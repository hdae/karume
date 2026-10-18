/**
 * The fused kernel of adaptive layer norm (the DiT's modulation). Folds the four nodes
 * `layer_norm → (1 + scale) → mul → add` into one dispatch (the body of the fusion rule `adaln` —
 * src/runtime/fusion.ts).
 *
 * The primitive sequence it replaces writes three intermediate f32 values to storage, each read
 * back by the next dispatch: `t = layer_norm(x)` → storage, `s = scale + 1` → storage,
 * `p = t · s` → storage, and finally `p + shift`. Written naively as one expression after fusion,
 * both `t · s + shift` and `t · (scale + 1)` are **textbook shapes for fma contraction**, and the
 * difference between one and two roundings shows up directly as a bit difference. So the bit
 * patterns of `s` and `p` are written to a workgroup `u32` and read back after a barrier, leaving
 * two explicit materialization points at the same positions as the plain sequence (the same
 * technique as silu.ts).
 *
 * `t` also has a storage boundary, but no third stage is placed there: the next step is a multiply
 * and does not form `a·b+c`, so no contraction can span it. However, WGSL allows reassociation as
 * well as contraction (the specification's section on reassociation and fusion), so omitting the
 * materialization of `t` is not guaranteed by the specification; it is a fact within the range
 * where the real-GPU A/B at adoption (below) confirmed bit equality. Contraction decisions
 * **inside** the expression for `t` line up because it shares the same string as the plain
 * layer_norm.
 *
 * **This "rounding barrier" is not guaranteed by the WGSL specification** — the specification only
 * allows floating-point fusion (contraction to fma) and does not require a workgroup memory round
 * trip to be respected as an optimisation barrier. The bit equality of finite values is a measured
 * fact used as the adoption check in per-backend real-GPU A/B (tests/gpu_adaln_fusion_test.ts);
 * **if a backend update breaks the PNG sha256 regression check, suspect this first** (removing the
 * fusion and going back to the primitive sequence isolates it). For NaN payloads, which WGSL does
 * not specify, only matching classification is part of the contract.
 *
 * MUST: the body of the normalisation (row statistics and affine) shares **the same string** with
 * the plain layer_norm ({@link LAYER_NORM_ROW_STATS_WGSL} / {@link LAYER_NORM_AFFINE_WGSL}).
 * Written separately, the primitive and the fused version could split in reduction order. The
 * price of sharing is fixed identifiers — the modulation's scale vector is called `scale_vec`
 * because the shared block uses `scale` for `1/dim`.
 * MUST: barriers sit only inside workgroup-uniform control flow. The row loop (the grid-stride of
 * `row = wid.x`) and the block loop (`0..blocks`) are both uniform, and layer_norm already runs
 * barriers inside a row loop of the same form. Iterating the output with a `o = lid` while loop
 * would give each lane a different iteration count and leave no place for barriers, so it is
 * restructured into the same block loop as silu.
 */

import {
  LAYER_NORM_AFFINE_WGSL,
  LAYER_NORM_ROW_STATS_WGSL,
  LAYER_NORM_WORKGROUP_SIZE,
} from "./layer-norm.ts";

/**
 * The uniform Params (16 bytes of rows / dim / eps). **The same contract as the plain layer_norm**,
 * so the generator itself is shared (including the MUST of storing eps as an f32 bit pattern).
 */
export { layerNormParams as adalnNormParams } from "./layer-norm.ts";

/**
 * MUST: the same value as the plain layer_norm. The shared row-statistics block expands the length
 * of `scratch` and the number of tree-reduction stages from this constant, so moving only one side
 * breaks the WGSL (or silently changes the reduction order).
 */
export const ADALN_NORM_WORKGROUP_SIZE = LAYER_NORM_WORKGROUP_SIZE;

export const ADALN_NORM_KEY = `adaln_norm:v1:lastdim:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`;

export const ADALN_NORM_WGSL: string = `// karume adaLN (layer_norm -> 1 + scale -> mul -> add, f32, staged modulation)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale_vec: array<f32>;
@group(0) @binding(5) var<storage, read> one: array<f32>;
@group(0) @binding(6) var<storage, read> shift: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${ADALN_NORM_WORKGROUP_SIZE}>;
var<workgroup> staged: array<u32, ${ADALN_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ADALN_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  // NOTE: the ceiling is not written as (dim + 255) / 256 — it overflows at dim = u32::MAX.
  let blocks = dim / ${ADALN_NORM_WORKGROUP_SIZE}u +
    select(0u, 1u, dim % ${ADALN_NORM_WORKGROUP_SIZE}u != 0u);
  var row = wid.x;
  while (row < params.rows) {
${LAYER_NORM_ROW_STATS_WGSL}

    var block = 0u;
    while (block < blocks) {
      let o = block * ${ADALN_NORM_WORKGROUP_SIZE}u + lid;
      // ③ materialize the modulation factor 1 + scale (where the add node writes to storage in the
      // plain sequence)
      if (o < dim) {
        staged[lid] = bitcast<u32>(scale_vec[o] + one[0u]);
      }
      workgroupBarrier();
      // ④ normalisation (the same expression as the plain layer_norm) → multiply. Rounded at the
      // same point as the plain mul.
      if (o < dim) {
        let modulation = bitcast<f32>(staged[lid]);
        staged[lid] = bitcast<u32>((${LAYER_NORM_AFFINE_WGSL}) * modulation);
      }
      workgroupBarrier();
      // ⑤ add shift (the plain add node)
      if (o < dim) {
        out[base + o] = bitcast<f32>(staged[lid]) + shift[o];
      }
      workgroupBarrier();
      block = block + 1u;
    }
    row = row + nwg.x;
  }
}
`;
