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
 *
 * ## Per-token modulation (Wan2.2's DiT)
 *
 * Wan2.2 picks the modulation per token: `where(mask, scale_c, scale_g)` and
 * `where(mask, shift_c, shift_g)` (`mask` `[..,1]` bool — the condition-side timestep of ADR 0121
 * decision 3) materialise `[.., dim]` tensors that the plain sequence then adds `one` to, multiplies
 * and adds. `where` only selects (no arithmetic), so the `token` variant reads the selected vector
 * element and runs **the same staged body**: the f32 operations and their order are those of the
 * plain sequence.
 * Folding it in one dispatch would need 10 storage bindings (x, weight, bias, mask, four vectors,
 * one, out), above the WebGPU default of 8 per stage, and the fusion pass does not depend on device
 * limits. So the fused step is two dispatches: {@link ADALN_MODULATION_PACK_WGSL} copies the four
 * vectors (as raw bits) into one table `[scale_c, scale_g, shift_c, shift_g]` of `4·dim`, and the
 * body binds that table (7 storage bindings, as the broadcast variant).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
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

/** The per-token body (the second dispatch of the fused step — see the module notes). */
export const ADALN_NORM_TOKEN_KEY = `adaln_norm:v1:lastdim:token:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`;

/**
 * Where the modulation vectors come from: `broadcast` = one `[1,…,1,dim]` scale / shift for every
 * row (anima's DiT); `token` = per row, `mask[row]` picks the condition or the generation vector
 * from the packed table (Wan2.2's DiT).
 */
export type AdalnModulation = "broadcast" | "token";

/** The parts of the body that differ per {@link AdalnModulation}; everything else is shared. */
type ModulationSource = {
  readonly title: string;
  /** Bindings 4–6 (x / weight / bias are 1–3 and out is 7 in both). */
  readonly bindings: string;
  /** Per-row lines after the row statistics (empty for broadcast). */
  readonly select: string;
  /** The scale element of column `o` (the value the plain `add(scale, one)` reads). */
  readonly scale: string;
  /** The shift element of column `o` (the value the plain final `add` reads). */
  readonly shift: string;
};

const MODULATION_SOURCES: Readonly<Record<AdalnModulation, ModulationSource>> = {
  broadcast: {
    title: "layer_norm -> 1 + scale -> mul -> add, f32, staged modulation",
    bindings: [
      "@group(0) @binding(4) var<storage, read> scale_vec: array<f32>;",
      "@group(0) @binding(5) var<storage, read> one: array<f32>;",
      "@group(0) @binding(6) var<storage, read> shift: array<f32>;",
    ].join("\n"),
    select: "",
    scale: "scale_vec[o]",
    shift: "shift[o]",
  },
  token: {
    title: "layer_norm -> 1 + scale -> mul -> add, f32, staged modulation, per-token where",
    bindings: [
      "@group(0) @binding(4) var<storage, read> mask: array<u32>;",
      "@group(0) @binding(5) var<storage, read> modulation_table: array<f32>;",
      "@group(0) @binding(6) var<storage, read> one: array<f32>;",
    ].join("\n"),
    // MUST: the same test as the plain where (`select(gen, cond, mask != 0u)`). The table rows are
    // [scale_c, scale_g, shift_c, shift_g]; 4 · dim fits in u32 (adalnModulationPackParams).
    select: [
      "",
      "    // where(mask, condition, generation) for this token: row 0 / 1 of the scale half of the",
      "    // packed table, and the same row of the shift half",
      "    let scale_base = select(dim, 0u, mask[row] != 0u);",
      "    let shift_base = scale_base + 2u * dim;",
      "",
    ].join("\n"),
    scale: "modulation_table[scale_base + o]",
    shift: "modulation_table[shift_base + o]",
  },
};

/**
 * MUST: one template for both variants — the staged body (③ → ④ → ⑤) is the one the bit-equality
 * A/B adopted, and the `token` variant differs only in where the scale / shift element is read.
 * The `broadcast` output is byte-identical to the key `adaln_norm:v1:lastdim` (snapshot).
 */
export const adalnNormWgsl = (modulation: AdalnModulation): string => {
  const source = MODULATION_SOURCES[modulation];
  return `// karume adaLN (${source.title})
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
${source.bindings}
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
${source.select}
    var block = 0u;
    while (block < blocks) {
      let o = block * ${ADALN_NORM_WORKGROUP_SIZE}u + lid;
      // ③ materialize the modulation factor 1 + scale (where the add node writes to storage in the
      // plain sequence)
      if (o < dim) {
        staged[lid] = bitcast<u32>(${source.scale} + one[0u]);
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
        out[base + o] = bitcast<f32>(staged[lid]) + ${source.shift};
      }
      workgroupBarrier();
      block = block + 1u;
    }
    row = row + nwg.x;
  }
}
`;
};

export const ADALN_NORM_WGSL: string = adalnNormWgsl("broadcast");

export const ADALN_MODULATION_PACK_WORKGROUP_SIZE = 256;

/** The first dispatch of the per-token fused step (see the module notes). */
export const ADALN_MODULATION_PACK_KEY = `adaln_modulation_pack:v1:u32:wg${ADALN_MODULATION_PACK_WORKGROUP_SIZE}`;

/**
 * Copies the four modulation vectors into one table `[scale_c, scale_g, shift_c, shift_g]` of
 * `4 · dim` words.
 *
 * MUST: a copy of raw `u32` words, no arithmetic — every bit pattern (subnormals, NaN payloads)
 * reaches the body as the plain `where` would hand it over. MUST: grid-stride over the columns
 * (every table word is written; `dim` can exceed one dispatch).
 */
export const ADALN_MODULATION_PACK_WGSL: string = [
  "// karume adaLN modulation pack (4 vectors -> one table, raw u32 copy, full-write)",
  "struct Params {",
  "  dim: u32,",
  "  reserved0: u32,",
  "  reserved1: u32,",
  "  reserved2: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> scale_condition: array<u32>;",
  "@group(0) @binding(2) var<storage, read> scale_generation: array<u32>;",
  "@group(0) @binding(3) var<storage, read> shift_condition: array<u32>;",
  "@group(0) @binding(4) var<storage, read> shift_generation: array<u32>;",
  "@group(0) @binding(5) var<storage, read_write> modulation_table: array<u32>;",
  "",
  `@compute @workgroup_size(${ADALN_MODULATION_PACK_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let dim = params.dim;",
  `  let stride = nwg.x * ${ADALN_MODULATION_PACK_WORKGROUP_SIZE}u;`,
  "  var o = gid.x;",
  "  while (o < dim) {",
  "    modulation_table[o] = scale_condition[o];",
  "    modulation_table[dim + o] = scale_generation[o];",
  "    modulation_table[2u * dim + o] = shift_condition[o];",
  "    modulation_table[3u * dim + o] = shift_generation[o];",
  "    o = o + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * 16-byte uniform (`dim` + padding). MUST: `4 · dim` fits in u32 — the body indexes the table up
 * to `3 · dim + dim - 1`.
 */
export const adalnModulationPackParams = (dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("adaln modulation pack params", { dim, "4 × dim": 4 * dim });
  if (dim < 1) {
    throw new CodegenError(`adaln modulation pack params: dim must be a positive integer (${dim})`);
  }
  return new Uint32Array([dim, 0, 0, 0]);
};
