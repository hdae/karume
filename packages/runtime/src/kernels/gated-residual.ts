/**
 * The fused kernel of the per-token gated residual (the body of the fusion rule `gatedResidual` —
 * src/runtime/fusion.ts): `residual + y · where(mask, gate_c, gate_g)` in one dispatch.
 *
 * The primitive sequence it replaces (Wan2.2's DiT, after the attention output projection and
 * after the FFN) writes two `[.., dim]` intermediates: `g = where(mask, gate_c, gate_g)` →
 * storage, `p = y · g` → storage, and finally `residual + p`. `where` only selects (no arithmetic),
 * so the kernel reads the selected gate element directly. `residual + y · g` is a textbook shape
 * for fma contraction, so the bit pattern of `p` is written to a workgroup `u32` and read back after
 * a barrier, leaving the materialization point of the plain mul in place (the same technique as
 * rope.ts / silu.ts).
 *
 * **This "rounding barrier" is not guaranteed by the WGSL specification** — the specification only
 * allows floating-point fusion (contraction to fma) and does not require a workgroup memory round
 * trip to be respected as an optimisation barrier. The bit equality of finite values is a measured
 * fact used as the adoption check in per-backend real-GPU A/B
 * (tests/gpu_gated_residual_fusion_test.ts); **if a backend update breaks the PNG sha256 regression
 * check, suspect this first** (removing the fusion and going back to the primitive sequence
 * isolates it). For NaN payloads, which WGSL does not specify, only matching classification is part
 * of the contract.
 *
 * MUST: the operand orders are those of the plain nodes: `y * gate` (mul(y, where)) and
 * `residual + product` (add(residual, mul)).
 * MUST: the gate selection is the plain where's test: `select(gate_g, gate_c, mask[row] != 0u)`
 * (bool is u32 0 / 1 — ADR 0009), with one mask element per row of `dim` elements.
 * MUST: every output element is written with grid-stride. Barriers only inside the
 * workgroup-uniform block loop. No optional features / subgroups / atomics. Six storage bindings.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const GATED_RESIDUAL_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const GATED_RESIDUAL_KEY = `gated_residual:v1:token:f32:wg${GATED_RESIDUAL_WORKGROUP_SIZE}`;

export const GATED_RESIDUAL_WGSL: string = `// karume gated residual (residual + y * where(mask, gate_c, gate_g), f32, staged product)
struct Params {
  n: u32,
  dim: u32,
  reserved0: u32,
  reserved1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> gate_condition: array<f32>;
@group(0) @binding(3) var<storage, read> gate_generation: array<f32>;
@group(0) @binding(4) var<storage, read> y: array<f32>;
@group(0) @binding(5) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
var<workgroup> products: array<u32, ${GATED_RESIDUAL_WORKGROUP_SIZE}>;

@compute @workgroup_size(${GATED_RESIDUAL_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${GATED_RESIDUAL_WORKGROUP_SIZE}u;
  var base = wid.x * ${GATED_RESIDUAL_WORKGROUP_SIZE}u;
  while (base < params.n) {
    let i = base + lid;
    if (i < params.n) {
      let row = i / params.dim;
      let o = i - row * params.dim;
      let gate = select(gate_generation[o], gate_condition[o], mask[row] != 0u);
      products[lid] = bitcast<u32>(y[i] * gate);
    }
    workgroupBarrier();
    if (i < params.n) {
      out[i] = residual[i] + bitcast<f32>(products[lid]);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
`;

/** 16-byte uniform: the element count and the row width (the gate vectors' length). */
export const gatedResidualParams = (n: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("gated residual params", { n, dim });
  if (dim < 1) throw new CodegenError(`gated residual params: dim must be at least 1 (${dim})`);
  if (n % dim !== 0) {
    throw new CodegenError(`gated residual params: n ${n} is not a whole number of rows of ${dim}`);
  }
  return new Uint32Array([n, dim, 0, 0]);
};
