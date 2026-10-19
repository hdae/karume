/**
 * The SiLU fusion kernel that keeps the intermediate f32 storage boundary of `sigmoid(x) → mul`
 * (the body of the fusion rule `silu` — src/runtime/fusion.ts).
 *
 * The primitive sequence it replaces writes the sigmoid result to storage, and the next dispatch's
 * mul reads it back. Written simply as `x * sigmoid_stable(x)`, the compiler could fold this
 * boundary, so the sigmoid's f32 bit pattern is written to a workgroup `u32` and read back after a
 * barrier, leaving an explicit materialization point.
 *
 * **This "rounding barrier" is not guaranteed by the WGSL specification** — the specification only
 * allows floating-point fusion (contraction to fma) and does not require a workgroup memory round
 * trip to be respected as an optimisation barrier. The bit equality of finite values is a measured
 * fact used as the adoption check in per-backend real-GPU A/B (tests/gpu_silu_fusion_test.ts);
 * **if a backend update breaks the PNG sha256 regression check, suspect this first** (removing the
 * fusion and going back to the primitive sequence isolates it). For NaN payloads, which WGSL does
 * not specify, only matching classification is part of the contract.
 *
 * MUST: barriers sit only inside the workgroup-uniform block loop (the second one makes the next
 * block's writes start only after every lane of the previous block has finished reading). It uses
 * no subgroups / atomics / optional features, so it works on WebGPU core's Metal / Dawn / wgpu
 * paths.
 * MUST: the sigmoid body shares **the same string** as the elementwise codegen (written
 * separately, the primitive and the fused version could split in rounding sequence).
 */

import { SIGMOID_STABLE_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** Input order of the replaced mul (commutative for finite values; NaN payloads may differ). */
export type SiluMulOrder = "x-sigmoid" | "sigmoid-x";

export const SILU_WORKGROUP_SIZE = 256;

const canonicalizeOrder = (order: SiluMulOrder): SiluMulOrder => {
  if (order === "x-sigmoid" || order === "sigmoid-x") return order;
  throw new CodegenError(`silu codegen: invalid input order of mul (${order})`);
};

export const siluKey = (order: SiluMulOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  return `silu:v1:${canonicalOrder}:f32:wg${SILU_WORKGROUP_SIZE}`;
};

/**
 * MUST: the input order stays in both the WGSL and the pipeline key. Multiplication is commutative
 * for finite values, but which NaN payload is chosen can differ per backend, so folding it on one
 * side only silently produces a different sequence.
 * NOTE: the ceiling of the block count is not written as `(n + 255) / 256` — it overflows at
 * n = u32::MAX.
 */
export const siluWgsl = (order: SiluMulOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  // The replaced mul also re-reads the original input in a separate dispatch. Written so it is not
  // carried over in a register.
  const product =
    canonicalOrder === "x-sigmoid"
      ? "x_for_mul * sigmoid_after_store"
      : "sigmoid_after_store * x_for_mul";
  return `// karume SiLU (${canonicalOrder}, f32, staged sigmoid, full-write)
struct Params {
  n: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> sigmoid_bits: array<u32, ${SILU_WORKGROUP_SIZE}>;

${SIGMOID_STABLE_WGSL}

@compute @workgroup_size(${SILU_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let blocks = params.n / ${SILU_WORKGROUP_SIZE}u +
    select(0u, 1u, params.n % ${SILU_WORKGROUP_SIZE}u != 0u);
  var block = wid.x;
  while (block < blocks) {
    let i = block * ${SILU_WORKGROUP_SIZE}u + lid;
    if (i < params.n) {
      let x_for_sigmoid = x[i];
      sigmoid_bits[lid] = bitcast<u32>(sigmoid_stable(x_for_sigmoid));
    }
    workgroupBarrier();
    if (i < params.n) {
      let x_for_mul = x[i];
      let sigmoid_after_store = bitcast<f32>(sigmoid_bits[lid]);
      out[i] = ${product};
    }
    workgroupBarrier();
    block = block + nwg.x;
  }
}
`;
};

/** 16-byte uniform params. n is the element count of the f32 input / output. */
export const siluParams = (n: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("silu params", { n });
  return new Uint32Array([n, 0, 0, 0]);
};
