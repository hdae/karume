/**
 * The kernel of the fusion rule `geluMulStaticQuantize` (src/runtime/fusion-rules/): the chain
 * `gelu_tanh(gate) → mul(·, other) → static_quantize` in one dispatch, one element per thread
 * (perf-ledger K-48 step 1b — Gemma 4 QAT's per-layer-input gate, `[1, M, 256]` f32).
 *
 * The plain chain stores two f32 values (the gelu output and the product) before the SRQ reads the
 * product's bits. Every stored value is re-materialised here with `bitcast<u32>(v) ^ mask`, the
 * mask words 0 **at run time** (uniform words, never constants): WGSL lets a compiler contract
 * floating-point operations across expressions, so without the barrier the gelu's last multiply and
 * the product could fuse into an fma whose rounding the plain path never performs. The same
 * discipline as the GEMV epilogue's `rounding_mask` (linear-gemv.ts) and rms_norm_add's.
 * **The barrier is measured behaviour, not a WGSL guarantee** (0 differing words over 786,432
 * adversarial words on Vulkan / NVIDIA, with mask injection after gelu and after the mul both
 * changing the output — perf-ledger K-48); if a backend update breaks the QAT sha256 regression
 * checks, suspect this fusion first (`fuseGeluMulStaticQuantize: false` isolates it).
 *
 * MUST: every expression is the product's own text — the gelu_tanh value from
 * {@link prologueUnaryWgsl} (the elementwise generator's), the product from
 * {@link fusedBinaryWgsl}, the SRQ from {@link STATIC_QUANTIZE_FN_WGSL} (the standalone v2 kernel's
 * `quantize`). A copy would be bit-identical only by accident.
 * MUST: the product keeps the node's operand order ({@link GeluMulOrder}). IEEE multiplication is
 * commutative, but which payload a backend returns for two NaNs is not specified, and the SRQ
 * passes NaN payloads through.
 */

import { CodegenError } from "../codegen/errors.ts";
import { fusedBinaryWgsl, prologueUnaryWgsl } from "../codegen/elementwise.ts";
import { assertU32Params } from "../codegen/params.ts";
import { STATIC_QUANTIZE_FN_WGSL, staticQuantizeParams } from "./static-quantize.ts";

/** The mul's slot of the gelu value: `gelu-other` = `mul(gelu(gate), other)`. */
export type GeluMulOrder = "gelu-other" | "other-gelu";

export const GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE = 128;

const canonicalizeOrder = (order: GeluMulOrder): GeluMulOrder => {
  if (order === "gelu-other" || order === "other-gelu") return order;
  throw new CodegenError(`gelu_mul_static_quantize codegen: invalid input order of mul (${order})`);
};

/** MUST: the operand order stays in the key (it changes the WGSL — {@link GeluMulOrder}). */
export const geluMulStaticQuantizeKey = (order: GeluMulOrder): string =>
  `gelu_mul_static_quantize:v1:${canonicalizeOrder(order)}:f32:wg${GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE}`;

export const geluMulStaticQuantizeWgsl = (order: GeluMulOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  const gelu = prologueUnaryWgsl("gelu_tanh", "v0");
  const product =
    canonicalOrder === "gelu-other"
      ? fusedBinaryWgsl("mul", "act", "v1")
      : fusedBinaryWgsl("mul", "v1", "act");
  return `// karume gelu_tanh -> mul -> static_quantize (${canonicalOrder}, f32, SRQ v2, u32 out)
struct Params {
  n: u32,
  mask_act: u32,
  mask_mul: u32,
  reserved: u32,
  srq: array<vec4<u32>, 65>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> other: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

${gelu.helpers.join("\n\n")}

fn word(index: u32) -> u32 { return params.srq[index >> 2u][index & 3u]; }
${STATIC_QUANTIZE_FN_WGSL}
@compute @workgroup_size(${GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * ${GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < params.n) {
    let v0 = gate[i];
    let v1 = other[i];
    // Rounding barriers: the two values the plain chain stores (masks are 0 at run time)
    let act = bitcast<f32>(bitcast<u32>(${gelu.value}) ^ params.mask_act);
    out[i] = quantize(bitcast<u32>(${product}) ^ params.mask_mul);
    i = i + stride;
  }
}
`;
};

/**
 * The uniform (264 words = 1056 B): `n`, the two rounding masks (0 — written as run-time words; do
 * not replace the XOR with a constant expression), a reserved word, then the 260-word table of
 * {@link staticQuantizeParams} at word 4 (the `srq` member — `array<vec4<u32>>` is 16-byte aligned).
 */
export const geluMulStaticQuantizeParams = (
  count: number,
  scale: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("gelu_mul_static_quantize params", { count });
  const params = new Uint32Array(264);
  params[0] = count;
  params.set(staticQuantizeParams(count, scale), 4);
  return params;
};
