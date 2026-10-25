/**
 * **rms_norm → pair RoPE in one kernel** (f32, token-major, full-write — perf-ledger F-3): the Wan
 * DiT's self-attention q / k prologue `rms_norm` over the whole token row, then (through two reshape
 * aliases) the pair-form RoPE `rope:v1:pair:bshd`, with the normalised row never stored.
 *
 * One workgroup normalises one token row (grid-stride over rows), with the plain kernel's reduction
 * ({@link rmsNormInvWgsl}); then the row is rotated 256 elements at a time. Each element's
 * normalised value and its pair partner's are recomputed from `x` with the plain kernel's expression
 * ({@link rmsNormValueWgsl}), and the RoPE products go through the workgroup array of the plain
 * RoPE ({@link ropeProductsWgsl}) before they are summed. The output index of element `o` of row `r`
 * is `r · dim + o`, where the plain RoPE writes it.
 *
 * Bit identity with the pair "rms_norm → rope:v1:pair:bshd" (measured at adoption on Vulkan /
 * NVIDIA, Node + Dawn and Chromium: 0 differing words at the Wan2.2 and Wan2.1 shapes, special
 * values and NaN payloads included — research of 2026-10-25, wave a stage 3):
 * - the reduction, the normalised value and the RoPE products are the plain kernels' strings, so the
 *   compiler's contraction decisions are theirs; the normalised value is parenthesised where it
 *   feeds the products, the form the adoption measurement compiled;
 * - the normalised value meets only multiplications before the products are staged, so no
 *   contraction spans the stored-then-read boundary the pair had; the staging itself is the RoPE
 *   rounding barrier (its MUST — without it 19–25% of the words move by one ulp on this backend);
 * - the backend measured flushes subnormals in both kernels alike; a backend that keeps them is
 *   unchecked.
 *
 * MUST: the row loop and the chunk loop are workgroup-uniform (every barrier is reached by every
 * invocation). MUST: `dim` is `heads · head_dim` with an even `head_dim` (a pair never straddles a
 * head) — the fusion rule checks it; Params carry `head_dim`, and the token of every element of row
 * `r` is `r` (token-major input).
 */

import { assertU32Params } from "../codegen/params.ts";
import { CodegenError } from "../codegen/errors.ts";
import { RMS_NORM_WORKGROUP_SIZE, rmsNormInvWgsl, rmsNormValueWgsl } from "./rms-norm.ts";
import { ROPE_SUM_WGSL, ropePairRotationWgsl, ropeProductsWgsl } from "./rope.ts";

export const RMS_NORM_ROPE_KEY = `rms_norm_rope:v1:pair:bshd:f32:wg${RMS_NORM_WORKGROUP_SIZE}`;

export const RMS_NORM_ROPE_WGSL: string = `// karume rms_norm → RoPE (pair rotation, token-major, f32, full-write) — the normalised row is never stored
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
  head_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> cos_table: array<f32>;
@group(0) @binding(4) var<storage, read> sin_table: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${RMS_NORM_WORKGROUP_SIZE}>;
var<workgroup> products: array<vec2<u32>, ${RMS_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${RMS_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

${rmsNormInvWgsl(RMS_NORM_WORKGROUP_SIZE)}
    // RoPE on the normalised row, ${RMS_NORM_WORKGROUP_SIZE} elements per chunk (a workgroup-uniform loop: the
    // barriers below need every invocation). Every element of this row belongs to token \`row\`.
    let token = row;
    var chunk = 0u;
    while (chunk < dim) {
      let o = chunk + lid;
      let i = base + o;
      if (o < dim) {
        let d = o % params.head_dim;
${ropePairRotationWgsl((offset) => `(${rmsNormValueWgsl(`o ${offset}`)})`)}
${ropeProductsWgsl(`(${rmsNormValueWgsl("o")})`)}
      } else {
        products[lid] = vec2<u32>(0u);
      }
      workgroupBarrier();
      if (o < dim) {
        let pair = products[lid];
        out[i] = ${ROPE_SUM_WGSL};
      }
      workgroupBarrier();
      chunk = chunk + ${RMS_NORM_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

/** Uniform Params (`{rows, dim, eps, head_dim}`, one 16-byte block). */
export const rmsNormRopeParams = (
  rows: number,
  dim: number,
  eps: number,
  headDim: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("rms_norm_rope params", { rows, dim, head_dim: headDim, size: rows * dim });
  if (headDim < 2 || headDim % 2 !== 0 || dim % headDim !== 0) {
    throw new CodegenError(
      `rms_norm_rope params: head_dim (${headDim}) must be even and divide dim (${dim})`,
    );
  }
  if (!Number.isFinite(eps)) {
    throw new CodegenError(`rms_norm_rope params: eps must be finite (${eps})`);
  }
  const words = new Uint32Array(4);
  words[0] = rows;
  words[1] = dim;
  new Float32Array(words.buffer, 8, 1)[0] = eps;
  words[3] = headDim;
  return words;
};
