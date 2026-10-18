/**
 * An f32 kernel that applies half-split RoPE in one pass (the body of the fusion rule `rope` —
 * src/runtime/fusion.ts).
 *
 * The input `x` is `[1,H,S,D]` with table `[1,1,S,D]`, or `[1,S,H,D]` with table `[1,S,1,D]`. The
 * last dimension is split in half, and with `rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])` it
 * outputs `x * cos + rotate_half(x) * sin`.
 *
 * MUST: cos / sin are read as is from the f32 tables the host prepared (anima = the raw tables
 * from torch / gemma4 = tables that the TS of ADR 0091 decisions 1 / 2 computes in f64 and rounds
 * to f32 — not bit-identical to the upstream torch tables, limitations.md). The kernel does not
 * recompute trigonometric functions or positions (including the H/W coordinates of anima's
 * tokens) (even for non-square inputs, only the token order is the contract).
 * MUST: the two products are written to a 2 KiB workgroup `u32` array and brought back to the
 * addition after a barrier. Written plainly as `x*cos + rot*sin`, the two f32 roundings of the
 * primitive sequence it replaces (three dispatches, mul → mul → add) could be contracted.
 * **This "rounding barrier" is not guaranteed by the WGSL specification** — the specification
 * allows floating-point fusion (contraction to fma) and does not forbid it, and whether a workgroup
 * memory round trip is respected as a boundary is an implementation-dependent measured fact.
 * So **if a backend update breaks the PNG sha256 regression check, suspect this first** (removing
 * the fusion and going back to the primitive sequence isolates it).
 * MUST: every output element is written with grid-stride. Barriers only inside workgroup-uniform
 * loops. No optional features / subgroups / atomics.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ROPE_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const ROPE_KEY = `rope:v1:half:f32:wg${ROPE_WORKGROUP_SIZE}`;

export const ROPE_BSHD_KEY = `rope:v1:half:bshd:f32:wg${ROPE_WORKGROUP_SIZE}`;
export type RopeLayout = "bhsd" | "bshd";

/** The axis order switches only the position-table index; the rounding barrier is shared. */
export const ropeWgsl = (layout: RopeLayout): string =>
  `// karume RoPE (half split, f32, full-write)
struct Params {
  n: u32,
  ${layout === "bhsd" ? "sequence" : "heads"}: u32,
  head_dim: u32,
  half_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read> sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> products: array<vec2<u32>, ${ROPE_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ROPE_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${ROPE_WORKGROUP_SIZE}u;
  var base = wid.x * ${ROPE_WORKGROUP_SIZE}u;
  while (base < params.n) {
    let i = base + lid;
    if (i < params.n) {
      let d = i % params.head_dim;
      let row = i / params.head_dim;
      let token = ${layout === "bhsd" ? "row % params.sequence" : "row / params.heads"};
      let row_base = i - d;
      var rotated_bits: u32;
      if (d < params.half_dim) {
        rotated_bits = bitcast<u32>(x[row_base + d + params.half_dim]) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>(x[row_base + d - params.half_dim]);
      }
      let table_index = token * params.head_dim + d;
      let direct_bits = bitcast<u32>(x[i] * cos_table[table_index]);
      let cross_bits = bitcast<u32>(bitcast<f32>(rotated_bits) * sin_table[table_index]);
      products[lid] = vec2<u32>(direct_bits, cross_bits);
    } else {
      products[lid] = vec2<u32>(0u);
    }
    workgroupBarrier();
    if (i < params.n) {
      let pair = products[lid];
      out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
`;

/** 16-byte uniform. The second value is S for BHSD, H for BSHD (the axis of the table index). */
export const ropeParams = (
  n: number,
  tableAxisSize: number,
  headDim: number,
  layout: RopeLayout = "bhsd",
): Uint32Array<ArrayBuffer> => {
  const axis = layout === "bhsd" ? "sequence" : "heads";
  assertU32Params("rope params", { n, [axis]: tableAxisSize, headDim });
  if (tableAxisSize === 0) throw new CodegenError(`rope params: ${axis} は 1 以上`);
  if (headDim === 0 || headDim % 2 !== 0) {
    throw new CodegenError(`rope params: headDim は正の偶数（${headDim}）`);
  }
  const rowSize = tableAxisSize * headDim;
  assertU32Params("rope params", { [`${axis} × headDim`]: rowSize });
  if (n % rowSize !== 0) {
    throw new CodegenError(
      `rope params: n ${n} が ${axis} ${tableAxisSize} × headDim ${headDim} の整数行でない`,
    );
  }
  return new Uint32Array([n, tableAxisSize, headDim, headDim / 2]);
};
