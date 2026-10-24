/**
 * An f32 kernel that applies RoPE in one pass (the body of the fusion rule `rope` —
 * src/runtime/fusion.ts). Two rotations exist, one per measured exporter form:
 *
 * - **half** (half split): the input `x` is `[1,H,S,D]` with table `[1,1,S,D]`, or `[1,S,H,D]`
 *   with table `[1,S,1,D]`. The last dimension is split in half, and with
 *   `rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])` it outputs `x * cos + rotate_half(x) * sin`.
 * - **pair** (adjacent pairs — the real form of the complex rotation in Wan's DiT): `x` is
 *   `[1,S,H,D]` with table `[1,S,1,D]`, and the rotation reads `pairs` `[1,S,H·D/2,2]`:
 *   `rot[2p] = -pairs[2p+1]`, `rot[2p+1] = pairs[2p]` (flat indices — the last axis of `pairs`
 *   has length 2, so the parity of the flat index is the position inside the pair). It outputs
 *   `x * cos + rot * sin`. `pairs` is bound separately from `x`, so the kernel computes exactly
 *   what the folded window computes without assuming where `pairs` comes from; in the measured
 *   graphs it is a reshape alias of `x`, and the same buffer is bound twice read-only (legal
 *   within one usage scope).
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
 * MUST: both rotations negate by flipping the sign bit (exact, as the primitive `neg` is for every
 * non-NaN value; for NaN payloads only the classification is part of the contract).
 * MUST: every output element is written with grid-stride. Barriers only inside workgroup-uniform
 * loops. No optional features / subgroups / atomics.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ROPE_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const ROPE_KEY = `rope:v1:half:f32:wg${ROPE_WORKGROUP_SIZE}`;

export const ROPE_BSHD_KEY = `rope:v1:half:bshd:f32:wg${ROPE_WORKGROUP_SIZE}`;

/** The pair rotation exists only in the measured BSHD layout (Wan's DiT). */
export const ROPE_PAIR_BSHD_KEY = `rope:v1:pair:bshd:f32:wg${ROPE_WORKGROUP_SIZE}`;

export type RopeLayout = "bhsd" | "bshd";
export type RopeRotation = "half" | "pair";

/** The storage inputs per rotation (bindings 1.., the output follows them). */
const ROPE_INPUTS: Readonly<Record<RopeRotation, readonly string[]>> = {
  half: ["x", "cos_table", "sin_table"],
  pair: ["x", "pairs", "cos_table", "sin_table"],
};

/** The read of `rotate(x)[i]` as f32 bits per rotation (the only other difference). */
const ROPE_ROTATION_WGSL: Readonly<Record<RopeRotation, string>> = {
  half: `      let row_base = i - d;
      var rotated_bits: u32;
      if (d < params.half_dim) {
        rotated_bits = bitcast<u32>(x[row_base + d + params.half_dim]) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>(x[row_base + d - params.half_dim]);
      }`,
  pair: `      var rotated_bits: u32;
      if ((i & 1u) == 0u) {
        rotated_bits = bitcast<u32>(pairs[i + 1u]) ^ 0x80000000u;
      } else {
        rotated_bits = bitcast<u32>(pairs[i - 1u]);
      }`,
};

/**
 * The axis order switches only the position-table index, and the rotation only the read of the
 * rotated element (and the `pairs` binding); the rounding barrier is shared.
 *
 * The pair rotation keeps the same Params as the half split ({@link ropeParams}); its WGSL does not
 * read `half_dim`.
 */
export const ropeWgsl = (layout: RopeLayout, rotation: RopeRotation = "half"): string => {
  if (rotation === "pair" && layout !== "bshd") {
    throw new CodegenError(`rope codegen: the pair rotation exists only for bshd (${layout})`);
  }
  const inputs = ROPE_INPUTS[rotation];
  const bindings = [
    ...inputs.map(
      (name, at) => `@group(0) @binding(${at + 1}) var<storage, read> ${name}: array<f32>;`,
    ),
    `@group(0) @binding(${inputs.length + 1}) var<storage, read_write> out: array<f32>;`,
  ].join("\n");
  return `// karume RoPE (${rotation === "half" ? "half split" : "pair rotation"}, f32, full-write)
struct Params {
  n: u32,
  ${layout === "bhsd" ? "sequence" : "heads"}: u32,
  head_dim: u32,
  half_dim: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
${bindings}
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
${ROPE_ROTATION_WGSL[rotation]}
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
};

/** 16-byte uniform. The second value is S for BHSD, H for BSHD (the axis of the table index). */
export const ropeParams = (
  n: number,
  tableAxisSize: number,
  headDim: number,
  layout: RopeLayout = "bhsd",
): Uint32Array<ArrayBuffer> => {
  const axis = layout === "bhsd" ? "sequence" : "heads";
  assertU32Params("rope params", { n, [axis]: tableAxisSize, headDim });
  if (tableAxisSize === 0) throw new CodegenError(`rope params: ${axis} must be at least 1`);
  if (headDim === 0 || headDim % 2 !== 0) {
    throw new CodegenError(`rope params: headDim must be a positive even number (${headDim})`);
  }
  const rowSize = tableAxisSize * headDim;
  assertU32Params("rope params", { [`${axis} × headDim`]: rowSize });
  if (n % rowSize !== 0) {
    throw new CodegenError(
      `rope params: n ${n} is not a whole number of rows of ${axis} ${tableAxisSize} × headDim ${headDim}`,
    );
  }
  return new Uint32Array([n, tableAxisSize, headDim, headDim / 2]);
};
