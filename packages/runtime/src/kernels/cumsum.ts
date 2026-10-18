/**
 * The fixed kernel of cumsum (prefix sum over the last dimension `out[…, j] = Σ_{i ≤ j} x[…, i]`,
 * f32).
 *
 * The input is a contiguous `[rows, dim]` layout with the leading dimensions flattened; the output
 * has the same shape and is contiguous.
 *
 * It uses **one invocation = a sequential scan of one row** (a different form from the row
 * reduce's "one workgroup = one row + tree reduction"). The reason is the measured row length: the
 * spline's cumwidths / cumheights have bins ≈ 10 (recon §2), and a Hillis–Steele or Blelloch
 * parallel scan would pay log2(dim) workgroupBarriers per row to fold 10 elements — slower than
 * sequential, adding only implementation complexity (barrier uniformity, crossing rows). The row
 * count is the source of parallelism, so **rows go through a grid-stride**.
 *
 * MUST: if demand for longer rows (rows of thousands of elements) shows up in measurement, revisit
 * this choice. The sequential scan pays O(dim) per row on one thread, and with a large dim the
 * parallelism is only the row count.
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535).
 * MUST: the accumulation runs **from the front** (`out[j]` is the sum of `x[0..j]`). Writing it
 * backwards changes neither the shape nor the element count, so a mixed-up direction is detectable
 * only in the values (tests/gpu_ops_test.ts pins it with an asymmetric sequence).
 */

import { assertU32Params } from "../codegen/params.ts";

export const CUMSUM_WORKGROUP_SIZE = 256;

export const CUMSUM_KEY = `cumsum:v1:f32:wg${CUMSUM_WORKGROUP_SIZE}`;

export const CUMSUM_WGSL: string = [
  "// karume cumsum (last dim, f32, sequential per row)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${CUMSUM_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${CUMSUM_WORKGROUP_SIZE}u;`,
  "  var row = gid.x;",
  "  while (row < params.rows) {",
  "    let base = row * params.dim;",
  "    var acc = 0.0;",
  "    var j = 0u;",
  "    while (j < params.dim) {",
  "      acc = acc + x[base + j];",
  "      out[base + j] = acc;",
  "      j = j + 1u;",
  "    }",
  "    row = row + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * The uniform Params (the same two words as the row-op family). 16 bytes are allocated even for
 * two words of content (allocating in 16-byte units is a project convention (the fusion pass fixes
 * the same form in ADR 0040), not a requirement of the specification).
 */
export const cumsumParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("cumsum params", { rows, dim });
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};
