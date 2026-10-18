/**
 * The fixed kernel of pad (constant-0 padding of the last dimension, f32).
 *
 * The input is a contiguous `[rows, in_len]` layout with the leading dimensions flattened; the
 * output is `[rows, left + in_len + right]`. The only measured use is `F.pad(p_attn, [w, w])` on
 * the value side of relative-position attention (recon §2), so there is a single form: last
 * dimension, constant 0, left and right padding.
 *
 * MUST: **one dispatch writes every byte of the output** (the full-write invariant — ADR 0014 /
 * 0004). In-range elements are copied and out-of-range elements are **written** as 0; it is not
 * the form "copy only the in-range part into a zero-initialised buffer". Recycled buffers from the
 * buffer pool are not zero-initialised, so the latter is incompatible with the pool (a trap known
 * from the prototype).
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535).
 */

import { assertU32Params } from "../codegen/params.ts";

export const PAD_WORKGROUP_SIZE = 256;

export const PAD_KEY = `pad:v1:f32:wg${PAD_WORKGROUP_SIZE}`;

export const PAD_WGSL: string = [
  "// karume pad (last dim, constant 0, f32, full-write)",
  "struct Params {",
  "  n: u32,",
  "  out_len: u32,",
  "  in_len: u32,",
  "  left: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${PAD_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${PAD_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < params.n) {",
  "    let col = i % params.out_len;",
  "    let row = i / params.out_len;",
  "    // always write 0 out of range too (not a single element is left unwritten)",
  "    var value = 0.0;",
  "    if (col >= params.left && col < params.left + params.in_len) {",
  "      value = x[row * params.in_len + (col - params.left)];",
  "    }",
  "    out[i] = value;",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * The uniform Params (4 words = 16 bytes, like the row-op family — the 16-byte unit of the project
 * convention (the fusion pass fixes the same form in ADR 0040)).
 */
export const padParams = (
  rows: number,
  lengthIn: number,
  left: number,
  right: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("pad params", { rows, in_len: lengthIn, left, right });
  const lengthOut = left + lengthIn + right;
  const n = rows * lengthOut;
  assertU32Params("pad params", { out_len: lengthOut, n });
  const params = new Uint32Array(4);
  params[0] = n;
  params[1] = lengthOut;
  params[2] = lengthIn;
  params[3] = left;
  return params;
};
