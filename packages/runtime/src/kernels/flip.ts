/**
 * The fixed kernel of flip (index reversal along a static axis, f32).
 *
 * The axis position is carried folded into three parts `[outer, len, inner]` (`outer` = the
 * product of the dimensions before the flipped axis, `inner` = the product of those after it).
 * Rank is not carried in params, so no rank limit is needed, and turning the strided family's
 * strides negative (a full revision of the existing params and the determinism snapshots) is
 * avoided too — exactly the reason ADR 0014 chose "one new kernel".
 *
 * MUST: every byte of the output is written (the full-write invariant — ADR 0014 / 0004). The
 * reversal is a bijection, so the grid-stride of one dispatch = one output element covers the
 * whole range as is.
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535).
 * MUST: the reversal is `len - 1 - c`. With `len - c` the first element reads out of range, and
 * since neither the shape nor the element count changes it is detectable only in the values
 * (tests/gpu_ops_test.ts pins it with an asymmetric sequence).
 */

import { assertU32Params } from "../codegen/params.ts";

export const FLIP_WORKGROUP_SIZE = 256;

export const FLIP_KEY = `flip:v1:f32:wg${FLIP_WORKGROUP_SIZE}`;

export const FLIP_WGSL: string = [
  "// karume flip (static axis, f32, full-write)",
  "struct Params {",
  "  n: u32,",
  "  len: u32,",
  "  inner: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${FLIP_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${FLIP_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < params.n) {",
  "    let tail = i % params.inner;",
  "    let rest = i / params.inner;",
  "    let axis = rest % params.len;",
  "    let head = rest / params.len;",
  "    let src = (head * params.len + (params.len - 1u - axis)) * params.inner + tail;",
  "    out[i] = x[src];",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * The uniform Params (three words, but 16 bytes are allocated — allocating in 16-byte units is a
 * project convention (the fusion pass fixes the same form in ADR 0040), not a requirement of the
 * specification).
 * `outer` is not carried (recovering it from `n = outer · len · inner` is unnecessary, and carrying
 * it would hold the same fact twice).
 */
export const flipParams = (
  outer: number,
  length: number,
  inner: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("flip params", { outer, len: length, inner });
  const n = outer * length * inner;
  assertU32Params("flip params", { n });
  const params = new Uint32Array(4);
  params[0] = n;
  params[1] = length;
  params[2] = inner;
  return params;
};
