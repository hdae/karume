/**
 * The fixed kernel of masked_fill (`out = mask ? value : x`, f32 values / bool condition, the fill
 * value a scalar in attrs).
 *
 * The output and `x` have the same shape and are contiguous; **only mask is read with
 * right-aligned broadcast strides** (measured in two shapes, `mask [1,1,T,T] × x [1,16,T,T]` and
 * `mask [1,T,1024] × x [1,T,1024]` — recon §2). Building the strides follows the same rule as
 * expand in the strided copy family (ADR 0011), so `expandSrcStrides` is shared — "an axis of
 * length 1 has stride 0 / missing leading axes have stride 0 too" is the definition of the same
 * right-aligned broadcast, and writing it twice would split the rule over two places.
 *
 * The rank is fixed at {@link STRIDED_RANK}, as in the strided family, to fold into one pipeline.
 * The caller left-pads a shorter real rank with 1 (dims) and 0 (strides).
 *
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535).
 * MUST: the fill value travels in params as **an f32 bit pattern**. Baking it into the WGSL as a
 * constant makes a separate kernel per value, and the pipeline cache grows by the number of
 * distinct fill values (-3.4e38 and 0 become different kernels).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { STRIDED_RANK } from "../codegen/strided.ts";

export const MASKED_FILL_WORKGROUP_SIZE = 256;

export const MASKED_FILL_KEY = `masked_fill:v1:f32:bool:r${STRIDED_RANK}:wg${MASKED_FILL_WORKGROUP_SIZE}`;

/**
 * Layout of params (storage, a u32 array):
 * `[0]=element count n, [1..STRIDED_RANK]=output dims, then as many mask strides, the fill value
 * last`. It is passed as storage, not uniform, because this family has no workgroupBarrier and is
 * not constrained by the uniformity analysis (the same reason as elementwise / strided).
 */
const dimAt = (d: number): number => 1 + d;
const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
const VALUE_AT = 1 + 2 * STRIDED_RANK;

export const MASKED_FILL_WGSL: string = [
  `// karume masked_fill (out = select(x, value, mask), rank ${STRIDED_RANK}, f32 / mask bool)`,
  "@group(0) @binding(0) var<storage, read> params: array<u32>;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read> mask: array<u32>;",
  "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${MASKED_FILL_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let n = params[0u];",
  `  let fill = bitcast<f32>(params[${VALUE_AT}u]);`,
  `  let stride = nwg.x * ${MASKED_FILL_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < n) {",
  "    var rem = i;",
  ...Array.from({ length: STRIDED_RANK - 1 }, (_, index) => {
    const d = STRIDED_RANK - 1 - index;
    return `    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`;
  }),
  "    let c0 = rem;",
  `    let mask_index = ${Array.from(
    { length: STRIDED_RANK },
    (_, d) => `c${d} * params[${strideAt(d)}u]`,
  ).join(" + ")};`,
  "    // bool is stored as u32 0 / 1 (ADR 0009) — only the true positions are replaced with the fill value",
  "    out[i] = select(x[i], fill, mask[mask_index] != 0u);",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * Builds the contents of the params buffer. A shorter real rank is padded **on the left** with
 * dims=1 / strides=0 (the same padding as the strided family — the coordinates of the extra
 * leading axes are always 0 and do not contribute to the read position).
 *
 * MUST: the fill value is rounded to f32 before it is stored. The IR's attrs are JSON f64, so
 * without rounding the CPU reference (`Math.fround`) and the GPU (an f32 word) could see different
 * values.
 */
export const maskedFillParams = (
  outShape: readonly number[],
  maskStrides: readonly number[],
  value: number,
): Uint32Array<ArrayBuffer> => {
  if (outShape.length < 1 || outShape.length > STRIDED_RANK) {
    throw new CodegenError(
      `masked_fill params: output rank ${outShape.length} is outside 1..${STRIDED_RANK}`,
    );
  }
  if (maskStrides.length !== outShape.length) {
    throw new CodegenError(
      `masked_fill params: mask stride count ${maskStrides.length} differs from output rank ${outShape.length}`,
    );
  }
  // MUST: the finiteness test is done **as an f32** (`1e39`, finite in f64, becomes `+Inf` as an
  // f32 word, and a non-finite value would be written to an output that specified "a finite fill
  // value" — the same check as the contract layer).
  if (!Number.isFinite(Math.fround(value))) {
    throw new CodegenError(`masked_fill params: fill value is not finite as f32 (${value})`);
  }
  const n = outShape.reduce((count, dim) => count * dim, 1);
  assertU32Params("masked_fill params", {
    ...Object.fromEntries(outShape.map((dim, d) => [`out_dims[${d}]`, dim])),
    ...Object.fromEntries(maskStrides.map((stride, d) => [`mask_strides[${d}]`, stride])),
    n,
  });
  const pad = STRIDED_RANK - outShape.length;
  const params = new Uint32Array(2 + 2 * STRIDED_RANK);
  params[0] = n;
  for (let d = 0; d < STRIDED_RANK; d += 1) {
    params[dimAt(d)] = d < pad ? 1 : outShape[d - pad];
    params[strideAt(d)] = d < pad ? 0 : maskStrides[d - pad];
  }
  new Float32Array(params.buffer)[VALUE_AT] = value;
  return params;
};
