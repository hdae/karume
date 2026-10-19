/**
 * WGSL generation for reduce kernels (one axis, no keepdim). It has **two variants**:
 *
 * - **Row reduce** (last dimension): treats the input as the contiguous layout `[rows, dim]` with
 *   the leading dimensions flattened. One row = one workgroup (256) + a 256-wide bit-reversed
 *   binary tree.
 * - **Axis reduce** (any dimension but the last): one thread = one output. The reduced axis is
 *   stepped by `inner` (= the product of the dimensions after the reduced axis), so **adjacent
 *   threads read contiguous addresses** (coalesced).
 *
 * MUST: the axis variant's **reduction order matches the row reduce exactly** (design and symbolic
 * verification in docs/research/2026-08-04-vae-axis-reduce-recon.md §5.2). That the output stays
 * bit-identical when permute + row reduce is replaced by axis reduce is what makes this family's
 * numeric contract hold (the PNG sha256 check, atol=0 in tests/gpu_reduce_axis_parity_test.ts).
 *
 * MUST: generation is deterministic (same key → byte-identical WGSL). Key and WGSL are built from
 * the same canonicalization.
 * MUST: one row = one workgroup as the base, while **the row direction is grid-stride too**. Using
 * the row count as the dispatch count makes real models exceed the per-dimension workgroup limit
 * (65535 by default in the spec).
 * MUST: the element type (of both input and output) goes into the key. `sum` has two forms, the f32
 * sum and **the count of a bool (→ i32)**; without it in the key, one form's pipeline is assigned to
 * the other's dispatch and the bit pattern is reinterpreted without an exception (ADR 0009).
 *
 * params are passed as a uniform because the row loop contains a workgroupBarrier. The loop
 * condition must be uniform within the workgroup, and loads from storage are not proven uniform by
 * the uniformity analysis.
 *
 * MUST: `amax` / `amin` return NaN when the reduced elements contain a NaN (the same as torch). The
 * driver's `max` / `min` swallow NaN, so propagation is carried by the bit-pattern check
 * {@link IS_NAN_BITS_WGSL}.
 */

import type { IrDtype } from "../format/ir.ts";
import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL, NAN_MIN_WGSL } from "./numerics-wgsl.ts";
import { outputDtypeOf, REDUCE_OPS, type ReduceOpName, resolveOpContract } from "../ops.ts";
import { CodegenError } from "./errors.ts";
import { assertU32Params } from "./params.ts";

const REDUCE_WORKGROUP_SIZE = 256;

export type ReduceSpec = {
  readonly op: ReduceOpName;
  /** Semantic dtype of the input (the output follows from the contract table's mapping). */
  readonly dtype: IrDtype;
};

type CanonicalSpec = {
  readonly op: ReduceOpName;
  readonly from: IrDtype;
  readonly to: IrDtype;
};

const canonicalize = (spec: ReduceSpec): CanonicalSpec => {
  const op = REDUCE_OPS.find((name) => name === spec.op);
  if (op === undefined) {
    throw new CodegenError(`reduce codegen: op '${spec.op}' is not in the reduce vocabulary`);
  }
  const contract = resolveOpContract(op);
  if (!contract.dtypes.includes(spec.dtype)) {
    throw new CodegenError(`reduce codegen: op '${op}' cannot run dtype '${spec.dtype}'`);
  }
  // Row reduce has a single output, so output slot 0.
  return { op, from: spec.dtype, to: outputDtypeOf(contract, 0, spec.dtype, "reduce codegen") };
};

/** Semantic dtype → WGSL scalar type. A bool is stored as u32 0 / 1 (ADR 0009). */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

/** The largest finite f32. WGSL has no infinity literal, so this serves as the identity. */
const F32_MAX = "3.402823466e38";

const IDENTITY: Readonly<Record<ReduceOpName, Readonly<Partial<Record<IrDtype, string>>>>> = {
  sum: { f32: "0.0", i32: "0i" },
  amax: { f32: `-${F32_MAX}` },
  amin: { f32: F32_MAX },
};

/**
 * NaN-propagating `max` / `min`. **The reduction has two stages (one thread's scan + the
 * workgroup tree)**, so propagation is closed inside the combine function (guarding only one stage
 * lets the identity swallow the NaN). For non-NaN values it is the plain `max` / `min`, so existing
 * numeric results do not move.
 */
const NAN_PROPAGATING_FN: Readonly<Partial<Record<ReduceOpName, string>>> = {
  amax: NAN_MAX_WGSL,
  amin: NAN_MIN_WGSL,
};

const COMBINE: Readonly<Record<ReduceOpName, (a: string, b: string) => string>> = {
  sum: (a, b) => `${a} + ${b}`,
  amax: (a, b) => `nan_max(${a}, ${b})`,
  amin: (a, b) => `nan_min(${a}, ${b})`,
};

/**
 * How a value is read into the accumulator. A bool input counts "the number of trues", so it drops
 * to an i32 of 0 / 1 (a form that counts by truth even when the storage convention is broken —
 * `i32(x)` would turn the bit pattern into an integer as is).
 */
const load = (expr: string, from: IrDtype): string =>
  from === "bool" ? `select(0i, 1i, ${expr} != 0u)` : expr;

/**
 * v2: amax / amin gained the NaN-propagating combine function and their WGSL body changed. The
 * version is bumped **per family** (`sum` only changes its key and its generated code is identical
 * — there is no per-op exception table).
 */
export const reduceKey = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  return `reduce:v2:${op}:${from}>${to}:wg${REDUCE_WORKGROUP_SIZE}`;
};

export const reduceWgsl = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  const combine = COMBINE[op];
  const identity = IDENTITY[op][to];
  if (identity === undefined) {
    throw new CodegenError(`reduce codegen: op '${op}' has no identity for output dtype '${to}'`);
  }
  const accumulator = WGSL_SCALAR[to];
  // When both types are the same, write one word (the f32 form's output is frozen byte for byte by
  // the snapshots).
  const label = from === to ? from : `${from}>${to}`;
  const propagate = NAN_PROPAGATING_FN[op];
  const helpers = propagate === undefined ? [] : [IS_NAN_BITS_WGSL, propagate];
  return [
    `// karume row reduce ${op} (last dim, ${label}, generated)`,
    "struct Params {",
    "  rows: u32,",
    "  dim: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    `@group(0) @binding(1) var<storage, read> x: array<${WGSL_SCALAR[from]}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${accumulator}>;`,
    "",
    ...helpers.flatMap((fn) => [fn, ""]),
    `var<workgroup> scratch: array<${accumulator}, ${REDUCE_WORKGROUP_SIZE}>;`,
    "",
    `@compute @workgroup_size(${REDUCE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(workgroup_id) wid: vec3<u32>,",
    "  @builtin(local_invocation_id) lid3: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let lid = lid3.x;",
    "  let dim = params.dim;",
    "  var row = wid.x;",
    "  while (row < params.rows) {",
    "    let base = row * dim;",
    `    var acc = ${identity};`,
    "    var i = lid;",
    "    while (i < dim) {",
    `      acc = ${combine("acc", load("x[base + i]", from))};`,
    `      i = i + ${REDUCE_WORKGROUP_SIZE}u;`,
    "    }",
    "    scratch[lid] = acc;",
    "    workgroupBarrier();",
    `    var stride = ${REDUCE_WORKGROUP_SIZE / 2}u;`,
    "    while (stride > 0u) {",
    "      if (lid < stride) {",
    `        scratch[lid] = ${combine("scratch[lid]", "scratch[lid + stride]")};`,
    "      }",
    "      workgroupBarrier();",
    "      stride = stride / 2u;",
    "    }",
    "    if (lid == 0u) {",
    "      out[row] = scratch[0u];",
    "    }",
    "    // line up the end of the scratch[0] reads before the next row overwrites scratch[lid]",
    "    workgroupBarrier();",
    "    row = row + nwg.x;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * The uniform Params. 16 bytes are allocated even for two words of content (allocating in 16-byte
 * units is a project convention <the fusion pass pins the same form in ADR 0040>, not a spec
 * requirement — WGSL's minimum binding size is the struct's actual size <8 bytes here>).
 */
export const reduceParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("reduce params", { rows, dim });
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  return params;
};

/**
 * The parallel shape of axis reduce (one thread = one output). **Being the same size as the row
 * reduce** is the condition for the reduction orders to match, so it is defined from
 * {@link REDUCE_WORKGROUP_SIZE} rather than written as a separate number.
 */
export const AXIS_REDUCE_WORKGROUP_SIZE = REDUCE_WORKGROUP_SIZE;

/**
 * log2 of the workgroup size (rejected at the generation entry when it is not a power of two).
 *
 * MUST: the carry-stack depth (`log2 + 1`) and the bit-reversal shift (`32 − log2`) are derived
 * from the size. Hand-copied constants side by side let one of them drift into **a silent wrong
 * value without an exception** (too few levels read a level that was never written; a shifted
 * shift puts the leaves in an order different from the row reduce's tree). A size that is not a
 * power of two not only makes the depth fractional, it also breaks the row reduce's tree reduction
 * (halving the stride).
 */
const workgroupSizeBits = (size: number): number => {
  const bits = Math.log2(size);
  if (!Number.isInteger(bits)) {
    throw new CodegenError(`reduce codegen: workgroup size ${size} is not a power of 2`);
  }
  return bits;
};

/**
 * Key of the axis reduce variant. It is **a separate pipeline** from the row reduce
 * ({@link reduceKey}), and the existing key strings do not change by a single byte (a separate
 * variant so that the default path's WGSL / keys do not move).
 */
export const axisReduceKey = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  return `reduce:v2:${op}:${from}>${to}:axis:wg${AXIS_REDUCE_WORKGROUP_SIZE}`;
};

/**
 * The variant that reduces an axis other than the last one. One thread owns one output and scans
 * the reduced axis in steps of `inner`.
 *
 * Reproducing the reduction order (**MUST** — the basis of bit identity):
 *
 * 1. The row reduce has "lane `i` fold `x[i], x[i+256], x[i+512], …` left-associatively starting
 *    from the identity", so slot `i` is built here in the same order starting from the same
 *    identity (the `+0.0` fold is not skipped — the row reduce actually adds the identity too, so
 *    the behaviour matches down to the one exception `-0.0 + 0.0 = +0.0`).
 * 2. The row reduce's tree is `scratch[lid] = combine(scratch[lid], scratch[lid + stride])`
 *    (stride = 128 → 1), so the first pairs formed are (0,128) / (1,129) … = **the leaf order is
 *    bit-reversed**. Pushing the slots onto a carry-stack (a binary counter) in `reverseBits` order
 *    makes the symbolic expression match this tree exactly (verified on 21 cases with dim 1–512 —
 *    recon §5.2).
 * 3. The combine argument order is "the side pushed first is the first argument" (the tree's
 *    `scratch[lid]` is the first argument). NaN propagation in `amax` / `amin` is symmetric so the
 *    value does not change, but `sum`'s rounding depends on the order.
 */
export const axisReduceWgsl = (spec: ReduceSpec): string => {
  const { op, from, to } = canonicalize(spec);
  const combine = COMBINE[op];
  const identity = IDENTITY[op][to];
  if (identity === undefined) {
    throw new CodegenError(`reduce codegen: op '${op}' has no identity for output dtype '${to}'`);
  }
  const accumulator = WGSL_SCALAR[to];
  const label = from === to ? from : `${from}>${to}`;
  const propagate = NAN_PROPAGATING_FN[op];
  const helpers = propagate === undefined ? [] : [IS_NAN_BITS_WGSL, propagate];
  const size = AXIS_REDUCE_WORKGROUP_SIZE;
  const bits = workgroupSizeBits(size);
  // After `size` pushes only the top level is valid, so `log2(size) + 1` levels are needed.
  const stackDepth = bits + 1;
  return [
    `// karume axis reduce ${op} (non-last dim, ${label}, generated)`,
    "struct Params {",
    "  out_count: u32,",
    "  axis_len: u32,",
    "  inner: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    `@group(0) @binding(1) var<storage, read> x: array<${WGSL_SCALAR[from]}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${accumulator}>;`,
    "",
    ...helpers.flatMap((fn) => [fn, ""]),
    `@compute @workgroup_size(${size})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let inner = params.inner;",
    "  let axis_len = params.axis_len;",
    `  let stride = nwg.x * ${size}u;`,
    "  var t = gid.x;",
    "  while (t < params.out_count) {",
    "    // output index t = outer·inner + i; back to the base address with only the reduced axis removed",
    "    let base = (t / inner) * axis_len * inner + (t % inner);",
    `    var acc: array<${accumulator}, ${stackDepth}>;`,
    `    for (var m = 0u; m < ${size}u; m = m + 1u) {`,
    "      // reproduce the leaf order of the row reduce (bit reversal) in one thread",
    `      let slot = reverseBits(m) >> ${32 - bits}u;`,
    `      var v = ${identity};`,
    "      var c = slot;",
    "      while (c < axis_len) {",
    `        v = ${combine("v", load("x[base + c * inner]", from))};`,
    `        c = c + ${size}u;`,
    "      }",
    "      var k = 0u;",
    "      while ((m & (1u << k)) != 0u) {",
    `        v = ${combine("acc[k]", "v")};`,
    "        k = k + 1u;",
    "      }",
    "      acc[k] = v;",
    "    }",
    `    out[t] = acc[${stackDepth - 1}u];`,
    "    t = t + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * The uniform Params of axis reduce (`{out_count, axis_len, inner}`, 16 bytes allocated).
 *
 * `inner` is the product of the dimensions after the reduced axis. It is 0 only when a later axis
 * has length 0, and then `out_count` is 0 too, so `t / inner` is never evaluated (the loop is not
 * entered).
 */
export const axisReduceParams = (
  outCount: number,
  axisLen: number,
  inner: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("axis reduce params", {
    out_count: outCount,
    axis_len: axisLen,
    inner,
  });
  const params = new Uint32Array(4);
  params[0] = outCount;
  params[1] = axisLen;
  params[2] = inner;
  return params;
};
