/**
 * CPU reference implementations of the ops (naive, GPU-independent). The oracle that GPU output is
 * compared against in the numeric verification of kernels (stage 2 of ADR 0005).
 *
 * Approach: a literal transcription with no thought for speed. f32 elementwise ops apply
 * `Math.fround` after every operation to follow f32 rounding; reductions (matmul / row reduce)
 * accumulate in f64 and round once when stored (the GPU side accumulates sequentially in f32, so
 * they do not match exactly — which is why allclose decides).
 * i32 / bool leave no room for approximation, so **exact match** is the expectation (the comparing
 * side does not use allclose).
 *
 * MUST: do not share the coordinate → index computation with codegen. Building the strides is
 * exactly what is being verified, and calling the same implementation would leave tests unable to
 * detect stride bugs (a deliberate double implementation).
 * MUST: write the value expressions in a different form from codegen too (cast's truncation is
 * `Math.trunc` against WGSL's `i32()`, conversion to bool is a comparison against `select(…)`).
 * Copying the same expression lets an error cancel out on both sides.
 */

import type { IrDtype } from "../format/ir.ts";
import {
  ARGMAX_OP,
  arityFits,
  assertAttrKeys,
  assertAttrValues,
  assertDtype,
  assertInputDtypes,
  assertSlotDtype,
  attentionScale,
  type BinaryOpName,
  castTargetDtype,
  catDim,
  computeOutputShape,
  conv1dAttrs,
  conv2dAttrs,
  conv3dAttrs,
  convTranspose1dAttrs,
  deformConv2dAttrs,
  describeArity,
  flipDim,
  type GruScanOpName,
  layerNormAttrs,
  maskedFillValue,
  numel,
  type OpContract,
  outputDtypeOf,
  padAttrs,
  permuteDims,
  reduceDim,
  type ReduceOpName,
  resolveOpContract,
  rmsNormEps,
  scalarParamValues,
  sliceAttrs,
  staticQuantizeScale,
  TOPK_OP,
  topkK,
  type UnaryOpName,
  upsampleBilinear2dAttrs,
} from "../ops.ts";

/**
 * A value passed to a reference implementation does not meet the contract (element count and shape
 * disagree, etc.).
 */
export class ReferenceOpError extends Error {
  override readonly name = "ReferenceOpError";
}

type RefTensorOf<D extends IrDtype, A> = {
  readonly dtype: D;
  readonly shape: readonly number[];
  readonly data: A;
};

/** The same dtype-discriminated union as the public Tensor (bool is u32 0 / 1 — ADR 0009). */
export type RefTensor =
  | RefTensorOf<"f32", Float32Array<ArrayBuffer>>
  | RefTensorOf<"i32", Int32Array<ArrayBuffer>>
  | RefTensorOf<"bool", Uint32Array<ArrayBuffer>>;

/**
 * Decides the dtype from the given TypedArray (the mapping between array type and discriminator
 * lives in one place).
 */
export const refTensor = (
  shape: readonly number[],
  data: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
): RefTensor => {
  // MUST: check each dimension. Checking only the product lets [-1,-1] pass as 1 element and builds
  // a meaningless expected value from a tensor with negative dimensions.
  const bad = shape.findIndex((dim) => !Number.isSafeInteger(dim) || dim < 0);
  if (bad >= 0) {
    throw new ReferenceOpError(
      `shape [${shape.join(",")}] の次元 ${bad}（${shape[bad]}）が非負の整数でない`,
    );
  }
  if (data.length !== numel(shape)) {
    throw new ReferenceOpError(`要素数 ${data.length} が shape [${shape.join(",")}] と合わない`);
  }
  if (data instanceof Float32Array) return { dtype: "f32", shape, data };
  if (data instanceof Int32Array) return { dtype: "i32", shape, data };
  return { dtype: "bool", shape, data };
};

/**
 * Stores a number sequence into the TypedArray of the dtype. For i32, storing into an `Int32Array`
 * itself wraps in two's complement, so it wraps the same way as the GPU side (WGSL i32 arithmetic).
 */
const materialize = (
  dtype: IrDtype,
  shape: readonly number[],
  values: readonly number[],
): RefTensor => {
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: Float32Array.from(values) };
    case "i32":
      return { dtype, shape, data: Int32Array.from(values) };
    case "bool":
      return { dtype, shape, data: Uint32Array.from(values) };
  }
};

/**
 * Maclaurin series of erf (summed in f64 until it converges). For |x| > 4, erfc(x) < 1.6e-8 cannot
 * be told apart from the f32 1.0, so it rounds to ±1.
 *
 * MUST: do not copy the GPU side's formula (Abramowitz–Stegun 7.1.26). With the same approximation
 * the errors cancel out and gelu's comparison test becomes vacuous.
 */
const erf = (x: number): number => {
  const a = Math.abs(x);
  if (a > 4) return Math.sign(x);
  let term = a;
  let sum = a;
  for (let n = 1; n < 200; n += 1) {
    term *= -(a * a) / n;
    const contribution = term / (2 * n + 1);
    sum += contribution;
    if (Math.abs(contribution) < 1e-18 * Math.abs(sum)) break;
  }
  return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
};

/**
 * f32 → f32 unary ops (bitwise_not is bool-only and the three comparisons return bool, so they
 * are handled separately).
 * The second argument is the scalars from attrs (in the order of {@link scalarParamValues}).
 *
 * MUST: `log1p` uses `Math.log1p` (does not copy the GPU side's corrected formula). Writing the
 * same formula would leave the comparison unable to detect that "the correction is off" — the same
 * discipline as erf / softmax.
 * MUST: `leaky_relu` / `clamp` do not copy the GPU side's select form either; they are written as
 * branches with the comparison reversed. Every comparison with NaN is false, so NaN passes through
 * in either form (= the same propagation as torch).
 */
const UNARY_F32: Readonly<
  Record<
    Exclude<UnaryOpName, "bitwise_not" | "ge_scalar" | "le_scalar" | "gt_scalar">,
    (x: number, scalars: readonly number[]) => number
  >
> = {
  neg: (x) => -x,
  abs: (x) => Math.abs(x),
  exp: (x) => Math.exp(x),
  log: (x) => Math.log(x),
  log1p: (x) => Math.log1p(x),
  sqrt: (x) => Math.sqrt(x),
  // Like sqrt / tanh, compared built-in against built-in (unlike log1p / erf, neither side is an
  // approximation, so no other formula exists to detect a copying mistake). NOTE: WGSL guarantees
  // the accuracy of `sin` only for |x| ≤ π, and agreement outside it is implementation-dependent
  // — the comparison is decided by allclose.
  sin: (x) => Math.sin(x),
  tanh: (x) => Math.tanh(x),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  relu: (x) => Math.max(x, 0),
  // torch's default gelu (approximate="none") = 0.5·x·(1 + erf(x/√2))
  gelu: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
  // torch's approximate="tanh". **Written with the same formula as the GPU side here** — unlike
  // erf / log1p, this formula is torch's definition, not an approximation, so a different formula
  // would compare against "a different function" (what should be detected is an implementation
  // mix-up, not an approximation miss). 0.7978845608028654 = √(2/π).
  gelu_tanh: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
  clamp: (x, [min, max]) => (x < min ? min : x > max ? max : x),
  // MUST: do not write the branch as `x >= min ? x : min` — NaN falls to false and returns min,
  // silently departing from torch's `clamp(NaN, min) = NaN` (this one line is the direction
  // detector).
  // The GPU side carries NaN propagation with the bit-pattern shell (src/codegen/elementwise.ts)
  // — the reference does not copy that expression and is written independently (the same
  // discipline as log1p / erf).
  clamp_min: (x, [min]) => (x < min ? min : x),
  leaky_relu: (x, [slope]) => (x < 0 ? slope * x : x),
};

/**
 * The empty list for looking up {@link UNARY_F32} without attrs scalars (the table is shared with
 * the clamp family, so the second argument is needed — the expressions are not copied just to call
 * them differently).
 */
const NO_SCALARS: readonly number[] = [];

/** f32 → bool comparisons (against a scalar from attrs). bool is u32 0 / 1 (ADR 0009). */
const COMPARE_SCALAR: Readonly<
  Record<"ge_scalar" | "le_scalar" | "gt_scalar", (x: number, value: number) => boolean>
> = {
  ge_scalar: (x, value) => x >= value,
  le_scalar: (x, value) => x <= value,
  gt_scalar: (x, value) => x > value,
};

/** f32 arithmetic (comparison / logic are separate tables — their output dtype differs). */
const BINARY_F32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => number>>> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
};

/**
 * i32 binary ops. JS numbers are f64, so the folding into 32 bits is explicit
 * (`Math.imul` / `| 0`) — a plain multiplication silently loses precision once it exceeds 2^53.
 * Only mul / sub are allowed (the contract table — only the forms that appear in real graphs).
 */
const BINARY_I32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => number>>> = {
  sub: (a, b) => (a - b) | 0,
  mul: (a, b) => Math.imul(a, b),
};

/** f32 × f32 → bool comparisons (the output is u32 0 / 1 — ADR 0009). */
const COMPARE_F32: Readonly<Partial<Record<BinaryOpName, (a: number, b: number) => boolean>>> = {
  ge: (a, b) => a >= b,
};

/**
 * bool × bool → bool logical ops.
 * MUST: a different form from the GPU side (`select` + `!= 0u`) — drop to JS booleans, write the
 * plain logical operation, and convert to a number only once at the end.
 */
const BINARY_BOOL: Readonly<Partial<Record<BinaryOpName, (a: boolean, b: boolean) => boolean>>> = {
  bitwise_and: (a, b) => a && b,
};

/**
 * Binary operator per dtype (a missing combination has no reference implementation even if it
 * passed the contract check = an internal inconsistency).
 */
const binaryEvaluator = (op: BinaryOpName, dtype: IrDtype): ((a: number, b: number) => number) => {
  if (dtype === "bool") {
    const logical = BINARY_BOOL[op];
    if (logical !== undefined) return (a, b) => (logical(a !== 0, b !== 0) ? 1 : 0);
  } else if (dtype === "i32") {
    const integer = BINARY_I32[op];
    if (integer !== undefined) return integer;
  } else {
    const compare = COMPARE_F32[op];
    if (compare !== undefined) return (a, b) => (compare(a, b) ? 1 : 0);
    const arithmetic = BINARY_F32[op];
    if (arithmetic !== undefined) return (a, b) => Math.fround(arithmetic(a, b));
  }
  throw new ReferenceOpError(`op '${op}' に dtype '${dtype}' の参照実装が無い`);
};

export const referenceUnary = (
  op: UnaryOpName,
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>> = {},
): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, x.dtype, "reference");
  const values = new Array<number>(x.data.length);
  if (op === "bitwise_not") {
    // Negation of a bool (u32 0 / 1). A different form from the WGSL side's select (the result
    // of a comparison converted to a number).
    for (let i = 0; i < values.length; i += 1) values[i] = x.data[i] === 0 ? 1 : 0;
    return materialize(x.dtype, [...x.shape], values);
  }
  // MUST: scalar attrs are rounded to f32 before use (the GPU side receives them as f32 words of
  // params).
  const scalars = scalarParamValues(contract, attrs, "reference").map((value) =>
    Math.fround(value),
  );
  const outDtype = outputDtypeOf(contract, 0, x.dtype, "reference");
  if (op === "ge_scalar" || op === "le_scalar" || op === "gt_scalar") {
    const compare = COMPARE_SCALAR[op];
    for (let i = 0; i < values.length; i += 1) values[i] = compare(x.data[i], scalars[0]) ? 1 : 0;
    return materialize(outDtype, [...x.shape], values);
  }
  const apply = UNARY_F32[op];
  for (let i = 0; i < values.length; i += 1) {
    values[i] = Math.fround(apply(x.data[i], scalars));
  }
  return materialize(outDtype, [...x.shape], values);
};

/**
 * Flat output index → flat input index (right-aligned broadcast).
 *
 * MUST: do not build a stride table. The codegen side (`elementwiseParams` in elementwise.ts)
 * builds strides as "accumulate a running product from the right and zero the dimensions of 1", so
 * writing the same formula on the reference side lets a stride-derivation error cancel out on both
 * sides and makes the comparison test against the GPU vacuous.
 * Here it is derived with a different structure — "decompose the output coordinates per element
 * with divmod → fold the input shape from the left by Horner's scheme". The basis of the
 * cross-check is the independence of the implementations itself, so neither helpers nor
 * expression shapes are shared with codegen.
 *
 * Leading dimensions that do not fall into the input rank under right alignment always have
 * coordinate 0 on the input side (they do not enter the Horner fold). Dimensions of length 1 also
 * collapse to coordinate 0 (this is the broadcast).
 */
const broadcastIndex = (
  outIndex: number,
  outShape: readonly number[],
  inShape: readonly number[],
): number => {
  const rank = outShape.length;
  const coords = new Array<number>(rank).fill(0);
  let rest = outIndex;
  for (let d = rank - 1; d >= 0; d -= 1) {
    coords[d] = rest % outShape[d];
    rest = Math.floor(rest / outShape[d]);
  }
  const offset = rank - inShape.length;
  let index = 0;
  for (let d = 0; d < inShape.length; d += 1) {
    const extent = inShape[d];
    index = index * extent + (extent === 1 ? 0 : coords[offset + d]);
  }
  return index;
};

export const referenceBinary = (op: BinaryOpName, a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  if (a.dtype !== b.dtype) {
    throw new ReferenceOpError(`op '${op}' の入力 dtype が混在（${a.dtype} / ${b.dtype}）`);
  }
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const evaluate = binaryEvaluator(op, a.dtype);
  const values = new Array<number>(numel(shape));
  for (let i = 0; i < values.length; i += 1) {
    const va = a.data[broadcastIndex(i, shape, a.shape)];
    const vb = b.data[broadcastIndex(i, shape, b.shape)];
    values[i] = evaluate(va, vb);
  }
  return materialize(outputDtypeOf(contract, 0, a.dtype, "reference"), shape, values);
};

/**
 * where `out = cond ? a : b` (all three right-aligned broadcast).
 *
 * The index derivation is {@link broadcastIndex} — "divmod the output coordinates per element →
 * Horner over the input shape from the left", a structure different from codegen's stride table
 * (the MUST at the top of this file).
 * MUST: the branch direction is torch's `where(cond, a, b)` (a when true). A mix-up changes neither
 * shape nor dtype, so only the values can detect it (tests fill a / b with different value ranges).
 */
export const referenceWhere = (cond: RefTensor, a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("where");
  assertSlotDtype(contract, 0, cond.dtype, "reference");
  assertSlotDtype(contract, 1, a.dtype, "reference");
  assertSlotDtype(contract, 2, b.dtype, "reference");
  const shape = computeOutputShape(contract, [cond.shape, a.shape, b.shape], "reference")[0];
  const values = new Array<number>(numel(shape));
  for (let i = 0; i < values.length; i += 1) {
    values[i] =
      cond.data[broadcastIndex(i, shape, cond.shape)] !== 0
        ? a.data[broadcastIndex(i, shape, a.shape)]
        : b.data[broadcastIndex(i, shape, b.shape)];
  }
  return materialize(outputDtypeOf(contract, 0, cond.dtype, "reference"), shape, values);
};

/**
 * Prefix sum over the last dimension `out[…, j] = Σ_{i ≤ j} x[…, i]`.
 *
 * MUST: compute **each output as the sum of an independent inner loop** (O(n²)). The kernel
 * carries an accumulator along its scan, so writing the same recurrence would let errors of
 * accumulation direction, initial value and store position cancel out on both sides. Rows are ~10
 * long as measured (recon §2) and tests are small, so the naive double sum is fast enough.
 * MUST: the inner loop **rounds with `Math.fround` after every term** (reproducing f32
 * accumulation step by step). JS numbers are f64, so summing everything and rounding once would
 * give the reference alone an f64-precision sum, and the longer the row, the more the gap to the
 * GPU opens through "a different rounding path, not an implementation bug" (at row length 300 the
 * oracle's resolution eats up the tolerance). The double sum's **independence is kept**; only the
 * rounding path is aligned with the kernel.
 */
export const referenceCumsum = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("cumsum");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    for (let last = 0; last < dim; last += 1) {
      let acc = 0;
      for (let i = 0; i <= last; i += 1) acc = Math.fround(acc + x.data[base + i]);
      out[base + last] = acc;
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * Conversion between semantic dtypes. As in the contract (src/ops.ts),
 * **f32 → integer truncates toward 0 (torch's truncate)**, **x → bool is x != 0**, and
 * bool → number reads the u32 0 / 1 as is.
 */
export const referenceCast = (x: RefTensor, to: IrDtype): RefTensor => {
  const values = new Array<number>(x.data.length);
  for (let i = 0; i < values.length; i += 1) {
    const v = x.data[i];
    values[i] = to === "bool" ? (v !== 0 ? 1 : 0) : to === "i32" ? Math.trunc(v) : Math.fround(v);
  }
  return materialize(to, [...x.shape], values);
};

/** Row-major C[m,n] = A[m,k] · B[k,n]. Accumulates in f64 and rounds to f32 when stored. */
export const referenceMatmul = (a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("matmul");
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const [m, n] = shape;
  const k = a.shape[1];
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      let acc = 0;
      for (let i = 0; i < k; i += 1) acc += a.data[row * k + i] * b.data[i * n + col];
      out[row * n + col] = Math.fround(acc);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * Batched matmul `C[b,m,n] = A[b,m,k] · B[b,k,n]`. Like matmul, accumulates in f64 and rounds to
 * f32 once when stored.
 *
 * MUST: build the batch offset from "the element count of one matrix" **independently per batch**.
 * The GPU side derives the same quantity from the dispatch's z axis, so compare on shapes where an
 * axis mix-up on either side always makes the values disagree (cases where B / M / K / N all have
 * different lengths).
 */
export const referenceBmm = (a: RefTensor, b: RefTensor): RefTensor => {
  const contract = resolveOpContract("bmm");
  assertDtype(contract, a.dtype, "reference");
  assertDtype(contract, b.dtype, "reference");
  const shape = computeOutputShape(contract, [a.shape, b.shape], "reference")[0];
  const [batch, m, n] = shape;
  const k = a.shape[2];
  const out = new Float32Array(batch * m * n);
  for (let item = 0; item < batch; item += 1) {
    const aBase = item * m * k;
    const bBase = item * k * n;
    const cBase = item * m * n;
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        for (let i = 0; i < k; i += 1) {
          acc += a.data[aBase + row * k + i] * b.data[bBase + i * n + col];
        }
        out[cBase + row * n + col] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * gather over the last dimension. `out[..., j] = src[..., index[..., j]]` (the leading dimensions
 * of src and index match).
 *
 * MUST: **an out-of-range index always throws**. The CPU reference is the contract's oracle, so it
 * is not relaxed to match the GPU side's ruling (out of range contaminates with NaN —
 * src/kernels/gather.ts). Matching it would make "a contract-violating graph give the same values
 * on both sides", and the comparison could no longer detect the violation.
 * MUST: decompose the index in a different form from codegen. The kernel divides the flat output
 * index into rows with `i / J`; here the flat index is built from a double loop over rows and
 * columns.
 */
export const referenceGather = (src: RefTensor, index: RefTensor): RefTensor => {
  const contract = resolveOpContract("gather");
  assertSlotDtype(contract, 0, src.dtype, "reference");
  assertSlotDtype(contract, 1, index.dtype, "reference");
  const shape = computeOutputShape(contract, [src.shape, index.shape], "reference")[0];
  const cols = shape[shape.length - 1];
  const srcCols = src.shape[src.shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const flat = row * cols + col;
      const pick = index.data[flat];
      if (pick < 0 || pick >= srcCols) {
        throw new ReferenceOpError(
          `gather の添字 index[${flat}] = ${pick} が src の最終次元 ${srcCols} の範囲外`,
        );
      }
      out[flat] = src.data[row * srcCols + pick];
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * The largest finite f32. The reduction identity of `amax` / `amin` (WGSL has no infinity literal,
 * so the GPU side uses this value as the identity — the reference aligns with it).
 *
 * MUST: do not import codegen's constant (written out from its decimal notation separately from
 * `src/codegen/reduce.ts` — the discipline that keeps the reference and codegen as a double
 * implementation).
 */
const F32_MAX = Math.fround(3.402823466e38);

/**
 * Reduce over one axis (no keepdim). The reduced axis `axis` is the attrs' `dim` (no default is
 * filled in).
 *
 * The scan "walks the reduced axis in steps of `inner` for each output element", which degenerates
 * to a contiguous scan for the last dimension (`inner == 1`). The GPU side splits into two
 * variants, but **a single version that handles the axis in the index expression** is enough for
 * the reference (accumulating in f64 and rounding once at the end does not depend on the axis).
 *
 * `sum` of a bool input is **the number of trues** (output i32 — the contract table's mapping). It
 * counts in f64 before dropping to the integer array, so it does not round even when the reduced
 * length exceeds 2^24.
 *
 * MUST: `amax` / `amin` fold **starting from** the identity (∓{@link F32_MAX}). Starting from the
 * first element, a row whose reduced axis is all −Infinity returns −Infinity in the reference
 * alone, and the value diverges without an exception from the GPU side (−F32_MAX), which folds
 * with the identity (the tolerance is EXACT, so it is not absorbed). Every finite input lies in
 * [−F32_MAX, F32_MAX], so this starting point does not move a value by a single bit.
 */
export const referenceRowReduce = (op: ReduceOpName, x: RefTensor, axis: number): RefTensor => {
  const contract = resolveOpContract(op);
  assertDtype(contract, x.dtype, "reference");
  const attrs = { dim: axis };
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const outDtype = outputDtypeOf(contract, 0, x.dtype, "reference");
  const dim = x.shape[axis];
  const inner = numel(x.shape.slice(axis + 1));
  const count = numel(shape);
  const values = new Array<number>(count);
  for (let index = 0; index < count; index += 1) {
    // Output index index = outer·inner + i. Map back to the start address with only the reduced
    // axis removed.
    const base = Math.floor(index / inner) * dim * inner + (index % inner);
    if (op === "sum") {
      let acc = 0;
      for (let i = 0; i < dim; i += 1) {
        const value = x.data[base + i * inner];
        acc += x.dtype === "bool" ? (value !== 0 ? 1 : 0) : value;
      }
      values[index] = outDtype === "f32" ? Math.fround(acc) : acc;
      continue;
    }
    let acc = op === "amax" ? -F32_MAX : F32_MAX;
    for (let i = 0; i < dim; i += 1) {
      const value = x.data[base + i * inner];
      acc = op === "amax" ? Math.max(acc, value) : Math.min(acc, value);
    }
    values[index] = acc;
  }
  return materialize(outDtype, shape, values);
};

/**
 * argmax over the last dimension (ADR 0068 decision 2). The output is **an i32 index**, and the
 * shape is the input with its last dimension collapsed to 1 (rank preserved).
 *
 * Fixed behaviour (following torch — measured 2026-08-17):
 *
 * - **Ties go to the smallest index** (`[1,3,3,2]` → 1).
 * - **An all −inf row gives index 0** (unlike `amax`, the identity is an index, not a value, so
 *   the answer is defined even for a −inf row).
 * - **NaN is the largest** (`[1,NaN,3,NaN]` → 1). With several NaNs, the smallest index.
 *
 * MUST: do not copy the GPU side's expression (src/kernels/argmax.ts). That one "folds the
 * lexicographic maximum of (value, index) pairs in a tree", so this one is written in **the naive
 * form that scans one element at a time from the start and updates** (NaN is detected with
 * `Number.isNaN` too, not by copying the bit-pattern check). Writing the same expression would let
 * an error in the tree's association order or a missing NaN check cancel out on both sides.
 */
const referenceArgmax = (x: RefTensor): RefTensor => {
  const contract = resolveOpContract(ARGMAX_OP);
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference")[0];
  const dim = x.shape[x.shape.length - 1];
  const rows = numel(shape);
  const values = new Array<number>(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let bestAt = 0;
    for (let i = 1; i < dim; i += 1) {
      const candidate = x.data[base + i];
      const best = x.data[base + bestAt];
      // NaN is the largest (every comparison with NaN is false, so the NaN side is picked up
      // explicitly).
      // Once a NaN is held, later elements never update it = NaN also gets the smallest index.
      if (Number.isNaN(best)) continue;
      if (Number.isNaN(candidate) || candidate > best) bestAt = i;
    }
    values[row] = bestAt;
  }
  return materialize(outputDtypeOf(contract, 0, x.dtype, "reference"), shape, values);
};

/**
 * top-k over the last dimension (ADR 0068 decision 3). It returns **two outputs** (slot 0 = the
 * values, f32 in descending order; slot 1 = the indices, i32), both `[…, k]`.
 *
 * Fixed behaviour (measured 2026-08-17 / torch 2.13.0+cpu):
 *
 * - **The value sequence is bit-identical to torch** (descending, with the same multiplicity of
 *   ties, so it matches even on rows full of duplicates).
 * - **The index sequence prefers the smallest index among ties**. This is **Karume's rule, not
 *   torch's** (torch does not guarantee the order of ties and contradicts itself with
 *   `topk([5,5,5,5],1)` = 2 against `argmax` = 0 — details in the NOTE of src/kernels/topk.ts).
 *   k=1 gives the same answer as argmax.
 * - **NaN is the largest** (torch puts it first too). With several, the smallest index.
 * - **An all −inf row** also has a defined answer (the identity is a rank, not a value, so k
 *   entries from the smallest index).
 *
 * MUST: do not copy the GPU side's expression (src/kernels/topk.ts). That one is "lane-local
 * top-k → a k-round tournament", so this one is written in **the naive form that sorts the indices
 * in lexicographic order and takes the first k** (NaN detected with `Number.isNaN` too). Writing
 * the same form would let an error at the merge boundaries or a missing NaN cancel out on both
 * sides.
 */
const referenceTopk = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): readonly RefTensor[] => {
  const contract = resolveOpContract(TOPK_OP);
  assertDtype(contract, x.dtype, "reference");
  const shapes = computeOutputShape(contract, [x.shape], "reference", { attrs });
  const k = topkK(attrs, "reference");
  const dim = x.shape[x.shape.length - 1];
  const rows = numel(x.shape) / dim;
  const values = new Array<number>(rows * k);
  const indices = new Array<number>(rows * k);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    // Sort the indices by "value descending, index ascending among ties" (NaN is the largest, so
    // NaN vs NaN / NaN vs number are settled before the comparison).
    const order = Array.from({ length: dim }, (_, i) => i).sort((a, b) => {
      const va = x.data[base + a];
      const vb = x.data[base + b];
      const na = Number.isNaN(va);
      const nb = Number.isNaN(vb);
      if (na !== nb) return na ? -1 : 1;
      if (!na && va !== vb) return va > vb ? -1 : 1;
      return a - b;
    });
    for (let slot = 0; slot < k; slot += 1) {
      values[row * k + slot] = x.data[base + order[slot]];
      indices[row * k + slot] = order[slot];
    }
  }
  return [
    materialize(outputDtypeOf(contract, 0, x.dtype, "reference"), shapes[0], values),
    materialize(outputDtypeOf(contract, 1, x.dtype, "reference"), shapes[1], indices),
  ];
};

/**
 * Reshaping that keeps the element order (ADR 0011's reshape). The data is copied contiguously as
 * is. The execution system aliases the buffer here and makes no copy, while the reference copies it
 * plainly and serves as the oracle that "aliasing does not change the values".
 */
export const referenceReshape = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  if (numel(target) !== x.data.length) {
    throw new ReferenceOpError(
      `reshape の要素数が合わない ${x.data.length} → [${target.join(",")}]`,
    );
  }
  return materialize(x.dtype, target, Array.from(x.data));
};

/**
 * Axis permutation. `dims[d]` = the input dimension that output dimension d takes.
 *
 * MUST: do not share the index computation with codegen (strided.ts). That one is a gather form
 * that reads through "output coordinates → dot product with the input strides", so this one is
 * written in the opposite scatter form — "decompose the flat **input** index into input
 * coordinates → permute into output coordinates → fold the output index by Horner from the left".
 * Writing it in the same direction would let an error in building the stride table cancel out on
 * both sides and make the comparison test vacuous.
 */
export const referencePermute = (x: RefTensor, dims: readonly number[]): RefTensor => {
  const rank = x.shape.length;
  if (dims.length !== rank || new Set(dims).size !== rank) {
    throw new ReferenceOpError(
      `permute の dims [${dims.join(",")}] が shape [${x.shape.join(",")}] の並べ替えでない`,
    );
  }
  const outShape = dims.map((dim) => {
    const extent = x.shape[dim];
    if (extent === undefined) {
      throw new ReferenceOpError(`permute の dims に軸 ${dim} が入っている（rank ${rank}）`);
    }
    return extent;
  });
  const values = new Array<number>(x.data.length);
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * outShape[d] + coords[dims[d]];
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, outShape, values);
};

/**
 * Replicates only the dimensions of length 1 to the target shape (right-aligned). The index
 * derivation is {@link broadcastIndex} — the same "divmod the output coordinates per element →
 * Horner over the input shape from the left" as binary elementwise, a structure different from
 * codegen's stride table (the MUST at the top of this file).
 */
export const referenceExpand = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  if (target.length < x.shape.length) {
    throw new ReferenceOpError(
      `expand は rank を下げられない [${x.shape.join(",")}] → [${target.join(",")}]`,
    );
  }
  const offset = target.length - x.shape.length;
  x.shape.forEach((extent, index) => {
    if (extent !== 1 && extent !== target[offset + index]) {
      throw new ReferenceOpError(
        `expand は長さ 1 でない次元 ${index}（${extent}）を ${target[offset + index]} にできない`,
      );
    }
  });
  const values = new Array<number>(numel(target));
  for (let i = 0; i < values.length; i += 1) {
    values[i] = x.data[broadcastIndex(i, target, x.shape)];
  }
  return materialize(x.dtype, target, values);
};

/**
 * Slicing with a static axis and a static range (ADR 0014). Extracts `[start, end)` along axis
 * `dim`.
 *
 * MUST: do not share the index computation with codegen (the read family of strided.ts). That one
 * is a gather form that reads through "output coordinates → dot product with the input strides +
 * offset", so this one is written in the opposite scatter form — "decompose the flat **input**
 * index into input coordinates → drop it when outside the window → fold the output coordinates by
 * Horner from the left" (the same approach as referenceSymPrefixSlice, so that an error in building
 * the offset does not cancel out on both sides).
 */
export const referenceSlice = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("slice");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const { dim, start } = sliceAttrs(attrs, "reference");
  const rank = x.shape.length;
  const values = new Array<number>(numel(shape));
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    const picked = coords[dim] - start;
    if (picked < 0 || picked >= shape[dim]) continue;
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * shape[d] + (d === dim ? picked : coords[d]);
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * Concatenation along a static axis (ADR 0014). Stacks the inputs in order along axis `dim`.
 *
 * MUST: the oracle side also checks that the output is **filled completely** (an unwritten hole
 * raises ReferenceOpError). The GPU side covers the whole range with partial writes per input, so if
 * the reference "treated unwritten elements as 0", a full-write violation would cancel out on both
 * sides.
 * MUST: the index comes from a scan per input (scatter form), and the write position is built by
 * **shifting the coordinate of the concatenation axis** — codegen folds it into a single offset
 * word, so the expression is not the same.
 */
export const referenceCat = (
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("cat");
  for (const input of inputs) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(
    contract,
    inputs.map((input) => input.shape),
    "reference",
    {
      attrs,
    },
  )[0];
  const dim = catDim(attrs, "reference");
  const rank = shape.length;
  const values = new Array<number | undefined>(numel(shape)).fill(undefined);
  const coords = new Array<number>(rank).fill(0);
  let written = 0;
  for (const input of inputs) {
    for (let flat = 0; flat < input.data.length; flat += 1) {
      let rest = flat;
      for (let d = rank - 1; d >= 0; d -= 1) {
        coords[d] = rest % input.shape[d];
        rest = Math.floor(rest / input.shape[d]);
      }
      let out = 0;
      for (let d = 0; d < rank; d += 1) {
        out = out * shape[d] + (d === dim ? coords[d] + written : coords[d]);
      }
      values[out] = input.data[flat];
    }
    written += input.shape[dim];
  }
  const hole = values.findIndex((value) => value === undefined);
  if (hole >= 0) {
    throw new ReferenceOpError(`cat の出力 [${shape.join(",")}] の要素 ${hole} が書かれていない`);
  }
  return materialize(inputs[0].dtype, shape, values as number[]);
};

/**
 * Constant zero padding of the last dimension (ADR 0014).
 *
 * MUST: **fill with 0, then copy** (scatter form). The GPU side is a gather form that decides "in
 * range or not" per output element and writes the copied value or 0, so copying the same branch
 * would let an off-by-one at the boundary cancel out on both sides. This form also shows in the
 * values that the zero region really is 0.
 */
export const referencePad = (x: RefTensor, attrs: Readonly<Record<string, unknown>>): RefTensor => {
  const contract = resolveOpContract("pad");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const { left } = padAttrs(attrs, "reference");
  const lengthIn = x.shape[x.shape.length - 1];
  const lengthOut = shape[shape.length - 1];
  const values = new Array<number>(numel(shape)).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    const row = Math.floor(flat / lengthIn);
    const col = flat % lengthIn;
    values[row * lengthOut + col + left] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * Index reversal along a static axis (ADR 0014).
 *
 * MUST: do not use the same direction as codegen (src/kernels/flip.ts). That one is a gather form
 * that reads the input at `len - 1 - c` from the output index, so this one is a scatter form that
 * scans the **input** and writes to the reversed position (so that an axis mix-up or an off-by-one
 * does not cancel out on both sides).
 */
export const referenceFlip = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("flip");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = flipDim(attrs, "reference");
  const rank = shape.length;
  const values = new Array<number>(x.data.length);
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    let out = 0;
    for (let d = 0; d < rank; d += 1) {
      out = out * shape[d] + (d === dim ? shape[d] - 1 - coords[d] : coords[d]);
    }
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, shape, values);
};

/**
 * Symbolic prefix slice (ADR 0010) — cuts **the head of each axis** of the constant `x` baked at
 * Tmax to `shape`. The prefix length itself is decided on the runtime side from
 * `coeff·sym+offset`, so the reference implementation only receives the decided lengths.
 *
 * MUST: do not share the index computation with codegen (strided.ts). That one is a gather form
 * that reads through "output coordinates → dot product with the input strides", so this one is
 * written in the opposite scatter form — "decompose the flat **input** index into Tmax-shaped
 * coordinates → drop it when outside the prefix → fold the output coordinates by Horner from the
 * left" (the same approach as referencePermute, so that an error in building the stride table does
 * not cancel out on both sides).
 */
export const referenceSymPrefixSlice = (x: RefTensor, shape: readonly number[]): RefTensor => {
  const target = [...shape];
  const rank = x.shape.length;
  if (target.length !== rank) {
    throw new ReferenceOpError(
      `sym_prefix_slice は rank を変えない [${x.shape.join(",")}] → [${target.join(",")}]`,
    );
  }
  target.forEach((extent, index) => {
    if (extent < 0 || extent > x.shape[index]) {
      throw new ReferenceOpError(
        `sym_prefix_slice の prefix [${target.join(",")}] が定数 [${x.shape.join(
          ",",
        )}] に収まらない`,
      );
    }
  });
  const values = new Array<number>(numel(target));
  const coords = new Array<number>(rank).fill(0);
  for (let flat = 0; flat < x.data.length; flat += 1) {
    let rest = flat;
    for (let d = rank - 1; d >= 0; d -= 1) {
      coords[d] = rest % x.shape[d];
      rest = Math.floor(rest / x.shape[d]);
    }
    if (coords.some((coord, d) => coord >= target[d])) continue;
    let out = 0;
    for (let d = 0; d < rank; d += 1) out = out * target[d] + coords[d];
    values[out] = x.data[flat];
  }
  return materialize(x.dtype, target, values);
};

/**
 * linear `out[…, n] = Σ_k x[…, k] · weight[n, k] + bias[n]` (a fused op — ADR 0012).
 *
 * MUST: build the weight index as `weight[n * K + k]`, **keeping the transposed layout**. The GPU
 * side reads the same layout, but this one "runs the leading dimensions as plain nested loops, not
 * tiles", so the handling of tile boundaries and remainders is not shared (that is what should be
 * verified).
 */
export const referenceLinear = (x: RefTensor, weight: RefTensor, bias: RefTensor): RefTensor => {
  const contract = resolveOpContract("linear");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference")[0];
  const outFeatures = weight.shape[0];
  const inFeatures = weight.shape[1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < outFeatures; col += 1) {
      let acc = 0;
      for (let i = 0; i < inFeatures; i += 1) {
        acc += x.data[row * inFeatures + i] * weight.data[col * inFeatures + i];
      }
      out[row * outFeatures + col] = Math.fround(acc + bias.data[col]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * layer_norm (last dimension, with affine). The variance is **the population variance (divided by
 * N, correction = 0)**, not `torch.var`'s default (N−1) (the matching row of the counter-examples
 * list in docs/op-vocabulary.md).
 *
 * MUST: written in two passes (mean → sum of squared deviations). The one-pass form
 * `E[x²] − E[x]²` can give a negative variance through cancellation, and copying the same shortcut
 * as the GPU side would let the errors cancel and make the comparison vacuous.
 */
export const referenceLayerNorm = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("layer_norm");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { eps } = layerNormAttrs(attrs, "reference");
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let sum = 0;
    for (let i = 0; i < dim; i += 1) sum += x.data[base + i];
    const mean = sum / dim;
    let squares = 0;
    for (let i = 0; i < dim; i += 1) {
      const deviation = x.data[base + i] - mean;
      squares += deviation * deviation;
    }
    // eps is a scalar attr carried as an f32 word, so the reference also rounds it to f32 before
    // use (the same discipline as the other scalars)
    const inv = 1 / Math.sqrt(squares / dim + Math.fround(eps));
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround((x.data[base + i] - mean) * inv * weight.data[i] + bias.data[i]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * rms_norm (last dimension, weight only).
 * `out[r, i] = x[r, i] / sqrt(mean_r(x²) + eps) · weight[i]`
 *
 * MUST: **do not subtract the mean** (do not copy layer_norm). rms_norm uses the mean of the
 * squares itself, so a sum of squared deviations silently drifts on inputs whose mean is not zero.
 * MUST: write the reciprocal square root as `1 / Math.sqrt(…)` (the GPU side uses the single
 * operation `inverseSqrt`). Copying the same built-in would let rounding errors cancel out on both
 * sides and make the comparison vacuous — the same discipline as erf / softmax.
 * MUST: the sum of squares accumulates in f64 and rounds once when stored (reductions are treated
 * like matmul / row reduce).
 */
export const referenceRmsNorm = (
  x: RefTensor,
  weight: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("rms_norm");
  for (const input of [x, weight]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape], "reference", { attrs })[0];
  const eps = rmsNormEps(attrs, "reference");
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let squares = 0;
    for (let i = 0; i < dim; i += 1) squares += x.data[base + i] * x.data[base + i];
    // eps is a scalar attr carried as an f32 word, so the reference also rounds it to f32 before
    // use (the same as layer_norm)
    const inv = 1 / Math.sqrt(squares / dim + Math.fround(eps));
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround(x.data[base + i] * inv * weight.data[i]);
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * softmax / safe_softmax (last dimension, safe-softmax).
 *
 * MUST: written **with amax subtracted**. The naive form breaks on large inputs, as the
 * counter-examples list in docs/op-vocabulary.md says, and if the oracle broke the same way the
 * comparison could not detect that "the kernel's safe form is off".
 *
 * `safe` is ADR 0044's safe_softmax (writes 0 to rows whose row max is −inf). MUST: the operation
 * sequence of rows with finite elements stays **identical** to the plain softmax — writing them
 * separately would let the comparison miss a state where "only the variant degenerated to the naive
 * form".
 */
const softmaxLike = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
  safe: boolean,
): RefTensor => {
  const contract = resolveOpContract(safe ? "safe_softmax" : "softmax");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = x.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, x.data[base + i]);
    // A row with every element −inf (`out` is already 0-initialized, so go on to the next row).
    if (safe && amax === Number.NEGATIVE_INFINITY) continue;
    let total = 0;
    for (let i = 0; i < dim; i += 1) total += Math.exp(x.data[base + i] - amax);
    for (let i = 0; i < dim; i += 1) {
      out[base + i] = Math.fround(Math.exp(x.data[base + i] - amax) / total);
    }
  }
  return { dtype: "f32", shape, data: out };
};

export const referenceSoftmax = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => softmaxLike(x, attrs, false);

/** safe_softmax (ADR 0044) — softmax + "a row whose row max is −inf is all 0". */
export const referenceSafeSoftmax = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => softmaxLike(x, attrs, true);

/**
 * Fused attention `out = softmax_lastdim((q·scale) @ (k·scale)ᵀ + mask) @ v` (ADR 0023).
 *
 * `mask` is an optional fourth input (an additive `[1,1,M,N]` term broadcast over all of B·H —
 * src/ops.ts). It is added **after S is rounded to f32** (the same position as the write-out
 * epilogue of the GPU side's ①QK).
 *
 * MUST: written as **three plain steps** (materialize S → materialize P with safe-softmax → PV).
 * The GPU side does not materialize P and evaluates `exp(S−m)·inv` while filling the A tile (a
 * one-pass fusion), so if the reference were written in the same fused form (online softmax or a
 * non-materialized P), errors in the fused steps would cancel out on both sides and the comparison
 * could not detect them — the same discipline as the "subtract amax" MUST of `referenceSoftmax`.
 * MUST: reductions accumulate in f64 and round to f32 once when stored (the same as
 * `referenceBmm`). Not matching the GPU's sequential f32 accumulation bit for bit is **correct**
 * (which is why allclose decides).
 * MUST: `scale` is applied to **both** q and k (the half-scale contract — src/ops.ts). Compared
 * with an implementation that applies it to one side only, the values differ by a factor of √, and
 * this oracle detects that error.
 */
export const referenceAttention = (
  q: RefTensor,
  k: RefTensor,
  v: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
  mask?: RefTensor,
): RefTensor => {
  const contract = resolveOpContract("attention");
  const inputs = mask === undefined ? [q, k, v] : [q, k, v, mask];
  for (const input of inputs) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(
    contract,
    inputs.map((input) => input.shape),
    "reference",
    { attrs },
  )[0];
  // The GPU side carries it as an f32 word of params, so the oracle also uses the value rounded to
  // f32 (the same as masked_fill).
  const scale = Math.fround(attentionScale(attrs, "reference"));
  const [batches, heads, rows, depth] = shape;
  const cols = k.shape[2];
  // GQA repeat count `r = H / Hkv` (the derived value of ADR 0067 decision 1 — the shape check
  // guarantees exact division).
  const repeat = heads / k.shape[1];
  const out = new Float32Array(numel(shape));
  // The f32 intermediates (S and P) plainly represent "three materialized steps" — one rounding
  // per step.
  const scores = new Float32Array(cols);
  const weights = new Float32Array(cols);
  for (let head = 0; head < batches * heads; head += 1) {
    const qBase = head * rows * depth;
    // MUST: the kv-head is mapped by **integer division** (the same identity as the GPU side's
    // `wid.z / r` — ADR 0067 decision 2. For `head = b·H + h` with `H = Hkv·r`,
    // `⌊head/r⌋ = b·Hkv + ⌊h/r⌋` holds exactly).
    // The remainder (`head % Hkv`) is a different mapping that agrees only when "Hkv = H and
    // B = 1", and it breaks both the head correspondence and the b term (the detectors are the
    // GQA / MQA cases in tests/helpers/gpu_op_cases.ts).
    const kvBase = Math.floor(head / repeat) * cols * depth;
    for (let row = 0; row < rows; row += 1) {
      // ① S[row, col] = Σ_d (q·scale)[row,d] · (k·scale)[col,d] (+ mask[row,col])
      for (let col = 0; col < cols; col += 1) {
        let acc = 0;
        for (let d = 0; d < depth; d += 1) {
          acc +=
            q.data[qBase + row * depth + d] * scale * (k.data[kvBase + col * depth + d] * scale);
        }
        // The mask is added **after** S is rounded to f32 (the same position as the GPU side's
        // write-out epilogue).
        // The mask is [1,1,M,N], so the index has no batch term = every head reads the same row.
        scores[col] =
          mask === undefined
            ? Math.fround(acc)
            : Math.fround(Math.fround(acc) + mask.data[row * cols + col]);
      }
      // ② safe-softmax (MUST: the amax-subtracted form — the same as referenceSoftmax). A row
      // with every column −inf (a fully masked row) fixes P at all 0 — the mirror of the empty-row
      // guard of the GPU side's attention_stats v2 (ADR 0044 addendum 2026-08-31). A row containing
      // NaN does not enter this branch because JS's Math.max propagates NaN, and it flows to the
      // output as NaN (the same classification as the GPU side's nan_max).
      let amax = scores[0];
      for (let col = 1; col < cols; col += 1) amax = Math.max(amax, scores[col]);
      if (amax === -Infinity) {
        weights.fill(0);
      } else {
        let total = 0;
        for (let col = 0; col < cols; col += 1) total += Math.exp(scores[col] - amax);
        for (let col = 0; col < cols; col += 1) {
          weights[col] = Math.fround(Math.exp(scores[col] - amax) / total);
        }
      }
      // ③ O[row, d] = Σ_col P[row,col] · v[col,d]
      for (let d = 0; d < depth; d += 1) {
        let acc = 0;
        for (let col = 0; col < cols; col += 1) {
          acc += weights[col] * v.data[kvBase + col * depth + d];
        }
        out[qBase + row * depth + d] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * embedding `out[…, h] = weight[index[…], h]`.
 *
 * MUST: **an out-of-range index always throws** (the same ruling as gather — the GPU side
 * contaminates with NaN). Relaxing the oracle to match the GPU side would make "a
 * contract-violating graph give the same values on both sides", and the comparison could no longer
 * detect the violation.
 * NOTE: the attrs' `padding_idx` has no effect on forward, so it is not taken (the contract —
 * src/ops.ts).
 */
export const referenceEmbedding = (weight: RefTensor, index: RefTensor): RefTensor => {
  const contract = resolveOpContract("embedding");
  assertSlotDtype(contract, 0, weight.dtype, "reference");
  assertSlotDtype(contract, 1, index.dtype, "reference");
  const shape = computeOutputShape(contract, [weight.shape, index.shape], "reference")[0];
  const vocab = weight.shape[0];
  const hidden = weight.shape[1];
  const out = new Float32Array(numel(shape));
  for (let row = 0; row < index.data.length; row += 1) {
    const pick = index.data[row];
    if (pick < 0 || pick >= vocab) {
      throw new ReferenceOpError(
        `embedding の添字 index[${row}] = ${pick} が語彙数 ${vocab} の範囲外`,
      );
    }
    for (let col = 0; col < hidden; col += 1) {
      out[row * hidden + col] = weight.data[pick * hidden + col];
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * masked_fill `out = mask ? value : x` (the mask is right-aligned broadcast).
 *
 * The index derivation is {@link broadcastIndex} — "divmod the output coordinates per element →
 * Horner over the input shape from the left", a structure different from codegen's stride table
 * (the MUST at the top of this file). The fill value is rounded with `Math.fround` to match the GPU
 * side's f32 word.
 */
export const referenceMaskedFill = (
  x: RefTensor,
  mask: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("masked_fill");
  assertSlotDtype(contract, 0, x.dtype, "reference");
  assertSlotDtype(contract, 1, mask.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, mask.shape], "reference")[0];
  const fill = Math.fround(maskedFillValue(attrs, "reference"));
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i += 1) {
    out[i] = mask.data[broadcastIndex(i, shape, mask.shape)] !== 0 ? fill : x.data[i];
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv1d `out[b, oc, ox] = Σ_{ic,k} x[b, ic, ox·s + k·d − p] · w[oc, ic − g·Cin/groups, k] + b[oc]`
 * (`g` is the group `oc` belongs to — ADR 0015 made groups / dilation attrs).
 *
 * MUST: the padding region is **skipped** (not written as adding 0). Collapsing it into adding 0
 * removes the out-of-range branch itself and would allow an implementation that passes even with a
 * negative index.
 */
export const referenceConv1d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv1d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding, dilation, groups } = conv1dAttrs(attrs, "reference");
  const [batch, channelsOut, lengthOut] = shape;
  const channelsIn = x.shape[1];
  const lengthIn = x.shape[2];
  const kernel = weight.shape[2];
  // The contract check (computeOutputShape) has already ensured exact division.
  const inPerGroup = channelsIn / groups;
  const outPerGroup = channelsOut / groups;
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      // The weight's second axis is Cin/groups — the input channels run over the group's band only
      const icBase = Math.floor(oc / outPerGroup) * inPerGroup;
      for (let ox = 0; ox < lengthOut; ox += 1) {
        let acc = bias.data[oc];
        for (let icRel = 0; icRel < inPerGroup; icRel += 1) {
          for (let k = 0; k < kernel; k += 1) {
            const ix = ox * stride + k * dilation - padding;
            if (ix < 0 || ix >= lengthIn) continue;
            acc +=
              x.data[(item * channelsIn + icBase + icRel) * lengthIn + ix] *
              weight.data[(oc * inPerGroup + icRel) * kernel + k];
          }
        }
        out[(item * channelsOut + oc) * lengthOut + ox] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv2d `out[b, oc, oy, ox] = Σ_{ic,kh,kw}
 *   x[b, ic, oy·sh + kh·dh − ph, ox·sw + kw·dw − pw] · w[oc, ic − g·Cin/groups, kh, kw] + b[oc]`
 * (`g` is the group `oc` belongs to — ADR 0017).
 *
 * MUST: the weight is `[Cout, Cin/groups, Kh, Kw]`, and **the order Kh, Kw** is part of the
 * contract too. With a square kernel (Kh == Kw) swapping them gives the same values, so tests pin
 * Kh ≠ Kw.
 * MUST: the padding region is **skipped** (not written as adding 0). Collapsing it into adding 0
 * removes the out-of-range branch itself and would allow an implementation that passes even with a
 * negative index.
 * MUST: build the H / W indices independently (do not flatten to `(y·W + x)` first). Collapsing
 * them into one lets "a stride with H and W swapped" agree on square inputs.
 */
export const referenceConv2d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv2d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding, dilation, groups } = conv2dAttrs(attrs, "reference");
  const [batch, channelsOut, heightOut, widthOut] = shape;
  const channelsIn = x.shape[1];
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const kernelH = weight.shape[2];
  const kernelW = weight.shape[3];
  // The contract check (computeOutputShape) has already ensured exact division.
  const inPerGroup = channelsIn / groups;
  const outPerGroup = channelsOut / groups;
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      // The weight's second axis is Cin/groups — the input channels run over the group's band only
      const icBase = Math.floor(oc / outPerGroup) * inPerGroup;
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let acc = bias.data[oc];
          for (let icRel = 0; icRel < inPerGroup; icRel += 1) {
            const plane = (item * channelsIn + icBase + icRel) * heightIn;
            const weightPlane = (oc * inPerGroup + icRel) * kernelH;
            for (let kh = 0; kh < kernelH; kh += 1) {
              const iy = oy * stride[0] + kh * dilation[0] - padding[0];
              if (iy < 0 || iy >= heightIn) continue;
              for (let kw = 0; kw < kernelW; kw += 1) {
                const ix = ox * stride[1] + kw * dilation[1] - padding[1];
                if (ix < 0 || ix >= widthIn) continue;
                acc +=
                  x.data[(plane + iy) * widthIn + ix] *
                  weight.data[(weightPlane + kh) * kernelW + kw];
              }
            }
          }
          out[((item * channelsOut + oc) * heightOut + oy) * widthOut + ox] = Math.fround(acc);
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv3d `out[oc, ot, oy, ox] = Σ_{ic,kt,kh,kw}
 *   x[ic, ot·st + kt·dt − pt, oy·sh + kh·dh − ph, ox·sw + kw·dw − pw] · w[oc, ic − g·Cin/groups, kt, kh, kw]
 *   + b[oc]` (**unbatched** — `g` is the group `oc` belongs to; ADR 0118 decision 1).
 *
 * It carries the full semantics (general groups, general stride / dilation / padding). It is wider
 * than the subset the GPU implements (implicit GEMM with groups == 1) because this oracle is the
 * only path that can check assets with groups > 1 on the CPU (ADR 0064 axis A — semantics and
 * subset are held separately).
 *
 * MUST: the weight is `[Cout, Cin/groups, Kt, Kh, Kw]`, and **the order Kt, Kh, Kw** is part of the
 * contract too. With a cubic kernel swapping them gives the same values, so tests pin
 * Kt ≠ Kh ≠ Kw.
 * MUST: the padding region is **skipped** (not written as adding 0 — the same reason as conv1d /
 * conv2d).
 * MUST: build the T / H / W indices independently (flattening first lets a stride with swapped
 * axes agree on cubic inputs).
 */
export const referenceConv3d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv3d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding, dilation, groups } = conv3dAttrs(attrs, "reference");
  const [channelsOut, timeOut, heightOut, widthOut] = shape;
  const [channelsIn, timeIn, heightIn, widthIn] = x.shape;
  const [, , kernelT, kernelH, kernelW] = weight.shape;
  // The contract check (computeOutputShape) has already ensured exact division.
  const inPerGroup = channelsIn / groups;
  const outPerGroup = channelsOut / groups;
  const out = new Float32Array(numel(shape));
  for (let oc = 0; oc < channelsOut; oc += 1) {
    // The weight's second axis is Cin/groups — the input channels run over the group's band only
    const icBase = Math.floor(oc / outPerGroup) * inPerGroup;
    for (let ot = 0; ot < timeOut; ot += 1) {
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let acc = bias.data[oc];
          for (let icRel = 0; icRel < inPerGroup; icRel += 1) {
            const volume = (icBase + icRel) * timeIn;
            const weightVolume = (oc * inPerGroup + icRel) * kernelT;
            for (let kt = 0; kt < kernelT; kt += 1) {
              const it = ot * stride[0] + kt * dilation[0] - padding[0];
              if (it < 0 || it >= timeIn) continue;
              const plane = (volume + it) * heightIn;
              const weightPlane = (weightVolume + kt) * kernelH;
              for (let kh = 0; kh < kernelH; kh += 1) {
                const iy = oy * stride[1] + kh * dilation[1] - padding[1];
                if (iy < 0 || iy >= heightIn) continue;
                for (let kw = 0; kw < kernelW; kw += 1) {
                  const ix = ox * stride[2] + kw * dilation[2] - padding[2];
                  if (ix < 0 || ix >= widthIn) continue;
                  acc +=
                    x.data[(plane + iy) * widthIn + ix] *
                    weight.data[(weightPlane + kh) * kernelW + kw];
                }
              }
            }
          }
          out[((oc * timeOut + ot) * heightOut + oy) * widthOut + ox] = Math.fround(acc);
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * conv_transpose1d `out[b, oc, ox] = Σ_{ic,k} x[b, ic, (ox + p − k)/s] · w[ic, oc, k] + b[oc]`
 * (only the `(ic, k)` for which `(ox + p − k)` is divisible by `s` and the quotient lies in
 * `[0, L)` — ADR 0015).
 *
 * MUST: the weight is `[Cin, Cout, K]` (transposed relative to conv1d).
 * MUST: a `k` that does not divide is **skipped** (there is no input there = zero contribution).
 * Collapsing it into adding 0 would let an implementation pass that has lost the divisibility test
 * itself.
 */
export const referenceConvTranspose1d = (
  x: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("conv_transpose1d");
  for (const input of [x, weight, bias]) assertDtype(contract, input.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape, weight.shape, bias.shape], "reference", {
    attrs,
  })[0];
  const { stride, padding } = convTranspose1dAttrs(attrs, "reference");
  const [batch, channelsOut, lengthOut] = shape;
  const channelsIn = x.shape[1];
  const lengthIn = x.shape[2];
  const kernel = weight.shape[2];
  const out = new Float32Array(numel(shape));
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      for (let ox = 0; ox < lengthOut; ox += 1) {
        let acc = bias.data[oc];
        for (let ic = 0; ic < channelsIn; ic += 1) {
          for (let k = 0; k < kernel; k += 1) {
            const shifted = ox + padding - k;
            if (shifted < 0 || shifted % stride !== 0) continue;
            const ix = shifted / stride;
            if (ix >= lengthIn) continue;
            acc +=
              x.data[(item * channelsIn + ic) * lengthIn + ix] *
              weight.data[(ic * channelsOut + oc) * kernel + k];
          }
        }
        out[(item * channelsOut + oc) * lengthOut + ox] = Math.fround(acc);
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * CPU reference of DCNv2 (`torchvision::deform_conv2d`) — ADR 0055.
 *
 * `out[b, oc, oy, ox] = Σ_{ic,kh,kw} (m[b, kh·Kw+kw, oy, ox] · bilinear(x[b,ic], y, x)) ·
 *   w[oc, ic, kh, kw] + bias[oc]`, `y = (oy − ph) + kh + off_y` / `x = (ox − pw) + kw + off_x`
 * (stride / dilation / groups / offset_groups have no field = fixed at 1).
 *
 * MUST: the reduction is in ascending `(ic, kh, kw)`, with **the same nesting** as the conv2d
 * reference. The main oracle of this op is that the degenerate case with offset all 0 and mask all
 * 1 matches conv2d, so reordering the loops (even with the same semantics) breaks that check
 * through rounding differences.
 * MUST: do not hoist the reads of offset / mask out of the `ic` loop (same reason).
 * MUST: the source coordinates are rounded to f32 with `Math.fround`. **Whether a tap is taken**
 * (the `(−1, in)` range test and the per-corner zero fill) can change with the rounding width, so
 * this part alone does not proceed in f64.
 * MUST: write the range test in **the positive form** (`> −1 && < in`). NaN falls to the false
 * side, so NaN alone is checked first and **propagated to the output** (dropping it to 0 would be a
 * silent wrong value — in the line of ADR 0020).
 * MUST: outside the border is **zero fill, not border clamp** (if the center is out of range the
 * whole tap is 0; even inside, an out-of-range corner alone is 0). torchvision's two-level scheme
 * is copied verbatim.
 */
export const referenceDeformConv2d = (
  x: RefTensor,
  weight: RefTensor,
  offset: RefTensor,
  mask: RefTensor,
  bias: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("deform_conv2d");
  for (const input of [x, weight, offset, mask, bias]) {
    assertDtype(contract, input.dtype, "reference");
  }
  const shape = computeOutputShape(
    contract,
    [x.shape, weight.shape, offset.shape, mask.shape, bias.shape],
    "reference",
    { attrs },
  )[0];
  const { padding } = deformConv2dAttrs(attrs, "reference");
  const [batch, channelsOut, heightOut, widthOut] = shape;
  const channelsIn = x.shape[1];
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const kernelH = weight.shape[2];
  const kernelW = weight.shape[3];
  const taps = kernelH * kernelW;
  const out = new Float32Array(numel(shape));
  /** Bilinear sample of the `x[item, channel]` plane (0 out of range; NaN propagates). */
  const sample = (plane: number, sourceY: number, sourceX: number): number => {
    if (Number.isNaN(sourceY) || Number.isNaN(sourceX)) return Number.NaN;
    if (!(sourceY > -1 && sourceY < heightIn && sourceX > -1 && sourceX < widthIn)) {
      return 0;
    }
    const y0 = Math.floor(sourceY);
    const x0 = Math.floor(sourceX);
    const ly1 = Math.fround(sourceY - y0);
    const lx1 = Math.fround(sourceX - x0);
    const ly0 = 1 - ly1;
    const lx0 = 1 - lx1;
    // The four corners are range-checked **individually** (not clamped). y0 is in [−1, H−1] as a
    // consequence of the range test.
    const at = (y: number, columnX: number): number =>
      y >= 0 && y < heightIn && columnX >= 0 && columnX < widthIn
        ? x.data[plane + y * widthIn + columnX]
        : 0;
    return (
      ly0 * lx0 * at(y0, x0) +
      ly0 * lx1 * at(y0, x0 + 1) +
      ly1 * lx0 * at(y0 + 1, x0) +
      ly1 * lx1 * at(y0 + 1, x0 + 1)
    );
  };
  for (let item = 0; item < batch; item += 1) {
    for (let oc = 0; oc < channelsOut; oc += 1) {
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let acc = bias.data[oc];
          for (let ic = 0; ic < channelsIn; ic += 1) {
            const plane = (item * channelsIn + ic) * heightIn * widthIn;
            for (let kh = 0; kh < kernelH; kh += 1) {
              for (let kw = 0; kw < kernelW; kw += 1) {
                const tap = kh * kernelW + kw;
                const pixel = oy * widthOut + ox;
                // The channel order of offset is **even = y / odd = x** (torchvision verbatim).
                const offsetBase = (item * 2 * taps + 2 * tap) * heightOut * widthOut;
                const sourceY = Math.fround(oy - padding[0] + kh + offset.data[offsetBase + pixel]);
                const sourceX = Math.fround(
                  ox - padding[1] + kw + offset.data[offsetBase + heightOut * widthOut + pixel],
                );
                // The modulator applies **after the interpolation** (`(m · v) · w` — the same
                // grouping as torchvision).
                const modulated =
                  mask.data[(item * taps + tap) * heightOut * widthOut + pixel] *
                  sample(plane, sourceY, sourceX);
                acc +=
                  modulated * weight.data[((oc * channelsIn + ic) * kernelH + kh) * kernelW + kw];
              }
            }
          }
          out[((item * channelsOut + oc) * heightOut + oy) * widthOut + ox] = Math.fround(acc);
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * Bilinear tap table for one axis (`align_corners = True`).
 * A literal translation of torch's `area_pixel_compute_scale` + `compute_source_index_and_lambda`.
 *
 * MUST: the scale and the source coordinates are rounded to f32 with `Math.fround`. **The choice
 * of index** (the result of `trunc` and the last-element special case) can change with the rounding
 * width, so this part alone does not proceed in f64 — it would be an off-by-one that no tolerance
 * absorbs.
 */
const bilinearTaps = (
  lengthIn: number,
  lengthOut: number,
): readonly { readonly low: number; readonly high: number; readonly lambda: number }[] => {
  const scale = lengthOut > 1 ? Math.fround((lengthIn - 1) / (lengthOut - 1)) : 0;
  return Array.from({ length: lengthOut }, (_unused, index) => {
    const source = Math.fround(scale * index);
    // The source coordinate is non-negative, so floor = trunc (the same result as torch's
    // static_cast<int64_t>). The upper bound in − 1 is the same as torch's guard_index_and_lambda
    // — once an axis exceeds roughly 2^23, f32 rounding lets the source coordinate reach in (an
    // actual case with in 8650802, out 34), and without the bound it reads the next plane.
    const low = Math.min(Math.floor(source), lengthIn - 1);
    return {
      low,
      // The last element has no second tap (`index1 = index0`).
      high: low + 1 < lengthIn ? low + 1 : low,
      lambda: Math.fround(source - low),
    };
  });
};

/**
 * Bilinear resample over the two spatial axes of NCHW (`align_corners = True` only — Core ATen layer).
 *
 * MUST: do not share the way indices are folded with codegen
 * (src/kernels/upsample-bilinear2d.ts). That one folds N·C into a single plane index and recovers
 * the coordinates from the flat index, so this one **loops over the coordinates directly in four
 * nested loops** (so that an axis mix-up or a plane offset does not cancel out on both sides).
 * Building the tap tables per axis first is also a deliberately different form; that one computes
 * them inline per element.
 */
export const referenceUpsampleBilinear2d = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("upsample_bilinear2d");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  upsampleBilinear2dAttrs(attrs, "reference");
  const [batch, channels, heightOut, widthOut] = shape;
  const heightIn = x.shape[2];
  const widthIn = x.shape[3];
  const rows = bilinearTaps(heightIn, heightOut);
  const columns = bilinearTaps(widthIn, widthOut);
  const out = new Float32Array(numel(shape));
  let flat = 0;
  for (let item = 0; item < batch; item += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const plane = (item * channels + channel) * heightIn * widthIn;
      for (const row of rows) {
        for (const column of columns) {
          const at = (y: number, index: number): number => x.data[plane + y * widthIn + index];
          // H outside, W inside (the same nesting as torch's expression tree).
          const top =
            (1 - column.lambda) * at(row.low, column.low) +
            column.lambda * at(row.low, column.high);
          const bottom =
            (1 - column.lambda) * at(row.high, column.low) +
            column.lambda * at(row.high, column.high);
          out[flat] = Math.fround((1 - row.lambda) * top + row.lambda * bottom);
          flat += 1;
        }
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * The hidden-side GRU scan (`gru_scan` / `gru_scan_reverse` — extended composite layer, ADR 0056).
 *
 * `gh = W_hh·h + b_hh` / `r = σ(gh_r + gi_r)` / `z = σ(gh_z + gi_z)` /
 * `n = tanh(gi_n + gh_n·r)` / `h' = (h − n)·z + n`.
 *
 * MUST: the update formula is `(h − n)·z + n` **verbatim** (the mathematically equivalent
 * `(1 − z)·n + z·h` is a different rounding sequence in f32 — measured: 44,345 of 100,000 elements
 * disagree). The order of the gate additions and of the `n` product also follows torch's
 * decomposed form (ADR 0056 decision 3).
 * MUST: drop to f32 with `Math.fround` at every step. The state is carried over T steps, so running
 * in f64 would open the gap to the GPU side in proportion to T and make the comparison's tolerance
 * shape-dependent.
 * MUST: the reverse direction reverses **only the scan order**; the output is written at the
 * forward time index (the same contract as the GPU kernel).
 * NOTE: the sequential reduction is in ascending `k` like the kernel, but this one is a plain dot
 * product that does not split rows into GEMM tiles, and runs the three gates in **separate loops**
 * (the way indices are folded is not shared — so that an axis mix-up does not cancel out on both
 * sides).
 */
export const referenceGruScan = (
  name: GruScanOpName,
  gi: RefTensor,
  initial: RefTensor,
  weight: RefTensor,
  bias: RefTensor,
): RefTensor => {
  const contract = resolveOpContract(name);
  for (const input of [gi, initial, weight, bias]) {
    assertDtype(contract, input.dtype, "reference");
  }
  const shape = computeOutputShape(
    contract,
    [gi.shape, initial.shape, weight.shape, bias.shape],
    "reference",
  )[0];
  const [time, batch, hidden] = shape;
  const gates = 3 * hidden;
  const out = new Float32Array(numel(shape));
  const f32 = Math.fround;
  /**
   * The hidden side of gate `g` (0 = r / 1 = z / 2 = n):
   * `Σ_k W_hh[g·H + j, k]·h[k] + b_hh[g·H + j]`.
   */
  const hiddenGate = (state: Float32Array<ArrayBuffer>, gate: number, unit: number): number => {
    const row = (gate * hidden + unit) * hidden;
    let acc = 0;
    for (let k = 0; k < hidden; k += 1) acc += weight.data[row + k] * state[k];
    // bias last (the GEMM epilogue form — not the conv family's bias-first)
    return f32(acc + bias.data[gate * hidden + unit]);
  };
  for (let item = 0; item < batch; item += 1) {
    const state = new Float32Array(hidden);
    for (let unit = 0; unit < hidden; unit += 1) state[unit] = initial.data[item * hidden + unit];
    for (let step = 0; step < time; step += 1) {
      const t = name === "gru_scan_reverse" ? time - 1 - step : step;
      const inputBase = (t * batch + item) * gates;
      const next = new Float32Array(hidden);
      for (let unit = 0; unit < hidden; unit += 1) {
        // In the gate additions **the hidden side is the first argument** (torch's decomposed
        // form verbatim)
        const reset = f32(
          UNARY_F32.sigmoid(
            f32(hiddenGate(state, 0, unit) + gi.data[inputBase + unit]),
            NO_SCALARS,
          ),
        );
        const update = f32(
          UNARY_F32.sigmoid(
            f32(hiddenGate(state, 1, unit) + gi.data[inputBase + hidden + unit]),
            NO_SCALARS,
          ),
        );
        // n = tanh(i_n + h_n·r) — the product is `h_n · r`; in the sum the input side is the first
        // argument
        const gated = f32(hiddenGate(state, 2, unit) * reset);
        const candidate = f32(
          UNARY_F32.tanh(f32(gi.data[inputBase + 2 * hidden + unit] + gated), NO_SCALARS),
        );
        // h' = (h − n)·z + n
        const decayed = f32(f32(state[unit] - candidate) * update);
        next[unit] = f32(decayed + candidate);
      }
      for (let unit = 0; unit < hidden; unit += 1) {
        state[unit] = next[unit];
        out[(t * batch + item) * hidden + unit] = next[unit];
      }
    }
  }
  return { dtype: "f32", shape, data: out };
};

/**
 * The arm of a single-output op (each arm states that it has one output — ADR 0068 decision 1).
 */
const sole = (tensor: RefTensor): readonly RefTensor[] => [tensor];

/**
 * The unified entry that branches on the contract table's kind (so that tests can walk a graph as
 * is). It returns **the sequence in ascending output slot order**, and only `topk` (ADR 0068
 * decision 3) returns two.
 *
 * `outShape` is required for ops where "the declared output shape is the target shape" (reshape /
 * expand — ADR 0011). It cannot be derived from the inputs, so when it is omitted the call fails
 * instead of guessing silently.
 */
export const applyReferenceOpOutputs = (
  op: string,
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>> = {},
  outShape?: readonly number[],
): readonly RefTensor[] => {
  const contract: OpContract = resolveOpContract(op);
  if (!arityFits(contract, inputs.length)) {
    throw new ReferenceOpError(
      `op '${op}' の入力数が ${inputs.length}（契約は ${describeArity(contract)}）`,
    );
  }
  // MUST: the attrs key set is checked with the same single check as the node contract check
  // (states is always empty — the reference has no states form). Without it, the reference alone
  // would silently ignore attrs the contract rejects (`window` of an attention without a states
  // field, etc.) and turn a contract violation into values.
  assertAttrKeys(contract, attrs, false, "reference");
  // MUST: attrs values, input dtypes and shapes (rank limit, axis range, consistency with the target
  // shape) are also checked with the same table as the node contract check. Without it, the
  // reference alone would return values for combinations the contract rejects (an i32 input to
  // permute, a rank-0 input to expand, etc.), and contract violations in test data would go
  // undetected. Only the acceptance conditions are shared; the reference keeps its own index and
  // numeric computations (so that it does not carry the same errors as the GPU side).
  assertAttrValues(contract, attrs, "reference");
  assertInputDtypes(
    contract,
    inputs.map((input) => input.dtype),
    (index) => `reference の入力 ${index}`,
    "reference",
  );
  const declared = (): readonly number[] => {
    if (outShape === undefined) {
      throw new ReferenceOpError(`op '${op}' は出力 shape が目標形（outShape が要る）`);
    }
    return outShape;
  };
  // The shape contract of sym_prefix_slice needs the bindings (`coeff·sym+offset`), and the
  // reference only receives the resolved output shape, so it is not run through it (attrs and dtype
  // were checked above). state_append fails below as "has no CPU reference".
  if (contract.kind !== "symPrefixSlice" && contract.kind !== "stateAppend") {
    computeOutputShape(
      contract,
      inputs.map((input) => input.shape),
      "reference",
      {
        attrs,
        ...(contract.kind === "reshape" || contract.kind === "expand"
          ? { declared: declared() }
          : {}),
      },
    );
  }
  switch (contract.kind) {
    case "unary":
      return sole(referenceUnary(contract.name, inputs[0], attrs));
    case "binary":
      return sole(referenceBinary(contract.name, inputs[0], inputs[1]));
    case "where":
      return sole(referenceWhere(inputs[0], inputs[1], inputs[2]));
    case "cumsum":
      return sole(referenceCumsum(inputs[0], attrs));
    case "matmul":
      return sole(referenceMatmul(inputs[0], inputs[1]));
    case "bmm":
      return sole(referenceBmm(inputs[0], inputs[1]));
    case "gather":
      return sole(referenceGather(inputs[0], inputs[1]));
    case "rowReduce":
      return sole(referenceRowReduce(contract.name, inputs[0], reduceDim(attrs, "reference")));
    case "argmax":
      return sole(referenceArgmax(inputs[0]));
    // The only multi-output arm (values + indices — ADR 0068 decision 3). Not going through
    // `sole` here is itself the declaration that "there are two outputs".
    case "topk":
      return referenceTopk(inputs[0], attrs);
    case "cast":
      return sole(referenceCast(inputs[0], castTargetDtype(attrs, "reference")));
    case "reshape":
      return sole(referenceReshape(inputs[0], declared()));
    case "permute":
      return sole(referencePermute(inputs[0], permuteDims(attrs, "reference")));
    case "expand":
      return sole(referenceExpand(inputs[0], declared()));
    case "slice":
      return sole(referenceSlice(inputs[0], attrs));
    case "cat":
      return sole(referenceCat(inputs, attrs));
    case "pad":
      return sole(referencePad(inputs[0], attrs));
    case "flip":
      return sole(referenceFlip(inputs[0], attrs));
    case "symPrefixSlice":
      // The prefix length follows from the bindings (`coeff·sym+offset`) — it cannot be derived
      // from the inputs, so as for reshape / expand the caller passes the resolved output shape.
      return sole(referenceSymPrefixSlice(inputs[0], declared()));
    case "linear":
      return sole(referenceLinear(inputs[0], inputs[1], inputs[2]));
    case "layerNorm":
      return sole(referenceLayerNorm(inputs[0], inputs[1], inputs[2], attrs));
    case "rmsNorm":
      return sole(referenceRmsNorm(inputs[0], inputs[1], attrs));
    case "staticQuantize":
      return sole(referenceStaticQuantize(inputs[0], attrs));
    case "softmax":
      return sole(referenceSoftmax(inputs[0], attrs));
    case "safeSoftmax":
      return sole(referenceSafeSoftmax(inputs[0], attrs));
    case "attention":
      return sole(referenceAttention(inputs[0], inputs[1], inputs[2], attrs, inputs[3]));
    // Ops that touch state have no CPU reference (they are effects on context-owned buffers, not
    // values, and RefTensor inputs and outputs alone cannot express their semantics — the
    // execution layer (wave D) holds the implementation).
    // MUST: do not let them pass. Silently returning an empty output sequence would allow a parity
    // test that counts it as "the reference and the GPU agreed".
    case "stateAppend":
      throw new ReferenceOpError(
        `op '${op}' は CPU 参照を持たない（state への effect は GenerationContext の担当）`,
      );
    case "embedding":
      return sole(referenceEmbedding(inputs[0], inputs[1]));
    case "maskedFill":
      return sole(referenceMaskedFill(inputs[0], inputs[1], attrs));
    case "conv1d":
      return sole(referenceConv1d(inputs[0], inputs[1], inputs[2], attrs));
    case "conv2d":
      return sole(referenceConv2d(inputs[0], inputs[1], inputs[2], attrs));
    case "conv3d":
      return sole(referenceConv3d(inputs[0], inputs[1], inputs[2], attrs));
    case "convTranspose1d":
      return sole(referenceConvTranspose1d(inputs[0], inputs[1], inputs[2], attrs));
    case "deformConv2d":
      return sole(
        referenceDeformConv2d(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], attrs),
      );
    case "upsampleBilinear2d":
      return sole(referenceUpsampleBilinear2d(inputs[0], attrs));
    case "gruScan":
      return sole(referenceGruScan(contract.name, inputs[0], inputs[1], inputs[2], inputs[3]));
  }
};

/**
 * The unified entry for single-output ops (slot 0 of {@link applyReferenceOpOutputs}).
 *
 * MUST: do not silently return only slot 0 of a multi-output op. Calling `topk` through
 * `applyReferenceOp` would allow a test where "only the values match and nobody compares the
 * indices", so a different count fails and points to the plural entry.
 */
export const applyReferenceOp = (
  op: string,
  inputs: readonly RefTensor[],
  attrs: Readonly<Record<string, unknown>> = {},
  outShape?: readonly number[],
): RefTensor => {
  const outputs = applyReferenceOpOutputs(op, inputs, attrs, outShape);
  if (outputs.length !== 1) {
    throw new ReferenceOpError(
      `op '${op}' は出力 ${outputs.length} 本（多出力は applyReferenceOpOutputs で受ける）`,
    );
  }
  return outputs[0];
};

/**
 * An independent reference computed from f32 division and round-half-to-even, without the GPU's
 * boundary table (ADR 0097).
 */
export const referenceStaticQuantize = (
  x: RefTensor,
  attrs: Readonly<Record<string, unknown>>,
): RefTensor => {
  const contract = resolveOpContract("static_quantize");
  assertDtype(contract, x.dtype, "reference");
  const shape = computeOutputShape(contract, [x.shape], "reference", { attrs })[0];
  const scale = staticQuantizeScale(attrs, "reference");
  const input = new Uint32Array(x.data.buffer, x.data.byteOffset, x.data.length);
  const output = new Float32Array(input.length);
  const bits = new Uint32Array(output.buffer);
  for (let i = 0; i < input.length; i++) {
    const value = input[i],
      sign = value & 0x80000000,
      magnitude = value & 0x7fffffff;
    if (scale === 0) {
      bits[i] = value;
      continue;
    }
    // Detect NaN before handing it to floating-point arithmetic, and make it quiet while keeping
    // its sign and payload.
    if (magnitude > 0x7f800000) {
      bits[i] = value | 0x00400000;
      continue;
    }
    const q = Math.fround(Math.abs(x.data[i]) / scale);
    const lower = Math.floor(q),
      fraction = q - lower;
    const rounded = fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0) ? lower + 1 : lower;
    output[i] = Math.fround(Math.min(rounded, sign === 0 ? 127 : 128) * scale);
    bits[i] |= sign;
  }
  return { dtype: "f32", shape, data: output };
};
