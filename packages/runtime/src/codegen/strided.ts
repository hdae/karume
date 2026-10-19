/**
 * WGSL generation for the strided materializing-copy kernels (ADR 0011).
 *
 * The input is read through `(offset, strides[STRIDED_RANK])` and **the output is always
 * contiguous**. permute is expressed as permuted strides, expand as stride 0 on the replicated
 * axes — a form that confines the materialization of layout ops to one kernel family and keeps the
 * elementwise / reduce / matmul codegen on its contiguous-input assumption (option B of ADR 0011).
 *
 * MUST: generation is deterministic — the same key gives byte-identical WGSL (key and WGSL are built
 * from the same canonicalization). **The element type is part of the key** (ADR 0009) — an expand
 * of a bool mask and an expand of i32 indices occur in practice, and without it in the key the same
 * pipeline is assigned to a dispatch of another element type and the bit pattern is reinterpreted
 * without an exception.
 * MUST: grid-stride loops are assumed. Real models exceed the per-dimension dispatch limit (65535 by
 * default in the spec).
 *
 * The rank is **fixed at STRIDED_RANK** so that one dtype = one pipeline. The caller pads the
 * missing rank on the left with 1 (dims) and 0 (strides) — the same padding as right-aligned
 * broadcast, so the coordinates of the extra leading axes are always 0 and do not contribute to the
 * read position.
 *
 * This file has **two families, read and write** (ADR 0014). The write family (cat) is the dual of
 * the read family — it reads the input contiguously and writes the output through
 * `(offset, strides)`. Both share the same params layout and the same deterministic key convention;
 * only the direction of traversal is reversed.
 */

import { type IrDtype, SEMANTIC_DTYPES } from "../format/ir.ts";
import { CodegenError } from "./errors.ts";
import { assertU32Params } from "./params.ts";

/**
 * Value ranks are capped at 4 (ADR 0011: its first premise, every DeBERTa-front value having
 * rank ≤ 4, is gone; ADR 0016 keeps 4 for Anima and ADR 0118 admits rank 5 only for conv3d
 * weights). Rank ≥ 5 values are lowered first by the rank-lowering normalization.
 */
export const STRIDED_RANK = 4;
export const STRIDED_WORKGROUP_SIZE = 256;

export type StridedSpec = {
  /** Semantic dtype of the elements (input and output alike — layout ops do not touch values). */
  readonly dtype: IrDtype;
};

/** Semantic dtype → WGSL scalar type. A bool is stored as u32 0 / 1 (ADR 0009). */
const WGSL_SCALAR: Readonly<Record<IrDtype, string>> = {
  f32: "f32",
  i32: "i32",
  bool: "u32",
};

const canonicalize = (spec: StridedSpec): IrDtype => {
  const found = SEMANTIC_DTYPES.find((dtype) => dtype === spec.dtype);
  if (found === undefined) {
    throw new CodegenError(`strided codegen: element type '${spec.dtype}' is not a semantic dtype`);
  }
  return found;
};

/**
 * MUST: every generation parameter baked into the WGSL goes into the key (rank, workgroup size,
 * element type). Changing a constant without it in the key reuses a pipeline compiled with the old
 * constant under the same key.
 */
export const stridedKey = (spec: StridedSpec): string =>
  `strided:v1:${canonicalize(spec)}:r${STRIDED_RANK}:wg${STRIDED_WORKGROUP_SIZE}`;

/**
 * The params layout (storage, a u32 array):
 * `[0]=element count n, [1..STRIDED_RANK]=output dims, then as many input strides, offset last`.
 * It is passed as storage rather than uniform because this family has no workgroupBarrier and is
 * not constrained by the uniformity analysis (the same reason as elementwise).
 */
export const stridedWgsl = (spec: StridedSpec): string => {
  const scalar = WGSL_SCALAR[canonicalize(spec)];
  const dimAt = (d: number): number => 1 + d;
  const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
  const offsetAt = 1 + 2 * STRIDED_RANK;

  // Decompose the flat output index with divmod from the last axis (the output is always
  // contiguous).
  const decode: string[] = ["    var rem = i;"];
  for (let d = STRIDED_RANK - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`);
  }
  decode.push("    let c0 = rem;");

  const terms = Array.from({ length: STRIDED_RANK }, (_, d) => `c${d} * params[${strideAt(d)}u]`);

  return [
    `// karume strided copy (rank ${STRIDED_RANK}, ${scalar}, generated)`,
    "@group(0) @binding(0) var<storage, read> params: array<u32>;",
    `@group(0) @binding(1) var<storage, read> src: array<${scalar}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${scalar}>;`,
    "",
    `@compute @workgroup_size(${STRIDED_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    `  let stride = nwg.x * ${STRIDED_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    `    let src_index = params[${offsetAt}u] + ${terms.join(" + ")};`,
    "    out[i] = src[src_index];",
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

const assertRank = (shape: readonly number[], what: string): void => {
  if (shape.length < 1 || shape.length > STRIDED_RANK) {
    throw new CodegenError(
      `strided params: ${what} rank ${shape.length} is outside 1..${STRIDED_RANK}`,
    );
  }
};

/** Strides of a contiguous layout (running product from the right). */
const contiguousStrides = (shape: readonly number[]): number[] => {
  const strides = new Array<number>(shape.length).fill(0);
  let running = 1;
  for (let d = shape.length - 1; d >= 0; d -= 1) {
    strides[d] = running;
    running *= shape[d];
  }
  return strides;
};

/**
 * Read strides of permute. `dims[d]` = the input dimension that output dimension d takes, so one
 * step along output dimension d is one step along input axis `dims[d]`.
 * Assumes the contract table has already checked the permutation (bijective, matching rank).
 */
export const permuteSrcStrides = (
  srcShape: readonly number[],
  dims: readonly number[],
): number[] => {
  assertRank(srcShape, "permute input");
  if (dims.length !== srcShape.length) {
    throw new CodegenError(
      `strided params: permute dims [${dims.join(",")}] do not match input rank ${srcShape.length}`,
    );
  }
  const contiguous = contiguousStrides(srcShape);
  return dims.map((dim) => {
    const stride = contiguous[dim];
    if (stride === undefined) {
      throw new CodegenError(`strided params: permute dims contain out-of-range axis ${dim}`);
    }
    return stride;
  });
};

/** The subject named in {@link expandSrcStrides}'s diagnostics (expand itself by default). */
type BroadcastLabels = {
  readonly src: string;
  readonly out: string;
};

const EXPAND_LABELS: BroadcastLabels = { src: "expand input", out: "expand output" };

/**
 * Read strides of right-aligned broadcast. A replicated dimension (input length 1) gets
 * **stride 0**, and so do the leading axes added by a larger rank. Assumes the contract side (only
 * dimensions of length 1 are expanded) has already been checked.
 *
 * MUST: the caller replaces the subject of the diagnostics. masked_fill's mask read shares this
 * rule too (a rule written in two places always diverges), and a message such as "the rank of the
 * expand input …" there would mislead the reader into suspecting an expand when the graph has
 * none.
 */
export const expandSrcStrides = (
  srcShape: readonly number[],
  outShape: readonly number[],
  labels: BroadcastLabels = EXPAND_LABELS,
): number[] => {
  assertRank(srcShape, labels.src);
  assertRank(outShape, labels.out);
  if (outShape.length < srcShape.length) {
    throw new CodegenError(
      `strided params: ${labels.out} rank ${outShape.length} is less than ${labels.src} rank ${srcShape.length}`,
    );
  }
  const contiguous = contiguousStrides(srcShape);
  const offset = outShape.length - srcShape.length;
  return outShape.map((_, d) =>
    d < offset || srcShape[d - offset] === 1 ? 0 : contiguous[d - offset],
  );
};

/**
 * Read strides of slice (ADR 0014). The read uses **the input's contiguous strides as they are**,
 * and the cut is expressed by the offset alone (what ADR 0011 anticipated: "the only variable point
 * is the one offset word in params").
 */
export const sliceSrcStrides = (srcShape: readonly number[]): number[] => {
  assertRank(srcShape, "slice input");
  return contiguousStrides(srcShape);
};

/**
 * Read start position of slice = `start × the input stride of that axis`.
 *
 * MUST: the strides are built from the **input** shape (the same table as {@link sliceSrcStrides}).
 * Built from the shrunken output shape, the steps of the axes before the sliced one shrink and the
 * second and later rows read other rows (the same trap as the MUST of sym_prefix_slice).
 */
export const sliceSrcOffset = (srcShape: readonly number[], dim: number, start: number): number => {
  assertRank(srcShape, "slice input");
  if (!Number.isSafeInteger(dim) || dim < 0 || dim >= srcShape.length) {
    throw new CodegenError(
      `strided params: slice dim ${dim} is outside input rank ${srcShape.length}`,
    );
  }
  if (!Number.isSafeInteger(start) || start < 0 || start > srcShape[dim]) {
    throw new CodegenError(
      `strided params: slice start ${start} is outside axis ${dim} (length ${srcShape[dim]})`,
    );
  }
  return start * contiguousStrides(srcShape)[dim];
};

/**
 * Read strides of sym_prefix_slice (ADR 0010). The input is **the full length of a constant baked
 * at Tmax**, and the output is a contiguous buffer cut from its head. The offset is 0 and the read
 * uses the input's contiguous strides as they are.
 *
 * MUST: the strides are built from **the input (Tmax-shaped) shape**. Built from the bound output
 * shape, the row step shrinks and the second and later rows read other rows of the constant table
 * (correct only when T = Tmax, silently broken otherwise).
 */
export const prefixSliceSrcStrides = (srcShape: readonly number[]): number[] => {
  assertRank(srcShape, "sym_prefix_slice input");
  return contiguousStrides(srcShape);
};

/**
 * Builds the contents of the params buffer. The missing rank is padded **on the left** with
 * dims=1 / strides=0 (the same padding as right-aligned broadcast — the coordinates of the extra
 * leading axes are always 0 and do not contribute to the read position).
 *
 * The read and write families use **the same layout** (the dims of the traversed side, and the
 * strides + offset of the other side) — only the direction of traversal is reversed, and the
 * meaning of params is exactly dual.
 */
const packStridedParams = (
  dims: readonly number[],
  strides: readonly number[],
  offset: number,
  labels: { readonly shape: string; readonly rank: string },
): Uint32Array<ArrayBuffer> => {
  assertRank(dims, labels.shape);
  if (strides.length !== dims.length) {
    throw new CodegenError(
      `strided params: stride count ${strides.length} does not match ${labels.rank} rank ${dims.length}`,
    );
  }
  const n = dims.reduce((count, dim) => count * dim, 1);
  assertU32Params("strided params", {
    ...Object.fromEntries(dims.map((dim, d) => [`dims[${d}]`, dim])),
    ...Object.fromEntries(strides.map((stride, d) => [`strides[${d}]`, stride])),
    n,
    offset,
  });
  const pad = STRIDED_RANK - dims.length;
  const params = new Uint32Array(2 + 2 * STRIDED_RANK);
  params[0] = n;
  for (let d = 0; d < STRIDED_RANK; d += 1) {
    params[1 + d] = d < pad ? 1 : dims[d - pad];
    params[1 + STRIDED_RANK + d] = d < pad ? 0 : strides[d - pad];
  }
  params[1 + 2 * STRIDED_RANK] = offset;
  return params;
};

export const stridedParams = (
  outShape: readonly number[],
  srcStrides: readonly number[],
  offset: number,
): Uint32Array<ArrayBuffer> =>
  packStridedParams(outShape, srcStrides, offset, { shape: "strided output", rank: "output" });

// ---- strided write-copy family (the dual of the read family — ADR 0014) ----------------------

export const STRIDED_WRITE_WORKGROUP_SIZE = 256;

/**
 * MUST: **the same deterministic key convention** as the read family (element type, rank and
 * workgroup size go in). Changing only the family name and keeping the rest also states that both
 * have the same variable points and no others.
 */
export const stridedWriteKey = (spec: StridedSpec): string =>
  `strided_write:v1:${canonicalize(spec)}:r${STRIDED_RANK}:wg${STRIDED_WRITE_WORKGROUP_SIZE}`;

/**
 * WGSL of the strided **write** copy (how cat runs — ADR 0014). It reads the input contiguously and
 * writes the output through `(offset, strides[STRIDED_RANK])` (the dual of {@link stridedWgsl}).
 *
 * The params layout is the same as the read family's:
 * `[0]=element count n, [1..STRIDED_RANK]=**input** dims, then as many **output** strides, offset
 * last`.
 *
 * MUST: one dispatch writes only a subregion of the output. **Covering the whole output per node**
 * is guaranteed by the caller (the executor's cat), which matches the sum of the offsets against
 * the output's axis length (the full-write invariant — ADR 0004 / 0014).
 */
export const stridedWriteWgsl = (spec: StridedSpec): string => {
  const scalar = WGSL_SCALAR[canonicalize(spec)];
  const dimAt = (d: number): number => 1 + d;
  const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
  const offsetAt = 1 + 2 * STRIDED_RANK;

  // Decompose the flat input index with divmod from the last axis (the input is always
  // contiguous).
  const decode: string[] = ["    var rem = i;"];
  for (let d = STRIDED_RANK - 1; d >= 1; d -= 1) {
    decode.push(`    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`);
  }
  decode.push("    let c0 = rem;");

  const terms = Array.from({ length: STRIDED_RANK }, (_, d) => `c${d} * params[${strideAt(d)}u]`);

  return [
    `// karume strided write (rank ${STRIDED_RANK}, ${scalar}, generated)`,
    "@group(0) @binding(0) var<storage, read> params: array<u32>;",
    `@group(0) @binding(1) var<storage, read> src: array<${scalar}>;`,
    `@group(0) @binding(2) var<storage, read_write> out: array<${scalar}>;`,
    "",
    `@compute @workgroup_size(${STRIDED_WRITE_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(global_invocation_id) gid: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let n = params[0u];",
    `  let stride = nwg.x * ${STRIDED_WRITE_WORKGROUP_SIZE}u;`,
    "  var i = gid.x;",
    "  while (i < n) {",
    ...decode,
    `    let out_index = params[${offsetAt}u] + ${terms.join(" + ")};`,
    "    out[out_index] = src[i];",
    "    i = i + stride;",
    "  }",
    "}",
    "",
  ].join("\n");
};

/**
 * params of the write family (the traversed side is the **input**, so dims are the input shape and
 * strides are the output side's).
 */
export const stridedWriteParams = (
  srcShape: readonly number[],
  outStrides: readonly number[],
  offset: number,
): Uint32Array<ArrayBuffer> =>
  packStridedParams(srcShape, outStrides, offset, {
    shape: "strided write input",
    rank: "input",
  });

/**
 * Write strides of cat (the output's contiguous strides) and the write start position of input
 * `k`.
 *
 * offset = "the sum of the lengths of the preceding inputs along the concatenation axis × the
 * output stride of that axis". Getting it wrong writes part of the output twice and leaves the rest
 * **unwritten** (the previous value when the buffer is reused).
 */
export const catOutStrides = (outShape: readonly number[]): number[] => {
  assertRank(outShape, "cat output");
  return contiguousStrides(outShape);
};

export const catOutOffset = (outShape: readonly number[], dim: number, written: number): number => {
  assertRank(outShape, "cat output");
  if (!Number.isSafeInteger(dim) || dim < 0 || dim >= outShape.length) {
    throw new CodegenError(
      `strided params: cat dim ${dim} is outside output rank ${outShape.length}`,
    );
  }
  if (!Number.isSafeInteger(written) || written < 0 || written > outShape[dim]) {
    throw new CodegenError(
      `strided params: cat write position ${written} is outside axis ${dim} (length ${outShape[dim]})`,
    );
  }
  return written * contiguousStrides(outShape)[dim];
};
