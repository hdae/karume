/**
 * The fixed kernel of argmax (last dimension, **rank-preserving**, i32 output) (ADR 0068
 * decision 2).
 *
 * The input is a contiguous `[rows, dim]` layout with the leading dimensions flattened; the output
 * is one word per row (the declared shape is `[…, 1]` = a fixed form with the last dimension
 * collapsed to 1; there is no `keepdim` field).
 *
 * The form is the same as the row reduce (src/codegen/reduce.ts) — one row = one workgroup (256) +
 * a 256-wide bit-reversed binary tree + **a grid-stride over rows too** (real models exceed the
 * one-dimensional workgroup limit of 65535). The only new part is that it **carries (value, index)
 * pairs**.
 *
 * MUST: do not piggyback on the reduce family (a separate family, a separate key). The identity of
 * the existing `amax` is `-F32_MAX` (a finite sentinel), which conflicts with the **−inf start**
 * this kernel requires. Merging them by bumping the reduce side's key version would move the
 * existing snapshots and the WAV / PNG checks wholesale, so keeping them apart is correct.
 *
 * MUST: **the tie-break is the smallest index** (as torch — measured 2026-08-17: a row with several
 * equal values returns the first). llama.cpp disagrees with itself in one repository, GPU side =
 * largest index / CPU sampler = smallest index (research §2); without writing it down, greedy
 * reproducibility splits over implementation differences. The mechanism that preserves it is
 * {@link F32_RANK_WGSL}: "the larger value wins; on a tie the smaller index wins" is **the maximum
 * in the lexicographic order of (value descending, index ascending)**, so the result is the same
 * however the tree combines (whichever child is the first argument).
 *
 * MUST: the initial row max is **−inf** (no finite sentinel). The initial index is the sentinel
 * `dim` (= "no candidate yet"), which makes "a row of all −inf → smallest index = 0"
 * deterministic: even with equal values (−inf vs −inf), a real element (`< dim`) always beats the
 * sentinel in the index comparison. The shape layer guarantees dim ≥ 1 (a last dimension of length
 * 0 is rejected by the contract), so lane 0 always scans at least one element and the final index
 * always falls in `[0, dim)`.
 *
 * MUST: NaN is **treated as the maximum** (as torch — measured 2026-08-17: `argmax` returns the
 * index of the NaN, the smallest one if there are several). The same discipline as NaN propagation
 * in `amax` / `amin` (reduce.ts); the comparison is **on the bit pattern** ({@link F32_RANK_WGSL})
 * — driver comparisons are all false with NaN, so a plain `>` makes NaN lose silently, and the
 * family disagrees with itself: "amax is NaN, argmax is another element".
 *
 * MUST: params are passed as a uniform (there is a workgroupBarrier inside the row loop, and the
 * loop condition must be uniform within the workgroup). The −inf bit pattern travels in the
 * existing params layout. Comparison and storage both stay integer, so subnormals are not flushed
 * to zero on the way (ADR 0068 addendum 10).
 */

import { F32_RANK_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ARGMAX_WORKGROUP_SIZE = 256;

export const ARGMAX_KEY = `argmax:v2:f32>i32:lastdim:minindex:wg${ARGMAX_WORKGROUP_SIZE}`;

/** f32 bit pattern of −inf (the third params word). */
export const ARGMAX_NEG_INF_BITS = 0xff800000;

export const ARGMAX_WGSL: string = [
  "// karume argmax (last dim, f32>i32, min-index tie-break, -inf identity)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "  neg_inf: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<u32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<i32>;",
  "",
  F32_RANK_WGSL,
  "",
  `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  `var<workgroup> scratch_index: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  "",
  `@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(workgroup_id) wid: vec3<u32>,",
  "  @builtin(local_invocation_id) lid3: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let lid = lid3.x;",
  "  let dim = params.dim;",
  "  let neg_inf = f32_rank_key(params.neg_inf);",
  "  var row = wid.x;",
  "  while (row < params.rows) {",
  "    let base = row * dim;",
  "    // the identity is the weakest in value and index (sentinel dim = no candidate) — even in",
  "    // an all -inf row a real element wins on index, so the result is always the smallest index",
  "    var best = neg_inf;",
  "    var best_at = dim;",
  "    var i = lid;",
  "    while (i < dim) {",
  "      let v = f32_rank_key(x[base + i]);",
  "      if (rank_key_beats(v, i, best, best_at)) {",
  "        best = v;",
  "        best_at = i;",
  "      }",
  `      i = i + ${ARGMAX_WORKGROUP_SIZE}u;`,
  "    }",
  "    scratch_value[lid] = best;",
  "    scratch_index[lid] = best_at;",
  "    workgroupBarrier();",
  `    var stride = ${ARGMAX_WORKGROUP_SIZE / 2}u;`,
  "    while (stride > 0u) {",
  "      if (lid < stride) {",
  "        let other = scratch_value[lid + stride];",
  "        let other_at = scratch_index[lid + stride];",
  "        if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
  "          scratch_value[lid] = other;",
  "          scratch_index[lid] = other_at;",
  "        }",
  "      }",
  "      workgroupBarrier();",
  "      stride = stride / 2u;",
  "    }",
  "    if (lid == 0u) {",
  "      out[row] = i32(scratch_index[0u]);",
  "    }",
  "    // make every read of scratch[0] finish before the next row overwrites scratch",
  "    workgroupBarrier();",
  "    row = row + nwg.x;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * The uniform Params (`{rows, dim, neg_inf}`). 16 bytes are allocated even for three words of
 * content (allocating in 16-byte units is a project convention (the fusion pass fixes the same
 * form in ADR 0040), not a requirement of the specification).
 *
 * MUST: `dim` is a positive integer (`dim = 0` is "a row with no maximum", and the sentinel index
 * would leak into the output as is). The contract side also rejects a last dimension of length 0,
 * so this is a double check.
 */
export const argmaxParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("argmax params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`argmax params: dim は正整数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = ARGMAX_NEG_INF_BITS;
  return params;
};

// ---- two-phase split (long rows — MTP stage 4-B ③) ---------------------------------------
//
// With one row = one workgroup, once a row reaches the vocabulary length (262,144 for gemma4's
// drafter) the latency of 256 threads each folding 1,024 elements sequentially becomes the
// bottleneck (measured 2026-09-09, RTX 3080 Ti: 0.5–0.6 ms per row, 1.5–2.1 ms/cycle over the
// drafter's three steps). Long rows are split into **two dispatches**:
//
// - **partial**: the row is cut into spans of {@link ARGMAX_SPLIT_SPAN} elements; one workgroup
//   per span emits the span's maximum `(rank key, index)` into a temporary buffer (u32
//   `[rows, groups, 2]` — the rank key ranks NaN first on ties).
// - **merge**: one workgroup per row folds the `groups` partial results and writes the index.
//
// MUST: the result is **bit-identical** to the one-dispatch form (the maximum in the lexicographic
// order (value descending, index ascending) does not depend on the combination order, so taking
// the maximum per span and then folding equals the maximum over the whole row. The partial
// results use the same identity −inf / sentinel `dim`, and no span is empty
// (`groups = ceil(dim / span)`), so each partial index falls in `[0, dim)`, and even for an all
// −inf row the merge returns the smallest index = 0).
// MUST: the split threshold and the span width are **pure functions of the shape**
// ({@link argmaxSplitGroups}) — the same shape always takes the same path with the same key (no
// run-time autotuning, ADR 0022).

/** Elements handled by one partial workgroup (256 threads × 16 elements). */
const ARGMAX_SPLIT_ELEMENTS = 16;
export const ARGMAX_SPLIT_SPAN = ARGMAX_WORKGROUP_SIZE * ARGMAX_SPLIT_ELEMENTS;
/**
 * The smallest row length that is split into two phases (= four spans or more). Shorter rows stay
 * in the one-dispatch form (key and WGSL unchanged — the existing snapshots / bit-identity checks
 * keep working as they are).
 */
export const ARGMAX_SPLIT_MIN_DIM = ARGMAX_SPLIT_SPAN * 4;

export const ARGMAX_SPLIT_PARTIAL_KEY = `argmax:v2:f32>i32:lastdim:minindex:split-partial:wg${ARGMAX_WORKGROUP_SIZE}e${ARGMAX_SPLIT_ELEMENTS}`;
export const ARGMAX_SPLIT_MERGE_KEY = `argmax:v2:f32>i32:lastdim:minindex:split-merge:wg${ARGMAX_WORKGROUP_SIZE}`;

/** Number of spans when a row of length `dim` is split into two phases (0 when not split). */
export const argmaxSplitGroups = (dim: number): number =>
  dim >= ARGMAX_SPLIT_MIN_DIM ? Math.ceil(dim / ARGMAX_SPLIT_SPAN) : 0;

/**
 * The split of topk(k=1) is limited to WebGPU's default dispatch limit.
 * Huge rows stay on the existing grid-stride single dispatch, so the accepted inputs do not shrink.
 * https://gpuweb.github.io/gpuweb/#dom-gpusupportedlimits-maxcomputeworkgroupsperdimension
 */
export const topkOneSplitGroups = (dim: number): number => {
  const groups = argmaxSplitGroups(dim);
  return groups <= 65_535 ? groups : 0;
};

/** Size of the partial-result temporary buffer (`[rows, groups]` × (u32 value + u32 index)). */
export const argmaxSplitPartialBytes = (rows: number, groups: number): number => rows * groups * 8;

/** Tree reduction (scratch's 256 pairs to one) — the same skeleton as the one-dispatch form. */
const ARGMAX_TREE_WGSL = [
  `    var stride = ${ARGMAX_WORKGROUP_SIZE / 2}u;`,
  "    while (stride > 0u) {",
  "      if (lid < stride) {",
  "        let other = scratch_value[lid + stride];",
  "        let other_at = scratch_index[lid + stride];",
  "        if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
  "          scratch_value[lid] = other;",
  "          scratch_index[lid] = other_at;",
  "        }",
  "      }",
  "      workgroupBarrier();",
  "      stride = stride / 2u;",
  "    }",
].join("\n");

export const ARGMAX_SPLIT_PARTIAL_WGSL: string = [
  "// karume argmax split partial (last dim, f32>i32, min-index tie-break, -inf identity)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "  groups: u32,",
  "  neg_inf: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<u32>;",
  "@group(0) @binding(2) var<storage, read_write> partial: array<u32>;",
  "",
  F32_RANK_WGSL,
  "",
  `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  `var<workgroup> scratch_index: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  "",
  `@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(workgroup_id) wid: vec3<u32>,",
  "  @builtin(local_invocation_id) lid3: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let lid = lid3.x;",
  "  let dim = params.dim;",
  "  let neg_inf = f32_rank_key(params.neg_inf);",
  "  // x axis = span (1 workgroup = 1 span; exceeding the limit fails loudly), y axis = row (grid-stride)",
  "  let group = wid.x;",
  `  let start = group * ${ARGMAX_SPLIT_SPAN}u;`,
  `  let end = min(start + ${ARGMAX_SPLIT_SPAN}u, dim);`,
  "  var row = wid.y;",
  "  while (row < params.rows) {",
  "    let base = row * dim;",
  "    var best = neg_inf;",
  "    var best_at = dim;",
  "    var i = start + lid;",
  "    while (i < end) {",
  "      let v = f32_rank_key(x[base + i]);",
  "      if (rank_key_beats(v, i, best, best_at)) {",
  "        best = v;",
  "        best_at = i;",
  "      }",
  `      i = i + ${ARGMAX_WORKGROUP_SIZE}u;`,
  "    }",
  "    scratch_value[lid] = best;",
  "    scratch_index[lid] = best_at;",
  "    workgroupBarrier();",
  ARGMAX_TREE_WGSL,
  "    if (lid == 0u) {",
  "      let at = (row * params.groups + group) * 2u;",
  "      partial[at] = scratch_value[0u];",
  "      partial[at + 1u] = scratch_index[0u];",
  "    }",
  "    workgroupBarrier();",
  "    row = row + nwg.y;",
  "  }",
  "}",
  "",
].join("\n");

const splitMergeWgsl = (valueOutput: boolean): string =>
  [
    valueOutput
      ? "// karume topk k=1 split merge (raw input bits and minimum index)"
      : "// karume argmax split merge (partials [rows, groups, 2] -> i32 index per row)",
    "struct Params {",
    "  rows: u32,",
    "  dim: u32,",
    "  groups: u32,",
    "  neg_inf: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    "@group(0) @binding(1) var<storage, read> partial: array<u32>;",
    "@group(0) @binding(2) var<storage, read_write> out: array<i32>;",
    ...(valueOutput
      ? [
          "@group(0) @binding(3) var<storage, read_write> value: array<u32>;",
          "@group(0) @binding(4) var<storage, read> input: array<u32>;",
        ]
      : []),
    "",
    F32_RANK_WGSL,
    "",
    `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
    `var<workgroup> scratch_index: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
    "",
    `@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(workgroup_id) wid: vec3<u32>,",
    "  @builtin(local_invocation_id) lid3: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let lid = lid3.x;",
    "  let groups = params.groups;",
    "  let neg_inf = f32_rank_key(params.neg_inf);",
    "  var row = wid.x;",
    "  while (row < params.rows) {",
    "    let base = row * groups * 2u;",
    "    var best = neg_inf;",
    "    var best_at = params.dim;",
    "    var g = lid;",
    "    while (g < groups) {",
    "      let v = partial[base + g * 2u];",
    "      let v_at = partial[base + g * 2u + 1u];",
    "      if (rank_key_beats(v, v_at, best, best_at)) {",
    "        best = v;",
    "        best_at = v_at;",
    "      }",
    `      g = g + ${ARGMAX_WORKGROUP_SIZE}u;`,
    "    }",
    "    scratch_value[lid] = best;",
    "    scratch_index[lid] = best_at;",
    "    workgroupBarrier();",
    ARGMAX_TREE_WGSL,
    "    if (lid == 0u) {",
    "      out[row] = i32(scratch_index[0u]);",
    ...(valueOutput ? ["      value[row] = input[row * params.dim + scratch_index[0u]];"] : []),
    "    }",
    "    workgroupBarrier();",
    "    row = row + nwg.x;",
    "  }",
    "}",
    "",
  ].join("\n");

export const ARGMAX_SPLIT_MERGE_WGSL: string = splitMergeWgsl(false);
export const TOPK_ONE_SPLIT_MERGE_KEY = `topk:v2:f32+i32:lastdim:desc:minindex:k1:split-merge:wg${ARGMAX_WORKGROUP_SIZE}`;
export const topkOneSplitMergeWgsl = (): string => splitMergeWgsl(true);

/**
 * The uniform of the two-phase form (`{rows, dim, groups, neg_inf}` — the same four words for
 * partial / merge).
 *
 * MUST: `groups` is the value of {@link argmaxSplitGroups} (1 or more) — calling with 0 is a shape
 * that should take the one-dispatch form, so it fails loudly.
 */
export const argmaxSplitParams = (
  rows: number,
  dim: number,
  groups: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("argmax split params", { rows, dim, groups });
  if (dim < 1 || groups < 1 || groups !== argmaxSplitGroups(dim)) {
    throw new CodegenError(
      `argmax split params: dim ${dim} / groups ${groups} が分割の形に合わない` +
        `（${argmaxSplitGroups(dim)} 区間が正）`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = groups;
  params[3] = ARGMAX_NEG_INF_BITS;
  return params;
};
