/**
 * The fixed kernel of topk (last dimension, **static k**, two outputs = f32 values in descending
 * order + i32 indices) (ADR 0068 decision 3 — the first user of node-level multi-output).
 *
 * The input is a contiguous `[rows, dim]` layout with the leading dimensions flattened; the output
 * is k words per row (both declared shapes are `[…, k]`). `k` is **a plan-time constant**
 * (attrs declaration required), so it is baked into the WGSL and not carried in params (one fact
 * is not held in two places).
 *
 * The form is **one row = one workgroup** in two phases, with a grid-stride over rows (real models
 * exceed the one-dimensional workgroup limit):
 *
 * 1. **Lane-local top-k** — lane `l` scans only the elements with `i ≡ l (mod W)` and updates its
 *    block (K words of `cand_*`) by insertion that keeps it **in descending order**. A candidate
 *    that cannot beat the tail (the weakest) is dropped there.
 * 2. **merge** — k rounds of a **W-way tournament**. Each round folds the maximum of "every lane's
 *    current head" over a tree; that is the maximum of the remaining set (each block is
 *    descending). Only the owner of the winning element advances its cursor by one.
 *
 * MUST: **no full-vocabulary argsort** (ADR 0068 decision 3). The candidate scan reads the row
 * once; the work is a scan of `dim/W` plus a fold of `k·log2(W)`. Flattening it into k row reduces
 * (repeated masked argmax) reads the row k times and lands on the expensive side the ADR avoids.
 * Only when writing out the selected values are the k elements of the original input re-read, to
 * keep the value bits.
 *
 * MUST: in this regular form **the scratch stays in workgroup storage** (no co-location in an
 * output buffer and no temporary buffer — structurally avoiding ADR 0068 decision 3's "the
 * allocation spec becomes unreadable"). In exchange, the **implementation limit** of k follows from
 * the device's workgroup storage limit ({@link topkMaxK}), so exceeding it does not fall back:
 * {@link assertTopkK} fails loudly with the limit. Long rows with k=1 use the same split form as
 * argmax (ADR 0068 addendum 9).
 *
 * MUST: the tie-break is **the smallest index** (ADR 0068 decision 3). The predicate
 * {@link F32_RANK_WGSL} is shared with argmax: a strict comparison in the lexicographic order of
 * (value descending, index ascending) — (1) the winner does not change whichever child is the
 * first argument, (2) the winner is the maximum of the candidate set. So the lane-local insertion,
 * the tree fold and the order across rounds all close over one predicate, and the result comes out
 * "values descending, ties by ascending index" regardless of the combination order. k=1 gives the
 * same answer as argmax (a check compares the two ops for disagreement).
 *
 * NOTE: torch's `topk` **does not guarantee the order of equal elements** (measured 2026-08-17 /
 * torch 2.13.0+cpu): `topk([5,5,5,5], 1)` returns index **2** while `argmax([5,5,5,5])` returns
 * **0** — torch disagrees with itself in the same repository (3 of 7 rows mismatch at k=1). Over
 * 700 columns of `[…9…9…9…]` it returns `[650,300,12]` = **descending indices**. So the
 * parenthetical of ADR 0068 decision 3, "the order of equal elements also matches torch", **does
 * not hold in measurement** — what matches is **the sequence of values** (descending with the same
 * multiplicities, so bit-identical to the (value descending, index ascending) implementation;
 * measured on a 200×4 case full of duplicates), and for the index sequence karume chose to
 * **define** the smallest index (the same family as argmax, deterministic).
 *
 * MUST: the initial row max is **−inf**, and the block identity is `(−inf, sentinel index = dim)`
 * (no finite sentinel — the same reason as argmax). A real element always beats the identity (even
 * with value −inf, `< dim` wins the index comparison), so even an all −inf row has the answer "k
 * entries from the smallest index". The contract layer (the shape rule) guarantees `1 ≤ k ≤ dim`,
 * so the union of the lane-local lists always holds at least k real elements
 * (`Σ min(K, count_l) ≥ min(K, Σ count_l) = K`), and the identity never leaks into the output.
 *
 * MUST: NaN is **treated as the maximum** (the same discipline as argmax; torch also puts NaN
 * first — measured as above). The comparison is **on the bit pattern** ({@link F32_RANK_WGSL}) —
 * driver comparisons are all false with NaN, so a plain `>` makes NaN lose silently.
 *
 * MUST: params are passed as a uniform (there is a workgroupBarrier inside the row loop, and the
 * loop condition must be uniform within the workgroup). The −inf bit pattern travels in the
 * existing params layout. Comparison and storage both stay integer, so subnormals are not flushed
 * to zero on the way (ADR 0068 addendum 10).
 */

import { F32_RANK_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertRowLoopBound, assertU32Params } from "../codegen/params.ts";

/**
 * Number of lanes that fold one row.
 *
 * MUST: do not increase it. Workgroup storage is `8·W·(k+1)` bytes (K (value, index) pairs per
 * lane + one tournament word), so doubling W halves the implementation limit of k
 * ({@link topkMaxK}). W=32 is the largest power of two that admits **k ≤ 63** within WebGPU's
 * default 16384 bytes — a choice that keeps the practical range of top-k sampling (k ≤ 50 or so)
 * on machines with the default limits.
 */
export const TOPK_WORKGROUP_SIZE = 32;

/** f32 bit pattern of −inf (the third params word — WGSL has no infinity literal). */
export const TOPK_NEG_INF_BITS = 0xff800000;

/** Workgroup storage for one candidate (an f32 value + a u32 index). */
const TOPK_CANDIDATE_BYTES = 8;

/**
 * Bytes of workgroup storage needed to admit `k` = `8·W·(k+1)`.
 * The breakdown is the per-lane candidate block of `k` words (`cand_value` / `cand_index`) and the
 * tournament head of one word (`head_value` / `head_index`).
 */
export const topkWorkgroupStorageBytes = (k: number): number =>
  TOPK_CANDIDATE_BYTES * TOPK_WORKGROUP_SIZE * (k + 1);

/**
 * **The implementation limit of k** given by the device's workgroup storage limit (ADR 0068
 * decision 3). The largest k with `8·W·(k+1) ≤ limit`; **63** at the WebGPU default (16384 bytes).
 */
export const topkMaxK = (storageLimitBytes: number): number =>
  Math.floor(storageLimitBytes / (TOPK_CANDIDATE_BYTES * TOPK_WORKGROUP_SIZE)) - 1;

/**
 * The limit on k guaranteed by the WebGPU core default (`maxComputeWorkgroupStorageSize` = 16384).
 * A k above it is a portability step: "it runs on this adapter but fails on a machine with the
 * defaults" (neither falls back silently — {@link assertTopkK} fails with the limit).
 */
export const TOPK_CORE_LIMIT_MAX_K = 63;

/**
 * Checks that k is within the implementation limit (ADR 0068 decision 3 — **no fallback**).
 *
 * MUST: the diagnostic carries **the limit itself**. "k is too large" alone does not tell the user
 * how far to lower k, and leaves them guessing a device-dependent value ({@link topkMaxK}).
 * MUST: `1 ≤ k ≤ last dimension` is not this function's job (an accepted range that does not
 * depend on the device, so it belongs to the contract layer — `TOPK_ATTRS` in src/ops/attrs.ts and
 * the shape rule in src/ops/shapes.ts).
 */
export const assertTopkK = (k: number, storageLimitBytes: number, where: string): void => {
  assertU32Params(`${where} topk`, { k });
  if (k < 1) {
    throw new CodegenError(`${where}: topk k must be a positive integer (${k})`);
  }
  const max = topkMaxK(storageLimitBytes);
  if (k > max) {
    throw new CodegenError(
      `${where}: topk k=${k} exceeds the implementation limit ${max} (workgroup storage needs ${topkWorkgroupStorageBytes(
        k,
      )} bytes but the device limit is ${storageLimitBytes} bytes, ${TOPK_WORKGROUP_SIZE} lanes per row)`,
    );
  }
};

/**
 * Pipeline key. **It includes k** (the WGSL changes with k; without it, the pipeline of the first
 * k is handed to dispatches of another k, and different bytes are written without an exception).
 */
export const topkKey = (k: number): string =>
  `topk:v2:f32+i32:lastdim:desc:minindex:k${k}:wg${TOPK_WORKGROUP_SIZE}`;

/** WGSL with k baked in (k determines the array lengths and the number of rounds). */
export const topkWgsl = (k: number): string => {
  const w = TOPK_WORKGROUP_SIZE;
  return `// karume topk (last dim, f32 values + i32 indices, descending, min-index tie-break, -inf identity, k=${k})
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> indices: array<i32>;

${F32_RANK_WGSL}

var<workgroup> cand_value: array<u32, ${k * w}>;
var<workgroup> cand_index: array<u32, ${k * w}>;
var<workgroup> head_value: array<u32, ${w}>;
var<workgroup> head_index: array<u32, ${w}>;

@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = f32_rank_key(params.neg_inf);
  let block = lid * ${k}u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    // phase 1: lane-local top-k. The identity is the weakest in both value and index (sentinel
    // dim = no candidate), so a real element always wins — even an all -inf row gives k entries
    // from the smallest index
    for (var s = 0u; s < ${k}u; s = s + 1u) {
      cand_value[block + s] = neg_inf;
      cand_index[block + s] = dim;
    }
    var i = lid;
    while (i < dim) {
      let v = f32_rank_key(x[base + i]);
      // a candidate that cannot beat the tail (the weakest) is dropped here. Only a winner goes on
      // to the insertion that keeps the descending order
      if (rank_key_beats(v, i, cand_value[block + ${k - 1}u], cand_index[block + ${k - 1}u])) {
        var s = ${k - 1}u;
        while (s > 0u && rank_key_beats(v, i, cand_value[block + s - 1u], cand_index[block + s - 1u])) {
          cand_value[block + s] = cand_value[block + s - 1u];
          cand_index[block + s] = cand_index[block + s - 1u];
          s = s - 1u;
        }
        cand_value[block + s] = v;
        cand_index[block + s] = i;
      }
      i = i + ${w}u;
    }
    // phase 2: merge. The maximum of the lanes' heads = the maximum of the remaining set (each
    // block is descending), so k rounds give values descending, ties by ascending index. The
    // cursor advances at most k-1 times, so block + cursor always stays inside the block
    var cursor = 0u;
    for (var r = 0u; r < ${k}u; r = r + 1u) {
      head_value[lid] = cand_value[block + cursor];
      head_index[lid] = cand_index[block + cursor];
      workgroupBarrier();
      var stride = ${w / 2}u;
      while (stride > 0u) {
        if (lid < stride) {
          let other = head_value[lid + stride];
          let other_at = head_index[lid + stride];
          if (rank_key_beats(other, other_at, head_value[lid], head_index[lid])) {
            head_value[lid] = other;
            head_index[lid] = other_at;
          }
        }
        workgroupBarrier();
        stride = stride / 2u;
      }
      let won = head_index[0u];
      if (lid == 0u) {
        values[row * ${k}u + r] = x[base + won];
        indices[row * ${k}u + r] = i32(won);
      }
      // the scan splits work by i ≡ lid (mod ${w}), so the winning element's owner is won % ${w}
      if (won % ${w}u == lid) {
        cursor = cursor + 1u;
      }
      // make every read of won finish before the next round overwrites head
      workgroupBarrier();
    }
    row = row + nwg.x;
  }
}
`;
};

/**
 * The uniform Params (`{rows, dim, neg_inf}`). 16 bytes are allocated even for three words of
 * content (allocating in 16-byte units is a project convention (the fusion pass fixes the same
 * form in ADR 0040), not a requirement of the specification).
 *
 * MUST: `k` is not carried (it is baked into the WGSL — holding one fact in two places lets a
 * dispatch with only one of them updated silently write a different count).
 * MUST: `dim` is a positive integer (a row with `dim = 0` has no "k largest"). The contract side
 * also requires `1 ≤ k ≤ last dimension`, so this is a double check.
 */
export const topkParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("topk params", { rows, dim });
  // MUST: the in-row loop steps by the workgroup size.
  assertRowLoopBound("topk params", { dim }, TOPK_WORKGROUP_SIZE);
  if (dim < 1) {
    throw new CodegenError(`topk params: dim must be a positive integer (${dim})`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = TOPK_NEG_INF_BITS;
  return params;
};
