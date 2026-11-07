/**
 * The **single formula** for the node temporaries of states-form attention (ADR 0067 decision 4).
 *
 * The execution plan (`buildStateAttention` in recipe-builders/attention.ts) and the estimate
 * (`stateAttentionTemps` in estimate.ts) must produce the same numbers, so the three formulas they
 * share — the two branches of the column capacity `colCap`, the row-block split
 * ({@link planRowBlocks}) and the byte formulas of S / row statistics — live in this module only.
 *
 * MUST: callers do not bypass this function and rewrite the formulas (the convention "do not keep a
 * derived value in an independently updated field"). Bypassing it lets the estimator keep claiming a
 * different number, with no exception and no warning, against an implementation where only one side
 * was fixed (the estimate does not stop execution, so a drift never turns red).
 *
 * It lives in `runtime/` because {@link planRowBlocks} lives in `runtime/fusion-rules/` (entry point
 * `fusion.ts`) (placing it in `kernels/` would create a reverse kernels → runtime import).
 */

import { STATE_STATS_STRIDE, stateSliding } from "../kernels/state-attention.ts";
import { stateStatsPvEligible } from "../kernels/state-attention-stats-pv.ts";
import type { StateStorage } from "../kernels/state-storage.ts";
import { planRowBlocks, type RowBlockLimit } from "./fusion.ts";
import type { StateAttentionReduce } from "./session-types.ts";

/**
 * Whether the states form asks for the fused stats + PV (the `fuseStatsPv` input of
 * {@link planStateAttention}; the plan then applies the shape's eligibility).
 *
 * MUST: execution and the estimate read this one condition. An f16 slot takes ③': the fused kernel
 * has no f16 form, because it needs a column limit of at most 1024, and a full slot's column limit is
 * its capacity, so the two meet only at capacities where halving the slot buys nothing.
 */
export const stateFusesStatsPv = (reduce: StateAttentionReduce, storage: StateStorage): boolean =>
  reduce === "parallel-fused" && storage === "f32";

/**
 * The geometry of one node (only the fields of `StateAttentionGeometry` that size the temporaries).
 *
 * `depth` / `kvRepeat` / `scale` do not affect the size of the temporaries, so they are not taken.
 */
export type StateAttentionExtent = {
  /** `B·H` (the axis folding batch and heads — the outer axis of S / row statistics). */
  readonly batchHeads: number;
  /** Physical chunk rows `M`. */
  readonly chunkRows: number;
  /** Capacity `C` of the state slot. */
  readonly capacity: number;
  /** Sliding window width `W`. The full variant is 0 or omitted (decided by {@link stateSliding} alone). */
  readonly window?: number;
  /** Given only by the parallel-fused states form. Readonly does not pass it. */
  readonly fuseStatsPv?: boolean;
};

/** The temporaries of one row block (allocation order — released in reverse). */
export type StateAttentionBlock = {
  /** Local row at the start of the block (`params.row_offset`). */
  readonly offset: number;
  /** Rows in the block (`params.rows_block`). */
  readonly rows: number;
  /** Scores S = `B·H × rows × colCap × 4` bytes. */
  readonly scoreBytes: number;
  /** Row statistics = `B·H × rows × STATE_STATS_STRIDE × 4` bytes. 0 when fused. */
  readonly statsBytes: number;
};

/** Result of {@link planStateAttention}. */
export type StateAttentionPlan = {
  /**
   * Upper bound of the column stride of S. Full: the capacity `C`; sliding: the resident window
   * `(W−1) + M` (the source of truth for the lower-bound formula is `assertStateGeometry` in
   * `src/kernels/state-attention.ts`).
   */
  readonly colCap: number;
  /** Whether the plan is the 2-dispatch one without an intermediate for row statistics. */
  readonly fusedStatsPv: boolean;
  /** Row blocks (the equal split of {@link planRowBlocks} — only one block's worth is alive at a time). */
  readonly blocks: readonly StateAttentionBlock[];
};

/**
 * Computes the column capacity of one states-form attention node and the temporary bytes per row
 * block (pure function).
 *
 * The equal split into the fewest blocks such that **both** one S block = `B·H · rows · colCap · 4`
 * bytes and one row-statistics block fit within `limit` (the device's granted
 * `maxStorageBufferBindingSize`, or the smaller of it and the budget when a budget is set — ADR
 * 0124); there is no run-time autotuning (ADR 0022). A shape where not even one row fits fails
 * loudly in {@link planRowBlocks}.
 *
 * @param limit The per-temporary limit. A number is read as the storage-binding limit in bytes
 *   (granted value, no budget); when a budget applies, pass the result of `rowBlockLimit`
 *   (fusion.ts) (the same input as {@link planRowBlocks}).
 * @param forced Gives the number of row blocks explicitly (the entry for the test-only
 *   `ROW_BLOCK_SPLIT` — production callers do not pass it).
 */
export const planStateAttention = (
  extent: StateAttentionExtent,
  limit: number | RowBlockLimit,
  forced?: number,
): StateAttentionPlan => {
  const { batchHeads, chunkRows, capacity } = extent;
  const window = extent.window ?? 0;
  const colCap = stateSliding(window) ? window - 1 + chunkRows : capacity;
  const fusedStatsPv = extent.fuseStatsPv === true && stateStatsPvEligible(chunkRows, colCap);
  const statsRowBytes = fusedStatsPv ? 0 : STATE_STATS_STRIDE * 4;
  // MUST: the block count is decided by the **larger** per-row size of the row-proportional
  // temporaries (S and row statistics). Deciding by S alone, a shape whose column capacity is
  // smaller than the word count of row statistics (colCap = 1) fits S while one row-statistics block
  // exceeds the limit, and a shape that would pass if split is rejected at allocation.
  const rowBytes = batchHeads * Math.max(colCap * 4, statsRowBytes);
  const blocks = planRowBlocks(chunkRows, rowBytes, limit, forced).map((block) => ({
    offset: block.offset,
    rows: block.rows,
    scoreBytes: batchHeads * block.rows * colCap * 4,
    statsBytes: batchHeads * block.rows * statsRowBytes,
  }));
  return { colCap, fusedStatsPv, blocks };
};
