/**
 * states 形 attention（ADR 0067 決定 4）のノード内一時の**算式 1 本**。
 *
 * 実行計画（recipe-builder の `#buildStateAttention`）と見積り（estimate.ts の
 * `stateAttentionTemps`）は同じ数を出さなければならないので、両者が分け合う 3 つの式
 * — 列容量 `colCap` の 2 分岐・行ブロックの割り方（{@link planRowBlocks}）・S / 行統計の
 * バイト式 — をこのモジュールにだけ置く。
 *
 * MUST: 呼び手はこの関数を迂回して式を書き直さない（規約「派生値を独立更新の欄に持たない」）。
 * 迂回すると、片方だけ直された実装に対して estimator が例外も警告も無く別の数を主張し続ける
 * （見積りは実行を止めないので、ずれても赤くならない）。
 *
 * 置き場が `runtime/` なのは {@link planRowBlocks} が `runtime/fusion.ts` に居るため
 * （`kernels/` へ置くと kernels → runtime の逆向き import になる）。
 */

import { STATE_STATS_STRIDE, stateSliding } from "../kernels/state-attention.ts";
import { planRowBlocks } from "./fusion.ts";

/**
 * 1 ノードぶんの幾何（`StateAttentionGeometry` のうち、一時の大きさを決める欄だけ）。
 *
 * `depth` / `kvRepeat` / `scale` は一時の大きさに効かないので受けない。
 */
export type StateAttentionExtent = {
  /** `B·H`（バッチと head を畳んだ軸 — S / 行統計の外側の軸）。 */
  readonly batchHeads: number;
  /** 物理 chunk 行 `M`。 */
  readonly chunkRows: number;
  /** state スロットの容量 `C`。 */
  readonly capacity: number;
  /** sliding の窓幅 `W`。full 変種は 0 か省略（判定は {@link stateSliding} の 1 本）。 */
  readonly window?: number;
};

/** 行ブロック 1 枚ぶんの一時（確保順 — 解放は逆順）。 */
export type StateAttentionBlock = {
  /** ブロック先頭の局所行（`params.row_offset`）。 */
  readonly offset: number;
  /** ブロックの行数（`params.rows_block`）。 */
  readonly rows: number;
  /** スコア S = `B·H × 行数 × colCap × 4` バイト。 */
  readonly scoreBytes: number;
  /** 行統計 = `B·H × 行数 × STATE_STATS_STRIDE × 4` バイト。 */
  readonly statsBytes: number;
};

/** {@link planStateAttention} の結果。 */
export type StateAttentionPlan = {
  /**
   * S の列ストライド上限。full は容量ぶん `C`・sliding は resident 窓 `(W−1) + M`
   * （下限式の正本は `src/kernels/state-attention.ts` の `assertStateGeometry`）。
   */
  readonly colCap: number;
  /** 行ブロック（{@link planRowBlocks} の等分 — 1 枚ぶんだけが同時生存する）。 */
  readonly blocks: readonly StateAttentionBlock[];
};

/**
 * states 形 attention 1 ノードの列容量と行ブロックごとの一時バイト数を出す（純関数）。
 *
 * S 1 枚 = `B·H · 行数 · colCap · 4` バイトが `limit`（device の granted
 * `maxStorageBufferBindingSize`）に収まる最小枚数の等分で、実行時オートチューンは持たない
 * （ADR 0022）。1 行でも収まらない形は {@link planRowBlocks} が fail loudly。
 *
 * @param limit ストレージ束縛の上限バイト数（granted 値）。
 * @param forced 行ブロックの枚数を明示する（テスト専用 `ROW_BLOCK_SPLIT` の受け口 —
 *   実運用の呼び手は渡さない）。
 */
export const planStateAttention = (
  extent: StateAttentionExtent,
  limit: number,
  forced?: number,
): StateAttentionPlan => {
  const { batchHeads, chunkRows, capacity } = extent;
  const window = extent.window ?? 0;
  const colCap = stateSliding(window) ? window - 1 + chunkRows : capacity;
  const blocks = planRowBlocks(chunkRows, batchHeads * colCap * 4, limit, forced).map((block) => ({
    offset: block.offset,
    rows: block.rows,
    scoreBytes: batchHeads * block.rows * colCap * 4,
    statsBytes: batchHeads * block.rows * STATE_STATS_STRIDE * 4,
  }));
  return { colCap, blocks };
};
