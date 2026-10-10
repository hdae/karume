/**
 * 融合 attention の**経路の判定**（席の要求 × 形 → 段ごとに実際に走る経路）— 判定の正本はここ 1 本。
 *
 * a8（`attentionCompute`）と s16（`attentionScoreStorage`）は、形の条件を満たさない段だけが f32 経路へ
 * 沈黙で縮退する（ADR 0030 決定 5 / ADR 0058 決定 3）。レシピ導出（recipe-builders/attention.ts の
 * `buildAttention`）と、系列の受理集合を通すテストが**同じ関数**を呼ぶ — 判定を二重に書くと、
 * 片方だけが動いて「テストでは適格・実行では縮退」が黙って起こる。
 *
 * NOTE: `export` はレシピ導出とテストのため（mod.ts の公開面には出さない — ADR 0008）。
 */

import { attentionPvI8a8Eligible, attentionQkI8a8Eligible } from "../kernels/attention-i8a8.ts";
import { attentionScoreUsesF16, type ScoreStorage } from "../kernels/score-storage.ts";
import type { ComputePrecision } from "./session-types.ts";

/** 段（①QK / ③PV）の経路。`dense` は f32 / f16 の GEMM 骨格（`attentionCompute` の f32 / f16）。 */
export type AttentionStageRoute = "i8a8" | "dense";

/** 1 つの融合 attention で実際に走る経路（②行統計は常に f32 計算で、S の格納形だけが動く）。 */
export type AttentionRoute = {
  readonly qk: AttentionStageRoute;
  readonly pv: AttentionStageRoute;
  readonly score: ScoreStorage;
};

/** 経路を決める席（Session の解決済みの値 — 省略値の代入は session-build.ts の責務）。 */
export type AttentionSeats = {
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
};

/**
 * 席の要求と形（`depth` = D・`cols` = キー長 N）から、段ごとに実際に走る経路を決める。
 *
 * - i8a8: MUST: 適格判定は**段ごとに独立**。① は q / k のパック方向が D・③ は P̃ / Vᵀ のパック方向が
 *   N なので、条件も別々になる（両方満たさない形・片方だけの形が実在する）。
 * - S の格納形: `pack2x16float` の 2 要素／語なので、書き手 ①QK が v4 経路を取る形
 *   （`D % 4 == 0 && N % 4 == 0`）だけが適格。MUST: S の格納形は 3 カーネル（①②③）で**同時に**
 *   切り替わる — 書き手と読み手で一致していなければならないので、段ごとの混成はあり得ない
 *   （i8a8 の段ごとの独立とは別の話で、① が i8a8 でも f32 でも S の格納形は 1 つに決まる）。
 *
 * MUST: mask × a8・GQA × a8 の拒否は、この戻り値ではなく**要求されたモード**
 * （`attentionCompute === "a8"`）で見る（buildAttention の MUST — 適格判定込みの値で見ると、
 * 縮退する形だけが拒否をすり抜ける）。
 */
export const attentionRoute = (
  seats: AttentionSeats,
  depth: number,
  cols: number,
): AttentionRoute => {
  const i8a8 = seats.attentionCompute === "a8";
  return {
    qk: i8a8 && attentionQkI8a8Eligible(depth) ? "i8a8" : "dense",
    pv: i8a8 && attentionPvI8a8Eligible(cols) ? "i8a8" : "dense",
    score: seats.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols)
      ? "f16"
      : "f32",
  };
};
