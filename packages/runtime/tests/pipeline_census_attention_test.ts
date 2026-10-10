/**
 * 融合 attention の完全適格の検査（tests/helpers/pipeline-census.ts の `assertAttentionSeatsComplete`）を、
 * 合成した census の行で固定する（GPU 不要）。キーは src の key 関数から組む — 実行が出すキーと同じ綴りで
 * 検査が当たることの裏取りで、文字列を二重に持たない。
 *
 * 故障注入の役: 宣言どおりの census に縮退した dispatch を 1 本混ぜると赤になる（恒真の検査でないこと）。
 */

import { assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { attentionPvKey, attentionQkKey, attentionStatsKey } from "../src/kernels/attention.ts";
import { attentionPvI8a8Key, attentionQkI8a8Key } from "../src/kernels/attention-i8a8.ts";
import { linearI8a8Key } from "../src/kernels/linear-i8a8.ts";
import type { ScoreStorage } from "../src/kernels/score-storage.ts";
import { type AttentionRoute, attentionRoute } from "../src/runtime/attention-route.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import {
  assertAttentionSeatsComplete,
  assertSeatsApplied,
  type CensusRow,
} from "./helpers/pipeline-census.ts";

const LAYERS = 30;
const A8_S16: SessionOptions = { attentionCompute: "a8", attentionScoreStorage: "f16" };

/**
 * 融合 attention 1 ノード × 行ブロック `blocks` 枚の census の行（①②③ が各 `blocks` 本）。経路は
 * `attentionRoute` の戻りをそのまま使う — 実行（buildAttention）と同じ判定から行を作る。
 */
const attentionRows = (route: AttentionRoute, blocks: number): readonly CensusRow[] => {
  const score: ScoreStorage = route.score;
  return [
    {
      key: route.qk === "i8a8"
        ? attentionQkI8a8Key(true, true, score)
        : attentionQkKey(true, "f32", score),
      dispatchCount: blocks,
    },
    { key: attentionStatsKey("f32", score, 16), dispatchCount: blocks },
    {
      key: route.pv === "i8a8"
        ? attentionPvI8a8Key(true, true, score)
        : attentionPvKey(true, "f32", score),
      dispatchCount: blocks,
    },
  ];
};

/**
 * a8 + s16 の席で、D = 128 の self（キー長 `selfCols`・行ブロック `blocks` 枚）と cross（N = 512・1 枚）を
 * transformer 30 層ぶん回した census（linear の i8a8 も混ぜる — 族の外の行を数えないことの確認）。
 */
const ditCensus = (selfCols: number, blocks: number): readonly CensusRow[] => {
  const seats = { attentionCompute: "a8", attentionScoreStorage: "f16" } as const;
  const rows = [
    ...attentionRows(attentionRoute(seats, 128, selfCols), blocks),
    ...attentionRows(attentionRoute(seats, 128, 512), 1),
  ];
  // 族ごとに束ねる（実物の census と同じく、同じキーは 1 行に畳まれる）。
  const counts = new Map<string, number>();
  for (const { key, dispatchCount } of rows) {
    counts.set(key, (counts.get(key) ?? 0) + dispatchCount * LAYERS);
  }
  return [
    { key: linearI8a8Key(true, true), dispatchCount: 300 },
    ...[...counts].map(([key, dispatchCount]) => ({ key, dispatchCount })),
  ];
};

describe("assertAttentionSeatsComplete: 宣言どおりの census", () => {
  it("self / cross とも適格な形（N = 4096 / 512）は、行ブロックの枚数に依らず緑", () => {
    for (const blocks of [1, 3]) {
      assertAttentionSeatsComplete(ditCensus(4096, blocks), A8_S16, `${blocks} 枚`);
    }
  });

  it("a8 / s16 を宣言しない席は、dense の attention があっても見ない", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f32" }, 128, 1089),
      1,
    );
    assertAttentionSeatsComplete(census, {}, "席なし");
    assertAttentionSeatsComplete(census, { attentionCompute: "f32" }, "参照値");
  });

  it("a8 だけの席は S の格納を見ない（f32 格納の i8a8 で緑）", () => {
    const census = attentionRows(
      attentionRoute({ attentionCompute: "a8", attentionScoreStorage: "f32" }, 128, 4096),
      2,
    );
    assertAttentionSeatsComplete(census, { attentionCompute: "a8" }, "a8 だけ");
  });
});

describe("assertAttentionSeatsComplete: 縮退を含む census（故障注入）", () => {
  it("dense の attention_pv:v1:f32 を 1 本混ぜると、③PV 族の i8a8 の欠けで赤", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionPvKey(true, "f32", "f16"), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "混入"),
      Error,
      'attentionCompute="a8" の attention_pv が',
    );
  });

  it("②行統計の :s16 が 1 本欠けると、s16 の欠けで赤（a8 は満たしたまま）", () => {
    const census: CensusRow[] = [
      ...ditCensus(4096, 1),
      { key: attentionStatsKey("f32", "f32", 16), dispatchCount: 1 },
    ];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "s16 の欠け"),
      Error,
      'attentionScoreStorage="f16" の attention_stats が',
    );
  });

  it("self が N % 4 != 0（N = 1089 — Anima の 528×528）の census は赤、assertSeatsApplied は緑のまま", () => {
    const census = ditCensus(1089, 1);
    // 「1 本以上」の検査は cross の i8a8 / s16 だけで満たされる — これが今回塞ぐ穴。
    assertSeatsApplied(census, A8_S16, "1 本以上");
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "self の縮退"),
      Error,
      "f32 へ縮退",
    );
  });

  it("融合 attention が 1 本も無い census は、宣言した族の欠落として赤（0 == 0 で緑にしない）", () => {
    const census: CensusRow[] = [{ key: linearI8a8Key(true, true), dispatchCount: 4 }];
    assertThrows(
      () => assertAttentionSeatsComplete(census, A8_S16, "attention なし"),
      Error,
      "1 本も無い",
    );
  });
});
