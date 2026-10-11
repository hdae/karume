// attention の行ブロックの一時 1 本ごとの予算（`SessionOptions.attentionRowBlockMaxBytes` — ADR 0124）の
// 導出の門（GPU 不要）。
//
// 守るのは 4 点: ①上限は束縛上限と予算の小さいほう（予算なしは束縛上限そのもの）②その上限で
// 「S 1 枚と行統計 1 枚のそれぞれが収まる最小枚数」が出る（融合 attention・states 形の両方）
// ③1 行ぶんにも満たない予算は予算を名指して fail loudly ④テスト専用の強制枚数は予算との小さいほうに
// 収まる枚数だけが通る。
//
// 期待値は**手計算した定数**で置く（実装と同じ式で組み直すと恒真化する）。Wan2.2 TI2V-5B の形の枚数は
// 設計（d6 の事実「行ブロックの枚数 — 計算値」）が node で独立に計算した値と同じ。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planRowBlocks, rowBlockLimit } from "../src/runtime/fusion.ts";
import { planAttentionBlocks } from "../src/runtime/recipe-builders/attention.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";

const MiB = 1 << 20;
const GiB = 1 << 30;

/** RTX 3080 Ti / B570 が granted する `maxStorageBufferBindingSize`（ADR 0118 決定 6 の表の前提）。 */
const GRANTED = 2_147_483_644;

/** Wan2.2 TI2V-5B の attention の head 数（B = 1 なので B·H = 24）。 */
const WAN22_HEADS = 24;
/** 1280×704 の潜在トークン数（17 フレーム = 4,400・121 フレーム = 27,280）とテキスト 512 トークン。 */
const TOKENS_17F = 4400;
const TOKENS_121F = 27280;
const TEXT_TOKENS = 512;

/** 予算ごとの融合 attention の枚数（S の 1 要素のバイト数で席を表す — 実用席 s16 = 2・参照席 f32 = 4）。 */
const blocksOf = (
  rows: number,
  cols: number,
  scoreBytes: number,
  budget: number | undefined,
): number =>
  planAttentionBlocks(
    { batch: WAN22_HEADS, rows, cols, scoreBytes },
    rowBlockLimit(GRANTED, budget),
  ).length;

describe("rowBlockLimit（行ブロックの一時 1 本ごとの上限の導出）", () => {
  it("予算なしは束縛上限そのものを名乗る（既定は予算の欄を足す前と同じ）", () => {
    assertEquals(rowBlockLimit(GRANTED, undefined), {
      bytes: GRANTED,
      source: "maxStorageBufferBindingSize",
    });
  });

  it("束縛上限より小さい予算は予算を名乗る", () => {
    assertEquals(rowBlockLimit(GRANTED, 512 * MiB), {
      bytes: 512 * MiB,
      source: "attentionRowBlockMaxBytes",
    });
  });

  it("束縛上限以上の予算は束縛上限を名乗る（予算で上限を緩める向きには効かない）", () => {
    assertEquals(rowBlockLimit(128 * MiB, 128 * MiB).source, "maxStorageBufferBindingSize");
    assertEquals(rowBlockLimit(128 * MiB, 4 * GiB), {
      bytes: 128 * MiB,
      source: "maxStorageBufferBindingSize",
    });
  });
});

describe("融合 attention の枚数（Wan2.2 TI2V-5B の形・束縛上限 2,147,483,644 B）", () => {
  describe("121 フレーム（S = 27,280）の self-attention", () => {
    it("実用席（S を s16 = 2 B で格納）は予算なし / 1 GiB / 512 MiB / 256 MiB で 17 / 34 / 67 / 134 枚", () => {
      // 1 行 = 24 × 27,280 × 2 = 1,309,440 B。1 枚に入る行は 1,640 / 820 / 410 / 205 行。
      assertEquals(
        [undefined, GiB, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TOKENS_121F, 2, budget)
        ),
        [17, 34, 67, 134],
      );
    });

    it("参照席（f32 = 4 B）は予算なし / 512 MiB / 256 MiB で 34 / 134 / 268 枚", () => {
      // 1 行 = 2,618,880 B。1 枚に入る行は 820 / 205 / 102 行。
      assertEquals(
        [undefined, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TOKENS_121F, 4, budget)
        ),
        [34, 134, 268],
      );
    });
  });

  describe("121 フレームの cross-attention（テキスト 512 列）も予算で割れる", () => {
    it("実用席は予算なし / 512 MiB / 256 MiB で 1 / 2 / 3 枚", () => {
      // 1 行 = 24 × 512 × 2 = 24,576 B。1 枚に入る行は 87,381 / 21,845 / 10,922 行。
      assertEquals(
        [undefined, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TEXT_TOKENS, 2, budget)
        ),
        [1, 2, 3],
      );
    });

    it("参照席は予算なし / 1 GiB / 512 MiB / 256 MiB で 1 / 2 / 3 / 5 枚", () => {
      // 1 行 = 49,152 B。1 枚に入る行は 43,690 / 21,845 / 10,922 / 5,461 行。
      assertEquals(
        [undefined, GiB, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TEXT_TOKENS, 4, budget)
        ),
        [1, 2, 3, 5],
      );
    });
  });

  describe("17 フレーム（S = 4,400）", () => {
    it("self は実用席で 1 / 2 / 4 枚・参照席で 1 / 4 / 7 枚（予算なし / 512 MiB / 256 MiB）", () => {
      // 実用席の 1 行 = 211,200 B（1 枚 10,168 / 2,542 / 1,271 行）。参照席の 1 行 = 422,400 B
      // （5,084 / 1,271 / 635 行）。
      const budgets = [undefined, 512 * MiB, 256 * MiB];
      assertEquals(budgets.map((budget) => blocksOf(TOKENS_17F, TOKENS_17F, 2, budget)), [1, 2, 4]);
      assertEquals(budgets.map((budget) => blocksOf(TOKENS_17F, TOKENS_17F, 4, budget)), [1, 4, 7]);
    });

    it("cross はどの席・どの予算でも 1 枚", () => {
      for (const scoreBytes of [2, 4]) {
        for (const budget of [undefined, 512 * MiB, 256 * MiB]) {
          assertEquals(blocksOf(TOKENS_17F, TEXT_TOKENS, scoreBytes, budget), 1);
        }
      }
    });
  });

  it("予算で割った 1 枚の S は予算以下で、S と行統計の合計は予算を超えうる（1 本ごとの上限）", () => {
    // B·H = 8・M = 4・N = 16・f32 → 1 行のスコア 512 B・行統計 64 B。予算 1,100 B では 2 行 / 枚
    // （2 行 1,024 B ≤ 1,100 B < 3 行 1,536 B）。S 1 枚 1,024 B と行統計 1 枚 128 B はそれぞれ予算以下。
    const blocks = planAttentionBlocks(
      { batch: 8, rows: 4, cols: 16, scoreBytes: 4 },
      rowBlockLimit(GRANTED, 1100),
    );
    assertEquals(blocks.map((block) => [block.scoreBytes, block.statsBytes]), [
      [1024, 128],
      [1024, 128],
    ]);
    // 合計 1,152 B は予算 1,100 B を超える — 予算は合計の上限ではない。
    assertEquals(blocks[0].scoreBytes + blocks[0].statsBytes > 1100, true);
  });
});

describe("1 行ぶんにも満たない予算", () => {
  it("融合 attention は予算を名指して fail loudly（束縛上限の文言にしない）", () => {
    // 121 フレームの実用席の 1 行 = 1,309,440 B。
    const error = assertThrows(
      () => blocksOf(TOKENS_121F, TOKENS_121F, 2, 1 * MiB),
      ExecutionError,
      "予算 attentionRowBlockMaxBytes",
    );
    assertEquals(error.message.includes("ストレージ束縛の上限を超える"), false, error.message);
  });

  it("readonly の states 形（1 行 = decode）でも、長い容量に小さい予算を渡すと予算を名指して落ちる", () => {
    // B·H = 8・M = 1・C = 131,072 → 1 行のスコア 8 × 131,072 × 4 = 4 MiB。予算 1 MiB では割り切れない。
    assertThrows(
      () =>
        planStateAttention(
          { batchHeads: 8, chunkRows: 1, capacity: 131072 },
          rowBlockLimit(GRANTED, 1 * MiB),
        ),
      ExecutionError,
      "予算 attentionRowBlockMaxBytes",
    );
  });
});

describe("states 形の枚数", () => {
  it("予算は束縛上限と同じ式で枚数を決める（B·H = 8・M = 4・C = 16 で予算 1,024 B は 2 行 × 2 枚）", () => {
    // 1 行 = 8 × 16 × 4 = 512 B（行統計 64 B より大きい）。
    const plan = planStateAttention(
      { batchHeads: 8, chunkRows: 4, capacity: 16 },
      rowBlockLimit(GRANTED, 1024),
    );
    assertEquals(plan.blocks, [
      { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
      { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    ]);
  });
});

describe("テスト専用の強制枚数（ROW_BLOCK_SPLIT）と予算の併用", () => {
  // 10 行 × 100 B。予算 400 B なら 1 枚 4 行まで。
  const limit = rowBlockLimit(GRANTED, 400);

  it("予算に収まる枚数は通る（強制が予算より細かく割るのは許す）", () => {
    assertEquals(planRowBlocks(10, 100, limit, 5).map((block) => block.rows), [2, 2, 2, 2, 2]);
    assertEquals(planRowBlocks(10, 100, limit, 3).map((block) => block.rows), [4, 3, 3]);
  });

  it("予算に収まらない枚数は予算を名指して fail loudly（強制で予算を緩めない）", () => {
    assertThrows(
      () => planRowBlocks(10, 100, limit, 2),
      ExecutionError,
      "2 枚では 1 枚 500B が予算 attentionRowBlockMaxBytesに収まらない",
    );
  });

  it("予算なしの強制は従来どおり束縛上限だけを見る", () => {
    assertEquals(planRowBlocks(10, 100, rowBlockLimit(GRANTED, undefined), 2).length, 2);
  });
});
