// 通常 attention（states 欄なし）のクエリ行ブロックの算式（`planAttentionBlocks`）の門。
//
// 守るのは「1 枚ぶんの一時（スコア S と行統計）が**どちらも**ストレージ束縛の上限に収まる最小枚数を
// 自動で選ぶ」こと。期待値は手計算した定数で置く（実装と同じ式で組み直すと恒真化する）。

import { assert, assertEquals } from "@std/assert";
import { planAttentionBlocks } from "../src/runtime/recipe-builders/attention.ts";

const MiB = 1 << 20;

Deno.test("キー長が長い通常の形はスコアの大きさだけで枚数が決まる", () => {
  // B·H = 8・M = 4・N = 16・f32 → 1 行のスコア 512B（行統計 64B より大きい）。上限 1,536B には
  // 3 行入るが、4 行は 2 枚へ等分するので 2 行 + 2 行。
  assertEquals(planAttentionBlocks({ batch: 8, rows: 4, cols: 16, scoreBytes: 4 }, 1536), [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
  ]);
});

Deno.test("キー長 1 の形は行統計も上限に収まる枚数を自動で選ぶ", () => {
  // f32・q = [1, 32, 1048576, 1]・k = v = [1, 32, 1, 1]・上限 128 MiB。1 行のスコアは 32×4 = 128B、
  // 行統計は 32×8 = 256B。スコアだけで割ると 1 枚になり、行統計 256 MiB が上限の 2 倍になる。
  const blocks = planAttentionBlocks(
    { batch: 32, rows: 1048576, cols: 1, scoreBytes: 4 },
    128 * MiB,
  );
  assertEquals(blocks, [
    { offset: 0, rows: 524288, scoreBytes: 64 * MiB, statsBytes: 128 * MiB },
    { offset: 524288, rows: 524288, scoreBytes: 64 * MiB, statsBytes: 128 * MiB },
  ]);
  for (const block of blocks) {
    assert(block.scoreBytes <= 128 * MiB && block.statsBytes <= 128 * MiB, JSON.stringify(block));
  }
});

Deno.test("f16 計算でキー長 4 未満の形も行統計で割る", () => {
  // B·H = 1・M = 8・N = 3・f16 → 1 行のスコア 6B・行統計 8B。上限 24B → 3 行 / 枚で 3 / 3 / 2。
  // 上限は「スコアだけで割ると行統計が溢れる」値に選ぶ: スコアだけなら 4 行 / 枚 × 2 枚で、
  // 1 枚の行統計 32B が上限 24B を超える（上限 16B では両式とも 2 行 / 枚になり区別できない）。
  const blocks = planAttentionBlocks({ batch: 1, rows: 8, cols: 3, scoreBytes: 2 }, 24);
  assertEquals(blocks.map((block) => block.rows), [3, 3, 2]);
  for (const block of blocks) {
    assert(block.scoreBytes <= 24 && block.statsBytes <= 24, JSON.stringify(block));
  }
});
