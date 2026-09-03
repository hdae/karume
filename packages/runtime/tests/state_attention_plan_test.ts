// states 形 attention のノード内一時の算式（`planStateAttention`）の門。
//
// この関数は実行計画（recipe-builder の `#buildStateAttention`）と見積り（estimate.ts）が
// 分け合う唯一の導出元なので、ここが守るのは「両者が同じ数を得る」ための性質そのもの:
// 列容量 `colCap` の 2 分岐・行ブロックの覆い（行を漏らさず重ねない）・1 枚が上限に収まること・
// 枚数の明示（テスト専用 `ROW_BLOCK_SPLIT` の受け口）が効くこと。
//
// 期待値は**手計算した定数**で置く — 実装と同じ式で組み直すと恒真化して、算式がずれたときに
// 1 件も落ちない。実行計画との突合（実 GPU）は estimate_test.ts の側にある。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";

/** 上限が効かない大きさ（行ブロックが常に 1 枚になる）。 */
const WIDE_LIMIT = 1 << 20;

Deno.test("full 変種の列容量はスロット容量 C（窓を渡さない形）", () => {
  // B·H = 8・M = 4・C = 16 → S = 8×4×16×4 = 2048 / 行統計 = 8×4×2×4 = 256
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT);
  assertEquals(plan.colCap, 16);
  assertEquals(plan.blocks, [{ offset: 0, rows: 4, scoreBytes: 2048, statsBytes: 256 }]);
  // window: 0 は「full」の綴り違い（`stateSliding` の 1 本で判定する）。
  assertEquals(
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16, window: 0 }, WIDE_LIMIT),
    plan,
  );
});

Deno.test("sliding 変種の列容量は W−1+M で、C を上げても動かない", () => {
  const at = (capacity: number) =>
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity, window: 6 }, WIDE_LIMIT);
  // 列容量 = 6−1+4 = 9 → S = 8×4×9×4 = 1152（行統計は列容量に依らない）
  assertEquals(at(16).colCap, 9);
  assertEquals(at(16).blocks, [{ offset: 0, rows: 4, scoreBytes: 1152, statsBytes: 256 }]);
  assertEquals(at(64), at(16));
});

Deno.test("行ブロックは chunk 行を漏らさず重ねずに覆い、1 枚が上限に収まる", () => {
  // 1 行 = 8×16×4 = 512B。上限 1,536B には 3 行入るが、4 行は 2 枚へ**等分**するので 2 行 + 2 行。
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 1536);
  assertEquals(plan.blocks, [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
  ]);
  // 覆いの性質（行の総和と offset の連続）を式ではなく結果から確かめる。
  assertEquals(plan.blocks.reduce((sum, block) => sum + block.rows, 0), 4);
  assertEquals(plan.blocks.map((block) => block.offset), [0, 2]);
  for (const block of plan.blocks) assert(block.scoreBytes <= 1536);
});

Deno.test("端数は先頭のブロックへ寄り、1 行狭いブロックが混ざる", () => {
  // M = 5・1 行 512B・上限 1,024B → 1 枚 2 行では足りず 3 枚（2 + 2 + 1 行）。
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 5, capacity: 16 }, 1024);
  assertEquals(plan.blocks, [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 4, rows: 1, scoreBytes: 512, statsBytes: 64 },
  ]);
});

Deno.test("枚数の明示（forced）は上限に余裕があっても割る", () => {
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT, 4);
  assertEquals(plan.blocks.length, 4);
  assertEquals(plan.blocks.map((block) => block.offset), [0, 1, 2, 3]);
  assertEquals(plan.blocks[0], { offset: 0, rows: 1, scoreBytes: 512, statsBytes: 64 });
  // 明示しなければ 1 枚（forced が実際に効いていることの対）。
  assertEquals(
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT).blocks.length,
    1,
  );
});

Deno.test("1 行でも上限に入らない形は fail loudly（行ブロックでは割り切れない）", () => {
  assertThrows(
    // 1 行 = 8×16×4 = 512B > 256B
    () => planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 256),
    ExecutionError,
    "既にストレージ束縛の上限を超える",
  );
});

Deno.test("収まらない枚数の明示は fail loudly（上限の検査を forced が迂回しない）", () => {
  assertThrows(
    // 4 行を 2 枚 = 1 枚 2 行 = 1,024B が上限 512B に入らない。
    () => planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 512, 2),
    ExecutionError,
    "2 枚では 1 枚 1024B が上限に収まらない",
  );
});
