// The check of the formulas for the in-node temporaries of states-form attention
// (`planStateAttention`).
//
// This function is the single derivation that the execution plan (`buildStateAttention` of
// recipe-builders/attention.ts) and the estimate (estimate.ts) share, so what this guards are the
// very properties that make both get the same numbers: the two branches of the column capacity
// `colCap`, the row blocks' coverage (no row left out, none overlapping), each block fitting
// under the limit, and an explicit block count (the hook for the test-only `ROW_BLOCK_SPLIT`)
// taking effect.
//
// The expected values are **hand-computed constants** — rebuilding them with the same formulas as
// the implementation would turn the test into a tautology that catches nothing when the formulas
// drift. The comparison with the execution plan (real GPU) lives in estimate_test.ts.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";

/** A size at which the limit never applies (there is always one row block). */
const WIDE_LIMIT = 1 << 20;

test("the full variant's column capacity is the slot capacity C (the form that passes no window)", () => {
  // B·H = 8, M = 4, C = 16 → S = 8×4×16×4 = 2048 / row stats = 8×4×2×4 = 256
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT);
  assertEquals(plan.colCap, 16);
  assertEquals(plan.blocks, [{ offset: 0, rows: 4, scoreBytes: 2048, statsBytes: 256 }]);
  // window: 0 is another spelling of "full" (decided by `stateSliding` alone).
  assertEquals(
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16, window: 0 }, WIDE_LIMIT),
    plan,
  );
});

test("the sliding variant's column capacity is W−1+M and does not move when C grows", () => {
  const at = (capacity: number) =>
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity, window: 6 }, WIDE_LIMIT);
  // column capacity = 6−1+4 = 9 → S = 8×4×9×4 = 1152 (the row stats do not depend on it)
  assertEquals(at(16).colCap, 9);
  assertEquals(at(16).blocks, [{ offset: 0, rows: 4, scoreBytes: 1152, statsBytes: 256 }]);
  assertEquals(at(64), at(16));
});

test("the row blocks cover the chunk rows with none left out or overlapping, and each fits under the limit", () => {
  // 1 row = 8×16×4 = 512B. A limit of 1,536B holds 3 rows, but 4 rows are split **evenly** into 2
  // blocks, so 2 rows + 2 rows.
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 1536);
  assertEquals(plan.blocks, [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
  ]);
  // Check the coverage properties (the sum of rows and contiguous offsets) from the result, not
  // from the formula.
  assertEquals(
    plan.blocks.reduce((sum, block) => sum + block.rows, 0),
    4,
  );
  assertEquals(
    plan.blocks.map((block) => block.offset),
    [0, 2],
  );
  for (const block of plan.blocks) assert(block.scoreBytes <= 1536);
});

test("the remainder goes to the leading blocks, so a block one row narrower is mixed in", () => {
  // M = 5, 1 row = 512B, limit 1,024B → 2 rows per block are not enough, so 3 blocks (2 + 2 + 1
  // rows).
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 5, capacity: 16 }, 1024);
  assertEquals(plan.blocks, [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 128 },
    { offset: 4, rows: 1, scoreBytes: 512, statsBytes: 64 },
  ]);
});

test("an explicit block count (forced) splits even when the limit has room to spare", () => {
  const plan = planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT, 4);
  assertEquals(plan.blocks.length, 4);
  assertEquals(
    plan.blocks.map((block) => block.offset),
    [0, 1, 2, 3],
  );
  assertEquals(plan.blocks[0], { offset: 0, rows: 1, scoreBytes: 512, statsBytes: 64 });
  // Without the explicit count there is one block (the counterpart showing forced really applies).
  assertEquals(
    planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, WIDE_LIMIT).blocks.length,
    1,
  );
});

test("a shape where even one row exceeds the limit fails loudly (row blocks cannot split it)", () => {
  assertThrows(
    // 1 row = 8×16×4 = 512B > 256B
    () => planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 256),
    ExecutionError,
    "already exceeds the storage binding limit",
  );
});

test("an explicit block count that does not fit fails loudly (forced does not bypass the limit check)", () => {
  assertThrows(
    // 4 rows in 2 blocks = 2 rows per block = 1,024B, which does not fit the 512B limit.
    () => planStateAttention({ batchHeads: 8, chunkRows: 4, capacity: 16 }, 512, 2),
    ExecutionError,
    "with 2 blocks, one block of 1024B does not fit the limit",
  );
});

test("the row-stats / PV fusion drops only the intermediate stats and keeps the columns and row blocks", () => {
  const extent = { batchHeads: 8, chunkRows: 4, capacity: 16 };
  const plain = planStateAttention(extent, 1536);
  const fused = planStateAttention({ ...extent, fuseStatsPv: true }, 1536);
  assertEquals(plain.fusedStatsPv, false);
  assertEquals(fused.fusedStatsPv, true);
  assertEquals(fused.colCap, plain.colCap);
  assertEquals(fused.blocks, [
    { offset: 0, rows: 2, scoreBytes: 1024, statsBytes: 0 },
    { offset: 2, rows: 2, scoreBytes: 1024, statsBytes: 0 },
  ]);
  const outside = { ...extent, capacity: 1025 };
  assertEquals(
    planStateAttention({ ...outside, fuseStatsPv: true }, WIDE_LIMIT),
    planStateAttention(outside, WIDE_LIMIT),
  );
});

test("a shape with column capacity 1 also splits so that the row stats fit under the limit", () => {
  // B·H = 1, M = 8, C = 1 → a row's scores are 4B while its row stats are 2 words = 8B. Looking
  // only at the scores, a 16B limit gives 4 rows × 2 blocks, and one block's row stats of 32B
  // exceed the limit. Splitting by the row stats' 8B gives 2 rows × 4 blocks.
  const plan = planStateAttention({ batchHeads: 1, chunkRows: 8, capacity: 1 }, 16);
  assertEquals(plan.blocks, [
    { offset: 0, rows: 2, scoreBytes: 8, statsBytes: 16 },
    { offset: 2, rows: 2, scoreBytes: 8, statsBytes: 16 },
    { offset: 4, rows: 2, scoreBytes: 8, statsBytes: 16 },
    { offset: 6, rows: 2, scoreBytes: 8, statsBytes: 16 },
  ]);
  for (const block of plan.blocks) {
    assert(block.scoreBytes <= 16 && block.statsBytes <= 16, JSON.stringify(block));
  }
});
