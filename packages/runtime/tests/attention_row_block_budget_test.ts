// The check of how the per-temporary budget of attention's row blocks
// (`SessionOptions.attentionRowBlockMaxBytes` — ADR 0124) is derived (no GPU needed).
//
// It protects four points: ① the limit is the smaller of the binding limit and the budget (with no
// budget it is the binding limit itself) ② that limit yields "the smallest block count at which one
// S block and one row-statistics block each fit" (both fused attention and the states form) ③ a
// budget smaller than one row fails loudly and names the budget ④ a test-only forced block count
// passes only when it fits the smaller of the limit and the budget.
//
// Expected values are **hand-computed constants** (rebuilding them with the implementation's
// formula would make the test tautological). The block counts for the Wan2.2 TI2V-5B shapes match
// the values the design (d6's fact "row block count — computed values") computed independently in
// node.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planRowBlocks, rowBlockLimit } from "../src/runtime/fusion.ts";
import { planAttentionBlocks } from "../src/runtime/recipe-builders/attention.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";

const MiB = 1 << 20;
const GiB = 1 << 30;

/**
 * The `maxStorageBufferBindingSize` an RTX 3080 Ti / B570 grants (the premise of ADR 0118 decision
 * 6's table).
 */
const GRANTED = 2_147_483_644;

/** The attention head count of Wan2.2 TI2V-5B (B = 1, so B·H = 24). */
const WAN22_HEADS = 24;
/** Latent token counts at 1280×704 (17 frames = 4,400, 121 frames = 27,280) and 512 text tokens. */
const TOKENS_17F = 4400;
const TOKENS_121F = 27280;
const TEXT_TOKENS = 512;

/**
 * Fused attention block count per budget (the preset is given by the bytes per S element —
 * practical preset s16 = 2, reference preset f32 = 4).
 */
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

describe("rowBlockLimit (deriving the per-temporary limit of row blocks)", () => {
  it("with no budget it reports the binding limit itself (the default is the same as before the budget field)", () => {
    assertEquals(rowBlockLimit(GRANTED, undefined), {
      bytes: GRANTED,
      source: "maxStorageBufferBindingSize",
    });
  });

  it("a budget below the binding limit reports the budget", () => {
    assertEquals(rowBlockLimit(GRANTED, 512 * MiB), {
      bytes: 512 * MiB,
      source: "attentionRowBlockMaxBytes",
    });
  });

  it("a budget at or above the binding limit reports the binding limit (a budget never loosens the limit)", () => {
    assertEquals(rowBlockLimit(128 * MiB, 128 * MiB).source, "maxStorageBufferBindingSize");
    assertEquals(rowBlockLimit(128 * MiB, 4 * GiB), {
      bytes: 128 * MiB,
      source: "maxStorageBufferBindingSize",
    });
  });
});

describe("fused attention block count (Wan2.2 TI2V-5B shapes, binding limit 2,147,483,644 B)", () => {
  describe("self-attention at 121 frames (S = 27,280)", () => {
    it("the practical preset (S stored as s16 = 2 B) gives 17 / 34 / 67 / 134 blocks with no budget / 1 GiB / 512 MiB / 256 MiB", () => {
      // 1 row = 24 × 27,280 × 2 = 1,309,440 B. A block holds 1,640 / 820 / 410 / 205 rows.
      assertEquals(
        [undefined, GiB, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TOKENS_121F, 2, budget),
        ),
        [17, 34, 67, 134],
      );
    });

    it("the reference preset (f32 = 4 B) gives 34 / 134 / 268 blocks with no budget / 512 MiB / 256 MiB", () => {
      // 1 row = 2,618,880 B. A block holds 820 / 205 / 102 rows.
      assertEquals(
        [undefined, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TOKENS_121F, 4, budget),
        ),
        [34, 134, 268],
      );
    });
  });

  describe("cross-attention at 121 frames (512 text columns) is also split by the budget", () => {
    it("the practical preset gives 1 / 2 / 3 blocks with no budget / 512 MiB / 256 MiB", () => {
      // 1 row = 24 × 512 × 2 = 24,576 B. A block holds 87,381 / 21,845 / 10,922 rows.
      assertEquals(
        [undefined, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TEXT_TOKENS, 2, budget),
        ),
        [1, 2, 3],
      );
    });

    it("the reference preset gives 1 / 2 / 3 / 5 blocks with no budget / 1 GiB / 512 MiB / 256 MiB", () => {
      // 1 row = 49,152 B. A block holds 43,690 / 21,845 / 10,922 / 5,461 rows.
      assertEquals(
        [undefined, GiB, 512 * MiB, 256 * MiB].map((budget) =>
          blocksOf(TOKENS_121F, TEXT_TOKENS, 4, budget),
        ),
        [1, 2, 3, 5],
      );
    });
  });

  describe("17 frames (S = 4,400)", () => {
    it("self gives 1 / 2 / 4 blocks with the practical preset and 1 / 4 / 7 with the reference preset (no budget / 512 MiB / 256 MiB)", () => {
      // Practical preset: 1 row = 211,200 B (10,168 / 2,542 / 1,271 rows per block). Reference
      // preset: 1 row = 422,400 B (5,084 / 1,271 / 635 rows).
      const budgets = [undefined, 512 * MiB, 256 * MiB];
      assertEquals(
        budgets.map((budget) => blocksOf(TOKENS_17F, TOKENS_17F, 2, budget)),
        [1, 2, 4],
      );
      assertEquals(
        budgets.map((budget) => blocksOf(TOKENS_17F, TOKENS_17F, 4, budget)),
        [1, 4, 7],
      );
    });

    it("cross is 1 block for every preset and every budget", () => {
      for (const scoreBytes of [2, 4]) {
        for (const budget of [undefined, 512 * MiB, 256 * MiB]) {
          assertEquals(blocksOf(TOKENS_17F, TEXT_TOKENS, scoreBytes, budget), 1);
        }
      }
    });
  });

  it("an S block split by the budget is within the budget, while S plus the row statistics may exceed it (a per-temporary limit)", () => {
    // B·H = 8, M = 4, N = 16, f32 → a row of scores is 512 B, row statistics 64 B. A budget of
    // 1,100 B gives 2 rows per block (2 rows 1,024 B ≤ 1,100 B < 3 rows 1,536 B). One S block of
    // 1,024 B and one row-statistics block of 128 B are each within the budget.
    const blocks = planAttentionBlocks(
      { batch: 8, rows: 4, cols: 16, scoreBytes: 4 },
      rowBlockLimit(GRANTED, 1100),
    );
    assertEquals(
      blocks.map((block) => [block.scoreBytes, block.statsBytes]),
      [
        [1024, 128],
        [1024, 128],
      ],
    );
    // The total of 1,152 B exceeds the 1,100 B budget — the budget is not a limit on the total.
    assertEquals(blocks[0].scoreBytes + blocks[0].statsBytes > 1100, true);
  });
});

// The block count must also bound the row statistics, not only S: these shapes have a row of S
// smaller than its row statistics, so splitting by S alone would overflow the binding limit.
describe("row statistics also bound the block count (plain attention, no states)", () => {
  it("a shape with key length 1 automatically picks a block count at which the row statistics fit the limit too", () => {
    // f32, q = [1, 32, 1048576, 1], k = v = [1, 32, 1, 1], limit 128 MiB. A row of scores is
    // 32×4 = 128B, the row statistics 32×8 = 256B. Splitting by the scores alone gives 1 block,
    // and the 256 MiB of row statistics would be twice the limit.
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

  it("f16 compute with a key length below 4 is also split by the row statistics", () => {
    // B·H = 1, M = 8, N = 3, f16 → a row of scores is 6B, row statistics 8B. Limit 24B → 3 rows per
    // block: 3 / 3 / 2. The limit is chosen so that "splitting by the scores alone overflows the
    // row statistics": by the scores alone it is 4 rows per block × 2 blocks, and the 32B of row
    // statistics in one block exceed the 24B limit (with a 16B limit both formulas give 2 rows per
    // block and cannot be told apart).
    const blocks = planAttentionBlocks({ batch: 1, rows: 8, cols: 3, scoreBytes: 2 }, 24);
    assertEquals(
      blocks.map((block) => block.rows),
      [3, 3, 2],
    );
    for (const block of blocks) {
      assert(block.scoreBytes <= 24 && block.statsBytes <= 24, JSON.stringify(block));
    }
  });
});

describe("a budget smaller than one row", () => {
  it("fused attention fails loudly naming the budget (not with the binding-limit message)", () => {
    // 1 row of the practical preset at 121 frames = 1,309,440 B.
    const error = assertThrows(
      () => blocksOf(TOKENS_121F, TOKENS_121F, 2, 1 * MiB),
      ExecutionError,
      "set the budget to at least one row",
    );
    assertEquals(error.message.includes("exceeds the storage binding limit"), false, error.message);
  });

  it("the readonly states form (1 row = decode) also fails naming the budget when a small budget meets a long capacity", () => {
    // B·H = 8, M = 1, C = 131,072 → a row of scores is 8 × 131,072 × 4 = 4 MiB. A 1 MiB budget
    // cannot split it.
    assertThrows(
      () =>
        planStateAttention(
          { batchHeads: 8, chunkRows: 1, capacity: 131072 },
          rowBlockLimit(GRANTED, 1 * MiB),
        ),
      ExecutionError,
      "set the budget to at least one row",
    );
  });
});

describe("block count of the states form", () => {
  it("the budget sets the block count with the same formula as the binding limit (B·H = 8, M = 4, C = 16 with a 1,024 B budget gives 2 rows × 2 blocks)", () => {
    // 1 row = 8 × 16 × 4 = 512 B (larger than the 64 B of row statistics).
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

describe("the test-only forced block count (ROW_BLOCK_SPLIT) together with a budget", () => {
  // 10 rows × 100 B. A 400 B budget allows up to 4 rows per block.
  const limit = rowBlockLimit(GRANTED, 400);

  it("a block count that fits the budget passes (forcing a finer split than the budget is allowed)", () => {
    assertEquals(
      planRowBlocks(10, 100, limit, 5).map((block) => block.rows),
      [2, 2, 2, 2, 2],
    );
    assertEquals(
      planRowBlocks(10, 100, limit, 3).map((block) => block.rows),
      [4, 3, 3],
    );
  });

  it("a block count that does not fit the budget fails loudly naming the budget (forcing does not loosen the budget)", () => {
    assertThrows(
      () => planRowBlocks(10, 100, limit, 2),
      ExecutionError,
      "with 2 blocks, one block of 500B does not fit the budget attentionRowBlockMaxBytes",
    );
  });

  it("forcing with no budget looks only at the binding limit, as before", () => {
    assertEquals(planRowBlocks(10, 100, rowBlockLimit(GRANTED, undefined), 2).length, 2);
  });
});
