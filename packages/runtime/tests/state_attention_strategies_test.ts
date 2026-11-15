/**
 * The selection of the states-form attention's stage forms
 * (`src/runtime/state-attention-strategies.ts`): which kernel ①QK and ③PV take for a plan's row
 * count, the layer's kind and the Session's `stateAttentionReduce`; the priority between forms whose
 * cases overlap; and the refusal of a plan that asks for bidirectional rows and cannot have them.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { stateQkFoldHeads } from "../src/kernels/state-attention.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  resolveStateAttention,
  selectStateAttention,
  STATE_ATTENTION_STRATEGIES,
  type StateAttentionFacts,
} from "../src/runtime/state-attention-strategies.ts";

/** A full layer without GQA, the Session's omitted option, no fusion, causal rows. */
const facts = (overrides: Partial<StateAttentionFacts> = {}): StateAttentionFacts => ({
  where: "attention (states) [1,8,1,256] × [1,8,1,256]",
  chunkRows: 1,
  depth: 256,
  kvRepeat: 1,
  sliding: false,
  reduce: "sequential",
  fusedStatsPv: false,
  bidirectionalRows: false,
  ...overrides,
});

const ids = (overrides: Partial<StateAttentionFacts>): readonly [string, string] => {
  const { qk, pv } = selectStateAttention(facts(overrides));
  return [qk.id, pv.id];
};

describe("with the option omitted (sequential)", () => {
  it("runs the sequential forms on short plans and the tiled forms from 16 rows", () => {
    assertEquals(ids({ chunkRows: 1 }), ["qk-sequential", "pv-sequential"]);
    assertEquals(ids({ chunkRows: 15 }), ["qk-sequential", "pv-sequential"]);
    assertEquals(ids({ chunkRows: 16 }), ["qk-tiled", "pv-tiled"]);
    assertEquals(ids({ chunkRows: 768 }), ["qk-tiled", "pv-tiled"]);
  });

  it("never leaves the forms whose level is exact", () => {
    for (const chunkRows of [1, 8, 16, 768]) {
      const { qk, pv } = selectStateAttention(facts({ chunkRows }));
      assertEquals(STATE_ATTENTION_STRATEGIES[qk.id].level, "exact");
      assertEquals(STATE_ATTENTION_STRATEGIES[pv.id].level, "exact");
    }
  });
});

describe('under stateAttentionReduce "parallel"', () => {
  const parallel = { reduce: "parallel" } as const;

  it("splits ①QK only on plans of at most 8 rows", () => {
    assertEquals(ids({ ...parallel, chunkRows: 1 })[0], "qk-parallel");
    assertEquals(ids({ ...parallel, chunkRows: 8 })[0], "qk-parallel");
    assertEquals(ids({ ...parallel, chunkRows: 9 })[0], "qk-sequential");
    assertEquals(ids({ ...parallel, chunkRows: 16 })[0], "qk-tiled");
  });

  it("splits ③PV below 16 rows and gives the tiled form priority over the option from 16", () => {
    assertEquals(ids({ ...parallel, chunkRows: 1 })[1], "pv-parallel");
    assertEquals(ids({ ...parallel, chunkRows: 15 })[1], "pv-parallel");
    assertEquals(ids({ ...parallel, chunkRows: 16 })[1], "pv-tiled");
  });

  it("folds ①QK over the query heads of one kv plane exactly where the kernel names a class", () => {
    // Every (r, D, M) the kernel's own function is asked about gives the same answer here.
    for (const kvRepeat of [1, 2, 4, 8]) {
      for (const depth of [64, 128, 256]) {
        for (const chunkRows of [1, 4, 8]) {
          const heads = stateQkFoldHeads(false, kvRepeat, depth, chunkRows);
          const { qk } = selectStateAttention(facts({ ...parallel, kvRepeat, depth, chunkRows }));
          assertEquals(
            qk,
            heads > 1 ? { id: "qk-folded", heads } : { id: "qk-parallel" },
            `r=${kvRepeat} D=${depth} M=${chunkRows}`,
          );
        }
      }
    }
  });

  it("takes the fused ②③ where the block plan chose it", () => {
    assertEquals(ids({ reduce: "parallel-fused", chunkRows: 1, fusedStatsPv: true }), [
      "qk-parallel",
      "stats-pv-fused",
    ]);
    // The option alone does not fuse: the block plan decides (slot storage, column limit).
    assertEquals(ids({ reduce: "parallel-fused", chunkRows: 1, fusedStatsPv: false }), [
      "qk-parallel",
      "pv-parallel",
    ]);
  });
});

describe("bidirectional rows", () => {
  it("are carried by the tiled ①QK of a sliding layer, and not by a full layer", () => {
    const sliding = selectStateAttention(
      facts({ sliding: true, chunkRows: 280, bidirectionalRows: true }),
    );
    assertEquals(sliding.qk, { id: "qk-tiled", bidirectional: true });
    const full = selectStateAttention(
      facts({ sliding: false, chunkRows: 280, bidirectionalRows: true }),
    );
    assertEquals(full.qk, { id: "qk-tiled", bidirectional: false });
  });

  it("refuse a sliding plan that does not take the tiled ①QK", () => {
    const short = facts({ sliding: true, chunkRows: 8, bidirectionalRows: true });
    assertEquals(resolveStateAttention(short), {
      refusal: `${short.where}: generation.bidirectionalRows needs a plan of at least 16 rows (this one has 8)`,
    });
    assertThrows(() => selectStateAttention(short), ExecutionError, "at least 16 rows");
    // A full layer of the same plan stays causal and is not refused.
    assertEquals(
      ids({ sliding: false, chunkRows: 8, bidirectionalRows: true })[0],
      "qk-sequential",
    );
  });
});
