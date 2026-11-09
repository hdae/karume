// The host tier of the exact expert cache without a GPU (ADR 0146 decision 5 — expertCache.hostBytes):
// the construction-time fill in read order up to the budget, the seal that refuses a half-filled
// entry, and one least-recently-used order over every layer by bytes (get refreshes, put evicts the
// least recent until the held bytes fit). The load's use of the tier is in expert_cache_test.ts
// (fake device) and gpu_expert_cache_test.ts (bit identity).
//
// Entry sizes and keys are written out by hand (deriving them from the code under test would make
// the checks tautological).

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ExpertHostTier } from "../src/runtime/expert-host-tier.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

/** Two layers of 8 experts; layer 0's entry is 8 bytes, layer 1's 4. */
const ENTRIES = [8, 4];

/** `length` bytes all equal to `value` (one recognisable range). */
const bytesOf = (value: number, length: number): Uint8Array<ArrayBuffer> =>
  new Uint8Array(length).fill(value);

/** Fills whole entries, one range of two halves each, in the given order, and seals. */
const filledTier = (budget: number, order: readonly (readonly [number, number])[]) => {
  const tier = new ExpertHostTier(budget, 8, ENTRIES);
  for (const [layer, expert] of order) {
    const half = ENTRIES[layer] / 2;
    tier.fill(layer, expert, 0, bytesOf(10 * layer + expert, half));
    tier.fill(layer, expert, half, bytesOf(100 + 10 * layer + expert, half));
  }
  tier.seal();
  return tier;
};

describe("the host tier's fill at construction", () => {
  it("holds each expert's ranges in its entry, in the layout's places", () => {
    const tier = filledTier(64, [
      [0, 3],
      [1, 5],
    ]);
    assertEquals([...(tier.get(0, 3) ?? [])], [3, 3, 3, 3, 103, 103, 103, 103]);
    assertEquals([...(tier.get(1, 5) ?? [])], [15, 15, 115, 115]);
    assertEquals([tier.heldExperts, tier.heldBytes], [2, 12]);
  });

  it("copies the ranges (a later change of the block bytes does not reach the entry)", () => {
    const tier = new ExpertHostTier(64, 8, ENTRIES);
    const block = bytesOf(7, 8);
    tier.fill(0, 1, 0, block.subarray(0, 4));
    tier.fill(0, 1, 4, block.subarray(4, 8));
    tier.seal();
    block.fill(0);
    assertEquals([...(tier.get(0, 1) ?? [])], [7, 7, 7, 7, 7, 7, 7, 7]);
  });

  it("holds, in read order, each expert whose entry still fits the budget", () => {
    // 20 bytes: layer 0's 2 and 6 (16 bytes), then 4 does not fit (24); layer 1's 5 fits (20).
    const tier = filledTier(20, [
      [0, 2],
      [0, 6],
      [0, 4],
      [1, 5],
      [1, 0],
    ]);
    assertEquals(
      [tier.get(0, 2) !== undefined, tier.get(0, 6) !== undefined, tier.get(0, 4)],
      [true, true, undefined],
    );
    assertEquals([tier.get(1, 5) !== undefined, tier.get(1, 0)], [true, undefined]);
    assertEquals([tier.heldExperts, tier.heldBytes], [3, 20]);
  });

  it("refuses a seal while a held entry is filled only in part, and a fill after the seal", () => {
    const tier = new ExpertHostTier(64, 8, ENTRIES);
    tier.fill(0, 1, 0, bytesOf(1, 4));
    assertThrows(() => tier.seal(), ExecutionError, "filled with 4 of its 8 bytes");
    const sealed = filledTier(64, [[0, 1]]);
    assertThrows(() => sealed.fill(0, 2, 0, bytesOf(1, 4)), ExecutionError, "after construction");
  });

  it("refuses a range outside the entry, an expert out of range, and a use before the seal", () => {
    const tier = new ExpertHostTier(64, 8, ENTRIES);
    assertThrows(
      () => tier.fill(1, 0, 2, bytesOf(1, 4)),
      ExecutionError,
      "outside its 4-byte entry",
    );
    assertThrows(() => tier.fill(0, 8, 0, bytesOf(1, 4)), ExecutionError, "not in [0, 8)");
    assertThrows(() => tier.get(0, 0), ExecutionError, "before construction sealed it");
  });

  it("refuses a budget that is not a positive safe integer", () => {
    for (const budget of [0, -1, 1.5, Number.NaN]) {
      assertThrows(() => new ExpertHostTier(budget, 8, ENTRIES), ExecutionError, "budget");
    }
  });
});

describe("the host tier's recency over every layer, by bytes", () => {
  it("evicts the least recently used first when a put goes over the budget", () => {
    // 20 bytes held: 0:1 (oldest), 1:2, 0:3 — a put of 1:4 (4 bytes) needs 4 more, evicting 0:1.
    const tier = filledTier(20, [
      [0, 1],
      [1, 2],
      [0, 3],
    ]);
    tier.put(1, 4, bytesOf(44, 4));
    assertEquals(tier.get(0, 1), undefined);
    assertEquals([tier.heldExperts, tier.heldBytes], [3, 16]);
    assertEquals([...(tier.get(1, 4) ?? [])], [44, 44, 44, 44]);
  });

  it("makes a got expert the most recent, so the next eviction passes it over", () => {
    const tier = filledTier(20, [
      [0, 1],
      [1, 2],
      [0, 3],
    ]);
    // 0:1 becomes the most recent: the eviction order is now 1:2, 0:3, 0:1.
    tier.get(0, 1);
    // An 8-byte put needs 8 bytes freed: 1:2 (4) and then 0:3 (8) go, 0:1 stays.
    tier.put(0, 7, bytesOf(77, 8));
    assertEquals([tier.get(1, 2), tier.get(0, 3)], [undefined, undefined]);
    assertEquals(tier.get(0, 1) !== undefined, true);
    assertEquals([tier.heldExperts, tier.heldBytes], [2, 16]);
  });

  it("gives back an evicted expert only once it is put again, with the bytes of that put", () => {
    const tier = filledTier(8, [[0, 1]]);
    tier.put(0, 2, bytesOf(22, 8));
    assertEquals(tier.get(0, 1), undefined);
    const again = bytesOf(11, 8);
    tier.put(0, 1, again);
    assertStrictEquals(tier.get(0, 1), again);
    assertEquals(tier.get(0, 2), undefined);
  });

  it("keeps the entry it was given (no copy) and refuses a put of a held expert or a wrong size", () => {
    const tier = filledTier(64, [[0, 1]]);
    const entry = bytesOf(5, 4);
    tier.put(1, 3, entry);
    assertStrictEquals(tier.get(1, 3), entry);
    assertThrows(() => tier.put(0, 1, bytesOf(1, 8)), ExecutionError, "while held");
    assertThrows(() => tier.put(1, 6, bytesOf(1, 8)), ExecutionError, "as 8 bytes, not 4");
  });

  it("holds nothing of an entry larger than the whole budget, and evicts nothing for it", () => {
    const tier = filledTier(6, [[1, 0]]);
    tier.put(0, 1, bytesOf(1, 8));
    assertEquals([tier.get(0, 1), tier.get(1, 0) !== undefined], [undefined, true]);
    assertEquals([tier.heldExperts, tier.heldBytes], [1, 4]);
  });

  it("drops every entry on clear", () => {
    const tier = filledTier(64, [
      [0, 1],
      [1, 2],
    ]);
    tier.clear();
    assertEquals([tier.heldExperts, tier.heldBytes, tier.get(0, 1)], [0, 0, undefined]);
  });
});
