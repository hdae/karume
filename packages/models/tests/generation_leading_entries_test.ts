// Behaviour tests of the merge of per-part selections into a row's leading entries
// (`src/generation/leading-entries.ts`). Neither a GPU nor real assets are needed: the GPU's `topk`
// per part is modelled on the host (`selectParts`).
//
// What this pins: the merged entries are the row's own leading entries in the ranking's order
// (value descending, the smaller id first on a tie, +0 = −0, NaN first) — or the merge says it does
// not hold them. A wrong order here raises no exception anywhere: the sampler's draw would walk the
// candidates in another order and pick another token for the same uniform value.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { leadingEntries, type PartSelections } from "../src/generation/leading-entries.ts";

/** "left ranks above right" in the ranking, written as a rank per id: NaN, then the number, then the id. */
const ranksAbove = (row: Float32Array<ArrayBuffer>, left: number, right: number): boolean => {
  const leftNaN = Number.isNaN(row[left]);
  const rightNaN = Number.isNaN(row[right]);
  if (leftNaN !== rightNaN) return leftNaN;
  if (!leftNaN && row[left] !== row[right]) return row[left] > row[right];
  return left < right;
};

/** `ids` of `row` in the ranking's order. */
const ranked = (row: Float32Array<ArrayBuffer>, ids: readonly number[]): number[] =>
  ids.toSorted((left, right) => (ranksAbove(row, left, right) ? -1 : 1));

const allIds = (row: Float32Array<ArrayBuffer>): number[] =>
  Array.from({ length: row.length }, (_unused, id) => id);

/**
 * What the GPU returns for `row` cut into `parts` by residue: of the ids `≡ p (mod parts)`, the
 * first `perPart` in the ranking's order, each as its place in the part (`(id − p) / parts`).
 */
const selectParts = (
  row: Float32Array<ArrayBuffer>,
  parts: number,
  perPart: number,
): PartSelections => {
  const width = row.length / parts;
  const values = new Float32Array(parts * perPart);
  const indices = new Int32Array(parts * perPart);
  for (let part = 0; part < parts; part += 1) {
    ranked(
      row,
      allIds(row).filter((id) => id % parts === part),
    )
      .slice(0, perPart)
      .forEach((id, at) => {
        values[part * perPart + at] = row[id];
        indices[part * perPart + at] = (id - part) / parts;
      });
  }
  return { values, indices, parts, perPart, width };
};

const stream = (seed: number): (() => number) => {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return (state + 0.5) / 4294967296;
  };
};

const rowOf = (
  size: number,
  fill: number,
  entries: readonly (readonly [id: number, value: number])[],
): Float32Array<ArrayBuffer> => {
  const row = new Float32Array(size).fill(fill);
  for (const [id, value] of entries) row[id] = value;
  return row;
};

describe("the leading entries of a row from per-part selections", () => {
  it("are the row's first entries in the ranking's order, with their ids in the whole row", () => {
    const uniform = stream(20261118);
    let merges = 0;
    for (let round = 0; round < 60; round += 1) {
      const row = new Float32Array(1024);
      // Few distinct values: ties inside a part and across parts on every row.
      for (let id = 0; id < row.length; id += 1) row[id] = Math.floor(uniform() * 60) / 4 - 7;
      const count = 1 + Math.floor(uniform() * 24);
      const merged = leadingEntries(selectParts(row, 16, 6), count);
      const expected = ranked(row, allIds(row)).slice(0, count);
      if (merged === undefined) {
        // Only when a part held 6 of the row's first `count`.
        const perPart = new Map<number, number>();
        for (const id of expected) perPart.set(id % 16, (perPart.get(id % 16) ?? 0) + 1);
        assert(Math.max(...perPart.values()) >= 6, `round ${round}: given up without cause`);
        continue;
      }
      merges += 1;
      assertEquals([...merged.indices], expected, `round ${round}`);
      assertEquals(
        [...merged.values],
        expected.map((id) => row[id]),
        `round ${round}`,
      );
    }
    assert(merges > 30, `only ${merges} of 60 rows were merged`);
  });

  it("keeps the smaller id first on a tie, inside a part and across parts", () => {
    // Part 2 of four holds 50, 18 and 2; the other ids are in parts 3, 1 and 0.
    const row = rowOf(64, -5, [
      [50, 7],
      [3, 7],
      [17, 7],
      [18, 7],
      [2, 7],
      [40, 9],
    ]);
    assertEquals(
      [...(leadingEntries(selectParts(row, 4, 8), 6)?.indices ?? [])],
      [40, 2, 3, 17, 18, 50],
    );
  });

  it("ranks +0 and −0 as one value and a NaN above every number, the smaller id first among NaNs", () => {
    const row = rowOf(64, -1, [
      [60, Number.NaN],
      [9, Number.NaN],
      [33, 0],
      [20, -0],
      [5, Number.POSITIVE_INFINITY],
    ]);
    assertEquals(
      [...(leadingEntries(selectParts(row, 4, 8), 5)?.indices ?? [])],
      [9, 60, 5, 20, 33],
    );
  });

  it("keeps the value bits of what it returns (−0 stays −0)", () => {
    const row = rowOf(16, -1, [[6, -0]]);
    const merged = leadingEntries(selectParts(row, 4, 2), 1);
    assert(Object.is(merged?.values[0], -0));
  });

  it("holds a run of consecutive ids apart: neighbours in the vocabulary are in different parts", () => {
    // Twelve leading entries at consecutive ids, four parts of three entries each.
    const row = rowOf(
      64,
      -5,
      Array.from({ length: 12 }, (_unused, at) => [20 + at, 9 - at * 0.25] as const),
    );
    assertEquals(
      [...(leadingEntries(selectParts(row, 4, 4), 12)?.indices ?? [])],
      Array.from({ length: 12 }, (_unused, at) => 20 + at),
    );
  });

  describe("when a part may hold more of them than it returned", () => {
    // Ids 1, 5, 9 and 13 are all in part 1 of four.
    const crowded = rowOf(64, -5, [
      [1, 9],
      [5, 8],
      [9, 7],
      [13, 6],
    ]);

    it("gives up when a part's whole list was taken before the last entry: what the part did not return would be the next", () => {
      // Part 1 returned its first three (ids 1, 5, 9). The row's fourth is id 13, which the part
      // did not return: the merge would take id 0 (−5) for it.
      assertEquals(leadingEntries(selectParts(crowded, 4, 3), 4), undefined);
    });

    it("does not give up when a part's last returned entry is the last one asked for: what the part did not return ranks below it", () => {
      assertEquals([...(leadingEntries(selectParts(crowded, 4, 3), 3)?.indices ?? [])], [1, 5, 9]);
      // Over rows with crowded parts, whenever the merge answers, it is the row's ranking.
      const uniform = stream(23);
      let answered = 0;
      for (let round = 0; round < 200; round += 1) {
        const row = new Float32Array(64);
        for (let id = 0; id < 64; id += 1) row[id] = Math.floor(uniform() * 12) - 6;
        const count = 1 + Math.floor(uniform() * 12);
        const entries = leadingEntries(selectParts(row, 4, 3), count);
        if (entries === undefined) continue;
        answered += 1;
        assertEquals(
          [...entries.indices],
          ranked(row, allIds(row)).slice(0, count),
          `round ${round}`,
        );
      }
      assert(answered > 20, `only ${answered} rounds answered`);
    });

    it("does not give up while every part still has an entry it returned and was not asked for", () => {
      assertEquals([...(leadingEntries(selectParts(crowded, 4, 4), 3)?.indices ?? [])], [1, 5, 9]);
    });

    it("does not give up for a part that returned everything it has", () => {
      // Parts of four ids, four returned per part: nothing is hidden.
      const row = rowOf(16, -5, [
        [1, 9],
        [5, 8],
        [9, 7],
        [13, 6],
      ]);
      assertEquals(
        [...(leadingEntries(selectParts(row, 4, 4), 5)?.indices ?? [])],
        [1, 5, 9, 13, 0],
      );
    });
  });

  describe("on selections that are not what the GPU half promises", () => {
    // Part 0 of four leads with ids 16 (9) and 40 (7), part 3 with id 3 (8), part 1 with id 41 (6).
    const row = rowOf(64, -5, [
      [16, 9],
      [3, 8],
      [40, 7],
      [41, 6],
    ]);
    const good = selectParts(row, 4, 8);

    it("gives up and never throws", () => {
      assertEquals([...good.values.subarray(0, 2)], [9, 7]);
      assertEquals([...good.indices.subarray(0, 2)], [4, 10]);
      const withValues = (change: (values: Float32Array<ArrayBuffer>) => void): PartSelections => {
        const values = good.values.slice();
        change(values);
        return { ...good, values };
      };
      const withIndices = (change: (indices: Int32Array<ArrayBuffer>) => void): PartSelections => {
        const indices = good.indices.slice();
        change(indices);
        return { ...good, indices };
      };
      for (const [name, selections, count] of [
        ["more entries asked than selected", good, 33],
        ["no entry asked", good, 0],
        ["a count that is no integer", good, 2.5],
        ["a layout of the wrong size", { ...good, values: good.values.slice(1) }, 3],
        ["a place outside its part", withIndices((indices) => (indices[0] = 16)), 3],
        ["a negative place", withIndices((indices) => (indices[0] = -1)), 3],
        // Part 0's list starts 9 (place 4), 7 (place 10): swapped values put the lower one first.
        [
          "a part's list out of order",
          withValues((values) => {
            values[0] = 7;
            values[1] = 9;
          }),
          4,
        ],
        // Equal values with the larger place first.
        [
          "a part's tie with the larger id first",
          {
            ...withValues((values) => (values[1] = 9)),
            indices: withIndices((indices) => {
              indices[0] = 10;
              indices[1] = 4;
            }).indices,
          },
          4,
        ],
        ["a place repeated in a part", withIndices((indices) => (indices[1] = indices[0])), 4],
      ] as const) {
        assertEquals(leadingEntries(selections, count), undefined, name);
      }
      // The selections it was derived from do merge.
      assertEquals([...(leadingEntries(good, 4)?.indices ?? [])], [16, 3, 40, 41]);
    });
  });
});
