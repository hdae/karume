// Pins the binary search over closed-range tables (`src/text/code-ranges.ts`) on its own. Needs
// neither a GPU nor real assets.
//
// Why it is pinned on its own: this one function is **the only classifier** of characters for anima
// (Qwen2 / T5 / spm) and sbv2 (DeBERTa), and a miss gives "a different character class" without an
// exception — the pre-token boundaries change and the id sequence quietly becomes something else,
// so from the families' e2e tests it stays green even when broken.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { type CodeRanges, inCodeRanges } from "../src/text/code-ranges.ts";

test("inCodeRanges: includes both ends of a range and excludes the value just outside", () => {
  const ranges: CodeRanges = [[0x41, 0x5a]];
  assertEquals(inCodeRanges(ranges, 0x40), false);
  assertEquals(inCodeRanges(ranges, 0x41), true);
  assertEquals(inCodeRanges(ranges, 0x4d), true);
  assertEquals(inCodeRanges(ranges, 0x5a), true);
  assertEquals(inCodeRanges(ranges, 0x5b), false);
});

test("inCodeRanges: does not miss single-point ranges or the ranges at both ends of the table", () => {
  // For the first / last range the binary search narrows towards one side, so they are checked
  // separately from the ranges in between.
  const ranges: CodeRanges = [
    [0, 0],
    [0x10, 0x12],
    [0x10ffff, 0x10ffff],
  ];
  assertEquals(inCodeRanges(ranges, 0), true);
  assertEquals(inCodeRanges(ranges, 1), false);
  assertEquals(inCodeRanges(ranges, 0x11), true);
  assertEquals(inCodeRanges(ranges, 0x10fffe), false);
  assertEquals(inCodeRanges(ranges, 0x10ffff), true);
});

test("inCodeRanges: an empty table contains nothing (a missing table is not something to notice by a pass)", () => {
  assertEquals(inCodeRanges([], 0), false);
  assertEquals(inCodeRanges([], 0x41), false);
});

test("inCodeRanges: matches a linear scan for every range count", () => {
  // The binary search's branches (even / odd length, the truncation of mid) are swept through the
  // table length, with a naive scan as the answer. The ranges are [4k, 4k+1], so the gaps (4k+2 /
  // 4k+3) are covered at the same time.
  for (let count = 1; count <= 16; count++) {
    const ranges: CodeRanges = Array.from({ length: count }, (_, k) => [4 * k, 4 * k + 1]);
    const naive = (cp: number): boolean => ranges.some(([start, end]) => cp >= start && cp <= end);
    for (let cp = 0; cp < 4 * count + 2; cp++) {
      assertEquals(inCodeRanges(ranges, cp), naive(cp), `count=${count} cp=${cp}`);
    }
  }
});
