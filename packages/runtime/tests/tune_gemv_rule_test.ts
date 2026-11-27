// The numbers of the GEMV derivation's rule of one steady record and its pure pieces
// (src/tune/gemv-rule.ts): a record's class from its summaries, a row's gap, a gain's break-even
// margin and the margin a derivation takes.
//
// Faults injected: 19 groups with a spread (no class); a share of 21 % beyond ×1.05 (unsteady);
// groups without a spread counted (they are not); a spread of exactly ×1.05 counted as beyond it
// (it is not); a clamped round of 0 taken as the smallest (it is left out); one positive round (no
// gap); a margin below 1, NaN and infinity (refused).

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  GEMV_LOWEST_MARGIN,
  GEMV_MARGIN,
  GEMV_MIN_PASS_MS,
  GEMV_STEADY_GROUP_SPREAD,
  GEMV_STEADY_MAX_SHARE,
  GEMV_STEADY_MIN_GROUPS,
  gemvBreakEven,
  gemvRoundsGap,
  gemvSteadiness,
} from "../tune.ts";
import { gemvMarginOf } from "../src/tune/gemv-rule.ts";

/** `steady` groups at ×1.01 and `unsteady` ones at ×1.2. */
const summaries = (steady: number, unsteady: number) => [
  ...Array.from({ length: steady }, () => ({ baselineSpread: 1.01 })),
  ...Array.from({ length: unsteady }, () => ({ baselineSpread: 1.2 })),
];

describe("the rule's numbers", () => {
  it("are the maintainer's of 2026-11-28: ×1.065, ×1.03 the lowest, 20 groups, 20 %, ×1.05, 20 ms", () => {
    assertEquals(
      [
        GEMV_MARGIN,
        GEMV_LOWEST_MARGIN,
        GEMV_STEADY_MIN_GROUPS,
        GEMV_STEADY_MAX_SHARE,
        GEMV_STEADY_GROUP_SPREAD,
        GEMV_MIN_PASS_MS,
      ],
      [1.065, 1.03, 20, 0.2, 1.05, 20],
    );
  });
});

describe("a record's class (gemvSteadiness)", () => {
  it("is steady with 20 groups or more of which at most 20 % spread beyond ×1.05", () => {
    assertEquals(gemvSteadiness(summaries(20, 0)), {
      class: "steady",
      groups: 20,
      unsteadyGroups: 0,
      share: 0,
    });
    assertEquals(gemvSteadiness(summaries(16, 4)), {
      class: "steady",
      groups: 20,
      unsteadyGroups: 4,
      share: 0.2,
    });
  });

  it("is unsteady past 20 % (fault: 21 of 100)", () => {
    assertEquals(gemvSteadiness(summaries(79, 21)).class, "unsteady");
    assertEquals(gemvSteadiness(summaries(80, 20)).class, "steady");
  });

  it("has no class with fewer than 20 groups that have a spread, whatever their share (fault: 19)", () => {
    assertEquals(gemvSteadiness(summaries(19, 0)), {
      class: "unclassified",
      groups: 19,
      unsteadyGroups: 0,
      share: 0,
    });
    assertEquals(gemvSteadiness(summaries(0, 5)).class, "unclassified");
    assertEquals(gemvSteadiness([]), { class: "unclassified", groups: 0, unsteadyGroups: 0 });
  });

  it("counts only the groups with a spread, and a spread of ×1.05 itself as steady (faults: 30 groups without a spread; ×1.05 taken as beyond)", () => {
    const withoutSpread = Array.from({ length: 30 }, () => ({}));
    assertEquals(gemvSteadiness([...withoutSpread, ...summaries(10, 0)]).class, "unclassified");
    const atBound = Array.from({ length: 20 }, () => ({
      baselineSpread: GEMV_STEADY_GROUP_SPREAD,
    }));
    assertEquals(gemvSteadiness(atBound).unsteadyGroups, 0);
  });
});

describe("a row's gap (gemvRoundsGap) and a gain's break-even margin (gemvBreakEven)", () => {
  it("is the second-smallest positive round ÷ the smallest, rounds clamped to 0 left out (fault: a 0 round)", () => {
    assertEquals(gemvRoundsGap([110, 100, 105, 130, 120]), 1.05);
    assertEquals(gemvRoundsGap([0, 100, 0, 102]), 1.02);
    assertEquals(gemvRoundsGap([100, 100]), 1);
  });

  it("is undefined with fewer than two positive rounds (fault: one)", () => {
    assertEquals(gemvRoundsGap([100]), undefined);
    assertEquals(gemvRoundsGap([0, 100]), undefined);
    assertEquals(gemvRoundsGap(undefined), undefined);
  });

  it("is the gain ÷ the two gaps: the largest margin at which the gain clears its bar", () => {
    const breakEven = gemvBreakEven(1.2, 1.02, 1.01);
    assertEquals(breakEven, 1.2 / (1.02 * 1.01));
    // At that margin the bar is the gain itself
    assertEquals(Math.abs(breakEven * 1.02 * 1.01 - 1.2) < 1e-12, true);
  });
});

describe("the margin a derivation takes (gemvMarginOf)", () => {
  it("is ×1.065 by default, and any finite number of at least 1 given", () => {
    assertEquals(gemvMarginOf(undefined), GEMV_MARGIN);
    assertEquals(gemvMarginOf(1.03), 1.03);
    assertEquals(gemvMarginOf(1), 1);
  });

  it("refuses a margin below 1, NaN and infinity (faults: each)", () => {
    for (const margin of [0.99, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => gemvMarginOf(margin), Error, "at least 1");
    }
  });
});
