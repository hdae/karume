import { describe, expect, it } from "vite-plus/test";

import { durationText, PRIOR_SECONDS, remainingEstimate } from "../src/estimate.ts";
import type { CaseKind } from "../src/plan.ts";

/** Two GEMV sentinels, two other GEMV cases, then a geometry class of three cases. */
const KINDS: readonly CaseKind[] = [
  "sentinel",
  "sentinel",
  "other",
  "other",
  "first-of-class",
  "later",
  "later",
];

const priorsOf = (kinds: readonly CaseKind[]): number =>
  kinds.reduce((sum, kind) => sum + PRIOR_SECONDS[kind], 0);

describe("remainingEstimate", () => {
  it("before anything is measured, sums the stated priors and the idle time before every case but the first", () => {
    const estimate = remainingEstimate(KINDS, { finished: new Map() }, 10);
    expect(estimate).toEqual({ seconds: priorsOf(KINDS) + 6 * 10, measured: false });
  });

  it("takes a kind's time from its finished cases: the mean of the same kind (fault: the prior kept)", () => {
    const finished = new Map([
      [0, 40],
      [1, 20],
    ]);
    const { seconds, measured } = remainingEstimate(KINDS, { finished }, 0);
    expect(measured).toBe(true);
    // The other GEMV kind is scaled by the sentinels' measured ÷ prior; the geometry kinds keep their priors
    const scale = 60 / (2 * PRIOR_SECONDS.sentinel);
    expect(seconds).toBeCloseTo(
      2 * PRIOR_SECONDS.other * scale + PRIOR_SECONDS["first-of-class"] + 2 * PRIOR_SECONDS.later,
      6,
    );
  });

  it("reads a kind with finished cases by their mean alone, whatever the other kind of the sweep took", () => {
    const finished = new Map([
      [0, 30],
      [1, 30],
      [2, 5],
      [3, 7],
      [4, 100],
    ]);
    const { seconds } = remainingEstimate(KINDS, { finished }, 0);
    // later: scaled by the first case of the class (100 ÷ its prior)
    expect(seconds).toBeCloseTo(
      2 * PRIOR_SECONDS.later * (100 / PRIOR_SECONDS["first-of-class"]),
      6,
    );
  });

  it("counts what the running case has left of its expectation, never below 0", () => {
    const finished = new Map([[0, 30]]);
    const halfway = remainingEstimate(KINDS, { finished, running: { index: 1, seconds: 10 } }, 0);
    const over = remainingEstimate(KINDS, { finished, running: { index: 1, seconds: 500 } }, 0);
    const after = remainingEstimate(
      KINDS,
      {
        finished: new Map([
          [0, 30],
          [1, 30],
        ]),
      },
      0,
    );
    expect(halfway.seconds - after.seconds).toBeCloseTo(20, 6);
    expect(over.seconds).toBeCloseTo(after.seconds, 6);
  });

  it("counts the idle wait's rest once, and the idle before every later case", () => {
    const finished = new Map([[0, 30]]);
    const idling = remainingEstimate(KINDS, { finished, idling: { index: 1, seconds: 4 } }, 10);
    const sentinel = 30;
    const scale = 30 / PRIOR_SECONDS.sentinel;
    expect(idling.seconds).toBeCloseTo(
      4 +
        sentinel +
        2 * PRIOR_SECONDS.other * scale +
        PRIOR_SECONDS["first-of-class"] +
        2 * PRIOR_SECONDS.later +
        5 * 10,
      6,
    );
  });

  it("is 0 when every case finished", () => {
    const finished = new Map(KINDS.map((_kind, index) => [index, 1]));
    expect(remainingEstimate(KINDS, { finished }, 10).seconds).toBe(0);
  });
});

describe("durationText", () => {
  it("says a duration as a person reads an estimate", () => {
    expect(durationText(20)).toBe("under a minute");
    expect(durationText(12 * 60 + 10)).toBe("about 12 min");
    expect(durationText(60 * 60)).toBe("about 1 h");
    expect(durationText(99 * 60 + 40)).toBe("about 1 h 40 min");
  });
});
