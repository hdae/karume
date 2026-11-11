/**
 * Hand-written `karume-session-ab/1` records for the session A/B page's tests: two arms, `base` and
 * `rule`, with the times, counts and identity each case needs. Plain data, as a stored record is.
 */

import type { SessionAbArmRecord, SessionAbRecord } from "@karume/runtime/tune";

export const RULE = "candidate/gelu_mul";

type Times = readonly (readonly [number, number])[];

const identical = (runs: number): SessionAbArmRecord["identity"] => ({
  runs,
  worst: { round: 0, visit: 1, run: 1, words: 96, failures: 0 },
});

/** The rule arm's tokens differ from round 2 on: 3 of 96 words, first at index 41. */
export const DIFFERING: SessionAbArmRecord["identity"] = {
  runs: 20,
  worst: {
    round: 2,
    visit: 2,
    run: 2,
    words: 96,
    failures: 3,
    first: { output: "tokens", cause: "word", index: 41, expected: 100, actual: 200 },
  },
};

export const abRecord = (input: {
  readonly base: Times;
  readonly rule: Times;
  readonly ruleIdentity?: SessionAbArmRecord["identity"];
  /** Leave the counts out of both arms (a run without `readCounts`). */
  readonly noCounts?: boolean;
}): SessionAbRecord => ({
  format: "karume-session-ab/1",
  environment: {
    adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
    userAgent: "Mozilla/5.0 Chrome/150.0.0.0",
    backend: "d3d12",
    date: "2026-11-11T12:00:00.000Z",
  },
  rounds: input.base.length,
  openPerVisit: true,
  arms: [
    {
      name: "base",
      ...(input.noCounts === true ? {} : { fusions: { rmsNormAdd: 30 }, dispatches: 1294 }),
      timesMs: input.base,
      identity: identical(20),
    },
    {
      name: "rule",
      ...(input.noCounts === true
        ? {}
        : { fusions: { rmsNormAdd: 30, [RULE]: 60 }, dispatches: 1234 }),
      timesMs: input.rule,
      identity: input.ruleIdentity ?? identical(20),
    },
  ],
});

/** Three rounds of one steady time per arm (visit 1 and 2 a little apart). */
export const steady = (ms: number): Times => [
  [ms, ms + 0.1],
  [ms + 0.1, ms],
  [ms, ms + 0.1],
];
