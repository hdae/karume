/**
 * What a reader sees of a session A/B record: one row per arm (identity and the per-round times
 * from the runner's record, the rule's hits and the dispatches from the arm's counts) and, per
 * compared arm, the judgement with its numbers and its reason in words. Pure, so `vp test` covers
 * it.
 */

import type {
  SessionAbArmRecord,
  SessionAbJudgement,
  SessionAbVerdict,
} from "@karume/runtime/tune";

import type { SessionAbArmCounts } from "./data.ts";

/** One arm's row. */
export type ArmRow = {
  readonly name: string;
  /** "identical (N runs)" or how many words of which run differ, and where first. */
  readonly identity: string;
  readonly identical: boolean;
  /** The rule's hits in the counted run; "—" when the counts were not probed. */
  readonly hits: string;
  /** The counted run's dispatches; "—" when the counts were not probed. */
  readonly dispatches: string;
  /** Per timed round, "visit 1 / visit 2" in ms. */
  readonly rounds: readonly string[];
};

const ms = (value: number, digits = 1): string => `${value.toFixed(digits)} ms`;

/** "identical (N runs)", or how many words of which run differ from the base's first, and where first. */
export const identityText = (arm: SessionAbArmRecord): string => {
  const { runs, worst } = arm.identity;
  if (worst.failures === 0) return `identical (${runs} runs)`;
  const where = `round ${worst.round}, visit ${worst.visit}, run ${worst.run}`;
  const first =
    worst.first === undefined
      ? ""
      : `; first at ${worst.first.output}[${worst.first.index}] (${worst.first.cause}): expected ${worst.first.expected}, got ${worst.first.actual}`;
  return `${worst.failures} of ${worst.words} words differ (${where}${first})`;
};

/**
 * The row of one arm: its record from the runner and its counts (absent when the counts were not
 * probed); `rule` names the counter the hits are read from.
 */
export const armRow = (
  arm: SessionAbArmRecord,
  counts: SessionAbArmCounts | undefined,
  rule: string,
): ArmRow => ({
  name: arm.name,
  identity: identityText(arm),
  identical: arm.identity.worst.failures === 0,
  hits: counts === undefined ? "—" : String(counts.fusions[rule] ?? 0),
  dispatches: counts === undefined ? "—" : String(counts.dispatches),
  rounds: arm.timesMs.map(([first, second]) => `${ms(first)} / ${ms(second)}`),
});

/** The word a verdict is shown with. */
export const VERDICT_LABELS: Readonly<Record<SessionAbVerdict, string>> = {
  adopt: "Adopt",
  reject: "Reject",
  inconclusive: "Inconclusive",
};

/** One compared arm's judgement as shown. */
export type JudgementView = {
  readonly arm: string;
  readonly base: string;
  readonly label: string;
  /** The numbers behind the verdict, as label and value. */
  readonly numbers: readonly (readonly [string, string])[];
  readonly reason: string;
};

const signedMs = (value: number): string => `${value >= 0 ? "+" : "−"}${ms(Math.abs(value), 3)}`;

const gainText = (judgement: SessionAbJudgement): string => `${(judgement.gain * 100).toFixed(2)}%`;

const intervalLabel = (judgement: SessionAbJudgement): string =>
  `Interval of the median difference (${(judgement.intervalCoverage * 100).toFixed(1)}% coverage)`;

const intervalText = (judgement: SessionAbJudgement): string =>
  `${signedMs(judgement.deltaLowMs)} to ${signedMs(judgement.deltaHighMs)}`;

export const judgementView = (judgement: SessionAbJudgement): JudgementView => ({
  arm: judgement.arm,
  base: judgement.base,
  label: VERDICT_LABELS[judgement.verdict],
  numbers: [
    ["Identical to the base", judgement.identical ? "yes" : "no"],
    [
      `Median paired difference (${judgement.arm} − ${judgement.base})`,
      signedMs(judgement.medianDeltaMs),
    ],
    ["Base median round", ms(judgement.baseMedianMs, 3)],
    ["Gain", gainText(judgement)],
    ["Noise floor (an arm against itself within a round)", ms(judgement.noiseFloorMs, 3)],
    [intervalLabel(judgement), intervalText(judgement)],
    ["Paired differences per round", judgement.deltasMs.map(signedMs).join(", ")],
  ],
  reason: judgement.reason,
});

/**
 * One compared arm's judgement in one line: the verdict, the gain and the interval of the median
 * difference, formatted as {@link judgementView} shows them.
 */
export const judgementLine = (judgement: SessionAbJudgement): string =>
  `${judgement.arm} against ${judgement.base}: ${VERDICT_LABELS[judgement.verdict]} · gain ${gainText(judgement)} · ${intervalLabel(judgement).toLowerCase()} ${intervalText(judgement)}`;

/**
 * An arm's counts as the runner's record keeps them (a target whose arms report their counts to
 * the runner, the `sessionArm`s of the Session targets); `undefined` when the arm reported none.
 */
export const recordedCounts = (arm: SessionAbArmRecord): SessionAbArmCounts | undefined =>
  arm.fusions === undefined || arm.dispatches === undefined
    ? undefined
    : { arm: arm.name, fusions: arm.fusions, dispatches: arm.dispatches };
