/**
 * What the session A/B page shows of a record: one row per arm (identity and the per-round times
 * from the runner's record, the rule's hits and the dispatches from the arm's count probe) and, per compared arm, the judgement with its numbers and its
 * reason in words. Pure, so `vp test` covers it.
 */

import type {
  SessionAbArmRecord,
  SessionAbJudgement,
  SessionAbVerdict,
} from "@karume/runtime/tune";

import type { SessionAbArmCounts } from "./record";

/** One arm's row. */
export type ArmRow = {
  readonly name: string;
  /** "identical (N runs)" or how many words of which run differ, and where first. */
  readonly identity: string;
  readonly identical: boolean;
  /** The rule's hits in the probe's counted run; "—" when the counts were not probed. */
  readonly hits: string;
  /** The probe's counted run's planned dispatches; "—" when the counts were not probed. */
  readonly dispatches: string;
  /** Per timed round, "visit 1 / visit 2" in ms. */
  readonly rounds: readonly string[];
};

const ms = (value: number, digits = 1): string => `${value.toFixed(digits)} ms`;

const identityText = (arm: SessionAbArmRecord): string => {
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
 * The row of one arm: its record from the runner and its probe's counts (absent when the counts
 * were not probed); `rule` names the counter the hits are read from.
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

/** The badge a verdict is shown with. */
export type VerdictTone = "default" | "destructive" | "secondary";

export const VERDICT_VIEW: Readonly<
  Record<SessionAbVerdict, { readonly label: string; readonly tone: VerdictTone }>
> = {
  adopt: { label: "Adopt", tone: "default" },
  reject: { label: "Reject", tone: "destructive" },
  inconclusive: { label: "Inconclusive", tone: "secondary" },
};

/** One compared arm's judgement as shown. */
export type JudgementView = {
  readonly arm: string;
  readonly base: string;
  readonly label: string;
  readonly tone: VerdictTone;
  /** The numbers behind the verdict, as label and value. */
  readonly numbers: readonly (readonly [string, string])[];
  readonly reason: string;
};

const signedMs = (value: number): string => `${value >= 0 ? "+" : "−"}${ms(Math.abs(value), 3)}`;

export const judgementView = (judgement: SessionAbJudgement): JudgementView => ({
  arm: judgement.arm,
  base: judgement.base,
  ...VERDICT_VIEW[judgement.verdict],
  numbers: [
    ["Identical to the base", judgement.identical ? "yes" : "no"],
    [
      `Median paired difference (${judgement.arm} − ${judgement.base})`,
      signedMs(judgement.medianDeltaMs),
    ],
    ["Base median round", ms(judgement.baseMedianMs, 3)],
    ["Gain", `${(judgement.gain * 100).toFixed(2)}%`],
    ["Noise floor (an arm against itself within a round)", ms(judgement.noiseFloorMs, 3)],
    [
      `Interval of the median difference (${(judgement.intervalCoverage * 100).toFixed(1)}% coverage)`,
      `${signedMs(judgement.deltaLowMs)} to ${signedMs(judgement.deltaHighMs)}`,
    ],
    ["Paired differences per round", judgement.deltasMs.map(signedMs).join(", ")],
  ],
  reason: judgement.reason,
});
