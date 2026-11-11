/**
 * The statistics and the verdict of a session A/B record (`karume-session-ab/1`) — pure functions
 * of the record, so a stored record can be judged again.
 *
 * Per non-base arm and timed round r: `base_r` and `arm_r` are the means of that arm's two visits in
 * the round (forward and reverse pass, so a drift linear within the round cancels), and
 * `delta_r = arm_r − base_r`. Over the rounds: `medianDeltaMs` = the median of `delta_r`,
 * `baseMedianMs` = the median of `base_r`, and `gain = −medianDeltaMs / baseMedianMs` (positive =
 * the arm is faster). The noise floor reads how far one arm moves against itself: for each arm, the
 * median over rounds of |visit 1 − visit 2|; `noiseFloorMs` is the larger of the base's and the
 * compared arm's. A median of an even count is the mean of the two middle values.
 *
 * The verdict, per non-base arm: `"inconclusive"` when |medianDeltaMs| ≤ noiseFloorMs; otherwise
 * `"adopt"` when gain ≥ minGain; otherwise `"reject"`. Identity comes first and is required by
 * default (a class E rule, ADR 0147 decision 2): an arm whose outputs differed from the base's in any
 * visit, or a base whose own runs differed, is `"reject"` whatever its times.
 *
 * What the verdict means: on this record's device, backend and work, the arm's end-to-end time
 * differed from the base's by more than the arms moved against themselves, by at least `minGain`.
 * What it does not mean: anything about another device, backend or input size, or that a
 * difference inside the noise floor is zero.
 */

import { assertSessionAbShape, SESSION_AB_FORMAT, type SessionAbRecord } from "./session-ab.ts";

/** The numbers behind one non-base arm's verdict. */
export type SessionAbComparison = {
  /** The compared arm's name. */
  readonly arm: string;
  /** The base arm's name. */
  readonly base: string;
  /** `delta_r = arm_r − base_r` per timed round, in round order. */
  readonly deltasMs: readonly number[];
  readonly medianDeltaMs: number;
  readonly baseMedianMs: number;
  /** `−medianDeltaMs / baseMedianMs`; positive when the arm is faster. */
  readonly gain: number;
  /** The larger of the two arms' medians of |visit 1 − visit 2|. */
  readonly noiseFloorMs: number;
};

/** The three verdicts of {@link judgeSessionAb}. */
export type SessionAbVerdict = "adopt" | "reject" | "inconclusive";

/** One non-base arm's verdict, with the numbers behind it and a reason in words. */
export type SessionAbJudgement = SessionAbComparison & {
  readonly verdict: SessionAbVerdict;
  /** Whether the arm's outputs (and the base's own runs) were u32-identical to the base's first. */
  readonly identical: boolean;
  readonly reason: string;
};

/** The options of {@link judgeSessionAb}. */
export type JudgeSessionAbOptions = {
  /** The smallest gain adopted, a finite non-negative fraction (0.01 = 1% of the base's time). */
  readonly minGain: number;
  /**
   * Whether an arm must be u32-identical to the base (default `true`, a class E rule). `false` for a
   * rule of another class, whose numbers are checked by other means.
   */
  readonly requireIdentity?: boolean;
};

const median = (values: readonly number[]): number => {
  const sorted = values.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const assertRecord = (record: SessionAbRecord): void => {
  if (record.format !== SESSION_AB_FORMAT) {
    throw new RangeError(
      `session A/B: the record's format is ${JSON.stringify(record.format)}, not "${SESSION_AB_FORMAT}"`,
    );
  }
  assertSessionAbShape(
    record.arms.map((arm) => arm.name),
    record.rounds,
  );
  for (const arm of record.arms) {
    const valid =
      arm.timesMs.length === record.rounds &&
      arm.timesMs.every(
        (visits) =>
          visits.length === 2 && visits.every((time) => Number.isFinite(time) && time >= 0),
      );
    if (!valid) {
      throw new RangeError(
        `session A/B: arm "${arm.name}" does not hold two finite non-negative times for each of the` +
          ` ${record.rounds} rounds`,
      );
    }
  }
};

const roundMeans = (times: readonly (readonly [number, number])[]): number[] =>
  times.map(([first, second]) => (first + second) / 2);

const selfSpread = (times: readonly (readonly [number, number])[]): number =>
  median(times.map(([first, second]) => Math.abs(first - second)));

/**
 * The statistics of every non-base arm against the base (the first arm), in the record's order.
 * Refuses (`RangeError`) a record of another format, fewer than two arms, a name used twice, rounds
 * that are not an integer of at least 3, times that are not two finite non-negative numbers per
 * round, and a base whose median round time is not positive.
 */
export const sessionAbStatistics = (record: SessionAbRecord): SessionAbComparison[] => {
  assertRecord(record);
  const [base, ...others] = record.arms;
  const baseMeans = roundMeans(base.timesMs);
  const baseMedianMs = median(baseMeans);
  if (!(baseMedianMs > 0)) {
    throw new RangeError(
      `session A/B: the base arm "${base.name}" has a median round time of ${baseMedianMs} ms`,
    );
  }
  const baseSpread = selfSpread(base.timesMs);
  return others.map((arm) => {
    const deltasMs = roundMeans(arm.timesMs).map((mean, round) => mean - baseMeans[round]);
    const medianDeltaMs = median(deltasMs);
    return {
      arm: arm.name,
      base: base.name,
      deltasMs,
      medianDeltaMs,
      baseMedianMs,
      gain: -medianDeltaMs / baseMedianMs,
      noiseFloorMs: Math.max(baseSpread, selfSpread(arm.timesMs)),
    };
  });
};

const ms = (value: number): string => `${value.toFixed(3)} ms`;
const percent = (value: number): string => `${(value * 100).toFixed(2)}%`;

/**
 * The verdict of every non-base arm, in the record's order (see the module doc for the rules).
 * Refuses (`RangeError`) a `minGain` that is not a finite non-negative number, and every record
 * {@link sessionAbStatistics} refuses.
 */
export const judgeSessionAb = (
  record: SessionAbRecord,
  options: JudgeSessionAbOptions,
): SessionAbJudgement[] => {
  const { minGain } = options;
  if (!Number.isFinite(minGain) || minGain < 0) {
    throw new RangeError(
      `session A/B: minGain must be a finite non-negative number (got ${minGain})`,
    );
  }
  const requireIdentity = options.requireIdentity ?? true;
  const comparisons = sessionAbStatistics(record);
  const [base, ...others] = record.arms;
  return comparisons.map((comparison, index) => {
    const arm = others[index];
    const identical = base.identity.identical && arm.identity.identical;
    if (requireIdentity && !identical) {
      const failing = arm.identity.identical ? base : arm;
      const { worst } = failing.identity;
      const where =
        worst.first === undefined
          ? ""
          : ` (first at output ${worst.first.output}[${worst.first.index}], ${worst.first.cause})`;
      const whose =
        failing === arm
          ? `arm "${arm.name}" differs from the base's outputs`
          : `the base arm "${base.name}" differs from its own first outputs`;
      return {
        ...comparison,
        verdict: "reject",
        identical,
        reason:
          `identity is required and ${whose}: ${worst.failures} of ${worst.words} words fail in` +
          ` round ${worst.round}, visit ${worst.visit}${where}`,
      };
    }
    const { medianDeltaMs, noiseFloorMs, gain } = comparison;
    const numbers =
      `median delta ${ms(medianDeltaMs)}, noise floor ${ms(noiseFloorMs)}, gain ${percent(gain)}` +
      ` of ${ms(comparison.baseMedianMs)}`;
    if (Math.abs(medianDeltaMs) <= noiseFloorMs) {
      return {
        ...comparison,
        verdict: "inconclusive",
        identical,
        reason: `the difference is inside the noise floor (${numbers})`,
      };
    }
    if (gain >= minGain) {
      return {
        ...comparison,
        verdict: "adopt",
        identical,
        reason: `faster by at least the minimum gain ${percent(minGain)} (${numbers})`,
      };
    }
    return {
      ...comparison,
      verdict: "reject",
      identical,
      reason:
        gain > 0
          ? `faster by less than the minimum gain ${percent(minGain)} (${numbers})`
          : `slower than the base (${numbers})`,
    };
  });
};
