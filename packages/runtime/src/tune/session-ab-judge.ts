/**
 * The statistics and the verdict of a session A/B record (`karume-session-ab/1`) — pure functions
 * of the record, so a stored record can be judged again.
 *
 * Per non-base arm and timed round r: `base_r` and `arm_r` are the means of that arm's two visits in
 * the round (forward and reverse pass, so a drift linear within the round cancels), and
 * `delta_r = arm_r − base_r`. Over the rounds: `medianDeltaMs` = the median of `delta_r`,
 * `baseMedianMs` = the median of `base_r`, and `gain = −medianDeltaMs / baseMedianMs` (positive =
 * the arm is faster). The noise floor reads two scatters and takes the larger:
 *
 * - `visitSpreadMs`, how far one arm moves against itself within a round: for each arm, the median
 *   over rounds of |visit 1 − visit 2|, the larger of the base's and the compared arm's;
 * - `deltaSpreadMs`, how far the paired differences scatter across rounds: the median over rounds of
 *   |delta_r − medianDeltaMs| (an arm whose speed moves from round to round while its two visits of
 *   a round agree has a visit spread near 0, and this is what catches it);
 * - `noiseFloorMs = max(visitSpreadMs, deltaSpreadMs)`.
 *
 * A median of an even count is the mean of the two middle values. The floor is conservative on
 * purpose and does not shrink with the number of rounds: more rounds make the medians steadier, they
 * do not let a gain smaller than the jitter of one visit through.
 *
 * The verdict, per non-base arm: `"inconclusive"` when |medianDeltaMs| ≤ noiseFloorMs; otherwise
 * `"adopt"` when gain ≥ minGain; otherwise `"reject"`. Identity comes first and is required by
 * default (a class E rule, ADR 0147 decision 2): an arm whose outputs differed from the base's in any
 * run, or a base whose own runs differed, is `"reject"` whatever its times. Identity is read from
 * each arm's worst run (`identity.worst.failures` 0 means identical).
 *
 * What the verdict means: on this record's device, backend and work, the arm's end-to-end time
 * differed from the base's by more than the arms moved against themselves, by at least `minGain`.
 * What it does not mean: anything about another device, backend or input size, or that a
 * difference inside the noise floor is zero.
 */

import {
  assertSessionAbShape,
  invalidTimesMessage,
  SESSION_AB_FORMAT,
  SESSION_AB_MISMATCH_CAUSES,
  type SessionAbArmRecord,
  type SessionAbRecord,
} from "./session-ab.ts";

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
  /** The larger of the two arms' medians over rounds of |visit 1 − visit 2|. */
  readonly visitSpreadMs: number;
  /** The median over rounds of |delta_r − medianDeltaMs|. */
  readonly deltaSpreadMs: number;
  /** `max(visitSpreadMs, deltaSpreadMs)`. */
  readonly noiseFloorMs: number;
};

/** The three verdicts of {@link judgeSessionAb}. */
export type SessionAbVerdict = "adopt" | "reject" | "inconclusive";

/** One non-base arm's verdict, with the numbers behind it and a reason in words. */
export type SessionAbJudgement = SessionAbComparison & {
  readonly verdict: SessionAbVerdict;
  /**
   * Whether the arm's outputs (and the base's own runs) were u32-identical to the base's first,
   * derived from both arms' worst runs.
   */
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

/** A field of a value read from a stored record; `undefined` when the value is not an object. */
const fieldOf = (value: unknown, name: string): unknown =>
  typeof value === "object" && value !== null && Object.hasOwn(value, name)
    ? Reflect.get(value, name)
    : undefined;

const isCount = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value >= 0;

const isWord = (value: unknown): boolean => isCount(value) && value <= 0xffffffff;

/** Whether a stored identity block holds what the judge reads (the record's types say it does). */
const validIdentity = (identity: unknown, rounds: number): boolean => {
  const worst = fieldOf(identity, "worst");
  const runs = fieldOf(identity, "runs");
  const round = fieldOf(worst, "round");
  const visit = fieldOf(worst, "visit");
  const run = fieldOf(worst, "run");
  const failures = fieldOf(worst, "failures");
  const first = fieldOf(worst, "first");
  const shaped =
    isCount(runs) &&
    runs > 0 &&
    isCount(round) &&
    round <= rounds &&
    (visit === 1 || visit === 2) &&
    (run === 1 || run === 2) &&
    isCount(fieldOf(worst, "words")) &&
    isCount(failures);
  if (!shaped) return false;
  if (failures === 0) return first === undefined;
  const cause = fieldOf(first, "cause");
  return (
    typeof fieldOf(first, "output") === "string" &&
    SESSION_AB_MISMATCH_CAUSES.some((known) => known === cause) &&
    isCount(fieldOf(first, "index")) &&
    isWord(fieldOf(first, "expected")) &&
    isWord(fieldOf(first, "actual"))
  );
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
    if (!valid) throw new RangeError(invalidTimesMessage(arm.name, record.rounds));
    if (!validIdentity(arm.identity, record.rounds)) {
      throw new RangeError(
        `session A/B: arm "${arm.name}" does not hold a valid identity block (a positive run count` +
          ` and a worst run with its round, visit, run, words, failures and, when failures > 0, its` +
          ` first mismatch)`,
      );
    }
  }
};

/** Whether an arm's every run was u32-identical to the base's first outputs. */
const identicalArm = (arm: SessionAbArmRecord): boolean => arm.identity.worst.failures === 0;

const roundMeans = (times: readonly (readonly [number, number])[]): number[] =>
  times.map(([first, second]) => (first + second) / 2);

const selfSpread = (times: readonly (readonly [number, number])[]): number =>
  median(times.map(([first, second]) => Math.abs(first - second)));

/**
 * The statistics of every non-base arm against the base (the first arm), in the record's order.
 * Refuses (`RangeError`) a record of another format, fewer than two arms, a name used twice, rounds
 * that are not an integer of at least 3, times that are not two finite non-negative numbers per
 * round, an identity block that does not hold what the judge reads, and a base whose median round
 * time is not positive.
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
    const visitSpreadMs = Math.max(baseSpread, selfSpread(arm.timesMs));
    const deltaSpreadMs = median(deltasMs.map((delta) => Math.abs(delta - medianDeltaMs)));
    return {
      arm: arm.name,
      base: base.name,
      deltasMs,
      medianDeltaMs,
      baseMedianMs,
      gain: -medianDeltaMs / baseMedianMs,
      visitSpreadMs,
      deltaSpreadMs,
      noiseFloorMs: Math.max(visitSpreadMs, deltaSpreadMs),
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
    const identical = identicalArm(base) && identicalArm(arm);
    if (requireIdentity && !identical) {
      const failing = identicalArm(arm) ? base : arm;
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
          ` round ${worst.round}, visit ${worst.visit}, run ${worst.run}${where}`,
      };
    }
    const { medianDeltaMs, noiseFloorMs, gain } = comparison;
    const numbers =
      `median delta ${ms(medianDeltaMs)}, noise floor ${ms(noiseFloorMs)}` +
      ` (visit spread ${ms(comparison.visitSpreadMs)}, delta spread` +
      ` ${ms(comparison.deltaSpreadMs)}), gain ${percent(gain)} of ${ms(comparison.baseMedianMs)}`;
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
