/**
 * The estimate of the time a run has left. Pure: the page calls it on every frame with what the run
 * has done so far.
 *
 * A case is estimated by its kind (`plan.ts`'s `CaseKind`): the mean of the finished cases of the
 * same kind in this run; before one finished, the stated prior of its kind ({@link PRIOR_SECONDS}),
 * scaled by how this machine fared against the priors on the finished cases of the same sweep (the
 * other kind) when there are any. Every case but the run's first has the idle time before it.
 *
 * SPEC: docs/spec/benchmark.md#the-time-left
 */

import type { CaseKind, SweepId } from "./plan.ts";

/**
 * The seconds a case of each kind takes before any case of its kind finished, device acquisition and
 * the case's buffers included.
 *
 * Source (records of the maintainer's machines): a GEMV sentinel is measured with every candidate,
 * which the two `full` GEMV sweep records of an Apple M5 of 2026-11-23 time at 26 s per case in its
 * row counts, plus 1.4 s per case besides; the explore set halves the time in the row counts over the
 * whole table (its replay on the records of 2026-11-18/19), which leaves 14 s per other case. A
 * geometry case first of its class measures every candidate of `full`, which the two `full` geometry
 * sweep records of an RTX 5070 Ti under Chrome of 2026-11-18 time at 58 s per such case; the explore
 * set takes an estimated 58 % less time in the cases (2,134 of 5,082 s), which leaves 19 s per later
 * case. The RTX 3080 Ti's explore run of the `i8a8-conv3d` cases (2026-11-26) took 65 s for the first
 * case and 14 to 23 s for each later one. The GEMV priors were 27 and 15 s (27.4 and 15.4 before
 * rounding). Since 2026-11-28 each group re-times its references once more after the rounds (about
 * 6 to 7 % more passes than the rounds take on the explore records of 2026-11-27 on a device without
 * a table, 8.5 to 8.9 % on one with an embedded table, whose choices are re-timed too) and M = 1
 * carries candidates down to ×1.03 (1.4 to 3.9 % more timed rows there) — about 4 % more of a GEMV
 * case, 28.5 and 16.0 s; and the explore set predicts every sentinel row ×1.03 faster than the
 * default's choice, not ×1.10, which adds 0.6 to 7.5 % to the other cases' timed passes on the four
 * devices' records (2.8 % on average), while a device's embedded table adds 1.3 to 4.8 % to the whole
 * set's (its choices timed and re-timed besides): over the four devices with and without their table
 * (outputs/misc/gemv-rule-2026-11-28/fix/explore-plan.txt), 28.9 and 16.7 s — 29 and 17 s.
 */
export const PRIOR_SECONDS: Readonly<Record<CaseKind, number>> = {
  sentinel: 29,
  other: 17,
  "first-of-class": 58,
  later: 19,
};

/** The sweep each kind belongs to. */
const SWEEP_OF: Readonly<Record<CaseKind, SweepId>> = {
  sentinel: "gemv",
  other: "gemv",
  "first-of-class": "geometry",
  later: "geometry",
};

/** Where a run stands, as the estimate reads it. */
export type EstimateState = {
  /** The seconds each finished case took, by its index in the run. */
  readonly finished: ReadonlyMap<number, number>;
  /** The case being measured: its index in the run and its seconds so far. */
  readonly running?: { readonly index: number; readonly seconds: number };
  /** The idle wait before the case at `index`: the seconds it has left. */
  readonly idling?: { readonly index: number; readonly seconds: number };
};

export type Estimate = {
  /** The seconds the run has left. */
  readonly seconds: number;
  /** Whether a case of the run finished (otherwise the estimate is the priors alone). */
  readonly measured: boolean;
};

/** The seconds a case of `kind` is expected to take, from the finished cases of the run. */
const kindSeconds = (
  kind: CaseKind,
  kinds: readonly CaseKind[],
  finished: ReadonlyMap<number, number>,
): number => {
  const done = [...finished].map(([index, seconds]) => ({ kind: kinds[index], seconds }));
  const same = done.filter((entry) => entry.kind === kind);
  if (same.length > 0) return same.reduce((sum, entry) => sum + entry.seconds, 0) / same.length;
  const sweep = done.filter((entry) => SWEEP_OF[entry.kind] === SWEEP_OF[kind]);
  if (sweep.length === 0) return PRIOR_SECONDS[kind];
  const measured = sweep.reduce((sum, entry) => sum + entry.seconds, 0);
  const prior = sweep.reduce((sum, entry) => sum + PRIOR_SECONDS[entry.kind], 0);
  return PRIOR_SECONDS[kind] * (measured / prior);
};

/**
 * The time a run of the cases `kinds` (in run order) has left at `state`, with `idleSeconds` before
 * every case but the first: what the running case has left of its expectation (never below 0), the
 * idle wait's rest, and every case after it with the idle time before it.
 */
export const remainingEstimate = (
  kinds: readonly CaseKind[],
  state: EstimateState,
  idleSeconds: number,
): Estimate => {
  const { finished, running, idling } = state;
  const expected = (index: number): number => kindSeconds(kinds[index], kinds, finished);
  const lastFinished = Math.max(-1, ...finished.keys());
  const next =
    running !== undefined
      ? running.index + 1
      : idling !== undefined
        ? idling.index
        : lastFinished + 1;
  let seconds = running === undefined ? 0 : Math.max(0, expected(running.index) - running.seconds);
  if (idling !== undefined) seconds += Math.max(0, idling.seconds);
  for (let index = next; index < kinds.length; index += 1) {
    seconds += expected(index);
    if (index > 0 && index !== idling?.index) seconds += idleSeconds;
  }
  return { seconds, measured: finished.size > 0 };
};

/** A duration as a person reads an estimate: `under a minute`, `about 12 min`, `about 1 h 5 min`. */
export const durationText = (seconds: number): string => {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return "under a minute";
  if (minutes < 60) return `about ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `about ${hours} h${rest === 0 ? "" : ` ${rest} min`}`;
};
