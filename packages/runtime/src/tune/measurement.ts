/**
 * Constants of the micro-benchmark measurement protocol and the repetition estimate (shared by the
 * sweep's measurement core `harness.ts` and the in-repository tool `tools/opbench` (removed in
 * 0.15) — moved here from the tool by ADR 0117 decision 1. This is the only definition).
 *
 * Source of the protocol = docs/research/2026-08-10-kernel-variant-sweep.md §5:
 * 1. Stack the same dispatch until one timed pass is ≈80ms (keeps the duty cycle up so the clocks
 *    stay pinned — measured from idle, the same variant fluctuated by 2x)
 * 2. The representative value is the **min** (the min absorbs thermal drift)
 * 3. Never compare absolute ms across days or machines — only ratios within one rig are valid
 */

/**
 * Target length of one timed pass (ms). The default is the measured value of research §5-1 as is.
 */
export const TARGET_PASS_MS = 80;

/**
 * Lower bound of the idle runs that pin the clocks before measuring (cumulative GPU time in ns).
 * Replaces research §5-2's "run idle before the pair until the memory clock is pinned" with a time
 * budget that does not depend on nvidia-smi. One ≈80ms pass from idle does not pin the clocks (the
 * same variant fluctuated by 2x in measurement).
 */
export const WARMUP_NS = 500e6;

/**
 * Minimum number of idle runs (even if the cumulative time fills first, a single run right after
 * pipeline creation is not enough).
 */
export const WARMUP_MIN_RUNS = 3;

/**
 * Number of rounds for the representative value (the min is taken, so no parity or pairing is
 * needed — repetitions of the same case).
 */
export const ROUNDS = 5;

/**
 * Decides, from the estimated ns of one dispatch, the repetition count that fills the target length
 * (at least 1, capped at the upper bound). An estimate of 0 or non-finite gives the upper bound
 * (too fast to measure = stack as many as possible).
 *
 * The caller passes the upper bound — the sweep overdraws the same buffer and so takes a large one
 * (harness.ts's `SWEEP_MAX_REPS`), while opbench (removed in 0.15) takes a small one because its
 * output readback grows with the repetitions (`MAX_REPS`).
 */
export const calibrateReps = (nsPerDispatch: number, targetMs: number, maxReps: number): number => {
  if (!Number.isFinite(nsPerDispatch) || nsPerDispatch <= 0) return maxReps;
  const reps = Math.ceil((targetMs * 1e6) / nsPerDispatch);
  return Math.max(1, Math.min(maxReps, reps));
};
