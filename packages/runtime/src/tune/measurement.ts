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
 *
 * The warm start that opens a case and the check of a calibration's pass were added from the two
 * GEMV sweeps of 2026-11-17 (the block of constants below {@link ROUNDS}).
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

/*
 * The warm start and the calibration check (harness.ts's `warmStart` and `calibrate`). The figures
 * they were chosen from are the two `quick` GEMV sweeps of 2026-11-17 on an RTX 3080 Ti under Node
 * (66 cases and 1,569 rows each, 5 rounds, 10 s idle between cases), whose first kernel of a case
 * was measured cold.
 */

/**
 * Time under load a warm start spends at least (ns) before stable passes may end it: twice
 * {@link WARMUP_NS}. In the second of those sweeps the first kernel of the 262,144-column i8 head
 * (k = 2560) got WARMUP_NS of load (its per-kernel warm-up, at 3 dispatches per pass) and still
 * read 2.9 times its re-measurement at the end of the case: WARMUP_NS from idle is known not to be
 * enough.
 */
export const WARM_START_MIN_NS = 1e9;

/**
 * Time under load a warm start spends at most (ns). Past it the case is measured as it stands and
 * the record shows how unsteady it was (the GEMV sweep's spread of the baseline's rounds, the
 * geometry sweep's drift ratio). It bounds the cost: the 66 `quick` cases of the GEMV sweep spend at
 * most 5.5 minutes in the warm starts that open them, against the 35 minutes those runs measured.
 */
export const WARM_START_MAX_NS = 5e9;

/**
 * Consecutive passes whose times must agree for a warm start to call the device steady: three, two
 * agreements in a row, so that two passes that happen to land on one step of a clock ramp are not
 * taken for its end.
 */
export const WARM_START_PASSES = 3;

/**
 * How far those passes may differ: largest ÷ smallest at most 1 + this. In those sweeps the five
 * rounds of a baseline whose pass was at least 40 ms spread by at most 1.0197 (86 rows, median
 * 1.0009): 2 % holds a steady device and still tells a ramp.
 */
export const WARM_START_TOLERANCE = 0.02;

/**
 * A calibration whose pass comes out below this share of {@link TARGET_PASS_MS} is done again after
 * more load. In those sweeps the shortest pass of every row whose reps were below the cap was
 * either 40.7 ms or longer, or 24.5 ms or shorter: 256 of those 2,933 rows under 20 ms (44 of them
 * baselines) and 2 between 20 and 25 ms. A quarter of the target catches all but those 2 and stays
 * a factor of two below the shortest healthy pass.
 */
export const RECALIBRATE_BELOW = 0.25;

/**
 * Calibrations of one launch at most. The last one is accepted whatever its pass: a dispatch that
 * stays that slow through two more warm starts is measured as it is.
 */
export const CALIBRATION_ATTEMPTS = 3;

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
