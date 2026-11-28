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
 * Time under load a warm start spends at least (ns) when the device is already under load: the GEMV
 * sweep's warm start of a row count that follows another row count of the same case, which timed a
 * kernel right before it with no idle time between (the first row count of a case, after the idle
 * time or the previous case, keeps {@link WARM_START_MIN_NS}). None: its three agreeing passes end
 * it. Two pairs of `explore` GEMV sweeps on an RTX 3080 Ti under Node on 2026-11-27, no idle time,
 * 5 rounds: with the full minimum in every row count 2,433 s and 2,408 s; with no minimum after a
 * case's first row count 2,145 s and 2,174 s. The derivation's decisions differed between the pairs
 * in 31 of 216 (11 slower, 17 faster by the full sweeps' timings), against 27 to 29 of 216 between
 * two mixes of the same four records (the repeat noise). Measured on that one device only.
 */
export const WARM_START_LOADED_MIN_NS = 0;

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
 * The quick look's margin (harness.ts's `quickLook` and `quickLookGoesOn`): after the warm start
 * that opens a group of kernels measured together, every kernel gets one short timed pass, and only
 * the references and the kernels whose quick time is at most this many times the bar of their class
 * go on to the calibration's check, the correctness check and the rounds. The bar is what the
 * derivation compares the class's kernels with in that group: the fastest quick time of the class
 * where it picks a winner (the GEMV sweep at M = 1, among the kernels whose check, read right after
 * their quick look, held, the subgroup form, which no table row holds, left out: a kernel the
 * derivation can never adopt sets no bar), the quick time of the reference where it asks
 * only whether a kernel clears a threshold over it (the parallel GEMV's other row counts, which must
 * not be slower than the default's choice; every case of the geometry sweep, where a geometry must
 * be `minSpeedup` faster than the default in every case of its class). `0` turns the quick look off.
 *
 * Source: the two `full` GEMV sweeps of 2026-11-17 (5,580 rows each, 528 groups of a case, a row
 * count and a class), on an RTX 5070 Ti under Chrome and an RTX 3080 Ti under Node, reading each
 * row's first round as its quick look. Per group, 1.9 of 10.6 candidates are within 10 % of the
 * fastest. Against the fastest of the class, ×1.15 keeps 23 % of the rows on the 5070 Ti and drops
 * the candidate that wins on the full timing in 0 of 528 groups (×1.25: 28 %, 0); on the 3080 Ti,
 * whose timing is unsteady, ×1.15 drops it in 25 groups and ×1.25 in 18. Against the fastest alone, the parallel candidate each record would adopt was screened
 * at M = 4 or 8 in 8 of 58 cases (5070 Ti) and 13 of 39 (3080 Ti) at ×1.25 — its no-regression
 * check would have had nothing to read; with the reference as the bar there, in none, the rows that
 * go on being 46 % and 48 % (the references included). The geometry sweep's `quick+` record of the
 * 5070 Ti of 2026-11-16 (821 rows): against the fastest of the case, ×1.25 changes one of its six
 * adopted fields (`i8a8.conv3d`); against the default, none, keeping 71 % of the rows (the M2's
 * record of 2026-11-17: none either way).
 */
export const QUICK_LOOK_MARGIN = 1.25;

/**
 * The quick look's default margin of the GEMV sweep's `explore` set ({@link QUICK_LOOK_MARGIN} is
 * every other set's).
 *
 * Source: replayed on the GEMV sweep records of 2026-11-18/19, a margin of ×1.10 left the
 * derivation's decisions unchanged on the RTX 5070 Ti (0 of 116) and changed 2 of 214 on the Apple
 * M2, each within 2 %; ×1.05 lost adoptions on both. NOTE: a replay under the rule of two records;
 * it has not been run again under the rule of one steady record.
 */
export const EXPLORE_QUICK_LOOK_MARGIN = 1.1;

/**
 * A sweep's quick-look margin: `fallback` (default {@link QUICK_LOOK_MARGIN}) when not given; 0
 * (off) or a finite number of at least 1 otherwise (below 1 would screen out every kernel but the
 * references, the fastest included). Throws for any other value — before a sweep touches the GPU.
 */
export const quickLookMarginOf = (
  margin: number | undefined,
  fallback: number = QUICK_LOOK_MARGIN,
): number => {
  if (margin === undefined) return fallback;
  if (margin === 0 || (Number.isFinite(margin) && margin >= 1)) return margin;
  throw new Error(
    `quickLookMargin must be 0 (the quick look off) or a number of at least 1 (${margin})`,
  );
};

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
