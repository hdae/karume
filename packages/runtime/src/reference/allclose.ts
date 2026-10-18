/**
 * Numeric agreement checks. Used to compare GPU execution with the CPU reference (stage 2 of
 * ADR 0005).
 *
 * MUST: the check is always `|x−y| ≤ atol + rtol·|ref|`. Relative error alone diverges at reference
 * values near zero; absolute error alone becomes pointlessly strict for large values.
 * MUST: NaN / ±Inf on either side fail. Every comparison with NaN is false, so a naive difference
 * check turns it into "difference 0" and an all-NaN output PASSes.
 */

import type { IrDtype } from "../format/ir.ts";

/**
 * The compared lengths or dtypes differ, or the tolerance itself is broken (a mix-up — not a
 * question of how large the error is).
 */
export class AllcloseError extends Error {
  override readonly name = "AllcloseError";
}

export type Tolerance = {
  readonly atol: number;
  readonly rtol: number;
};

/**
 * The default band when none is given. **It is no longer the check of the op vocabulary sweep**
 * (2026-08-31 — the M1 follow-up done): the sweep (checkAll in tests/gpu_ops_test.ts) and the GEMM
 * comparisons explicitly use the per-op table derived from measurement
 * (tests/helpers/op-tolerance.ts; source of truth =
 * docs/research/2026-08-31-op-tolerance-measurement.md).
 * What remains is "a temporary value until the caller has its own band" and the default of wiring
 * tests; **do not use this default for a new numeric check** (add a row to the table, or pass an
 * explicit value from measurement).
 */
export const DEFAULT_TOLERANCE: Tolerance = { atol: 1e-5, rtol: 1e-3 };

/**
 * The tolerance for comparing integer types (i32 / bool) = exact match (ADR 0009).
 * Integer arithmetic has neither rounding differences nor reduction-order differences, so any
 * deviation, even by 1, is an implementation bug.
 */
export const EXACT_TOLERANCE: Tolerance = { atol: 0, rtol: 0 };

type AllcloseReport = {
  readonly pass: boolean;
  /** Number of elements that broke the tolerance. */
  readonly failCount: number;
  readonly maxAbsError: number;
  readonly maxRelError: number;
  /** Position of maxAbsError (of the first non-finite element when there is one). */
  readonly worstIndex: number;
  /** Number of elements where either side was NaN / ±Inf. */
  readonly nonFiniteCount: number;
};

export const allclose = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: Tolerance = DEFAULT_TOLERANCE,
): AllcloseReport => {
  if (actual.length !== expected.length) {
    throw new AllcloseError(`長さ不一致: actual ${actual.length} vs expected ${expected.length}`);
  }
  // MUST: reject a NaN or negative tolerance. With NaN the right-hand side of the check is NaN and
  // `abs > rhs` is always false (any error passes); with a negative value even error 0 fails. e2e
  // tests deliberately pass atol = +Infinity as a record-only band, so atol is not required to be
  // finite.
  if (Number.isNaN(tolerance.atol) || tolerance.atol < 0) {
    throw new AllcloseError(`許容差 atol が NaN か負である: ${tolerance.atol}`);
  }
  // MUST: rtol is limited to finite non-negative values. With rtol = +Infinity an element whose
  // reference is 0 gives Infinity·0 = NaN, the right-hand side becomes NaN and it passes whatever
  // atol is (a record-only band is expressed on the atol side).
  if (!Number.isFinite(tolerance.rtol) || tolerance.rtol < 0) {
    throw new AllcloseError(`許容差 rtol が有限の非負でない: ${tolerance.rtol}`);
  }
  let failCount = 0;
  let maxAbsError = 0;
  let maxRelError = 0;
  let worstIndex = 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const x = actual[i];
    const y = expected[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (nonFiniteCount === 0) worstIndex = i;
      nonFiniteCount += 1;
      failCount += 1;
      maxAbsError = Number.POSITIVE_INFINITY;
      maxRelError = Number.POSITIVE_INFINITY;
      continue;
    }
    const abs = Math.abs(x - y);
    if (abs > maxAbsError && nonFiniteCount === 0) {
      maxAbsError = abs;
      worstIndex = i;
    }
    maxRelError = Math.max(maxRelError, abs / Math.max(Math.abs(y), 1e-12));
    if (abs > tolerance.atol + tolerance.rtol * Math.abs(y)) {
      failCount += 1;
    }
  }
  return { pass: failCount === 0, failCount, maxAbsError, maxRelError, worstIndex, nonFiniteCount };
};

/**
 * What is compared (the smallest surface that both the public Tensor and the CPU reference's
 * RefTensor fit structurally).
 */
type NumericTensor = {
  readonly dtype: IrDtype;
  readonly data: ArrayLike<number>;
};

/**
 * A comparison that picks the tolerance by dtype. f32 shows rounding and reduction-order
 * differences by construction, so it uses allclose; i32 / bool use {@link EXACT_TOLERANCE} (exact
 * match).
 *
 * MUST: do not compare when the dtypes disagree. Every element type is 4 bytes and the values can
 * still be read as numbers, so the comparison would "happen to pass / happen to fail" and the
 * verification would lose its meaning.
 */
export const compareTensors = (
  actual: NumericTensor,
  expected: NumericTensor,
  floatTolerance: Tolerance = DEFAULT_TOLERANCE,
): AllcloseReport => {
  if (actual.dtype !== expected.dtype) {
    throw new AllcloseError(`dtype 不一致: actual ${actual.dtype} vs expected ${expected.dtype}`);
  }
  return allclose(
    actual.data,
    expected.data,
    expected.dtype === "f32" ? floatTolerance : EXACT_TOLERANCE,
  );
};

/** For test failure messages (shows in one line where and by how much it deviated). */
export const formatAllclose = (report: AllcloseReport): string =>
  `fail=${report.failCount} nonFinite=${report.nonFiniteCount} maxAbs=${report.maxAbsError} maxRel=${report.maxRelError} at=${report.worstIndex}`;
