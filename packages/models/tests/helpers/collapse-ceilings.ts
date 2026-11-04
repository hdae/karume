// The collapse ceilings of the DeBERTa w8a8 mirror gate (output.2 onwards — ADR 0026 addendum
// 2026-10-07) and their judgement. They live outside the e2e test so the asset-free unit test
// (`collapse_ceilings_test.ts`) can import them without registering the GPU cases.

import { AllcloseError } from "../../../runtime/src/reference/allclose.ts";

/** The number of outputs of the full-24layer graph (embedding exit + 24 layer exits). */
export const GRAPH_OUTPUTS = 25;

/**
 * The number of outputs checked with the strict tolerance (output.0 and output.1 — up to the exit
 * of layer 1, where the discriminating band still holds).
 */
export const STRICT_OUTPUTS = 2;

/**
 * The table of **collapse ceilings** applied from output.2 onwards = output name → the ceiling of
 * the relative RMS error ‖gpu − golden‖₂ / ‖golden‖₂ (**declared values derived from measurements
 * on this environment** — 2026-10-07, RTX 3080 Ti / Ryzen 5 7600, ADR 0026 addendum 2026-10-07). It
 * is a declaration, not a per-environment-key row (ADR 0110 decision 4).
 *
 * Derivation: per output (observation point = output.k), the worst of 60 legitimate samples (4
 * cases × [1 GPU a8 run + 6 different samples of the torch mirror re-taken on this machine's CPU
 * (AVX512 / AVX2 / DEFAULT × 1 / 6 threads) + a perturbation ensemble of 8 runs with relative
 * ±2^-23 noise injected at the mirror's embedding exit]) × 2, rounded up to 2 significant digits.
 * This check is not the A/B check of ADR 0110 decision 5 (the comparison partner is the torch
 * mirror golden, not the same machine's reference tier), but its ceiling derivation alone follows
 * decision 5-3 (an E2E collapse ceiling = about 2 × the measurement at the observation point) and
 * does not break decision 4's MUST NOT (using "5–10 × the measurement" to derive an E2E band).
 * 5-3's "keep consistency with the theoretical value for a quantized preset" is not applied:
 * activation rounding is discontinuous, and an upstream difference of the order of 1e-5 causes ±1
 * jumps of the rounding step that saturate within a few layers (ADR 0026 "detection limit"), so
 * there is no theoretical upper bound usable for the relative RMS error of output.k — the numeric
 * correctness of a8 is owned by the atol=0 numeric contract of `tests/gpu_i8a8_test.ts`. Dumps of
 * the GPU f32 path are not a8 samples, so they are not included. The worst case is smaller the
 * shallower the output (output.2 0.0099 → output.24 0.1620), so a single scalar would loosen the
 * ceilings of the shallow outputs by orders of magnitude — that is why it is a table.
 *
 * Reproduction: `derive` of `tools/export-recipes/deberta/collapse_ceilings.py` writes this table
 * to `ceilings` of `derive.json` in the working dir (how the samples are made, and in what order,
 * is the "re-derivation procedure" of the ADR 0026 addendum). Fault injection is `faults` of
 * `tools/diag/deberta-collapse.ts` (it imports {@link collapseCeilingOf} and {@link judgeCollapse}
 * as they are). The 2026-10-07 material is in `outputs/diag/deberta-w8a8-2026-10-07/` (not tracked
 * by git).
 *
 * This is not numeric parity: it covers only non-finite values ({@link judgeCollapse} looks at both
 * sides and every element) and collapses where the whole output becomes something other than the
 * golden (all elements 0 = 1.0, sign flip ≈ 2.0, every element × 2 ≈ 1.0, × 1.5 ≈ 0.5). It does not
 * catch a one-element jump in an intermediate layer or a silent fallback to the f32 path — finer
 * defects are owned by the strict tolerance of output.0 / 1, the census, and the atol=0 numeric
 * contract of `tests/gpu_i8a8_test.ts`. Rather than widening a ceiling to pass, first find out
 * which of the i8a8 scale / accumulator / eligibility decision moved.
 */
export const COLLAPSE_CEILINGS: Readonly<Record<string, number>> = {
  "output.2": 0.02,
  "output.3": 0.037,
  "output.4": 0.051,
  "output.5": 0.062,
  "output.6": 0.071,
  "output.7": 0.085,
  "output.8": 0.095,
  "output.9": 0.11,
  "output.10": 0.12,
  "output.11": 0.13,
  "output.12": 0.14,
  "output.13": 0.15,
  "output.14": 0.15,
  "output.15": 0.17,
  "output.16": 0.17,
  "output.17": 0.19,
  "output.18": 0.21,
  "output.19": 0.22,
  "output.20": 0.24,
  "output.21": 0.26,
  "output.22": 0.28,
  "output.23": 0.31,
  "output.24": 0.33,
};

/**
 * The collapse ceiling of an io output name (`output.<k>`). An output missing from the table throws
 * (an output without a ceiling is not passed silently).
 */
export const collapseCeilingOf = (output: string): number => {
  if (!Object.hasOwn(COLLAPSE_CEILINGS, output)) {
    throw new Error(`${output} has no declared collapse ceiling (COLLAPSE_CEILINGS)`);
  }
  return COLLAPSE_CEILINGS[output];
};

/** The collapse-ceiling judgement of one output ({@link judgeCollapse}). */
export type CollapseReport = {
  readonly pass: boolean;
  /**
   * ‖actual − expected‖₂ / ‖expected‖₂ (accumulated in f64). +Inf when there is a non-finite value.
   */
  readonly relRms: number;
  /** Recorded only — not used in the judgement. +Inf when there is a non-finite value. */
  readonly maxAbs: number;
  /** The number of elements that were NaN / ±Inf on either side (every element is looked at). */
  readonly nonFiniteCount: number;
};

/**
 * The judgement of one output from output.2 onwards: zero non-finite values (both sides, every
 * element) and a relative RMS error ≤ `ceiling` (that output's ceiling —
 * {@link collapseCeilingOf}).
 *
 * Why not judge by maxAbs: the intermediate layers sit at the sign branch point of an outlier
 * channel (channel 686, \|ref\| up to 28.8), and a flipped rounding step of activation quantization
 * moves that one element a lot. Even between different samples of the torch mirror on CPU that
 * element moves by 7.17 (ADR 0026 addendum 2026-10-07).
 */
export const judgeCollapse = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  ceiling: number,
): CollapseReport => {
  if (actual.length !== expected.length) {
    throw new AllcloseError(
      `length mismatch: actual ${actual.length} vs expected ${expected.length}`,
    );
  }
  let difference = 0;
  let norm = 0;
  let maxAbs = 0;
  let nonFiniteCount = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const x = actual[index];
    const y = expected[index];
    // MUST: non-finite values are counted and fail the check rather than being left to the size of
    // the relative RMS — the verdict must not depend on the direction of a comparison (every
    // comparison with NaN is false), and the report shows what happened.
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      nonFiniteCount += 1;
      continue;
    }
    const delta = x - y;
    difference += delta * delta;
    norm += y * y;
    maxAbs = Math.max(maxAbs, Math.abs(delta));
  }
  if (nonFiniteCount > 0) {
    return {
      pass: false,
      relRms: Number.POSITIVE_INFINITY,
      maxAbs: Number.POSITIVE_INFINITY,
      nonFiniteCount,
    };
  }
  const relRms = Math.sqrt(difference) / Math.sqrt(norm);
  // ‖expected‖₂ = 0 gives 0/0 = NaN or +Inf, so `<=` returns false and the check fails.
  return { pass: relRms <= ceiling, relRms, maxAbs, nonFiniteCount };
};
