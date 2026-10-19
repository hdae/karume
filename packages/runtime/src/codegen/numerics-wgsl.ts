/**
 * Home of the WGSL fragments for numerical stability that **several kernel families share
 * verbatim** (a neutral leaf module — the same discipline as codegen/errors.ts and
 * codegen/limits.ts).
 *
 * MUST: the source of truth of a shared fragment is here, in one place (the NaN propagation
 * contract of ADR 0020 holds only if "every kernel shares the same single copy"). A copy can make
 * the rounding sequence of the primitive and the fused version diverge. Only fragments that any of
 * elementwise / reduce / attention / GRU / quantization may reference live here; expressions
 * confined to one family stay in that family's module.
 */

import { CodegenError } from "./errors.ts";

/**
 * Always makes integer values decimal literals too (never falls to WGSL's AbstractInt).
 *
 * MUST: append `.0` only when `String(value)` contains none of `.`, `e`, `E`. Appending it to an
 * integer in exponent notation such as `1e21` gives the invalid WGSL `1e+21.0` (exponent notation
 * is already a float literal as is). Non-finite values have no WGSL literal, so they fail loudly.
 */
export const f32Literal = (value: number): string => {
  if (!Number.isFinite(value)) {
    throw new CodegenError(`f32Literal: a non-finite value cannot be a WGSL literal (${value})`);
  }
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.0`;
};

/**
 * MUST: not the naive `1/(1+exp(-x))`. WGSL specifies the result of a floating-point overflow as
 * indeterminate, and implementations exist that can produce a silent NaN for x ≲ -88.
 * Pinning the exp argument to -|x| keeps the result in (0,1] at all times, so overflow cannot
 * happen by construction.
 *
 * MUST: the SiLU fusion kernel (src/kernels/silu.ts) shares this body as is. Copying the same
 * expression can make the rounding sequence of the primitive sigmoid and the fused version diverge
 * (fusion presupposes bit identity).
 */
export const SIGMOID_STABLE_WGSL = `fn sigmoid_stable(x: f32) -> f32 {
  let t = exp(-abs(x));
  return select(1.0 / (1.0 + t), t / (1.0 + t), x < 0.0);
}`;

/**
 * Detects an f32 NaN **by its bit pattern** (sign dropped, exponent all ones + non-zero mantissa).
 *
 * MUST: do not detect NaN with a floating-point comparison. A comparison alone
 * (`select(0.0, 1.0, NaN < m)`) is false as the spec says, yet **the whole `select(x, m, x < m)` is
 * folded by the shader compiler into the `max` idiom, and the driver's `max` swallows NaN**
 * (measured 2026-08-02 on this repository's verification environment: `clamp_min(NaN, min=0) = 0` /
 * `clamp(NaN, -1, 1) = -1` / `relu(NaN) = 0`). WGSL has no way to forbid folding of operations, so
 * the check uses **integer operations that are not folding targets** (`&` and `>`). This is the
 * only reason {@link nanGuard} is used.
 */
export const IS_NAN_BITS_WGSL = `fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`;

/**
 * NaN-propagating `max` (ADR 0020). The driver's `max` swallows NaN (the WGSL spec also drops NaN
 * with "e2 if e1 < e2, otherwise e1"), so a reduction that wants to keep NaN uses this.
 * For two non-NaN arguments it equals the plain `max` — existing numeric results do not move by a
 * single bit.
 * Requires {@link IS_NAN_BITS_WGSL} (the injecting side lists the dependency).
 */
export const NAN_MAX_WGSL = `fn nan_max(a: f32, b: f32) -> f32 {
  return select(select(max(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`;

/**
 * NaN-propagating `min` (the counterpart of {@link NAN_MAX_WGSL}). The `amin` reduction needs it to
 * keep NaN.
 * Requires {@link IS_NAN_BITS_WGSL} (the injecting side lists the dependency).
 */
export const NAN_MIN_WGSL = `fn nan_min(a: f32, b: f32) -> f32 {
  return select(select(min(a, b), b, is_nan_bits(b)), a, is_nan_bits(a));
}`;

/**
 * An outer shell that returns a NaN input as is. The inner value expression (`finite`) handles
 * **only the non-NaN path**, so an existing expression can be wrapped unchanged (= non-NaN numeric
 * results do not move bitwise).
 */
export const nanGuard = (a: string, finite: string): string =>
  `select(${finite}, ${a}, is_nan_bits(${a}))`;

/**
 * The threshold that clips the `tanh` argument in the saturated range. **This value can only be
 * chosen where two conditions hold at once** (rationale in the doc of {@link TANH_STABLE_WGSL}):
 *
 * 1. The f32 `tanh` already rounds to exactly ±1.0 here (the measured lower bound is 9.011).
 *    Below it, clipping **changes the value** = the bit-identity claim breaks.
 * 2. `exp(2·t)` stays far from the f32 overflow (argument 88.72 = value 3.40e38).
 *    At t = 9.5, `exp(19)` = 1.78e8, with 30 orders of magnitude to spare.
 *
 * It sits near the lower end of the band [9.011, 44.36) — moving it up gains nothing (the clipped
 * value is ±1.0 everywhere), and the closer to the lower end, the larger the margin of condition 2.
 * tests/codegen_wgsl_test.ts pins both conditions numerically.
 */
export const TANH_SATURATION = 9.5;

/**
 * The **saturation-clipped** `tanh`. It clamps the argument to ±{@link TANH_SATURATION} before
 * passing it to the built-in `tanh`.
 *
 * MUST: do not call the plain `tanh(x)`. WGSL does not specify how `tanh` is implemented, and an
 * implementation that computes `(exp(2x) − 1)/(exp(2x) + 1)` (Metal's fast-math path — measured)
 * **overflows f32 in exp(2x) for |x| > 44.36** and returns `Inf/Inf` = a silent NaN.
 * The inner argument of `gelu_tanh`, `√(2/π)·(x + 0.044715x³)`, reaches it at a pre-activation
 * x ≳ 10.05, which real models' activations (measured maximum 11.45) hit on every token. Flattening
 * the saturated range first keeps the intermediate exponent from growing at all = overflow cannot
 * happen **by construction** (the same technique as {@link SIGMOID_STABLE_WGSL} pinning the exp
 * argument to -|x|).
 *
 * MUST: clipping is allowed only at a threshold that **does not change non-NaN values bitwise**.
 * The f32 tanh rounds to exactly ±1.0 for |x| ≥ 9.011, so against an IEEE-faithful implementation
 * the result does not move by a single bit with or without clipping (the clipped side is already
 * ±1.0). The threshold's rationale is in {@link TANH_SATURATION}.
 *
 * MUST: NaN propagation is carried by the bit-pattern check ({@link IS_NAN_BITS_WGSL}), not left to
 * the clipping comparisons. The shader compiler may fold a two-step select into the `clamp` idiom
 * (mechanism in the doc of IS_NAN_BITS_WGSL), so written naively `clamp(NaN)` turns into the
 * threshold and **NaN is silently swallowed into ±1.0**.
 *
 * MUST: do not copy this body (the same reason as {@link SIGMOID_STABLE_WGSL} — a copy can make the
 * rounding sequence of the primitive and the fused version diverge).
 *
 * ## Remaining concerns (what adding the clipping did or did not introduce)
 *
 * 1. The possibility that "a tanh implementation that is not IEEE-faithful returns a different
 *    value even **below** the threshold" is outside the reach of this fix. It is a pre-existing
 *    difference that clipping neither worsens nor improves (below the threshold the expression's
 *    input is unchanged).
 * 2. Values below the threshold are bit-identical, but a theoretical room remains that **the shader
 *    compiler reacts to the changed shape of the expression and the rounding moves** (it cannot be
 *    ruled out in principle while WGSL has no way to forbid reassociation). Measurement has ruled
 *    it out — the generated code is pinned by the WGSL snapshots and the values by the golden sets
 *    (6 sbv2 WAV sha256 values, gemma4's exact token-sequence match), all unchanged before and
 *    after this fix.
 * 3. **The same hazard class** "overflows in the middle but the final value is finite" has been
 *    audited across the board (docs/research/2026-08-31-op-numerics-review.md). Every exp path is
 *    safe by construction with max subtraction or the -|x| pin, softplus is protected by its
 *    threshold-where decomposition, and the one leftover, the built-in `tanh` of
 *    src/kernels/gru-scan.ts, was also resolved by sharing this function.
 */
export const TANH_STABLE_WGSL = `fn tanh_stable(x: f32) -> f32 {
  let lo = select(x, ${f32Literal(-TANH_SATURATION)}, x < ${f32Literal(-TANH_SATURATION)});
  let t = select(lo, ${f32Literal(TANH_SATURATION)}, x > ${f32Literal(TANH_SATURATION)});
  return ${nanGuard("x", "tanh(t)")};
}`;

/**
 * Maps the value bits of an f32 to an integer in magnitude order. NaN ranks highest, ±0 tie.
 * Convert once on input; during the reduction compare the integer key and the smallest index.
 * When returning a value, read the original value bits at the chosen index (keeps NaN and ±0).
 * DECIDED: docs/decisions/0068-decode-exit-multi-output.md addendum 10.
 */
export const F32_RANK_WGSL = `fn f32_rank_key(bits: u32) -> u32 {
  let magnitude = bits & 0x7fffffffu;
  let zeroed = select(bits, 0u, magnitude == 0u);
  let ordered = select(zeroed ^ 0x80000000u, ~zeroed, (zeroed & 0x80000000u) != 0u);
  return select(ordered, 0xffffffffu, magnitude > 0x7f800000u);
}

fn rank_key_beats(kb: u32, ib: u32, ka: u32, ia: u32) -> bool {
  return kb > ka || (kb == ka && ib < ia);
}`;
