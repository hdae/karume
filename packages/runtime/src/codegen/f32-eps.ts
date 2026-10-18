// The f32 range of a normalization eps (the single check shared by the op contract
// <ops/attrs.ts> and the params of the normalization kernels).
//
// When attrs and params each hold the same condition, only one side gets fixed (there is a
// precedent where only params let subnormals through), so the check lives here in one place. The
// caller decides the exception type and message.

/** The smallest normal f32 (2^-126). */
const F32_MIN_NORMAL = 2 ** -126;

/**
 * Why an eps cannot be used as an f32.
 *
 * - `not-positive`: not a finite positive number (0, negative, NaN, ±Inf)
 * - `subnormal`: rounds to 0 or a subnormal in f32. On a GPU with FTZ (treating subnormals as 0 —
 *   WGSL allows it) it runs with eps = 0, and an all-zero row becomes a division by zero
 * - `overflow`: rounds to infinity in f32 (only the GPU side gives `1/sqrt(… + Inf) = 0`, all bias)
 */
export type F32EpsDefect = "not-positive" | "subnormal" | "overflow";

/** `undefined` when eps is finite and rounds to a normal f32; otherwise the reason. */
export const f32EpsDefect = (value: number): F32EpsDefect | undefined => {
  if (!Number.isFinite(value) || value <= 0) return "not-positive";
  const rounded = Math.fround(value);
  if (rounded < F32_MIN_NORMAL) return "subnormal";
  if (!Number.isFinite(rounded)) return "overflow";
  return undefined;
};
