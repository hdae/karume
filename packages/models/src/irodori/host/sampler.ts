/**
 * The part of the Euler sampler left on the host (ADR 0047 decision 4) — t schedule, CFG
 * combination, update of x.
 *
 * MUST: f32 rounding is applied one operation at a time with `Math.fround` (the same section of
 * `anima/sampler.ts`).
 */

const f32 = Math.fround;

/**
 * The t sequence of `[steps+1]` (**closed form** `initScale·(1 − i/steps)`).
 *
 * NOTE: upstream builds it as `(1 − linspace(0,1,n+1))·initScale`. The two are not bit-identical
 * (`linspace` accumulates `start + i·step` in f32, so the last ulp splits at some points — measured
 * on the exporter side, `tScheduleClosedFormMaxAbs` = 5.96e-8 @ steps=40). t enters only
 * `t_embed` and the step size `t_next − t`, and the effect of this difference on the final latent
 * is 3 orders of magnitude below the golden comparison threshold.
 * **The closed form is chosen because index and value read 1 to 1**, not because of a constraint
 * based on measured rounding.
 */
export const tSchedule = (steps: number, initScale: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`steps ${steps} が 1 以上の整数でない`);
  }
  const schedule = new Float32Array(steps + 1);
  for (let index = 0; index <= steps; index += 1) {
    schedule[index] = f32(initScale * f32(1 - index / steps));
  }
  return schedule;
};

/** One CFG variant (the result of an uncond forward and its strength). */
export type CfgVariant = {
  /**
   * Strength.
   * MUST: **a value exactly representable in f32** — here it enters the multiply as f64, so a value
   * that is not exact in f32 splits by 1 to 2 ulp from the resident path, which queues the same
   * operation in f32 (`parseCfgScales` rejects it at the distribution's declaration step).
   */
  readonly scale: number;
  /** The velocity field run with the mask that drops only that condition. */
  readonly velocity: Float32Array<ArrayBuffer>;
};

/**
 * The CFG-independent combination `v = v_cond + Σ_k s_k·(v_cond − v_k)`.
 *
 * MUST: keep the variant order (text → speaker → caption). Addition is not associative, so a
 * different order changes the last digits (upstream queues the tensor operations one variant at a
 * time in this order).
 * MUST: return a new array even with 0 variants (so the caller does not enter the next run while
 * holding on to a Session output buffer).
 */
export const combineCfg = (
  cond: Float32Array<ArrayBuffer>,
  variants: readonly CfgVariant[],
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(cond.length);
  for (const variant of variants) {
    if (variant.velocity.length !== cond.length) {
      throw new Error(`CFG 変種の長さ ${variant.velocity.length} が cond ${cond.length} と違う`);
    }
  }
  for (let index = 0; index < cond.length; index += 1) {
    const base = cond[index];
    let value = base;
    for (const variant of variants) {
      value = f32(value + f32(variant.scale * f32(base - variant.velocity[index])));
    }
    out[index] = value;
  }
  return out;
};

/** Euler update `x + v·Δt` (Δt = `t_next − t`, negative). */
export const eulerStep = (
  x: Float32Array<ArrayBuffer>,
  velocity: Float32Array<ArrayBuffer>,
  deltaT: number,
): Float32Array<ArrayBuffer> => {
  if (velocity.length !== x.length) {
    throw new Error(`速度場の長さ ${velocity.length} が x ${x.length} と違う`);
  }
  const out = new Float32Array(x.length);
  for (let index = 0; index < x.length; index += 1) {
    out[index] = f32(x[index] + f32(deltaT * velocity[index]));
  }
  return out;
};
