/**
 * Anima's sampler — holds the steps that are not in the IR, in TS.
 *
 * Three parts: ① the sigma sequence of `FlowMatchEulerDiscreteScheduler` ② the sinusoidal timestep
 * embedding ③ CFG mixing + the Euler update. Whether CFG is needed ({@link needsUncond}) is a
 * one-line invariant independent of the update rule, so it is borrowed from the family-independent
 * `src/generation/` (no copy). The Anima-side context stays here — the 2× of turbo operation comes
 * from **this branch**, independent of the step reduction, and the reference fixtures do not run
 * the uncond branch at CFG=1 either.
 *
 * MUST: apply the f32 rounding one operation at a time with `Math.fround`. JS numbers are f64, so
 * computing in bulk and rounding at the end changes the last digit against torch / numpy's
 * sequential f32 computation.
 *
 * NOTE: **the manifest's `pipelineConfig.scheduler` is the source of truth** for `shift` /
 * `numTrainTimesteps` (numbers that can vary per distribution, so they are not constants here —
 * ADR 0038 §1).
 */

import { ModelInputError } from "../errors.ts";
import { needsUncond } from "../generation/dpm-solver-multistep.ts";

const f32 = Math.fround;

/** `max_period` of `get_timestep_embedding` (diffusers' default 10000). */
const MAX_PERIOD = 10000;

/**
 * Upper bound of `steps` ({@link assertAcceptableSteps}).
 *
 * NOTE: a practical upper bound aligned with Wan (`wanUniPcSchedule` in `wan/scheduler.ts` — 1000 =
 * `num_train_timesteps`). Wan's reason "beyond it integer timesteps always repeat" does not apply
 * to Anima — Anima's timestep embedding uses the continuous σ value as is ({@link timestepsProj}).
 * The bound's purpose is to stop at the entry before a huge steps value allocates the ladder's
 * `Float32Array` and runs a synchronous loop (which cannot even be aborted). The manifest default
 * check (`config.ts`) calls {@link assertAcceptableSteps} rather than reading the value.
 */
const ANIMA_MAX_STEPS = 1000;

/**
 * The check of the accepted set of `steps` (**an integer from 2 to {@link ANIMA_MAX_STEPS}**).
 * **This is the single owner**; both the generation entry (`generate` in `pipeline.ts`) and
 * {@link sigmaSchedule} itself call it.
 *
 * NOTE: the generation entry calls this first, alongside the other knobs, so input-caused errors
 * come back as {@link ModelInputError} before the prompt layer (tokenization). The call in
 * {@link sigmaSchedule} is the contract for using the low-level function on its own (tests, tools).
 * The ladder itself is also built at the generation entry before touching the GPU (`generate` in
 * `pipeline.ts`). Writing the accepted set in both places would always let one side drift looser,
 * so the conditions live only here.
 *
 * Input-caused (the generation knob the caller passed itself), so it throws {@link ModelInputError}
 * — the ladder's structural checks (head is 1 / strictly decreasing) are internal assumptions about
 * the distribution's `shift`, so they stay `RangeError`, and the types tell them apart.
 */
export const assertAcceptableSteps = (steps: number): void => {
  if (!Number.isInteger(steps) || steps < 2) {
    throw new ModelInputError(
      `steps ${steps} が 2 以上の整数でない（sigma の linspace が組めない）`,
    );
  }
  if (steps > ANIMA_MAX_STEPS) {
    throw new ModelInputError(`steps ${steps} が ${ANIMA_MAX_STEPS} 以下でない（steps の上限）`);
  }
};

// Third-party code notice. `sigmaSchedule` below is adapted from `AnimaSetTimestepsStep.__call__`
// (its default `np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)` ladder) in
// huggingface/diffusers (`src/diffusers/modular_pipelines/anima/before_denoise.py`) and from the
// static shift path of `FlowMatchEulerDiscreteScheduler.set_timesteps`
// (`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`), both `diffusers==0.39.0`.
// The ladder, the shift and the terminal zero are carried over with each f32 operation written out;
// the structural checks on the ladder are added here. License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notices, copied verbatim from
// the headers of those files:
//
//   src/diffusers/modular_pipelines/anima/before_denoise.py:
//     Copyright 2026 The HuggingFace Team. All rights reserved.
//   src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:
//     Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.

/**
 * TS implementation of `FlowMatchEulerDiscreteScheduler.set_timesteps` (the static shift path).
 * A terminal 0 is appended, so the length is `steps + 1`.
 *
 * The construction copies numpy's `linspace` (build `step = (stop − start) / (num − 1)` first, then
 * `i · step + start`; the last element is assigned `stop` directly).
 *
 * NOTE (fault-injection measurement): with steps=32 / shift=3, neither ① reordering to
 * `(i · delta) / div + start` nor ② rounding to f32 once at the end instead of per operation
 * **changes a single bit of the 33 elements**. The formula is well-conditioned within the range
 * [0.088, 1] and does not lose enough digits for rounding to matter. **The parity test pins the
 * values, not the shape of the formula** — the copy is there so it can be followed 1:1 against the
 * upstream formula, not a constraint from a rounding measurement. What actually fails is a mix-up
 * of the linspace endpoints, the shift or the terminal 0 (confirmed by fault injection).
 *
 * ## MUST: check the ladder's structure before returning (source of truth for the `shift` range)
 *
 * `dpm-solver-multistep.ts` states "the ladder starts at exactly 1 and ends at exactly 0" as an
 * explicit assumption and derives step 0's `alphaS0 = 0` from it. The accepted set of
 * `pipelineConfig.scheduler.shift` ("positive finite number" in `config.ts`) is **wider** than that
 * assumption: the f32 step near 1.0 is 2⁻²⁴ ≈ 5.96e-8, so with `shift ≲ 3e-8` the denominator
 * `f32(shift − 1) + 1` collapses to exactly 0 and `sigmas[0] = +Inf` (Euler turns the latent into
 * NaN, which propagates in a form whose real cause cannot be read until the non-finite check of
 * `imageToRgba`), and with `shift ≈ 1e-7` the head is no longer 1 (DPM++ 2M throws nothing and
 * silently runs a different trajectory with the assumption broken). Writing the accepted set in two
 * places would always let one side drift looser, so the check is this single one on the ladder
 * side.
 *
 * NOTE (the head check is one-sided — `1 − 4ulp ≤ σ₀ ≤ 1`): upstream diffusers evaluates the same
 * formula in f32, and with `shift = 1.18 / 1.32 / 1.43 …` (111 of the 1,901 two-decimal values in
 * `[1,20]`) the head is 1–2 ulp away from 1. The handling depends on the direction:
 * - **Below (σ₀ < 1, e.g. 1.18) is accepted** — an exact comparison would reject these ordinary
 *   values too and narrow the distribution's freedom. dpm-solver's "head is exactly 1" is the
 *   assumption of a derivation that treats `alphaS0 = 1 − σ₀` as 0, and a downward offset gives
 *   `alphaS0 ≈ 6e-8`, which does not affect the picture. The width is {@link SIGMA_HEAD_TOLERANCE}
 *   (4 ulp).
 * - **Above (σ₀ > 1, e.g. 1.32 / 1.57 …, 55 of the same 1,901 values) is rejected** — DPM++ 2M
 *   takes `log(1 − σ₀)` at step 0, so the log of a negative number gives NaN and every step runs on
 *   NaN without an exception (it fails at the final RGBA conversion in a form whose real cause
 *   cannot be read, or the VAE's clamp turns it into ordinary pixels). The ladder is one regardless
 *   of the update rule and the check sits only here, so it is rejected for Euler too (which does
 *   not break above) — only the theoretical freedom of a fractional `shift` is lost.
 */
/** How far the ladder head may be **below** 1 (f32 4 ulp = 2⁻²²; 0 above — a one-sided check). */
const SIGMA_HEAD_TOLERANCE = 2 ** -22;

export const sigmaSchedule = (steps: number, shift: number): Float32Array<ArrayBuffer> => {
  assertAcceptableSteps(steps);
  const start = 1;
  const stop = 1 / steps;
  const step = (stop - start) / (steps - 1);
  const sigmas = new Float32Array(steps + 1);
  for (let index = 0; index < steps; index += 1) {
    const sigma = f32(index === steps - 1 ? stop : index * step + start);
    sigmas[index] = f32(f32(shift * sigma) / f32(f32((shift - 1) * sigma) + 1));
  }
  // The head may be a few ulp of f32 rounding below 1 only ({@link SIGMA_HEAD_TOLERANCE}). Above,
  // even 1 ulp makes DPM++ 2M's `log(1 − σ₀)` NaN, so it is not allowed (+Inf also fails on this
  // upper branch). NaN compares false, so it fails on the same branch.
  if (!(sigmas[0] <= 1 && 1 - sigmas[0] <= SIGMA_HEAD_TOLERANCE)) {
    throw new RangeError(
      `sigma 梯子の先頭が ${sigmas[0]}（1 を超えている / 1 から下へ ${SIGMA_HEAD_TOLERANCE} を` +
        `超えて離れている）— shift ${shift} は梯子の成立域の外`,
    );
  }
  for (let index = 1; index < sigmas.length; index += 1) {
    if (!(sigmas[index] < sigmas[index - 1])) {
      throw new RangeError(
        `sigma 梯子が狭義単調減少でない（index ${index - 1} → ${index} が` +
          ` ${sigmas[index - 1]} → ${sigmas[index]}）— shift ${shift} は梯子の成立域の外`,
      );
    }
  }
  return sigmas;
};

// Third-party code notice. `timestepsProj` below is adapted from `get_timestep_embedding` in
// huggingface/diffusers (`src/diffusers/models/embeddings.py`, `diffusers==0.39.0`), specialized to
// `flip_sin_to_cos=True`, `downscale_freq_shift=0`, `scale=1` and an even width, with each f32
// operation written out in the upstream order. License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
// header of that file:
//
//   Copyright 2026 The HuggingFace Team. All rights reserved.

/**
 * Timestep embedding `[width]` for one step (a copy of `Timesteps(2048, flip_sin_to_cos=True,
 * downscale_freq_shift=0.0)` = `get_timestep_embedding`). **cos in the first half, sin in the
 * second.**
 *
 * The input is sigma itself. Upstream round-trips through `timesteps[i] / num_train_timesteps`, so
 * the order is copied, but **the f32 round trip changes no bit** (same as the exporter-side
 * measurement NOTE).
 *
 * `width` comes from the static dimension of the graph input `timesteps_proj` (callers do not write
 * 2048).
 *
 * ## MUST: this is **not** bit-identical to the reference (measured — do not tighten for a match)
 *
 * torch CPU's f32 `exp` / `sin` / `cos` are SLEEF's 1.0 ULP implementations, and their last bit
 * differs from JS `Math.*` (computed in f64 and rounded to f32 = about 0.5 ULP). Measured (3
 * reference fixtures = 72 rows in total): mismatches 4.01–4.03% / at most 2 ULP / largest absolute
 * difference 5.96e-8. Of these, the `exp` step accounts for 9 of 1,024 (1 ULP) and the rest are the
 * `sin` / `cos` step. Bit identity would require copying SLEEF's polynomial approximations, which
 * is not worth the complexity for a 6e-8 difference on a table in the range [-1,1]. The parity test
 * pins it with atol 6e-7 derived from this measurement (about 10× the measured worst) —
 * implementation bugs differ by orders of magnitude more (measured: **row shift 1.68e-1** = 5.4
 * orders above the threshold / **cos·sin swap 1.00e+0** = 6.2 orders above), so detection power
 * remains while allowing rounding differences.
 */
export const timestepsProj = (
  sigma: number,
  width: number,
  numTrainTimesteps: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(width) || width <= 0 || width % 2 !== 0) {
    throw new RangeError(`timesteps_proj の幅 ${width} が正の偶数でない`);
  }
  if (!Number.isInteger(numTrainTimesteps) || numTrainTimesteps <= 0) {
    throw new RangeError(`num_train_timesteps ${numTrainTimesteps} が正の整数でない`);
  }
  const half = width / 2;
  const timestep = f32(f32(sigma * numTrainTimesteps) / numTrainTimesteps);
  // `-math.log(max_period)` drops to f32 in the product with an f32 tensor (torch's scalar
  // promotion). Multiplying in f64 was measured to raise the mismatches from 825 to 2,503 and the
  // maximum to 17 ULP.
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const out = new Float32Array(width);
  for (let index = 0; index < half; index += 1) {
    // downscale_freq_shift = 0, so the denominator is half itself.
    const exponent = f32(f32(logMaxPeriod * index) / half);
    const angle = f32(timestep * f32(Math.exp(exponent)));
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};

/**
 * CFG mixing (`uncond + scale·(cond − uncond)`) and the Euler update (`x + Δσ·noise`).
 *
 * MUST: the presence of `uncond` matches {@link needsUncond} exactly (a mismatch fails loudly).
 * "Passing uncond at CFG=1" is a branch missed by the caller; passed through, **values that differ
 * from the reference at the 1 ULP level get absorbed into the tolerance as "GPU error"**.
 *
 * NOTE (why lengths are not compared): the lengths of `previous` / `cond` / `uncond` are **made
 * equal structurally by the caller**. The only caller is the denoise loop in `pipeline.ts`, and
 * cond / uncond are outputs of the same dyn graph run with the same input shape, so their lengths
 * cannot differ; a mismatch with the latent side is caught first, by name, in `unpatchifyTokens`
 * (`dit-tokens.ts`). The family-independent shared surface (`generation/dpm-solver-multistep.ts`)
 * is different, since its callers can grow.
 */
export const cfgEulerStep = (
  previous: Float32Array<ArrayBuffer>,
  cond: Float32Array<ArrayBuffer>,
  uncond: Float32Array<ArrayBuffer> | undefined,
  sigmaDelta: number,
  guidance: number,
): Float32Array<ArrayBuffer> => {
  const wants = needsUncond(guidance);
  if (wants && uncond === undefined) {
    throw new Error(`CFG scale ${guidance} は uncond 側の DiT 出力を要求する`);
  }
  if (!wants && uncond !== undefined) {
    throw new Error("CFG scale 1 で uncond が渡された（uncond 分岐を計算しない経路のはず）");
  }
  // Drop the scale to f32 first (torch drops a Python float scalar to f32 before the product with
  // an f32 tensor — multiplying in f64 makes the mix 1 ulp off for a scale f32 cannot represent,
  // such as 1.1).
  const scale = f32(guidance);
  const next = new Float32Array(previous.length);
  for (let index = 0; index < previous.length; index += 1) {
    const noise =
      uncond === undefined
        ? cond[index]
        : f32(uncond[index] + f32(scale * f32(cond[index] - uncond[index])));
    next[index] = f32(previous[index] + f32(sigmaDelta * noise));
  }
  return next;
};
