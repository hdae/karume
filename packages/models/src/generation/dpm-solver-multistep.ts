/**
 * DPM++ 2M — second-order multistep DPM-Solver++ (**family-independent shared code**).
 *
 * It advances at second order along the sigma ladder of flow matching
 * (`prediction_type="flow_prediction"`), remembering only the previous step's single x0
 * prediction. It does **not own** how the ladder is built (static shift / dynamic shifting / how many
 * steps) — it only takes the ladder the caller built as an array, so any image / generative family
 * that uses the same update rule can run on it. That is why it lives in `src/generation/` and not in
 * a family directory (the same standing as `greedy.ts`).
 *
 * MUST: import nothing from the family side (`src/anima/` and so on). The ladder, CFG scale and step
 * count are all arguments; the moment a dependency in this direction (generation → family) appears,
 * this stops being shared code. {@link needsUncond} has a form that can be shared under this
 * constraint (a one-line invariant independent of the update rule), so **its one definition is
 * here** and the family side (`src/anima/sampler.ts`) imports it — the dependency still points
 * family → generation, and no copy is kept.
 *
 * MUST: apply f32 rounding one operation at a time with `Math.fround` (the same discipline as
 * `src/anima/sampler.ts`). JS numbers are f64, so computing in bulk and rounding afterwards changes
 * the last digit against torch's sequential f32 computation.
 *
 * The source of truth is diffusers 0.39.0's `DPMSolverMultistepScheduler`
 * (`algorithm_type="dpmsolver++"` × `solver_order=2` × `solver_type="midpoint"` ×
 * `final_sigmas_type="zero"`). The rig that generates the golden is
 * `tools/export-recipes/anima/dpmsolver_ref.py`.
 */

const f32 = Math.fround;

/**
 * Whether the uncond side (the negative prompt's forward) is needed.
 *
 * With `guidance === 1` the CFG formula `uncond + 1·(cond − uncond)` collapses mathematically to
 * `cond`, so the uncond computation is not needed at all.
 *
 * MUST: do not fall back to the naive "compute both and combine". Under floating-point rounding
 * there is no guarantee that `uncond + (cond − uncond)` is bit-identical to `cond` (Sterbenz's lemma
 * does not hold for general values).
 */
export const needsUncond = (guidance: number): boolean => guidance !== 1;

/** Input of {@link dpmSolverMultistepStep}. */
export type DpmSolverMultistepInput = {
  /**
   * The current x (the initial noise at step 0, afterwards the previous step's
   * {@link DpmSolverMultistepUpdate.sample}).
   */
  readonly sample: Float32Array<ArrayBuffer>;
  /** Model output of the cond side (the flow's velocity field). */
  readonly cond: Float32Array<ArrayBuffer>;
  /** Model output of the uncond side. Its presence must match {@link needsUncond} exactly. */
  readonly uncond: Float32Array<ArrayBuffer> | undefined;
  /** CFG scale. */
  readonly guidance: number;
  /**
   * The sigma ladder. Length `steps + 1`, **the last entry exactly 0** (paired with
   * `final_sigmas_type="zero"`).
   */
  readonly sigmas: Float32Array<ArrayBuffer>;
  /** Index of the step about to be taken (`0 ≤ index < steps`). */
  readonly index: number;
  /** The previous step's {@link DpmSolverMultistepUpdate.x0}. `undefined` only when `index === 0`. */
  readonly previousX0: Float32Array<ArrayBuffer> | undefined;
};

/** Output of {@link dpmSolverMultistepStep}. Pass `x0` unchanged as the next step's `previousX0`. */
export type DpmSolverMultistepUpdate = {
  /** The updated x. */
  readonly sample: Float32Array<ArrayBuffer>;
  /** This step's x0 prediction (`x − σ·v`). This is the only state. */
  readonly x0: Float32Array<ArrayBuffer>;
};

// Third-party code notice. `dpmSolverMultistepStep` below is adapted from
// `DPMSolverMultistepScheduler.convert_model_output`, `dpm_solver_first_order_update`,
// `multistep_dpm_solver_second_order_update` and `step` in huggingface/diffusers
// (`src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`, `diffusers==0.39.0`), specialized
// to `algorithm_type="dpmsolver++"`, `solver_order=2`, `solver_type="midpoint"`,
// `final_sigmas_type="zero"` and `prediction_type="flow_prediction"`, with each f32 operation
// written out in the upstream order and the CFG combination folded into the same loop. License:
// Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright
// notice, copied verbatim from the header of that file:
//
//   Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.

/**
 * One step of the DPM++ 2M update (CFG combination → x0 prediction → first / second-order state
 * update).
 *
 * The order switch is a copy of diffusers' `step()`. With `solver_order=2`, `lower_order_second` is
 * structurally unreachable, so **only two steps drop to first order: step 0 (`lower_order_nums < 1`)
 * and the final step (`final_sigmas_type="zero"` ⇒ `lower_order_final` is always true)**; all the
 * others are second order.
 *
 * ## MUST: do not pre-empt the ∓inf that appears at the ends of the ladder
 *
 * The ladder starts at exactly 1 and ends at exactly 0, so the flow path's `alpha = 1 − σ` makes
 * `alphaS0 = 0` at step 0 and `sigmaT = 0` at the final step; `lambda = log(alpha) − log(σ)` becomes
 * ∓inf and `h` becomes +inf. The formulas do not break down: (1) step 0 has `exp(−h) = 0`, giving
 * `x ← σ_t·x + α_t·x0` (equivalent to one Euler step); (2) step 1 has `lambdaS1 = −inf`, so
 * `r0 = inf → D1 = 0` and the second-order term drops out naturally; (3) the final step has
 * coefficients `(0, −1)`, and `x ← x0` is bit-identical. Adding special cases **changes the values**
 * (the path no longer follows what the rig measured).
 *
 * ## MUST: this function is **not** bit-identical to the reference (measured — do not tighten a tolerance expecting a match)
 *
 * The `log` / `exp` that build the coefficients split in the last bit between torch CPU (SLEEF's
 * 1.0 ULP implementation) and JS `Math.*` (computed in f64 and rounded to f32 = about 0.5 ULP); the
 * difference **multiplies every element as a scalar coefficient** and then accumulates across steps
 * through the two states x and x0. Measured (against the rig's golden):
 *
 * - Parity fixture (steps=8 / shift=3 / 4 elements = 32 states + 32 x0 predictions):
 *   **25/64 mismatches, max absolute difference 3.576e-07, at most 8 ULP**.
 * - Sweep (steps ∈ {2,3,4,8,16,32,50} × shift ∈ {1,3,7} × 16 elements, seed 7 = 11,040 values in
 *   total): **max absolute difference 1.192e-06** (steps=50 / shift=1). It grows cumulatively with
 *   the step count.
 * - steps ≤ 4 is **bit-identical** for every swept shift (1 / 3 / 7) (it only touches lattice points
 *   whose coefficients are built from exact values).
 *   Concluding "it is bit-identical" from this case alone turns red the moment the step count grows.
 *
 * The parity test is pinned at **atol 4e-6**, derived from these measurements (about 11× the worst
 * fixture measurement and about 3.4× the worst sweep value). Implementation bugs differ by orders
 * of magnitude more (measured max absolute difference on the fixture: **D0/D1 coefficients swapped
 * 5.337e-01** = 5.1 orders above the threshold / **step order swapped 2.034e+00** = 5.7 orders above /
 * **midpoint's 0.5 dropped 1.748e-01** = 4.6 orders above / **the previous step's x0 mixed up
 * 1.360e-01** = 4.5 orders above), so detection power remains even with the rounding difference
 * allowed.
 */
export const dpmSolverMultistepStep = (
  input: DpmSolverMultistepInput,
): DpmSolverMultistepUpdate => {
  const { sample, cond, uncond, guidance, sigmas, index, previousX0 } = input;
  const steps = sigmas.length - 1;
  if (steps < 1) {
    throw new RangeError(`sigma 梯子の長さ ${sigmas.length} が 2 未満（step が 1 つも無い）`);
  }
  if (sigmas[steps] !== 0) {
    throw new RangeError(
      `sigma 梯子の終端 ${sigmas[steps]} が 0 でない（最終 step の 1 次落ちが成り立たない）`,
    );
  }
  if (!Number.isInteger(index) || index < 0 || index >= steps) {
    throw new RangeError(`step の添字 ${index} が [0, ${steps}) の整数でない`);
  }
  const wants = needsUncond(guidance);
  if (wants && uncond === undefined) {
    throw new Error(`CFG scale ${guidance} は uncond 側のモデル出力を要求する`);
  }
  if (!wants && uncond !== undefined) {
    throw new Error("CFG scale 1 で uncond が渡された（uncond 分岐を計算しない経路のはず）");
  }
  // If the state hand-off is broken, do not silently drop to first order (first and second order
  // give different pictures).
  if ((index === 0) !== (previousX0 === undefined)) {
    throw new Error(`step ${index} と直前 x0 の有無が食い違う（step 0 でのみ undefined を渡す）`);
  }

  const sigmaS0 = sigmas[index];
  const sigmaT = sigmas[index + 1];
  const alphaT = f32(1 - sigmaT);
  const alphaS0 = f32(1 - sigmaS0);
  const lambdaT = f32(f32(Math.log(alphaT)) - f32(Math.log(sigmaT)));
  const lambdaS0 = f32(f32(Math.log(alphaS0)) - f32(Math.log(sigmaS0)));
  const h = f32(lambdaT - lambdaS0);
  const sampleScale = f32(sigmaT / sigmaS0);
  const d0Scale = f32(alphaT * f32(f32(Math.exp(-h)) - 1));

  // The two steps that drop to first order (doc above) do not read the previous x0 — the final step
  // discards the state even when it receives one.
  const previous = index === 0 || index === steps - 1 ? undefined : previousX0;
  // Second-order term coefficients. Midpoint has `d1Scale = d0Scale/2`; `invR0` is the reciprocal
  // of the step-size ratio to the previous step.
  let d1Scale = 0;
  let invR0 = 0;
  if (previous !== undefined) {
    const sigmaS1 = sigmas[index - 1];
    const lambdaS1 = f32(f32(Math.log(f32(1 - sigmaS1))) - f32(Math.log(sigmaS1)));
    invR0 = f32(1 / f32(f32(lambdaS0 - lambdaS1) / h));
    d1Scale = f32(0.5 * d0Scale);
  }

  // Round the CFG scale to f32 first (torch rounds a Python float scalar to f32 before multiplying
  // it with an f32 tensor — multiplying in f64 shifts the combination by 1 ulp for a scale such as
  // 1.1 that f32 cannot represent).
  const scale = f32(guidance);
  const next = new Float32Array(sample.length);
  const x0 = new Float32Array(sample.length);
  for (let element = 0; element < sample.length; element += 1) {
    const velocity =
      uncond === undefined
        ? cond[element]
        : f32(uncond[element] + f32(scale * f32(cond[element] - uncond[element])));
    const predicted = f32(sample[element] - f32(sigmaS0 * velocity));
    x0[element] = predicted;
    const base = f32(f32(sampleScale * sample[element]) - f32(d0Scale * predicted));
    if (previous === undefined) {
      next[element] = base;
      continue;
    }
    const d1 = f32(invR0 * f32(predicted - previous[element]));
    next[element] = f32(base - f32(d1Scale * d1));
  }
  return { sample: next, x0 };
};
