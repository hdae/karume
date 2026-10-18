/**
 * The host side of Wan2.1's sampler — flow matching UniPC (diffusers 0.39.0's
 * `UniPCMultistepScheduler`) and the CFG combination (ADR 0118 decision 5).
 *
 * Only the branches Wan2.1's scheduler config takes are ported: `use_flow_sigmas`, static shift,
 * `prediction_type = flow_prediction`, `predict_x0`, `final_sigmas_type = zero`,
 * `lower_order_final`, `disable_corrector = []`, no `thresholding`, no `solver_p`.
 * `solver_order` accepts 1 / 2 and `solver_type` bh1 / bh2 (Wan's values are 2 / bh2 —
 * {@link WAN_UNIPC_CONFIG}; order 1 and bh1 are used by the unit tests' fault injection).
 *
 * ## σ sequence and timesteps (the flow branch of `set_timesteps`)
 *
 * σ is **computed in f64 and then dropped to f32** (decision 5): numpy's
 * `linspace(1, 1/N, steps+1)[:-1]` → `shift·σ / (1 + (shift−1)·σ)` → subtract `1e-6` when σ[0] is 1
 * → the timestep is the f64 `σ·N` truncated to an integer → finally append 0 and convert to f32. The
 * same operations run in the same order as upstream, so the f32 sequence is bit-identical to
 * upstream and the timesteps match exactly (the unit tests pin this with a fixture).
 *
 * ## Update (`step` / predictor `multistep_uni_p_bh_update` / corrector `multistep_uni_c_bh_update`)
 *
 * Upstream computes with f32 tensors on the CPU. The coefficients (ratio of σ, λ, h, expm1 …) are
 * 0-d f32 tensors, and each elementwise operation is a separate kernel (no fusion). Here the same
 * operations are copied in the same order, applying `Math.fround` after every operation.
 *
 * ## MUST: the trajectory is **not** bit-identical to the reference (do not tighten to a match)
 *
 * f32 `log` / `expm1` can split in the last bit between torch (SLEEF / libm) and JS's `Math.*`
 * (computed in f64 and rounded to f32). The corrector's 2×2 system (upstream: `torch.linalg.solve`
 * = LAPACK's LU) is copied in the operation order of the reference sgetrf2 + sgetrs, but the
 * internal order of the implementation torch calls (MKL) is not guaranteed. The comparison uses an
 * atol derived from measurements (`packages/models/tests/wan_scheduler_test.ts` — the measured
 * values are in its doc).
 *
 * MUST: modules are side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

const f32 = Math.fround;

// Third-party code notice. `flowShiftedSigmas`, `wanUniPcSchedule`, `lambdaOf` and
// `WanUniPcSampler` below are adapted from `UniPCMultistepScheduler.set_timesteps`,
// `convert_model_output`, `step`, `multistep_uni_p_bh_update` and `multistep_uni_c_bh_update` in
// huggingface/diffusers (`src/diffusers/schedulers/scheduling_unipc_multistep.py`,
// `diffusers==0.39.0`). Only the branches the Wan2.1 scheduler config takes are carried over (see
// the module doc), each f32 operation is written out in the upstream order, and `solve2x2` stands
// in for `torch.linalg.solve`. License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
// header of that file:
//
//   Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.

/** The amount upstream's flow branch subtracts from σ[0] (`eps` in `set_timesteps`). */
const SIGMA_EPSILON = 1e-6;

/**
 * The predictor's order-2 coefficient (upstream's "order 2 is the simplified version" —
 * `rhos_p = 0.5`) and the corrector's order-1 coefficient.
 */
const HALF = 0.5;

/** The values of diffusers' scheduler config that this port reads. */
export type WanUniPcConfig = {
  /** `num_train_timesteps` (the lower end `1/N` of the σ linspace and `σ·N` of the timestep). */
  readonly numTrainTimesteps: number;
  /** `solver_order` (the upper limit of the multistep order). */
  readonly solverOrder: 1 | 2;
  /** `solver_type` (how `B(h)` is chosen — `hh` for bh1, `expm1(hh)` for bh2). */
  readonly solverType: "bh1" | "bh2";
};

/**
 * The scheduler config of Wan2.1 T2V 1.3B (`scheduler/scheduler_config.json` of upstream
 * `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` — the values at the pinned revision; the same as the fixture's
 * `scheduler_config`).
 *
 * NOTE: shift (`flow_shift` 3.0) is not kept here — it is a knob of the generation request
 * (`WanGenerateRequest.shift`).
 */
export const WAN_UNIPC_CONFIG: WanUniPcConfig = {
  numTrainTimesteps: 1000,
  solverOrder: 2,
  solverType: "bh2",
};

/** The σ sequence and the timestep sequence ({@link wanUniPcSchedule}). */
export type WanUniPcSchedule = {
  /** f32 σ (length `steps + 1`, the last is 0). */
  readonly sigmas: Float32Array<ArrayBuffer>;
  /** Integer timesteps (length `steps`, the values passed to the DiT's `timesteps_proj`). */
  readonly timesteps: readonly number[];
};

/**
 * The f64 σ sequence of upstream's flow branch (before the σ[0] correction, length `steps`).
 *
 * numpy's `linspace(start, stop, num)` first computes `step = (stop − start) / (num − 1)` and then
 * adds `start` to `i·step` (two roundings — JS does not fuse them either). The shift is evaluated as
 * `(shift·σ) / (1 + (shift − 1)·σ)` (the evaluation order of upstream's expression).
 *
 * NOTE: the `export` lets the unit tests build the f64 sequence and the fault injection (dropping the
 * σ[0] correction) (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const flowShiftedSigmas = (
  steps: number,
  shift: number,
  numTrainTimesteps: number,
): number[] => {
  const start = 1;
  const stop = 1 / numTrainTimesteps;
  const step = (stop - start) / steps;
  const sigmas: number[] = [];
  for (let index = 0; index < steps; index += 1) {
    const base = index * step + start;
    sigmas.push((shift * base) / (1 + (shift - 1) * base));
  }
  return sigmas;
};

/**
 * The σ sequence (f32) and the timestep sequence of `set_timesteps(steps)` (module doc, "σ sequence
 * and timesteps").
 *
 * MUST: check the structure of the σ sequence before returning it (positive, strictly decreasing,
 * last is 0). If two adjacent σ become equal after the drop to f32, `h = 0` makes the update NaN,
 * which shows up as a distant symptom midway through generation.
 *
 * MUST: the upper limit of steps is `numTrainTimesteps` (1000 for Wan), checked before allocating
 * the arrays. An integer timestep is `trunc(σ·N)`, so there are only N values 0 to N−1, and steps
 * above N always duplicate a timestep. Without the limit, a huge steps would run a synchronous loop
 * building the whole σ sequence (which cannot be cancelled either) before failing.
 */
export const wanUniPcSchedule = (
  steps: number,
  shift: number,
  numTrainTimesteps: number,
): WanUniPcSchedule => {
  if (!Number.isInteger(steps) || steps < 1 || steps > numTrainTimesteps) {
    throw new RangeError(
      `steps ${steps} が 1 以上 ${numTrainTimesteps} 以下の整数でない` +
        "（上限は num_train_timesteps — 超えると整数の timestep が必ず重複する）",
    );
  }
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new RangeError(`shift ${shift} が正の有限値でない`);
  }
  const sigmas64 = flowShiftedSigmas(steps, shift, numTrainTimesteps);
  // Upstream's `np.fabs(sigmas[0] - 1) < eps` branch (avoids divergence of `log(alpha)` in the
  // first update).
  if (Math.abs(sigmas64[0] - 1) < SIGMA_EPSILON) sigmas64[0] -= SIGMA_EPSILON;
  // The timestep is the f64 σ·N truncated to int64 (built from σ before the drop to f32).
  const timesteps = sigmas64.map((sigma) => Math.trunc(sigma * numTrainTimesteps));
  const sigmas = Float32Array.from([...sigmas64, 0]);
  for (let index = 0; index < steps; index += 1) {
    const current = sigmas[index];
    if (!(current > sigmas[index + 1]) || !(current <= 1)) {
      throw new RangeError(
        `σ 列が (0, 1] で狭義単調減少でない（steps ${steps}・shift ${shift}・` +
          `σ[${index}] = ${current}・σ[${index + 1}] = ${sigmas[index + 1]}）`,
      );
    }
  }
  return { sigmas, timesteps };
};

/** Upstream's `lambda = log(alpha) − log(sigma)` (flow's `alpha = 1 − σ`, an f32 0-d tensor). */
const lambdaOf = (sigma: number): number =>
  f32(f32(Math.log(f32(1 - sigma))) - f32(Math.log(sigma)));

/**
 * Solves the 2×2 system `a·x = b` in f32 (the operation order of the LAPACK reference sgetrf2 +
 * sgetrs — partial pivoting, L's column multiplied by the pivot's reciprocal, U's back substitution
 * by division).
 */
const solve2x2 = (
  matrix: readonly [readonly [number, number], readonly [number, number]],
  rhs: readonly [number, number],
): readonly [number, number] => {
  let [[a00, a01], [a10, a11]] = matrix;
  let [b0, b1] = rhs;
  // isamax picks the first on a tie, so swap only when strictly larger.
  if (Math.abs(a10) > Math.abs(a00)) {
    [a00, a01, a10, a11] = [a10, a11, a00, a01];
    [b0, b1] = [b1, b0];
  }
  const lower = f32(a10 * f32(1 / a00));
  const upper11 = f32(a11 - f32(lower * a01));
  const y1 = f32(b1 - f32(b0 * lower));
  const x1 = f32(y1 / upper11);
  const x0 = f32(f32(b0 - f32(x1 * a01)) / a00);
  return [x0, x1];
};

/** The coefficients of one interval `[σ_s0 → σ_t]` shared by predictor and corrector (copies of 0-d f32 tensors). */
type Interval = {
  readonly h: number;
  readonly lambdaS0: number;
  /** `(σ_t / σ_s0)` — the coefficient applied to `x`. */
  readonly ratio: number;
  /** `alpha_t · h_phi_1` — the coefficient applied to `m0`. */
  readonly x0Weight: number;
  /** `alpha_t · B(h)` — the coefficient applied to the higher-order correction. */
  readonly correctionWeight: number;
  readonly hh: number;
  readonly hPhi1: number;
  readonly bH: number;
};

/**
 * The state machine of flow matching UniPC (bh1 / bh2, order 1 / 2). Call
 * {@link WanUniPcSampler.step} once for every call of upstream's `step` (the step index is counted
 * internally — the same as after upstream's `set_begin_index(0)`).
 *
 * The state is the x0 predictions of the last `solverOrder` steps (upstream's `model_outputs`), the
 * previous step's corrected latent (`last_sample`), the warmup count (`lower_order_nums`) and the
 * previous step's order (`this_order`). It is rebuilt for every generation (just as upstream's
 * `set_timesteps` initializes the state).
 */
export class WanUniPcSampler {
  readonly #sigmas: Float32Array<ArrayBuffer>;
  readonly #steps: number;
  readonly #config: WanUniPcConfig;
  #index = 0;
  #lowerOrderNums = 0;
  #thisOrder = 0;
  #lastSample: Float32Array<ArrayBuffer> | undefined;
  /** The latest x0 predictions (oldest first, length up to `solverOrder`). */
  #history: Float32Array<ArrayBuffer>[] = [];

  constructor(schedule: { readonly sigmas: Float32Array<ArrayBuffer> }, config: WanUniPcConfig) {
    if (schedule.sigmas.length < 2) {
      throw new RangeError(`σ 列の長さ ${schedule.sigmas.length} が 2 未満（steps ≥ 1 が要る）`);
    }
    this.#sigmas = schedule.sigmas;
    this.#steps = schedule.sigmas.length - 1;
    this.#config = config;
  }

  /** The number of steps done. */
  get index(): number {
    return this.#index;
  }

  /**
   * Advances one step: returns the next latent from `modelOutput` (flow's velocity field — already
   * CFG-combined) and the current latent `sample`. Modifies neither.
   */
  step(
    modelOutput: Float32Array<ArrayBuffer>,
    sample: Float32Array<ArrayBuffer>,
  ): Float32Array<ArrayBuffer> {
    const index = this.#index;
    if (index >= this.#steps) {
      throw new RangeError(`step ${index + 1} 回目は σ 列（${this.#steps} step）の外`);
    }
    if (modelOutput.length !== sample.length) {
      throw new RangeError(
        `モデル出力 ${modelOutput.length} 要素と潜在 ${sample.length} 要素が違う`,
      );
    }
    const { solverOrder } = this.#config;
    // convert_model_output (flow_prediction, predict_x0): x0 = sample − σ·v (built from the latent
    // before conversion).
    const sigma = this.#sigmas[index];
    const x0 = new Float32Array(sample.length);
    for (let element = 0; element < sample.length; element += 1) {
      x0[element] = f32(sample[element] - f32(sigma * modelOutput[element]));
    }
    const lastSample = this.#lastSample;
    const corrected =
      index > 0 && lastSample !== undefined
        ? this.#correct(x0, lastSample, this.#thisOrder)
        : sample;
    this.#history.push(x0);
    if (this.#history.length > solverOrder) this.#history.shift();
    // lower_order_final: the last few steps lower the order to the number of remaining steps. During
    // warmup the order is at most the steps done + 1.
    const thisOrder = Math.min(solverOrder, this.#steps - index, this.#lowerOrderNums + 1);
    this.#thisOrder = thisOrder;
    this.#lastSample = corrected;
    const next = this.#predict(corrected, thisOrder);
    if (this.#lowerOrderNums < solverOrder) this.#lowerOrderNums += 1;
    this.#index += 1;
    return next;
  }

  /** The coefficients of the interval `[σ[s0] → σ[t]]`. */
  #interval(t: number, s0: number): Interval {
    const sigmaT = this.#sigmas[t];
    const sigmaS0 = this.#sigmas[s0];
    const alphaT = f32(1 - sigmaT);
    const lambdaS0 = lambdaOf(sigmaS0);
    const h = f32(lambdaOf(sigmaT) - lambdaS0);
    // predict_x0, so hh = −h.
    const hh = -h;
    const hPhi1 = f32(Math.expm1(hh));
    const bH = this.#config.solverType === "bh2" ? f32(Math.expm1(hh)) : hh;
    return {
      h,
      lambdaS0,
      ratio: f32(sigmaT / sigmaS0),
      x0Weight: f32(alphaT * hPhi1),
      correctionWeight: f32(alphaT * bH),
      hh,
      hPhi1,
      bH,
    };
  }

  /** `rk = (λ_si − λ_s0) / h` (the step ratio of the order-2 term). */
  #ratioOf(si: number, interval: Interval): number {
    return f32(f32(lambdaOf(this.#sigmas[si]) - interval.lambdaS0) / interval.h);
  }

  /** The predictor (upstream `multistep_uni_p_bh_update` — interval `[σ[k] → σ[k+1]]`). */
  #predict(sample: Float32Array<ArrayBuffer>, order: number): Float32Array<ArrayBuffer> {
    const index = this.#index;
    const interval = this.#interval(index + 1, index);
    const m0 = this.#history[this.#history.length - 1];
    const out = new Float32Array(sample.length);
    if (order === 2) {
      const m1 = this.#history[this.#history.length - 2];
      const rk = this.#ratioOf(index - 1, interval);
      for (let element = 0; element < sample.length; element += 1) {
        const d1 = f32(f32(m1[element] - m0[element]) / rk);
        const predRes = f32(HALF * d1);
        const base = f32(
          f32(interval.ratio * sample[element]) - f32(interval.x0Weight * m0[element]),
        );
        out[element] = f32(base - f32(interval.correctionWeight * predRes));
      }
      return out;
    }
    // Order 1: upstream computes `x_t_ − alpha_t·B_h·0` (subtracts a 0-d tensor multiplied by the
    // integer 0 — copied down to the sign of ±0).
    // oxlint-disable-next-line oxc/erasing-op -- deliberate: multiplying by 0 keeps the sign of ±0 (and NaN) that upstream subtracts
    const zero = f32(interval.correctionWeight * 0);
    for (let element = 0; element < sample.length; element += 1) {
      const base = f32(
        f32(interval.ratio * sample[element]) - f32(interval.x0Weight * m0[element]),
      );
      out[element] = f32(base - zero);
    }
    return out;
  }

  /**
   * The corrector (upstream `multistep_uni_c_bh_update` — interval `[σ[k−1] → σ[k]]`). `modelT` is
   * this step's x0, `lastSample` the previous step's corrected latent, `order` the previous step's
   * order. The history (m0 = the previous step's x0) is read **before** the append.
   */
  #correct(
    modelT: Float32Array<ArrayBuffer>,
    lastSample: Float32Array<ArrayBuffer>,
    order: number,
  ): Float32Array<ArrayBuffer> {
    const index = this.#index;
    const interval = this.#interval(index, index - 1);
    const m0 = this.#history[this.#history.length - 1];
    // b_i = h_phi_k · i! / B_h (the recurrence of h_phi_k — verbatim from upstream's for loop).
    let hPhiK = f32(f32(interval.hPhi1 / interval.hh) - 1);
    let factorial = 1;
    const b: number[] = [];
    for (let i = 1; i <= order; i += 1) {
      b.push(f32(f32(hPhiK * factorial) / interval.bH));
      factorial *= i + 1;
      hPhiK = f32(f32(hPhiK / interval.hh) - f32(1 / factorial));
    }
    const out = new Float32Array(lastSample.length);
    if (order === 1) {
      for (let element = 0; element < out.length; element += 1) {
        const base = f32(
          f32(interval.ratio * lastSample[element]) - f32(interval.x0Weight * m0[element]),
        );
        const d1t = f32(modelT[element] - m0[element]);
        // Upstream computes `0 + rhos_c[-1]·D1_t` (corr_res is the integer 0).
        const inner = f32(0 + f32(HALF * d1t));
        out[element] = f32(base - f32(interval.correctionWeight * inner));
      }
      return out;
    }
    const m1 = this.#history[this.#history.length - 2];
    const rk = this.#ratioOf(index - 2, interval);
    // R = stack([rks^0, rks^1]) = [[1, 1], [rk, 1]] (rks = [rk, 1]).
    const [rho0, rho1] = solve2x2(
      [
        [1, 1],
        [rk, 1],
      ],
      [b[0], b[1]],
    );
    for (let element = 0; element < out.length; element += 1) {
      const base = f32(
        f32(interval.ratio * lastSample[element]) - f32(interval.x0Weight * m0[element]),
      );
      const d1 = f32(f32(m1[element] - m0[element]) / rk);
      const corrRes = f32(rho0 * d1);
      const d1t = f32(modelT[element] - m0[element]);
      const inner = f32(corrRes + f32(rho1 * d1t));
      out[element] = f32(base - f32(interval.correctionWeight * inner));
    }
    return out;
  }
}

/**
 * The CFG combination `uncond + g·(cond − uncond)` (verbatim upstream `pipeline_wan.py` — one f32
 * operation at a time; `g` is dropped to f32 before multiplying, as with torch's Python scalars).
 */
export const wanClassifierFreeGuidance = (
  cond: Float32Array<ArrayBuffer>,
  uncond: Float32Array<ArrayBuffer>,
  guidance: number,
): Float32Array<ArrayBuffer> => {
  if (cond.length !== uncond.length) {
    throw new RangeError(`CFG: cond ${cond.length} 要素と uncond ${uncond.length} 要素が違う`);
  }
  const scale = f32(guidance);
  const out = new Float32Array(cond.length);
  for (let element = 0; element < cond.length; element += 1) {
    out[element] = f32(uncond[element] + f32(scale * f32(cond[element] - uncond[element])));
  }
  return out;
};
