/**
 * Wan2.1 のサンプラのホスト側 — flow matching の UniPC（diffusers 0.39.0 の
 * `UniPCMultistepScheduler`）と CFG の合成（ADR 0118 決定 5）。
 *
 * 移植するのは Wan2.1 の scheduler config が通る分岐だけ: `use_flow_sigmas`・静的 shift・
 * `prediction_type = flow_prediction`・`predict_x0`・`final_sigmas_type = zero`・
 * `lower_order_final`・`disable_corrector = []`・`thresholding` なし・`solver_p` なし。
 * `solver_order` は 1 / 2、`solver_type` は bh1 / bh2 を受ける（Wan の値は 2 / bh2 —
 * {@link WAN_UNIPC_CONFIG}。1 次と bh1 は単体テストの故障注入が使う）。
 *
 * ## σ 列と timestep（`set_timesteps` の flow の分岐）
 *
 * σ は **f64 で計算してから f32 へ落とす**（決定 5）: numpy の `linspace(1, 1/N, steps+1)[:-1]`
 * → `shift·σ / (1 + (shift−1)·σ)` → σ[0] が 1 なら `1e-6` を引く → timestep は f64 の `σ·N` を
 * 切り捨てた整数 → 最後に 0 を足して f32。上流と同じ演算を同じ順で踏むので、f32 の列は上流と
 * ビット一致し、timestep は完全一致する（単体テストが fixture で縛る）。
 *
 * ## 更新（`step` / 予測子 `multistep_uni_p_bh_update` / 修正子 `multistep_uni_c_bh_update`）
 *
 * 上流は CPU の f32 テンソルで計算する。係数（σ の比・λ・h・expm1 …）は 0-d の f32 テンソルで、
 * 要素ごとの演算は 1 本ずつ別の kernel（融合しない）。ここでは同じ演算を同じ順で `Math.fround`
 * を 1 演算ずつ踏んで写す。
 *
 * ## MUST: 軌跡は参照とビット一致し**ない**（一致を期待して締めない）
 *
 * f32 の `log` / `expm1` は torch（SLEEF / libm）と JS の `Math.*`（f64 で計算して f32 へ丸める）で
 * 最終ビットが割れうる。修正子の 2×2 の連立（上流は `torch.linalg.solve` = LAPACK の LU）は
 * 参照実装の sgetrf2 + sgetrs の演算順で写したが、torch が呼ぶ実装（MKL）の内部の順は保証されない。
 * 突き合わせは実測から導いた atol で行う（`packages/models/tests/wan_scheduler_test.ts` — 実測値は
 * そちらの doc）。
 *
 * MUST: モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

const f32 = Math.fround;

/** 上流の flow の分岐が σ[0] から引く量（`set_timesteps` の `eps`）。 */
const SIGMA_EPSILON = 1e-6;

/** 予測子の 2 次の係数（上流の「order 2 は簡略版」— `rhos_p = 0.5`）と修正子の 1 次の係数。 */
const HALF = 0.5;

/** diffusers の scheduler config のうち、この移植が読む値。 */
export type WanUniPcConfig = {
  /** `num_train_timesteps`（σ の linspace の下端 `1/N` と timestep の `σ·N`）。 */
  readonly numTrainTimesteps: number;
  /** `solver_order`（多段の次数の上限）。 */
  readonly solverOrder: 1 | 2;
  /** `solver_type`（`B(h)` の選び方 — bh1 は `hh`・bh2 は `expm1(hh)`）。 */
  readonly solverType: "bh1" | "bh2";
};

/**
 * Wan2.1 T2V 1.3B の scheduler config（上流 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` の
 * `scheduler/scheduler_config.json` — pin した revision の値。fixture の `scheduler_config` と同じ）。
 *
 * NOTE: shift（`flow_shift` 3.0）はここに持たない — 生成の要求のノブ（`WanGenerateRequest.shift`）。
 */
export const WAN_UNIPC_CONFIG: WanUniPcConfig = {
  numTrainTimesteps: 1000,
  solverOrder: 2,
  solverType: "bh2",
};

/** σ 列と timestep 列（{@link wanUniPcSchedule}）。 */
export type WanUniPcSchedule = {
  /** f32 の σ（長さ `steps + 1`・最後は 0）。 */
  readonly sigmas: Float32Array<ArrayBuffer>;
  /** 整数の timestep（長さ `steps`・DiT の `timesteps_proj` へ渡す値）。 */
  readonly timesteps: readonly number[];
};

/**
 * 上流の flow の分岐の f64 の σ 列（σ[0] の補正の前・長さ `steps`）。
 *
 * numpy の `linspace(start, stop, num)` は `step = (stop − start) / (num − 1)` を先に作って
 * `i·step` に `start` を足す（2 回の丸め — JS も融合しない）。shift は
 * `(shift·σ) / (1 + (shift − 1)·σ)` の順（上流の式の評価順）。
 *
 * NOTE: `export` は単体テストが f64 の列と故障注入（σ[0] の補正を外す）を組むため（`mod.ts` /
 * サブパス面には出さない — ADR 0008）。
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
 * `set_timesteps(steps)` の σ 列（f32）と timestep 列（モジュール doc の「σ 列と timestep」）。
 *
 * MUST: 返す前に σ 列の構造を見る（正・狭義単調減少・最後が 0）。f32 へ落とした隣り合う σ が
 * 等しいと `h = 0` で更新が NaN になり、生成の途中で遠くの症状として出る。
 */
export const wanUniPcSchedule = (
  steps: number,
  shift: number,
  numTrainTimesteps: number,
): WanUniPcSchedule => {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`steps ${steps} が 1 以上の整数でない`);
  }
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new RangeError(`shift ${shift} が正の有限値でない`);
  }
  const sigmas64 = flowShiftedSigmas(steps, shift, numTrainTimesteps);
  // 上流の `np.fabs(sigmas[0] - 1) < eps` の分岐（最初の更新の `log(alpha)` の発散よけ）。
  if (Math.abs(sigmas64[0] - 1) < SIGMA_EPSILON) sigmas64[0] -= SIGMA_EPSILON;
  // timestep は f64 の σ·N を int64 へ切り捨てた値（f32 へ落とす前の σ から作る）。
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

/** 上流の `lambda = log(alpha) − log(sigma)`（flow の `alpha = 1 − σ`・f32 の 0-d テンソル）。 */
const lambdaOf = (sigma: number): number =>
  f32(f32(Math.log(f32(1 - sigma))) - f32(Math.log(sigma)));

/**
 * 2×2 の連立 `a·x = b` を f32 で解く（LAPACK 参照実装の sgetrf2 + sgetrs の演算順 — 部分 pivot・
 * L の列は pivot の逆数を掛ける・U の後退代入は除算）。
 */
const solve2x2 = (
  matrix: readonly [readonly [number, number], readonly [number, number]],
  rhs: readonly [number, number],
): readonly [number, number] => {
  let [[a00, a01], [a10, a11]] = matrix;
  let [b0, b1] = rhs;
  // isamax は同値なら先頭を選ぶので、入れ替えるのは厳密に大きいときだけ。
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

/** 予測子・修正子が共有する 1 区間 `[σ_s0 → σ_t]` の係数（0-d の f32 テンソルの写し）。 */
type Interval = {
  readonly h: number;
  readonly lambdaS0: number;
  /** `(σ_t / σ_s0)` — `x` に掛ける係数。 */
  readonly ratio: number;
  /** `alpha_t · h_phi_1` — `m0` に掛ける係数。 */
  readonly x0Weight: number;
  /** `alpha_t · B(h)` — 高次の補正に掛ける係数。 */
  readonly correctionWeight: number;
  readonly hh: number;
  readonly hPhi1: number;
  readonly bH: number;
};

/**
 * flow matching の UniPC（bh1 / bh2・次数 1 / 2）の状態機械。上流の `step` を 1 回呼ぶごとに
 * {@link WanUniPcSampler.step} を 1 回呼ぶ（step の添字は内部で数える — 上流の
 * `set_begin_index(0)` の後と同じ）。
 *
 * 状態は直近 `solverOrder` 本の x0 の予測（上流の `model_outputs`）・前の step の修正後の潜在
 * （`last_sample`）・warmup の数（`lower_order_nums`）・前の step の次数（`this_order`）。
 * 生成 1 本ごとに作り直す（上流の `set_timesteps` が状態を初期化するのと同じ）。
 */
export class WanUniPcSampler {
  readonly #sigmas: Float32Array;
  readonly #steps: number;
  readonly #config: WanUniPcConfig;
  #index = 0;
  #lowerOrderNums = 0;
  #thisOrder = 0;
  #lastSample: Float32Array | undefined;
  /** 直近の x0 の予測（古い順・長さは `solverOrder` まで）。 */
  #history: Float32Array[] = [];

  constructor(schedule: { readonly sigmas: Float32Array }, config: WanUniPcConfig) {
    if (schedule.sigmas.length < 2) {
      throw new RangeError(`σ 列の長さ ${schedule.sigmas.length} が 2 未満（steps ≥ 1 が要る）`);
    }
    this.#sigmas = schedule.sigmas;
    this.#steps = schedule.sigmas.length - 1;
    this.#config = config;
  }

  /** 済んだ step 数。 */
  get index(): number {
    return this.#index;
  }

  /**
   * 1 step 進める: `modelOutput`（flow の速度場 — CFG 合成済み）と現在の潜在 `sample` から次の
   * 潜在を返す。どちらも書き換えない。
   */
  step(modelOutput: Float32Array, sample: Float32Array): Float32Array<ArrayBuffer> {
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
    // convert_model_output（flow_prediction・predict_x0）: x0 = sample − σ·v（変換前の潜在で作る）。
    const sigma = this.#sigmas[index];
    const x0 = new Float32Array(sample.length);
    for (let element = 0; element < sample.length; element += 1) {
      x0[element] = f32(sample[element] - f32(sigma * modelOutput[element]));
    }
    const lastSample = this.#lastSample;
    const corrected = index > 0 && lastSample !== undefined
      ? this.#correct(x0, lastSample, this.#thisOrder)
      : sample;
    this.#history.push(x0);
    if (this.#history.length > solverOrder) this.#history.shift();
    // lower_order_final: 最後の数 step は残りの step 数まで次数を落とす。warmup は済んだ数 + 1 まで。
    const thisOrder = Math.min(solverOrder, this.#steps - index, this.#lowerOrderNums + 1);
    this.#thisOrder = thisOrder;
    this.#lastSample = corrected;
    const next = this.#predict(corrected, thisOrder);
    if (this.#lowerOrderNums < solverOrder) this.#lowerOrderNums += 1;
    this.#index += 1;
    return next;
  }

  /** 区間 `[σ[s0] → σ[t]]` の係数。 */
  #interval(t: number, s0: number): Interval {
    const sigmaT = this.#sigmas[t];
    const sigmaS0 = this.#sigmas[s0];
    const alphaT = f32(1 - sigmaT);
    const lambdaS0 = lambdaOf(sigmaS0);
    const h = f32(lambdaOf(sigmaT) - lambdaS0);
    // predict_x0 なので hh = −h。
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

  /** `rk = (λ_si − λ_s0) / h`（2 次の項の刻み比）。 */
  #ratioOf(si: number, interval: Interval): number {
    return f32(f32(lambdaOf(this.#sigmas[si]) - interval.lambdaS0) / interval.h);
  }

  /** 予測子（上流 `multistep_uni_p_bh_update` — 区間 `[σ[k] → σ[k+1]]`）。 */
  #predict(sample: Float32Array, order: number): Float32Array<ArrayBuffer> {
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
    // 1 次: 上流は `x_t_ − alpha_t·B_h·0`（整数 0 を掛けた 0-d テンソルを引く — ±0 の符号まで写す）。
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
   * 修正子（上流 `multistep_uni_c_bh_update` — 区間 `[σ[k−1] → σ[k]]`）。`modelT` はこの step の
   * x0・`lastSample` は前の step の修正後の潜在・`order` は前の step の次数。履歴（m0 = 前の step の
   * x0）は追加の**前**に読む。
   */
  #correct(modelT: Float32Array, lastSample: Float32Array, order: number): Float32Array {
    const index = this.#index;
    const interval = this.#interval(index, index - 1);
    const m0 = this.#history[this.#history.length - 1];
    // b_i = h_phi_k · i! / B_h（h_phi_k の漸化 — 上流の for 文の逐語）。
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
        // 上流は `0 + rhos_c[-1]·D1_t`（corr_res が整数 0）。
        const inner = f32(0 + f32(HALF * d1t));
        out[element] = f32(base - f32(interval.correctionWeight * inner));
      }
      return out;
    }
    const m1 = this.#history[this.#history.length - 2];
    const rk = this.#ratioOf(index - 2, interval);
    // R = stack([rks^0, rks^1]) = [[1, 1], [rk, 1]]（rks = [rk, 1]）。
    const [rho0, rho1] = solve2x2([[1, 1], [rk, 1]], [b[0], b[1]]);
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
 * CFG の合成 `uncond + g·(cond − uncond)`（上流 `pipeline_wan.py` の逐語 — f32 で 1 演算ずつ。
 * `g` は torch の Python スカラと同じく f32 へ落としてから掛ける）。
 */
export const wanClassifierFreeGuidance = (
  cond: Float32Array,
  uncond: Float32Array,
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
