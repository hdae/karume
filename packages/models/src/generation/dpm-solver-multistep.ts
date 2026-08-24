/**
 * DPM++ 2M — multistep DPM-Solver++ の 2 次（**ファミリ非依存の共通処理**）。
 *
 * flow matching（`prediction_type="flow_prediction"`）の sigma 梯子の上を、直前 step の x0
 * 予測 1 本だけを覚えて 2 次で進む。梯子の作り方（静的 shift / dynamic shifting / 何 step か）は
 * **持たない** — 呼び出し側が作った梯子を配列で受けるだけなので、同じ更新式を使う画像 / 生成系
 * ならファミリを問わず載る。したがってファミリのディレクトリではなく `src/generation/` に置く
 * （`greedy.ts` と同じ位置づけ）。
 *
 * MUST: ファミリ側（`src/anima/` など）から何も import しない。梯子・CFG scale・step 数は
 * すべて引数で、この向きの依存（generation → ファミリ）を作った時点で共通処理でなくなる。
 * {@link needsUncond} が `src/anima/sampler.ts` の同名述語と同じ式なのはこの制約の帰結で、
 * 意図的な重複（1 行の不変条件を借りるために依存の向きを反転させない）。
 *
 * MUST: f32 の丸めを 1 演算ずつ `Math.fround` で踏む（`src/anima/sampler.ts` と同じ規律）。
 * JS の数値は f64 なので、まとめて計算してから丸めると torch の f32 逐次計算と最終桁が変わる。
 *
 * 正本は diffusers 0.39.0 の `DPMSolverMultistepScheduler`（`algorithm_type="dpmsolver++"` ×
 * `solver_order=2` × `solver_type="midpoint"` × `final_sigmas_type="zero"`）。golden を焼く
 * リグは `tools/export-recipes/anima/dpmsolver_ref.py`。
 */

const f32 = Math.fround;

/**
 * uncond 側（ネガティブプロンプトの forward）が要るか。
 *
 * `guidance === 1` のとき CFG 式 `uncond + 1·(cond − uncond)` は数学的に `cond` へ潰れるので、
 * uncond 側の計算は丸ごと不要になる。
 *
 * MUST: 素朴に「両方計算して合成する」に倒さない。浮動小数の丸めでは
 * `uncond + (cond − uncond)` が `cond` とビット一致する保証が無い（Sterbenz の補題は一般の値では
 * 成立しない）。
 */
export const needsUncond = (guidance: number): boolean => guidance !== 1;

/** {@link dpmSolverMultistepStep} の入力。 */
export type DpmSolverMultistepInput = {
  /** 現在の x（step 0 では初期ノイズ、以降は前 step の {@link DpmSolverMultistepUpdate.sample}）。 */
  readonly sample: Float32Array;
  /** cond 側のモデル出力（flow の速度場）。 */
  readonly cond: Float32Array;
  /** uncond 側のモデル出力。有無を {@link needsUncond} とちょうど一致させる。 */
  readonly uncond: Float32Array | undefined;
  /** CFG scale。 */
  readonly guidance: number;
  /** sigma 梯子。長さ `steps + 1`・**末尾は厳密に 0**（`final_sigmas_type="zero"` と対）。 */
  readonly sigmas: Float32Array;
  /** これから踏む step の添字（`0 ≤ index < steps`）。 */
  readonly index: number;
  /** 直前 step の {@link DpmSolverMultistepUpdate.x0}。`index === 0` でのみ `undefined`。 */
  readonly previousX0: Float32Array | undefined;
};

/** {@link dpmSolverMultistepStep} の出力。`x0` は次 step の `previousX0` にそのまま渡す。 */
export type DpmSolverMultistepUpdate = {
  /** 更新後の x。 */
  readonly sample: Float32Array<ArrayBuffer>;
  /** この step の x0 予測（`x − σ·v`）。状態はこの 1 本だけ。 */
  readonly x0: Float32Array<ArrayBuffer>;
};

/**
 * 1 step ぶんの DPM++ 2M 更新（CFG 合成 → x0 予測 → 1 次 / 2 次の状態更新）。
 *
 * 次数の切り替えは diffusers の `step()` の写し。`solver_order=2` では
 * `lower_order_second` が構造上到達しないので、**1 次に落ちるのは step 0（`lower_order_nums < 1`）と
 * 最終 step（`final_sigmas_type="zero"` ⇒ `lower_order_final` が常に真）の 2 つだけ**で、
 * 残りは全て 2 次。
 *
 * ## MUST: 梯子の端で出る ∓inf を前もって潰さない
 *
 * 梯子の先頭が厳密に 1・末尾が厳密に 0 なので、flow 経路の `alpha = 1 − σ` が step 0 で
 * `alphaS0 = 0`・最終 step で `sigmaT = 0` を作り、`lambda = log(alpha) − log(σ)` が ∓inf、
 * `h` が +inf になる。式は破綻せず ①step 0 は `exp(−h) = 0` で `x ← σ_t·x + α_t·x0`（Euler 1 段と
 * 同値）②step 1 は `lambdaS1 = −inf` から `r0 = inf → D1 = 0` で 2 次項が自然に落ちる
 * ③最終 step は係数が `(0, −1)` で `x ← x0` がビット同一、に落ちる。特別扱いを足すと**値が変わる**
 * （リグの実測どおりの経路を通らなくなる）。
 *
 * ## MUST: この関数は参照とビット一致し**ない**（実測 — 一致を期待して締めない）
 *
 * 係数を作る `log` / `exp` が torch CPU（SLEEF の 1.0 ULP 実装）と JS の `Math.*`（f64 で計算して
 * f32 へ丸める = 0.5 ULP 相当）で最終ビットが割れ、その差が**スカラ係数として全要素へ乗り**、
 * さらに x と x0 の 2 本の状態を通って step 間で累積する。実測（リグ golden との突合）:
 *
 * - パリティ用フィクスチャ（steps=8 / shift=3 / 4 要素 = 状態 32 個 + x0 予測 32 個）:
 *   **不一致 25/64・最大絶対差 3.576e-07・最大 8 ULP**。
 * - 掃引（steps ∈ {2,3,4,8,16,32,50} × shift ∈ {1,3,7} × 16 要素・seed 7 = 計 11,040 値）:
 *   **最大絶対差 1.192e-06**（steps=50 / shift=1）。step 数が増えるほど累積で伸びる。
 * - steps ≤ 4 は掃引した全 shift（1 / 3 / 7）で**ビット一致**（係数が厳密値だけで作れる格子点
 *   しか踏まない）。
 *   ここだけを見て「ビット一致する」と締めると、step を増やした瞬間に赤くなる。
 *
 * パリティテストはこの実測から導いた **atol 4e-6**（フィクスチャ実測最悪の約 11 倍・掃引最悪の
 * 約 3.4 倍）で固定する。実装バグの差は桁違いに大きい（フィクスチャ上の最大絶対差の実測:
 * **D0/D1 の係数取り違え 5.337e-01** = 閾値の 5.1 桁上 / **step 順序の入れ替え 2.034e+00** =
 * 5.7 桁上 / **midpoint の 0.5 落ち 1.748e-01** = 4.6 桁上 / **前 step の x0 の取り違え
 * 1.360e-01** = 4.5 桁上）ので、丸め差を許しても検出力は残る。
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
  // 状態の受け渡しが切れていたら黙って 1 次へ落ちない（1 次と 2 次は別の絵になる）。
  if ((index === 0) !== (previousX0 === undefined)) {
    throw new Error(
      `step ${index} と直前 x0 の有無が食い違う（step 0 でのみ undefined を渡す）`,
    );
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

  // 1 次に落ちる 2 つの step（上の doc）は直前 x0 を読まない — 最終 step は状態を受け取っても捨てる。
  const previous = index === 0 || index === steps - 1 ? undefined : previousX0;
  // 2 次項の係数。midpoint は `d1Scale = d0Scale/2`、`invR0` は前 step との刻み比の逆数。
  let d1Scale = 0;
  let invR0 = 0;
  if (previous !== undefined) {
    const sigmaS1 = sigmas[index - 1];
    const lambdaS1 = f32(f32(Math.log(f32(1 - sigmaS1))) - f32(Math.log(sigmaS1)));
    invR0 = f32(1 / f32(f32(lambdaS0 - lambdaS1) / h));
    d1Scale = f32(0.5 * d0Scale);
  }

  const next = new Float32Array(sample.length);
  const x0 = new Float32Array(sample.length);
  for (let element = 0; element < sample.length; element += 1) {
    const velocity = uncond === undefined
      ? cond[element]
      : f32(uncond[element] + f32(guidance * f32(cond[element] - uncond[element])));
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
