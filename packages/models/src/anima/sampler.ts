/**
 * Anima のサンプラ — IR に載らない段を TS で持つ。
 *
 * 中身は 4 つ: ①`FlowMatchEulerDiscreteScheduler` の sigma 列 ②timestep の正弦波埋め込み
 * ③CFG の要否判定 ④CFG 合成 + Euler 更新。
 *
 * MUST: f32 の丸めを 1 演算ずつ `Math.fround` で踏む。JS の数値は f64 なので、まとめて
 * 計算してから丸めると torch / numpy の f32 逐次計算と最終桁が変わる。
 *
 * NOTE: `shift` / `numTrainTimesteps` は **manifest の `pipelineConfig.scheduler` が正本**
 * （配布物ごとに変わりうる数なので、ここに定数として持たない — ADR 0038 §1）。
 */

const f32 = Math.fround;

/** `get_timestep_embedding` の `max_period`（diffusers の既定 10000）。 */
const MAX_PERIOD = 10000;

/**
 * `FlowMatchEulerDiscreteScheduler.set_timesteps`（静的 shift 経路）の TS 実装。
 * 末尾に終端 0 を 1 つ足すので長さは `steps + 1`。
 *
 * 組み方は numpy の `linspace` の写し（`step = (stop − start) / (num − 1)` を先に作って
 * `i · step + start`、最終要素は `stop` を直接代入）。
 *
 * NOTE（故障注入の実測）: steps=32 / shift=3 では ①`(i · delta) / div + start` への並べ替え
 * ②f32 への丸めを 1 演算ずつではなく最後にまとめる、のどちらも **33 要素が 1 ビットも
 * 変わらない**。この式は値域 [0.088, 1] に収まる well-conditioned な形で、丸めが効くほど桁が
 * 落ちない。**パリティテストが固定しているのは値であって式の形ではない** — 写しにしてあるのは
 * 上流の式と 1 対 1 で追えるようにするためで、丸めの実測に基づく制約ではない。実際に落ちるのは
 * linspace の端点・shift・終端 0 の取り違え（故障注入で確認）。
 */
export const sigmaSchedule = (steps: number, shift: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(steps) || steps < 2) {
    throw new RangeError(`steps ${steps} が 2 以上の整数でない（linspace の分母が 0 になる）`);
  }
  const start = 1;
  const stop = 1 / steps;
  const step = (stop - start) / (steps - 1);
  const sigmas = new Float32Array(steps + 1);
  for (let index = 0; index < steps; index += 1) {
    const sigma = f32(index === steps - 1 ? stop : index * step + start);
    sigmas[index] = f32(f32(shift * sigma) / f32(f32((shift - 1) * sigma) + 1));
  }
  return sigmas;
};

/**
 * 1 step ぶんの timestep 埋め込み `[width]`（`Timesteps(2048, flip_sin_to_cos=True,
 * downscale_freq_shift=0.0)` = `get_timestep_embedding` の写し）。**前半 cos・後半 sin**。
 *
 * 入力は sigma そのもの。上流は `timesteps[i] / num_train_timesteps` と往復するので順序を
 * 写してあるが、**f32 の往復は 1 ビットも変えない**（エクスポータ側の実測 NOTE と同じ）。
 *
 * `width` はグラフ入力 `timesteps_proj` の静的次元から渡す（呼び出し側に 2048 を書かない）。
 *
 * ## MUST: この関数は参照とビット一致し**ない**（実測 — 一致を期待して締めない）
 *
 * torch CPU の f32 `exp` / `sin` / `cos` は SLEEF の 1.0 ULP 実装で、JS の `Math.*`（f64 で
 * 計算して f32 へ丸める = 0.5 ULP 相当）とは最終ビットが割れる。実測（参照フィクスチャ 3 本 =
 * 計 72 行）: 不一致 4.01〜4.03% / 最大 2 ULP / 最大絶対差 5.96e-8。内訳は `exp` 段が 1,024 件中
 * 9 件（1 ULP）で、残りは `sin` / `cos` 段。ビット一致させるには SLEEF の多項式近似を写す必要が
 * あり、値域 [-1,1] の表に対して 6e-8 の差のために持ち込む複雑さではない。パリティテストは
 * この実測から導いた atol 6e-7（実測最悪の約 10 倍）で固定する — 実装バグの差は桁違いに大きい
 * （実測: **行ずれ 1.68e-1** = 閾値の 5.4 桁上 / **cos·sin 反転 1.00e+0** = 6.2 桁上）ので、
 * 丸め差を許しても検出力は残る。
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
  // `-math.log(max_period)` は f32 テンソルとの積で f32 へ落ちる（torch のスカラ昇格）。
  // f64 のまま掛けると不一致が 825 → 2,503 件・最大 17 ULP へ増えるのを実測済み。
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const out = new Float32Array(width);
  for (let index = 0; index < half; index += 1) {
    // downscale_freq_shift = 0 なので分母は half そのもの。
    const exponent = f32(f32(logMaxPeriod * index) / half);
    const angle = f32(timestep * f32(Math.exp(exponent)));
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};

/**
 * uncond 側（ネガティブプロンプトの text encode と、毎 step の DiT forward）が要るか。
 *
 * `guidance === 1` のとき CFG 式 `uncond + 1·(cond − uncond)` は数学的に `cond` へ潰れるので、
 * uncond 側の計算は**丸ごと不要**になる（turbo 運用の 2× の出所は step 削減とは独立にここ）。
 *
 * MUST: 素朴に「両方計算して合成する」に倒さない。浮動小数の丸めでは
 * `uncond + (cond − uncond)` が `cond` とビット一致する保証が無く（Sterbenz の補題は一般の
 * 値では成立しない）、参照フィクスチャ側も CFG=1 では uncond 分岐を実行していない。
 */
export const needsUncond = (guidance: number): boolean => guidance !== 1;

/**
 * CFG 混合（`uncond + scale·(cond − uncond)`）と Euler 更新（`x + Δσ·noise`）。
 *
 * MUST: `uncond` の有無は {@link needsUncond} とちょうど一致させる（食い違いは fail loudly）。
 * 「CFG=1 なのに uncond を渡す」は呼び出し側の分岐漏れで、素通しすると
 * **参照と 1 ULP 級で割れた値が「GPU の誤差」として tolerance に吸われる**。
 */
export const cfgEulerStep = (
  previous: Float32Array,
  cond: Float32Array,
  uncond: Float32Array | undefined,
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
  const next = new Float32Array(previous.length);
  for (let index = 0; index < previous.length; index += 1) {
    const noise = uncond === undefined
      ? cond[index]
      : f32(uncond[index] + f32(guidance * f32(cond[index] - uncond[index])));
    next[index] = f32(previous[index] + f32(sigmaDelta * noise));
  }
  return next;
};
