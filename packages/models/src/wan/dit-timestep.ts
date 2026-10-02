/**
 * Wan2.1 の DiT の時刻埋め込みの sinusoidal 段のホスト側（グラフ入力 `timesteps_proj [1,freq_dim]` —
 * ADR 0118 決定 3）。MLP（`freq_dim → dim → dim → 6·dim`）はグラフの中に残る。
 *
 * 正本は上流 `WanTimeTextImageEmbedding.timesteps_proj` =
 * `Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)`（diffusers の
 * `get_timestep_embedding`）。**前半 cos・後半 sin**。入力は上流のパイプラインが渡す整数の timestep
 * （flow matching の UniPC の `σ·1000` を int64 に切り捨てた値 — 決定 5）。
 */

const f32 = Math.fround;

/** 上流 `get_timestep_embedding` の既定 `max_period`。 */
const MAX_PERIOD = 10000;

/**
 * 1 step ぶんの `timesteps_proj` `[width]`（上流の f32 の演算順を 1 演算ずつ写したもの）。
 *
 * 演算順: `exponent_i = f32(f32(−ln(max_period)) · i) / half`（f32 の除算）→ `f32(exp(exponent_i))`
 * → `angle_i = f32(f32(timestep) · freq_i)` → 前半 `cos(angle_i)`・後半 `sin(angle_i)`。
 * `-math.log(max_period)` は f32 テンソルとの積で f32 へ落ちる（torch のスカラ昇格 — Anima の
 * `timestepsProj` と同じ扱い）。
 *
 * ## MUST: この関数は参照とビット一致し**ない**（一致を期待して締めない）
 *
 * torch CPU の f32 `exp` / `sin` / `cos` は SLEEF の 1.0 ULP 実装で、JS の `Math.*`（f64 で計算して
 * f32 へ丸める）とは最終ビットが割れうる。突き合わせは実測から導いた atol で行う
 * （`packages/models/tests/wan_dit_host_test.ts` — 実測値はそちらの doc）。
 *
 * `width` はグラフ入力 `timesteps_proj` の静的次元から渡す（呼び出し側に 256 を書かない）。
 */
export const timestepsProj = (timestep: number, width: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(width) || width <= 0 || width % 2 !== 0) {
    throw new RangeError(`timesteps_proj の幅 ${width} が正の偶数でない`);
  }
  if (!Number.isInteger(timestep) || timestep < 0) {
    throw new RangeError(`timestep ${timestep} が 0 以上の整数でない（上流は int64 の timestep）`);
  }
  const half = width / 2;
  const time = f32(timestep);
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const out = new Float32Array(width);
  for (let index = 0; index < half; index += 1) {
    // downscale_freq_shift = 0 なので分母は half そのもの。
    const exponent = f32(f32(logMaxPeriod * index) / half);
    const angle = f32(time * f32(Math.exp(exponent)));
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};
