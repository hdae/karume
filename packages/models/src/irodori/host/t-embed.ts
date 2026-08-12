/**
 * `t_embed` のホスト生成（上流 `get_timestep_embedding` の写し）。
 *
 * グラフに載せられないのは `cos` が IR の op 語彙に無いため（`sin` だけを足した第 1 層の
 * 判断 — ADR 0043 / 0047 決定 4）。
 *
 * MUST: f32 の丸めを 1 演算ずつ `Math.fround` で踏む。JS の数値は f64 なので、まとめて
 * 計算してから丸めると torch の f32 逐次計算と最終桁が変わる（`anima/sampler.ts` の同節）。
 *
 * ## NOTE: 参照とビット一致は**しない**（一致を期待して締めない）
 *
 * torch CPU の f32 `exp` / `sin` / `cos` は SLEEF の 1.0 ULP 実装で、JS の `Math.*`（f64 で
 * 計算して f32 へ丸める）とは最終ビットが割れる。突合は golden 表からの実測で導いた atol
 * （テスト側の定数 docstring）で締める。
 */

const f32 = Math.fround;

/** 上流の `max_period`（`1000 · exp(−ln(10000)·k/half)` の 10000）。 */
const MAX_PERIOD = 10000;
/** 周波数の全体倍率（上流の `1000.0`）。 */
const FREQ_SCALE = 1000;

/**
 * 周波数表 `[dim/2]`。t に依らないので 1 回作って使い回す（40 step ぶんの再計算を避ける）。
 *
 * 式の順序も上流の写し: `−log(10000)` を f32 へ落としてから `k` を掛け、`half` で割り、
 * `exp` してから `1000` を掛ける。まとめて f64 で計算すると最終ビットが変わる。
 */
export const timestepFrequencies = (dim: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new RangeError(`t_embed の幅 ${dim} が正の偶数でない`);
  }
  const half = dim / 2;
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const freqs = new Float32Array(half);
  for (let index = 0; index < half; index += 1) {
    const exponent = f32(f32(logMaxPeriod * index) / half);
    freqs[index] = f32(FREQ_SCALE * f32(Math.exp(exponent)));
  }
  return freqs;
};

/**
 * 1 step ぶんの `t_embed` `[2·freqs.length]`。**前半 cos・後半 sin**（上流の `torch.cat`）。
 *
 * MUST: 前後半を入れ替えない。値域が同じ `[-1,1]` なので、取り違えても shape も統計も
 * 変わらず、出力だけが別物になる。
 */
export const timestepEmbedding = (
  t: number,
  freqs: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const half = freqs.length;
  const out = new Float32Array(half * 2);
  for (let index = 0; index < half; index += 1) {
    const angle = f32(t * freqs[index]);
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};
