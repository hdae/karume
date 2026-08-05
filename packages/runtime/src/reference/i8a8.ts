/**
 * w8a8 linear（活性 per-token i8 × 重み per-channel i8）の **CPU 参照**（ADR 0005 の段 2）。
 *
 * 本経路の数値契約は **GPU と atol=0 で一致する**ことで、これは他のカーネル（f32 の縮約順序が
 * 実装で変わる）と決定的に違う性質になる。成立の根拠は 2 つ:
 *
 * 1. 内積が `i32` の**厳密加算**なので、タイル分割・端数 0 埋め・加算順序のどれも値を変えない。
 * 2. 浮動小数の演算は出力 1 要素あたり 3 つだけ — `f32(acc)` / `xs·wscale` / `fma`。どれも
 *    IEEE の round-to-nearest-even で決定的。
 *
 * MUST: GPU 側のタイル構造を写さない（src/kernels/linear-i8a8.ts の 64×64 タイル・共有
 * メモリ・K 端数 0 埋めは**ここには無い**）。素の 3 重ループで書くことが、タイル境界と端数の
 * 扱いを検証対象として残す唯一の方法（referenceLinear と同じ規律）。
 *
 * ## 丸めの同点規則（実測で確定・2026-08-03）
 *
 * WGSL の `round` は**偶数丸め**（実測: `round(0.5)=0` / `round(1.5)=2` / `round(-0.5)=-0` /
 * `round(2.5)=2`）で、`torch.round` と一致する。JS の `Math.round` は half-up（+∞ 方向）で
 * **一致しない** — 正本は {@link roundTiesToEven} 1 本に閉じる。
 *
 * ## `fma` の鏡像（実測で確定・2026-08-03）
 *
 * GPU 側の最終式は `fma(f32(acc), xs·ws, bias)`（積を丸めずに 1 度だけ丸める）。JS には fma が
 * 無いが、`f32(acc)` と `xs·ws` はどちらも f32 なので**倍精度での積は厳密**（24+24 = 48 ≤ 53
 * ビット）で、`Math.fround(a * s + b)` が単一丸めの結果と一致する。本機で 100,000 サンプルの
 * 全数一致を実測済み（素の `a*b+c` / 明示 `fma` / `bitcast` 固定の 3 形とも同値だった）。
 * NOTE: 厳密和が 54 ビットを超える組では倍精度の中間丸めが理論上効きうる（`|f32(acc)·s|` と
 * `|bias|` の指数差が大きい場合）。実測では 1 件も出ていないが、原理的な保証ではない。
 *
 * ## 除算だけは atol=0 の外（実測で判明・2026-08-03）
 *
 * WGSL の f32 除算は仕様上 2.5 ULP まで許され、本機の実測でも `a / b` が IEEE の正しい丸めと
 * 200,000 サンプル中 55,605 件で 1 ULP 割れた（乗算・加算・`fma` は 0 件）。したがって
 * `q = round(x / s)` は **`x/s` が半整数の近傍にある要素でだけ ±1 段揺れうる**。
 * scale の側は `amax * (1/127)` の**乗算**で作るので厳密に一致する（src/kernels/
 * quantize-rows.ts の MUST）。atol=0 の突合は「丸め境界から十分離れたデータ」でのみ成立する
 * 契約で、余裕はテスト側が実測して門にする（tests/gpu_i8a8_test.ts）。
 */

/**
 * 量子化の格子の端と scale の床。
 *
 * MUST: src/kernels/quantize-rows.ts の定数を**輸入しない**。共有すると「格子を ±128 に
 * する」「床を落とす」といった退行が参照にも同じだけ乗り、突合が恒真化する（故障注入②で
 * 実証済み）。参照は仕様（設計 doc §4.2）から独立に書き下す。
 */
const ABS_MAX = 127;
/** f32 の最小 normal（`torch.finfo(float32).tiny`）。 */
const F32_TINY = 1.1754943508222875e-38;
/** MUST: scale は 127 での除算ではなく 1/127 との**乗算**（除算は正しく丸められない）。 */
const INV_ABS_MAX = Math.fround(1 / ABS_MAX);

/**
 * 偶数丸め（round-half-to-even）。WGSL の `round` / `torch.round` と同じ規則。
 *
 * MUST: `Math.round` を使わない（half-up なので `-0.5 → 0` は合うが `0.5 → 1` /
 * `2.5 → 3` で割れる）。量子化の格子の境界にちょうど乗る入力は実データでも普通に出る。
 */
export const roundTiesToEven = (value: number): number => {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5) return floor;
  // 同点は偶数側へ（floor が偶数なら floor、奇数なら floor + 1）
  return floor % 2 === 0 ? floor : floor + 1;
};

export type QuantizedRows = {
  /**
   * 量子化した活性（`[rows, dim]` の平坦 i8 — GPU の `xq` を 4 詰めから展開した形）。
   *
   * NOTE: 行の scale が非有限（NaN / Inf）のとき、この値は**契約の外**（GPU 側は
   * `vec4<i32>(NaN)` が不定値になる）。突合の対象は `scale` と最終出力だけにする。
   */
  readonly q: Int8Array<ArrayBuffer>;
  /** 行ごとの scale（`[rows]`）。 */
  readonly scale: Float32Array<ArrayBuffer>;
};

/**
 * 行（= token）ごとの symmetric i8 量子化（src/kernels/quantize-rows.ts の鏡像）。
 *
 * MUST: NaN は行の scale へ伝播させる（`amax` が NaN なら `s` も NaN）。GPU 側はビット列
 * 判定でこれを保証しており、参照が素の `Math.max` で NaN を飲むと突合が恒真化する。
 */
export const quantizeRowsReference = (
  x: ArrayLike<number>,
  rows: number,
  dim: number,
): QuantizedRows => {
  const q = new Int8Array(rows * dim);
  const scale = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = 0;
    let sawNan = false;
    for (let i = 0; i < dim; i += 1) {
      const value = Math.abs(x[base + i]);
      if (Number.isNaN(value)) sawNan = true;
      else if (value > amax) amax = value;
    }
    const s = sawNan ? Number.NaN : Math.max(Math.fround(amax * INV_ABS_MAX), F32_TINY);
    scale[row] = s;
    for (let i = 0; i < dim; i += 1) {
      const rounded = roundTiesToEven(Math.fround(x[base + i] / s));
      q[base + i] = Math.max(-ABS_MAX, Math.min(ABS_MAX, rounded));
    }
  }
  return { q, scale };
};

/**
 * `x/s` が丸め境界（半整数）からどれだけ離れているかの最小値。
 *
 * GPU の除算は 2.5 ULP まで許されるので、**この余裕が大きいデータでだけ** GPU と参照の
 * 量子化値が完全に一致する（上の docstring）。テストは atol=0 を主張する前にこの値を
 * 実測して門にする — 余裕が縮んだのに気づかずに atol=0 を掲げると、実質は「たまたま通って
 * いるだけ」になる。|x/s| ≤ 127 での 2.5 ULP は約 1.9e-5 なので、1e-3 も取れば桁で安全。
 */
export const quantizeRowsTieMargin = (
  x: ArrayLike<number>,
  rows: number,
  dim: number,
): number => {
  const { scale } = quantizeRowsReference(x, rows, dim);
  let margin = Number.POSITIVE_INFINITY;
  for (let row = 0; row < rows; row += 1) {
    for (let i = 0; i < dim; i += 1) {
      const ratio = Math.fround(x[row * dim + i] / scale[row]);
      if (!Number.isFinite(ratio)) continue;
      const fraction = ratio - Math.floor(ratio);
      margin = Math.min(margin, Math.abs(fraction - 0.5));
    }
  }
  return margin;
};

export type LinearI8a8Input = {
  /** 活性 `[m, k]`（f32・量子化前）。 */
  readonly x: ArrayLike<number>;
  /** 重み `[n, k]` の**量子化済み整数値**（±127）。 */
  readonly weight: ArrayLike<number>;
  /** 重みの per-channel scale `[n]`。 */
  readonly weightScale: ArrayLike<number>;
  /** bias `[n]`（常に f32 — ADR 0006）。 */
  readonly bias: ArrayLike<number>;
  readonly m: number;
  readonly n: number;
  readonly k: number;
};

/**
 * w8a8 linear の参照 — `out = f32(Σ xq·wq) · (xs[row] · wscale[col]) + bias[col]`。
 *
 * MUST: 乗算の順序は GPU と同じ（`xs · wscale` を先に 1 つの f32 へ畳む）。逐次形
 * （`f32(acc) · xs · wscale`）にすると丸めの位置が動いて atol=0 が崩れる。
 */
export const referenceLinearI8a8 = (input: LinearI8a8Input): Float32Array<ArrayBuffer> => {
  const { x, weight, weightScale, bias, m, n, k } = input;
  const { q, scale } = quantizeRowsReference(x, m, k);
  const out = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    for (let col = 0; col < n; col += 1) {
      // 整数の厳密和（|acc| ≤ k·127² は 2^53 に遠く及ばないので JS の number で厳密）
      let acc = 0;
      for (let i = 0; i < k; i += 1) acc += q[row * k + i] * weight[col * k + i];
      const combined = Math.fround(scale[row] * weightScale[col]);
      out[row * n + col] = Math.fround(Math.fround(acc) * combined + bias[col]);
    }
  }
  return out;
};

/**
 * P̃ の量子化格子の端と `1/127`。
 *
 * MUST: 上の {@link ABS_MAX} と**束ねない**。あちらは「行 amax から作る適応 scale」の格子で、
 * こちらは「`P̃ = exp(S−m)` の行内 max が構造的に 1.0 だから scale が 1/127 に縮退する」という
 * 別の事実（設計 §2.2）。値が一致するのは偶然に近く、束ねると片方の変更が黙って伝播する。
 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

/**
 * ③PV が A タイル充填で作る量子化列 `qP = round(127·exp(S−m))` の参照（`[batch·m, n]` 平坦）。
 *
 * **GPU とビット一致はしない**（`exp` の実装差 — 設計 §4.2）。この参照の役目は
 * 「不一致は必ず ±1 段」という門と不一致率の記録で、テスト側がそれを assert する。
 * MUST: 丸めは偶数丸め（{@link roundTiesToEven}）。`Math.round` は half-up で割れる。
 * NOTE: clamp を置かないのは GPU 側と同じ理由（`m` が行の厳密な最大なら `exp(S−m) ≤ 1`）。
 * `rowMax` に行の最大でない値を渡すと 127 を超えうるが、それは呼び出し側の契約違反。
 */
export const referenceAttentionPvQuant = (
  s: ArrayLike<number>,
  rowMax: ArrayLike<number>,
  rows: number,
  n: number,
): Int8Array<ArrayBuffer> => {
  const qp = new Int8Array(rows * n);
  for (let row = 0; row < rows; row += 1) {
    const max = rowMax[row];
    for (let i = 0; i < n; i += 1) {
      const shifted = Math.fround(s[row * n + i] - max);
      const p = Math.fround(Math.fround(Math.exp(shifted)) * P_ABS_MAX);
      qp[row * n + i] = roundTiesToEven(p);
    }
  }
  return qp;
};

export type AttentionPvI8a8CoreInput = {
  /** `qP`（`[batch, m, n]` の平坦・0..127 の整数）。GPU が A タイルで作った列そのもの。 */
  readonly qp: ArrayLike<number>;
  /** Vᵀ の量子化値（`[batch, d, n]` の平坦・±127 の整数 — **N 連続**）。 */
  readonly vq: ArrayLike<number>;
  /** 行統計の `1/Σexp(S−m)`（`[batch·m]`）。 */
  readonly rowInv: ArrayLike<number>;
  /** Vᵀ の行 scale（`[batch·d]`）= V の**列** scale。 */
  readonly vs: ArrayLike<number>;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * 融合 attention ③PV の i8a8 変種の参照 — **整数を受け取ってからの純関数**。
 *
 * `O[b,m,d] = f32(Σ_n qP·vq) · ((inv·(1/127)) · vs[b,d])` で、浮動小数の演算は
 * `f32(acc)` / `inv·(1/127)` / `prow·vs` / 積 の 4 つだけ。したがって **GPU と atol=0**
 * （`qP` の生成だけが `exp` の実装差を持ち、それは {@link referenceAttentionPvQuant} 側の
 * 別の門に切り出してある — 設計 §4.2 の 2 段分割）。
 *
 * MUST: 乗算の順序は GPU と同じ（`prow · vs` を先に 1 つの f32 へ畳む）。逐次形
 * （`f32(acc)·prow·vs`）にすると丸めの位置が動いて atol=0 が崩れる。
 * MUST: `1/127` は乗算で作る（GPU 側と同じ — 除算は正しく丸められない）。
 * MUST: GPU 側のタイル構造を写さない（素の 4 重ループ — 他の参照と同じ規律）。
 */
export const referenceAttentionPvI8a8Core = (
  input: AttentionPvI8a8CoreInput,
): Float32Array<ArrayBuffer> => {
  const { qp, vq, rowInv, vs, batch, m, n, d } = input;
  const out = new Float32Array(batch * m * d);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const prow = Math.fround(rowInv[b * m + row] * INV_P_ABS_MAX);
      for (let col = 0; col < d; col += 1) {
        // 整数の厳密和（|acc| ≤ n·127² は 2^53 に遠く及ばないので JS の number で厳密）
        let acc = 0;
        for (let i = 0; i < n; i += 1) {
          acc += qp[(b * m + row) * n + i] * vq[(b * d + col) * n + i];
        }
        const combined = Math.fround(prow * vs[b * d + col]);
        out[(b * m + row) * d + col] = Math.fround(Math.fround(acc) * combined);
      }
    }
  }
  return out;
};

export type AttentionQkI8a8Input = {
  /** q `[batch, m, d]`（f32・量子化前。batch は B·H を畳んだ 1 軸）。 */
  readonly q: ArrayLike<number>;
  /** k `[batch, n, d]`（f32・量子化前）。 */
  readonly k: ArrayLike<number>;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
  /** **半スケール**（ADR 0023 の `√scale_factor`）。q 側と k 側の両方へ掛かる。 */
  readonly scale: number;
};

/**
 * 融合 attention ①QK の i8a8 変種の参照 —
 * `S[b,m,n] = f32(Σ_d qq·kq) · ((qs[b,m]·scale) · (ks[b,n]·scale))`。
 *
 * MUST: 半スケールは**量子化の後**（dequant 側）で掛ける。量子化の前に掛ける形にすると
 * 丸めが 1 段増え、GPU（src/kernels/attention-i8a8.ts）との atol=0 が崩れる。
 * MUST: 乗算の順序も GPU と同じ（`qs'·ks'` を先に 1 つの f32 へ畳む）。逐次形
 * （`f32(acc)·qs'·ks'`）にすると丸めの位置が動く。
 * MUST: GPU 側のタイル構造を写さない（素の 4 重ループで書くことが、タイル境界と K 端数の
 * 扱いを検証対象として残す唯一の方法 — {@link referenceLinearI8a8} と同じ規律）。
 */
export const referenceAttentionQkI8a8 = (
  input: AttentionQkI8a8Input,
): Float32Array<ArrayBuffer> => {
  const { batch, m, n, d } = input;
  const scale = Math.fround(input.scale);
  // 縮約軸 D が q / k とも最内連続なので、量子化は行 = (b, m) / (b, n) の per-token そのもの
  const query = quantizeRowsReference(input.q, batch * m, d);
  const key = quantizeRowsReference(input.k, batch * n, d);
  const out = new Float32Array(batch * m * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const qs = Math.fround(query.scale[b * m + row] * scale);
      for (let col = 0; col < n; col += 1) {
        // 整数の厳密和（|acc| ≤ d·127² は 2^53 に遠く及ばないので JS の number で厳密）
        let acc = 0;
        for (let i = 0; i < d; i += 1) {
          acc += query.q[(b * m + row) * d + i] * key.q[(b * n + col) * d + i];
        }
        const ks = Math.fround(key.scale[b * n + col] * scale);
        out[(b * m + row) * n + col] = Math.fround(Math.fround(acc) * Math.fround(qs * ks));
      }
    }
  }
  return out;
};
