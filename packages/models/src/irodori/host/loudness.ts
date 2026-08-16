/**
 * ITU-R BS.1770-4 の integrated loudness（LUFS）。参照音声の LUFS 正規化にだけ使う。
 *
 * 上流は `descript-audiotools` の `AudioSignal.loudness()`（= `pyloudnorm` の Meter を torch で
 * 書き直したもの）を通す。ここはその**実装仕様**の移植で、規格の一般形ではない — 規格が
 * 選択の余地を残している箇所（ゲーティングで残らなかったときの値・0.5 秒未満の扱い・ブロックの
 * 数え方）は上流の綴りに合わせてある。
 *
 * ## 段
 *
 * 1. 0.5 秒未満はゼロ pad してから測る（`AudioSignal.loudness` の `signal_duration < 0.5`）。
 *    pad は測定のためだけで、返るのは元の波形（呼び出し側は何も切らなくてよい）
 * 2. K-weighting = high_shelf → high_pass の 2 段 biquad（{@link kWeightingFilters}）
 * 3. 0.4 秒のブロックを 75% 重ねて（stride 0.1 秒）並べ、ブロックごとの平均二乗 z を採る。
 *    末尾の欠けたブロックは捨てずゼロ pad する（`F = 1 + ceil((T − 19200)/4800)`）
 * 4. l = −0.691 + 10·log10(z) を絶対閾 −70 LKFS で選び、残った z の平均から相対閾（−10 LU）を
 *    出し、**両方を通った**ブロックの z 平均で integrated loudness を出す
 * 5. −70 LKFS で下から clamp する
 *
 * ## MUST: f64 で回す（上流は f32 の `lfilter`）
 *
 * 上流は係数もデータも f32 のまま `torchaudio.functional.lfilter` を通す。ここを
 * `Math.fround` の逐次で写す選択もあるが、**採らない** — IIR の逐次差分方程式は丸めの入り方が
 * ライブラリの内部展開順（torchaudio は係数を正規化してから差分式を回す）に依存し、写しても
 * ビット一致は得られない。結果として残る差は refDb の 1e-5 LU 級（`irodori_reference_test.ts`
 * の実測表）で、正規化利得に直すと相対 1e-6 — 波形の tolerance に埋もれる。写経で偽の
 * 一致を演出するより、差を実測して門の閾値に載せるほうが素性が良い。
 */

/** biquad 1 段（分母は `a[0] = 1` に正規化済み）。 */
type Biquad = {
  readonly b: readonly [number, number, number];
  readonly a: readonly [number, number, number];
};

/** 測定ブロック長（秒）。 */
const BLOCK_SECONDS = 0.4;
/** ブロックの重なり（75% → stride は 0.1 秒）。 */
const BLOCK_OVERLAP = 0.75;
/** 測定の最小長（秒）。これ未満はゼロ pad してから測る。 */
const MIN_MEASURE_SECONDS = 0.5;
/** BS.1770 のオフセット項（`l = OFFSET + 10·log10(z)`）。 */
const LOUDNESS_OFFSET = -0.691;
/** 絶対ゲート（LKFS）。返り値の下限でもある。 */
const ABSOLUTE_GATE_DB = -70;
/** 相対ゲート（LU — 絶対ゲートを通ったブロックの平均から下げる幅）。 */
const RELATIVE_GATE_LU = -10;

/** K-weighting 第 1 段（high shelf）の設計値（RBJ cookbook のパラメタ）。 */
const SHELF_GAIN_DB = 4.0;
const SHELF_Q = Math.SQRT1_2;
const SHELF_FC_HZ = 1500;
/** K-weighting 第 2 段（high pass）の設計値。ゲインは持たない（G = 0dB）。 */
const HIGHPASS_Q = 0.5;
const HIGHPASS_FC_HZ = 38;

/**
 * RBJ cookbook の high shelf。
 *
 * MUST: 係数を数値で焼かない（式から作る）。48kHz 以外の参照音声はパイプラインが受け付け
 * ないが、係数を直書きすると「48kHz 用の係数で別の周波数を測る」誤りが**沈黙で**通る。
 * 48kHz での値そのものは `irodori_reference_test.ts` が固定する（式の写し間違いの検出）。
 */
const highShelf = (sampleRate: number, gainDb: number, q: number, fc: number): Biquad => {
  const amplitude = Math.pow(10, gainDb / 40);
  const w0 = 2 * Math.PI * (fc / sampleRate);
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const root = Math.sqrt(amplitude);
  const b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cos + 2 * root * alpha);
  const b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos);
  const b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cos - 2 * root * alpha);
  const a0 = (amplitude + 1) - (amplitude - 1) * cos + 2 * root * alpha;
  const a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cos);
  const a2 = (amplitude + 1) - (amplitude - 1) * cos - 2 * root * alpha;
  return { b: [b0 / a0, b1 / a0, b2 / a0], a: [1, a1 / a0, a2 / a0] };
};

/** RBJ cookbook の high pass（2 次）。 */
const highPass = (sampleRate: number, q: number, fc: number): Biquad => {
  const w0 = 2 * Math.PI * (fc / sampleRate);
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const b0 = (1 + cos) / 2;
  const a0 = 1 + alpha;
  return {
    b: [b0 / a0, -(1 + cos) / a0, b0 / a0],
    a: [1, (-2 * cos) / a0, (1 - alpha) / a0],
  };
};

/** K-weighting の 2 段（high shelf → high pass の順で当てる）。 */
export const kWeightingFilters = (sampleRate: number): readonly [Biquad, Biquad] => [
  highShelf(sampleRate, SHELF_GAIN_DB, SHELF_Q, SHELF_FC_HZ),
  highPass(sampleRate, HIGHPASS_Q, HIGHPASS_FC_HZ),
];

/**
 * biquad を逐次差分方程式で当てる（`lfilter` 相当 — 初期状態はゼロ）。
 *
 * MUST: 周波数領域や FIR 近似で代用しない。上流は厳密な IIR で、近似すると LUFS が 0.1 LU
 * 級でずれる（正規化利得が 1% 動く = 波形が全サンプルで食い違う）。
 */
const applyBiquad = (
  samples: Float64Array<ArrayBuffer>,
  filter: Biquad,
): Float64Array<ArrayBuffer> => {
  const [b0, b1, b2] = filter.b;
  const [, a1, a2] = filter.a;
  const out = new Float64Array(samples.length) as Float64Array<ArrayBuffer>;
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const x0 = samples[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
};

/**
 * integrated loudness（LUFS / LKFS）を返す。mono 専用（チャネル重み G = 1）。
 *
 * ゲートを 1 ブロックも通らなかったときは −70 を返す（上流は `nan_to_num` → `log10(0)` →
 * `-inf` → `maximum(-70)` の順で同じ値に落ちる）。
 */
export const integratedLoudness = (samples: Float32Array, sampleRate: number): number => {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`integratedLoudness: サンプリング周波数 ${sampleRate} が正の整数でない`);
  }
  if (samples.length === 0) throw new RangeError("integratedLoudness: 波形が空");
  // 0.5 秒未満は**測定のためだけに**末尾へゼロを足す（上流 `AudioSignal.loudness`）。
  const minimum = Math.round(MIN_MEASURE_SECONDS * sampleRate);
  const measured = new Float64Array(Math.max(samples.length, minimum)) as Float64Array<ArrayBuffer>;
  measured.set(samples);

  let filtered = measured;
  for (const filter of kWeightingFilters(sampleRate)) filtered = applyBiquad(filtered, filter);

  // MUST: ブロックの数え方は `julius.core.unfold`（audiotools が `torch.nn.Unfold` の代わりに
  // 使う）— **末尾の欠けたブロックを捨てず、ゼロ pad して数に入れる**。torch の Unfold と
  // 読み違えると、長さが stride の倍数でない参照音声だけ LUFS が 0.07 LU ずれる（実測 —
  // golden の ref-odd が 27 ブロックと 28 ブロックを判別する唯一のケース）。
  // pad は**フィルタを当てた後**に足す（`_unfold` の中なので、IIR の尾はここへ伸びない）。
  const kernel = Math.trunc(BLOCK_SECONDS * sampleRate);
  const stride = Math.trunc(BLOCK_SECONDS * sampleRate * (1 - BLOCK_OVERLAP));
  // MUST: floor 導出なので、極小の周波数では窓そのものが消える — stride 0 はブロック数が
  // Infinity（確保の不透明なエラー）、kernel 0 は `squared / kernel` が NaN。どちらも
  // 「その周波数では測れない」ことなので、導出値の下限で明示的に落とす。
  if (kernel < 1 || stride < 1) {
    throw new RangeError(
      `integratedLoudness: サンプリング周波数 ${sampleRate} では loudness 窓が導出できない` +
        `（${BLOCK_SECONDS} 秒 = ${kernel} サンプル / stride ${stride} サンプル — 各 1 以上が要る）`,
    );
  }
  const blocks = Math.ceil((Math.max(filtered.length, kernel) - kernel) / stride) + 1;
  const power = new Float64Array(blocks);
  for (let block = 0; block < blocks; block += 1) {
    const from = block * stride;
    const end = Math.min(from + kernel, filtered.length);
    let squared = 0;
    for (let i = from; i < end; i += 1) squared += filtered[i] * filtered[i];
    power[block] = squared / kernel;
  }

  // 絶対ゲート → その平均から相対ゲート → 両方を通ったブロックの平均。
  const level = (mean: number): number => LOUDNESS_OFFSET + 10 * Math.log10(mean);
  const gatedMean = (threshold: number, relative: number): number => {
    let sum = 0;
    let count = 0;
    for (const value of power) {
      const decibel = level(value);
      if (decibel > threshold && decibel > relative) {
        sum += value;
        count += 1;
      }
    }
    return count === 0 ? 0 : sum / count;
  };
  const absoluteMean = gatedMean(ABSOLUTE_GATE_DB, Number.NEGATIVE_INFINITY);
  const relativeGate = level(absoluteMean) + RELATIVE_GATE_LU;
  const loudness = level(gatedMean(ABSOLUTE_GATE_DB, relativeGate));
  return Math.max(loudness, ABSOLUTE_GATE_DB);
};
