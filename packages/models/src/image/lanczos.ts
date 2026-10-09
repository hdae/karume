/**
 * RGB8 の LANCZOS リサンプル — Pillow の `Image.resize(size, Image.LANCZOS)` の 8bit 経路
 * （`libImaging/Resample.c` の `ImagingResampleInner` と `*_8bpc`）の逐語の移植。
 *
 * Wan2.2 の I2V の条件画像は、公式実装（`textimage2video.py`）も diffusers（`VaeImageProcessor` の
 * `resample="lanczos"`）も Pillow の LANCZOS で作る。その uint8 をビット単位で再現するために置く
 * （ADR 0121 決定 11）。
 *
 * ## `preprocess.ts` の `resizeRgb8` と経路を分けている理由
 *
 * MUST: `resizeRgb8` の経路（f64 の重み・`/ filterScale`・`floor(v + 0.5)`）に LANCZOS のカーネルを足すだけでは
 * Pillow と一致しない。Pillow の 8bit 経路は係数を 22 bit の固定小数点の整数へ丸め、積和を整数で取る。f64 のままだと
 * 丸め境界に載った標本が 1〜2 LSB ずれる（段 9 の調査の Python の写しで 0.002〜0.275%）。一方 `resizeRgb8` は
 * torchvision の antialias 経路に合わせた SigLIP2 / Depth Anything / BiRefNet の前処理で、こちらに寄せると彼らの
 * 出力が動く。よって経路を分け、公開の `resizeRgb8` / `ResampleFilter` は広げない（ADR 0008）。
 *
 * ## 逐語に写している点（Pillow 12.3.0 — fixture `tests/fixtures/wan-i2v/lanczos.json`〈画素は
 * `lanczos.safetensors`〉が版を記録する）
 *
 * - カーネル: support 3・`sinc(x)·sinc(x / 3)`（`-3 ≤ x < 3` の外は 0）
 * - 係数の引数は `(x + xmin − center + 0.5) · (1 / filterscale)`（逆数を掛ける — Pillow の形のまま。割る形との
 *   差は推測で、2026-10-09 のレビューが 1,600 幾何〈入力 ≤ 4000・出力 ≤ 1400〉の係数表を比べた範囲では差が
 *   出ていない）
 * - `xmin` / `xmax` は `(int)` の切り捨ての後に `[0, inSize]` へ収める
 * - 総和 `ww` が 0 でないときだけ正規化する
 * - 係数は `(int)(±0.5 + w·2^22)`（符号で丸めの向きを変える）
 * - 積和は整数で、初期値 2^21 から `>> 22` して 0〜255 に飽和する
 * - 横 → 縦の順・中間は uint8・寸法が変わらない軸はパスを飛ばす（`need_horizontal` / `need_vertical`）
 * - 横パスは縦パスが読む入力の行（`ybox_first`〜`ybox_last`）だけを回す
 *
 * ## 窓だけを計算する（{@link resizeRgb8LanczosWindow}）
 *
 * 係数は出力の添字ごとに独立し（`center = (i + 0.5)·scale`）、中間の uint8 も出力の列ごとに独立している。よって
 * 伸縮後の画像の一部の矩形は、その矩形の出力の添字だけを計算すれば、全体を伸縮してから切り出した値と uint8 で
 * 同じになる。I2V の crop は覆う寸法へ伸縮してから中央を切り出すので、縦横比の極端な入力（例 4096×4 → 覆う寸法
 * 720,896×704）でも、手間と中間の大きさを出力の窓の大きさに抑えられる。
 *
 * NOTE: 係数の `sin` は V8 の `Math.sin`、Pillow は C の libm の `sin`。最終 ulp が割れると整数係数が 1 変わりうる
 * （推測）。fixture は出力の uint8 だけを見るので、係数の ±1 はまず検出できない（ずれた係数が出力を変えるのは積和が
 * 2^22 の境界をまたぐときだけ — 1 標本あたりおよそ 画素値 / 2^22）。2026-10-09 のレビューが係数表そのもの（`xmin` と
 * 整数係数の全列）を Resample.c の逐語の Python の写し（glibc の `sin`）と 1,600 幾何で比べ、不一致は 0 だった（その
 * 場の確かめで、常設の門ではない）。
 *
 * NOTE: `export` は Wan の I2V の前処理とテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */

import { assertRgb8, type Rgb8Image } from "./preprocess.ts";

/** チャネル数（RGB）。 */
const CHANNELS = 3;

/** 係数の固定小数点の小数部のビット数（Pillow の `PRECISION_BITS` = 32 − 8 − 2）。 */
const PRECISION_BITS = 32 - 8 - 2;

/** 係数の 1.0（`1 << PRECISION_BITS`）。 */
const FIXED_ONE = 1 << PRECISION_BITS;

/** 積和の初期値（`1 << (PRECISION_BITS - 1)` — `>> PRECISION_BITS` を四捨五入にする半分）。 */
const FIXED_HALF = 1 << (PRECISION_BITS - 1);

/** LANCZOS の台の半径（Pillow の `LANCZOS.support`）。 */
const SUPPORT = 3;

const sinc = (x: number): number => {
  if (x === 0) return 1;
  const radians = x * Math.PI;
  return Math.sin(radians) / radians;
};

/** Pillow の `lanczos_filter`（台の外は 0 — 右端 `x = 3` は台の外）。 */
const lanczos = (x: number): number => (-SUPPORT <= x && x < SUPPORT ? sinc(x) * sinc(x / 3) : 0);

/** 出力 1 点ぶんの入力範囲と固定小数点の係数。 */
type FixedTap = {
  readonly start: number;
  readonly coefficients: Int32Array;
};

/** 1 軸の出力の窓（伸縮後の添字 `offset` から `length` 個）。 */
type AxisWindow = {
  readonly offset: number;
  readonly length: number;
};

/**
 * 1 軸ぶんの係数（Pillow の `precompute_coeffs` → `normalize_coeffs_8bpc`）のうち、出力の窓 `window` の添字の分
 * （添字ごとに独立 — モジュール doc「窓だけを計算する」）。
 *
 * NOTE: Pillow の `scale` は `(double)(in1 − in0) / outSize`（`in0` / `in1` は float の箱）で、箱を取らない resize では
 * `inSize / outSize` と同じ値になる（2^24 未満の整数は float で厳密）。
 */
const fixedTaps = (inSize: number, outSize: number, window: AxisWindow): readonly FixedTap[] => {
  const scale = inSize / outSize;
  const filterScale = scale < 1 ? 1 : scale;
  const support = SUPPORT * filterScale;
  const inverseFilterScale = 1 / filterScale;
  const taps: FixedTap[] = [];
  for (let index = window.offset; index < window.offset + window.length; index += 1) {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const count = Math.min(inSize, Math.trunc(center + support + 0.5)) - start;
    const weights = new Float64Array(count);
    let total = 0;
    for (let x = 0; x < count; x += 1) {
      const weight = lanczos((x + start - center + 0.5) * inverseFilterScale);
      weights[x] = weight;
      total += weight;
    }
    const coefficients = new Int32Array(count);
    for (let x = 0; x < count; x += 1) {
      const weight = total !== 0 ? weights[x] / total : weights[x];
      coefficients[x] = weight < 0
        ? Math.trunc(-0.5 + weight * FIXED_ONE)
        : Math.trunc(0.5 + weight * FIXED_ONE);
    }
    taps.push({ start, coefficients });
  }
  return taps;
};

/**
 * 固定小数点の積和を 8bit へ戻す（Pillow の `clip8` — `>> PRECISION_BITS` の後に 0〜255 へ飽和）。
 *
 * 積和は Pillow と同じく int32 に収まる（|係数の和| ≲ 1.3·2^22 × 255 + 2^21 < 2^31）ので、`>>` の ToInt32 は値を
 * 変えない。
 */
const clip8 = (sum: number): number => {
  const value = sum >> PRECISION_BITS;
  return value < 0 ? 0 : value > 255 ? 255 : value;
};

/**
 * 横パス（Pillow の `ImagingResampleHorizontal_8bpc` の 3 バンドの枝）。入力の行 `rows` だけを回す（Pillow の
 * `ybox_first`〜`ybox_last`）。出力の列は `taps` の本数（窓の列）。
 */
const resampleHorizontal = (
  image: Rgb8Image,
  taps: readonly FixedTap[],
  rows: AxisWindow,
): Rgb8Image => {
  const width = taps.length;
  const sourceStride = image.width * CHANNELS;
  const targetStride = width * CHANNELS;
  const data = new Uint8Array(rows.length * targetStride);
  for (let y = 0; y < rows.length; y += 1) {
    const sourceRow = (rows.offset + y) * sourceStride;
    const targetRow = y * targetStride;
    for (let x = 0; x < width; x += 1) {
      const { start, coefficients } = taps[x];
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        let sum = FIXED_HALF;
        for (let j = 0; j < coefficients.length; j += 1) {
          sum += image.data[sourceRow + (start + j) * CHANNELS + channel] * coefficients[j];
        }
        data[targetRow + x * CHANNELS + channel] = clip8(sum);
      }
    }
  }
  return { data, width, height: rows.length };
};

/**
 * 縦パス（Pillow の `ImagingResampleVertical_8bpc` の 3 バンドの枝）。`image` の行 0 は入力の行 `rowOffset`（Pillow
 * が横パスの後に `bounds_vert` から `ybox_first` を引くのと同じ読み替え）。出力の行は `taps` の本数（窓の行）。
 */
const resampleVertical = (
  image: Rgb8Image,
  taps: readonly FixedTap[],
  rowOffset: number,
): Rgb8Image => {
  const stride = image.width * CHANNELS;
  const data = new Uint8Array(taps.length * stride);
  for (let y = 0; y < taps.length; y += 1) {
    const { start, coefficients } = taps[y];
    const sourceRow = start - rowOffset;
    const targetRow = y * stride;
    for (let x = 0; x < stride; x += 1) {
      let sum = FIXED_HALF;
      for (let j = 0; j < coefficients.length; j += 1) {
        sum += image.data[(sourceRow + j) * stride + x] * coefficients[j];
      }
      data[targetRow + x] = clip8(sum);
    }
  }
  return { data, width: image.width, height: taps.length };
};

/** 列 `columns`・行 `rows` の矩形の写し（パスを飛ばした軸の切り出し — 範囲の内側だけを呼ぶ）。 */
const cropRgb8 = (image: Rgb8Image, columns: AxisWindow, rows: AxisWindow): Rgb8Image => {
  const rowBytes = columns.length * CHANNELS;
  const data = new Uint8Array(rows.length * rowBytes);
  for (let y = 0; y < rows.length; y += 1) {
    const from = ((rows.offset + y) * image.width + columns.offset) * CHANNELS;
    data.set(image.data.subarray(from, from + rowBytes), y * rowBytes);
  }
  return { data, width: columns.length, height: rows.length };
};

/** 伸縮後の画像から切り出す矩形（左上 `left`, `top` から `width × height`）。 */
export type Rgb8Window = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/**
 * 縦パスが読む入力の行（Pillow の `ybox_first`〜`ybox_last`）。tap の起点も終点も出力の添字について単調なので、
 * 両端の tap で決まる。
 */
const rowsReadBy = (taps: readonly FixedTap[]): AxisWindow => {
  const first = taps[0];
  const last = taps[taps.length - 1];
  return { offset: first.start, length: last.start + last.coefficients.length - first.start };
};

/**
 * RGB8 を `width × height` へ LANCZOS でリサンプルし（Pillow の `Image.resize(..., Image.LANCZOS)` と uint8 で一致 —
 * モジュール doc）、その矩形 `window` を切り出したものを返す。計算するのは窓の出力だけ（モジュール doc「窓だけを
 * 計算する」— 全体を伸縮してから切り出した値と同じ）。縦横比は保たない。寸法が変わらない軸はパスを飛ばし（その軸は
 * 入力の写しを切り出す）、両軸とも変わらなければ写しを返す（Pillow の `ImagingCopy`）。
 *
 * 画像の検査は `assertRgb8`（呼び手が渡した画像 — `ModelInputError`）。出力寸法と窓は内部の事前条件（呼び手は受理
 * 集合とそこから導いた寸法しか渡さない）なので `RangeError`。
 */
export const resizeRgb8LanczosWindow = (
  image: Rgb8Image,
  width: number,
  height: number,
  window: Rgb8Window,
): Rgb8Image => {
  assertRgb8(image);
  if (!isPositiveInteger(width) || !isPositiveInteger(height)) {
    throw new RangeError(`LANCZOS の出力寸法 ${width}×${height} が正の整数でない`);
  }
  const { left, top } = window;
  if (
    !Number.isInteger(left) || !Number.isInteger(top) || left < 0 || top < 0 ||
    !isPositiveInteger(window.width) || !isPositiveInteger(window.height) ||
    left + window.width > width || top + window.height > height
  ) {
    throw new RangeError(
      `LANCZOS の窓 (${left}, ${top}) ${window.width}×${window.height} が出力 ${width}×${height} の内側でない`,
    );
  }
  const columns = { offset: left, length: window.width };
  const verticalTaps = height === image.height
    ? undefined
    : fixedTaps(image.height, height, { offset: top, length: window.height });
  // 横パスが回す入力の行（縦を飛ばすなら窓の行そのもの）。
  const rows = verticalTaps === undefined
    ? { offset: top, length: window.height }
    : rowsReadBy(verticalTaps);
  const band = width === image.width
    ? cropRgb8(image, columns, rows)
    : resampleHorizontal(image, fixedTaps(image.width, width, columns), rows);
  return verticalTaps === undefined ? band : resampleVertical(band, verticalTaps, rows.offset);
};

/**
 * RGB8 を `width × height` へ LANCZOS でリサンプルする（{@link resizeRgb8LanczosWindow} の窓 = 全体）。
 */
export const resizeRgb8Lanczos = (image: Rgb8Image, width: number, height: number): Rgb8Image =>
  resizeRgb8LanczosWindow(image, width, height, { left: 0, top: 0, width, height });
