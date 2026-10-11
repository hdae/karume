/**
 * Pillow の 8bit の分離型リサンプル — `Image.resize` の 8bit 経路（`libImaging/Resample.c` の
 * `ImagingResampleInner` と `*_8bpc`）の逐語の移植。フィルタ（重み関数と台の半径）とチャネル数を
 * 引数に取る**共通の核**で、Pillow の値をビット単位で再現する経路はここに 1 本だけ置く（Pillow の版を
 * 上げるときの追従を 1 か所にする — ADR 0125）。
 *
 * 載っている経路:
 *
 * - {@link resizeU8PlaneBicubic}（1 バンド・BICUBIC）: BiRefNet の α マットの `u8-pil-bicubic`
 *   （`src/birefnet/matte.ts` — HR の上流の `pred_pil.resize(image.size)`）。
 *
 * ## `preprocess.ts` の `resizeRgb8` と経路を分けている理由
 *
 * MUST: `resizeRgb8` の経路（f64 の重み・`/ filterScale`・`floor(v + 0.5)`）にカーネルを足すだけでは
 * Pillow と一致しない。Pillow の 8bit 経路は係数を 22 bit の固定小数点の整数へ丸め、積和を整数で
 * 取る。f64 のままだと丸め境界に載った標本が 1〜2 LSB ずれる。一方 `resizeRgb8` は torchvision の
 * antialias 経路に合わせた SigLIP2 / Depth Anything / BiRefNet の前処理で、こちらに寄せると彼らの
 * 出力が動く。よって経路を分ける（公開の `resizeRgb8` / `ResampleFilter` は広げない — ADR 0008）。
 *
 * ## 逐語に写している点（Pillow 12.3.0 — 版を記録した fixture が縛る）
 *
 * - 係数の引数は `(x + xmin − center + 0.5) · (1 / filterscale)`（逆数を掛ける — Pillow の形のまま）
 * - `xmin` / `xmax` は `(int)` の切り捨ての後に `[0, inSize]` へ収める
 * - 総和 `ww` が 0 でないときだけ正規化する
 * - 係数は `(int)(±0.5 + w·2^22)`（符号で丸めの向きを変える）
 * - 積和は整数で、初期値 2^21 から `>> 22` して 0〜255 に飽和する（`clip8`）
 * - 横 → 縦の順・中間は uint8・寸法が変わらない軸はパスを飛ばす（`need_horizontal` /
 *   `need_vertical`）。両軸とも変わらなければ写し（`ImagingCopy` — `Image.resize` も同寸なら
 *   `copy()` を返す）
 * - 横パスは縦パスが読む入力の行（`ybox_first`〜`ybox_last`）だけを回す
 * - チャネルごとの積和は互いに独立（Pillow の 1 バンドの枝 `image8` と多バンドの枝は同じ式）
 *
 * ## 窓だけを計算する（{@link resampleU8Window}）
 *
 * 係数は出力の添字ごとに独立し（`center = (i + 0.5)·scale`）、中間の uint8 も出力の列ごとに独立して
 * いる。よって伸縮後の画像の一部の矩形は、その矩形の出力の添字だけを計算すれば、全体を伸縮してから
 * 切り出した値と uint8 で同じになる。
 *
 * NOTE: `export` は `src/` の内部の呼び手とテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */

/** Pillow のフィルタ（`Resample.c` の `struct filter` — 重み関数と、縮尺 1 のときの台の半径）。 */
export type PillowFilter = {
  readonly support: number;
  readonly weight: (x: number) => number;
};

/**
 * Pillow の `bicubic_filter`（`a = −0.5`・式の形と演算の順序は `Resample.c` のまま）。`|x| ∈ [1, 2)` で
 * 負の重みを持つので、段差では値域を越えた積和が出る（{@link clip8} の飽和が効く経路）。
 *
 * NOTE: `preprocess.ts` の `cubic` と同じ式だが共有しない — あちらは torchvision の antialias 経路の
 * 写しで、こちらは Pillow の逐語の写し（正本が違うので、片方の版の追従がもう片方を動かさない形に保つ）。
 */
const bicubic = (x: number): number => {
  const a = -0.5;
  const abs = x < 0 ? -x : x;
  if (abs < 1) return ((a + 2) * abs - (a + 3)) * abs * abs + 1;
  if (abs < 2) return (((abs - 5) * abs + 8) * abs - 4) * a;
  return 0;
};

/** Pillow の BICUBIC（`Resample.c` の `BICUBIC = {bicubic_filter, 2.0}`）。 */
export const PILLOW_BICUBIC: PillowFilter = { support: 2, weight: bicubic };

/** 係数の固定小数点の小数部のビット数（Pillow の `PRECISION_BITS` = 32 − 8 − 2）。 */
const PRECISION_BITS = 32 - 8 - 2;

/** 係数の 1.0（`1 << PRECISION_BITS`）。 */
const FIXED_ONE = 1 << PRECISION_BITS;

/** 積和の初期値（`1 << (PRECISION_BITS - 1)` — `>> PRECISION_BITS` を四捨五入にする半分）。 */
const FIXED_HALF = 1 << (PRECISION_BITS - 1);

/**
 * 8bit の画素列（行優先・画素あたり `channels` バイト）と、その寸法。返す側は新しく確保した
 * `ArrayBuffer` の配列（`U8Image<ArrayBuffer>`）で、受ける側は共有メモリも受ける。
 */
export type U8Image<Buffer extends ArrayBufferLike = ArrayBufferLike> = {
  readonly data: Uint8Array<Buffer>;
  readonly width: number;
  readonly height: number;
  readonly channels: number;
};

/** 伸縮後の画像から切り出す矩形（左上 `left`, `top` から `width × height`）。 */
export type PixelWindow = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

/** 出力 1 点ぶんの入力範囲と固定小数点の係数。 */
type FixedTap = {
  readonly start: number;
  readonly coefficients: Int32Array<ArrayBuffer>;
};

/** 1 軸の出力の窓（伸縮後の添字 `offset` から `length` 個）。 */
type AxisWindow = {
  readonly offset: number;
  readonly length: number;
};

/**
 * 1 軸ぶんの係数（Pillow の `precompute_coeffs` → `normalize_coeffs_8bpc`）のうち、出力の窓 `window` の
 * 添字の分（添字ごとに独立 — モジュール doc「窓だけを計算する」）。
 *
 * NOTE: Pillow の `scale` は `(double)(in1 − in0) / outSize`（`in0` / `in1` は float の箱）で、箱を取らない
 * resize では `inSize / outSize` と同じ値になる（2^24 未満の整数は float で厳密）。
 */
const fixedTaps = (
  inSize: number,
  outSize: number,
  filter: PillowFilter,
  window: AxisWindow,
): readonly FixedTap[] => {
  const scale = inSize / outSize;
  const filterScale = scale < 1 ? 1 : scale;
  const support = filter.support * filterScale;
  const inverseFilterScale = 1 / filterScale;
  const taps: FixedTap[] = [];
  for (let index = window.offset; index < window.offset + window.length; index += 1) {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const count = Math.min(inSize, Math.trunc(center + support + 0.5)) - start;
    const weights = new Float64Array(count);
    let total = 0;
    for (let x = 0; x < count; x += 1) {
      const weight = filter.weight((x + start - center + 0.5) * inverseFilterScale);
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
 * 積和は Pillow と同じく int32 に収まる（|係数の和| ≲ 1.3·2^22 × 255 + 2^21 < 2^31）ので、`>>` の
 * ToInt32 は値を変えない。
 */
const clip8 = (sum: number): number => {
  const value = sum >> PRECISION_BITS;
  return value < 0 ? 0 : value > 255 ? 255 : value;
};

/**
 * 横パス（Pillow の `ImagingResampleHorizontal_8bpc`）。入力の行 `rows` だけを回す（Pillow の
 * `ybox_first`〜`ybox_last`）。出力の列は `taps` の本数（窓の列）。
 */
const resampleHorizontal = (
  image: U8Image,
  taps: readonly FixedTap[],
  rows: AxisWindow,
): U8Image<ArrayBuffer> => {
  const { channels } = image;
  const width = taps.length;
  const sourceStride = image.width * channels;
  const targetStride = width * channels;
  const data = new Uint8Array(rows.length * targetStride);
  for (let y = 0; y < rows.length; y += 1) {
    const sourceRow = (rows.offset + y) * sourceStride;
    const targetRow = y * targetStride;
    for (let x = 0; x < width; x += 1) {
      const { start, coefficients } = taps[x];
      for (let channel = 0; channel < channels; channel += 1) {
        let sum = FIXED_HALF;
        for (let j = 0; j < coefficients.length; j += 1) {
          sum += image.data[sourceRow + (start + j) * channels + channel] * coefficients[j];
        }
        data[targetRow + x * channels + channel] = clip8(sum);
      }
    }
  }
  return { data, width, height: rows.length, channels };
};

/**
 * 縦パス（Pillow の `ImagingResampleVertical_8bpc`）。`image` の行 0 は入力の行 `rowOffset`（Pillow が
 * 横パスの後に `bounds_vert` から `ybox_first` を引くのと同じ読み替え）。出力の行は `taps` の本数
 * （窓の行）。
 */
const resampleVertical = (
  image: U8Image,
  taps: readonly FixedTap[],
  rowOffset: number,
): U8Image<ArrayBuffer> => {
  const stride = image.width * image.channels;
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
  return { data, width: image.width, height: taps.length, channels: image.channels };
};

/** 列 `columns`・行 `rows` の矩形の写し（パスを飛ばした軸の切り出し — 範囲の内側だけを呼ぶ）。 */
const cropU8 = (image: U8Image, columns: AxisWindow, rows: AxisWindow): U8Image<ArrayBuffer> => {
  const { channels } = image;
  const rowBytes = columns.length * channels;
  const data = new Uint8Array(rows.length * rowBytes);
  for (let y = 0; y < rows.length; y += 1) {
    const from = ((rows.offset + y) * image.width + columns.offset) * channels;
    data.set(image.data.subarray(from, from + rowBytes), y * rowBytes);
  }
  return { data, width: columns.length, height: rows.length, channels };
};

/**
 * 縦パスが読む入力の行（Pillow の `ybox_first`〜`ybox_last`）。tap の起点も終点も出力の添字について
 * 単調なので、両端の tap で決まる。
 */
const rowsReadBy = (taps: readonly FixedTap[]): AxisWindow => {
  const first = taps[0];
  const last = taps[taps.length - 1];
  return { offset: first.start, length: last.start + last.coefficients.length - first.start };
};

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/**
 * 8bit の画像を `width × height` へ `filter` でリサンプルし（Pillow の `Image.resize` と uint8 で一致 —
 * モジュール doc）、その矩形 `window` を切り出したものを返す。計算するのは窓の出力だけ（全体を伸縮
 * してから切り出した値と同じ）。縦横比は保たない。寸法が変わらない軸はパスを飛ばし（その軸は入力の
 * 写しを切り出す）、両軸とも変わらなければ写しを返す。
 *
 * 入力・出力寸法・窓の検査は内部の事前条件（呼び手は検査済みの画像と、受理集合や資産から導いた寸法
 * しか渡さない）なので `RangeError`。呼び手が受けた画像そのものの検査（`ModelInputError`）は呼び手が
 * 先に持つ。
 */
export const resampleU8Window = (
  image: U8Image,
  width: number,
  height: number,
  window: PixelWindow,
  filter: PillowFilter,
): U8Image<ArrayBuffer> => {
  const { channels } = image;
  if (
    !isPositiveInteger(image.width) || !isPositiveInteger(image.height) ||
    !isPositiveInteger(channels) || image.data.length !== image.width * image.height * channels
  ) {
    throw new RangeError(
      `8bit の入力 ${image.width}×${image.height}×${channels} と長さ ${image.data.length} が合わない`,
    );
  }
  if (!isPositiveInteger(width) || !isPositiveInteger(height)) {
    throw new RangeError(`リサンプルの出力寸法 ${width}×${height} が正の整数でない`);
  }
  const { left, top } = window;
  if (
    !Number.isInteger(left) || !Number.isInteger(top) || left < 0 || top < 0 ||
    !isPositiveInteger(window.width) || !isPositiveInteger(window.height) ||
    left + window.width > width || top + window.height > height
  ) {
    throw new RangeError(
      `リサンプルの窓 (${left}, ${top}) ${window.width}×${window.height} が出力 ${width}×${height}` +
        " の内側でない",
    );
  }
  const columns = { offset: left, length: window.width };
  const verticalTaps = height === image.height
    ? undefined
    : fixedTaps(image.height, height, filter, { offset: top, length: window.height });
  // 横パスが回す入力の行（縦を飛ばすなら窓の行そのもの）。
  const rows = verticalTaps === undefined
    ? { offset: top, length: window.height }
    : rowsReadBy(verticalTaps);
  const band = width === image.width
    ? cropU8(image, columns, rows)
    : resampleHorizontal(image, fixedTaps(image.width, width, filter, columns), rows);
  return verticalTaps === undefined ? band : resampleVertical(band, verticalTaps, rows.offset);
};

/**
 * 8bit の 1 バンドの平面（行優先）を `width × height` へ Pillow の BICUBIC でリサンプルする（Pillow の
 * L モードの `Image.resize((width, height))` — resample の既定 — と uint8 で一致）。
 */
export const resizeU8PlaneBicubic = (
  plane: Uint8Array<ArrayBufferLike>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Uint8Array<ArrayBuffer> =>
  // 写しの経路（両軸とも同寸）も含め、返るのは新しく確保した配列（入力と領域を共有しない）。
  resampleU8Window(
    { data: plane, width: sourceWidth, height: sourceHeight, channels: 1 },
    width,
    height,
    { left: 0, top: 0, width, height },
    PILLOW_BICUBIC,
  ).data;
