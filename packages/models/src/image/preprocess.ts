/**
 * RGB8 の画素列 → モデル入力（resize → rescale → normalize → NCHW f32）。
 *
 * **パイプライン非依存の共通処理**（画像を入力に取るモデルは総じてこの 3 段を通る）なので、
 * ファミリのディレクトリではなく `src/image/` に置く（`png.ts` と同じ位置づけ）。
 *
 * **decode は karume の責務ではない**。入口は「RGB8 の画素列 + 幅 + 高さ」で、PNG / JPEG の
 * デコーダは持たない（ブラウザなら `createImageBitmap` + canvas、Deno なら任意のデコーダ）。
 * 一方で resize / rescale / normalize は**モデルカードの一部**（定数を間違えると静かに精度が
 * 落ちる）なので、利用者ではなくこちらが持つ。
 *
 * 画素ごとの出力を持つモデル（セグメンテーションのマット）は、焼かれた解像度で出た地図を
 * 元画像の解像度へ**戻す**段も通る。台と重みは入口と同じものなので、その 1 本
 * （{@link resizePlaneF32}）もここが持つ。
 *
 * ## MUST: 合わせている参照実装
 *
 * transformers 5.14.1 の **`SiglipImageProcessor`（`TorchvisionBackend`）** — v5 の
 * `AutoImageProcessor` が既定で返す fast 側で、PIL 側（`SiglipImageProcessorPil`）ではない。
 * resize は `torchvision.transforms.v2.functional.resize(..., antialias=True)` で、この
 * antialias 経路は Pillow の `ImagingResample` と同じ「support を縮尺で伸ばす分離型
 * リサンプリング」（{@link buildTaps}）。
 *
 * MUST: 補間フィルタは**モデルカードの一部**なので呼び出し側が明示する（既定は bilinear）。
 * `preprocessor_config.json` の `"resample"` は **PIL の定数**（NEAREST=0 / LANCZOS=1 /
 * **BILINEAR=2** / **BICUBIC=3** / BOX=4 / HAMMING=5）で、SigLIP2 は 2 = bilinear、
 * Depth Anything V2（`DPTImageProcessor`）は 3 = bicubic。transformers のクラス属性の既定は
 * どちらのクラスも `PILImageResampling.BICUBIC` なので「SigLIP は bicubic」と読み違えやすいが、
 * SigLIP2 はチェックポイントの config が既定を上書きしている。取り違えの実測は SigLIP2 側で
 * 最大 47/255、DA-V2 側で `pixel_values` の最大 0.59（1 LSB = 0.0175 の 34 倍）。
 * `resample` の実測は `tools/exporter/siglip2_preprocess.py` と
 * `tools/exporter/export_depth_anything.py` が emit のたびに行う。
 *
 * MUST: 2 パスの**間**で uint8 へ丸め直す（{@link resizeRgb8}）。PIL / torchvision の uint8
 * 経路は中間バッファを uint8 で持つので、f64 のまま縦パスへ渡すと参照とずれる標本が
 * **0.26% → 9.5%** へ跳ねる（18 幾何 × 224×224 = 2,709,504 標本の実測）。
 */

/** RGB8 の画素列（行優先・画素あたり 3 バイト）と、その寸法。 */
export type Rgb8Image = {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
};

/** チャネル数（RGB）。アルファは入口で受け取らない。 */
const CHANNELS = 3;

/** リサンプルの補間フィルタ（`preprocessor_config.json` の `resample` に対応する綴り）。 */
export type ResampleFilter = "bilinear" | "bicubic";

/** 1 つのフィルタ = 重み関数とその台の半径（縮小時はここに縮尺が掛かって台が伸びる）。 */
type Kernel = {
  /** 台の半径（縮尺 1 のときの値）。 */
  readonly support: number;
  readonly weight: (x: number) => number;
};

/** 三角（bilinear）フィルタ。 */
const triangle = (x: number): number => {
  const abs = Math.abs(x);
  return abs < 1 ? 1 - abs : 0;
};

/**
 * Catmull-Rom 系の三次フィルタ（PIL の `bicubic_filter` / torchvision の antialias 経路と
 * 同じ `a = -0.5`）。`|x| ∈ [1, 2)` で**負の重み**を持つので、階段状のエッジでは入力の値域を
 * 越えた overshoot が出る（{@link round8} の clamp が効く唯一の経路）。
 */
const cubic = (x: number): number => {
  const a = -0.5;
  const abs = Math.abs(x);
  if (abs < 1) return ((a + 2) * abs - (a + 3)) * abs * abs + 1;
  if (abs < 2) return (((abs - 5) * abs + 8) * abs - 4) * a;
  return 0;
};

const KERNELS: Readonly<Record<ResampleFilter, Kernel>> = {
  bilinear: { support: 1, weight: triangle },
  bicubic: { support: 2, weight: cubic },
};

/** 出力 1 点ぶんの入力範囲と重み。 */
type Tap = {
  readonly start: number;
  readonly weights: Float64Array;
};

/**
 * 1 軸ぶんのリサンプル重みを組む（Pillow / torchvision の antialias 経路と同じ式）。
 *
 * 縮小（`scale > 1`）では台を縮尺ぶん伸ばして平均化し、拡大では台を 1 に固定する。端では
 * 台が入力の外へはみ出した分を**切り詰めて**から総和 1 に正規化する（`ImagingResample` の
 * 規約）— この切り詰めのせいで、拡大の端は近傍 1 点そのままになることがある。
 *
 * 重み和が 0 にならないことは式から出る: `center` を含む入力位置は必ず `[start, stop)` に
 * 入り、その重みは（bilinear で 0.5 / bicubic で 0.75 以上）正の値を持つ。負のローブを持つ
 * bicubic でも総和が 0 へ潰れる組み合わせは無いので、0 除算の分岐は要らない。
 */
const buildTaps = (inSize: number, outSize: number, kernel: Kernel): readonly Tap[] => {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = kernel.support * filterScale;
  const taps: Tap[] = [];
  for (let index = 0; index < outSize; index += 1) {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const stop = Math.min(inSize, Math.floor(center + support + 0.5));
    const weights = new Float64Array(stop - start);
    let total = 0;
    for (let j = 0; j < weights.length; j += 1) {
      const weight = kernel.weight((start + j - center + 0.5) / filterScale);
      weights[j] = weight;
      total += weight;
    }
    for (let j = 0; j < weights.length; j += 1) weights[j] /= total;
    taps.push({ start, weights });
  }
  return taps;
};

/**
 * 丸めて 8bit に収める。
 *
 * MUST: clamp を外さない。`Uint8Array` への代入は範囲外を **mod 256 で巻き戻す**ので、
 * 負のローブを持つ bicubic の overshoot が、飽和すべき画素の黒白反転として静かに通る
 * （bilinear は重みが非負で総和 1 なので掛からない経路）。
 */
const round8 = (value: number): number => Math.min(255, Math.max(0, Math.floor(value + 0.5)));

/** 入口の契約（正の整数寸法・長さの整合）を検査する。 */
const assertRgb8 = (image: Rgb8Image): void => {
  const { data, width, height } = image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`画像サイズ ${width}×${height} が正の整数でない`);
  }
  if (data.length !== width * height * CHANNELS) {
    throw new Error(`RGB8 の長さ ${data.length} が 3×${width}×${height} と違う`);
  }
};

/**
 * RGB8 を指定寸法へリサンプルする（antialias 付き — モジュール docstring の参照実装）。
 *
 * アスペクト比は**保たない**（`size.height` / `size.width` へそのまま伸縮する）。SigLIP2 の
 * `preprocessor_config.json` が `size` を高さ・幅の対で持ち、crop も pad も無いため。DA-V2 は
 * `keep_aspect_ratio` を立てているが、正方入力 → 正方 `size` では恒等になる（アスペクト比を
 * 保つ経路が要るモデルが出るまで、この関数は寸法を受けるだけに留める）。
 *
 * `filter` の既定が bilinear なのは既存の呼び出し側（SigLIP2 / BiRefNet）の値で、**モデル
 * カードの一部**なので新しいファミリは必ず自分の `resample` を明示する。
 */
export const resizeRgb8 = (
  image: Rgb8Image,
  width: number,
  height: number,
  filter: ResampleFilter = "bilinear",
): Rgb8Image => {
  assertRgb8(image);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`出力サイズ ${width}×${height} が正の整数でない`);
  }
  const kernel = KERNELS[filter];
  const horizontal = buildTaps(image.width, width, kernel);
  const middleStride = width * CHANNELS;
  const middle = new Uint8Array(image.height * middleStride);
  for (let y = 0; y < image.height; y += 1) {
    const sourceRow = y * image.width * CHANNELS;
    const targetRow = y * middleStride;
    for (let x = 0; x < width; x += 1) {
      const { start, weights } = horizontal[x];
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        let sum = 0;
        for (let j = 0; j < weights.length; j += 1) {
          sum += image.data[sourceRow + (start + j) * CHANNELS + channel] * weights[j];
        }
        middle[targetRow + x * CHANNELS + channel] = round8(sum);
      }
    }
  }

  const vertical = buildTaps(image.height, height, kernel);
  const data = new Uint8Array(height * middleStride);
  for (let y = 0; y < height; y += 1) {
    const { start, weights } = vertical[y];
    const targetRow = y * middleStride;
    for (let x = 0; x < width; x += 1) {
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        let sum = 0;
        for (let j = 0; j < weights.length; j += 1) {
          sum += middle[(start + j) * middleStride + x * CHANNELS + channel] * weights[j];
        }
        data[targetRow + x * CHANNELS + channel] = round8(sum);
      }
    }
  }
  return { data, width, height };
};

/**
 * **単一チャネルの f32 平面**を指定寸法へリサンプルする（{@link buildTaps} を
 * {@link resizeRgb8} と共有する・フィルタは bilinear 固定）。
 *
 * 画素ごとの出力を持つモデル（セグメンテーションのマット等）は、グラフが焼かれた解像度で
 * 出した地図を**元画像の解像度へ戻す**必要がある。参照実装は
 * `torchvision.transforms.functional.resize`（テンソル枝・bilinear・antialias 既定）で、
 * 重みの組み方は {@link resizeRgb8} が合わせている Pillow の `ImagingResample` と同じ。
 *
 * MUST: {@link resizeRgb8} と違って**2 パスの間で丸めない**。あちらの `round8` は PIL /
 * torchvision の **uint8 経路**が中間バッファを uint8 で持つことに合わせた再現で、f32 の
 * テンソル枝には無い段。ここで真似ると参照から離れるうえ、`[0, 1]` のマットが 8bit の段に
 * 潰れて境界の勾配が失われる。
 *
 * 重みは非負で総和 1 なので、入力の値域は出力でも保たれる（`[0, 1]` のマットが範囲外へ
 * 出ることはない — clamp を置いていないのはこの理由）。
 */
export const resizePlaneF32 = (
  plane: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Float32Array<ArrayBuffer> => {
  for (
    const [size, label] of [[sourceWidth, "入力幅"], [sourceHeight, "入力高さ"], [
      width,
      "出力幅",
    ], [height, "出力高さ"]] as const
  ) {
    if (!Number.isInteger(size) || size <= 0) {
      throw new RangeError(`${label} ${size} が正の整数でない`);
    }
  }
  if (plane.length !== sourceWidth * sourceHeight) {
    throw new Error(`平面の長さ ${plane.length} が ${sourceWidth}×${sourceHeight} と違う`);
  }

  const horizontal = buildTaps(sourceWidth, width, KERNELS.bilinear);
  const middle = new Float64Array(sourceHeight * width);
  for (let y = 0; y < sourceHeight; y += 1) {
    const sourceRow = y * sourceWidth;
    const targetRow = y * width;
    for (let x = 0; x < width; x += 1) {
      const { start, weights } = horizontal[x];
      let sum = 0;
      for (let j = 0; j < weights.length; j += 1) sum += plane[sourceRow + start + j] * weights[j];
      middle[targetRow + x] = sum;
    }
  }

  const vertical = buildTaps(sourceHeight, height, KERNELS.bilinear);
  const out = new Float32Array(width * height) as Float32Array<ArrayBuffer>;
  for (let y = 0; y < height; y += 1) {
    const { start, weights } = vertical[y];
    const targetRow = y * width;
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let j = 0; j < weights.length; j += 1) {
        sum += middle[(start + j) * width + x] * weights[j];
      }
      out[targetRow + x] = sum;
    }
  }
  return out;
};

/**
 * rescale の除数（`rescale_factor` = 1/255 の逆数）。
 * f64 で `1 / 0.00392156862745098` はちょうど 255.0（Python 側が毎回実測する）。
 */
const RESCALE_DIVISOR = 255;

/**
 * RGB8（インターリーブ）→ 正規化済み planar f32 `[3, height, width]`（batch 1 は畳む）。
 *
 * MUST: rescale と normalize を**畳んだ形**で計算する — 参照実装
 * （`TorchvisionBackend._fuse_mean_std_and_rescale_factor`）は `mean · 255` と `std · 255` を
 * 先に作って `(x - mean) / std` を 1 回だけ掛ける。`x / 255` を先に通すと丸めが 1 回増え、
 * 素の f32 比較でずれる（Python 側はこの畳んだ形とフルパイプラインのビット同一を実測）。
 *
 * `mean` / `std` は `[0, 1]` 尺度（`preprocessor_config.json` の綴りのまま）で受ける。
 */
export const normalizeToNchw = (
  image: Rgb8Image,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array<ArrayBuffer> => {
  assertRgb8(image);
  const plane = image.width * image.height;
  const out = new Float32Array(plane * CHANNELS) as Float32Array<ArrayBuffer>;
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    const offset = mean[channel] * RESCALE_DIVISOR;
    const divisor = std[channel] * RESCALE_DIVISOR;
    for (let index = 0; index < plane; index += 1) {
      out[channel * plane + index] = (image.data[index * CHANNELS + channel] - offset) / divisor;
    }
  }
  return out;
};
