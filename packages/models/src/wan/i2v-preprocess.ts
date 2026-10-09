/**
 * Wan2.2 TI2V の I2V の条件画像の前処理（ホスト・GPU に触らない — ADR 0121 決定 11・段 9a）:
 * 出力寸法の選択 → 寸法合わせ（crop / stretch・Pillow の LANCZOS）→ `[-1, 1]` → patchify（VAE encoder の入力
 * `[3·p², 1, H/p, W/p]`）。encoder の出口の mu の正規化は `latents.ts` の `normalizeWanLatents`。
 *
 * 寸法合わせの正本は 2 つ:
 *
 * - fit = `"crop"`（既定）: 公式 Wan2.2 の `textimage2video.py` の i2v — `scale = max(ow / iw, oh / ih)` →
 *   `round(iw·scale)`・`round(ih·scale)`（Python の `round`）→ LANCZOS → 中央クロップ `(rw − ow) // 2` →
 *   `TF.to_tensor(img).sub_(0.5).div_(0.5)`
 * - fit = `"stretch"`: diffusers の `WanImageToVideoPipeline`（`VideoProcessor.preprocess` の resize_mode `"default"`
 *   = `(width, height)` へ直接 LANCZOS・`reducing_gap` 無し → `2·(x / 255) − 1`）
 *
 * 出力寸法の選び方だけは公式と違う: 公式は面積の上限の下で自由な寸法（16:9 の画像なら 1248×704）を選ぶが、ここは
 * 受理集合（記述子の `acceptedSizes`）の中から公式の比較式で選ぶ（ADR 0121 の追記「受理寸法を公式の 2 寸法へ」）。
 *
 * NOTE: `export` はパイプラインの内部とテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */

import { ModelInputError } from "../errors.ts";
import { resizeRgb8Lanczos, resizeRgb8LanczosWindow } from "../image/lanczos.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";

/** 条件画像の寸法の合わせ方（公式の中央クロップ / diffusers の直接の伸縮 — モジュール doc）。 */
export type WanI2vFit = "crop" | "stretch";

/** 省いたときの合わせ方（公式 Wan2.2 の I2V と同じ）。 */
const DEFAULT_FIT: WanI2vFit = "crop";

/** チャネル数（RGB）。 */
const CHANNELS = 3;

type Size = { readonly width: number; readonly height: number };

/**
 * 公式の比較式 `max(r / rc, rc / r)`（r = 画像の幅 / 高さ・rc = 候補の幅 / 高さ）を `max(a, b) / min(a, b)`
 * （a = iw·ch・b = ih·cw — 整数の積）の形で計算する。数学的には同じ値で、丸めが割り算 1 回だけになるので、転置した
 * 2 候補に対する正方形の画像が浮動小数の丸めに依らず厳密に同点になる（公式の形は割り算を 3 回重ねる）。
 */
const aspectDistance = (image: Size, candidate: Size): number => {
  const a = image.width * candidate.height;
  const b = image.height * candidate.width;
  return Math.max(a, b) / Math.min(a, b);
};

const isLandscape = ({ width, height }: Size): boolean => width >= height;

/** 要求が明示した寸法の欄の綴り（拒否の文言用）。 */
const describeRequested = (requested: Partial<Size>): string => {
  if (requested.width !== undefined && requested.height !== undefined) {
    return `${requested.width}×${requested.height}`;
  }
  return requested.width !== undefined ? `width ${requested.width}` : `height ${requested.height}`;
};

/**
 * I2V の出力寸法を選ぶ: 受理集合（記述子の `acceptedSizes`）のうち、要求が明示した欄（width / height）に合う候補の
 * 中から、画像の縦横比に最も近いもの。近さは公式 Wan2.2 の `best_output_size` の比較式（{@link aspectDistance}）で
 * 測り、同点は横長（幅 ≥ 高さ）を取る。受理集合が 2 寸法（1280×704 / 704×1280）なら「画像の幅 ≥ 高さ → 1280×704」と
 * 同じ結果になる。
 *
 * MUST: 差の絶対値 `|r − rc|` で測らない — 2 寸法では同点が r ≈ 1.18 へ移り、正方形の画像が縦長になる（公式と逆）。
 *
 * 両方を明示すれば完全一致の 1 つ、片方だけならその欄が合う候補の中から縦横比で選ぶ。合う候補が無ければ
 * `ModelInputError`（受理集合の外 — 入力起因）。
 */
export const selectWanI2vSize = (
  image: Rgb8Image,
  requested: Partial<Size>,
  generation: Pick<WanGenerationDescriptor, "acceptedSizes">,
): Size => {
  assertRgb8(image);
  const candidates = generation.acceptedSizes.filter((size) =>
    (requested.width === undefined || size.width === requested.width) &&
    (requested.height === undefined || size.height === requested.height)
  );
  if (candidates.length === 0) {
    throw new ModelInputError(
      `${describeRequested(requested)} が受理集合（${
        generation.acceptedSizes.map((size) => `${size.width}×${size.height}`).join(" / ")
      }）に無い`,
    );
  }
  let best = candidates[0];
  let bestDistance = aspectDistance(image, best);
  for (const candidate of candidates.slice(1)) {
    const distance = aspectDistance(image, candidate);
    if (
      distance < bestDistance ||
      (distance === bestDistance && isLandscape(candidate) && !isLandscape(best))
    ) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return { width: best.width, height: best.height };
};

/**
 * Python の `round(x)`（float — .5 ちょうどは偶数側）。`x − floor(x)` は |x| < 2^52 で厳密なので、.5 の判定に
 * 丸めは入らない。
 */
const roundHalfEven = (value: number): number => {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
};

/**
 * fit = `"crop"` の縮尺後の寸法（公式: `scale = max(ow / iw, oh / ih)` → `round(iw·scale)`・`round(ih·scale)`）。
 * 縮尺は覆う側（両軸とも目標以上）を取る。
 *
 * MUST: 丸めは Python の `round`（.5 ちょうどは偶数側）。JS の `Math.round`（.5 は +∞ 側）で写すと、例えば
 * 512×301 → 1280×704 で 301 × 2.5 = 752.5 が 753 になり、中央クロップの位置ごと 1 px ずれる。
 */
export const wanI2vCoverSize = (image: Size, target: Size): Size => {
  const scale = Math.max(target.width / image.width, target.height / image.height);
  return {
    width: roundHalfEven(image.width * scale),
    height: roundHalfEven(image.height * scale),
  };
};

/**
 * 画像を出力寸法 `size` に合わせる（モジュール doc の 2 つの正本）。`"crop"` は覆う寸法（{@link wanI2vCoverSize}）へ
 * LANCZOS で伸縮してから中央を切り出し（`(rw − ow) // 2` — 余りの 1 px は右・下に残る）、`"stretch"` は `size` へ
 * 直接 LANCZOS で伸縮する（縦横比は保たない）。
 *
 * crop は覆う寸法の全体を作らず、切り出す窓の出力だけを計算する（`resizeRgb8LanczosWindow` — 全体を伸縮してから
 * 切り出した値と uint8 で同じ）。全体を作ると、縦横比の極端な入力で手間と中間が際限なく膨らむ（例 4096×4 →
 * 1280×704 は覆う寸法が 720,896×704 で、中間だけで 1.5 GB）。
 *
 * NOTE: fit の綴りの拒否は素の `Error`（名前の綴り違い — ADR 0107 決定 3 の sampler 名と同じ扱い）。要求の欄として
 * 受ける門（段 9b）が別の型にするなら、そちらで先に弾く。
 */
export const fitWanI2vImage = (image: Rgb8Image, size: Size, fit: WanI2vFit): Rgb8Image => {
  if (fit === "stretch") return resizeRgb8Lanczos(image, size.width, size.height);
  if (fit !== "crop") {
    throw new Error(`I2V の fit ${JSON.stringify(fit)} は未対応（期待 'crop' / 'stretch'）`);
  }
  const cover = wanI2vCoverSize(image, size);
  // 縮尺を決めた側の軸は `iw·(ow / iw)` の丸め誤差（0.5 未満）が丸めで消えて目標ちょうど、もう片方の軸は目標以上
  // になる。下回るなら式の写し間違い（公式も `assert img.width == ow and img.height == oh` で同じことを見る）。
  if (cover.width < size.width || cover.height < size.height) {
    throw new Error(
      `I2V の crop: 縮尺後の ${cover.width}×${cover.height} が出力 ${size.width}×${size.height} を覆わない`,
    );
  }
  return resizeRgb8LanczosWindow(image, cover.width, cover.height, {
    left: Math.floor((cover.width - size.width) / 2),
    top: Math.floor((cover.height - size.height) / 2),
    width: size.width,
    height: size.height,
  });
};

/**
 * RGB8（インターリーブ）→ planar `[3, H, W]` の `[-1, 1]`: `f32(2·f32(x / 255) − 1)`。
 *
 * MUST: この順で書く — 公式の `TF.to_tensor(img).sub_(0.5).div_(0.5)` と diffusers の `2·(x / 255) − 1` は f32 で
 * ビット一致し、この式がその値（f64 で計算して f32 へ 1 回丸める形は、f64 の精度が f32 の 2 倍 + 2 bit 以上あるので
 * f32 の演算と同じ値）。`normalizeToNchw`（mean = std = 0.5 の畳んだ形 `(x − 127.5) / 127.5`）は 256 値のうち
 * 128 値で最終ビットがずれるので使わない。
 */
export const wanI2vPixels = (image: Rgb8Image): Float32Array<ArrayBuffer> => {
  assertRgb8(image);
  const plane = image.width * image.height;
  const out = new Float32Array(CHANNELS * plane);
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    for (let index = 0; index < plane; index += 1) {
      const unit = Math.fround(image.data[index * CHANNELS + channel] / 255);
      out[channel * plane + index] = Math.fround(2 * unit - 1);
    }
  }
  return out;
};

/**
 * patchify: planar `[C, H, W]`（1 フレーム）→ `[C·p², H/p, W/p]`（上流 diffusers の `patchify` と同じ並び —
 * `view(c, f, h, r, w, q).permute(c, q, r, f, h, w)`・r = q = p）。添字の対応は
 * `out[c·p² + dx·p + dy, y, x] = in[c, y·p + dy, x·p + dx]`（チャネル内のずれは幅方向 dx が上位・高さ方向 dy が
 * 下位）— `vae-tiles.ts` の `unpatchifyWanVaeFrames` の逆。VAE encoder の入力 `[C·p², 1, H/p, W/p]` は同じバイト列。
 *
 * 写しだけで浮動小数の演算は無い（値はビット単位で保たれる）。形・倍率・長さの食い違いは fail loudly。
 */
export const patchifyWanImage = (
  pixels: Float32Array,
  shape: readonly [number, number, number],
  patchSize: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(patchSize) || patchSize < 1) {
    throw new Error(`patchify の倍率 ${patchSize} が正の整数でない`);
  }
  if (shape.some((dim) => !Number.isInteger(dim) || dim < 1)) {
    throw new Error(`patchify の入力の形 [${shape.join(", ")}] の次元が正の整数でない`);
  }
  const [channels, height, width] = shape;
  if (height % patchSize !== 0 || width % patchSize !== 0) {
    throw new Error(`patchify の入力 ${width}×${height} が倍率 ${patchSize} で割り切れない`);
  }
  if (pixels.length !== channels * height * width) {
    throw new Error(
      `patchify の入力の要素数 ${pixels.length} が [${shape.join(", ")}]（${
        channels * height * width
      }）と違う`,
    );
  }
  const rows = height / patchSize;
  const columns = width / patchSize;
  const out = new Float32Array(pixels.length);
  // 出力の並び（c → dx → dy → y → x）で回すので、書き込み先は 1 ずつ進む。
  let to = 0;
  for (let channel = 0; channel < channels; channel += 1) {
    const plane = channel * height * width;
    for (let dx = 0; dx < patchSize; dx += 1) {
      for (let dy = 0; dy < patchSize; dy += 1) {
        for (let y = 0; y < rows; y += 1) {
          const row = plane + (y * patchSize + dy) * width + dx;
          for (let x = 0; x < columns; x += 1) {
            out[to] = pixels[row + x * patchSize];
            to += 1;
          }
        }
      }
    }
  }
  return out;
};

/** I2V の条件画像の前処理の結果（VAE encoder の入力と、選んだ出力寸法）。 */
export type WanI2vImageInput = {
  /** 出力寸法（受理集合の中 — 生成の width / height になる）。 */
  readonly width: number;
  readonly height: number;
  /** encoder の入力（patchify 済み・`[-1, 1]`）。形は {@link WanI2vImageInput.shape}。 */
  readonly pixels: Float32Array<ArrayBuffer>;
  /** `[3·p², 1, H/p, W/p]`（p は記述子の `vaePatchSize`）。 */
  readonly shape: readonly [number, number, number, number];
};

/**
 * 条件画像 → VAE encoder の入力（寸法の選択 {@link selectWanI2vSize} → 寸法合わせ {@link fitWanI2vImage} →
 * `[-1, 1]` {@link wanI2vPixels} → patchify {@link patchifyWanImage}）。`fit` を省けば `"crop"`（公式）。
 */
export const preprocessWanI2vImage = (
  image: Rgb8Image,
  request: Partial<Size> & { readonly fit?: WanI2vFit },
  generation: Pick<WanGenerationDescriptor, "acceptedSizes" | "vaePatchSize">,
): WanI2vImageInput => {
  const size = selectWanI2vSize(image, request, generation);
  const fitted = fitWanI2vImage(image, size, request.fit ?? DEFAULT_FIT);
  const patchSize = generation.vaePatchSize;
  const pixels = patchifyWanImage(
    wanI2vPixels(fitted),
    [CHANNELS, size.height, size.width],
    patchSize,
  );
  return {
    width: size.width,
    height: size.height,
    pixels,
    shape: [
      CHANNELS * patchSize * patchSize,
      1,
      size.height / patchSize,
      size.width / patchSize,
    ],
  };
};
