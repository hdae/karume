/**
 * Preprocessing of the I2V condition image for Wan2.2 TI2V (on the host, no GPU — ADR 0121
 * decision 11, stage 9a): output size selection → size fitting (crop / stretch, Pillow's LANCZOS) →
 * `[-1, 1]` → patchify (the VAE encoder input `[3·p², 1, H/p, W/p]`). Normalizing the mu at the
 * encoder exit is `normalizeWanLatents` in `latents.ts`.
 *
 * Size fitting has two sources of truth:
 *
 * - fit = `"crop"` (default): i2v in official Wan2.2's `textimage2video.py` —
 *   `scale = max(ow / iw, oh / ih)` → `round(iw·scale)`, `round(ih·scale)` (Python's `round`) →
 *   LANCZOS → centre crop `(rw − ow) // 2` → `TF.to_tensor(img).sub_(0.5).div_(0.5)`
 * - fit = `"stretch"`: diffusers' `WanImageToVideoPipeline` (`VideoProcessor.preprocess` with
 *   resize_mode `"default"` = LANCZOS directly to `(width, height)`, no `reducing_gap` →
 *   `2·(x / 255) − 1`)
 *
 * Only the choice of output size differs from the official code: the official code picks a free
 * size under an area limit (1248×704 for a 16:9 image), while this picks from the accepted set (the
 * descriptor's `acceptedSizes`) with the official comparison (ADR 0121 addendum "accepted sizes
 * moved to the two official sizes").
 *
 * NOTE: the `export`s are for the pipeline internals and the tests (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008). The only exception is the type {@link WanI2vFit}, which is vocabulary
 * of the public request type (`fit` of `WanTi2vGenerateRequest`) and is therefore exported from
 * `./wan` and the barrel.
 */

import { ModelInputError } from "../errors.ts";
import { resizeRgb8Lanczos, resizeRgb8LanczosWindow } from "../image/lanczos.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";

/** How the condition image is fitted to the size (the official centre crop / diffusers' direct resize — module doc). */
export type WanI2vFit = "crop" | "stretch";

/**
 * Whether the value is one of the spellings of {@link WanI2vFit} (the request check in `plan.ts`
 * rejects misspellings that slipped past the type).
 */
export const isWanI2vFit = (value: unknown): value is WanI2vFit =>
  value === "crop" || value === "stretch";

/** The fit used when omitted (the same as official Wan2.2's I2V). */
const DEFAULT_FIT: WanI2vFit = "crop";

/** The channel count (RGB). */
const CHANNELS = 3;

type Size = { readonly width: number; readonly height: number };

/**
 * Computes the official comparison `max(r / rc, rc / r)` (r = image width / height, rc = candidate
 * width / height) in the form `max(a, b) / min(a, b)` (a = iw·ch, b = ih·cw — integer products). It
 * is mathematically the same value, and since the only rounding is one division, a square image
 * ties exactly between two transposed candidates regardless of floating-point rounding (the official
 * form chains three divisions).
 */
const aspectDistance = (image: Size, candidate: Size): number => {
  const a = image.width * candidate.height;
  const b = image.height * candidate.width;
  return Math.max(a, b) / Math.min(a, b);
};

const isLandscape = ({ width, height }: Size): boolean => width >= height;

/** The spelling of the size fields the request gave (for the rejection message). */
const describeRequested = (requested: Partial<Size>): string => {
  if (requested.width !== undefined && requested.height !== undefined) {
    return `${requested.width}×${requested.height}`;
  }
  return requested.width !== undefined ? `width ${requested.width}` : `height ${requested.height}`;
};

/**
 * Selects the I2V output size: among the accepted sizes (the descriptor's `acceptedSizes`) that match
 * the fields the request gave (width / height), the one closest to the image's aspect ratio.
 * Closeness is measured with the comparison of official Wan2.2's `best_output_size`
 * ({@link aspectDistance}), and a tie goes to landscape (width ≥ height). With an accepted set of two
 * sizes (1280×704 / 704×1280) the result equals "image width ≥ height → 1280×704".
 *
 * MUST: do not measure by the absolute difference `|r − rc|` — with two sizes the tie moves to
 * r ≈ 1.18, and a square image becomes portrait (the opposite of the official code).
 *
 * With both given it is the one exact match; with only one, the aspect ratio picks among the
 * candidates matching that field. With no matching candidate it is a `ModelInputError` (outside the
 * accepted set — an input error).
 */
export const selectWanI2vSize = (
  image: Rgb8Image,
  requested: Partial<Size>,
  generation: Pick<WanGenerationDescriptor, "acceptedSizes">,
): Size => {
  assertRgb8(image);
  const candidates = generation.acceptedSizes.filter(
    (size) =>
      (requested.width === undefined || size.width === requested.width) &&
      (requested.height === undefined || size.height === requested.height),
  );
  if (candidates.length === 0) {
    throw new ModelInputError(
      `${describeRequested(requested)} が受理集合（${generation.acceptedSizes
        .map((size) => `${size.width}×${size.height}`)
        .join(" / ")}）に無い`,
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
 * Python's `round(x)` (float — an exact .5 goes to the even side). `x − floor(x)` is exact for
 * |x| < 2^52, so no rounding enters the .5 test.
 */
const roundHalfEven = (value: number): number => {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
};

/**
 * The scaled size for fit = `"crop"` (official: `scale = max(ow / iw, oh / ih)` → `round(iw·scale)`,
 * `round(ih·scale)`). The scale takes the covering side (both axes at least the target).
 *
 * MUST: the rounding is Python's `round` (an exact .5 goes to the even side). Copying it with JS's
 * `Math.round` (.5 goes toward +∞) turns, for example, 301 × 2.5 = 752.5 for 512×301 → 1280×704 into
 * 753, shifting the centre crop's position by 1 px.
 */
export const wanI2vCoverSize = (image: Size, target: Size): Size => {
  const scale = Math.max(target.width / image.width, target.height / image.height);
  return {
    width: roundHalfEven(image.width * scale),
    height: roundHalfEven(image.height * scale),
  };
};

/**
 * Fits the image to the output size `size` (the two sources of truth in the module doc). `"crop"`
 * LANCZOS-resizes to the covering size ({@link wanI2vCoverSize}) and then cuts out the centre
 * (`(rw − ow) // 2` — a leftover 1 px stays on the right / bottom); `"stretch"` LANCZOS-resizes
 * directly to `size` (the aspect ratio is not kept).
 *
 * crop does not build the whole covering image; it computes only the output of the window it cuts
 * out (`resizeRgb8LanczosWindow` — equal in uint8 to resizing the whole image and then cutting).
 * Building the whole image would let the work and the intermediate grow without bound for extreme
 * aspect ratios (e.g. 4096×4 → 1280×704 has a covering size of 720,896×704, 1.5 GB for the
 * intermediate alone).
 *
 * NOTE: rejecting a fit spelling is a plain `Error` (a misspelt name — the same treatment as the
 * sampler name in ADR 0107 decision 3). The check that accepts it as a request field (`plan.ts`)
 * rejects it first with the same plain `Error` ({@link isWanI2vFit}), so this is the guard for calls
 * that bypass that check.
 */
export const fitWanI2vImage = (image: Rgb8Image, size: Size, fit: WanI2vFit): Rgb8Image => {
  if (fit === "stretch") return resizeRgb8Lanczos(image, size.width, size.height);
  if (fit !== "crop") {
    throw new Error(`I2V の fit ${JSON.stringify(fit)} は未対応（期待 'crop' / 'stretch'）`);
  }
  const cover = wanI2vCoverSize(image, size);
  // On the axis that set the scale, the rounding error of `iw·(ow / iw)` (under 0.5) disappears in
  // the rounding and the size equals the target; the other axis is at least the target. Anything
  // smaller is a mistake in copying the formula (the official code checks the same with
  // `assert img.width == ow and img.height == oh`).
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
 * RGB8 (interleaved) → planar `[3, H, W]` in `[-1, 1]`: `f32(2·f32(x / 255) − 1)`.
 *
 * MUST: write it in this order — the official `TF.to_tensor(img).sub_(0.5).div_(0.5)` and diffusers'
 * `2·(x / 255) − 1` are bit-identical in f32, and this expression gives that value (computing in f64
 * and rounding to f32 once gives the same value as f32 arithmetic, because f64 has at least twice
 * f32's precision + 2 bits). `normalizeToNchw` (the folded form `(x − 127.5) / 127.5` for
 * mean = std = 0.5) is not used because its last bit differs on 128 of the 256 values.
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
 * patchify: planar `[C, H, W]` (one frame) → `[C·p², H/p, W/p]` (the same order as upstream diffusers'
 * `patchify` — `view(c, f, h, r, w, q).permute(c, q, r, f, h, w)`, r = q = p). The index mapping is
 * `out[c·p² + dx·p + dy, y, x] = in[c, y·p + dy, x·p + dx]` (within a channel the width offset dx is
 * the major and the height offset dy the minor) — the inverse of `unpatchifyWanVaeFrames` in
 * `vae-tiles.ts`. The VAE encoder input `[C·p², 1, H/p, W/p]` is the same bytes.
 *
 * Only copies, no floating-point arithmetic (values are preserved bit for bit). A mismatch in shape,
 * factor or length fails loudly.
 */
export const patchifyWanImage = (
  pixels: Float32Array<ArrayBuffer>,
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
  // The loops follow the output order (c → dx → dy → y → x), so the write position advances by 1.
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

/** The result of preprocessing the I2V condition image (the VAE encoder input and the chosen output size). */
export type WanI2vImageInput = {
  /** The output size (inside the accepted set — it becomes the generation's width / height). */
  readonly width: number;
  readonly height: number;
  /** The encoder input (patchified, `[-1, 1]`). Its shape is {@link WanI2vImageInput.shape}. */
  readonly pixels: Float32Array<ArrayBuffer>;
  /** `[3·p², 1, H/p, W/p]` (p is the descriptor's `vaePatchSize`). */
  readonly shape: readonly [number, number, number, number];
};

/**
 * Condition image → VAE encoder input (size selection {@link selectWanI2vSize} → size fitting
 * {@link fitWanI2vImage} → `[-1, 1]` {@link wanI2vPixels} → patchify {@link patchifyWanImage}).
 * Without `fit` it is `"crop"` (official).
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
    shape: [CHANNELS * patchSize * patchSize, 1, size.height / patchSize, size.width / patchSize],
  };
};
