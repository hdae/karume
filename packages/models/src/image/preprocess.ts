/**
 * RGB8 pixels → model input (resize → rescale → normalize → NCHW f32).
 *
 * This is **pipeline-independent shared processing** (models that take an image as input generally
 * go through these 3 steps), so it lives in `src/image/` rather than in a family directory (the same
 * standing as `png.ts`).
 *
 * **Decoding is not karume's responsibility**. The entry is "RGB8 pixels + width + height", and no
 * PNG / JPEG decoder is included (in a browser `createImageBitmap` + canvas, in Node / Deno any
 * decoder). On the other hand, resize / rescale / normalize are **part of the model card** (a wrong
 * constant silently costs accuracy), so this side owns them, not the user.
 *
 * Models with per-pixel output (segmentation mattes) also go through a step that **brings back** the
 * map produced at the exported resolution to the original image's resolution. The support and the
 * weights are the same as at the entry, so that one function ({@link resizePlaneF32}) lives here too.
 *
 * ## MUST: the reference implementation being matched
 *
 * transformers 5.14.1's **`SiglipImageProcessor` (`TorchvisionBackend`)** — the fast side that v5's
 * `AutoImageProcessor` returns by default, not the PIL side (`SiglipImageProcessorPil`). Resize is
 * `torchvision.transforms.v2.functional.resize(..., antialias=True)`, and this antialias path is the
 * same "separable resampling that stretches the support by the scale" as Pillow's
 * `ImagingResample` ({@link buildTaps}).
 *
 * MUST: the interpolation filter is **part of the model card**, so the caller states it explicitly
 * (the default is bilinear). `"resample"` in `preprocessor_config.json` is **a PIL constant**
 * (NEAREST=0 / LANCZOS=1 / **BILINEAR=2** / **BICUBIC=3** / BOX=4 / HAMMING=5): SigLIP2 has
 * 2 = bilinear, Depth Anything V2 (`DPTImageProcessor`) 3 = bicubic. The class-attribute default in
 * transformers is `PILImageResampling.BICUBIC` for both classes, which makes "SigLIP is bicubic" an
 * easy misreading, but for SigLIP2 the checkpoint's config overrides the default. Measured effect of
 * the mix-up: up to 47/255 on the SigLIP2 side, and up to 0.59 in `pixel_values` on the DA-V2 side
 * (34 LSB, with 1 LSB = 0.0175). `resample` is measured on every emit by
 * `tools/export-recipes/siglip2/preprocess.py` and `tools/export-recipes/depth_anything/export.py`.
 *
 * MUST: round back to uint8 **between** the 2 passes ({@link resizeRgb8}). The uint8 paths of PIL /
 * torchvision keep the intermediate buffer in uint8, so passing f64 on to the vertical pass makes the
 * samples that differ from the reference jump from **0.26% → 9.5%** (measured over 18 geometries ×
 * 224×224 = 2,709,504 samples).
 */

import { ModelInputError } from "../errors.ts";

/** RGB8 pixels (row-major, 3 bytes per pixel) and their size. */
export type Rgb8Image = {
  readonly data: Uint8Array<ArrayBufferLike>;
  readonly width: number;
  readonly height: number;
};

/** Number of channels (RGB). Alpha is not accepted at the entry. */
const CHANNELS = 3;

/** Interpolation filter of the resampling (the spelling that corresponds to `resample` in `preprocessor_config.json`). */
export type ResampleFilter = "bilinear" | "bicubic";

/** One filter = weight function and the radius of its support (when downscaling, the scale multiplies it and the support grows). */
type Kernel = {
  /** Radius of the support (the value at scale 1). */
  readonly support: number;
  readonly weight: (x: number) => number;
};

/** Triangle (bilinear) filter. */
const triangle = (x: number): number => {
  const abs = Math.abs(x);
  return abs < 1 ? 1 - abs : 0;
};

/**
 * Catmull-Rom-style cubic filter (`a = -0.5`, the same as PIL's `bicubic_filter` / torchvision's
 * antialias path). It has **negative weights** for `|x| ∈ [1, 2)`, so step edges overshoot beyond the
 * input's value range (the only path where the clamp of {@link round8} matters).
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

/** Input range and weights for one output point. */
type Tap = {
  readonly start: number;
  readonly weights: Float64Array<ArrayBuffer>;
};

/**
 * Builds the resampling weights of one axis (the same formula as Pillow / torchvision's antialias
 * path).
 *
 * When downscaling (`scale > 1`) the support is stretched by the scale to average; when upscaling
 * the support is fixed at 1. At the edges, the part of the support that sticks out of the input is
 * **truncated** before normalising to a sum of 1 (`ImagingResample`'s convention) — because of this
 * truncation an upscaled edge can end up as its single nearest sample.
 *
 * That the weight sum never becomes 0 follows from the formula: the input position containing
 * `center` always falls inside `[start, stop)`, and its weight is positive (at least 0.5 for
 * bilinear / 0.75 for bicubic). Even bicubic, with its negative lobes, has no combination whose sum
 * collapses to 0, so no division-by-zero branch is needed.
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
 * Rounds and fits into 8 bits.
 *
 * MUST: keep the clamp. Assigning to a `Uint8Array` **wraps out-of-range values modulo 256**, so
 * the overshoot of bicubic's negative lobes would pass silently as a black/white inversion of pixels
 * that should saturate (bilinear has non-negative weights summing to 1, so it never takes this
 * path).
 */
const round8 = (value: number): number => Math.min(255, Math.max(0, Math.floor(value + 0.5)));

/**
 * Checks the entry contract (positive integer size, consistent length).
 *
 * NOTE: it is a {@link ModelInputError} because what it checks is **the image the caller passed
 * itself** (the 3 families' `preprocessPixelValues` pass on the `Rgb8Image` that `estimate` /
 * `segment` / `embedImage` received as is). The only remedy is "fix the image you pass" = what HTTP
 * would call a 400 (ADR 0107 decision 2).
 *
 * NOTE: it is `export`ed because Wan's I2V preprocessing (`lanczos.ts` and `wan/i2v-preprocess.ts`)
 * uses the same check (not exposed on `mod.ts` — ADR 0008).
 */
export const assertRgb8 = (image: Rgb8Image): void => {
  const { data, width, height } = image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new ModelInputError(`Image size ${width}×${height} is not a positive integer`);
  }
  if (data.length !== width * height * CHANNELS) {
    throw new ModelInputError(`RGB8 length ${data.length} does not match 3×${width}×${height}`);
  }
};

/**
 * Resamples RGB8 to the given size (antialiased — the reference implementation of the module
 * docstring).
 *
 * The aspect ratio is **not preserved** (it stretches straight to `size.height` / `size.width`).
 * SigLIP2's `preprocessor_config.json` holds `size` as a height/width pair with neither crop nor
 * pad. DA-V2 sets `keep_aspect_ratio`, but a square input → a square `size` makes it the identity
 * (until a model that needs an aspect-preserving path appears, this function only takes a size).
 *
 * `filter` is **part of the model card**, so all 3 family call sites pass `config.interpolation`
 * explicitly (no caller relies on the default any more — the remaining bilinear default is a
 * shorthand for tests that look only at geometry).
 */
export const resizeRgb8 = (
  image: Rgb8Image,
  width: number,
  height: number,
  filter: ResampleFilter = "bilinear",
): Rgb8Image => {
  assertRgb8(image);
  // NOTE: the output size is input-caused too (ModelInputError). The `config.imageWidth` /
  // `imageHeight` the 3 families pass have gone through `isPositiveInteger` at parse time
  // (`readNumber` in `siglip2/config.ts` and the 2 other families), so **the pipeline path never
  // reaches here** — the only remaining path is a caller hitting `resizeRgb8` directly from the
  // barrel, where the size is the caller's request itself.
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new ModelInputError(`Output size ${width}×${height} is not a positive integer`);
  }
  // MUST: look up with `Object.hasOwn` — when a JS caller passes `"toString"` etc., a member of
  // Object.prototype would be taken as the kernel, support becomes NaN → every tap is empty → an
  // all-black image is returned silently. It is a misspelled name, so a plain `Error` (ADR 0107
  // decision 3 — the same treatment as sampler names).
  if (!Object.hasOwn(KERNELS, filter)) {
    throw new Error(
      `resizeRgb8: interpolation filter ${JSON.stringify(filter)} is not supported (expected ${Object.keys(
        KERNELS,
      )
        .map((name) => `'${name}'`)
        .join(" / ")})`,
    );
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
 * Resamples **a single-channel f32 plane** to the given size (shares {@link buildTaps} with
 * {@link resizeRgb8}; the filter is fixed to bilinear).
 *
 * Models with per-pixel output (segmentation mattes etc.) need to **bring the map back to the
 * original image's resolution** from the resolution the graph was exported at. The reference
 * implementation is `torchvision.transforms.functional.resize` (tensor branch, bilinear, antialias
 * by default), whose weights are built the same way as Pillow's `ImagingResample`, which
 * {@link resizeRgb8} matches.
 *
 * MUST: unlike {@link resizeRgb8}, **do not round between the 2 passes**. The `round8` there
 * reproduces the fact that the **uint8 paths** of PIL / torchvision keep the intermediate buffer in
 * uint8; the f32 tensor branch has no such step. Imitating it here would move away from the
 * reference and, on top of that, crush a `[0, 1]` matte into 8-bit steps and lose the gradient at
 * the boundary.
 *
 * The weights are non-negative and sum to 1, so the input's value range is preserved in the output
 * (a `[0, 1]` matte never leaves its range — that is why there is no clamp).
 *
 * NOTE: the 2 checks below stay `RangeError` / plain `Error` (unlike {@link resizeRgb8}, not
 * {@link ModelInputError}). This function is exposed neither on the barrel nor on a subpath
 * surface; its only 2 callers are `birefnet`'s `matteFromLogits` and `depth-anything`'s
 * `resampleDepth`, and both pass **only asset values (`config.imageWidth` / `imageHeight`) and
 * already-checked image sizes**. So these are **preconditions of an internal helper**, on the side
 * that ADR 0107 decision 3 excludes from its scope.
 */
export const resizePlaneF32 = (
  plane: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Float32Array<ArrayBuffer> => {
  for (const [size, label] of [
    [sourceWidth, "Input width"],
    [sourceHeight, "Input height"],
    [width, "Output width"],
    [height, "Output height"],
  ] as const) {
    if (!Number.isInteger(size) || size <= 0) {
      throw new RangeError(`${label} ${size} is not a positive integer`);
    }
  }
  if (plane.length !== sourceWidth * sourceHeight) {
    throw new Error(`Plane length ${plane.length} does not match ${sourceWidth}×${sourceHeight}`);
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
  const out = new Float32Array(width * height);
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
 * Divisor of rescale (the reciprocal of `rescale_factor` = 1/255).
 * In f64, `1 / 0.00392156862745098` is exactly 255.0 (the Python side measures it every time).
 */
const RESCALE_DIVISOR = 255;

/**
 * A message describing the first channel whose normalised values do not stay finite in f32
 * (`undefined` when every channel fits).
 *
 * MUST: finiteness of each mean / std alone is not enough. Normalisation is computed in f64 and
 * stored into a `Float32Array`, so with an extremely small std (e.g. 1e-50) or an extremely large
 * mean, a quotient that is finite in double precision exceeds the f32 maximum and becomes ±Inf, and
 * the whole plane reaches the GPU non-finite.
 * The value range is found by applying **the same formula** as {@link normalizeToNchw} to the pixel
 * extremes (0 and 255) — the formula is monotonic in the pixel (divisor > 0) and rounding to f32 is
 * monotonic too, so if both extremes are finite every pixel is finite, and the check needs no pass
 * over the pixels.
 *
 * Precondition: `std` is a positive finite number and `mean` is finite (the caller checks first).
 */
export const normalizationOverflow = (
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): string | undefined => {
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    const offset = mean[channel] * RESCALE_DIVISOR;
    const divisor = std[channel] * RESCALE_DIVISOR;
    const low = (0 - offset) / divisor;
    const high = (255 - offset) / divisor;
    if (!Number.isFinite(Math.fround(low)) || !Number.isFinite(Math.fround(high))) {
      return (
        `channel ${channel}: the normalized range [${low}, ${high}] is not finite in f32` +
        ` (mean ${mean[channel]} / std ${std[channel]})`
      );
    }
  }
  return undefined;
};

/**
 * RGB8 (interleaved) → normalised planar f32 `[3, height, width]` (batch 1 is folded away).
 *
 * MUST: compute rescale and normalize in **folded form** — the reference implementation
 * (`TorchvisionBackend._fuse_mean_std_and_rescale_factor`) builds `mean · 255` and `std · 255`
 * first and applies `(x - mean) / std` once. Applying `x / 255` first adds one more rounding and
 * differs in a plain f32 comparison (the Python side measured this folded form to be bit-identical
 * to the full pipeline).
 *
 * `mean` / `std` are taken on the `[0, 1]` scale (as spelled in `preprocessor_config.json`).
 */
export const normalizeToNchw = (
  image: Rgb8Image,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array<ArrayBuffer> => {
  assertRgb8(image);
  // mean / std are exactly the values the caller passes (families pass only parsed config values —
  // only a direct barrel call reaches here). A std of 0 / non-finite would silently return an
  // Inf / NaN plane through division by zero.
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    if (!Number.isFinite(mean[channel])) {
      throw new ModelInputError(
        `normalizeToNchw: mean[${channel}] = ${mean[channel]} is not finite`,
      );
    }
    if (!Number.isFinite(std[channel]) || std[channel] <= 0) {
      throw new ModelInputError(
        `normalizeToNchw: std[${channel}] = ${std[channel]} is not a positive finite number`,
      );
    }
  }
  // Even finite mean / std can produce combinations whose normalised result overflows f32 (±Inf for
  // a tiny std, −Inf for a huge mean, Inf / Inf = NaN when both mean and std are huge —
  // {@link normalizationOverflow}). Checked before allocating the output — a huge image is not
  // allocated for coefficients that are going to be rejected. On the family path the config parse
  // rejects with the same test first, so only a direct barrel call reaches here (the same
  // classification as above).
  const overflow = normalizationOverflow(mean, std);
  if (overflow !== undefined) throw new ModelInputError(`normalizeToNchw: ${overflow}`);
  const plane = image.width * image.height;
  const out = new Float32Array(plane * CHANNELS);
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    const offset = mean[channel] * RESCALE_DIVISOR;
    const divisor = std[channel] * RESCALE_DIVISOR;
    for (let index = 0; index < plane; index += 1) {
      out[channel * plane + index] = (image.data[index * CHANNELS + channel] - offset) / divisor;
    }
  }
  return out;
};
