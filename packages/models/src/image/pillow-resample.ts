/**
 * Pillow's 8-bit separable resampling — a verbatim port of `Image.resize`'s 8-bit path
 * (`ImagingResampleInner` and `*_8bpc` in `libImaging/Resample.c`). It is **the shared core** that
 * takes the filter (weight function and support radius) and the channel count as arguments; the path
 * that reproduces Pillow's values bit for bit lives here, once (so that following a Pillow version
 * bump happens in one place — ADR 0125).
 *
 * Paths built on it:
 *
 * - `lanczos.ts` (3 channels, LANCZOS, window): the condition image of Wan2.2's I2V (the official
 *   implementation and diffusers' `Image.resize(..., LANCZOS)` — ADR 0121 decision 11).
 * - {@link resizeU8PlaneBicubic} (1 band, BICUBIC): `u8-pil-bicubic` of BiRefNet's alpha matte
 *   (`src/birefnet/matte.ts` — HR's upstream `pred_pil.resize(image.size)`).
 *
 * ## Why this path is separate from `resizeRgb8` in `preprocess.ts`
 *
 * MUST: adding a kernel to the path of `resizeRgb8` (f64 weights, `/ filterScale`,
 * `floor(v + 0.5)`) does not match Pillow. Pillow's 8-bit path rounds the coefficients to 22-bit
 * fixed-point integers and accumulates in integers. Staying in f64, samples that sit on a rounding
 * boundary are off by 1–2 LSB (0.002–0.275% with LANCZOS, in the Python copy of the Wan2.2 stage 9
 * investigation). Meanwhile `resizeRgb8` is the preprocessing of SigLIP2 / Depth Anything / BiRefNet
 * matched to torchvision's antialias path, and pulling it towards Pillow would move their output.
 * So the paths are kept separate (the public `resizeRgb8` / `ResampleFilter` are not widened —
 * ADR 0008).
 *
 * ## What is copied verbatim (Pillow 12.3.0 — pinned by fixtures that record the version)
 *
 * - The coefficient argument is `(x + xmin − center + 0.5) · (1 / filterscale)` (multiplying by the
 *   reciprocal — kept in Pillow's form)
 * - `xmin` / `xmax` are clamped to `[0, inSize]` after the `(int)` truncation
 * - Normalisation happens only when the sum `ww` is not 0
 * - The coefficients are `(int)(±0.5 + w·2^22)` (the rounding direction follows the sign)
 * - Accumulation is in integers, starts at 2^21, is shifted `>> 22` and saturated to 0–255
 *   (`clip8`)
 * - Horizontal → vertical order, uint8 intermediate, and the pass is skipped for an axis whose size
 *   does not change (`need_horizontal` / `need_vertical`). When neither axis changes, a copy
 *   (`ImagingCopy` — `Image.resize` also returns `copy()` for the same size)
 * - The horizontal pass runs only over the input rows the vertical pass reads (`ybox_first` to
 *   `ybox_last`)
 * - The per-channel accumulations are independent of each other (Pillow's single-band branch
 *   `image8` and its multi-band branch use the same formula)
 *
 * ## Computing only the window ({@link resampleU8Window})
 *
 * The coefficients are independent per output index (`center = (i + 0.5)·scale`), and the uint8
 * intermediate is independent per output column. So for a rectangle of the resized image,
 * computing only that rectangle's output indexes gives the same uint8 values as resizing the whole
 * image and then cropping. Wan2.2's I2V crop resizes to the covering size and then crops the
 * centre, so even for inputs with extreme aspect ratios (e.g. 4096×4 → covering size 720,896×704)
 * the work and the intermediate size stay bounded by the size of the output window.
 *
 * NOTE: `export` serves internal callers in `src/` and the tests (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */

// Third-party code notice. The filters, the fixed-point coefficients and the separable passes below
// (`PILLOW_BICUBIC`, `PILLOW_LANCZOS`, `fixedTaps`, `clip8`, `resampleHorizontal`,
// `resampleVertical` and `resampleU8Window`) are ported from the 8-bit path of
// `ImagingResampleInner` (its `*_8bpc` passes and the BICUBIC / LANCZOS filters) in
// python-pillow/Pillow (`src/libImaging/Resample.c`, Pillow 12.3.0). The window-only computation
// and the channel-count parameter are additions (see the module doc). License: MIT-CMU License — the
// full copyright and permission notice is in `THIRD_PARTY_NOTICES.md` at the package root. Upstream
// copyright notice, copied verbatim from Pillow's `LICENSE`:
//
//   Copyright © 1997-2011 by Secret Labs AB
//   Copyright © 1995-2011 by Fredrik Lundh and contributors
//   Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

/**
 * A Pillow filter (`struct filter` in `Resample.c` — the weight function and the support radius at
 * scale 1). `name` is used only in diagnostic messages.
 */
export type PillowFilter = {
  readonly name: string;
  readonly support: number;
  readonly weight: (x: number) => number;
};

/**
 * Pillow's `bicubic_filter` (`a = −0.5`; the shape of the formula and the order of operations are as
 * in `Resample.c`). It has negative weights for `|x| ∈ [1, 2)`, so steps produce sums beyond the
 * value range (the path where the saturation of {@link clip8} matters).
 *
 * NOTE: the same formula as `cubic` in `preprocess.ts`, but not shared — that one copies
 * torchvision's antialias path, this one is a verbatim copy of Pillow (the canonical sources differ,
 * so following one version must not move the other).
 */
const bicubic = (x: number): number => {
  const a = -0.5;
  const abs = x < 0 ? -x : x;
  if (abs < 1) return ((a + 2) * abs - (a + 3)) * abs * abs + 1;
  if (abs < 2) return (((abs - 5) * abs + 8) * abs - 4) * a;
  return 0;
};

/** Pillow's BICUBIC (`BICUBIC = {bicubic_filter, 2.0}` in `Resample.c`). */
export const PILLOW_BICUBIC: PillowFilter = { name: "BICUBIC", support: 2, weight: bicubic };

const sinc = (x: number): number => {
  if (x === 0) return 1;
  const radians = x * Math.PI;
  return Math.sin(radians) / radians;
};

/** Pillow's `lanczos_filter` (0 outside the support — the right end `x = 3` is outside). */
const lanczos = (x: number): number => (-3 <= x && x < 3 ? sinc(x) * sinc(x / 3) : 0);

/**
 * Pillow's LANCZOS (`LANCZOS = {lanczos_filter, 3.0}` in `Resample.c`).
 *
 * NOTE: the coefficients' `sin` is V8's `Math.sin`, while Pillow uses the C libm `sin`. If the last
 * ulp differs, an integer coefficient can change by 1 (speculation). The fixtures look only at the
 * output uint8, so a ±1 in a coefficient is hardly ever detectable (a shifted coefficient changes the
 * output only when the sum crosses a 2^22 boundary — roughly pixel value / 2^22 per sample).
 * The review of 2026-10-09 compared the coefficient table itself (`xmin` and every integer
 * coefficient) against a verbatim Python copy of Resample.c (glibc's `sin`) over 1,600 geometries,
 * with 0 mismatches (a one-off verification, not a standing check). BICUBIC is a polynomial and does
 * not use `sin`, so it does not have this problem.
 */
export const PILLOW_LANCZOS: PillowFilter = { name: "LANCZOS", support: 3, weight: lanczos };

/** Number of fractional bits of the fixed-point coefficients (Pillow's `PRECISION_BITS` = 32 − 8 − 2). */
const PRECISION_BITS = 32 - 8 - 2;

/** 1.0 as a coefficient (`1 << PRECISION_BITS`). */
const FIXED_ONE = 1 << PRECISION_BITS;

/** Initial value of the sum (`1 << (PRECISION_BITS - 1)` — the half that makes `>> PRECISION_BITS` round to nearest). */
const FIXED_HALF = 1 << (PRECISION_BITS - 1);

/**
 * 8-bit pixels (row-major, `channels` bytes per pixel) and their size. Results are arrays on a newly
 * allocated `ArrayBuffer` (`U8Image<ArrayBuffer>`); inputs may also be on shared memory.
 */
export type U8Image<Buffer extends ArrayBufferLike = ArrayBufferLike> = {
  readonly data: Uint8Array<Buffer>;
  readonly width: number;
  readonly height: number;
  readonly channels: number;
};

/** A rectangle cropped from the resized image (`width × height` from the top-left `left`, `top`). */
export type PixelWindow = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

/** Input range and fixed-point coefficients for one output point. */
type FixedTap = {
  readonly start: number;
  readonly coefficients: Int32Array<ArrayBuffer>;
};

/** The output window of one axis (`length` entries from the resized index `offset`). */
type AxisWindow = {
  readonly offset: number;
  readonly length: number;
};

/**
 * The coefficients of one axis (Pillow's `precompute_coeffs` → `normalize_coeffs_8bpc`), for the
 * indexes of the output window `window` only (independent per index — module doc "Computing only
 * the window").
 *
 * NOTE: Pillow's `scale` is `(double)(in1 − in0) / outSize` (`in0` / `in1` are the float box), which
 * for a resize without a box equals `inSize / outSize` (integers below 2^24 are exact in float).
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
      coefficients[x] =
        weight < 0 ? Math.trunc(-0.5 + weight * FIXED_ONE) : Math.trunc(0.5 + weight * FIXED_ONE);
    }
    taps.push({ start, coefficients });
  }
  return taps;
};

/**
 * Brings a fixed-point sum back to 8 bits (Pillow's `clip8` — saturates to 0–255 after
 * `>> PRECISION_BITS`).
 *
 * As in Pillow, the sum fits in int32 (|sum of coefficients| ≲ 1.3·2^22 × 255 + 2^21 < 2^31), so the
 * ToInt32 of `>>` does not change the value.
 */
const clip8 = (sum: number): number => {
  const value = sum >> PRECISION_BITS;
  return value < 0 ? 0 : value > 255 ? 255 : value;
};

/**
 * Horizontal pass (Pillow's `ImagingResampleHorizontal_8bpc`). Runs only over the input rows `rows`
 * (Pillow's `ybox_first` to `ybox_last`). The output columns are the number of `taps` (the window's
 * columns).
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
 * Vertical pass (Pillow's `ImagingResampleVertical_8bpc`). Row 0 of `image` is input row `rowOffset`
 * (the same re-indexing Pillow does when it subtracts `ybox_first` from `bounds_vert` after the
 * horizontal pass). The output rows are the number of `taps` (the window's rows).
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

/** Copy of the rectangle of columns `columns` and rows `rows` (the crop on an axis whose pass is skipped — called only within range). */
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
 * The input rows the vertical pass reads (Pillow's `ybox_first` to `ybox_last`). Both the start and
 * the end of a tap are monotonic in the output index, so the taps at the two ends decide it.
 */
const rowsReadBy = (taps: readonly FixedTap[]): AxisWindow => {
  const first = taps[0];
  const last = taps[taps.length - 1];
  return { offset: first.start, length: last.start + last.coefficients.length - first.start };
};

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/**
 * Resamples an 8-bit image to `width × height` with `filter` (uint8-identical to Pillow's
 * `Image.resize` — module doc) and returns the rectangle `window` cropped from it. Only the window's
 * output is computed (the same values as resizing the whole image and cropping). The aspect ratio is
 * not preserved. The pass is skipped for an axis whose size does not change (that axis crops a copy
 * of the input), and when neither axis changes a copy is returned.
 *
 * The checks of the input, the output size and the window are internal preconditions (callers pass
 * only checked images and sizes derived from the acceptance set or from assets), hence
 * `RangeError`. The check of the image the caller received itself (`ModelInputError`) belongs to the
 * caller, earlier.
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
    !isPositiveInteger(image.width) ||
    !isPositiveInteger(image.height) ||
    !isPositiveInteger(channels) ||
    image.data.length !== image.width * image.height * channels
  ) {
    throw new RangeError(
      `${filter.name}: input ${image.width}×${image.height}×${channels} and length ${image.data.length}` +
        " do not match",
    );
  }
  if (!isPositiveInteger(width) || !isPositiveInteger(height)) {
    throw new RangeError(
      `${filter.name}: output size ${width}×${height} is not a positive integer`,
    );
  }
  const { left, top } = window;
  if (
    !Number.isInteger(left) ||
    !Number.isInteger(top) ||
    left < 0 ||
    top < 0 ||
    !isPositiveInteger(window.width) ||
    !isPositiveInteger(window.height) ||
    left + window.width > width ||
    top + window.height > height
  ) {
    throw new RangeError(
      `${filter.name}: window (${left}, ${top}) ${window.width}×${window.height} and output ${width}×${height}` +
        ": the window is not inside the output",
    );
  }
  const columns = { offset: left, length: window.width };
  const verticalTaps =
    height === image.height
      ? undefined
      : fixedTaps(image.height, height, filter, { offset: top, length: window.height });
  // The input rows the horizontal pass runs over (the window's own rows when the vertical pass is skipped).
  const rows =
    verticalTaps === undefined ? { offset: top, length: window.height } : rowsReadBy(verticalTaps);
  const band =
    width === image.width
      ? cropU8(image, columns, rows)
      : resampleHorizontal(image, fixedTaps(image.width, width, filter, columns), rows);
  return verticalTaps === undefined ? band : resampleVertical(band, verticalTaps, rows.offset);
};

/**
 * Resamples an 8-bit single-band plane (row-major) to `width × height` with Pillow's BICUBIC
 * (uint8-identical to Pillow's L-mode `Image.resize((width, height))` — the resample default).
 */
export const resizeU8PlaneBicubic = (
  plane: Uint8Array<ArrayBufferLike>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): Uint8Array<ArrayBuffer> =>
  // Including the copy path (both axes the same size), the result is a newly allocated array (it shares no memory with the input).
  resampleU8Window(
    { data: plane, width: sourceWidth, height: sourceHeight, channels: 1 },
    width,
    height,
    { left: 0, top: 0, width, height },
    PILLOW_BICUBIC,
  ).data;
