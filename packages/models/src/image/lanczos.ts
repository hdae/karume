/**
 * LANCZOS resampling of RGB8 — a verbatim port of the 8-bit path of Pillow's
 * `Image.resize(size, Image.LANCZOS)` (`ImagingResampleInner` and `*_8bpc` in
 * `libImaging/Resample.c`). The body is the shared core of Pillow's 8-bit separable resampling
 * (`pillow-resample.ts` — filter `PILLOW_LANCZOS`, 3 channels); this file holds only the RGB8 entry
 * check and the wiring (the verbatim copy of Pillow is consolidated into one — ADR 0125).
 *
 * The condition image of Wan2.2's I2V is made with Pillow's LANCZOS both by the official
 * implementation (`textimage2video.py`) and by diffusers (`resample="lanczos"` of
 * `VaeImageProcessor`). This exists to reproduce that uint8 bit for bit (ADR 0121 decision 11). The
 * canonical source for what is copied verbatim, why the path is separate from `resizeRgb8` in
 * `preprocess.ts`, and why only the window is computed is the module doc of `pillow-resample.ts`.
 * The version is recorded by the fixture `tests/fixtures/wan-i2v/lanczos.json` (pixels in
 * `lanczos.safetensors`).
 *
 * NOTE: `export` serves Wan's I2V preprocessing and the tests (not exposed on `mod.ts` / the subpath
 * surface — ADR 0008).
 */

import { assertRgb8, type Rgb8Image } from "./preprocess.ts";
import { PILLOW_LANCZOS, type PixelWindow, resampleU8Window } from "./pillow-resample.ts";

/** Number of channels (RGB). */
const CHANNELS = 3;

/** A rectangle cropped from the resized image (`width × height` from the top-left `left`, `top`). */
export type Rgb8Window = PixelWindow;

/**
 * Resamples RGB8 to `width × height` with LANCZOS (uint8-identical to Pillow's
 * `Image.resize(..., Image.LANCZOS)` — module doc) and returns the rectangle `window` cropped from
 * it. Only the window's output is computed (the same values as resizing the whole image and
 * cropping). The aspect ratio is not preserved. The pass is skipped for an axis whose size does not
 * change (that axis crops a copy of the input), and when neither axis changes a copy is returned
 * (Pillow's `ImagingCopy`).
 *
 * The image is checked by `assertRgb8` (the image the caller passed — `ModelInputError`). The output
 * size and the window are internal preconditions (callers pass only the acceptance set and sizes
 * derived from it), hence `RangeError` (thrown by the shared core).
 */
export const resizeRgb8LanczosWindow = (
  image: Rgb8Image,
  width: number,
  height: number,
  window: Rgb8Window,
): Rgb8Image => {
  assertRgb8(image);
  const resized = resampleU8Window(
    { data: image.data, width: image.width, height: image.height, channels: CHANNELS },
    width,
    height,
    window,
    PILLOW_LANCZOS,
  );
  return { data: resized.data, width: resized.width, height: resized.height };
};

/**
 * Resamples RGB8 to `width × height` with LANCZOS (the window of {@link resizeRgb8LanczosWindow} =
 * the whole image).
 */
export const resizeRgb8Lanczos = (image: Rgb8Image, width: number, height: number): Rgb8Image =>
  resizeRgb8LanczosWindow(image, width, height, { left: 0, top: 0, width, height });
