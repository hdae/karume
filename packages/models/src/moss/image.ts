/**
 * A generated MOSS-Image image (`[3, H, W]`, range `[-1, 1]`) → interleaved 8-bit RGBA.
 *
 * The rule is Anima's `imageToRgba` and Wan's `frameToRgba`: `v = clamp(x/2 + 0.5, 0, 1)` and `v·255`
 * in **f64**, then `Math.round` (0.5 rounds up). Families do not import each other, so this keeps
 * its own copy (a change to one does not silently move the others' outputs).
 *
 * NOTE: this rule does not match upstream at 8 bits. Upstream's `output_type="pil"` goes through
 * diffusers' `numpy_to_pil`, f32 `(images * 255).round()` (round half to even), so the two can
 * split by ±1 near an 8-bit boundary. The sha rows of this family are taken over the f32 image, not
 * over these bytes.
 *
 * MUST: do not move the clamp **before** the `[0, 1]` conversion (negative values would collapse to
 * 0 and then get +0.5, pinning the shadows to mid-grey).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { GeneratedImage } from "./pipeline.ts";

export const imageToRgba = (
  image: Pick<GeneratedImage, "width" | "height" | "data">,
): Uint8ClampedArray<ArrayBuffer> => {
  const { width, height, data } = image;
  // A length check alone cannot pin the dimensions (width 0.5 × height 2 also has the right
  // length) — the entry contract of Anima's and Wan's conversions.
  if (![width, height].every((side) => Number.isInteger(side) && side > 0)) {
    throw new RangeError(`Image size ${width}×${height} is not a positive integer`);
  }
  const plane = width * height;
  if (data.length !== 3 * plane) {
    throw new Error(`Image element count ${data.length} does not match [3, ${height}, ${width}]`);
  }
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let channel = 0; channel < 3; channel += 1) {
    for (let index = 0; index < plane; index += 1) {
      const raw = data[channel * plane + index];
      // MUST: never turn a non-finite value into a pixel silently (`Uint8ClampedArray` makes NaN 0).
      if (!Number.isFinite(raw)) {
        throw new Error(
          `Pixel (x=${index % width}, y=${Math.floor(index / width)}): channel ${channel} is non-finite`,
        );
      }
      rgba[index * 4 + channel] = Math.round(Math.min(1, Math.max(0, raw / 2 + 0.5)) * 255);
    }
  }
  for (let index = 0; index < plane; index += 1) rgba[index * 4 + 3] = 255;
  return rgba;
};
