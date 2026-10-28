/**
 * One frame of a generated video (`[3, F, H, W]`, range `[-1, 1]`) → interleaved 8-bit RGBA.
 *
 * The rule is the same as anima's `imageToRgba`: compute `v = clamp(x/2 + 0.5, 0, 1)` and `v·255` in
 * **f64**, then convert to 8 bits with `Math.round` (0.5 rounds up) (ADR 0118 decision 8: "uint8
 * conversion as for Anima's images"). Families do not import each other, so this keeps its own copy
 * (so that a change to one does not silently move the other's sha256 rows).
 *
 * NOTE: this rule does not match upstream at 8 bits. diffusers' `numpy_to_pil` is f32
 * `(images * 255).round()` (numpy's round-half-to-even), so the two split by ±1 near an 8-bit
 * boundary (e.g. x = −0.4941176176 is 64.5 → 64 in f32, and 64.500004 → 65 in f64 here). Upstream
 * Wan's default is `output_type="np"` (no uint8 conversion), and the `export_to_video` its example
 * uses **truncates** with `astype(uint8)`. Assume this difference when comparing 8-bit output with
 * upstream.
 *
 * MUST: do not move the clamp **before** the `[0,1]` conversion (negative values would collapse to 0
 * and then get +0.5, pinning the shadows to mid-grey).
 */

import type { GeneratedVideo } from "./pipeline.ts";

export const frameToRgba = (
  video: Pick<GeneratedVideo, "frames" | "width" | "height" | "data">,
  frame: number,
): Uint8ClampedArray<ArrayBuffer> => {
  const { frames, width, height, data } = video;
  // A length check alone cannot pin the dimensions (width 0.5 × height 2 also has the right
  // length) — the same entry contract as Anima's `imageToRgba`.
  if (![frames, width, height].every((dim) => Number.isInteger(dim) && dim > 0)) {
    throw new RangeError(
      `Video size ${frames} frames × ${width}×${height} is not a positive integer`,
    );
  }
  if (!Number.isInteger(frame) || frame < 0 || frame >= frames) {
    throw new RangeError(`Frame ${frame} is outside [0, ${frames})`);
  }
  const plane = width * height;
  if (data.length !== 3 * frames * plane) {
    throw new Error(
      `Video element count ${data.length} does not match [3, ${frames}, ${height}, ${width}]`,
    );
  }
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let channel = 0; channel < 3; channel += 1) {
    const base = (channel * frames + frame) * plane;
    for (let index = 0; index < plane; index += 1) {
      const raw = data[base + index];
      // MUST: never turn a non-finite value into a pixel silently (`Uint8ClampedArray` makes NaN 0).
      if (!Number.isFinite(raw)) {
        throw new Error(
          `Frame ${frame}, pixel (x=${index % width}, y=${Math.floor(index / width)}): ` +
            `channel ${channel} is non-finite`,
        );
      }
      rgba[index * 4 + channel] = Math.round(Math.min(1, Math.max(0, raw / 2 + 0.5)) * 255);
    }
  }
  for (let index = 0; index < plane; index += 1) rgba[index * 4 + 3] = 255;
  return rgba;
};
