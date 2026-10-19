/**
 * VAE decode output (planar `[1,3,H,W]`, range `[-1,1]`) → interleaved 8-bit RGBA.
 *
 * The rule computes `VaeImageProcessor.postprocess`'s `v = clamp(x/2 + 0.5, 0, 1)` and `v·255` in
 * **f64** and rounds to 8 bits with `Math.round` (0.5 rounds up).
 *
 * NOTE: this rule does not match upstream to the 8-bit value. diffusers' `numpy_to_pil` is the f32
 * `(images * 255).round()` (numpy's round-half-to-even), so values near an 8-bit boundary differ by
 * ±1 (e.g. x = −0.4941176176 is 64.5 → 64 in f32, here f64 64.500004 → 65). Pixel values and the
 * sha256 reference rows are fixed by this rule (do not change it — review finding X-3 of the
 * ADR 0118 review of 2026-10-03 was recorded only). When comparing the 8-bit output with upstream,
 * assume this difference.
 *
 * NOTE: **the double clamp is intentional** (harmless). The `clamp(-1,1)` of
 * `AutoencoderKLQwenImage._decode` is baked into the IR, so on the execution path the values are
 * already saturated, but this function takes "a planar image in [-1,1]" as its entry contract.
 * MUST: do not move the clamp **before** the `[0,1]` conversion (confused with the clamp from
 * postprocess, negative values would collapse to 0 before +0.5 and the shadows would stick to
 * mid-grey).
 *
 * Independent of the DOM (returns bytes instead of creating `ImageData`), so the result can go
 * straight to the browser's `putImageData` or to PNG writing.
 */
export const imageToRgba = (
  image: Float32Array<ArrayBuffer>,
  width: number,
  height: number,
): Uint8ClampedArray<ArrayBuffer> => {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`Image size ${width}×${height} is not a positive integer`);
  }
  const plane = width * height;
  if (image.length !== plane * 3) {
    throw new Error(`Image element count ${image.length} does not match 3×${width}×${height}`);
  }
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let index = 0; index < plane; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const raw = image[channel * plane + index];
      // MUST: do not return non-finite values silently. RGBA quantization (assignment to
      // `Uint8ClampedArray`) turns NaN into 0 and ±Inf into 0/255, i.e. "normal pixels", so this is
      // the last checkpoint.
      if (!Number.isFinite(raw)) {
        const x = index % width;
        const y = Math.floor(index / width);
        throw new Error(`Pixel (x=${x}, y=${y}): channel ${channel} is non-finite`);
      }
      const value = Math.min(1, Math.max(0, raw / 2 + 0.5));
      rgba[index * 4 + channel] = Math.round(value * 255);
    }
    rgba[index * 4 + 3] = 255;
  }
  return rgba;
};
