/**
 * The three views of a matte: the matte itself (opaque grayscale), the cut-out (the image with the
 * matte as straight alpha, for display over a checkerboard) and the cut-out composited on white
 * (opaque, so `encodePng` accepts it). The pipeline returns the matte only; compositing is the
 * page's choice (the module doc of `BirefnetPipeline`). Pure, so `vp test` covers it.
 */

import type { AlphaMatte, Rgb8Image } from "@karume/models/birefnet";

import type { RgbaPixels } from "@/lib/image";

const assertMatte = (matte: AlphaMatte): void => {
  if (matte.data.length !== matte.width * matte.height) {
    throw new RangeError(
      `The matte has ${matte.data.length} bytes, expected ${matte.width}×${matte.height}`,
    );
  }
};

/** The matte and the image it was made from must have the same size (the pipeline's contract). */
const assertPair = (image: Rgb8Image, matte: AlphaMatte): void => {
  assertMatte(matte);
  if (image.width !== matte.width || image.height !== matte.height) {
    throw new RangeError(
      `The matte is ${matte.width}×${matte.height} but the image is ${image.width}×${image.height}`,
    );
  }
  if (image.data.length !== image.width * image.height * 3) {
    throw new RangeError(
      `The image has ${image.data.length} RGB bytes, expected 3×${image.width}×${image.height}`,
    );
  }
};

/** The matte as an opaque grayscale image (white = foreground). */
export const matteRgba = (matte: AlphaMatte): RgbaPixels => {
  assertMatte(matte);
  const data = new Uint8ClampedArray(matte.data.length * 4);
  for (let pixel = 0; pixel < matte.data.length; pixel += 1) {
    const alpha = matte.data[pixel] ?? 0;
    data[pixel * 4] = alpha;
    data[pixel * 4 + 1] = alpha;
    data[pixel * 4 + 2] = alpha;
    data[pixel * 4 + 3] = 255;
  }
  return { data, width: matte.width, height: matte.height };
};

/** The image with the matte as straight alpha (translucent: for display, not for `encodePng`). */
export const cutoutRgba = (image: Rgb8Image, matte: AlphaMatte): RgbaPixels => {
  assertPair(image, matte);
  const data = new Uint8ClampedArray(matte.data.length * 4);
  for (let pixel = 0; pixel < matte.data.length; pixel += 1) {
    data[pixel * 4] = image.data[pixel * 3] ?? 0;
    data[pixel * 4 + 1] = image.data[pixel * 3 + 1] ?? 0;
    data[pixel * 4 + 2] = image.data[pixel * 3 + 2] ?? 0;
    data[pixel * 4 + 3] = matte.data[pixel] ?? 0;
  }
  return { data, width: matte.width, height: matte.height };
};

/**
 * The cut-out composited on white: `round(c · a + 255 · (1 − a))` per channel with `a = α / 255`
 * (the usage example of `BirefnetPipeline`), alpha 255.
 */
export const compositeOnWhite = (image: Rgb8Image, matte: AlphaMatte): RgbaPixels => {
  assertPair(image, matte);
  const data = new Uint8ClampedArray(matte.data.length * 4);
  for (let pixel = 0; pixel < matte.data.length; pixel += 1) {
    const alpha = (matte.data[pixel] ?? 0) / 255;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = image.data[pixel * 3 + channel] ?? 0;
      data[pixel * 4 + channel] = Math.round(value * alpha + 255 * (1 - alpha));
    }
    data[pixel * 4 + 3] = 255;
  }
  return { data, width: matte.width, height: matte.height };
};
