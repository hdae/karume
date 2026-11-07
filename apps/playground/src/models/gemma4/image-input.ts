/**
 * The chat page's attached images: a file → the decoded RGB8 the session takes, and a small RGBA copy
 * to draw as a thumbnail. Decoding is the page's: the models package takes pixels (`Rgb8Image`) and
 * holds no image decoder. The conversions are pure, so `vp test` covers them.
 */

import type { Rgb8Image } from "@karume/models/gemma";

/**
 * RGBA → RGB8, the alpha dropped (as upstream's `convert("RGB")` does): a transparent pixel keeps the
 * color it stores, it is not composited over a background.
 */
export const rgb8OfRgba = (rgba: Uint8ClampedArray, width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    data[pixel * 3] = rgba[pixel * 4];
    data[pixel * 3 + 1] = rgba[pixel * 4 + 1];
    data[pixel * 3 + 2] = rgba[pixel * 4 + 2];
  }
  return { data, width, height };
};

/** A thumbnail as `ImageData` takes it. */
export type Thumbnail = {
  readonly data: Uint8ClampedArray<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/**
 * An opaque RGBA copy with its longest side at most `maxSide` (nearest neighbor: it is only looked at
 * in a chat bubble, and a photo's full size would be drawn for nothing on every mount).
 */
export const thumbnailOf = (image: Rgb8Image, maxSide: number): Thumbnail => {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(image.height - 1, Math.floor(y / scale));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(image.width - 1, Math.floor(x / scale));
      const from = (sourceY * image.width + sourceX) * 3;
      const to = (y * width + x) * 4;
      data[to] = image.data[from];
      data[to + 1] = image.data[from + 1];
      data[to + 2] = image.data[from + 2];
      data[to + 3] = 255;
    }
  }
  return { data, width, height };
};

/** Decodes one file in the page (`createImageBitmap` + `OffscreenCanvas`). */
export const decodeImageFile = async (file: Blob): Promise<Rgb8Image> => {
  // The file's bytes, not the display's: no color-space conversion (as PIL reads them). A 2D canvas
  // stores premultiplied alpha, so a transparent pixel reads back black, not its stored color
  // (README).
  const bitmap = await createImageBitmap(file, { colorSpaceConversion: "none" });
  try {
    const { width, height } = bitmap;
    const context = new OffscreenCanvas(width, height).getContext("2d");
    if (context === null) throw new Error("This browser gives no 2D context on an OffscreenCanvas");
    context.drawImage(bitmap, 0, 0);
    return rgb8OfRgba(context.getImageData(0, 0, width, height).data, width, height);
  } finally {
    bitmap.close();
  }
};
