/**
 * Images in and out of the vision pages: a picked file → RGBA (for the preview) and RGB8 (the
 * `Rgb8Image` the BiRefNet / Depth Anything / SigLIP 2 pipelines take).
 */

import type { Rgb8Image } from "@karume/models";

/** 8-bit RGBA pixels, row-major (an `ImageData`, an Anima image, a composited matte…). */
export type RgbaPixels = {
  readonly data: Uint8ClampedArray<ArrayBuffer> | Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** A decoded file: its pixels as RGBA and as the pipelines' RGB8. */
export type LoadedImage = {
  readonly name: string;
  readonly rgba: RgbaPixels;
  readonly rgb: Rgb8Image;
};

/** Drops the alpha channel (pure). */
export const rgbaToRgb8 = (
  rgba: RgbaPixels,
): Rgb8Image & { readonly data: Uint8Array<ArrayBuffer> } => {
  const { data, width, height } = rgba;
  if (data.length !== width * height * 4) {
    throw new RangeError(`RGBA length ${data.length} does not match 4×${width}×${height}`);
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    rgb[pixel * 3] = data[pixel * 4] ?? 0;
    rgb[pixel * 3 + 1] = data[pixel * 4 + 1] ?? 0;
    rgb[pixel * 3 + 2] = data[pixel * 4 + 2] ?? 0;
  }
  return { data: rgb, width, height };
};

/**
 * Decodes an image file in the browser (`createImageBitmap` without color-space conversion, then a 2D
 * canvas). NOTE: a canvas keeps pixels premultiplied, so a fully transparent pixel comes out black —
 * the RGB a PNG stores under alpha 0 does not survive (the pipelines take RGB only).
 */
export const decodeImageFile = async (file: File): Promise<LoadedImage> => {
  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    if (context === null) throw new Error("OffscreenCanvas has no 2D context");
    context.drawImage(bitmap, 0, 0);
    const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    const rgba: RgbaPixels = { data: image.data, width: image.width, height: image.height };
    return { name: file.name, rgba, rgb: rgbaToRgb8(rgba) };
  } finally {
    bitmap.close();
  }
};
