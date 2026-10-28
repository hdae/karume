import { describe, expect, it } from "vite-plus/test";

import { compositeOnWhite, cutoutRgba, matteRgba } from "../src/models/birefnet/pixels.ts";

/** A 3×1 image: red, green, blue. */
const image = { data: Uint8Array.from([255, 0, 0, 0, 255, 0, 0, 0, 255]), width: 3, height: 1 };
/** Background, half, foreground. */
const matte = { data: Uint8Array.from([0, 128, 255]), width: 3, height: 1 };

describe("matteRgba", () => {
  it("draws the matte as opaque gray, white for foreground", () => {
    const rgba = matteRgba(matte);
    expect([rgba.width, rgba.height]).toEqual([3, 1]);
    expect([...rgba.data]).toEqual([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
  });

  it("refuses a matte whose bytes do not match its size", () => {
    expect(() => matteRgba({ data: new Uint8Array(2), width: 3, height: 1 })).toThrow(
      /2 bytes, expected 3×1/,
    );
  });
});

describe("cutoutRgba", () => {
  it("keeps the image's colors and takes the matte as straight alpha", () => {
    expect([...cutoutRgba(image, matte).data]).toEqual([
      255, 0, 0, 0, 0, 255, 0, 128, 0, 0, 255, 255,
    ]);
  });

  it("refuses a matte of another size than the image", () => {
    expect(() => cutoutRgba(image, { data: new Uint8Array(3), width: 1, height: 3 })).toThrow(
      /matte is 1×3 but the image is 3×1/,
    );
  });
});

describe("compositeOnWhite", () => {
  it("is white where the matte is 0, the image where it is 255, and opaque everywhere", () => {
    const rgba = compositeOnWhite(image, matte);
    expect([...rgba.data.subarray(0, 4)]).toEqual([255, 255, 255, 255]);
    expect([...rgba.data.subarray(8, 12)]).toEqual([0, 0, 255, 255]);
    expect(rgba.data.filter((_, index) => index % 4 === 3).every((alpha) => alpha === 255)).toBe(
      true,
    );
  });

  it("blends a partial alpha with white, rounding to the nearest value", () => {
    // Green at α = 128: 0 · 128/255 + 255 · 127/255 = 127 for red and blue; 255 stays 255.
    expect([...compositeOnWhite(image, matte).data.subarray(4, 8)]).toEqual([127, 255, 127, 255]);
  });

  it("refuses an image whose RGB bytes do not match its size", () => {
    expect(() => compositeOnWhite({ data: new Uint8Array(6), width: 3, height: 1 }, matte)).toThrow(
      /6 RGB bytes, expected 3×3×1/,
    );
  });
});
