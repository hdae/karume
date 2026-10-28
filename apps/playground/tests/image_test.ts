import { describe, expect, it } from "vite-plus/test";

import { rgbaToRgb8 } from "../src/lib/image.ts";

describe("rgbaToRgb8", () => {
  it("keeps red, green and blue of every pixel in order and drops alpha", () => {
    const rgba = Uint8Array.from([1, 2, 3, 255, 4, 5, 6, 0, 7, 8, 9, 128]);
    const rgb = rgbaToRgb8({ data: rgba, width: 3, height: 1 });
    expect(rgb.width).toBe(3);
    expect(rgb.height).toBe(1);
    expect([...rgb.data]).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("refuses pixels that do not match the size", () => {
    expect(() => rgbaToRgb8({ data: new Uint8Array(12), width: 2, height: 2 })).toThrow(
      /does not match 4×2×2/,
    );
  });
});
