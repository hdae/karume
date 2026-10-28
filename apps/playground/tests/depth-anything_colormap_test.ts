import { describe, expect, it } from "vite-plus/test";

import { depthRange, depthToRgba } from "../src/models/depth-anything/colormap.ts";

const pixel = (rgba: Uint8ClampedArray, index: number): number[] =>
  Array.from(rgba.subarray(index * 4, index * 4 + 4));

describe("depthRange", () => {
  it("takes min and max over the map", () => {
    expect(depthRange(new Float32Array([2, -1, 5, 0]))).toEqual({ min: -1, max: 5 });
  });

  it("refuses an empty map instead of returning an infinite range", () => {
    expect(() => depthRange(new Float32Array(0))).toThrow(RangeError);
  });
});

describe("depthToRgba", () => {
  describe("grayscale", () => {
    it("maps the minimum to black and the maximum (nearest) to white, fully opaque", () => {
      const rgba = depthToRgba(new Float32Array([10, 20, 15, 10]), 2, 2, "grayscale");
      expect(pixel(rgba, 0)).toEqual([0, 0, 0, 255]);
      expect(pixel(rgba, 1)).toEqual([255, 255, 255, 255]);
      expect(pixel(rgba, 2)).toEqual([128, 128, 128, 255]);
    });

    it("does not depend on the scale or offset of the depth (relative depth has neither)", () => {
      const a = depthToRgba(new Float32Array([0, 1, 3, 4]), 4, 1, "grayscale");
      const b = depthToRgba(new Float32Array([100, 102, 106, 108]), 4, 1, "grayscale");
      expect(Array.from(b)).toEqual(Array.from(a));
    });
  });

  describe("inferno", () => {
    it("runs from near-black (far) to light yellow (near), as matplotlib's inferno", () => {
      const rgba = depthToRgba(new Float32Array([0, 1]), 2, 1, "inferno");
      expect(pixel(rgba, 0)).toEqual([0, 0, 4, 255]);
      expect(pixel(rgba, 1)).toEqual([252, 255, 164, 255]);
    });

    it("gets lighter as the depth grows (perceptual order is kept)", () => {
      const steps = 32;
      const depth = Float32Array.from({ length: steps }, (_, index) => index);
      const rgba = depthToRgba(depth, steps, 1, "inferno");
      const luma = Array.from({ length: steps }, (_, index) => {
        const [r, g, b] = pixel(rgba, index);
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      });
      for (let index = 1; index < steps; index += 1) {
        expect(luma[index]).toBeGreaterThan(luma[index - 1]);
      }
    });
  });

  it("draws a flat map as one colour (no gradient the data does not have)", () => {
    const rgba = depthToRgba(new Float32Array([7, 7, 7]), 3, 1, "grayscale");
    expect([pixel(rgba, 0), pixel(rgba, 1), pixel(rgba, 2)]).toEqual([
      [0, 0, 0, 255],
      [0, 0, 0, 255],
      [0, 0, 0, 255],
    ]);
  });

  it("refuses a map whose length does not match its size", () => {
    expect(() => depthToRgba(new Float32Array(5), 2, 2, "grayscale")).toThrow(RangeError);
  });
});
