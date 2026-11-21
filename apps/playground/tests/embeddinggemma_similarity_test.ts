import { describe, expect, it } from "vite-plus/test";

import {
  dotProduct,
  embeddingStats,
  FIRST_VALUES,
  similarityMatrix,
} from "../src/models/embeddinggemma/similarity.ts";

/** `values` scaled to unit length (what the pipeline returns). */
const unit = (values: readonly number[]): Float32Array => {
  const norm = Math.hypot(...values);
  return Float32Array.from(values, (value) => value / norm);
};

describe("dotProduct", () => {
  it("is the cosine of two unit vectors: 1 the same way, -1 opposite, 0 orthogonal", () => {
    const vector = unit([1, 2, 3]);
    expect(dotProduct(vector, vector)).toBeCloseTo(1, 6);
    expect(dotProduct(vector, unit([-1, -2, -3]))).toBeCloseTo(-1, 6);
    expect(dotProduct(unit([1, 0]), unit([0, 1]))).toBe(0);
    expect(dotProduct(unit([1, 0]), unit([1, 1]))).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it("accumulates in f64 (no f32 rounding of the running sum)", () => {
    // 3 × (2^-12)^2 = 3 · 2^-24: lost next to 1 in f32, kept in f64.
    const left = new Float32Array([1, 2 ** -12, 2 ** -12, 2 ** -12]);
    expect(dotProduct(left, left)).toBe(1 + 3 * 2 ** -24);
  });

  it("refuses two different widths and non-finite values instead of returning a number or NaN", () => {
    expect(() => dotProduct(new Float32Array(768), new Float32Array(256))).toThrow(
      /different lengths \(768 and 256\)/,
    );
    expect(() => dotProduct(new Float32Array([1, Number.NaN]), new Float32Array([1, 1]))).toThrow(
      /Element 1 is not finite/,
    );
  });
});

describe("similarityMatrix", () => {
  it("is symmetric, about 1 on the diagonal, at six decimals", () => {
    const matrix = similarityMatrix([unit([1, 0, 0]), unit([1, 1, 0]), unit([0, 0, 1])]);
    expect(matrix).toEqual([
      [1, 0.707107, 0],
      [0.707107, 1, 0],
      [0, 0, 1],
    ]);
  });

  it("leaves the row and the column of a refused row empty and compares the others", () => {
    const matrix = similarityMatrix([unit([1, 0]), undefined, unit([0, 1])]);
    expect(matrix).toEqual([
      [1, null, 0],
      [null, null, null],
      [0, null, 1],
    ]);
  });

  it("leaves a vector with non-finite values out of every pair (its statistics count them)", () => {
    const matrix = similarityMatrix([unit([1, 0]), new Float32Array([Number.NaN, 0])]);
    expect(matrix).toEqual([
      [1, null],
      [null, null],
    ]);
  });

  it("fails loudly on two widths in one run (the page asks every row for one width)", () => {
    expect(() => similarityMatrix([new Float32Array(4), new Float32Array(2)])).toThrow(
      /different lengths/,
    );
  });
});

describe("embeddingStats", () => {
  it("gives the width, the L2 norm and the first values of a unit vector", () => {
    const values = Float32Array.from({ length: 16 }, (_, index) =>
      index % 2 === 0 ? 0.25 : -0.25,
    );
    const stats = embeddingStats(values);
    expect(stats).toMatchObject({ dimension: 16, nonFinite: 0, l2Norm: 1 });
    expect(stats.first).toHaveLength(FIRST_VALUES);
    expect(stats.first.slice(0, 2)).toEqual([0.25, -0.25]);
  });

  it("counts non-finite values, gives no norm and keeps them apart through JSON", () => {
    const stats = embeddingStats(
      new Float32Array([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0.5]),
    );
    expect(stats.nonFinite).toBe(3);
    expect(stats.l2Norm).toBeNull();
    const saved: unknown = JSON.parse(JSON.stringify(stats));
    expect(saved).toMatchObject({ first: ["NaN", "Infinity", "-Infinity", 0.5] });
  });
});
