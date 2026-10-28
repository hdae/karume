import { describe, expect, it } from "vite-plus/test";

import { cosineSimilarity, embeddingStats, FIRST_VALUES } from "../src/models/siglip2/embedding.ts";

describe("embeddingStats", () => {
  it("gives the L2 norm, min, max and mean of a finite embedding", () => {
    const stats = embeddingStats(new Float32Array([3, -4, 0, 1]));
    expect(stats.dimension).toBe(4);
    expect(stats.nonFinite).toBe(0);
    expect(stats.summary).toEqual({ l2Norm: 5.09902, min: -4, max: 3, mean: 0 });
  });

  it("keeps only the first values, rounded to six decimals", () => {
    const values = Float32Array.from({ length: 20 }, (_, index) => index / 3);
    const stats = embeddingStats(values);
    expect(stats.first).toHaveLength(FIRST_VALUES);
    expect(stats.first[1]).toBe(0.333333);
  });

  it("counts non-finite values and gives no summary when there are any", () => {
    const stats = embeddingStats(new Float32Array([1, Number.NaN, Number.POSITIVE_INFINITY, 2]));
    expect(stats.nonFinite).toBe(2);
    expect(stats.summary).toBeUndefined();
    expect(stats.first[1]).toBe("NaN");
  });

  it("keeps NaN, Infinity and -Infinity apart through JSON", () => {
    const stats = embeddingStats(
      new Float32Array([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0.5]),
    );
    const saved: unknown = JSON.parse(JSON.stringify(stats));
    expect(saved).toMatchObject({ first: ["NaN", "Infinity", "-Infinity", 0.5] });
  });
});

describe("cosineSimilarity", () => {
  it("is 1 for vectors pointing the same way and -1 for opposite ones, whatever their norm", () => {
    const vector = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(vector, new Float32Array([2, 4, 6]))).toBeCloseTo(1, 12);
    expect(cosineSimilarity(vector, new Float32Array([-1, -2, -3]))).toBeCloseTo(-1, 12);
  });

  it("is 0 for orthogonal vectors", () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 5]))).toBe(0);
  });

  it("refuses embeddings of different lengths (two different models)", () => {
    expect(() => cosineSimilarity(new Float32Array(768), new Float32Array(1152))).toThrow(
      /different lengths/,
    );
  });

  it("refuses non-finite values and zero vectors instead of returning NaN", () => {
    expect(() =>
      cosineSimilarity(new Float32Array([1, Number.NaN]), new Float32Array([1, 1])),
    ).toThrow(/not finite/);
    expect(() => cosineSimilarity(new Float32Array([0, 0]), new Float32Array([1, 1]))).toThrow(
      /zero vector/,
    );
  });
});
