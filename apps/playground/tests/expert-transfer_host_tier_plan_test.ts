import { describe, expect, it } from "vite-plus/test";

import {
  allocationPlan,
  drawDistinct,
  fillByRepeating,
  GIB,
  HOST_TIER_CHUNK_BYTES,
  HOST_TIER_FORMS,
  leastSquaresSlope,
  parseTargetGib,
  perExpertMs,
  placeExperts,
} from "../src/models/expert-transfer/host-tier-plan.ts";
import { EXPERT_BYTES } from "../src/models/expert-transfer/stats.ts";

describe("the host tier forms", () => {
  it("run the upload buffers first, so the arrays start once B's memory is destroyed", () => {
    expect(HOST_TIER_FORMS).toEqual(["upload-buffer", "array"]);
  });
});

describe("parseTargetGib", () => {
  it("accepts an integer from 1 to 64 with spaces around it", () => {
    expect(parseTargetGib(" 13 ")).toBe(13);
    expect(parseTargetGib("1")).toBe(1);
    expect(parseTargetGib("64")).toBe(64);
  });

  it("rejects an empty text, a fraction, zero and a value above 64", () => {
    for (const text of ["", " ", "1.5", "0", "65", "-3"]) {
      expect(parseTargetGib(text)).toBeUndefined();
    }
  });
});

describe("allocationPlan", () => {
  it("splits 13 GiB into 13 arrays of 1 GiB or 26 upload buffers of 512 MiB", () => {
    expect(allocationPlan(13 * GIB, HOST_TIER_CHUNK_BYTES.array)).toEqual(
      Array.from({ length: 13 }, () => GIB),
    );
    const upload = allocationPlan(13 * GIB, HOST_TIER_CHUNK_BYTES["upload-buffer"]);
    expect(upload).toHaveLength(26);
    expect(upload.every((size) => size === GIB / 2)).toBe(true);
  });

  it("ends with a smaller chunk when the target is not a whole number of chunks", () => {
    expect(allocationPlan(20, 8)).toEqual([8, 8, 4]);
  });

  it("refuses sizes that are not positive multiples of 4", () => {
    expect(() => allocationPlan(12, 3)).toThrow("chunkBytes");
    expect(() => allocationPlan(0, 4)).toThrow("targetBytes");
    expect(() => allocationPlan(6, 4)).toThrow("targetBytes");
  });
});

describe("placeExperts", () => {
  it("packs whole experts from the start of each chunk and leaves the short tail empty", () => {
    const places = placeExperts([2 * EXPERT_BYTES + 100, EXPERT_BYTES]);
    expect(places).toEqual([
      { chunk: 0, offset: 0 },
      { chunk: 0, offset: EXPERT_BYTES },
      { chunk: 1, offset: 0 },
    ]);
  });

  it("never lets an expert straddle the end of its chunk, at copy-aligned offsets", () => {
    const sizes = allocationPlan(2 * GIB, GIB / 2);
    const places = placeExperts(sizes);
    expect(places).toHaveLength(4 * Math.floor(GIB / 2 / EXPERT_BYTES));
    for (const { chunk, offset } of places) {
      expect(offset % 4).toBe(0);
      expect(offset + EXPERT_BYTES).toBeLessThanOrEqual(sizes[chunk] ?? 0);
    }
  });

  it("places nothing in a chunk smaller than one expert", () => {
    expect(placeExperts([EXPERT_BYTES - 4])).toEqual([]);
  });
});

describe("drawDistinct", () => {
  it("draws the asked number of distinct indices inside the range", () => {
    const picks = drawDistinct(32, 4160, 9);
    expect(picks).toHaveLength(32);
    expect(new Set(picks).size).toBe(32);
    expect(picks.every((pick) => Number.isInteger(pick) && pick >= 0 && pick < 4160)).toBe(true);
  });

  it("is a permutation when it draws the whole range", () => {
    expect(drawDistinct(50, 50, 3).toSorted((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, index) => index),
    );
  });

  it("is fixed by its seed and moves with it", () => {
    expect(drawDistinct(8, 1000, 5)).toEqual(drawDistinct(8, 1000, 5));
    expect(drawDistinct(8, 1000, 5)).not.toEqual(drawDistinct(8, 1000, 6));
  });

  it("refuses more draws than there are experts", () => {
    expect(() => drawDistinct(33, 32, 1)).toThrow("cannot draw 33");
  });
});

describe("fillByRepeating", () => {
  it("writes every byte, cutting the last repeat short", () => {
    const target = new Uint8Array(11);
    fillByRepeating(target, new Uint8Array([1, 2, 3, 4]));
    expect([...target]).toEqual([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]);
  });

  it("handles a target shorter than the pattern", () => {
    const target = new Uint8Array(2);
    fillByRepeating(target, new Uint8Array([7, 8, 9]));
    expect([...target]).toEqual([7, 8]);
  });

  it("refuses an empty pattern", () => {
    expect(() => fillByRepeating(new Uint8Array(4), new Uint8Array(0))).toThrow("non-empty");
  });
});

describe("the per-expert cost", () => {
  it("takes the fence off a batch and divides by its experts", () => {
    expect(perExpertMs(5.5, 2.5, 8)).toBeCloseTo(0.375);
  });

  it("is the least-squares slope over the batches: the fixed cost falls into the intercept", () => {
    const points = [1, 8, 16, 32].map((x) => ({ x, y: 2.7 + 0.43 * x }));
    expect(leastSquaresSlope(points)).toBeCloseTo(0.43);
  });

  it("fits noisy points by least squares", () => {
    // y = 1, 3, 2 at x = 0, 1, 2: slope = Σ(dx·dy) / Σdx² = (−1·−1 + 1·0) / 2 = 0.5.
    expect(
      leastSquaresSlope([
        { x: 0, y: 1 },
        { x: 1, y: 3 },
        { x: 2, y: 2 },
      ]),
    ).toBeCloseTo(0.5);
  });

  it("has no slope with fewer than two distinct batches", () => {
    expect(leastSquaresSlope([])).toBeUndefined();
    expect(leastSquaresSlope([{ x: 8, y: 3 }])).toBeUndefined();
    expect(
      leastSquaresSlope([
        { x: 8, y: 3 },
        { x: 8, y: 4 },
      ]),
    ).toBeUndefined();
  });
});
