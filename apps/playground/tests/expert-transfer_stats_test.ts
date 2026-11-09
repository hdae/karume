import { describe, expect, it } from "vite-plus/test";

import {
  EXPERT_BYTES,
  gigabytesPerSecond,
  parseBatches,
  summarize,
  transferCases,
} from "../src/models/expert-transfer/stats.ts";

describe("the expert transfer cases", () => {
  it("runs the fence floor and the readback first, then each batch by ring and by writeBuffer", () => {
    expect(transferCases([1, 8]).map(({ id }) => id)).toEqual([
      "fence-empty",
      "fence-copy",
      "readback-router",
      "upload-ring-1",
      "upload-writeBuffer-1",
      "upload-ring-8",
      "upload-writeBuffer-8",
    ]);
  });

  it("sizes an upload as whole experts", () => {
    const upload = transferCases([4]).find(({ id }) => id === "upload-ring-4");
    expect(upload?.bytes).toBe(4 * EXPERT_BYTES);
  });
});

describe("parseBatches", () => {
  it("keeps the order given and ignores spaces and empty entries", () => {
    expect(parseBatches(" 8, 1,,4 ", 32)).toEqual([8, 1, 4]);
  });

  it("rejects an empty list, a fraction, zero, a value above the cap and a repeat", () => {
    for (const text of ["", " , ", "1.5", "0", "33", "4,4"]) {
      expect(parseBatches(text, 32)).toBeUndefined();
    }
  });
});

describe("summarize", () => {
  it("takes the nearest-rank median and p95 of unsorted samples", () => {
    const samples = Array.from({ length: 20 }, (_, index) => 20 - index);
    expect(summarize(samples)).toEqual({ min: 1, median: 10, p95: 19, max: 20, mean: 10.5 });
  });

  it("returns the value itself for one sample", () => {
    expect(summarize([2.5])).toEqual({ min: 2.5, median: 2.5, p95: 2.5, max: 2.5, mean: 2.5 });
  });

  it("refuses an empty list", () => {
    expect(() => summarize([])).toThrow("at least one sample");
  });
});

describe("gigabytesPerSecond", () => {
  it("is bytes per millisecond over 10⁶", () => {
    expect(gigabytesPerSecond(7_000_000, 1)).toBe(7);
  });

  it("is undefined when nothing moved or no time passed", () => {
    expect(gigabytesPerSecond(0, 1)).toBeUndefined();
    expect(gigabytesPerSecond(1, 0)).toBeUndefined();
  });
});
