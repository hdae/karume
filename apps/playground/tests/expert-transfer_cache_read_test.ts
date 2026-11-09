import type { AssetRangeReader, FileRef } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  drawRanges,
  EXPERT_RANGES,
  runCacheRead,
} from "../src/models/expert-transfer/cache-read.ts";
import { EXPERT_BYTES } from "../src/models/expert-transfer/stats.ts";

const part = (path: string, size: number): FileRef => ({ path, size, sha256: "0".repeat(64) });

const PARTS = [part("a.krm", 40_000_000), part("b.krm", 30_000_000)];

/** A reader that records each read and returns exactly the bytes asked for. */
const recordingReader = (
  path: string,
  reads: { path: string; offset: number; length: number }[],
): AssetRangeReader => ({
  cost: "seek",
  read: async (offset, length) => {
    reads.push({ path, offset, length });
    return new Uint8Array(length);
  },
});

describe("drawRanges", () => {
  it("draws the four ranges of each expert, aligned and inside their part", () => {
    const ranges = drawRanges(PARTS, 3, 7);
    expect(ranges.map(({ length }) => length)).toEqual([
      ...EXPERT_RANGES,
      ...EXPERT_RANGES,
      ...EXPERT_RANGES,
    ]);
    for (const { part: drawn, offset, length } of ranges) {
      expect(offset % 4).toBe(0);
      expect(offset + length).toBeLessThanOrEqual(drawn.size);
    }
  });

  it("adds up to one expert's bytes per expert", () => {
    expect(EXPERT_RANGES.reduce((sum, length) => sum + length, 0)).toBe(EXPERT_BYTES);
  });

  it("is fixed by its seed and moves with it", () => {
    expect(drawRanges(PARTS, 2, 5)).toEqual(drawRanges(PARTS, 2, 5));
    expect(drawRanges(PARTS, 2, 5)).not.toEqual(drawRanges(PARTS, 2, 6));
  });

  it("refuses a part smaller than a range", () => {
    expect(() => drawRanges([part("tiny.krm", 1000)], 1, 1)).toThrow("smaller than");
  });
});

describe("runCacheRead", () => {
  it("reads every sample's ranges twice, opens each part once and keeps the samples after the warm-up", async () => {
    const reads: { path: string; offset: number; length: number }[] = [];
    const opened: string[] = [];
    const results = await runCacheRead({
      parts: PARTS,
      open: async (opening) => {
        opened.push(opening.path);
        return recordingReader(opening.path, reads);
      },
      batches: [2],
      samples: 3,
      warmup: 1,
      signal: new AbortController().signal,
      onProgress: () => undefined,
    });
    // 4 samples (1 warm-up) × 2 experts × 4 ranges × 2 reads.
    expect(reads).toHaveLength(4 * 2 * 4 * 2);
    // The second read of a sample repeats its first read range for range.
    const sampleSize = 2 * 4;
    const first = reads.slice(0, sampleSize);
    expect(reads.slice(sampleSize, 2 * sampleSize)).toEqual(first);
    expect(new Set(opened).size).toBe(opened.length);
    expect(results).toHaveLength(1);
    expect(results[0]?.samples).toHaveLength(3);
    expect(results[0]?.warmSamples).toHaveLength(3);
    expect(results[0]?.bytes).toBe(2 * EXPERT_BYTES);
  });

  it("fails on a short read", async () => {
    await expect(
      runCacheRead({
        parts: PARTS,
        open: async () => ({ cost: "seek", read: async () => new Uint8Array(1) }),
        batches: [1],
        samples: 1,
        warmup: 0,
        signal: new AbortController().signal,
        onProgress: () => undefined,
      }),
    ).rejects.toThrow("returned 1 bytes");
  });

  it("stops before the next sample once aborted", async () => {
    const controller = new AbortController();
    controller.abort(new Error("stop"));
    await expect(
      runCacheRead({
        parts: PARTS,
        open: async (opening) => recordingReader(opening.path, []),
        batches: [1],
        samples: 1,
        warmup: 0,
        signal: controller.signal,
        onProgress: () => undefined,
      }),
    ).rejects.toThrow("stop");
  });
});
