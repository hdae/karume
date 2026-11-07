import { describe, expect, it } from "vite-plus/test";
import {
  fitCapacityCandidates,
  GEMMA4_FIT_MARGIN_BYTES,
  pickFitCapacity,
} from "../src/gemma/fit-capacity.ts";

describe("capacity: fit", () => {
  describe("the candidates", () => {
    it("go from the position limit down the powers of two to the default", () => {
      expect(fitCapacityCandidates(4096, 32768)).toEqual([32768, 16384, 8192, 4096]);
      expect(fitCapacityCandidates(4096, 262144)).toEqual([
        262144, 131072, 65536, 32768, 16384, 8192, 4096,
      ]);
    });

    it("keep a position limit that is not a power of two as the first one", () => {
      expect(fitCapacityCandidates(4096, 20000)).toEqual([20000, 16384, 8192, 4096]);
    });

    it("are the default alone when it is the position limit", () => {
      expect(fitCapacityCandidates(4096, 4096)).toEqual([4096]);
    });
  });

  describe("the pick", () => {
    it("takes the first candidate whose need and margin fit", async () => {
      const tried: number[] = [];
      const picked = await pickFitCapacity(
        [8192, 4096],
        (capacity) => capacity,
        (bytes) => {
          tried.push(bytes);
          return Promise.resolve(bytes < GEMMA4_FIT_MARGIN_BYTES + 5000);
        },
      );
      expect(picked).toBe(4096);
      expect(tried).toEqual([8192 + GEMMA4_FIT_MARGIN_BYTES, 4096 + GEMMA4_FIT_MARGIN_BYTES]);
    });

    it("skips a candidate this device cannot hold, without a trial allocation", async () => {
      const tried: number[] = [];
      const picked = await pickFitCapacity(
        [8192, 4096],
        (capacity) => (capacity === 8192 ? undefined : capacity),
        (bytes) => {
          tried.push(bytes);
          return Promise.resolve(true);
        },
      );
      expect(picked).toBe(4096);
      expect(tried).toHaveLength(1);
    });

    it("returns undefined when nothing fits", async () => {
      const picked = await pickFitCapacity(
        [4096],
        (capacity) => capacity,
        () => Promise.resolve(false),
      );
      expect(picked).toBeUndefined();
    });
  });
});
