import { describe, expect, it } from "vite-plus/test";

import { BUILD_NOTE, loadingProgress } from "../src/models/birefnet/progress.ts";

const MIB = 2 ** 20;

describe("loadingProgress", () => {
  describe("while parts are still arriving", () => {
    it("reports the download with the file being read and a MiB counter", () => {
      expect(
        loadingProgress({ loaded: 100 * MIB, total: 432 * MIB, path: "shared/matte/part-3.krm" }),
      ).toEqual({
        stage: "loading weights",
        count: { done: 100, total: 432, unit: "MiB" },
        note: "shared/matte/part-3.krm",
      });
    });
  });

  describe("once every part is in (Cached: in the browser's cache; Direct: at once)", () => {
    it("names the Session build as its own stage, not cancellable, and says why", () => {
      const progress = loadingProgress({
        loaded: 432 * MIB,
        total: 432 * MIB,
        path: "shared/matte/part-4.krm",
      });
      expect(progress.stage).toBe("building the Session");
      expect(progress.count).toEqual({ done: 432, total: 432, unit: "MiB" });
      expect(progress.cancellable).toBe(false);
      expect(progress.cancelNote).toBe(BUILD_NOTE);
      expect(BUILD_NOTE).toMatch(/Cancel is unavailable until it ends/);
    });
  });
});
