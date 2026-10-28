import type { AssetProgress } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { loadProgress, SESSION_BUILD_NOTE } from "../src/models/depth-anything/progress.ts";

const MIB = 2 ** 20;

const event = (loaded: number, total: number): AssetProgress => ({
  phase: loaded >= total ? "complete" : "downloading",
  path: "depth/f32/part-1.krm",
  loaded,
  total,
  fileLoaded: loaded,
  fileTotal: total,
});

describe("loadProgress", () => {
  it("counts the fetch in MiB with the file being read, while files are still coming in", () => {
    expect(loadProgress(event(30 * MIB, 99 * MIB))).toEqual({
      stage: "fetching the weights",
      count: { done: 30, total: 99, unit: "MiB" },
      note: "depth/f32/part-1.krm",
    });
  });

  it("switches to the Session build, not cancellable, once every file is in", () => {
    expect(loadProgress(event(99 * MIB, 99 * MIB))).toEqual({
      stage: "building the Session",
      cancellable: false,
      cancelNote: SESSION_BUILD_NOTE,
    });
    expect(SESSION_BUILD_NOTE).toMatch(/Cancel is unavailable/);
  });
});
