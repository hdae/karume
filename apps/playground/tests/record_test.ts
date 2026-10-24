import { createHash } from "node:crypto";
import { wanFrameToRgba } from "@karume/models/wan";
import { describe, expect, it } from "vite-plus/test";

import {
  attentionPipelines,
  buildCostOf,
  countNonFinite,
  foldComponentDiagnostics,
  intervalSeconds,
  pipelineOp,
  sha256Hex,
  stackRgbaFrames,
  summarizeDevice,
  summarizeDiagnostics,
  summarizeIntervals,
  totalBuildCost,
} from "../src/lib/record.ts";

describe("countNonFinite", () => {
  it("counts NaN and both infinities, not finite extremes", () => {
    const data = Float32Array.of(0, Number.NaN, 1, Infinity, -Infinity, 3.4e38, -0);
    expect(countNonFinite(data)).toBe(3);
  });

  it("is 0 for an all-finite buffer", () => {
    expect(countNonFinite(new Float32Array(16))).toBe(0);
  });
});

describe("frames → RGBA → sha256", () => {
  // A [3, frames = 2, height = 1, width = 2] video in [-1, 1] (channel-major, as Wan returns it).
  const video = {
    frames: 2,
    width: 2,
    height: 1,
    // R plane: frame 0 = (-1, 1), frame 1 = (0, 0.5); G plane: …; B plane: …
    data: Float32Array.of(-1, 1, 0, 0.5, 1, -1, -0.5, 0, 0, 0, 1, 1),
  };
  // clamp(x/2 + 0.5) · 255, rounded: -1 → 0, -0.5 → 64, 0 → 128, 0.5 → 191, 1 → 255.
  const expected = Uint8Array.of(
    // frame 0: pixel 0 (R -1, G 1, B 0), pixel 1 (R 1, G -1, B 0)
    0,
    255,
    128,
    255,
    255,
    0,
    128,
    255,
    // frame 1: pixel 0 (R 0, G -0.5, B 1), pixel 1 (R 0.5, G 0, B 1)
    128,
    64,
    255,
    255,
    191,
    128,
    255,
    255,
  );

  const stack = stackRgbaFrames(video.frames, video.width, video.height, (frame) =>
    wanFrameToRgba(video, frame),
  );

  it("lays the frames out one after another", () => {
    expect(Array.from(stack.bytes)).toEqual(Array.from(expected));
    expect(Array.from(stack.frame(1))).toEqual(Array.from(expected.subarray(8)));
  });

  it("hashes the concatenated RGBA bytes", async () => {
    const reference = createHash("sha256").update(expected).digest("hex");
    expect(await sha256Hex(stack.bytes)).toBe(reference);
  });

  it("rejects a frame of the wrong size", () => {
    expect(() => stackRgbaFrames(1, 2, 2, () => new Uint8ClampedArray(4))).toThrow(/expected 16/);
  });
});

describe("summarizeDiagnostics", () => {
  const report = (allocatedBytes: number, submitCount: number) => ({
    weights: { allocatedBytes },
    submit: { submitCount, dispatchCount: submitCount * 10 },
    geometryProfile: "default",
  });

  it("counts runs and keeps the latest figures", () => {
    const first = summarizeDiagnostics(undefined, report(2 ** 30, 3));
    const second = summarizeDiagnostics(first, report(1.5 * 2 ** 30, 7));
    expect(second).toEqual({
      runs: 2,
      weightsAllocatedGiB: 1.5,
      submitCount: 7,
      dispatchCount: 70,
      geometryProfile: "default",
    });
  });
});

const buildStats = (scale: number) => ({
  partCount: 2 * scale,
  supplyWaitMs: 1000.04 * scale,
  decodeMs: 0,
  bufferCreateMs: 2.26 * scale,
  writeBufferIssueMs: 30 * scale,
  uploadFenceMs: 400.56 * scale,
  uploadedBytes: 2 ** 30 * scale,
});

describe("build cost", () => {
  it("keeps the host and GPU-wait fields apart at 0.1 ms", () => {
    expect(buildCostOf(buildStats(1))).toEqual({
      partCount: 2,
      supplyWaitMs: 1000,
      decodeMs: 0,
      bufferCreateMs: 2.3,
      writeBufferIssueMs: 30,
      uploadFenceMs: 400.6,
      uploadedBytes: 2 ** 30,
    });
  });

  it("sums the components field by field", () => {
    const total = totalBuildCost([buildCostOf(buildStats(1)), buildCostOf(buildStats(3))]);
    expect(total).toEqual({
      partCount: 8,
      supplyWaitMs: 4000.1,
      decodeMs: 0,
      bufferCreateMs: 9.1,
      writeBufferIssueMs: 120,
      uploadFenceMs: 1602.3,
      uploadedBytes: 4 * 2 ** 30,
    });
  });

  it("is all zero for no components", () => {
    expect(totalBuildCost([]).supplyWaitMs).toBe(0);
  });
});

describe("foldComponentDiagnostics", () => {
  const report = (scale: number, keys: readonly string[] | undefined) => ({
    weights: { allocatedBytes: 2 ** 30 },
    submit: { submitCount: scale, dispatchCount: scale * 10 },
    geometryProfile: "nvidia-blackwell",
    buildStats: buildStats(scale),
    lastRunPipelines: keys?.map((key, index) => ({ key, dispatchCount: index + 1 })),
  });

  it("counts runs and keeps the build cost and pipelines of the latest report", () => {
    const first = foldComponentDiagnostics(undefined, report(1, ["linear:v1:a"]));
    const second = foldComponentDiagnostics(
      first,
      report(2, ["attention_online:v1:x", "linear:v1:a"]),
    );
    expect(second.runs).toBe(2);
    expect(second.build.partCount).toBe(4);
    expect(second.lastRunPipelines).toEqual([
      { key: "attention_online:v1:x", dispatchCount: 1 },
      { key: "linear:v1:a", dispatchCount: 2 },
    ]);
  });

  it("leaves the pipelines out when the last run reported none", () => {
    expect(foldComponentDiagnostics(undefined, report(1, undefined))).not.toHaveProperty(
      "lastRunPipelines",
    );
  });
});

describe("the attention route", () => {
  const pipelines = [
    { key: "attention_qk:v3:i8a8:reg:dp4a", dispatchCount: 30 },
    { key: "attention_stats:v2:f32:lastdim", dispatchCount: 30 },
    { key: "attention_pv:v3:i8a8:reg:dp4a", dispatchCount: 30 },
    { key: "linear:v1:i8a8", dispatchCount: 300 },
  ];

  it("picks the materialized attention kernels", () => {
    expect(attentionPipelines(pipelines).map(({ key }) => pipelineOp(key))).toEqual([
      "attention_qk",
      "attention_stats",
      "attention_pv",
    ]);
  });

  it("picks the online kernel and nothing else", () => {
    const online = [{ key: "attention_online:v1:i8a8:d128", dispatchCount: 30 }, pipelines[3]];
    expect(attentionPipelines(online)).toEqual([online[0]]);
  });

  it("reads a key without a colon as its own op", () => {
    expect(pipelineOp("copy")).toBe("copy");
  });
});

describe("summarizeDevice", () => {
  it("sorts the sets and keeps the three limits and the flags", () => {
    expect(
      summarizeDevice({
        features: new Set(["subgroups", "shader-f16", "timestamp-query"]),
        wgslLanguageFeatures: new Set(["subgroup_id", "packed_4x8_integer_dot_product"]),
        subgroupLanesLinear: true,
        gpuTimingEnabled: false,
        limits: {
          maxComputeWorkgroupStorageSize: 32768,
          maxStorageBufferBindingSize: 2 ** 31,
          maxBufferSize: 2 ** 32,
        },
      }),
    ).toEqual({
      features: ["shader-f16", "subgroups", "timestamp-query"],
      wgslLanguageFeatures: ["packed_4x8_integer_dot_product", "subgroup_id"],
      subgroupLanesLinear: true,
      gpuTimingEnabled: false,
      limits: {
        maxComputeWorkgroupStorageSize: 32768,
        maxStorageBufferBindingSize: 2 ** 31,
        maxBufferSize: 2 ** 32,
      },
    });
  });
});

describe("the timeline", () => {
  it("measures the first interval from the stage start and the rest from the previous mark", () => {
    expect(intervalSeconds(1000, [3500, 4000, 4600.4])).toEqual([2.5, 0.5, 0.6]);
  });

  it("has no intervals without marks", () => {
    expect(intervalSeconds(1000, [])).toEqual([]);
  });

  it("keeps the first interval apart from the spread of the rest", () => {
    expect(summarizeIntervals([9, 3, 1, 2])).toEqual({
      first: 9,
      rest: { count: 3, min: 1, median: 2, max: 3 },
    });
  });

  it("takes the mean of the middle two for an even count", () => {
    expect(summarizeIntervals([9, 4, 1, 2, 3])?.rest?.median).toBe(2.5);
  });

  it("has no spread for a single interval and no summary for none", () => {
    expect(summarizeIntervals([7])).toEqual({ first: 7 });
    expect(summarizeIntervals([])).toBeUndefined();
  });
});
