import { createHash } from "node:crypto";
import { wanFrameToRgba } from "@karume/models/wan";
import { describe, expect, it } from "vite-plus/test";

import type { PipelineSource } from "../src/lib/distributions.ts";
import {
  attentionPipelines,
  buildCostOf,
  countNonFinite,
  f32Bytes,
  f32Sha256,
  foldComponentDiagnostics,
  foldStageEvent,
  intervalSeconds,
  NO_SUBMITS,
  perRunSubmitCounts,
  pipelineOp,
  rgb8Sha256,
  sha256Hex,
  sourceRecord,
  stackRgbaFrames,
  type StageTimes,
  stageSecondsOf,
  submitCountsSince,
  summarizeDevice,
  summarizeDiagnostics,
  summarizeIntervals,
  tallyTiming,
  timingOf,
  type TimingTally,
  topOpShares,
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

describe("GPU time per component", () => {
  // One run's `lastRunTiming` (ns, as the runtime reports them).
  const run = (
    entries: readonly { key: string; ns: number; dispatchCount: number }[],
    clampedNegativeSamples = 0,
  ) => ({
    entries: entries.map((entry) => ({ ...entry, workgroupCount: entry.dispatchCount * 64 })),
    totalNs: entries.reduce((total, { ns }) => total + ns, 0),
    dispatchCount: entries.reduce((total, { dispatchCount }) => total + dispatchCount, 0),
    clampedNegativeSamples,
  });
  const tallyOf = (runs: readonly ReturnType<typeof run>[]): TimingTally | undefined =>
    runs.reduce<TimingTally | undefined>((tally, timing) => tallyTiming(tally, timing), undefined);
  const forward = run([
    { key: "linear:v1:i8a8:n3072", ns: 1_000_040, dispatchCount: 30 },
    { key: "linear:v1:i8a8:n9216", ns: 2_000_030, dispatchCount: 30 },
    { key: "attention_online:v1:i8a8:d128", ns: 700_000, dispatchCount: 30 },
    { key: "ew", ns: 300_000, dispatchCount: 90 },
  ]);

  describe("when every run is timed", () => {
    const tally = tallyOf([forward, forward, forward]);
    const timing = tally === undefined ? undefined : timingOf(tally);

    it("sums each pipeline key over the runs, in ms at 0.1 ms, most expensive first", () => {
      expect(timing?.runs).toBe(3);
      expect(timing?.keys).toEqual([
        { key: "linear:v1:i8a8:n9216", ms: 6, dispatchCount: 90 },
        { key: "linear:v1:i8a8:n3072", ms: 3, dispatchCount: 90 },
        { key: "attention_online:v1:i8a8:d128", ms: 2.1, dispatchCount: 90 },
        { key: "ew", ms: 0.9, dispatchCount: 270 },
      ]);
    });

    it("folds the keys per op (the text before the first ':')", () => {
      expect(timing?.ops).toEqual([
        { op: "linear", ms: 9, dispatchCount: 180 },
        { op: "attention_online", ms: 2.1, dispatchCount: 90 },
        { op: "ew", ms: 0.9, dispatchCount: 270 },
      ]);
      expect(timing?.totalMs).toBe(12);
      expect(timing?.dispatchCount).toBe(540);
    });

    it("rounds once, after summing: 0.04 ms a run still adds up over three runs", () => {
      const small = tallyOf(
        [1, 2, 3].map(() => run([{ key: "copy", ns: 40_000, dispatchCount: 1 }])),
      );
      const summed = small === undefined ? undefined : timingOf(small);
      expect(summed?.totalMs).toBe(0.1);
      expect(summed?.keys).toEqual([{ key: "copy", ms: 0.1, dispatchCount: 3 }]);
    });
  });

  describe("when a run reports no breakdown", () => {
    it("has no tally on a device without timing", () => {
      expect(tallyTiming(undefined, undefined)).toBeUndefined();
    });

    it("has no tally for runs with nothing timed (a batch section on a timing device)", () => {
      expect(tallyOf([run([]), run([])])).toBeUndefined();
    });

    it("keeps the earlier runs and does not count the untimed ones", () => {
      const tally = tallyOf([forward, run([])]);
      expect(tally?.runs).toBe(1);
      expect(tallyTiming(tally, undefined)).toBe(tally);
    });
  });

  it("adds up the clamped samples so a questionable breakdown stays visible", () => {
    const tally = tallyOf([run(forward.entries, 2), run(forward.entries, 1)]);
    expect(tally === undefined ? undefined : timingOf(tally).clampedNegativeSamples).toBe(3);
  });

  it("does not change the tally it folds into", () => {
    const first = tallyTiming(undefined, forward);
    tallyTiming(first, forward);
    expect(first?.keys.get("ew")).toEqual({ ns: 300_000, dispatchCount: 90 });
  });

  it("names the most expensive ops with their share of the component's total", () => {
    const tally = tallyTiming(undefined, forward);
    const shares = tally === undefined ? [] : topOpShares(timingOf(tally), 2);
    expect(shares.map(({ op, ms }) => [op, ms])).toEqual([
      ["linear", 3],
      ["attention_online", 0.7],
    ]);
    expect(shares[0].share).toBeCloseTo(0.75, 6);
    expect(shares[1].share).toBeCloseTo(0.175, 6);
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

describe("f32Bytes / f32Sha256", () => {
  it("views exactly the values' bytes, even inside a larger buffer", () => {
    const backing = new Float32Array([9, 1.5, -2, 9]);
    const bytes = f32Bytes(backing.subarray(1, 3));
    expect(bytes.byteLength).toBe(8);
    expect(Array.from(new Float32Array(bytes.slice().buffer))).toEqual([1.5, -2]);
  });

  it("hashes the little-endian bytes of the f32 values, not the whole buffer", async () => {
    const view = new Float32Array([9, 1, 0.5, 9]).subarray(1, 3);
    expect(Array.from(f32Bytes(view))).toEqual([0, 0, 0x80, 0x3f, 0, 0, 0, 0x3f]);
    // sha256 of 0000803f0000003f (Node's crypto as the oracle).
    expect(await f32Sha256(view)).toBe(
      createHash("sha256")
        .update(Uint8Array.of(0, 0, 0x80, 0x3f, 0, 0, 0, 0x3f))
        .digest("hex"),
    );
  });

  it("writes little-endian f32 (1.0 = 00 00 80 3f, -2 = 00 00 00 c0)", () => {
    expect([...f32Bytes(new Float32Array([1, -2]))]).toEqual([
      0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0,
    ]);
  });

  it("gives the empty sha256 for no values", async () => {
    expect(await f32Sha256(new Float32Array(0))).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("tells one changed sample apart", async () => {
    const left = await f32Sha256(new Float32Array([0.1, 0.2]));
    const right = await f32Sha256(new Float32Array([0.1, 0.2000001]));
    expect(left).not.toBe(right);
  });

  it("tells -0 from 0 (the bytes differ, so a sign flip shows in the hash)", async () => {
    expect(await f32Sha256(new Float32Array([0]))).not.toBe(
      await f32Sha256(new Float32Array([-0])),
    );
  });
});

describe("rgb8Sha256", () => {
  it("hashes only the image's bytes when its data is a view into a larger buffer", async () => {
    // A 2×1 image (6 bytes) in the middle of a 10-byte buffer.
    const data = new Uint8Array([9, 9, 0, 0, 0x80, 0x3f, 0, 0, 9, 9]).subarray(2, 8);
    expect(await rgb8Sha256({ data })).toBe(
      createHash("sha256")
        .update(Uint8Array.of(0, 0, 0x80, 0x3f, 0, 0))
        .digest("hex"),
    );
  });
});

describe("sourceRecord", () => {
  it("keeps the cached route's repo reference (not the cache generation), and nothing for direct", () => {
    const cached: PipelineSource = {
      kind: "cached",
      ref: {
        repo: "hdae/karume-depth-anything-v2",
        revision: "a".repeat(40),
        hubUrl: "http://x/hf",
      },
      cacheGeneration: 3,
    };
    expect(sourceRecord(cached)).toEqual({
      kind: "cached",
      repo: "hdae/karume-depth-anything-v2",
      revision: "a".repeat(40),
      hubUrl: "http://x/hf",
    });
    expect(sourceRecord({ kind: "direct" })).toEqual({ kind: "direct" });
  });
});

describe("submitCountsSince", () => {
  it("is the submits and dispatches between two reports of one Session", () => {
    expect(
      submitCountsSince(
        { submitCount: 3, dispatchCount: 120 },
        { submitCount: 5, dispatchCount: 200 },
      ),
    ).toEqual({ submitCount: 2, dispatchCount: 80 });
    expect(submitCountsSince(NO_SUBMITS, { submitCount: 2, dispatchCount: 60 })).toEqual({
      submitCount: 2,
      dispatchCount: 60,
    });
  });

  it("throws when a count went down (the reports came from different Sessions)", () => {
    expect(() =>
      submitCountsSince(
        { submitCount: 30, dispatchCount: 1200 },
        { submitCount: 2, dispatchCount: 60 },
      ),
    ).toThrow(/went down/);
  });
});

describe("perRunSubmitCounts", () => {
  const report = (submitCount: number, dispatchCount: number) => ({
    submit: { submitCount, dispatchCount, measuredCount: submitCount },
    geometryProfile: "default",
  });

  it("gives each report of one Session the counts of its own run, not the Session's totals", () => {
    const perRun = perRunSubmitCounts();
    // A Session that submits 2 and dispatches 300 per run reports 2/300, 4/600, 6/900 over its life.
    expect([perRun(report(2, 300)), perRun(report(4, 600)), perRun(report(6, 900))]).toEqual([
      report(2, 300),
      { ...report(2, 300), submit: { submitCount: 2, dispatchCount: 300, measuredCount: 4 } },
      { ...report(2, 300), submit: { submitCount: 2, dispatchCount: 300, measuredCount: 6 } },
    ]);
  });

  it("keeps a separate count per Session (a new pipeline starts from zero)", () => {
    const first = perRunSubmitCounts();
    first(report(5, 500));
    expect(perRunSubmitCounts()(report(2, 300)).submit).toMatchObject({
      submitCount: 2,
      dispatchCount: 300,
    });
  });

  it("throws when a report's counts go down (reports of two Sessions mixed)", () => {
    const perRun = perRunSubmitCounts();
    perRun(report(5, 500));
    expect(() => perRun(report(2, 300))).toThrow(/went down/);
  });
});

describe("foldStageEvent / stageSecondsOf", () => {
  type Stage = "encode" | "denoise" | "decode";
  const ORDER: readonly Stage[] = ["encode", "denoise", "decode"];

  it("records a stage's start and end, and leaves the other stages alone", () => {
    let times: StageTimes<Stage> = { encode: { start: 0, end: 5 } };
    times = foldStageEvent(times, "denoise", "start", 10);
    times = foldStageEvent(times, "denoise", "end", 250);
    expect(times).toEqual({ encode: { start: 0, end: 5 }, denoise: { start: 10, end: 250 } });
  });

  it("gives the seconds of the stages that ended, in the given order, at millisecond precision", () => {
    let times: StageTimes<Stage> = {};
    times = foldStageEvent(times, "decode", "start", 2000);
    times = foldStageEvent(times, "decode", "end", 3234.5678);
    times = foldStageEvent(times, "encode", "start", 0);
    times = foldStageEvent(times, "encode", "end", 1500);
    const stages = stageSecondsOf(times, ORDER);
    expect(stages).toEqual({ encode: 1.5, decode: 1.235 });
    expect(Object.keys(stages)).toEqual(["encode", "decode"]);
  });

  it("counts an end without a start from the end itself (zero seconds)", () => {
    const times = foldStageEvent<Stage>({}, "decode", "end", 50);
    expect(times).toEqual({ decode: { start: 50, end: 50 } });
    expect(stageSecondsOf(times, ORDER)).toEqual({ decode: 0 });
  });

  it("leaves out a stage that has not ended", () => {
    expect(stageSecondsOf(foldStageEvent<Stage>({}, "denoise", "start", 10), ORDER)).toEqual({});
  });
});
