import { describe, expect, it } from "vite-plus/test";

import type { PipelineSource } from "../src/lib/distributions.ts";
import type { RecordEnvironment } from "../src/lib/environment.ts";
import type { ComponentRecord, TimingTally } from "../src/lib/record.ts";
import {
  buildDepthRecord,
  type DepthRunFacts,
  RECORD_FORMAT,
  summaryRows,
} from "../src/models/depth-anything/record.ts";

const environment: RecordEnvironment = {
  recordedAt: "2026-10-28T12:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "D3D12" },
  device: {
    features: [],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2 ** 31,
      maxBufferSize: 2 ** 32,
    },
  },
};

const cached: PipelineSource = {
  kind: "cached",
  ref: { repo: "hdae/karume-depth-anything-v2", revision: "a".repeat(40), hubUrl: "http://x/hf" },
  cacheGeneration: 0,
};

const component: ComponentRecord = {
  runs: 1,
  weightsAllocatedGiB: 0.092,
  submitCount: 1,
  dispatchCount: 300,
  geometryProfile: "default",
  build: {
    partCount: 3,
    supplyWaitMs: 120.5,
    decodeMs: 1,
    bufferCreateMs: 2,
    writeBufferIssueMs: 3,
    uploadFenceMs: 40,
    uploadedBytes: 98_000_000,
  },
  lastRunPipelines: [{ key: "linear:v1", dispatchCount: 48 }],
};

const facts: DepthRunFacts = {
  directory: "models/karume-depth-anything-v2",
  manifestSha256: "f".repeat(64),
  source: cached,
  model: "small",
  defaultModel: "small",
  quant: "f32",
  quantLabel: "f32",
  defaultQuant: "f32",
  image: { name: "cat.png", width: 640, height: 480, rgbSha256: "c".repeat(64) },
  modelInput: { width: 518, height: 518 },
  pipelineLoadMs: 2345.6789,
  estimateMs: 80.12345,
  postprocessMs: 12.3,
  wallMs: 2440.7,
  output: { width: 640, height: 480, range: { min: 0, max: 7.25 } },
  depthSha256: "d".repeat(64),
  diagnostics: component,
  timing: undefined,
};

describe("buildDepthRecord", () => {
  it("starts with the format and the environment, and marks the defaults", () => {
    const record = buildDepthRecord(environment, facts);
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-depth-anything/1");
    expect(record.page).toBe("depth-anything");
    expect(record.adapter).toEqual(environment.adapter);
    expect(record.model).toEqual({ name: "small", isDefault: true });
    expect(record.preset).toEqual({ quant: "f32", label: "f32", isDefault: true });
    expect(buildDepthRecord(environment, { ...facts, quant: "f16" }).preset.isDefault).toBe(false);
  });

  it("keeps the request and the output sizes apart from the model's input size", () => {
    const record = buildDepthRecord(environment, facts);
    expect(record.request).toEqual({
      image: { name: "cat.png", width: 640, height: 480, rgbSha256: "c".repeat(64) },
      modelInput: { width: 518, height: 518 },
    });
    expect(
      buildDepthRecord(environment, { ...facts, modelInput: undefined }).request.modelInput,
    ).toBeNull();
    expect(record.output).toEqual({
      width: 640,
      height: 480,
      min: 0,
      max: 7.25,
      depthSha256: "d".repeat(64),
    });
  });

  it("writes timings as seconds at millisecond precision, and null load for a reused pipeline", () => {
    expect(buildDepthRecord(environment, facts).timings).toEqual({
      pipelineLoadSeconds: 2.346,
      estimateSeconds: 0.08,
      postprocessSeconds: 0.012,
      wallSeconds: 2.441,
    });
    expect(
      buildDepthRecord(environment, { ...facts, pipelineLoadMs: undefined }).timings
        .pipelineLoadSeconds,
    ).toBeNull();
  });

  it("records the depth component's diagnostics, with GPU time only when a run was timed", () => {
    expect(buildDepthRecord(environment, facts).diagnostics).toEqual({ depth: component });
    expect(buildDepthRecord(environment, { ...facts, diagnostics: undefined }).diagnostics).toEqual(
      {},
    );
    const tally: TimingTally = {
      runs: 1,
      clampedNegativeSamples: 0,
      keys: new Map([
        ["linear:v1:a", { ns: 3_000_000, dispatchCount: 48 }],
        ["conv2d:v1:b", { ns: 1_000_000, dispatchCount: 12 }],
      ]),
    };
    const timed = buildDepthRecord(environment, { ...facts, timing: tally }).diagnostics.depth;
    expect(timed?.timing?.totalMs).toBe(4);
    expect(timed?.timing?.ops.map(({ op }) => op)).toEqual(["linear", "conv2d"]);
  });
});

describe("summaryRows", () => {
  const value = (rows: readonly (readonly [string, string])[], label: string) =>
    rows.find(([name]) => name === label)?.[1];

  it("shows the sizes, the timings and the output hash", () => {
    const rows = summaryRows(buildDepthRecord(environment, facts));
    expect(value(rows, "Adapter")).toBe("nvidia · blackwell · D3D12");
    expect(value(rows, "Source")).toBe(`cached (hdae/karume-depth-anything-v2@${"a".repeat(40)})`);
    expect(value(rows, "Input")).toBe("cat.png · 640×480");
    expect(value(rows, "Model input")).toBe("518×518");
    expect(value(rows, "Output")).toBe("640×480");
    expect(value(rows, "Pipeline load")).toBe("2.346 s");
    expect(value(rows, "Depth sha256 (f32)")).toBe("d".repeat(64));
    expect(value(rows, "GPU time")).toBeUndefined();
  });

  it("says when the pipeline was reused instead of showing a load time", () => {
    const rows = summaryRows(
      buildDepthRecord(environment, { ...facts, pipelineLoadMs: undefined }),
    );
    expect(value(rows, "Pipeline load")).toMatch(/reused/);
  });
});
