import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  assembleRecord,
  type MatteReport,
  RECORD_FORMAT,
  type RecordInput,
} from "../src/models/birefnet/record.ts";

const hex = (character: string): string => character.repeat(64);

/** `1024` is the default model; its default preset `f32` is labelled, `f16` is not. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "1024",
    models: Object.fromEntries(
      ["1024", "2048"].map((name) => [
        name,
        {
          pipeline: "birefnet/2",
          assets: {},
          weights: {
            matte: Object.fromEntries(
              ["f32", "f16"].map((dtype) => [
                dtype,
                {
                  container: {
                    descriptor: {
                      graph: { length: 8, sha256: hex("d") },
                      model: { length: 8, sha256: hex("e") },
                    },
                    parts: [
                      {
                        path: `${name}/matte/model.${dtype}-00001-of-00002.krm`,
                        // container-v1 §8: header 24 + graph 8 + model descriptor 8.
                        size: 40,
                        sha256: hex(dtype === "f32" ? "a" : "b"),
                      },
                      {
                        path: `${name}/matte/model.${dtype}-00002-of-00002.krm`,
                        size: 10,
                        sha256: hex(dtype === "f32" ? "c" : "f"),
                      },
                    ],
                  },
                },
              ]),
            ),
          },
          quants: {
            f32: { weights: { matte: "f32" }, session: {}, label: "Full precision (f32)" },
            f16: { weights: { matte: "f16" }, session: {} },
          },
          defaultQuant: "f32",
          pipelineConfig: {},
        },
      ]),
    ),
  }),
);

const environment: RecordInput["environment"] = {
  recordedAt: "2026-10-28T00:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "driver" },
  device: {
    features: [],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32_768,
      maxStorageBufferBindingSize: 2 ** 30,
      maxBufferSize: 2 ** 31,
    },
  },
};

const report = (timed: boolean): MatteReport => ({
  weights: { allocatedBytes: 2 ** 29 },
  submit: { submitCount: 3, dispatchCount: 2500 },
  geometryProfile: "nvidia-ampere",
  buildStats: {
    partCount: 4,
    supplyWaitMs: 1234.56,
    decodeMs: 0,
    bufferCreateMs: 1,
    writeBufferIssueMs: 2,
    uploadFenceMs: 300.04,
    uploadedBytes: 2 ** 29,
  },
  lastRunPipelines: [{ key: "conv2d:v1:a", dispatchCount: 2500 }],
  lastRunTiming: timed
    ? {
        entries: [{ key: "conv2d:v1:a", ns: 12_345_678, dispatchCount: 2500, workgroupCount: 1 }],
        totalNs: 12_345_678,
        dispatchCount: 2500,
        clampedNegativeSamples: 0,
      }
    : undefined,
});

const input = (patch: Partial<RecordInput> = {}): RecordInput => ({
  environment,
  checkpoint: "lucida",
  directory: "models/karume-lucida",
  manifest,
  manifestSha256: hex("9"),
  source: { kind: "direct" },
  selection: { model: "1024", quant: "f32" },
  image: { name: "cat.png", width: 640, height: 480, rgbSha256: hex("1") },
  pipeline: { reused: false, loadMs: 12_345.6789, segmentRun: 1 },
  segmentMs: 1_400.4444,
  wallMs: 13_800.1234,
  matte: { width: 640, height: 480, sha256: hex("2") },
  report: report(false),
  ...patch,
});

describe("assembleRecord", () => {
  it("opens with the format and the page, then the environment", () => {
    const record = assembleRecord(input());
    expect(record.format).toBe(RECORD_FORMAT);
    expect(RECORD_FORMAT).toBe("karume-playground-birefnet/1");
    expect(record.page).toBe("birefnet");
    expect(record.adapter.vendor).toBe("nvidia");
    expect(record.device.limits.maxStorageBufferBindingSize).toBe(2 ** 30);
  });

  it("names the distribution, the input and the matte by sha256", () => {
    const record = assembleRecord(input());
    expect(record.distribution).toEqual({
      checkpoint: "lucida",
      directory: "models/karume-lucida",
      manifestSha256: hex("9"),
    });
    expect(record.request.image.rgbSha256).toBe(hex("1"));
    expect(record.output).toEqual({ width: 640, height: 480, matteSha256: hex("2") });
  });

  it("marks the default model and preset and takes the preset's label", () => {
    const record = assembleRecord(input());
    expect(record.model).toEqual({ name: "1024", isDefault: true });
    expect(record.preset).toEqual({
      quant: "f32",
      label: "Full precision (f32)",
      isDefault: true,
    });
  });

  it("falls back to the preset's name for an unlabelled preset, not marked default", () => {
    const record = assembleRecord(input({ selection: { model: "2048", quant: "f16" } }));
    expect(record.model).toEqual({ name: "2048", isDefault: false });
    expect(record.preset).toEqual({ quant: "f16", label: "f16", isDefault: false });
  });

  it("keeps the cached route's repo and revision, and nothing for direct", () => {
    const ref = { repo: "hdae/karume-lucida", revision: hex("3").slice(0, 40), hubUrl: "x/hf" };
    expect(
      assembleRecord(input({ source: { kind: "cached", ref, cacheGeneration: 2 } })).source,
    ).toEqual({ kind: "cached", ...ref });
    expect(assembleRecord(input()).source).toEqual({ kind: "direct" });
  });

  it("gives the times in seconds at millisecond precision", () => {
    const record = assembleRecord(input());
    expect(record.pipeline).toEqual({ reused: false, loadSeconds: 12.346, segmentRun: 1 });
    expect(record.timings).toEqual({ segmentSeconds: 1.4, wallSeconds: 13.8 });
  });

  it("keeps the matte Session's report, without GPU time on an untimed run", () => {
    const { diagnostics } = assembleRecord(input());
    expect(Object.keys(diagnostics)).toEqual(["matte"]);
    expect(diagnostics["matte"]).toMatchObject({
      runs: 1,
      weightsAllocatedGiB: 0.5,
      submitCount: 3,
      dispatchCount: 2500,
      geometryProfile: "nvidia-ampere",
      build: { supplyWaitMs: 1234.6, uploadFenceMs: 300 },
      lastRunPipelines: [{ key: "conv2d:v1:a", dispatchCount: 2500 }],
    });
    expect(diagnostics["matte"]).not.toHaveProperty("timing");
  });

  it("adds the GPU time when the run was timed", () => {
    const { diagnostics } = assembleRecord(input({ report: report(true) }));
    expect(diagnostics["matte"]?.timing).toMatchObject({
      runs: 1,
      totalMs: 12.3,
      ops: [{ op: "conv2d", ms: 12.3, dispatchCount: 2500 }],
    });
  });

  it("has no diagnostics when the run reported none", () => {
    expect(assembleRecord(input({ report: undefined })).diagnostics).toEqual({});
  });

  it("fails loudly for a model or preset the manifest does not have", () => {
    expect(() => assembleRecord(input({ selection: { model: "512", quant: "f32" } }))).toThrow(
      /no model '512'/,
    );
    expect(() => assembleRecord(input({ selection: { model: "1024", quant: "i8" } }))).toThrow(
      /no preset 'i8'/,
    );
  });
});
