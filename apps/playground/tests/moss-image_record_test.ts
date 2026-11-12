import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import {
  assembleRecord,
  RECORD_FORMAT,
  type RunObservation,
} from "../src/models/moss-image/record.ts";

const environment: RecordEnvironment = {
  recordedAt: "2026-11-01T12:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "driver" },
  device: {
    features: ["subgroups"],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: true,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBufferBindingSize: 2 ** 31,
      maxBufferSize: 2 ** 32,
    },
  },
};

const observation = (patch: Partial<RunObservation> = {}): RunObservation => ({
  environment,
  directory: "models/karume-moss-image",
  manifestSha256: "f".repeat(64),
  source: { kind: "direct" },
  model: "sft",
  defaultModel: "sft",
  quant: "f32+dit8",
  quantLabel: "int8 transformer, f32 compute",
  defaultQuant: "f32+dit8-a8-attn8-online",
  request: { prompt: "a red fox", width: 512, height: 512, seed: 42 },
  effective: {
    width: 512,
    height: 512,
    steps: 50,
    guidance: 3,
    shift: 5,
    negativePrompt: "",
  },
  openMilliseconds: 1250,
  generateMilliseconds: 95_432.1,
  output: { width: 512, height: 512, sha256: "0".repeat(64) },
  ...patch,
});

describe("assembleRecord", () => {
  it("starts with the format, the page and the environment", () => {
    const record = assembleRecord(observation());
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-moss-image/1");
    expect(record.page).toBe("moss-image");
    expect(record.adapter).toEqual(environment.adapter);
    expect(record.device).toEqual(environment.device);
    expect(record.recordedAt).toBe(environment.recordedAt);
  });

  it("names the distribution and the direct route it was read over", () => {
    const record = assembleRecord(observation());
    expect(record.distribution).toEqual({
      mount: "moss-image",
      directory: "models/karume-moss-image",
      manifestSha256: "f".repeat(64),
    });
    expect(record.source).toEqual({ kind: "direct" });
  });

  it("keeps the cached route's repo reference and leaves the cache generation out", () => {
    const record = assembleRecord(
      observation({
        source: {
          kind: "cached",
          ref: { repo: "hdae/karume-moss-image", revision: "0".repeat(40), hubUrl: "http://x/hf" },
          cacheGeneration: 2,
        },
      }),
    );
    expect(record.source).toEqual({
      kind: "cached",
      repo: "hdae/karume-moss-image",
      revision: "0".repeat(40),
      hubUrl: "http://x/hf",
    });
  });

  it("marks whether the model and the preset are the manifest's defaults", () => {
    const record = assembleRecord(observation());
    expect(record.model).toEqual({ name: "sft", isDefault: true });
    expect(record.preset).toEqual({
      quant: "f32+dit8",
      label: "int8 transformer, f32 compute",
      isDefault: false,
    });
    const online = assembleRecord(observation({ quant: "f32+dit8-a8-attn8-online" }));
    expect(online.preset.isDefault).toBe(true);
  });

  it("keeps the request as sent and the values the run used", () => {
    const record = assembleRecord(observation());
    expect(record.request).toEqual({ prompt: "a red fox", width: 512, height: 512, seed: 42 });
    expect(record.effective).toEqual({
      width: 512,
      height: 512,
      steps: 50,
      guidance: 3,
      shift: 5,
      negativePrompt: "",
    });
  });

  it("times the generation and tells a reused pipeline from one opened by this run", () => {
    const record = assembleRecord(observation());
    expect(record.generateSeconds).toBe(95.432);
    expect(record.pipeline).toEqual({ reused: false, openSeconds: 1.25 });
    expect(assembleRecord(observation({ openMilliseconds: undefined })).pipeline).toEqual({
      reused: true,
      openSeconds: null,
    });
  });

  it("keeps the output's size and the sha256 of its RGBA bytes", () => {
    const record = assembleRecord(
      observation({ output: { width: 768, height: 512, sha256: "1".repeat(64) } }),
    );
    expect(record.output).toEqual({ width: 768, height: 512, rgbaSha256: "1".repeat(64) });
  });

  it("survives a JSON round trip unchanged (what Download and Copy save)", () => {
    const record = assembleRecord(observation());
    expect(JSON.parse(JSON.stringify(record))).toEqual(record);
  });
});

describe("assembleRecord: the experimental fusion rule", () => {
  it("names the rule the run's pipeline was opened with", () => {
    const record = assembleRecord(observation({ fusionRule: "candidate/ew_chain" }));
    expect(record.experimentalFusionRule).toBe("candidate/ew_chain");
  });

  it("is null when no rule ran (left out or undefined)", () => {
    expect(assembleRecord(observation()).experimentalFusionRule).toBeNull();
    expect(
      assembleRecord(observation({ fusionRule: undefined })).experimentalFusionRule,
    ).toBeNull();
  });
});
