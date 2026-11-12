import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import type { ComponentRecord } from "../src/lib/record.ts";
import {
  assembleRecord,
  buildsOfThisRun,
  carriedOverAfter,
  RECORD_FORMAT,
  residentAfter,
  type RunObservation,
} from "../src/models/anima/record.ts";

const environment: RecordEnvironment = {
  recordedAt: "2026-10-28T12:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "driver" },
  device: {
    features: ["shader-f16"],
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

const transformerRecord: ComponentRecord = {
  runs: 8,
  weightsAllocatedGiB: 1.9,
  submitCount: 3,
  dispatchCount: 900,
  geometryProfile: "default",
  build: {
    partCount: 11,
    supplyWaitMs: 1200,
    decodeMs: 10,
    bufferCreateMs: 5,
    writeBufferIssueMs: 40,
    uploadFenceMs: 300,
    uploadedBytes: 2 ** 31,
  },
};

const observation = (patch: Partial<RunObservation> = {}): RunObservation => ({
  environment,
  mount: "anima",
  directory: "models/karume-anima",
  manifestSha256: "f".repeat(64),
  source: { kind: "direct" },
  model: "anima-turbo-v1.1",
  defaultModel: "anima-turbo-v1.1",
  quant: "bf16",
  quantLabel: "Original weights (bf16)",
  defaultQuant: "f16+dit8-a8-attn8-s16",
  request: { prompt: "1girl", seed: 42, residency: "per-stage" },
  effective: {
    steps: 8,
    guidanceScale: 1,
    resolution: "1024x1024",
    sampler: "euler",
    negativePrompt: null,
  },
  livePreview: true,
  openMilliseconds: 2500,
  residencyEvents: [],
  residentAtStart: false,
  transformerCarriedOver: false,
  stages: {
    text_encoder: { start: 0, end: 1000 },
    text_conditioner: { start: 1000, end: 1100 },
    transformer: { start: 1100, end: 9100 },
    vae_decoder: { start: 9100, end: 10600 },
  },
  stepMarks: [3100, 4100, 5100],
  tileMarks: [9600, 10100],
  generateMilliseconds: 10650,
  output: { width: 1024, height: 1024, sha256: "0".repeat(64) },
  diagnostics: new Map([["transformer", transformerRecord]]),
  ...patch,
});

describe("residentAfter", () => {
  it("holds a DiT after retained and none after released or evicted", () => {
    expect(residentAfter("retained")).toBe(true);
    expect(residentAfter("released")).toBe(false);
    expect(residentAfter("evicted")).toBe(false);
  });
});

describe("carriedOverAfter", () => {
  it("stays carried over through a retain or an eviction after the DiT stage", () => {
    expect(carriedOverAfter(true, "retained", true)).toBe(true);
    expect(carriedOverAfter(true, "evicted", true)).toBe(true);
    expect(carriedOverAfter(true, "released", false)).toBe(true);
  });

  it("ends when the resident DiT is evicted before the DiT stage ends", () => {
    expect(carriedOverAfter(true, "evicted", false)).toBe(false);
  });

  it("never starts during a run that began without a resident DiT", () => {
    expect(carriedOverAfter(false, "retained", false)).toBe(false);
    expect(carriedOverAfter(false, "retained", true)).toBe(false);
  });
});

describe("assembleRecord", () => {
  it("starts with the format, the page and the environment", () => {
    const record = assembleRecord(observation());
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-anima/1");
    expect(record.page).toBe("anima");
    expect(record.adapter).toEqual(environment.adapter);
    expect(record.recordedAt).toBe(environment.recordedAt);
  });

  it("marks whether the model and the preset are the manifest's defaults", () => {
    const record = assembleRecord(observation());
    expect(record.model).toEqual({ name: "anima-turbo-v1.1", isDefault: true });
    expect(record.preset).toEqual({
      quant: "bf16",
      label: "Original weights (bf16)",
      isDefault: false,
    });
  });

  it("times every stage and every step and tile from its stage start", () => {
    const record = assembleRecord(observation());
    expect(record.stageSeconds).toEqual({
      text_encoder: 1,
      text_conditioner: 0.1,
      transformer: 8,
      vae_decoder: 1.5,
    });
    expect(record.timeline).toEqual({
      denoiseStepSeconds: [2, 1, 1],
      vaeTileSeconds: [0.5, 0.5],
    });
    expect(record.generateSeconds).toBe(10.65);
  });

  it("has an empty timeline for a stage that never started", () => {
    const record = assembleRecord(observation({ stages: {}, stepMarks: [], tileMarks: [] }));
    expect(record.timeline).toEqual({ denoiseStepSeconds: [], vaeTileSeconds: [] });
  });

  it("tells a reused pipeline from one opened by this run", () => {
    expect(assembleRecord(observation()).pipeline).toEqual({ reused: false, openSeconds: 2.5 });
    expect(assembleRecord(observation({ openMilliseconds: undefined })).pipeline).toEqual({
      reused: true,
      openSeconds: null,
    });
  });

  it("keeps the requested residency and the residency events in order", () => {
    const record = assembleRecord(
      observation({
        request: { prompt: "1girl", seed: 1, residency: "transformer" },
        residencyEvents: [
          { action: "evicted", reason: "headroom" },
          { action: "released", reason: "downgraded" },
        ],
      }),
    );
    expect(record.residency).toEqual({
      requested: "transformer",
      events: [
        { action: "evicted", reason: "headroom" },
        { action: "released", reason: "downgraded" },
      ],
      residentAtStart: false,
      transformerBuildCarriedOver: false,
    });
  });

  it("marks a DiT carried over from an earlier run", () => {
    const record = assembleRecord(
      observation({
        request: { prompt: "1girl", seed: 1, residency: "transformer" },
        residencyEvents: [{ action: "retained", reason: "request" }],
        residentAtStart: true,
        transformerCarriedOver: true,
      }),
    );
    expect(record.residency.residentAtStart).toBe(true);
    expect(record.residency.transformerBuildCarriedOver).toBe(true);
  });

  it("keeps the source, the output hash and the diagnostics per component", () => {
    const record = assembleRecord(
      observation({
        source: {
          kind: "cached",
          repo: "hdae/karume-anima",
          revision: "b".repeat(40),
          hubUrl: "http://127.0.0.1:8790/hf",
        },
      }),
    );
    expect(record.source).toEqual({
      kind: "cached",
      repo: "hdae/karume-anima",
      revision: "b".repeat(40),
      hubUrl: "http://127.0.0.1:8790/hf",
    });
    expect(record.output).toEqual({ width: 1024, height: 1024, rgbaSha256: "0".repeat(64) });
    expect(record.diagnostics).toEqual({ transformer: transformerRecord });
  });

  it("survives a JSON round trip unchanged (what Download and Copy save)", () => {
    const record = assembleRecord(observation());
    expect(JSON.parse(JSON.stringify(record))).toEqual(record);
  });
});

describe("buildsOfThisRun", () => {
  const vaeRecord: ComponentRecord = {
    ...transformerRecord,
    runs: 4,
    build: { ...transformerRecord.build, uploadedBytes: 2 ** 27, supplyWaitMs: 50 },
  };
  const diagnostics = new Map([
    ["transformer", transformerRecord],
    ["vae_decoder", vaeRecord],
  ]);

  it("sums every component's build when the DiT was built in this run", () => {
    const record = assembleRecord(observation({ diagnostics }));
    expect(buildsOfThisRun(record)).toEqual([transformerRecord.build, vaeRecord.build]);
  });

  it("leaves out the build of a DiT carried over from an earlier run", () => {
    const record = assembleRecord(
      observation({ diagnostics, residentAtStart: true, transformerCarriedOver: true }),
    );
    expect(buildsOfThisRun(record)).toEqual([vaeRecord.build]);
  });

  it("keeps the DiT's build when the resident DiT was evicted and rebuilt", () => {
    const record = assembleRecord(
      observation({ diagnostics, residentAtStart: true, transformerCarriedOver: false }),
    );
    expect(buildsOfThisRun(record)).toEqual([transformerRecord.build, vaeRecord.build]);
  });
});

describe("assembleRecord: the experimental fusion rule", () => {
  it("names the rule the run's pipeline was opened with", () => {
    const record = assembleRecord(observation({ fusionRule: "candidate/vae_rms_silu_g8" }));
    expect(record.experimentalFusionRule).toBe("candidate/vae_rms_silu_g8");
  });

  it("is null when no rule ran (left out or undefined)", () => {
    expect(assembleRecord(observation()).experimentalFusionRule).toBeNull();
    expect(
      assembleRecord(observation({ fusionRule: undefined })).experimentalFusionRule,
    ).toBeNull();
  });
});
