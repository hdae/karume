import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { type ComponentRecord, f32Sha256 } from "../src/lib/record.ts";
import {
  assembleRecord,
  linearDispatchesOf,
  RECORD_FORMAT,
  realTimeFactor,
  recordRequest,
  type RecordInput,
} from "../src/models/irodori/record.ts";

const environment: RecordEnvironment = {
  recordedAt: "2026-10-28T00:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "v", architecture: "a", device: "", description: "d" },
  device: {
    features: [],
    wgslLanguageFeatures: [],
    subgroupLanesLinear: false,
    gpuTimingEnabled: false,
    limits: {
      maxComputeWorkgroupStorageSize: 1,
      maxStorageBufferBindingSize: 2,
      maxBufferSize: 3,
    },
  },
};

describe("realTimeFactor", () => {
  it("is synthesis seconds per audio second", () => {
    expect(realTimeFactor(2, 8)).toBe(0.25);
  });

  it("is null without audio", () => {
    expect(realTimeFactor(2, 0)).toBeNull();
  });
});

describe("linearDispatchesOf", () => {
  it("counts the int8-activation linears apart from every other linear, and ignores other ops", () => {
    expect(
      linearDispatchesOf([
        { key: "linear:v4:i8a8:g1:dp4a", dispatchCount: 30 },
        { key: "linear:v5:i8a8:g1:dp4a", dispatchCount: 2 },
        { key: "linear:v3:f32:t64", dispatchCount: 5 },
        { key: "linear_gemv:v1:f32:c4u2", dispatchCount: 1 },
        { key: "attention_online:v2:i8a8:d64", dispatchCount: 12 },
        { key: "quantize_rows:v1", dispatchCount: 32 },
      ]),
    ).toEqual({ i8a8: 32, other: 6 });
  });

  it("is zero for a run without linears", () => {
    expect(linearDispatchesOf([])).toEqual({ i8a8: 0, other: 0 });
  });
});

describe("recordRequest", () => {
  it("names a caption request, the defaults spelled out", () => {
    expect(recordRequest({ text: "t", caption: "c" }, undefined, 40)).toEqual({
      text: "t",
      voice: { kind: "caption", caption: "c" },
      seed: 0,
      durationSeconds: "predicted",
      codecTileFrames: "default",
      steps: 40,
    });
  });

  it("names a request without voice design by an empty caption", () => {
    expect(recordRequest({ text: "t", seed: 3 }, undefined, 40).voice).toEqual({
      kind: "caption",
      caption: "",
    });
  });

  it("names a reference request by its file, decoder, length and sample hash", () => {
    const data = new Float32Array(24_000);
    const voice = recordRequest(
      { text: "t", speaker: { audio: { data, sampleRate: 48000 } }, durationSeconds: 2 },
      { file: "me.wav", decoder: { kind: "decodeWav" }, f32Sha256: "ab" },
      40,
    );
    expect(voice.voice).toEqual({
      kind: "reference",
      file: "me.wav",
      decoder: { kind: "decodeWav" },
      sampleRate: 48000,
      samples: 24_000,
      seconds: 0.5,
      f32Sha256: "ab",
    });
    expect(voice.durationSeconds).toBe(2);
  });

  it("refuses a reference request without the file's record", () => {
    const audio = { data: new Float32Array(1), sampleRate: 48000 };
    expect(() => recordRequest({ text: "t", speaker: { audio } }, undefined, 40)).toThrow();
  });
});

describe("assembleRecord", () => {
  const input = (patch: Partial<RecordInput> = {}): RecordInput => ({
    environment,
    distribution: { mount: "irodori-v4.1", directory: "models/x", manifestSha256: "m" },
    source: { kind: "direct" },
    model: "v4.1-small",
    preset: { quant: "i8-a8", label: "Balanced", isDefault: true },
    linearCompute: "preset",
    linearComputeResolved: "a8",
    request: recordRequest({ text: "t", caption: "c", seed: 1 }, undefined, 40),
    ditLoop: "host",
    stageEvents: true,
    pipeline: { built: true, loadSeconds: 1.2 },
    diagnostics: {},
    stages: { dit: { start: 1000, end: 3000 } },
    stepMarks: [1500, 2000, 2500],
    runStart: 0,
    generateStart: 500,
    generateEnd: 4500,
    audio: {
      data: new Float32Array(96_000),
      sampleRate: 48000,
      frames: 50,
      forwards: 160,
      seed: 1,
    },
    ...patch,
  });

  it("keeps the format, the environment, the timings and the output facts", async () => {
    const record = await assembleRecord(input());
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-irodori/1");
    expect(record.page).toBe("irodori");
    expect(record.recordedAt).toBe(environment.recordedAt);
    expect(record.stageSeconds).toEqual({ dit: 2 });
    expect(record.timeline.denoiseStepSeconds).toEqual([0.5, 0.5, 0.5]);
    expect(record.generateSeconds).toBe(4);
    expect(record.wallSeconds).toBe(4.5);
    expect(record.output).toMatchObject({
      sampleRate: 48000,
      samples: 96_000,
      seconds: 2,
      latentFrames: 50,
      forwards: 160,
      seed: 1,
      realTimeFactor: 2,
    });
    expect(record.output.f32Sha256).toBe(await f32Sha256(new Float32Array(96_000)));
  });

  it("records no stages and no step timeline for a run without stage events", async () => {
    const record = await assembleRecord(
      input({ stages: {}, stepMarks: [], stageEvents: false, ditLoop: "resident" }),
    );
    expect(record.stageSeconds).toEqual({});
    expect(record.timeline.denoiseStepSeconds).toEqual([]);
  });

  it("keeps the asked-for linearCompute and the dit's observed linear dispatches", async () => {
    const withoutDit = await assembleRecord(input());
    expect(withoutDit.linearComputeResolved).toBe("a8");
    expect(withoutDit.ditLinearDispatches).toBeNull();
    const dit = {
      runs: 40,
      weightsAllocatedGiB: 0.1,
      submitCount: 1,
      dispatchCount: 3,
      geometryProfile: "default",
      build: {
        partCount: 1,
        supplyWaitMs: 0,
        decodeMs: 0,
        bufferCreateMs: 0,
        writeBufferIssueMs: 0,
        uploadFenceMs: 0,
        uploadedBytes: 0,
      },
      lastRunPipelines: [{ key: "linear:v3:f32:t64", dispatchCount: 3 }],
    } satisfies ComponentRecord;
    const floatWeights = await assembleRecord(input({ diagnostics: { dit } }));
    expect(floatWeights.ditLinearDispatches).toEqual({ i8a8: 0, other: 3 });
  });

  it("records a null seed when the API reports none", async () => {
    const record = await assembleRecord(
      input({
        audio: { data: new Float32Array(10), sampleRate: 48000, frames: 1, forwards: 1 },
      }),
    );
    expect(record.output.seed).toBeNull();
  });
});
