import type { Sbv2RunComponent, Sbv2Utterance } from "@karume/models/sbv2";
import { describe, expect, it } from "vite-plus/test";

import type { ComponentRecord, TimingTally } from "../src/lib/record.ts";
import {
  buildSbv2Record,
  realTimeFactor,
  type Sbv2RunFacts,
  stageSeconds,
} from "../src/models/sbv2/record.ts";

describe("stageSeconds", () => {
  it("measures each graph from the previous mark, the first from the generate call", () => {
    expect(stageSeconds(1000, { text_encoder: 1500, front: 1750, voice: 3000 })).toEqual({
      text_encoder: 0.5,
      front: 0.25,
      voice: 1.25,
    });
  });

  it("stops at the first graph that has no mark (the run failed there)", () => {
    expect(stageSeconds(0, { text_encoder: 400, voice: 900 })).toEqual({ text_encoder: 0.4 });
  });
});

describe("realTimeFactor", () => {
  it("relates the synthesis seconds to the audio seconds", () => {
    expect(realTimeFactor(0.5, 2)).toBe(0.25);
    expect(realTimeFactor(1, 3)).toBe(0.333);
  });

  it("refuses an empty clip instead of dividing by zero", () => {
    expect(() => realTimeFactor(1, 0)).toThrow(RangeError);
  });
});

/** ame — kana spelled in romaji (any string is a kana to the record). */
const utterance: Sbv2Utterance = {
  leadingPunctuations: [],
  moras: [
    { kana: "a", vowel: "a", tone: 1 },
    { kana: "me", consonant: "m", vowel: "e", tone: 0, punctuations: ["."] },
  ],
  words: [{ surface: "ame", phones: ["a", "m", "e"] }],
};

const componentRecord = (dispatchCount: number): ComponentRecord => ({
  runs: 1,
  weightsAllocatedGiB: 0.1,
  submitCount: 2,
  dispatchCount,
  geometryProfile: "default",
  build: {
    partCount: 1,
    supplyWaitMs: 5,
    decodeMs: 1,
    bufferCreateMs: 1,
    writeBufferIssueMs: 1,
    uploadFenceMs: 1,
    uploadedBytes: 1024,
  },
});

/** One finished run: 44 100 samples at 44.1 kHz (1 s), generate from 1100 ms to 1600 ms. */
const facts = (overrides: Partial<Sbv2RunFacts> = {}): Sbv2RunFacts => ({
  environment: {
    recordedAt: "2026-10-28T00:00:00.000Z",
    userAgent: "test",
    crossOriginIsolated: true,
    adapter: { vendor: "v", architecture: "a", device: "d", description: "x" },
    device: {
      features: ["subgroups"],
      wgslLanguageFeatures: [],
      subgroupLanesLinear: true,
      gpuTimingEnabled: false,
      limits: {
        maxComputeWorkgroupStorageSize: 32_768,
        maxStorageBufferBindingSize: 2 ** 30,
        maxBufferSize: 2 ** 30,
      },
    },
  },
  distribution: { directory: "models/karume-sbv2-jvnv", manifestSha256: "f".repeat(64) },
  source: { kind: "direct" },
  model: "F1",
  preset: { quant: "i8", label: "i8 (int8)", defaultQuant: "i8+bert4" },
  pipelineLoadSeconds: 1.23456,
  request: {
    text: "ame.",
    options: {
      speaker: "jvnv-F1-jp",
      style: "Neutral",
      styleWeight: 1,
      sdpRatio: 0.2,
      noiseScale: 0.6,
      noiseScaleW: 0.8,
      lengthScale: 1,
      seed: 0,
    },
  },
  analyzer: { name: "@hdae/yomi", version: "0.4.2" },
  utterance,
  clock: {
    startedAt: 1000,
    analysedAt: 1100,
    marks: { text_encoder: 1200, front: 1300, voice: 1550 },
    synthesizedAt: 1600,
    finishedAt: 1700,
  },
  audio: { sampleRate: 44_100, samples: 44_100 },
  hashes: { f32Sha256: "1".repeat(64), wavSha256: "2".repeat(64) },
  diagnostics: new Map<Sbv2RunComponent, ComponentRecord>([
    ["text_encoder", componentRecord(10)],
    ["front", componentRecord(20)],
    ["voice", componentRecord(30)],
  ]),
  timings: new Map<Sbv2RunComponent, TimingTally>(),
  ...overrides,
});

describe("buildSbv2Record", () => {
  it("records a direct source by its kind only", () => {
    expect(buildSbv2Record(facts()).source).toEqual({ kind: "direct" });
  });

  it("records a cached source by its coordinates, without the page's rebuild counter", () => {
    const record = buildSbv2Record(
      facts({
        source: {
          kind: "cached",
          ref: { repo: "karume/sbv2-jvnv", revision: "abc", hubUrl: "/hf" },
          cacheGeneration: 3,
        },
      }),
    );
    expect(record.source).toEqual({
      kind: "cached",
      repo: "karume/sbv2-jvnv",
      revision: "abc",
      hubUrl: "/hf",
    });
  });

  it("marks the preset as the default only when it is the model's defaultQuant", () => {
    expect(buildSbv2Record(facts()).preset).toEqual({
      quant: "i8",
      label: "i8 (int8)",
      isDefault: false,
    });
    const chosenDefault = facts({
      preset: { quant: "i8+bert4", label: "i8+bert4", defaultQuant: "i8+bert4" },
    });
    expect(buildSbv2Record(chosenDefault).preset.isDefault).toBe(true);
  });

  it("derives the seconds, the duration and the real-time factor from the clock", () => {
    const record = buildSbv2Record(facts());
    expect(record.timings).toEqual({
      pipelineLoadSeconds: 1.235,
      analysisSeconds: 0.1,
      stageSeconds: { text_encoder: 0.1, front: 0.1, voice: 0.25 },
      synthesisSeconds: 0.5,
      wallSeconds: 0.7,
    });
    expect(record.output).toEqual({
      sampleRate: 44_100,
      samples: 44_100,
      durationSeconds: 1,
      realTimeFactor: 0.5,
      f32Sha256: "1".repeat(64),
      wavSha256: "2".repeat(64),
    });
  });

  it("keeps the request, the analyser and the exact utterance with its bracketed reading", () => {
    const record = buildSbv2Record(facts());
    expect(record.format).toBe("karume-playground-sbv2/1");
    expect(record.page).toBe("sbv2");
    expect(record.request.options.seed).toBe(0);
    expect(record.analysis).toEqual({
      analyzer: "@hdae/yomi",
      analyzerVersion: "0.4.2",
      reading: "[a]me.",
      utterance,
    });
  });

  it("adds GPU time only to the graphs that were timed", () => {
    const tally: TimingTally = {
      runs: 1,
      clampedNegativeSamples: 0,
      keys: new Map([["matmul:v1:x", { ns: 2_000_000, dispatchCount: 4 }]]),
    };
    const record = buildSbv2Record(facts({ timings: new Map([["voice", tally]]) }));
    expect(record.diagnostics.text_encoder?.timing).toBeUndefined();
    expect(record.diagnostics.front?.timing).toBeUndefined();
    expect(record.diagnostics.voice?.timing).toMatchObject({
      runs: 1,
      totalMs: 2,
      dispatchCount: 4,
      ops: [{ op: "matmul", ms: 2, dispatchCount: 4 }],
    });
    expect(record.diagnostics.voice?.dispatchCount).toBe(30);
  });

  it("records only the graphs that reported (none of them is invented)", () => {
    const record = buildSbv2Record(
      facts({ diagnostics: new Map([["text_encoder", componentRecord(10)]]) }),
    );
    expect(Object.keys(record.diagnostics)).toEqual(["text_encoder"]);
  });
});
