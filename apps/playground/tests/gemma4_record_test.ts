import type { ChatStop } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import {
  buildGemmaRecord,
  NO_RUNS,
  RECORD_FORMAT,
  stopRecordOf,
  type TurnRecord,
} from "../src/models/gemma4/chat-record.ts";
import type { RecordEnvironment } from "../src/lib/environment.ts";
import { foldComponentDiagnostics } from "../src/lib/record.ts";

const environment: RecordEnvironment = {
  recordedAt: "2026-10-28T00:00:00.000Z",
  userAgent: "test",
  crossOriginIsolated: true,
  adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
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

const turn: TurnRecord = {
  index: 1,
  userText: "Hi",
  replyText: "Hello.",
  request: { maxNewTokens: 64, sampler: { temperature: 0, seed: 0 } },
  outcome: { kind: "stopped", stop: { reason: "eos", tokens: 3, token: 106 } },
  prefill: { promptTokens: 12, carriedPositions: 0, chunks: 1 },
  timing: { totalSeconds: 0.5, decodeTokens: 2 },
  generatedTokenIds: [9259, 236761],
  droppedEntries: 0,
  inHistory: true,
  kvUsedAfter: 15,
  runs: { ...NO_RUNS, prefill: 1, decode: 2 },
};

describe("stopRecordOf", () => {
  it("keeps the stop token of an EOS / stop-token stop", () => {
    const stop: ChatStop = { reason: "eos", token: 106, tokens: 3 };
    expect(stopRecordOf(stop)).toEqual({ reason: "eos", tokens: 3, token: 106 });
  });

  it("keeps the matched stop string and the speculation counters", () => {
    const speculation = {
      cycles: 2,
      draftRuns: 6,
      drafted: 6,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2, 0],
    };
    const stop: ChatStop = {
      reason: "stop-string",
      stopString: "END",
      tokens: 7,
      speculation,
    };
    expect(stopRecordOf(stop)).toEqual({
      reason: "stop-string",
      tokens: 7,
      stopString: "END",
      speculation,
    });
  });

  it("has no token for a max-tokens or aborted stop", () => {
    expect(stopRecordOf({ reason: "max-tokens", tokens: 64 })).toEqual({
      reason: "max-tokens",
      tokens: 64,
    });
    expect(stopRecordOf({ reason: "aborted", tokens: 5 })).toEqual({
      reason: "aborted",
      tokens: 5,
    });
  });

  it("keeps the decode run-ahead counters of a run-ahead turn, and has none otherwise", () => {
    const decodeAhead = { depth: 4, batches: 3, steps: 11 };
    expect(stopRecordOf({ reason: "eos", token: 106, tokens: 10, decodeAhead })).toEqual({
      reason: "eos",
      tokens: 10,
      token: 106,
      decodeAhead,
    });
    expect(Object.hasOwn(stopRecordOf({ reason: "max-tokens", tokens: 4 }), "decodeAhead")).toBe(
      false,
    );
  });
});

describe("buildGemmaRecord", () => {
  const input = {
    environment,
    directory: "models/karume-gemma4",
    manifestSha256: "f".repeat(64),
    source: { kind: "direct" } as const,
    model: "e2b",
    preset: { quant: "i4-fast", label: "Packed int4", isDefault: true },
    speculative: false,
    loadSeconds: 12.345,
    chat: { capacity: 4096, overflow: "drop-oldest" as const },
    turns: [turn],
    diagnostics: undefined,
  };

  it("names f16 full slots only when they were asked for", () => {
    expect(buildGemmaRecord({ ...input, mount: "gemma4" })).not.toHaveProperty("fullStateStorage");
    expect(
      buildGemmaRecord({ ...input, mount: "gemma4", fullStateStorage: "f16" }).fullStateStorage,
    ).toBe("f16");
  });

  it("names the expert cache and its latest report only when the pipeline was built with it", () => {
    const without = buildGemmaRecord({ ...input, mount: "gemma4-qat", model: "26b" });
    expect(without).not.toHaveProperty("expertCache");
    expect(without).not.toHaveProperty("expertCacheReport");
    const expertCacheReport = {
      runs: 12,
      missedRuns: 4,
      loads: 4,
      loadedExperts: 96,
      evictedExperts: 32,
      hostExperts: 90,
      containerExperts: 6,
      bytesRead: 6 * 3_345_408,
      bytesUploaded: 96 * 3_345_408,
      loadMs: 210,
      readMs: 25,
      uploadMs: 50,
      hostHeldExperts: 3840,
      hostHeldBytes: 3840 * 3_345_408,
      slots: 64,
      committedRuns: 8,
      retries: 4,
    };
    const record = buildGemmaRecord({
      ...input,
      mount: "gemma4-qat",
      model: "26b",
      expertCache: { slots: 64 },
      expertCacheReport,
    });
    expect(record.expertCache).toEqual({ slots: 64 });
    expect(record.expertCacheReport).toEqual(expertCacheReport);
  });

  it("keeps the host tier's budget inside the expert cache, through JSON", () => {
    const record = buildGemmaRecord({
      ...input,
      mount: "gemma4-qat",
      model: "26b",
      expertCache: { slots: 32, hostBytes: Number.MAX_SAFE_INTEGER },
    });
    const saved: unknown = JSON.parse(JSON.stringify(record));
    expect(saved).toMatchObject({ expertCache: { slots: 32, hostBytes: Number.MAX_SAFE_INTEGER } });
  });

  it("keeps the self-speculation depth and the cycles' counters through JSON", () => {
    const expertCacheReport = {
      runs: 50,
      missedRuns: 6,
      loads: 12,
      loadedExperts: 140,
      evictedExperts: 76,
      hostExperts: 140,
      containerExperts: 0,
      bytesRead: 0,
      bytesUploaded: 140 * 3_345_408,
      loadMs: 300,
      readMs: 0,
      uploadMs: 120,
      hostHeldExperts: 3840,
      hostHeldBytes: 3840 * 3_345_408,
      slots: 64,
      committedRuns: 3,
      retries: 1,
      speculation: {
        depth: 4,
        cycles: 10,
        draftRuns: 34,
        acceptedDrafts: 31,
        verifyRetries: 2,
        partialCommits: 3,
      },
    };
    const record = buildGemmaRecord({
      ...input,
      mount: "gemma4-qat",
      model: "26b",
      expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER, depth: 4 },
      expertCacheReport,
    });
    const saved: unknown = JSON.parse(JSON.stringify(record));
    expect(saved).toMatchObject({
      expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER, depth: 4 },
      expertCacheReport,
    });
  });

  it("starts with the format and the environment and names the entry point by mount", () => {
    const record = buildGemmaRecord({ ...input, mount: "gemma4" });
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-gemma4/4");
    expect(record.page).toBe("gemma4");
    expect(record.recordedAt).toBe(environment.recordedAt);
    expect(record.pipeline).toBe("gemma.Pipeline");
    expect(buildGemmaRecord({ ...input, mount: "gemma4-qat-mobile" }).pipeline).toBe(
      "gemma4QatMobile.Pipeline",
    );
  });

  it("keeps the turns with their token ids", () => {
    const record = buildGemmaRecord({ ...input, mount: "gemma4" });
    expect(record.turns[0]?.generatedTokenIds).toEqual([9259, 236761]);
    expect(record.distribution).toEqual({
      mount: "gemma4",
      directory: "models/karume-gemma4",
      manifestSha256: "f".repeat(64),
    });
  });

  it("says whether diagnostics were collected and carries them only then", () => {
    const without = buildGemmaRecord({ ...input, mount: "gemma4" });
    expect(without.diagnosticsCollected).toBe(false);
    expect("diagnostics" in without).toBe(false);
    const target = foldComponentDiagnostics(undefined, {
      weights: { allocatedBytes: 2 ** 30 },
      submit: { submitCount: 1, dispatchCount: 30 },
      geometryProfile: "nvidia-blackwell",
      buildStats: {
        partCount: 1,
        supplyWaitMs: 0,
        decodeMs: 0,
        bufferCreateMs: 0,
        writeBufferIssueMs: 0,
        uploadFenceMs: 0,
        uploadedBytes: 0,
      },
      lastRunPipelines: undefined,
    });
    const withDiagnostics = buildGemmaRecord({
      ...input,
      mount: "gemma4",
      diagnostics: { target },
    });
    expect(withDiagnostics.diagnosticsCollected).toBe(true);
    expect(withDiagnostics.diagnostics).toEqual({ target });
  });
});
