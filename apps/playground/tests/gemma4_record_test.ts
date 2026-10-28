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

  it("starts with the format and the environment and names the entry point by mount", () => {
    const record = buildGemmaRecord({ ...input, mount: "gemma4" });
    expect(record.format).toBe(RECORD_FORMAT);
    expect(record.format).toBe("karume-playground-gemma4/1");
    expect(record.page).toBe("gemma4");
    expect(record.recordedAt).toBe(environment.recordedAt);
    expect(record.pipeline).toBe("gemma.Pipeline");
    expect(buildGemmaRecord({ ...input, mount: "gemma4-qat" }).pipeline).toBe("Gemma4QatPipeline");
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
