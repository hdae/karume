import type { Manifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import type { RecordEnvironment } from "../src/lib/environment.ts";
import { type ArmDraft, armLabel, resolveArm } from "../src/models/gemma4-bench/plan.ts";
import { buildBenchRecord } from "../src/models/gemma4-bench/record.ts";
import type { RunArm } from "../src/models/gemma4-bench/run.ts";

/** A QAT manifest with one preset (`i4`, the default). */
const manifest = {
  defaultModel: "e2b",
  models: {
    e2b: {
      defaultQuant: "i4",
      quants: { i4: { label: "int4", weights: {}, session: {} } },
      weights: {},
      pipelineConfig: {
        capacity: 4096,
        chunkLength: 768,
        maxChunkLength: 768,
        maxPosition: 32768,
        rope: {
          sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
          full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
        },
      },
    },
  },
} as unknown as Manifest;

const draft = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  id: 0,
  key: { mount: "gemma4-qat", model: "e2b" },
  tap: true,
  ...patch,
});

describe("the PLE-on-GPU and decode-ahead switches", () => {
  describe("an arm", () => {
    it("passes pleResidency and decodeAhead as pipeline options", () => {
      const arm = resolveArm(draft({ pleOnGpu: true, decodeAhead: "4" }), {
        "gemma4-qat": manifest,
      });
      expect(arm.ok && arm.arm.pipeline).toEqual({ pleResidency: "gpu", decodeAhead: 4 });
      // The pipeline options are not Session options.
      expect(arm.ok && arm.arm.session).toBeUndefined();
    });

    it("passes the PLE placement alone at depth 1, and nothing when both are off", () => {
      const ple = resolveArm(draft({ pleOnGpu: true, decodeAhead: "1" }), {
        "gemma4-qat": manifest,
      });
      expect(ple.ok && ple.arm.pipeline).toEqual({ pleResidency: "gpu" });
      for (const off of [draft(), draft({ pleOnGpu: false, decodeAhead: "1" })]) {
        const arm = resolveArm(off, { "gemma4-qat": manifest });
        expect(arm.ok && arm.arm.pipeline).toBeUndefined();
      }
    });

    it("cannot run ahead without the PLE on the GPU, and says why", () => {
      const arm = resolveArm(draft({ decodeAhead: "2" }), { "gemma4-qat": manifest });
      expect(arm.ok).toBe(false);
      expect(!arm.ok && arm.error).toContain("PLE on GPU");
    });

    it("is labelled with the placement and the depth, before the tap's own marks", () => {
      const arm = { mount: "gemma4-qat", model: "e2b", quant: "i4" } as const;
      expect(
        armLabel({ ...arm, tap: true, pipeline: { pleResidency: "gpu", decodeAhead: 8 } }),
      ).toBe("QAT e2b i4 (PLE GPU) (ahead 8)");
      expect(
        armLabel({
          ...arm,
          tap: false,
          session: { linearGemvPlanes: true },
          pipeline: { pleResidency: "gpu" },
        }),
      ).toBe("QAT e2b i4 (planes) (PLE GPU) (tap off)");
    });
  });

  describe("the record", () => {
    const ENVIRONMENT: RecordEnvironment = {
      recordedAt: "2026-11-03T10:00:00.000Z",
      userAgent: "test",
      crossOriginIsolated: true,
      adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "" },
      device: {
        features: [],
        wgslLanguageFeatures: [],
        subgroupLanesLinear: false,
        gpuTimingEnabled: false,
        limits: {
          maxComputeWorkgroupStorageSize: 32768,
          maxStorageBufferBindingSize: 4294967292,
          maxBufferSize: 4294967292,
        },
      },
    };
    const armOf = (pipeline?: RunArm["pipeline"]): RunArm => ({
      mount: "gemma4-qat",
      model: "e2b",
      quant: "i4",
      quantLabel: "int4",
      isDefaultQuant: true,
      tap: true,
      ...(pipeline === undefined ? {} : { pipeline }),
      sampler: { temperature: 0 },
    });

    it("records each arm's pipeline options ({} when none)", () => {
      const record = buildBenchRecord({
        environment: ENVIRONMENT,
        timestampQuery: true,
        complete: true,
        settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
        arms: [armOf(), armOf({ pleResidency: "gpu", decodeAhead: 4 })],
        distributions: { "gemma4-qat": { directory: "qat", manifestSha256: "00" } },
        source: { kind: "direct" },
        visits: [],
        chats: [],
      });
      expect(record.arms.map(({ pipelineOptions }) => pipelineOptions)).toEqual([
        {},
        { pleResidency: "gpu", decodeAhead: 4 },
      ]);
    });
  });
});
