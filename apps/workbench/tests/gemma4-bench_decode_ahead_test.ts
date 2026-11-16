import type { Manifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { buildBenchData } from "../src/gemma4-bench/data.ts";
import { type ArmDraft, armLabel, resolveArm } from "../src/gemma4-bench/plan.ts";
import type { RunArm } from "../src/gemma4-bench/run.ts";

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

/** The same preset table under the model name that has no PLE. */
const manifestWith12b = {
  defaultModel: "12b",
  models: { "12b": manifest.models.e2b },
} as unknown as Manifest;

const draft = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  key: { mount: "gemma4-qat-mobile", model: "e2b" },
  tap: true,
  ...patch,
});

describe("the PLE-on-GPU and decode-ahead switches", () => {
  describe("an arm", () => {
    it("passes pleResidency and decodeAhead as pipeline options", () => {
      const arm = resolveArm(draft({ pleOnGpu: true, decodeAhead: "4" }), {
        "gemma4-qat-mobile": manifest,
      });
      expect(arm.ok && arm.arm.pipeline).toEqual({ pleResidency: "gpu", decodeAhead: 4 });
      // The pipeline options are not Session options.
      expect(arm.ok && arm.arm.session).toBeUndefined();
    });

    it("passes the PLE placement alone at depth 1, and nothing when both are off", () => {
      const ple = resolveArm(draft({ pleOnGpu: true, decodeAhead: "1" }), {
        "gemma4-qat-mobile": manifest,
      });
      expect(ple.ok && ple.arm.pipeline).toEqual({ pleResidency: "gpu" });
      for (const off of [draft(), draft({ pleOnGpu: false, decodeAhead: "1" })]) {
        const arm = resolveArm(off, { "gemma4-qat-mobile": manifest });
        expect(arm.ok && arm.arm.pipeline).toBeUndefined();
      }
    });

    it("cannot run ahead without the PLE on the GPU, and says why", () => {
      const arm = resolveArm(draft({ decodeAhead: "2" }), { "gemma4-qat-mobile": manifest });
      expect(arm.ok).toBe(false);
      expect(!arm.ok && arm.error).toContain("PLE on GPU");
    });

    it("runs ahead on 12B, which has no PLE, and refuses to place one", () => {
      const twelve = (extra: Partial<ArmDraft>): ArmDraft => ({
        key: { mount: "gemma4-qat", model: "12b" },
        tap: false,
        ...extra,
      });
      const qat = { "gemma4-qat": manifestWith12b };
      const arm = resolveArm(twelve({ decodeAhead: "4" }), qat);
      expect(arm.ok && arm.arm.pipeline).toEqual({ decodeAhead: 4 });
      const placed = resolveArm(twelve({ pleOnGpu: true }), qat);
      expect(!placed.ok && placed.error).toContain("no PLE");
    });

    it("is labelled with the placement and the depth, before the tap's own marks", () => {
      const arm = { mount: "gemma4-qat-mobile", model: "e2b", quant: "i4" } as const;
      expect(
        armLabel({ ...arm, tap: true, pipeline: { pleResidency: "gpu", decodeAhead: 8 } }),
      ).toBe("mobile QAT e2b i4 (PLE GPU) (ahead 8)");
      expect(
        armLabel({
          ...arm,
          tap: false,
          session: { integerGemv: true },
          pipeline: { pleResidency: "gpu" },
        }),
      ).toBe("mobile QAT e2b i4 (integer) (PLE GPU) (tap off)");
    });
  });

  describe("the data", () => {
    const armOf = (pipeline?: RunArm["pipeline"]): RunArm => ({
      mount: "gemma4-qat-mobile",
      model: "e2b",
      quant: "i4",
      quantLabel: "int4",
      isDefaultQuant: true,
      tap: true,
      ...(pipeline === undefined ? {} : { pipeline }),
      sampler: { temperature: 0 },
    });

    it("records each arm's pipeline options ({} when none)", () => {
      const data = buildBenchData({
        timestampQuery: true,
        complete: true,
        settings: { chats: 2, newTokens: 2, rounds: 1, sampler: "greedy", prompts: ["few-ids"] },
        arms: [armOf(), armOf({ pleResidency: "gpu", decodeAhead: 4 })],
        distributions: {
          "gemma4-qat-mobile": { directory: "qat", manifestSha256: "00", read: "direct" },
        },
        visits: [],
        chats: [],
      });
      expect(data.arms.map(({ pipelineOptions }) => pipelineOptions)).toEqual([
        {},
        { pleResidency: "gpu", decodeAhead: 4 },
      ]);
    });
  });
});
