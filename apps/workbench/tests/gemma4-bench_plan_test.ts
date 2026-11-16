import type { Manifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  type ArmDraft,
  armLabel,
  BENCH_PROMPTS,
  defaultArmDrafts,
  meetsDistinctIdCondition,
  resolveArm,
  samplerOf,
  visitOrder,
} from "../src/gemma4-bench/plan.ts";

const pipelineConfig = {
  capacity: 4096,
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 32768,
  rope: {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  },
  sampler: { temperature: 1, topK: 64, topP: 0.95 },
};

/** A manifest with the given models, each with presets `i4` and `i4-fast` (`i4-fast` the default). */
const manifest = (defaultModel: string, models: readonly string[]): Manifest =>
  ({
    defaultModel,
    models: Object.fromEntries(
      models.map((model) => [
        model,
        {
          defaultQuant: "i4-fast",
          quants: { i4: { label: "int4" }, "i4-fast": { label: "int4 fast" } },
          weights: {},
          pipelineConfig,
        },
      ]),
    ),
  }) as unknown as Manifest;

const draft = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  key: { mount: "gemma4-qat-mobile", model: "e2b" },
  tap: true,
  ...patch,
});

describe("the prompt set", () => {
  it("has one prompt on each side of the PLE distinct-id threshold", () => {
    expect(BENCH_PROMPTS.map(({ distinctIds }) => distinctIds)).toEqual(["below", "at-least"]);
  });

  it("checks a measured distinct count against the side its prompt is meant for", () => {
    const [few, many] = BENCH_PROMPTS;
    expect(meetsDistinctIdCondition(few, 31)).toBe(true);
    expect(meetsDistinctIdCondition(few, 32)).toBe(false);
    expect(meetsDistinctIdCondition(many, 32)).toBe(true);
    expect(meetsDistinctIdCondition(many, 31)).toBe(false);
  });
});

describe("the arms", () => {
  describe("by default", () => {
    it("takes each loaded distribution's default model, regular Gemma 4 first, tap on", () => {
      const drafts = defaultArmDrafts({
        "gemma4-qat-mobile": manifest("e2b", ["e2b", "e4b"]),
        gemma4: manifest("e2b", ["e2b", "e4b"]),
      });
      expect(drafts.map(({ key, tap }) => ({ ...key, tap }))).toEqual([
        { mount: "gemma4", model: "e2b", tap: true },
        { mount: "gemma4-qat-mobile", model: "e2b", tap: true },
      ]);
    });

    it("leaves out a distribution whose manifest is not loaded", () => {
      expect(defaultArmDrafts({ "gemma4-qat-mobile": manifest("e4b", ["e4b"]) })).toEqual([
        { key: { mount: "gemma4-qat-mobile", model: "e4b" }, tap: true },
      ]);
    });
  });

  describe("resolved against the manifests", () => {
    const manifests = { "gemma4-qat-mobile": manifest("e2b", ["e2b", "e4b", "e12b"]) };

    it("preselects the model's default preset", () => {
      const resolved = resolveArm(draft(), manifests);
      expect(resolved).toMatchObject({
        ok: true,
        arm: { quant: "i4-fast", quantLabel: "int4 fast", isDefaultQuant: true },
      });
    });

    it("keeps a chosen preset and says it is not the default", () => {
      expect(resolveArm(draft({ quant: "i4" }), manifests)).toMatchObject({
        ok: true,
        arm: { quant: "i4", isDefaultQuant: false },
      });
    });

    it("refuses a preset the model does not have, naming its presets", () => {
      expect(resolveArm(draft({ quant: "bf16" }), manifests)).toEqual({
        ok: false,
        error:
          "The gemma4-qat-mobile model 'e2b' has no preset 'bf16'; its presets are i4, i4-fast.",
      });
    });

    it("carries the distribution's declared sampler", () => {
      expect(resolveArm(draft(), manifests)).toMatchObject({
        ok: true,
        arm: { declaredSampler: { temperature: 1, topK: 64, topP: 0.95 } },
      });
    });

    it("refuses a QAT model the QAT entry point does not accept", () => {
      expect(
        resolveArm(draft({ key: { mount: "gemma4-qat-mobile", model: "e12b" } }), manifests),
      ).toEqual({ ok: false, error: "gemma4QatMobile.Pipeline does not accept the model 'e12b'." });
    });

    it("refuses a model of a manifest that is not loaded, or that it does not have", () => {
      expect(resolveArm(draft({ key: { mount: "gemma4", model: "e2b" } }), manifests).ok).toBe(
        false,
      );
      expect(
        resolveArm(draft({ key: { mount: "gemma4-qat-mobile", model: "e9b" } }), manifests).ok,
      ).toBe(false);
    });

    it("names the models the manifest has when it refuses one", () => {
      expect(
        resolveArm(draft({ key: { mount: "gemma4-qat-mobile", model: "e9b" } }), manifests),
      ).toEqual({
        ok: false,
        error: "The gemma4-qat-mobile manifest has no model 'e9b'; its models are e2b, e4b, e12b.",
      });
    });
  });

  it("labels an arm by distribution, model and preset, marking a tap-off control", () => {
    expect(armLabel({ mount: "gemma4", model: "e2b", quant: "i4-fast", tap: true })).toBe(
      "Gemma 4 e2b i4-fast",
    );
    expect(armLabel({ mount: "gemma4-qat-mobile", model: "e4b", quant: "i4", tap: false })).toBe(
      "mobile QAT e4b i4 (tap off)",
    );
  });
});

describe("the sampler", () => {
  const declared = { temperature: 1, topK: 64, topP: 0.95 };

  // The default (greedy) is the entry's `sampler` parameter: gemma4-bench_entry_test.ts.
  it("is greedy at temperature 0", () => {
    expect(samplerOf("greedy", declared)).toEqual({ temperature: 0 });
  });

  it("takes the distribution's declared sampler with seed 0 when asked", () => {
    expect(samplerOf("distribution", declared)).toEqual({ ...declared, seed: 0 });
  });

  it("stays greedy when the distribution declares none", () => {
    expect(samplerOf("distribution", undefined)).toEqual({ temperature: 0 });
  });
});

describe("the visit order", () => {
  it("is ABBA for two arms", () => {
    expect(
      visitOrder(2, 4)
        .map(({ arm }) => "AB"[arm])
        .join(""),
    ).toBe("ABBAABBA");
  });

  it("reverses every odd round for more arms, numbering visits and rounds", () => {
    expect(visitOrder(3, 2)).toEqual([
      { visit: 0, round: 0, arm: 0 },
      { visit: 1, round: 0, arm: 1 },
      { visit: 2, round: 0, arm: 2 },
      { visit: 3, round: 1, arm: 2 },
      { visit: 4, round: 1, arm: 1 },
      { visit: 5, round: 1, arm: 0 },
    ]);
  });

  it("visits every arm once per round", () => {
    const visits = visitOrder(3, 5);
    for (let round = 0; round < 5; round += 1) {
      expect(
        visits
          .filter((visit) => visit.round === round)
          .map(({ arm }) => arm)
          .sort((left, right) => left - right),
      ).toEqual([0, 1, 2]);
    }
  });
});
