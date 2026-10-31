import type { Manifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  type ArmDraft,
  armLabel,
  BENCH_PROMPTS,
  defaultArmDrafts,
  INITIAL_BENCH_FORM,
  meetsDistinctIdCondition,
  parseBenchForm,
  resolveArm,
  samplerOf,
  visitOrder,
} from "../src/models/gemma4-bench/plan.ts";

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
  id: 0,
  key: { mount: "gemma4-qat", model: "e2b" },
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
        "gemma4-qat": manifest("e2b", ["e2b", "e4b"]),
        gemma4: manifest("e2b", ["e2b", "e4b"]),
      });
      expect(drafts.map(({ key, tap }) => ({ ...key, tap }))).toEqual([
        { mount: "gemma4", model: "e2b", tap: true },
        { mount: "gemma4-qat", model: "e2b", tap: true },
      ]);
      expect(new Set(drafts.map(({ id }) => id)).size).toBe(2);
    });

    it("leaves out a distribution whose manifest is not loaded", () => {
      expect(defaultArmDrafts({ "gemma4-qat": manifest("e4b", ["e4b"]) })).toEqual([
        { id: 1, key: { mount: "gemma4-qat", model: "e4b" }, tap: true },
      ]);
    });
  });

  describe("resolved against the manifests", () => {
    const manifests = { "gemma4-qat": manifest("e2b", ["e2b", "e4b", "e12b"]) };

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

    it("falls back to the default preset when the chosen one is not the model's", () => {
      expect(resolveArm(draft({ quant: "bf16" }), manifests)).toMatchObject({
        ok: true,
        arm: { quant: "i4-fast" },
      });
    });

    it("carries the distribution's declared sampler", () => {
      expect(resolveArm(draft(), manifests)).toMatchObject({
        ok: true,
        arm: { declaredSampler: { temperature: 1, topK: 64, topP: 0.95 } },
      });
    });

    it("refuses a QAT model the QAT entry point does not accept", () => {
      expect(resolveArm(draft({ key: { mount: "gemma4-qat", model: "e12b" } }), manifests)).toEqual(
        { ok: false, error: "gemma4Qat.Pipeline does not accept the model 'e12b'." },
      );
    });

    it("refuses a model of a manifest that is not loaded, or that it does not have", () => {
      expect(resolveArm(draft({ key: { mount: "gemma4", model: "e2b" } }), manifests).ok).toBe(
        false,
      );
      expect(resolveArm(draft({ key: { mount: "gemma4-qat", model: "e9b" } }), manifests).ok).toBe(
        false,
      );
    });
  });

  it("labels an arm by distribution, model and preset, marking a tap-off control", () => {
    expect(armLabel({ mount: "gemma4", model: "e2b", quant: "i4-fast", tap: true })).toBe(
      "Gemma 4 e2b i4-fast",
    );
    expect(armLabel({ mount: "gemma4-qat", model: "e4b", quant: "i4", tap: false })).toBe(
      "QAT e4b i4 (tap off)",
    );
  });
});

describe("the sampler", () => {
  const declared = { temperature: 1, topK: 64, topP: 0.95 };

  it("is greedy by default", () => {
    expect(INITIAL_BENCH_FORM.sampler).toBe("greedy");
    expect(samplerOf("greedy", declared)).toEqual({ temperature: 0 });
  });

  it("takes the distribution's declared sampler with seed 0 when asked", () => {
    expect(samplerOf("distribution", declared)).toEqual({ ...declared, seed: 0 });
  });

  it("stays greedy when the distribution declares none", () => {
    expect(samplerOf("distribution", undefined)).toEqual({ temperature: 0 });
  });
});

describe("the workload form", () => {
  it("defaults to 3 chats of 128 new tokens over 4 rounds, both prompts", () => {
    expect(parseBenchForm(INITIAL_BENCH_FORM)).toEqual({
      ok: true,
      value: {
        chats: 3,
        newTokens: 128,
        rounds: 4,
        sampler: "greedy",
        prompts: ["few-ids", "many-ids"],
      },
    });
  });

  it("keeps the prompts in the set's order whatever order they were ticked in", () => {
    const parsed = parseBenchForm({ ...INITIAL_BENCH_FORM, prompts: ["many-ids", "few-ids"] });
    expect(parsed.ok && parsed.value.prompts).toEqual(["few-ids", "many-ids"]);
  });

  it("needs a warm chat, a decode token, a round and a prompt — and reports every problem", () => {
    const parsed = parseBenchForm({
      chats: "1",
      newTokens: "1",
      rounds: "0",
      sampler: "greedy",
      prompts: [],
    });
    expect(parsed.ok).toBe(false);
    expect(parsed.ok ? [] : parsed.errors).toHaveLength(4);
  });

  it("refuses text that is not a non-negative integer", () => {
    expect(parseBenchForm({ ...INITIAL_BENCH_FORM, chats: "2.5" }).ok).toBe(false);
    expect(parseBenchForm({ ...INITIAL_BENCH_FORM, rounds: "-1" }).ok).toBe(false);
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
