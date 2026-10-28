import type { Prompt } from "@karume/models/wan";
import { describe, expect, it } from "vite-plus/test";

import {
  matchingScenario,
  type Scenario,
  SCENARIOS,
  scenarioSettings,
  ScenarioUnavailableError,
  type ScenarioContext,
} from "../src/models/wan22/scenarios.ts";
import { buildRequest, initialForm, type WanForm } from "../src/models/wan22/settings.ts";

const prompts: readonly Prompt[] = [
  {
    name: "neon-city",
    role: "positive",
    prompt: "A neon city at night.",
    normalized: "a neon city",
  },
  {
    name: "boxing-cats",
    role: "positive",
    prompt: "Two anthropomorphic cats box on a stage.",
    normalized: "two anthropomorphic cats box on a stage.",
  },
  { name: "fixed-negative", role: "negative", prompt: "blurry", normalized: "blurry" },
];

const context: ScenarioContext = {
  defaultQuant: "i8-a8-attn8-online-conv8",
  prompts,
  cachedSource: true,
  gpuTextEncoder: true,
};

const scenarioById = (id: Scenario["id"]): Scenario => {
  const found = SCENARIOS.find((scenario) => scenario.id === id);
  if (found === undefined) throw new Error(`no scenario '${id}'`);
  return found;
};

describe("SCENARIOS", () => {
  it("lists the four scenarios in order with distinct ids", () => {
    expect(SCENARIOS.map(({ id }) => id)).toEqual([
      "quick-check",
      "bench-33f",
      "bench-33f-tile32",
      "bench-121f",
    ]);
  });
});

describe("scenarioSettings", () => {
  describe("when the page can apply the scenario", () => {
    it("gives the 33-frame bench the fixed request on the default quant through the cache", () => {
      const { form, source } = scenarioSettings(scenarioById("bench-33f"), context);
      expect(source).toBe("cached");
      expect(form).toEqual({
        quant: "i8-a8-attn8-online-conv8",
        vaeTile: 16,
        size: "1280x704",
        frames: 33,
        steps: "50",
        seed: "42",
        shift: "",
        textEncoder: "gpu",
        promptName: "boxing-cats",
        prompt: "Two anthropomorphic cats box on a stage.",
        negativePrompt: "",
      });
    });

    it.each([
      ["quick-check", 17, 2, 16],
      ["bench-33f", 33, 50, 16],
      ["bench-33f-tile32", 33, 50, 32],
      ["bench-121f", 121, 50, 32],
    ] as const)(
      "builds a valid request for %s (%i frames, %i steps, tile %i) with the official negative prompt and the manifest's shift",
      (id, frames, steps, vaeTile) => {
        const request = buildRequest(scenarioSettings(scenarioById(id), context).form, prompts);
        expect(request).toEqual({
          prompt: "Two anthropomorphic cats box on a stage.",
          seed: 42,
          steps,
          frames,
          width: 1280,
          height: 704,
          vaeTile,
        });
      },
    );
  });

  describe("when the page cannot apply the scenario", () => {
    const quick = scenarioById("quick-check");

    it("refuses without the HF-compatible route", () => {
      expect(() => scenarioSettings(quick, { ...context, cachedSource: false })).toThrow(
        /HF-compatible route/,
      );
    });

    it("refuses without the umT5 distribution", () => {
      expect(() => scenarioSettings(quick, { ...context, gpuTextEncoder: false })).toThrow(
        /umT5 distribution is missing/,
      );
    });

    it("refuses when the asset has no positive 'boxing-cats' row", () => {
      const withoutRow = prompts.filter(({ name }) => name !== "boxing-cats");
      expect(() => scenarioSettings(quick, { ...context, prompts: withoutRow })).toThrow(
        /does not have/,
      );
    });

    it("refuses a 'boxing-cats' row that is a negative prompt", () => {
      const negativeOnly: readonly Prompt[] = [
        { name: "boxing-cats", role: "negative", prompt: "x", normalized: "x" },
      ];
      expect(() => scenarioSettings(quick, { ...context, prompts: negativeOnly })).toThrow(
        ScenarioUnavailableError,
      );
    });

    it("refuses while the prompt list is not known", () => {
      expect(() => scenarioSettings(quick, { ...context, prompts: [] })).toThrow(/not known/);
    });
  });
});

describe("matchingScenario", () => {
  const loaded = (id: Scenario["id"]): WanForm => scenarioSettings(scenarioById(id), context).form;

  it.each(SCENARIOS.map(({ id }) => id))("recognizes the settings %s applies", (id) => {
    expect(matchingScenario(loaded(id), "cached", context)).toBe(id);
  });

  it("tells the two 33-frame benches apart by the VAE tile", () => {
    const form = { ...loaded("bench-33f"), vaeTile: 32 as const };
    expect(matchingScenario(form, "cached", context)).toBe("bench-33f-tile32");
  });

  it("is custom once a field the request reads is edited", () => {
    const edits: readonly Partial<WanForm>[] = [
      { quant: "i8" },
      { size: "704x1280" },
      { frames: 21 },
      { steps: "49" },
      { seed: "43" },
      { shift: "5" },
      { textEncoder: "precomputed" },
      { prompt: "Two cats box." },
      { negativePrompt: "blurry" },
    ];
    for (const edit of edits) {
      expect(matchingScenario({ ...loaded("bench-121f"), ...edit }, "cached", context)).toBe(
        "custom",
      );
    }
  });

  it("is custom on the direct source", () => {
    expect(matchingScenario(loaded("bench-33f"), "direct", context)).toBe("custom");
  });

  it("ignores the precomputed prompt choice, which the GPU path does not read", () => {
    const form = { ...loaded("quick-check"), promptName: "neon-city" };
    expect(matchingScenario(form, "cached", context)).toBe("quick-check");
  });

  it("is custom for the page's initial form", () => {
    expect(matchingScenario(initialForm(context.defaultQuant), "cached", context)).toBe("custom");
  });

  it("is custom when no scenario can apply", () => {
    const form = loaded("bench-33f");
    expect(matchingScenario(form, "cached", { ...context, prompts: [] })).toBe("custom");
  });
});
