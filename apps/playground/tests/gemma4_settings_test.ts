import { type Manifest, parseManifest } from "@karume/hub";
import type { ChatMessage } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import {
  defaultModelKey,
  effectiveModelKey,
  entryPointOf,
  gemmaModelChoices,
  hasDrafter,
  isQatModel,
  modelConfig,
  overflowPolicy,
  parseChatForm,
  parseModelKey,
  parseTurnForm,
  sameChatSettings,
  type TurnForm,
  turnFormOf,
} from "../src/models/gemma4/settings.ts";

const hex = (character: string): string => character.repeat(64);

const container = (path: string) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [
      { path: `${path}-00001`, size: 40, sha256: hex("a") },
      { path: `${path}-00002`, size: 8, sha256: hex("b") },
    ],
  },
});

const PIPELINE_CONFIG = {
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 131072,
  capacity: 4096,
  rope: {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  },
  sampler: { temperature: 1, topK: 64, topP: 0.95 },
};

/**
 * A `karume/5` manifest shaped like the Gemma 4 distributions: each model has the presets `i4` and
 * `i4-fast` (default); `withDrafter` adds drafter weights (mapped by both presets).
 */
const gemmaManifest = (models: readonly string[], options: { withDrafter: boolean }): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: models[0],
      models: Object.fromEntries(
        models.map((name) => [
          name,
          {
            pipeline: "gemma4/1",
            assets: {},
            weights: {
              model: { i4: container(`${name}/model.krm`) },
              ...(options.withDrafter ? { drafter: { i8: container(`${name}/drafter.krm`) } } : {}),
            },
            quants: {
              i4: {
                weights: { model: "i4", ...(options.withDrafter ? { drafter: "i8" } : {}) },
                session: {},
              },
              "i4-fast": {
                weights: { model: "i4", ...(options.withDrafter ? { drafter: "i8" } : {}) },
                session: {},
              },
            },
            defaultQuant: "i4-fast",
            pipelineConfig: PIPELINE_CONFIG,
          },
        ]),
      ),
    }),
  );

const regular = gemmaManifest(["e2b"], { withDrafter: true });
const qat = gemmaManifest(["e2b", "e4b"], { withDrafter: false });

describe("model keys", () => {
  it("round-trips a mount and a model through the select value", () => {
    expect(parseModelKey("gemma4-qat-mobile/e4b")).toEqual({
      mount: "gemma4-qat-mobile",
      model: "e4b",
    });
  });

  it("rejects an unknown mount or an empty model", () => {
    expect(parseModelKey("anima/e2b")).toBeUndefined();
    expect(parseModelKey("gemma4/")).toBeUndefined();
    expect(parseModelKey("gemma4")).toBeUndefined();
  });

  it("accepts only the models gemma4QatMobile.Pipeline takes", () => {
    expect(isQatModel("e2b")).toBe(true);
    expect(isQatModel("e4b")).toBe(true);
    expect(isQatModel("12b")).toBe(false);
  });
});

describe("gemmaModelChoices", () => {
  it("lists every model of both mounts, each manifest's default marked", () => {
    expect(gemmaModelChoices({ gemma4: regular, "gemma4-qat-mobile": qat })).toEqual([
      { value: "gemma4/e2b", label: "Gemma 4 — e2b (default)" },
      { value: "gemma4-qat-mobile/e2b", label: "Gemma 4 mobile QAT — e2b (default)" },
      { value: "gemma4-qat-mobile/e4b", label: "Gemma 4 mobile QAT — e4b" },
    ]);
  });

  it("disables a QAT model the entry point does not accept", () => {
    const odd = gemmaManifest(["e2b", "12b"], { withDrafter: false });
    expect(gemmaModelChoices({ "gemma4-qat-mobile": odd }).at(-1)).toEqual({
      value: "gemma4-qat-mobile/12b",
      label: "Gemma 4 mobile QAT — 12b (not accepted by gemma4QatMobile.Pipeline)",
      disabled: true,
    });
  });

  it("lists nothing for a mount whose manifest is not there", () => {
    expect(gemmaModelChoices({ "gemma4-qat-mobile": qat }).map(({ value }) => value)).toEqual([
      "gemma4-qat-mobile/e2b",
      "gemma4-qat-mobile/e4b",
    ]);
  });
});

describe("the model the page starts with", () => {
  it("prefers regular Gemma 4's default model", () => {
    expect(defaultModelKey({ gemma4: regular, "gemma4-qat-mobile": qat })).toEqual({
      mount: "gemma4",
      model: "e2b",
    });
  });

  it("falls back to the QAT distribution when regular Gemma 4 is missing", () => {
    expect(defaultModelKey({ "gemma4-qat-mobile": qat })).toEqual({
      mount: "gemma4-qat-mobile",
      model: "e2b",
    });
  });

  it("keeps a chosen model while its manifest holds it, else returns to the default", () => {
    const manifests = { gemma4: regular, "gemma4-qat-mobile": qat };
    expect(effectiveModelKey({ mount: "gemma4-qat-mobile", model: "e4b" }, manifests)).toEqual({
      mount: "gemma4-qat-mobile",
      model: "e4b",
    });
    expect(effectiveModelKey({ mount: "gemma4", model: "e4b" }, manifests)).toEqual({
      mount: "gemma4",
      model: "e2b",
    });
  });
});

describe("hasDrafter", () => {
  it("is true only for a model of the regular entry point with drafter weights", () => {
    expect(hasDrafter(regular, { mount: "gemma4", model: "e2b" })).toBe(true);
    expect(hasDrafter(qat, { mount: "gemma4", model: "e2b" })).toBe(false);
    expect(hasDrafter(regular, { mount: "gemma4-qat-mobile", model: "e2b" })).toBe(false);
  });

  it("is true for the q4_0 QAT line, which the regular entry point reads with its drafters", () => {
    const line = gemmaManifest(["e2b", "e4b", "12b"], { withDrafter: true });
    for (const model of ["e2b", "e4b", "12b"]) {
      expect(hasDrafter(line, { mount: "gemma4-qat", model })).toBe(true);
    }
  });
});

describe("entryPointOf", () => {
  it("builds only the mobile QAT mount with the mobile entry point", () => {
    expect(entryPointOf("gemma4")).toBe("gemma.Pipeline");
    expect(entryPointOf("gemma4-qat")).toBe("gemma.Pipeline");
    expect(entryPointOf("gemma4-qat-mobile")).toBe("gemma4QatMobile.Pipeline");
  });
});

describe("modelConfig", () => {
  it("reads the validated pipelineConfig of the model", () => {
    const config = modelConfig(regular, "e2b");
    expect(config.capacity).toBe(4096);
    expect(config.sampler).toEqual({ temperature: 1, topK: 64, topP: 0.95 });
  });

  it("fails loudly for a model the manifest does not have", () => {
    expect(() => modelConfig(regular, "e4b")).toThrow(/no model 'e4b'/);
  });
});

describe("the per-turn form", () => {
  const form = turnFormOf({ temperature: 1, topK: 64, topP: 0.95 });

  it("starts from the distribution's sampler, seed 0 and 512 new tokens", () => {
    expect(form).toEqual({
      temperature: "1",
      topK: "64",
      topP: "0.95",
      seed: "0",
      maxNewTokens: "512",
    });
    expect(parseTurnForm(form)).toEqual({
      ok: true,
      value: { maxNewTokens: 512, sampler: { temperature: 1, topK: 64, topP: 0.95, seed: 0 } },
    });
  });

  it("starts greedy when the distribution declares no sampler", () => {
    expect(parseTurnForm(turnFormOf(undefined))).toEqual({
      ok: true,
      value: { maxNewTokens: 512, sampler: { temperature: 0, seed: 0 } },
    });
  });

  it("leaves an empty top-k / top-p out of the sampler instead of filling a default", () => {
    const result = parseTurnForm({ ...form, topK: "", topP: " " });
    expect(result.ok && result.value.sampler).toEqual({ temperature: 1, seed: 0 });
  });

  it("reports every out-of-range field at once", () => {
    const bad: TurnForm = {
      temperature: "-0.5",
      topK: "1.5",
      topP: "1.2",
      seed: "-1",
      maxNewTokens: "0",
    };
    const result = parseTurnForm(bad);
    expect(result.ok).toBe(false);
    expect(result.ok ? [] : result.errors).toHaveLength(5);
  });

  it("accepts top-p 1 and rejects top-p 0", () => {
    expect(parseTurnForm({ ...form, topP: "1" }).ok).toBe(true);
    expect(parseTurnForm({ ...form, topP: "0" }).ok).toBe(false);
  });
});

describe("the chat form", () => {
  const limits = { capacity: 4096, chunkLength: 768, maxPosition: 131072 };

  it("takes the distribution's capacity when empty and no system message when blank", () => {
    expect(parseChatForm({ system: "  ", capacity: "", overflow: "drop-oldest" }, limits)).toEqual({
      ok: true,
      value: { capacity: 4096, overflow: "drop-oldest" },
    });
  });

  it("keeps the system prompt as typed", () => {
    const result = parseChatForm(
      { system: " Be brief. ", capacity: "8192", overflow: "refuse" },
      limits,
    );
    expect(result).toEqual({
      ok: true,
      value: { system: " Be brief. ", capacity: 8192, overflow: "refuse" },
    });
  });

  it("accepts the capacity bounds the session checks and refuses outside them", () => {
    const at = (capacity: string) =>
      parseChatForm({ system: "", capacity, overflow: "drop-oldest" }, limits).ok;
    expect(at("768")).toBe(true);
    expect(at("131072")).toBe(true);
    expect(at("767")).toBe(false);
    expect(at("131073")).toBe(false);
    expect(at("4096.5")).toBe(false);
  });

  it("compares settings by value", () => {
    const base = { capacity: 4096, overflow: "drop-oldest" as const };
    expect(sameChatSettings(base, { ...base })).toBe(true);
    expect(sameChatSettings(base, { ...base, system: "x" })).toBe(false);
    expect(sameChatSettings(base, { ...base, capacity: 2048 })).toBe(false);
  });
});

describe("overflowPolicy", () => {
  const turns: readonly ChatMessage[] = [
    { role: "system", content: "Be brief." },
    { role: "user", content: "Hi" },
    { role: "assistant", content: "Hello." },
    { role: "user", content: "Again" },
    { role: "assistant", content: "Hello again." },
    { role: "user", content: "Third" },
  ];
  const context = { turns, system: "Be brief.", capacity: 1024, needed: 1500 };

  it("refuses by returning the history unchanged (the session then fails the turn) and reports 0", async () => {
    const reported: number[] = [];
    const next = await overflowPolicy("refuse", (count) => reported.push(count))(context);
    expect(next).toEqual(turns);
    expect(reported).toEqual([0]);
  });

  it("drops the oldest user / assistant pair, keeps the system message and reports the count", async () => {
    const reported: number[] = [];
    const next = await overflowPolicy("drop-oldest", (count) => reported.push(count))(context);
    expect(next).toEqual([turns[0], ...turns.slice(3)]);
    expect(reported).toEqual([turns.length - next.length]);
    expect(reported).toEqual([2]);
  });
});
