import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  dimensionChoices,
  effectiveDimension,
  promptChoices,
  promptOfValue,
  promptValue,
  readEmbeddingConfig,
} from "../src/models/embeddinggemma/config.ts";

const hex = (character: string): string => character.repeat(64);

/** The pipelineConfig the EmbeddingGemma 2 dist recipe writes (three of its twenty prompts). */
const CONFIG: Readonly<Record<string, unknown>> = {
  prompts: {
    query: "task: search result | query: ",
    document: "title: none | text: ",
    STS: "task: sentence similarity | query: ",
  },
  dimensions: [768, 512, 256, 128],
  maxTokens: 2048,
  placeholderTokens: { image: 258880, audio: 258881, video: 258884 },
};

/** A minimal `karume/6` manifest with one model `text` (pipeline `embeddinggemma/1`). */
const manifestWith = (pipelineConfig: Readonly<Record<string, unknown>>) =>
  parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "test",
      defaultModel: "text",
      models: {
        text: {
          pipeline: "embeddinggemma/1",
          assets: {},
          weights: {
            model: {
              bf16: {
                container: {
                  descriptor: {
                    graph: { length: 8, sha256: hex("d") },
                    model: { length: 8, sha256: hex("e") },
                  },
                  parts: [
                    {
                      path: "text/model/model.bf16-00001-of-00002.krm",
                      size: 40,
                      sha256: hex("a"),
                    },
                    { path: "text/model/model.bf16-00002-of-00002.krm", size: 8, sha256: hex("b") },
                  ],
                },
              },
            },
          },
          quants: { bf16: { weights: { model: "bf16" }, session: {} } },
          defaultQuant: "bf16",
          pipelineConfig,
        },
      },
    }),
  );

describe("readEmbeddingConfig", () => {
  it("reads the prompts in the manifest's order with their prefixes, the dimensions and the limit", () => {
    expect(readEmbeddingConfig(manifestWith(CONFIG), "text")).toEqual({
      prompts: [
        { name: "query", prefix: "task: search result | query: " },
        { name: "document", prefix: "title: none | text: " },
        { name: "STS", prefix: "task: sentence similarity | query: " },
      ],
      dimensions: [768, 512, 256, 128],
      maxTokens: 2048,
    });
  });

  it("refuses a model the manifest does not have", () => {
    expect(() => readEmbeddingConfig(manifestWith(CONFIG), "image")).toThrow(/no model 'image'/);
  });

  it("names the field that is off instead of guessing", () => {
    const read = (change: Readonly<Record<string, unknown>>) => () =>
      readEmbeddingConfig(manifestWith({ ...CONFIG, ...change }), "text");
    expect(read({ prompts: ["query"] })).toThrow(/pipelineConfig\.prompts is not an object/);
    expect(read({ prompts: { query: 1 } })).toThrow(/prompts\.query is not a string/);
    expect(read({ dimensions: [] })).toThrow(/dimensions is not a non-empty descending/);
    expect(read({ dimensions: [256, 768] })).toThrow(/dimensions is not a non-empty descending/);
    expect(read({ dimensions: [768, 768] })).toThrow(/dimensions is not a non-empty descending/);
    expect(read({ dimensions: [768, 0.5] })).toThrow(/dimensions is not a non-empty descending/);
    expect(read({ maxTokens: "2048" })).toThrow(/maxTokens is not a positive integer/);
  });
});

describe("the prompt select", () => {
  const config = readEmbeddingConfig(manifestWith(CONFIG), "text");

  it("offers no prompt first, then every declared prompt with its prefix", () => {
    expect(promptChoices(config, undefined)).toEqual([
      { value: "none", label: "none (no prefix)" },
      { value: "prompt:query", label: 'query — "task: search result | query: "' },
      { value: "prompt:document", label: 'document — "title: none | text: "' },
      { value: "prompt:STS", label: 'STS — "task: sentence similarity | query: "' },
    ]);
  });

  it("lists a row's prompt the model does not declare, marked, so the select shows what is sent", () => {
    const choices = promptChoices(config, "Retrieval");
    expect(choices).toHaveLength(5);
    expect(choices.at(-1)).toEqual({
      value: "prompt:Retrieval",
      label: "Retrieval (not declared by this model)",
    });
  });

  it("round-trips a prompt name through its value, even one spelled like the no-prompt value", () => {
    for (const prompt of [undefined, "query", "none", "prompt:x", ""]) {
      expect(promptOfValue(promptValue(prompt))).toBe(prompt);
    }
  });
});

describe("the dimension select", () => {
  const config = readEmbeddingConfig(manifestWith(CONFIG), "text");

  it("offers every declared width, the first marked as the full default", () => {
    expect(dimensionChoices(config).map(({ label }) => label)).toEqual([
      "768 (full, default)",
      "512",
      "256",
      "128",
    ]);
  });

  it("keeps a declared choice and falls back to the full width for anything else", () => {
    expect(effectiveDimension(config, 256)).toBe(256);
    expect(effectiveDimension(config, undefined)).toBe(768);
    expect(effectiveDimension(config, 300)).toBe(768);
  });
});
