import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { manifestPresets, modelChoices, presetChoices } from "../src/lib/presets.ts";

const hex = (character: string): string => character.repeat(64);

/** A minimal `karume/5` manifest with two models; `b` is the default, its `i8` preset is labelled. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "b",
    models: Object.fromEntries(
      ["a", "b"].map((name) => [
        name,
        {
          pipeline: "test/1",
          assets: {},
          weights: {
            net: {
              f16: {
                container: {
                  descriptor: {
                    graph: { length: 8, sha256: hex("d") },
                    model: { length: 8, sha256: hex("e") },
                  },
                  parts: [
                    { path: `${name}/model-00001-of-00002.krm`, size: 40, sha256: hex("a") },
                    { path: `${name}/model-00002-of-00002.krm`, size: 8, sha256: hex("b") },
                  ],
                },
              },
            },
          },
          quants: {
            f16: { weights: { net: "f16" }, session: {} },
            i8: {
              weights: { net: "f16" },
              session: {},
              label: "int8 weights",
              description: "Weights in int8.",
            },
          },
          defaultQuant: name === "b" ? "i8" : "f16",
          pipelineConfig: {},
        },
      ]),
    ),
  }),
);

describe("manifestPresets", () => {
  it("lists the default model's presets with label and description, and its default", () => {
    expect(manifestPresets(manifest)).toEqual({
      presets: [
        { name: "f16", label: "f16", description: "" },
        { name: "i8", label: "int8 weights", description: "Weights in int8." },
      ],
      defaultQuant: "i8",
    });
  });

  it("reads the model it is given", () => {
    expect(manifestPresets(manifest, "a").defaultQuant).toBe("f16");
  });

  it("fails loudly for a model the manifest does not have", () => {
    expect(() => manifestPresets(manifest, "c")).toThrow(/no model 'c'/);
  });
});

describe("presetChoices", () => {
  it("marks the default preset", () => {
    const { presets, defaultQuant } = manifestPresets(manifest);
    expect(presetChoices(presets, defaultQuant)).toEqual([
      { value: "f16", label: "f16 — f16" },
      { value: "i8", label: "i8 — int8 weights (default)" },
    ]);
  });
});

describe("modelChoices", () => {
  it("keeps the manifest's order and marks the default model", () => {
    expect(modelChoices(manifest)).toEqual([
      { value: "a", label: "a" },
      { value: "b", label: "b (default)" },
    ]);
  });
});
