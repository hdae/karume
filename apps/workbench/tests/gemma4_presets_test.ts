// Cases copied from the Playground's tests of its preset helpers (apps/playground/tests/
// presets_test.ts), for the function the workbench copied (src/gemma4/presets.ts).

import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { manifestPresets } from "../src/gemma4/presets.ts";

import { container } from "./helpers/session-ab-manifests.ts";

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
          weights: { net: { f16: container(`${name}/model.krm`) } },
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
