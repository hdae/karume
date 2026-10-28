import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  modelInputSize,
  type RunReadiness,
  runBlocker,
  settleForm,
} from "../src/models/depth-anything/settings.ts";

const hex = (character: string): string => character.repeat(64);

/** Two models; `small` is the default with `f32` as its default preset, `base` defaults to `f16`. */
const manifest = parseManifest(
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "small",
    models: Object.fromEntries(
      ["small", "base"].map((name) => [
        name,
        {
          pipeline: "depth-anything/1",
          assets: {},
          weights: {
            depth: {
              f32: {
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
            f32: { weights: { depth: "f32" }, session: {} },
            f16: { weights: { depth: "f32" }, session: {} },
          },
          defaultQuant: name === "small" ? "f32" : "f16",
          pipelineConfig: { imageWidth: 518, imageHeight: 518 },
        },
      ]),
    ),
  }),
);

describe("settleForm", () => {
  it("fills an empty form with the default model and its default preset", () => {
    expect(settleForm({ model: "", quant: "" }, manifest)).toEqual({
      model: "small",
      quant: "f32",
    });
  });

  it("gives a newly chosen model its own default preset", () => {
    expect(settleForm({ model: "base", quant: "" }, manifest)).toEqual({
      model: "base",
      quant: "f16",
    });
  });

  it("keeps a valid choice as it is (the same object, so nothing re-renders for it)", () => {
    const form = { model: "base", quant: "f32" };
    expect(settleForm(form, manifest)).toBe(form);
  });

  it("replaces a model or preset the manifest does not have with the defaults", () => {
    expect(settleForm({ model: "large", quant: "f32" }, manifest)).toEqual({
      model: "small",
      quant: "f32",
    });
    expect(settleForm({ model: "small", quant: "i8" }, manifest)).toEqual({
      model: "small",
      quant: "f32",
    });
  });
});

describe("modelInputSize", () => {
  it("reads the square input size of the pipeline config", () => {
    expect(modelInputSize({ imageWidth: 518, imageHeight: 518 })).toEqual({
      width: 518,
      height: 518,
    });
  });

  it("says nothing when the config does not state it as positive integers", () => {
    expect(modelInputSize({})).toBeUndefined();
    expect(modelInputSize({ imageWidth: 518, imageHeight: "518" })).toBeUndefined();
    expect(modelInputSize({ imageWidth: 0, imageHeight: 518 })).toBeUndefined();
    expect(modelInputSize({ imageWidth: 518.5, imageHeight: 518 })).toBeUndefined();
  });
});

describe("runBlocker", () => {
  const ready: RunReadiness = {
    gpuReady: true,
    manifestReady: true,
    formSettled: true,
    hasImage: true,
    running: false,
  };

  it("lets a run start when everything is there", () => {
    expect(runBlocker(ready)).toBeUndefined();
  });

  it("names what is missing", () => {
    expect(runBlocker({ ...ready, hasImage: false })).toMatch(/image/);
    expect(runBlocker({ ...ready, gpuReady: false })).toMatch(/GPU/);
    expect(runBlocker({ ...ready, manifestReady: false })).toMatch(/manifest/);
  });

  it("refuses a second run while one is in progress", () => {
    expect(runBlocker({ ...ready, running: true })).toMatch(/in progress/);
  });
});
