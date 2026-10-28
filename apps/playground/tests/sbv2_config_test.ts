import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { readSbv2Config, speakerModelChoices } from "../src/models/sbv2/config.ts";

const hex = (character: string): string => character.repeat(64);

/**
 * The pipelineConfig of a jvnv model holding speaker `speaker` (styles written out of row order),
 * with `defaults` overriding fields of its defaults block.
 */
const jvnvConfig = (
  speaker: string,
  defaults: Readonly<Record<string, unknown>> = {},
): Record<string, unknown> => ({
  styles: { Neutral: 0, Happy: 2, Angry: 1 },
  speakers: { [speaker]: 0 },
  maxTokens: 512,
  maxFrames: 4096,
  defaults: {
    speaker,
    style: "Neutral",
    styleWeight: 1,
    sdpRatio: 0.2,
    noiseScale: 0.6,
    noiseScaleW: 0.8,
    lengthScale: 1,
    ...defaults,
  },
});

/** A minimal `karume/5` manifest with the given models' pipelineConfig; the last model is the default. */
const manifestOf = (configs: Readonly<Record<string, Record<string, unknown>>>) =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: Object.keys(configs).at(-1),
      models: Object.fromEntries(
        Object.entries(configs).map(([name, pipelineConfig]) => [
          name,
          {
            pipeline: "sbv2/1",
            assets: {},
            weights: {
              net: {
                i8: {
                  container: {
                    descriptor: {
                      graph: { length: 8, sha256: hex("d") },
                      model: { length: 8, sha256: hex("e") },
                    },
                    parts: [
                      { path: `${name}/net-00001-of-00002.krm`, size: 40, sha256: hex("a") },
                      { path: `${name}/net-00002-of-00002.krm`, size: 8, sha256: hex("b") },
                    ],
                  },
                },
              },
            },
            quants: { i8: { weights: { net: "i8" }, session: {} } },
            defaultQuant: "i8",
            pipelineConfig,
          },
        ]),
      ),
    }),
  );

describe("readSbv2Config", () => {
  it("lists the speakers and the styles in table-row order with the distribution's defaults", () => {
    const manifest = manifestOf({ F1: jvnvConfig("jvnv-F1-jp") });
    expect(readSbv2Config(manifest, "F1")).toEqual({
      speakers: ["jvnv-F1-jp"],
      styles: ["Neutral", "Angry", "Happy"],
      maxTokens: 512,
      maxFrames: 4096,
      defaults: {
        speaker: "jvnv-F1-jp",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    });
  });

  it("refuses a model the manifest does not have", () => {
    const manifest = manifestOf({ F1: jvnvConfig("jvnv-F1-jp") });
    expect(() => readSbv2Config(manifest, "F9")).toThrow("The manifest has no model 'F9'");
  });

  it("refuses a model name that exists only on Object.prototype", () => {
    const manifest = manifestOf({ F1: jvnvConfig("jvnv-F1-jp") });
    expect(() => readSbv2Config(manifest, "constructor")).toThrow(
      "The manifest has no model 'constructor'",
    );
  });

  it("refuses a block that lacks the frame limit instead of reading an inherited field", () => {
    const { maxFrames: _dropped, ...withoutFrames } = jvnvConfig("jvnv-F1-jp");
    const manifest = manifestOf({ F1: withoutFrames });
    expect(() => readSbv2Config(manifest, "F1")).toThrow(
      "models.F1.pipelineConfig.maxFrames is not a finite number (got undefined)",
    );
  });

  it("refuses a default style that is not one of the styles, naming the field", () => {
    const manifest = manifestOf({ F1: jvnvConfig("jvnv-F1-jp", { style: "Whisper" }) });
    expect(() => readSbv2Config(manifest, "F1")).toThrow(
      'models.F1.pipelineConfig.defaults.style "Whisper" is not one of Neutral / Angry / Happy',
    );
  });

  it("refuses a knob default that is not a number", () => {
    const manifest = manifestOf({ F1: jvnvConfig("jvnv-F1-jp", { lengthScale: "1" }) });
    expect(() => readSbv2Config(manifest, "F1")).toThrow(
      'models.F1.pipelineConfig.defaults.lengthScale is not a finite number (got "1")',
    );
  });

  it("refuses an empty speaker table", () => {
    const manifest = manifestOf({ F1: { ...jvnvConfig("jvnv-F1-jp"), speakers: {} } });
    expect(() => readSbv2Config(manifest, "F1")).toThrow(
      "models.F1.pipelineConfig.speakers is empty",
    );
  });
});

describe("speakerModelChoices", () => {
  it("names each model with the speakers it holds and marks the default model", () => {
    const manifest = manifestOf({
      F1: jvnvConfig("jvnv-F1-jp"),
      M2: jvnvConfig("jvnv-M2-jp"),
    });
    expect(speakerModelChoices(manifest)).toEqual([
      { value: "F1", label: "F1 — jvnv-F1-jp" },
      { value: "M2", label: "M2 — jvnv-M2-jp (default)" },
    ]);
  });
});
