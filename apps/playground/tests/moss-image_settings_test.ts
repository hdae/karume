import { type Manifest, parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { manifestPresets } from "../src/lib/presets.ts";
import {
  buildRequest,
  effectiveRequest,
  initialForm,
  type MossDefaults,
  type MossForm,
  modelDefaults,
  usesNegativePrompt,
} from "../src/models/moss-image/settings.ts";

const hex = (character: string): string => character.repeat(64);

const container = (prefix: string) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [1, 2].map((part) => ({
      path: `${prefix}/model-0000${part}-of-00002.krm`,
      size: part === 1 ? 40 : 8,
      sha256: hex(part === 1 ? "a" : "b"),
    })),
  },
});

/**
 * A `karume/5` manifest shaped like `karume-moss-image`: one model, the two presets (the online one
 * the default) and upstream's `pipelineConfig` (the empty negative prompt).
 */
const mossLikeManifest = (pipelineConfig: unknown): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: "sft",
      models: {
        sft: {
          pipeline: "moss-image/1",
          assets: {
            tokenizer: { path: "sft/tokenizer/tokenizer.json", size: 16, sha256: hex("c") },
          },
          weights: {
            text_encoder: { i8: container("sft/text_encoder") },
            dit: { i8: container("sft/dit") },
          },
          quants: {
            "f32+dit8": {
              weights: { text_encoder: "i8", dit: "i8" },
              session: {},
              label: "int8 transformer, f32 compute",
            },
            "f32+dit8-a8-attn8-online": {
              weights: { text_encoder: "i8", dit: "i8" },
              session: {
                linearCompute: "a8",
                attentionCompute: "a8",
                attentionScoreStorage: "none",
              },
              label: "int8 transformer, int8 activations, online attention",
            },
          },
          defaultQuant: "f32+dit8-a8-attn8-online",
          pipelineConfig,
        },
      },
    }),
  );

const UPSTREAM_CONFIG = {
  scheduler: { shift: 5 },
  defaults: { width: 1024, height: 1024, steps: 50, guidance: 3, negativePrompt: "" },
};

const manifest = mossLikeManifest(UPSTREAM_CONFIG);
const defaults: MossDefaults = modelDefaults(manifest, "sft");
const noCfg: MossDefaults = { ...defaults, guidance: 1 };

const formWith = (patch: Partial<MossForm>): MossForm => ({
  ...initialForm("a red fox"),
  ...patch,
});

describe("modelDefaults", () => {
  it("reads the manifest's request defaults and the scheduler's shift", () => {
    expect(defaults).toEqual({
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 3,
      shift: 5,
      negativePrompt: "",
    });
  });

  it("leaves out what the manifest does not declare", () => {
    const partial = modelDefaults(mossLikeManifest({ defaults: { steps: 8 } }), "sft");
    expect(partial).toEqual({ steps: 8 });
  });

  it("refuses a model the manifest does not have", () => {
    expect(() => modelDefaults(manifest, "base")).toThrow(/no model 'base'/);
  });
});

describe("the presets of the manifest", () => {
  it("offers both quants with their labels and the online preset as the default", () => {
    expect(manifestPresets(manifest, "sft")).toEqual({
      presets: [
        { name: "f32+dit8", label: "int8 transformer, f32 compute", description: "" },
        {
          name: "f32+dit8-a8-attn8-online",
          label: "int8 transformer, int8 activations, online attention",
          description: "",
        },
      ],
      defaultQuant: "f32+dit8-a8-attn8-online",
    });
  });
});

describe("buildRequest", () => {
  it("sends only the prompt and the seed when every other field is empty", () => {
    expect(buildRequest(formWith({}), defaults)).toEqual({ prompt: "a red fox", seed: 42 });
  });

  it("sends every field the form sets", () => {
    expect(
      buildRequest(
        formWith({
          negativePrompt: "blurry",
          width: "1152",
          height: "896",
          steps: "28",
          guidance: "4.5",
          shift: "3",
          seed: "7",
        }),
        defaults,
      ),
    ).toEqual({
      prompt: "a red fox",
      negativePrompt: "blurry",
      width: 1152,
      height: 896,
      steps: 28,
      guidance: 4.5,
      shift: 3,
      seed: 7,
    });
  });

  it("does not send a negative prompt at guidance 1 (the pipeline would refuse it)", () => {
    expect(
      buildRequest(formWith({ negativePrompt: "blurry", guidance: "1" }), defaults).negativePrompt,
    ).toBeUndefined();
    expect(buildRequest(formWith({ negativePrompt: "blurry" }), noCfg).negativePrompt).toBe(
      undefined,
    );
  });

  it("leaves an empty negative prompt to the manifest's default", () => {
    expect(buildRequest(formWith({ negativePrompt: "  " }), defaults)).not.toHaveProperty(
      "negativePrompt",
    );
  });

  it("accepts guidance 0 and refuses a negative guidance", () => {
    expect(buildRequest(formWith({ guidance: "0" }), defaults).guidance).toBe(0);
    expect(() => buildRequest(formWith({ guidance: "-0.5" }), defaults)).toThrow(/Guidance/);
  });

  it("refuses a side that is not a positive multiple of 16", () => {
    expect(() => buildRequest(formWith({ width: "1000" }), defaults)).toThrow(/Width/);
    expect(() => buildRequest(formWith({ height: "0" }), defaults)).toThrow(/Height/);
    expect(() => buildRequest(formWith({ height: "-16" }), defaults)).toThrow(/Height/);
    expect(() => buildRequest(formWith({ width: "512.5" }), defaults)).toThrow(/Width/);
    expect(buildRequest(formWith({ width: "16", height: "2048" }), defaults)).toMatchObject({
      width: 16,
      height: 2048,
    });
  });

  it("names the field of any other value it rejects", () => {
    expect(() => buildRequest(formWith({ prompt: "  " }), defaults)).toThrow(/prompt/);
    expect(() => buildRequest(formWith({ steps: "0" }), defaults)).toThrow(/Steps/);
    expect(() => buildRequest(formWith({ steps: "2.5" }), defaults)).toThrow(/Steps/);
    expect(() => buildRequest(formWith({ shift: "0" }), defaults)).toThrow(/Shift/);
    expect(() => buildRequest(formWith({ shift: "abc" }), defaults)).toThrow(/Shift/);
    expect(() => buildRequest(formWith({ guidance: "abc" }), defaults)).toThrow(/Guidance/);
    expect(() => buildRequest(formWith({ seed: "-1" }), defaults)).toThrow(/Seed/);
    expect(() => buildRequest(formWith({ seed: "" }), defaults)).toThrow(/Seed/);
  });
});

describe("usesNegativePrompt", () => {
  it("follows the form's guidance, else the manifest's", () => {
    expect(usesNegativePrompt(formWith({}), defaults)).toBe(true);
    expect(usesNegativePrompt(formWith({ guidance: "1" }), defaults)).toBe(false);
    expect(usesNegativePrompt(formWith({}), noCfg)).toBe(false);
    expect(usesNegativePrompt(formWith({ guidance: "2" }), noCfg)).toBe(true);
  });
});

describe("effectiveRequest", () => {
  it("fills the fields the request leaves out from the manifest", () => {
    expect(effectiveRequest(buildRequest(formWith({}), defaults), defaults)).toEqual({
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 3,
      shift: 5,
      negativePrompt: "",
    });
  });

  it("keeps the request's values over the manifest's", () => {
    const request = buildRequest(
      formWith({ width: "512", height: "768", steps: "20", shift: "3", negativePrompt: "x" }),
      defaults,
    );
    expect(effectiveRequest(request, defaults)).toEqual({
      width: 512,
      height: 768,
      steps: 20,
      guidance: 3,
      shift: 3,
      negativePrompt: "x",
    });
  });

  it("names no negative prompt when no negative branch runs", () => {
    const request = buildRequest(formWith({ guidance: "1", negativePrompt: "x" }), defaults);
    expect(effectiveRequest(request, defaults).negativePrompt).toBeNull();
  });
});
