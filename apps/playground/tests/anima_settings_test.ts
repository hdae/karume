import { type Manifest, parseManifest } from "@karume/hub";
import { ModelInputError } from "@karume/models/anima";
import { describe, expect, it } from "vite-plus/test";

import {
  animaModelChoices,
  buildRequest,
  crossRepoMounts,
  defaultModelRef,
  effectiveRequest,
  initialForm,
  type ModelDefaults,
  modelDefaults,
  modelKey,
  parseModelKey,
  usesNegativePrompt,
  type AnimaForm,
} from "../src/models/anima/settings.ts";

const hex = (character: string): string => character.repeat(64);
const OFFICIAL = "hdae/karume-anima";
const REVISION = "a".repeat(40);

const container = (prefix: string, crossRepo: boolean) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [1, 2].map((part) => ({
      path: `${prefix}/model-0000${part}-of-00002.krm`,
      size: part === 1 ? 40 : 8,
      sha256: hex(part === 1 ? "a" : "b"),
      ...(crossRepo ? { repo: OFFICIAL, revision: REVISION } : {}),
    })),
  },
});

/**
 * A `karume/5` manifest shaped like the Anima ones: per model a transformer of its own and a text
 * encoder (cross-repo to the official repository when `crossRepo`), a tokenizer asset likewise, and
 * an Anima-like `pipelineConfig`.
 */
const animaLikeManifest = (
  models: readonly string[],
  options: { readonly crossRepo: boolean; readonly guidance: number; readonly negative?: string },
): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: models[0],
      models: Object.fromEntries(
        models.map((name) => [
          name,
          {
            pipeline: "anima/1",
            assets: {
              tokenizer: {
                path: "shared/tokenizer/tokenizer.json",
                size: 16,
                sha256: hex("c"),
                ...(options.crossRepo ? { repo: OFFICIAL, revision: REVISION } : {}),
              },
            },
            weights: {
              transformer: { f16: container(`${name}/transformer`, false) },
              text_encoder: { f16: container("shared/text_encoder", options.crossRepo) },
            },
            quants: {
              f16: { weights: { transformer: "f16", text_encoder: "f16" }, session: {} },
              bf16: {
                weights: { transformer: "f16", text_encoder: "f16" },
                session: {},
                label: "Original weights (bf16)",
              },
            },
            defaultQuant: "f16",
            pipelineConfig: {
              scheduler: { type: "euler", shift: 3, numTrainTimesteps: 1000 },
              defaults: {
                steps: 8,
                guidanceScale: options.guidance,
                resolution: { width: 1024, height: 1024 },
                ...(options.negative === undefined ? {} : { negativePrompt: options.negative }),
              },
            },
          },
        ]),
      ),
    }),
  );

const official = animaLikeManifest(["turbo", "base"], {
  crossRepo: false,
  guidance: 1,
  negative: "low quality",
});
const extra = animaLikeManifest(["wai"], { crossRepo: true, guidance: 4 });

const turboDefaults: ModelDefaults = modelDefaults(official, "turbo");
const cfgDefaults: ModelDefaults = { ...turboDefaults, guidanceScale: 4 };

const formWith = (patch: Partial<AnimaForm>): AnimaForm => ({ ...initialForm("1girl"), ...patch });

describe("model keys", () => {
  it("round-trips a mount and a model", () => {
    const ref = { mount: "anima-extra", model: "anima-wai-v1.0" } as const;
    expect(parseModelKey(modelKey(ref))).toEqual(ref);
  });

  it("refuses a key of no known mount", () => {
    expect(parseModelKey("wan22/x")).toBeUndefined();
    expect(parseModelKey("anima")).toBeUndefined();
  });
});

describe("animaModelChoices", () => {
  it("lists both mounts' models in mount order, each mount's default marked", () => {
    expect(animaModelChoices({ anima: official, "anima-extra": extra })).toEqual([
      { value: "anima/turbo", label: "Official · turbo (default)" },
      { value: "anima/base", label: "Official · base" },
      { value: "anima-extra/wai", label: "Fine-tune · wai (default)" },
    ]);
  });

  it("offers only the manifests that loaded", () => {
    expect(animaModelChoices({ "anima-extra": extra }).map(({ value }) => value)).toEqual([
      "anima-extra/wai",
    ]);
  });
});

describe("defaultModelRef", () => {
  it("prefers the official default model", () => {
    expect(defaultModelRef({ anima: official, "anima-extra": extra })).toEqual({
      mount: "anima",
      model: "turbo",
    });
  });

  it("falls back to the fine-tunes' default without the official manifest", () => {
    expect(defaultModelRef({ "anima-extra": extra })).toEqual({
      mount: "anima-extra",
      model: "wai",
    });
  });

  it("has none before any manifest loaded", () => {
    expect(defaultModelRef({})).toBeUndefined();
  });
});

describe("crossRepoMounts", () => {
  it("maps the repository a fine-tune references to the mount that holds it", () => {
    expect(crossRepoMounts(extra, "wai", "f16", { [OFFICIAL]: "anima" })).toEqual({
      [OFFICIAL]: "anima",
    });
  });

  it("is empty for a model that references no other repository", () => {
    expect(crossRepoMounts(official, "turbo", "bf16", { [OFFICIAL]: "anima" })).toEqual({});
  });

  it("fails loudly for a referenced repository no mount holds", () => {
    expect(() => crossRepoMounts(extra, "wai", "f16", {})).toThrow(
      /references hdae\/karume-anima, which no mount/,
    );
  });
});

describe("modelDefaults", () => {
  it("reads the manifest's request defaults and scheduler", () => {
    expect(turboDefaults).toEqual({
      steps: 8,
      guidanceScale: 1,
      resolution: { width: 1024, height: 1024 },
      negativePrompt: "low quality",
      sampler: "euler",
      shift: 3,
    });
  });

  it("leaves out what the manifest does not declare", () => {
    expect(modelDefaults(extra, "wai").negativePrompt).toBeUndefined();
  });
});

describe("buildRequest", () => {
  it("sends only the prompt, the seed and the residency when every other field is empty", () => {
    expect(buildRequest(formWith({}), turboDefaults)).toEqual({
      prompt: "1girl",
      seed: 42,
      residency: "per-stage",
    });
  });

  it("sends every field the form sets", () => {
    expect(
      buildRequest(
        formWith({
          guidanceScale: "4.5",
          negativePrompt: "blurry",
          steps: "20",
          resolution: "1344x768",
          sampler: "dpmpp-2m",
          seed: "7",
          residency: "transformer",
        }),
        turboDefaults,
      ),
    ).toEqual({
      prompt: "1girl",
      negativePrompt: "blurry",
      guidanceScale: 4.5,
      steps: 20,
      resolution: { width: 1344, height: 768 },
      sampler: "dpmpp-2m",
      seed: 7,
      residency: "transformer",
    });
  });

  it("does not send a negative prompt at guidance 1 (the pipeline would refuse it)", () => {
    const request = buildRequest(formWith({ negativePrompt: "blurry" }), turboDefaults);
    expect(request.negativePrompt).toBeUndefined();
  });

  it("sends the negative prompt when the manifest's guidance computes the uncond side", () => {
    expect(buildRequest(formWith({ negativePrompt: "blurry" }), cfgDefaults).negativePrompt).toBe(
      "blurry",
    );
  });

  it("refuses an empty negative prompt when CFG runs and the model has no default one", () => {
    const noDefault = modelDefaults(extra, "wai");
    expect(() => buildRequest(formWith({}), noDefault)).toThrow(/no default negative prompt/);
    expect(buildRequest(formWith({ negativePrompt: "x" }), noDefault).negativePrompt).toBe("x");
  });

  it("names the field of a value it rejects", () => {
    expect(() => buildRequest(formWith({ prompt: "  " }), turboDefaults)).toThrow(/prompt/);
    expect(() => buildRequest(formWith({ steps: "1" }), turboDefaults)).toThrow(/Steps/);
    expect(() => buildRequest(formWith({ steps: "2.5" }), turboDefaults)).toThrow(/Steps/);
    expect(() => buildRequest(formWith({ seed: "-1" }), turboDefaults)).toThrow(/Seed/);
    expect(() => buildRequest(formWith({ guidanceScale: "abc" }), turboDefaults)).toThrow(
      /Guidance/,
    );
  });

  it("rejects a resolution outside the pipeline's accepted set with its own error", () => {
    expect(() => buildRequest(formWith({ resolution: "1000x1000" }), turboDefaults)).toThrow(
      ModelInputError,
    );
    expect(() => buildRequest(formWith({ resolution: "256" }), turboDefaults)).toThrow(
      ModelInputError,
    );
  });

  it("accepts the square shorthand", () => {
    expect(buildRequest(formWith({ resolution: "768" }), turboDefaults).resolution).toEqual({
      width: 768,
      height: 768,
    });
  });
});

describe("usesNegativePrompt", () => {
  it("follows the form's guidance, else the manifest's", () => {
    expect(usesNegativePrompt(formWith({}), turboDefaults)).toBe(false);
    expect(usesNegativePrompt(formWith({ guidanceScale: "3" }), turboDefaults)).toBe(true);
    expect(usesNegativePrompt(formWith({}), cfgDefaults)).toBe(true);
    expect(usesNegativePrompt(formWith({ guidanceScale: "1" }), cfgDefaults)).toBe(false);
  });
});

describe("effectiveRequest", () => {
  it("fills the fields the request leaves out from the manifest", () => {
    expect(effectiveRequest(buildRequest(formWith({}), turboDefaults), turboDefaults)).toEqual({
      steps: 8,
      guidanceScale: 1,
      resolution: "1024x1024",
      sampler: "euler",
      negativePrompt: null,
    });
  });

  it("names the negative prompt a CFG run uses: the request's, else the manifest's", () => {
    const withCfg = buildRequest(formWith({ guidanceScale: "4" }), turboDefaults);
    expect(effectiveRequest(withCfg, turboDefaults).negativePrompt).toBe("low quality");
    const own = buildRequest(formWith({ guidanceScale: "4", negativePrompt: "x" }), turboDefaults);
    expect(effectiveRequest(own, turboDefaults).negativePrompt).toBe("x");
  });

  it("keeps the request's values over the manifest's", () => {
    const request = buildRequest(
      formWith({ steps: "30", resolution: "832x1216", sampler: "dpmpp-2m" }),
      turboDefaults,
    );
    expect(effectiveRequest(request, turboDefaults)).toMatchObject({
      steps: 30,
      resolution: "832x1216",
      sampler: "dpmpp-2m",
    });
  });
});
