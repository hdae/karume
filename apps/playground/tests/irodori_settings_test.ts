import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import { pageAt } from "../src/models/registry.ts";
import {
  buildRequest,
  changedKeyFields,
  DEFAULT_MOUNT,
  emptyCaptionText,
  FormError,
  initialForm,
  type IrodoriForm,
  linearComputeNote,
  MODEL_CHOICES,
  readIrodoriConfig,
  readPresetFacts,
  referenceLimits,
  resolveLinearCompute,
  samePipelineKey,
  samplerText,
} from "../src/models/irodori/settings.ts";

const PIPELINE_CONFIG = {
  maxTextLen: 256,
  maxCaptionLen: 512,
  speakerRows: 751,
  ditSymMax: 750,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
  latentDim: 32,
  speakerPatchSize: 4,
  steps: 40,
  minSeconds: 0.5,
  maxSeconds: 30,
};

const manifestWith = (pipelineConfig: Record<string, unknown>) =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "test",
      defaultModel: "v4.1-small",
      models: {
        "v4.1-small": {
          pipeline: "irodori/1",
          weights: {
            dit: {
              i8: {
                container: {
                  descriptor: {
                    graph: { length: 8, sha256: "d".repeat(64) },
                    model: { length: 8, sha256: "e".repeat(64) },
                  },
                  parts: [
                    { path: "dit-00001-of-00002.krm", size: 40, sha256: "a".repeat(64) },
                    { path: "dit-00002-of-00002.krm", size: 8, sha256: "b".repeat(64) },
                  ],
                },
              },
            },
          },
          assets: {},
          quants: { i8: { weights: { dit: "i8" }, session: {} } },
          defaultQuant: "i8",
          pipelineConfig,
        },
      },
    }),
  );

/** The MeanFlow declaration (ADR 0150): its own step count, the flow key, none of the CFG keys. */
const MEANFLOW_CONFIG = { ...PIPELINE_CONFIG, flowParameterization: "meanflow", steps: 4 };

const form = (patch: Partial<IrodoriForm> = {}): IrodoriForm => ({ ...initialForm(), ...patch });

describe("MODEL_CHOICES", () => {
  it("offers exactly the mounts the page registry lists for the Irodori page", () => {
    const mounts = pageAt("/irodori")?.mounts ?? [];
    expect(MODEL_CHOICES.map(({ value }) => value).sort()).toEqual([...mounts].sort());
  });

  it("offers the MeanFlow model next to the two rectified-flow ones, v4.1 staying the default", () => {
    expect(MODEL_CHOICES.map(({ value }) => value)).toContain("irodori-v4.1-mf");
    expect(DEFAULT_MOUNT).toBe("irodori-v4.1");
  });
});

const reference = { data: new Float32Array(20_000), sampleRate: 48000 };

describe("initialForm", () => {
  it("starts on v4.1-small with a caption, seed 0 and stage events on", () => {
    const initial = initialForm();
    expect(initial.mount).toBe("irodori-v4.1");
    expect(initial.voice).toBe("caption");
    expect(initial.caption).not.toBe("");
    expect(initial.text).not.toBe("");
    expect(initial.seed).toBe("0");
    expect(initial.stageEvents).toBe(true);
  });

  it("starts with the rare-token split box checked (the request leaves it to the model)", () => {
    expect(initialForm().splitRareTokens).toBe(true);
  });
});

describe("buildRequest", () => {
  it("sends the caption in caption mode and no speaker", () => {
    const request = buildRequest(form({ caption: "voice", text: "text" }), reference);
    expect(request).toEqual({ text: "text", caption: "voice", seed: 0 });
  });

  it("sends no caption for an empty one (no voice design)", () => {
    const request = buildRequest(form({ caption: "  " }), undefined);
    expect(request.caption).toBeUndefined();
    expect(request.speaker).toBeUndefined();
  });

  it("sends only the reference in reference mode, even with a caption typed", () => {
    const request = buildRequest(form({ voice: "reference", caption: "voice" }), reference);
    expect(request.caption).toBeUndefined();
    expect(request.speaker).toEqual({ audio: reference });
  });

  it("refuses reference mode without a reference", () => {
    expect(() => buildRequest(form({ voice: "reference" }), undefined)).toThrow(FormError);
  });

  it("refuses a blank text", () => {
    expect(() => buildRequest(form({ text: " \n" }), undefined)).toThrow("Enter the text");
  });

  it("sends the text as typed (the pipeline normalizes it)", () => {
    expect(buildRequest(form({ text: " a b " }), undefined).text).toBe(" a b ");
  });

  it("parses the seed, the duration and the codec tile", () => {
    const request = buildRequest(
      form({ seed: "42", durationSeconds: "3.5", codecTileFrames: "64" }),
      undefined,
    );
    expect(request).toMatchObject({ seed: 42, durationSeconds: 3.5, codecTileFrames: 64 });
  });

  it("leaves duration and codec tile out when empty", () => {
    const request = buildRequest(form({ durationSeconds: " ", codecTileFrames: "" }), undefined);
    expect("durationSeconds" in request).toBe(false);
    expect("codecTileFrames" in request).toBe(false);
  });

  it.each([
    ["seed", { seed: "" }],
    ["seed", { seed: "1.5" }],
    ["seed", { seed: "x" }],
    ["duration", { durationSeconds: "abc" }],
    ["codec tile", { codecTileFrames: "10.5" }],
  ] as const)("refuses a malformed %s", (_, patch) => {
    expect(() => buildRequest(form(patch), undefined)).toThrow(FormError);
  });

  it("leaves splitRareTokens out while the box is checked (the model's default)", () => {
    const request = buildRequest(form({ splitRareTokens: true }), undefined);
    expect("splitRareTokens" in request).toBe(false);
  });

  it("sends splitRareTokens false when the box is unchecked, and never true", () => {
    expect(buildRequest(form({ splitRareTokens: false }), undefined).splitRareTokens).toBe(false);
    for (const splitRareTokens of [true, false]) {
      expect(buildRequest(form({ splitRareTokens }), undefined).splitRareTokens).not.toBe(true);
    }
  });

  it("leaves the range checks to the pipeline (a negative seed passes the form)", () => {
    expect(buildRequest(form({ seed: "-1" }), undefined).seed).toBe(-1);
  });
});

describe("readIrodoriConfig", () => {
  it("reads the numbers the page needs from pipelineConfig", () => {
    const config = readIrodoriConfig(manifestWith(PIPELINE_CONFIG));
    expect(config).toEqual({
      sampleRate: 48000,
      maxTextLen: 256,
      maxCaptionLen: 512,
      hopLength: 1920,
      speakerPatchSize: 4,
      speakerRows: 751,
      codecHaloFrames: 8,
      minSeconds: 0.5,
      maxSeconds: 30,
      steps: 40,
      flow: "rf_velocity",
    });
  });

  it("reads the MeanFlow declaration's flow and step count from the manifest", () => {
    const config = readIrodoriConfig(manifestWith(MEANFLOW_CONFIG));
    expect(config.flow).toBe("meanflow");
    expect(config.steps).toBe(4);
  });

  it("takes an explicit rf_velocity as the rectified flow", () => {
    const config = readIrodoriConfig(
      manifestWith({ ...PIPELINE_CONFIG, flowParameterization: "rf_velocity" }),
    );
    expect(config.flow).toBe("rf_velocity");
  });

  it.each([["x0"], [1], [null]])("fails loudly on the flow parameterization %j", (value) => {
    expect(() =>
      readIrodoriConfig(manifestWith({ ...PIPELINE_CONFIG, flowParameterization: value })),
    ).toThrow("pipelineConfig.flowParameterization");
  });

  it("fails loudly on a missing field", () => {
    const { sampleRate: _, ...rest } = PIPELINE_CONFIG;
    expect(() => readIrodoriConfig(manifestWith(rest))).toThrow("pipelineConfig.sampleRate");
  });

  it("fails loudly on an unknown model", () => {
    expect(() => readIrodoriConfig(manifestWith(PIPELINE_CONFIG), "v9")).toThrow("no model 'v9'");
  });
});

describe("samplerText", () => {
  it("names guidance for the rectified flow and its absence for MeanFlow", () => {
    expect(samplerText("rf_velocity")).toContain("with classifier-free guidance");
    expect(samplerText("meanflow")).toContain("no classifier-free guidance");
  });
});

describe("emptyCaptionText", () => {
  it("says the caption's guidance goes off only on the rectified flow", () => {
    expect(emptyCaptionText("rf_velocity")).toContain("guidance is off");
    expect(emptyCaptionText("meanflow")).toContain("runs no guidance");
  });

  it("claims nothing about guidance before the manifest is known", () => {
    expect(emptyCaptionText(undefined)).not.toContain("guidance");
  });
});

describe("readPresetFacts", () => {
  it("names each quant's declared linearCompute and its dit weights label", () => {
    expect(readPresetFacts(manifestWith(PIPELINE_CONFIG))).toEqual({
      i8: { declaredLinearCompute: undefined, ditWeights: "i8" },
    });
  });

  it("fails loudly on an unknown model", () => {
    expect(() => readPresetFacts(manifestWith(PIPELINE_CONFIG), "v9")).toThrow("no model 'v9'");
  });
});

describe("resolveLinearCompute", () => {
  it("takes the page's override over the preset's declaration", () => {
    expect(resolveLinearCompute("f32", "a8")).toBe("f32");
    expect(resolveLinearCompute("a8", undefined)).toBe("a8");
  });

  it("takes the preset's declaration, and f32 when it declares nothing", () => {
    expect(resolveLinearCompute("preset", "a8")).toBe("a8");
    expect(resolveLinearCompute("preset", undefined)).toBe("f32");
  });
});

describe("linearComputeNote", () => {
  it("warns when a8 meets DiT weights labelled float", () => {
    expect(linearComputeNote("a8", "f16")).toContain("'f16'");
    expect(linearComputeNote("a8", "f32")).toContain("f32 path");
  });

  it("says nothing for integer weights or another compute", () => {
    expect(linearComputeNote("a8", "i8")).toBeUndefined();
    expect(linearComputeNote("a8", "i4")).toBeUndefined();
    expect(linearComputeNote("f32", "f16")).toBeUndefined();
  });
});

describe("referenceLimits", () => {
  it("gives the documented 13,441-sample minimum and 120 s maximum of the small distributions", () => {
    const limits = referenceLimits(readIrodoriConfig(manifestWith(PIPELINE_CONFIG)));
    expect(limits.minSamples).toBe(13_441);
    expect(limits.maxSamples / 48000).toBe(120);
  });
});

describe("samePipelineKey", () => {
  const gpu = {};
  const source = {};
  const key = {
    gpu,
    source,
    mount: "irodori-v4.1",
    model: "v4.1-small",
    quant: "i8-a8",
    linearCompute: "preset",
  } as const;

  it("reuses a pipeline only for the same GPU and source objects and the same choices", () => {
    expect(samePipelineKey(key, { ...key })).toBe(true);
    expect(samePipelineKey(key, { ...key, gpu: {} })).toBe(false);
    expect(samePipelineKey(key, { ...key, source: {} })).toBe(false);
    expect(samePipelineKey(key, { ...key, quant: "f16" })).toBe(false);
    expect(samePipelineKey(key, { ...key, linearCompute: "a8" })).toBe(false);
    expect(samePipelineKey(key, { ...key, mount: "irodori-v4", model: "v4-small" })).toBe(false);
  });

  it("names the fields that changed, in key order", () => {
    expect(changedKeyFields(key, { ...key })).toEqual([]);
    expect(changedKeyFields(key, { ...key, linearCompute: "f32", gpu: {} })).toEqual([
      "gpu",
      "linearCompute",
    ]);
  });
});
