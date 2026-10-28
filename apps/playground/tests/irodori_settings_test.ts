import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  buildRequest,
  changedKeyFields,
  FormError,
  initialForm,
  type IrodoriForm,
  linearComputeNote,
  readIrodoriConfig,
  readPresetFacts,
  referenceLimits,
  resolveLinearCompute,
  samePipelineKey,
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

const form = (patch: Partial<IrodoriForm> = {}): IrodoriForm => ({ ...initialForm(), ...patch });

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
    });
  });

  it("fails loudly on a missing field", () => {
    const { sampleRate: _, ...rest } = PIPELINE_CONFIG;
    expect(() => readIrodoriConfig(manifestWith(rest))).toThrow("pipelineConfig.sampleRate");
  });

  it("fails loudly on an unknown model", () => {
    expect(() => readIrodoriConfig(manifestWith(PIPELINE_CONFIG), "v9")).toThrow("no model 'v9'");
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
