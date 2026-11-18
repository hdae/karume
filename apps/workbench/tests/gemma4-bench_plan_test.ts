import { parseManifest } from "@karume/hub";
import { describe, expect, it } from "vite-plus/test";

import {
  type ArmDraft,
  armLabel,
  type ArmResolution,
  BENCH_PROMPTS,
  defaultArmDrafts,
  integerGemvSession,
  meetsDistinctIdCondition,
  resolveArm,
  samplerOf,
  visitOrder,
} from "../src/gemma4-bench/plan.ts";

import { MOBILE, QAT, REGULAR } from "./helpers/gemma4-bench-manifests.ts";

const MANIFESTS = {
  "gemma4-qat": parseManifest(QAT),
  "gemma4-qat-mobile": parseManifest(MOBILE),
};

/** A mobile QAT E2B arm, tap on (the default preset, `i4-fast`, declares the packed activations). */
const draft = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  key: { mount: "gemma4-qat-mobile", model: "e2b" },
  tap: true,
  ...patch,
});

/** A QAT 12B arm (the model without a PLE), tap off. */
const twelve = (patch: Partial<ArmDraft> = {}): ArmDraft => ({
  key: { mount: "gemma4-qat", model: "12b" },
  tap: false,
  ...patch,
});

const armOf = (resolution: ArmResolution) => {
  if (!resolution.ok) throw new Error(resolution.error);
  return resolution.arm;
};

describe("the prompt set", () => {
  it("has one prompt on each side of the PLE distinct-id threshold", () => {
    expect(BENCH_PROMPTS.map(({ distinctIds }) => distinctIds)).toEqual(["below", "at-least"]);
  });

  it("checks a measured distinct count against the side its prompt is meant for", () => {
    const [few, many] = BENCH_PROMPTS;
    expect(meetsDistinctIdCondition(few, 31)).toBe(true);
    expect(meetsDistinctIdCondition(few, 32)).toBe(false);
    expect(meetsDistinctIdCondition(many, 32)).toBe(true);
    expect(meetsDistinctIdCondition(many, 31)).toBe(false);
  });
});

describe("the arms", () => {
  describe("by default", () => {
    it("takes each loaded distribution's default model, regular Gemma 4 first, tap on", () => {
      const drafts = defaultArmDrafts({
        "gemma4-qat-mobile": parseManifest(MOBILE),
        gemma4: parseManifest(REGULAR),
      });
      expect(drafts.map(({ key, tap }) => ({ ...key, tap }))).toEqual([
        { mount: "gemma4", model: "e2b", tap: true },
        { mount: "gemma4-qat-mobile", model: "e2b", tap: true },
      ]);
    });

    it("leaves out a distribution whose manifest is not loaded", () => {
      expect(defaultArmDrafts({ "gemma4-qat-mobile": parseManifest(MOBILE) })).toEqual([
        { key: { mount: "gemma4-qat-mobile", model: "e2b" }, tap: true },
      ]);
    });
  });

  describe("resolved against the manifests", () => {
    it("preselects the model's default preset with its label, and carries the distribution's declared sampler", () => {
      expect(armOf(resolveArm(draft(), MANIFESTS))).toMatchObject({
        quant: "i4-fast",
        quantLabel: "i4-fast label",
        isDefaultQuant: true,
        declaredSampler: { temperature: 1, topK: 64, topP: 0.95 },
      });
    });

    it("keeps a chosen preset and says it is not the default", () => {
      expect(armOf(resolveArm(draft({ quant: "i4" }), MANIFESTS))).toMatchObject({
        quant: "i4",
        isDefaultQuant: false,
      });
    });

    it("passes the Session and pipeline options of its switches, and the split as the arms parser left it", () => {
      for (const [name, given, session, pipeline, split] of [
        ["no switch", draft(), undefined, undefined, undefined],
        // The default preset declares the packed activations: integerGemv alone.
        ["integer", draft({ integerGemv: true }), { integerGemv: true }, undefined, undefined],
        [
          "integer on i4",
          draft({ quant: "i4", integerGemv: true }),
          {
            integerGemv: true,
            packedStaticQuantize: true,
            numerics: { ops: { linear: "reorder" } },
          },
          undefined,
          undefined,
        ],
        ["integer off", draft({ integerGemv: false }), undefined, undefined, undefined],
        // `i4` declares no level: the switch passes linear at reorder, and nothing else.
        [
          "linear reorder on i4",
          draft({ quant: "i4", linearReorder: true }),
          { numerics: { ops: { linear: "reorder" } } },
          undefined,
          undefined,
        ],
        [
          "linear reorder and integer on i4",
          draft({ quant: "i4", linearReorder: true, integerGemv: true }),
          {
            integerGemv: true,
            packedStaticQuantize: true,
            numerics: { ops: { linear: "reorder" } },
          },
          undefined,
          undefined,
        ],
        ["linear reorder off", draft({ linearReorder: false }), undefined, undefined, undefined],
        [
          "PLE on GPU, ahead 4",
          draft({ pleOnGpu: true, decodeAhead: "4" }),
          undefined,
          { pleResidency: "gpu", decodeAhead: 4 },
          undefined,
        ],
        [
          "PLE on GPU, ahead 1",
          draft({ pleOnGpu: true, decodeAhead: "1" }),
          undefined,
          { pleResidency: "gpu" },
          undefined,
        ],
        [
          "PLE on the host, ahead 1",
          draft({ pleOnGpu: false, decodeAhead: "1" }),
          undefined,
          undefined,
          undefined,
        ],
        // 12B has no PLE: it runs ahead without one.
        ["12B ahead 4", twelve({ decodeAhead: "4" }), undefined, { decodeAhead: 4 }, undefined],
        ["split", draft({ split: true }), undefined, undefined, true],
      ] as const) {
        const arm = armOf(resolveArm(given, MANIFESTS));
        expect([arm.session, arm.pipeline, arm.split], name).toEqual([session, pipeline, split]);
      }
    });

    it("refuses an arm it cannot run, saying why and naming what the manifest has", () => {
      for (const [given, error] of [
        [
          draft({ quant: "bf16" }),
          "The gemma4-qat-mobile model 'e2b' has no preset 'bf16'; its presets are i4, i4-gemvpar, i4-fast.",
        ],
        [
          draft({ key: { mount: "gemma4-qat-mobile", model: "12b" } }),
          "gemma4QatMobile.Pipeline does not accept the model '12b'.",
        ],
        [draft({ key: { mount: "gemma4", model: "e2b" } }), "The gemma4 manifest is not loaded."],
        [
          draft({ key: { mount: "gemma4-qat-mobile", model: "e9b" } }),
          "The gemma4-qat-mobile manifest has no model 'e9b'; its models are e2b, e4b, 12b.",
        ],
        [
          draft({ decodeAhead: "2" }),
          "Decode ahead needs PLE on GPU (each step's token stays on the GPU, so its PLE gather must run there).",
        ],
        [twelve({ pleOnGpu: true }), "12B has no PLE to place on the GPU."],
        // The arm would run what its base runs, under a label that says otherwise
        [
          draft({ linearReorder: true }),
          "The preset 'i4-fast' declares linear at the level reorder already.",
        ],
        [
          draft({ quant: "i4-gemvpar", linearReorder: true }),
          "The preset 'i4-gemvpar' declares linear at the level reorder already.",
        ],
      ] as const) {
        expect(resolveArm(given, MANIFESTS)).toEqual({ ok: false, error });
      }
    });
  });

  it("labels an arm by distribution, model and preset, then its integer GEMV or linear at reorder, PLE placement and depth, then the tap's own mark", () => {
    const mobile = { mount: "gemma4-qat-mobile", model: "e2b" } as const;
    const integer = { integerGemv: true } as const;
    for (const [arm, label] of [
      [{ mount: "gemma4", model: "e2b", quant: "i4-fast", tap: true }, "Gemma 4 e2b i4-fast"],
      [{ ...mobile, model: "e4b", quant: "i4", tap: false }, "mobile QAT e4b i4 (tap off)"],
      [{ ...mobile, quant: "i4", tap: true, split: true }, "mobile QAT e2b i4 (split)"],
      [
        { ...mobile, quant: "i4-fast", tap: true, session: integer },
        "mobile QAT e2b i4-fast (integer)",
      ],
      [
        { ...mobile, quant: "i4-fast", tap: true, split: true, session: integer },
        "mobile QAT e2b i4-fast (integer) (split)",
      ],
      [
        { ...mobile, quant: "i4", tap: false, session: integer },
        "mobile QAT e2b i4 (integer) (tap off)",
      ],
      [
        { ...mobile, quant: "i4", tap: true, pipeline: { pleResidency: "gpu", decodeAhead: 8 } },
        "mobile QAT e2b i4 (PLE GPU) (ahead 8)",
      ],
      [
        {
          mount: "gemma4",
          model: "e2b",
          quant: "bf16",
          tap: true,
          session: { numerics: { ops: { linear: "reorder" } } },
        },
        "Gemma 4 e2b bf16 (linear reorder)",
      ],
      // The integer GEMV's options hold linear at reorder too: its own mark says it
      [
        {
          ...mobile,
          quant: "i4",
          tap: true,
          session: { ...integer, numerics: { ops: { linear: "reorder" } } },
        },
        "mobile QAT e2b i4 (integer)",
      ],
      [
        { ...mobile, quant: "i4", tap: false, session: integer, pipeline: { pleResidency: "gpu" } },
        "mobile QAT e2b i4 (integer) (PLE GPU) (tap off)",
      ],
    ] as const) {
      expect(armLabel(arm)).toBe(label);
    }
  });
});

describe("the integer GEMV switch's explicit Session options", () => {
  it("is integerGemv alone when the preset declares the packed activations", () => {
    expect(
      integerGemvSession({ numerics: { level: "reorder" }, packedStaticQuantize: true }),
    ).toEqual({ integerGemv: true });
  });

  it("adds the packed activations and the parallel GEMV a preset does not declare", () => {
    expect(integerGemvSession({})).toEqual({
      integerGemv: true,
      packedStaticQuantize: true,
      numerics: { ops: { linear: "reorder" } },
    });
    expect(integerGemvSession({ numerics: { ops: { linear: "reorder" } } })).toEqual({
      integerGemv: true,
      packedStaticQuantize: true,
    });
    expect(
      integerGemvSession({
        numerics: { ops: { linear: "reorder" } },
        packedStaticQuantize: false,
      }),
    ).toEqual({ integerGemv: true, packedStaticQuantize: true });
    expect(
      integerGemvSession({
        numerics: { level: "reorder", ops: { linear: "exact" } },
        packedStaticQuantize: true,
      }),
    ).toEqual({ integerGemv: true, numerics: { ops: { linear: "reorder" } } });
  });
});

describe("the sampler", () => {
  const declared = { temperature: 1, topK: 64, topP: 0.95 };

  // The default (greedy) is the entry's `sampler` parameter: gemma4-bench_entry_test.ts.
  it("is greedy at temperature 0", () => {
    expect(samplerOf("greedy", declared)).toEqual({ temperature: 0 });
  });

  it("takes the distribution's declared sampler with seed 0 when asked", () => {
    expect(samplerOf("distribution", declared)).toEqual({ ...declared, seed: 0 });
  });

  it("stays greedy when the distribution declares none", () => {
    expect(samplerOf("distribution", undefined)).toEqual({ temperature: 0 });
  });
});

describe("the visit order", () => {
  it("is ABBA for two arms", () => {
    expect(
      visitOrder(2, 4)
        .map(({ arm }) => "AB"[arm])
        .join(""),
    ).toBe("ABBAABBA");
  });

  it("reverses every odd round for more arms, numbering visits and rounds", () => {
    expect(visitOrder(3, 2)).toEqual([
      { visit: 0, round: 0, arm: 0 },
      { visit: 1, round: 0, arm: 1 },
      { visit: 2, round: 0, arm: 2 },
      { visit: 3, round: 1, arm: 2 },
      { visit: 4, round: 1, arm: 1 },
      { visit: 5, round: 1, arm: 0 },
    ]);
  });

  it("visits every arm once per round", () => {
    const visits = visitOrder(3, 5);
    for (let round = 0; round < 5; round += 1) {
      expect(
        visits
          .filter((visit) => visit.round === round)
          .map(({ arm }) => arm)
          .sort((left, right) => left - right),
      ).toEqual([0, 1, 2]);
    }
  });
});
