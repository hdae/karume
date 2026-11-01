/**
 * The MOSS-Image pipeline's manifest resolution (`../src/moss/pipeline.ts`, `../src/moss/config.ts`
 * and `mossPresetOf` of `../src/moss/preset.ts` — ADR 0138 addendum of 2026-11-01). CPU only, no
 * asset: a synthetic `karume/5` manifest shaped like the recipe's (`tools/export-recipes/moss/
 * distribution.py` — seven components, the tokenizer asset, the two presets, upstream's defaults).
 *
 * Pinned:
 *
 * 1. **The preset**: without `quant` the model's `defaultQuant`, with it that preset; its knobs are
 *    the family table's, and a `session` that declares other knobs under the name, a knob the
 *    family does not accept, a preset the table does not know, or a quant / model not in the
 *    manifest fail loudly, naming what there is.
 * 2. **The defaults** of `pipelineConfig` fill exactly the fields a request leaves out; the
 *    negative prompt only while guidance ≠ 1; the seed 0 only when neither a seed nor latents is
 *    given (both pass through, so the acceptance check refuses them).
 * 3. **The schema**: unknown keys, missing fields and values outside the request checks' ranges
 *    fail as plain `Error`s (an asset inconsistency, not an input error).
 * 4. **The contract**: another pipeline name or major, and a manifest without the tokenizer asset.
 *
 * Faults injected while writing these (each turned at least one case red, 2026-11-01): the default
 * quant ignored (`options.quant ?? "f32+dit8"`), the knob comparison removed from `mossPresetOf`,
 * the default negative applied at guidance 1, the seed default applied over given latents, the major
 * check dropped, the `negativePrompt` type check dropped, the tokenizer-asset check dropped.
 */

import { describe, expect, it } from "vite-plus/test";
import { assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import { type MossPipelineConfig, parseMossPipelineConfig } from "../src/moss/config.ts";
import { mossRequestOf, mossSelectionOf } from "../src/moss/pipeline.ts";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_REFERENCE_PRESET,
  mossPreset,
  mossPresetOf,
} from "../src/moss/preset.ts";
import { declaredContainer } from "./helpers/container-fixture.ts";

const MODEL = "sft";

/** The recipe's components and their storage labels (`MOSS_STORAGE_LABELS`). */
const LABELS: Readonly<Record<string, string>> = {
  text_encoder: "i8",
  dit_text: "i8",
  dit: "i8",
  dit_skip22: "i8",
  vae_decoder_pre: "f32",
  vae_decoder_attn: "f32",
  vae_decoder_post: "f32",
};

const PRACTICAL_SESSION = {
  linearCompute: "a8",
  attentionCompute: "a8",
  attentionScoreStorage: "none",
};

/** The recipe's `MOSS_PIPELINE_CONFIG`. */
const PIPELINE_CONFIG = {
  scheduler: { shift: 5.0 },
  defaults: { width: 1024, height: 1024, steps: 50, guidance: 3.0, negativePrompt: "" },
};

/** The manifest JSON; `patch` overrides fields of the model entry. */
const manifestText = (patch: Record<string, unknown> = {}): string => {
  const weights = Object.fromEntries(
    Object.entries(LABELS).map(([role, label]) => [
      role,
      { [label]: declaredContainer(`${MODEL}/${role}/model.${label}`) },
    ]),
  );
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.15.0",
    defaultModel: MODEL,
    models: {
      [MODEL]: {
        pipeline: "moss-image/1",
        weights,
        assets: {
          tokenizer: {
            path: `${MODEL}/tokenizer/tokenizer.json`,
            size: 16,
            sha256: "e".repeat(64),
          },
        },
        quants: {
          [MOSS_REFERENCE_PRESET]: { weights: LABELS, session: {} },
          [MOSS_PRACTICAL_PRESET]: { weights: LABELS, session: PRACTICAL_SESSION },
        },
        defaultQuant: MOSS_PRACTICAL_PRESET,
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });
};

const manifest = (patch: Record<string, unknown> = {}) => parseManifest(manifestText(patch));

describe("MOSS-Image pipeline: the preset from the manifest", () => {
  it("runs the model's defaultQuant when no quant is given — the practical preset", () => {
    const selection = mossSelectionOf(manifest(), {});
    expect(selection.quantName).toBe(MOSS_PRACTICAL_PRESET);
    expect(selection.preset).toBe(mossPreset(MOSS_PRACTICAL_PRESET));
    // No GPU feature is required (ADR 0128 C1) — the online kernel's subgroups are optional.
    expect(selection.gpuFeatures).toBeUndefined();
  });

  it("runs the named preset when quant is given", () => {
    const selection = mossSelectionOf(manifest(), { quant: MOSS_REFERENCE_PRESET });
    expect(selection.quantName).toBe(MOSS_REFERENCE_PRESET);
    expect(selection.preset).toBe(mossPreset(MOSS_REFERENCE_PRESET));
  });

  it("follows another defaultQuant (the default is the manifest's, not the code's)", () => {
    const selection = mossSelectionOf(manifest({ defaultQuant: MOSS_REFERENCE_PRESET }), {});
    expect(selection.preset.name).toBe(MOSS_REFERENCE_PRESET);
  });

  it("refuses a quant that is not in the manifest, naming the presets there are", () => {
    assertThrows(
      () => mossSelectionOf(manifest(), { quant: "i8" }),
      Error,
      `quant 'i8' is not in the manifest (available: ${MOSS_REFERENCE_PRESET} / ${MOSS_PRACTICAL_PRESET})`,
    );
  });

  it("refuses a model that is not in the manifest", () => {
    assertThrows(
      () => mossSelectionOf(manifest(), { model: "base" }),
      Error,
      "model 'base' is not in the manifest (available: sft)",
    );
  });

  it("refuses a preset name the family table does not know", () => {
    const quants = {
      [MOSS_REFERENCE_PRESET]: { weights: LABELS, session: {} },
      "f32+dit8-a8": { weights: LABELS, session: { linearCompute: "a8" } },
    };
    assertThrows(
      () => mossSelectionOf(manifest({ quants, defaultQuant: "f32+dit8-a8" }), {}),
      Error,
      "this implementation has no preset 'f32+dit8-a8'",
    );
  });

  it("refuses a checked name that declares other knobs", () => {
    const quants = {
      [MOSS_REFERENCE_PRESET]: { weights: LABELS, session: {} },
      [MOSS_PRACTICAL_PRESET]: {
        weights: LABELS,
        session: { ...PRACTICAL_SESSION, attentionScoreStorage: "f16" },
      },
    };
    assertThrows(
      () => mossSelectionOf(manifest({ quants }), {}),
      Error,
      `the manifest declares the knobs`,
    );
  });

  it("refuses a knob the family does not accept, as an asset inconsistency", () => {
    const error = assertThrows(
      () => mossPresetOf(MOSS_REFERENCE_PRESET, { convCompute: "a8" }, "where"),
      Error,
      "where: session.convCompute is not supported",
    );
    expect(error).not.toBeInstanceOf(ModelInputError);
  });
});

describe("MOSS-Image pipeline: the manifest contract", () => {
  it("refuses another pipeline name and another major", () => {
    for (const pipeline of ["wan/1", "moss-image/2"]) {
      assertThrows(
        () => mossSelectionOf(manifest({ pipeline }), {}),
        Error,
        `the manifest's pipeline is '${pipeline}'`,
      );
    }
  });

  it("refuses a manifest without the tokenizer asset before the weights are fetched", () => {
    assertThrows(
      () => mossSelectionOf(manifest({ assets: {} }), {}),
      Error,
      "the manifest's assets have no 'tokenizer'",
    );
  });
});

describe("MOSS-Image pipeline: the request defaults from pipelineConfig", () => {
  const config: MossPipelineConfig = parseMossPipelineConfig(PIPELINE_CONFIG);

  it("fills every field a request leaves out with upstream's defaults", () => {
    expect(mossRequestOf({ prompt: "a cat" }, config)).toEqual({
      prompt: "a cat",
      negativePrompt: "",
      width: 1024,
      height: 1024,
      steps: 50,
      guidance: 3,
      shift: 5,
      seed: 0,
    });
  });

  it("keeps every field a request gives", () => {
    const latents = new Float32Array(128 * 16 * 16);
    expect(
      mossRequestOf(
        {
          prompt: "a cat",
          negativePrompt: "blurry",
          overflow: "truncate",
          width: 256,
          height: 384,
          steps: 2,
          guidance: 4.5,
          shift: 3,
          latents,
        },
        config,
      ),
    ).toEqual({
      prompt: "a cat",
      negativePrompt: "blurry",
      overflow: "truncate",
      width: 256,
      height: 384,
      steps: 2,
      guidance: 4.5,
      shift: 3,
      latents,
    });
  });

  it("adds no default negative prompt at guidance 1, and keeps an explicit one for the check to refuse", () => {
    expect(mossRequestOf({ prompt: "a cat", guidance: 1 }, config)).not.toHaveProperty(
      "negativePrompt",
    );
    expect(mossRequestOf({ prompt: "a cat", guidance: 1, negativePrompt: "" }, config)).toEqual(
      expect.objectContaining({ negativePrompt: "" }),
    );
  });

  it("passes both a seed and latents through (the acceptance check refuses them)", () => {
    const latents = new Float32Array(1);
    expect(mossRequestOf({ prompt: "a cat", seed: 7, latents }, config)).toEqual(
      expect.objectContaining({ seed: 7, latents }),
    );
    expect(mossRequestOf({ prompt: "a cat", latents }, config)).not.toHaveProperty("seed");
  });
});

describe("MOSS-Image pipeline: the pipelineConfig schema", () => {
  const withDefaults = (patch: Record<string, unknown>) => ({
    ...PIPELINE_CONFIG,
    defaults: { ...PIPELINE_CONFIG.defaults, ...patch },
  });

  it("reads the recipe's declaration", () => {
    expect(parseMossPipelineConfig(PIPELINE_CONFIG)).toEqual(PIPELINE_CONFIG);
  });

  it("refuses unknown keys and missing fields", () => {
    assertThrows(
      () => parseMossPipelineConfig({ ...PIPELINE_CONFIG, sampler: {} }),
      Error,
      "pipelineConfig: unknown key 'sampler'",
    );
    assertThrows(
      () => parseMossPipelineConfig(withDefaults({ guidance_scale: 3 })),
      Error,
      "pipelineConfig.defaults: unknown key 'guidance_scale'",
    );
    const { negativePrompt: _dropped, ...defaults } = PIPELINE_CONFIG.defaults;
    assertThrows(
      () => parseMossPipelineConfig({ ...PIPELINE_CONFIG, defaults }),
      Error,
      "pipelineConfig.defaults.negativePrompt: missing",
    );
  });

  it("refuses values outside the request checks' ranges, as plain errors", () => {
    const cases: readonly [Record<string, unknown>, string][] = [
      [withDefaults({ width: 1000 }), "pipelineConfig.defaults.width"],
      [withDefaults({ height: 0 }), "pipelineConfig.defaults.height"],
      [withDefaults({ steps: 0 }), "pipelineConfig.defaults.steps"],
      [withDefaults({ guidance: -1 }), "pipelineConfig.defaults.guidance"],
      [withDefaults({ guidance: 1e39 }), "pipelineConfig.defaults.guidance"],
      [withDefaults({ negativePrompt: 0 }), "pipelineConfig.defaults.negativePrompt"],
      [{ ...PIPELINE_CONFIG, scheduler: { shift: 0 } }, "pipelineConfig.scheduler.shift"],
    ];
    for (const [raw, field] of cases) {
      const error = assertThrows(() => parseMossPipelineConfig(raw), Error, field);
      expect(error).not.toBeInstanceOf(ModelInputError);
    }
  });
});
