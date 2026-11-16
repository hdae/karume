/**
 * Hand-written `karume/5` manifests shaped like the three Gemma 4 mirrors (the models, the presets
 * and the default preset of each, with a valid `pipelineConfig`), for the decode bench's tests:
 * `memoryDistributions` serves them, so an entry's run reaches its arm checks without a mirror.
 */

import { container } from "./session-ab-manifests.ts";

/** A Gemma 4 `pipelineConfig` as the mirrors declare it (with the distribution's sampler). */
export const PIPELINE_CONFIG = {
  chunkLength: 768,
  maxChunkLength: 768,
  maxPosition: 131072,
  capacity: 4096,
  rope: {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  },
  sampler: { temperature: 1, topK: 64, topP: 0.95 },
};

/**
 * The manifest's JSON text: `models` (the first is the default), each with `presets` (name →
 * session declaration, labelled `<name> label`) and `defaultPreset`.
 */
export const gemmaBenchManifestText = (input: {
  readonly models: readonly string[];
  readonly presets: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly defaultPreset: string;
}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: input.models[0],
    models: Object.fromEntries(
      input.models.map((model) => [
        model,
        {
          pipeline: "gemma4/1",
          assets: {},
          weights: { model: { i4: container(`${model}/model.krm`) } },
          quants: Object.fromEntries(
            Object.entries(input.presets).map(([name, session]) => [
              name,
              { weights: { model: "i4" }, session, label: `${name} label` },
            ]),
          ),
          defaultQuant: input.defaultPreset,
          pipelineConfig: PIPELINE_CONFIG,
        },
      ]),
    ),
  });

/** Regular Gemma 4: E2B (default) and E4B at `bf16`. */
export const REGULAR = gemmaBenchManifestText({
  models: ["e2b", "e4b"],
  presets: { bf16: {} },
  defaultPreset: "bf16",
});

/** The q4_0 QAT line: E2B (default), E4B, 12B and 26B at `i4` (default) and `i4-fast`. */
export const QAT = gemmaBenchManifestText({
  models: ["e2b", "e4b", "12b", "26b"],
  presets: { i4: {}, "i4-fast": { numerics: { level: "reorder" }, packedStaticQuantize: true } },
  defaultPreset: "i4",
});

/**
 * The mobile QAT distribution: E2B (default) and E4B, plus a 12B its entry point refuses, at `i4`,
 * `i4-gemvpar` and `i4-fast` (default, declaring the packed activations).
 */
export const MOBILE = gemmaBenchManifestText({
  models: ["e2b", "e4b", "12b"],
  presets: {
    i4: {},
    "i4-gemvpar": { numerics: { level: "reorder" } },
    "i4-fast": { numerics: { level: "reorder" }, packedStaticQuantize: true },
  },
  defaultPreset: "i4-fast",
});
