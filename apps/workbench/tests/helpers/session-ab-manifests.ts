/**
 * Hand-written `karume/5` manifests for the session A/B entries' tests, and an in-memory
 * distribution that serves one, so an entry's run reaches its parameter checks without a mirror.
 */

import { MountUnavailableError } from "@karume/local-distributions/browser";
import {
  type DistributionSource,
  localDirectory,
  MANIFEST_FILENAME,
  type Manifest,
  parseManifest,
} from "@karume/hub";

import type { OpenDistribution, OpenedDistribution } from "../../src/entry.ts";

export const hex = (character: string): string => character.repeat(64);

/** One container of two parts (part 0, the descriptor, and one weight part). */
export const container = (path: string) => ({
  container: {
    descriptor: {
      graph: { length: 8, sha256: hex("d") },
      model: { length: 8, sha256: hex("e") },
    },
    parts: [
      { path: `${path}-00001`, size: 40, sha256: hex("a") },
      { path: `${path}-00002`, size: 8, sha256: hex("b") },
    ],
  },
});

/**
 * The manifest's JSON text: `models` (the first is the default), each with `presets` (name →
 * session). A preset loads `component` at i8 unless `weights` names its own storages (component →
 * storage); every storage a preset names has its container.
 */
export const manifestText = (input: {
  readonly pipeline: string;
  readonly models: readonly string[];
  readonly component: string;
  readonly presets: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly weights?: Readonly<Record<string, Readonly<Record<string, string>>>>;
  readonly defaultPreset: string;
}): string => {
  const weightsOf = (preset: string): Readonly<Record<string, string>> =>
    input.weights !== undefined && Object.hasOwn(input.weights, preset)
      ? input.weights[preset]
      : { [input.component]: "i8" };
  const storages = new Map<string, Set<string>>();
  for (const preset of Object.keys(input.presets)) {
    for (const [component, storage] of Object.entries(weightsOf(preset))) {
      storages.set(component, (storages.get(component) ?? new Set()).add(storage));
    }
  }
  return JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: input.models[0],
    models: Object.fromEntries(
      input.models.map((model) => [
        model,
        {
          pipeline: input.pipeline,
          assets: {},
          weights: Object.fromEntries(
            [...storages].map(([component, names]) => [
              component,
              Object.fromEntries(
                [...names].map((storage) => [
                  storage,
                  container(`${model}/${component}.${storage}.krm`),
                ]),
              ),
            ]),
          ),
          quants: Object.fromEntries(
            Object.entries(input.presets).map(([name, session]) => [
              name,
              { weights: weightsOf(name), session },
            ]),
          ),
          defaultQuant: input.defaultPreset,
          pipelineConfig: {},
        },
      ]),
    ),
  });
};

/** A Gemma-shaped distribution of `models`, each with the presets `i4` and `i4-fast` (default). */
export const gemmaManifestText = (models: readonly string[]): string =>
  manifestText({
    pipeline: "gemma4/1",
    models,
    component: "model",
    presets: { i4: {}, "i4-fast": {} },
    defaultPreset: "i4-fast",
  });

export const manifestOf = (text: string): Manifest => parseManifest(text);

/** A distribution that serves `text` as its manifest and nothing else. */
export const memoryDistribution = (text: string): DistributionSource =>
  localDirectory(
    {
      readFile: (path) =>
        path === MANIFEST_FILENAME
          ? Promise.resolve(new TextEncoder().encode(text))
          : Promise.reject(new Error(`memory distribution: no ${path}`)),
    },
    { label: "memory" },
  );

/**
 * An `openDistribution` over in-memory manifests by mount; a mount without one throws as a missing
 * mirror does. `opened` records every mount asked for, in order.
 */
export const memoryDistributions =
  (manifests: Readonly<Record<string, string>>, opened: string[] = []): OpenDistribution =>
  async (mount, source): Promise<OpenedDistribution> => {
    opened.push(`${mount}:${source}`);
    const text = Object.hasOwn(manifests, mount) ? manifests[mount] : undefined;
    if (text === undefined) throw new MountUnavailableError(mount, "its mirror is missing");
    return {
      ref: memoryDistribution(text),
      facts: { directory: `models/karume-${mount}`, manifestSha256: hex("f"), read: "directory" },
    };
  };

/** The MOSS-Image mirror's shape: one model, the reference preset and the practical one (default). */
export const MOSS_MANIFEST = manifestText({
  pipeline: "moss-image/1",
  models: ["sft"],
  component: "dit",
  presets: {
    "f32+dit8": {},
    "f32+dit8-a8-attn8-online": {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "none",
    },
  },
  defaultPreset: "f32+dit8-a8-attn8-online",
});

/**
 * The Wan2.2 mirror's shape around the next chunk graph: a preset with the f16 VAE, the i8 one,
 * and the conv8 one (the default).
 */
export const WAN_MANIFEST = manifestText({
  pipeline: "wan-ti2v/1",
  models: ["ti2v-5b"],
  component: "vae_decoder_next",
  presets: {
    "f16+dit8-a8-attn8-online": {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "none",
    },
    i8: {},
    "i8-a8-attn8-online-conv8": {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "none",
      convCompute: "a8",
    },
  },
  weights: { "f16+dit8-a8-attn8-online": { vae_decoder_next: "f16" } },
  defaultPreset: "i8-a8-attn8-online-conv8",
});
