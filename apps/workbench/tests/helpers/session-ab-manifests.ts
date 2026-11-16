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

/** The manifest's JSON text: `models` (the first is the default), each with `presets` of `component`. */
export const manifestText = (input: {
  readonly pipeline: string;
  readonly models: readonly string[];
  readonly component: string;
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
          pipeline: input.pipeline,
          assets: {},
          weights: { [input.component]: { i8: container(`${model}/${input.component}.krm`) } },
          quants: Object.fromEntries(
            Object.entries(input.presets).map(([name, session]) => [
              name,
              { weights: { [input.component]: "i8" }, session },
            ]),
          ),
          defaultQuant: input.defaultPreset,
          pipelineConfig: {},
        },
      ]),
    ),
  });

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
