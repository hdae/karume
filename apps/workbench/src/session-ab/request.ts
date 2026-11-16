/**
 * The session A/B's request on the Gemma 4 decode: the settings, their defaults and bounds (the
 * entries' parameters declare them), and the fixed work every timed run does (one prompt, a greedy
 * sampler with a fixed seed, one capacity); and the preset rule both entries share. Pure, so
 * `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import type { SourceKind } from "@karume/local-distributions/browser";

import { ParameterError } from "../entry.ts";
import type { ExpertCacheSlots, GemmaMount } from "../gemma4/settings.ts";

/**
 * How a browser reads the distribution (`openDistribution`'s `source`): `cached` first, the default —
 * an A/B reads the weights again for every pipeline or Session it builds, and the direct route would
 * fetch them anew each time. Under Node the mirror's directory is read either way.
 */
export const SOURCE_KINDS = ["cached", "direct"] as const satisfies readonly SourceKind[];

/** The prompt of every timed run (the 2026-11-10 measurement's first prompt). */
export const SESSION_AB_PROMPT =
  "Explain in detail how a refrigerator keeps food cold, step by step, for a curious teenager.";

/** Greedy with a fixed seed: every run of every arm decodes the same way. */
export const SESSION_AB_SAMPLER = { temperature: 0, seed: 0 } as const;

/**
 * The capacity of every run's sequence. It holds the prompt and up to
 * {@link SESSION_AB_LIMITS}`.maxTokens` tokens, and is at least every Gemma 4 model's chunk length.
 */
export const SESSION_AB_CAPACITY = 1024;

/**
 * What the settings accept. Rounds: the runner's minimum, and a generous maximum. Tokens: at least
 * 2 (the first token comes from the prefill, so a run of one token has no decode run for the probe
 * to count), at most what {@link SESSION_AB_CAPACITY} holds next to the prompt. Minimum gain: a
 * fraction of the base's time, 0 to 1 (a gain cannot exceed the whole time).
 */
export const SESSION_AB_LIMITS = {
  minRounds: 3,
  maxRounds: 1000,
  minTokens: 2,
  maxTokens: 512,
  minGain: 0,
  maxGain: 1,
} as const;

/** What an A/B runs with besides the model. */
export type SessionAbSettings = {
  readonly rounds: number;
  readonly tokens: number;
  readonly minGain: number;
  /** Open and close each arm's pipeline around every visit (two large pipelines do not fit together). */
  readonly openPerVisit: boolean;
  /**
   * Probe each arm's fusion and dispatch counts before the A/B (one extra load and run per arm,
   * built with `onRunDiagnostics`; the arms themselves never are).
   */
  readonly probeCounts: boolean;
};

/** The defaults: 4 rounds of 96 tokens, a minimum gain of 1%, open per visit, the counts probed. */
export const SESSION_AB_DEFAULTS: SessionAbSettings = {
  rounds: 4,
  tokens: 96,
  minGain: 0.01,
  openPerVisit: true,
  probeCounts: true,
};

/** The pipeline every arm loads: the model, the preset and the load options. */
export type SessionAbModel = {
  readonly mount: GemmaMount;
  readonly model: string;
  readonly quant: string;
  /** Absent = f32 full state slots. */
  readonly fullStateStorage?: "f16";
  /** Absent = every expert resident (a mixture-of-experts model only). */
  readonly expertCache?: {
    readonly slots: ExpertCacheSlots;
    readonly hostBytes?: number;
    readonly depth?: number;
  };
};

/** What one run asked for, as the record keeps it. */
export type SessionAbRequest = SessionAbModel &
  SessionAbSettings & {
    readonly prompt: string;
    readonly sampler: typeof SESSION_AB_SAMPLER;
    readonly capacity: number;
    /** The arms in order: the first is the base. */
    readonly arms: readonly { readonly name: string; readonly fusionRules: readonly string[] }[];
  };

/** The request of a run: the model and the settings, with the fixed work and the arms named. */
export const sessionAbRequest = (
  model: SessionAbModel,
  settings: SessionAbSettings,
  ruleName: string,
): SessionAbRequest => ({
  ...model,
  ...settings,
  prompt: SESSION_AB_PROMPT,
  sampler: SESSION_AB_SAMPLER,
  capacity: SESSION_AB_CAPACITY,
  arms: [
    { name: "base", fusionRules: [] },
    { name: "rule", fusionRules: [ruleName] },
  ],
});

/**
 * The preset a run loads: `chosen` when it is a preset of `model`, the model's `defaultQuant` when
 * `chosen` is empty. Anything else is a {@link ParameterError} of `preset` that lists the presets.
 */
export const presetOf = (manifest: Manifest, model: string, chosen: string): string => {
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  if (chosen === "") return entry.defaultQuant;
  if (Object.hasOwn(entry.quants, chosen)) return chosen;
  throw new ParameterError(
    "preset",
    `'${chosen}' is not a preset of model '${model}'; its presets are ${Object.keys(entry.quants).join(", ")}`,
  );
};
