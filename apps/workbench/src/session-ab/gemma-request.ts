/**
 * The session A/B's request on the Gemma 4 decode: the settings, and the fixed work every timed run
 * does (one prompt, a greedy sampler with a fixed seed, one capacity). Pure, so `vp test` covers it.
 */

import type { GemmaLoad } from "../gemma4/pipeline.ts";
import { type RequestArm, type RoundsAndGain, twoArms } from "./shared.ts";

/** The prompt of every timed run (the 2026-11-10 measurement's first prompt). */
export const SESSION_AB_PROMPT =
  "Explain in detail how a refrigerator keeps food cold, step by step, for a curious teenager.";

/** Greedy with a fixed seed: every run of every arm decodes the same way. */
export const SESSION_AB_SAMPLER = { temperature: 0, seed: 0 } as const;

/**
 * The capacity of every run's sequence. It holds the prompt and up to 512 tokens (the most the
 * `tokens` parameter takes), and is at least every Gemma 4 model's chunk length.
 */
export const SESSION_AB_CAPACITY = 1024;

/** What an A/B runs with besides the model. */
export type SessionAbSettings = RoundsAndGain & {
  readonly tokens: number;
  /** Open and close each arm's pipeline around every visit (two large pipelines do not fit together). */
  readonly openPerVisit: boolean;
  /**
   * Probe each arm's fusion and dispatch counts before the A/B (one extra load and run per arm,
   * built with `onRunDiagnostics`; the arms themselves never are).
   */
  readonly probeCounts: boolean;
};

/** The pipeline every arm loads: the model, the preset and the load options. */
export type SessionAbModel = GemmaLoad;

/** What one run asked for, as the record keeps it. */
export type SessionAbRequest = SessionAbModel &
  SessionAbSettings & {
    readonly prompt: string;
    readonly sampler: typeof SESSION_AB_SAMPLER;
    readonly capacity: number;
    /** The arms in order: the first is the base. */
    readonly arms: readonly RequestArm[];
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
  arms: twoArms(ruleName),
});
