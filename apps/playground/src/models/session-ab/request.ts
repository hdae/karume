/**
 * The session A/B page's request: the form as typed, its defaults and its validation, and the
 * fixed work every timed run does (one prompt, a greedy sampler with a fixed seed, one capacity).
 * Pure, so `vp test` covers it.
 */

import type { ExpertCacheSlots, GemmaMount, Parsed } from "@/models/gemma4/settings";

/**
 * The backends the user picks from (WebGPU does not expose the backend, so the record states what
 * the user says).
 */
export const SESSION_AB_BACKENDS = ["d3d12", "metal", "vulkan", "other"] as const;
export type SessionAbBackend = (typeof SESSION_AB_BACKENDS)[number];

const isBackend = (value: string): value is SessionAbBackend =>
  SESSION_AB_BACKENDS.some((backend) => backend === value);

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
 * What the form accepts. Rounds: the runner's minimum. Tokens: at least 2 (the first token comes
 * from the prefill, so a run of one token has no decode run and no counts to report), at most what
 * {@link SESSION_AB_CAPACITY} holds next to the prompt.
 */
export const SESSION_AB_LIMITS = { minRounds: 3, minTokens: 2, maxTokens: 512 } as const;

/** The form, as typed (parsed by {@link parseSessionAbForm}). */
export type SessionAbForm = {
  readonly rounds: string;
  readonly tokens: string;
  readonly minGain: string;
  /** Open and close each arm's pipeline around every visit (two large pipelines do not fit together). */
  readonly openPerVisit: boolean;
  /** Read the fusion and dispatch counts through `onRunDiagnostics`. */
  readonly readCounts: boolean;
  /** Empty until the user picks one: the record must not name a backend nobody chose. */
  readonly backend: string;
};

export const INITIAL_SESSION_AB_FORM: SessionAbForm = {
  rounds: "4",
  tokens: "96",
  minGain: "0.01",
  openPerVisit: true,
  readCounts: true,
  backend: "",
};

/** What the form gives once it is valid. */
export type SessionAbSettings = {
  readonly rounds: number;
  readonly tokens: number;
  readonly minGain: number;
  readonly openPerVisit: boolean;
  readonly readCounts: boolean;
  readonly backend: SessionAbBackend;
};

const integerOf = (text: string): number | undefined => {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : undefined;
};

const fractionOf = (text: string): number | undefined => {
  const trimmed = text.trim();
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : undefined;
};

/** The form → the settings, or every problem at once. */
export const parseSessionAbForm = (form: SessionAbForm): Parsed<SessionAbSettings> => {
  const errors: string[] = [];
  const rounds = integerOf(form.rounds);
  if (rounds === undefined || rounds < SESSION_AB_LIMITS.minRounds) {
    errors.push(`Rounds must be an integer ≥ ${SESSION_AB_LIMITS.minRounds}.`);
  }
  const tokens = integerOf(form.tokens);
  if (
    tokens === undefined ||
    tokens < SESSION_AB_LIMITS.minTokens ||
    tokens > SESSION_AB_LIMITS.maxTokens
  ) {
    errors.push(
      `Tokens per run must be an integer from ${SESSION_AB_LIMITS.minTokens} to ${SESSION_AB_LIMITS.maxTokens}.`,
    );
  }
  const minGain = fractionOf(form.minGain);
  if (minGain === undefined) {
    errors.push("Minimum gain must be a number ≥ 0 (a fraction: 0.01 = 1% of the base's time).");
  }
  const backend = form.backend;
  if (!isBackend(backend)) {
    errors.push(`Choose the backend (${SESSION_AB_BACKENDS.join(" / ")}).`);
  }
  if (
    errors.length > 0 ||
    rounds === undefined ||
    tokens === undefined ||
    minGain === undefined ||
    !isBackend(backend)
  ) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    value: {
      rounds,
      tokens,
      minGain,
      openPerVisit: form.openPerVisit,
      readCounts: form.readCounts,
      backend,
    },
  };
};

/** The pipeline every arm loads: the model, the preset and the load options the page offers. */
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

/** What one press of Run asked for, as the record keeps it. */
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
