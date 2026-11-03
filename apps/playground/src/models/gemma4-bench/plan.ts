/**
 * What the decode bench runs: the prompt set, the arms (a model of one of the two Gemma 4
 * distributions × a preset × the tap on, on and split per dispatch, or off × the integer GEMV on or
 * off × the byte-plane GEMV on or off), the workload form (chats, new tokens, rounds,
 * sampler) and the order the arms are visited in. Pure, so `vp test` covers it.
 */

import type { Manifest, SessionSpec } from "@karume/hub";
import type { DefaultSampler, FromPretrainedOptions, SamplerSpec } from "@karume/models/gemma";

import type { Choice } from "@/components/choice-select";
import { manifestPresets } from "@/lib/presets";
import {
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  isQatModel,
  type ModelKey,
  modelConfig,
} from "@/models/gemma4/settings";

/**
 * Distinct token ids at which a prompt makes a PLE block whole-resident on the host (ADR 0085): a
 * prompt with this many or more is the condition under which the row cache once stopped admitting
 * rows (perf-ledger H-41), so the set has one prompt on each side of it.
 */
export const PLE_DISTINCT_ID_THRESHOLD = 32;

export type BenchPromptId = "few-ids" | "many-ids";

export type BenchPrompt = {
  readonly id: BenchPromptId;
  readonly label: string;
  readonly text: string;
  /** Which side of {@link PLE_DISTINCT_ID_THRESHOLD} the rendered prompt is meant to fall on. */
  readonly distinctIds: "below" | "at-least";
};

/**
 * The fixed prompts, in the order a visit runs them (the one below the threshold first, so it runs
 * on a pipeline no larger prompt has touched yet). Both ask for more than 128 tokens of text, so a
 * greedy run normally reaches the token limit. With the chat template the Gemma 4 tokenizer renders
 * them to 21 tokens (17 distinct) and 60 tokens (48 distinct); the page counts them again at run time.
 */
export const BENCH_PROMPTS: readonly BenchPrompt[] = [
  {
    id: "few-ids",
    label: "Story request",
    text: "Write a long story about a lighthouse keeper and the sea.",
    distinctIds: "below",
  },
  {
    id: "many-ids",
    label: "GPU explainer",
    text: "Explain in about 300 words how a GPU runs thousands of threads at once: cover warps or wavefronts, the schedulers, how memory latency is hidden, registers, shared memory and occupancy, and why branches inside a warp cost time.",
    distinctIds: "at-least",
  },
];

export const benchPrompt = (id: BenchPromptId): BenchPrompt => {
  const prompt = BENCH_PROMPTS.find((candidate) => candidate.id === id);
  if (prompt === undefined) throw new Error(`No bench prompt '${id}'`);
  return prompt;
};

/** Whether a rendered prompt's distinct id count falls on the side its prompt is meant for. */
export const meetsDistinctIdCondition = (prompt: BenchPrompt, distinctIds: number): boolean =>
  prompt.distinctIds === "below"
    ? distinctIds < PLE_DISTINCT_ID_THRESHOLD
    : distinctIds >= PLE_DISTINCT_ID_THRESHOLD;

/** One arm as the form holds it (`quant` absent = the model's default preset). */
export type ArmDraft = {
  readonly id: number;
  readonly key: ModelKey;
  readonly quant?: string;
  readonly tap: boolean;
  /** Split the tap per dispatch (applies only with the tap on). */
  readonly split?: boolean;
  /** Run the eligible decode linears on the integer GEMV (the Session option `integerGemv`). */
  readonly integerGemv?: boolean;
  /** Run the QAT lm_head on the byte-plane GEMV (the Session option `linearGemvPlanes`). */
  readonly linearGemvPlanes?: boolean;
};

/**
 * The Session options an arm passes to `fromPretrained` explicitly. The pipeline resolves each one
 * as explicit > the preset's declaration > the runtime default, so only what the arm changes is
 * passed (absent = the preset alone).
 */
export type ArmSessionOptions = Pick<
  FromPretrainedOptions,
  "integerGemv" | "packedStaticQuantize" | "linearGemvReduce" | "linearGemvPlanes"
>;

/**
 * The explicit options of the integer GEMV switch: `integerGemv: true`, plus what it requires —
 * packed activations, which need the parallel GEMV — where the preset does not declare it already
 * (QAT `i4-fast` declares both, so it passes `integerGemv` alone).
 */
export const integerGemvSession = (declared: SessionSpec): ArmSessionOptions => ({
  integerGemv: true,
  ...(declared.packedStaticQuantize === true ? {} : { packedStaticQuantize: true }),
  ...(declared.linearGemvReduce === "parallel" ? {} : { linearGemvReduce: "parallel" }),
});

/** One arm, checked against the manifests. */
export type BenchArm = {
  readonly mount: GemmaMount;
  readonly model: string;
  readonly quant: string;
  readonly quantLabel: string;
  readonly isDefaultQuant: boolean;
  /** Whether the tap records and times this arm's chats (off = a wall-clock control). */
  readonly tap: boolean;
  /**
   * Whether the tap runs split — one timed pass per dispatch, labelled with its pipeline key — for
   * the per-kernel attribution (only with the tap on; absent = the product's own passes).
   */
  readonly split?: boolean;
  /** The Session options passed explicitly (absent = none: the preset's declaration alone). */
  readonly session?: ArmSessionOptions;
  /** The distribution's declared sampler (for the "distribution's sampler" choice). */
  readonly declaredSampler?: DefaultSampler;
};

export const MOUNT_SHORT_LABELS: Readonly<Record<GemmaMount, string>> = {
  gemma4: "Gemma 4",
  "gemma4-qat": "QAT",
};

export const armLabel = (
  arm: Pick<BenchArm, "mount" | "model" | "quant" | "tap" | "split" | "session">,
): string =>
  `${MOUNT_SHORT_LABELS[arm.mount]} ${arm.model} ${arm.quant}${
    arm.session?.integerGemv === true ? " (integer)" : ""
  }${arm.session?.linearGemvPlanes === true ? " (planes)" : ""}${
    arm.tap ? (arm.split === true ? " (split)" : "") : " (tap off)"
  }`;

/**
 * The arms the page starts with: the default model of each loaded distribution at its default
 * preset, tap on (regular Gemma 4 first, so it is arm 1, the one the others are compared with).
 */
export const defaultArmDrafts = (manifests: GemmaManifests): readonly ArmDraft[] =>
  GEMMA_MOUNTS.flatMap((mount, index) => {
    const manifest = manifests[mount];
    if (manifest === undefined) return [];
    return [{ id: index, key: { mount, model: manifest.defaultModel }, tap: true }];
  });

export type ArmResolution =
  | { readonly ok: true; readonly arm: BenchArm }
  | { readonly ok: false; readonly error: string };

/** A draft → its arm, or why it cannot run (a missing manifest or model, or a QAT model the entry point refuses). */
export const resolveArm = (draft: ArmDraft, manifests: GemmaManifests): ArmResolution => {
  const { mount, model } = draft.key;
  const manifest: Manifest | undefined = manifests[mount];
  if (manifest === undefined) return { ok: false, error: `The ${mount} manifest is not loaded.` };
  if (manifest.models[model] === undefined) {
    return { ok: false, error: `The ${mount} manifest has no model '${model}'.` };
  }
  if (mount === "gemma4-qat" && !isQatModel(model)) {
    return { ok: false, error: `gemma4Qat.Pipeline does not accept the model '${model}'.` };
  }
  const { presets, defaultQuant } = manifestPresets(manifest, model);
  const preset =
    presets.find(({ name }) => name === draft.quant) ??
    presets.find(({ name }) => name === defaultQuant);
  if (preset === undefined) {
    return { ok: false, error: `The ${mount} model '${model}' has no preset '${defaultQuant}'.` };
  }
  const declaredSampler = modelConfig(manifest, model).sampler;
  // The byte-plane GEMV needs no other option, so it joins whatever the integer switch passes.
  const session: ArmSessionOptions = {
    ...(draft.integerGemv === true
      ? integerGemvSession(manifest.models[model].quants[preset.name].session)
      : {}),
    ...(draft.linearGemvPlanes === true ? { linearGemvPlanes: true } : {}),
  };
  return {
    ok: true,
    arm: {
      mount,
      model,
      quant: preset.name,
      quantLabel: preset.label,
      isDefaultQuant: preset.name === defaultQuant,
      tap: draft.tap,
      ...(draft.tap && draft.split === true ? { split: true } : {}),
      ...(Object.keys(session).length === 0 ? {} : { session }),
      ...(declaredSampler === undefined ? {} : { declaredSampler }),
    },
  };
};

/** Which sampler every chat of every arm is sent with. */
export type SamplerKind = "greedy" | "distribution";

export const SAMPLER_CHOICES: readonly Choice<SamplerKind>[] = [
  { value: "greedy", label: "Greedy (temperature 0, the same tokens in every round)" },
  { value: "distribution", label: "The distribution's sampler, seed 0" },
];

/**
 * The sampler a chat is sent with: greedy, or the distribution's declared sampler with seed 0
 * (greedy when it declares none). Always passed explicitly, so the record names exactly what ran.
 */
export const samplerOf = (kind: SamplerKind, declared: DefaultSampler | undefined): SamplerSpec =>
  kind === "greedy" || declared === undefined
    ? { temperature: 0 }
    : { temperature: declared.temperature, topK: declared.topK, topP: declared.topP, seed: 0 };

/** The workload form, as typed. */
export type BenchForm = {
  readonly chats: string;
  readonly newTokens: string;
  readonly rounds: string;
  readonly sampler: SamplerKind;
  readonly prompts: readonly BenchPromptId[];
};

export const INITIAL_BENCH_FORM: BenchForm = {
  chats: "3",
  newTokens: "128",
  rounds: "4",
  sampler: "greedy",
  prompts: BENCH_PROMPTS.map(({ id }) => id),
};

export type BenchSettings = {
  /** Chats per prompt per visit; chat 0 is the cold one and is left out of the summary. */
  readonly chats: number;
  /** `maxNewTokens` of every chat. */
  readonly newTokens: number;
  readonly rounds: number;
  readonly sampler: SamplerKind;
  /** In {@link BENCH_PROMPTS} order. */
  readonly prompts: readonly BenchPromptId[];
};

export type Parsed<Value> =
  | { readonly ok: true; readonly value: Value }
  | { readonly ok: false; readonly errors: readonly string[] };

const integerOf = (text: string): number | undefined => {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : undefined;
};

/**
 * The form → the settings, or every problem at once. Two chats at least (chat 0 is discarded), two
 * new tokens at least (one decode token), one round, one prompt.
 */
export const parseBenchForm = (form: BenchForm): Parsed<BenchSettings> => {
  const errors: string[] = [];
  const chats = integerOf(form.chats);
  if (chats === undefined || chats < 2) {
    errors.push("Chats must be an integer ≥ 2 (chat 0 is the cold one and is left out).");
  }
  const newTokens = integerOf(form.newTokens);
  if (newTokens === undefined || newTokens < 2) {
    errors.push("New tokens must be an integer ≥ 2 (the first token comes from the prefill).");
  }
  const rounds = integerOf(form.rounds);
  if (rounds === undefined || rounds < 1) errors.push("Rounds must be an integer ≥ 1.");
  const prompts = BENCH_PROMPTS.map(({ id }) => id).filter((id) => form.prompts.includes(id));
  if (prompts.length === 0) errors.push("Choose at least one prompt.");
  if (errors.length > 0 || chats === undefined || newTokens === undefined || rounds === undefined) {
    return { ok: false, errors };
  }
  return { ok: true, value: { chats, newTokens, rounds, sampler: form.sampler, prompts } };
};

/** One visit of an arm: its pipeline is built, runs every chat of the visit, and is released. */
export type Visit = { readonly visit: number; readonly round: number; readonly arm: number };

/**
 * The visit order: even rounds take the arms in order, odd rounds in reverse (A B, B A, A B, … —
 * ABBA for two arms), so a slow drift of the machine (heat, clocks) lands on every arm alike.
 */
export const visitOrder = (armCount: number, rounds: number): readonly Visit[] => {
  const visits: Visit[] = [];
  for (let round = 0; round < rounds; round += 1) {
    for (let step = 0; step < armCount; step += 1) {
      const arm = round % 2 === 0 ? step : armCount - 1 - step;
      visits.push({ visit: visits.length, round, arm });
    }
  }
  return visits;
};
