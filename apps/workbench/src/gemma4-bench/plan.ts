/**
 * What the decode bench runs: the prompt set, the arms (a model of one of the three Gemma 4
 * distributions × a preset × the tap on, on and split per dispatch, in the host mode, or off × linear
 * at the level `reorder` or as the preset declares × the integer GEMV on or off × the PLE on the host
 * or the GPU × the decode run-ahead depth × a signal passed or not × the bench's sampler or the
 * distribution's), the workload (chats, new tokens, rounds, sampler) and the order the arms are
 * visited in. Pure, so `vp test` covers it.
 */

import {
  GEMMA_MOUNTS,
  type GemmaManifests,
  type GemmaMount,
  type ModelKey,
  modelConfig,
} from "@karume/gemma4-app-support/settings";
import type { Manifest, SessionSpec } from "@karume/hub";
import type { DefaultSampler, SamplerSpec } from "@karume/models/gemma";
import { DEFAULT_SUBMIT_POLICY } from "@karume/runtime";

import { modelRefusal } from "../gemma4/open.ts";
import type { GemmaPipelineOptions, GemmaSessionOptions } from "../gemma4/pipeline.ts";
import { modelPreset } from "../model-preset.ts";

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
 * them to 21 tokens (17 distinct) and 60 tokens (48 distinct); the bench counts them again at run
 * time.
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

/**
 * The prompt ids in the set's order. Marked pure so a production build, which leaves out the
 * entries that read a distribution, drops it with them.
 */
export const BENCH_PROMPT_IDS: readonly BenchPromptId[] = /* @__PURE__ */ BENCH_PROMPTS.map(
  ({ id }) => id,
);

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

/** One arm as the `arms` parameter names it (`quant` absent = the model's default preset). */
export type ArmDraft = {
  readonly key: ModelKey;
  readonly quant?: string;
  readonly tap: boolean;
  /** Split the tap per dispatch (applies only with the tap on). */
  readonly split?: boolean;
  /**
   * Run linear at the level `reorder` (the Session option `numerics` with `ops.linear`) on a preset
   * that does not declare it: the arm against the same preset at `exact`.
   */
  readonly linearReorder?: boolean;
  /** Run the tap in its host mode: wall-clock marks at the WebGPU calls, the passes untouched. */
  readonly hostMarks?: boolean;
  /** Send the chats without an `AbortSignal` (absent = with the run's signal). */
  readonly noSignal?: boolean;
  /** Send the chats with the distribution's sampler, whatever the bench's `sampler` is. */
  readonly sampled?: boolean;
  /** Run the eligible decode linears on the integer GEMV (the Session option `integerGemv`). */
  readonly integerGemv?: boolean;
  /** Keep the PLE on the GPU and gather it there (the pipeline option `pleResidency: "gpu"`). */
  readonly pleOnGpu?: boolean;
  /** Greedy decode steps per GPU batch (the pipeline option `decodeAhead`; absent = 1). */
  readonly decodeAhead?: DecodeAhead;
  /**
   * The first chunk of a run's submits (the Session option `submitPolicy` with that
   * `firstChunkSize`, or without one for `"off"`; absent = the pipeline's own policy).
   */
  readonly firstChunk?: FirstChunk;
};

/** The first-chunk sizes the bench offers, and `"off"` for a policy without the ramp. */
export type FirstChunk = "8" | "16" | "32" | "64" | "off";

/**
 * The dispatch cap per submit of the arms that set their own first chunk: the cap of the Gemma
 * pipelines' own policy (perf-ledger H-40), so such an arm differs from the pipeline's policy in
 * the first chunk alone.
 */
export const FIRST_CHUNK_MAX_CHUNK_SIZE = 128;

/**
 * The submit policy of an arm that sets its first chunk: the runtime default with the Gemma cap,
 * and a first chunk of that size (`"off"`: none — every chunk at the cap).
 */
export const firstChunkPolicy = (
  firstChunk: FirstChunk,
): NonNullable<ArmSessionOptions["submitPolicy"]> => ({
  ...DEFAULT_SUBMIT_POLICY,
  maxChunkSize: FIRST_CHUNK_MAX_CHUNK_SIZE,
  ...(firstChunk === "off" ? {} : { firstChunkSize: Number(firstChunk) }),
});

/** The run-ahead depths the bench offers (1 = off — one step per batch). */
export type DecodeAhead = "1" | "2" | "4" | "8";

/**
 * The pipeline options an arm passes to `fromPretrained` (absent = the default: the PLE on the
 * host, one decode step per batch).
 */
export type ArmPipelineOptions = GemmaPipelineOptions;

/**
 * The Session options an arm passes to `fromPretrained` explicitly. The pipeline resolves each one
 * as explicit > the preset's declaration > the runtime default, so only what the arm changes is
 * passed (absent = the preset alone).
 */
export type ArmSessionOptions = Omit<GemmaSessionOptions, "fusionRules">;

/**
 * The explicit options of the integer GEMV switch: `integerGemv: true`, plus what it requires —
 * packed activations, which need linear at the level `reorder` (the parallel GEMV) — where the
 * preset does not declare it already (mobile QAT `i4-fast` declares both, so it passes
 * `integerGemv` alone). Only the linear level is passed: the pipeline composes `numerics` per
 * field, so the levels the preset declares for the other op kinds stay.
 */
export const integerGemvSession = (declared: SessionSpec): ArmSessionOptions => {
  const linearLevel = declaredLinearLevel(declared);
  return {
    integerGemv: true,
    ...(declared.packedStaticQuantize === true ? {} : { packedStaticQuantize: true }),
    ...(linearLevel === "reorder" ? {} : { numerics: { ops: { linear: "reorder" } } }),
  };
};

/** The level a preset's declaration gives linear (absent: `exact`). */
const declaredLinearLevel = (declared: SessionSpec): "exact" | "reorder" | undefined =>
  declared.numerics?.ops?.linear ?? declared.numerics?.level;

/**
 * The explicit options of the linear-reorder switch: linear at the level `reorder`, the other op
 * kinds as the preset declares them (the pipeline composes `numerics` per field).
 */
export const LINEAR_REORDER_SESSION: ArmSessionOptions = {
  numerics: { ops: { linear: "reorder" } },
};

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
  /**
   * Whether the arm asked for linear at `reorder` on a preset that does not declare it (the
   * `linear-reorder` switch). The label names it; the integer GEMV's options hold the same level, so
   * {@link session} alone does not tell the two switches apart.
   */
  readonly linearReorder?: true;
  /**
   * Whether the tap runs in its host mode — wall-clock marks at the WebGPU calls and at the checks
   * of the chat's signal, no pass timing — for the host's segments of a decode token (only with the
   * tap on and not split).
   */
  readonly hostMarks?: boolean;
  /**
   * Whether the chats are sent without an `AbortSignal`: the generation loop then does not yield to
   * the event loop before a run, and a cancel takes effect between two chats (absent = the run's
   * signal is passed).
   */
  readonly noSignal?: boolean;
  /** The sampler kind of this arm alone (absent = the bench's `sampler`). */
  readonly samplerKind?: SamplerKind;
  /** The Session options passed explicitly (absent = none: the preset's declaration alone). */
  readonly session?: ArmSessionOptions;
  /** The pipeline options passed explicitly (absent = none — {@link ArmPipelineOptions}). */
  readonly pipeline?: ArmPipelineOptions;
  /** The distribution's declared sampler (for the "distribution's sampler" choice). */
  readonly declaredSampler?: DefaultSampler;
};

export const MOUNT_SHORT_LABELS: Readonly<Record<GemmaMount, string>> = {
  gemma4: "Gemma 4",
  "gemma4-qat": "QAT q4_0",
  "gemma4-qat-mobile": "mobile QAT",
};

export const armLabel = (
  arm: Pick<
    BenchArm,
    | "mount"
    | "model"
    | "quant"
    | "tap"
    | "split"
    | "hostMarks"
    | "noSignal"
    | "samplerKind"
    | "linearReorder"
    | "session"
    | "pipeline"
  >,
): string =>
  `${MOUNT_SHORT_LABELS[arm.mount]} ${arm.model} ${arm.quant}${
    arm.linearReorder === true ? " (linear reorder)" : ""
  }${arm.session?.integerGemv === true ? " (integer)" : ""}${
    arm.pipeline?.pleResidency === "gpu" ? " (PLE GPU)" : ""
  }${(arm.pipeline?.decodeAhead ?? 1) >= 2 ? ` (ahead ${arm.pipeline?.decodeAhead})` : ""}${
    arm.session?.submitPolicy === undefined
      ? ""
      : ` (first chunk ${arm.session.submitPolicy.firstChunkSize ?? "off"})`
  }${arm.samplerKind === "distribution" ? " (sampled)" : ""}${arm.noSignal === true ? " (no signal)" : ""}${
    arm.tap
      ? arm.split === true
        ? " (split)"
        : arm.hostMarks === true
          ? " (host)"
          : ""
      : " (tap off)"
  }`;

/**
 * The arms of an empty `arms` parameter: the default model of each loaded distribution at its
 * default preset, tap on (regular Gemma 4 first, so it is arm 1, the one the others are compared
 * with).
 */
export const defaultArmDrafts = (manifests: GemmaManifests): readonly ArmDraft[] =>
  GEMMA_MOUNTS.flatMap((mount) => {
    const manifest = manifests[mount];
    if (manifest === undefined) return [];
    return [{ key: { mount, model: manifest.defaultModel }, tap: true }];
  });

export type ArmResolution =
  | { readonly ok: true; readonly arm: BenchArm }
  | { readonly ok: false; readonly error: string };

/**
 * A draft → its arm, or why it cannot run (a missing manifest, a model or a preset the manifest does
 * not have — named with the ones it has —, a QAT model the entry point refuses, linear at `reorder`
 * asked of a preset that declares it — the arm would be its base under another label —, or a
 * run-ahead depth without the PLE on the GPU — the pipeline refuses that pair too, but only after the
 * arm was started).
 */
export const resolveArm = (draft: ArmDraft, manifests: GemmaManifests): ArmResolution => {
  const { mount, model } = draft.key;
  const manifest: Manifest | undefined = manifests[mount];
  if (manifest === undefined) return { ok: false, error: `The ${mount} manifest is not loaded.` };
  const refusal = modelRefusal(draft.key, manifest);
  if (refusal?.kind === "unknown") {
    return {
      ok: false,
      error: `The ${mount} manifest has no model '${model}'; its models are ${refusal.models.join(", ")}.`,
    };
  }
  if (refusal?.kind === "not-mobile") {
    return { ok: false, error: `gemma4QatMobile.Pipeline does not accept the model '${model}'.` };
  }
  const preset = modelPreset(manifest, model, draft.quant);
  if (!preset.ok) {
    return {
      ok: false,
      error: `The ${mount} model '${model}' has no preset '${preset.chosen}'; its presets are ${preset.presets.join(", ")}.`,
    };
  }
  const declaredSampler = modelConfig(manifest, model).sampler;
  const decodeAhead = Number(draft.decodeAhead ?? "1");
  // 12B has no PLE: its token id feeds the graph directly, and there is nothing to place.
  if (model === "12b" && draft.pleOnGpu === true) {
    return { ok: false, error: "12B has no PLE to place on the GPU." };
  }
  if (decodeAhead >= 2 && model !== "12b" && draft.pleOnGpu !== true) {
    return {
      ok: false,
      error:
        "Decode ahead needs PLE on GPU (each step's token stays on the GPU, so its PLE gather must run there).",
    };
  }
  const pipeline: ArmPipelineOptions = {
    ...(draft.pleOnGpu === true ? { pleResidency: "gpu" } : {}),
    ...(decodeAhead >= 2 ? { decodeAhead } : {}),
  };
  if (
    draft.linearReorder === true &&
    declaredLinearLevel(preset.declaration.session) === "reorder"
  ) {
    return {
      ok: false,
      error: `The preset '${preset.name}' declares linear at the level reorder already.`,
    };
  }
  // The integer GEMV's options hold linear at `reorder` where the preset does not: either switch
  // alone, or both, gives one `numerics`.
  const session: ArmSessionOptions = {
    ...(draft.linearReorder === true ? LINEAR_REORDER_SESSION : {}),
    ...(draft.integerGemv === true ? integerGemvSession(preset.declaration.session) : {}),
    ...(draft.firstChunk === undefined ? {} : { submitPolicy: firstChunkPolicy(draft.firstChunk) }),
  };
  return {
    ok: true,
    arm: {
      mount,
      model,
      quant: preset.name,
      quantLabel: preset.declaration.label ?? preset.name,
      isDefaultQuant: preset.isDefault,
      tap: draft.tap,
      // The arms parser refuses split and host without the tap, and the two together
      // (`parseArmsText`): it owns those rules.
      ...(draft.split === true ? { split: true } : {}),
      ...(draft.linearReorder === true ? { linearReorder: true } : {}),
      ...(draft.hostMarks === true ? { hostMarks: true } : {}),
      ...(draft.noSignal === true ? { noSignal: true } : {}),
      ...(draft.sampled === true ? { samplerKind: "distribution" as const } : {}),
      ...(Object.keys(session).length === 0 ? {} : { session }),
      ...(Object.keys(pipeline).length === 0 ? {} : { pipeline }),
      ...(declaredSampler === undefined ? {} : { declaredSampler }),
    },
  };
};

/**
 * Which sampler every chat of every arm is sent with: greedy (temperature 0, the same tokens in
 * every round) or the distribution's sampler with seed 0.
 */
export const SAMPLER_KINDS = ["greedy", "distribution"] as const;
export type SamplerKind = (typeof SAMPLER_KINDS)[number];

/**
 * The sampler a chat is sent with: greedy, or the distribution's declared sampler with seed 0
 * (greedy when it declares none). Always passed explicitly, so the record names exactly what ran.
 */
export const samplerOf = (kind: SamplerKind, declared: DefaultSampler | undefined): SamplerSpec =>
  kind === "greedy" || declared === undefined
    ? { temperature: 0 }
    : { temperature: declared.temperature, topK: declared.topK, topP: declared.topP, seed: 0 };

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
