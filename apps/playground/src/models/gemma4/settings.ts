/**
 * The Gemma 4 chat page's settings: the models of the three mounts (regular Gemma 4, the q4_0 QAT
 * line and the mobile QAT distribution), the per-turn form (sampler and max new tokens) and the chat
 * form (the settings a `ChatSession` takes at construction), with their parsing. Pure, so `vp test`
 * covers it.
 */

import type { Manifest } from "@karume/hub";
import {
  dropOldestTurns,
  type ChatOverflowPolicy,
  type DefaultSampler,
  type PipelineConfig,
  parsePipelineConfig,
  type SamplerSpec,
} from "@karume/models/gemma";

import type { Choice } from "@/components/choice-select";

import { SAMPLE_TOOL_NAMES, type SampleToolName } from "./sample-tools";

/**
 * The mounts this page reads, each built with its family's entry point ({@link entryPointOf}):
 * regular Gemma 4 (`karume-gemma4`), the q4_0 QAT line (`karume-gemma4-qat` — a `gemma4/1`
 * distribution with MTP drafters) and the mobile QAT distribution (`karume-gemma4-qat-mobile`).
 */
export const GEMMA_MOUNTS = ["gemma4", "gemma4-qat", "gemma4-qat-mobile"] as const;
export type GemmaMount = (typeof GEMMA_MOUNTS)[number];

export const MOUNT_LABELS: Readonly<Record<GemmaMount, string>> = {
  gemma4: "Gemma 4",
  "gemma4-qat": "Gemma 4 QAT (q4_0)",
  "gemma4-qat-mobile": "Gemma 4 mobile QAT",
};

/** The entry point a saved record names. */
export type GemmaEntryPoint = "gemma.Pipeline" | "gemma4QatMobile.Pipeline";

/**
 * The entry point a mount is built with: `gemma4QatMobile.Pipeline` for the mobile QAT
 * distribution, the regular `Pipeline` for the others (the q4_0 QAT line is a `gemma4/1`
 * distribution).
 */
export const entryPointOf = (mount: GemmaMount): GemmaEntryPoint =>
  mount === "gemma4-qat-mobile" ? "gemma4QatMobile.Pipeline" : "gemma.Pipeline";

/** One model of one mount (the select's value is `<mount>/<model>`). */
export type ModelKey = { readonly mount: GemmaMount; readonly model: string };

export const modelKeyValue = ({ mount, model }: ModelKey): string => `${mount}/${model}`;

const isGemmaMount = (value: string): value is GemmaMount =>
  GEMMA_MOUNTS.some((mount) => mount === value);

export const parseModelKey = (value: string): ModelKey | undefined => {
  const slash = value.indexOf("/");
  if (slash < 0) return undefined;
  const mount = value.slice(0, slash);
  const model = value.slice(slash + 1);
  return isGemmaMount(mount) && model !== "" ? { mount, model } : undefined;
};

/** The models `gemma4QatMobile.Pipeline.fromPretrained` accepts (`model` is typed `"e2b" | "e4b"`). */
export type QatModel = "e2b" | "e4b";
export const isQatModel = (model: string): model is QatModel => model === "e2b" || model === "e4b";

/** The manifests that have loaded, by mount (a missing or failed one is absent). */
export type GemmaManifests = Partial<Record<GemmaMount, Manifest>>;

/**
 * Every model of every loaded manifest, in mount order then manifest order, each manifest's
 * `defaultModel` marked "(default)". A mobile QAT model the entry point does not accept is listed disabled
 * (choosing it could only fail at load).
 */
export const gemmaModelChoices = (manifests: GemmaManifests): readonly Choice<string>[] =>
  GEMMA_MOUNTS.flatMap((mount) => {
    const manifest = manifests[mount];
    if (manifest === undefined) return [];
    return Object.keys(manifest.models).map((model) => {
      const unsupported = mount === "gemma4-qat-mobile" && !isQatModel(model);
      return {
        value: modelKeyValue({ mount, model }),
        label: `${MOUNT_LABELS[mount]} — ${model}${model === manifest.defaultModel ? " (default)" : ""}${
          unsupported ? " (not accepted by gemma4QatMobile.Pipeline)" : ""
        }`,
        ...(unsupported ? { disabled: true } : {}),
      };
    });
  });

/**
 * The model the page starts with: the default model of the first mount (in {@link GEMMA_MOUNTS}
 * order) whose manifest is there.
 */
export const defaultModelKey = (manifests: GemmaManifests): ModelKey | undefined => {
  for (const mount of GEMMA_MOUNTS) {
    const manifest = manifests[mount];
    if (manifest !== undefined) return { mount, model: manifest.defaultModel };
  }
  return undefined;
};

/** The chosen model when the manifests still hold it, else the default. */
export const effectiveModelKey = (
  chosen: ModelKey | undefined,
  manifests: GemmaManifests,
): ModelKey | undefined =>
  chosen !== undefined && manifests[chosen.mount]?.models[chosen.model] !== undefined
    ? chosen
    : defaultModelKey(manifests);

/** The model's static wiring as the distribution declares it (`pipelineConfig`, validated). */
export const modelConfig = (manifest: Manifest, model: string): PipelineConfig => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  return parsePipelineConfig(entry.pipelineConfig);
};

/**
 * Whether the model has drafter weights, i.e. whether the MTP drafter (`speculative` of
 * `Pipeline.fromPretrained`) can be built (every preset then maps them — a quant is a complete
 * mapping of the model's weights). The regular entry point only: `gemma4QatMobile.Pipeline` takes no
 * drafter.
 */
export const hasDrafter = (manifest: Manifest, { mount, model }: ModelKey): boolean =>
  entryPointOf(mount) === "gemma.Pipeline" &&
  manifest.models[model]?.weights["drafter"] !== undefined;

/**
 * Whether the model has the image embedder's weights, i.e. whether the page loads with `vision` and
 * shows the attach control (ADR 0144; every preset then maps them). The regular entry point only.
 */
export const hasVision = (manifest: Manifest, { mount, model }: ModelKey): boolean =>
  entryPointOf(mount) === "gemma.Pipeline" &&
  manifest.models[model]?.weights["vision"] !== undefined;

/**
 * Whether the model is a mixture of experts, i.e. whether the page offers the exact expert cache
 * (`expertCache` of `Pipeline.fromPretrained` — ADR 0146). Decided by the model key: neither the
 * manifest nor `pipelineConfig` declares the routing (only the graph inside the container has
 * `expert_linear`, and reading it means opening the container), and the one such model the mounts
 * hold is Gemma 4 26B A4B of the q4_0 QAT line. The pipeline still refuses a graph without
 * `expert_linear`, so a wrong entry here fails at Load rather than running something else.
 */
export const hasExpertCache = ({ mount, model }: ModelKey): boolean =>
  mount === "gemma4-qat" && model === "26b";

/**
 * The slot range the page accepts: 2 × 26B's top-k 8 (a prefill chunk has at least 2 rows, and all
 * of one chunk's picks must fit) to its 128 experts per layer — the bounds the pipeline checks.
 */
export const EXPERT_CACHE_SLOTS = { min: 16, max: 128 } as const;

/**
 * The "Expert cache slots" field → the slots (`undefined` = empty: no cache, every expert resident),
 * or the problem. Anything else must be an integer in {@link EXPERT_CACHE_SLOTS}.
 */
export const parseExpertCacheSlots = (text: string): Parsed<number | undefined> => {
  if (text.trim() === "") return { ok: true, value: undefined };
  const slots = integerOf(text);
  if (slots === undefined || slots < EXPERT_CACHE_SLOTS.min || slots > EXPERT_CACHE_SLOTS.max) {
    return {
      ok: false,
      errors: [
        `Expert cache slots must be an integer from ${EXPERT_CACHE_SLOTS.min} to ${EXPERT_CACHE_SLOTS.max}, or empty for none (every expert resident).`,
      ],
    };
  }
  return { ok: true, value: slots };
};

const GIB = 2 ** 30;

/**
 * The "Host tier (GiB)" field's word for every expert: the budget `Number.MAX_SAFE_INTEGER`, which
 * the runtime reads as "at or above every expert's bytes" (`expertCache.hostBytes`).
 */
export const EXPERT_HOST_TIER_ALL = "all";

/**
 * The "Host tier (GiB)" field → `expertCache.hostBytes` (`undefined` = empty: no host tier, every
 * miss reads the distribution), or the problem. {@link EXPERT_HOST_TIER_ALL} (any case) is every
 * expert; anything else must be a positive decimal number of GiB, taken down to whole bytes. A
 * budget below one expert passes here and is refused by the pipeline at Load (an expert's bytes are
 * the distribution's, which the page has not read).
 */
export const parseExpertHostTier = (text: string): Parsed<number | undefined> => {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: true, value: undefined };
  if (trimmed.toLowerCase() === EXPERT_HOST_TIER_ALL) {
    return { ok: true, value: Number.MAX_SAFE_INTEGER };
  }
  const bytes = /^\d+(\.\d+)?$/.test(trimmed) ? Math.floor(Number(trimmed) * GIB) : undefined;
  if (bytes === undefined || !Number.isSafeInteger(bytes) || bytes < 1) {
    return {
      ok: false,
      errors: [
        `Host tier must be a positive number of GiB, "${EXPERT_HOST_TIER_ALL}" for every expert, or empty for none (every miss reads the distribution).`,
      ],
    };
  }
  return { ok: true, value: bytes };
};

/**
 * The range the "Self-speculation depth" field accepts (`expertCache.depth` — ADR 0146 decision 8).
 * The bound that depends on the slots (`depth + 1 ≤ ⌊slots / 8⌋` on 26B) is the pipeline's to check:
 * a depth it refuses fails at Load with its own message.
 */
export const EXPERT_SPECULATION_DEPTH = { min: 1, max: 4 } as const;

/**
 * The "Self-speculation depth" field → `expertCache.depth` (`undefined` = empty: no
 * self-speculation, every decode token is one run re-run after a miss), or the problem. Anything
 * else must be an integer in {@link EXPERT_SPECULATION_DEPTH}.
 */
export const parseExpertSpeculationDepth = (text: string): Parsed<number | undefined> => {
  if (text.trim() === "") return { ok: true, value: undefined };
  const depth = integerOf(text);
  if (
    depth === undefined ||
    depth < EXPERT_SPECULATION_DEPTH.min ||
    depth > EXPERT_SPECULATION_DEPTH.max
  ) {
    return {
      ok: false,
      errors: [
        `Self-speculation depth must be an integer from ${EXPERT_SPECULATION_DEPTH.min} to ${EXPERT_SPECULATION_DEPTH.max}, or empty for none (one run per decode token).`,
      ],
    };
  }
  return { ok: true, value: depth };
};

/**
 * The `speculative` a chat passes for what was loaded. MUST: `"always"` with a drafter on f16 full
 * slots, because a turn that leaves it out selects the self-tuning gate, which the pipeline refuses
 * there (ADR 0143 decision 8). Everywhere else the session keeps its default.
 */
export const chatSpeculative = (loaded: {
  readonly speculative: boolean;
  readonly fullStateStorage?: "f16";
}): { readonly speculative?: "always" } =>
  loaded.speculative && loaded.fullStateStorage === "f16" ? { speculative: "always" } : {};

/** The per-turn form, as typed (parsed by {@link parseTurnForm}). */
export type TurnForm = {
  readonly temperature: string;
  readonly topK: string;
  readonly topP: string;
  readonly seed: string;
  readonly maxNewTokens: string;
};

const DEFAULT_MAX_NEW_TOKENS = 512;

/**
 * The form filled from the distribution's declared sampler (greedy when it declares none), seed 0
 * and {@link DEFAULT_MAX_NEW_TOKENS}. The page always passes the sampler it shows, so the form is
 * where the distribution's default enters.
 */
export const turnFormOf = (sampler: DefaultSampler | undefined): TurnForm => ({
  temperature: String(sampler?.temperature ?? 0),
  topK: sampler === undefined ? "" : String(sampler.topK),
  topP: sampler === undefined ? "" : String(sampler.topP),
  seed: "0",
  maxNewTokens: String(DEFAULT_MAX_NEW_TOKENS),
});

/** What one turn is sent with. */
export type TurnSettings = {
  /** Absent = no limit: the turn generates until the model stops or the context is full. */
  readonly maxNewTokens?: number;
  readonly sampler: SamplerSpec;
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

const numberOf = (text: string): number | undefined => {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : undefined;
};

/**
 * The per-turn form → the turn's `maxNewTokens` and sampler, or every problem at once. The ranges
 * are the sampler's own (`SamplerSpec`): temperature ≥ 0, top-k an integer ≥ 1, 0 < top-p ≤ 1, seed
 * a non-negative safe integer. An empty top-k / top-p is left out (no narrowing) — never silently
 * replaced by a default, so the record names exactly what ran. An empty max new tokens is left out
 * the same way (the turn then has no limit); anything else must be an integer ≥ 1.
 */
export const parseTurnForm = (form: TurnForm): Parsed<TurnSettings> => {
  const errors: string[] = [];
  const temperature = numberOf(form.temperature);
  if (temperature === undefined || temperature < 0) {
    errors.push("Temperature must be a number ≥ 0.");
  }
  let topK: number | undefined;
  if (form.topK.trim() !== "") {
    topK = integerOf(form.topK);
    if (topK === undefined || topK < 1) errors.push("Top-k must be an integer ≥ 1 (or empty).");
  }
  let topP: number | undefined;
  if (form.topP.trim() !== "") {
    topP = numberOf(form.topP);
    if (topP === undefined || topP <= 0 || topP > 1) {
      errors.push("Top-p must be in (0, 1] (or empty).");
    }
  }
  const seed = integerOf(form.seed);
  if (seed === undefined) errors.push("Seed must be a non-negative integer.");
  let maxNewTokens: number | undefined;
  if (form.maxNewTokens.trim() !== "") {
    maxNewTokens = integerOf(form.maxNewTokens);
    if (maxNewTokens === undefined || maxNewTokens < 1) {
      errors.push("Max new tokens must be an integer ≥ 1 (or empty).");
    }
  }
  if (errors.length > 0 || temperature === undefined || seed === undefined) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    value: {
      ...(maxNewTokens === undefined ? {} : { maxNewTokens }),
      sampler: {
        temperature,
        ...(topK === undefined ? {} : { topK }),
        ...(topP === undefined ? {} : { topP }),
        seed,
      },
    },
  };
};

/**
 * What the session does when a turn does not fit the capacity: drop the oldest user / assistant pair
 * until it fits (`dropOldestTurns`, the session's default), or refuse the turn with
 * `GenerationCapacityError`.
 */
export type OverflowMode = "drop-oldest" | "refuse";

export const OVERFLOW_CHOICES: readonly Choice<OverflowMode>[] = [
  { value: "drop-oldest", label: "Drop the oldest turns (dropOldestTurns)" },
  { value: "refuse", label: "Refuse the turn (GenerationCapacityError)" },
];

/**
 * The session's overflow policy for a mode, reporting how many history entries it removed (its input
 * length minus its output length — the page reads this to tell which entries left the context).
 * Refusing returns the history unchanged: the session then fails the turn with
 * `GenerationCapacityError` (its documented contract for a policy that cannot shrink the history).
 */
export const overflowPolicy =
  (mode: OverflowMode, onDropped: (count: number) => void): ChatOverflowPolicy =>
  async (context) => {
    const next = mode === "refuse" ? context.turns : await dropOldestTurns(context);
    onDropped(context.turns.length - next.length);
    return next;
  };

/** The chat form, as typed (parsed by {@link parseChatForm}). */
export type ChatForm = {
  readonly system: string;
  /** Empty = the distribution's capacity. */
  readonly capacity: string;
  readonly overflow: OverflowMode;
  /** Thinking mode (`enableThinking`); absent = off. */
  readonly thinking?: boolean;
  /** The sample tools checked (any order); absent = none. */
  readonly tools?: readonly SampleToolName[];
};

export const INITIAL_CHAT_FORM: ChatForm = {
  system: "",
  capacity: "",
  overflow: "drop-oldest",
  thinking: false,
  tools: [],
};

/** What one `ChatSession` is built with (fixed for the session's life). */
export type ChatSettings = {
  /** `undefined` = no system message. */
  readonly system?: string;
  readonly capacity: number;
  readonly overflow: OverflowMode;
  /** Present (true) only when thinking mode is on. */
  readonly enableThinking?: true;
  /** The sample tools the session declares, in {@link SAMPLE_TOOL_NAMES} order; absent = none. */
  readonly tools?: readonly SampleToolName[];
};

/**
 * Whether the session streams the template's structure spellings (`<|channel>`, `<|tool_call>`,
 * `<|"|>`): it does with tools or thinking (the NOTE of `ChatSession.send`). The streamed text is
 * then not the text the history keeps.
 */
export const streamsStructure = (settings: ChatSettings): boolean =>
  settings.enableThinking === true || settings.tools !== undefined;

/** The limits a capacity is checked against (from the model's `pipelineConfig`). */
export type CapacityLimits = Pick<PipelineConfig, "capacity" | "chunkLength" | "maxPosition">;

/**
 * The chat form → the session's settings. A whitespace-only system prompt means none. The capacity
 * must satisfy what the session checks at construction: an integer with `chunkLength ≤ capacity ≤
 * maxPosition`. Thinking and the tools appear only when chosen, so a plain chat's settings are what
 * they were before either existed; the tools come out in {@link SAMPLE_TOOL_NAMES} order (the
 * declaration order the session renders, and what {@link sameChatSettings} compares).
 */
export const parseChatForm = (form: ChatForm, limits: CapacityLimits): Parsed<ChatSettings> => {
  let capacity = limits.capacity;
  if (form.capacity.trim() !== "") {
    const parsed = integerOf(form.capacity);
    if (parsed === undefined || parsed < limits.chunkLength || parsed > limits.maxPosition) {
      return {
        ok: false,
        errors: [
          `Capacity must be an integer from ${limits.chunkLength} (the chunk length) to ${limits.maxPosition} (the position limit), or empty for the distribution's ${limits.capacity}.`,
        ],
      };
    }
    capacity = parsed;
  }
  const chosen = form.tools ?? [];
  const tools = SAMPLE_TOOL_NAMES.filter((name) => chosen.includes(name));
  return {
    ok: true,
    value: {
      ...(form.system.trim() === "" ? {} : { system: form.system }),
      capacity,
      overflow: form.overflow,
      ...(form.thinking === true ? { enableThinking: true } : {}),
      ...(tools.length === 0 ? {} : { tools }),
    },
  };
};

const sameTools = (
  left: readonly SampleToolName[] | undefined,
  right: readonly SampleToolName[] | undefined,
): boolean =>
  (left ?? []).length === (right ?? []).length &&
  (left ?? []).every((name, index) => right?.[index] === name);

export const sameChatSettings = (left: ChatSettings, right: ChatSettings): boolean =>
  left.system === right.system &&
  left.capacity === right.capacity &&
  left.overflow === right.overflow &&
  (left.enableThinking ?? false) === (right.enableThinking ?? false) &&
  sameTools(left.tools, right.tools);
