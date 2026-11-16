/**
 * The Gemma 4 lookups and text forms the workbench's Gemma entries use: the three mounts and the
 * entry point each is built with, a model's declared `pipelineConfig`, which model takes the exact
 * expert cache, and the text forms of the cache's slots and host tier with their parsing. Copied
 * from the Playground's chat page settings (`apps/playground/src/models/gemma4/settings.ts`), which
 * keeps its own: the workbench and the Playground share no module. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import { type PipelineConfig, parsePipelineConfig } from "@karume/models/gemma";

/**
 * The mounts the Gemma entries read, each built with its family's entry point ({@link entryPointOf}):
 * regular Gemma 4 (`karume-gemma4`), the q4_0 QAT line (`karume-gemma4-qat` — a `gemma4/1`
 * distribution with MTP drafters) and the mobile QAT distribution (`karume-gemma4-qat-mobile`).
 */
export const GEMMA_MOUNTS = ["gemma4", "gemma4-qat", "gemma4-qat-mobile"] as const;
export type GemmaMount = (typeof GEMMA_MOUNTS)[number];

export const isGemmaMount = (value: string): value is GemmaMount =>
  GEMMA_MOUNTS.some((mount) => mount === value);

/** The entry point a record names. */
export type GemmaEntryPoint = "gemma.Pipeline" | "gemma4QatMobile.Pipeline";

/**
 * The entry point a mount is built with: `gemma4QatMobile.Pipeline` for the mobile QAT
 * distribution, the regular `Pipeline` for the others (the q4_0 QAT line is a `gemma4/1`
 * distribution).
 */
export const entryPointOf = (mount: GemmaMount): GemmaEntryPoint =>
  mount === "gemma4-qat-mobile" ? "gemma4QatMobile.Pipeline" : "gemma.Pipeline";

/** One model of one mount (written `<mount>/<model>`). */
export type ModelKey = { readonly mount: GemmaMount; readonly model: string };

/** The models `gemma4QatMobile.Pipeline.fromPretrained` accepts (`model` is typed `"e2b" | "e4b"`). */
export type QatModel = "e2b" | "e4b";
export const isQatModel = (model: string): model is QatModel => model === "e2b" || model === "e4b";

/** The manifests that have loaded, by mount (a mount whose mirror is missing is absent). */
export type GemmaManifests = Partial<Record<GemmaMount, Manifest>>;

/** The model's static wiring as the distribution declares it (`pipelineConfig`, validated). */
export const modelConfig = (manifest: Manifest, model: string): PipelineConfig => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  return parsePipelineConfig(entry.pipelineConfig);
};

/**
 * Whether the model is a mixture of experts, i.e. whether the exact expert cache (`expertCache` of
 * `Pipeline.fromPretrained`) applies. Decided by the model key: neither the manifest nor
 * `pipelineConfig` declares the routing (only the graph inside the container has `expert_linear`,
 * and reading it means opening the container), and the one such model the mounts hold is Gemma 4
 * 26B A4B of the q4_0 QAT line. The pipeline still refuses a graph without `expert_linear`, so a
 * wrong entry here fails at the build rather than running something else.
 */
export const hasExpertCache = ({ mount, model }: ModelKey): boolean =>
  mount === "gemma4-qat" && model === "26b";

/**
 * The slot range accepted (for each layer of a list too): 2 × 26B's top-k 8 (a prefill chunk has at
 * least 2 rows, and all of one chunk's picks must fit) to its 128 experts per layer — the bounds the
 * pipeline checks.
 */
export const EXPERT_CACHE_SLOTS = { min: 16, max: 128 } as const;

/**
 * The slots of `expertCache`: one count for every routed layer, or one count per routed layer in
 * layer order (the pipeline checks the length against the model).
 */
export type ExpertCacheSlots = number | readonly number[];

/** The slots' text form's word for {@link FITTED_EXPERT_SLOTS}. */
export const EXPERT_CACHE_FITTED = "fitted";

/**
 * 26B's per-layer slots fitted at a mean of 64, layer 0 first: the allocation of
 * docs/research/2026-11-08-gemma4-26b-moe-cache.md §4.2, fitted on CPU traces by a local search that
 * keeps the mean at 64 — the same total VRAM as 64 slots in every layer, moved to the early layers
 * so a run's first miss comes later. Marked pure so a production build, which leaves out the entries
 * that read a distribution, drops it with them.
 */
export const FITTED_EXPERT_SLOTS: readonly number[] = /* @__PURE__ */ Object.freeze([
  96, 108, 84, 88, 84, 68, 72, 72, 68, 60, 52, 56, 64, 64, 60, 68, 52, 56, 64, 56, 48, 40, 44, 48,
  52, 48, 56, 60, 64, 68,
]);

export type Parsed<Value> =
  | { readonly ok: true; readonly value: Value }
  | { readonly ok: false; readonly errors: readonly string[] };

const integerOf = (text: string): number | undefined => {
  const trimmed = text.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : undefined;
};

const slotCountOf = (text: string): number | undefined => {
  const slots = integerOf(text);
  return slots === undefined || slots < EXPERT_CACHE_SLOTS.min || slots > EXPERT_CACHE_SLOTS.max
    ? undefined
    : slots;
};

/**
 * The slots' text → the slots (`undefined` = empty: no cache, every expert resident), or the
 * problem. {@link EXPERT_CACHE_FITTED} (any case) is {@link FITTED_EXPERT_SLOTS}; text with a comma
 * is a list, one count per routed layer; anything else is one count for every layer. Every count
 * must be an integer in {@link EXPERT_CACHE_SLOTS}. A list of the wrong length passes here and is
 * refused by the pipeline at the build (the layer count is the model's, which is not read here).
 */
export const parseExpertCacheSlots = (text: string): Parsed<ExpertCacheSlots | undefined> => {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: true, value: undefined };
  if (trimmed.toLowerCase() === EXPERT_CACHE_FITTED) {
    return { ok: true, value: FITTED_EXPERT_SLOTS };
  }
  const forms = `Expert cache slots must be an integer from ${EXPERT_CACHE_SLOTS.min} to ${EXPERT_CACHE_SLOTS.max}, a comma-separated list of such integers (one per routed layer, layer 0 first), "${EXPERT_CACHE_FITTED}" for the fitted allocation of 26B, or empty for none (every expert resident).`;
  if (!trimmed.includes(",")) {
    const slots = slotCountOf(trimmed);
    return slots === undefined ? { ok: false, errors: [forms] } : { ok: true, value: slots };
  }
  const entries = trimmed.split(",").map((entry) => entry.trim());
  const counts: number[] = [];
  const wrong: string[] = [];
  entries.forEach((entry, layer) => {
    const count = slotCountOf(entry);
    if (count === undefined) wrong.push(`Layer ${layer} ("${entry}") is not.`);
    else counts.push(count);
  });
  return wrong.length > 0 ? { ok: false, errors: [forms, ...wrong] } : { ok: true, value: counts };
};

const GIB = 2 ** 30;

/**
 * The host tier's word for every expert: the budget `Number.MAX_SAFE_INTEGER`, which the runtime
 * reads as "at or above every expert's bytes" (`expertCache.hostBytes`).
 */
export const EXPERT_HOST_TIER_ALL = "all";

/**
 * The host tier's text (GiB) → `expertCache.hostBytes` (`undefined` = empty: no host tier, every
 * miss reads the distribution), or the problem. {@link EXPERT_HOST_TIER_ALL} (any case) is every
 * expert; anything else must be a positive decimal number of GiB, taken down to whole bytes. A
 * budget below one expert passes here and is refused by the pipeline at the build (an expert's
 * bytes are the distribution's, which are not read here).
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
 * The range of the self-speculation depth (`expertCache.depth`). The bound that depends on the slots
 * (`depth + 1 ≤ ⌊slots / 8⌋` on 26B, with the fewest layer's slots for a list) is the pipeline's to
 * check: a depth it refuses fails at the build with its own message.
 */
export const EXPERT_SPECULATION_DEPTH = { min: 1, max: 4 } as const;
