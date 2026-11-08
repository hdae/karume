/**
 * Schema validation of gemma4's `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns
 * and validates its schema).
 *
 * hub passes `pipelineConfig` through (it checks only the removal of forbidden keys and the size
 * limits), so **this module is canonical for the shape**, and every hand-written check runs at
 * parse time. The baking side's canonical source is `tools/export-recipes/gemma4/distribution.py`
 * (`gemma4_pipeline_config` / `gemma4_sampler`).
 *
 * MUST: unknown keys fail loudly (if a misspelling such as `{ capacity: 640, capasity: 640 }`
 * silently degraded to the default, the declaration the distributor intended and the execution
 * would differ unnoticed).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 *
 * NOTE: the public distribution repository tables ({@link SOURCES}, and {@link QAT_MOBILE_SOURCES}
 * of the mobile QAT family) also live here. They are not values derivable from the manifest but the
 * "which manifest to fetch" side, so a distribution cannot carry them (ADR 0073).
 */

import type { HubRepoRef } from "@karume/hub";

import { assertAllowedKeys, isRecord } from "../config/readers.ts";
import {
  assertGemma4RopeSpec,
  GEMMA4_ROPE_LAYER_TYPES,
  type RopeLayerSpec,
  type RopeLayerType,
  type RopeSpec,
} from "./rope.ts";
import { CHAT_DIALECTS, type ChatDialect, isChatDialect } from "./text/chat-dialect.ts";
import { maxImageSoftTokens, SOFT_TOKEN_BUDGETS } from "./vision.ts";

/** The contract name of `pipeline` and the major this implementation accepts (ADR 0038 §1). */
export const GEMMA4_PIPELINE_NAME = "gemma4";
export const GEMMA4_PIPELINE_MAJOR = 1;

/**
 * The **public distribution repository table** of the Gemma 4 family (ADR 0092 — one table per
 * family, **no default entry**). The values are **the sources this package version verified**
 * (pinned commit SHAs — ADR 0073).
 *
 * The key is the basename of the HF repository name without `karume-` (`"karume-" + key` gives back
 * the repository basename — this invariant is checked by `tests/sources_test.ts`). One entry per
 * repository, several models in each (non-default models are chosen with `{ model }`): the regular
 * line `"gemma4"` (E2B / E4B), and the q4_0 QAT line `"gemma4-qat"` (E2B / E4B / 12B, each with its
 * QAT MTP drafter).
 *
 * Pass it **as an opt-in when you want to follow the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of using this table
 * (`fromPretrained` has no default).
 *
 * MUST: revisions are pinned to commit SHAs — branches and tags can be moved on the distribution
 * side, so the bytes this published package reads would change silently at the network's
 * convenience (an accident on the unrecoverable side). A SHA removes the revision-resolution
 * request itself, so offline start-up with a full cache works at the same time (ADR 0038). A user
 * who needs to follow main explicitly chooses `{ ...SOURCES["gemma4"], revision: "main" }`.
 */
// NOTE: the revision is updated to main's SHA after the upload, in §3 of the release runbook
// (docs/release-runbook.md) (inheriting the maintenance duty of ADR 0073 decision 3 — hand-written
// + runbook gate).
export const SOURCES = {
  gemma4: {
    repo: "hdae/karume-gemma4",
    revision: "a3016952bb63cd4665dfbe68ade9cbac5d2092e8",
  },
  "gemma4-qat": {
    repo: "hdae/karume-gemma4-qat",
    revision: "2ae106960e77c53a921177396524dfcd9a3d9816",
  },
} as const satisfies Record<string, HubRepoRef>;

/**
 * The **public distribution repository table** of the fixed mobile QAT family
 * (`@karume/models/gemma4-qat-mobile` exports it as `SOURCES` — ADR 0092, one table per family,
 * **no default entry**). The values are **the sources this package version verified** (pinned
 * commit SHAs — ADR 0073).
 *
 * The key follows the same rule as {@link SOURCES} (`"karume-" + key` gives back the repository
 * basename — `tests/sources_test.ts`):
 *
 * - `"gemma4-qat-mobile"` = `hdae/karume-gemma4-qat-mobile` (E2B and E4B in one repository, default
 *   E2B; chosen with `{ model }` — self-contained, no cross-repository reference)
 *
 * It lives in this module next to regular Gemma's table because both families share this
 * directory, and the recipe-side check (`tools/export-recipes/tests/test_dist_driver.py`) reads the
 * repositories of `packages/models/src/<family>/config.ts`.
 *
 * MUST: the revision is a commit SHA, for the reasons given on {@link SOURCES}.
 */
// NOTE: the revision is updated to main's SHA after the upload, in §3 of the release runbook
// (docs/release-runbook.md) (inheriting the maintenance duty of ADR 0073 decision 3 — hand-written
// + runbook gate).
export const QAT_MOBILE_SOURCES = {
  "gemma4-qat-mobile": {
    repo: "hdae/karume-gemma4-qat-mobile",
    revision: "d4685bf2ba3026115cc3dec5459e82f9fe6ded98",
  },
} as const satisfies Record<string, HubRepoRef>;

/**
 * The sampler defaults a distribution can declare (a **subset** of `SamplerSpec`).
 *
 * MUST: do not open up the whole `SamplerSpec` — `logitBias` / `seed` / `repetitionPenalty` are not
 * values of the "recommended by the distributor" kind (the upstream `generation_config.json` also
 * carries only these three). Keeping the type as `SamplerSpec` would make it **wider than the
 * acceptance set**, growing fields on the public surface that "pass the type but fail the parser as
 * unknown keys". Add fields when a real need appears.
 *
 * MUST: all three fields are required ({@link parsePipelineConfig} rejects a missing field).
 * Allowing partial declarations would create combinations found **nowhere upstream**, such as
 * "temperature recommended, top-k left at the low-level default".
 */
export type DefaultSampler = {
  /** Temperature dividing the logits before softmax (a finite number ≥ 0). */
  readonly temperature: number;
  /** Narrows the candidates to the top `topK` by probability (an integer ≥ 1). */
  readonly topK: number;
  /**
   * Narrows to the smallest candidate set whose cumulative probability reaches `topP` (0 < topP ≤
   * 1).
   */
  readonly topP: number;
};

/**
 * What a host needs to turn an image into prompt rows (`pipelineConfig.vision` — ADR 0144; the
 * writer's source of truth is `gemma4_vision_config` of `gemma4/distribution.py`).
 */
export type PipelineVisionConfig = {
  /** The most soft tokens one image becomes (upstream's `max_soft_tokens`). */
  readonly softTokenBudget: number;
  /** `<|image|>`: the id of every soft token. */
  readonly imageTokenId: number;
  /** `<|image>`: the marker in front of an image's soft tokens. */
  readonly beginImageTokenId: number;
  /** `<image|>`: the marker behind them. */
  readonly endImageTokenId: number;
  /** The width of one soft token's vector (the embedder's output and `soft_embeds`). */
  readonly hiddenSize: number;
};

/**
 * The **numbers that change with each asset generation** among the static wiring (declared by the
 * distribution's `pipelineConfig`).
 *
 * NOTE: the symbol (the full slots' capacity symbol) is not put here — keeping something derivable
 * from the graph in the declaration as well lets one copy go stale (`capacitySymbolOf` in
 * `./admission.ts`).
 */
export type PipelineConfig = {
  /**
   * The **default** row count of a fixed-length prefill chunk (ADR 0066 decision 4 — a plan-time
   * constant of the context).
   *
   * It is also a runtime option (`PipelineOptions.chunkLength`) — the graph's chunk rows are
   * the symbol `M`, so it can be chosen without re-baking the asset (limited by
   * {@link maxChunkLength}).
   */
  readonly chunkLength: number;
  /**
   * The **upper limit** of values selectable for `chunkLength` (the top of the trace range the
   * symbol `M` was baked with).
   *
   * MUST: required. The IR's `symbols` is only a list of names and carries no symbol limit
   * (`format/ir.ts`), so this is **the only number not derivable from the asset**. Without the
   * declaration, a `chunkLength` outside the trace range would run through without an exception
   * (the values are not necessarily broken, but it is outside the guarantee — against the
   * cross-cutting invariant "fail loudly on the unexpected"). The baking side's canonical source is
   * `gemma4.export.SYM_MAX` (the distribution recipe holds a copy).
   */
  readonly maxChunkLength: number;
  /**
   * The exclusive upper limit of absolute positions **the model declares** (upstream
   * `max_position_embeddings`).
   *
   * NOTE: this used to be "the row count of the baked RoPE tables". The tables left the
   * distribution and the host builds cos / sin only for the chunk (`./rope.ts`), so this number is
   * no longer the physical shape of the asset but **the model's declaration**. What it means to the
   * reader (`capacity` cannot exceed it) is unchanged.
   */
  readonly maxPosition: number;
  /**
   * The **default** capacity of the full slots (the maximum logical length a conversation can use).
   *
   * It is also a runtime option (`capacity` of `createGenerationSequence` /
   * `Pipeline.sequence`) — the capacity is the physical allocation of the state slots itself,
   * so the caller chooses the trade between long conversations and VRAM.
   */
  readonly capacity: number;
  /**
   * Parameters for building RoPE's cos / sin on the host (two, one per layer type — a revision of
   * ADR 0067 decision 4).
   *
   * MUST: required. In a distribution whose tables left the graph, this declaration is **the only
   * source of position encoding** (running without it would run attention without rotation, with no
   * exception). The canonical formula and value ranges live in `./rope.ts`.
   */
  readonly rope: RopeSpec;
  /**
   * The sampler defaults the distribution declares (ADR 0083 decision 7).
   *
   * Used when a request omits `sampler`. Without a declaration the low-level default (temperature 0
   * = greedy) stays, positioned as "no sampler declared = the degenerate temperature-0 form" (the
   * parity check keeps living on this path).
   */
  readonly sampler?: DefaultSampler;
  /**
   * The chat template the model follows (`./text/chat-dialect.ts`). Absent = `"gemma4"` (the E2B /
   * E4B template); gemma-4-12B-it declares `"gemma4-unified"`.
   *
   * Optional so that the manifests written before the 12B model stay byte-identical. When absent the
   * field is omitted from the parsed value as well (the same shape as {@link sampler}); the one
   * place that reads it is the tokenizer construction in `./pipeline.ts`, which carries the dialect
   * to `chatPrompt` / `chatTurn`.
   */
  readonly chatDialect?: ChatDialect;
  /**
   * Token ids the model never generates (upstream `generation_config.json` `suppress_tokens` —
   * gemma-4-12B-it declares `[258882, 258883]`, `<image|>` / `<audio|>`). Their logits are
   * `-Infinity` before every token selection — greedy, sampling and the speculative acceptance —
   * which is what HF `generate()` does with the field in greedy and sampling alike. A request
   * cannot lift it (`GenerationWiring.suppressTokens` in `../generation/program.ts`).
   *
   * Optional, and absent from the parsed value when not declared (the E2B / E4B manifests stay
   * byte-identical; their selection is unchanged). When declared: a non-empty array of distinct
   * non-negative integers in ascending order (the writer sorts them; a declaration in another
   * order is not normalized here — it was not written by that writer). The vocabulary bound is
   * checked at admission, where the vocabulary is first known (`admitGemma4` in `./admission.ts`).
   */
  readonly suppressTokens?: readonly number[];
  /**
   * The image input's declaration (gemma-4-12B-it only). Optional, and absent from the parsed value
   * when not declared (the E2B / E4B manifests stay byte-identical). Having it does not load
   * anything: `PipelineOptions.vision` asks for the embedder.
   */
  readonly vision?: PipelineVisionConfig;
};

const ROOT_KEYS: readonly string[] = [
  "chunkLength",
  "maxChunkLength",
  "maxPosition",
  "capacity",
  "rope",
  "sampler",
  "chatDialect",
  "suppressTokens",
  "vision",
];

/** Accepted fields of `vision` (the type's canonical source is {@link PipelineVisionConfig}). */
const VISION_KEYS: readonly string[] = [
  "softTokenBudget",
  "imageTokenId",
  "beginImageTokenId",
  "endImageTokenId",
  "hiddenSize",
];

/**
 * Accepted sampler fields (the type's canonical source is {@link DefaultSampler} — so is the
 * reason to add any).
 */
const SAMPLER_KEYS: readonly string[] = ["temperature", "topK", "topP"];

/**
 * Accepted fields per rope layer type (the type's canonical source is {@link RopeLayerSpec}
 * in `./rope.ts`).
 */
const ROPE_LAYER_KEYS: readonly string[] = ["theta", "headDim", "rotaryDim"];

/** Checks only that the field exists and is a number (the caller adds the value range). */
const readRawNumber = (raw: Record<string, unknown>, key: string, where: string): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (typeof value !== "number")
    throw new Error(`${where}.${key}: not a number (${String(value)})`);
  return value;
};

const readNumber = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): number => {
  const value = readRawNumber(raw, key, where);
  if (!check(value)) throw new Error(`${where}.${key}: ${requirement} (${String(value)})`);
  return value;
};

/**
 * Reads `sampler` (when the whole field is absent, `undefined` = the low-level default = greedy).
 *
 * The value ranges are the same as `assertSpec` in `generation/sampler.ts` — rejected when the
 * distribution is read, instead of the shape "found out only after generating once".
 */
const parseSampler = (raw: unknown, where: string): DefaultSampler => {
  const record = isRecord(raw) ? raw : undefined;
  if (record === undefined) throw new Error(`${where}: missing / not an object`);
  assertAllowedKeys(record, SAMPLER_KEYS, where);
  return {
    temperature: readNumber(
      record,
      "temperature",
      where,
      (value) => Number.isFinite(value) && value >= 0,
      "not a finite number ≥ 0",
    ),
    topK: readNumber(
      record,
      "topK",
      where,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "not a positive integer",
    ),
    topP: readNumber(
      record,
      "topP",
      where,
      (value) => Number.isFinite(value) && value > 0 && value <= 1,
      "not in the range 0 < topP ≤ 1",
    ),
  };
};

/**
 * Reads the rope parameters of one layer type.
 *
 * MUST: value ranges are not checked here — the numeric acceptance set (positive theta, even
 * headDim, `2 ≤ rotaryDim ≤ headDim`) is canonical in `assertGemma4RopeLayerSpec` of `./rope.ts`,
 * and with the same formula in two implementations one goes stale. This function holds only **the
 * shape of the declaration** (unknown keys, missing fields, types).
 */
const parseRopeLayer = (raw: unknown, where: string): RopeLayerSpec => {
  if (!isRecord(raw)) throw new Error(`${where}: not an object (${String(raw)})`);
  assertAllowedKeys(raw, ROPE_LAYER_KEYS, where);
  return {
    theta: readRawNumber(raw, "theta", where),
    headDim: readRawNumber(raw, "headDim", where),
    rotaryDim: readRawNumber(raw, "rotaryDim", where),
  };
};

/** Reads both layer types and leaves the value ranges to the check in `./rope.ts`. */
const parseRope = (raw: unknown, where: string): RopeSpec => {
  if (!isRecord(raw)) throw new Error(`${where}: not an object (${String(raw)})`);
  assertAllowedKeys(raw, GEMMA4_ROPE_LAYER_TYPES, where);
  const layer = (layerType: RopeLayerType): RopeLayerSpec => {
    if (!Object.hasOwn(raw, layerType)) throw new Error(`${where}.${layerType}: missing`);
    return parseRopeLayer(raw[layerType], `${where}.${layerType}`);
  };
  const spec: RopeSpec = {
    sliding_attention: layer("sliding_attention"),
    full_attention: layer("full_attention"),
  };
  assertGemma4RopeSpec(where, spec);
  return spec;
};

/**
 * Reads `chatDialect` (called only when the key is present — absence is the default dialect, a
 * present value outside the closed set fails loudly).
 */
const parseChatDialect = (raw: unknown, where: string): ChatDialect => {
  if (!isChatDialect(raw)) {
    throw new Error(
      `${where}: ${JSON.stringify(raw)} is not one of ${CHAT_DIALECTS.join(" / ")}` +
        ` (rendering a conversation in another model's template still generates text, so an unknown` +
        ` dialect is not mapped to the default)`,
    );
  }
  return raw;
};

/**
 * Reads `suppressTokens` (called only when the key is present — absence means none, and a present
 * value must declare at least one id: an empty list would be a declaration that does nothing).
 */
const parseSuppressTokens = (raw: unknown, where: string): readonly number[] => {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error(`${where}: ${JSON.stringify(raw)} is not a non-empty array of token ids`);
  }
  const tokens: number[] = [];
  for (const [index, token] of raw.entries()) {
    if (typeof token !== "number" || !Number.isSafeInteger(token) || token < 0) {
      throw new Error(`${where}[${index}]: ${JSON.stringify(token)} is not a non-negative integer`);
    }
    // Strictly ascending covers both "sorted" and "no duplicates" in one comparison.
    const previous = tokens.at(-1);
    if (previous !== undefined && token <= previous) {
      throw new Error(
        `${where}[${index}]: ${token} does not follow ${previous} in strictly ascending order` +
          ` (the ids are distinct and written sorted)`,
      );
    }
    tokens.push(token);
  }
  return tokens;
};

/**
 * Reads `vision` (called only when the key is present). The vocabulary bound of the ids is checked
 * at admission, where the vocabulary is first known (`assertGemma4Vision` in `./admission.ts`).
 */
const parseVision = (raw: unknown, where: string): PipelineVisionConfig => {
  if (!isRecord(raw)) throw new Error(`${where}: not an object (${String(raw)})`);
  assertAllowedKeys(raw, VISION_KEYS, where);
  const tokenId = (key: string): number =>
    readNumber(
      raw,
      key,
      where,
      (value) => Number.isSafeInteger(value) && value >= 0,
      "not a non-negative integer",
    );
  const vision: PipelineVisionConfig = {
    softTokenBudget: readNumber(
      raw,
      "softTokenBudget",
      where,
      (value) => SOFT_TOKEN_BUDGETS.includes(value),
      `not one of ${SOFT_TOKEN_BUDGETS.join(" / ")}`,
    ),
    imageTokenId: tokenId("imageTokenId"),
    beginImageTokenId: tokenId("beginImageTokenId"),
    endImageTokenId: tokenId("endImageTokenId"),
    hiddenSize: readNumber(
      raw,
      "hiddenSize",
      where,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "not a positive integer",
    ),
  };
  // A soft token that is also its own marker would render a prompt the model never saw.
  const ids = [vision.imageTokenId, vision.beginImageTokenId, vision.endImageTokenId];
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${where}: the three image token ids are not distinct (${ids.join(", ")})`);
  }
  return vision;
};

/**
 * Checks an `unknown` (the raw declaration hub passed through) and turns it into a
 * {@link PipelineConfig}.
 *
 * MUST: check the numeric relations as well — `chunkLength ≤ maxChunkLength` and `chunkLength ≤
 * capacity ≤ maxPosition`. A conversation that fills the capacity reaches position `capacity - 1`,
 * so a default above the model's declared position limit would fail at run time **only in long
 * conversations** (the baking side's `gemma4_pipeline_config` checks the same relations, but other
 * people's distributions are read on the assumption that they skipped the check). The runtime
 * options (the sequence's `capacity`, the pipeline's `chunkLength`) are subject to the same
 * relations (`generation/sequence.ts` / `./admission.ts`).
 *
 * ## Why it is public — the entry for consumers of `fromAssets`
 *
 * `Pipeline.fromPretrained` runs this internally, so there is no need to call it. It is
 * needed by callers who pass their own bytes to `Pipeline.fromAssets`: reading `karume.json`
 * themselves, what they hold is hub's `ModelEntry.pipelineConfig` (`Record<string, unknown>`).
 * Without this entry a consumer could only cast with `as PipelineConfig` or keep the three
 * numbers both in the distribution and in their own code (either silently diverges when the
 * distribution moves).
 */
export const parsePipelineConfig = (raw: unknown): PipelineConfig => {
  const where = "pipelineConfig";
  if (!isRecord(raw)) throw new Error(`${where}: not an object (${String(raw)})`);
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positiveInteger = (value: number): boolean => Number.isSafeInteger(value) && value >= 1;
  // MUST: chunk row counts are at least 2 (the same lower bound as the runtime option's check
  // `assertChunkLength` — a one-row chunk is reserved for the decode shape). Without closing it at
  // the declaration step, a distribution mismatch would fail with the type for caller-caused input
  // (`ModelInputError`) (ADR 0107 decision 2). The writer `gemma4_pipeline_config` also bakes [2,
  // max].
  const chunkRows = (value: number): boolean => Number.isSafeInteger(value) && value >= 2;
  const chunkLength = readNumber(raw, "chunkLength", where, chunkRows, "not an integer ≥ 2");
  const maxChunkLength = readNumber(raw, "maxChunkLength", where, chunkRows, "not an integer ≥ 2");
  const maxPosition = readNumber(
    raw,
    "maxPosition",
    where,
    positiveInteger,
    "not a positive integer",
  );
  const capacity = readNumber(raw, "capacity", where, positiveInteger, "not a positive integer");
  if (chunkLength > maxChunkLength) {
    throw new Error(
      `${where}: chunkLength ${chunkLength} exceeds maxChunkLength ${maxChunkLength}` +
        ` (outside the trace range the symbol M was baked with)`,
    );
  }
  if (chunkLength > capacity) {
    throw new Error(
      `${where}: chunkLength ${chunkLength} exceeds capacity ${capacity}` +
        ` (a capacity that does not fit even one chunk)`,
    );
  }
  if (capacity > maxPosition) {
    throw new Error(
      `${where}: capacity ${capacity} exceeds maxPosition ${maxPosition}` +
        ` (a conversation filling the capacity would look up beyond the position table)`,
    );
  }
  if (!Object.hasOwn(raw, "rope")) throw new Error(`${where}.rope: missing`);
  const rope = parseRope(raw["rope"], `${where}.rope`);
  const vision = Object.hasOwn(raw, "vision")
    ? parseVision(raw["vision"], `${where}.vision`)
    : undefined;
  // An image's block is never split over chunks (ADR 0144 decision 3), so a budget whose largest
  // image needs more rows than any chunk the graph takes declares images that could never run.
  if (vision !== undefined && maxImageSoftTokens(vision.softTokenBudget) > maxChunkLength) {
    throw new Error(
      `${where}.vision.softTokenBudget ${vision.softTokenBudget}: one image can become that many ` +
        `soft tokens, more than maxChunkLength ${maxChunkLength} (an image's block is never split ` +
        `over chunks)`,
    );
  }
  return {
    chunkLength,
    maxChunkLength,
    maxPosition,
    capacity,
    rope,
    ...(Object.hasOwn(raw, "sampler")
      ? { sampler: parseSampler(raw["sampler"], `${where}.sampler`) }
      : {}),
    ...(Object.hasOwn(raw, "chatDialect")
      ? { chatDialect: parseChatDialect(raw["chatDialect"], `${where}.chatDialect`) }
      : {}),
    ...(Object.hasOwn(raw, "suppressTokens")
      ? { suppressTokens: parseSuppressTokens(raw["suppressTokens"], `${where}.suppressTokens`) }
      : {}),
    ...(vision === undefined ? {} : { vision }),
  };
};
