/**
 * The `pipeline` contract name of the EmbeddingGemma family and the schema check of its
 * `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and validates its schema).
 *
 * The family serves the text model of EmbeddingGemma 2 (distribution `karume-embeddinggemma-2`,
 * model `text`), and **every model value lives in the manifest** — the prompt names and prefixes,
 * the output dimensions, the token limit and the placeholder ids are declared by the dist recipe
 * (`tools/export-recipes/embeddinggemma2/distribution.py`, which reads them from the pinned
 * upstream files and checks them against the graph and the tokenizer asset) and read here,
 * never held as TS constants.
 *
 * The four required fields and the two optional ones:
 *
 * - `prompts` — prompt name → the prefix prepended to the text (upstream's
 *   `config_sentence_transformers.json`, verbatim and in its order).
 * - `dimensions` — the output widths a caller may ask for, in descending order: the full width
 *   first (the default), then the shorter leading slices the model was trained to give
 *   (Matryoshka).
 * - `maxTokens` — the longest token sequence, `<bos>` and `<eos>` included (the bound of the graph's
 *   symbolic length; the pipeline checks the two agree at construction).
 * - `placeholderTokens` — modality → the id of that modality's placeholder token. The text tower
 *   alone cannot take them (upstream's composite model replaces them with the pad id before the
 *   lookup, and sentence-transformers refuses such a text), so a text whose tokens include one is
 *   refused.
 * - `vision` (optional — a model whose distribution ships the vision tower as the weights role
 *   `vision`) — what the host needs to turn an image into the tower's rows and the text graph's
 *   soft rows: the soft-token budget of the resize rule, the ids of the soft token and of the two
 *   markers around an image's soft tokens, and the width of a soft row. The shape is the Gemma 4
 *   family's `pipelineConfig.vision`, read by its reader ({@link parseVision}); the pipeline checks
 *   it against the two graphs and the tokenizer asset when the tower is asked for.
 * - `audio` (optional — a model whose distribution ships the audio tower as the weights role
 *   `audio`) — what the host needs to turn an audio into the tower's frames and the text graph's
 *   soft rows ({@link PipelineAudioConfig}): the ids of the soft token and of the two markers around
 *   an audio's soft tokens, the width of a soft row, the most soft tokens one audio may take (the
 *   bound of the tower graph's symbol) and the sample rate of the samples. The front end's own
 *   settings (frame, hop, transform, mel filters) are the host's constants (`./audio.ts`), held to
 *   upstream's feature extractor by the dist recipe; the pipeline checks the declaration against the
 *   two graphs and the tokenizer asset when the tower is asked for.
 *
 * MUST: unknown keys fail loudly, and so does a missing required one (a misspelt field silently
 * left out would let the pipeline run with a limit or a prompt table the distributor did not
 * intend).
 * MUST: maps are read only through `Object.hasOwn` / own entries (cross-cutting invariant).
 *
 * This family has no table of public distribution repositories: nothing of it is published yet, so
 * there is no `SOURCES` (the same as `vowel-detector` — docs/release-runbook.md §3 adds it at the
 * first publication).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { assertAllowedKeys, isRecord, readNumber, readRecord } from "../config/readers.ts";
import { type PipelineVisionConfig, parseVision } from "../gemma/config.ts";
import {
  AUDIO_MAX_SAMPLES,
  AUDIO_SAMPLE_RATE,
  audioSoftTokens,
  audioValidFrames,
} from "./audio.ts";

/** The `pipeline` contract name and the major this implementation accepts (ADR 0038 §1). */
export const EMBEDDINGGEMMA_PIPELINE_NAME = "embeddinggemma";
export const EMBEDDINGGEMMA_PIPELINE_MAJOR = 1;

/** The six fields, in the order the dist recipe writes them (`vision` and `audio` are optional). */
const ROOT_KEYS: readonly string[] = [
  "prompts",
  "dimensions",
  "maxTokens",
  "placeholderTokens",
  "vision",
  "audio",
];

/** The fields of `audio`, in the order the dist recipe writes them (all required). */
const AUDIO_KEYS: readonly string[] = [
  "audioTokenId",
  "beginAudioTokenId",
  "endAudioTokenId",
  "hiddenSize",
  "maxSoftTokens",
  "sampleRate",
];

/**
 * The shortest token sequence: `<bos>` and `<eos>` alone (the empty text). The graph's symbolic
 * length starts at 2 as well (`torch.export`'s 0 / 1 specialisation is avoided), so a declared
 * maximum below it could hold no input at all.
 */
const MIN_TOKENS = 2;

/** The largest token id (ids reach the graph as an `Int32Array`). */
const MAX_TOKEN_ID = 2147483647;

/** The modality key of `placeholderTokens` whose id `vision.imageTokenId` must be. */
const IMAGE_MODALITY = "image";
/** The modality key of `placeholderTokens` whose id `audio.audioTokenId` must be. */
const AUDIO_MODALITY = "audio";

/**
 * What the host needs for audio input (`pipelineConfig.audio`).
 *
 * An audio is `beginAudioTokenId`, its n soft tokens (each `audioTokenId`) and `endAudioTokenId`,
 * no separator on either side (upstream's processor). n comes from the audio's length alone
 * (`./audio.ts`).
 */
export type PipelineAudioConfig = {
  /** The id under every soft row of an audio (= `placeholderTokens.audio`). */
  readonly audioTokenId: number;
  /** The marker before an audio's soft tokens (upstream's `boa_token_id`, `<|audio>`). */
  readonly beginAudioTokenId: number;
  /** The marker after them (upstream's `eoa_token_index`, `<audio|>`). */
  readonly endAudioTokenId: number;
  /** The width of a soft token (= the text graph's soft rows). */
  readonly hiddenSize: number;
  /**
   * The most soft tokens one audio may take: the bound of the tower graph's symbol, at most the
   * soft tokens of the front end's longest audio (750 for 30 s).
   */
  readonly maxSoftTokens: number;
  /** The rate of the samples (16,000 — the only rate the host's front end is built for). */
  readonly sampleRate: number;
};

/** The checked `pipelineConfig` of one EmbeddingGemma model. */
export type EmbeddingGemmaPipelineConfig = {
  /** Prompt name → prefix, in the declared order. */
  readonly prompts: ReadonlyMap<string, string>;
  /** The output widths, strictly descending; `dimensions[0]` is the full width and the default. */
  readonly dimensions: readonly number[];
  /** The longest token sequence, `<bos>` and `<eos>` included. */
  readonly maxTokens: number;
  /** Modality → the id of its placeholder token, in the declared order. */
  readonly placeholderTokens: ReadonlyMap<string, number>;
  /** What the host needs for image input (absent: the distribution ships no vision tower). */
  readonly vision?: PipelineVisionConfig;
  /** What the host needs for audio input (absent: the distribution ships no audio tower). */
  readonly audio?: PipelineAudioConfig;
};

const isSafeCount = (value: number): boolean => Number.isSafeInteger(value) && value > 0;

/** Reads the prompt table (any names; every prefix a string). */
const readPrompts = (raw: Record<string, unknown>, where: string): ReadonlyMap<string, string> => {
  const at = `${where}.prompts`;
  if (!Object.hasOwn(raw, "prompts")) throw new Error(`${at}: missing`);
  const table = readRecord(raw["prompts"], at);
  const prompts = new Map<string, string>();
  for (const [name, prefix] of Object.entries(table)) {
    if (name.length === 0) throw new Error(`${at}: a prompt has an empty name`);
    if (typeof prefix !== "string") {
      throw new Error(`${at}.${name}: the prefix is not a string (${JSON.stringify(prefix)})`);
    }
    prompts.set(name, prefix);
  }
  return prompts;
};

/** Reads the dimensions (a non-empty array of positive integers, strictly descending). */
const readDimensions = (raw: Record<string, unknown>, where: string): readonly number[] => {
  const at = `${where}.dimensions`;
  if (!Object.hasOwn(raw, "dimensions")) throw new Error(`${at}: missing`);
  const value = raw["dimensions"];
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${at}: not a non-empty array (${JSON.stringify(value)})`);
  }
  const dimensions: number[] = [];
  for (const entry of value) {
    if (typeof entry !== "number" || !isSafeCount(entry)) {
      throw new Error(
        `${at}: has an element that is not a positive integer (${JSON.stringify(value)})`,
      );
    }
    // Strictly descending: the first is the full width (the default), and no width repeats.
    const previous = dimensions.at(-1);
    if (previous !== undefined && entry >= previous) {
      throw new Error(`${at}: not strictly descending (${JSON.stringify(value)})`);
    }
    dimensions.push(entry);
  }
  return dimensions;
};

/** Reads the placeholder ids (modality → token id; no id repeats). */
const readPlaceholderTokens = (
  raw: Record<string, unknown>,
  where: string,
): ReadonlyMap<string, number> => {
  const at = `${where}.placeholderTokens`;
  if (!Object.hasOwn(raw, "placeholderTokens")) throw new Error(`${at}: missing`);
  const table = readRecord(raw["placeholderTokens"], at);
  const tokens = new Map<string, number>();
  const seen = new Set<number>();
  for (const [modality, id] of Object.entries(table)) {
    if (modality.length === 0) throw new Error(`${at}: a modality has an empty name`);
    if (typeof id !== "number" || !Number.isInteger(id) || id < 0 || id > MAX_TOKEN_ID) {
      throw new Error(
        `${at}.${modality}: not a token id (an integer in 0..${MAX_TOKEN_ID}) (${JSON.stringify(id)})`,
      );
    }
    if (seen.has(id)) throw new Error(`${at}.${modality}: id ${id} is declared twice`);
    seen.add(id);
    tokens.set(modality, id);
  }
  return tokens;
};

/**
 * Reads `audio` (called only when the key is present). The vocabulary bound of the ids is checked
 * at admission, where the vocabulary is first known (`admitAudio` in `./pipeline.ts`).
 */
const parseAudio = (raw: unknown, where: string): PipelineAudioConfig => {
  if (!isRecord(raw)) throw new Error(`${where}: not an object (${String(raw)})`);
  assertAllowedKeys(raw, AUDIO_KEYS, where);
  // The soft tokens of the front end's longest audio (750 — `maxSoftTokens` may not exceed it).
  // Computed here, not at module scope: a call there would survive tree-shaking.
  const softTokensMax = audioSoftTokens(audioValidFrames(AUDIO_MAX_SAMPLES));
  const tokenId = (key: string): number =>
    readNumber(
      raw,
      key,
      where,
      (value) => Number.isSafeInteger(value) && value >= 0 && value <= MAX_TOKEN_ID,
      `not a token id (an integer in 0..${MAX_TOKEN_ID})`,
    );
  const audio: PipelineAudioConfig = {
    audioTokenId: tokenId("audioTokenId"),
    beginAudioTokenId: tokenId("beginAudioTokenId"),
    endAudioTokenId: tokenId("endAudioTokenId"),
    hiddenSize: readNumber(raw, "hiddenSize", where, isSafeCount, "not a positive integer"),
    maxSoftTokens: readNumber(
      raw,
      "maxSoftTokens",
      where,
      (value) => isSafeCount(value) && value <= softTokensMax,
      `not an integer in 1..${softTokensMax} (the soft tokens of ${AUDIO_MAX_SAMPLES} samples, the longest audio the front end reads)`,
    ),
    // MUST: the host's front end (its hop, frame and mel range) is built for one rate — another
    // declared rate would embed every audio through frames of other lengths.
    sampleRate: readNumber(
      raw,
      "sampleRate",
      where,
      (value) => value === AUDIO_SAMPLE_RATE,
      `not ${AUDIO_SAMPLE_RATE} (the only rate the host's audio front end is built for)`,
    ),
  };
  // A soft token that is also its own marker would render a sequence the model never saw.
  const ids = [audio.audioTokenId, audio.beginAudioTokenId, audio.endAudioTokenId];
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${where}: the three audio token ids are not distinct (${ids.join(", ")})`);
  }
  return audio;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseEmbeddingGemmaPipelineConfig = (raw: unknown): EmbeddingGemmaPipelineConfig => {
  const where = "pipelineConfig";
  if (!isRecord(raw)) throw new Error(`${where}: missing / not an object`);
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const config: EmbeddingGemmaPipelineConfig = {
    prompts: readPrompts(raw, where),
    dimensions: readDimensions(raw, where),
    maxTokens: readNumber(
      raw,
      "maxTokens",
      where,
      (value) => Number.isSafeInteger(value) && value >= MIN_TOKENS,
      `not an integer of at least ${MIN_TOKENS} (<bos> and <eos>)`,
    ),
    placeholderTokens: readPlaceholderTokens(raw, where),
  };
  return {
    ...config,
    ...(Object.hasOwn(raw, "vision") ? { vision: readVision(raw["vision"], config, where) } : {}),
    ...(Object.hasOwn(raw, "audio") ? { audio: readAudio(raw["audio"], config, where) } : {}),
  };
};

/** Reads `vision` and ties its soft token to the image placeholder. */
const readVision = (
  raw: unknown,
  config: EmbeddingGemmaPipelineConfig,
  where: string,
): PipelineVisionConfig => {
  const vision = parseVision(raw, `${where}.vision`);
  // MUST: the soft token is the image placeholder — the text path refuses that id in text, which
  // is what keeps a soft row from being spelt without an image behind it.
  const placeholder = config.placeholderTokens.get(IMAGE_MODALITY);
  if (placeholder !== vision.imageTokenId) {
    throw new Error(
      `${where}.vision.imageTokenId ${vision.imageTokenId} is not placeholderTokens.${IMAGE_MODALITY}` +
        ` (${placeholder === undefined ? "not declared" : placeholder}) — an image's soft tokens are` +
        " its placeholder, which a text may not hold",
    );
  }
  return vision;
};

/** Reads `audio` and ties its soft token to the audio placeholder. */
const readAudio = (
  raw: unknown,
  config: EmbeddingGemmaPipelineConfig,
  where: string,
): PipelineAudioConfig => {
  const audio = parseAudio(raw, `${where}.audio`);
  // MUST: the soft token is the audio placeholder, for the image's reason.
  const placeholder = config.placeholderTokens.get(AUDIO_MODALITY);
  if (placeholder !== audio.audioTokenId) {
    throw new Error(
      `${where}.audio.audioTokenId ${audio.audioTokenId} is not placeholderTokens.${AUDIO_MODALITY}` +
        ` (${placeholder === undefined ? "not declared" : placeholder}) — an audio's soft tokens are` +
        " its placeholder, which a text may not hold",
    );
  }
  return audio;
};
