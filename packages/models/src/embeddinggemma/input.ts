/**
 * What one `embed` call becomes on the host before the GPU is touched: the input → its content
 * parts → one text → the token ids, and the width of the result; and, after the run, the leading
 * slice of a shorter width.
 *
 * ## The input: a string, or an array of content parts
 *
 * A string is one text part. An array is read in order; today the only part type is
 * `{ type: "text", text }` (the shape of Gemma chat's content parts), and the text parts are
 * concatenated with nothing between them and tokenized as one string — what upstream does with one
 * string. An empty array is refused, and so is a part of any type the loaded model does not accept
 * (the message names the type and the accepted ones).
 *
 * The parts are where a later model widens the input: an image, audio or video part (a video as an
 * array of frames — the maintainer's ruling of 2026-11-21) would join the union and the part
 * reader, and a text-only input would keep this path as it is — the prefix, then the joined text.
 * Nothing of another modality is built. Placeholders are never written in text: a text whose
 * tokens include a placeholder id is refused (the text tower alone has no use for them —
 * upstream's composite model replaces such an id with the pad id before the lookup, and
 * sentence-transformers refuses the text).
 *
 * ## The rest of a request
 *
 * - `prompt` — a prompt name the manifest declares; its prefix is prepended to the text. None: no
 *   prefix (upstream's `default_prompt_name` is null).
 * - `dimension` — one of the declared widths; default the full width. A shorter width is the
 *   leading slice of the unit vector, re-normalised to unit length ({@link truncateEmbedding}).
 * - The token sequence with `<bos>` and `<eos>` (the tokenizer asset's post-processor) may not
 *   exceed the declared maximum. Nothing is truncated silently.
 *
 * Every refusal of the request is a `ModelInputError` (fixing the input makes it pass — ADR 0107);
 * a tokenizer asset that does not fit the manifest is a plain `Error` raised at construction
 * ({@link embedTextRules}).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { isRecord } from "../config/readers.ts";
import { ModelInputError } from "../errors.ts";
import type { Tokenizer } from "../gemma/text/tokenizer.ts";
import type { EmbeddingGemmaPipelineConfig } from "./config.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/** One part of an input given as an array (today text only). */
export type EmbedContentPart = { readonly type: "text"; readonly text: string };

/** Options for one call of `embed`. */
export type EmbedOptions = {
  /**
   * A prompt name the model declares (`"query"`, `"document"`, …); its prefix is prepended to the
   * text. Omitted: no prefix. An unknown name is refused with the declared names.
   */
  readonly prompt?: string;
  /**
   * The width of the result: one of the widths the model declares (768, 512, 256, 128 for
   * EmbeddingGemma 2). Omitted: the full width. A shorter one is the leading slice re-normalised to
   * unit length.
   */
  readonly dimension?: number;
  /**
   * Cancels the call. Checked after waiting for the turn on the pipeline's serialization chain and
   * before the run, after yielding to the event loop once (a run is indivisible). On cancellation it
   * throws `signal.reason` as is; the pipeline can run the next `embed` afterwards.
   */
  readonly signal?: AbortSignal;
};

/** The part types the text path takes. */
const TEXT_PART_TYPES: readonly string[] = ["text"];

/**
 * The divisor floor of the re-normalisation (upstream's `torch.nn.functional.normalize` default,
 * which sentence-transformers uses after truncating).
 */
const NORM_EPSILON = 1e-12;

/** What turns a request into token ids (checked once, at construction — {@link embedTextRules}). */
export type EmbedTextRules = {
  readonly tokenizer: Tokenizer;
  readonly config: EmbeddingGemmaPipelineConfig;
  /** Placeholder id → how a refusal names it (`the image placeholder token "<|image|>" (id 258880)`). */
  readonly placeholders: ReadonlyMap<number, string>;
};

/** One request, ready for the run. */
export type EmbedRequest = {
  /** The token ids, `<bos>` … `<eos>` (lent to the run — not modified until it settles). */
  readonly ids: Int32Array<ArrayBuffer>;
  /** The width of the result. */
  readonly dimension: number;
};

/** A value as a refusal message shows it. */
const shown = (value: unknown): string =>
  typeof value === "string" ? JSON.stringify(value) : String(value);

/**
 * Checks the tokenizer asset against the manifest and builds the rules of the text path.
 *
 * MUST: the asset wraps the sequence in `<bos>` … `<eos>` — the graph was exported and checked
 * against upstream with that sequence, so an asset that adds nothing would embed every text
 * silently differently.
 * MUST: every declared placeholder id is an added token of the asset — the refusal looks for the id
 * in the tokenized text, and an id the asset cannot produce from its spelling would make the check
 * dead while upstream's composite model still treats the spelling as a placeholder.
 */
export const embedTextRules = (
  tokenizer: Tokenizer,
  config: EmbeddingGemmaPipelineConfig,
): EmbedTextRules => {
  if (tokenizer.postProcessor !== "bos-eos") {
    throw new Error(
      `${OWNER}: the tokenizer asset's post-processor is '${tokenizer.postProcessor}' — the model` +
        " embeds <bos> … <eos> sequences ('bos-eos')",
    );
  }
  const placeholders = new Map<number, string>();
  for (const [modality, id] of config.placeholderTokens) {
    const where = `${OWNER}: pipelineConfig.placeholderTokens.${modality} (id ${id})`;
    let spelling: string;
    try {
      spelling = tokenizer.decode([id], { skipSpecialTokens: false });
    } catch (cause) {
      throw new Error(`${where} is not a token of the tokenizer asset`, { cause });
    }
    if (tokenizer.addedTokenId(spelling) !== id) {
      throw new Error(
        `${where} is not an added token of the tokenizer asset (its spelling ${JSON.stringify(spelling)}` +
          " does not tokenize to it)",
      );
    }
    placeholders.set(
      id,
      `the ${modality} placeholder token ${JSON.stringify(spelling)} (id ${id})`,
    );
  }
  return { tokenizer, config, placeholders };
};

/** The prefix of the requested prompt ("" without one). */
const promptPrefix = (config: EmbeddingGemmaPipelineConfig, prompt: unknown): string => {
  if (prompt === undefined) return "";
  const prefix = typeof prompt === "string" ? config.prompts.get(prompt) : undefined;
  if (prefix === undefined) {
    throw new ModelInputError(
      `${OWNER}: prompt ${shown(prompt)} is not one of this model's prompts` +
        ` (${[...config.prompts.keys()].join(" / ")})`,
    );
  }
  return prefix;
};

/** The requested width (the full width without one). */
const outputDimension = (config: EmbeddingGemmaPipelineConfig, dimension: unknown): number => {
  if (dimension === undefined) return config.dimensions[0];
  if (typeof dimension !== "number" || !config.dimensions.includes(dimension)) {
    throw new ModelInputError(
      `${OWNER}: dimension ${shown(dimension)} is not one of this model's output dimensions` +
        ` (${config.dimensions.join(" / ")})`,
    );
  }
  return dimension;
};

/** Reads one part of an array input. */
const contentPart = (
  part: unknown,
  index: number,
  accepted: readonly string[],
): EmbedContentPart => {
  const where = `${OWNER}: content part ${index}`;
  if (!isRecord(part)) throw new ModelInputError(`${where} is not an object (${shown(part)})`);
  const type = part["type"];
  if (typeof type !== "string" || !accepted.includes(type)) {
    throw new ModelInputError(
      `${where} has type ${shown(type)} — this model accepts ${accepted.map(shown).join(" / ")} parts`,
    );
  }
  const text = part["text"];
  if (typeof text !== "string") {
    throw new ModelInputError(
      `${where} is a text part whose text is not a string (${shown(text)})`,
    );
  }
  return { type: "text", text };
};

/** The content parts of an input (a string is one text part). */
const contentParts = (input: unknown, accepted: readonly string[]): readonly EmbedContentPart[] => {
  if (typeof input === "string") return [{ type: "text", text: input }];
  if (!Array.isArray(input)) {
    throw new ModelInputError(
      `${OWNER}: the input is neither a string nor an array of content parts (${shown(input)})`,
    );
  }
  const parts: readonly unknown[] = input;
  if (parts.length === 0) {
    throw new ModelInputError(
      `${OWNER}: the input has no content part (pass a string, or an array of at least one part)`,
    );
  }
  return parts.map((part, index) => contentPart(part, index, accepted));
};

/** Refuses a token sequence holding a declared placeholder id (module doc). */
const assertNoPlaceholder = (
  placeholders: ReadonlyMap<number, string>,
  ids: readonly number[],
): void => {
  for (const [position, id] of ids.entries()) {
    const label = placeholders.get(id);
    if (label === undefined) continue;
    throw new ModelInputError(
      `${OWNER}: the text holds ${label} at token ${position} — placeholder tokens are not taken from` +
        " text (the text tower embeds text only, and sentence-transformers refuses such a text too);" +
        " remove that spelling",
    );
  }
};

/** Refuses a token sequence longer than the declared maximum (nothing is truncated). */
const assertTokenLimit = (count: number, maxTokens: number): void => {
  if (count <= maxTokens) return;
  throw new ModelInputError(
    `${OWNER}: the input is ${count} tokens (<bos> and <eos> included), over this model's limit of` +
      ` ${maxTokens} — nothing is truncated; shorten the text or split it`,
  );
};

/**
 * Turns one request into token ids and the result width, or refuses it with a `ModelInputError`
 * (the module doc). Pure: no GPU, no state.
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath
 * surface — ADR 0008).
 */
export const planEmbedRequest = (
  rules: EmbedTextRules,
  input: string | readonly EmbedContentPart[],
  options: EmbedOptions,
): EmbedRequest => {
  const { config, tokenizer } = rules;
  const prefix = promptPrefix(config, options.prompt);
  const dimension = outputDimension(config, options.dimension);
  const text = contentParts(input, TEXT_PART_TYPES)
    .map((part) => part.text)
    .join("");
  const ids = tokenizer.encodeWithSpecialTokens(prefix + text);
  assertNoPlaceholder(rules.placeholders, ids);
  assertTokenLimit(ids.length, config.maxTokens);
  return { ids: Int32Array.from(ids), dimension };
};

/**
 * The result of width `dimension` from the full unit vector: the vector itself at the full width,
 * otherwise its leading `dimension` values divided by `max(norm, 1e-12)` (computed in f64, stored
 * as f32) — sentence-transformers' `truncate_dim` followed by its normalisation.
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath
 * surface — ADR 0008).
 */
export const truncateEmbedding = (
  vector: Float32Array<ArrayBuffer>,
  dimension: number,
): Float32Array<ArrayBuffer> => {
  if (dimension === vector.length) return vector;
  if (!Number.isInteger(dimension) || dimension <= 0 || dimension > vector.length) {
    throw new Error(
      `${OWNER}: internal inconsistency: width ${dimension} is not a leading slice of ${vector.length}`,
    );
  }
  let squares = 0;
  for (let index = 0; index < dimension; index += 1) squares += vector[index] * vector[index];
  const norm = Math.max(Math.sqrt(squares), NORM_EPSILON);
  const truncated = new Float32Array(dimension);
  for (let index = 0; index < dimension; index += 1) truncated[index] = vector[index] / norm;
  return truncated;
};
