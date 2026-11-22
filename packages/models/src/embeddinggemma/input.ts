/**
 * What one `embed` call becomes on the host before the GPU is touched: the input → its content
 * parts → the token ids (text runs and image blocks), the images' places, and the width of the
 * result; and, after the run, the leading slice of a shorter width.
 *
 * ## The input: a string, or an array of content parts
 *
 * A string is one text part. An array is read in order; its parts are `{ type: "text", text }` and
 * `{ type: "image", image }` (the shape of Gemma chat's content parts; the image is decoded RGB8 —
 * `Rgb8Image`). An empty array is refused, and so is a part of any type the pipeline does not
 * accept (the message names the type and the accepted ones): an image part is accepted only by a
 * pipeline built with the vision tower (`PipelineOptions.vision`), and its refusal says so.
 *
 * ## The sequence (upstream's sentence-transformers path)
 *
 * `<bos>`, then the parts in order, then `<eos>`. Adjacent text parts — and the prompt's prefix in
 * front of the first part — are joined with nothing between them into one string, tokenized on its
 * own; an image is `<|image>`, its n soft tokens (each the image placeholder id) and `<image|>`, with
 * no separator on either side. The image ids are written by the host and never pass through the
 * tokenizer, and a text run is tokenized as upstream's whole string tokenizes it (its added
 * vocabulary splits around those spellings). An input that starts with an image under a prompt
 * gets the prefix as a run of its own before the image (sentence-transformers renders the prompt
 * as a leading system message, before the first part). For a text-only input the sequence is the
 * prefix and the joined text, tokenized once with `<bos>` … `<eos>` — the text path as it was.
 *
 * n comes from the image's size alone: the aspect-preserving resize rule at
 * `pipelineConfig.vision.softTokenBudget` (upstream's default, 280 — {@link imageSoftTokens}).
 *
 * Placeholders are never written in text: a text run whose tokens include a placeholder id is
 * refused (the text tower alone has no use for them — upstream's composite model replaces such an
 * id with the pad id before the lookup, and sentence-transformers refuses the text). The begin /
 * end markers of an image in text are tokens like any other, as upstream tokenizes them.
 *
 * ## The rest of a request
 *
 * - `prompt` — a prompt name the manifest declares; its prefix is prepended to the text. None: no
 *   prefix (upstream's `default_prompt_name` is null).
 * - `dimension` — one of the declared widths; default the full width. A shorter width is the
 *   leading slice of the unit vector, re-normalised to unit length ({@link truncateEmbedding}).
 * - The token sequence with `<bos>` and `<eos>` (the tokenizer asset's post-processor) may not
 *   exceed the declared maximum; every image counts its n soft tokens and its two markers. Nothing
 *   is truncated silently.
 *
 * `tokenize` takes the same step from the input and the prompt to the token ids
 * ({@link tokenizeEmbedInput} — an image's soft rows as the placeholder id) and stops there: the
 * limit and the placeholder refusal are `embed`'s.
 *
 * Every refusal of the request is a `ModelInputError` (fixing the input makes it pass — ADR 0107);
 * a tokenizer asset that does not fit the manifest is a plain `Error` raised at construction
 * ({@link embedTextRules}).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { isRecord } from "../config/readers.ts";
import { ModelInputError } from "../errors.ts";
import type { PipelineVisionConfig } from "../gemma/config.ts";
import type { Tokenizer } from "../gemma/text/tokenizer.ts";
import { imageSoftTokens } from "../gemma/vision.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import type { EmbeddingGemmaPipelineConfig } from "./config.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/** One part of an input given as an array: a text, or a decoded RGB8 image (Gemma chat's shape). */
export type EmbedContentPart =
  | { readonly type: "text"; readonly text: string }
  /** Accepted by a pipeline built with the vision tower (`PipelineOptions.vision`). */
  | { readonly type: "image"; readonly image: Rgb8Image };

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

/** Options for one call of `tokenize`: the prompt, as `embed` takes it. */
export type TokenizeOptions = Pick<EmbedOptions, "prompt">;

/** The part types of a pipeline without the vision tower, and of one with it. */
const TEXT_PART_TYPES: readonly string[] = ["text"];
const IMAGE_PART_TYPES: readonly string[] = ["text", "image"];

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
  /** The ids around a sequence (the asset's `<bos>` … `<eos>`). */
  readonly bos: number;
  readonly eos: number;
  /**
   * The image rules (`pipelineConfig.vision`) when the pipeline holds the vision tower; absent, an
   * image part is refused.
   */
  readonly images?: PipelineVisionConfig;
};

/** One image of a request: where its soft rows start in the sequence and how many there are. */
export type EmbedImageBlock = {
  readonly image: Rgb8Image;
  /** The row of its first soft token (just after its `<|image>`). */
  readonly start: number;
  readonly softTokens: number;
};

/** One request, ready for the run. */
export type EmbedRequest = {
  /** The token ids, `<bos>` … `<eos>` (lent to the run — not modified until it settles). */
  readonly ids: Int32Array<ArrayBuffer>;
  /** The width of the result. */
  readonly dimension: number;
  /** The images, in sequence order (none for a text). */
  readonly images: readonly EmbedImageBlock[];
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
 * MUST (`images`): the two image markers are added tokens of the asset too — they are upstream's
 * spellings of the ids the host writes around an image, so an id that is another token would be
 * a sequence upstream never sees.
 */
export const embedTextRules = (
  tokenizer: Tokenizer,
  config: EmbeddingGemmaPipelineConfig,
  options: {
    /** The pipeline holds the vision tower: image parts are accepted (`pipelineConfig.vision`). */
    readonly images?: boolean;
  } = {},
): EmbedTextRules => {
  if (tokenizer.postProcessor !== "bos-eos") {
    throw new Error(
      `${OWNER}: the tokenizer asset's post-processor is '${tokenizer.postProcessor}' — the model` +
        " embeds <bos> … <eos> sequences ('bos-eos')",
    );
  }
  // The empty text is `<bos>` `<eos>` alone under that post-processor.
  const [bos, eos] = tokenizer.encodeWithSpecialTokens("");
  const addedSpelling = (id: number, where: string): string => {
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
    return spelling;
  };
  const placeholders = new Map<number, string>();
  for (const [modality, id] of config.placeholderTokens) {
    const spelling = addedSpelling(
      id,
      `${OWNER}: pipelineConfig.placeholderTokens.${modality} (id ${id})`,
    );
    placeholders.set(
      id,
      `the ${modality} placeholder token ${JSON.stringify(spelling)} (id ${id})`,
    );
  }
  const rules = { tokenizer, config, placeholders, bos, eos };
  if (options.images !== true) return rules;
  const { vision } = config;
  if (vision === undefined) {
    throw new Error(`${OWNER}: image input was asked for, but pipelineConfig declares no 'vision'`);
  }
  for (const key of ["beginImageTokenId", "endImageTokenId"] as const) {
    addedSpelling(vision[key], `${OWNER}: pipelineConfig.vision.${key} (id ${vision[key]})`);
  }
  return { ...rules, images: vision };
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

/**
 * Why an image part is refused by a pipeline without the vision tower: how to build one that takes
 * it, or that the distribution has none.
 */
const imageRefusal = (config: EmbeddingGemmaPipelineConfig): string =>
  config.vision === undefined
    ? " (this model's distribution ships no vision tower)"
    : "; image parts need the vision tower — build the pipeline with { vision: true }" +
      " (Pipeline.fromPretrained / Pipeline.fromAssets)";

/** Reads the image of an image part (decoded RGB8 pixels and their size). */
const imageOfPart = (part: Record<string, unknown>, where: string): Rgb8Image => {
  const image = part["image"];
  if (!isRecord(image) || !(image["data"] instanceof Uint8Array)) {
    throw new ModelInputError(
      `${where} is an image part whose image is not RGB8 pixels (a decoded Rgb8Image: data, width,` +
        " height)",
    );
  }
  const { data, width, height } = image;
  if (typeof width !== "number" || typeof height !== "number") {
    throw new ModelInputError(
      `${where} is an image part whose width / height are not numbers (${shown(width)} × ${shown(height)})`,
    );
  }
  const pixels = { data, width, height };
  assertRgb8(pixels);
  return pixels;
};

/** Reads one part of an array input. */
const contentPart = (part: unknown, index: number, rules: EmbedTextRules): EmbedContentPart => {
  const where = `${OWNER}: content part ${index}`;
  if (!isRecord(part)) throw new ModelInputError(`${where} is not an object (${shown(part)})`);
  const type = part["type"];
  const accepted = rules.images === undefined ? TEXT_PART_TYPES : IMAGE_PART_TYPES;
  if (typeof type !== "string" || !accepted.includes(type)) {
    throw new ModelInputError(
      `${where} has type ${shown(type)} — this model accepts ${accepted.map(shown).join(" / ")} parts` +
        (type === "image" ? imageRefusal(rules.config) : ""),
    );
  }
  if (type === "image") return { type: "image", image: imageOfPart(part, where) };
  const text = part["text"];
  if (typeof text !== "string") {
    throw new ModelInputError(
      `${where} is a text part whose text is not a string (${shown(text)})`,
    );
  }
  return { type: "text", text };
};

/** The content parts of an input (a string is one text part). */
const contentParts = (input: unknown, rules: EmbedTextRules): readonly EmbedContentPart[] => {
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
  return parts.map((part, index) => contentPart(part, index, rules));
};

/** Refuses a token sequence holding a declared placeholder id outside every image (module doc). */
const assertNoPlaceholder = (
  placeholders: ReadonlyMap<number, string>,
  sequence: InputSequence,
): void => {
  for (const [start, end] of sequence.textRanges) {
    for (let position = start; position < end; position += 1) {
      const label = placeholders.get(sequence.ids[position]);
      if (label === undefined) continue;
      throw new ModelInputError(
        `${OWNER}: the text holds ${label} at token ${position} — placeholder tokens are not taken` +
          " from text (the text tower embeds text only, and sentence-transformers refuses such a text" +
          " too); remove that spelling",
      );
    }
  }
};

/**
 * How the token count of an input with `images` images is made up (the wording of the refusals of
 * the model's limit here and of the device's limit in `pipeline.ts`).
 *
 * NOTE: the `export` is for the pipeline (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const tokenCountIncludes = (images: number): string =>
  images === 0
    ? "<bos> and <eos> included"
    : `<bos>, <eos> and the soft tokens of ${images === 1 ? "the image" : `${images} images`} with` +
      " their two markers each included";

/** Refuses a token sequence longer than the declared maximum (nothing is truncated). */
const assertTokenLimit = (count: number, maxTokens: number, images: number): void => {
  if (count <= maxTokens) return;
  throw new ModelInputError(
    `${OWNER}: the input is ${count} tokens (${tokenCountIncludes(images)}), over this model's limit of` +
      ` ${maxTokens} — nothing is truncated; ${images === 0 ? "shorten the text or split it" : "shorten the text or pass fewer images"}`,
  );
};

/** An input as token ids: the whole sequence, where its text runs lie and where its images do. */
type InputSequence = {
  readonly ids: readonly number[];
  /** `[start, end)` of every text run's ids (the placeholder refusal reads these only). */
  readonly textRanges: readonly (readonly [number, number])[];
  readonly images: readonly EmbedImageBlock[];
};

/**
 * The one step from an input to its token ids, shared by {@link tokenizeEmbedInput} and
 * {@link planEmbedRequest} (module doc, "The sequence"): `<bos>`, the prefix and the parts — each
 * run of adjacent text tokenized as one string, each image as its block — then `<eos>`. It refuses
 * only an input it cannot read (the parts, an image's size); the limit and the placeholders are the
 * request's checks.
 */
const inputSequence = (
  rules: EmbedTextRules,
  prefix: string,
  input: string | readonly EmbedContentPart[],
): InputSequence => {
  const { tokenizer, images: vision } = rules;
  const ids: number[] = [rules.bos];
  const textRanges: (readonly [number, number])[] = [];
  const images: EmbedImageBlock[] = [];
  let run = prefix;
  const flush = (): void => {
    const start = ids.length;
    // One element at a time (a long text would hit the argument limit of a spread push).
    for (const id of tokenizer.encode(run)) ids.push(id);
    textRanges.push([start, ids.length]);
    run = "";
  };
  for (const part of contentParts(input, rules)) {
    if (part.type === "text") {
      run += part.text;
      continue;
    }
    if (vision === undefined) {
      throw new Error(
        `${OWNER}: internal inconsistency: an image part was read without image rules`,
      );
    }
    flush();
    const softTokens = imageSoftTokens(part.image, vision.softTokenBudget);
    ids.push(vision.beginImageTokenId);
    images.push({ image: part.image, start: ids.length, softTokens });
    for (let row = 0; row < softTokens; row += 1) ids.push(vision.imageTokenId);
    ids.push(vision.endImageTokenId);
  }
  flush();
  ids.push(rules.eos);
  return { ids, textRanges, images };
};

/**
 * The token ids `embed` would run for this input and prompt, without the request's checks: a
 * sequence over the limit or one holding a placeholder id is returned, not refused (a caller sees
 * the count and the ids before `embed` refuses them). An input that cannot be read and an unknown
 * prompt are refused with the same `ModelInputError` as {@link planEmbedRequest}. Pure: no GPU, no
 * state.
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath
 * surface — ADR 0008).
 */
export const tokenizeEmbedInput = (
  rules: EmbedTextRules,
  input: string | readonly EmbedContentPart[],
  options: TokenizeOptions,
): Int32Array<ArrayBuffer> =>
  Int32Array.from(inputSequence(rules, promptPrefix(rules.config, options.prompt), input).ids);

/**
 * Turns one request into token ids, its images' places and the result width, or refuses it with a
 * `ModelInputError` (the module doc). Pure: no GPU, no state — an image is read for its size only
 * (its pixels are the pipeline's, at issue).
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath
 * surface — ADR 0008).
 */
export const planEmbedRequest = (
  rules: EmbedTextRules,
  input: string | readonly EmbedContentPart[],
  options: EmbedOptions,
): EmbedRequest => {
  const { config } = rules;
  const prefix = promptPrefix(config, options.prompt);
  const dimension = outputDimension(config, options.dimension);
  const sequence = inputSequence(rules, prefix, input);
  assertNoPlaceholder(rules.placeholders, sequence);
  assertTokenLimit(sequence.ids.length, config.maxTokens, sequence.images.length);
  return { ids: Int32Array.from(sequence.ids), dimension, images: sequence.images };
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
