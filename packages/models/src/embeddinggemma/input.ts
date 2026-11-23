/**
 * What one `embed` call becomes on the host before the GPU is touched: the input → its content
 * parts → the token ids (text runs, image blocks and audio blocks), the images' and the audios'
 * places, and the width of the result; and, after the run, the leading slice of a shorter width.
 *
 * ## The input: a string, or an array of content parts
 *
 * A string is one text part. An array is read in order; its parts are `{ type: "text", text }`,
 * `{ type: "image", image }` (the shape of Gemma chat's content parts; the image is decoded RGB8 —
 * `Rgb8Image`) and `{ type: "audio", audio }` (mono f32 samples at 16 kHz — `MonoAudio`). An empty
 * array is refused, and so is a part of any type the pipeline does not accept (the message names the
 * type and the accepted ones): an image part is accepted only by a pipeline built with the vision
 * tower (`PipelineOptions.vision`), an audio part only by one built with the audio tower
 * (`PipelineOptions.audio`), and each refusal says so.
 *
 * An audio is refused unless it is 16,000 Hz (nothing is resampled — the caller resamples), at
 * most 480,000 samples (30 s — upstream's feature extractor truncates a longer one silently; here
 * nothing is truncated), long enough for one mel frame (more than 160 samples) and finite in every
 * sample.
 *
 * ## The sequence (upstream's sentence-transformers path)
 *
 * `<bos>`, then the parts in order, then `<eos>`. Adjacent text parts — and the prompt's prefix in
 * front of the first part — are joined with nothing between them into one string, tokenized on its
 * own; an image is `<|image>`, its n soft tokens (each the image placeholder id) and `<image|>`, with
 * no separator on either side, and an audio is `<|audio>`, its n soft tokens (each the audio
 * placeholder id) and `<audio|>` the same way. The ids of both are written by the host and never
 * pass through the tokenizer, and a text run is tokenized as upstream's whole string tokenizes it
 * (its added vocabulary splits around those spellings). An input that starts with an image under a prompt
 * gets the prefix as a run of its own before the image (sentence-transformers renders the prompt
 * as a leading system message, before the first part). For a text-only input the sequence is the
 * prefix and the joined text, tokenized once with `<bos>` … `<eos>` — the text path as it was.
 *
 * n comes from the image's size alone: the aspect-preserving resize rule at
 * `pipelineConfig.vision.softTokenBudget` (upstream's default, 280 — {@link imageSoftTokens}); and
 * from the audio's length alone: `ceil(L / 4)` of its `L = ceil(samples / 160) − 1` mel frames
 * (`./audio.ts` — 25 for 1 s, 750 for 30 s).
 *
 * Placeholders are never written in text: a text run whose tokens include a placeholder id is
 * refused (the text tower alone has no use for them — upstream's composite model replaces such an
 * id with the pad id before the lookup, and sentence-transformers refuses the text). The begin /
 * end markers of an image or an audio in text are tokens like any other, as upstream tokenizes
 * them.
 *
 * ## The rest of a request
 *
 * - `prompt` — a prompt name the manifest declares; its prefix is prepended to the text. None: no
 *   prefix (upstream's `default_prompt_name` is null).
 * - `dimension` — one of the declared widths; default the full width. A shorter width is the
 *   leading slice of the unit vector, re-normalised to unit length ({@link truncateEmbedding}).
 * - The token sequence with `<bos>` and `<eos>` (the tokenizer asset's post-processor) may not
 *   exceed the declared maximum; every image and every audio counts its n soft tokens and its two
 *   markers. Nothing is truncated silently.
 *
 * `tokenize` takes the same step from the input and the prompt to the token ids
 * ({@link tokenizeEmbedInput} — the soft rows of an image or an audio as its placeholder id) and
 * stops there: the limit and the placeholder refusal are `embed`'s.
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
import { AUDIO_MAX_SAMPLES, audioSoftTokens, audioValidFrames, type MonoAudio } from "./audio.ts";
import type { EmbeddingGemmaPipelineConfig, PipelineAudioConfig } from "./config.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/**
 * One part of an input given as an array: a text, a decoded RGB8 image or a mono 16 kHz audio
 * (Gemma chat's shape).
 */
export type EmbedContentPart =
  | { readonly type: "text"; readonly text: string }
  /** Accepted by a pipeline built with the vision tower (`PipelineOptions.vision`). */
  | { readonly type: "image"; readonly image: Rgb8Image }
  /** Accepted by a pipeline built with the audio tower (`PipelineOptions.audio`). */
  | { readonly type: "audio"; readonly audio: MonoAudio };

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

/** The part types a pipeline accepts: text, then image with the vision tower, audio with the audio one. */
const partTypes = (rules: EmbedTextRules): readonly string[] => [
  "text",
  ...(rules.images === undefined ? [] : ["image"]),
  ...(rules.audio === undefined ? [] : ["audio"]),
];

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
  /**
   * The audio rules (`pipelineConfig.audio`) when the pipeline holds the audio tower; absent, an
   * audio part is refused.
   */
  readonly audio?: PipelineAudioConfig;
};

/** One image of a request: where its soft rows start in the sequence and how many there are. */
export type EmbedImageBlock = {
  readonly image: Rgb8Image;
  /** The row of its first soft token (just after its `<|image>`). */
  readonly start: number;
  readonly softTokens: number;
};

/** One audio of a request: where its soft rows start in the sequence and how many there are. */
export type EmbedAudioBlock = {
  readonly audio: MonoAudio;
  /** The row of its first soft token (just after its `<|audio>`). */
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
  /** The audios, in sequence order (none for a text). */
  readonly audios: readonly EmbedAudioBlock[];
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
 * MUST (`audio`): the two audio markers likewise.
 */
export const embedTextRules = (
  tokenizer: Tokenizer,
  config: EmbeddingGemmaPipelineConfig,
  options: {
    /** The pipeline holds the vision tower: image parts are accepted (`pipelineConfig.vision`). */
    readonly images?: boolean;
    /** The pipeline holds the audio tower: audio parts are accepted (`pipelineConfig.audio`). */
    readonly audio?: boolean;
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
  const images = options.images === true ? config.vision : undefined;
  if (options.images === true) {
    if (images === undefined) {
      throw new Error(
        `${OWNER}: image input was asked for, but pipelineConfig declares no 'vision'`,
      );
    }
    for (const key of ["beginImageTokenId", "endImageTokenId"] as const) {
      addedSpelling(images[key], `${OWNER}: pipelineConfig.vision.${key} (id ${images[key]})`);
    }
  }
  const audio = options.audio === true ? config.audio : undefined;
  if (options.audio === true) {
    if (audio === undefined) {
      throw new Error(
        `${OWNER}: audio input was asked for, but pipelineConfig declares no 'audio'`,
      );
    }
    for (const key of ["beginAudioTokenId", "endAudioTokenId"] as const) {
      addedSpelling(audio[key], `${OWNER}: pipelineConfig.audio.${key} (id ${audio[key]})`);
    }
  }
  return {
    ...rules,
    ...(images === undefined ? {} : { images }),
    ...(audio === undefined ? {} : { audio }),
  };
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

/**
 * Why an audio part is refused by a pipeline without the audio tower: how to build one that takes
 * it, or that the distribution has none.
 */
const audioRefusal = (config: EmbeddingGemmaPipelineConfig): string =>
  config.audio === undefined
    ? " (this model's distribution ships no audio tower)"
    : "; audio parts need the audio tower — build the pipeline with { audio: true }" +
      " (Pipeline.fromPretrained / Pipeline.fromAssets)";

/** Whether `value` is f32 samples over an `ArrayBuffer` (what {@link MonoAudio} holds). */
const isSampleArray = (value: unknown): value is Float32Array<ArrayBuffer> =>
  value instanceof Float32Array && value.buffer instanceof ArrayBuffer;

/**
 * Reads the audio of an audio part (mono f32 samples and their rate) and refuses one the front end
 * cannot take as it is (the module doc): another rate, longer than 30 s, too short for one frame,
 * more soft tokens than the declared bound, or a non-finite sample.
 */
const audioOfPart = (
  part: Record<string, unknown>,
  where: string,
  rules: PipelineAudioConfig,
): MonoAudio => {
  const audio = part["audio"];
  const data = isRecord(audio) ? audio["data"] : undefined;
  if (!isRecord(audio) || !isSampleArray(data)) {
    throw new ModelInputError(
      `${where} is an audio part whose audio is not mono f32 samples (data: a Float32Array over an` +
        " ArrayBuffer, sampleRate)",
    );
  }
  const { sampleRate } = audio;
  if (typeof sampleRate !== "number") {
    throw new ModelInputError(
      `${where} is an audio part whose sampleRate is not a number (${shown(sampleRate)})`,
    );
  }
  if (sampleRate !== rules.sampleRate) {
    throw new ModelInputError(
      `${where} is audio at ${sampleRate} Hz — this model takes ${rules.sampleRate} Hz mono samples` +
        ` only, and nothing is resampled; resample it to ${rules.sampleRate} Hz first`,
    );
  }
  const samples = data.length;
  if (samples > AUDIO_MAX_SAMPLES) {
    throw new ModelInputError(
      `${where} is ${samples} samples (${(samples / sampleRate).toFixed(2)} s), over the` +
        ` ${AUDIO_MAX_SAMPLES} (${AUDIO_MAX_SAMPLES / sampleRate} s) this model reads — nothing is` +
        " truncated; shorten the audio or split it",
    );
  }
  const frames = audioValidFrames(samples);
  if (frames === 0) {
    throw new ModelInputError(
      `${where} is ${samples} samples, too short for one mel frame (more than 160 samples — 10 ms —` +
        " are needed)",
    );
  }
  const softTokens = audioSoftTokens(frames);
  if (softTokens > rules.maxSoftTokens) {
    throw new ModelInputError(
      `${where} is ${samples} samples, ${softTokens} soft tokens — over the ${rules.maxSoftTokens}` +
        " this model's audio tower takes; nothing is truncated — shorten the audio or split it",
    );
  }
  for (let sample = 0; sample < samples; sample += 1) {
    if (!Number.isFinite(data[sample])) {
      throw new ModelInputError(
        `${where} is an audio whose sample ${sample} is non-finite (${String(data[sample])})`,
      );
    }
  }
  return { data, sampleRate };
};

/** Reads one part of an array input. */
const contentPart = (part: unknown, index: number, rules: EmbedTextRules): EmbedContentPart => {
  const where = `${OWNER}: content part ${index}`;
  if (!isRecord(part)) throw new ModelInputError(`${where} is not an object (${shown(part)})`);
  const type = part["type"];
  const accepted = partTypes(rules);
  if (typeof type !== "string" || !accepted.includes(type)) {
    throw new ModelInputError(
      `${where} has type ${shown(type)} — this model accepts ${accepted.map(shown).join(" / ")} parts` +
        (type === "image" ? imageRefusal(rules.config) : "") +
        (type === "audio" ? audioRefusal(rules.config) : ""),
    );
  }
  if (type === "image") return { type: "image", image: imageOfPart(part, where) };
  if (type === "audio") {
    if (rules.audio === undefined) {
      throw new Error(
        `${OWNER}: internal inconsistency: an audio part was accepted without audio rules`,
      );
    }
    return { type: "audio", audio: audioOfPart(part, where, rules.audio) };
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

/** Refuses a token sequence holding a declared placeholder id outside every soft block (module doc). */
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

/** The soft blocks of an input: how many images and audios it holds. */
export type SoftBlockCounts = { readonly images: number; readonly audios: number };

/**
 * How the token count of an input with these soft blocks is made up (the wording of the refusals
 * of the model's limit here and of the device's limit in `pipeline.ts`).
 *
 * NOTE: the `export` is for the pipeline (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const tokenCountIncludes = ({ images, audios }: SoftBlockCounts): string => {
  const blocks = [
    ...(images === 0 ? [] : [images === 1 ? "the image" : `${images} images`]),
    ...(audios === 0 ? [] : [audios === 1 ? "the audio" : `${audios} audios`]),
  ];
  return blocks.length === 0
    ? "<bos> and <eos> included"
    : `<bos>, <eos> and the soft tokens of ${blocks.join(" and ")} with their two markers each` +
        " included";
};

/**
 * What shortens an input over a token limit, by what it holds (the end of both refusals).
 *
 * NOTE: the `export` is for the pipeline (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const tokenLimitRemedy = ({ images, audios }: SoftBlockCounts): string => {
  if (images === 0 && audios === 0) return "shorten the text or split it";
  if (audios === 0) return "shorten the text or pass fewer images";
  if (images === 0) return "shorten the text or pass fewer or shorter audios";
  return "shorten the text or pass fewer images or audios";
};

/** Refuses a token sequence longer than the declared maximum (nothing is truncated). */
const assertTokenLimit = (count: number, maxTokens: number, blocks: SoftBlockCounts): void => {
  if (count <= maxTokens) return;
  throw new ModelInputError(
    `${OWNER}: the input is ${count} tokens (${tokenCountIncludes(blocks)}), over this model's limit of` +
      ` ${maxTokens} — nothing is truncated; ${tokenLimitRemedy(blocks)}`,
  );
};

/** An input as token ids: the whole sequence, where its text runs lie and where its soft blocks do. */
type InputSequence = {
  readonly ids: readonly number[];
  /** `[start, end)` of every text run's ids (the placeholder refusal reads these only). */
  readonly textRanges: readonly (readonly [number, number])[];
  readonly images: readonly EmbedImageBlock[];
  readonly audios: readonly EmbedAudioBlock[];
};

/**
 * The one step from an input to its token ids, shared by {@link tokenizeEmbedInput} and
 * {@link planEmbedRequest} (module doc, "The sequence"): `<bos>`, the prefix and the parts — each
 * run of adjacent text tokenized as one string, each image and each audio as its block — then
 * `<eos>`. It refuses only an input it cannot read (the parts, an image's size, an audio's rate,
 * length and samples); the limit and the placeholders are the request's checks.
 */
const inputSequence = (
  rules: EmbedTextRules,
  prefix: string,
  input: string | readonly EmbedContentPart[],
): InputSequence => {
  const { tokenizer, images: vision, audio: audioRules } = rules;
  const ids: number[] = [rules.bos];
  const textRanges: (readonly [number, number])[] = [];
  const images: EmbedImageBlock[] = [];
  const audios: EmbedAudioBlock[] = [];
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
    if (part.type === "audio") {
      if (audioRules === undefined) {
        throw new Error(
          `${OWNER}: internal inconsistency: an audio part was read without audio rules`,
        );
      }
      flush();
      const softTokens = audioSoftTokens(audioValidFrames(part.audio.data.length));
      ids.push(audioRules.beginAudioTokenId);
      audios.push({ audio: part.audio, start: ids.length, softTokens });
      for (let row = 0; row < softTokens; row += 1) ids.push(audioRules.audioTokenId);
      ids.push(audioRules.endAudioTokenId);
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
  return { ids, textRanges, images, audios };
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
 * Turns one request into token ids, its images' and audios' places and the result width, or refuses
 * it with a `ModelInputError` (the module doc). Pure: no GPU, no state — an image is read for its
 * size only and an audio for its rate, its length and the finiteness of its samples (the pixels and
 * the frames are the pipeline's, at issue).
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
  assertTokenLimit(sequence.ids.length, config.maxTokens, {
    images: sequence.images.length,
    audios: sequence.audios.length,
  });
  return {
    ids: Int32Array.from(sequence.ids),
    dimension,
    images: sequence.images,
    audios: sequence.audios,
  };
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
