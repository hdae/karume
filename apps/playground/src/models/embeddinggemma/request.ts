/**
 * The page's form logic: the list of rows (each a text, its own prompt and, on a model that ships
 * the vision tower, an optional image, on one that ships the audio tower, an optional audio), what a
 * row sends to the pipeline, its token count or the pipeline's refusal of it, which model and preset
 * are in effect, whether the held pipeline serves a run, and why Run is disabled. Pure, so `vp test`
 * covers it.
 */

import type { Manifest } from "@karume/hub";
import { type EmbedContentPart, ModelInputError } from "@karume/models/embeddinggemma";

import type { LoadedImage } from "@/lib/image";

import type { LoadedAudio } from "./audio";

/**
 * One input of a run: a text, the prompt whose prefix goes before it (`undefined` = none), an image
 * embedded after the text and an audio after that (`undefined` = none).
 */
export type TextRow = {
  /** Stable across edits and removals (the React key); never reused within one page. */
  readonly id: number;
  readonly text: string;
  readonly prompt?: string;
  readonly image?: LoadedImage;
  readonly audio?: LoadedAudio;
};

/**
 * The rows the page opens with: a query, a document that answers it and one that does not, so the
 * first run already shows the similarity the prompts are for. The prompt names are EmbeddingGemma's
 * upstream names; a model that lacks one shows it as not declared and the pipeline refuses that row.
 */
export const SAMPLE_ROWS: readonly TextRow[] = [
  { id: 0, text: "how do I reset my password", prompt: "query" },
  {
    id: 1,
    text: "To reset your password, open Settings, choose Security and follow the link sent to your email address.",
    prompt: "document",
  },
  {
    id: 2,
    text: "The museum is open from 10 am to 6 pm every day except Mondays.",
    prompt: "document",
  },
];

/** The id the next added row gets (one past the largest in use, so ids stay unique). */
const nextId = (rows: readonly TextRow[]): number =>
  rows.reduce((largest, { id }) => Math.max(largest, id), -1) + 1;

/** `rows` with an empty row (no prompt) at the end. */
export const addRow = (rows: readonly TextRow[]): readonly TextRow[] => [
  ...rows,
  { id: nextId(rows), text: "" },
];

/** `rows` without row `id`; the last row is kept (a run needs at least one). */
export const removeRow = (rows: readonly TextRow[], id: number): readonly TextRow[] =>
  rows.length <= 1 ? rows : rows.filter((row) => row.id !== id);

/**
 * `rows` with row `id` changed (`prompt: undefined` clears the prompt, `image: undefined` the
 * image, `audio: undefined` the audio).
 */
export const updateRow = (
  rows: readonly TextRow[],
  id: number,
  change:
    | { readonly text: string }
    | { readonly prompt: string | undefined }
    | { readonly image: LoadedImage | undefined }
    | { readonly audio: LoadedAudio | undefined },
): readonly TextRow[] =>
  rows.map((row) => {
    if (row.id !== id) return row;
    if ("text" in change) return { ...row, text: change.text };
    // A cleared field leaves its key out (no prompt, no image, no audio), as a new row has none.
    const next =
      "prompt" in change
        ? { ...row, prompt: change.prompt }
        : "image" in change
          ? { ...row, image: change.image }
          : { ...row, audio: change.audio };
    return {
      id: next.id,
      text: next.text,
      ...(next.prompt === undefined ? {} : { prompt: next.prompt }),
      ...(next.image === undefined ? {} : { image: next.image }),
      ...(next.audio === undefined ? {} : { audio: next.audio }),
    };
  });

/**
 * What a row sends to `embed` / `tokenize`: its text as a string, or — with an image or an audio —
 * the text part, then the image part (the RGB8 pixels), then the audio part (the mono samples); an
 * empty text adds no token, so an image or an audio alone is that.
 */
export const rowInput = (row: TextRow): string | readonly EmbedContentPart[] =>
  row.image === undefined && row.audio === undefined
    ? row.text
    : [
        { type: "text", text: row.text },
        ...(row.image === undefined ? [] : [{ type: "image", image: row.image.rgb } as const]),
        ...(row.audio === undefined ? [] : [{ type: "audio", audio: row.audio.audio } as const]),
      ];

/** What counts a row's tokens: a pipeline's `tokenize`, narrowed so a test can stand in for it. */
export type RowTokenizer = (
  input: string | readonly EmbedContentPart[],
  options: { readonly prompt?: string },
) => { readonly length: number };

/**
 * What the pipeline makes of `row` on the host: the number of tokens it would embed (the prompt's
 * prefix, `<bos>` and `<eos>` included, and an image's or an audio's soft tokens with its two
 * markers), or its refusal when it cannot tokenize the row at all (a `ModelInputError` — a prompt
 * the model does not declare, an image or an audio on a pipeline without its tower, an audio that is
 * not 16 kHz or is longer than 30 s), with the library's message. A count over the limit is returned
 * as is: `embed` refuses it. Any other failure propagates.
 */
export const rowTokens = (
  tokenize: RowTokenizer,
  row: TextRow,
): { readonly count: number } | { readonly refusal: string } => {
  try {
    return {
      count: tokenize(rowInput(row), row.prompt === undefined ? {} : { prompt: row.prompt }).length,
    };
  } catch (error) {
    if (error instanceof ModelInputError) return { refusal: error.message };
    throw error;
  }
};

/** The count of {@link rowTokens}, or `undefined` for a row the pipeline refuses. */
export const rowTokenCount = (tokenize: RowTokenizer, row: TextRow): number | undefined => {
  const tokens = rowTokens(tokenize, row);
  return "count" in tokens ? tokens.count : undefined;
};

/** The model and preset a run builds with, and whether each is the manifest's default. */
export type Selection = {
  readonly model: string;
  readonly quant: string;
  readonly modelIsDefault: boolean;
  readonly quantIsDefault: boolean;
};

/**
 * The selection in effect: the user's choices where the manifest has them, else the defaults. A
 * preset chosen for another model falls back to the chosen model's `defaultQuant`.
 */
export const effectiveSelection = (
  manifest: Manifest,
  chosenModel: string | undefined,
  chosenQuant: string | undefined,
): Selection => {
  const model =
    chosenModel !== undefined && Object.hasOwn(manifest.models, chosenModel)
      ? chosenModel
      : manifest.defaultModel;
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const quant =
    chosenQuant !== undefined && Object.hasOwn(entry.quants, chosenQuant)
      ? chosenQuant
      : entry.defaultQuant;
  return {
    model,
    quant,
    modelIsDefault: model === manifest.defaultModel,
    quantIsDefault: quant === entry.defaultQuant,
  };
};

/** What a pipeline was built for: the GPU, the source, the model and the preset. */
export type PipelineKey<Gpu, Source> = {
  readonly gpu: Gpu;
  readonly source: Source;
  readonly model: string;
  readonly quant: string;
};

/**
 * Whether the held pipeline serves a run that wants `wanted` as is (the prompt and the dimension
 * are per call, so they never rebuild it). Otherwise the page disposes it before building the next
 * one, so at most one Session is open at a time. The GPU and the source compare by identity: a new
 * GPU is a new device, and the page's source is memoized, so a new source object is a changed
 * choice.
 */
export const canReuse = <Gpu, Source>(
  held: PipelineKey<Gpu, Source> | undefined,
  wanted: PipelineKey<Gpu, Source>,
): boolean =>
  held !== undefined &&
  held.gpu === wanted.gpu &&
  held.source === wanted.source &&
  held.model === wanted.model &&
  held.quant === wanted.quant;

/**
 * Why Run is disabled now (`undefined` = it can run). The first reason that holds wins. An empty
 * text is not a reason: the pipeline accepts it (`<bos>` `<eos>` alone).
 */
export const runBlocker = (input: {
  readonly gpuReady: boolean;
  readonly manifestReady: boolean;
  readonly configError: boolean;
  readonly running: boolean;
  readonly rows: readonly TextRow[];
}): string | undefined => {
  if (input.running) return "A run is in progress.";
  if (!input.manifestReady) return "The manifest is not loaded.";
  if (input.configError) return "The model's pipelineConfig cannot be read.";
  if (!input.gpuReady) return "The GPU is not ready.";
  if (input.rows.length === 0) return "Add a row.";
  return undefined;
};
