/**
 * The page's form logic: the list of text rows (each with its own prompt), a row's token count,
 * which model and preset are in effect, whether the held pipeline serves a run, and why Run is
 * disabled. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import { ModelInputError } from "@karume/models/embeddinggemma";

/** One input of a run: a text and the prompt whose prefix goes before it (`undefined` = none). */
export type TextRow = {
  /** Stable across edits and removals (the React key); never reused within one page. */
  readonly id: number;
  readonly text: string;
  readonly prompt?: string;
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

/** `rows` with row `id` changed (`prompt: undefined` clears the prompt). */
export const updateRow = (
  rows: readonly TextRow[],
  id: number,
  change: { readonly text: string } | { readonly prompt: string | undefined },
): readonly TextRow[] =>
  rows.map((row) => {
    if (row.id !== id) return row;
    if ("text" in change) return { ...row, text: change.text };
    return {
      id: row.id,
      text: row.text,
      ...(change.prompt === undefined ? {} : { prompt: change.prompt }),
    };
  });

/** What counts a row's tokens: a pipeline's `tokenize`, narrowed so a test can stand in for it. */
export type RowTokenizer = (
  text: string,
  options: { readonly prompt?: string },
) => { readonly length: number };

/**
 * The number of tokens the pipeline would embed for `row` (the prompt's prefix, `<bos>` and `<eos>`
 * included), or `undefined` when it cannot tokenize the row at all (a `ModelInputError` — today a
 * prompt the model does not declare). A count over the limit is returned as is: `embed` refuses it.
 * Any other failure propagates.
 */
export const rowTokenCount = (tokenize: RowTokenizer, row: TextRow): number | undefined => {
  try {
    return tokenize(row.text, row.prompt === undefined ? {} : { prompt: row.prompt }).length;
  } catch (error) {
    if (error instanceof ModelInputError) return undefined;
    throw error;
  }
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
