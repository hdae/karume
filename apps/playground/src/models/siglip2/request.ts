/**
 * The page's form logic: which model and preset are in effect, what the chosen model declares about
 * its input and output, which images a run embeds and why Run is disabled. Pure, so `vp test`
 * covers it.
 */

import type { Manifest } from "@karume/hub";

import type { LoadedImage } from "@/lib/image";

/** The model and preset a run builds with, and whether each is the manifest's default. */
export type Selection = {
  readonly model: string;
  readonly quant: string;
  readonly modelIsDefault: boolean;
  readonly quantIsDefault: boolean;
};

/**
 * The selection in effect: the user's choices where the manifest has them, else the defaults. A
 * preset is a quant of one model, so a preset chosen for another model falls back to the chosen
 * model's `defaultQuant` (base and so400m may not share preset names).
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

/**
 * What the model's `pipelineConfig` declares about the input size and the embedding width, for the
 * hints and the record. The pipeline owns and validates that schema (not on the public surface), so
 * this only reads the three numbers when they are positive integers and leaves the rest absent —
 * the pipeline itself fails loudly on a malformed config when it is built.
 */
export type ModelFacts = {
  readonly imageWidth?: number;
  readonly imageHeight?: number;
  readonly hiddenDim?: number;
};

const positiveInteger = (
  config: Readonly<Record<string, unknown>>,
  key: string,
): number | undefined => {
  if (!Object.hasOwn(config, key)) return undefined;
  const value = config[key];
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
};

export const modelFacts = (manifest: Manifest, model: string): ModelFacts => {
  const entry = manifest.models[model];
  if (entry === undefined) return {};
  const config = entry.pipelineConfig;
  const imageWidth = positiveInteger(config, "imageWidth");
  const imageHeight = positiveInteger(config, "imageHeight");
  const hiddenDim = positiveInteger(config, "hiddenDim");
  return {
    ...(imageWidth === undefined ? {} : { imageWidth }),
    ...(imageHeight === undefined ? {} : { imageHeight }),
    ...(hiddenDim === undefined ? {} : { hiddenDim }),
  };
};

/** `224×224 input, 768-dim embedding` (the parts the config declares). */
export const modelFactsText = (facts: ModelFacts): string => {
  const parts: string[] = [];
  if (facts.imageWidth !== undefined && facts.imageHeight !== undefined) {
    parts.push(`${facts.imageWidth}×${facts.imageHeight} input`);
  }
  if (facts.hiddenDim !== undefined) parts.push(`${facts.hiddenDim}-dim embedding`);
  return parts.join(", ");
};

/** What a pipeline was built for: the GPU, the source, the model and the preset. */
export type PipelineKey<Gpu, Source> = {
  readonly gpu: Gpu;
  readonly source: Source;
  readonly model: string;
  readonly quant: string;
};

/**
 * Whether the held pipeline serves a run that wants `wanted` as is. Otherwise the page disposes it
 * before building the next one, so at most one Session (up to 1.7 GB of weights) is open at a time.
 * The GPU and the source compare by identity: a new GPU is a new device (the old Session cannot run
 * on it), and the page's source is memoized, so a new source object means a changed choice or a new
 * cache generation (fresh readers) — rebuilding then is the safe side.
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

/** The two image slots: A is required, B is optional (with B the run also compares the two). */
export type ImageSlot = "a" | "b";

export type SlotImage = { readonly slot: ImageSlot; readonly image: LoadedImage };

/** The images a run embeds, in order (A first). */
export const requestImages = (
  a: LoadedImage | undefined,
  b: LoadedImage | undefined,
): readonly SlotImage[] => [
  ...(a === undefined ? [] : [{ slot: "a" as const, image: a }]),
  ...(b === undefined ? [] : [{ slot: "b" as const, image: b }]),
];

/** Why Run is disabled now (`undefined` = it can run). The first reason that holds wins. */
export const runBlocker = (input: {
  readonly gpuReady: boolean;
  readonly manifestReady: boolean;
  readonly running: boolean;
  readonly images: readonly SlotImage[];
}): string | undefined => {
  if (input.running) return "A run is in progress.";
  if (!input.manifestReady) return "The manifest is not loaded.";
  if (!input.gpuReady) return "The GPU is not ready.";
  if (!input.images.some(({ slot }) => slot === "a")) return "Choose image A.";
  return undefined;
};
