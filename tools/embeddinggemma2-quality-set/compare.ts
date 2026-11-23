/**
 * The pure part of `compare-presets.ts`: the quality set's manifest as that command reads it, the
 * command line, and the numbers it prints — the cosine of each item between two presets, the
 * cross-modal top-1 of each preset, and the summary of the per-item times. No GPU, no file.
 *
 * Every cosine is in f64 over the leading `width` values of the full vectors. A shorter width of
 * the pipeline is that slice re-normalised, and a cosine does not see the norm, so the cosine of
 * the slices is the cosine of the pipeline's shorter vectors.
 */

import { type AudioItem, type ImageItem, MANIFEST_FORMAT, type QualitySetManifest } from "./set.ts";

/** The widths the command reports: the full width and the narrowest the model declares. */
export const WIDTHS = [768, 128] as const;

/** The rate every clip must have (the library takes 16 kHz only — nothing is resampled). */
export const CLIP_SAMPLE_RATE = 16000;

/** A manifest or a command line the command cannot act on. */
export class CompareInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CompareInputError";
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const stringField = (entry: Record<string, unknown>, key: string, where: string): string => {
  const value = Object.hasOwn(entry, key) ? entry[key] : undefined;
  if (typeof value !== "string" || value === "") {
    throw new CompareInputError(`${where}: '${key}' is not a non-empty string`);
  }
  return value;
};

/**
 * Reads a quality-set manifest (`karume-embeddinggemma2-quality-set/1`): every image with its
 * file and caption, every clip with its file, transcript and rate. Anything else fails loudly — a
 * missing caption would otherwise drop an item from retrieval silently.
 */
export const parseQualitySetManifest = (text: string, where: string): QualitySetManifest => {
  const document: unknown = JSON.parse(text);
  if (!isRecord(document) || document["format"] !== MANIFEST_FORMAT) {
    throw new CompareInputError(`${where}: not a ${MANIFEST_FORMAT} manifest`);
  }
  const list = (key: "images" | "audio"): Record<string, unknown>[] => {
    const value = document[key];
    if (!Array.isArray(value) || value.length < 2 || !value.every(isRecord)) {
      throw new CompareInputError(`${where}: '${key}' is not a list of at least two entries`);
    }
    return value;
  };
  const images: ImageItem[] = list("images").map((entry, index) => {
    const at = `${where} images[${index}]`;
    const seed = entry["seed"];
    if (typeof seed !== "number") throw new CompareInputError(`${at}: 'seed' is not a number`);
    return {
      file: stringField(entry, "file", at),
      caption: stringField(entry, "caption", at),
      generator: stringField(entry, "generator", at),
      prompt: stringField(entry, "prompt", at),
      seed,
    };
  });
  const audio: AudioItem[] = list("audio").map((entry, index) => {
    const at = `${where} audio[${index}]`;
    const sampleRate = entry["sampleRate"];
    if (typeof sampleRate !== "number" || !Number.isInteger(sampleRate) || sampleRate <= 0) {
      throw new CompareInputError(`${at}: 'sampleRate' is not a positive integer`);
    }
    return {
      file: stringField(entry, "file", at),
      transcript: stringField(entry, "transcript", at),
      generator: stringField(entry, "generator", at),
      sampleRate,
    };
  });
  return { format: MANIFEST_FORMAT, images, audio };
};

/** The distinct texts in the order of their first appearance, and each item's text index. */
export const distinctTexts = (
  texts: readonly string[],
): { readonly texts: readonly string[]; readonly itemText: readonly number[] } => {
  const order = new Map<string, number>();
  for (const text of texts) if (!order.has(text)) order.set(text, order.size);
  return { texts: [...order.keys()], itemText: texts.map((text) => order.get(text) ?? -1) };
};

/** The cosine of the leading `width` values of two vectors, in f64. */
export const cosine = (
  left: Float32Array<ArrayBuffer>,
  right: Float32Array<ArrayBuffer>,
  width: number,
): number => {
  if (left.length < width || right.length < width) {
    throw new RangeError(
      `a cosine at width ${width} of vectors ${left.length} and ${right.length} wide`,
    );
  }
  let dot = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  for (let index = 0; index < width; index += 1) {
    dot += left[index] * right[index];
    leftSquares += left[index] * left[index];
    rightSquares += right[index] * right[index];
  }
  return dot / Math.sqrt(leftSquares * rightSquares);
};

/** The lowest and the mean of named values, and the name of the lowest. */
export type CosineSummary = { readonly min: number; readonly mean: number; readonly worst: string };

export const summarizeCosines = (
  values: readonly { readonly name: string; readonly value: number }[],
): CosineSummary => {
  if (values.length === 0) throw new RangeError("a summary of no values");
  let lowest = values[0];
  let sum = 0;
  for (const entry of values) {
    if (entry.value < lowest.value) lowest = entry;
    sum += entry.value;
  }
  return { min: lowest.value, mean: sum / values.length, worst: lowest.name };
};

/** The index of the row's best candidate (the first of equal ones). */
const best = (
  query: Float32Array<ArrayBuffer>,
  candidates: readonly Float32Array<ArrayBuffer>[],
  width: number,
): number => {
  let chosen = 0;
  let score = -Infinity;
  candidates.forEach((candidate, index) => {
    const similarity = cosine(query, candidate, width);
    if (similarity > score) {
      score = similarity;
      chosen = index;
    }
  });
  return chosen;
};

/**
 * Cross-modal top-1 of one preset at one width (the screening's definitions): each item ranks the
 * distinct texts and hits when its own text is first; each text ranks the items and hits when the
 * first is an item of that text (several clips share a transcript). `choices` are the chosen
 * candidates, so two presets can be compared choice by choice.
 */
export type Retrieval = {
  readonly itemToText: { readonly hits: number; readonly choices: readonly number[] };
  readonly textToItem: { readonly hits: number; readonly choices: readonly number[] };
};

export const retrieval = (
  items: readonly Float32Array<ArrayBuffer>[],
  texts: readonly Float32Array<ArrayBuffer>[],
  itemText: readonly number[],
  width: number,
): Retrieval => {
  if (items.length !== itemText.length) {
    throw new RangeError(`${items.length} items and ${itemText.length} text indices`);
  }
  const forward = items.map((item) => best(item, texts, width));
  const backward = texts.map((text) => best(text, items, width));
  return {
    itemToText: {
      hits: forward.filter((chosen, item) => chosen === itemText[item]).length,
      choices: forward,
    },
    textToItem: {
      hits: backward.filter((chosen, text) => itemText[chosen] === text).length,
      choices: backward,
    },
  };
};

/** How many choices two presets share (the same candidate for the same query). */
export const agreement = (left: readonly number[], right: readonly number[]): number => {
  if (left.length !== right.length) throw new RangeError("choices of two lengths");
  return left.filter((chosen, index) => chosen === right[index]).length;
};

/** The mean, the median (the mean of the two middle values for an even count), min and max. */
export type TimeSummary = {
  readonly mean: number;
  readonly median: number;
  readonly min: number;
  readonly max: number;
};

export const summarizeTimes = (values: readonly number[]): TimeSummary => {
  if (values.length === 0) throw new RangeError("a summary of no times");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length >> 1;
  const median =
    sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  return {
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    median,
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
};

export const COMPARE_USAGE =
  "usage: node tools/embeddinggemma2-quality-set/compare-presets.ts [<record directory>]";

/** Reads `[<record directory>]` (`undefined` = the dated default). */
export const parseCompareCommandLine = (args: readonly string[]): string | undefined => {
  const options = args.filter((arg) => arg.startsWith("-"));
  if (options.length > 0)
    throw new CompareInputError(`unknown option ${options[0]}\n${COMPARE_USAGE}`);
  if (args.length > 1)
    throw new CompareInputError(`more than one record directory\n${COMPARE_USAGE}`);
  return args[0];
};
