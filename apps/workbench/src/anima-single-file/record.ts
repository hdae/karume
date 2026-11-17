/**
 * The record of the `anima-single-file` entry, its pure parts: the arms' timings and image hashes
 * folded into `karume-anima-single-file/1` and the summary lines.
 *
 * SPEC: docs/spec/workbench.md#the-anima-single-file-record
 */

import type { DistributionFacts, InputFacts } from "../entry.ts";

export const ANIMA_SINGLE_FILE_FORMAT = "karume-anima-single-file/1";

/** Where the text conditioner comes from: the file (when it holds one) or the distribution. */
export const TEXT_CONDITIONER_SOURCES = ["file", "distribution"] as const;
export type TextConditionerSource = (typeof TEXT_CONDITIONER_SOURCES)[number];

/** One generation of an arm. */
export type GenerationRecord = {
  readonly seconds: number;
  /** The sha256 of the image's RGB8 pixels (what the Anima e2e hashes). */
  readonly pixelSha256: string;
};

/** One pipeline: how it was built and its two generations. */
export type ArmRecord = {
  readonly name: "fromSingleFile" | "fromPretrained";
  /** Building the pipeline (`fromSingleFile` / `fromPretrained` until it resolves). */
  readonly buildSeconds: number;
  readonly generations: readonly GenerationRecord[];
};

export type AnimaSingleFileRequest = {
  readonly model: string;
  /** The quant the run used (the manifest's default resolved). */
  readonly quant: string;
  readonly textConditioner: TextConditionerSource;
  /** `retainConverted` of the `fromSingleFile` arm (the converted weights kept in memory). */
  readonly retainConverted: boolean;
  readonly compare: boolean;
  readonly prompt: string;
  readonly resolution: { readonly width: number; readonly height: number };
  readonly steps: number;
  readonly seed: number;
};

export type AnimaSingleFileData = {
  readonly format: typeof ANIMA_SINGLE_FILE_FORMAT;
  readonly request: AnimaSingleFileRequest;
  readonly file: InputFacts & {
    /** The name map's layout that recognised the file, and the keys it placed per component. */
    readonly layout: string;
    readonly placed: Readonly<Record<string, number>>;
  };
  readonly distribution: DistributionFacts;
  /** The components that came from the file. */
  readonly fromFile: readonly string[];
  /** `fromSingleFile` first, then `fromPretrained` when the request compares. */
  readonly arms: readonly ArmRecord[];
  readonly equal: {
    /** Every image of the run has the same pixels. */
    readonly all: boolean;
    /** The first images of the two arms have the same pixels; `null` without a comparison. */
    readonly arms: boolean | null;
  };
};

/** Folds the arms into the record (the equalities derived from the hashes). */
export const animaSingleFileData = (
  fields: Omit<AnimaSingleFileData, "format" | "equal">,
): AnimaSingleFileData => {
  const hashes = fields.arms.flatMap((arm) => arm.generations.map((image) => image.pixelSha256));
  const [first, second] = fields.arms;
  return {
    format: ANIMA_SINGLE_FILE_FORMAT,
    ...fields,
    equal: {
      all: hashes.every((hash) => hash === hashes[0]),
      arms:
        first === undefined || second === undefined
          ? null
          : first.generations[0]?.pixelSha256 === second.generations[0]?.pixelSha256,
    },
  };
};

const seconds = (value: number): string => `${value.toFixed(1)} s`;

/** The lines a person reads first. */
export const animaSingleFileSummary = (data: AnimaSingleFileData): string[] => [
  `${data.file.path} (${data.file.layout}) → ${data.request.model} / ${data.request.quant}: from the file ${data.fromFile.join(", ")}${data.request.retainConverted ? "; the converted weights kept in memory" : ""}`,
  ...data.arms.map(
    (arm) =>
      `${arm.name}: built in ${seconds(arm.buildSeconds)}; ${arm.generations
        .map(
          (image, index) =>
            `generation ${index + 1} ${seconds(image.seconds)} ${image.pixelSha256.slice(0, 16)}`,
        )
        .join("; ")}`,
  ),
  data.equal.arms === null
    ? `images equal within the run: ${data.equal.all ? "yes" : "no"}`
    : `fromSingleFile and fromPretrained give ${data.equal.arms ? "the same image" : "different images"}; every image equal: ${data.equal.all ? "yes" : "no"}`,
];
