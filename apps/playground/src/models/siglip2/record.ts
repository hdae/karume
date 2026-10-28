/**
 * The saved record of one SigLIP 2 run (JSON; the page shows a summary of it) and its assembly from
 * what the run measured. Pure apart from the types, so `vp test` covers it; the sha256 digests are
 * taken by the page (WebCrypto is async) and handed in.
 */

import type { SessionDiagnostics } from "@karume/runtime";

import type { RecordEnvironment } from "@/lib/environment";
import type { PipelineSource } from "@/lib/distributions";
import {
  type ComponentRecord,
  foldComponentDiagnostics,
  seconds,
  type SourceRecord,
  sourceRecord,
  type SubmitCounts,
  submitCountsSince,
  tallyTiming,
  timingOf,
  type TimingTally,
} from "@/lib/record";

import { cosineSimilarity, type EmbeddingStats, embeddingStats } from "./embedding";
import type { ImageSlot, ModelFacts, Selection } from "./request";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-siglip2/1";

/** One input image as the record names it. */
export type ImageRecord = {
  readonly slot: ImageSlot;
  readonly name: string;
  readonly width: number;
  readonly height: number;
  /** sha256 of the RGB8 pixels the pipeline was given (alpha already dropped). */
  readonly rgbSha256: string;
};

/** One embedding: its statistics and the sha256 of its raw f32 bytes (little-endian). */
export type EmbeddingRecord = EmbeddingStats & {
  readonly slot: ImageSlot;
  readonly sha256: string;
};

/** Present only for a run with two images. */
export type ComparisonRecord =
  | { readonly cosineSimilarity: number }
  | { readonly cosineSimilarity: null; readonly reason: string };

export type Siglip2Record = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "siglip2";
  readonly distribution: { readonly directory: string; readonly manifestSha256: string };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: SourceRecord;
  readonly model: ModelFacts & { readonly name: string; readonly isDefault: boolean };
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly request: { readonly images: readonly ImageRecord[] };
  readonly timings: {
    /** true: the pipeline (and its Session) built by an earlier run was used as is. */
    readonly pipelineReused: boolean;
    /**
     * Seconds of `fromPretrained` (manifest, weight download or cache read, Session build and
     * weight upload); null when the pipeline was reused.
     */
    readonly buildSeconds: number | null;
    /** Seconds of each `embed` call, in request order (preprocessing, one vision run, readback). */
    readonly embedSeconds: readonly number[];
    readonly wallSeconds: number;
  };
  readonly output: {
    readonly embeddings: readonly EmbeddingRecord[];
    readonly comparison?: ComparisonRecord;
  };
  /**
   * The vision Session's diagnostics over this run's embeddings (one report per `embed`). `build`
   * is the Session's construction cost — from an earlier run when `timings.pipelineReused`.
   */
  readonly diagnostics: { readonly vision?: VisionRecord };
};

/**
 * The vision component's record. The Session's counters only grow and the page keeps one Session
 * across runs, so `submitCount` / `dispatchCount` are this run's share (the latest report minus the
 * counters before the run) — comparable between a fresh and a reused pipeline — and the totals the
 * report gave are kept beside them.
 */
export type VisionRecord = ComponentRecord & {
  readonly sessionSubmitCount: number;
  readonly sessionDispatchCount: number;
};

/** What one `onRunDiagnostics` report contributes (a `SessionDiagnostics` satisfies it). */
export type DiagnosticsReport = Parameters<typeof foldComponentDiagnostics>[1] & {
  readonly lastRunTiming: SessionDiagnostics["lastRunTiming"];
};

/**
 * Folds the run's reports into the vision component's record (`undefined` for none). `before` = the
 * Session's counters when the run started: `NO_SUBMITS` (of the kit) for a pipeline the run built,
 * the last report of the earlier runs for a reused one.
 */
export const visionDiagnostics = (
  reports: readonly DiagnosticsReport[],
  before: SubmitCounts,
): VisionRecord | undefined => {
  let folded: ComponentRecord | undefined;
  let tally: TimingTally | undefined;
  for (const report of reports) {
    folded = foldComponentDiagnostics(folded, report);
    tally = tallyTiming(tally, report.lastRunTiming);
  }
  if (folded === undefined) return undefined;
  const vision: VisionRecord = {
    ...folded,
    ...submitCountsSince(before, folded),
    sessionSubmitCount: folded.submitCount,
    sessionDispatchCount: folded.dispatchCount,
  };
  return tally === undefined ? vision : { ...vision, timing: timingOf(tally) };
};

/** The cosine of the two embeddings, or why there is none. */
export const compareEmbeddings = (left: Float32Array, right: Float32Array): ComparisonRecord => {
  try {
    return { cosineSimilarity: Number(cosineSimilarity(left, right).toFixed(6)) };
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return { cosineSimilarity: null, reason: error.message };
  }
};

/** One embedded image as the run measured it. */
export type MeasuredImage = {
  readonly slot: ImageSlot;
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly rgbSha256: string;
  readonly embedding: Float32Array;
  readonly embeddingSha256: string;
  readonly embedMs: number;
};

export const assembleRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly directory: string;
  readonly manifestSha256: string;
  readonly source: PipelineSource;
  readonly selection: Selection;
  readonly facts: ModelFacts;
  readonly presetLabel: string;
  readonly images: readonly MeasuredImage[];
  /** `undefined` = the pipeline was reused. */
  readonly buildMs: number | undefined;
  readonly wallMs: number;
  readonly reports: readonly DiagnosticsReport[];
  /** The vision Session's counters when the run started ({@link visionDiagnostics}). */
  readonly countersBefore: SubmitCounts;
}): Siglip2Record => {
  const { selection, images } = input;
  const [first, second] = images;
  const vision = visionDiagnostics(input.reports, input.countersBefore);
  return {
    format: RECORD_FORMAT,
    page: "siglip2",
    ...input.environment,
    distribution: { directory: input.directory, manifestSha256: input.manifestSha256 },
    source: sourceRecord(input.source),
    model: { name: selection.model, isDefault: selection.modelIsDefault, ...input.facts },
    preset: {
      quant: selection.quant,
      label: input.presetLabel,
      isDefault: selection.quantIsDefault,
    },
    request: {
      images: images.map(({ slot, name, width, height, rgbSha256 }) => ({
        slot,
        name,
        width,
        height,
        rgbSha256,
      })),
    },
    timings: {
      pipelineReused: input.buildMs === undefined,
      buildSeconds: input.buildMs === undefined ? null : seconds(input.buildMs),
      embedSeconds: images.map(({ embedMs }) => seconds(embedMs)),
      wallSeconds: seconds(input.wallMs),
    },
    output: {
      embeddings: images.map(({ slot, embedding, embeddingSha256 }) => ({
        slot,
        ...embeddingStats(embedding),
        sha256: embeddingSha256,
      })),
      ...(first === undefined || second === undefined
        ? {}
        : { comparison: compareEmbeddings(first.embedding, second.embedding) }),
    },
    diagnostics: vision === undefined ? {} : { vision },
  };
};
