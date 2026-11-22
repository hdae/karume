/**
 * The saved record of one EmbeddingGemma run (JSON; the page shows a summary of it) and its assembly
 * from what the run measured. Pure apart from the types, so `vp test` covers it; the sha256 digests
 * are taken by the page (WebCrypto is async) and handed in.
 */

import type { PipelineSource } from "@karume/local-distributions/browser";
import type { SessionDiagnostics } from "@karume/runtime";

import type { RecordEnvironment } from "@/lib/environment";
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

import type { Selection } from "./request";
import { type EmbeddingStats, embeddingStats, similarityMatrix } from "./similarity";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-embeddinggemma/1";

/**
 * One row as it was sent: the text as typed, the prompt name (null = none) and the token count
 * `tokenize` gave for them (the prefix, `<bos>` and `<eos>` included — over `maxTokens` for a row
 * the pipeline refused for the model's limit, at or below it for a row refused for the device's
 * smaller buffer limit; null when the pipeline could not tokenize the row, an undeclared prompt).
 */
export type RowRequestRecord = {
  readonly text: string;
  readonly prompt: string | null;
  readonly tokens: number | null;
};

/** A refusal of the pipeline as it was raised (`ModelInputError`: fixing the input makes it pass). */
export type RefusalRecord = { readonly name: string; readonly message: string };

/** What one row gave: its embedding, or the pipeline's refusal of it. */
export type RowOutputRecord =
  | (EmbeddingStats & {
      readonly kind: "embedding";
      /** sha256 of the vector's raw f32 bytes (little-endian). */
      readonly sha256: string;
      /** Seconds of the `embed` call (host tokenization, one Session run, readback). */
      readonly embedSeconds: number;
    })
  | {
      readonly kind: "refused";
      readonly error: RefusalRecord;
      readonly embedSeconds: number;
    };

/**
 * The model Session's record. The Session's counters only grow and the page keeps one Session
 * across runs, so `submitCount` / `dispatchCount` are this run's share and the totals the report
 * gave are kept beside them (as on the SigLIP 2 page).
 */
export type ModelSessionRecord = ComponentRecord & {
  readonly sessionSubmitCount: number;
  readonly sessionDispatchCount: number;
};

export type EmbeddingGemmaRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "embeddinggemma";
  readonly distribution: { readonly directory: string; readonly manifestSha256: string };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: SourceRecord;
  readonly model: { readonly name: string; readonly isDefault: boolean };
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly request: {
    /** The width every row was asked for. */
    readonly dimension: number;
    readonly rows: readonly RowRequestRecord[];
  };
  readonly timings: {
    /** true: the pipeline (and its Session) built by an earlier run was used as is. */
    readonly pipelineReused: boolean;
    /** Seconds of `fromPretrained`; null when the pipeline was reused. */
    readonly buildSeconds: number | null;
    /** From Run to the last row's result (hashing excluded). */
    readonly wallSeconds: number;
  };
  readonly output: {
    /** One per row, in request order. */
    readonly rows: readonly RowOutputRecord[];
    /** `similarity[i][j]` = the dot product of rows i and j ({@link similarityMatrix}). */
    readonly similarity: readonly (readonly (number | null)[])[];
  };
  /** The model Session's diagnostics over this run's embeddings (one report per embedded row). */
  readonly diagnostics: { readonly model?: ModelSessionRecord };
};

/** What one `onRunDiagnostics` report contributes (a `SessionDiagnostics` satisfies it). */
export type DiagnosticsReport = Parameters<typeof foldComponentDiagnostics>[1] & {
  readonly lastRunTiming: SessionDiagnostics["lastRunTiming"];
};

/**
 * Folds the run's reports into the model Session's record (`undefined` for none — every row
 * refused). `before` = the Session's counters when the run started: `NO_SUBMITS` for a pipeline the
 * run built, the last report of the earlier runs for a reused one.
 */
export const modelDiagnostics = (
  reports: readonly DiagnosticsReport[],
  before: SubmitCounts,
): ModelSessionRecord | undefined => {
  let folded: ComponentRecord | undefined;
  let tally: TimingTally | undefined;
  for (const report of reports) {
    folded = foldComponentDiagnostics(folded, report);
    tally = tallyTiming(tally, report.lastRunTiming);
  }
  if (folded === undefined) return undefined;
  const record: ModelSessionRecord = {
    ...folded,
    ...submitCountsSince(before, folded),
    sessionSubmitCount: folded.submitCount,
    sessionDispatchCount: folded.dispatchCount,
  };
  return tally === undefined ? record : { ...record, timing: timingOf(tally) };
};

/** One row as the run measured it. */
export type MeasuredRow = {
  readonly text: string;
  readonly prompt: string | undefined;
  /** The row's token count (`rowTokenCount`); `undefined` when it could not be tokenized. */
  readonly tokens: number | undefined;
  readonly embedMs: number;
  readonly outcome:
    | {
        readonly kind: "embedding";
        readonly values: Float32Array<ArrayBuffer>;
        readonly sha256: string;
      }
    | { readonly kind: "refused"; readonly error: RefusalRecord };
};

const rowOutput = ({ embedMs, outcome }: MeasuredRow): RowOutputRecord =>
  outcome.kind === "embedding"
    ? {
        kind: "embedding",
        ...embeddingStats(outcome.values),
        sha256: outcome.sha256,
        embedSeconds: seconds(embedMs),
      }
    : { kind: "refused", error: outcome.error, embedSeconds: seconds(embedMs) };

export const assembleRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly directory: string;
  readonly manifestSha256: string;
  readonly source: PipelineSource;
  readonly selection: Selection;
  readonly presetLabel: string;
  readonly dimension: number;
  readonly rows: readonly MeasuredRow[];
  /** `undefined` = the pipeline was reused. */
  readonly buildMs: number | undefined;
  readonly wallMs: number;
  readonly reports: readonly DiagnosticsReport[];
  /** The model Session's counters when the run started ({@link modelDiagnostics}). */
  readonly countersBefore: SubmitCounts;
}): EmbeddingGemmaRecord => {
  const { selection, rows } = input;
  const model = modelDiagnostics(input.reports, input.countersBefore);
  return {
    format: RECORD_FORMAT,
    page: "embeddinggemma",
    ...input.environment,
    distribution: { directory: input.directory, manifestSha256: input.manifestSha256 },
    source: sourceRecord(input.source),
    model: { name: selection.model, isDefault: selection.modelIsDefault },
    preset: {
      quant: selection.quant,
      label: input.presetLabel,
      isDefault: selection.quantIsDefault,
    },
    request: {
      dimension: input.dimension,
      rows: rows.map(({ text, prompt, tokens }) => ({
        text,
        prompt: prompt ?? null,
        tokens: tokens ?? null,
      })),
    },
    timings: {
      pipelineReused: input.buildMs === undefined,
      buildSeconds: input.buildMs === undefined ? null : seconds(input.buildMs),
      wallSeconds: seconds(input.wallMs),
    },
    output: {
      rows: rows.map(rowOutput),
      similarity: similarityMatrix(
        rows.map(({ outcome }) => (outcome.kind === "embedding" ? outcome.values : undefined)),
      ),
    },
    diagnostics: model === undefined ? {} : { model },
  };
};
