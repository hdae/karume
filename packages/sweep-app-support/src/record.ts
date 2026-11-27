/**
 * The record of one workbench run: the envelope every entry shares (what ran, with which
 * parameters, where and on which checkout) around the entry's own result. Pure apart from the clock
 * {@link recordRun} reads: the workbench's browser shell, its dev server and its Node run, and the
 * benchmark app, build and serialize records with these functions, so the saved bytes are the same
 * everywhere.
 *
 * SPEC: docs/spec/workbench.md#the-record
 */

import type { ParameterValues } from "./parameters.ts";

/** The version of the envelope; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-workbench/1";

/** An entry id: kebab-case. The registry and the results directory both hold ids to it. */
export const ENTRY_ID = /^[a-z0-9]+(-[a-z0-9]+)*$/;

/** What a check-like entry concludes. Absent for an entry that only measures. */
export type Verdict = "pass" | "fail" | "incomplete";

/** What a run resolves to: the record's `verdict`, `summary` and `data`. */
export type EntryResult = {
  readonly verdict?: Verdict;
  /** A few plain lines a person reads first. */
  readonly summary: readonly string[];
  /** The entry's own record body. */
  readonly data: unknown;
};

export type AdapterSummary = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/** WebGPU does not expose the backend: a person declares it in a browser, Dawn's flags say it under Node. */
export type Backend = {
  readonly name: string;
  readonly source: "declared" | "dawn-flag" | "dawn-default";
};

/** Where a run ran: the browser shell or the Node command. */
export type Host = "browser" | "node";

export type RecordEnvironment = {
  readonly host: Host;
  readonly userAgent: string;
  /** Browser only. */
  readonly crossOriginIsolated?: boolean;
  readonly backend?: Backend;
  readonly adapter: AdapterSummary;
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
  readonly wgslLanguageFeatures: readonly string[];
};

export type Checkout = {
  readonly commit: string;
  readonly dirty: boolean;
  /** The version of `packages/runtime/package.json`. */
  readonly runtimeVersion: string;
};

export type Outcome = "done" | "error" | "cancelled";

/**
 * The batch a run was a step of. Absent from the record of a single run.
 *
 * SPEC: docs/spec/workbench.md#batches
 */
export type RecordBatch = {
  /** The batch's id: its file name without `.json`. */
  readonly id: string;
  /** When the batch run started (ISO 8601): the same in every record of that batch run. */
  readonly startedAt: string;
  /** The step's 1-based number in the file. */
  readonly step: number;
  /** The number of steps in the file. */
  readonly steps: number;
};

export type WorkbenchRecord = {
  readonly format: typeof RECORD_FORMAT;
  /** The typed parameter values the run used. */
  readonly entry: { readonly id: string; readonly parameters: ParameterValues };
  /** Present only for a run that was a step of a batch. */
  readonly batch?: RecordBatch;
  /** When the run ended (ISO 8601). */
  readonly recordedAt: string;
  /** Wall time of the entry's `run`. */
  readonly seconds: number;
  readonly outcome: Outcome;
  /** The stack (or message) of what `run` threw, when the outcome is `error`. */
  readonly error?: string;
  readonly environment: RecordEnvironment;
  readonly checkout: Checkout;
  readonly verdict?: Verdict;
  readonly summary?: readonly string[];
  readonly data?: unknown;
};

/** How `run` settled: its result or what it threw, and whether the signal was aborted by then. */
export type Settlement = {
  readonly aborted: boolean;
} & (
  | { readonly kind: "resolved"; readonly result: EntryResult }
  | { readonly kind: "rejected"; readonly error: unknown }
);

/** Seconds with millisecond precision (records stay readable and diff-friendly). */
export const seconds = (milliseconds: number): number => Number((milliseconds / 1000).toFixed(3));

/** What a thrown value says: an Error's stack (or message), otherwise its text. */
export const errorText = (error: unknown): string => {
  if (error instanceof Error) return error.stack ?? error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error) ?? typeof error;
  } catch {
    return typeof error;
  }
};

/** What a thrown value says in one line of a summary: an Error's message, otherwise its text. */
export const messageText = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

/** `2026-11-16T09-30-12`: a UTC timestamp that is safe in file names on every OS. */
export const utcStamp = (at: Date): string => at.toISOString().slice(0, 19).replaceAll(":", "-");

/**
 * The record of a run. A run that settled after its signal was aborted is `cancelled` (with the
 * result it returned, if any); one that threw otherwise is `error`, without a result. `batch`, when
 * given, sits right after `entry`; without it the record has no such key.
 */
export const buildRecord = (facts: {
  readonly entry: { readonly id: string; readonly parameters: ParameterValues };
  readonly batch?: RecordBatch;
  readonly endedAt: Date;
  readonly milliseconds: number;
  readonly environment: RecordEnvironment;
  readonly checkout: Checkout;
  readonly settlement: Settlement;
}): WorkbenchRecord => {
  const { settlement } = facts;
  const outcome: Outcome = settlement.aborted
    ? "cancelled"
    : settlement.kind === "rejected"
      ? "error"
      : "done";
  const result = settlement.kind === "resolved" ? settlement.result : undefined;
  return {
    format: RECORD_FORMAT,
    entry: facts.entry,
    ...(facts.batch === undefined ? {} : { batch: facts.batch }),
    recordedAt: facts.endedAt.toISOString(),
    seconds: seconds(facts.milliseconds),
    outcome,
    ...(outcome === "error" && settlement.kind === "rejected"
      ? { error: errorText(settlement.error) }
      : {}),
    environment: facts.environment,
    checkout: facts.checkout,
    ...(result?.verdict === undefined ? {} : { verdict: result.verdict }),
    ...(result === undefined ? {} : { summary: result.summary, data: result.data }),
  };
};

/** The text that is saved, downloaded and copied: one function, so the bytes are the same everywhere. */
export const recordText = (record: WorkbenchRecord): string =>
  `${JSON.stringify(record, null, 1)}\n`;

/**
 * Runs `run` and wraps how it settled in the record ({@link buildRecord}): `seconds` is its wall
 * time, `recordedAt` the moment it settled, and it is `cancelled` when `signal` was aborted by then.
 * Never rejects for what `run` throws (that is the record's `error`).
 */
export const recordRun = async (
  facts: {
    readonly entry: { readonly id: string; readonly parameters: ParameterValues };
    readonly batch?: RecordBatch;
    readonly environment: RecordEnvironment;
    readonly checkout: Checkout;
    readonly signal: AbortSignal;
  },
  run: () => Promise<EntryResult>,
): Promise<WorkbenchRecord> => {
  const started = performance.now();
  let settlement: Settlement;
  try {
    const result = await run();
    settlement = { kind: "resolved", result, aborted: facts.signal.aborted };
  } catch (error) {
    settlement = { kind: "rejected", error, aborted: facts.signal.aborted };
  }
  const milliseconds = performance.now() - started;
  return buildRecord({
    entry: facts.entry,
    ...(facts.batch === undefined ? {} : { batch: facts.batch }),
    endedAt: new Date(),
    milliseconds,
    environment: facts.environment,
    checkout: facts.checkout,
    settlement,
  });
};
