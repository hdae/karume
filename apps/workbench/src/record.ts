/**
 * The record of one workbench run: the envelope every entry shares (what ran, with which
 * parameters, where and on which checkout) around the entry's own result. Pure: the browser shell,
 * the dev server and the Node run build and serialize records with these functions, so the saved
 * bytes are the same everywhere.
 *
 * SPEC: docs/spec/workbench.md#the-record
 */

import type { EntryResult, ParameterValues, Verdict } from "./entry.ts";

/** The version of the envelope; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-workbench/1";

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

export type WorkbenchRecord = {
  readonly format: typeof RECORD_FORMAT;
  /** The typed parameter values the run used. */
  readonly entry: { readonly id: string; readonly parameters: ParameterValues };
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

/** `2026-11-16T09-30-12`: a UTC timestamp that is safe in file names on every OS. */
export const utcStamp = (at: Date): string => at.toISOString().slice(0, 19).replaceAll(":", "-");

/**
 * The record of a run. A run that settled after its signal was aborted is `cancelled` (with the
 * result it returned, if any); one that threw otherwise is `error`, without a result.
 */
export const buildRecord = (facts: {
  readonly entry: { readonly id: string; readonly parameters: ParameterValues };
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
