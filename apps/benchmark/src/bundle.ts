/**
 * The file a run leaves (`karume-benchmark/1`): the run's workbench records exactly as the
 * workbench writes them (`karume-workbench/1`, entries `gemv-sweep` and `geometry-sweep`), wrapped
 * with what the page knows of the run — when it ran, whether it ran to the end and why not, the
 * trial the URL asked for and whether the page was hidden. The reader (`unpack.ts`) gives every
 * record back as the file the workbench would have written, at the path it would have written it to.
 * Pure.
 *
 * SPEC: docs/spec/benchmark.md#the-file
 */

import { GEMV_SWEEP_ENTRY_ID } from "@karume/sweep-app-support/gemv-sweep/settings";
import { GEOMETRY_SWEEP_ENTRY_ID } from "@karume/sweep-app-support/geometry-sweep/settings";
import {
  ENTRY_ID,
  RECORD_FORMAT,
  recordText,
  utcStamp,
  type WorkbenchRecord,
} from "@karume/sweep-app-support/record";
import { adapterDirectory, recordPath } from "@karume/sweep-app-support/record-path";

import type { Trial } from "./plan.ts";

export const BUNDLE_FORMAT = "karume-benchmark/1";

/** Why a run ended. */
export type BundleStop =
  | { readonly kind: "completed" }
  | { readonly kind: "cancelled" }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string }
  | { readonly kind: "error"; readonly message: string };

/**
 * The page's screen wake lock when the run ended (`wake-lock.ts`): `held`; `not-held` (the browser
 * released it, as it does while the page is hidden, and the page had not got it back — or had not
 * been granted it yet); `failed` (the browser refused the page's last request); `unavailable` (no
 * Screen Wake Lock API).
 */
export type WakeLockState = "held" | "not-held" | "unavailable" | "failed";

const WAKE_LOCK_STATES = [
  "held",
  "not-held",
  "unavailable",
  "failed",
] as const satisfies readonly WakeLockState[];

/** What the page saw of itself while the run went. */
export type PageFacts = {
  /** How often the page was hidden (another tab, a minimised window), and for how long in all. */
  readonly hiddenCount: number;
  readonly hiddenSeconds: number;
  readonly wakeLock: WakeLockState;
};

export type Bundle = {
  readonly format: typeof BUNDLE_FORMAT;
  /** When the run started and ended (ISO 8601). */
  readonly startedAt: string;
  readonly endedAt: string;
  /** Every planned sweep ran to its end. */
  readonly complete: boolean;
  readonly stop: BundleStop;
  /** The entries of the planned sweeps, in run order. */
  readonly sweeps: readonly string[];
  /** What the URL asked for (`{}` for a run of the defaults). */
  readonly trial: Trial;
  readonly page: PageFacts;
  /** One record per sweep that ran, as the workbench writes it. */
  readonly records: readonly WorkbenchRecord[];
};

/** The bundle's text: one-space JSON with a final newline, as a record's. */
export const bundleText = (bundle: Bundle): string => `${JSON.stringify(bundle, null, 1)}\n`;

/**
 * The bundle's file name: `karume-benchmark_<host>-<adapter name>_<UTC end time>.json` (the adapter
 * name by the record directory's rule), with `_incomplete` before `.json` when the run stopped short.
 */
export const bundleFileName = (bundle: Bundle): string => {
  const environment = bundle.records[0]?.environment;
  const adapter = adapterDirectory(environment?.host ?? "browser", environment?.adapter ?? {});
  return `karume-benchmark_${adapter}_${utcStamp(new Date(bundle.endedAt))}${bundle.complete ? "" : "_incomplete"}.json`;
};

/** A file that is not a bundle this reader takes. */
export class BundleError extends Error {
  override name = "BundleError";
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isString = (value: unknown): value is string => typeof value === "string";

const isStrings = (value: unknown): value is readonly string[] =>
  Array.isArray(value) && value.every(isString);

const isOptional = (value: unknown, check: (value: unknown) => boolean): boolean =>
  value === undefined || check(value);

const isDate = (value: unknown): value is string =>
  isString(value) && !Number.isNaN(Date.parse(value));

const isParameterValue = (value: unknown): boolean =>
  typeof value === "boolean" ||
  (typeof value === "number" && Number.isFinite(value)) ||
  isString(value) ||
  isStrings(value);

/** The reason a value is not a workbench record of one of the two sweeps, or nothing. */
const recordProblem = (value: unknown): string | undefined => {
  if (!isRecord(value) || value.format !== RECORD_FORMAT)
    return `its format is not ${RECORD_FORMAT}`;
  const { entry, environment, checkout } = value;
  if (
    !isRecord(entry) ||
    !isString(entry.id) ||
    !ENTRY_ID.test(entry.id) ||
    !isRecord(entry.parameters) ||
    !Object.values(entry.parameters).every(isParameterValue)
  ) {
    return "it has no entry with an id and parameters";
  }
  if (entry.id !== GEMV_SWEEP_ENTRY_ID && entry.id !== GEOMETRY_SWEEP_ENTRY_ID) {
    return `its entry ${entry.id} is neither ${GEMV_SWEEP_ENTRY_ID} nor ${GEOMETRY_SWEEP_ENTRY_ID}`;
  }
  if (value.batch !== undefined) return "it is a step of a batch";
  if (!isDate(value.recordedAt) || typeof value.seconds !== "number") {
    return "it has no recordedAt and seconds";
  }
  if (value.outcome !== "done" && value.outcome !== "error" && value.outcome !== "cancelled") {
    return "its outcome is not done, error or cancelled";
  }
  if (!isOptional(value.error, isString)) return "its error is not a text";
  const adapter = isRecord(environment) ? environment.adapter : undefined;
  if (
    !isRecord(environment) ||
    (environment.host !== "browser" && environment.host !== "node") ||
    !isString(environment.userAgent) ||
    !isOptional(environment.crossOriginIsolated, (one) => typeof one === "boolean") ||
    !isOptional(
      environment.backend,
      (one) => isRecord(one) && isString(one.name) && isString(one.source),
    ) ||
    !isRecord(adapter) ||
    !["vendor", "architecture", "device", "description"].every((key) => isString(adapter[key])) ||
    !isStrings(environment.features) ||
    !isRecord(environment.limits) ||
    !Object.values(environment.limits).every((limit) => typeof limit === "number") ||
    !isStrings(environment.wgslLanguageFeatures)
  ) {
    return "its environment is not a host, a user agent, an adapter, features, limits and WGSL features";
  }
  if (
    !isRecord(checkout) ||
    !isString(checkout.commit) ||
    typeof checkout.dirty !== "boolean" ||
    !isString(checkout.runtimeVersion)
  ) {
    return "its checkout is not a commit, dirty and a runtime version";
  }
  if (
    !isOptional(value.verdict, (one) => one === "pass" || one === "fail" || one === "incomplete")
  ) {
    return "its verdict is not pass, fail or incomplete";
  }
  if (!isOptional(value.summary, isStrings)) return "its summary is not lines of text";
  return undefined;
};

/** A workbench record of one of the two sweeps, every envelope field checked (its data is the entry's). */
const isSweepRecord = (value: unknown): value is WorkbenchRecord =>
  recordProblem(value) === undefined;

const isStop = (value: unknown): value is BundleStop =>
  isRecord(value) &&
  (value.kind === "completed" ||
    value.kind === "cancelled" ||
    (value.kind === "deviceLost" && isString(value.reason) && isString(value.message)) ||
    (value.kind === "error" && isString(value.message)));

const isTrial = (value: unknown): value is Trial =>
  isRecord(value) &&
  isOptional(value.cases, isStrings) &&
  isOptional(value.idleSeconds, (one) => Number.isInteger(one));

const isPageFacts = (value: unknown): value is PageFacts =>
  isRecord(value) &&
  typeof value.hiddenCount === "number" &&
  typeof value.hiddenSeconds === "number" &&
  WAKE_LOCK_STATES.some((state) => value.wakeLock === state);

/**
 * Reads a bundle's text. Throws {@link BundleError} naming what is wrong: not JSON, another format,
 * a field of the wrong shape, or a record that is not a workbench record of one of the two sweeps.
 */
export const parseBundle = (text: string): Bundle => {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (cause) {
    throw new BundleError("The file is not JSON", { cause });
  }
  if (!isRecord(value) || value.format !== BUNDLE_FORMAT) {
    throw new BundleError(`The file's format is not ${BUNDLE_FORMAT}`);
  }
  const { startedAt, endedAt, complete, stop, sweeps, trial, page, records } = value;
  if (!isDate(startedAt) || !isDate(endedAt))
    throw new BundleError("The file has no startedAt and endedAt");
  if (typeof complete !== "boolean" || !isStop(stop)) {
    throw new BundleError("The file does not say whether the run completed and why it stopped");
  }
  if (!isStrings(sweeps) || !isTrial(trial) || !isPageFacts(page)) {
    throw new BundleError("The file has no sweeps, trial and page facts");
  }
  if (!Array.isArray(records)) throw new BundleError("The file has no records");
  const checked: WorkbenchRecord[] = [];
  for (const [index, record] of records.entries()) {
    if (!isSweepRecord(record)) {
      throw new BundleError(
        `Record ${index + 1} of the file: ${recordProblem(record) ?? "unreadable"}`,
      );
    }
    checked.push(record);
  }
  return {
    format: BUNDLE_FORMAT,
    startedAt,
    endedAt,
    complete,
    stop,
    sweeps,
    trial,
    page,
    records: checked,
  };
};

/** A record of a bundle as the workbench would have saved it. */
export type RecordFile = {
  /** Repository-relative (with `/`): `outputs/workbench/<host>-<adapter name>/<entry id>_<stamp>.json`. */
  readonly path: string;
  /** The record's text, byte for byte the workbench's. */
  readonly text: string;
};

/**
 * Every record of a bundle as a file of the workbench: its path by the workbench's rule, stamped with
 * the moment the record was made (`recordedAt`), and its text by the shared record code.
 */
export const bundleRecordFiles = (bundle: Bundle): RecordFile[] =>
  bundle.records.map((record) => ({
    path: recordPath(
      {
        entryId: record.entry.id,
        host: record.environment.host,
        adapter: record.environment.adapter,
      },
      utcStamp(new Date(record.recordedAt)),
    ),
    text: recordText(record),
  }));
