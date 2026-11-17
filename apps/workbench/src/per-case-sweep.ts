/**
 * What the sweep entries (`geometry-sweep`, `gemv-sweep`) share: one sweep call per planned case with
 * idle time before every case but the first (cooling — heat moves the re-measured default or
 * baseline a record's ratios rest on), the cancel and device-loss rules of that loop, the live view
 * folded from its events, the checks and fields of merging the per-case records into one, the verdict
 * order, and the summary's text helpers. Each entry supplies its sweep call and its record's own
 * fields; the loop is tested without a GPU through the injected sweep and idle wait.
 */

import type { GeometrySweepTimingUnit } from "@karume/runtime/tune";

import type { Verdict } from "./entry.ts";
import { messageText } from "./record.ts";

/** What the loop reports while it goes. */
export type PerCaseEvent<P> =
  /** Case `index` of the plan starts (its sweep acquires its own device). */
  | { readonly kind: "case"; readonly index: number; readonly caseId: string }
  /** Idle before case `index`, until `until` (`performance.now()` milliseconds). */
  | { readonly kind: "idle"; readonly index: number; readonly until: number }
  /** An event of the current case's sweep. */
  | { readonly kind: "sweep"; readonly progress: P };

/** Why the loop ended. */
export type PerCaseStop =
  | { readonly kind: "completed" }
  | { readonly kind: "cancelled" }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string }
  | { readonly kind: "error"; readonly error: unknown };

/** The per-case records in run order (also the ones before a cancel, a device loss or an error). */
export type PerCaseOutcome<R> = {
  readonly reports: readonly R[];
  readonly stop: PerCaseStop;
};

/** What the loop, the merge and the verdict read of a per-case record. */
export type PerCaseRecord<Row = unknown> = {
  readonly rows: readonly Row[];
  /** A cancel stopped the case midway. */
  readonly aborted: boolean;
  readonly deviceLost: { readonly reason: string; readonly message: string } | null;
};

export type PerCaseDeps<R, P> = {
  readonly signal: AbortSignal;
  readonly onEvent: (event: PerCaseEvent<P>) => void;
  /** Sweeps one case on a device of its own. */
  readonly sweep: (
    caseId: string,
    signal: AbortSignal,
    onProgress: (progress: P) => void,
  ) => Promise<R>;
  /** Waits `ms`, rejecting with the signal's reason on abort. Default {@link idle}. */
  readonly wait?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Default `performance.now`. */
  readonly now?: () => number;
};

/** `setTimeout` as a promise that an abort ends at once (with the signal's reason). */
export const idle = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });

/**
 * Sweeps the cases one call each, idling before every case but the first. Never rejects: the records
 * so far come back with the reason the run ended.
 *
 * Cancel points: before a case starts (checked before its sweep acquires a device, so a run whose
 * signal was aborted before it started acquires nothing), during the idle wait (at once), and inside
 * a case where the sweep takes its signal (once the current measurement is done — that record is
 * kept, with `aborted: true`). A sweep checks no signal while it acquires its device: a cancel then
 * waits for the device, and the record it returns without a row is dropped (it measured nothing). An
 * adapter or device request that never settles therefore holds the run for good: a cancel does not
 * end it, only reloading the page or ending the process. A cancel that lands after the last case has
 * measured everything changes nothing: the run is completed. A device loss ends the run after the
 * case's record; the next case would acquire a new device, but the loss means the machine's state
 * changed mid-sweep.
 *
 * Only the abort itself — a rejection with the signal's reason, as the idle wait's — ends the run as
 * cancelled. Anything else a sweep throws is the run's error, also when the signal was aborted
 * meanwhile, so a failure is never hidden behind a cancel.
 */
export const runPerCase = async <R extends PerCaseRecord, P>(
  caseIds: readonly string[],
  idleSeconds: number,
  deps: PerCaseDeps<R, P>,
): Promise<PerCaseOutcome<R>> => {
  const { signal, onEvent } = deps;
  const wait = deps.wait ?? idle;
  const now = deps.now ?? (() => performance.now());
  const reports: R[] = [];
  try {
    for (const [index, caseId] of caseIds.entries()) {
      if (index > 0 && idleSeconds > 0) {
        const ms = idleSeconds * 1000;
        onEvent({ kind: "idle", index, until: now() + ms });
        await wait(ms, signal);
      }
      if (signal.aborted) return { reports, stop: { kind: "cancelled" } };
      onEvent({ kind: "case", index, caseId });
      const report = await deps.sweep(caseId, signal, (progress) =>
        onEvent({ kind: "sweep", progress }),
      );
      const measuredNothing = report.rows.length === 0 && report.deviceLost === null;
      if (!(signal.aborted && measuredNothing)) reports.push(report);
      if (report.deviceLost !== null) {
        return { reports, stop: { kind: "deviceLost", ...report.deviceLost } };
      }
      const last = index === caseIds.length - 1;
      if (signal.aborted && (report.aborted || !last)) {
        return { reports, stop: { kind: "cancelled" } };
      }
    }
    return { reports, stop: { kind: "completed" } };
  } catch (error) {
    const cancelled = signal.aborted && error === signal.reason;
    return {
      reports,
      stop: cancelled ? { kind: "cancelled" } : { kind: "error", error },
    };
  }
};

/** The progress both sweeps report; the payload of `started` is each sweep's own. */
export type PerCaseProgress =
  | { readonly kind: "started" }
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "row" }
  | { readonly kind: "case" }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

/** The live view of a run. */
export type PerCaseLive = {
  /** The case that runs now (or ran last), as an index into the plan. */
  readonly caseIndex: number;
  readonly phase:
    | { readonly kind: "waiting" }
    | { readonly kind: "idle"; readonly until: number }
    | { readonly kind: "sweeping"; readonly status?: string };
  /** The cases finished so far, over all calls. */
  readonly casesDone: number;
  /** The rows of the case that runs now (each call sweeps one case). */
  readonly caseRows: number;
  readonly deviceLost?: { readonly reason: string; readonly message: string };
};

/** Before the first case starts. */
export const INITIAL_PER_CASE_LIVE: PerCaseLive = {
  caseIndex: 0,
  phase: { kind: "waiting" },
  casesDone: 0,
  caseRows: 0,
};

/** The live view after one event; a sweep's `started` event goes to `onStarted`. */
export const foldPerCaseEvent = <L extends PerCaseLive, P extends PerCaseProgress>(
  live: L,
  event: PerCaseEvent<P>,
  onStarted: (live: L, started: P) => L,
): L => {
  switch (event.kind) {
    case "case":
      return { ...live, caseIndex: event.index, phase: { kind: "sweeping" }, caseRows: 0 };
    case "idle":
      return { ...live, caseIndex: event.index, phase: { kind: "idle", until: event.until } };
    case "sweep": {
      const progress: PerCaseProgress = event.progress;
      switch (progress.kind) {
        case "started":
          return onStarted(live, event.progress);
        case "status":
          return { ...live, phase: { kind: "sweeping", status: progress.message } };
        case "row":
          return { ...live, caseRows: live.caseRows + 1 };
        case "case":
          return { ...live, casesDone: live.casesDone + 1 };
        case "deviceLost":
          return { ...live, deviceLost: { reason: progress.reason, message: progress.message } };
      }
    }
  }
};

/**
 * Shows the live line after every event, and once a second while the run idles (the idle wait
 * reports nothing for its whole length: the ticker counts it down). `stop` clears the ticker.
 */
export const liveLine = (
  show: () => void,
): { readonly update: (idling: boolean) => void; readonly stop: () => void } => {
  let ticker: ReturnType<typeof setInterval> | undefined;
  return {
    update: (idling) => {
      if (idling) {
        ticker ??= setInterval(show, 1000);
      } else {
        clearInterval(ticker);
        ticker = undefined;
      }
      show();
    },
    stop: () => clearInterval(ticker),
  };
};

/** ns (GPU time, or wall clock submit → completion) as ms; raw Deno ticks as they are. */
export const formatPerDispatch = (
  value: number | undefined,
  unit: GeometrySweepTimingUnit | undefined,
): string =>
  value === undefined
    ? ""
    : unit === "ns"
      ? `${(value / 1e6).toFixed(3)} ms`
      : unit === "wall"
        ? `${(value / 1e6).toFixed(3)} ms wall`
        : `${value.toFixed(0)} ${unit ?? ""}`;

/** A ratio as `×1.234`, or empty. */
export const ratioText = (value: number | undefined): string =>
  value === undefined ? "" : `×${value.toFixed(3)}`;

/** The non-empty parts joined with ` · `. */
export const joined = (parts: readonly (string | undefined)[]): string =>
  parts.filter((part) => part !== undefined && part !== "").join(" · ");

/**
 * The live status as one line: the cases done, then the case being swept (with its rows so far, the
 * sweep's own status and `detail` — the timing once the sweep started) or the seconds left of the
 * idle wait before the next case.
 */
export const perCaseProgressLine = (
  caseIds: readonly string[],
  live: PerCaseLive,
  now: number,
  detail: string | undefined,
): string => {
  const total = caseIds.length;
  const position = `case ${live.caseIndex + 1} / ${total}`;
  const done = `${live.casesDone} / ${total} cases done`;
  if (live.deviceLost !== undefined) {
    return `device lost (${live.deviceLost.reason}): ${live.deviceLost.message} — the sweep stops after ${position}`;
  }
  const { phase } = live;
  switch (phase.kind) {
    case "waiting":
      return `starting ${total} case${total === 1 ? "" : "s"}`;
    case "idle":
      return `${done} · idle before ${position}: ${Math.max(0, (phase.until - now) / 1000).toFixed(0)} s left`;
    case "sweeping":
      return joined([
        done,
        `${position} ${caseIds[live.caseIndex] ?? ""}`,
        `${live.caseRows} rows`,
        phase.status ?? "acquiring the device",
        detail,
      ]);
  }
};

/** Why the run stopped, or nothing when it completed. */
export const stopText = (stop: PerCaseStop): string | undefined => {
  switch (stop.kind) {
    case "completed":
      return undefined;
    case "cancelled":
      return "stopped: cancelled";
    case "deviceLost":
      return `stopped: device lost (${stop.reason}): ${stop.message}`;
    case "error":
      return `stopped: ${messageText(stop.error)}`;
  }
};

/** A record a cancel or a device loss cut short (its rows so far are kept; it is not a finished case). */
export const cutShort = (report: PerCaseRecord): boolean =>
  report.aborted || report.deviceLost !== null;

/**
 * A run's verdict, checked in this order: `fail` when a row of a finished case is `wrong` (a wrong
 * result, whatever else happened); `incomplete` when a device loss or an error stopped the run short
 * of its plan (rows that errored in such a run are that stop's consequence); `fail` when a row of a
 * finished case `errored`; none when the run was cancelled (the record says `cancelled`); otherwise
 * `pass`. A finished case is one whose record neither a cancel nor a device loss cut short. Timing is
 * not part of it (it depends on the machine).
 */
export const perCaseVerdict = <Row>(
  reports: readonly PerCaseRecord<Row>[],
  stop: PerCaseStop,
  rows: { readonly wrong: (row: Row) => boolean; readonly errored: (row: Row) => boolean },
): Verdict | undefined => {
  const finishedRows = reports.filter((report) => !cutShort(report)).flatMap(({ rows }) => rows);
  if (finishedRows.some(rows.wrong)) return "fail";
  if (stop.kind === "deviceLost" || stop.kind === "error") return "incomplete";
  if (finishedRows.some(rows.errored)) return "fail";
  if (stop.kind === "cancelled") return undefined;
  return "pass";
};

export type MergeOptions = {
  /** Every case the sweep was planned to run, in order (`settings.cases` of the merged record). */
  readonly plannedCaseIds: readonly string[];
  /** Cancel stopped the sweep before every planned case had a record. */
  readonly cancelledBetweenCases: boolean;
};

/** What the merge checks and reads of a per-case record besides its rows. */
export type MergeableRecord = PerCaseRecord & {
  readonly startedAt: string;
  readonly date: string;
  readonly settings: { readonly cases?: readonly string[] };
};

/** The one case a per-case record swept (`settings.cases` of a call with `cases: [id]`). */
const caseOf = (report: MergeableRecord, index: number): string => {
  const ids = report.settings.cases ?? [];
  if (ids.length !== 1) {
    throw new Error(
      `Record ${index} is not a per-case record (settings.cases is ${JSON.stringify(ids)})`,
    );
  }
  return ids[0];
};

/**
 * Checks that per-case records belong to one sweep and returns the fields every merged record takes
 * from them: `startedAt` of the first and `date` of the last; `aborted` when Cancel stopped a case
 * midway or the sweep between cases (a sweep stopped by a device loss or an error has
 * `aborted: false` and fewer cases than planned, as `aborted` means a cancel); `deviceLost`, the
 * first loss.
 *
 * Throws when there is no record, when a record's `shared` fields differ from the first's (named by
 * `sharedText`), when a record is not per-case, or when a case is unplanned or comes twice.
 */
export const mergePerCase = <R extends MergeableRecord>(
  reports: readonly R[],
  options: MergeOptions,
  shared: (report: R) => unknown,
  sharedText: string,
): {
  readonly first: R;
  readonly startedAt: string;
  readonly date: string;
  readonly aborted: boolean;
  readonly deviceLost: R["deviceLost"];
} => {
  const [first] = reports;
  const last = reports.at(-1);
  if (first === undefined || last === undefined) throw new Error("No per-case record to merge");
  const expected = JSON.stringify(shared(first));
  const seen = new Set<string>();
  reports.forEach((report, index) => {
    if (JSON.stringify(shared(report)) !== expected) {
      throw new Error(`Record ${index} differs from record 0 in ${sharedText}`);
    }
    const caseId = caseOf(report, index);
    if (!options.plannedCaseIds.includes(caseId)) {
      throw new Error(`Record ${index} swept ${caseId}, which is not a planned case`);
    }
    if (seen.has(caseId)) throw new Error(`Case ${caseId} has two records`);
    seen.add(caseId);
  });
  return {
    first,
    startedAt: first.startedAt,
    date: last.date,
    aborted: options.cancelledBetweenCases || reports.some((report) => report.aborted),
    deviceLost: reports.find((report) => report.deviceLost !== null)?.deviceLost ?? null,
  };
};
