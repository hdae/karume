/**
 * The geometry sweep page's run: one `runGeometrySweep` call per planned case with idle time
 * between them (cooling — a case takes minutes, and heat moves the default's re-measurement), and
 * the live view folded from its events. The sweep and the idle wait are injected, so the order,
 * the cancel points and the fold are tested without a GPU.
 */

import {
  type GeometrySweepCaseSummary,
  type GeometrySweepOptions,
  type GeometrySweepProgress,
  type GeometrySweepReport,
  type GeometrySweepRow,
  type GeometrySweepTimingUnit,
  runGeometrySweep,
} from "@karume/runtime/tune";

import type { SweepPlan } from "./settings";

/** What the run reports while it goes. */
export type SweepRunEvent =
  /** Case `index` of the plan starts (its sweep acquires its own device). */
  | { readonly kind: "case"; readonly index: number; readonly caseId: string }
  /** Idle before case `index`, until `until` (`performance.now()` milliseconds). */
  | { readonly kind: "idle"; readonly index: number; readonly until: number }
  /** An event of the current case's sweep. */
  | { readonly kind: "sweep"; readonly progress: GeometrySweepProgress };

/** Why the run ended. */
export type SweepStop =
  | { readonly kind: "completed" }
  | { readonly kind: "cancelled" }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string }
  | { readonly kind: "error"; readonly error: unknown };

/** The per-case records in run order (also the ones before a cancel, a device loss or an error). */
export type SweepRunOutcome = {
  readonly reports: readonly GeometrySweepReport[];
  readonly stop: SweepStop;
};

export type SweepRunDeps = {
  readonly signal: AbortSignal;
  readonly onEvent: (event: SweepRunEvent) => void;
  /** Default `runGeometrySweep`. */
  readonly sweep?: (options: GeometrySweepOptions) => Promise<GeometrySweepReport>;
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
 * Sweeps the plan's cases one call each, idling before every case but the first. Never rejects:
 * the records so far come back with the reason the run ended.
 *
 * Cancel points: before a case starts (checked before its sweep acquires a device, so a run whose
 * signal was aborted while it waited for the app's GPU queue acquires nothing), during the idle
 * wait (at once), and inside a case where `runGeometrySweep` takes its signal (once the current
 * geometry is measured — that record is kept, with `aborted: true`). `runGeometrySweep` checks no
 * signal while it acquires its device: a cancel then waits for the device, and the record it
 * returns without a row is dropped (it measured nothing). An adapter or device request that never
 * settles therefore holds the run — and with it the app's GPU queue (`runOnGpuQueue`) — for good:
 * neither a cancel nor leaving the page ends it, and only reloading the tab recovers. A cancel that
 * lands after the last case has measured everything changes nothing: the run is completed. A device
 * loss ends the run after the case's record; the next case would acquire a new device, but the loss
 * means the machine's state changed mid-sweep.
 *
 * Only the abort itself — a rejection with the signal's reason, as the idle wait's — ends the run
 * as cancelled. Anything else a sweep throws is the run's error, also when the signal was aborted
 * meanwhile, so a failure is never hidden behind a cancel.
 */
export const runSweepPlan = async (
  plan: SweepPlan,
  deps: SweepRunDeps,
): Promise<SweepRunOutcome> => {
  const { signal, onEvent } = deps;
  const sweep = deps.sweep ?? runGeometrySweep;
  const wait = deps.wait ?? idle;
  const now = deps.now ?? (() => performance.now());
  const reports: GeometrySweepReport[] = [];
  try {
    for (const [index, caseId] of plan.caseIds.entries()) {
      if (index > 0 && plan.idleSeconds > 0) {
        const ms = plan.idleSeconds * 1000;
        onEvent({ kind: "idle", index, until: now() + ms });
        await wait(ms, signal);
      }
      if (signal.aborted) return { reports, stop: { kind: "cancelled" } };
      onEvent({ kind: "case", index, caseId });
      const report = await sweep({
        candidateSet: plan.candidateSet,
        ops: plan.ops,
        cases: [caseId],
        rounds: plan.rounds,
        signal,
        onProgress: (progress) => onEvent({ kind: "sweep", progress }),
      });
      const measuredNothing = report.rows.length === 0 && report.deviceLost === null;
      if (!(signal.aborted && measuredNothing)) reports.push(report);
      if (report.deviceLost !== null) {
        return { reports, stop: { kind: "deviceLost", ...report.deviceLost } };
      }
      const last = index === plan.caseIds.length - 1;
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

/** The live view of a run. */
export type SweepLive = {
  /** The case that runs now (or ran last), as an index into the plan. */
  readonly caseIndex: number;
  readonly phase:
    | { readonly kind: "waiting" }
    | { readonly kind: "idle"; readonly until: number }
    | { readonly kind: "sweeping"; readonly status?: string };
  /** From the first sweep's `started` event. */
  readonly unit?: GeometrySweepTimingUnit;
  readonly dp4a?: boolean;
  /** Every row and every finished case so far, over all calls. */
  readonly rows: readonly GeometrySweepRow[];
  readonly cases: readonly GeometrySweepCaseSummary[];
  readonly deviceLost?: { readonly reason: string; readonly message: string };
};

/** Before the first case: waiting for the app's GPU queue (the previous page's device). */
export const INITIAL_LIVE: SweepLive = {
  caseIndex: 0,
  phase: { kind: "waiting" },
  rows: [],
  cases: [],
};

/** The live view after one event. */
export const foldSweepEvent = (live: SweepLive, event: SweepRunEvent): SweepLive => {
  switch (event.kind) {
    case "case":
      return { ...live, caseIndex: event.index, phase: { kind: "sweeping" } };
    case "idle":
      return { ...live, caseIndex: event.index, phase: { kind: "idle", until: event.until } };
    case "sweep": {
      const { progress } = event;
      switch (progress.kind) {
        case "started":
          return {
            ...live,
            unit: progress.unit,
            dp4a: progress.dp4a,
          };
        case "status":
          return { ...live, phase: { kind: "sweeping", status: progress.message } };
        case "row":
          return { ...live, rows: [...live.rows, progress.row] };
        case "case":
          return { ...live, cases: [...live.cases, progress.summary] };
        case "deviceLost":
          return {
            ...live,
            deviceLost: { reason: progress.reason, message: progress.message },
          };
      }
    }
  }
};
