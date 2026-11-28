/**
 * A run of the benchmark: the planned sweeps one after another, each one workbench record made by
 * the shared code (its per-case loop with idle time, cancel and device-loss rules, its merged record
 * and summary, the record's envelope), with the idle time before a later sweep's first case too,
 * then the bundle of those records. A run is complete only when every planned sweep finished and its
 * signal never aborted; a complete run downloads the bundle at once. A run a cancel, a device loss or
 * an error stopped keeps what it measured and leaves the download to the person (the page's button),
 * and so does a complete run whose download threw. Everything that touches the GPU, the clock or the
 * page is injected, so the run is tested against fake sweeps without a GPU.
 *
 * The events are only handed on (`onEvent`): the page stores them and paints on its own frame, as the
 * sweeps' callbacks run between timed passes.
 *
 * SPEC: docs/spec/benchmark.md#while-it-runs
 */

import {
  type GemvSweepOptions,
  type GemvSweepProgress,
  type GemvSweepReport,
  type GemvSweepSession,
  type GeometrySweepOptions,
  type GeometrySweepProgress,
  type GeometrySweepReport,
  runGemvSweep,
  runGeometrySweep,
} from "@karume/runtime/tune";
import { runGemvPlan } from "@karume/sweep-app-support/gemv-sweep/run";
import { gemvSweepResult } from "@karume/sweep-app-support/gemv-sweep/summary";
import { runSweepPlan } from "@karume/sweep-app-support/geometry-sweep/run";
import { geometrySweepResult } from "@karume/sweep-app-support/geometry-sweep/summary";
import {
  idle,
  type PerCaseEvent,
  type PerCaseStop,
} from "@karume/sweep-app-support/per-case-sweep";
import {
  type Checkout,
  errorText,
  messageText,
  type RecordEnvironment,
  recordRun,
  type WorkbenchRecord,
} from "@karume/sweep-app-support/record";

import {
  type Bundle,
  BUNDLE_FORMAT,
  bundleFileName,
  type BundleStop,
  bundleText,
  type PageFacts,
} from "./bundle.ts";
import type { BenchmarkPlan, PlannedSweep, SweepId } from "./plan.ts";

/** What a run reports, its case indices counted over the whole run. */
export type BenchmarkEvent =
  /** A sweep starts; `first` is the run index of its first case. */
  | { readonly kind: "sweep"; readonly sweep: SweepId; readonly first: number }
  /** Idle before the run's case `index`, until `until` (`performance.now()` milliseconds). */
  | { readonly kind: "idle"; readonly index: number; readonly until: number }
  /** The run's case `index` starts (its sweep acquires a device of its own). */
  | { readonly kind: "case"; readonly index: number }
  | { readonly kind: "progress"; readonly sweep: "gemv"; readonly progress: GemvSweepProgress }
  /** A geometry sweep's event, of its `explore` set (`geometry`) or its `grid` set (`grid`). */
  | {
      readonly kind: "progress";
      readonly sweep: "geometry" | "grid";
      readonly progress: GeometrySweepProgress;
    }
  /**
   * The case's sweep call returned its per-case record (the one the sweep's record merges; a cancel
   * or a device loss may have cut it short).
   */
  | { readonly kind: "record"; readonly sweep: "gemv"; readonly report: GemvSweepReport }
  | {
      readonly kind: "record";
      readonly sweep: "geometry" | "grid";
      readonly report: GeometrySweepReport;
    };

/** An event and the moment it came (`performance.now()` milliseconds). */
export type StampedEvent = { readonly at: number; readonly event: BenchmarkEvent };

export type BenchmarkDeps = {
  readonly signal: AbortSignal;
  /** Called with every event as it comes; MUST only store it (the sweeps call it between timed passes). */
  readonly onEvent: (event: StampedEvent) => void;
  /** The record's environment, read from an adapter the page requests itself (once per sweep). */
  readonly readEnvironment: () => Promise<RecordEnvironment>;
  /** The checkout the page was built from. */
  readonly checkout: Checkout;
  /** Hands the bundle's text to the person as a file. */
  readonly download: (fileName: string, text: string) => void;
  /** What the page saw of itself, read when the bundle is made. */
  readonly pageFacts: () => PageFacts;
  /** The WebGPU implementation (default `navigator.gpu`). */
  readonly gpu?: GPU;
  /**
   * The page load's session, made when the page loaded: every GEMV record of its runs names it, so
   * two runs in one page load are told from runs in two.
   */
  readonly session?: GemvSweepSession;
  /** One GEMV sweep call (one case). Default `runGemvSweep`. */
  readonly gemvSweep?: (options: GemvSweepOptions) => Promise<GemvSweepReport>;
  /** One geometry sweep call (one case). Default `runGeometrySweep`. */
  readonly geometrySweep?: (options: GeometrySweepOptions) => Promise<GeometrySweepReport>;
  /** The idle wait (rejects with the signal's reason on abort). Default: the shared one. */
  readonly wait?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Default `performance.now`. */
  readonly now?: () => number;
  /** Default `new Date()`. */
  readonly clock?: () => Date;
};

/** How a run ended: its bundle and whether the page downloaded it already. */
export type RunEnding = {
  readonly bundle: Bundle;
  readonly text: string;
  readonly fileName: string;
  /** The run is complete and the bundle was downloaded; otherwise the page offers it. */
  readonly downloaded: boolean;
  /** Why the download of a complete run failed (what it threw, in one line). */
  readonly downloadError?: string;
  /** Some record holds measurements (a run stopped before any case finished holds none). */
  readonly measured: boolean;
};

const stopOf = (stop: PerCaseStop): BundleStop => {
  switch (stop.kind) {
    case "completed":
    case "cancelled":
      return { kind: stop.kind };
    case "deviceLost":
      return { kind: "deviceLost", reason: stop.reason, message: stop.message };
    case "error":
      return { kind: "error", message: errorText(stop.error) };
  }
};

/**
 * Runs `plan` and returns its ending. A sweep that stopped short ends the run after its record, and
 * so does a sweep whose result could not be made (the record's `error`); a download that throws
 * leaves the bundle to the page's button. It rejects only when the bundle itself cannot be made
 * (`pageFacts` or the bundle's text threw), which leaves no file to offer.
 */
export const runBenchmark = async (
  plan: BenchmarkPlan,
  deps: BenchmarkDeps,
): Promise<RunEnding> => {
  const { signal } = deps;
  const now = deps.now ?? (() => performance.now());
  const clock = deps.clock ?? (() => new Date());
  const wait = deps.wait ?? idle;
  const gemvSweep = deps.gemvSweep ?? runGemvSweep;
  const geometrySweep = deps.geometrySweep ?? runGeometrySweep;
  const emit = (event: BenchmarkEvent): void => deps.onEvent({ at: now(), event });
  const startedAt = clock();
  const records: WorkbenchRecord[] = [];
  let stop: BundleStop = { kind: "completed" };
  let first = 0;

  /** The per-case loop's events, their indices counted over the run. */
  const relay =
    <P>(progress: (value: P) => BenchmarkEvent) =>
    (event: PerCaseEvent<P>): void => {
      switch (event.kind) {
        case "case":
          emit({ kind: "case", index: first + event.index });
          return;
        case "idle":
          emit({ kind: "idle", index: first + event.index, until: event.until });
          return;
        case "sweep":
          emit(progress(event.progress));
      }
    };

  /** One sweep's record, and how its loop stopped. */
  const sweepRecord = async (
    sweep: PlannedSweep,
    environment: RecordEnvironment,
  ): Promise<{ readonly record: WorkbenchRecord; readonly stop: PerCaseStop | undefined }> => {
    let loopStop: PerCaseStop | undefined;
    const common = {
      ...(deps.gpu === undefined ? {} : { gpu: deps.gpu }),
      signal,
      wait,
      now,
    };
    const record = await recordRun(
      {
        entry: { id: sweep.entryId, parameters: sweep.parameters },
        environment,
        checkout: deps.checkout,
        signal,
      },
      async () => {
        if (sweep.sweep === "gemv") {
          const outcome = await runGemvPlan(sweep.plan, {
            ...common,
            ...(deps.session === undefined ? {} : { session: deps.session }),
            sweep: async (options) => {
              const report = await gemvSweep(options);
              emit({ kind: "record", sweep: "gemv", report });
              return report;
            },
            onEvent: relay((progress: GemvSweepProgress) => ({
              kind: "progress",
              sweep: "gemv",
              progress,
            })),
          });
          loopStop = outcome.stop;
          return gemvSweepResult(sweep.plan, outcome);
        }
        const set = sweep.sweep;
        const outcome = await runSweepPlan(sweep.plan, {
          ...common,
          sweep: async (options) => {
            const report = await geometrySweep(options);
            emit({ kind: "record", sweep: set, report });
            return report;
          },
          onEvent: relay((progress: GeometrySweepProgress) => ({
            kind: "progress",
            sweep: set,
            progress,
          })),
        });
        loopStop = outcome.stop;
        return geometrySweepResult(sweep.plan, outcome);
      },
    );
    return { record, stop: loopStop };
  };

  for (const [position, sweep] of plan.sweeps.entries()) {
    try {
      if (position > 0 && plan.idleSeconds > 0) {
        const ms = plan.idleSeconds * 1000;
        emit({ kind: "idle", index: first, until: now() + ms });
        await wait(ms, signal);
      }
      if (signal.aborted) {
        stop = { kind: "cancelled" };
        break;
      }
      emit({ kind: "sweep", sweep: sweep.sweep, first });
      const made = await sweepRecord(sweep, await deps.readEnvironment());
      records.push(made.record);
      stop =
        made.record.outcome === "error"
          ? { kind: "error", message: made.record.error ?? "the sweep's result could not be made" }
          : made.stop === undefined
            ? { kind: "completed" }
            : stopOf(made.stop);
    } catch (error) {
      stop =
        signal.aborted && error === signal.reason
          ? { kind: "cancelled" }
          : { kind: "error", message: errorText(error) };
    }
    if (stop.kind !== "completed") break;
    first += sweep.plan.caseIds.length;
  }
  // One rule for a complete run: every planned sweep finished and the signal never aborted. A
  // cancel that lands while a sweep's last case re-measures (where its sweep takes no signal) leaves
  // the loop `completed`, but the sweep's record `cancelled` (`recordRun`)
  if (stop.kind === "completed" && signal.aborted) stop = { kind: "cancelled" };
  const complete = stop.kind === "completed" && records.length === plan.sweeps.length;

  const bundle: Bundle = {
    format: BUNDLE_FORMAT,
    startedAt: startedAt.toISOString(),
    endedAt: clock().toISOString(),
    complete,
    stop,
    sweeps: plan.sweeps.map(({ entryId }) => entryId),
    trial: plan.trial,
    page: deps.pageFacts(),
    records,
  };
  const text = bundleText(bundle);
  const fileName = bundleFileName(bundle);
  let downloadError: string | undefined;
  if (complete) {
    try {
      deps.download(fileName, text);
    } catch (error) {
      downloadError = messageText(error);
    }
  }
  return {
    bundle,
    text,
    fileName,
    downloaded: complete && downloadError === undefined,
    ...(downloadError === undefined ? {} : { downloadError }),
    measured: records.some((record) => record.data !== undefined),
  };
};
