/**
 * The geometry sweep entry's run: one `runGeometrySweep` call per planned case through the loop the
 * sweep entries share (`../per-case-sweep.ts` — idle time between cases, cancel, device loss), and
 * the live view folded from its events. The sweep and the idle wait are injected, so the order, the
 * cancel points and the fold are tested without a GPU.
 */

import {
  type GeometrySweepOptions,
  type GeometrySweepProgress,
  type GeometrySweepReport,
  type GeometrySweepTimingUnit,
  runGeometrySweep,
} from "@karume/runtime/tune";

import {
  foldPerCaseEvent,
  idle,
  INITIAL_PER_CASE_LIVE,
  type PerCaseEvent,
  type PerCaseLive,
  type PerCaseOutcome,
  type PerCaseStop,
  runPerCase,
} from "../per-case-sweep.ts";
import type { SweepPlan } from "./settings.ts";

export { idle };

/** What the run reports while it goes. */
export type SweepRunEvent = PerCaseEvent<GeometrySweepProgress>;

/** Why the run ended. */
export type SweepStop = PerCaseStop;

/** The per-case records in run order (also the ones before a cancel, a device loss or an error). */
export type SweepRunOutcome = PerCaseOutcome<GeometrySweepReport>;

export type SweepRunDeps = {
  /** The WebGPU implementation every call acquires its device from (default `navigator.gpu`). */
  readonly gpu?: GPU;
  readonly signal: AbortSignal;
  readonly onEvent: (event: SweepRunEvent) => void;
  /** Default `runGeometrySweep`. */
  readonly sweep?: (options: GeometrySweepOptions) => Promise<GeometrySweepReport>;
  /** Waits `ms`, rejecting with the signal's reason on abort. Default {@link idle}. */
  readonly wait?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Default `performance.now`. */
  readonly now?: () => number;
};

/**
 * Sweeps the plan's cases one `runGeometrySweep` call each with the plan's candidate set, op
 * families, geometries, quick-look margin and rounds, idling before every case but the first (`runPerCase` — its cancel and
 * device-loss rules). Never rejects: the records so far come back with the reason the run ended.
 */
export const runSweepPlan = (plan: SweepPlan, deps: SweepRunDeps): Promise<SweepRunOutcome> => {
  const sweep = deps.sweep ?? runGeometrySweep;
  return runPerCase<GeometrySweepReport, GeometrySweepProgress>(plan.caseIds, plan.idleSeconds, {
    signal: deps.signal,
    onEvent: deps.onEvent,
    sweep: (caseId, signal, onProgress) =>
      sweep({
        gpu: deps.gpu,
        candidateSet: plan.candidateSet,
        ops: plan.ops,
        cases: [caseId],
        ...(plan.geometries.length === 0 ? {} : { geometries: plan.geometries }),
        quickLookMargin: plan.quickLookMargin,
        rounds: plan.rounds,
        signal,
        onProgress,
      }),
    ...(deps.wait === undefined ? {} : { wait: deps.wait }),
    ...(deps.now === undefined ? {} : { now: deps.now }),
  });
};

/** The live view of a run, with the timing the first sweep started with. */
export type SweepLive = PerCaseLive & {
  /** From the first sweep's `started` event. */
  readonly unit?: GeometrySweepTimingUnit;
  readonly dp4a?: boolean;
};

/** Before the first case starts. */
export const INITIAL_LIVE: SweepLive = INITIAL_PER_CASE_LIVE;

/** The live view after one event. */
export const foldSweepEvent = (live: SweepLive, event: SweepRunEvent): SweepLive =>
  foldPerCaseEvent(live, event, (current, started) =>
    started.kind === "started" ? { ...current, unit: started.unit, dp4a: started.dp4a } : current,
  );
