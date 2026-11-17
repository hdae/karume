/**
 * The GEMV sweep entry's run: one `runGemvSweep` call per planned case through the loop the sweep
 * entries share (`../per-case-sweep.ts` — idle time between cases, cancel, device loss), and the live
 * view folded from its events. The sweep and the idle wait are injected, so the calls are tested
 * without a GPU.
 */

import {
  type GemvSweepOptions,
  type GemvSweepProgress,
  type GemvSweepReport,
  type GeometrySweepTimingUnit,
  runGemvSweep,
} from "@karume/runtime/tune";

import {
  foldPerCaseEvent,
  INITIAL_PER_CASE_LIVE,
  type PerCaseEvent,
  type PerCaseLive,
  type PerCaseOutcome,
  runPerCase,
} from "../per-case-sweep.ts";
import type { GemvSweepPlan } from "./settings.ts";

/** What the run reports while it goes. */
export type GemvRunEvent = PerCaseEvent<GemvSweepProgress>;

/** The per-case records in run order and why the run ended. */
export type GemvRunOutcome = PerCaseOutcome<GemvSweepReport>;

export type GemvRunDeps = {
  /** The WebGPU implementation every call acquires its device from (default `navigator.gpu`). */
  readonly gpu?: GPU;
  readonly signal: AbortSignal;
  readonly onEvent: (event: GemvRunEvent) => void;
  /** Default `runGemvSweep`. */
  readonly sweep?: (options: GemvSweepOptions) => Promise<GemvSweepReport>;
  /** Waits `ms`, rejecting with the signal's reason on abort. Default: the shared idle wait. */
  readonly wait?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Default `performance.now`. */
  readonly now?: () => number;
};

/**
 * Sweeps the plan's cases one `runGemvSweep` call each with the plan's candidate set, row counts,
 * candidates, quick-look margin and rounds, idling before every case but the first (`runPerCase` — its cancel and device-loss rules). Never
 * rejects: the records so far come back with the reason the run ended.
 */
export const runGemvPlan = (plan: GemvSweepPlan, deps: GemvRunDeps): Promise<GemvRunOutcome> => {
  const sweep = deps.sweep ?? runGemvSweep;
  return runPerCase<GemvSweepReport, GemvSweepProgress>(plan.caseIds, plan.idleSeconds, {
    signal: deps.signal,
    onEvent: deps.onEvent,
    sweep: (caseId, signal, onProgress) =>
      sweep({
        ...(deps.gpu === undefined ? {} : { gpu: deps.gpu }),
        candidateSet: plan.candidateSet,
        cases: [caseId],
        ...(plan.rows.length === 0 ? {} : { rows: plan.rows }),
        ...(plan.candidates.length === 0 ? {} : { candidates: plan.candidates }),
        quickLookMargin: plan.quickLookMargin,
        rounds: plan.rounds,
        signal,
        onProgress,
      }),
    ...(deps.wait === undefined ? {} : { wait: deps.wait }),
    ...(deps.now === undefined ? {} : { now: deps.now }),
  });
};

/** The live view of a run, with what the first sweep started with. */
export type GemvLive = PerCaseLive & {
  readonly unit?: GeometrySweepTimingUnit;
  /** Whether the subgroup form is measured on this device. */
  readonly subgroups?: boolean;
};

/** Before the first case starts. */
export const INITIAL_GEMV_LIVE: GemvLive = INITIAL_PER_CASE_LIVE;

/** The live view after one event. */
export const foldGemvEvent = (live: GemvLive, event: GemvRunEvent): GemvLive =>
  foldPerCaseEvent(live, event, (current, started) =>
    started.kind === "started"
      ? { ...current, unit: started.unit, subgroups: started.subgroups }
      : current,
  );
