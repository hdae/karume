/**
 * What the page shows of a run while it goes, folded from the run's events: the whole run (how many
 * of its cases are done), the case being measured (which step of how many, from the sweeps' `plan`
 * and `step` events, and what it does now in words), the time left (`estimate.ts`) and the patterns
 * found so far (`patterns.ts`). A case is finished when its sweep call returned a record that neither
 * a cancel nor a device loss cut short (`cutShort`). Pure: the page folds the events it stored and
 * paints the view on its own frame.
 *
 * SPEC: docs/spec/benchmark.md#while-it-runs
 */

import {
  GEMV_LOWEST_MARGIN,
  GEMV_MARGIN,
  GEMV_STEADY_GROUP_SPREAD,
  GEMV_STEADY_MAX_SHARE,
  GEMV_STEADY_MIN_GROUPS,
  type GemvSteadiness,
  type GemvSweepCaseSummary,
  type GemvSweepPhase,
  gemvSteadiness,
  GEOMETRY_MIN_SPEEDUP,
  type GeometrySweepPhase,
  type GeometrySweepReport,
} from "@karume/runtime/tune";
import { cutShort } from "@karume/sweep-app-support/per-case-sweep";

import type { BenchmarkEvent, StampedEvent } from "./controller.ts";
import { durationText, remainingEstimate } from "./estimate.ts";
import {
  type GemvPatternCount,
  gemvPatterns,
  geometryPatterns,
  type PatternCount,
} from "./patterns.ts";
import type { BenchmarkPlan, CaseKind, SweepId } from "./plan.ts";

/** The step a case is at: a phase of its sweep and the units of that phase done of all. */
type Step = {
  readonly phase: GemvSweepPhase | GeometrySweepPhase;
  readonly done: number;
  readonly total: number;
  /** The GEMV row count the step is of. */
  readonly m?: number;
};

/** What the case being measured has said of itself. */
type Current = {
  readonly index: number;
  readonly startedAt: number;
  /** The GEMV case's row counts (its `plan` event). */
  readonly rowCounts?: readonly number[];
  /** The geometry case's planned geometries and the candidates it skips (its `plan` event). */
  readonly geometries?: { readonly planned: number; readonly skipped: number };
  readonly step?: Step;
  /** The sweep's own status line. */
  readonly status?: string;
};

export type ProgressState = {
  readonly current?: Current;
  /** The idle wait before the run's case `index`, from `since` to `until` (ms). */
  readonly idle?: { readonly index: number; readonly since: number; readonly until: number };
  /** The seconds each finished case took, by its run index. */
  readonly finished: ReadonlyMap<number, number>;
  /**
   * The patterns of the finished cases. A GEMV group is one case's, so its count adds up case by
   * case; a geometry class spans cases, so its count is read again over every finished case's record.
   */
  readonly patterns: { readonly gemv: GemvPatternCount; readonly geometry: PatternCount };
  /**
   * The summaries of the finished GEMV cases, which the run's class is read from (the class of the
   * record they merge into: `gemvSteadiness`).
   */
  readonly gemvSummaries: readonly GemvSweepCaseSummary[];
  /** The per-case records of the finished geometry cases. */
  readonly geometryRecords: readonly GeometrySweepReport[];
  readonly deviceLost?: { readonly reason: string; readonly message: string };
};

export const INITIAL_PROGRESS: ProgressState = {
  finished: new Map(),
  patterns: { gemv: { atMargin: 0, atLowest: 0, of: 0 }, geometry: { found: 0, of: 0 } },
  gemvSummaries: [],
  geometryRecords: [],
};

/** A sweep's progress event as the fold reads it, whichever sweep sent it. */
type CaseProgress =
  | { readonly kind: "none" }
  | { readonly kind: "row-counts"; readonly rowCounts: readonly number[] }
  | {
      readonly kind: "geometries";
      readonly geometries: { readonly planned: number; readonly skipped: number };
    }
  | { readonly kind: "step"; readonly step: Step }
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

const caseProgressOf = (event: Extract<BenchmarkEvent, { kind: "progress" }>): CaseProgress => {
  if (event.sweep === "gemv") {
    const { progress } = event;
    switch (progress.kind) {
      case "started":
        return { kind: "none" };
      case "plan":
        return { kind: "row-counts", rowCounts: progress.rowCounts.map(({ m }) => m) };
      case "step":
        return {
          kind: "step",
          step: {
            phase: progress.phase,
            done: progress.done,
            total: progress.total,
            m: progress.m,
          },
        };
      case "status":
        return { kind: "status", message: progress.message };
      // A case's rows and its end are read from the record its call returns
      case "row":
      case "case":
        return { kind: "none" };
      case "deviceLost":
        return { kind: "deviceLost", reason: progress.reason, message: progress.message };
    }
  }
  const { progress } = event;
  switch (progress.kind) {
    case "started":
      return { kind: "none" };
    case "plan":
      return {
        kind: "geometries",
        geometries: { planned: progress.geometries, skipped: progress.skipped },
      };
    case "step":
      return {
        kind: "step",
        step: { phase: progress.phase, done: progress.done, total: progress.total },
      };
    case "status":
      return { kind: "status", message: progress.message };
    case "row":
    case "case":
      return { kind: "none" };
    case "deviceLost":
      return { kind: "deviceLost", reason: progress.reason, message: progress.message };
  }
};

/** The sum of two GEMV counts. */
const plus = (a: GemvPatternCount, b: GemvPatternCount): GemvPatternCount => ({
  atMargin: a.atMargin + b.atMargin,
  atLowest: a.atLowest + b.atLowest,
  of: a.of + b.of,
});

/**
 * The state after the case being measured returned its record at `at`: finished, with its patterns,
 * unless a cancel or a device loss cut the record short (its rows are no finished case's).
 */
const finish = (
  state: ProgressState,
  current: Current,
  at: number,
  record: Extract<BenchmarkEvent, { kind: "record" }>,
): ProgressState => {
  if (cutShort(record.report)) return { ...state, current: undefined };
  const finished = new Map(state.finished).set(current.index, (at - current.startedAt) / 1000);
  if (record.sweep === "gemv") {
    const { summaries, rows, gpuTiming } = record.report;
    return {
      ...state,
      current: undefined,
      finished,
      patterns: {
        ...state.patterns,
        gemv: plus(state.patterns.gemv, gemvPatterns(summaries, rows, gpuTiming.unit)),
      },
      gemvSummaries: [...state.gemvSummaries, ...summaries],
    };
  }
  const geometryRecords = [...state.geometryRecords, record.report];
  return {
    ...state,
    current: undefined,
    finished,
    geometryRecords,
    patterns: { ...state.patterns, geometry: geometryPatterns(geometryRecords) },
  };
};

/** The state after one event. */
export const foldProgress = (state: ProgressState, { at, event }: StampedEvent): ProgressState => {
  switch (event.kind) {
    case "sweep":
      return state;
    case "idle":
      return {
        ...state,
        current: undefined,
        idle: { index: event.index, since: at, until: event.until },
      };
    case "case":
      return {
        ...state,
        idle: undefined,
        current: { index: event.index, startedAt: at },
      };
    case "record":
      return state.current === undefined ? state : finish(state, state.current, at, event);
    case "progress": {
      const progress = caseProgressOf(event);
      if (progress.kind === "deviceLost") {
        return { ...state, deviceLost: { reason: progress.reason, message: progress.message } };
      }
      const { current } = state;
      if (current === undefined) return state;
      switch (progress.kind) {
        case "none":
          return state;
        case "row-counts":
          return { ...state, current: { ...current, rowCounts: progress.rowCounts } };
        case "geometries":
          return { ...state, current: { ...current, geometries: progress.geometries } };
        case "step":
          return { ...state, current: { ...current, step: progress.step } };
        case "status":
          return { ...state, current: { ...current, status: progress.message } };
      }
    }
  }
};

/** A bar of the page: `done` of `total`, and what it stands for. */
export type Bar = {
  readonly done: number;
  readonly total: number;
  readonly text: string;
};

/** The bar of the case being measured, under a title that names the case. */
export type CaseBar = Bar & {
  readonly title: string;
  /** The sweep's own status line, for a person who wants the detail. */
  readonly detail?: string;
};

export type ProgressView = {
  readonly overall: Bar;
  /** Absent before the first case starts. */
  readonly current?: CaseBar;
  /** The time left, in words. */
  readonly remaining: string;
  /** One line per sweep of the run: the patterns found so far. */
  readonly patterns: readonly string[];
  /** A device loss, which stops the run after the case. */
  readonly warning?: string;
};

const SWEEP_LABEL: Readonly<Record<SweepId, string>> = {
  gemv: "GEMV",
  geometry: "tile geometry",
};

const KIND_LABEL: Readonly<Record<CaseKind, string>> = {
  sentinel: "a sentinel: every candidate",
  other: "the candidates its sentinels predict",
  "first-of-class": "first of its class: every candidate",
  later: "the candidates its class kept",
};

/** What a phase does, in words, with the unit its count is of. */
const PHASE_TEXT: Readonly<Record<GemvSweepPhase | GeometrySweepPhase, string>> = {
  prepare: "Building the kernels",
  "warm-start": "Warming up the GPU",
  "quick-look": "A quick look at every candidate",
  calibrate: "Calibrating the kernels that go on",
  rounds: "Timing rounds",
  timing: "Timing the geometries that go on",
  repeat: "Timing the default again",
};

const ratio = (value: number): string => `×${value.toFixed(2)}`;

const ratio3 = (value: number): string => `×${value.toFixed(3)}`;

/**
 * What the GEMV run's class says to the person who runs it: steady — this run is enough for a
 * table; unsteady — run the page once more at another time; no class yet — too few groups so far.
 */
export const steadinessLine = (steadiness: GemvSteadiness): string => {
  const counted = `${steadiness.unsteadyGroups} of ${steadiness.groups} groups spread beyond ${ratio(GEMV_STEADY_GROUP_SPREAD)}`;
  switch (steadiness.class) {
    case "steady":
      return `The GPU's timings held steady in this run (${counted}): this run alone is enough for a table`;
    case "unsteady":
      return `The GPU's timings moved during this run (${counted}, more than ${Math.round(GEMV_STEADY_MAX_SHARE * 100)} %): please run the page once more at another time — a table needs two such runs`;
    case "unclassified":
      return `Too few GEMV groups so far to tell whether the timings hold steady (${steadiness.groups} of the ${GEMV_STEADY_MIN_GROUPS} needed)`;
  }
};

/**
 * The pattern lines of the sweeps a run plans (the same words live and at the end): per sweep its
 * count, and for the GEMV sweep its count at the default margin and the lowest one side by side and
 * the run's class.
 */
export const patternLines = (
  plan: BenchmarkPlan,
  state: Pick<ProgressState, "patterns" | "gemvSummaries">,
): string[] => {
  const { patterns } = state;
  return plan.sweeps.flatMap(({ sweep }) =>
    sweep === "gemv"
      ? [
          `GEMV: ${patterns.gemv.atMargin} of ${patterns.gemv.of} shapes measured so far have a kernel that beats today's choice by ${ratio3(GEMV_MARGIN)} times its timing noise (${patterns.gemv.atLowest} by ${ratio3(GEMV_LOWEST_MARGIN)})`,
          steadinessLine(gemvSteadiness(state.gemvSummaries)),
        ]
      : [
          `Tile geometry: ${patterns.geometry.found} of ${patterns.geometry.of} classes measured so far have a geometry at least ${ratio(GEOMETRY_MIN_SPEEDUP)} faster than the default in every case`,
        ],
  );
};

/** The case bar's title: its place in the run, its sweep, its id and what it measures. */
const caseTitle = (plan: BenchmarkPlan, index: number): string => {
  const runCase = plan.cases[index];
  const place = `Case ${index + 1} of ${plan.cases.length}`;
  return runCase === undefined
    ? place
    : `${place} · ${SWEEP_LABEL[runCase.sweep]} · ${runCase.caseId} (${KIND_LABEL[runCase.kind]})`;
};

/** The case bar: the step it is at, or that it acquires its device. */
const caseBar = (plan: BenchmarkPlan, current: Current): CaseBar => {
  const title = caseTitle(plan, current.index);
  const detail = current.status === undefined ? {} : { detail: current.status };
  const { step } = current;
  if (step === undefined) return { title, text: "Acquiring the GPU", done: 0, total: 1, ...detail };
  const rowCounts = current.rowCounts ?? [];
  const position = step.m === undefined ? -1 : rowCounts.indexOf(step.m);
  const rowCount =
    step.m === undefined
      ? ""
      : `M = ${step.m}${position < 0 ? "" : ` (row count ${position + 1} of ${rowCounts.length})`} · `;
  const { geometries } = current;
  const planned =
    geometries === undefined
      ? ""
      : `${geometries.planned} geometries${geometries.skipped === 0 ? "" : `, ${geometries.skipped} skipped`} · `;
  return {
    title,
    text: `${rowCount}${planned}${PHASE_TEXT[step.phase]}: ${step.done} of ${step.total}`,
    done: step.done,
    total: step.total,
    ...detail,
  };
};

/** The view of `state` at `now` (ms). */
export const progressView = (
  plan: BenchmarkPlan,
  state: ProgressState,
  now: number,
): ProgressView => {
  const total = plan.cases.length;
  const done = state.finished.size;
  const bySweep = plan.sweeps.map(({ sweep }) => {
    const indices = plan.cases.flatMap((runCase, index) =>
      runCase.sweep === sweep ? [index] : [],
    );
    const finished = indices.filter((index) => state.finished.has(index)).length;
    return `${SWEEP_LABEL[sweep]} ${finished} of ${indices.length}`;
  });
  const { idle, current } = state;
  const estimate = remainingEstimate(
    plan.cases.map(({ kind }) => kind),
    {
      finished: state.finished,
      ...(current === undefined
        ? {}
        : { running: { index: current.index, seconds: (now - current.startedAt) / 1000 } }),
      ...(idle === undefined
        ? {}
        : { idling: { index: idle.index, seconds: (idle.until - now) / 1000 } }),
    },
    plan.idleSeconds,
  );
  const left = durationText(estimate.seconds);
  const remaining = `${left.charAt(0).toUpperCase()}${left.slice(1)} left${estimate.measured ? "" : " (a first estimate from other machines' runs; it is refined as cases finish)"}`;
  const caseView: CaseBar | undefined =
    current !== undefined
      ? caseBar(plan, current)
      : idle !== undefined
        ? {
            title: caseTitle(plan, idle.index),
            text: `Cooling down before it: ${Math.max(0, Math.ceil((idle.until - now) / 1000))} s left`,
            done: Math.min(now, idle.until) - idle.since,
            total: idle.until - idle.since,
          }
        : undefined;
  return {
    overall: {
      done,
      total,
      text: `${done} of ${total} cases done${bySweep.length > 1 ? ` (${bySweep.join(" · ")})` : ""}`,
    },
    ...(caseView === undefined ? {} : { current: caseView }),
    remaining,
    patterns: patternLines(plan, state),
    ...(state.deviceLost === undefined
      ? {}
      : {
          warning: `The GPU device was lost (${state.deviceLost.reason}): ${state.deviceLost.message}. The run stops after this case.`,
        }),
  };
};
