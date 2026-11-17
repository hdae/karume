// Synthetic GEMV sweep records (`karume-gemv-sweep/2`) for the tests of the GEMV derivation
// (src/tune/gemv-derive.ts) and of the generator that uses it (tools/geometry-profiles — tools may
// reach into tests/helpers: ADR 0008 addendum 2026-09-05). The rows are the sweep's own measurement
// plans (`gemvMeasurementPlans`, the full set) of the current runtime, and the record is assembled by
// the sweep's own `gemvSweepReport`, so its fingerprints and its default-profile choices are what a
// sweep of the current runtime writes. Every kernel takes `BASE_TIME` unless a case gives it another
// time, so a candidate's gain is `BASE_TIME ÷ its time`. No GPU.

import { opTolerance } from "../../src/reference/op-tolerance.ts";
import { type GemvGates, gemvMeasurementPlans } from "../../src/tune/gemv-candidates.ts";
import { gemvCaseOf } from "../../src/tune/gemv-cases.ts";
import {
  choiceRecord,
  type GemvSweepCaseSummary,
  type GemvSweepReport,
  type GemvSweepRow,
} from "../../src/tune/gemv-report.ts";
import { gemvSweepReport } from "../../src/tune/gemv-sweep.ts";

/** The time of every kernel a case does not time otherwise (ns per dispatch). */
export const BASE_TIME = 1000;

export const GEMV_ADAPTER = {
  vendor: "testvendor",
  architecture: "testarch",
  device: "0x1234",
  description: "Test GPU",
} as const;

const GATES: GemvGates = { subgroups: true, maxComputeWorkgroupsPerDimension: 65535 };

/** One case of a record. */
export type GemvCaseSpec = {
  readonly caseId: string;
  /** The row counts measured (default: M = 1 only). */
  readonly rows?: readonly number[];
  /** The time per dispatch of a kernel by row count and candidate name (default {@link BASE_TIME}). */
  readonly times?: Readonly<Record<number, Readonly<Record<string, number>>>>;
  /** Fields laid over a row by row count and candidate name (a failed check, an error). */
  readonly fields?: Readonly<Record<number, Readonly<Record<string, Partial<GemvSweepRow>>>>>;
  /** The baseline spread of every row count (default 1.01). */
  readonly spread?: number;
};

export type GemvRecordOptions = {
  readonly adapter?: GemvSweepReport["adapter"];
  readonly candidateSet?: "quick" | "full";
};

/** A record of the cases, its rows the full set's plans at each case's row counts. */
export const gemvRecord = (
  date: string,
  specs: readonly GemvCaseSpec[],
  options: GemvRecordOptions = {},
): GemvSweepReport => {
  const rows: GemvSweepRow[] = [];
  const summaries: GemvSweepCaseSummary[] = [];
  for (const spec of specs) {
    const gemvCase = gemvCaseOf(spec.caseId);
    if (gemvCase === undefined) throw new Error(`no case ${spec.caseId}`);
    const counts = spec.rows ?? [1];
    for (const plan of gemvMeasurementPlans(gemvCase, "full", GATES)) {
      if (!counts.includes(plan.m)) continue;
      const time = (name: string): number => spec.times?.[plan.m]?.[name] ?? BASE_TIME;
      const baseline = time(plan.baseline.name);
      for (const [index, candidate] of [plan.baseline, ...plan.candidates].entries()) {
        const isBaseline = index === 0;
        const perDispatch = time(candidate.name);
        rows.push({
          caseId: gemvCase.id,
          m: plan.m,
          candidate: candidate.name,
          class: candidate.class,
          isBaseline,
          choice: choiceRecord(candidate.choice, gemvCase.storage),
          reps: 100,
          // Off Chrome's 100 µs step, so the record does not look quantized
          rounds: [perDispatch * 100 + 1, perDispatch * 101 + 1],
          perDispatch,
          ...(isBaseline ? {} : { speedupVsBaseline: baseline / perDispatch }),
          ...(candidate.class === "identical"
            ? isBaseline
              ? {}
              : { identicalToBaseline: true, differingWords: 0 }
            : { maxAbsError: 1e-6, bandUse: 0.01, withinBand: true, nonFiniteOutputs: 0 }),
          ...spec.fields?.[plan.m]?.[candidate.name],
        });
      }
      summaries.push({
        caseId: gemvCase.id,
        m: plan.m,
        elapsedMs: 1000,
        profile: plan.profile,
        baselineSpread: spec.spread ?? 1.01,
      });
    }
  }
  const cases = specs.flatMap(({ caseId }) => {
    const gemvCase = gemvCaseOf(caseId);
    return gemvCase === undefined ? [] : [gemvCase];
  });
  const candidateSet =
    options.candidateSet ??
    (specs.some(({ rows: counts }) => counts?.some((m) => m > 1)) ? "full" : "quick");
  return gemvSweepReport({
    startedAt: date,
    date,
    aborted: false,
    userAgent: "Node.js/24",
    adapter: options.adapter ?? GEMV_ADAPTER,
    gates: {
      subgroups: true,
      features: { subgroups: true, subgroupSizeControl: true, subgroupId: true },
      maxComputeWorkgroupsPerDimension: 65535,
    },
    timing: { feature: true, unit: "ns" },
    candidateSet,
    caseIds: cases.map(({ id }) => id),
    rounds: 2,
    tolerance: opTolerance("linear"),
    deviceLost: null,
    cases,
    summaries,
    rows,
  });
};
