// Synthetic GEMV sweep records (`karume-gemv-sweep/4`) for the tests of the GEMV derivation
// (src/tune/gemv-derive.ts) and of the generator that uses it (tools/geometry-profiles — tools may
// reach into tests/helpers: ADR 0008 addendum 2026-09-05). The rows are the sweep's own measurement
// plans (`gemvMeasurementPlans`, the full set or the row-blocks set by row count) of the current
// runtime, each with the fingerprint of its kernel, and the record is assembled by the sweep's own
// `gemvSweepReport`, so its fingerprints and its default-profile choices are what a sweep of the
// current runtime writes. Every kernel takes `BASE_TIME` unless a case gives it another time, so a
// candidate's gain is `BASE_TIME ÷ its time`. No GPU.

import {
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
} from "../../src/kernels/geometry-profile.ts";
import { LINEAR_GEMV_PARALLEL_MAX_ROWS } from "../../src/kernels/linear-gemv.ts";
import { opTolerance } from "../../src/reference/op-tolerance.ts";
import {
  type GemvGates,
  gemvMeasurementPlans,
  gemvProgram,
} from "../../src/tune/gemv-candidates.ts";
import { gemvCaseOf } from "../../src/tune/gemv-cases.ts";
import { gemvKernelId } from "../../src/tune/gemv-fingerprint.ts";
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
  /**
   * Fields laid over a row by row count and candidate name (a failed check, an error, an old
   * kernel fingerprint). A row given `screened: true` loses its time and its checks, as the sweep
   * writes it; a row given `skipped: true` loses its quick time too.
   */
  readonly fields?: Readonly<Record<number, Readonly<Record<string, Partial<GemvSweepRow>>>>>;
  /** The baseline spread of every row count (default 1.01), or per row count. */
  readonly spread?: number | Readonly<Record<number, number>>;
  /** Fields laid over the summary of a row count (another default-profile choice, say). */
  readonly summaries?: Readonly<Record<number, Partial<GemvSweepCaseSummary>>>;
};

export type GemvRecordOptions = {
  readonly adapter?: GemvSweepReport["adapter"];
  readonly candidateSet?: "quick" | "full" | "row-blocks";
  /**
   * The profile the sweep's device resolved to (default: the default profile): the record names its
   * id, and the GEMM rows of a float-stored case carry the fingerprint of the GEMM it builds.
   */
  readonly profile?: GeometryProfile;
};

/** A row without its measurement: its identity and launch fields, and its time as `quick`. */
const unmeasured = (row: GemvSweepRow): { rest: GemvSweepRow; quick?: number } => {
  const {
    reps: _reps,
    rounds: _rounds,
    wallRounds: _wallRounds,
    perDispatch,
    speedupVsBaseline: _speedup,
    identicalToBaseline: _identical,
    differingWords: _differing,
    maxAbsError: _error,
    bandUse: _bandUse,
    withinBand: _within,
    nonFiniteOutputs: _nonFinite,
    ...rest
  } = row;
  return { rest, ...(perDispatch === undefined ? {} : { quick: perDispatch }) };
};

/** A row the quick look screened out: its identity, launch fields and quick time only. */
const screenedRow = (row: GemvSweepRow): GemvSweepRow => {
  const { rest, quick } = unmeasured(row);
  return { ...rest, ...(quick === undefined ? {} : { quickPerDispatch: quick }) };
};

/** A row the M = 1 group left out: its identity and launch fields only. */
const skippedRow = (row: GemvSweepRow): GemvSweepRow => unmeasured(row).rest;

/** A record of the cases, its rows the plans of the sets at each case's row counts. */
export const gemvRecord = (
  date: string,
  specs: readonly GemvCaseSpec[],
  options: GemvRecordOptions = {},
): GemvSweepReport => {
  const rows: GemvSweepRow[] = [];
  const summaries: GemvSweepCaseSummary[] = [];
  const profile = options.profile ?? DEFAULT_GEOMETRY_PROFILE;
  for (const spec of specs) {
    const gemvCase = gemvCaseOf(spec.caseId);
    if (gemvCase === undefined) throw new Error(`no case ${spec.caseId}`);
    const counts = spec.rows ?? [1];
    const plans = [
      ...gemvMeasurementPlans(gemvCase, "full", GATES),
      ...gemvMeasurementPlans(gemvCase, "row-blocks", GATES),
    ];
    for (const plan of plans) {
      if (!counts.includes(plan.m)) continue;
      const time = (name: string): number => spec.times?.[plan.m]?.[name] ?? BASE_TIME;
      const baseline = time(plan.baseline.name);
      let screened = 0;
      let skipped = 0;
      for (const [index, candidate] of [plan.baseline, ...plan.candidates].entries()) {
        const isBaseline = index === 0;
        const perDispatch = time(candidate.name);
        const row: GemvSweepRow = {
          caseId: gemvCase.id,
          m: plan.m,
          candidate: candidate.name,
          class: candidate.class,
          isBaseline,
          choice: choiceRecord(candidate.choice, gemvCase.storage),
          kernel: gemvKernelId(gemvCase, plan.m, candidate.choice, (one, m, choice) =>
            gemvProgram(one, m, choice, profile),
          ),
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
        };
        if (row.screened === true) screened += 1;
        if (row.skipped === true) skipped += 1;
        rows.push(
          row.screened === true ? screenedRow(row) : row.skipped === true ? skippedRow(row) : row,
        );
      }
      const spread =
        typeof spec.spread === "number" ? spec.spread : (spec.spread?.[plan.m] ?? 1.01);
      summaries.push({
        caseId: gemvCase.id,
        m: plan.m,
        elapsedMs: 1000,
        profile: plan.profile,
        baselineSpread: spread,
        screened,
        skipped,
        ...spec.summaries?.[plan.m],
      });
    }
  }
  const cases = specs.flatMap(({ caseId }) => {
    const gemvCase = gemvCaseOf(caseId);
    return gemvCase === undefined ? [] : [gemvCase];
  });
  const measured = specs.flatMap(({ rows: counts }) => counts ?? [1]);
  const candidateSet =
    options.candidateSet ??
    (measured.some((m) => m > LINEAR_GEMV_PARALLEL_MAX_ROWS)
      ? "row-blocks"
      : measured.some((m) => m > 1)
        ? "full"
        : "quick");
  const record = gemvSweepReport({
    startedAt: date,
    date,
    aborted: false,
    userAgent: "Node.js/24",
    adapter: options.adapter ?? GEMV_ADAPTER,
    geometryProfile: profile.id,
    gates: {
      subgroups: true,
      features: { subgroups: true, subgroupSizeControl: true, subgroupId: true },
      maxComputeWorkgroupsPerDimension: 65535,
    },
    timing: { feature: true, unit: "ns" },
    candidateSet,
    caseIds: cases.map(({ id }) => id),
    selection: {},
    rounds: 2,
    quickLookMargin: 1.25,
    tolerance: opTolerance("linear"),
    deviceLost: null,
    cases,
    summaries,
    rows,
  });
  // The planned cases list the row counts the record measured (a test may mix the sets' counts)
  return {
    ...record,
    cases: record.cases.map((entry) => {
      const spec = specs.find(({ caseId }) => caseId === entry.id);
      return { ...entry, rows: spec?.rows ?? [1] };
    }),
  };
};
