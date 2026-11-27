/**
 * Complete GEMV sweep records for the gemv-sweep entry's tests: what one
 * `runGemvSweep({ cases: [caseId] })` returns, with the rows of the baseline, one identical and one
 * reorder candidate at M = 1. Complete, so a field the entry's code starts to read is never silently
 * missing from a fake. The cases are the runtime's own; the fingerprints are placeholders (the
 * entry reads none — the derivation does).
 */

import {
  type GemvSweepCaseSummary,
  type GemvSweepReport,
  type GemvSweepRow,
  gemvSweepCases,
} from "@karume/runtime/tune";

/** Two i8 cases of the table (no group): E2B QAT's per_layer_projection and per_layer_input_gate. */
export const CASE_A = "i8-n1536-k256";
export const CASE_B = "i8-n256-k1536";
/** A sentinel of the table that comes after CASE_A and CASE_B in it (the explore set moves it first). */
export const CASE_C = "i4g32-n2112-k2816";

export const ADAPTER = {
  vendor: "nvidia",
  architecture: "ampere",
  device: "0x2208",
  description: "NVIDIA GeForce RTX 3080 Ti",
};

/** One row of `caseId` at M = 1: `kind` picks the baseline, an identical or a reorder candidate. */
export const gemvRow = (
  caseId: string,
  kind: "baseline" | "identical" | "reorder",
  extra: Partial<GemvSweepRow> = {},
): GemvSweepRow =>
  ({
    caseId,
    m: 1,
    candidate: kind === "baseline" ? "c32u4" : kind === "identical" ? "c64u8" : "parallel-l4",
    class: kind === "reorder" ? "reorder" : "identical",
    isBaseline: kind === "baseline",
    choice:
      kind === "reorder"
        ? { id: "gemv-parallel", storage: "i8", lanes: 4, variant: { cols: 32, unroll: 4 } }
        : {
            id: "gemv-m1",
            storage: "i8",
            variant: kind === "baseline" ? { cols: 32, unroll: 4 } : { cols: 64, unroll: 8 },
          },
    kernel: "0123456789abcdef",
    keys: ["key"],
    reps: 100,
    rounds: [80_000_000, 80_100_000],
    perDispatch: kind === "baseline" ? 20_000 : kind === "identical" ? 16_000 : 10_000,
    ...(kind === "baseline"
      ? { maxAbsError: 1e-5, bandUse: 0.05, withinBand: true, nonFiniteOutputs: 0 }
      : kind === "identical"
        ? { speedupVsBaseline: 1.25, identicalToBaseline: true, differingWords: 0 }
        : {
            speedupVsBaseline: 2,
            maxAbsError: 2e-5,
            bandUse: 0.1,
            withinBand: true,
            nonFiniteOutputs: 0,
          }),
    ...extra,
  }) satisfies GemvSweepRow;

export type PerCaseOptions = {
  readonly rows?: readonly GemvSweepRow[];
  readonly aborted?: boolean;
  readonly deviceLost?: GemvSweepReport["deviceLost"];
  readonly quantized?: boolean;
  readonly date?: string;
  readonly gates?: GemvSweepReport["gates"];
  /** The profile the sweep's device resolved to (default `default`). */
  readonly geometryProfile?: string;
  readonly summary?: Partial<GemvSweepCaseSummary>;
};

/** What one `runGemvSweep({ cases: [caseId] })` returns. */
export const gemvPerCase = (caseId: string, options: PerCaseOptions = {}): GemvSweepReport => {
  const rows = options.rows ?? [
    gemvRow(caseId, "baseline"),
    gemvRow(caseId, "identical"),
    gemvRow(caseId, "reorder"),
  ];
  const gemvCase = gemvSweepCases().find(({ id }) => id === caseId);
  if (gemvCase === undefined) throw new Error(`no case ${caseId}`);
  return {
    format: "karume-gemv-sweep/4",
    date: options.date ?? "2026-11-17T10:05:00.000Z",
    startedAt: "2026-11-17T10:00:00.000Z",
    aborted: options.aborted ?? false,
    userAgent: "Node.js/24",
    adapter: ADAPTER,
    geometryProfile: options.geometryProfile ?? "default",
    gates: options.gates ?? {
      subgroups: true,
      features: { subgroups: true, subgroupSizeControl: true, subgroupId: true },
      maxComputeWorkgroupsPerDimension: 65535,
    },
    gpuTiming: { feature: true, unit: "ns", quantized: options.quantized ?? false },
    settings: {
      candidateSet: "quick",
      cases: [caseId],
      rounds: 5,
      quickLookMargin: 1.25,
      targetPassMs: 80,
      maxReps: 16384,
      warmStart: { minNs: 1e9, maxNs: 5e9, maxPasses: 125, stablePasses: 3, tolerance: 0.02 },
      recalibrateBelow: 0.25,
      calibrationAttempts: 3,
      wallTimingNote: "note",
      tolerance: { atol: 2e-4, rtol: 4e-6 },
      notMeasured: "packed",
    },
    deviceLost: options.deviceLost ?? null,
    cases: [{ ...gemvCase, rows: [1], fingerprint: "fedcba9876543210" }],
    summaries: [
      {
        caseId,
        m: 1,
        elapsedMs: 30_000,
        profile: { exact: "c32u4", reorder: "parallel-l4" },
        baselineSpread: 1.004,
        screened: rows.filter((one) => one.screened === true).length,
        skipped: rows.filter((one) => one.skipped === true).length,
        ...options.summary,
      },
    ],
    rows,
  };
};
