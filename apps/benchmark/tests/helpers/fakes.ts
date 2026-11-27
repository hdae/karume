/**
 * Fakes for the benchmark's tests: complete per-case records of the two sweeps' `explore` sets (what
 * one `runGemvSweep` / `runGeometrySweep` call with `cases: [id]` returns), fake runtime sweeps that
 * report the progress events a real one does around those records, the record environment of a
 * browser and an idle wait that does not wait. The cases are the runtime's own; the fingerprints the
 * derivations read are placeholders except where the merge needs them.
 */

import {
  type GemvSweepCaseSummary,
  type GemvSweepOptions,
  type GemvSweepReport,
  type GemvSweepRow,
  gemvSweepCases,
  type GeometrySweepOptions,
  type GeometrySweepReport,
  type GeometrySweepRow,
  sweepCandidateKernelsId,
  sweepCaseSetIds,
  sweepDefaultKernelsIds,
} from "@karume/runtime/tune";
import type { Checkout, RecordEnvironment } from "@karume/sweep-app-support/record";

/** Two i8 cases of the GEMV table and two conv3d cases of the geometry table (one class). */
export const GEMV_A = "i8-n1536-k256";
export const GEMV_B = "i8-n256-k1536";
export const CONV3D_A = "i8a8-conv3d-cin256-cout256-k333-in6x128x128";
export const CONV3D_B = "i8a8-conv3d-cin512-cout512-k333-in6x64x64";

export const ADAPTER = {
  vendor: "nvidia",
  architecture: "blackwell",
  device: "",
  description: "NVIDIA GeForce RTX 5070 Ti",
};

export const ENVIRONMENT: RecordEnvironment = {
  host: "browser",
  userAgent: "Mozilla/5.0 Chrome/155 test",
  crossOriginIsolated: true,
  adapter: ADAPTER,
  features: ["shader-f16", "timestamp-query"],
  limits: { maxBufferSize: 4294967296 },
  wgslLanguageFeatures: [],
};

export const CHECKOUT: Checkout = {
  commit: "0123456789abcdef",
  dirty: false,
  runtimeVersion: "0.24.0",
};

/** One GEMV row of `caseId` at `m`: the baseline, an identical or a reorder candidate. */
export const gemvRow = (
  caseId: string,
  kind: "baseline" | "identical" | "reorder" | "subgroup",
  extra: Partial<GemvSweepRow> = {},
): GemvSweepRow => ({
  caseId,
  m: 1,
  candidate:
    kind === "baseline"
      ? "c32u4"
      : kind === "identical"
        ? "c64u8"
        : kind === "reorder"
          ? "parallel-l4"
          : "subgroup32-l8",
  class: kind === "baseline" || kind === "identical" ? "identical" : "reorder",
  isBaseline: kind === "baseline",
  choice:
    kind === "reorder"
      ? { id: "gemv-parallel", storage: "i8", lanes: 4, variant: { cols: 32, unroll: 4 } }
      : kind === "subgroup"
        ? {
            id: "gemv-parallel-subgroup32",
            storage: "i8",
            lanes: 8,
            variant: { cols: 32, unroll: 4 },
          }
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
});

/** The summary of `caseId` at M = 1, the default's choices `c32u4` (exact) and `parallel-l4` (reorder). */
export const gemvSummary = (
  caseId: string,
  extra: Partial<GemvSweepCaseSummary> = {},
): GemvSweepCaseSummary => ({
  caseId,
  m: 1,
  elapsedMs: 15_000,
  profile: { exact: "c32u4", reorder: "parallel-l4" },
  baselineSpread: 1.004,
  screened: 0,
  skipped: 0,
  ...extra,
});

/** What one `runGemvSweep({ candidateSet: "explore", cases: [caseId] })` returns. */
export const gemvReport = (
  caseId: string,
  options: {
    readonly rows?: readonly GemvSweepRow[];
    readonly deviceLost?: GemvSweepReport["deviceLost"];
    readonly aborted?: boolean;
  } = {},
): GemvSweepReport => {
  const gemvCase = gemvSweepCases().find(({ id }) => id === caseId);
  if (gemvCase === undefined) throw new Error(`no GEMV case ${caseId}`);
  const rows = options.rows ?? [
    gemvRow(caseId, "baseline"),
    gemvRow(caseId, "identical"),
    gemvRow(caseId, "reorder"),
  ];
  return {
    format: "karume-gemv-sweep/4",
    date: "2026-11-26T10:05:00.000Z",
    startedAt: "2026-11-26T10:00:00.000Z",
    aborted: options.aborted ?? false,
    userAgent: ENVIRONMENT.userAgent,
    adapter: ADAPTER,
    geometryProfile: "nvidia-blackwell",
    gates: {
      subgroups: true,
      features: { subgroups: true, subgroupSizeControl: true, subgroupId: true },
      maxComputeWorkgroupsPerDimension: 65535,
    },
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    settings: {
      candidateSet: "explore",
      sentinels: [],
      cases: [caseId],
      rounds: 5,
      quickLookMargin: 1.1,
      targetPassMs: 80,
      maxReps: 16384,
      warmStart: {
        minNs: 1e9,
        loadedMinNs: 0,
        maxNs: 5e9,
        maxPasses: 125,
        stablePasses: 3,
        tolerance: 0.02,
      },
      recalibrateBelow: 0.25,
      calibrationAttempts: 3,
      wallTimingNote: "note",
      tolerance: { atol: 2e-4, rtol: 4e-6 },
      notMeasured: "packed",
    },
    deviceLost: options.deviceLost ?? null,
    cases: [{ ...gemvCase, rows: [1], fingerprint: "fedcba9876543210" }],
    summaries: rows.length === 0 ? [] : [gemvSummary(caseId)],
    rows,
  };
};

/** The conv3d default (`tile128x64r8x8w8x16k16`) and one candidate (`tile64x64r8x4w16x8k16`). */
const DEFAULT_GEOMETRY = { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 };
const CANDIDATE_GEOMETRY = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };

/** A conv3d row of `caseId`: the default's or the candidate's (×1.25 and identical unless changed). */
export const geometryRow = (
  caseId: string,
  isDefault: boolean,
  extra: Partial<GeometrySweepRow> = {},
): GeometrySweepRow => ({
  caseId,
  op: "i8a8-conv3d",
  shape: caseId,
  censusCount: 1,
  geometry: isDefault ? "tile128x64r8x8w8x16k16" : "tile64x64r8x4w16x8k16",
  geometryParams: isDefault ? DEFAULT_GEOMETRY : CANDIDATE_GEOMETRY,
  isDefault,
  reps: 4,
  rounds: [80_000_000, 80_100_000],
  perDispatch: isDefault ? 20_000_000 : 16_000_000,
  ...(isDefault ? {} : { speedupVsDefault: 1.25, identicalToDefault: true }),
  ...extra,
});

/** The family keys of the conv3d family alone (what a record of one conv3d case holds). */
const conv3dOnly = (ids: GeometrySweepReport["caseSets"]): GeometrySweepReport["caseSets"] => {
  const id = ids["i8a8-conv3d"];
  if (id === undefined) throw new Error("the full set has no i8a8-conv3d case");
  return { "i8a8-conv3d": id };
};

/** What one `runGeometrySweep({ candidateSet: "explore", cases: [caseId] })` returns. */
export const geometryReport = (
  caseId: string,
  options: {
    readonly rows?: readonly GeometrySweepRow[];
    readonly deviceLost?: GeometrySweepReport["deviceLost"];
  } = {},
): GeometrySweepReport => {
  const rows = options.rows ?? [geometryRow(caseId, true), geometryRow(caseId, false)];
  return {
    format: "karume-geometry-sweep/4",
    date: "2026-11-26T11:05:00.000Z",
    startedAt: "2026-11-26T11:00:00.000Z",
    aborted: false,
    caseSets: conv3dOnly(sweepCaseSetIds("full")),
    defaultKernels: conv3dOnly(sweepDefaultKernelsIds("full")),
    candidateKernels: sweepCandidateKernelsId(rows, true),
    userAgent: ENVIRONMENT.userAgent,
    adapter: ADAPTER,
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    dp4a: true,
    settings: {
      candidateSet: "explore",
      quick: false,
      ops: ["i8a8-conv3d"],
      cases: [caseId],
      rounds: 5,
      quickLookMargin: 1.25,
      targetPassMs: 80,
      maxReps: 16384,
      warmupNs: 1,
      warmupMinRuns: 1,
      wallTimingNote: "note",
    },
    deviceLost: options.deviceLost ?? null,
    cases:
      rows.length === 0
        ? []
        : [
            {
              caseId,
              elapsedMs: 20_000,
              screened: 0,
              defaultRepeat: { perDispatch: 20_000_000, driftRatio: 1.01 },
            },
          ],
    rows,
  };
};

/** The one case a per-case call sweeps. */
const caseOf = (cases: readonly string[] | undefined): string => {
  const [caseId] = cases ?? [];
  if (caseId === undefined || cases?.length !== 1) throw new Error("not a per-case call");
  return caseId;
};

/**
 * A runtime GEMV sweep that reports `started`, its plan, a step, its rows and the case, and returns
 * `report(caseId)` (default {@link gemvReport}); a report with a device loss reports the loss instead
 * of the case.
 */
export const fakeGemvSweep =
  (report: (caseId: string) => GemvSweepReport = (caseId) => gemvReport(caseId)) =>
  async (options: GemvSweepOptions): Promise<GemvSweepReport> => {
    const caseId = caseOf(options.cases);
    const made = report(caseId);
    const progress = options.onProgress ?? (() => {});
    progress({ kind: "started", adapter: ADAPTER, unit: "ns", subgroups: true, caseCount: 1 });
    progress({ kind: "plan", caseId, index: 0, rowCounts: [{ m: 1, kernels: made.rows.length }] });
    progress({ kind: "step", caseId, m: 1, phase: "rounds", done: 0, total: 10 });
    for (const row of made.rows) progress({ kind: "row", row });
    if (made.deviceLost !== null) progress({ kind: "deviceLost", ...made.deviceLost });
    else progress({ kind: "case", caseId, summaries: made.summaries });
    await Promise.resolve();
    return made;
  };

/** A runtime geometry sweep, as {@link fakeGemvSweep}. */
export const fakeGeometrySweep =
  (report: (caseId: string) => GeometrySweepReport = (caseId) => geometryReport(caseId)) =>
  async (options: GeometrySweepOptions): Promise<GeometrySweepReport> => {
    const caseId = caseOf(options.cases);
    const made = report(caseId);
    const progress = options.onProgress ?? (() => {});
    progress({ kind: "started", adapter: ADAPTER, unit: "ns", dp4a: true, caseCount: 1 });
    progress({ kind: "plan", caseId, index: 0, geometries: made.rows.length, skipped: 0 });
    progress({ kind: "step", caseId, phase: "timing", done: 1, total: 2 });
    for (const row of made.rows) progress({ kind: "row", row });
    const [summary] = made.cases;
    if (made.deviceLost !== null) progress({ kind: "deviceLost", ...made.deviceLost });
    else if (summary !== undefined) progress({ kind: "case", summary });
    await Promise.resolve();
    return made;
  };

/** An idle wait that does not wait (and rejects at once with the reason of an aborted signal). */
export const noWait = (_ms: number, signal: AbortSignal): Promise<void> =>
  signal.aborted ? Promise.reject(signal.reason) : Promise.resolve();
