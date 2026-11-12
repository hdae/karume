// Synthetic fixture of sweep records (`karume-geometry-sweep/3`) — shared by the tests of the
// generator (src/tune/derive.ts). Every record carries the per-family matching keys a writer records
// for every case of the sweep's table (`recordKeys` — the generator requires them for every family a
// record has rows of).
//
// The runtime's generator tests (tune_profile_test.ts, tune_profile_test.ts) and the tests of the
// tool's output rendering and CLI (tools/geometry-sweep/profile_test.ts, render_test.ts — tools may
// reach into tests/helpers: ADR 0008 addendum 2026-09-05) build tables from the same records. No
// GPU.

import { defaultGemmGeometry, type GemmGeometry } from "../../src/kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../../src/kernels/i8a8-geometry.ts";
import type { SweepOp } from "../../src/tune/cases.ts";
import { gemmCandidate } from "../../src/tune/geometries.ts";
import { parseSweepReport, type SlotVerdict, type SweepSource } from "../../src/tune/derive.ts";
import type { GeometryProfileFamilyIds } from "../../src/tune/families.ts";
import { sweepCaseSetIds, sweepDefaultKernelsIds } from "../../src/tune/fingerprint.ts";
import { REPORT_FORMAT } from "../../src/tune/report.ts";

let currentKeys:
  | {
      readonly caseSets: GeometryProfileFamilyIds;
      readonly defaultKernels: GeometryProfileFamilyIds;
    }
  | undefined;

/**
 * The current runtime's `caseSets` and `defaultKernels` of every family (what `runGeometrySweep`
 * writes for a sweep of every case — derived once, the default fingerprint runs the codegen of every
 * case).
 */
export const recordKeys = (): {
  readonly caseSets: GeometryProfileFamilyIds;
  readonly defaultKernels: GeometryProfileFamilyIds;
} => {
  currentKeys ??= { caseSets: sweepCaseSetIds(), defaultKernels: sweepDefaultKernelsIds() };
  return currentKeys;
};

// --- Records of the rule-extraction tests (tune_profile_test.ts): cases and candidates as rows ---

export const BIG = defaultGemmGeometry();
export const A: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };
export const B: GemmGeometry = { regM: 8, regN: 4, wgX: 16, wgY: 8 };
/** Geometry of the ≤ 64 bucket of the default table (A is the 65〜512 bucket, BIG the > 512 one). */
export const SMALL: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };

/** Field names of the 7 buckets (expected spelling of the generator — written by hand). */
export const ROWS_SLOT_NAMES = [
  "gemmRows ≤ 16",
  "gemmRows 17〜32",
  "gemmRows 33〜64",
  "gemmRows 65〜128",
  "gemmRows 129〜256",
  "gemmRows 257〜512",
  "gemmRows > 512",
] as const;

/** Field names of the 3 i8a8.linear buckets (expected spelling of the generator — written by hand). */
export const I8A8_LINEAR_SLOT_NAMES = [
  "i8a8.linear ≤ 256",
  "i8a8.linear 257〜512",
  "i8a8.linear > 512",
] as const;

export type CaseRef = { readonly caseId: string; readonly op: SweepOp; readonly shape: string };

export const linearCase = (m: number): CaseRef => ({
  caseId: `linear-m${m}`,
  op: "linear",
  shape: `M${m} N64 K64`,
});

/** The matmul shape is spelled like linear (caseShape in cases.ts). */
export const matmulCase = (m: number): CaseRef => ({
  caseId: `matmul-m${m}`,
  op: "matmul",
  shape: `M${m} N64 K64`,
});

/**
 * The bmm shape puts the batch first (caseShape in cases.ts — the bucket is the M of one matrix).
 */
export const bmmCase = (batch: number, m: number): CaseRef => ({
  caseId: `bmm-b${batch}-m${m}`,
  op: "bmm",
  shape: `B${batch} M${m} N64 K64`,
});

export type Measured = {
  readonly speedup?: number;
  readonly identical?: boolean;
  readonly error?: string;
  /**
   * Pass time per measurement round (input to the rounding-error upper bound — omitted from the row
   * if left out).
   */
  readonly rounds?: readonly number[];
  readonly reps?: number;
};

/** One sweep row (the fields of `SweepRow` in `report.ts` the generator reads + a few more). */
export const row = (
  ref: CaseRef,
  candidate: { readonly name: string; readonly geometry: GemmGeometry | I8a8Geometry },
  isDefault: boolean,
  measured: Measured = {},
): Record<string, unknown> => ({
  ...ref,
  censusCount: 1,
  geometry: candidate.name,
  geometryParams: candidate.geometry,
  isDefault,
  ...(isDefault ? { speedupVsDefault: 1, identicalToDefault: true } : {}),
  ...(measured.speedup === undefined ? {} : { speedupVsDefault: measured.speedup }),
  ...(measured.identical === undefined ? {} : { identicalToDefault: measured.identical }),
  ...(measured.error === undefined ? {} : { error: measured.error }),
  ...(measured.rounds === undefined ? {} : { rounds: measured.rounds }),
  ...(measured.reps === undefined ? {} : { reps: measured.reps }),
});

/**
 * The rows of one case (default + a measurement per candidate; the default row's rounds and the
 * like come from `defaultMeasured`).
 */
export const caseRows = (
  ref: CaseRef,
  defaultGeometry: GemmGeometry,
  candidates: readonly (readonly [GemmGeometry, Measured])[],
  defaultMeasured: Measured = {},
): Record<string, unknown>[] => [
  row(ref, gemmCandidate(defaultGeometry), true, defaultMeasured),
  ...candidates.map(([geometry, measured]) =>
    row(ref, gemmCandidate(geometry), false, { identical: true, ...measured }),
  ),
];

export const ADAPTER = {
  vendor: "apple",
  architecture: "metal-3",
  device: "",
  description: "Test GPU",
};

/**
 * Re-measurement of the default per case (`cases[]`). By default every case has ratio 1 (not
 * excluded from the input).
 */
export type Repeats = Readonly<Record<string, Record<string, unknown>>>;

export const report = (
  rows: readonly Record<string, unknown>[],
  adapter: Record<string, string> = ADAPTER,
  repeats: Repeats = {},
): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-09-27T00:00:00.000Z",
  ...recordKeys(),
  adapter,
  settings: { candidateSet: "full" },
  gpuTiming: { feature: true, unit: "ns", quantized: false },
  cases: [...new Set(rows.map((entry) => String(entry.caseId)))].map(
    (caseId) => repeats[caseId] ?? { caseId, defaultRepeat: { perDispatch: 1, driftRatio: 1 } },
  ),
  rows,
});

export const source = (
  rows: readonly Record<string, unknown>[],
  name = "sweep.json",
  adapter?: Record<string, string>,
): SweepSource => parseSweepReport(report(rows, adapter), { path: name, sha256: `sha-${name}` });

export const OPTIONS = { vendor: "apple", architecture: "metal-3", minSpeedup: 1.05 } as const;

export const verdictOf = (
  verdicts: readonly SlotVerdict[],
  slot: SlotVerdict["slot"],
): SlotVerdict => {
  const found = verdicts.find((verdict) => verdict.slot === slot);
  if (found === undefined) throw new Error(`${slot} is missing`);
  return found;
};

export const rejectionOf = (verdict: SlotVerdict, geometry: GemmGeometry): string => {
  const name = gemmCandidate(geometry).name;
  const found = verdict.rejected.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`${name} is not in the rejected list`);
  return found.reason;
};

// --- Records for the injection-table tests (tune_profile_test.ts): one linear case, two sweeps ---

export const DEFAULT = defaultGemmGeometry();
export const FAST: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };

/**
 * One linear M = 1024 case (the default, and one geometry 1.2× faster than the default with
 * matching output).
 */
export const sweep = (name: string, sha256: string): SweepSource =>
  parseSweepReport(
    {
      format: REPORT_FORMAT,
      date: "2026-09-29T00:00:00.000Z",
      ...recordKeys(),
      adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
      settings: { candidateSet: "full" },
      gpuTiming: { feature: true, unit: "ns", quantized: false },
      cases: [{ caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } }],
      rows: [
        {
          caseId: "linear-m1024",
          op: "linear",
          shape: "M1024 N64 K64",
          geometry: gemmCandidate(DEFAULT).name,
          geometryParams: DEFAULT,
          isDefault: true,
          speedupVsDefault: 1,
          identicalToDefault: true,
        },
        {
          caseId: "linear-m1024",
          op: "linear",
          shape: "M1024 N64 K64",
          geometry: gemmCandidate(FAST).name,
          geometryParams: FAST,
          isDefault: false,
          speedupVsDefault: 1.2,
          identicalToDefault: true,
        },
      ],
    },
    { path: name, sha256 },
  );

export const SPEC = {
  from: ["a.json", "b.json"],
  id: "test-gpu",
  vendor: "apple",
  architecture: "metal-3",
  out: "profiles/test-gpu.ts",
  minSpeedup: 1.05,
} as const;

export const SOURCES = [sweep("a.json", "sha-a"), sweep("b.json", "sha-b")];
