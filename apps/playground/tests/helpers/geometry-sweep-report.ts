/**
 * Complete sweep records for the geometry sweep page's tests: rows of the conv3d default and one
 * candidate, and what one `runGeometrySweep({ cases: [caseId] })` returns. Complete, so a field the
 * page's code starts to read is never silently missing from a fake.
 */

import {
  type GeometrySweepAdapter,
  type GeometrySweepReport,
  type GeometrySweepRow,
  geometrySweepCases,
  sweepCandidateKernelsId,
  sweepCaseSetId,
} from "@karume/runtime/tune";

export const CONV3D = geometrySweepCases()
  .filter(({ op }) => op === "i8a8-conv3d")
  .map(({ id }) => id);
export const [CASE_A = "", CASE_B = "", CASE_C = ""] = CONV3D;

/** The conv3d default (`tile128x64r8x8w8x16k16`) and one candidate (`tile64x64r8x4w16x8k16`). */
const DEFAULT_GEOMETRY = { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 };
export const CANDIDATE_GEOMETRY = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };

export const ADAPTER: GeometrySweepAdapter = {
  vendor: "nvidia",
  architecture: "blackwell",
  device: "0x2c05",
  description: "NVIDIA GeForce RTX 5070 Ti",
};

export const row = (caseId: string, isDefault: boolean, extra: Partial<GeometrySweepRow> = {}) =>
  ({
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
  }) satisfies GeometrySweepRow;

export type PerCaseOptions = {
  readonly date?: string;
  readonly rows?: readonly GeometrySweepRow[];
  readonly aborted?: boolean;
  readonly quantized?: boolean;
  readonly deviceLost?: GeometrySweepReport["deviceLost"];
  readonly adapter?: GeometrySweepAdapter;
  readonly candidateKernels?: string;
};

/**
 * What one `runGeometrySweep({ cases: [caseId] })` returns (its own candidateKernels, unless
 * `candidateKernels` is given — the fingerprint needs a case id of the runtime's table).
 */
export const perCase = (caseId: string, options: PerCaseOptions = {}): GeometrySweepReport => {
  const rows = options.rows ?? [row(caseId, true), row(caseId, false)];
  return {
    format: "karume-geometry-sweep/2",
    date: options.date ?? "2026-10-28T10:05:00.000Z",
    startedAt: "2026-10-28T10:00:00.000Z",
    aborted: options.aborted ?? false,
    caseSet: sweepCaseSetId(),
    defaultKernels: "default-kernels",
    candidateKernels: options.candidateKernels ?? sweepCandidateKernelsId(rows, true),
    userAgent: "Chrome/155 test",
    adapter: options.adapter ?? ADAPTER,
    gpuTiming: { feature: true, unit: "ns", quantized: options.quantized ?? false },
    dp4a: true,
    settings: {
      candidateSet: "full",
      quick: false,
      ops: ["i8a8-conv3d"],
      cases: [caseId],
      rounds: 5,
      targetPassMs: 80,
      maxReps: 16384,
      warmupNs: 1,
      warmupMinRuns: 1,
      wallTimingNote: "note",
    },
    deviceLost: options.deviceLost ?? null,
    cases: [
      { caseId, elapsedMs: 120_000, defaultRepeat: { perDispatch: 20_000_000, driftRatio: 1.01 } },
    ],
    rows,
  };
};
