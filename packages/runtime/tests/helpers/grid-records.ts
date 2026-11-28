/**
 * Hand-built records of the geometry sweep's `grid` set for the derivation of the grid-fill rule
 * (src/tune/gemm-grid-derive.ts) and the generator's tests: a `karume-geometry-sweep/4` record with
 * one row per (grid case, ladder tile), the default row the default profile's tile, every row's
 * rounds the given time per dispatch × reps × the record's round factors (aligned across the rows,
 * as the interleaved rounds are), and the current runtime's matching keys of the grid table. No GPU.
 *
 * `RECON_GRID_TIMES` holds, for every grid case, the recon of 2026-11-28's time of each ladder tile
 * (µs): today's tile on `nvidia-ampere` (the recon's reference) × the tile's median ratio over the
 * interleaved rounds (outputs/misc/gemm-geometry-recon-2026-11-28/final.tsv).
 */

import { DEFAULT_GEOMETRY_PROFILE, gemmShapeGeometry } from "../../src/kernels/geometry-profile.ts";
import {
  type BmmCase,
  caseShape,
  gemmCaseShape,
  GRID_CASES,
  type LinearCase,
} from "../../src/tune/cases.ts";
import {
  candidateSetCaseSetIds,
  candidateSetDefaultKernelsIds,
  sweepCandidateKernelsId,
} from "../../src/tune/fingerprint.ts";
import { GEMM_GRID_LADDER } from "../../src/tune/gemm-grid.ts";
import { gemmCandidate } from "../../src/tune/geometries.ts";
import { REPORT_FORMAT, type SweepRow } from "../../src/tune/report.ts";

export const GRID_GEMM_CASES: readonly (LinearCase | BmmCase)[] = GRID_CASES.filter(
  (sweepCase): sweepCase is LinearCase | BmmCase =>
    sweepCase.op === "linear" || sweepCase.op === "bmm",
);

/** The recon's times (µs) of the ladder's five tiles, largest first, per grid case. */
export const RECON_GRID_TIMES: Readonly<Record<string, readonly number[]>> = {
  "grid-linear-m1370-n384-k1536-wf32": [370.0, 179.82, 195.36, 276.39, 460.65],
  "grid-linear-m1370-n384-k384-wf32": [85.1, 46.63, 50.97, 70.46, 116.59],
  "grid-linear-m196-n768-k3072-wf32": [731.28, 260.8, 234.2, 250.63, 302.01],
  "grid-linear-m729-n1152-k1152-wf32": [294.3, 202.18, 226.91, 329.03, 573.0],
  "grid-linear-m729-n4304-k1152-wf32": [756.8, 710.64, 789.34, 1170.01, 2156.12],
  // final.tsv's 2,015 µs for 128 × 128 is a re-run whose reference hit a transient slow state (table.tsv:
  // its rounds spread ×0.25 to ×0.63); the recon's own reading, 64 × 32 at ×0.675 of today, gives 758.2
  "grid-linear-m1024-n768-k3072-wf32": [758.2, 535.99, 511.81, 795.93, 1422.59],
  "grid-linear-m4096-n3072-k768-wf32": [1676.3, 1676.3, 1931.1, 3097.8, 5841.91],
  "grid-linear-m17424-n384-k384-wf32": [500.0, 461.0, 511.0, 816.5, 1500.5],
  "grid-linear-m65536-n192-k768-wf32": [2061.8, 1673.1, 1934.71, 3062.28, 5706.79],
  "grid-linear-m256-n1536-k6144-wf32": [1507.16, 789.5, 671.07, 855.03, 1478.73],
  "grid-linear-m1024-n768-k3072-wf16": [729.9, 548.15, 518.23, 809.46, 1467.1],
  "grid-linear-m4096-n768-k3072-wf16": [1940.8, 1735.08, 2010.67, 3173.21, 5781.64],
  "grid-linear-m256-n512-k512-wbf16": [112.92, 38.2, 30.18, 32.93, 48.28],
  "grid-linear-m64-n2048-k512-wbf16": [109.39, 37.95, 30.07, 33.44, 47.5],
  "grid-linear-m64-n512-k512-wbf16": [108.96, 37.8, 29.66, 23.64, 19.8],
  "grid-linear-m768-n256-k1536-wbf16": [336.9, 116.9, 125.66, 124.99, 192.03],
  "grid-linear-m768-n6144-k1536-wbf16": [1301.2, 1346.74, 1505.49, 2426.74, 4472.22],
  "grid-linear-m768-n12288-k1536-wbf16": [2601.7, 2612.11, 3010.17, 4857.37, 8983.67],
  "grid-linear-m64-n3072-k1024-wbf16": [235.48, 83.77, 88.62, 86.84, 127.7],
  "grid-linear-m512-n1024-k1024-wbf16": [232.07, 123.8, 137.4, 189.47, 317.67],
  "grid-linear-m768-n2048-k1536-wi4g32": [636.6, 452.62, 525.19, 828.22, 1471.18],
  "grid-linear-m768-n2560-k10240-wi4g32": [4401.7, 3706.23, 4340.08, 6840.24, 12624.08],
  "grid-linear-m768-n512-k2560-wi4g32": [648.2, 340.31, 328.64, 385.68, 680.61],
  "grid-linear-m768-n15360-k3840-wi4g32": [7324.2, 8210.43, 9499.49, 15021.93, 27700.12],
  "grid-linear-m768-n128-k2816-wi4g32": [645.2, 241.95, 207.75, 218.08, 190.33],
  "grid-linear-m128-n1024-k1024-wi4g32": [221.36, 77.59, 63.7, 70.52, 97.46],
  "grid-bmm-b6-m1370-n64-k1370": [392.7, 169.65, 202.63, 228.16, 396.23],
  "grid-bmm-b6-m1370-n1370-k64": [309.5, 175.18, 194.99, 239.24, 432.37],
  "grid-bmm-b12-m196-n64-k196": [38.36, 13.3, 15.15, 12.94, 14.9],
  "grid-bmm-b16-m729-n72-k729": [304.5, 215.89, 184.83, 227.46, 357.79],
  "grid-bmm-b864-m144-n32-k144": [564.42, 202.3, 131.5, 148.49, 239.73],
  "grid-bmm-b16-m64-n64-k128": [23.92, 8.67, 7.08, 5.65, 5.9],
  "grid-bmm-b64-m512-n64-k512": [398.44, 177.21, 195.6, 263.08, 474.53],
};

export const GRID_ADAPTER = {
  vendor: "nvidia",
  architecture: "ampere",
  device: "0x2208",
  description: "NVIDIA test driver",
} as const;

/** Round factors of a steady record (spread ×1.006) and of one that is not (×1.2). */
export const STEADY_ROUNDS: readonly number[] = [1, 1.002, 1.004, 1.006, 1.001, 1.003, 1.005];
export const UNSTEADY_ROUNDS: readonly number[] = [1, 1.2, 1.1, 1.05, 1.15, 1.02, 1.08];

export type GridRecordOptions = {
  /** A tile's time per dispatch (µs) on a case; absent: the recon's. */
  readonly timeOf?: (caseId: string, step: number) => number;
  /** The rounds' factors (default {@link STEADY_ROUNDS}). */
  readonly rounds?: readonly number[];
  /** Whether a tile wrote the default's words (default: every one did). */
  readonly identical?: (caseId: string, step: number) => boolean;
  /** A tile whose row is a failure. */
  readonly failed?: (caseId: string, step: number) => boolean;
  readonly date?: string;
  readonly adapter?: Readonly<Record<string, string>>;
};

const REPS = 1000;

/** A grid record (the runtime's record, as the workbench's envelope holds it in `data`). */
export const gridRecord = (options: GridRecordOptions = {}): Record<string, unknown> => {
  const factors = options.rounds ?? STEADY_ROUNDS;
  const timeOf =
    options.timeOf ??
    ((caseId: string, step: number): number => {
      const times = RECON_GRID_TIMES[caseId];
      if (times === undefined) throw new Error(`no recon time for ${caseId}`);
      return times[step];
    });
  const rows: SweepRow[] = GRID_GEMM_CASES.flatMap((sweepCase) => {
    const defaultName = gemmCandidate(
      gemmShapeGeometry(DEFAULT_GEOMETRY_PROFILE, gemmCaseShape(sweepCase)),
    ).name;
    const ladder = GEMM_GRID_LADDER.map((geometry, step) => ({
      step,
      candidate: gemmCandidate(geometry),
    }));
    const ordered = [
      ...ladder.filter(({ candidate }) => candidate.name === defaultName),
      ...ladder.filter(({ candidate }) => candidate.name !== defaultName),
    ];
    const defaultTime = timeOf(sweepCase.id, ordered[0].step) * 1000;
    return ordered.map(({ step, candidate }, index): SweepRow => {
      const head = {
        caseId: sweepCase.id,
        op: sweepCase.op,
        shape: caseShape(sweepCase),
        censusCount: sweepCase.censusCount,
        geometry: candidate.name,
        geometryParams: candidate.geometry,
        isDefault: index === 0,
      };
      if (options.failed?.(sweepCase.id, step) === true) return { ...head, error: "test failure" };
      const perDispatch = timeOf(sweepCase.id, step) * 1000;
      const rounds = factors.map((factor) => perDispatch * REPS * factor);
      const least = Math.min(...rounds) / REPS;
      return {
        ...head,
        reps: REPS,
        rounds,
        perDispatch: least,
        speedupVsDefault: defaultTime / least,
        outputSha256: options.identical?.(sweepCase.id, step) === false ? "other" : "same",
        ...(index === 0
          ? {}
          : { identicalToDefault: options.identical?.(sweepCase.id, step) ?? true }),
      };
    });
  });
  return {
    format: REPORT_FORMAT,
    date: options.date ?? "2026-11-28T00:00:00.000Z",
    startedAt: "2026-11-28T00:00:00.000Z",
    aborted: false,
    caseSets: candidateSetCaseSetIds("grid", ["gemm"]),
    defaultKernels: candidateSetDefaultKernelsIds("grid", ["gemm"]),
    candidateKernels: sweepCandidateKernelsId(rows, false),
    userAgent: "Node.js/24",
    adapter: options.adapter ?? GRID_ADAPTER,
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    dp4a: false,
    settings: {
      candidateSet: "grid",
      quick: false,
      ops: ["linear", "bmm"],
      rounds: factors.length,
      quickLookMargin: 0,
      targetPassMs: 80,
      maxReps: 16384,
      warmupNs: 500e6,
      warmupMinRuns: 3,
      wallTimingNote: "",
    },
    deviceLost: null,
    cases: GRID_GEMM_CASES.map(({ id }) => ({ caseId: id, elapsedMs: 1, screened: 0 })),
    rows,
  };
};
