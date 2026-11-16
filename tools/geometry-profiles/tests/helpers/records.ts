/**
 * Small hand-built records for the generator's tests: a `karume-geometry-sweep/3` record of one
 * linear case (the default geometry and one ×1.2 faster candidate with identical output) carrying
 * the matching keys of the current runtime, and the workbench envelope around a record. Built from
 * the public surface `@karume/runtime/tune` only, so the keys follow the runtime without a GPU.
 */

import {
  type GeometrySweepRow,
  sweepCandidateKernelsId,
  sweepCaseSetIds,
  sweepDefaultKernelsIds,
} from "@karume/runtime/tune";

/** A case of the sweep's table in the > 512 rows bucket of `gemmRows`. */
export const CASE_ID = "linear-m1024-n2048-k2048";

/** The runtime's default geometry of the bucket and the candidate the record makes faster. */
export const DEFAULT_GEOMETRY = { regM: 8, regN: 8, wgX: 16, wgY: 16 } as const;
export const FAST_GEOMETRY = { regM: 4, regN: 4, wgX: 8, wgY: 16 } as const;

export const ADAPTER = {
  vendor: "testvendor",
  architecture: "testarch",
  device: "0x1234",
  description: "Test GPU",
} as const;

export const CHROME =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36";

const row = (
  geometry: string,
  geometryParams: GeometrySweepRow["geometryParams"],
  isDefault: boolean,
  speedupVsDefault: number,
): GeometrySweepRow => ({
  caseId: CASE_ID,
  op: "linear",
  shape: "M1024 N2048 K2048",
  censusCount: 1,
  geometry,
  geometryParams,
  isDefault,
  reps: 100,
  rounds: [80_000_000, 80_100_000],
  speedupVsDefault,
  identicalToDefault: true,
});

/** One sweep record; `date` tells two records apart (their bytes, and so their sha256, differ). */
export const sweepRecord = (date: string, speedup = 1.2): Record<string, unknown> => {
  const rows = [
    row("reg128x128r8x8w16", DEFAULT_GEOMETRY, true, 1),
    row("reg64x32r4x4w8", FAST_GEOMETRY, false, speedup),
  ];
  const caseSet = sweepCaseSetIds().gemm;
  const defaultKernels = sweepDefaultKernelsIds().gemm;
  if (caseSet === undefined || defaultKernels === undefined) {
    throw new Error("the sweep's table has no gemm case");
  }
  return {
    format: "karume-geometry-sweep/3",
    date,
    startedAt: date,
    aborted: false,
    caseSets: { gemm: caseSet },
    defaultKernels: { gemm: defaultKernels },
    candidateKernels: sweepCandidateKernelsId(rows, true),
    userAgent: CHROME,
    adapter: ADAPTER,
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    dp4a: true,
    settings: { candidateSet: "full", quick: false, ops: ["linear"], cases: [CASE_ID] },
    deviceLost: null,
    cases: [{ caseId: CASE_ID, defaultRepeat: { perDispatch: 1_000_000, driftRatio: 1 } }],
    rows,
  };
};

/** A workbench record (`karume-workbench/1`) around `data`; without `data` it is a failed run. */
export const workbenchRecord = (
  entryId: string,
  data: unknown,
  options: { readonly outcome?: string; readonly dirty?: boolean } = {},
): Record<string, unknown> => ({
  format: "karume-workbench/1",
  entry: { id: entryId, parameters: {} },
  recordedAt: "2026-11-16T00:00:00.000Z",
  seconds: 1,
  outcome: options.outcome ?? "done",
  environment: {
    host: "browser",
    userAgent: CHROME,
    crossOriginIsolated: true,
    backend: { name: "d3d12", source: "declared" },
    adapter: ADAPTER,
  },
  checkout: {
    commit: "0123456789abcdef0123456789abcdef01234567",
    dirty: options.dirty ?? false,
    runtimeVersion: "0.0.0",
  },
  summary: [],
  ...(data === undefined ? {} : { data }),
});

/** The bytes of a record as a writer stores it. */
export const bytesOf = (record: unknown): Uint8Array =>
  new TextEncoder().encode(`${JSON.stringify(record, null, 1)}\n`);
