/**
 * Small hand-built records for the generator's tests: a `karume-geometry-sweep/4` record of one
 * linear case (the default geometry and one ×1.2 faster candidate with identical output) carrying
 * the matching keys of the current runtime, the same with candidates that carry a skeleton form and
 * conv2d's counter decode (perf-ledger K-88), and the workbench envelope around a record. Built from
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

/** A conv2d case of the sweep's table in the 64-row m-tile class, and its default geometry. */
export const CONV2D_CASE_ID = "conv2d-c192-256x256";
export const CONV2D_DEFAULT_GEOMETRY = { regM: 8, regN: 8, wgX: 16, wgY: 8 } as const;

type CaseRef = Pick<GeometrySweepRow, "caseId" | "op" | "shape">;

const LINEAR: CaseRef = { caseId: CASE_ID, op: "linear", shape: "M1024 N2048 K2048" };
const CONV2D: CaseRef = {
  caseId: CONV2D_CASE_ID,
  op: "conv2d",
  shape: "Cin192 Cout192 256x256 3x3",
};

const row = (
  geometry: string,
  geometryParams: GeometrySweepRow["geometryParams"],
  isDefault: boolean,
  speedupVsDefault: number,
  ref: CaseRef = LINEAR,
): GeometrySweepRow => ({
  ...ref,
  censusCount: 1,
  geometry,
  geometryParams,
  isDefault,
  reps: 100,
  rounds: [80_000_000, 80_100_000],
  speedupVsDefault,
  identicalToDefault: true,
});

/** The record around `rows`, with the current runtime's matching keys of `families`. */
const recordOf = (
  date: string,
  rows: readonly GeometrySweepRow[],
  families: readonly ("gemm" | "conv2d")[],
  adapter: Readonly<Record<string, string>> = ADAPTER,
): Record<string, unknown> => {
  const keys = (ids: ReturnType<typeof sweepCaseSetIds>): Record<string, string> =>
    Object.fromEntries(
      families.map((family) => {
        const id = ids[family];
        if (id === undefined) throw new Error(`the sweep's table has no ${family} case`);
        return [family, id];
      }),
    );
  const caseIds = [...new Set(rows.map((entry) => entry.caseId))];
  return {
    format: "karume-geometry-sweep/4",
    date,
    startedAt: date,
    aborted: false,
    caseSets: keys(sweepCaseSetIds()),
    defaultKernels: keys(sweepDefaultKernelsIds()),
    candidateKernels: sweepCandidateKernelsId(rows, true),
    userAgent: CHROME,
    adapter,
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    dp4a: true,
    settings: {
      candidateSet: "full",
      quick: false,
      ops: [...new Set(rows.map((entry) => entry.op))],
      cases: caseIds,
      quickLookMargin: 1.25,
    },
    deviceLost: null,
    cases: caseIds.map((caseId) => ({
      caseId,
      screened: 0,
      defaultRepeat: { perDispatch: 1_000_000, driftRatio: 1 },
    })),
    rows,
  };
};

/**
 * One sweep record; `date` tells two records apart (their bytes, and so their sha256, differ), and
 * `adapter` is the adapter it names (default {@link ADAPTER}).
 */
export const sweepRecord = (
  date: string,
  speedup = 1.2,
  adapter: Readonly<Record<string, string>> = ADAPTER,
): Record<string, unknown> =>
  recordOf(
    date,
    [
      row("reg128x128r8x8w16", DEFAULT_GEOMETRY, true, 1),
      row("reg64x32r4x4w8", FAST_GEOMETRY, false, speedup),
    ],
    ["gemm"],
    adapter,
  );

/** A second adapter of {@link ADAPTER}'s vendor and architecture (another model). */
export const OTHER_ADAPTER = { ...ADAPTER, description: "Other GPU" } as const;

/** The geometries the record of {@link fieldsSweepRecord} makes faster. */
export const FORM_GEOMETRY = { ...DEFAULT_GEOMETRY, form: "tb-un" } as const;
export const CONV2D_FIELDS_GEOMETRY = {
  ...CONV2D_DEFAULT_GEOMETRY,
  form: "tas-tb-un",
  decode: "counter",
} as const;

/**
 * A sweep record whose faster candidates carry the K-88 fields: the linear case's default tile with
 * the skeleton form `tb-un` (×1.2), and the conv2d case's default tile with the form `tas-tb-un`
 * and the counter decode (×1.3).
 */
export const fieldsSweepRecord = (date: string): Record<string, unknown> =>
  recordOf(
    date,
    [
      row("reg128x128r8x8w16", DEFAULT_GEOMETRY, true, 1),
      row("reg128x128r8x8w16:tb-un", FORM_GEOMETRY, false, 1.2),
      row("igemm64x128:wg16x8", CONV2D_DEFAULT_GEOMETRY, true, 1, CONV2D),
      row("igemm64x128:wg16x8:tas-tb-un:dec", CONV2D_FIELDS_GEOMETRY, false, 1.3, CONV2D),
    ],
    ["gemm", "conv2d"],
  );

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
