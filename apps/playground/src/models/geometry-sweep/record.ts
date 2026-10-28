/**
 * The geometry sweep page's record: the per-case records of `runGeometrySweep` (one call per case,
 * idle time between them) merged into one sweep record that `deriveGeometryProfile` reads as it is
 * (`karume-geometry-sweep/2` — no playground envelope), its text and file name, the table target
 * the page derives with, and the per-case summary the page shows. Pure.
 */

import {
  type GeometryProfileTarget,
  type GeometrySweepAdapter,
  type GeometrySweepCaseSummary,
  type GeometrySweepReport,
  type GeometrySweepRow,
  sweepCandidateKernelsId,
} from "@karume/runtime/tune";

import { fileStamp } from "@/lib/download";

/** The fields every per-case record of one sweep shares (anything else is a different sweep). */
const sharedFields = (report: GeometrySweepReport) => {
  const { cases: _cases, ...settings } = report.settings;
  return {
    format: report.format,
    caseSet: report.caseSet,
    defaultKernels: report.defaultKernels,
    userAgent: report.userAgent,
    adapter: report.adapter,
    dp4a: report.dp4a,
    timingFeature: report.gpuTiming.feature,
    timingUnit: report.gpuTiming.unit,
    settings,
  };
};

/** The one case a per-case record swept (`settings.cases` of a call with `cases: [id]`). */
const caseOf = (report: GeometrySweepReport, index: number): string => {
  const ids = report.settings.cases ?? [];
  if (ids.length !== 1) {
    throw new Error(
      `Record ${index} is not a per-case record (settings.cases is ${JSON.stringify(ids)})`,
    );
  }
  return ids[0];
};

export type MergeOptions = {
  /** Every case the sweep was planned to run, in order (`settings.cases` of the merged record). */
  readonly plannedCaseIds: readonly string[];
  /** Cancel stopped the sweep before every planned case had a record. */
  readonly cancelledBetweenCases: boolean;
};

/**
 * Merges the per-case records into one record in the shape one `runGeometrySweep` call over the
 * planned cases writes: rows and cases in run order, `startedAt` of the first and `date` of the
 * last, `settings.cases` = the planned cases, `candidateKernels` recomputed over the merged rows
 * (each record fingerprints only its own rows, and the derivation re-derives it from all the rows).
 *
 * - `aborted`: Cancel stopped a case midway or the sweep between cases (a sweep stopped by a device
 *   loss or an error has `aborted: false` and fewer cases than planned, as `aborted` means a cancel).
 * - `gpuTiming.quantized`: true when any record's rounds looked quantized. This differs from one
 *   call, which judges all the rows together (true only when every round of every row looks
 *   quantized): the runtime does not export that judgement, and a merged record with mixed states
 *   then takes the larger rounding step for its error bound, never the smaller (the safe side — more
 *   observations leave the ratio data).
 * - `deviceLost`: the first loss.
 *
 * Throws when the records do not belong to one sweep (format, case set, default kernels, browser,
 * adapter, dp4a, timing, settings other than `cases`), when a record is not per-case, or when a
 * case is unplanned or comes twice.
 */
export const mergeSweepReports = (
  reports: readonly GeometrySweepReport[],
  options: MergeOptions,
): GeometrySweepReport => {
  const [first] = reports;
  const last = reports.at(-1);
  if (first === undefined || last === undefined) throw new Error("No per-case record to merge");
  const shared = JSON.stringify(sharedFields(first));
  const seen = new Set<string>();
  reports.forEach((report, index) => {
    if (JSON.stringify(sharedFields(report)) !== shared) {
      throw new Error(
        `Record ${index} differs from record 0 in its format, case set, default kernels, browser, adapter, dp4a, timing or settings`,
      );
    }
    const caseId = caseOf(report, index);
    if (!options.plannedCaseIds.includes(caseId)) {
      throw new Error(`Record ${index} swept ${caseId}, which is not a planned case`);
    }
    if (seen.has(caseId)) throw new Error(`Case ${caseId} has two records`);
    seen.add(caseId);
  });
  const rows = reports.flatMap((report) => report.rows);
  return {
    ...first,
    date: last.date,
    startedAt: first.startedAt,
    aborted: options.cancelledBetweenCases || reports.some((report) => report.aborted),
    candidateKernels: sweepCandidateKernelsId(rows, first.dp4a),
    gpuTiming: {
      ...first.gpuTiming,
      quantized: reports.some((report) => report.gpuTiming.quantized),
    },
    settings: { ...first.settings, cases: [...options.plannedCaseIds] },
    deviceLost: reports.find((report) => report.deviceLost !== null)?.deviceLost ?? null,
    cases: reports.flatMap((report) => report.cases),
    rows,
  };
};

/** The bytes the page downloads, copies and hashes (one text, so the sha256 names the file). */
export const sweepRecordText = (report: GeometrySweepReport): string =>
  `${JSON.stringify(report, null, 2)}\n`;

/** Lower-case words joined by `-`, safe in file names (`NVIDIA GeForce RTX 5070 Ti` → `nvidia-geforce-rtx-5070-ti`). */
const slug = (text: string): string =>
  text
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "-")
    .replaceAll(/^-+|-+$/g, "");

/**
 * The adapter in a file name: its description, else vendor, architecture and device (Chrome without
 * developer flags may report an empty description).
 */
const adapterSlug = (adapter: GeometrySweepAdapter): string => {
  const described = slug(adapter.description);
  if (described !== "") return described;
  const fields = slug([adapter.vendor, adapter.architecture, adapter.device].join(" "));
  return fields === "" ? "unknown-adapter" : fields;
};

/**
 * `geometry-sweep_<adapter>_<candidate set>_<UTC timestamp of the record's date>.json`. The candidate
 * set is in the name so a quick dry run is never mistaken for the full record.
 */
export const sweepFileName = (report: GeometrySweepReport): string =>
  `geometry-sweep_${adapterSlug(report.adapter)}_${report.settings.candidateSet.replace("+", "-plus")}_${fileStamp(new Date(report.date))}.json`;

/**
 * The table the page derives from its own record (to show the verdicts before the file is sent):
 * `<vendor>-<architecture>` matching the adapter's vendor and architecture, the id form of the
 * built-in tables (`nvidia-blackwell`). An adapter without a vendor gets an injection-only table.
 */
export const deriveTarget = (
  adapter: GeometrySweepAdapter,
): { readonly id: string } & GeometryProfileTarget => {
  if (adapter.vendor === "") return { id: "in-app", optIn: true };
  if (adapter.architecture === "") return { id: adapter.vendor, vendor: adapter.vendor };
  return {
    id: `${adapter.vendor}-${adapter.architecture}`,
    vendor: adapter.vendor,
    architecture: adapter.architecture,
  };
};

/** One case of the summary table. */
export type CaseSummaryRow = {
  readonly caseId: string;
  /** Rows measured so far (the default and every candidate, failures included). */
  readonly rows: number;
  /** The default geometry's time per dispatch, in the record's unit. */
  readonly defaultPerDispatch?: number;
  /** The default re-measured at the end of the case ÷ the first (absent until the case is done). */
  readonly driftRatio?: number;
  /** Why the re-measurement failed. */
  readonly driftError?: string;
  /** The fastest candidate whose output matched the default bit for bit. */
  readonly fastest?: { readonly geometry: string; readonly speedup: number };
  readonly errors: number;
  /** Candidates whose output differed from the default (never adopted). */
  readonly mismatches: number;
  /** The case's wall seconds (absent until the case is done). */
  readonly seconds?: number;
};

/** Per case, in the order the rows came: what the page's table shows (live, and for a record). */
export const caseSummaryRows = (
  rows: readonly GeometrySweepRow[],
  cases: readonly GeometrySweepCaseSummary[],
): CaseSummaryRow[] => {
  const byCase = new Map<string, GeometrySweepRow[]>();
  for (const row of rows) byCase.set(row.caseId, [...(byCase.get(row.caseId) ?? []), row]);
  return [...byCase].map(([caseId, caseRows]) => {
    const summary = cases.find((entry) => entry.caseId === caseId);
    const defaultRow = caseRows.find((row) => row.isDefault && row.error === undefined);
    const fastest = caseRows
      .filter((row) => !row.isDefault && row.error === undefined && row.identicalToDefault === true)
      .reduce<GeometrySweepRow | undefined>(
        (best, row) =>
          row.speedupVsDefault !== undefined &&
          (best?.speedupVsDefault === undefined || row.speedupVsDefault > best.speedupVsDefault)
            ? row
            : best,
        undefined,
      );
    return {
      caseId,
      rows: caseRows.length,
      ...(defaultRow?.perDispatch === undefined
        ? {}
        : { defaultPerDispatch: defaultRow.perDispatch }),
      ...(summary?.defaultRepeat === undefined
        ? {}
        : { driftRatio: summary.defaultRepeat.driftRatio }),
      ...(summary?.defaultRepeatError === undefined
        ? {}
        : { driftError: summary.defaultRepeatError }),
      ...(fastest?.speedupVsDefault === undefined
        ? {}
        : { fastest: { geometry: fastest.geometry, speedup: fastest.speedupVsDefault } }),
      errors: caseRows.filter((row) => row.error !== undefined).length,
      mismatches: caseRows.filter((row) => row.identicalToDefault === false).length,
      ...(summary === undefined ? {} : { seconds: summary.elapsedMs / 1000 }),
    };
  });
};
