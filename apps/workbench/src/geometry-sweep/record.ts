/**
 * The geometry sweep entry's record: the per-case records of `runGeometrySweep` (one call per case,
 * idle time between them) merged into one sweep record that `deriveGeometryProfile` reads as it is
 * (`karume-geometry-sweep/3` — no envelope of its own), and the per-case summary. Pure.
 */

import {
  type GeometryProfileFamilyIds,
  type GeometrySweepCaseSummary,
  type GeometrySweepReport,
  type GeometrySweepRow,
  sweepCandidateKernelsId,
} from "@karume/runtime/tune";

import { type MergeOptions, mergePerCase } from "../per-case-sweep.ts";

export type { MergeOptions };

/**
 * The fields every per-case record of one sweep shares (anything else is a different sweep). The
 * per-family keys `caseSets` / `defaultKernels` are not among them: each per-case record holds the
 * op family of its own case only ({@link unionFamilyKeys}).
 */
const sharedFields = (report: GeometrySweepReport) => {
  const { cases: _cases, ...settings } = report.settings;
  return {
    format: report.format,
    userAgent: report.userAgent,
    adapter: report.adapter,
    dp4a: report.dp4a,
    timingFeature: report.gpuTiming.feature,
    timingUnit: report.gpuTiming.unit,
    settings,
  };
};

/**
 * The union of the per-case records' per-family keys (`caseSets` or `defaultKernels` — one entry
 * per op family the record has a case of). Throws when two records give one family different values
 * (records of different runtimes or candidate sets, not one sweep).
 */
const unionFamilyKeys = (
  reports: readonly GeometrySweepReport[],
  key: "caseSets" | "defaultKernels",
): GeometryProfileFamilyIds => {
  const seen = new Map<string, { readonly id: string; readonly index: number }>();
  reports.forEach((report, index) => {
    for (const [family, id] of Object.entries(report[key])) {
      if (id === undefined) continue;
      const earlier = seen.get(family);
      if (earlier !== undefined && earlier.id !== id) {
        throw new Error(
          `Record ${index} differs from record ${earlier.index} in ${key}.${family} (${id} / ${earlier.id})`,
        );
      }
      seen.set(family, { id, index });
    }
  });
  return reports.reduce<GeometryProfileFamilyIds>(
    (merged, report) => ({ ...merged, ...report[key] }),
    {},
  );
};

/**
 * Merges the per-case records into one record in the shape one `runGeometrySweep` call over the
 * planned cases writes: rows and cases in run order, `startedAt` of the first and `date` of the
 * last, `settings.cases` = the planned cases, `candidateKernels` recomputed over the merged rows
 * (each record fingerprints only its own rows, and the derivation re-derives it from all the rows),
 * `caseSets` and `defaultKernels` the union of the records' (each holds its own case's op family).
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
 * Throws when the records do not belong to one sweep (format, a family's case set or default kernels,
 * browser, adapter, dp4a, timing, settings other than `cases`), when a record is not per-case, or
 * when a case is unplanned or comes twice.
 */
export const mergeSweepReports = (
  reports: readonly GeometrySweepReport[],
  options: MergeOptions,
): GeometrySweepReport => {
  const { first, ...envelope } = mergePerCase(
    reports,
    options,
    sharedFields,
    "its format, browser, adapter, dp4a, timing or settings",
  );
  const rows = reports.flatMap((report) => report.rows);
  return {
    ...first,
    caseSets: unionFamilyKeys(reports, "caseSets"),
    defaultKernels: unionFamilyKeys(reports, "defaultKernels"),
    ...envelope,
    candidateKernels: sweepCandidateKernelsId(rows, first.dp4a),
    gpuTiming: {
      ...first.gpuTiming,
      quantized: reports.some((report) => report.gpuTiming.quantized),
    },
    settings: { ...first.settings, cases: [...options.plannedCaseIds] },
    cases: reports.flatMap((report) => report.cases),
    rows,
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

/** Per case, in the order the rows came: one line of the entry's summary each. */
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
