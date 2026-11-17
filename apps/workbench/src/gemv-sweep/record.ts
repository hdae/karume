/**
 * The GEMV sweep entry's record: the per-case records of `runGemvSweep` (one call per case, idle
 * time between them) merged into one `karume-gemv-sweep/3` record in the shape one call over the
 * planned cases writes. Pure: every fingerprint is a row's or a case's own, so nothing is
 * recomputed.
 */

import type { GemvSweepReport } from "@karume/runtime/tune";

import { type MergeOptions, mergePerCase } from "../per-case-sweep.ts";

/** The fields every per-case record of one sweep shares (anything else is a different sweep). */
const sharedFields = (report: GemvSweepReport) => {
  const { cases: _cases, ...settings } = report.settings;
  return {
    format: report.format,
    userAgent: report.userAgent,
    adapter: report.adapter,
    gates: report.gates,
    timingFeature: report.gpuTiming.feature,
    timingUnit: report.gpuTiming.unit,
    settings,
  };
};

/**
 * Merges the per-case records: rows, cases and summaries in run order, the envelope of
 * `mergePerCase` (`startedAt`, `date`, `aborted`, `deviceLost`), `settings.cases` = the planned
 * cases, and `gpuTiming.quantized` when any record's rounds looked quantized (the safe side, as for
 * the geometry sweep's merge).
 *
 * Throws when the records do not belong to one sweep (format, browser, adapter, gates, timing,
 * settings other than `cases`), when a record is not per-case, or when a case is unplanned or comes
 * twice.
 */
export const mergeGemvSweepReports = (
  reports: readonly GemvSweepReport[],
  options: MergeOptions,
): GemvSweepReport => {
  const { first, ...envelope } = mergePerCase(
    reports,
    options,
    sharedFields,
    "its format, browser, adapter, gates, timing or settings",
  );
  const rows = reports.flatMap((report) => report.rows);
  return {
    ...first,
    ...envelope,
    gpuTiming: {
      ...first.gpuTiming,
      quantized: reports.some((report) => report.gpuTiming.quantized),
    },
    settings: { ...first.settings, cases: [...options.plannedCaseIds] },
    cases: reports.flatMap((report) => report.cases),
    summaries: reports.flatMap((report) => report.summaries),
    rows,
  };
};
