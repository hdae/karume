/**
 * The GEMV sweep entry's record: the per-case records of `runGemvSweep` (one call per case, idle
 * time between them) merged into one `karume-gemv-sweep/4` record in the shape one call over the
 * planned cases writes. Pure: every fingerprint is a row's or a case's own, so nothing is
 * recomputed but the record's class, which is its summaries' (`gemvSteadiness` over the merged
 * summaries — a per-case record holds a case's two or three groups, too few for a class).
 */

import { type GemvSweepReport, gemvSteadiness } from "@karume/runtime/tune";

import { mergedQuantized, type MergeOptions, mergePerCase } from "../per-case-sweep.ts";

/** The fields every per-case record of one sweep shares (anything else is a different sweep). */
const sharedFields = (report: GemvSweepReport) => {
  const { cases: _cases, ...settings } = report.settings;
  return {
    format: report.format,
    userAgent: report.userAgent,
    adapter: report.adapter,
    // One page load or process: the id, not the time since its start (each call's own)
    sessionId: report.session?.id,
    geometryProfile: report.geometryProfile,
    gates: report.gates,
    timingFeature: report.gpuTiming.feature,
    timingUnit: report.gpuTiming.unit,
    settings,
  };
};

/**
 * Merges the per-case records: rows, cases and summaries in run order, the envelope of
 * `mergePerCase` (`startedAt`, `date`, `aborted`, `deviceLost`), the first record's `session` (the
 * time since the session's start to the first case's sweep), the class of the merged summaries,
 * `settings.cases` = the planned cases, and `gpuTiming.quantized` when any record's rounds looked
 * quantized (the safe side, as for the geometry sweep's merge).
 *
 * Throws when the records do not belong to one sweep (format, browser, adapter, session, geometry
 * profile, gates, timing, settings other than `cases`), when a record is not per-case, or when a
 * case is unplanned or comes twice.
 */
export const mergeGemvSweepReports = (
  reports: readonly GemvSweepReport[],
  options: MergeOptions,
): GemvSweepReport => {
  const { first, ...envelope } = mergePerCase(
    reports,
    options,
    sharedFields,
    "its format, browser, adapter, session, geometry profile, gates, timing or settings",
  );
  const rows = reports.flatMap((report) => report.rows);
  const summaries = reports.flatMap((report) => report.summaries);
  return {
    ...first,
    ...envelope,
    steadiness: gemvSteadiness(summaries),
    gpuTiming: {
      ...first.gpuTiming,
      quantized: mergedQuantized(reports),
    },
    settings: { ...first.settings, cases: [...options.plannedCaseIds] },
    cases: reports.flatMap((report) => report.cases),
    summaries,
    rows,
  };
};
