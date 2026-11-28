/**
 * What a person reads of a geometry sweep run: the live status line, and after the run the totals
 * and one line per finished case — for the evaluation set `k88`, the K-88 summary instead, and for
 * the `grid` set the grid summary (./grid.ts). Pure,
 * over the same per-case records the record body is merged from.
 */

import {
  type GeometrySweepReport,
  type GeometrySweepTimingUnit,
  geometrySweepCases,
} from "@karume/runtime/tune";

import type { EntryResult } from "../record.ts";
import {
  cutShort,
  formatPerDispatch,
  joined,
  perCaseProgressLine,
  perCaseVerdict,
  ratioText,
  stopText,
} from "../per-case-sweep.ts";
import { gridLines } from "./grid.ts";
import { K88_SET, type K88Summary, k88Summary } from "./k88.ts";
import { type CaseSummaryRow, caseSummaryRows, mergeSweepReports } from "./record.ts";
import type { SweepLive, SweepRunOutcome, SweepStop } from "./run.ts";
import { GRID_SET, type SweepPlan } from "./settings.ts";

/**
 * The live status as one line: the cases done, then the case being swept (with its rows so far, the
 * sweep's own status — the geometry measured now — and the timing once it started) or the seconds left
 * of the idle wait before the next case.
 */
export const progressLine = (plan: SweepPlan, live: SweepLive, now: number): string =>
  perCaseProgressLine(
    plan.caseIds,
    live,
    now,
    live.unit === undefined ? undefined : `timing ${live.unit}, dp4a ${String(live.dp4a)}`,
  );

/**
 * Cases planned and finished, a case cut short (Cancel or a device loss stopped it midway: its rows so
 * far are in the record, and it is not a finished case), rows, the rows the quick look screened out
 * (with its margin), for the `explore` set the rows skipped because an earlier case of their class
 * dropped the geometry, failed rows, rows whose output differed, and why the run stopped.
 */
const totalsLine = (
  plan: SweepPlan,
  reports: readonly GeometrySweepReport[],
  report: GeometrySweepReport | undefined,
  stop: SweepStop,
): string => {
  const rows = report?.rows ?? [];
  const short = reports.filter(cutShort).length;
  return joined([
    `${reports.length - short} of ${plan.caseIds.length} planned cases finished`,
    short === 0 ? undefined : `${short} cut short`,
    `${rows.length} rows`,
    plan.quickLookMargin === 0
      ? "quick look off"
      : `${rows.filter((row) => row.screened === true).length} screened by the quick look (margin ×${plan.quickLookMargin})`,
    plan.candidateSet === "explore"
      ? `${rows.filter((row) => row.skipped === true).length} skipped as dropped for their class`
      : undefined,
    `${rows.filter((row) => row.error !== undefined).length} with an error`,
    `${rows.filter((row) => row.identicalToDefault === false).length} not identical to the default`,
    stopText(stop),
  ]);
};

const caseLine = (row: CaseSummaryRow, unit: GeometrySweepTimingUnit): string =>
  `${row.caseId}: ${joined([
    `${row.rows} rows`,
    row.defaultPerDispatch === undefined
      ? undefined
      : `default ${formatPerDispatch(row.defaultPerDispatch, unit)}`,
    row.driftRatio !== undefined
      ? `drift ${row.driftRatio.toFixed(3)}`
      : row.driftError === undefined
        ? undefined
        : `drift failed: ${row.driftError}`,
    row.fastest === undefined
      ? undefined
      : `fastest identical ${row.fastest.geometry} ×${row.fastest.speedup.toFixed(3)}`,
    `${row.errors} errors`,
    `${row.mismatches} mismatches`,
    row.seconds === undefined ? undefined : `${row.seconds.toFixed(0)} s`,
  ])}`;

/**
 * The K-88 summary as lines: per unit the census-weighted sums of the table geometry and of its best
 * form, decode and both (each with its ratio to the table's), then per case the table geometry and
 * each variant's ratio (table time ÷ variant time: above ×1 the variant is faster).
 */
const k88Lines = (summary: K88Summary, unit: GeometrySweepTimingUnit): string[] => {
  const sumText = (label: string, value: number | undefined, table: number | undefined) =>
    value === undefined
      ? undefined
      : `${label} ${formatPerDispatch(value, unit)}${table === undefined ? "" : ` (${ratioText(table / value)})`}`;
  return [
    `K-88 against the tile of this adapter's table geometry (${summary.tableId}, without the fields); a ratio is the tile's time ÷ the variant's (above ×1 the variant is faster)`,
    ...summary.sums.map((entry) =>
      entry.measured < entry.planned
        ? `${entry.unit}: ${entry.measured} / ${entry.planned} cases, no sum`
        : `${entry.unit}: ${joined([
            `${entry.nodes} nodes`,
            entry.table === undefined ? undefined : `table ${formatPerDispatch(entry.table, unit)}`,
            sumText("best form", entry.bestForm, entry.table),
            sumText("decode", entry.decode, entry.table),
            sumText("both", entry.bestBoth, entry.table),
          ])}`,
    ),
    ...summary.cases.map(
      (entry) =>
        `${entry.caseId} (${entry.censusCount === 0 ? "guard" : `×${entry.censusCount}`}): ${joined(
          [
            entry.table === undefined
              ? undefined
              : `table ${entry.table.geometry} ${formatPerDispatch(entry.table.perDispatch, unit)}`,
            entry.formVsTable === undefined
              ? undefined
              : `best form ÷ table ${ratioText(entry.formVsTable)}`,
            entry.decodeVsTable === undefined
              ? undefined
              : `decode ÷ table ${ratioText(entry.decodeVsTable)}`,
            entry.bothVsTable === undefined
              ? undefined
              : `both ÷ table ${ratioText(entry.bothVsTable)}`,
            `identical ${entry.identical ? "yes" : "no"}`,
            `${entry.mismatches} mismatches`,
            `${entry.errors} errors`,
          ],
        )}`,
    ),
  ];
};

/**
 * The run's verdict (`perCaseVerdict` — its order): a row of a finished case that is not identical to
 * its case's default is wrong. No candidate of any set is allowed to differ from the default: a match
 * is the sweep's own expectation (f32 by the K reduction order, i8a8 by exact integer arithmetic).
 */
const sweepVerdict = (
  reports: readonly GeometrySweepReport[],
  stop: SweepStop,
): EntryResult["verdict"] =>
  perCaseVerdict(reports, stop, {
    wrong: (row) => row.identicalToDefault === false,
    errored: (row) => row.error !== undefined,
  });

/** The entry's result: the merged record (absent when no case finished) and its summary. */
export type GeometrySweepResult = EntryResult & { readonly data: GeometrySweepReport | undefined };

/**
 * The result of a run's outcome: the per-case records merged into one `karume-geometry-sweep/4`
 * record (the cases that finished before a cancel, a device loss or an error included), the totals
 * line, and one line per finished case — the K-88 lines for the evaluation set — with the verdict
 * of {@link sweepVerdict}. A run that measured nothing because a sweep threw rethrows that error; a
 * merge that fails throws.
 */
export const geometrySweepResult = (
  plan: SweepPlan,
  outcome: SweepRunOutcome,
): GeometrySweepResult => {
  const { reports, stop } = outcome;
  const concluded = sweepVerdict(reports, stop);
  const verdict = concluded === undefined ? {} : { verdict: concluded };
  if (reports.length === 0) {
    if (stop.kind === "error") throw stop.error;
    return { ...verdict, summary: [totalsLine(plan, reports, undefined, stop)], data: undefined };
  }
  const report = mergeSweepReports(reports, {
    plannedCaseIds: plan.caseIds,
    cancelledBetweenCases: stop.kind === "cancelled" && reports.length < plan.caseIds.length,
  });
  const unit = report.gpuTiming.unit;
  const details =
    report.settings.candidateSet === K88_SET
      ? k88Lines(k88Summary(report, geometrySweepCases(K88_SET)), unit)
      : report.settings.candidateSet === GRID_SET
        ? gridLines(report)
        : caseSummaryRows(report.rows, report.cases).map((row) => caseLine(row, unit));
  return {
    ...verdict,
    summary: [totalsLine(plan, reports, report, stop), ...details],
    data: report,
  };
};
