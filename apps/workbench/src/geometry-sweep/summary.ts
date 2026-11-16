/**
 * What a person reads of a geometry sweep run: the live status line, and after the run the totals
 * and one line per finished case — for the evaluation set `k88`, the K-88 summary instead. Pure,
 * over the same per-case records the record body is merged from.
 */

import {
  type GeometrySweepReport,
  type GeometrySweepTimingUnit,
  geometrySweepCases,
} from "@karume/runtime/tune";

import type { EntryResult } from "../entry.ts";
import { K88_SET, type K88Summary, k88Summary } from "./k88.ts";
import { type CaseSummaryRow, caseSummaryRows, mergeSweepReports } from "./record.ts";
import type { SweepLive, SweepRunOutcome, SweepStop } from "./run.ts";
import type { SweepPlan } from "./settings.ts";

/** ns (GPU time, or wall clock submit → completion) as ms; raw Deno ticks as they are. */
const formatPerDispatch = (
  value: number | undefined,
  unit: GeometrySweepTimingUnit | undefined,
): string =>
  value === undefined
    ? ""
    : unit === "ns"
      ? `${(value / 1e6).toFixed(3)} ms`
      : unit === "wall"
        ? `${(value / 1e6).toFixed(3)} ms wall`
        : `${value.toFixed(0)} ${unit ?? ""}`;

/** A ratio as `×1.234`, or empty. */
const ratioText = (value: number | undefined): string =>
  value === undefined ? "" : `×${value.toFixed(3)}`;

const joined = (parts: readonly (string | undefined)[]): string =>
  parts.filter((part) => part !== undefined && part !== "").join(" · ");

/**
 * The live status as one line: the cases done, then the case being swept (with its rows so far and
 * the sweep's own status — the geometry measured now) or the seconds left of the idle wait before the
 * next case.
 */
export const progressLine = (plan: SweepPlan, live: SweepLive, now: number): string => {
  const total = plan.caseIds.length;
  const position = `case ${live.caseIndex + 1} / ${total}`;
  const done = `${live.cases.length} / ${total} cases done`;
  if (live.deviceLost !== undefined) {
    return `device lost (${live.deviceLost.reason}): ${live.deviceLost.message} — the sweep stops after ${position}`;
  }
  const { phase } = live;
  switch (phase.kind) {
    case "waiting":
      return `starting ${total} case${total === 1 ? "" : "s"}`;
    case "idle":
      return `${done} · idle before ${position}: ${Math.max(0, (phase.until - now) / 1000).toFixed(0)} s left`;
    case "sweeping": {
      const caseId = plan.caseIds[live.caseIndex] ?? "";
      const rows = live.rows.filter((row) => row.caseId === caseId).length;
      return joined([
        done,
        `${position} ${caseId}`,
        `${rows} rows`,
        phase.status ?? "acquiring the device",
        live.unit === undefined ? undefined : `timing ${live.unit}, dp4a ${String(live.dp4a)}`,
      ]);
    }
  }
};

const stopText = (stop: SweepStop): string | undefined => {
  switch (stop.kind) {
    case "completed":
      return undefined;
    case "cancelled":
      return "stopped: cancelled";
    case "deviceLost":
      return `stopped: device lost (${stop.reason}): ${stop.message}`;
    case "error":
      return `stopped: ${stop.error instanceof Error ? stop.error.message : String(stop.error)}`;
  }
};

/**
 * Cases planned and finished, a case cut short (Cancel or a device loss stopped it midway: its rows so
 * far are in the record, and it is not a finished case), rows, failed rows, rows whose output
 * differed, and why the run stopped.
 */
const totalsLine = (
  plan: SweepPlan,
  reports: readonly GeometrySweepReport[],
  report: GeometrySweepReport | undefined,
  stop: SweepStop,
): string => {
  const rows = report?.rows ?? [];
  const cutShort = reports.filter((one) => one.aborted || one.deviceLost !== null).length;
  return joined([
    `${reports.length - cutShort} of ${plan.caseIds.length} planned cases finished`,
    cutShort === 0 ? undefined : `${cutShort} cut short`,
    `${rows.length} rows`,
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
    `K-88 against this adapter's table geometry (${summary.tableId}); a ratio is the table's time ÷ the variant's (above ×1 the variant is faster)`,
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

/** The entry's result: the merged record (absent when no case finished) and its summary. */
export type GeometrySweepResult = EntryResult & { readonly data: GeometrySweepReport | undefined };

/**
 * The result of a run's outcome: the per-case records merged into one `karume-geometry-sweep/3`
 * record (the cases that finished before a cancel, a device loss or an error included), the totals
 * line, and one line per finished case — the K-88 lines for the evaluation set. A run that a device
 * loss or an error stopped short of its plan has the verdict `incomplete`, so the record does not read
 * as a finished sweep (a cancelled run is `cancelled` in the record already). A run that measured
 * nothing because a sweep threw rethrows that error; a merge that fails throws.
 */
export const geometrySweepResult = (
  plan: SweepPlan,
  outcome: SweepRunOutcome,
): GeometrySweepResult => {
  const { reports, stop } = outcome;
  const verdict =
    stop.kind === "deviceLost" || stop.kind === "error" ? ({ verdict: "incomplete" } as const) : {};
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
      : caseSummaryRows(report.rows, report.cases).map((row) => caseLine(row, unit));
  return {
    ...verdict,
    summary: [totalsLine(plan, reports, report, stop), ...details],
    data: report,
  };
};
