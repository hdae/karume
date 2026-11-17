/**
 * What a person reads of a GEMV sweep run: the live status line, and after the run the totals and one
 * line per case and row count — the baseline's time and the spread of its rounds (how steady the
 * device was), the fastest identical candidate and the fastest reorder candidate with their speed
 * against the baseline, what the default profile chooses at each level with its share of the
 * fastest of its class's speed, and how many kernels the quick look screened out. Pure, over the
 * per-case records the record body is merged from.
 */

import type {
  GemvSweepCaseSummary,
  GemvSweepReport,
  GemvSweepRow,
  GeometrySweepTimingUnit,
} from "@karume/runtime/tune";

import type { EntryResult } from "../entry.ts";
import {
  cutShort,
  formatPerDispatch,
  joined,
  type PerCaseOutcome,
  perCaseProgressLine,
  perCaseVerdict,
  ratioText,
  stopText,
} from "../per-case-sweep.ts";
import { mergeGemvSweepReports } from "./record.ts";
import type { GemvLive } from "./run.ts";
import type { GemvSweepPlan } from "./settings.ts";

/** The live status as one line (the shared form, with the timing and the subgroup gate once started). */
export const gemvProgressLine = (plan: GemvSweepPlan, live: GemvLive, now: number): string =>
  perCaseProgressLine(
    plan.caseIds,
    live,
    now,
    live.unit === undefined
      ? undefined
      : `timing ${live.unit}, subgroups ${String(live.subgroups)}`,
  );

/** A row the check of its class failed: words off the baseline's, or outside the band. */
const wrong = (row: GemvSweepRow): boolean =>
  (row.class === "identical" && row.identicalToBaseline === false) ||
  (row.class === "reorder" && row.withinBand === false);

/** A row that passed its class's check and has a time. */
const passed = (row: GemvSweepRow): boolean =>
  row.error === undefined &&
  row.perDispatch !== undefined &&
  row.perDispatch > 0 &&
  (row.class === "identical" ? row.identicalToBaseline === true : row.withinBand === true);

/** The fastest of `rows` (the smallest time), or nothing. */
const fastest = (rows: readonly GemvSweepRow[]): GemvSweepRow | undefined =>
  rows.reduce<GemvSweepRow | undefined>(
    (best, row) =>
      row.perDispatch !== undefined &&
      (best?.perDispatch === undefined || row.perDispatch < best.perDispatch)
        ? row
        : best,
    undefined,
  );

/** A candidate's name and its speed against the baseline (`fastest identical c64u8 ×1.200`). */
const fastestText = (label: string, row: GemvSweepRow | undefined): string | undefined => {
  if (row === undefined) return undefined;
  const speed = ratioText(row.speedupVsBaseline);
  return `fastest ${label} ${row.candidate}${speed === "" ? "" : ` ${speed}`}`;
};

/**
 * The profile's choice at a level and its share of the fastest of its class's speed (the fastest
 * time ÷ the choice's: ×1.000 when the choice is the fastest). The baseline's class is the identical
 * one, the baseline included.
 */
const choiceText = (
  level: "exact" | "reorder",
  name: string,
  rows: readonly GemvSweepRow[],
  baseline: GemvSweepRow | undefined,
): string => {
  const row = rows.find((one) => one.candidate === name);
  if (row === undefined) return `${level} ${name} not measured`;
  if (row.error !== undefined) return `${level} ${name} failed`;
  if (!row.isBaseline && !passed(row)) {
    return `${level} ${name} ${row.class === "identical" ? "not identical" : "outside the band"}`;
  }
  const classRows = rows.filter(
    (one) =>
      one.class === row.class && (one.isBaseline ? baseline?.error === undefined : passed(one)),
  );
  const best = fastest(classRows);
  const share =
    best?.perDispatch !== undefined && row.perDispatch !== undefined && row.perDispatch > 0
      ? best.perDispatch / row.perDispatch
      : undefined;
  return share === undefined
    ? `${level} ${name}`
    : `${level} ${name} ${ratioText(share)} of the fastest ${row.class}`;
};

/** The line of one case at one row count. */
const caseLine = (
  summary: GemvSweepCaseSummary,
  rows: readonly GemvSweepRow[],
  unit: GeometrySweepTimingUnit,
): string => {
  const baseline = rows.find((row) => row.isBaseline);
  const candidates = rows.filter((row) => !row.isBaseline && passed(row));
  return `${summary.caseId} M=${summary.m}: ${joined([
    baseline === undefined
      ? undefined
      : baseline.error === undefined
        ? `baseline ${baseline.candidate} ${formatPerDispatch(baseline.perDispatch, unit)}`
        : `baseline ${baseline.candidate} failed: ${baseline.error}`,
    summary.baselineSpread === undefined
      ? undefined
      : `spread ${summary.baselineSpread.toFixed(3)}`,
    fastestText("identical", fastest(candidates.filter((row) => row.class === "identical"))),
    fastestText("reorder", fastest(candidates.filter((row) => row.class === "reorder"))),
    choiceText("exact", summary.profile.exact, rows, baseline),
    choiceText("reorder", summary.profile.reorder, rows, baseline),
    summary.screened === 0 ? undefined : `${summary.screened} screened`,
  ])}`;
};

/** One line per case and row count of the record, in its order. */
export const gemvCaseLines = (report: GemvSweepReport): string[] =>
  report.summaries.map((summary) =>
    caseLine(
      summary,
      report.rows.filter((row) => row.caseId === summary.caseId && row.m === summary.m),
      report.gpuTiming.unit,
    ),
  );

/**
 * Cases planned and finished, the cases cut short (Cancel or a device loss stopped them midway),
 * rows, the rows the quick look screened out (with its margin), failed rows, identical-class rows
 * whose words differ, reorder rows outside the band, and why the run stopped.
 */
const totalsLine = (
  plan: GemvSweepPlan,
  outcome: PerCaseOutcome<GemvSweepReport>,
  rows: readonly GemvSweepRow[],
): string => {
  const short = outcome.reports.filter(cutShort).length;
  return joined([
    `${outcome.reports.length - short} of ${plan.caseIds.length} planned cases finished`,
    short === 0 ? undefined : `${short} cut short`,
    `${rows.length} rows`,
    plan.quickLookMargin === 0
      ? "quick look off"
      : `${rows.filter((row) => row.screened === true).length} screened by the quick look (margin ×${plan.quickLookMargin})`,
    `${rows.filter((row) => row.error !== undefined).length} with an error`,
    `${rows.filter((row) => row.class === "identical" && row.identicalToBaseline === false).length} not identical to the baseline`,
    `${rows.filter((row) => row.class === "reorder" && row.withinBand === false).length} outside the band`,
    stopText(outcome.stop),
  ]);
};

/**
 * The run's verdict (`perCaseVerdict` — its order): a row of a finished case is wrong when it is in
 * the identical class and its words differ from the baseline's, or in the reorder class and outside
 * the band. Timing is not part of it.
 */
export const gemvSweepVerdict = (
  outcome: PerCaseOutcome<GemvSweepReport>,
): EntryResult["verdict"] =>
  perCaseVerdict(outcome.reports, outcome.stop, {
    wrong,
    errored: (row) => row.error !== undefined,
  });

/** The entry's result: the merged record (absent when no case finished) and its summary. */
export type GemvSweepResult = EntryResult & { readonly data: GemvSweepReport | undefined };

/**
 * The result of a run's outcome: the per-case records merged into one `karume-gemv-sweep/3` record
 * (the cases that finished before a cancel, a device loss or an error included), the totals line and
 * one line per case and row count, with the verdict of {@link gemvSweepVerdict}. A run that measured
 * nothing because a sweep threw rethrows that error; a merge that fails throws.
 */
export const gemvSweepResult = (
  plan: GemvSweepPlan,
  outcome: PerCaseOutcome<GemvSweepReport>,
): GemvSweepResult => {
  const { reports, stop } = outcome;
  const concluded = gemvSweepVerdict(outcome);
  const verdict = concluded === undefined ? {} : { verdict: concluded };
  if (reports.length === 0) {
    if (stop.kind === "error") throw stop.error;
    return { ...verdict, summary: [totalsLine(plan, outcome, [])], data: undefined };
  }
  const report = mergeGemvSweepReports(reports, {
    plannedCaseIds: plan.caseIds,
    cancelledBetweenCases: stop.kind === "cancelled" && reports.length < plan.caseIds.length,
  });
  return {
    ...verdict,
    summary: [totalsLine(plan, outcome, report.rows), ...gemvCaseLines(report)],
    data: report,
  };
};
