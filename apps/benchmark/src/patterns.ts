/**
 * The optimisable patterns a run found, as the page counts them for the person who runs it: read by
 * the derivations' own rules, predicates and exclusions (`@karume/runtime/tune`), over the records
 * of the cases that finished. The GEMV count is the rule of one steady record's at M = 1, at the
 * default margin and at the lowest one side by side, with the run's class beside it: on a steady run
 * a table can take them from this run alone; on an unsteady one only after a second run agrees. The
 * geometry count is a candidate from one run. Pure.
 *
 * SPEC: docs/spec/benchmark.md#the-patterns-found
 */

import {
  GEMV_LOWEST_MARGIN,
  GEMV_MARGIN,
  type GemvSweepCaseSummary,
  type GemvSweepRow,
  GEOMETRY_MIN_SPEEDUP,
  gemvClassReference,
  gemvGroupRole,
  gemvReadingIsEvidence,
  gemvSteadyReading,
  gemvTableCandidate,
  geometryCaseClass,
  geometryCaseExclusion,
  geometryRowExclusion,
  type GeometrySweepReport,
  type GeometrySweepRow,
  geometryTimestampQuantum,
} from "@karume/runtime/tune";
import { mergedQuantized } from "@karume/sweep-app-support/per-case-sweep";

/** How many of the groups (or classes) a count read hold a pattern. */
export type PatternCount = {
  readonly found: number;
  /** The groups (GEMV) or classes (geometry) read. */
  readonly of: number;
};

/** The GEMV count: the groups that hold a pattern at the default margin and at the lowest one. */
export type GemvPatternCount = {
  /** At gemv-rule's `GEMV_MARGIN` (×1.065). */
  readonly atMargin: number;
  /** At `GEMV_LOWEST_MARGIN` (×1.03), the alternative the maintainer weighs. */
  readonly atLowest: number;
  /** The groups read. */
  readonly of: number;
};

/**
 * The largest margin at which a group holds a pattern for one class (0 when it holds none): over the
 * candidates the derivation could adopt there — the group's role for the class is `winner`
 * (`gemvGroupRole`) and the row is one a table holds (`gemvTableCandidate`) —, read against the
 * class's reference (`gemvClassReference`, the current default profile's choice) in the same group
 * by the rule of one steady record (`gemvSteadyReading`: the reference's row is one, the candidate's
 * check held, both rows long enough and with a gap), each reading's break-even margin, where its
 * reference's re-timing lets it be evidence at that margin (`gemvReadingIsEvidence`).
 */
const groupBreakEven = (
  summary: GemvSweepCaseSummary,
  rows: readonly GemvSweepRow[],
  candidateClass: GemvSweepRow["class"],
  unit: string,
): ((margin: number) => boolean) => {
  if (gemvGroupRole(summary.m, candidateClass) !== "winner") return () => false;
  const name = gemvClassReference(summary.caseId, summary.m, candidateClass);
  const reference = rows.find((row) => row.candidate === name);
  if (reference === undefined) return () => false;
  const readings = rows.flatMap((row) => {
    if (row.class !== candidateClass || !gemvTableCandidate(row)) return [];
    const reading = gemvSteadyReading(row, reference, candidateClass, unit);
    return reading === undefined ? [] : [reading];
  });
  return (margin) =>
    readings.some(
      (reading) => reading.breakEven >= margin && gemvReadingIsEvidence(reading, margin),
    );
};

/**
 * The GEMV sweep's count: of the (case, row count) groups of `summaries` at which the derivation
 * picks a winner (`gemvGroupRole`'s `winner`: M = 1), those in which a candidate a table could hold
 * beat its class's reference by the rule of one steady record — at ×1.065 and at ×1.03 —, in a
 * record timed in `unit`.
 */
export const gemvPatterns = (
  summaries: readonly GemvSweepCaseSummary[],
  rows: readonly GemvSweepRow[],
  unit: string,
): GemvPatternCount => {
  const read = summaries.filter(
    (summary) =>
      gemvGroupRole(summary.m, "identical") === "winner" ||
      gemvGroupRole(summary.m, "reorder") === "winner",
  );
  const holds = read.map((summary) => {
    const group = rows.filter((row) => row.caseId === summary.caseId && row.m === summary.m);
    const identical = groupBreakEven(summary, group, "identical", unit);
    const reorder = groupBreakEven(summary, group, "reorder", unit);
    return (margin: number) => identical(margin) || reorder(margin);
  });
  return {
    atMargin: holds.filter((holdsAt) => holdsAt(GEMV_MARGIN)).length,
    atLowest: holds.filter((holdsAt) => holdsAt(GEMV_LOWEST_MARGIN)).length,
    of: read.length,
  };
};

/** A candidate's row that is ahead of the default by the derivation's threshold, its output the default's. */
const ahead = (row: GeometrySweepRow): boolean =>
  !row.isDefault &&
  row.error === undefined &&
  row.identicalToDefault === true &&
  row.speedupVsDefault !== undefined &&
  row.speedupVsDefault >= GEOMETRY_MIN_SPEEDUP;

/**
 * The geometries of one case that the derivation could take from it: none when it drops the case
 * from the ratio material (`geometryCaseExclusion`: the default's re-measurement, the default row's
 * rounding), otherwise those ahead whose observation it keeps (`geometryRowExclusion`).
 */
const caseGeometries = (
  caseRows: readonly GeometrySweepRow[],
  summaries: GeometrySweepReport["cases"],
  timestampQuantum: number,
): Set<string> => {
  const defaultRow = caseRows.find((row) => row.isDefault);
  const [first] = caseRows;
  const summary = summaries.find((entry) => entry.caseId === first?.caseId);
  if (
    defaultRow === undefined ||
    geometryCaseExclusion(summary, defaultRow, timestampQuantum) !== undefined
  ) {
    return new Set();
  }
  return new Set(
    caseRows
      .filter(
        (row) =>
          ahead(row) && geometryRowExclusion(row, defaultRow, timestampQuantum) === undefined,
      )
      .map((row) => row.geometry),
  );
};

/**
 * The geometry sweep's count over the per-case records of the finished cases: of the classes
 * (`geometryCaseClass`, the field the derivation reads a case in) the records hold a case of, those
 * in which one candidate is ahead of the default by `GEOMETRY_MIN_SPEEDUP` with the default's output
 * in every case measured so far, its observation kept as ratio material there — what the derivation
 * asks of a candidate in every case of the class. The rounding step is the one of the record the
 * records merge into (`mergedQuantized`).
 */
export const geometryPatterns = (reports: readonly GeometrySweepReport[]): PatternCount => {
  const timestampQuantum = geometryTimestampQuantum(mergedQuantized(reports));
  const summaries = reports.flatMap((report) => report.cases);
  const classes = new Map<string, Map<string, GeometrySweepRow[]>>();
  for (const row of reports.flatMap((report) => report.rows)) {
    const caseClass = geometryCaseClass(row.caseId);
    const cases = classes.get(caseClass) ?? new Map<string, GeometrySweepRow[]>();
    cases.set(row.caseId, [...(cases.get(row.caseId) ?? []), row]);
    classes.set(caseClass, cases);
  }
  let found = 0;
  for (const cases of classes.values()) {
    const [first, ...others] = [...cases.values()].map((caseRows) =>
      caseGeometries(caseRows, summaries, timestampQuantum),
    );
    if (
      first !== undefined &&
      [...first].some((geometry) => others.every((set) => set.has(geometry)))
    ) {
      found += 1;
    }
  }
  return { found, of: classes.size };
};
