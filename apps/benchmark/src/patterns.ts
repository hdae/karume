/**
 * The optimisable patterns a run found, as the page counts them for the person who runs it: read by
 * the derivations' own rules, predicates and exclusions (`@karume/runtime/tune`), over the records
 * of the cases that finished. The GEMV count is the rule of one steady record's at M = 1 — against
 * the device table's choice where the record names it, as the derivation keeps a table's choice —,
 * at the default margin and at the lowest one side by side, with the run's class beside it: on a
 * steady run a table can take them from this run alone; on an unsteady one only after a second run
 * agrees. The
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
  type GemvPairReading,
  gemvReadingIsEvidence,
  gemvRowReading,
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

/** Whether a reading clears the bar at `margin` and is evidence there (its reference's re-timing). */
const clears = (reading: GemvPairReading | undefined, margin: number): boolean =>
  reading !== undefined && reading.breakEven >= margin && gemvReadingIsEvidence(reading, margin);

/**
 * Whether a group holds a pattern for one class at a margin: a choice the derivation would write in
 * place of the one the device's table holds. Over the candidates the derivation could adopt there —
 * the group's role for the class is `winner` (`gemvGroupRole`) and the row is one a table holds
 * (`gemvTableCandidate`) —, read by the rule of one steady record (`gemvSteadyReading`: the
 * reference's row is one, the candidate's check held, both rows long enough and with a gap; a
 * reading clears its bar when its break-even margin is at least the margin and its reference's
 * re-timing lets it be evidence there, `gemvReadingIsEvidence`):
 *
 * - against the class's reference (`gemvClassReference`, the current default profile's choice) in
 *   the same group, where the record names no table choice for the class (`summary.table`, a record
 *   written before it), or names the default's, or names one whose row failed or whose check failed
 *   (the derivation then beats the default's choice);
 * - otherwise against the table's choice, as the derivation keeps it: a candidate that clears its
 *   bar over the default's choice and over the table's, or the default's choice that clears it over
 *   the table's; none where the record has no time of the table's choice (a record that did not
 *   time it does not move it).
 */
const groupHolds = (
  summary: GemvSweepCaseSummary,
  rows: readonly GemvSweepRow[],
  candidateClass: GemvSweepRow["class"],
  unit: string,
): ((margin: number) => boolean) => {
  if (gemvGroupRole(summary.m, candidateClass) !== "winner") return () => false;
  const name = gemvClassReference(summary.caseId, summary.m, candidateClass);
  const reference = rows.find((row) => row.candidate === name);
  if (reference === undefined) return () => false;
  const candidates = rows.filter(
    (row) => row.class === candidateClass && gemvTableCandidate(row) && row.candidate !== name,
  );
  const overDefault = candidates.map((row) => ({
    row,
    reading: gemvSteadyReading(row, reference, candidateClass, unit),
  }));
  const tableName = summary.table?.[candidateClass === "identical" ? "exact" : "reorder"];
  const tableRow =
    tableName === undefined || tableName === name
      ? undefined
      : rows.find((row) => row.candidate === tableName);
  // The table's choice is no longer held when its row failed or its check (of its own class) did
  const check =
    tableRow === undefined
      ? undefined
      : gemvRowReading(tableRow, tableRow.class, tableRow.perDispatch).kind;
  const held = check !== "failed" && check !== "unshown" && check !== "outside-band";
  if (tableName === undefined || tableName === name || !held) {
    return (margin) => overDefault.some(({ reading }) => clears(reading, margin));
  }
  if (tableRow === undefined) return () => false;
  const overTable = (row: GemvSweepRow) => gemvSteadyReading(row, tableRow, candidateClass, unit);
  const byDefault = overTable(reference);
  const byCandidates = overDefault
    .filter(({ row }) => row.candidate !== tableName)
    .map(({ row, reading }) => ({ reading, againstTable: overTable(row) }));
  return (margin) =>
    clears(byDefault, margin) ||
    byCandidates.some(
      ({ reading, againstTable }) => clears(reading, margin) && clears(againstTable, margin),
    );
};

/**
 * The GEMV sweep's count: of the (case, row count) groups of `summaries` at which the derivation
 * picks a winner (`gemvGroupRole`'s `winner`: M = 1), those that hold a pattern for a class
 * ({@link groupHolds}: a choice the rule of one steady record would write in place of the device
 * table's, or of the default's where the record names no table choice) — at ×1.065 and at ×1.03 —,
 * in a record timed in `unit`.
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
    const identical = groupHolds(summary, group, "identical", unit);
    const reorder = groupHolds(summary, group, "reorder", unit);
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
