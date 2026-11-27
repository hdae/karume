/**
 * The optimisable patterns a run found, as the page counts them for the person who runs it: read by
 * the derivations' own threshold, predicates and exclusions (`@karume/runtime/tune`), over the
 * records of the cases that finished. They are candidates from one run: a table takes a pattern
 * only once a second run on the same machine agrees, which this count does not check. Pure.
 *
 * SPEC: docs/spec/benchmark.md#the-patterns-found
 */

import {
  GEMV_MIN_SPEEDUP,
  type GemvSweepCaseSummary,
  type GemvSweepRow,
  GEOMETRY_MIN_SPEEDUP,
  gemvClassReference,
  gemvGroupRole,
  gemvReferenceRefusal,
  gemvRowReading,
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

/**
 * Whether a group holds a pattern for one class: a candidate the derivation could adopt there — the
 * group's role for the class is `winner` (`gemvGroupRole`) and the row is one a table holds
 * (`gemvTableCandidate`) — that reads as evidence (`gemvRowReading`: timed, its check held) at least
 * `GEMV_MIN_SPEEDUP` faster than the class's reference (`gemvClassReference`, the current default
 * profile's choice), whose row in the group is one (`gemvReferenceRefusal`). The reference's time
 * is the group's own: one record has no second one to repeat it.
 */
const groupHolds = (
  summary: GemvSweepCaseSummary,
  rows: readonly GemvSweepRow[],
  candidateClass: GemvSweepRow["class"],
): boolean => {
  if (gemvGroupRole(summary.m, candidateClass) !== "winner") return false;
  const name = gemvClassReference(summary.caseId, summary.m, candidateClass);
  const reference = rows.find((row) => row.candidate === name);
  if (reference === undefined || gemvReferenceRefusal(reference, candidateClass) !== undefined) {
    return false;
  }
  return rows.some((row) => {
    if (row.class !== candidateClass || !gemvTableCandidate(row)) return false;
    const reading = gemvRowReading(row, candidateClass, reference.perDispatch);
    return reading.kind === "evidence" && reading.gain >= GEMV_MIN_SPEEDUP;
  });
};

/**
 * The GEMV sweep's count: of the (case, row count) groups of `summaries` at which the derivation
 * picks a winner (`gemvGroupRole`'s `winner`: M = 1), those in which a candidate a table could hold
 * beat its class's reference by at least the derivation's threshold with its check held.
 */
export const gemvPatterns = (
  summaries: readonly GemvSweepCaseSummary[],
  rows: readonly GemvSweepRow[],
): PatternCount => {
  const read = summaries.filter(
    (summary) =>
      gemvGroupRole(summary.m, "identical") === "winner" ||
      gemvGroupRole(summary.m, "reorder") === "winner",
  );
  const found = read.filter((summary) => {
    const group = rows.filter((row) => row.caseId === summary.caseId && row.m === summary.m);
    return groupHolds(summary, group, "identical") || groupHolds(summary, group, "reorder");
  });
  return { found: found.length, of: read.length };
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
