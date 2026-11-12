/**
 * The K-85 summary of a sweep record of the evaluation set `k85` (perf-ledger K-85 — the vec4
 * skeleton of the i8a8 linear): per case, this adapter's table geometry against the fastest
 * skeleton candidate whose output matched the default bit for bit, and the census-weighted sum over
 * one Wan2.2 TI2V-5B 33f DiT forward (each case's `censusCount` × its time per dispatch), so one
 * number answers what the forward would gain. Pure.
 *
 * "This adapter's table" is the embedded table the record names (`geometryProfile` — the one a
 * Session on that adapter selects); a record without the field, or naming the default table, is read
 * against the sweep's default row. A profile field holds one geometry for every M, so besides the
 * per-case best the sum is also given for the one skeleton geometry with the smallest sum.
 *
 * NOTE: the opt-in Node run of the same set (packages/e2e/tests/e2e_wan_ti2v_k85_linear_sweep_test.ts)
 * computes its forward sum separately from the same record fields.
 */

import {
  BUILTIN_GEOMETRY_PROFILES,
  type GemmGeometry,
  type GeometryProfile,
  type I8a8Geometry,
} from "@karume/runtime";
import type {
  GeometrySweepCandidateSet,
  GeometrySweepCase,
  GeometrySweepReport,
  GeometrySweepRow,
} from "@karume/runtime/tune";

/** The evaluation set this summary reads (no table is derived from its record). */
export const K85_SET = "k85" satisfies GeometrySweepCandidateSet;

/** A measured geometry and its time per dispatch (the record's unit). */
export type Timed = { readonly geometry: string; readonly perDispatch: number };

export type K85Case = {
  readonly caseId: string;
  /** Nodes in one Wan2.2 33f DiT forward (0 = a guard shape of another model, outside the sum). */
  readonly censusCount: number;
  readonly table?: Timed;
  /** The fastest skeleton candidate whose output matched the default. */
  readonly bestSkeleton?: Timed;
  /** Table time ÷ best skeleton time (> 1: the skeleton is faster). */
  readonly skeletonVsTable?: number;
  /** The best skeleton's speed-up over the default geometry (the record's own ratio). */
  readonly skeletonVsDefault?: number;
  readonly mismatches: number;
  readonly errors: number;
};

export type K85Forward = {
  /** The forward's nodes the sum covers (the cases with a count). */
  readonly nodes: number;
  /** The forward cases the record has rows for, of those the case list names. */
  readonly measured: number;
  readonly planned: number;
  /** Σ count × the table geometry's time (absent while a forward case lacks it). */
  readonly table?: number;
  /** Σ count × the per-case best skeleton's time. */
  readonly perCaseBestSkeleton?: number;
  /** The one skeleton geometry with the smallest sum, among those identical on every forward case. */
  readonly singleBestSkeleton?: Timed;
};

export type K85Summary = {
  /** The table the record names (`"default"` when none matched or the field is absent). */
  readonly tableId: string;
  readonly cases: readonly K85Case[];
  readonly forward: K85Forward;
};

const isI8a8 = (geometry: GemmGeometry | I8a8Geometry): geometry is I8a8Geometry =>
  "tileK" in geometry;

/** Whether a row measured exactly this i8a8 geometry (the skeleton included). */
const isGeometry = (row: GeometrySweepRow, geometry: I8a8Geometry): boolean => {
  const params = row.geometryParams;
  return (
    isI8a8(params) &&
    params.regM === geometry.regM &&
    params.regN === geometry.regN &&
    params.wgX === geometry.wgX &&
    params.wgY === geometry.wgY &&
    params.tileK === geometry.tileK &&
    params.skeleton === geometry.skeleton
  );
};

const isSkeleton = (row: GeometrySweepRow): boolean =>
  isI8a8(row.geometryParams) && row.geometryParams.skeleton === "vec4";

/** A row that ran and whose output matched the default (the default row itself included). */
const usable = (row: GeometrySweepRow): row is GeometrySweepRow & { perDispatch: number } =>
  row.error === undefined &&
  row.perDispatch !== undefined &&
  (row.isDefault || row.identicalToDefault === true);

const timed = (row: GeometrySweepRow & { perDispatch: number }): Timed => ({
  geometry: row.geometry,
  perDispatch: row.perDispatch,
});

/**
 * The i8a8 linear geometry of the table the record names, or the sweep's default row's geometry for
 * the default table (and for a record without the field).
 */
export const k85TableGeometry = (
  report: Pick<GeometrySweepReport, "geometryProfile" | "rows">,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): I8a8Geometry | undefined => {
  const named = profiles.find((profile) => profile.id === report.geometryProfile);
  if (named !== undefined) {
    // One geometry per table: the summary has no row count to pick a rule by
    const [rule, ...rest] = named.i8a8.linear;
    if (rule === undefined || rest.length > 0) {
      throw new Error(
        `Table '${named.id}' has ${named.i8a8.linear.length} i8a8.linear rules (the K-85 summary reads one)`,
      );
    }
    return rule.geometry;
  }
  const params = report.rows.find((row) => row.isDefault)?.geometryParams;
  return params !== undefined && isI8a8(params) ? params : undefined;
};

/** Σ count × value over the forward cases, absent while one of them has no value. */
const weighted = (
  entries: readonly { readonly censusCount: number; readonly value: number | undefined }[],
): number | undefined =>
  entries.reduce<number | undefined>(
    (total, { censusCount, value }) =>
      total === undefined || value === undefined ? undefined : total + censusCount * value,
    0,
  );

/**
 * The summary of a `k85` record: one row per case of `cases` (the set's case list with its counts)
 * that the record measured, in the list's order, and the forward sum over the cases with a count.
 */
export const k85Summary = (
  report: Pick<GeometrySweepReport, "geometryProfile" | "rows">,
  cases: readonly GeometrySweepCase[],
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): K85Summary => {
  const table = k85TableGeometry(report, profiles);
  const tableId = profiles.some((profile) => profile.id === report.geometryProfile)
    ? (report.geometryProfile ?? "default")
    : "default";
  const rowsOf = (caseId: string): GeometrySweepRow[] =>
    report.rows.filter((row) => row.caseId === caseId);
  const summaries = cases
    .filter(({ id }) => rowsOf(id).length > 0)
    .map(({ id, censusCount }): K85Case => {
      const rows = rowsOf(id);
      const tableRow = table === undefined ? undefined : rows.find((row) => isGeometry(row, table));
      const best = rows
        .filter(isSkeleton)
        .filter(usable)
        .reduce<(GeometrySweepRow & { perDispatch: number }) | undefined>(
          (fastest, row) =>
            fastest === undefined || row.perDispatch < fastest.perDispatch ? row : fastest,
          undefined,
        );
      const tableTimed = tableRow !== undefined && usable(tableRow) ? timed(tableRow) : undefined;
      return {
        caseId: id,
        censusCount: censusCount ?? 0,
        ...(tableTimed === undefined ? {} : { table: tableTimed }),
        ...(best === undefined ? {} : { bestSkeleton: timed(best) }),
        ...(tableTimed === undefined || best === undefined
          ? {}
          : { skeletonVsTable: tableTimed.perDispatch / best.perDispatch }),
        ...(best?.speedupVsDefault === undefined
          ? {}
          : { skeletonVsDefault: best.speedupVsDefault }),
        mismatches: rows.filter((row) => row.identicalToDefault === false).length,
        errors: rows.filter((row) => row.error !== undefined).length,
      };
    });
  const planned = cases.filter(({ censusCount }) => (censusCount ?? 0) > 0);
  const forward = summaries.filter(({ censusCount }) => censusCount > 0);
  const complete = forward.length === planned.length;
  const skeletonNames = [...new Set(report.rows.filter(isSkeleton).map((row) => row.geometry))];
  const single = skeletonNames
    .map((geometry) => ({
      geometry,
      perDispatch: weighted(
        forward.map(({ caseId, censusCount }) => ({
          censusCount,
          value: rowsOf(caseId)
            .filter(usable)
            .find((row) => row.geometry === geometry)?.perDispatch,
        })),
      ),
    }))
    .filter((entry): entry is Timed => entry.perDispatch !== undefined)
    .reduce<Timed | undefined>(
      (best, entry) => (best === undefined || entry.perDispatch < best.perDispatch ? entry : best),
      undefined,
    );
  const sumOf = (pick: (entry: K85Case) => number | undefined): number | undefined =>
    complete
      ? weighted(forward.map((entry) => ({ censusCount: entry.censusCount, value: pick(entry) })))
      : undefined;
  const tableSum = sumOf((entry) => entry.table?.perDispatch);
  const perCaseBest = sumOf((entry) => entry.bestSkeleton?.perDispatch);
  return {
    tableId,
    cases: summaries,
    forward: {
      nodes: forward.reduce((total, { censusCount }) => total + censusCount, 0),
      measured: forward.length,
      planned: planned.length,
      ...(tableSum === undefined ? {} : { table: tableSum }),
      ...(perCaseBest === undefined ? {} : { perCaseBestSkeleton: perCaseBest }),
      ...(complete && single !== undefined ? { singleBestSkeleton: single } : {}),
    },
  };
};
