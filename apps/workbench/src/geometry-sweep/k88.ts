/**
 * The K-88 summary of a sweep record of the evaluation set `k88` (perf-ledger K-88 — the f32 GEMM
 * skeleton forms and the conv2d counter decode, the optional geometry fields `form` / `decode`):
 * per case, this adapter's table geometry against the same geometry with the fastest form, with the
 * decode alone and with the fastest decode + form, each as table time ÷ variant time (> 1: the
 * variant is faster), and whether every candidate's output matched the default bit for bit; then,
 * per unit the cases count in (`censusGroup` — one gemma4 E2B prefill chunk, one Wan2.2 VAE next
 * chunk, …), the census-weighted sum (each case's `censusCount` × its time per dispatch). Pure.
 *
 * "This adapter's table" is the embedded table the record names (`geometryProfile` — the one a
 * Session on that adapter selects): its `gemmRows` rule of the case's M (read from the row's shape
 * `M{m} …`, the spelling derive.ts reads too) or its conv2d field of the case's class (the tile edge
 * of the sweep's default row, 64 or 32 rows). A record without the field, or naming the default
 * table, is read against the sweep's default row.
 *
 * NOTE: the opt-in Node run of the same set (packages/e2e/tests/e2e_k88_gemm_sweep_test.ts)
 * computes its summary separately from the same record fields.
 */

import {
  BUILTIN_GEOMETRY_PROFILES,
  type GemmGeometry,
  type GeometryProfile,
} from "@karume/runtime";
import type {
  GeometrySweepCandidateSet,
  GeometrySweepCase,
  GeometrySweepReport,
  GeometrySweepRow,
} from "@karume/runtime/tune";

/** The evaluation set this summary reads (no table is derived from its record). */
export const K88_SET = "k88" satisfies GeometrySweepCandidateSet;

/** A measured geometry and its time per dispatch (the record's unit). */
export type Timed = { readonly geometry: string; readonly perDispatch: number };

export type K88Case = {
  readonly caseId: string;
  /** Nodes in its unit (0 = a shape of no model, outside the sums). */
  readonly censusCount: number;
  readonly censusGroup?: string;
  /** The table geometry without the fields. */
  readonly table?: Timed;
  /** The table geometry with the faster of the two forms. */
  readonly bestForm?: Timed;
  /** The table geometry with the counter decode alone (conv2d). */
  readonly decode?: Timed;
  /** The table geometry with the decode and the faster form (conv2d). */
  readonly bestBoth?: Timed;
  /** Table time ÷ variant time (> 1: the variant is faster). */
  readonly formVsTable?: number;
  readonly decodeVsTable?: number;
  readonly bothVsTable?: number;
  /** Every row of the case ran and every candidate's output matched the default's words. */
  readonly identical: boolean;
  readonly mismatches: number;
  readonly errors: number;
};

/** The census-weighted sum over the cases of one unit (each absent while a case lacks it). */
export type K88Sum = {
  readonly unit: string;
  /** The unit's nodes the sum covers. */
  readonly nodes: number;
  /** The unit's cases the record has rows for, of those the case list names. */
  readonly measured: number;
  readonly planned: number;
  readonly table?: number;
  readonly bestForm?: number;
  readonly decode?: number;
  readonly bestBoth?: number;
};

export type K88Summary = {
  /** The table the record names (`"default"` when none matched or the field is absent). */
  readonly tableId: string;
  readonly cases: readonly K88Case[];
  readonly sums: readonly K88Sum[];
};

type Fields = Pick<GemmGeometry, "form" | "decode">;

const FORMS = ["tb-un", "tas-tb-un"] as const;

const isF32 = (params: GeometrySweepRow["geometryParams"]): params is GemmGeometry =>
  !("tileK" in params);

/** Whether a row measured exactly `base` with exactly `fields`. */
const isGeometry = (row: GeometrySweepRow, base: GemmGeometry, fields: Fields): boolean => {
  const params = row.geometryParams;
  return (
    isF32(params) &&
    params.regM === base.regM &&
    params.regN === base.regN &&
    params.wgX === base.wgX &&
    params.wgY === base.wgY &&
    params.form === fields.form &&
    params.decode === fields.decode
  );
};

/** A row that ran and whose output matched the default (the default row itself included). */
const usable = (
  row: GeometrySweepRow | undefined,
): row is GeometrySweepRow & { perDispatch: number } =>
  row !== undefined &&
  row.error === undefined &&
  row.perDispatch !== undefined &&
  (row.isDefault || row.identicalToDefault === true);

const timed = (row: GeometrySweepRow & { perDispatch: number }): Timed => ({
  geometry: row.geometry,
  perDispatch: row.perDispatch,
});

/**
 * The geometry this adapter's table gives a case of the record: the named embedded table's
 * `gemmRows` rule of the case's M or its conv2d field of the case's class, or the sweep's default
 * row for the default table (and an unknown id or a record without the field). Undefined while the
 * case has no default row.
 */
export const k88TableGeometry = (
  report: Pick<GeometrySweepReport, "geometryProfile" | "rows">,
  sweepCase: Pick<GeometrySweepCase, "id" | "op">,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GemmGeometry | undefined => {
  const defaultRow = report.rows.find((row) => row.caseId === sweepCase.id && row.isDefault);
  if (defaultRow === undefined || !isF32(defaultRow.geometryParams)) return undefined;
  const params = defaultRow.geometryParams;
  const profile = profiles.find((entry) => entry.id === report.geometryProfile);
  if (profile === undefined) return params;
  if (sweepCase.op === "conv2d") {
    return params.regM * params.wgY === 32 ? profile.conv2d.rows32 : profile.conv2d.rows64;
  }
  const m = Number(/(?:^| )M(\d+) /.exec(defaultRow.shape)?.[1]);
  return profile.gemmRows.find((rule) => m <= rule.maxRows)?.geometry;
};

/** The faster of two timed rows (absent ones skipped). */
const faster = (entries: readonly (Timed | undefined)[]): Timed | undefined =>
  entries.reduce<Timed | undefined>(
    (best, entry) =>
      entry === undefined || (best !== undefined && best.perDispatch <= entry.perDispatch)
        ? best
        : entry,
    undefined,
  );

/**
 * The summary of a `k88` record: one row per case of `cases` (the set's case list with its counts
 * and units) that the record measured, in the list's order, and one sum per unit of those cases.
 */
export const k88Summary = (
  report: Pick<GeometrySweepReport, "geometryProfile" | "rows">,
  cases: readonly GeometrySweepCase[],
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): K88Summary => {
  const tableId = profiles.some((profile) => profile.id === report.geometryProfile)
    ? (report.geometryProfile ?? "default")
    : "default";
  const rowsOf = (caseId: string): GeometrySweepRow[] =>
    report.rows.filter((row) => row.caseId === caseId);
  const summaries = cases
    .filter(({ id }) => rowsOf(id).length > 0)
    .map((sweepCase): K88Case => {
      const rows = rowsOf(sweepCase.id);
      const table = k88TableGeometry(report, sweepCase, profiles);
      const of = (fields: Fields): Timed | undefined => {
        if (table === undefined) return undefined;
        const row = rows.find((candidate) => isGeometry(candidate, table, fields));
        return usable(row) ? timed(row) : undefined;
      };
      const tableTimed = of({});
      const bestForm = faster(FORMS.map((form) => of({ form })));
      const conv = sweepCase.op === "conv2d";
      const decode = conv ? of({ decode: "counter" }) : undefined;
      const bestBoth = conv
        ? faster(FORMS.map((form) => of({ form, decode: "counter" })))
        : undefined;
      const ratio = (variant: Timed | undefined): number | undefined =>
        tableTimed === undefined || variant === undefined
          ? undefined
          : tableTimed.perDispatch / variant.perDispatch;
      const mismatches = rows.filter((row) => row.identicalToDefault === false).length;
      const errors = rows.filter((row) => row.error !== undefined).length;
      const formVsTable = ratio(bestForm);
      const decodeVsTable = ratio(decode);
      const bothVsTable = ratio(bestBoth);
      return {
        caseId: sweepCase.id,
        censusCount: sweepCase.censusCount ?? 0,
        ...(sweepCase.censusGroup === undefined ? {} : { censusGroup: sweepCase.censusGroup }),
        ...(tableTimed === undefined ? {} : { table: tableTimed }),
        ...(bestForm === undefined ? {} : { bestForm }),
        ...(decode === undefined ? {} : { decode }),
        ...(bestBoth === undefined ? {} : { bestBoth }),
        ...(formVsTable === undefined ? {} : { formVsTable }),
        ...(decodeVsTable === undefined ? {} : { decodeVsTable }),
        ...(bothVsTable === undefined ? {} : { bothVsTable }),
        identical: mismatches === 0 && errors === 0,
        mismatches,
        errors,
      };
    });
  const units = [
    ...new Set(
      cases.flatMap(({ censusGroup, censusCount }) =>
        censusGroup !== undefined && (censusCount ?? 0) > 0 ? [censusGroup] : [],
      ),
    ),
  ];
  const sums = units.map((unit): K88Sum => {
    const planned = cases.filter(
      ({ censusGroup, censusCount }) => censusGroup === unit && (censusCount ?? 0) > 0,
    );
    const members = summaries.filter(
      (entry) => entry.censusGroup === unit && entry.censusCount > 0,
    );
    const complete = members.length === planned.length;
    const sum = (pick: (entry: K88Case) => Timed | undefined): number | undefined =>
      complete
        ? members.reduce<number | undefined>((total, entry) => {
            const value = pick(entry)?.perDispatch;
            return total === undefined || value === undefined
              ? undefined
              : total + entry.censusCount * value;
          }, 0)
        : undefined;
    const table = sum((entry) => entry.table);
    const bestForm = sum((entry) => entry.bestForm);
    const decode = sum((entry) => entry.decode);
    const bestBoth = sum((entry) => entry.bestBoth);
    return {
      unit,
      nodes: members.reduce((total, { censusCount }) => total + censusCount, 0),
      measured: members.length,
      planned: planned.length,
      ...(table === undefined ? {} : { table }),
      ...(bestForm === undefined ? {} : { bestForm }),
      ...(decode === undefined ? {} : { decode }),
      ...(bestBoth === undefined ? {} : { bestBoth }),
    };
  });
  return { tableId, cases: summaries, sums };
};
