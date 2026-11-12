import { BUILTIN_GEOMETRY_PROFILES, type GemmGeometry } from "@karume/runtime";
import {
  type GeometrySweepCase,
  type GeometrySweepRow,
  sweepCaseSetIds,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { k88Summary, k88TableGeometry } from "../src/models/geometry-sweep/k88.ts";

/** The default geometry of the linear cases' bucket and of conv2d's 64-row class. */
const LINEAR_DEFAULT: GemmGeometry = { regM: 8, regN: 8, wgX: 16, wgY: 16 };
const CONV_DEFAULT: GemmGeometry = { regM: 8, regN: 8, wgX: 16, wgY: 8 };
/** The test table's geometries (differ from the defaults). */
const LINEAR_TABLE: GemmGeometry = { regM: 4, regN: 4, wgX: 16, wgY: 16 };
const CONV_TABLE: GemmGeometry = { regM: 4, regN: 4, wgX: 16, wgY: 16 };

const nameOf = (geometry: GemmGeometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}x${geometry.wgY}${
    geometry.form === undefined ? "" : `:${geometry.form}`
  }${geometry.decode === undefined ? "" : ":dec"}`;

/** One row of `caseId` at `geometry` taking `ms` per dispatch (identical unless said otherwise). */
const row = (
  caseId: string,
  op: GeometrySweepRow["op"],
  shape: string,
  geometry: GemmGeometry,
  ms: number,
  extra: Partial<GeometrySweepRow> = {},
): GeometrySweepRow => {
  const isDefault = geometry === LINEAR_DEFAULT || geometry === CONV_DEFAULT;
  return {
    caseId,
    op,
    shape,
    censusCount: 1,
    geometry: nameOf(geometry),
    geometryParams: geometry,
    isDefault,
    reps: 4,
    rounds: [80_000_000],
    perDispatch: ms * 1e6,
    ...(isDefault ? {} : { speedupVsDefault: 10 / ms, identicalToDefault: true }),
    ...extra,
  };
};

const gemm = "one gemma4 E2B prefill chunk (M = 768)";
const wan = "one Wan2.2 TI2V-5B VAE decoder next chunk (tile 16)";

const CASES: readonly GeometrySweepCase[] = [
  { id: "up", op: "linear", censusCount: 30, censusGroup: gemm },
  { id: "down", op: "linear", censusCount: 15, censusGroup: gemm },
  { id: "guard", op: "linear", censusCount: 0 },
  { id: "conv", op: "conv2d", censusCount: 1, censusGroup: wan },
];

const linearRows = (caseId: string, table: number, tb: number, tas: number) => [
  row(caseId, "linear", "M768 N6144 K1536", LINEAR_DEFAULT, 10),
  row(caseId, "linear", "M768 N6144 K1536", LINEAR_TABLE, table),
  row(caseId, "linear", "M768 N6144 K1536", { ...LINEAR_TABLE, form: "tb-un" }, tb),
  row(caseId, "linear", "M768 N6144 K1536", { ...LINEAR_TABLE, form: "tas-tb-un" }, tas),
  // A form of the default geometry is never "the table's form"
  row(caseId, "linear", "M768 N6144 K1536", { ...LINEAR_DEFAULT, form: "tb-un" }, 1),
];

const ROWS: readonly GeometrySweepRow[] = [
  ...linearRows("up", 8, 5, 4),
  ...linearRows("down", 8, 6, 7),
  ...linearRows("guard", 8, 9, 10),
  row("conv", "conv2d", "B2 Cin1024 Cout1024 32x32 3x3", CONV_DEFAULT, 10),
  row("conv", "conv2d", "B2 Cin1024 Cout1024 32x32 3x3", CONV_TABLE, 8),
  row("conv", "conv2d", "B2 Cin1024 Cout1024 32x32 3x3", { ...CONV_TABLE, form: "tb-un" }, 7),
  row("conv", "conv2d", "B2 Cin1024 Cout1024 32x32 3x3", { ...CONV_TABLE, form: "tas-tb-un" }, 7.5),
  row("conv", "conv2d", "B2 Cin1024 Cout1024 32x32 3x3", { ...CONV_TABLE, decode: "counter" }, 6),
  row(
    "conv",
    "conv2d",
    "B2 Cin1024 Cout1024 32x32 3x3",
    { ...CONV_TABLE, form: "tb-un", decode: "counter" },
    5,
  ),
  row(
    "conv",
    "conv2d",
    "B2 Cin1024 Cout1024 32x32 3x3",
    { ...CONV_TABLE, form: "tas-tb-un", decode: "counter" },
    4,
  ),
];

const [BASE] = BUILTIN_GEOMETRY_PROFILES;
if (BASE === undefined) throw new Error("no embedded table");
const PROFILES = [
  {
    ...BASE,
    id: "test-table",
    // M 768 falls in the last rule; conv2d's 64-row class
    gemmRows: BASE.gemmRows.map((rule) =>
      rule.maxRows === Number.POSITIVE_INFINITY ? { ...rule, geometry: LINEAR_TABLE } : rule,
    ),
    conv2d: { ...BASE.conv2d, rows64: CONV_TABLE },
  },
];

describe("k88TableGeometry", () => {
  it("is the named table's gemmRows rule of the case's M and its conv2d field of the case's class", () => {
    const report = { geometryProfile: "test-table", rows: ROWS };
    expect(k88TableGeometry(report, { id: "up", op: "linear" }, PROFILES)).toEqual(LINEAR_TABLE);
    expect(k88TableGeometry(report, { id: "conv", op: "conv2d" }, PROFILES)).toEqual(CONV_TABLE);
  });

  it("is the sweep's default row for the default table, an unknown id and a record without the field", () => {
    for (const geometryProfile of ["default", "unknown", undefined]) {
      const report = { geometryProfile, rows: ROWS };
      expect(k88TableGeometry(report, { id: "up", op: "linear" }, PROFILES)).toEqual(
        LINEAR_DEFAULT,
      );
      expect(k88TableGeometry(report, { id: "conv", op: "conv2d" }, PROFILES)).toEqual(
        CONV_DEFAULT,
      );
    }
  });
});

describe("k88Summary", () => {
  const summary = k88Summary({ geometryProfile: "test-table", rows: ROWS }, CASES, PROFILES);

  it("puts, per case, the table geometry against its own fastest form, decode and decode + form (never another geometry's form)", () => {
    expect(summary.tableId).toBe("test-table");
    expect(
      summary.cases.map((entry) => [
        entry.caseId,
        entry.table?.geometry,
        entry.bestForm?.geometry,
        entry.formVsTable,
        entry.decodeVsTable,
        entry.bothVsTable,
        entry.identical,
      ]),
    ).toEqual([
      ["up", "r4x4w16x16", "r4x4w16x16:tas-tb-un", 8 / 4, undefined, undefined, true],
      ["down", "r4x4w16x16", "r4x4w16x16:tb-un", 8 / 6, undefined, undefined, true],
      ["guard", "r4x4w16x16", "r4x4w16x16:tb-un", 8 / 9, undefined, undefined, true],
      ["conv", "r4x4w16x16", "r4x4w16x16:tb-un", 8 / 7, 8 / 6, 8 / 4, true],
    ]);
    expect(summary.cases[3]?.bestBoth?.geometry).toBe("r4x4w16x16:tas-tb-un:dec");
  });

  it("sums each unit over its cases with a count (the guard in none), in the record's unit", () => {
    expect(summary.sums).toEqual([
      {
        unit: gemm,
        nodes: 45,
        measured: 2,
        planned: 2,
        table: (30 * 8 + 15 * 8) * 1e6,
        bestForm: (30 * 4 + 15 * 6) * 1e6,
      },
      {
        unit: wan,
        nodes: 1,
        measured: 1,
        planned: 1,
        table: 8e6,
        bestForm: 7e6,
        decode: 6e6,
        bestBoth: 4e6,
      },
    ]);
  });

  it("never counts a candidate whose output differed or failed, marks the case not identical, and gives no sum while a unit's case is missing", () => {
    const broken = ROWS.map((entry) =>
      entry.caseId === "up" && entry.geometry === "r4x4w16x16:tas-tb-un"
        ? { ...entry, identicalToDefault: false }
        : entry.caseId === "conv" && entry.geometry === "r4x4w16x16:dec"
          ? { ...entry, error: "device lost", perDispatch: undefined }
          : entry,
    );
    const result = k88Summary({ geometryProfile: "test-table", rows: broken }, CASES, PROFILES);
    const [up, , , conv] = result.cases;
    expect(up?.bestForm?.geometry).toBe("r4x4w16x16:tb-un");
    expect(up?.identical).toBe(false);
    expect(up?.mismatches).toBe(1);
    expect(conv?.decodeVsTable).toBeUndefined();
    expect(conv?.errors).toBe(1);
    expect(conv?.identical).toBe(false);
    const partial = k88Summary(
      { geometryProfile: "test-table", rows: ROWS.filter(({ caseId }) => caseId !== "down") },
      CASES,
      PROFILES,
    );
    expect(partial.sums[0]).toEqual({ unit: gemm, nodes: 30, measured: 1, planned: 2 });
  });
});

describe("the case-set ids the page shows (the chosen set's, per op family)", () => {
  it("differ per evaluation set and are the sweep's own for the table-deriving sets", () => {
    const sweep = sweepCaseSetIds();
    const k85 = sweepCaseSetIds("k85");
    const k88 = sweepCaseSetIds("k88");
    expect(sweepCaseSetIds("full")).toEqual(sweep);
    expect(Object.keys(k85)).toEqual(["i8a8-linear"]);
    expect(k85["i8a8-linear"]).not.toBe(sweep["i8a8-linear"]);
    expect(Object.keys(k88).length).toBeGreaterThan(0);
    const sweepIds = new Map(Object.entries(sweep));
    for (const [family, id] of Object.entries(k88)) {
      expect(id, family).not.toBe(sweepIds.get(family));
    }
    expect(k88).not.toEqual(k85);
  });
});
