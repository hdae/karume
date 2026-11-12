import { BUILTIN_GEOMETRY_PROFILES, type I8a8Geometry } from "@karume/runtime";
import {
  type GeometrySweepCase,
  type GeometrySweepRow,
  geometrySweepCases,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { k85Summary, k85TableGeometry } from "../src/models/geometry-sweep/k85.ts";

const DEFAULT: I8a8Geometry = { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 };
const TABLE: I8a8Geometry = { regM: 8, regN: 4, wgX: 16, wgY: 16, tileK: 16 };
const SKELETON_A: I8a8Geometry = {
  regM: 16,
  regN: 4,
  wgX: 16,
  wgY: 8,
  tileK: 64,
  skeleton: "vec4",
};
const SKELETON_B: I8a8Geometry = { ...SKELETON_A, regM: 8, regN: 8, wgX: 8, wgY: 16 };
/** The same tile as SKELETON_A without the skeleton (a word-fill candidate). */
const WORD_A: I8a8Geometry = { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64 };

const nameOf = (geometry: I8a8Geometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}x${geometry.wgY}k${geometry.tileK}${
    geometry.skeleton === undefined ? "" : "s"
  }`;

/** One row of `caseId` at `geometry` taking `ms` per dispatch (identical unless said otherwise). */
const row = (
  caseId: string,
  geometry: I8a8Geometry,
  ms: number,
  extra: Partial<GeometrySweepRow> = {},
): GeometrySweepRow => {
  const isDefault = geometry === DEFAULT;
  return {
    caseId,
    op: "i8a8-linear",
    shape: caseId,
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

const CASES: readonly GeometrySweepCase[] = [
  { id: "qkvo", op: "i8a8-linear", censusCount: 180 },
  { id: "up", op: "i8a8-linear", censusCount: 30 },
  { id: "guard", op: "i8a8-linear", censusCount: 0 },
];

/** Default 10 ms, table 8 ms, A 5 / 4 / 9, B 6 / 3 / 7, word A 7 everywhere (ms per dispatch). */
const ROWS: readonly GeometrySweepRow[] = [
  row("qkvo", DEFAULT, 10),
  row("qkvo", TABLE, 8),
  row("qkvo", SKELETON_A, 5),
  row("qkvo", SKELETON_B, 6),
  row("qkvo", WORD_A, 7),
  row("up", DEFAULT, 10),
  row("up", TABLE, 8),
  row("up", SKELETON_A, 4),
  row("up", SKELETON_B, 3),
  row("up", WORD_A, 7),
  row("guard", DEFAULT, 10),
  row("guard", TABLE, 8),
  row("guard", SKELETON_A, 9),
  row("guard", SKELETON_B, 7),
];

const PROFILES = [
  {
    ...BUILTIN_GEOMETRY_PROFILES[0],
    id: "test-table",
    i8a8: {
      ...BUILTIN_GEOMETRY_PROFILES[0].i8a8,
      linear: [{ maxRows: Number.POSITIVE_INFINITY, geometry: TABLE }],
    },
  },
];

describe("k85TableGeometry", () => {
  it("is the named embedded table's i8a8.linear", () => {
    expect(k85TableGeometry({ geometryProfile: "test-table", rows: ROWS }, PROFILES)).toEqual(
      TABLE,
    );
  });

  it("is the sweep's default row for the default table, an unknown id and a record without the field", () => {
    for (const geometryProfile of ["default", "unknown", undefined]) {
      expect(k85TableGeometry({ geometryProfile, rows: ROWS }, PROFILES)).toEqual(DEFAULT);
    }
  });
});

describe("k85Summary", () => {
  const summary = k85Summary({ geometryProfile: "test-table", rows: ROWS }, CASES, PROFILES);

  it("puts, per case, the table geometry against the fastest skeleton candidate (a word-fill candidate is never the skeleton)", () => {
    expect(summary.tableId).toBe("test-table");
    expect(
      summary.cases.map(({ caseId, table, bestSkeleton, skeletonVsTable, censusCount }) => [
        caseId,
        censusCount,
        table?.geometry,
        bestSkeleton?.geometry,
        skeletonVsTable,
      ]),
    ).toEqual([
      ["qkvo", 180, "r8x4w16x16k16", "r16x4w16x8k64s", 8 / 5],
      ["up", 30, "r8x4w16x16k16", "r8x8w8x16k64s", 8 / 3],
      ["guard", 0, "r8x4w16x16k16", "r8x8w8x16k64s", 8 / 7],
    ]);
    expect(summary.cases[0]?.skeletonVsDefault).toBe(2);
  });

  it("sums count × time over the forward cases only (the guard is outside), per-case best and the one best skeleton geometry", () => {
    expect(summary.forward).toEqual({
      nodes: 210,
      measured: 2,
      planned: 2,
      table: (180 * 8 + 30 * 8) * 1e6,
      perCaseBestSkeleton: (180 * 5 + 30 * 3) * 1e6,
      // A: 180·5 + 30·4 = 1,020; B: 180·6 + 30·3 = 1,170
      singleBestSkeleton: { geometry: "r16x4w16x8k64s", perDispatch: (180 * 5 + 30 * 4) * 1e6 },
    });
  });

  it("never counts a candidate whose output differed or that failed", () => {
    const rows = ROWS.map((entry) =>
      entry.caseId === "qkvo" && entry.geometry === "r16x4w16x8k64s"
        ? { ...entry, identicalToDefault: false }
        : entry.caseId === "up" && entry.geometry === "r8x8w8x16k64s"
          ? { ...entry, error: "compile failed" }
          : entry,
    );
    const changed = k85Summary({ geometryProfile: "test-table", rows }, CASES, PROFILES);
    expect(changed.cases.map(({ bestSkeleton }) => bestSkeleton?.geometry)).toEqual([
      "r8x8w8x16k64s",
      "r16x4w16x8k64s",
      "r8x8w8x16k64s",
    ]);
    expect(changed.cases.map(({ mismatches, errors }) => [mismatches, errors])).toEqual([
      [1, 0],
      [0, 1],
      [0, 0],
    ]);
    // A differs on qkvo and B failed on up: no skeleton geometry is usable on every forward case
    expect(changed.forward.singleBestSkeleton).toBeUndefined();
    expect(changed.forward.perCaseBestSkeleton).toBe((180 * 6 + 30 * 4) * 1e6);
  });

  it("leaves the forward sums out while a forward case is missing from the record (a cancelled sweep)", () => {
    const partial = k85Summary(
      { geometryProfile: "test-table", rows: ROWS.filter(({ caseId }) => caseId !== "up") },
      CASES,
      PROFILES,
    );
    expect(partial.cases.map(({ caseId }) => caseId)).toEqual(["qkvo", "guard"]);
    expect(partial.forward).toEqual({ nodes: 180, measured: 1, planned: 2 });
  });

  it("reads the runtime's k85 case list: five forward shapes with counts and one guard", () => {
    const cases = geometrySweepCases("k85");
    expect(cases.map(({ censusCount }) => censusCount)).toEqual([180, 30, 30, 61, 2, 0]);
    expect(cases.every(({ op }) => op === "i8a8-linear")).toBe(true);
  });
});
