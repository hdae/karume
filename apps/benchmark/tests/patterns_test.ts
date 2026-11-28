import {
  GEMV_LOWEST_MARGIN,
  GEMV_MARGIN,
  type GemvSweepCaseSummary,
  type GemvSweepRow,
  GEOMETRY_MIN_SPEEDUP,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { gemvPatterns, geometryPatterns } from "../src/patterns.ts";
import {
  CONV3D_A,
  CONV3D_B,
  GEMV_A,
  GEMV_B,
  gemvRow,
  gemvSummary,
  geometryReport,
  geometryRow,
} from "./helpers/fakes.ts";

/** The baseline at 20 µs and the default's choice at reorder (`parallel-l4`) at 12 µs. */
const references = (caseId: string) => [
  gemvRow(caseId, "baseline"),
  gemvRow(caseId, "reorder", { perDispatch: 12_000 }),
];

describe("gemvPatterns", () => {
  // Every fake row's rounds are 80 ms and 80.1 ms: a gap of 1.00125, so a candidate's bar over its
  // reference is the margin × 1.00125² (×1.0677 at ×1.065, ×1.0326 at ×1.03)
  const BAR = GEMV_MARGIN * 1.00125 ** 2;
  const LOW_BAR = GEMV_LOWEST_MARGIN * 1.00125 ** 2;
  const count = (summaries: readonly GemvSweepCaseSummary[], rows: readonly GemvSweepRow[]) =>
    gemvPatterns(summaries, rows, "ns");

  it("counts a group whose sequential variant clears the bar at ×1.065 over the baseline, its words the baseline's — and at ×1.03 too", () => {
    const rows = [
      ...references(GEMV_A),
      gemvRow(GEMV_A, "identical", { perDispatch: 20_000 / (BAR + 0.001) }),
    ];
    expect(count([gemvSummary(GEMV_A)], rows)).toEqual({ atMargin: 1, atLowest: 1, of: 1 });
  });

  it("counts one between the two margins at ×1.03 only, and none below the lower bar or whose words differ (faults)", () => {
    const between = [
      ...references(GEMV_A),
      gemvRow(GEMV_A, "identical", { perDispatch: 20_000 / (BAR - 0.001) }),
    ];
    expect(count([gemvSummary(GEMV_A)], between)).toEqual({ atMargin: 0, atLowest: 1, of: 1 });
    const below = [
      ...references(GEMV_A),
      gemvRow(GEMV_A, "identical", { perDispatch: 20_000 / (LOW_BAR - 0.001) }),
    ];
    const differs = [
      ...references(GEMV_A),
      gemvRow(GEMV_A, "identical", { perDispatch: 10_000, identicalToBaseline: false }),
    ];
    expect(count([gemvSummary(GEMV_A)], below).atLowest).toBe(0);
    expect(count([gemvSummary(GEMV_A)], differs).atLowest).toBe(0);
  });

  it("reads a parallel row against the default's choice at reorder, not the baseline, and holds it to the band", () => {
    const parallel = (perDispatch: number, extra = {}) => ({
      ...gemvRow(GEMV_A, "reorder", { perDispatch, ...extra }),
      candidate: "staged-l8",
      choice: {
        id: "gemv-parallel-staged" as const,
        storage: "i8" as const,
        lanes: 8 as const,
        variant: { cols: 32, unroll: 4 },
      },
    });
    const summary = [gemvSummary(GEMV_A)];
    // ×1.9 over the baseline but only ×1.05 over parallel-l4 at 12 µs: no pattern at ×1.065 (fault: read against the baseline)
    expect(count(summary, [...references(GEMV_A), parallel(12_000 / 1.05)]).atMargin).toBe(0);
    expect(count(summary, [...references(GEMV_A), parallel(12_000 / 1.2)]).atMargin).toBe(1);
    expect(
      count(summary, [...references(GEMV_A), parallel(12_000 / 1.2, { withinBand: false })])
        .atLowest,
    ).toBe(0);
  });

  it("takes no subgroup form (a Session option a table cannot hold), and no screened or failed row", () => {
    const summary = [gemvSummary(GEMV_A)];
    const fast = { perDispatch: 1_000 };
    expect(
      count(summary, [...references(GEMV_A), gemvRow(GEMV_A, "subgroup", fast)]).atLowest,
    ).toBe(0);
    expect(
      count(summary, [
        ...references(GEMV_A),
        gemvRow(GEMV_A, "identical", { ...fast, screened: true }),
      ]).atLowest,
    ).toBe(0);
    expect(
      count(summary, [
        ...references(GEMV_A),
        gemvRow(GEMV_A, "identical", { ...fast, error: "boom" }),
      ]).atLowest,
    ).toBe(0);
  });

  it("is no evidence from a pass under 20 ms, a reference re-timed beyond the bar, or timing not in ns (faults: 10 ms rounds, a re-timing ×0.9, Deno's raw ticks)", () => {
    const summary = [gemvSummary(GEMV_A)];
    const fast = gemvRow(GEMV_A, "identical", { perDispatch: 10_000 });
    expect(count(summary, [...references(GEMV_A), fast]).atMargin).toBe(1);
    const short = gemvRow(GEMV_A, "identical", { perDispatch: 10_000, rounds: [10e6, 10.01e6] });
    expect(count(summary, [...references(GEMV_A), short]).atLowest).toBe(0);
    const [baseline, reorder] = references(GEMV_A);
    if (baseline === undefined || reorder === undefined) throw new Error("no references");
    const moved = { ...baseline, retimedPerDispatch: 20_000 * 0.9 };
    expect(count(summary, [moved, reorder, fast]).atLowest).toBe(0);
    expect(gemvPatterns(summary, [...references(GEMV_A), fast], "deno-raw-tick").atLowest).toBe(0);
  });

  it("reads only the groups at which the derivation picks a winner (M = 1), and none without a timed reference", () => {
    const atFour = gemvSummary(GEMV_A, { m: 4 });
    const rows = [...references(GEMV_A), gemvRow(GEMV_A, "identical", { perDispatch: 1_000 })].map(
      (row) => ({ ...row, m: 4 }),
    );
    expect(count([atFour], rows)).toEqual({ atMargin: 0, atLowest: 0, of: 0 });
    const failedBaseline = [
      gemvRow(GEMV_A, "baseline", { error: "boom" }),
      gemvRow(GEMV_A, "identical", { perDispatch: 1_000 }),
    ];
    expect(count([gemvSummary(GEMV_A)], failedBaseline)).toEqual({
      atMargin: 0,
      atLowest: 0,
      of: 1,
    });
  });

  it("reads against the current default profile's choice, whatever the group's summary names", () => {
    // The summary names the fast variant as the choice at exact; the derivation reads c32u4 there
    const summary = gemvSummary(GEMV_A, { profile: { exact: "c64u8", reorder: "parallel-l4" } });
    const rows = [...references(GEMV_A), gemvRow(GEMV_A, "identical", { perDispatch: 10_000 })];
    expect(count([summary], rows)).toEqual({ atMargin: 1, atLowest: 1, of: 1 });
  });

  describe("where the record names the device table's choice (summary.table)", () => {
    /** The table's exception at exact: c128u8, at `gain` over the baseline (its words the baseline's). */
    const tableChoice = (gain: number, extra: Partial<GemvSweepRow> = {}) => ({
      ...gemvRow(GEMV_A, "identical", { perDispatch: 20_000 / gain }),
      candidate: "c128u8",
      choice: { id: "gemv-m1" as const, storage: "i8" as const, variant: { cols: 128, unroll: 8 } },
      ...extra,
    });
    const variant = (gain: number) => gemvRow(GEMV_A, "identical", { perDispatch: 20_000 / gain });
    const summary = [gemvSummary(GEMV_A, { table: { exact: "c128u8", reorder: "parallel-l4" } })];

    it("counts a candidate only where it beats the table's choice by the bar too (fault: read against the default's choice alone, which counts ×1.038 over the table's ×1.3)", () => {
      const near = [...references(GEMV_A), tableChoice(1.3), variant(1.35)];
      expect(count(summary, near)).toEqual({ atMargin: 0, atLowest: 1, of: 1 });
      const clear = [...references(GEMV_A), tableChoice(1.3), variant(1.5)];
      expect(count(summary, clear)).toEqual({ atMargin: 1, atLowest: 1, of: 1 });
    });

    it("counts the default's choice that beats the table's by the bar (the derivation goes back to it)", () => {
      expect(count(summary, [...references(GEMV_A), tableChoice(0.8)])).toEqual({
        atMargin: 1,
        atLowest: 1,
        of: 1,
      });
    });

    it("counts nothing where the record has no time of the table's choice, and reads against the default's where its check failed", () => {
      expect(count(summary, [...references(GEMV_A), variant(1.5)])).toEqual({
        atMargin: 0,
        atLowest: 0,
        of: 1,
      });
      const failed = [
        ...references(GEMV_A),
        tableChoice(1.3, { identicalToBaseline: false }),
        variant(1.5),
      ];
      expect(count(summary, failed)).toEqual({ atMargin: 1, atLowest: 1, of: 1 });
    });
  });

  it("counts groups, not candidates: two cases, one with two winners", () => {
    const rows = [
      ...references(GEMV_A),
      gemvRow(GEMV_A, "identical", { perDispatch: 10_000 }),
      gemvRow(GEMV_A, "reorder", {
        candidate: "parallel-l8",
        perDispatch: 6_000,
        choice: { id: "gemv-parallel", storage: "i8", lanes: 8, variant: { cols: 32, unroll: 4 } },
      }),
      ...references(GEMV_B),
    ];
    expect(count([gemvSummary(GEMV_A), gemvSummary(GEMV_B)], rows)).toEqual({
      atMargin: 1,
      atLowest: 1,
      of: 2,
    });
  });
});

describe("geometryPatterns", () => {
  const candidate = (caseId: string, speedup: number, extra = {}) =>
    geometryRow(caseId, false, { speedupVsDefault: speedup, ...extra });
  /** The per-case record of `caseId`: its default row, `rows` and its re-measurement within range. */
  const record = (caseId: string, ...rows: ReturnType<typeof geometryRow>[]) =>
    geometryReport(caseId, { rows: [geometryRow(caseId, true), ...rows] });

  it("counts a class whose candidate is the derivation's threshold ahead, with the default's output, in every case so far", () => {
    const records = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      record(CONV3D_B, candidate(CONV3D_B, GEOMETRY_MIN_SPEEDUP)),
    ];
    expect(geometryPatterns(records)).toEqual({ found: 1, of: 1 });
  });

  it("does not count a class where the candidate falls short in one case, or wrote another output there (faults)", () => {
    const short = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      record(CONV3D_B, candidate(CONV3D_B, GEOMETRY_MIN_SPEEDUP - 0.001)),
    ];
    const differs = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      record(CONV3D_B, candidate(CONV3D_B, 1.3, { identicalToDefault: false })),
    ];
    expect(geometryPatterns(short)).toEqual({ found: 0, of: 1 });
    expect(geometryPatterns(differs)).toEqual({ found: 0, of: 1 });
  });

  it("does not count a candidate that a later case skipped (not ahead there), and counts one class once", () => {
    const skipped = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      record(CONV3D_B, {
        ...geometryRow(CONV3D_B, false),
        skipped: true as const,
        speedupVsDefault: undefined,
        identicalToDefault: undefined,
      }),
    ];
    expect(geometryPatterns(skipped)).toEqual({ found: 0, of: 1 });
  });

  it("reads the classes apart: a linear case is another class than the conv3d cases", () => {
    const linear = "linear-m1024-n2048-k2048";
    const records = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      // The fake fingerprints conv3d rows only: the linear record is the conv3d one, its case renamed
      {
        ...geometryReport(CONV3D_A),
        cases: geometryReport(CONV3D_A).cases.map((summary) => ({ ...summary, caseId: linear })),
        rows: [
          { ...geometryRow(linear, true), op: "linear" as const },
          { ...geometryRow(linear, false), op: "linear" as const, speedupVsDefault: 1.01 },
        ],
      },
    ];
    expect(geometryPatterns(records)).toEqual({ found: 1, of: 2 });
  });

  it("counts no class with a case the derivation drops: its default moved out of range when re-measured", () => {
    const moved = record(CONV3D_B, candidate(CONV3D_B, 1.3));
    const records = [
      record(CONV3D_A, candidate(CONV3D_A, 1.3)),
      {
        ...moved,
        cases: moved.cases.map((summary) => ({
          ...summary,
          defaultRepeat: { perDispatch: 24_000_000, driftRatio: 1.2 },
        })),
      },
    ];
    expect(geometryPatterns(records)).toEqual({ found: 0, of: 1 });
  });

  it("does not count a candidate whose observation the derivation drops for its rounding error, and reads the merged record's step", () => {
    // A 5 ms minimum round under 100 µs rounding: E ≥ 2% (over 1%) — kept without rounding
    const coarse = candidate(CONV3D_A, 1.3, { rounds: [5_000_000, 5_100_000] });
    const quantized = (report: ReturnType<typeof geometryReport>, value: boolean) => ({
      ...report,
      gpuTiming: { ...report.gpuTiming, quantized: value },
    });
    expect(geometryPatterns([quantized(record(CONV3D_A, coarse), false)])).toEqual({
      found: 1,
      of: 1,
    });
    expect(geometryPatterns([quantized(record(CONV3D_A, coarse), true)])).toEqual({
      found: 0,
      of: 1,
    });
    // The record of the run takes the step when any case's rounds looked rounded
    expect(
      geometryPatterns([
        quantized(record(CONV3D_A, coarse), false),
        quantized(record(CONV3D_B, candidate(CONV3D_B, 1.3)), true),
      ]),
    ).toEqual({ found: 0, of: 1 });
  });
});
