import { gemvSteadiness } from "@karume/runtime/tune";
import { mergeGemvSweepReports } from "@karume/sweep-app-support/gemv-sweep/record";
import { describe, expect, it } from "vite-plus/test";

import type { BenchmarkEvent, StampedEvent } from "../src/controller.ts";
import { durationText, PRIOR_SECONDS } from "../src/estimate.ts";
import { planBenchmark } from "../src/plan.ts";
import {
  foldProgress,
  INITIAL_PROGRESS,
  type ProgressState,
  progressView,
  steadinessLine,
} from "../src/progress.ts";
import {
  ADAPTER,
  CONV3D_A,
  CONV3D_B,
  GEMV_A,
  GEMV_B,
  gemvReport,
  gemvRow,
  gemvSummary,
  geometryReport,
  geometryRow,
  GRID_A,
  GRID_B,
  gridReport,
} from "./helpers/fakes.ts";

const PLAN = planBenchmark(
  { gemv: true, geometry: true, grid: false },
  { cases: [GEMV_A, GEMV_B, CONV3D_A, CONV3D_B], idleSeconds: 10 },
);

/** The run's cases in its order (the table's order within a sweep). */
const [FIRST = "", SECOND = "", , FOURTH = ""] = PLAN.cases.map(({ caseId }) => caseId);

const fold = (events: readonly (readonly [number, BenchmarkEvent])[]): ProgressState =>
  events.reduce<ProgressState>(
    (state, [at, event]) => foldProgress(state, { at, event } satisfies StampedEvent),
    INITIAL_PROGRESS,
  );

/**
 * GEMV case `index` of the run (`caseId`) started at `at`, measured its rows, finished at `end`
 * (its sweep call returned its record, `aborted` when a cancel cut it short).
 */
const gemvCase = (
  index: number,
  caseId: string,
  at: number,
  end: number,
  aborted = false,
): (readonly [number, BenchmarkEvent])[] => [
  [at, { kind: "case", index }],
  [
    at,
    {
      kind: "progress",
      sweep: "gemv",
      progress: { kind: "started", adapter: ADAPTER, unit: "ns", subgroups: true, caseCount: 1 },
    },
  ],
  [
    at,
    {
      kind: "progress",
      sweep: "gemv",
      progress: { kind: "plan", caseId, index: 0, rowCounts: [{ m: 1, kernels: 3 }] },
    },
  ],
  ...gemvRows(caseId).map((row): readonly [number, BenchmarkEvent] => [
    at,
    { kind: "progress", sweep: "gemv", progress: { kind: "row", row } },
  ]),
  [
    end,
    {
      kind: "progress",
      sweep: "gemv",
      progress: { kind: "case", caseId, summaries: [gemvSummary(caseId)] },
    },
  ],
  [
    end,
    {
      kind: "record",
      sweep: "gemv",
      report: gemvReport(caseId, { rows: gemvRows(caseId), aborted }),
    },
  ],
];

/** A GEMV case's rows: the baseline, and a sequential variant twice as fast. */
const gemvRows = (caseId: string) => [
  gemvRow(caseId, "baseline"),
  gemvRow(caseId, "identical", { perDispatch: 10_000 }),
];

describe("the progress view", () => {
  it("before the run: nothing done, no case bar, the time left from the priors and the idle times", () => {
    const view = progressView(PLAN, INITIAL_PROGRESS, 0);
    expect(view.overall).toEqual({
      done: 0,
      total: 4,
      text: "0 of 4 cases done (GEMV 0 of 2 · tile geometry 0 of 2)",
    });
    expect(view.current).toBeUndefined();
    // The two GEMV trial cases are no sentinels: their prior is `other`
    const expected =
      2 * PRIOR_SECONDS.other + PRIOR_SECONDS["first-of-class"] + PRIOR_SECONDS.later + 3 * 10;
    expect(view.remaining).toBe(
      `${durationText(expected).replace(/^a/, "A")} left (a first estimate from other machines' runs; it is refined as cases finish)`,
    );
  });

  it("shows the case being measured: its place, sweep, id and kind, then the step it is at in words", () => {
    const acquiring = progressView(PLAN, fold([[1000, { kind: "case", index: 0 }]]), 1500);
    expect(acquiring.current).toEqual({
      title: `Case 1 of 4 · GEMV · ${FIRST} (the candidates its sentinels predict)`,
      text: "Acquiring the GPU",
      done: 0,
      total: 1,
    });
    const stepping = fold([
      [1000, { kind: "case", index: 0 }],
      [
        1000,
        {
          kind: "progress",
          sweep: "gemv",
          progress: {
            kind: "plan",
            caseId: GEMV_A,
            index: 0,
            rowCounts: [
              { m: 1, kernels: 30 },
              { m: 4, kernels: 9 },
              { m: 8, kernels: 9 },
            ],
          },
        },
      ],
      [
        1100,
        { kind: "progress", sweep: "gemv", progress: { kind: "status", message: "M=4 rounds" } },
      ],
      [
        1200,
        {
          kind: "progress",
          sweep: "gemv",
          progress: { kind: "step", caseId: GEMV_A, m: 4, phase: "rounds", done: 24, total: 60 },
        },
      ],
    ]);
    expect(progressView(PLAN, stepping, 2000).current).toEqual({
      title: `Case 1 of 4 · GEMV · ${FIRST} (the candidates its sentinels predict)`,
      text: "M = 4 (row count 2 of 3) · Timing rounds: 24 of 60",
      done: 24,
      total: 60,
      detail: "M=4 rounds",
    });
  });

  it("names a geometry case's planned and skipped geometries and its phase", () => {
    const state = fold([
      [0, { kind: "case", index: 3 }],
      [
        0,
        {
          kind: "progress",
          sweep: "geometry",
          progress: { kind: "plan", caseId: CONV3D_B, index: 0, geometries: 12, skipped: 36 },
        },
      ],
      [
        0,
        {
          kind: "progress",
          sweep: "geometry",
          progress: { kind: "step", caseId: CONV3D_B, phase: "timing", done: 3, total: 12 },
        },
      ],
    ]);
    expect(progressView(PLAN, state, 0).current?.text).toBe(
      "12 geometries, 36 skipped · Timing the geometries that go on: 3 of 12",
    );
    expect(progressView(PLAN, state, 0).current?.title).toBe(
      `Case 4 of 4 · tile geometry · ${FOURTH} (the candidates its class kept)`,
    );
  });

  it("counts a finished case in the whole run and refines the time left from it", () => {
    const state = fold(gemvCase(0, GEMV_A, 1000, 21_000));
    const view = progressView(PLAN, state, 21_000);
    expect(view.overall.done).toBe(1);
    expect(view.overall.text).toBe("1 of 4 cases done (GEMV 1 of 2 · tile geometry 0 of 2)");
    expect(state.finished).toEqual(new Map([[0, 20]]));
    // Case 2 at the measured 20 s, the geometry cases at their priors, three idle waits
    const left = 20 + PRIOR_SECONDS["first-of-class"] + PRIOR_SECONDS.later + 3 * 10;
    expect(view.remaining).toBe(`${durationText(left).replace(/^a/, "A")} left`);
  });

  it("counts the idle wait down before the next case", () => {
    const state = fold([
      ...gemvCase(0, GEMV_A, 0, 20_000),
      [20_000, { kind: "idle", index: 1, until: 30_000 }],
    ]);
    const view = progressView(PLAN, state, 23_500);
    expect(view.current).toEqual({
      title: `Case 2 of 4 · GEMV · ${SECOND} (the candidates its sentinels predict)`,
      text: "Cooling down before it: 7 s left",
      done: 3500,
      total: 10_000,
    });
  });

  it("counts the patterns as cases finish, live, in the derivations' words", () => {
    const state = fold([
      ...gemvCase(0, GEMV_A, 0, 1000),
      ...gemvCase(1, GEMV_B, 2000, 3000),
      [4000, { kind: "case", index: 2 }],
      [
        4000,
        {
          kind: "progress",
          sweep: "geometry",
          progress: { kind: "row", row: geometryRow(CONV3D_A, true) },
        },
      ],
      [
        4000,
        {
          kind: "progress",
          sweep: "geometry",
          progress: { kind: "row", row: geometryRow(CONV3D_A, false) },
        },
      ],
      [
        5000,
        {
          kind: "progress",
          sweep: "geometry",
          progress: { kind: "case", summary: { caseId: CONV3D_A, elapsedMs: 1000, screened: 0 } },
        },
      ],
      [5000, { kind: "record", sweep: "geometry", report: geometryReport(CONV3D_A) }],
    ]);
    expect(state.patterns).toEqual({
      gemv: { atMargin: 2, atLowest: 2, of: 2 },
      geometry: { found: 1, of: 1 },
    });
    // The GEMV count at both margins, then the run's class (two groups: none yet)
    expect(progressView(PLAN, state, 5000).patterns).toEqual([
      "GEMV: 2 of 2 shapes measured so far have a kernel that beats today's choice by ×1.065 times its timing noise (2 by ×1.030)",
      "Too few GEMV groups so far to tell whether the timings hold steady (2 of the 20 needed)",
      "Tile geometry: 1 of 1 classes measured so far have a geometry at least ×1.05 faster than the default in every case",
    ]);
  });

  it("says a steady run is enough for a table only when the derivation would read it so: GPU timestamps in ns, not rounded (faults: rounded, the wall clock, raw ticks)", () => {
    const steady = { class: "steady", groups: 40, unsteadyGroups: 1, share: 0.025 } as const;
    expect(steadinessLine(steady, { unit: "ns", quantized: false })).toContain(
      "this run alone is enough for a table",
    );
    for (const timing of [
      { unit: "ns", quantized: true },
      { unit: "wall", quantized: false },
    ]) {
      const line = steadinessLine(steady, timing);
      expect(line).not.toContain("enough");
      expect(line).toContain("no evidence for a table");
    }
    const ticks = steadinessLine(steady, { unit: "deno-raw-tick", quantized: false });
    expect(ticks).not.toContain("enough");
    expect(ticks).toContain("a table needs a second such run");
  });

  it("keeps the finished GEMV cases' timing as the merged record states it: rounded when any case was", () => {
    const rounded = fold([
      ...gemvCase(0, GEMV_A, 0, 1000),
      [2000, { kind: "case", index: 1 }],
      [
        3000,
        {
          kind: "record",
          sweep: "gemv",
          report: {
            ...gemvReport(GEMV_B),
            gpuTiming: { ...gemvReport(GEMV_B).gpuTiming, quantized: true },
          },
        },
      ],
    ]);
    expect(rounded.gemvTiming).toEqual({ unit: "ns", quantized: true });
  });

  it("does not count the rows of a case that did not finish", () => {
    const state = fold([
      [0, { kind: "case", index: 0 }],
      [
        0,
        {
          kind: "progress",
          sweep: "gemv",
          progress: { kind: "row", row: gemvRow(GEMV_A, "identical", { perDispatch: 1 }) },
        },
      ],
    ]);
    expect(state.patterns.gemv).toEqual({ atMargin: 0, atLowest: 0, of: 0 });
  });

  it("does not count a case its sweep stopped midway as finished, nor its patterns — but its finished groups count in the run's class, as the merged record counts them (fault: the class line over finished cases alone)", () => {
    const state = fold([...gemvCase(0, GEMV_A, 0, 1000, true)]);
    const view = progressView(PLAN, state, 1000);
    expect(view.overall.done).toBe(0);
    expect(state.finished.size).toBe(0);
    expect(state.patterns.gemv).toEqual({ atMargin: 0, atLowest: 0, of: 0 });
    expect(view.current).toBeUndefined();
    // The class the page states is the class the merged record states
    const finished = gemvReport(GEMV_B);
    const stopped = gemvReport(GEMV_A, { aborted: true });
    const merged = mergeGemvSweepReports([finished, stopped], {
      plannedCaseIds: [GEMV_B, GEMV_A],
      cancelledBetweenCases: false,
    });
    const both = fold([
      [0, { kind: "case", index: 0 }],
      [1000, { kind: "record", sweep: "gemv", report: finished }],
      [2000, { kind: "case", index: 1 }],
      [3000, { kind: "record", sweep: "gemv", report: stopped }],
    ]);
    expect(gemvSteadiness(both.gemvSummaries)).toEqual(merged.steadiness);
    expect(both.gemvSummaries).toEqual(merged.summaries);
  });

  it("warns of a device loss", () => {
    const state = fold([
      [0, { kind: "case", index: 0 }],
      [
        0,
        {
          kind: "progress",
          sweep: "gemv",
          progress: { kind: "deviceLost", reason: "unknown", message: "gone" },
        },
      ],
    ]);
    expect(progressView(PLAN, state, 0).warning).toBe(
      "The GPU device was lost (unknown): gone. The run stops after this case.",
    );
  });
});

describe("the progress view of the grid set", () => {
  /** A run of one tile geometry case, then the grid set's two cases. */
  const GRID_PLAN = planBenchmark(
    { gemv: true, geometry: true, grid: true },
    { cases: [CONV3D_A, GRID_A, GRID_B] },
  );

  /** Grid case `index` of the run (`caseId`) started at `at` and finished at `end`. */
  const gridCase = (
    index: number,
    caseId: string,
    at: number,
    end: number,
  ): (readonly [number, BenchmarkEvent])[] => [
    [at, { kind: "case", index }],
    [
      at,
      {
        kind: "progress",
        sweep: "grid",
        progress: { kind: "plan", caseId, index: 0, geometries: 5, skipped: 0 },
      },
    ],
    [
      end,
      {
        kind: "progress",
        sweep: "grid",
        progress: { kind: "case", summary: { caseId, elapsedMs: end - at, screened: 0 } },
      },
    ],
    [end, { kind: "record", sweep: "grid", report: gridReport(caseId) }],
  ];

  it("names a grid case's sweep and kind, and its timing as rounds of every tile", () => {
    const state = fold([
      [0, { kind: "case", index: 1 }],
      [
        0,
        {
          kind: "progress",
          sweep: "grid",
          progress: { kind: "plan", caseId: GRID_A, index: 0, geometries: 5, skipped: 0 },
        },
      ],
      [
        0,
        {
          kind: "progress",
          sweep: "grid",
          progress: { kind: "step", caseId: GRID_A, phase: "timing", done: 3, total: 5 },
        },
      ],
    ]);
    const view = progressView(GRID_PLAN, state, 0);
    expect(view.current?.title).toBe(
      `Case 2 of 3 · GEMM tile ladder · ${GRID_A} (every tile of the ladder)`,
    );
    expect(view.current?.text).toBe(
      "5 geometries · Timing rounds of every tile, interleaved: 3 of 5",
    );
  });

  it("counts a finished grid case in the whole run but not among the tile geometry's classes, and gives the grid set no count of its own (fault: the grid record read as a geometry class)", () => {
    const state = fold(gridCase(1, GRID_A, 0, 12_000));
    expect(state.finished).toEqual(new Map([[1, 12]]));
    expect(state.gridCases).toBe(1);
    expect(state.patterns.geometry).toEqual({ found: 0, of: 0 });
    expect(state.geometryRecords).toEqual([]);
    const view = progressView(GRID_PLAN, state, 12_000);
    expect(view.overall.text).toBe(
      "1 of 3 cases done (tile geometry 0 of 1 · GEMM tile ladder 1 of 2)",
    );
    expect(view.patterns.at(-1)).toBe(
      "GEMM tile ladder: 1 of 2 cases measured so far — which tile suits which grid size is decided from all of them together, after the run",
    );
    // What is left is the second grid case, at the first one's measured 12 s
    expect(view.remaining).toBe(`${durationText(12).replace(/^u/, "U")} left`);
  });

  it("does not count a grid case a cancel cut short", () => {
    const state = fold([
      [0, { kind: "case", index: 2 }],
      [500, { kind: "record", sweep: "grid", report: { ...gridReport(GRID_B), aborted: true } }],
    ]);
    expect(state.finished.size).toBe(0);
    expect(state.gridCases).toBe(0);
  });
});
