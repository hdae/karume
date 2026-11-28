import type { GemvSweepReport, GeometrySweepReport } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import type { PageFacts } from "../src/bundle.ts";
import { type BenchmarkDeps, runBenchmark, type StampedEvent } from "../src/controller.ts";
import { planBenchmark } from "../src/plan.ts";
import {
  CHECKOUT,
  CONV3D_A,
  CONV3D_B,
  ENVIRONMENT,
  fakeGemvSweep,
  fakeGeometrySweep,
  GEMV_A,
  GEMV_B,
  gemvReport,
  GRID_A,
  GRID_B,
  noWait,
} from "./helpers/fakes.ts";

const PLAN = planBenchmark(
  { gemv: true, geometry: true, grid: true },
  { cases: [GEMV_A, GEMV_B, CONV3D_A, CONV3D_B, GRID_A, GRID_B], idleSeconds: 3 },
);

const PAGE: PageFacts = { hiddenCount: 0, hiddenSeconds: 0, wakeLock: "held" };

/** A run of {@link PLAN} on fake sweeps (the shared per-case loop is the real one). */
const harness = (
  options: {
    readonly gemv?: (caseId: string) => GemvSweepReport;
    readonly geometry?: (caseId: string) => GeometrySweepReport;
    /** Aborts the run during the n-th GEMV call (1-based), which then measures nothing. */
    readonly abortAtGemvCall?: number;
    /**
     * Aborts the run during the n-th geometry call (1-based, the grid set's calls after the explore
     * set's) after it measured everything — a cancel while the case re-measures its default, where
     * the sweep takes no signal.
     */
    readonly abortAfterGeometryCall?: number;
    /** The page's download (default: kept in `downloads`). */
    readonly download?: (fileName: string, text: string) => void;
  } = {},
) => {
  const controller = new AbortController();
  const events: StampedEvent[] = [];
  const downloads: { readonly fileName: string; readonly text: string }[] = [];
  const waits: number[] = [];
  let gemvCalls = 0;
  let geometryCalls = 0;
  let clock = Date.parse("2026-11-26T12:00:00.000Z");
  const gemvSweep = fakeGemvSweep(options.gemv);
  const geometrySweep = fakeGeometrySweep(options.geometry);
  const deps: BenchmarkDeps = {
    signal: controller.signal,
    onEvent: (event) => events.push(event),
    readEnvironment: () => Promise.resolve(ENVIRONMENT),
    checkout: CHECKOUT,
    download: options.download ?? ((fileName, text) => downloads.push({ fileName, text })),
    pageFacts: () => PAGE,
    gemvSweep: (sweepOptions) => {
      gemvCalls += 1;
      if (gemvCalls !== options.abortAtGemvCall) return gemvSweep(sweepOptions);
      // A cancel during a case: the sweep stops, and the row count it was in leaves no row
      controller.abort();
      const [caseId = ""] = sweepOptions.cases ?? [];
      return Promise.resolve(gemvReport(caseId, { rows: [], aborted: true }));
    },
    geometrySweep: async (sweepOptions) => {
      geometryCalls += 1;
      const report = await geometrySweep(sweepOptions);
      if (geometryCalls === options.abortAfterGeometryCall) controller.abort();
      return report;
    },
    wait: (ms, signal) => {
      waits.push(ms);
      return noWait(ms, signal);
    },
    now: () => 1000 + events.length,
    clock: () => new Date((clock += 60_000)),
  };
  return {
    run: () => runBenchmark(PLAN, deps),
    events,
    downloads,
    waits,
    calls: () => ({ gemv: gemvCalls, geometry: geometryCalls }),
  };
};

describe("runBenchmark", () => {
  it("runs the three sweeps to the end and downloads one complete bundle once, its records one per sweep", async () => {
    const run = harness();
    const ending = await run.run();
    expect(run.calls()).toEqual({ gemv: 2, geometry: 4 });
    expect(run.downloads).toHaveLength(1);
    expect(run.downloads[0]).toEqual({ fileName: ending.fileName, text: ending.text });
    expect(ending.downloaded).toBe(true);
    expect(ending.bundle.complete).toBe(true);
    expect(ending.bundle.stop).toEqual({ kind: "completed" });
    expect(
      ending.bundle.records.map((record) => [record.entry.id, record.outcome, record.verdict]),
    ).toEqual([
      ["gemv-sweep", "done", "pass"],
      ["geometry-sweep", "done", "pass"],
      ["geometry-sweep", "done", "pass"],
    ]);
    // The geometry sweep's two records: the explore set's, then the grid set's
    expect(ending.bundle.records.map((record) => record.entry.parameters["candidateSet"])).toEqual([
      "explore",
      "explore",
      "grid",
    ]);
    expect(ending.bundle.sweeps).toEqual(["gemv-sweep", "geometry-sweep", "geometry-sweep"]);
    expect(ending.fileName).toBe(
      "karume-benchmark_browser-nvidia-geforce-rtx-5070-ti_2026-11-26T12-02-00.json",
    );
  });

  it("idles before every case but the run's first, a later sweep's first case included", async () => {
    const run = harness();
    await run.run();
    expect(run.waits).toEqual([3000, 3000, 3000, 3000, 3000]);
    const idles = run.events.flatMap(({ event }) => (event.kind === "idle" ? [event.index] : []));
    expect(idles).toEqual([1, 2, 3, 4, 5]);
  });

  it("counts the case events over the whole run, and hands every sweep event on", async () => {
    const run = harness();
    await run.run();
    const cases = run.events.flatMap(({ event }) => (event.kind === "case" ? [event.index] : []));
    expect(cases).toEqual([0, 1, 2, 3, 4, 5]);
    const done = run.events.filter(
      ({ event }) => event.kind === "progress" && event.progress.kind === "case",
    );
    expect(done).toHaveLength(6);
    // The grid set's events are told from the explore set's
    expect(
      run.events.flatMap(({ event }) =>
        event.kind === "progress" && event.progress.kind === "case" ? [event.sweep] : [],
      ),
    ).toEqual(["gemv", "gemv", "geometry", "geometry", "grid", "grid"]);
  });

  it("on cancel keeps what was measured, downloads nothing and offers it, marked incomplete; the next sweep does not run", async () => {
    const run = harness({ abortAtGemvCall: 2 });
    const ending = await run.run();
    expect(run.downloads).toEqual([]);
    expect(ending.downloaded).toBe(false);
    expect(ending.measured).toBe(true);
    expect(ending.bundle.complete).toBe(false);
    expect(ending.bundle.stop).toEqual({ kind: "cancelled" });
    expect(run.calls().geometry).toBe(0);
    expect(ending.bundle.records.map((record) => record.outcome)).toEqual(["cancelled"]);
    expect(ending.fileName).toMatch(/_incomplete\.json$/);
  });

  it("stops at a device loss after that case's record, offering the incomplete bundle", async () => {
    const lost = { reason: "unknown", message: "the device was lost" };
    const run = harness({
      gemv: (caseId) => gemvReport(caseId, caseId === GEMV_B ? { deviceLost: lost } : {}),
    });
    const ending = await run.run();
    expect(run.downloads).toEqual([]);
    expect(ending.bundle.stop).toEqual({ kind: "deviceLost", ...lost });
    expect(ending.bundle.records.map((record) => record.verdict)).toEqual(["incomplete"]);
    expect(run.calls().geometry).toBe(0);
    expect(ending.measured).toBe(true);
  });

  it("does not download a run cancelled while its last case re-measured: it is incomplete and offered", async () => {
    const run = harness({ abortAfterGeometryCall: 4 });
    const ending = await run.run();
    expect(run.calls()).toEqual({ gemv: 2, geometry: 4 });
    expect(run.downloads).toEqual([]);
    expect(ending.downloaded).toBe(false);
    expect(ending.measured).toBe(true);
    expect(ending.bundle.complete).toBe(false);
    expect(ending.bundle.stop).toEqual({ kind: "cancelled" });
    expect(ending.bundle.records.map((record) => record.outcome)).toEqual([
      "done",
      "done",
      "cancelled",
    ]);
    expect(ending.fileName).toMatch(/_incomplete\.json$/);
  });

  it("runs no later sweep once a cancel landed in an earlier one's last case", async () => {
    const run = harness({ abortAfterGeometryCall: 2 });
    const ending = await run.run();
    expect(run.calls()).toEqual({ gemv: 2, geometry: 2 });
    expect(ending.bundle.records.map((record) => record.outcome)).toEqual(["done", "cancelled"]);
    expect(ending.bundle.stop).toEqual({ kind: "cancelled" });
  });

  it("keeps a complete run's bundle to offer when its download throws, and says why", async () => {
    const run = harness({
      download: () => {
        throw new Error("no download here");
      },
    });
    const ending = await run.run();
    expect(ending.bundle.complete).toBe(true);
    expect(ending.downloaded).toBe(false);
    expect(ending.downloadError).toBe("no download here");
    expect(ending.text).toContain("karume-benchmark/1");
  });

  it("hands each case's record on as its sweep call returns it", async () => {
    const run = harness();
    await run.run();
    const records = run.events.flatMap(({ event }) =>
      event.kind === "record" ? [[event.sweep, event.report.settings.cases]] : [],
    );
    expect(records).toEqual(PLAN.cases.map(({ sweep, caseId }) => [sweep, [caseId]]));
  });

  it("measures nothing when cancelled before the first case, and says so", async () => {
    const run = harness({ abortAtGemvCall: 1 });
    const ending = await run.run();
    expect(ending.measured).toBe(false);
    expect(ending.bundle.stop).toEqual({ kind: "cancelled" });
    expect(run.downloads).toEqual([]);
  });
});
