import type {
  GeometrySweepOptions,
  GeometrySweepProgress,
  GeometrySweepReport,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import {
  foldSweepEvent,
  idle,
  INITIAL_LIVE,
  runSweepPlan,
  type SweepRunEvent,
} from "../src/models/geometry-sweep/run.ts";
import type { SweepPlan } from "../src/models/geometry-sweep/settings.ts";
import { type PerCaseOptions, perCase, row } from "./helpers/geometry-sweep-report.ts";

const PLAN: SweepPlan = {
  caseIds: ["a", "b", "c"],
  ops: ["i8a8-conv3d"],
  candidateSet: "full",
  rounds: 5,
  idleSeconds: 90,
};

/**
 * The complete record of the case a call swept (the run reads its rows, `aborted` and device loss).
 * The fake case ids are no case of the runtime's table, so the record carries a placeholder
 * `candidateKernels`.
 */
const reportFor = (
  options: GeometrySweepOptions,
  extra: Pick<PerCaseOptions, "deviceLost" | "rows" | "aborted"> = {},
): GeometrySweepReport =>
  perCase(options.cases?.[0] ?? "", { ...extra, candidateKernels: "unused" });

/** A fake sweep and idle wait that log what the run did, in order. */
const harness = (
  sweepBehavior: (options: GeometrySweepOptions, call: number) => Promise<GeometrySweepReport>,
) => {
  const log: string[] = [];
  const calls: GeometrySweepOptions[] = [];
  const events: SweepRunEvent[] = [];
  return {
    log,
    calls,
    events,
    deps: (signal: AbortSignal) => ({
      signal,
      onEvent: (event: SweepRunEvent) => events.push(event),
      sweep: (options: GeometrySweepOptions) => {
        calls.push(options);
        log.push(`sweep ${options.cases?.join(",") ?? ""}`);
        return sweepBehavior(options, calls.length);
      },
      wait: (ms: number, waitSignal: AbortSignal) => {
        log.push(`idle ${ms}`);
        return idle(0, waitSignal);
      },
      now: () => 1000,
    }),
  };
};

describe("runSweepPlan", () => {
  describe("when every case runs", () => {
    it("calls the sweep once per case in order, idling before every case but the first", async () => {
      const run = harness(async (options) => reportFor(options));
      const outcome = await runSweepPlan(PLAN, run.deps(new AbortController().signal));
      expect(run.log).toEqual(["sweep a", "idle 90000", "sweep b", "idle 90000", "sweep c"]);
      expect(outcome.stop).toEqual({ kind: "completed" });
      expect(outcome.reports).toHaveLength(3);
    });

    it("passes the plan's settings and its signal to every call", async () => {
      const run = harness(async (options) => reportFor(options));
      const signal = new AbortController().signal;
      await runSweepPlan(PLAN, run.deps(signal));
      expect(run.calls[1]).toMatchObject({
        candidateSet: "full",
        ops: ["i8a8-conv3d"],
        cases: ["b"],
        rounds: 5,
        signal,
      });
    });

    it("reports the idle's end and each case's start", async () => {
      const run = harness(async (options) => reportFor(options));
      await runSweepPlan({ ...PLAN, caseIds: ["a", "b"] }, run.deps(new AbortController().signal));
      expect(run.events).toEqual([
        { kind: "case", index: 0, caseId: "a" },
        { kind: "idle", index: 1, until: 91_000 },
        { kind: "case", index: 1, caseId: "b" },
      ]);
    });

    it("does not idle at all with 0 idle seconds", async () => {
      const run = harness(async (options) => reportFor(options));
      await runSweepPlan({ ...PLAN, idleSeconds: 0 }, run.deps(new AbortController().signal));
      expect(run.log).toEqual(["sweep a", "sweep b", "sweep c"]);
    });
  });

  describe("when Cancel comes during a case", () => {
    it("keeps that case's record and starts no further case", async () => {
      const controller = new AbortController();
      const run = harness(async (options, call) => {
        if (call === 1) controller.abort(new Error("Cancelled by the user"));
        return reportFor(options);
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(run.log).toEqual(["sweep a"]);
      expect(outcome.reports).toHaveLength(1);
      expect(outcome.stop).toEqual({ kind: "cancelled" });
    });
  });

  describe("when Cancel comes while a case acquires its device (before any geometry is measured)", () => {
    it("drops that case's record without a row and starts no further case", async () => {
      const controller = new AbortController();
      const run = harness(async (options, call) => {
        if (call === 2) {
          controller.abort(new Error("Cancelled by the user"));
          return reportFor(options, { rows: [], aborted: true });
        }
        return reportFor(options);
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(run.log).toEqual(["sweep a", "idle 90000", "sweep b"]);
      expect(outcome.reports.map((report) => report.settings.cases)).toEqual([["a"]]);
      expect(outcome.stop).toEqual({ kind: "cancelled" });
    });
  });

  describe("when Cancel comes after the last case has measured everything", () => {
    it("completes, as every planned case ran to its end", async () => {
      const controller = new AbortController();
      const run = harness(async (options, call) => {
        if (call === 3) controller.abort(new Error("Cancelled by the user"));
        return reportFor(options);
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(outcome.reports).toHaveLength(3);
      expect(outcome.stop).toEqual({ kind: "completed" });
    });

    it("is cancelled when the last case's record says the cancel left something out", async () => {
      const controller = new AbortController();
      const run = harness(async (options, call) => {
        if (call !== 3) return reportFor(options);
        controller.abort(new Error("Cancelled by the user"));
        return reportFor(options, { aborted: true });
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(outcome.reports).toHaveLength(3);
      expect(outcome.stop).toEqual({ kind: "cancelled" });
    });
  });

  describe("when Cancel comes during the idle wait", () => {
    it("ends at once without the next case", async () => {
      const controller = new AbortController();
      const run = harness(async (options) => reportFor(options));
      const deps = run.deps(controller.signal);
      const outcome = await runSweepPlan(PLAN, {
        ...deps,
        wait: (ms, signal) => {
          run.log.push(`idle ${ms}`);
          const waiting = idle(60_000, signal);
          controller.abort(new Error("Cancelled by the user"));
          return waiting;
        },
      });
      expect(run.log).toEqual(["sweep a", "idle 90000"]);
      expect(outcome.reports).toHaveLength(1);
      expect(outcome.stop).toEqual({ kind: "cancelled" });
    });
  });

  describe("when the signal was aborted before the run started (the page left while it waited for the GPU queue)", () => {
    it("acquires nothing", async () => {
      const controller = new AbortController();
      controller.abort(new Error("The page was left during the sweep"));
      const run = harness(async (options) => reportFor(options));
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(run.log).toEqual([]);
      expect(outcome).toEqual({ reports: [], stop: { kind: "cancelled" } });
    });
  });

  describe("when the device is lost in a case", () => {
    it("stops after that case's record", async () => {
      const lost = { reason: "unknown", message: "GPU process crashed" };
      const run = harness(async (options, call) =>
        reportFor(options, { deviceLost: call === 2 ? lost : null }),
      );
      const outcome = await runSweepPlan(PLAN, run.deps(new AbortController().signal));
      expect(run.log).toEqual(["sweep a", "idle 90000", "sweep b"]);
      expect(outcome.reports).toHaveLength(2);
      expect(outcome.stop).toEqual({ kind: "deviceLost", ...lost });
    });
  });

  describe("when a sweep throws (no device could be acquired)", () => {
    it("keeps the records before and returns the error", async () => {
      const failure = new Error("GpuUnavailableError");
      const run = harness(async (options, call) => {
        if (call === 2) throw failure;
        return reportFor(options);
      });
      const outcome = await runSweepPlan(PLAN, run.deps(new AbortController().signal));
      expect(outcome.reports).toHaveLength(1);
      expect(outcome.stop).toEqual({ kind: "error", error: failure });
    });

    it("still returns the error when Cancel came before it was thrown", async () => {
      const controller = new AbortController();
      const failure = new Error("GpuUnavailableError");
      const run = harness(async () => {
        controller.abort(new Error("Cancelled by the user"));
        throw failure;
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(outcome).toEqual({ reports: [], stop: { kind: "error", error: failure } });
    });

    it("is cancelled when what it throws is the signal's own reason (the abort itself)", async () => {
      const controller = new AbortController();
      const reason = new Error("Cancelled by the user");
      const run = harness(async () => {
        controller.abort(reason);
        throw reason;
      });
      const outcome = await runSweepPlan(PLAN, run.deps(controller.signal));
      expect(outcome).toEqual({ reports: [], stop: { kind: "cancelled" } });
    });
  });
});

describe("idle", () => {
  it("rejects with the signal's reason when aborted while waiting", async () => {
    const controller = new AbortController();
    const reason = new Error("Cancelled by the user");
    const waiting = idle(60_000, controller.signal);
    controller.abort(reason);
    await expect(waiting).rejects.toBe(reason);
  });

  it("rejects at once on an aborted signal", async () => {
    const controller = new AbortController();
    controller.abort("gone");
    await expect(idle(60_000, controller.signal)).rejects.toBe("gone");
  });
});

describe("foldSweepEvent (the live view)", () => {
  const measured = row("a", true);
  const sweep = (progress: GeometrySweepProgress): SweepRunEvent => ({ kind: "sweep", progress });

  it("follows the run: waiting → case → status, rows and the finished case → idle → next case", () => {
    const events: SweepRunEvent[] = [
      { kind: "case", index: 0, caseId: "a" },
      sweep({
        kind: "started",
        adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
        unit: "ns",
        dp4a: true,
        caseCount: 1,
      }),
      sweep({ kind: "status", message: "a: g (1/48)" }),
      sweep({ kind: "row", row: measured }),
      sweep({ kind: "case", summary: { caseId: "a", elapsedMs: 1 } }),
      { kind: "idle", index: 1, until: 5000 },
    ];
    const states = events.reduce(
      (all, event) => [...all, foldSweepEvent(all.at(-1) ?? INITIAL_LIVE, event)],
      [INITIAL_LIVE],
    );
    expect(states.map(({ phase }) => phase)).toEqual([
      { kind: "waiting" },
      { kind: "sweeping" },
      { kind: "sweeping" },
      { kind: "sweeping", status: "a: g (1/48)" },
      { kind: "sweeping", status: "a: g (1/48)" },
      { kind: "sweeping", status: "a: g (1/48)" },
      { kind: "idle", until: 5000 },
    ]);
    const last = states.at(-1);
    expect(last).toMatchObject({ caseIndex: 1, unit: "ns", dp4a: true, rows: [measured] });
    expect(last?.cases).toEqual([{ caseId: "a", elapsedMs: 1 }]);
  });

  it("keeps a device loss", () => {
    const live = foldSweepEvent(
      INITIAL_LIVE,
      sweep({ kind: "deviceLost", reason: "unknown", message: "crashed" }),
    );
    expect(live.deviceLost).toEqual({ reason: "unknown", message: "crashed" });
  });
});
