import {
  type GemvSweepOptions,
  type GemvSweepReport,
  gemvSweepCases,
  QUICK_LOOK_MARGIN,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { parseParameters } from "../src/entry.ts";
import entry from "../src/entries/gemv-sweep.ts";
import {
  foldGemvEvent,
  type GemvRunEvent,
  type GemvLive,
  INITIAL_GEMV_LIVE,
  runGemvPlan,
} from "../src/gemv-sweep/run.ts";
import {
  GEMV_CANDIDATE_NAMES,
  GEMV_CASE_IDS,
  GEMV_ROW_COUNTS,
  type GemvSweepPlan,
  gemvSweepSelection,
  planGemvSweep,
} from "../src/gemv-sweep/settings.ts";
import { gemvProgressLine } from "../src/gemv-sweep/summary.ts";
import { idle } from "../src/per-case-sweep.ts";
import { CASE_A, CASE_B, gemvPerCase } from "./helpers/gemv-sweep-report.ts";

const PLAN: GemvSweepPlan = {
  caseIds: [CASE_A, CASE_B],
  candidateSet: "full",
  rows: [],
  candidates: [],
  quickLookMargin: 1.25,
  rounds: 3,
  idleSeconds: 10,
};

describe("the gemv-sweep entry's settings", () => {
  it("default to the quick set over every case of the runtime's table, every row count and candidate, the quick look at ×1.25, 5 rounds and 10 s idle", () => {
    const plan = planGemvSweep(gemvSweepSelection(parseParameters(entry.parameters, {})));
    expect(plan).toEqual({
      caseIds: gemvSweepCases().map(({ id }) => id),
      candidateSet: "quick",
      rows: [],
      candidates: [],
      quickLookMargin: QUICK_LOOK_MARGIN,
      rounds: 5,
      idleSeconds: 10,
    });
    expect(GEMV_CASE_IDS).toEqual(plan.caseIds);
  });

  it("take the row counts as numbers, the candidates by name and the margin, and refuse a margin between 0 and 1", () => {
    const plan = planGemvSweep(
      gemvSweepSelection(
        parseParameters(entry.parameters, {
          candidateSet: "row-blocks",
          rows: "64,16",
          candidates: "c32u4r2,c32u4r1",
          quickLookMargin: "0",
        }),
      ),
    );
    // Choices come back in the order of their options
    expect([plan.candidateSet, plan.rows, plan.quickLookMargin]).toEqual([
      "row-blocks",
      [16, 64],
      0,
    ]);
    expect([...plan.candidates].sort()).toEqual(["c32u4r1", "c32u4r2"]);
    expect(GEMV_ROW_COUNTS).toEqual(["1", "4", "8", "16", "64"]);
    expect(GEMV_CANDIDATE_NAMES).toContain("staged-l32");
    expect(() =>
      gemvSweepSelection(parseParameters(entry.parameters, { quickLookMargin: "0.5" })),
    ).toThrow(/Parameter quickLookMargin: 0.5 is neither 0/);
    expect(() => parseParameters(entry.parameters, { candidates: "parallel-l3" })).toThrow(
      /Parameter candidates/,
    );
  });

  it("keep the chosen cases in the table's order", () => {
    const plan = planGemvSweep(
      gemvSweepSelection(
        parseParameters(entry.parameters, { cases: `${CASE_B},${CASE_A}`, candidateSet: "full" }),
      ),
    );
    const order = GEMV_CASE_IDS.filter((id) => id === CASE_A || id === CASE_B);
    expect(plan.caseIds).toEqual(order);
    expect(plan.candidateSet).toBe("full");
  });

  it("refuse a case the table does not have and a set that does not exist", () => {
    expect(() => parseParameters(entry.parameters, { cases: "i4g32-n1-k1" })).toThrow(
      /Parameter cases/,
    );
    expect(() => parseParameters(entry.parameters, { candidateSet: "quick+" })).toThrow(
      /Parameter candidateSet/,
    );
  });
});

describe("runGemvPlan", () => {
  const harness = () => {
    const log: string[] = [];
    const calls: GemvSweepOptions[] = [];
    const events: GemvRunEvent[] = [];
    return {
      log,
      calls,
      events,
      deps: (signal: AbortSignal) => ({
        signal,
        onEvent: (event: GemvRunEvent) => events.push(event),
        sweep: async (options: GemvSweepOptions): Promise<GemvSweepReport> => {
          calls.push(options);
          log.push(`sweep ${options.cases?.join(",") ?? ""}`);
          return gemvPerCase(options.cases?.[0] ?? "");
        },
        wait: (ms: number, waitSignal: AbortSignal) => {
          log.push(`idle ${ms}`);
          return idle(0, waitSignal);
        },
        now: () => 1000,
      }),
    };
  };

  it("calls runGemvSweep once per case with the plan's set and rounds, idling before every case but the first", async () => {
    const run = harness();
    const signal = new AbortController().signal;
    const outcome = await runGemvPlan(PLAN, run.deps(signal));
    expect(run.log).toEqual([`sweep ${CASE_A}`, "idle 10000", `sweep ${CASE_B}`]);
    expect(run.calls[1]).toMatchObject({
      candidateSet: "full",
      cases: [CASE_B],
      quickLookMargin: 1.25,
      rounds: 3,
      signal,
    });
    // Nothing narrowed: no row counts and no candidates are passed
    expect(Object.keys(run.calls[1] ?? {})).not.toContain("rows");
    expect(Object.keys(run.calls[1] ?? {})).not.toContain("candidates");
    expect(outcome.stop).toEqual({ kind: "completed" });
    expect(outcome.reports).toHaveLength(2);
  });

  it("passes the plan's row counts and candidates to every call", async () => {
    const run = harness();
    const outcome = await runGemvPlan(
      { ...PLAN, rows: [4], candidates: ["staged-l8"], quickLookMargin: 0 },
      run.deps(new AbortController().signal),
    );
    expect(run.calls).toHaveLength(2);
    for (const call of run.calls) {
      expect(call).toMatchObject({ rows: [4], candidates: ["staged-l8"], quickLookMargin: 0 });
    }
    expect(outcome.stop).toEqual({ kind: "completed" });
  });

  it("stops after the case the device was lost in (the shared loop's rule)", async () => {
    const lost = { reason: "unknown", message: "GPU process crashed" };
    const run = harness();
    const outcome = await runGemvPlan(PLAN, {
      ...run.deps(new AbortController().signal),
      sweep: async (options) => gemvPerCase(options.cases?.[0] ?? "", { deviceLost: lost }),
    });
    expect(outcome.reports).toHaveLength(1);
    expect(outcome.stop).toEqual({ kind: "deviceLost", ...lost });
  });
});

describe("the live view and its line", () => {
  const after = (events: readonly GemvRunEvent[]): GemvLive =>
    events.reduce(foldGemvEvent, INITIAL_GEMV_LIVE);

  it("names the case, its rows, the sweep's status and the timing with the subgroup gate", () => {
    const live = after([
      { kind: "case", index: 1, caseId: CASE_B },
      {
        kind: "sweep",
        progress: {
          kind: "started",
          adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "" },
          unit: "ns",
          subgroups: false,
          caseCount: 1,
        },
      },
      { kind: "sweep", progress: { kind: "status", message: `${CASE_B} M=1: c32u4 (1/20)` } },
    ]);
    expect(gemvProgressLine(PLAN, live, 0)).toBe(
      `0 / 2 cases done · case 2 / 2 ${CASE_B} · 0 rows · ${CASE_B} M=1: c32u4 (1/20) · timing ns, subgroups false`,
    );
  });

  it("counts a finished case and the idle wait down", () => {
    const live = after([
      { kind: "sweep", progress: { kind: "case", caseId: CASE_A, summaries: [] } },
      { kind: "idle", index: 1, until: 11_000 },
    ]);
    expect(gemvProgressLine(PLAN, live, 1000)).toBe(
      "1 / 2 cases done · idle before case 2 / 2: 10 s left",
    );
  });
});
