import type { GemvSweepReport } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";
import { mergeGemvSweepReports } from "@karume/sweep-app-support/gemv-sweep/record";
import type { GemvRunOutcome } from "@karume/sweep-app-support/gemv-sweep/run";
import type { GemvSweepPlan } from "@karume/sweep-app-support/gemv-sweep/settings";
import { gemvSweepResult, gemvSweepVerdict } from "@karume/sweep-app-support/gemv-sweep/summary";
import type { PerCaseStop } from "@karume/sweep-app-support/per-case-sweep";

import { CASE_A, CASE_B, CASE_C, gemvPerCase, gemvRow } from "./helpers/gemv-sweep-report.ts";

const PLANNED = { plannedCaseIds: [CASE_A, CASE_B], cancelledBetweenCases: false };

const PLAN: GemvSweepPlan = {
  caseIds: [CASE_A, CASE_B],
  candidateSet: "quick",
  rows: [],
  candidates: [],
  quickLookMargin: 1.25,
  rounds: 5,
  idleSeconds: 10,
};

const COMPLETED: PerCaseStop = { kind: "completed" };

describe("mergeGemvSweepReports", () => {
  describe("when every planned case has its record", () => {
    const merged = mergeGemvSweepReports(
      [gemvPerCase(CASE_A), gemvPerCase(CASE_B, { date: "2026-11-17T10:09:00.000Z" })],
      PLANNED,
    );

    it("keeps the rows, cases and summaries in run order and the planned cases in the settings", () => {
      expect(merged.rows.map(({ caseId }) => caseId)).toEqual([
        CASE_A,
        CASE_A,
        CASE_A,
        CASE_B,
        CASE_B,
        CASE_B,
      ]);
      expect(merged.cases.map(({ id }) => id)).toEqual([CASE_A, CASE_B]);
      expect(merged.summaries.map(({ caseId }) => caseId)).toEqual([CASE_A, CASE_B]);
      expect(merged.settings.cases).toEqual([CASE_A, CASE_B]);
    });

    it("keeps each row's and each case's own fingerprint as the per-case records wrote them", () => {
      expect(merged.rows.map(({ kernel }) => kernel)).toEqual(
        [gemvPerCase(CASE_A), gemvPerCase(CASE_B)].flatMap(({ rows }) =>
          rows.map(({ kernel }) => kernel),
        ),
      );
      expect(merged.cases.map(({ fingerprint }) => fingerprint)).toEqual([
        gemvPerCase(CASE_A).cases[0]?.fingerprint,
        gemvPerCase(CASE_B).cases[0]?.fingerprint,
      ]);
      expect(Object.keys(merged)).not.toContain("kernels");
    });

    it("takes the first record's start and the last record's end", () => {
      expect(merged.startedAt).toBe("2026-11-17T10:00:00.000Z");
      expect(merged.date).toBe("2026-11-17T10:09:00.000Z");
      expect(merged.aborted).toBe(false);
      expect(merged.deviceLost).toBeNull();
    });
  });

  it("carries each case's baseline spread", () => {
    const merged = mergeGemvSweepReports(
      [gemvPerCase(CASE_A), gemvPerCase(CASE_B, { summary: { baselineSpread: 1.25 } })],
      PLANNED,
    );
    expect(merged.summaries.map(({ caseId, baselineSpread }) => [caseId, baselineSpread])).toEqual([
      [CASE_A, 1.004],
      [CASE_B, 1.25],
    ]);
  });

  it("marks the record aborted when Cancel stopped the sweep between cases, and keeps the first device loss", () => {
    const lost = { reason: "unknown", message: "GPU process crashed" };
    const cancelled = mergeGemvSweepReports([gemvPerCase(CASE_A)], {
      ...PLANNED,
      cancelledBetweenCases: true,
    });
    expect(cancelled.aborted).toBe(true);
    const lostRun = mergeGemvSweepReports([gemvPerCase(CASE_A, { deviceLost: lost })], PLANNED);
    expect(lostRun.deviceLost).toEqual(lost);
    expect(lostRun.aborted).toBe(false);
  });

  it("marks the record quantized when any record's rounds looked quantized", () => {
    const merged = mergeGemvSweepReports(
      [gemvPerCase(CASE_A), gemvPerCase(CASE_B, { quantized: true })],
      PLANNED,
    );
    expect(merged.gpuTiming.quantized).toBe(true);
  });

  it("states the class of the merged summaries, not the first record's (fault: one case's group alone)", () => {
    const base = gemvPerCase(CASE_A);
    const [summary] = base.summaries;
    const [entry] = base.cases;
    if (summary === undefined || entry === undefined) throw new Error("no summary or case");
    const ids = Array.from({ length: 20 }, (_, index) => `case-${index}`);
    const records = ids.map((id, index): GemvSweepReport => ({
      ...base,
      // One group each: none has a class of its own; one in five spreads beyond ×1.05
      steadiness: { class: "unclassified", groups: 1, unsteadyGroups: 0, share: 0 },
      summaries: [{ ...summary, caseId: id, baselineSpread: index % 5 === 0 ? 1.2 : 1.01 }],
      cases: [{ ...entry, id }],
      settings: { ...base.settings, cases: [id] },
      rows: [],
    }));
    const merged = mergeGemvSweepReports(records, {
      plannedCaseIds: ids,
      cancelledBetweenCases: false,
    });
    expect(merged.steadiness).toEqual({
      class: "steady",
      groups: 20,
      unsteadyGroups: 4,
      share: 0.2,
    });
  });

  it("keeps the session of its records, the first one's time since its start, and throws for another session (fault: two page loads)", () => {
    const session = (id: string, elapsedMs: number) => ({ session: { id, elapsedMs } });
    const merged = mergeGemvSweepReports(
      [
        { ...gemvPerCase(CASE_A), ...session("load-1", 100) },
        { ...gemvPerCase(CASE_B), ...session("load-1", 900) },
      ],
      PLANNED,
    );
    expect(merged.session).toEqual({ id: "load-1", elapsedMs: 100 });
    expect(() =>
      mergeGemvSweepReports(
        [
          { ...gemvPerCase(CASE_A), ...session("load-1", 100) },
          { ...gemvPerCase(CASE_B), ...session("load-2", 900) },
        ],
        PLANNED,
      ),
    ).toThrow(/Record 1 differs from record 0/);
  });

  describe("when the records do not belong to one sweep", () => {
    it("throws for another geometry profile (its GEMM references were built with another geometry)", () => {
      const other = gemvPerCase(CASE_B, { geometryProfile: "nvidia-ampere" });
      expect(() => mergeGemvSweepReports([gemvPerCase(CASE_A), other], PLANNED)).toThrow(
        /Record 1 differs from record 0/,
      );
      const same = [
        gemvPerCase(CASE_A, { geometryProfile: "nvidia-ampere" }),
        gemvPerCase(CASE_B, { geometryProfile: "nvidia-ampere" }),
      ];
      expect(mergeGemvSweepReports(same, PLANNED).geometryProfile).toBe("nvidia-ampere");
    });

    it("throws for other gates (another device's candidates)", () => {
      const other = gemvPerCase(CASE_B, {
        gates: {
          subgroups: false,
          features: { subgroups: false, subgroupSizeControl: false, subgroupId: false },
          maxComputeWorkgroupsPerDimension: 65535,
        },
      });
      expect(() => mergeGemvSweepReports([gemvPerCase(CASE_A), other], PLANNED)).toThrow(
        /Record 1 differs from record 0/,
      );
    });

    it("throws for another quick-look margin", () => {
      const base = gemvPerCase(CASE_B);
      const other = { ...base, settings: { ...base.settings, quickLookMargin: 0 } };
      expect(() => mergeGemvSweepReports([gemvPerCase(CASE_A), other], PLANNED)).toThrow(
        /Record 1 differs/,
      );
    });

    it("throws for an unplanned case, a case twice and no record", () => {
      expect(() => mergeGemvSweepReports([gemvPerCase(CASE_C)], PLANNED)).toThrow(
        /not a planned case/,
      );
      expect(() =>
        mergeGemvSweepReports([gemvPerCase(CASE_A), gemvPerCase(CASE_A)], PLANNED),
      ).toThrow(/has two records/);
      expect(() => mergeGemvSweepReports([], PLANNED)).toThrow(/No per-case record/);
    });
  });
});

describe("the summary", () => {
  it("sums up the totals, then one line per case and row count", () => {
    const result = gemvSweepResult(PLAN, {
      reports: [gemvPerCase(CASE_A), gemvPerCase(CASE_B)],
      stop: COMPLETED,
    });
    const line = (caseId: string): string =>
      `${caseId} M=1: baseline c32u4 0.020 ms · spread 1.004 · fastest identical c64u8 ×1.250 · fastest reorder parallel-l4 ×2.000 · exact c32u4 ×0.800 of the fastest identical · reorder parallel-l4 ×1.000 of the fastest reorder`;
    expect(result.summary).toEqual([
      "2 of 2 planned cases finished · 6 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the baseline · 0 outside the band",
      line(CASE_A),
      line(CASE_B),
    ]);
    expect(result.data?.format).toBe("karume-gemv-sweep/4");
  });

  it("shows no spread for a row count without one (its baseline failed)", () => {
    const rows = [gemvRow(CASE_A, "baseline", { error: "compile error" })];
    const report = gemvPerCase(CASE_A, { rows, summary: { baselineSpread: undefined } });
    const result = gemvSweepResult(
      { ...PLAN, caseIds: [CASE_A] },
      { reports: [report], stop: COMPLETED },
    );
    expect(result.summary[1]).toBe(
      `${CASE_A} M=1: baseline c32u4 failed: compile error · exact c32u4 failed · reorder parallel-l4 not measured`,
    );
  });

  it("leaves a candidate that failed its check out of the fastest, and says the profile's choice failed it", () => {
    const rows = [
      gemvRow(CASE_A, "baseline"),
      gemvRow(CASE_A, "identical", { identicalToBaseline: false, differingWords: 3 }),
      gemvRow(CASE_A, "reorder", { withinBand: false, bandUse: 1.5 }),
    ];
    const result = gemvSweepResult(
      { ...PLAN, caseIds: [CASE_A] },
      { reports: [gemvPerCase(CASE_A, { rows })], stop: COMPLETED },
    );
    expect(result.summary).toEqual([
      "1 of 1 planned cases finished · 3 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 1 not identical to the baseline · 1 outside the band",
      `${CASE_A} M=1: baseline c32u4 0.020 ms · spread 1.004 · exact c32u4 ×1.000 of the fastest identical · reorder parallel-l4 outside the band`,
    ]);
  });

  it("counts the kernels the quick look screened out, per case and in the totals, and says when it was off; a screened row is never wrong", () => {
    const {
      perDispatch: _time,
      identicalToBaseline: _same,
      ...identical
    } = gemvRow(CASE_A, "identical");
    const rows = [
      gemvRow(CASE_A, "baseline"),
      { ...identical, screened: true as const, quickPerDispatch: 40_000 },
      gemvRow(CASE_A, "reorder"),
    ];
    const outcome = { reports: [gemvPerCase(CASE_A, { rows })], stop: COMPLETED };
    const result = gemvSweepResult({ ...PLAN, caseIds: [CASE_A] }, outcome);
    expect(result.summary[0]).toContain("3 rows · 1 screened by the quick look (margin ×1.25)");
    expect(result.summary[1]).toContain("· 1 screened");
    expect(result.summary[1]).not.toContain("fastest identical c64u8");
    expect(result.verdict).toBe("pass");
    const off = gemvSweepResult({ ...PLAN, caseIds: [CASE_A], quickLookMargin: 0 }, outcome);
    expect(off.summary[0]).toContain("3 rows · quick look off ·");
  });

  it("counts the kernels skipped as not adoptable from M = 1, per case and in the totals; a skipped row is never wrong", () => {
    const {
      perDispatch: _time,
      reps: _reps,
      rounds: _rounds,
      speedupVsBaseline: _speed,
      maxAbsError: _error,
      bandUse: _use,
      withinBand: _within,
      nonFiniteOutputs: _nonFinite,
      ...reorder
    } = gemvRow(CASE_A, "reorder", { m: 4 });
    const rows = [
      gemvRow(CASE_A, "baseline", { m: 4 }),
      { ...reorder, candidate: "staged-l32", skipped: true as const },
      gemvRow(CASE_A, "reorder", { m: 4 }),
    ];
    const report = gemvPerCase(CASE_A, { rows, summary: { m: 4 } });
    const result = gemvSweepResult(
      { ...PLAN, caseIds: [CASE_A] },
      { reports: [report], stop: COMPLETED },
    );
    expect(result.summary[0]).toContain(
      "0 screened by the quick look (margin ×1.25) · 1 skipped (not adoptable from M = 1) · 0 with an error",
    );
    expect(result.summary[1]).toMatch(/· 1 skipped \(not adoptable from M = 1\)$/);
    expect(result.verdict).toBe("pass");
  });

  it("says why the run stopped, and rethrows the error of a run that measured nothing", () => {
    const failure = new Error("GpuUnavailableError");
    expect(() =>
      gemvSweepResult(PLAN, { reports: [], stop: { kind: "error", error: failure } }),
    ).toThrow(failure);
    const cancelled = gemvSweepResult(PLAN, {
      reports: [gemvPerCase(CASE_A)],
      stop: { kind: "cancelled" },
    });
    expect(cancelled.summary[0]).toBe(
      "1 of 2 planned cases finished · 3 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the baseline · 0 outside the band · stopped: cancelled",
    );
    expect(cancelled.data?.aborted).toBe(true);
  });
});

describe("an explore run's record and summary", () => {
  const SENTINELS = [CASE_C];
  /** An explore per-case record: its sentinels, and `narrowedTo` on the M = 1 summary of a case that is none. */
  const explore = (caseId: string, rows = gemvPerCase(caseId).rows): GemvSweepReport => {
    const report = gemvPerCase(caseId, { rows });
    return {
      ...report,
      settings: {
        ...report.settings,
        candidateSet: "explore",
        sentinels: SENTINELS,
        quickLookMargin: 1.1,
      },
      summaries: report.summaries.map((summary) =>
        SENTINELS.includes(caseId) ? summary : { ...summary, narrowedTo: ["c64u8", "c32u8"] },
      ),
    };
  };
  const PLANNED_EXPLORE: GemvSweepPlan = {
    ...PLAN,
    caseIds: [CASE_C, CASE_A],
    candidateSet: "explore",
    quickLookMargin: 1.1,
  };

  it("merges into the record one call over the cases writes: the sentinels in the settings, each group's narrowed names", () => {
    const merged = mergeGemvSweepReports([explore(CASE_C), explore(CASE_A)], {
      plannedCaseIds: [CASE_C, CASE_A],
      cancelledBetweenCases: false,
    });
    expect(merged.settings.sentinels).toEqual(SENTINELS);
    expect(merged.cases.map(({ id }) => id)).toEqual([CASE_C, CASE_A]);
    expect(merged.summaries.map(({ caseId, narrowedTo }) => [caseId, narrowedTo])).toEqual([
      [CASE_C, undefined],
      [CASE_A, ["c64u8", "c32u8"]],
    ]);
    // Records of other sentinels are not of one sweep
    const other = explore(CASE_A);
    expect(() =>
      mergeGemvSweepReports(
        [explore(CASE_C), { ...other, settings: { ...other.settings, sentinels: [CASE_A] } }],
        { plannedCaseIds: [CASE_C, CASE_A], cancelledBetweenCases: false },
      ),
    ).toThrow(/Record 1 differs from record 0/);
  });

  it("says how many candidates a group was narrowed to, and that the kernels it skipped were not carried from M = 1 (fault: the full set's words, not adoptable)", () => {
    const {
      perDispatch: _time,
      reps: _reps,
      rounds: _rounds,
      speedupVsBaseline: _speed,
      maxAbsError: _error,
      bandUse: _use,
      withinBand: _within,
      nonFiniteOutputs: _nonFinite,
      ...reorder
    } = gemvRow(CASE_A, "reorder", { m: 4 });
    const result = gemvSweepResult(PLANNED_EXPLORE, {
      reports: [
        explore(CASE_C),
        explore(CASE_A, [
          ...gemvPerCase(CASE_A).rows,
          gemvRow(CASE_A, "baseline", { m: 4 }),
          { ...reorder, candidate: "staged-l32", skipped: true as const },
        ]),
      ],
      stop: COMPLETED,
    });
    expect(result.summary[0]).toContain(
      "0 screened by the quick look (margin ×1.1) · 1 skipped (not carried from M = 1)",
    );
    expect(result.summary[1]).not.toContain("narrowed");
    expect(result.summary[2]).toContain("· narrowed to 2 predicted candidates");
    expect(result.verdict).toBe("pass");
  });
});

describe("gemvSweepVerdict", () => {
  const verdict = (outcome: GemvRunOutcome) => gemvSweepVerdict(outcome);
  const clean = gemvPerCase(CASE_A);
  const lost: PerCaseStop = { kind: "deviceLost", reason: "unknown", message: "crashed" };
  const differing = gemvPerCase(CASE_A, {
    rows: [
      gemvRow(CASE_A, "baseline"),
      gemvRow(CASE_A, "identical", { identicalToBaseline: false }),
    ],
  });
  const outside = gemvPerCase(CASE_A, {
    rows: [gemvRow(CASE_A, "baseline"), gemvRow(CASE_A, "reorder", { withinBand: false })],
  });
  const errored = gemvPerCase(CASE_A, {
    rows: [gemvRow(CASE_A, "baseline"), gemvRow(CASE_A, "reorder", { error: "compile error" })],
  });

  it("is fail when an identical row of a finished case differs from the baseline, even when the run then stopped", () => {
    expect(verdict({ reports: [differing], stop: COMPLETED })).toBe("fail");
    expect(verdict({ reports: [differing, gemvPerCase(CASE_B)], stop: lost })).toBe("fail");
  });

  it("is fail when a reorder row of a finished case is outside the band", () => {
    expect(verdict({ reports: [outside], stop: COMPLETED })).toBe("fail");
  });

  it("is incomplete when a device loss or an error stopped the run, errored rows notwithstanding", () => {
    expect(verdict({ reports: [errored], stop: lost })).toBe("incomplete");
    expect(verdict({ reports: [clean], stop: { kind: "error", error: new Error("x") } })).toBe(
      "incomplete",
    );
  });

  it("is fail when a row of a finished case errored", () => {
    expect(verdict({ reports: [errored], stop: COMPLETED })).toBe("fail");
  });

  it("is none when the run was cancelled, and does not count the case the cancel cut short", () => {
    const cutShort = gemvPerCase(CASE_B, {
      aborted: true,
      rows: [gemvRow(CASE_B, "baseline"), gemvRow(CASE_B, "reorder", { withinBand: false })],
    });
    expect(verdict({ reports: [clean, cutShort], stop: { kind: "cancelled" } })).toBeUndefined();
  });

  it("is pass when every planned case finished clean", () => {
    expect(verdict({ reports: [clean, gemvPerCase(CASE_B)], stop: COMPLETED })).toBe("pass");
  });

  it("does not hold the baseline to the band (its reading is shown, not judged)", () => {
    const baselineOutside = gemvPerCase(CASE_A, {
      rows: [gemvRow(CASE_A, "baseline", { withinBand: false }), gemvRow(CASE_A, "reorder")],
    });
    expect(verdict({ reports: [baselineOutside], stop: COMPLETED })).toBe("pass");
  });
});
