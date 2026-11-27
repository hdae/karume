import {
  type GeometrySweepOptions,
  type GeometrySweepReport,
  geometrySweepCases,
  sweepCandidateKernelsId,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";
import {
  foldSweepEvent,
  INITIAL_LIVE,
  runSweepPlan,
  type SweepLive,
  type SweepRunEvent,
  type SweepRunOutcome,
} from "@karume/sweep-app-support/geometry-sweep/run";
import type { SweepPlan } from "@karume/sweep-app-support/geometry-sweep/settings";
import {
  geometrySweepResult,
  progressLine,
} from "@karume/sweep-app-support/geometry-sweep/summary";

import {
  CANDIDATE_GEOMETRY,
  CASE_A,
  CASE_B,
  type PerCaseOptions,
  perCase,
  row,
} from "./helpers/geometry-sweep-report.ts";

const PLAN: SweepPlan = {
  caseIds: [CASE_A, CASE_B],
  ops: ["i8a8-conv3d"],
  candidateSet: "full",
  geometries: [],
  quickLookMargin: 1.25,
  rounds: 5,
  idleSeconds: 0,
};

/** A run of the plan with a fake sweep: each call returns its case's record, `extra` by call. */
const runWith = (
  signal: AbortSignal,
  extra: (call: number, options: GeometrySweepOptions) => PerCaseOptions = () => ({}),
): Promise<SweepRunOutcome> => {
  let call = 0;
  return runSweepPlan(PLAN, {
    signal,
    onEvent: () => undefined,
    sweep: async (options) => {
      call += 1;
      return perCase(options.cases?.[0] ?? "", extra(call, options));
    },
  });
};

/** The result's record, which a run that finished a case always has. */
const recordOf = (data: GeometrySweepReport | undefined): GeometrySweepReport => {
  if (data === undefined) throw new Error("the result has no record");
  return data;
};

const CASE_LINE = (caseId: string): string =>
  `${caseId}: 2 rows · default 20.000 ms · drift 1.010 · fastest identical tile64x64r8x4w16x8k16 ×1.250 · 0 errors · 0 mismatches · 120 s`;

describe("geometrySweepResult", () => {
  describe("when every planned case ran", () => {
    it("keeps the record merged from every case as its data", async () => {
      const result = geometrySweepResult(PLAN, await runWith(new AbortController().signal));
      const record = recordOf(result.data);
      expect(record.format).toBe("karume-geometry-sweep/4");
      expect(record.settings.cases).toEqual([CASE_A, CASE_B]);
    });

    it("sums up the totals, then one line per case", async () => {
      const result = geometrySweepResult(PLAN, await runWith(new AbortController().signal));
      expect(result.summary).toEqual([
        "2 of 2 planned cases finished · 4 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default",
        CASE_LINE(CASE_A),
        CASE_LINE(CASE_B),
      ]);
      expect(result.verdict).toBe("pass");
    });

    it("counts the rows with an error and the rows not identical to the default", async () => {
      const outcome = await runWith(new AbortController().signal, (call, options) =>
        call === 1
          ? {
              rows: [
                row(options.cases?.[0] ?? "", true),
                row(options.cases?.[0] ?? "", false, { identicalToDefault: false }),
                { ...row(options.cases?.[0] ?? "", false), error: "compile failed" },
              ],
              candidateKernels: "unused",
            }
          : {},
      );
      const [totals] = geometrySweepResult(PLAN, outcome).summary;
      expect(totals).toBe(
        "2 of 2 planned cases finished · 5 rows · 0 screened by the quick look (margin ×1.25) · 1 with an error · 1 not identical to the default",
      );
    });
  });

  describe("when Cancel stopped the run after the first case", () => {
    it("marks its record aborted and says why the run stopped", async () => {
      const controller = new AbortController();
      const outcome = await runWith(controller.signal, (call) => {
        if (call === 1) controller.abort(new Error("Cancelled by the user"));
        return {};
      });
      const result = geometrySweepResult(PLAN, outcome);
      // The cancel between cases is what marks the record aborted (the case itself finished).
      expect(recordOf(result.data).aborted).toBe(true);
      expect(result.summary).toEqual([
        "1 of 2 planned cases finished · 2 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default · stopped: cancelled",
        CASE_LINE(CASE_A),
      ]);
    });
  });

  describe("when Cancel stopped the first case midway", () => {
    it("counts that case as cut short, not as finished", async () => {
      const controller = new AbortController();
      const outcome = await runWith(controller.signal, () => {
        controller.abort(new Error("Cancelled by the user"));
        return { aborted: true };
      });
      const result = geometrySweepResult(PLAN, outcome);
      expect(result.summary[0]).toBe(
        "0 of 2 planned cases finished · 1 cut short · 2 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default · stopped: cancelled",
      );
    });
  });

  describe("when the device was lost in the first case", () => {
    it("counts that case as cut short, names the loss and reads as incomplete", async () => {
      const outcome = await runWith(new AbortController().signal, () => ({
        deviceLost: { reason: "unknown", message: "GPU process crashed" },
      }));
      const result = geometrySweepResult(PLAN, outcome);
      expect(result.summary[0]).toBe(
        "0 of 2 planned cases finished · 1 cut short · 2 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default · stopped: device lost (unknown): GPU process crashed",
      );
      expect(result.verdict).toBe("incomplete");
    });
  });

  describe("when a sweep threw", () => {
    const failure = new Error("GpuUnavailableError: no adapter");

    it("keeps the cases before, names the error and reads as incomplete", () => {
      const outcome: SweepRunOutcome = {
        reports: [perCase(CASE_A)],
        stop: { kind: "error", error: failure },
      };
      const result = geometrySweepResult(PLAN, outcome);
      expect(recordOf(result.data).cases).toHaveLength(1);
      expect(result.summary[0]).toMatch(/· stopped: GpuUnavailableError: no adapter$/);
      expect(result.verdict).toBe("incomplete");
    });

    it("throws that error when no case finished", () => {
      expect(() =>
        geometrySweepResult(PLAN, { reports: [], stop: { kind: "error", error: failure } }),
      ).toThrow(failure);
    });
  });

  describe("when Cancel came before any case finished", () => {
    it("has no record, and says so in the totals", () => {
      const result = geometrySweepResult(PLAN, { reports: [], stop: { kind: "cancelled" } });
      expect(result.data).toBeUndefined();
      expect(result.summary).toEqual([
        "0 of 2 planned cases finished · 0 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default · stopped: cancelled",
      ]);
      expect(result.verdict).toBeUndefined();
    });
  });

  describe("for the evaluation set k88", () => {
    it("gives the K-88 summary instead of the per-case lines", () => {
      const [k88Case] = geometrySweepCases("k88");
      if (k88Case === undefined) throw new Error("the k88 table is empty");
      const table = { regM: 8, regN: 8, wgX: 16, wgY: 16 };
      const f32Row = (form: "tb-un" | undefined, perDispatch: number) => ({
        ...row(k88Case.id, form === undefined),
        op: "linear" as const,
        shape: "M768 N6144 K1536",
        geometry: form === undefined ? "r8x8w16x16" : `r8x8w16x16:${form}`,
        geometryParams: form === undefined ? table : { ...table, form },
        perDispatch,
      });
      const report = perCase(k88Case.id, {
        rows: [f32Row(undefined, 10e6), f32Row("tb-un", 8e6)],
        candidateKernels: "unused",
      });
      const k88Report: GeometrySweepReport = {
        ...report,
        settings: { ...report.settings, candidateSet: "k88" },
      };
      const plan: SweepPlan = { ...PLAN, caseIds: [k88Case.id], candidateSet: "k88" };
      const { summary } = geometrySweepResult(plan, {
        reports: [k88Report],
        stop: { kind: "completed" },
      });
      expect(summary[0]).toBe(
        "1 of 1 planned cases finished · 2 rows · 0 screened by the quick look (margin ×1.25) · 0 with an error · 0 not identical to the default",
      );
      expect(summary[1]).toBe(
        "K-88 against the tile of this adapter's table geometry (default, without the fields); a ratio is the tile's time ÷ the variant's (above ×1 the variant is faster)",
      );
      expect(summary.at(-1)).toBe(
        `${k88Case.id} (×${k88Case.censusCount}): table r8x8w16x16 10.000 ms · best form ÷ table ×1.250 · identical yes · 0 mismatches · 0 errors`,
      );
      expect(summary.some((line) => line.startsWith(`${k88Case.id}: `))).toBe(false);
    });
  });
});

describe("the verdict of geometrySweepResult", () => {
  const LOST = { reason: "unknown", message: "GPU process crashed" } as const;
  const mismatch = (caseId: string): PerCaseOptions => ({
    rows: [row(caseId, true), row(caseId, false, { identicalToDefault: false })],
    candidateKernels: "unused",
  });
  const errored = (caseId: string): PerCaseOptions => ({
    rows: [row(caseId, true), { ...row(caseId, false), error: "compile failed" }],
    candidateKernels: "unused",
  });
  const caseOf = (options: GeometrySweepOptions): string => options.cases?.[0] ?? "";
  const verdictOf = async (
    extra: (call: number, options: GeometrySweepOptions) => PerCaseOptions,
    controller = new AbortController(),
  ) => geometrySweepResult(PLAN, await runWith(controller.signal, extra)).verdict;

  describe("when a row of a finished case is not identical to its default", () => {
    it("is fail", async () => {
      expect(
        await verdictOf((call, options) => (call === 1 ? mismatch(caseOf(options)) : {})),
      ).toBe("fail");
    });

    it("is fail when a device loss then stopped the run short (a wrong result whatever else happened)", async () => {
      expect(
        await verdictOf((call, options) =>
          call === 1 ? mismatch(caseOf(options)) : { deviceLost: LOST },
        ),
      ).toBe("fail");
    });

    it("is fail when the run was then cancelled", async () => {
      const controller = new AbortController();
      expect(
        await verdictOf((call, options) => {
          if (call === 1) controller.abort(new Error("Cancelled by the user"));
          return mismatch(caseOf(options));
        }, controller),
      ).toBe("fail");
    });
  });

  describe("when a device loss stopped the run short of its plan", () => {
    it("is incomplete although a finished case before it had a row that errored (the loss comes first)", async () => {
      expect(
        await verdictOf((call, options) =>
          call === 1 ? errored(caseOf(options)) : { ...errored(caseOf(options)), deviceLost: LOST },
        ),
      ).toBe("incomplete");
    });

    it("is incomplete when only the case it cut short has a mismatch (that case did not finish)", async () => {
      expect(
        await verdictOf((call, options) =>
          call === 1 ? {} : { ...mismatch(caseOf(options)), deviceLost: LOST },
        ),
      ).toBe("incomplete");
    });
  });

  describe("when an error stopped the run short of its plan", () => {
    it("is incomplete although a finished case before it had a row that errored", () => {
      const outcome: SweepRunOutcome = {
        reports: [perCase(CASE_A, errored(CASE_A))],
        stop: { kind: "error", error: new Error("GpuUnavailableError: no adapter") },
      };
      expect(geometrySweepResult(PLAN, outcome).verdict).toBe("incomplete");
    });
  });

  describe("when every planned case finished and a row errored", () => {
    it("is fail", async () => {
      expect(await verdictOf((call, options) => (call === 2 ? errored(caseOf(options)) : {}))).toBe(
        "fail",
      );
    });
  });

  describe("when the run was cancelled", () => {
    it("has none when only the case it cut short has a row that errored or differed", async () => {
      const controller = new AbortController();
      expect(
        await verdictOf((call, options) => {
          if (call === 2) controller.abort(new Error("Cancelled by the user"));
          return call === 2
            ? {
                rows: [
                  row(caseOf(options), true),
                  { ...row(caseOf(options), false), error: "compile failed" },
                  row(caseOf(options), false, { identicalToDefault: false }),
                ],
                candidateKernels: "unused",
                aborted: true,
              }
            : {};
        }, controller),
      ).toBeUndefined();
    });

    it("is fail when a finished case before the cancel had a row that errored (the errored row comes first)", async () => {
      const controller = new AbortController();
      expect(
        await verdictOf((call, options) => {
          if (call === 1) controller.abort(new Error("Cancelled by the user"));
          return errored(caseOf(options));
        }, controller),
      ).toBe("fail");
    });
  });

  describe("when every planned case finished with every row identical and none errored", () => {
    it("is pass, however the candidates timed against the default (timing is not part of the verdict)", async () => {
      expect(
        await verdictOf((_call, options) => ({
          rows: [
            row(caseOf(options), true),
            row(caseOf(options), false, { perDispatch: 40_000_000, speedupVsDefault: 0.5 }),
          ],
          candidateKernels: "unused",
        })),
      ).toBe("pass");
    });
  });
});

describe("progressLine", () => {
  const plan: SweepPlan = { ...PLAN, caseIds: ["a", "b", "c"], idleSeconds: 90 };
  const after = (events: readonly SweepRunEvent[]): SweepLive =>
    events.reduce(foldSweepEvent, INITIAL_LIVE);

  it("says the run is starting before the first case", () => {
    expect(progressLine(plan, INITIAL_LIVE, 0)).toBe("starting 3 cases");
  });

  it("names the cases done, the case, its rows, the sweep's status and the timing once the case started", () => {
    const acquiring = after([{ kind: "case", index: 1, caseId: "b" }]);
    expect(progressLine(plan, acquiring, 0)).toBe(
      "0 / 3 cases done · case 2 / 3 b · 0 rows · acquiring the device",
    );
    // Case a finished (its summary is the sweep's last event of it), then case b started.
    const measuring = after([
      {
        kind: "sweep",
        progress: { kind: "case", summary: { caseId: "a", elapsedMs: 1, screened: 0 } },
      },
      { kind: "case", index: 1, caseId: "b" },
      {
        kind: "sweep",
        progress: {
          kind: "started",
          adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
          unit: "ns",
          dp4a: true,
          caseCount: 1,
        },
      },
      { kind: "sweep", progress: { kind: "row", row: row("b", true) } },
      { kind: "sweep", progress: { kind: "status", message: "b: g (2/48)" } },
    ]);
    expect(progressLine(plan, measuring, 0)).toBe(
      "1 / 3 cases done · case 2 / 3 b · 1 rows · b: g (2/48) · timing ns, dp4a true",
    );
  });

  it("counts the idle wait down", () => {
    const idle = after([{ kind: "idle", index: 1, until: 100_000 }]);
    expect(progressLine(plan, idle, 55_000)).toBe(
      "0 / 3 cases done · idle before case 2 / 3: 45 s left",
    );
    expect(progressLine(plan, idle, 120_000)).toBe(
      "0 / 3 cases done · idle before case 2 / 3: 0 s left",
    );
  });

  it("names a device loss", () => {
    const lost = after([
      { kind: "case", index: 0, caseId: "a" },
      { kind: "sweep", progress: { kind: "deviceLost", reason: "unknown", message: "crashed" } },
    ]);
    expect(progressLine(plan, lost, 0)).toBe(
      "device lost (unknown): crashed — the sweep stops after case 1 / 3",
    );
  });
});

describe("geometrySweepResult for the explore set", () => {
  const EXPLORE: SweepPlan = { ...PLAN, candidateSet: "explore" };
  /** The run of the explore plan: the second case skips the candidate the first ruled out. */
  const exploreRun = (): Promise<SweepRunOutcome> =>
    runSweepPlan(EXPLORE, {
      signal: new AbortController().signal,
      onEvent: () => undefined,
      sweep: async (options) => {
        const caseId = options.cases?.[0] ?? "";
        const later = (options.exploreEvidence ?? []).length > 0;
        const report = perCase(caseId, {
          rows: later
            ? [
                row(caseId, true),
                {
                  caseId,
                  op: "i8a8-conv3d",
                  shape: caseId,
                  censusCount: 1,
                  geometry: "tile64x64r8x4w16x8k16",
                  geometryParams: CANDIDATE_GEOMETRY,
                  isDefault: false,
                  skipped: true,
                },
              ]
            : [row(caseId, true), row(caseId, false, { speedupVsDefault: 0.9 })],
        });
        return { ...report, settings: { ...report.settings, candidateSet: "explore" } };
      },
    });

  it("says in the totals how many rows were skipped as dropped for their class (fault: none said)", async () => {
    const [totals] = geometrySweepResult(EXPLORE, await exploreRun()).summary;
    expect(totals).toBe(
      "2 of 2 planned cases finished · 4 rows · 0 screened by the quick look (margin ×1.25) · 1 skipped as dropped for their class · 0 with an error · 0 not identical to the default",
    );
  });

  it("merges the records into what one call over the cases holds: the cases in measuring order, the fingerprint over every row", async () => {
    const record = recordOf(geometrySweepResult(EXPLORE, await exploreRun()).data);
    expect(record.settings.candidateSet).toBe("explore");
    expect(record.settings.cases).toEqual(EXPLORE.caseIds);
    expect(record.rows.filter((one) => one.skipped === true).map(({ caseId }) => caseId)).toEqual([
      CASE_B,
    ]);
    expect(record.candidateKernels).toBe(sweepCandidateKernelsId(record.rows, record.dp4a));
  });
});
