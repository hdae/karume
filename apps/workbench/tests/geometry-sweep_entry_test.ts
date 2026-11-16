import {
  deriveGeometryProfile,
  type GeometrySweepOptions,
  type GeometrySweepReport,
  geometrySweepCases,
} from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import geometrySweep from "../src/entries/geometry-sweep.ts";
import {
  foldSweepEvent,
  INITIAL_LIVE,
  runSweepPlan,
  type SweepLive,
  type SweepRunEvent,
  type SweepRunOutcome,
} from "../src/geometry-sweep/run.ts";
import type { SweepPlan } from "../src/geometry-sweep/settings.ts";
import { geometrySweepResult, progressLine } from "../src/geometry-sweep/summary.ts";
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

describe("the geometry-sweep entry", () => {
  it("declares the candidate set, the op families, the cases, the rounds and the idle seconds", () => {
    expect(geometrySweep.id).toBe("geometry-sweep");
    expect(geometrySweep.parameters.map(({ name, kind }) => `${name}:${kind}`)).toEqual([
      "candidateSet:choice",
      "ops:choices",
      "cases:choices",
      "rounds:integer",
      "idleSeconds:integer",
    ]);
  });
});

describe("geometrySweepResult", () => {
  describe("when every planned case ran", () => {
    it("keeps the merged record, which deriveGeometryProfile accepts as one input", async () => {
      const result = geometrySweepResult(PLAN, await runWith(new AbortController().signal));
      const record = recordOf(result.data);
      expect(record.format).toBe("karume-geometry-sweep/3");
      expect(record.settings.cases).toEqual([CASE_A, CASE_B]);
      const { profile } = deriveGeometryProfile(
        [{ report: record, path: "record.json", sha256: "sha" }],
        { id: "nvidia-blackwell", vendor: "nvidia", architecture: "blackwell" },
      );
      expect(profile.i8a8.conv3d).toEqual(CANDIDATE_GEOMETRY);
    });

    it("sums up the totals, then one line per case", async () => {
      const result = geometrySweepResult(PLAN, await runWith(new AbortController().signal));
      expect(result.summary).toEqual([
        "2 of 2 planned cases finished · 4 rows · 0 with an error · 0 not identical to the default",
        CASE_LINE(CASE_A),
        CASE_LINE(CASE_B),
      ]);
      expect(result.verdict).toBeUndefined();
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
        "2 of 2 planned cases finished · 5 rows · 1 with an error · 1 not identical to the default",
      );
    });
  });

  describe("when Cancel stopped the run after the first case", () => {
    it("keeps that case in an aborted record and says why the run stopped", async () => {
      const controller = new AbortController();
      const outcome = await runWith(controller.signal, (call) => {
        if (call === 1) controller.abort(new Error("Cancelled by the user"));
        return {};
      });
      const result = geometrySweepResult(PLAN, outcome);
      const record = recordOf(result.data);
      expect(record.aborted).toBe(true);
      expect(record.cases.map(({ caseId }) => caseId)).toEqual([CASE_A]);
      expect(record.settings.cases).toEqual([CASE_A, CASE_B]);
      expect(result.summary).toEqual([
        "1 of 2 planned cases finished · 2 rows · 0 with an error · 0 not identical to the default · stopped: cancelled",
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
      expect(recordOf(result.data).aborted).toBe(true);
      expect(result.summary[0]).toBe(
        "0 of 2 planned cases finished · 1 cut short · 2 rows · 0 with an error · 0 not identical to the default · stopped: cancelled",
      );
    });
  });

  describe("when the device was lost in the first case", () => {
    it("keeps that case's rows as a case cut short, names the loss and reads as incomplete", async () => {
      const outcome = await runWith(new AbortController().signal, () => ({
        deviceLost: { reason: "unknown", message: "GPU process crashed" },
      }));
      const result = geometrySweepResult(PLAN, outcome);
      expect(recordOf(result.data).cases).toHaveLength(1);
      expect(result.summary[0]).toBe(
        "0 of 2 planned cases finished · 1 cut short · 2 rows · 0 with an error · 0 not identical to the default · stopped: device lost (unknown): GPU process crashed",
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
        "0 of 2 planned cases finished · 0 rows · 0 with an error · 0 not identical to the default · stopped: cancelled",
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
        "1 of 1 planned cases finished · 2 rows · 0 with an error · 0 not identical to the default",
      );
      expect(summary[1]).toBe(
        "K-88 against this adapter's table geometry (default); a ratio is the table's time ÷ the variant's (above ×1 the variant is faster)",
      );
      expect(summary.at(-1)).toBe(
        `${k88Case.id} (×${k88Case.censusCount}): table r8x8w16x16 10.000 ms · best form ÷ table ×1.250 · identical yes · 0 mismatches · 0 errors`,
      );
      expect(summary.some((line) => line.startsWith(`${k88Case.id}: `))).toBe(false);
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
      { kind: "sweep", progress: { kind: "case", summary: { caseId: "a", elapsedMs: 1 } } },
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
