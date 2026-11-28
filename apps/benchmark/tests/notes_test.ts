import { type GemvSweepReport, gemvSweepCases } from "@karume/runtime/tune";
import { gemvSweepResult } from "@karume/sweep-app-support/gemv-sweep/summary";
import { geometrySweepResult } from "@karume/sweep-app-support/geometry-sweep/summary";
import { recordRun, type WorkbenchRecord } from "@karume/sweep-app-support/record";
import { describe, expect, it } from "vite-plus/test";

import { durationText } from "../src/estimate.ts";
import { afterRunNotes, beforeRunNotes, deviceView, gemvClassText } from "../src/notes.ts";
import { planBenchmark } from "../src/plan.ts";
import { hiddenLog } from "../src/visibility.ts";
import {
  CHECKOUT,
  CONV3D_A,
  ENVIRONMENT,
  GEMV_A,
  gemvReport,
  geometryReport,
} from "./helpers/fakes.ts";

const GPU = {
  vendor: "nvidia",
  architecture: "blackwell",
  description: "NVIDIA GeForce RTX 5070 Ti",
};

describe("deviceView", () => {
  it("shows the adapter and lets a run start when the adapter gives GPU timestamps", () => {
    const view = deviceView({ ...GPU, timestamps: true });
    expect(view.rows).toEqual([
      ["Vendor", "nvidia"],
      ["Architecture", "blackwell"],
      ["Description", "NVIDIA GeForce RTX 5070 Ti"],
      ["GPU timestamps (timestamp-query)", "available"],
    ]);
    expect(view.blocker).toBeUndefined();
    expect(view.notes).toEqual([]);
  });

  it("does not start without timestamp-query, saying the run would not be usable as evidence", () => {
    const view = deviceView({ ...GPU, timestamps: false });
    expect(view.blocker).toMatch(/could not be used as evidence, so it does not start/);
    expect(view.rows.at(-1)).toEqual(["GPU timestamps (timestamp-query)", "not available"]);
  });

  it("says what an empty description means, and shows it as not reported", () => {
    const view = deviceView({ ...GPU, description: "", timestamps: true });
    expect(view.rows[2]).toEqual(["Description", "(not reported by the browser)"]);
    expect(view.notes[0]).toMatch(/enable-webgpu-developer-features/);
  });
});

describe("beforeRunNotes", () => {
  it("says what runs, how long, what to avoid, why a second run matters and what the file holds", () => {
    const plan = planBenchmark({ gemv: true, geometry: true }, {});
    const notes = beforeRunNotes(plan, 5940);
    expect(notes).toHaveLength(5);
    expect(notes[0]).toMatch(
      new RegExp(
        `the GEMV sweep .*${gemvSweepCases().length} cases.* and the tile geometry sweep .*78 cases`,
      ),
    );
    expect(notes[1]).toContain(`How long: ${durationText(5940)}`);
    expect(notes[2]).toMatch(/keep this tab in front/);
    expect(notes[3]).toMatch(/second run/);
    expect(notes[4]).toMatch(/nothing else about you/);
  });

  it("asks for a choice when no sweep is chosen", () => {
    const notes = beforeRunNotes(planBenchmark({ gemv: false, geometry: false }, {}), 0);
    expect(notes[0]).toBe("Choose at least one of the two measurements below.");
  });
});

describe("hiddenLog", () => {
  it("counts the times the page was hidden and their total, the one still going included", () => {
    const log = hiddenLog(false, 0);
    log.note(true, 1000);
    log.note(true, 1500);
    log.note(false, 3000);
    log.note(true, 10_000);
    expect(log.summary(10_500)).toEqual({ hiddenCount: 2, hiddenSeconds: 2.5 });
  });

  it("counts a page hidden from the start", () => {
    const log = hiddenLog(true, 0);
    log.note(false, 400);
    expect(log.summary(1000)).toEqual({ hiddenCount: 1, hiddenSeconds: 0.4 });
  });
});

describe("afterRunNotes", () => {
  const plan = planBenchmark({ gemv: true, geometry: true }, { cases: [GEMV_A, CONV3D_A] });

  /** The run's records as the shared record code makes them, the GEMV case's timing `quantized`. */
  const records = async (quantized: boolean, sweeps = plan.sweeps): Promise<WorkbenchRecord[]> => {
    const signal = new AbortController().signal;
    const timed = (report: GemvSweepReport): GemvSweepReport => ({
      ...report,
      gpuTiming: { ...report.gpuTiming, quantized },
    });
    const made: WorkbenchRecord[] = [];
    for (const sweep of sweeps) {
      made.push(
        await recordRun(
          {
            entry: { id: sweep.entryId, parameters: sweep.parameters },
            environment: ENVIRONMENT,
            checkout: CHECKOUT,
            signal,
          },
          () =>
            Promise.resolve(
              sweep.sweep === "gemv"
                ? gemvSweepResult(sweep.plan, {
                    reports: sweep.plan.caseIds.map((caseId) => timed(gemvReport(caseId))),
                    stop: { kind: "completed" },
                  })
                : geometrySweepResult(sweep.plan, {
                    reports: sweep.plan.caseIds.map((caseId) => geometryReport(caseId)),
                    stop: { kind: "completed" },
                  }),
            ),
        ),
      );
    }
    return made;
  };

  it("says the GEMV part will not count when its record reports quantized GPU timing, why, and the flag", async () => {
    const notes = afterRunNotes(await records(true));
    expect(notes).toHaveLength(1);
    expect(notes[0]).toMatch(/^The GEMV part of this result will not be accepted as evidence/);
    expect(notes[0]).toMatch(/rounded the GPU timestamps to 100 µs/);
    expect(notes[0]).toContain("chrome://flags/#enable-webgpu-developer-features");
    expect(notes[0]).toMatch(/The tile geometry part is still used\.$/);
  });

  it("names no geometry part when no geometry sweep ran", async () => {
    const gemvOnly = plan.sweeps.filter(({ sweep }) => sweep === "gemv");
    const notes = afterRunNotes(await records(true, gemvOnly));
    expect(notes).toHaveLength(1);
    expect(notes[0]).not.toMatch(/tile geometry/);
  });

  /** A record's data that is a GEMV sweep record (the merged one of the run). */
  const isGemvData = (data: unknown): data is GemvSweepReport =>
    typeof data === "object" &&
    data !== null &&
    "format" in data &&
    data.format === "karume-gemv-sweep/4";

  it("says nothing when the GEMV timing was not quantized and the run was steady, or the GEMV sweep measured nothing", async () => {
    const steady = (await records(false)).map((record) =>
      isGemvData(record.data)
        ? {
            ...record,
            data: {
              ...record.data,
              steadiness: { class: "steady" as const, groups: 40, unsteadyGroups: 1, share: 0.025 },
            },
          }
        : record,
    );
    expect(afterRunNotes(steady)).toEqual([]);
    const [gemv] = await records(true);
    if (gemv === undefined) throw new Error("no GEMV record");
    expect(afterRunNotes([{ ...gemv, data: undefined }])).toEqual([]);
  });

  it("asks for one more run after an unsteady GEMV run, and says when a short one has too few groups to tell (faults: unsteady, two cases)", async () => {
    const made = await records(false);
    // The trial's two cases: too few groups for a class
    const [few] = afterRunNotes(made);
    expect(few).toContain("too few groups");
    const unsteady = made.map((record) =>
      isGemvData(record.data)
        ? {
            ...record,
            data: {
              ...record.data,
              steadiness: {
                class: "unsteady" as const,
                groups: 40,
                unsteadyGroups: 20,
                share: 0.5,
              },
            },
          }
        : record,
    );
    const [again] = afterRunNotes(unsteady);
    expect(again).toContain("please run the page once more at another time");
  });

  /** The run's records with the GEMV record's data steady and timed as `timing`. */
  const steadyTimed = async (timing: { unit: string; quantized: boolean }) =>
    (await records(false)).map((record) =>
      isGemvData(record.data)
        ? {
            ...record,
            data: {
              ...record.data,
              gpuTiming: { ...record.data.gpuTiming, ...timing },
              steadiness: { class: "steady" as const, groups: 40, unsteadyGroups: 1, share: 0.025 },
            },
          }
        : record,
    );

  it("says a steady GEMV record is enough only when the derivation reads it so: timed in ns, not rounded (faults: rounded, the wall clock, raw ticks)", async () => {
    const gemvOf = (made: readonly WorkbenchRecord[]) => {
      const found = made.find((record) => isGemvData(record.data));
      if (found === undefined) throw new Error("no GEMV record");
      return found;
    };
    const steady = await steadyTimed({ unit: "ns", quantized: false });
    expect(gemvClassText(gemvOf(steady))).toBe("steady: one record is enough for a table");
    const rounded = await steadyTimed({ unit: "ns", quantized: true });
    expect(gemvClassText(gemvOf(rounded))).toBe(
      "its GPU timestamps were rounded: no GEMV evidence, whatever its class",
    );
    expect(afterRunNotes(rounded)[0]).toMatch(/rounded the GPU timestamps to 100 µs/);
    const wall = await steadyTimed({ unit: "wall", quantized: false });
    expect(gemvClassText(gemvOf(wall))).toBe(
      "timed without GPU timestamps: no GEMV evidence, whatever its class",
    );
    expect(afterRunNotes(wall)[0]).toMatch(/^The GEMV part of this result will not be accepted/);
    const ticks = await steadyTimed({ unit: "deno-raw-tick", quantized: false });
    expect(gemvClassText(gemvOf(ticks))).toContain("a table needs a second record");
  });
});
