import { deriveGeometryProfile, type GeometrySweepReport } from "@karume/runtime/tune";
import { describe, expect, it, vi } from "vite-plus/test";

import { downloadJson } from "../src/lib/download.ts";
import {
  caseSummaryRows,
  deriveTarget,
  mergeSweepReports,
  sweepFileName,
  sweepRecordText,
} from "../src/models/geometry-sweep/record.ts";
import {
  ADAPTER,
  CANDIDATE_GEOMETRY,
  CASE_A,
  CASE_B,
  CASE_C,
  perCase,
  row,
} from "./helpers/geometry-sweep-report.ts";

const PLAN = { plannedCaseIds: [CASE_A, CASE_B], cancelledBetweenCases: false };

/**
 * What the derivation reads from the saved file. `defaultKernels` is left out: it is the runtime's
 * fingerprint of its default table, which the public surface does not give (records without the
 * field are not matched on it), while `caseSet` and `candidateKernels` stay and are matched.
 */
const derive = (report: GeometrySweepReport) => {
  const { defaultKernels: _unmatched, ...saved } = report;
  return deriveGeometryProfile([{ report: saved, path: "merged.json", sha256: "sha" }], {
    id: "nvidia-blackwell",
    vendor: "nvidia",
    architecture: "blackwell",
  });
};

describe("mergeSweepReports", () => {
  describe("when every planned case has its record", () => {
    const first = perCase(CASE_A, { date: "2026-10-28T10:05:00.000Z" });
    const second = perCase(CASE_B, { date: "2026-10-28T10:12:00.000Z" });
    const merged = mergeSweepReports([first, second], PLAN);

    it("holds the rows and cases of both in run order, the first start and the last end", () => {
      expect(merged.rows).toEqual([...first.rows, ...second.rows]);
      expect(merged.cases.map(({ caseId }) => caseId)).toEqual([CASE_A, CASE_B]);
      expect(merged.startedAt).toBe(first.startedAt);
      expect(merged.date).toBe("2026-10-28T10:12:00.000Z");
      expect(merged.settings.cases).toEqual([CASE_A, CASE_B]);
      expect(merged.aborted).toBe(false);
    });

    it("carries a candidateKernels the derivation accepts, which neither per-case value is", () => {
      expect(merged.candidateKernels).not.toBe(first.candidateKernels);
      expect(merged.candidateKernels).not.toBe(second.candidateKernels);
      const { profile } = derive(merged);
      expect(profile.i8a8.conv3d).toEqual(CANDIDATE_GEOMETRY);
    });

    it("would be rejected with the first record's candidateKernels (a merge that does not recompute it)", () => {
      expect(() => derive({ ...merged, candidateKernels: first.candidateKernels })).toThrow(
        /candidateKernels/,
      );
    });

    it("stays in the sweep record's format (no envelope)", () => {
      expect(merged.format).toBe("karume-geometry-sweep/2");
      expect(Object.keys(merged).sort()).toEqual(Object.keys(first).sort());
    });
  });

  describe("when Cancel stopped the sweep between cases", () => {
    it("marks the record aborted and keeps every planned case in settings.cases", () => {
      const merged = mergeSweepReports([perCase(CASE_A)], {
        plannedCaseIds: [CASE_A, CASE_B],
        cancelledBetweenCases: true,
      });
      expect(merged.aborted).toBe(true);
      expect(merged.settings.cases).toEqual([CASE_A, CASE_B]);
      expect(merged.cases).toHaveLength(1);
    });
  });

  describe("when Cancel stopped a case midway", () => {
    it("marks the record aborted", () => {
      const merged = mergeSweepReports([perCase(CASE_A), perCase(CASE_B, { aborted: true })], PLAN);
      expect(merged.aborted).toBe(true);
    });
  });

  describe("when the device was lost in a case", () => {
    it("keeps the loss and is not marked aborted (aborted means a cancel)", () => {
      const lost = { reason: "unknown", message: "GPU process crashed" };
      const merged = mergeSweepReports([perCase(CASE_A, { deviceLost: lost })], PLAN);
      expect(merged.deviceLost).toEqual(lost);
      expect(merged.aborted).toBe(false);
    });
  });

  describe("when some cases' rounds looked quantized", () => {
    it("marks the merged record quantized (the larger rounding bound)", () => {
      const merged = mergeSweepReports(
        [perCase(CASE_A), perCase(CASE_B, { quantized: true })],
        PLAN,
      );
      expect(merged.gpuTiming.quantized).toBe(true);
      expect(mergeSweepReports([perCase(CASE_A), perCase(CASE_B)], PLAN).gpuTiming.quantized).toBe(
        false,
      );
    });
  });

  describe("when the records do not belong to one sweep", () => {
    it("throws for another adapter", () => {
      const other = perCase(CASE_B, { adapter: { ...ADAPTER, device: "0x2c02" } });
      expect(() => mergeSweepReports([perCase(CASE_A), other], PLAN)).toThrow(/Record 1 differs/);
    });

    it("throws for other settings", () => {
      const other = perCase(CASE_B);
      const changed = { ...other, settings: { ...other.settings, rounds: 3 } };
      expect(() => mergeSweepReports([perCase(CASE_A), changed], PLAN)).toThrow(/Record 1 differs/);
    });

    it("throws for an unplanned case, a case twice, and a record of more than one case", () => {
      expect(() => mergeSweepReports([perCase(CASE_C)], PLAN)).toThrow(/not a planned case/);
      expect(() => mergeSweepReports([perCase(CASE_A), perCase(CASE_A)], PLAN)).toThrow(
        /has two records/,
      );
      const both = perCase(CASE_A);
      expect(() =>
        mergeSweepReports(
          [{ ...both, settings: { ...both.settings, cases: [CASE_A, CASE_B] } }],
          PLAN,
        ),
      ).toThrow(/not a per-case record/);
    });

    it("throws for no record at all", () => {
      expect(() => mergeSweepReports([], PLAN)).toThrow(/No per-case record/);
    });
  });
});

describe("sweepRecordText", () => {
  it("is the record's JSON, which reads back to the same record", () => {
    const report = perCase(CASE_A);
    const text = sweepRecordText(report);
    expect(text.endsWith("\n")).toBe(true);
    expect(JSON.parse(text)).toEqual(report);
  });

  it("is the very text the page's Download record button saves (downloadJson), so the sha256 names the file", async () => {
    const report = perCase(CASE_A);
    const blobs: Blob[] = [];
    vi.stubGlobal("document", { createElement: () => ({ click: () => undefined }) });
    vi.spyOn(URL, "createObjectURL").mockImplementation((object) => {
      if (object instanceof Blob) blobs.push(object);
      return "blob:test";
    });
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    try {
      downloadJson(report, "record.json");
      // downloadBlob revokes the URL on the next task.
      await new Promise((resolve) => setTimeout(resolve, 0));
    } finally {
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
    }
    expect(blobs).toHaveLength(1);
    expect(await blobs[0]?.text()).toBe(sweepRecordText(report));
  });
});

describe("sweepFileName", () => {
  it("names the adapter's description, the candidate set and the record's UTC end", () => {
    expect(sweepFileName(perCase(CASE_A, { date: "2026-10-28T10:05:09.123Z" }))).toBe(
      "geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T10-05-09.json",
    );
  });

  it("falls back to vendor, architecture and device when the description is empty", () => {
    const report = perCase(CASE_A, { adapter: { ...ADAPTER, description: "" } });
    expect(sweepFileName(report)).toMatch(/^geometry-sweep_nvidia-blackwell-0x2c05_full_/);
  });

  it("spells quick+ apart from quick", () => {
    const report = perCase(CASE_A);
    const quickPlus = {
      ...report,
      settings: { ...report.settings, candidateSet: "quick+" as const },
    };
    expect(sweepFileName(quickPlus)).toMatch(/_quick-plus_/);
  });

  it("names an adapter that reports nothing as unknown", () => {
    const report = perCase(CASE_A, {
      adapter: { vendor: "", architecture: "", device: "", description: "" },
    });
    expect(sweepFileName(report)).toMatch(/^geometry-sweep_unknown-adapter_/);
  });
});

describe("deriveTarget", () => {
  it("names the table <vendor>-<architecture> and matches both, like the built-in tables", () => {
    expect(deriveTarget(ADAPTER)).toEqual({
      id: "nvidia-blackwell",
      vendor: "nvidia",
      architecture: "blackwell",
    });
  });

  it("matches the vendor alone when the architecture is empty", () => {
    expect(deriveTarget({ ...ADAPTER, architecture: "" })).toEqual({
      id: "nvidia",
      vendor: "nvidia",
    });
  });

  it("makes an injection-only table when the vendor is empty", () => {
    expect(deriveTarget({ ...ADAPTER, vendor: "", architecture: "" })).toEqual({
      id: "in-app",
      optIn: true,
    });
  });
});

describe("caseSummaryRows", () => {
  it("summarizes each case: its default, drift, fastest identical candidate, errors and mismatches", () => {
    const rows = [
      row(CASE_A, true),
      row(CASE_A, false, { geometry: "slow", speedupVsDefault: 1.1 }),
      row(CASE_A, false, {
        geometry: "fast-but-different",
        speedupVsDefault: 2,
        identicalToDefault: false,
      }),
      row(CASE_A, false, { geometry: "fast", speedupVsDefault: 1.3 }),
      { ...row(CASE_A, false), geometry: "broken", error: "compile failed" },
    ];
    expect(
      caseSummaryRows(rows, [
        { caseId: CASE_A, elapsedMs: 90_000, defaultRepeat: { perDispatch: 1, driftRatio: 1.02 } },
      ]),
    ).toEqual([
      {
        caseId: CASE_A,
        rows: 5,
        defaultPerDispatch: 20_000_000,
        driftRatio: 1.02,
        fastest: { geometry: "fast", speedup: 1.3 },
        errors: 1,
        mismatches: 1,
        seconds: 90,
      },
    ]);
  });

  it("shows a case that is still running without drift or seconds, in the order rows came", () => {
    const summary = caseSummaryRows([row(CASE_B, true), row(CASE_A, true)], []);
    expect(summary.map(({ caseId }) => caseId)).toEqual([CASE_B, CASE_A]);
    expect(summary[0]).toEqual({
      caseId: CASE_B,
      rows: 1,
      defaultPerDispatch: 20_000_000,
      errors: 0,
      mismatches: 0,
    });
  });

  it("carries the re-measurement's failure", () => {
    const [summary] = caseSummaryRows(
      [row(CASE_A, true)],
      [{ caseId: CASE_A, elapsedMs: 1000, defaultRepeatError: "out of memory" }],
    );
    expect(summary?.driftError).toBe("out of memory");
  });
});
