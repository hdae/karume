// The GEMV sweep on a real GPU (src/tune/gemv-sweep.ts): one small shape per storage with every
// candidate kind — the column variants, a row-block height, the parallel GEMV plain and staged, and
// the subgroup form where the device has it — measured through the sweep's own path, and the public
// entry point over one case of the table. The GPU sections are SKIPped explicitly without an adapter.
//
// Faults injected, one per check: a reordering kernel put into the identical class (its words must
// differ from the baseline's); a band of zero width (every reordering row and the baseline must read
// outside it).

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { gemvSweepKernelsId, runGemvSweep } from "../tune.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import {
  type GemvCandidate,
  type GemvGates,
  gemvBaseline,
  gemvCandidates,
  type GemvMeasurementPlan,
  gemvProfileChoice,
} from "../src/tune/gemv-candidates.ts";
import type { GemvCase } from "../src/tune/gemv-cases.ts";
import { gemvCaseTableId } from "../src/tune/gemv-fingerprint.ts";
import type { GemvSweepRow } from "../src/tune/gemv-report.ts";
import {
  type GemvCaseResult,
  type GemvCaseSettings,
  gemvSweepGates,
  sweepGemvCase,
} from "../src/tune/gemv-sweep.ts";
import { createSweepContext, destroySweepContext } from "../src/tune/harness.ts";

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

/** A small shape per storage: 100 columns (an edge workgroup), k = 1024 (a reduction worth reordering). */
const SMALL: readonly GemvCase[] = [
  {
    id: "test-i4g32-n100-k1024",
    storage: "i4",
    group: 32,
    n: 100,
    k: 1024,
    rows: [],
    source: "test",
  },
  { id: "test-i8-n100-k1024", storage: "i8", n: 100, k: 1024, rows: [], source: "test" },
  { id: "test-i2-n100-k1024", storage: "i2", n: 100, k: 1024, rows: [], source: "test" },
];

/** The candidates of `names` among the full set's (each must exist). */
const pick = (
  gemvCase: GemvCase,
  m: number,
  gates: GemvGates,
  names: readonly string[],
): GemvCandidate[] =>
  names.map((name) => {
    const found = gemvCandidates(gemvCase, m, "full", gates).find((one) => one.name === name);
    if (found === undefined) throw new Error(`${gemvCase.id} M=${m}: no candidate ${name}`);
    return found;
  });

const planOf = (
  gemvCase: GemvCase,
  m: number,
  candidates: GemvCandidate[],
): GemvMeasurementPlan => ({
  m,
  baseline: gemvBaseline(gemvCase, m),
  candidates,
  profile: {
    exact: gemvBaseline(gemvCase, m).name,
    reorder: gemvProfileChoice(gemvCase, m, "reorder").name,
  },
});

/** Every candidate kind: M = 1 a column variant and the reorder forms, M = 4 a height and a lane count, M = 16 a height. */
const everyKind = (gemvCase: GemvCase, gates: GemvGates): GemvMeasurementPlan[] => [
  planOf(
    gemvCase,
    1,
    pick(gemvCase, 1, gates, [
      "c64u8",
      "parallel-l4",
      "staged-l32",
      ...(gates.subgroups ? ["subgroup32-l8"] : []),
    ]),
  ),
  planOf(gemvCase, 4, pick(gemvCase, 4, gates, ["c32u4r2", "parallel-l8"])),
  planOf(gemvCase, 16, pick(gemvCase, 16, gates, ["c32u4r4"])),
];

/** Measures plans of one case on a fresh device with the sweep's per-case path. */
const sweepWith = async (
  gemvCase: GemvCase,
  plansFor: (gates: GemvGates) => GemvMeasurementPlan[],
  settings: Partial<GemvCaseSettings> = {},
): Promise<GemvCaseResult> => {
  const gpu = await acquireGpu({
    subgroups: "available",
    ...(timestampQuery ? { gpuTiming: true } : {}),
  });
  const context = await createSweepContext(gpu, "ns");
  try {
    const gates = gemvSweepGates(gpu);
    return await sweepGemvCase(
      context,
      gemvCase,
      plansFor({
        subgroups: gates.subgroups,
        maxComputeWorkgroupsPerDimension: gates.maxComputeWorkgroupsPerDimension,
      }),
      { rounds: 1, tolerance: opTolerance("linear"), ...settings },
    );
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

const rowLabel = (row: GemvSweepRow): string => `${row.caseId} M=${row.m} ${row.candidate}`;

describe.skipIf(adapter === null)("the GEMV sweep on a real GPU", () => {
  for (const gemvCase of SMALL) {
    it(`measures every candidate kind of ${gemvCase.id}: identical words equal the baseline's, reordered outputs stay in the band`, async () => {
      const result = await sweepWith(gemvCase, (gates) => everyKind(gemvCase, gates));
      assertEquals(result.aborted, false);
      assert(result.rows.length >= 9, `${result.rows.length} rows`);
      for (const row of result.rows) {
        assertEquals(row.error, undefined, rowLabel(row));
        assert(row.perDispatch !== undefined && row.perDispatch > 0, rowLabel(row));
        assertEquals(row.keys?.length, 1, rowLabel(row));
        if (row.isBaseline) continue;
        assert(row.speedupVsBaseline !== undefined && row.speedupVsBaseline > 0, rowLabel(row));
        if (row.class === "identical") {
          assertEquals(row.identicalToBaseline, true, rowLabel(row));
          assertEquals(row.differingWords, 0, rowLabel(row));
        } else {
          assertEquals(row.withinBand, true, rowLabel(row));
          assertEquals(row.nonFiniteOutputs, 0, rowLabel(row));
        }
      }
      // The baseline's own distance from the oracle is read at the row counts with reorder rows.
      const baselines = result.rows.filter((row) => row.isBaseline);
      assertEquals(
        baselines.map((row) => [row.m, row.withinBand]),
        [
          [1, true],
          [4, true],
          [16, undefined],
        ],
      );
      assertEquals(
        result.summaries.map(({ m }) => m),
        [1, 4, 16],
      );
      for (const summary of result.summaries) {
        assert(summary.baselineRepeat !== undefined && summary.baselineRepeat.driftRatio > 0);
      }
    });
  }

  it("fails the identical check of a kernel that sums in another order (fault: parallel-l32 put into the identical class)", async () => {
    const [gemvCase] = SMALL;
    const result = await sweepWith(gemvCase, (gates) => {
      const [reordered] = pick(gemvCase, 1, gates, ["parallel-l32"]);
      return [planOf(gemvCase, 1, [{ ...reordered, class: "identical" }])];
    });
    const [, fault] = result.rows;
    assertEquals(fault.error, undefined);
    assertEquals(fault.identicalToBaseline, false);
    assert(fault.differingWords !== undefined && fault.differingWords > 0);
    assert(fault.firstDifference !== undefined);
  });

  it("reads every reordered output outside a band of zero width (fault: atol = rtol = 0)", async () => {
    const [gemvCase] = SMALL;
    const result = await sweepWith(
      gemvCase,
      (gates) => [planOf(gemvCase, 1, pick(gemvCase, 1, gates, ["parallel-l4", "staged-l32"]))],
      { tolerance: { atol: 0, rtol: 0 } },
    );
    for (const row of result.rows) {
      assertEquals(row.error, undefined, rowLabel(row));
      assertEquals(row.withinBand, false, rowLabel(row));
    }
  });

  it("returns the record of the public entry point over one case of the table", async () => {
    const caseId = "i8-n1536-k256";
    const record = await runGemvSweep({ candidateSet: "quick", cases: [caseId], rounds: 1 });
    assertEquals(record.format, "karume-gemv-sweep/1");
    assertEquals(record.deviceLost, null);
    assertEquals(record.aborted, false);
    assertEquals(record.settings.cases, [caseId]);
    assertEquals(
      record.cases.map(({ id, rows }) => [id, rows]),
      [[caseId, [1]]],
    );
    assertEquals(record.caseTable, gemvCaseTableId());
    assertEquals(record.kernels, gemvSweepKernelsId(record.rows));
    assertEquals(record.gpuTiming.unit, timestampQuery ? "ns" : "wall");
    assertEquals(record.rows.filter((row) => row.isBaseline).length, 1);
    for (const row of record.rows) {
      assertEquals(row.error, undefined, rowLabel(row));
      if (row.isBaseline) continue;
      if (row.class === "identical") assertEquals(row.identicalToBaseline, true, rowLabel(row));
      else assertEquals(row.withinBand, true, rowLabel(row));
    }
    assertEquals(
      record.rows.some((row) => row.choice.id === "gemv-parallel-subgroup32"),
      record.gates.subgroups,
    );
    assertEquals(record.summaries.length, 1);
    assertEquals(record.summaries[0]?.profile, { exact: "c32u4", reorder: "parallel-l4" });
  });
});
