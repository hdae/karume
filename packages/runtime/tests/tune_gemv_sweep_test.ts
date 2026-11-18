// The GEMV sweep on a real GPU (src/tune/gemv-sweep.ts): one small shape per storage with every
// candidate kind — the column variants, a row-block height, the parallel GEMV plain and staged, and
// the subgroup form where the device has it; for the float storages the column variants and the
// parallel GEMV against the GEMM past one row — measured through the sweep's own path, the public
// entry point over one case of the table, and the calibration of a case's first kernel after the
// workbench entry's idle time. The GPU sections are SKIPped explicitly without an adapter.
//
// Faults injected, one per check: a reordering kernel put into the identical class (its words must
// differ from the baseline's); a band of zero width (every reordering row and the baseline must read
// outside it); a case that starts after 10 s of idle (the cold device the warm start exists for:
// measured without it, the baseline of i4g512-n256-k1536 got 8 and 7 reps where most kernels after
// it got thousands, in both quick runs of 2026-11-17 on an RTX 3080 Ti).

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { runGemvSweep } from "../tune.ts";
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
import { GEMV_CASES, type GemvCase } from "../src/tune/gemv-cases.ts";
import { gemvCaseFingerprint, gemvKernelId } from "../src/tune/gemv-fingerprint.ts";
import { choiceOfRecord, type GemvSweepRow } from "../src/tune/gemv-report.ts";
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

/** A small shape per float storage: no scale buffer, no row block, the GEMM past one row. */
const SMALL_FLOAT: readonly GemvCase[] = (["bf16", "f16", "f32"] as const).map((storage) => ({
  id: `test-${storage}-n100-k1024`,
  storage,
  n: 100,
  k: 1024,
  rows: [],
  source: "test",
}));

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

/** Every candidate kind of a float storage: M = 1 a column variant and the reorder forms, M = 4 two lane counts against the GEMM. */
const everyFloatKind = (gemvCase: GemvCase, gates: GemvGates): GemvMeasurementPlan[] => [
  planOf(gemvCase, 1, pick(gemvCase, 1, gates, ["c64u8", "parallel-l4", "staged-l32"])),
  planOf(gemvCase, 4, pick(gemvCase, 4, gates, ["parallel-l8", "staged-l2"])),
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
      // Every kernel timed (no quick look): these cases check every candidate kind's output
      { rounds: 1, tolerance: opTolerance("linear"), quickLookMargin: 0, ...settings },
    );
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

const rowLabel = (row: GemvSweepRow): string => `${row.caseId} M=${row.m} ${row.candidate}`;

/** The idle time before every case but the first of the workbench's gemv-sweep entry (its default). */
const ENTRY_IDLE_MS = 10_000;

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
      // One round: the baseline's largest round is its smallest.
      for (const summary of result.summaries) assertEquals(summary.baselineSpread, 1);
    });
  }

  for (const gemvCase of SMALL_FLOAT) {
    it(`measures every candidate kind of ${gemvCase.id}: the column variant's words equal the sequential GEMV's, the parallel GEMV stays in the band at one row and against the GEMM at four`, async () => {
      const result = await sweepWith(gemvCase, (gates) => everyFloatKind(gemvCase, gates));
      assertEquals(result.aborted, false);
      assertEquals(
        result.rows.map((row) => [row.m, row.candidate, row.class, row.isBaseline]),
        [
          [1, "c32u4", "identical", true],
          [1, "c64u8", "identical", false],
          [1, "parallel-l4", "reorder", false],
          [1, "staged-l32", "reorder", false],
          [4, "gemm", "identical", true],
          [4, "parallel-l8", "reorder", false],
          [4, "staged-l2", "reorder", false],
        ],
      );
      for (const row of result.rows) {
        assertEquals(row.error, undefined, rowLabel(row));
        assert(row.perDispatch !== undefined && row.perDispatch > 0, rowLabel(row));
        assertEquals(row.keys?.length, 1, rowLabel(row));
        // The baselines' own distance from the oracle: the sequential GEMV, and the GEMM
        if (row.isBaseline || row.class === "reorder") {
          assertEquals(row.withinBand, true, rowLabel(row));
          assertEquals(row.nonFiniteOutputs, 0, rowLabel(row));
          assert(row.maxAbsError !== undefined && row.maxAbsError > 0, rowLabel(row));
        } else {
          assertEquals(row.identicalToBaseline, true, rowLabel(row));
        }
      }
      const gemm = result.rows.find((row) => row.candidate === "gemm");
      assertEquals(gemm?.choice, { id: "gemm-f32", storage: gemvCase.storage, v4: true });
      assert(gemm?.keys?.[0]?.startsWith("linear:"), gemm?.keys?.[0]);
    });
  }

  it("calibrates the first kernel of a case that starts after the entry's idle time like the same kernel later in the case (fault: a cold device)", async () => {
    const gemvCase = GEMV_CASES.find(({ id }) => id === "i4g512-n256-k1536");
    if (gemvCase === undefined) throw new Error("no case i4g512-n256-k1536");
    const baseline = gemvBaseline(gemvCase, 1);
    await new Promise((resolve) => setTimeout(resolve, ENTRY_IDLE_MS));
    const result = await sweepWith(gemvCase, (gates) => [
      planOf(gemvCase, 1, [
        ...pick(gemvCase, 1, gates, ["c16u2", "c64u8"]),
        // The baseline's kernel once more, as the last kernel of the row count.
        { ...baseline, name: "baseline-again" },
      ]),
    ]);
    const first = result.rows.find((row) => row.isBaseline);
    const again = result.rows.find((row) => row.candidate === "baseline-again");
    assert(first?.reps !== undefined && again?.reps !== undefined, "a row has no reps");
    assertEquals(again.identicalToBaseline, true);
    const ratio = Math.max(first.reps, again.reps) / Math.min(first.reps, again.reps);
    assert(ratio <= 2, `reps ${first.reps} first, ${again.reps} later in the case`);
  });

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
    assertEquals(record.format, "karume-gemv-sweep/4");
    assertEquals(record.deviceLost, null);
    assertEquals(record.aborted, false);
    assertEquals(record.settings.cases, [caseId]);
    assertEquals(
      record.cases.map(({ id, rows }) => [id, rows]),
      [[caseId, [1]]],
    );
    const planned = GEMV_CASES.find(({ id }) => id === caseId);
    assert(planned !== undefined);
    assertEquals(
      record.cases.map(({ fingerprint }) => fingerprint),
      [gemvCaseFingerprint(planned)],
    );
    for (const row of record.rows) {
      assertEquals(
        row.kernel,
        gemvKernelId(planned, row.m, choiceOfRecord(row.choice)),
        rowLabel(row),
      );
    }
    assertEquals(record.settings.quickLookMargin, 1.25);
    assertEquals(record.gpuTiming.unit, timestampQuery ? "ns" : "wall");
    assertEquals(record.rows.filter((row) => row.isBaseline).length, 1);
    for (const row of record.rows) {
      assertEquals(row.error, undefined, rowLabel(row));
      if (row.isBaseline) continue;
      // The quick look times in full only the kernels near enough to matter: a screened row has its
      // quick time and no check
      if (row.screened === true) {
        assert(row.quickPerDispatch !== undefined && row.rounds === undefined, rowLabel(row));
        continue;
      }
      if (row.class === "identical") assertEquals(row.identicalToBaseline, true, rowLabel(row));
      else assertEquals(row.withinBand, true, rowLabel(row));
    }
    // The references always go on
    for (const name of ["c32u4", "parallel-l4"]) {
      assertEquals(record.rows.find((row) => row.candidate === name)?.screened, undefined, name);
    }
    assertEquals(
      record.rows.some((row) => row.choice.id === "gemv-parallel-subgroup32"),
      record.gates.subgroups,
    );
    assertEquals(record.summaries.length, 1);
    assertEquals(record.summaries[0]?.profile, { exact: "c32u4", reorder: "parallel-l4" });
  });
});
