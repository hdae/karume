// The GEMV sweep on a real GPU (src/tune/gemv-sweep.ts): one small shape per storage with every
// candidate kind — the column variants, a row-block height, the parallel GEMV plain and staged, and
// the subgroup form where the device has it; for the float storages the column variants and the
// parallel GEMV against the GEMM past one row (the GEMM of the profile the device resolves to: an
// injected table with another GEMM geometry changes the reference's kernel and no other) — measured
// through the sweep's own path, the public
// entry point over one case of the table, and the calibration of a case's first kernel after the
// workbench entry's idle time. The GPU sections are SKIPped explicitly without an adapter.
//
// Faults injected, one per check: a reordering kernel put into the identical class (its words must
// differ from the baseline's); a band of zero width (every reordering row and the baseline must read
// outside it); a case that starts after 10 s of idle (the cold device the warm start exists for:
// measured without it, the baseline of i4g512-n256-k1536 got 8 and 7 reps where most kernels after
// it got thousands, in both quick runs of 2026-11-17 on an RTX 3080 Ti); a fast kernel whose words
// differ (it must not set its class's bar: the baseline's own kernel goes on).
//
// One explore run over a sentinel and a predicted case: its order, its narrowing at M = 1, its
// skipping at M = 4 and the progress it reports, each read against the record it returns.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  EXPLORE_QUICK_LOOK_MARGIN,
  gemvExploreSelection,
  gemvSentinels,
  gemvSweepExploreEvidence,
  type GemvSweepProgress,
  QUICK_LOOK_MARGIN,
  runGemvSweep,
} from "../tune.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { defaultGemmGeometry } from "../src/kernels/gemm-geometry.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import {
  type GemvCandidate,
  type GemvGates,
  gemvBaseline,
  gemvCandidates,
  gemvExploreCarriedFromM1,
  type GemvMeasurementPlan,
  gemvPlanReferences,
  gemvProfileChoice,
  gemvProgram,
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
  // A device without a table of its own runs the default's choices
  table: {
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
  /** A table injected in place of the one the adapter selects. */
  geometryProfile?: GeometryProfile,
): Promise<GemvCaseResult> => {
  const gpu = await acquireGpu({
    subgroups: "available",
    ...(timestampQuery ? { gpuTiming: true } : {}),
    ...(geometryProfile === undefined ? {} : { geometryProfile }),
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

  it("measures a float-stored case's GEMM with the tile geometry of the profile its device resolves to (an injected table with one GEMM geometry for every row count)", async () => {
    const [gemvCase] = SMALL_FLOAT;
    // The default profile and every embedded table take reg16x16r1x4w4 at 4 rows; this table does not
    const oneBucket: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      id: "test-one-bucket",
      gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: defaultGemmGeometry() }],
    };
    const plans = (gates: GemvGates): GemvMeasurementPlan[] => [
      planOf(gemvCase, 4, pick(gemvCase, 4, gates, ["parallel-l8"])),
    ];
    const selected = await sweepWith(gemvCase, plans);
    const injected = await sweepWith(gemvCase, plans, {}, oneBucket);
    const rowOf = (result: GemvCaseResult, name: string): GemvSweepRow => {
      const row = result.rows.find((one) => one.candidate === name);
      if (row === undefined) throw new Error(`no row ${name}`);
      assertEquals(row.error, undefined, rowLabel(row));
      assertEquals(row.withinBand, true, rowLabel(row));
      return row;
    };
    assertEquals(rowOf(selected, "gemm").keys, ["linear:v2:f32:reg16x16r1x4w4v4:wbf16"]);
    assertEquals(rowOf(injected, "gemm").keys, ["linear:v2:f32:reg128x128r8x8w16v4:wbf16"]);
    const { choice } = gemvBaseline(gemvCase, 4);
    assertEquals(
      rowOf(injected, "gemm").kernel,
      gemvKernelId(gemvCase, 4, choice, (one, m, chosen) => gemvProgram(one, m, chosen, oneBucket)),
    );
    assert(rowOf(injected, "gemm").kernel !== rowOf(selected, "gemm").kernel);
    // The candidate is the same kernel under both
    assertEquals(rowOf(injected, "parallel-l8").kernel, rowOf(selected, "parallel-l8").kernel);
  });

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

  it("lets only a kernel whose check held set its class's bar: a faster kernel whose words differ does not screen out the baseline's own kernel (fault: parallel-l32 put into the identical class)", async () => {
    // k = 16384 over 100 columns: the sequential kernel loops 4096 words a column on a few
    // workgroups, the parallel one spreads them over 32 lanes — far faster
    const gemvCase: GemvCase = {
      id: "test-i8-n100-k16384",
      storage: "i8",
      n: 100,
      k: 16384,
      rows: [],
      source: "test",
    };
    const baseline = gemvBaseline(gemvCase, 1);
    const result = await sweepWith(
      gemvCase,
      (gates) => {
        const [reordered] = pick(gemvCase, 1, gates, ["parallel-l32"]);
        return [
          planOf(gemvCase, 1, [
            { ...baseline, name: "baseline-again" },
            { ...reordered, name: "wrong-words", class: "identical" },
          ]),
        ];
      },
      { quickLookMargin: QUICK_LOOK_MARGIN },
    );
    const again = result.rows.find((row) => row.candidate === "baseline-again");
    const wrong = result.rows.find((row) => row.candidate === "wrong-words");
    assert(again?.quickPerDispatch !== undefined && wrong?.quickPerDispatch !== undefined);
    // Precondition: the wrong kernel's quick time alone would screen the baseline's own kernel out
    assert(
      wrong.quickPerDispatch * QUICK_LOOK_MARGIN < again.quickPerDispatch,
      `quick ${wrong.quickPerDispatch} against ${again.quickPerDispatch}: the fault would not bite`,
    );
    assertEquals(
      [again.screened, again.rounds?.length, again.identicalToBaseline],
      [undefined, 1, true],
    );
    // The wrong kernel goes on by its own time, and its row records step 4's check
    assertEquals([wrong.screened, wrong.identicalToBaseline], [undefined, false]);
  });

  it("measures an explore run over a sentinel and a predicted case: the sentinel first with every candidate, the other narrowed at M = 1 to its prediction, M = 4 to what M = 1 carries; its progress counts what the record holds", async () => {
    const sentinelId = "i8-n2048-k256";
    const otherId = "i8-n1536-k256";
    const progress: GemvSweepProgress[] = [];
    const record = await runGemvSweep({
      candidateSet: "explore",
      cases: [otherId, sentinelId],
      rows: [1, 4],
      rounds: 1,
      onProgress: (event) => progress.push(event),
    });
    assertEquals(record.format, "karume-gemv-sweep/4");
    assertEquals(record.deviceLost, null);
    assertEquals(record.settings.candidateSet, "explore");
    assertEquals(record.settings.sentinels, gemvSentinels(GEMV_CASES));
    assertEquals(record.settings.quickLookMargin, EXPLORE_QUICK_LOOK_MARGIN);
    assertEquals(
      record.cases.map(({ id }) => id),
      [sentinelId, otherId],
    );
    // In measuring order, as the workbench's merged record holds them
    assertEquals(record.settings.cases, [sentinelId, otherId]);
    const gates: GemvGates = {
      subgroups: record.gates.subgroups,
      maxComputeWorkgroupsPerDimension: record.gates.maxComputeWorkgroupsPerDimension,
    };
    const rowsOf = (caseId: string, m: number) =>
      record.rows.filter((row) => row.caseId === caseId && row.m === m);
    const summaryOf = (caseId: string, m: number) => {
      const found = record.summaries.find(
        (summary) => summary.caseId === caseId && summary.m === m,
      );
      if (found === undefined) throw new Error(`no summary of ${caseId} M=${m}`);
      return found;
    };
    const caseOf = (id: string): GemvCase => {
      const found = GEMV_CASES.find((one) => one.id === id);
      if (found === undefined) throw new Error(`no case ${id}`);
      return found;
    };
    // The sentinel measures every candidate at M = 1
    const sentinel = caseOf(sentinelId);
    assertEquals(
      rowsOf(sentinelId, 1).map(({ candidate }) => candidate),
      [gemvBaseline(sentinel, 1), ...gemvCandidates(sentinel, 1, "explore", gates)].map(
        ({ name }) => name,
      ),
    );
    assertEquals(summaryOf(sentinelId, 1).narrowedTo, undefined);
    // The other case: narrowed to what the sentinel of its storage predicts (its references always
    // go on, so it has a winner of each class)
    const other = caseOf(otherId);
    const selection = gemvExploreSelection(other, gemvSweepExploreEvidence(record));
    assert(selection !== undefined, `${sentinelId} predicts nothing for ${otherId}`);
    assertEquals(summaryOf(otherId, 1).narrowedTo, selection);
    /** The references of a group: the default's choices and the device table's (always measured). */
    const referencesOf = (caseId: string, m: number): ReadonlySet<string> => {
      const { profile, table } = summaryOf(caseId, m);
      assert(table !== undefined, `${caseId} M=${m}: the summary names no table choices`);
      return gemvPlanReferences({ profile, table });
    };
    for (const row of rowsOf(otherId, 1)) {
      if (row.isBaseline || referencesOf(otherId, 1).has(row.candidate)) continue;
      assert(selection.includes(row.candidate), `${row.candidate} is not in ${selection.join()}`);
    }
    // M = 4: a reorder candidate is measured exactly when it is a reference or the case's M = 1
    // group carries it
    for (const caseId of [sentinelId, otherId]) {
      const first = { ...summaryOf(caseId, 1), rows: rowsOf(caseId, 1) };
      const references = summaryOf(caseId, 4).profile;
      const four = gemvCandidates(caseOf(caseId), 4, "explore", gates);
      for (const row of rowsOf(caseId, 4)) {
        const candidate = four.find(({ name }) => name === row.candidate);
        if (row.isBaseline || candidate === undefined) continue;
        assertEquals(
          row.skipped === true,
          !referencesOf(caseId, 4).has(row.candidate) &&
            !gemvExploreCarriedFromM1(4, candidate, references, first),
          rowLabel(row),
        );
      }
      assert(summaryOf(caseId, 4).skipped > 0, `${caseId}: nothing skipped at M = 4`);
    }
    // Every reference that was timed is timed once more after the rounds, and only a reference
    for (const summary of record.summaries) {
      const references = referencesOf(summary.caseId, summary.m);
      for (const row of rowsOf(summary.caseId, summary.m)) {
        const timedReference = references.has(row.candidate) && row.perDispatch !== undefined;
        assertEquals(row.retimedPerDispatch !== undefined, timedReference, rowLabel(row));
      }
    }
    for (const row of record.rows) {
      assertEquals(row.error, undefined, rowLabel(row));
      if (row.screened === true || row.skipped === true || row.isBaseline) continue;
      if (row.class === "identical") assertEquals(row.identicalToBaseline, true, rowLabel(row));
      else assertEquals(row.withinBand, true, rowLabel(row));
    }
    // The progress: one plan per case in measuring order, and steps whose totals are the record's
    const plans = progress.filter((event) => event.kind === "plan");
    assertEquals(
      plans.map(({ caseId, index, rowCounts }) => [caseId, index, rowCounts.map(({ m }) => m)]),
      [
        [sentinelId, 0, [1, 4]],
        [otherId, 1, [1, 4]],
      ],
    );
    const steps = progress.filter((event) => event.kind === "step");
    for (const plan of plans) {
      for (const { m, kernels } of plan.rowCounts) {
        const rows = rowsOf(plan.caseId, m);
        assertEquals(kernels, rows.length, `${plan.caseId} M=${m} kernels`);
        const totalOf = (phase: string) =>
          steps.find((step) => step.caseId === plan.caseId && step.m === m && step.phase === phase)
            ?.total;
        const measured = rows.filter((row) => row.skipped !== true).length;
        const timed = rows.filter((row) => row.skipped !== true && row.screened !== true).length;
        // One round, and one pass more for each reference re-timed after it
        const passes = timed + rows.filter((row) => row.retimedPerDispatch !== undefined).length;
        assertEquals(
          [totalOf("prepare"), totalOf("quick-look"), totalOf("calibrate"), totalOf("rounds")],
          [measured, measured, timed, passes],
          `${plan.caseId} M=${m}`,
        );
        const last = steps.filter((step) => step.caseId === plan.caseId && step.m === m).at(-1);
        assertEquals([last?.phase, last?.done, last?.total], ["rounds", passes, passes]);
      }
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
    // The profile a Session on this adapter runs with (nothing is injected here)
    assert(adapter !== null);
    assertEquals(record.geometryProfile, selectGeometryProfile(adapter.info).id);
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
