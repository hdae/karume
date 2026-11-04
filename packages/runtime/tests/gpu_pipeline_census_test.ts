// Real-GPU check that pins `SessionDiagnostics.lastRunPipelines` (the planned dispatch counts of the latest
// run, per pipeline key). The behaviour of the pure function is in runtime_pipeline_census_test.ts.
//
// Three things are checked:
// ① On a device with timing, the key set and counts match the breakdown of the dispatches actually issued
//    (`lastRunTiming`) (on both a miss run and a hit run — a hit run skips the derivation phase, so a census
//    sourced from the Session's cumulative key set would test nothing).
// ② It is filled on the resident path without timing too (enqueue — ADR 0054) (the reason this field exists).
// ③ Right after a run that failed in recipe generation it is undefined (the table of the previous successful run is not kept).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import type { GpuTimingStats } from "../src/gpu/submit.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { createSessionFromContainer, type Session } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

type Census = readonly { readonly key: string; readonly dispatchCount: number }[];

/**
 * y = relu(neg(relu(x))) + relu(x). relu is used twice, so one row carries 2 dispatches on the same key
 * (elementwise keys are determined by op / rank / dtype — evidence that the count aggregation is not vacuous).
 * MUST: the graph has no states (= every workgroup count is a static triple). A states-form dispatch computed
 * from a logical length is not actually encoded for a zero-work step, so the planned count (census) comes out
 * higher than the actual issue (timing) — here we want to assert an exact match, so only shapes where that
 * difference cannot arise are used.
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["relu", "neg", "add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [4, 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [4, 4] },
    b: { dtype: "f32", shape: [4, 4] },
    c: { dtype: "f32", shape: [4, 4] },
    y: { dtype: "f32", shape: [4, 4] },
  },
  nodes: [
    { op: "relu", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "neg", ins: ["a"], outs: ["b"], attrs: {} },
    { op: "relu", ins: ["b"], outs: ["c"], attrs: {} },
    { op: "add", ins: ["c", "a"], outs: ["y"], attrs: {} },
  ],
};

const input = () => ({ x: fill([4, 4], (i) => (i % 7) - 3) });

const openSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(GRAPH), GRAPH_NAME);

const byKey = (a: { readonly key: string }, b: { readonly key: string }): number =>
  a.key < b.key ? -1 : a.key > b.key ? 1 : 0;

/** Maps the actually issued breakdown into the same shape as the census (keys in lexicographic order; `entries` is in descending ns). */
const issued = (timing: GpuTimingStats): Census =>
  timing.entries.map(({ key, dispatchCount }) => ({ key, dispatchCount })).sort(byKey);

/** Checks that the census is not vacuous (two empty tables matching must not pass). */
const assertNonTrivial = (census: Census | undefined): Census => {
  assert(census !== undefined, "lastRunPipelines is not filled");
  assert(census.length > 0, "the census is empty (a miscount hidden behind a match)");
  assert(
    census.some((entry) => entry.dispatchCount >= 2),
    "no row carries 2 or more dispatches on the same key (the count aggregation is not exercised)",
  );
  return census;
};

test.skipIf(!GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE)(
  "lastRunPipelines matches the key set and counts of the timed actual issue (miss / hit run; real GPU)",
  async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    const session = await openSession(gpu);
    try {
      assertEquals(session.diagnostics().lastRunPipelines, undefined, "undefined before any run");
      for (const hit of [false, true]) {
        await session.run(input());
        const diagnostics = session.diagnostics();
        assertEquals(diagnostics.lastRunPrepared?.hit, hit);
        const timing = diagnostics.lastRunTiming;
        assert(timing !== undefined, "no breakdown despite gpuTiming: true");
        const census = assertNonTrivial(diagnostics.lastRunPipelines);
        assertEquals(
          census,
          issued(timing),
          `hit=${hit}: planned counts and actual issue disagree`,
        );
        assertEquals(
          census.reduce((total, entry) => total + entry.dispatchCount, 0),
          timing.dispatchCount,
        );
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "lastRunPipelines is filled on the resident path without timing (enqueue) too, with the same table as run (real GPU)",
  async () => {
    // A device that does not request timing (with timing enabled, beginBatch is rejected — ADR 0054).
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const batch = await gpu.beginBatch();
      // The first is derived on the enqueue side (miss), the second is a hit. It must be filled right after both.
      // `#prepareInvocation` resets it to undefined every time, so what is filled right after the second is the
      // value the second wrote itself (not a leftover from the first).
      const seen: Census[] = [];
      for (const hit of [false, true]) {
        await session.enqueue(input(), { batch });
        const diagnostics = session.diagnostics();
        assertEquals(diagnostics.lastRunPrepared?.hit, hit);
        assertEquals(
          diagnostics.lastRunTiming,
          undefined,
          "the resident path does not open a timing window",
        );
        seen.push(assertNonTrivial(diagnostics.lastRunPipelines));
      }
      await batch.finish();
      assertEquals(seen[1], seen[0], "the table changes between miss and hit of the same plan");

      // Control: the same table as a run with the same inputs (which hits the same derived plan).
      await session.run(input());
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunPipelines, seen[0]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "right after a run that failed in recipe generation, lastRunPipelines is undefined (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // The fault is injected in **recipe generation** (the matmul of `RecipeBuilder.buildRecipes` throws for
    // exceeding the dispatch limit — the same setup as gpu_params_cache_test.ts). The output tile edge is derived
    // from the default geometry (if it were written as a constant, changing the edge would stop the throw and only
    // assertRejects would fail, quietly).
    // NOTE: failures of `bindSymbols` (input shape validation) and `planSteps` (planning and fusion decisions) are
    // out of scope — both throw before `#prepareInvocation` resets the "latest run" slot.
    const tileN = gemmTileN(defaultGemmGeometry());
    const huge = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["relu", "matmul"] },
      symbols: ["N"],
      inputs: [
        { name: "x0", dtype: "f32", shape: [1, 1] },
        { name: "x1", dtype: "f32", shape: [1, "N"] },
      ],
      outputs: ["y"],
      initializers: {},
      values: {
        t: { dtype: "f32", shape: [1, 1] },
        y: { dtype: "f32", shape: [1, "N"] },
      },
      nodes: [
        { op: "relu", ins: ["x0"], outs: ["t"], attrs: {} },
        { op: "matmul", ins: ["t", "x1"], outs: ["y"], attrs: {} },
      ],
    };
    const session = await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME);
    const small = () => ({ x0: fill([1, 1], () => 1), x1: fill([1, 4], () => 1) });
    try {
      // Control: it is filled right after a successful run (proof that the failed run's undefined is not "empty from the start").
      await session.run(small());
      const before = session.diagnostics().lastRunPipelines;
      assert(before !== undefined && before.length > 0);

      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, huge], () => 1) }),
        DispatchLimitError,
      );
      const failed = session.diagnostics();
      assertEquals(
        failed.lastRunPipelines,
        undefined,
        "the failed run reports the previous run's table",
      );
      // The fusion count stays because it is the failed run's own planning result (only the "slot after the recipe list is fixed" was reset).
      assertEquals(failed.lastRunFusions !== undefined, true);

      // Recovery: a run with the same bindings hits the original plan and returns the same table.
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunPipelines, before);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);
