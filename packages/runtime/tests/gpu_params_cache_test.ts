// The content-addressed cache of params buffers (resident in the Session). Every byte of a params buffer is a
// pure function of the graph, node.attrs and the resolved shapes, so the invariant is that params with the same
// content are created only once, both within a run and across runs.
//
// NOTE: the second run with the same bindings hits the derived plan (see the prepared-plan section below) and
// skips the whole derivation phase, so it never reaches #writeParams (lastRunParams is {0,0}). This check
// therefore observes reuse at 2 points: ① duplicate params within one run (2 nodes of the same shape), and
// ② a re-derivation run **after eviction** from the derived plans (prepared miss, every params a hit) — which also
// pins that the lifetimes of the 2 caches are independent (params unbounded, prepared with an LRU limit).
//
// This check looks at 2 things: "0 new allocations with reuse > 0" and "exactly equal values". The first alone
// stays green even if the cache hands out the wrong buffer; the second alone stays green even if the cache
// silently misses and falls back to allocating per dispatch (neither throws).

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import {
  createSessionFromContainer,
  PREPARED_PLAN_CAPACITY,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { injectOutOfMemory } from "./helpers/oom-injection.ts";

/**
 * y[T] = Σ_c ((x[T,3] · w[3,2] + b[2]) + b[2]). Besides 3 nodes with different params shapes (matmul / add /
 * sum), **one more add of the same shape** is placed — every params byte of the second add matches the first,
 * so one reuse can be observed within the first run.
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add", "sum"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    g2: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T"] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "add", ins: ["g", "b"], outs: ["g2"], attrs: {} },
    { op: "sum", ins: ["g2"], outs: ["y"], attrs: { dim: 1 } },
  ],
};

const openedModel = () =>
  openGraphModel(GRAPH, [
    {
      graph: "model",
      initializer: "w",
      bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
      encoding: { codec: "f32" },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes([0.125, -0.5]),
      encoding: { codec: "f32" },
    },
  ]);

const input = (rows: number): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5),
});

test.skipIf(!GPU_AVAILABLE)(
  "params with the same content fold into one within a run, and a hit run touches none (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      assertEquals(
        session.diagnostics().lastRunParams,
        undefined,
        "the diagnostics are undefined before any run",
      );

      const first = await session.run({ x: input(4) });
      const firstParams = session.diagnostics().lastRunParams;
      assertEquals(
        firstParams?.reuseCount,
        1,
        "the second add of the same shape is reused within the first run",
      );
      assertEquals((firstParams?.allocCount ?? 0) > 0, true, "the first run makes new allocations");
      // The weights arena also owns the params buffers (they are destroyed along with the weights' dispose).
      const weightsAfterFirst = session.diagnostics().weights.allocCount;

      // The second run hits the derived plan and skips the whole derivation phase — zero params GPU operations as well
      // (not because the cache missed, but because the recipes reference the params buffers directly).
      const second = await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: 0 });
      assertEquals(
        session.diagnostics().weights.allocCount,
        weightsAfterFirst,
        "a hit run adds nothing to the arena",
      );

      // Reused params do not change the values (the cache did not hand out a buffer with different content)
      assertEquals(second["y"].shape, first["y"].shape);
      assertEquals(Array.from(second["y"].data), Array.from(first["y"].data));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the params cache survives eviction from the derived plans and hits on everything (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      await session.run({ x: input(4) });
      const narrow = session.diagnostics().lastRunParams;
      const narrowAlloc = narrow?.allocCount ?? 0;

      // When a symbolic dimension changes, the resolved shapes change and so does the params content (= a different key).
      await session.run({ x: input(9) }, { T: 9 });
      assertEquals(
        session.diagnostics().lastRunParams?.allocCount,
        narrowAlloc,
        "a different shape is rebuilt",
      );

      // Evict T=4 from the derived plans (LRU limit) for sure — running limit + 1 different shapes always drops the
      // oldest, T=4. The params cache is unbounded, so the params created here remain.
      // MUST: derive the count from the constant (a copy would stop evicting when the limit moves, and the very
      // observation point "prepared miss yet every params a hit" would disappear).
      for (let rows = 5; rows <= 5 + PREPARED_PLAN_CAPACITY; rows += 1) {
        await session.run({ x: input(rows) });
      }

      // A re-derivation run (prepared misses) yet not a single params is rebuilt — the first run's buffers are still
      // resident in the Session (+1 is the in-run reuse of the same-shape add).
      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "the plan has been evicted");
      assertEquals(session.diagnostics().lastRunParams, {
        allocCount: 0,
        reuseCount: narrowAlloc + 1,
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Pins that the params cache is **unbounded over the Session's lifetime** (by design — docs/limitations.md). It
 * turns red if an eviction mechanism is added, so adding one means re-deciding together with this check.
 *
 * The other caches of the same scope (`SubmitScheduler.MEASURED_HISTORY` / `PREPARED_PLAN_CAPACITY`) state
 * "MUST: bounded", and only this one is outside that rule. Without pinning it in numbers, the constraint "running
 * variable shapes for a long time grows GPU buffers monotonically" is invisible from the behaviour (no exception,
 * no warning; only `diagnostics().weights.allocCount` grows).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the params cache is unbounded over the Session's lifetime (grows monotonically per variable-shape run; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      // Starts from a state holding only the 2 initializers.
      assertEquals(session.diagnostics().weights.allocCount, 2);

      // Change the symbolic dimension T each time = the params content changes each time = never a hit.
      // The plans are evicted from the derived plans (LRU limit), but not a single params buffer is returned.
      const RUNS = PREPARED_PLAN_CAPACITY + 1;
      for (let i = 0; i < RUNS; i += 1) {
        await session.run({ x: input(2 + i) });
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          `run ${i}: 3 kinds of params newly allocated (only the second same-shape add is reused within the run)`,
        );
      }

      assertEquals(
        session.diagnostics().weights.allocCount,
        2 + RUNS * 3,
        "nothing evicted at all (2 initializers + runs × 3 node kinds)",
      );
      assertEquals(
        session.diagnostics().lastRunPrepared?.cachedPlans,
        PREPARED_PLAN_CAPACITY,
        "only the derived plans level off at the limit (their lifetime is independent of the params cache)",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** The output of a single run on a new Session (what the values after rollback are compared against). */
const freshOutput = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  rows: number,
): Promise<readonly number[]> => {
  const session = await createSessionFromContainer(gpu, await openedModel(), "model");
  try {
    return Array.from((await session.run({ x: input(rows) }))["y"].data);
  } finally {
    await session.dispose();
  }
};

// A non-fatal OOM (`createBuffer` merely returns an invalid buffer without throwing) is discovered only in the
// errorScope **after** params and the plan have been put into the caches. Without a rollback, a retry after
// memory frees up hits a plan that references the same invalid buffer directly, and validation errors continue
// for the rest of the Session's lifetime. The rollback discards only "what that execution newly added"; existing
// healthy entries are kept.
describe("after an execution whose params allocation failed with a non-fatal OOM (injected; real GPU)", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "discards only the params and plan the failed run added, rebuilds them on the next run, and keeps the existing plans",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const oom = injectOutOfMemory(gpu.device);
      try {
        await session.run({ x: input(4) });

        oom.arm();
        await assertRejects(
          () => session.run({ x: input(9) }, { T: 9 }),
          GpuOutOfMemoryError,
          "注入: 非致命 OOM",
        );
        assertEquals(
          oom.fired(),
          1,
          "the injection did not hit the run's pop (the check would test nothing)",
        );

        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "discards even existing plans unrelated to the failure",
        );

        const retried = await session.run({ x: input(9) }, { T: 9 });
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          false,
          "the plan registered by the failed run remains",
        );
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          "reuses from the cache the params allocated by the failed run",
        );
        assertEquals(Array.from(retried["y"].data), await freshOutput(gpu, 9));
      } finally {
        oom.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "discards the params and plans added by the enqueue of a batch that settled with a failure, and rebuilds them in the next batch",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const oom = injectOutOfMemory(gpu.device);
      try {
        const failed = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch: failed });
        assertEquals(
          session.diagnostics().lastRunParams?.allocCount,
          3,
          "the derivation did not run",
        );
        oom.arm();
        await assertRejects(() => failed.finish(), GpuOutOfMemoryError, "注入: 非致命 OOM");
        assertEquals(
          oom.fired(),
          1,
          "the injection did not hit finish's pop (the check would test nothing)",
        );

        const retry = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch: retry });
        await retry.finish();
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          false,
          "the plan registered by the failed batch's enqueue remains",
        );
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          "reuses from the cache the params allocated by the failed batch's enqueue",
        );
        assertEquals(
          Array.from((await session.run({ x: input(4) }))["y"].data),
          await freshOutput(gpu, 4),
        );
      } finally {
        oom.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

// Unless the failure is on the GPU side, the params and plans that were created are valid, so they are not
// removed. Removing them would allocate new params on every failure that fails the same way every time (an input
// error), and since they are only removed from the cache, they would pile up until `dispose` (params are not
// destroyed; the weights arena owns them — `RecipeBuilder.forgetParams`).
// enqueue derives before checking its inputs, so even an input error fails after the params were put in.
describe("when an enqueue that fails on a host-side error (wrong input element count) is repeated (real GPU)", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "keeps the params and plan created the first time, and does not allocate new params on repetition",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      // A wrong element count passes binding resolution and fails at the input check after derivation.
      const wrongCount: Tensor = { dtype: "f32", shape: [4, 3], data: new Float32Array(11) };
      const failOnce = async (): Promise<void> => {
        const batch = await gpu.beginBatch();
        await assertRejects(() => session.enqueue({ x: wrongCount }, { batch }), ExecutionError);
        await assertRejects(() => batch.finish(), ExecutionError);
      };
      try {
        const initial = session.diagnostics().weights.allocCount;
        await failOnce();
        const derived = session.diagnostics().weights.allocCount;
        assertEquals(
          derived - initial,
          3,
          "the first enqueue did not derive (the check would test nothing)",
        );

        await failOnce();
        await failOnce();
        assertEquals(
          session.diagnostics().weights.allocCount,
          derived,
          "allocates new params on every identical host-side failure",
        );

        const batch = await gpu.beginBatch();
        await session.enqueue({ x: input(4) }, { batch });
        await batch.finish();
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "discards a plan that was created, on a host-side failure",
        );
        assertEquals(
          Array.from((await session.run({ x: input(4) }))["y"].data),
          await freshOutput(gpu, 4),
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

// --- Prepared-plan cache ---
//
// The Session-resident cache of derived execution plans (key = the resolved bindings). Planning, fusion decisions
// and recipe derivation are pure functions of the graph and the bindings only, so from the second run with the
// same bindings on, the whole derivation phase can be skipped (#preparedKey in src/runtime/executor.ts).
//
// This check looks at 2 things: "the hit / miss decision" and "the hit run's output is bit-identical to the miss
// run's". The first alone stays green even if a wrong plan is handed out; the second alone stays green even if
// the cache silently misses and falls back to deriving on every run (neither throws).

/**
 * y = silu(x·w + b) (x: [T,3] → y: [T,2]). The last 2 nodes are caught by silu fusion, so the fusion counters
 * (the standing diagnostics of ADR 0040 §3) can be checked on the hit run as well.
 */
const SILU_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add", "sigmoid", "mul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    s: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T", 2] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "sigmoid", ins: ["g"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["g", "s"], outs: ["y"], attrs: {} },
  ],
};

const openProjection = (): Promise<OpenedContainer> =>
  openGraphModel(SILU_GRAPH, [
    {
      graph: "model",
      initializer: "w",
      bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
      encoding: { codec: "f32" },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes([0.125, -0.5]),
      encoding: { codec: "f32" },
    },
  ]);

/** The output bit pattern. Values are compared by "byte equality" (a rounding mix-up is not tolerated). */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

test.skipIf(!GPU_AVAILABLE)(
  "the second run with the same bindings hits the derived plan, and the output is bit-identical (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      assertEquals(
        session.diagnostics().lastRunPrepared,
        undefined,
        "the diagnostics are undefined before any run",
      );

      const first = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 1 },
        "the first run derives and then stores",
      );

      const second = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: true, cachedPlans: 1 },
        "the second run skips the derivation phase",
      );
      // Only the derivation is skipped; the command sequence put on the GPU is identical — the output is the only detector.
      assertEquals(second["y"].shape, first["y"].shape);
      assertEquals(bits(second["y"]), bits(first["y"]));

      // The whole derivation phase is skipped, so the params GPU operations are zero too (the meaning of the values is unchanged).
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: 0 });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a run with changed bindings re-derives, and switching back hits again (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      const narrow = await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });

      // When a symbolic dimension changes, the resolved bindings change (= a different key).
      await session.run({ x: input(9) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 2 },
        "different bindings are re-derived and stored as a second plan",
      );

      const again = await session.run({ x: input(4) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: true, cachedPlans: 2 },
        "the plan for the earlier bindings is still resident in the Session",
      );
      assertEquals(bits(again["y"]), bits(narrow["y"]));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the derived plans level off at the limit count and are evicted oldest first (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    const capacity = PREPARED_PLAN_CAPACITY;
    try {
      for (let rows = 1; rows <= capacity; rows += 1) {
        await session.run({ x: input(rows) });
        assertEquals(
          session.diagnostics().lastRunPrepared,
          { hit: false, cachedPlans: rows },
          `T=${rows} is new`,
        );
      }

      // The limit + 1-th kind hits the ceiling. It is stored and then the oldest (T=1) dropped, so the count does not change.
      await session.run({ x: input(capacity + 1) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: capacity });

      // Proof of eviction: T=1, which hit once, misses again.
      await session.run({ x: input(1) });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: capacity },
        "the oldest bindings have been dropped",
      );
      // The rest are intact (the other side of eviction being one at a time).
      await session.run({ x: input(capacity) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: capacity });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a hit run reports the fusion counts too (the cache does not erase the standing diagnostics; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openProjection(), "model");
    try {
      await session.run({ x: input(4) });
      const derived = session.diagnostics().lastRunFusions;
      assertEquals(
        derived?.silu,
        1,
        "the first run catches the silu fusion (without it the check would test nothing)",
      );

      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      const cached = session.diagnostics().lastRunFusions;
      assertEquals(cached === undefined, false, "not undefined even on a hit run");
      assertEquals(cached, derived, "the counts decided at planning are reported as they are");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the partial recipes of a run that failed in the derivation phase are not cached (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // Exceeding the limit is decided by the matmul's **output tile edge** (`ceil(n / tileN) > limit`). The edge is
    // derived from the default geometry, not a constant — written directly, changing the edge would stop the throw
    // and only assertRejects would fail, quietly (the same reason as runtime_executor_test.ts).
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
    const session = await createSessionFromContainer(gpu, await openGraphModel(graph), "model");
    const small = () => ({ x0: fill([1, 1], () => 1), x1: fill([1, 4], () => 1) });
    try {
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });

      // The second node (matmul) fails the limit in the derivation phase. The recipe list exists up to the first
      // node, but registration happens only after #buildRecipes completes.
      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, huge], () => 1) }),
        DispatchLimitError,
      );
      assertEquals(
        session.diagnostics().lastRunPrepared,
        undefined,
        "a run that did not settle in the derivation phase leaves no record",
      );

      // The count did not grow = the partial recipes of the failed run were not stored.
      await session.run(small());
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);
