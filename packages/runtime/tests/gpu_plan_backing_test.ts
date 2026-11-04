// Checks of the GPU backing of transient slots (the Session-resident buffers used by a run that hits a derived plan).
//
// A backed run bypasses the arena's allocation and reference counting and writes straight into **buffers that
// still hold the previous run's leftovers**. Its correctness rests on full-write (ADR 0014) alone, so the detector
// is "does the output of runs with the same bindings and changing values match a non-backed execution bit for
// bit" — if even one stale slot slips in, the previous run's values remain and not a single exception is thrown.
//
// It also pins: ① a one-off run pays no slot memory at all; ② the backings are held as **an LRU set with a byte
// budget** (ADR 0095 decision 1 — signatures within the budget are kept rather than rebuilt on a switch; the excess
// is retired oldest first; a shape that exceeds the budget on its own retires everything and keeps only itself; a
// budget of 0 is the former capacity of 1. What the budget counts is **the regions + the input buffers the backing
// owns** = `residentBytes + inputBytes` of the diagnostics); ③ backings are released by the LRU eviction of plans;
// ④ the total slot bytes match the current run's pool allocation (footprint unchanged — a claim about **one
// backing**). If ④ breaks, the VRAM premise (going resident creates no new peak) breaks.

import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import {
  createSessionFromContainer,
  PREPARED_PLAN_CAPACITY,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { PlanBackingStats } from "../src/runtime/session-types.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { injectOutOfMemory } from "./helpers/oom-injection.ts";

/** y = silu(x·w + b) (x: [T,3] → y: [T,2]). The last 2 nodes are caught by silu fusion. */
const GRAPH: DeclarationJson = {
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

/** An input whose values change per `phase` (handing out the same values would not detect a stale slot). */
const input = (rows: number, phase = 0): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/** The output bit pattern. Values are compared by "byte equality" (a rounding mix-up is not tolerated). */
const bits = (tensor: Tensor): readonly number[] =>
  Array.from(new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length));

/** Rebuilds the Session and runs it once = always non-backed (a backing is built only on a hit run). */
const runFresh = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  inputs: Parameters<Session["run"]>[0],
  model = openedModel(),
): Promise<Tensor> => {
  const session = await createSessionFromContainer(gpu, await model, "model");
  try {
    const outputs = await session.run(inputs);
    assertEquals(
      session.diagnostics().planBacking,
      { residentBytes: 0, inputBytes: 0, retainedCount: 0, buildCount: 0 },
      "a one-off run pays no slot memory at all",
    );
    return outputs["y"];
  } finally {
    await session.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the output of a backed run is bit-identical to a non-backed execution (stale-slot check; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      // Run 1 = miss (arena path) / run 2 = backing construction / run 3 = backed fast path.
      const phases = [0, 1, 2];
      const actual: Tensor[] = [];
      for (const phase of phases) actual.push((await session.run({ x: input(4, phase) }))["y"]);
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });
      assertEquals(session.diagnostics().planBacking.buildCount, 1, "built only once");

      // Vacuity check: if the output stays the same when the input changes, a leftover stale slot would still be green.
      assert(
        JSON.stringify(bits(actual[0])) !== JSON.stringify(bits(actual[1])),
        "the output does not change per phase (the detector would test nothing)",
      );

      for (const phase of phases) {
        const expected = await runFresh(gpu, { x: input(4, phase) });
        assertEquals(actual[phase].shape, expected.shape);
        assertEquals(bits(actual[phase]), bits(expected), `phase ${phase} bit identity`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the arena allocations of a backed run are only the readback staging (fixed-input check; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      await session.run({ x: input(4, 0) });
      const miss = session.diagnostics().lastRun;
      await session.run({ x: input(4, 1) });
      const built = session.diagnostics().lastRun;
      await session.run({ x: input(4, 2) });
      const backed = session.diagnostics().lastRun;
      assert(miss !== undefined && built !== undefined && backed !== undefined);

      // The graph has 1 output, so 1 staging buffer. If even one input upload remained in the arena, this would be
      // 2 or more (the intermediates have already moved to resident slots).
      assertEquals(
        backed.allocCount,
        1,
        "a backed run allocates only the 1 readback staging buffer",
      );
      assertEquals(built.allocCount, 1, "the same for the first hit (construction) run");
      assertEquals(backed.transientBytes, 0, "intermediates do not go through the arena");
      assert(
        miss.allocCount > backed.allocCount,
        `the allocations did not decrease from the miss run (${miss.allocCount} → ${backed.allocCount})`,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

describe("the release wait before building a backing", () => {
  // Dawn returns a destroyed buffer's VRAM only when a later submit completes, so a run that
  // allocates the slot regions right after the previous run's arena was destroyed holds both at once
  // (Wan2.2 121 frames went out of memory there on 12 GiB). The detector is the order of the calls
  // the device sees: the building run must issue an empty submit and see it complete before it
  // allocates the first region.
  type Event = "storage-buffer" | "empty-submit" | "settled";

  const recordDevice = (device: GPUDevice): { readonly events: Event[]; restore(): void } => {
    const events: Event[] = [];
    const createBuffer = device.createBuffer.bind(device);
    const submit = device.queue.submit.bind(device.queue);
    const onSubmittedWorkDone = device.queue.onSubmittedWorkDone.bind(device.queue);
    device.createBuffer = (descriptor) => {
      if ((descriptor.usage & BUFFER_USAGE.STORAGE) !== 0) events.push("storage-buffer");
      return createBuffer(descriptor);
    };
    device.queue.submit = (buffers) => {
      if ([...buffers].length === 0) events.push("empty-submit");
      submit(buffers);
    };
    device.queue.onSubmittedWorkDone = () =>
      onSubmittedWorkDone().then(() => {
        events.push("settled");
        return undefined;
      });
    return {
      events,
      restore: () => {
        device.createBuffer = createBuffer;
        device.queue.submit = submit;
        device.queue.onSubmittedWorkDone = onSubmittedWorkDone;
      },
    };
  };

  it.skipIf(!GPU_AVAILABLE)(
    "settles an empty submit before the first region of a new backing is allocated",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      try {
        await session.run({ x: input(4, 0) });
        const recorder = recordDevice(gpu.device);
        try {
          await session.run({ x: input(4, 1) });
        } finally {
          recorder.restore();
        }
        assertEquals(session.diagnostics().planBacking.buildCount, 1, "this run built the backing");
        const firstRegion = recorder.events.indexOf("storage-buffer");
        assert(firstRegion >= 0, `no region was allocated: ${recorder.events.join(", ")}`);
        assertEquals(
          recorder.events.slice(0, firstRegion).slice(-2),
          ["empty-submit", "settled"],
          `the empty submit did not settle before the regions: ${recorder.events.join(", ")}`,
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "adds no empty submit to a miss run or to a run on a retained backing",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      try {
        const missRun = recordDevice(gpu.device);
        try {
          await session.run({ x: input(4, 0) });
        } finally {
          missRun.restore();
        }
        await session.run({ x: input(4, 1) });
        const backedRun = recordDevice(gpu.device);
        try {
          await session.run({ x: input(4, 2) });
        } finally {
          backedRun.restore();
        }
        assertEquals(session.diagnostics().planBacking.buildCount, 1, "the backing was retained");
        assert(!missRun.events.includes("empty-submit"), missRun.events.join(", "));
        assert(!backedRun.events.includes("empty-submit"), backedRun.events.join(", "));
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

test.skipIf(!GPU_AVAILABLE)(
  "concurrent runs of the same signature do not mix up even with a shared input buffer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      // The inputs of a backed run are overwritten into **the one** buffer the backing owns. Issuing them
      // concurrently without await makes the test fail if the serialization (#chain) and "a run returns only after
      // its flush completes" break — then a later run's writeBuffer would overtake an earlier run's unsubmitted
      // dispatch, and the values of adjacent phases would mix without an exception.
      const phases = [0, 1, 2, 3, 4];
      const actual = await Promise.all(phases.map((phase) => session.run({ x: input(4, phase) })));
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      for (const phase of phases) {
        const expected = await runFresh(gpu, { x: input(4, phase) });
        assertEquals(bits(actual[phase]["y"]), bits(expected), `phase ${phase} bit identity`);
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** A shape whose graph output aliases an input (reshape) + one real dispatch. */
const ALIAS_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["reshape", "sigmoid"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [4, 3] }],
  outputs: ["y", "s"],
  initializers: {},
  values: {
    y: { dtype: "f32", shape: [12] },
    s: { dtype: "f32", shape: [4, 3] },
  },
  nodes: [
    { op: "reshape", ins: ["x"], outs: ["y"], attrs: {} },
    { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
  ],
};

test.skipIf(!GPU_AVAILABLE)(
  "a backed run can read back even when the graph output aliases an input (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const opened = await openGraphModel(ALIAS_GRAPH);
    const session = await createSessionFromContainer(gpu, opened, "model");
    try {
      for (const phase of [0, 1, 2]) {
        const x = input(4, phase);
        const outputs = await session.run({ x });
        // An aliased output reads back the input buffer itself. The values changing per phase is the detector for
        // "the resident input buffer is written on every run".
        assertEquals(outputs["y"].shape, [12]);
        assertEquals(bits(outputs["y"]), bits(x), `phase ${phase} aliased output`);

        const reference = await createSessionFromContainer(gpu, opened, "model");
        try {
          assertEquals(bits(outputs["s"]), bits((await reference.run({ x }))["s"]));
        } finally {
          await reference.dispose();
        }
      }
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** Measurements of one backing (regions = `residentBytes` / owned input buffers = `inputBytes`). */
type Measured = { readonly residentBytes: number; readonly inputBytes: number };

/**
 * The amount the budget counts (the same formula as `#evictBackingsFor` — both the regions and the owned input
 * buffers).
 *
 * MUST: the budget checks are built from this. Building them from the regions alone makes the real count exceed
 * the budget by the input buffers, so a Session meant to be "exactly at budget" runs on the over-budget side
 * (= always retiring everything) from the start.
 */
const accountedBytes = (measured: readonly Measured[]): number =>
  measured.reduce((total, one) => total + one.residentBytes + one.inputBytes, 0);

/** The diagnostics when the retained set is `retained` (**every field** — no retreat to a partial match). */
const expectStats = (retained: readonly Measured[], buildCount: number): PlanBackingStats => ({
  residentBytes: retained.reduce((total, one) => total + one.residentBytes, 0),
  inputBytes: retained.reduce((total, one) => total + one.inputBytes, 0),
  retainedCount: retained.length,
  buildCount,
});

/**
 * Measures the bytes the backing of one signature (`T = rows`) keeps resident, on **a separate Session that ran
 * only that shape**.
 *
 * The budget checks (the 3 below) are built from this measurement. Written as a constant, a change to how the slot
 * table is packed (ADR 0093) would silently shift what "exactly at budget" means, and checks meant to look at the
 * LRU order would turn into a "everything fits" or a "nothing fits" shape.
 */
const backingBytes = async (
  gpu: Awaited<ReturnType<typeof acquireGpu>>,
  rows: number,
): Promise<Measured> => {
  const session = await createSessionFromContainer(gpu, await openedModel(), "model");
  try {
    // Run 1 = miss (plan derivation) / run 2 = hit (backing construction).
    await session.run({ x: input(rows) });
    await session.run({ x: input(rows) });
    const stats = session.diagnostics().planBacking;
    assertEquals(
      stats.retainedCount,
      1,
      `the Session that ran only the T=${rows} shape does not retain exactly 1`,
    );
    assert(
      stats.residentBytes > 0,
      `the T=${rows} backing is 0 bytes (the check would test nothing)`,
    );
    // The buffers of normal inputs (non-resident `Tensor`) are owned by the backing = counted by the budget. If this
    // is 0, "inputs are not counted", so the budget checks below would have been built from the regions alone.
    assert(
      stats.inputBytes > 0,
      `the owned input buffers of T=${rows} are 0 bytes (the budget does not count them)`,
    );
    return { residentBytes: stats.residentBytes, inputBytes: stats.inputBytes };
  } finally {
    await session.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "2 signatures within the budget stay retained and are not rebuilt even when alternated (default budget; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);
      assert(
        accountedBytes([wide]) > accountedBytes([narrow]),
        `the T=9 backing is not larger than T=4 (${JSON.stringify(narrow)} / ${JSON.stringify(
          wide,
        )})`,
      );

      // Default budget (256 MiB) — both shapes are tens of bytes, so both fit.
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      try {
        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([], 0),
          "a miss run does not build a backing",
        );

        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "the first hit makes 1 slot set resident",
        );

        // Another signature is not built on a miss run; it is **added** only on a hit (not a replacement).
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "unchanged on a miss run of another signature",
        );
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow, wide], 2),
          "the second retired the old backing (the total is not the sum of the separate measurements)",
        );

        // Not rebuilding across 3 round trips = the observation point that the per-switch rebuild (≈ 40 ms of
        // perf-ledger H-15) is gone. The values are bit-identical to a non-backed execution too (if a retained set
        // held buffers of a different shape, this would break without an exception).
        for (const phase of [1, 2, 3]) {
          const narrowOut = (await session.run({ x: input(4, phase) }))["y"];
          const wideOut = (await session.run({ x: input(9, phase) }))["y"];
          assertEquals(
            session.diagnostics().planBacking,
            expectStats([narrow, wide], 2),
            `phase ${phase}: the backing is rebuilt on a switch`,
          );
          assertEquals(bits(narrowOut), bits(await runFresh(gpu, { x: input(4, phase) })));
          assertEquals(bits(wideOut), bits(await runFresh(gpu, { x: input(9, phase) })));
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a Session with budget 0 rebuilds on every switch (reproducing the former capacity of 1; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);

      const session = await createSessionFromContainer(gpu, await openedModel(), "model", {
        planBackingBudgetBytes: 0,
      });
      try {
        await session.run({ x: input(4) });
        await session.run({ x: input(4) });
        assertEquals(session.diagnostics().planBacking, expectStats([narrow], 1));

        // With budget 0 a new backing is always over budget, so everything retained is retired.
        await session.run({ x: input(9) });
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([wide], 2),
          "the old backing remains at budget 0 (the capacity-1 behaviour is not reproduced)",
        );

        // The plan stays in the LRU, so the run that switches back hits and **rebuilds**.
        const narrowOut = (await session.run({ x: input(4, 1) }))["y"];
        assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 3),
          "no rebuild on a switch (budget 0 is not in effect)",
        );
        assertEquals(bits(narrowOut), bits(await runFresh(gpu, { x: input(4, 1) })));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a shape that exceeds the budget on its own retires everything retained and keeps only itself (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const narrow = await backingBytes(gpu, 4);
      const wide = await backingBytes(gpu, 9);
      // A budget that any single backing exceeds on its own (= residency is only the "largest one" side of
      // `max(budget, largest one)` (ADR 0095 decision 1)). The count is regions + owned inputs, so subtract from that sum.
      const budget = accountedBytes([narrow]) - 1;
      assert(
        budget >= 0,
        `the measurement of narrow ${JSON.stringify(narrow)} is too small to make a budget`,
      );

      const session = await createSessionFromContainer(gpu, await openedModel(), "model", {
        planBackingBudgetBytes: budget,
      });
      try {
        await session.run({ x: input(4) });
        await session.run({ x: input(4) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([narrow], 1),
          "the first backing that exceeds the budget was not allocated (the backed path is gone)",
        );

        await session.run({ x: input(9) });
        await session.run({ x: input(9) });
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([wide], 2),
          "a new over-budget backing leaves the old one in place (residency exceeds max(budget, largest one))",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "when a backed plan is evicted from the LRU its backing is released too, and later runs complete (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openedModel(), "model");
    try {
      await session.run({ x: input(4) });
      await session.run({ x: input(4) });
      assert(session.diagnostics().planBacking.residentBytes > 0);

      // T=4 is the oldest. Running **exactly the limit** of other shapes evicts only T=4 (the T=5 hit is checked
      // below, so evicting more would turn the check red for another reason). The count comes from the constant —
      // a copy would stop evicting when the limit moves, and this check would test nothing.
      for (let rows = 5; rows < 5 + PREPARED_PLAN_CAPACITY; rows += 1) {
        await session.run({ x: input(rows) });
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([], 1),
        "the backing of an evicted plan is released (no rebuild happened)",
      );

      // After eviction + destruction, hit runs of the remaining plans still complete correctly.
      const hit = (await session.run({ x: input(5) }))["y"];
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 2);
      assertEquals(bits(hit), bits(await runFresh(gpu, { x: input(5) })));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * attention (2 node-local temporaries) + reshape (alias, graph output). One graph covers the 3 paths of slot
 * derivation (output allocation / nested lifetimes of temporaries / pinning through an alias).
 */
const HALF_SCALE = Math.fround(Math.sqrt(1 / Math.sqrt(4)));

const TEMP_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "reshape"] },
  symbols: ["M"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 1, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 1, 4, 4] },
    { name: "v", dtype: "f32", shape: [1, 1, 4, 4] },
  ],
  outputs: ["y"],
  initializers: {},
  values: {
    a: { dtype: "f32", shape: [1, 1, "M", 4] },
    y: { dtype: "f32", shape: ["M", 4] },
  },
  nodes: [
    { op: "attention", ins: ["q", "k", "v"], outs: ["a"], attrs: { scale: HALF_SCALE } },
    { op: "reshape", ins: ["a"], outs: ["y"], attrs: {} },
  ],
};

const ROWS = 6;
const tempInputs = (phase: number) => ({
  q: fill([1, 1, ROWS, 4], (i) => (((i + phase) % 7) - 3) * 0.25),
  k: fill([1, 1, 4, 4], (i) => ((i % 5) - 2) * 0.5),
  v: fill([1, 1, 4, 4], (i) => ((i % 3) - 1) * 0.75),
});

/** The size the arena hands out for one buffer (minimum 4 bytes + 4-byte alignment). */
const bufferBytes = (count: number): number => Math.max(4, count * 4);

test.skipIf(!GPU_AVAILABLE)(
  "the total slot bytes match the pool allocation of a non-backed run (footprint-unchanged check; real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // Uploads of the 3 graph inputs (q: [1,1,ROWS,4] / k, v: [1,1,4,4]). In a backed run the backing owns buffers
    // of the same size = also the expected value of `inputBytes` in the diagnostics.
    const inputHostBytes = bufferBytes(ROWS * 4) + bufferBytes(16) * 2;
    // The part outside the pool (input uploads and readback staging). The arena's allocatedBytes minus this is "the
    // real allocation of the output storage the dispatches write" = the total bytes of the slot table.
    const hostBytes = inputHostBytes + bufferBytes(ROWS * 4);
    const reference = await createSessionFromContainer(
      gpu,
      await openGraphModel(TEMP_GRAPH),
      "model",
    );
    let pooledBytes = 0;
    try {
      await reference.run(tempInputs(0));
      const stats = reference.diagnostics().lastRun;
      assert(stats !== undefined, "no arena record for run 1");
      pooledBytes = stats.allocatedBytes - hostBytes;
      assert(
        pooledBytes > 0,
        `the pool allocation is 0 (the check would test nothing): ${stats.allocatedBytes}`,
      );
    } finally {
      await reference.dispose();
    }

    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(TEMP_GRAPH),
      "model",
    );
    try {
      const first = (await session.run(tempInputs(1)))["y"];
      const second = (await session.run(tempInputs(1)))["y"];
      // MUST: measure with exactly 1 backing retained (`residentBytes` in the diagnostics is the sum over the retained
      // set, so measuring on a Session that ran 2 shapes would change the very claim "matches one run's pool allocation").
      assertEquals(
        session.diagnostics().planBacking.retainedCount,
        1,
        "footprint-unchanged is a claim about one backing (measured with other than 1 retained)",
      );
      assertEquals(
        session.diagnostics().planBacking.residentBytes,
        pooledBytes,
        "the slot table disagrees with the current pool allocation (the VRAM premise breaks)",
      );
      // The same claim on the input side — the input buffers the backing owns are exactly the upload targets the
      // arena path allocated per run (the budget counts this sum, so if this drifts, the budget's meaning departs
      // from the real VRAM).
      assertEquals(
        session.diagnostics().planBacking.inputBytes,
        inputHostBytes,
        "the backing's owned inputs disagree with the upload targets of the arena path",
      );
      // Even in a shape with temporaries and aliases the backed output matches the non-backed one (pinning through an alias is checked too).
      assertEquals(bits(second), bits(first));
      assertEquals(second.shape, [ROWS, 4]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** Record of one storage buffer (one element per allocation in order = the destroy count of that buffer). */
type BufferRecord = { readonly destroyCounts: number[] };

/**
 * Injection that fails `#activateBacking` in the middle of its allocations.
 *
 * A real GPU can neither "fail only the N-th createBuffer" nor tell "whether an allocated buffer was released
 * exactly once", so the device's `createBuffer` is wrapped (a substitute for opening an injection surface in
 * production code — the same shape as `injectFaults` in `gpu_generation_context_test.ts`).
 * MUST: the returned objects stay **real GPUBuffers** (only `destroy` is wrapped). Replacing them would make the
 * objects passed to bind group creation the test's own fakes, and the check would stop testing the implementation.
 */
const injectStorageFault = (
  device: GPUDevice,
): {
  /** Puts later storage allocations on a new record (synchronous throw at allocation `failAt`). */
  readonly record: (failAt?: number) => BufferRecord;
  readonly restore: () => void;
} => {
  const original = device.createBuffer.bind(device);
  let current: BufferRecord = { destroyCounts: [] };
  let creations = 0;
  let failAt: number | undefined;
  device.createBuffer = ((descriptor: GPUBufferDescriptor): GPUBuffer => {
    // Only slot / input storage is counted (the readback staging takes another path with MAP_READ).
    if ((descriptor.usage & BUFFER_USAGE.STORAGE) === 0) return original(descriptor);
    creations += 1;
    if (creations === failAt) {
      throw new Error(`injected: synchronous throw from storage createBuffer #${creations}`);
    }
    const buffer = original(descriptor);
    const counts = current.destroyCounts;
    const index = counts.push(0) - 1;
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = (): undefined => {
      counts[index] += 1;
      return destroy();
    };
    return buffer;
  }) as typeof device.createBuffer;
  return {
    record: (next?: number): BufferRecord => {
      creations = 0;
      failAt = next;
      current = { destroyCounts: [] };
      return current;
    },
    restore: (): void => {
      device.createBuffer = original;
    },
  };
};

// `#activateBacking` in executor.ts wraps "allocation → registration in the retained set (`#backings`)
// (establishing ownership)" in a try/catch. A buffer leaked in this window is reachable from neither
// `#retireBacking()` nor `dispose()`, and its size is the largest in this Session (the total bytes of the slot
// table). A check of the same shape exists on the `GenerationContext.create` side but was missing on the executor side.
//
// The budget decides only **what happens to the old backing** (ADR 0095 decision 2 — the failure path retires
// "the one this run newly built"): within the budget the old backing stays retained; at budget 0 the retirement
// before allocation has already taken effect, so the old backing is released too. That the buffers leaked in the
// middle of allocation are released exactly once is the same under both budgets.
const assertPartialBackingFault = async (
  planBackingBudgetBytes: number | undefined,
  expected: {
    /** Whether the old backing (T=4) stays retained after the failed run. */
    readonly retainsOld: boolean;
  },
): Promise<void> => {
  const gpu = await acquireGpu();
  const session = await createSessionFromContainer(
    gpu,
    await openedModel(),
    "model",
    planBackingBudgetBytes === undefined ? {} : { planBackingBudgetBytes },
  );
  // Resident inputs (the side whose baked references are retained / released). The backing does not own input
  // buffers, so the storage record is the slot table itself.
  const narrow = await gpu.createResident(4 * 3 * 4, "narrow");
  const wide = await gpu.createResident(9 * 3 * 4, "wide");
  const fault = injectStorageFault(gpu.device);
  try {
    narrow.write(input(4, 0).data);
    wide.write(input(9, 0).data);

    // The miss run's (arena path) allocations go to a separate record. From here on only backing construction is counted.
    // MUST: bind the symbol T explicitly (resident inputs have no shape, so they cannot be a binding source).
    fault.record();
    await session.run({ x: narrow }, { T: 4 });

    // The T=4 hit run = backing construction. The storage allocations in this window are the slot table itself
    // (resident inputs are not owned by the backing, and the readback staging takes another path with MAP_READ).
    const built = fault.record();
    await session.run({ x: narrow }, { T: 4 });
    assertEquals(session.diagnostics().planBacking.buildCount, 1);
    const old: Measured = {
      residentBytes: session.diagnostics().planBacking.residentBytes,
      inputBytes: session.diagnostics().planBacking.inputBytes,
    };
    // Resident inputs are owned by the GpuContext = not by the backing. If this stops being 0, the premise below that
    // "the storage record is the slot table itself" (= how the buffers are counted) breaks.
    assertEquals(old.inputBytes, 0, "resident inputs are counted as owned by the backing");
    assertEquals(
      narrow.bakedReferences,
      1,
      "no baked reference is held (the check would test nothing)",
    );
    const slots = built.destroyCounts.length;
    assert(slots >= 2, `only ${slots} slots, so "fail in the middle" cannot happen`);

    // T=9 builds no backing on a miss run. The next hit run is the construction window.
    fault.record();
    await session.run({ x: wide }, { T: 9 });

    const partial = fault.record(2);
    await assertRejects(
      () => session.run({ x: wide }, { T: 9 }),
      Error,
      "injected: synchronous throw from storage createBuffer #2",
    );

    assertEquals(partial.destroyCounts, [1], "the 1 allocated buffer is released exactly once");
    assertEquals(
      built.destroyCounts,
      Array(slots).fill(expected.retainsOld ? 0 : 1),
      expected.retainsOld
        ? "an unrelated failure retired the old backing (the entrance to rebuild thrashing)"
        : "the retired old backing is also released exactly once each",
    );
    assertEquals(
      session.diagnostics().planBacking,
      expected.retainsOld ? expectStats([old], 1) : expectStats([], 1),
      "the failed construction sits in place as a backing",
    );
    assertEquals(
      narrow.bakedReferences,
      expected.retainsOld ? 1 : 0,
      expected.retainsOld
        ? "the baked references of the still-retained backing were released"
        : "retirement did not release the baked references",
    );
    assertEquals(wide.bakedReferences, 0, "the failed construction leaves baked references behind");

    // The next run of the same signature rebuilds, completes, and is bit-identical to a non-backed execution.
    fault.restore();
    const actual = (await session.run({ x: wide }, { T: 9 }))["y"];
    assertEquals(session.diagnostics().planBacking.buildCount, 2, "not rebuilt");
    assertEquals(bits(actual), bits(await runFresh(gpu, { x: input(9, 0) })));
  } finally {
    fault.restore();
    await session.dispose();
    narrow.dispose();
    wide.dispose();
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "a failure in the middle of backing allocation releases only what was allocated, once, and keeps the old backing (default budget; fault injection; real GPU)",
  () => assertPartialBackingFault(undefined, { retainsOld: true }),
);

test.skipIf(!GPU_AVAILABLE)(
  "at budget 0 the old backing is retired before allocation and released exactly once along with what was allocated (fault injection; real GPU)",
  () => assertPartialBackingFault(0, { retainsOld: false }),
);

/**
 * Puts a third signature into a Session exactly at budget and checks that the one retired is **the least recently
 * used** (not the one touched just before).
 *
 * `touched` is the one touched again before the third (= moved to the most recent end of the LRU). Both variants
 * are kept because with only one of them an implementation that "always retires the one built first" (insertion
 * order only, not moved on a hit) would stay green.
 */
const assertLruOrder = async (touched: 4 | 9): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    const narrow = await backingBytes(gpu, 4);
    const wide = await backingBytes(gpu, 9);
    const small = await backingBytes(gpu, 2);
    // Unless the third is of a size that "always fits after retiring one", retirement would extend to 2 and the LRU
    // order could not be observed (the check would test nothing). What is compared is what the budget counts (regions + owned inputs).
    assert(
      accountedBytes([small]) <= accountedBytes([narrow]) &&
        accountedBytes([small]) <= accountedBytes([wide]),
      `the measurement of T=2 ${JSON.stringify(small)} is not at most the other shapes` +
        `（${JSON.stringify(narrow)} / ${JSON.stringify(wide)}）`,
    );

    const session = await createSessionFromContainer(gpu, await openedModel(), "model", {
      planBackingBudgetBytes: accountedBytes([narrow, wide]),
    });
    const fault = injectStorageFault(gpu.device);
    try {
      // The miss runs go to a discarded record, and only the construction windows are counted per shape.
      fault.record();
      await session.run({ x: input(4) });
      const narrowRecord = fault.record();
      await session.run({ x: input(4) });
      fault.record();
      await session.run({ x: input(9) });
      const wideRecord = fault.record();
      await session.run({ x: input(9) });
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([narrow, wide], 2),
        "the 2 backings exactly at budget are not both retained",
      );

      // Touch again (move to the most recent end of the LRU). It is a hit, so no rebuild happens.
      await session.run({ x: input(touched) });
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        2,
        "touching again rebuilds the backing",
      );

      // The third. Exactly at budget, so only the least recently used one is retired.
      fault.record();
      await session.run({ x: input(2) });
      const smallRecord = fault.record();
      await session.run({ x: input(2) });

      const kept =
        touched === 4
          ? { record: narrowRecord, measured: narrow, rows: 4 }
          : { record: wideRecord, measured: wide, rows: 9 };
      const evicted =
        touched === 4 ? { record: wideRecord, rows: 9 } : { record: narrowRecord, rows: 4 };
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([kept.measured, small], 3),
        `the one other than T=${touched}, touched just before, was not retired (the LRU order stays the insertion order)`,
      );
      assert(
        evicted.record.destroyCounts.length > 0,
        "the record of the retired side is empty (the check would test nothing)",
      );
      assertEquals(
        evicted.record.destroyCounts,
        Array(evicted.record.destroyCounts.length).fill(1),
        `the buffers of the retired T=${evicted.rows} were not released exactly once each`,
      );
      assertEquals(
        kept.record.destroyCounts,
        Array(kept.record.destroyCounts.length).fill(0),
        `the buffers of the retained T=${kept.rows} were released (destroying buffers in use)`,
      );
      assertEquals(
        smallRecord.destroyCounts,
        Array(smallRecord.destroyCounts.length).fill(0),
        "the newly allocated backing was released on the spot",
      );

      // Retained shapes run without rebuilding, and the values match a non-backed execution.
      const hit = (await session.run({ x: input(kept.rows, 1) }))["y"];
      assertEquals(session.diagnostics().planBacking.buildCount, 3, "a retained shape was rebuilt");
      assertEquals(bits(hit), bits(await runFresh(gpu, { x: input(kept.rows, 1) })));
    } finally {
      fault.restore();
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "over-budget retirement takes the least recently used one (T=4, touched just before, remains; real GPU)",
  () => assertLruOrder(4),
);

test.skipIf(!GPU_AVAILABLE)(
  "over-budget retirement takes the least recently used one (T=9, touched just before, remains; real GPU)",
  () => assertLruOrder(9),
);

// Failure settlement of a batch (the batch version of ADR 0095 decision 2). A non-fatal OOM (`createBuffer` merely
// returns an invalid buffer without throwing) lets the enqueue succeed and is discovered only in the errorScope of
// `finish()`. Unless the settlement retires it, the broken backing stays in the retained set, and the next batch /
// run of the same shape keeps reusing it without rebuilding even after memory frees up.
//
// Plan derivation is done beforehand by a preliminary run (the params created by enqueue's derivation use storage
// too, and mixing them into the same record would break the count of "the backing's buffers are released exactly once").
describe("when finish of a batch discovers a GPU failure (non-fatal OOM) (injected; real GPU)", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "the backing that batch newly built is retired and released exactly once, and the next batch rebuilds the same shape",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const storage = injectStorageFault(gpu.device);
      const oom = injectOutOfMemory(gpu.device);
      try {
        storage.record();
        await session.run({ x: input(4, 0) });

        const built = storage.record();
        const failed = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 0) }, { batch: failed });
        assertEquals(
          session.diagnostics().planBacking.buildCount,
          1,
          "enqueue did not build a backing",
        );
        oom.arm();
        await assertRejects(() => failed.finish(), GpuOutOfMemoryError, "injected: non-fatal OOM");
        assertEquals(
          oom.fired(),
          1,
          "the injection did not hit finish's pop (the check would test nothing)",
        );
        assertEquals(
          session.diagnostics().planBacking,
          expectStats([], 1),
          "the backing of the batch that settled with a failure remains in the retained set",
        );
        // The old backing's VRAM is released **before** the rebuild (destruction is queued on the Session's
        // serialization chain and does not ride on the cleanup of a later enqueue). To advance the chain by exactly
        // one step, a run that fails at binding resolution is sent — it runs on the chain but fails without touching
        // either the pending destruction or the backing.
        await assertRejects(
          () => session.run({ x: { dtype: "f32", shape: [4, 5], data: new Float32Array(20) } }),
          ExecutionError,
        );
        assert(
          built.destroyCounts.length > 0,
          "the construction record is empty (the check would test nothing)",
        );
        assertEquals(
          built.destroyCounts,
          Array(built.destroyCounts.length).fill(1),
          "the backing of the batch that settled with a failure was not released at the first step of the serialization chain",
        );

        storage.record();
        const retry = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 1) }, { batch: retry });
        await retry.finish();
        assertEquals(
          session.diagnostics().planBacking.buildCount,
          2,
          "reuses the failed batch's backing without rebuilding",
        );
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "discards even plans unrelated to the failed batch (registered by an earlier run)",
        );
        assert(
          built.destroyCounts.length > 0,
          "the construction record is empty (the check would test nothing)",
        );
        assertEquals(
          built.destroyCounts,
          Array(built.destroyCounts.length).fill(1),
          "the buffers of the retired backing were not released exactly once each",
        );

        // A hit run on the rebuilt backing is bit-identical to a non-backed execution.
        const actual = (await session.run({ x: input(4, 2) }))["y"];
        assertEquals(session.diagnostics().planBacking.buildCount, 2);
        assertEquals(bits(actual), bits(await runFresh(gpu, { x: input(4, 2) })));
      } finally {
        oom.restore();
        storage.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "a failure of a batch that merely hit an existing backing does not retire it (no rebuild thrashing)",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const storage = injectStorageFault(gpu.device);
      const oom = injectOutOfMemory(gpu.device);
      try {
        storage.record();
        await session.run({ x: input(4, 0) });

        const built = storage.record();
        const first = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 0) }, { batch: first });
        await first.finish();
        const retained = session.diagnostics().planBacking;
        assertEquals(retained.buildCount, 1);

        storage.record();
        const failed = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 1) }, { batch: failed });
        oom.arm();
        await assertRejects(() => failed.finish(), GpuOutOfMemoryError, "injected: non-fatal OOM");
        assertEquals(
          oom.fired(),
          1,
          "the injection did not hit finish's pop (the check would test nothing)",
        );

        const next = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 2) }, { batch: next });
        await next.finish();
        assertEquals(
          session.diagnostics().planBacking,
          retained,
          "an unrelated failure retires an existing backing",
        );
        assert(
          built.destroyCounts.length > 0,
          "the construction record is empty (the check would test nothing)",
        );
        assertEquals(
          built.destroyCounts,
          Array(built.destroyCounts.length).fill(0),
          "destroys the buffers of a retained backing",
        );
      } finally {
        oom.restore();
        storage.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});

// A batch whose scope body (the errorScope check) passed and that settled only on the host-side failure (input
// error) of one enqueue. The backing another enqueue of the same scope built is valid, so the settlement does not
// retire it — retiring would rebuild the backing on every unrelated input error (the shape ADR 0095 decision 2 avoids).
describe("when a batch settles as failed on a host-side failure only (real GPU)", () => {
  it.skipIf(!GPU_AVAILABLE)(
    "keeps the backing newly built by another enqueue of the same scope retained",
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openedModel(), "model");
      const storage = injectStorageFault(gpu.device);
      try {
        storage.record();
        await session.run({ x: input(4, 0) });

        const built = storage.record();
        const batch = await gpu.beginBatch();
        await session.enqueue({ x: input(4, 0) }, { batch });
        // A wrong element count passes binding resolution and fails at the input check after derivation (before backing construction).
        const wrongCount: Tensor = { dtype: "f32", shape: [4, 3], data: new Float32Array(11) };
        await assertRejects(() => session.enqueue({ x: wrongCount }, { batch }), ExecutionError);
        await assertRejects(() => batch.finish(), ExecutionError);

        const stats = session.diagnostics().planBacking;
        assertEquals(
          [stats.retainedCount, stats.buildCount],
          [1, 1],
          "a settlement on a host-side failure alone retires the backing another enqueue built",
        );
        assert(
          built.destroyCounts.length > 0,
          "the construction record is empty (the check would test nothing)",
        );
        assertEquals(
          built.destroyCounts,
          Array(built.destroyCounts.length).fill(0),
          "destroys the buffers of a retained backing",
        );
      } finally {
        storage.restore();
        await session.dispose();
        gpu.destroy();
      }
    },
  );
});
