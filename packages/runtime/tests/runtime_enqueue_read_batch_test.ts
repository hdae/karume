// The `options.batch` that `Session.enqueue` / `Session.enqueueRead` receive is **a property of the
// caller's object**, so a getter can be planted there. Passing a getter that returns a different
// value on each read makes the scope the entry check saw, the scope the in-flight lease was taken
// on and the scope remembered in `#readBatch` diverge, so that
//
// ① the check "a Session with an unsettled run cannot be enqueued to" is bypassed and it
//    self-deadlocks
// ② a throw from the getter leaks as a **synchronous throw** instead of a rejection of the returned
//    Promise
// ③ `enqueueRead`'s check "reject later enqueues until settled" applies to another scope, and the
//    slot being read is overwritten by a later enqueue (a silent wrong value)
//
// happen. What this pins is "`options.batch` is read once in the synchronous part of issuing, and
// only that value is used afterwards" — the three above follow from it.
//
// No real GPU is used. The check and the lease acquisition happen in a synchronous part that issues
// no dispatch at all, so a fake device (`helpers/fake-gpu.ts`) pins them deterministically.

import { assertEquals, assertInstanceOf, assertNotEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { IrGraph } from "../src/format/ir.ts";
import {
  type BatchScope,
  BatchScopeError,
  GpuContext,
  readAdapterInfo,
  type RequiredLimits,
} from "../src/gpu/device.ts";
import { Session } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

// ---------------------------------------------------------------------------
// Fixtures (empty graph, fake device, observing settlement)
// ---------------------------------------------------------------------------

/**
 * A graph with 0 nodes. The entry check does not look at a single byte of what is enqueued, so this
 * is enough.
 */
const emptyGraph = (): IrGraph =>
  mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: [],
    inputs: [],
    outputs: [],
    initializers: {},
    values: {},
    nodes: [],
  });

/**
 * Any limits that do not trip a check will do (what is checked is the entry point, not allocation
 * sizes).
 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 1 << 20,
  maxStorageBufferBindingSize: 1 << 20,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

/**
 * A device that adds `limits` to the minimal fake in `helpers/fake-gpu.ts`, because the build phase
 * (`buildSessionState`) reads `minStorageBufferOffsetAlignment` once.
 */
const deviceWithLimits = (): GPUDevice =>
  Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } });

const gpuContext = (): GpuContext =>
  new GpuContext(deviceWithLimits(), readAdapterInfo({}), LIMITS, new Set());

const emptySession = async (gpu: GpuContext): Promise<Session> => {
  async function* noShards(): AsyncGenerator<never> {}
  return await Session.build(gpu, emptyGraph(), new Map(), noShards(), {});
};

/**
 * Observes settlement (`"resolved"` / `"rejected"` / `"pending"`).
 *
 * A regression leaves the target **pending forever** (a self-deadlock raises no exception and no
 * diagnostic), so this gives up instead of waiting it out, letting a test assert "it stayed
 * pending".
 */
const outcomeOf = async (
  awaited: Promise<unknown>,
): Promise<{ readonly state: string; readonly cause?: unknown }> => {
  const cutoff = Promise.withResolvers<{ readonly state: string }>();
  const timer = setTimeout(() => cutoff.resolve({ state: "pending" }), 500);
  try {
    return await Promise.race([
      awaited.then(
        () => ({ state: "resolved" }),
        (cause) => ({ state: "rejected", cause }),
      ),
      cutoff.promise,
    ]);
  } finally {
    clearTimeout(timer);
  }
};

const rejectionOf = async (awaited: Promise<unknown>): Promise<unknown> => {
  const outcome = await outcomeOf(awaited);
  assertEquals(outcome.state, "rejected");
  return outcome.cause;
};

// ---------------------------------------------------------------------------

describe("the enqueue entry point reads options.batch only once", () => {
  it("in the normal form without a getter it reads once, and admission, read-back and scope settlement all pass", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    let reads = 0;
    const read = session.enqueueRead(
      {},
      {
        get batch(): BatchScope {
          reads += 1;
          return batch;
        },
      },
    );
    const finished = outcomeOf(batch.finish());
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    assertEquals(await outcomeOf(read.outputs), { state: "resolved" });
    assertEquals(await finished, { state: "resolved" });
    // That it is read only once is itself why the three regressions below cannot happen.
    assertEquals(reads, 1);
    await session.dispose();
  });

  it("even with a getter that returns another scope from the second read on, the scope rejecting later enqueues stays the one entered", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    // Scopes are exclusive per device, so the substitute is opened on another GpuContext.
    const other = gpuContext();
    const entered = await gpu.beginBatch();
    const decoy = await other.beginBatch();
    let reads = 0;
    const read = session.enqueueRead(
      {},
      {
        get batch(): BatchScope {
          reads += 1;
          return reads === 1 ? entered : decoy;
        },
      },
    );
    // Observing that the entered scope is `entered`: had it entered `decoy`, admission itself would
    // fail with "cannot enqueue to a batch opened on another GpuContext".
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    // `#readBatch` is also still `entered` = later enqueues before settlement are rejected.
    const refused = await rejectionOf(session.enqueue({}, { batch: entered }));
    assertInstanceOf(refused, BatchScopeError);
    assertStringIncludes(refused.message, "A batch holding an enqueueRead");
    assertEquals(await outcomeOf(entered.finish()), { state: "resolved" });
    // After settlement it is no longer the same scope (the other side of the check being lifted).
    assertEquals(await outcomeOf(decoy.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("even with a getter that throws from the second read on, enqueueRead does not throw synchronously", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    let reads = 0;
    // If a synchronous throw leaked, this call itself would fail here (the return value could not
    // be received).
    const read = session.enqueueRead(
      {},
      {
        get batch(): BatchScope {
          reads += 1;
          if (reads >= 2) throw new Error("second read");
          return batch;
        },
      },
    );
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("a getter that throws on the first read becomes a rejection of the returned Promise, not a synchronous throw", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    const read = session.enqueueRead(
      {},
      {
        get batch(): BatchScope {
          throw new Error("first read");
        },
      },
    );
    assertInstanceOf(await rejectionOf(read.admitted), Error);
    assertInstanceOf(await rejectionOf(read.outputs), Error);
    // An enqueue that failed admission put nothing into the scope (no lease remains and finish
    // settles).
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    await session.dispose();
  });

  it("even if the getter issues a run on the same Session, admission and the scope settle", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    const issued: Promise<unknown>[] = [];
    const read = session.enqueueRead(
      {},
      {
        get batch(): BatchScope {
          if (issued.length === 0) issued.push(session.run({}));
          return batch;
        },
      },
    );
    // An enqueue on a Session with an unsettled run would self-deadlock, so the entry point turns
    // it into a typed exception.
    const refused = await rejectionOf(read.admitted);
    assertInstanceOf(refused, BatchScopeError);
    assertStringIncludes(refused.message, "unsettled runs");
    // It failed before taking the lease, so the scope settles.
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    // The run also gets as far as taking the scope lock (the body fails because the fake device has
    // no way to build a command sequence — what this pins is "it does not stay pending forever
    // waiting for the lock").
    assertEquals(issued.length, 1);
    assertNotEquals((await outcomeOf(issued[0])).state, "pending");
    await session.dispose();
  });
});

describe("Session.dispose is not accepted before a batch holding an enqueueRead has settled", () => {
  it("a dispose before settlement is rejected before intake closes, and a dispose after finish passes", async () => {
    const gpu = gpuContext();
    const session = await emptySession(gpu);
    const batch = await gpu.beginBatch();
    const read = session.enqueueRead({}, { batch });
    assertEquals(await outcomeOf(read.admitted), { state: "resolved" });
    // The read-back at settlement copies from the slot backing's raw buffer (no borrow count) —
    // letting dispose through would destroy the backing before finish's copy.
    const refused = await rejectionOf(session.dispose());
    assertInstanceOf(refused, ExecutionError);
    assertStringIncludes(refused.message, "enqueueRead readback");
    // The rejection comes before intake closes = the Session is still usable (later scopes can be
    // enqueued to).
    assertEquals(await outcomeOf(batch.finish()), { state: "resolved" });
    assertEquals(await outcomeOf(read.outputs), { state: "resolved" });
    const next = await gpu.beginBatch();
    assertEquals(await outcomeOf(session.enqueue({}, { batch: next })), { state: "resolved" });
    assertEquals(await outcomeOf(next.finish()), { state: "resolved" });
    assertEquals(await outcomeOf(session.dispose()), { state: "resolved" });
  });
});

describe("the read-back total limit of enqueueRead is checked before dispatch", () => {
  // The limit lets one read-back (12 bytes) pass and is exceeded by two (24 bytes).
  const READBACK_LIMITS: RequiredLimits = { ...LIMITS, maxBufferSize: 16 };

  /**
   * A graph that outputs input `x` (f32 × 3 = 12 bytes) as is (with 0 nodes it still carries a
   * 12-byte read-back).
   */
  const passThroughGraph = (): IrGraph =>
    mergeGraph({
      format: "karume-ir",
      version: 2,
      requires: { ops: [] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [3] }],
      outputs: ["x"],
      initializers: {},
      values: {},
      nodes: [],
    });

  /** A fake device that takes backing allocations and input writes (counting the issued writes). */
  const recordingGpu = (): { readonly gpu: GpuContext; readonly writes: () => number } => {
    let writes = 0;
    const device = fakeDevice();
    Object.assign(device, {
      limits: { minStorageBufferOffsetAlignment: 256 },
      createBuffer: (descriptor: GPUBufferDescriptor) => ({
        size: descriptor.size,
        destroy: (): void => undefined,
      }),
    });
    Object.assign(device.queue, {
      writeBuffer: (): void => {
        writes += 1;
      },
      submit: (): void => undefined,
    });
    return {
      gpu: new GpuContext(device, readAdapterInfo({}), READBACK_LIMITS, new Set()),
      writes: () => writes,
    };
  };

  const passThroughSession = async (gpu: GpuContext): Promise<Session> => {
    async function* noShards(): AsyncGenerator<never> {}
    return await Session.build(gpu, passThroughGraph(), new Map(), noShards(), {});
  };

  const x = () => ({ dtype: "f32" as const, shape: [3], data: Float32Array.of(1, 2, 3) });

  it("an enqueueRead whose registered total plus this one exceeds the limit is rejected without writing inputs or holding a backing", async () => {
    const { gpu, writes } = recordingGpu();
    const first = await passThroughSession(gpu);
    const second = await passThroughSession(gpu);
    const batch = await gpu.beginBatch();
    const accepted = first.enqueueRead({ x: x() }, { batch });
    assertEquals(await outcomeOf(accepted.admitted), { state: "resolved" });
    const writesBefore = writes();

    const refusedRead = second.enqueueRead({ x: x() }, { batch });
    const refused = await rejectionOf(refusedRead.admitted);
    assertInstanceOf(refused, BatchScopeError);
    assertStringIncludes(refused.message, "maxBufferSize");
    // If the check ran only at registration (after submit), it would fail after writing the inputs
    // and submitting the dispatch (with a generation, after writing state = poisoning the context).
    assertEquals(writes() - writesBefore, 0);
    // The failure path before dispatch retires the backing this enqueue newly built.
    assertEquals(second.diagnostics().planBacking.retainedCount, 0);

    // The scope's settlement returns the rejected enqueue's failure as the one it is attributed to
    // (the read-back does not run on the fake).
    await outcomeOf(batch.finish());
    await outcomeOf(accepted.outputs);
    await outcomeOf(refusedRead.outputs);
    await first.dispose();
    await second.dispose();
  });
});
