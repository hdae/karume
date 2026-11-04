// Regressions of the rule that the admission checks run again "after the copy" (executor.ts
// `#assertEnqueueAdmissible` / `run`, device.ts `finishAndRead`), and the handling of the measurement
// window of a batch that did not reach its fence (device.ts `#completion`).
//
// `Object.entries`, which copies the input and output Records, runs the caller's getters synchronously.
// Re-entering the same Session / batch from inside them slips past the admission checks done at call
// time. These tests build the re-entry from a getter on purpose and pin (1) that it is rejected and
// (2) that the inner operation settles correctly.
//
// The file also holds the batch scope x unsettled run self-deadlock detector cases (bottom section).

import { test as vitestTest } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import {
  acquireGpu,
  BatchScopeError,
  type GpuContext,
  type ResidentTensor,
} from "../src/gpu/device.ts";
import { GpuValidationError } from "../src/gpu/error-scope.ts";
import { DEFAULT_SUBMIT_POLICY } from "../src/gpu/submit.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { EnqueueRead, RunOutputs } from "../src/runtime/session-types.ts";
import {
  type DeclarationJson,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const input = (): Tensor => ({ dtype: "f32", shape: [3], data: Float32Array.of(1, 2, 3) });
const session = async (gpu: GpuContext) =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(singleOpDeclaration("neg", [[3]], [[3]])),
    "model",
  );
const test = (name: string, fn: () => Promise<void>): void =>
  vitestTest.skipIf(!GPU_AVAILABLE)(name, fn);

/**
 * A self-deadlock regression shows up as "never settles", so the wait is bounded (the timer is always
 * cleared so that the sanitizer sees nothing left).
 */
const settleWithin = async <T>(work: Promise<T>, label: string): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label}: did not settle within 5 seconds (self-deadlock)`)),
      5000,
    );
  });
  try {
    return await Promise.race([work, timeout]);
  } finally {
    clearTimeout(timer);
  }
};

test("batch: drops the window whose readback failed before the fence and does not update the submit estimate", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu),
    sink = await gpu.createResident(12);
  try {
    const batch = await gpu.beginBatch();
    await target.enqueue({ x0: input() }, { batch, copyOutputs: { y: sink } });
    // Put one validation error into the range's errorScope (MAP_READ combines only with COPY_DST — it
    // raises no synchronous exception and shows up only in the scope). The settling readback fails at the
    // scope check that precedes `mapAsync`.
    gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.STORAGE });
    await assertRejects(() => batch.finishAndRead({ s: sink }), GpuValidationError);
    const stats = target.diagnostics().submit;
    assertEquals(
      stats.msPerWorkgroup,
      undefined,
      "a measurement of a window that failed before the fence entered the estimate",
    );
    assertEquals(stats.currentChunkSize, DEFAULT_SUBMIT_POLICY.initialChunkSize);
    assertEquals(sink.useReferences, 0);
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch enqueue: an enqueueRead re-entered from an input getter is rejected by the re-check after the copy", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    const batch = await gpu.beginBatch();
    let read: EnqueueRead | undefined;
    const outer = target.enqueue(
      {
        get x0(): Tensor {
          read = target.enqueueRead({ x0: input() }, { batch });
          return { dtype: "f32", shape: [3], data: Float32Array.of(10, 20, 30) };
        },
      },
      { batch },
    );
    await assertRejects(() => outer, BatchScopeError, "batch holding an enqueueRead");
    await batch.finish();
    assertEquals(
      Array.from((await read!.outputs).y.data),
      [-1, -2, -3],
      "a later enqueue's input got mixed into the settling readback (silently wrong value)",
    );
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueue: a run re-entered from an input getter becomes a BatchScopeError, not a self-deadlock", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    const batch = await gpu.beginBatch();
    let run: Promise<RunOutputs> | undefined;
    const outer = target.enqueue(
      {
        get x0(): Tensor {
          run = target.run({ x0: input() });
          return input();
        },
      },
      { batch },
    );
    await settleWithin(
      assertRejects(() => outer, BatchScopeError, "unsettled run"),
      "enqueue",
    );
    await settleWithin(batch.finish(), "finish");
    assertEquals(Array.from((await settleWithin(run!, "run")).y.data), [-1, -2, -3]);
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("run: a dispose re-entered from an input getter is rejected as disposed and creates no backing after disposal", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    let disposal: Promise<void> | undefined;
    await assertRejects(
      () =>
        target.run({
          get x0(): Tensor {
            disposal = target.dispose();
            return input();
          },
        }),
      ExecutionError,
      "disposed",
    );
    await disposal!;
    assertEquals(
      target.diagnostics().planBacking.retainedCount,
      0,
      "a backing was allocated after disposal",
    );
  } finally {
    gpu.destroy();
  }
});

test("batch finishAndRead: re-entry from an output getter is rejected by the re-check after the copy and keeps the inner registration", async () => {
  const gpu = await acquireGpu(),
    a = await gpu.createResident(4),
    b = await gpu.createResident(4);
  a.write(Float32Array.of(1));
  b.write(Float32Array.of(2));
  try {
    const batch = await gpu.beginBatch();
    let inner: Promise<Readonly<Record<string, ArrayBuffer>>> | undefined;
    const outer = batch.finishAndRead({
      get o(): ResidentTensor {
        inner = batch.finishAndRead({ i: b });
        return a;
      },
    });
    await assertRejects(() => outer, BatchScopeError, "only once");
    const read = await inner!;
    assertEquals(Array.from(new Float32Array(read.i)), [2]);
    assertEquals(
      [a.useReferences, b.useReferences],
      [0, 0],
      "the use reservation of the overwritten registration was not returned",
    );
  } finally {
    a.dispose();
    b.dispose();
    gpu.destroy();
  }
});

// Checks for the self-deadlock detector of a batch range x an unsettled run.
//
// The cycle closes over 4 edges: "the BatchScope body holds the errorScope range lock while it waits
// for an in-flight lease to come back -> the lease comes back only in the enqueue body's finally ->
// the enqueue body waits on the Session's serialization chain for the earlier run to settle -> the
// run body waits for that range lock". `withScopeLock` cannot tell "a legitimate queue" from
// "re-entry", so no re-entry detector can sit there, and without detection the process **hangs
// forever with no exception and no diagnostic**. These tests pin three things: (1) that sequence
// becomes a typed exception; (2) the same holds for a run issued **before** the batch was opened
// (beginBatch takes the lock synchronously in its constructor, so within the same tick the batch
// always comes first); (3) correct usage does not change by a single byte.
//
// NOTE: these check a hang, so **they do not confirm the rendezvous with await** (awaiting a
// sequence that never settles keeps the test runner itself from returning). After the detector has
// passed, always close the range with finish() first, then collect the runs still queued.

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;

/** y = x + x (= 2x). The smallest graph usable by both run and enqueue. */
const DOUBLE_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

/** An input whose values change with `phase` (handing out the same values would hide a mix-up). */
const phasedInput = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/** Reference values (rounded in f32 independently of the implementation). */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = phasedInput(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const openDoubleSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(DOUBLE_GRAPH), "model");

test("an unawaited run issued inside a batch range makes enqueue fail loudly (real GPU)", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  try {
    const batch = await gpu.beginBatch();
    // MUST NOT await. Awaiting would make the run wait for the range lock and hang **here**, so
    // what the detector looks at is the side of "a run queued and not yet settled".
    const pending = target.run({ x: phasedInput(0) });
    try {
      await assertRejects(
        () => target.enqueue({ x: phasedInput(1) }, { batch }),
        BatchScopeError,
        "unsettled run",
      );
    } finally {
      // Closing the range returns the lock, and the queued run can run to completion.
      await batch.finish();
    }
    assertEquals(
      bits((await pending)["y"].data),
      bits(doubled(0)),
      "the run itself completes unharmed",
    );
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("an unawaited run issued before the batch is opened forms the same cycle (true positive, real GPU)", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  try {
    // Even when it is issued **outside** the range, the run body goes for the lock only one microtask
    // later. beginBatch, on the other hand, takes the lock synchronously in the BatchScope constructor, so
    // within the same tick the batch always comes first = the form that hangs even while following the
    // current doc's "do not await a run during the range".
    const pending = target.run({ x: phasedInput(2) });
    const batch = await gpu.beginBatch();
    try {
      await assertRejects(
        () => target.enqueue({ x: phasedInput(1) }, { batch }),
        BatchScopeError,
        "self-deadlock",
      );
    } finally {
      await batch.finish();
    }
    assertEquals(bits((await pending)["y"].data), bits(doubled(2)));
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("the correct sequence (only enqueue inside the range, run outside it) does not trip the detector (real GPU)", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  const sink = await gpu.createResident(COUNT * 4, "sink");
  try {
    const before = await target.run({ x: phasedInput(0) });

    const batch = await gpu.beginBatch();
    await target.enqueue({ x: phasedInput(1) }, { batch, copyOutputs: { y: sink } });
    // An unawaited enqueue is outside the detector's scope (the cycle needs an unsettled **run**).
    const queued = target.enqueue({ x: phasedInput(2) }, { batch, copyOutputs: { y: sink } });
    await batch.finish();
    await queued;
    assertEquals(bits(await sink.read()), bits(doubled(2)), "runs through to the last enqueue");

    // Runs outside the range are undisturbed before and after (a settled run leaves nothing in the bookkeeping).
    const after = await target.run({ x: phasedInput(0) });
    assertEquals(bits(after["y"].data), bits(before["y"].data));
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});
