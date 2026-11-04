// Checks for resident tensors (the fourth lifetime class) and fenceless enqueue (H-5).
//
// The aim is "bring the run-boundary fences of the generation loop down to a few", so these
// checks pin ① correct values (both resident inputs and copy targets are bit-identical to a run
// through the host) ② **that the fences really decrease** (one onSubmittedWorkDone for the whole
// batch) ③ no overtaking (the invariant of eager submit) ④ lifetime violations fail loudly (use
// after dispose, dispose while referenced).
//
// Without ②, an internal fallback to flush would still pass with correct values (no exception, no
// warning). Without ③, swapping the inputs of consecutive enqueues would go unnoticed, since "both
// are computed anyway".

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  type BatchScope,
  BatchScopeError,
  type GpuContext,
  GpuDeviceLostError,
  ResidentTensorError,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { GpuValidationError, popFailureScopes, pushFailureScopes } from "../src/gpu/error-scope.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, f32Bytes, openGraphModel } from "./helpers/model-fixture.ts";
import { countFences } from "./helpers/fences.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";
import { DEFAULT_SUBMIT_POLICY } from "../src/gpu/submit.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/** y = x + x (= 2x). The producer side. */
const PRODUCER: DeclarationJson = {
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

/** w = z * z. The consumer side (receives the producer's output through a resident tensor). */
const CONSUMER: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["mul"] },
  symbols: [],
  inputs: [{ name: "z", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["w"],
  initializers: {},
  values: { w: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "mul", ins: ["z", "z"], outs: ["w"], attrs: {} }],
};

/** An input whose values change with `phase` (handing out the same values would hide a mix-up). */
const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/**
 * Reference values, computed by hand while rounding to f32 at every node (independent of the
 * implementation).
 */
const expectedChain = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  const out = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i += 1) {
    const doubled = Math.fround(x[i] + x[i]);
    out[i] = Math.fround(doubled * doubled);
  }
  return out;
};

/**
 * Bit-pattern comparison (does not tolerate a rounding mix-up). A resident tensor reads back as a
 * plain ArrayBuffer.
 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const producerSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(PRODUCER), "model");

const consumerSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(CONSUMER), "model");

test.skipIf(!GPU_AVAILABLE)(
  "a resident input is bound without writeBuffer and is bit-identical to a run with host input (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const resident = await gpu.createResident(BYTES, "x");
    const session = await producerSession(gpu);
    const reference = await producerSession(gpu);
    try {
      for (const phase of [0, 1, 2]) {
        const tensor = input(phase);
        resident.write(tensor.data);
        // Run 1 is not backed (the arena path); run 2 onwards is backed. On both paths a resident
        // input "binds the storage as is", so both must give the same values.
        const actual = await session.run({ x: resident });
        const expected = await reference.run({ x: tensor });
        assertEquals(bits(actual["y"].data), bits(expected["y"].data), `phase ${phase}`);
      }
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      await reference.dispose();
      resident.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a chain of enqueues passes values through resident tensors with a single fence for the batch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const fences = countFences(gpu);
    try {
      const batch = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });
      // The producer's output remains only in the resident tensor. The consumer takes it **as is**
      // as its input.
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      assertEquals(fences.count(), 0, "enqueue sets no fence at all");

      await batch.finish();
      assertEquals(fences.count(), 1, "one onSubmittedWorkDone for the whole batch");

      assertEquals(bits(await sink.read()), bits(expectedChain(0)));
      // Both enqueues are backed from the first call (no silent fallback to the arena path).
      assertEquals(producer.diagnostics().planBacking.buildCount, 1);
      assertEquals(consumer.diagnostics().planBacking.buildCount, 1);
      assertEquals(producer.diagnostics().lastRun, undefined, "enqueue does not use the arena");
    } finally {
      fences.restore();
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "settle() sets a single fence in the middle of the section and switches the later chunk cap to the estimate (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const fences = countFences(gpu);
    try {
      const batch = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });
      const before = producer.diagnostics().submit;
      assertEquals(before.measuredCount, 0, "no window closes in the middle of the section");
      assertEquals(before.currentChunkSize, DEFAULT_SUBMIT_POLICY.initialChunkSize);

      await batch.settle();
      assertEquals(fences.count(), 1, "settle() sets one fence");
      const after = producer.diagnostics().submit;
      assertEquals(after.measuredCount, 1, "settle() closes one window");
      // A measurement below the resolution (0) is no evidence (invariant 1) — this shape is a
      // single dispatch of a few µs, so the cap switch is checked only when evidence exists.
      if (after.msPerWorkgroup !== undefined) {
        assertEquals(after.currentChunkSize, DEFAULT_SUBMIT_POLICY.maxChunkSize);
      }

      // The section continues: the later enqueue joins the same section and completes with
      // finish's fence.
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      await batch.finish();
      assertEquals(fences.count(), 2, "one from settle() + one from finish");
      assertEquals(bits(await sink.read()), bits(expectedChain(0)));

      await assertRejects(() => batch.settle(), BatchScopeError, "after finish()");
    } finally {
      fences.restore();
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "settle() fails loudly while an enqueue is in flight (keeps the window attribution, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      const pending = producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });
      await assertRejects(() => batch.settle(), BatchScopeError, "in-flight");
      await pending;
      await batch.settle();
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      await batch.finish();
      assertEquals(bits(await sink.read()), bits(expectedChain(0)));
    } finally {
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "during settle(), enqueue, a second settle() and finish() all fail loudly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });

      // Take the three rejections together in **the synchronous part** that does not await
      // `settle()`. With an await in between, the fence could resolve midway and the section return
      // to open, so whether a rejection is hit would be a matter of run-time luck (the check would
      // never fire).
      const settling = batch.settle();
      const duplicated = batch.settle();
      const late = producer.enqueue({ x: input(1) }, { batch, copyOutputs: { y: carrier } });
      const closing = batch.finish();

      await assertRejects(() => duplicated, BatchScopeError, "cannot overlap");
      await assertRejects(() => late, BatchScopeError, "in the middle of settle()");
      await assertRejects(() => closing, BatchScopeError, "Cannot finish()");
      await settling;

      // Confirm by value that the rejected enqueue did not enter the section: had it gone through,
      // carrier would be overwritten with twice phase 1 and the consumer's output would be
      // expectedChain(1).
      assert(
        JSON.stringify(bits(expectedChain(0))) !== JSON.stringify(bits(expectedChain(1))),
        "the expected values do not change per phase (the detector would never fire)",
      );
      // The rejections do not break the section: finish after settle goes through as before, and
      // neither leases nor bookkeeping remain.
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      await batch.finish();
      assertEquals(bits(await sink.read()), bits(expectedChain(0)));
    } finally {
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "input writes of consecutive enqueues do not overtake the preceding dispatch (the eager-submit check, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const first = await gpu.createResident(BYTES, "first");
    const second = await gpu.createResident(BYTES, "second");
    try {
      // Same signature = **one** input buffer, owned by the backing. The order queued is enqueue →
      // writeBuffer (the next value) → enqueue, so without eager submit the second write would
      // overtake the first dispatch and both would get the new value (no exception).
      const batch = await gpu.beginBatch();
      await session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: first } });
      await session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: second } });
      await batch.finish();

      const doubled = (phase: number): Float32Array<ArrayBuffer> => {
        const x = input(phase).data;
        return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
      };
      // A check against tautology: if the two phases expected the same values, a mix-up could not
      // be detected.
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(1))),
        "the expected values do not change per phase (the detector would never fire)",
      );
      assertEquals(
        bits(await first.read()),
        bits(doubled(0)),
        "the earlier enqueue reads the old value",
      );
      assertEquals(
        bits(await second.read()),
        bits(doubled(1)),
        "the later enqueue reads the new value",
      );
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        1,
        "the same signature is not rebuilt",
      );
    } finally {
      await session.dispose();
      first.dispose();
      second.dispose();
      gpu.destroy();
    }
  },
);

// clear() replaces a host write of zeros (`write(new Float32Array(n))`) for state reset between
// runs. A no-op clear would leave the previous contents, so every case starts from non-zero data.
test.skipIf(!GPU_AVAILABLE)(
  "clear() zeroes every byte of a resident that held data and keeps its size (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const resident = await gpu.createResident(BYTES, "state");
    try {
      // Every word non-zero, so a clear that skipped any range shows up in the read-back.
      const pattern = Float32Array.from({ length: COUNT }, (_, i) => i + 1);
      resident.write(pattern);
      assertEquals(bits(await resident.read()), bits(pattern), "the write did not land (vacuous)");

      resident.clear();
      assertEquals(bits(await resident.read()), new Array<number>(COUNT).fill(0));
      assertEquals(resident.byteLength, BYTES, "clear changed the size");

      // The resident stays writable after a clear (it is a content reset, not a lifetime event).
      resident.write(pattern);
      assertEquals(bits(await resident.read()), bits(pattern));
    } finally {
      resident.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "clear() between settled enqueues lands in queue order: after the earlier copy, before the later dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const zeros = (): readonly number[] => new Array<number>(COUNT).fill(0);
    try {
      // Detector check: without the clear the consumer sees the producer's non-zero output.
      assert(
        JSON.stringify(bits(expectedChain(0))) !== JSON.stringify(zeros()),
        "the chain's expected value is zero (the detector is vacuous)",
      );
      const batch = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });
      // If the clear overtook the producer's copy, carrier would end as 2x and the consumer would
      // read it; if the consumer's dispatch overtook the clear, the same. Only queue order gives 0.
      carrier.clear();
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      await batch.finish();
      assertEquals(
        bits(await sink.read()),
        zeros(),
        "the consumer did not read the cleared carrier",
      );
      assertEquals(bits(await carrier.read()), zeros());
    } finally {
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

// finishAndRead takes a use reservation on each member until the batch settles (ADR 0054 addendum
// 2026-09-12); clear() checks the same counter as for an issued run / enqueue.
test.skipIf(!GPU_AVAILABLE)(
  "clear() refuses a resident held by a pending finishAndRead and goes through after the batch settles (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const state = await gpu.createResident(BYTES, "state");
    try {
      const pattern = Float32Array.from({ length: COUNT }, (_, i) => i + 1);
      state.write(pattern);
      const batch = await gpu.beginBatch();
      const reading = batch.finishAndRead({ state });
      assertEquals(state.useReferences, 1, "finishAndRead took no use reservation (vacuous)");
      // A clear here would be submitted before the batch's readback copy, which would read zeros.
      assertThrows(() => state.clear(), ResidentTensorError, "cannot be cleared");
      assertEquals(
        bits((await reading)["state"]),
        bits(pattern),
        "the refused clear changed the data",
      );
      assertEquals(state.useReferences, 0);

      state.clear();
      assertEquals(bits(await state.read()), new Array<number>(COUNT).fill(0));
    } finally {
      state.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the identity of a resident input is part of the derived plan's key (swapping re-derives, swapping back hits, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const a = await gpu.createResident(BYTES, "a");
    const b = await gpu.createResident(BYTES, "b");
    try {
      a.write(input(0).data);
      b.write(input(1).data);

      await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });
      await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });

      // A different resident tensor is a different signature. Baked bind groups must not be
      // reused.
      const swapped = await session.run({ x: b });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 2 },
        "swapping the resident input gives a different key",
      );
      const doubled = (phase: number): readonly number[] => {
        const x = input(phase).data;
        return bits(Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i])));
      };
      assertEquals(bits(swapped["y"].data), doubled(1), "after the swap the new storage is read");

      const back = await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 2 });
      assertEquals(bits(back["y"].data), doubled(0), "swapping back reads the original storage");

      // Host input (Tensor) and resident input have separate keys (a different signature even with
      // the same shape).
      await session.run({ x: input(0) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 3 });
    } finally {
      await session.dispose();
      a.dispose();
      b.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a resident tensor referenced by baked bind groups cannot be disposed (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const bound = await gpu.createResident(BYTES, "bound");
    const loose = await gpu.createResident(BYTES, "loose");
    try {
      bound.write(input(0).data);
      assertEquals(bound.bakedReferences, 0);
      // Run 1 is not backed (nothing baked) — at this point it should still be disposable.
      await session.run({ x: bound });
      assertEquals(bound.bakedReferences, 0, "a non-backed run creates no baked reference");
      // Run 2 builds the backing, and the bind groups bake this storage in.
      await session.run({ x: bound });
      assertEquals(bound.bakedReferences, 1);

      assertThrows(() => bound.dispose(), ResidentTensorError, "referenced by baked bind groups");
      assertEquals(bound.disposed, false, "a refused dispose does not corrupt the state");

      // A check against tautology: an unreferenced resident tensor passes the same call (evidence
      // that the check does not always fail).
      loose.dispose();
      assertEquals(loose.disposed, true);

      // Going through the only paths that drop the reference (disposing the Session / switching
      // signature) makes it disposable.
      await session.dispose();
      assertEquals(bound.bakedReferences, 0);
      bound.dispose();
      assertEquals(bound.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "using a disposed resident tensor fails loudly on every surface (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const dead = await gpu.createResident(BYTES, "dead");
    try {
      dead.dispose();
      dead.dispose(); // the second call is a no-op (dispose is idempotent)

      assertThrows(
        () => dead.write(new Float32Array(COUNT)),
        ResidentTensorError,
        "a disposed resident tensor",
      );
      await assertRejects(() => dead.read(), ResidentTensorError, "a disposed resident tensor");
      assertThrows(() => dead.clear(), ResidentTensorError, "cannot be used (clear)");
      await assertRejects(() => session.run({ x: dead }), ExecutionError, "disposed");

      const batch = await gpu.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: dead } }),
          ExecutionError,
          "disposed",
        );
      } finally {
        await batch.finish();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

// On a lost device, `queue.writeBuffer` becomes a **silent no-op** on a destroyed buffer (not even
// a warning), and the errorScope of `createBuffer` does not catch the loss either (pop resolves
// with null), so a resident holding an invalid buffer comes back as a success. Both turn loud only
// at the next fence, and the diagnostics in between are misleading — only a synchronous check at
// the entry can stop it.
test.skipIf(!GPU_AVAILABLE)(
  "resident tensors are refused right after device.destroy() (before the lost reaction) (real GPU)",
  async () => {
    // This case breaks the device, so it takes a GpuContext of its own.
    const gpu = await acquireGpu();
    const resident = await gpu.createResident(BYTES, "victim");

    gpu.destroy();
    // MUST: lost is **not yet recorded** here. If it were, the existing lost check alone would
    // pass, which would be no evidence that "the synchronous destroy flag is checked" (the check
    // would be a tautology).
    assertStrictEquals(gpu.lost, undefined, "the lost reaction ran right after destroy");
    assert(gpu.destroyRequested, "the destroy request was not set synchronously");

    assertThrows(
      () => resident.write(new Float32Array(COUNT)),
      GpuDeviceLostError,
      "device was lost",
    );
    // read goes through the same check. Without it, read would allocate staging on the destroyed
    // device and submit a copy, and popErrorScope after the loss resolves with null, so the failure
    // would not be caught (this one call is the only evidence for the check — after lost is
    // recorded below, raceDeviceLost already fails with the same type).
    await assertRejects(() => resident.read(), GpuDeviceLostError, "device was lost");
    // clear() takes the same guard: commands on a destroyed device are silent no-ops, so the state
    // would silently keep its previous contents.
    assertThrows(() => resident.clear(), GpuDeviceLostError, "device was lost");
    await assertRejects(
      () => gpu.createResident(BYTES, "after-destroy"),
      GpuDeviceLostError,
      "device was lost",
    );

    await gpu.device.lost;
    assert(gpu.lost !== undefined, "the loss was not recorded (the check would never fire)");
    assertThrows(() => resident.write(new Float32Array(COUNT)), GpuDeviceLostError);
    assertThrows(() => resident.clear(), GpuDeviceLostError);
    await assertRejects(() => resident.read(), GpuDeviceLostError);
    await assertRejects(() => gpu.createResident(BYTES, "after-lost"), GpuDeviceLostError);
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a resident tensor size mismatch fails loudly on both the input side and the copy-target side (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const wrong = await gpu.createResident(BYTES + 4, "wrong");
    const right = await gpu.createResident(BYTES, "right");
    try {
      assertThrows(() => right.write(new Float32Array(COUNT + 1)), ResidentTensorError, "write");
      await assertRejects(() => session.run({ x: wrong }), ExecutionError, "bytes, but shape");

      const batch = await gpu.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: wrong } }),
          ExecutionError,
          "bytes, but shape",
        );
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { nope: right } }),
          ExecutionError,
          "is not a graph output",
        );
      } finally {
        // The section itself can still settle after the failed enqueues (the leftovers are
        // discarded). But settling attributes **the first host-side failure** — the first call's
        // size mismatch shows rather than the second's "not a graph output", which is why only one
        // record is kept (a derived failure does not hide the root cause).
        await assertRejects(() => batch.finish(), ExecutionError, "bytes, but shape");
      }
    } finally {
      await session.dispose();
      wrong.dispose();
      right.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "createResident accepts only positive sizes that are multiples of 4",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const bad of [0, -4, 6, 1.5]) {
        await assertRejects(
          () => gpu.createResident(bad),
          ResidentTensorError,
          "multiple of 4",
          `byteLength=${bad}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a settled batch accepts no enqueue, and finish returns the same settlement every time (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      await session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: sink } });
      await batch.finish();
      await batch.finish();
      assertEquals(batch.finished, true);
      await assertRejects(
        () => session.enqueue({ x: input(0) }, { batch }),
        BatchScopeError,
        "after finish()",
      );

      // An ordinary run after the section closes works as before (the scope lock was released).
      const outputs = await session.run({ x: input(0) });
      assertEquals(outputs["y"].shape, [ROWS, COLS]);
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE)(
  "a batch on a timing device runs untimed, and the runs around it keep only their own breakdown (real GPU, timestamp-query)",
  async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      await session.run({ x: input(0) });
      const before = session.diagnostics().lastRunTiming;
      assert(before !== undefined && before.entries.length > 0, "the run before has a breakdown");

      const batch = await gpu.beginBatch();
      for (const phase of [1, 2, 3]) {
        await session.enqueue({ x: input(phase) }, { batch, copyOutputs: { y: sink } });
      }
      const read = await batch.finishAndRead({ sink });
      const doubled = Float32Array.from(input(3).data, (value) => Math.fround(value + value));
      assertEquals(bits(read["sink"]), bits(doubled), "the batch's values (y = x + x)");

      await session.run({ x: input(4) });
      const after = session.diagnostics().lastRunTiming;
      assert(after !== undefined, "the run after has a breakdown");
      // A timed batch chunk would leave timestamps that the next run's flush collects into that
      // run's totals, so the run after the batch would count more dispatches than the run before.
      assertEquals(after.dispatchCount, before.dispatchCount, "the run after counts only its own");
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a batch opened on another GpuContext accepts no enqueue (real GPU)",
  async () => {
    const owner = await acquireGpu();
    const other = await acquireGpu();
    const session = await producerSession(other);
    try {
      const batch = await owner.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch }),
          BatchScopeError,
          "another GpuContext",
        );
      } finally {
        await batch.finish();
      }
    } finally {
      await session.dispose();
      owner.destroy();
      other.destroy();
    }
  },
);

/**
 * y = reshape(x) ([ROWS,COLS] → [COUNT]). A degenerate graph whose output is an **alias** of the
 * input.
 */
const ALIAS: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [COUNT] } },
  nodes: [{ op: "reshape", ins: ["x"], outs: ["y"], attrs: {} }],
};

test.skipIf(!GPU_AVAILABLE)(
  "finish() right after queuing non-awaited enqueues still takes all of them into the section (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      // MUST: await none of them. Without the in-flight lease, finish() would settle first and
      // the result would be a silent miss: ① all three reject with BatchScopeError ② sink is all 0
      // (zero dispatches) ③ finish still returns success.
      // NOTE: phase has period 3 (the `input` expression), so queue three within the range where
      // the values differ.
      const pending = [0, 1, 2].map((phase) =>
        session.enqueue({ x: input(phase) }, { batch, copyOutputs: { y: sink } }),
      );
      await batch.finish();
      await Promise.all(pending);

      const doubled = (phase: number): Float32Array<ArrayBuffer> => {
        const x = input(phase).data;
        return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
      };
      // A check against tautology: if the first and the last expected values were the same, "the
      // last one went in" could not be confirmed.
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(2))),
        "the expected values do not change per phase (the detector would never fire)",
      );
      assertEquals(bits(await sink.read()), bits(doubled(2)), "every enqueue up to the last ran");
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a resident tensor of another GpuContext fails loudly on the hit path too (the ID-collision check, real GPU)",
  async () => {
    const owner = await acquireGpu();
    const other = await acquireGpu();
    const session = await producerSession(owner);
    const mine = await owner.createResident(BYTES, "mine");
    const foreign = await other.createResident(BYTES, "foreign");
    try {
      // Confirm the premise: IDs are assigned independently per GpuContext, so they collide (if
      // they did not, the derived plan's keys would split and this check would detect nothing).
      assertEquals(
        mine[RUNTIME_INTERNAL].id,
        foreign[RUNTIME_INTERNAL].id,
        "the first residents of different contexts share an ID (the premise this check guards)",
      );
      assertEquals(mine.byteLength, foreign.byteLength, "the size check cannot tell them apart");

      mine.write(input(0).data);
      await session.run({ x: mine }); // run 1 = miss (the arena path)
      await session.run({ x: mine }); // run 2 = hit. The backing bakes in mine
      assertEquals(session.diagnostics().planBacking.buildCount, 1);

      // A baked backing is reused as is when the key matches, so without the check not a single
      // byte of foreign would be read and mine's old values would come back (no exception, no
      // warning).
      await assertRejects(
        () => session.run({ x: foreign }),
        ResidentTensorError,
        "another GpuContext",
      );
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        1,
        "the refused run does not rebuild the backing",
      );
    } finally {
      await session.dispose();
      mine.dispose();
      foreign.dispose();
      owner.destroy();
      other.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "copyOutputs with the same buffer as source and destination fails at enqueue (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openGraphModel(ALIAS), "model");
    const state = await gpu.createResident(BYTES, "state");
    try {
      const batch = await gpu.beginBatch();
      try {
        // y is an alias of x (reshape), so the copy source's storage is state itself. Without the
        // check a self-copy would be queued and the validation failure delayed until
        // batch.finish() (= unrelated enqueues in the same section would go down with it, and the
        // enqueue at fault could not be identified).
        await assertRejects(
          () => session.enqueue({ x: state }, { batch, copyOutputs: { y: state } }),
          ExecutionError,
          "source and destination are the same buffer",
        );
      } finally {
        // The failed enqueue is also attributed to the settlement as this section's first
        // host-side failure (RC1-2).
        await assertRejects(
          () => batch.finish(),
          ExecutionError,
          "source and destination are the same buffer",
        );
      }
    } finally {
      await session.dispose();
      state.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a resident tensor bound by a run in progress cannot be disposed (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const bound = await gpu.createResident(BYTES, "bound");
    const device = gpu.device;
    const original = device.createComputePipelineAsync.bind(device);
    let injected = 0;
    try {
      bound.write(input(0).data);
      // A miss run proceeds as "bind the storage into env → await pipeline creation → encode".
      // This injection tries dispose in that await window (= during pipeline creation). There are
      // still no baked references, so without the binding reservation this would slip through.
      device.createComputePipelineAsync = (
        descriptor: GPUComputePipelineDescriptor,
      ): Promise<GPUComputePipeline> => {
        if (injected === 0) {
          injected += 1;
          assertEquals(bound.bakedReferences, 0, "the window where no baked reference exists yet");
          assertEquals(bound.boundReferences, 1, "the binding reservation is held");
          assertThrows(() => bound.dispose(), ResidentTensorError, "bound as an input");
        }
        return original(descriptor);
      };

      const outputs = await session.run({ x: bound });
      assertEquals(injected, 1, "the injection never ran (the window was not hit)");
      const x = input(0).data;
      assertEquals(
        bits(outputs["y"].data),
        bits(Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]))),
        "the refused dispose does not corrupt the run's values",
      );

      // Once the run completes, the reservation is released (failing to release it would make the
      // tensor undisposable from then on).
      assertEquals(bound.boundReferences, 0);
      assertEquals(bound.disposed, false, "a refused dispose does not corrupt the state");
      bound.dispose();
      assertEquals(bound.disposed, true);
    } finally {
      device.createComputePipelineAsync = original;
      await session.dispose();
      gpu.destroy();
    }
  },
);

/** The reference value of 2x (PRODUCER's output). */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

// When a non-awaited enqueue failed in its body, the section used to settle as "success with one
// dispatch fewer" (the errorScope catches only GPU-side failures, and a host-side throw shows only
// in the returned Promise = unless someone holds it, it just escapes as an unhandled rejection).
// This check pins that attribution.
test.skipIf(!GPU_AVAILABLE)(
  "the failure of a non-awaited enqueue is attributed to finish() (the first one, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const wrong = await gpu.createResident(BYTES + 4, "wrong");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      // MUST: await neither (the form this surface explicitly allows). The first fails in its body
      // because the copy target's size does not match; the second is queued normally.
      const pending = [
        session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: wrong } }),
        session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } }),
      ];
      // Attach the handler in the same synchronous part as the issue, so no unhandled rejection
      // arises.
      const settled = Promise.allSettled(pending);

      const attributed = await assertRejects(
        () => batch.finish(),
        ExecutionError,
        "bytes, but shape",
      );

      const results = await settled;
      assert(
        results[0].status === "rejected",
        "the returned Promise of the failed enqueue rejects",
      );
      assertStrictEquals(
        results[0].reason,
        attributed,
        "finish throws the very exception the enqueue failed with (one fact seen through two paths)",
      );
      assertEquals(results[1].status, "fulfilled", "the healthy enqueue does not go down with it");

      // A check against tautology: if the expected values were all 0, "the copy happened" could
      // not be confirmed.
      assert(
        bits(doubled(1)).some((word) => word !== 0),
        "the expected values are all 0 (whether the copy happened cannot be told)",
      );
      assertEquals(
        bits(await sink.read()),
        bits(doubled(1)),
        "the healthy enqueue's copy target is updated",
      );
    } finally {
      await session.dispose();
      wrong.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

// The binding reservation (`#bindInput`) is taken in the execution body = one microtask later, so
// a resident with neither baked nor binding references had a window where "dispose goes through in
// the same tick right after the API accepted the call". The use reservation at issue makes what the
// dispose doc promises — "a mistake fails at the dispose call site with its true cause" — hold for
// accepted runs / enqueues too.
test.skipIf(!GPU_AVAILABLE)(
  "a resident tensor used by an issued run / enqueue cannot be disposed even in the same tick (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const fresh = await gpu.createResident(BYTES, "fresh");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      fresh.write(input(0).data);
      // Run 1 is on the miss path (no backing) — confirms the window where no baked reference
      // exists.
      assertEquals(
        fresh.bakedReferences,
        0,
        "a baked reference already exists (the check would never fire)",
      );

      const pending = session.run({ x: fresh });
      assertEquals(
        fresh.useReferences,
        1,
        "no use reservation in the synchronous part of the issue",
      );
      assertEquals(
        fresh.boundReferences,
        0,
        "the binding reservation is not taken until the body (the evidence for this window)",
      );
      assertThrows(() => fresh.dispose(), ResidentTensorError, "in use");
      assertEquals(fresh.disposed, false, "a refused dispose does not corrupt the state");
      // clear() refuses in the same window (unlike write, which leaves it as a caller contract): a
      // clear here would land before the run's dispatch and zero the input it reads.
      assertThrows(() => fresh.clear(), ResidentTensorError, "cannot be cleared");

      // The refused dispose does not break the run (the accepted execution runs to completion).
      assertEquals(bits((await pending)["y"].data), bits(doubled(0)));
      assertEquals(fresh.useReferences, 0, "the use reservation was not released on settlement");

      // The copy target (copyOutputs) gets the same lifetime protection.
      const batch = await gpu.beginBatch();
      const queued = session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } });
      assertEquals(sink.useReferences, 1, "no use reservation on the copy target");
      assertThrows(() => sink.dispose(), ResidentTensorError, "in use");
      assertThrows(() => sink.clear(), ResidentTensorError, "cannot be cleared");
      await queued;
      await batch.finish();
      assertEquals(sink.useReferences, 0);
      assertEquals(bits(await sink.read()), bits(doubled(1)));

      // After settlement it goes through (evidence that the reservations were released — the check
      // is not a tautology).
      // The refused clears left the values intact (the run read input(0), the copy wrote doubled(1)),
      // and a clear after settlement goes through.
      fresh.clear();
      sink.clear();
      assertEquals(bits(await sink.read()), new Array<number>(COUNT).fill(0));
      fresh.dispose();
      sink.dispose();
      assertEquals(fresh.disposed, true);
      assertEquals(sink.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a failure midway through taking use reservations leaks none of those already taken (the rollback check, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const alive = await gpu.createResident(BYTES, "alive");
    const spare = await gpu.createResident(BYTES, "spare");
    const dead = await gpu.createResident(BYTES, "dead");
    try {
      dead.dispose();
      const batch = await gpu.beginBatch();
      // The shape that fails on the copy target (dead) after taking the input's (alive)
      // reservation. Without the rollback, alive could never be disposed afterwards.
      await assertRejects(
        () => session.enqueue({ x: alive }, { batch, copyOutputs: { y: dead } }),
        ExecutionError,
        "disposed",
      );
      assertEquals(alive.useReferences, 0, "the use reservation already taken was not rolled back");
      // An enqueue that failed before entering the section is not a failure of the section (it
      // took no lease).
      await batch.finish();

      // The same holds on the path where batch.enter() fails (here it fails after taking both the
      // input and the copy target).
      await assertRejects(
        () => session.enqueue({ x: alive }, { batch, copyOutputs: { y: spare } }),
        BatchScopeError,
        "after finish()",
      );
      assertEquals(alive.useReferences, 0, "the use reservation leaked on the enter failure");
      assertEquals(spare.useReferences, 0, "the use reservation leaked on the enter failure");

      alive.dispose();
      spare.dispose();
      assertEquals(alive.disposed, true);
      assertEquals(spare.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

// createResident is async because it wraps the call in an errorScope (createBuffer over the limit
// throws no synchronous exception and **returns an invalid buffer**). Removing push/pop keeps the
// values correct, so without this check the regression "an invalid resident comes back as a
// success and later writes become silent no-ops" would pass all green.
test.skipIf(!GPU_AVAILABLE)(
  "createResident over the limit becomes a typed exception through the errorScope (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // MUST: stay over the limit and a multiple of 4 (otherwise the synchronous argument check
      // fires first and a different check gets tested). `maxBufferSize` itself is not necessarily a
      // multiple of 4 (Intel Arc B570 / Deno wgpu report 2³¹−1 — measured 2026-09-20), so round up
      // to the next multiple of 4 above the limit.
      const tooBig = Math.ceil((gpu.limits.maxBufferSize + 1) / 4) * 4;
      assert(tooBig > gpu.limits.maxBufferSize && tooBig % 4 === 0, `tooBig ${tooBig}`);
      // By the spec, size > maxBufferSize is a validation error **without an allocation** (it uses
      // no real memory).
      const error = await assertRejects(
        () => gpu.createResident(tooBig, "too-big"),
        GpuValidationError,
      );
      assert(error.message.startsWith("resident 'too-big' allocation: "), error.message);

      // The scope balance is 0 (if the failure path left a pop undone, this turns red).
      pushFailureScopes(gpu.device);
      assertEquals(await popFailureScopes(gpu.device, "balance check"), undefined);

      // A check against tautology: the check is not of the "always fails" kind.
      const ok = await gpu.createResident(BYTES, "ok");
      ok.write(input(0).data);
      assertEquals(bits(await ok.read()), bits(input(0).data));
      ok.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

// A section keeps holding the per-device errorScope section lock, so leaving without finish()
// would make later runs wait forever with neither an exception nor a diagnostic. `await using` is a
// style that makes that slip harder; without asyncDispose it is a type error in TS and a TypeError
// in JS (and only after the lock is taken).
test.skipIf(!GPU_AVAILABLE)(
  "a batch section opened with await using is settled when the block exits (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    let scope: BatchScope | undefined;
    try {
      {
        await using batch = await gpu.beginBatch();
        scope = batch;
        await session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: sink } });
        assertEquals(
          batch.finished,
          false,
          "settled in the middle of the section (the check would never fire)",
        );
      }
      assertEquals(scope.finished, true, "finish() was not called on leaving the section");
      assertEquals(bits(await sink.read()), bits(doubled(0)), "the section's work has completed");

      // Evidence that the section lock was released. Otherwise this run would wait forever (with
      // neither an exception nor a diagnostic).
      const outputs = await session.run({ x: input(1) });
      assertEquals(bits(outputs["y"].data), bits(doubled(1)));
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

/**
 * A graph output that names an initializer (the IR allows it). Its value is not in the baked value
 * mapping, so it cannot be a `copyOutputs` target (it would only copy the same constant within the
 * GPU on every execution).
 */
const INITIALIZER_OUTPUT: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y", "c"],
  initializers: { c: {} },
  values: {
    c: { dtype: "f32", shape: [ROWS, COLS] },
    y: { dtype: "f32", shape: [ROWS, COLS] },
  },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

const openInitializerOutputModel = () =>
  openGraphModel(INITIALIZER_OUTPUT, [
    {
      graph: "model",
      initializer: "c",
      bytes: f32Bytes(Array.from({ length: COUNT }, (_, i) => i * 0.25)),
      encoding: { codec: "f32" },
    },
  ]);

test.skipIf(!GPU_AVAILABLE)(
  "copyOutputs to a graph output from an initializer fails before queuing and does not take down the rest of the section (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openInitializerOutputModel(),
      "model",
    );
    const healthy = await producerSession(gpu);
    const constSink = await gpu.createResident(BYTES, "const-sink");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      // MUST: await neither (the same form as the existing one that collects failure attribution
      // in finish).
      const pending = [
        // In IR v2 the initializer name is itself the storage key, so `c` becomes the output name
        // as declared.
        session.enqueue({ x: input(0) }, { batch, copyOutputs: { c: constSink } }),
        healthy.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } }),
      ];
      const settled = Promise.allSettled(pending);

      const attributed = await assertRejects(
        () => batch.finish(),
        ExecutionError,
        "not a node output",
      );

      const results = await settled;
      assert(
        results[0].status === "rejected",
        "the returned Promise of the failed enqueue rejects",
      );
      assertStrictEquals(results[0].reason, attributed);
      assertEquals(
        results[1].status,
        "fulfilled",
        "the other enqueue in the same section does not go down with it",
      );
      assertEquals(
        session.diagnostics().submit.dispatchCount,
        0,
        "it did not fail before queuing a single dispatch",
      );
      assertEquals(session.diagnostics().submit.discardedDispatches, 0);
      assertEquals(
        bits(await sink.read()),
        bits(doubled(1)),
        "the healthy enqueue's copy target is updated",
      );
    } finally {
      await session.dispose();
      await healthy.dispose();
      constSink.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);
