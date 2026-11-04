// Real-GPU checks of the in-flight submit bound (ADR 0123).
//
// Pinned: ① a long run's values do not change, and between steps one completion marker appears
// every 64 submits after the previous one; ② a run that exceeds the bound waits for a marker
// between steps (holding the section lock); ③ an enqueue places no markers and does not wait — an
// enqueue of another Session, issued next without await, that reads the previous enqueue's output
// through a resident tensor does not overtake the call order (ADR 0123 decision 4).
//
// Without ③, going back to an implementation that waits in enqueue stays green, since a single
// enqueue's values are still correct. What breaks is only the shape where "another Session's body
// runs during the wait and reads a resident tensor before it is written", with no exception and no
// warning. Placing markers in enqueue slows a batch's generation loop on Deno (a marker stops the
// host until the GPU completes).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type BatchScope, type GpuContext } from "../src/gpu/device.ts";
import {
  IN_FLIGHT_MARKER_INTERVAL,
  IN_FLIGHT_MAX_MARKERS,
  type SubmitPolicy,
} from "../src/gpu/submit.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, openGraphModel } from "./helpers/model-fixture.ts";
import { countFences } from "./helpers/fences.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/** Enough add stages to exceed the bound (the pending markers reach the bound + 1). */
const CHAIN = IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1) + 8;

/** 1 dispatch = 1 submit (so the marker positions follow from the dispatch count). */
const ONE_PER_SUBMIT: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 1,
  minChunkSize: 1,
  maxChunkSize: 1,
};

/** y = x + x + … + x ({@link CHAIN} add stages in a chain). */
const CHAIN_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: Object.fromEntries(
    Array.from({ length: CHAIN }, (_, i) => [
      i === CHAIN - 1 ? "y" : `v${i}`,
      { dtype: "f32", shape: [ROWS, COLS] },
    ]),
  ),
  nodes: Array.from({ length: CHAIN }, (_, i) => ({
    op: "add",
    ins: [i === 0 ? "x" : `v${i - 1}`, "x"],
    outs: [i === CHAIN - 1 ? "y" : `v${i}`],
    attrs: {},
  })),
};

/** w = z * z (receives the output of {@link CHAIN_GRAPH} through a resident tensor). */
const SQUARE_GRAPH: DeclarationJson = {
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

/**
 * An input whose values change with `phase` (with the same values, reading a stale value could
 * not be detected).
 */
const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => ((i + phase * 5) % 9) - 4),
});

/**
 * Reference values (computed by hand, rounding to f32 at every node — independent of the
 * implementation).
 */
const expectedChain = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => {
    let value = x[i];
    for (let step = 0; step < CHAIN; step += 1) value = Math.fround(value + x[i]);
    return value;
  });
};

const expectedSquare = (phase: number): Float32Array<ArrayBuffer> =>
  Float32Array.from(expectedChain(phase), (value) => Math.fround(value * value));

/**
 * Bit comparison (a rounding mix-up is not tolerated). A resident tensor reads back as a plain
 * ArrayBuffer.
 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const session = async (
  gpu: GpuContext,
  json: DeclarationJson,
  submitPolicy?: SubmitPolicy,
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(json), "model", { submitPolicy });

test.skipIf(!GPU_AVAILABLE)(
  "a long run keeps its values and places one completion marker between steps every 64 submits after the previous one (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const chain = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const fences = countFences(gpu);
    try {
      // The 1st run is unbacked (the arena path), the 2nd is the run that creates the backing, the
      // 3rd uses the retained backing. All of them wait between steps.
      for (const phase of [0, 1, 2]) {
        const before = { fences: fences.count(), submits: chain.diagnostics().submit.submitCount };
        const outputs = await chain.run({ x: input(phase) });
        assertEquals(bits(outputs["y"].data), bits(expectedChain(phase)), `phase ${phase}`);
        const submits = chain.diagnostics().submit.submitCount - before.submits;
        assert(
          submits > IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1),
          `the run is not long enough to exceed the bound (${submits} submits — the check would miss)`,
        );
        // A run's fence is the one mapAsync (H-1), so onSubmittedWorkDone is only the completion
        // markers (only the 2nd run, which creates the backing, adds one wait for releases before
        // the allocation). Markers are placed only between steps, so the readback copy queued after
        // the last step (1 copy = 1 submit) is not counted.
        assertEquals(
          fences.count() - before.fences,
          Math.floor((submits - 1) / IN_FLIGHT_MARKER_INTERVAL) + (phase === 1 ? 1 : 0),
          `phase ${phase}: completion markers for ${submits} submits`,
        );
      }
    } finally {
      fences.restore();
      await chain.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "enqueue does not wait — enqueues of another Session, issued for two rounds without await, do not overtake the order through a resident tensor (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const consumer = await session(gpu, SQUARE_GRAPH);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const later = await gpu.createResident(BYTES, "later");
    try {
      // The 1st batch runs with await to bring both Sessions to a derived plan + backing (the path
      // whose body has no await — the bodies run to completion within microtasks in call order).
      const warm = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch: warm, copyOutputs: { y: carrier } });
      await consumer.enqueue({ z: carrier }, { batch: warm, copyOutputs: { w: sink } });
      await warm.finish();
      assertEquals(bits(await sink.read()), bits(expectedSquare(0)), "1st batch");

      // The 2nd batch queues two rounds without await. If the producer waited midway, the
      // consumer's body would run in the meantime, submit first and read the previous carrier. Even
      // waiting at the end, the 2nd-round producer would be held by the 1st round's wait through
      // the same Session's serialization, and only the 2nd-round consumer would run ahead.
      const batch = await gpu.beginBatch();
      await Promise.all([
        producer.enqueue({ x: input(1) }, { batch, copyOutputs: { y: carrier } }),
        consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } }),
        producer.enqueue({ x: input(2) }, { batch, copyOutputs: { y: carrier } }),
        consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: later } }),
      ]);
      await batch.finish();
      assertEquals(bits(await sink.read()), bits(expectedSquare(1)), "2nd batch, 1st round");
      assertEquals(bits(await later.read()), bits(expectedSquare(2)), "2nd batch, 2nd round");
    } finally {
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      later.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Holds the resolution of `onSubmittedWorkDone` (completion markers and fences) by hand. GPU
 * execution and `mapAsync` are not held.
 * MUST: always call `release` in finally (it opens the gate and restores the original surface —
 * otherwise the cleanup flush hangs / it leaks into other tests).
 */
const gateWorkDone = (
  gpu: GpuContext,
): {
  /** The number of held `onSubmittedWorkDone` calls (completion markers and fences). */
  readonly calls: () => number;
  readonly open: () => void;
  readonly release: () => void;
} => {
  const queue = gpu.device.queue;
  const original = queue.onSubmittedWorkDone.bind(queue);
  const gate = Promise.withResolvers<void>();
  let calls = 0;
  Object.defineProperty(queue, "onSubmittedWorkDone", {
    configurable: true,
    writable: true,
    value: () => {
      calls += 1;
      return original().then(() => gate.promise);
    },
  });
  return {
    calls: () => calls,
    open: () => gate.resolve(),
    release: () => {
      gate.resolve();
      Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        writable: true,
        value: original,
      });
    },
  };
};

/** Watches a Promise so whether it has settled can be read from outside. */
const watch = <T>(
  promise: Promise<T>,
): { readonly settled: () => boolean; readonly done: Promise<T> } => {
  let settled = false;
  const done = promise.finally(() => {
    settled = true;
  });
  return { settled: () => settled, done };
};

test.skipIf(!GPU_AVAILABLE)(
  "a run that exceeds the bound waits for a completion marker between steps, does not settle until then, and keeps the section lock (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const chain = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    let gate: ReturnType<typeof gateWorkDone> | undefined;
    let entered: ReturnType<typeof watch<BatchScope>> | undefined;
    try {
      // Bring it to a derived plan + backing (removing the wait for pipeline creation and the
      // backing-creating run's wait for releases before the allocation — a backing is made only on
      // a hit run, so two runs are needed).
      await chain.run({ x: input(0) });
      await chain.run({ x: input(0) });

      gate = gateWorkDone(gpu);
      const running = watch(chain.run({ x: input(1) }));
      // Advance until the run has placed more markers than the bound (markers are placed inside the
      // section lock). Calling beginBatch first would let the batch take the lock before the run's
      // body (which is a microtask away).
      for (let tick = 0; tick < 200 && gate.calls() <= IN_FLIGHT_MAX_MARKERS; tick += 1) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      assert(
        gate.calls() > IN_FLIGHT_MAX_MARKERS,
        "the run did not place more completion markers than the bound",
      );
      // During the wait the run still holds the errorScope section lock — the batch cannot take it
      // (if it could, another Session's enqueue would cut into the same queue during the wait).
      entered = watch(gpu.beginBatch());
      // A run's fence is mapAsync (not held), so a run that did not wait for markers would settle
      // in this time.
      await new Promise((resolve) => setTimeout(resolve, 200));
      assertEquals(
        running.settled(),
        false,
        "the run does not settle until the completion marker resolves",
      );
      assertEquals(
        entered.settled(),
        false,
        "during the wait the batch cannot take the section lock",
      );

      gate.open();
      const outputs = await running.done;
      assertEquals(bits(outputs["y"].data), bits(expectedChain(1)));
    } finally {
      gate?.release();
      // Close the batch before disposing the Session (dispose hangs while the batch holds the
      // section lock).
      if (entered !== undefined) await (await entered.done).finish();
      await chain.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "enqueue places no completion marker and does not wait — even beyond the bound it settles without calling onSubmittedWorkDone (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const producer = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const carrier = await gpu.createResident(BYTES, "carrier");
    let gate: ReturnType<typeof gateWorkDone> | undefined;
    try {
      // Bring it to a derived plan + backing (the path whose body has no await — without waits the
      // body settles within microtasks).
      const warm = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch: warm, copyOutputs: { y: carrier } });
      await warm.finish();

      gate = gateWorkDone(gpu);
      const batch = await gpu.beginBatch();
      const enqueued = watch(
        producer.enqueue({ x: input(1) }, { batch, copyOutputs: { y: carrier } }),
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(
        enqueued.settled(),
        true,
        "even beyond the bound, enqueue does not wait for markers",
      );
      assertEquals(gate.calls(), 0, "enqueue places no completion markers");

      gate.open();
      await enqueued.done;
      await batch.finish();
      assertEquals(bits(await carrier.read()), bits(expectedChain(1)));
    } finally {
      gate?.release();
      await producer.dispose();
      carrier.dispose();
      gpu.destroy();
    }
  },
);
