// Checks for the fence layout of an ordinary run (H-1).
//
// The aim is "fold the wait of one run into a single mapAsync", so these checks pin
// ① a run with gpuTiming OFF sets **no** `onSubmittedWorkDone` at all (the only exception is the
// one wait for releases before allocation in a run that creates a new backing — once per signature)
// ② the values are still bit-identical to the two-stage wait path ③ the two conditions that fall
// back to the two-stage wait (gpuTiming ON / zero graph outputs) **do not silently go fenceless**
// ④ the arena cleanup that only submits without waiting has not regressed into "destroy while
// leaving unsubmitted work".
//
// Without ①, an internal return to flush would still pass with correct values (no exception, no
// warning). The zero-output case of ③ is the shape where no copy is queued = there is nothing to
// wait on; passed through the single-fence path, the run would return before the GPU finishes.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { RunArena, STORAGE_USAGE } from "../src/gpu/arena.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, openGraphModel } from "./helpers/model-fixture.ts";
import { countFences } from "./helpers/fences.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;

/** y = x + x, s = y * y. **Two outputs** (actually exercises waiting on mapAsync in parallel). */
const graph = (outputs: readonly string[]): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: [...outputs],
  initializers: {},
  values: {
    y: { dtype: "f32", shape: [ROWS, COLS] },
    s: { dtype: "f32", shape: [ROWS, COLS] },
  },
  nodes: [
    { op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} },
    { op: "mul", ins: ["y", "y"], outs: ["s"], attrs: {} },
  ],
});

const TWO_OUTPUTS = graph(["y", "s"]);
/** Zero outputs (the IR accepts an empty `graph.outputs`). */
const NO_OUTPUT = graph([]);

const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/**
 * Reference values (computed by hand, rounding to f32 at every node — independent of the
 * implementation).
 */
const expected = (
  phase: number,
): {
  readonly y: Float32Array<ArrayBuffer>;
  readonly s: Float32Array<ArrayBuffer>;
} => {
  const x = input(phase).data;
  const y = Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
  return { y, s: Float32Array.from(y, (value) => Math.fround(value * value)) };
};

/** Bit-pattern comparison (does not tolerate a rounding mix-up). */
const bits = (data: Tensor["data"] | Float32Array<ArrayBuffer>): readonly number[] =>
  Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));

const session = async (gpu: GpuContext, json: DeclarationJson): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(json), "model");

test.skipIf(!GPU_AVAILABLE)(
  "a run with gpuTiming OFF folds its fence into a single mapAsync (zero onSubmittedWorkDone calls, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const target = await session(gpu, TWO_OUTPUTS);
    // Start counting **after** the Session is built (which sets one for the weight upload).
    const fences = countFences(gpu);
    try {
      // Run 1 is not backed (the arena path), run 2 creates the backing, run 3 uses the retained
      // backing. Runs set no fence; the only one set is run 2's wait for releases before allocation
      // (counted cumulatively).
      const waits = [0, 1, 1];
      for (const phase of [0, 1, 2]) {
        const outputs = await target.run({ x: input(phase) });
        const reference = expected(phase);
        assertEquals(bits(outputs["y"].data), bits(reference.y), `y of phase ${phase}`);
        assertEquals(bits(outputs["s"].data), bits(reference.s), `s of phase ${phase}`);
        assertEquals(
          fences.count(),
          waits[phase],
          `onSubmittedWorkDone calls through run ${phase + 1} (regression to the two-stage wait / missing release wait)`,
        );
      }
      assertEquals(target.diagnostics().lastRunPrepared?.hit, true, "run 3 uses the derived plan");
      assertEquals(target.diagnostics().lastRunTiming, undefined, "a device with timing disabled");
      // One measurement window closes per run (folding the fence keeps the input of the adaptive
      // control alive).
      assertEquals(target.diagnostics().submit.measuredCount, 3, "one window closes per run");
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE)(
  "a run with gpuTiming ON keeps the two-stage wait (timestamp collection rides on flush, real GPU)",
  async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    const target = await session(gpu, TWO_OUTPUTS);
    const fences = countFences(gpu);
    try {
      const outputs = await target.run({ x: input(0) });
      const reference = expected(0);
      // In timing mode the values are bit-identical to the single-fence path too (1 dispatch =
      // 1 pass does not change semantics).
      assertEquals(bits(outputs["y"].data), bits(reference.y));
      assertEquals(bits(outputs["s"].data), bits(reference.s));
      // Two: the flush of the run itself and the flush of arena.destroy (the layout before H-1, as
      // is).
      assertEquals(fences.count(), 2, "gpuTiming ON keeps the two-stage wait");
      const timing = target.diagnostics().lastRunTiming;
      assert(timing !== undefined, "with timing enabled the breakdown is always available");
      assert(timing.entries.length > 0, "the breakdown is empty (collection left the flush)");
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a run with zero graph outputs falls back to the original fence (does not return fenceless, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const target = await session(gpu, NO_OUTPUT);
    const fences = countFences(gpu);
    try {
      const outputs = await target.run({ x: input(0) });
      assertEquals(Object.keys(outputs).length, 0, "there is no output to read back");
      // There is nothing to wait on (no queued copy), so mapAsync cannot serve as the fence.
      assert(
        fences.count() > 0,
        "a zero-output run set no onSubmittedWorkDone (it returns before the GPU finishes)",
      );
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "an ArenaFlush that only submits without waiting leaves nothing unsubmitted before destroy (real GPU)",
  async () => {
    // Exactly the cleanup a single-fence run passes to RunArena (`submitPending` only; it does not
    // wait for completion). Fault injection goes two ways:
    // ① drop `submitPending()` from the closure → the copy is discarded unsubmitted and the
    //   readback is all zeros (submitCount stays 0 too).
    // ② move the submit after destroy → the copy source is already destroyed, the encoding fails
    //   validation and stays in the errorScope.
    const gpu = await acquireGpu();
    const scheduler = new SubmitScheduler(gpu);
    const arena = new RunArena(gpu.device, () => {
      scheduler.submitPending();
      return Promise.resolve();
    });
    const values = Float32Array.from({ length: COUNT }, (_, i) => i * 1.5 - 3);
    // Keep the readback target **outside** the arena (owned by the arena, it would vanish on
    // destroy and its contents could not be checked).
    const staging = gpu.device.createBuffer({
      size: values.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const source = arena.allocHostWritten(values.byteLength, STORAGE_USAGE);
      gpu.device.queue.writeBuffer(source, 0, values);
      // Call destroy while unsubmitted (flushing the queued copy is the injected closure's job).
      scheduler.copyBuffer(source, 0, staging, 0, values.byteLength);
      assertEquals(scheduler.stats.submitCount, 0, "nothing has been submitted at this point");

      gpu.device.pushErrorScope("validation");
      await arena.destroy();
      assertEquals(scheduler.stats.submitCount, 1, "the destroy cleanup submitted the copy");
      assertEquals(
        await gpu.device.popErrorScope(),
        null,
        "submitted while still referencing a destroyed buffer (the submit ran after destroy)",
      );

      await gpu[RUNTIME_INTERNAL].raceDeviceLost(
        staging.mapAsync(GPUMapMode.READ),
        "readback check",
      );
      assertEquals(
        bits(new Float32Array(staging.getMappedRange().slice(0))),
        bits(values),
        "the queued copy did not run (it was discarded unsubmitted)",
      );
      staging.unmap();
    } finally {
      staging.destroy();
      gpu.destroy();
    }
  },
);
