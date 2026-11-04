/**
 * Checks of `Session.enqueueRead` (ADR 0054 addendum, batched readback of graph outputs).
 *
 * What is checked: ① the outputs have the same values as run and come back with a single map at the
 * end of the batch (1 staging, 0 queue waits); ② `outputs` does not resolve before the batch
 * settles; ③ it joins the same staging as the resident `finishAndRead`; ④ later enqueues of the
 * same Session are rejected until that batch settles; ⑤ an admission failure and a GPU failure of
 * the batch show in `outputs` too, and no success data is returned; ⑥ an enqueueRead whose output
 * aliases a resident input is rejected at admission; ⑦ Session.dispose before the batch settles is
 * rejected before admission closes.
 */
import { test as vitestTest } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { acquireGpu, BatchScopeError, type GpuContext, ResidentTensor } from "../src/gpu/device.ts";
import { GpuValidationError } from "../src/gpu/error-scope.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { countFences } from "./helpers/fences.ts";
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
/** Waits until the next macrotask (to see that `outputs` has not resolved before settling). */
const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));
/** Counts the allocations and maps of the settling staging (label `batch-readback`). */
const countStagings = (gpu: GpuContext): { stagings(): number; maps(): number } => {
  const create = gpu.device.createBuffer.bind(gpu.device);
  let stagings = 0,
    maps = 0;
  gpu.device.createBuffer = (descriptor): GPUBuffer => {
    const buffer = create(descriptor);
    if (descriptor.label === "batch-readback") {
      stagings += 1;
      const map = buffer.mapAsync.bind(buffer);
      buffer.mapAsync = (...args) => {
        maps += 1;
        return map(...args);
      };
    }
    return buffer;
  };
  return { stagings: () => stagings, maps: () => maps };
};

test("batch enqueueRead: reads graph outputs back with a single map at the end and returns the same values as run", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  const fences = countFences(gpu),
    staging = countStagings(gpu);
  try {
    const expected = await target.run({ x0: input() });
    const before = fences.count();
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch });
    let settled = false;
    read.outputs.then(
      () => (settled = true),
      () => (settled = true),
    );
    await read.admitted;
    await tick();
    assertEquals(settled, false, "outputs resolved before the batch settled");
    await batch.finish();
    const outputs = await read.outputs;
    assertEquals(outputs, expected);
    assertEquals(Array.from(outputs.y.data), [-1, -2, -3]);
    assertEquals(fences.count() - before, 0, "a queue wait slipped in");
    assertEquals([staging.stagings(), staging.maps()], [1, 1]);
  } finally {
    fences.restore();
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: joins the single staging of the resident finishAndRead", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu),
    sink = await gpu.createResident(12);
  const staging = countStagings(gpu);
  try {
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch, copyOutputs: { y: sink } });
    const residents = await batch.finishAndRead({ s: sink });
    assertEquals(Array.from(new Float32Array(residents.s)), [-1, -2, -3]);
    assertEquals(Array.from((await read.outputs).y.data), [-1, -2, -3]);
    assertEquals([staging.stagings(), staging.maps()], [1, 1]);
    assertEquals(sink.useReferences, 0);
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: later enqueues of the same Session are rejected until settled, and pass in the next batch", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch });
    await assertRejects(() => target.enqueue({ x0: input() }, { batch }), BatchScopeError);
    const second = target.enqueueRead({ x0: input() }, { batch });
    const failure = await assertRejects(() => second.admitted, BatchScopeError);
    assertStrictEquals(await assertRejects(() => second.outputs), failure);
    await batch.finish();
    assertEquals(Array.from((await read.outputs).y.data), [-1, -2, -3]);
    const next = await gpu.beginBatch();
    await target.enqueue({ x0: input() }, { batch: next });
    await next.finish();
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: an admission failure shows in both admitted and outputs and is attributed to finish too", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    const wrong = { dtype: "f32", shape: [1], data: Float32Array.of(1) } satisfies Tensor;
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: wrong }, { batch });
    const failure = await assertRejects(() => read.admitted);
    assertStrictEquals(await assertRejects(() => batch.finish()), failure);
    assertStrictEquals(await assertRejects(() => read.outputs), failure);
    // A batch that fails without anyone holding outputs is no unhandled rejection either (the
    // settlement is received internally).
    const orphaned = await gpu.beginBatch();
    target.enqueueRead({ x0: wrong }, { batch: orphaned });
    await assertRejects(() => orphaned.finish());
    await tick();
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: a GPU failure of the batch rejects outputs too and returns no success data", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  const bad = new ResidentTensor(
    gpu,
    987653,
    gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.COPY_DST }),
    4,
    "missing-copy-src",
  );
  try {
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch });
    await read.admitted;
    await assertRejects(() => batch.finishAndRead({ bad }), GpuValidationError);
    await assertRejects(() => read.outputs, GpuValidationError);
    assertEquals(bad.useReferences, 0);
    // After a failed batch the same Session can read again (#readBatch is released at settlement).
    const next = await gpu.beginBatch();
    const again = target.enqueueRead({ x0: input() }, { batch: next });
    await next.finish();
    assertEquals(Array.from((await again.outputs).y.data), [-1, -2, -3]);
  } finally {
    bad.dispose();
    await target.dispose();
    gpu.destroy();
  }
});

/** y = reshape(x) ([2,2] → [4]) — a degenerate graph whose output aliases its input. */
const aliasSession = async (gpu: GpuContext) =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(singleOpDeclaration("reshape", [[2, 2]], [[4]])),
    "model",
  );

test("batch enqueueRead: an output aliasing a resident input fails at admission and queues nothing in the batch", async () => {
  const gpu = await acquireGpu(),
    target = await aliasSession(gpu);
  const resident = await gpu.createResident(16, "r");
  try {
    resident.write(Float32Array.of(1, 2, 3, 4));
    const batch = await gpu.beginBatch();
    // The copy runs at settlement, so letting it through would mix writes made after admitted and
    // before finish into the readback (a value different from the one at enqueue time comes back
    // silently).
    const read = target.enqueueRead({ x0: resident }, { batch });
    const failure = await assertRejects(() => read.admitted, ExecutionError, "is an alias of");
    assertEquals(
      [failure.message.includes("'y'"), failure.message.includes("'x0'")],
      [true, true],
      failure.message,
    );
    assertStrictEquals(await assertRejects(() => read.outputs), failure);
    assertStrictEquals(await assertRejects(() => batch.finish()), failure);
    assertEquals(resident.useReferences, 0);
  } finally {
    await target.dispose();
    resident.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: an output aliasing a host input passes and returns the input's value", async () => {
  // The storage of a host input is owned by the backing, and the only thing that can overwrite it
  // before settlement is a later enqueue of the same Session (blocked by #readBatch), so it is
  // allowed.
  const gpu = await acquireGpu(),
    target = await aliasSession(gpu);
  try {
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead(
      { x0: { dtype: "f32", shape: [2, 2], data: Float32Array.of(1, 2, 3, 4) } },
      { batch },
    );
    await read.admitted;
    await batch.finish();
    assertEquals(Array.from((await read.outputs).y.data), [1, 2, 3, 4]);
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: Session.dispose before settlement is rejected before admission closes, and passes after finish", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu);
  try {
    const expected = await target.run({ x0: input() });
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch });
    await read.admitted;
    // Letting it through would have the copy at settlement queue that storage as its source after
    // the dispose body destroyed the slot backing.
    await assertRejects(() => target.dispose(), ExecutionError, "enqueueRead readback");
    await batch.finish();
    const outputs = await read.outputs;
    const u32 = (tensor: Tensor): number[] => {
      const data = tensor.data as Float32Array;
      return Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));
    };
    assertEquals(u32(outputs.y), u32(expected.y));
    await target.dispose();
  } finally {
    // dispose returns the same completion on the second and later calls.
    await target.dispose();
    gpu.destroy();
  }
});
