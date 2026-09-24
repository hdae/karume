import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  BatchScopeError,
  type GpuContext,
  GpuDeviceLostError,
  ResidentTensor,
  ResidentTensorError,
} from "../src/gpu/device.ts";
import { GpuValidationError } from "../src/gpu/error-scope.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
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
  Deno.test({ name, ignore: !GPU_AVAILABLE, fn });

test("batch read: 未await enqueueと出力snapshotを1map・queue待ち0回で返す", async () => {
  const gpu = await acquireGpu(),
    target = await session(gpu),
    sink = await gpu.createResident(12),
    marker = await gpu.createResident(4),
    decoy = await gpu.createResident(12);
  marker.write(Uint32Array.of(0x7fa01234));
  const fences = countFences(gpu), create = gpu.device.createBuffer.bind(gpu.device);
  let maps = 0, destroyed = 0;
  gpu.device.createBuffer = (desc): GPUBuffer => {
    const buffer = create(desc);
    if (desc.label === "batch-readback") {
      const map = buffer.mapAsync.bind(buffer), destroy = buffer.destroy.bind(buffer);
      buffer.mapAsync = (...args) => {
        maps++;
        return map(...args);
      };
      buffer.destroy = () => {
        destroyed++;
        destroy();
      };
    }
    return buffer;
  };
  try {
    const batch = await gpu.beginBatch();
    const enqueued = target.enqueue({ x0: input() }, { batch, copyOutputs: { y: sink } });
    const selected: Record<string, ResidentTensor> = { a: sink, b: marker };
    Object.defineProperty(selected, "__proto__", { value: marker, enumerable: true });
    const pending = batch.finishAndRead(selected), finish = batch.finish();
    assertStrictEquals(finish, batch.finish());
    assertThrows(() => marker.dispose(), ResidentTensorError);
    selected.a = decoy;
    await assertRejects(() => batch.finishAndRead({ a: sink }), BatchScopeError);
    const result = await pending;
    await Promise.all([enqueued, finish]);
    assertEquals(Array.from(new Float32Array(result.a)), [-1, -2, -3]);
    assertEquals(Array.from(new Uint32Array(result.b)), [0x7fa01234]);
    assertEquals(Array.from(new Uint32Array(result["__proto__"])), [0x7fa01234]);
    assertEquals(Object.keys(result), ["a", "b", "__proto__"]);
    assertEquals([fences.count(), maps, destroyed], [0, 1, 1]);
    assertEquals([sink.useReferences, marker.useReferences], [0, 0]);
    marker.dispose();
  } finally {
    gpu.device.createBuffer = create;
    fences.restore();
    await target.dispose();
    sink.dispose();
    marker.dispose();
    decoy.dispose();
    gpu.destroy();
  }
});

test("batch read: 空集合は従来の完了フェンスで閉じる", async () => {
  const gpu = await acquireGpu(), fences = countFences(gpu);
  try {
    const batch = await gpu.beginBatch();
    assertEquals(await batch.finishAndRead({}), {});
    assertEquals(fences.count(), 1);
  } finally {
    fences.restore();
    gpu.destroy();
  }
});

test("batch read: finish先行・settle中を拒否しsettle後は読める", async () => {
  const gpu = await acquireGpu(), sink = await gpu.createResident(4);
  sink.write(Int32Array.of(91));
  try {
    const first = await gpu.beginBatch();
    await first.finish();
    await assertRejects(() => first.finishAndRead({ sink }), BatchScopeError);
    const queue = gpu.device.queue,
      original = queue.onSubmittedWorkDone.bind(queue),
      gate = Promise.withResolvers<undefined>();
    const second = await gpu.beginBatch();
    queue.onSubmittedWorkDone = () => gate.promise;
    const settling = second.settle();
    try {
      await assertRejects(() => second.finishAndRead({ sink }), BatchScopeError);
      assertEquals(sink.useReferences, 0);
    } finally {
      gate.resolve(undefined);
      queue.onSubmittedWorkDone = original;
      await settling;
    }
    assertEquals(Array.from(new Int32Array((await second.finishAndRead({ sink })).sink)), [91]);
  } finally {
    sink.dispose();
    gpu.destroy();
  }
});

test("batch read: 異なるdeviceと不正memberは発行時に拒否し予約を戻す", async () => {
  const gpu = await acquireGpu(),
    other = await acquireGpu(),
    sink = await gpu.createResident(4),
    foreign = await other.createResident(4);
  const batch = await gpu.beginBatch();
  try {
    await assertRejects(() => batch.finishAndRead({ sink, foreign }), BatchScopeError);
    assertEquals(sink.useReferences, 0);
    assertEquals(batch.finished, false);
    await assertRejects(
      () => Reflect.apply(batch.finishAndRead, batch, [{ sink, wrong: {} }]),
      BatchScopeError,
    );
    assertEquals(sink.useReferences, 0);
    await batch.finishAndRead({ sink });
  } finally {
    await batch.finish();
    sink.dispose();
    foreign.dispose();
    gpu.destroy();
    other.destroy();
  }
});

test("batch read: 破棄済みと合計上限超過も部分予約を漏らさない", async () => {
  const gpu = await acquireGpu(),
    sink = await gpu.createResident(4),
    dead = await gpu.createResident(4);
  dead.dispose();
  const raw = gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.COPY_SRC });
  const oversized = new ResidentTensor(
    gpu,
    987654,
    raw,
    gpu.limits.maxBufferSize,
    "synthetic-limit",
  );
  const batch = await gpu.beginBatch();
  try {
    await assertRejects(() => batch.finishAndRead({ sink, dead }), ResidentTensorError);
    assertEquals(sink.useReferences, 0);
    await assertRejects(
      () => batch.finishAndRead({ sink, oversized }),
      BatchScopeError,
      "maxBufferSize",
    );
    assertEquals([sink.useReferences, oversized.useReferences], [0, 0]);
    assertEquals(batch.finished, false);
  } finally {
    await batch.finish();
    sink.dispose();
    oversized.dispose();
    gpu.destroy();
  }
});

for (const fault of ["create", "copy", "map", "getMappedRange"] as const) {
  test(`batch read: ${fault}例外でscopeと予約・stagingを解放する`, async () => {
    const gpu = await acquireGpu(),
      sink = await gpu.createResident(4),
      create = gpu.device.createBuffer.bind(gpu.device),
      encoder = gpu.device.createCommandEncoder.bind(gpu.device),
      error = new Error(`injected ${fault}`);
    sink.write(Int32Array.of(43));
    let created = 0, destroyed = 0;
    gpu.device.createBuffer = (desc): GPUBuffer => {
      if (desc.label === "batch-readback" && fault === "create") throw error;
      const b = create(desc);
      if (desc.label === "batch-readback") {
        created++;
        const destroy = b.destroy.bind(b);
        b.destroy = () => {
          destroyed++;
          destroy();
        };
        if (fault === "map") b.mapAsync = () => Promise.reject(error);
        if (fault === "getMappedRange") {
          b.getMappedRange = () => {
            throw error;
          };
        }
      }
      return b;
    };
    if (fault === "copy") {
      gpu.device.createCommandEncoder = (desc) => {
        const e = encoder(desc);
        e.copyBufferToBuffer = () => {
          throw error;
        };
        return e;
      };
    }
    try {
      const batch = await gpu.beginBatch();
      const pending = batch.finishAndRead({ sink });
      const caught = await assertRejects(() => pending, Error, `injected ${fault}`);
      assertStrictEquals(caught, error);
      assertEquals(sink.useReferences, 0);
      assertEquals(created, destroyed);
      gpu.device.createBuffer = create;
      gpu.device.createCommandEncoder = encoder;
      const next = await gpu.beginBatch();
      assertEquals(Array.from(new Int32Array((await next.finishAndRead({ sink })).sink)), [43]);
    } finally {
      gpu.device.createBuffer = create;
      gpu.device.createCommandEncoder = encoder;
      sink.dispose();
      gpu.destroy();
    }
  });
}

test("batch read: copyのGPU validationをmap例外で隠さない", async () => {
  const gpu = await acquireGpu();
  const bad = new ResidentTensor(
    gpu,
    987653,
    gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.COPY_DST }),
    4,
    "missing-copy-src",
  );
  try {
    const batch = await gpu.beginBatch();
    await assertRejects(() => batch.finishAndRead({ bad }), GpuValidationError);
    assertEquals(bad.useReferences, 0);
    const next = await gpu.beginBatch();
    await next.finish();
  } finally {
    bad.dispose();
    gpu.destroy();
  }
});

test("batch read: enqueue側の失敗でも成功データを返さない", async () => {
  const gpu = await acquireGpu(), target = await session(gpu), sink = await gpu.createResident(12);
  try {
    const batch = await gpu.beginBatch();
    const wrong = { dtype: "f32", shape: [1], data: Float32Array.of(1) } satisfies Tensor;
    const failure = await assertRejects(() => target.enqueue({ x0: wrong }, { batch }));
    const caught = await assertRejects(() => batch.finishAndRead({ sink }));
    assertStrictEquals(caught, failure);
    assertEquals(sink.useReferences, 0);
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch read: map未決着のdevice lossでもハングせず予約を戻す", async () => {
  const gpu = await acquireGpu(),
    sink = await gpu.createResident(4),
    create = gpu.device.createBuffer.bind(gpu.device),
    entered = Promise.withResolvers<void>();
  gpu.device.createBuffer = (desc): GPUBuffer => {
    const b = create(desc);
    if (desc.label === "batch-readback") {
      b.mapAsync = () => {
        entered.resolve();
        return new Promise<undefined>(() => {});
      };
    }
    return b;
  };
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const batch = await gpu.beginBatch(), pending = batch.finishAndRead({ sink });
    void pending.catch(() => undefined);
    await entered.promise;
    gpu.destroy();
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(Error("device loss timeout")), 5000);
    });
    await assertRejects(() => Promise.race([pending, timeout]), GpuDeviceLostError);
    assertEquals(sink.useReferences, 0);
  } finally {
    clearTimeout(timer);
    gpu.device.createBuffer = create;
    sink.dispose();
    gpu.destroy();
  }
});

test("batch read: GPU失敗とenqueue失敗が重なってもホスト側の原因を保つ", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
  const create = gpu.device.createBuffer.bind(gpu.device);
  const bad = new ResidentTensor(
    gpu,
    987652,
    gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.COPY_DST }),
    4,
    "missing-copy-src",
  );
  gpu.device.createBuffer = (desc): GPUBuffer => {
    const buffer = create(desc);
    if (desc.label === "batch-readback") {
      buffer.mapAsync = () => Promise.reject(new Error("secondary map failure"));
    }
    return buffer;
  };
  try {
    const batch = await gpu.beginBatch();
    const wrong = { dtype: "f32", shape: [1], data: Float32Array.of(1) } satisfies Tensor;
    const failure = await assertRejects(() => target.enqueue({ x0: wrong }, { batch }));
    const caught = await assertRejects(() => batch.finishAndRead({ bad }), GpuValidationError);
    assertStrictEquals(caught.cause, failure);
    assertEquals(bad.useReferences, 0);
  } finally {
    gpu.device.createBuffer = create;
    await target.dispose();
    bad.dispose();
    gpu.destroy();
  }
});
