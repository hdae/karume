/**
 * `Session.enqueueRead`（ADR 0054 追記〈グラフ出力の一括読み戻し〉）の門。
 *
 * 見るのは ①出力が run と同じ値で、batch の終端 map 1 回（staging 1 本・queue 待ち 0 回）で
 * 返ること ②`outputs` は決着前に解決しないこと ③常駐の `finishAndRead` と同じ staging に
 * 連結すること ④同じ Session の後続 enqueue をその batch の決着まで拒むこと ⑤受理の失敗と
 * 区間の GPU 失敗が `outputs` にも出て、成功データを返さないこと。
 */
import { assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { acquireGpu, BatchScopeError, type GpuContext, ResidentTensor } from "../src/gpu/device.ts";
import { GpuValidationError } from "../src/gpu/error-scope.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { openModel } from "../src/format/container.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { countFences } from "./helpers/fences.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const input = (): Tensor => ({ dtype: "f32", shape: [3], data: Float32Array.of(1, 2, 3) });
const session = (gpu: GpuContext) =>
  createSession(gpu, openModel(graphModelBuffer(singleOpGraph("neg", [[3]], [[3]]))));
const test = (name: string, fn: () => Promise<void>): void =>
  Deno.test({ name, ignore: !GPU_AVAILABLE, fn });
/** 次のマクロタスクまで待つ（決着前の `outputs` が解決していないことを見るため）。 */
const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));
/** 決着の staging（label `batch-readback`）の確保数と map 数を数える。 */
const countStagings = (gpu: GpuContext): { stagings(): number; maps(): number } => {
  const create = gpu.device.createBuffer.bind(gpu.device);
  let stagings = 0, maps = 0;
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

test("batch enqueueRead: グラフ出力を終端の map 1 回で読み戻し run と同じ値を返す", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
  const fences = countFences(gpu), staging = countStagings(gpu);
  try {
    const expected = await target.run({ x0: input() });
    const before = fences.count();
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: input() }, { batch });
    let settled = false;
    read.outputs.then(() => (settled = true), () => (settled = true));
    await read.admitted;
    await tick();
    assertEquals(settled, false, "決着前に outputs が解決した");
    await batch.finish();
    const outputs = await read.outputs;
    assertEquals(outputs, expected);
    assertEquals(Array.from(outputs.y.data), [-1, -2, -3]);
    assertEquals(fences.count() - before, 0, "queue 待ちが混ざった");
    assertEquals([staging.stagings(), staging.maps()], [1, 1]);
  } finally {
    fences.restore();
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: finishAndRead の常駐と同じ 1 本の staging に連結する", async () => {
  const gpu = await acquireGpu(), target = await session(gpu), sink = await gpu.createResident(12);
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

test("batch enqueueRead: 同じ Session の後続 enqueue は決着まで拒否し、次の batch では通る", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
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

test("batch enqueueRead: 受理の失敗は admitted と outputs の両方に出て finish にも帰属する", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
  try {
    const wrong = { dtype: "f32", shape: [1], data: Float32Array.of(1) } satisfies Tensor;
    const batch = await gpu.beginBatch();
    const read = target.enqueueRead({ x0: wrong }, { batch });
    const failure = await assertRejects(() => read.admitted);
    assertStrictEquals(await assertRejects(() => batch.finish()), failure);
    assertStrictEquals(await assertRejects(() => read.outputs), failure);
    // outputs を握らないまま落ちる区間も未処理拒否にならない（内部で決着を受け取っている）。
    const orphaned = await gpu.beginBatch();
    target.enqueueRead({ x0: wrong }, { batch: orphaned });
    await assertRejects(() => orphaned.finish());
    await tick();
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueueRead: 区間の GPU 失敗では outputs も拒否し成功データを返さない", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
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
    // 失敗した区間の後も同じ Session で読める（#readBatch が決着で解けている）。
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
