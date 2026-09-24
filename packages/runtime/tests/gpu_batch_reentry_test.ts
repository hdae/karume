// 受け口の検査を「写しの後」にもう一度行う規律の回帰（executor.ts `#assertEnqueueAdmissible` /
// `run`、device.ts `finishAndRead`）と、フェンスに到達しなかった batch の計測窓の扱い
// （device.ts `#completion`）。
//
// 入力・出力の Record を写す `Object.entries` は利用者の getter を同期で走らせる。その中から同じ
// Session / batch を再入すると、呼び出し時点で済ませた受け口の検査がすり抜ける。ここでは getter
// から再入する形を意図的に作り、①拒否されること ②内側の操作が正しく決着すること を固定する。

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
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { EnqueueRead, RunOutputs } from "../src/runtime/session-types.ts";
import { openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
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

/**
 * 自己デッドロックの回帰は「決着しない」形で現れるので、待ちに上限を置く（timer は必ず解除して
 * sanitizer に残さない）。
 */
const settleWithin = async <T>(work: Promise<T>, label: string): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label}: 5 秒以内に決着しない（自己デッドロック）`)),
      5000,
    );
  });
  try {
    return await Promise.race([work, timeout]);
  } finally {
    clearTimeout(timer);
  }
};

test("batch: 読み戻しがフェンス前に落ちた窓は捨て、submit の推定を更新しない", async () => {
  const gpu = await acquireGpu(), target = await session(gpu), sink = await gpu.createResident(12);
  try {
    const batch = await gpu.beginBatch();
    await target.enqueue({ x0: input() }, { batch, copyOutputs: { y: sink } });
    // 区間の errorScope に validation エラーを 1 つ入れる（MAP_READ は COPY_DST としか組めない —
    // 同期例外にならず scope にだけ現れる）。決着の読み戻しは `mapAsync` より前の scope 検査で落ちる。
    gpu.device.createBuffer({ size: 4, usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.STORAGE });
    await assertRejects(() => batch.finishAndRead({ s: sink }), GpuValidationError);
    const stats = target.diagnostics().submit;
    assertEquals(stats.msPerWorkgroup, undefined, "フェンス前に落ちた窓の実測が推定に入った");
    assertEquals(stats.currentChunkSize, DEFAULT_SUBMIT_POLICY.initialChunkSize);
    assertEquals(sink.useReferences, 0);
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch enqueue: 入力 getter からの enqueueRead 再入は写しの後の再検査で拒否する", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
  try {
    const batch = await gpu.beginBatch();
    let read: EnqueueRead | undefined;
    const outer = target.enqueue({
      get x0(): Tensor {
        read = target.enqueueRead({ x0: input() }, { batch });
        return { dtype: "f32", shape: [3], data: Float32Array.of(10, 20, 30) };
      },
    }, { batch });
    await assertRejects(() => outer, BatchScopeError, "enqueueRead を積んだ batch");
    await batch.finish();
    assertEquals(
      Array.from((await read!.outputs).y.data),
      [-1, -2, -3],
      "後続 enqueue の入力が決着時の読み戻しに混ざった（沈黙誤値）",
    );
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueue: 入力 getter からの run 再入は自己デッドロックでなく BatchScopeError にする", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
  try {
    const batch = await gpu.beginBatch();
    let run: Promise<RunOutputs> | undefined;
    const outer = target.enqueue({
      get x0(): Tensor {
        run = target.run({ x0: input() });
        return input();
      },
    }, { batch });
    await settleWithin(assertRejects(() => outer, BatchScopeError, "未決着の run"), "enqueue");
    await settleWithin(batch.finish(), "finish");
    assertEquals(Array.from((await settleWithin(run!, "run")).y.data), [-1, -2, -3]);
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("run: 入力 getter からの dispose 再入は破棄済みとして拒否し、破棄後に backing を作らない", async () => {
  const gpu = await acquireGpu(), target = await session(gpu);
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
      "dispose 済み",
    );
    await disposal!;
    assertEquals(
      target.diagnostics().planBacking.retainedCount,
      0,
      "破棄後に backing が確保された",
    );
  } finally {
    gpu.destroy();
  }
});

test("batch finishAndRead: 出力 getter からの再入は写しの後の再検査で拒否し、内側の登録を守る", async () => {
  const gpu = await acquireGpu(), a = await gpu.createResident(4), b = await gpu.createResident(4);
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
    await assertRejects(() => outer, BatchScopeError, "1回だけ");
    const read = await inner!;
    assertEquals(Array.from(new Float32Array(read.i)), [2]);
    assertEquals(
      [a.useReferences, b.useReferences],
      [0, 0],
      "上書きされた登録の使用予約が返らない",
    );
  } finally {
    a.dispose();
    b.dispose();
    gpu.destroy();
  }
});
