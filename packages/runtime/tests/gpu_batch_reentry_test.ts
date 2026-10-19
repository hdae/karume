// 受け口の検査を「写しの後」にもう一度行う規律の回帰（executor.ts `#assertEnqueueAdmissible` /
// `run`、device.ts `finishAndRead`）と、フェンスに到達しなかった batch の計測窓の扱い
// （device.ts `#completion`）。
//
// 入力・出力の Record を写す `Object.entries` は利用者の getter を同期で走らせる。その中から同じ
// Session / batch を再入すると、呼び出し時点で済ませた受け口の検査がすり抜ける。ここでは getter
// から再入する形を意図的に作り、①拒否されること ②内側の操作が正しく決着すること を固定する。
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
  const gpu = await acquireGpu(),
    target = await session(gpu),
    sink = await gpu.createResident(12);
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
      "後続 enqueue の入力が決着時の読み戻しに混ざった（沈黙誤値）",
    );
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch enqueue: 入力 getter からの run 再入は自己デッドロックでなく BatchScopeError にする", async () => {
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

test("run: 入力 getter からの dispose 再入は破棄済みとして拒否し、破棄後に backing を作らない", async () => {
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
      "破棄後に backing が確保された",
    );
  } finally {
    gpu.destroy();
  }
});

test("batch finishAndRead: 出力 getter からの再入は写しの後の再検査で拒否し、内側の登録を守る", async () => {
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
      "上書きされた登録の使用予約が返らない",
    );
  } finally {
    a.dispose();
    b.dispose();
    gpu.destroy();
  }
});

// batch 区間 × 未決着 run の自己デッドロック検出器の門。
//
// 閉路は 4 辺で閉じる:「BatchScope 本体が errorScope 区間ロックを握ったまま in-flight リースの
// 返却を待つ → リースは enqueue 本体の finally でしか返らない → enqueue 本体は Session の
// 直列化チェーンで先行 run の決着を待つ → run 本体はその区間ロックを待つ」。`withScopeLock` は
// 「正当な待ち行列」と「再入」を区別できないため再入検出器を置けず、検出しなければ**例外も
// 診断も出ないまま永久にハングする**。ここが固定するのは ①その列が型付き例外になること
// ②batch を開く**前**に発行した run でも同じであること（beginBatch はコンストラクタで同期に
// ロックを先取りするので、同一 tick なら常に batch が先）③正しい使い方は 1 バイトも変わらない
// こと、の 3 点。
//
// NOTE: ハングの門なので、**待ち合わせを await で確かめない**（決着しない列を待つと
// テストランナー自体が返らない）。検出器を通った後は必ず finish() で区間を閉じてから、
// 積んだままの run を回収する。

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;

/** y = x + x（= 2x）。run と enqueue の両方で使える最小のグラフ。 */
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

/** `phase` ごとに値が変わる入力（同じ値を配ると取り違えが検出できない）。 */
const phasedInput = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/** 参照値（実装とは独立に f32 で丸める）。 */
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

test("batch 区間の内側で発行した未 await run は enqueue を fail loudly にする（実 GPU）", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  try {
    const batch = await gpu.beginBatch();
    // MUST: await しない。await すると run が区間ロックを待って**ここで**ハングするので、
    // 検出器が見るのは「積まれたまま決着していない run」の側。
    const pending = target.run({ x: phasedInput(0) });
    try {
      await assertRejects(
        () => target.enqueue({ x: phasedInput(1) }, { batch }),
        BatchScopeError,
        "unsettled run",
      );
    } finally {
      // 区間を閉じるとロックが返り、積んだままの run が完走できるようになる。
      await batch.finish();
    }
    assertEquals(bits((await pending)["y"].data), bits(doubled(0)), "run 自体は無傷で完走する");
  } finally {
    await target.dispose();
    gpu.destroy();
  }
});

test("batch を開く前に発行した未 await run でも同じ閉路になる（真陽性・実 GPU）", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  try {
    // 区間の**外**で発行しても、run 本体はマイクロタスク 1 段後にしかロックを取りに行かない。
    // 一方 beginBatch は BatchScope のコンストラクタで同期にロックを先取りするので、
    // 同一 tick では常に batch が先 = 現行 doc の「区間中に run を await しない」を
    // 守っていてもハングする形。
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

test("正しい列（区間中は enqueue のみ・run は区間の外）は検出器に掛からない（実 GPU）", async () => {
  const gpu = await acquireGpu();
  const target = await openDoubleSession(gpu);
  const sink = await gpu.createResident(COUNT * 4, "sink");
  try {
    const before = await target.run({ x: phasedInput(0) });

    const batch = await gpu.beginBatch();
    await target.enqueue({ x: phasedInput(1) }, { batch, copyOutputs: { y: sink } });
    // 未 await の enqueue は検出器の対象外（閉路に必要なのは未決着の **run** の方）。
    const queued = target.enqueue({ x: phasedInput(2) }, { batch, copyOutputs: { y: sink } });
    await batch.finish();
    await queued;
    assertEquals(bits(await sink.read()), bits(doubled(2)), "最後の enqueue まで実行される");

    // 区間の外の run は前後とも無風（決着した run は簿記に残らない）。
    const after = await target.run({ x: phasedInput(0) });
    assertEquals(bits(after["y"].data), bits(before["y"].data));
  } finally {
    await target.dispose();
    sink.dispose();
    gpu.destroy();
  }
});
