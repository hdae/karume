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

import { assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, BatchScopeError, type GpuContext } from "../src/gpu/device.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;

/** y = x + x（= 2x）。run と enqueue の両方で使える最小のグラフ。 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

/** `phase` ごとに値が変わる入力（同じ値を配ると取り違えが検出できない）。 */
const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 参照値（実装とは独立に f32 で丸める）。 */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const openSession = (gpu: GpuContext): Promise<Session> =>
  createSession(gpu, openModel(graphModelBuffer(GRAPH)));

Deno.test({
  name: "batch 区間の内側で発行した未 await run は enqueue を fail loudly にする（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const batch = await gpu.beginBatch();
      // MUST: await しない。await すると run が区間ロックを待って**ここで**ハングするので、
      // 検出器が見るのは「積まれたまま決着していない run」の側。
      const pending = session.run({ x: input(0) });
      try {
        await assertRejects(
          () => session.enqueue({ x: input(1) }, { batch }),
          BatchScopeError,
          "未決着の run",
        );
      } finally {
        // 区間を閉じるとロックが返り、積んだままの run が完走できるようになる。
        await batch.finish();
      }
      assertEquals(bits((await pending)["y"].data), bits(doubled(0)), "run 自体は無傷で完走する");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "batch を開く前に発行した未 await run でも同じ閉路になる（真陽性・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      // 区間の**外**で発行しても、run 本体はマイクロタスク 1 段後にしかロックを取りに行かない。
      // 一方 beginBatch は BatchScope のコンストラクタで同期にロックを先取りするので、
      // 同一 tick では常に batch が先 = 現行 doc の「区間中に run を await しない」を
      // 守っていてもハングする形。
      const pending = session.run({ x: input(2) });
      const batch = await gpu.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(1) }, { batch }),
          BatchScopeError,
          "自己デッドロック",
        );
      } finally {
        await batch.finish();
      }
      assertEquals(bits((await pending)["y"].data), bits(doubled(2)));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "正しい列（区間中は enqueue のみ・run は区間の外）は検出器に掛からない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    const sink = await gpu.createResident(COUNT * 4, "sink");
    try {
      const before = await session.run({ x: input(0) });

      const batch = await gpu.beginBatch();
      await session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } });
      // 未 await の enqueue は検出器の対象外（閉路に必要なのは未決着の **run** の方）。
      const queued = session.enqueue({ x: input(2) }, { batch, copyOutputs: { y: sink } });
      await batch.finish();
      await queued;
      assertEquals(bits(await sink.read()), bits(doubled(2)), "最後の enqueue まで実行される");

      // 区間の外の run は前後とも無風（決着した run は簿記に残らない）。
      const after = await session.run({ x: input(0) });
      assertEquals(bits(after["y"].data), bits(before["y"].data));
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});
