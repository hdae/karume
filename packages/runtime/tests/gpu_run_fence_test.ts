// 通常 run のフェンス構成（H-1）の門。
//
// 狙いは「run 1 本の待ちを mapAsync の 1 本に畳む」ことなので、この門が固定するのは
// ①gpuTiming OFF の run が `onSubmittedWorkDone` を **1 本も張らない**こと ②それでも値が
// 二段待ち経路とビット単位で一致すること ③二段待ちへ落ちる 2 条件（gpuTiming ON / グラフ出力
// 0 本）では**黙って無フェンスにしない**こと ④submit だけして待たないアリーナ後始末が
// 「未 submit を残して destroy」に退化していないこと。
//
// ① が無いと、内部で flush へ戻しても値は正しいまま緑になる（例外も警告も出ない）。
// ③ の出力 0 本は、copy が 1 件も積まれない = 待つ相手が居ない形で、単一フェンス経路のまま
// 通すと GPU 実行の完了前に run が戻る。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { RunArena, STORAGE_USAGE } from "../src/gpu/arena.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import { countFences } from "./helpers/fences.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;

/** y = x + x、s = y * y。**出力 2 本**（mapAsync を並列に待つ形を実際に通す）。 */
const graph = (outputs: readonly string[]): GraphJson => ({
  format: "karume-ir",
  version: 1,
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
/** 出力 0 本（IR は空の `graph.outputs` を受理する）。 */
const NO_OUTPUT = graph([]);

const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 参照値（ノードごとに f32 へ丸めながら手計算する — 実装とは独立）。 */
const expected = (phase: number): {
  readonly y: Float32Array<ArrayBuffer>;
  readonly s: Float32Array<ArrayBuffer>;
} => {
  const x = input(phase).data;
  const y = Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
  return { y, s: Float32Array.from(y, (value) => Math.fround(value * value)) };
};

/** ビット列比較（丸めの取り違えを許容しない）。 */
const bits = (data: Tensor["data"] | Float32Array<ArrayBuffer>): readonly number[] =>
  Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));

const session = (gpu: GpuContext, json: GraphJson): Promise<Session> =>
  createSession(gpu, openModel(graphModelBuffer(json)));

Deno.test({
  name: "gpuTiming OFF の run はフェンスを mapAsync 1 本に畳む（onSubmittedWorkDone 0 回・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const target = await session(gpu, TWO_OUTPUTS);
    // 計数は Session 構築（重みアップロードの 1 本）の**後**から始める。
    const fences = countFences(gpu);
    try {
      // 1 run 目は非 backed（アリーナ経路）、2 run 目は backed。どちらもフェンスは 0 本。
      for (const phase of [0, 1]) {
        const outputs = await target.run({ x: input(phase) });
        const reference = expected(phase);
        assertEquals(bits(outputs["y"].data), bits(reference.y), `phase ${phase} の y`);
        assertEquals(bits(outputs["s"].data), bits(reference.s), `phase ${phase} の s`);
        assertEquals(
          fences.count(),
          0,
          `run ${phase + 1} 本目で onSubmittedWorkDone が張られている（二段待ちへ退化）`,
        );
      }
      assertEquals(target.diagnostics().lastRunPrepared?.hit, true, "2 run 目は導出済み計画");
      assertEquals(target.diagnostics().lastRunTiming, undefined, "計測は無効な device");
      // 計測窓は run ごとに 1 本閉じている（フェンスを畳んでも適応制御の材料は生きている）。
      assertEquals(target.diagnostics().submit.measuredCount, 2, "run ごとに窓が 1 本閉じる");
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "gpuTiming ON の run は二段待ちのまま（timestamp 回収が flush に乗る・実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    const target = await session(gpu, TWO_OUTPUTS);
    const fences = countFences(gpu);
    try {
      const outputs = await target.run({ x: input(0) });
      const reference = expected(0);
      // 計測モードでも値は単一フェンス経路とビット一致（1 dispatch = 1 pass は意味論を変えない）。
      assertEquals(bits(outputs["y"].data), bits(reference.y));
      assertEquals(bits(outputs["s"].data), bits(reference.s));
      // run 本体の flush と arena.destroy の flush で 2 本（H-1 前の構成そのまま）。
      assertEquals(fences.count(), 2, "gpuTiming ON では二段待ちを据え置く");
      const timing = target.diagnostics().lastRunTiming;
      assert(timing !== undefined, "計測が有効なら内訳は必ず取れる");
      assert(timing.entries.length > 0, "内訳が空（回収が flush から外れている）");
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "グラフ出力 0 本の run は従来のフェンスへ落ちる（無フェンスで返さない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const target = await session(gpu, NO_OUTPUT);
    const fences = countFences(gpu);
    try {
      const outputs = await target.run({ x: input(0) });
      assertEquals(Object.keys(outputs).length, 0, "読み戻す出力が無い");
      // 待つ相手（積んだ copy）が 1 件も無いので、mapAsync はフェンスになりえない。
      assert(
        fences.count() > 0,
        "出力 0 本の run が onSubmittedWorkDone を 1 本も張っていない（GPU 実行の完了前に返る）",
      );
    } finally {
      fences.restore();
      await target.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "submit だけして待たない ArenaFlush でも destroy 前に未 submit を残さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 単一フェンス run が RunArena へ渡す後始末そのもの（`submitPending` のみ・完了は待たない）。
    // 故障注入は 2 方向:
    // ①closure から `submitPending()` を抜く → copy が未 submit のまま捨てられ、読み戻しが
    //   全 0 になる（submitCount も 0 のまま）。
    // ②submit を destroy の後に回す → コピー元が破棄済みでエンコードが validation で落ち、
    //   errorScope に残る。
    const gpu = await acquireGpu();
    const scheduler = new SubmitScheduler(gpu);
    const arena = new RunArena(gpu.device, () => {
      scheduler.submitPending();
      return Promise.resolve();
    });
    const values = Float32Array.from({ length: COUNT }, (_, i) => i * 1.5 - 3);
    // 読み戻し先はアリーナの**外**に置く（アリーナ所有だと destroy で消えて内容を確かめられない）。
    const staging = gpu.device.createBuffer({
      size: values.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const source = arena.allocHostWritten(values.byteLength, STORAGE_USAGE);
      gpu.device.queue.writeBuffer(source, 0, values);
      // 未 submit のまま destroy を呼ぶ（積んだ copy を出し切るのは注入した closure の責務）。
      scheduler.copyBuffer(source, 0, staging, 0, values.byteLength);
      assertEquals(scheduler.stats.submitCount, 0, "この時点では 1 度も submit していない");

      gpu.device.pushErrorScope("validation");
      await arena.destroy();
      assertEquals(scheduler.stats.submitCount, 1, "destroy の後始末が copy を submit した");
      assertEquals(
        await gpu.device.popErrorScope(),
        null,
        "破棄済みバッファを参照したまま submit している（submit が destroy の後に回っている）",
      );

      await gpu[RUNTIME_INTERNAL].raceDeviceLost(staging.mapAsync(GPUMapMode.READ), "検算");
      assertEquals(
        bits(new Float32Array(staging.getMappedRange().slice(0))),
        bits(values),
        "積んだ copy が実行されていない（未 submit のまま捨てられた）",
      );
      staging.unmap();
    } finally {
      staging.destroy();
      gpu.destroy();
    }
  },
});
