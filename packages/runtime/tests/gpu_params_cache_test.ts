// params バッファの内容アドレスキャッシュ（Session 常駐）。params の全バイトは
// グラフ・node.attrs・解決済み shape の純関数なので、同じ shape で走り直す 2 run 目は
// 1 本も新規確保・転送されないのが不変条件。
//
// この門が見るのは「2 run 目の新規生成が 0・再利用 > 0」と「値が 1 run 目と完全一致」の 2 つ。
// 前者だけだとキャッシュが誤ったバッファを配っても緑になり、後者だけだとキャッシュが黙って
// 外れて毎 dispatch 確保に戻っても緑になる（どちらも例外は出ない）。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** y[T] = Σ_c (x[T,3] · w[3,2] + b[2])。params の形が異なる 3 ノード（matmul / add / sum）。 */
const GRAPH: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "sum"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "proj.weight", storage: { dtype: "f32" } },
    b: { tensor: "proj.bias", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [3, 2] },
    b: { dtype: "f32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 2] },
    g: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T"] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "sum", ins: ["g"], outs: ["y"], attrs: { dim: 1 } },
  ],
};

const modelBytes = (): ArrayBuffer =>
  graphModelBuffer(GRAPH, [
    {
      name: "proj.weight",
      dtype: "F32",
      shape: [3, 2],
      data: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1]),
    },
    { name: "proj.bias", dtype: "F32", shape: [2], data: f32Bytes([0.125, -0.5]) },
  ]);

const input = (rows: number): Tensor => ({
  dtype: "f32",
  shape: [rows, 3],
  data: Float32Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5),
});

Deno.test({
  name: "同じ shape の 2 run 目は params を 1 本も新規生成せず全て再利用する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      assertEquals(session.diagnostics().lastRunParams, undefined, "未実行なら診断は undefined");

      const first = await session.run({ x: input(4) });
      const firstParams = session.diagnostics().lastRunParams;
      assertEquals(firstParams?.reuseCount, 0, "1 run 目は再利用ゼロ");
      assertEquals((firstParams?.allocCount ?? 0) > 0, true, "1 run 目は新規生成がある");
      // 重みアリーナが params の実体も所有する（破棄は weights の dispose に相乗り）。
      const weightsAfterFirst = session.diagnostics().weights.allocCount;

      const second = await session.run({ x: input(4) });
      const secondParams = session.diagnostics().lastRunParams;
      assertEquals(secondParams?.allocCount, 0, "2 run 目の新規生成は 0");
      assertEquals(secondParams?.reuseCount, firstParams?.allocCount, "1 run 目のぶんを全て再利用");
      assertEquals(
        session.diagnostics().weights.allocCount,
        weightsAfterFirst,
        "再利用 run はアリーナに 1 本も足さない",
      );

      // 使い回した params で値が変わらない（キャッシュが別の内容のバッファを配っていない）
      assertEquals(second["y"].shape, first["y"].shape);
      assertEquals(Array.from(second["y"].data), Array.from(first["y"].data));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "shape が変わる run は params を新規生成し、戻ると再び再利用に乗る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4) });
      const narrow = session.diagnostics().lastRunParams?.allocCount ?? 0;

      // 記号次元が変われば解決済み shape が変わり、params の内容も変わる（= 別のキー）。
      await session.run({ x: input(9) }, { T: 9 });
      assertEquals(session.diagnostics().lastRunParams?.allocCount, narrow, "別 shape は作り直す");
      assertEquals(session.diagnostics().lastRunParams?.reuseCount, 0);

      // 先の shape へ戻せば、そのぶんは Session 常駐のまま残っている。
      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: narrow });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
