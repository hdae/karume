// params バッファの内容アドレスキャッシュ（Session 常駐）。params の全バイトは
// グラフ・node.attrs・解決済み shape の純関数なので、同じ内容の params は run 内・run 跨ぎの
// どちらでも 1 本しか作られないのが不変条件。
//
// NOTE: 同一 bindings の 2 run 目は導出済み計画（gpu_prepared_plan_test.ts）に当たり、導出相
// ごと飛ぶので #writeParams に到達しない（lastRunParams は {0,0}）。したがってこの門は
// ①同一 run 内の重複 params（同形ノード 2 本）と、②導出済み計画から**追い出された後**の
// 再導出 run（prepared ミス・params は全ヒット）の 2 点で再利用を観測する — 2 つのキャッシュの
// 寿命が独立であること（params は無上限・prepared は LRU 上限あり）もここで固定される。
//
// この門が見るのは「新規生成 0 で再利用 > 0」と「値が完全一致」の 2 つ。前者だけだと
// キャッシュが誤ったバッファを配っても緑になり、後者だけだとキャッシュが黙って外れて
// 毎 dispatch 確保に戻っても緑になる（どちらも例外は出ない）。

import { assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { f32Bytes, type GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * y[T] = Σ_c ((x[T,3] · w[3,2] + b[2]) + b[2])。params の形が異なる 3 ノード（matmul / add /
 * sum）に加えて、**同形の add をもう 1 本**置く — 2 本目の add は params の全バイトが
 * 1 本目と一致するので、1 run 目の中で再利用が 1 件観測できる。
 */
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
    g2: { dtype: "f32", shape: ["T", 2] },
    y: { dtype: "f32", shape: ["T"] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "add", ins: ["g", "b"], outs: ["g2"], attrs: {} },
    { op: "sum", ins: ["g2"], outs: ["y"], attrs: { dim: 1 } },
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
  name: "同じ内容の params は run 内で 1 本に畳まれ、ヒット run は 1 本も触らない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      assertEquals(session.diagnostics().lastRunParams, undefined, "未実行なら診断は undefined");

      const first = await session.run({ x: input(4) });
      const firstParams = session.diagnostics().lastRunParams;
      assertEquals(firstParams?.reuseCount, 1, "同形 add の 2 本目は 1 run 目の中で再利用");
      assertEquals((firstParams?.allocCount ?? 0) > 0, true, "1 run 目は新規生成がある");
      // 重みアリーナが params の実体も所有する（破棄は weights の dispose に相乗り）。
      const weightsAfterFirst = session.diagnostics().weights.allocCount;

      // 2 run 目は導出済み計画に当たり導出相ごと飛ぶ — params の GPU 操作もゼロ
      // （キャッシュが外れたのではなく、レシピが params 実体を直参照している）。
      const second = await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().lastRunParams, { allocCount: 0, reuseCount: 0 });
      assertEquals(
        session.diagnostics().weights.allocCount,
        weightsAfterFirst,
        "ヒット run はアリーナに 1 本も足さない",
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
  name: "導出済み計画から追い出されても params キャッシュは生き残り全ヒットする（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      await session.run({ x: input(4) });
      const narrow = session.diagnostics().lastRunParams;
      const narrowAlloc = narrow?.allocCount ?? 0;

      // 記号次元が変われば解決済み shape が変わり、params の内容も変わる（= 別のキー）。
      await session.run({ x: input(9) }, { T: 9 });
      assertEquals(
        session.diagnostics().lastRunParams?.allocCount,
        narrowAlloc,
        "別 shape は作り直す",
      );

      // T=4 を導出済み計画（LRU 上限 4）から確実に追い出す。params キャッシュは無上限なので
      // ここで作られた params が消えることはない。
      for (const rows of [5, 6, 7, 8, 10]) {
        await session.run({ x: input(rows) });
      }

      // 再導出 run（prepared はミス）だが params は 1 本も作り直さない — 1 run 目の実体が
      // Session 常駐のまま残っている（+1 は同形 add の run 内再利用）。
      await session.run({ x: input(4) });
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "計画は追い出し済み");
      assertEquals(session.diagnostics().lastRunParams, {
        allocCount: 0,
        reuseCount: narrowAlloc + 1,
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * params キャッシュが **Session 寿命で無界**であること（by-design — docs/limitations.md）を
 * 固定する。追い出し機構が入れば赤くなるので、入れるときはこの門ごと裁定し直すこと。
 *
 * 同範囲の他キャッシュ（`SubmitScheduler.MEASURED_HISTORY` / `PREPARED_PLAN_CAPACITY`）は
 * 「MUST: 有界」を明示しており、ここだけが規律の外にある。数値で固定しておかないと、
 * 「可変 shape を長時間回すと GPU バッファが単調増加する」という制約が挙動からは見えない
 * （例外も警告も出ず、`diagnostics().weights.allocCount` が伸びるだけ）。
 */
Deno.test({
  name: "params キャッシュは Session 寿命で無界（可変 shape の run ごとに単調増加・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(modelBytes()));
    try {
      // initializer 2 本ぶんだけが載った状態から始まる。
      assertEquals(session.diagnostics().weights.allocCount, 2);

      // 記号次元 T を毎回変える = params の内容が毎回変わる = 1 度も当たらない。
      // 導出済み計画（LRU 上限 4）からは追い出されるが、params の実体は 1 本も返らない。
      const RUNS = 5;
      for (let i = 0; i < RUNS; i += 1) {
        await session.run({ x: input(2 + i) });
        assertEquals(
          session.diagnostics().lastRunParams,
          { allocCount: 3, reuseCount: 1 },
          `run ${i}: 3 種の params を新規生成（同形 add の 2 本目だけ run 内再利用）`,
        );
      }

      assertEquals(
        session.diagnostics().weights.allocCount,
        2 + RUNS * 3,
        "追い出しは 1 つも無い（initializer 2 本 + run 数 × ノード種 3）",
      );
      assertEquals(
        session.diagnostics().lastRunPrepared?.cachedPlans,
        4,
        "導出済み計画だけが上限で頭打ちになる（params キャッシュとは寿命が独立）",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
