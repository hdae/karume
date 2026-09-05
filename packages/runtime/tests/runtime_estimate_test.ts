// 見積りの中間バッファ規則（`src/runtime/estimate.ts` の `transientSlotBytes`）の門。
// GPU を 1 つも触らない純関数テスト。
//
// 縛るのは 1 点 —— **exact-size プールの slot を累積した総量**であって「同時生存バイトの最大」
// では**ない**こと。規則の正本は 3 写し（`RunArena` / `derivePlanSlots` / `transientSlotBytes`）
// にあり、割れたときの症状は沈黙（例外も警告も出ず、数字だけがずれる）なので、算式そのものを
// 手計算の定数で押さえる。

import { assertEquals } from "@std/assert";
import { type KarumeModel, openModel } from "../src/format/container.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";

const openGraph = (graph: GraphJson, tensors: readonly TensorSpec[] = []): KarumeModel =>
  openModel(graphModelBuffer(graph, tensors));

const workspaceBytes = (model: KarumeModel): number => {
  const report = estimateSessionMemory(model);
  assertEquals(report.scenarios.map((scenario) => scenario.name), ["run"]);
  return report.scenarios[0].workspaceBytes;
};

/**
 * `x[1,2] → matmul(w1[2,2]) → h1[1,2] → matmul(w2[2,3]) → h2[1,3] → matmul(w3[3,1]) → h3[1,1]`
 * の 3 段。中間は 8 / 12 / 4 バイトで、**同時に生きるのは高々 2 本**（8+12 = 20）。
 */
const chainModel = (): KarumeModel =>
  openGraph({
    format: "karume-ir",
    version: 1,
    requires: { ops: ["matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
    outputs: ["h3"],
    initializers: {
      w1: { tensor: "m.w1", storage: { dtype: "f32" } },
      w2: { tensor: "m.w2", storage: { dtype: "f32" } },
      w3: { tensor: "m.w3", storage: { dtype: "f32" } },
    },
    values: {
      w1: { dtype: "f32", shape: [2, 2] },
      w2: { dtype: "f32", shape: [2, 3] },
      w3: { dtype: "f32", shape: [3, 1] },
      h1: { dtype: "f32", shape: [1, 2] },
      h2: { dtype: "f32", shape: [1, 3] },
      h3: { dtype: "f32", shape: [1, 1] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w1"], outs: ["h1"], attrs: {} },
      { op: "matmul", ins: ["h1", "w2"], outs: ["h2"], attrs: {} },
      { op: "matmul", ins: ["h2", "w3"], outs: ["h3"], attrs: {} },
    ],
  }, [
    { name: "m.w1", dtype: "F32", shape: [2, 2], data: f32Bytes(new Array(4).fill(1)) },
    { name: "m.w2", dtype: "F32", shape: [2, 3], data: f32Bytes(new Array(6).fill(1)) },
    { name: "m.w3", dtype: "F32", shape: [3, 1], data: f32Bytes(new Array(3).fill(1)) },
  ]);

Deno.test("中間の総量は slot の累積であって同時生存バイトの最大ではない", () => {
  // 8（h1）+ 12（h2）+ 4（h3）。サイズクラスが 3 種とも違うので 1 本も再利用されない。
  // 生存ピークは 20（h1 と h2 が同時に生きる瞬間）で、24 との差がこの規則そのもの —— ここが
  // 20 になったら best-fit / offset 配置へ規則が変わったということ。
  assertEquals(workspaceBytes(chainModel()), 24);
});

/**
 * `relu` の鎖（形が動かないので中間は全て同じサイズクラス）。`outputs` を差し替えて
 * 「pinned が 1 本増えると slot が 1 本増える」を対にして見る。
 */
const reluChainModel = (outputs: readonly string[]): KarumeModel =>
  openGraph({
    format: "karume-ir",
    version: 1,
    requires: { ops: ["relu"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
    outputs: [...outputs],
    initializers: {},
    values: {
      h1: { dtype: "f32", shape: [1, 2] },
      h2: { dtype: "f32", shape: [1, 2] },
      h3: { dtype: "f32", shape: [1, 2] },
    },
    nodes: [
      { op: "relu", ins: ["x"], outs: ["h1"], attrs: {} },
      { op: "relu", ins: ["h1"], outs: ["h2"], attrs: {} },
      { op: "relu", ins: ["h2"], outs: ["h3"], attrs: {} },
    ],
  });

Deno.test("同じサイズクラスの 2 本目はプールから配り直され、総量が増えない", () => {
  // 中間 3 本（8B × 3）だが、h3 は消費済みの h1 の slot を掴むので実確保は 2 本。
  assertEquals(workspaceBytes(reluChainModel(["h3"])), 16);
});

Deno.test("グラフ出力（pinned）はプールへ戻らず、1 本増やすと slot が 1 本ぶん増える", () => {
  // h1 も出力にすると、その slot は refs が 0 になってもプールへ返らない。h3 は配り直しを
  // 受けられず新しい slot が生え、8 バイトちょうど増える。
  assertEquals(workspaceBytes(reluChainModel(["h1", "h3"])), 24);
});
