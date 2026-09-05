// 見積りの中間バッファ規則（`src/runtime/estimate.ts` の `transientSlotBytes`）の門。
// GPU を 1 つも触らない純関数テスト。
//
// 縛るのは 1 点 —— 中間の総量は**計画の領域の総和**（ADR 0093: 生存区間の first-fit 配置・
// 256 バイト整列・同じ dispatch で読み書きされる 2 本は別領域）であって「同時生存バイトの最大」
// では**ない**こと。規則の正本は `planTransients` 1 本だが、見積りが組む確保プログラム（1 ノード
// 1 dispatch の近似）が実行相とずれたときの症状は沈黙（例外も警告も出ず、数字だけがずれる）
// なので、算式そのものを手計算の定数で押さえる。

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

Deno.test("中間の総量は領域の総和（読み書きの同居禁止で領域が割れ、配り直しはサイズに依らない）", () => {
  // 8（h1）+ 12（h2）+ 4（h3）。h2 は h1 を読む dispatch が書くので別領域（12 / 8）。h3 は h2 を
  // 読むので h2 の領域には置けず、生存を終えた h1 の区間（8 バイト）を掴む = 12 + 8 = 20
  // （旧 exact-size プールでは 3 本累積の 24 だった）。
  assertEquals(workspaceBytes(chainModel()), 20);
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

Deno.test("生存を終えた区間は配り直され、総量が増えない", () => {
  // 中間 3 本（8B × 3）。h2 は h1 を読むので別領域、h3 は h2 を読むので h1 側の領域へ戻り、
  // 生存を終えた h1 の区間を掴む = 領域 2 本 × 8 = 16。
  assertEquals(workspaceBytes(reluChainModel(["h3"])), 16);
});

Deno.test("グラフ出力（pinned）は生存を終えず、後続は整列した次の区間へ回る", () => {
  // h1 も出力にすると、その区間は refs が 0 になっても生存を終えない。h3 は h1 側の領域で
  // h1 と重なるので 256 整列の次の offset（256）へ置かれ、領域は 264 + h2 の 8 = 272
  // （offset 整列は device の minStorageBufferOffsetAlignment・仕様既定 256 — ADR 0093 決定 1）。
  assertEquals(workspaceBytes(reluChainModel(["h1", "h3"])), 272);
});
