// Session ライフサイクル系テストの共有フィクスチャ（runtime_executor_test.ts と
// gpu_runtime_executor_test.ts が同じ鎖グラフを使う）。宣言は IR v2、実体は容器
// （`krm` のバイト列 → `openContainer`）で渡す。
//
// 宣言の組み立ての小道具（`f32Bytes` / `fill` / `singleOpDeclaration` / `openGraphModel`）は
// `helpers/model-fixture.ts` に 1 本きり — ここが持つのは「鎖グラフ」という形そのものだけ。

import type { TensorInput } from "./container-write.ts";
import type { OpenedContainer } from "../../src/format/container/open.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, openGraphModel } from "./model-fixture.ts";

/** y = relu(x·w + b)（x: [T,4] → y: [T,3]）— 中間値 2 本を持つ最小の鎖。 */
export const chainGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add", "matmul", "relu"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["y"], attrs: {} },
  ],
});

export const W = Float32Array.from([0.5, -1, 0.25, 2, 0.125, -0.5, -3, 1.5, 0.75, 1, -0.25, 0.5]);
export const B = Float32Array.from([1, -2, 0.5]);

/** {@link chainGraph} の initializer 2 本（丸ごと f32）。 */
export const chainTensors = (): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: f32Bytes(W),
    encoding: { codec: "f32" },
  },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes(B),
    encoding: { codec: "f32" },
  },
];

/** 鎖グラフ（差し替え可）を容器に書いて開く。 */
export const openChainModel = (
  graph: DeclarationJson = chainGraph(),
  tensors: readonly TensorInput[] = chainTensors(),
): Promise<OpenedContainer> => openGraphModel(graph, tensors);
