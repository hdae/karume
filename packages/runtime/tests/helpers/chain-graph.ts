// Session ライフサイクル系テストの共有グラフ（runtime_executor_test.ts と
// gpu_runtime_executor_test.ts が同じ鎖グラフを使う）。
import { f32Bytes, type GraphJson } from "./format.ts";
import { graphModelBuffer } from "./graph.ts";

/** y = relu(x·w + b)（x: [T,4] → y: [T,3]）— 中間値 2 本を持つ最小の鎖。 */
export const chainGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add", "relu"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "enc.w", storage: { dtype: "f32" } },
    b: { tensor: "enc.b", storage: { dtype: "f32" } },
  },
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

export const chainModelBuffer = (graph: GraphJson = chainGraph()): ArrayBuffer =>
  graphModelBuffer(graph, [
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes([...W]) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([...B]) },
  ]);
