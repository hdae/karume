// Shared fixture of the Session lifecycle tests (runtime_executor_test.ts and
// gpu_runtime_executor_test.ts use the same chain graph). The declaration is IR v2; the data is
// passed as a container (`krm` bytes → `openContainer`).
//
// The declaration-building utilities (`f32Bytes` / `fill` / `singleOpDeclaration` /
// `openGraphModel`) live only in `helpers/model-fixture.ts` — this file holds just the shape of the
// "chain graph" itself.

import type { TensorInput } from "./container-write.ts";
import type { OpenedContainer } from "../../src/format/container/open.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, openGraphModel } from "./model-fixture.ts";

/** y = relu(x·w + b) (x: [T,4] → y: [T,3]) — the smallest chain with two intermediate values. */
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

/** The two initializers of {@link chainGraph} (plain f32). */
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

/** Writes the chain graph (replaceable) into a container and opens it. */
export const openChainModel = (
  graph: DeclarationJson = chainGraph(),
  tensors: readonly TensorInput[] = chainTensors(),
): Promise<OpenedContainer> => openGraphModel(graph, tensors);
