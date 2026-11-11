/**
 * Small graphs for the memory estimate's accounting of node temporaries (estimate_fused_steps_test.ts
 * pins each one's `workspaceBytes` to a literal derived by hand; gpu_estimate_test.ts compares it
 * with the slot table of a real Session): the i8a8 linear with and without a folded producer.
 *
 * The expected numbers stay in the tests (moving hand-derived constants here would make the checks
 * tautological — the rule of estimate-graphs.ts).
 */

import type { SessionOptions } from "../../src/runtime/executor.ts";
import type { TensorInput } from "./container-write.ts";
import { weight } from "./estimate-graphs.ts";
import { type DeclarationJson, f32Bytes, fill, type FilledTensor } from "./model-fixture.ts";
import { quantizeI8 } from "./i8.ts";

/** One case: the graph, its supplies, the inputs of a run and the options of both sides. */
export type FusedWindowCase = {
  readonly name: string;
  /** The counter the window moves in `lastRunFusions` / the plan's counts. */
  readonly counter: string;
  /** How many windows the counter counts. */
  readonly windows: number;
  readonly graph: DeclarationJson;
  readonly tensors: readonly TensorInput[];
  readonly inputs: Readonly<Record<string, FilledTensor>>;
  /** The Session options the window needs, given to the estimate too. */
  readonly options: SessionOptions;
  /** Symbol bindings of the estimate (the run binds them from the inputs). */
  readonly bindings?: Readonly<Record<string, number>>;
};

const f32 = (shape: readonly number[]) => ({ dtype: "f32", shape: [...shape] });

const declaration = (parts: {
  readonly inputs: DeclarationJson["inputs"];
  readonly outputs: readonly string[];
  readonly values: DeclarationJson["values"];
  readonly nodes: DeclarationJson["nodes"];
  readonly initializers?: DeclarationJson["initializers"];
}): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...new Set(parts.nodes.map((node) => node.op))] },
  symbols: [],
  inputs: parts.inputs,
  outputs: [...parts.outputs],
  initializers: parts.initializers ?? {},
  values: parts.values,
  nodes: parts.nodes,
});

/** Deterministic inputs for every graph input (bool inputs alternate 0 / 1 by row). */
const inputsOf = (
  graph: DeclarationJson,
  bindings: Readonly<Record<string, number>> = {},
): Record<string, FilledTensor> =>
  Object.fromEntries(
    graph.inputs.map((spec, slot) => {
      const shape = spec.shape.map((dim) => (typeof dim === "number" ? dim : bindings[dim]));
      return [
        spec.name,
        spec.dtype === "bool"
          ? fill(shape, (i) => Math.floor(i / shape[shape.length - 1]) % 2, "bool")
          : fill(shape, (i) => Math.sin(i * 0.37 + slot) * 1.5),
      ];
    }),
  );

/**
 * `linear(x, w, b)` (`prologue` false) or `linear(gelu_tanh(x), w, b)` with `w [16, 64]` resident as
 * i8 and `x [1, 4, 64]` — the a8 linear's quantize → GEMM pair with its two node temporaries, and the
 * same with the producer folded into its step ① (perf-ledger F-4).
 */
export const a8LinearCase = (prologue: boolean): FusedWindowCase => {
  const [rows, k, n] = [4, 64, 16];
  const row = [1, rows, k];
  const quantized = quantizeI8(fill([n, k], (i) => Math.sin(i * 0.61)).data, [n, k], 0);
  const graph = declaration({
    inputs: [{ name: "x", ...f32(row) }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: f32([n, k]),
      b: f32([n]),
      ...(prologue ? { g: f32(row) } : {}),
      y: f32([1, rows, n]),
    },
    nodes: [
      ...(prologue ? [{ op: "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} }] : []),
      { op: "linear", ins: [prologue ? "g" : "x", "w", "b"], outs: ["y"], attrs: {} },
    ],
  });
  return {
    name: prologue ? "a8 linear with a folded gelu_tanh" : "a8 linear",
    counter: "linearPrologue",
    windows: prologue ? 1 : 0,
    graph,
    tensors: [
      weight("b", f32Bytes(new Array(n).fill(0.5))),
      weight("w", quantized.bytes, {
        codec: "int8-sym",
        groupSize: k,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      }),
    ],
    inputs: inputsOf(graph),
    options: { linearCompute: "a8" },
  };
};
