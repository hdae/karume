/**
 * The video / image VAE's channel RMS norm as the exporter lowers it — `mul(x, x) → sum{dim} → sqrt
 * → clamp_min{min} → reshape → div(x, ·) → mul(·, scale) → mul(·, gamma)`, optionally followed by
 * the SiLU `sigmoid → mul(m2, ·)` — as a small synthetic IR graph: the window of the candidate rule
 * `candidate/vae_rms_silu_g8` / `_g4` (packages/candidate-rules/src/vae-rms-silu.ts). Shared by the
 * rule's planning test (vae_rms_silu_rule_plan_test.ts) and its GPU check against the unfused plan
 * (e2e_vae_rms_silu_rule_test.ts). `scale` and `gamma` are f32 initializers, as in the exported
 * graphs. Pure; no side effects.
 */

import type { DeclarationJson } from "../../../runtime/tests/helpers/model-fixture.ts";

/** One node: its op, inputs, output, the output's shape and the node's attrs. */
export type NormNode = {
  readonly op: string;
  readonly ins: readonly string[];
  readonly out: string;
  readonly shape: readonly number[];
  readonly attrs?: Readonly<Record<string, unknown>>;
};

export type NormGraph = {
  /** The f32 graph inputs: name → shape. */
  readonly inputs: Readonly<Record<string, readonly number[]>>;
  /** The f32 initializers: name → shape. */
  readonly weights: Readonly<Record<string, readonly number[]>>;
  readonly nodes: readonly NormNode[];
  /** The graph outputs (default: the last node's output). */
  readonly outputs?: readonly string[];
};

/** Where the channels are: `x` is `[1…, C, P…]` and the sum reduces `axis` (the channel axis). */
export type NormLayout = {
  readonly x: readonly number[];
  readonly axis: number;
  /** gamma's shape: C elements on the channel axis (right-aligned against `x`). */
  readonly gamma: readonly number[];
};

/** `"silu"`: the chain ends with `sigmoid(m2) → mul(m2, ·)`; `"none"`: it ends at `m2`. */
export type NormTail = "silu" | "none";

/** The clamp's floor in the exported graphs (`F.normalize`'s eps). */
export const NORM_FLOOR = 1e-12;

/**
 * The chain over `layout`. The values are named `sq, s, r, n, k, d, m1, m2` and, with the SiLU,
 * `g, y`; the last one is the graph's output.
 */
export const normChain = (layout: NormLayout, tail: NormTail): NormGraph => {
  const shape = layout.x;
  const reduced = shape.filter((_, at) => at !== layout.axis);
  const kept = shape.map((dim, at) => (at === layout.axis ? 1 : dim));
  const norm: NormNode[] = [
    { op: "mul", ins: ["x", "x"], out: "sq", shape },
    { op: "sum", ins: ["sq"], out: "s", shape: reduced, attrs: { dim: layout.axis } },
    { op: "sqrt", ins: ["s"], out: "r", shape: reduced },
    { op: "clamp_min", ins: ["r"], out: "n", shape: reduced, attrs: { min: NORM_FLOOR } },
    { op: "reshape", ins: ["n"], out: "k", shape: kept },
    { op: "div", ins: ["x", "k"], out: "d", shape },
    { op: "mul", ins: ["d", "scale"], out: "m1", shape },
    { op: "mul", ins: ["m1", "gamma"], out: "m2", shape },
  ];
  const silu: NormNode[] = [
    { op: "sigmoid", ins: ["m2"], out: "g", shape },
    { op: "mul", ins: ["m2", "g"], out: "y", shape },
  ];
  return {
    inputs: { x: shape },
    weights: { scale: [1], gamma: layout.gamma },
    nodes: tail === "silu" ? [...norm, ...silu] : norm,
  };
};

/** The IR v2 declaration of `graph` (every value f32). */
export const normDeclaration = (graph: NormGraph): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...new Set(graph.nodes.map(({ op }) => op))].sort() },
  symbols: [],
  inputs: Object.entries(graph.inputs).map(([name, shape]) => ({
    name,
    dtype: "f32",
    shape: [...shape],
  })),
  outputs: [...(graph.outputs ?? [graph.nodes[graph.nodes.length - 1].out])],
  initializers: Object.fromEntries(Object.keys(graph.weights).map((name) => [name, {}])),
  values: Object.fromEntries([
    ...Object.entries(graph.weights).map(([name, shape]) => [
      name,
      { dtype: "f32", shape: [...shape] },
    ]),
    ...graph.nodes.map(({ out, shape }) => [out, { dtype: "f32", shape: [...shape] }]),
  ]),
  nodes: graph.nodes.map(({ op, ins, out, attrs }) => ({
    op,
    ins: [...ins],
    outs: [out],
    attrs: { ...attrs },
  })),
});
