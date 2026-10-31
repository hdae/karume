/**
 * The `gelu_tanh → mul → static_quantize` chain of Gemma 4 QAT's per-layer-input gate as a
 * weightless graph (perf-ledger K-48). Each option changes one eligibility condition of the
 * `geluMulStaticQuantize` fusion; the unfused graph always runs.
 */

import type { GeluMulOrder } from "../../src/kernels/gelu-mul-static-quantize.ts";
import type { DeclarationJson } from "./model-fixture.ts";

export type GeluMulStaticQuantizeGraphOptions = {
  /** M (the product of the leading dims). Default 4. */
  readonly rows?: number;
  /** The last dim. Default 256. */
  readonly dim?: number;
  readonly order?: GeluMulOrder;
  readonly scale?: number;
  /**
   * The mul's other operand: `"slice"` = the QAT form (a materialised slice of a `[1, M, L, dim]`
   * input, then a reshape alias — `slice_tensor` → `squeeze_dims`); `"input"` = a graph input of
   * the chain's shape. Default `"slice"`.
   */
  readonly other?: "slice" | "input";
  /** The other operand is `[1, 1, dim]` (a broadcast mul). */
  readonly broadcast?: boolean;
  /** The other operand is the gate itself (`mul(gelu(gate), gate)`). */
  readonly otherIsGate?: boolean;
  readonly geluOutput?: boolean;
  readonly mulOutput?: boolean;
  readonly extraGeluConsumer?: boolean;
  readonly extraMulConsumer?: boolean;
  /** A `neg` of the gate between gelu_tanh and mul (the chain is not adjacent). */
  readonly interpose?: boolean;
};

/** The per-layer input's layer count of the `"slice"` form (the slice takes layer 1). */
export const PLE_LAYERS = 3;

export const geluMulStaticQuantizeGraph = (
  options: GeluMulStaticQuantizeGraphOptions = {},
): DeclarationJson => {
  const rows = options.rows ?? 4,
    dim = options.dim ?? 256;
  const shape = [1, rows, dim];
  const otherShape = options.broadcast ? [1, 1, dim] : shape;
  const inputs: DeclarationJson["inputs"] = [{ name: "gate", dtype: "f32", shape }];
  const values: DeclarationJson["values"] = {
    g: { dtype: "f32", shape },
    m: { dtype: "f32", shape },
    y: { dtype: "f32", shape },
    copy: { dtype: "f32", shape },
  };
  const nodes: DeclarationJson["nodes"] = [];
  const ops = new Set(["gelu_tanh", "mul", "static_quantize", "neg"]);
  let other = "gate";
  if (!options.otherIsGate) {
    if ((options.other ?? "slice") === "slice") {
      const otherRows = otherShape[1];
      inputs.push({ name: "pli", dtype: "f32", shape: [1, otherRows, PLE_LAYERS, dim] });
      values.slice = { dtype: "f32", shape: [1, otherRows, 1, dim] };
      values.ple = { dtype: "f32", shape: otherShape };
      nodes.push(
        { op: "slice", ins: ["pli"], outs: ["slice"], attrs: { dim: 2, start: 1, end: 2 } },
        { op: "reshape", ins: ["slice"], outs: ["ple"], attrs: {} },
      );
      ops.add("slice").add("reshape");
    } else {
      inputs.push({ name: "ple", dtype: "f32", shape: otherShape });
    }
    other = "ple";
  }
  nodes.push({ op: "gelu_tanh", ins: ["gate"], outs: ["g"], attrs: {} });
  if (options.interpose) {
    values.between = { dtype: "f32", shape };
    nodes.push({ op: "neg", ins: ["gate"], outs: ["between"], attrs: {} });
  }
  nodes.push(
    {
      op: "mul",
      ins: (options.order ?? "gelu-other") === "gelu-other" ? ["g", other] : [other, "g"],
      outs: ["m"],
      attrs: {},
    },
    {
      op: "static_quantize",
      ins: ["m"],
      outs: ["y"],
      attrs: { scale: options.scale ?? Math.fround(0.0371) },
    },
    // The gate must stay alive after the fold: a release too early shows in `copy`.
    { op: "neg", ins: ["gate"], outs: ["copy"], attrs: {} },
  );
  const outputs = [
    "y",
    "copy",
    ...(options.interpose ? ["between"] : []),
    ...(options.geluOutput ? ["g"] : []),
    ...(options.mulOutput ? ["m"] : []),
  ];
  if (options.extraGeluConsumer) {
    values.g_copy = { dtype: "f32", shape };
    nodes.push({ op: "neg", ins: ["g"], outs: ["g_copy"], attrs: {} });
    outputs.push("g_copy");
  }
  if (options.extraMulConsumer) {
    values.m_copy = { dtype: "f32", shape };
    nodes.push({ op: "neg", ins: ["m"], outs: ["m_copy"], attrs: {} });
    outputs.push("m_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...ops] },
    symbols: [],
    inputs,
    outputs,
    initializers: {},
    values,
    nodes,
  };
};
