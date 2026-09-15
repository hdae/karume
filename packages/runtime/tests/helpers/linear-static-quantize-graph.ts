import { buildSafetensors, f32Bytes, type GraphJson } from "./format.ts";

export type LinearStaticQuantizeOptions = {
  m?: number;
  n?: number;
  k?: number;
  scale?: number;
  storage?: "i2" | "i4" | "i8";
  group?: number;
  publicLinear?: boolean;
  sharedLinear?: boolean;
  interpose?: boolean;
  sharedWeight?: boolean;
};

/** privacyと隣接条件を変えても非融合の実行が成立する検査用グラフ。 */
export const linearStaticQuantizeGraph = (o: LinearStaticQuantizeOptions = {}): GraphJson => {
  const m = o.m ?? 4, n = o.n ?? 256, k = o.k ?? 1536, storage = o.storage ?? "i8";
  const weight = {
    dtype: storage,
    scale: "s",
    ...(storage === "i4" ? { group_size: o.group ?? 512 } : {}),
  };
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear", "static_quantize", "neg"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y", "copy", ...(o.publicLinear ? ["linear"] : [])],
    initializers: {
      w: o.sharedWeight
        ? {
          shared: { tensor: "w" },
          storage: { dtype: storage },
        }
        : { tensor: "w", storage: weight },
      b: { tensor: "b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      linear: { dtype: "f32", shape: [m, n] },
      y: { dtype: "f32", shape: [m, n] },
      copy: { dtype: "f32", shape: [m, k] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["linear"], attrs: {} }],
  };
  const copy = { op: "neg", ins: ["x"], outs: ["copy"], attrs: {} };
  if (o.interpose) graph.nodes.push(copy);
  graph.nodes.push({
    op: "static_quantize",
    ins: ["linear"],
    outs: ["y"],
    attrs: { scale: o.scale ?? Math.fround(.00071) },
  });
  if (!o.interpose) graph.nodes.push(copy);
  if (o.sharedLinear) {
    graph.values.extra = { dtype: "f32", shape: [m, n] };
    graph.nodes.push({ op: "neg", ins: ["linear"], outs: ["extra"], attrs: {} });
    graph.outputs.push("extra");
  }
  return graph;
};

export const linearStaticQuantizeModel = (o: LinearStaticQuantizeOptions = {}): ArrayBuffer => {
  const graph = linearStaticQuantizeGraph(o),
    n = o.n ?? 256,
    k = o.k ?? 1536,
    storage = o.storage ?? "i8";
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8,
    group = storage === "i4" ? o.group ?? 512 : undefined;
  const scaleShape = group === undefined ? [n, 1] : [n, k / group];
  return buildSafetensors([
    ...o.sharedWeight ? [] : [{
      name: "w",
      dtype: storage === "i2" ? "I2" : storage === "i4" ? "I4" : "I8",
      shape: [n, k],
      data: Uint8Array.from(
        { length: n * k * bits / 8 },
        (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
      ),
    }, {
      name: "s",
      dtype: "F32",
      shape: scaleShape,
      data: f32Bytes(
        Array.from({ length: scaleShape[0] * scaleShape[1] }, (_, i) => (i % 17 + 1) * .00017),
      ),
    }],
    {
      name: "b",
      dtype: "F32",
      shape: [n],
      data: f32Bytes(Array.from({ length: n }, (_, i) => (i % 7 - 3) * .11)),
    },
  ], { karume_ir: JSON.stringify(graph) });
};
