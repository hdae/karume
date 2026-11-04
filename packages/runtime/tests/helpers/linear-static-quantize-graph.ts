import type { BoundContainer } from "../../src/format/container/bind.ts";
import type { EncodingInput, TensorInput } from "./container-write.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, memoryModel } from "./model-fixture.ts";

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

/**
 * A test graph whose unfused execution holds whatever the adjacency (IR v2 declaration — no
 * storage).
 */
export const linearStaticQuantizeGraph = (o: LinearStaticQuantizeOptions = {}): DeclarationJson => {
  const m = o.m ?? 4,
    n = o.n ?? 256,
    k = o.k ?? 1536;
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "static_quantize", "neg"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y", "copy", ...(o.publicLinear ? ["linear"] : [])],
    initializers: {
      // A borrowed declaration is a name only (MUST equal the lender's initializer name — no
      // storage, no data).
      w: o.sharedWeight ? { shared: true } : {},
      b: {},
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
    attrs: { scale: o.scale ?? Math.fround(0.00071) },
  });
  if (!o.interpose) graph.nodes.push(copy);
  if (o.sharedLinear) {
    graph.values.extra = { dtype: "f32", shape: [m, n] };
    graph.nodes.push({ op: "neg", ins: ["linear"], outs: ["extra"], attrs: {} });
    graph.outputs.push("extra");
  }
  return graph;
};

/**
 * The supply side decides storage (codec / group / companion scale) — where v1's `storage`
 * declaration went.
 */
const weightEncoding = (o: LinearStaticQuantizeOptions): EncodingInput => {
  const n = o.n ?? 256,
    k = o.k ?? 1536,
    storage = o.storage ?? "i8";
  const groupSize = storage === "i4" ? (o.group ?? 512) : k;
  const groups = k / groupSize;
  return {
    codec: storage === "i2" ? "int2-off" : storage === "i4" ? "int4-sym-g" : "int8-sym",
    groupSize,
    scale: {
      bytes: f32Bytes(Array.from({ length: n * groups }, (_, i) => ((i % 17) + 1) * 0.00017)),
      dtype: "f32",
    },
  };
};

/** The supply for {@link linearStaticQuantizeGraph} (the borrowed `w` is not supplied). */
export const linearStaticQuantizeTensors = (
  o: LinearStaticQuantizeOptions = {},
): readonly TensorInput[] => {
  const n = o.n ?? 256,
    k = o.k ?? 1536,
    storage = o.storage ?? "i8";
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  return [
    ...(o.sharedWeight
      ? []
      : [
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: Uint8Array.from(
              { length: (n * k * bits) / 8 },
              (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
            ),
            encoding: weightEncoding(o),
          },
        ]),
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes(Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11)),
      encoding: { codec: "f32" } satisfies EncodingInput,
    },
  ];
};

/**
 * In-memory container joining declaration + supply (input to `createSessionFromContainer` /
 * `mergedGraph`).
 */
export const linearStaticQuantizeModel = (o: LinearStaticQuantizeOptions = {}): BoundContainer =>
  memoryModel(linearStaticQuantizeGraph(o), linearStaticQuantizeTensors(o));
