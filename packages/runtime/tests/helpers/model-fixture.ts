// Builders for container-surface tests (IR v2 declaration JSON → `krm` bytes / an opened
// container).
//
// The "correct shape" the tests expect lives in one place, and each failure-case test breaks
// exactly one point of it. The same input builds both the **krm path** (`writeModelContainer` →
// `openContainer`) and the **in-memory container** (`openMemoryContainer`), so an A/B of the two
// sources is built from this one file too.
//
// The small tools that build the declarations themselves (`fill` / `f32Bytes` /
// `singleOpDeclaration` / `withStateReaders`) live only here as well. Tests that need the joined
// `IrGraph` use `helpers/merged-graph.ts`.

import type { BoundContainer } from "../../src/format/container/bind.ts";
import { openContainer, type OpenedContainer } from "../../src/format/container/open.ts";
import { type MemoryTensor, openMemoryContainer } from "../../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../../src/format/ir.ts";
import { type TensorInput, writeModelContainer, type WriteOptions } from "./container-write.ts";

/**
 * Name of the graph this helper writes (a container can hold several graphs; the fixture has one).
 */
export const GRAPH_NAME = "model";

/** f32 values as raw bytes (the smallest tool for building weight / scale data). */
export const f32Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/** Mutable copy of an IR v2 declaration JSON (loosely typed so a test can break one point). */
export type DeclarationJson = {
  format: string;
  version: number;
  requires: { ops: string[] };
  symbols: string[];
  inputs: { name: string; dtype: string; shape: (number | string)[] }[];
  outputs: string[];
  /**
   * An IR v2 initializer declaration has only two forms: `{}` (owns data) or `{ shared: true }`
   * (borrowed).
   */
  initializers: Record<string, { shared?: true }>;
  values: Record<string, { dtype: string; shape: (number | string)[] }>;
  /** Optional state slot section (ADR 0066 decision 2) — the default declaration has none. */
  states?: Record<string, { dtype: string; shape: (number | string)[]; external?: boolean }>;
  nodes: {
    op: string;
    ins: string[];
    outs: string[];
    attrs: unknown;
    /** References to state slots by name (ADR 0067 decision 4) — the default nodes have none. */
    states?: Record<string, string>;
  }[];
};

/** Smallest valid graph: y = x·w + b (x: T×4 → y: T×3). */
export const baseDeclaration = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  // MUST: fields treated as sets (`requires.ops` / `symbols`) are written in code point order.
  // `krm` goes through canonical serialization, so the order matches and the graphs of the krm
  // path and the memory path can be compared whole.
  requires: { ops: ["add", "matmul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

/** Weights for {@link baseDeclaration} (one whole f32 tensor each). */
export const baseTensors = (): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: f32Bytes(new Array(12).fill(0.5)),
    encoding: { codec: "f32" },
  },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes([1, 2, 3]),
    encoding: { codec: "f32" },
  },
];

const declarationOf = (graph: DeclarationJson): IrDeclaration => parseIrDeclarationValue(graph);

/** Declaration + weights as `krm` bytes (single-file form). */
export const modelBytes = async (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
  options?: WriteOptions,
): Promise<Uint8Array<ArrayBuffer>> => {
  const written = await writeModelContainer(
    {
      graphs: { [GRAPH_NAME]: declarationOf(graph) },
      consts: [],
      weights: tensors,
      assets: [],
      provenance: { license: "test" },
    },
    options ?? {},
  );
  return written.single;
};

/**
 * {@link modelBytes}, opened (no expected values for the two documents — the test / local-load
 * surface).
 */
export const openModelBytes = async (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
  options?: WriteOptions,
): Promise<OpenedContainer> =>
  await openContainer({ kind: "bytes", bytes: await modelBytes(graph, tensors, options) });

/** Any graph (zero weights is fine) as `krm` bytes. */
export const graphModelBytes = (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): Promise<Uint8Array<ArrayBuffer>> => modelBytes(graph, tensors);

/**
 * Writes any graph to `krm` and opens it (a graph without weights omits `tensors`).
 *
 * The only difference from {@link openModelBytes} is the default supply — {@link baseTensors}
 * there, none here.
 */
export const openGraphModel = async (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): Promise<OpenedContainer> =>
  await openContainer({ kind: "bytes", bytes: await graphModelBytes(graph, tensors) });

/** Maps the container writer's input to an in-memory container supply (one whole tensor each). */
export const memoryTensorOf = (tensor: TensorInput): MemoryTensor => ({
  bytes: tensor.bytes,
  encoding: {
    codec: tensor.encoding.codec,
    ...(tensor.encoding.rowAxis === undefined ? {} : { rowAxis: tensor.encoding.rowAxis }),
    ...(tensor.encoding.groupSize === undefined ? {} : { groupSize: tensor.encoding.groupSize }),
    ...(tensor.encoding.scale === undefined ? {} : { scale: tensor.encoding.scale.bytes }),
  },
});

/**
 * Builds an in-memory container from **the same input** as {@link modelBytes} (one side of the A/B
 * with the krm path).
 */
export const memoryModel = (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
): BoundContainer =>
  openMemoryContainer({
    graphs: { [GRAPH_NAME]: declarationOf(graph) },
    tensors: {
      [GRAPH_NAME]: Object.fromEntries(
        tensors.map((tensor) => [tensor.initializer, memoryTensorOf(tensor)]),
      ),
    },
  });

/** Host-side TypedArray per semantic dtype (bool is u32 0/1 — ADR 0009). */
export type FilledTensor =
  | {
      readonly dtype: "f32";
      readonly shape: readonly number[];
      readonly data: Float32Array<ArrayBuffer>;
    }
  | {
      readonly dtype: "i32";
      readonly shape: readonly number[];
      readonly data: Int32Array<ArrayBuffer>;
    }
  | {
      readonly dtype: "bool";
      readonly shape: readonly number[];
      readonly data: Uint32Array<ArrayBuffer>;
    };

/** Deterministic data generation (no random numbers — a failure would not reproduce). */
export const fill = (
  shape: readonly number[],
  generator: (index: number) => number,
  dtype: "f32" | "i32" | "bool" = "f32",
): FilledTensor => {
  const count = shape.reduce((total, dim) => total * dim, 1);
  const values = Array.from({ length: count }, (_, i) => generator(i));
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: Float32Array.from(values) };
    case "i32":
      return { dtype, shape, data: Int32Array.from(values) };
    case "bool":
      return { dtype, shape, data: Uint32Array.from(values) };
  }
};

/**
 * Value name of an output slot. Slot 0 is `y` (existing tests read `outputs["y"]` as is), and
 * slot 1 onward are `y1`, `y2`, … (multi-output ops — ADR 0068 decision 1).
 */
export const outputName = (slot: number): string => (slot === 0 ? "y" : `y${slot}`);

/** dtype / attrs options of a single-node declaration (default: all f32, empty attrs). */
export type SingleOpOptions = {
  readonly inDtypes?: readonly string[];
  /**
   * dtypes in ascending output-slot order (default: f32 for every slot — {@link
   * singleOpDeclaration}).
   */
  readonly outDtypes?: readonly string[];
  readonly attrs?: Record<string, unknown>;
  readonly symbols?: readonly string[];
};

/**
 * Declaration of a single node that takes only inputs (for per-op numeric checks).
 *
 * `outShapes` is **a list in ascending output-slot order** (ADR 0068 decision 1 — length 1 for a
 * single-output op).
 */
export const singleOpDeclaration = (
  op: string,
  inputShapes: readonly (readonly (number | string)[])[],
  outShapes: readonly (readonly (number | string)[])[],
  options: SingleOpOptions = {},
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [op] },
  symbols: [...(options.symbols ?? [])],
  inputs: inputShapes.map((shape, index) => ({
    name: `x${index}`,
    dtype: options.inDtypes?.[index] ?? "f32",
    shape: [...shape],
  })),
  outputs: outShapes.map((_, slot) => outputName(slot)),
  initializers: {},
  values: Object.fromEntries(
    outShapes.map((shape, slot) => [
      outputName(slot),
      { dtype: options.outDtypes?.[slot] ?? "f32", shape: [...shape] },
    ]),
  ),
  nodes: [
    {
      op,
      ins: inputShapes.map((_, index) => `x${index}`),
      outs: outShapes.map((_, slot) => outputName(slot)),
      attrs: { ...options.attrs },
    },
  ],
});

/**
 * Adds to each declared state slot the smallest `state_append` node that references it
 * (referential completeness — ADR 0067 decisions 4 / 5).
 *
 * A graph with declarations only fails with "a declaration no node references", so tests that want
 * to look at **the slot declaration itself** use this to provide the referencing side. The parser
 * is a layer that checks neither shapes nor contracts, so the only thing that matters about `ins`
 * is "it is a defined value name" (shape checks belong to shape inference / the contract layer).
 *
 * A graph without any slot gets nothing added (if only `requires.ops` gained `state_append`, the
 * declaration would fail as "unused", since no node references it).
 */
export const withStateReaders = (graph: DeclarationJson): DeclarationJson => {
  const slots = Object.keys(graph.states ?? {});
  if (slots.length === 0) return graph;
  for (const slot of slots) {
    graph.nodes.push({ op: "state_append", ins: ["x"], outs: [], attrs: {}, states: { slot } });
  }
  if (!graph.requires.ops.includes("state_append")) graph.requires.ops.push("state_append");
  return graph;
};
