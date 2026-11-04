// Shared graphs and small utilities of the estimator checks (so estimate_test.ts and
// gpu_estimate_test.ts both take the same shapes. The expected values themselves stay in each test
// — moving hand-computed constants into a helper makes the tests tautological).

import { assertEquals } from "@std/assert";
import type { AdmissionReport, AdmissionScenario } from "../../src/runtime/estimate.ts";
import { prepareContainer, type PreparedModel } from "../../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./container-write.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./f16.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, memoryModel } from "./model-fixture.ts";

/**
 * Declaration + supplies → `PreparedModel` (the entry to admission).
 *
 * The source is an **in-memory container**: the estimate depends only on the merged `IrGraph` and
 * not on a single byte of the `krm` index, so there is no reason to go through the path that
 * writes a container (= the asynchronous sha256 pass).
 */
export const openGraph = (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): PreparedModel => prepareContainer(memoryModel(graph, tensors), GRAPH_NAME);

/**
 * One supply (in IR v2 **the initializer name is the data's key as is**, so the declared name is
 * what is passed). The default storage is `f32` (one whole tensor, no companion scale).
 */
export const weight = (
  initializer: string,
  bytes: Uint8Array<ArrayBuffer>,
  encoding: EncodingInput = { codec: "f32" },
): TensorInput => ({ graph: GRAPH_NAME, initializer, bytes, encoding });

/** f16 bytes (filled with 0 because the values are never looked at — only the byte count is). */
export const f16Zeros = (count: number): Uint8Array<ArrayBuffer> =>
  f16BytesFromBits(new Array(count).fill(f32ToF16Bits(0)));

/**
 * A graph with k of symbolic capacity `C` and v of numeric capacity (one append per slot is a
 * MUST).
 */
export const stateGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "state_append"] },
  symbols: ["T", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { w: {}, chunk: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    chunk: { dtype: "f32", shape: [1, 2, 4, 4] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  states: {
    k: { dtype: "f32", shape: [1, 2, "C", 4] },
    v: { dtype: "f32", shape: [1, 2, 6, 4] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "v" } },
  ],
});

/** The supplies of {@link stateGraph} (filled with 0 because the values are never looked at). */
export const stateTensors = (): readonly TensorInput[] => [
  weight("w", f32Bytes(new Array(12).fill(0))),
  weight("chunk", f32Bytes(new Array(32).fill(0))),
];

export const stateModel = (): PreparedModel => openGraph(stateGraph(), stateTensors());

/**
 * One states-form attention + two `state_append`s (the minimal version of the executed shape of
 * gpu_state_execution_test). `B=1` / `Hkv=2` (GQA) / `D=8`; `M` is the physical chunk rows and `C`
 * the slot capacity.
 *
 * Passing `window` gives the sliding variant (reads and writes MUST use the same formula — it goes
 * on both the attention and the appends). `heads` (default 4) is the axis that **moves only the
 * byte size of one row of S, `H·colCap·4`** — a state slot is `Hkv·C·D·4` and does not depend on
 * H, so raising H builds, on a device with tightened limits, the shape where "the slot fits the
 * binding limit but one row of S does not" (= the row blocks split into several).
 */
export const stateAttentionGraph = (window?: number, heads = 4): DeclarationJson => {
  const windowAttrs: Record<string, number> = window === undefined ? {} : { window };
  const append = (name: string, slot: string) => ({
    op: "state_append",
    ins: [name],
    outs: [] as string[],
    attrs: { ...windowAttrs },
    states: { slot },
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "state_append"] },
    symbols: ["M", "C"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", 8] },
      { name: "k", dtype: "f32", shape: [1, 2, "M", 8] },
      { name: "v", dtype: "f32", shape: [1, 2, "M", 8] },
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, heads, "M", 8] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 2, "C", 8] },
      vslot: { dtype: "f32", shape: [1, 2, "C", 8] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: 0.5, ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      },
      append("k", "kslot"),
      append("v", "vslot"),
    ],
  };
};

/**
 * The 9-node window of a decomposed attention as the exporter emits it (the irodori DiT spelling —
 * the shape that fuses into the row-block attention, ADR 0060). Every value is f32 and only `y` is
 * a graph output. The scores are `heads·queries·keys·4` bytes.
 *
 * `interpose` only puts one zero-byte alias between the bmm and the reshape, which breaks just the
 * window's adjacency condition (the same computation, with the fusion not matching).
 */
export const decomposedAttentionGraph = (
  dims: {
    readonly heads: number;
    readonly queries: number;
    readonly keys: number;
    readonly headDim: number;
  },
  interpose = false,
): DeclarationJson => {
  const { heads, queries, keys, headDim } = dims;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = interpose ? "s3_alias" : "s3";
  const nodes: DeclarationJson["nodes"] = [
    { op: "bmm", ins: ["q", "kt"], outs: [qkOut], attrs: {} },
    ...(interpose ? [{ op: "reshape", ins: [qkOut], outs: ["s3"], attrs: {} }] : []),
    { op: "reshape", ins: ["s3"], outs: ["s4"], attrs: {} },
    { op: "add", ins: ["s4", "mask"], outs: ["masked"], attrs: {} },
    { op: "safe_softmax", ins: ["masked"], outs: ["p4"], attrs: { dim: 3 } },
    { op: "expand", ins: ["p4"], outs: ["pe"], attrs: {} },
    { op: "reshape", ins: ["pe"], outs: ["p3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["ve"], attrs: {} },
    { op: "reshape", ins: ["ve"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["p3", "v3"], outs: ["y"], attrs: {} },
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      { name: "kt", dtype: "f32", shape: [heads, headDim, keys] },
      { name: "mask", dtype: "f32", shape: [1, 1, 1, keys] },
      { name: "v", dtype: "f32", shape: [1, heads, keys, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      ...(interpose ? { s3_alias: { dtype: "f32", shape: scores3 } } : {}),
      s3: { dtype: "f32", shape: scores3 },
      s4: { dtype: "f32", shape: scores4 },
      masked: { dtype: "f32", shape: scores4 },
      p4: { dtype: "f32", shape: scores4 },
      pe: { dtype: "f32", shape: scores4 },
      p3: { dtype: "f32", shape: scores3 },
      ve: { dtype: "f32", shape: [1, heads, keys, headDim] },
      v3: { dtype: "f32", shape: [heads, keys, headDim] },
      y: { dtype: "f32", shape: [heads, queries, headDim] },
    },
    nodes,
  };
};

/** Takes the prefill / decode pair by name (pinning the assumed order here as well). */
export const bothScenarios = (
  report: AdmissionReport,
): { readonly prefill: AdmissionScenario; readonly decode: AdmissionScenario } => {
  assertEquals(
    report.scenarios.map((scenario) => scenario.name),
    ["prefill", "decode"],
  );
  return { prefill: report.scenarios[0], decode: report.scenarios[1] };
};
