/**
 * Unit tests of fusion candidate enumeration (`enumerateUnfusedWindows`). No GPU; synthetic IR
 * only.
 *
 * What is checked is how windows are formed = "which contiguous windows become candidates and which
 * are dropped", with a control for each eligibility condition (single output / no state access /
 * f32 / the chain's internal values do not escape).
 *
 * The cross-check against real assets (reproducing known hit counts, detecting double counting)
 * and the aggregation live in the tool's tools/fusion-hints/enumerate_test.ts — discovering,
 * binding and aggregating assets are the tool's concern, and runtime tests do not import tools
 * (ADR 0008 addendum 2026-09-03).
 */

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import type { IrGraph } from "../src/format/ir.ts";
import {
  enumerateUnfusedWindows,
  type ExecStep,
  planAliases,
  planFusions,
  type UnfusedWindow,
} from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

/** Device capabilities used for the decision (WebGPU core defaults — 128MiB / 65535). */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

const shaped = (name: string, shape: readonly number[]) => ({
  name,
  dtype: "f32",
  shape: [...shape],
});

/** Graph JSON → planned node sequence + the fusion decision context. */
const plan = (graph: DeclarationJson, inputShapes: Readonly<Record<string, readonly number[]>>) => {
  const ir: IrGraph = mergeGraph(graph);
  const nodes = planGraph(ir, bindSymbols(ir, inputShapes)).nodes;
  return {
    nodes,
    context: {
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: TEST_LIMITS,
    },
  };
};

/** A plan with fusion turned off (every node is a plain step). */
const unfused = (nodes: ReturnType<typeof plan>["nodes"]): readonly ExecStep[] => {
  const aliases = planAliases(nodes);
  return nodes.map((node) => ({ kind: "node", plan: node, aliasesInput: aliases.has(node) }));
};

/** A window summary (op names / window width / first node position / chain output name). */
const outline = (windows: readonly UnfusedWindow[]): readonly string[] =>
  windows.map(
    (window) =>
      `${window.ops.join(",")}@${window.nodeIndex}+${window.windowSize}=${window.outputName}`,
  );

/**
 * A plain 3-node chain (`neg → add → mul`). It matches no rule's spelling, so the plan is always a
 * plain sequence (= a straightforward input for the enumerator).
 */
const chainGraph = (
  options: {
    readonly extraConsumer?: boolean;
    readonly internalOutput?: boolean;
  } = {},
): DeclarationJson => {
  const shape = [4, 8];
  const nodes: DeclarationJson["nodes"] = [
    { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "add", ins: ["a", "w"], outs: ["b"], attrs: {} },
    { op: "mul", ins: ["b", "w"], outs: ["c"], attrs: {} },
  ];
  const values: DeclarationJson["values"] = {
    a: { dtype: "f32", shape: [...shape] },
    b: { dtype: "f32", shape: [...shape] },
    c: { dtype: "f32", shape: [...shape] },
  };
  if (options.extraConsumer) {
    values.d = { dtype: "f32", shape: [...shape] };
    nodes.push({ op: "neg", ins: ["b"], outs: ["d"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [shaped("x", shape), shaped("w", shape)],
    outputs: [
      ...(options.internalOutput ? ["b"] : []),
      "c",
      ...(options.extraConsumer ? ["d"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

const CHAIN_INPUTS = { x: [4, 8], w: [4, 8] } as const;

test("enumerates every window of length 2 to maxWindow from a plain node sequence", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  assertEquals(planFusions(nodes, context).steps.length, 3, "precondition: no rule matches");
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 3)), [
    "neg,add@0+2=b",
    "neg,add,mul@0+3=c",
    "add,mul@1+2=c",
  ]);
  // The window width limit only narrows the accepted set (the length-3 window disappears).
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 2)), [
    "neg,add@0+2=b",
    "add,mul@1+2=c",
  ]);
});

test("a window whose chain's internal value escapes is not a candidate (another consumer / graph output)", () => {
  for (const [label, graph] of [
    ["another consumer", chainGraph({ extraConsumer: true })],
    ["graph output", chainGraph({ internalOutput: true })],
  ] as const) {
    const { nodes, context } = plan(graph, CHAIN_INPUTS);
    // Only the window that makes `b` its **final output** remains (a fused step materializes the
    // final output, so the final node's output escaping is eligible).
    assertEquals(
      outline(enumerateUnfusedWindows(unfused(nodes), context, 4)),
      ["neg,add@0+2=b"],
      label,
    );
  }
});

/**
 * Splitting out an in-window passthrough (an in-window node not in the chain). The same shape as
 * the `sym_prefix_slice` of RoPE's cos / sin tables — it flows into the chain's `add` but is also
 * read elsewhere (2 consumers), so it does not join the chain, and only the window width becomes
 * 1 larger than the chain.
 */
test("an in-window passthrough leaves the chain, and only the window width grows beyond the chain", () => {
  const shape = [4, 8];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["neg", "add", "mul"] },
    symbols: [],
    inputs: [shaped("x", shape), shaped("w", shape)],
    outputs: ["c", "e"],
    initializers: {},
    values: {
      a: { dtype: "f32", shape: [...shape] },
      t: { dtype: "f32", shape: [...shape] },
      b: { dtype: "f32", shape: [...shape] },
      c: { dtype: "f32", shape: [...shape] },
      e: { dtype: "f32", shape: [...shape] },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
      // t is an in-window passthrough (read by the add below and by the mul outside the window).
      { op: "neg", ins: ["w"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["a", "t"], outs: ["b"], attrs: {} },
      { op: "mul", ins: ["b", "w"], outs: ["c"], attrs: {} },
      { op: "mul", ins: ["t", "w"], outs: ["e"], attrs: {} },
    ],
  };
  const { nodes, context } = plan(graph, CHAIN_INPUTS);
  const windows = enumerateUnfusedWindows(unfused(nodes), context, 4);
  // `neg,add` is picked up with window width 3 (with the passthrough neg in between).
  assertEquals(outline(windows).includes("neg,add@0+3=b"), true, outline(windows).join(" / "));
  // A window starting at the passthrough itself (double counting of `neg,add`) does not appear —
  // windows whose first node is not the chain's first node are dropped.
  assertEquals(
    windows.filter((window) => window.ops.join(",") === "neg,add").map((w) => w.nodeIndex),
    [0],
  );
});

test("a window containing a multi-output op is not a candidate", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["neg", "topk"] },
    symbols: [],
    inputs: [shaped("x", [4, 8])],
    outputs: ["y", "i"],
    initializers: {},
    values: {
      a: { dtype: "f32", shape: [4, 8] },
      v: { dtype: "f32", shape: [4, 3] },
      i: { dtype: "i32", shape: [4, 3] },
      y: { dtype: "f32", shape: [4, 3] },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
      { op: "topk", ins: ["a"], outs: ["v", "i"], attrs: { k: 3 } },
      { op: "neg", ins: ["v"], outs: ["y"], attrs: {} },
    ],
  };
  const { nodes, context } = plan(graph, { x: [4, 8] });
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 3)), []);
});

test("a window containing a node that touches state is not a candidate", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  // In the current op vocabulary only `attention` / `state_append` can have a `states` field, and a
  // window of this spelling cannot be reached from a real graph. As an invariant of the window
  // mechanism, a synthetic node pins the decision point.
  const tainted = nodes.map((node, index) =>
    index === 1 ? { ...node, node: { ...node.node, states: { k: "kv" } } } : node,
  );
  assertEquals(outline(enumerateUnfusedWindows(unfused(tainted), context, 3)), []);
});

test("a chain that is not f32 is not a candidate", () => {
  const reshapeGraph = (dtype: "f32" | "i32"): DeclarationJson => ({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["reshape"] },
    symbols: [],
    inputs: [{ name: "x", dtype, shape: [4, 8] }],
    outputs: ["c"],
    initializers: {},
    values: {
      a: { dtype, shape: [32] },
      b: { dtype, shape: [8, 4] },
      c: { dtype, shape: [2, 16] },
    },
    nodes: [
      { op: "reshape", ins: ["x"], outs: ["a"], attrs: {} },
      { op: "reshape", ins: ["a"], outs: ["b"], attrs: {} },
      { op: "reshape", ins: ["b"], outs: ["c"], attrs: {} },
    ],
  });
  const f32 = plan(reshapeGraph("f32"), { x: [4, 8] });
  // Control: the same spelling in f32 yields 3 (the flip side showing that i32's 0 is not "there
  // are no windows at all").
  assertEquals(enumerateUnfusedWindows(unfused(f32.nodes), f32.context, 3).length, 3);
  const i32 = plan(reshapeGraph("i32"), { x: [4, 8] });
  assertEquals(enumerateUnfusedWindows(unfused(i32.nodes), i32.context, 3), []);
});

test("no window crosses a fused step (no double counting of chains already matched)", () => {
  const shape = [4, 8];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["sigmoid", "mul", "neg", "add"] },
    symbols: [],
    inputs: [shaped("x", shape)],
    outputs: ["o"],
    initializers: {},
    values: {
      s: { dtype: "f32", shape: [...shape] },
      y: { dtype: "f32", shape: [...shape] },
      z: { dtype: "f32", shape: [...shape] },
      o: { dtype: "f32", shape: [...shape] },
    },
    nodes: [
      { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
      { op: "mul", ins: ["x", "s"], outs: ["y"], attrs: {} },
      { op: "neg", ins: ["y"], outs: ["z"], attrs: {} },
      { op: "add", ins: ["z", "x"], outs: ["o"], attrs: {} },
    ],
  };
  const { nodes, context } = plan(graph, { x: shape });
  const fused = planFusions(nodes, context);
  assertEquals(fused.counts.silu, 1, "precondition: SiLU is matched");
  // With fusion off, sigmoid,mul also shows up as a candidate.
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 2)), [
    "sigmoid,mul@0+2=y",
    "mul,neg@1+2=z",
    "neg,add@2+2=o",
  ]);
  // The current plan does not cross fused steps, so only the runs of remaining plain nodes are
  // candidates. `nodeIndex` advances by the number of folded nodes (fused step = 2 nodes → neg is
  // 2).
  assertEquals(outline(enumerateUnfusedWindows(fused.steps, context, 2)), ["neg,add@2+2=o"]);
});

test("fails loudly if the window width limit is below 2", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  for (const maxWindow of [1, 0, -1, 2.5, Number.NaN]) {
    assertThrows(
      () => enumerateUnfusedWindows(unfused(nodes), context, maxWindow),
      ExecutionError,
      undefined,
      `maxWindow=${maxWindow}`,
    );
  }
});

test("a linear with a folded producer (perf-ledger F-4) is a grabbed pair: no window crosses it, and the node index advances by its two nodes", () => {
  const dtype = "f32";
  const shape = [1, 4, 8];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["gelu_tanh", "linear", "neg"] },
    symbols: [],
    inputs: [shaped("x", shape)],
    outputs: ["d"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype, shape: [8, 8] },
      b: { dtype, shape: [8] },
      a: { dtype, shape },
      n: { dtype, shape },
      g: { dtype, shape },
      y: { dtype, shape },
      c: { dtype, shape },
      d: { dtype, shape },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
      { op: "neg", ins: ["a"], outs: ["n"], attrs: {} },
      { op: "gelu_tanh", ins: ["n"], outs: ["g"], attrs: {} },
      { op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} },
      { op: "neg", ins: ["y"], outs: ["c"], attrs: {} },
      { op: "neg", ins: ["c"], outs: ["d"], attrs: {} },
    ],
  };
  const { nodes, context } = plan(graph, { x: shape });
  const a8: Parameters<typeof planFusions>[1] = {
    ...context,
    linearCompute: "a8",
    weightLayouts: new Map([["w", { storage: "i8" }]]),
  };
  const fused = planFusions(nodes, a8);
  assertEquals(fused.counts.linearPrologue, 1, "precondition: the pair is folded");
  assertEquals(outline(enumerateUnfusedWindows(fused.steps, a8, 3)), [
    "neg,neg@0+2=n",
    "neg,neg@4+2=d",
  ]);
});

/**
 * `neg → neg` into q, the q / k / v permutes, the attention, O permuted back, then `neg → neg`: the
 * Wan DiT form between plain chains, at sequence length `s` (S = N) and head width `d` (H = 2).
 */
const attentionHintsGraph = (s: number, d: number): DeclarationJson => {
  const dtype = "f32";
  const token = [1, s, 2, d];
  const head = [1, 2, s, d];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "neg", "permute"] },
    symbols: [],
    inputs: [shaped("x", token), shaped("kt", token), shaped("vt", token)],
    outputs: ["d"],
    initializers: {},
    values: {
      a: { dtype, shape: token },
      qt: { dtype, shape: token },
      q: { dtype, shape: head },
      k: { dtype, shape: head },
      v: { dtype, shape: head },
      o: { dtype, shape: head },
      ot: { dtype, shape: token },
      c: { dtype, shape: token },
      d: { dtype, shape: token },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
      { op: "neg", ins: ["a"], outs: ["qt"], attrs: {} },
      { op: "permute", ins: ["qt"], outs: ["q"], attrs: { dims: [0, 2, 1, 3] } },
      { op: "permute", ins: ["kt"], outs: ["k"], attrs: { dims: [0, 2, 1, 3] } },
      { op: "permute", ins: ["vt"], outs: ["v"], attrs: { dims: [0, 2, 1, 3] } },
      { op: "attention", ins: ["q", "k", "v"], outs: ["o"], attrs: { scale: 0.125 } },
      { op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } },
      { op: "neg", ins: ["ot"], outs: ["c"], attrs: {} },
      { op: "neg", ins: ["c"], outs: ["d"], attrs: {} },
    ],
  };
};

test("an attention that absorbed its permutes (perf-ledger F-3 / F-8) is a grabbed window: no window crosses it, and the node index advances by its four nodes (the output permute stays a plain node)", () => {
  const token = [1, 5, 2, 8];
  const { nodes, context } = plan(attentionHintsGraph(5, 8), { x: token, kt: token, vt: token });
  const a8: Parameters<typeof planFusions>[1] = { ...context, attentionCompute: "a8" };
  const fused = planFusions(nodes, a8);
  assertEquals(fused.counts.attentionPermute, 3, "precondition: the input permutes are absorbed");
  // The output permute (node 6) opens the next run of plain nodes.
  assertEquals(outline(enumerateUnfusedWindows(fused.steps, a8, 3)), [
    "neg,neg@0+2=qt",
    "permute,neg@6+2=c",
    "permute,neg,neg@6+3=d",
    "neg,neg@7+2=d",
  ]);
});

test("an attention that also absorbed its output permute (the online route, perf-ledger F-3 A4) advances the node index by its five nodes", () => {
  const token = [1, 8, 2, 64];
  const { nodes, context } = plan(attentionHintsGraph(8, 64), { x: token, kt: token, vt: token });
  const online: Parameters<typeof planFusions>[1] = {
    ...context,
    attentionCompute: "a8",
    attentionScoreStorage: "none",
    attentionDevice: { subgroups: true, maxComputeWorkgroupStorageSize: 32768 },
  };
  const fused = planFusions(nodes, online);
  assertEquals(fused.counts.attentionPermute, 4, "precondition: all four permutes are absorbed");
  // The attention step stands for nodes 2..6; the next run starts at node 7.
  assertEquals(outline(enumerateUnfusedWindows(fused.steps, online, 3)), [
    "neg,neg@0+2=qt",
    "neg,neg@7+2=d",
  ]);
});
