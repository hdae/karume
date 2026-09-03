/**
 * 融合候補の列挙（`enumerateUnfusedWindows`）の単体検証。GPU 不要・合成 IR だけを使う。
 *
 * 見るのは窓の作り方 =「どの連続窓を候補にし、どれを落とすか」で、適格条件（単一出力 /
 * state 不触 / f32 / 鎖の内部値が外へ出ない）の 1 本ずつに対照を置く。
 *
 * 実資産に掛けた答え合わせ（既知のヒット数の再現・二重計上の検出）と集計は、道具側の
 * tools/fusion-hints/enumerate_test.ts にある — 資産の発見・束縛・集計は道具の面で、
 * runtime のテストから tools を import しない（ADR 0008 追記 2026-09-03）。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import {
  aliasesInput,
  enumerateUnfusedWindows,
  type ExecStep,
  planFusions,
  type UnfusedWindow,
} from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";

/** 判定に使う device の能力（WebGPU core 既定 — 128MiB / 65535）。 */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

const shaped = (name: string, shape: readonly number[]) => ({
  name,
  dtype: "f32",
  shape: [...shape],
});

/** グラフ JSON → 計画済みノード列 + 融合の判定文脈。 */
const plan = (graph: GraphJson, inputShapes: Readonly<Record<string, readonly number[]>>) => {
  const ir: IrGraph = parseIrGraph(JSON.stringify(graph));
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

/** 融合を切った計画（全ノードが素のステップ）。 */
const unfused = (nodes: ReturnType<typeof plan>["nodes"]): readonly ExecStep[] =>
  nodes.map((node) => ({ kind: "node", plan: node, aliasesInput: aliasesInput(node) }));

/** 窓の要約（op 名列 / 窓幅 / 先頭ノード位置 / 鎖の出力名）。 */
const outline = (windows: readonly UnfusedWindow[]): readonly string[] =>
  windows.map((window) =>
    `${window.ops.join(",")}@${window.nodeIndex}+${window.windowSize}=${window.outputName}`
  );

/**
 * 素の 3 ノード鎖（`neg → add → mul`）。どのルールの綴りにも当たらないので、計画は必ず素の
 * 列になる（= 列挙器の入力として素直な形）。
 */
const chainGraph = (options: {
  readonly extraConsumer?: boolean;
  readonly internalOutput?: boolean;
} = {}): GraphJson => {
  const shape = [4, 8];
  const nodes: GraphJson["nodes"] = [
    { op: "neg", ins: ["x"], outs: ["a"], attrs: {} },
    { op: "add", ins: ["a", "w"], outs: ["b"], attrs: {} },
    { op: "mul", ins: ["b", "w"], outs: ["c"], attrs: {} },
  ];
  const values: GraphJson["values"] = {
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
    version: 1,
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

Deno.test("素のノード列から長さ 2〜maxWindow の窓を全て列挙する", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  assertEquals(planFusions(nodes, context).steps.length, 3, "前提: どのルールも掴まない");
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 3)), [
    "neg,add@0+2=b",
    "neg,add,mul@0+3=c",
    "add,mul@1+2=c",
  ]);
  // 窓幅の上限は受理集合を絞るだけ（長さ 3 の窓が消える）。
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 2)), [
    "neg,add@0+2=b",
    "add,mul@1+2=c",
  ]);
});

Deno.test("鎖の内部値が外へ出る窓は候補にならない（別 consumer / graph output）", () => {
  for (
    const [label, graph] of [
      ["別 consumer", chainGraph({ extraConsumer: true })],
      ["graph output", chainGraph({ internalOutput: true })],
    ] as const
  ) {
    const { nodes, context } = plan(graph, CHAIN_INPUTS);
    // 残るのは `b` を**最終出力**にする窓だけ（融合ステップは最終出力を実体化するので、
    // 最終ノードの出力が外へ出ることは適格）。
    assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 4)), [
      "neg,add@0+2=b",
    ], label);
  }
});

/**
 * 窓内 passthrough（鎖に入らない窓内ノード）の切り出し。RoPE の cos / sin 表の
 * `sym_prefix_slice` と同じ形 — 鎖の `add` へ流れ込むが、他でも読まれる（消費者 2 本）ので
 * 鎖には入らず、窓幅だけが鎖より 1 大きくなる。
 */
Deno.test("窓内 passthrough は鎖から外れ、窓幅だけが鎖より大きくなる", () => {
  const shape = [4, 8];
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
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
      // t は窓内 passthrough（下の add と、窓の外の mul の 2 本が読む）。
      { op: "neg", ins: ["w"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["a", "t"], outs: ["b"], attrs: {} },
      { op: "mul", ins: ["b", "w"], outs: ["c"], attrs: {} },
      { op: "mul", ins: ["t", "w"], outs: ["e"], attrs: {} },
    ],
  };
  const { nodes, context } = plan(graph, CHAIN_INPUTS);
  const windows = enumerateUnfusedWindows(unfused(nodes), context, 4);
  // `neg,add` は窓幅 3（passthrough の neg を挟む）で拾える。
  assertEquals(outline(windows).includes("neg,add@0+3=b"), true, outline(windows).join(" / "));
  // passthrough 自身を先頭にした窓（`neg,add` の重複計上）は出ない — 鎖の先頭が窓の先頭で
  // ない窓は落とすため。
  assertEquals(
    windows.filter((window) => window.ops.join(",") === "neg,add").map((w) => w.nodeIndex),
    [0],
  );
});

Deno.test("多出力 op を含む窓は候補にならない", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
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

Deno.test("state を触るノードを含む窓は候補にならない", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  // 現行の op 語彙では `states` 欄を持てるのが `attention` / `state_append` だけで、この綴りの
  // 窓は実グラフから踏めない。窓の仕組み側の不変条件として、合成ノードで判定点を固定する。
  const tainted = nodes.map((node, index) =>
    index === 1 ? { ...node, node: { ...node.node, states: { k: "kv" } } } : node
  );
  assertEquals(outline(enumerateUnfusedWindows(unfused(tainted), context, 3)), []);
});

Deno.test("f32 でない鎖は候補にならない", () => {
  const reshapeGraph = (dtype: "f32" | "i32"): GraphJson => ({
    format: "karume-ir",
    version: 1,
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
  // 対照: 同じ綴りの f32 なら 3 本出る（i32 の 0 が「そもそも窓が無い」でないことの裏）。
  assertEquals(enumerateUnfusedWindows(unfused(f32.nodes), f32.context, 3).length, 3);
  const i32 = plan(reshapeGraph("i32"), { x: [4, 8] });
  assertEquals(enumerateUnfusedWindows(unfused(i32.nodes), i32.context, 3), []);
});

Deno.test("融合ステップを跨いだ窓は作らない（掴めている鎖の二重計上なし）", () => {
  const shape = [4, 8];
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
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
  assertEquals(fused.counts.silu, 1, "前提: SiLU が掴めている");
  // 融合を切れば sigmoid,mul も候補に出る。
  assertEquals(outline(enumerateUnfusedWindows(unfused(nodes), context, 2)), [
    "sigmoid,mul@0+2=y",
    "mul,neg@1+2=z",
    "neg,add@2+2=o",
  ]);
  // 現行計画では融合ステップを跨がないので、残る素のノードの走りだけが候補になる。
  // `nodeIndex` は畳んだノード数だけ進む（融合ステップ = 2 ノード → neg は 2）。
  assertEquals(outline(enumerateUnfusedWindows(fused.steps, context, 2)), ["neg,add@2+2=o"]);
});

Deno.test("窓幅の上限が 2 未満なら fail loudly", () => {
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
