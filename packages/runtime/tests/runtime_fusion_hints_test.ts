/**
 * 融合候補の列挙（`enumerateUnfusedWindows`）と、それを実資産に掛ける道具
 * （tools/fusion-hints）の答え合わせ。GPU 不要。
 *
 * 列挙器が背負う性質は 2 つで、どちらも「数えそこない」と「数えすぎ」の対になっている:
 *
 * 1. **融合を切った計画**に掛けると、現行 5 ルールが実資産で掴んでいる鎖が候補として
 *    そのまま出る（op 名列 × 本数が既知のヒット数と一致する）。出なければ列挙器が窓を
 *    取りこぼしている。
 * 2. **現行計画**に掛けると、既に掴めている鎖は 1 本も出ない（融合ステップを跨いだ窓を
 *    作らない = 二重計上しない）。
 *
 * 実資産の期待値の出典は tests/assets_fusion_counts_test.ts（同じ資産・同じ読み方）。資産の
 * 無い環境は理由を出して**明示 SKIP** する。
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
import {
  bindGraphSymbols,
  discoverGraphs,
  readIrGraph,
} from "../../../tools/fusion-hints/assets.ts";
import { aggregate, enumerateGraph } from "../../../tools/fusion-hints/enumerate.ts";

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

Deno.test("候補の集計は op 名列ごとに数え、同じ先頭からの最長窓を極大として印す", () => {
  const { nodes, context } = plan(chainGraph(), CHAIN_INPUTS);
  const rows = aggregate(enumerateUnfusedWindows(unfused(nodes), context, 3));
  assertEquals(
    rows.map((row) => [row.ops.join(","), row.count, row.maximal, row.windowSizes]),
    [
      // 同数なら鎖の長い順。`neg,add` は同じ先頭（node 0）に長さ 3 の窓があるので極大でない。
      ["neg,add,mul", 1, 1, [3]],
      ["add,mul", 1, 1, [2]],
      ["neg,add", 1, 0, [2]],
    ],
  );
});

// ------------------------------------------------------- 実資産の答え合わせ

/**
 * 資産 1 件ぶんの門。`expected` は**融合を切った計画**で出るべき候補（op 名列 → 本数）、
 * `fusedExpected` は**現行計画**で残る本数（掴めている鎖はここで 0 になる）。
 */
type AssetCase = {
  readonly source: string;
  readonly graph: string;
  readonly binds: Readonly<Record<string, number>>;
  readonly expected: Readonly<Record<string, number>>;
  readonly fusedExpected: Readonly<Record<string, number>>;
};

const SILU = "sigmoid,mul";
/** RoPE の 7 ノード窓（`mul` 先行形と、後置形 = 実際のノード順が違う 2 綴り）。 */
const ROPE_DIRECT_FIRST = "mul,slice,slice,neg,cat,mul,add";
const ROPE_DIRECT_LAST = "slice,slice,neg,cat,mul,mul,add";
/** adaLN の鎖（窓内 passthrough の reshape を除いた並び）。 */
const ADALN = "layer_norm,reshape,reshape,add,mul,add";
const ROW_BLOCK_ATTENTION = "bmm,reshape,add,safe_softmax,expand,reshape,expand,reshape,bmm";

const ASSET_CASES: readonly AssetCase[] = [
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/transformer",
    binds: { S: 4096 },
    expected: { [ROPE_DIRECT_LAST]: 56, [ADALN]: 85, [SILU]: 2 },
    fusedExpected: { [ROPE_DIRECT_LAST]: 0, [ADALN]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/text_encoder",
    binds: { T: 64 },
    // rope 56 のうち 1 本は窓幅 8（cos / sin 表の `sym_prefix_slice` を窓内 passthrough として
    // 跨ぐ形）。窓内 passthrough を切り出せないと 55 になる。
    expected: { [ROPE_DIRECT_FIRST]: 56, [SILU]: 28 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "models/karume-anima",
    graph: "anima-turbo-v1.1/vae_decoder",
    binds: {},
    expected: { [SILU]: 29 },
    fusedExpected: { [SILU]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    graph: "v4-small/dit",
    binds: { S: 750 },
    // silu 29 = 掴めている 17 + ゲート 12（`mul(v, sigmoid(u))` — 自分自身に掛からないので
    // SILU_RULE の受理集合の外）。op 名列の n-gram はルールの受理集合より広い。
    expected: { [SILU]: 29, [ROW_BLOCK_ATTENTION]: 12 },
    fusedExpected: { [SILU]: 12, [ROW_BLOCK_ATTENTION]: 0 },
  },
  {
    source: "models/karume-irodori-v4-small",
    graph: "v4-small/backbone",
    binds: { T: 256 },
    expected: { [ROPE_DIRECT_FIRST]: 50 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/embeddinggemma-300m",
    graph: "embeddinggemma-300m",
    binds: { T: 318 },
    expected: { [ROPE_DIRECT_FIRST]: 48 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0 },
  },
  {
    source: "outputs/series/minicpm5-1b-decode",
    graph: "minicpm5-1b-decode",
    binds: { M: 1, C: 640 },
    expected: { [ROPE_DIRECT_FIRST]: 48, [SILU]: 24 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 0, [SILU]: 0 },
  },
  {
    source: "outputs/series/gemma4-e2b-decode",
    graph: "gemma4-e2b-decode",
    binds: { M: 1, C: 640 },
    // **既知の穴**: 綴りは 50 箇所とも並ぶのに計画は 15 本しか掴まない（機序は未特定 —
    // docs/research/2026-08-30-gemma4-decode-wallclock.md §4）。残る 35 本が候補に出る。
    expected: { [ROPE_DIRECT_FIRST]: 50 },
    fusedExpected: { [ROPE_DIRECT_FIRST]: 35 },
  },
];

const REPO = new URL("../../../", import.meta.url);

/**
 * 資産の有無。
 * MUST: NotFound 以外は伝播させる — 全 I/O エラーを「未生成」に丸めると、資産ルートの
 * マウント異常が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
const exists = async (url: URL): Promise<boolean> => {
  try {
    await Deno.stat(url);
    return true;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const AVAILABLE = new Map<string, boolean>();
for (const source of new Set(ASSET_CASES.map((entry) => entry.source))) {
  const present = await exists(new URL(source, REPO));
  AVAILABLE.set(source, present);
  if (!present) {
    console.warn(
      `[karume] ${source} が無いため融合候補の答え合わせを SKIP する` +
        "（列挙器の取りこぼしは実資産でしか検出できない）",
    );
  }
}
const ASSETS_AVAILABLE = [...AVAILABLE.values()].every((present) => present);

/** 資産 1 件を計画して、`keys` の op 名列ごとの本数を引く。 */
const assetCounts = async (
  entry: AssetCase,
  fused: boolean,
  keys: readonly string[],
): Promise<Record<string, number>> => {
  const root = new URL(`${entry.source}/`, REPO);
  const sources = await discoverGraphs(root);
  const found = sources.find((source) => source.name === entry.graph);
  if (found === undefined) {
    throw new Error(
      `${entry.source} に '${entry.graph}' が無い（見えたのは ${
        sources.map((source) => source.name).join(" / ")
      }）`,
    );
  }
  const graph = await readIrGraph(found.url);
  const bound = bindGraphSymbols(graph, entry.binds, undefined);
  const rows = aggregate(enumerateGraph(graph, bound, { maxWindow: 9, fused }).windows);
  const counts: Record<string, number> = Object.fromEntries(keys.map((key) => [key, 0]));
  for (const row of rows) {
    const key = row.ops.join(",");
    if (Object.hasOwn(counts, key)) counts[key] = row.count;
  }
  return counts;
};

Deno.test({
  name: "実資産: 融合を切った計画に掛けると既知のヒット数が候補として再現する",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    for (const entry of ASSET_CASES) {
      const keys = Object.keys(entry.expected);
      assertEquals(await assetCounts(entry, false, keys), entry.expected, entry.graph);
    }
  },
});

Deno.test({
  name: "実資産: 現行計画では掴めている鎖が候補に出ない（二重計上の検出）",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    for (const entry of ASSET_CASES) {
      const keys = Object.keys(entry.fusedExpected);
      assertEquals(await assetCounts(entry, true, keys), entry.fusedExpected, entry.graph);
    }
  },
});
