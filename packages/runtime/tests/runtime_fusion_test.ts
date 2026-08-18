/**
 * 融合パス（src/runtime/fusion.ts）の判定だけを GPU 非依存で固定する。
 *
 * 融合の受理集合は「掴めた形は速く、掴めない形は素の列で必ず正しい」という 2 段構えなので、
 * **反例側の網羅がそのまま安全性の根拠**になる。実 GPU テスト（gpu_*_fusion_test.ts）は
 * 値のビット parity を見る役で、こちらは反例を GPU 抜きで細かく積む役。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { ADALN_NORM_KEY, adalnNormParams } from "../src/kernels/adaln-norm.ts";
import { bmmKey } from "../src/kernels/bmm.ts";
import { ROPE_KEY } from "../src/kernels/rope.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { SAFE_SOFTMAX_KEY } from "../src/kernels/softmax.ts";
import { UPSAMPLE_2X_KEY } from "../src/kernels/upsample2x.ts";
import {
  type ExecStep,
  FUSION_RULES,
  type FusionPlan,
  planFusions,
  planRowBlocks,
} from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";

const parse = (graph: GraphJson): IrGraph => parseIrGraph(JSON.stringify(graph));

/**
 * 判定に使う device の能力（WebGPU core 既定 — 128MiB / 65535）。行ブロック分割の枚数だけが
 * これを読む（既存 4 ルールは device の能力に依らない）。
 */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/** グラフ JSON → 融合済みステップ列（executor が run ごとに作るのと同じ入力）。 */
const fuse = (
  graph: GraphJson,
  inputShapes: Readonly<Record<string, readonly number[]>>,
): FusionPlan => {
  const ir = parse(graph);
  const plan = planGraph(ir, bindSymbols(ir, inputShapes));
  return planFusions(plan.nodes, {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
  });
};

/** ステップ列の要約（rule 名、素のノードは op 名）。 */
const outline = (steps: readonly ExecStep[]): readonly string[] =>
  steps.map((step) => step.kind === "fused" ? `fused:${step.rule}` : step.plan.node.op);

const fusedAt = (plan: FusionPlan, index: number) => {
  const step = plan.steps[index];
  if (step.kind !== "fused") throw new Error(`steps[${index}] が融合ステップでない`);
  return step;
};

Deno.test("ルール表の先頭 op は互いに素で、適用順が結果に効かない", () => {
  assertEquals(FUSION_RULES.map((rule) => rule.name), [
    "silu",
    "upsample2x",
    "rope",
    "adaln",
    "rowBlockAttention",
  ]);
  const seen = new Set<string>();
  for (const rule of FUSION_RULES) {
    for (const head of rule.heads) {
      assertEquals(seen.has(head), false, `先頭 op '${head}' が複数ルールで重なっている`);
      seen.add(head);
    }
  }
  assertEquals([...seen].sort(), ["bmm", "layer_norm", "mul", "reshape", "sigmoid", "slice"]);
});

// ---------------------------------------------------------------- SiLU

type SiluOptions = {
  readonly order?: "x-sigmoid" | "sigmoid-x";
  readonly interpose?: boolean;
  readonly sigmoidOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly gateWithOther?: boolean;
  readonly broadcast?: boolean;
};

const siluGraph = (options: SiluOptions = {}): GraphJson => {
  const shape = [4, 8];
  const values: GraphJson["values"] = {
    s: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  const inputs: GraphJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...shape] }];
  const nodes: GraphJson["nodes"] = [{ op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} }];
  let mulSigmoid = "s";
  if (options.interpose) {
    values.s_alias = { dtype: "f32", shape: [...shape] };
    nodes.push({ op: "reshape", ins: ["s"], outs: ["s_alias"], attrs: {} });
    mulSigmoid = "s_alias";
  }
  // broadcast は「片側だけ長さ 1 の軸」で、mul としては合法だが SiLU ではない。
  if (options.broadcast) inputs[0] = { name: "x", dtype: "f32", shape: [1, 8] };
  if (options.broadcast) values.s = { dtype: "f32", shape: [1, 8] };
  if (options.gateWithOther) inputs.push({ name: "z", dtype: "f32", shape: [...shape] });
  const mulIns = options.gateWithOther
    ? [mulSigmoid, "z"]
    : options.order === "sigmoid-x"
    ? [mulSigmoid, "x"]
    : options.broadcast
    ? ["x2", mulSigmoid]
    : ["x", mulSigmoid];
  if (options.broadcast) inputs.push({ name: "x2", dtype: "f32", shape: [...shape] });
  nodes.push({ op: "mul", ins: mulIns, outs: ["y"], attrs: {} });
  if (options.extraConsumer) {
    values.s_copy = { dtype: "f32", shape: [...(options.broadcast ? [1, 8] : shape)] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [
      ...(options.sigmoidOutput ? ["s"] : []),
      "y",
      ...(options.extraConsumer ? ["s_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

const siluInputs = (options: SiluOptions = {}): Readonly<Record<string, readonly number[]>> => ({
  x: options.broadcast ? [1, 8] : [4, 8],
  ...(options.gateWithOther ? { z: [4, 8] } : {}),
  ...(options.broadcast ? { x2: [4, 8] } : {}),
});

Deno.test("SiLU は sigmoid→mul の両順を掴み、x を延べ 2 回消費するステップにする", () => {
  for (const order of ["x-sigmoid", "sigmoid-x"] as const) {
    const plan = fuse(siluGraph({ order }), siluInputs());
    assertEquals(outline(plan.steps), ["fused:silu"], order);
    const step = fusedAt(plan, 0);
    assertEquals(step.ins, ["x", "x"], `${order}: 外部入力の延べ列`);
    assertEquals(step.binds, ["x"], `${order}: bind 順`);
    assertEquals(step.nodeCount, 2, order);
    assertEquals(step.outputName, "y", order);
    assertEquals(step.dispatches[0].key, siluKey(order), order);
    assertEquals(plan.counts.silu, 1, order);
  }
});

Deno.test("SiLU の反例（別名 / 内部 output / 別 consumer / 別 gate / broadcast）は素の列へ落ちる", () => {
  const cases: readonly (readonly [string, SiluOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { sigmoidOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["mul(s,z)", { gateWithOther: true }],
    ["broadcast mul", { broadcast: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(siluGraph(options), siluInputs(options));
    assertEquals(plan.counts.silu, 0, `${label}: 融合カウンタ`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

/**
 * state を触るノードを含む窓は**全ルールで**掴まない（ADR 0067 決定 5b）。
 *
 * state 参照はテンソルのデータ辺を張らないので、融合が窓内 passthrough を前へ動かすと
 * `nodes` 配列順が崩れても shape 検査も参照計数も何も落ちない（「今 step の k/v を過去として
 * 二重に読む」が例外なしに出る）。
 *
 * NOTE: 現行の op 語彙では `states` 欄を持てるのが `attention` / `state_append` の 2 本だけで、
 * どちらも既存 5 ルールの窓の綴りに現れない — つまり**実グラフからは踏めない**。それでも窓の
 * 仕組み側の不変条件として置いてある門なので、掴めることが確定している窓（SiLU）の片方に
 * `states` 欄を差した合成ノードで、判定点が実際に効いていることを固定する。
 */
Deno.test("state を触るノードを含む窓はどのルールも掴まない（窓の仕組み側の不変条件）", () => {
  const ir = parse(siluGraph());
  const plan = planGraph(ir, bindSymbols(ir, siluInputs()));
  const context = {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
  };
  // 対照: 素のままなら掴める（下の 0 が「そもそも掴めない窓」でないことの裏）。
  assertEquals(planFusions(plan.nodes, context).counts.silu, 1);

  for (const tainted of [0, 1]) {
    const nodes = plan.nodes.map((node, index) =>
      index === tainted ? { ...node, node: { ...node.node, states: { k: "kv" } } } : node
    );
    const fused = planFusions(nodes, context);
    assertEquals(fused.counts.silu, 0, `nodes[${tainted}] に states 欄があるのに掴んだ`);
    assertEquals(
      fused.steps.every((step) => step.kind === "node"),
      true,
      `nodes[${tainted}]: ${outline(fused.steps).join(",")}`,
    );
  }
});

// ---------------------------------------------------------- upsample2x

type UpsampleOptions = {
  readonly scaleWidth?: number;
  readonly interpose?: boolean;
  readonly internalOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly dtype?: "f32" | "i32";
};

const upsampleGraph = (options: UpsampleOptions = {}): GraphJson => {
  const dtype = options.dtype ?? "f32";
  const scale = options.scaleWidth ?? 2;
  const outW = 7 * scale;
  const values: GraphJson["values"] = {
    a: { dtype, shape: [30, 7, 1] },
    wide: { dtype, shape: [30, 7, scale] },
    b: { dtype, shape: [6, 5, outW] },
    c: { dtype, shape: [6, 5, 1, outW] },
    tall: { dtype, shape: [6, 5, 2, outW] },
    y: { dtype, shape: [2, 3, 10, outW] },
  };
  const nodes: GraphJson["nodes"] = [{ op: "reshape", ins: ["x"], outs: ["a"], attrs: {} }];
  let expandSource = "a";
  if (options.interpose) {
    values.a_alias = { dtype, shape: [30, 7, 1] };
    nodes.push({ op: "reshape", ins: ["a"], outs: ["a_alias"], attrs: {} });
    expandSource = "a_alias";
  }
  nodes.push(
    { op: "expand", ins: [expandSource], outs: ["wide"], attrs: {} },
    { op: "reshape", ins: ["wide"], outs: ["b"], attrs: {} },
    { op: "reshape", ins: ["b"], outs: ["c"], attrs: {} },
    { op: "expand", ins: ["c"], outs: ["tall"], attrs: {} },
    { op: "reshape", ins: ["tall"], outs: ["y"], attrs: {} },
  );
  if (options.extraConsumer) {
    values.wide_copy = { dtype, shape: [30, 7, scale] };
    nodes.push({ op: "neg", ins: ["wide"], outs: ["wide_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [{ name: "x", dtype, shape: [2, 3, 5, 7] }],
    outputs: [
      ...(options.internalOutput ? ["wide"] : []),
      "y",
      ...(options.extraConsumer ? ["wide_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

Deno.test("VAE nearest x2 は 6 ノード鎖を掴み、外部入力 1 本だけを消費する", () => {
  const plan = fuse(upsampleGraph(), { x: [2, 3, 5, 7] });
  assertEquals(outline(plan.steps), ["fused:upsample2x"]);
  const step = fusedAt(plan, 0);
  assertEquals(step.ins, ["x"]);
  assertEquals(step.binds, ["x"]);
  assertEquals(step.nodeCount, 6);
  assertEquals(step.dispatches[0].key, UPSAMPLE_2X_KEY);
  // params は [入力要素数, width, 出力 width, reserved]
  assertEquals([...step.dispatches[0].params], [210, 7, 14, 0]);
  assertEquals(plan.counts.upsample2x, 1);
});

Deno.test("upsample x2 の反例（near-shape / 別名 / 内部 output / 別 consumer / dtype）は掴まない", () => {
  const cases: readonly (readonly [string, UpsampleOptions])[] = [
    ["x3 near-shape", { scaleWidth: 3 }],
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["i32 dtype", { dtype: "i32" }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(upsampleGraph(options), { x: [2, 3, 5, 7] });
    assertEquals(plan.counts.upsample2x, 0, `${label}: 融合カウンタ`);
  }
});

// ---------------------------------------------------------------- RoPE

type RopeOptions = {
  readonly order?: "slice-first" | "direct-first";
  readonly heads?: number;
  readonly interpose?: boolean;
  readonly internalOutput?: boolean;
  readonly extraConsumer?: boolean;
  /** 半分割の位置をずらす（head 幅の 1/4 で切る — 「式は同じだが半分割でない」反例）。 */
  readonly nearSplit?: boolean;
  /** head 幅（実測は 128（Anima）と 256（Gemma 系）。既定 128）。 */
  readonly headDim?: number;
  /** batch を 2 にする（受理形は B=1 決め打ち）。 */
  readonly batch2?: boolean;
  /** add の入力順を入れ替える（値は同じで結線パターンが違う反例）。 */
  readonly swappedAdd?: boolean;
  /** cos / sin の table を x と同形にする（broadcast されない反例）。 */
  readonly nearTable?: boolean;
  /**
   * sin 表を Tmax 定数からの `sym_prefix_slice` で作り、`cat` と cross mul の間に置く
   * （実測形 — 表は θ 系統ごとに 1 度だけ作るので、初出の 1 箇所だけ窓内 passthrough になる）。
   */
  readonly prefixSlicedSin?: boolean;
  /** 同じ隙間に `sym_prefix_slice` 以外（reshape）を挟む反例。 */
  readonly gapReshape?: boolean;
};

const ropeGraph = (options: RopeOptions = {}): GraphJson => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const split = headDim / (options.nearSplit ? 4 : 2);
  const full = [batch, heads, 5, headDim];
  const low = [batch, heads, 5, split];
  const high = [batch, heads, 5, headDim - split];
  const table = options.nearTable ? full : [1, 1, 5, headDim];
  const values: GraphJson["values"] = {
    first: { dtype: "f32", shape: low },
    second: { dtype: "f32", shape: high },
    negative: { dtype: "f32", shape: high },
    rotated: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const direct: GraphJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: GraphJson["nodes"] = [];
  if (options.order === "direct-first") nodes.push(direct);
  nodes.push({ op: "slice", ins: ["x"], outs: ["first"], attrs: { dim: 3, start: 0, end: split } });
  let catFirst = "first";
  if (options.interpose) {
    values.first_alias = { dtype: "f32", shape: low };
    nodes.push({ op: "reshape", ins: ["first"], outs: ["first_alias"], attrs: {} });
    catFirst = "first_alias";
  }
  nodes.push(
    { op: "slice", ins: ["x"], outs: ["second"], attrs: { dim: 3, start: split, end: headDim } },
    { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
    { op: "cat", ins: ["negative", catFirst], outs: ["rotated"], attrs: { dim: 3 } },
  );
  // 窓内 passthrough の位置は `cat` の直後（実測形）。sym_prefix_slice なら掴めたまま、
  // それ以外の op が来れば窓が崩れて掴めなくなることを 2 つの選択肢で分ける。
  if (options.prefixSlicedSin) {
    // Tmax 形（S=8）の定数を実行時の T=5 へ縮める。宣言 shape が記号を含まないことが
    // sym_prefix_slice の契約なので、表側は静的形で置く。
    values.sin_table = { dtype: "f32", shape: [1, 1, 8, headDim] };
    values.sin = { dtype: "f32", shape: [1, 1, "T", headDim] };
    nodes.push({
      op: "sym_prefix_slice",
      ins: ["sin_table"],
      outs: ["sin"],
      attrs: { sym: "T", slices: [{ dim: 2, coeff: 1, offset: 0 }] },
    });
  }
  if (options.gapReshape) {
    values.cos_view = { dtype: "f32", shape: [table[0], table[1] * table[2], table[3]] };
    nodes.push({ op: "reshape", ins: ["cos"], outs: ["cos_view"], attrs: {} });
  }
  if (options.order !== "direct-first") nodes.push(direct);
  nodes.push(
    { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["cross", "direct"] : ["direct", "cross"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.extraConsumer) {
    values.first_copy = { dtype: "f32", shape: low };
    nodes.push({ op: "neg", ins: ["first"], outs: ["first_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    // 記号は sym_prefix_slice を置く形でだけ要る（T は補助入力 `bind` から束縛する）。
    symbols: options.prefixSlicedSin ? ["T"] : [],
    inputs: [
      { name: "x", dtype: "f32", shape: full },
      { name: "cos", dtype: "f32", shape: table },
      ...(options.prefixSlicedSin
        ? [{ name: "bind", dtype: "f32", shape: ["T"] }]
        : [{ name: "sin", dtype: "f32", shape: table }]),
    ],
    outputs: [
      ...(options.internalOutput ? ["first"] : []),
      "y",
      ...(options.gapReshape ? ["cos_view"] : []),
      ...(options.extraConsumer ? ["first_copy"] : []),
    ],
    initializers: options.prefixSlicedSin
      ? { sin_table: { tensor: "sin_table", storage: { dtype: "f32" } } }
      : {},
    values,
    nodes,
  };
};

const ropeInputs = (options: RopeOptions = {}): Readonly<Record<string, readonly number[]>> => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const full = [batch, heads, 5, headDim];
  const table = options.nearTable ? full : [1, 1, 5, headDim];
  return options.prefixSlicedSin
    ? { x: full, cos: table, bind: [5] }
    : { x: full, cos: table, sin: table };
};

Deno.test("half-split RoPE は 2 順序を掴み、x を延べ 3 回・cos / sin を 1 回ずつ消費する", () => {
  for (
    const [order, expectedIns] of [
      ["slice-first", ["x", "x", "x", "cos", "sin"]],
      ["direct-first", ["x", "cos", "x", "x", "sin"]],
    ] as const
  ) {
    const plan = fuse(ropeGraph({ order, heads: 4 }), ropeInputs({ heads: 4 }));
    assertEquals(outline(plan.steps), ["fused:rope"], order);
    const step = fusedAt(plan, 0);
    // MUST: 延べ列は**実際のノード順**（direct-first は cos が 2 番目に来る）。
    assertEquals(step.ins, expectedIns, `${order}: 外部入力の延べ列`);
    assertEquals(step.binds, ["x", "cos", "sin"], `${order}: bind 順`);
    assertEquals(step.nodeCount, 7, order);
    assertEquals(step.dispatches[0].key, ROPE_KEY, order);
    // params は [全要素数, sequence, head_dim, half_dim]
    assertEquals([...step.dispatches[0].params], [1 * 4 * 5 * 128, 5, 128, 64], order);
    assertEquals(plan.counts.rope, 1, order);
  }
});

// 半分割の位置は head 幅から導く（カーネルは head_dim / half_dim を uniform で受けるので、
// 幅が変わっても WGSL は 1 バイトも変わらない）。実測は Anima の 128 と Gemma 系の 256 だが、
// 「掴める幅」が実測 2 点の決め打ちに戻っていないことは別幅の 64 で押さえる。
Deno.test("half-split RoPE は head 幅を分割位置から導き、幅ごとに params だけが変わる", () => {
  for (const headDim of [64, 128, 256]) {
    const options: RopeOptions = { order: "direct-first", heads: 4, headDim };
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(outline(plan.steps), ["fused:rope"], `head 幅 ${headDim}`);
    const step = fusedAt(plan, 0);
    assertEquals(step.dispatches[0].key, ROPE_KEY, `head 幅 ${headDim}: カーネルは 1 本`);
    assertEquals(
      [...step.dispatches[0].params],
      [4 * 5 * headDim, 5, headDim, headDim / 2],
      `head 幅 ${headDim}: params`,
    );
  }
});

// 実測形: cos / sin の表は θ 系統ごとに 1 度だけ Tmax 定数から切り出すので、その初出だけが
// 鎖の隙間（cat と cross mul の間）に落ちる。窓内 passthrough として融合ステップの**前**へ
// 動かせなければ、その 1 箇所だけ黙って融合が外れる。
Deno.test("RoPE は cat 直後の sym_prefix_slice を窓内 passthrough として跨ぐ", () => {
  const options: RopeOptions = { order: "direct-first", heads: 4, prefixSlicedSin: true };
  const plan = fuse(ropeGraph(options), ropeInputs(options));
  // MUST: passthrough が先（融合ステップは その出力 `sin` を入力に取る）。
  assertEquals(outline(plan.steps), ["sym_prefix_slice", "fused:rope"]);
  const step = fusedAt(plan, 1);
  assertEquals(step.ins, ["x", "cos", "x", "x", "sin"], "外部入力の延べ列");
  assertEquals(step.binds, ["x", "cos", "sin"], "bind 順");
  // 畳んだのは 7 本（走査幅 8 と別物 — passthrough は素のノードのまま残る）。
  assertEquals(step.nodeCount, 7);
  assertEquals(plan.counts.rope, 1);
});

Deno.test("RoPE の反例（別名 / 内部 output / 別 consumer / 分割位置 / batch / add 順 / table 形 / 隙間の別 op）は掴まない", () => {
  const cases: readonly (readonly [string, RopeOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["32/96 split", { nearSplit: true }],
    ["batch 2", { batch2: true }],
    ["add(cross, direct)", { swappedAdd: true }],
    // head ごとに別表を渡す形（カーネルは token*head_dim+d と引くので黙って誤値になる）
    ["per-head table", { nearTable: true, heads: 4 }],
    // 隙間に入れるのは sym_prefix_slice だけ（「何でも 1 本なら跨ぐ」に広げない）
    ["gap reshape", { gapReshape: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(plan.counts.rope, 0, `${label}: 融合カウンタ`);
  }
});

// -------------------------------------------------------- identity expand

const expandGraph = (outShape: readonly number[]): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "expand", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [3, 1] }],
  outputs: ["y", "r"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [3, 1] },
    y: { dtype: "f32", shape: [...outShape] },
    r: { dtype: "f32", shape: [1, 3] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "expand", ins: ["h"], outs: ["y"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["r"], attrs: {} },
  ],
});

Deno.test("恒等 expand だけが 0 dispatch の別名になり、複製軸があれば実体化コピーへ戻る", () => {
  const identity = fuse(expandGraph([3, 1]), { x: [3, 1] });
  assertEquals(identity.steps.map((step) => step.kind === "node" && step.aliasesInput), [
    false, // neg
    true, // 恒等 expand
    true, // reshape は無条件に別名
  ]);
  assertEquals(identity.counts.identityExpand, 1);

  const replicated = fuse(expandGraph([3, 4]), { x: [3, 1] });
  assertEquals(replicated.steps.map((step) => step.kind === "node" && step.aliasesInput), [
    false,
    false, // 1 軸でも複製があれば別名化しない
    true,
  ]);
  assertEquals(replicated.counts.identityExpand, 0);
});

// --------------------------------------------------------------- adaLN

const ROWS = 4;
const DIM = 8;

type AdalnOptions = {
  /**
   * 変調ベクトルを `reshape` で作る本数（= 窓内 passthrough の本数）。実測は 3（shift /
   * scale / gate）と、末層だけ 2（gate 無し）。減らした分は直接 graph input にする。
   */
  readonly gaps?: number;
  /** passthrough を 1 本増やし、それが layer_norm 出力を消費する（並べ替えが非合法）。 */
  readonly gapConsumesNorm?: boolean;
  /** layer_norm 出力を graph output にする。 */
  readonly normOutput?: boolean;
  /** layer_norm 出力に別 consumer を足す。 */
  readonly normExtraConsumer?: boolean;
  /** `1 + scale` の結果に別 consumer を足す。 */
  readonly modulationExtraConsumer?: boolean;
  /** mul の入力順を入れ替える。 */
  readonly swappedMul?: boolean;
  /** shift を足す add の入力順を入れ替える。 */
  readonly swappedAdd?: boolean;
  /** `1` の定数を `[1]` でなく `[dim]` にする（カーネルは `one[0]` しか読まない）。 */
  readonly wideOne?: boolean;
  /** 変調を broadcast でなく行ごとにする（near-shape）。 */
  readonly perRowScale?: boolean;
  /** layer_norm の weight と bias に同じ値名を渡す（bind 面が重複する形）。 */
  readonly sharedAffine?: boolean;
};

/**
 * 実 IR（anima DiT）の adaLN 鎖。`layer_norm` と `mul` の間に変調ベクトルの `reshape` が
 * 挟まる**非隣接**の形をそのまま作る。
 */
const adalnGraph = (options: AdalnOptions = {}): GraphJson => {
  const dtype = "f32";
  const gaps = options.gaps ?? 3;
  const row = [1, ROWS, DIM];
  const modulation = options.perRowScale ? row : [1, 1, DIM];
  const biasName = options.sharedAffine ? "ln_weight" : "ln_bias";
  const values: GraphJson["values"] = {
    t: { dtype, shape: [...row] },
    s: { dtype, shape: [...modulation] },
    p: { dtype, shape: [...row] },
    y: { dtype, shape: [...row] },
  };
  const inputs: GraphJson["inputs"] = [
    { name: "x", dtype, shape: [...row] },
    { name: "ln_weight", dtype, shape: [DIM] },
    ...(options.sharedAffine ? [] : [{ name: "ln_bias", dtype, shape: [DIM] }]),
    { name: "one", dtype, shape: options.wideOne ? [DIM] : [1] },
  ];
  const nodes: GraphJson["nodes"] = [{
    op: "layer_norm",
    ins: ["x", "ln_weight", biasName],
    outs: ["t"],
    attrs: { normalized_shape: [DIM], eps: 0.000001 },
  }];
  const extraOutputs: string[] = [];

  /** 変調ベクトル 1 本。`viaReshape` なら窓内 passthrough に、でなければ直接 graph input に。 */
  const declare = (name: string, shape: readonly number[], viaReshape: boolean): void => {
    if (!viaReshape) {
      inputs.push({ name, dtype, shape: [...shape] });
      return;
    }
    inputs.push({ name: `${name}_src`, dtype, shape: [shape[1], shape[2]] });
    values[name] = { dtype, shape: [...shape] };
    nodes.push({ op: "reshape", ins: [`${name}_src`], outs: [name], attrs: {} });
  };

  if (options.gapConsumesNorm) {
    // 窓内 passthrough が鎖の値を読む形。融合ステップより前へ動かせない（同時に
    // layer_norm 出力の consumer も 2 本になり、内部値の private 条件も外れる）。
    values.norm_alias = { dtype, shape: [...row] };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["norm_alias"], attrs: {} });
    extraOutputs.push("norm_alias");
  }
  // 実測の並び（shift → scale → gate）。gate は鎖の外で消費される。
  declare("shift", [1, 1, DIM], gaps >= 1);
  declare("scale", modulation, gaps >= 2);
  for (const [at, name] of ["gate", "spare"].entries()) {
    if (gaps < 3 + at) continue;
    declare(name, [1, 1, DIM], true);
    extraOutputs.push(name);
  }

  nodes.push(
    { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: options.swappedMul ? ["s", "t"] : ["t", "s"], outs: ["p"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["shift", "p"] : ["p", "shift"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.normExtraConsumer) {
    values.t_copy = { dtype, shape: [...row] };
    nodes.push({ op: "neg", ins: ["t"], outs: ["t_copy"], attrs: {} });
    extraOutputs.push("t_copy");
  }
  if (options.modulationExtraConsumer) {
    values.s_copy = { dtype, shape: [...modulation] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
    extraOutputs.push("s_copy");
  }
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.normOutput ? ["t"] : []), "y", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

/** 宣言 shape がそのまま実 shape（記号次元を使っていない）。 */
const adalnInputs = (graph: GraphJson): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(graph.inputs.map((spec) => [spec.name, spec.shape as readonly number[]]));

const fuseAdaln = (options: AdalnOptions = {}): FusionPlan => {
  const graph = adalnGraph(options);
  return fuse(graph, adalnInputs(graph));
};

Deno.test("adaLN は窓 7 / 窓 6 を掴み、reshape は融合ステップの前に素のまま並ぶ", () => {
  for (const gaps of [3, 2] as const) {
    const label = `窓 ${gaps + 4}`;
    const plan = fuseAdaln({ gaps });
    assertEquals(outline(plan.steps), [...Array(gaps).fill("reshape"), "fused:adaln"], label);
    const step = fusedAt(plan, gaps);
    // 外部入力の延べ列は**畳んだ 4 ノードだけ**から出る（passthrough は素のノードが数える）。
    const operands = ["x", "ln_weight", "ln_bias", "scale", "one", "shift"];
    assertEquals(step.ins, operands, `${label}: 外部入力の延べ列`);
    assertEquals(step.binds, operands, `${label}: bind 順（カーネルの binding 1〜6）`);
    assertEquals(step.nodeCount, 4, `${label}: 畳んだ本数は窓幅ではない`);
    assertEquals(step.outputName, "y", label);
    assertEquals(step.outputShape, [1, ROWS, DIM], label);
    assertEquals(step.dispatches[0].key, ADALN_NORM_KEY, label);
    // params は素の layer_norm と同一（rows / dim / eps のビット列）。
    assertEquals([...step.dispatches[0].params], [...adalnNormParams(ROWS, DIM, 1e-6)], label);
    assertEquals(
      step.dispatches[0].workgroups,
      { kind: "gridStride", items: ROWS, size: 1 },
      `${label}: 1 行 = 1 workgroup・行数がそのまま workgroup 数`,
    );
    assertEquals(plan.counts.adaln, 1, label);
  }
});

Deno.test("adaLN の窓内 passthrough は他ルールの受理位置を潰さない", () => {
  const graph = adalnGraph();
  const ir = parse(graph);
  const plan = planGraph(ir, bindSymbols(ir, adalnInputs(graph)));
  const context = {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
  };
  // 窓（layer_norm + reshape×3 + add + mul + add）の**全ての開始位置**で adaln 以外が
  // 1 つも掴まないことを見る。`reshape` は upsample2x の先頭 op でもあるので、窓ごと
  // 読み飛ばすことで他ルールの機会を奪っていないかはここでしか分からない。
  for (let index = 0; index < 7; index += 1) {
    for (const rule of FUSION_RULES) {
      if (rule.name === "adaln") continue;
      assertEquals(
        rule.apply(plan.nodes, index, context),
        undefined,
        `窓の ${index} 番目で '${rule.name}' が掴んでいる`,
      );
    }
  }
});

// NOTE: dtype 違いの反例は**構成できない** — layer_norm の契約が f32 専業で、続く
// add / mul も dtype 一様を要求するので、planGraph が融合パスより先に落とす。
// `allF32` は全ルール共通の適格条件として残す（層が変われば効き始める）。
Deno.test("adaLN の反例（gap の本数 / gap が norm を消費 / 内部 output / 別 consumer / 入力順 / 定数形 / near-shape / bind 重複）は掴まない", () => {
  const cases: readonly (readonly [string, AdalnOptions])[] = [
    ["gap 無し（隣接鎖）", { gaps: 0 }],
    ["gap 1 本", { gaps: 1 }],
    ["gap 4 本", { gaps: 4 }],
    ["gap が layer_norm 出力を消費", { gapConsumesNorm: true }],
    ["layer_norm 出力が graph output", { normOutput: true }],
    ["layer_norm 出力に別 consumer", { normExtraConsumer: true }],
    ["1 + scale に別 consumer", { modulationExtraConsumer: true }],
    ["mul(s, t)", { swappedMul: true }],
    ["add(shift, p)", { swappedAdd: true }],
    ["定数が [dim]", { wideOne: true }],
    ["行ごとの scale（broadcast でない）", { perRowScale: true }],
    ["weight と bias が同じ値名", { sharedAffine: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseAdaln(options);
    assertEquals(plan.counts.adaln, 0, `${label}: 融合カウンタ`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

Deno.test("カウンタは融合が並んだグラフでルール別に積み上がる", () => {
  const graph = ropeGraph({ heads: 2 });
  // RoPE 鎖の直後に SiLU 鎖を継ぎ足す（先頭 op が互いに素なので順に掴まれる）。
  graph.values.sig = { dtype: "f32", shape: [1, 2, 5, 128] };
  graph.values.gated = { dtype: "f32", shape: [1, 2, 5, 128] };
  graph.nodes.push(
    { op: "sigmoid", ins: ["y"], outs: ["sig"], attrs: {} },
    { op: "mul", ins: ["y", "sig"], outs: ["gated"], attrs: {} },
  );
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  graph.outputs = ["gated"];

  const plan = fuse(graph, ropeInputs({ heads: 2 }));
  assertEquals(outline(plan.steps), ["fused:rope", "fused:silu"]);
  assertEquals(plan.counts, {
    silu: 1,
    upsample2x: 0,
    rope: 1,
    adaln: 0,
    rowBlockAttention: 0,
    identityExpand: 0,
  });
});

// ------------------------------------------- 行ブロック attention（rowBlockAttention）

type AttentionShape = {
  readonly heads: number;
  readonly queries: number;
  readonly keys: number;
  readonly headDim: number;
};

const ATTENTION: AttentionShape = { heads: 3, queries: 17, keys: 19, headDim: 13 };

/** 窓の 1 点だけを壊すためのつまみ（既定は実測どおりの綴り）。 */
type AttentionOptions = {
  /** bmm と reshape の間に 0 dispatch の別名を 1 本挟む（隣接条件だけを外す）。 */
  readonly interpose?: boolean;
  /** mask を `[1,1,M,N]`（行ごと）にする。 */
  readonly rowMask?: boolean;
  /** safe_softmax を素の softmax にする。 */
  readonly plainSoftmax?: boolean;
  /** V 側の expand を**複製軸を持つ** broadcast にする（恒等でなくなる）。 */
  readonly broadcastV?: boolean;
  /** 最後の bmm の入力順を入れ替える。 */
  readonly swapPv?: boolean;
  /** 中間 S を graph output にする。 */
  readonly scoresOutput?: boolean;
  /** softmax 出力にもう 1 本 consumer を足す。 */
  readonly extraConsumer?: boolean;
  /** q と kᵀ を同じ入力にする（bind 面が重複する形 — 正方形のときだけ作れる）。 */
  readonly sameQk?: boolean;
};

const attentionGraph = (
  shape: AttentionShape = ATTENTION,
  options: AttentionOptions = {},
): GraphJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = options.interpose ? "scores3_alias" : "scores3";
  const maskShape = options.rowMask ? [1, 1, queries, keys] : [1, 1, 1, keys];
  // 非恒等 expand を作るには、元を 1 軸だけ 1 にして複製させる。
  const vSource = options.broadcastV ? [1, 1, keys, headDim] : [1, heads, keys, headDim];
  const values: GraphJson["values"] = {
    scores3: { dtype: "f32", shape: scores3 },
    scores4: { dtype: "f32", shape: scores4 },
    masked: { dtype: "f32", shape: scores4 },
    probs: { dtype: "f32", shape: scores4 },
    probsExpanded: { dtype: "f32", shape: scores4 },
    probs3: { dtype: "f32", shape: scores3 },
    vExpanded: { dtype: "f32", shape: [1, heads, keys, headDim] },
    v3: { dtype: "f32", shape: [heads, keys, headDim] },
    y: { dtype: "f32", shape: [heads, queries, headDim] },
  };
  if (options.interpose) values.scores3_alias = { dtype: "f32", shape: scores3 };
  if (options.extraConsumer) values.probsCopy = { dtype: "f32", shape: scores4 };
  const nodes: GraphJson["nodes"] = [
    { op: "bmm", ins: ["q", options.sameQk ? "q" : "kt"], outs: [qkOut], attrs: {} },
    ...(options.interpose ? [{ op: "reshape", ins: [qkOut], outs: ["scores3"], attrs: {} }] : []),
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    { op: "add", ins: ["scores4", "mask"], outs: ["masked"], attrs: {} },
    {
      op: options.plainSoftmax ? "softmax" : "safe_softmax",
      ins: ["masked"],
      outs: ["probs"],
      attrs: { dim: 3 },
    },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    {
      op: "bmm",
      ins: options.swapPv ? ["v3", "probs3"] : ["probs3", "v3"],
      outs: ["y"],
      attrs: {},
    },
    ...(options.extraConsumer
      ? [{ op: "neg", ins: ["probs"], outs: ["probsCopy"], attrs: {} }]
      : []),
  ];
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      ...(options.sameQk
        ? []
        : [{ name: "kt", dtype: "f32" as const, shape: [heads, headDim, keys] }]),
      { name: "mask", dtype: "f32", shape: maskShape },
      { name: "v", dtype: "f32", shape: vSource },
    ],
    outputs: options.scoresOutput ? ["y", "scores4"] : ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const attentionInputs = (
  shape: AttentionShape = ATTENTION,
  options: AttentionOptions = {},
): Readonly<Record<string, readonly number[]>> => {
  const { heads, queries, keys, headDim } = shape;
  return {
    q: [heads, queries, headDim],
    ...(options.sameQk ? {} : { kt: [heads, headDim, keys] }),
    mask: options.rowMask ? [1, 1, queries, keys] : [1, 1, 1, keys],
    v: options.broadcastV ? [1, 1, keys, headDim] : [1, heads, keys, headDim],
  };
};

Deno.test("分解 attention の 9 ノード窓を 1 ステップへ畳み、外部入力を延べ 4 回だけ消費する", () => {
  const plan = fuse(attentionGraph(), attentionInputs());
  assertEquals(outline(plan.steps), ["fused:rowBlockAttention"]);
  const step = fusedAt(plan, 0);
  const operands = ["q", "kt", "mask", "v"];
  assertEquals(step.ins, operands, "外部入力の延べ列（元 9 ノードの外部消費と厳密一致）");
  assertEquals(step.binds, operands, "bind 面（重複無し）");
  assertEquals(step.nodeCount, 9);
  assertEquals(step.outputName, "y");
  assertEquals(step.outputShape, [ATTENTION.heads, ATTENTION.queries, ATTENTION.headDim]);
  assertEquals(plan.counts.rowBlockAttention, 1);
  // 128MiB 上限に対して S は小さいので 1 枚 = 素の 4 dispatch 列。
  assertEquals(step.dispatches.length, 4, "1 枚は 4 dispatch");
  assertEquals(step.temps.length, 3, "S / mask 済み S / P の 3 本");
  assertEquals(
    step.dispatches.map((dispatch) => dispatch.key),
    [
      bmmKey(false, ATTENTION.queries),
      elementwiseKey({ op: "add", rank: 4, dtype: "f32" }),
      SAFE_SOFTMAX_KEY,
      bmmKey(false, ATTENTION.queries),
    ],
    "1 枚のキーは素の bmm（行窓変種を使わない）",
  );
});

Deno.test("行ブロックは上限に収まる最小枚数で等分され、一時は 1 枚ぶんまで縮む", () => {
  const shape: AttentionShape = { heads: 2, queries: 100, keys: 64, headDim: 8 };
  // 1 行 = 2·64·4 = 512B。上限を 1 行 30 枚ぶんに絞ると ceil(100/30) = 4 枚（25 行 × 4）。
  const ir = parse(attentionGraph(shape));
  const plan = planFusions(planGraph(ir, bindSymbols(ir, attentionInputs(shape))).nodes, {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: { maxStorageBufferBindingSize: 512 * 30, maxComputeWorkgroupsPerDimension: 65535 },
  });
  const step = fusedAt(plan, 0);
  assertEquals(step.dispatches.length, 16, "4 枚 × 4 dispatch");
  assertEquals(step.temps.length, 12, "4 枚 × 3 本");
  assertEquals(
    [...new Set(step.temps.map((temp) => temp.byteLength))],
    [25 * 512],
    "一時はブロック 1 枚ぶん（全 M ではない）",
  );
  assertEquals(
    step.dispatches.map((dispatch) => dispatch.key).slice(0, 4),
    [
      bmmKey(true, 25, "a"),
      elementwiseKey({ op: "add", rank: 4, dtype: "f32" }),
      SAFE_SOFTMAX_KEY,
      bmmKey(true, 25, "c"),
    ],
    "2 枚以上では bmm が行窓変種へ振り替わる",
  );
});

Deno.test("行ブロック分割は上限に収まる最小枚数を等分で返し、収まらない形は fail loudly", () => {
  // ちょうど上限（10 行 × 100B = 1000B）は 1 枚。
  assertEquals(planRowBlocks(10, 100, 1000), [{ offset: 0, rows: 10 }]);
  // 1 要素ぶん超えたら 2 枚（5 / 5）。
  assertEquals(planRowBlocks(10, 100, 999), [{ offset: 0, rows: 5 }, { offset: 5, rows: 5 }]);
  // 端数は先頭から 1 行ずつ配る（3 枚 = 4 / 3 / 3）。
  assertEquals(planRowBlocks(10, 100, 400), [
    { offset: 0, rows: 4 },
    { offset: 4, rows: 3 },
    { offset: 7, rows: 3 },
  ]);
  // 強制枚数（テスト専用の内部面）も等分。
  assertEquals(planRowBlocks(10, 100, 1000, 3), [
    { offset: 0, rows: 4 },
    { offset: 4, rows: 3 },
    { offset: 7, rows: 3 },
  ]);
  // 1 行でも収まらない形は fail loudly（黙って素の列へ落とすと確保が失敗するだけ）。
  assertThrows(
    () => planRowBlocks(10, 1001, 1000),
    ExecutionError,
    "クエリ 1 行ぶんのスコア",
  );
  // 強制枚数が上限に収まらないのも fail loudly（緩める向きには使えない）。
  assertThrows(() => planRowBlocks(10, 100, 400, 2), ExecutionError, "2 枚では 1 枚 500B");
  // 行数より多い枚数・0 枚は取れない。
  assertThrows(() => planRowBlocks(10, 100, 1000, 11), ExecutionError);
  assertThrows(() => planRowBlocks(10, 100, 1000, 0), ExecutionError);
});

Deno.test("行ブロック attention の反例（隣接 / mask 形 / op 違い / 非恒等 expand / 入力順 / 内部 output / 別 consumer / bind 重複）は掴まない", () => {
  // 正方形（M = N = D）だけが「PV の入力順を入れ替える」「q と kᵀ を同じ入力にする」を
  // 契約違反にせず作れる。**その土台自体が掴めること**を先に見ないと、下の 2 件が
  // 「元から掴めない形」を見ているだけの恒真になる。
  const square: AttentionShape = { heads: 2, queries: 4, keys: 4, headDim: 4 };
  assertEquals(
    fuse(attentionGraph(square), attentionInputs(square)).counts.rowBlockAttention,
    1,
    "正方形の土台が掴めていない（下の 2 反例が恒真になる）",
  );
  const cases:
    readonly (readonly [string, GraphJson, Readonly<Record<string, readonly number[]>>])[] = [
      ["別名を 1 本挟む", attentionGraph(ATTENTION, { interpose: true }), attentionInputs()],
      [
        "mask が行ごと [1,1,M,N]",
        attentionGraph(ATTENTION, { rowMask: true }),
        attentionInputs(ATTENTION, { rowMask: true }),
      ],
      ["素の softmax", attentionGraph(ATTENTION, { plainSoftmax: true }), attentionInputs()],
      [
        "V 側 expand が複製軸を持つ",
        attentionGraph(ATTENTION, { broadcastV: true }),
        attentionInputs(ATTENTION, { broadcastV: true }),
      ],
      ["PV の入力順が逆", attentionGraph(square, { swapPv: true }), attentionInputs(square)],
      [
        "中間 S が graph output",
        attentionGraph(ATTENTION, { scoresOutput: true }),
        attentionInputs(),
      ],
      ["P に別 consumer", attentionGraph(ATTENTION, { extraConsumer: true }), attentionInputs()],
      [
        "q と kᵀ が同じ入力（bind 重複）",
        attentionGraph(square, { sameQk: true }),
        attentionInputs(square, { sameQk: true }),
      ],
    ];
  for (const [label, graph, inputs] of cases) {
    const plan = fuse(graph, inputs);
    assertEquals(
      plan.counts.rowBlockAttention,
      0,
      `${label}: 掴んでいる（${outline(plan.steps).join(",")}）`,
    );
  }
});
