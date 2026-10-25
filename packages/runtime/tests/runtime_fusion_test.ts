/**
 * 融合パス（src/runtime/fusion.ts）の判定だけを GPU 非依存で固定する。
 *
 * 融合の受理集合は「掴めた形は速く、掴めない形は素の列で必ず正しい」という 2 段構えなので、
 * **反例側の網羅がそのまま安全性の根拠**になる。実 GPU テスト（gpu_*_fusion_test.ts）は
 * 値のビット parity を見る役で、こちらは反例を GPU 抜きで細かく積む役。
 */

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
  adalnNormParams,
} from "../src/kernels/adaln-norm.ts";
import { bmmKey } from "../src/kernels/bmm.ts";
import { GATED_RESIDUAL_KEY } from "../src/kernels/gated-residual.ts";
import { ROPE_BSHD_KEY, ROPE_KEY, ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { QUANTIZE_ROWS_PROLOGUE_MAX_DIM } from "../src/kernels/quantize-rows.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { SAFE_SOFTMAX_KEY } from "../src/kernels/softmax.ts";
import { UPSAMPLE_2X_KEY } from "../src/kernels/upsample2x.ts";
import {
  type ExecStep,
  FUSION_RULES,
  type FusionPlan,
  LINEAR_PROLOGUE_OPS,
  planFusions,
  planRowBlocks,
} from "../src/runtime/fusion.ts";
import { bindSymbols, countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

const parse = mergeGraph;

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
  graph: DeclarationJson,
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
  steps.map((step) => (step.kind === "fused" ? `fused:${step.rule}` : step.plan.node.op));

const fusedAt = (plan: FusionPlan, index: number) => {
  const step = plan.steps[index];
  if (step.kind !== "fused") throw new Error(`steps[${index}] が融合ステップでない`);
  return step;
};

test("ルール表の先頭 op は互いに素で、適用順が結果に効かない", () => {
  assertEquals(
    FUSION_RULES.map((rule) => rule.name),
    [
      "linearStaticQuantize",
      "rmsNormAdd",
      "silu",
      "upsample2x",
      "rope",
      "adaln",
      "gatedResidual",
      "rowBlockAttention",
    ],
  );
  const seen = new Set<string>();
  for (const rule of FUSION_RULES) {
    for (const head of rule.heads) {
      assertEquals(seen.has(head), false, `先頭 op '${head}' が複数ルールで重なっている`);
      seen.add(head);
    }
  }
  assertEquals([...seen].sort(), [
    "bmm",
    "layer_norm",
    "linear",
    "mul",
    "reshape",
    "rms_norm",
    "sigmoid",
    "slice",
    "where",
  ]);
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

const siluGraph = (options: SiluOptions = {}): DeclarationJson => {
  const shape = [4, 8];
  const values: DeclarationJson["values"] = {
    s: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...shape] }];
  const nodes: DeclarationJson["nodes"] = [{ op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} }];
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
    version: 2,
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

test("SiLU は sigmoid→mul の両順を掴み、x を延べ 2 回消費するステップにする", () => {
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

test("SiLU の反例（別名 / 内部 output / 別 consumer / 別 gate / broadcast）は素の列へ落ちる", () => {
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
test("state を触るノードを含む窓はどのルールも掴まない（窓の仕組み側の不変条件）", () => {
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
      index === tainted ? { ...node, node: { ...node.node, states: { k: "kv" } } } : node,
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

const upsampleGraph = (options: UpsampleOptions = {}): DeclarationJson => {
  const dtype = options.dtype ?? "f32";
  const scale = options.scaleWidth ?? 2;
  const outW = 7 * scale;
  const values: DeclarationJson["values"] = {
    a: { dtype, shape: [30, 7, 1] },
    wide: { dtype, shape: [30, 7, scale] },
    b: { dtype, shape: [6, 5, outW] },
    c: { dtype, shape: [6, 5, 1, outW] },
    tall: { dtype, shape: [6, 5, 2, outW] },
    y: { dtype, shape: [2, 3, 10, outW] },
  };
  const nodes: DeclarationJson["nodes"] = [{ op: "reshape", ins: ["x"], outs: ["a"], attrs: {} }];
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
    version: 2,
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

test("VAE nearest x2 は 6 ノード鎖を掴み、外部入力 1 本だけを消費する", () => {
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

test("upsample x2 の反例（near-shape / 別名 / 内部 output / 別 consumer / dtype）は掴まない", () => {
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
  readonly layout?: "bhsd" | "bshd";
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

const ropeGraph = (options: RopeOptions = {}): DeclarationJson => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const split = headDim / (options.nearSplit ? 4 : 2);
  const full = options.layout === "bshd" ? [batch, 5, heads, headDim] : [batch, heads, 5, headDim];
  const low = options.layout === "bshd" ? [batch, 5, heads, split] : [batch, heads, 5, split];
  const high =
    options.layout === "bshd"
      ? [batch, 5, heads, headDim - split]
      : [batch, heads, 5, headDim - split];
  const table = options.nearTable
    ? full
    : options.layout === "bshd"
      ? [1, 5, 1, headDim]
      : [1, 1, 5, headDim];
  const values: DeclarationJson["values"] = {
    first: { dtype: "f32", shape: low },
    second: { dtype: "f32", shape: high },
    negative: { dtype: "f32", shape: high },
    rotated: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const direct: DeclarationJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: DeclarationJson["nodes"] = [];
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
    values.sin_table = {
      dtype: "f32",
      shape: options.layout === "bshd" ? [1, 8, 1, headDim] : [1, 1, 8, headDim],
    };
    values.sin = {
      dtype: "f32",
      shape: options.layout === "bshd" ? [1, "T", 1, headDim] : [1, 1, "T", headDim],
    };
    nodes.push({
      op: "sym_prefix_slice",
      ins: ["sin_table"],
      outs: ["sin"],
      attrs: {
        sym: "T",
        slices: [{ dim: options.layout === "bshd" ? 1 : 2, coeff: 1, offset: 0 }],
      },
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
    version: 2,
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
    initializers: options.prefixSlicedSin ? { sin_table: {} } : {},
    values,
    nodes,
  };
};

const ropeInputs = (options: RopeOptions = {}): Readonly<Record<string, readonly number[]>> => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const full = options.layout === "bshd" ? [batch, 5, heads, headDim] : [batch, heads, 5, headDim];
  const table = options.nearTable
    ? full
    : options.layout === "bshd"
      ? [1, 5, 1, headDim]
      : [1, 1, 5, headDim];
  return options.prefixSlicedSin
    ? { x: full, cos: table, bind: [5] }
    : { x: full, cos: table, sin: table };
};

test("half-split RoPE は 2 順序を掴み、x を延べ 3 回・cos / sin を 1 回ずつ消費する", () => {
  for (const [order, expectedIns] of [
    ["slice-first", ["x", "x", "x", "cos", "sin"]],
    ["direct-first", ["x", "cos", "x", "x", "sin"]],
  ] as const) {
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
test("half-split RoPE は head 幅を分割位置から導き、幅ごとに params だけが変わる", () => {
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
test("RoPE は cat 直後の sym_prefix_slice を窓内 passthrough として跨ぐ", () => {
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

test("RoPE の反例（別名 / 内部 output / 別 consumer / 分割位置 / batch / add 順 / table 形 / 隙間の別 op）は掴まない", () => {
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

const expandGraph = (outShape: readonly number[]): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
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

test("恒等 expand だけが 0 dispatch の別名になり、複製軸があれば実体化コピーへ戻る", () => {
  const identity = fuse(expandGraph([3, 1]), { x: [3, 1] });
  assertEquals(
    identity.steps.map((step) => step.kind === "node" && step.aliasesInput),
    [
      false, // neg
      true, // 恒等 expand
      true, // reshape は無条件に別名
    ],
  );
  assertEquals(identity.counts.identityExpand, 1);

  const replicated = fuse(expandGraph([3, 4]), { x: [3, 1] });
  assertEquals(
    replicated.steps.map((step) => step.kind === "node" && step.aliasesInput),
    [
      false,
      false, // 1 軸でも複製があれば別名化しない
      true,
    ],
  );
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
const adalnGraph = (options: AdalnOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const gaps = options.gaps ?? 3;
  const row = [1, ROWS, DIM];
  const modulation = options.perRowScale ? row : [1, 1, DIM];
  const biasName = options.sharedAffine ? "ln_weight" : "ln_bias";
  const values: DeclarationJson["values"] = {
    t: { dtype, shape: [...row] },
    s: { dtype, shape: [...modulation] },
    p: { dtype, shape: [...row] },
    y: { dtype, shape: [...row] },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype, shape: [...row] },
    { name: "ln_weight", dtype, shape: [DIM] },
    ...(options.sharedAffine ? [] : [{ name: "ln_bias", dtype, shape: [DIM] }]),
    { name: "one", dtype, shape: options.wideOne ? [DIM] : [1] },
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", biasName],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
  ];
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
    version: 2,
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
const adalnInputs = (graph: DeclarationJson): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(graph.inputs.map((spec) => [spec.name, spec.shape as readonly number[]]));

const fuseAdaln = (options: AdalnOptions = {}): FusionPlan => {
  const graph = adalnGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("adaLN は窓 7 / 窓 6 を掴み、reshape は融合ステップの前に素のまま並ぶ", () => {
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

test("adaLN の窓内 passthrough は他ルールの受理位置を潰さない", () => {
  const graph = adalnGraph();
  const ir = parse(graph);
  const plan = planGraph(ir, bindSymbols(ir, adalnInputs(graph)));
  const context = {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    packedValues: new Map<string, number>(),
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
test("adaLN の反例（gap の本数 / gap が norm を消費 / 内部 output / 別 consumer / 入力順 / 定数形 / near-shape / bind 重複）は掴まない", () => {
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

type AdalnTokenOptions = {
  /** The shift `where` reads a different mask from the scale `where`. */
  readonly separateMasks?: boolean;
  /** The mask is `[1,1,1]` (one flag for every row — not one per token). */
  readonly broadcastMask?: boolean;
  /** The mask is `[1,rows,dim]` (per element — not one per token). */
  readonly fullMask?: boolean;
  /** The condition-side scale is `[1,rows,dim]` (per row — not a broadcast vector). */
  readonly perRowVector?: boolean;
  /** `mul(s, t)` instead of `mul(t, s)`. */
  readonly swappedMul?: boolean;
  /** `add(one, sel_scale)` instead of `add(sel_scale, one)`. */
  readonly swappedModulate?: boolean;
  /** `add(sel_shift, p)` instead of `add(p, sel_shift)`. */
  readonly swappedOffset?: boolean;
  /** `one` as `[dim]` (the kernel reads `one[0]` only). */
  readonly wideOne?: boolean;
  /** The layer_norm output is a graph output. */
  readonly normOutput?: boolean;
  /** The selected scale gets a second consumer. */
  readonly selectExtraConsumer?: boolean;
  /** `where(mask, scale_c, scale_c)` (the same value in two bind slots). */
  readonly sharedVector?: boolean;
  /** A zero-dispatch reshape of the selected scale between the `where` and its `add`. */
  readonly gap?: boolean;
};

/**
 * Wan2.2's per-token adaLN: the modulation vectors are picked per token by
 * `where(mask, condition, generation)` and the window is the 6 contiguous nodes.
 */
const adalnTokenGraph = (options: AdalnTokenOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const mask = options.broadcastMask ? [1, 1, 1] : options.fullMask ? row : [1, ROWS, 1];
  const selected = options.broadcastMask ? vector : row;
  const values: DeclarationJson["values"] = {
    t: { dtype, shape: row },
    sel_scale: { dtype, shape: selected },
    s: { dtype, shape: selected },
    p: { dtype, shape: row },
    sel_shift: { dtype, shape: selected },
    y: { dtype, shape: row },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype, shape: row },
    { name: "ln_weight", dtype, shape: [DIM] },
    { name: "ln_bias", dtype, shape: [DIM] },
    { name: "mask", dtype: "bool", shape: mask },
    ...(options.separateMasks ? [{ name: "mask2", dtype: "bool", shape: mask }] : []),
    { name: "one", dtype, shape: options.wideOne ? [DIM] : [1] },
    { name: "scale_c", dtype, shape: options.perRowVector ? row : vector },
    { name: "scale_g", dtype, shape: vector },
    { name: "shift_c", dtype, shape: vector },
    { name: "shift_g", dtype, shape: vector },
  ];
  const extraOutputs: string[] = [];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
    {
      op: "where",
      ins: ["mask", "scale_c", options.sharedVector ? "scale_c" : "scale_g"],
      outs: ["sel_scale"],
      attrs: {},
    },
  ];
  let selectedScale = "sel_scale";
  if (options.gap) {
    values.sel_scale_alias = { dtype, shape: selected };
    nodes.push({ op: "reshape", ins: ["sel_scale"], outs: ["sel_scale_alias"], attrs: {} });
    selectedScale = "sel_scale_alias";
  }
  nodes.push(
    {
      op: "add",
      ins: options.swappedModulate ? ["one", selectedScale] : [selectedScale, "one"],
      outs: ["s"],
      attrs: {},
    },
    { op: "mul", ins: options.swappedMul ? ["s", "t"] : ["t", "s"], outs: ["p"], attrs: {} },
    {
      op: "where",
      ins: [options.separateMasks ? "mask2" : "mask", "shift_c", "shift_g"],
      outs: ["sel_shift"],
      attrs: {},
    },
    {
      op: "add",
      ins: options.swappedOffset ? ["sel_shift", "p"] : ["p", "sel_shift"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.selectExtraConsumer) {
    values.sel_copy = { dtype, shape: selected };
    nodes.push({ op: "neg", ins: ["sel_scale"], outs: ["sel_copy"], attrs: {} });
    extraOutputs.push("sel_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.normOutput ? ["t"] : []), "y", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

const fuseAdalnToken = (options: AdalnTokenOptions = {}): FusionPlan => {
  const graph = adalnTokenGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("per-token adaLN folds the 6 contiguous nodes into a pack dispatch and the shared staged body", () => {
  const plan = fuseAdalnToken();
  assertEquals(outline(plan.steps), ["fused:adaln"]);
  const step = fusedAt(plan, 0);
  // MUST: the total-use list follows the actual node order (the mask is used by both wheres).
  assertEquals(step.ins, [
    "x",
    "ln_weight",
    "ln_bias",
    "mask",
    "scale_c",
    "scale_g",
    "one",
    "mask",
    "shift_c",
    "shift_g",
  ]);
  assertEquals(step.binds, [
    "x",
    "ln_weight",
    "ln_bias",
    "mask",
    "scale_c",
    "scale_g",
    "one",
    "shift_c",
    "shift_g",
  ]);
  assertEquals(step.nodeCount, 6);
  assertEquals(step.outputName, "y");
  // The packed table `[scale_c, scale_g, shift_c, shift_g]` lives across the two dispatches only.
  assertEquals(step.temps, [{ byteLength: 4 * DIM * 4, allocBefore: 0, releaseAfter: 1 }]);
  const [pack, body] = step.dispatches;
  assertEquals(step.dispatches.length, 2);
  assertEquals(pack.key, ADALN_MODULATION_PACK_KEY);
  assertEquals([...pack.params], [DIM, 0, 0, 0]);
  assertEquals(pack.operands, [
    { kind: "bind", index: 4 },
    { kind: "bind", index: 5 },
    { kind: "bind", index: 7 },
    { kind: "bind", index: 8 },
    { kind: "temp", id: 0 },
  ]);
  assertEquals(pack.workgroups, { kind: "gridStride", items: DIM, size: 256 });
  assertEquals(body.key, ADALN_NORM_TOKEN_KEY);
  assertEquals([...body.params], [...adalnNormParams(ROWS, DIM, 1e-6)]);
  // x, weight, bias, mask, table, one, out = 7 storage bindings (the WebGPU default is 8).
  assertEquals(body.operands, [
    { kind: "bind", index: 0 },
    { kind: "bind", index: 1 },
    { kind: "bind", index: 2 },
    { kind: "bind", index: 3 },
    { kind: "temp", id: 0 },
    { kind: "bind", index: 6 },
    { kind: "output" },
  ]);
  assertEquals(body.workgroups, { kind: "gridStride", items: ROWS, size: 1 });
  assertEquals(plan.counts.adaln, 1);
  assertEquals(plan.counts.gatedResidual, 0);
});

test("per-token adaLN counterexamples (two masks / mask and vector shapes / input orders / one's shape / internal output / extra consumer / bind duplicate / gap) are not grabbed", () => {
  const cases: readonly (readonly [string, AdalnTokenOptions])[] = [
    ["two different masks", { separateMasks: true }],
    ["mask [1,1,1]", { broadcastMask: true }],
    ["mask per element", { fullMask: true }],
    ["per-row scale vector", { perRowVector: true }],
    ["mul(s, t)", { swappedMul: true }],
    ["add(one, sel_scale)", { swappedModulate: true }],
    ["add(sel_shift, p)", { swappedOffset: true }],
    ["one as [dim]", { wideOne: true }],
    ["layer_norm output is a graph output", { normOutput: true }],
    ["selected scale has another consumer", { selectExtraConsumer: true }],
    ["where(mask, v, v)", { sharedVector: true }],
    ["reshape between where and add", { gap: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseAdalnToken(options);
    assertEquals(plan.counts.adaln, 0, `${label}: adaln counter`);
    // Neither `where` of the window may be taken by the gated-residual rule instead.
    assertEquals(plan.counts.gatedResidual, 0, `${label}: gatedResidual counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("カウンタは融合が並んだグラフでルール別に積み上がる", () => {
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
    gatedResidual: 0,
    rmsNormAdd: 0,
    linearStaticQuantize: 0,
    rowBlockAttention: 0,
    identityExpand: 0,
    packedStaticQuantize: 0,
    linearPrologue: 0,
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
): DeclarationJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = options.interpose ? "scores3_alias" : "scores3";
  const maskShape = options.rowMask ? [1, 1, queries, keys] : [1, 1, 1, keys];
  // 非恒等 expand を作るには、元を 1 軸だけ 1 にして複製させる。
  const vSource = options.broadcastV ? [1, 1, keys, headDim] : [1, heads, keys, headDim];
  const values: DeclarationJson["values"] = {
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
  const nodes: DeclarationJson["nodes"] = [
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
    version: 2,
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

test("分解 attention の 9 ノード窓を 1 ステップへ畳み、外部入力を延べ 4 回だけ消費する", () => {
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

test("行ブロックは上限に収まる最小枚数で等分され、一時は 1 枚ぶんまで縮む", () => {
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

test("予算 attentionRowBlockMaxBytes は束縛上限との小さいほうで分解 attention を割る（ADR 0124）", () => {
  const shape: AttentionShape = { heads: 2, queries: 100, keys: 64, headDim: 8 };
  const ir = parse(attentionGraph(shape));
  const nodes = planGraph(ir, bindSymbols(ir, attentionInputs(shape))).nodes;
  const planWith = (budget: number | undefined) =>
    planFusions(nodes, {
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      // 束縛上限は 128 MiB（予算なしでは 1 枚）。予算だけが枚数を動かす。
      limits: TEST_LIMITS,
      ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
    });
  // 予算なしは素の 4 dispatch 列（既定は予算の欄を足す前と同一）。
  assertEquals(fusedAt(planWith(undefined), 0).dispatches.length, 4);
  // 1 行 = 2·64·4 = 512B。予算を 1 行 30 枚ぶんにすると、束縛上限で絞ったときと同じ 4 枚（25 行 × 4）。
  const budgeted = fusedAt(planWith(512 * 30), 0);
  assertEquals(budgeted.dispatches.length, 16, "4 枚 × 4 dispatch");
  assertEquals([...new Set(budgeted.temps.map((temp) => temp.byteLength))], [25 * 512]);
  assertEquals(budgeted.dispatches[0].key, bmmKey(true, 25, "a"), "行窓変種へ振り替わる");
  // 束縛上限以上の予算は効かない（予算で上限を緩める向きには使えない）。
  assertEquals(
    fusedAt(planWith(TEST_LIMITS.maxStorageBufferBindingSize * 2), 0).dispatches.length,
    4,
  );
  // 1 行ぶんに満たない予算は、掴めた窓を素の列へ落とさず予算を名指して fail loudly。
  assertThrows(() => planWith(511), ExecutionError, "set the budget to at least one row");
});

test("行ブロック分割は上限に収まる最小枚数を等分で返し、収まらない形は fail loudly", () => {
  // ちょうど上限（10 行 × 100B = 1000B）は 1 枚。
  assertEquals(planRowBlocks(10, 100, 1000), [{ offset: 0, rows: 10 }]);
  // 1 要素ぶん超えたら 2 枚（5 / 5）。
  assertEquals(planRowBlocks(10, 100, 999), [
    { offset: 0, rows: 5 },
    { offset: 5, rows: 5 },
  ]);
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
    "the temporary for one query row",
  );
  // 強制枚数が上限に収まらないのも fail loudly（緩める向きには使えない）。
  assertThrows(
    () => planRowBlocks(10, 100, 400, 2),
    ExecutionError,
    "with 2 blocks, one block of 500B",
  );
  // 行数より多い枚数・0 枚は取れない。
  assertThrows(() => planRowBlocks(10, 100, 1000, 11), ExecutionError);
  assertThrows(() => planRowBlocks(10, 100, 1000, 0), ExecutionError);
});

test("行ブロック attention の反例（隣接 / mask 形 / op 違い / 非恒等 expand / 入力順 / 内部 output / 別 consumer / bind 重複）は掴まない", () => {
  // 正方形（M = N = D）だけが「PV の入力順を入れ替える」「q と kᵀ を同じ入力にする」を
  // 契約違反にせず作れる。**その土台自体が掴めること**を先に見ないと、下の 2 件が
  // 「元から掴めない形」を見ているだけの恒真になる。
  const square: AttentionShape = { heads: 2, queries: 4, keys: 4, headDim: 4 };
  assertEquals(
    fuse(attentionGraph(square), attentionInputs(square)).counts.rowBlockAttention,
    1,
    "正方形の土台が掴めていない（下の 2 反例が恒真になる）",
  );
  const cases: readonly (readonly [
    string,
    DeclarationJson,
    Readonly<Record<string, readonly number[]>>,
  ])[] = [
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

/**
 * BiRefNet の分解 attention（窓 attention）の最小形。実測グラフ 96 本のうち **0 本**が
 * `rowBlockAttention` に掴まれない形で、外れる理由が 3 つ同時に立っている:
 *
 * ① mask が `[1,H,S,S]`（head ごとの相対位置バイアス）で、ルールが要求する `[1,1,1,N]` ではない
 * ② `safe_softmax` ではなく素の `softmax`
 * ③ 窓の途中に `embedding`（相対位置表の引き）と、その `reshape` が挟まる
 *
 * ACTIVE_DESIGN が「**row-block だけは外れ方が性能でなく資源**」と書いているのはこの形のこと
 * で、掴めないと `bmm` の出力 `[nW·H,S,S]` が**ノード出力スロット**として実体化する
 * （原理的に行ブロックへ分割できない）。matcher を広げた／狭めたときにどちらの向きの退行も
 * 沈黙するので、掴まないこと自体と資源側の帰結を観測点として置く。
 */
const birefnetAttentionGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["bmm", "reshape", "embedding", "add", "softmax", "expand"] },
  symbols: [],
  inputs: [
    { name: "q", dtype: "f32", shape: [6, 4, 5] },
    { name: "kt", dtype: "f32", shape: [6, 5, 4] },
    { name: "idx", dtype: "i32", shape: [48] },
    { name: "v", dtype: "f32", shape: [2, 3, 4, 5] },
  ],
  outputs: ["y"],
  initializers: { table: {} },
  values: {
    table: { dtype: "f32", shape: [8, 1] },
    scores3: { dtype: "f32", shape: [6, 4, 4] },
    scores4: { dtype: "f32", shape: [2, 3, 4, 4] },
    biasFlat: { dtype: "f32", shape: [48, 1] },
    bias4: { dtype: "f32", shape: [1, 3, 4, 4] },
    masked: { dtype: "f32", shape: [2, 3, 4, 4] },
    probs: { dtype: "f32", shape: [2, 3, 4, 4] },
    probsExpanded: { dtype: "f32", shape: [2, 3, 4, 4] },
    probs3: { dtype: "f32", shape: [6, 4, 4] },
    vExpanded: { dtype: "f32", shape: [2, 3, 4, 5] },
    v3: { dtype: "f32", shape: [6, 4, 5] },
    y: { dtype: "f32", shape: [6, 4, 5] },
  },
  nodes: [
    { op: "bmm", ins: ["q", "kt"], outs: ["scores3"], attrs: {} },
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    // ③ 窓の途中の相対位置表の引き（2 ノード）。
    { op: "embedding", ins: ["table", "idx"], outs: ["biasFlat"], attrs: { padding_idx: -1 } },
    { op: "reshape", ins: ["biasFlat"], outs: ["bias4"], attrs: {} },
    // ① mask は head ごと（`[1,H,S,S]`）。
    { op: "add", ins: ["scores4", "bias4"], outs: ["masked"], attrs: {} },
    // ② 素の softmax。
    { op: "softmax", ins: ["masked"], outs: ["probs"], attrs: { dim: 3 } },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["probs3", "v3"], outs: ["y"], attrs: {} },
  ],
});

const birefnetInputs: Readonly<Record<string, readonly number[]>> = {
  q: [6, 4, 5],
  kt: [6, 5, 4],
  idx: [48],
  v: [2, 3, 4, 5],
};

/** 実体化されるノード出力（別名は確保を出さないので除く）の「名前 → バイト数」。 */
const allocatedOutputs = (plan: FusionPlan): ReadonlyMap<string, number> => {
  const bytes = new Map<string, number>();
  for (const step of plan.steps) {
    if (step.kind !== "node" || step.aliasesInput) continue;
    for (const output of step.plan.outputs) {
      bytes.set(output.name, output.shape.reduce((size, dim) => size * dim, 1) * 4);
    }
  }
  return bytes;
};

test("birefnet 形の分解 attention は掴めず、S がノード出力スロットとして実体化する", () => {
  const plan = fuse(birefnetAttentionGraph(), birefnetInputs);
  assertEquals(plan.counts.rowBlockAttention, 0, outline(plan.steps).join(","));

  // 資源側の帰結: S = nW·H·S·S·4 = 2·3·4·4·4 が**ノード出力**として並ぶ。融合ステップの
  // 内側の一時なら行ブロックで 1 枚ぶんまで縮むが、ノード出力スロットは分割できない。
  assertEquals(allocatedOutputs(plan).get("scores3"), 2 * 3 * 4 * 4 * 4);

  // 対照: mask が [1,1,1,N]・safe_softmax・窓が連続 9 ノードの形は掴み、同じ S は
  // ステップ内一時（= 行ブロックで縮む側）になる。
  const fused = fuse(attentionGraph(), attentionInputs());
  assertEquals(fused.counts.rowBlockAttention, 1);
  const scoreBytes = ATTENTION.heads * ATTENTION.queries * ATTENTION.keys * 4;
  assertEquals(
    fusedAt(fused, 0).temps.filter((temp) => temp.byteLength === scoreBytes).length,
    3,
    "S / mask 済み S / P がステップ内一時になっていない",
  );
  assertEquals(allocatedOutputs(fused).has("scores3"), false, "S がノード出力として残っている");
});

test("BSHD RoPEは両発行順でhead数をuniformへ渡し、私有中間だけを畳む", () => {
  for (const order of ["slice-first", "direct-first"] as const) {
    const options: RopeOptions = { layout: "bshd", heads: 4, headDim: 256, order };
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(outline(plan.steps), ["fused:rope"]);
    const step = fusedAt(plan, 0);
    assertEquals(step.binds, ["x", "cos", "sin"]);
    assertEquals(step.nodeCount, 7);
    assertEquals(step.dispatches[0].key, ROPE_BSHD_KEY);
    assertEquals([...step.dispatches[0].params], [4 * 5 * 256, 4, 256, 128]);
    for (const negative of [
      { interpose: true },
      { internalOutput: true },
      { extraConsumer: true },
      { nearSplit: true },
      { batch2: true },
      { swappedAdd: true },
      { nearTable: true },
      { gapReshape: true },
    ]) {
      const rejected = { ...options, ...negative };
      assertEquals(
        fuse(ropeGraph(rejected), ropeInputs(rejected)).counts.rope,
        0,
        JSON.stringify(rejected),
      );
    }
  }
});

test("BSHD RoPEはcos/sinの軸が別なら掴まず、共通の位置表のsliceは保持する", () => {
  const options: RopeOptions = { layout: "bshd", heads: 4, headDim: 128, order: "direct-first" };
  const graph = ropeGraph(options);
  graph.inputs[1].shape = [1, 1, 4, 128];
  assertEquals(fuse(graph, { ...ropeInputs(options), cos: [1, 1, 4, 128] }).counts.rope, 0);
  const sliced = { ...options, prefixSlicedSin: true };
  const plan = fuse(ropeGraph(sliced), ropeInputs(sliced));
  assertEquals(outline(plan.steps), ["sym_prefix_slice", "fused:rope"]);
  assertEquals(fusedAt(plan, 1).dispatches[0].key, ROPE_BSHD_KEY);
});

// ------------------------------------------------------------ RoPE (pair form)

type RopePairOptions = {
  readonly heads?: number;
  readonly headDim?: number;
  /** `pairs` is a graph input of its own instead of the reshape of `x` (the kernel binds both). */
  readonly separatePairs?: boolean;
  /** A zero-dispatch reshape between `cat` and the reshape back (breaks the exact wiring only). */
  readonly interpose?: boolean;
  /** `odd` is a graph output (an internal value must stay private). */
  readonly internalOutput?: boolean;
  /** `negative` gets a second consumer. */
  readonly extraConsumer?: boolean;
  /** The two slices swap their boundaries (`0..1` is negated — the opposite rotation). */
  readonly swappedSlices?: boolean;
  /** `cat(even, negative)` instead of `cat(negative, even)`. */
  readonly swappedCat?: boolean;
  /** `add(cross, direct)` instead of `add(direct, cross)`. */
  readonly swappedAdd?: boolean;
  /** cos / sin as `[1,1,S,D]` (the BHSD broadcast) while x stays `[1,S,H,D]`. */
  readonly bhsdTables?: boolean;
  /** cos / sin as `[1,S,H,D]` (per-head tables — no broadcast). */
  readonly perHeadTables?: boolean;
  /**
   * Irodori's first use of the shared tables: a `sym_prefix_slice` of each table sits between the
   * reshape back and its `mul` (two in-window nodes the pair form does not accept).
   */
  readonly prefixSlicedTables?: boolean;
  /** Batch 2 (the accepted form is B=1). */
  readonly batch2?: boolean;
};

/**
 * Wan's DiT RoPE (the complex rotation in real numbers): `x` `[1,S,H,D]` is viewed as pairs
 * `[1,S,H·D/2,2]`, the pair is swapped with the second element negated, viewed back and combined
 * with the `[1,S,1,D]` tables. S = 5.
 */
const ropePairGraph = (options: RopePairOptions = {}): DeclarationJson => {
  const heads = options.heads ?? 2;
  const headDim = options.headDim ?? 128;
  const batch = options.batch2 ? 2 : 1;
  const sequence = 5;
  const pairCount = (heads * headDim) / 2;
  const full = [batch, sequence, heads, headDim];
  const pairs = [batch, sequence, pairCount, 2];
  const half = [batch, sequence, pairCount, 1];
  const table = options.perHeadTables
    ? full
    : options.bhsdTables
      ? [1, 1, sequence, headDim]
      : [1, sequence, 1, headDim];
  const values: DeclarationJson["values"] = {
    odd: { dtype: "f32", shape: half },
    negative: { dtype: "f32", shape: half },
    even: { dtype: "f32", shape: half },
    rotated: { dtype: "f32", shape: pairs },
    rot: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: full }];
  const nodes: DeclarationJson["nodes"] = [];
  if (options.separatePairs) {
    inputs.push({ name: "pairs", dtype: "f32", shape: pairs });
  } else {
    values.pairs = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["x"], outs: ["pairs"], attrs: {} });
  }
  const [oddRange, evenRange] = options.swappedSlices
    ? [
        { start: 0, end: 1 },
        { start: 1, end: 2 },
      ]
    : [
        { start: 1, end: 2 },
        { start: 0, end: 1 },
      ];
  nodes.push(
    { op: "slice", ins: ["pairs"], outs: ["odd"], attrs: { dim: 3, ...oddRange } },
    { op: "neg", ins: ["odd"], outs: ["negative"], attrs: {} },
    { op: "slice", ins: ["pairs"], outs: ["even"], attrs: { dim: 3, ...evenRange } },
    {
      op: "cat",
      ins: options.swappedCat ? ["even", "negative"] : ["negative", "even"],
      outs: ["rotated"],
      attrs: { dim: 3 },
    },
  );
  let rotatedForView = "rotated";
  if (options.interpose) {
    values.rotated_alias = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["rotated"], outs: ["rotated_alias"], attrs: {} });
    rotatedForView = "rotated_alias";
  }
  nodes.push({ op: "reshape", ins: [rotatedForView], outs: ["rot"], attrs: {} });
  const prefixSlice = (name: string): void => {
    values[`${name}_table`] = { dtype: "f32", shape: [1, 8, 1, headDim] };
    values[name] = { dtype: "f32", shape: [1, "T", 1, headDim] };
    nodes.push({
      op: "sym_prefix_slice",
      ins: [`${name}_table`],
      outs: [name],
      attrs: { sym: "T", slices: [{ dim: 1, coeff: 1, offset: 0 }] },
    });
  };
  if (options.prefixSlicedTables) prefixSlice("cos");
  nodes.push({ op: "mul", ins: ["x", "cos"], outs: ["direct"], attrs: {} });
  if (options.prefixSlicedTables) prefixSlice("sin");
  nodes.push(
    { op: "mul", ins: ["rot", "sin"], outs: ["cross"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["cross", "direct"] : ["direct", "cross"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.extraConsumer) {
    values.negative_copy = { dtype: "f32", shape: half };
    nodes.push({ op: "neg", ins: ["negative"], outs: ["negative_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: options.prefixSlicedTables ? ["T"] : [],
    inputs: [
      ...inputs,
      ...(options.prefixSlicedTables
        ? [{ name: "bind", dtype: "f32", shape: ["T"] }]
        : [
            { name: "cos", dtype: "f32", shape: table },
            { name: "sin", dtype: "f32", shape: table },
          ]),
    ],
    outputs: [
      ...(options.internalOutput ? ["odd"] : []),
      "y",
      ...(options.extraConsumer ? ["negative_copy"] : []),
    ],
    initializers: options.prefixSlicedTables ? { cos_table: {}, sin_table: {} } : {},
    values,
    nodes,
  };
};

const ropePairInputs = (graph: DeclarationJson): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [spec.name, spec.name === "bind" ? [5] : (spec.shape as number[])]),
  );

const fuseRopePair = (options: RopePairOptions = {}): FusionPlan => {
  const graph = ropePairGraph(options);
  return fuse(graph, ropePairInputs(graph));
};

test("pair-form RoPE folds the 8-node window into one step that binds x and its pair view separately", () => {
  for (const [heads, headDim] of [
    [2, 128],
    [24, 128],
    [10, 64],
    [3, 6],
  ] as const) {
    const label = `H=${heads} D=${headDim}`;
    const plan = fuseRopePair({ heads, headDim });
    // The reshape x → pairs lies before the window and stays a zero-dispatch alias.
    assertEquals(outline(plan.steps), ["reshape", "fused:rope"], label);
    const step = fusedAt(plan, 1);
    // MUST: the total-use list follows the actual node order (slice, slice, then the direct mul).
    assertEquals(step.ins, ["pairs", "pairs", "x", "cos", "sin"], `${label}: total uses`);
    assertEquals(step.binds, ["x", "pairs", "cos", "sin"], `${label}: bind order`);
    assertEquals(step.nodeCount, 8, label);
    assertEquals(step.outputName, "y", label);
    assertEquals(step.dispatches.length, 1, label);
    assertEquals(step.dispatches[0].key, ROPE_PAIR_BSHD_KEY, label);
    // params are [element count, heads (the table divisor), head_dim, half_dim].
    assertEquals(
      [...step.dispatches[0].params],
      [5 * heads * headDim, heads, headDim, headDim / 2],
      label,
    );
    assertEquals(plan.counts.rope, 1, label);
  }
  // The kernel reads the rotation from `pairs` itself, so a pair view that is not a reshape of x
  // (here a graph input of its own) is folded as well — the window's semantics, not the producer.
  const separate = fuseRopePair({ separatePairs: true });
  assertEquals(outline(separate.steps), ["fused:rope"]);
  assertEquals(fusedAt(separate, 0).binds, ["x", "pairs", "cos", "sin"]);
});

test("pair-form RoPE counterexamples (alias / internal output / extra consumer / slice boundaries / cat order / add order / table shapes / batch / Irodori's first use) are not grabbed", () => {
  const cases: readonly (readonly [string, RopePairOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["swapped slice boundaries", { swappedSlices: true }],
    ["cat(even, negative)", { swappedCat: true }],
    ["add(cross, direct)", { swappedAdd: true }],
    // H = S so that the `[1,1,S,D]` table broadcasts at all (a legal mul, not this rotation).
    ["BHSD tables with BSHD x", { bhsdTables: true, heads: 5 }],
    ["per-head tables", { perHeadTables: true }],
    ["batch 2", { batch2: true }],
    // Irodori's DiT / speaker: the first use of the shared tables slices them inside the window.
    // Only the later uses (exactly the Wan spelling) are folded.
    ["two sym_prefix_slice in the window", { prefixSlicedTables: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseRopePair(options);
    assertEquals(plan.counts.rope, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("pair-form RoPE does not take the half-split window and the half split does not take the pair form", () => {
  // The two forms share the leading `slice`; the second op decides. A half-split window keeps
  // its own key, and a pair window never becomes a half-split step.
  const half = fuse(
    ropeGraph({ layout: "bshd", heads: 4 }),
    ropeInputs({ layout: "bshd", heads: 4 }),
  );
  assertEquals(fusedAt(half, 0).dispatches[0].key, ROPE_BSHD_KEY);
  const pair = fuseRopePair({ heads: 4 });
  assertEquals(fusedAt(pair, 1).dispatches[0].key, ROPE_PAIR_BSHD_KEY);
});

// ------------------------------------------------------------ gated residual (gatedResidual)

type GatedResidualOptions = {
  /** `mul(g, y)` instead of `mul(y, g)`. */
  readonly swappedMul?: boolean;
  /** `add(p, residual)` instead of `add(residual, p)`. */
  readonly swappedAdd?: boolean;
  /** The mask is `[1,1,1]` (one flag for every row). */
  readonly broadcastMask?: boolean;
  /** The mask is `[1,rows,dim]` (per element). */
  readonly fullMask?: boolean;
  /** The condition-side gate is `[1,rows,dim]` (not a broadcast vector). */
  readonly perRowGate?: boolean;
  /** `y` is a broadcast `[1,1,dim]` vector. */
  readonly broadcastY?: boolean;
  /** The selected gate is a graph output. */
  readonly gateOutput?: boolean;
  /** The product gets a second consumer. */
  readonly productExtraConsumer?: boolean;
  /** `where(mask, gate_c, gate_c)`. */
  readonly sharedGate?: boolean;
  /** The residual is `y` itself (`add(y, mul(y, g))` — one value in two bind slots). */
  readonly residualIsY?: boolean;
  /** A zero-dispatch reshape of the selected gate between the `where` and the `mul`. */
  readonly gap?: boolean;
};

/** Wan2.2's gated residual: `add(residual, mul(y, where(mask, gate_c, gate_g)))`. */
const gatedResidualGraph = (options: GatedResidualOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const mask = options.broadcastMask ? [1, 1, 1] : options.fullMask ? row : [1, ROWS, 1];
  const gate = options.broadcastMask && !options.perRowGate ? vector : row;
  const values: DeclarationJson["values"] = {
    g: { dtype, shape: gate },
    p: { dtype, shape: row },
    out: { dtype, shape: row },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "mask", dtype: "bool", shape: mask },
    { name: "gate_c", dtype, shape: options.perRowGate ? row : vector },
    { name: "gate_g", dtype, shape: vector },
    { name: "y", dtype, shape: options.broadcastY ? vector : row },
    ...(options.residualIsY ? [] : [{ name: "residual", dtype, shape: row }]),
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "where",
      ins: ["mask", "gate_c", options.sharedGate ? "gate_c" : "gate_g"],
      outs: ["g"],
      attrs: {},
    },
  ];
  let selected = "g";
  if (options.gap) {
    values.g_alias = { dtype, shape: gate };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
    selected = "g_alias";
  }
  const residual = options.residualIsY ? "y" : "residual";
  nodes.push(
    {
      op: "mul",
      ins: options.swappedMul ? [selected, "y"] : ["y", selected],
      outs: ["p"],
      attrs: {},
    },
    {
      op: "add",
      ins: options.swappedAdd ? ["p", residual] : [residual, "p"],
      outs: ["out"],
      attrs: {},
    },
  );
  const extraOutputs: string[] = [];
  if (options.productExtraConsumer) {
    values.p_copy = { dtype, shape: row };
    nodes.push({ op: "neg", ins: ["p"], outs: ["p_copy"], attrs: {} });
    extraOutputs.push("p_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.gateOutput ? ["g"] : []), "out", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

const fuseGatedResidual = (options: GatedResidualOptions = {}): FusionPlan => {
  const graph = gatedResidualGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("the gated residual folds where → mul → add into one step that reads the gate per token", () => {
  const plan = fuseGatedResidual();
  assertEquals(outline(plan.steps), ["fused:gatedResidual"]);
  const step = fusedAt(plan, 0);
  assertEquals(step.ins, ["mask", "gate_c", "gate_g", "y", "residual"]);
  assertEquals(step.binds, ["mask", "gate_c", "gate_g", "y", "residual"]);
  assertEquals(step.nodeCount, 3);
  assertEquals(step.outputName, "out");
  assertEquals(step.temps, []);
  assertEquals(step.dispatches.length, 1);
  assertEquals(step.dispatches[0].key, GATED_RESIDUAL_KEY);
  assertEquals([...step.dispatches[0].params], [ROWS * DIM, DIM, 0, 0]);
  assertEquals(step.dispatches[0].workgroups, {
    kind: "gridStride",
    items: ROWS * DIM,
    size: 256,
  });
  assertEquals(plan.counts.gatedResidual, 1);
});

test("gated residual counterexamples (input orders / mask, gate and y shapes / internal output / extra consumer / bind duplicates / gap) are not grabbed", () => {
  const cases: readonly (readonly [string, GatedResidualOptions])[] = [
    ["mul(g, y)", { swappedMul: true }],
    ["add(p, residual)", { swappedAdd: true }],
    ["mask [1,1,1]", { broadcastMask: true }],
    ["mask per element", { fullMask: true }],
    ["per-row gate", { perRowGate: true }],
    ["broadcast y", { broadcastY: true }],
    ["selected gate is a graph output", { gateOutput: true }],
    ["product has another consumer", { productExtraConsumer: true }],
    ["where(mask, g, g)", { sharedGate: true }],
    ["residual is y", { residualIsY: true }],
    ["reshape between where and mul", { gap: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseGatedResidual(options);
    assertEquals(plan.counts.gatedResidual, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("a Wan2.2-like block folds the gated residual and the per-token adaLN next to each other", () => {
  // where → mul → add (the attention's gated residual) feeds the next layer_norm of the per-token
  // adaLN (the FFN's modulation), as in Wan2.2's DiT blocks.
  const graph = gatedResidualGraph();
  const token = adalnTokenGraph();
  const rename = (name: string): string =>
    name === "x" ? "out" : name === "mask" ? "mask" : `ffn_${name}`;
  graph.inputs.push(
    ...token.inputs
      .filter((spec) => spec.name !== "x" && spec.name !== "mask")
      .map((spec) => ({ ...spec, name: rename(spec.name) })),
  );
  for (const [name, spec] of Object.entries(token.values)) graph.values[rename(name)] = spec;
  graph.nodes.push(
    ...token.nodes.map((node) => ({
      ...node,
      ins: node.ins.map(rename),
      outs: node.outs.map(rename),
    })),
  );
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  graph.outputs = ["ffn_y"];
  const plan = fuse(graph, adalnInputs(graph));
  assertEquals(outline(plan.steps), ["fused:gatedResidual", "fused:adaln"]);
  assertEquals(plan.counts.gatedResidual, 1);
  assertEquals(plan.counts.adaln, 1);
  // The mask is consumed by both steps (1 + 2 total uses), and the residual output feeds the
  // adaLN step.
  assertEquals(fusedAt(plan, 1).ins.slice(0, 1), ["out"]);
});

// ------------------------------------------------ linear prologue (perf-ledger F-4)

/** Wan2.2's FFN down-projection row (K 14,336). N is shrunk: only K enters the decision. */
const PROLOGUE_K = 14336;
const PROLOGUE_N = 16;
const PROLOGUE_ROWS = 7;

type ProloguePairOptions = {
  /** The producer op (default `gelu_tanh`). */
  readonly op?: string;
  /** The rows S of the activation `[1, S, K]` (default 7). */
  readonly rows?: number;
  /** The reduction length K (default {@link PROLOGUE_K}). */
  readonly k?: number;
  /** The producer's output gets a second consumer. */
  readonly extraConsumer?: boolean;
  /** The producer's output is a graph output. */
  readonly producerOutput?: boolean;
  /** The consumer is a `matmul` reading the output as its activation (rank 2). */
  readonly matmulConsumer?: boolean;
  /** The linear reads the producer's output in its bias slot (the activation is another input). */
  readonly biasSlot?: boolean;
  /** An unrelated node between the producer and the linear. */
  readonly gap?: boolean;
};

/** `y = linear(gelu_tanh(x), w, b)` — the shape of the Wan DiT's FFN down projection. */
const prologuePairGraph = (options: ProloguePairOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const k = options.k ?? PROLOGUE_K;
  const n = PROLOGUE_N;
  const rows = options.rows ?? PROLOGUE_ROWS;
  const matmul = options.matmulConsumer === true;
  const row = matmul ? [rows, k] : [1, rows, k];
  const produced = options.biasSlot ? [n] : row;
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype, shape: produced }];
  const values: DeclarationJson["values"] = {
    g: { dtype, shape: produced },
    w: { dtype, shape: matmul ? [k, n] : [n, k] },
    y: { dtype, shape: matmul ? [rows, n] : [1, rows, n] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  const nodes: DeclarationJson["nodes"] = [
    { op: options.op ?? "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} },
  ];
  const outputs = options.producerOutput ? ["g"] : [];
  if (options.gap) {
    inputs.push({ name: "z", dtype, shape: [4] });
    values.z_neg = { dtype, shape: [4] };
    nodes.push({ op: "neg", ins: ["z"], outs: ["z_neg"], attrs: {} });
    outputs.push("z_neg");
  }
  if (matmul) {
    nodes.push({ op: "matmul", ins: ["g", "w"], outs: ["y"], attrs: {} });
  } else if (options.biasSlot) {
    inputs.push({ name: "a", dtype, shape: row });
    nodes.push({ op: "linear", ins: ["a", "w", "g"], outs: ["y"], attrs: {} });
  } else {
    values.b = { dtype, shape: [n] };
    initializers.b = {};
    nodes.push({ op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} });
  }
  outputs.push("y");
  if (options.extraConsumer) {
    values.g_neg = { dtype, shape: produced };
    nodes.push({ op: "neg", ins: ["g"], outs: ["g_neg"], attrs: {} });
    outputs.push("g_neg");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs,
    initializers,
    values,
    nodes,
  };
};

type PrologueContext = Partial<Parameters<typeof planFusions>[1]>;

/** The a8 Session's facts: `linearCompute: "a8"` and the weight resident as i8. */
const A8_I8: PrologueContext = {
  linearCompute: "a8",
  weightLayouts: new Map([["w", { storage: "i8" }]]),
};

const fusePrologue = (
  options: ProloguePairOptions = {},
  context: PrologueContext = A8_I8,
): FusionPlan => {
  const graph = prologuePairGraph(options);
  const ir = parse(graph);
  return planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...context,
  });
};

const NO_FUSIONS = {
  silu: 0,
  upsample2x: 0,
  rope: 0,
  adaln: 0,
  gatedResidual: 0,
  rmsNormAdd: 0,
  linearStaticQuantize: 0,
  rowBlockAttention: 0,
  identityExpand: 0,
  packedStaticQuantize: 0,
  linearPrologue: 0,
} as const;

/** A node step's `prologue` (undefined for fused steps and plain nodes). */
const prologueOf = (step: ExecStep) => (step.kind === "node" ? step.prologue : undefined);

test("the linear prologue folds gelu_tanh into the a8 linear that reads it: one node step whose input is the producer's input", () => {
  const cases: readonly (readonly [string, ProloguePairOptions, PrologueContext])[] = [
    ["Wan2.2 down projection (K 14,336, i8)", {}, A8_I8],
    [
      "i4 residency (w4a8)",
      {},
      { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "i4", groupSize: 32 }]]) },
    ],
    ["the prologue bound (K 16,384)", { k: QUANTIZE_ROWS_PROLOGUE_MAX_DIM }, A8_I8],
    ["the smallest row (K 4)", { k: 4 }, A8_I8],
    ["one row", { rows: 1 }, A8_I8],
  ];
  for (const [label, options, context] of cases) {
    const plan = fusePrologue(options, context);
    assertEquals(outline(plan.steps), ["linear"], label);
    const [step] = plan.steps;
    assertEquals(prologueOf(step), { op: "gelu_tanh" }, label);
    if (step.kind !== "node") throw new Error(`${label}: not a node step`);
    assertEquals(step.plan.node.ins, ["x", "w", "b"], `${label}: re-pointed input`);
    assertEquals(
      step.plan.outputs.map((out) => out.name),
      ["y"],
      label,
    );
    assertEquals(step.aliasesInput, false, label);
    assertEquals(plan.counts, { ...NO_FUSIONS, linearPrologue: 1 }, label);
  }
});

test("the linear prologue keeps the total-use accounting exact: the producer's input trades the producer's use for the linear's, and the folded value is never consumed or defined", () => {
  // up projection → gelu_tanh → down projection, with the up projection's output `h` also read by
  // a residual add after the down projection (so `h` has two uses before and after the fold).
  const dtype = "f32";
  const row = [1, 4, 8];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "gelu_tanh", "linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype, shape: row }],
    outputs: ["out"],
    initializers: { w_up: {}, b_up: {}, w: {}, b: {} },
    values: {
      w_up: { dtype, shape: [8, 8] },
      b_up: { dtype, shape: [8] },
      w: { dtype, shape: [8, 8] },
      b: { dtype, shape: [8] },
      h: { dtype, shape: row },
      g: { dtype, shape: row },
      y: { dtype, shape: row },
      out: { dtype, shape: row },
    },
    nodes: [
      { op: "linear", ins: ["x", "w_up", "b_up"], outs: ["h"], attrs: {} },
      { op: "gelu_tanh", ins: ["h"], outs: ["g"], attrs: {} },
      { op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} },
      { op: "add", ins: ["y", "h"], outs: ["out"], attrs: {} },
    ],
  };
  const ir = parse(graph);
  const uses = countUses(ir);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, { x: row })).nodes, {
    useCounts: uses,
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    linearCompute: "a8",
    weightLayouts: new Map([
      ["w_up", { storage: "i8" }],
      ["w", { storage: "i8" }],
    ]),
  });
  assertEquals(outline(plan.steps), ["linear", "linear", "add"]);
  assertEquals(plan.steps.map(prologueOf), [undefined, { op: "gelu_tanh" }, undefined]);
  // The release bookkeeping reads `plan.node.ins` of a node step and `ins` of a fused step.
  const consumed = new Map<string, number>();
  const defined = new Set<string>();
  for (const step of plan.steps) {
    const names = step.kind === "node" ? step.plan.node.ins : step.ins;
    for (const name of names) consumed.set(name, (consumed.get(name) ?? 0) + 1);
    if (step.kind === "node") for (const out of step.plan.outputs) defined.add(out.name);
    else defined.add(step.outputName);
  }
  assertEquals(uses.get("g"), 1, "the folded value had exactly one use");
  assertEquals(consumed.get("g"), undefined, "the folded value is never consumed");
  assertEquals(defined.has("g"), false, "the folded value is never defined");
  for (const [name, count] of uses) {
    if (name === "g") continue;
    assertEquals(consumed.get(name), count, `total uses of '${name}'`);
  }
  assertEquals(consumed.get("h"), 2, "h: the down projection and the residual add");
});

test("linear prologue counterexamples (use count 2 / graph output / not a linear / another slot / a gap / K / erf gelu / no rows) keep both nodes as they are", () => {
  const cases: readonly (readonly [string, ProloguePairOptions])[] = [
    ["the output has a second consumer", { extraConsumer: true }],
    ["the output is a graph output", { producerOutput: true }],
    ["the consumer is a matmul", { matmulConsumer: true }],
    ["the linear reads it in the bias slot", { biasSlot: true }],
    ["an unrelated node between the two", { gap: true }],
    ["K % 4 != 0", { k: PROLOGUE_K - 2 }],
    ["K above the prologue bound", { k: QUANTIZE_ROWS_PROLOGUE_MAX_DIM + 4 }],
    ["erf gelu (not accepted yet)", { op: "gelu" }],
    ["no rows", { rows: 0 }],
  ];
  for (const [label, options] of cases) {
    const plan = fusePrologue(options);
    assertEquals(plan.counts.linearPrologue, 0, `${label}: counter`);
    assertEquals(
      outline(plan.steps),
      prologuePairGraph(options).nodes.map((node) => node.op),
      `${label}: plain nodes`,
    );
    assertEquals(
      plan.steps.map(prologueOf),
      plan.steps.map(() => undefined),
      label,
    );
  }
});

test("the linear prologue needs the a8 Session's i8a8 route: f32 / f16 / bf16 weights and linearCompute f32 / f16 / unset do not fold", () => {
  const cases: readonly (readonly [string, PrologueContext])[] = [
    ["f32 weight (not resident compressed)", { linearCompute: "a8", weightLayouts: new Map() }],
    ["no weight layouts", { linearCompute: "a8" }],
    ["f16 weight", { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "f16" }]]) }],
    ["bf16 weight", { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "bf16" }]]) }],
    ["linearCompute f32", { ...A8_I8, linearCompute: "f32" }],
    ["linearCompute f16", { ...A8_I8, linearCompute: "f16" }],
    ["linearCompute unset", { weightLayouts: A8_I8.weightLayouts }],
  ];
  for (const [label, context] of cases) {
    const plan = fusePrologue({}, context);
    assertEquals(plan.counts.linearPrologue, 0, `${label}: counter`);
    assertEquals(outline(plan.steps), ["gelu_tanh", "linear"], label);
  }
});

test("the linear prologue rejects a producer that is not f32 in and out", () => {
  // The contract runs gelu_tanh on f32 only, so a planned graph cannot carry another dtype; the
  // matcher's own check is exercised on a plan with the producer's dtypes replaced.
  const graph = prologuePairGraph();
  const ir = parse(graph);
  const nodes = planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes;
  const [producer, linear] = nodes;
  for (const [label, replaced] of [
    ["input i32", { ...producer, inputDtypes: ["i32"] }],
    ["output i32", { ...producer, outputs: [{ ...producer.outputs[0], dtype: "i32" }] }],
  ] as const) {
    const plan = planFusions([replaced, linear], {
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: TEST_LIMITS,
      ...A8_I8,
    });
    assertEquals(plan.counts.linearPrologue, 0, label);
    assertEquals(outline(plan.steps), ["gelu_tanh", "linear"], label);
  }
});

test("no fusion rule's head is a prologue op (the rules are tried first at each index, so the order cannot change the result)", () => {
  const heads = new Set(FUSION_RULES.flatMap((rule) => rule.heads));
  assertEquals(
    LINEAR_PROLOGUE_OPS.filter((op) => heads.has(op)),
    [],
  );
  assertEquals(LINEAR_PROLOGUE_OPS, ["gelu_tanh"]);
});
