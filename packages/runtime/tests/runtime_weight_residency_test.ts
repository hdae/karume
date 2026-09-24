// 常駐プランナ（src/runtime/weight-residency.ts）の門。GPU を一切使わない
// （容器（golden の `krm` / 合成グラフのメモリ内容器）を開いて、宣言だけから決まる席と
// バイト数を見る）。
//
// 見るのは 2 点:
// ① **席**（どの重みが f16 / i8 / i4 常駐・CPU 展開・生バイト常駐のどれに落ちるか）—
//    期待値は fixture ごとに手書きの定数で置く。プランナと同じ述語で組み直すと恒真化する。
// ② **宣言由来バイト数 = 容器の供給計画のバイト数**（payload / scale とも）— プランナは
//    束縛表も目次も見ずに宣言 shape と席だけで数えるので、ここが「宣言と現物が同じ数を指す」
//    唯一の突合になる。i8 の scale は**チャネル軸の取り違えが即バイト数の違いになる**
//    （conv_transpose1d の `[Cin,Cout,K]` は軸 1 — 軸 0 と読むと golden の実 scale と一致しない）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { type InitializerSupply, mergedGraph } from "../src/format/container/bind.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planWeightResidency, type WeightResidency } from "../src/runtime/weight-residency.ts";
import { openSeriesContainer } from "./helpers/container-files.ts";
import { autoTensors, type StorageMap } from "./helpers/merged-graph.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const GOLDEN_ROOT = new URL("./fixtures/golden/", import.meta.url);
/** golden 1 件の容器の代表 path（`goldens.py` の `MODEL_FILE`）。 */
const MODEL_FILE = "model.krm";

/** 席とバイト数の突合に要るぶんだけの面（合流後のグラフと供給計画）。 */
type ResidencyTarget = {
  readonly graph: IrGraph;
  readonly supplies: ReadonlyMap<string, InitializerSupply>;
};

/**
 * golden 1 件の**容器**を開き、合流後のグラフと供給計画を返す（グラフ名 = 置き場の
 * ディレクトリ名 — `goldens.py` の `graph_name=spec.name`）。
 *
 * ここは GPU も Session も通さないので、`prepareContainer` ではなく素の合流面で開く。
 * 読むのはヘッダと 2 文書だけで、block へは 1 バイトも進まない。
 */
const openGoldenContainer = async (model: string): Promise<ResidencyTarget> => {
  const opened = await openSeriesContainer(new URL(`${model}/${MODEL_FILE}`, GOLDEN_ROOT));
  const bound = opened.graphs[model];
  assert(
    bound !== undefined,
    `容器にグラフ '${model}' が無い（在るのは ${Object.keys(opened.graphs).join(" / ")}）`,
  );
  return { graph: mergedGraph(bound, model), supplies: bound.supplies };
};

/**
 * 合成グラフを**メモリ内容器**で開く（`krm` を書かずに同じ合流面を得る）。供給の中身は
 * 見ないので、宣言 shape と codec から決まる長さの 0 埋めでよい（helpers/merged-graph.ts）。
 */
const openGraph = (graph: DeclarationJson, storage: StorageMap = {}): ResidencyTarget => {
  const bound = memoryModel(graph, autoTensors(graph, storage)).graphs[GRAPH_NAME];
  return { graph: mergedGraph(bound, GRAPH_NAME), supplies: bound.supplies };
};

/** 名前 → 席（期待値との突合は席だけを見る — バイト数は現物との突合が別に見る）。 */
const seats = (model: { readonly graph: IrGraph }): Record<string, WeightResidency["seat"]> =>
  Object.fromEntries(
    [...planWeightResidency(model.graph)].map(([name, plan]) => [name, plan.seat]),
  );

/**
 * 宣言由来のバイト数を**容器の供給計画**と突き合わせる（payload と scale の両方）。
 *
 * 常駐プランナ（`weight-residency.ts`）と容器の合流層（`format/container/bind.ts`）は同じ量を
 * **別々に**導く — 前者は IR の宣言 shape と席から、後者は束縛表の encoding と目次から。
 * したがってここは 2 実装の突合であって恒真ではない。チャネル軸の取り違え
 * （conv_transpose1d の `[Cin,Cout,K]` は行軸 1）は scale のバイト数の違いとして出る。
 *
 * NOTE: ただしチャネル軸の主張が効くのは **golden の側だけ**。合成グラフの供給を作る
 * `autoTensors` は消費側 op を見ずに rowAxis 0 固定で scale 長を決めるので、合成側の突合は
 * 「宣言 shape から同じ軸で 2 度数えた」形にしかならない。
 */
const assertDeclaredBytesMatchContainer = (golden: ResidencyTarget): void => {
  const { graph, supplies } = golden;
  const plan = planWeightResidency(graph);
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const seat = plan.get(name);
    assert(seat !== undefined, `initializer '${name}' の席が無い`);
    // 共有宣言（借り物 — ADR 0096 段 2 §1.3）は実体を容器に持たないので、この助手の対象外
    // （対象の資産に 1 本も無いことを門にする）。
    assert(initializer.shared === undefined, `initializer '${name}' が共有宣言`);
    assert(seat.seat !== "shared", `initializer '${name}' の席が shared`);
    const supply = supplies.get(name);
    assert(supply !== undefined, `initializer '${name}' の供給計画が無い`);
    // piece 分割された実体は block 列で来る（container-v1 §5）ので payload は足し合わせる。
    const payloadBytes = supply.blocks.reduce((total, block) => total + block.payloadBytes, 0);
    assertEquals(seat.payloadBytes, payloadBytes, `${name} の payload バイト数`);
    if (seat.seat !== "i8" && seat.seat !== "i4") continue;
    assert(supply.scale !== undefined, `initializer '${name}' に scale の供給が無い`);
    assertEquals(seat.scaleBytes, supply.scale.payloadBytes, `${name} の scale バイト数`);
  }
};

// ---------------------------------------------------------------------------
// 既存 fixture（実エクスポータが書いた配布形）との突合
// ---------------------------------------------------------------------------

Deno.test("golden `mlp`: 圧縮しない格納は全て生バイト常駐の席", async () => {
  const model = await openGoldenContainer("mlp");
  assertEquals(seats(model), { w1: "raw", b1: "raw", w2: "raw", b2: "raw" });
  assertDeclaredBytesMatchContainer(model);
});

Deno.test("golden `i8_weights`: 重みスロット消費の i8 は全て i8 常駐（bias は f32 のまま）", async () => {
  const model = await openGoldenContainer("i8_weights");
  assertEquals(seats(model), {
    // embedding / linear / conv1d / conv_transpose1d / conv2d の重みスロット
    "table.weight": "i8",
    "dense.weight": "i8",
    "conv.weight": "i8",
    "up.weight": "i8",
    "image.weight": "i8",
    // bias は適格判定に載らないスロット（ADR 0006 が名指しした「bias が weight を道連れに
    // 降格させる」形を作らない）ので、f32 のまま生バイト常駐
    "dense.bias": "raw",
    "conv.bias": "raw",
    "up.bias": "raw",
    "image.bias": "raw",
  });
  // conv_transpose1d の `up.weight` は `[Cin,Cout,K]` = [5,2,3] でチャネル軸が **1**。
  // 軸 0 と取り違えると scale が 5 要素（20 バイト）になり、容器側の 2 要素（8 バイト）と外れる。
  assertDeclaredBytesMatchContainer(model);
});

Deno.test("golden `conv_transpose` / `embedding_lookup`: 宣言由来バイト数が容器と一致", async () => {
  assertDeclaredBytesMatchContainer(await openGoldenContainer("conv_transpose"));
  assertDeclaredBytesMatchContainer(await openGoldenContainer("embedding_lookup"));
});

// ---------------------------------------------------------------------------
// 席の分岐（適格 / 適格外）
// ---------------------------------------------------------------------------

/**
 * linear の重み（適格）と mul の被演算子（適格外）に**同じ格納**を置ける形のグラフ。
 * IR v2 の宣言は格納を持たないので、`w` / `g` の codec は供給側（`openGraph` の第 2 引数）が決める。
 */
const twoPathGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 3] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
    g: {},
  },
  values: {
    w: { dtype: "f32", shape: [3, 3] },
    b: { dtype: "f32", shape: [3] },
    g: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [2, 3] },
    y: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "g"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("f16: 重みスロットだけの消費は f16 席・重みスロット以外の消費は CPU 展開の席", () => {
  const model = openGraph(twoPathGraph(), { w: "f16", g: "f16" });
  const plan = planWeightResidency(model.graph);
  assertEquals(seats(model), { w: "f16", b: "raw", g: "expanded" });
  // f16 の payload は numel×2（整列の詰め物は転送側の責務なので席には現れない）
  assertEquals(plan.get("w"), { seat: "f16", payloadBytes: 18 });
  // 展開後は f32 の numel×4 — 宣言由来の payload（6 バイト）とは別欄で持つ
  assertEquals(plan.get("g"), { seat: "expanded", payloadBytes: 6, expandedBytes: 12 });
  assertDeclaredBytesMatchContainer(model);
});

Deno.test("i8: 席は scale のバイト数とチャネル軸を伴う", () => {
  const model = openGraph(twoPathGraph(), { w: "int8-sym", g: "f16" });
  // linear の重みは `[out,in]` なので行の軸 0 → scale は 3 要素 = 12 バイト
  assertEquals(planWeightResidency(model.graph).get("w"), {
    seat: "i8",
    payloadBytes: 9,
    scaleBytes: 12,
    rowAxis: 0,
  });
  assertDeclaredBytesMatchContainer(model);
});

/** `conv1d(x, w, b)` 1 本のグラフ（w は i4 + rank 2 の group scale）。 */
const i4Conv1dGraph = (groups: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 32, 6] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
  },
  values: {
    // 行長 = Cin/groups · K（groups == 1 なら 32·2 = 64 = g16 が 4 つ）
    w: { dtype: "f32", shape: [4, 32 / groups, 2] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 5] },
  },
  nodes: [{
    op: "conv1d",
    ins: ["x", "w", "b"],
    outs: ["y"],
    attrs: { stride: 1, padding: 0, dilation: 1, groups },
  }],
});

const i4Conv1dModel = (groups: number): ResidencyTarget =>
  openGraph(i4Conv1dGraph(groups), { w: { codec: "int4-sym-g", groupSize: 16 } });

Deno.test("i4: 展開経路のある conv1d(groups==1) は i4 席・groups>1 は CPU 展開の席", () => {
  const direct = i4Conv1dModel(1);
  // numel 256 → packed 128 バイト / group scale は [4, 4] の 16 要素 = 64 バイト
  assertEquals(planWeightResidency(direct.graph).get("w"), {
    seat: "i4",
    payloadBytes: 128,
    scaleBytes: 64,
    groupSize: 16,
  });
  assertDeclaredBytesMatchContainer(direct);

  // groups > 1 は直接カーネルへ流れる = 展開経路が無いので CPU 展開へ落ちる
  const grouped = i4Conv1dModel(2);
  // numel 128 → packed 64 バイト・展開後は 128×4 = 512 バイト
  assertEquals(planWeightResidency(grouped.graph).get("w"), {
    seat: "expanded",
    payloadBytes: 64,
    expandedBytes: 512,
  });
  assertDeclaredBytesMatchContainer(grouped);
});

Deno.test("グラフ出力に載った initializer は圧縮常駐しない（readback が f32 を仮定する）", () => {
  const graph = twoPathGraph();
  const model = openGraph({ ...graph, outputs: ["y", "w"] }, { w: "f16", g: "f16" });
  assertEquals(seats(model)["w"], "expanded");
});

// ---------------------------------------------------------------------------
// fail loudly
// ---------------------------------------------------------------------------

Deno.test("チャネル軸が消費側で食い違う i8 は席を決めずに落ちる", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "conv_transpose1d"] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [2, 3] },
      { name: "z", dtype: "f32", shape: [1, 3, 4] },
    ],
    outputs: ["h", "u"],
    initializers: {
      // linear は軸 0・conv_transpose1d は軸 1 を要求する（ADR 0019）
      w: {},
      b: {},
      c: {},
    },
    values: {
      w: { dtype: "f32", shape: [3, 3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      h: { dtype: "f32", shape: [2, 3] },
      u: { dtype: "f32", shape: [1, 3, 5] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
      {
        op: "conv_transpose1d",
        ins: ["z", "w", "c"],
        outs: ["u"],
        attrs: { stride: 1, padding: 0 },
      },
    ],
  };
  // チャネル軸は消費側 op から導くので、常駐計画が席を決める前に落ちる（合流層は消費側を
  // 見ないので、落ちるのは `openGraph` ではなく `planWeightResidency` の側）。
  const model = openGraph(graph, { w: "int8-sym" });
  assertThrows(
    () => planWeightResidency(model.graph),
    ExecutionError,
    "チャネル軸が消費側で食い違う",
  );
});
