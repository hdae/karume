// 常駐プランナ（src/runtime/weight-residency.ts）の門。GPU を一切使わない
// （配布形を container まで開いて、宣言だけから決まる席とバイト数を見る）。
//
// 見るのは 2 点:
// ① **席**（どの重みが f16 / i8 / i4 常駐・CPU 展開・生バイト常駐のどれに落ちるか）—
//    期待値は fixture ごとに手書きの定数で置く。プランナと同じ述語で組み直すと恒真化する。
// ② **宣言由来バイト数 = 実テンソルのバイト数**（payload / scale とも）— プランナは
//    safetensors を一切見ずに数えるので、ここが「宣言と現物が同じ数を指す」唯一の突合になる。
//    i8 の scale は**チャネル軸の取り違えが即バイト数の違いになる**（conv_transpose1d の
//    `[Cin,Cout,K]` は軸 1 — 軸 0 と読むと golden の実 scale と一致しない）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { type KarumeModel, openModel } from "../src/format/container.ts";
import { parseIrGraph } from "../src/format/ir.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planWeightResidency, type WeightResidency } from "../src/runtime/weight-residency.ts";
import { f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./helpers/f16.ts";
import { graphModelBuffer } from "./helpers/graph.ts";

const GOLDEN_ROOT = new URL("./fixtures/golden/", import.meta.url);

const openGolden = (model: string): KarumeModel => {
  const bytes = Deno.readFileSync(new URL(`${model}/model.safetensors`, GOLDEN_ROOT));
  return openModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};

const openGraph = (graph: GraphJson, tensors: readonly TensorSpec[] = []): KarumeModel =>
  openModel(graphModelBuffer(graph, tensors));

/** f16 のバイト列（値そのものは見ないので 0 で埋める — 見るのはバイト数と席だけ）。 */
const f16Zeros = (count: number): Uint8Array<ArrayBuffer> =>
  f16BytesFromBits(new Array(count).fill(f32ToF16Bits(0)));

/** 名前 → 席（期待値との突合は席だけを見る — バイト数は現物との突合が別に見る）。 */
const seats = (model: KarumeModel): Record<string, WeightResidency["seat"]> =>
  Object.fromEntries(
    [...planWeightResidency(model.graph)].map(([name, plan]) => [name, plan.seat]),
  );

/**
 * 宣言由来のバイト数を実テンソルと突き合わせる（全 initializer・payload と scale の両方）。
 * 圧縮常駐しない席にも payload の突合は掛かる（宣言由来の数え方は席に依らない）。
 */
const assertDeclaredBytesMatchFile = (model: KarumeModel): void => {
  const { graph, file } = model;
  const plan = planWeightResidency(graph);
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const seat = plan.get(name);
    assert(seat !== undefined, `initializer '${name}' の席が無い`);
    const view = file.tensors.get(initializer.tensor);
    assert(view !== undefined, `テンソル '${initializer.tensor}' が無い`);
    assertEquals(seat.payloadBytes, view.byteLength, `${name} の payload バイト数`);
    if (seat.seat !== "i8" && seat.seat !== "i4") continue;
    const scaleKey = initializer.storage.scale;
    assert(scaleKey !== undefined, `initializer '${name}' に scale が無い`);
    const scale = file.tensors.get(scaleKey);
    assert(scale !== undefined, `scale テンソル '${scaleKey}' が無い`);
    assertEquals(seat.scaleBytes, scale.byteLength, `${name} の scale バイト数`);
  }
};

// ---------------------------------------------------------------------------
// 既存 fixture（実エクスポータが書いた配布形）との突合
// ---------------------------------------------------------------------------

Deno.test("golden `mlp`: 圧縮しない格納は全て生バイト常駐の席", () => {
  const model = openGolden("mlp");
  assertEquals(seats(model), { p_w1: "raw", p_b1: "raw", p_w2: "raw", p_b2: "raw" });
  assertDeclaredBytesMatchFile(model);
});

Deno.test("golden `i8_weights`: 重みスロット消費の i8 は全て i8 常駐（bias は f32 のまま）", () => {
  const model = openGolden("i8_weights");
  assertEquals(seats(model), {
    // embedding / linear / conv1d / conv_transpose1d / conv2d の重みスロット
    p_table_weight: "i8",
    p_dense_weight: "i8",
    p_conv_weight: "i8",
    p_up_weight: "i8",
    p_image_weight: "i8",
    // bias は適格判定に載らないスロット（ADR 0006 が名指しした「bias が weight を道連れに
    // 降格させる」形を作らない）ので、f32 のまま生バイト常駐
    p_dense_bias: "raw",
    p_conv_bias: "raw",
    p_up_bias: "raw",
    p_image_bias: "raw",
  });
  // conv_transpose1d の `p_up_weight` は `[Cin,Cout,K]` = [5,2,3] でチャネル軸が **1**。
  // 軸 0 と取り違えると scale が 5 要素（20 バイト）になり、現物の 2 要素（8 バイト）と外れる。
  assertDeclaredBytesMatchFile(model);
});

Deno.test("golden `conv_transpose` / `embedding_lookup`: 宣言由来バイト数が現物と一致", () => {
  assertDeclaredBytesMatchFile(openGolden("conv_transpose"));
  assertDeclaredBytesMatchFile(openGolden("embedding_lookup"));
});

// ---------------------------------------------------------------------------
// 席の分岐（適格 / 適格外）
// ---------------------------------------------------------------------------

/** linear の重み（適格）と mul の被演算子（適格外）に同じ格納 dtype を置くグラフ。 */
const twoPathGraph = (storage: Record<string, unknown>): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 3] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { ...storage } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
    g: { tensor: "m.g", storage: { dtype: "f16" } },
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
  const model = openGraph(twoPathGraph({ dtype: "f16" }), [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.w", dtype: "F16", shape: [3, 3], data: f16Zeros(9) },
    { name: "m.g", dtype: "F16", shape: [3], data: f16Zeros(3) },
  ]);
  const plan = planWeightResidency(model.graph);
  assertEquals(seats(model), { w: "f16", b: "raw", g: "expanded" });
  // f16 の payload は numel×2（整列の詰め物は転送側の責務なので席には現れない）
  assertEquals(plan.get("w"), { seat: "f16", payloadBytes: 18 });
  // 展開後は f32 の numel×4 — 宣言由来の payload（6 バイト）とは別欄で持つ
  assertEquals(plan.get("g"), { seat: "expanded", payloadBytes: 6, expandedBytes: 12 });
  assertDeclaredBytesMatchFile(model);
});

Deno.test("i8: 席は scale のバイト数とチャネル軸を伴う", () => {
  // 整列降順（F32 → F16 → I8）で詰める（整列単位を跨がせない）
  const model = openGraph(twoPathGraph({ dtype: "i8", scale: "m.s" }), [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
    { name: "m.g", dtype: "F16", shape: [3], data: f16Zeros(3) },
    { name: "m.w", dtype: "I8", shape: [3, 3], data: new Uint8Array(9) },
  ]);
  // linear の重みは `[out,in]` なのでチャネル軸 0 → scale は 3 要素 = 12 バイト
  assertEquals(planWeightResidency(model.graph).get("w"), {
    seat: "i8",
    payloadBytes: 9,
    scaleBytes: 12,
    channelAxis: 0,
  });
  assertDeclaredBytesMatchFile(model);
});

/** `conv1d(x, w, b)` 1 本のグラフ（w は i4 + rank 2 の group scale）。 */
const i4Conv1dGraph = (groups: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 32, 6] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: 16 } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
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

const i4Conv1dModel = (groups: number): KarumeModel => {
  const rowLength = (32 / groups) * 2;
  return openGraph(i4Conv1dGraph(groups), [
    { name: "m.b", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    {
      name: "m.s",
      dtype: "F32",
      shape: [4, rowLength / 16],
      data: f32Bytes(new Array(4 * (rowLength / 16)).fill(1)),
    },
    {
      name: "m.w",
      dtype: "I4",
      shape: [4, 32 / groups, 2],
      data: new Uint8Array((4 * rowLength) / 2),
    },
  ]);
};

Deno.test("i4: 展開経路のある conv1d(groups==1) は i4 席・groups>1 は CPU 展開の席", () => {
  const direct = i4Conv1dModel(1);
  // numel 256 → packed 128 バイト / group scale は [4, 4] の 16 要素 = 64 バイト
  assertEquals(planWeightResidency(direct.graph).get("w"), {
    seat: "i4",
    payloadBytes: 128,
    scaleBytes: 64,
    groupSize: 16,
  });
  assertDeclaredBytesMatchFile(direct);

  // groups > 1 は直接カーネルへ流れる = 展開経路が無いので CPU 展開へ落ちる
  const grouped = i4Conv1dModel(2);
  // numel 128 → packed 64 バイト・展開後は 128×4 = 512 バイト
  assertEquals(planWeightResidency(grouped.graph).get("w"), {
    seat: "expanded",
    payloadBytes: 64,
    expandedBytes: 512,
  });
  assertDeclaredBytesMatchFile(grouped);
});

Deno.test("グラフ出力に載った initializer は圧縮常駐しない（readback が f32 を仮定する）", () => {
  const graph = twoPathGraph({ dtype: "f16" });
  const model = openGraph({ ...graph, outputs: ["y", "w"] }, [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.w", dtype: "F16", shape: [3, 3], data: f16Zeros(9) },
    { name: "m.g", dtype: "F16", shape: [3], data: f16Zeros(3) },
  ]);
  assertEquals(seats(model).w, "expanded");
});

// ---------------------------------------------------------------------------
// fail loudly
// ---------------------------------------------------------------------------

Deno.test("チャネル軸が消費側で食い違う i8 は席を決めずに落ちる", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear", "conv_transpose1d"] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [2, 3] },
      { name: "z", dtype: "f32", shape: [1, 3, 4] },
    ],
    outputs: ["h", "u"],
    initializers: {
      // linear は軸 0・conv_transpose1d は軸 1 を要求する（ADR 0019）
      w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
      c: { tensor: "m.c", storage: { dtype: "f32" } },
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
  const model = openGraph(graph, [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.c", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
    { name: "m.w", dtype: "I8", shape: [3, 3], data: new Uint8Array(9) },
  ]);
  assertThrows(
    () => planWeightResidency(model.graph),
    ExecutionError,
    "チャネル軸が消費側で食い違う",
  );
});

Deno.test("配布形を開かずグラフだけで席とバイト数が決まる（実テンソルを見ない）", () => {
  // 配布形（safetensors）を 1 バイトも用意せずに同じ答えが出ることを固定する。実テンソルを
  // 覗く実装へ戻ると、ここが「テンソルが無い」で落ちる。
  const plan = planWeightResidency(parseIrGraph(JSON.stringify(i4Conv1dGraph(1))));
  assertEquals(plan.get("w"), {
    seat: "i4",
    payloadBytes: 128,
    scaleBytes: 64,
    groupSize: 16,
  });
  assertEquals(plan.get("b"), { seat: "raw", payloadBytes: 16 });
});
