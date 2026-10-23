// conv3d の w8a8 の重みの並べ替え（ADR 0133 決定 4）の GPU を要らない面:
// ① 並べ替える重みの集合（`conv3dTapMajorInitializers` — グラフだけから決まる）
// ② バイト列の並べ替え（`reorderConv3dWeightTapMajor` — 行ごとの置換・piece 単位で同じ）
//
// MUST: 集合の判定は「i8 格納 × 消費が全て適格な conv3d の weight slot × グラフ出力でない」。
// 1 つでも外れる消費があれば並べ替えない（f32 カーネルが置換済みのバイト列を読む誤値を生まない）。

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { reorderConv3dWeightTapMajor } from "../src/kernels/conv3d-i8a8.ts";
import { conv3dTapMajorInitializers } from "../src/runtime/plan.ts";

const CONV_ATTRS = { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups: 1 };

/** 検査に要る欄だけを持つグラフ（契約検査は別層 — gpu_f16_i8_weights_test.ts と同じ流儀）。 */
const graphOf = (
  nodes: readonly { op: string; ins: string[]; outs: string[]; attrs: Record<string, unknown> }[],
  storage: Readonly<Record<string, string>>,
  shapes: Readonly<Record<string, number[]>>,
  outputs: string[] = ["y"],
) =>
  ({
    initializers: Object.fromEntries(
      Object.entries(storage).map(([name, codec]) => [name, { storage: { codec } }]),
    ),
    values: Object.fromEntries(
      Object.entries(shapes).map(([name, shape]) => [name, { dtype: "f32", shape }]),
    ),
    nodes,
    outputs,
  }) as never;

const W_SHAPE = [6, 8, 3, 3, 3];

test("conv3dTapMajorInitializers: i8 の重みを weight slot だけで消費する適格な conv3d の重みを選ぶ", () => {
  const graph = graphOf(
    [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS }],
    { w: "int8-sym", b: "f32" },
    { w: W_SHAPE, b: [6] },
  );
  assertEquals([...conv3dTapMajorInitializers(graph)], ["w"]);
});

test("conv3dTapMajorInitializers: f16 格納・Cin % 4 ≠ 0・groups > 1・他の消費・グラフ出力は外れる", () => {
  const cases: readonly [string, Parameters<typeof conv3dTapMajorInitializers>[0]][] = [
    [
      "f16 格納",
      graphOf(
        [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS }],
        { w: "f16", b: "f32" },
        { w: W_SHAPE, b: [6] },
      ),
    ],
    [
      "Cin % 4 ≠ 0",
      graphOf(
        [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS }],
        { w: "int8-sym", b: "f32" },
        { w: [6, 6, 3, 3, 3], b: [6] },
      ),
    ],
    [
      "groups > 1",
      graphOf(
        [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: { ...CONV_ATTRS, groups: 2 } }],
        { w: "int8-sym", b: "f32" },
        { w: [6, 4, 3, 3, 3], b: [6] },
      ),
    ],
    [
      "weight slot 以外の消費が 1 つある",
      graphOf(
        [
          { op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS },
          { op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} },
        ],
        { w: "int8-sym", b: "f32" },
        { w: W_SHAPE, b: [6] },
      ),
    ],
    [
      "グラフ出力に並ぶ",
      graphOf(
        [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS }],
        { w: "int8-sym", b: "f32" },
        { w: W_SHAPE, b: [6] },
        ["y", "w"],
      ),
    ],
  ];
  for (const [name, graph] of cases) {
    assertEquals([...conv3dTapMajorInitializers(graph)], [], name);
  }
});

test("conv3dTapMajorInitializers: 2 本の適格な conv3d が同じ重みを使えば並べ替える（1 本でも不適格なら外す）", () => {
  const both = graphOf(
    [
      { op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS },
      { op: "conv3d", ins: ["y", "w", "b"], outs: ["z"], attrs: CONV_ATTRS },
    ],
    { w: "int8-sym", b: "f32" },
    { w: [8, 8, 3, 3, 3], b: [8] },
    ["z"],
  );
  assertEquals([...conv3dTapMajorInitializers(both)], ["w"]);
  const mixed = graphOf(
    [
      { op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS },
      { op: "conv3d", ins: ["y", "w", "b"], outs: ["z"], attrs: { ...CONV_ATTRS, groups: 2 } },
    ],
    { w: "int8-sym", b: "f32" },
    { w: [8, 8, 3, 3, 3], b: [8] },
    ["z"],
  );
  assertEquals([...conv3dTapMajorInitializers(mixed)], []);
});

test("reorderConv3dWeightTapMajor: [row][Cin][taps] → [row][taps][Cin] を行ごとに置換する（piece 単位で同じ）", () => {
  const rows = 2;
  const channelsIn = 4;
  const taps = 3;
  // 値 = row·100 + ci·10 + tap（元の位置が読める）
  const bytes = new Uint8Array(rows * channelsIn * taps);
  for (let row = 0; row < rows; row += 1) {
    for (let ci = 0; ci < channelsIn; ci += 1) {
      for (let tap = 0; tap < taps; tap += 1) {
        bytes[(row * channelsIn + ci) * taps + tap] = row * 100 + ci * 10 + tap;
      }
    }
  }
  const whole = reorderConv3dWeightTapMajor(bytes, rows, channelsIn, taps);
  for (let row = 0; row < rows; row += 1) {
    for (let tap = 0; tap < taps; tap += 1) {
      for (let ci = 0; ci < channelsIn; ci += 1) {
        assertEquals(whole[(row * taps + tap) * channelsIn + ci], row * 100 + ci * 10 + tap);
      }
    }
  }
  // 行 1 だけの piece を並べ替えると、全体の行 1 の部分と一致する
  const perRow = channelsIn * taps;
  const piece = reorderConv3dWeightTapMajor(bytes.slice(perRow, 2 * perRow), 1, channelsIn, taps);
  assertEquals([...piece], [...whole.slice(perRow, 2 * perRow)]);
  // 元の配列は触らない
  assertEquals(bytes[1], 1);
  assertThrows(() => reorderConv3dWeightTapMajor(bytes, 3, channelsIn, taps), CodegenError);
});
