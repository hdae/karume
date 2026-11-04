// The GPU-free side of reordering conv3d w8a8 weights (ADR 0133 decision 4): ① the set of weights
// to reorder (`conv3dTapMajorInitializers` — determined by the graph alone) ② reordering the bytes
// (`reorderConv3dWeightTapMajor` — a per-row permutation, the same per piece)
//
// MUST: the set is "i8 storage × every consumer is the weight slot of an eligible conv3d × not a
// graph output". If even one consumer falls outside, the weight is not reordered (so an f32 kernel
// never reads permuted bytes and produces wrong values).

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { reorderConv3dWeightTapMajor } from "../src/kernels/conv3d-i8a8.ts";
import { conv3dTapMajorInitializers } from "../src/runtime/plan.ts";

const CONV_ATTRS = { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups: 1 };

/**
 * A graph with only the fields the check needs (contract checks are a separate layer — the same
 * approach as gpu_f16_i8_weights_test.ts).
 */
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

test("conv3dTapMajorInitializers: selects i8 weights of eligible conv3d consumed only through the weight slot", () => {
  const graph = graphOf(
    [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: CONV_ATTRS }],
    { w: "int8-sym", b: "f32" },
    { w: W_SHAPE, b: [6] },
  );
  assertEquals([...conv3dTapMajorInitializers(graph)], ["w"]);
});

test("conv3dTapMajorInitializers: f16 storage, Cin % 4 ≠ 0, groups > 1, other consumers and graph outputs are excluded", () => {
  const cases: readonly [string, Parameters<typeof conv3dTapMajorInitializers>[0]][] = [
    [
      "f16 storage",
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
      "one consumer other than the weight slot",
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
      "listed as a graph output",
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

test("conv3dTapMajorInitializers: reorders a weight shared by two eligible conv3d (excluded if even one is ineligible)", () => {
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

test("reorderConv3dWeightTapMajor: permutes [row][Cin][taps] → [row][taps][Cin] per row (the same per piece)", () => {
  const rows = 2;
  const channelsIn = 4;
  const taps = 3;
  // value = row·100 + ci·10 + tap (the original position can be read off)
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
  // Reordering a piece holding only row 1 matches the row-1 part of the whole
  const perRow = channelsIn * taps;
  const piece = reorderConv3dWeightTapMajor(bytes.slice(perRow, 2 * perRow), 1, channelsIn, taps);
  assertEquals([...piece], [...whole.slice(perRow, 2 * perRow)]);
  // The original array is not modified
  assertEquals(bytes[1], 1);
  assertThrows(() => reorderConv3dWeightTapMajor(bytes, 3, channelsIn, taps), CodegenError);
});
