/**
 * The recipe of conv3d's w8a8 path (ADR 0133 — `buildConv3d` with a tap-major i8 weight), without
 * a GPU: the dispatch list the builder queues, read off a {@link StepRecipeBuilder}.
 *
 * The channel-major quantiser is grid-stride and bit-neutral in its workgroup count, so a wrong
 * count (a different geometry's, or one workgroup per 256 pixels) gives the same values on every
 * GPU test and shows only in timing — the small-P starvation the geometry rule exists to avoid.
 * This pins the count to the geometry in the key.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertNotEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { CONV3D_I8A8_QUANTIZE_GEOMETRY } from "../src/kernels/conv3d-i8a8.ts";
import {
  QUANTIZE_ROWS_WORKGROUP_SIZE,
  quantizeRowsChannelMajorKey,
  quantizeRowsChannelMajorWorkgroups,
} from "../src/kernels/quantize-rows.ts";
import { bindSymbols, planGraph, type NodePlan } from "../src/runtime/plan.ts";
import { StepRecipeBuilder } from "../src/runtime/recipe.ts";
import type { RecipeBuildFace } from "../src/runtime/recipe-builder.ts";
import { buildConv3d } from "../src/runtime/recipe-builders/conv.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

/** One `conv3d(x, w, b)` node, 1×1×1, `x` = [channelsIn, ...input]. */
const conv3dNode = (channelsIn: number, input: readonly [number, number, number]): NodePlan => {
  const channelsOut = 8;
  const xShape = [channelsIn, ...input];
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["conv3d"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: xShape }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [channelsOut, channelsIn, 1, 1, 1] },
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [channelsOut, ...input] },
    },
    nodes: [
      {
        op: "conv3d",
        ins: ["x", "w", "b"],
        outs: ["y"],
        attrs: { stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1 },
      },
    ],
  };
  const graph = mergeGraph(declaration, { w: "int8-sym" });
  const [node] = planGraph(graph, bindSymbols(graph, { x: xShape })).nodes;
  return node;
};

/**
 * The surface the conv3d builder reads, with the weight flagged tap-major (the w8a8 path). GPU
 * objects are opaque placeholders: the builder only passes them through to the recipe.
 */
const w8a8Face = (limit: number): RecipeBuildFace => {
  const opaque = {};
  const face = {
    state: {
      gpu: { limits: { maxComputeWorkgroupsPerDimension: limit } },
      cache: {
        get: async () => ({ pipeline: opaque, layout: opaque, roles: opaque }),
      },
      convCompute: "a8",
      linearI8a8Dot: "dp4a",
    },
    writeParams: () => opaque,
    weightStorage: () => "i8",
    weightLayout: () => "conv3d-tap-major",
    weightScaleBindings: () => [],
    weightGroupSize: () => 1,
  };
  // MUST: only the fields above are read; a new read reaches `undefined` and fails the test.
  return face as unknown as RecipeBuildFace;
};

const quantiserWorkgroups = async (
  channelsIn: number,
  input: readonly [number, number, number],
  limit: number,
): Promise<{ readonly key: string; readonly workgroups: unknown }> => {
  const builder = new StepRecipeBuilder();
  await buildConv3d(
    w8a8Face(limit),
    conv3dNode(channelsIn, input),
    [
      { kind: "value", name: "x" },
      { kind: "value", name: "w" },
      { kind: "value", name: "b" },
    ],
    [{ kind: "value", name: "y" }],
    builder,
  );
  assertEquals(builder.dispatches.length, 2, "one node = quantiser + GEMM");
  const [quantiser] = builder.dispatches;
  return { key: quantiser.key, workgroups: quantiser.workgroups };
};

describe("conv3d w8a8 recipe: the quantiser's workgroup count", () => {
  it("is quantizeRowsChannelMajorWorkgroups(P, CONV3D_I8A8_QUANTIZE_GEOMETRY, limit), with the geometry of its key", async () => {
    const limit = 65535;
    // P = 768 (the 1024-channel layer at 3×16², the geometry rule's weakest shape) and a P that is
    // not a multiple of the pixel block.
    for (const [channelsIn, input] of [
      [1024, [3, 16, 16]],
      [48, [1, 7, 9]],
    ] as const) {
      const pixels = input[0] * input[1] * input[2];
      const { key, workgroups } = await quantiserWorkgroups(channelsIn, input, limit);
      assertEquals(key, quantizeRowsChannelMajorKey(CONV3D_I8A8_QUANTIZE_GEOMETRY));
      assertEquals(workgroups, [
        quantizeRowsChannelMajorWorkgroups(pixels, CONV3D_I8A8_QUANTIZE_GEOMETRY, limit),
        1,
        1,
      ]);
      // Not one workgroup per 256 pixels (the starvation at small P the rule avoids)
      assertNotEquals(workgroups, [
        gridStrideWorkgroups(pixels, QUANTIZE_ROWS_WORKGROUP_SIZE, limit),
        1,
        1,
      ]);
    }
    // The literal at the production geometry (16 pixels per workgroup): 768 / 16
    assertEquals((await quantiserWorkgroups(1024, [3, 16, 16], limit)).workgroups, [48, 1, 1]);
  });

  it("clamps to the device's per-dimension limit (the kernel grid-strides over the rest)", async () => {
    const { workgroups } = await quantiserWorkgroups(1024, [3, 16, 16], 40);
    assertEquals(workgroups, [40, 1, 1]);
  });
});
