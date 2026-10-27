/**
 * The recipe of conv3d's w8a8 path (ADR 0133 — `buildConv3d` with a tap-major i8 weight), without
 * a GPU: the dispatch list the builder queues, read off a {@link StepRecipeBuilder}.
 *
 * The channel-major quantiser is grid-stride and bit-neutral in its workgroup count, so a wrong
 * count (a different geometry's, or one workgroup per 256 pixels) gives the same values on every
 * GPU test and shows only in timing — the small-P starvation the geometry rule exists to avoid.
 * This pins the count to the geometry in the key.
 *
 * The GEMM's tile geometry is the Session's profile's `i8a8.conv3d` (ADR 0115 addendum 2026-10-26).
 * A geometry that only one of key, WGSL and grid followed would give a key that lies about the
 * kernel, or output tiles left unwritten, so the three are checked against an injected profile.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertNotEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import {
  CONV3D_I8A8_QUANTIZE_GEOMETRY,
  conv3dI8a8Key,
  conv3dI8a8UsesVec4,
  conv3dI8a8Wgsl,
} from "../src/kernels/conv3d-i8a8.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import type { I8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
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
const conv3dNode = (
  channelsIn: number,
  input: readonly [number, number, number],
  channelsOut = 8,
): NodePlan => {
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
 * objects are opaque placeholders: the builder only passes them through to the recipe. The pipeline
 * cache records the WGSL of every key it is asked for in `compiled`.
 */
const w8a8Face = (
  limit: number,
  geometryProfile: GeometryProfile = DEFAULT_GEOMETRY_PROFILE,
  compiled: Map<string, string> = new Map(),
): RecipeBuildFace => {
  const opaque = {};
  const face = {
    state: {
      gpu: { limits: { maxComputeWorkgroupsPerDimension: limit } },
      cache: {
        get: async (key: string, wgsl: string) => {
          compiled.set(key, wgsl);
          return { pipeline: opaque, layout: opaque, roles: opaque };
        },
      },
      convCompute: "a8",
      linearI8a8Dot: "dp4a",
      geometryProfile,
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

/** The integer dot-product GEMM (the second dispatch) of one node under `profile`. */
const gemmUnder = async (
  profile: GeometryProfile,
  channelsIn: number,
  input: readonly [number, number, number],
  channelsOut: number,
): Promise<{
  readonly key: string;
  readonly wgsl: string | undefined;
  readonly workgroups: unknown;
}> => {
  const builder = new StepRecipeBuilder();
  const compiled = new Map<string, string>();
  await buildConv3d(
    w8a8Face(65535, profile, compiled),
    conv3dNode(channelsIn, input, channelsOut),
    [
      { kind: "value", name: "x" },
      { kind: "value", name: "w" },
      { kind: "value", name: "b" },
    ],
    [{ kind: "value", name: "y" }],
    builder,
  );
  assertEquals(builder.dispatches.length, 2, "one node = quantiser + GEMM");
  const gemm = builder.dispatches[1];
  return { key: gemm.key, wgsl: compiled.get(gemm.key), workgroups: gemm.workgroups };
};

const profileWith = (conv3d: I8a8Geometry): GeometryProfile => ({
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "conv3d-injected",
  i8a8: { ...DEFAULT_GEOMETRY_PROFILE.i8a8, conv3d },
});

describe("conv3d w8a8 recipe: the GEMM's geometry is the profile's i8a8.conv3d", () => {
  // Cout 200 over P = 3·16·16 = 768 output pixels: every geometry below gives another grid
  const [channelsIn, input, channelsOut] = [64, [3, 16, 16], 200] as const;
  const pixels = input[0] * input[1] * input[2];

  it("key, WGSL and grid all follow the injected geometry (two non-default geometries and the default)", async () => {
    for (const [geometry, grid] of [
      // The default (tile 128×64): [ceil(768 / 64), ceil(200 / 128), 1]
      [DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d, [12, 2, 1]],
      // tile 64×64 (r8×4 wg16×8 — nvidia-ampere's i8a8.linear)
      [{ regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 }, [12, 4, 1]],
      // tile 32×32 (r4×4 wg8×8)
      [{ regM: 4, regN: 4, wgX: 8, wgY: 8, tileK: 16 }, [24, 7, 1]],
    ] as const) {
      const where = JSON.stringify(geometry);
      const v4 = conv3dI8a8UsesVec4(pixels);
      const gemm = await gemmUnder(profileWith(geometry), channelsIn, input, channelsOut);
      assertEquals(gemm.key, conv3dI8a8Key(v4, true, geometry), where);
      assertEquals(gemm.wgsl, conv3dI8a8Wgsl(v4, true, geometry), where);
      assertEquals(gemm.workgroups, grid, where);
    }
  });

  it("DEFAULT_GEOMETRY_PROFILE in the build face gives the default-geometry key a default machine has always had", async () => {
    const gemm = await gemmUnder(DEFAULT_GEOMETRY_PROFILE, channelsIn, input, channelsOut);
    assertEquals(gemm.key, "conv3d:v1:i8a8:tile128x64r8x8w8x16k16v4:dp4a");
  });
});
