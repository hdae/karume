// The gated residual `add(residual, mul(y, where(mask, gate_c, gate_g)))` as one dispatch (fusion
// rule gatedResidual — src/runtime/fusion.ts): bit parity with the unfused primitive sequence on
// the same backend, and the dispatch reduction.
//
// The fused kernel drops the storage round trips of the selected gate and of the product, so a
// rounding point that moved (an fma contraction of `residual + y * g`) would show up directly as a
// bit difference of finite values. Whether the workgroup u32 staging works as a rounding barrier on
// a real backend is visible only here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { GATED_RESIDUAL_KEY } from "../src/kernels/gated-residual.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * `interpose=true` puts a zero-dispatch reshape on the selected gate: the values and the physical
 * dispatches stay the same and only the exact matcher stops matching, so it is the unfused
 * reference on the same backend.
 */
const gatedResidualGraph = (rows: number, dim: number, interpose: boolean): DeclarationJson => {
  const row = [1, rows, dim];
  const vector = [1, 1, dim];
  const values: DeclarationJson["values"] = {
    g: { dtype: "f32", shape: row },
    p: { dtype: "f32", shape: row },
    out: { dtype: "f32", shape: row },
  };
  const nodes: DeclarationJson["nodes"] = [
    { op: "where", ins: ["mask", "gate_c", "gate_g"], outs: ["g"], attrs: {} },
  ];
  if (interpose) {
    values.g_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
  }
  nodes.push(
    { op: "mul", ins: ["y", interpose ? "g_alias" : "g"], outs: ["p"], attrs: {} },
    { op: "add", ins: ["residual", "p"], outs: ["out"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "mask", dtype: "bool", shape: [1, rows, 1] },
      { name: "gate_c", dtype: "f32", shape: vector },
      { name: "gate_g", dtype: "f32", shape: vector },
      { name: "y", dtype: "f32", shape: row },
      { name: "residual", dtype: "f32", shape: row },
    ],
    outputs: ["out"],
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/**
 * Inputs whose product and residual are of similar magnitude with both signs (so a contracted
 * `residual + y * g` would round differently), the two gates differ in value and sign (so picking
 * the wrong side changes the bits), and ±0 / subnormals / ±Inf / NaN payloads sit in fixed slots.
 */
const inputs = (
  rows: number,
  dim: number,
  mask: (row: number) => number,
): Readonly<Record<string, Tensor>> => {
  const gateCondition = fill([1, 1, dim], (i) => Math.cos(i * 0.29) * 1.7 - 0.3);
  const gateGeneration = fill([1, 1, dim], (i) => -Math.sin(i * 0.13) * 2.1 + 0.2);
  const y = fill([1, rows, dim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  const residual = fill([1, rows, dim], (i) => -Math.cos(i * 0.23) * 2.9 + i / 1301);
  words(gateCondition).set([0x80000000, 0x00000001], 0);
  words(gateGeneration).set([0x807fffff, 0x7fc01234], 2);
  words(y).set(
    [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000, 0x7fc01234],
    dim,
  );
  words(residual).set([0x80000000, 0x007fffff, 0xff800000, 0x00000000, 0xff801234], dim + 2);
  return {
    mask: fill([1, rows, 1], mask, "bool"),
    gate_c: gateCondition,
    gate_g: gateGeneration,
    y,
    residual,
  };
};

const MASKS: readonly (readonly [string, (row: number) => number])[] = [
  ["mixed mask", (row) => (row % 3 === 1 ? 0 : 1)],
  ["all condition", () => 1],
  ["all generation", () => 0],
];

test.skipIf(!GPU_AVAILABLE)(
  "gated residual fusion matches the unfused sequence bit for bit (finite) / by NaN class for mixed and uniform masks, and folds 3 dispatches into 1 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // 257 = a row that is not a multiple of the workgroup; 3072 = Wan2.2's width; 8 = rows
      // shorter than one workgroup.
      for (const [rows, dim] of [
        [5, 257],
        [7, 3072],
        [4, 8],
      ] as const) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(gatedResidualGraph(rows, dim, false), []),
          GRAPH_NAME,
        );
        const primitive = await createSessionFromContainer(
          gpu,
          await openModelBytes(gatedResidualGraph(rows, dim, true), []),
          GRAPH_NAME,
        );
        try {
          for (const [maskLabel, mask] of MASKS) {
            const label = `rows=${rows} dim=${dim} ${maskLabel}`;
            assertCheckCase(
              await checkFusionTwin(gpu, {
                label: `${label}: fused vs primitive`,
                fused,
                plain: primitive,
                feeds: inputs(rows, dim, mask),
                counter: "gatedResidual",
                hits: 1,
                nan: "class",
                // One pass against where + mul + add.
                dispatches: { fused: 1, plain: 3 },
              }),
            );
            assertEquals(
              requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
              [GATED_RESIDUAL_KEY],
              `${label}: census`,
            );
          }
        } finally {
          await fused.dispose();
          await primitive.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "gated residual fusion matches the unfused sequence over several grid-stride passes (real GPU)",
  async () => {
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const rows = 9;
    const dim = 300;
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(gatedResidualGraph(rows, dim, false), []),
        GRAPH_NAME,
      );
      const primitive = await createSessionFromContainer(
        gpu,
        await openModelBytes(gatedResidualGraph(rows, dim, true), []),
        GRAPH_NAME,
      );
      try {
        assertCheckCase(
          await checkFusionTwin(gpu, {
            label: "grid-stride",
            fused,
            plain: primitive,
            feeds: inputs(rows, dim, (row) => row % 2),
            counter: "gatedResidual",
            hits: 1,
            nan: "class",
          }),
        );
      } finally {
        await fused.dispose();
        await primitive.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
