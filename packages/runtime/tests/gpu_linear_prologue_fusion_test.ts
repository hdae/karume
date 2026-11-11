// gelu_tanh and the erf gelu folded into the step ① of the a8 linear that reads them (perf-ledger
// F-4 — the linear prologue of src/runtime/fusion.ts, executed by `buildLinearI8a8`): bit parity
// with the unfused pair on the same backend, the counter, and the dispatch reduction, for both
// integer dot forms.
//
// The fused Session never stores `op(x)`; its quantize reads `x` and applies the op in
// registers. The reference is the same graph with a zero-dispatch reshape between the two nodes:
// the values and the physical kernels stay the production pair, only the exact match stops
// matching.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { checkFusionTwin } from "@karume/kernel-checks";
import { elementwiseKey, type PrologueUnaryOp } from "../src/codegen/elementwise.ts";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { linearI8a8Key, linearI8a8UsesVec4 } from "../src/kernels/linear-i8a8.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsPrologueKey,
} from "../src/kernels/quantize-rows.ts";
import {
  createSessionFromContainer,
  I8A8_DOT,
  type I8a8Dot,
  type Tensor,
} from "../src/runtime/executor.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * `y = linear(op(x), w, b)` with `w` resident as i8. `interpose = true` puts a zero-dispatch
 * reshape on the op's output (the unfused reference on the same backend).
 */
const geluLinearGraph = (
  op: PrologueUnaryOp,
  rows: number,
  k: number,
  n: number,
  interpose: boolean,
): DeclarationJson => {
  const row = [1, rows, k];
  const values: DeclarationJson["values"] = {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    g: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: [1, rows, n] },
  };
  const nodes: DeclarationJson["nodes"] = [{ op, ins: ["x"], outs: ["g"], attrs: {} }];
  if (interpose) {
    values.g_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
  }
  nodes.push({
    op: "linear",
    ins: [interpose ? "g_alias" : "g", "w", "b"],
    outs: ["y"],
    attrs: {},
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: row }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values,
    nodes,
  };
};

/** The i8 weight (per-output-channel scales of different sizes) and the bias. */
const weightTensors = (k: number, n: number) => {
  const weight = fill([n, k], (i) => Math.sin(i * 0.61) * (1 + (Math.floor(i / k) % 5) * 0.5));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], (i) => Math.cos(i * 0.7) * 0.5);
  return [
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    },
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int8-sym",
        groupSize: perChannelGroupSize(k),
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    },
  ] as const;
};

/**
 * Activations of the DiT's pre-gelu magnitude (both signs, up to |x| ≈ 14.7: past the point where
 * both ops saturate to x and to −0), one all-zero row and one row with a NaN (its output row is NaN
 * in both forms).
 */
const activations = (rows: number, k: number): Tensor => {
  const x = fill([1, rows, k], (i) => Math.sin(i * 0.37) * 3.1 + ((i % 101) - 50) / 4.3);
  x.data.fill(0, k, 2 * k);
  if (rows > 2) x.data[2 * k + 5] = Number.NaN;
  return x;
};

/**
 * (rows, K, N): a remainder row length, Wan2.2's text embedder (M 512 → 300 here), DeBERTa's
 * intermediate (K 4096), anima's FFN down projection (K 8192), Wan2.2's FFN down projection.
 */
const SHAPES: readonly (readonly [number, number, number])[] = [
  [7, 1028, 68],
  [300, 3072, 16],
  [11, 4096, 24],
  [9, 8192, 16],
  [5, 14336, 8],
];

/** The ops the fusion pass folds into the linear's step ① (`LINEAR_PROLOGUE_OPS`). */
const OPS = ["gelu_tanh", "gelu"] as const satisfies readonly PrologueUnaryOp[];

test.skipIf(!GPU_AVAILABLE)(
  "the gelu_tanh / erf gelu prologue of an a8 linear matches the unfused pair bit for bit (finite) / by NaN class, counts linearPrologue and drops one dispatch, for dp4a and the emulated dot (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const [rows, k, n] of SHAPES) {
        const tensors = weightTensors(k, n);
        const x = activations(rows, k);
        for (const [op, dot] of OPS.flatMap((op) =>
          (["dp4a", "emu"] satisfies I8a8Dot[]).map((dot) => [op, dot] as const),
        )) {
          const label = `${op} rows=${rows} K=${k} N=${n} ${dot}`;
          const options = { linearCompute: "a8", [I8A8_DOT]: dot } as const;
          const fused = await createSessionFromContainer(
            gpu,
            await openModelBytes(geluLinearGraph(op, rows, k, n, false), tensors),
            GRAPH_NAME,
            options,
          );
          const primitive = await createSessionFromContainer(
            gpu,
            await openModelBytes(geluLinearGraph(op, rows, k, n, true), tensors),
            GRAPH_NAME,
            options,
          );
          try {
            assertCheckCase(
              await checkFusionTwin(gpu, {
                label: `${label}: fused vs pair`,
                fused,
                plain: primitive,
                feeds: { x },
                counter: "linearPrologue",
                hits: 1,
                nan: "class",
                // The prologue quantize + the GEMM against the op + quantize_rows + the GEMM.
                dispatches: { fused: 2, plain: 3 },
              }),
            );
            const gemm = linearI8a8Key(linearI8a8UsesVec4(n), dot === "dp4a", profile.i8a8.linear);
            assertEquals(
              requireCensus(fused.diagnostics().lastRunPipelines, label)
                .map((row) => row.key)
                .sort(),
              [gemm, quantizeRowsPrologueKey(op, k)].sort(),
              `${label}: fused census`,
            );
            assertEquals(
              requireCensus(primitive.diagnostics().lastRunPipelines, label)
                .map((row) => row.key)
                .sort(),
              [
                gemm,
                quantizeRowsKey(quantizeRowsGeometry(k)),
                elementwiseKey({ op, rank: 3, dtype: "f32" }),
              ].sort(),
              `${label}: reference census`,
            );
          } finally {
            await fused.dispose();
            await primitive.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
