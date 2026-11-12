// gelu_tanh and the erf gelu folded into the step ① of the a8 linear that reads them (perf-ledger
// F-4 — the linear prologue of src/runtime/fusion.ts, executed by `buildLinearI8a8`): bit parity
// with the unfused pair on the same backend, the counter, and the dispatch reduction, for both
// integer dot forms.
//
// The fused Session never stores `op(x)`; its quantize reads `x` and applies the op in
// registers. The reference is the same graph with a zero-dispatch reshape between the two nodes:
// the values and the physical kernels stay the production pair, only the exact match stops
// matching.
//
// The twin cases live in @karume/kernel-checks (checkLinearPrologueTwins), which the playground's
// /kernel-check page runs on the browser's backend too; the census of both sides stays here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkLinearPrologueTwins,
  LINEAR_PROLOGUE_FORMS,
  LINEAR_PROLOGUE_SHAPES,
  LINEAR_PROLOGUE_TWIN_CASES,
  linearPrologueActivations,
  linearPrologueGraph,
  linearPrologueLabel,
  linearPrologueWeights,
} from "@karume/kernel-checks";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { linearI8a8Key, linearI8a8UsesVec4 } from "../src/kernels/linear-i8a8.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsPrologueKey,
} from "../src/kernels/quantize-rows.ts";
import { createSessionFromContainer, I8A8_DOT } from "../src/runtime/executor.ts";
import { deviceGeometryProfile, expectedI8a8LinearGeometry } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

test.skipIf(!GPU_AVAILABLE)(
  "the gelu_tanh / erf gelu prologue of an a8 linear matches the unfused pair bit for bit (finite) / by NaN class, counts linearPrologue and drops one dispatch, for dp4a and the emulated dot (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkLinearPrologueTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(LINEAR_PROLOGUE_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the fused a8 linear runs the prologue quantize and the GEMM, the pair runs the op, quantize_rows and the GEMM (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of LINEAR_PROLOGUE_SHAPES) {
        const [rows, k, n] = shape;
        const tensors = linearPrologueWeights(k, n);
        const x = linearPrologueActivations(rows, k);
        for (const [op, dot] of LINEAR_PROLOGUE_FORMS) {
          const label = linearPrologueLabel(op, shape, dot);
          const options = { linearCompute: "a8", [I8A8_DOT]: dot } as const;
          const fused = await createSessionFromContainer(
            gpu,
            await openModelBytes(linearPrologueGraph(op, rows, k, n, false), tensors),
            GRAPH_NAME,
            options,
          );
          const primitive = await createSessionFromContainer(
            gpu,
            await openModelBytes(linearPrologueGraph(op, rows, k, n, true), tensors),
            GRAPH_NAME,
            options,
          );
          try {
            await fused.run({ x });
            await primitive.run({ x });
            const gemm = linearI8a8Key(
              linearI8a8UsesVec4(n),
              dot === "dp4a",
              expectedI8a8LinearGeometry(profile, rows),
            );
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
