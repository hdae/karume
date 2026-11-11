// The gated residual `add(residual, mul(y, where(mask, gate_c, gate_g)))` as one dispatch (fusion
// rule gatedResidual — src/runtime/fusion.ts): bit parity with the unfused primitive sequence on
// the same backend, and the dispatch reduction.
//
// The fused kernel drops the storage round trips of the selected gate and of the product, so a
// rounding point that moved (an fma contraction of `residual + y * g`) would show up directly as a
// bit difference of finite values. Whether the workgroup u32 staging works as a rounding barrier on
// a real backend is visible only here.
//
// The twin cases live in @karume/kernel-checks (checkGatedResidualTwins), which the playground's
// /kernel-check page runs on the browser's backend too; the census and the grid-stride run on a
// capped device stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkFusionTwin,
  checkGatedResidualTwins,
  GATED_RESIDUAL_MASKS,
  GATED_RESIDUAL_SHAPES,
  GATED_RESIDUAL_TWIN_CASES,
  gatedResidualGraph,
  gatedResidualInputs,
} from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { GATED_RESIDUAL_KEY } from "../src/kernels/gated-residual.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

test.skipIf(!GPU_AVAILABLE)(
  "gated residual fusion matches the unfused sequence bit for bit (finite) / by NaN class for mixed and uniform masks, and folds 3 dispatches into 1 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkGatedResidualTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(GATED_RESIDUAL_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "gated residual fusion runs only the gated residual kernel (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [rows, dim] of GATED_RESIDUAL_SHAPES) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(gatedResidualGraph(rows, dim, false), []),
          GRAPH_NAME,
        );
        try {
          for (const [maskLabel, mask] of GATED_RESIDUAL_MASKS) {
            const label = `rows=${rows} dim=${dim} ${maskLabel}`;
            await fused.run(gatedResidualInputs(rows, dim, mask));
            assertEquals(
              requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
              [GATED_RESIDUAL_KEY],
              `${label}: census`,
            );
          }
        } finally {
          await fused.dispose();
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
            feeds: gatedResidualInputs(rows, dim, (row) => row % 2),
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
