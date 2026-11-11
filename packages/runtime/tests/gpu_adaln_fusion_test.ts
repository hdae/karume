// adaLN = the strict peephole for `layer_norm → (1 + scale) → mul → add` (fusion rule adaln —
// src/runtime/fusion.ts). Only the **non-adjacent windows of 6 / 7 nodes** that the exporter emits
// are folded into 1 dispatch; a shape it cannot match falls back to the plain sequence.
//
// This check looks at one thing: bit parity of the values. The fused kernel removes the storage
// round trips of the 3 intermediates (normalized result / `1 + scale` / product), so if its
// rounding points drift from the plain sequence the bits of finite values move. Whether the
// rounding barrier of the workgroup u32 staging works on real backends can only be seen here.
//
// The twin cases of both forms live in @karume/kernel-checks (checkAdalnTwins), which the
// playground's /kernel-check page runs on the browser's backend too; the census, the fallback
// counterexamples and the grid-stride run on a capped device stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  ADALN_GATES,
  ADALN_TOKEN_MASKS,
  ADALN_TOKEN_SHAPES,
  ADALN_TWIN_CASES,
  adalnGraph,
  type AdalnGraphOptions,
  adalnInputs,
  adalnTokenGraph,
  adalnTokenInputs,
  checkAdalnTwins,
  checkFusionTwin,
} from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
} from "../src/kernels/adaln-norm.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

test.skipIf(!GPU_AVAILABLE)(
  "adaLN fusion matches primitives in finite bits / NaN class for both windows 6 / 7 (4→1 dispatch) and the per-token form for mixed and uniform masks (6→2 dispatches) (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkAdalnTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(ADALN_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "adaLN fusion runs only the adaLN kernel in both windows (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const gate of ADALN_GATES) {
        const label = `window ${gate ? 7 : 6}`;
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnGraph({ gate }), []),
          GRAPH_NAME,
        );
        try {
          await fused.run(adalnInputs({ gate }));
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
            [ADALN_NORM_KEY],
            `${label}: census key`,
          );
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
  "adaLN with an internal output / another consumer / an alias falls back from the strict matcher (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = adalnInputs();
    const cases: readonly { readonly name: string; readonly options: AdalnGraphOptions }[] = [
      { name: "internal output", options: { normOutput: true } },
      { name: "extra consumer", options: { modulationExtraConsumer: true } },
      { name: "interposed alias", options: { interpose: true } },
    ];
    try {
      // Every counterexample has the shape "values stay correct, dispatches go back to 4". Check
      // down to bit equality with the reference (fused) y.
      const reference = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnGraph(), []),
        GRAPH_NAME,
      );
      try {
        for (const testCase of cases) {
          const session = await createSessionFromContainer(
            gpu,
            await openModelBytes(adalnGraph(testCase.options), []),
            GRAPH_NAME,
          );
          try {
            // The counterexample is the plain side: it falls back, so its counter stays 0.
            assertCheckCase(
              await checkFusionTwin(gpu, {
                label: `${testCase.name}: fallback values`,
                fused: reference,
                plain: session,
                feeds: inputs,
                counter: "adaln",
                hits: 1,
                nan: "class",
                outputs: ["y"],
                dispatches: { plain: testCase.name === "extra consumer" ? 5 : 4 },
              }),
            );
          } finally {
            await session.dispose();
          }
        }
      } finally {
        await reference.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ------------------------------------------------------------ per-token form (Wan2.2's DiT)

test.skipIf(!GPU_AVAILABLE)(
  "per-token adaLN fusion runs only the modulation pack and the per-token body (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [rows, dim] of ADALN_TOKEN_SHAPES) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnTokenGraph(rows, dim, false), []),
          GRAPH_NAME,
        );
        try {
          for (const [maskLabel, mask] of ADALN_TOKEN_MASKS) {
            const label = `rows=${rows} dim=${dim} ${maskLabel}`;
            await fused.run(adalnTokenInputs(rows, dim, mask));
            assertEquals(
              requireCensus(fused.diagnostics().lastRunPipelines, label)
                .map((row) => row.key)
                .sort(),
              [ADALN_MODULATION_PACK_KEY, ADALN_NORM_TOKEN_KEY].sort(),
              `${label}: census keys`,
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
  "per-token adaLN matches the unfused sequence when both the pack and the rows run several grid-stride passes (real GPU)",
  async () => {
    // Two workgroups per dispatch: the pack covers 1,100 columns in three passes and the body
    // covers 5 rows in three passes.
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const rows = 5;
    const dim = 1100;
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnTokenGraph(rows, dim, false), []),
        GRAPH_NAME,
      );
      const primitive = await createSessionFromContainer(
        gpu,
        await openModelBytes(adalnTokenGraph(rows, dim, true), []),
        GRAPH_NAME,
      );
      try {
        assertCheckCase(
          await checkFusionTwin(gpu, {
            label: "grid-stride",
            fused,
            plain: primitive,
            feeds: adalnTokenInputs(rows, dim, (row) => row % 2),
            counter: "adaln",
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
