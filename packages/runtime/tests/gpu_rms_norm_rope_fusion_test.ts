// The fusion rule rmsNormRope (src/runtime/fusion-rules/rms-norm-rope.ts) in a Session: the Wan
// DiT's interleaved q / k prologues — rms_norm → two reshapes → the pair-form RoPE, with k's nodes
// deferred behind q's fused step — against the same graph with a zero-dispatch reshape after each
// rms_norm, which breaks the match and leaves the plain path (rms_norm, then the pair RoPE folded
// alone by the rope rule) on the same backend. The outputs are compared as u32 words, with the
// counters, the dispatch counts and the census. No adapter = explicit SKIP.
//
// The twin cases live in @karume/kernel-checks (checkRmsNormRopeTwins), which the playground's
// /kernel-check page runs on the browser's backend too; the shapes, the rope counters, the census
// and the grid-stride run on a capped device stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkFusionTwin,
  checkRmsNormRopeTwins,
  RMS_NORM_ROPE_OUTPUTS,
  RMS_NORM_ROPE_SHAPES,
  RMS_NORM_ROPE_TWIN_CASES,
  rmsNormRopeInputs,
} from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { rmsNormWorkgroupKey } from "../src/kernels/rms-norm.ts";
import { RMS_NORM_ROPE_KEY } from "../src/kernels/rms-norm-rope.ts";
import { ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { countDispatches, requireCensus } from "./helpers/pipeline-census.ts";
import { rmsNormRopeGraph } from "./helpers/rms-norm-rope-graph.ts";

test.skipIf(!GPU_AVAILABLE)(
  "rmsNormRope folds the interleaved q / k prologues into one dispatch each, bit-identical to rms_norm → pair RoPE, at the Wan shapes and with a partial last chunk (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkRmsNormRopeTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(RMS_NORM_ROPE_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rmsNormRope keeps the output shapes, leaves no pair RoPE of its own on the fused side and runs the plain path's kernels on the reference (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, headDim, sequence] of RMS_NORM_ROPE_SHAPES) {
        const label = `H=${heads} D=${headDim} S=${sequence}`;
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(rmsNormRopeGraph({ heads, headDim }), []),
          GRAPH_NAME,
        );
        const plain = await createSessionFromContainer(
          gpu,
          await openModelBytes(rmsNormRopeGraph({ heads, headDim, interpose: true }), []),
          GRAPH_NAME,
        );
        try {
          const given = rmsNormRopeInputs(heads, headDim, sequence);
          const actual = await fused.run(given);
          const expected = await plain.run(given);
          for (const name of RMS_NORM_ROPE_OUTPUTS) {
            assertEquals(actual[name].shape, expected[name].shape, `${label}: ${name} shape`);
          }
          const fusedRun = fused.diagnostics();
          const plainRun = plain.diagnostics();
          assertEquals(fusedRun.lastRunFusions?.rope, 0, `${label}: fused counters`);
          assertEquals(plainRun.lastRunFusions?.rope, 2, `${label}: plain counters`);
          const census = requireCensus(fusedRun.lastRunPipelines, label);
          assertEquals(countDispatches(census, RMS_NORM_ROPE_KEY), 2, `${label}: census`);
          assertEquals(
            countDispatches(census, rmsNormWorkgroupKey(heads * headDim)),
            0,
            `${label}: no plain rms_norm`,
          );
          assertEquals(countDispatches(census, ROPE_PAIR_BSHD_KEY), 0, `${label}: no plain RoPE`);
          const plainCensus = requireCensus(plainRun.lastRunPipelines, label);
          assertEquals(
            countDispatches(plainCensus, rmsNormWorkgroupKey(heads * headDim)),
            2,
            `${label}: plain census`,
          );
          assertEquals(
            countDispatches(plainCensus, ROPE_PAIR_BSHD_KEY),
            2,
            `${label}: plain census`,
          );
        } finally {
          await fused.dispose();
          await plain.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rmsNormRope matches the plain path over several grid-stride passes of its rows (real GPU)",
  async () => {
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const heads = 12;
    const headDim = 128;
    const sequence = 17;
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(rmsNormRopeGraph({ heads, headDim }), []),
        GRAPH_NAME,
      );
      const plain = await createSessionFromContainer(
        gpu,
        await openModelBytes(rmsNormRopeGraph({ heads, headDim, interpose: true }), []),
        GRAPH_NAME,
      );
      try {
        assertCheckCase(
          await checkFusionTwin(gpu, {
            label: "grid-stride",
            fused,
            plain,
            feeds: rmsNormRopeInputs(heads, headDim, sequence),
            counter: "rmsNormRope",
            hits: 2,
            nan: "payload",
            outputs: RMS_NORM_ROPE_OUTPUTS,
          }),
        );
      } finally {
        await fused.dispose();
        await plain.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
