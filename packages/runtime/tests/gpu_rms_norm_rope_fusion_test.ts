// The fusion rule rmsNormRope (src/runtime/fusion-rules/rms-norm-rope.ts) in a Session: the Wan
// DiT's interleaved q / k prologues — rms_norm → two reshapes → the pair-form RoPE, with k's nodes
// deferred behind q's fused step — against the same graph with a zero-dispatch reshape after each
// rms_norm, which breaks the match and leaves the plain path (rms_norm, then the pair RoPE folded
// alone by the rope rule) on the same backend. The outputs are compared as u32 words, with the
// counters, the dispatch counts and the census. No adapter = explicit SKIP.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { rmsNormWorkgroupKey } from "../src/kernels/rms-norm.ts";
import { RMS_NORM_ROPE_KEY } from "../src/kernels/rms-norm-rope.ts";
import { ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { countDispatches, requireCensus } from "./helpers/pipeline-census.ts";
import { rmsNormRopeGraph } from "./helpers/rms-norm-rope-graph.ts";

/**
 * The graph inputs: q / k / v rows, the two norm weights and the tables, with special values in
 * q's first row (±0, a subnormal, 2e19, ±inf, a NaN payload) and an all-zero second row of k.
 */
const inputs = (
  heads: number,
  headDim: number,
  sequence: number,
): Readonly<Record<string, Tensor>> => {
  const dim = heads * headDim;
  const row = [1, sequence, dim];
  const table = [1, sequence, 1, headDim];
  const q = fill(row, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  new Uint32Array(q.data.buffer).set([
    0x00000000, 0x80000000, 0x00000001, 0x5f0ac723, 0x7f800000, 0xff800000, 0x7fc01234,
  ]);
  const k = fill(row, (i) => Math.cos(i * 0.29) * 2.3 - i / 1531);
  if (sequence > 1) k.data.fill(0, dim, 2 * dim);
  return {
    q_x: q,
    k_x: k,
    v_x: fill(row, (i) => Math.sin(i * 0.11) - 0.5),
    q_w: fill([dim], (i) => 0.2 + Math.cos(i * 0.17)),
    k_w: fill([dim], (i) => 1.1 - Math.sin(i * 0.13)),
    cos: fill(table, (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill(table, (i) => Math.sin(i * 0.23 - 0.2)),
  };
};

const OUTPUTS = ["q_y", "k_y", "v_out"] as const;

test.skipIf(!GPU_AVAILABLE)(
  "rmsNormRope folds the interleaved q / k prologues into one dispatch each, bit-identical to rms_norm → pair RoPE, at the Wan shapes and with a partial last chunk (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Wan2.2 (24 × 128), Wan2.1 (12 × 128), and a row of 384 (its second chunk half full).
      for (const [heads, headDim, sequence] of [
        [24, 128, 7],
        [12, 128, 5],
        [3, 128, 1],
      ] as const) {
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
          const given = inputs(heads, headDim, sequence);
          assertCheckCase(
            await checkFusionTwin(gpu, {
              label,
              fused,
              plain,
              feeds: given,
              counter: "rmsNormRope",
              hits: 2,
              nan: "payload",
              outputs: OUTPUTS,
              // q and k one dispatch each, plus v's neg; the plain path runs both rms_norms and
              // both pair RoPEs (the reshapes are zero-dispatch aliases).
              dispatches: { fused: 3, plain: 5 },
            }),
          );
          // The shapes, from a run of each side of its own (the harness returns a case).
          const actual = await fused.run(given);
          const expected = await plain.run(given);
          for (const name of OUTPUTS) {
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
            feeds: inputs(heads, headDim, sequence),
            counter: "rmsNormRope",
            hits: 2,
            nan: "payload",
            outputs: OUTPUTS,
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
