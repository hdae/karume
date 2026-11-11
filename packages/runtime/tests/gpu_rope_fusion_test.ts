// The exact peephole for half-split RoPE (fusion rule rope — src/runtime/fusion.ts): pins bit
// parity with the existing primitive sequence and the dispatch reduction directly.
//
// The twin cases of every form live in @karume/kernel-checks (checkRopeTwins), which the
// playground's /kernel-check page runs on the browser's backend too; the census, the output shapes,
// the fallback cases and the grid-stride runs on a capped device stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkFusionTwin,
  checkRopeTwins,
  ROPE_BSHD_SHAPES,
  ROPE_ORDERS,
  ROPE_PAIR_SHAPES,
  ROPE_TWIN_CASES,
  ropeBshdInputs,
  ropeGraph,
  ropeInputs,
  ropePairGraph,
  ropePairInputs,
  ropePairLabel,
} from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { ROPE_BSHD_KEY, ROPE_KEY, ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

test.skipIf(!GPU_AVAILABLE)(
  "RoPE fusion matches the primitive sequence in finite bits and NaN class — half-split [1,H,S,D] at H = 1 and H ≥ 2 in both orders, [1,S,H,D] across axes, widths and tails, and the pair form — in 1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkRopeTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(ROPE_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). Only the keys show that the fusion needs a single RoPE kernel
 * (the dispatch count alone misses a fusion that turned into one different kernel).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (it works on machines
 * without timing support). The census is checked unconditionally (an empty or undefined one is
 * failed by `requireCensus` in tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "half-split RoPE fusion runs only a single RoPE kernel (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = ropeInputs(1, 3);
    try {
      for (const order of ROPE_ORDERS) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, false), []),
          GRAPH_NAME,
        );
        try {
          await fused.run(inputs);
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, order).map((row) => row.key),
            [ROPE_KEY],
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

// With H=1 alone, the head-direction folding of `row / sequence` is never exercised (row stays
// below sequence, so token = row). The real distribution has H=16, so this is the production index.
test.skipIf(!GPU_AVAILABLE)(
  "RoPE fusion with H≥2 returns the [1,H,S,D] shape (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = ropeInputs(3, 5);
    try {
      for (const order of ROPE_ORDERS) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, false, 3), []),
          GRAPH_NAME,
        );
        try {
          assertEquals((await fused.run(inputs)).y.shape, [1, 3, 5, 128], `${order}: output shape`);
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
  "RoPE does not fuse when an inner value is a graph output or has another consumer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const sequence = 2;
    const inputs = {
      x: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.11) + i / 4093),
      cos: fill([1, 1, sequence, 128], (i) => Math.cos(i * 0.07)),
      sin: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.13)),
    };
    const x = inputs.x;
    try {
      const outputGraph = ropeGraph("slice-first", false);
      outputGraph.outputs = ["first", "y"];
      const outputSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(outputGraph, []),
        GRAPH_NAME,
      );
      try {
        const outputs = await outputSession.run(inputs);
        assertEquals(
          outputSession.diagnostics().submit.dispatchCount,
          8,
          "an inner output causes a fallback",
        );
        assertEquals(outputSession.diagnostics().lastRunFusions?.rope, 0, "fusion counter 0");
        const first = outputs.first.data;
        for (let row = 0; row < sequence; row++) {
          for (let d = 0; d < 64; d++) {
            assertEquals(first[row * 64 + d], x.data[row * 128 + d], `row=${row}:d=${d}`);
          }
        }
      } finally {
        await outputSession.dispose();
      }

      const directOutputGraph = ropeGraph("direct-first", false);
      directOutputGraph.outputs = ["direct", "y"];
      const directOutputSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(directOutputGraph, []),
        GRAPH_NAME,
      );
      try {
        const outputs = await directOutputSession.run(inputs);
        assertEquals(
          directOutputSession.diagnostics().submit.dispatchCount,
          8,
          "a leading inner output of direct-first causes a fallback",
        );
        assertEquals(directOutputSession.diagnostics().lastRunFusions?.rope, 0, "fusion counter 0");
        const direct = outputs.direct.data;
        for (let i = 0; i < direct.length; i++) {
          assertEquals(direct[i], Math.fround(x.data[i] * inputs.cos.data[i]), `direct i=${i}`);
        }
      } finally {
        await directOutputSession.dispose();
      }

      const consumerGraph = ropeGraph("slice-first", false);
      consumerGraph.values.first_copy = { dtype: "f32", shape: [1, 1, "S", 64] };
      consumerGraph.nodes.push({
        op: "neg",
        ins: ["first"],
        outs: ["first_copy"],
        attrs: {},
      });
      consumerGraph.outputs = ["first_copy", "y"];
      const consumerSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(consumerGraph, []),
        GRAPH_NAME,
      );
      try {
        const outputs = await consumerSession.run(inputs);
        assertEquals(
          consumerSession.diagnostics().submit.dispatchCount,
          9,
          "an inner value with use count 2 falls back to the 8-dispatch sequence + the consumer",
        );
        assertEquals(consumerSession.diagnostics().lastRunFusions?.rope, 0, "fusion counter 0");
        const firstCopy = outputs.first_copy.data;
        for (let row = 0; row < sequence; row++) {
          for (let d = 0; d < 64; d++) {
            assertEquals(
              firstCopy[row * 64 + d],
              -x.data[row * 128 + d],
              `consumer row=${row}:d=${d}`,
            );
          }
        }
      } finally {
        await consumerSession.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "BSHD RoPE returns the [1,S,H,D] shape across the token/head axes, partial workgroups and widths (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, sequence, dim] of ROPE_BSHD_SHAPES) {
        const inputs = ropeBshdInputs(heads, sequence, dim);
        for (const order of ROPE_ORDERS) {
          const fused = await createSessionFromContainer(
            gpu,
            await openModelBytes(ropeGraph(order, false, heads, dim, "bshd"), []),
            GRAPH_NAME,
          );
          try {
            assertEquals((await fused.run(inputs)).y.shape, [1, sequence, heads, dim]);
          } finally {
            await fused.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "BSHD RoPE is bit-identical to the primitive sequence across multiple grid-stride passes too (real GPU)",
  async () => {
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const heads = 8,
      sequence = 17,
      dim = 128;
    const inputs = {
      x: fill([1, sequence, heads, dim], (i) => Math.sin(i * 0.013)),
      cos: fill([1, sequence, 1, dim], (i) => Math.cos(i * 0.017)),
      sin: fill([1, sequence, 1, dim], (i) => Math.sin(i * 0.019)),
    };
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(ropeGraph("direct-first", false, heads, dim, "bshd"), []),
        GRAPH_NAME,
      );
      const split = await createSessionFromContainer(
        gpu,
        await openModelBytes(ropeGraph("direct-first", true, heads, dim, "bshd"), []),
        GRAPH_NAME,
      );
      try {
        assertCheckCase(
          await checkFusionTwin(gpu, {
            label: "BSHD grid-stride",
            fused,
            plain: split,
            feeds: inputs,
            counter: "rope",
            hits: 1,
            nan: "class",
          }),
        );
      } finally {
        await fused.dispose();
        await split.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)("BSHD RoPE runs only its dedicated keys (real GPU)", async () => {
  const gpu = await acquireGpu();
  try {
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(ropeGraph("direct-first", false, 3, 128, "bshd"), []),
      GRAPH_NAME,
    );
    try {
      await session.run({
        x: fill([1, 5, 3, 128], (i) => Math.sin(i)),
        cos: fill([1, 5, 1, 128], (i) => Math.cos(i)),
        sin: fill([1, 5, 1, 128], (i) => Math.sin(i)),
      });
      assertEquals(
        requireCensus(session.diagnostics().lastRunPipelines, "bshd").map((row) => row.key),
        [ROPE_BSHD_KEY],
      );
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
});

// ------------------------------------------------------------ pair form (Wan's DiT)

test.skipIf(!GPU_AVAILABLE)(
  "pair-form RoPE fusion returns the [1,S,H,D] shape and runs only the pair RoPE kernel across heads, widths and tails (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, sequence, headDim, separatePairs] of ROPE_PAIR_SHAPES) {
        const label = ropePairLabel(heads, sequence, headDim, separatePairs);
        const inputs = ropePairInputs(heads, sequence, headDim, separatePairs);
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropePairGraph(heads, headDim, false, separatePairs), []),
          GRAPH_NAME,
        );
        try {
          assertEquals((await fused.run(inputs)).y.shape, [1, sequence, heads, headDim], label);
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
            [ROPE_PAIR_BSHD_KEY],
            `${label}: census`,
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
  "pair-form RoPE matches the unfused sequence over several grid-stride passes (real GPU)",
  async () => {
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const heads = 8;
    const sequence = 17;
    const headDim = 128;
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const inputs = ropePairInputs(heads, sequence, headDim, false);
      const fused = await createSessionFromContainer(
        gpu,
        await openModelBytes(ropePairGraph(heads, headDim, false), []),
        GRAPH_NAME,
      );
      const split = await createSessionFromContainer(
        gpu,
        await openModelBytes(ropePairGraph(heads, headDim, true), []),
        GRAPH_NAME,
      );
      try {
        assertCheckCase(
          await checkFusionTwin(gpu, {
            label: "pair grid-stride",
            fused,
            plain: split,
            feeds: inputs,
            counter: "rope",
            hits: 1,
            nan: "class",
          }),
        );
      } finally {
        await fused.dispose();
        await split.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
