// The exact peephole for half-split RoPE (fusion rule rope — src/runtime/fusion.ts): pins bit
// parity with the existing primitive sequence and the dispatch reduction directly.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { ROPE_BSHD_KEY, ROPE_KEY, ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

// With H=1, S=3, n=384, which also exercises a partial trailing 256-thread workgroup.
type RopeOrder = "slice-first" | "direct-first";

/**
 * `interpose=true` inserts a zero-dispatch reshape right after the first slice, which breaks only
 * the exact matcher without changing the expression. This lets the existing 8-dispatch sequence on
 * the same backend serve as the reference for a direct comparison.
 *
 * `heads` is the H axis of x. The kernel folds the head direction by dividing `row = i / head_dim`
 * by `sequence` to get the token, so that index is never exercised unless H≥2 runs.
 */
const ropeGraph = (
  order: RopeOrder,
  interpose: boolean,
  heads = 1,
  headDim = 128,
  layout: "bhsd" | "bshd" = "bhsd",
): DeclarationJson => {
  const shape =
    layout === "bshd" ? ([1, "S", heads, headDim] as const) : ([1, heads, "S", headDim] as const);
  const tableShape =
    layout === "bshd" ? ([1, "S", 1, headDim] as const) : ([1, 1, "S", headDim] as const);
  const halfShape =
    layout === "bshd"
      ? ([1, "S", heads, headDim / 2] as const)
      : ([1, heads, "S", headDim / 2] as const);
  const values: DeclarationJson["values"] = {
    first: { dtype: "f32", shape: [...halfShape] },
    second: { dtype: "f32", shape: [...halfShape] },
    negative: { dtype: "f32", shape: [...halfShape] },
    rotated: { dtype: "f32", shape: [...shape] },
    direct: { dtype: "f32", shape: [...shape] },
    cross: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  if (interpose) values.first_alias = { dtype: "f32", shape: [...halfShape] };
  const direct: DeclarationJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: DeclarationJson["nodes"] = [];
  if (order === "direct-first") nodes.push(direct);
  nodes.push({
    op: "slice",
    ins: ["x"],
    outs: ["first"],
    attrs: { dim: 3, start: 0, end: headDim / 2 },
  });
  if (interpose) {
    nodes.push({ op: "reshape", ins: ["first"], outs: ["first_alias"], attrs: {} });
  }
  nodes.push(
    {
      op: "slice",
      ins: ["x"],
      outs: ["second"],
      attrs: { dim: 3, start: headDim / 2, end: headDim },
    },
    { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
    {
      op: "cat",
      ins: ["negative", interpose ? "first_alias" : "first"],
      outs: ["rotated"],
      attrs: { dim: 3 },
    },
  );
  if (order === "slice-first") nodes.push(direct);
  nodes.push(
    { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
    { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: ["S"],
    inputs: [
      { name: "x", dtype: "f32", shape: [...shape] },
      { name: "cos", dtype: "f32", shape: [...tableShape] },
      { name: "sin", dtype: "f32", shape: [...tableShape] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertFloatParity = (actual: Uint32Array, expected: Uint32Array): void => {
  assertEquals(actual.length, expected.length);
  const word = new Uint32Array(1);
  const value = new Float32Array(word.buffer);
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === expected[i]) continue;
    word[0] = actual[i];
    const actualValue = value[0];
    word[0] = expected[i];
    const expectedValue = value[0];
    assertEquals(
      Number.isNaN(actualValue) && Number.isNaN(expectedValue),
      true,
      "non-NaN values must be bit-identical index=" +
        i +
        " actual=0x" +
        actual[i].toString(16) +
        " expected=0x" +
        expected[i].toString(16),
    );
  }
};

/** The full input set, with ±0 / subnormals / ±Inf / NaN payloads written at the start of x. */
const ropeInputs = (heads: number, sequence: number): Readonly<Record<string, Tensor>> => {
  const x = fill([1, heads, sequence, 128], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  new Uint32Array(x.data.buffer).set([
    0x00000000, // +0
    0x80000000, // -0
    0x00000001, // smallest positive subnormal
    0x80000001, // smallest-magnitude negative subnormal
    0x007fffff, // largest positive subnormal
    0x807fffff, // largest-magnitude negative subnormal
    0x7f800000, // +Inf
    0xff800000, // -Inf
    0x7fc01234, // quiet NaN payload
    0xff801234, // negative signaling NaN payload
  ]);
  return {
    x,
    cos: fill([1, 1, sequence, 128], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.23 - 0.2)),
  };
};

test.skipIf(!GPU_AVAILABLE)(
  "half-split RoPE fusion matches in finite bits and NaN class in both slice/direct-first orders and folds into 1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = ropeInputs(1, 3);
    try {
      for (const order of ["slice-first", "direct-first"] as const) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, false), []),
          GRAPH_NAME,
        );
        const split = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, true), []),
          GRAPH_NAME,
        );
        try {
          const fusedOut = (await fused.run(inputs)).y;
          const splitOut = (await split.run(inputs)).y;
          assertFloatParity(bits(fusedOut), bits(splitOut));
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: RoPE 1 pass`);
          assertEquals(
            split.diagnostics().submit.dispatchCount,
            8,
            `${order}: slice×2 + neg + cat×2 + mul×2 + add`,
          );
          assertEquals(fused.diagnostics().lastRunFusions?.rope, 1, `${order}: fusion counter`);
          assertEquals(
            split.diagnostics().lastRunFusions?.rope,
            0,
            `${order}: the counter of the counterexample is 0`,
          );
        } finally {
          await fused.dispose();
          await split.dispose();
        }
      }
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
      for (const order of ["slice-first", "direct-first"] as const) {
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
  "RoPE fusion's head-direction index is bit-identical to the primitive sequence with H≥2 too (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = ropeInputs(3, 5);
    try {
      for (const order of ["slice-first", "direct-first"] as const) {
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, false, 3), []),
          GRAPH_NAME,
        );
        const split = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropeGraph(order, true, 3), []),
          GRAPH_NAME,
        );
        try {
          const fusedOut = (await fused.run(inputs)).y;
          const splitOut = (await split.run(inputs)).y;
          assertEquals(fusedOut.shape, [1, 3, 5, 128], `${order}: output shape`);
          assertFloatParity(bits(fusedOut), bits(splitOut));
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: RoPE 1 pass`);
          assertEquals(fused.diagnostics().lastRunFusions?.rope, 1, `${order}: fusion counter`);
        } finally {
          await fused.dispose();
          await split.dispose();
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
  "BSHD RoPE handles the token/head axes, partial workgroups, widths and special values bit-identically to the primitive sequence (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, sequence, dim] of [
        [8, 1, 256],
        [16, 17, 64],
        [3, 17, 6],
        [4, 3, 128],
        [8, 9, 256],
        [2, 5, 512],
      ]) {
        const inputs = {
          x: fill([1, sequence, heads, dim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997),
          cos: fill([1, sequence, 1, dim], (i) => Math.cos(i * 0.19 + 0.3)),
          sin: fill([1, sequence, 1, dim], (i) => Math.sin(i * 0.23 - 0.2)),
        };
        new Uint32Array(inputs.x.data.buffer).set([
          0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000, 0x7fc01234,
          0xff801234,
        ]);
        for (const order of ["slice-first", "direct-first"] as const) {
          const fused = await createSessionFromContainer(
            gpu,
            await openModelBytes(ropeGraph(order, false, heads, dim, "bshd"), []),
            GRAPH_NAME,
          );
          const split = await createSessionFromContainer(
            gpu,
            await openModelBytes(ropeGraph(order, true, heads, dim, "bshd"), []),
            GRAPH_NAME,
          );
          try {
            const actual = (await fused.run(inputs)).y;
            const expected = (await split.run(inputs)).y;
            assertEquals(actual.shape, [1, sequence, heads, dim]);
            assertFloatParity(bits(actual), bits(expected));
            assertEquals(fused.diagnostics().lastRunFusions?.rope, 1);
            assertEquals(split.diagnostics().lastRunFusions?.rope, 0);
            assertEquals(fused.diagnostics().submit.dispatchCount, 1);
          } finally {
            await fused.dispose();
            await split.dispose();
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
        assertFloatParity(bits((await fused.run(inputs)).y), bits((await split.run(inputs)).y));
        assertEquals(fused.diagnostics().lastRunFusions?.rope, 1);
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

/**
 * The pair-form RoPE as the exporter writes it for Wan's DiT: `x` `[1,S,H,D]` is viewed as pairs
 * `[1,S,H·D/2,2]`, `slice(1..2), neg, slice(0..1), cat`, viewed back and combined with the
 * `[1,S,1,D]` tables.
 *
 * `interpose=true` puts a zero-dispatch reshape between `cat` and the reshape back: the values and
 * the physical dispatches stay the same, but the exact matcher no longer matches, so it is the
 * unfused reference on the same backend.
 * `separatePairs=true` makes `pairs` a graph input of its own, so the kernel's `pairs` binding is a
 * different buffer from `x` (the rotation must come from `pairs`, not from `x`).
 */
const ropePairGraph = (
  heads: number,
  headDim: number,
  interpose: boolean,
  separatePairs = false,
): DeclarationJson => {
  const full = [1, "S", heads, headDim];
  const pairs = [1, "S", (heads * headDim) / 2, 2];
  const half = [1, "S", (heads * headDim) / 2, 1];
  const values: DeclarationJson["values"] = {
    odd: { dtype: "f32", shape: half },
    negative: { dtype: "f32", shape: half },
    even: { dtype: "f32", shape: half },
    rotated: { dtype: "f32", shape: pairs },
    rot: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const nodes: DeclarationJson["nodes"] = [];
  if (!separatePairs) {
    values.pairs = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["x"], outs: ["pairs"], attrs: {} });
  }
  nodes.push(
    { op: "slice", ins: ["pairs"], outs: ["odd"], attrs: { dim: 3, start: 1, end: 2 } },
    { op: "neg", ins: ["odd"], outs: ["negative"], attrs: {} },
    { op: "slice", ins: ["pairs"], outs: ["even"], attrs: { dim: 3, start: 0, end: 1 } },
    { op: "cat", ins: ["negative", "even"], outs: ["rotated"], attrs: { dim: 3 } },
  );
  if (interpose) {
    values.rotated_alias = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["rotated"], outs: ["rotated_alias"], attrs: {} });
  }
  nodes.push(
    { op: "reshape", ins: [interpose ? "rotated_alias" : "rotated"], outs: ["rot"], attrs: {} },
    { op: "mul", ins: ["x", "cos"], outs: ["direct"], attrs: {} },
    { op: "mul", ins: ["rot", "sin"], outs: ["cross"], attrs: {} },
    { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: ["S"],
    inputs: [
      { name: "x", dtype: "f32", shape: full },
      ...(separatePairs ? [{ name: "pairs", dtype: "f32", shape: pairs }] : []),
      { name: "cos", dtype: "f32", shape: [1, "S", 1, headDim] },
      { name: "sin", dtype: "f32", shape: [1, "S", 1, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/** ±0 / subnormals / ±Inf / NaN payloads at the head of a tensor (both signs, both pair slots). */
const SPECIAL_WORDS = [
  0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000,
  0x7fc01234, 0xff801234,
];

const ropePairInputs = (
  heads: number,
  sequence: number,
  headDim: number,
  separatePairs: boolean,
): Readonly<Record<string, Tensor>> => {
  const x = fill([1, sequence, heads, headDim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  new Uint32Array(x.data.buffer).set(SPECIAL_WORDS.slice(0, x.data.length));
  const inputs: Record<string, Tensor> = {
    x,
    cos: fill([1, sequence, 1, headDim], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, sequence, 1, headDim], (i) => Math.sin(i * 0.23 - 0.2)),
  };
  if (separatePairs) {
    // Different values from x (and the special words shifted by one slot), so a kernel that
    // rotated x instead of pairs would differ.
    const pairs = fill([1, sequence, (heads * headDim) / 2, 2], (i) => Math.cos(i * 0.41) * -2.3);
    new Uint32Array(pairs.data.buffer).set(SPECIAL_WORDS.slice(0, pairs.data.length - 1), 1);
    inputs.pairs = pairs;
  }
  return inputs;
};

test.skipIf(!GPU_AVAILABLE)(
  "pair-form RoPE fusion matches the unfused primitive sequence bit for bit (finite) / by NaN class across heads, widths and tails, and folds 8 dispatches into 1 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Wan2.2 (24 × 128), Irodori (10 × 64 / 12 × 64), a narrow width, one token, a wide head,
      // and lengths that leave a partial last workgroup.
      for (const [heads, sequence, headDim, separatePairs] of [
        [24, 5, 128, false],
        [10, 17, 64, false],
        [12, 3, 64, false],
        [3, 17, 6, false],
        [4, 1, 128, false],
        [2, 9, 256, false],
        [3, 7, 128, true],
      ] as const) {
        const label = `H=${heads} S=${sequence} D=${headDim}${separatePairs ? " separate pairs" : ""}`;
        const inputs = ropePairInputs(heads, sequence, headDim, separatePairs);
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropePairGraph(heads, headDim, false, separatePairs), []),
          GRAPH_NAME,
        );
        const split = await createSessionFromContainer(
          gpu,
          await openModelBytes(ropePairGraph(heads, headDim, true, separatePairs), []),
          GRAPH_NAME,
        );
        try {
          const actual = (await fused.run(inputs)).y;
          const expected = (await split.run(inputs)).y;
          assertEquals(actual.shape, [1, sequence, heads, headDim], label);
          assertFloatParity(bits(actual), bits(expected));
          assertEquals(fused.diagnostics().lastRunFusions?.rope, 1, `${label}: fused counter`);
          assertEquals(split.diagnostics().lastRunFusions?.rope, 0, `${label}: reference counter`);
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${label}: one pass`);
          assertEquals(
            split.diagnostics().submit.dispatchCount,
            8,
            `${label}: slice + neg + slice + cat×2 + mul×2 + add`,
          );
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
            [ROPE_PAIR_BSHD_KEY],
            `${label}: census`,
          );
        } finally {
          await fused.dispose();
          await split.dispose();
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
        assertFloatParity(bits((await fused.run(inputs)).y), bits((await split.run(inputs)).y));
        assertEquals(fused.diagnostics().lastRunFusions?.rope, 1);
      } finally {
        await fused.dispose();
        await split.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
