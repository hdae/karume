// adaLN = the strict peephole for `layer_norm → (1 + scale) → mul → add` (fusion rule adaln —
// src/runtime/fusion.ts). Only the **non-adjacent windows of 6 / 7 nodes** that the exporter emits
// are folded into 1 dispatch; a shape it cannot match falls back to the plain sequence.
//
// This check looks at one thing: bit parity of the values. The fused kernel removes the storage
// round trips of the 3 intermediates (normalized result / `1 + scale` / product), so if its
// rounding points drift from the plain sequence the bits of finite values move. Whether the
// rounding barrier of the workgroup u32 staging works on real backends can only be seen here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
} from "../src/kernels/adaln-norm.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * The row length exercises a remainder of the 256-thread workgroup; the row count makes grid-stride
 * loop at least once.
 */
const ROWS = 5;
const DIM = 257;
const ROW_SHAPE = [1, ROWS, DIM] as const;
const MOD_SHAPE = [1, 1, DIM] as const;

type AdalnGraphOptions = {
  /** Also place the gate's reshape, as measured (window 7). false gives window 6. */
  readonly gate?: boolean;
  /**
   * Interpose one 0-dispatch alias on the normalized result. It removes only the wiring condition
   * of the strict matcher without changing values or the physical dispatch count (= builds the
   * reference on the same backend).
   */
  readonly interpose?: boolean;
  /**
   * Make the normalized result a graph output (a counterexample removing the private-internal-value
   * condition).
   */
  readonly normOutput?: boolean;
  /** A counterexample adding another consumer to `1 + scale`. */
  readonly modulationExtraConsumer?: boolean;
};

const adalnGraph = (options: AdalnGraphOptions = {}): DeclarationJson => {
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: [...ROW_SHAPE] },
    shift: { dtype: "f32", shape: [...MOD_SHAPE] },
    scale: { dtype: "f32", shape: [...MOD_SHAPE] },
    s: { dtype: "f32", shape: [...MOD_SHAPE] },
    p: { dtype: "f32", shape: [...ROW_SHAPE] },
    y: { dtype: "f32", shape: [...ROW_SHAPE] },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype: "f32", shape: [...ROW_SHAPE] },
    { name: "ln_weight", dtype: "f32", shape: [DIM] },
    { name: "ln_bias", dtype: "f32", shape: [DIM] },
    { name: "one", dtype: "f32", shape: [1] },
    { name: "shift_src", dtype: "f32", shape: [1, DIM] },
    { name: "scale_src", dtype: "f32", shape: [1, DIM] },
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
  ];
  let normForMul = "t";
  if (options.interpose) {
    values.t_alias = { dtype: "f32", shape: [...ROW_SHAPE] };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
    normForMul = "t_alias";
  }
  nodes.push(
    { op: "reshape", ins: ["shift_src"], outs: ["shift"], attrs: {} },
    { op: "reshape", ins: ["scale_src"], outs: ["scale"], attrs: {} },
  );
  if (options.gate !== false) {
    inputs.push({ name: "gate_src", dtype: "f32", shape: [1, DIM] });
    values.gate = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "reshape", ins: ["gate_src"], outs: ["gate"], attrs: {} });
  }
  nodes.push(
    { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [normForMul, "s"], outs: ["p"], attrs: {} },
    { op: "add", ins: ["p", "shift"], outs: ["y"], attrs: {} },
  );
  if (options.modulationExtraConsumer) {
    values.s_copy = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [
      ...(options.normOutput ? ["t"] : []),
      "y",
      // gate is a value consumed outside the chain (in real IR the next block consumes it).
      ...(options.gate !== false ? ["gate"] : []),
      ...(options.modulationExtraConsumer ? ["s_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/**
 * Each row has a different character. Rows 0–2 are finite (small and large variance, signed zeros),
 * row 3 pushes the magnitude to hit cancellation in the normalization, and row 4 mixes in ±Inf /
 * NaN to make the whole row NaN.
 */
const testInput = (): Tensor => {
  const x = fill(ROW_SHAPE, (i) => {
    const row = Math.floor(i / DIM);
    const at = i % DIM;
    if (row === 0) return Math.sin(at * 0.37) * 3.1 + at / 997;
    if (row === 1) return Math.cos(at * 0.11) * 1e-4;
    if (row === 2) return (at % 2 === 0 ? 1 : -1) * (at / 3);
    if (row === 3) return Math.sin(at * 0.017) * 1e5 + 1e6;
    return Math.tan(at * 0.05);
  });
  // Inject non-finite values into row 4 (mean becomes NaN and the whole row falls into the NaN
  // class).
  words(x).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * DIM);
  // The head of row 2 is ±0 / subnormal / the smallest normal number.
  words(x).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * DIM);
  return x;
};

const testInputs = (options: AdalnGraphOptions = {}): Readonly<Record<string, Tensor>> => ({
  x: testInput(),
  ln_weight: fill([DIM], (i) => 1 + Math.sin(i * 0.21) * 0.4),
  ln_bias: fill([DIM], (i) => Math.cos(i * 0.13) * 0.2),
  one: fill([1], () => 1),
  // scale ranges from near ±0 to fairly large (if `t · (scale + 1)` gets distributed, finite values
  // break).
  shift_src: fill([1, DIM], (i) => Math.sin(i * 0.07) * 2.5),
  scale_src: fill([1, DIM], (i) => Math.cos(i * 0.29) * 3 - 1),
  ...(options.gate !== false ? { gate_src: fill([1, DIM], (i) => i / 31) } : {}),
});

test.skipIf(!GPU_AVAILABLE)(
  "adaLN fusion matches primitives in finite bits / NaN class for both windows 6 / 7 and folds 4→1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const gate of [true, false] as const) {
        const label = `window ${gate ? 7 : 6}`;
        const inputs = testInputs({ gate });
        const fused = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnGraph({ gate }), []),
          GRAPH_NAME,
        );
        const primitive = await createSessionFromContainer(
          gpu,
          await openModelBytes(adalnGraph({ gate, interpose: true }), []),
          GRAPH_NAME,
        );
        try {
          // If the fusion silently drops off, the values stay correct and only dispatches grow. Pin
          // it directly with the counter.
          assertCheckCase(
            await checkFusionTwin(gpu, {
              label: `${label}: fused vs primitive`,
              fused,
              plain: primitive,
              feeds: inputs,
              counter: "adaln",
              hits: 1,
              nan: "class",
              outputs: ["y"],
              // One dispatch against layer_norm + add + mul + add.
              dispatches: { fused: 1, plain: 4 },
            }),
          );
          assertEquals(
            requireCensus(fused.diagnostics().lastRunPipelines, label).map((row) => row.key),
            [ADALN_NORM_KEY],
            `${label}: census key`,
          );
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
  "adaLN with an internal output / another consumer / an alias falls back from the strict matcher (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const inputs = testInputs();
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

/**
 * Wan2.2's per-token adaLN: `layer_norm, where(mask, scale_c, scale_g), add(+one), mul,
 * where(mask, shift_c, shift_g), add` — 6 contiguous nodes.
 *
 * `interpose=true` puts a zero-dispatch reshape on the normalised value (the same trick as the
 * broadcast form): the values and the physical dispatches stay the same and only the exact matcher
 * stops matching, so it is the unfused reference on the same backend.
 */
const adalnTokenGraph = (rows: number, dim: number, interpose: boolean): DeclarationJson => {
  const row = [1, rows, dim];
  const vector = [1, 1, dim];
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: row },
    sel_scale: { dtype: "f32", shape: row },
    s: { dtype: "f32", shape: row },
    p: { dtype: "f32", shape: row },
    sel_shift: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: row },
  };
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [dim], eps: 0.000001 },
    },
  ];
  if (interpose) {
    values.t_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
  }
  nodes.push(
    { op: "where", ins: ["mask", "scale_c", "scale_g"], outs: ["sel_scale"], attrs: {} },
    { op: "add", ins: ["sel_scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [interpose ? "t_alias" : "t", "s"], outs: ["p"], attrs: {} },
    { op: "where", ins: ["mask", "shift_c", "shift_g"], outs: ["sel_shift"], attrs: {} },
    { op: "add", ins: ["p", "sel_shift"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: row },
      { name: "ln_weight", dtype: "f32", shape: [dim] },
      { name: "ln_bias", dtype: "f32", shape: [dim] },
      { name: "mask", dtype: "bool", shape: [1, rows, 1] },
      { name: "one", dtype: "f32", shape: [1] },
      { name: "scale_c", dtype: "f32", shape: vector },
      { name: "scale_g", dtype: "f32", shape: vector },
      { name: "shift_c", dtype: "f32", shape: vector },
      { name: "shift_g", dtype: "f32", shape: vector },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * Rows of different character (as {@link testInput}): wide / tiny variance, signed alternation,
 * a large offset that cancels in the mean, a row with ±Inf / NaN, and ±0 / subnormals at the head
 * of row 2.
 */
const tokenX = (rows: number, dim: number): Tensor => {
  const x = fill([1, rows, dim], (i) => {
    const row = Math.floor(i / dim);
    const at = i % dim;
    switch (row % 5) {
      case 0:
        return Math.sin(at * 0.37 + row) * 3.1 + at / 997;
      case 1:
        return Math.cos(at * 0.11) * 1e-4;
      case 2:
        return (at % 2 === 0 ? 1 : -1) * (at / 3);
      case 3:
        return Math.sin(at * 0.017) * 1e5 + 1e6;
      default:
        return Math.tan(at * 0.05);
    }
  });
  if (rows > 4) words(x).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * dim);
  if (rows > 2) words(x).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * dim);
  return x;
};

/**
 * The four vectors have different values and signs (so picking the wrong side changes the bits),
 * with -0, a subnormal and a NaN payload in fixed slots.
 */
const tokenInputs = (
  rows: number,
  dim: number,
  mask: (row: number) => number,
): Readonly<Record<string, Tensor>> => {
  const scaleCondition = fill([1, 1, dim], (i) => Math.cos(i * 0.29) * 3 - 1);
  const scaleGeneration = fill([1, 1, dim], (i) => Math.sin(i * 0.31) * 2.5 + 0.5);
  const shiftCondition = fill([1, 1, dim], (i) => Math.sin(i * 0.07) * 2.5);
  const shiftGeneration = fill([1, 1, dim], (i) => -Math.cos(i * 0.05) * 1.5);
  words(scaleCondition).set([0x80000000, 0x00000001], 0);
  words(scaleGeneration).set([0x807fffff], 1);
  words(shiftCondition).set([0x7fc01234], 2);
  words(shiftGeneration).set([0x80000000, 0x00000003], 3);
  return {
    x: tokenX(rows, dim),
    ln_weight: fill([dim], (i) => 1 + Math.sin(i * 0.21) * 0.4),
    ln_bias: fill([dim], (i) => Math.cos(i * 0.13) * 0.2),
    mask: fill([1, rows, 1], mask, "bool"),
    one: fill([1], () => 1),
    scale_c: scaleCondition,
    scale_g: scaleGeneration,
    shift_c: shiftCondition,
    shift_g: shiftGeneration,
  };
};

/** Mixed (the I2V shape: the first latent frame is the condition side), all condition, all generation. */
const MASKS: readonly (readonly [string, (row: number) => number])[] = [
  ["mixed mask", (row) => (row % 3 === 1 ? 0 : 1)],
  ["all condition", () => 1],
  ["all generation", () => 0],
];

test.skipIf(!GPU_AVAILABLE)(
  "per-token adaLN fusion matches the unfused sequence bit for bit (finite) / by NaN class for mixed and uniform masks, and folds 6 dispatches into 2 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // 257 = a partial last block; 3072 = Wan2.2's width; 8 = narrower than one block.
      for (const [rows, dim] of [
        [5, 257],
        [7, 3072],
        [4, 8],
      ] as const) {
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
          for (const [maskLabel, mask] of MASKS) {
            const label = `rows=${rows} dim=${dim} ${maskLabel}`;
            assertCheckCase(
              await checkFusionTwin(gpu, {
                label: `${label}: fused vs primitive`,
                fused,
                plain: primitive,
                feeds: tokenInputs(rows, dim, mask),
                counter: "adaln",
                hits: 1,
                nan: "class",
                // pack + body against layer_norm + where + add + mul + where + add.
                dispatches: { fused: 2, plain: 6 },
              }),
            );
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
          await primitive.dispose();
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
            feeds: tokenInputs(rows, dim, (row) => row % 2),
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
