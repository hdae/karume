// Fault injection for the full-write invariant (ADR 0014 / arena invariant ⑤ of ADR 0004).
//
// It demonstrates "every kernel writes every byte of its output buffer" by **planting poison
// values in pool-reused buffers first** and then running cat / pad / flip / conv_transpose1d.
// Buffers reused from the buffer pool are not zero-initialized (only a fresh createBuffer
// guarantees zeros — src/gpu/arena.ts), so any element left unwritten keeps the poison value in
// the output.
//
// How it is planted: the output of a preceding node (an identity cast) is **the poison value
// itself**, and that node's output has zero consumers, so it returns to the pool at the node
// boundary. The output allocation of the next node is in the same size class, so the pool hands
// out the same buffer (reuseCount confirms that the reuse really happened — without reuse the check
// looks at nothing).
//
// MUST: the poison value is **a nonzero finite value**, not an "impossible value". The point is to
// catch a mix-up with zero filling, and a poison value of 0 could not be told apart from pad's
// zero region.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type AcquireGpuOptions } from "../src/gpu/device.ts";
import type { Conv3dDims } from "../src/kernels/conv3d.ts";
import { referenceConv3dI8a8 } from "../src/reference/conv3d-i8a8.ts";
import { quantizeRowsTieMargin, referenceLinearI8a8 } from "../src/reference/i8a8.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

/** The poison value 0xDEADBEEF read as f32 (-6.259854e18 — a nonzero finite value). */
const POISON = new Float32Array(new Uint32Array([0xdeadbeef]).buffer)[0];

/**
 * A 2-node graph: "an identity cast that makes the poison" + "the node under test".
 *
 * `seed` is a graph input filled with the poison value, and the output `poison` of cast(f32 → f32)
 * is **consumed by nobody**, so it returns to the pool as is at the node boundary. The element
 * count of `poison` matches the output under test, so the next allocation hands the same buffer out
 * again.
 */
const poisonGraph = (
  outShape: readonly number[],
  operands: readonly { readonly name: string; readonly shape: readonly number[] }[],
  node: { readonly op: string; readonly attrs: Record<string, unknown> },
): DeclarationJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", node.op] },
    symbols: [],
    inputs: [
      { name: "seed", dtype: "f32", shape: [count] },
      ...operands.map((operand) => ({
        name: operand.name,
        dtype: "f32",
        shape: [...operand.shape],
      })),
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      poison: { dtype: "f32", shape: [count] },
      y: { dtype: "f32", shape: [...outShape] },
    },
    nodes: [
      { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
      { op: node.op, ins: operands.map((operand) => operand.name), outs: ["y"], attrs: node.attrs },
    ],
  };
};

const runPoisoned = async (
  graph: DeclarationJson,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
  acquire: AcquireGpuOptions = {},
): Promise<{ readonly output: Tensor; readonly reuseCount: number }> => {
  const gpu = await acquireGpu(acquire);
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
    options,
  );
  try {
    const count = graph.inputs[0].shape[0] as number;
    const outputs = await session.run({ seed: fill([count], () => POISON), ...inputs });
    return { output: outputs["y"], reuseCount: session.diagnostics().lastRun?.reuseCount ?? 0 };
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "cat leaves not a single word of the poison in a pool-reused buffer (full-write / real GPU)",
  async () => {
    // [2,1] + [2,2] → [2,3]. The poison buffer has 6 elements, the same size class as cat's output.
    const graph = poisonGraph(
      [2, 3],
      [
        { name: "a", shape: [2, 1] },
        { name: "b", shape: [2, 2] },
      ],
      {
        op: "cat",
        attrs: { dim: 1 },
      },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      a: fill([2, 1], (i) => 10 + i),
      b: fill([2, 2], (i) => 20 + i),
    });
    // Without reuse the poison check looks at nothing (the check against a vacuous pass)
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals([...output.data], [10, 20, 21, 11, 22, 23]);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "pad's zero region is exactly 0, not the poison (full-write / real GPU)",
  async () => {
    // Pad [2,2] by [1,1] → [2,4]. If the zero region kept the poison, it turns red at once.
    const graph = poisonGraph([2, 4], [{ name: "x", shape: [2, 2] }], {
      op: "pad",
      attrs: { left: 1, right: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, { x: fill([2, 2], (i) => 1 + i) });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals([...output.data], [0, 1, 2, 0, 0, 3, 4, 0]);
    // MUST: the zero region is not just "not the poison" but **exactly 0** (a signed zero is +0
    // too)
    for (const index of [0, 3, 4, 7]) {
      assertEquals(Object.is(output.data[index], 0), true, `element ${index} is not +0`);
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "flip leaves not a single word of the poison in a pool-reused buffer (full-write / real GPU)",
  async () => {
    // flip is a **bijection** (1 output element ← 1 input element), so if anything is left
    // unwritten it is a grid-stride cut-off or an index jump, and both show up as remaining poison.
    // Reverse axis 1 of [2,3] → [2,3] (the poison buffer has 6 elements, the same size class).
    const graph = poisonGraph([2, 3], [{ name: "x", shape: [2, 3] }], {
      op: "flip",
      attrs: { dim: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, { x: fill([2, 3], (i) => 1 + i) });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals([...output.data], [3, 2, 1, 6, 5, 4]);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "upsample_bilinear2d leaves not a single word of the poison (full-write / real GPU)",
  async () => {
    // 1 output element = 1 thread, but **N·C is folded into one plane index**, so splitting the
    // planes wrongly leaves whole planes unwritten. Exercised with B·C = 4 planes (`[2,2,…]`).
    // [2,2,2,3] → [2,2,3,2] (24 elements, so input and output share a size class = the poison
    // buffer is handed out again).
    const graph = poisonGraph([2, 2, 3, 2], [{ name: "x", shape: [2, 2, 2, 3] }], {
      op: "upsample_bilinear2d",
      attrs: { output_size: [3, 2] },
    });
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([2, 2, 2, 3], (i) => 1 + i),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(output.shape, [2, 2, 3, 2]);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
    // The corners of each plane match the input's corners (a shifted plane index shows the
    // neighbouring plane's values)
    for (let plane = 0; plane < 4; plane += 1) {
      assertEquals(output.data[plane * 6], 1 + plane * 6, `top left of plane ${plane}`);
      assertEquals(output.data[plane * 6 + 5], 6 + plane * 6, `bottom right of plane ${plane}`);
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm leaves no poison word (full-write / real GPU)",
  async () => {
    // The apply stage writes one element per thread over the flat index and splits it into
    // (row, offset, channel); a wrong split leaves whole groups or samples unwritten. Two samples ×
    // two groups, 240 elements — the poison buffer has the output's size class, the node's two
    // temporaries (4 partials, 8 statistics) do not.
    const shape = [2, 4, 6, 5];
    const graph = poisonGraph(
      shape,
      [
        { name: "x", shape },
        { name: "w", shape: [4] },
        { name: "b", shape: [4] },
      ],
      { op: "group_norm", attrs: { num_groups: 2, eps: 1e-5 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill(shape, (i) => ((i * 7) % 13) - 6),
      w: fill([4], (i) => 1 + i),
      b: fill([4], (i) => 0.5 * i),
    });
    assert(reuseCount >= 1, `no pool reuse happened (reuseCount=${reuseCount})`);
    assertEquals(output.shape, shape);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison left in the output",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "deform_conv2d leaves not a single word of the poison (full-write / real GPU)",
  async () => {
    // 1 output element = 1 thread, but the flat index is decomposed into the 4 levels B / Cout /
    // Hout / Wout, so splitting a level wrongly leaves whole channels or batches unwritten.
    // Exercised with batch 2. x[2,2,3,4] * W[3,2,2,3], padding [1,1] → Hout 4 / Wout 4 → [2,3,4,4]
    // (96 elements).
    const outShape = [2, 3, 4, 4];
    const graph = poisonGraph(
      outShape,
      [
        { name: "x", shape: [2, 2, 3, 4] },
        { name: "w", shape: [3, 2, 2, 3] },
        { name: "off", shape: [2, 12, 4, 4] },
        { name: "mask", shape: [2, 6, 4, 4] },
        { name: "b", shape: [3] },
      ],
      { op: "deform_conv2d", attrs: { padding: [1, 1] } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([2, 2, 3, 4], (i) => 1 + i),
      w: fill([3, 2, 2, 3], (i) => 0.25 + i * 0.5),
      // The degenerate form with offset 0 and mask 1 (another test checks that the values match
      // plain conv2d)
      off: fill([2, 12, 4, 4], () => 0),
      mask: fill([2, 6, 4, 4], () => 1),
      b: fill([3], () => 0),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(output.shape, outShape);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
    // MUST: the 2 batches hold **different values** (dropping the B level makes the second a copy
    // of the first, which slips through the poison check alone).
    assertEquals(
      [...output.data.slice(0, 48)].every((value, index) => value === output.data[48 + index]),
      false,
      "the 2 batches are identical (the B index decomposition has no effect)",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "gru_scan leaves not a single word of the poison (full-write / real GPU)",
  async () => {
    // The output `[T,N,H]` is written by the 3 levels "time loop × batch grid-stride × lane (hidden
    // unit)". If any level is reduced, only that step / batch / unit keeps the poison (the contract
    // check and the comparison with the CPU reference **only look at written elements**, so this is
    // the only check).
    // MUST: exercise with batch 2 (1 workgroup = 1 batch element, so a grid-stride cut-off only
    // shows with N ≥ 2).
    const outShape = [3, 2, 4];
    const graph = poisonGraph(
      outShape,
      [
        { name: "gi", shape: [3, 2, 12] },
        { name: "h0", shape: [2, 4] },
        { name: "w", shape: [12, 4] },
        { name: "b", shape: [12] },
      ],
      { op: "gru_scan", attrs: {} },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      gi: fill([3, 2, 12], (i) => ((i % 11) - 5) * 0.17),
      // A different initial state for each of the 2 batches (exercises an implementation that skips
      // reading h0 and a reduced B level at once)
      h0: fill([2, 4], (i) => (i < 4 ? 0.6 : -0.4)),
      w: fill([12, 4], (i) => ((i % 7) - 3) * 0.09),
      b: fill([12], (i) => ((i % 5) - 2) * 0.13),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(output.shape, outShape);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
    // MUST: the 2 batches hold **different values** (dropping the B level makes the second a copy
    // of the first, which slips through the poison check alone). h0 differs per batch, so if the
    // level works they always differ.
    for (let step = 0; step < outShape[0]; step += 1) {
      const base = step * 8;
      assertEquals(
        [...output.data.slice(base, base + 4)].every(
          (value, index) => value === output.data[base + 4 + index],
        ),
        false,
        `t=${step}: the 2 batches are identical (the B level has no effect)`,
      );
    }
    // MUST: the values also move along time (the time loop is not cut off after 1 step).
    assertEquals(
      [...output.data.slice(0, 8)].every((value, index) => value === output.data[8 + index]),
      false,
      "t=0 and t=1 are identical (the time loop is reduced)",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "gated_delta_scan leaves not one poisoned word (full-write / real GPU)",
  async () => {
    // o[B,T,Hv,128] is written by the slice-0 thread of each column, one token late (c_prev) and
    // the last token in the epilogue (ADR 0135 — kernels/gated-delta-scan.ts). A skipped token, the
    // epilogue or a column block shows up as a poisoned word.
    // MUST: T = 9 crosses the token block of 8 (the second block holds one token), B = 2 and
    // Hv = 2 exercise the batch / head decomposition of the workgroup index.
    const [batch, time, valueHeads, dim] = [2, 9, 2, 128];
    const outShape = [batch, time, valueHeads, dim];
    const graph = poisonGraph(
      outShape,
      [
        { name: "q", shape: [batch, time, 1, dim] },
        { name: "k", shape: [batch, time, 1, dim] },
        { name: "v", shape: [batch, time, valueHeads, dim] },
        { name: "g", shape: [batch, time, valueHeads] },
        { name: "beta", shape: [batch, time, valueHeads] },
      ],
      { op: "gated_delta_scan", attrs: {} },
    );
    // q / k rows of norm 1 (two entries of ±1/√2 at positions that move with the row)
    const unitRow = (i: number): number => {
      const row = Math.floor(i / dim);
      const column = i % dim;
      if (column === row % dim) return Math.SQRT1_2;
      return column === (row * 5 + 3) % dim ? -Math.SQRT1_2 : 0;
    };
    const { output, reuseCount } = await runPoisoned(graph, {
      q: fill([batch, time, 1, dim], unitRow),
      k: fill([batch, time, 1, dim], (i) => unitRow(i + 7 * dim)),
      v: fill([batch, time, valueHeads, dim], (i) => ((i % 13) - 6) * 0.11 + Math.floor(i / 2304)),
      g: fill([batch, time, valueHeads], (i) => -0.05 * ((i % 5) + 1)),
      beta: fill([batch, time, valueHeads], (i) => 0.2 + 0.05 * (i % 7)),
    });
    assert(reuseCount >= 1, `the pool reused no buffer (reuseCount=${reuseCount})`);
    assertEquals(output.shape, outShape);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poisoned words left",
    );
    assert([...output.data].every(Number.isFinite), "a non-finite word");
    // MUST: the two batch elements differ (dropping the B decomposition would copy the first)
    const half = output.data.length / 2;
    assertEquals(
      [...output.data.slice(0, half)].every((value, index) => value === output.data[half + index]),
      false,
      "batch elements 0 and 1 are identical",
    );
    // MUST: the last token differs from the one before (the epilogue writes it)
    const row = valueHeads * dim;
    assertEquals(
      [...output.data.slice((time - 1) * row, time * row)].every(
        (value, index) => value === output.data[(time - 2) * row + index],
      ),
      false,
      "tokens T-2 and T-1 are identical",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the conv2d direct kernel (groups>1) leaves not a single word of the poison (full-write / real GPU)",
  async () => {
    // Writes every byte of the output with 1 element per thread (ADR 0017). A grid-stride cut-off
    // or an error in decomposing the flat index (the 4 levels B / Cout / H / W) shows up as
    // unwritten elements = remaining poison.
    // MUST: **groups > 1** (the side that the dispatch split of ADR 0024 sends to the direct
    // kernel). groups == 1 goes to implicit GEMM, so with that alone this one test would never
    // exercise the direct kernel.
    // x[1,2,3,4] * W[4,1,2,3] groups=2 pad=[1,1] → [1,4,4,4] (the poison buffer has 64 elements).
    const graph = poisonGraph(
      [1, 4, 4, 4],
      [
        { name: "x", shape: [1, 2, 3, 4] },
        { name: "w", shape: [4, 1, 2, 3] },
        { name: "b", shape: [4] },
      ],
      {
        op: "conv2d",
        attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 2 },
      },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([1, 2, 3, 4], (i) => 1 + i),
      w: fill([4, 1, 2, 3], (i) => 0.25 + i * 0.5),
      b: fill([4], () => 0),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(output.shape, [1, 4, 4, 4]);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);

/**
 * The implicit GEMM of conv2d (`groups == 1` — ADR 0024) writes tiles the same way as the GEMM
 * skeleton, so its only safety nets are the fail loudly of `tiledWorkgroups` and the poison
 * injection here.
 *
 * MUST: **keep the v4 path and the scalar path as a pair** (the write guard is a different
 * expression per variant).
 * MUST: the shapes cross 64 (collapsed below 1 tile, the guard is never false).
 * MUST: have a batch-2 case — the batch is the dispatch's **z axis**, so an error that fixes z to 1
 * stays green in the B = 1 cases and with B ≥ 2 leaves "the whole second batch unwritten".
 */
const CONV2D_IGEMM_POISON_CASES: readonly {
  readonly name: string;
  readonly outShape: readonly number[];
  readonly x: readonly number[];
  readonly w: readonly number[];
  readonly attrs: Record<string, unknown>;
}[] = [
  {
    // v4 (kFlat = 4·1·1 = 4 / Wout = 8 / stride_w = 1). m = 70 gives 2 row tiles,
    // n = Hout·Wout = 72 gives 2 column tiles (the last tile has 2 valid quads out of 16).
    name: "conv2d igemm v4 [1,4,9,8] * W[70,4,1,1]",
    outShape: [1, 70, 9, 8],
    x: [1, 4, 9, 8],
    w: [70, 4, 1, 1],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // Scalar (kFlat = 2·3·3 = 18 is not a multiple of 4). m = 70 / n = 9·7 = 63.
    name: "conv2d igemm scalar [1,2,9,7] * W[70,2,3,3]",
    outShape: [1, 70, 9, 7],
    x: [1, 2, 9, 7],
    w: [70, 2, 3, 3],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // Batch 2 (both batches of the z axis are written)
    name: "conv2d igemm batch 2 [2,4,9,8] * W[70,4,1,1]",
    outShape: [2, 70, 9, 8],
    x: [2, 4, 9, 8],
    w: [70, 4, 1, 1],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "the implicit GEMM of conv2d leaves not a single word of the poison (full-write of the tile writes / real GPU)",
  async () => {
    for (const testCase of CONV2D_IGEMM_POISON_CASES) {
      const graph = poisonGraph(
        testCase.outShape,
        [
          { name: "x", shape: testCase.x },
          { name: "w", shape: testCase.w },
          { name: "b", shape: [testCase.w[0]] },
        ],
        { op: "conv2d", attrs: testCase.attrs },
      );
      const { output, reuseCount } = await runPoisoned(graph, {
        x: fill(testCase.x, (i) => 0.5 + (i % 11) * 0.25),
        w: fill(testCase.w, (i) => 0.25 + (i % 7) * 0.5),
        b: fill([testCase.w[0]], () => 0),
      });
      assert(reuseCount >= 1, `${testCase.name}: pool reuse did not happen`);
      assertEquals(output.shape, [...testCase.outShape], testCase.name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${testCase.name}: poison remains`,
      );
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rms_norm leaves not a single word of the poison (full-write of a row kernel / real GPU)",
  async () => {
    // A reduced row loop (grid-stride) or a cut-off inner loop leaves "unwritten rows / columns".
    // [3,4] (the poison buffer has 12 elements, the same size class).
    const graph = poisonGraph(
      [3, 4],
      [
        { name: "x", shape: [3, 4] },
        { name: "w", shape: [4] },
      ],
      { op: "rms_norm", attrs: { eps: 1e-6 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([3, 4], (i) => 1 + i),
      w: fill([4], () => 1),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);

/**
 * Tile writes of the 3 GEMM ops (matmul / bmm / linear) (64×64 register tiles —
 * src/kernels/gemm.ts).
 *
 * MUST: **keep the v4 path and the scalar path as a pair**. The write guard is a different
 * expression per variant (v4 = 1 quad guard / scalar = 4, one per component), so one of them alone
 * does not check that "the tiles cover the whole range exactly once with `ceil(m/64) ×
 * ceil(n/64)`".
 * MUST: the shapes cross 64 (collapsed below 1 tile, the guard is never false).
 */
const GEMM_POISON_CASES: readonly {
  readonly name: string;
  readonly outShape: readonly number[];
  readonly operands: readonly { readonly name: string; readonly shape: readonly number[] }[];
  readonly op: string;
}[] = [
  // v4 (k=20 / n=68 both multiples of 4). With n=68 the last column tile has 1 valid quad out of
  // 16.
  {
    name: "matmul v4 [65,20] × [20,68]",
    op: "matmul",
    outShape: [65, 68],
    operands: [
      { name: "a", shape: [65, 20] },
      { name: "b", shape: [20, 68] },
    ],
  },
  // Scalar (k=19 / n=23 neither a multiple of 4)
  {
    name: "matmul scalar [70,19] × [19,23]",
    op: "matmul",
    outShape: [70, 23],
    operands: [
      { name: "a", shape: [70, 19] },
      { name: "b", shape: [19, 23] },
    ],
  },
  // bmm must also cover the whole batch axis (z)
  {
    name: "bmm v4 [3,68,20] × [3,20,12]",
    op: "bmm",
    outShape: [3, 68, 12],
    operands: [
      { name: "a", shape: [3, 68, 20] },
      { name: "b", shape: [3, 20, 12] },
    ],
  },
  {
    name: "bmm scalar [2,70,19] × [2,19,23]",
    op: "bmm",
    outShape: [2, 70, 23],
    operands: [
      { name: "a", shape: [2, 70, 19] },
      { name: "b", shape: [2, 19, 23] },
    ],
  },
  // linear's write expression differs by the bias addition (the path with f32 weight storage)
  {
    name: "linear v4 [65,20] × W[68,20]",
    op: "linear",
    outShape: [65, 68],
    operands: [
      { name: "x", shape: [65, 20] },
      { name: "w", shape: [68, 20] },
      { name: "b", shape: [68] },
    ],
  },
  {
    name: "linear scalar [70,37] × W[23,37]",
    op: "linear",
    outShape: [70, 23],
    operands: [
      { name: "x", shape: [70, 37] },
      { name: "w", shape: [23, 37] },
      { name: "b", shape: [23] },
    ],
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "the 3 GEMM ops leave not a single word of the poison (full-write of the tile writes / real GPU)",
  async () => {
    for (const testCase of GEMM_POISON_CASES) {
      const graph = poisonGraph(testCase.outShape, testCase.operands, {
        op: testCase.op,
        attrs: {},
      });
      const inputs: Record<string, Tensor> = {};
      testCase.operands.forEach((operand, index) => {
        inputs[operand.name] = fill(operand.shape, (i) => 0.25 + ((i + index) % 11) * 0.5);
      });
      const { output, reuseCount } = await runPoisoned(graph, inputs);
      assert(reuseCount >= 1, `${testCase.name}: pool reuse did not happen`);
      assertEquals(output.shape, [...testCase.outShape], testCase.name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${testCase.name}: poison remains`,
      );
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv_transpose1d leaves not a single word of the poison (full-write of the gather form / real GPU)",
  async () => {
    // A plain scatter implementation (spreading 1 input point to several output points) assumes
    // zero initialization, so in a pool-reused buffer the poison stays where nothing contributes.
    // The gather form always writes those positions.
    // x[1,1,3] * W[1,2,4] stride=2 padding=1 → [1,2,6] (the poison buffer has 12 elements, the same
    // size).
    const graph = poisonGraph(
      [1, 2, 6],
      [
        { name: "x", shape: [1, 1, 3] },
        { name: "w", shape: [1, 2, 4] },
        { name: "b", shape: [2] },
      ],
      { op: "conv_transpose1d", attrs: { stride: 2, padding: 1 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([1, 1, 3], (i) => 1 + i),
      w: fill([1, 2, 4], (i) => 0.5 + i),
      b: fill([2], () => 0),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);

/**
 * Full-write of the **node-internal temporary buffers** `xq` / `xs` of w8a8 linear (1 node = 2
 * dispatches).
 *
 * Unlike the other families, the poison here goes not into the node's output but into the 2
 * temporary regions that `quantize_rows` writes. Something left unwritten shows up as "the integer
 * dot product consuming the earlier, unquantized values", so the detector is **an atol=0
 * comparison with the TS reference** rather than "the poison itself remaining" (the poison always
 * turns into some other integer, so a single remaining word moves the values).
 *
 * MUST: the size of the poison buffer must **not match the node output `y`**. y is allocated before
 * xq / xs, so with the same size class the poison would be handed to y and never reach the
 * temporary regions.
 */
const i8a8PoisonGraph = (
  poisonCount: number,
  m: number,
  n: number,
  k: number,
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cast", "linear"] },
  symbols: [],
  inputs: [
    { name: "seed", dtype: "f32", shape: [poisonCount] },
    { name: "x", dtype: "f32", shape: [m, k] },
  ],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    poison: { dtype: "f32", shape: [poisonCount] },
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
  },
  nodes: [
    { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
    { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "the temporary buffers (xq / xs) of i8a8 linear leave not a single word of the poison (full-write / real GPU)",
  async () => {
    // m=4 / n=5 / k=8 → y = 80 bytes, xq = m·k = 32 bytes, xs = m·4 = 16 bytes.
    // All 3 are in different size classes, so the size of the poison buffer selects where the
    // poison goes.
    const m = 4;
    const n = 5;
    const k = 8;
    const cases: readonly { readonly name: string; readonly poisonCount: number }[] = [
      { name: "xq (the quantized activations)", poisonCount: (m * k) / 4 },
      { name: "xs（per-token scale）", poisonCount: m },
    ];
    const weight = fill([n, k], (i) => 0.125 + (i % 17) * 0.5 * (1 + (Math.floor(i / k) % 5)));
    const quantized = quantizeI8(weight.data, [n, k], 0);
    const bias = fill([n], (i) => ((i % 7) - 3) * 0.25);
    // Steps away from the rounding boundaries (the condition for claiming atol=0 — the MUST in
    // tests/gpu_i8a8_test.ts)
    const x = fill([m, k], (i) => ((i % 29) - 14) * 0.3717 + 0.0131);
    const expected = referenceLinearI8a8({
      x: x.data,
      weight: new Int8Array(quantized.bytes.buffer, quantized.bytes.byteOffset, n * k),
      weightScale: quantized.scale,
      bias: bias.data,
      m,
      n,
      k,
    });
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const model = await openModelBytes(i8a8PoisonGraph(testCase.poisonCount, m, n, k), [
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(bias.data),
            encoding: { codec: "f32" },
          },
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: quantized.bytes,
            // Per-channel (groupSize = row length K); `rowAxis` is the channel axis 0 of linear's
            // weights.
            encoding: {
              codec: "int8-sym",
              groupSize: k,
              scale: { dtype: "f32", bytes: f32Bytes(quantized.scale) },
            },
          },
        ]);
        const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
          linearCompute: "a8",
        });
        try {
          const outputs = await session.run({
            seed: fill([testCase.poisonCount], () => POISON),
            x,
          });
          const reuseCount = session.diagnostics().lastRun?.reuseCount ?? 0;
          assert(reuseCount >= 1, `${testCase.name}: pool reuse did not happen`);
          assertEquals(
            [...outputs["y"].data],
            [...expected],
            `${testCase.name}: atol=0 comparison`,
          );
          // The check against a vacuous pass: with a constant output the comparison checks nothing
          assert(new Set([...outputs["y"].data]).size > 1, `${testCase.name}: constant output`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The node-local temporaries `xq` / `xs` of conv3d w8a8 (one node = two dispatches — the
 * channel-major `quantize_rows` reads the conv input in place and writes them; perf-ledger V-6).
 * As for linear's i8a8 temporaries above, a word the quantiser leaves unwritten is eaten by the
 * integer GEMM as a different integer or scale, so the detector is the atol = 0 match against the
 * TS reference.
 *
 * How the poison reaches them: the liveness packer (src/runtime/transient-plan.ts, ADR 0093) places
 * slots largest first, each in the region it grows least, and keeps a slot out of every region that
 * holds a partner (a slot read in a dispatch where it is written, or the reverse). The GEMM reads
 * `xq` / `xs` and writes `y`, so `y` never shares their region. With the poison (512 B) the
 * largest slot, then `xq` (96 B), `y` (48 B) and `xs` (48 B): the poison opens region 0, `xq` takes
 * its first bytes, `y` is kept out and opens region 1, and `xs` goes after `xq` in region 0 (at
 * most 256 + 48 B — the offset alignment), still inside the poison's bytes.
 * MUST: the poison is the largest slot and `y` is smaller than `xq` — a `y` placed before `xq` has
 * no partner placed yet and takes the poison's bytes (as a poison smaller than `y` would land on
 * `y`'s), and the temporaries get fresh, zeroed bytes. The run checks the result: exactly two slots
 * share bytes with an earlier one (`xq` and `xs` on the poison).
 * MUST: the poison word is a scale and a packed word that change the output when left in place —
 * the file's {@link POISON} is about −6e-18 as an f32, so a poisoned scale times the integer sum
 * would round to the bias, the same output as a zeroed scale.
 */
const CONV3D_TEMP_POISON = new Float32Array(new Uint32Array([0x41424344]).buffer)[0];
const CONV3D_TEMP_POISON_COUNT = 128;

const conv3dI8a8PoisonGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cast", "conv3d"] },
  symbols: [],
  inputs: [
    { name: "seed", dtype: "f32", shape: [CONV3D_TEMP_POISON_COUNT] },
    { name: "x", dtype: "f32", shape: [8, 1, 3, 4] },
  ],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    poison: { dtype: "f32", shape: [CONV3D_TEMP_POISON_COUNT] },
    w: { dtype: "f32", shape: [1, 8, 1, 1, 1] },
    b: { dtype: "f32", shape: [1] },
    y: { dtype: "f32", shape: [1, 1, 3, 4] },
  },
  nodes: [
    { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
    {
      op: "conv3d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1 },
    },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "conv3d w8a8's temporaries (xq / xs) keep no poisoned word (full-write / real GPU)",
  async () => {
    const channelsIn = 8;
    const channelsOut = 1;
    const pixels = 12;
    const dims: Conv3dDims = {
      channelsIn,
      channelsOut,
      timeIn: 1,
      heightIn: 3,
      widthIn: 4,
      timeOut: 1,
      heightOut: 3,
      widthOut: 4,
      kernelT: 1,
      kernelH: 1,
      kernelW: 1,
      strideT: 1,
      strideH: 1,
      strideW: 1,
      paddingT: 0,
      paddingH: 0,
      paddingW: 0,
      dilationT: 1,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
    };
    const weight = fill(
      [channelsOut, channelsIn, 1, 1, 1],
      (i) => ((i * 7) % 23) * 0.0625 - 0.6875,
    );
    assert(weight.dtype === "f32");
    const quantized = quantizeI8(weight.data, [channelsOut, channelsIn, 1, 1, 1], 0);
    const bias = fill([channelsOut], (i) => ((i % 7) - 3) * 0.25);
    assert(bias.dtype === "f32");
    // Per-pixel channel ratios of 127·{1, 0.8, −0.6, 0.4}: far from the rounding boundaries (the
    // condition of the atol = 0 claim — tests/gpu_conv3d_i8a8_test.ts)
    const fractions = [1, 0.8, -0.6, 0.4];
    const x = fill(
      [channelsIn, 1, 3, 4],
      (i) =>
        (0.5 + ((i % pixels) % 13) * 0.37) *
        fractions[(Math.floor(i / pixels) + (i % pixels)) % fractions.length],
    );
    assert(x.dtype === "f32");
    const transposed = new Float32Array(pixels * channelsIn);
    for (let c = 0; c < channelsIn; c += 1) {
      for (let p = 0; p < pixels; p += 1) transposed[p * channelsIn + c] = x.data[c * pixels + p];
    }
    assert(quantizeRowsTieMargin(transposed, pixels, channelsIn) > 1e-3, "rounding margin");
    const weightInts = new Float32Array(quantized.values.length);
    for (let i = 0; i < weightInts.length; i += 1) {
      weightInts[i] = Math.round(quantized.values[i] / quantized.scale[Math.floor(i / channelsIn)]);
    }
    const expected = referenceConv3dI8a8({
      x: x.data,
      weight: weightInts,
      weightScale: quantized.scale,
      bias: bias.data,
      dims,
    });
    const gpu = await acquireGpu();
    try {
      const model = await openModelBytes(conv3dI8a8PoisonGraph(), [
        {
          graph: GRAPH_NAME,
          initializer: "b",
          bytes: f32Bytes(bias.data),
          encoding: { codec: "f32" },
        },
        {
          graph: GRAPH_NAME,
          initializer: "w",
          bytes: quantized.bytes,
          encoding: {
            codec: "int8-sym",
            groupSize: channelsIn,
            scale: { dtype: "f32", bytes: f32Bytes(quantized.scale) },
          },
        },
      ]);
      const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
        convCompute: "a8",
      });
      try {
        const outputs = await session.run({
          seed: fill([CONV3D_TEMP_POISON_COUNT], () => CONV3D_TEMP_POISON),
          x,
        });
        // xq and xs on the poison's bytes, y alone (a poison on y's bytes would count 1)
        assertEquals(session.diagnostics().lastRun?.reuseCount, 2, "slots sharing the poison");
        assertEquals([...outputs["y"].data], [...expected], "atol = 0 match");
        // The tautology gate: a constant output would make the match say nothing
        assert(new Set([...outputs["y"].data]).size > 1, "constant output");
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "attention writes every byte of its output even with 3 dispatches (no poison remains / real GPU)",
  async () => {
    // The fused attention is 1 node = 3 dispatches (① QK → ② row statistics → ③ PV — ADR 0023).
    // Full-write applies to **the node's output O**: if the writes of ③ miss an edge tile, the
    // poison of the pool-reused buffer stays in the result (the same "1 node from several
    // dispatches" as cat).
    // [1,2,3,4] × [1,2,5,4] → O has 24 elements. S (30 elements) and the row statistics (12
    // elements) are in other size classes, so O is the only one that gets the poison buffer.
    const graph = poisonGraph(
      [1, 2, 3, 4],
      [
        { name: "q", shape: [1, 2, 3, 4] },
        { name: "k", shape: [1, 2, 5, 4] },
        { name: "v", shape: [1, 2, 5, 4] },
      ],
      { op: "attention", attrs: { scale: 0.7071067811865476 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      q: fill([1, 2, 3, 4], (i) => ((i % 7) - 3) * 0.5),
      k: fill([1, 2, 5, 4], (i) => 0.25 + (i % 5) * 0.5),
      v: fill([1, 2, 5, 4], (i) => ((i % 9) - 4) * 0.25),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
    // The check against a vacuous pass: if every output value is the same, "no poison" checks
    // nothing
    assert(new Set([...output.data]).size > 1, "constant output (the poison check is vacuous)");
  },
);

/**
 * Full-write of the **f16-storage** variant of S (option γ, phase 1).
 *
 * The node's output O stays f32, so the writes of ③PV themselves run with the same expression as
 * the f32 variant. What this looks at is two things: whether making S `array<u32>` with 2 elements
 * per word **shifted the allocation width of the node output**, and whether the two-element word
 * writes of ①QK miss anything at an edge tile.
 *
 * NOTE: with s16, S takes **half the bytes**, so its size class changes. The 2 shapes below are
 * chosen so that S / row statistics / O are all in different size classes, and O is the only one
 * that gets the poison buffer.
 * MUST: use eligible shapes (`D % 4 == 0 && N % 4 == 0`) — an ineligible shape falls back to f32
 * storage and never exercises this variant.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the f16-storage variant of S also leaves not a single word of the poison (the two-element word writes of ①QK / real GPU)",
  async () => {
    for (const testCase of [
      // The smallest shape that fits in 1 tile (S 144B / row statistics 48B / O 96B are all
      // different classes)
      { name: "attention s16 small M3 N12 D4", m: 3, n: 12, d: 4 },
      // A shape whose rows and columns both cross the 64 tile (the two-element word writes at the
      // edge tile are the target)
      { name: "attention s16 tile edge M65 N68 D20", m: 65, n: 68, d: 20 },
    ]) {
      const { name, m, n, d } = testCase;
      const { output, reuseCount } = await runPoisoned(
        poisonGraph(
          [1, 1, m, d],
          [
            { name: "q", shape: [1, 1, m, d] },
            { name: "k", shape: [1, 1, n, d] },
            { name: "v", shape: [1, 1, n, d] },
          ],
          { op: "attention", attrs: { scale: 0.7071067811865476 } },
        ),
        {
          q: fill([1, 1, m, d], (i) => ((i % 7) - 3) * 0.5),
          k: fill([1, 1, n, d], (i) => 0.25 + (i % 5) * 0.5),
          v: fill([1, 1, n, d], (i) => ((i % 9) - 4) * 0.25),
        },
        { attentionScoreStorage: "f16" },
      );
      assert(reuseCount >= 1, `${name}: pool reuse did not happen (${reuseCount})`);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${name}: poison remains`,
      );
      // The check against a vacuous pass: if every output value is the same, "no poison" checks
      // nothing
      assert(new Set([...output.data]).size > 1, `${name}: constant output`);
    }
  },
);

/**
 * Full-write of the f16 **compute** variants (ADR 0028). The family has grown, so it lists the 3
 * whose write guard is a different expression per variant (the f32 output of attention ③ / linear's
 * v4 / linear's scalar).
 *
 * NOTE: in the f16 variant the S of the fused attention takes **half the bytes**, so its size class
 * changes, and the premise that only the node output O gets the poison buffer is the same as for
 * the f32 variant.
 */
test.skipIf(!GPU_AVAILABLE || !SHADER_F16_AVAILABLE)(
  "the f16 compute variants also leave not a single word of the poison (full-write of the tile writes / real GPU)",
  async () => {
    const acquire = { shaderF16: true } as const;

    const attention = await runPoisoned(
      poisonGraph(
        [1, 2, 3, 4],
        [
          { name: "q", shape: [1, 2, 3, 4] },
          { name: "k", shape: [1, 2, 5, 4] },
          { name: "v", shape: [1, 2, 5, 4] },
        ],
        { op: "attention", attrs: { scale: 0.7071067811865476 } },
      ),
      {
        q: fill([1, 2, 3, 4], (i) => ((i % 7) - 3) * 0.5),
        k: fill([1, 2, 5, 4], (i) => 0.25 + (i % 5) * 0.5),
        v: fill([1, 2, 5, 4], (i) => ((i % 9) - 4) * 0.25),
      },
      { attentionCompute: "f16" },
      acquire,
    );
    assert(attention.reuseCount >= 1, "attention f16: pool reuse did not happen");
    assertEquals(
      [...attention.output.data].filter((value) => value === POISON),
      [],
      "attention f16: poison remains",
    );
    assert(new Set([...attention.output.data]).size > 1, "attention f16: constant output");

    // linear's write guard is a different expression for v4 and scalar (use shapes that cross 64)
    for (const testCase of [
      { name: "linear f16 v4 [65,20] × W[68,20]", m: 65, n: 68, k: 20 },
      { name: "linear f16 scalar [70,37] × W[23,37]", m: 70, n: 23, k: 37 },
    ]) {
      const { name, m, n, k } = testCase;
      const { output, reuseCount } = await runPoisoned(
        poisonGraph(
          [m, n],
          [
            { name: "x", shape: [m, k] },
            { name: "w", shape: [n, k] },
            { name: "b", shape: [n] },
          ],
          { op: "linear", attrs: {} },
        ),
        {
          x: fill([m, k], (i) => 0.25 + (i % 11) * 0.5),
          w: fill([n, k], (i) => 0.125 + (i % 7) * 0.25),
          b: fill([n], (i) => ((i % 5) - 2) * 0.5),
        },
        { linearCompute: "f16" },
        acquire,
      );
      assert(reuseCount >= 1, `${name}: pool reuse did not happen`);
      assertEquals(output.shape, [m, n], name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${name}: poison remains`,
      );
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "axis reduce leaves not a single word of the poison (full-write / real GPU)",
  async () => {
    // The axis variant writes `out[t]` exactly once with "1 thread = 1 output". A grid-stride
    // cut-off or mixing up `out_count` shows up as unwritten elements = remaining poison (unlike
    // row reduce it is not per workgroup, so the granularity of a cut-off differs too = a separate
    // target).
    // Fold axis 1 of [3,4,5] → [3,5] (the poison buffer has 15 elements, the same size class).
    const graph = poisonGraph([3, 5], [{ name: "x", shape: [3, 4, 5] }], {
      op: "sum",
      attrs: { dim: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([3, 4, 5], (i) => 1 + i),
    });
    assert(reuseCount >= 1, `pool reuse did not happen (reuseCount=${reuseCount})`);
    assertEquals(output.shape, [3, 5]);
    // Σ_j x[i,j,k] = Σ_j (1 + 20i + 5j + k) = 4 + 80i + 30 + 4k
    assertEquals(
      [...output.data],
      [34, 38, 42, 46, 50, 114, 118, 122, 126, 130, 194, 198, 202, 206, 210],
    );
    assertEquals(
      [...output.data].filter((value) => value === POISON),
      [],
      "poison remains",
    );
  },
);
