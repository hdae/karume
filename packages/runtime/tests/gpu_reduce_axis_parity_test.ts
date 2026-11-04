// The **bit-identity check** of the axis reduce variant.
//
// On the same input buffer it runs
//
//   ① the old form (a `permute` moves the reduced axis to the last dimension → row reduce
//      `reduce:v2:…:wg256`)
//   ② the axis reduce variant at the production split (`reduce:v2:…:axis:s<split>:wg256` —
//      `AXIS_REDUCE_SPLIT` threads share one output, no permute)
//
// on **the same real GPU** and compares the outputs as **f32 bit patterns**. The point is exact
// `Uint32Array` equality rather than allclose: a **change of the reduction order** is a few ulp, which
// a tolerance lets through, so only this check sees it.
//
// Why it holds: the row reduce's order is "a 256-lane grid-stride fold + a 256-wide bit-reversed
// binary tree". Each axis thread rebuilds one aligned subtree of that tree with its own bit-reversed
// carry stack, and `log2(split)` levels in workgroup memory combine the subtree roots in the tree's
// operand order. The `+0.0` (identity) fold is not skipped. The tree itself is checked symbolically,
// without a GPU, in tests/reduce_axis_tree_test.ts (ADR 0035 addendum 2026-10-26).
//
// MUST: the check must not become vacuous. ① and ② are **different pipelines with different WGSL**, and their
// generator functions differ too.
// MUST: always carry at least one **axis case with a cycle length of 3 or more** (the countermeasure for the
// ACTIVE_DESIGN pitfall "every permute of the measured graphs is an involution" — a mixed-up axis reorder goes
// unnoticed if only involution cases are tested).
// MUST: rows (= number of output elements) hit **both sides** of 65,535. On the row reduce side the row count
// becomes the workgroup count directly, so exceeding the limit takes the degraded path (the production path).
// MUST: carry **NaN cases** for `amax` / `amin` (one on each side of a reduction length of 256).
// The ADR 0020 decision 1 level of "confirm by actually running on a real GPU" exists only here for the axis
// variant (the NaN cases of tests/gpu_ops_test.ts only hit the last dimension of rank 2 = the row variant).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import {
  AXIS_REDUCE_SPLIT,
  axisReduceKey,
  axisReduceOutputsPerGroup,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../src/codegen/reduce.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../src/codegen/strided.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import type { IrDtype } from "../src/format/ir.ts";
import { numel, type ReduceOpName } from "../src/ops.ts";
import { allclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceRowReduce, refTensor } from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * Tolerance for the comparison with the CPU reference. The reference reduces in JS f64, so its rounding differs
 * from the two-stage f32 reduction and bit identity cannot hold in principle (`amax` / `amin` / bool `sum` have
 * no rounding, so they measure 0).
 *
 * Measured (2026-08-04, a plain comparison with `atol=rtol=0`, the 9 NaN-free cases of {@link CASES} below):
 * `[300,7] axis=0` maxAbs **7.15e-7** / maxRel **2.37e-6** (worst),
 * `[1,96,255,257]` 5.96e-7 / 1.67e-6, `[1,384,128,128]` 5.36e-7 / 7.01e-7.
 * atol 5e-6 is about 7.0× the worst and rtol 2e-5 about 8.4×. **The error of an implementation bug is O(1)**, so
 * the orders of magnitude are far apart.
 */
const CPU_REFERENCE_TOLERANCE = { atol: 5e-6, rtol: 2e-5 } as const;

/**
 * The CPU tolerance of the `[1024,2,32,32]` case. Its columns of `SIGNED` nearly cancel (the
 * reference is near 0 for many outputs, so rtol does not help) while f32 partial sums over 1024
 * terms carry rounding of a few ulp of values in the tens. Measured on 2026-10-26
 * (`atol = rtol = 0`): `[1024,2,32,32]` maxAbs 1.24e-5, past the default atol; atol 1e-4 is about 8
 * times that. `[1024,1,16,16]` (maxAbs 1.43e-6) fits {@link CPU_REFERENCE_TOLERANCE} and keeps it.
 * An implementation bug is still O(1). The bit comparison with the row reduce, not this check, is
 * the gate.
 */
const C1024_CPU_REFERENCE_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 2e-5 };

/** A deterministic data sequence (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
/** A bool input (counting the true values — not a 3-element period, so that 0/1 are not skewed). */
const BOOLEAN = (i: number): number => ((i * 5) % 7 < 3 ? 1 : 0);

/**
 * Input generator that scatters NaN along the reduced axis (NaN propagation of `amax` / `amin` — ADR 0020 decision 1).
 *
 * For each reduction column of one output element, the NaN position is distributed over 4 cases: **first /
 * middle / last / none**. MUST: always mix in **NaN-free control columns** — checking only that the positions
 * match does not catch an implementation that "always returns NaN" (the same reason as `NAN_REDUCE_ROWS` in
 * tests/gpu_ops_test.ts).
 */
/**
 * Input generator that makes reduction columns **entirely ±Infinity** (the identity check of `amax` / `amin` — E-R2-2).
 *
 * Of the reduction columns of one output element, even-numbered ones are all `sign`·Infinity and odd-numbered ones
 * are finite control columns.
 * MUST: the `amax` of an all −Inf column is **−F32_MAX** on both the GPU and the CPU reference (fold starting from
 * the identity — the MUST of `reference/ops.ts`). When the reference started from the first element it returned
 * −Infinity and disagreed with the GPU on every such column.
 */
const infAlongAxis = (
  shape: readonly number[],
  axis: number,
  sign: -1 | 1,
): ((i: number) => number) => {
  const axisLen = shape[axis];
  const inner = numel(shape.slice(axis + 1));
  return (i: number): number => {
    const out = Math.floor(i / (axisLen * inner)) * inner + (i % inner);
    return out % 2 === 0 ? sign * Number.POSITIVE_INFINITY : SIGNED(i);
  };
};

const nanAlongAxis = (shape: readonly number[], axis: number): ((i: number) => number) => {
  const axisLen = shape[axis];
  const inner = numel(shape.slice(axis + 1));
  /** NaN position per reduction column (-1 = a NaN-free control). */
  const positions = [0, axisLen >> 1, axisLen - 1, -1];
  return (i: number): number => {
    // Decompose the input index i = (outer·axis_len + c)·inner + j
    const c = Math.floor(i / inner) % axisLen;
    const out = Math.floor(i / (axisLen * inner)) * inner + (i % inner);
    return c === positions[out % positions.length] ? Number.NaN : SIGNED(i);
  };
};

type ParityCase = {
  readonly name: string;
  readonly shape: readonly number[];
  /** The reduced axis (the last dimension belongs to the row reduce, so here it is always before the last dimension). */
  readonly axis: number;
  readonly op: ReduceOpName;
  readonly dtype: IrDtype;
  /** Whether to use a NaN input ({@link nanAlongAxis}). By default a deterministic NaN-free sequence. */
  readonly nan?: true;
  /** Makes reduction columns entirely ±Infinity ({@link infAlongAxis} — the identity check). */
  readonly inf?: -1 | 1;
  /** The CPU-reference tolerance when it is not {@link CPU_REFERENCE_TOLERANCE}. */
  readonly tolerance?: Tolerance;
};

const readbackBits = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Uint32Array<ArrayBuffer>> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Uint32Array(copy, 0, count);
};

/**
 * Runs "permute + row reduce" and "axis reduce" on **the same input buffer** and returns the output bit patterns
 * as a pair. Writing from the host twice would turn "a difference in writes" into "a difference in kernels", so
 * only one input is allocated and bound to both.
 */
const runBoth = async (
  gpu: GpuContext,
  testCase: ParityCase,
): Promise<{
  readonly viaPermute: Uint32Array;
  readonly viaAxis: Uint32Array;
  readonly permuteDims: readonly number[];
  readonly reference: Float32Array | Int32Array;
}> => {
  const { shape, axis, op, dtype } = testCase;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const generator =
      testCase.nan === true
        ? nanAlongAxis(shape, axis)
        : testCase.inf !== undefined
          ? infAlongAxis(shape, axis, testCase.inf)
          : dtype === "bool"
            ? BOOLEAN
            : SIGNED;
    const x = fill(shape, generator, dtype === "bool" ? "bool" : "f32");
    const upload = (data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, data.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const xBuffer = upload(x.data);

    const axisLen = shape[axis];
    const outShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    const outCount = numel(outShape);
    const inner = numel(shape.slice(axis + 1));
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;

    const allocOut = (): GPUBuffer => {
      const buffer = arena.allocRegion(Math.max(4, outCount * 4));
      return buffer;
    };
    const bind = (
      layout: GPUBindGroupLayout,
      params: Uint32Array<ArrayBuffer>,
      usage: number,
      input: GPUBuffer,
      out: GPUBuffer,
    ): GPUBindGroup => {
      const paramsBuffer = arena.allocHostWritten(params.byteLength, usage);
      gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
      return gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: out } },
        ],
      });
    };

    // ① a permute that moves the reduced axis to the last dimension (the form the current exporter used to emit) → row reduce
    const permuteDims = [...shape.keys()].filter((d) => d !== axis).concat(axis);
    const permutedShape = permuteDims.map((d) => shape[d]);
    const permuted = arena.allocRegion(Math.max(4, numel(shape) * 4));
    const { pipeline: stridedPipeline, layout: stridedLayout } = await cache.get(
      stridedKey({ dtype }),
      stridedWgsl({ dtype }),
    );
    const stridedGroup = gpu.device.createBindGroup({
      layout: stridedLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: (() => {
              const params = stridedParams(permutedShape, permuteSrcStrides(shape, permuteDims), 0);
              const buffer = arena.allocHostWritten(params.byteLength, STORAGE_IN);
              gpu.device.queue.writeBuffer(buffer, 0, params);
              return buffer;
            })(),
          },
        },
        { binding: 1, resource: { buffer: xBuffer } },
        { binding: 2, resource: { buffer: permuted } },
      ],
    });
    scheduler.dispatch(
      stridedPipeline,
      stridedGroup,
      [gridStrideWorkgroups(numel(shape), STRIDED_WORKGROUP_SIZE, limit), 1, 1],
      stridedKey({ dtype }),
    );

    const rowKey = reduceKey({ op, dtype });
    const { pipeline: rowPipeline, layout: rowLayout } = await cache.get(
      rowKey,
      reduceWgsl({ op, dtype }),
    );
    const rowOut = allocOut();
    scheduler.dispatch(
      rowPipeline,
      bind(rowLayout, reduceParams(outCount, axisLen), UNIFORM_IN, permuted, rowOut),
      [gridStrideWorkgroups(outCount, 1, limit), 1, 1],
      rowKey,
    );

    // ② the axis reduce variant (no permute). The split is passed explicitly, as `buildRowReduce`
    // does, so the key, the WGSL and the grid are the production ones, not the parameter defaults.
    const axisKey = axisReduceKey({ op, dtype }, AXIS_REDUCE_SPLIT);
    const { pipeline: axisPipeline, layout: axisLayout } = await cache.get(
      axisKey,
      axisReduceWgsl({ op, dtype }, AXIS_REDUCE_SPLIT),
    );
    const axisOut = allocOut();
    scheduler.dispatch(
      axisPipeline,
      bind(axisLayout, axisReduceParams(outCount, axisLen, inner), UNIFORM_IN, xBuffer, axisOut),
      [gridStrideWorkgroups(outCount, axisReduceOutputsPerGroup(AXIS_REDUCE_SPLIT), limit), 1, 1],
      axisKey,
    );

    await scheduler.flush();
    const reference = referenceRowReduce(op, refTensor(shape, x.data), axis);
    return {
      viaPermute: await readbackBits(gpu.device, rowOut, outCount),
      viaAxis: await readbackBits(gpu.device, axisOut, outCount),
      permuteDims,
      reference: reference.data as Float32Array | Int32Array,
    };
  } finally {
    await arena.destroy();
  }
};

const CASES: readonly ParityCase[] = [
  {
    // A real shape from the census (VAE decoder 1024px): rows 16,384 / dim 384 (11 nodes).
    // rows < 65,535, so the row reduce side does not degrade.
    name: "real shape [1,384,128,128] axis=1 (rows 16,384 / dim 384)",
    shape: [1, 384, 128, 128],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // A real census shape: rows 65,536 / dim 192 (reduced from the 6-node side). **rows > 65,535**, so the row
    // reduce side takes the degraded grid-stride path (the production path).
    name: "real shape [1,192,256,256] axis=1 (rows 65,536 > 65,535 / dim 192)",
    shape: [1, 192, 256, 256],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // rows = exactly 65,535 (the clamp boundary itself). dim 96 is the shape that holds 58% of the elements in the
    // census; against a workgroup width of 256, 160 lanes go round the tree as identity.
    name: "real shape [1,96,255,257] axis=1 (rows 65,535 = boundary / dim 96)",
    shape: [1, 96, 255, 257],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // rows more than twice the limit (makes the degraded path go round twice for sure). dim 96.
    name: "real shape [1,96,512,257] axis=1 (rows 131,584 / dim 96)",
    shape: [1, 96, 512, 257],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // **An axis case with a cycle length of 3 or more (MUST)**: reducing the middle axis of rank 4 gives the permute
    // [0,2,3,1] with inverse [0,3,1,2] (= not itself). And all 4 dimensions differ, so mixing up any of outer /
    // inner / axis_len breaks the values.
    name: "cycle length 3 [2,3,4,5] axis=1 (permute [0,2,3,1] is not an involution)",
    shape: [2, 3, 4, 5],
    axis: 1,
    op: "sum",
    dtype: "f32",
  },
  {
    // The reduction length crosses 256 (the per-slot scan goes round twice = the carry-stack leaves are not "1 element").
    // It also covers a reduction of the leading axis (outer = 1).
    name: "reduction length 300 > 256 [300,7] axis=0 (slot scan goes round twice)",
    shape: [300, 7],
    axis: 0,
    op: "sum",
    dtype: "f32",
  },
  {
    // The Wan2.2 decoder's smallest channel norm: P = 256 outputs (exactly one workgroup of the
    // single-thread form, several at the production split), C = 1024 (4 sweeps per slot).
    name: "[1024,1,16,16] axis=0 (P 256 / C 1024)",
    shape: [1024, 1, 16, 16],
    axis: 0,
    op: "sum",
    dtype: "f32",
  },
  {
    // The Wan2.2 next chunk's 32×32 stage: P = 2,048, C = 1024, a time axis of 2 inside P.
    name: "[1024,2,32,32] axis=0 (P 2,048 / C 1024)",
    shape: [1024, 2, 32, 32],
    axis: 0,
    op: "sum",
    dtype: "f32",
    tolerance: C1024_CPU_REFERENCE_TOLERANCE,
  },
  {
    // P = 1,023 is not a multiple of the outputs per group (the last workgroup has idle lanes that
    // still take part in the barriers), and C = 300 > 256 (some slots sweep twice).
    name: "[300,33,31] axis=0 (P 1,023, odd / C 300)",
    shape: [300, 33, 31],
    axis: 0,
    op: "sum",
    dtype: "f32",
  },
  {
    // amax (the NaN-propagating fold function and the ±F32_MAX identity). The axis is the middle one of rank 3.
    name: "amax [5,9,11] axis=1",
    shape: [5, 9, 11],
    axis: 1,
    op: "amax",
    dtype: "f32",
  },
  {
    // amin has the same skeleton (only the sign of the identity differs — mixing them up makes every element +F32_MAX)
    name: "amin [5,9,11] axis=1",
    shape: [5, 9, 11],
    axis: 1,
    op: "amin",
    dtype: "f32",
  },
  {
    // **The real-GPU check of NaN propagation (ADR 0020 decisions 1 and 4)**. The string check of the generated
    // WGSL covers the axis variant too, but the fold failure ADR 0020 names as "strings are not enough" (a backend
    // folding `nan_max` into a plain `max`) is visible only by running. Reduction length 9 < 256 = the per-slot
    // scan goes round once (a leaf is 1 element).
    name: "amax NaN [5,9,11] axis=1 (reduction length 9 < 256 / scan goes round once)",
    shape: [5, 9, 11],
    axis: 1,
    op: "amax",
    dtype: "f32",
    nan: true,
  },
  {
    // amin has the same skeleton (swallowing NaN turns it into the +F32_MAX identity)
    name: "amin NaN [5,9,11] axis=1 (reduction length 9 < 256 / scan goes round once)",
    shape: [5, 9, 11],
    axis: 1,
    op: "amin",
    dtype: "f32",
    nan: true,
  },
  {
    // Reduction length 300 > 256 = the per-slot scan goes round twice and the carry-stack actually stacks levels.
    // This checks that NaN passes through **both** the scan loop and the fold loop.
    name: "amax NaN [300,7] axis=0 (reduction length 300 > 256 / scan goes round twice)",
    shape: [300, 7],
    axis: 0,
    op: "amax",
    dtype: "f32",
    nan: true,
  },
  {
    name: "amin NaN [300,7] axis=0 (reduction length 300 > 256 / scan goes round twice)",
    shape: [300, 7],
    axis: 0,
    op: "amin",
    dtype: "f32",
    nan: true,
  },
  {
    // **The identity check (E-R2-2)**: the `amax` of a reduction column that is entirely −Inf returns −F32_MAX (the
    // GPU folds starting from the identity. Unless the CPU reference starts from the same point, it returns
    // −Infinity and disagrees on every such column — finite control columns are mixed in to check the `SIGNED` side too).
    name: "amax all −Inf columns [4,9,6] axis=1 (identity −F32_MAX check)",
    shape: [4, 9, 6],
    axis: 1,
    op: "amax",
    dtype: "f32",
    inf: -1,
  },
  {
    name: "amin all +Inf columns [4,9,6] axis=1 (identity +F32_MAX check)",
    shape: [4, 9, 6],
    axis: 1,
    op: "amin",
    dtype: "f32",
    inf: 1,
  },
  {
    // bool → i32 (the number of true values). The accumulator type and the truthiness of `load` are in the axis variant too.
    name: "bool sum [3,13,5] axis=1 (number of true values → i32)",
    shape: [3, 13, 5],
    axis: 1,
    op: "sum",
    dtype: "bool",
  },
];

/**
 * The I2V encoder's C = 4 shortcut sum (outer > 1): P = 40·167·167 = 1,115,560 outputs. At the
 * production split its natural axis grid is past 65,535 workgroups, so on an adapter with that
 * per-dimension limit the grid clamps and some workgroups run the grid-stride loop (with its
 * barriers) for more than one block. The row reduce side clamps as well.
 */
const CLAMPED_CASE: ParityCase = {
  name: "[40,4,167,167] axis=1 (P 1,115,560, the axis grid clamps / C 4)",
  shape: [40, 4, 167, 167],
  axis: 1,
  op: "sum",
  dtype: "f32",
};

/** Makes bit-pattern mismatches (the first 4) human-readable. */
const firstMismatches = (a: Uint32Array, b: Uint32Array): readonly string[] => {
  const found: string[] = [];
  for (let i = 0; i < a.length && found.length < 4; i += 1) {
    if (a[i] !== b[i]) found.push(`[${i}] 0x${a[i].toString(16)} vs 0x${b[i].toString(16)}`);
  }
  return found;
};

test.skipIf(!GPU_AVAILABLE)(
  "the axis reduce is **bit-identical** to permute + row reduce (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    let sawNonInvolution = false;
    try {
      for (const testCase of CASES) {
        const result = await runBoth(gpu, testCase);
        const { viaPermute, viaAxis, permuteDims, reference } = result;
        assertEquals(
          firstMismatches(viaPermute, viaAxis),
          [],
          `${testCase.name}: bit patterns differ from permute + row reduce`,
        );
        // Vacuity check (if the output is constant, the match verifies nothing)
        assert(
          new Set(viaAxis).size > 1,
          `${testCase.name}: the output is constant (the bit identity is vacuous)`,
        );
        // Detector for both kernels being wrong **together** (bit comparison alone cannot notice it)
        const values =
          testCase.dtype === "bool"
            ? new Int32Array(viaAxis.buffer, viaAxis.byteOffset, viaAxis.length)
            : new Float32Array(viaAxis.buffer, viaAxis.byteOffset, viaAxis.length);
        if (testCase.nan === true) {
          // MUST: allclose fails NaN on either side, so it cannot be used for NaN cases —
          // compare by **matching isNaN patterns + exact equality of the non-NaN elements**
          // (the same form as the NaN check of the row variant in tests/gpu_ops_test.ts).
          const actualValues = [...values];
          const expectedValues = [...reference];
          const actualNan = actualValues.map((value) => Number.isNaN(value));
          const expectedNan = expectedValues.map((value) => Number.isNaN(value));
          assert(expectedNan.includes(true), `${testCase.name}: the reference has no NaN at all`);
          // MUST: have control elements (an implementation that "always returns NaN" passes a position match)
          assert(expectedNan.includes(false), `${testCase.name}: no control elements`);
          assertEquals(actualNan, expectedNan, `${testCase.name}: NaN positions`);
          // The non-NaN side is `amax` / `amin`, so no rounding difference enters (NaN never compares equal, so
          // only those elements are flattened to 0 before checking)
          const finiteOnly = (
            source: readonly number[],
            isNan: readonly boolean[],
          ): readonly number[] => source.map((value, index) => (isNan[index] ? 0 : value));
          assertEquals(
            finiteOnly(actualValues, actualNan),
            finiteOnly(expectedValues, expectedNan),
            `${testCase.name}: non-NaN values`,
          );
        } else {
          const report = allclose(values, reference, testCase.tolerance ?? CPU_REFERENCE_TOLERANCE);
          assert(
            report.pass,
            `${testCase.name}: disagrees with the CPU reference (maxAbs ${report.maxAbsError} @${report.worstIndex} / violations ${report.failCount})`,
          );
        }
        // At least one case with a cycle length of 3 or more (inverse not itself) must be covered
        const inverse = permuteDims.map((_, d) => permuteDims.indexOf(d));
        if (inverse.some((value, index) => value !== permuteDims[index])) sawNonInvolution = true;
      }
      assert(
        sawNonInvolution,
        "only involution permute cases (a mixed-up axis would go unnoticed)",
      );
    } finally {
      gpu.destroy();
    }
  },
);

// The clamp is a property of the adapter's limit, not of the shape: an adapter that grants more than
// the natural grid would run the case unclamped, so it is skipped there with the reason instead of
// failing (or passing without the clamp it exists for).
test.skipIf(!GPU_AVAILABLE)(
  "the axis reduce matches permute + row reduce bit for bit when its grid clamps (real GPU)",
  async (context) => {
    const gpu = await acquireGpu();
    try {
      const { shape, axis } = CLAMPED_CASE;
      const natural = Math.ceil(
        numel(shape) / shape[axis] / axisReduceOutputsPerGroup(AXIS_REDUCE_SPLIT),
      );
      const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
      context.skip(
        natural <= limit,
        `maxComputeWorkgroupsPerDimension ${limit} admits the natural grid of ${natural} workgroups, so the case would not clamp`,
      );
      const { viaPermute, viaAxis, reference } = await runBoth(gpu, CLAMPED_CASE);
      assertEquals(
        firstMismatches(viaPermute, viaAxis),
        [],
        "bits differ from permute + row reduce",
      );
      assert(new Set(viaAxis).size > 1, "the output is constant (the bit match is vacuous)");
      const values = new Float32Array(viaAxis.buffer, viaAxis.byteOffset, viaAxis.length);
      const report = allclose(values, reference, CPU_REFERENCE_TOLERANCE);
      assert(
        report.pass,
        `differs from the CPU reference (maxAbs ${report.maxAbsError} @${report.worstIndex} / ${report.failCount} failing)`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Which kernel the executor dispatches. The last dimension = the row reduce (the existing key), any other = the
 * axis variant.
 *
 * MUST: check by **the pipeline key that actually ran**, not by values. Both kernels are correct, so matching
 * outputs are no evidence of the dispatch (the same discipline as ADR 0024 / 0030).
 */
const reduceKeysUsed = async (
  shape: readonly number[],
  axis: number,
): Promise<ReadonlySet<string>> => {
  const gpu = await acquireGpu();
  const outShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
  const graph = singleOpDeclaration("sum", [shape], [outShape], { attrs: { dim: axis } });
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    await session.run({ x0: fill(shape, SIGNED) });
    return new Set(
      requireCensus(session.diagnostics().lastRunPipelines, "sum").map((row) => row.key),
    );
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the executor dispatches to 2 kernels by the reduced axis (last dimension = row / any other = axis; real GPU)",
  async () => {
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 2),
      new Set([reduceKey({ op: "sum", dtype: "f32" })]),
    );
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 1),
      new Set([axisReduceKey({ op: "sum", dtype: "f32" }, AXIS_REDUCE_SPLIT)]),
    );
    assertEquals(
      await reduceKeysUsed([4, 6, 8], 0),
      new Set([axisReduceKey({ op: "sum", dtype: "f32" }, AXIS_REDUCE_SPLIT)]),
    );
  },
);
