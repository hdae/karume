// **Oracle checks** of deform_conv2d (DCNv2 — ADR 0055). A new primitive has no existing path to pair
// with, so the case for its correctness is built in 3 parts:
//
//   (1) degenerate = **bit-identical** to conv2d (`offset` all 0, `mask` all 1). Compared by exact
//       `Uint32Array` equality rather than a tolerance — a change in the reduction order (hoisting the
//       offset read out of the ic loop, changing what the mask multiplies, adding the bias at the
//       store) moves only the rounding sequence, so allclose would stay silent. The substance of
//       ADR 0055 decision 2 / decision 4.
//   (2) **The boundary semantics** (centre out of range -> the whole tap is 0 / inside, an
//       out-of-range corner is 0 for that corner only), compared by **exact equality** with
//       hand-computed values of a 1×1 kernel. A border-clamp implementation turns red only here
//       (with allclose one can build shapes where the difference between clamp and zero fill hides
//       in the tolerance).
//   (3) **NaN propagation**. A range test of the positive form alone would turn NaN into a 0
//       contribution, a silently wrong value, so this pins that NaN really comes out on a real GPU
//       (in the line of ADR 0020 — the driver may fold the test away, so only a measurement can
//       guarantee it).
//
// The comparison with torch (torchvision) itself is the golden `deform_conv2d_block` (e2e_golden_test.ts).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  CONV2D_WORKGROUP_SIZE,
  conv2dKey,
  conv2dParams,
  conv2dWgsl,
} from "../src/kernels/conv2d.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../src/kernels/deform-conv2d.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/**
 * MUST: the bias is non-zero. In elements whose output is exactly 0, the `+0.0` deform adds flips
 * conv2d's `−0.0` to `+0.0` (the only numeric difference — the same shape as ADR 0055 decision 4 /
 * ADR 0024 decision 3).
 */
const BIAS = (i: number): number => 0.375 + (i % 5) * 0.25;

type ParityCase = {
  readonly name: string;
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly paddingH: number;
  readonly paddingW: number;
};

const PARITY_CASES: readonly ParityCase[] = [
  {
    // The same shape as BiRefNet's k=3 branch (same-size output). Cin ≠ Cout and Kh ≠ Kw also catch an axis mix-up
    name: "deform [1,3,6,7] * W[5,3,3,2] padding=[1,0]",
    batch: 1,
    channelsIn: 3,
    channelsOut: 5,
    heightIn: 6,
    widthIn: 7,
    kernelH: 3,
    kernelW: 2,
    paddingH: 1,
    paddingW: 0,
  },
  {
    // k=1 (ASPP's aspp1). padding 0, so not a single tap is out of range
    name: "deform k=1 [2,4,5,6] * W[3,4,1,1] padding=[0,0]",
    batch: 2,
    channelsIn: 4,
    channelsOut: 3,
    heightIn: 5,
    widthIn: 6,
    kernelH: 1,
    kernelW: 1,
    paddingH: 0,
    paddingW: 0,
  },
  {
    // A large k (the same shape as BiRefNet's k=7 branch). The padding is close to the input length, and out-of-range taps dominate
    name: "deform k=5 [1,2,5,5] * W[4,2,5,5] padding=[2,2]",
    batch: 1,
    channelsIn: 2,
    channelsOut: 4,
    heightIn: 5,
    widthIn: 5,
    kernelH: 5,
    kernelW: 5,
    paddingH: 2,
    paddingW: 2,
  },
];

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
 * Runs deform (offset all 0, mask all 1) and the conv2d direct kernel on **the same input buffers**
 * and returns the output bit patterns as a pair. For x / weight / bias both dispatches bind the same
 * GPUBuffer (writing them twice on the host would turn "a difference in the writes" into "a
 * difference in the kernels").
 *
 * MUST: the counterpart is conv2d's **direct kernel** (the side kept permanently for `groups > 1`).
 * implicit GEMM is tiled and folds its fmas differently, so comparing against it would not show
 * that "the order is the same".
 */
const runBoth = async (
  gpu: GpuContext,
  testCase: ParityCase,
): Promise<{ readonly deform: Uint32Array; readonly conv: Uint32Array }> => {
  const heightOut = testCase.heightIn + 2 * testCase.paddingH - (testCase.kernelH - 1);
  const widthOut = testCase.widthIn + 2 * testCase.paddingW - (testCase.kernelW - 1);
  const taps = testCase.kernelH * testCase.kernelW;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const x = fill(
      [testCase.batch, testCase.channelsIn, testCase.heightIn, testCase.widthIn],
      SIGNED,
    );
    const weight = fill(
      [testCase.channelsOut, testCase.channelsIn, testCase.kernelH, testCase.kernelW],
      WEIGHT,
    );
    const bias = fill([testCase.channelsOut], BIAS);
    const offset = fill([testCase.batch, 2 * taps, heightOut, widthOut], () => 0);
    const mask = fill([testCase.batch, taps, heightOut, widthOut], () => 1);

    const upload = (data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, data.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const xBuffer = upload(x.data);
    const weightBuffer = upload(weight.data);
    const biasBuffer = upload(bias.data);
    const offsetBuffer = upload(offset.data);
    const maskBuffer = upload(mask.data);

    const count = testCase.batch * testCase.channelsOut * heightOut * widthOut;
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
    const outputs: GPUBuffer[] = [];
    const bind = (
      layout: GPUBindGroupLayout,
      params: Uint32Array<ArrayBuffer>,
      inputs: readonly GPUBuffer[],
    ): GPUBindGroup => {
      const paramsBuffer = arena.allocHostWritten(params.byteLength, UNIFORM_IN);
      gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
      const out = arena.allocRegion(Math.max(4, count * 4));
      outputs.push(out);
      return gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          ...inputs.map((buffer, index) => ({
            binding: index + 1,
            resource: { buffer },
          })),
          { binding: inputs.length + 1, resource: { buffer: out } },
        ],
      });
    };

    const { pipeline: deformPipeline, layout: deformLayout } = await cache.get(
      DEFORM_CONV2D_KEY,
      DEFORM_CONV2D_WGSL,
    );
    const deformGroup = bind(
      deformLayout,
      deformConv2dParams({
        batch: testCase.batch,
        channelsIn: testCase.channelsIn,
        channelsOut: testCase.channelsOut,
        heightIn: testCase.heightIn,
        widthIn: testCase.widthIn,
        heightOut,
        widthOut,
        kernelH: testCase.kernelH,
        kernelW: testCase.kernelW,
        paddingH: testCase.paddingH,
        paddingW: testCase.paddingW,
      }),
      [xBuffer, weightBuffer, offsetBuffer, maskBuffer, biasBuffer],
    );
    scheduler.dispatch(
      deformPipeline,
      deformGroup,
      [gridStrideWorkgroups(count, DEFORM_CONV2D_WORKGROUP_SIZE, limit), 1, 1],
      DEFORM_CONV2D_KEY,
    );

    const convKey = conv2dKey("f32");
    const { pipeline: convPipeline, layout: convLayout } = await cache.get(
      convKey,
      conv2dWgsl("f32"),
    );
    const convGroup = bind(
      convLayout,
      conv2dParams({
        batch: testCase.batch,
        channelsIn: testCase.channelsIn,
        channelsOut: testCase.channelsOut,
        heightIn: testCase.heightIn,
        widthIn: testCase.widthIn,
        heightOut,
        widthOut,
        kernelH: testCase.kernelH,
        kernelW: testCase.kernelW,
        strideH: 1,
        strideW: 1,
        paddingH: testCase.paddingH,
        paddingW: testCase.paddingW,
        dilationH: 1,
        dilationW: 1,
        groups: 1,
      }),
      [xBuffer, weightBuffer, biasBuffer],
    );
    scheduler.dispatch(
      convPipeline,
      convGroup,
      [gridStrideWorkgroups(count, CONV2D_WORKGROUP_SIZE, limit), 1, 1],
      convKey,
    );

    await scheduler.flush();
    const [deform, conv] = await Promise.all(
      outputs.map((buffer) => readbackBits(gpu.device, buffer, count)),
    );
    return { deform, conv };
  } finally {
    await arena.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "deform_conv2d with offset 0 and mask 1 is bit-identical to the conv2d direct kernel (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of PARITY_CASES) {
        const { deform, conv } = await runBoth(gpu, testCase);
        assert(deform.length > 0, `${testCase.name}: the output is empty`);
        // Exact Uint32 equality (a tolerance would stay silent about a change in the rounding sequence)
        assertEquals(
          [...deform],
          [...conv],
          `${testCase.name}: the bit pattern of the degenerate case differs from conv2d`,
        );
        // The vacuity check: the output is not all bias (= the reduction never ran)
        assert(
          new Set(deform).size > 1,
          `${testCase.name}: the output is a single value (a vacuous comparison in which the reduction has no effect)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/** Runs deform_conv2d once in a single-node graph (the executor path). */
const runDeform = async (
  gpu: GpuContext,
  inputs: readonly Tensor[],
  outShape: readonly number[],
  padding: readonly [number, number],
): Promise<Tensor> => {
  const graph = singleOpDeclaration(
    "deform_conv2d",
    inputs.map((input) => input.shape),
    [outShape],
    { attrs: { padding: [...padding] } },
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    const named: Record<string, Tensor> = {};
    inputs.forEach((input, index) => {
      named[`x${index}`] = input;
    });
    return (await session.run(named))["y"];
  } finally {
    await session.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the deform_conv2d boundary is zero fill, not clamp (exact equality with hand-computed values / real GPU)",
  async () => {
    // A 1×1 kernel, Cin=Cout=1, weight 1, bias 0, so the output is exactly
    // `mask · bilinear(x, oy + off_y, ox + off_x)` (the values are exact in f32).
    const x = fill([1, 1, 2, 2], (i) => 1 + i);
    const weight = fill([1, 1, 1, 1], () => 1);
    const bias = fill([1], () => 0);
    const mask = fill([1, 1, 2, 2], () => 1);
    const gpu = await acquireGpu();
    try {
      const run = async (shiftY: number, shiftX: number): Promise<readonly number[]> => {
        // offset is `[4 elements of the y plane, 4 elements of the x plane]` (even channels = y / odd = x)
        const offset = fill([1, 2, 2, 2], (i) => (i < 4 ? shiftY : shiftX));
        const out = await runDeform(gpu, [x, weight, offset, mask, bias], [1, 1, 2, 2], [0, 0]);
        return [...out.data];
      };
      // Inside the boundary — only the lower corners are out of range (where clamp would give 3, 4, this gives 1.5, 2)
      assertEquals(await run(0.5, 0), [2, 3, 1.5, 2]);
      // The centre at −1.5 ≤ −1 -> the whole tap is 0 (clamp would give 1, 2)
      assertEquals(await run(-1.5, 0), [0, 0, 0.5, 1]);
      // MUST: even = y / odd = x. Pinned with a shape in which swapping them gives entirely different values
      assertEquals(await run(1, 0), [3, 4, 0, 0]);
      assertEquals(await run(0, 1), [2, 0, 4, 0]);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a NaN offset in deform_conv2d does not fall to 0 and propagates to the output (real GPU)",
  async () => {
    // A range test of the positive form (`> −1 && < in`) falls to the false side on NaN, so unless NaN is
    // split off first by a bit-pattern test, **this output becomes 0** (a silently wrong value).
    const x = fill([1, 1, 2, 2], (i) => 1 + i);
    const weight = fill([1, 1, 1, 1], () => 1);
    const bias = fill([1], () => 0);
    const mask = fill([1, 1, 2, 2], () => 1);
    // Only the first element of the y plane is NaN, the second is +Inf (Inf is "out of range = 0", as in torch)
    const offset = fill([1, 2, 2, 2], (i) => {
      if (i === 0) return Number.NaN;
      if (i === 1) return Number.POSITIVE_INFINITY;
      return 0;
    });
    const gpu = await acquireGpu();
    try {
      const out = await runDeform(gpu, [x, weight, offset, mask, bias], [1, 1, 2, 2], [0, 0]);
      assertEquals(Number.isNaN(out.data[0]), true, "a NaN offset fell to 0");
      assertEquals(out.data[1], 0, "a ±Inf offset is out of range = 0 (not NaN)");
      assertEquals([...out.data.slice(2)], [3, 4], "the other elements are not caught up in it");
    } finally {
      gpu.destroy();
    }
  },
);
