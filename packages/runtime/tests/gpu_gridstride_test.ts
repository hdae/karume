// Checks that grid-stride really works by deliberately cutting the dispatch count (real GPU).
//
// Reaching the 1-D workgroup limit (65535 by default) with real element counts needs buffers of
// tens of MB. Whether a kernel copes with a reduced dispatch can be checked equivalently by asking
// "does it fill the whole range with far fewer workgroups than needed", so small inputs get a cut
// dispatch.
//
// MUST: list **every kernel family** that claims grid-stride here. The tests with measured shapes
// (gpu_ops_test.ts) only pass "fairly large inputs", where the needed workgroups really are
// assigned, so they never take the reduced path — making `stride` a constant instead of
// `nwg.x * WG` stays green there.
// The element-wise families (elementwise / strided read and write / gather / embedding /
// masked_fill / the conv family / pad / flip / upsample_bilinear2d / deform_conv2d / **axis
// reduce**) exercise 1-D grid-stride over `gid.x`,
// the row-wise families (row reduce / **argmax** / **topk** / softmax /
// layer_norm / rms_norm / **gru_scan**) exercise the row loop that advances `workgroup_id.x` by
// `num_workgroups.x` (a "row" of gru_scan is a batch element; the time loop is inside the kernel).
// **Both reduce variants are listed** (last dimension = row-wise / otherwise = element-wise; they
// run differently). cumsum is **grid-stride over `gid.x` with a row as the unit** (1 invocation = a
// sequential scan of 1 row), and the needed count is `ceil(rows / workgroup size)`, not the row
// count.
//
// NOTE: conv2d was split into 2 kernels (ADR 0024). Only the **direct kernel** is listed here (the
// grid-stride side that stays permanently for groups > 1); the implicit GEMM variant is "1
// workgroup = 1 output tile" and so a different category — its safety nets are the fail loudly of
// `tiledWorkgroups` and the poison-value injection in tests/gpu_full_write_test.ts (the same
// decision as for the 3 GEMM ops).
//
// NOTE: the one that always takes the reduced path in real use is **pad** (on the value side of
// the relative-position attention of flow / voice, `F.pad(p_attn, [w, w])` needs more than 65535
// workgroups at T ≳ 2900). The other families merely "do not exceed it with today's measured
// shapes", so whether they reach the limit does not decide whether they are listed.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrDtype } from "../src/format/ir.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { refTensor } from "../src/reference/ops.ts";
import { WEIGHT_STORAGES } from "../src/kernels/weight-storage.ts";
import {
  adalnNormCase,
  argmaxCase,
  attentionStatsCase,
  attentionStatsF16Case,
  attentionStatsS16Case,
  axisReduceCase,
  conv1dCase,
  conv2dCase,
  convTranspose1dCase,
  cumsumCase,
  deformConv2dCase,
  type DegenerateCase,
  elementwiseCase,
  embeddingCase,
  flipCase,
  gatherCase,
  groupNormApplyCase,
  groupNormPartialCase,
  gruScanCase,
  layerNormCase,
  maskedFillCase,
  padCase,
  quantizeRowsCase,
  quantizeRowsChannelMajorCase,
  quantizeRowsGroupedCase,
  reduceCase,
  rmsNormCase,
  rmsNormPrefetchCase,
  siluCase,
  softmaxCase,
  stridedCase,
  stridedWriteCase,
  topkCase,
  upsample2xCase,
  upsampleBilinear2dCase,
} from "./helpers/gridstride_cases.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * Reads the output buffer **in the semantic dtype of the expected value** (the physical
 * representation of ADR 0009 — bool is a u32 0/1).
 *
 * MUST: do not hard-code the dtype to f32. `argmax` writes i32 (indices), so reading it as f32
 * reinterprets the bits and makes the comparison entirely meaningless (it always fails, but the
 * failure looks like "the kernel does not cope with a reduced dispatch" and hides the cause).
 */
const readback = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
  dtype: IrDtype,
): Promise<Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>> => {
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
  switch (dtype) {
    case "f32":
      return new Float32Array(copy, 0, count);
    case "i32":
      return new Int32Array(copy, 0, count);
    case "bool":
      return new Uint32Array(copy, 0, count);
  }
};

const runDegenerate = async (gpu: GpuContext, testCase: DegenerateCase): Promise<void> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const { pipeline, layout } = await cache.get(testCase.key, testCase.wgsl);
    const entries: GPUBindGroupEntry[] = [];
    const params = arena.allocHostWritten(
      testCase.params.byteLength,
      testCase.uniformParams ? UNIFORM_IN : STORAGE_IN,
    );
    gpu.device.queue.writeBuffer(params, 0, testCase.params);
    entries.push({ binding: 0, resource: { buffer: params } });
    testCase.inputs.forEach((input, index) => {
      const payload = input instanceof Uint8Array ? input : input.data;
      const buffer = arena.allocHostWritten(payload.byteLength, STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, payload);
      entries.push({ binding: index + 1, resource: { buffer } });
    });
    if (testCase.sideOutputBytes !== undefined) {
      const side = arena.allocRegion(Math.max(4, testCase.sideOutputBytes));
      entries.push({ binding: entries.length, resource: { buffer: side } });
    }
    const count = testCase.expected.data.length;
    const dst = arena.allocRegion(Math.max(4, count * 4));
    entries.push({ binding: entries.length, resource: { buffer: dst } });
    if (testCase.weightScale !== undefined) {
      const scale = arena.allocHostWritten(testCase.weightScale.byteLength, STORAGE_IN);
      gpu.device.queue.writeBuffer(scale, 0, testCase.weightScale);
      entries.push({ binding: entries.length, resource: { buffer: scale } });
    }

    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries,
    });
    scheduler.dispatch(pipeline, bindGroup, [testCase.groups, 1, 1], testCase.key);
    await scheduler.flush();

    const actual = await readback(gpu.device, dst, count, testCase.expected.dtype);
    const report = compareTensors(refTensor(testCase.expected.shape, actual), testCase.expected);
    assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
  } finally {
    await arena.destroy();
  }
};

const CASES: readonly DegenerateCase[] = [
  elementwiseCase(),
  stridedCase(),
  gatherCase(),
  maskedFillCase(),
  // The 4 families with weight-storage variants are listed **once per variant** (a different WGSL
  // body = a separate target). linear uses fixed tiles and does not claim grid-stride, so it is
  // outside this harness.
  ...WEIGHT_STORAGES.flatMap((storage) => [
    embeddingCase(storage),
    conv1dCase(storage),
    conv2dCase(storage),
    convTranspose1dCase(storage),
  ]),
  padCase(),
  flipCase(),
  upsampleBilinear2dCase(),
  deformConv2dCase(),
  stridedWriteCase(),
  cumsumCase(),
  reduceCase(),
  axisReduceCase(),
  // argmax is a **separate family** with the same row-wise grid-stride as row reduce (identity −inf
  // + index tracking).
  // Its output is i32, so readback takes a view in the expected value's dtype.
  argmaxCase(),
  // topk is another separate family with the same row-wise grid-stride (lane-local top-k →
  // tournament merge, 2 outputs).
  // The comparison is on the index column; the value column goes to sideOutputBytes (doc of
  // helpers/gridstride_cases.ts).
  topkCase(),
  softmaxCase(),
  // Of the 3 kernels of the fused attention, only ② is row-wise grid-stride (①③ are tiled, a
  // different category — their safety nets are the fail loudly of tiledWorkgroups + the full-write
  // test).
  attentionStatsCase(),
  // The f16-storage variant of S (option γ, phase 1) is core WGSL, so it is listed here (the `:c16`
  // version needs another device)
  attentionStatsS16Case(),
  // Activation quantization of w8a8 (the first of 1 node's 2 dispatches — the i8a8 GEMM side is
  // tiled, a different category)
  quantizeRowsCase(),
  quantizeRowsGroupedCase(),
  // The channel-major form conv3d w8a8 reads its input with (perf-ledger V-6)
  quantizeRowsChannelMajorCase(),
  layerNormCase(),
  // group_norm: the partial stage (chunks over rows) and the apply stage (elements)
  groupNormPartialCase(),
  groupNormApplyCase(),
  rmsNormCase(),
  rmsNormCase(true),
  rmsNormPrefetchCase(),
  gruScanCase(),
  // Fused kernels (outside the op vocabulary — the 3 families src/runtime/fusion.ts inserts by
  // rewriting). Their own tests have small shapes where the needed workgroups are assigned as is,
  // so the reduced path is exercised only here.
  upsample2xCase(),
  siluCase(),
  adalnNormCase(),
];

test.skipIf(!GPU_AVAILABLE)(
  "every grid-stride kernel family writes the whole range even with far fewer workgroups than needed (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        // Evidence that the dispatch is reduced: the plain split needs this many, yet it runs with
        // `groups`.
        assertEquals(
          testCase.natural > testCase.groups * 10,
          true,
          `${testCase.name}: ${testCase.natural} workgroups needed, so a dispatch of ${testCase.groups} does not count as reduced`,
        );
        await runDegenerate(gpu, testCase);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Families of the f16 compute variants (ADR 0028). They **need another device** (`shader-f16` can
 * only be requested when the device is created), so they are a separate test from the combined
 * harness above. SKIP on adapters that do not list it.
 */
const F16_CASES: readonly DegenerateCase[] = [attentionStatsF16Case()];

test.skipIf(!GPU_AVAILABLE || !SHADER_F16_AVAILABLE)(
  "the grid-stride families of the f16 compute variants also write the whole range with a reduced dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      for (const testCase of F16_CASES) {
        assertEquals(
          testCase.natural > testCase.groups * 10,
          true,
          `${testCase.name}: ${testCase.natural} workgroups needed, so a dispatch of ${testCase.groups} does not count as reduced`,
        );
        await runDegenerate(gpu, testCase);
      }
    } finally {
      gpu.destroy();
    }
  },
);
