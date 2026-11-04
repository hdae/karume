// **Bit-identity check** of conv1d implicit GEMM (the 1D version of ADR 0024).
//
// On the same input buffers,
//
//   (1) the direct convolution kernel (`conv1d:v2:…:direct:wg256` — kept permanently for groups > 1)
//   (2) the implicit GEMM kernel (`conv1d:v4:…:igemm64x128…`)
//
// run on **the same real GPU**, and their outputs are compared as **f32 bit patterns**. The point is
// to compare by exact `Uint32Array` equality rather than allclose: a change in the rounding sequence
// that a tolerance would hide (e.g. adding the bias at the store instead of as the initial value of
// `acc` / splitting the flat k in `(k, ic)` order) can be detected only here.
//
// Why it holds: ascending flat k = the direct kernel's double ascending `(ic, k)`, ascending K tiles
// of 16, the bias comes **before** the reduction in both, and for padding "skip without adding" and
// "multiply by 0 and add" agree because `a + 0.0 == a` (for finite `a`).
//
// **The only exception = signed zero**: adding a padding-derived `+0.0` where the partial sum is
// exactly `−0.0` flips it to `+0.0` (the direct kernel adds nothing, so `−0.0` remains). This file's
// main test avoids that region by **making the bias non-zero in every case**; the mechanism itself
// is pinned by the negative case at the end.
//
// MUST: do not let the check become vacuous. (1) and (2) are **different pipelines and different
// WGSL**, from different generator functions.
// MUST: cover every variant (v4 / scalar) and every storage (f32 / f16 / i8). The variant each case
// exercises is pinned explicitly with `expectVec4` (so that a mix-up in the predicate cannot hide
// behind "both green on the same variant").

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { alignF16Payload } from "../src/format/f16.ts";
import { alignI8Payload } from "../src/format/i8.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { GEMM_MTILE_SMALL, gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import {
  CONV1D_SCALE_BINDING,
  CONV1D_WORKGROUP_SIZE,
  type Conv1dDims,
  conv1dIgemmKey,
  conv1dIgemmParams,
  conv1dIgemmWgsl,
  conv1dKey,
  conv1dParams,
  conv1dUsesVec4,
  conv1dWgsl,
} from "../src/kernels/conv1d.ts";
import { conv2dIgemmMTile } from "../src/kernels/conv2d.ts";
import { allclose } from "../src/reference/allclose.ts";
import { referenceConv1d, refTensor } from "../src/reference/ops.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * Tolerance of the comparison with the CPU reference. The reference reduces in JS f64, which rounds
 * differently from sequential f32 addition, so bit identity cannot hold in principle (a few ulp for
 * this set of cases: K ≤ 288, ranges |x| ≤ 2 / |w| ≤ 0.25). **An implementation bug's error is
 * O(1)**, more than 6 orders of magnitude away.
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/**
 * Weights only for the signed-zero negative case (**strictly positive**). The default {@link WEIGHT}
 * contains 0, and `x · 0` produces signed zeros of its own, so whether the negative case's mechanism
 * comes from padding could not be isolated.
 */
const POSITIVE_WEIGHT = (i: number): number => 0.05 + ((i * 11) % 19) * 0.011;
/** MUST: the bias is non-zero (avoids the signed-zero region — the caveat at the top). */
const BIAS = (i: number): number => 0.375 + (i % 5) * 0.25;

type ParityCase = {
  readonly name: string;
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly kernel: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly storage: WeightStorage;
  /** The variant the case should exercise (stated explicitly so that a predicate mix-up cannot hide behind "both on the same variant"). */
  readonly expectVec4: boolean;
  /** By default a non-zero bias / weights that contain 0. Only the signed-zero negative case replaces them. */
  readonly bias?: (index: number) => number;
  readonly weight?: (index: number) => number;
};

const outLength = (
  input: number,
  padding: number,
  dilation: number,
  kernel: number,
  stride: number,
): number => Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

const dimsOf = (testCase: ParityCase): Conv1dDims => ({
  batch: testCase.batch,
  channelsIn: testCase.channelsIn,
  channelsOut: testCase.channelsOut,
  lengthIn: testCase.lengthIn,
  lengthOut: outLength(
    testCase.lengthIn,
    testCase.padding,
    testCase.dilation,
    testCase.kernel,
    testCase.stride,
  ),
  kernel: testCase.kernel,
  stride: testCase.stride,
  padding: testCase.padding,
  dilation: testCase.dilation,
  groups: 1,
});

/**
 * The payload for each weight storage variant (f32 as is / f16 and i8 as aligned raw bytes).
 *
 * `values` is **the f32 decoded back from the storage** (the weights after fake-quant). The CPU
 * reference uses it as the weights — handing it the plain f32 would turn the storage's rounding error
 * into "a kernel error".
 */
const weightPayload = (
  weight: FilledTensor,
  storage: WeightStorage,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly scale?: Float32Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
} => {
  if (storage === "f32") {
    return {
      bytes: new Uint8Array(weight.data.buffer, weight.data.byteOffset, weight.data.byteLength),
      values: weight.data as Float32Array<ArrayBuffer>,
    };
  }
  if (storage === "f16") {
    const quantized = quantizeF16(weight.data);
    return { bytes: alignF16Payload(quantized.bytes), values: quantized.values };
  }
  // The axis of conv1d's per-channel scale is 0 (output channels — ADR 0019)
  const quantized = quantizeI8(weight.data, weight.shape, 0);
  return {
    bytes: alignI8Payload(quantized.bytes),
    scale: quantized.scale,
    values: quantized.values,
  };
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

type ParityResult = {
  readonly direct: Uint32Array;
  readonly igemm: Uint32Array;
  readonly v4: boolean;
  readonly reference: Float32Array;
};

/**
 * Runs the direct kernel and implicit GEMM on **the same input buffers** and returns the output bit patterns as a pair.
 *
 * For the inputs (x / weights / bias / scale) both dispatches bind the same GPUBuffer — writing them
 * twice on the host would turn "a difference in the writes" into "a difference in the kernels".
 */
const runBoth = async (
  gpu: GpuContext,
  testCase: ParityCase,
  mTile: number = GEMM_TILE,
): Promise<ParityResult> => {
  const dims = dimsOf(testCase);
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const x = fill([dims.batch, dims.channelsIn, dims.lengthIn], SIGNED);
    const weight = fill(
      [dims.channelsOut, dims.channelsIn, dims.kernel],
      testCase.weight ?? WEIGHT,
    );
    const bias = fill([dims.channelsOut], testCase.bias ?? BIAS);
    const payload = weightPayload(weight, testCase.storage);

    const upload = (data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      // Round up to 16 bytes so that the vec4 binding (the f32 v4 path) does not drop the last quad
      const buffer = arena.allocHostWritten(
        Math.ceil(Math.max(4, data.byteLength) / 16) * 16,
        STORAGE_IN,
      );
      gpu.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const xBuffer = upload(x.data);
    const weightBuffer = upload(payload.bytes);
    const biasBuffer = upload(bias.data);
    const scaleBuffer = payload.scale === undefined ? undefined : upload(payload.scale);

    const count = dims.batch * dims.channelsOut * dims.lengthOut;
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
    const bind = (
      layout: GPUBindGroupLayout,
      params: Uint32Array<ArrayBuffer>,
    ): { readonly group: GPUBindGroup; readonly out: GPUBuffer } => {
      const paramsBuffer = arena.allocHostWritten(params.byteLength, UNIFORM_IN);
      gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
      const out = arena.allocRegion(Math.max(4, count * 4));
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: xBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } },
        { binding: 3, resource: { buffer: biasBuffer } },
        { binding: 4, resource: { buffer: out } },
      ];
      if (scaleBuffer !== undefined) {
        entries.push({ binding: CONV1D_SCALE_BINDING, resource: { buffer: scaleBuffer } });
      }
      return { group: gpu.device.createBindGroup({ layout, entries }), out };
    };

    // (1) direct kernel (grid-stride)
    const directKey = conv1dKey(testCase.storage);
    const { pipeline: directPipeline, layout: directLayout } = await cache.get(
      directKey,
      conv1dWgsl(testCase.storage),
    );
    const direct = bind(directLayout, conv1dParams(dims));
    scheduler.dispatch(
      directPipeline,
      direct.group,
      [gridStrideWorkgroups(count, CONV1D_WORKGROUP_SIZE, limit), 1, 1],
      directKey,
    );

    // (2) implicit GEMM (1 workgroup = 1 output tile)
    const kFlat = dims.channelsIn * dims.kernel;
    const v4 = conv1dUsesVec4(kFlat, dims.lengthOut, dims.stride);
    const igemmKey = conv1dIgemmKey(testCase.storage, v4, mTile);
    const { pipeline: igemmPipeline, layout: igemmLayout } = await cache.get(
      igemmKey,
      conv1dIgemmWgsl(testCase.storage, v4, mTile),
    );
    const igemm = bind(igemmLayout, conv1dIgemmParams(dims));
    // MUST: the dispatch edges are derived from **the same resolution point** as generation and the key
    // (`gemmMTileGeometry`). The m-tile variant moves only the row edge; the n edge is the geometry's
    // tileN (the same for both variants).
    const geometry = gemmMTileGeometry(mTile);
    scheduler.dispatch(
      igemmPipeline,
      igemm.group,
      [
        tiledWorkgroups(dims.lengthOut, gemmTileN(geometry), limit, testCase.name),
        tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, testCase.name),
        tiledWorkgroups(dims.batch, 1, limit, testCase.name),
      ],
      igemmKey,
    );

    await scheduler.flush();
    // The CPU reference is built from **the quantized weights expanded back** (so that the f16 / i8
    // storage error does not turn into "a kernel error"). Dequant is per element (ADR 0019).
    const reference = referenceConv1d(
      refTensor(x.shape, x.data),
      refTensor(weight.shape, payload.values),
      refTensor(bias.shape, bias.data),
      {
        stride: dims.stride,
        padding: dims.padding,
        dilation: dims.dilation,
        groups: 1,
      },
    );
    return {
      direct: await readbackBits(gpu.device, direct.out, count),
      igemm: await readbackBits(gpu.device, igemm.out, count),
      v4,
      reference: reference.data as Float32Array,
    };
  } finally {
    await arena.destroy();
  }
};

const CASES: readonly ParityCase[] = [
  {
    // A shrunk version of a measured shape (dacvae decoder): 3 taps, pad 1, same. Pins that with B=2 the
    // batch folds into N, and covers 2 m tiles (Cout=70) and a K remainder (24 = 16 + 8) in one case.
    name: "shrunk measured shape K=3 pad1 [2,8,20] * W[70,8,3] (f32/v4)",
    batch: 2,
    channelsIn: 8,
    channelsOut: 70,
    lengthIn: 20,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    storage: "f32",
    expectVec4: true,
  },
  {
    // kFlat = 1·7 = 7 -> `kFlat % 4 ≠ 0` selects the scalar variant, and `K % 16 ≠ 0` exercises the
    // zero fill of the K remainder. Also the detector for **a shape where deciding by `Cin % 4` would
    // pick v4** (Cin=1).
    name: "K remainder scalar Cin=1 K=7 [1,1,19] * W[5,1,7] pad3 (f32)",
    batch: 1,
    channelsIn: 1,
    channelsOut: 5,
    lengthIn: 19,
    kernel: 7,
    stride: 1,
    padding: 3,
    dilation: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // stride 2 (output and input positions no longer correspond 1:1)
    name: "stride=2 [1,3,21] * W[4,3,3] pad1 (f32)",
    batch: 1,
    channelsIn: 3,
    channelsOut: 4,
    lengthIn: 21,
    kernel: 3,
    stride: 2,
    padding: 1,
    dilation: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // **Detector for the stride condition of the v4 predicate**: kFlat = 16 and Lout = 8 are both
    // multiples of 4, so overlooking the stride would pick v4, and the 4-column contiguous read would
    // read positions 8 apart as contiguous. Also covers Cout=16 (1 m tile with 3/4 of its rows empty).
    name: "stride=8 [1,4,64] * W[16,4,4] pad0 (f32)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 16,
    lengthIn: 64,
    kernel: 4,
    stride: 8,
    padding: 0,
    dilation: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // dilation 3 (the tap spacing — applying it on the output-position side is indistinguishable from stride)
    name: "dilation=3 [1,4,24] * W[12,4,3] pad3 (f32/v4)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 12,
    lengthIn: 24,
    kernel: 3,
    stride: 1,
    padding: 3,
    dilation: 3,
    storage: "f32",
    expectVec4: true,
  },
  {
    // dilation 9 (the largest measured shape — taps reach far outside the sequence)
    name: "dilation=9 [1,2,40] * W[6,2,3] pad9 (f32)",
    batch: 1,
    channelsIn: 2,
    channelsOut: 6,
    lengthIn: 40,
    kernel: 3,
    stride: 1,
    padding: 9,
    dilation: 9,
    storage: "f32",
    expectVec4: false,
  },
  {
    // Cout=1 (1 m tile with 63/64 rows empty — pins that the acc of the empty rows is discarded by the
    // store's row guard, and that the out-of-range read of bias-first does not show in the values)
    name: "Cout=1 [1,4,12] * W[1,4,3] pad1 (f32/v4)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 1,
    lengthIn: 12,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    storage: "f32",
    expectVec4: true,
  },
  {
    // **Detector for `Lout % 4`**: kFlat = 12 is a multiple of 4 and stride = 1, so overlooking Lout
    // would pick v4, and the store's quad write would cross the end of the output.
    name: "Lout=15 (kFlat%4==0 but Lout%4≠0) [1,4,15] * W[8,4,3] pad1 (f32)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 8,
    lengthIn: 15,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // Padding wider than the kernel span (both ends of the output see only the padding area = the
    // error of reading out-of-range positions with clamped indices always shows in the values)
    name: "excess padding=3 [1,4,4] * W[5,4,3] (f32/v4)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 5,
    lengthIn: 4,
    kernel: 3,
    stride: 1,
    padding: 3,
    dilation: 1,
    storage: "f32",
    expectVec4: true,
  },
  {
    // f16 storage x v4 (the quad expansion `dequant4` relies on the flat index being a multiple of 4)
    name: "f16 v4 [1,8,16] * W[12,8,3] pad1",
    batch: 1,
    channelsIn: 8,
    channelsOut: 12,
    lengthIn: 16,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    storage: "f16",
    expectVec4: true,
  },
  {
    // f16 storage x scalar. kFlat = 1·9 = 9 is **odd**, so the start of a weight row alternates between
    // the high and low half of a word (the detector for choosing the pair by the parity of the flat
    // index — the pitfall of ADR 0018).
    name: "f16 scalar row length 9 (odd) [1,1,12] * W[4,1,9] pad4",
    batch: 1,
    channelsIn: 1,
    channelsOut: 4,
    lengthIn: 12,
    kernel: 9,
    stride: 1,
    padding: 4,
    dilation: 1,
    storage: "f16",
    expectVec4: false,
  },
  {
    // **Detector for the i8 scale axis**: the scale runs along rows (= output channels). With Cout = 96
    // there are 2 m tiles of 64 rows or 3 of 32 rows, so a mix-up that takes a tile-relative row or a
    // column instead of `arow` shows in the values (a shape that fits in one tile matches by accident
    // with `wid.y = 0`). Since `96 % 64 == 32`, it is also the main shape for the m-tile predicate.
    name: "i8 v4 Cout=96（M%64==32）[1,4,16] * W[96,4,1]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 96,
    lengthIn: 16,
    kernel: 1,
    stride: 1,
    padding: 0,
    dilation: 1,
    storage: "i8",
    expectVec4: true,
  },
  {
    // i8 x the scalar variant (taking `unpack4xI8`'s remainder mod 4 from the flat index + 2 m tiles).
    // kFlat = 2·3 = 6 is not a multiple of 4, so it falls to scalar.
    name: "i8 scalar 2 m tiles [1,2,10] * W[66,2,3] pad1",
    batch: 1,
    channelsIn: 2,
    channelsOut: 66,
    lengthIn: 10,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    storage: "i8",
    expectVec4: false,
  },
];

/** Formats the bit-pattern mismatches (the first 4) for a human reader. */
const firstMismatches = (direct: Uint32Array, igemm: Uint32Array): readonly string[] => {
  const found: string[] = [];
  for (let i = 0; i < direct.length && found.length < 4; i += 1) {
    if (direct[i] !== igemm[i]) {
      found.push(`[${i}] 0x${direct[i].toString(16)} vs 0x${igemm[i].toString(16)}`);
    }
  }
  return found;
};

/**
 * The checks for one case.
 *
 * (1) **exact Uint32 bit equality** with the direct kernel (a change in the rounding sequence shows
 *     only here)
 * (2) numeric agreement with the CPU reference (`referenceConv1d`) (the detector for both kernels
 *     being wrong **together** — a bit comparison alone cannot notice "they make the same mistake")
 * (3) the vacuity check (if the output is constant, agreement verifies nothing)
 */
const assertParity = (testCase: ParityCase, where: string, result: ParityResult): void => {
  const { direct, igemm, v4, reference } = result;
  assertEquals(
    v4,
    testCase.expectVec4,
    `${where}: the variant exercised differs from the expected one`,
  );
  assertEquals(
    firstMismatches(direct, igemm),
    [],
    `${where}: the bit pattern differs from the direct kernel`,
  );
  const values = new Float32Array(igemm.buffer, igemm.byteOffset, igemm.length);
  const report = allclose(values, reference, CPU_REFERENCE_TOLERANCE);
  assert(
    report.pass,
    `${where}: disagrees with the CPU reference (maxAbs ${report.maxAbsError} @${report.worstIndex} / violations ${report.failCount})`,
  );
  assert(
    new Set(direct).size > 1,
    `${where}: the output is constant (bit equality has become vacuous)`,
  );
};

test.skipIf(!GPU_AVAILABLE)(
  "conv1d implicit GEMM **matches the direct kernel bit for bit** (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        assertParity(testCase, testCase.name, await runBoth(gpu, testCase));
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The permanent check for **the 32-row m-tile variant**. Runs every case **forcibly** with 32-row
 * tiles (including shapes for which the predicate picks 64 rows). The tile shape is designed to
 * change only "which workgroup handles which outputs", so the same direct-kernel comparison as for
 * the 64-row version must pass unchanged.
 *
 * MUST: cover every shape independently of whether the predicate (`conv2dIgemmMTile`, shared with
 * conv2d) picks it. Checking only the shapes the predicate picks would let an error that shifts the
 * predicate's boundary pass green, with the variant "merely no longer running".
 */
test.skipIf(!GPU_AVAILABLE)(
  "the conv1d 32-row m-tile variant also matches the direct kernel bit for bit (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        assertParity(
          testCase,
          `m32 / ${testCase.name}`,
          await runBoth(gpu, testCase, GEMM_MTILE_SMALL),
        );
      }
      // CASES contains a shape for which the predicate actually picks 32 rows (the check that the test is not vacuous)
      assert(
        CASES.some((testCase) => conv2dIgemmMTile(testCase.channelsOut) === GEMM_MTILE_SMALL),
        "no case for which the predicate picks 32 rows",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Signed zero = the only exception to bit identity** (negative case). Records the mechanism as a
 * test and doubles as a detector for a future model whose bias contains `−0.0`.
 *
 * Shape: `Lin = 1 / K = 3 / padding = 3`, so for both ends of the output (ox = 0 and ox = 4) **the
 * whole receptive field is padding**. With the bias `−0.0` and all weights positive,
 *
 * - direct kernel … **does not add** the padding, so `−0.0` remains as is
 * - implicit GEMM … adds `0.0 × w = +0.0`, so it flips to `−0.0 + 0.0 = +0.0`
 *
 * MUST: the difference appears in **these 2 elements only**; the rest is bit-identical (the flip side
 * of `a + 0.0 == a` holding for finite `a`). As values `−0.0 == +0.0`, so neither allclose nor `===`
 * **can ever detect it** — only a bit comparison sees this difference.
 */
test.skipIf(!GPU_AVAILABLE)(
  "conv1d: with a −0.0 bias, the signed zero splits in outputs that see only padding (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const testCase: ParityCase = {
        name: "signed zero [1,1,1] * W[4,1,3] pad3 bias=-0.0",
        batch: 1,
        channelsIn: 1,
        channelsOut: 4,
        lengthIn: 1,
        kernel: 3,
        stride: 1,
        padding: 3,
        dilation: 1,
        storage: "f32",
        expectVec4: false,
        bias: () => -0.0,
        weight: POSITIVE_WEIGHT,
      };
      const { direct, igemm } = await runBoth(gpu, testCase);
      const lengthOut = 5;
      // Output positions whose receptive field is all padding (origin = ox − 3; the only valid input index is 0)
      const paddingOnly = new Set([0, 4]);
      const splits: number[] = [];
      for (let index = 0; index < direct.length; index += 1) {
        const ox = index % lengthOut;
        if (!paddingOnly.has(ox)) {
          assertEquals(direct[index], igemm[index], `[${index}] must match`);
          continue;
        }
        assertEquals(direct[index], 0x8000_0000, `[${index}] the direct kernel keeps −0.0`);
        assertEquals(igemm[index], 0x0000_0000, `[${index}] implicit GEMM flips to +0.0`);
        splits.push(index);
      }
      // The vacuity check (if every case fell on the "match" side, the negative case would not hold)
      assertEquals(splits.length, testCase.channelsOut * paddingOnly.size);
      // MUST: equal as values (this difference is detectable by neither allclose nor ===)
      const directValues = new Float32Array(direct.buffer, direct.byteOffset, direct.length);
      const igemmValues = new Float32Array(igemm.buffer, igemm.byteOffset, igemm.length);
      for (const index of splits) {
        assert(
          directValues[index] === igemmValues[index],
          `[${index}] as values −0.0 === +0.0 should hold`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The executor's dispatch between kernels. `groups == 1` goes to implicit GEMM, `groups > 1` to the
 * direct kernel (kept as a permanent differential oracle).
 *
 * MUST: look at **the pipeline keys that actually ran**, not at values. Both kernels are correct, so
 * matching outputs are no evidence of the dispatch.
 */
const conv1dKeysUsed = async (channels: number, groups: number): Promise<ReadonlySet<string>> => {
  const gpu = await acquireGpu();
  const graph = singleOpDeclaration(
    "conv1d",
    [[1, channels, 8], [channels, channels / groups, 3], [channels]],
    [[1, channels, 8]],
    { attrs: { stride: 1, padding: 1, dilation: 1, groups } },
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    await session.run({
      x0: fill([1, channels, 8], SIGNED),
      x1: fill([channels, channels / groups, 3], WEIGHT),
      x2: fill([channels], BIAS),
    });
    return new Set(
      requireCensus(session.diagnostics().lastRunPipelines, "conv").map((row) => row.key),
    );
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the executor dispatches between the 2 kernels by groups (1 = igemm / >1 = direct, real GPU)",
  async () => {
    // Cin = Cout = 6, K=3 -> kFlat = 18 (groups=1) is not a multiple of 4, so the scalar variant.
    // Cout=6 gives `6 % 64 == 6`, so the m tile has 32 rows.
    assertEquals(
      await conv1dKeysUsed(6, 1),
      new Set([conv1dIgemmKey("f32", false, GEMM_MTILE_SMALL)]),
    );
    assertEquals(await conv1dKeysUsed(6, 3), new Set([conv1dKey("f32")]));
    // depthwise (groups = Cin = Cout) also goes to the direct kernel
    assertEquals(await conv1dKeysUsed(6, 6), new Set([conv1dKey("f32")]));
  },
);

/**
 * The check that the m-tile choice is **wired into the executor**.
 *
 * MUST: look at **the pipeline keys that actually ran**, not at values. Both tile shapes give
 * bit-identical outputs, so matching values are no evidence of the wiring (with the predicate
 * knocked out, every numeric test would stay green).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the conv1d executor dispatches between 64/32-row m tiles by M%64 (real GPU)",
  async () => {
    // Cout = 96 (`96 % 64 == 32`) -> 32 rows. kFlat = 288 and Lout = 8, so v4.
    assertEquals(
      await conv1dKeysUsed(96, 1),
      new Set([conv1dIgemmKey("f32", true, GEMM_MTILE_SMALL)]),
    );
    // Cout = 64 (`64 % 64 == 0`) -> stays at 64 rows
    assertEquals(await conv1dKeysUsed(64, 1), new Set([conv1dIgemmKey("f32", true, GEMM_TILE)]));
  },
);
