// Real-GPU checks of conv3d (unbatched, implicit GEMM — ADR 0118 decision 1).
//
// 1. **Comparison with the CPU reference** (`referenceConv3d` — an f64 reduction, so the band is
//    the same atol 1e-5 / rtol 1e-5 as the conv2d parity test). 8 shapes = the 4 shapes of the chunk
//    graph (16→384 k3, 384→768 (3,1,1), 96→96 k3, 96→3 k3) **shrunk in space** + small shapes with a
//    K remainder, Wout%4≠0, stride≠1 and dilation≠1. Each runs with both 64- and 32-row m tiles,
//    and the 2 outputs must also match as Uint32 (the tile shape changes only the work split).
// 2. **Identity check (1)** (exact Uint32 equality): conv3d with Kt = 1 ≡ the implicit GEMM of
//    conv2d with the frames placed in the batch. With Kt = 1 the flat K order `(ic, kt, kh, kw)`
//    becomes the same sequence as conv2d's `(ic, kh, kw)`, and bias-first and the out-of-range 0 are
//    held by the shared skeleton too, so even the rounding sequence matches. 3 small shapes, and the
//    **full size** of the chunk graph's 4 shapes (sizes that would take minutes with the JS CPU
//    reference — compared GPU against GPU). A change in the rounding sequence that a tolerance would
//    hide (adding the bias at the store, etc.) **can be detected only here**.
// 3. **Identity check (2)**: Kt = 3 with the first 2 time frames zero ≡ Kt = 1 with the last slice
//    of the weights. The products of the zero frames are ±0, and `a + (±0) = a` (a ≠ 0), so the
//    values match. The only exception is signed zero (±0 flips only when the partial sum is exactly
//    0 — ADR 0024 decision 3), so a difference is allowed there only.
// 4. **Poison values of full-write**, **fail loudly at the dispatch limit**, **rejection of
//    groups > 1 at build time**, **the dispatch keys** and **f16 residency** go through a Session
//    (which also covers the wiring into the executor).
//
// MUST: do not let the checks become vacuous. (1) and (2) use different pipelines, different WGSL
// and different generator functions, and the outputs are also checked not to be constant. The bias
// is non-zero in every case (avoids the signed-zero region).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { alignF16Payload } from "../src/format/f16.ts";
import { alignI8Payload } from "../src/format/i8.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dUsesVec4,
} from "../src/kernels/conv2d.ts";
import {
  CONV3D_SCALE_BINDING,
  type Conv3dDims,
  conv3dIgemmKey,
  conv3dIgemmParams,
  conv3dIgemmWgsl,
  conv3dIgemmWorkgroups,
  conv3dUsesVec4,
} from "../src/kernels/conv3d.ts";
import { GEMM_MTILE_SMALL, gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { allclose } from "../src/reference/allclose.ts";
import { referenceConv3d, refTensor } from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * Tolerance of the comparison with the CPU reference (the same band as the conv2d parity test — the
 * same skeleton, the same reduction order). The reference reduces in JS f64, which rounds differently
 * from sequential f32 addition, so bit identity cannot hold in principle. **An implementation bug's
 * error is O(1)**, orders of magnitude away.
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
/** Small-ish weights (the output stays O(1) even for a reduction over K = 2,592 — keeps the band's rtol at a meaningful size). */
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.0041;
/** MUST: the bias is non-zero (avoids the signed-zero region — the caveat at the top). */
const BIAS = (i: number): number => 0.375 + (i % 5) * 0.25;

/** The large-size version of `fill` (does not go through a JS array for the 37.7 million elements of the full size). */
const generate = (count: number, value: (index: number) => number): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(count);
  for (let i = 0; i < count; i += 1) data[i] = value(i);
  return data;
};

type Triple = readonly [number, number, number];

type Conv3dCase = {
  readonly name: string;
  readonly channelsIn: number;
  readonly channelsOut: number;
  /** The input's [T, H, W]. */
  readonly input: Triple;
  /** The kernel's [Kt, Kh, Kw]. */
  readonly kernel: Triple;
  readonly stride: Triple;
  readonly padding: Triple;
  readonly dilation: Triple;
  readonly storage: WeightStorage;
};

const outLength = (
  input: number,
  padding: number,
  dilation: number,
  kernel: number,
  stride: number,
): number => Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

const dimsOf = (testCase: Conv3dCase): Conv3dDims => {
  const out = [0, 1, 2].map((axis) =>
    outLength(
      testCase.input[axis],
      testCase.padding[axis],
      testCase.dilation[axis],
      testCase.kernel[axis],
      testCase.stride[axis],
    ),
  );
  return {
    channelsIn: testCase.channelsIn,
    channelsOut: testCase.channelsOut,
    timeIn: testCase.input[0],
    heightIn: testCase.input[1],
    widthIn: testCase.input[2],
    timeOut: out[0],
    heightOut: out[1],
    widthOut: out[2],
    kernelT: testCase.kernel[0],
    kernelH: testCase.kernel[1],
    kernelW: testCase.kernel[2],
    strideT: testCase.stride[0],
    strideH: testCase.stride[1],
    strideW: testCase.stride[2],
    paddingT: testCase.padding[0],
    paddingH: testCase.padding[1],
    paddingW: testCase.padding[2],
    dilationT: testCase.dilation[0],
    dilationH: testCase.dilation[1],
    dilationW: testCase.dilation[2],
    groups: 1,
  };
};

const outCount = (dims: Conv3dDims): number =>
  dims.channelsOut * dims.timeOut * dims.heightOut * dims.widthOut;

/**
 * The payload for each weight storage variant. `values` is **the f32 decoded back from the storage**
 * (the weights after fake-quant); the CPU reference and the other side of the identity checks use it
 * as the weights (handing them the plain f32 would turn the storage's rounding into "a kernel
 * error").
 */
const weightPayload = (
  data: Float32Array<ArrayBuffer>,
  shape: readonly number[],
  storage: WeightStorage,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly scale?: Float32Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
} => {
  if (storage === "f32") {
    return { bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), values: data };
  }
  if (storage === "f16") {
    const quantized = quantizeF16(data);
    return { bytes: alignF16Payload(quantized.bytes), values: quantized.values };
  }
  // The axis of conv3d's per-channel scale is 0 (output channels — ADR 0019 / 0024 decision 6)
  const quantized = quantizeI8(data, shape, 0);
  return {
    bytes: alignI8Payload(quantized.bytes),
    scale: quantized.scale,
    values: quantized.values,
  };
};

/** One unit of GPU work (bundles the cache, the scheduler and the arena, and always disposes them at the end). */
type Rig = {
  readonly gpu: GpuContext;
  readonly cache: PipelineCache;
  readonly scheduler: SubmitScheduler;
  readonly arena: RunArena;
};

const withRig = async <T>(gpu: GpuContext, body: (rig: Rig) => Promise<T>): Promise<T> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    return await body({ gpu, cache, scheduler, arena });
  } finally {
    await arena.destroy();
  }
};

const upload = (rig: Rig, data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
  // Round up to 16 bytes so that the vec4 binding (the f32 v4 path) does not drop the last quad
  const buffer = rig.arena.allocHostWritten(
    Math.ceil(Math.max(4, data.byteLength) / 16) * 16,
    STORAGE_IN,
  );
  rig.gpu.device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

type Operands = {
  readonly x: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly scale?: GPUBuffer;
};

/** Builds the bindings (0 dims / 1 x / 2 weights / 3 bias / 4 output / 5 scale) and queues 1 dispatch. */
const dispatchGemm = async (
  rig: Rig,
  key: string,
  wgsl: string,
  params: Uint32Array<ArrayBuffer>,
  operands: Operands,
  count: number,
  workgroups: readonly [number, number, number],
): Promise<GPUBuffer> => {
  const { pipeline, layout } = await rig.cache.get(key, wgsl);
  const paramsBuffer = rig.arena.allocHostWritten(params.byteLength, UNIFORM_IN);
  rig.gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
  const out = rig.arena.allocRegion(Math.max(16, Math.ceil((count * 4) / 16) * 16));
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: paramsBuffer } },
    { binding: 1, resource: { buffer: operands.x } },
    { binding: 2, resource: { buffer: operands.weight } },
    { binding: 3, resource: { buffer: operands.bias } },
    { binding: 4, resource: { buffer: out } },
  ];
  if (operands.scale !== undefined) {
    entries.push({ binding: CONV3D_SCALE_BINDING, resource: { buffer: operands.scale } });
  }
  rig.scheduler.dispatch(
    pipeline,
    rig.gpu.device.createBindGroup({ layout, entries }),
    [...workgroups],
    key,
  );
  return out;
};

/** Queues one conv3d implicit GEMM (the dispatch is the same `conv3dIgemmWorkgroups` as in production). */
const dispatchConv3d = async (
  rig: Rig,
  dims: Conv3dDims,
  storage: WeightStorage,
  mTile: number,
  operands: Operands,
): Promise<{ readonly out: GPUBuffer; readonly v4: boolean }> => {
  const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  const v4 = conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW);
  const out = await dispatchGemm(
    rig,
    conv3dIgemmKey(storage, v4, mTile),
    conv3dIgemmWgsl(storage, v4, mTile),
    conv3dIgemmParams(dims),
    operands,
    outCount(dims),
    conv3dIgemmWorkgroups(dims, mTile, rig.gpu.limits.maxComputeWorkgroupsPerDimension, "conv3d"),
  );
  return { out, v4 };
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

const asFloats = (bits: Uint32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(bits.buffer, bits.byteOffset, bits.length);

/** The bit-pattern mismatches (the first 4; with `allowSignedZero`, ±0 against ±0 counts as a match). */
const firstMismatches = (
  expected: Uint32Array,
  actual: Uint32Array,
  allowSignedZero = false,
): readonly string[] => {
  assertEquals(actual.length, expected.length, "number of elements compared");
  const found: string[] = [];
  for (let i = 0; i < expected.length && found.length < 4; i += 1) {
    if (expected[i] === actual[i]) continue;
    if (allowSignedZero && (expected[i] & 0x7fffffff) === 0 && (actual[i] & 0x7fffffff) === 0) {
      continue;
    }
    found.push(`[${i}] 0x${expected[i].toString(16)} vs 0x${actual[i].toString(16)}`);
  }
  return found;
};

/** The vacuity check (if the output is constant, agreement verifies nothing). */
const assertNotConstant = (bits: Uint32Array, where: string): void => {
  const first = bits[0];
  assert(
    bits.some((value) => value !== first),
    `${where}: the output is constant (agreement has become vacuous)`,
  );
};

// ---------------------------------------------------------------------------
// 1. Comparison with the CPU reference (8 shapes x m tiles of 64 / 32)
// ---------------------------------------------------------------------------

const ONE: Triple = [1, 1, 1];

type ParityCase = Conv3dCase & {
  /** The variant the case should exercise (stated explicitly so that a predicate mix-up cannot hide behind "both on the same variant"). */
  readonly expectVec4: boolean;
};

const PARITY_CASES: readonly ParityCase[] = [
  {
    // A spatially shrunk version of the chunk graph's conv_in (16→384 k3, temporal padding 0 <cache 2 +
    // 1 frame>, spatial same). M = 384 gives 6 m tiles; K = 432 = 16·27.
    name: "shrunk chunk conv_in 16→384 k3 [16,3,6,8]",
    channelsIn: 16,
    channelsOut: 384,
    input: [3, 6, 8],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
    expectVec4: true,
  },
  {
    // A spatially shrunk version of time_conv (384→768, kernel (3,1,1)). f16 storage (Wan's preset) x v4.
    name: "shrunk chunk time_conv 384→768 (3,1,1) [384,3,4,8] f16",
    channelsIn: 384,
    channelsOut: 768,
    input: [3, 4, 8],
    kernel: [3, 1, 1],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    storage: "f16",
    expectVec4: true,
  },
  {
    // A spatially shrunk version of up3's 96→96 k3 (Tout = 4 = a chunk after the temporal upsample).
    // M = 96 (96%64 == 32, so the m-tile predicate picks 32 rows) x i8 — exercises the row scale with 3
    // m tiles (32 rows) / 2 (64 rows).
    name: "shrunk chunk 96→96 k3 [96,6,6,8] i8",
    channelsIn: 96,
    channelsOut: 96,
    input: [6, 6, 8],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "i8",
    expectVec4: true,
  },
  {
    // A spatially shrunk version of head (96→3 k3). M = 3 (under a tenth of one tile), N = 4·6·8 = 192 (2 n tiles).
    name: "shrunk chunk head 96→3 k3 [96,6,6,8]",
    channelsIn: 96,
    channelsOut: 3,
    input: [6, 6, 8],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
    expectVec4: true,
  },
  {
    // K remainder: kFlat = 2·27 = 54 (not a multiple of 4 -> the scalar variant; not a multiple of 16 ->
    // the last K tile is zero-filled). padding 1 on every axis exercises the out-of-range 0 on all 3 axes
    // (the detector for clamped reads). Cin ≠ Cout.
    name: "K remainder scalar [2,4,5,7] * W[5,2,3,3,3] padding 1",
    channelsIn: 2,
    channelsOut: 5,
    input: [4, 5, 7],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [1, 1, 1],
    dilation: ONE,
    storage: "f32",
    expectVec4: false,
  },
  {
    // **Detector for the second v4 condition (Wout%4)**: Tout = Hout = Wout = 2 gives N = 8 (%4 == 0),
    // kFlat = 4·5 = 20 (%4 == 0), strideW = 1. Writing the predicate as N%4 would pick v4, and since
    // Win = 6 the quad's contiguous-read guard lets it through too, so 4 columns that span output rows
    // are read from the same input row.
    name: "Wout=2 (N%4==0 but Wout%4≠0) [4,2,2,6] * W[3,4,1,1,5]",
    channelsIn: 4,
    channelsOut: 3,
    input: [2, 2, 6],
    kernel: [1, 1, 5],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    storage: "f32",
    expectVec4: false,
  },
  {
    // stride ≠ 1 on all 3 axes, asymmetric (T 2 / H 2 / W 3); Kt≠Kh≠Kw; padding asymmetric too.
    // T: (7+2−2−1)/2+1 = 4 / H: (9+2−1−1)/2+1 = 5 / W: (11−3−1)/3+1 = 3.
    name: "stride=[2,2,3] Kt≠Kh≠Kw [3,7,9,11] * W[4,3,3,2,4]",
    channelsIn: 3,
    channelsOut: 4,
    input: [7, 9, 11],
    kernel: [3, 2, 4],
    stride: [2, 2, 3],
    padding: [1, 1, 0],
    dilation: ONE,
    storage: "f32",
    expectVec4: false,
  },
  {
    // dilation ≠ 1 on all 3 axes (T 3 / H 3 / W 2); exercises a `kt·dt` / `kh·dh` / `kw·dw` mix-up on the v4 path.
    // T: 8−3−1+1 = 5 / H: 12−6−1+1 = 6 / W: 10−2−1+1 = 8.
    name: "dilation=[3,3,2] [4,8,12,10] * W[6,4,2,3,2] (v4)",
    channelsIn: 4,
    channelsOut: 6,
    input: [8, 12, 10],
    kernel: [2, 3, 2],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: [3, 3, 2],
    storage: "f32",
    expectVec4: true,
  },
];

/** The CPU reference and the GPU (with both 64- and 32-row m tiles) for one case. */
const runParity = async (
  gpu: GpuContext,
  testCase: ParityCase,
): Promise<{
  readonly m64: Uint32Array<ArrayBuffer>;
  readonly m32: Uint32Array<ArrayBuffer>;
  readonly v4: boolean;
  readonly reference: Float32Array;
}> => {
  const dims = dimsOf(testCase);
  const xShape = [dims.channelsIn, dims.timeIn, dims.heightIn, dims.widthIn];
  const wShape = [dims.channelsOut, dims.channelsIn, dims.kernelT, dims.kernelH, dims.kernelW];
  const x = generate(
    xShape.reduce((a, b) => a * b, 1),
    SIGNED,
  );
  const w = generate(
    wShape.reduce((a, b) => a * b, 1),
    WEIGHT,
  );
  const bias = generate(dims.channelsOut, BIAS);
  const payload = weightPayload(w, wShape, testCase.storage);
  return await withRig(gpu, async (rig) => {
    const operands: Operands = {
      x: upload(rig, x),
      weight: upload(rig, payload.bytes),
      bias: upload(rig, bias),
      scale: payload.scale === undefined ? undefined : upload(rig, payload.scale),
    };
    const m64 = await dispatchConv3d(rig, dims, testCase.storage, GEMM_TILE, operands);
    const m32 = await dispatchConv3d(rig, dims, testCase.storage, GEMM_MTILE_SMALL, operands);
    await rig.scheduler.flush();
    const reference = referenceConv3d(
      refTensor(xShape, x),
      refTensor(wShape, payload.values),
      refTensor([dims.channelsOut], bias),
      {
        stride: testCase.stride,
        padding: testCase.padding,
        dilation: testCase.dilation,
        groups: 1,
      },
    );
    return {
      m64: await readbackBits(gpu.device, m64.out, outCount(dims)),
      m32: await readbackBits(gpu.device, m32.out, outCount(dims)),
      v4: m64.v4,
      reference: reference.data as Float32Array,
    };
  });
};

test.skipIf(!GPU_AVAILABLE)(
  "conv3d implicit GEMM matches the CPU reference (8 shapes x m tiles of 64 / 32, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of PARITY_CASES) {
        const { m64, m32, v4, reference } = await runParity(gpu, testCase);
        const where = testCase.name;
        assertEquals(
          v4,
          testCase.expectVec4,
          `${where}: the variant exercised differs from the expected one`,
        );
        // The tile shape changes only "which workgroup handles which outputs" (the numeric path is shared)
        assertEquals(
          firstMismatches(m64, m32),
          [],
          `${where}: the bit patterns differ between 64- and 32-row m tiles`,
        );
        const report = allclose(asFloats(m64), reference, CPU_REFERENCE_TOLERANCE);
        assert(
          report.pass,
          `${where}: disagrees with the CPU reference (maxAbs ${report.maxAbsError} @${report.worstIndex} / violations ${report.failCount})`,
        );
        assertNotConstant(m64, where);
      }
      // Both shapes for which the predicate actually picks 32 rows and shapes for which it picks 64 are included (the check that the test is not vacuous)
      const tiles = new Set(PARITY_CASES.map((testCase) => conv2dIgemmMTile(testCase.channelsOut)));
      assertEquals(
        [...tiles].sort((a, b) => a - b),
        [GEMM_MTILE_SMALL, GEMM_TILE].sort((a, b) => a - b),
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// 2. Identity check (1): conv3d with Kt = 1 ≡ conv2d per frame (exact Uint32 equality)
// ---------------------------------------------------------------------------

/** Returns, as a pair, the outputs of conv3d with Kt = 1 and of conv2d with the frames placed in the batch. */
const runFrameIdentity = async (
  gpu: GpuContext,
  testCase: Conv3dCase,
): Promise<{
  readonly conv3d: Uint32Array<ArrayBuffer>;
  readonly conv2d: Uint32Array<ArrayBuffer>;
}> => {
  assertEquals(
    testCase.kernel[0],
    1,
    `${testCase.name}: identity check (1) takes only shapes with Kt = 1`,
  );
  assertEquals(testCase.padding[0], 0, `${testCase.name}: temporal padding 0`);
  assertEquals(testCase.stride[0], 1, `${testCase.name}: temporal stride 1`);
  const dims = dimsOf(testCase);
  const { channelsIn: cin, timeIn: time, heightIn: height, widthIn: width } = dims;
  const plane = height * width;
  const x = generate(cin * time * plane, SIGNED);
  // Frames into the batch: [Cin, T, H, W] -> [T, Cin, H, W]
  const frames = new Float32Array(x.length);
  for (let c = 0; c < cin; c += 1) {
    for (let t = 0; t < time; t += 1) {
      frames.set(
        x.subarray((c * time + t) * plane, (c * time + t + 1) * plane),
        (t * cin + c) * plane,
      );
    }
  }
  const wShape = [dims.channelsOut, cin, 1, dims.kernelH, dims.kernelW];
  const w = generate(
    wShape.reduce((a, b) => a * b, 1),
    WEIGHT,
  );
  // The weights are **the same bytes** as [Cout, Cin, 1, Kh, Kw] and as [Cout, Cin, Kh, Kw] (the same buffer is bound)
  const payload = weightPayload(w, wShape, testCase.storage);
  const bias = generate(dims.channelsOut, BIAS);
  const conv2dDims = {
    batch: time,
    channelsIn: cin,
    channelsOut: dims.channelsOut,
    heightIn: height,
    widthIn: width,
    heightOut: dims.heightOut,
    widthOut: dims.widthOut,
    kernelH: dims.kernelH,
    kernelW: dims.kernelW,
    strideH: dims.strideH,
    strideW: dims.strideW,
    paddingH: dims.paddingH,
    paddingW: dims.paddingW,
    dilationH: dims.dilationH,
    dilationW: dims.dilationW,
    groups: 1,
  };
  return await withRig(gpu, async (rig) => {
    const shared = {
      weight: upload(rig, payload.bytes),
      bias: upload(rig, bias),
      scale: payload.scale === undefined ? undefined : upload(rig, payload.scale),
    };
    const mTile = conv2dIgemmMTile(dims.channelsOut);
    const three = await dispatchConv3d(rig, dims, testCase.storage, mTile, {
      x: upload(rig, x),
      ...shared,
    });
    const kFlat = cin * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const geometry = gemmMTileGeometry(mTile);
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
    const two = await dispatchGemm(
      rig,
      conv2dIgemmKey(testCase.storage, v4, mTile),
      conv2dIgemmWgsl(testCase.storage, v4, mTile),
      conv2dIgemmParams(conv2dDims),
      { x: upload(rig, frames), ...shared },
      outCount(dims),
      [
        tiledWorkgroups(dims.heightOut * dims.widthOut, gemmTileN(geometry), limit, "conv2d"),
        tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, "conv2d"),
        tiledWorkgroups(time, 1, limit, "conv2d"),
      ],
    );
    await rig.scheduler.flush();
    const conv3dBits = await readbackBits(gpu.device, three.out, outCount(dims));
    const conv2dBits = await readbackBits(gpu.device, two, outCount(dims));
    // conv2d's output [T, Cout, Hout·Wout] -> conv3d's order [Cout, T, Hout·Wout]
    const planeOut = dims.heightOut * dims.widthOut;
    const reordered = new Uint32Array(conv2dBits.length);
    for (let t = 0; t < time; t += 1) {
      for (let oc = 0; oc < dims.channelsOut; oc += 1) {
        reordered.set(
          conv2dBits.subarray(
            (t * dims.channelsOut + oc) * planeOut,
            (t * dims.channelsOut + oc + 1) * planeOut,
          ),
          (oc * time + t) * planeOut,
        );
      }
    }
    return { conv3d: conv3dBits, conv2d: reordered };
  });
};

const IDENTITY_SMALL: readonly Conv3dCase[] = [
  {
    // The scalar variant; stride / dilation / padding asymmetric between H / W; Cin ≠ Cout
    name: "small scalar [3,4,5,6] * W[4,3,1,3,2] stride=[1,2,1] dilation=[1,1,2]",
    channelsIn: 3,
    channelsOut: 4,
    input: [4, 5, 6],
    kernel: [1, 3, 2],
    stride: [1, 2, 1],
    padding: [0, 1, 0],
    dilation: [1, 1, 2],
    storage: "f32",
  },
  {
    // The v4 variant (kFlat = 72, Wout = 8)
    name: "small v4 [8,3,6,8] * W[12,8,1,3,3]",
    channelsIn: 8,
    channelsOut: 12,
    input: [3, 6, 8],
    kernel: [1, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
  },
  {
    // M = 70 (3 m tiles of 32 rows) x i8 (row scale) x 2 n tiles (N = 5·4·8 = 160)
    name: "small i8 m32 [4,5,4,8] * W[70,4,1,1,3]",
    channelsIn: 4,
    channelsOut: 70,
    input: [5, 4, 8],
    kernel: [1, 1, 3],
    stride: ONE,
    padding: [0, 0, 1],
    dilation: ONE,
    storage: "i8",
  },
];

/**
 * The **full size** of the chunk graph's 4 shapes (32×32 latent tiles — ADR 0118 decision 2) with
 * Kt = 1. Identity check (2) covers the temporal kernel at full size, so these exercise the 3D gather
 * (n -> (ot, oy, ox)) and the dispatch edges at the full size of channels, space and frame count
 * (96→96 gives N = 6·256² = 393,216, i.e. 3,072 n tiles).
 */
const IDENTITY_REAL: readonly Conv3dCase[] = [
  {
    name: "full size conv_in 16→384 (1,3,3) [16,3,32,32]",
    channelsIn: 16,
    channelsOut: 384,
    input: [3, 32, 32],
    kernel: [1, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size time_conv 384→768 (1,1,1) [384,3,32,32]",
    channelsIn: 384,
    channelsOut: 768,
    input: [3, 32, 32],
    kernel: [1, 1, 1],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size up3 96→96 (1,3,3) [96,6,256,256]",
    channelsIn: 96,
    channelsOut: 96,
    input: [6, 256, 256],
    kernel: [1, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size head 96→3 (1,3,3) [96,6,256,256]",
    channelsIn: 96,
    channelsOut: 3,
    input: [6, 256, 256],
    kernel: [1, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "identity check (1): conv3d with Kt = 1 matches conv2d per frame as Uint32 (3 small shapes + 4 full-size shapes, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of [...IDENTITY_SMALL, ...IDENTITY_REAL]) {
        const { conv3d, conv2d } = await runFrameIdentity(gpu, testCase);
        assertEquals(
          firstMismatches(conv2d, conv3d),
          [],
          `${testCase.name}: the bit pattern differs from conv2d per frame`,
        );
        assertNotConstant(conv3d, testCase.name);
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// 3. Identity check (2): Kt = 3 with the first 2 time frames zero ≡ Kt = 1 with the last slice of the weights
// ---------------------------------------------------------------------------

/**
 * Returns, as a pair, conv3d with Kt = 3 where the first 2 of x's 3 time frames are zero (temporal
 * padding 0 -> Tout = 1 — the same shape as the chunk graph's first chunk <2 zero cache frames + 1
 * frame>) and conv3d with Kt = 1 on the third frame only and the last slice of the weights only.
 */
const runZeroFrameIdentity = async (
  gpu: GpuContext,
  testCase: Conv3dCase,
): Promise<{ readonly kt3: Uint32Array<ArrayBuffer>; readonly kt1: Uint32Array<ArrayBuffer> }> => {
  assertEquals(
    testCase.kernel[0],
    3,
    `${testCase.name}: identity check (2) takes only shapes with Kt = 3`,
  );
  assertEquals(testCase.input[0], 3, `${testCase.name}: 3 time frames (cache 2 + 1)`);
  // The i8 scale is re-derived from the amax of each row (output channel), so it differs between Kt = 3
  // and Kt = 1 — this check is set up only for f32 / f16 (storages without a scale).
  assert(
    testCase.storage !== "i8",
    `${testCase.name}: identity check (2) takes only storages without a scale`,
  );
  const dims = dimsOf(testCase);
  assertEquals(dims.timeOut, 1);
  const { channelsIn: cin, channelsOut: cout, heightIn: height, widthIn: width } = dims;
  const plane = height * width;
  const x = generate(cin * 3 * plane, (i) => (Math.floor(i / plane) % 3 < 2 ? 0 : SIGNED(i)));
  const lastFrame = new Float32Array(cin * plane);
  for (let c = 0; c < cin; c += 1) {
    lastFrame.set(x.subarray((c * 3 + 2) * plane, (c * 3 + 3) * plane), c * plane);
  }
  const taps = dims.kernelH * dims.kernelW;
  const w3Shape = [cout, cin, 3, dims.kernelH, dims.kernelW];
  const w3 = weightPayload(generate(cout * cin * 3 * taps, WEIGHT), w3Shape, testCase.storage);
  // The last slice is cut from **the values decoded back from the storage** (aligns the f16 / i8 rounding on both sides)
  const sliced = new Float32Array(cout * cin * taps);
  for (let oc = 0; oc < cout; oc += 1) {
    for (let ic = 0; ic < cin; ic += 1) {
      const from = ((oc * cin + ic) * 3 + 2) * taps;
      sliced.set(w3.values.subarray(from, from + taps), (oc * cin + ic) * taps);
    }
  }
  const w1Shape = [cout, cin, 1, dims.kernelH, dims.kernelW];
  const w1 = weightPayload(sliced, w1Shape, testCase.storage);
  assertEquals(
    [...w1.values],
    [...sliced],
    `${testCase.name}: re-storing the last slice moved the values`,
  );
  const bias = generate(cout, BIAS);
  const dims1 = { ...dims, timeIn: 1, kernelT: 1 };
  return await withRig(gpu, async (rig) => {
    const biasBuffer = upload(rig, bias);
    const mTile = conv2dIgemmMTile(cout);
    const kt3 = await dispatchConv3d(rig, dims, testCase.storage, mTile, {
      x: upload(rig, x),
      weight: upload(rig, w3.bytes),
      bias: biasBuffer,
    });
    const kt1 = await dispatchConv3d(rig, dims1, testCase.storage, mTile, {
      x: upload(rig, lastFrame),
      weight: upload(rig, w1.bytes),
      bias: biasBuffer,
    });
    await rig.scheduler.flush();
    return {
      kt3: await readbackBits(gpu.device, kt3.out, outCount(dims)),
      kt1: await readbackBits(gpu.device, kt1.out, outCount(dims1)),
    };
  });
};

const ZERO_FRAME_CASES: readonly Conv3dCase[] = [
  {
    // Small, scalar (kFlat = 2·3·3·2 = 36, but Wout = 5 gives scalar), Kh ≠ Kw
    name: "small [2,3,4,5] * W[3,2,3,3,2] padding=[0,1,1]",
    channelsIn: 2,
    channelsOut: 3,
    input: [3, 4, 5],
    kernel: [3, 3, 2],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
  },
  {
    name: "full size conv_in 16→384 k3 [16,3,32,32]",
    channelsIn: 16,
    channelsOut: 384,
    input: [3, 32, 32],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size time_conv 384→768 (3,1,1) [384,3,32,32]",
    channelsIn: 384,
    channelsOut: 768,
    input: [3, 32, 32],
    kernel: [3, 1, 1],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size up3 96→96 k3 [96,3,256,256]",
    channelsIn: 96,
    channelsOut: 96,
    input: [3, 256, 256],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f16",
  },
  {
    name: "full size head 96→3 k3 [96,3,256,256]",
    channelsIn: 96,
    channelsOut: 3,
    input: [3, 256, 256],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    storage: "f32",
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "identity check (2): Kt = 3 with the first 2 frames zero matches Kt = 1 with the last slice (only signed zero allowed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of ZERO_FRAME_CASES) {
        const { kt3, kt1 } = await runZeroFrameIdentity(gpu, testCase);
        assertEquals(
          firstMismatches(kt1, kt3, true),
          [],
          `${testCase.name}: the bit pattern differs from Kt = 1 with the last slice`,
        );
        assertNotConstant(kt3, testCase.name);
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// 4. Through a Session (wiring into the executor)
// ---------------------------------------------------------------------------

/** The poison value 0xDEADBEEF read as f32 (a non-zero finite value — the same as tests/gpu_full_write_test.ts). */
const POISON = new Float32Array(new Uint32Array([0xdeadbeef]).buffer)[0];

const conv3dAttrs = (padding: Triple, groups = 1): Record<string, unknown> => ({
  stride: [1, 1, 1],
  padding: [...padding],
  dilation: [1, 1, 1],
  groups,
});

/**
 * A 2-node graph of "an identity cast that produces the poison values" + conv3d (the same setup as
 * `poisonGraph` in tests/gpu_full_write_test.ts — the cast's output has no consumers, returns to the
 * pool at the node boundary and is handed out again for conv3d's output allocation).
 */
const poisonGraph = (
  outShape: readonly number[],
  x: readonly number[],
  w: readonly number[],
  padding: Triple,
): DeclarationJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", "conv3d"] },
    symbols: [],
    inputs: [
      { name: "seed", dtype: "f32", shape: [count] },
      { name: "x", dtype: "f32", shape: [...x] },
      { name: "w", dtype: "f32", shape: [...w] },
      { name: "b", dtype: "f32", shape: [w[0]] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      poison: { dtype: "f32", shape: [count] },
      y: { dtype: "f32", shape: [...outShape] },
    },
    nodes: [
      { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
      { op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: conv3dAttrs(padding) },
    ],
  };
};

/**
 * MUST: **have the v4 path and the scalar path as a pair** (the write-out guard is a different
 * expression per variant). The shapes span 2 or more m tiles and n tiles (collapsed into less than
 * one tile, the guard is never false).
 */
const POISON_CASES: readonly {
  readonly name: string;
  readonly outShape: readonly number[];
  readonly x: readonly number[];
  readonly w: readonly number[];
  readonly padding: Triple;
  readonly key: string;
}[] = [
  {
    // v4 (kFlat = 4, Wout = 8). M = 70 -> 3 m tiles of 32 rows; N = 3·9·8 = 216 -> 2 n tiles
    name: "conv3d v4 [4,3,9,8] * W[70,4,1,1,1]",
    outShape: [70, 3, 9, 8],
    x: [4, 3, 9, 8],
    w: [70, 4, 1, 1, 1],
    padding: [0, 0, 0],
    key: conv3dIgemmKey("f32", true, GEMM_MTILE_SMALL),
  },
  {
    // scalar (kFlat = 2·9 = 18), M = 70, N = 3·9·7 = 189
    name: "conv3d scalar [2,3,9,7] * W[70,2,1,3,3]",
    outShape: [70, 3, 9, 7],
    x: [2, 3, 9, 7],
    w: [70, 2, 1, 3, 3],
    padding: [0, 1, 1],
    key: conv3dIgemmKey("f32", false, GEMM_MTILE_SMALL),
  },
  {
    // M = 128 (2 m tiles of 64 rows — the side where the predicate picks 64 rows), v4, with temporal padding
    name: "conv3d v4 m64 [4,2,3,8] * W[128,4,3,1,1]",
    outShape: [128, 2, 3, 8],
    x: [4, 2, 3, 8],
    w: [128, 4, 3, 1, 1],
    padding: [1, 0, 0],
    key: conv3dIgemmKey("f32", true, GEMM_TILE),
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "conv3d leaves not a single poison word, and the executor uses the implicit GEMM of the default geometry (full-write / real GPU)",
  async () => {
    for (const testCase of POISON_CASES) {
      const gpu = await acquireGpu();
      const graph = poisonGraph(testCase.outShape, testCase.x, testCase.w, testCase.padding);
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(graph, []),
        GRAPH_NAME,
      );
      try {
        const count = testCase.outShape.reduce((total, dim) => total * dim, 1);
        const outputs = await session.run({
          seed: fill([count], () => POISON),
          x: fill(testCase.x, (i) => 0.5 + (i % 11) * 0.25),
          w: fill(testCase.w, (i) => 0.25 + (i % 7) * 0.5),
          b: fill([testCase.w[0]], (i) => 0.125 + i * 0.5),
        });
        const diagnostics = session.diagnostics();
        assert(
          (diagnostics.lastRun?.reuseCount ?? 0) >= 1,
          `${testCase.name}: no pool reuse happened (the poison check sees nothing)`,
        );
        assertEquals(outputs["y"].shape, [...testCase.outShape], testCase.name);
        assertEquals(
          [...outputs["y"].data].filter((value) => value === POISON),
          [],
          `${testCase.name}: poison values remain`,
        );
        // The keys of the pipelines that actually ran (the values are the same for every variant, so only the keys are evidence of the dispatch)
        const keys = requireCensus(diagnostics.lastRunPipelines, testCase.name)
          .map((row) => row.key)
          .filter((key) => key.startsWith("conv3d:"));
        assertEquals(keys, [testCase.key], `${testCase.name}: the variant exercised`);
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a conv3d shape whose N tile count exceeds the dispatch limit is a DispatchLimitError (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // MUST: the threshold is derived from **the default geometry's tileN** (writing the edge as a
    // constant would drift: the moment the geometry changes, no throw happens and only assertRejects
    // fails quietly — the same as gpu_runtime_executor_test.ts).
    const tileN = gemmTileN(gemmMTileGeometry(GEMM_MTILE_SMALL));
    const n = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph = singleOpDeclaration(
      "conv3d",
      [[1, 1, 1, n], [1, 1, 1, 1, 1], [1]],
      [[1, 1, 1, n]],
      {
        attrs: conv3dAttrs([0, 0, 0]),
      },
    );
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(graph, []),
      GRAPH_NAME,
    );
    try {
      await assertRejects(
        () =>
          session.run({
            x0: fill([1, 1, 1, n], () => 1),
            x1: fill([1, 1, 1, 1, 1], () => 2),
            x2: fill([1], () => 0.5),
          }),
        DispatchLimitError,
      );
      assertEquals(session.diagnostics().submit.dispatchCount, 0, "a failed run queues nothing");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d with groups > 1 fails at Session build (the subset the GPU implements — ADR 0118 decision 1)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const graph = singleOpDeclaration(
        "conv3d",
        [[6, 2, 4, 4], [6, 2, 1, 3, 3], [6]],
        [[6, 2, 4, 4]],
        { attrs: conv3dAttrs([0, 1, 1], 3) },
      );
      await assertRejects(
        async () =>
          await createSessionFromContainer(gpu, await openModelBytes(graph, []), GRAPH_NAME),
        ExecutionError,
        "conv3d groups 3 cannot run on the GPU",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * f16-stored weights (Wan's quant preset — ADR 0118 decision 7) stay **resident in compressed form**
 * and the `:wf16` variant runs. Covers the wiring eligibility (WEIGHT_SLOTS) -> residency -> the
 * derivation phase's `weightStorage` -> the key in one test.
 */
test.skipIf(!GPU_AVAILABLE)(
  "conv3d's f16 weights stay resident in compressed form, and the w=f16 variant matches the CPU reference (real GPU)",
  async () => {
    const xShape = [4, 3, 5, 8];
    const wShape = [10, 4, 3, 3, 3];
    const outShape = [10, 1, 5, 8];
    const x = fill(xShape, SIGNED);
    const quantized = quantizeF16(
      generate(
        wShape.reduce((a, b) => a * b, 1),
        WEIGHT,
      ),
    );
    const bias = generate(10, BIAS);
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["conv3d"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: xShape }],
      outputs: ["y"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: wShape },
        b: { dtype: "f32", shape: [10] },
        y: { dtype: "f32", shape: outShape },
      },
      nodes: [{ op: "conv3d", ins: ["x", "w", "b"], outs: ["y"], attrs: conv3dAttrs([0, 1, 1]) }],
    };
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(declaration, [
        { graph: GRAPH_NAME, initializer: "w", bytes: quantized.bytes, encoding: { codec: "f16" } },
        { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
      ]),
      GRAPH_NAME,
    );
    try {
      const outputs = await session.run({ x });
      const diagnostics = session.diagnostics();
      // Resident in compressed form (not fallen back to CPU expansion) — the element count 1,080 is even, so there is no padding
      assertEquals(diagnostics.storage.residentCompressedBytes, quantized.bytes.byteLength);
      const keys = requireCensus(diagnostics.lastRunPipelines, "conv3d f16")
        .map((row) => row.key)
        .filter((key) => key.startsWith("conv3d:"));
      // kFlat = 108, Wout = 8 -> v4 / M = 10 -> 32 rows
      assertEquals(keys, [conv3dIgemmKey("f16", true, GEMM_MTILE_SMALL)]);
      const reference = referenceConv3d(
        refTensor(xShape, x.data),
        refTensor(wShape, quantized.values),
        refTensor([10], bias),
        conv3dAttrs([0, 1, 1]),
      );
      assertEquals(outputs["y"].shape, outShape);
      const report = allclose(
        outputs["y"].data as Float32Array,
        reference.data as Float32Array,
        CPU_REFERENCE_TOLERANCE,
      );
      assert(report.pass, `disagrees with the CPU reference (maxAbs ${report.maxAbsError})`);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);
