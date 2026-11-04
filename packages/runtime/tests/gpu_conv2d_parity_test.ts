// **Bit-identity check** of conv2d implicit GEMM (ADR 0024). The most important test of this
// iteration.
//
// On the same input buffers,
//
//   (1) the direct convolution kernel (`conv2d:v1:…:direct:wg256` — kept permanently for groups > 1)
//   (2) the implicit GEMM kernel (`conv2d:v3:…:igemm64x64…`)
//
// run on **the same real GPU**, and their outputs are compared as **f32 bit patterns**. The point is
// to compare by exact `Uint32Array` equality rather than allclose: a change in the rounding sequence
// that a tolerance would hide (e.g. adding the bias at the store instead of as the initial value of
// `acc`) can be detected only here.
//
// Why it holds (design recon §5.3): ascending flat k = the direct kernel's triple ascending
// `(ic, kh, kw)`, ascending K tiles of 16, the bias comes **before** the reduction in both, and for
// padding "skip without adding" and "multiply by 0 and add" agree because `a + 0.0 == a` (for
// finite `a`).
//
// **The only exception = signed zero**: adding a padding-derived `+0.0` where the partial sum is
// exactly `−0.0` flips it to `+0.0` (the direct kernel adds nothing, so `−0.0` remains). It happens
// only in elements whose output is exactly 0, so this test avoids that region by **making the bias
// non-zero in every case**.
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
  CONV2D_SCALE_BINDING,
  CONV2D_WORKGROUP_SIZE,
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dKey,
  conv2dParams,
  conv2dUsesVec4,
  conv2dWgsl,
} from "../src/kernels/conv2d.ts";
import { allclose } from "../src/reference/allclose.ts";
import { referenceConv2d, refTensor } from "../src/reference/ops.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import type { GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
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
 * this set of cases: K ≤ 72, ranges |x| ≤ 2 / |w| ≤ 0.25). **An implementation bug's error is
 * O(1)**, more than 6 orders of magnitude away.
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/** MUST: the bias is non-zero (avoids the signed-zero region — the caveat at the top). */
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
  readonly strideH: number;
  readonly strideW: number;
  readonly paddingH: number;
  readonly paddingW: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly storage: WeightStorage;
  /** The variant the case should exercise (stated explicitly so that a predicate mix-up cannot hide behind "both on the same variant"). */
  readonly expectVec4: boolean;
};

const outLength = (
  input: number,
  padding: number,
  dilation: number,
  kernel: number,
  stride: number,
): number => Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

const dimsOf = (testCase: ParityCase): Conv2dDims => ({
  batch: testCase.batch,
  channelsIn: testCase.channelsIn,
  channelsOut: testCase.channelsOut,
  heightIn: testCase.heightIn,
  widthIn: testCase.widthIn,
  heightOut: outLength(
    testCase.heightIn,
    testCase.paddingH,
    testCase.dilationH,
    testCase.kernelH,
    testCase.strideH,
  ),
  widthOut: outLength(
    testCase.widthIn,
    testCase.paddingW,
    testCase.dilationW,
    testCase.kernelW,
    testCase.strideW,
  ),
  kernelH: testCase.kernelH,
  kernelW: testCase.kernelW,
  strideH: testCase.strideH,
  strideW: testCase.strideW,
  paddingH: testCase.paddingH,
  paddingW: testCase.paddingW,
  dilationH: testCase.dilationH,
  dilationW: testCase.dilationW,
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
  // The axis of conv2d's per-channel scale is 0 (output channels — ADR 0019)
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
): Promise<{
  readonly direct: Uint32Array;
  readonly igemm: Uint32Array;
  readonly v4: boolean;
  readonly reference: Float32Array;
}> => {
  const dims = dimsOf(testCase);
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const x = fill([dims.batch, dims.channelsIn, dims.heightIn, dims.widthIn], SIGNED);
    const weight = fill([dims.channelsOut, dims.channelsIn, dims.kernelH, dims.kernelW], WEIGHT);
    const bias = fill([dims.channelsOut], BIAS);
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

    const count = dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut;
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
        entries.push({ binding: CONV2D_SCALE_BINDING, resource: { buffer: scaleBuffer } });
      }
      return {
        group: gpu.device.createBindGroup({ layout, entries }),
        out,
      };
    };

    // (1) direct kernel (grid-stride)
    const directKey = conv2dKey(testCase.storage);
    const { pipeline: directPipeline, layout: directLayout } = await cache.get(
      directKey,
      conv2dWgsl(testCase.storage),
    );
    const direct = bind(directLayout, conv2dParams(dims));
    scheduler.dispatch(
      directPipeline,
      direct.group,
      [gridStrideWorkgroups(count, CONV2D_WORKGROUP_SIZE, limit), 1, 1],
      directKey,
    );

    // (2) implicit GEMM (1 workgroup = 1 output tile)
    const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const igemmKey = conv2dIgemmKey(testCase.storage, v4, mTile);
    const { pipeline: igemmPipeline, layout: igemmLayout } = await cache.get(
      igemmKey,
      conv2dIgemmWgsl(testCase.storage, v4, mTile),
    );
    const igemm = bind(igemmLayout, conv2dIgemmParams(dims));
    // MUST: the dispatch edges are derived from **the same resolution point** as generation and the key
    // (`gemmMTileGeometry`). The m-tile variant moves only the row edge; the n edge is the geometry's
    // tileN (the same for both variants).
    const geometry = gemmMTileGeometry(mTile);
    scheduler.dispatch(
      igemmPipeline,
      igemm.group,
      [
        tiledWorkgroups(dims.heightOut * dims.widthOut, gemmTileN(geometry), limit, testCase.name),
        tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, testCase.name),
        tiledWorkgroups(dims.batch, 1, limit, testCase.name),
      ],
      igemmKey,
    );

    await scheduler.flush();
    // The CPU reference is built from **the quantized weights expanded back** (so that the f16 / i8
    // storage error does not turn into "a kernel error"). Dequant is per element (ADR 0019).
    const reference = referenceConv2d(
      refTensor(x.shape, x.data),
      refTensor(weight.shape, payload.values),
      refTensor(bias.shape, bias.data),
      {
        stride: [dims.strideH, dims.strideW],
        padding: [dims.paddingH, dims.paddingW],
        dilation: [dims.dilationH, dims.dilationW],
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
    // A shrunk version of an actual census shape (VAE decoder): 3×3, pad 1, same. Pins that with B=2 the
    // batch folds into N, and covers 2 m tiles (Cout=70), 2 n tiles (N=96) and a K remainder
    // (72 = 16·4 + 8) in one case.
    name: "shrunk census shape 3×3 pad1 [2,8,6,8] * W[70,8,3,3] (f32/v4)",
    batch: 2,
    channelsIn: 8,
    channelsOut: 70,
    heightIn: 6,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: true,
  },
  {
    // kFlat = 2·3·3 = 18 -> `kFlat % 4 ≠ 0` selects the scalar variant, and `K % 16 ≠ 0` exercises the
    // zero fill of the K remainder (in the census every K is a multiple of 16, so the real E2E has no
    // detector for this path).
    name: "K remainder scalar [1,2,5,7] * W[5,2,3,3] (f32)",
    batch: 1,
    channelsIn: 2,
    channelsOut: 5,
    heightIn: 5,
    widthIn: 7,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // Kh ≠ Kw, stride / padding with H≠W, Cin ≠ Cout (both at least 2). stride_w = 3, so the v4
    // 4-column contiguous read does not hold and it falls to scalar.
    name: "asymmetric Kh≠Kw stride=[2,3] padding=[1,0] [1,3,9,11] * W[4,3,2,4]",
    batch: 1,
    channelsIn: 3,
    channelsOut: 4,
    heightIn: 9,
    widthIn: 11,
    kernelH: 2,
    kernelW: 4,
    strideH: 2,
    strideW: 3,
    paddingH: 1,
    paddingW: 0,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // dilation with H≠W too (exercises a `kh·dh` / `kw·dw` mix-up on the v4 path)
    name: "dilation=[3,2] [1,4,12,10] * W[6,4,3,2] (f32/v4)",
    batch: 1,
    channelsIn: 4,
    channelsOut: 6,
    heightIn: 12,
    widthIn: 10,
    kernelH: 3,
    kernelW: 2,
    strideH: 1,
    strideW: 1,
    paddingH: 0,
    paddingW: 0,
    dilationH: 3,
    dilationW: 2,
    storage: "f32",
    expectVec4: true,
  },
  {
    // **The only detector for MUST ②**: Hout = Wout = 2 gives N = 4. `kFlat = 20` and `stride_w = 1`
    // meet the v4 conditions, so writing the predicate as `N % 4` would pick v4. Moreover Win = 6, so
    // the quad's contiguous-read guard (`ix0 + 3 < width_in`) lets it through, and 4 columns that span
    // output rows are read from the same input row = a wrong value with no exception. A mix-up that
    // would never surface on the measured shapes (every Wout is a multiple of 4); only this case turns
    // it red.
    name: "Hout=Wout=2 (N%4==0 but Wout%4≠0) [1,4,2,6] * W[3,4,1,5]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 3,
    heightIn: 2,
    widthIn: 6,
    kernelH: 1,
    kernelW: 5,
    strideH: 1,
    strideW: 1,
    paddingH: 0,
    paddingW: 0,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // f16 storage x v4 (the quad expansion `dequant4` relies on the flat index being a multiple of 4)
    name: "f16 v4 [1,8,6,8] * W[12,8,3,3]",
    batch: 1,
    channelsIn: 8,
    channelsOut: 12,
    heightIn: 6,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f16",
    expectVec4: true,
  },
  {
    // f16 storage x scalar. kFlat = 1·3·3 = 9 is **odd**, so the start of a weight row alternates
    // between the high and low half of a word (the detector for choosing the pair by the parity of the
    // flat index — the pitfall of ADR 0018).
    name: "f16 scalar row length 9 (odd) [1,1,5,6] * W[4,1,3,3]",
    batch: 1,
    channelsIn: 1,
    channelsOut: 4,
    heightIn: 5,
    widthIn: 6,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f16",
    expectVec4: false,
  },
  {
    // **Detector for MUST ④**: the i8 scale runs along rows (= output channels). Cout = 70 gives 2 m
    // tiles, so a mix-up that takes a tile-relative row or a column instead of `arow` shows in the
    // values (a shape that fits in one tile matches by accident with `wid.y = 0` — the same shape as
    // detection limit ③ of ADR 0022).
    name: "i8 v4 2 m tiles [1,4,2,4] * W[70,4,1,1]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 70,
    heightIn: 2,
    widthIn: 4,
    kernelH: 1,
    kernelW: 1,
    strideH: 1,
    strideW: 1,
    paddingH: 0,
    paddingW: 0,
    dilationH: 1,
    dilationW: 1,
    storage: "i8",
    expectVec4: true,
  },
  {
    // i8 x the scalar variant (taking `unpack4xI8`'s remainder mod 4 from the flat index + 2 m tiles).
    // kFlat = 2·3·3 = 18 is not a multiple of 4, so it falls to scalar.
    name: "i8 scalar 2 m tiles [1,2,4,5] * W[66,2,3,3]",
    batch: 1,
    channelsIn: 2,
    channelsOut: 66,
    heightIn: 4,
    widthIn: 5,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "i8",
    expectVec4: false,
  },
  {
    // **The main shape for the 32-row m-tile variant** (next to ADR 0024): the census's Cout=96
    // (`96 % 64 == 32`). 64 rows give 2 tiles (1.33× waste), 32 rows give 3 tiles (no waste), and the
    // variant's premise is that **the output is bit-identical either way**. Being i8, it also doubles as
    // the detector for 3 m tiles x the row scale.
    name: "i8 Cout=96（M%64==32）[1,4,4,8] * W[96,4,3,3]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 96,
    heightIn: 4,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "i8",
    expectVec4: true,
  },
  {
    // A shape with M % 64 == 0 (the predicate picks 64 rows). **Forcing** 32 rows splits it into 2 tiles,
    // so the claim "changing the tile shape does not change the values" is exercised with a moving tile
    // boundary.
    name: "Cout=64（M%64==0）[1,3,3,8] * W[64,3,3,3]",
    batch: 1,
    channelsIn: 3,
    channelsOut: 64,
    heightIn: 3,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: false,
  },
  {
    // The 33–64 band (the predicate picks 64 rows). Forcing 32 rows makes **the second tile use only one
    // row**, which exercises the edge tile's row guard (`orow < dims.m`) and the out-of-range read of
    // bias-first.
    name: "f16 Cout=33 (33–64 band) [1,4,3,8] * W[33,4,3,3]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 33,
    heightIn: 3,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "f16",
    expectVec4: true,
  },
  {
    // M < 32 (one tile with 2/3 of its rows empty). It is one tile with both 32 and 64 rows, but this
    // pins on both sides that the `acc` of the empty rows is discarded by the store's row guard.
    name: "i8 Cout=20（M<32）[2,4,3,8] * W[20,4,3,3]",
    batch: 2,
    channelsIn: 4,
    channelsOut: 20,
    heightIn: 3,
    widthIn: 8,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    storage: "i8",
    expectVec4: true,
  },
  {
    // Padding wider than the kernel span (both ends of the output see only the padding area = the
    // error of reading out-of-range positions with clamped indices always shows in the values)
    name: "padding=[2,0] excess padding [1,4,2,8] * W[5,4,3,1]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 5,
    heightIn: 2,
    widthIn: 8,
    kernelH: 3,
    kernelW: 1,
    strideH: 1,
    strideW: 1,
    paddingH: 2,
    paddingW: 0,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
    expectVec4: true,
  },
  {
    // **The only detector for the third v4 condition (`strideW == 1`)**. kFlat = 4·1·4 = 16 (%4==0)
    // and Wout = (16−4)/4 + 1 = 4 (%4==0) meet the other 2 conditions, but strideW = 4, so v4 cannot be
    // chosen (a 4-column contiguous read would read stride-apart positions as contiguous = a wrong value
    // with no exception). Dropping `strideW === 1` from the predicate turns only this case red.
    name: "strideW=4 does not fall into v4 [1,4,4,16] * W[8,4,1,4]",
    batch: 1,
    channelsIn: 4,
    channelsOut: 8,
    heightIn: 4,
    widthIn: 16,
    kernelH: 1,
    kernelW: 4,
    strideH: 1,
    strideW: 4,
    paddingH: 0,
    paddingW: 0,
    dilationH: 1,
    dilationW: 1,
    storage: "f32",
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
 * (2) numeric agreement with the CPU reference (`referenceConv2d`) (the detector for both kernels
 *     being wrong **together** — a bit comparison alone cannot notice "they make the same mistake")
 * (3) the vacuity check (if the output is constant, agreement verifies nothing)
 */
const assertParity = (
  testCase: ParityCase,
  where: string,
  result: {
    readonly direct: Uint32Array;
    readonly igemm: Uint32Array;
    readonly v4: boolean;
    readonly reference: Float32Array;
  },
): void => {
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
  "conv2d implicit GEMM **matches the direct kernel bit for bit** (real GPU)",
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
 * The permanent check for **the 32-row m-tile variant** (next to ADR 0024).
 *
 * Runs every case **forcibly** with 32-row tiles (including shapes for which the predicate picks 64
 * rows). The tile shape is designed to change only "which workgroup handles which outputs", so the
 * same direct-kernel comparison as for the 64-row version must pass unchanged.
 *
 * MUST: cover every shape independently of whether the predicate ({@link conv2dIgemmMTile}) picks it.
 * Checking only the shapes the predicate picks would let an error that shifts the predicate's
 * boundary pass green, with the variant "merely no longer running".
 *
 * Coverage: M a multiple of 32 (96 / 64 / 20 give 1–3 tiles) / not a multiple (70 / 66 / 33 / 5 / 4 /
 * 3 / 12) / the 33–64 band (33 / the second tile of 70) / M < 32 (20 / 12 / 5 / 4 / 3) / **2 or more m
 * tiles x i8** (96 = 3 tiles, 70 = 3 tiles, 66 = 3 tiles — the only detector of a scale-axis mix-up).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the conv2d 32-row m-tile variant also matches the direct kernel bit for bit (real GPU)",
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
 * The executor's dispatch between kernels (ADR 0024). `groups == 1` goes to implicit GEMM,
 * `groups > 1` to the direct kernel (kept as a permanent differential oracle).
 *
 * MUST: look at **the pipeline keys that actually ran**, not at values. Both kernels are correct, so
 * matching outputs are no evidence of the dispatch.
 */
const conv2dKeysUsed = async (
  channels: number,
  groups: number,
): Promise<{ readonly keys: ReadonlySet<string>; readonly profile: GeometryProfile }> => {
  const gpu = await acquireGpu();
  const profile = deviceGeometryProfile(gpu);
  const graph = singleOpDeclaration(
    "conv2d",
    [[1, channels, 4, 4], [channels, channels / groups, 3, 3], [channels]],
    [[1, channels, 4, 4]],
    { attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups } },
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    await session.run({
      x0: fill([1, channels, 4, 4], SIGNED),
      x1: fill([channels, channels / groups, 3, 3], WEIGHT),
      x2: fill([channels], BIAS),
    });
    return {
      keys: new Set(
        requireCensus(session.diagnostics().lastRunPipelines, "conv").map((row) => row.key),
      ),
      profile,
    };
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the executor dispatches between the 2 kernels by groups (1 = igemm / >1 = direct, real GPU)",
  async () => {
    // Cin = Cout = 6, 3×3 -> kFlat = 54 (groups=1) is not a multiple of 4, so the scalar variant.
    // Cout=6 gives `6 % 64 == 6`, so the m tile has 32 rows ({@link conv2dIgemmMTile}).
    const small = await conv2dKeysUsed(6, 1);
    assertEquals(
      small.keys,
      new Set([conv2dIgemmKey("f32", false, GEMM_MTILE_SMALL, small.profile.conv2d.rows32)]),
    );
    assertEquals((await conv2dKeysUsed(6, 3)).keys, new Set([conv2dKey("f32")]));
    // depthwise (groups = Cin = Cout) also goes to the direct kernel
    assertEquals((await conv2dKeysUsed(6, 6)).keys, new Set([conv2dKey("f32")]));
  },
);

/**
 * The check that the m-tile choice is **wired into the executor** (next to ADR 0024).
 *
 * MUST: look at **the pipeline keys that actually ran**, not at values. Both tile shapes give
 * bit-identical outputs, so matching values are no evidence of the wiring (with the predicate
 * knocked out, every numeric test would stay green).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the executor dispatches between 64/32-row m tiles by M%64 (real GPU)",
  async () => {
    // Cout = 96 (the census's main shape — `96 % 64 == 32`) -> 32 rows
    const rows32 = await conv2dKeysUsed(96, 1);
    assertEquals(
      rows32.keys,
      new Set([conv2dIgemmKey("f32", true, GEMM_MTILE_SMALL, rows32.profile.conv2d.rows32)]),
    );
    // Cout = 64 (`64 % 64 == 0`) -> stays at 64 rows (also confirms that the existing key does not move)
    const rows64 = await conv2dKeysUsed(64, 1);
    assertEquals(
      rows64.keys,
      new Set([conv2dIgemmKey("f32", true, GEMM_TILE, rows64.profile.conv2d.rows64)]),
    );
  },
);
