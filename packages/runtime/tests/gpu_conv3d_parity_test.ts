// conv3d（unbatched・implicit GEMM — ADR 0118 決定 1）の実 GPU の門。
//
// 1. **CPU 参照との照合**（`referenceConv3d` — f64 縮約なので帯は conv2d の parity テストと同じ
//    atol 1e-5 / rtol 1e-5）。形 8 種 = chunk グラフの 4 形（16→384 k3・384→768 (3,1,1)・96→96 k3・
//    96→3 k3）を**空間を縮めた縮小形** + K 端数・Wout%4≠0・stride≠1・dilation≠1 の小形。
//    m タイル 64 / 32 の両方で流し、2 つの出力が Uint32 で一致することも見る（タイル形は担当割り
//    だけを変える）。
// 2. **恒等門 ①**（Uint32 完全一致）: Kt = 1 の conv3d ≡ フレームをバッチに置いた conv2d の
//    implicit GEMM。平坦 K の並び `(ic, kt, kh, kw)` は Kt = 1 で conv2d の `(ic, kh, kw)` と同じ列に
//    なり、bias-first・範囲外 0 も共有の骨格が持つので、丸め列まで一致する。小形 3 種と、chunk
//    グラフの 4 形の**実寸**（JS の CPU 参照では分単位になる大きさ — GPU どうしで見る）。
//    tolerance に隠れる丸め列の変化（bias を store 側で足す等）は**ここでしか検出できない**。
// 3. **恒等門 ②**: 時間の先頭 2 枚がゼロの Kt = 3 ≡ 重みの最終スライスの Kt = 1。ゼロフレームの
//    積は ±0 で、`a + (±0) = a`（a ≠ 0）なので値は一致する。唯一の例外は符号付きゼロ（部分和が
//    ちょうど 0 のときだけ ±0 が転ぶ — ADR 0024 決定 3）なので、そこだけ差を許す。
// 4. **full-write の毒値**・**dispatch 上限の fail loudly**・**groups > 1 の構築時拒否**・**踏み分け
//    キー**・**f16 常駐**は Session 経由（executor への結線を兼ねる）。
//
// MUST: 恒真化しないこと。①と②は別のパイプライン・別の WGSL・別の生成関数で、出力が定数で
// ないことも見る。bias は全ケース非ゼロ（符号付きゼロの領域を避ける）。

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
 * CPU 参照との突合の許容誤差（conv2d の parity テストと同じ帯 — 同じ骨格・同じ縮約順）。参照は
 * JS の f64 で縮約するので f32 の逐次加算とは丸めが違い、ビット一致は原理的に成立しない。
 * **実装バグの誤差は O(1)** なので桁で離れている。
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
/** 重みは小さめ（K = 2,592 の縮約でも出力が O(1) に収まる — 帯の rtol を意味のある大きさに保つ）。 */
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.0041;
/** MUST: bias は非ゼロ（符号付きゼロの領域を避ける — 冒頭の caveat）。 */
const BIAS = (i: number): number => 0.375 + (i % 5) * 0.25;

/** `fill` の大寸法版（実寸の 3,770 万要素で JS 配列を経由しない）。 */
const generate = (
  count: number,
  value: (index: number) => number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(count);
  for (let i = 0; i < count; i += 1) data[i] = value(i);
  return data;
};

type Triple = readonly [number, number, number];

type Conv3dCase = {
  readonly name: string;
  readonly channelsIn: number;
  readonly channelsOut: number;
  /** 入力の [T, H, W]。 */
  readonly input: Triple;
  /** カーネルの [Kt, Kh, Kw]。 */
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
    )
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
 * 重み格納の変種ぶんのペイロード。`values` は**格納から復号し直した f32**（fake-quant 後の重み）で、
 * CPU 参照と恒等門の相手側はこれを重みとして使う（素の f32 を渡すと格納の丸めが「カーネルの
 * 誤り」に化ける）。
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
  // conv3d の per-channel scale の軸は 0（出力チャネル — ADR 0019 / 0024 決定 6）
  const quantized = quantizeI8(data, shape, 0);
  return {
    bytes: alignI8Payload(quantized.bytes),
    scale: quantized.scale,
    values: quantized.values,
  };
};

/** 1 回の GPU 作業単位（キャッシュ・スケジューラ・アリーナを束ねて最後に必ず破棄する）。 */
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
  // vec4 束縛（f32 の v4 経路）が末尾 quad を落とさないよう 16 バイトへ丸める
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

/** 束縛（0 dims / 1 x / 2 重み / 3 bias / 4 出力 / 5 scale）を組んで 1 dispatch 積む。 */
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

/** conv3d の implicit GEMM を 1 本積む（dispatch は本番と同じ `conv3dIgemmWorkgroups`）。 */
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

/** ビット列の食い違い（先頭 4 件・`allowSignedZero` なら ±0 どうしは一致扱い）。 */
const firstMismatches = (
  expected: Uint32Array,
  actual: Uint32Array,
  allowSignedZero = false,
): readonly string[] => {
  assertEquals(actual.length, expected.length, "比較する要素数");
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

/** 恒真化の門（出力が定数なら一致は何も検証していない）。 */
const assertNotConstant = (bits: Uint32Array, where: string): void => {
  const first = bits[0];
  assert(bits.some((value) => value !== first), `${where}: 出力が定数（一致が恒真になっている）`);
};

// ---------------------------------------------------------------------------
// 1. CPU 参照との照合（形 8 種 × m タイル 64 / 32）
// ---------------------------------------------------------------------------

const ONE: Triple = [1, 1, 1];

type ParityCase = Conv3dCase & {
  /** 踏むはずの変種（判定の取り違えが「両方同じ変種」で紛れないように明示する）。 */
  readonly expectVec4: boolean;
};

const PARITY_CASES: readonly ParityCase[] = [
  {
    // chunk グラフの conv_in（16→384 k3・時間 padding 0〈cache 2 + 1 フレーム〉・空間 same）の
    // 空間縮小形。M = 384 で m タイル 6 枚・K = 432 = 16·27。
    name: "chunk conv_in 縮小 16→384 k3 [16,3,6,8]",
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
    // time_conv（384→768・カーネル (3,1,1)）の空間縮小形。f16 格納（Wan の席）× v4。
    name: "chunk time_conv 縮小 384→768 (3,1,1) [384,3,4,8] f16",
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
    // up3 の 96→96 k3 の空間縮小形（Tout = 4 = 時間 upsample 後の chunk）。M = 96（96%64 == 32 で
    // m タイル述語は 32 行）× i8 — m タイル 3 枚（32 行）/ 2 枚（64 行）で行 scale を踏む。
    name: "chunk 96→96 k3 縮小 [96,6,6,8] i8",
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
    // head（96→3 k3）の空間縮小形。M = 3（タイル 1 枚の 1 割未満）・N = 4·6·8 = 192（n タイル 2 枚）。
    name: "chunk head 縮小 96→3 k3 [96,6,6,8]",
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
    // K 端数: kFlat = 2·27 = 54（4 の倍数でない → スカラ変種・16 の倍数でない → 最終 K タイルが 0 埋め）。
    // 全軸 padding 1 で範囲外 0 を 3 軸とも踏む（クランプ読みの検出器）。Cin ≠ Cout。
    name: "K 端数 スカラ [2,4,5,7] * W[5,2,3,3,3] padding 1",
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
    // **v4 判定 2 条件目（Wout%4）の検出器**: Tout = Hout = Wout = 2 で N = 8（%4 == 0）・
    // kFlat = 4·5 = 20（%4 == 0）・strideW = 1。判定を N%4 で書くと v4 が選ばれ、Win = 6 なので
    // quad の連続読みガードも素通りして、出力行をまたいだ 4 列が同じ入力行から読まれる。
    name: "Wout=2（N%4==0 だが Wout%4≠0）[4,2,2,6] * W[3,4,1,1,5]",
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
    // stride ≠ 1 を 3 軸とも非対称に（T 2 / H 2 / W 3）・Kt≠Kh≠Kw・padding も非対称。
    // T: (7+2−2−1)/2+1 = 4 / H: (9+2−1−1)/2+1 = 5 / W: (11−3−1)/3+1 = 3。
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
    // dilation ≠ 1 を 3 軸とも（T 3 / H 3 / W 2）・v4 経路で `kt·dt` / `kh·dh` / `kw·dw` の取り違えを踏む。
    // T: 8−3−1+1 = 5 / H: 12−6−1+1 = 6 / W: 10−2−1+1 = 8。
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

/** CPU 参照と GPU（m タイル 64 / 32 の両方）を 1 ケースぶん。 */
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
  const wShape = [
    dims.channelsOut,
    dims.channelsIn,
    dims.kernelT,
    dims.kernelH,
    dims.kernelW,
  ];
  const x = generate(xShape.reduce((a, b) => a * b, 1), SIGNED);
  const w = generate(wShape.reduce((a, b) => a * b, 1), WEIGHT);
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

Deno.test({
  name: "conv3d の implicit GEMM は CPU 参照と一致する（形 8 種 × m タイル 64 / 32・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of PARITY_CASES) {
        const { m64, m32, v4, reference } = await runParity(gpu, testCase);
        const where = testCase.name;
        assertEquals(v4, testCase.expectVec4, `${where}: 踏んだ変種が想定と違う`);
        // タイル形は「どの workgroup がどの出力を担当するか」だけを変える（数値経路は共通）
        assertEquals(firstMismatches(m64, m32), [], `${where}: m タイル 64 と 32 でビット列が違う`);
        const report = allclose(asFloats(m64), reference, CPU_REFERENCE_TOLERANCE);
        assert(
          report.pass,
          `${where}: CPU 参照と食い違う（maxAbs ${report.maxAbsError} @${report.worstIndex} / 破り ${report.failCount}）`,
        );
        assertNotConstant(m64, where);
      }
      // 述語が実際に 32 行を選ぶ形と 64 行を選ぶ形の両方が含まれていること（空振りしない門）
      const tiles = new Set(PARITY_CASES.map((testCase) => conv2dIgemmMTile(testCase.channelsOut)));
      assertEquals([...tiles].sort(), [GEMM_MTILE_SMALL, GEMM_TILE].sort());
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 2. 恒等門 ①: Kt = 1 の conv3d ≡ フレームごとの conv2d（Uint32 完全一致）
// ---------------------------------------------------------------------------

/** Kt = 1 の conv3d と、フレームをバッチに置いた conv2d の出力を組で返す。 */
const runFrameIdentity = async (
  gpu: GpuContext,
  testCase: Conv3dCase,
): Promise<{
  readonly conv3d: Uint32Array<ArrayBuffer>;
  readonly conv2d: Uint32Array<ArrayBuffer>;
}> => {
  assertEquals(testCase.kernel[0], 1, `${testCase.name}: 恒等門 ① は Kt = 1 の形だけ`);
  assertEquals(testCase.padding[0], 0, `${testCase.name}: 時間 padding 0`);
  assertEquals(testCase.stride[0], 1, `${testCase.name}: 時間 stride 1`);
  const dims = dimsOf(testCase);
  const { channelsIn: cin, timeIn: time, heightIn: height, widthIn: width } = dims;
  const plane = height * width;
  const x = generate(cin * time * plane, SIGNED);
  // フレームをバッチへ: [Cin, T, H, W] → [T, Cin, H, W]
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
  const w = generate(wShape.reduce((a, b) => a * b, 1), WEIGHT);
  // 重みは [Cout, Cin, 1, Kh, Kw] と [Cout, Cin, Kh, Kw] で**同じバイト列**（同じバッファを束縛する）
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
    // conv2d の出力 [T, Cout, Hout·Wout] → conv3d の並び [Cout, T, Hout·Wout]
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
    // スカラ変種・stride / dilation / padding が H / W で非対称・Cin ≠ Cout
    name: "小形 スカラ [3,4,5,6] * W[4,3,1,3,2] stride=[1,2,1] dilation=[1,1,2]",
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
    // v4 変種（kFlat = 72・Wout = 8）
    name: "小形 v4 [8,3,6,8] * W[12,8,1,3,3]",
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
    // M = 70（m タイル 32 行 × 3 枚）× i8（行 scale）× n タイル 2 枚（N = 5·4·8 = 160）
    name: "小形 i8 m32 [4,5,4,8] * W[70,4,1,1,3]",
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
 * chunk グラフの 4 形の**実寸**（潜在 32×32 タイル — ADR 0118 決定 2）を Kt = 1 にした形。
 * 時間カーネルの扱いは恒等門 ② が実寸で見るので、ここはチャネル・空間・フレーム数の実寸で
 * 3D の gather（n → (ot, oy, ox)）と dispatch の辺を踏む（96→96 は N = 6·256² = 393,216 で
 * n タイル 3,072 枚）。
 */
const IDENTITY_REAL: readonly Conv3dCase[] = [
  {
    name: "実寸 conv_in 16→384 (1,3,3) [16,3,32,32]",
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
    name: "実寸 time_conv 384→768 (1,1,1) [384,3,32,32]",
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
    name: "実寸 up3 96→96 (1,3,3) [96,6,256,256]",
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
    name: "実寸 head 96→3 (1,3,3) [96,6,256,256]",
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

Deno.test({
  name:
    "恒等門 ①: Kt = 1 の conv3d はフレームごとの conv2d と Uint32 で一致する（小形 3 種 + 実寸 4 形・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of [...IDENTITY_SMALL, ...IDENTITY_REAL]) {
        const { conv3d, conv2d } = await runFrameIdentity(gpu, testCase);
        assertEquals(
          firstMismatches(conv2d, conv3d),
          [],
          `${testCase.name}: フレームごとの conv2d とビット列が違う`,
        );
        assertNotConstant(conv3d, testCase.name);
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 3. 恒等門 ②: 時間の先頭 2 枚がゼロの Kt = 3 ≡ 重みの最終スライスの Kt = 1
// ---------------------------------------------------------------------------

/**
 * x の時間 3 枚のうち先頭 2 枚をゼロにした Kt = 3 の conv3d（時間 padding 0 → Tout = 1 — chunk
 * グラフの最初の chunk〈ゼロの cache 2 枚 + 1 フレーム〉と同じ形）と、3 枚目だけ・重みの最終
 * スライスだけの Kt = 1 の conv3d を組で返す。
 */
const runZeroFrameIdentity = async (
  gpu: GpuContext,
  testCase: Conv3dCase,
): Promise<{ readonly kt3: Uint32Array<ArrayBuffer>; readonly kt1: Uint32Array<ArrayBuffer> }> => {
  assertEquals(testCase.kernel[0], 3, `${testCase.name}: 恒等門 ② は Kt = 3 の形だけ`);
  assertEquals(testCase.input[0], 3, `${testCase.name}: 時間 3 枚（cache 2 + 1）`);
  // i8 の scale は行（出力チャネル）の amax から引き直すので Kt = 3 と Kt = 1 で別物になる — この門は
  // f32 / f16（scale を持たない格納）だけで張る。
  assert(testCase.storage !== "i8", `${testCase.name}: 恒等門 ② は scale を持たない格納だけ`);
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
  // 最終スライスは**格納から復号し直した値**から切る（f16 / i8 の丸めを両側で揃える）
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
    `${testCase.name}: 最終スライスの再格納で値が動いた`,
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
    // 小形・スカラ（kFlat = 2·3·3·2 = 36 だが Wout = 5 でスカラ）・Kh ≠ Kw
    name: "小形 [2,3,4,5] * W[3,2,3,3,2] padding=[0,1,1]",
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
    name: "実寸 conv_in 16→384 k3 [16,3,32,32]",
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
    name: "実寸 time_conv 384→768 (3,1,1) [384,3,32,32]",
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
    name: "実寸 up3 96→96 k3 [96,3,256,256]",
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
    name: "実寸 head 96→3 k3 [96,3,256,256]",
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

Deno.test({
  name:
    "恒等門 ②: 先頭 2 枚がゼロの Kt = 3 は最終スライスの Kt = 1 と一致する（符号付きゼロのみ許容・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of ZERO_FRAME_CASES) {
        const { kt3, kt1 } = await runZeroFrameIdentity(gpu, testCase);
        assertEquals(
          firstMismatches(kt1, kt3, true),
          [],
          `${testCase.name}: 最終スライスの Kt = 1 とビット列が違う`,
        );
        assertNotConstant(kt3, testCase.name);
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 4. Session 経由（executor への結線）
// ---------------------------------------------------------------------------

/** 毒値 0xDEADBEEF を f32 として読んだもの（0 でない有限値 — tests/gpu_full_write_test.ts と同じ）。 */
const POISON = new Float32Array(new Uint32Array([0xDEADBEEF]).buffer)[0];

const conv3dAttrs = (padding: Triple, groups = 1): Record<string, unknown> => ({
  stride: [1, 1, 1],
  padding: [...padding],
  dilation: [1, 1, 1],
  groups,
});

/**
 * 「毒値を作る恒等 cast」+ conv3d の 2 ノードグラフ（tests/gpu_full_write_test.ts の `poisonGraph` と
 * 同じ仕込み — cast の出力は消費者ゼロでノード境界でプールへ戻り、conv3d の出力確保に配り直される）。
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
 * MUST: **v4 経路とスカラ経路を対で持つ**（書き出しのガードが変種ごとに別の式）。形は m タイル・
 * n タイルとも 2 枚以上に跨がせる（1 タイル未満に潰れるとガードが一度も偽にならない）。
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
    // v4（kFlat = 4・Wout = 8）。M = 70 → m タイル 32 行 × 3 枚・N = 3·9·8 = 216 → n タイル 2 枚
    name: "conv3d v4 [4,3,9,8] * W[70,4,1,1,1]",
    outShape: [70, 3, 9, 8],
    x: [4, 3, 9, 8],
    w: [70, 4, 1, 1, 1],
    padding: [0, 0, 0],
    key: conv3dIgemmKey("f32", true, GEMM_MTILE_SMALL),
  },
  {
    // スカラ（kFlat = 2·9 = 18）・M = 70・N = 3·9·7 = 189
    name: "conv3d スカラ [2,3,9,7] * W[70,2,1,3,3]",
    outShape: [70, 3, 9, 7],
    x: [2, 3, 9, 7],
    w: [70, 2, 1, 3, 3],
    padding: [0, 1, 1],
    key: conv3dIgemmKey("f32", false, GEMM_MTILE_SMALL),
  },
  {
    // M = 128（m タイル 64 行 × 2 枚 — 述語が 64 行を選ぶ側）・v4・時間 padding あり
    name: "conv3d v4 m64 [4,2,3,8] * W[128,4,3,1,1]",
    outShape: [128, 2, 3, 8],
    x: [4, 2, 3, 8],
    w: [128, 4, 3, 1, 1],
    padding: [1, 0, 0],
    key: conv3dIgemmKey("f32", true, GEMM_TILE),
  },
];

Deno.test({
  name:
    "conv3d は毒値を 1 語も残さず、executor は既定幾何の implicit GEMM を踏む（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
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
          `${testCase.name}: プール再利用が起きていない（毒値検査が何も見ていない）`,
        );
        assertEquals(outputs["y"].shape, [...testCase.outShape], testCase.name);
        assertEquals(
          [...outputs["y"].data].filter((value) => value === POISON),
          [],
          `${testCase.name}: 毒値の残存`,
        );
        // 実際に走ったパイプラインのキー（値はどの変種でも同じなので、踏み分けの証拠はキーだけ）
        const keys = requireCensus(diagnostics.lastRunPipelines, testCase.name)
          .map((row) => row.key)
          .filter((key) => key.startsWith("conv3d:"));
        assertEquals(keys, [testCase.key], `${testCase.name}: 踏んだ変種`);
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    }
  },
});

Deno.test({
  name: "conv3d の N タイル数が dispatch 上限を超える形は DispatchLimitError（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // MUST: 閾値は**既定幾何の tileN** から導く（辺を定数で書くと、幾何を変えた瞬間に throw が
    // 起きず assertRejects だけが静かに落ちるドリフトになる — gpu_runtime_executor_test.ts と同じ）。
    const tileN = gemmTileN(gemmMTileGeometry(GEMM_MTILE_SMALL));
    const n = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph = singleOpDeclaration("conv3d", [[1, 1, 1, n], [1, 1, 1, 1, 1], [1]], [[
      1,
      1,
      1,
      n,
    ]], {
      attrs: conv3dAttrs([0, 0, 0]),
    });
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
      assertEquals(session.diagnostics().submit.dispatchCount, 0, "失敗した run は何も積まない");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "conv3d の groups > 1 は Session の構築時に落ちる（GPU の実装済み subset — ADR 0118 決定 1）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
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
        "conv3d の groups 3 は GPU で実行できない",
      );
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * f16 格納の重み（Wan の quant 席 — ADR 0118 決定 7）が**圧縮のまま常駐**し、`:wf16` の変種が走る。
 * 適格判定（WEIGHT_SLOTS）→ 常駐 → 導出相の `weightStorage` → キーの結線を 1 本で見る。
 */
Deno.test({
  name: "conv3d の f16 重みは圧縮のまま常駐し、w=f16 変種が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const xShape = [4, 3, 5, 8];
    const wShape = [10, 4, 3, 3, 3];
    const outShape = [10, 1, 5, 8];
    const x = fill(xShape, SIGNED);
    const quantized = quantizeF16(generate(wShape.reduce((a, b) => a * b, 1), WEIGHT));
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
      // 圧縮のまま常駐（CPU 展開に落ちていない）— 要素数 1,080 は偶数なので詰め物は無い
      assertEquals(diagnostics.storage.residentCompressedBytes, quantized.bytes.byteLength);
      const keys = requireCensus(diagnostics.lastRunPipelines, "conv3d f16")
        .map((row) => row.key)
        .filter((key) => key.startsWith("conv3d:"));
      // kFlat = 108・Wout = 8 → v4 / M = 10 → 32 行
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
      assert(report.pass, `CPU 参照と食い違う（maxAbs ${report.maxAbsError}）`);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});
