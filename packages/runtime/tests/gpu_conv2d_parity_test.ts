// conv2d implicit GEMM（ADR 0024）の**ビット同一の門**。本波で最重要のテスト。
//
// 同じ入力バッファに対し、
//
//   ① 直接畳み込みカーネル（`conv2d:v1:…:direct:wg256` — groups > 1 用に恒久で残る）
//   ② implicit GEMM カーネル（`conv2d:v2:…:igemm64x64…`）
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる丸め列の変化（例: bias を
// `acc` の初期値ではなく store 側で足す）はここでしか検出できない。
//
// 成立の根拠（設計 recon §5.3）: 平坦 k 昇順 = 直接カーネルの `(ic, kh, kw)` 三重昇順、
// K タイル 16 昇順、bias は両方とも縮約の**前**、padding は「加算せず読み飛ばす」と
// 「0 を掛けて足す」で `a + 0.0 == a`（`a` が有限）。
//
// **唯一の例外 = 符号付きゼロ**: 部分和がちょうど `−0.0` の位置に padding 由来の `+0.0` を
// 足すと `+0.0` に転ぶ（直接カーネルは足さないので `−0.0` が残る）。出力が厳密に 0 になる
// 要素でしか起こらないため、本テストは **bias を全ケース非ゼロ**にしてその領域を避ける。
//
// MUST: 恒真化しないこと。①② は**別のパイプライン・別の WGSL**で、生成関数も別。
// MUST: 変種（v4 / スカラ）と格納（f32 / f16 / i8）を全て踏む。踏んだ変種は `expectVec4` で
// 明示的に固定する（判定の取り違えが「両方とも同じ変種で緑」に紛れないため）。

import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { alignF16Payload } from "../src/format/f16.ts";
import { alignI8Payload } from "../src/format/i8.ts";
import { openModel } from "../src/format/container.ts";
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
import { createSession } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * CPU 参照との突合の許容誤差。参照は JS の f64 で縮約するので f32 の逐次加算とは丸めが違い、
 * ビット一致は原理的に成立しない（K ≤ 72・値域 |x| ≤ 2 / |w| ≤ 0.25 の本ケース群では
 * 数 ulp 級）。**実装バグの誤差は O(1)** なので 6 桁以上離れている。
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/** MUST: bias は非ゼロ（符号付きゼロの領域を避ける — 冒頭の caveat）。 */
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
  /** 踏むはずの変種（判定の取り違えが「両方同じ変種」で紛れないように明示する）。 */
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
 * 重み格納の変種ぶんのペイロード（f32 はそのまま / f16・i8 は整列済みの生バイト）。
 *
 * `values` は**格納から復号し直した f32**（fake-quant 後の重み）。CPU 参照はこれを重みとして
 * 使う — 素の f32 を渡すと格納の丸め誤差が「カーネルの誤り」に化ける。
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
  // conv2d の per-channel scale の軸は 0（出力チャネル — ADR 0019）
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
 * 直接カーネルと implicit GEMM を**同じ入力バッファ**で走らせ、出力のビット列を組で返す。
 *
 * 入力（x / 重み / bias / scale）はどちらの dispatch も同じ GPUBuffer を束縛する — ホスト側で
 * 2 度書くと「書き込みの差」が「カーネルの差」に化ける。
 */
const runBoth = async (
  gpu: GpuContext,
  testCase: ParityCase,
  mTile: number = GEMM_TILE,
): Promise<
  {
    readonly direct: Uint32Array;
    readonly igemm: Uint32Array;
    readonly v4: boolean;
    readonly reference: Float32Array;
  }
> => {
  const dims = dimsOf(testCase);
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const x = fill(
      [dims.batch, dims.channelsIn, dims.heightIn, dims.widthIn],
      SIGNED,
    );
    const weight = fill(
      [dims.channelsOut, dims.channelsIn, dims.kernelH, dims.kernelW],
      WEIGHT,
    );
    const bias = fill([dims.channelsOut], BIAS);
    const payload = weightPayload(weight, testCase.storage);

    const upload = (data: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      // vec4 束縛（f32 の v4 経路）が末尾 quad を落とさないよう 16 バイトへ丸める
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

    // ① 直接カーネル（grid-stride）
    const directKey = conv2dKey(testCase.storage);
    const { pipeline: directPipeline, layout: directLayout } = await cache.get(
      directKey,
      conv2dWgsl(testCase.storage),
    );
    const direct = bind(directLayout, conv2dParams(dims));
    scheduler.dispatch(directPipeline, direct.group, [
      gridStrideWorkgroups(count, CONV2D_WORKGROUP_SIZE, limit),
      1,
      1,
    ], directKey);

    // ② implicit GEMM（1 workgroup = 1 出力タイル）
    const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const igemmKey = conv2dIgemmKey(testCase.storage, v4, mTile);
    const { pipeline: igemmPipeline, layout: igemmLayout } = await cache.get(
      igemmKey,
      conv2dIgemmWgsl(testCase.storage, v4, mTile),
    );
    const igemm = bind(igemmLayout, conv2dIgemmParams(dims));
    // MUST: dispatch の辺は生成・キーと**同じ解決点**（`gemmMTileGeometry`）から導く。
    // m タイルの変種が動かすのは行の辺だけで、n の辺は幾何の tileN（両変種で同じ）。
    const geometry = gemmMTileGeometry(mTile);
    scheduler.dispatch(igemmPipeline, igemm.group, [
      tiledWorkgroups(dims.heightOut * dims.widthOut, gemmTileN(geometry), limit, testCase.name),
      tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, testCase.name),
      tiledWorkgroups(dims.batch, 1, limit, testCase.name),
    ], igemmKey);

    await scheduler.flush();
    // CPU 参照は**量子化した重みを展開し直したもの**で作る（f16 / i8 の格納誤差を
    // 「カーネルの誤り」に化けさせない）。dequant は要素ごと（ADR 0019）。
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
    // census（VAE decoder）の実形状を縮めたもの: 3×3 pad 1 same。B=2 でバッチが N に畳まれる
    // ことを固定し、m タイル 2 枚（Cout=70）・n タイル 2 枚（N=96）・K 端数（72 = 16·4 + 8）を
    // 1 本で踏む。
    name: "census 縮小 3×3 pad1 [2,8,6,8] * W[70,8,3,3] (f32/v4)",
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
    // kFlat = 2·3·3 = 18 → `kFlat % 4 ≠ 0` でスカラ変種、かつ `K % 16 ≠ 0` で K 端数 0 埋めを
    // 踏む（census では K が全て 16 の倍数なので実 E2E に検出器が無い経路）。
    name: "端数 K スカラ [1,2,5,7] * W[5,2,3,3] (f32)",
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
    // Kh ≠ Kw・stride / padding が H≠W・Cin ≠ Cout（どちらも 2 以上）。stride_w = 3 なので
    // v4 の連続 4 列読みは成立せずスカラへ落ちる。
    name: "非対称 Kh≠Kw stride=[2,3] padding=[1,0] [1,3,9,11] * W[4,3,2,4]",
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
    // dilation も H≠W（v4 経路で `kh·dh` / `kw·dw` の取り違えを踏む）
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
    // **MUST ② の唯一の検出器**: Hout = Wout = 2 で N = 4。`kFlat = 20` と `stride_w = 1` は
    // v4 の条件を満たすので、判定を `N % 4` で書くと v4 が選ばれる。しかも Win = 6 なので
    // quad の連続読みガード（`ix0 + 3 < width_in`）を素通りし、出力行をまたいだ 4 列が
    // 同じ入力行から読まれる = 例外なしの誤値になる。実測形（Wout は全て 4 の倍数）では
    // 絶対に露見しない取り違えで、ここだけが赤くする。
    name: "Hout=Wout=2（N%4==0 だが Wout%4≠0）[1,4,2,6] * W[3,4,1,5]",
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
    // f16 格納 × v4（quad 展開 `dequant4` は平坦添字が 4 の倍数であることに依存する）
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
    // f16 格納 × スカラ。kFlat = 1·3·3 = 9 が**奇数**なので重み行の先頭が語の上位・下位に
    // 交互に来る（平坦添字の偶奇で対を選ぶことの検出器 — ADR 0018 の罠）。
    name: "f16 スカラ 行長 9（奇数）[1,1,5,6] * W[4,1,3,3]",
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
    // **MUST ④ の検出器**: i8 の scale は行（= 出力チャネル）。Cout = 70 で m タイルが 2 枚に
    // なるので、`arow` の代わりにタイル内相対の行や列を引く取り違えが値に出る
    // （1 タイルに収まる形では `wid.y = 0` で偶然一致してしまう — ADR 0022 の検出限界③と同型）。
    name: "i8 v4 m タイル 2 枚 [1,4,2,4] * W[70,4,1,1]",
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
    // i8 × スカラ変種（`unpack4xI8` の 4 剰余を平坦添字から取ること + m タイル 2 枚）。
    // kFlat = 2·3·3 = 18 は 4 の倍数でないのでスカラへ落ちる。
    name: "i8 スカラ m タイル 2 枚 [1,2,4,5] * W[66,2,3,3]",
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
    // **32 行 m タイル変種の本命形**（ADR 0024 隣接）: census の Cout=96（`96 % 64 == 32`）。
    // 64 行では 2 タイル（無駄 1.33×）・32 行では 3 タイル（無駄ゼロ）で、**どちらでも
    // 出力はビット同一**であることがこの変種の前提。i8 なので m タイル 3 枚 × 行 scale の
    // 検出器も兼ねる。
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
    // M % 64 == 0 の形（述語は 64 行を選ぶ）。32 行を**強制**すると 2 タイルに割れるので、
    // 「タイル形を変えても値は変わらない」という主張をタイル境界が動く形で踏む。
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
    // 33〜64 の帯（述語は 64 行）。32 行を強制すると **2 枚目のタイルが 1 行しか使わない**
    // ので、端タイルの行ガード（`orow < dims.m`）と bias-first の範囲外読みを踏む。
    name: "f16 Cout=33（33〜64 帯）[1,4,3,8] * W[33,4,3,3]",
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
    // M < 32（タイル 1 枚のまま行の 2/3 が空く形）。32 行でも 64 行でも 1 タイルだが、
    // 空き行の `acc` が store の行ガードで捨てられることを両側で固定する。
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
    // padding がカーネル張りを超える形（出力の両端が padding 域だけを見る = 範囲外を
    // クランプ添字で読む誤りが必ず値に出る）
    name: "padding=[2,0] 過剰 padding [1,4,2,8] * W[5,4,3,1]",
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
    // **v4 判定 3 条件目（`strideW == 1`）の唯一の検出器**。kFlat = 4·1·4 = 16（%4==0）・
    // Wout = (16−4)/4 + 1 = 4（%4==0）で他 2 条件は満たすが、strideW = 4 なので v4 は選べない
    // （4 列連続読みは stride 飛びの位置を連続として読む = 例外の出ない誤値になる）。
    // 述語から `strideW === 1` を外すとこの 1 本だけが赤くなる。
    name: "strideW=4 で v4 に落ちない [1,4,4,16] * W[8,4,1,4]",
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

/** ビット列の食い違い（先頭 4 件）を人が読める形にする。 */
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
 * 1 ケースぶんの検査。
 *
 * ① 直接カーネルとの **Uint32 ビット完全一致**（丸め列の変化はここでしか出ない）
 * ② CPU 参照（`referenceConv2d`）との数値一致（両カーネルが**揃って**誤る形の検出器 —
 *    ビット比較だけだと「同じ間違いをしている」に気づけない）
 * ③ 恒真化の門（出力が定数なら一致は何も検証していない）
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
  assertEquals(v4, testCase.expectVec4, `${where}: 踏んだ変種が想定と違う`);
  assertEquals(
    firstMismatches(direct, igemm),
    [],
    `${where}: 直接カーネルとビット列が違う`,
  );
  const values = new Float32Array(igemm.buffer, igemm.byteOffset, igemm.length);
  const report = allclose(values, reference, CPU_REFERENCE_TOLERANCE);
  assert(
    report.pass,
    `${where}: CPU 参照と食い違う（maxAbs ${report.maxAbsError} @${report.worstIndex} / 破り ${report.failCount}）`,
  );
  assert(new Set(direct).size > 1, `${where}: 出力が定数（ビット一致が恒真になっている）`);
};

Deno.test({
  name: "conv2d の implicit GEMM は直接カーネルと**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        assertParity(testCase, testCase.name, await runBoth(gpu, testCase));
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **32 行 m タイル変種**（ADR 0024 隣接）の恒久の門。
 *
 * 全ケースを 32 行タイルで**強制的に**流す（述語が 64 行を選ぶ形も含む）。タイル形は
 * 「どの workgroup がどの出力を担当するか」だけを変える設計なので、64 行版と同じ
 * 直接カーネル比較がそのまま通らなければならない。
 *
 * MUST: 述語（{@link conv2dIgemmMTile}）が選ぶかどうかと切り離して全形状を踏む。述語に
 * 従って選ばれた形だけを検査すると、述語の境界をずらす誤りが「変種が走らなくなるだけ」で
 * 緑のまま通る。
 *
 * カバー: M が 32 の倍数（96 / 64 / 20 は 1〜3 枚）/ 非倍数（70 / 66 / 33 / 5 / 4 / 3 / 12）/
 * 33〜64 の帯（33 / 70 の 2 枚目）/ M < 32（20 / 12 / 5 / 4 / 3）/ **m タイル 2 枚以上 × i8**
 * （96 = 3 枚・70 = 3 枚・66 = 3 枚 — scale 軸取り違えの唯一の検出器）。
 */
Deno.test({
  name: "conv2d の 32 行 m タイル変種も直接カーネルとビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        assertParity(
          testCase,
          `m32 / ${testCase.name}`,
          await runBoth(gpu, testCase, GEMM_MTILE_SMALL),
        );
      }
      // 述語が実際に 32 行を選ぶ形が CASES に含まれていること（テストが空振りしない門）
      assert(
        CASES.some((testCase) => conv2dIgemmMTile(testCase.channelsOut) === GEMM_MTILE_SMALL),
        "述語が 32 行を選ぶケースが 1 本も無い",
      );
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * executor の踏み分け（ADR 0024）。`groups == 1` は implicit GEMM、`groups > 1` は直接カーネル
 * （恒久の差分オラクルとして温存）。
 *
 * MUST: 値ではなく**実際に走ったパイプラインキー**で見る。どちらのカーネルも正しいので、
 * 出力の一致は踏み分けの証拠にならない。
 */
const conv2dKeysUsed = async (
  channels: number,
  groups: number,
): Promise<ReadonlySet<string>> => {
  const gpu = await acquireGpu({ gpuTiming: true });
  const graph = singleOpGraph(
    "conv2d",
    [[1, channels, 4, 4], [channels, channels / groups, 3, 3], [channels]],
    [[1, channels, 4, 4]],
    { attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups } },
  );
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    await session.run({
      x0: fill([1, channels, 4, 4], SIGNED),
      x1: fill([channels, channels / groups, 3, 3], WEIGHT),
      x2: fill([channels], BIAS),
    });
    const timing = session.diagnostics().lastRunTiming;
    assert(timing !== undefined, "timestamp-query が無効（キー別内訳が取れない）");
    return new Set(timing.entries.map((entry) => entry.key));
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

Deno.test({
  name: "executor は groups で 2 カーネルを踏み分ける（1 = igemm / >1 = 直接・実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    // Cin = Cout = 6・3×3 → kFlat = 54（groups=1）で 4 の倍数でないのでスカラ変種。
    // Cout=6 は `6 % 64 == 6` なので m タイルは 32 行（{@link conv2dIgemmMTile}）。
    assertEquals(
      await conv2dKeysUsed(6, 1),
      new Set([conv2dIgemmKey("f32", false, GEMM_MTILE_SMALL)]),
    );
    assertEquals(await conv2dKeysUsed(6, 3), new Set([conv2dKey("f32")]));
    // depthwise（groups = Cin = Cout）も直接カーネル側
    assertEquals(await conv2dKeysUsed(6, 6), new Set([conv2dKey("f32")]));
  },
});

/**
 * m タイルの選択が **executor に結線されている**ことの門（ADR 0024 隣接）。
 *
 * MUST: 値ではなく**実際に走ったパイプラインキー**で見る。どちらのタイル形も出力はビット
 * 同一なので、値の一致は結線の証拠にならない（述語を潰しても数値テストは全て緑のまま）。
 */
Deno.test({
  name: "executor は M%64 で m タイル 64/32 を踏み分ける（実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    // Cout = 96（census の本命 — `96 % 64 == 32`）→ 32 行
    assertEquals(
      await conv2dKeysUsed(96, 1),
      new Set([conv2dIgemmKey("f32", true, GEMM_MTILE_SMALL)]),
    );
    // Cout = 64（`64 % 64 == 0`）→ 64 行のまま（既存キーが動かないことの確認も兼ねる）
    assertEquals(await conv2dKeysUsed(64, 1), new Set([conv2dIgemmKey("f32", true, GEMM_TILE)]));
  },
});
