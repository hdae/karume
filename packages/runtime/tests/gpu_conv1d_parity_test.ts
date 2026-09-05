// conv1d implicit GEMM（ADR 0024 の 1D 版）の**ビット同一の門**。
//
// 同じ入力バッファに対し、
//
//   ① 直接畳み込みカーネル（`conv1d:v2:…:direct:wg256` — groups > 1 用に恒久で残る）
//   ② implicit GEMM カーネル（`conv1d:v3:…:igemm64x128…`）
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる丸め列の変化（例: bias を
// `acc` の初期値ではなく store 側で足す / 平坦 k を `(k, ic)` の順に割る）はここでしか
// 検出できない。
//
// 成立の根拠: 平坦 k 昇順 = 直接カーネルの `(ic, k)` 二重昇順、K タイル 16 昇順、bias は
// 両方とも縮約の**前**、padding は「加算せず読み飛ばす」と「0 を掛けて足す」で
// `a + 0.0 == a`（`a` が有限）。
//
// **唯一の例外 = 符号付きゼロ**: 部分和がちょうど `−0.0` の位置に padding 由来の `+0.0` を
// 足すと `+0.0` に転ぶ（直接カーネルは足さないので `−0.0` が残る）。本ファイルの主テストは
// **bias を全ケース非ゼロ**にしてその領域を避け、機序そのものは末尾の負ケースが固定する。
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
import { createSession } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * CPU 参照との突合の許容誤差。参照は JS の f64 で縮約するので f32 の逐次加算とは丸めが違い、
 * ビット一致は原理的に成立しない（K ≤ 288・値域 |x| ≤ 2 / |w| ≤ 0.25 の本ケース群では
 * 数 ulp 級）。**実装バグの誤差は O(1)** なので 6 桁以上離れている。
 */
const CPU_REFERENCE_TOLERANCE = { atol: 1e-5, rtol: 1e-5 } as const;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/**
 * 符号付きゼロの負ケース専用の重み（**厳密に正**）。既定の {@link WEIGHT} は 0 を含み、
 * `x · 0` が符号付きゼロを作ってしまうので、負ケースの機序が padding 由来かどうかを
 * 分離できない。
 */
const POSITIVE_WEIGHT = (i: number): number => 0.05 + ((i * 11) % 19) * 0.011;
/** MUST: bias は非ゼロ（符号付きゼロの領域を避ける — 冒頭の caveat）。 */
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
  /** 踏むはずの変種（判定の取り違えが「両方同じ変種」で紛れないように明示する）。 */
  readonly expectVec4: boolean;
  /** 既定は非ゼロ bias / 0 を含む重み。符号付きゼロの負ケースだけが差し替える。 */
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
  // conv1d の per-channel scale の軸は 0（出力チャネル — ADR 0019）
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
 * 直接カーネルと implicit GEMM を**同じ入力バッファ**で走らせ、出力のビット列を組で返す。
 *
 * 入力（x / 重み / bias / scale）はどちらの dispatch も同じ GPUBuffer を束縛する — ホスト側で
 * 2 度書くと「書き込みの差」が「カーネルの差」に化ける。
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

    const count = dims.batch * dims.channelsOut * dims.lengthOut;
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
    const bind = (
      layout: GPUBindGroupLayout,
      params: Uint32Array<ArrayBuffer>,
    ): { readonly group: GPUBindGroup; readonly out: GPUBuffer } => {
      const paramsBuffer = arena.allocHostWritten(params.byteLength, UNIFORM_IN);
      gpu.device.queue.writeBuffer(paramsBuffer, 0, params);
      const out = arena.allocStorage(Math.max(4, count * 4));
      arena.retain(out, 0, { pinned: true });
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

    // ① 直接カーネル（grid-stride）
    const directKey = conv1dKey(testCase.storage);
    const { pipeline: directPipeline, layout: directLayout } = await cache.get(
      directKey,
      conv1dWgsl(testCase.storage),
    );
    const direct = bind(directLayout, conv1dParams(dims));
    scheduler.dispatch(directPipeline, direct.group, [
      gridStrideWorkgroups(count, CONV1D_WORKGROUP_SIZE, limit),
      1,
      1,
    ], directKey);

    // ② implicit GEMM（1 workgroup = 1 出力タイル）
    const kFlat = dims.channelsIn * dims.kernel;
    const v4 = conv1dUsesVec4(kFlat, dims.lengthOut, dims.stride);
    const igemmKey = conv1dIgemmKey(testCase.storage, v4, mTile);
    const { pipeline: igemmPipeline, layout: igemmLayout } = await cache.get(
      igemmKey,
      conv1dIgemmWgsl(testCase.storage, v4, mTile),
    );
    const igemm = bind(igemmLayout, conv1dIgemmParams(dims));
    // MUST: dispatch の辺は生成・キーと**同じ解決点**（`gemmMTileGeometry`）から導く。
    // m タイルの変種が動かすのは行の辺だけで、n の辺は幾何の tileN（両変種で同じ）。
    const geometry = gemmMTileGeometry(mTile);
    scheduler.dispatch(igemmPipeline, igemm.group, [
      tiledWorkgroups(dims.lengthOut, gemmTileN(geometry), limit, testCase.name),
      tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, testCase.name),
      tiledWorkgroups(dims.batch, 1, limit, testCase.name),
    ], igemmKey);

    await scheduler.flush();
    // CPU 参照は**量子化した重みを展開し直したもの**で作る（f16 / i8 の格納誤差を
    // 「カーネルの誤り」に化けさせない）。dequant は要素ごと（ADR 0019）。
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
    // 実測形（dacvae decoder）を縮めたもの: 3 tap pad 1 same。B=2 でバッチが N に畳まれる
    // ことを固定し、m タイル 2 枚（Cout=70）・K 端数（24 = 16 + 8）を 1 本で踏む。
    name: "実測縮小 K=3 pad1 [2,8,20] * W[70,8,3] (f32/v4)",
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
    // kFlat = 1·7 = 7 → `kFlat % 4 ≠ 0` でスカラ変種、かつ `K % 16 ≠ 0` で K 端数 0 埋めを
    // 踏む。**`Cin % 4` で判定すると v4 が選ばれてしまう形**（Cin=1）の検出器も兼ねる。
    name: "端数 K スカラ Cin=1 K=7 [1,1,19] * W[5,1,7] pad3 (f32)",
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
    // stride 2（出力位置と入力位置の対応が 1:1 でなくなる）
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
    // **v4 述語の stride 条件の検出器**: kFlat = 16 も Lout = 8 も 4 の倍数なので、
    // stride を見落とすと v4 が選ばれ、連続 4 列読みが 8 飛びの位置を連続として読む。
    // Cout=16（m タイル 1 枚・行の 3/4 が空く形）も兼ねる。
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
    // dilation 3（tap の間隔 — 出力位置側に掛ける誤りは stride と見分けがつかない）
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
    // dilation 9（実測形の最大 — tap が系列の外へ大きくはみ出す）
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
    // Cout=1（m タイル 1 枚で 63/64 行が空く — 空き行の acc が store の行ガードで
    // 捨てられること、bias-first の範囲外読みが値に出ないことを固定する）
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
    // **`Lout % 4` の検出器**: kFlat = 12 は 4 の倍数・stride = 1 なので、Lout を見落とすと
    // v4 が選ばれ、store の quad 書きが出力の端を越える。
    name: "Lout=15（kFlat%4==0 だが Lout%4≠0）[1,4,15] * W[8,4,3] pad1 (f32)",
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
    // padding がカーネル張りを超える形（出力の両端が padding 域だけを見る = 範囲外を
    // クランプ添字で読む誤りが必ず値に出る）
    name: "過剰 padding=3 [1,4,4] * W[5,4,3] (f32/v4)",
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
    // f16 格納 × v4（quad 展開 `dequant4` は平坦添字が 4 の倍数であることに依存する）
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
    // f16 格納 × スカラ。kFlat = 1·9 = 9 が**奇数**なので重み行の先頭が語の上位・下位に
    // 交互に来る（平坦添字の偶奇で対を選ぶことの検出器 — ADR 0018 の罠）。
    name: "f16 スカラ 行長 9（奇数）[1,1,12] * W[4,1,9] pad4",
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
    // **i8 の scale 軸の検出器**: scale は行（= 出力チャネル）。Cout = 96 で m タイルが
    // 64 行なら 2 枚・32 行なら 3 枚になるので、`arow` の代わりにタイル内相対の行や列を
    // 引く取り違えが値に出る（1 タイルに収まる形では `wid.y = 0` で偶然一致してしまう）。
    // `96 % 64 == 32` なので m タイル述語の本命形でもある。
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
    // i8 × スカラ変種（`unpack4xI8` の 4 剰余を平坦添字から取ること + m タイル 2 枚）。
    // kFlat = 2·3 = 6 は 4 の倍数でないのでスカラへ落ちる。
    name: "i8 スカラ m タイル 2 枚 [1,2,10] * W[66,2,3] pad1",
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
 * ② CPU 参照（`referenceConv1d`）との数値一致（両カーネルが**揃って**誤る形の検出器 —
 *    ビット比較だけだと「同じ間違いをしている」に気づけない）
 * ③ 恒真化の門（出力が定数なら一致は何も検証していない）
 */
const assertParity = (testCase: ParityCase, where: string, result: ParityResult): void => {
  const { direct, igemm, v4, reference } = result;
  assertEquals(v4, testCase.expectVec4, `${where}: 踏んだ変種が想定と違う`);
  assertEquals(firstMismatches(direct, igemm), [], `${where}: 直接カーネルとビット列が違う`);
  const values = new Float32Array(igemm.buffer, igemm.byteOffset, igemm.length);
  const report = allclose(values, reference, CPU_REFERENCE_TOLERANCE);
  assert(
    report.pass,
    `${where}: CPU 参照と食い違う（maxAbs ${report.maxAbsError} @${report.worstIndex} / 破り ${report.failCount}）`,
  );
  assert(new Set(direct).size > 1, `${where}: 出力が定数（ビット一致が恒真になっている）`);
};

Deno.test({
  name: "conv1d の implicit GEMM は直接カーネルと**ビット単位で一致**する（実 GPU）",
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
 * **32 行 m タイル変種**の恒久の門。全ケースを 32 行タイルで**強制的に**流す（述語が
 * 64 行を選ぶ形も含む）。タイル形は「どの workgroup がどの出力を担当するか」だけを変える
 * 設計なので、64 行版と同じ直接カーネル比較がそのまま通らなければならない。
 *
 * MUST: 述語（conv2d と共有の `conv2dIgemmMTile`）が選ぶかどうかと切り離して全形状を踏む。
 * 述語に従って選ばれた形だけを検査すると、述語の境界をずらす誤りが「変種が走らなくなる
 * だけ」で緑のまま通る。
 */
Deno.test({
  name: "conv1d の 32 行 m タイル変種も直接カーネルとビット単位で一致する（実 GPU）",
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
 * **符号付きゼロ = ビット同一の唯一の例外**（負ケース）。機序をテストとして記録し、将来
 * bias に `−0.0` が入るモデルが来たときの検出器を兼ねる。
 *
 * 形: `Lin = 1 / K = 3 / padding = 3` なので出力の両端（ox = 0 と ox = 4）は**受容野が
 * まるごと padding 域**になる。bias を `−0.0`・重みを全て正にすると、
 *
 * - 直接カーネル … padding を**加算しない**ので `−0.0` がそのまま残る
 * - implicit GEMM … `0.0 × w = +0.0` を足すので `−0.0 + 0.0 = +0.0` に転ぶ
 *
 * MUST: 差が出るのは**この 2 要素だけ**で、残りはビット一致（`a + 0.0 == a` が有限の `a` で
 * 成り立つことの裏返し）。値としては `−0.0 == +0.0` なので、allclose でも `===` でも
 * **絶対に検出できない** — ビット比較だけがこの差を見る。
 */
Deno.test({
  name: "conv1d: bias が −0.0 のとき padding だけの出力で符号付きゼロが割れる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const testCase: ParityCase = {
        name: "符号付きゼロ [1,1,1] * W[4,1,3] pad3 bias=-0.0",
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
      // 受容野が全て padding 域になる出力位置（origin = ox − 3、有効な入力添字は 0 だけ）
      const paddingOnly = new Set([0, 4]);
      const splits: number[] = [];
      for (let index = 0; index < direct.length; index += 1) {
        const ox = index % lengthOut;
        if (!paddingOnly.has(ox)) {
          assertEquals(direct[index], igemm[index], `[${index}] は一致しなければならない`);
          continue;
        }
        assertEquals(direct[index], 0x8000_0000, `[${index}] 直接カーネルは −0.0 を残す`);
        assertEquals(igemm[index], 0x0000_0000, `[${index}] implicit GEMM は +0.0 に転ぶ`);
        splits.push(index);
      }
      // 空振りの門（全ケースが「一致」側に落ちていたら負ケースとして成立していない）
      assertEquals(splits.length, testCase.channelsOut * paddingOnly.size);
      // MUST: 値としては等しい（この差は allclose でも === でも検出できない）
      const directValues = new Float32Array(direct.buffer, direct.byteOffset, direct.length);
      const igemmValues = new Float32Array(igemm.buffer, igemm.byteOffset, igemm.length);
      for (const index of splits) {
        assert(
          directValues[index] === igemmValues[index],
          `[${index}] 値としては −0.0 === +0.0 のはず`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * executor の踏み分け。`groups == 1` は implicit GEMM、`groups > 1` は直接カーネル
 * （恒久の差分オラクルとして温存）。
 *
 * MUST: 値ではなく**実際に走ったパイプラインキー**で見る。どちらのカーネルも正しいので、
 * 出力の一致は踏み分けの証拠にならない。
 */
const conv1dKeysUsed = async (
  channels: number,
  groups: number,
): Promise<ReadonlySet<string>> => {
  const gpu = await acquireGpu({ gpuTiming: true });
  const graph = singleOpGraph(
    "conv1d",
    [[1, channels, 8], [channels, channels / groups, 3], [channels]],
    [[1, channels, 8]],
    { attrs: { stride: 1, padding: 1, dilation: 1, groups } },
  );
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    await session.run({
      x0: fill([1, channels, 8], SIGNED),
      x1: fill([channels, channels / groups, 3], WEIGHT),
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
    // Cin = Cout = 6・K=3 → kFlat = 18（groups=1）で 4 の倍数でないのでスカラ変種。
    // Cout=6 は `6 % 64 == 6` なので m タイルは 32 行。
    assertEquals(
      await conv1dKeysUsed(6, 1),
      new Set([conv1dIgemmKey("f32", false, GEMM_MTILE_SMALL)]),
    );
    assertEquals(await conv1dKeysUsed(6, 3), new Set([conv1dKey("f32")]));
    // depthwise（groups = Cin = Cout）も直接カーネル側
    assertEquals(await conv1dKeysUsed(6, 6), new Set([conv1dKey("f32")]));
  },
});

/**
 * m タイルの選択が **executor に結線されている**ことの門。
 *
 * MUST: 値ではなく**実際に走ったパイプラインキー**で見る。どちらのタイル形も出力はビット
 * 同一なので、値の一致は結線の証拠にならない（述語を潰しても数値テストは全て緑のまま）。
 */
Deno.test({
  name: "conv1d の executor は M%64 で m タイル 64/32 を踏み分ける（実 GPU）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    // Cout = 96（`96 % 64 == 32`）→ 32 行。kFlat = 288・Lout = 8 なので v4。
    assertEquals(
      await conv1dKeysUsed(96, 1),
      new Set([conv1dIgemmKey("f32", true, GEMM_MTILE_SMALL)]),
    );
    // Cout = 64（`64 % 64 == 0`）→ 64 行のまま
    assertEquals(await conv1dKeysUsed(64, 1), new Set([conv1dIgemmKey("f32", true, GEMM_TILE)]));
  },
});
