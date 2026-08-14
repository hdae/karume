// grid-stride の実効性を、dispatch 数を意図的に絞って直接確かめる（実 GPU）。
//
// 1 次元の workgroup 上限（既定 65535）を本物の要素数で踏むには数十 MB のバッファが要る。
// カーネル側の縮退耐性そのものは「必要数より遥かに少ない workgroup で全域が埋まるか」で
// 等価に検証できるので、小さい入力に絞った dispatch を当てる。
//
// MUST: grid-stride を謳う**全カーネル族**をここに載せる。実測形のテスト（gpu_ops_test.ts）は
// 「大きめの入力」を通すだけで、必要数の workgroup が実際に割り当たるため縮退経路を 1 度も
// 踏まない — `stride` を `nwg.x * WG` ではなく定数にする誤りは、そちらでは緑のまま通る。
// 要素方向の族（elementwise / strided 読み書き / gather / embedding / masked_fill / conv 族 /
// pad / flip / upsample_bilinear2d / deform_conv2d / **軸 reduce**）は `gid.x` の 1 次元
// grid-stride、
// 行方向の族（行 reduce / softmax /
// layer_norm / rms_norm / **gru_scan**）は `workgroup_id.x` を `num_workgroups.x` で送る
// 行ループを踏ませる（gru_scan の「行」はバッチ要素で、時間ループはカーネル内）。
// **reduce 族は 2 変種とも載せる**（最終次元 = 行方向 / それ以外 = 要素方向で、走らせ方が違う）。
// cumsum は**行を単位とした
// `gid.x` の grid-stride**（1 invocation = 1 行の逐次走査）で、必要数は行数ではなく
// `ceil(行数 / workgroup サイズ)`。
//
// NOTE: conv2d は 2 カーネルに分かれた（ADR 0024）。ここが載せるのは **直接カーネル**
// （groups > 1 用に恒久で残る grid-stride 側）だけで、implicit GEMM 変種は「1 workgroup =
// 1 出力タイル」なのでカテゴリ違い — 安全網は `tiledWorkgroups` の fail loudly と
// tests/gpu_full_write_test.ts の毒値注入の 2 本（GEMM 3 op と同じ判断）。
//
// NOTE: 実運用で必ず縮退経路に入るのは **pad**（flow / voice の相対位置注意の value 側
// `F.pad(p_attn, [w, w])` は T ≳ 2900 で必要 workgroup 数が 65535 を超える）。他の族は
// 「今の実測形では超えない」だけなので、上限を踏むかどうかで載せる・載せないを決めない。

import { assertEquals } from "@std/assert";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  elementwiseWgsl,
} from "../src/codegen/elementwise.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../src/codegen/reduce.ts";
import {
  expandSrcStrides,
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  STRIDED_WRITE_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../src/codegen/strided.ts";
import {
  CONV1D_WORKGROUP_SIZE,
  conv1dKey,
  conv1dParams,
  conv1dWgsl,
} from "../src/kernels/conv1d.ts";
import {
  CONV2D_WORKGROUP_SIZE,
  conv2dKey,
  conv2dParams,
  conv2dWgsl,
} from "../src/kernels/conv2d.ts";
import {
  CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../src/kernels/conv-transpose1d.ts";
import {
  CUMSUM_KEY,
  CUMSUM_WGSL,
  CUMSUM_WORKGROUP_SIZE,
  cumsumParams,
} from "../src/kernels/cumsum.ts";
import {
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../src/kernels/embedding.ts";
import { FLIP_KEY, FLIP_WGSL, FLIP_WORKGROUP_SIZE, flipParams } from "../src/kernels/flip.ts";
import {
  GATHER_KEY,
  GATHER_WGSL,
  GATHER_WORKGROUP_SIZE,
  gatherParams,
} from "../src/kernels/gather.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../src/kernels/layer-norm.ts";
import { RMS_NORM_KEY, RMS_NORM_WGSL, rmsNormParams } from "../src/kernels/rms-norm.ts";
import {
  MASKED_FILL_KEY,
  MASKED_FILL_WGSL,
  MASKED_FILL_WORKGROUP_SIZE,
  maskedFillParams,
} from "../src/kernels/masked-fill.ts";
import { PAD_KEY, PAD_WGSL, PAD_WORKGROUP_SIZE, padParams } from "../src/kernels/pad.ts";
import {
  UPSAMPLE_BILINEAR2D_KEY,
  UPSAMPLE_BILINEAR2D_WGSL,
  UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
  upsampleBilinear2dParams,
} from "../src/kernels/upsample-bilinear2d.ts";
import { gruScanKey, gruScanParams, gruScanWgsl } from "../src/kernels/gru-scan.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../src/kernels/deform-conv2d.ts";
import { SOFTMAX_KEY, SOFTMAX_WGSL, softmaxParams } from "../src/kernels/softmax.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { quantizeRowsReference } from "../src/reference/i8a8.ts";
import {
  ATTENTION_STATS_KEY,
  ATTENTION_STATS_STRIDE,
  ATTENTION_STATS_WGSL,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsWgsl,
} from "../src/kernels/attention.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { alignF16Payload, roundToF16 } from "../src/format/f16.ts";
import { alignI8Payload } from "../src/format/i8.ts";
import { WEIGHT_STORAGES, type WeightStorage } from "../src/kernels/weight-storage.ts";
import { f32ToF16Bits, quantizeF16 } from "./helpers/f16.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

const readback = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array<ArrayBuffer>> => {
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
  return new Float32Array(copy, 0, count);
};

/**
 * 1 カーネル族ぶんの縮退ケース。binding は `0 = params` / `1..n = 入力（並び順）` /
 * 末尾 = 出力で、全カーネル族が共有する配置。
 */
type DegenerateCase = {
  readonly name: string;
  readonly key: string;
  readonly wgsl: string;
  readonly params: Uint32Array<ArrayBuffer>;
  /** params を uniform で受けるカーネル（行ループに workgroupBarrier がある族）。 */
  readonly uniformParams: boolean;
  /**
   * 入力バッファに書く中身。`Uint8Array` は**生バイト**（f16 / i8 格納の重みは 2 / 4 要素を
   * 1 語に詰めていてホストの数値配列型では表せない — ADR 0018 / 0019）。
   */
  readonly inputs: readonly (FilledTensor | Uint8Array<ArrayBuffer>)[];
  /**
   * 比較しない**もう 1 本の書き込み先**のバイト数（束縛は入力の次・比較対象の出力の 1 つ前）。
   * `quantize_rows` のように 1 dispatch が 2 本書く族のためのもので、こちらの中身は f32 では
   * 表せない（i8 の 4 詰め）ので突合の対象にしない — 縮退耐性は行方向のループが担うので、
   * 行ごとに 1 語書く `xs` 側を比較すれば同じ経路を踏める。
   */
  readonly sideOutputBytes?: number;
  /**
   * i8 変種の per-channel scale（ADR 0019）。束縛は**出力の次の番号**なので、出力を積んだ
   * 後に足す（カーネル側の `*_SCALE_BINDING` と同じ位置になる）。
   */
  readonly weightScale?: Float32Array<ArrayBuffer>;
  /** CPU 参照が出した期待出力（f32 の族のみ載せる）。 */
  readonly expected: RefTensor;
  /** 素直に割ったときの必要 workgroup 数（縮退していることの証拠として固定する）。 */
  readonly natural: number;
  /** 意図的に絞った dispatch 数。 */
  readonly groups: number;
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
      const side = arena.allocStorage(Math.max(4, testCase.sideOutputBytes));
      arena.retain(side, 0, { pinned: true });
      entries.push({ binding: entries.length, resource: { buffer: side } });
    }
    const count = testCase.expected.data.length;
    const dst = arena.allocStorage(Math.max(4, count * 4));
    arena.retain(dst, 0, { pinned: true });
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

    const actual = await readback(gpu.device, dst, count);
    const report = compareTensors(
      refTensor(testCase.expected.shape, actual),
      testCase.expected,
    );
    assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
  } finally {
    await arena.destroy();
  }
};

/** 要素方向の族（`gid.x` の 1 次元 grid-stride）。 */
const elementwiseCase = (): DegenerateCase => {
  const count = 100_000;
  const spec = { op: "neg", rank: 1, dtype: "f32" } as const;
  const input = fill([count], SIGNED);
  return {
    name: "elementwise neg",
    key: elementwiseKey(spec),
    wgsl: elementwiseWgsl(spec),
    params: elementwiseParams([count], [[count]]),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("neg", [input], {}, [count]),
    natural: Math.ceil(count / ELEMENTWISE_WORKGROUP_SIZE),
    groups: 3,
  };
};

const stridedCase = (): DegenerateCase => {
  const srcShape = [400, 250];
  const outShape = [250, 400];
  const dims = [1, 0];
  const input = fill(srcShape, SIGNED);
  return {
    name: "strided copy (permute [1,0])",
    key: stridedKey({ dtype: "f32" }),
    wgsl: stridedWgsl({ dtype: "f32" }),
    params: stridedParams(outShape, permuteSrcStrides(srcShape, dims), 0),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("permute", [input], { dims }, outShape),
    natural: Math.ceil(100_000 / STRIDED_WORKGROUP_SIZE),
    groups: 3,
  };
};

const gatherCase = (): DegenerateCase => {
  const rows = 20_000;
  const srcCols = 9;
  const cols = 6;
  const src = fill([rows, srcCols], SIGNED);
  // 添字は 0..srcCols-1 を一巡する（恒等でも単調でもない列 — 1 ずれが必ず値に出る）。
  const index = fill([rows, cols], (i) => (i * 5) % srcCols, "i32");
  return {
    name: "gather",
    key: GATHER_KEY,
    wgsl: GATHER_WGSL,
    params: gatherParams(rows * cols, cols, srcCols),
    uniformParams: true,
    inputs: [src, index],
    expected: applyReferenceOp("gather", [src, index], {}, [rows, cols]),
    natural: Math.ceil((rows * cols) / GATHER_WORKGROUP_SIZE),
    groups: 3,
  };
};

/**
 * 重み格納の変種（ADR 0018 / 0019）を縮退ハーネスに載せるための小道具。**f16 / i8 変種は
 * WGSL 本体が違う = 別の検出対象**なので、f32 版と同じ形の縮退ケースを 1 本ずつ持つ
 * （このファイル冒頭の MUST）。CPU 参照の重みは**丸め後の値**（fake-quant — ADR 0006）にする。
 */
type VariantWeight = {
  /** 入力バッファに載せる中身（f32 はそのまま / f16・i8 は整列済みの生バイト）。 */
  readonly payload: FilledTensor | Uint8Array<ArrayBuffer>;
  /** CPU 参照が使う重み（低精度格納なら丸め後）。 */
  readonly reference: RefTensor;
  /** i8 変種の per-channel scale（他の変種は undefined）。 */
  readonly scale?: Float32Array<ArrayBuffer>;
};

/**
 * `channelAxis` は per-channel scale が掛かる軸（conv_transpose1d だけ 1 — ADR 0019）。
 * f32 / f16 では使われない。
 */
const variantWeight = (
  weight: FilledTensor,
  storage: WeightStorage,
  channelAxis: number,
): VariantWeight => {
  if (storage === "f32") return { payload: weight, reference: weight };
  if (storage === "f16") {
    const quantized = quantizeF16(weight.data);
    return {
      payload: alignF16Payload(quantized.bytes),
      reference: refTensor(weight.shape, quantized.values),
    };
  }
  const quantized = quantizeI8(weight.data, weight.shape, channelAxis);
  return {
    payload: alignI8Payload(quantized.bytes),
    reference: refTensor(weight.shape, quantized.values),
    scale: quantized.scale,
  };
};

const embeddingCase = (storage: WeightStorage): DegenerateCase => {
  const vocab = 64;
  // 行長を奇数にする（f16 変種で行の先頭が語の上位・下位に交互に来る — 対の取り違えの検出器）
  const hidden = 5;
  const rows = 20_000;
  const weight = variantWeight(fill([vocab, hidden], SIGNED), storage, 0);
  const index = fill([rows], (i) => (i * 7) % vocab, "i32");
  return {
    name: `embedding (w=${storage})`,
    key: embeddingKey(storage),
    wgsl: embeddingWgsl(storage),
    params: embeddingParams(rows * hidden, hidden, vocab),
    uniformParams: true,
    inputs: [weight.payload, index],
    weightScale: weight.scale,
    expected: applyReferenceOp("embedding", [weight.reference, index], {}, [rows, hidden]),
    natural: Math.ceil((rows * hidden) / EMBEDDING_WORKGROUP_SIZE),
    groups: 3,
  };
};

const maskedFillCase = (): DegenerateCase => {
  const rows = 25_000;
  const cols = 4;
  const outShape = [rows, cols];
  const value = -3.4028234663852886e38;
  const x = fill(outShape, SIGNED);
  // mask は最終次元を stride 0 で複製する形（実測の右詰め broadcast と同型）。
  const mask = fill([rows, 1], (i) => i % 3 === 0 ? 1 : 0, "bool");
  return {
    name: "masked_fill",
    key: MASKED_FILL_KEY,
    wgsl: MASKED_FILL_WGSL,
    params: maskedFillParams(outShape, expandSrcStrides(mask.shape, outShape), value),
    uniformParams: false,
    inputs: [x, mask],
    expected: applyReferenceOp("masked_fill", [x, mask], { value }, outShape),
    natural: Math.ceil((rows * cols) / MASKED_FILL_WORKGROUP_SIZE),
    groups: 3,
  };
};

const conv1dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  const channelsIn = 2;
  const channelsOut = 2;
  const lengthIn = 25_000;
  // K を奇数にする（f16 変種で重み行の先頭が語の上位・下位に交互に来る）
  const kernel = 3;
  const stride = 1;
  const padding = 1;
  const dilation = 1;
  const groups = 1;
  const lengthOut = lengthIn;
  const x = fill([batch, channelsIn, lengthIn], SIGNED);
  const weight = variantWeight(fill([channelsOut, channelsIn, kernel], POSITIVE), storage, 0);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv1d (w=${storage})`,
    key: conv1dKey(storage),
    wgsl: conv1dWgsl(storage),
    params: conv1dParams({
      batch,
      channelsIn,
      channelsOut,
      lengthIn,
      lengthOut,
      kernel,
      stride,
      padding,
      dilation,
      groups,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp(
      "conv1d",
      [x, weight.reference, bias],
      { stride, padding, dilation, groups },
      [batch, channelsOut, lengthOut],
    ),
    natural: Math.ceil((batch * channelsOut * lengthOut) / CONV1D_WORKGROUP_SIZE),
    groups: 2,
  };
};

const convTranspose1dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  // 非対称チャネル（重み [Cin,Cout,K] を [Cout,Cin,K] と読む誤りをここでも赤にする）
  const channelsIn = 3;
  const channelsOut = 2;
  const lengthIn = 12_000;
  // K は契約（2·padding == K − stride）に縛られてここでは偶数（奇数長の対選択は
  // stride 1 が取れる tests/gpu_f16_weights_test.ts が見る）
  const kernel = 4;
  const stride = 2;
  const padding = 1;
  const lengthOut = lengthIn * stride;
  const x = fill([batch, channelsIn, lengthIn], SIGNED);
  // MUST: チャネル軸は 1（重み [Cin, Cout, K] の転置レイアウト — ADR 0019）
  const weight = variantWeight(fill([channelsIn, channelsOut, kernel], POSITIVE), storage, 1);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv_transpose1d (w=${storage})`,
    key: convTranspose1dKey(storage),
    wgsl: convTranspose1dWgsl(storage),
    params: convTranspose1dParams({
      batch,
      channelsIn,
      channelsOut,
      lengthIn,
      lengthOut,
      kernel,
      stride,
      padding,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp("conv_transpose1d", [x, weight.reference, bias], {
      stride,
      padding,
    }, [
      batch,
      channelsOut,
      lengthOut,
    ]),
    natural: Math.ceil((batch * channelsOut * lengthOut) / CONV_TRANSPOSE1D_WORKGROUP_SIZE),
    groups: 2,
  };
};

const conv2dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  // Cin ≠ Cout・Kh ≠ Kw（縮退経路でも重みの軸取り違えが赤くなる形にする）
  const channelsIn = 2;
  const channelsOut = 3;
  const heightIn = 4;
  const widthIn = 8_000;
  // NOTE: Kh·Kw が偶数なので重み行の先頭は常に語境界。f16 の対選択（奇数長）を踏むのは
  // tests/gpu_f16_weights_test.ts の担当で、ここは縮退耐性だけを見る。
  const kernelH = 3;
  const kernelW = 2;
  const paddingH = 1;
  const paddingW = 0;
  const heightOut = heightIn;
  const widthOut = widthIn - (kernelW - 1);
  const x = fill([batch, channelsIn, heightIn, widthIn], SIGNED);
  const weight = variantWeight(
    fill([channelsOut, channelsIn, kernelH, kernelW], POSITIVE),
    storage,
    0,
  );
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv2d (w=${storage})`,
    key: conv2dKey(storage),
    wgsl: conv2dWgsl(storage),
    params: conv2dParams({
      batch,
      channelsIn,
      channelsOut,
      heightIn,
      widthIn,
      heightOut,
      widthOut,
      kernelH,
      kernelW,
      strideH: 1,
      strideW: 1,
      paddingH,
      paddingW,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp(
      "conv2d",
      [x, weight.reference, bias],
      {
        stride: [1, 1],
        padding: [paddingH, paddingW],
        dilation: [1, 1],
        groups: 1,
      },
      [batch, channelsOut, heightOut, widthOut],
    ),
    natural: Math.ceil((batch * channelsOut * heightOut * widthOut) / CONV2D_WORKGROUP_SIZE),
    groups: 2,
  };
};

const padCase = (): DegenerateCase => {
  const rows = 4;
  const lengthIn = 25_000;
  // 左右を**別の幅**にする（対称幅だと left / right の取り違えが出力に出ない）。
  const left = 3;
  const right = 5;
  const x = fill([rows, lengthIn], SIGNED);
  const outShape = [rows, left + lengthIn + right];
  return {
    name: "pad",
    key: PAD_KEY,
    wgsl: PAD_WGSL,
    params: padParams(rows, lengthIn, left, right),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp("pad", [x], { left, right }, outShape),
    natural: Math.ceil((rows * outShape[1]) / PAD_WORKGROUP_SIZE),
    groups: 2,
  };
};

const flipCase = (): DegenerateCase => {
  // 反転軸の長さは 3 以上（2 の反転は off-by-one が対称に消えて検出器にならない）。
  const outer = 2;
  const length = 3;
  const inner = 20_000;
  const shape = [outer, length, inner];
  const x = fill(shape, SIGNED);
  return {
    name: "flip",
    key: FLIP_KEY,
    wgsl: FLIP_WGSL,
    params: flipParams(outer, length, inner),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp("flip", [x], { dim: 1 }, shape),
    natural: Math.ceil((outer * length * inner) / FLIP_WORKGROUP_SIZE),
    groups: 2,
  };
};

const deformConv2dCase = (): DegenerateCase => {
  const batch = 1;
  // Cin ≠ Cout・Kh ≠ Kw（縮退経路でも重みの軸取り違えと offset の y/x 順が赤くなる形）
  const channelsIn = 2;
  const channelsOut = 3;
  const heightIn = 4;
  const widthIn = 6_000;
  const kernelH = 3;
  const kernelW = 2;
  const paddingH = 1;
  const paddingW = 0;
  const heightOut = heightIn;
  const widthOut = widthIn - (kernelW - 1);
  const taps = kernelH * kernelW;
  const x = fill([batch, channelsIn, heightIn, widthIn], SIGNED);
  const weight = fill([channelsOut, channelsIn, kernelH, kernelW], POSITIVE);
  // offset は ±1.5 の非整数（境界の内外を跨ぐ）・mask は BiRefNet と同じ [0,2]
  const offset = fill([batch, 2 * taps, heightOut, widthOut], (i) => ((i % 7) - 3) * 0.5);
  const mask = fill([batch, taps, heightOut, widthOut], (i) => (i % 5) * 0.5);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: "deform_conv2d",
    key: DEFORM_CONV2D_KEY,
    wgsl: DEFORM_CONV2D_WGSL,
    params: deformConv2dParams({
      batch,
      channelsIn,
      channelsOut,
      heightIn,
      widthIn,
      heightOut,
      widthOut,
      kernelH,
      kernelW,
      paddingH,
      paddingW,
    }),
    uniformParams: true,
    inputs: [x, weight, offset, mask, bias],
    expected: applyReferenceOp(
      "deform_conv2d",
      [x, weight, offset, mask, bias],
      { padding: [paddingH, paddingW] },
      [batch, channelsOut, heightOut, widthOut],
    ),
    natural: Math.ceil(
      (batch * channelsOut * heightOut * widthOut) / DEFORM_CONV2D_WORKGROUP_SIZE,
    ),
    groups: 2,
  };
};

const upsampleBilinear2dCase = (): DegenerateCase => {
  // 実モデルの形（BiRefNet の decoder 本流は 512² → 2048²）を要素数だけ縮めた比。倍率は
  // 非整数（H は 39/399・W は 41/419）で、出力位置ごとに重みが違う形を縮退経路で踏む。
  const [heightIn, widthIn] = [40, 42];
  const [heightOut, widthOut] = [400, 420];
  const shape = [1, 1, heightIn, widthIn];
  const outShape = [1, 1, heightOut, widthOut];
  const x = fill(shape, SIGNED);
  return {
    name: "upsample_bilinear2d",
    key: UPSAMPLE_BILINEAR2D_KEY,
    wgsl: UPSAMPLE_BILINEAR2D_WGSL,
    params: upsampleBilinear2dParams(
      heightOut * widthOut,
      heightIn,
      widthIn,
      heightOut,
      widthOut,
    ),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp(
      "upsample_bilinear2d",
      [x],
      { output_size: [heightOut, widthOut] },
      outShape,
    ),
    natural: Math.ceil((heightOut * widthOut) / UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE),
    groups: 2,
  };
};

/**
 * cat の実行カーネル（strided **書き**族）。1 dispatch = 出力の部分領域という cat の実形は
 * 残りが未書き込みで比較できないので、**出力全域を覆う置換**（読み族の permute [1,0] の双対）
 * としてパラメータを与える。書き族の grid-stride 上限 `n` は params[0]（= 入力要素数）だけで
 * 決まり stride 値に依らないので、縮退耐性はこの形で等価に踏める。
 */
const stridedWriteCase = (): DegenerateCase => {
  const srcShape = [400, 250];
  const outShape = [250, 400];
  const input = fill(srcShape, SIGNED);
  // 入力座標 (a, b) → 出力添字 b·400 + a。軸 0 の送りが 1、軸 1 の送りが 400。
  const outStrides = [1, srcShape[0]];
  return {
    name: "strided write (cat の書き族)",
    key: stridedWriteKey({ dtype: "f32" }),
    wgsl: stridedWriteWgsl({ dtype: "f32" }),
    params: stridedWriteParams(srcShape, outStrides, 0),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("permute", [input], { dims: [1, 0] }, outShape),
    natural: Math.ceil(100_000 / STRIDED_WRITE_WORKGROUP_SIZE),
    groups: 2,
  };
};

/** 行を単位とした `gid.x` の grid-stride（1 invocation = 1 行の逐次走査）。 */
const cumsumCase = (): DegenerateCase => {
  const rows = 20_000;
  const dim = 6;
  // SIGNED は 0.75 の整数倍なので、6 項までの部分和は f32 で厳密（丸めの寄与を混ぜない）。
  const input = fill([rows, dim], SIGNED);
  return {
    name: "cumsum",
    key: CUMSUM_KEY,
    wgsl: CUMSUM_WGSL,
    params: cumsumParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("cumsum", [input], { dim: 1 }, [rows, dim]),
    // 行内は 1 スレッドの逐次走査なので、必要数は行数ではなく ceil(行数 / workgroup サイズ)。
    natural: Math.ceil(rows / CUMSUM_WORKGROUP_SIZE),
    groups: 2,
  };
};

/** 行方向の族（`workgroup_id.x` を `num_workgroups.x` で送る行ループ）。 */
const reduceCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 3;
  const input = fill([rows, dim], POSITIVE);
  return {
    name: "行 reduce sum",
    key: reduceKey({ op: "sum", dtype: "f32" }),
    wgsl: reduceWgsl({ op: "sum", dtype: "f32" }),
    params: reduceParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("sum", [input], { dim: 1 }, [rows]),
    // 1 行 = 1 workgroup（行長は workgroup 内で畳む）ので必要数は行数そのもの。
    natural: rows,
    groups: 2,
  };
};

/**
 * 軸 reduce 変種（最終次元以外）。行方向ではなく **`gid.x` の 1 次元 grid-stride**（1 スレッド =
 * 1 出力）なので、必要数は出力要素数 ÷ workgroup サイズ。
 */
const axisReduceCase = (): DegenerateCase => {
  const shape = [40, 3, 500];
  const axis = 1;
  const input = fill(shape, POSITIVE);
  return {
    name: "軸 reduce sum",
    key: axisReduceKey({ op: "sum", dtype: "f32" }),
    wgsl: axisReduceWgsl({ op: "sum", dtype: "f32" }),
    // out_count = 40·500 = 20,000 / axis_len = 3 / inner = 500
    params: axisReduceParams(20_000, shape[axis], 500),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("sum", [input], { dim: axis }, [40, 500]),
    natural: Math.ceil(20_000 / AXIS_REDUCE_WORKGROUP_SIZE),
    groups: 2,
  };
};

const softmaxCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  // 行ごとに違う分布にする（全行同じだと「別の行を読む」誤りが値に出ない）。
  const input = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  return {
    name: "softmax",
    key: SOFTMAX_KEY,
    wgsl: SOFTMAX_WGSL,
    params: softmaxParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

/**
 * 融合 attention の ②（行統計 — ADR 0023）。行方向 grid-stride の族で、期待値は
 * 「行ごとの `amax` と `1/Σexp(x − amax)`」。
 *
 * MUST: 期待値を GPU と同じ式で組まない。ここは `applyReferenceOp` を通せない
 * （行統計は op 単体ではない）ので、**safe-softmax の CPU 参照が出す分布から逆算**して
 * `inv` を作る — 走査順や縮約の段が壊れれば allclose で落ちる。
 */
const attentionStatsCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  // 行ごとに違う分布にする（全行同じだと「別の行を読む」誤りが値に出ない）。
  const input = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    // `p[j] = exp(x[j] − amax) · inv` なので、最大要素（exp(0) = 1）の確率がそのまま inv。
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  return {
    name: "attention_stats",
    key: ATTENTION_STATS_KEY,
    wgsl: ATTENTION_STATS_WGSL,
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * 融合 attention ② の **f16 計算変種**（S を f16 で読む — ADR 0028）。行方向 grid-stride の
 * 族が 1 本増えるので、族の MUST に従ってここにも載せる。
 *
 * MUST: 期待値は f16 に丸めた**後**の値から作る（`quantizeF16` の `values` は本番の
 * `decodeF16` が戻した値）。丸める前の値から組むと、丸め誤差ぶんだけ allclose の余裕を
 * 食い潰して「縮退の検出」ではなく「丸めの検出」になる。
 */
const attentionStatsF16Case = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const raw = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const quantized = quantizeF16(raw.data);
  const input = refTensor([rows, dim], quantized.values);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  return {
    name: "attention_stats c16",
    key: attentionStatsKey("f16"),
    wgsl: attentionStatsWgsl("f16"),
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [alignF16Payload(quantized.bytes)],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * 融合 attention ② の **S f16 格納変種**（`pack2x16float` の 2 要素／語 — 案 γ 波 1）。
 * 行方向 grid-stride の族がもう 1 本増えるので、族の MUST に従ってここにも載せる。
 *
 * `:c16` 版（上）と違い **`shader-f16` を要らない**（core WGSL）ので、別 device に分けず
 * 一括ハーネスへ入れる — これ自体が「案 γ は feature 非依存」の実行側の証拠になっている。
 * MUST: 期待値は f16 に丸めた**後**の値から作る（丸める前から組むと「縮退の検出」ではなく
 * 「丸めの検出」になる）。
 */
const attentionStatsS16Case = (): DegenerateCase => {
  const rows = 5_000;
  // MUST: dim は 4 の倍数（s16 の適格形は N % 4 == 0）かつ workgroup 幅 256 の倍数でない
  const dim = 4;
  const raw = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const rounded = Float32Array.from(raw.data, roundToF16);
  const input = refTensor([rows, dim], rounded);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  // GPU と同じ格納形（下位半分が偶数添字）へ詰める
  const words = new Uint32Array(rounded.length / 2);
  for (let i = 0; i < words.length; i += 1) {
    words[i] = (f32ToF16Bits(rounded[2 * i]) & 0xffff) |
      ((f32ToF16Bits(rounded[2 * i + 1]) & 0xffff) << 16);
  }
  return {
    name: "attention_stats s16",
    key: attentionStatsKey("f32", "f16"),
    wgsl: attentionStatsWgsl("f32", "f16"),
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [new Uint8Array(words.buffer)],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * 活性の per-token 量子化（w8a8 の前段 — src/kernels/quantize-rows.ts）。行方向 grid-stride の
 * 族で、比較対象は行ごとの scale `xs`（`xq` は i8 の 4 詰めで f32 として比較できないため
 * {@link DegenerateCase.sideOutputBytes} で確保だけする）。
 *
 * MUST: 行ごとに違う大きさの入力にする（全行同じ scale だと「別の行を読む」誤りが値に出ない）。
 */
const quantizeRowsCase = (): DegenerateCase => {
  const rows = 5_000;
  // dim は 4 の倍数（i8 の 4 詰めの前提）かつ workgroup 幅 256 の倍数でない
  const dim = 20;
  const input = fill([rows, dim], (i) => SIGNED(i) * (1 + (Math.floor(i / dim) % 7) * 0.5));
  return {
    name: "quantize_rows",
    key: QUANTIZE_ROWS_KEY,
    wgsl: QUANTIZE_ROWS_WGSL,
    params: quantizeRowsParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    sideOutputBytes: rows * dim,
    expected: refTensor([rows], quantizeRowsReference(input.data, rows, dim).scale),
    // 1 行 = 1 workgroup（行長は workgroup 内で畳む）ので必要数は行数そのもの
    natural: rows,
    groups: 2,
  };
};

const layerNormCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const eps = 1e-5;
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4);
  const weight = fill([dim], POSITIVE);
  const bias = fill([dim], SIGNED);
  return {
    name: "layer_norm",
    key: LAYER_NORM_KEY,
    wgsl: LAYER_NORM_WGSL,
    params: layerNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight, bias],
    expected: applyReferenceOp("layer_norm", [input, weight, bias], {
      normalized_shape: [dim],
      eps,
    }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

const gruScanCase = (): DegenerateCase => {
  // 行方向の族と同型（1 workgroup = 1 バッチ要素を `num_workgroups.x` で送る）。時間ループは
  // カーネル内なので縮退の対象はバッチ軸だけで、必要数は N そのもの。
  // MUST: バッチごとに違う `h0` を渡す（全バッチ同じだと「別のバッチを走査する」誤りが値に
  // 出ず、縮退耐性の検査が恒真化する）。
  const time = 3;
  const batch = 2_000;
  const hidden = 4;
  const gates = 3 * hidden;
  const gi = fill([time, batch, gates], (i) => ((i * 7) % 23) * 0.13 - 1.4);
  const initial = fill([batch, hidden], (i) => ((i * 5) % 19) * 0.09 - 0.8);
  const weight = fill([gates, hidden], (i) => ((i % 11) - 5) * 0.07);
  const bias = fill([gates], SIGNED);
  return {
    name: "gru_scan",
    key: gruScanKey("forward"),
    wgsl: gruScanWgsl("forward"),
    params: gruScanParams({ time, batch, hidden }),
    uniformParams: true,
    inputs: [gi, initial, weight, bias],
    expected: applyReferenceOp("gru_scan", [gi, initial, weight, bias], {}, [
      time,
      batch,
      hidden,
    ]),
    natural: batch,
    groups: 2,
  };
};

const rmsNormCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const eps = 1e-6;
  // 行ごとに違う二乗和にする（全行同じだと「別の行を読む」誤りが値に出ない）
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4);
  const weight = fill([dim], POSITIVE);
  return {
    name: "rms_norm",
    key: RMS_NORM_KEY,
    wgsl: RMS_NORM_WGSL,
    params: rmsNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight],
    expected: applyReferenceOp("rms_norm", [input, weight], { eps }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

const CASES: readonly DegenerateCase[] = [
  elementwiseCase(),
  stridedCase(),
  gatherCase(),
  maskedFillCase(),
  // 重み格納の変種を持つ 4 族は**変種ぶん**載せる（WGSL 本体が違う = 別の検出対象）。
  // linear はタイル固定で grid-stride を謳わないのでこのハーネスの対象外。
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
  softmaxCase(),
  // 融合 attention の 3 カーネルのうち行方向 grid-stride なのは ② だけ（①③ はタイル系で
  // カテゴリ違い — 安全網は tiledWorkgroups の fail loudly + full-write テスト）。
  attentionStatsCase(),
  // S の f16 格納変種（案 γ 波 1）は core WGSL なのでここに載る（`:c16` 版は別 device）
  attentionStatsS16Case(),
  // w8a8 の活性量子化（1 ノード 2 dispatch の前段 — i8a8 GEMM 側はタイル系でカテゴリ違い）
  quantizeRowsCase(),
  layerNormCase(),
  rmsNormCase(),
  gruScanCase(),
];

Deno.test({
  name: "grid-stride の全カーネル族が必要数より遥かに少ない workgroup でも全域を書く（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        // 縮退させていることの証拠。素直に割った必要数がこれだけあるのに groups 本で回す。
        assertEquals(
          testCase.natural > testCase.groups * 10,
          true,
          `${testCase.name}: 必要数 ${testCase.natural} に対し dispatch ${testCase.groups} は縮退と言えない`,
        );
        await runDegenerate(gpu, testCase);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * f16 計算変種（ADR 0028）の族。**別の device が要る**（`shader-f16` は device 作成時にしか
 * 要求できない）ので、上の一括ハーネスとは別のテストに分ける。列挙しないアダプタでは SKIP。
 */
const F16_CASES: readonly DegenerateCase[] = [attentionStatsF16Case()];

Deno.test({
  name: "f16 計算変種の grid-stride 族も縮退した dispatch で全域を書く（実 GPU）",
  ignore: !GPU_AVAILABLE || !SHADER_F16_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      for (const testCase of F16_CASES) {
        assertEquals(
          testCase.natural > testCase.groups * 10,
          true,
          `${testCase.name}: 必要数 ${testCase.natural} に対し dispatch ${testCase.groups} は縮退と言えない`,
        );
        await runDegenerate(gpu, testCase);
      }
    } finally {
      gpu.destroy();
    }
  },
});
