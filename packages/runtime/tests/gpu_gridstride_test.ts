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
// 行方向の族（行 reduce / **argmax** / **topk** / softmax /
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
import type { IrDtype } from "../src/format/ir.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { refTensor } from "../src/reference/ops.ts";
import { WEIGHT_STORAGES } from "../src/kernels/weight-storage.ts";
import {
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
  gruScanCase,
  layerNormCase,
  maskedFillCase,
  padCase,
  quantizeRowsCase,
  reduceCase,
  rmsNormCase,
  softmaxCase,
  stridedCase,
  stridedWriteCase,
  topkCase,
  upsampleBilinear2dCase,
} from "./helpers/gridstride_cases.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * 出力バッファを**期待値の意味論 dtype で**読む（ADR 0009 の実表現 — bool は u32 の 0/1）。
 *
 * MUST: dtype を固定で f32 にしない。`argmax` は i32（添字）を書くので、f32 として読むと
 * ビット列の読み替えになって突合が丸ごと無意味になる（比較は必ず落ちるが、落ち方が
 * 「縮退耐性の欠陥」に見えて原因を隠す）。
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

    const actual = await readback(gpu.device, dst, count, testCase.expected.dtype);
    const report = compareTensors(
      refTensor(testCase.expected.shape, actual),
      testCase.expected,
    );
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
  // argmax は行 reduce と同じ行方向 grid-stride の**別族**（identity −inf + index 追跡）。
  // 出力が i32 なので readback は期待値の dtype で view を張る。
  argmaxCase(),
  // topk も同じ行方向 grid-stride の別族（レーン局所 top-k → トーナメント merge・出力 2 本）。
  // 突合は添字の列で、値の列は sideOutputBytes 側（helpers/gridstride_cases.ts の doc）。
  topkCase(),
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
