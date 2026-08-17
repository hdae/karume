// deform_conv2d（DCNv2 — ADR 0055）の**オラクル門**。新規原子には対になる既存経路が無いので、
// 正しさの根拠を 3 本に分けて立てる:
//
//   ① 退化 = conv2d の**ビット同一**（`offset` 全 0・`mask` 全 1）。tolerance ではなく
//      `Uint32Array` の完全一致で見る — 縮約順の変化（offset の読みを ic ループ外へ巻き上げる、
//      mask の掛け先を変える、bias を store 側で足す）は丸め列だけを動かすので、allclose では
//      沈黙する。ADR 0055 決定 2 / 決定 4 の実体。
//   ② **境界の意味論**（中心が範囲外 → タップ全体 0 / 内側でも範囲外の隅はその隅だけ 0）を、
//      1×1 カーネルの手計算値と**厳密一致**で突き合わせる。border clamp 実装はここでしか
//      赤くならない（allclose だと clamp と 0 埋めの差が許容差に紛れる形が作れる）。
//   ③ **NaN の伝播**。正の形の範囲判定だけだと NaN が 0 寄与へ落ちて沈黙誤値になるので、
//      実 GPU で本当に NaN が出ることを固定する（ADR 0020 の系譜 — ドライバが判定ごと畳む
//      可能性があるので、実測でしか担保できない）。
//
// torch（torchvision）そのものとの突合は golden `deform_conv2d_block`（e2e_golden_test.ts）。

import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { openModel } from "../src/format/container.ts";
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
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const SIGNED = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const WEIGHT = (i: number): number => (((i * 11) % 19) - 9) * 0.023;
/**
 * MUST: bias は非ゼロ。出力がちょうど 0 になる要素では、deform が足す `+0.0` が conv2d 側の
 * `−0.0` を `+0.0` に転ばせる（唯一の数値差分 — ADR 0055 決定 4 / ADR 0024 決定 3 と同型）。
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
    // BiRefNet の k=3 分岐と同型（同サイズ出力）。Cin ≠ Cout・Kh ≠ Kw で軸の取り違えも見る
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
    // k=1（ASPP の aspp1）。padding 0 なので範囲外タップが 1 本も出ない形
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
    // 大きい k（BiRefNet の k=7 分岐と同型）。padding が入力長に近く、範囲外タップが支配的
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
 * deform（offset 全 0・mask 全 1）と conv2d 直接カーネルを**同じ入力バッファ**で走らせ、
 * 出力のビット列を組で返す。x / weight / bias はどちらの dispatch も同じ GPUBuffer を束縛する
 * （ホスト側で 2 度書くと「書き込みの差」が「カーネルの差」に化ける）。
 *
 * MUST: 相手は conv2d の**直接カーネル**（`groups > 1` 用に恒久で残る側）。implicit GEMM は
 * タイル系で fma の畳み方が別なので、比較相手にすると「順序が同じこと」を見ていることに
 * ならない。
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
      const out = arena.allocStorage(Math.max(4, count * 4));
      arena.retain(out, 0, { pinned: true });
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
    scheduler.dispatch(deformPipeline, deformGroup, [
      gridStrideWorkgroups(count, DEFORM_CONV2D_WORKGROUP_SIZE, limit),
      1,
      1,
    ], DEFORM_CONV2D_KEY);

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
    scheduler.dispatch(convPipeline, convGroup, [
      gridStrideWorkgroups(count, CONV2D_WORKGROUP_SIZE, limit),
      1,
      1,
    ], convKey);

    await scheduler.flush();
    const [deform, conv] = await Promise.all(
      outputs.map((buffer) => readbackBits(gpu.device, buffer, count)),
    );
    return { deform, conv };
  } finally {
    await arena.destroy();
  }
};

Deno.test({
  name: "deform_conv2d は offset 0・mask 1 で conv2d 直接カーネルとビット同一（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of PARITY_CASES) {
        const { deform, conv } = await runBoth(gpu, testCase);
        assert(deform.length > 0, `${testCase.name}: 出力が空`);
        // Uint32 の完全一致（tolerance では丸め列の変化が沈黙する）
        assertEquals(
          [...deform],
          [...conv],
          `${testCase.name}: 退化ケースのビット列が conv2d と違う`,
        );
        // 恒真化の門: 出力が bias 一色（= 縮約が 1 度も走っていない）でないこと
        assert(
          new Set(deform).size > 1,
          `${testCase.name}: 出力が単一値（縮約が効いていない恒真な比較）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/** 単一ノードグラフで deform_conv2d を 1 回走らせる（executor 経路）。 */
const runDeform = async (
  gpu: GpuContext,
  inputs: readonly Tensor[],
  outShape: readonly number[],
  padding: readonly [number, number],
): Promise<Tensor> => {
  const graph = singleOpGraph(
    "deform_conv2d",
    inputs.map((input) => input.shape),
    [outShape],
    { attrs: { padding: [...padding] } },
  );
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
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

Deno.test({
  name: "deform_conv2d の境界は clamp ではなくゼロ埋め（手計算値と厳密一致 / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 1×1 カーネル・Cin=Cout=1・weight 1・bias 0 なので、出力は
    // `mask · bilinear(x, oy + off_y, ox + off_x)` そのもの（値は f32 で厳密）。
    const x = fill([1, 1, 2, 2], (i) => 1 + i);
    const weight = fill([1, 1, 1, 1], () => 1);
    const bias = fill([1], () => 0);
    const mask = fill([1, 1, 2, 2], () => 1);
    const gpu = await acquireGpu();
    try {
      const run = async (shiftY: number, shiftX: number): Promise<readonly number[]> => {
        // offset は `[y 平面 4 要素, x 平面 4 要素]`（偶数チャネル = y / 奇数 = x）
        const offset = fill([1, 2, 2, 2], (i) => (i < 4 ? shiftY : shiftX));
        const out = await runDeform(gpu, [x, weight, offset, mask, bias], [1, 1, 2, 2], [0, 0]);
        return [...out.data];
      };
      // 境界の内側 — 下側の隅だけ範囲外（clamp なら 3, 4 が出るところが 1.5, 2 になる）
      assertEquals(await run(0.5, 0), [2, 3, 1.5, 2]);
      // 中心が −1.5 ≤ −1 → タップ全体 0（clamp なら 1, 2 が出る）
      assertEquals(await run(-1.5, 0), [0, 0, 0.5, 1]);
      // MUST: 偶数 = y / 奇数 = x。入れ替えると値が別物になる形で固定する
      assertEquals(await run(1, 0), [3, 4, 0, 0]);
      assertEquals(await run(0, 1), [2, 0, 4, 0]);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "deform_conv2d の NaN offset は 0 に落ちず出力へ伝播する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 正の形の範囲判定（`> −1 && < in`）は NaN で false 側に落ちるので、ビット列判定で
    // NaN を先に分けていなければ**この出力は 0 になる**（沈黙誤値）。
    const x = fill([1, 1, 2, 2], (i) => 1 + i);
    const weight = fill([1, 1, 1, 1], () => 1);
    const bias = fill([1], () => 0);
    const mask = fill([1, 1, 2, 2], () => 1);
    // y 平面の先頭だけ NaN・2 番目は +Inf（Inf は torch と同じく「範囲外 = 0」）
    const offset = fill([1, 2, 2, 2], (i) => {
      if (i === 0) return Number.NaN;
      if (i === 1) return Number.POSITIVE_INFINITY;
      return 0;
    });
    const gpu = await acquireGpu();
    try {
      const out = await runDeform(gpu, [x, weight, offset, mask, bias], [1, 1, 2, 2], [0, 0]);
      assertEquals(Number.isNaN(out.data[0]), true, "NaN の offset が 0 に落ちている");
      assertEquals(out.data[1], 0, "±Inf の offset は範囲外 = 0（NaN ではない）");
      assertEquals([...out.data.slice(2)], [3, 4], "他の要素は巻き添えにならない");
    } finally {
      gpu.destroy();
    }
  },
});
