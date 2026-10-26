// conv3d の w8a8 変種（`convCompute: "a8"` — ADR 0133）の実 GPU の門。
//
// 1. **TS 鏡像との atol=0**（`referenceConv3dI8a8` — tap ごとの整数内積と fma の flush を同じ順で
//    行う。クラス E）。形は chunk グラフの縮小形 + タイル端（Cout > 128 で行タイル 2 枚・N % 4 ≠ 0 の
//    スカラ書き出し・Cin % 16 ≠ 0 の K 端数）+ 3 軸 padding + stride ≠ 1 + dilation ≠ 1 + Kt ≠ Kh ≠ Kw。
// 2. **dp4a 版とエミュ版が atol=0 で一致**（拡張の有無は速度だけ — linear と同じ主張の conv 側の検出器）。
// 3. **既定（f32 の implicit GEMM）とは値が割れる**（活性量子化が効いている裏取り — 一致するなら
//    経路が切り替わっていない）。
// 4. **適格でない node は f32 経路に落ち、キーがそれを示す**（Cin % 4 ≠ 0）。
// 5. 鏡像の **tap の並び**は Kt, Kh, Kw の昇順であること（Kh / Kw を入れ替えた鏡像は GPU と割れる —
//    恒真化の防止）。
//
// MUST: 比較は `assertExact`（`===` の全数比較）。許容誤差を入れた時点でクラス E の主張が消える。
// MUST: 活性の丸め境界からの余裕（`quantizeRowsTieMargin` — 転置した [画素, Cin] で測る）を門にする
// （GPU の除算は 2.5 ULP まで許されるので、余裕の無いデータでは ±1 段のずれが出る — linear と同じ）。

import { test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { type Conv3dDims, conv3dIgemmKey, conv3dUsesVec4 } from "../src/kernels/conv3d.ts";
import {
  CONV3D_I8A8_QUANTIZE_GEOMETRY,
  conv3dI8a8Eligible,
  conv3dI8a8Geometry,
  conv3dI8a8Key,
  conv3dI8a8UsesVec4,
} from "../src/kernels/conv3d-i8a8.ts";
import { conv2dIgemmMTile } from "../src/kernels/conv2d.ts";
import { quantizeRowsChannelMajorKey } from "../src/kernels/quantize-rows.ts";
import { referenceConv3dI8a8 } from "../src/reference/conv3d-i8a8.ts";
import { quantizeRowsTieMargin } from "../src/reference/i8a8.ts";
import {
  createSessionFromContainer,
  I8A8_DOT,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** 符号付き・画素ごとに大きさの違うデータ（画素ごとの scale が全画素同じにならないように）。 */
const SIGNED = (i: number): number => ((i % 29) - 14) * 0.3717 + 0.0131;
/** 行（= 出力チャネル）ごとに大きさの違う重み（全行同じ scale だと軸の取り違えが値に出ない）。 */
const WEIGHT = (i: number, rowLength: number): number =>
  (((i * 7) % 23) - 11) * 0.0625 * (1 + (Math.floor(i / rowLength) % 5) * 0.5);
const BIAS = (i: number): number => ((i % 11) - 5) * 0.25 + 0.0625;
const TIE_MARGIN = 1e-3;

const assertExact = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: 長さ`);
  for (let i = 0; i < actual.length; i += 1) {
    assert(actual[i] === expected[i], `${where}[${i}]: GPU ${actual[i]} / 参照 ${expected[i]}`);
  }
};

type Triple = readonly [number, number, number];
const ONE: Triple = [1, 1, 1];

type Conv3dCase = {
  readonly name: string;
  readonly channelsIn: number;
  readonly channelsOut: number;
  /** 入力 [T, H, W]。 */
  readonly input: Triple;
  readonly kernel: Triple;
  readonly stride: Triple;
  readonly padding: Triple;
  readonly dilation: Triple;
  /** 踏むはずの書き出し変種（判定の取り違えが「両方同じ変種」で紛れないように明示する）。 */
  readonly expectVec4: boolean;
  /**
   * 入力の生成（既定は SIGNED）。Cin が小さい形は画素ごとの scale が 4〜6 値から決まり、SIGNED では
   * x/s が丸め境界に寄るので、比が 127 の既知の分数になる生成に差し替える（余裕の門が通る形）。
   */
  readonly x?: (channel: number, pixel: number) => number;
};

/** 画素ごとの最大チャネルを回しながら、比が 127·{1, 0.8, −0.6, 0.4, …} に落ちる入力。 */
const fractionsOf =
  (fractions: readonly number[]) =>
  (channel: number, pixel: number): number =>
    (0.5 + (pixel % 13) * 0.37) * fractions[(channel + pixel) % fractions.length];

const outExtent = (
  input: number,
  kernel: number,
  stride: number,
  padding: number,
  dilation: number,
): number => Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

const dimsOf = (testCase: Conv3dCase): Conv3dDims => ({
  channelsIn: testCase.channelsIn,
  channelsOut: testCase.channelsOut,
  timeIn: testCase.input[0],
  heightIn: testCase.input[1],
  widthIn: testCase.input[2],
  timeOut: outExtent(
    testCase.input[0],
    testCase.kernel[0],
    testCase.stride[0],
    testCase.padding[0],
    testCase.dilation[0],
  ),
  heightOut: outExtent(
    testCase.input[1],
    testCase.kernel[1],
    testCase.stride[1],
    testCase.padding[1],
    testCase.dilation[1],
  ),
  widthOut: outExtent(
    testCase.input[2],
    testCase.kernel[2],
    testCase.stride[2],
    testCase.padding[2],
    testCase.dilation[2],
  ),
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
});

const CASES: readonly Conv3dCase[] = [
  {
    // chunk グラフの conv_in（48→1024 k3・時間 padding 0・空間 same）の縮小形。Cin = 48 は 16 の倍数
    // （tap の中で K 端数なし）・N = 1·6·8 = 48（v4）
    name: "chunk conv_in 縮小 48→96 k3 [48,3,6,8]",
    channelsIn: 48,
    channelsOut: 96,
    input: [3, 6, 8],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    expectVec4: true,
  },
  {
    // Cout = 130 > タイル辺 128（行タイル 2 枚・2 枚目は 2 行だけ）× 1×1×1（tap 1 本 = flush 1 回）×
    // N = 2·5·7 = 70（スカラ書き出し）× Cin = 8（パック 2 つ = K タイル 1 枚の 4 パック中 2 パック）
    name: "行タイル端 130 出力チャネル 1×1×1 [8,2,5,7]",
    channelsIn: 8,
    channelsOut: 130,
    input: [2, 5, 7],
    kernel: [1, 1, 1],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    expectVec4: false,
  },
  {
    // stride ≠ 1 を 3 軸とも非対称に・Kt ≠ Kh ≠ Kw・padding 非対称・Cin = 20（16 の倍数でない → tap の
    // 2 枚目の K タイルは 4 パック中 1 パック）。T: (5+2−2−1)/2+1 = 3 / H: (9+2−1−1)/2+1 = 5 /
    // W: (10−3−1)/3+1 = 3 → N = 45（スカラ）
    name: "stride=[2,2,3] Kt≠Kh≠Kw Cin 20 [20,5,9,10] * W[6,20,3,2,4]",
    channelsIn: 20,
    channelsOut: 6,
    input: [5, 9, 10],
    kernel: [3, 2, 4],
    stride: [2, 2, 3],
    padding: [1, 1, 0],
    dilation: ONE,
    expectVec4: false,
  },
  {
    // dilation ≠ 1 を 3 軸とも・Cin = 4（1 パック = K タイル 1 枚の 4 パック中 1 パック）・N = 5·6·8 = 240
    // （列タイル 64 で 4 枚・v4）
    name: "dilation=[3,3,2] Cin 4 [4,8,12,10] * W[6,4,2,3,2]",
    channelsIn: 4,
    channelsOut: 6,
    input: [8, 12, 10],
    kernel: [2, 3, 2],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: [3, 3, 2],
    expectVec4: true,
    x: fractionsOf([1, 0.8, -0.6, 0.4]),
  },
  {
    // 全軸 padding 1（範囲外 0 を 3 軸とも踏む — 範囲外の tap が scale 0 で畳まれる検出器）。
    // N = 4·5·7 = 140（列タイル 3 枚・v4）
    name: "全軸 padding 1 [12,4,5,7] * W[7,12,3,3,3]",
    channelsIn: 12,
    channelsOut: 7,
    input: [4, 5, 7],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [1, 1, 1],
    dilation: ONE,
    expectVec4: true,
  },
];

type Prepared = {
  readonly dims: Conv3dDims;
  readonly model: Promise<BoundContainer>;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
  /** 鏡像の入力（tap の並びの故障注入に使う）。 */
  readonly mirror: {
    readonly x: Float32Array<ArrayBuffer>;
    readonly weight: Float32Array<ArrayBuffer>;
    readonly weightScale: Float32Array<ArrayBuffer>;
    readonly bias: Float32Array<ArrayBuffer>;
  };
};

/** `conv3d(x, w, b)` 1 本の宣言と供給（w は i8 + per-channel scale）。 */
const prepare = (testCase: Conv3dCase): Prepared => {
  const dims = dimsOf(testCase);
  const { channelsIn, channelsOut } = dims;
  const xShape = [channelsIn, dims.timeIn, dims.heightIn, dims.widthIn];
  const wShape = [channelsOut, channelsIn, dims.kernelT, dims.kernelH, dims.kernelW];
  const rowLength = channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  const weight = fill(wShape, (i) => WEIGHT(i, rowLength));
  assert(weight.dtype === "f32");
  const quantized = quantizeI8(weight.data, wShape, 0);
  const bias = fill([channelsOut], BIAS);
  assert(bias.dtype === "f32");
  const pixels = dims.timeIn * dims.heightIn * dims.widthIn;
  const generate = testCase.x;
  const x = fill(
    xShape,
    generate === undefined ? SIGNED : (i) => generate(Math.floor(i / pixels), i % pixels),
  );
  assert(x.dtype === "f32");
  // 丸め境界からの余裕は転置した [画素, Cin] で測る（カーネルが量子化する並び）
  const transposed = new Float32Array(pixels * channelsIn);
  for (let ci = 0; ci < channelsIn; ci += 1) {
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      transposed[pixel * channelsIn + ci] = x.data[ci * pixels + pixel];
    }
  }
  const margin = quantizeRowsTieMargin(transposed, pixels, channelsIn);
  assert(margin > TIE_MARGIN, `${testCase.name}: 丸め境界からの余裕が ${margin} しかない`);
  // 鏡像は量子化後の整数値を重みに取る（fake-quant 値 ÷ scale — 整数に正確）
  const weightInts = new Float32Array(quantized.values.length);
  for (let i = 0; i < weightInts.length; i += 1) {
    const channel = Math.floor(i / rowLength);
    weightInts[i] = Math.round(quantized.values[i] / quantized.scale[channel]);
  }
  const mirror = { x: x.data, weight: weightInts, weightScale: quantized.scale, bias: bias.data };
  // 適格でない形（Cin % 4 ≠ 0）は鏡像の外（f32 経路に落ちることだけ見る）
  const expected = conv3dI8a8Eligible(dims, "i8")
    ? referenceConv3dI8a8({ ...mirror, dims })
    : new Float32Array(0);
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
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [channelsOut, dims.timeOut, dims.heightOut, dims.widthOut] },
    },
    nodes: [
      {
        op: "conv3d",
        ins: ["x", "w", "b"],
        outs: ["y"],
        attrs: {
          stride: [...testCase.stride],
          padding: [...testCase.padding],
          dilation: [...testCase.dilation],
          groups: 1,
        },
      },
    ],
  };
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    },
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int8-sym",
        groupSize: perChannelGroupSize(rowLength),
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    },
  ];
  return {
    dims,
    model: openModelBytes(declaration, tensors),
    x: { dtype: "f32", shape: xShape, data: x.data },
    expected,
    mirror,
  };
};

const run = async (
  gpu: GpuContext,
  prepared: Prepared,
  options: SessionOptions,
): Promise<{ readonly y: Tensor; readonly keys: string[] }> => {
  const session = await createSessionFromContainer(gpu, await prepared.model, GRAPH_NAME, options);
  try {
    const outputs = await session.run({ x: prepared.x });
    return {
      y: outputs["y"],
      keys: requireCensus(session.diagnostics().lastRunPipelines, "conv3d")
        .map((row) => row.key)
        .sort(),
    };
  } finally {
    await session.dispose();
  }
};

/**
 * The two keys that must run (per-pixel quantisation read straight from the channel-major input →
 * the integer dot-product implicit GEMM; no transposing copy — ADR 0133 addendum 2026-10-26).
 */
const expectedKeys = (dims: Conv3dDims, dp4a: boolean): string[] =>
  [
    quantizeRowsChannelMajorKey(CONV3D_I8A8_QUANTIZE_GEOMETRY),
    conv3dI8a8Key(
      conv3dI8a8UsesVec4(dims.timeOut * dims.heightOut * dims.widthOut),
      dp4a,
      conv3dI8a8Geometry(),
    ),
  ].sort();

test.skipIf(!GPU_AVAILABLE)(
  "convCompute:'a8' の conv3d が TS 鏡像と atol=0 で一致する（chunk の縮小形 / タイル端 / K 端数 / 3 軸 padding / stride / dilation・実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const prepared = prepare(testCase);
        const { dims } = prepared;
        assert(conv3dI8a8Eligible(dims, "i8"), `${testCase.name}: 適格でない形`);
        assertEquals(
          conv3dI8a8UsesVec4(dims.timeOut * dims.heightOut * dims.widthOut),
          testCase.expectVec4,
          `${testCase.name}: 踏む書き出し変種が想定と違う`,
        );
        const actual = await run(gpu, prepared, { convCompute: "a8" });
        assertEquals(
          actual.y.shape,
          [dims.channelsOut, dims.timeOut, dims.heightOut, dims.widthOut],
          testCase.name,
        );
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // 恒真化の門: 出力が定数なら「一致」は何も検証していない
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: 出力が定数`);
        assertEquals(actual.keys, expectedKeys(dims, true), `${testCase.name}: 走ったキー`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8: dot4I8Packed 版とエミュ版が atol=0 で一致する（拡張の有無は速度だけ・実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const prepared = prepare(testCase);
        const results: Partial<Record<"dp4a" | "emu", Tensor>> = {};
        for (const dot of ["dp4a", "emu"] as const) {
          const actual = await run(gpu, prepared, { convCompute: "a8", [I8A8_DOT]: dot });
          assertEquals(
            actual.keys,
            expectedKeys(prepared.dims, dot === "dp4a"),
            `${testCase.name}: ${dot} のキー`,
          );
          results[dot] = actual.y;
        }
        assert(results.dp4a !== undefined && results.emu !== undefined);
        assertExact(results.emu.data, results.dp4a.data, `${testCase.name}: emu vs dp4a`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8 は opt-in のときだけ効き、既定の f32 implicit GEMM とは値が割れる（活性量子化の裏取り・実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    try {
      const testCase = CASES[0];
      const prepared = prepare(testCase);
      const { dims } = prepared;
      const baseline = await run(gpu, prepared, {});
      const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
      assertEquals(
        baseline.keys,
        [
          conv3dIgemmKey(
            "i8",
            conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW),
            conv2dIgemmMTile(dims.channelsOut),
          ),
        ],
        "既定のキー（f32 の implicit GEMM 1 本）",
      );
      const a8 = await run(gpu, prepared, { convCompute: "a8" });
      // MUST: 既定の値は活性 f32 のままなので w8a8 の鏡像とは一致しない（一致するなら活性量子化が
      // 効いていない）
      assertNotEquals([...baseline.y.data], [...a8.y.data], "a8 が既定と同じ値");
      assertExact(a8.y.data, prepared.expected, "a8");
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8: Cin % 4 ≠ 0 の node は opt-in でも f32 経路に落ち、キーがそれを示す（実 GPU）",
  async () => {
    const gpu = await acquireGpu();
    try {
      const testCase: Conv3dCase = {
        ...CASES[3],
        name: "Cin 6",
        channelsIn: 6,
        x: fractionsOf([1, 0.8, -0.6, 0.4, -0.2, 0.9]),
      };
      const dims = dimsOf(testCase);
      assert(!conv3dI8a8Eligible(dims, "i8"));
      // prepare は余裕の門を持つが、適格でない形は鏡像を比較しないので門だけ通す
      const prepared = prepare(testCase);
      const actual = await run(gpu, prepared, { convCompute: "a8" });
      const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
      assertEquals(actual.keys, [
        conv3dIgemmKey(
          "i8",
          conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW),
          conv2dIgemmMTile(dims.channelsOut),
        ),
      ]);
    } finally {
      gpu.destroy();
    }
  },
);

test("conv3d i8a8 の鏡像は tap を Kt, Kh, Kw の昇順に畳む（Kh / Kw を入れ替えた鏡像は割れる）", () => {
  // Kh ≠ Kw の形で、重みの (kh, kw) を入れ替えて渡すと値が変わる（対称な核なら恒真になる）
  const testCase = CASES[2];
  const prepared = prepare(testCase);
  const { dims, mirror } = prepared;
  const swapped = new Float32Array(mirror.weight.length);
  const { channelsOut, channelsIn, kernelT, kernelH, kernelW } = dims;
  for (let co = 0; co < channelsOut; co += 1) {
    for (let ci = 0; ci < channelsIn; ci += 1) {
      for (let kt = 0; kt < kernelT; kt += 1) {
        for (let kh = 0; kh < kernelH; kh += 1) {
          for (let kw = 0; kw < kernelW; kw += 1) {
            const from = (((co * channelsIn + ci) * kernelT + kt) * kernelH + kh) * kernelW + kw;
            // (kh, kw) → (kw % Kh, kh % Kw): 核が非正方なので全単射ではないが、値が動くことだけ見る
            const to =
              (((co * channelsIn + ci) * kernelT + kt) * kernelH + (kw % kernelH)) * kernelW +
              (kh % kernelW);
            swapped[to] = mirror.weight[from];
          }
        }
      }
    }
  }
  const other = referenceConv3dI8a8({ ...mirror, weight: swapped, dims });
  assertNotEquals([...other], [...prepared.expected]);
});
