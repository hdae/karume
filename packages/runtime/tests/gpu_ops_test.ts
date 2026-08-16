// 実 GPU 実行 vs CPU 参照の数値突合（ADR 0005 の段 2）。M0 の全 op を代表 shape で回す。
// MUST: 乱数を使わない — 失敗が再現しないと原因の切り分けができない。

import { assertEquals, assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp } from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import {
  ATTENTION_CASES,
  BINARY_CASES,
  BMM_CASES,
  BOUNDARY_CASES,
  DEFORM_CASES,
  DTYPE_CASES,
  FUSED_CASES,
  GATHER_CASES,
  GRU_SCAN_CASES,
  INTEGERS,
  LAYOUT2_CASES,
  LAYOUT_CASES,
  MATH_CASES,
  MATMUL_CASES,
  type OpCase,
  POSITIVE,
  REDUCE_CASES,
  SIGNED,
  UNARY_CASES,
  UPSAMPLE_CASES,
} from "./helpers/gpu_op_cases.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const runCase = async (gpu: GpuContext, testCase: OpCase): Promise<Tensor> => {
  const graph = singleOpGraph(
    testCase.op,
    testCase.inputs.map((input) => input.shape),
    testCase.outShape,
    {
      inDtypes: testCase.inputs.map((input) => input.dtype),
      outDtype: testCase.outDtype ?? testCase.inputs[0].dtype,
      attrs: testCase.attrs,
    },
  );
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    const named: Record<string, Tensor> = {};
    testCase.inputs.forEach((input, index) => {
      named[`x${index}`] = input;
    });
    const outputs = await session.run(named);
    return outputs["y"];
  } finally {
    await session.dispose();
  }
};

const checkAll = async (cases: readonly OpCase[]): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    for (const testCase of cases) {
      const actual = await runCase(gpu, testCase);
      const expected = applyReferenceOp(
        testCase.op,
        testCase.inputs,
        testCase.attrs ?? {},
        testCase.outShape,
      );
      assertEquals(actual.shape, expected.shape, testCase.name);
      assertEquals(actual.dtype, expected.dtype, `${testCase.name}: dtype`);
      // f32 は allclose、i32 / bool は厳密一致（整数演算に丸め差は無い — ADR 0009）
      const report = compareTensors(actual, expected);
      assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
    }
  } finally {
    gpu.destroy();
  }
};

Deno.test({
  name: "unary 9 種が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(UNARY_CASES),
});

Deno.test({
  name: "binary 4 種が broadcast 込みで CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(BINARY_CASES),
});

Deno.test({
  name: "i32 / bool の elementwise（mask 経路）が CPU 参照と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(DTYPE_CASES),
});

Deno.test({
  name:
    "波3 の数理 op（where / 比較 / clamp / leaky_relu / log1p / cumsum / bool sum）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(MATH_CASES),
});

/**
 * leaky_relu の NaN 伝播（ADR 0015 の裁定を実 GPU で固定する）。
 *
 * torch は `leaky_relu(NaN) = NaN` だが、WGSL の `max` は NaN 伝播を保証しない実装が
 * ありうる（relu / clamp が既に同じ乖離を抱えていることは limitations.md が既知化済み）。
 * MUST: allclose は NaN をどちらの側でも不合格にするので、この検査は checkAll に載せられない
 * — 生の値を見る専用テストとして持つ。
 */
Deno.test({
  name: "leaky_relu は NaN を伝播する（select 形の裁定 — 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "leaky_relu NaN",
        op: "leaky_relu",
        inputs: [fill([4], (i) => [Number.NaN, -2, 0, 3][i])],
        outShape: [4],
        attrs: { negative_slope: 0.1 },
      });
      const values = [...actual.data];
      assertEquals(values.map((v) => Number.isNaN(v)), [true, false, false, false]);
      // NaN 以外は通常経路（負側は slope 倍・正側は素通し）
      assertEquals(values[1], Math.fround(-0.2));
      assertEquals(values[2], 0);
      assertEquals(values[3], 3);
    } finally {
      gpu.destroy();
    }
  },
});

/** NaN の位置（先頭 / 中間 / 末尾）。elementwise の入力長 4 に対する添字。 */
const NAN_POSITIONS = [0, 2, 3] as const;
/** NaN を差し込む前の素の列。clamp（min=-1 / max=1）の下限側・上限側・素通しを跨ぐ。 */
const NAN_BASE = [-3, -0.5, 0.5, 3] as const;

/** 同じ op について「NaN の位置を変えた 3 本 + NaN 無しの対照 1 本」を組む。 */
const nanUnaryCases = (op: string, attrs?: Record<string, unknown>): readonly OpCase[] => [
  ...NAN_POSITIONS.map((position) => ({
    name: `${op} NaN@${position}`,
    op,
    inputs: [fill([4], (i) => (i === position ? Number.NaN : NAN_BASE[i]))],
    outShape: [4],
    attrs,
  })),
  {
    name: `${op} NaN 無し（対照）`,
    op,
    inputs: [fill([4], (i) => NAN_BASE[i])],
    outShape: [4],
    attrs,
  },
];

/** 行ごとの NaN 位置（-1 = NaN 無しの対照行）。縮約は「行内のどこにあっても」伝播する。 */
const NAN_REDUCE_AT = [0, 2, 3, -1] as const;
const NAN_REDUCE_ROWS = (index: number): number => {
  const row = Math.floor(index / 4);
  const col = index % 4;
  return col === NAN_REDUCE_AT[row] ? Number.NaN : (col - 1.5) * (row + 1);
};

const NAN_CASES: readonly OpCase[] = [
  ...nanUnaryCases("clamp", { min: -1, max: 1 }),
  ...nanUnaryCases("clamp_min", { min: 0 }),
  ...nanUnaryCases("relu"),
  ...(["amax", "amin"] as const).flatMap((op): readonly OpCase[] => [
    {
      name: `${op} 行内の位置を変えた NaN [4,4]`,
      op,
      inputs: [fill([4, 4], NAN_REDUCE_ROWS)],
      outShape: [4],
      attrs: { dim: 1 },
    },
    {
      // 行長 300 > workgroup サイズ 256 → 添字 299 は 1 スレッドの走査ループの **2 周目**で
      // 読まれる（[4,4] は 1 周目しか回らないので、この経路はここでしか踏まない）。
      // 行 1 は NaN 無しの対照。
      name: `${op} 走査ループ 2 周目の NaN [2,300]`,
      op,
      inputs: [fill([2, 300], (i) => (i === 299 ? Number.NaN : SIGNED(i)))],
      outShape: [2],
      attrs: { dim: 1 },
    },
  ]),
];

/**
 * clamp / clamp_min / relu / amax / amin の NaN 伝播（ビット列判定の裁定を実 GPU で固定）。
 *
 * torch は 5 op とも「入力（縮約なら縮約対象の語群）に NaN があれば結果は NaN」。ドライバの
 * `max` / `min` は NaN を飲むので、伝播を担うのは生成 WGSL のビット列判定だけ
 * （機序は src/codegen/elementwise.ts の IS_NAN_FN）。
 * MUST: allclose は NaN をどちらの側でも不合格にするので checkAll には載せられない —
 * **CPU 参照との isNaN パターン一致**で突合する（NaN のビットパターン一致までは求めない）。
 */
Deno.test({
  name: "clamp / clamp_min / relu / amax / amin が NaN を伝播する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of NAN_CASES) {
        const actual = await runCase(gpu, testCase);
        const expected = applyReferenceOp(
          testCase.op,
          testCase.inputs,
          testCase.attrs ?? {},
          testCase.outShape,
        );
        const actualValues = [...actual.data];
        const expectedValues = [...expected.data];
        const actualNan = actualValues.map((value) => Number.isNaN(value));
        const expectedNan = expectedValues.map((value) => Number.isNaN(value));
        assertEquals(actualNan, expectedNan, `${testCase.name}: NaN の位置`);
        // MUST: 対照要素を持つこと。「常に NaN を返す」実装は位置の一致だけでは落ちない。
        assertEquals(expectedNan.includes(false), true, `${testCase.name}: 対照要素が無い`);
        // 非 NaN 側は丸め差の入らない値なので生で厳密一致（NaN 同士は比較が成立しないので
        // 該当要素だけ 0 に潰してから見る）
        const finiteOnly = (values: readonly number[], isNan: readonly boolean[]): number[] =>
          values.map((value, index) => (isNan[index] ? 0 : value));
        assertEquals(
          finiteOnly(actualValues, actualNan),
          finiteOnly(expectedValues, expectedNan),
          `${testCase.name}: 非 NaN の値`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * log1p の精度（実装方針の実測根拠 — src/codegen/elementwise.ts の LOG1P_FN）。
 *
 * 素朴な `log(1 + x)` は |x| ≪ 1 で `1 + x` の丸めが有効桁を食う。ここは f32 で
 * 「1 + x が 1 に丸まる領域」を含む x を与え、CPU 参照（`Math.log1p`）との**相対誤差**が
 * f32 の数 ulp に収まることを実測する。素朴形なら x = 1e-8 で出力 0（相対誤差 1）になり、
 * この閾値では通らない。
 */
Deno.test({
  name: "log1p は 1+x が丸まる領域でも相対誤差 f32 数 ulp に収まる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const smalls = [1e-8, -1e-8, 1e-7, 5e-7, -3e-6, 1e-4, -1e-3, 0.5, 2, 100];
    const gpu = await acquireGpu();
    let worst = 0;
    try {
      const actual = await runCase(gpu, {
        name: "log1p small",
        op: "log1p",
        inputs: [fill([smalls.length], (i) => smalls[i])],
        outShape: [smalls.length],
      });
      smalls.forEach((x, index) => {
        const expected = Math.log1p(Math.fround(x));
        const relative = Math.abs((actual.data[index] - expected) / expected);
        worst = Math.max(worst, relative);
      });
    } finally {
      gpu.destroy();
    }
    // f32 の eps は 1.19e-7。log / 除算・乗算の合成で数 ulp を見込んで 8 倍を上限にする
    // （素朴形の誤差は x = 1e-8 で 1.0 = この閾値の 7 桁上）。
    assertEquals(worst < 1e-6, true, `log1p の最悪相対誤差 ${worst}`);
  },
});

Deno.test({
  name: "matmul がタイル端数込みで CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(MATMUL_CASES),
});

Deno.test({
  name: "bmm が非対称形・タイル端数込みで CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(BMM_CASES),
});

Deno.test({
  name: "融合 attention が CPU 参照と一致する（軸全異・端数・D≠128 込み / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(ATTENTION_CASES),
});

Deno.test({
  name: "gather（最終次元固定）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(GATHER_CASES),
});

/**
 * 範囲外添字の裁定（src/kernels/gather.ts）を実 GPU で固定する。
 *
 * GPU は例外を投げられないので、範囲外の要素**だけ**を NaN で汚染する（別の要素を静かに
 * 返さない）。オラクル側の CPU 参照は同じ入力で必ず throw する — 両者が一致してしまうと
 * 「契約違反のグラフが突合を通る」ことになるので、ここは意図的に非対称であることを固定する。
 */
Deno.test({
  name: "gather の範囲外添字は GPU で NaN 汚染・CPU 参照で throw（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const src = fill([2, 4], POSITIVE);
    // 添字 4（上限超え）と -1（負）をそれぞれ 1 本ずつ混ぜ、残りは正しい列を引く
    const index = fill([2, 3], (i) => [1, 4, 0, 2, -1, 3][i], "i32");
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "gather oob",
        op: "gather",
        inputs: [src, index],
        outShape: [2, 3],
      });
      const values = [...actual.data];
      assertEquals(values.map((v) => Number.isNaN(v)), [false, true, false, false, true, false]);
      // 契約内の添字は通常どおり自分の行から引く（汚染は範囲外の要素だけ）
      assertEquals(values[0], src.data[1]);
      assertEquals(values[3], src.data[4 + 2]);
    } finally {
      gpu.destroy();
    }
    assertThrows(() => applyReferenceOp("gather", [src, index], {}, [2, 3]));
  },
});

Deno.test({
  name: "行 reduce 3 種が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(REDUCE_CASES),
});

Deno.test({
  name: "レイアウト 3 種（別名 reshape / strided permute・expand）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(LAYOUT_CASES),
});

Deno.test({
  name: "レイアウト第 2 群（slice / cat / pad / flip）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(LAYOUT2_CASES),
});

Deno.test({
  name:
    "融合 op 10 種（linear / layer_norm / rms_norm / softmax / safe_softmax / embedding / masked_fill / conv1d / conv2d / conv_transpose1d）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(FUSED_CASES),
});

Deno.test({
  name: "双線形 resample（拡大 / 縮小 / scale 0）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(UPSAMPLE_CASES),
});

Deno.test({
  name: "DCNv2（k の 2 形 / バッチ 2 / 非対称形）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(DEFORM_CASES),
});

Deno.test({
  name: "GRU 隠れ側スキャン（2 方向 × バッチ 1 / 3）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(GRU_SCAN_CASES),
});

/**
 * `align_corners = True` の**定義そのもの**を実 GPU で固定する — 出力の 1 行目 / 最終行と
 * 1 列目 / 最終列が、入力の対応する端と**厳密に一致**する（λ が 0 か 1 に潰れるので丸めが
 * 入らない）。
 *
 * MUST: allclose ではなく厳密一致で見る。`align_corners = False` の座標式へ取り違えると
 * 端が半画素ずれるが、ずれ幅は許容差より小さいことがあり、checkAll では**両側が同じ誤りを
 * 共有していれば緑のまま**通る（CPU 参照も GPU もこのテストの対象）。ここは参照実装を
 * 経由せず、入力テンソルの端の値そのものと突き合わせる。
 */
Deno.test({
  name:
    "双線形 resample は (in−1)/(out−1) が f32 で往復する形で出力の端を入力の端へ厳密一致させる（align_corners — 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // H≠W / Hout≠Wout（軸の取り違えが端の値に出る）。値は全て相異なる列にする。
    const [heightIn, widthIn] = [4, 5];
    const [heightOut, widthOut] = [9, 7];
    const input = fill([1, 1, heightIn, widthIn], (i) => 1 + i * 0.5);
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "upsample_bilinear2d edges",
        op: "upsample_bilinear2d",
        inputs: [input],
        outShape: [1, 1, heightOut, widthOut],
        attrs: { output_size: [heightOut, widthOut] },
      });
      const at = (row: number, column: number): number => actual.data[row * widthOut + column];
      const src = (row: number, column: number): number => input.data[row * widthIn + column];
      assertEquals(at(0, 0), src(0, 0), "左上");
      assertEquals(at(0, widthOut - 1), src(0, widthIn - 1), "右上");
      assertEquals(at(heightOut - 1, 0), src(heightIn - 1, 0), "左下");
      assertEquals(at(heightOut - 1, widthOut - 1), src(heightIn - 1, widthIn - 1), "右下");
      // 中は補間される（端だけ見ると恒等コピーでも通ってしまう）。scale_h = 3/8 なので
      // 出力行 4 の源座標は 1.5 = 入力行 1 と 2 の中点 — 端の厳密一致と両立する内点。
      assertEquals(at(4, 0), Math.fround((src(1, 0) + src(2, 0)) / 2), "H の中点");
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * embedding の範囲外添字の裁定（src/kernels/embedding.ts）を実 GPU で固定する。gather と
 * 同じ非対称性 — GPU は範囲外の**行だけ**を NaN で汚染し、オラクルの CPU 参照は throw する。
 */
Deno.test({
  name: "embedding の範囲外添字は GPU で NaN 汚染・CPU 参照で throw（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const weight = fill([3, 2], POSITIVE);
    // 添字 3（語彙数超え）と -1（負）をそれぞれ 1 本ずつ混ぜ、残りは正しい行を引く
    const index = fill([4], (i) => [1, 3, -1, 0][i], "i32");
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "embedding oob",
        op: "embedding",
        inputs: [weight, index],
        outShape: [4, 2],
        attrs: { padding_idx: -1 },
      });
      const values = [...actual.data];
      assertEquals(
        values.map((v) => Number.isNaN(v)),
        [false, false, true, true, true, true, false, false],
      );
      // 契約内の添字は通常どおり自分の行を引く（汚染は範囲外の行だけ）
      assertEquals(values[0], weight.data[2]);
      assertEquals(values[1], weight.data[3]);
      assertEquals(values[6], weight.data[0]);
    } finally {
      gpu.destroy();
    }
    assertThrows(() => applyReferenceOp("embedding", [weight, index], { padding_idx: -1 }, [4, 2]));
  },
});

Deno.test({
  name: "境界（大きめ 1 本 / 行数が workgroup 上限超え）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(BOUNDARY_CASES),
});

/**
 * sym_prefix_slice（ADR 0010）— Tmax で焼いた定数の**先頭**を実行時の長さで切り出す。
 * 実行は strided 実体化コピーの流用（新カーネル無し）で、読み出し stride は入力（Tmax 形）
 * の連続 stride。
 *
 * MUST: **T < Tmax のケースだけを置く**。T = Tmax では「stride を束縛後の出力 shape から
 * 組む」誤りと正しい実装が同じ答えを出すため、その形は検出器にならない（波 2 の permute で
 * 対合の並べ替えが空振りしたのと同型の罠）。
 * MUST: 縮める軸は最終次元だけでなく**先行次元**も踏む。最終次元だけを縮める形は行の
 * 送り幅が変わらない特殊ケースで、stride の取り違えが値に出ない。
 */
const prefixSliceGraph = (
  constShape: readonly number[],
  outShape: readonly (number | string)[],
  slices: readonly { dim: number; coeff: number; offset: number }[],
  dtype: "f32" | "i32",
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["sym_prefix_slice"] },
  symbols: ["T"],
  // 束縛は入力 shape の次元位置からしか取れない（docs/ir-v1.md）ので、T を素の形で運ぶ
  // ダミー入力を 1 本置く。
  inputs: [{ name: "bind", dtype: "f32", shape: ["T"] }],
  outputs: ["y"],
  initializers: { table: { tensor: "table", storage: { dtype } } },
  values: {
    table: { dtype, shape: [...constShape] },
    y: { dtype, shape: [...outShape] },
  },
  nodes: [{
    op: "sym_prefix_slice",
    ins: ["table"],
    outs: ["y"],
    attrs: { sym: "T", slices: slices.map((slice) => ({ ...slice })) },
  }],
});

Deno.test({
  name: "sym_prefix_slice が Tmax 定数の先頭を切り出す（実 GPU / T < Tmax）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const bound = 3;
    const cases = [
      {
        name: "2 軸とも縮める（相対位置バケット表 i32 [6,5] → [3,3]）",
        constShape: [6, 5],
        outShape: ["T", "T"],
        resolved: [3, 3],
        slices: [{ dim: 0, coeff: 1, offset: 0 }, { dim: 1, coeff: 1, offset: 0 }],
        dtype: "i32" as const,
        generator: INTEGERS,
      },
      {
        name: "先行次元だけ縮める（位置テーブル f32 [7,4] → [3,4]）",
        constShape: [7, 4],
        outShape: ["T", 4],
        resolved: [3, 4],
        slices: [{ dim: 0, coeff: 1, offset: 0 }],
        dtype: "f32" as const,
        generator: SIGNED,
      },
      {
        name: "係数付き（f32 [16] → [2T+1] = [7]）",
        constShape: [16],
        outShape: ["2T+1"],
        resolved: [7],
        slices: [{ dim: 0, coeff: 2, offset: 1 }],
        dtype: "f32" as const,
        generator: POSITIVE,
      },
      {
        name: "rank4 の中間軸を縮める（i32 [2,6,3,4] → [2,3,3,4]）",
        constShape: [2, 6, 3, 4],
        outShape: [2, "T", 3, 4],
        resolved: [2, 3, 3, 4],
        slices: [{ dim: 1, coeff: 1, offset: 0 }],
        dtype: "i32" as const,
        generator: INTEGERS,
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const table = fill(testCase.constShape, testCase.generator, testCase.dtype);
        const graph = prefixSliceGraph(
          testCase.constShape,
          testCase.outShape,
          testCase.slices,
          testCase.dtype,
        );
        const buffer = graphModelBuffer(graph, [{
          name: "table",
          dtype: testCase.dtype === "i32" ? "I32" : "F32",
          shape: [...testCase.constShape],
          data: new Uint8Array(table.data.buffer, table.data.byteOffset, table.data.byteLength),
        }]);
        const session = await createSession(gpu, openModel(buffer));
        try {
          const outputs = await session.run({ bind: fill([bound], SIGNED) });
          // MUST: 期待 shape は**リテラル**（`resolved`）で持つ。実出力の shape を
          // applyReferenceOp に渡して同じものと突き合わせると、束縛から prefix 長を導く経路が
          // 丸ごと壊れていても恒真で通る。
          assertEquals(outputs["y"].shape, testCase.resolved, `${testCase.name}: 束縛後の shape`);
          const expected = applyReferenceOp(
            "sym_prefix_slice",
            [table],
            graph.nodes[0].attrs as Record<string, unknown>,
            testCase.resolved,
          );
          const report = compareTensors(outputs["y"], expected);
          assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 記号軸 cat（ADR 0046）— 連結軸が `T+定数` / `2T` に解決される形を実 GPU で閉じる。
 *
 * 緩めたのは**宣言レベルのガードだけ**で、束縛後は数値 shape なので strided 書きコピーの
 * カーネルは無変更のはず。「はず」を実測に置き換えるのがこのテストで、書き込み offset を
 * 静的軸の総和から組んでいる経路が残っていれば必ずずれる。
 *
 * MUST: 期待 shape は**リテラル**（`resolved`）で持つ。実出力の shape を参照側にも渡すと、
 * 束縛から連結軸長を導く経路が丸ごと壊れていても恒真で通る。
 */
Deno.test({
  name: "記号軸 cat（T+定数 / 2T）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const cases = [
      {
        name: "記号 + 定数 [1,T,4] + [1,3,4] dim=1（DiT の KV 連結と同型）",
        inShapes: [[1, "T", 4], [1, 3, 4]] as const,
        outShape: [1, "T+3", 4] as const,
        bound: 5,
        resolved: [1, 8, 4],
        dim: 1,
      },
      {
        name: "同一シンボルどうし [T,3] + [T,3] dim=0",
        inShapes: [["T", 3], ["T", 3]] as const,
        outShape: ["2T", 3] as const,
        bound: 4,
        resolved: [8, 3],
        dim: 0,
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const graph = singleOpGraph(
          "cat",
          testCase.inShapes.map((shape) => [...shape]),
          [...testCase.outShape],
          { symbols: ["T"], attrs: { dim: testCase.dim } },
        );
        // 束縛は入力 shape の次元位置から取る（'T' が素の形で現れる位置）。
        const inputs = testCase.inShapes.map((shape) =>
          fill(shape.map((dim) => (dim === "T" ? testCase.bound : dim as number)), SIGNED)
        );
        const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
        try {
          const outputs = await session.run(
            Object.fromEntries(inputs.map((input, index) => [`x${index}`, input])),
          );
          assertEquals(outputs["y"].shape, testCase.resolved, `${testCase.name}: 束縛後の shape`);
          const expected = applyReferenceOp(
            "cat",
            inputs,
            { dim: testCase.dim },
            testCase.resolved,
          );
          const report = compareTensors(outputs["y"], expected);
          assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
