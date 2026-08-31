// 実 GPU 実行 vs CPU 参照の数値突合（ADR 0005 の段 2）。M0 の全 op を代表 shape で回す。
// MUST: 乱数を使わない — 失敗が再現しないと原因の切り分けができない。

import { assertEquals, assertNotEquals, assertRejects, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { TOPK_WORKGROUP_SIZE } from "../src/kernels/topk.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import {
  applyReferenceOp,
  applyReferenceOpOutputs,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import {
  ARGMAX_CASES,
  ARGMAX_TIEBREAK_INPUT,
  ARGMAX_TIEBREAK_TORCH,
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
  TOPK_CASES,
  UNARY_CASES,
  UPSAMPLE_CASES,
} from "./helpers/gpu_op_cases.ts";
import { fill, graphModelBuffer, outputName, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  type RankMutation,
  type RankPipelineCache,
  runArgmaxDirect,
  runTopkDirect,
} from "./helpers/rank-dispatch.ts";
import { assertMutated } from "./helpers/state-dispatch.ts";

/**
 * 1 ケースを実 GPU で走らせ、**出力 slot 昇順の列**を返す（ADR 0068 決定 1）。単一出力 op では
 * 長さ 1 の列で、`topk` だけが 2 本になる。
 */
const runOutputs = async (gpu: GpuContext, testCase: OpCase): Promise<readonly Tensor[]> => {
  const graph = singleOpGraph(
    testCase.op,
    testCase.inputs.map((input) => input.shape),
    testCase.outShapes,
    {
      inDtypes: testCase.inputs.map((input) => input.dtype),
      outDtypes: testCase.outShapes.map((_, slot) =>
        testCase.outDtypes?.[slot] ?? testCase.inputs[0].dtype
      ),
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
    return testCase.outShapes.map((_, slot) => outputs[outputName(slot)]);
  } finally {
    await session.dispose();
  }
};

/** 単一出力ケースの実行（多出力 op は {@link runOutputs} で受ける）。 */
const runCase = async (gpu: GpuContext, testCase: OpCase): Promise<Tensor> => {
  const outputs = await runOutputs(gpu, testCase);
  assertEquals(outputs.length, 1, `${testCase.name}: 単一出力ケース`);
  return outputs[0];
};

const checkAll = async (cases: readonly OpCase[]): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    for (const testCase of cases) {
      const actual = await runOutputs(gpu, testCase);
      const expected = applyReferenceOpOutputs(
        testCase.op,
        testCase.inputs,
        testCase.attrs ?? {},
        testCase.outShapes[0],
      );
      // MUST: 本数から突き合わせる（多出力 op で片方を読み飛ばした形が緑になるのを防ぐ）
      assertEquals(actual.length, expected.length, `${testCase.name}: 出力本数`);
      actual.forEach((tensor, slot) => {
        const label = expected.length === 1 ? testCase.name : `${testCase.name} 出力 ${slot}`;
        assertEquals(tensor.shape, expected[slot].shape, label);
        assertEquals(tensor.dtype, expected[slot].dtype, `${label}: dtype`);
        // f32 は allclose、i32 / bool は厳密一致（整数演算に丸め差は無い — ADR 0009）
        const report = compareTensors(tensor, expected[slot]);
        assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
      });
    }
  } finally {
    gpu.destroy();
  }
};

Deno.test({
  name: "unary 11 種が CPU 参照と一致する（実 GPU）",
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
        outShapes: [[4]],
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
    outShapes: [[4]],
    attrs,
  })),
  {
    name: `${op} NaN 無し（対照）`,
    op,
    inputs: [fill([4], (i) => NAN_BASE[i])],
    outShapes: [[4]],
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
      outShapes: [[4]],
      attrs: { dim: 1 },
    },
    {
      // 行長 300 > workgroup サイズ 256 → 添字 299 は 1 スレッドの走査ループの **2 周目**で
      // 読まれる（[4,4] は 1 周目しか回らないので、この経路はここでしか踏まない）。
      // 行 1 は NaN 無しの対照。
      name: `${op} 走査ループ 2 周目の NaN [2,300]`,
      op,
      inputs: [fill([2, 300], (i) => (i === 299 ? Number.NaN : SIGNED(i)))],
      outShapes: [[2]],
      attrs: { dim: 1 },
    },
  ]),
];

/**
 * clamp / clamp_min / relu / amax / amin の NaN 伝播（ビット列判定の裁定を実 GPU で固定）。
 *
 * torch は 5 op とも「入力（縮約なら縮約対象の語群）に NaN があれば結果は NaN」。ドライバの
 * `max` / `min` は NaN を飲むので、伝播を担うのは生成 WGSL のビット列判定だけ
 * （機序は src/codegen/elementwise.ts の IS_NAN_BITS_WGSL）。
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
          testCase.outShapes[0],
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
        outShapes: [[smalls.length]],
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

/**
 * tanh / gelu_tanh の**飽和域**（src/codegen/elementwise.ts の TANH_STABLE_WGSL）。
 *
 * **この 2 本は「Metal でだけ赤 → 修正後に緑」になるカナリア**で、Vulkan / DXC のように
 * tanh を IEEE 忠実に実装するドライバでは打ち切りの前後どちらでも緑のまま（打ち切りは
 * 非 NaN の値をビット単位で変えない）。赤くなるのは tanh を `(exp(2x)−1)/(exp(2x)+1)` で
 * 計算する実装（Metal の fast-math 経路 — 実測）だけで、そこでは
 *   - `tanh` は |x| > 44.36
 *   - `gelu_tanh` は前活性 x ≳ 10.05（内側引数 √(2/π)·(x + 0.044715x³) が 44.36 を超える）
 * で exp(2x) が f32 のオーバーフローに入り `Inf/Inf` = 沈黙 NaN を返す。実モデル
 * （gemma4 E2B の活性は実測最大 11.45）が毎 token 踏んでいた穴がこれ。
 *
 * MUST: 入力に**飽和帯の内と外の両方**を含める。打ち切り閾値（9.5）の内側だけを見ると
 * 「打ち切りが効いていない実装」も緑で通る。
 * MUST: NaN 伝播を対で見る。打ち切りを `clamp` イディオムで書くと NaN が閾値に化けて
 * ±1.0 に飲まれる（機序は src/codegen/elementwise.ts の IS_NAN_BITS_WGSL）— そこを見張る 1 本。
 */
const TANH_SATURATION_INPUTS = [
  0,
  5,
  -5,
  9,
  -9,
  // 9.2 = 「打ち切らない（< 9.5）が f32 では既に 1.0（> 9.011）」の帯 — 閾値を 9.011 未満へ
  // 下げる退行と、飽和境界そのもののずれをここで検出する。9.5 は打ち切り境界ちょうど。
  9.2,
  -9.2,
  9.5,
  -9.5,
  10.6,
  -10.6,
  12,
  -12,
  50,
  -50,
  88,
  -88,
  120,
  -120,
] as const;

Deno.test({
  name: "tanh / gelu_tanh が飽和域で CPU 参照と一致する（NaN を吐かない — 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of ["tanh", "gelu_tanh"] as const) {
        const actual = await runCase(gpu, {
          name: `${op} 飽和域`,
          op,
          inputs: [fill([TANH_SATURATION_INPUTS.length], (i) => TANH_SATURATION_INPUTS[i])],
          outShapes: [[TANH_SATURATION_INPUTS.length]],
        });
        const expected = applyReferenceOp(
          op,
          [fill([TANH_SATURATION_INPUTS.length], (i) => TANH_SATURATION_INPUTS[i])],
          {},
          [TANH_SATURATION_INPUTS.length],
        );
        // MUST: NaN を先に落とす。下の許容差判定は NaN をどちらの側でも不合格にするが、
        // 「どの入力で NaN になったか」がメッセージに出ないと Metal の切り分けに使えない。
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          assertEquals(
            Number.isNaN(actual.data[index]),
            false,
            `${op}(${x}) が NaN（tanh の exp 経由実装が中間でオーバーフローしている）`,
          );
        });
        // 絶対 + 相対の合成で見る（gelu_tanh の負の飽和側は期待値が ±0 になり、相対だけでは
        // 判定できない）。WGSL は tanh の精度を保証しないので厳密一致は求めない。
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          const want = expected.data[index];
          const gap = Math.abs(actual.data[index] - want);
          assertEquals(
            gap <= 1e-6 * Math.max(1, Math.abs(want)),
            true,
            `${op}(${x}): GPU ${actual.data[index]} vs CPU 参照 ${want}`,
          );
        });
        // MUST: 飽和域は**厳密値**で固定する（打ち切りの中核主張 = 非 NaN の値をビット単位で
        // 変えない、の device 側の門）。上の 1e-6 帯は 2.5 ulp 低い 0.99999985 も通すため、
        // 「打ち切りが値を変えていない」ことはここでしか見えない。tanh は ±1.0、gelu_tanh は
        // 正側 = x そのもの・負側 = **−0.0**（0.5·x·(1 + (−1.0)) の符号付きゼロ）。
        // NOTE: WGSL は tanh に絶対誤差 1e-5 を許すので、仕様適合のままこの門が赤くなる
        // ドライバは理論上ありうる — そのときはビット同一 golden 群も同時に割れているはずで、
        // この門は「その事実を単体テストで最初に知る」カナリア（機で緩めない）。
        TANH_SATURATION_INPUTS.forEach((x, index) => {
          const v = actual.data[index];
          if (op === "tanh" && Math.abs(x) >= 9.2) {
            assertEquals(
              Object.is(v, Math.sign(x)),
              true,
              `tanh(${x}) = ${v}（飽和域が厳密な ±1.0 でない）`,
            );
          }
          if (op === "gelu_tanh" && x >= 9) {
            assertEquals(
              Object.is(v, Math.fround(x)),
              true,
              `gelu_tanh(${x}) = ${v}（正の飽和域が x 厳密でない）`,
            );
          }
          if (op === "gelu_tanh" && x <= -9) {
            assertEquals(
              Object.is(v, -0),
              true,
              `gelu_tanh(${x}) = ${v}（負の飽和域が −0.0 厳密でない）`,
            );
          }
        });
        // MUST: 恒真化しない。飽和域の外（x = ±5）が定数へ潰れていないことを見る
        // （「全部 ±1 を返す」実装は上の 2 つの判定だけでは落ちない）。
        assertNotEquals(actual.data[1], actual.data[3], `${op}: x=5 と x=9 が同値`);
      }

      // ±Inf（打ち切りが副次的に塞いだ穴 — 素の exp 経由実装では tanh(±Inf) = Inf/Inf = NaN）。
      // gelu_tanh(−Inf) は NaN が正: 0.5·(−Inf)·0.0 の 0·Inf で、torch も NaN
      //（F.gelu(-inf) 実測 torch 2.13.0 — approximate 両方）。CPU 参照とも三者一致。
      const infInput = fill([2], (i) => i === 0 ? Infinity : -Infinity);
      const tanhInf = await runCase(gpu, {
        name: "tanh ±Inf",
        op: "tanh",
        inputs: [infInput],
        outShapes: [[2]],
      });
      assertEquals(Object.is(tanhInf.data[0], 1), true, `tanh(+Inf) = ${tanhInf.data[0]}`);
      assertEquals(Object.is(tanhInf.data[1], -1), true, `tanh(-Inf) = ${tanhInf.data[1]}`);
      const geluInf = await runCase(gpu, {
        name: "gelu_tanh ±Inf",
        op: "gelu_tanh",
        inputs: [infInput],
        outShapes: [[2]],
      });
      assertEquals(geluInf.data[0], Infinity, `gelu_tanh(+Inf) = ${geluInf.data[0]}`);
      assertEquals(
        Number.isNaN(geluInf.data[1]),
        true,
        `gelu_tanh(-Inf) = ${geluInf.data[1]}（torch と同じ NaN のはず）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 上のカナリアの NaN 側。`tanh` はビット列判定の外殻、`gelu_tanh` は式の外に残る因子 `x` で
 * それぞれ NaN を運ぶ（機序が違うので op ごとに見る）。
 */
Deno.test({
  name: "tanh / gelu_tanh が NaN を伝播する（打ち切りが NaN を飲まない — 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of ["tanh", "gelu_tanh"] as const) {
        // 飽和帯の内（0.5）・外（12）と NaN を混ぜる。対照が無いと「常に NaN」で緑になる。
        const inputs = [Number.NaN, 0.5, 12, Number.NaN] as const;
        const actual = await runCase(gpu, {
          name: `${op} NaN`,
          op,
          inputs: [fill([4], (i) => inputs[i])],
          outShapes: [[4]],
        });
        assertEquals(
          [...actual.data].map((value) => Number.isNaN(value)),
          [true, false, false, true],
          `${op}: NaN の位置`,
        );
      }
    } finally {
      gpu.destroy();
    }
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
        outShapes: [[2, 3]],
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
  name: "argmax（最終次元・rank 保存）が CPU 参照と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(ARGMAX_CASES),
});

/**
 * argmax のタイブレーク直接門（ADR 0068 決定 2 の 3 つの MUST を**期待値リテラル**で固定）。
 *
 * MUST: CPU 参照との突合（checkAll）では代替できない。両側が同じ向きに間違えれば緑になる
 * 軸（タイブレークの向き・NaN の扱い・全 −inf 行の答え）なので、torch の実測値そのものを
 * リテラルで置く（表と行の意味は `helpers/gpu_op_cases.ts` の `ARGMAX_TIEBREAK_INPUT`）。
 * ここは **GPU の出力**だけを見る — 同じ表に対する CPU 参照の向きは、アダプタ無しの環境でも
 * 走る `reference_ops_test.ts` 側の門が縛る。
 */
Deno.test({
  name: "argmax のタイブレーク / NaN / 全 −inf 行が torch の実測値と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const actual = await runCase(gpu, {
        name: "argmax tie-break",
        op: "argmax",
        inputs: [ARGMAX_TIEBREAK_INPUT],
        outShapes: [[ARGMAX_TIEBREAK_TORCH.length, 1]],
        outDtypes: ["i32"],
      });
      assertEquals(
        [...actual.data],
        [...ARGMAX_TIEBREAK_TORCH],
        "GPU の argmax が torch の実測値と違う",
      );
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "topk（最終次元・static-k・値 + 添字の 2 出力）が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => checkAll(TOPK_CASES),
});

/** NaN を "NaN" に潰した表示形（allclose も assertEquals も NaN 同士を一致と見ないため）。 */
const nanSafe = (values: Iterable<number>): readonly (number | "NaN")[] =>
  [...values].map((value) => (Number.isNaN(value) ? "NaN" : value));

/**
 * topk の同値 tie / NaN / 全 −inf 行（ADR 0068 決定 3 の固定挙動を**期待値リテラル**で固定）。
 *
 * MUST: CPU 参照との突合（checkAll）では代替できない。両側が同じ向きに間違えれば緑になる軸
 * （タイブレークの向き・NaN の扱い・全 −inf 行の答え）なので、リテラルで置く。
 *
 * **torch との関係（実測 2026-08-17 / torch 2.13.0+cpu）**:
 *
 * - **値の列は torch とビット一致**（`VALUES` は torch の実測値そのもの）。降順で同値の
 *   多重度も同じなので、重複だらけの行でも一致する（`torch.topk` の values 列を
 *   (値降順, index昇順) 実装と 200×4 ケースで突合した結果も全一致）。
 * - **添字の列は torch と一致しない**（`INDICES` は karume の規定）。torch の topk は同値
 *   要素の順序を保証せず、`[5,5,5,5]` の k=1 で index **2** を返すのに `argmax` は **0** を
 *   返す（同一リポ内の自己矛盾）。karume は argmax と同族の**最小 index 優先**に規定した。
 *   下表で torch と割れるのは行 1 / 2 / 4 / 5 / 6（torch は順に [2,3,0,1] / [2,3,0,1] /
 *   [2,3,0,1] / [1,3,2,0] / [0,2,3,1]）。
 *
 * 行の意味:
 * 0. 同値が 2 つ（3.0 が index 1,2）→ 最小 index が先（torch も同じ）
 * 1. 全要素が同値 → index 昇順（「最後の最大値」実装なら降順になる）
 * 2. 全要素 −inf → index 昇順（有限 sentinel の identity だと番兵 index が漏れる）
 * 3. NaN が 2 つ（index 1,3）→ **NaN は最大**・NaN 同士は最小 index（torch も同じ）
 * 4. 全要素 NaN → index 昇順
 * 5. −inf と同値の最大が混在 → 2.0 が先・−inf は index 昇順
 * 6. 同値が 2 組 → 2.0 の組が先、その中は index 昇順
 */
Deno.test({
  name: "topk の同値 tie / NaN / 全 −inf 行が固定挙動どおり（値は torch とビット一致 — 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const NEG = Number.NEGATIVE_INFINITY;
    const NAN = Number.NaN;
    const rows = [
      [1, 3, 3, 2],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [1, NAN, 3, NAN],
      [NAN, NAN, NAN, NAN],
      [NEG, 2, NEG, 2],
      [2, 1, 2, 1],
    ];
    // torch 実測の values 列（k = 4 = 最終次元 = 行全体の降順）
    const VALUES = [
      [3, 3, 2, 1],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [NAN, NAN, 3, 1],
      [NAN, NAN, NAN, NAN],
      [2, 2, NEG, NEG],
      [2, 2, 1, 1],
    ];
    // karume の規定（最小 index 優先 — torch は同値の順序を保証しない）
    const INDICES = [
      [1, 2, 3, 0],
      [0, 1, 2, 3],
      [0, 1, 2, 3],
      [1, 3, 2, 0],
      [0, 1, 2, 3],
      [1, 3, 0, 2],
      [0, 2, 1, 3],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const testCase: OpCase = {
      name: "topk tie-break",
      op: "topk",
      inputs: [input],
      outShapes: [[rows.length, 4], [rows.length, 4]],
      outDtypes: ["f32", "i32"],
      attrs: { k: 4 },
    };
    const gpu = await acquireGpu();
    try {
      const actual = await runOutputs(gpu, testCase);
      assertEquals(nanSafe(actual[0].data), nanSafe(VALUES.flat()), "GPU の topk 値列");
      assertEquals([...actual[1].data], INDICES.flat(), "GPU の topk 添字列");
    } finally {
      gpu.destroy();
    }
    // 同じ表を CPU 参照にも当てて、オラクル側の向きも同時に固定する
    const expected = applyReferenceOpOutputs("topk", [input], { k: 4 });
    assertEquals(nanSafe(expected[0].data), nanSafe(VALUES.flat()), "CPU 参照の topk 値列");
    assertEquals([...expected[1].data], INDICES.flat(), "CPU 参照の topk 添字列");
  },
});

/**
 * k=1 の topk が argmax と一致する（族間の食い違い検出）。
 *
 * MUST: 同じ入力で 2 つの op を実走させて突き合わせる。両者はカーネルもキーも別なので、
 * 述語の本文を片方だけ書き換えても shape も dtype も合ったまま答えだけが割れる。
 * 入力は**タイと NaN と全 −inf 行**を含む形（一意な最大値だけの入力では、どちらの
 * タイブレーク規律でも同じ答えになって検出器にならない）。
 */
Deno.test({
  name: "k=1 の topk の添字が argmax と一致する（タイ / NaN / 全 −inf 行込み・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const NEG = Number.NEGATIVE_INFINITY;
    const NAN = Number.NaN;
    const rows = [
      [1, 3, 3, 2],
      [5, 5, 5, 5],
      [NEG, NEG, NEG, NEG],
      [1, NAN, 3, NAN],
      [NEG, 2, NEG, 2],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const gpu = await acquireGpu();
    try {
      const topk = await runOutputs(gpu, {
        name: "topk k=1",
        op: "topk",
        inputs: [input],
        outShapes: [[rows.length, 1], [rows.length, 1]],
        outDtypes: ["f32", "i32"],
        attrs: { k: 1 },
      });
      const argmax = await runCase(gpu, {
        name: "argmax",
        op: "argmax",
        inputs: [input],
        outShapes: [[rows.length, 1]],
        outDtypes: ["i32"],
      });
      assertEquals([...topk[1].data], [...argmax.data], "k=1 の topk が argmax と食い違う");
      // 値の側も amax と同じ要素を返している（添字だけ合っていて値が別の要素という形を潰す）
      assertEquals(
        nanSafe(topk[0].data),
        nanSafe([...argmax.data].map((at, row) => rows[row][at])),
        "k=1 の topk 値が添字の指す要素と違う",
      );
    } finally {
      gpu.destroy();
    }
  },
});

/** f32 テンソルの**ビット列**（±0.0 は数値としては同値なので、符号ビットはここでしか見えない）。 */
const f32Bits = (tensor: Tensor | RefTensor): readonly number[] => {
  if (tensor.dtype !== "f32") throw new Error(`f32 でない dtype: ${tensor.dtype}`);
  return [...new Uint32Array(tensor.data.slice().buffer)];
};

/**
 * `+0.0` と `-0.0` が混じる tie 行（**選ばれた添字の要素のビットがそのまま出る**こと）。
 *
 * MUST: 期待値は **karume の規範**（同値類は最小 index 優先・値は選ばれた要素のビットそのもの）
 * で、torch 突合ではない。`±0.0` は比較上同値なので、どちらの符号を書いても数値の突合
 * （allclose も assertEquals も `-0 === +0`）は緑になる — 符号ビットは **Uint32 のビット列**
 * でしか見えない。ここが緩むと、比較を `max` イディオムへ畳んだ実装（ドライバの
 * `max(-0.0, +0.0)` は `+0.0` を返しうる）や、値を添字から引き直さず正規化して書く実装が
 * 黙って通る。
 *
 * NOTE: NaN payload の差（同じ NaN でも仮数部のビットが違う形）は対象外 — WGSL / ドライバは
 * payload の保存を保証しないので、規範として固定できない。
 *
 * 行の意味:
 * 0. 全要素が ±0.0 の tie → 添字昇順・値は各添字の符号がそのまま
 * 1. 有限値 1 本 + ±0.0 の tie → 1.0 が先頭、零は添字昇順
 * 2. ±0.0 の tie + 負値 → 零 3 本が先（添字昇順）、−1.0 が最後
 */
Deno.test({
  name: "topk は ±0.0 の tie でも選択添字どおりのビットを書く（値をビット列で固定・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const rows = [
      [-0, 0, -0, 0],
      [0, -0, 1, -0],
      [-0, 0, -1, -0],
    ];
    // 値のビット列（0x80000000 = −0.0 / 0x00000000 = +0.0 / 0x3f800000 = 1.0 / 0xbf800000 = −1.0）
    const VALUE_BITS = [
      [0x80000000, 0x00000000, 0x80000000, 0x00000000],
      [0x3f800000, 0x00000000, 0x80000000, 0x80000000],
      [0x80000000, 0x00000000, 0x80000000, 0xbf800000],
    ];
    const INDICES = [
      [0, 1, 2, 3],
      [2, 0, 1, 3],
      [0, 1, 3, 2],
    ];
    const input = fill([rows.length, 4], (i) => rows[Math.floor(i / 4)][i % 4]);
    const testCase: OpCase = {
      name: "topk signed-zero tie",
      op: "topk",
      inputs: [input],
      outShapes: [[rows.length, 4], [rows.length, 4]],
      outDtypes: ["f32", "i32"],
      attrs: { k: 4 },
    };
    const gpu = await acquireGpu();
    try {
      const actual = await runOutputs(gpu, testCase);
      assertEquals(f32Bits(actual[0]), VALUE_BITS.flat(), "GPU の topk 値ビット列");
      assertEquals([...actual[1].data], INDICES.flat(), "GPU の topk 添字列");
    } finally {
      gpu.destroy();
    }
    // 同じ表を CPU 参照にも当てて、オラクル側の符号も同時に固定する
    const expected = applyReferenceOpOutputs("topk", [input], { k: 4 });
    assertEquals(f32Bits(expected[0]), VALUE_BITS.flat(), "CPU 参照の topk 値ビット列");
    assertEquals([...expected[1].data], INDICES.flat(), "CPU 参照の topk 添字列");
  },
});

/**
 * topk の **k の実装上限**（ADR 0068 決定 3 — 縮退しない・上限値つきで fail loudly）。
 *
 * MUST: **絞った device**（`LIMIT_CAPS`）で見る。手元のアダプタは workgroup storage を
 * 49152 バイト出すので上限 k は 191 になり、既定の機（16384 → 上限 63）で落ちる k を
 * 実走で確かめる手段が他に無い（列挙は「動く」の証拠にならない、と同じ規律）。
 * 上限の内側（k=2）が**同じ絞った device で緑**であることも対で見る — 全部落ちる実装でも
 * 拒否側の門だけなら通ってしまう。
 */
Deno.test({
  name:
    "topk は workgroup storage 由来の k 上限を超えたら上限値つきで落ちる（絞った device / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 8·W·(k+1) ≤ 1024 → 上限 k = 3
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupStorageSize: 1024 } });
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupStorageSize, 1024);
      const input = fill([2, 8], SIGNED);
      const outputs = await runOutputs(gpu, {
        name: "topk k=2（上限の内側）",
        op: "topk",
        inputs: [input],
        outShapes: [[2, 2], [2, 2]],
        outDtypes: ["f32", "i32"],
        attrs: { k: 2 },
      });
      const expected = applyReferenceOpOutputs("topk", [input], { k: 2 });
      assertEquals([...outputs[0].data], [...expected[0].data]);
      assertEquals([...outputs[1].data], [...expected[1].data]);
      // 上限超過は縮退せず落ちる（診断に上限値 3 が出る）
      const graph = singleOpGraph("topk", [[2, 8]], [[2, 4], [2, 4]], {
        outDtypes: ["f32", "i32"],
        attrs: { k: 4 },
      });
      const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
      try {
        await assertRejects(
          () => session.run({ x0: input }),
          CodegenError,
          "実装上限 3 を超える",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 故障注入（ADR 0068 受入条件③）— argmax / topk の門が**実際に検出器**であることの実証。
 *
 * MUST: Session 経由（{@link runOutputs}）では撃てない。`PipelineCache` は**キー**でキャッシュ
 * するので、変異版 WGSL を流し込む口が無い（口を作っても正常版のパイプラインが配られる）。
 * ここは `helpers/rank-dispatch.ts` の直接 dispatch で、**最終 WGSL 文字列**をキーにした
 * キャッシュを使う（states 形の `helpers/state-dispatch.ts` と同じ流儀）。
 *
 * MUST: 入力は**両系統**を用意する。P2E の実測（`.claude/reviews/2026-08-19_b04f589/findings/
 * P2E-topk-merge-fuzz.md` §3・18,000 ケース）が示すとおり、topk の変異は検出できる入力の族が
 * 相補的で、片方だけでは変異が**常に緑**になる:
 *
 * - **単一レーン入力**（上位 k 本が全て `i ≡ l (mod W)`）でしか③（レーン局所挿入の末尾比較）は
 *   出ない — 上位 k 本が 1 レーンに載らないと局所リストが溢れず、末尾比較そのものが働かない。
 * - **分散入力**（同値がレーンに散る）でしか①②（カーソル前進・tie 分岐）は出ない — 単一レーンが
 *   全ラウンド勝つ入力ではカーソル前進の分岐が無害化し、tie も起きない。
 *
 * 本テストでもその相補性を再現した（2026-08-19 実測 / RTX 3080 Ti）: ①を単一レーン入力へ、
 * ③を分散入力へ当てるとどちらも緑（= 検出できず）になる。
 *
 * NOTE: 検討したが**入れなかった**変異が 2 本ある（恒真テストを置かないため。同上の実測）:
 *
 * - **topk の最終 `workgroupBarrier()` 削除**: rows=256・dim=320・k=16 の 20 回試行で
 *   mismatch 0。W=32 は NVIDIA の warp 幅そのもので workgroup が lockstep 実行されるため、
 *   `won` の読みと次ラウンドの `head` 上書きが競合しない。P2E §4-1 の「逐次シミュレートでは
 *   原理的に検出不能」は、この機では**実 GPU でも同じ**という実測。
 * - **argmax の番兵 `dim` → `0`**: 番兵が答えに効くのは行の最大値が −inf のとき（= 全要素が
 *   −inf）だけで、その行の正解添字は 0 なので**変異版も同じ 0** を返す。原理的に値へ出ない。
 *   代わりに同じ MUST（有限 sentinel 禁止 — src/kernels/argmax.ts）を撃つ**番兵値**の変異
 *   （`neg_inf` → `0.0`）を置いた。こちらは全負値の行で番兵 index `dim` が出力へ漏れる。
 */
type RankInjection = {
  readonly label: string;
  readonly mutate: RankMutation;
};

/** WGSL の該当箇所を書き換える（空振り = 文言変更で置換対象が消えた形は `assertMutated`）。 */
const injectRank = (from: string, to: string): RankMutation => (wgsl) => {
  const mutated = wgsl.replaceAll(from, to);
  assertMutated(wgsl, mutated, from);
  return mutated;
};

/** topk の故障注入 1 件（どの入力系統で撃つかまで含めて 1 件 — 上の相補性）。 */
type TopkInjection = RankInjection & { readonly input: TopkFaultInput };

type TopkFaultInput = {
  readonly rows: number;
  readonly dim: number;
  readonly k: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * 分散入力（`W = 32` の 3 周ぶん）。行 0 = 全要素同値、行 1 = 8 要素ずつの同値ブロックが
 * レーン境界を跨ぐ形。どちらも「同値の集合から k 本を最小 index 順で取る」ので、カーソル前進を
 * 1 レーンでも余計に進めると答えが飛ぶ。
 */
const TOPK_SPREAD_INPUT: TopkFaultInput = {
  rows: 2,
  dim: 96,
  k: 8,
  data: Float32Array.from(
    Array.from({ length: 2 * 96 }, (_, i) => (i < 96 ? 7.5 : -Math.floor((i - 96) / 8))),
  ),
};

/**
 * 単一レーン入力（上位 5 本が全て `i ≡ 7 (mod 32)` = レーン 7 の担当）。k=3 なのでレーン 7 の
 * 局所リストは 5 候補で 2 回溢れ、末尾比較が実際に候補を捨てる経路に乗る。
 */
const TOPK_SINGLE_LANE_INPUT: TopkFaultInput = {
  rows: 1,
  dim: 160,
  k: 3,
  data: Float32Array.from(
    Array.from({ length: 160 }, (_, i) => (i % 32 === 7 ? 9 - (i - 7) / 32 : -1)),
  ),
};

const TOPK_INJECTIONS: readonly TopkInjection[] = [
  // ① 勝った要素の持ち主だけが進むカーソルを**全レーン前進**にする（merge の中核）
  {
    label: "カーソル前進条件を恒真にする",
    input: TOPK_SPREAD_INPUT,
    mutate: injectRank(`won % ${TOPK_WORKGROUP_SIZE}u == lid`, "true"),
  },
  // ② 勝者述語の tie 分岐を反転（最小 index 規範 → 最大 index）
  {
    label: "tie 分岐を ib > ia へ反転",
    input: TOPK_SPREAD_INPUT,
    mutate: injectRank("(vb == va && ib < ia)", "(vb == va && ib > ia)"),
  },
  // ③ レーン局所挿入の末尾（最弱）比較を 1 つ手前へずらす（k 本目が入らなくなる）
  {
    label: "レーン局所挿入の末尾比較を 1 ずらす",
    input: TOPK_SINGLE_LANE_INPUT,
    mutate: injectRank(
      `block + ${TOPK_SINGLE_LANE_INPUT.k - 1}u`,
      `block + ${TOPK_SINGLE_LANE_INPUT.k - 2}u`,
    ),
  },
];

/**
 * argmax の故障注入用の行。0 = NaN 混在（NaN 分岐の検出線）/ 1 = 全負値（番兵値の検出線 —
 * 有限 sentinel なら番兵 index `dim` が漏れる）/ 2 = 全 −inf（identity の順位が効く行）。
 * `dim = 8 < 256` なので**担当要素を持たないレーン**が居る = 番兵がそのまま木へ入る。
 */
const ARGMAX_FAULT_INPUT = {
  rows: 3,
  dim: 8,
  data: Float32Array.from([
    1,
    Number.NaN,
    3,
    Number.NaN,
    2,
    0,
    -1,
    5,
    -5,
    -2,
    -9,
    -2,
    -7,
    -3,
    -4,
    -8,
    ...Array.from({ length: 8 }, () => Number.NEGATIVE_INFINITY),
  ]),
};

const ARGMAX_INJECTIONS: readonly RankInjection[] = [
  // ① identity の**値**を有限 sentinel にする（番兵 index が [0, dim) の外へ漏れる）
  {
    label: "行 max の identity を有限 sentinel にする",
    mutate: injectRank("var best = neg_inf;", "var best = 0.0;"),
  },
  // ② 述語から NaN 分岐を落とす（NaN は比較で全て false になり黙って負ける）
  {
    label: "argmax_beats の NaN 分岐を削除",
    mutate: injectRank(
      [
        "  let na = is_nan_bits(va);",
        "  let nb = is_nan_bits(vb);",
        "  if (na != nb) {",
        "    return nb;",
        "  }",
        "  if (na) {",
        "    return ib < ia;",
        "  }",
        "",
      ].join("\n"),
      "",
    ),
  },
];

/** topk の 2 出力を 1 つの比較対象へ畳む（片方だけ見る形を潰す）。 */
const topkPair = (
  result: { readonly values: Float32Array; readonly indices: Int32Array },
): readonly (readonly number[])[] => [[...result.values], [...result.indices]];

const referenceTopkPair = (input: TopkFaultInput): readonly (readonly number[])[] => {
  const expected = applyReferenceOpOutputs(
    "topk",
    [refTensor([input.rows, input.dim], input.data)],
    { k: input.k },
  );
  return [[...expected[0].data], [...expected[1].data]];
};

Deno.test({
  name: "故障注入: argmax / topk の変異版はいずれも参照突合で赤くなる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache: RankPipelineCache = new Map();
    try {
      // 正常版が期待値と一致することが前提（ここが赤なら「変異版が不一致」は何の証拠にもならない）
      for (const input of [TOPK_SPREAD_INPUT, TOPK_SINGLE_LANE_INPUT]) {
        const actual = await runTopkDirect(gpu.device, input, input.data, { cache });
        assertEquals(
          topkPair(actual),
          referenceTopkPair(input),
          `正常版の topk（dim=${input.dim} k=${input.k}）が CPU 参照と違う`,
        );
      }
      const argmaxExpected = [
        ...applyReferenceOp(
          "argmax",
          [refTensor([ARGMAX_FAULT_INPUT.rows, ARGMAX_FAULT_INPUT.dim], ARGMAX_FAULT_INPUT.data)],
          {},
          [ARGMAX_FAULT_INPUT.rows, 1],
        ).data,
      ];
      assertEquals(
        [
          ...await runArgmaxDirect(gpu.device, ARGMAX_FAULT_INPUT, ARGMAX_FAULT_INPUT.data, {
            cache,
          }),
        ],
        argmaxExpected,
        "正常版の argmax が CPU 参照と違う",
      );
      for (const injection of TOPK_INJECTIONS) {
        const input = injection.input;
        const actual = await runTopkDirect(gpu.device, input, input.data, {
          cache,
          mutate: injection.mutate,
        });
        assertNotEquals(
          topkPair(actual),
          referenceTopkPair(input),
          `topk の故障注入 '${injection.label}' が検出されなかった`,
        );
      }
      for (const injection of ARGMAX_INJECTIONS) {
        const actual = await runArgmaxDirect(
          gpu.device,
          ARGMAX_FAULT_INPUT,
          ARGMAX_FAULT_INPUT.data,
          { cache, mutate: injection.mutate },
        );
        assertNotEquals(
          [...actual],
          argmaxExpected,
          `argmax の故障注入 '${injection.label}' が検出されなかった`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
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
        outShapes: [[1, 1, heightOut, widthOut]],
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
        outShapes: [[4, 2]],
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
          [[...testCase.outShape]],
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
