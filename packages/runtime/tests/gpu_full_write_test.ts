// full-write 不変条件（ADR 0014 / ADR 0004 のアリーナ不変条件⑤）のフォールト注入。
//
// 「全カーネルは出力バッファの全バイトを書く」を、**プール再利用バッファに毒値を仕込んでから**
// cat / pad / flip / conv_transpose1d を実行して実証する。バッファプールの再利用バッファは
// ゼロ初期化されない（ゼロ保証は新規 createBuffer のみ — src/gpu/arena.ts）ので、書き漏らした
// 要素があれば毒値がそのまま出力に残る。
//
// 仕込み方: 先行ノード（恒等 cast）の出力を**毒値そのもの**にし、そのノードの出力が消費者
// ゼロでノード境界でプールへ戻る形にする。次のノードの出力確保は同じサイズクラスなので
// プールから同じバッファが配られる（reuseCount で実際に再利用されたことを確かめる — 再利用が
// 起きていなければ検査は何も見ていない）。
//
// MUST: 毒値は「ありえない値」ではなく **0 でない有限値**にする。ゼロ埋めの取り違えを見たいので、
// 0 を毒値にすると pad のゼロ領域と区別できない。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type AcquireGpuOptions } from "../src/gpu/device.ts";
import { referenceLinearI8a8 } from "../src/reference/i8a8.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

/** 毒値 0xDEADBEEF を f32 として読んだもの（-6.259854e18 — 0 でない有限値）。 */
const POISON = new Float32Array(new Uint32Array([0xDEADBEEF]).buffer)[0];

/**
 * 「毒値を作る恒等 cast」+ 「検査対象ノード」の 2 ノードグラフ。
 *
 * `seed` は毒値で埋めたグラフ入力で、cast(f32 → f32) の出力 `poison` は**誰にも消費されない**
 * ため、ノード境界でそのままプールへ戻る。`poison` の要素数を検査対象の出力と揃えてあるので、
 * 次の確保で同じバッファが配り直される。
 */
const poisonGraph = (
  outShape: readonly number[],
  operands: readonly { readonly name: string; readonly shape: readonly number[] }[],
  node: { readonly op: string; readonly attrs: Record<string, unknown> },
): GraphJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["cast", node.op] },
    symbols: [],
    inputs: [
      { name: "seed", dtype: "f32", shape: [count] },
      ...operands.map((operand) => ({
        name: operand.name,
        dtype: "f32",
        shape: [...operand.shape],
      })),
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      poison: { dtype: "f32", shape: [count] },
      y: { dtype: "f32", shape: [...outShape] },
    },
    nodes: [
      { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
      { op: node.op, ins: operands.map((operand) => operand.name), outs: ["y"], attrs: node.attrs },
    ],
  };
};

const runPoisoned = async (
  graph: GraphJson,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
  acquire: AcquireGpuOptions = {},
): Promise<{ readonly output: Tensor; readonly reuseCount: number }> => {
  const gpu = await acquireGpu(acquire);
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    const count = graph.inputs[0].shape[0] as number;
    const outputs = await session.run({ seed: fill([count], () => POISON), ...inputs });
    return { output: outputs["y"], reuseCount: session.diagnostics().lastRun?.reuseCount ?? 0 };
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

Deno.test({
  name: "cat はプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // [2,1] + [2,2] → [2,3]。毒値バッファは 6 要素で、cat の出力とサイズクラスが一致する。
    const graph = poisonGraph(
      [2, 3],
      [{ name: "a", shape: [2, 1] }, { name: "b", shape: [2, 2] }],
      {
        op: "cat",
        attrs: { dim: 1 },
      },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      a: fill([2, 1], (i) => 10 + i),
      b: fill([2, 2], (i) => 20 + i),
    });
    // 再利用が起きていなければ毒値検査は何も見ていない（恒真化の門）
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data], [10, 20, 21, 11, 22, 23]);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});

Deno.test({
  name: "pad のゼロ領域は毒値ではなく厳密に 0 になる（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // [2,2] を [1,1] で埋める → [2,4]。ゼロ領域が毒値のまま残れば即座に赤くなる。
    const graph = poisonGraph([2, 4], [{ name: "x", shape: [2, 2] }], {
      op: "pad",
      attrs: { left: 1, right: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, { x: fill([2, 2], (i) => 1 + i) });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data], [0, 1, 2, 0, 0, 3, 4, 0]);
    // MUST: ゼロ領域は「毒値でない」だけでなく**厳密に 0**（符号付きゼロも +0）
    for (const index of [0, 3, 4, 7]) {
      assertEquals(Object.is(output.data[index], 0), true, `要素 ${index} が +0 でない`);
    }
  },
});

Deno.test({
  name: "flip はプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // flip は**全単射**（出力 1 要素 ← 入力 1 要素）なので、書き漏らしが起こるとすれば
    // grid-stride の打ち切りか添字の飛びで、どちらも毒値の残存として出る。
    // [2,3] の軸 1 を反転 → [2,3]（毒値バッファは 6 要素で同サイズクラス）。
    const graph = poisonGraph([2, 3], [{ name: "x", shape: [2, 3] }], {
      op: "flip",
      attrs: { dim: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, { x: fill([2, 3], (i) => 1 + i) });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data], [3, 2, 1, 6, 5, 4]);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});

Deno.test({
  name: "conv2d 直接カーネル（groups>1）は毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 出力の全バイトを 1 スレッド 1 要素で書く形（ADR 0017）。grid-stride の打ち切りや
    // 平坦添字の分解（B / Cout / H / W の 4 段）の誤りは、書かれない要素 = 毒値の残存に出る。
    // MUST: **groups > 1**（ADR 0024 の踏み分けで直接カーネルへ流れる側）。groups == 1 は
    // implicit GEMM へ行くので、この 1 本だけでは直接カーネルを 1 度も踏まなくなる。
    // x[1,2,3,4] * W[4,1,2,3] groups=2 pad=[1,1] → [1,4,4,4]（毒値バッファは 64 要素）。
    const graph = poisonGraph(
      [1, 4, 4, 4],
      [
        { name: "x", shape: [1, 2, 3, 4] },
        { name: "w", shape: [4, 1, 2, 3] },
        { name: "b", shape: [4] },
      ],
      {
        op: "conv2d",
        attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 2 },
      },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([1, 2, 3, 4], (i) => 1 + i),
      w: fill([4, 1, 2, 3], (i) => 0.25 + i * 0.5),
      b: fill([4], () => 0),
    });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals(output.shape, [1, 4, 4, 4]);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});

/**
 * conv2d の implicit GEMM（`groups == 1` — ADR 0024）は GEMM 骨格と同じタイル書き出しなので、
 * 安全網は `tiledWorkgroups` の fail loudly とここの毒値注入の 2 本しかない。
 *
 * MUST: **v4 経路とスカラ経路を対で持つ**（書き出しのガードが変種ごとに別の式）。
 * MUST: 形は 64 を跨がせる（1 タイル未満に潰れるとガードが一度も偽にならない）。
 * MUST: バッチ 2 のケースを持つ — バッチは dispatch の **z 軸**なので、z を 1 に固定する
 * 誤りは B = 1 のケースでは緑のまま通り、B ≥ 2 で「2 枚目が丸ごと未書き込み」になる。
 */
const CONV2D_IGEMM_POISON_CASES: readonly {
  readonly name: string;
  readonly outShape: readonly number[];
  readonly x: readonly number[];
  readonly w: readonly number[];
  readonly attrs: Record<string, unknown>;
}[] = [
  {
    // v4（kFlat = 4·1·1 = 4 / Wout = 8 / stride_w = 1）。m = 70 で行タイル 2 枚、
    // n = Hout·Wout = 72 で列タイル 2 枚（最終タイルの有効 quad は 16 中 2）。
    name: "conv2d igemm v4 [1,4,9,8] * W[70,4,1,1]",
    outShape: [1, 70, 9, 8],
    x: [1, 4, 9, 8],
    w: [70, 4, 1, 1],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // スカラ（kFlat = 2·3·3 = 18 が 4 の倍数でない）。m = 70 / n = 9·7 = 63。
    name: "conv2d igemm スカラ [1,2,9,7] * W[70,2,3,3]",
    outShape: [1, 70, 9, 7],
    x: [1, 2, 9, 7],
    w: [70, 2, 3, 3],
    attrs: { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 },
  },
  {
    // バッチ 2（z 軸が 2 枚とも書かれること）
    name: "conv2d igemm バッチ 2 [2,4,9,8] * W[70,4,1,1]",
    outShape: [2, 70, 9, 8],
    x: [2, 4, 9, 8],
    w: [70, 4, 1, 1],
    attrs: { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 },
  },
];

Deno.test({
  name: "conv2d の implicit GEMM は毒値を 1 語も残さない（タイル書き出しの full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    for (const testCase of CONV2D_IGEMM_POISON_CASES) {
      const graph = poisonGraph(
        testCase.outShape,
        [
          { name: "x", shape: testCase.x },
          { name: "w", shape: testCase.w },
          { name: "b", shape: [testCase.w[0]] },
        ],
        { op: "conv2d", attrs: testCase.attrs },
      );
      const { output, reuseCount } = await runPoisoned(graph, {
        x: fill(testCase.x, (i) => 0.5 + (i % 11) * 0.25),
        w: fill(testCase.w, (i) => 0.25 + (i % 7) * 0.5),
        b: fill([testCase.w[0]], () => 0),
      });
      assert(reuseCount >= 1, `${testCase.name}: プール再利用が起きていない`);
      assertEquals(output.shape, [...testCase.outShape], testCase.name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${testCase.name}: 毒値の残存`,
      );
    }
  },
});

Deno.test({
  name: "rms_norm は毒値を 1 語も残さない（行カーネルの full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 行ループの縮退（grid-stride）や行内ループの打ち切りは「書かれない行 / 列」を残す。
    // [3,4]（毒値バッファは 12 要素で同サイズクラス）。
    const graph = poisonGraph(
      [3, 4],
      [{ name: "x", shape: [3, 4] }, { name: "w", shape: [4] }],
      { op: "rms_norm", attrs: { eps: 1e-6 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([3, 4], (i) => 1 + i),
      w: fill([4], () => 1),
    });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});

/**
 * GEMM 3 op（matmul / bmm / linear）のタイル書き出し（64×64 レジスタタイル — src/kernels/gemm.ts）。
 *
 * MUST: **v4 経路とスカラ経路を対で持つ**。書き出しのガードは変種ごとに別の式
 * （v4 = quad ガード 1 本 / スカラ = 成分ごと 4 本）なので、片方だけでは「タイルが
 * `ceil(m/64) × ceil(n/64)` で全域をちょうど 1 回覆う」を検証できていない。
 * MUST: 形は 64 を跨がせる（1 タイル未満に潰れるとガードが一度も偽にならない）。
 */
const GEMM_POISON_CASES: readonly {
  readonly name: string;
  readonly outShape: readonly number[];
  readonly operands: readonly { readonly name: string; readonly shape: readonly number[] }[];
  readonly op: string;
}[] = [
  // v4（k=20 / n=68 とも 4 の倍数）。n=68 は最終列タイルの有効 quad が 16 中 1。
  {
    name: "matmul v4 [65,20] × [20,68]",
    op: "matmul",
    outShape: [65, 68],
    operands: [{ name: "a", shape: [65, 20] }, { name: "b", shape: [20, 68] }],
  },
  // スカラ（k=19 / n=23 とも 4 の倍数でない）
  {
    name: "matmul スカラ [70,19] × [19,23]",
    op: "matmul",
    outShape: [70, 23],
    operands: [{ name: "a", shape: [70, 19] }, { name: "b", shape: [19, 23] }],
  },
  // bmm はバッチ軸（z）も全域を覆う必要がある
  {
    name: "bmm v4 [3,68,20] × [3,20,12]",
    op: "bmm",
    outShape: [3, 68, 12],
    operands: [{ name: "a", shape: [3, 68, 20] }, { name: "b", shape: [3, 20, 12] }],
  },
  {
    name: "bmm スカラ [2,70,19] × [2,19,23]",
    op: "bmm",
    outShape: [2, 70, 23],
    operands: [{ name: "a", shape: [2, 70, 19] }, { name: "b", shape: [2, 19, 23] }],
  },
  // linear は bias 加算ぶんだけ書き出し式が違う（重み格納 f32 の経路）
  {
    name: "linear v4 [65,20] × W[68,20]",
    op: "linear",
    outShape: [65, 68],
    operands: [
      { name: "x", shape: [65, 20] },
      { name: "w", shape: [68, 20] },
      { name: "b", shape: [68] },
    ],
  },
  {
    name: "linear スカラ [70,37] × W[23,37]",
    op: "linear",
    outShape: [70, 23],
    operands: [
      { name: "x", shape: [70, 37] },
      { name: "w", shape: [23, 37] },
      { name: "b", shape: [23] },
    ],
  },
];

Deno.test({
  name: "GEMM 3 op は毒値を 1 語も残さない（タイル書き出しの full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    for (const testCase of GEMM_POISON_CASES) {
      const graph = poisonGraph(testCase.outShape, testCase.operands, {
        op: testCase.op,
        attrs: {},
      });
      const inputs: Record<string, Tensor> = {};
      testCase.operands.forEach((operand, index) => {
        inputs[operand.name] = fill(operand.shape, (i) => 0.25 + ((i + index) % 11) * 0.5);
      });
      const { output, reuseCount } = await runPoisoned(graph, inputs);
      assert(reuseCount >= 1, `${testCase.name}: プール再利用が起きていない`);
      assertEquals(output.shape, [...testCase.outShape], testCase.name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${testCase.name}: 毒値の残存`,
      );
    }
  },
});

Deno.test({
  name: "conv_transpose1d は毒値を 1 語も残さない（gather 形の full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 素直な scatter 実装（入力 1 点を出力の複数点へ散らす）はゼロ初期化前提なので、
    // プール再利用バッファでは寄与の無い位置に毒値が残る。gather 形はここを必ず書く。
    // x[1,1,3] * W[1,2,4] stride=2 padding=1 → [1,2,6]（毒値バッファは 12 要素で同サイズ）。
    const graph = poisonGraph(
      [1, 2, 6],
      [
        { name: "x", shape: [1, 1, 3] },
        { name: "w", shape: [1, 2, 4] },
        { name: "b", shape: [2] },
      ],
      { op: "conv_transpose1d", attrs: { stride: 2, padding: 1 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([1, 1, 3], (i) => 1 + i),
      w: fill([1, 2, 4], (i) => 0.5 + i),
      b: fill([2], () => 0),
    });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});

/**
 * w8a8 linear（1 ノード = 2 dispatch）の**ノード内一時バッファ** `xq` / `xs` の full-write。
 *
 * 他の族と違い、ここで毒値を仕込む相手はノードの出力ではなく `quantize_rows` が書く 2 本の
 * 一時領域。書き漏らしは「量子化されていない前の値を整数内積が食う」形で現れるので、検出器は
 * 「毒値そのものの残存」ではなく **TS 参照との atol=0 突合**になる（毒値は必ず別の整数へ
 * 化けるので、1 語でも残れば値が動く）。
 *
 * MUST: 毒値バッファのサイズを**ノード出力 `y` と一致させない**。y は xq / xs より先に確保
 * されるので、同じサイズクラスだと毒値が y に配られて一時領域には届かない。
 */
const i8a8PoisonGraph = (
  poisonCount: number,
  m: number,
  n: number,
  k: number,
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["cast", "linear"] },
  symbols: [],
  inputs: [
    { name: "seed", dtype: "f32", shape: [poisonCount] },
    { name: "x", dtype: "f32", shape: [m, k] },
  ],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    poison: { dtype: "f32", shape: [poisonCount] },
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
  },
  nodes: [
    { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
    { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
  ],
});

Deno.test({
  name: "i8a8 linear の一時バッファ（xq / xs）が毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // m=4 / n=5 / k=8 → y = 80 バイト・xq = m·k = 32 バイト・xs = m·4 = 16 バイト。
    // 3 つとも別のサイズクラスなので、毒値の行き先を毒値バッファの大きさで選び分けられる。
    const m = 4;
    const n = 5;
    const k = 8;
    const cases: readonly { readonly name: string; readonly poisonCount: number }[] = [
      { name: "xq（量子化した活性）", poisonCount: (m * k) / 4 },
      { name: "xs（per-token scale）", poisonCount: m },
    ];
    const weight = fill([n, k], (i) => 0.125 + (i % 17) * 0.5 * (1 + (Math.floor(i / k) % 5)));
    const quantized = quantizeI8(weight.data, [n, k], 0);
    const bias = fill([n], (i) => ((i % 7) - 3) * 0.25);
    // 丸め境界から離れた刻み（atol=0 を主張する条件 — tests/gpu_i8a8_test.ts の MUST）
    const x = fill([m, k], (i) => ((i % 29) - 14) * 0.3717 + 0.0131);
    const expected = referenceLinearI8a8({
      x: x.data,
      weight: new Int8Array(quantized.bytes.buffer, quantized.bytes.byteOffset, n * k),
      weightScale: quantized.scale,
      bias: bias.data,
      m,
      n,
      k,
    });
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const model = openModel(buildSafetensors([
          { name: "m.b", dtype: "F32", shape: [n], data: f32Bytes(bias.data) },
          { name: "m.s", dtype: "F32", shape: [n, 1], data: f32Bytes(quantized.scale) },
          { name: "m.w", dtype: "I8", shape: [n, k], data: quantized.bytes },
        ], {
          karume_ir: JSON.stringify(i8a8PoisonGraph(testCase.poisonCount, m, n, k)),
        }));
        const session = await createSession(gpu, model, { linearCompute: "i8a8" });
        try {
          const outputs = await session.run({
            seed: fill([testCase.poisonCount], () => POISON),
            x,
          });
          const reuseCount = session.diagnostics().lastRun?.reuseCount ?? 0;
          assert(reuseCount >= 1, `${testCase.name}: プール再利用が起きていない`);
          assertEquals([...outputs["y"].data], [...expected], `${testCase.name}: atol=0 突合`);
          // 恒真化の門: 出力が定数なら突合は何も検証していない
          assert(new Set([...outputs["y"].data]).size > 1, `${testCase.name}: 出力が定数`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "attention は 3 dispatch でも出力の全バイトを書く（毒値の残存なし / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 融合 attention は 1 ノード = 3 dispatch（① QK → ② 行統計 → ③ PV — ADR 0023）。
    // full-write が掛かるのは**ノードの出力 O** で、③ の書き出しが端タイルを取りこぼすと
    // プール再利用バッファの毒値がそのまま結果に残る（cat と同じ「複数 dispatch で 1 ノード」）。
    // [1,2,3,4] × [1,2,5,4] → O は 24 要素。S（30 要素）と行統計（12 要素）はサイズクラスが
    // 違うので、毒値バッファが配り直されるのは O だけになる。
    const graph = poisonGraph(
      [1, 2, 3, 4],
      [
        { name: "q", shape: [1, 2, 3, 4] },
        { name: "k", shape: [1, 2, 5, 4] },
        { name: "v", shape: [1, 2, 5, 4] },
      ],
      { op: "attention", attrs: { scale: 0.7071067811865476 } },
    );
    const { output, reuseCount } = await runPoisoned(graph, {
      q: fill([1, 2, 3, 4], (i) => ((i % 7) - 3) * 0.5),
      k: fill([1, 2, 5, 4], (i) => 0.25 + (i % 5) * 0.5),
      v: fill([1, 2, 5, 4], (i) => ((i % 9) - 4) * 0.25),
    });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
    // 恒真化の門: 出力が全て同じ値なら「毒値が無い」は何も検証していない
    assert(new Set([...output.data]).size > 1, "出力が定数（毒値検査が恒真になっている）");
  },
});

/**
 * S の **f16 格納**変種（案 γ 波 1）の full-write。
 *
 * ノードの出力 O は f32 のままなので、③PV の書き出し自体は f32 変種と同じ式で走る。ここが
 * 見ているのは「S を `array<u32>` の 2 要素／語にしたことで**ノード出力の確保幅**が
 * ずれていないか」と「①QK の 2 語書きが端タイルで取りこぼしていないか」の 2 つ。
 *
 * NOTE: S は s16 で**半分のバイト数**になるのでサイズクラスが変わる。下の 2 形は S / 行統計 /
 * O がどれも別のサイズクラスになるように選んであり、毒値バッファが配り直されるのは O だけ。
 * MUST: 適格形（`D % 4 == 0 && N % 4 == 0`）を使う — 非適格だと f32 格納へ縮退して
 * この変種を 1 度も踏まない。
 */
Deno.test({
  name: "S の f16 格納変種も毒値を 1 語も残さない（①QK の 2 語書き / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    for (
      const testCase of [
        // 1 タイルに収まる最小形（S 144B / 行統計 48B / O 96B は全て別クラス）
        { name: "attention s16 小 M3 N12 D4", m: 3, n: 12, d: 4 },
        // 行 / 列とも 64 タイルを跨ぐ形（端タイルの 2 語書きが検出対象）
        { name: "attention s16 タイル端 M65 N68 D20", m: 65, n: 68, d: 20 },
      ]
    ) {
      const { name, m, n, d } = testCase;
      const { output, reuseCount } = await runPoisoned(
        poisonGraph(
          [1, 1, m, d],
          [
            { name: "q", shape: [1, 1, m, d] },
            { name: "k", shape: [1, 1, n, d] },
            { name: "v", shape: [1, 1, n, d] },
          ],
          { op: "attention", attrs: { scale: 0.7071067811865476 } },
        ),
        {
          q: fill([1, 1, m, d], (i) => ((i % 7) - 3) * 0.5),
          k: fill([1, 1, n, d], (i) => 0.25 + (i % 5) * 0.5),
          v: fill([1, 1, n, d], (i) => ((i % 9) - 4) * 0.25),
        },
        { attentionScoreStorage: "f16" },
      );
      assert(reuseCount >= 1, `${name}: プール再利用が起きていない（${reuseCount}）`);
      assertEquals([...output.data].filter((value) => value === POISON), [], `${name}: 毒値の残存`);
      // 恒真化の門: 出力が全て同じ値なら「毒値が無い」は何も検証していない
      assert(new Set([...output.data]).size > 1, `${name}: 出力が定数`);
    }
  },
});

/**
 * f16 **計算**変種（ADR 0028）の full-write。族が増えたので、書き出しのガードが変種ごとに
 * 別の式になっている 3 本（attention ③ の f32 出力 / linear の v4 / linear のスカラ）を載せる。
 *
 * NOTE: 融合 attention の S は f16 変種では**半分のバイト数**になるためサイズクラスが変わり、
 * 毒値バッファが配り直されるのはノード出力 O だけ、という前提は f32 変種と同じまま。
 */
Deno.test({
  name: "f16 計算変種も毒値を 1 語も残さない（タイル書き出しの full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE || !SHADER_F16_AVAILABLE,
  fn: async () => {
    const acquire = { shaderF16: true } as const;

    const attention = await runPoisoned(
      poisonGraph(
        [1, 2, 3, 4],
        [
          { name: "q", shape: [1, 2, 3, 4] },
          { name: "k", shape: [1, 2, 5, 4] },
          { name: "v", shape: [1, 2, 5, 4] },
        ],
        { op: "attention", attrs: { scale: 0.7071067811865476 } },
      ),
      {
        q: fill([1, 2, 3, 4], (i) => ((i % 7) - 3) * 0.5),
        k: fill([1, 2, 5, 4], (i) => 0.25 + (i % 5) * 0.5),
        v: fill([1, 2, 5, 4], (i) => ((i % 9) - 4) * 0.25),
      },
      { attentionCompute: "f16" },
      acquire,
    );
    assert(attention.reuseCount >= 1, "attention f16: プール再利用が起きていない");
    assertEquals(
      [...attention.output.data].filter((value) => value === POISON),
      [],
      "attention f16: 毒値の残存",
    );
    assert(new Set([...attention.output.data]).size > 1, "attention f16: 出力が定数");

    // linear は v4 とスカラで書き出しのガードが別の式（64 を跨ぐ形にする）
    for (
      const testCase of [
        { name: "linear f16 v4 [65,20] × W[68,20]", m: 65, n: 68, k: 20 },
        { name: "linear f16 スカラ [70,37] × W[23,37]", m: 70, n: 23, k: 37 },
      ]
    ) {
      const { name, m, n, k } = testCase;
      const { output, reuseCount } = await runPoisoned(
        poisonGraph([m, n], [
          { name: "x", shape: [m, k] },
          { name: "w", shape: [n, k] },
          { name: "b", shape: [n] },
        ], { op: "linear", attrs: {} }),
        {
          x: fill([m, k], (i) => 0.25 + (i % 11) * 0.5),
          w: fill([n, k], (i) => 0.125 + (i % 7) * 0.25),
          b: fill([n], (i) => ((i % 5) - 2) * 0.5),
        },
        { linearCompute: "f16" },
        acquire,
      );
      assert(reuseCount >= 1, `${name}: プール再利用が起きていない`);
      assertEquals(output.shape, [m, n], name);
      assertEquals(
        [...output.data].filter((value) => value === POISON),
        [],
        `${name}: 毒値の残存`,
      );
    }
  },
});

Deno.test({
  name: "軸 reduce は毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 軸変種は「1 スレッド = 1 出力」で `out[t]` を 1 回だけ書く。grid-stride の打ち切りや
    // `out_count` の取り違えは、書かれない要素 = 毒値の残存として出る（行 reduce と違い
    // workgroup 単位ではないので、打ち切りの粒度も違う = 別の検出対象）。
    // [3,4,5] の軸 1 を畳む → [3,5]（毒値バッファは 15 要素で同サイズクラス）。
    const graph = poisonGraph([3, 5], [{ name: "x", shape: [3, 4, 5] }], {
      op: "sum",
      attrs: { dim: 1 },
    });
    const { output, reuseCount } = await runPoisoned(graph, {
      x: fill([3, 4, 5], (i) => 1 + i),
    });
    assert(reuseCount >= 1, `プール再利用が起きていない（reuseCount=${reuseCount}）`);
    assertEquals(output.shape, [3, 5]);
    // Σ_j x[i,j,k] = Σ_j (1 + 20i + 5j + k) = 4 + 80i + 30 + 4k
    assertEquals([...output.data], [
      34,
      38,
      42,
      46,
      50,
      114,
      118,
      122,
      126,
      130,
      194,
      198,
      202,
      206,
      210,
    ]);
    assertEquals([...output.data].filter((value) => value === POISON), [], "毒値の残存");
  },
});
