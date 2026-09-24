// f16 格納の実行経路（ADR 0018）— 適格判定・ロード経路・カーネル変種・診断の通し検証。
//
// この波で入った 2 経路を両方踏む:
//   適格（消費が融合 5 op の weight スロットだけ）→ 生バイトのまま GPU 常駐し、dequant は
//     カーネル内（`unpack2x16float`）
//   適格外（bias / 混在消費 / その他）→ ロード時に CPU で f32 展開（VRAM 削減ゼロ）
//
// MUST: 数値ケースは **in-features / K / Cin·Kh·Kw を奇数**にする。f16 は 2 要素を 1 語に
// 詰めるので、行長が偶数だと重み行の先頭が必ず語境界に来る — 「対の選択を行内の相対添字で
// 取る」誤りが偶数長では一切値に出ない（波Q1 の故障注入で実証）。
// MUST: 重みの総要素数も奇数にする（末尾 2 バイトのゼロ詰めが無いと writeBuffer が
// validation で落ちる経路をここで踏む）。
// MUST: 期待値は**丸め後の重み**（fake-quant — ADR 0006）で作る。丸め前の f32 で比較すると
// 量子化誤差と実装誤差が混ざり、tolerance を緩める圧力になる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { CodecName } from "../src/format/container/codecs.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { decodeF16, f16BitsToF32 } from "../src/format/f16.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { RUNTIME_SUPPORT } from "../src/ops.ts";
import { assertRuntimeSupport, RuntimeSupportError } from "../src/ops/support.ts";
import { eligibleCompressedInitializers } from "../src/runtime/plan.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { f16BytesFromBits, quantizeF16 } from "./helpers/f16.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/**
 * 宣言の initializer 全部に指定 codec の空バイト列を当てて合流した `IrGraph`（同期）。
 * 適格判定はグラフ構造だけを見るので、実体の中身は要らない — 合流層だけ通す。
 *
 * 既定は `f16`（この系列が見るのは f16 格納の適格判定なので、名指ししない initializer も
 * f16 で載る — bias が f16 のまま適格から外れることが、ここの主張そのもの）。
 */
const mergedFor = (
  declaration: DeclarationJson,
  codecs: Readonly<Record<string, CodecName>> = {},
): IrGraph =>
  mergeGraph(
    declaration,
    Object.fromEntries(
      Object.keys(declaration.initializers).map((name) => [name, codecs[name] ?? "f16"]),
    ),
  );

// ---------------------------------------------------------------------------
// 適格判定（GPU 非依存 — グラフ構造だけで決まる）
// ---------------------------------------------------------------------------

/** `linear(x, w, b)` 1 本の宣言。`extra` で w の消費を足せる（混在消費を作るため）。 */
const linearDeclaration = (
  extra: DeclarationJson["nodes"] = [],
  extraValues: DeclarationJson["values"] = {},
  extraOutputs: readonly string[] = [],
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", ...extra.map((node) => node.op)] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
  outputs: ["y", ...extraOutputs],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 4] },
    b: { dtype: "f32", shape: [3] },
    y: { dtype: "f32", shape: [2, 3] },
    ...extraValues,
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }, ...extra],
});

Deno.test("適格判定は融合 5 op の weight スロット消費だけを通す", () => {
  const eligible = (declaration: DeclarationJson): readonly string[] =>
    [...eligibleCompressedInitializers(mergedFor(declaration))].sort();

  // 単独の weight 消費は適格。bias は同じグラフにいても**絶対に適格にならない**
  // （プロトタイプの f16 降格バグの逆 — ADR 0006 が名指しした規則）。
  assertEquals(eligible(linearDeclaration()), ["w"]);

  // MUST: 同じ initializer が weight 以外でも消費されたら適格を失う（混在消費）。
  // 圧縮のまま上げると elementwise 側のカーネルが u32 を f32 として読む沈黙誤値になる。
  assertEquals(
    eligible(linearDeclaration(
      [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
      { z: { dtype: "f32", shape: [3, 4] } },
      ["z"],
    )),
    [],
  );

  // 消費ゼロの initializer も適格外（実行に使われないバイトを「常駐圧縮」と数えない）
  const unused = linearDeclaration();
  unused.initializers["dead"] = {};
  unused.values["dead"] = { dtype: "f32", shape: [2] };
  assertEquals(eligible(unused), ["w"]);
});

Deno.test("適格判定は 5 op それぞれの weight スロット位置を見る（bias / index は適格にしない）", () => {
  // op → (ins の並び, weight の位置)。embedding だけスロット 0（他は 1）。
  const cases: readonly (readonly [string, readonly string[], readonly string[]])[] = [
    ["linear", ["x", "w", "b"], ["w"]],
    ["conv1d", ["x", "w", "b"], ["w"]],
    ["conv2d", ["x", "w", "b"], ["w"]],
    ["conv_transpose1d", ["x", "w", "b"], ["w"]],
    ["embedding", ["w", "x"], ["w"]],
    // 重みスロットを持たない op はどのスロットも適格にしない
    ["layer_norm", ["x", "w", "b"], []],
    ["rms_norm", ["x", "w"], []],
    ["masked_fill", ["w", "x"], []],
  ];
  for (const [op, ins, expected] of cases) {
    // 適格判定はグラフの構造（op 名 × スロット位置）だけを見るので、shape / dtype 宣言の
    // 整合はここでは要らない（契約検査は別層 — plan.ts の validateGraphContracts）。
    const graph = {
      initializers: {
        w: { storage: { codec: "f16" } },
        b: { storage: { codec: "f16" } },
      },
      nodes: [{ op, ins: [...ins], outs: ["y"], attrs: {} }],
    };
    assertEquals(
      [...eligibleCompressedInitializers(graph as never)].sort(),
      [...expected],
      `${op}(${ins.join(", ")})`,
    );
  }
});

Deno.test("graph 出力になった initializer は weight スロット消費だけでも適格外", () => {
  const eligible = (declaration: DeclarationJson): readonly string[] =>
    [...eligibleCompressedInitializers(mergedFor(declaration))].sort();

  // MUST: 消費が weight スロットだけでも、値そのものがグラフ出力なら適格を失う。readback は
  // semantic f32（4 バイト / 要素）を仮定して重みバッファから写すので、圧縮のまま常駐させると
  // copy が実バッファをはみ出す（極小サイズではビット列の読み替えが黙って返る）。
  assertEquals(eligible(linearDeclaration([], {}, ["w"])), []);

  // 失格は**出力に載った名前だけ**に効く（同じグラフの別の重みは適格のまま）
  const twoWeights = linearDeclaration(
    [{ op: "linear", ins: ["y", "w2", "b"], outs: ["y2"], attrs: {} }],
    { y2: { dtype: "f32", shape: [2, 3] } },
    ["y2", "w"],
  );
  // requires.ops は重複を拒む（linear 2 本ぶんの宣言は 1 つ）
  twoWeights.requires.ops = ["linear"];
  twoWeights.initializers["w2"] = {};
  twoWeights.values["w2"] = { dtype: "f32", shape: [3, 3] };
  assertEquals(eligible(twoWeights), ["w2"]);
});

// ---------------------------------------------------------------------------
// CPU 側の展開（ホスト鏡像の仕様）
// ---------------------------------------------------------------------------

Deno.test("f16 → f32 の展開は ±0 / subnormal / ±Inf / NaN / 最大最小 normal を正しく扱う", () => {
  assert(Object.is(f16BitsToF32(0x0000), 0), "+0");
  assert(Object.is(f16BitsToF32(0x8000), -0), "-0");
  // subnormal: 最小（2^-24）と最大（2^-14 に 1 ulp 足りない）
  assertEquals(f16BitsToF32(0x0001), 2 ** -24);
  assertEquals(f16BitsToF32(0x03ff), 1023 * 2 ** -24);
  assertEquals(f16BitsToF32(0x8001), -(2 ** -24));
  // normal: 最小（2^-14）と最大（65504）
  assertEquals(f16BitsToF32(0x0400), 2 ** -14);
  assertEquals(f16BitsToF32(0x7bff), 65504);
  assertEquals(f16BitsToF32(0xfbff), -65504);
  assertEquals(f16BitsToF32(0x3c00), 1);
  assertEquals(f16BitsToF32(0xbc00), -1);
  assertEquals(f16BitsToF32(0x7c00), Infinity);
  assertEquals(f16BitsToF32(0xfc00), -Infinity);
  assert(Number.isNaN(f16BitsToF32(0x7e00)), "quiet NaN");
  assert(Number.isNaN(f16BitsToF32(0xfc01)), "signaling NaN（符号付き）");
  // 展開後は必ず f32 で厳密（丸めが 1 度も起きない）
  for (const bits of [0x0001, 0x03ff, 0x0400, 0x7bff, 0x3555]) {
    assertEquals(Math.fround(f16BitsToF32(bits)), f16BitsToF32(bits), `bits=${bits}`);
  }
});

// ---------------------------------------------------------------------------
// GPU 実行（変種カーネル）
// ---------------------------------------------------------------------------

type WeightedCase = {
  readonly name: string;
  readonly op: string;
  /** グラフ入力（x / index）。 */
  readonly inputs: readonly (readonly [string, FilledTensor])[];
  readonly weight: FilledTensor;
  /** bias は**常に f32 格納**（ADR 0006）。embedding のように bias が無い op は undefined。 */
  readonly bias?: FilledTensor;
  readonly outShape: readonly number[];
  readonly attrs?: Record<string, unknown>;
};

/** weight を f16 initializer にした単一ノードの宣言 + 開いた容器。 */
const weightedModel = (
  testCase: WeightedCase,
  weightBytes: Uint8Array<ArrayBuffer>,
): Promise<OpenedContainer> => {
  const values: DeclarationJson["values"] = {
    w: { dtype: "f32", shape: [...testCase.weight.shape] },
    y: { dtype: "f32", shape: [...testCase.outShape] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  // NOTE: 要素数が奇数の f16（42 バイト）でも並べる順は効かない — 容器は block ごとに
  // 先頭 64 B 整列 + 末尾のゼロ詰めを書き手が焼くので、隣に何を置いても整列は崩れない。
  const tensors: TensorInput[] = [];
  if (testCase.bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...testCase.bias.shape] };
    initializers["b"] = {};
    tensors.push({
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(testCase.bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    });
  }
  tensors.push({
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: weightBytes,
    encoding: { codec: "f16" },
  });
  // ins の並びは契約どおり（embedding は weight が先頭・他は x の次）
  const ins = testCase.op === "embedding" ? ["w", ...testCase.inputs.map(([name]) => name)] : [
    ...testCase.inputs.map(([name]) => name),
    "w",
    ...(testCase.bias === undefined ? [] : ["b"]),
  ];
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: [testCase.op] },
    symbols: [],
    inputs: testCase.inputs.map(([name, tensor]) => ({
      name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
    })),
    outputs: ["y"],
    initializers,
    values,
    nodes: [{ op: testCase.op, ins, outs: ["y"], attrs: { ...testCase.attrs } }],
  };
  return openModelBytes(declaration, tensors);
};

const runWeighted = async (
  gpu: GpuContext,
  testCase: WeightedCase,
): Promise<{ readonly output: Tensor; readonly residentBytes: number }> => {
  const quantized = quantizeF16(testCase.weight.data);
  const session = await createSessionFromContainer(
    gpu,
    await weightedModel(testCase, quantized.bytes),
    GRAPH_NAME,
  );
  try {
    const named: Record<string, Tensor> = {};
    for (const [name, tensor] of testCase.inputs) named[name] = tensor;
    const outputs = await session.run(named);
    return {
      output: outputs["y"],
      residentBytes: session.diagnostics().storage.residentCompressedBytes,
    };
  } finally {
    await session.dispose();
  }
};

/**
 * 5 変種の数値ケース。**重みの総要素数が全て奇数**（ゼロ詰め経路）で、
 * **重み行の長さも奇数**（対の選択が平坦添字であることの検出器）。
 */
const WEIGHTED_CASES: readonly WeightedCase[] = [
  {
    // k = 7（奇数）/ 重み 3·7 = 21 要素（奇数）
    name: "linear [3,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [["x", fill([3, 7], SIGNED)]],
    weight: fill([3, 7], POSITIVE),
    bias: fill([3], SIGNED),
    outShape: [3, 3],
  },
  {
    // Cin·K = 9（奇数）/ 重み 5·3·3 = 45 要素（奇数）
    name: "conv1d [1,3,9] * W[5,3,3] stride=1 padding=1",
    op: "conv1d",
    inputs: [["x", fill([1, 3, 9], SIGNED)]],
    weight: fill([5, 3, 3], POSITIVE),
    bias: fill([5], SIGNED),
    outShape: [1, 5, 9],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  {
    // Cin·Kh·Kw = 9（奇数）/ 重み 5·3·3·1 = 45 要素（奇数）。Kh ≠ Kw で軸取り違えも赤にする
    name: "conv2d [1,3,5,4] * W[5,3,3,1]",
    op: "conv2d",
    inputs: [["x", fill([1, 3, 5, 4], SIGNED)]],
    weight: fill([5, 3, 3, 1], POSITIVE),
    bias: fill([5], SIGNED),
    outShape: [1, 5, 5, 4],
    attrs: { stride: [1, 1], padding: [1, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // K = 3（奇数・契約 2·padding == K − stride を満たす stride 1）/ 重み 3·5·3 = 45 要素
    name: "conv_transpose1d [1,3,7] * W[3,5,3] stride=1 padding=1",
    op: "conv_transpose1d",
    inputs: [["x", fill([1, 3, 7], SIGNED)]],
    weight: fill([3, 5, 3], POSITIVE),
    bias: fill([5], SIGNED),
    outShape: [1, 5, 7],
    attrs: { stride: 1, padding: 1 },
  },
  {
    // H = 3（奇数）/ 重み 5·3 = 15 要素（奇数）
    name: "embedding W[5,3] × index [2,4]",
    op: "embedding",
    inputs: [["x", fill([2, 4], (i) => (i * 3) % 5, "i32")]],
    weight: fill([5, 3], SIGNED),
    outShape: [2, 4, 3],
    attrs: { padding_idx: -1 },
  },
];

Deno.test({
  name: "w=f16 変種 5 種が CPU 参照（丸め後の重み）と一致する（奇数長・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of WEIGHTED_CASES) {
        const quantized = quantizeF16(testCase.weight.data);
        // 総要素数が奇数 = 生バイト長が 4 の倍数でない（ゼロ詰めが要る形）
        assertEquals(
          quantized.bytes.byteLength % 4,
          2,
          `${testCase.name}: 重みの総要素数が奇数でない（ゼロ詰め経路を踏まない）`,
        );
        const { output, residentBytes } = await runWeighted(gpu, testCase);
        // 圧縮のまま常駐している（= CPU 展開に落ちていない）。ゼロ詰めで +2 バイト。
        assertEquals(
          residentBytes,
          quantized.bytes.byteLength + 2,
          `${testCase.name}: GPU 常駐圧縮バイト数`,
        );
        const reference = refTensor(testCase.weight.shape, quantized.values);
        const operands: RefTensor[] = testCase.op === "embedding"
          ? [reference, ...testCase.inputs.map(([, tensor]) => tensor as RefTensor)]
          : [
            ...testCase.inputs.map(([, tensor]) => tensor as RefTensor),
            reference,
            ...(testCase.bias === undefined ? [] : [testCase.bias as RefTensor]),
          ];
        const expected = applyReferenceOp(
          testCase.op,
          operands,
          testCase.attrs ?? {},
          testCase.outShape,
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * linear の GEMM タイル境界（64×64 レジスタタイル — src/kernels/gemm.ts）を f16 重みで踏む。
 *
 * MUST: **v4 経路とスカラ経路を対で持つ**。
 * - v4（`k%4==0 && n%4==0`）は u32 **2 語**で 4 要素を組む経路で、2 語目のオフセット誤りは
 *   k が 4 の倍数の形なら必ず値に出る（k=20）。
 * - スカラ経路は 1 語 2 要素の偶奇選択で、上の {@link WEIGHTED_CASES}（k=7）と合わせて
 *   「行長が 2 の倍数でない」形が唯一の検出器（k=37）。v4 経路では行頭が常に語境界に来るので
 *   この罠を**踏めない** — スカラ側のケースを削ると検出力が消える。
 */
Deno.test({
  name: "w=f16 の linear が GEMM タイル境界（v4 / スカラ）で CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const cases: readonly WeightedCase[] = [
      {
        // k=20（4 の倍数・16 の倍数でない）/ n=68（最終タイルの有効 quad が 16 中 1）/
        // m=65（行タイル 2 枚）。m/n/k は互いに違う長さ。
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], POSITIVE),
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        // k=37（2 の倍数でない = 平坦添字の偶奇の検出器）/ n=23 / m=70（行タイル 2 枚）
        name: "linear スカラ [70,37] × W[23,37] + b[23]",
        op: "linear",
        inputs: [["x", fill([70, 37], SIGNED)]],
        weight: fill([23, 37], POSITIVE),
        bias: fill([23], SIGNED),
        outShape: [70, 23],
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const quantized = quantizeF16(testCase.weight.data);
        const { output } = await runWeighted(gpu, testCase);
        const expected = applyReferenceOp(
          testCase.op,
          [
            testCase.inputs[0][1] as RefTensor,
            refTensor(testCase.weight.shape, quantized.values),
            testCase.bias as RefTensor,
          ],
          {},
          testCase.outShape,
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * GPU の `unpack2x16float` と CPU の {@link f16BitsToF32} が**全 65,536 パターンで
 * ビット一致**する（ADR 0018 の検証条件）。
 *
 * 運び方: embedding の weight を [256, 256] の f16 とし、`bits = row·256 + col` を並べて
 * 全パターンを 1 度に敷き詰める。embedding は行 gather で算術を 1 つも挟まないので、
 * 出力は dequant 結果そのもの。
 *
 * MUST: NaN だけは**双方 NaN**で見る。WGSL は NaN のペイロード伝播を保証しないため、
 * ビット列の一致を要求すると仕様外の性質に依存したテストになる。
 */
Deno.test({
  name: "GPU の unpack2x16float と CPU 展開が全 65,536 パターンでビット一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const vocab = 256;
    const hidden = 256;
    const bytes = f16BytesFromBits(
      Array.from({ length: vocab * hidden }, (_, i) => i),
    );
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "index", dtype: "i32", shape: [vocab] }],
      outputs: ["y"],
      initializers: { w: {} },
      values: {
        w: { dtype: "f32", shape: [vocab, hidden] },
        y: { dtype: "f32", shape: [vocab, hidden] },
      },
      nodes: [{
        op: "embedding",
        ins: ["w", "index"],
        outs: ["y"],
        attrs: { padding_idx: -1 },
      }],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes, encoding: { codec: "f16" } },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    let actual: Float32Array<ArrayBuffer>;
    try {
      const outputs = await session.run({ index: fill([vocab], (i) => i, "i32") });
      actual = outputs["y"].data as Float32Array<ArrayBuffer>;
      // 圧縮のまま常駐していなければ「GPU の展開」を見ていない
      assertEquals(session.diagnostics().storage.residentCompressedBytes, bytes.byteLength);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
    const expected = decodeF16(bytes);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    let nan = 0;
    let subnormal = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (Number.isNaN(expected[i])) {
        nan += 1;
        assert(Number.isNaN(actual[i]), `pattern 0x${i.toString(16)}: GPU 側が NaN でない`);
        continue;
      }
      if (((i >> 10) & 0x1f) === 0 && (i & 0x3ff) !== 0) subnormal += 1;
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `pattern 0x${i.toString(16)}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
    }
    // 網羅の証拠（NaN 2046 本・subnormal 2046 本を実際に踏んでいる）
    assertEquals(nan, 2046, "NaN パターン数");
    assertEquals(subnormal, 2046, "subnormal パターン数");
  },
});

/**
 * 変種経路の full-write（ADR 0014）。プール再利用バッファに毒値を仕込んでから f16 変種を
 * 実行し、1 語も残らないことを見る。
 *
 * 仕込み方は tests/gpu_full_write_test.ts と同じ（消費者ゼロの恒等 cast の出力が
 * ノード境界でプールへ戻り、次の確保で配り直される）。
 */
const poisonedF16Graph = (
  op: string,
  node: { readonly ins: readonly string[]; readonly attrs: Record<string, unknown> },
  inputs: readonly (readonly [string, FilledTensor])[],
  weightShape: readonly number[],
  outShape: readonly number[],
  bias?: readonly number[],
): DeclarationJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  const values: DeclarationJson["values"] = {
    poison: { dtype: "f32", shape: [count] },
    w: { dtype: "f32", shape: [...weightShape] },
    y: { dtype: "f32", shape: [...outShape] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  if (bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...bias] };
    initializers["b"] = {};
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", op] },
    symbols: [],
    inputs: [
      { name: "seed", dtype: "f32", shape: [count] },
      ...inputs.map(([name, tensor]) => ({
        name,
        dtype: tensor.dtype,
        shape: [...tensor.shape],
      })),
    ],
    outputs: ["y"],
    initializers,
    values,
    nodes: [
      { op: "cast", ins: ["seed"], outs: ["poison"], attrs: { to: "f32" } },
      { op, ins: [...node.ins], outs: ["y"], attrs: { ...node.attrs } },
    ],
  };
};

/** 毒値 0xDEADBEEF を f32 として読んだもの（0 でない有限値）。 */
const POISON = new Float32Array(new Uint32Array([0xDEADBEEF]).buffer)[0];

Deno.test({
  name: "w=f16 変種もプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // linear（タイル形）と embedding（grid-stride + 範囲外分岐）の 2 本で踏む
      const cases: readonly {
        readonly name: string;
        readonly graph: DeclarationJson;
        readonly tensors: readonly TensorInput[];
        readonly inputs: Record<string, Tensor>;
        readonly count: number;
      }[] = [
        (() => {
          const weight = fill([3, 7], POSITIVE);
          const bias = fill([3], SIGNED);
          return {
            name: "linear",
            graph: poisonedF16Graph(
              "linear",
              { ins: ["x", "w", "b"], attrs: {} },
              [["x", fill([4, 7], SIGNED)]],
              [3, 7],
              [4, 3],
              [3],
            ),
            tensors: [
              {
                graph: GRAPH_NAME,
                initializer: "b",
                bytes: new Uint8Array(bias.data.buffer.slice(0)),
                encoding: { codec: "f32" },
              },
              {
                graph: GRAPH_NAME,
                initializer: "w",
                bytes: quantizeF16(weight.data).bytes,
                encoding: { codec: "f16" },
              },
            ],
            inputs: { x: fill([4, 7], SIGNED) },
            count: 12,
          };
        })(),
        (() => {
          const weight = fill([5, 3], SIGNED);
          return {
            name: "embedding",
            graph: poisonedF16Graph(
              "embedding",
              { ins: ["w", "idx"], attrs: { padding_idx: -1 } },
              [["idx", fill([4], (i) => i % 5, "i32")]],
              [5, 3],
              [4, 3],
            ),
            tensors: [
              {
                graph: GRAPH_NAME,
                initializer: "w",
                bytes: quantizeF16(weight.data).bytes,
                encoding: { codec: "f16" },
              },
            ],
            inputs: { idx: fill([4], (i) => i % 5, "i32") },
            count: 12,
          };
        })(),
      ];
      for (const testCase of cases) {
        const opened = await openModelBytes(testCase.graph, testCase.tensors);
        const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
        try {
          const outputs = await session.run({
            seed: fill([testCase.count], () => POISON),
            ...testCase.inputs,
          });
          const reuseCount = session.diagnostics().lastRun?.reuseCount ?? 0;
          // 再利用が起きていなければ毒値検査は何も見ていない（恒真化の門）
          assert(reuseCount >= 1, `${testCase.name}: プール再利用が起きていない`);
          assertEquals(
            [...outputs["y"].data].filter((value) => value === POISON),
            [],
            `${testCase.name}: 毒値の残存`,
          );
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
 * 適格外（混在消費）は CPU で f32 展開され、**適格経路と同じ値**を出す。
 *
 * MUST: 2 経路の出力を突き合わせる。片方だけを CPU 参照と比べると、両経路が同じだけ
 * ずれている形（例: 展開表が誤っていて GPU 側も同じ誤りを持つ）を見逃す。
 */
Deno.test({
  name: "混在消費の f16 は CPU 展開へ落ち、適格経路とビット単位で同じ値を出す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = quantizeF16(weight.data);
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: quantized.bytes,
        encoding: { codec: "f16" },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(bias.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ];
    const x = fill([2, 4], SIGNED);
    const run = async (
      gpu: GpuContext,
      declaration: DeclarationJson,
    ): Promise<{ readonly y: Tensor; readonly resident: number; readonly expanded: number }> => {
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(declaration, tensors),
        GRAPH_NAME,
      );
      try {
        const outputs = await session.run({ x });
        const storage = session.diagnostics().storage;
        return {
          y: outputs["y"],
          resident: storage.residentCompressedBytes,
          expanded: storage.hostExpandedBytes,
        };
      } finally {
        await session.dispose();
      }
    };
    const gpu = await acquireGpu();
    try {
      const eligible = await run(gpu, linearDeclaration());
      // w を add でも消費する = 混在消費（同じ linear ノードは残す）
      const mixed = await run(
        gpu,
        linearDeclaration(
          [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
          { z: { dtype: "f32", shape: [3, 4] } },
          ["z"],
        ),
      );
      assertEquals(eligible.resident, quantized.bytes.byteLength, "適格側は圧縮のまま常駐");
      assertEquals(eligible.expanded, 0);
      assertEquals(mixed.resident, 0, "混在消費は 1 バイトも圧縮常駐しない");
      assertEquals(mixed.expanded, quantized.values.byteLength, "CPU 展開バイト数（f32 換算）");
      // 同じ重み・同じ演算なので出力はビット単位で一致する（丸めの差も出ない）
      assertEquals([...mixed.y.data], [...eligible.y.data]);
      // オラクル: 丸め後の重みでの CPU 参照
      const expected = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 4], quantized.values), bias as RefTensor],
        {},
        [2, 3],
      );
      const report = compareTensors(eligible.y, expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 診断（ADR 0006 の常設義務）の検算。適格・適格外・非圧縮を 1 つのグラフに混ぜ、
 * 両方の数字が**実バイト数**と一致することを見る。
 *
 * bias を f16 で宣言しても**必ず CPU 展開**に落ちる（bias スロットは適格判定に載らない —
 * プロトタイプの降格バグの逆）。
 */
Deno.test({
  name: "診断は適格 / 適格外 / 非圧縮を実バイト数で区別する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 重み 3·7 = 21 要素（奇数 → ゼロ詰めで 44 バイト）/ bias 3 要素を **f16 宣言**
    const weight = quantizeF16(fill([3, 7], POSITIVE).data);
    const bias = quantizeF16(fill([3], SIGNED).data);
    const scale = fill([3], POSITIVE);
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear", "mul"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [2, 7] }],
      outputs: ["y"],
      initializers: { w: {}, b: {}, s: {} },
      values: {
        w: { dtype: "f32", shape: [3, 7] },
        b: { dtype: "f32", shape: [3] },
        s: { dtype: "f32", shape: [3] },
        h: { dtype: "f32", shape: [2, 3] },
        y: { dtype: "f32", shape: [2, 3] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
        { op: "mul", ins: ["h", "s"], outs: ["y"], attrs: {} },
      ],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes: weight.bytes, encoding: { codec: "f16" } },
      { graph: GRAPH_NAME, initializer: "b", bytes: bias.bytes, encoding: { codec: "f16" } },
      {
        graph: GRAPH_NAME,
        initializer: "s",
        bytes: new Uint8Array(scale.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    try {
      const storage = session.diagnostics().storage;
      // 適格: 21 要素 × 2 バイト = 42 → 4 バイト整列で 44
      assertEquals(weight.bytes.byteLength, 42);
      assertEquals(storage.residentCompressedBytes, 44);
      // 適格外: bias 3 要素を f32 展開 = 12 バイト（f32 格納の scale は**どちらにも入らない**）
      assertEquals(storage.hostExpandedBytes, 12);
      // 展開した bias が値としても正しい（診断だけ合っていて中身が壊れている形を塞ぐ）
      const x = fill([2, 7], SIGNED);
      const outputs = await session.run({ x });
      const linear = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 7], weight.values), refTensor([3], bias.values)],
        {},
        [2, 3],
      );
      const expected = applyReferenceOp("mul", [linear, scale as RefTensor], {}, [2, 3]);
      const report = compareTensors(outputs["y"], expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * initializer 自身をグラフ出力にした形（IR が許す — format/ir.ts の定義済み検査）。
 *
 * readback は宣言 dtype の semantic f32 を 4 バイト / 要素で重みバッファから写すので、圧縮の
 * まま常駐していると copy が実バッファ（2 バイト / 要素）をはみ出す。適格判定が `graph.outputs`
 * を見ていれば CPU 展開へ落ち、丸め後の重みがそのまま読める。
 */
Deno.test({
  name:
    "graph 出力にした f16 initializer は CPU 展開へ落ち、丸め後の値が f32 として読める（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = quantizeF16(weight.data);
    const opened = await openModelBytes(linearDeclaration([], {}, ["w"]), [
      { graph: GRAPH_NAME, initializer: "w", bytes: quantized.bytes, encoding: { codec: "f16" } },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(bias.data.buffer.slice(0)),
        encoding: { codec: "f32" },
      },
    ]);
    const x = fill([2, 4], SIGNED);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
    try {
      const storage = session.diagnostics().storage;
      assertEquals(
        storage.residentCompressedBytes,
        0,
        "グラフ出力の重みは 1 バイトも圧縮常駐しない",
      );
      assertEquals(storage.hostExpandedBytes, quantized.values.byteLength, "CPU 展開バイト数");
      const outputs = await session.run({ x });
      // 重みは実行に依らない定数なので、丸め後の値とビット単位で一致する
      assertEquals(outputs["w"].shape, [3, 4]);
      assertEquals([...outputs["w"].data], [...quantized.values]);
      // 同じ run の計算側も従来どおり（展開経路でも値は変わらない）
      const expected = applyReferenceOp(
        "linear",
        [x as RefTensor, refTensor([3, 4], quantized.values), bias as RefTensor],
        {},
        [2, 3],
      );
      const report = compareTensors(outputs["y"], expected, GEMM_TOLERANCE);
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * bf16 は宣言としては valid なまま**実行できない**ので、capability 不足として initializer
 * 名つきで列挙される（i8 は ADR 0019 で門が開いた — tests/gpu_i8_weights_test.ts）。
 */
Deno.test("bf16 は capability 不足で fail loudly（f16 の門は bf16 まで開かない）", () => {
  const error = assertThrows(
    // 圧縮重み + f32 bias（`b` を省くと `mergedFor` の既定 f16 になり、fixture の意図が変わる）。
    () =>
      assertRuntimeSupport(
        mergedFor(linearDeclaration(), { w: "bf16", b: "f32" }),
        RUNTIME_SUPPORT,
      ),
    RuntimeSupportError,
    "capability 不足",
  );
  assertEquals(error.message.includes("非対応 格納 'bf16' (1): w"), true, error.message);
  // f16 は同じ門を通る（適格かどうかは実行可否と別軸）
  assertRuntimeSupport(mergedFor(linearDeclaration(), { b: "f32" }), RUNTIME_SUPPORT);
});
