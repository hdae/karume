// i8（per-channel symmetric int8）格納の実行経路（ADR 0019）— ロード時検証・適格判定の共用・
// カーネル変種・診断の通し検証。
//
// f16（ADR 0018 / tests/gpu_f16_weights_test.ts）と同じ 2 経路を踏む:
//   適格（消費が融合 5 op の weight スロットだけ）→ 生バイトのまま GPU 常駐し、dequant は
//     カーネル内（`unpack4xI8` + per-channel scale）
//   適格外（bias / 混在消費 / その他）→ ロード時に CPU で f32 展開（VRAM 削減ゼロ）
//
// MUST: 数値ケースは **重み行の長さを 4 の倍数にしない**。i8 は 4 要素を 1 語に詰めるので、
// 行長が 4 の倍数だと重み行の先頭が必ず語境界に来る — 「語とレーンの選択を行内の相対添字で
// 取る」誤りが 4 の倍数長では一切値に出ない（f16 の偶奇 MUST と同型の罠）。
// MUST: 重みの総要素数も 4 の倍数にしない（末尾ゼロ詰めが無いと writeBuffer が validation で
// 落ちる経路をここで踏む）。
// MUST: 期待値は**丸め後の重み**（fake-quant — ADR 0006）で作る。丸め前の f32 で比較すると
// 量子化誤差と実装誤差が混ざり、tolerance を緩める圧力になる。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { assertRuntimeSupport, ContainerError, openModel } from "../src/format/container.ts";
import { alignI8Payload, decodeI8, I8Error } from "../src/format/i8.ts";
import { IrError, parseIrGraph } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { RUNTIME_SUPPORT } from "../src/ops.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  weightChannelAxes,
} from "../src/runtime/plan.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { buildSafetensors, f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { i8BytesFrom, quantizeI8 } from "./helpers/i8.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/** `linear(x, w, b)` 1 本のグラフ（w は i8 + scale）。`extra` で w の消費を足せる。 */
const linearGraph = (
  storage: Record<string, unknown>,
  extra: GraphJson["nodes"] = [],
  extraValues: GraphJson["values"] = {},
  extraOutputs: readonly string[] = [],
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear", ...extra.map((node) => node.op)] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
  outputs: ["y", ...extraOutputs],
  initializers: {
    w: { tensor: "m.w", storage },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [3, 4] },
    b: { dtype: "f32", shape: [3] },
    y: { dtype: "f32", shape: [2, 3] },
    ...extraValues,
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }, ...extra],
});

// ---------------------------------------------------------------------------
// CPU 側の展開（ホスト鏡像の仕様）
// ---------------------------------------------------------------------------

Deno.test("decodeI8: 符号付き 8bit を per-channel scale で展開する（keepdim broadcast）", () => {
  // [2,3] の重み。チャネル軸 0 の scale [2,1] が行ごとに掛かる。
  const bytes = i8BytesFrom([127, -128, 0, 1, -1, 64]);
  const scale = Float32Array.from([0.5, 0.25]);
  assertEquals(
    [...decodeI8(bytes, [2, 3], scale, [2, 1])],
    [63.5, -64, 0, 0.25, -0.25, 16],
  );
  // チャネル軸 1（conv_transpose1d の形）。列ごとに掛かる。
  assertEquals(
    [...decodeI8(bytes, [2, 3], Float32Array.from([1, 2, 4]), [1, 3])],
    [127, -256, 0, 1, -2, 256],
  );
  // 全軸 1 の scale（per-tensor 形）も broadcast としては成立する
  assertEquals(
    [...decodeI8(bytes, [2, 3], Float32Array.from([2]), [1, 1])],
    [254, -256, 0, 2, -2, 128],
  );
});

Deno.test("decodeI8: rank 3 以上でも軸ごとに正しく broadcast する", () => {
  // [2,3,2]（conv1d の [Cout,Cin,K] 形）。チャネル軸 0 = 出力チャネル。
  const q = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6];
  const bytes = i8BytesFrom(q);
  const scale = Float32Array.from([0.5, 4]);
  const decoded = [...decodeI8(bytes, [2, 3, 2], scale, [2, 1, 1])];
  assertEquals(decoded, [
    ...q.slice(0, 6).map((value) => value * 0.5),
    ...q.slice(6).map((value) => value * 4),
  ]);
  // 中央の軸（conv_transpose1d の [Cin,Cout,K] 形 — 軸 1 が出力チャネル）
  const middle = [...decodeI8(bytes, [2, 3, 2], Float32Array.from([1, 2, 4]), [1, 3, 1])];
  assertEquals(middle, [1, 2, 6, 8, 20, 24, -1, -2, -6, -8, -20, -24]);
});

Deno.test("decodeI8: 展開の丸めは f32 の 1 回だけ（GPU の f32 乗算と同値）", () => {
  // 0.1 は f32 で丸められる値。q·s は f64 で厳密（7bit × 24bit）なので、f32 へ 1 度だけ
  // 丸めた結果が GPU の f32 乗算の正しい丸めと一致する。
  const scale = Float32Array.from([0.1]);
  const decoded = decodeI8(i8BytesFrom([3, 7, 127]), [3, 1], scale, [1, 1]);
  for (const [index, q] of [3, 7, 127].entries()) {
    assertEquals(decoded[index], Math.fround(q * scale[0]), `q=${q}`);
    assertEquals(Math.fround(decoded[index]), decoded[index], `q=${q}: f32 で厳密`);
  }
});

Deno.test("decodeI8: broadcast できない scale と要素数不一致を拒否する", () => {
  const bytes = i8BytesFrom([1, 2, 3, 4, 5, 6]);
  assertThrows(
    () => decodeI8(bytes, [2, 3], Float32Array.from([1, 2]), [2]),
    I8Error,
    "rank",
  );
  // 軸 1 が 2（重みの 3 と違う）— 要素数は shape と整合しているので broadcast 検査が見る
  assertThrows(
    () => decodeI8(bytes, [2, 3], Float32Array.from([1, 2, 3, 4]), [2, 2]),
    I8Error,
    "broadcast できない",
  );
  assertThrows(
    () => decodeI8(bytes, [2, 4], Float32Array.from([1, 2]), [2, 1]),
    I8Error,
    "ペイロード",
  );
});

Deno.test("alignI8Payload: 4 バイト境界までゼロ詰めし、整列済みはコピーしない", () => {
  const aligned = i8BytesFrom([1, 2, 3, 4]);
  assertEquals(alignI8Payload(aligned), aligned);
  assert(alignI8Payload(aligned) === aligned, "整列済みは同じ view を返す");
  for (const length of [1, 2, 3, 5, 6, 7]) {
    const padded = alignI8Payload(i8BytesFrom(new Array(length).fill(7)));
    assertEquals(padded.byteLength % 4, 0, `${length} バイト`);
    assertEquals(padded.byteLength, length + (4 - (length % 4)), `${length} バイト`);
    assertEquals([...padded.slice(length)], new Array(padded.byteLength - length).fill(0));
  }
});

// ---------------------------------------------------------------------------
// 適格判定（f16 と共用）とチャネル軸（GPU 非依存）
// ---------------------------------------------------------------------------

Deno.test("i8 の適格判定は f16 と同じ 1 本の判定を通る（新設していない）", () => {
  const eligible = (graph: GraphJson): readonly string[] =>
    [...eligibleCompressedInitializers(parseIrGraph(JSON.stringify(graph)))].sort();
  const storage = { dtype: "i8", scale: "m.s" };
  assertEquals(eligible(linearGraph(storage)), ["w"]);
  // 混在消費（weight スロット以外でも消費）は適格を失う
  assertEquals(
    eligible(linearGraph(
      storage,
      [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
      { z: { dtype: "f32", shape: [3, 4] } },
      ["z"],
    )),
    [],
  );
});

Deno.test("graph 出力になった i8 initializer も同じ 1 本の判定で適格外", () => {
  // MUST: readback は semantic f32（4 バイト / 要素）を仮定して重みバッファから写すので、
  // i8（1 バイト / 要素 + scale）のまま常駐させると copy が実バッファをはみ出す。
  const eligible = eligibleCompressedInitializers(
    parseIrGraph(JSON.stringify(linearGraph({ dtype: "i8", scale: "m.s" }, [], {}, ["w"]))),
  );
  assertEquals([...eligible], []);
});

Deno.test("チャネル軸は消費側 op から決まる（conv_transpose1d だけ 1）", () => {
  const axisOf = (op: string, ins: readonly string[]): number | undefined => {
    const graph = {
      initializers: { w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } } },
      nodes: [{ op, ins: [...ins], outs: ["y"], attrs: {} }],
    };
    return weightChannelAxes(graph as never).get("w");
  };
  assertEquals(axisOf("linear", ["x", "w", "b"]), 0);
  assertEquals(axisOf("conv1d", ["x", "w", "b"]), 0);
  assertEquals(axisOf("conv2d", ["x", "w", "b"]), 0);
  // MUST: 重み [Cin, Cout, K] の転置レイアウトなので出力チャネルは軸 1
  assertEquals(axisOf("conv_transpose1d", ["x", "w", "b"]), 1);
  assertEquals(axisOf("embedding", ["w", "x"]), 0);
  // 重みスロットを持たない op / weight スロット以外の位置は軸を持たない
  assertEquals(axisOf("layer_norm", ["x", "w", "b"]), undefined);
  assertEquals(axisOf("embedding", ["x", "w"]), undefined);
});

Deno.test("同じ重みを軸の違う 2 op が食う形は fail loudly", () => {
  const graph = {
    initializers: { w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } } },
    nodes: [
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "conv_transpose1d", ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
    ],
  };
  assertThrows(() => weightChannelAxes(graph as never), ExecutionError, "チャネル軸が消費側で");
});

// ---------------------------------------------------------------------------
// ロード時検証（scale の宣言と実テンソル — GPU 非依存）
// ---------------------------------------------------------------------------

/** i8 の linear モデル（scale は keepdim [3,1]）。`mutate` で 1 点だけ壊せる。 */
const i8LinearModel = (
  mutate: (parts: {
    graph: GraphJson;
    tensors: TensorSpec[];
  }) => void = () => {},
): ArrayBuffer => {
  const weight = fill([3, 4], POSITIVE);
  const quantized = quantizeI8(weight.data, [3, 4], 0);
  const graph = linearGraph({ dtype: "i8", scale: "m.s" });
  // MUST: I8 はファイル末尾（1 バイト要素なので後続テンソルの整列を壊す）
  const tensors: TensorSpec[] = [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0.5, -0.25, 1]) },
    { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes(quantized.scale) },
    { name: "m.w", dtype: "I8", shape: [3, 4], data: quantized.bytes },
  ];
  mutate({ graph, tensors });
  return buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
};

Deno.test("i8 は scale の宣言が必須（既定 1.0 で補完しない）", () => {
  assertThrows(
    () =>
      openModel(i8LinearModel(({ graph }) => {
        graph.initializers["w"].storage = { dtype: "i8" };
      })),
    IrError,
    "scale",
  );
});

Deno.test("scale テンソルの実在・dtype・keepdim 形・名前衝突をロード時に見る", () => {
  // 実在しない
  assertThrows(
    () =>
      openModel(i8LinearModel(({ graph }) => {
        graph.initializers["w"].storage = { dtype: "i8", scale: "m.missing" };
      })),
    ContainerError,
    "がファイルに無い",
  );
  // F32 でない（f16 のビット列として読むと全チャネルが桁違いになる）
  assertThrows(
    () =>
      openModel(i8LinearModel(({ tensors }) => {
        tensors[1] = { name: "m.s", dtype: "F16", shape: [3, 1], data: new Uint8Array(6) };
      })),
    ContainerError,
    "F32 が必要",
  );
  // rank が重みと違う（[3] は keepdim 形ではない）
  assertThrows(
    () =>
      openModel(i8LinearModel(({ tensors }) => {
        tensors[1] = { name: "m.s", dtype: "F32", shape: [3], data: f32Bytes([1, 1, 1]) };
      })),
    ContainerError,
    "rank",
  );
  // broadcast できない（軸 0 が 2 で重みの 3 と違う）
  assertThrows(
    () =>
      openModel(i8LinearModel(({ tensors }) => {
        tensors[1] = { name: "m.s", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 1]) };
      })),
    ContainerError,
    "broadcast できない",
  );
  // 実テンソルとの名前衝突（別の initializer の実体を scale として読む形）
  assertThrows(
    () =>
      openModel(i8LinearModel(({ graph }) => {
        graph.initializers["w"].storage = { dtype: "i8", scale: "m.b" };
      })),
    ContainerError,
    "実体と同じキー",
  );
});

// group 量子化を受理する格納は i4 だけ（ADR 0069 決定 2）— i8 に付いた group_size は
// 従来どおり capability 不足で落とす（group ごとの scale を per-channel として読む沈黙誤値）。
Deno.test("i4 以外の格納 dtype に付いた group_size は capability 不足で落とす（ADR 0069）", () => {
  const model = openModel(i8LinearModel(({ graph }) => {
    graph.initializers["w"].storage = { dtype: "i8", scale: "m.s", group_size: 32 };
  }));
  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, RUNTIME_SUPPORT),
    ContainerError,
    "capability 不足",
  );
  assertEquals(error.message.includes("非対応 group 量子化 (1): w"), true, error.message);
  assertEquals(error.message.includes("i4 のみ"), true, error.message);
});

Deno.test("bf16 は従来どおり capability 不足で fail loudly（i8 の門が開いても変わらない）", () => {
  const graph = linearGraph({ dtype: "bf16" });
  const model = openModel(buildSafetensors([
    { name: "m.b", dtype: "F32", shape: [3], data: new Uint8Array(12) },
    { name: "m.w", dtype: "BF16", shape: [3, 4], data: new Uint8Array(24) },
  ], { karume_ir: JSON.stringify(graph) }));
  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, RUNTIME_SUPPORT),
    ContainerError,
    "capability 不足",
  );
  assertEquals(error.message.includes("非対応 格納 dtype 'bf16' (1): w"), true, error.message);
  // i8 は同じ門を通る（適格かどうかは実行可否と別軸）
  assertRuntimeSupport(openModel(i8LinearModel()).graph, RUNTIME_SUPPORT);
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
  /** per-channel scale のチャネル軸（conv_transpose1d だけ 1）。 */
  readonly channelAxis: number;
  /** bias は**常に f32 格納**（ADR 0006）。embedding のように bias が無い op は undefined。 */
  readonly bias?: FilledTensor;
  readonly outShape: readonly number[];
  readonly attrs?: Record<string, unknown>;
};

/** weight を i8 initializer にした単一ノードのグラフ + 配布形バイト列。 */
const weightedModel = (
  testCase: WeightedCase,
  quantized: ReturnType<typeof quantizeI8>,
): ArrayBuffer => {
  const values: GraphJson["values"] = {
    w: { dtype: "f32", shape: [...testCase.weight.shape] },
    y: { dtype: "f32", shape: [...testCase.outShape] },
  };
  const initializers: GraphJson["initializers"] = {
    w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
  };
  // MUST: I8（1 バイト要素）は**ファイル末尾**に置く。後続の F32 の絶対 offset が 4 の倍数から
  // 外れてリーダの整列検査で落ちる（格納の並べ方の制約であって i8 経路の問題ではない）。
  const tensors: TensorSpec[] = [];
  if (testCase.bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...testCase.bias.shape] };
    initializers["b"] = { tensor: "m.b", storage: { dtype: "f32" } };
    tensors.push({
      name: "m.b",
      dtype: "F32",
      shape: [...testCase.bias.shape],
      data: new Uint8Array(testCase.bias.data.buffer.slice(0)),
    });
  }
  tensors.push({
    name: "m.s",
    dtype: "F32",
    shape: [...quantized.scaleShape],
    data: f32Bytes(quantized.scale),
  });
  tensors.push({
    name: "m.w",
    dtype: "I8",
    shape: [...testCase.weight.shape],
    data: quantized.bytes,
  });
  // ins の並びは契約どおり（embedding は weight が先頭・他は x の次）
  const ins = testCase.op === "embedding" ? ["w", ...testCase.inputs.map(([name]) => name)] : [
    ...testCase.inputs.map(([name]) => name),
    "w",
    ...(testCase.bias === undefined ? [] : ["b"]),
  ];
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
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
  return buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
};

/**
 * 5 変種の数値ケース。**重みの総要素数が全て 4 の倍数でない**（ゼロ詰め経路）で、
 * **重み行の長さも 4 の倍数でない**（語とレーンの選択が平坦添字であることの検出器）。
 */
const WEIGHTED_CASES: readonly WeightedCase[] = [
  {
    // k = 7（4 の倍数でない）/ 重み 3·7 = 21 要素（4 の倍数でない）
    name: "linear [3,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [["x", fill([3, 7], SIGNED)]],
    weight: fill([3, 7], POSITIVE),
    channelAxis: 0,
    bias: fill([3], SIGNED),
    outShape: [3, 3],
  },
  {
    // Cin·K = 9 / 重み 5·3·3 = 45 要素
    name: "conv1d [1,3,9] * W[5,3,3] stride=1 padding=1",
    op: "conv1d",
    inputs: [["x", fill([1, 3, 9], SIGNED)]],
    weight: fill([5, 3, 3], POSITIVE),
    channelAxis: 0,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 9],
    attrs: { stride: 1, padding: 1, dilation: 1, groups: 1 },
  },
  {
    // Cin·Kh·Kw = 9 / 重み 5·3·3·1 = 45 要素。Kh ≠ Kw で軸取り違えも赤にする
    name: "conv2d [1,3,5,4] * W[5,3,3,1]",
    op: "conv2d",
    inputs: [["x", fill([1, 3, 5, 4], SIGNED)]],
    weight: fill([5, 3, 3, 1], POSITIVE),
    channelAxis: 0,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 5, 4],
    attrs: { stride: [1, 1], padding: [1, 0], dilation: [1, 1], groups: 1 },
  },
  {
    // K = 3 / 重み 3·5·3 = 45 要素。**チャネル軸は 1**（[Cin,Cout,K] の転置レイアウト）で、
    // Cin ≠ Cout なので軸を 0 と取り違えると shape 検査で赤くなる。
    name: "conv_transpose1d [1,3,7] * W[3,5,3] stride=1 padding=1",
    op: "conv_transpose1d",
    inputs: [["x", fill([1, 3, 7], SIGNED)]],
    weight: fill([3, 5, 3], POSITIVE),
    channelAxis: 1,
    bias: fill([5], SIGNED),
    outShape: [1, 5, 7],
    attrs: { stride: 1, padding: 1 },
  },
  {
    // H = 3 / 重み 5·3 = 15 要素
    name: "embedding W[5,3] × index [2,4]",
    op: "embedding",
    inputs: [["x", fill([2, 4], (i) => (i * 3) % 5, "i32")]],
    weight: fill([5, 3], SIGNED),
    channelAxis: 0,
    outShape: [2, 4, 3],
    attrs: { padding_idx: -1 },
  },
];

Deno.test({
  name: "w=i8 変種 5 種が CPU 参照（丸め後の重み）と一致する（行長 4 の倍数でない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of WEIGHTED_CASES) {
        const quantized = quantizeI8(
          testCase.weight.data,
          testCase.weight.shape,
          testCase.channelAxis,
        );
        // 総要素数が 4 の倍数でない = 生バイト長が 4 の倍数でない（ゼロ詰めが要る形）
        assert(
          quantized.bytes.byteLength % 4 !== 0,
          `${testCase.name}: 重みの総要素数が 4 の倍数（ゼロ詰め経路を踏まない）`,
        );
        const session = await createSession(gpu, openModel(weightedModel(testCase, quantized)));
        let output: Tensor;
        let residentBytes: number;
        try {
          const named: Record<string, Tensor> = {};
          for (const [name, tensor] of testCase.inputs) named[name] = tensor;
          output = (await session.run(named))["y"];
          residentBytes = session.diagnostics().storage.residentCompressedBytes;
        } finally {
          await session.dispose();
        }
        // 圧縮のまま常駐している（= CPU 展開に落ちていない）。ゼロ詰めぶんと scale を含む。
        const padded = quantized.bytes.byteLength + (4 - (quantized.bytes.byteLength % 4));
        assertEquals(
          residentBytes,
          padded + quantized.scale.byteLength,
          `${testCase.name}: GPU 常駐圧縮バイト数（scale 込み）`,
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
 * linear の GEMM タイル境界（64×64 レジスタタイル — src/kernels/gemm.ts）を i8 重みで踏む。
 *
 * MUST: **v4 経路とスカラ経路を対で持ち、どちらも n / k を違う長さ**にする。scale は
 * **出力チャネルごと**（`wscale[wcol]`）で、別の軸（行 / k / 出力列以外）から引く誤りは
 * 「チャネルごとに scale が違う」形でしか値に出ない。quantizeI8 は行ごとに最大値から
 * scale を決めるので、行ごとに値の違う重みを入れれば scale も行ごとに違う。
 * MUST: v4 経路でも scale は**成分ごとの f32 乗算**（`vec4 * scalar`）— 縮約の外へ出すと
 * ADR 0019 の「(Σ x·q)·s は CPU 展開とのビット一致を失う」に触れる。
 */
Deno.test({
  name: "w=i8 の linear が GEMM タイル境界（v4 / スカラ）で CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 行（= 出力チャネル）ごとに大きさが違う重み。POSITIVE のままだと各行の最大値が
    // 一致して scale が全チャネル同じになり、軸の取り違えが数値に出ない。
    const rowVarying = (cols: number) => (i: number): number =>
      POSITIVE(i) * (1 + (Math.floor(i / cols) % 7) * 0.25);
    const cases: readonly WeightedCase[] = [
      {
        // k=20（4 の倍数・16 の倍数でない）/ n=68（最終タイルの有効 quad が 16 中 1）/
        // m=65（行タイル 2 枚）。m/n/k は互いに違う長さで、scale は 68 チャネルぶん。
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], rowVarying(20)),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        // n=19（4 の倍数でない → スカラ経路・行内の 4 剰余の罠を踏む形）/ k=20 / m=17
        name: "linear スカラ [17,20] × W[19,20] + b[19]",
        op: "linear",
        inputs: [["x", fill([17, 20], SIGNED)]],
        weight: fill([19, 20], rowVarying(20)),
        channelAxis: 0,
        bias: fill([19], SIGNED),
        outShape: [17, 19],
      },
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const quantized = quantizeI8(
          testCase.weight.data,
          testCase.weight.shape,
          testCase.channelAxis,
        );
        // scale がチャネルごとに違う（同じなら軸の取り違えが数値に出ない — 恒真化の門）
        assert(
          new Set(quantized.scale).size > 1,
          `${testCase.name}: scale が全チャネルで同じ`,
        );
        const session = await createSession(gpu, openModel(weightedModel(testCase, quantized)));
        let output: Tensor;
        try {
          output = (await session.run({ x: testCase.inputs[0][1] }))["y"];
        } finally {
          await session.dispose();
        }
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
 * GPU の `unpack4xI8` + scale 乗算と CPU の {@link decodeI8} が**全 256 値 × 4 レーン位置で
 * ビット一致**する（ADR 0019 の検証条件）。
 *
 * 運び方: embedding の weight を [256, 4] の i8 とし、行 r の 4 要素すべてに値 `r − 128` を
 * 置く。平坦添字は `r·4 + col` なので、同じ値が 4 つのレーン位置すべてに現れる。embedding は
 * 行 gather で算術を 1 つも挟まないので、出力は dequant 結果そのもの。
 *
 * scale は行ごとに違う非 2 冪の値にする（scale を掛け忘れる誤り・行を取り違える誤りの検出器で、
 * 同時に「f32 乗算の丸め 1 回」がホストと一致することの検証になる）。
 */
Deno.test({
  name: "GPU の unpack4xI8 と CPU 展開が全 256 値 × 4 レーンでビット一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const vocab = 256;
    const hidden = 4;
    const quantized: number[] = [];
    for (let row = 0; row < vocab; row += 1) {
      for (let col = 0; col < hidden; col += 1) quantized.push(row - 128);
    }
    const bytes = i8BytesFrom(quantized);
    const scale = Float32Array.from({ length: vocab }, (_, row) => 0.1 + row * 0.003);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["embedding"] },
      symbols: [],
      inputs: [{ name: "index", dtype: "i32", shape: [vocab] }],
      outputs: ["y"],
      initializers: { w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } } },
      values: {
        w: { dtype: "f32", shape: [vocab, hidden] },
        y: { dtype: "f32", shape: [vocab, hidden] },
      },
      nodes: [{ op: "embedding", ins: ["w", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
    };
    const model = buildSafetensors([
      { name: "m.s", dtype: "F32", shape: [vocab, 1], data: f32Bytes(scale) },
      { name: "m.w", dtype: "I8", shape: [vocab, hidden], data: bytes },
    ], { karume_ir: JSON.stringify(graph) });
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(model));
    let actual: Float32Array<ArrayBuffer>;
    try {
      const outputs = await session.run({ index: fill([vocab], (i) => i, "i32") });
      actual = outputs["y"].data as Float32Array<ArrayBuffer>;
      // 圧縮のまま常駐していなければ「GPU の展開」を見ていない
      assertEquals(
        session.diagnostics().storage.residentCompressedBytes,
        bytes.byteLength + scale.byteLength,
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
    const expected = decodeI8(bytes, [vocab, hidden], scale, [vocab, 1]);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    const seen = new Set<string>();
    for (let i = 0; i < expected.length; i += 1) {
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `q=${quantized[i]} lane=${i & 3}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
      seen.add(`${quantized[i]}:${i & 3}`);
    }
    // 網羅の証拠（256 値 × 4 レーンを実際に踏んでいる）
    assertEquals(seen.size, 256 * 4, "値 × レーンの網羅数");
  },
});

/**
 * 変種経路の full-write（ADR 0014）。プール再利用バッファに毒値を仕込んでから i8 変種を
 * 実行し、1 語も残らないことを見る（仕込み方は tests/gpu_full_write_test.ts と同じ）。
 */
const poisonedI8Graph = (
  op: string,
  node: { readonly ins: readonly string[]; readonly attrs: Record<string, unknown> },
  inputs: readonly (readonly [string, FilledTensor])[],
  weightShape: readonly number[],
  outShape: readonly number[],
  bias?: readonly number[],
): GraphJson => {
  const count = outShape.reduce((total, dim) => total * dim, 1);
  const values: GraphJson["values"] = {
    poison: { dtype: "f32", shape: [count] },
    w: { dtype: "f32", shape: [...weightShape] },
    y: { dtype: "f32", shape: [...outShape] },
  };
  const initializers: GraphJson["initializers"] = {
    w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
  };
  if (bias !== undefined) {
    values["b"] = { dtype: "f32", shape: [...bias] };
    initializers["b"] = { tensor: "m.b", storage: { dtype: "f32" } };
  }
  return {
    format: "karume-ir",
    version: 1,
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
  name: "w=i8 変種もプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // linear（タイル形）と embedding（grid-stride + 範囲外分岐）の 2 本で踏む
      const cases: readonly {
        readonly name: string;
        readonly graph: GraphJson;
        readonly tensors: readonly TensorSpec[];
        readonly inputs: Record<string, Tensor>;
        readonly count: number;
      }[] = [
        (() => {
          const quantized = quantizeI8(fill([3, 7], POSITIVE).data, [3, 7], 0);
          const bias = fill([3], SIGNED);
          return {
            name: "linear",
            graph: poisonedI8Graph(
              "linear",
              { ins: ["x", "w", "b"], attrs: {} },
              [["x", fill([4, 7], SIGNED)]],
              [3, 7],
              [4, 3],
              [3],
            ),
            // I8 は末尾（後続 F32 の整列が崩れるため — weightedModel の MUST）
            tensors: [
              {
                name: "m.b",
                dtype: "F32",
                shape: [3],
                data: new Uint8Array(bias.data.buffer.slice(0)),
              },
              { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes(quantized.scale) },
              { name: "m.w", dtype: "I8", shape: [3, 7], data: quantized.bytes },
            ],
            inputs: { x: fill([4, 7], SIGNED) },
            count: 12,
          };
        })(),
        (() => {
          const quantized = quantizeI8(fill([5, 3], SIGNED).data, [5, 3], 0);
          return {
            name: "embedding",
            graph: poisonedI8Graph(
              "embedding",
              { ins: ["w", "idx"], attrs: { padding_idx: -1 } },
              [["idx", fill([4], (i) => i % 5, "i32")]],
              [5, 3],
              [4, 3],
            ),
            tensors: [
              { name: "m.s", dtype: "F32", shape: [5, 1], data: f32Bytes(quantized.scale) },
              { name: "m.w", dtype: "I8", shape: [5, 3], data: quantized.bytes },
            ],
            inputs: { idx: fill([4], (i) => i % 5, "i32") },
            count: 12,
          };
        })(),
      ];
      for (const testCase of cases) {
        const model = openModel(
          buildSafetensors(testCase.tensors, { karume_ir: JSON.stringify(testCase.graph) }),
        );
        const session = await createSession(gpu, model);
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
 * 適格外（混在消費）は CPU で f32 展開され、**適格経路とビット単位で同じ値**を出す
 * （ADR 0019 の「要素ごと dequant」がこの一致の根拠）。
 *
 * MUST: 2 経路の出力を突き合わせる。片方だけを CPU 参照と比べると、両経路が同じだけ
 * ずれている形（例: 展開が誤っていて GPU 側も同じ誤りを持つ）を見逃す。
 */
Deno.test({
  name: "混在消費の i8 は CPU 展開へ落ち、適格経路とビット単位で同じ値を出す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const weight = fill([3, 4], POSITIVE);
    const quantized = quantizeI8(weight.data, [3, 4], 0);
    const bias = fill([3], SIGNED);
    const tensors: readonly TensorSpec[] = [
      { name: "m.b", dtype: "F32", shape: [3], data: new Uint8Array(bias.data.buffer.slice(0)) },
      { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes(quantized.scale) },
      { name: "m.w", dtype: "I8", shape: [3, 4], data: quantized.bytes },
    ];
    const x = fill([2, 4], SIGNED);
    const run = async (
      gpu: GpuContext,
      graph: GraphJson,
    ): Promise<{ readonly y: Tensor; readonly resident: number; readonly expanded: number }> => {
      const session = await createSession(
        gpu,
        openModel(buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) })),
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
      const storage = { dtype: "i8", scale: "m.s" };
      const eligible = await run(gpu, linearGraph(storage));
      // w を add でも消費する = 混在消費（同じ linear ノードは残す）
      const mixed = await run(
        gpu,
        linearGraph(
          storage,
          [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
          { z: { dtype: "f32", shape: [3, 4] } },
          ["z"],
        ),
      );
      // 適格側: 12 バイト（4 の倍数でゼロ詰め無し）+ scale 3 要素 12 バイト
      assertEquals(eligible.resident, 12 + 12, "適格側は圧縮のまま常駐（scale 込み）");
      assertEquals(eligible.expanded, 0);
      assertEquals(mixed.resident, 0, "混在消費は 1 バイトも圧縮常駐しない");
      assertEquals(mixed.expanded, 48, "CPU 展開バイト数（f32 換算 12 要素）");
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
 * initializer 自身をグラフ出力にした形（IR が許す — format/ir.ts の定義済み検査）。
 *
 * readback は宣言 dtype の semantic f32 を 4 バイト / 要素で重みバッファから写すので、i8 の
 * まま常駐していると copy が実バッファ（1 バイト / 要素）をはみ出す。適格判定が
 * `graph.outputs` を見ていれば CPU 展開へ落ち、丸め後の重みがそのまま読める。
 */
Deno.test({
  name:
    "graph 出力にした i8 initializer は CPU 展開へ落ち、丸め後の値が f32 として読める（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const weight = fill([3, 4], POSITIVE);
    const quantized = quantizeI8(weight.data, [3, 4], 0);
    const bias = fill([3], SIGNED);
    const model = openModel(buildSafetensors([
      { name: "m.b", dtype: "F32", shape: [3], data: new Uint8Array(bias.data.buffer.slice(0)) },
      { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes(quantized.scale) },
      { name: "m.w", dtype: "I8", shape: [3, 4], data: quantized.bytes },
    ], {
      karume_ir: JSON.stringify(linearGraph({ dtype: "i8", scale: "m.s" }, [], {}, ["w"])),
    }));
    const x = fill([2, 4], SIGNED);
    const gpu = await acquireGpu();
    const session = await createSession(gpu, model);
    try {
      const storage = session.diagnostics().storage;
      assertEquals(storage.residentCompressedBytes, 0, "グラフ出力の重みは圧縮常駐しない");
      assertEquals(storage.hostExpandedBytes, 48, "CPU 展開バイト数（f32 換算 12 要素）");
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
 * 診断（ADR 0006 の常設義務）の検算。適格・適格外・非圧縮を 1 つのグラフに混ぜ、
 * 両方の数字が**実バイト数**と一致することを見る。
 *
 * bias を i8 で宣言しても**必ず CPU 展開**に落ちる（bias スロットは適格判定に載らない —
 * プロトタイプの降格バグの逆）。
 */
Deno.test({
  name: "診断は適格 / 適格外 / 非圧縮を実バイト数で区別する（scale 込み・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 重み 3·7 = 21 要素（4 の倍数でない → ゼロ詰めで 24 バイト）/ bias 3 要素を **i8 宣言**
    const weight = quantizeI8(fill([3, 7], POSITIVE).data, [3, 7], 0);
    const bias = quantizeI8(fill([3], SIGNED).data, [3], 0);
    const scale = fill([3], POSITIVE);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["linear", "mul"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [2, 7] }],
      outputs: ["y"],
      initializers: {
        w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.ws" } },
        b: { tensor: "m.b", storage: { dtype: "i8", scale: "m.bs" } },
        s: { tensor: "m.s", storage: { dtype: "f32" } },
      },
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
    const model = openModel(buildSafetensors([
      { name: "m.ws", dtype: "F32", shape: [3, 1], data: f32Bytes(weight.scale) },
      { name: "m.bs", dtype: "F32", shape: [3], data: f32Bytes(bias.scale) },
      { name: "m.s", dtype: "F32", shape: [3], data: new Uint8Array(scale.data.buffer.slice(0)) },
      { name: "m.b", dtype: "I8", shape: [3], data: bias.bytes },
      { name: "m.w", dtype: "I8", shape: [3, 7], data: weight.bytes },
    ], { karume_ir: JSON.stringify(graph) }));
    const gpu = await acquireGpu();
    const session = await createSession(gpu, model);
    try {
      const storage = session.diagnostics().storage;
      // 適格: 21 バイト → 4 バイト整列で 24、+ scale 3 要素 12 バイト
      assertEquals(weight.bytes.byteLength, 21);
      assertEquals(storage.residentCompressedBytes, 24 + 12);
      // 適格外: bias 3 要素を f32 展開 = 12 バイト（f32 格納の s は**どちらにも入らない**。
      // bias の scale は GPU に上がらないので常駐にも展開にも数えない）
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
 * 適格経路の scale は**平坦添字で引ける形**でなければならない（ADR 0019）。broadcast 可能
 * なだけの形（重み `[3,4]` に対する `[1,4]`）は openModel を通るが、カーネルは `wscale[col]` と
 * 読むので沈黙誤値になる — Session 構築で落ちることを固定する。
 */
Deno.test({
  name: "チャネル軸と食い違う scale は Session 構築で fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const model = openModel(i8LinearModel(({ tensors }) => {
      // 重み [3,4] の軸 1 に沿った scale。broadcast は可能だがチャネル軸（0）ではない。
      tensors[1] = { name: "m.s", dtype: "F32", shape: [1, 4], data: f32Bytes([1, 1, 1, 1]) };
    }));
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSession(gpu, model),
        ExecutionError,
        "keepdim 形でない",
      );
    } finally {
      gpu.destroy();
    }
  },
});
