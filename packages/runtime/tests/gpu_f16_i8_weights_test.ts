// Weight storages shared by the five fused kernels: f16 (ADR 0018), bf16 (ADR 0131) and per-channel
// i8 (ADR 0019). The storages mirror each other test-for-test, so the mirrored tests run once per
// entry of STORAGES (each keeps its own title); storage-specific parts (host decoding, the exhaustive
// unpack check, the i8 channel axis and rowAxis admission) stay separate. bf16 shares f16's packing
// (two elements per u32 word, the same odd-length traps) and differs only in the unpack (a bitcast).
//
// f16 格納の実行経路（ADR 0018）— 適格判定・ロード経路・カーネル変種・診断の通し検証。
// i8（per-channel symmetric int8）格納の実行経路（ADR 0019）— ロード時検証・適格判定の共用・
// カーネル変種・診断の通し検証。
//
// この波で入った 2 経路を両方踏む:
//   適格（消費が WEIGHT_SLOTS に載った op の weight スロットだけ）→ 生バイトのまま GPU
//     常駐し、dequant はカーネル内（`unpack2x16float`）
//   適格外（bias / 混在消費 / その他）→ ロード時に CPU で f32 展開（VRAM 削減ゼロ）
// (i8 dequantizes in the kernel with `unpack4xI8` + the per-channel scale.)
//
// MUST: 数値ケースは **in-features / K / Cin·Kh·Kw を奇数**にする。f16 は 2 要素を 1 語に
// 詰めるので、行長が偶数だと重み行の先頭が必ず語境界に来る — 「対の選択を行内の相対添字で
// 取る」誤りが偶数長では一切値に出ない（波Q1 の故障注入で実証）。
// MUST: 重みの総要素数も奇数にする（末尾 2 バイトのゼロ詰めが無いと writeBuffer が
// validation で落ちる経路をここで踏む）。
// MUST: 数値ケースは **重み行の長さを 4 の倍数にしない**。i8 は 4 要素を 1 語に詰めるので、
// 行長が 4 の倍数だと重み行の先頭が必ず語境界に来る — 「語とレーンの選択を行内の相対添字で
// 取る」誤りが 4 の倍数長では一切値に出ない（f16 の偶奇 MUST と同型の罠）。
// MUST: 重みの総要素数も 4 の倍数にしない（末尾ゼロ詰めが無いと writeBuffer が validation で
// 落ちる経路をここで踏む）。
// MUST: 期待値は**丸め後の重み**（fake-quant — ADR 0006）で作る。丸め前の f32 で比較すると
// 量子化誤差と実装誤差が混ざり、tolerance を緩める圧力になる。

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type CodecName, perChannelGroupSize } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { bf16BitsToF32, decodeBf16 } from "../src/format/bf16.ts";
import { decodeF16, f16BitsToF32 } from "../src/format/f16.ts";
import { alignI8Payload, decodeI8, I8Error } from "../src/format/i8.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { RUNTIME_SUPPORT, WEIGHT_SLOTS } from "../src/ops.ts";
import { assertRuntimeSupport } from "../src/ops/support.ts";
import {
  eligibleCompressedInitializers,
  ExecutionError,
  weightChannelAxes,
} from "../src/runtime/plan.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./helpers/container-write.ts";
import { bf16BytesFromBits, quantizeBf16 } from "./helpers/bf16.ts";
import { f16BytesFromBits, quantizeF16 } from "./helpers/f16.ts";
import { i8BytesFrom, quantizeI8 } from "./helpers/i8.ts";
import { mergeGraph, mergeTensors } from "./helpers/merged-graph.ts";
import {
  type DeclarationJson,
  f32Bytes,
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
 * `fallback` replaces that default for the other storage (the i8 run loads every unnamed
 * initializer, bias included, as int8-sym).
 */
const mergedFor = (
  declaration: DeclarationJson,
  codecs: Readonly<Record<string, CodecName>> = {},
  fallback: CodecName = "f16",
): IrGraph =>
  mergeGraph(
    declaration,
    Object.fromEntries(
      Object.keys(declaration.initializers).map((name) => [name, codecs[name] ?? fallback]),
    ),
  );

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

/** A weight rounded to one storage: the bytes to load and the values the CPU reference uses. */
type QuantizedWeight = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** 量子化後の値（fake-quant — CPU 参照はこちらを重みとして使う）。 */
  readonly values: Float32Array<ArrayBuffer>;
  readonly encoding: EncodingInput;
  /** The per-channel scale (i8 only); it is resident next to the payload. */
  readonly scale?: Float32Array<ArrayBuffer>;
};

/** One storage under test, with the expectations that differ between the storages. */
type WeightStorageCase = {
  readonly storage: "f16" | "bf16" | "i8";
  readonly codec: CodecName;
  readonly quantize: (weight: FilledTensor, channelAxis: number) => QuantizedWeight;
  /** linear cases on the GEMM tile edges (v4 path and scalar path). */
  readonly tileCases: readonly WeightedCase[];
  /** Resident bytes of the [3,4] linear weight on the eligible path (literal oracle). */
  readonly mixedResidentBytes: number;
  /** Payload / resident bytes of the [3,7] weight in the diagnostics graph (literal oracles). */
  readonly diagnostics: { readonly payloadBytes: number; readonly residentBytes: number };
  readonly titles: {
    readonly eligibility: string;
    readonly graphOutputEligibility: string;
    readonly variants: string;
    readonly tiles: string;
    readonly poison: string;
    readonly mixed: string;
    readonly diagnostics: string;
    readonly graphOutput: string;
  };
};

// 行（= 出力チャネル）ごとに大きさが違う重み。POSITIVE のままだと各行の最大値が
// 一致して scale が全チャネル同じになり、軸の取り違えが数値に出ない。
const rowVarying =
  (cols: number) =>
  (i: number): number =>
    POSITIVE(i) * (1 + (Math.floor(i / cols) % 7) * 0.25);

const STORAGES: readonly WeightStorageCase[] = [
  {
    storage: "f16",
    codec: "f16",
    quantize: (weight) => {
      const quantized = quantizeF16(weight.data);
      return { ...quantized, encoding: { codec: "f16" } };
    },
    tileCases: [
      {
        // k=20（4 の倍数・16 の倍数でない）/ n=68（最終タイルの有効 quad が 16 中 1）/
        // m=65（行タイル 2 枚）。m/n/k は互いに違う長さ。
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], POSITIVE),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        // k=37（2 の倍数でない = 平坦添字の偶奇の検出器）/ n=23 / m=70（行タイル 2 枚）
        name: "linear スカラ [70,37] × W[23,37] + b[23]",
        op: "linear",
        inputs: [["x", fill([70, 37], SIGNED)]],
        weight: fill([23, 37], POSITIVE),
        channelAxis: 0,
        bias: fill([23], SIGNED),
        outShape: [70, 23],
      },
    ],
    // 12 elements × 2 bytes, already 4-byte aligned.
    mixedResidentBytes: 24,
    // 適格: 21 要素 × 2 バイト = 42 → 4 バイト整列で 44
    diagnostics: { payloadBytes: 42, residentBytes: 44 },
    titles: {
      eligibility: "適格判定は WEIGHT_SLOTS の op の weight スロット消費だけを通す",
      graphOutputEligibility: "graph 出力になった initializer は weight スロット消費だけでも適格外",
      variants: "w=f16 変種 5 種が CPU 参照（丸め後の重み）と一致する（奇数長・実 GPU）",
      tiles: "w=f16 の linear が GEMM タイル境界（v4 / スカラ）で CPU 参照と一致する（実 GPU）",
      poison: "w=f16 変種もプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
      mixed: "混在消費の f16 は CPU 展開へ落ち、適格経路とビット単位で同じ値を出す（実 GPU）",
      diagnostics: "診断は適格 / 適格外 / 非圧縮を実バイト数で区別する（実 GPU）",
      graphOutput:
        "graph 出力にした f16 initializer は CPU 展開へ落ち、丸め後の値が f32 として読める（実 GPU）",
    },
  },
  {
    // bf16 = f32 の上位 16 bit（ADR 0131）。詰め方は f16 と同じ（2 要素 / 語）なので、奇数長の罠と
    // 整列の詰め物もそのまま共有する。丸めは f32 の下位 16 bit を落とす RNE（helpers/bf16.ts）。
    storage: "bf16",
    codec: "bf16",
    quantize: (weight) => {
      const quantized = quantizeBf16(weight.data);
      return { ...quantized, encoding: { codec: "bf16" } };
    },
    tileCases: [
      {
        // f16 と同じ形（k=20 の v4 経路 / k=37 のスカラ経路 — 偶奇の検出器）
        name: "linear v4 [65,20] × W[68,20] + b[68]",
        op: "linear",
        inputs: [["x", fill([65, 20], SIGNED)]],
        weight: fill([68, 20], POSITIVE),
        channelAxis: 0,
        bias: fill([68], SIGNED),
        outShape: [65, 68],
      },
      {
        name: "linear スカラ [70,37] × W[23,37] + b[23]",
        op: "linear",
        inputs: [["x", fill([70, 37], SIGNED)]],
        weight: fill([23, 37], POSITIVE),
        channelAxis: 0,
        bias: fill([23], SIGNED),
        outShape: [70, 23],
      },
    ],
    // 12 elements × 2 bytes, already 4-byte aligned.
    mixedResidentBytes: 24,
    // 適格: 21 要素 × 2 バイト = 42 → 4 バイト整列で 44
    diagnostics: { payloadBytes: 42, residentBytes: 44 },
    titles: {
      eligibility: "bf16 の適格判定は f16 と同じ 1 本の判定を通る（新設していない）",
      graphOutputEligibility: "graph 出力になった bf16 initializer も同じ 1 本の判定で適格外",
      variants: "w=bf16 変種 5 種が CPU 参照（丸め後の重み）と一致する（奇数長・実 GPU）",
      tiles: "w=bf16 の linear が GEMM タイル境界（v4 / スカラ）で CPU 参照と一致する（実 GPU）",
      poison: "w=bf16 変種もプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
      mixed: "混在消費の bf16 は CPU 展開へ落ち、適格経路とビット単位で同じ値を出す（実 GPU）",
      diagnostics: "診断は適格 / 適格外 / 非圧縮を実バイト数で区別する（bf16・実 GPU）",
      graphOutput:
        "graph 出力にした bf16 initializer は CPU 展開へ落ち、丸め後の値が f32 として読める（実 GPU）",
    },
  },
  {
    storage: "i8",
    codec: "int8-sym",
    quantize: (weight, channelAxis) => {
      const quantized = quantizeI8(weight.data, weight.shape, channelAxis);
      // per-channel の groupSize は台帳（`codecs.ts`）から引く（行長 = numel / shape[rowAxis]・
      // groups は 1）— テスト側で数え直すと、供給の長さと期待の長さが同じ誤りを共有する。
      const numel = weight.shape.reduce((count, dim) => count * dim, 1);
      return {
        bytes: quantized.bytes,
        values: quantized.values,
        scale: quantized.scale,
        encoding: {
          codec: "int8-sym",
          rowAxis: channelAxis === 1 ? 1 : 0,
          groupSize: perChannelGroupSize(numel / weight.shape[channelAxis]),
          scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
        },
      };
    },
    tileCases: [
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
    ],
    // 適格側: 12 バイト（4 の倍数でゼロ詰め無し）+ scale 3 要素 12 バイト
    mixedResidentBytes: 12 + 12,
    // 適格: 21 バイト → 4 バイト整列で 24、+ scale 3 要素 12 バイト
    diagnostics: { payloadBytes: 21, residentBytes: 24 + 12 },
    titles: {
      eligibility: "i8 の適格判定は f16 と同じ 1 本の判定を通る（新設していない）",
      graphOutputEligibility: "graph 出力になった i8 initializer も同じ 1 本の判定で適格外",
      variants:
        "w=i8 変種 5 種が CPU 参照（丸め後の重み）と一致する（行長 4 の倍数でない・実 GPU）",
      tiles: "w=i8 の linear が GEMM タイル境界（v4 / スカラ）で CPU 参照と一致する（実 GPU）",
      poison: "w=i8 変種もプール再利用バッファの毒値を 1 語も残さない（full-write / 実 GPU）",
      mixed: "混在消費の i8 は CPU 展開へ落ち、適格経路とビット単位で同じ値を出す（実 GPU）",
      diagnostics: "診断は適格 / 適格外 / 非圧縮を実バイト数で区別する（scale 込み・実 GPU）",
      graphOutput:
        "graph 出力にした i8 initializer は CPU 展開へ落ち、丸め後の値が f32 として読める（実 GPU）",
    },
  },
];

// ---------------------------------------------------------------------------
// 適格判定（GPU 非依存 — グラフ構造だけで決まる）
// ---------------------------------------------------------------------------

for (const storage of STORAGES) {
  test(storage.titles.eligibility, () => {
    const eligible = (declaration: DeclarationJson): readonly string[] =>
      [...eligibleCompressedInitializers(mergedFor(declaration, {}, storage.codec))].sort();

    // 単独の weight 消費は適格。bias は同じグラフにいても**絶対に適格にならない**
    // （プロトタイプの f16 降格バグの逆 — ADR 0006 が名指しした規則）。
    assertEquals(eligible(linearDeclaration()), ["w"]);

    // MUST: 同じ initializer が weight 以外でも消費されたら適格を失う（混在消費）。
    // 圧縮のまま上げると elementwise 側のカーネルが u32 を f32 として読む沈黙誤値になる。
    assertEquals(
      eligible(
        linearDeclaration(
          [{ op: "add", ins: ["w", "w"], outs: ["z"], attrs: {} }],
          { z: { dtype: "f32", shape: [3, 4] } },
          ["z"],
        ),
      ),
      [],
    );

    // 消費ゼロの initializer も適格外（実行に使われないバイトを「常駐圧縮」と数えない）
    const unused = linearDeclaration();
    unused.initializers["dead"] = {};
    unused.values["dead"] = { dtype: "f32", shape: [2] };
    assertEquals(eligible(unused), ["w"]);
  });
}

test("適格判定は WEIGHT_SLOTS の op それぞれの weight スロット位置を見る（bias / index は適格にしない）", () => {
  // op → (ins の並び, weight の位置)。embedding だけスロット 0（他は 1）。
  const cases: readonly (readonly [string, readonly string[], readonly string[]])[] = [
    ["linear", ["x", "w", "b"], ["w"]],
    ["conv1d", ["x", "w", "b"], ["w"]],
    ["conv2d", ["x", "w", "b"], ["w"]],
    ["conv3d", ["x", "w", "b"], ["w"]],
    ["conv_transpose1d", ["x", "w", "b"], ["w"]],
    ["embedding", ["w", "x"], ["w"]],
    // 重みスロットを持たない op はどのスロットも適格にしない
    ["layer_norm", ["x", "w", "b"], []],
    ["rms_norm", ["x", "w"], []],
    ["masked_fill", ["w", "x"], []],
  ];
  // 表が WEIGHT_SLOTS の全 op を覆うことを固定する（op を足したのに表が古いまま緑になるのを防ぐ）
  assertEquals(
    new Set(cases.filter(([, , expected]) => expected.length > 0).map(([op]) => op)),
    new Set(WEIGHT_SLOTS.keys()),
  );
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

for (const storage of STORAGES) {
  test(storage.titles.graphOutputEligibility, () => {
    const eligible = (declaration: DeclarationJson): readonly string[] =>
      [...eligibleCompressedInitializers(mergedFor(declaration, {}, storage.codec))].sort();

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
}

test("チャネル軸は消費側 op から決まる（conv_transpose1d だけ 1）", () => {
  const axisOf = (op: string, ins: readonly string[]): number | undefined => {
    const graph = {
      initializers: { w: { storage: { codec: "int8-sym", groupSize: 4, rowAxis: 0 } } },
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

test("同じ重みを軸の違う 2 op が食う形は fail loudly", () => {
  const graph = {
    initializers: { w: { storage: { codec: "int8-sym", groupSize: 4, rowAxis: 0 } } },
    nodes: [
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "conv_transpose1d", ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
    ],
  };
  assertThrows(
    () => weightChannelAxes(graph as never),
    ExecutionError,
    "consumers disagree on the channel axis",
  );
});

// ---------------------------------------------------------------------------
// 合流層の検証（宣言 × 供給 — GPU 非依存）
// ---------------------------------------------------------------------------

/** {@link linearDeclaration} の重み [3,4] を i8 量子化したもの（scale は行ごと 3 本）。 */
const LINEAR_W = quantizeI8(fill([3, 4], POSITIVE).data, [3, 4], 0);

/** {@link linearDeclaration} への i8 供給。`encoding` を差し替えて 1 点だけ壊せる。 */
const linearTensors = (encoding: EncodingInput): TensorInput[] => [
  { graph: GRAPH_NAME, initializer: "w", bytes: LINEAR_W.bytes, encoding },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes([0.5, -0.25, 1]),
    encoding: { codec: "f32" },
  },
];

// group 量子化を受理する codec は int4-sym-g だけ（ADR 0069 決定 2）— per-channel codec に
// 行長と違う groupSize を付けた形は合流層が落とす（group ごとの scale を per-channel として
// 読む沈黙誤値）。
test("per-channel codec に行長と違う groupSize を付けると合流で落とす（ADR 0069）", () => {
  const error = assertThrows(
    () =>
      mergeTensors(
        linearDeclaration(),
        linearTensors({
          codec: "int8-sym",
          groupSize: 2,
          scale: { bytes: f32Bytes(LINEAR_W.scale), dtype: "f32" },
        }),
      ),
    ContainerFormatError,
    "is per-channel, so groupSize MUST equal the row length 4",
  );
  assertEquals(error.message.includes("initializer 'w'"), true, error.message);
});

/**
 * f16 / bf16 / i8 は同じ capability 門を通る（適格かどうかは実行可否と別軸）。bf16 は ADR 0131 まで
 * 「宣言としては valid なまま実行できない」格納だったが、今は f16 と同じ経路で実行する。
 */
test("f16 / bf16 / i8 は capability 門を通る（適格かどうかは実行可否と別軸）", () => {
  for (const storage of STORAGES) {
    // 圧縮重み + f32 bias（`b` を省くと `mergedFor` の既定 f16 になり、fixture の意図が変わる）。
    assertRuntimeSupport(
      mergedFor(linearDeclaration(), { b: "f32" }, storage.codec),
      RUNTIME_SUPPORT,
    );
  }
});

/**
 * 適格経路の scale は**平坦添字で引ける形**でなければならない（ADR 0019）。broadcast 可能
 * なだけの形（重み `[3,4]` に対する `[1,4]`）はカーネルが `wscale[col]` と読むので沈黙誤値に
 * なる — 消費側 op の軸と突き合わせる admission（`planWeightResidency`）で落ちることを固定する。
 */
test("消費側の軸と食い違う rowAxis は admission（prepareContainer）で fail loudly", async () => {
  // 重み [3,4] の軸 1 を行と宣言した形。容器としては整合するが、消費側（linear）の
  // チャネル軸は 0 — 通すとカーネルが別の軸の scale を引く沈黙誤値になる。
  const opened = await openModelBytes(
    linearDeclaration(),
    linearTensors({
      codec: "int8-sym",
      rowAxis: 1,
      groupSize: 3,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
  );
  const error = assertThrows(
    () => prepareContainer(opened, GRAPH_NAME),
    ExecutionError,
    "the declared rowAxis 1 differs from the consuming op's channel axis 0",
  );
  assertEquals(error.message.includes("initializer 'w'"), true, error.message);
});

// ---------------------------------------------------------------------------
// CPU 側の展開（ホスト鏡像の仕様）
// ---------------------------------------------------------------------------

test("f16 → f32 の展開は ±0 / subnormal / ±Inf / NaN / 最大最小 normal を正しく扱う", () => {
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

test("bf16 → f32 の展開は上位 16 bit の写し（±0 / subnormal / ±Inf / NaN / 最大最小 normal）", () => {
  assert(Object.is(bf16BitsToF32(0x0000), 0), "+0");
  assert(Object.is(bf16BitsToF32(0x8000), -0), "-0");
  // subnormal: 最小（2^-133）と最大（2^-126 に 1 ulp 足りない）— f32 の subnormal にそのまま当たる
  assertEquals(bf16BitsToF32(0x0001), 2 ** -133);
  assertEquals(bf16BitsToF32(0x007f), 127 * 2 ** -133);
  // normal: 最小（2^-126）と最大（(2 - 2^-7) · 2^127）
  assertEquals(bf16BitsToF32(0x0080), 2 ** -126);
  assertEquals(bf16BitsToF32(0x7f7f), (2 - 2 ** -7) * 2 ** 127);
  assertEquals(bf16BitsToF32(0xff7f), -(2 - 2 ** -7) * 2 ** 127);
  assertEquals(bf16BitsToF32(0x3f80), 1);
  assertEquals(bf16BitsToF32(0xbf80), -1);
  assertEquals(bf16BitsToF32(0x7f80), Infinity);
  assertEquals(bf16BitsToF32(0xff80), -Infinity);
  assert(Number.isNaN(bf16BitsToF32(0x7fc0)), "quiet NaN");
  assert(Number.isNaN(bf16BitsToF32(0xff81)), "signaling NaN（符号付き）");
  // 展開はビットの写しなので、f32 → bf16 の往復（下位 16 bit が 0 の値）は恒等
  for (const bits of [0x0001, 0x007f, 0x0080, 0x7f7f, 0x3555]) {
    const value = bf16BitsToF32(bits);
    assertEquals(Math.fround(value), value, `bits=${bits}`);
    assertEquals(decodeBf16(bf16BytesFromBits([bits]))[0], value, `bits=${bits}`);
  }
});

test("decodeI8: 符号付き 8bit を per-channel scale で展開する（keepdim broadcast）", () => {
  // [2,3] の重み。チャネル軸 0 の scale [2,1] が行ごとに掛かる。
  const bytes = i8BytesFrom([127, -128, 0, 1, -1, 64]);
  const scale = Float32Array.from([0.5, 0.25]);
  assertEquals([...decodeI8(bytes, [2, 3], scale, [2, 1])], [63.5, -64, 0, 0.25, -0.25, 16]);
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

test("decodeI8: rank 3 以上でも軸ごとに正しく broadcast する", () => {
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

test("decodeI8: 展開の丸めは f32 の 1 回だけ（GPU の f32 乗算と同値）", () => {
  // 0.1 は f32 で丸められる値。q·s は f64 で厳密（7bit × 24bit）なので、f32 へ 1 度だけ
  // 丸めた結果が GPU の f32 乗算の正しい丸めと一致する。
  const scale = Float32Array.from([0.1]);
  const decoded = decodeI8(i8BytesFrom([3, 7, 127]), [3, 1], scale, [1, 1]);
  for (const [index, q] of [3, 7, 127].entries()) {
    assertEquals(decoded[index], Math.fround(q * scale[0]), `q=${q}`);
    assertEquals(Math.fround(decoded[index]), decoded[index], `q=${q}: f32 で厳密`);
  }
});

test("decodeI8: broadcast できない scale と要素数不一致を拒否する", () => {
  const bytes = i8BytesFrom([1, 2, 3, 4, 5, 6]);
  assertThrows(() => decodeI8(bytes, [2, 3], Float32Array.from([1, 2]), [2]), I8Error, "rank");
  // 軸 1 が 2（重みの 3 と違う）— 要素数は shape と整合しているので broadcast 検査が見る
  assertThrows(
    () => decodeI8(bytes, [2, 3], Float32Array.from([1, 2, 3, 4]), [2, 2]),
    I8Error,
    "cannot be broadcast",
  );
  assertThrows(
    () => decodeI8(bytes, [2, 4], Float32Array.from([1, 2]), [2, 1]),
    I8Error,
    "payload",
  );
});

test("alignI8Payload: 4 バイト境界までゼロ詰めし、整列済みはコピーしない", () => {
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
// GPU 実行（変種カーネル）
// ---------------------------------------------------------------------------

/** weight を圧縮 initializer にした単一ノードの宣言 + 開いた容器。 */
const weightedModel = (
  testCase: WeightedCase,
  quantized: QuantizedWeight,
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
    bytes: quantized.bytes,
    encoding: quantized.encoding,
  });
  // ins の並びは契約どおり（embedding は weight が先頭・他は x の次）
  const ins =
    testCase.op === "embedding"
      ? ["w", ...testCase.inputs.map(([name]) => name)]
      : [
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
  quantized: QuantizedWeight,
): Promise<{ readonly output: Tensor; readonly residentBytes: number }> => {
  const session = await createSessionFromContainer(
    gpu,
    await weightedModel(testCase, quantized),
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
 * 5 変種の数値ケース。**重みの総要素数が全て奇数かつ 4 の倍数でない**（ゼロ詰め経路）で、
 * **重み行の長さも奇数かつ 4 の倍数でない**（対の選択 / 語とレーンの選択が平坦添字である
 * ことの検出器）。
 */
const WEIGHTED_CASES: readonly WeightedCase[] = [
  {
    // k = 7（奇数）/ 重み 3·7 = 21 要素（奇数）
    name: "linear [3,7] × W[3,7] + b[3]",
    op: "linear",
    inputs: [["x", fill([3, 7], SIGNED)]],
    weight: fill([3, 7], POSITIVE),
    channelAxis: 0,
    bias: fill([3], SIGNED),
    outShape: [3, 3],
  },
  {
    // Cin·K = 9（奇数）/ 重み 5·3·3 = 45 要素（奇数）
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
    // Cin·Kh·Kw = 9（奇数）/ 重み 5·3·3·1 = 45 要素（奇数）。Kh ≠ Kw で軸取り違えも赤にする
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
    // K = 3（奇数・契約 2·padding == K − stride を満たす stride 1）/ 重み 3·5·3 = 45 要素
    // **チャネル軸は 1**（[Cin,Cout,K] の転置レイアウト）で、
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
    // H = 3（奇数）/ 重み 5·3 = 15 要素（奇数）
    name: "embedding W[5,3] × index [2,4]",
    op: "embedding",
    inputs: [["x", fill([2, 4], (i) => (i * 3) % 5, "i32")]],
    weight: fill([5, 3], SIGNED),
    channelAxis: 0,
    outShape: [2, 4, 3],
    attrs: { padding_idx: -1 },
  },
];

for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.variants, async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of WEIGHTED_CASES) {
        const quantized = storage.quantize(testCase.weight, testCase.channelAxis);
        // A payload that is not 4-byte aligned needs the zero padding (f16: an odd element
        // count; i8: an element count that is not a multiple of 4).
        assert(
          quantized.bytes.byteLength % 4 !== 0,
          `${testCase.name}: payload is 4-byte aligned (the zero-padding path is not taken)`,
        );
        const { output, residentBytes } = await runWeighted(gpu, testCase, quantized);
        // 圧縮のまま常駐している（= CPU 展開に落ちていない）。ゼロ詰めぶんと scale を含む。
        const padded = quantized.bytes.byteLength + (4 - (quantized.bytes.byteLength % 4));
        assertEquals(
          residentBytes,
          padded + (quantized.scale?.byteLength ?? 0),
          `${testCase.name}: GPU 常駐圧縮バイト数（scale 込み）`,
        );
        const reference = refTensor(testCase.weight.shape, quantized.values);
        const operands: RefTensor[] =
          testCase.op === "embedding"
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
  });
}

/**
 * linear の GEMM タイル境界（64×64 レジスタタイル — src/kernels/gemm.ts）を圧縮重みで踏む。
 *
 * f16 — MUST: **v4 経路とスカラ経路を対で持つ**。
 * - v4（`k%4==0 && n%4==0`）は u32 **2 語**で 4 要素を組む経路で、2 語目のオフセット誤りは
 *   k が 4 の倍数の形なら必ず値に出る（k=20）。
 * - スカラ経路は 1 語 2 要素の偶奇選択で、上の {@link WEIGHTED_CASES}（k=7）と合わせて
 *   「行長が 2 の倍数でない」形が唯一の検出器（k=37）。v4 経路では行頭が常に語境界に来るので
 *   この罠を**踏めない** — スカラ側のケースを削ると検出力が消える。
 *
 * i8 — MUST: **v4 経路とスカラ経路を対で持ち、どちらも n / k を違う長さ**にする。scale は
 * **出力チャネルごと**（`wscale[wcol]`）で、別の軸（行 / k / 出力列以外）から引く誤りは
 * 「チャネルごとに scale が違う」形でしか値に出ない。quantizeI8 は行ごとに最大値から
 * scale を決めるので、行ごとに値の違う重みを入れれば scale も行ごとに違う。
 * MUST: v4 経路でも scale は**成分ごとの f32 乗算**（`vec4 * scalar`）— 縮約の外へ出すと
 * ADR 0019 の「(Σ x·q)·s は CPU 展開とのビット一致を失う」に触れる。
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.tiles, async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of storage.tileCases) {
        const quantized = storage.quantize(testCase.weight, testCase.channelAxis);
        // scale がチャネルごとに違う（同じなら軸の取り違えが数値に出ない — 恒真化の門）
        if (quantized.scale !== undefined) {
          assert(new Set(quantized.scale).size > 1, `${testCase.name}: scale が全チャネルで同じ`);
        }
        const { output } = await runWeighted(gpu, testCase, quantized);
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
  });
}

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
test.skipIf(!GPU_AVAILABLE)(
  "GPU の unpack2x16float と CPU 展開が全 65,536 パターンでビット一致（実 GPU）",
  async () => {
    const vocab = 256;
    const hidden = 256;
    const bytes = f16BytesFromBits(Array.from({ length: vocab * hidden }, (_, i) => i));
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
      nodes: [
        {
          op: "embedding",
          ins: ["w", "index"],
          outs: ["y"],
          attrs: { padding_idx: -1 },
        },
      ],
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
);

/**
 * GPU の bf16 展開（`bitcast<f32>` — ADR 0131）と CPU の {@link bf16BitsToF32} が**全 65,536
 * パターンで一致**する。運び方は f16 の門と同じ（embedding の [256, 256]・`bits = row·256 + col`）。
 *
 * MUST: NaN は**双方 NaN**で見る（WGSL は NaN のペイロード伝播を保証しない — f16 と同じ）。
 * MUST: f32 の subnormal に当たるパターン（指数 0・仮数 ≠ 0）は **GPU が 0 へ flush してもよい**
 * （WGSL は subnormal の保持を保証しない）— それ以外はビット一致を要求する。
 */
test.skipIf(!GPU_AVAILABLE)(
  "GPU の bf16 bitcast 展開と CPU 展開が全 65,536 パターンで一致（実 GPU・subnormal は flush を許す）",
  async () => {
    const vocab = 256;
    const hidden = 256;
    const bytes = bf16BytesFromBits(Array.from({ length: vocab * hidden }, (_, i) => i));
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
      nodes: [
        {
          op: "embedding",
          ins: ["w", "index"],
          outs: ["y"],
          attrs: { padding_idx: -1 },
        },
      ],
    };
    const opened = await openModelBytes(declaration, [
      { graph: GRAPH_NAME, initializer: "w", bytes, encoding: { codec: "bf16" } },
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
    const expected = decodeBf16(bytes);
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    let nan = 0;
    let subnormal = 0;
    let flushed = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (Number.isNaN(expected[i])) {
        nan += 1;
        assert(Number.isNaN(actual[i]), `pattern 0x${i.toString(16)}: GPU 側が NaN でない`);
        continue;
      }
      if (((i >> 7) & 0xff) === 0 && (i & 0x7f) !== 0) {
        subnormal += 1;
        if (actual[i] === 0) {
          flushed += 1;
          continue;
        }
      }
      assertEquals(
        actualBits[i],
        expectedBits[i],
        `pattern 0x${i.toString(16)}: GPU ${actual[i]} / CPU ${expected[i]}`,
      );
    }
    // 網羅の証拠（NaN 254 本・subnormal 254 本を実際に踏んでいる）。flush の有無は機ごとの事実で、
    // 門の条件ではない（ここでは数だけ残す）。
    assertEquals(nan, 254, "NaN パターン数");
    assertEquals(subnormal, 254, "subnormal パターン数");
    assert(flushed <= subnormal, "flush の数が subnormal の数を超えている");
  },
);

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
test.skipIf(!GPU_AVAILABLE)(
  "GPU の unpack4xI8 と CPU 展開が全 256 値 × 4 レーンでビット一致（実 GPU）",
  async () => {
    const vocab = 256;
    const hidden = 4;
    const quantized: number[] = [];
    for (let row = 0; row < vocab; row += 1) {
      for (let col = 0; col < hidden; col += 1) quantized.push(row - 128);
    }
    const bytes = i8BytesFrom(quantized);
    const scale = Float32Array.from({ length: vocab }, (_, row) => 0.1 + row * 0.003);
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
      nodes: [{ op: "embedding", ins: ["w", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
    };
    const opened = await openModelBytes(declaration, [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes,
        encoding: {
          codec: "int8-sym",
          groupSize: hidden,
          scale: { bytes: f32Bytes(scale), dtype: "f32" },
        },
      },
    ]);
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
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
);

/**
 * 変種経路の full-write（ADR 0014）。プール再利用バッファに毒値を仕込んでから圧縮変種を
 * 実行し、1 語も残らないことを見る。
 *
 * 仕込み方は tests/gpu_full_write_test.ts と同じ（消費者ゼロの恒等 cast の出力が
 * ノード境界でプールへ戻り、次の確保で配り直される）。
 */
const poisonedGraph = (
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
const POISON = new Float32Array(new Uint32Array([0xdeadbeef]).buffer)[0];

for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.poison, async () => {
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
          const quantized = storage.quantize(fill([3, 7], POSITIVE), 0);
          const bias = fill([3], SIGNED);
          return {
            name: "linear",
            graph: poisonedGraph(
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
                bytes: quantized.bytes,
                encoding: quantized.encoding,
              },
            ],
            inputs: { x: fill([4, 7], SIGNED) },
            count: 12,
          };
        })(),
        (() => {
          const quantized = storage.quantize(fill([5, 3], SIGNED), 0);
          return {
            name: "embedding",
            graph: poisonedGraph(
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
                bytes: quantized.bytes,
                encoding: quantized.encoding,
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
  });
}

/**
 * 適格外（混在消費）は CPU で f32 展開され、**適格経路と同じ値**を出す
 * （i8 では ADR 0019 の「要素ごと dequant」がこの一致の根拠）。
 *
 * MUST: 2 経路の出力を突き合わせる。片方だけを CPU 参照と比べると、両経路が同じだけ
 * ずれている形（例: 展開表が誤っていて GPU 側も同じ誤りを持つ）を見逃す。
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.mixed, async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = storage.quantize(weight, 0);
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: quantized.bytes,
        encoding: quantized.encoding,
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
    ): Promise<{
      readonly y: Tensor;
      readonly resident: number;
      readonly expanded: number;
    }> => {
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(declaration, tensors),
        GRAPH_NAME,
      );
      try {
        const outputs = await session.run({ x });
        const diagnostics = session.diagnostics().storage;
        return {
          y: outputs["y"],
          resident: diagnostics.residentCompressedBytes,
          expanded: diagnostics.hostExpandedBytes,
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
      assertEquals(
        eligible.resident,
        storage.mixedResidentBytes,
        "適格側は圧縮のまま常駐（scale 込み）",
      );
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
  });
}

/**
 * 診断（ADR 0006 の常設義務）の検算。適格・適格外・非圧縮を 1 つのグラフに混ぜ、
 * 両方の数字が**実バイト数**と一致することを見る。
 *
 * bias を圧縮格納で宣言しても**必ず CPU 展開**に落ちる（bias スロットは適格判定に載らない —
 * プロトタイプの降格バグの逆）。
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.diagnostics, async () => {
    // 重み 3·7 = 21 要素（ゼロ詰めが要る長さ）/ bias 3 要素を **圧縮格納で宣言**
    const weight = storage.quantize(fill([3, 7], POSITIVE), 0);
    const bias = storage.quantize(fill([3], SIGNED), 0);
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
      { graph: GRAPH_NAME, initializer: "w", bytes: weight.bytes, encoding: weight.encoding },
      // i8: bias は rank 1（行 = 3・行長 1）— per-channel の groupSize は行長 1。
      { graph: GRAPH_NAME, initializer: "b", bytes: bias.bytes, encoding: bias.encoding },
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
      const diagnostics = session.diagnostics().storage;
      assertEquals(weight.bytes.byteLength, storage.diagnostics.payloadBytes);
      assertEquals(diagnostics.residentCompressedBytes, storage.diagnostics.residentBytes);
      // 適格外: bias 3 要素を f32 展開 = 12 バイト（f32 格納の s は**どちらにも入らない**。
      // bias の scale は GPU に上がらないので常駐にも展開にも数えない）
      assertEquals(diagnostics.hostExpandedBytes, 12);
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
  });
}

/**
 * initializer 自身をグラフ出力にした形（IR が許す — format/ir.ts の定義済み検査）。
 *
 * readback は宣言 dtype の semantic f32 を 4 バイト / 要素で重みバッファから写すので、圧縮の
 * まま常駐していると copy が実バッファ（f16 は 2 バイト / 要素・i8 は 1 バイト / 要素）を
 * はみ出す。適格判定が `graph.outputs` を見ていれば CPU 展開へ落ち、丸め後の重みがそのまま
 * 読める。
 */
for (const storage of STORAGES) {
  test.skipIf(!GPU_AVAILABLE)(storage.titles.graphOutput, async () => {
    const weight = fill([3, 4], POSITIVE);
    const bias = fill([3], SIGNED);
    const quantized = storage.quantize(weight, 0);
    const opened = await openModelBytes(linearDeclaration([], {}, ["w"]), [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: quantized.bytes,
        encoding: quantized.encoding,
      },
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
      const diagnostics = session.diagnostics().storage;
      assertEquals(
        diagnostics.residentCompressedBytes,
        0,
        "グラフ出力の重みは 1 バイトも圧縮常駐しない",
      );
      assertEquals(diagnostics.hostExpandedBytes, 48, "CPU 展開バイト数（f32 換算 12 要素）");
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
  });
}
