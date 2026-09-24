// companion scale の形（`groupScaleShape` — src/format/container/codecs.ts）を消費する 4 経路の突合。
// GPU を一切使わない。
//
// scale の形を前提にする経路は 4 つある — 合流層（scale block の長さ）・常駐プランナ（scale の
// バイト数）・容器から Session 構築へ渡す scale 形（`containerBatches`）・CPU 展開 `decodeI4` の
// 突合。どれか 1 つが自前の式を持つと、受理した形と展開が読む形が静かに食い違う
// （group scale が 1 チャネル 1 値として配られる沈黙誤値）。
//
// 表の 1 行ごとに見るのは 2 段:
// ① 共有の関数が手書きの形を返す（式そのものの固定 — 期待値は手で書く。関数から引くと恒真）。
// ② 4 経路の受理 / 数 / 形が、その関数の値と一致する（式を変えれば 4 経路とも同時に動き、
//    1 経路だけ別の式に戻すとその経路の行が赤になる）。
// 行の軸 1（conv_transpose1d の `[Cin,Cout,K]`）と行長 0 の退化形は、局所の式が最も割れやすい
// 形として必ず表に置く。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { mergedGraph } from "../src/format/container/bind.ts";
import { type CodecName, groupScaleShape, payloadBytes } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { decodeI4, I4Error } from "../src/format/i4.ts";
import { containerBatches } from "../src/runtime/executor.ts";
import type { ReadyScale } from "../src/runtime/session-build.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const zeros = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(new ArrayBuffer(length));

/** 重み `w` 1 本を消費する op（経路が重みスロットの消費として扱うもの）。 */
type Consumer = "linear" | "conv1d" | "conv_transpose1d";

type Case = {
  readonly label: string;
  readonly consumer: Consumer;
  readonly codec: CodecName;
  readonly shape: readonly number[];
  readonly rowAxis: 0 | 1;
  readonly groupSize: number;
  /** 手書きの期待形（共有の関数から引かない）。 */
  readonly want: readonly [number, number];
};

const CASES: readonly Case[] = [
  {
    label: "i4 linear [8,64]・group 16",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [8, 64],
    rowAxis: 0,
    groupSize: 16,
    want: [8, 4],
  },
  {
    label: "i4 linear [8,64]・group 64（行長 = group 長）",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [8, 64],
    rowAxis: 0,
    groupSize: 64,
    want: [8, 1],
  },
  {
    label: "i4 conv1d [4,16,4]・group 16（行長は Cin·K = 64 に平坦化）",
    consumer: "conv1d",
    codec: "int4-sym-g",
    shape: [4, 16, 4],
    rowAxis: 0,
    groupSize: 16,
    want: [4, 4],
  },
  {
    label: "i4 linear [4,0]・group 16（行長 0 の退化形は group 数 1）",
    consumer: "linear",
    codec: "int4-sym-g",
    shape: [4, 0],
    rowAxis: 0,
    groupSize: 16,
    want: [4, 1],
  },
  {
    label: "i8 linear [6,16]・per-channel",
    consumer: "linear",
    codec: "int8-sym",
    shape: [6, 16],
    rowAxis: 0,
    groupSize: 16,
    want: [6, 1],
  },
  {
    label: "i8 conv_transpose1d [4,6,2]・行の軸 1（行は Cout = 6・行長 8）",
    consumer: "conv_transpose1d",
    codec: "int8-sym",
    shape: [4, 6, 2],
    rowAxis: 1,
    groupSize: 8,
    want: [6, 1],
  },
  {
    // per-channel の要素数は `numel / groupSize` で行の軸に依らないので、バイト数だけを見る経路
    // （合流層・常駐プランナ）で軸の取り違えが数に出るのは行長 0 の形だけ（軸 0 と読むと 0 行）。
    label: "i8 conv_transpose1d [0,6,2]・行の軸 1・行長 0（軸 0 と読むと scale 0 要素）",
    consumer: "conv_transpose1d",
    codec: "int8-sym",
    shape: [0, 6, 2],
    rowAxis: 1,
    groupSize: 1,
    want: [6, 1],
  },
];

/** `w` を 1 本の op で消費するグラフ（bias `b` は f32）。 */
const declarationOf = (testCase: Case): DeclarationJson => {
  const { consumer, shape } = testCase;
  const common = {
    format: "karume-ir",
    version: 2,
    requires: { ops: [consumer] },
    symbols: [],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
  };
  if (consumer === "linear") {
    const [rows, width] = shape;
    return {
      ...common,
      inputs: [{ name: "x", dtype: "f32", shape: [2, width] }],
      values: {
        w: { dtype: "f32", shape: [...shape] },
        b: { dtype: "f32", shape: [rows] },
        y: { dtype: "f32", shape: [2, rows] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    };
  }
  if (consumer === "conv1d") {
    const [channelsOut, channelsIn, kernel] = shape;
    const length = 6;
    return {
      ...common,
      inputs: [{ name: "x", dtype: "f32", shape: [1, channelsIn, length] }],
      values: {
        w: { dtype: "f32", shape: [...shape] },
        b: { dtype: "f32", shape: [channelsOut] },
        y: { dtype: "f32", shape: [1, channelsOut, length - kernel + 1] },
      },
      nodes: [{
        op: "conv1d",
        ins: ["x", "w", "b"],
        outs: ["y"],
        attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
      }],
    };
  }
  const [channelsIn, channelsOut, kernel] = shape;
  const length = 5;
  return {
    ...common,
    inputs: [{ name: "x", dtype: "f32", shape: [1, channelsIn, length] }],
    values: {
      w: { dtype: "f32", shape: [...shape] },
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [1, channelsOut, length - 1 + kernel] },
    },
    nodes: [{
      op: "conv_transpose1d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0 },
    }],
  };
};

/** `w` の scale を `scaleElements` 要素ぶん持つ供給（中身は見ないので 0 埋め）。 */
const tensorsOf = (testCase: Case, scaleElements: number): readonly TensorInput[] => {
  const { codec, shape, rowAxis, groupSize } = testCase;
  const rows = shape[rowAxis];
  return [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: zeros(payloadBytes(codec, numel(shape), "w")),
      encoding: {
        codec,
        ...(rowAxis === 0 ? {} : { rowAxis }),
        groupSize,
        scale: { bytes: zeros(scaleElements * 4), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: zeros(rows * 4),
      encoding: { codec: "f32" },
    },
  ];
};

/** `containerBatches` が `w` の item に同乗させる scale（GPU を通らない Session 構築の入口）。 */
const yieldedScale = async (testCase: Case): Promise<ReadyScale | undefined> => {
  const expected = groupScaleShape(testCase.shape, testCase.rowAxis, testCase.groupSize);
  const opened = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)));
  for await (const batch of containerBatches(opened, GRAPH_NAME)) {
    for await (const item of batch.items) {
      if (item.name === "w") return item.scale;
    }
  }
  return undefined;
};

describe("companion scale の形: 4 経路が groupScaleShape の 1 本に従う", () => {
  for (const testCase of CASES) {
    const { shape, rowAxis, groupSize } = testCase;
    const expected = groupScaleShape(shape, rowAxis, groupSize);

    describe(testCase.label, () => {
      it("共有の関数は手書きの形を返す", () => {
        assertEquals(expected, testCase.want);
      });

      it("合流層はその形ぶんの scale block を受理し、1 要素多い block を拒否する", () => {
        const bound = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)))
          .graphs[GRAPH_NAME];
        assertEquals(bound.supplies.get("w")?.scale?.payloadBytes, numel(expected) * 4);
        assertThrows(
          () => memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected) + 1)),
          ContainerFormatError,
        );
      });

      it("常駐プランナの scale バイト数はその形の要素数 × 4", () => {
        const bound = memoryModel(declarationOf(testCase), tensorsOf(testCase, numel(expected)))
          .graphs[GRAPH_NAME];
        const seat = planWeightResidency(mergedGraph(bound, GRAPH_NAME)).get("w");
        // 行長 0 の i4 も含め、表の重みはどれも圧縮のまま常駐する席に載る（席の判定は
        // runtime_weight_residency_test の担当 — ここは scale のバイト数だけを見る）。
        if (seat?.seat !== "i4" && seat?.seat !== "i8") {
          throw new Error(`w の席が圧縮常駐でない: ${seat?.seat}`);
        }
        assertEquals(seat.scaleBytes, numel(expected) * 4);
      });

      it("容器から Session 構築へ渡る scale の形がその形", async () => {
        const scale = await yieldedScale(testCase);
        assertEquals(scale?.shape, expected);
        assertEquals(scale?.bytes.byteLength, numel(expected) * 4);
      });

      if (testCase.codec !== "int4-sym-g") return;

      it("decodeI4 はその形を受理し、group 数が 1 つ違う形を拒否する", () => {
        const bytes = zeros(numel(shape) / 2);
        const decoded = decodeI4(
          bytes,
          shape,
          new Float32Array(numel(expected)),
          expected,
          groupSize,
        );
        assertEquals(decoded.length, numel(shape));
        const wider: readonly [number, number] = [expected[0], expected[1] + 1];
        assertThrows(
          () => decodeI4(bytes, shape, new Float32Array(numel(wider)), wider, groupSize),
          I4Error,
          "group 形",
        );
      });
    });
  }
});
