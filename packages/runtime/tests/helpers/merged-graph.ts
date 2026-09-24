// 宣言（IR v2）→ 合流後の `IrGraph` を組む小物。
//
// IR v2 の宣言は格納を持たないので、plan / fusion / 契約検査のように**格納つきグラフの形**
// だけを見るテストでも、`IrGraph` を得るには供給が 1 本ずつ要る。ここが作るのは
// 「中身を見ない供給」— 宣言 shape と codec から決まる長さの 0 埋めバイト列で、合流は
// メモリ内容器 1 本（`openMemoryContainer` → `bindDeclarations`）を通す。
//
// MUST: 長さの式は台帳（`codecs.ts`）から引く。テスト側で数え直すと、合流層が期待する長さと
// 供給の長さが**同じ誤りを共有**して突合が恒真になる。

import { mergedGraph } from "../../src/format/container/bind.ts";
import {
  codecEntry,
  type CodecName,
  groupCount,
  payloadBytes,
  perChannelGroupSize,
  scaleBytes,
} from "../../src/format/container/codecs.ts";
import type { IrGraph } from "../../src/format/ir.ts";
import type { EncodingInput, TensorInput } from "./container-write.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./model-fixture.ts";

/** 格納の指定（量子化 codec は group codec だけが `groupSize` を要る）。 */
export type StorageSpec =
  | CodecName
  | { readonly codec: CodecName; readonly groupSize?: number; readonly rowAxis?: 0 | 1 };

/** initializer 名 → 格納。省略した名前は値の意味論 dtype から `f32` / `i32`。 */
export type StorageMap = Readonly<Record<string, StorageSpec>>;

const zeros = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(new ArrayBuffer(length));

const encodingOf = (
  spec: StorageSpec | undefined,
  dtype: string,
  shape: readonly number[],
  numel: number,
  name: string,
): EncodingInput => {
  const codec: CodecName = spec === undefined
    ? (dtype === "i32" ? "i32" : "f32")
    : typeof spec === "string"
    ? spec
    : spec.codec;
  const entry = codecEntry(codec);
  if (entry.scale === "forbidden") return { codec };
  const rowAxis = typeof spec === "object" ? spec.rowAxis ?? 0 : 0;
  const rowCount = shape[rowAxis];
  const rowLength = rowCount === 0 ? 0 : numel / rowCount;
  const groupSize = entry.grouping === "channel"
    ? perChannelGroupSize(rowLength)
    : typeof spec === "object" && spec.groupSize !== undefined
    ? spec.groupSize
    : rowLength;
  return {
    codec,
    ...(rowAxis === 0 ? {} : { rowAxis }),
    groupSize,
    scale: {
      bytes: zeros(scaleBytes(rowCount * groupCount(rowLength, groupSize), `${name} の scale`)),
      dtype: "f32",
    },
  };
};

/** 宣言の全 initializer（`shared` を除く）に、中身を見ない供給を 1 本ずつ作る。 */
export const autoTensors = (
  declaration: DeclarationJson,
  storage: StorageMap = {},
): readonly TensorInput[] =>
  Object.entries(declaration.initializers)
    .filter(([, init]) => init.shared !== true)
    .map(([name]) => {
      const value = declaration.values[name];
      const shape = value.shape.map(Number);
      const numel = shape.reduce((count, dim) => count * dim, 1);
      const encoding = encodingOf(storage[name], value.dtype, shape, numel, name);
      return {
        graph: GRAPH_NAME,
        initializer: name,
        bytes: zeros(payloadBytes(encoding.codec, numel, name)),
        encoding,
      };
    });

/**
 * 宣言 + **明示の供給** → 合流後の `IrGraph`。中身のあるバイト列（実際に量子化した重みなど）を
 * 渡したいテストはこちら。中身を見ないテストは {@link mergeGraph}。
 */
export const mergeTensors = (
  declaration: DeclarationJson,
  tensors: readonly TensorInput[],
): IrGraph => mergedGraph(memoryModel(declaration, tensors).graphs[GRAPH_NAME], GRAPH_NAME);

/**
 * 宣言 → 合流後の `IrGraph`。供給は {@link autoTensors} が宣言から作る（中身は 0 埋め）。
 * 格納を動かしたいテストは `storage` に initializer 名で codec を渡す。
 */
export const mergeGraph = (
  declaration: DeclarationJson,
  storage: StorageMap = {},
): IrGraph => mergeTensors(declaration, autoTensors(declaration, storage));
