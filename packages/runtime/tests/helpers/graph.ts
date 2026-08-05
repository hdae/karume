// ランタイム側テストの組み立てヘルパ（グラフ JSON → 配布形バイト列）。
// src/ を一切 import しない — 公開面のみで書く E2E テストからも使えるようにするため。

import { buildSafetensors, type GraphJson, type TensorSpec } from "./format.ts";

export const graphModelBuffer = (
  graph: GraphJson,
  tensors: readonly TensorSpec[] = [],
): ArrayBuffer => buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });

/** 単一ノードグラフの dtype / attrs 指定（既定は全て f32・attrs 空）。 */
export type SingleOpOptions = {
  readonly inDtypes?: readonly string[];
  readonly outDtype?: string;
  readonly attrs?: Record<string, unknown>;
  readonly symbols?: readonly string[];
};

/** 入力だけを取る単一ノードのグラフ（op 単位の数値検証用）。 */
export const singleOpGraph = (
  op: string,
  inputShapes: readonly (readonly (number | string)[])[],
  outShape: readonly (number | string)[],
  options: SingleOpOptions = {},
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: [op] },
  symbols: [...(options.symbols ?? [])],
  inputs: inputShapes.map((shape, index) => ({
    name: `x${index}`,
    dtype: options.inDtypes?.[index] ?? "f32",
    shape: [...shape],
  })),
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: options.outDtype ?? "f32", shape: [...outShape] } },
  nodes: [
    {
      op,
      ins: inputShapes.map((_, index) => `x${index}`),
      outs: ["y"],
      attrs: { ...options.attrs },
    },
  ],
});

/** 意味論 dtype ごとのホスト側 TypedArray（bool は u32 の 0/1 — ADR 0009）。 */
export type FilledTensor =
  | {
    readonly dtype: "f32";
    readonly shape: readonly number[];
    readonly data: Float32Array<ArrayBuffer>;
  }
  | {
    readonly dtype: "i32";
    readonly shape: readonly number[];
    readonly data: Int32Array<ArrayBuffer>;
  }
  | {
    readonly dtype: "bool";
    readonly shape: readonly number[];
    readonly data: Uint32Array<ArrayBuffer>;
  };

/** 決定的なデータ生成（乱数は使わない — 失敗が再現しないため）。 */
export const fill = (
  shape: readonly number[],
  generator: (index: number) => number,
  dtype: "f32" | "i32" | "bool" = "f32",
): FilledTensor => {
  const count = shape.reduce((total, dim) => total * dim, 1);
  const values = Array.from({ length: count }, (_, i) => generator(i));
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: Float32Array.from(values) };
    case "i32":
      return { dtype, shape, data: Int32Array.from(values) };
    case "bool":
      return { dtype, shape, data: Uint32Array.from(values) };
  }
};
