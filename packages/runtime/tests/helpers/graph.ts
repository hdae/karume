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
  /** 出力 slot 昇順の dtype（既定は全 slot f32 — {@link singleOpGraph}）。 */
  readonly outDtypes?: readonly string[];
  readonly attrs?: Record<string, unknown>;
  readonly symbols?: readonly string[];
};

/**
 * 出力 slot の値名。slot 0 は `y`（既存テストがそのまま `outputs["y"]` を読む）で、
 * slot 1 以降は `y1`, `y2`, …（多出力 op — ADR 0068 決定 1）。
 */
export const outputName = (slot: number): string => (slot === 0 ? "y" : `y${slot}`);

/**
 * 入力だけを取る単一ノードのグラフ（op 単位の数値検証用）。
 *
 * `outShapes` は**出力 slot 昇順の列**（ADR 0068 決定 1 — 単一出力 op では長さ 1）。
 */
export const singleOpGraph = (
  op: string,
  inputShapes: readonly (readonly (number | string)[])[],
  outShapes: readonly (readonly (number | string)[])[],
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
  outputs: outShapes.map((_, slot) => outputName(slot)),
  initializers: {},
  values: Object.fromEntries(outShapes.map((shape, slot) => [
    outputName(slot),
    { dtype: options.outDtypes?.[slot] ?? "f32", shape: [...shape] },
  ])),
  nodes: [
    {
      op,
      ins: inputShapes.map((_, index) => `x${index}`),
      outs: outShapes.map((_, slot) => outputName(slot)),
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
