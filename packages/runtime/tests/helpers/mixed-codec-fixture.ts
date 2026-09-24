// 4 codec 混在（f32 / f16 / int8-sym / int4-sym-g）のモデル素材 — 容器面の共有 fixture。
// 「どれか 1 codec が経路から欠けても沈黙しない」形を 1 本のモデルで用意する。
// src/ からは型と合流面だけを引く（バイト列の組み立ては helpers 側に閉じる）。

import type { BoundContainer } from "../../src/format/container/bind.ts";
import type { TensorInput } from "./container-write.ts";
import { quantizeF16 } from "./f16.ts";
import { quantizeI4 } from "./i4.ts";
import { quantizeI8 } from "./i8.ts";
import { type DeclarationJson, fill, type FilledTensor, GRAPH_NAME } from "./model-fixture.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const VARYING = (i: number): number => (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);

const rawBytes = (filled: FilledTensor): Uint8Array<ArrayBuffer> =>
  new Uint8Array(filled.data.buffer);

const f32Tensor = (initializer: string, filled: FilledTensor): TensorInput => ({
  graph: GRAPH_NAME,
  initializer,
  bytes: rawBytes(filled),
  encoding: { codec: "f32" },
});

export type MixedCodecFixture = {
  readonly declaration: DeclarationJson;
  readonly tensors: readonly TensorInput[];
  readonly x: FilledTensor;
};

/**
 * linear 3 段（w1 = int4-sym-g g16 / w2 = f16 / w3 = int8-sym）+ f32 bias 群のモデル素材。
 * 4 つの codec が全て「圧縮のまま常駐」の適格になる形（消費は linear の重みスロットのみ）。
 */
export const buildFixture = (): MixedCodecFixture => {
  const w1 = fill([16, 32], VARYING);
  const w2 = fill([8, 16], VARYING);
  const w3 = fill([4, 8], VARYING);
  const q1 = quantizeI4(w1.data, w1.shape, 16);
  const q2 = quantizeF16(w2.data);
  const q3 = quantizeI8(w3.data, w3.shape, 0);
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 32] }],
    outputs: ["y"],
    initializers: { w1: {}, w2: {}, w3: {}, b1: {}, b2: {}, b3: {} },
    values: {
      w1: { dtype: "f32", shape: [16, 32] },
      w2: { dtype: "f32", shape: [8, 16] },
      w3: { dtype: "f32", shape: [4, 8] },
      b1: { dtype: "f32", shape: [16] },
      b2: { dtype: "f32", shape: [8] },
      b3: { dtype: "f32", shape: [4] },
      h1: { dtype: "f32", shape: [2, 16] },
      h2: { dtype: "f32", shape: [2, 8] },
      y: { dtype: "f32", shape: [2, 4] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w1", "b1"], outs: ["h1"], attrs: {} },
      { op: "linear", ins: ["h1", "w2", "b2"], outs: ["h2"], attrs: {} },
      { op: "linear", ins: ["h2", "w3", "b3"], outs: ["y"], attrs: {} },
    ],
  };
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "w1",
      bytes: q1.bytes,
      // group scale は rank 2 `[rows, groups]` の行優先バイト列（container-v1 §6.1）。
      encoding: {
        codec: "int4-sym-g",
        groupSize: 16,
        scale: { bytes: new Uint8Array(q1.scale.buffer), dtype: "f32" },
      },
    },
    { graph: GRAPH_NAME, initializer: "w2", bytes: q2.bytes, encoding: { codec: "f16" } },
    {
      graph: GRAPH_NAME,
      initializer: "w3",
      bytes: q3.bytes,
      // per-channel は groupSize = 行長・groups = 1（scale は rows × 4 バイト）。
      encoding: {
        codec: "int8-sym",
        groupSize: 8,
        scale: { bytes: new Uint8Array(q3.scale.buffer), dtype: "f32" },
      },
    },
    f32Tensor("b1", fill([16], SIGNED)),
    f32Tensor("b2", fill([8], SIGNED)),
    f32Tensor("b3", fill([4], SIGNED)),
  ];
  return { declaration, tensors, x: fill([2, 32], SIGNED) };
};

/**
 * 開いた容器を「block を何回取ったか数える」面で包む。
 *
 * 2 段境界（ADR 0070 決定 5 / graph-first）の主張は「重みの block を 1 つも取る前に admission が
 * 終わる」なので、検出器は**取得の回数**そのものになる。合流（`graphs`）は包まずそのまま通す。
 */
export const countingContainer = (
  opened: BoundContainer,
): { readonly container: BoundContainer; reads(): number } => {
  let reads = 0;
  return {
    container: {
      graphs: opened.graphs,
      readBlock: (id) => {
        reads += 1;
        return opened.readBlock(id);
      },
    },
    reads: () => reads,
  };
};
