// shard 面（ADR 0070 決定 3）の共有 fixture — 4 格納混在のモデル素材と shard 列の組み立て。
// src/ を一切 import しない（helpers の規律 — shard の型は構造互換の形で書く）。

import { buildSafetensors, type TensorSpec } from "./format.ts";
import { fill } from "./graph.ts";
import { quantizeF16 } from "./f16.ts";
import { quantizeI4 } from "./i4.ts";
import { quantizeI8 } from "./i8.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const VARYING = (i: number): number => (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);

/**
 * linear 3 段（w1 = i4 g16 / w2 = f16 / w3 = i8）+ f32 bias 群のモデル素材。
 * 4 つの格納 dtype が全て「圧縮のまま常駐」の適格になる形（消費は linear の重みスロットのみ）。
 */
export const buildFixture = () => {
  const w1 = fill([16, 32], VARYING);
  const w2 = fill([8, 16], VARYING);
  const w3 = fill([4, 8], VARYING);
  const q1 = quantizeI4(w1.data, w1.shape, 16);
  const q2 = quantizeF16(w2.data);
  const q3 = quantizeI8(w3.data, w3.shape, 0);
  const graph = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 32] }],
    outputs: ["y"],
    initializers: {
      w1: { tensor: "m.w1", storage: { dtype: "i4", scale: "m.s1", group_size: 16 } },
      w2: { tensor: "m.w2", storage: { dtype: "f16" } },
      w3: { tensor: "m.w3", storage: { dtype: "i8", scale: "m.s3" } },
      b1: { tensor: "m.b1", storage: { dtype: "f32" } },
      b2: { tensor: "m.b2", storage: { dtype: "f32" } },
      b3: { tensor: "m.b3", storage: { dtype: "f32" } },
    },
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
  const metadata = { karume_ir: JSON.stringify(graph) };
  const f32Tensor = (name: string, filled: ReturnType<typeof fill>): TensorSpec => ({
    name,
    dtype: "F32",
    shape: [...filled.shape],
    data: new Uint8Array(filled.data.buffer),
  });
  const biases = [
    f32Tensor("m.b1", fill([16], SIGNED)),
    f32Tensor("m.b2", fill([8], SIGNED)),
    f32Tensor("m.b3", fill([4], SIGNED)),
  ];
  // 各 shard 内の並びは整列降順（F32 → I4 → F16 → I8 — 先頭 offset の整列規則を満たす並び）
  const s1: TensorSpec = {
    name: "m.s1",
    dtype: "F32",
    shape: [...q1.scaleShape],
    data: new Uint8Array(q1.scale.buffer),
  };
  const s3: TensorSpec = {
    name: "m.s3",
    dtype: "F32",
    shape: [...q3.scaleShape],
    data: new Uint8Array(q3.scale.buffer),
  };
  const t = {
    w1: { name: "m.w1", dtype: "I4", shape: [16, 32], data: q1.bytes } satisfies TensorSpec,
    w2: { name: "m.w2", dtype: "F16", shape: [8, 16], data: q2.bytes } satisfies TensorSpec,
    w3: { name: "m.w3", dtype: "I8", shape: [4, 8], data: q3.bytes } satisfies TensorSpec,
    s1,
    s3,
  };
  return {
    metadata,
    biases,
    tensors: t,
    x: fill([2, 32], SIGNED),
    /** 全量面のファイル（グラフ + 全テンソル）。 */
    fullBuffer: (): ArrayBuffer =>
      buildSafetensors([...biases, s1, s3, t.w1, t.w2, t.w3], metadata),
    /** 既定の 3 分割（graph shard = bias 群 / shard1 = w1+s1 / shard2 = w3+s3+w2）。 */
    shards: (): ArrayBuffer[] => [
      buildSafetensors(biases, metadata),
      buildSafetensors([s1, t.w1], undefined),
      buildSafetensors([s3, t.w2, t.w3], undefined),
    ],
  };
};

/** shard 1 本の面（runtime の `ModelShard` と構造互換 — helpers は src を import しない）。 */
export type FixtureShard = {
  readonly id: string;
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/**
 * ArrayBuffer 列を shard 面の入力（実名 + tight view の逐次列）にする。id は既定で
 * 「配布形のファイル名らしい実名」を振る（帰属の検出器が連番と取り違えないため）。
 */
export const shardStream = async function* (
  buffers: readonly ArrayBuffer[],
  ids?: readonly string[],
): AsyncGenerator<FixtureShard, void, unknown> {
  for (const [index, buffer] of buffers.entries()) {
    yield {
      id: ids?.[index] ?? `fixture/model-0000${index}.safetensors`,
      bytes: new Uint8Array(buffer),
    };
  }
};
