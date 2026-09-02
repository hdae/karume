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

/**
 * テンソルを**先頭次元（行）の連続範囲**で割った piece 列（キーは `<親名>#NNNNN-of-NNNNN`）。
 *
 * 綴りは読み手契約そのものなので、テスト側でも 1 箇所で作る（実装から引くと往復が恒真化する
 * ため、正規表現ではなく組み立て側をここに書き下す）。末尾以外の piece が 4 の倍数バイトに
 * なる行数を選ぶのは呼び手の責任（行あたりのバイト長から決まる）。
 */
const splitRows = (tensor: TensorSpec, rows: readonly number[]): TensorSpec[] => {
  const rowBytes = tensor.data.byteLength / tensor.shape[0];
  let offset = 0;
  return rows.map((take, index) => {
    const piece: TensorSpec = {
      name: `${tensor.name}#${String(index + 1).padStart(5, "0")}-of-${
        String(rows.length).padStart(5, "0")
      }`,
      dtype: tensor.dtype,
      shape: [take, ...tensor.shape.slice(1)],
      data: tensor.data.subarray(offset * rowBytes, (offset + take) * rowBytes),
    };
    offset += take;
    return piece;
  });
};

/**
 * テンソル分割（piece）の fixture — 分割なしの 1 本と、同じ資産を先頭次元で割った shard 列。
 *
 * MUST: 分割対象は「生バイト席（f32）/ 圧縮常駐席（f16・i8・i4）/ 展開席（i8・i4）」を全て
 * 含む（席ごとにバイト位置の出し方と scale の扱いが違うので、1 つでも欠けると門が沈黙する）。
 * `w2` は 15 要素の f16 = **末尾 piece が奇数要素**になる形で、末尾詰め物の経路を通す。
 * 展開席（`e` / `e2`）は消費が重みスロット以外（`add`）— piece ごとに scale を行で切り出す
 * 経路がここでだけ走る。
 */
export const buildPieceFixture = () => {
  const w1 = fill([16, 32], VARYING);
  const w3 = fill([5, 16], VARYING);
  const w2 = fill([3, 5], VARYING);
  const w5 = fill([4, 3], VARYING);
  const e = fill([2, 4], SIGNED);
  const e2 = fill([2, 16], VARYING);
  const q1 = quantizeI4(w1.data, w1.shape, 16);
  const q3 = quantizeI8(w3.data, w3.shape, 0);
  const q2 = quantizeF16(w2.data);
  const qe = quantizeI8(e.data, e.shape, 0);
  const qe2 = quantizeI4(e2.data, e2.shape, 16);
  const graph = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear", "add"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 32] }],
    outputs: ["y", "z"],
    initializers: {
      w1: { tensor: "m.w1", storage: { dtype: "i4", scale: "m.s1", group_size: 16 } },
      b1: { tensor: "m.b1", storage: { dtype: "f32" } },
      w3: { tensor: "m.w3", storage: { dtype: "i8", scale: "m.s3" } },
      b3: { tensor: "m.b3", storage: { dtype: "f32" } },
      w2: { tensor: "m.w2", storage: { dtype: "f16" } },
      b2: { tensor: "m.b2", storage: { dtype: "f32" } },
      w5: { tensor: "m.w5", storage: { dtype: "f32" } },
      b5: { tensor: "m.b5", storage: { dtype: "f32" } },
      e: { tensor: "m.e", storage: { dtype: "i8", scale: "m.se" } },
      e2: { tensor: "m.e2", storage: { dtype: "i4", scale: "m.se2", group_size: 16 } },
    },
    values: {
      w1: { dtype: "f32", shape: [16, 32] },
      b1: { dtype: "f32", shape: [16] },
      w3: { dtype: "f32", shape: [5, 16] },
      b3: { dtype: "f32", shape: [5] },
      w2: { dtype: "f32", shape: [3, 5] },
      b2: { dtype: "f32", shape: [3] },
      w5: { dtype: "f32", shape: [4, 3] },
      b5: { dtype: "f32", shape: [4] },
      e: { dtype: "f32", shape: [2, 4] },
      e2: { dtype: "f32", shape: [2, 16] },
      h1: { dtype: "f32", shape: [2, 16] },
      h2: { dtype: "f32", shape: [2, 5] },
      h3: { dtype: "f32", shape: [2, 3] },
      h4: { dtype: "f32", shape: [2, 4] },
      y: { dtype: "f32", shape: [2, 4] },
      z: { dtype: "f32", shape: [2, 16] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w1", "b1"], outs: ["h1"], attrs: {} },
      { op: "linear", ins: ["h1", "w3", "b3"], outs: ["h2"], attrs: {} },
      { op: "linear", ins: ["h2", "w2", "b2"], outs: ["h3"], attrs: {} },
      { op: "linear", ins: ["h3", "w5", "b5"], outs: ["h4"], attrs: {} },
      { op: "add", ins: ["h4", "e"], outs: ["y"], attrs: {} },
      { op: "add", ins: ["h1", "e2"], outs: ["z"], attrs: {} },
    ],
  };
  const metadata = { karume_ir: JSON.stringify(graph) };
  const f32Tensor = (name: string, filled: ReturnType<typeof fill>): TensorSpec => ({
    name,
    dtype: "F32",
    shape: [...filled.shape],
    data: new Uint8Array(filled.data.buffer),
  });
  const scaleTensor = (
    name: string,
    quantized: {
      readonly scale: Float32Array<ArrayBuffer>;
      readonly scaleShape: readonly number[];
    },
  ): TensorSpec => ({
    name,
    dtype: "F32",
    shape: [...quantized.scaleShape],
    data: new Uint8Array(quantized.scale.buffer),
  });
  const whole = {
    s1: scaleTensor("m.s1", q1),
    s3: scaleTensor("m.s3", q3),
    se: scaleTensor("m.se", qe),
    se2: scaleTensor("m.se2", qe2),
    b1: f32Tensor("m.b1", fill([16], SIGNED)),
    b2: f32Tensor("m.b2", fill([3], SIGNED)),
    b3: f32Tensor("m.b3", fill([5], SIGNED)),
    b5: f32Tensor("m.b5", fill([4], SIGNED)),
    w5: f32Tensor("m.w5", w5),
    w1: { name: "m.w1", dtype: "I4", shape: [16, 32], data: q1.bytes } satisfies TensorSpec,
    e2: { name: "m.e2", dtype: "I4", shape: [2, 16], data: qe2.bytes } satisfies TensorSpec,
    w2: { name: "m.w2", dtype: "F16", shape: [3, 5], data: q2.bytes } satisfies TensorSpec,
    w3: { name: "m.w3", dtype: "I8", shape: [5, 16], data: q3.bytes } satisfies TensorSpec,
    e: { name: "m.e", dtype: "I8", shape: [2, 4], data: qe.bytes } satisfies TensorSpec,
  };
  // 行の割り方は「末尾以外の piece が 4 の倍数バイト」を満たすものを選ぶ（1 行 = f32 4B /
  // i4 [.,32] 16B / i8 [.,16] 16B / f16 [.,5] 10B / i8 [.,4] 4B / i4 [.,16] 8B）。
  const w1p = splitRows(whole.w1, [8, 8]);
  const b1p = splitRows(whole.b1, [8, 8]);
  const w3p = splitRows(whole.w3, [3, 2]);
  const w2p = splitRows(whole.w2, [2, 1]);
  const ep = splitRows(whole.e, [1, 1]);
  const e2p = splitRows(whole.e2, [1, 1]);
  return {
    metadata,
    tensors: whole,
    x: fill([2, 32], SIGNED),
    /** 分割なしの 1 本（A/B の対照側）。 */
    fullBuffer: (): ArrayBuffer =>
      buildSafetensors([
        whole.s1,
        whole.s3,
        whole.se,
        whole.se2,
        whole.b1,
        whole.b2,
        whole.b3,
        whole.b5,
        whole.w5,
        whole.w1,
        whole.e2,
        whole.w2,
        whole.w3,
        whole.e,
      ], metadata),
    /**
     * 3 分割（graph shard = 分割しない小テンソル / shard1 = piece 1 + 全 scale /
     * shard2 = piece 2）。各 shard 内の並びは整列降順（F32 / I4 → F16 → I8）。
     */
    shards: (): ArrayBuffer[] => [
      buildSafetensors([whole.b2, whole.b3, whole.b5, whole.w5], metadata),
      buildSafetensors([
        whole.s1,
        whole.s3,
        whole.se,
        whole.se2,
        b1p[0],
        w1p[0],
        e2p[0],
        w2p[0],
        w3p[0],
        ep[0],
      ], undefined),
      buildSafetensors([b1p[1], w1p[1], e2p[1], w2p[1], w3p[1], ep[1]], undefined),
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
