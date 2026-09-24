// 容器面テストの組み立てヘルパ（IR v2 の宣言 JSON → `krm` のバイト列 / 開いた容器）。
//
// テストが期待する「正しい形」を 1 箇所に置き、異常系は各テストがそこから 1 点だけ壊す。
// 同じ入力から **krm 経路**（`writeModelContainer` → `openContainer`）と**メモリ内容器**
// （`openMemoryContainer`）の両方を作れるので、2 つの供給元の A/B もここ 1 本で組める。
//
// 宣言そのものを組む小道具（`fill` / `f32Bytes` / `singleOpDeclaration` / `withStateReaders`）も
// ここ 1 本きり。合流後の `IrGraph` が要るテストは `helpers/merged-graph.ts` を使う。

import type { BoundContainer } from "../../src/format/container/bind.ts";
import { openContainer, type OpenedContainer } from "../../src/format/container/open.ts";
import { type MemoryTensor, openMemoryContainer } from "../../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../../src/format/ir.ts";
import { type TensorInput, writeModelContainer, type WriteOptions } from "./container-write.ts";

/** このヘルパが書くグラフの名前（容器は複数グラフを持てるが、fixture は 1 本きり）。 */
export const GRAPH_NAME = "model";

/** f32 の値列をそのままバイト列にする（重み / scale の実体を組む最小の道具）。 */
export const f32Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/** IR v2 の宣言 JSON の可変な写し（テストが 1 点だけ壊せるよう型は緩く持つ）。 */
export type DeclarationJson = {
  format: string;
  version: number;
  requires: { ops: string[] };
  symbols: string[];
  inputs: { name: string; dtype: string; shape: (number | string)[] }[];
  outputs: string[];
  /** IR v2 の initializer 宣言は `{}`（実体を持つ）か `{ shared: true }`（借り物）の 2 形だけ。 */
  initializers: Record<string, { shared?: true }>;
  values: Record<string, { dtype: string; shape: (number | string)[] }>;
  /** 省略可能な state スロット節（ADR 0066 決定 2）— 既定の宣言は持たない。 */
  states?: Record<
    string,
    { dtype: string; shape: (number | string)[]; external?: boolean }
  >;
  nodes: {
    op: string;
    ins: string[];
    outs: string[];
    attrs: unknown;
    /** state スロットの名前参照（ADR 0067 決定 4）— 既定のノードは持たない。 */
    states?: Record<string, string>;
  }[];
};

/** 最小の正常系グラフ: y = x·w + b（x: T×4 → y: T×3）。 */
export const baseDeclaration = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  // MUST: 集合として扱う欄（`requires.ops` / `symbols`）は code point 順で書く。`krm` は正準
  // 直列化を通るので並びが揃い、krm 経路とメモリ経路のグラフを丸ごと比較できる。
  requires: { ops: ["add", "matmul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

/** {@link baseDeclaration} に対応する重み（f32 の丸ごと 1 本ずつ）。 */
export const baseTensors = (): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: f32Bytes(new Array(12).fill(0.5)),
    encoding: { codec: "f32" },
  },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes([1, 2, 3]),
    encoding: { codec: "f32" },
  },
];

const declarationOf = (graph: DeclarationJson): IrDeclaration => parseIrDeclarationValue(graph);

/** 宣言 + 重みを `krm`（単一形）のバイト列にする。 */
export const modelBytes = async (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
  options?: WriteOptions,
): Promise<Uint8Array<ArrayBuffer>> => {
  const written = await writeModelContainer({
    graphs: { [GRAPH_NAME]: declarationOf(graph) },
    consts: [],
    weights: tensors,
    assets: [],
    provenance: { license: "test" },
  }, options ?? {});
  return written.single;
};

/** {@link modelBytes} を開いたもの（2 文書の期待値は渡さない — テストとローカル読み込みの面）。 */
export const openModelBytes = async (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
  options?: WriteOptions,
): Promise<OpenedContainer> =>
  await openContainer({ kind: "bytes", bytes: await modelBytes(graph, tensors, options) });

/** 任意のグラフ（重み 0 本でもよい）を `krm` のバイト列にする。 */
export const graphModelBytes = (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): Promise<Uint8Array<ArrayBuffer>> => modelBytes(graph, tensors);

/**
 * 任意のグラフを `krm` に焼いて開く（重みを持たないグラフは `tensors` を省く）。
 *
 * {@link openModelBytes} との違いは既定の供給だけ — あちらは {@link baseTensors}、ここは 0 本。
 */
export const openGraphModel = async (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): Promise<OpenedContainer> =>
  await openContainer({ kind: "bytes", bytes: await graphModelBytes(graph, tensors) });

/** 容器の書き手と同じ入力を、メモリ内容器の供給（丸ごと 1 本ずつ）へ写す。 */
export const memoryTensorOf = (tensor: TensorInput): MemoryTensor => ({
  bytes: tensor.bytes,
  encoding: {
    codec: tensor.encoding.codec,
    ...(tensor.encoding.rowAxis === undefined ? {} : { rowAxis: tensor.encoding.rowAxis }),
    ...(tensor.encoding.groupSize === undefined ? {} : { groupSize: tensor.encoding.groupSize }),
    ...(tensor.encoding.scale === undefined ? {} : { scale: tensor.encoding.scale.bytes }),
  },
});

/** {@link modelBytes} と**同じ入力**からメモリ内容器を組む（krm 経路との A/B の片側）。 */
export const memoryModel = (
  graph: DeclarationJson = baseDeclaration(),
  tensors: readonly TensorInput[] = baseTensors(),
): BoundContainer =>
  openMemoryContainer({
    graphs: { [GRAPH_NAME]: declarationOf(graph) },
    tensors: {
      [GRAPH_NAME]: Object.fromEntries(
        tensors.map((tensor) => [tensor.initializer, memoryTensorOf(tensor)]),
      ),
    },
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

/**
 * 出力 slot の値名。slot 0 は `y`（既存テストがそのまま `outputs["y"]` を読む）で、
 * slot 1 以降は `y1`, `y2`, …（多出力 op — ADR 0068 決定 1）。
 */
export const outputName = (slot: number): string => (slot === 0 ? "y" : `y${slot}`);

/** 単一ノード宣言の dtype / attrs 指定（既定は全て f32・attrs 空）。 */
export type SingleOpOptions = {
  readonly inDtypes?: readonly string[];
  /** 出力 slot 昇順の dtype（既定は全 slot f32 — {@link singleOpDeclaration}）。 */
  readonly outDtypes?: readonly string[];
  readonly attrs?: Record<string, unknown>;
  readonly symbols?: readonly string[];
};

/**
 * 入力だけを取る単一ノードの宣言（op 単位の数値検証用）。
 *
 * `outShapes` は**出力 slot 昇順の列**（ADR 0068 決定 1 — 単一出力 op では長さ 1）。
 */
export const singleOpDeclaration = (
  op: string,
  inputShapes: readonly (readonly (number | string)[])[],
  outShapes: readonly (readonly (number | string)[])[],
  options: SingleOpOptions = {},
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
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

/**
 * 宣言済みの各 state スロットへ、それを参照する最小の `state_append` ノードを足す
 * （参照完全性 — ADR 0067 決定 4 / 5）。
 *
 * 宣言だけのグラフは「どのノードからも参照されない宣言」で落ちるので、**スロット宣言そのもの**
 * を見たいテストはこれで参照側を用意する。パーサは shape も契約も見ない層なので、`ins` に
 * 何を置くかは「定義済みの値名であること」しか効かない（形の検査は shape 計算 / 契約層の担当）。
 *
 * スロットが 1 本も無いグラフには何も足さない（`requires.ops` にだけ `state_append` が増えると、
 * 参照するノードが無いぶん「余剰」で宣言が落ちる）。
 */
export const withStateReaders = (graph: DeclarationJson): DeclarationJson => {
  const slots = Object.keys(graph.states ?? {});
  if (slots.length === 0) return graph;
  for (const slot of slots) {
    graph.nodes.push({ op: "state_append", ins: ["x"], outs: [], attrs: {}, states: { slot } });
  }
  if (!graph.requires.ops.includes("state_append")) graph.requires.ops.push("state_append");
  return graph;
};
