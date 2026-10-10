// estimator の門が使う共有グラフと小物（estimate_test.ts と gpu_estimate_test.ts の両方から
// 同じ形を引くため。期待値そのものは各テストに置く — 手計算定数を helper へ逃がすと恒真化する）。

import { assertEquals } from "@std/assert";
import type { AdmissionReport, AdmissionScenario } from "../../src/runtime/estimate.ts";
import { prepareContainer, type PreparedModel } from "../../src/runtime/executor.ts";
import type { EncodingInput, TensorInput } from "./container-write.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./f16.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME, memoryModel } from "./model-fixture.ts";

/**
 * 宣言 + 供給 → `PreparedModel`（admission の入口）。
 *
 * 供給元は**メモリ内容器**にしてある: 見積りは合流後の `IrGraph` だけで決まり `krm` の目次には
 * 1 バイトも依らないので、容器を書く（= sha256 を掛ける非同期）経路を通す理由が無い。
 */
export const openGraph = (
  graph: DeclarationJson,
  tensors: readonly TensorInput[] = [],
): PreparedModel => prepareContainer(memoryModel(graph, tensors), GRAPH_NAME);

/**
 * 供給 1 本（IR v2 では**initializer 名がそのまま実体の鍵**なので、渡すのは宣言の名前）。
 * 既定の格納は `f32`（丸ごと 1 本・companion scale なし）。
 */
export const weight = (
  initializer: string,
  bytes: Uint8Array<ArrayBuffer>,
  encoding: EncodingInput = { codec: "f32" },
): TensorInput => ({ graph: GRAPH_NAME, initializer, bytes, encoding });

/** f16 のバイト列（値そのものは見ないので 0 で埋める — 見るのはバイト数だけ）。 */
export const f16Zeros = (count: number): Uint8Array<ArrayBuffer> =>
  f16BytesFromBits(new Array(count).fill(f32ToF16Bits(0)));

/** 記号容量 `C` の k と数値容量の v を持つグラフ（append は 1 スロット 1 本 MUST）。 */
export const stateGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "state_append"] },
  symbols: ["T", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { w: {}, chunk: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    chunk: { dtype: "f32", shape: [1, 2, 4, 4] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  states: {
    k: { dtype: "f32", shape: [1, 2, "C", 4] },
    v: { dtype: "f32", shape: [1, 2, 6, 4] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "v" } },
  ],
});

/** {@link stateGraph} に対応する供給（値は見ないので 0 で埋める）。 */
export const stateTensors = (): readonly TensorInput[] => [
  weight("w", f32Bytes(new Array(12).fill(0))),
  weight("chunk", f32Bytes(new Array(32).fill(0))),
];

export const stateModel = (): PreparedModel => openGraph(stateGraph(), stateTensors());

/**
 * states 形 attention 1 本 + `state_append` 2 本（gpu_state_execution_test の実行形と同じ姿の
 * 最小版）。`B=1` / `Hkv=2`（GQA）/ `D=8` で、`M` が物理 chunk 行・`C` がスロット容量。
 *
 * `window` を渡すと sliding 変種（読み書き同式 MUST — attention と append の両方に載せる）。
 * `heads`（既定 4）は **S の 1 行バイト数 `H·colCap·4` だけを動かす**軸 — state スロットは
 * `Hkv·C·D·4` で H に依らないので、H を上げると「スロットは束縛上限に収まるが S の 1 行は
 * 収まらない」形（= 行ブロックが複数枚に割れる形）を絞った device 上で作れる。
 */
export const stateAttentionGraph = (window?: number, heads = 4): DeclarationJson => {
  const windowAttrs: Record<string, number> = window === undefined ? {} : { window };
  const append = (name: string, slot: string) => ({
    op: "state_append",
    ins: [name],
    outs: [] as string[],
    attrs: { ...windowAttrs },
    states: { slot },
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "state_append"] },
    symbols: ["M", "C"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", 8] },
      { name: "k", dtype: "f32", shape: [1, 2, "M", 8] },
      { name: "v", dtype: "f32", shape: [1, 2, "M", 8] },
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, heads, "M", 8] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 2, "C", 8] },
      vslot: { dtype: "f32", shape: [1, 2, "C", 8] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: 0.5, ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      },
      append("k", "kslot"),
      append("v", "vslot"),
    ],
  };
};

/**
 * エクスポータが出す分解 attention の 9 ノード窓（irodori DiT の綴り — 行ブロック attention に
 * 融合される形・ADR 0060）。全値 f32・`y` だけがグラフ出力。スコアは `heads·queries·keys·4` バイト。
 *
 * `interpose` は bmm と reshape の間に 0 バイトの別名を 1 本挟むだけで、窓の隣接条件だけを外す
 * （融合が成立しない同じ計算）。
 */
export const decomposedAttentionGraph = (
  dims: {
    readonly heads: number;
    readonly queries: number;
    readonly keys: number;
    readonly headDim: number;
  },
  interpose = false,
): DeclarationJson => {
  const { heads, queries, keys, headDim } = dims;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = interpose ? "s3_alias" : "s3";
  const nodes: DeclarationJson["nodes"] = [
    { op: "bmm", ins: ["q", "kt"], outs: [qkOut], attrs: {} },
    ...(interpose ? [{ op: "reshape", ins: [qkOut], outs: ["s3"], attrs: {} }] : []),
    { op: "reshape", ins: ["s3"], outs: ["s4"], attrs: {} },
    { op: "add", ins: ["s4", "mask"], outs: ["masked"], attrs: {} },
    { op: "safe_softmax", ins: ["masked"], outs: ["p4"], attrs: { dim: 3 } },
    { op: "expand", ins: ["p4"], outs: ["pe"], attrs: {} },
    { op: "reshape", ins: ["pe"], outs: ["p3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["ve"], attrs: {} },
    { op: "reshape", ins: ["ve"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["p3", "v3"], outs: ["y"], attrs: {} },
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      { name: "kt", dtype: "f32", shape: [heads, headDim, keys] },
      { name: "mask", dtype: "f32", shape: [1, 1, 1, keys] },
      { name: "v", dtype: "f32", shape: [1, heads, keys, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      ...(interpose ? { s3_alias: { dtype: "f32", shape: scores3 } } : {}),
      s3: { dtype: "f32", shape: scores3 },
      s4: { dtype: "f32", shape: scores4 },
      masked: { dtype: "f32", shape: scores4 },
      p4: { dtype: "f32", shape: scores4 },
      pe: { dtype: "f32", shape: scores4 },
      p3: { dtype: "f32", shape: scores3 },
      ve: { dtype: "f32", shape: [1, heads, keys, headDim] },
      v3: { dtype: "f32", shape: [heads, keys, headDim] },
      y: { dtype: "f32", shape: [heads, queries, headDim] },
    },
    nodes,
  };
};

/** prefill / decode の 2 本を名前つきで引く（並びの前提もここで一緒に押さえる）。 */
export const bothScenarios = (
  report: AdmissionReport,
): { readonly prefill: AdmissionScenario; readonly decode: AdmissionScenario } => {
  assertEquals(report.scenarios.map((scenario) => scenario.name), ["prefill", "decode"]);
  return { prefill: report.scenarios[0], decode: report.scenarios[1] };
};
