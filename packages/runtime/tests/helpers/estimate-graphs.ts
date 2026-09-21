// estimator の門が使う共有グラフと小物（estimate_test.ts と gpu_estimate_test.ts の両方から
// 同じ形を引くため。期待値そのものは各テストに置く — 手計算定数を helper へ逃がすと恒真化する）。

import { assertEquals } from "@std/assert";
import { type KarumeModel, openModel } from "../../src/format/container.ts";
import type { AdmissionReport, AdmissionScenario } from "../../src/runtime/estimate.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./f16.ts";
import { f32Bytes, type GraphJson, type TensorSpec } from "./format.ts";
import { graphModelBuffer } from "./graph.ts";

export const openGraph = (
  graph: GraphJson,
  tensors: readonly TensorSpec[] = [],
): KarumeModel => openModel(graphModelBuffer(graph, tensors));

/** f16 のバイト列（値そのものは見ないので 0 で埋める — 見るのはバイト数だけ）。 */
export const f16Zeros = (count: number): Uint8Array<ArrayBuffer> =>
  f16BytesFromBits(new Array(count).fill(f32ToF16Bits(0)));

/** 記号容量 `C` の k と数値容量の v を持つグラフ（append は 1 スロット 1 本 MUST）。 */
export const stateGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "state_append"] },
  symbols: ["T", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    chunk: { tensor: "m.chunk", storage: { dtype: "f32" } },
  },
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

export const stateModel = (): KarumeModel =>
  openGraph(stateGraph(), [
    { name: "m.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0)) },
    {
      name: "m.chunk",
      dtype: "F32",
      shape: [1, 2, 4, 4],
      data: f32Bytes(new Array(32).fill(0)),
    },
  ]);

/**
 * states 形 attention 1 本 + `state_append` 2 本（gpu_state_execution_test の実行形と同じ姿の
 * 最小版）。`B=1` / `Hkv=2`（GQA）/ `D=8` で、`M` が物理 chunk 行・`C` がスロット容量。
 *
 * `window` を渡すと sliding 変種（読み書き同式 MUST — attention と append の両方に載せる）。
 * `heads`（既定 4）は **S の 1 行バイト数 `H·colCap·4` だけを動かす**軸 — state スロットは
 * `Hkv·C·D·4` で H に依らないので、H を上げると「スロットは束縛上限に収まるが S の 1 行は
 * 収まらない」形（= 行ブロックが複数枚に割れる形）を絞った device 上で作れる。
 */
export const stateAttentionGraph = (window?: number, heads = 4): GraphJson => {
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
    version: 1,
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

/** prefill / decode の 2 本を名前つきで引く（並びの前提もここで一緒に押さえる）。 */
export const bothScenarios = (
  report: AdmissionReport,
): { readonly prefill: AdmissionScenario; readonly decode: AdmissionScenario } => {
  assertEquals(report.scenarios.map((scenario) => scenario.name), ["prefill", "decode"]);
  return { prefill: report.scenarios[0], decode: report.scenarios[1] };
};
