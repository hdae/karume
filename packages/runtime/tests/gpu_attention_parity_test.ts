// 融合 attention（ADR 0023）の**ビット同一の門**。本波で最重要のテスト。
//
// 同じ乱数入力に対し、
//
//   ① 既存 GPU op 列で組んだ分解経路（mul → permute → mul → bmm → softmax → expand → bmm）
//   ② `attention` op 1 ノード
//
// を**同じ実 GPU**で流し、出力を **f32 のビット列**で突き合わせる。allclose ではなく
// `Uint32Array` の完全一致で見るのが要点で、tolerance に隠れる丸め列の変化（例: scale を
// 内積の後に 1 度だけ掛ける「全スケール」化）はここでしか検出できない。
//
// MUST: 恒真化しないこと。①② は**別々のグラフ**を別々の Session で走らせる（同じカーネルを
// 2 回呼ぶ形にすると常に一致する）。分解経路のノード列は Anima DiT の実測グラフ（設計 recon
// §1.1 の #60〜#74）と同じ順序・同じ op で組んである。
//
// MUST: B / H / M / N / D は複数形状（端数込み）を回す。カーネルは B と H を 1 本のバッチ軸へ
// 畳むので、B=1 だけでは軸の取り違えが値に出ない。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 半スケール（torch math decomp の `√scale_factor`）。D から導く契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

type Shape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * 分解経路のグラフ（torch SDPA の math decomp を IR へ写したもの）。
 *
 * `sc` は半スケールを 1 要素で渡す入力で、実グラフの `const.6953fe58410d6c34` に対応する
 * （initializer でも入力でも `mul` の右詰め broadcast は同じ）。恒等 `expand` まで含めるのは、
 * 実グラフの P 側 56 本と同じコピーを経由させて「融合で消える 1 枚」を再現するため。
 */
const decomposedGraph = (shape: Shape): GraphJson => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["mul", "permute", "reshape", "bmm", "softmax", "expand"] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [b, h, m, d] },
      { name: "k", dtype: "f32", shape: [b, h, n, d] },
      { name: "v", dtype: "f32", shape: [b, h, n, d] },
      { name: "sc", dtype: "f32", shape: [1] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      qs: { dtype: "f32", shape: [b, h, m, d] },
      kt: { dtype: "f32", shape: [b, h, d, n] },
      kts: { dtype: "f32", shape: [b, h, d, n] },
      qs3: { dtype: "f32", shape: [heads, m, d] },
      kts3: { dtype: "f32", shape: [heads, d, n] },
      scores3: { dtype: "f32", shape: [heads, m, n] },
      scores4: { dtype: "f32", shape: [b, h, m, n] },
      probs: { dtype: "f32", shape: [b, h, m, n] },
      probsExpanded: { dtype: "f32", shape: [b, h, m, n] },
      probs3: { dtype: "f32", shape: [heads, m, n] },
      v3: { dtype: "f32", shape: [heads, n, d] },
      out3: { dtype: "f32", shape: [heads, m, d] },
      y: { dtype: "f32", shape: [b, h, m, d] },
    },
    nodes: [
      // #60 mul q × scale（半スケールの q 側）
      { op: "mul", ins: ["q", "sc"], outs: ["qs"], attrs: {} },
      // #61 permute k → kᵀ
      { op: "permute", ins: ["k"], outs: ["kt"], attrs: { dims: [0, 1, 3, 2] } },
      // #62 mul kᵀ × scale（半スケールの k 側 — 同じ定数）
      { op: "mul", ins: ["kt"], outs: ["kts"], attrs: {} },
      { op: "reshape", ins: ["qs"], outs: ["qs3"], attrs: {} },
      { op: "reshape", ins: ["kts"], outs: ["kts3"], attrs: {} },
      // #67 bmm → S
      { op: "bmm", ins: ["qs3", "kts3"], outs: ["scores3"], attrs: {} },
      { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
      // #69 softmax dim=-1
      { op: "softmax", ins: ["scores4"], outs: ["probs"], attrs: { dim: 3 } },
      // #70 恒等 expand（P のフルコピー — 融合で消える 1 枚）
      { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
      { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
      { op: "reshape", ins: ["v"], outs: ["v3"], attrs: {} },
      // #74 bmm → O
      { op: "bmm", ins: ["probs3", "v3"], outs: ["out3"], attrs: {} },
      { op: "reshape", ins: ["out3"], outs: ["y"], attrs: {} },
    ],
  };
};

/** 上のグラフの `mul` は 2 本とも同じ scale 入力を取る（k 側の ins を後から差す）。 */
const withScaleInput = (graph: GraphJson): GraphJson => {
  for (const node of graph.nodes) {
    if (node.op === "mul" && node.ins.length === 1) node.ins.push("sc");
  }
  return graph;
};

const run = async (
  gpu: GpuContext,
  graph: GraphJson,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<Tensor> => {
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    return (await session.run(inputs))["y"];
  } finally {
    await session.dispose();
  }
};

/** f32 のビット列（`0.0` と `-0.0` の差も、末尾 1 ulp の差も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const SHAPES: readonly Shape[] = [
  // B / H / M / N / D が全て違う cross-attention 形（軸の取り違えが値に出る）
  { name: "全異 B2 H3 M5 N11 D7", b: 2, h: 3, m: 5, n: 11, d: 7 },
  // 全てタイル端数（M % 64 ≠ 0 / N % 64 ≠ 0 / D % 4 ≠ 0 = スカラ変種）
  { name: "端数 B3 H1 M17 N19 D13", b: 3, h: 1, m: 17, n: 19, d: 13 },
  // v4 経路（① は D%4 && N%4・③ は N%4 && D%4）で行タイル 2 枚を跨ぐ
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, m: 68, n: 20, d: 12 },
  // ① が v4 で ③ がスカラに落ちる形（変種の踏み分けが ① と ③ で違うことの固定）
  { name: "混成 B2 H2 M9 N8 D6", b: 2, h: 2, m: 9, n: 8, d: 6 },
  // DiT の self-attention を縮めた形（D=128・M/N とも 64 の倍数 = 実モデルの経路）
  { name: "DiT 形 B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

Deno.test({
  name: "attention 1 ノードの出力が分解経路（bmm/softmax/bmm）と**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const sc = fill([1], () => scale);

        const fused = await run(
          gpu,
          singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], {
            attrs: { scale },
          }),
          { x0: q, x1: k, x2: v },
        );
        const split = await run(gpu, withScaleInput(decomposedGraph(shape)), { q, k, v, sc });

        assertEquals(fused.shape, split.shape, shape.name);
        const a = bits(fused);
        const c = bits(split);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
          if (a[i] !== c[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches,
          [],
          `${shape.name}: 分解経路とビット列が違う（最初の食い違い: ${
            mismatches.map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`).join(" / ")
          }）`,
        );
        // 恒真化の門: 出力が全て同じ値なら「一致」は何も検証していない
        assert(
          new Set(a).size > 1,
          `${shape.name}: 出力が定数（ビット一致が恒真になっている）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});
