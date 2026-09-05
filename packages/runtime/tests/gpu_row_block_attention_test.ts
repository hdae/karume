/**
 * 分解 attention の**行ブロック実行**（融合ルール `rowBlockAttention` — src/runtime/fusion.ts）の
 * 実 GPU の門。見るのは 2 つで、どちらも実機でしか立たない。
 *
 * ① **強制分割 parity**: 同じ入力に対し、
 *    - 掴めない綴り（bmm と reshape の間に 0 dispatch の別名を 1 本挟んだだけ）の**素の 9 ノード列**
 *    - 融合した 1 枚（既定 = 素の 4 dispatch 列と同一）
 *    - 融合した n 枚（`ROW_BLOCK_SPLIT` で 2 / 3 / 端数割を強制）
 *    を回して **f32 のビット列**で突き合わせる。allclose ではなく `Uint32Array` の完全一致で
 *    見るのが要点 — 行ブロック化は「どの workgroup がどの行を担当するか」しか変えないので、
 *    1 ulp でも動いたら K 縮約順か丸め列が動いている。
 *
 * ② **ポータビリティ門**: `maxStorageBufferBindingSize` を WebGPU core 既定（128MiB）へ**絞った
 *    device**（`LIMIT_CAPS`）で、S が 128MiB を超える合成グラフが緑になること。**同じグラフを
 *    掴めない綴りにすると同じ device で落ちること**も対で見る（落ちない形を緑にしても、
 *    行ブロックが効いている証拠にならない）。
 *
 * MUST: 恒真化しないこと。①は**別々のグラフ / 別々の Session** を回す（同じ経路を 2 回撃つ形は
 * 常に一致する）。出力が定数でないことも毎回確かめる。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { planRowBlocks } from "../src/runtime/fusion.ts";
import { createSession, ROW_BLOCK_SPLIT, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** WebGPU core 既定のストレージ束縛上限。ポータビリティ門はここを再現する。 */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

type Shape = {
  readonly name: string;
  /** B·H を畳んだバッチ軸。 */
  readonly heads: number;
  /** クエリ行数 M。 */
  readonly queries: number;
  /** キー列数 N。 */
  readonly keys: number;
  /** head 幅 D。 */
  readonly headDim: number;
};

/** 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/**
 * `[1,1,1,N]` の加算 mask（実測形と同じ「列ごと」の broadcast）。
 *
 * 4 列に 1 本 −inf を置いて safe_softmax の −inf 経路を通す。MUST: 全列を塞がないこと —
 * 全 −inf の行は safe_softmax が全 0 を書くので、ビット比較が「0 と 0」の恒真になる。
 */
const columnMask = (keys: number): FilledTensor =>
  fill([1, 1, 1, keys], (i) => (i % 4 === 3 ? Number.NEGATIVE_INFINITY : 0));

/**
 * 分解 attention の実測綴り（irodori DiT の 9 ノード窓そのもの）。
 *
 * `interpose` は bmm と reshape の間に 0 dispatch の別名を 1 本挟むだけ。値も物理 dispatch 数も
 * 変えずに窓の隣接条件だけを外すので、**同じバックエンド上の正本**（融合前の経路）になる。
 */
const attentionGraph = (shape: Shape, interpose = false): GraphJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = interpose ? "scores3_alias" : "scores3";
  const nodes: GraphJson["nodes"] = [
    { op: "bmm", ins: ["q", "kt"], outs: [qkOut], attrs: {} },
    ...(interpose ? [{ op: "reshape", ins: [qkOut], outs: ["scores3"], attrs: {} }] : []),
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    { op: "add", ins: ["scores4", "mask"], outs: ["masked"], attrs: {} },
    { op: "safe_softmax", ins: ["masked"], outs: ["probs"], attrs: { dim: 3 } },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["probs3", "v3"], outs: ["y"], attrs: {} },
  ];
  return {
    format: "karume-ir",
    version: 1,
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
      ...(interpose ? { scores3_alias: { dtype: "f32", shape: scores3 } } : {}),
      scores3: { dtype: "f32", shape: scores3 },
      scores4: { dtype: "f32", shape: scores4 },
      masked: { dtype: "f32", shape: scores4 },
      probs: { dtype: "f32", shape: scores4 },
      probsExpanded: { dtype: "f32", shape: scores4 },
      probs3: { dtype: "f32", shape: scores3 },
      vExpanded: { dtype: "f32", shape: [1, heads, keys, headDim] },
      v3: { dtype: "f32", shape: [heads, keys, headDim] },
      y: { dtype: "f32", shape: [heads, queries, headDim] },
    },
    nodes,
  };
};

const attentionInputs = (shape: Shape): Readonly<Record<string, FilledTensor>> => ({
  q: fill([shape.heads, shape.queries, shape.headDim], QUERY),
  kt: fill([shape.heads, shape.headDim, shape.keys], KEY),
  mask: columnMask(shape.keys),
  v: fill([1, shape.heads, shape.keys, shape.headDim], VALUE),
});

/** 出力と、その run が実際に確保した中間バッファのピーク（行ブロックの効きの観測点）。 */
type RunResult = {
  readonly output: Tensor;
  readonly peakTransientBytes: number;
};

const run = async (
  gpu: GpuContext,
  graph: GraphJson,
  inputs: Readonly<Record<string, FilledTensor>>,
  split?: number,
): Promise<RunResult> => {
  const session = await createSession(
    gpu,
    openModel(graphModelBuffer(graph)),
    split === undefined ? {} : { [ROW_BLOCK_SPLIT]: split },
  );
  try {
    const output = (await session.run(inputs))["y"];
    return {
      output,
      peakTransientBytes: session.diagnostics().lastRun?.peakTransientBytes ?? 0,
    };
  } finally {
    await session.dispose();
  }
};

/** f32 のビット列（`0.0` と `-0.0` の差も、末尾 1 ulp の差も見える形）。 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertSameBits = (actual: Tensor, expected: Tensor, label: string): void => {
  assertEquals(actual.shape, expected.shape, label);
  const a = bits(actual);
  const b = bits(expected);
  const mismatches: number[] = [];
  for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
    if (a[i] !== b[i]) mismatches.push(i);
  }
  assertEquals(
    mismatches,
    [],
    `${label}: ビット列が違う（最初の食い違い: ${
      mismatches.map((i) => `[${i}] ${actual.data[i]} vs ${expected.data[i]}`).join(" / ")
    }）`,
  );
  // 恒真化の門: 出力が定数なら「一致」は何も検証していない。
  assert(new Set(a).size > 1, `${label}: 出力が定数（ビット一致が恒真になっている）`);
};

const SHAPES: readonly Shape[] = [
  // 全軸が端数（M / N / D とも 4 の倍数でない = 2 本の bmm ともスカラ変種）
  { name: "端数 H3 M17 N19 D13", heads: 3, queries: 17, keys: 19, headDim: 13 },
  // v4 経路（D % 4 == 0 && N % 4 == 0）で M が行タイル 2 枚を跨ぐ。分割すると幾何のバケット
  // （M ≤ 64 / ≤ 512）も跨ぐので、「幾何が変わっても値は動かない」がここで確かめられる。
  { name: "v4 H2 M68 N20 D12", heads: 2, queries: 68, keys: 20, headDim: 12 },
  // DiT の self-attention を縮めた形（D = 128・M / N とも 64 の倍数 = 実モデルの経路）
  { name: "DiT 形 H4 M64 N64 D128", heads: 4, queries: 64, keys: 64, headDim: 128 },
];

/** 強制する枚数（1 = 既定と同じ形・2/3 は等分・5 は端数割）。 */
const SPLITS: readonly number[] = [1, 2, 3, 5];

Deno.test({
  name: "行ブロック実行は枚数を変えても素の 9 ノード列と**ビット単位で一致**する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const inputs = attentionInputs(shape);
        // 正本 = 掴めない綴り（融合が 1 度も掛からない素の 9 ノード列）。
        const reference = await run(gpu, attentionGraph(shape, true), inputs);
        let previousPeak = Number.POSITIVE_INFINITY;
        for (const split of SPLITS) {
          const fused = await run(gpu, attentionGraph(shape), inputs, split);
          const label = `${shape.name} / ${split} 枚`;
          assertSameBits(fused.output, reference.output, label);
          // 枚数を増やすほど中間のピークは下がる（値が同じでも**実体化幅が縮んでいる**ことの
          // 観測点 — ここが単調でなければ一時の寿命宣言かプール再利用が壊れている）。
          assert(
            fused.peakTransientBytes < previousPeak,
            `${label}: 中間ピーク ${fused.peakTransientBytes}B が ${previousPeak}B から縮んでいない`,
          );
          previousPeak = fused.peakTransientBytes;
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * ポータビリティ門の合成グラフ。**実資産を要らない**形で `S = H·M·N·4` を 128MiB 超へ持ち上げる
 * （4 × 2048 × 4200 × 4 = 137,625,600B）。D を 8 に絞ってあるので GEMM の仕事量自体は小さく、
 * 見ているのは「上限を超える中間を実体化せずに済むか」の 1 点だけ。
 */
const OVERSIZE: Shape = {
  name: "上限超 H4 M2048 N4200 D8",
  heads: 4,
  queries: 2048,
  keys: 4200,
  headDim: 8,
};

Deno.test({
  name:
    "maxStorageBufferBindingSize=128MiB へ絞った device で、S が上限を超えるグラフが行ブロックで通る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const scoreBytes = OVERSIZE.heads * OVERSIZE.queries * OVERSIZE.keys * 4;
    assert(
      scoreBytes > CORE_STORAGE_BINDING_LIMIT,
      `合成グラフの S ${scoreBytes}B が core 既定の上限を超えていない（門が空振りする）`,
    );
    // 実行が読むのと**同じ純関数**で、この形が 2 枚に割れて 1 枚が上限の内側へ収まることを
    // 先に固定する（GPU 側が緑になった理由を「たまたま」にしない）。
    const blocks = planRowBlocks(
      OVERSIZE.queries,
      OVERSIZE.heads * OVERSIZE.keys * 4,
      CORE_STORAGE_BINDING_LIMIT,
    );
    assertEquals(blocks.length, 2, "上限に収まる最小枚数");
    for (const block of blocks) {
      assert(
        block.rows * OVERSIZE.heads * OVERSIZE.keys * 4 <= CORE_STORAGE_BINDING_LIMIT,
        `ブロック ${block.offset}+${block.rows} が上限に収まらない`,
      );
    }
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits が絞られていない（絞れていなければ門は何も見ていない）",
      );
      const inputs = attentionInputs(OVERSIZE);
      const fused = await run(gpu, attentionGraph(OVERSIZE), inputs);
      assertEquals(fused.output.shape, [OVERSIZE.heads, OVERSIZE.queries, OVERSIZE.headDim]);
      assert(new Set(bits(fused.output)).size > 1, "出力が定数（門が空振りしている）");

      // 門の効力証明: **同じ device・同じ形**でも、窓を崩して素の 9 ノード列に落とすと
      // S を丸ごと実体化しようとして落ちる。ここが緑のままなら上の緑は行ブロックの
      // おかげではない。落とすのは計画時の上限 preflight（ADR 0093 決定 5 — GPU の validation
      // より前に、上限を超える中間をノード名つきで全件列挙する）。
      await assertRejects(
        () => run(gpu, attentionGraph(OVERSIZE, true), inputs),
        ExecutionError,
        `中間バッファが device の上限を超える`,
      );
    } finally {
      gpu.destroy();
    }
  },
});
