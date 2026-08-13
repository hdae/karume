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

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { attentionPvKey, attentionQkKey } from "../src/kernels/attention.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

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
 *
 * `masked` は加算 mask を持つ形（ADR 0023 改訂）。**S を実体化した後の `add` 1 本**として
 * 書くのが要点で、融合側は同じ加算を ①QK の書き出し epilogue で行う（丸めの位置も回数も
 * 同じ = ビット同一の根拠）。
 */
const decomposedGraph = (shape: Shape, masked = false): GraphJson => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;
  return {
    format: "karume-ir",
    version: 1,
    requires: {
      ops: ["mul", "permute", "reshape", "bmm", "softmax", "expand", ...(masked ? ["add"] : [])],
    },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [b, h, m, d] },
      { name: "k", dtype: "f32", shape: [b, h, n, d] },
      { name: "v", dtype: "f32", shape: [b, h, n, d] },
      { name: "sc", dtype: "f32", shape: [1] },
      ...(masked ? [{ name: "mk", dtype: "f32" as const, shape: [1, 1, m, n] }] : []),
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
      ...(masked ? { scoresMasked: { dtype: "f32" as const, shape: [b, h, m, n] } } : {}),
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
      // 加算 mask（S を実体化した後の 1 加算 — [1,1,M,N] は右詰め broadcast で B·H へ広がる）
      ...(masked ? [{ op: "add", ins: ["scores4", "mk"], outs: ["scoresMasked"], attrs: {} }] : []),
      // #69 softmax dim=-1
      {
        op: "softmax",
        ins: [masked ? "scoresMasked" : "scores4"],
        outs: ["probs"],
        attrs: {
          dim: 3,
        },
      },
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

/**
 * 実測形の band mask（EmbeddingGemma の双方向 sliding window と同じ作り）。
 *
 * MUST: **全ての行に非マスク列が 1 本以上**あること（行が丸ごとマスクだと amax が −inf に
 * なり、`exp(−inf − (−inf))` が NaN になる — 融合側も分解側も同じ NaN を出すので
 * ビット比較は通るが、検証としては何も見ていない状態になる）。対称バンドは中心が必ず
 * 非マスクなので、この条件が構造で満たされる。
 */
const bandMask = (
  m: number,
  n: number,
  width: number,
  blocked: number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    // M ≠ N（cross-attention 形）でも「行に対応する列」を中心に置く
    const center = Math.floor((row * n) / m);
    for (let col = 0; col < n; col += 1) {
      data[row * n + col] = Math.abs(col - center) <= width ? 0 : blocked;
    }
  }
  return data;
};

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

/**
 * **加算 mask 版のビット同一の門**（ADR 0023 改訂）。融合側は ①QK の書き出し epilogue で
 * `S' = fl(S + mask[m·N+n])` を足すだけなので、分解側の `bmm`（S を実体化）→ `add`（mask）と
 * 丸めの位置も回数も一致する — 一致しなくなるのは添字（行/列・バッチ base）を取り違えたときで、
 * それは tolerance に隠れずビット比較で必ず出る。
 *
 * mask の値は **0 と大きい負値の混在**（実測の band mask）で、最後の 1 形だけ `-Infinity`
 * （エクスポータが −inf を折り込む形）を通す。
 */
Deno.test({
  name:
    "加算 mask 付き attention の出力が分解経路（bmm → add → softmax → bmm）とビット単位で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const [index, shape] of SHAPES.entries()) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const sc = fill([1], () => scale);
        // 最後の形だけ −inf（残りは実測 mask の大きい負値 — masked_fill の埋め値と同じ桁）
        const blocked = index === SHAPES.length - 1 ? -Infinity : -3.4028234663852886e38;
        const mask = {
          dtype: "f32" as const,
          shape: [1, 1, m, n],
          data: bandMask(m, n, 2, blocked),
        };
        const where = `${shape.name} (blocked=${blocked})`;

        const fused = await run(
          gpu,
          singleOpGraph("attention", [q.shape, k.shape, v.shape, mask.shape], [b, h, m, d], {
            attrs: { scale },
          }),
          { x0: q, x1: k, x2: v, x3: mask },
        );
        const split = await run(gpu, withScaleInput(decomposedGraph(shape, true)), {
          q,
          k,
          v,
          sc,
          mk: mask,
        });

        assertEquals(fused.shape, split.shape, where);
        const a = bits(fused);
        const c = bits(split);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
          if (a[i] !== c[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches,
          [],
          `${where}: 分解経路とビット列が違う（最初の食い違い: ${
            mismatches.map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`).join(" / ")
          }）`,
        );
        // 恒真化の門 ①: 出力が定数なら「一致」は何も検証していない
        assert(new Set(a).size > 1, `${where}: 出力が定数（ビット一致が恒真になっている）`);
        // 恒真化の門 ②: mask が結果に効いていること（mask 無しの同じ入力と**違う**値になる）。
        // 添字を取り違えても値は変わるが、この門が守るのは「mask が丸ごと無視されていない」形。
        const unmasked = await run(
          gpu,
          singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], {
            attrs: { scale },
          }),
          { x0: q, x1: k, x2: v },
        );
        const plain = bits(unmasked);
        assert(
          a.some((word, i) => word !== plain[i]),
          `${where}: mask 付きと mask 無しの出力が同一（mask が効いていない）`,
        );
        // NaN が出ていない（band mask は全行に非マスク列を残す — 上の MUST の実測確認）
        assert(
          fused.data.every((value) => Number.isFinite(value)),
          `${where}: 出力に非有限値（行が丸ごとマスクされている）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * Session 経路の結線（ADR 0023 改訂）。mask で増えるのは **①QK の束縛 1 本とキーの 1 語**
 * だけで、dispatch は 3 本のまま（②③ は mask の存在を知らない）。
 *
 * i8a8 の ①QK は別カーネルで epilogue を持たないので、組み合わせは **fail loudly**
 * （黙って f32 へ縮退すると「i8a8 を頼んだのに効かない」沈黙になり、mask を落とすと値が壊れる）。
 */
Deno.test({
  name: "加算 mask は ①QK のキーと束縛だけを増やし、i8a8 とは組めない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const { b, h, m, n, d } = { b: 1, h: 2, m: 8, n: 8, d: 8 };
      const q = fill([b, h, m, d], QUERY);
      const k = fill([b, h, n, d], KEY);
      const v = fill([b, h, n, d], VALUE);
      const mask = { dtype: "f32" as const, shape: [1, 1, m, n], data: bandMask(m, n, 2, -1e30) };
      const graph = singleOpGraph(
        "attention",
        [q.shape, k.shape, v.shape, mask.shape],
        [b, h, m, d],
        { attrs: { scale: halfScale(d) } },
      );
      const inputs = { x0: q, x1: k, x2: v, x3: mask };

      const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
      try {
        await session.run(inputs);
        const entries = session.diagnostics().lastRunTiming?.entries ?? [];
        if (entries.length > 0) {
          const keys = entries.map((entry) => entry.key);
          // 1 ノード = 3 dispatch のまま（mask で dispatch は増えない）
          assertEquals(entries.reduce((sum, entry) => sum + entry.dispatchCount, 0), 3);
          assertEquals(keys.includes(attentionQkKey(true, "f32", "f32", true)), true, "mask 変種");
          assertEquals(keys.includes(attentionQkKey(true)), false, "mask 無しの ①QK が残っている");
          // ②③ は mask なしと同じキー（生成物も同じ = スナップショットが固定）
          assertEquals(keys.includes(attentionPvKey(true)), true, "③PV は mask で変わらない");
        }
      } finally {
        await session.dispose();
      }

      // i8a8 との組み合わせは fail loudly（縮退しない）
      const i8a8 = await createSession(gpu, openModel(graphModelBuffer(graph)), {
        attentionCompute: "i8a8",
      });
      try {
        await assertRejects(() => i8a8.run(inputs), ExecutionError, "i8a8");
      } finally {
        await i8a8.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * mask × `attentionCompute: "i8a8"` の拒否は **D / N の 4 の倍数性に依らない**（ADR 0023 /
 * docs/limitations.md の「無条件 fail loudly」）。段ごとの適格判定（① は `D % 4`・③ は
 * `N % 4`）で拒否を決めると `D % 4 != 0` の形だけ素通りし、**f32 の ①QK と i8a8 の ③PV の
 * 混成**で走ってしまう — 値は正しくても「組めない」という契約が破れる。
 *
 * 恒真化の門: **同じ形の mask 無し**は i8a8 で走ること（拒否の原因が mask であって
 * 「その形が i8a8 で走らない」ではないことを固定する）。
 */
Deno.test({
  name: "mask × attentionCompute 'i8a8' は D%4 / N%4 に依らず一貫して拒否される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // ①QK 適格（D%4）× ③PV 適格（N%4）の 3 通り。真×真 は従来から拒否され、残り 2 つが
    // 素通りしていた（偽×真 = 混成 / 偽×偽 = 全段 f32 への沈黙の縮退）。
    const shapes: readonly Shape[] = [
      { name: "①③とも適格 D8 N8", b: 1, h: 2, m: 8, n: 8, d: 8 },
      { name: "③のみ適格 D13 N20", b: 1, h: 2, m: 8, n: 20, d: 13 },
      { name: "①③とも非適格 D13 N19", b: 1, h: 2, m: 8, n: 19, d: 13 },
    ];
    const gpu = await acquireGpu();
    try {
      for (const shape of shapes) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const mask = { dtype: "f32" as const, shape: [1, 1, m, n], data: bandMask(m, n, 2, -1e30) };
        const attrs = { scale: halfScale(d) };

        const masked = await createSession(
          gpu,
          openModel(graphModelBuffer(
            singleOpGraph("attention", [q.shape, k.shape, v.shape, mask.shape], [b, h, m, d], {
              attrs,
            }),
          )),
          { attentionCompute: "i8a8" },
        );
        try {
          // 文言・型は D%4==0 経路の拒否と同じもの（分岐ごとに別の拒否を作らない）
          await assertRejects(
            () => masked.run({ x0: q, x1: k, x2: v, x3: mask }),
            ExecutionError,
            "加算 mask 付きの attention は attentionCompute 'i8a8' と組めない",
          );
        } finally {
          await masked.dispose();
        }

        const plain = await createSession(
          gpu,
          openModel(graphModelBuffer(
            singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], { attrs }),
          )),
          { attentionCompute: "i8a8" },
        );
        try {
          const y = (await plain.run({ x0: q, x1: k, x2: v }))["y"];
          assertEquals(y.shape, [b, h, m, d], shape.name);
          assert(
            y.data.every((value) => Number.isFinite(value)),
            `${shape.name}: mask 無しの i8a8 出力に非有限値`,
          );
        } finally {
          await plain.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});
