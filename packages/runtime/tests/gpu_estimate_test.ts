// メモリ必要量 estimator（ADR 0070 決定 5）の実 GPU 突合（GPU に依らない門は estimate_test.ts）。
//
// 厳密一致を主張できるのは診断が実測している 2 カテゴリ（圧縮常駐・展開）と state 容量、および
// 融合が 1 本も掛からない states 形 attention の slot 表。中間ピークは**近似**なので突合しない
// （融合が中間を消し、行ブロック分割が一時を足す — どちらも estimator の unaccounted 欄が
// 認めている差）。アダプタ無しは明示 SKIP。

import { assertEquals } from "@std/assert";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import type { PreparedModel } from "../src/runtime/executor.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";
import {
  bothScenarios,
  f16Zeros,
  openGraph,
  stateAttentionGraph,
  stateModel,
  weight,
} from "./helpers/estimate-graphs.ts";
import { type DeclarationJson, f32Bytes, fill } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";

/**
 * S / 行統計の算式が recipe-builder と同じ導出元（`planStateAttention`）から出ていることの
 * 唯一の実測門。
 *
 * このグラフは融合が 1 本も掛からない（states を触るノードは窓を掴まない — ADR 0067 決定 5b）
 * ので、`workspaceBytes` は slot 表の総バイト = `planBacking.residentBytes` と**厳密一致**する。
 * 算式そのものは両者が共有する（= ここでは割れない）が、共有関数へ**渡す材料**（`B·H`・窓・
 * 容量の出どころ）と、返ったバイト数を実行相が確保する位置・サイズクラス再利用の規則が
 * estimator の写しとずれれば、ここが例外なしで割れる。
 *
 * 呼ぶのは full / sliding の 2 変種（下の 2 本）— `colCap` は変種で式が分かれる唯一の欄なので、
 * 片方だけでは分岐のもう一方が無門のままになる。
 *
 * `limitCap` を渡すと device の `maxStorageBufferBindingSize` を絞って**行ブロックを複数枚に
 * 割る**（estimator は `ROW_BLOCK_SPLIT` の受け口を持たないので、枚数を寄せる手は形と上限しか
 * 無い）。複数枚でだけ効く 2 つの規則 — ブロック跨ぎのプール再利用と、端数で 1 行狭い
 * ブロックが混ざったときのサイズクラス 2 種 — は、絞らない呼び方では 1 度も踏まれない。
 * MUST: 絞ったときは `expectedBlocks` を渡して枚数を先に固定する（1 枚に落ちた形で緑になると
 * 「複数枚での一致」を見たことにならない）。
 */
const assertPlanBackingMatchesEstimate = async (
  variant: {
    readonly capacity: number;
    readonly window?: number;
    readonly heads?: number;
    readonly chunkLength?: number;
    readonly limitCap?: number;
    readonly expectedBlocks?: number;
  },
): Promise<void> => {
  const heads = variant.heads ?? 4;
  const chunkLength = variant.chunkLength ?? 4;
  const gpu = await acquireGpu(
    variant.limitCap === undefined
      ? {}
      : { [LIMIT_CAPS]: { maxStorageBufferBindingSize: variant.limitCap } },
  );
  try {
    const limit = gpu.limits.maxStorageBufferBindingSize;
    if (variant.limitCap !== undefined) {
      assertEquals(limit, variant.limitCap, "requiredLimits が絞られていない（門が空振りする）");
    }
    if (variant.expectedBlocks !== undefined) {
      // 実行と見積りが共有する純関数そのもので枚数を固定する（前提の可視化）。
      const blocks = planStateAttention({
        batchHeads: heads,
        chunkRows: chunkLength,
        capacity: variant.capacity,
        window: variant.window,
      }, limit).blocks;
      assertEquals(blocks.length, variant.expectedBlocks, "行ブロックの枚数");
    }
    const model = openGraph(stateAttentionGraph(variant.window, heads));
    const generation = { chunkLength, bindings: { C: variant.capacity } };
    const { prefill } = bothScenarios(
      model.estimate({ generation, maxStorageBufferBindingSize: limit }),
    );
    const session = await model.createContainerSession(gpu);
    try {
      const context = await session.createGenerationContext(generation);
      try {
        // slot backing は同じ signature の 2 run 目で組まれる（1 run 目はアリーナ経路）。
        for (let step = 0; step < 2; step += 1) {
          await session.run(
            {
              q: fill([1, heads, chunkLength, 8], (i) => ((i % 5) - 2) / 4),
              k: fill([1, 2, chunkLength, 8], (i) => ((i % 3) - 1) / 4),
              v: fill([1, 2, chunkLength, 8], (i) => ((i % 7) - 3) / 4),
            },
            {},
            { context, queryLength: chunkLength },
          );
        }
      } finally {
        await context.dispose();
      }
      assertEquals(session.diagnostics().planBacking.residentBytes, prefill.workspaceBytes);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

Deno.test({
  name: "states 形 attention の中間総量が実行計画の slot 表と厳密一致する（full 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = C = 16（S は容量に比例する側）。
  fn: () => assertPlanBackingMatchesEstimate({ capacity: 16 }),
});

Deno.test({
  name: "states 形 attention の中間総量が実行計画の slot 表と厳密一致する（sliding 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = W−1+M = 8−1+4 = 11（容量 C とは別の式 — full と同じ数にならない形を選ぶ）。
  fn: () => assertPlanBackingMatchesEstimate({ capacity: 8, window: 8 }),
});

/**
 * 行ブロック**複数枚**での 2 実装一致。H=64 / Hkv=2 / D=8 / C=16 では S の 1 行が
 * 64·16·4 = 4096B なのに対し state スロットは 2·16·8·4 = 1024B なので、上限 8192B に絞ると
 * 「スロットは束縛できるが S 4 行は束縛できない」形になり、行ブロックが必ず割れる。
 */
Deno.test({
  name: "states 形 attention の中間総量が複数枚でも slot 表と厳密一致する（full 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 1 枚 2 行の 2 枚（8192 ÷ 4096 = 2 行／枚・M=4）。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 4,
      limitCap: 8192,
      expectedBlocks: 2,
    }),
});

Deno.test({
  name: "states 形 attention の中間総量が端数ブロックでも slot 表と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  // M=5 を 2 行／枚で割ると 3 枚（2 行 + 2 行 + 1 行）— サイズクラスが 2 種同時に生きる唯一の形。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 5,
      // 出力 o [1,64,5,8] = 10240B が束縛上限に収まる最小の絞り（S は 1 行 4096B → 2 行 × 3 枚）
      limitCap: 10240,
      expectedBlocks: 3,
    }),
});

Deno.test({
  name: "states 形 attention の中間総量が複数枚でも slot 表と厳密一致する（sliding 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = W−1+M = 8−1+4 = 11 → 1 行 64·11·4 = 2816B。上限 5632B で 2 行／枚の 2 枚。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      window: 8,
      chunkLength: 4,
      // 出力 o [1,64,4,8] = 8192B が束縛上限に収まる絞り（S は 1 行 64×11×4 = 2816B → 2 行 × 2 枚）
      limitCap: 8192,
      expectedBlocks: 2,
    }),
});

// ---------------------------------------------------------------------------
// 実 GPU 突合
// ---------------------------------------------------------------------------

/**
 * 3 つの欄が全て非 0 になるモデル — i8 の embedding 表（適格 = 圧縮のまま常駐）・f16 の
 * mul 被演算子（適格外 = f32 展開）・f32 の add 被演算子（非圧縮のまま常駐）。
 *
 * 圧縮 / 展開の 2 欄は診断 `storage` と厳密一致を主張でき、f32 は診断に現れない
 * （`weights.uncompressedBytes` の欄を分けている理由そのもの）ので手計算定数と突合する。
 */
const gpuWeightModel = (): PreparedModel => {
  const table = fill([5, 3], (i) => (i % 7) - 3);
  const quantized = quantizeI8(table.data, [5, 3], 0);
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "mul", "add"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y"],
    initializers: {
      w: {},
      g: {},
      c: {},
    },
    values: {
      w: { dtype: "f32", shape: [5, 3] },
      g: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      e: { dtype: "f32", shape: [2, 3] },
      h: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "embedding", ins: ["w", "ids"], outs: ["e"], attrs: { padding_idx: -1 } },
      { op: "mul", ins: ["e", "g"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "c"], outs: ["y"], attrs: {} },
    ],
  };
  return openGraph(graph, [
    weight("c", f32Bytes([1, 2, 3])),
    weight("g", f16Zeros(3), { codec: "f16" }),
    // per-channel（行長 3・group 1 本）— scale のバイト列は v1 の `[5,1]` と同じ。
    weight("w", quantized.bytes, {
      codec: "int8-sym",
      groupSize: 3,
      scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
    }),
  ]);
};

Deno.test({
  name: "estimator の重み・state が実測診断と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const weightModel = gpuWeightModel();
      const estimate = weightModel.estimate();
      const session = await weightModel.createContainerSession(gpu);
      try {
        await session.run({ ids: fill([2], (i) => i, "i32") });
        const storage = session.diagnostics().storage;
        const { weights } = estimate.resident;
        assertEquals(weights.compressedBytes, storage.residentCompressedBytes);
        assertEquals(weights.expandedBytes, storage.hostExpandedBytes);
        // f32 の c は診断に現れない（ADR 0006 の storage 診断は低精度格納だけ）— 手計算 3×4
        assertEquals(weights.uncompressedBytes, 12);
      } finally {
        await session.dispose();
      }

      const model = stateModel();
      const generation = { chunkLength: 4, bindings: { C: 8 } };
      const stateEstimate = model.estimate({ bindings: { T: 2 }, generation });
      const stateSession = await model.createContainerSession(gpu);
      try {
        const context = await stateSession.createGenerationContext(generation);
        assertEquals(
          stateEstimate.resident.stateBytes,
          stateSession.diagnostics().stateBacking.residentBytes,
        );
        await context.dispose();
      } finally {
        await stateSession.dispose();
      }
      // workspaceBytes は突合しない — 融合が中間を消し、行ブロック分割が一時を足すので
      // 実測（planBacking.residentBytes / lastRun.peakTransientBytes）とは原理的にずれる
      // （estimator の unaccounted 欄が認めている差そのもの）。
    } finally {
      gpu.destroy();
    }
  },
});
