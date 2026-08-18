// states 形 attention + `state_append` の**実行統合**門（波 D-3 — ADR 0066 / 0067）。
//
// カーネル単体の格子は gpu_state_attention_test.ts（直接 dispatch）が持つ。こちらが見るのは
// 「Session.run に GenerationContext を渡した 1 step が、CPU 参照を**鎖にした**オラクルと一致
// し続けるか」— つまり実行結線（レシピ / 束縛 / 論理長の搬送 / 進行）そのもの:
//
//   ① 複数 step の parity（prefill → decode ×n・pastLength の進行・2 step 目以降のキャッシュ）
//   ② sliding の ring 跨ぎ / ③GQA（r=2）と census / ④rewind の可否 / ⑤full 容量の run 前検査
//   ⑥queryLength の 3 拒否 / ⑦計画鍵に context が載らないこと / ⑧stateless 実行の拒否
//   ⑨state を submit した run の失敗が context を poison すること
//
// 波 D-4（ADR 0066 決定 5 の焼き込み単位の分離）で足したのは次の 4 本:
//
//   ⑩ 切替 A/B（再導出ゼロ + backing 再構築ゼロ + 焼き直しは context ごと 1 度 + 取り違えゼロ）
//   ⑪ backing が別 signature に入れ替わったときの復帰（世代識別子で焼き直す）
//   ⑫ 故障注入（context 側の束を取り違えると parity が落ちる = ⑩ が空振りでない証明）
//   ⑬ ①の backed 移行（3 run 目以降は slot backing で走り、移行点で値が変わらない）
//
// MUST: オラクルは**ホスト側でスロットを持ち回る**（`referenceStateAppend` の戻りを次 step の
// `referenceStateAttention` へ食わせる）。1 step だけの突合では「append が書いた行を次の step が
// 過去として読む」という結線そのものが検証されない。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import { stateQkKey } from "../src/kernels/state-attention.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { openModel } from "../src/format/container.ts";
import {
  createSession,
  ROW_BLOCK_SPLIT,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import type { BakedGroups } from "../src/runtime/recipe.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

/**
 * 突合の許容誤差。根拠は gpu_state_attention_test.ts の `STATE_TOLERANCE` と同じ
 * （参照は f64 積算・GPU は f32 逐次累積で、差は `D` と `live` に比例する）。
 *
 * MUST: `rtol = 0`。states 形は**厳密 0 が正解の要素**（空行・述語外の寄与）を正規に含むので、
 * 相対項は 0 近傍で効かず、大きい要素の誤りを隠す側にだけ働く。
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/** 決定的な入力列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** 1 モデルぶんの形（`B = 1` 固定 — batch 軸は ADR 0066 決定 8 のスコープ外）。 */
type StateModel = {
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /** スロットの行容量。記号 `"C"` にすると `createGenerationContext(bindings)` が決める。 */
  readonly capacity: number | "C";
  /** `W`（省略 = 全 context）。 */
  readonly window?: number;
};

/**
 * states 形 1 層ぶんのグラフ（attention 1 本 + `state_append` 2 本 — ADR 0067 決定 5b の
 * 発行規約どおり「全読者 → 書き」の順）。
 *
 * `M`（物理 chunk 行数）は**記号**にしてある。prefill（M=4）と decode（M=1）は宣言 shape が
 * 違う別の計画で、1 つの Session / 1 つの context がその 2 本を跨ぐのが ADR 0066 決定 4 の
 * 実行形そのもの。
 */
const stateGraph = (model: StateModel): GraphJson => {
  const { heads, kvHeads, depth, capacity, window } = model;
  const windowAttrs = window === undefined ? {} : { window };
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
    symbols: capacity === "C" ? ["M", "C"] : ["M"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", depth] },
      { name: "k", dtype: "f32", shape: [1, kvHeads, "M", depth] },
      { name: "v", dtype: "f32", shape: [1, kvHeads, "M", depth] },
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, heads, "M", depth] } },
    states: {
      kslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
      vslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: halfScale(depth), ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      },
      append("k", "kslot"),
      append("v", "vslot"),
    ],
  };
};

const stateSession = (
  gpu: GpuContext,
  model: StateModel,
  options: SessionOptions = {},
): Promise<Session> => createSession(gpu, openModel(graphModelBuffer(stateGraph(model))), options);

/** 1 step ぶんの入力（`M` 行ぶん — 有効なのは先頭 `queryLength` 行）。 */
type StepInputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
};

/**
 * step ごとの入力。`salt` を step で変えることで「前 step の入力を読み直している」実装が
 * 値で落ちるようにする（同じ入力を配ると鎖の誤りが自己相殺しうる）。
 */
const stepInputs = (model: StateModel, chunkRows: number, salt: number): StepInputs => ({
  q: seeded(model.heads * chunkRows * model.depth, (i) => QUERY(i + salt)),
  k: seeded(model.kvHeads * chunkRows * model.depth, (i) => KEY(i + salt)),
  v: seeded(model.kvHeads * chunkRows * model.depth, (i) => VALUE(i + salt)),
});

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** ホスト側のスロット内容と論理長（GPU 側 context の鏡）。 */
type Oracle = {
  past: number;
  slotK: Float32Array<ArrayBuffer>;
  slotV: Float32Array<ArrayBuffer>;
};

/** 生成直後の context は**ゼロ初期化されたスロット**を持つ（WebGPU のバッファ契約）。 */
const newOracle = (model: StateModel, capacity: number): Oracle => ({
  past: 0,
  slotK: new Float32Array(model.kvHeads * capacity * model.depth),
  slotV: new Float32Array(model.kvHeads * capacity * model.depth),
});

const asF32 = (data: Float32Array | Int32Array | Uint32Array): Float32Array<ArrayBuffer> => {
  if (!(data instanceof Float32Array)) throw new Error("参照の戻りが f32 でない");
  return data as Float32Array<ArrayBuffer>;
};

/**
 * オラクルを 1 step 進める（**読んでから書く** — ノード配列順そのもの）。返すのは今 step の
 * 期待出力 `[1,H,M,D]`。
 */
const advanceOracle = (
  model: StateModel,
  capacity: number,
  state: Oracle,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
): Float32Array<ArrayBuffer> => {
  const common = {
    chunkRows,
    depth: model.depth,
    capacity,
    window: model.window ?? 0,
    past: state.past,
    query,
  };
  const expected = referenceStateAttention({
    ...common,
    batch: 1,
    heads: model.heads,
    kvHeads: model.kvHeads,
    q: inputs.q,
    insK: inputs.k,
    insV: inputs.v,
    slotK: state.slotK,
    slotV: state.slotV,
    scale: halfScale(model.depth),
  });
  const appendBase = { ...common, kvPlanes: model.kvHeads };
  state.slotK = asF32(
    referenceStateAppend({ ...appendBase, x: inputs.k, slot: state.slotK }).data,
  );
  state.slotV = asF32(
    referenceStateAppend({ ...appendBase, x: inputs.v, slot: state.slotV }).data,
  );
  state.past += query;
  return asF32(expected.data);
};

/** GPU 側の 1 step（`run` の第 3 引数が generation 面）。 */
const runStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
  bindings: Record<string, number> = {},
): Promise<Float32Array<ArrayBuffer>> => {
  const outputs = await session.run(
    {
      q: tensor([1, model.heads, chunkRows, model.depth], inputs.q),
      k: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.k),
      v: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.v),
    },
    bindings,
    { context, queryLength: query },
  );
  const out = outputs["o"];
  assertEquals(out.dtype, "f32");
  return out.data as Float32Array<ArrayBuffer>;
};

/** 1 step 走らせてオラクルと突き合わせる（両側を 1 本にまとめた検査点）。 */
const assertStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  capacity: number,
  state: Oracle,
  chunkRows: number,
  query: number,
  salt: number,
  label: string,
  bindings: Record<string, number> = {},
): Promise<void> => {
  const inputs = stepInputs(model, chunkRows, salt);
  const actual = await runStep(session, context, model, inputs, chunkRows, query, bindings);
  const expected = advanceOracle(model, capacity, state, inputs, chunkRows, query);
  // MUST: 期待値が自明でないことを毎回見る（両側が全 0 なら突合は恒真になる — states 形は
  // 空行の厳密 0 が正規に出るので、この確認が無いと「何も計算していない」実装が緑で通る）。
  assert(expected.some((value) => Math.abs(value) > 1e-3), `${label}: 期待出力が自明（全 ~0）`);
  const report = compareTensors(
    { dtype: "f32", data: actual },
    { dtype: "f32", data: expected },
    STATE_TOLERANCE,
  );
  assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
  assertEquals(context.pastLength, state.past, `${label}: pastLength がオラクルとずれた`);
};

const FULL: StateModel = { heads: 4, kvHeads: 4, depth: 4, capacity: 8 };

Deno.test({
  name: "prefill → decode の鎖が CPU 参照と一致し、2 step 目以降はレシピがキャッシュに当たる",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // prefill（M=4 / Q=4）。1 本目は導出が走る。
      await assertStep(session, context, FULL, 8, state, 4, 4, 0, "prefill");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "prefill は導出 run");
      assertEquals(context.pastLength, 4);

      // decode（M=1 / Q=1）。M が変わるので 1 本目は別の計画 = ミス。
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "decode 1");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "M が変われば別鍵");
      // 2 本目の decode は**同じ鍵**に当たる（ADR 0066 受入条件③の「レシピ再導出ゼロ」）。
      await assertStep(session, context, FULL, 8, state, 1, 1, 23, "decode 2");
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "同一 context・同一 M の 2 本目がキャッシュに当たっていない",
      );
      await assertStep(session, context, FULL, 8, state, 1, 1, 37, "decode 3");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(context.pastLength, 7, "論理長は 4 → 5 → 6 → 7 と run の成功でだけ進む");

      // 波 D-4: generation run も slot backing に載る（ADR 0066 決定 5 の分離焼き込み）。載るのは
      // **Session 所有の実体を束ねる dispatch だけ**で、state を束ねる位置は context 側が焼く。
      // decode 2 / 3 はその backed 経路で走っており、**アリーナ → backed の移行点で値が変わらない**
      // ことは上の各 step の突合そのものが押さえている。
      const diagnostics = session.diagnostics();
      assertEquals(
        diagnostics.planBacking.buildCount,
        1,
        "同一鍵の連続 decode で backing を作り直している（切替スラッシング）",
      );
      assert(
        diagnostics.planBacking.residentBytes > 0,
        "generation run が backed 経路に載っていない（slot が常駐していない）",
      );
      assertEquals(
        diagnostics.stateBacking.rebindCount,
        1,
        "同一 (context, backing) の連続 run で context 側 bind group を焼き直している",
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "pad 行（queryLength < M）は KV に書かれず、次 step の過去にも現れない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // M=4 の chunk で Q=2 だけ有効。pad 2 行を書く実装は、次 step の past が 2 行ぶん
      // ずれる（オラクルは Q 行しか書かない）ので下の突合が落ちる。
      await assertStep(session, context, FULL, 8, state, 4, 2, 5, "pad つき prefill");
      assertEquals(context.pastLength, 2);
      await assertStep(session, context, FULL, 8, state, 4, 3, 41, "pad つき 2 本目");
      assertEquals(context.pastLength, 5);
      await assertStep(session, context, FULL, 8, state, 1, 1, 59, "decode");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

const SLIDING: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 };

Deno.test({
  name: "sliding は ring の wrap を跨いでも CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SLIDING);
    const context = await session.createGenerationContext({ chunkLength: 3 });
    const state = newOracle(SLIDING, 4);
    try {
      // prefill 3 行 → decode ×5。容量 4 の ring は past=4 で一周し、以後は毎 step 上書き。
      await assertStep(session, context, SLIDING, 4, state, 3, 3, 3, "prefill");
      for (let step = 0; step < 5; step += 1) {
        await assertStep(
          session,
          context,
          SLIDING,
          4,
          state,
          1,
          1,
          71 + step * 13,
          `decode ${step} (past=${state.past})`,
        );
      }
      assertEquals(context.pastLength, 8, "容量 4 の ring でも論理長は進み続ける");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

const GQA: StateModel = { heads: 4, kvHeads: 2, depth: 4, capacity: 8 };

Deno.test({
  name: "GQA（r=2）の states 形が CPU 参照と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, GQA);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(GQA, 8);
    try {
      await assertStep(session, context, GQA, 8, state, 4, 4, 2, "prefill");
      await assertStep(session, context, GQA, 8, state, 1, 1, 17, "decode 1");
      await assertStep(session, context, GQA, 8, state, 1, 1, 29, "decode 2");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * **census**（ADR 0058 決定 4）— GQA 変種のキーが**実際に走った**ことを見る。
 *
 * MUST: 計測を要求しない device では明示 SKIP し、走るときは空の内訳を無条件に FAIL にする
 * （`entries` が空なら素通り、にすると全ケースが無検査のまま緑になる）。
 */
Deno.test({
  name:
    "states 形 attention の dispatch は変種キーどおりに走る（census・実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [label, model, sliding] of [
          ["full r=1", FULL, false],
          ["full r=2", GQA, false],
          ["sliding r=1", SLIDING, true],
        ] as const
      ) {
        const session = await stateSession(gpu, model);
        const context = await session.createGenerationContext({ chunkLength: 2 });
        try {
          await runStep(session, context, model, stepInputs(model, 2, 7), 2, 2);
          const keys = session.diagnostics().lastRunTiming?.entries.map((entry) => entry.key) ?? [];
          assert(keys.length > 0, `${label}: 内訳が空（キー検査が空振りしている）`);
          assertEquals(
            keys.includes(stateQkKey(sliding, model.heads !== model.kvHeads)),
            true,
            `${label}: 期待した ①QK の変種キーが出ていない（${keys.join(" / ")}）`,
          );
          // 別族であることの裏 — 融合 attention のキーは 1 本も出ない。
          assertEquals(
            keys.filter((key) => key.startsWith("attention_qk")),
            [],
            `${label}: 融合 attention のキーが混ざっている`,
          );
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "rewind は full のみの context で効き、sliding を含む context では全拒否（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const full = await stateSession(gpu, FULL);
    const sliding = await stateSession(gpu, SLIDING);
    const context = await full.createGenerationContext({ chunkLength: 4 });
    const slidingContext = await sliding.createGenerationContext({ chunkLength: 2 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(full, context, FULL, 8, state, 4, 4, 0, "prefill");
      await assertStep(full, context, FULL, 8, state, 1, 1, 101, "decode（捨てる）");
      assertEquals(context.pastLength, 5);

      // 巻き戻し: 論理位置 4 へ戻し、別の入力で同じ位置を上書きする。
      context.rewind(4);
      state.past = 4;
      assertEquals(context.pastLength, 4);
      await assertStep(full, context, FULL, 8, state, 1, 1, 211, "rewind 後の decode");
      assertEquals(context.pastLength, 5, "巻き戻した位置から進み直す");

      // sliding を 1 本でも含む context は位置指定 rewind を全拒否（ADR 0066 追記 2）。
      const rejected = assertThrows(() => slidingContext.rewind(0), ExecutionError);
      assert(rejected.message.includes("sliding"), rejected.message);
      assert(rejected.message.includes("kslot"), rejected.message);
    } finally {
      await context.dispose();
      await slidingContext.dispose();
      await full.dispose();
      await sliding.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 行ブロックの**強制分割 parity**（ADR 0067 決定 7 / ADR 0060 と同じ流儀）。
 *
 * 既定の枚数は device の `maxStorageBufferBindingSize` から静的に決まるので、上限に余裕のある
 * 機では常に 1 枚 = 複数ブロックの経路（ブロック跨ぎの row_offset・ブロックごとの一時）が
 * 1 度も走らない。`ROW_BLOCK_SPLIT` はその経路を実機で回すための唯一の手段で、値は
 * **ビット単位で同一**でなければならない（畳んでいるのは中間の実体化幅だけ）。
 */
Deno.test({
  name: "行ブロックを強制分割しても出力はビット同一（states 形・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const single = await stateSession(gpu, FULL);
    const split = await stateSession(gpu, FULL, { [ROW_BLOCK_SPLIT]: 2 });
    const inputs = stepInputs(FULL, 4, 97);
    try {
      const contexts = await Promise.all([
        single.createGenerationContext({ chunkLength: 4 }),
        split.createGenerationContext({ chunkLength: 4 }),
      ]);
      try {
        const one = await runStep(single, contexts[0], FULL, inputs, 4, 4);
        const many = await runStep(split, contexts[1], FULL, inputs, 4, 4);
        assertEquals(
          Array.from(new Uint32Array(many.buffer, many.byteOffset, many.length)),
          Array.from(new Uint32Array(one.buffer, one.byteOffset, one.length)),
          "2 枚に割った実行が 1 枚実行とビット単位で一致しない",
        );
      } finally {
        for (const context of contexts) await context.dispose();
      }
    } finally {
      await single.dispose();
      await split.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 数値変種（`attentionCompute` / `attentionScoreStorage`）× states 形は **fail loudly**
 * （ADR 0058 決定 3 —「未実装の組は縮退でなく fail loudly」）。
 *
 * MUST: 黙って f32 経路で走らせない。opt-in を指定したのに効かない状態は、値も診断も変わらない
 * まま「速くなっていない」だけで残る。
 */
Deno.test({
  name: "states 形は f32 以外の数値変種と組めない（縮退でなく fail loudly・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cases: readonly (readonly [string, SessionOptions, string])[] = [
      ["attentionCompute i8a8", { attentionCompute: "i8a8" }, "attentionCompute"],
      ["attentionScoreStorage f16", { attentionScoreStorage: "f16" }, "attentionScoreStorage"],
    ];
    try {
      for (const [label, options, needle] of cases) {
        const session = await stateSession(gpu, FULL, options);
        const context = await session.createGenerationContext({ chunkLength: 1 });
        try {
          const error = await assertRejects(
            () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 1),
            ExecutionError,
          );
          assert(error.message.includes(needle), `${label}: ${error.message}`);
          assertEquals(context.pastLength, 0, `${label}: 拒否された run は進めない`);
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

const SMALL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 6 };

Deno.test({
  name: "full スロットの容量超過はエンコード前に落ち、context を汚染しない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SMALL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(SMALL, 6);
    try {
      await assertStep(session, context, SMALL, 6, state, 4, 4, 0, "prefill");
      assertEquals(context.pastLength, 4);

      // 4 + 4 > 6。dispatch を 1 本も積まずに落ちる。
      const inputs = stepInputs(SMALL, 4, 13);
      const error = await assertRejects(
        () => runStep(session, context, SMALL, inputs, 4, 4),
        ExecutionError,
      );
      assert(error.message.includes("full スロット"), error.message);
      // MUST: 汚染していない（poison していれば pastLength の読みが落ちる）。
      assertEquals(context.pastLength, 4, "拒否された run は論理長も汚染も動かさない");

      // 容量に収まる次の run は通る（拒否が context を壊していないことの裏）。
      await assertStep(session, context, SMALL, 6, state, 4, 2, 31, "収まる run");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "queryLength は M / chunkLength / 正整数の 3 つで拒否される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    // chunkLength=2 に対し M=4 の chunk を渡せるので、2 つの上限を別々に踏める。
    const context = await session.createGenerationContext({ chunkLength: 2 });
    try {
      // ① Q > M（宣言 shape の外 — 導出相が集めた chunkRows が見る）
      const overRows = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 2),
        ExecutionError,
      );
      assert(overRows.message.includes("物理 chunk 行数"), overRows.message);

      // ② Q > chunkLength（context の計画時定数 — writeLengths が見る）
      const overChunk = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 4, 0), 4, 3),
        ExecutionError,
      );
      assert(overChunk.message.includes("chunkLength"), overChunk.message);

      // ③ Q = 0（「何も進めない run」— 進行と物理内容の対応が観測できなくなる）
      const zero = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 4, 0), 4, 0),
        ExecutionError,
      );
      assert(zero.message.includes("queryLength"), zero.message);

      assertEquals(context.pastLength, 0, "拒否された run は 1 つも進めない");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

const SYMBOLIC: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "C" };

Deno.test({
  name: "計画鍵は容量で決まり、context の識別子では決まらない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SYMBOLIC);
    const first = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 8 } });
    const second = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 8 } });
    const wider = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 12 } });
    const inputs = stepInputs(SYMBOLIC, 1, 0);
    try {
      // MUST: 容量記号 `C` は run の bindings に**要らない**（束縛点は createGenerationContext
      // だけ — ADR 0066 追記 7 の効く範囲の分担。渡させると context との二重簿記になる）。
      await runStep(session, first, SYMBOLIC, inputs, 1, 1);
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "1 本目は導出 run");

      // MUST: **別 context・同じ容量**は同じ鍵（context の識別子が鍵に載っていればここで
      // 再導出が起き、decode のホットパスが毎シーケンス全滅する — ADR 0066 決定 5）。
      await runStep(session, second, SYMBOLIC, inputs, 1, 1);
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "同容量の別 context が別鍵になっている",
      );
      // ヒット run なので slot backing が立つ（generation run も backed — 波 D-4）。
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      assertEquals(session.diagnostics().stateBacking.rebindCount, 1, "second が 1 度焼く");

      // 容量が違えば別鍵（レシピは容量を params と S の確保サイズへ焼き込む）。
      await runStep(session, wider, SYMBOLIC, inputs, 1, 1);
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "容量の違う context が同じレシピを使い回している",
      );
      // MUST: ミス run は backing を作らない（単発 run に slot メモリを払わせない門）ので、
      // 容量の違う context を挟んでも活性 backing は据え置き = 焼き直しも起きない。
      assertEquals(session.diagnostics().planBacking.buildCount, 1, "ミス run が backing を作った");
      assertEquals(session.diagnostics().stateBacking.rebindCount, 1);

      // 戻ると最初の鍵にまた当たる（LRU に両方載っている）。
      await runStep(session, first, SYMBOLIC, inputs, 1, 1);
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      // backing は同じ実体のまま（世代識別子が動かない）で、焼くのは first のぶん 1 度だけ。
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      assertEquals(session.diagnostics().stateBacking.rebindCount, 2, "first が 1 度焼く");

      // states 専用記号を run の bindings に書いた形は fail loudly（黙って受けて鍵だけ割れる
      // 形にしない — 同じ計画が別鍵で重複導出される沈黙劣化）。
      const rejected = await assertRejects(
        () => runStep(session, first, SYMBOLIC, inputs, 1, 1, { C: 8 }),
        ExecutionError,
      );
      assert(rejected.message.includes("states 専用記号"), rejected.message);
    } finally {
      for (const context of [first, second, wider]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "state 参照グラフを generation 無しで run すると fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    try {
      const inputs = stepInputs(FULL, 1, 0);
      // 落ちるのは shape 計算層（スロットの解決済み shape が無い run はそこまで進めない）。
      const error = await assertRejects(
        () =>
          session.run({
            q: tensor([1, FULL.heads, 1, FULL.depth], inputs.q),
            k: tensor([1, FULL.kvHeads, 1, FULL.depth], inputs.k),
            v: tensor([1, FULL.kvHeads, 1, FULL.depth], inputs.v),
          }),
        OpContractError,
      );
      assert(error.message.includes("GenerationContext"), error.message);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 内部面（論理長の進行を直に駆動する / 焼き込み束を覗く）。**故障注入と前提づくり専用**で、
 * 通常の結線は全て `Session.run` 越しに見る。
 */
const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/**
 * poison の結線（ADR 0066 追記 3）。
 *
 * 決定的な失敗注入として、**論理長を u32 上限の直前まで進めてから** state を書く run を 1 本
 * 通す。dispatch は全て成功して submit されるが、成功後の `advance` が u32 の上限で落ちるので、
 * 「state を submit した run が例外で終わる」形になる — スナップショット比較が働けば context は
 * poison され、働かなければ「物理 ring だけ進んだ context」が正常値を返し続ける。
 *
 * NOTE: これが踏むのは判定の結線であって、GPU 側の失敗（device 消失・validation）そのものでは
 * ない。そちらの注入面は波 C の保留（L8 fake device）待ち。
 */
Deno.test({
  name: "state を submit した run の失敗は context を poison する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // sliding にするのは full 容量の run 前検査（past + query ≤ C）を避けるため — ring は
    // 論理長がいくつでも回る。
    const session = await stateSession(gpu, SLIDING);
    const context = await session.createGenerationContext({ chunkLength: 0xffffffff });
    try {
      internals(context).advance(0xffffffff - 1);
      assertEquals(context.pastLength, 0xfffffffe);

      const inputs = stepInputs(SLIDING, 2, 0);
      const error = await assertRejects(
        () => runStep(session, context, SLIDING, inputs, 2, 2),
        ExecutionError,
      );
      assert(error.message.includes("u32 の上限"), error.message);

      // MUST: 以後の全操作が拒否される（読みも含む — 物理 ring は書かれてしまっている）。
      const poisoned = assertThrows(() => context.pastLength, ExecutionError);
      assert(poisoned.message.includes("汚染された"), poisoned.message);
      assert(
        poisoned.message.includes("state 変更 dispatch を submit した run が失敗した"),
        `真因が残っていない: ${poisoned.message}`,
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * **切替 A/B**（ADR 0066 受入条件③の完成形 — 決定 5 の焼き込み単位の分離）。
 *
 * 同じ Session・同じ容量の context 2 本を交互に decode する。分離が効いていれば同時に 3 つが
 * 成り立つ: ①レシピ再導出ゼロ（鍵に context が載らない）②backing 再構築ゼロ（Session 所有の
 * 焼き込みは context に依らない）③stale 読みゼロ（state を束ねる bind group は context ごと）。
 *
 * MUST: 入力の salt を context ごとに変える — KV の取り違えは例外を出さず**値にしか出ない**ので、
 * 同じ入力を配ると鎖の誤りが自己相殺しうる。
 */
Deno.test({
  name: "context を交互に切り替えても再導出・backing 再構築ゼロで KV を取り違えない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const a = await session.createGenerationContext({ chunkLength: 1 });
    const b = await session.createGenerationContext({ chunkLength: 1 });
    const stateA = newOracle(FULL, 8);
    const stateB = newOracle(FULL, 8);
    try {
      // 導出が走るのは M=1 の 1 本目だけ。以後は全て同じ鍵に当たる。
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 0, "A 初回");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "1 本目は導出 run");

      for (let step = 0; step < 3; step += 1) {
        await assertStep(session, b, FULL, 8, stateB, 1, 1, 301 + step * 7, `B decode ${step}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          `B decode ${step}: context 切替で再導出が起きた（鍵に context が載っている）`,
        );
        await assertStep(session, a, FULL, 8, stateA, 1, 1, 401 + step * 7, `A decode ${step}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          `A decode ${step}: context 切替で再導出が起きた`,
        );
      }

      const diagnostics = session.diagnostics();
      assertEquals(
        diagnostics.planBacking.buildCount,
        1,
        "context 切替が slot backing を作り直している（決定 5 が避けた全再構築スラッシング）",
      );
      // 焼き直しは **context ごとに 1 度**きり。束が有効なのは backing 実体に対してなので、
      // 切替では無効にならない（切替のたびに増える形は run 数に比例する再構築の入り口）。
      assertEquals(
        diagnostics.stateBacking.rebindCount,
        2,
        "context 側 bind group の焼き直しが context ごと 1 度で収まっていない",
      );
      assertEquals(diagnostics.stateBacking.contextCount, 2);
      assertEquals(a.pastLength, 4);
      assertEquals(b.pastLength, 3);
    } finally {
      for (const context of [a, b]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * backing が**別 signature に入れ替わった**後の復帰（ADR 0066 決定 5 の世代識別子）。
 *
 * M=4（prefill 形）と M=1（decode 形）は別鍵なので、容量 1 の slot backing を奪い合う。退役した
 * backing の slot / 入力バッファは run の後始末で `destroy()` されるため、context 側が古い束を
 * 掴んだまま回れば**破棄済みバッファを束ねた dispatch**になる（値か例外のどちらかで必ず壊れる）。
 * 焼き直しが backing の再構築に追随していることを、値の正しさと回数の両方で押さえる。
 */
Deno.test({
  name: "backing の入れ替わりに追随して context 側 bind group を焼き直す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // 2 つの鍵をそれぞれ導出させる（ここまでは全てミス run = backing 不使用）。
      await assertStep(session, context, FULL, 8, state, 4, 1, 0, "M=4 初回（導出）");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "M=1 初回（導出）");
      assertEquals(session.diagnostics().planBacking.buildCount, 0, "ミス run が backing を作った");

      /** 各 run 決着時の `[backing 構築回数, 焼き直し回数]`。 */
      const builds: [number, number][] = [];
      for (const [rows, salt] of [[1, 23], [4, 31], [1, 43], [4, 53]] as const) {
        await assertStep(session, context, FULL, 8, state, rows, 1, salt, `M=${rows} へ切替`);
        assertEquals(session.diagnostics().lastRunPrepared?.hit, true, "レシピは再導出しない");
        const diagnostics = session.diagnostics();
        builds.push([diagnostics.planBacking.buildCount, diagnostics.stateBacking.rebindCount]);
      }
      // 鍵が交互に変わるので backing は毎 run 作り直しになり、context 側も毎 run 焼き直す。
      // MUST: 焼き直し回数が backing の構築回数に追随すること — 追随しないなら、退役した
      // backing のバッファを束ねた束が使い回されている。
      assertEquals(
        builds,
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        "backing の再構築に context 側の焼き直しが追随していない",
      );
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * **故障注入** — 切替 A/B（上）が空振りでないことの実証。
 *
 * context B の焼き込み束を **A のもの**で差し替える = state を含む bind group を context 跨ぎで
 * 共有した実装（波 D-3 の DECIDED が名指ししていた「前の context の KV を束ねたまま回る」形）を
 * 再現する。A / B は同容量なのでバッファの大きさは 1 バイトも違わず、**validation は通って値だけが
 * 静かに変わる** — この突合が落ちなければ、切替 A/B の parity は何も守っていない。
 */
Deno.test({
  name: "context 側の束を取り違えると出力が前の context の KV を読む（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const a = await session.createGenerationContext({ chunkLength: 1 });
    const b = await session.createGenerationContext({ chunkLength: 1 });
    const stateA = newOracle(FULL, 8);
    const stateB = newOracle(FULL, 8);
    try {
      // 焼き込みが起きるのはヒット run からなので、A を backed まで温めてから束を横取りする。
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 0, "A 初回（導出）");
      const internalsA = internals(a);
      const setBaked = internalsA.setBakedGroups;
      let stolen: BakedGroups | undefined;
      internalsA.setBakedGroups = (token, groups) => {
        stolen = groups;
        setBaked(token, groups);
      };
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 13, "A backed");
      assert(stolen !== undefined, "A の context 側 bind group が焼かれていない（注入が空振り）");

      // 注入: B は自分の束の代わりに A の束（A の KV スロットと A の論理長 uniform）を使う。
      internals(b).bakedGroups = () => stolen;
      const inputs = stepInputs(FULL, 1, 77);
      const actual = await runStep(session, b, FULL, inputs, 1, 1);
      const expected = advanceOracle(FULL, 8, stateB, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(
        report.pass,
        false,
        "A の束で走った B の run が B のオラクルと一致した（切替 A/B の突合が空振りしている）",
      );
    } finally {
      for (const context of [a, b]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});
