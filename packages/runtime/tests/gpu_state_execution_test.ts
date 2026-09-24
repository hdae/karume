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
//   ⑪ backing が退役して作り直されたときの復帰（世代識別子で焼き直す）— ADR 0095（予算つきの
//      複数保持）で 3 本になった: 予算 0 は切替のたびに作り直して焼き直す / 予算内で保持した形の
//      往復は焼き直しを増やさない / 退役（予算超過）した形は束ごと捨てられ、戻った run で 1 回
//      だけ焼き直す
//   ⑫ 故障注入（context 側の束を取り違えると parity が落ちる = ⑩ が空振りでない証明）
//   ⑬ ①の backed 移行（3 run 目以降は slot backing で走り、移行点で値が変わらない）
//
// MUST: オラクルは**ホスト側でスロットを持ち回る**（`referenceStateAppend` の戻りを次 step の
// `referenceStateAttention` へ食わせる）。1 step だけの突合では「append が書いた行を次の step が
// 過去として読む」という結線そのものが検証されない。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import {
  statePvKey,
  statePvParallelKey,
  statePvTiledKey,
  stateQkKey,
  stateQkParallelKey,
  stateQkTiledKey,
} from "../src/kernels/state-attention.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  ROW_BLOCK_SPLIT,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import type { PlanBackingStats } from "../src/runtime/session-types.ts";
import type { BakedGroups } from "../src/runtime/recipe.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";
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
  /**
   * スロットの行容量。記号 `"C"`（states 専用記号）にすると `createGenerationContext(bindings)`
   * だけが決める。記号 `"M"` は**入力 shape にも現れる記号**で、束縛点が 2 つある形になる
   * （ADR 0066 追記 7 — 2 つの束縛点が割れていないことを実行時に照合する）。
   */
  readonly capacity: number | "C" | "M";
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
const stateGraph = (model: StateModel): DeclarationJson => {
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
    version: 2,
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

const stateSession = async (
  gpu: GpuContext,
  model: StateModel,
  options: SessionOptions = {},
): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(stateGraph(model)),
    GRAPH_NAME,
    options,
  );

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

// ---------------------------------------------------------------------------
// 投機デコードの verify 形（sliding の余裕 + deferred commit）
// ---------------------------------------------------------------------------

/**
 * 余裕の行数（sliding スロットの `capacity − window`）。`Q` 行を論理長より先に書いて `a` 行だけ
 * 受理する形が成立する条件は `C ≥ W + Q`（棄却行 `P+i` が潰す論理列 `P+i−C` が、確定後の読者の
 * 窓の下端より必ず小さい — 下端が最も低いのは借り手 = readonly 読者の `P+a−W`）。したがって
 * 余裕 8 が支える `Q` の上限は 8。
 */
const VERIFY_SLACK = 8;
const VERIFY_WINDOW = 8;
const VERIFY_QUERY = VERIFY_SLACK;
/** 受理行数（残り 5 行は棄却 = 物理 ring に書かれたまま論理長に載らない）。 */
const VERIFY_COMMIT = 3;
/** verify 形の前に積む past（`P > W` — 窓の下限述語が効いている状態から始める）。 */
const VERIFY_PREFILL = 12;

const verifyModel = (capacity: number): StateModel => ({
  heads: 2,
  kvHeads: 2,
  depth: 4,
  capacity,
  window: VERIFY_WINDOW,
});

/** `[1, planes, M, D]` の 1 行を `[1, planes, 1, D]` として切り出す。 */
const takeRow = (
  data: Float32Array<ArrayBuffer>,
  planes: number,
  chunkRows: number,
  depth: number,
  row: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let d = 0; d < depth; d += 1) {
      out[plane * depth + d] = data[(plane * chunkRows + row) * depth + d];
    }
  }
  return out;
};

const bitsOfF32 = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

/**
 * 「`Q` 行を deferred で書いて `a` 行だけ commit した context」の次 step の出力を返す。
 *
 * 参照側（`speculate = false`）は同じ token 列を **1 行ずつ通常 run** で流す（= 投機を使わない
 * 生成そのもの）。両者がビット同一なら、棄却された 5 行の書き込みが live な過去 KV を 1 語も
 * 壊していないことになる。
 */
const runVerifyChain = async (
  gpu: GpuContext,
  capacity: number,
  speculate: boolean,
): Promise<Float32Array<ArrayBuffer>> => {
  const model = verifyModel(capacity);
  const session = await stateSession(gpu, model);
  const context = await session.createGenerationContext({
    chunkLength: 16,
    chunkBuckets: [VERIFY_QUERY],
  });
  try {
    // ① 共通の prefill（P = 12 > W = 8）
    const prefill = stepInputs(model, 16, 5);
    await runStep(session, context, model, prefill, 16, VERIFY_PREFILL);

    // ② draft 9 行ぶんの入力（両側で**同じ値**を使う — 参照側は先頭 3 行を 1 行ずつ流す）
    const draft = stepInputs(model, VERIFY_QUERY, 41);
    if (speculate) {
      const outputs = await session.run(
        {
          q: tensor([1, model.heads, VERIFY_QUERY, model.depth], draft.q),
          k: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.k),
          v: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.v),
        },
        {},
        { context, queryLength: VERIFY_QUERY, commit: "deferred" },
      );
      assertEquals(outputs["o"].shape, [1, model.heads, VERIFY_QUERY, model.depth]);
      assertEquals(
        context.pendingCommit,
        { pastLength: VERIFY_PREFILL, queryLength: VERIFY_QUERY },
        "deferred run が保留を作っていない",
      );
      context.commit(VERIFY_COMMIT);
    } else {
      for (let row = 0; row < VERIFY_COMMIT; row += 1) {
        await runStep(
          session,
          context,
          model,
          {
            q: takeRow(draft.q, model.heads, VERIFY_QUERY, model.depth, row),
            k: takeRow(draft.k, model.kvHeads, VERIFY_QUERY, model.depth, row),
            v: takeRow(draft.v, model.kvHeads, VERIFY_QUERY, model.depth, row),
          },
          1,
          1,
        );
      }
    }
    assertEquals(context.pastLength, VERIFY_PREFILL + VERIFY_COMMIT, "論理長が受理行数と合わない");

    // ③ 受理後の 1 step（ここで読む過去 KV に棄却行が混ざっていないか）
    const next = stepInputs(model, 1, 97);
    return await runStep(session, context, model, next, 1, 1);
  } finally {
    await context.dispose();
    await session.dispose();
  }
};

Deno.test({
  name: "deferred commit + sliding の余裕: 棄却行は次 step の出力を 1 語も動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const slack = VERIFY_WINDOW + VERIFY_SLACK;
      const speculative = await runVerifyChain(gpu, slack, true);
      const sequential = await runVerifyChain(gpu, slack, false);
      assertEquals(
        [...bitsOfF32(speculative)],
        [...bitsOfF32(sequential)],
        "余裕つきの投機 + commit が 1 行ずつの生成とビット同一でない",
      );

      // 余裕を 0（capacity = window）に戻すと、棄却行が live 窓の中の論理列を潰す（例外も NaN も
      // 出ない沈黙破壊）。その形は run 発行の同期区間で落ちる — deferred run の queryLength は
      // 「余裕」まで（`GenerationContext.slidingSlack` の門）。潰れる値そのものは公開面から
      // 到達できないので、ここで見るのは「守りが効いていること」= 発行が拒否されること。
      const noSlack = await assertRejects(
        () => runVerifyChain(gpu, VERIFY_WINDOW, true),
        ExecutionError,
      );
      assert(noSlack.message.includes("余裕 0"), noSlack.message);
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 借り手（drafter 相当）: 貸し手と**同名**の external スロットを readonly attention 1 本が
 * 読むだけのグラフ（ADR 0096 段 2 §2.1）。書き手（`state_append`）を持たないので、論理長を
 * 進めるのも確定させるのも貸し手の側。
 */
const verifyBorrowerGraph = (capacity: number): DeclarationJson => {
  const model = verifyModel(capacity);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention"] },
    symbols: [],
    inputs: [{ name: "q", dtype: "f32", shape: [1, model.heads, 1, model.depth] }],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, model.heads, 1, model.depth] } },
    states: {
      kslot: { dtype: "f32", shape: [1, model.kvHeads, capacity, model.depth], external: true },
      vslot: { dtype: "f32", shape: [1, model.kvHeads, capacity, model.depth], external: true },
    },
    nodes: [{
      op: "attention",
      ins: ["q"],
      outs: ["o"],
      attrs: { scale: halfScale(model.depth), window: VERIFY_WINDOW, readonly: true },
      states: { k: "kslot", v: "vslot" },
    }],
  };
};

/**
 * 「`Q = 余裕` 行を deferred で書いて `m` 行だけ commit した貸し手」を、**借り手**（readonly
 * 読者）が 1 回読んだ出力を返す。参照側（`speculate = false`）は同じ token 列の先頭 `m` 行を
 * 1 行ずつ通常 run で流す（= 棄却行を 1 語も書かなかった走行）。
 *
 * `runVerifyChain` との違いは読者の種類だけ。states 形の読者は今 step の ins のぶん窓の下端が
 * 1 列高い（`P+m−(W−1)`）のに対し、借り手の readonly 読者は `P+m−W` まで下がる（
 * `src/kernels/state-attention.ts` の readonly 節 `column_base = P − min(P, W)`）。門を
 * `Q ≤ slack + 1` から `Q ≤ slack` へ締めた根拠はこの 1 列で、それを踏むのはこちらの形だけ。
 */
const runBorrowedVerifyChain = async (
  gpu: GpuContext,
  speculate: boolean,
  commitRows: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const capacity = VERIFY_WINDOW + VERIFY_SLACK;
  const model = verifyModel(capacity);
  const lender = await stateSession(gpu, model);
  const borrower = await createSessionFromContainer(
    gpu,
    await openGraphModel(verifyBorrowerGraph(capacity)),
    GRAPH_NAME,
  );
  const lenderContext = await lender.createGenerationContext({
    chunkLength: 16,
    chunkBuckets: [VERIFY_QUERY],
  });
  const borrowed = await borrower.createGenerationContext({
    chunkLength: 1,
    borrow: lenderContext,
  });
  try {
    // ① 共通の prefill（P = 12 > W = 8）
    await runStep(lender, lenderContext, model, stepInputs(model, 16, 5), 16, VERIFY_PREFILL);

    // ② draft 8 行ぶんの入力（両側で**同じ値**を使う — 参照側は先頭 m 行を 1 行ずつ流す）
    const draft = stepInputs(model, VERIFY_QUERY, 41);
    if (speculate) {
      await lender.run(
        {
          q: tensor([1, model.heads, VERIFY_QUERY, model.depth], draft.q),
          k: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.k),
          v: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.v),
        },
        {},
        { context: lenderContext, queryLength: VERIFY_QUERY, commit: "deferred" },
      );
      lenderContext.commit(commitRows);
    } else {
      for (let row = 0; row < commitRows; row += 1) {
        await runStep(
          lender,
          lenderContext,
          model,
          {
            q: takeRow(draft.q, model.heads, VERIFY_QUERY, model.depth, row),
            k: takeRow(draft.k, model.kvHeads, VERIFY_QUERY, model.depth, row),
            v: takeRow(draft.v, model.kvHeads, VERIFY_QUERY, model.depth, row),
          },
          1,
          1,
        );
      }
    }
    assertEquals(
      lenderContext.pastLength,
      VERIFY_PREFILL + commitRows,
      "論理長が受理行数と合わない",
    );

    // ③ 借り手の 1 run（`[P−W, P)` を読む — 棄却行が混ざっていないか）
    const outputs = await borrower.run(
      {
        q: tensor(
          [1, model.heads, 1, model.depth],
          seeded(model.heads * model.depth, (i) => QUERY(i + 97)),
        ),
      },
      {},
      { context: borrowed, queryLength: 1 },
    );
    assertEquals(
      borrowed.pastLength,
      VERIFY_PREFILL + commitRows,
      "借り手が写した P が貸し手とずれた",
    );
    return outputs["o"].data as Float32Array<ArrayBuffer>;
  } finally {
    await borrowed.dispose();
    await lenderContext.dispose();
    await borrower.dispose();
    await lender.dispose();
  }
};

Deno.test({
  name:
    "deferred commit + 借り手の readonly 窓: 棄却行は drafter が読む列を 1 語も動かさない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // `m = 0`（全棄却 = 例外復旧の `commit(0)`）が最悪ケース。C = 16 / W = 8 / P = 12 で
      // 棄却行 j = 0..7 が潰す物理 row は 12,13,14,15,0,1,2,3、借り手が読む列 `[4,12)` の
      // 物理 row は 4..11 で、**ちょうど 1 行だけ空いている**（`Q` を 9 に緩めると 9 本目の
      // 論理列 20 が物理 row 4 = 借り手の窓の下端を潰す）。この形が緑であることが、締めた門
      // `Q ≤ slidingSlack` の安全性そのもの。`m = VERIFY_COMMIT` は受理行が混じる側の対照。
      for (const commitRows of [0, VERIFY_COMMIT]) {
        const speculative = await runBorrowedVerifyChain(gpu, true, commitRows);
        const sequential = await runBorrowedVerifyChain(gpu, false, commitRows);
        // MUST: 期待値が自明でない（両側が全 0 なら突合は恒真）。
        assert(
          speculative.some((value) => Math.abs(value) > 1e-3),
          `commit(${commitRows}): 借り手の出力が自明（全 ~0）`,
        );
        assertEquals(
          [...bitsOfF32(speculative)],
          [...bitsOfF32(sequential)],
          `commit(${commitRows}): 棄却行が借り手の読む列を壊した`,
        );
      }
    } finally {
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

/**
 * **census**（ADR 0058 決定 4 ③）— `stateAttentionReduce: "parallel"` を指定したとき、①' と ③'
 * のキー（`:par`）が**期待した計画で実際に走り**、対の逐次キーが 1 本も出ないことを見る。逆に
 * 既定では ① ③ だけが出る（opt-in が黙って既定へ落ちる / 既定が黙って変種へ上がる、の両方向を
 * 塞ぐ）。
 *
 * MUST: 2 段を**両方**見る（席は 1 つで 2 段を一緒に切り替えるので、片方だけの検査だと
 * 「①' が結線から落ちて ① が走っている」が素通りする — 値は帯の内側なので数値門も鳴らない）。
 * MUST: **①' だけは席に適用条件が掛かる**（`M ≤ 8` = decode と投機の verify）。この表は M=1 / M=2 で
 * 「①' + ③'」を見て、適用外（M=9 → ① + ③'）は下の 3 経路の census が持つ（この模型の full スロットは
 * 容量 8 で M=9 を載せられない）。③' は M に依らない（適用条件と実測は src/kernels/state-attention.ts の
 * `stateQkParallelEligible`）。門を条件式から独立させるため、期待は行ごとに `qkParallel` の
 * 真偽で直書きする（判定を輸入すると実装と一緒に間違える）。
 */
Deno.test({
  name:
    "states 形 attention ①QK / ③PV の縮約形は席と計画の M どおりに走る（census・実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [label, model, sliding, reduce, chunkRows, qkParallel, pvParallel] of [
          ["full r=2 parallel decode M=1", GQA, false, "parallel", 1, true, true],
          // M=2（投機の verify 相当・M ≤ 8）は ①' も ③' も上がる
          ["full r=2 parallel verify M=2", GQA, false, "parallel", 2, true, true],
          ["sliding r=1 parallel decode M=1", SLIDING, true, "parallel", 1, true, true],
          ["sliding r=1 parallel verify M=2", SLIDING, true, "parallel", 2, true, true],
          // 席が既定なら M=1 でも上がらない（適用条件だけを見て席を無視する実装を落とす）
          ["full r=2 sequential decode M=1", GQA, false, "sequential", 1, false, false],
          ["full r=2 sequential prefill M=2", GQA, false, "sequential", 2, false, false],
        ] as const
      ) {
        const session = await stateSession(gpu, model, { stateAttentionReduce: reduce });
        // chunkLength は行の M そのもの（M=1 の decode 形も M=9 の prefill 形も同じ context 契約の内側）。
        const context = await session.createGenerationContext({
          chunkLength: Math.max(chunkRows, 2),
        });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 7),
            chunkRows,
            chunkRows,
          );
          const keys = session.diagnostics().lastRunTiming?.entries.map((entry) => entry.key) ?? [];
          assert(keys.length > 0, `${label}: 内訳が空（キー検査が空振りしている）`);
          const gqa = model.heads !== model.kvHeads;
          for (
            const [stage, expected, other] of [
              [
                "①QK",
                qkParallel ? stateQkParallelKey(sliding, gqa) : stateQkKey(sliding, gqa),
                qkParallel ? stateQkKey(sliding, gqa) : stateQkParallelKey(sliding, gqa),
              ],
              [
                "③PV",
                pvParallel ? statePvParallelKey(sliding, gqa) : statePvKey(sliding, gqa),
                pvParallel ? statePvKey(sliding, gqa) : statePvParallelKey(sliding, gqa),
              ],
            ] as const
          ) {
            assertEquals(
              keys.includes(expected),
              true,
              `${label}: 期待した ${stage} のキーが出ていない（${keys.join(" / ")}）`,
            );
            assertEquals(
              keys.includes(other),
              false,
              `${label}: 席と計画から期待されない ${stage} のキーが混ざっている（${
                keys.join(" / ")
              }）`,
            );
          }
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

/** ①ₜ / ③ₜ（`M ≥ 16` で選ぶ行タイル共有経路）を踏ませるための形（`M = 16` が入る容量）。 */
const TILED: StateModel = { heads: 4, kvHeads: 2, depth: 4, capacity: 64 };

const TILED_SLIDING: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 16, window: 16 };

/**
 * **census**（ADR 0058 決定 4 ③ のタイル経路版 — perf-ledger K-13）— ①QK と ③PV の
 * **3 経路ずつ**が計画の `M` と席どおりに走ることを見る。
 *
 * ①ₜ / ③ₜ（K / V の行タイル共有）は **① / ③ とビット同一**なので**席に依らない既定経路**で、
 * `M ≥ 16` の計画だけが選ぶ（適用条件は src/kernels/state-attention.ts の
 * `stateQkTiledEligible` / `statePvTiledEligible`）。したがって期待は「M ≥ 16 → タイル経路
 * （席が `"parallel"` でも変わらない）・M < 16 → 席どおり」。**①' だけは席の中でさらに M ≤ 8 に
 * 限られる**ので、`M = 9` の parallel は「①QK は逐次・③PV は ③'」という**段で違う**行になる。
 *
 * MUST: 3 経路を**全て**見る（期待した 1 本が出ていることと、残り 2 本が 1 本も出ていないことの
 * 両方）。片側だけだと「タイル経路が結線から落ちて参照経路が走っている」が素通りする —
 * ①ₜ / ③ₜ は参照経路と値が 1 ビットも違わないので、数値門は原理的に鳴らない。
 * MUST: 期待は行ごとに**段ごとに**直書きする（判定を輸入すると実装と一緒に間違える）。
 */
Deno.test({
  name:
    "states 形 attention ①QK / ③PV の 3 経路は計画の M と席どおりに走る（census・実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [label, model, sliding, reduce, chunkRows, qkExpected, pvExpected] of [
          ["full r=2 prefill M=16", TILED, false, "sequential", 16, "tiled", "tiled"],
          // 席はタイル経路を動かさない（ビット同一なので既定経路 — 席で切り替える対象ではない）
          ["full r=2 parallel prefill M=16", TILED, false, "parallel", 16, "tiled", "tiled"],
          ["sliding r=1 prefill M=16", TILED_SLIDING, true, "sequential", 16, "tiled", "tiled"],
          // M=2 はどちらのタイル経路も適用外で ①'（M ≤ 8）も ③' も席どおり上がる
          ["full r=2 parallel M=2", TILED, false, "parallel", 2, "parallel", "parallel"],
          // M=9 は ①' の適用外（M ≤ 8 の外・タイル経路の手前）なので ①QK は逐次のまま、
          // ③PV だけが席どおり ③' へ上がる（段で適用範囲が違うことの直接の観測点）
          ["full r=2 parallel M=9", TILED, false, "parallel", 9, "sequential", "parallel"],
          // M=1 も適用外。席どおりに 2 段とも分かれる
          ["full r=2 decode M=1", TILED, false, "sequential", 1, "sequential", "sequential"],
          ["full r=2 parallel decode M=1", TILED, false, "parallel", 1, "parallel", "parallel"],
        ] as const
      ) {
        const session = await stateSession(gpu, model, { stateAttentionReduce: reduce });
        const context = await session.createGenerationContext({ chunkLength: chunkRows });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 7),
            chunkRows,
            chunkRows,
          );
          const keys = session.diagnostics().lastRunTiming?.entries.map((entry) => entry.key) ?? [];
          assert(keys.length > 0, `${label}: 内訳が空（キー検査が空振りしている）`);
          const gqa = model.heads !== model.kvHeads;
          const shown = keys.join(" / ");
          for (
            const [stage, expected, routes] of [
              ["①QK", qkExpected, [
                ["sequential", stateQkKey(sliding, gqa)],
                ["parallel", stateQkParallelKey(sliding, gqa)],
                ["tiled", stateQkTiledKey(sliding, gqa, chunkRows)],
              ]],
              ["③PV", pvExpected, [
                ["sequential", statePvKey(sliding, gqa)],
                ["parallel", statePvParallelKey(sliding, gqa)],
                ["tiled", statePvTiledKey(sliding, gqa, chunkRows)],
              ]],
            ] as const
          ) {
            for (const [route, key] of routes) {
              assertEquals(
                keys.includes(key),
                route === expected,
                `${label}: ${stage} の ${route} 経路キー ${key} の有無が期待と違う（走った内訳: ${shown}）`,
              );
            }
          }
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
 * **進行中 run のリース**（波 D-7）。
 *
 * run は頭で捕捉した `pastLength` で uniform を書き・dispatch 数を算出し・成功時にその値を
 * 基準に進める。途中で `rewind` が論理長を動かすと「GPU が読んだ P」と「進行の基準にした P」が
 * 分裂し、例外も NaN も出ないまま KV の論理位置だけがずれる。リースは `Session.run` の
 * **同期区間**で立つので、`run()` を await しない並びでも捕まえられる（本体でリースを取ると
 * マイクロタスク 1 段のぶんだけ窓が開く）。
 */
Deno.test({
  name: "進行中の generation run がある間の rewind は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 1 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 1, 1, 0, "decode 1");
      await assertStep(session, context, FULL, 8, state, 1, 1, 17, "decode 2");
      assertEquals(context.pastLength, 2);

      // **await しない**まま rewind を撃つ（本体でリースを取る実装ではここが素通りする）。
      const inputs = stepInputs(FULL, 1, 29);
      const running = runStep(session, context, FULL, inputs, 1, 1);
      const rejected = assertThrows(() => context.rewind(0), ExecutionError);
      assert(rejected.message.includes("進行中の generation run"), rejected.message);

      // run 自体は無傷（rewind の拒否が run を巻き添えにしない）。
      const actual = await running;
      const expected = advanceOracle(FULL, 8, state, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(report.pass, true, `リース中の run: ${formatAllclose(report)}`);
      assertEquals(context.pastLength, 3);

      // 対照: 決着後の rewind は通る（リースがちゃんと返っている = 永久拒否になっていない）。
      context.rewind(1);
      assertEquals(context.pastLength, 1);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * **dispose の 2 段化**（波 D-7）。
 *
 * 同期に立つのは 1 段目（新規受付の終了）だけで、内部面の遮断は Session チェーンに積んだ破棄
 * 本体が走る 2 段目。破棄本体は先行 run の**後**に走るので、`run(); context.dispose();` の
 * 非 await 並びは Session の `run(); session.dispose();` と同じ意味論になる（受理済み run は
 * 完走し、その後で実体が返る）。1 段で閉じる実装ではここで run が「dispose 済み」で落ちる。
 */
Deno.test({
  name: "dispose は新規受付だけを同期に閉じ、受理済み run は完走する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 1 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 1, 1, 0, "decode 1");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "decode 2");
      assertEquals(context.pastLength, 2);

      // **await しない**まま dispose。受理済み run は past=2 の形で最後まで走り切る。
      const inputs = stepInputs(FULL, 1, 23);
      const running = runStep(session, context, FULL, inputs, 1, 1);
      const disposing = context.dispose();
      const actual = await running;
      const expected = advanceOracle(FULL, 8, state, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(report.pass, true, `dispose と並んだ run: ${formatAllclose(report)}`);
      await disposing;

      // 1 段目の効き: 利用者面は dispose の**呼び出し時点**から閉じている。
      assertThrows(() => context.pastLength, ExecutionError, "dispose 済み");
      // 新規 run は admission で拒否（チェーンへ積む前 = 同期区間の判定）。
      const rejected = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 41), 1, 1),
        ExecutionError,
      );
      assert(rejected.message.includes("dispose 済み"), rejected.message);
    } finally {
      await session.dispose();
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
      ["attentionCompute a8", { attentionCompute: "a8" }, "attentionCompute"],
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

/**
 * 実行形の門（ADR 0066 決定 4 — 固定長 chunk / decode の 2 本だけ）。
 *
 * `M ∈ 許可集合`（この context は `chunkBuckets` を宣言していないので `{chunkLength, 1}`）を
 * 実行時に課すようになった（波 D-7）ので、`queryLength` の上限は
 * `Q ≤ M` の 1 本に**畳まれている**: `M = chunkLength` なら `Q ≤ M` が `Q ≤ chunkLength` を
 * 含意し、`M = 1` なら `Q = 1` を含意する。つまり `Session.run` 経由で `Q > chunkLength` だけを
 * 単独で踏む形は**構造的に作れない**（作るには M ∉ 許可集合が要り、それは先に落ちる）。
 * `GenerationContext` 側の `queryLength ≤ chunkLength` 検査は内部面の防波堤として残っており、
 * 直接駆動する門は tests/gpu_generation_context_test.ts が持つ（二重簿記にしない）。
 */
Deno.test({
  name: "実行形は固定長 chunk と decode の 2 本だけ・queryLength は M / 正整数で拒否（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    // chunkLength=2 に対し M=2（prefill 形）と M=1（decode 形）の 2 本だけが通る。
    const context = await session.createGenerationContext({ chunkLength: 2 });
    try {
      // ① Q > M（decode 形の宣言 shape の外 — 導出相が集めた chunkRows が見る）
      const overRows = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 2),
        ExecutionError,
      );
      assert(overRows.message.includes("物理 chunk 行数"), overRows.message);

      // ② Q > M（prefill 形 — chunkLength の上限もこの 1 本に畳まれている）
      const overChunk = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 2, 0), 2, 3),
        ExecutionError,
      );
      assert(overChunk.message.includes("物理 chunk 行数"), overChunk.message);

      // ③ Q = 0（「何も進めない run」— 進行と物理内容の対応が観測できなくなる）
      const zero = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 2, 0), 2, 0),
        ExecutionError,
      );
      assert(zero.message.includes("queryLength"), zero.message);

      // ④ M ∉ 許可集合（この context は chunkBuckets を宣言していないので {1, chunkLength}）:
      //    任意の M を通すと M の種類ぶん別鍵の計画が増え、PreparedPlan の LRU を汚して
      //    decode のホットパスが静かに再導出へ落ちる（ADR 0066 決定 4 の「PreparedPlan は
      //    2 本が定常」— 追記〈バケット〉が広げるのは「宣言した本数だけ」）。
      for (const chunkRows of [3, 4]) {
        const wrongForm = await assertRejects(
          () => runStep(session, context, FULL, stepInputs(FULL, chunkRows, 0), chunkRows, 1),
          ExecutionError,
        );
        assert(wrongForm.message.includes("固定 chunk 契約"), wrongForm.message);
      }

      assertEquals(context.pastLength, 0, "拒否された run は 1 つも進めない");

      // 対照: 2 本の実行形はそのまま通る（門が「何でも赤くする」形になっていない裏）。
      const state = newOracle(FULL, 8);
      await assertStep(session, context, FULL, 8, state, 2, 2, 71, "prefill 形（M=chunkLength）");
      await assertStep(session, context, FULL, 8, state, 1, 1, 83, "decode 形（M=1）");
      assertEquals(context.pastLength, 3);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** 新規の拒否形の追加検査（chunkLength=4 に M=2 / M=3 — decode 形の 1 とも一致しない）。 */
Deno.test({
  name: "chunkLength と 1 のどちらでもない M は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      for (const chunkRows of [2, 3]) {
        const error = await assertRejects(
          () => runStep(session, context, FULL, stepInputs(FULL, chunkRows, 0), chunkRows, 1),
          ExecutionError,
        );
        assert(error.message.includes(`物理 chunk 行数 ${chunkRows}`), error.message);
        // 診断は許可集合を列挙する（バケット宣言が無いので prefill 形は chunkLength の 1 本）。
        assert(error.message.includes("prefill 形の {4}"), error.message);
      }
      assertEquals(context.pastLength, 0, "拒否された run は 1 つも進めない");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 有効行 `query` 行ぶんの値を物理 `rows` 行の `[1, planes, rows, depth]` へ 0 詰めで並べる
 * （pad 領域の入力値は 0 埋め MUST — ADR 0066 追記 6）。
 *
 * バケット形（M=4）と prefill 形（M=8）で**同じ有効行**を渡すのに要る。`stepInputs` の平坦列は
 * M ごとに行の切れ目が変わるので、そのまま両方へ渡すと別の値を比べることになる。
 */
const scatterRows = (
  planes: number,
  rows: number,
  depth: number,
  query: number,
  valid: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * rows * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    const from = plane * query * depth;
    out.set(valid.subarray(from, from + query * depth), plane * rows * depth);
  }
  return out as Float32Array<ArrayBuffer>;
};

/** `[1, planes, rows, depth]` の先頭 `query` 行だけを取り出す（pad 行を落とした比較用）。 */
const takeRows = (
  planes: number,
  rows: number,
  depth: number,
  query: number,
  data: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * query * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    const from = plane * rows * depth;
    out.set(data.subarray(from, from + query * depth), plane * query * depth);
  }
  return out as Float32Array<ArrayBuffer>;
};

/** 同じ有効行を持つ M 行ぶんの入力（`stepInputs` の `salt` 規則をそのまま使う）。 */
const bucketInputs = (
  model: StateModel,
  chunkRows: number,
  query: number,
  salt: number,
): StepInputs => ({
  q: scatterRows(
    model.heads,
    chunkRows,
    model.depth,
    query,
    seeded(model.heads * query * model.depth, (i) => QUERY(i + salt)),
  ),
  k: scatterRows(
    model.kvHeads,
    chunkRows,
    model.depth,
    query,
    seeded(model.kvHeads * query * model.depth, (i) => KEY(i + salt)),
  ),
  v: scatterRows(
    model.kvHeads,
    chunkRows,
    model.depth,
    query,
    seeded(model.kvHeads * query * model.depth, (i) => VALUE(i + salt)),
  ),
});

/**
 * chunkBuckets が許す追加の prefill 形（ADR 0066 追記〈バケット〉）。
 *
 * 見るのは 3 つ: ①バケット行が通る ②同じ行数がバケット無しの context では落ちる（許可が
 * バケット由来であることの裏）③バケット形の**有効行の出力が prefill 形（M=chunkLength）と
 * 一致する** — 短い chunk を pad 無しで回すのはこの一致が成り立つ限りでの高速化なので、
 * 一致が崩れたら「速いが違う値」になる（例外も警告も出ない）。
 */
Deno.test({
  name: "chunkBuckets の物理行数は通り、有効行の出力は M=chunkLength と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    // chunkLength=8 / バケット [4]。許可集合は {1, 4, 8}。
    const bucketed = await session.createGenerationContext({ chunkLength: 8, chunkBuckets: [4] });
    const plain = await session.createGenerationContext({ chunkLength: 8 });
    try {
      // ② 集合の隙間（3）は落ちる。
      const gap = await assertRejects(
        () => runStep(session, bucketed, FULL, stepInputs(FULL, 3, 0), 3, 3),
        ExecutionError,
      );
      assert(gap.message.includes("prefill 形の {4, 8}"), gap.message);

      // ② 同じ M=4 が、バケットを宣言していない context では落ちる。
      const withoutBuckets = await assertRejects(
        () => runStep(session, plain, FULL, bucketInputs(FULL, 4, 3, 0), 4, 3),
        ExecutionError,
      );
      assert(withoutBuckets.message.includes("prefill 形の {8}"), withoutBuckets.message);
      assertEquals(bucketed.pastLength, 0, "拒否された run は 1 つも進めない");
      assertEquals(plain.pastLength, 0);

      // ① / ③ 同じ有効行 3 行を M=4（バケット）と M=8（prefill 形）で回す。
      const QUERY_ROWS = 3;
      const bucketState = newOracle(FULL, 8);
      const plainState = newOracle(FULL, 8);
      const bucketIn = bucketInputs(FULL, 4, QUERY_ROWS, 0);
      const plainIn = bucketInputs(FULL, 8, QUERY_ROWS, 0);

      const bucketOut = await runStep(session, bucketed, FULL, bucketIn, 4, QUERY_ROWS);
      const plainOut = await runStep(session, plain, FULL, plainIn, 8, QUERY_ROWS);
      assertEquals(bucketed.pastLength, QUERY_ROWS);
      assertEquals(plain.pastLength, QUERY_ROWS);

      // 両方とも CPU 参照と一致する（片側だけ見ると「同じように壊れた」形を見逃す）。
      for (
        const [label, state, inputs, actual, chunkRows] of [
          ["バケット形（M=4）", bucketState, bucketIn, bucketOut, 4],
          ["prefill 形（M=8）", plainState, plainIn, plainOut, 8],
        ] as const
      ) {
        const expected = advanceOracle(FULL, 8, state, inputs, chunkRows, QUERY_ROWS);
        const valid = takeRows(FULL.heads, chunkRows, FULL.depth, QUERY_ROWS, expected);
        assert(valid.some((value) => Math.abs(value) > 1e-3), `${label}: 期待出力が自明（全 ~0）`);
        const report = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        );
        assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
      }

      // ③ 有効行そのものの突合（pad 行を落として直接比べる）。
      const report = compareTensors(
        { dtype: "f32", data: takeRows(FULL.heads, 4, FULL.depth, QUERY_ROWS, bucketOut) },
        { dtype: "f32", data: takeRows(FULL.heads, 8, FULL.depth, QUERY_ROWS, plainOut) },
        STATE_TOLERANCE,
      );
      assertEquals(report.pass, true, `バケット形と prefill 形の有効行: ${formatAllclose(report)}`);

      // decode 形（M=1）はバケットを宣言しても従来どおり通る。
      await assertStep(session, bucketed, FULL, 8, bucketState, 1, 1, 37, "decode 形（M=1）");
    } finally {
      await bucketed.dispose();
      await plain.dispose();
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

/**
 * **両方に現れる記号**の照合（ADR 0066 追記 7 の「効く範囲の分担」の実行時執行）。
 *
 * スロット容量に入力と同じ記号 `M` を使うと、束縛点が 2 つになる: 入力 shape（run —
 * `bindSymbols`）と `spec.bindings`（context）。割れたまま走ると「確保容量は 4・計画と dispatch は
 * 3」のような分裂が例外も警告も無く成立するので、エンコード前に落とす。
 */
Deno.test({
  name: "states と入力の両方に現れる記号は 2 つの束縛点で一致しないと fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const shared: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "M" };
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, shared);
    // 容量 M=4 で確保した context（chunkLength も 4 = prefill 形の M）。
    const context = await session.createGenerationContext({ chunkLength: 4, bindings: { M: 4 } });
    try {
      // 割れた形: run の入力 shape が M=3 を解決する（context は 4 で確保済み）。
      const split = await assertRejects(
        () => runStep(session, context, shared, stepInputs(shared, 3, 0), 3, 3),
        ExecutionError,
      );
      assert(split.message.includes("記号 'M'"), split.message);
      assert(split.message.includes("4"), split.message);
      assert(split.message.includes("3"), split.message);
      assertEquals(context.pastLength, 0, "拒否された run は進めない");

      // 対照: 一致していれば通る（門が「常に赤い」形になっていない裏）。
      const state = newOracle(shared, 4);
      await assertStep(session, context, shared, 4, state, 4, 4, 13, "M=4 で一致");
      assertEquals(context.pastLength, 4);
    } finally {
      await context.dispose();
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
 * 決定的な失敗注入として、**論理長を u32 上限まで進めてから** state を書く decode 形の run を
 * 1 本通す。dispatch は全て成功して submit されるが、成功後の `advance` が u32 の上限で落ちるので、
 * 「state を submit した run が例外で終わる」形になる — スナップショット比較が働けば context は
 * poison され、働かなければ「物理 ring だけ進んだ context」が正常値を返し続ける。
 *
 * NOTE: 走らせるのは **M=1 の decode 形**（固定 chunk 契約で `M ∈ {chunkLength, 1}` — 大きな
 * `chunkLength` は「手動 advance の 1 回で u32 上限まで飛ばす」ために要るだけ）。
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
      internals(context).advance(0, 0xffffffff);
      assertEquals(context.pastLength, 0xffffffff);

      const inputs = stepInputs(SLIDING, 1, 0);
      const error = await assertRejects(
        () => runStep(session, context, SLIDING, inputs, 1, 1),
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
 * backing が**退役して作り直された**後の復帰（ADR 0066 決定 5 の世代識別子）。
 *
 * M=4（prefill 形）と M=1（decode 形）は別鍵で、予算 0 の Session では 1 本の slot backing を
 * 奪い合う（従来の容量 1 — ADR 0095 決定 1）。退役した backing の slot / 入力バッファは run の
 * 後始末で `destroy()` されるため、context 側が古い束を掴んだまま回れば**破棄済みバッファを
 * 束ねた dispatch** になる（値か例外のどちらかで必ず壊れる）。焼き直しが backing の再構築に
 * 追随していることを、値の正しさと回数の両方で押さえる。
 */
Deno.test({
  name: "予算 0 では切替のたびに backing を作り直し、context 側も焼き直す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL, { planBackingBudgetBytes: 0 });
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
      // 予算 0 では鍵が変わるたびに退役 → 再構築なので、context 側も毎 run 焼き直す。
      // MUST: 焼き直し回数が backing の構築回数に追随すること — 追随しないなら、退役した
      // backing のバッファを束ねた束が使い回されている。
      assertEquals(
        builds,
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        "backing の再構築に context 側の焼き直しが追随していない",
      );
      assertEquals(
        session.diagnostics().planBacking.retainedCount,
        1,
        "予算 0 で 2 本以上保持した",
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
 * 予算内で**保持した形**の間の行き来では焼き直しが増えない（ADR 0095 決定 3）。
 *
 * 束の寿命は (context, backing 実体) の組のままで、保持中の backing ごとに 1 束持つ。上の
 * 予算 0 の門と同じ往復を既定予算で回して、`rebindCount` が形ごとの初回 1 回ずつで止まること
 * を見る（生成 1 ターンの prefill ⇄ decode の切替で毎回焼き直す形が、この波で消えた点）。
 */
Deno.test({
  name: "予算内で保持した 2 形の往復は焼き直しを増やさない（既定予算・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 4, 1, 0, "M=4 初回（導出）");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "M=1 初回（導出）");

      const builds: [number, number][] = [];
      for (const [rows, salt] of [[1, 23], [4, 31], [1, 43], [4, 53], [1, 61], [4, 71]] as const) {
        await assertStep(session, context, FULL, 8, state, rows, 1, salt, `M=${rows} へ切替`);
        assertEquals(session.diagnostics().lastRunPrepared?.hit, true, "レシピは再導出しない");
        const diagnostics = session.diagnostics();
        builds.push([diagnostics.planBacking.buildCount, diagnostics.stateBacking.rebindCount]);
      }
      // 3 往復しても構築は形ごとに 1 回・焼き直しも形ごとに 1 回で止まる。値の正しさは
      // 各 step の突合（`assertStep`）が持つので、「焼き直さないまま別の形の実体を束ねている」
      // 形はここが緑でも値で落ちる。
      assertEquals(
        builds,
        [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        "保持した形の往復で backing の作り直し / 焼き直しが起きている",
      );
      assertEquals(session.diagnostics().planBacking.retainedCount, 2, "2 形が保持されていない");
      assertEquals(context.pastLength, 8);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** backing 1 本ぶんの実測（領域 = `residentBytes` / 所有する入力バッファ = `inputBytes`）。 */
type Measured = { readonly residentBytes: number; readonly inputBytes: number };

/**
 * 予算が勘定する量（`#evictBackingsFor` と同じ算式 — 領域と所有入力バッファの両方）。
 * 領域だけで組むと、入力バッファのぶんだけ実際の勘定が予算を上回り、「予算ちょうど」の
 * つもりの Session が最初から超過側（= 常に全退役）で回る。
 */
const accountedBytes = (measured: readonly Measured[]): number =>
  measured.reduce((total, one) => total + one.residentBytes + one.inputBytes, 0);

/** 保持集合が `retained` のときの診断（**全欄** — 部分一致に逃げない）。 */
const expectStats = (retained: readonly Measured[], buildCount: number): PlanBackingStats => ({
  residentBytes: retained.reduce((total, one) => total + one.residentBytes, 0),
  inputBytes: retained.reduce((total, one) => total + one.inputBytes, 0),
  retainedCount: retained.length,
  buildCount,
});

/**
 * 形（M=`rows`）1 本ぶんの slot backing のバイト数を、**その形だけを回した別 Session** で
 * 測る（予算の門をこの実測から組むのは `gpu_plan_backing_test.ts` の同名 helper と同じ理由 —
 * 定数で書くと slot 表の詰め方が変わったときに「予算ちょうど」の意味が黙ってずれる）。
 */
const stateBackingBytes = async (
  gpu: GpuContext,
  model: StateModel,
  spec: Parameters<Session["createGenerationContext"]>[0],
  rows: number,
): Promise<Measured> => {
  const session = await stateSession(gpu, model);
  const context = await session.createGenerationContext(spec);
  try {
    const inputs = stepInputs(model, rows, 0);
    // 1 run 目 = ミス（導出）/ 2 run 目 = ヒット（backing の構築）。
    await runStep(session, context, model, inputs, rows, 1);
    await runStep(session, context, model, inputs, rows, 1);
    const stats = session.diagnostics().planBacking;
    assertEquals(stats.retainedCount, 1, `M=${rows} を 1 形だけ回した Session の保持が 1 本でない`);
    assert(stats.residentBytes > 0, `M=${rows} の backing が 0 バイト（門が空振る）`);
    // q / k / v は常駐でない `Tensor` なので、そのバッファは backing 所有 = 予算の勘定に入る。
    assert(stats.inputBytes > 0, `M=${rows} の所有入力バッファが 0 バイト（予算が勘定していない）`);
    return { residentBytes: stats.residentBytes, inputBytes: stats.inputBytes };
  } finally {
    await context.dispose();
    await session.dispose();
  }
};

/** 予算の門で使う形（capacity は往復 10 step ぶんの論理長を飲む余裕を取る）。 */
const BUDGET_MODEL: StateModel = { heads: 4, kvHeads: 4, depth: 4, capacity: 32 };
/** M ∈ {1, 2, 4} の 3 形を 1 つの context から出す指定（バケットは ADR 0066 追記〈バケット〉）。 */
const BUDGET_SPEC = { chunkLength: 4, chunkBuckets: [2] } as const;

/**
 * **退役 → 再構築で焼き直しが 1 回増える**（= 退役と同時に束を捨てている — ADR 0095 決定 3 の MUST）。
 *
 * 予算を 2 形（M=4 + M=1）ちょうどに絞り、3 形目（M=2）を入れて最古の M=4 を 1 本だけ
 * 退役させる。見るのは 2 つ:
 * ①退役した backing の世代の束が context から**消えている**（`bakedGroups` が undefined）—
 * 捨てないと、破棄済みバッファを束ねた group が context の参照ぶんだけ生き残る ②戻った run が
 * 焼き直しを 1 回増やす。保持したままの形の束は残っている（①が「全部捨てる」ではないことの対）。
 */
Deno.test({
  name: "退役した backing の束は捨てられ、戻った run で 1 回だけ焼き直す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const wide = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 4);
    const decode = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 1);
    const mid = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 2);
    // 予算 = M=4 + M=1 に対し、最古（M=4）を 1 本退役させれば M=2 が必ず収まる条件
    // （`M=1 + M=2 ≤ M=4 + M=1` ⟺ `M=2 ≤ M=4`）。崩れると退役が 2 本に及んで、見ているのが
    // 「最古 1 本の退役」ではなくなる。比べるのは予算が勘定する量（領域 + 所有入力）。
    assert(
      accountedBytes([mid]) <= accountedBytes([wide]),
      `M=2 の実測 ${JSON.stringify(mid)} が M=4 の ${
        JSON.stringify(wide)
      } 以下でない（門が空振る）`,
    );

    const session = await stateSession(gpu, BUDGET_MODEL, {
      planBackingBudgetBytes: accountedBytes([wide, decode]),
    });
    const context = await session.createGenerationContext(BUDGET_SPEC);
    const state = newOracle(BUDGET_MODEL, BUDGET_MODEL.capacity as number);
    const capacity = BUDGET_MODEL.capacity as number;
    /** 焼いた束の世代識別子（`#generationGroups` が預ける token を順に拾う）。 */
    const tokens: number[] = [];
    const inner = internals(context);
    const setBaked = inner.setBakedGroups;
    inner.setBakedGroups = (token, groups) => {
      tokens.push(token);
      setBaked(token, groups);
    };
    try {
      // M=4 と M=1 を 2 本ずつ回して両方を保持させる（1 本目は導出 = backing 不使用）。
      for (const [rows, salt] of [[4, 0], [4, 7], [1, 13], [1, 19]] as const) {
        await assertStep(
          session,
          context,
          BUDGET_MODEL,
          capacity,
          state,
          rows,
          1,
          salt,
          `M=${rows}`,
        );
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([wide, decode], 2),
        "予算ちょうどの 2 形が保持されていない",
      );
      assertEquals(tokens.length, 2, "焼き直しが形ごとに 1 回で収まっていない");
      const [wideToken, decodeToken] = tokens;

      // M=1 を触り直しても焼き直しは増えない（束が生きている）。LRU の順は [M=4, M=1] のまま。
      await assertStep(session, context, BUDGET_MODEL, capacity, state, 1, 1, 23, "M=1 触り直し");
      assertEquals(session.diagnostics().stateBacking.rebindCount, 2, "保持中の形を焼き直した");

      // 3 形目（M=2 のバケット形）。予算ちょうどなので最古の M=4 が 1 本だけ退役する。
      for (const salt of [29, 31]) {
        await assertStep(session, context, BUDGET_MODEL, capacity, state, 2, 1, salt, "M=2");
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([decode, mid], 3),
        "退役したのが最古（M=4）1 本ではない",
      );
      // 3 形目そのものの焼き込みが 1 回（保持中の M=1 は焼き直さない）。
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        3,
        "3 形目の焼き込みが 1 回でない",
      );
      // ① 退役した世代の束は context から消えている（捨てなければ実体の寿命が延びる）。
      assertEquals(
        inner.bakedGroups(wideToken),
        undefined,
        "退役した backing の束が context に残っている（破棄済みバッファを束ねた group が生き残る）",
      );
      assert(
        inner.bakedGroups(decodeToken) !== undefined,
        "保持中の backing の束まで捨てている（往復のたびに焼き直す形へ戻る）",
      );

      // ② 戻った run は作り直し + 焼き直しをちょうど 1 回ずつ増やす。
      await assertStep(session, context, BUDGET_MODEL, capacity, state, 4, 1, 37, "M=4 へ復帰");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true, "レシピは再導出しない");
      assertEquals(session.diagnostics().planBacking.buildCount, 4, "退役した形を作り直していない");
      assertEquals(session.diagnostics().stateBacking.rebindCount, 4, "復帰で焼き直していない");
      assertEquals(tokens.length, 4);
      assert(tokens[3] !== wideToken, "作り直した backing が同じ世代識別子を名乗っている");
    } finally {
      inner.setBakedGroups = setBaked;
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

/**
 * 席 `"parallel"`（①' の D 並列縮約 + ③' の KV 並列縮約）を **`Session.run` 経由で値まで**
 * 見る門。
 *
 * 現状 runtime の検証は「直接 dispatch の帯」（tests/gpu_state_attention_parallel_test.ts）と
 * 「executor 経由のキー検査」（上の census — `runStep` の戻りを捨てている）に割れており、
 * **その間**（レシピが ①' / ③' 用の workgroup 数・束縛・S / stats の確保を正しく組むか）は
 * 資産つきの models e2e にしか落ちていない（ミラー無しの環境では明示 SKIP される）。ここは
 * その隙間を資産なしで塞ぐ。
 *
 * MUST: ビット同一は要求しない（参照経路と ①' / ③' は縮約順が違う — 形が小さいと差が出ない
 * だけで、
 * 一致を契約にすると別の主張になる）。見るのは「オラクルと帯で一致」と「席をまたいでも
 * 同じ帯に収まる」の 2 点。
 */
Deno.test({
  name: "stateAttentionReduce:'parallel' は Session.run 経由でも参照鎖と一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const parallel = await stateSession(gpu, GQA, { stateAttentionReduce: "parallel" });
    const sequential = await stateSession(gpu, GQA);
    const parallelContext = await parallel.createGenerationContext({ chunkLength: 4 });
    const sequentialContext = await sequential.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(GQA, 8);
    try {
      // prefill（M=4 / Q=4）→ decode ×3（M=1 / Q=1）。salt は step ごとに変える
      for (
        const [chunkRows, query, salt, label] of [
          [4, 4, 2, "prefill"],
          [1, 1, 17, "decode 1"],
          [1, 1, 29, "decode 2"],
          [1, 1, 43, "decode 3"],
        ] as const
      ) {
        const inputs = stepInputs(GQA, chunkRows, salt);
        const actual = await runStep(parallel, parallelContext, GQA, inputs, chunkRows, query);
        const baseline = await runStep(
          sequential,
          sequentialContext,
          GQA,
          inputs,
          chunkRows,
          query,
        );
        const expected = advanceOracle(GQA, 8, state, inputs, chunkRows, query);
        // 恒真化の門: 期待出力が自明でない（両側が全 0 なら突合は何も見ていない）
        assert(
          expected.some((value) => Math.abs(value) > 1e-3),
          `${label}: 期待出力が自明（全 ~0）`,
        );
        // ① ③' が参照鎖と一致する
        const against = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        );
        assertEquals(against.pass, true, `${label} parallel: ${formatAllclose(against)}`);
        // ② ③' と ③ が同じ帯に収まる（席の違いが値へ漏れない）
        const cross = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: baseline },
          STATE_TOLERANCE,
        );
        assertEquals(cross.pass, true, `${label} parallel vs sequential: ${formatAllclose(cross)}`);
        // ③ 論理長の進行は席に依らない
        assertEquals(parallelContext.pastLength, state.past, `${label}: parallel の pastLength`);
        assertEquals(
          sequentialContext.pastLength,
          state.past,
          `${label}: sequential の pastLength`,
        );
      }
    } finally {
      await parallelContext.dispose();
      await sequentialContext.dispose();
      await parallel.dispose();
      await sequential.dispose();
      gpu.destroy();
    }
  },
});

/** 融合の適用・非適用が分かれる形（`chunkRows <= 8` かつ `colCap <= 1024` が適用条件）。 */
const FUSION_CASES = [
  [1, 64, undefined, true],
  [8, 39, 31, true],
  [9, 64, undefined, false],
  [1, 1025, undefined, false],
  [8, 1024, undefined, true],
] as const;

/**
 * 席 `"parallel-fused"`（② と ③' の融合）を `Session.run` 経由で見る門の 1 本目 = 値。
 *
 * MUST: 数値一致とキー確認は**別テストに割る**（キー確認は下）。`TIMING_ACQUIRE_OPTIONS` は
 * timestamp-query 不在で `gpuTiming: false` に落ち、`lastRunTiming` が undefined になるので
 * キー確認は timestamp 付き device に閉じるしかない。一方 u32 一致は timestamp を要らないので、
 * 1 本にまとめると Session を通した唯一の数値一致まで timestamp 不在の device で丸ごと
 * skip され、結線の誤りを自動検証が 1 件も捕まえなくなる。
 */
Deno.test({
  name: "parallel-fused は state 更新後も parallel と u32 一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const [chunkRows, capacity, window] of FUSION_CASES) {
        const model: StateModel = { heads: 4, kvHeads: 1, depth: 17, capacity, window };
        const sessions = [
          await stateSession(gpu, model, { stateAttentionReduce: "parallel" }),
          await stateSession(gpu, model, { stateAttentionReduce: "parallel-fused" }),
        ];
        const contexts = await Promise.all(
          sessions.map((s) => s.createGenerationContext({ chunkLength: chunkRows })),
        );
        try {
          for (let step = 0; step < (window ? 8 : 4); step++) {
            const input = stepInputs(model, chunkRows, step * 13);
            const before = await runStep(
              sessions[0],
              contexts[0],
              model,
              input,
              chunkRows,
              chunkRows,
            );
            const after = await runStep(
              sessions[1],
              contexts[1],
              model,
              input,
              chunkRows,
              chunkRows,
            );
            assertEquals(new Uint32Array(after.buffer), new Uint32Array(before.buffer));
          }
          assertEquals(contexts[1].pastLength, contexts[0].pastLength);
        } finally {
          for (const c of contexts) await c.dispose();
          for (const s of sessions) await s.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 2 本目 = キー。適用形では融合キーが 1 本だけ立ち、非適用形では従来の ② / ③' が立つ。
 *
 * NOTE: 融合の採否はレシピ構築時に `chunkRows` と `colCap` だけで決まるので、1 step 走らせれば
 * 立つキーは出揃う（値の進行は上のテストが見る）。
 */
Deno.test({
  name: "parallel-fused は適用形だけstats/PVをまとめる（実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const [chunkRows, capacity, window, expectedFusion] of FUSION_CASES) {
        const model: StateModel = { heads: 4, kvHeads: 1, depth: 17, capacity, window };
        const session = await stateSession(gpu, model, {
          stateAttentionReduce: "parallel-fused",
        });
        const context = await session.createGenerationContext({ chunkLength: chunkRows });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 13),
            chunkRows,
            chunkRows,
          );
          const entries = session.diagnostics().lastRunTiming?.entries ?? [];
          assert(entries.length > 0);
          assertEquals(
            entries.some((e) => e.key.startsWith("attention_state_stats_pv:")),
            expectedFusion,
          );
          assertEquals(
            entries.some((e) => e.key.startsWith("attention_state_stats:")),
            !expectedFusion,
          );
          assertEquals(
            entries.some((e) => e.key.startsWith("attention_state_pv:")),
            !expectedFusion,
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
