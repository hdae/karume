// 借り手 context（drafter）と共有重みの**実行統合**門（ADR 0096 段 2 §2.1〜§2.3）。
//
// 宣言と契約の門は tests/runtime_external_states_test.ts（GPU 非依存）が持つ。こちらが見るのは
// 「2 つの Session が 1 本の KV と 1 本の重みを共有して走る」結線そのもの:
//
//   ① 貸し手を数 step 走らせた後の借り手 run が、**貸し手スロットの実内容**から作った f64 参照
//      （`referenceStateAttentionReadonly`）と帯内（P < W / P = W / ring 一周後の 3 相を通る）
//   ② 直列化: 貸し手の進行中 run / commit 待ち / poison の 3 つが借り手 run の拒否理由になる
//   ③ 寿命: 借り手が生きている間の貸し手 context dispose は ExecutionError・借り手 dispose 後は通る
//   ④ 計画鍵: 容量の違う貸し手を束ねた 2 本の借り手 context は**別鍵**（external スロットの容量が
//      鍵に載っていることの検出器）
//   ⑤ 共有重み: 貸し手の i8 embedding 表を借り手が読み、値が貸し手と**ビット一致**する。
//      shape / dtype / 席の不一致は fail loudly・借り手が生きている間の貸し手 dispose も拒否
//
// MUST: 参照に渡すスロットは**GPU から読み戻す**（ホスト側オラクルの再現ではない）。借り手が
// 読むのは貸し手が実際に書いた行なので、読み戻しでしか「同じ実体を見ている」ことを示せない。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAttentionReadonly } from "../src/reference/state-attention.ts";
import { decodeI8 } from "../src/format/i8.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { openModel } from "../src/format/container.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import { LENGTHS_BYTES } from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { i8BytesFrom } from "./helpers/i8.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** 帯は states 形の実行統合門と同値・同根拠（参照は f64 積算・GPU は f32 逐次累積）。 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const HEADS = 4;
const KV_HEADS = 1;
const DEPTH = 8;
/** sliding の窓（貸し手・借り手で同値 MUST — 読み書き同式）。 */
const WINDOW = 8;

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

const SCALE = halfScale(DEPTH);

/** 貸し手（target 相当）: sliding な自前スロット 2 本 + states 形 attention + append 2 本。 */
const lenderGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, HEADS, "M", DEPTH] },
    { name: "k", dtype: "f32", shape: [1, KV_HEADS, "M", DEPTH] },
    { name: "v", dtype: "f32", shape: [1, KV_HEADS, "M", DEPTH] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, "M", DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH] },
    vslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: SCALE, window: WINDOW },
      states: { k: "kslot", v: "vslot" },
    },
    {
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "kslot" },
    },
    {
      op: "state_append",
      ins: ["v"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "vslot" },
    },
  ],
});

/** 借り手（drafter 相当）: 貸し手と**同名**の external スロットを readonly attention 1 本が読む。 */
const borrowerGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention"] },
  symbols: ["C"],
  inputs: [{ name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] }],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH], external: true },
    vslot: { dtype: "f32", shape: [1, KV_HEADS, "C", DEPTH], external: true },
  },
  nodes: [{
    op: "attention",
    ins: ["q"],
    outs: ["o"],
    attrs: { scale: SCALE, window: WINDOW, readonly: true },
    states: { k: "kslot", v: "vslot" },
  }],
});

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** GPU バッファを f32 として読み戻す（helpers/state-dispatch.ts の readRaw と同型）。 */
const readSlot = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const size = Math.max(4, count * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Float32Array(copy, 0, count);
};

const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/** 貸し手 context のスロット 2 本を読み戻す（借り手が実際に読む実体そのもの）。 */
const readLenderSlots = async (
  gpu: GpuContext,
  context: GenerationContext,
  capacity: number,
): Promise<{ slotK: Float32Array<ArrayBuffer>; slotV: Float32Array<ArrayBuffer> }> => {
  const count = KV_HEADS * capacity * DEPTH;
  const slots = internals(context).slots;
  const k = slots.get("kslot");
  const v = slots.get("vslot");
  assert(k !== undefined && v !== undefined, "貸し手のスロットが引けない");
  return {
    slotK: await readSlot(gpu.device, k.buffer, count),
    slotV: await readSlot(gpu.device, v.buffer, count),
  };
};

/** 貸し手を 1 step 進める（`chunkRows` 行 × `query` 有効行）。 */
const lenderStep = async (
  session: Session,
  context: GenerationContext,
  chunkRows: number,
  query: number,
  salt: number,
  commit: "immediate" | "deferred" = "immediate",
): Promise<void> => {
  await session.run(
    {
      q: tensor(
        [1, HEADS, chunkRows, DEPTH],
        seeded(HEADS * chunkRows * DEPTH, (i) => QUERY(i + salt)),
      ),
      k: tensor(
        [1, KV_HEADS, chunkRows, DEPTH],
        seeded(KV_HEADS * chunkRows * DEPTH, (i) => KEY(i + salt)),
      ),
      v: tensor(
        [1, KV_HEADS, chunkRows, DEPTH],
        seeded(KV_HEADS * chunkRows * DEPTH, (i) => VALUE(i + salt)),
      ),
    },
    {},
    { context, queryLength: query, commit },
  );
};

/** 借り手を 1 回走らせて、貸し手スロットの実内容から作った参照と突き合わせる。 */
const assertDraft = async (
  gpu: GpuContext,
  borrower: Session,
  borrowed: GenerationContext,
  lender: GenerationContext,
  capacity: number,
  salt: number,
  label: string,
): Promise<void> => {
  const past = lender.pastLength;
  const q = seeded(HEADS * DEPTH, (i) => QUERY(i + salt));
  const { slotK, slotV } = await readLenderSlots(gpu, lender, capacity);
  const outputs = await borrower.run(
    { q: tensor([1, HEADS, 1, DEPTH], q) },
    {},
    { context: borrowed, queryLength: 1 },
  );
  const actual = outputs["o"].data as Float32Array<ArrayBuffer>;
  const expected = referenceStateAttentionReadonly({
    batch: 1,
    heads: HEADS,
    kvHeads: KV_HEADS,
    depth: DEPTH,
    capacity,
    window: WINDOW,
    past,
    q,
    slotK,
    slotV,
    scale: SCALE,
  });
  // MUST: 期待値が自明でないことを毎回見る（P = 0 以外では非零 — 恒真な突合を避ける）。
  assert(
    (expected.data as Float32Array).some((value) => Math.abs(value) > 1e-3),
    `${label}: 期待出力が自明（全 ~0）`,
  );
  const report = compareTensors(
    { dtype: "f32", data: actual },
    { dtype: "f32", data: expected.data as Float32Array<ArrayBuffer> },
    STATE_TOLERANCE,
  );
  assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
  // 借り手は論理長を 1 つも動かさない（進めるのは貸し手の run だけ）。
  assertEquals(lender.pastLength, past, `${label}: 借り手 run が貸し手の論理長を動かした`);
  assertEquals(borrowed.pastLength, past, `${label}: 借り手の pastLength は最後に写した P`);
};

Deno.test({
  name:
    "借り手 context の run が貸し手スロットの実内容と一致する（P < W / = W / ring 一周後・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      // P = 0（列 0 本 = 空行 → 厳密 0）。参照も 0 なので自明チェックは掛けない。
      const empty = await borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context: borrowed, queryLength: 1 },
      );
      for (const value of empty["o"].data as Float32Array) {
        assert(Object.is(value, 0), `P = 0 の出力が厳密 0 でない（${value}）`);
      }

      // prefill（M=4）→ decode を重ねて 3 相を通す: P < W（4）/ P = W（8）/ ring 一周後（18）。
      await lenderStep(lender, lenderContext, 4, 4, 0);
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 101, "P=4 < W");
      for (let step = 0; step < 4; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 11 + step * 7);
      }
      assertEquals(lenderContext.pastLength, 8);
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 202, "P=8 = W");
      for (let step = 0; step < 10; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 71 + step * 5);
      }
      assertEquals(lenderContext.pastLength, 18, "ring を一周している（C=16）");
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 303, "P=18（一周後）");

      // 2 本目以降の draft は導出済み計画に当たる（借り手も通常の計画キャッシュに載る）。
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, true);
      // 借り手は state を 1 行も書かないので、貸し手の論理長は draft で動かない。
      assertEquals(lenderContext.pastLength, 18);

      // 診断は Session ごとに分かれる（借り手 run は貸し手の stateBacking を汚さない）。
      // 借り手が抱えるのは論理長 uniform の 8 バイトだけ — スロットは貸し手の所有物なので、
      // 両方が数えると同じ VRAM の二重計上になる。
      const lenderState = lender.diagnostics().stateBacking;
      const borrowerState = borrower.diagnostics().stateBacking;
      assertEquals(borrowerState.contextCount, 1);
      assertEquals(borrowerState.residentBytes, LENGTHS_BYTES);
      assertEquals(lenderState.contextCount, 1);
      assertEquals(
        lenderState.residentBytes,
        2 * KV_HEADS * capacity * DEPTH * 4 + LENGTHS_BYTES,
        "貸し手はスロット 2 本 + 論理長ぶんを数える",
      );
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

/**
 * 上の parity 門が**空振りでない**証明（故障注入）。参照側を 3 通り壊し、どれも帯の外へ出る
 * ことを見る:
 *
 *   ① P を 1 つ古くする — 借り手が「貸し手の**現在の** P」を写していることの検出器
 *   ② k / v スロットを取り違える — 束縛の向きの検出器（2 本は同形なので shape では捕まらない）
 *   ③ live 窓の内側の 1 行を潰す — 借り手が実際にその行を読んでいることの検出器
 *
 * MUST: 3 通りとも「参照側だけ」を壊す（GPU 側は正しいまま）。両側を同じだけ壊すと誤りが
 * 相殺して恒真になる。
 */
Deno.test({
  name: "借り手 parity の故障注入（P のずれ・k/v 取り違え・live 行の欠落）は帯で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      for (let step = 0; step < 3; step += 1) {
        await lenderStep(lender, lenderContext, 1, 1, 21 + step * 3);
      }
      const past = lenderContext.pastLength;
      assertEquals(past, 7);
      const q = seeded(HEADS * DEPTH, (i) => QUERY(i + 55));
      const { slotK, slotV } = await readLenderSlots(gpu, lenderContext, capacity);
      const outputs = await borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], q) },
        {},
        { context: borrowed, queryLength: 1 },
      );
      const actual = outputs["o"].data as Float32Array<ArrayBuffer>;
      const base = {
        batch: 1,
        heads: HEADS,
        kvHeads: KV_HEADS,
        depth: DEPTH,
        capacity,
        window: WINDOW,
        q,
        scale: SCALE,
      };
      const inBand = (expected: Float32Array<ArrayBuffer>): boolean =>
        compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        ).pass;

      // 正しい参照は帯の中（門が成立していることの前提）。
      assert(
        inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "正しい参照が帯の外（門そのものが壊れている）",
      );
      // ① P のずれ
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past: past - 1, slotK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "P を 1 つ古くしても帯の中（借り手が貸し手の現在の P を見ていない可能性）",
      );
      // ② k / v の取り違え
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK: slotV, slotV: slotK })
            .data as Float32Array<ArrayBuffer>,
        ),
        "k / v を入れ替えても帯の中（束縛の向きが検出できていない）",
      );
      // ③ live 窓の内側の 1 行（論理位置 P−1 = 直近確定 token の行）を潰す
      const damagedK = new Float32Array(slotK);
      const row = (past - 1) % capacity;
      for (let d = 0; d < DEPTH; d += 1) damagedK[row * DEPTH + d] = 0;
      assert(
        !inBand(
          referenceStateAttentionReadonly({ ...base, past, slotK: damagedK, slotV })
            .data as Float32Array<ArrayBuffer>,
        ),
        "live 行を潰しても帯の中（その行を読んでいない可能性）",
      );
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "借り手 run は貸し手の進行中 run / commit 待ち / poison で拒否される（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    const draft = (): Promise<unknown> =>
      borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context: borrowed, queryLength: 1 },
      );
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      await draft();

      // ① 進行中の貸し手 run（リースは発行の同期区間で立つので、await しない並びで踏める）。
      const running = lenderStep(lender, lenderContext, 1, 1, 5);
      const concurrent = await assertRejects(draft, ExecutionError);
      assert(concurrent.message.includes("進行中の generation run"), concurrent.message);
      await running;

      // ② commit 待ち（deferred な貸し手 run — draft は commit の後）。
      await lenderStep(lender, lenderContext, 1, 1, 9, "deferred");
      const pending = await assertRejects(draft, ExecutionError);
      assert(pending.message.includes("commit 待ち"), pending.message);
      lenderContext.commit(1);
      await draft();

      // ③ 借り手 run に deferred は指定できない（確定させる相手が居ない）。
      const deferred = await assertRejects(
        () =>
          borrower.run(
            { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
            {},
            { context: borrowed, queryLength: 1, commit: "deferred" },
          ),
        ExecutionError,
      );
      assert(deferred.message.includes("借り手 context"), deferred.message);

      // ④ 借り手の commit / rewind は常に拒否（論理長を持たない）。
      assertThrows(() => borrowed.commit(0), ExecutionError, "借り手 context");
      assertThrows(() => borrowed.rewind(0), ExecutionError, "借り手 context");

      // ⑤ 貸し手の poison は借り手 run の拒否理由になる（故障注入）。
      internals(lenderContext).poison("テストの故障注入");
      const poisoned = await assertRejects(draft, ExecutionError);
      assert(poisoned.message.includes("汚染された"), poisoned.message);
    } finally {
      await borrowed.dispose();
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "借り手が生きている間の貸し手 context dispose は拒否され、dispose 後は通る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const capacity = 16;
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: capacity },
    });
    const borrowed = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: lenderContext,
    });
    try {
      await lenderStep(lender, lenderContext, 4, 4, 0);
      const refused = await assertRejects(() => lenderContext.dispose(), ExecutionError);
      assert(refused.message.includes("借りている context"), refused.message);
      // 拒否された dispose は貸し手を壊さない（受付終了フラグも立っていない）。
      assertEquals(lenderContext.pastLength, 4);
      await assertDraft(gpu, borrower, borrowed, lenderContext, capacity, 404, "拒否の後");

      await borrowed.dispose();
      // 借り手が退いた後は通る。
      await lenderContext.dispose();
    } finally {
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "借り手 context の spec の門（borrow と external は対・chunkLength 1・継承する束縛・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: 16 },
    });
    try {
      // ① external があるのに borrow 無し（借り物を自分で確保すると空の過去を読む）
      const noBorrow = await assertRejects(
        () => borrower.createGenerationContext({ chunkLength: 1 }),
        ExecutionError,
      );
      assert(noBorrow.message.includes("external な state スロット"), noBorrow.message);

      // ② 自前スロットのグラフに borrow（誰も読まないスロットを貸し手から掴む）
      const ownSlots = await assertRejects(
        () => lender.createGenerationContext({ chunkLength: 4, borrow: lenderContext }),
        ExecutionError,
      );
      assert(ownSlots.message.includes("全スロットが external"), ownSlots.message);

      // ③ chunkLength は 1 ちょうど（借り手の実行形は decode 1 本）
      const chunk = await assertRejects(
        () => borrower.createGenerationContext({ chunkLength: 2, borrow: lenderContext }),
        ExecutionError,
      );
      assert(chunk.message.includes("chunkLength"), chunk.message);

      // ④ bindings は継承する（渡すと二重の束縛点になる）
      const bindings = await assertRejects(
        () =>
          borrower.createGenerationContext({
            chunkLength: 1,
            borrow: lenderContext,
            bindings: { C: 16 },
          }),
        ExecutionError,
      );
      assert(bindings.message.includes("継承"), bindings.message);

      // ⑤ chunkBuckets は宣言できない（`chunkLength = 1` なので既存のバケット門が落とす）
      const buckets = await assertRejects(
        () =>
          borrower.createGenerationContext({
            chunkLength: 1,
            borrow: lenderContext,
            chunkBuckets: [2],
          }),
        ExecutionError,
      );
      assert(buckets.message.includes("chunkBuckets"), buckets.message);

      // ⑥ 窓が貸し手と食い違う借り手（読み書き同式の破れ — 窓外の行を過去として読む）
      const wideWindow = borrowerGraph();
      wideWindow.nodes[0].attrs = { scale: SCALE, window: WINDOW + 2, readonly: true };
      const wide = await createSession(gpu, openModel(graphModelBuffer(wideWindow)));
      try {
        const mismatch = await assertRejects(
          () => wide.createGenerationContext({ chunkLength: 1, borrow: lenderContext }),
          ExecutionError,
        );
        assert(mismatch.message.includes("貸し手と食い違う"), mismatch.message);
      } finally {
        await wide.dispose();
      }

      // 正常系: 貸し手の bindings（C=16）を継承し、余裕も写す。
      const borrowed = await borrower.createGenerationContext({
        chunkLength: 1,
        borrow: lenderContext,
      });
      assertEquals(borrowed.chunkLength, 1);
      assertEquals(borrowed.chunkBuckets, []);
      assertEquals(borrowed.slidingSlack, lenderContext.slidingSlack, "余裕は貸し手の写し");
      assertEquals(borrowed.slidingSlack, 16 - WINDOW);
      await borrowed.dispose();
    } finally {
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "確保の await を跨いだ chunkLength の書き換えは借り手 context に届かない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const lenderContext = await lender.createGenerationContext({
      chunkLength: 4,
      bindings: { C: 16 },
    });
    try {
      // 「chunkLength は 1 ちょうど」の検査と constructor 渡しの間には論理長確保の await がある。
      // `spec` から読み直す形だと、検査は 1 で通ったのに実行形が M=4 の借り手ができてしまう
      // （readonly attention は M 1 固定 — ADR 0096 段 2 §1.2 の形検査と対なので、この形は
      // 例外なしに崩れる）。
      const spec = { chunkLength: 1, borrow: lenderContext };
      const pending = borrower.createGenerationContext(spec);
      spec.chunkLength = 4;
      const borrowed = await pending;
      try {
        assertEquals(borrowed.chunkLength, 1, "検査を通った 1 のまま");
        assertEquals([...internals(borrowed).allowedRows], [1], "許可する物理形は decode 1 本だけ");
      } finally {
        await borrowed.dispose();
      }
    } finally {
      await lenderContext.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "external スロットの容量は計画鍵に載る（容量の違う貸し手を束ねた 2 本は別鍵・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(graphModelBuffer(lenderGraph())));
    const borrower = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph())));
    const small = await lender.createGenerationContext({ chunkLength: 4, bindings: { C: 16 } });
    const large = await lender.createGenerationContext({ chunkLength: 4, bindings: { C: 32 } });
    const borrowedSmall = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: small,
    });
    const borrowedLarge = await borrower.createGenerationContext({
      chunkLength: 1,
      borrow: large,
    });
    const draft = (context: GenerationContext): Promise<unknown> =>
      borrower.run(
        { q: tensor([1, HEADS, 1, DEPTH], seeded(HEADS * DEPTH, QUERY)) },
        {},
        { context, queryLength: 1 },
      );
    try {
      await draft(borrowedSmall);
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, false, "1 本目は導出 run");
      await draft(borrowedSmall);
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, true, "同じ容量なら同じ鍵");
      await draft(borrowedLarge);
      const stats = borrower.diagnostics().lastRunPrepared;
      assertEquals(stats?.hit, false, "容量が違えば別鍵（external の容量が鍵に載っていない）");
      assertEquals(stats?.cachedPlans, 2, "容量ごとに 1 本ずつ計画が載る");
      await draft(borrowedLarge);
      assertEquals(borrower.diagnostics().lastRunPrepared?.hit, true);
    } finally {
      await borrowedSmall.dispose();
      await borrowedLarge.dispose();
      await small.dispose();
      await large.dispose();
      await borrower.dispose();
      await lender.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 共有 initializer（§2.2）— 貸し手の i8 embedding 表を借り手が読む
// ---------------------------------------------------------------------------

const VOCAB = 8;
const HIDDEN = 4;

/** i8 の embedding 表 1 本（行ごとに違う scale — 行の取り違えが値に出る形）。 */
const embedTable = (): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly scale: Float32Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
} => {
  const quantized: number[] = [];
  for (let row = 0; row < VOCAB; row += 1) {
    for (let col = 0; col < HIDDEN; col += 1) quantized.push(((row * 7 + col * 13) % 251) - 125);
  }
  const bytes = i8BytesFrom(quantized);
  const scale = Float32Array.from({ length: VOCAB }, (_, row) => 0.1 + row * 0.003);
  return { bytes, scale, values: decodeI8(bytes, [VOCAB, HIDDEN], scale, [VOCAB, 1]) };
};

/** 貸し手（重みの持ち主）: i8 の embedding 表を 1 本だけ持つ最小グラフ。 */
const weightLenderGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { embed: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } } },
  values: {
    embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [{ op: "embedding", ins: ["embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
});

/** 借り手（バイトを持たない側）: 同じ表を `shared` 宣言で借りて読む。 */
const weightBorrowerGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { borrowed_embed: { shared: { tensor: "m.w" }, storage: { dtype: "i8" } } },
  values: {
    borrowed_embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [
    { op: "embedding", ins: ["borrowed_embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } },
  ],
});

const weightLenderModel = (table: ReturnType<typeof embedTable>): ArrayBuffer =>
  buildSafetensors([
    { name: "m.s", dtype: "F32", shape: [VOCAB, 1], data: f32Bytes(table.scale) },
    { name: "m.w", dtype: "I8", shape: [VOCAB, HIDDEN], data: table.bytes },
  ], { karume_ir: JSON.stringify(weightLenderGraph()) });

Deno.test({
  name: "共有 initializer: 借り手が貸し手の i8 表を読み、値がビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(weightLenderModel(table)));
    const borrower = await createSession(
      gpu,
      openModel(graphModelBuffer(weightBorrowerGraph())),
      { sharedWeights: { borrowed_embed: lender.exportWeight("embed") } },
    );
    try {
      const index = fill([VOCAB], (i) => VOCAB - 1 - i, "i32");
      const lent = await lender.run({ index });
      const borrowedOut = await borrower.run({ index });
      const a = lent["y"].data as Float32Array<ArrayBuffer>;
      const b = borrowedOut["y"].data as Float32Array<ArrayBuffer>;
      const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
      const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
      for (let i = 0; i < aBits.length; i += 1) {
        assertEquals(bBits[i], aBits[i], `要素 ${i}: 貸し手 ${a[i]} / 借り手 ${b[i]}`);
      }
      // 期待値そのもの（decodeI8 の行 gather）とも一致する = 表を丸ごと読めている。
      for (let row = 0; row < VOCAB; row += 1) {
        for (let col = 0; col < HIDDEN; col += 1) {
          const source = (VOCAB - 1 - row) * HIDDEN + col;
          assertEquals(b[row * HIDDEN + col], table.values[source], `行 ${row} 列 ${col}`);
        }
      }
      // 借り手は 1 バイトも常駐させない（バイトを持つのは貸し手だけ）。
      assertEquals(borrower.diagnostics().storage.residentCompressedBytes, 0);
      assertEquals(borrower.diagnostics().storage.hostExpandedBytes, 0);

      // 借り手が生きている間の貸し手 dispose は拒否される。
      const refused = await assertRejects(() => lender.dispose(), ExecutionError);
      assert(refused.message.includes("貸し出している"), refused.message);
      // 拒否された dispose は貸し手を壊さない。
      await lender.run({ index });
    } finally {
      await borrower.dispose();
      // 借り手が退いた後は通る。
      await lender.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "共有 initializer の門: 過不足・shape / dtype・席の不一致は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSession(gpu, openModel(weightLenderModel(table)));
    try {
      const shared = lender.exportWeight("embed");
      const build = (graph: GraphJson, weights?: Record<string, typeof shared>): Promise<Session> =>
        createSession(
          gpu,
          openModel(graphModelBuffer(graph)),
          weights === undefined ? {} : { sharedWeights: weights },
        );

      // ① 渡し忘れ（宣言 1 本に対して 0 本）
      const missing = await assertRejects(
        () => build(weightBorrowerGraph()),
        ExecutionError,
      );
      assert(missing.message.includes("不足 [borrowed_embed]"), missing.message);

      // ② 余剰（宣言に無い名前）
      const surplus = await assertRejects(
        () => build(weightBorrowerGraph(), { borrowed_embed: shared, unknown: shared }),
        ExecutionError,
      );
      assert(surplus.message.includes("余剰 [unknown]"), surplus.message);

      // ③ 宣言 shape の不一致
      const wrongShape = weightBorrowerGraph();
      wrongShape.values["borrowed_embed"].shape = [HIDDEN, VOCAB];
      wrongShape.values["y"].shape = [VOCAB, VOCAB];
      const shapeError = await assertRejects(
        () => build(wrongShape, { borrowed_embed: shared }),
        ExecutionError,
      );
      assert(shapeError.message.includes("宣言 shape"), shapeError.message);

      // ④ 格納 dtype の不一致（貸し手は i8）
      const wrongDtype = weightBorrowerGraph();
      wrongDtype.initializers["borrowed_embed"].storage = { dtype: "f32" };
      const dtypeError = await assertRejects(
        () => build(wrongDtype, { borrowed_embed: shared }),
        ExecutionError,
      );
      assert(dtypeError.message.includes("格納 dtype"), dtypeError.message);

      // ⑤ 消費席の不一致。借り手が同じ表を**重みスロット以外**（elementwise の被演算子）で
      // 食う形は圧縮常駐の適格外 = 席 `expanded` で、貸し手の i8 席と組めない。通すと
      // 「packed な i8 バイト列を f32 として読む」沈黙誤値そのものになる。
      const wrongSeat: GraphJson = {
        format: "karume-ir",
        version: 1,
        requires: { ops: ["add"] },
        symbols: [],
        inputs: [{ name: "x", dtype: "f32", shape: [VOCAB, HIDDEN] }],
        outputs: ["y"],
        initializers: { borrowed_embed: { shared: { tensor: "m.w" }, storage: { dtype: "i8" } } },
        values: {
          borrowed_embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
          y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
        },
        nodes: [{ op: "add", ins: ["x", "borrowed_embed"], outs: ["y"], attrs: {} }],
      };
      const seatError = await assertRejects(
        () => build(wrongSeat, { borrowed_embed: shared }),
        ExecutionError,
      );
      assert(seatError.message.includes("消費席が貸し手と互換でない"), seatError.message);

      // ⑥ 借り物の再輸出は拒否する（借用の連鎖は持たない）。
      const chained = await build(weightBorrowerGraph(), { borrowed_embed: shared });
      try {
        assertThrows(
          () => chained.exportWeight("borrowed_embed"),
          ExecutionError,
          "借り物の再輸出",
        );
      } finally {
        await chained.dispose();
      }
    } finally {
      await lender.dispose();
      gpu.destroy();
    }
  },
});
