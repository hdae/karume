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
//      shape / 意味論 dtype / 席 / 行の軸の不一致は fail loudly・借り手が生きている間の貸し手 dispose も拒否
//
// MUST: 参照に渡すスロットは**GPU から読み戻す**（ホスト側オラクルの再現ではない）。借り手が
// 読むのは貸し手が実際に書いた行なので、読み戻しでしか「同じ実体を見ている」ことを示せない。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertThrows,
  fail,
} from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAttentionReadonly } from "../src/reference/state-attention.ts";
import { decodeI8 } from "../src/format/i8.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import { LENGTHS_BYTES } from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
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
const lenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
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
const borrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
      const wide = await createSessionFromContainer(gpu, await openGraphModel(wideWindow), "model");
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
    const lender = await createSessionFromContainer(
      gpu,
      await openGraphModel(lenderGraph()),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(borrowerGraph()),
      "model",
    );
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
const weightLenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { embed: {} },
  values: {
    embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [{ op: "embedding", ins: ["embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
});

/** 借り手（バイトを持たない側）: 同じ表を `shared` 宣言で借りて読む。 */
const weightBorrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "index", dtype: "i32", shape: [VOCAB] }],
  outputs: ["y"],
  initializers: { embed: { shared: true } },
  values: {
    embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
    y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
  },
  nodes: [
    { op: "embedding", ins: ["embed", "index"], outs: ["y"], attrs: { padding_idx: -1 } },
  ],
});

/**
 * 行の軸の門の材料（貸し手）: conv1d の重みスロット = 行の軸 **0**（`[Cout,Cin,K]`）。
 * 値は見ない（門は席と軸だけを見る）ので scale は恒等・payload は定数で埋める。
 */
const convLenderGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 2, 6] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 2, 3] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 4] },
  },
  nodes: [{
    op: "conv1d",
    ins: ["x", "w", "b"],
    outs: ["y"],
    attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
  }],
});

/** 同じバイト列を conv_transpose1d（`[Cin,Cout,K]` = 行の軸 **1**）で食う借り手。 */
const convBorrowerGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv_transpose1d"] },
  symbols: [],
  inputs: [{ name: "z", dtype: "f32", shape: [1, 4, 6] }],
  outputs: ["u"],
  initializers: { w: { shared: true }, c: {} },
  values: {
    w: { dtype: "f32", shape: [4, 2, 3] },
    c: { dtype: "f32", shape: [2] },
    u: { dtype: "f32", shape: [1, 2, 8] },
  },
  nodes: [{
    op: "conv_transpose1d",
    ins: ["z", "w", "c"],
    outs: ["u"],
    attrs: { stride: 1, padding: 0 },
  }],
});

const openConvLenderModel = () =>
  openGraphModel(convLenderGraph(), [
    {
      graph: "model",
      initializer: "w",
      bytes: i8BytesFrom(new Array(24).fill(1)),
      // per-channel scale の行は消費側 op のチャネル軸（conv1d = 0）で、行長 6 = groupSize。
      encoding: {
        codec: "int8-sym",
        rowAxis: 0,
        groupSize: 6,
        scale: { bytes: f32Bytes(Float32Array.from([1, 1, 1, 1])), dtype: "f32" },
      },
    },
    {
      graph: "model",
      initializer: "b",
      bytes: f32Bytes(new Float32Array(4)),
      encoding: { codec: "f32" },
    },
  ]);

const openConvBorrowerModel = () =>
  openGraphModel(convBorrowerGraph(), [
    {
      graph: "model",
      initializer: "c",
      bytes: f32Bytes(new Float32Array(2)),
      encoding: { codec: "f32" },
    },
  ]);

const openWeightLenderModel = (table: ReturnType<typeof embedTable>) =>
  openGraphModel(weightLenderGraph(), [
    {
      graph: "model",
      initializer: "embed",
      bytes: table.bytes,
      // embedding のチャネル軸は 0・行長 HIDDEN = groupSize（per-channel）。
      encoding: {
        codec: "int8-sym",
        rowAxis: 0,
        groupSize: HIDDEN,
        scale: { bytes: f32Bytes(table.scale), dtype: "f32" },
      },
    },
  ]);

Deno.test({
  name: "共有 initializer: 借り手が貸し手の i8 表を読み、値がビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openWeightLenderModel(table),
      "model",
    );
    const borrower = await createSessionFromContainer(
      gpu,
      await openGraphModel(weightBorrowerGraph()),
      "model",
      { sharedWeights: { embed: lender.exportWeight("embed") } },
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
  name: "共有 initializer の門: 過不足・shape・席・行の軸の不一致は fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const table = embedTable();
    const gpu = await acquireGpu();
    const lender = await createSessionFromContainer(
      gpu,
      await openWeightLenderModel(table),
      "model",
    );
    try {
      const shared = lender.exportWeight("embed");
      const build = async (
        graph: DeclarationJson,
        weights?: Record<string, typeof shared>,
      ): Promise<Session> =>
        await createSessionFromContainer(
          gpu,
          await openGraphModel(graph),
          "model",
          weights === undefined ? {} : { sharedWeights: weights },
        );

      // ① 渡し忘れ（宣言 1 本に対して 0 本）
      const missing = await assertRejects(
        () => build(weightBorrowerGraph()),
        ExecutionError,
      );
      assert(missing.message.includes("不足 [embed]"), missing.message);

      // ② 余剰（宣言に無い名前）
      const surplus = await assertRejects(
        () => build(weightBorrowerGraph(), { embed: shared, unknown: shared }),
        ExecutionError,
      );
      assert(surplus.message.includes("余剰 [unknown]"), surplus.message);

      // ③ 宣言 shape の不一致
      const wrongShape = weightBorrowerGraph();
      wrongShape.values["embed"].shape = [HIDDEN, VOCAB];
      wrongShape.values["y"].shape = [VOCAB, VOCAB];
      const shapeError = await assertRejects(
        () => build(wrongShape, { embed: shared }),
        ExecutionError,
      );
      assert(shapeError.message.includes("宣言 shape"), shapeError.message);

      // ④ 行の軸の不一致。借り手は格納を宣言しない（期待する席は貸し手の codec から導く）ので、
      // 読み方の割れはこの軸が受け持つ: per-channel scale の軸は**消費側 op** から決まり、
      // 貸し手 conv1d（軸 0）と借り手 conv_transpose1d（軸 1）では同じバイト列に別の行の
      // scale が掛かる（例外は 1 つも出ない沈黙誤値）。
      const axisLender = await createSessionFromContainer(
        gpu,
        await openConvLenderModel(),
        "model",
      );
      try {
        const axisError = await assertRejects(
          async () =>
            await createSessionFromContainer(gpu, await openConvBorrowerModel(), "model", {
              sharedWeights: { w: axisLender.exportWeight("w") },
            }),
          ExecutionError,
        );
        assert(axisError.message.includes("行の軸"), axisError.message);
      } finally {
        await axisLender.dispose();
      }

      // ⑤ 消費席の不一致。借り手が同じ表を**重みスロット以外**（elementwise の被演算子）で
      // 食う形は圧縮常駐の適格外 = 席 `expanded` で、貸し手の i8 席と組めない。通すと
      // 「packed な i8 バイト列を f32 として読む」沈黙誤値そのものになる。
      const wrongSeat: DeclarationJson = {
        format: "karume-ir",
        version: 2,
        requires: { ops: ["add"] },
        symbols: [],
        inputs: [{ name: "x", dtype: "f32", shape: [VOCAB, HIDDEN] }],
        outputs: ["y"],
        initializers: { embed: { shared: true } },
        values: {
          embed: { dtype: "f32", shape: [VOCAB, HIDDEN] },
          y: { dtype: "f32", shape: [VOCAB, HIDDEN] },
        },
        nodes: [{ op: "add", ins: ["x", "embed"], outs: ["y"], attrs: {} }],
      };
      const seatError = await assertRejects(
        () => build(wrongSeat, { embed: shared }),
        ExecutionError,
      );
      assert(seatError.message.includes("消費席が貸し手と互換でない"), seatError.message);

      // ⑥ 借り物の再輸出は拒否する（借用の連鎖は持たない）。
      const chained = await build(weightBorrowerGraph(), { embed: shared });
      try {
        assertThrows(
          () => chained.exportWeight("embed"),
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

// ---------------------------------------------------------------------------
// 共有 initializer の門: 意味論 dtype（§2.2）
// ---------------------------------------------------------------------------

/** dtype の門の材料の要素数（値は見ない — 門は宣言だけを見る）。 */
const DTYPE_LENGTH = 4;

/** 貸し手: 意味論 dtype が `dtype` の initializer `w` を 1 本持ち、f32 へ cast して返す。 */
const dtypeLenderGraph = (dtype: "f32" | "i32"): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cast"] },
  symbols: [],
  inputs: [],
  outputs: ["y"],
  initializers: { w: {} },
  values: {
    w: { dtype, shape: [DTYPE_LENGTH] },
    y: { dtype: "f32", shape: [DTYPE_LENGTH] },
  },
  nodes: [{ op: "cast", ins: ["w"], outs: ["y"], attrs: { to: "f32" } }],
});

const openDtypeLenderModel = (dtype: "f32" | "i32") =>
  openGraphModel(dtypeLenderGraph(dtype), [{
    graph: "model",
    initializer: "w",
    bytes: dtype === "f32"
      ? f32Bytes([1, 2, -3, 0.5])
      : new Uint8Array(Int32Array.from([1, 2, -3, 5]).buffer),
    encoding: { codec: dtype },
  }]);

/**
 * 借り手: `w` を `shared` で借り（意味論 dtype は `dtype` と宣言）、自前の f32 initializer
 * `bias` も 1 本持つ。`bias` は「門が重みのアップロードより後ろに在れば借り手側の GPU 確保が
 * 必ず起きる」ことを保証する材料（確保ゼロの観測を空振りにしない）。
 *
 * bool は IR パーサが initializer の語彙外として落とすので、パース済みの宣言を書き換えて作る —
 * 公開面の `openMemoryContainer` は `IrDeclaration` を型付きの値で受け取り、パーサを通らない。
 */
const dtypeBorrowerContainer = (dtype: "f32" | "i32" | "bool") => {
  const parsed = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", "add"] },
    symbols: [],
    inputs: [],
    outputs: ["y"],
    initializers: { w: { shared: true }, bias: {} },
    values: {
      w: { dtype: dtype === "bool" ? "f32" : dtype, shape: [DTYPE_LENGTH] },
      bias: { dtype: "f32", shape: [DTYPE_LENGTH] },
      c: { dtype: "f32", shape: [DTYPE_LENGTH] },
      y: { dtype: "f32", shape: [DTYPE_LENGTH] },
    },
    nodes: [
      { op: "cast", ins: ["w"], outs: ["c"], attrs: { to: "f32" } },
      { op: "add", ins: ["c", "bias"], outs: ["y"], attrs: {} },
    ],
  });
  const declaration: IrDeclaration = dtype === "bool"
    ? { ...parsed, values: { ...parsed.values, w: { ...parsed.values["w"], dtype: "bool" } } }
    : parsed;
  return openMemoryContainer({
    graphs: { model: declaration },
    tensors: {
      model: {
        bias: { bytes: f32Bytes(new Float32Array(DTYPE_LENGTH)), encoding: { codec: "f32" } },
      },
    },
  });
};

/** `fn` の間に device が受けた `createBuffer` の回数を数え、`fn` の結果と返す（終わったら元へ戻す）。 */
const countCreateBuffer = async <T>(
  gpu: GpuContext,
  fn: () => Promise<T>,
): Promise<{ allocated: number; result: T }> => {
  const create = gpu.device.createBuffer.bind(gpu.device);
  let allocated = 0;
  gpu.device.createBuffer = (descriptor): GPUBuffer => {
    allocated += 1;
    return create(descriptor);
  };
  try {
    const result = await fn();
    return { allocated, result };
  } finally {
    gpu.device.createBuffer = create;
  }
};

Deno.test({
  name:
    "共有 initializer の門: 貸し手と借り手の意味論 dtype の不一致は、借り手の GPU 確保より前に fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // 貸し手の構築も try の内側に置き、作れたものだけをここに積んで finally で返す — 2 本目の構築が
    // 落ちても 1 本目と device の後始末を飛ばさない。
    const lenders: Session[] = [];
    let disposed: PromiseSettledResult<void>[] = [];
    try {
      const f32Lender = await createSessionFromContainer(
        gpu,
        await openDtypeLenderModel("f32"),
        "model",
      );
      lenders.push(f32Lender);
      const i32Lender = await createSessionFromContainer(
        gpu,
        await openDtypeLenderModel("i32"),
        "model",
      );
      lenders.push(i32Lender);
      const build = (lender: Session, dtype: "f32" | "i32" | "bool"): Promise<Session> =>
        createSessionFromContainer(gpu, dtypeBorrowerContainer(dtype), "model", {
          sharedWeights: { w: lender.exportWeight("w") },
        });
      const rejectsBeforeAllocation = async (
        lender: Session,
        dtype: "f32" | "i32" | "bool",
        lenderDtype: string,
      ): Promise<void> => {
        const { allocated, result: outcome } = await countCreateBuffer(
          gpu,
          () =>
            build(lender, dtype).then(
              (borrower) => ({ borrower }),
              (error: unknown) => ({ error }),
            ),
        );
        if ("borrower" in outcome) {
          // 借り手を先に返す — 掴んだままだと finally の貸し手 dispose が「貸し出し中」で拒否され、
          // 本来の失敗（門が効いていない）がその拒否に隠れる。fail を finally に置くのは、借り手の
          // dispose 自体が拒否されても、その拒否ではなく本来の失敗の文言を表に出すため。
          try {
            await outcome.borrower.dispose();
          } finally {
            fail(
              `借り手（${dtype}）の構築が貸し手（${lenderDtype}）の重みで通った — 意味論 dtype の門が効いていない`,
            );
          }
        }
        assertInstanceOf(outcome.error, ExecutionError);
        const message = outcome.error.message;
        assert(message.includes("意味論 dtype"), message);
        assert(message.includes(`'${dtype}'`) && message.includes(`'${lenderDtype}'`), message);
        // 仕様（借り手の重みのアップロード前）より強い条件として構築全体の確保 0 回を見ている。将来ここが
        // 赤になったら、門より前の確保の中身（重みか、重み以外か）を見て判断する。
        assertEquals(allocated, 0, "門は借り手の重みを 1 バイトも上げる前に落とす");
      };

      // 対照: dtype が一致すれば構築は通り、借り手は自前の `bias` を確保する（= 確保ゼロの観測は
      // 空振りではない）。
      const { allocated: matchedAllocations, result: matched } = await countCreateBuffer(
        gpu,
        () => build(f32Lender, "f32"),
      );
      await matched.dispose();
      assert(matchedAllocations > 0, `一致した借り手の確保回数 ${matchedAllocations}`);

      // ① 貸し手 f32・借り手 i32 — 通すと同じ 4 バイトが整数として読まれる（1.0 → 1065353216）。
      await rejectsBeforeAllocation(f32Lender, "i32", "f32");
      // ② 貸し手 i32・借り手 f32 — 逆向きの同じ取り違え。
      await rejectsBeforeAllocation(i32Lender, "f32", "i32");
      // ③ 借り手 bool — 常駐形式（raw）は同じなので席の門は区別しない。
      await rejectsBeforeAllocation(f32Lender, "bool", "f32");
    } finally {
      // 片方の dispose の拒否で残りの後始末（もう 1 本の貸し手と device）を飛ばさない。
      disposed = await Promise.allSettled(lenders.map((lender) => lender.dispose()));
      gpu.destroy();
    }
    // dispose の拒否は本体が通ったときだけここで出す（finally で投げると本体の失敗を上書きする）。
    for (const settled of disposed) {
      if (settled.status === "rejected") throw settled.reason;
    }
  },
});
