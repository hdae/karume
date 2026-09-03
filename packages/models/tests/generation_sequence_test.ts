// 1 会話ぶんの生成面（`src/generation/sequence.ts`）の挙動テスト。GPU も実資産も要らない。
//
// ここで縛るのは ADR 0083 決定 1〜5・10 の契約:
//
// - **多ターンで token を 1 個も落とさない**（`pendingToken` 連結 prefill — 決定 4）。落ちても
//   例外は出ず「直前 assistant の最後の 1 token が履歴から消える」だけなので、実 GPU の
//   parity 門では気づけない。EOS 停止後 / max-tokens 停止後 / `break` 中断後の **3 経路**を見る。
// - 可変状態は context と `pendingToken` の 2 つだけ（位置は `context.pastLength` から都度導出）。
//   観測口（`used` / `GenerationStop.tokens` / 容量例外の欄）も**同じ 2 つからの導出**であること。
// - 「generate 1 回ぶん」の直列化・`AbortSignal` の素通し・容量超過の専用型。
//
// Session と context は narrow interface（`GenerationSession` / `GenerationContextFace`）で
// 受けるので、fake は素の object 1 個で足りる。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type {
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  Session,
  SymbolBindings,
} from "@karume/runtime";
import {
  createGenerationProgram,
  type DerivedRunInputs,
  type GenerationGraph,
  type GenerationProgramSpec,
  type GenerationWiring,
} from "../src/generation/program.ts";
import {
  assertGenerationRequestValues,
  createGenerationSequence,
  GenerationCapacityError,
  type GenerationEvent,
  type GenerationRequest,
  type GenerationSession,
  type GenerationStop,
} from "../src/generation/sequence.ts";

const VOCAB = 16;
const IDS = "input_ids";
const LAST_ROW = "last_row";
const LOGITS = "logits";
const DERIVED = "per_layer_inputs";
const CHUNK_LENGTH = 4;

const graphOf = (): GenerationGraph => ({
  symbols: ["C", "M"],
  inputs: [
    { name: IDS, dtype: "i32", shape: [1, "M"] },
    { name: DERIVED, dtype: "f32", shape: [1, "M", 2] },
    { name: LAST_ROW, dtype: "i32", shape: [1] },
  ],
  outputs: [LOGITS],
  values: { [LOGITS]: { dtype: "f32", shape: [1, 1, VOCAB] } },
});

/**
 * 静的配線。**派生入力の席は fake が持つ**（既定は位置列を記録する実装）。
 *
 * 位置は run の入力ではなくなった（`position_ids` はグラフから消え、位置に依存するホスト入力は
 * 派生入力の席が受ける）ので、「どの run にどの位置が渡ったか」を見られる唯一の場所がここである。
 */
const programOf = (
  fake: FakeSession,
  override: Partial<GenerationProgramSpec> = {},
): GenerationWiring =>
  createGenerationProgram({
    graph: graphOf(),
    inputIds: IDS,
    lastRow: LAST_ROW,
    logits: LOGITS,
    chunkLength: CHUNK_LENGTH,
    maxPosition: 128,
    capacity: 64,
    vocabSize: VOCAB,
    stopTokens: [],
    capacitySymbol: "C",
    derivedInputs: fake.derivedInputs,
    ...override,
  });

/** run 1 回ぶんの記録（呼び出し列だけで step の形が全部読める粒度）。 */
type RunCall = {
  readonly ids: readonly number[];
  readonly idsShape: readonly number[];
  readonly positions: readonly number[];
  readonly lastRow: number;
  readonly lastRowShape: readonly number[];
  readonly queryLength: number;
  readonly pastBefore: number;
  readonly bindings: SymbolBindings | undefined;
  readonly extra: readonly string[];
  readonly sameContext: boolean;
};

const readRow = (
  inputs: RunInputs,
  name: string,
): { readonly shape: readonly number[]; readonly values: readonly number[] } => {
  if (!Object.hasOwn(inputs, name)) throw new Error(`fake: 入力 '${name}' が渡っていない`);
  const tensor = inputs[name];
  if (!("data" in tensor)) throw new Error(`fake: 入力 '${name}' がホストテンソルでない`);
  if (tensor.dtype !== "i32") throw new Error(`fake: 入力 '${name}' が i32 でない`);
  return { shape: tensor.shape, values: [...tensor.data] };
};

type FakeOptions = {
  /** run ごとに argmax が指すべき token id（call 番号で引く）。 */
  readonly tokens?: readonly number[];
  /** この回数目（0 始まり）の run を失敗させる。 */
  readonly failAt?: number;
};

type FakeSession = ReturnType<typeof fakeSession>;

const fakeSession = (options: FakeOptions = {}) => {
  const calls: RunCall[] = [];
  const specs: GenerationContextSpec[] = [];
  let pastLength = 0;
  let disposals = 0;
  /** 直前の `derive` が受けた位置列（run の記録へ合流させる — 位置は run の入力ではない）。 */
  let derivedPositions: readonly number[] = [];
  /** 既定の派生入力の席（`[1,M,2]` の f32 を返しつつ、渡った位置列を記録する）。 */
  const derivedInputs: DerivedRunInputs = {
    names: [DERIVED],
    derive: (ids, positions) => {
      if (ids.length !== positions.length) {
        throw new Error(`fake: ids ${ids.length} と positions ${positions.length} の長さが違う`);
      }
      derivedPositions = [...positions];
      return Promise.resolve(
        {
          [DERIVED]: {
            dtype: "f32",
            shape: [1, ids.length, 2],
            data: new Float32Array(ids.length * 2),
          },
        } satisfies RunInputs,
      );
    },
  };
  const context = {
    get pastLength(): number {
      return pastLength;
    },
    dispose: (): Promise<void> => {
      disposals += 1;
      return Promise.resolve();
    },
  };
  const session: GenerationSession<typeof context> = {
    createGenerationContext: (spec) => {
      specs.push(spec);
      return Promise.resolve(context);
    },
    // deno-lint-ignore require-await
    run: async (inputs, bindings, generation): Promise<RunOutputs> => {
      const call = calls.length;
      const ids = readRow(inputs, IDS);
      const lastRow = readRow(inputs, LAST_ROW);
      calls.push({
        ids: ids.values,
        idsShape: ids.shape,
        // この run の直前に `derive` が受けた位置列（run の入力には無い）。
        positions: derivedPositions,
        lastRow: lastRow.values[0],
        lastRowShape: lastRow.shape,
        queryLength: generation.queryLength,
        pastBefore: pastLength,
        bindings,
        extra: Object.keys(inputs).filter((name) => name !== IDS && name !== LAST_ROW),
        sameContext: generation.context === context,
      });
      if (options.failAt === call) throw new Error("run が落ちた");
      // 論理長の進行は run の成功で起きる（実 context と同じ順序）。
      pastLength += generation.queryLength;
      const id = options.tokens?.[call] ?? (call + 1) % VOCAB;
      const data = new Float32Array(VOCAB);
      data[id] = 10;
      return { [LOGITS]: { dtype: "f32", shape: [1, 1, VOCAB], data } };
    },
  };
  return {
    session,
    derivedInputs,
    calls,
    specs,
    disposals: (): number => disposals,
    pastLength: (): number => pastLength,
  };
};

/** イベントを全部汲む（`done` も一緒に返す）。 */
const drain = async (
  stream: ReturnType<Awaited<ReturnType<typeof createGenerationSequence>>["generate"]>,
): Promise<{ readonly events: GenerationEvent[]; readonly stop: GenerationStop }> => {
  const events: GenerationEvent[] = [];
  for await (const event of stream) events.push(event);
  return { events, stop: await stream.done };
};

const tokenIds = (events: readonly GenerationEvent[]): number[] =>
  events.filter((event) => event.kind === "token").map((event) => event.id);

Deno.test("GenerationSequence: prefill は固定長 chunk・pad 0・位置は絶対値、decode は 1 行", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const { events, stop } = await drain(
    sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }),
  );

  // 容量記号の束縛点は context 生成だけ（ADR 0066 追記 7）。
  assertEquals(fake.specs.length, 1);
  assertEquals(fake.specs[0].bindings, { C: 64 });
  assertEquals(fake.specs[0].chunkLength, CHUNK_LENGTH);
  assertEquals(fake.calls.map((call) => call.bindings), [undefined, undefined, undefined]);
  assertEquals(fake.calls.every((call) => call.sameContext), true);

  // prefill: 物理 4 行に有効 3 行 + pad 0・位置は絶対値・last_row は最終有効行。
  assertEquals(fake.calls[0].idsShape, [1, CHUNK_LENGTH]);
  assertEquals(fake.calls[0].ids, [1, 2, 3, 0]);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 0]);
  assertEquals(fake.calls[0].lastRow, 2);
  assertEquals(fake.calls[0].lastRowShape, [1]);
  assertEquals(fake.calls[0].queryLength, 3);

  // decode: 1 行固定・前 step の token をそのまま食う・位置は続き。
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[1].ids, [5]);
  assertEquals(fake.calls[1].positions, [3]);
  assertEquals(fake.calls[1].lastRow, 0);
  assertEquals(fake.calls[2].ids, [6]);
  assertEquals(fake.calls[2].positions, [4]);

  assertEquals(events, [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "token", id: 5, position: 3 },
    { kind: "token", id: 6, position: 4 },
    { kind: "token", id: 7, position: 5 },
  ]);
  assertEquals(stop, { reason: "max-tokens", tokens: 3 });
});

Deno.test("GenerationSequence: 長い prompt は chunk ごとに prefill イベントを出す", async () => {
  const fake = fakeSession({ tokens: [1, 2, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const { events } = await drain(
    sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 1 }),
  );

  assertEquals(fake.calls.length, 2);
  assertEquals(fake.calls[0].ids, [1, 2, 3, 4]);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 3]);
  assertEquals(fake.calls[1].ids, [5, 6, 0, 0]);
  assertEquals(fake.calls[1].positions, [4, 5, 0, 0]);
  assertEquals(fake.calls[1].lastRow, 1);
  // 生成の起点は**最終 chunk の最終有効行**（途中 chunk の出力は捨てる）。
  assertEquals(events, [
    { kind: "prefill", chunk: 1, chunks: 2 },
    { kind: "prefill", chunk: 2, chunks: 2 },
    { kind: "token", id: 2, position: 6 },
  ]);
});

// ---- 多ターン: 「直前 assistant の最後の token が落ちない」直接門（3 経路） ----

Deno.test("多ターン: max-tokens 停止後の pendingToken が次ターン prompt の先頭に連結される", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 9] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const first = await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }));
  assertEquals(tokenIds(first.events), [5, 6, 7]);
  // K token 生成後の pastLength は T + K − 1（最後の 7 は未 commit の frontier）。
  assertEquals(fake.pastLength(), 5);

  await drain(sequence.generate({ prompt: [8, 9], maxNewTokens: 1 }));
  // 直前の 7 が先頭へ連結され、位置も途切れない（落とすと会話から 1 token 消える）。
  assertEquals(fake.calls[3].ids, [7, 8, 9, 0]);
  assertEquals(fake.calls[3].positions, [5, 6, 7, 0]);
  assertEquals(fake.calls[3].queryLength, 3);
  assertEquals(fake.calls[3].pastBefore, 5);
});

Deno.test("多ターン: EOS 停止でも停止 token は会話に残る（イベントには出さない）", async () => {
  const fake = fakeSession({ tokens: [5, 11, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { stopTokens: [11] }),
  });
  const first = await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 4 }));

  // 停止 token 自体は本文でなく終端記号なので `token` イベントに出さない。
  assertEquals(tokenIds(first.events), [5]);
  // 停止 token 自体も 1 個として数える（抽選 1 回 = run 1 回 — `GenerationStop.tokens` の doc）。
  assertEquals(first.stop, { reason: "eos", token: 11, tokens: 2 });
  // 停止で decode を打ち切る（maxNewTokens まで回さない）。
  assertEquals(fake.calls.length, 2);

  await drain(sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  // 停止 token は次ターンの prefill 先頭へ（chat の `<turn|>` を落とすと会話が壊れる）。
  assertEquals(fake.calls[2].ids, [11, 4, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

Deno.test("要求の stopTokens: 配布形の集合が空でも、そのターンだけ停止 token を足せる", async () => {
  const fake = fakeSession({ tokens: [5, 11, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { stopTokens: [] }),
  });
  const first = await drain(
    sequence.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [11] }),
  );

  // 停止 token 自体は本文でないので `token` イベントに出さない（EOS 停止と同じ扱い）。
  assertEquals(tokenIds(first.events), [5]);
  // 理由だけが EOS と別（配布形の終端記号ではなく、この要求の都合で止めた）。
  assertEquals(first.stop, { reason: "stop-token", token: 11, tokens: 2 });
  assertEquals(fake.calls.length, 2, "停止で decode を打ち切る");

  // 会話には残る = 次ターンの prefill 先頭へ連結される（EOS 停止後と同じ後始末）。
  await drain(sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  assertEquals(fake.calls[2].ids, [11, 4, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

Deno.test("要求の stopTokens: 配布形の EOS 集合との和集合で判定する（EOS は常に効く）", async () => {
  // 要求が停止集合を渡しても、配布形の EOS を上書きはしない（両方が効く = 和集合）。
  const eosFake = fakeSession({ tokens: [5, 11, 3] });
  const eos = await createGenerationSequence({
    session: eosFake.session,
    program: programOf(eosFake, { stopTokens: [11] }),
  });
  assertEquals(
    (await drain(eos.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [9] }))).stop,
    { reason: "eos", token: 11, tokens: 2 },
    "配布形の EOS で止まったターン",
  );

  // 両方に居る id は `eos` で閉じる（配布形の終端記号としての意味が優先する）。
  const bothFake = fakeSession({ tokens: [5, 11, 3] });
  const both = await createGenerationSequence({
    session: bothFake.session,
    program: programOf(bothFake, { stopTokens: [11] }),
  });
  assertEquals(
    (await drain(both.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [11] }))).stop,
    { reason: "eos", token: 11, tokens: 2 },
    "両方の集合に居る停止 token",
  );
});

Deno.test("要求の stopTokens: 語彙外・重複は同期に落ちる（効かない停止条件を残さない）", async () => {
  // 停止 token は「出力に現れない id」なので、間違っていても生成は普通に完走する（その id が
  // 抽選されないだけ）— 出力が伸び続けることでしか気づけないので、入口で落とす。
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [VOCAB] }),
    Error,
    `stopTokens[0] ${VOCAB} が語彙 0..${VOCAB - 1} の外`,
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [1, -1] }),
    Error,
    "stopTokens[1] -1",
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [3, 5, 3] }),
    Error,
    "stopTokens に token 3 が 2 度出る",
  );
  assertEquals(fake.calls.length, 0);
});

Deno.test("要求の stopTokens: 発行後に配列へ足しても走行中の生成には効かない", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  // 呼び手が握ったままの可変な停止集合（発行時は空 = 止まらない指定）。
  const stopTokens: number[] = [];
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3, stopTokens });
  // 本体は最初の `next()` まで走らないので、汲む前がいちばん広い書き換えの窓である。
  stopTokens.push(6);

  const { events, stop } = await drain(stream);
  // 後付けが効いていれば 6 で止まって [5] になる。
  assertEquals(tokenIds(events), [5, 6, 7]);
  assertEquals(stop, { reason: "max-tokens", tokens: 3 });
});

Deno.test("多ターン: break 中断後も pendingToken が残り、次ターンが 1 token も落とさない", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8, 9, 10] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });

  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 5 });
  const seen: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    seen.push(event.id);
    // 2 個目を受け取ったところで打ち切る（`return()` 経由で finally へ入る経路）。
    if (seen.length === 2) break;
  }
  assertEquals(seen, [5, 6]);
  assertEquals(await stream.done, { reason: "closed", tokens: 2 });
  const callsAtBreak = fake.calls.length;
  assertEquals(callsAtBreak, 2);
  assertEquals(fake.pastLength(), 3);

  await drain(sequence.generate({ prompt: [3], maxNewTokens: 1 }));
  assertEquals(fake.calls[2].ids, [6, 3, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

Deno.test("多ターン: 続きだけのターン（prompt 空）は pendingToken を decode 形で流す", async () => {
  // 有効行 1 本の chunk は物理 1 行 = decode 形。中断からの再開が「中断しなかった走り」と
  // **同じ形の run** になり、prefill 形（M=chunkLength）を余分に踏まない。
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 5 });
  for await (const event of stream) {
    if (event.kind === "token") break;
  }
  assertEquals(fake.calls.length, 1);

  await drain(sequence.generate({ prompt: [], maxNewTokens: 2 }));
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[1].ids, [5]);
  assertEquals(fake.calls[1].positions, [2]);
  assertEquals(fake.calls[1].lastRow, 0);
  assertEquals(fake.calls[1].queryLength, 1);
});

Deno.test("GenerationSequence: prompt が空で pendingToken も無ければ fail loudly", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await assertRejects(
    () => drain(sequence.generate({ prompt: [], maxNewTokens: 1 })),
    Error,
    "prompt が空",
  );
  assertEquals(fake.calls.length, 0);
});

// ---- 中断（AbortSignal）と直列化 ----

Deno.test("GenerationSequence: abort は signal.reason を包まずそのまま throw する", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const controller = new AbortController();
  const reason = new Error("呼び手が止めた");
  const stream = sequence.generate({
    prompt: [1, 2],
    maxNewTokens: 5,
    signal: controller.signal,
  });

  let caught: unknown;
  const seen: number[] = [];
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  // 包まない（消費側が `error === controller.signal.reason` で自分の中断を識別できる）。
  assert(caught === reason, `中断の例外が包まれている: ${String(caught)}`);
  assertEquals(seen, [5]);
  assertEquals(await stream.done, { reason: "aborted", tokens: 1 });
  // 中断は段の境目（次の run の直前）で効く — 走行中の run を殺しはしない。
  assertEquals(fake.calls.length, 1);

  // 中断でも会話は「成功した run のぶんだけ」進み、frontier は残る。
  assertEquals(fake.pastLength(), 2);
  await drain(sequence.generate({ prompt: [3], maxNewTokens: 1 }));
  assertEquals(fake.calls[1].ids, [5, 3, 0, 0]);
});

Deno.test("GenerationSequence: 発行済みの signal は 1 本も run を出さずに落ちる", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const controller = new AbortController();
  const reason = new Error("最初から中断");
  controller.abort(reason);
  const stream = sequence.generate({
    prompt: [1, 2],
    maxNewTokens: 2,
    signal: controller.signal,
  });
  let caught: unknown;
  try {
    for await (const _event of stream) { /* 1 個も来ない */ }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason);
  assertEquals(fake.calls.length, 0);
  assertEquals(await stream.done, { reason: "aborted", tokens: 0 });
});

Deno.test("GenerationSequence: 順番待ちの間に届いた中断は run を 1 本も出さずに閉じる", async () => {
  // 待っている間に abort されたリクエストは、順番が回ってきても**何も判定せずに**閉じる。
  // 予算超過を先に返すと「止めたのに容量エラーが出た」になり、呼び手は自分の中断を識別できない。
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });

  const first = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  const iterator = first[Symbol.asyncIterator]();
  await iterator.next();

  const controller = new AbortController();
  const reason = new Error("順番待ちの間に止めた");
  // past 4 + prompt(連結 1 + 6) + K 4 − 1 = 14 > 容量 8 = 自分の番が来れば必ず容量超過。
  const queued = sequence.generate({
    prompt: [1, 2, 3, 4, 5, 6],
    maxNewTokens: 4,
    signal: controller.signal,
  });
  const consumed = (async () => {
    for await (const _event of queued) { /* 1 個も来ない */ }
  })();
  // 順番待ちに入らせてから中断する。
  await new Promise((resolve) => setTimeout(resolve, 0));
  controller.abort(reason);

  while (!(await iterator.next()).done) { /* 席を空ける */ }
  const callsAfterFirst = fake.calls.length;
  let caught: unknown;
  try {
    await consumed;
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `中断ではなく別の失敗で閉じた: ${String(caught)}`);
  assertEquals(await queued.done, { reason: "aborted", tokens: 0 });
  assertEquals(fake.calls.length, callsAfterFirst);
});

Deno.test("GenerationSequence: 2 本の generate は直列化される（run が混ざらない）", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8, 9, 10] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });

  const log: string[] = [];
  const consume = async (label: string, prompt: readonly number[]): Promise<void> => {
    for await (const event of sequence.generate({ prompt, maxNewTokens: 2 })) {
      log.push(`${label}:${event.kind}`);
    }
  };
  await Promise.all([consume("a", [1, 2]), consume("b", [3])]);

  // 片方の全イベントがもう片方より前に並ぶ（混ざったら鎖が効いていない）。
  const boundary = log.findIndex((entry) => entry.startsWith("b:"));
  assert(boundary > 0, `直列化されていない: ${log.join(" ")}`);
  assertEquals(log.slice(0, boundary).every((entry) => entry.startsWith("a:")), true);
  assertEquals(log.slice(boundary).every((entry) => entry.startsWith("b:")), true);
  // b は a の frontier（6）を連結した prompt で始まる = 順番待ちの間に状態が進んでいる。
  assertEquals(fake.calls[2].ids, [6, 3, 0, 0]);
});

// ---- 容量（ADR 0083 決定 10 / 可変 capacity 波の実行時ノブ） ----

Deno.test("capacity ノブ: 既定は program の宣言・渡した値が context の束縛になる", async () => {
  const declared = fakeSession();
  const byDefault = await createGenerationSequence({
    session: declared.session,
    program: programOf(declared),
  });
  assertEquals(byDefault.capacity, 64, "省略時は配布形の既定");
  assertEquals(declared.specs[0].bindings, { C: 64 }, "容量記号の束縛も既定");

  const chosen = fakeSession();
  const narrowed = await createGenerationSequence({
    session: chosen.session,
    program: programOf(chosen),
    capacity: 16,
  });
  // 束縛点は context 生成だけ（ADR 0066 追記 7）— state スロットの物理確保がこの値で決まる。
  assertEquals(narrowed.capacity, 16);
  assertEquals(chosen.specs[0].bindings, { C: 16 });
  assertEquals(chosen.specs[0].chunkLength, CHUNK_LENGTH, "chunk 長は program のまま");
});

Deno.test("capacity ノブ: 予算検査は program の既定ではなく選んだ容量で行う", async () => {
  const fake = fakeSession({ tokens: [5] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
    capacity: 8,
  });
  // peak = 0 + 6 + 4 − 1 = 9 > 8（program の既定 64 なら通ってしまう形）。
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 4 })),
    GenerationCapacityError,
    "state 容量を超える",
  );
  assertEquals(error.limit, 8, "例外が運ぶ上限も選んだ容量");
  assertEquals(fake.calls.length, 0);
});

Deno.test("capacity ノブ: 関係を破る値は sequence を作る時点で fail loudly", async () => {
  const fake = fakeSession();
  // 1 chunk すら入らない容量（run を 1 本も出せない context を作らせない）。
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: CHUNK_LENGTH - 1,
      }),
    Error,
    `capacity ${CHUNK_LENGTH - 1} が chunkLength ${CHUNK_LENGTH} を下回る`,
  );
  // モデルの位置上限の外まで容量を取る形（容量いっぱいの会話が学習外の位置を踏む）。
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: 129,
      }),
    Error,
    "capacity 129 が maxPosition 128 を超えた",
  );
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: 8.5,
      }),
    Error,
    "capacity 8.5 が 1 以上の整数でない",
  );
  assertEquals(fake.specs.length, 0, "context を 1 本も確保していない");
});

Deno.test("GenerationSequence: 容量超過は GenerationCapacityError（run の前に落ちる）", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });
  // peak = past 0 + prompt 6 + maxNewTokens 4 − 1 = 9 > 8。
  await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 4 })),
    GenerationCapacityError,
    "state 容量を超える",
  );
  assertEquals(fake.calls.length, 0);
  // ちょうど収まる形は通る（peak = 8）。
  await drain(sequence.generate({ prompt: [1, 2, 3, 4, 5], maxNewTokens: 4 }));
  assertEquals(fake.pastLength(), 8);
});

Deno.test("GenerationSequence: used は pastLength + 未 commit frontier から都度導出する", async () => {
  // 独立 counter を持たない（ADR 0066 決定 6 の二重簿記の禁止）ので、走行中に読めば
  // 「その時点で成功済みの run まで」が出る。切り詰めの判断は生成の合間に読んだ値で行う。
  const fake = fakeSession({ tokens: [5, 6, 7, 9] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertEquals(sequence.used, 0, "生成前");

  const stream = sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next(); // prefill イベント（frontier は KV に入り、pendingToken は空）。
  assertEquals(sequence.used, 3, "prefill 直後");
  await iterator.next(); // 最初の token（未 commit の frontier が 1 つ立つ）。
  assertEquals(sequence.used, 4, "1 token 目の直後");
  while (!(await iterator.next()).done) { /* 汲み切る */ }

  // pastLength は T + K − 1（最後の 1 個は未 commit）だが、会話が占めているのは T + K。
  assertEquals(fake.pastLength(), 5);
  assertEquals(sequence.used, 6, "1 ターン目の後");

  // 次ターンは used を起点に積み上がる（連結される pendingToken を二重に数えない）。
  await drain(sequence.generate({ prompt: [8, 9], maxNewTokens: 1 }));
  assertEquals(sequence.used, 9, "2 ターン目の後（6 + 新規 2 + 生成 1）");
});

Deno.test("GenerationCapacityError: 切り詰めに要る実値を欄で運ぶ（文言を読み解かせない）", async () => {
  const fake = fakeSession({ tokens: [5] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });
  await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 }));
  assertEquals(sequence.used, 3);

  // past 2 + prompt(連結 1 + 4) 5 + K 4 − 1 = 10 > 容量 8。
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: 4 })),
    GenerationCapacityError,
    "state 容量を超える",
  );
  assertEquals(error.constraint, "capacity");
  assertEquals(error.pastLength, 2);
  assertEquals(error.promptLength, 5, "pendingToken の連結後の実効長（呼び手の 4 ではない）");
  assertEquals(error.requestedNewTokens, 4);
  assertEquals(error.limit, 8);
  assertEquals(error.maxNewTokens, 2);
  // 欄どおりに縮めれば本当に通る（この 1 本が「導ける」を恒真でなくする）。
  await drain(
    sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: error.maxNewTokens }),
  );
});

Deno.test("GenerationCapacityError: 位置表の上限も同じ欄で運ぶ（constraint だけが違う）", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { maxPosition: 5, capacity: 5 }),
  });
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 4 })),
    GenerationCapacityError,
    "位置表の外",
  );
  assertEquals(error.constraint, "maxPosition");
  assertEquals(error.limit, 5);
  assertEquals(error.pastLength, 0);
  assertEquals(error.promptLength, 3);
  assertEquals(error.requestedNewTokens, 4);
  // 排他上限だが式は容量側と同じ形に畳める（5 − 0 − 3 + 1 = 3）。
  assertEquals(error.maxNewTokens, 3);
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: error.maxNewTokens }));
});

Deno.test("GenerationSequence: 位置表の上限超過も同じ型で落とす", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { maxPosition: 5, capacity: 5 }),
  });
  // 最終位置 = 0 + 3 + 4 − 2 = 5（排他的上限 5 の外）。
  await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 4 })),
    GenerationCapacityError,
    "位置表の外",
  );
  assertEquals(fake.calls.length, 0);
  // 最終位置 4 はぎりぎり適法。
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }));
  assertEquals(fake.calls.length, 3);
});

Deno.test("GenerationSequence: 容量は自分の順番が来てから見る（先行ターンの進行を含める）", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 10 }),
  });
  await drain(sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: 4 }));
  assertEquals(fake.pastLength(), 7);
  // past 7 + prompt(1 連結 + 2) 3 + K 2 − 1 = 11 > 10。発行時点では past を知らない。
  await assertRejects(
    () => drain(sequence.generate({ prompt: [5, 6], maxNewTokens: 2 })),
    GenerationCapacityError,
    "既存 7",
  );
});

// ---- ホスト由来の per-chunk 入力（PLE の席） ----

Deno.test("DerivedRunInputs: pad 行込みの id 列と位置列が渡り、入力として結線される", async () => {
  const fake = fakeSession({ tokens: [5, 6] });
  const seen: { readonly ids: readonly number[]; readonly positions: readonly number[] }[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, {
      derivedInputs: {
        names: [DERIVED],
        derive: (ids, positions) => {
          seen.push({ ids: [...ids], positions: [...positions] });
          return Promise.resolve(
            {
              [DERIVED]: {
                dtype: "f32",
                shape: [1, ids.length, 2],
                data: new Float32Array(ids.length * 2),
              },
            } satisfies RunInputs,
          );
        },
      },
    }),
  });
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 2 }));

  // prefill は物理行数ぶん（pad 行にも `input_ids` と同じ 0 が入る）・decode は 1 行。位置も
  // 同じ規約（pad 行は 0）で、絶対位置が prompt の続きから 1 ずつ進む。
  assertEquals(seen, [
    { ids: [1, 2, 3, 0], positions: [0, 1, 2, 0] },
    { ids: [5], positions: [3] },
  ]);
  assertEquals(fake.calls.map((call) => call.extra), [[DERIVED], [DERIVED]]);
});

Deno.test("DerivedRunInputs: 生成の signal が derive まで降りる（best-effort の材料）", async () => {
  // 派生入力の材料は配布形によっては GB 級の遅延ロード（gemma4 の PLE sidecar）。降ろさないと
  // 「停止を押しても shard を読み終わるまで返らない」区間ができる。
  const fake = fakeSession({ tokens: [5, 6] });
  const seen: (AbortSignal | undefined)[] = [];
  const controller = new AbortController();
  const derived = (): GenerationWiring["derivedInputs"] => ({
    names: [DERIVED],
    derive: (ids, _positions, options) => {
      seen.push(options?.signal);
      return Promise.resolve(
        {
          [DERIVED]: {
            dtype: "f32",
            shape: [1, ids.length, 2],
            data: new Float32Array(ids.length * 2),
          },
        } satisfies RunInputs,
      );
    },
  });

  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { derivedInputs: derived() }),
  });
  await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 2, signal: controller.signal }));
  assertEquals(seen.length, 2);
  assert(seen.every((signal) => signal === controller.signal), "derive に別の signal が渡っている");

  // 購読していない要求には捏造しない。
  const bareFake = fakeSession();
  const bare = await createGenerationSequence({
    session: bareFake.session,
    program: programOf(bareFake, { derivedInputs: derived() }),
  });
  await drain(bare.generate({ prompt: [1], maxNewTokens: 1 }));
  assertEquals(seen[2], undefined);
});

Deno.test("DerivedRunInputs: derive の await 明けに届いた中断は run を出さない", async () => {
  // 中断の観測点が「ループ先頭」だけだと、派生入力を作っている間に届いた中断の後に run が
  // 1 本まるごと進む（prefill 先頭は常に cold miss なので、送信直後の停止で必ず踏む）。
  for (const [name, maxNewTokens, tokensBefore] of [["prefill", 2, 0], ["decode", 3, 1]] as const) {
    const fake = fakeSession({ tokens: [5, 6, 7] });
    const controller = new AbortController();
    const reason = new Error(`${name} の派生入力の途中で止めた`);
    let derives = 0;
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, {
        derivedInputs: {
          names: [DERIVED],
          derive: async (ids) => {
            derives += 1;
            // この derive の「読み」の最中に中断が届く形。
            if (derives > tokensBefore) {
              controller.abort(reason);
              await new Promise((resolve) => setTimeout(resolve, 0));
            }
            return {
              [DERIVED]: {
                dtype: "f32",
                shape: [1, ids.length, 2],
                data: new Float32Array(ids.length * 2),
              },
            } satisfies RunInputs;
          },
        },
      }),
    });

    const stream = sequence.generate({
      prompt: [1, 2],
      maxNewTokens,
      signal: controller.signal,
    });
    const events: GenerationEvent[] = [];
    let caught: unknown;
    try {
      for await (const event of stream) events.push(event);
    } catch (error) {
      caught = error;
    }
    assert(caught === reason, `${name}: 中断が包まれている（${String(caught)}）`);
    assertEquals(await stream.done, { reason: "aborted", tokens: tokensBefore }, name);
    // 中断が届いた後の run は 1 本も出ていない = token も 1 個も余分に届いていない。
    assertEquals(fake.calls.length, tokensBefore, `${name}: 中断後に run が進んだ`);
    assertEquals(tokenIds(events).length, tokensBefore === 0 ? 0 : 1, `${name}: 余分な token`);
  }
});

Deno.test("DerivedRunInputs: 宣言と違うキーを返したら fail loudly", async () => {
  const cases: readonly (readonly [string, RunInputs, string])[] = [
    ["欠け", {}, "欠け: per_layer_inputs"],
    [
      "余り",
      {
        [DERIVED]: { dtype: "f32", shape: [1, 4, 2], data: new Float32Array(8) },
        other: { dtype: "f32", shape: [1], data: new Float32Array(1) },
      },
      "余り: other",
    ],
  ];
  for (const [name, result, message] of cases) {
    const fake = fakeSession();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, {
        derivedInputs: { names: [DERIVED], derive: () => Promise.resolve(result) },
      }),
    });
    await assertRejects(
      () => drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 })),
      Error,
      message,
      name,
    );
  }
});

// ---- 受理集合・失敗・寿命 ----

Deno.test("GenerationSequence: 受理集合は同期に落ちる（順番待ちにも GPU にも入らない）", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 0 }),
    Error,
    "maxNewTokens 0",
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1.5 }),
    Error,
    "maxNewTokens 1.5",
  );
  // `Int32Array` の wrap も語彙外も黙って別 token に化ける（後者は範囲外 gather = NaN 汚染）。
  assertThrows(
    () => sequence.generate({ prompt: [1, VOCAB], maxNewTokens: 1 }),
    Error,
    `prompt[1] ${VOCAB} が語彙 0..${VOCAB - 1} の外`,
  );
  assertThrows(
    () => sequence.generate({ prompt: [4294967297], maxNewTokens: 1 }),
    Error,
    "prompt[0] 4294967297",
  );
  // sampler の指定も発行時に落とす（抽選は decode の途中で走るので、遅らせると深く潜る）。
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, sampler: { temperature: -1 } }),
    RangeError,
    "temperature -1",
  );
  assertEquals(fake.calls.length, 0);
});

Deno.test("受理集合の値域検査は 1 本の純関数（高レベル面と generate が同じ式を見る）", async (t) => {
  // 高レベル面（`Gemma4Pipeline.chat` / `Gemma4ChatSession.send`）は `generate` を async
  // generator の本体で呼ぶので、発行時に落とすにはこの関数を自分で呼ぶしかない。式を写して
  // 2 本持つと片方だけが古くなるので、正本が 1 本であること自体をここで縛る。
  await t.step("受理される要求は通る（門が恒真に落ちていないことの対）", () => {
    assertGenerationRequestValues(VOCAB, { maxNewTokens: 1 });
    assertGenerationRequestValues(VOCAB, {
      maxNewTokens: 8,
      stopTokens: [0, VOCAB - 1],
      sampler: { temperature: 1, topK: 4 },
    });
  });

  await t.step("maxNewTokens の値域", () => {
    assertThrows(() => assertGenerationRequestValues(VOCAB, { maxNewTokens: 0 }), Error, "0 が 1");
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1.5 }),
      Error,
      "maxNewTokens 1.5",
    );
  });

  await t.step("stopTokens の語彙外と重複", () => {
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1, stopTokens: [VOCAB] }),
      Error,
      `stopTokens[0] ${VOCAB} が語彙 0..${VOCAB - 1} の外`,
    );
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1, stopTokens: [2, 2] }),
      Error,
      "token 2 が 2 度出る",
    );
  });

  await t.step("sampler の指定（抽選は decode の途中で走るので発行時に落とす）", () => {
    assertThrows(
      () =>
        assertGenerationRequestValues(VOCAB, {
          maxNewTokens: 1,
          sampler: { temperature: -1 },
        }),
      RangeError,
      "temperature -1",
    );
  });

  await t.step("generate は同じ関数を通る（片方だけ緩む形を作らない）", async () => {
    // 故障注入: 関数が受理する値は generate も受理し、拒む値は generate も拒む。
    const fake = fakeSession();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake),
    });
    const request = { prompt: [1], maxNewTokens: 1, stopTokens: [2, 2] };
    assertThrows(() => assertGenerationRequestValues(VOCAB, request), Error, "2 度出る");
    assertThrows(() => sequence.generate(request), Error, "2 度出る");
    assertEquals(fake.calls.length, 0);
  });
});

/** 発行後に書き換えるための可変版（公開型は全欄 readonly）。 */
type MutableRequest = { -readonly [Field in keyof GenerationRequest]: GenerationRequest[Field] };

Deno.test("GenerationSequence: 要求は発行時に写す（発行後の書き換えは走行中の生成に効かない）", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  // 呼び手が握ったままの可変な要求（配列も sampler の bias も後から触れる形）。
  const prompt = [1, 2];
  const bias: [number, number][] = [];
  const request: MutableRequest = { prompt, maxNewTokens: 2, sampler: { logitBias: bias } };

  const stream = sequence.generate(request);
  // 本体は最初の `next()` まで走らないので、汲む前がいちばん広い書き換えの窓である。
  prompt.push(VOCAB + 5); // 受理集合の検査を通った後で語彙外 id を足す
  request.prompt = [3, 4]; // 別の配列へ差し替える
  request.maxNewTokens = 4; // 走行中に上限を伸ばす
  bias.push([6, Number.NEGATIVE_INFINITY]); // 2 個目に出るはずの token を禁止する
  const controller = new AbortController();
  request.signal = controller.signal; // 発行時に無かった中断を後から挿す
  controller.abort();

  const { events, stop } = await drain(stream);
  // 流れたのは発行時の prompt（長さ 2 = 有効 2 行 + pad 2 行）。
  assertEquals(fake.calls.length, 2);
  assertEquals(fake.calls[0].ids, [1, 2, 0, 0]);
  // token も停止も発行時の指定どおり（bias の後付けが効けば 6 は別 id に化ける）。
  assertEquals(tokenIds(events), [5, 6]);
  assertEquals(stop, { reason: "max-tokens", tokens: 2 });
});

Deno.test("GenerationSequence: run の失敗は iterable と done の両方へ同じ例外で届く", async () => {
  const fake = fakeSession({ failAt: 1 });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  await assertRejects(() => drain(stream), Error, "run が落ちた");
  await assertRejects(() => stream.done, Error, "run が落ちた");
});

Deno.test("GenerationSequence: done を読まない失敗が unhandled rejection にならない", async () => {
  const fake = fakeSession({ failAt: 0 });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 2 });
  await assertRejects(
    async () => {
      for await (const _event of stream) { /* 例外だけ受ける */ }
    },
    Error,
    "run が落ちた",
  );
  // `done` は握らない。ここでプロセスが落ちなければ、内部で 1 度握られている。
  await new Promise((resolve) => setTimeout(resolve, 0));
});

Deno.test("GenerationSequence: dispose は走行中の生成の後に走り、2 度目も同じ完了を返す", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next();

  const disposal = sequence.dispose();
  assertEquals(fake.disposals(), 0, "走行中の生成を追い越して dispose した");
  // 残りを汲み切ると席が空き、dispose が走る。
  while (!(await iterator.next()).done) { /* 汲み切る */ }
  await disposal;
  assertEquals(fake.disposals(), 1);
  await sequence.dispose();
  assertEquals(fake.disposals(), 1);
});

Deno.test("GenerationSequence: dispose 済みの generate は同期に落ちる（自分の文言で）", async () => {
  // context は外へ出さない（決定 3）ので、呼び手が寿命を確かめる術はこの検査しかない。遅らせると
  // 初反復まで落ちず、しかも `GenerationContext` 側の汎用文言になって真因が読めない。
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await sequence.dispose();
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1 }),
    Error,
    "GenerationSequence: dispose 済み",
  );
  assertEquals(fake.calls.length, 0);
  // dispose 発行直後（完了前）も同じ — `disposal` が立った時点で新しい生成は受けない。
  const running = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const disposal = running.dispose();
  assertThrows(
    () => running.generate({ prompt: [1], maxNewTokens: 1 }),
    Error,
    "GenerationSequence: dispose 済み",
  );
  await disposal;
});

Deno.test("GenerationSession: 実 Session の面を型で満たす（綴りのドリフト検出）", () => {
  // 型検査だけの門（実行時は何もしない）。runtime 側で `run` /
  // `createGenerationContext` / `pastLength` の綴りが変わるとコンパイルエラーになる。
  const asGenerationSession = (session: Session): GenerationSession => session;
  assertEquals(typeof asGenerationSession, "function");
});
