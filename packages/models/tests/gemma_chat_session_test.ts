// `Gemma4ChatSession`（`src/gemma/chat-session.ts`）の会話管理の門。GPU は使わない。
//
// この層が持つのは**会話（発話の並び）と KV の対応**だけで、生成そのものは
// `GenerationSequence` に、描画・復号・停止文字列は `chat()` と同じ実装に乗る。だから門も
// 「どんな prompt を、どの sequence へ流したか」で書ける — 台本どおりに答える偽 sequence を
// 差し（`Gemma4ChatSessionHost` = セッションがパイプラインから読む面）、tokenizer は
// `gemma_chat_test.ts` と同じフィクスチャ資産で**本物**を組む（差分描画の等式は本物の綴りでしか
// 見られない）。
//
// 門は 7 本:
//
// ① **KV の継続**: 2 ターン目は差分（`gemma4ChatTurn`）だけを流し、sequence を作り直さない
// ② **継げない停止の後**（max-tokens）は sequence を捨てて履歴を全体描画で撃ち直す — ①だけだと
//    「いつでも継ぐ」実装が緑のままになる（model turn が閉じていない会話へ差分を継ぐと、例外
//    ゼロで turn の区切りが壊れる）
// ③ **消費側の `break`** が内側のイベント列まで届き（`done` が決着する）、出た片だけが履歴へ入る
// ④ **溢れ処理**: 送る前に判定し、既定ポリシー（`dropOldestTurns`）が最古の対を落として system を
//    残す。落とした履歴は KV と対応しないので sequence を作り直す
// ⑤ **落とせるものが尽きたら fail loudly**（`GenerationCapacityError`）— 黙って縮めない。
//    無かったことになったターンの発話は履歴に残らない
// ⑥ **ポリシーの注入**が効く（履歴を返す / throw をそのまま素通しする）。縮まない履歴を返す
//    ポリシーは再試行の無限ループではなく fail loudly になる
// ⑦ **1 セッション = 1 生成**（同時 send / dispose 後の send は同期に throw）と、`send` が返す
//    stream が `chat()` と同じ契約（停止理由・`text()`・1 通りにしか消費できない）

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  dropOldestTurns,
  type Gemma4ChatOverflow,
  Gemma4ChatSession,
  type Gemma4ChatSessionHost,
} from "../src/gemma/chat-session.ts";
import type { GenerationProgram } from "../src/generation/program.ts";
import {
  GenerationCapacityError,
  type GenerationEvent,
  type GenerationRequest,
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../src/generation/sequence.ts";
import { createBpeModel } from "../src/text/bpe.ts";
import type { GemmaTokenizerAssets } from "../src/gemma/text/asset.ts";
import { GemmaTokenizer } from "../src/gemma/text/tokenizer.ts";
import {
  type Gemma4ChatMessage,
  gemma4ChatPrompt,
  gemma4ChatTurn,
  gemma4StopTokens,
} from "../src/gemma/text/chat.ts";

type Fixture = {
  readonly asset: {
    readonly vocab: [number, string][];
    readonly merges: [number, number, number][];
    readonly byteIds: number[];
    readonly addedTokens: [string, number][];
    readonly specialIds: number[];
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
  };
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as Fixture;

/** フィクスチャの部分集合から資産表を組む（`gemma_chat_test.ts` と同じ形）。 */
const assets: GemmaTokenizerAssets = {
  spec: { postProcessor: "none" },
  model: createBpeModel({
    vocab: fixture.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: fixture.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: fixture.asset.byteIds,
  }),
  addedTokens: new Map(fixture.asset.addedTokens),
  specialIds: new Set(fixture.asset.specialIds),
  unkId: fixture.asset.unkId,
  bosId: fixture.asset.bosId,
  eosId: fixture.asset.eosId,
};
const tokenizer = new GemmaTokenizer(assets);
const STOP_TOKENS = gemma4StopTokens(tokenizer);
/** 配布形が宣言する停止 token の 1 つ（turn を閉じる札）。 */
const END_OF_TURN = tokenizer.addedTokenId("<turn|>") as number;

const SYSTEM = "You are terse.";
const SYSTEM_MESSAGE: Gemma4ChatMessage = { role: "system", content: SYSTEM };
const MAX_NEW_TOKENS = 16;

/** 台本 1 ターンぶん。`closes` = 配布形の停止 token で閉じる（次ターンを差分で継げる）。 */
type Answer = { readonly text: string; readonly closes: boolean };

/** そのターンが会話へ積む論理位置の数（本文 + 停止 token）。 */
const answerCost = (answer: Answer): number =>
  tokenizer.encode(answer.text).length + (answer.closes ? 1 : 0);

const programOf = (capacity: number): GenerationProgram =>
  Object.freeze({
    chunkLength: 32,
    // 位置表は十分広く取る（この門で効かせたい上限は capacity 側だけ）。
    maxPosition: 4096,
    capacity,
    vocabSize: 262144,
    stopTokens: Object.freeze([...STOP_TOKENS]),
  });

type FakeHost = Gemma4ChatSessionHost & {
  /** 作った sequence の数（作り直したかがここに出る）。 */
  created(): number;
  /** 畳んだ sequence の数。 */
  disposed(): number;
  /** `generate` が受けた prompt を発行順に。 */
  prompts(): readonly (readonly number[])[];
  /** `sequence()` が受けた容量を発行順に（セッションのノブが降りているか）。 */
  capacities(): readonly (number | undefined)[];
};

/**
 * 台本どおりに答える偽 sequence を配るホスト。
 *
 * `used` は本物と同じ導出（prompt の長さ + 抽選した token の数 — 停止 token も 1 個）で進める。
 * これがずれると溢れ判定の門が意味を失うので、ここだけは本物の契約を写す。
 */
const fakeHost = (answers: readonly Answer[], program: GenerationProgram): FakeHost => {
  const prompts: number[][] = [];
  const capacities: (number | undefined)[] = [];
  let created = 0;
  let disposed = 0;
  let turn = 0;
  return {
    tokenizer,
    program,
    defaultSampler: undefined,
    created: () => created,
    disposed: () => disposed,
    prompts: () => prompts,
    capacities: () => capacities,
    sequence: (options = {}): Promise<GenerationSequence> => {
      created += 1;
      capacities.push(options.capacity);
      const capacity = options.capacity ?? program.capacity;
      let used = 0;
      let gone = false;
      return Promise.resolve({
        capacity,
        get used(): number {
          return used;
        },
        generate(request: GenerationRequest): GenerationStream {
          assert(!gone, "dispose 済みの sequence へ generate した");
          prompts.push([...request.prompt]);
          const answer = answers[turn];
          assert(answer !== undefined, `台本に ${turn + 1} ターン目が無い`);
          turn += 1;
          const ids = tokenizer.encode(answer.text);
          used += request.prompt.length;
          const stop: GenerationStop = answer.closes
            ? { reason: "eos", token: END_OF_TURN, tokens: ids.length + 1 }
            : { reason: "max-tokens", tokens: ids.length };

          let settle!: (value: GenerationStop) => void;
          const done = new Promise<GenerationStop>((resolve) => {
            settle = resolve;
          });
          const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
            let emitted = 0;
            let stopped: GenerationStop | undefined;
            try {
              yield { kind: "prefill", chunk: 1, chunks: 1 };
              for (const id of ids) {
                used += 1;
                emitted += 1;
                yield { kind: "token", id, position: used };
              }
              // 停止 token は列に出さないが会話には残る（未 commit frontier）。
              if (answer.closes) used += 1;
              stopped = stop;
            } finally {
              // `break` / `return()` で畳まれた場合は `closed`（本物と同じ分け方）。
              settle(stopped ?? { reason: "closed", tokens: emitted });
            }
          };
          const iterable = events();
          return { [Symbol.asyncIterator]: () => iterable, done };
        },
        dispose(): Promise<void> {
          gone = true;
          disposed += 1;
          return Promise.resolve();
        },
      });
    },
  };
};

Deno.test("ChatSession: 2 ターン目は差分だけを流し、sequence を作り直さない", async () => {
  const host = fakeHost(
    [{ text: "Blue.", closes: true }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  assertEquals(await session.send("Name a color.").text(), "Blue.");
  assertEquals(await session.send("Another one.").text(), "Red.");

  assertEquals(host.created(), 1, "KV を継いだなら sequence は 1 本だけ");
  assertEquals(
    host.prompts()[0],
    gemma4ChatPrompt(tokenizer, [SYSTEM_MESSAGE, { role: "user", content: "Name a color." }]),
    "初回は `<bos>` 込みの全体描画",
  );
  assertEquals(
    host.prompts()[1],
    gemma4ChatTurn(tokenizer, { role: "user", content: "Another one." }),
    "2 ターン目は差分だけ（前 turn を閉じる `<turn|>` は pendingToken が前置する）",
  );
  assertEquals(session.turns, [
    SYSTEM_MESSAGE,
    { role: "user", content: "Name a color." },
    { role: "assistant", content: "Blue." },
    { role: "user", content: "Another one." },
    { role: "assistant", content: "Red." },
  ]);
});

Deno.test("ChatSession: 停止 token で閉じなかったターンの次は全体を描き直す", async () => {
  // max-tokens で打ち切ったターンの後ろは model turn が閉じていない = 差分の前提が無い。
  const host = fakeHost(
    [{ text: "Blue", closes: false }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  const first = session.send("Name a color.");
  assertEquals(await first.text(), "Blue");
  assertEquals(await first.done, {
    reason: "max-tokens",
    tokens: tokenizer.encode("Blue").length,
  });
  assertEquals(await session.send("Another one.").text(), "Red.");

  assertEquals(host.created(), 2, "打ち切ったターンの後は sequence を作り直す");
  assertEquals(host.disposed(), 1, "捨てた sequence は畳む");
  assertEquals(
    host.prompts()[1],
    gemma4ChatPrompt(tokenizer, [
      SYSTEM_MESSAGE,
      { role: "user", content: "Name a color." },
      { role: "assistant", content: "Blue" },
      { role: "user", content: "Another one." },
    ]),
    "2 ターン目は履歴の全体描画",
  );
});

Deno.test("ChatSession: 汲むのをやめたターンは内側まで畳み、出た本文だけを残す", async () => {
  // 消費側の `break` は「中断と同じ後始末」で内側のイベント列まで届く必要がある（届かないと
  // sequence が走行中のまま残り、`done` も決着しない）。
  const host = fakeHost(
    [{ text: "Blue.", closes: true }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const stream = session.send("Name a color.");
  let first = "";
  for await (const chunk of stream) {
    first = chunk;
    break;
  }
  assertEquals((await stream.done).reason, "closed", "内側のイベント列も畳まれている");
  assert(first !== "" && "Blue.".startsWith(first), `最初の片が本文の頭でない: ${first}`);
  assertEquals(session.turns, [
    { role: "user", content: "Name a color." },
    { role: "assistant", content: first },
  ], "流した片だけが履歴へ入る");

  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "閉じていないターンの後は sequence を作り直す");
});

Deno.test("ChatSession 溢れ: 既定ポリシーが最古の対を落とし、system を残して組み直す", async () => {
  const answers: readonly Answer[] = [
    { text: "Blue.", closes: true },
    { text: "Red.", closes: true },
  ];
  const first = { role: "user", content: "Name a color." } as const;
  const second = { role: "user", content: "Another one." } as const;
  // 2 ターン目が **1 だけ** 入らない容量にする（差分を継いだままなら通る形を作らない）。
  const used = gemma4ChatPrompt(tokenizer, [SYSTEM_MESSAGE, first]).length + answerCost(answers[0]);
  const delta = gemma4ChatTurn(tokenizer, second).length;
  const host = fakeHost(answers, programOf(used + delta + MAX_NEW_TOKENS - 2));
  const seen: Gemma4ChatOverflow[] = [];
  const session = new Gemma4ChatSession(host, {
    system: SYSTEM,
    maxNewTokens: MAX_NEW_TOKENS,
    onOverflow: (context) => {
      seen.push(context);
      return dropOldestTurns(context);
    },
  });

  assertEquals(await session.send(first.content).text(), "Blue.");
  assertEquals(await session.send(second.content).text(), "Red.");

  assertEquals(seen.length, 1, "溢れ処理は 1 回で収まった");
  assertEquals(seen[0].system, SYSTEM);
  assertEquals(seen[0].needed, seen[0].capacity + 1, "1 だけ足りない");
  assertEquals(seen[0].turns.length, 4, "呼ばれた時点の履歴（system + 1 往復 + 今の発話）");
  assertEquals(session.turns, [
    SYSTEM_MESSAGE,
    second,
    { role: "assistant", content: "Red." },
  ], "最古の user / assistant の対が落ち、system は残る");
  assertEquals(host.created(), 2, "落とした履歴は KV と対応しない = sequence を作り直す");
  assertEquals(
    host.prompts()[1],
    gemma4ChatPrompt(tokenizer, [SYSTEM_MESSAGE, second]),
    "切り詰めた履歴の全体描画",
  );
});

Deno.test("ChatSession 溢れ: 落とせるものが尽きたら fail loudly（黙って縮めない）", async () => {
  // system + 今の発話しか無い = 既定ポリシーには落とせるものが無い。
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(8));
  const session = new Gemma4ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  const error = await assertRejects(
    () => session.send("Name a color.").text(),
    GenerationCapacityError,
    "会話が入り切らない",
  );
  assertEquals(error.constraint, "capacity");
  assertEquals(error.limit, 8);
  assertEquals(error.requestedNewTokens, MAX_NEW_TOKENS);
  // 「今なら通る maxNewTokens」は負値 = 何 token 溢れているか（ADR 0083 追記 2026-08-31）。
  assert(error.maxNewTokens < 0, `溢れ幅が読めない: ${error.maxNewTokens}`);
  assertEquals(session.turns, [SYSTEM_MESSAGE], "起きなかったターンの発話は履歴に残らない");
  assertEquals(host.disposed(), 1, "掴んだ sequence は畳む");
});

Deno.test("ChatSession 溢れ: ポリシーの注入が効く（履歴を返す / throw を素通しする）", async (t) => {
  await t.step("返した履歴で撃ち直す（system も落とせる）", async () => {
    const asked = { role: "user", content: "Name a color." } as const;
    // system 込みでは 1 だけ足りず、system を落とせば通る容量。
    const capacity = gemma4ChatPrompt(tokenizer, [SYSTEM_MESSAGE, asked]).length +
      MAX_NEW_TOKENS - 2;
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(capacity));
    const session = new Gemma4ChatSession(host, {
      system: SYSTEM,
      maxNewTokens: MAX_NEW_TOKENS,
      // 何が起きても「今の発話だけ」にする（system も捨てる = 既定と違う意味論）。
      onOverflow: ({ turns }) => turns.slice(-1),
    });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(session.turns, [
      { role: "user", content: "Name a color." },
      { role: "assistant", content: "Blue." },
    ]);
    assertEquals(
      host.prompts()[0],
      gemma4ChatPrompt(tokenizer, [{ role: "user", content: "Name a color." }]),
      "system の無い履歴で描き直す",
    );
  });

  await t.step("throw はそのまま呼び手へ届く（包まない）", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(8));
    const refusal = new Error("ホストが切り詰めを拒否した");
    const session = new Gemma4ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: () => {
        throw refusal;
      },
    });
    const caught = await session.send("Name a color.").text().then(
      () => undefined,
      (error: unknown) => error,
    );
    assertEquals(caught, refusal, "ポリシーの例外が同一性のまま届く");
  });

  await t.step("縮まない履歴は無限ループにせず fail loudly", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(8));
    let calls = 0;
    const session = new Gemma4ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: ({ turns }) => {
        calls += 1;
        return turns;
      },
    });
    await assertRejects(
      () => session.send("Name a color.").text(),
      GenerationCapacityError,
      "縮めなかった",
    );
    // 縮まない結果は 1 回で打ち切る（再試行の回数は履歴の件数に張り付かない）。
    assertEquals(calls, 1);
  });
});

Deno.test("ChatSession capacity: 省略時は program の既定・渡せばそれが sequence と溢れ判定の物差し", async (t) => {
  await t.step("省略時は配布形の既定がそのまま降りる", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    const session = new Gemma4ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.capacities(), [640], "sequence へ渡る容量");
  });

  await t.step("渡した容量が sequence へ降り、切り詰めの判断もその値で行う", async () => {
    // program の既定は十分広い（640）が、セッションは狭い容量を選ぶ — 既定で判断していれば
    // 溢れ処理は 1 度も呼ばれず、選んだ容量で判断していれば呼ばれる。
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    let overflows = 0;
    const session = new Gemma4ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      capacity: 8,
      onOverflow: (context) => {
        overflows += 1;
        return dropOldestTurns(context);
      },
    });
    const error = await assertRejects(
      () => session.send("Name a color.").text(),
      GenerationCapacityError,
      "会話が入り切らない",
    );
    assertEquals(error.limit, 8, "選んだ容量が上限として運ばれる");
    assertEquals(overflows, 1, "溢れ処理は選んだ容量で呼ばれる");
    assertEquals(host.capacities(), [8], "sequence にも同じ容量が降りる");
  });
});

Deno.test("ChatSession: onPrefill が chunk ごとの進捗を運ぶ（本文には出ない情報）", async () => {
  const host = fakeHost(
    [{ text: "Blue.", closes: true }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const seen: { readonly chunk: number; readonly chunks: number }[] = [];

  assertEquals(
    await session.send("Name a color.", { onPrefill: (progress) => seen.push(progress) }).text(),
    "Blue.",
  );
  assertEquals(seen, [{ chunk: 1, chunks: 1 }], "台本の prefill イベントがそのまま届く");

  // ターンごとの指定なので、渡さなかったターンには届かない（購読を持ち越さない）。
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(seen.length, 1);
});

Deno.test("ChatSession: 1 セッション = 1 生成（同時 send と dispose 後は同期に throw）", async () => {
  const host = fakeHost(
    [{ text: "Blue.", closes: true }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  // 発行しただけ（まだ汲んでいない）のターンが走っている間は 2 本目を受けない。
  const first = session.send("Name a color.");
  assertThrows(
    () => session.send("Another one."),
    Error,
    "前のターンがまだ終わっていない",
  );
  assertEquals(await first.text(), "Blue.");
  // 汲み切ったターンの後は受ける。
  assertEquals(await session.send("Another one.").text(), "Red.");

  await session.dispose();
  assertEquals(host.disposed(), 1, "dispose は sequence を畳む");
  assertThrows(() => session.send("hi"), Error, "dispose 済み");
});

Deno.test("ChatSession: 停止文字列で切ったターンは chat() と同じ理由を運び、KV を継がない", async () => {
  const host = fakeHost(
    [{ text: "Blue.\nUser: hi", closes: true }, { text: "Red.", closes: true }],
    programOf(640),
  );
  const session = new Gemma4ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const stream = session.send("Name a color.", { stopStrings: ["\nUser:"] });
  assertEquals(await stream.text(), "Blue.", "停止文字列そのものと、その後ろは流れない");
  const stop = await stream.done;
  assertEquals(stop.reason, "stop-string");
  assert(stop.reason === "stop-string" && stop.stopString === "\nUser:");
  assert(stop.tokens > 0, "内側の数をそのまま運ぶ");
  // 1 つのストリームは 1 通りにしか消費できない（`chat()` と同じ契約）。
  assertThrows(() => stream[Symbol.asyncIterator](), Error, "1 通りにしか消費できない");

  assertEquals(session.turns, [
    { role: "user", content: "Name a color." },
    { role: "assistant", content: "Blue." },
  ], "流した本文だけが履歴へ入る");
  // model turn は閉じていない = 差分の前提が無い。
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "停止文字列で切ったターンの後は sequence を作り直す");
});
