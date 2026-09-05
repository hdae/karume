// gemma4 の chat フォーマット（`src/gemma/text/chat.ts`）の上流パリティ検証。GPU は使わない。
//
// フィクスチャ `gemma-text/gemma4-chat.json` は export recipe（`tools/export-recipes/gemma4/
// chat.py`）が生成する。期待値は **HF `apply_chat_template` を独立に呼んだ**もので、こちらの
// レンダラも compile 台本の畳み込みも通っていない（ADR
// [0084](../../../docs/decisions/0084-gemma-tokenizer-chat.md) 決定 7 — 共有すると parity が
// 恒真化する）。
//
// 門は 6 本:
//
// ① **描画**が上流のレンダリング結果と文字単位で一致する（割れたときに「描画」と「符号化」の
//    どちらの段かが読み手に伝わるよう、id 列とは別に見る）
// ② **id 列**が `apply_chat_template(tokenize=True)` と一致する（段 4 の合格線そのもの）
// ③ **射程外は fail loudly**（tools / thinking / tool_call / 画像パート / 未知 role / 空の
//    会話）。黙って無視すると「tool を渡したのに使われない」が例外なしで通るので、拒否経路は
//    直接叩くしかない
// ④ **停止 token 集合**が資産の追加語彙から導出でき、上流の `generation_config.json` の宣言と
//    一致する（ADR 0083 決定 8 / 0084 決定 5 — chat 形式と EOS 集合は同じ digest set）
// ⑤ **増分描画の turn-local 契約**（`gemma4ChatTurn`）— 「初回の全体描画 + 生成された本文 +
//    `<turn|>` + 以後の差分」の連結が、全会話を 1 度に描いたものと token 単位で一致する
// ⑥ **ターンの後始末**（`closeChatTurn`）— 後始末が失敗しても直列化席は返り、本体の失敗と
//    後始末の失敗はどちらも消えない（席が返らないと以後の `chat` / `dispose` が永久に待つ）
//
// 実資産（`outputs/series/gemma4-e2b-tokenizer/`）が在れば**全語彙**でも同じ id 列になること
// を併せて見る（部分集合と full の食い違いを塞ぐ門）。無い環境では SKIP する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { createBpeModel } from "../src/text/bpe.ts";
import {
  createStopStringFilter,
  type DetokenizerSource,
  StreamingDetokenizer,
} from "../src/text/detokenizer.ts";
import {
  chatStreamOf,
  closeChatTurn,
  decodeChatChunks,
  type Gemma4ChatStop,
  withRunDiagnostics,
} from "../src/gemma/pipeline.ts";
import type {
  GenerationEvent,
  GenerationStop,
  GenerationStream,
} from "../src/generation/sequence.ts";
import { type GemmaTokenizerAssets, parseGemmaTokenizerAsset } from "../src/gemma/text/asset.ts";
import { GemmaTokenizer } from "../src/gemma/text/tokenizer.ts";
import {
  type Gemma4ChatMessage,
  gemma4ChatPrompt,
  gemma4ChatTurn,
  gemma4StopTokens,
  renderGemma4Chat,
  renderGemma4ChatTurn,
} from "../src/gemma/text/chat.ts";

type ChatCase = {
  readonly name: string;
  readonly why: string;
  readonly messages: Gemma4ChatMessage[];
  readonly rendered: string;
  readonly ids: number[];
};

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
  readonly stopTokens: number[];
  readonly stopTokenSpellings: string[];
  readonly chat: ChatCase[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as Fixture;

/** フィクスチャの部分集合から資産表を直接組む（JSON パーサの門は tokenizer 側が見る）。 */
const assetsOf = (source: Fixture): GemmaTokenizerAssets => ({
  spec: { postProcessor: "none" },
  model: createBpeModel({
    vocab: source.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: source.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: source.asset.byteIds,
  }),
  addedTokens: new Map(source.asset.addedTokens),
  specialIds: new Set(source.asset.specialIds),
  unkId: source.asset.unkId,
  bosId: source.asset.bosId,
  eosId: source.asset.eosId,
});

const tokenizer = new GemmaTokenizer(assetsOf(fixture));

/** 系列側の実資産（無ければ undefined = compile していない）。 */
const seriesBytes = await Deno.readFile(
  new URL("../../../outputs/series/gemma4-e2b-tokenizer/tokenizer.json", import.meta.url),
).catch(() => undefined);

if (seriesBytes === undefined) {
  console.warn(
    "[karume] outputs/series/gemma4-e2b-tokenizer/tokenizer.json が無いため gemma4 chat の " +
      "実資産突合を SKIP する。生成: cd tools/export-recipes && uv run python -m gemma4.tokenizer",
  );
}

Deno.test("gemma4 chat 描画: 上流 apply_chat_template と文字列がビット一致", async (t) => {
  for (const testCase of fixture.chat) {
    await t.step(`${testCase.name} — ${testCase.why}`, () => {
      assertEquals(renderGemma4Chat(testCase.messages), testCase.rendered);
    });
  }
});

Deno.test("gemma4 chat 符号化: 上流 apply_chat_template(tokenize=True) と id 列が一致", async (t) => {
  for (const testCase of fixture.chat) {
    await t.step(`${testCase.name} — ${testCase.why}`, () => {
      assertEquals(gemma4ChatPrompt(tokenizer, testCase.messages), testCase.ids);
    });
  }
});

Deno.test("gemma4 chat: `<bos>` の所有者は chat 関数だけ（double-BOS を作らない）", () => {
  // `encode` が `<bos>` を付けないこと（ADR 0084 決定 5 の MUST）と、chat が 1 個だけ付ける
  // ことを対で見る。片方だけだと「両方が付ける」形が通る。
  const bos = fixture.asset.bosId;
  assertEquals(tokenizer.encode("hello").includes(bos), false, "encode は付けない");
  const ids = gemma4ChatPrompt(tokenizer, [{ role: "user", content: "hello" }]);
  assertEquals(ids[0], bos, "chat は先頭に付ける");
  assertEquals(ids.filter((id) => id === bos).length, 1, "付けるのは 1 個だけ");
});

Deno.test("gemma4 chat 射程: 素の会話の外は fail loudly（黙って無視しない）", async (t) => {
  const reject = (
    label: string,
    messages: readonly unknown[],
    expected: string,
  ): Promise<boolean> =>
    t.step(label, () => {
      assertThrows(
        () => renderGemma4Chat(messages as readonly Gemma4ChatMessage[]),
        Error,
        expected,
      );
    });

  await reject("空の会話", [], "会話が空");
  await reject(
    "tools（関数宣言）",
    [{ role: "user", content: "hi", tools: [{ name: "search" }] }],
    "射程外の欄",
  );
  await reject(
    "reasoning（thinking チャネル）",
    [{ role: "assistant", content: "hi", reasoning: "…" }],
    "射程外の欄",
  );
  await reject(
    "tool_calls",
    [{ role: "assistant", content: "", tool_calls: [{ function: { name: "f" } }] }],
    "射程外の欄",
  );
  await reject("role が tool", [{ role: "tool", content: "…" }], "role");
  await reject(
    "role が model（template の出力側の綴り）",
    [{ role: "model", content: "…" }],
    "role",
  );
  await reject(
    "画像パート（content が配列）",
    [{ role: "user", content: [{ type: "image_url", image_url: { url: "…" } }] }],
    "content",
  );
  await reject(
    "assistant 本文の thinking 綴り（上流は strip_thinking で巻き戻す）",
    [{ role: "assistant", content: "a<|channel>thought\nx<channel|>b" }],
    "thinking",
  );
});

// ---- 増分描画（turn-local 契約） ----------------------------------------
//
// `gemma4ChatTurn` は「新しい発話 1 つ」を**次の `generate` へ渡す差分 token 列**として描く。
// 成立させたい等式は（`⧺` は連結）:
//
//   gemma4ChatPrompt(全会話)
//     = gemma4ChatPrompt(先頭 turn まで) ⧺ 生成された本文 ⧺ [<turn|>] ⧺ gemma4ChatTurn(次の発話) ⧺ …
//
// `<turn|>` は `GenerationSequence` の `pendingToken`（未 commit frontier）が前置するので差分には
// 入らない。この門が無いと、消費者は「全体を描いて先頭の `<bos>` を剥がす」当て推量を書くしかなく、
// 当たっているかは実 GPU の逐語一致でしか分からない（= template が turn-local でなくなった日に、
// 例外ゼロで別の会話が組み上がる）。

/** 会話を「先頭ブロック + (生成された応答, 次の発話) の並び」で書いた形。 */
type TurnCase = {
  readonly name: string;
  /** 最初の `gemma4ChatPrompt` に渡す発話（system 等 + 最初の user）。 */
  readonly head: readonly Gemma4ChatMessage[];
  readonly turns: readonly {
    readonly answer: string;
    readonly next: Gemma4ChatMessage;
  }[];
};

/** 差分の末尾（turn を閉じる札 + 生成プロンプト）— `renderGemma4Chat` の末尾と同じ綴り。 */
const TURN_TAIL = "<turn|>\n<|turn>model\n";
const END_OF_TURN = "<turn|>";

/**
 * 「model が実際に出した本文 token 列」の代役。
 *
 * 生成プロンプト（`…<|turn>model\n`）に続けて本文を書いた符号化の**続き**で、実運用の decode ループ
 * が出す列そのものである。境界で BPE が跨いでいないことは**その場で assert する**（跨いだら
 * この代役が成り立たないので、門が黙って別の意味になるのを防ぐ）。
 */
const modelAnswerTokens = (
  tokenizer: GemmaTokenizer,
  prefix: readonly Gemma4ChatMessage[],
  answer: string,
): { readonly body: string; readonly ids: number[] } => {
  const head = renderGemma4Chat(prefix);
  const closed = renderGemma4Chat([...prefix, { role: "assistant", content: answer }]);
  assert(closed.startsWith(head), "描画が turn-local でない（prefix が接頭辞になっていない）");
  assert(closed.endsWith(TURN_TAIL), `model turn の閉じ方が変わった: ${JSON.stringify(closed)}`);
  // 上流と同じ trim を再実装せず、描画結果から本文だけを取り出す。
  const body = closed.slice(head.length, closed.length - TURN_TAIL.length);
  const withBody = tokenizer.encode(head + body);
  const bare = tokenizer.encode(head);
  assertEquals(withBody.slice(0, bare.length), bare, "生成プロンプトの境界で BPE が跨いだ");
  return { body, ids: withBody.slice(bare.length) };
};

/** fixture のケース（[…, assistant, …] の形）を {@link TurnCase} へ読み替える。 */
const fromFixture = (name: string): TurnCase => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `フィクスチャに chat ケース '${name}' が無い`);
  const first = found.messages.findIndex((message) => message.role === "assistant");
  assert(first > 0, `${name}: assistant を挟まないケースは増分の門に使えない`);
  const turns: { answer: string; next: Gemma4ChatMessage }[] = [];
  for (let index = first; index + 1 < found.messages.length; index += 2) {
    assertEquals(found.messages[index].role, "assistant", `${name}: 交互になっていない`);
    turns.push({ answer: found.messages[index].content, next: found.messages[index + 1] });
  }
  return { name, head: found.messages.slice(0, first), turns };
};

const TURN_CASES: readonly TurnCase[] = [
  // 上流パリティ済みの会話（`renderGemma4Chat` の門が同じ文字列を見ている）。
  fromFixture("round-trip"),
  fromFixture("japanese"),
  // 応答が空白で始まる / 終わる形（差分側でも上流と同じ trim が掛かること）。
  fromFixture("assistant-whitespace"),
  {
    name: "3 往復（差分を積み上げても system ブロックが動かない）",
    head: [
      { role: "system", content: "You are terse." },
      { role: "user", content: "Name a color." },
    ],
    turns: [
      { answer: "Blue.", next: { role: "user", content: "Another one." } },
      { answer: "Red.", next: { role: "user", content: "One more, please." } },
      { answer: "Green.", next: { role: "user", content: "Thanks!" } },
    ],
  },
  {
    name: "途中の developer（先頭でないので system ブロックへ落ちない）",
    head: [{ role: "user", content: "Hello" }],
    turns: [
      { answer: "Hi!", next: { role: "developer", content: "Answer in one word." } },
      { answer: "OK", next: { role: "user", content: "What is 2+2?" } },
    ],
  },
  {
    name: "空の応答（停止 token だけを出したターン）",
    head: [{ role: "user", content: "…" }],
    turns: [{ answer: "", next: { role: "user", content: "もう一度" } }],
  },
  {
    name: "改行・記号を含む応答",
    head: [{ role: "user", content: "Write two lines." }],
    turns: [
      { answer: "line one\nline two", next: { role: "user", content: "  余白付き  " } },
    ],
  },
];

Deno.test("gemma4 chat 増分: 差分の積み上げが全会話の描画と文字列一致（turn-local）", async (t) => {
  for (const testCase of TURN_CASES) {
    await t.step(testCase.name, () => {
      let prefix: Gemma4ChatMessage[] = [...testCase.head];
      for (const turn of testCase.turns) {
        const { body } = modelAnswerTokens(tokenizer, prefix, turn.answer);
        const grown: Gemma4ChatMessage[] = [
          ...prefix,
          { role: "assistant", content: turn.answer },
          turn.next,
        ];
        // 差分は「生成プロンプトの続き = 本文 + 閉じ札」の**後ろ**から始まる。
        assertEquals(
          renderGemma4Chat(grown),
          renderGemma4Chat(prefix) + body + END_OF_TURN + renderGemma4ChatTurn(turn.next),
          "全会話の描画 = prefix + 本文 + <turn|> + 差分",
        );
        prefix = grown;
      }
    });
  }
});

Deno.test("gemma4 chat 増分: 差分 token 列の連結が全会話の id 列と一致（pendingToken 込み）", async (t) => {
  const stopId = tokenizer.addedTokenId(END_OF_TURN);
  assert(stopId !== undefined, "資産に <turn|> が無い");
  for (const testCase of TURN_CASES) {
    await t.step(testCase.name, () => {
      let prefix: Gemma4ChatMessage[] = [...testCase.head];
      // 実運用の経路をそのまま写す: 初回だけ全体を描き、以後は「本文 + pendingToken + 差分」。
      let rebuilt = gemma4ChatPrompt(tokenizer, prefix);
      for (const turn of testCase.turns) {
        const answer = modelAnswerTokens(tokenizer, prefix, turn.answer);
        rebuilt = [...rebuilt, ...answer.ids, stopId, ...gemma4ChatTurn(tokenizer, turn.next)];
        prefix = [...prefix, { role: "assistant", content: turn.answer }, turn.next];
      }
      assertEquals(
        rebuilt,
        gemma4ChatPrompt(tokenizer, prefix),
        "積み上げた id 列が全会話の描画・符号化と一致しない（turn-local 契約が壊れている）",
      );
    });
  }
});

Deno.test("gemma4 chat 増分: 差分に `<turn|>` を含めない（pendingToken と二重にしない）", () => {
  // 前 turn を閉じる札は生成が出して sequence が握っている（ADR 0083 決定 4）。差分の先頭は
  // その**直後**の改行で、`<|turn>` から始まってはならない。
  const rendered = renderGemma4ChatTurn({ role: "user", content: "hi" });
  assertEquals(rendered, "\n<|turn>user\nhi<turn|>\n<|turn>model\n");
  assertEquals(rendered.includes(END_OF_TURN), true, "自分の turn は閉じる");
  assertEquals(rendered.startsWith(`\n<|turn>`), true, "先頭は `<turn|>` の直後の改行");
  // 会話全体（`<bos>` 込み）とは別物であること — 取り違えると double-BOS になる。
  assertEquals(rendered.includes("<bos>"), false, "差分に <bos> は入らない");
});

Deno.test("gemma4 chat 増分 射程: 素の発話の外は fail loudly", async (t) => {
  const reject = (label: string, message: unknown, expected: string): Promise<boolean> =>
    t.step(label, () => {
      assertThrows(
        () => renderGemma4ChatTurn(message as Gemma4ChatMessage),
        Error,
        expected,
      );
    });

  // model turn は生成が埋める席で、template も連続 assistant を 1 つへ畳む（差分では描けない）。
  await reject("assistant", { role: "assistant", content: "…" }, "'assistant' は差分にできない");
  await reject("role が tool", { role: "tool", content: "…" }, "role");
  await reject("role が model", { role: "model", content: "…" }, "role");
  await reject("tools（射程外の欄）", { role: "user", content: "hi", tools: [] }, "射程外の欄");
  await reject("content が配列", { role: "user", content: [{ type: "text" }] }, "content");
});

Deno.test("gemma4 chat 停止 token: 資産から導出した集合が上流の宣言と一致", () => {
  assertEquals(fixture.stopTokenSpellings, ["<eos>", "<turn|>", "<|tool_response>"], "上流の綴り");
  assertEquals(
    [...gemma4StopTokens(tokenizer)].sort((a, b) => a - b),
    [...fixture.stopTokens].sort((a, b) => a - b),
    "generation_config.json の eos_token_id",
  );
});

Deno.test("gemma4 chat 停止 token: 綴りが欠けた資産は fail loudly", () => {
  // 集合が痩せたまま通すと「`<turn|>` で止まらず次の turn を自分で書き始める」形になり、
  // 例外は 1 つも出ない（golden でしか気づけない沈黙劣化）。
  const assets = assetsOf(fixture);
  const thinned = new Map(assets.addedTokens);
  thinned.delete("<|tool_response>");
  assertThrows(
    () => gemma4StopTokens(new GemmaTokenizer({ ...assets, addedTokens: thinned })),
    Error,
    "<|tool_response>",
  );
});

// ---- 停止文字列と一括受け取り（chat 層 — ADR 0083 追記 2026-09-02）----------
//
// 停止条件は 2 層に分かれる: **token** は sequence 層（`generation_sequence_test.ts` の門）、
// **文字列**は復号の後 = chat 層でしか判定できない（1 つの停止文字列が複数 token に割れることも、
// 1 つの token が停止文字列の末尾と次の本文をまたぐこともある）。ここで縛るのはその chat 層側で、
// 実体は `decodeChatChunks`（`src/gemma/pipeline.ts`）+ `createStopStringFilter`
// （`src/text/detokenizer.ts`）である。id → 綴りの対応をテストが決められるよう、復号器は実
// `StreamingDetokenizer` に**フィクスチャの綴り表**（`DetokenizerSource`）を差して組む — token の
// 割れ方を握らないと「境界を跨いだ停止文字列」を再現できない。

/** id → 綴りの表（`undefined` = skip 対象の特殊トークン）。byte_fallback は `bytes` で与える。 */
const sourceOf = (
  spellings: readonly string[],
  bytes: Readonly<Record<number, number>> = {},
): DetokenizerSource => ({
  byteOf: (id) => Object.hasOwn(bytes, id) ? bytes[id] : undefined,
  textOf: (id) => {
    assert(id < spellings.length, `テストの綴り表に id ${id} が無い`);
    return spellings[id];
  },
});

/** 「どこまで汲まれたか」を観測できる token イベント列（早期終了の畳み方を見るため）。 */
const fakeEvents = (ids: readonly number[]) => {
  const state = { emitted: 0, exhausted: false, closed: false };
  const events = (async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
    try {
      yield { kind: "prefill", chunk: 1, chunks: 1 };
      for (const [index, id] of ids.entries()) {
        state.emitted += 1;
        yield { kind: "token", id, position: index };
      }
      state.exhausted = true;
    } finally {
      // `return()`（= 早期終了）でも通常終了でも走る。`exhausted` との組で「途中で畳まれた」が読める。
      state.closed = true;
    }
  })();
  return { events, state };
};

/** chat 層の復号 1 回ぶん（返り値の停止文字列まで受けるので `for await` は使わない）。 */
const decodeChat = async (
  spellings: readonly string[],
  ids: readonly number[],
  stopStrings: readonly string[],
  bytes: Readonly<Record<number, number>> = {},
): Promise<{
  readonly parts: string[];
  readonly matched: string | undefined;
  readonly state: { emitted: number; exhausted: boolean; closed: boolean };
}> => {
  const { events, state } = fakeEvents(ids);
  const chunks = decodeChatChunks(
    events,
    new StreamingDetokenizer(sourceOf(spellings, bytes)),
    createStopStringFilter(stopStrings),
  );
  const parts: string[] = [];
  for (;;) {
    const step = await chunks.next();
    if (step.done) return { parts, matched: step.value, state };
    parts.push(step.value);
  }
};

/** 綴り表: 0..3 で "Hello " + 停止文字列 "END" が **token 境界を跨いで** 現れる形。 */
const SPELLINGS = ["Hel", "lo ", "EN", "D!", "X!", " tail"];

Deno.test("chat 停止文字列: token 境界を跨いでも止まり、停止文字列は出力に入らない", async () => {
  // "EN" と "D!" の 2 token に割れた "END"。復号後の本文でしか判定できない形である。
  const { parts, matched, state } = await decodeChat(SPELLINGS, [0, 1, 2, 3, 5], ["END"]);
  assertEquals(matched, "END", "一致した停止文字列");
  assertEquals(parts.join(""), "Hello ", "停止文字列そのものと、その後ろは流れない");
  assertEquals(parts.filter((part) => part === "").length, 0, "空の片を作らない");
  // 消費はそこで止まる（残りの token は要求していない）= sequence の早期終了が畳まれている。
  assertEquals(state.emitted, 4, "停止を確定させた token までしか汲まない");
  assertEquals(state.exhausted, false, "イベント列を最後まで汲んでいる（止まっていない）");
  assertEquals(state.closed, true, "イベント列の return() が呼ばれていない（後始末が漏れる）");
});

Deno.test("chat 停止文字列: 一致しなければ保留ぶんが最後に流れる（1 文字も落とさない）", async () => {
  // 保留は判定のための遅延であって出力の切り詰めではない — "EN" の次が "D" でなければ全部出る。
  const { parts, matched, state } = await decodeChat(SPELLINGS, [0, 1, 2, 4], ["END"]);
  assertEquals(matched, undefined, "止まっていない");
  assertEquals(parts.join(""), "Hello ENX!", "保留していた接頭辞まで含めて全部流れる");
  assertEquals(state.exhausted, true, "イベント列は最後まで汲まれる");
});

Deno.test("chat 停止文字列: 保留するのは接頭辞になっている間だけ（描画を止めっぱなしにしない）", async () => {
  // 片の**並び**まで見る（連結だけ見ると「最後にまとめて出す」偽 streaming が通る）。停止文字列の
  // 接頭辞になった "EN" の間だけ保留し、外れた時点でまとめて流す。
  const { parts } = await decodeChat(SPELLINGS, [0, 1, 2, 4], ["END"]);
  assertEquals(parts, ["Hel", "lo ", "ENX!"], "接頭辞の間だけ保留（3 番目の片で追い付く）");
  // 停止文字列と無縁のターンでは 1 文字も保留しない（毎片そのまま流れる）。
  const bare = await decodeChat(SPELLINGS, [0, 1, 5], ["ZZZ"]);
  assertEquals(bare.parts, ["Hel", "lo ", " tail"]);
});

Deno.test("chat 停止文字列: byte run の確定ぶんも判定に入る（列の終わりで止まる）", async () => {
  // byte_fallback の run は次の非 byte token か `finish()` まで確定しない（ADR 0084 決定 4）。
  // その確定ぶんを判定へ通さないと、run の中で完成した停止文字列を取りこぼす。
  const spellings = ["Hi ", "EN", "<0x44>"];
  const { parts, matched } = await decodeChat(spellings, [0, 1, 2], ["END"], { 2: 0x44 });
  assertEquals(matched, "END", "列の終わりで確定した 'D' が停止文字列を完成させる");
  assertEquals(parts.join(""), "Hi ");
});

Deno.test("chat 停止文字列: 空文字列と重複は fail loudly", () => {
  // 空文字列は「常に一致する」= 1 文字も出せない指定、重複は同じ条件の二重書き（どちらも取り違え）。
  assertThrows(() => createStopStringFilter([""]), Error, "stopStrings[0] が空文字列");
  assertThrows(
    () => createStopStringFilter(["END", "END"]),
    Error,
    'stopStrings に "END" が 2 度出る',
  );
});

Deno.test("chat 停止文字列: 複数の停止文字列は本文に先に現れた方で切る", async () => {
  // 宣言の順ではなく**流れを実際に切った方**を運ぶ（"d" を先に宣言しても本文では "b" が先）。
  const { parts, matched } = await decodeChat(["ab", "cd"], [0, 1], ["d", "b"]);
  assertEquals(matched, "b");
  assertEquals(parts.join(""), "a");
});

Deno.test("chat prefill: 進捗は onPrefill が受け、本文の列は 1 文字も変わらない", async () => {
  // 長い prompt では最初の文字が出るまでの無音時間が prefill そのもので、文字列の面には
  // その進捗を運ぶ片が無い（本文はまだ 1 文字も出ていない）。観測席が唯一の経路である。
  const events = (async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
    yield { kind: "prefill", chunk: 1, chunks: 3 };
    yield { kind: "prefill", chunk: 2, chunks: 3 };
    yield { kind: "prefill", chunk: 3, chunks: 3 };
    yield { kind: "token", id: 0, position: 0 };
    yield { kind: "token", id: 1, position: 1 };
  })();
  const seen: { readonly chunk: number; readonly chunks: number }[] = [];
  const chunks = decodeChatChunks(
    events,
    new StreamingDetokenizer(sourceOf(SPELLINGS)),
    createStopStringFilter([]),
    (progress) => seen.push(progress),
  );
  const parts: string[] = [];
  for await (const part of chunks) parts.push(part);

  assertEquals(seen, [
    { chunk: 1, chunks: 3 },
    { chunk: 2, chunks: 3 },
    { chunk: 3, chunks: 3 },
  ], "commit 済み chunk 数がそのまま届く");
  assertEquals(parts, ["Hel", "lo "], "本文の片は観測席の有無で変わらない");
});

Deno.test("chat 一括: text() は片の連結と一致し、done も併せて読める", async () => {
  const stop: Gemma4ChatStop = { reason: "max-tokens", tokens: 3 };
  const chunks = (async function* (): AsyncGenerator<string, void, undefined> {
    yield "Hel";
    yield "lo ";
    yield "world";
  })();
  const stream = chatStreamOf(chunks, Promise.resolve(stop));
  assertEquals(await stream.text(), "Hello world");
  assertEquals(await stream.done, stop, "停止理由は text() の後でも読める");
});

Deno.test("chat 一括: 1 つのストリームは 1 通りにしか消費できない（同期に落ちる）", async () => {
  const stop: Gemma4ChatStop = { reason: "max-tokens", tokens: 1 };
  const streamOf = () =>
    chatStreamOf(
      (async function* (): AsyncGenerator<string, void, undefined> {
        yield "a";
      })(),
      Promise.resolve(stop),
    );
  // 生成は 1 度しか走らないので、2 通り目には「残り」しか流れない（例外にならない取り違え）。
  const iterated = streamOf();
  for await (const _part of iterated) { /* 汲み切る */ }
  assertThrows(() => iterated.text(), Error, "1 通りにしか消費できない");

  const collected = streamOf();
  assertEquals(await collected.text(), "a");
  assertThrows(() => collected[Symbol.asyncIterator](), Error, "1 通りにしか消費できない");

  const twice = streamOf();
  twice[Symbol.asyncIterator]();
  assertThrows(() => twice[Symbol.asyncIterator](), Error, "1 通りにしか消費できない");
});

// ---- ターンの後始末（`closeChatTurn`）---------------------------------------
//
// `Gemma4Pipeline.chat` の直列化席は「後始末が失敗したら返らない」形だと、鎖が前段の決着を
// 得られないまま以後の `chat` / `dispose` を永久に待つ（device 消失で `context.dispose` が
// `flush` の失敗を伝播させる経路が実在する = 例外 1 つで二度と動かないパイプライン）。
// 席の返却と例外の畳み方はこの 1 本に集めてあるので、故障注入もここで書ける（GPU 不要）。

Deno.test("chat 後始末: 席は無条件に返り、後始末の失敗は本体の例外と併せて運ぶ", async (t) => {
  const failing = (error: unknown) => (): Promise<void> => Promise.reject(error);
  const caughtOf = (work: Promise<void>): Promise<unknown> =>
    work.then(() => undefined, (error: unknown) => error);

  await t.step("成功した後始末は席を返すだけ", async () => {
    let released = 0;
    await closeChatTurn("test", undefined, () => Promise.resolve(), () => {
      released += 1;
    });
    assertEquals(released, 1);
  });

  await t.step("後始末が失敗しても席は返る（鎖を握ったままにしない）", async () => {
    const boom = new Error("context.dispose が flush の失敗を伝播した");
    let released = 0;
    const caught = await caughtOf(
      closeChatTurn("test", undefined, failing(boom), () => {
        released += 1;
      }),
    );
    assertEquals(caught, boom, "本体が成功していれば後始末の例外がそのまま届く");
    assertEquals(released, 1, "席は無条件に返る");
  });

  await t.step("本体も失敗していれば 2 本とも運ぶ（どちらの事実も消さない）", async () => {
    const body = new Error("ターン本体");
    const cleanup = new Error("後始末");
    let released = 0;
    const caught = await caughtOf(
      closeChatTurn("test", { error: body }, failing(cleanup), () => {
        released += 1;
      }),
    );
    assert(caught instanceof AggregateError, `AggregateError でない: ${caught}`);
    assertEquals(caught.errors, [body, cleanup], "errors[0] が本体・errors[1] が後始末");
    assertEquals(released, 1);
  });

  await t.step("席を持たない呼び手（send）でも畳み方は同じ", async () => {
    const body = new Error("ターン本体");
    const cleanup = new Error("後始末");
    const caught = await caughtOf(closeChatTurn("test", { error: body }, failing(cleanup)));
    assert(caught instanceof AggregateError, `AggregateError でない: ${caught}`);
    assertEquals(caught.errors, [body, cleanup]);
  });
});

Deno.test({
  name: "gemma4 chat 実資産突合: 全語彙でもフィクスチャと同じ id 列（部分集合の穴を塞ぐ）",
  ignore: seriesBytes === undefined,
  fn: () => {
    const full = new GemmaTokenizer(parseGemmaTokenizerAsset(seriesBytes as Uint8Array));
    for (const testCase of fixture.chat) {
      assertEquals(gemma4ChatPrompt(full, testCase.messages), testCase.ids, testCase.name);
    }
    assertEquals(
      [...gemma4StopTokens(full)].sort((a, b) => a - b),
      [...fixture.stopTokens].sort((a, b) => a - b),
      "実資産から導出した停止 token 集合",
    );
  },
});

// ---- 観測席の呼び出し規則（`withRunDiagnostics`）----------------------------
//
// doc（`src/gemma/pipeline.ts`）は 2 つの非自明な規則を宣言している —— ①prefill 直後の最初の
// token は「最終 chunk の logits から抽選しただけ」で run を伴わないので席を呼ばない
// ②停止 token を引いた最後の decode run の診断は列に出ないので**毎ターン 1 本欠ける**。
// 唯一の観測（`e2e_gemma4_pretrained_test.ts` の census）はキーの集合を集めるだけで**回数**を
// 見ていないため、規則が壊れても赤くならなかった。

/** イベント列を手で書いた偽 `GenerationStream`（`closed` で早期終了の伝播が読める）。 */
const fakeStream = (events: readonly GenerationEvent[]) => {
  const state = { closed: false };
  const iterable = (async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
    try {
      for (const event of events) yield event;
    } finally {
      state.closed = true;
    }
  })();
  const stream: GenerationStream = {
    [Symbol.asyncIterator]: () => iterable,
    done: Promise.resolve<GenerationStop>({ reason: "closed", tokens: 0 }),
  };
  return { stream, state };
};

const prefill = (chunk: number, chunks: number): GenerationEvent => ({
  kind: "prefill",
  chunk,
  chunks,
});
const token = (id: number, position: number): GenerationEvent => ({ kind: "token", id, position });

/** 診断は素通しされるだけ（中身を読まない）ので、席が受けた値をそのまま数える。 */
const diagnosticsSeat = () => {
  const seen: number[] = [];
  let ticket = 0;
  return {
    seen,
    seat: {
      session: {
        diagnostics: (): number => {
          ticket += 1;
          return ticket;
        },
      },
      onRunDiagnostics: (diagnostics: number): void => {
        seen.push(diagnostics);
      },
    },
  };
};

Deno.test("withRunDiagnostics: run を伴わない最初の token では席を呼ばない", async () => {
  const { stream, state } = fakeStream([
    prefill(1, 3),
    prefill(2, 3),
    prefill(3, 3),
    token(10, 0),
    token(11, 1),
    token(12, 2),
    token(13, 3),
  ]);
  const { seen, seat } = diagnosticsSeat();
  const wrapped = withRunDiagnostics(stream, seat);
  const drained: GenerationEvent[] = [];
  for await (const event of wrapped) drained.push(event);

  assertEquals(drained.length, 7, "イベントは 1 つも落とさず素通しする");
  // prefill 3 本 + token 4 本のうち最初の 1 本を飛ばす = 6 回。
  assertEquals(seen.length, 6, "呼び出し回数");
  assertEquals(seen, [1, 2, 3, 4, 5, 6], "席が受けるのは呼ぶたびの新しい診断");
  assertEquals(state.closed, true, "内側の列が閉じていない");
});

Deno.test("withRunDiagnostics: 最初の抽選が停止 token だったターンは prefill ぶんだけ", async () => {
  // 本文が 1 文字も出ないターン（列に token が 1 つも現れない）。
  const { stream } = fakeStream([prefill(1, 1)]);
  const { seen, seat } = diagnosticsSeat();
  for await (const _event of withRunDiagnostics(stream, seat)) { /* 汲み切る */ }
  assertEquals(seen.length, 1);
});

Deno.test("withRunDiagnostics: 席が無ければ元の列をそのまま返す（包みを 1 枚も足さない）", () => {
  const { stream } = fakeStream([prefill(1, 1), token(10, 0)]);
  assert(
    withRunDiagnostics(stream, { session: { diagnostics: () => 0 } }) === stream,
    "席が無いのに包みが増えている（中断や return() の伝播経路が 1 枚深くなる）",
  );
});

Deno.test("withRunDiagnostics: 消費側の break が内側の return() まで伝わる", async () => {
  const { stream, state } = fakeStream([
    prefill(1, 1),
    token(10, 0),
    token(11, 1),
    token(12, 2),
  ]);
  const { seen, seat } = diagnosticsSeat();
  for await (const event of withRunDiagnostics(stream, seat)) {
    if (event.kind === "token") break;
  }
  assertEquals(state.closed, true, "包みが中断経路を切っている（内側が走行中のまま残る）");
  assertEquals(seen.length, 1, "prefill ぶんだけ（最初の token は席を呼ばない）");
});
