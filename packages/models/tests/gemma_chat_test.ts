// gemma4 の chat フォーマット（`src/gemma/text/chat.ts`）の上流パリティ検証。GPU は使わない。
//
// フィクスチャ `gemma-text/gemma4-chat.json` は export recipe（`tools/export-recipes/gemma4/
// chat.py`）が生成する。期待値は **HF `apply_chat_template` を独立に呼んだ**もので、こちらの
// レンダラも compile 台本の畳み込みも通っていない（ADR
// [0084](../../../docs/decisions/0084-gemma-tokenizer-chat.md) 決定 7 — 共有すると parity が
// 恒真化する）。
//
// 門は 4 本:
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
//
// 実資産（`outputs/series/gemma4-e2b-tokenizer/`）が在れば**全語彙**でも同じ id 列になること
// を併せて見る（部分集合と full の食い違いを塞ぐ門）。無い環境では SKIP する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { createBpeModel } from "../src/text/bpe.ts";
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
