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
//
// 実資産（`outputs/series/gemma4-e2b-tokenizer/`）が在れば**全語彙**でも同じ id 列になること
// を併せて見る（部分集合と full の食い違いを塞ぐ門）。無い環境では SKIP する。

import { assertEquals, assertThrows } from "@std/assert";
import { createBpeModel } from "../src/text/bpe.ts";
import { type GemmaTokenizerAssets, parseGemmaTokenizerAsset } from "../src/gemma/text/asset.ts";
import { GemmaTokenizer } from "../src/gemma/text/tokenizer.ts";
import {
  type Gemma4ChatMessage,
  gemma4ChatPrompt,
  gemma4StopTokens,
  renderGemma4Chat,
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
