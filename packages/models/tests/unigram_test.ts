// Unigram 共通層（src/text/unigram.ts）の振る舞い検証。
//
// 実資産には依存しない — 合成した小語彙で「格子探索の同点規則」「未知の融合」「byte_fallback の
// 展開」を直接見る。実資産との突合（id 列が Python と一致するか）はファミリ側のパリティテスト
// （anima_tokenizer_test.ts）の担当で、ここは**規則そのもの**を落とせる形に置く。

import { assertEquals } from "@std/assert";
import { toCodePoints } from "../src/anima/text/code-ranges.ts";
import { type UnigramModel, unigramTokenize } from "../src/text/unigram.ts";

const UNK_ID = 2;
/** byte_fallback の `<0x00>` に当たる id。実資産と重ならない値にして展開先を見分ける。 */
const BYTE_BASE = 100;

/** `[トークン, id, スコア]` の並びから模型を組む。minScore / maxTokenLength は語彙全体から導く。 */
const buildModel = (
  vocab: readonly (readonly [string, number, number])[],
  options: { readonly byteBaseId?: number } = {},
): UnigramModel => ({
  vocab: new Map(vocab.map(([token, id, score]) => [token, { id, score }])),
  minScore: vocab.length === 0 ? 0 : Math.min(...vocab.map(([, , score]) => score)),
  maxTokenLength: Math.max(1, ...vocab.map(([token]) => toCodePoints(token).length)),
  unkId: UNK_ID,
  ...options,
});

const tokenize = (model: UnigramModel, text: string): number[] =>
  unigramTokenize(model, toCodePoints(text));

Deno.test("格子探索: 経路はスコアの和で決まる", async (t) => {
  await t.step("同点なら長い断片が勝つ（比較は厳密な `>`）", () => {
    // a(-1) + b(-1) と ab(-2) がちょうど同点。正本（tokenizers の Unigram）は同じ位置で終わる
    // ノードを開始位置の昇順に積み、同点なら先に積まれた = 長い方を残す。`>=` にすると
    // [4, 5] へ黙って変わる。
    const model = buildModel([["a", 4, -1], ["b", 5, -1], ["ab", 6, -2]]);
    assertEquals(tokenize(model, "ab"), [6]);
  });

  await t.step("同点でなければスコアの高い経路が勝つ（常に最長ではない）", () => {
    // 同じ形で ab だけを -3 に落とすと、短い 2 個の方が和で上回る。上の規則が
    // 「いつも最長を返すだけ」に退化していないことをここで捕まえる。
    const model = buildModel([["a", 4, -1], ["b", 5, -1], ["ab", 6, -3]]);
    assertEquals(tokenize(model, "ab"), [4, 5]);
  });
});

Deno.test("未知の扱い: byte_fallback なし", async (t) => {
  const model = buildModel([["a", 4, -1]]);

  await t.step("連続する未知は 1 つの unk に融合される（fuse_unk）", () => {
    // 融合しないと未知が 1 文字ずつ並び [4, 2, 2, 4] になる。
    assertEquals(tokenize(model, "axya"), [4, UNK_ID, 4]);
  });

  await t.step("既知を挟んだ未知は別々の unk になる（融合は連続したものだけ）", () => {
    assertEquals(tokenize(model, "xay"), [UNK_ID, 4, UNK_ID]);
  });

  await t.step("全文が未知でも unk 1 個", () => {
    assertEquals(tokenize(model, "xyz"), [UNK_ID]);
  });
});

Deno.test("未知の扱い: byte_fallback あり", async (t) => {
  const model = buildModel([["a", 4, -1]], { byteBaseId: BYTE_BASE });

  await t.step("3 バイト文字は UTF-8 の 3 バイトぶんの id に展開される", () => {
    // U+3042「あ」= E3 81 82 → 100 + 各バイト。
    assertEquals(tokenize(model, "あ"), [327, 229, 230]);
  });

  await t.step("4 バイト文字（BMP 外）も同じ規則で 4 id になる", () => {
    // U+1F600「😀」= F0 9F 98 80。サロゲート対ではなくコードポイントで数えていないと
    // 3 id や不正なバイトになる。
    assertEquals(tokenize(model, "😀"), [340, 259, 252, 228]);
  });

  await t.step("融合された未知断片は全体がバイト列になる（unk へは落ちない）", () => {
    assertEquals(tokenize(model, "aあ😀a"), [4, 327, 229, 230, 340, 259, 252, 228, 4]);
  });

  await t.step("語彙にある断片はバイト展開されない", () => {
    const withKana = buildModel([["a", 4, -1], ["あ", 7, -1]], { byteBaseId: BYTE_BASE });
    assertEquals(tokenize(withKana, "aあ"), [4, 7]);
  });
});

Deno.test("縮退: 空入力と空語彙", async (t) => {
  await t.step("空文字列は空の id 列（unk も出ない）", () => {
    assertEquals(tokenize(buildModel([["a", 4, -1]]), ""), []);
    assertEquals(tokenize(buildModel([], { byteBaseId: BYTE_BASE }), ""), []);
  });

  await t.step("語彙が空なら全文が 1 つの unk になる", () => {
    assertEquals(tokenize(buildModel([]), "abc"), [UNK_ID]);
  });

  await t.step("語彙が空 + byte_fallback なら全文がバイト列になる", () => {
    // "abc" = 61 62 63 → 100 + 各バイト。
    assertEquals(tokenize(buildModel([], { byteBaseId: BYTE_BASE }), "abc"), [197, 198, 199]);
  });
});
