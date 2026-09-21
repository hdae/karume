// 資産 JSON の値域門（`src/text/asset-gates.ts`）のうち、**家族横断で共有している id 系**の
// 門を直接叩く。GPU も実資産も要らない。
//
// ここを単体で縛る理由: これらの門が守るのは「例外にならず別のトークン / 別の行を指す」形
// （沈黙誤値）で、家族側の e2e からは**壊しても緑のまま**に見える位置にある。anima と irodori
// が同じ 1 本を使うようになったので、規律の正本もここ 1 本に置く。
//
// NOTE: 文字列・数値・オブジェクトの素の型検査（`asString` / `asNumber` / `asRecord`）は id 系の
// 門の土台なので、id の検査が「型検査を通り抜けた値」だけを見ていることが判る最小限だけ書く。

import { assertEquals, assertThrows } from "@std/assert";
import {
  asId,
  asNumber,
  asRecord,
  asString,
  asVocabId,
  parseAddedTokens,
} from "../src/text/asset-gates.ts";

Deno.test("asRecord / asString / asNumber: 型が違う値は読まずに落とす", () => {
  assertEquals(asRecord({ a: 1 }, "表"), { a: 1 });
  // `typeof null === "object"` を通さない（`null.foo` ではなく門で落とす）。
  assertThrows(() => asRecord(null, "表"), Error, "表: オブジェクトでない");
  assertThrows(() => asRecord("{}", "表"), Error, "表: オブジェクトでない");

  assertEquals(asString("abc", "欄"), "abc");
  assertThrows(() => asString(12, "欄"), Error, "欄: 文字列でない");

  assertEquals(asNumber(12, "欄"), 12);
  assertThrows(() => asNumber("12", "欄"), Error, "欄: 数値でない");
});

Deno.test("asId: `Int32Array` へ書くと黙って別の id になる値を全て落とす", () => {
  // 非整数は `Int32Array` が切り捨てる = 別のトークンを指す（門が無ければ例外は出ない）。
  assertEquals(Int32Array.of(1.5)[0], 1);
  assertThrows(() => asId(1.5, "unkId"), Error, "unkId: トークン id が 0..2147483647 の整数でない");

  // i32 の範囲外は wrap する（2147483648 → -2147483648）。
  assertEquals(Int32Array.of(2147483648)[0], -2147483648);
  assertThrows(() => asId(2147483648, "eosId"), Error, "トークン id が 0..2147483647 の整数でない");

  assertThrows(() => asId(-1, "eosId"), Error, "トークン id が 0..2147483647 の整数でない");
  assertThrows(() => asId("3", "eosId"), Error, "eosId: 数値でない");

  // 両端は受理集合の中（0 と i32 の上限）。
  assertEquals(asId(0, "unkId"), 0);
  assertEquals(asId(2147483647, "unkId"), 2147483647);
});

Deno.test("asVocabId: 語彙の行数以上の id は落とす（gather が別の行を引く前に）", () => {
  assertEquals(asVocabId(2, "unkId", 3), 2);
  assertThrows(() => asVocabId(3, "unkId", 3), Error, "unkId: トークン id 3 が語彙の行数 3 以上");
  // 土台の {@link asId} の規律もそのまま効く。
  assertThrows(() => asVocabId(1.5, "unkId", 3), Error, "整数でない");
});

Deno.test("parseAddedTokens: 対の列を写像にし、重複鍵は後勝ちにせず落とす", () => {
  const parsed = parseAddedTokens([["<s>", 1], ["</s>", 2]], "added");
  assertEquals([...parsed], [["<s>", 1], ["</s>", 2]]);

  // 重複鍵を `Map.set` の後勝ちで飲み込むと、分割規則だけが静かに変わる。
  assertThrows(
    () => parseAddedTokens([["<s>", 1], ["<s>", 9]], "added"),
    Error,
    'added: 鍵 "<s>" が重複している',
  );

  assertThrows(() => parseAddedTokens({ "<s>": 1 }, "added"), Error, "added: 配列でない");
  assertThrows(() => parseAddedTokens([["<s>"]], "added"), Error, "added: [文字列, id] でない");
  assertThrows(() => parseAddedTokens([[1, 1]], "added"), Error, "added: 文字列でない");
  assertThrows(() => parseAddedTokens([["<s>", 1.5]], "added"), Error, "added: トークン id が");
});

Deno.test("parseAddedTokens: 追加トークンは語彙の行数で縛らない（語彙表の外へ採番される形）", () => {
  // Qwen2 の実資産がこの形（語彙 151643 行に対し追加トークンは 151643..151668）。
  const parsed = parseAddedTokens([["<|im_start|>", 151644]], "added");
  assertEquals(parsed.get("<|im_start|>"), 151644);
});
