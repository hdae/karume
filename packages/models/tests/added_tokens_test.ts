// 追加語彙の切り出し（`src/text/added-tokens.ts`）の照合規則と計算量。GPU も実資産も要らない。
//
// 縛るのは 2 つ:
//
// - **照合規則**: leftmost-longest（位置が同じなら長い方・位置が違えば左）。期待側は「各位置で
//   全候補を総当たりして最長を採る」素朴な形をこのファイルに書き下し、共通接頭辞の多い合成語彙で
//   乱択の入力と突き合わせる（索引の持ち方を変えても規則が割れないことの門）。
// - **計算量**: 全候補が同じ文字（`<`）で始まる語彙でも、照合は入力長 × 最長の候補長に収まる。
//   先頭 1 文字で束ねるだけだと、本文の `<` ごとに全候補を走査する。

import { assert, assertEquals } from "@std/assert";
import { createAddedTokenMatcher, splitAddedTokens } from "../src/text/added-tokens.ts";

type Chunk = { text: string; added: boolean };

/** 期待側: 各位置で全候補を総当たりして最長を採る（実装と別の形）。 */
const referenceSplit = (text: string, added: readonly string[]): Chunk[] => {
  const out: Chunk[] = [];
  let buffer = "";
  let i = 0;
  while (i < text.length) {
    let hit = "";
    for (const token of added) {
      if (token.length > hit.length && text.startsWith(token, i)) hit = token;
    }
    if (hit === "") {
      buffer += text[i];
      i += 1;
      continue;
    }
    if (buffer !== "") out.push({ text: buffer, added: false });
    buffer = "";
    out.push({ text: hit, added: true });
    i += hit.length;
  }
  if (buffer !== "") out.push({ text: buffer, added: false });
  return out;
};

/** 決定的な擬似乱数（`[0, bound)` の整数）。 */
const randomIndex = (() => {
  let state = 20261010;
  return (bound: number): number => {
    // 32 bit の LCG（`Math.imul` — 素の乗算は 2^53 を超えて下位ビットが潰れる）。下位ビットは
    // 周期が短いので上位 16 bit を使う。
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state >>> 16) % bound;
  };
})();

Deno.test("追加語彙の切り出し: 共通接頭辞の多い語彙でも leftmost-longest が総当たりと一致する", () => {
  // 接頭辞が入れ子（`<a` ⊂ `<ab` ⊂ `<abc>`）・途中まで一致して外れる候補・BMP 外の文字を含む。
  const added = ["<a", "<ab", "<abc>", "<b>", "<", "ab>", "<😀>", "c<a"];
  const alphabet = ["<", "a", "b", "c", ">", "😀", "x"];
  const matcher = createAddedTokenMatcher(added);
  let addedChunks = 0;
  for (let sample = 0; sample < 5_000; sample += 1) {
    const length = randomIndex(16);
    const text = Array.from({ length }, () => alphabet[randomIndex(alphabet.length)]).join("");
    const expected = referenceSplit(text, added);
    addedChunks += expected.filter((chunk) => chunk.added).length;
    assertEquals(matcher.split(text), expected, JSON.stringify(text));
    assertEquals(splitAddedTokens(text, added), expected, JSON.stringify(text));
  }
  // 乱択が切り出す側を実際に踏んでいること（全部が素通しだと比較が恒真になる）。
  assert(addedChunks >= 1_000, `切り出した断片が ${addedChunks} 件しかない`);
});

Deno.test("追加語彙の切り出し: 全候補が同じ文字で始まっても照合は入力長に比例する", () => {
  // EmbeddingGemma 級（6,400 本）の追加語彙が全て `<` で始まる形。先頭 1 文字で束ねるだけだと
  // 本文の `<` ごとに 6,400 本へ startsWith を回す（`<` × 65,536 で数秒）。
  const added = Array.from({ length: 6_400 }, (_, index) => `<unused${index}>`);
  const matcher = createAddedTokenMatcher(added);
  const text = "<".repeat(65_536);
  const started = performance.now();
  const chunks = matcher.split(text);
  const elapsed = performance.now() - started;
  assertEquals(chunks, [{ text, added: false }]);
  assert(elapsed < 500, `\`<\` × ${text.length} の照合に ${elapsed.toFixed(0)} ms かかった`);
});
