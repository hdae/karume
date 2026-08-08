/**
 * base word2ph（add_blank 前）の生成。BERT テキスト層の中核。
 *
 * word2ph = 「DeBERTa の各トークン（= 文字）に音素を何個割り当てるか」。厳密アライメントは
 * 不可能（熟字訓）なので、参照実装は「単語ごとに音素数を文字数へ均等分配」する近似を採る
 * （`g2p.py` の `__distribute_phone`）。
 *
 * スコープ: ここが作るのは add_blank **前**の base word2ph。
 *
 * - `sum(base word2ph) === given_phone 長`（両端 PAD と句読点を含む `toSbv2PhoneTone` の出力長）
 * - `len(base word2ph) === Σ tokenize(語 surface) + 2`
 *
 * add_blank の `*2` / `[0]+=1` は `symbols.ts` の `addBlankWord2ph` の責務。
 */

import type { WordPhones } from "@hdae/yomi";
import type { DebertaTokenizer } from "./tokenizer.ts";

/**
 * n 個の音素を `wordLength` 文字へ均等分配する（`g2p.py __distribute_phone` の忠実移植）。
 * 左から右へ 1 ずつ、常に「最小の文字」へ足していく。
 */
export const distributePhone = (phoneCount: number, wordLength: number): number[] => {
  if (!Number.isInteger(wordLength) || wordLength <= 0) {
    throw new Error(`分配先の文字数 ${wordLength} が 1 以上の整数でない`);
  }
  if (!Number.isInteger(phoneCount) || phoneCount < 0) {
    throw new Error(`分配する音素数 ${phoneCount} が 0 以上の整数でない`);
  }
  const perWord = new Array<number>(wordLength).fill(0);
  for (let k = 0; k < phoneCount; k += 1) {
    // 最小値のインデックス（同値なら最左）。
    let minIndex = 0;
    for (let i = 1; i < wordLength; i += 1) {
      if (perWord[i] < perWord[minIndex]) minIndex = i;
    }
    perWord[minIndex] += 1;
  }
  return perWord;
};

/**
 * 語アライメントと DeBERTa トークナイザから base word2ph を作る。
 *
 * 不変条件（破れたら throw）:
 *
 * - `word2ph.length === Σ tokenize(surface) + 2`（`bert_feature.py` の
 *   `len(word2ph) == len(text)+2`。text は語 surface の連結で、文字トークナイザは加算的
 *   なので `Σ tokenize(word) === tokenize(joined)` が成り立つ）。
 * - `sum(word2ph) === givenPhoneLength`（`g2p.py` の `len(phones) == sum(word2ph)`）。
 *
 * NOTE: 参照実装は句読点語だけトークナイズを省いて 1 文字扱いにするが、こちらは全語を
 * トークナイズする。参照側は `normalize_text` が "…" を "..." へ潰しているので実質同値で、
 * yomi は "…" を 1 記号のまま残すため、トークナイズ側に揃えないと DeBERTa の実トークン数
 * （NFKC で "..." の 3 個）と長さが合わなくなる。
 *
 * @param words `analyzeWithWords(...).words`（両端 PAD を含まない）。
 * @param tokenizer DeBERTa 文字トークナイザ（各語 surface のトークン数算出に使う）。
 * @param givenPhoneLength `toSbv2PhoneTone(...).phones.length`（両端 PAD 込み）。
 */
export const buildBaseWord2ph = (
  words: readonly WordPhones[],
  tokenizer: DebertaTokenizer,
  givenPhoneLength: number,
): number[] => {
  const word2ph: number[] = [1]; // 先頭 PAD 用の番兵。
  let tokenTotal = 0;
  for (const word of words) {
    const wordLength = tokenizer.tokenize(word.surface).length;
    if (wordLength <= 0) {
      throw new Error(`語 surface が 0 トークンに正規化された: ${JSON.stringify(word.surface)}`);
    }
    tokenTotal += wordLength;
    for (const count of distributePhone(word.phones.length, wordLength)) word2ph.push(count);
  }
  word2ph.push(1); // 末尾 PAD 用の番兵。

  const expectedLength = tokenTotal + 2;
  if (word2ph.length !== expectedLength) {
    throw new Error(
      `word2ph 長 ${word2ph.length} が Σtokenize(surface)+2 (${expectedLength}) と違う`,
    );
  }
  const total = word2ph.reduce((sum, count) => sum + count, 0);
  if (total !== givenPhoneLength) {
    throw new Error(`sum(word2ph) ${total} が given_phone 長 ${givenPhoneLength} と違う`);
  }
  return word2ph;
};
