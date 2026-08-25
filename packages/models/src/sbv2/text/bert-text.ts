/**
 * DeBERTa 入力テキスト（参照実装の `"".join(text_to_sep_kata(...)[0])` 相当）の組み立て。
 *
 * 参照実装は `normalize_text` の `replace_punctuation` で句読点を正規形（! ? … , . ' -）へ
 * 写してから BERT に入れる。語アライメントは記号要素の surface に**生の 1 文字**
 * （。 ！ 等）を保持するため、そのまま連結すると DeBERTa が見るトークン ID がずれる
 * （例: "。" vs "."）。記号要素だけ正規形（= その要素の唯一の phone）へ置き換えて連結する。
 *
 * NOTE: この置換で語ごとのトークン数は変わらない（正規形も生形も文字トークナイザで同数に
 * 割れる）。そのため word2ph（Σ tokenize(surface) ベース）との整合は崩れない。
 *
 * MUST: bert 入力テキストは必ずこの関数で組む（手書きの surface 連結は word2ph との整合を
 * 二経路化し、トークン数不一致を実行時まで隠す）。
 */

import type { Sbv2Word } from "./utterance.ts";

/** 記号要素（phones が正規形句読点 1 個だけの語）か。かな語の音素と正規形字母は交差しない。 */
const isPunctuationWord = (word: Sbv2Word, punctuations: ReadonlySet<string>): boolean =>
  word.phones.length === 1 && punctuations.has(word.phones[0]);

/**
 * 語アライメントから DeBERTa 入力テキストを組む。記号要素は正規形句読点へ置き換え、
 * それ以外は surface をそのまま連結する。
 */
export const toBertText = (
  words: readonly Sbv2Word[],
  punctuations: ReadonlySet<string>,
): string =>
  words
    .map((word) => (isPunctuationWord(word, punctuations) ? word.phones[0] : word.surface))
    .join("");
