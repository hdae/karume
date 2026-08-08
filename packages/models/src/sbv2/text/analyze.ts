/**
 * テキスト → SBV2 の中間表現一式（front を回す前後で使う値をまとめて返す）。
 *
 * MUST: 解析は 1 回だけ。`analyzeWithWords` が result（音素・トーン用）と words（word2ph 用）を
 * **同一解析**から返すので、両者は必ず整合する（二重解析は `sum(word2ph) === P` の不変条件を
 * 壊す — 解析は決定的だが、将来オーバーレイ辞書等の状態が入ると割れる）。
 *
 * GPU 実行はここに入れない — この層は決定的な純関数に保ち、資産さえあれば単体で検証できる形に
 * する（テストは `packages/models/tests/sbv2_text_test.ts`）。
 */

import { analyzeWithWords, type JtdDictionary } from "@hdae/yomi";
import type { DebertaTokenizer } from "./tokenizer.ts";
import { toBertText } from "./bert-text.ts";
import { toSbv2PhoneTone } from "./phone-tone.ts";
import { buildBaseWord2ph } from "./word2ph.ts";
import {
  addBlankWord2ph,
  type JpExtraRules,
  type ModelIdSequences,
  phonesTonesToModelIds,
} from "./symbols.ts";

export type Sbv2TextAnalysis = {
  /** given_phone（両端 PAD 込み、add_blank 前）。 */
  readonly phones: readonly string[];
  /** given_tone（0/1、phones と同長）。 */
  readonly tones: readonly number[];
  /** DeBERTa 入力テキスト（参照実装の sep_text 連結相当）。 */
  readonly bertText: string;
  /** add_blank 前の word2ph（両端の番兵込み）。 */
  readonly baseWord2ph: readonly number[];
  /** DeBERTa の input_ids（`[CLS]`…`[SEP]`）。長さ === word2ph.length。 */
  readonly inputIds: readonly number[];
  /** add_blank 後の word2ph（BERT 特徴の音素レベル展開に使う）。 */
  readonly word2ph: readonly number[];
  /** add_blank 後の front 入力 ID 列。 */
  readonly ids: ModelIdSequences;
};

/**
 * テキスト 1 本を SBV2 の中間表現へ落とす（GPU 実行なし）。
 * 呼び出し側は `inputIds` で DeBERTa を回し、`word2ph` で BERT 特徴を音素レベルへ展開する。
 */
export const analyzeSbv2Text = (
  dictionary: JtdDictionary,
  text: string,
  tokenizer: DebertaTokenizer,
  rules: JpExtraRules,
): Sbv2TextAnalysis => {
  const { result, words } = analyzeWithWords(dictionary, text);
  const { phones, tones } = toSbv2PhoneTone(result, rules.pad);
  const bertText = toBertText(words, rules.punctuations);
  const baseWord2ph = buildBaseWord2ph(words, tokenizer, phones.length);
  const inputIds = tokenizer.encode(bertText);
  // 参照実装 bert_feature.py の `assert len(word2ph) == len(text) + 2` と同じ検査。
  // ここで落とせば、DeBERTa を回した後の shape 不一致（原因が遠い失敗）にならない。
  if (inputIds.length !== baseWord2ph.length) {
    throw new Error(
      `input_ids 長(${inputIds.length}) !== word2ph 長(${baseWord2ph.length})` +
        `（bertText=${JSON.stringify(bertText)}）`,
    );
  }
  return {
    phones,
    tones,
    bertText,
    baseWord2ph,
    inputIds,
    word2ph: addBlankWord2ph(baseWord2ph),
    ids: phonesTonesToModelIds(rules, phones, tones),
  };
};
