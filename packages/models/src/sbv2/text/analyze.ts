/**
 * テキスト → SBV2 の中間表現一式（front を回す前後で使う値をまとめて返す）。
 *
 * MUST: 解析は 1 回だけ。`analyzeWithWords` が result（音素・トーン用）と words（word2ph 用）を
 * **同一解析**から返すので、両者は必ず整合する（二重解析は `sum(word2ph) === P` の不変条件を
 * 壊す — 解析は決定的だが、オーバーレイ辞書を渡す経路ができた今は、同じ overlay を通さない
 * 2 回目の解析が別の語割りを返して実際に割れる）。
 *
 * GPU 実行はここに入れない — この層は決定的な純関数に保ち、資産さえあれば単体で検証できる形に
 * する（テストは `packages/models/tests/sbv2_text_test.ts` と
 * `packages/models/tests/sbv2_tone_injection_test.ts`）。
 */

import { analyzeWithWords, type JtdDictionary, type OverlayDictionary } from "@hdae/yomi";
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

/** 1 回の解析に効かせる差し替え席（省略時は辞書そのまま・解析どおりのトーン）。 */
export type Sbv2AnalyzeOptions = {
  /** 修正辞書（読み・アクセントの差し替え）。解決済みのものを渡す。 */
  readonly overlay?: OverlayDictionary;
  /** トーンの直接指定（0/1 の生値・解析の `phones` と同長）。 */
  readonly givenTone?: readonly number[];
};

/**
 * トーンの直接指定を検める門。**差し替え点はここ 1 箇所**（解析の tones を置き換えてから
 * `phonesTonesToModelIds` へ渡す）。
 *
 * MUST: 長さと値域を**呼び出し側の生値のまま**見る。`toneStart` を足した後で見ると、0/1 以外の
 * 値でも記号表の範囲には収まってしまい（例外が出ない）、別の tone 行を引いたまま音だけが崩れる。
 *
 * NOTE: `export` は門を直接叩くテストのため（この関数へ届く経路は辞書 19MB を要する
 * {@link analyzeSbv2Text} しかない）。公開面（`mod.ts` / `sbv2.ts`）には出さない（ADR 0008）。
 */
export const resolveGivenTone = (
  analyzed: readonly number[],
  given: readonly number[] | undefined,
): readonly number[] => {
  if (given === undefined) return analyzed;
  if (given.length !== analyzed.length) {
    throw new Error(
      `givenTone の長さ ${given.length} が解析の音素数 ${analyzed.length} と違う` +
        "（add_blank 前・両端 PAD 込みの長さで与える — analyzeProsody が返す tones と同じ形）",
    );
  }
  for (const [index, tone] of given.entries()) {
    if (tone !== 0 && tone !== 1) {
      throw new Error(`givenTone[${index}] = ${tone} が 0/1 でない（生の 2 値トーンで与える）`);
    }
  }
  return given;
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
  options: Sbv2AnalyzeOptions = {},
): Sbv2TextAnalysis => {
  const { result, words } = analyzeWithWords(dictionary, text, options.overlay);
  const { phones, tones: analyzed } = toSbv2PhoneTone(result, rules.pad);
  // トーンだけを差し替える（音素列・語割りは解析のまま — bertText / word2ph はトーンを読まない）。
  const tones = resolveGivenTone(analyzed, options.givenTone);
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
