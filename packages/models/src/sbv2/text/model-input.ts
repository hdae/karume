/**
 * 発話 → SBV2 の中間表現一式（front を回す前後で使う値をまとめて返す）。
 *
 * MUST: モーラ列（音素・トーン）と語アライメント（word2ph・BERT 入力テキスト）は**同一の
 * 解析から採られたもの**でなければならない。karume 側は解析器を持たないので、その整合は
 * {@link assertWordPhones} と `sum(word2ph) === P`（`buildBaseWord2ph`）の 2 本で入口に
 * 門を張って検める — 通せば「音素とトーンが 1 個ずれたまま front まで届く」沈黙誤値になる。
 *
 * GPU 実行はここに入れない — この層は決定的な純関数に保ち、資産さえあれば単体で検証できる形に
 * する（テストは `packages/models/tests/sbv2_text_test.ts` と
 * `packages/models/tests/sbv2_tone_injection_test.ts`）。
 */

import type { DebertaTokenizer } from "./tokenizer.ts";
import { toBertText } from "./bert-text.ts";
import { toSbv2PhoneTone } from "./phone-tone.ts";
import type { Sbv2Utterance, Sbv2Word } from "./utterance.ts";
import { buildBaseWord2ph } from "./word2ph.ts";
import { Sbv2InputError } from "../errors.ts";
import {
  addBlankWord2ph,
  type JpExtraRules,
  type ModelIdSequences,
  phonesTonesToModelIds,
} from "./symbols.ts";

export type Sbv2ModelInput = {
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
 * モーラ列から組んだ音素列が、語アライメントの音素内容と一致することを検める門。
 * **この席の本体**（長さだけでなく位置ごとの内容まで見る）。
 *
 * 長さだけを見ると、モーラの読み替え・記号の入れ替えが素通りする — 長さは合うのに音だけが
 * 崩れ、word2ph の分配（語ごとの音素数）だけが別の語に付く。位置ごとに見れば、別の解析から
 * 採った `moras` と `words` を混ぜた発話がここで止まる。
 *
 * NOTE: `export` は門を直接叩くテストのため。公開面（`mod.ts` / `sbv2.ts`）には出さない
 * （ADR 0008）。
 *
 * @param phones {@link toSbv2PhoneTone} の出力（両端 PAD 込み）。
 * @param words 語アライメント（PAD を含まない）。
 */
export const assertWordPhones = (
  phones: readonly string[],
  words: readonly Sbv2Word[],
): void => {
  const aligned = words.flatMap((word) => word.phones);
  // 両端 PAD は「語」ではないので語アライメントには現れない。
  const derived = phones.slice(1, -1);
  if (aligned.length !== derived.length) {
    throw new Sbv2InputError(
      `moras から組んだ音素数 ${derived.length} が words の音素数 ${aligned.length} と違う` +
        "（moras と words は同じ解析から採ったものを渡す — 読みを変えたいなら解析からやり直す）",
    );
  }
  for (const [index, phone] of derived.entries()) {
    if (phone !== aligned[index]) {
      throw new Sbv2InputError(
        `moras から組んだ音素[${index}] = ${JSON.stringify(phone)} が words の` +
          ` ${JSON.stringify(aligned[index])} と違う（動かせるのは tone だけ —` +
          "音素を変える編集〈vowel / consonant の書き換え・記号の増減〉は受け付けない）",
      );
    }
  }
};

/**
 * 発話 1 本を SBV2 の中間表現へ落とす（GPU 実行なし）。
 * 呼び出し側は `inputIds` で DeBERTa を回し、`word2ph` で BERT 特徴を音素レベルへ展開する。
 */
export const buildSbv2ModelInput = (
  utterance: Sbv2Utterance,
  tokenizer: DebertaTokenizer,
  rules: JpExtraRules,
): Sbv2ModelInput => {
  const { phones, tones } = toSbv2PhoneTone(utterance, rules.pad);
  assertWordPhones(phones, utterance.words);
  const bertText = toBertText(utterance.words, rules.punctuations);
  const baseWord2ph = buildBaseWord2ph(utterance.words, tokenizer, phones.length);
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
