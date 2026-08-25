/**
 * 発話（モーラ列）→ SBV2 の given_phone / given_tone。
 *
 * MUST: 音素・トーンを組む経路は**ここ 1 本だけ**。二経路化すると、経路ごとに梱包規則が
 * ズレる（長さは合うのに音だけ崩れる）。
 *
 * SBV2 の音素・トーン規約:
 *
 * - 音素列は先頭・末尾に PAD（記号表の `pad`、tone 0）を必ず含む完全な列。
 * - 1 モーラは `consonant` があれば `[consonant, vowel]`、無ければ `[vowel]`。促音は `"q"`
 *   1 個・撥音は `"N"` 1 個（`Sbv2Mora.vowel` が音素そのものを持つ）。
 * - トーンは 0/1 の 2 値で、子音・母音とも同一トーン（モーラ単位の値を音素へ配る）。
 * - 記号はテキストに実在した句読点（正規形）だけを tone 0 で出す。実在しない記号は
 *   合成しない（参照実装 g2p と同方針）。
 *
 * MUST: 出力の非 PAD 部分は語アライメントと完全一致する（`words.flatMap(w => w.phones)` ===
 * `leadingPunctuations` + Σ（モーラ音素 + そのモーラの punctuations））— word2ph の
 * `sum(word2ph) === phones.length` がこの一致に依存する。突合は `model-input.ts` の
 * `assertWordPhones` が持つ。
 */

import { Sbv2InputError } from "../errors.ts";
import type { Sbv2Utterance } from "./utterance.ts";

type Sbv2PhoneTone = {
  /** given_phone（両端 PAD 込み、add_blank 前）。 */
  readonly phones: readonly string[];
  /** given_tone（0/1、phones と同長）。 */
  readonly tones: readonly number[];
};

/**
 * 発話を given_phone / given_tone へ変換する（位置ごとに対応、tone は音素単位）。
 *
 * MUST: `tone` の値域を**ここで**見る。型は 0|1 だが JS からの呼び出しには効かず、`toneStart`
 * を足した後では 0/1 以外でも記号表の範囲に収まってしまう（例外が出ないまま別の tone 行を
 * 引く）。
 */
export const toSbv2PhoneTone = (utterance: Sbv2Utterance, pad: string): Sbv2PhoneTone => {
  const phones: string[] = [pad];
  const tones: number[] = [0];

  const pushPunctuations = (punctuations: readonly string[]): void => {
    for (const punctuation of punctuations) {
      phones.push(punctuation);
      tones.push(0);
    }
  };

  // 先頭モーラより前の実在記号（記号だけの入力ではモーラが無く全てここに入る）。
  pushPunctuations(utterance.leadingPunctuations);

  for (const [index, mora] of utterance.moras.entries()) {
    const { tone } = mora;
    if (tone !== 0 && tone !== 1) {
      throw new Sbv2InputError(
        `moras[${index}]（${mora.kana}）の tone = ${tone} が 0/1 でない` +
          "（生の 2 値トーンで与える）",
      );
    }
    // 1 モーラを [consonant, vowel] に展開したとき、子音・母音とも同一トーンを振る。
    if (mora.consonant !== undefined) {
      phones.push(mora.consonant);
      tones.push(tone);
    }
    phones.push(mora.vowel);
    tones.push(tone);
    // モーラ直後にテキスト上実在した記号（正規形・出現順）。
    if (mora.punctuations !== undefined) pushPunctuations(mora.punctuations);
  }

  phones.push(pad);
  tones.push(0);

  if (phones.length !== tones.length) {
    throw new Error(`phones(${phones.length}) と tones(${tones.length}) の長さが不一致`);
  }
  return { phones, tones };
};
