/**
 * yomi の `FrontendResult` → SBV2 の given_phone / given_tone。
 *
 * 責務境界（@hdae/yomi はモデル非依存）: yomi が提供するのは建材（`moraToPhones` /
 * `moraTones` / `punctuations` / `leadingPunctuations`）までで、SBV2 固有の梱包 —— 両端の
 * PAD、トーンの音素単位割当、実在記号の音素化 —— は呼び出し側（= ここ）で組む。
 *
 * SBV2 の音素・トーン規約:
 *
 * - 音素列は先頭・末尾に PAD（記号表の `pad`、tone 0）を必ず含む完全な列。
 * - 促音は "q" 1 個・撥音は "N" 1 個（`moraToPhones` が畳む）。長音は直前母音に解決済み。
 * - トーンは 0/1 の 2 値。各アクセント句で独立に 0 から立ち上がる（`moraTones`）。
 * - 記号はテキストに実在した句読点（正規形）だけを tone 0 で出す。実在しない記号は
 *   合成しない（参照実装 g2p と同方針）。
 *
 * MUST: 出力の非 PAD 部分は yomi の `wordPhoneAlignment` と完全一致させる
 * （`flatMap(w => w.phones)` === `leadingPunctuations` + Σ 句の（モーラ音素 + punctuations））
 * — word2ph の `sum(word2ph) === phones.length` がこの一致に依存する。
 */

import { type FrontendResult, moraTones, moraToPhones } from "@hdae/yomi";

type Sbv2PhoneTone = {
  /** given_phone（両端 PAD 込み、add_blank 前）。 */
  readonly phones: readonly string[];
  /** given_tone（0/1、phones と同長）。 */
  readonly tones: readonly number[];
};

/** `FrontendResult` を given_phone / given_tone へ変換する（位置ごとに対応、tone は音素単位）。 */
export const toSbv2PhoneTone = (result: FrontendResult, pad: string): Sbv2PhoneTone => {
  const phones: string[] = [pad];
  const tones: number[] = [0];

  // 先頭句より前の実在記号（記号だけの入力は句が作られず全てここに入る）。
  for (const punct of result.leadingPunctuations) {
    phones.push(punct);
    tones.push(0);
  }

  for (const phrase of result.accentPhrases) {
    const perMoraTone = moraTones(phrase.accentNucleus, phrase.moras.length);
    for (const [index, mora] of phrase.moras.entries()) {
      const tone = perMoraTone[index];
      // 1 モーラを [consonant, vowel] に展開したとき、子音・母音とも同一トーンを振る。
      for (const phone of moraToPhones(mora)) {
        phones.push(phone);
        tones.push(tone);
      }
    }
    // 句直後にテキスト上実在した記号（正規形・出現順）。pauseAfter からの合成はしない。
    for (const punct of phrase.punctuations) {
      phones.push(punct);
      tones.push(0);
    }
  }

  phones.push(pad);
  tones.push(0);

  if (phones.length !== tones.length) {
    throw new Error(`phones(${phones.length}) と tones(${tones.length}) の長さが不一致`);
  }
  return { phones, tones };
};
