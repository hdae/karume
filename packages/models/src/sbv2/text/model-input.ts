/**
 * Utterance → the full set of SBV2 intermediate representations (returns together the values used
 * before and after running front).
 *
 * MUST: the mora sequence (phonemes, tones) and the word alignment (word2ph, BERT input text)
 * **must come from the same analysis**. The karume side has no analyser, so that consistency is
 * checked at the entry by two checks, {@link assertWordPhones} and `sum(word2ph) === P`
 * (`buildBaseWord2ph`) — letting it through gives a silent wrong value where "phonemes and tones
 * reach front shifted by one".
 *
 * GPU execution does not go here — this layer stays a deterministic pure function, in a form that
 * can be verified alone given the assets (tests are `packages/models/tests/sbv2_text_test.ts` and
 * `packages/models/tests/sbv2_tone_injection_test.ts`).
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
  /** given_phone (including PAD at both ends, before add_blank). */
  readonly phones: readonly string[];
  /** given_tone (0/1, same length as phones). */
  readonly tones: readonly number[];
  /**
   * DeBERTa input text (the equivalent of the reference implementation's sep_text concatenation).
   */
  readonly bertText: string;
  /** word2ph before add_blank (including the sentinels at both ends). */
  readonly baseWord2ph: readonly number[];
  /** DeBERTa's input_ids (`[CLS]`…`[SEP]`). Length === word2ph.length. */
  readonly inputIds: readonly number[];
  /** word2ph after add_blank (used to expand the BERT features to phoneme level). */
  readonly word2ph: readonly number[];
  /** front's input ID sequences after add_blank. */
  readonly ids: ModelIdSequences;
};

/**
 * Check that the phoneme sequence built from the mora sequence matches the phoneme content of the
 * word alignment. **The core of this entry check** (it looks not only at the length but at the
 * content position by position).
 *
 * Looking only at the length lets mora rereadings and swapped symbols through — the length matches
 * but only the sound breaks, and only the word2ph distribution (phonemes per word) lands on a
 * different word. Looking position by position stops here an utterance that mixes `moras` and
 * `words` taken from different analyses.
 *
 * NOTE: it is `export`ed for tests that call the check directly. It is not on the public surface
 * (`mod.ts` / `sbv2.ts`) (ADR 0008).
 *
 * @param phones the output of {@link toSbv2PhoneTone} (including PAD at both ends).
 * @param words word alignment (without PAD).
 */
export const assertWordPhones = (phones: readonly string[], words: readonly Sbv2Word[]): void => {
  const aligned = words.flatMap((word) => word.phones);
  // The PADs at both ends are not "words", so they do not appear in the word alignment.
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
 * Lowers one utterance to SBV2's intermediate representation (no GPU execution).
 * The caller runs DeBERTa with `inputIds` and expands the BERT features to phoneme level with
 * `word2ph`.
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
  // The same check as `assert len(word2ph) == len(text) + 2` in the reference implementation's
  // bert_feature.py. Failing here avoids a shape mismatch after running DeBERTa (a failure far from
  // its cause).
  if (inputIds.length !== baseWord2ph.length) {
    // Reachable only through an utterance whose words' surfaces were rewritten (a symbol word's
    // normal form yielding a different token count, etc.), so it is input-caused = 400. The caller
    // fixes it by restoring words to what the analysis produced.
    throw new Sbv2InputError(
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
