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
import type { Utterance, Word } from "./utterance.ts";
import { buildBaseWord2ph } from "./word2ph.ts";
import { InputError } from "../errors.ts";
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
export const assertWordPhones = (phones: readonly string[], words: readonly Word[]): void => {
  const aligned = words.flatMap((word) => word.phones);
  // The PADs at both ends are not "words", so they do not appear in the word alignment.
  const derived = phones.slice(1, -1);
  if (aligned.length !== derived.length) {
    throw new InputError(
      `The phoneme count ${derived.length} built from moras differs from the phoneme count ${aligned.length} of words` +
        " (pass moras and words taken from the same analysis — to change the reading, redo the analysis)",
    );
  }
  for (const [index, phone] of derived.entries()) {
    if (phone !== aligned[index]) {
      throw new InputError(
        `Phoneme[${index}] = ${JSON.stringify(phone)} built from moras differs from words'` +
          ` ${JSON.stringify(aligned[index])} (only tone may be changed —` +
          " edits that change phonemes (rewriting vowel / consonant, adding or removing symbols) are not accepted)",
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
  utterance: Utterance,
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
    throw new InputError(
      `input_ids length(${inputIds.length}) !== word2ph length(${baseWord2ph.length})` +
        ` (bertText=${JSON.stringify(bertText)})`,
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
