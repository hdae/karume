/**
 * Utterance (mora sequence) → SBV2's given_phone / given_tone.
 *
 * MUST: **this is the only path** that builds phonemes and tones. With two paths, the packing rules
 * drift per path (the lengths match but only the sound breaks).
 *
 * SBV2's phoneme and tone conventions:
 *
 * - The phoneme sequence is a complete sequence that always includes PAD (the symbol table's `pad`,
 *   tone 0) at the start and the end.
 * - A mora is `[consonant, vowel]` when it has a `consonant`, otherwise `[vowel]`. The geminate is
 *   one `"q"` and the moraic nasal one `"N"` (`Mora.vowel` holds the phoneme itself).
 * - Tone is binary 0/1, and consonant and vowel share the same tone (the per-mora value is spread
 *   to the phonemes).
 * - Symbols are emitted with tone 0 only for punctuation actually present in the text (normal
 *   form). Symbols that are not present are not synthesised (the same policy as the reference
 *   implementation's g2p).
 *
 * MUST: the non-PAD part of the output matches the word alignment exactly
 * (`words.flatMap(w => w.phones)` === `leadingPunctuations` + Σ (mora phonemes + that mora's
 * punctuations)) — word2ph's `sum(word2ph) === phones.length` depends on this match. The
 * comparison is owned by `assertWordPhones` in `model-input.ts`.
 */

import { InputError } from "../errors.ts";
import type { Utterance } from "./utterance.ts";

type Sbv2PhoneTone = {
  /** given_phone (including PAD at both ends, before add_blank). */
  readonly phones: readonly string[];
  /** given_tone (0/1, same length as phones). */
  readonly tones: readonly number[];
};

/**
 * Converts an utterance to given_phone / given_tone (position by position; tone per phoneme).
 *
 * MUST: check the range of `tone` **here**. The type is 0|1 but has no effect on calls from JS, and
 * once `toneStart` is added, values other than 0/1 can still land inside the table's range (looking
 * up a different tone row without an exception).
 */
export const toSbv2PhoneTone = (utterance: Utterance, pad: string): Sbv2PhoneTone => {
  const phones: string[] = [pad];
  const tones: number[] = [0];

  const pushPunctuations = (punctuations: readonly string[]): void => {
    for (const punctuation of punctuations) {
      phones.push(punctuation);
      tones.push(0);
    }
  };

  // Symbols actually present before the first mora (with symbol-only input there are no moras and
  // all of them go here).
  pushPunctuations(utterance.leadingPunctuations);

  for (const [index, mora] of utterance.moras.entries()) {
    const { tone } = mora;
    if (tone !== 0 && tone !== 1) {
      throw new InputError(
        `moras[${index}] (${mora.kana}) has tone = ${tone}, not 0/1` +
          " (give it as a raw binary tone)",
      );
    }
    // When a mora expands to [consonant, vowel], consonant and vowel get the same tone.
    if (mora.consonant !== undefined) {
      phones.push(mora.consonant);
      tones.push(tone);
    }
    phones.push(mora.vowel);
    tones.push(tone);
    // Symbols actually present in the text right after the mora (normal form, in order of
    // appearance).
    if (mora.punctuations !== undefined) pushPunctuations(mora.punctuations);
  }

  phones.push(pad);
  tones.push(0);

  if (phones.length !== tones.length) {
    throw new Error(`phones(${phones.length}) and tones(${tones.length}) differ in length`);
  }
  return { phones, tones };
};
