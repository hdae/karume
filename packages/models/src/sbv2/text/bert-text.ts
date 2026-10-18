/**
 * Assembly of the DeBERTa input text (the equivalent of the reference implementation's
 * `"".join(text_to_sep_kata(...)[0])`).
 *
 * The reference implementation maps punctuation to its normal form (! ? … , . ' -) with
 * `replace_punctuation` in `normalize_text` before feeding BERT. The word alignment keeps the **raw
 * single character** (。 ！ etc.) in the surface of a symbol element, so concatenating as is shifts
 * the token IDs DeBERTa sees (e.g. "。" vs "."). Only symbol elements are replaced by their normal
 * form (= the element's only phone) before concatenation.
 *
 * NOTE: this replacement does not change the token count per word (the normal form and the raw form
 * split into the same number of tokens under the character tokenizer). So the consistency with
 * word2ph (based on Σ tokenize(surface)) does not break.
 *
 * MUST: the bert input text is always built with this function (hand-written surface concatenation
 * creates a second path to the consistency with word2ph and hides token-count mismatches until run
 * time).
 */

import type { Sbv2Word } from "./utterance.ts";

/**
 * Whether this is a symbol element (a word whose phones are one normal-form punctuation mark).
 * Kana phonemes and normal-form characters do not intersect.
 */
const isPunctuationWord = (word: Sbv2Word, punctuations: ReadonlySet<string>): boolean =>
  word.phones.length === 1 && punctuations.has(word.phones[0]);

/**
 * Builds the DeBERTa input text from the word alignment. Symbol elements are replaced by the
 * normal-form punctuation; everything else concatenates the surface as is.
 */
export const toBertText = (words: readonly Sbv2Word[], punctuations: ReadonlySet<string>): string =>
  words
    .map((word) => (isPunctuationWord(word, punctuations) ? word.phones[0] : word.surface))
    .join("");
