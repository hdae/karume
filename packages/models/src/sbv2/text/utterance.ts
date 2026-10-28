/**
 * SBV2's input vocabulary (utterance = mora sequence + word alignment) and the conversion from the
 * phrase structure.
 *
 * Responsibility boundary (the re-ruling of 0.6.0): text analysis (dictionary, correction
 * dictionary) is **the caller's responsibility**, and this package accepts only the analysed
 * structure. So every type here is karume-owned and imports no analyser types —
 * {@link Phrases} describes "the shape the upstream analysis result satisfies
 * **structurally**", not a dependency on a particular analyser.
 *
 * MUST: the utterance is **flat** (no nested phrases). A phrase is a structure only for deciding
 * "nucleus → tone", and what SBV2 reads is the binary tone of each mora itself, so keeping phrases
 * would create a dual representation of "which is authoritative, the nucleus or the tones".
 * {@link toUtterance} expands the nucleus fully before handing it over.
 */

import { InputError } from "../errors.ts";

/**
 * One element of the word alignment (one word, or one actually present symbol character).
 *
 * `phones` is **the phoneme sequence the word produces** (without PAD at either end). A symbol
 * element is one normal form (e.g. `["!"]`), and `surface` is the raw single character in the
 * normalised text (e.g. `"！"`).
 */
export type Word = {
  /** The word's surface (the source of DeBERTa's token count and of the BERT input text). */
  readonly surface: string;
  /** The phoneme sequence the word produces (without PAD at either end). */
  readonly phones: readonly string[];
};

/**
 * One mora of the utterance. **This is the editing surface** (fixing `tone` moves the accent).
 *
 * MUST: phonemes are derived from `consonant` / `vowel` (`[consonant, vowel]` when there is a
 * `consonant`, otherwise `[vowel]`). Do not keep the derived phoneme sequence in a separate field —
 * a fix-and-return round trip would carry the stale derived field along, leaving a choice between
 * "ignore it (silently discard the edit)" and "reject it (a legitimate round trip fails)".
 */
export type Mora = {
  /** One katakana mora (a contracted sound is one mora). For display and comparison. */
  readonly kana: string;
  /** Consonant phoneme (e.g. `"ky"`). Absent for vowel-only, moraic nasal and geminate moras. */
  readonly consonant?: string;
  /**
   * Vowel phoneme (`"a"`…`"o"` / moraic nasal `"N"` / geminate `"q"`). Long vowels are already
   * resolved to the preceding vowel.
   *
   * NOTE: **the phoneme itself** goes here (the geminate is `"q"`). When the upstream analysis
   * result writes the geminate as `"cl"`, {@link toUtterance} does the folding.
   */
  readonly vowel: string;
  /** Binary tone (0 = low, 1 = high). Consonant and vowel get the same tone. */
  readonly tone: 0 | 1;
  /**
   * The normal-form sequence of symbols actually present in the text **right after** this mora (in
   * order of appearance). Nothing is synthesised from pauses (symbols that are not present do not
   * become phonemes — the same policy as the reference implementation's g2p).
   */
  readonly punctuations?: readonly string[];
};

/**
 * The utterance passed to one synthesis (the first argument of {@link Pipeline.generate}).
 *
 * `moras` is the editing surface, and `words` is **read-only** (the source of the BERT input text
 * and word2ph). Rewriting `words` makes the phoneme sequence and the word split disagree, and the
 * synthesis fails at the entry check — to change a reading, redo it from the analysis side (the
 * caller's dictionary / correction dictionary).
 */
export type Utterance = {
  /**
   * Normal-form sequence of symbols present before the first mora (with symbol-only input, all of
   * them go here).
   */
  readonly leadingPunctuations: readonly string[];
  /** The utterance's mora sequence (phrase boundaries are not kept — the module doc's MUST). */
  readonly moras: readonly Mora[];
  /** Word alignment (read-only — the source of the BERT input text and word2ph). */
  readonly words: readonly Word[];
};

/**
 * The analysis result of the phrase structure (the input of {@link toUtterance}).
 *
 * It describes the shape that the output of an analyser returning readings as "accent phrases and
 * their nuclei" satisfies **structurally** (extra fields pass). The karume side reads only this
 * subset.
 */
export type Phrases = {
  readonly result: {
    /** Normal-form sequence of symbols present before the first phrase. */
    readonly leadingPunctuations: readonly string[];
    readonly accentPhrases: readonly {
      readonly moras: readonly {
        readonly kana: string;
        readonly consonant?: string;
        readonly vowel: string;
      }[];
      /** Accent nucleus (1-origin, 0 = flat). {@link toUtterance} clamps the upper end. */
      readonly accentNucleus: number;
      /**
       * Normal-form sequence of symbols present right after the phrase (in order of appearance).
       */
      readonly punctuations: readonly string[];
    }[];
  };
  /**
   * Word alignment taken from the same analysis (the phoneme sequence and the word split must be
   * consistent).
   */
  readonly words: readonly Word[];
};

/**
 * Expands an accent nucleus into the binary tones of the moras in the phrase.
 *
 * Rules (identical to upstream SBV2's g2p):
 *
 * - Flat (k = 0): only the first mora is 0, the rest 1.
 * - Head-high (k = 1): only the first mora is 1, the rest 0.
 * - Middle-high / tail-high (k > 1): the first mora 0, 2..k are 1, k+1 onwards 0.
 *
 * MUST: the **upper end** of the nucleus is clamped to the mora count (an out-of-range nucleus
 * routinely comes from dictionary differences and correction dictionaries). The clamped value
 * produces the same tone sequence, so the sound does not change. The **lower end** is not clamped
 * — a negative nucleus is "input that misread the meaning of the nucleus", and silently falling
 * back to flat would synthesise a different accent from the one specified.
 */
const phraseTones = (accentNucleus: number, moraCount: number, where: string): (0 | 1)[] => {
  if (!Number.isInteger(accentNucleus) || accentNucleus < 0) {
    throw new InputError(
      `${where}.accentNucleus = ${accentNucleus} is not a non-negative integer` +
        " (0 = flat, a 1-origin nucleus position, the mora count = tail-high)",
    );
  }
  const k = Math.min(accentNucleus, moraCount);
  if (k === 0) return Array.from({ length: moraCount }, (_, index) => (index === 0 ? 0 : 1));
  if (k === 1) return Array.from({ length: moraCount }, (_, index) => (index === 0 ? 1 : 0));
  return Array.from({ length: moraCount }, (_, index) => (index >= 1 && index + 1 <= k ? 1 : 0));
};

/**
 * Folding for when upstream writes the geminate as `"cl"` (`Mora.vowel` is the phoneme itself).
 */
const toVowelPhone = (vowel: string): string => (vowel === "cl" ? "q" : vowel);

/**
 * Lowers the phrase-structure analysis result to an utterance (pure function, no GPU).
 *
 * The nucleus is fully expanded here by {@link phraseTones} into a binary tone per mora. Symbols
 * right after a phrase are moved to **the last mora of that phrase** (flattening does not change
 * the order of the phonemes). Symbols before the first phrase move as is to
 * {@link Utterance.leadingPunctuations}.
 *
 * NOTE: symbols of a phrase with no moras are attached to the preceding mora (or, if there is none,
 * to the leading symbol sequence). Their position in the phoneme sequence is the same; only where
 * they are attached changes.
 */
export const toUtterance = (phrases: Phrases): Utterance => {
  const leadingPunctuations = [...phrases.result.leadingPunctuations];
  // Symbols are added afterwards, so this is mutable only while building (it is put on the
  // read-only surface before returning).
  const moras: {
    kana: string;
    consonant?: string;
    vowel: string;
    tone: 0 | 1;
    punctuations?: string[];
  }[] = [];

  for (const [index, phrase] of phrases.result.accentPhrases.entries()) {
    const tones = phraseTones(phrase.accentNucleus, phrase.moras.length, `accentPhrases[${index}]`);
    for (const [at, mora] of phrase.moras.entries()) {
      moras.push({
        kana: mora.kana,
        ...(mora.consonant === undefined ? {} : { consonant: mora.consonant }),
        vowel: toVowelPhone(mora.vowel),
        tone: tones[at],
      });
    }
    if (phrase.punctuations.length === 0) continue;
    const last = moras[moras.length - 1];
    if (last === undefined) {
      leadingPunctuations.push(...phrase.punctuations);
      continue;
    }
    last.punctuations = [...(last.punctuations ?? []), ...phrase.punctuations];
  }

  return {
    leadingPunctuations,
    moras,
    words: phrases.words.map((word) => ({ surface: word.surface, phones: [...word.phones] })),
  };
};
