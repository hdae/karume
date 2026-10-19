/**
 * Generation of the base word2ph (before add_blank). The core of the BERT text layer.
 *
 * word2ph = "how many phonemes are assigned to each DeBERTa token (= character)". An exact
 * alignment is impossible (jukujikun readings), so the reference implementation takes the
 * approximation "distribute each word's phoneme count evenly over its characters"
 * (`__distribute_phone` in `g2p.py`).
 *
 * Scope: what is built here is the base word2ph **before** add_blank.
 *
 * - `sum(base word2ph) === given_phone length` (the output length of `toSbv2PhoneTone`, including
 *   PAD at both ends and punctuation)
 * - `len(base word2ph) === Σ tokenize(word surface) + 2`
 *
 * add_blank's `*2` / `[0]+=1` is the responsibility of `addBlankWord2ph` in `symbols.ts`.
 */

import type { DebertaTokenizer } from "./tokenizer.ts";
import type { Sbv2Word } from "./utterance.ts";
import { Sbv2InputError } from "../errors.ts";

/**
 * Distributes n phonemes evenly over `wordLength` characters (the same distribution as
 * `g2p.py __distribute_phone`).
 *
 * Upstream iterates "from left to right, add 1 at a time, always to the smallest character (the
 * leftmost on ties)", but the result has a closed form: with `q = ⌊n / W⌋` and `r = n mod W`, the
 * first `r` characters get `q + 1` and the rest `q`. Each round of the iteration raises every
 * character by 1, and the final remainder `r` fills from the left.
 *
 * MUST: do not go back to copying the iteration — it is O(phonemes × characters) and runs before
 * the length limit check (on the pipeline side), so one huge word occupies the CPU for seconds.
 * The closed form is O(W), and the distribution is identical to upstream.
 */
export const distributePhone = (phoneCount: number, wordLength: number): number[] => {
  if (!Number.isInteger(wordLength) || wordLength <= 0) {
    throw new Error(`Target character count ${wordLength} is not an integer >= 1`);
  }
  if (!Number.isInteger(phoneCount) || phoneCount < 0) {
    throw new Error(`Phoneme count to distribute ${phoneCount} is not a non-negative integer`);
  }
  const quotient = Math.floor(phoneCount / wordLength);
  const remainder = phoneCount % wordLength;
  return Array.from({ length: wordLength }, (_, index) => quotient + (index < remainder ? 1 : 0));
};

/**
 * Builds the base word2ph from the word alignment and the DeBERTa tokenizer.
 *
 * Invariants (throw when broken):
 *
 * - `word2ph.length === Σ tokenize(surface) + 2` (`len(word2ph) == len(text)+2` in
 *   `bert_feature.py`. text is the concatenation of the word surfaces, and the character tokenizer
 *   is additive, so `Σ tokenize(word) === tokenize(joined)` holds).
 * - `sum(word2ph) === givenPhoneLength` (`len(phones) == sum(word2ph)` in `g2p.py`).
 *
 * NOTE: the reference implementation skips tokenisation only for punctuation words and treats them
 * as one character, whereas this tokenises every word. On the reference side `normalize_text` has
 * already collapsed "…" to "...", so it is equivalent in practice; when the upstream analysis
 * keeps "…" as one symbol, not aligning with the tokenising side makes the length disagree with
 * DeBERTa's real token count (3 for "..." under NFKC).
 *
 * @param words word alignment (`Sbv2Utterance.words` — without PAD at either end).
 * @param tokenizer DeBERTa character tokenizer (used to count the tokens of each word surface).
 * @param givenPhoneLength `toSbv2PhoneTone(...).phones.length` (including PAD at both ends).
 */
export const buildBaseWord2ph = (
  words: readonly Sbv2Word[],
  tokenizer: DebertaTokenizer,
  givenPhoneLength: number,
): number[] => {
  const word2ph: number[] = [1]; // Sentinel for the leading PAD.
  let tokenTotal = 0;
  for (const word of words) {
    const wordLength = tokenizer.tokenize(word.surface).length;
    if (wordLength <= 0) {
      // Reachable only through the surface the caller passed (a whitespace-only word and the like),
      // so it is input-caused = 400.
      throw new Sbv2InputError(
        `Word surface normalised to 0 tokens: ${JSON.stringify(word.surface)}`,
      );
    }
    tokenTotal += wordLength;
    for (const count of distributePhone(word.phones.length, wordLength)) word2ph.push(count);
  }
  word2ph.push(1); // Sentinel for the trailing PAD.

  const expectedLength = tokenTotal + 2;
  if (word2ph.length !== expectedLength) {
    throw new Error(
      `word2ph length ${word2ph.length} differs from Σtokenize(surface)+2 (${expectedLength})`,
    );
  }
  const total = word2ph.reduce((sum, count) => sum + count, 0);
  if (total !== givenPhoneLength) {
    throw new Error(`sum(word2ph) ${total} differs from given_phone length ${givenPhoneLength}`);
  }
  return word2ph;
};
