/**
 * Port of the DeBERTa character tokenizer (transformers `BertJapaneseTokenizer`, word=basic /
 * subword=character). The target is `ku-nlp/deberta-v2-large-japanese-char-wwm`, the same as
 * `outputs/series/deberta/`.
 *
 * The reference implementation's path is BasicTokenizer → CharacterTokenizer in
 * `tokenization_bert_japanese.py`, and the effective steps are:
 *
 * 1. `BasicTokenizer._clean_text`: remove cp==0 / cp==0xFFFD / Unicode category C*, and turn
 *    whitespace (" " \t \n \r and Zs) into an ASCII space. **This decision table is baked into the
 *    asset** — reimplementing Unicode categories on the TS side would turn ICU version differences
 *    into silent mismatches (the canonical decision is Python's).
 * 2. NFC normalisation.
 * 3. Whitespace split (whitespace is discarded).
 * 4. NFKC-normalise each token → split into single code points.
 * 5. vocab lookup (`[UNK]` if absent). `encode` is `[CLS] + ids + [SEP]`.
 *
 * NOTE: the reference implementation inserts `_run_split_on_punc` (cutting punctuation out into
 * separate tokens) between 3 and 4, but the CharacterTokenizer right after splits every token into
 * single characters, so the output string does not change (NFKC composition happens only between a
 * combining character and its base character and does not cross punctuation boundaries). So it is
 * not ported.
 *
 * "One character" is per Unicode code point. The token count ties directly to the length of
 * word2ph, so if the split rules here drift from the reference implementation, the BERT features
 * are misplaced onto phonemes (the audio comes out but is broken). That mismatch is checked on
 * real data by `python -m sbv2.demo reference` (`tools/export-recipes/sbv2/demo.py`): "does the
 * same bert_text give the same input_ids".
 */

import { assertUniqueLines } from "../../text/asset-gates.ts";
import { inCodeRanges } from "../../text/code-ranges.ts";

/**
 * Code-point ranges that `clean_text` removes / turns into spaces (an ascending list of closed
 * intervals, both ends inclusive).
 */
export type CleanRanges = {
  /** Code-point ranges to remove (not output). */
  readonly removed: readonly (readonly [number, number])[];
  /** Code-point ranges to replace with an ASCII space. */
  readonly spaced: readonly (readonly [number, number])[];
};

/** Vocabulary IDs of `[CLS]` / `[SEP]` / `[UNK]`. */
type SpecialTokens = {
  readonly clsId: number;
  readonly sepId: number;
  readonly unkId: number;
};

export class DebertaTokenizer {
  readonly #vocab: ReadonlyMap<string, number>;
  readonly #clean: CleanRanges;
  readonly #special: SpecialTokens;

  constructor(vocab: ReadonlyMap<string, number>, clean: CleanRanges, special: SpecialTokens) {
    this.#vocab = vocab;
    this.#clean = clean;
    this.#special = special;
  }

  /**
   * Builds from the equivalent of `vocab.txt` (1 line = 1 token, 0-origin line number = ID).
   *
   * MUST: check that the special IDs point at rows of the vocabulary. An out-of-range ID rides the
   * id sequence as is and does not surface until the graph's embedding gather reads an
   * out-of-range index (NaN pollution on the GPU).
   * MUST: also check that lines are unique. A duplicate line makes the earlier ID unreachable
   * through `Map`'s last-write-wins, and the lookup quietly moves to **a different, valid row**
   * (no exception, only the BERT features change).
   */
  static fromVocabText(
    vocabText: string,
    clean: CleanRanges,
    special: SpecialTokens,
  ): DebertaTokenizer {
    // With mixed CRLF, split("\n") alone leaves "\r" at the end of every token and every lookup
    // falls to [UNK] (no error, only the BERT features quietly break).
    const lines = vocabText.split(/\r?\n/);
    // An empty line from the trailing newline is not a token. There are no empty tokens within
    // lines.
    if (lines[lines.length - 1] === "") lines.pop();
    assertUniqueLines(lines, "vocabText");
    const vocab = new Map<string, number>();
    for (const [id, line] of lines.entries()) vocab.set(line, id);
    const size = lines.length;
    for (const [name, id] of Object.entries(special)) {
      if (id >= size) throw new Error(`${name} ${id} が語彙の行数 ${size} 以上`);
    }
    return new DebertaTokenizer(vocab, clean, special);
  }

  /**
   * Applies clean_text + NFC + whitespace split + NFKC and returns an array of tokens (one code
   * point each). CLS / SEP are not included — the per-word token count for word2ph is counted with
   * this.
   */
  tokenize(text: string): string[] {
    let cleaned = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (cp === undefined) continue;
      if (inCodeRanges(this.#clean.removed, cp)) continue;
      cleaned += inCodeRanges(this.#clean.spaced, cp) ? " " : ch;
    }
    // NFC belongs to the reference implementation's BasicTokenizer side and NFKC to the
    // CharacterTokenizer side. A whitespace split sits between them, so they cannot be merged into
    // one (characters for which NFKC produces a space change the split result).
    const tokens: string[] = [];
    for (const word of cleaned.normalize("NFC").split(/\s+/)) {
      if (word === "") continue;
      for (const ch of word.normalize("NFKC")) tokens.push(ch);
    }
    return tokens;
  }

  /**
   * text to input_ids (`[CLS]` + each character ID + `[SEP]`). Matches the reference
   * `tokenizer(text)`.
   */
  encode(text: string): number[] {
    const ids = [this.#special.clsId];
    for (const token of this.tokenize(text)) {
      ids.push(this.#vocab.get(token) ?? this.#special.unkId);
    }
    ids.push(this.#special.sepId);
    return ids;
  }
}
