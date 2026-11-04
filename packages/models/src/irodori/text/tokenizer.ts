/**
 * The Irodori-TTS v4 tokenizer (Unigram + byte_fallback). Both text and caption go through this
 * same one.
 *
 * The canonical path (`tokenizer/tokenizer.json`) is
 *   AddedVocabulary (splits out the 18 special tokens leftmost-longest)
 *   → normalizer is **null** (normalization is done by the earlier pipeline step `normalize.ts`)
 *   → Metaspace{replacement=▁, prepend_scheme=never, split=false}
 *      = only replaces U+0020 with ▁ (**does not prepend ▁ / does not split into pieces at ▁**)
 *   → Unigram (Viterbi, unk_id=0, **with byte_fallback**, fuse_unk)
 *   → post_processor adds no special tokens (BOS is prepended by
 *      {@link IrodoriTokenizer.encodePadded} below = the counterpart of upstream's
 *      `PretrainedTextTokenizer.batch_encode`)
 *
 * The Unigram core (Viterbi, tie handling, `fuse_unk`, byte expansion) is family-independent, so
 * `src/text/unigram.ts` holds it. This module holds only Irodori's own front end (AddedVocabulary,
 * Metaspace) and back end (BOS prepend, truncation, right padding, mask).
 */

import { type AddedTokenMatcher, createAddedTokenMatcher } from "../../text/added-tokens.ts";
import {
  asFiniteNumber,
  asId,
  asRecord,
  assertUniqueLines,
  asString,
  asVocabId,
  parseAddedTokensInVocab,
} from "../../text/asset-gates.ts";
import { assertEncodableText, toCodePoints } from "../../text/code-points.ts";
import { type UnigramModel, unigramTokenize, type UnigramVocabEntry } from "../../text/unigram.ts";

/** Metaspace replacement character (U+2581). */
const METASPACE = "▁";

export type IrodoriTokenizerAssets = {
  readonly vocab: ReadonlyMap<string, UnigramVocabEntry>;
  /**
   * Minimum score of the **whole** vocabulary. Pass the whole-vocabulary value even when passing a
   * subset (the weight of unknown nodes).
   */
  readonly minScore: number;
  /** Code-point count of the longest token of the **whole** vocabulary (prefix-search width). */
  readonly maxTokenLength: number;
  readonly unkId: number;
  /** Id of `<0x00>` (an unknown piece expands to UTF-8 bytes → `byteBaseId + byte value`). */
  readonly byteBaseId: number;
  readonly bosId: number;
  readonly padId: number;
  /** Added vocabulary (special tokens). Split out **before** normalization. */
  readonly addedTokens: ReadonlyMap<string, number>;
};

/** Final form of upstream's `batch_encode` (graph input dtypes — i32 and bool = u32, ADR 0009). */
type IrodoriPaddedIds = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

export class IrodoriTokenizer {
  readonly #assets: IrodoriTokenizerAssets;
  readonly #added: AddedTokenMatcher;

  constructor(assets: IrodoriTokenizerAssets) {
    this.#assets = assets;
    this.#added = createAddedTokenMatcher([...assets.addedTokens.keys()]);
  }

  /**
   * Normalized text → plain id sequence (no special tokens added).
   *
   * MUST: the input is the string after `normalize.ts`. The canonical `tokenizer.json` has a null
   * normalizer, so normalizing here would apply it twice (the `…` collapsing and so on would run
   * twice).
   */
  encode(text: string): number[] {
    // The acceptance check of the body text itself runs before the lattice is solved (input-caused
    // per ADR 0107 decision 2). The asset parsing below (`parseIrodoriTokenizerAsset`) goes
    // through the same condition, but a violation there is an asset mismatch, so it stays a plain
    // `Error`.
    assertEncodableText(text, "irodori text");
    const ids: number[] = [];
    for (const chunk of this.#added.split(text)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      // Metaspace has `split=false`, so the text is not split into pieces — the whole text is
      // solved as one lattice.
      const piece = chunk.text.replaceAll(" ", METASPACE);
      // IrodoriTokenizerAssets satisfies the UnigramModel surface as is. The piece is the whole
      // text, so the id sequence can be long too — spreading arguments (`push(...ids)`) would hit
      // the stack limit.
      for (const id of unigramTokenize(this.#assets satisfies UnigramModel, toCodePoints(piece))) {
        ids.push(id);
      }
    }
    return ids;
  }

  /**
   * The same final form as upstream's `PretrainedTextTokenizer.batch_encode([text], max_length)`:
   * BOS prepended + right padding + valid-length mask.
   *
   * MUST: the body budget is `maxLength - 1` (leaving room for the one BOS). Naively cutting at
   * `maxLength` overflows the whole by one — writing with {@link Int32Array.prototype.set} is there
   * so that the overflow is not silently discarded (index assignment ignores out-of-range writes,
   * so the overflow would be swallowed silently).
   */
  encodePadded(text: string, maxLength: number): IrodoriPaddedIds {
    if (!Number.isInteger(maxLength) || maxLength <= 0) {
      throw new Error(`maxLength must be an integer >= 1 (received: ${maxLength})`);
    }
    const body = this.encode(text).slice(0, maxLength - 1);
    const ids = new Int32Array(maxLength).fill(this.#assets.padId);
    const mask = new Uint32Array(maxLength);
    ids[0] = this.#assets.bosId;
    mask[0] = 1;
    ids.set(body, 1);
    mask.fill(1, 1, body.length + 1);
    return { ids, mask };
  }
}

/**
 * First id of the byte fallback ({@link asId}, plus the whole run of 256 consecutive ids being
 * inside the vocabulary).
 *
 * MUST: {@link asVocabId} for a single id (`< vocabSize`) is not enough — `byteBaseId` is not a
 * single id but the start of the 256 consecutive ids an unknown piece expands to, and the consumer
 * (`base + byte` in `unigram.ts`, byte = 0..255) emits up to `byteBaseId + 255`. Missing this, an
 * out-of-vocabulary id appears the moment an unknown piece shows up, a silent wrong value that does
 * not surface until the OOB of the embedding gather (non-finite — ADR 0061).
 *
 * MUST: the spellings of the 256 rows are matched too (`<0x00>`..`<0xFF>` in order = the
 * distribution contract). The consumer computes ids by arithmetic on the run, not by spelling, so
 * an asset with swapped rows would silently emit ids different from HF (which looks ids up by
 * name). Paired with the run check on the exporter side
 * (`tools/export-recipes/irodori/tokenizer_ref.py`), the reader rejects the same contract
 * violations.
 */
const asByteBaseId = (value: unknown, label: string, tokens: readonly string[]): number => {
  const id = asId(value, label);
  if (id + 255 >= tokens.length) {
    throw new Error(
      `${label}: the 256 bytes (${id}..${id + 255}) do not fit in the vocabulary row count ${tokens.length}`,
    );
  }
  for (let byte = 0; byte < 256; byte += 1) {
    const want = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    if (tokens[id + byte] !== want) {
      throw new Error(
        `${label}: row ${id + byte} is not ${want} (${JSON.stringify(tokens[id + byte])})`,
      );
    }
  }
  return id;
};

/**
 * Asset JSON (the exporter's `tokenizer.json`) → the asset table. It is an external boundary, so the
 * structure is checked before use.
 *
 * MUST: the vocabulary row count and the number of `scores` are matched. Letting a mismatch through
 * shifts every id, but Viterbi does not fail (a piece missing from the vocabulary escapes to byte
 * expansion), so it becomes a **silent wrong value**.
 * MUST: row uniqueness is checked too ({@link assertUniqueLines} — matching the row count cannot
 * catch duplicates).
 * MUST: a single special id is an integer below the vocabulary row count ({@link asVocabId} — the
 * reason is in its doc).
 * MUST: `byteBaseId` is not a single id, so {@link asVocabId} is not enough — the whole run of 256
 * consecutive ids must be inside the vocabulary ({@link asByteBaseId} — the reason is in its doc).
 * MUST: the added tokens are matched against the vocabulary table too
 * ({@link parseAddedTokensInVocab} — the reason is in its doc).
 */
export const parseIrodoriTokenizerAsset = (
  raw: unknown,
  label: string = "tokenizer",
): IrodoriTokenizerAssets => {
  const obj = asRecord(raw, label);
  // 0-origin row number = id. That no token in the vocabulary contains a newline, tab or raw space
  // was checked at emit time.
  const tokens = asString(obj["vocabText"], `${label}.vocabText`).split("\n");
  const rawScores = obj["scores"];
  if (!Array.isArray(rawScores) || rawScores.length !== tokens.length) {
    throw new Error(
      `${label}: the length of scores does not match the vocabulary row count ${tokens.length}`,
    );
  }
  assertUniqueLines(tokens, `${label}.vocabText`);
  const vocab = new Map<string, UnigramVocabEntry>();
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asFiniteNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, toCodePoints(token).length);
  }
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asVocabId(obj["unkId"], `${label}.unkId`, tokens.length),
    byteBaseId: asByteBaseId(obj["byteBaseId"], `${label}.byteBaseId`, tokens),
    bosId: asVocabId(obj["bosId"], `${label}.bosId`, tokens.length),
    padId: asVocabId(obj["padId"], `${label}.padId`, tokens.length),
    addedTokens: parseAddedTokensInVocab(obj["addedTokens"], `${label}.addedTokens`, vocab),
  };
};
