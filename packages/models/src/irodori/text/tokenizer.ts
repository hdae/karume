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
 * The **rare-token split** (ADR 0149 — upstream's `RareTokenSplitter`) is a second path through
 * the same front end: the lattice is solved by a second Unigram model whose disabled tokens carry
 * the replaced score, and every id found in the forced table is then replaced by its pieces. The
 * asset carries the result of upstream's rule as a table (`rareTokenSplit`), never the rule, and
 * which path a call takes is the caller's explicit flag ({@link IrodoriEncodeOptions}).
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

/**
 * The rare-token split table (ADR 0149): the result of upstream's `RareTokenSplitter` on this
 * vocabulary.
 */
export type IrodoriRareTokenSplit = {
  /** The score the disabled tokens carry in the split model (upstream: `-1e9`). */
  readonly disabledScore: number;
  /** Ids whose score is replaced by {@link IrodoriRareTokenSplit.disabledScore} (ascending). */
  readonly disabledIds: readonly number[];
  /**
   * Minimum score of the **whole replaced-score** vocabulary (the unknown-node weight of the split
   * model). Like {@link IrodoriTokenizerAssets.minScore}, pass the whole-vocabulary value even with
   * a subset.
   */
  readonly minScore: number;
  /** Token id → the ids it is replaced by after the lattice is solved. */
  readonly forcedSplits: ReadonlyMap<number, readonly number[]>;
};

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
  /**
   * The rare-token split table. Present only in assets of a tokenizer upstream splits
   * (`sbintuitions/modernbert-ja-310m`); an older asset revision has none.
   */
  readonly rareTokenSplit?: IrodoriRareTokenSplit;
};

/**
 * Which path {@link IrodoriTokenizer.encode} takes. Required on every call: the caller decides
 * (the pipeline resolves the request's `splitRareTokens` once and passes the result to the text
 * path, and `false` to the caption path).
 */
export type IrodoriEncodeOptions = {
  /**
   * `true`: encode with the rare-token split (needs an asset with the table — an asset without it
   * fails loudly). `false`: the plain Unigram path.
   */
  readonly splitRareTokens: boolean;
};

/** The split path: its own Unigram model and the forced expansion. */
type SplitPath = {
  readonly model: UnigramModel;
  readonly forcedSplits: ReadonlyMap<number, readonly number[]>;
};

/**
 * The Unigram model of the split path: the same vocabulary with the disabled tokens' scores
 * replaced, and **its own** minimum score.
 *
 * MUST: not approximated by removing the disabled entries. The unknown-node weight is "minimum score
 * − 10", so with the replaced minimum (-1e9) an unknown piece costs more than a disabled token: a
 * disabled token whose characters have no other piece is still emitted, where a vocabulary without
 * it would fall back to bytes. Upstream copies the tokenizer and replaces scores, so it is solved
 * the same way here.
 */
const buildSplitPath = (
  assets: IrodoriTokenizerAssets,
  split: IrodoriRareTokenSplit,
): SplitPath => {
  const disabled = new Set(split.disabledIds);
  const vocab = new Map<string, UnigramVocabEntry>();
  for (const [token, entry] of assets.vocab) {
    vocab.set(token, disabled.has(entry.id) ? { id: entry.id, score: split.disabledScore } : entry);
  }
  return {
    model: {
      vocab,
      minScore: split.minScore,
      maxTokenLength: assets.maxTokenLength,
      unkId: assets.unkId,
      byteBaseId: assets.byteBaseId,
    },
    forcedSplits: split.forcedSplits,
  };
};

/** Final form of upstream's `batch_encode` (graph input dtypes — i32 and bool = u32, ADR 0009). */
type IrodoriPaddedIds = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

export class IrodoriTokenizer {
  readonly #assets: IrodoriTokenizerAssets;
  readonly #added: AddedTokenMatcher;
  /**
   * Built at construction, not lazily: the table exists only for the tokenizer whose default is
   * the split (ADR 0149), so the first default request needs it anyway, and building it here keeps
   * the field immutable and moves the one pass over the vocabulary to pipeline construction
   * (before the GPU is acquired) instead of the first generation.
   */
  readonly #split: SplitPath | undefined;

  constructor(assets: IrodoriTokenizerAssets) {
    this.#assets = assets;
    this.#added = createAddedTokenMatcher([...assets.addedTokens.keys()]);
    this.#split =
      assets.rareTokenSplit === undefined
        ? undefined
        : buildSplitPath(assets, assets.rareTokenSplit);
  }

  /** Whether the asset carries the rare-token split table (the split path is available). */
  get hasRareTokenSplit(): boolean {
    return this.#split !== undefined;
  }

  /**
   * Normalized text → plain id sequence (no special tokens added).
   *
   * MUST: the input is the string after `normalize.ts`. The canonical `tokenizer.json` has a null
   * normalizer, so normalizing here would apply it twice (the `…` collapsing and so on would run
   * twice).
   *
   * With `splitRareTokens`, the lattice is solved by the split model and each id of the forced
   * table is replaced by its pieces — over the whole sequence, as upstream's `RareTokenSplitter.encode`
   * does.
   */
  encode(text: string, options: IrodoriEncodeOptions): number[] {
    // The acceptance check of the body text itself runs before the lattice is solved (input-caused
    // per ADR 0107 decision 2). The asset parsing below (`parseIrodoriTokenizerAsset`) goes
    // through the same condition, but a violation there is an asset mismatch, so it stays a plain
    // `Error`.
    assertEncodableText(text, "irodori text");
    let model: UnigramModel = this.#assets;
    let forced: ReadonlyMap<number, readonly number[]> | undefined;
    if (options.splitRareTokens) {
      if (this.#split === undefined) {
        throw new Error(
          "irodori tokenizer: the rare-token split was requested, but the asset carries no rareTokenSplit table",
        );
      }
      model = this.#split.model;
      forced = this.#split.forcedSplits;
    }
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
      for (const id of unigramTokenize(model, toCodePoints(piece))) ids.push(id);
    }
    if (forced === undefined) return ids;
    const expanded: number[] = [];
    for (const id of ids) {
      const pieces = forced.get(id);
      if (pieces === undefined) expanded.push(id);
      else for (const piece of pieces) expanded.push(piece);
    }
    return expanded;
  }

  /**
   * The same final form as upstream's `PretrainedTextTokenizer.batch_encode([text], max_length)`:
   * BOS prepended + right padding + valid-length mask.
   *
   * MUST: the body budget is `maxLength - 1` (leaving room for the one BOS). Naively cutting at
   * `maxLength` overflows the whole by one — writing with {@link Int32Array.prototype.set} is there
   * so that the overflow is not silently discarded (index assignment ignores out-of-range writes,
   * so the overflow would be swallowed silently).
   *
   * MUST: the body is cut **after** the split (upstream: `splitter.encode(text)[:body_max_length]`)
   * — the split lengthens the body, so cutting first would keep tokens upstream drops.
   */
  encodePadded(text: string, maxLength: number, options: IrodoriEncodeOptions): IrodoriPaddedIds {
    if (!Number.isInteger(maxLength) || maxLength <= 0) {
      throw new Error(`maxLength must be an integer >= 1 (received: ${maxLength})`);
    }
    const body = this.encode(text, options).slice(0, maxLength - 1);
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

/** An array, or an asset error naming the field. */
const asArray = (value: unknown, label: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label}: not an array`);
  return value;
};

/**
 * Ascending vocabulary ids ({@link asVocabId} each, strictly increasing — so also unique).
 *
 * MUST: strictly increasing, not only in range. The exporter writes them sorted, so a duplicate or
 * an out-of-order id means a table edited or assembled by something else — and a duplicate forced
 * key would be later-wins in a `Map`, silently choosing one of two expansions.
 */
const asAscendingVocabIds = (
  values: readonly unknown[],
  label: string,
  vocabSize: number,
): number[] => {
  const ids: number[] = [];
  for (const [index, value] of values.entries()) {
    const id = asVocabId(value, `${label}[${index}]`, vocabSize);
    if (ids.length > 0 && id <= ids[ids.length - 1]) {
      throw new Error(
        `${label}[${index}]: id ${id} is not above the previous id ${ids[ids.length - 1]} (ascending and unique)`,
      );
    }
    ids.push(id);
  }
  return ids;
};

/**
 * `rareTokenSplit` of the asset (ADR 0149) → the split table, with the minimum score of the
 * replaced-score vocabulary computed from the **whole** vocabulary here.
 *
 * MUST: the same strictness as the rest of the asset — every id inside the vocabulary, the disabled
 * ids and the forced keys ascending and unique, every forced expansion non-empty, a finite
 * `disabledScore`. A malformed table raises no exception later; it only splits differently from
 * upstream (a silent wrong value).
 */
const parseRareTokenSplit = (
  raw: unknown,
  label: string,
  scores: readonly number[],
): IrodoriRareTokenSplit => {
  const obj = asRecord(raw, label);
  const vocabSize = scores.length;
  const disabledScore = asFiniteNumber(obj["disabledScore"], `${label}.disabledScore`);
  const disabledIds = asAscendingVocabIds(
    asArray(obj["disabledIds"], `${label}.disabledIds`),
    `${label}.disabledIds`,
    vocabSize,
  );
  const pairs = asArray(obj["forcedSplits"], `${label}.forcedSplits`).map((row, index) => {
    const at = `${label}.forcedSplits[${index}]`;
    const pair = asArray(row, at);
    if (pair.length !== 2) throw new Error(`${at}: not a [tokenId, pieces] pair`);
    return pair;
  });
  const keys = asAscendingVocabIds(
    pairs.map((pair) => pair[0]),
    `${label}.forcedSplits[].tokenId`,
    vocabSize,
  );
  const forcedSplits = new Map<number, readonly number[]>();
  for (const [index, key] of keys.entries()) {
    const at = `${label}.forcedSplits[${index}][1]`;
    const pieces = asArray(pairs[index][1], at);
    if (pieces.length === 0) throw new Error(`${at}: token ${key} has no pieces`);
    forcedSplits.set(
      key,
      pieces.map((piece, position) => asVocabId(piece, `${at}[${position}]`, vocabSize)),
    );
  }
  const disabled = new Set(disabledIds);
  let minScore = Number.POSITIVE_INFINITY;
  for (const [id, score] of scores.entries()) {
    minScore = Math.min(minScore, disabled.has(id) ? disabledScore : score);
  }
  return { disabledScore, disabledIds, minScore, forcedSplits };
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
 * The optional `rareTokenSplit` goes through {@link parseRareTokenSplit} (an asset without it is
 * an older revision or another tokenizer, and has no split path).
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
  const scores: number[] = [];
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asFiniteNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    scores.push(score);
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, toCodePoints(token).length);
  }
  const rawSplit = obj["rareTokenSplit"];
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asVocabId(obj["unkId"], `${label}.unkId`, tokens.length),
    byteBaseId: asByteBaseId(obj["byteBaseId"], `${label}.byteBaseId`, tokens),
    bosId: asVocabId(obj["bosId"], `${label}.bosId`, tokens.length),
    padId: asVocabId(obj["padId"], `${label}.padId`, tokens.length),
    addedTokens: parseAddedTokensInVocab(obj["addedTokens"], `${label}.addedTokens`, vocab),
    ...(rawSplit === undefined
      ? {}
      : { rareTokenSplit: parseRareTokenSplit(rawSplit, `${label}.rareTokenSplit`, scores) }),
  };
};
