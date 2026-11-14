/**
 * The one narrow surface the Irodori pipeline reads its text tokenizer through, whatever the
 * tokenizer kind of the distribution: the Unigram tokenizer of the Small models
 * (`./tokenizer.ts`) or the Gemma BPE tokenizer of v4-Large (`./gemma-tokenizer.ts`). Which one a
 * distribution carries is decided by its asset (`./asset.ts`).
 *
 * Both kinds share Irodori's own back end, the counterpart of upstream's
 * `PretrainedTextTokenizer.batch_encode`: the plain encoding of the body (no special tokens added
 * by the tokenizer), BOS prepended by Irodori itself, truncation and right padding on the host
 * ({@link encodePaddedIds}). The back end is written once here, so the two kinds cannot drift
 * apart on it.
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

/**
 * Which path {@link IrodoriTextTokenizer.encode} takes. Required on every call: the caller decides
 * (the pipeline resolves the request's `splitRareTokens` once and passes the result to the text
 * path, and `false` to the caption path).
 */
export type IrodoriEncodeOptions = {
  /**
   * `true`: encode with the rare-token split (ADR 0149 — needs an asset with the table; a tokenizer
   * without it fails loudly with {@link RARE_TOKEN_SPLIT_MISSING}). `false`: the plain path.
   */
  readonly splitRareTokens: boolean;
};

/** The failure of a split request on a tokenizer that has no split table (both kinds say the same). */
export const RARE_TOKEN_SPLIT_MISSING =
  "irodori tokenizer: the rare-token split was requested, but the asset carries no rareTokenSplit table";

/** What the pipeline, `host/pack.ts` and the request checks need from a text tokenizer. */
export type IrodoriTextTokenizer = {
  /** The id Irodori prepends to every text and caption (upstream's `bos_token_id`). */
  readonly bosId: number;
  /** The id the right padding is filled with (upstream's `pad_token_id`). */
  readonly padId: number;
  /** Whether the asset carries the rare-token split table (the split path is available). */
  readonly hasRareTokenSplit: boolean;
  /**
   * Normalized body → plain id sequence (no special tokens added — the tokenizer's post_processor
   * never runs; upstream encodes with `add_special_tokens=False`).
   *
   * MUST: the input is the string after `normalize.ts` (text) or after `strip` (caption); the
   * tokenizer does not normalize it again.
   */
  encode(text: string, options: IrodoriEncodeOptions): number[];
};

/** Final form of upstream's `batch_encode` (graph input dtypes — i32 and bool = u32, ADR 0009). */
export type IrodoriPaddedIds = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

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
export const encodePaddedIds = (
  tokenizer: IrodoriTextTokenizer,
  text: string,
  maxLength: number,
  options: IrodoriEncodeOptions,
): IrodoriPaddedIds => {
  if (!Number.isInteger(maxLength) || maxLength <= 0) {
    throw new Error(`maxLength must be an integer >= 1 (received: ${maxLength})`);
  }
  const body = tokenizer.encode(text, options).slice(0, maxLength - 1);
  const ids = new Int32Array(maxLength).fill(tokenizer.padId);
  const mask = new Uint32Array(maxLength);
  ids[0] = tokenizer.bosId;
  mask[0] = 1;
  ids.set(body, 1);
  mask.fill(1, 1, body.length + 1);
  return { ids, mask };
};
