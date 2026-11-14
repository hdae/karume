/**
 * Raw text → a **packed** token sequence (the content of the graph input `[1,T]`).
 *
 * Upstream calls backbone with a fixed length right-padded to `max_text_len` / `max_caption_len`,
 * but at B=1 this is mathematically equivalent to "calling with the fully packed length" (dropping
 * the padding with a bidirectional mask ⇔ never passing the padding). The exporter's
 * `_static_scheme_evidence` measures the equivalence on every emit.
 *
 * MUST: normalization happens once only (the doc of `text/tokenizer.ts` — the canonical
 * `tokenizer.json` has a null normalizer, so it is never applied again on the tokenizer side).
 *
 * MUST: **the preprocessing differs between text and caption**. Upstream's
 * `inference_runtime._synthesize` applies `normalize_text(...).strip()` to text, but only
 * `str(...).strip()` to caption (`normalize_text` is for text only). Normalizing the caption too
 * would silently change the conditioning by the outer-bracket stripping, NFKC and symbol deletion.
 *
 * MUST: the rare-token split (ADR 0149) is **text only**. Upstream enables it on the text tokenizer
 * and builds the caption tokenizer as a separate instance that is never switched, so the caption
 * path always encodes without it, whatever the request says.
 */

import { ModelInputError } from "../../errors.ts";
import { normalizeText } from "../text/normalize.ts";
import type { IrodoriEncodeOptions, IrodoriTokenizer } from "../text/tokenizer.ts";

/**
 * Preprocessed body → a packed sequence with BOS prepended, cut at `maxLength-1`.
 *
 * NOTE: it uses `encodePadded` and drops the padding so that the spelling of the BOS prepend and
 * the truncation does not exist in 2 places (the mask is always a contiguous prefix from the start,
 * so a scan gives the real length).
 */
const packBody = (
  tokenizer: IrodoriTokenizer,
  body: string,
  maxLength: number,
  options: IrodoriEncodeOptions,
): Int32Array<ArrayBuffer> => {
  const padded = tokenizer.encodePadded(body, maxLength, options);
  let used = 0;
  while (used < padded.mask.length && padded.mask[used] === 1) used += 1;
  return padded.ids.slice(0, used);
};

/**
 * `normalize_text` → `strip` → BOS prepend + the body cut at `maxLength-1`, for the **text side**.
 *
 * MUST: empty after normalization fails loudly. Upstream (`_packed_ids` in
 * `irodori/pipeline_ref.py`) fails at the same place — letting a BOS-only sequence through
 * generates **an utterance with no content that is not even silence**.
 *
 * `options.splitRareTokens` is the flag the pipeline resolved from the request
 * (`resolveSplitRareTokens` in `pipeline.ts`).
 */
export const packIds = (
  tokenizer: IrodoriTokenizer,
  text: string,
  maxLength: number,
  where: string,
  options: IrodoriEncodeOptions,
): Int32Array<ArrayBuffer> => {
  const normalized = normalizeText(text).trim();
  if (normalized.length === 0) {
    // The body is exactly the `GenerateRequest.text` the caller passed, so the 400 side
    // (ADR 0107). The empty check of `packCaptionIds` just below is an internal premise that is
    // reached only after the pipeline has routed an empty caption to another path, so it stays a
    // plain `Error`.
    throw new ModelInputError(
      `irodori: ${where} is empty after normalisation (there is no text to synthesise)`,
    );
  }
  return packBody(tokenizer, normalized, maxLength, options);
};

/**
 * `strip` → BOS prepend + the body cut at `maxLength-1`, for the **caption side** (no normalization
 * — the MUST of the module doc).
 *
 * An empty caption is filtered out before the call (the pipeline decides whether there is a caption
 * from the raw string, and an empty one goes to the no-caption path = upstream's
 * `caption_mask.zero_()`), so an empty one arriving here fails loudly — a BOS-only sequence as the
 * condition would be yet another condition, different even from no caption.
 */
export const packCaptionIds = (
  tokenizer: IrodoriTokenizer,
  caption: string,
  maxLength: number,
): Int32Array<ArrayBuffer> => {
  const stripped = caption.trim();
  if (stripped.length === 0) {
    throw new Error(
      "irodori: caption is empty after strip (input that belongs on the no-caption path)",
    );
  }
  // Never split (the MUST of the module doc).
  return packBody(tokenizer, stripped, maxLength, { splitRareTokens: false });
};
