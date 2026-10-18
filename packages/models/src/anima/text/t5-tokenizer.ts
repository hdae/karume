/**
 * T5 tokenizer (Unigram). Builds the input of Anima's text_conditioner side.
 *
 * The canonical path (`t5_tokenizer/tokenizer.json`) is
 *   AddedVocabulary (cuts out the 103 special tokens leftmost-longest)
 *   → Precompiled normalisation (`spm-normalizer.ts`)
 *   → WhitespaceSplit → Metaspace (replacement=▁ / prepend_scheme=always / split)
 *   → Unigram (Viterbi, unk_id=2, no byte_fallback, fuse_unk)
 *   → TemplateProcessing (`</s>` at the end)
 *
 * The body of the path (added vocabulary, Metaspace, Unigram, `</s>`) lives in the cross-family
 * `src/text/t5-tokenizer.ts`. Only three Anima-specific points live here: the asset includes the
 * Precompiled normalisation table, out-of-vocabulary text fuses into one unk, and input over 512 is
 * truncated as upstream does.
 */

import { type T5Tables, T5UnigramTokenizer } from "../../text/t5-tokenizer.ts";
import { normalizeSpm, type SpmTables } from "./spm-normalizer.ts";

export type T5Assets = T5Tables & { readonly normalizer: SpmTables };

export class T5Tokenizer {
  readonly #inner: T5UnigramTokenizer;

  constructor(assets: T5Assets) {
    // NOTE: `fuse_unk` in `tokenizer.json` is `null`, but the Unigram of `tokenizers` fuses even
    // when unspecified (the Rust-side default). Without fusion a Japanese prompt yields one unk per
    // character and the comparison with the canonical tokenizer fails in the `japanese` case.
    // byte_fallback is false, so unknown text becomes one unk.
    this.#inner = new T5UnigramTokenizer(assets, {
      normalize: (text) => normalizeSpm(assets.normalizer, text),
      unknown: "unk",
      overflow: "truncate",
    });
  }

  /**
   * The same id sequence as
   * `tokenizer([text], padding="longest", max_length=512, truncation=True)`. The result always ends
   * with `</s>` and its length is always at least 1 (the truncation rule is the shared layer's
   * `encode`).
   */
  encode(text: string): number[] {
    return this.#inner.encode(text);
  }
}
