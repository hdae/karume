/**
 * The Gemma tokenizer (SPM-BPE + byte_fallback) — string ⇄ id sequence.
 *
 * The reference pipeline (the same configuration for both assets — ADR 0084 Context 2):
 *   AddedVocabulary (cuts out the added vocabulary leftmost-longest, **before** normalization)
 *   → Replace(" " → "▁") → Split(" ", MergedWithPrevious) → BPE (byte_fallback)
 *   → TemplateProcessing (gemma4 = pass-through / EmbeddingGemma = `<bos>` … `<eos>`)
 * Decoding is Sequence[Replace("▁" → " "), ByteFallback, Fuse].
 *
 * MUST: **the pre_tokenizer never splits after normalization**. `Split(" ")` runs on the string
 * after the normalizer has replaced every U+0020 with U+2581, so by construction it finds no
 * separator (measured as well: `pre_tokenize_str("hello▁world▁foo")` is one piece). **The piece is
 * the whole input**, so the BPE search gets long for English as well as Japanese — which is why
 * `bpeEncode` is written with a merge queue (ADR 0084 decision 3). The scan itself is not ported
 * because it would always be the identity.
 *
 * MUST: {@link GemmaTokenizer.encode} does not add `<bos>` (matching the measured gemma4
 * post_processor). Only the chat function adds it — without that split, chat would produce a double
 * BOS (ADR 0084 decision 5). When the asset declares that they are added, as EmbeddingGemma does,
 * {@link GemmaTokenizer.encodeWithSpecialTokens} adds them.
 *
 * MUST: **do not build tables at module scope** (the cross-cutting invariant "zero module side
 * effects"). Both the added-vocabulary order and the decoding hooks are built once per instance.
 */

import { type AddedTokenMatcher, createAddedTokenMatcher } from "../../text/added-tokens.ts";
import { bpeEncode } from "../../text/bpe.ts";
import { assertEncodableText } from "../../text/code-points.ts";
import {
  detokenize,
  type DetokenizerSource,
  StreamingDetokenizer,
} from "../../text/detokenizer.ts";
import { type GemmaTokenizerAssets, METASPACE } from "./asset.ts";

/** Decoding behaviour. */
export type GemmaDecodeOptions = {
  /**
   * Whether to drop special tokens (`<bos>` / `<eos>` / `<|turn>` …) from the output. Default
   * `true` (the same as the reference `Tokenizer.decode`'s default; the form that shows generated
   * output to people as is).
   *
   * NOTE: a dropped token **does not break a byte run** — the reference also skips while building
   * the spelling sequence, so bytes on both sides are folded into one character as if adjacent.
   */
  readonly skipSpecialTokens?: boolean;
};

export class GemmaTokenizer {
  readonly #assets: GemmaTokenizerAssets;
  readonly #added: AddedTokenMatcher;
  readonly #tokenOfAdded: Map<number, string>;

  constructor(assets: GemmaTokenizerAssets) {
    this.#assets = assets;
    this.#added = createAddedTokenMatcher([...assets.addedTokens.keys()]);
    this.#tokenOfAdded = new Map(
      [...assets.addedTokens].map(([token, id]) => [id, token] as const),
    );
  }

  /** The post_processor form the asset declares. */
  get postProcessor(): GemmaTokenizerAssets["spec"]["postProcessor"] {
    return this.#assets.spec.postProcessor;
  }

  /**
   * The id of `<eos>` the asset declares (the basis of the stop condition — ADR 0083 decision 8).
   */
  get eosId(): number {
    return this.#assets.eosId;
  }

  /**
   * The **largest token id** this asset can produce (cross-check of the id spaces — ADR 0085
   * decision 5).
   *
   * Added vocabulary may be numbered **outside** the vocabulary table (EmbeddingGemma's
   * `<image_soft_token>` is id 262144 / 262,144 vocabulary rows), so the vocabulary row count alone
   * is not the limit. An asset combination where this exceeds the main embedding's row count reads
   * not out of bounds but **a valid row of another token**.
   */
  get maxTokenId(): number {
    let max = this.#assets.model.pairStride - 1;
    for (const id of this.#assets.addedTokens.values()) if (id > max) max = id;
    return max;
  }

  /**
   * Added-vocabulary spelling → id (`undefined` when absent).
   *
   * The entry that reads both the chat spellings (`<|turn>` / `<turn|>`) and the stop tokens
   * (`<|tool_response>`) from **the same asset** (ADR 0084 decision 5's "same digest set" — picking
   * them up from separate places lets one go stale). Failing on a missing spelling by name is the
   * caller's responsibility, so this only returns `undefined` (this surface by itself cannot treat
   * "an unknown spelling" as an error).
   */
  addedTokenId(token: string): number | undefined {
    return this.#assets.addedTokens.get(token);
  }

  /**
   * The same id sequence as `tokenizer.encode(text, add_special_tokens=False)`.
   *
   * The post_processor is not applied — the owner of `<bos>` is the chat function (ADR 0084
   * decision 5).
   *
   * An unpaired surrogate is a `ModelInputError` (fixed when the caller fixes the text — ADR 0107
   * decision 2). Only the caller's text (including the rendered chat) comes in, so the thrown type
   * is raised here at the entry.
   */
  encode(text: string): number[] {
    assertEncodableText(text, "Gemma の本文");
    const ids: number[] = [];
    for (const chunk of this.#added.split(text)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      // The normalizer is a single `Replace(" " → "▁")` (it touches neither Unicode categories nor
      // NFC). Push one element at a time — the whole text is one piece, so spreading arguments
      // (`push(...)`) hits V8's argument limit as a RangeError at about 130,000 tokens, failing
      // before the caller's capacity check is reached.
      for (const id of bpeEncode(this.#assets.model, chunk.text.replaceAll(" ", METASPACE))) {
        ids.push(id);
      }
    }
    return ids;
  }

  /**
   * The same id sequence as `tokenizer.encode(text, add_special_tokens=True)`.
   *
   * What is added is decided by the asset's declaration (nothing for gemma4 / `<bos>` … `<eos>` for
   * EmbeddingGemma). The entry that spares callers from branching on the family.
   */
  encodeWithSpecialTokens(text: string): number[] {
    const ids = this.encode(text);
    if (this.#assets.spec.postProcessor === "none") return ids;
    return [this.#assets.bosId, ...ids, this.#assets.eosId];
  }

  /**
   * The same string as `tokenizer.decode(ids, …)` (goes through the same state machine as streaming
   * decoding).
   */
  decode(ids: Iterable<number>, options: GemmaDecodeOptions = {}): string {
    return detokenize(this.#decoderSource(options), ids);
  }

  /**
   * Builds a streaming decoder `push(id) → settled string` (one iteration of ADR 0083 decision 2 =
   * one push).
   */
  createDetokenizer(options: GemmaDecodeOptions = {}): StreamingDetokenizer {
    return new StreamingDetokenizer(this.#decoderSource(options));
  }

  #decoderSource(options: GemmaDecodeOptions): DetokenizerSource {
    const skipSpecialTokens = options.skipSpecialTokens ?? true;
    const { model, specialIds } = this.#assets;
    return {
      byteOf: (id) => model.byteOf.get(id),
      textOf: (id) => {
        if (skipSpecialTokens && specialIds.has(id)) return undefined;
        // Added vocabulary first (the same order as the reference's `id_to_token` — ids outside the
        // vocabulary table exist only there).
        const token = this.#tokenOfAdded.get(id) ?? model.tokenOf.get(id);
        if (token === undefined) {
          // The reference silently skips unknown ids; this fails (the cross-cutting invariant: fail
          // loudly).
          throw new Error(`復号: id ${id} が語彙にも追加語彙にも無い`);
        }
        // The first step of the decoder chain, `Replace("▁" → " ")`. byte_fallback spellings
        // contain no metaspace, so the order of this replacement and the byte check does not change
        // the result.
        return token.replaceAll(METASPACE, " ");
      },
    };
  }
}
