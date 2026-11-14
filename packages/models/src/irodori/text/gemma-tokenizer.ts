/**
 * The text tokenizer of Irodori-TTS v4-Large: the Gemma BPE tokenizer of T5Gemma 2
 * (`google/t5gemma-2-1b-1b`), read through the shared Gemma reader (`src/gemma/text/`) and adapted
 * to Irodori's surface ({@link IrodoriTextTokenizer}).
 *
 * Upstream wraps the `transformers` tokenizer in the same `PretrainedTextTokenizer` as the Small
 * models, so the back end is Irodori's, not Gemma's: the body is encoded with
 * `add_special_tokens=False` (the asset's post_processor — `bos` for this tokenizer — never runs),
 * and Irodori prepends `bos_token_id` and right-pads with `pad_token_id` itself
 * (`encodePaddedIds` of `./text-tokenizer.ts`).
 *
 * - **pad**: the Gemma asset has no `padId` field; the pad id is the added token `<pad>` (the
 *   recipe checks that it agrees with the checkpoint config and with upstream's wrapper —
 *   `check_gemma_special_ids` of `tools/export-recipes/irodori/tokenizer_ref.py`). An asset
 *   without `<pad>` fails at construction.
 * - **No rare-token split** (ADR 0149): upstream enables it only for the Small models' Unigram
 *   tokenizer, and its splitter refuses a BPE model. A split request fails loudly with the same
 *   message as an Unigram asset without the table.
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import { assertEncodableText } from "../../text/code-points.ts";
import type { GemmaTokenizerAssets } from "../../gemma/text/asset.ts";
import { Tokenizer } from "../../gemma/text/tokenizer.ts";
import {
  type IrodoriEncodeOptions,
  type IrodoriTextTokenizer,
  RARE_TOKEN_SPLIT_MISSING,
} from "./text-tokenizer.ts";

/** The added token upstream pads with (`pad_token` of the T5Gemma 2 tokenizer). */
const PAD_TOKEN = "<pad>";

export class IrodoriGemmaTokenizer implements IrodoriTextTokenizer {
  readonly #tokenizer: Tokenizer;
  readonly #bosId: number;
  readonly #padId: number;

  constructor(assets: GemmaTokenizerAssets) {
    this.#tokenizer = new Tokenizer(assets);
    this.#bosId = assets.bosId;
    const padId = assets.addedTokens.get(PAD_TOKEN);
    if (padId === undefined) {
      throw new Error(
        `irodori tokenizer: the Gemma asset has no added token ${PAD_TOKEN} (the pad id upstream uses)`,
      );
    }
    this.#padId = padId;
  }

  /** The asset's `bosId`. */
  get bosId(): number {
    return this.#bosId;
  }

  /** The id of the added token `<pad>`. */
  get padId(): number {
    return this.#padId;
  }

  /** Always `false` — this kind has no rare-token split (module doc). */
  get hasRareTokenSplit(): boolean {
    return false;
  }

  /**
   * Normalized body → the same ids as upstream's `tokenizer.encode(text, add_special_tokens=False)`
   * (added tokens typed in the text, special ones included, are still cut out, as upstream does).
   */
  encode(text: string, options: IrodoriEncodeOptions): number[] {
    // The acceptance check of the body runs under Irodori's label (input-caused per ADR 0107
    // decision 2), the same as the Unigram kind.
    assertEncodableText(text, "irodori text");
    if (options.splitRareTokens) throw new Error(RARE_TOKEN_SPLIT_MISSING);
    return this.#tokenizer.encode(text);
  }
}
