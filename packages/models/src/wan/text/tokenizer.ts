/**
 * Wan2.1's prompt layer — preprocessing (a mirror of `prompt_clean`) → the umT5 tokenizer (including
 * the accepted-set check) (ADR 0119 decisions 1, 2 and 4).
 *
 * The tokenizer and the accepted-set check are the single implementation shared with the standalone
 * umT5 (`Umt5Tokenizer` in `src/umt5/tokenizer.ts` — the source of truth for the path; the rejection
 * rules and their order are in that module's doc); Wan only runs upstream's preprocessing before it.
 * Where it departs from upstream is "never make ids silently", and each case is rejected with a
 * `ModelInputError` (an input error): input the preprocessing rejects (`PromptCleanError` —
 * `./prompt-clean.ts`), out-of-vocabulary characters (decision 1), added tokens in the text and `▁`
 * right after whitespace (U+2581 — forms whose id sequence splits between versions), more than 512
 * tokens (decision 4), and 1 token (empty or whitespace-only, giving only `</s>`). The shared check's
 * "whitespace other than U+0020, and a trailing U+0020" cannot happen on this path, because the
 * preprocessing collapses whitespace into a single U+0020 and strips both ends.
 *
 * The id sequence of an accepted input matches both paths, transformers 5.14.1 and `tokenizer.json`
 * (the recipe confirms this when generating the fixture, and `wan_text_tokenizer_test.ts` compares
 * with both).
 *
 * The asset (series `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json` — in the distribution, the
 * asset `umt5_tokenizer` of Wan's manifest) holds the vocabulary, scores, added tokens, whitespace set
 * and the preprocessing tables (`promptClean`) in one file. One run of the same recipe writes them,
 * so the tokenizer and the preprocessing cannot go stale separately. It does no I/O (fetching is the
 * hub's job).
 */

import { asRecord, asString } from "../../text/asset-gates.ts";
import { parseT5Tables, type T5Tables } from "../../text/t5-tokenizer.ts";
import { UMT5_MIN_TOKENS } from "../../umt5/relative-position.ts";
import { Umt5Tokenizer } from "../../umt5/tokenizer.ts";
import { cleanPrompt, parsePromptCleanTables, type PromptCleanTables } from "./prompt-clean.ts";

/** The version of the asset format (the recipe's `ASSET_FORMAT`). */
export const WAN_TOKENIZER_FORMAT = "karume-wan-umt5-tokenizer/1";

/**
 * The lower limit of the accepted set (an alias of `UMT5_MIN_TOKENS`, the lower limit of the umT5
 * graph's valid length — ADR 0119 decision 4. The value's source of truth is
 * `src/umt5/relative-position.ts`).
 */
export const WAN_PROMPT_MIN_TOKENS = UMT5_MIN_TOKENS;

/** Wan's tokenizer asset, parsed. */
export type WanTokenizerAssets = {
  readonly t5: T5Tables;
  readonly promptClean: PromptCleanTables;
};

/** Turns one prompt into a umT5 id sequence. */
export class WanPromptEncoder {
  readonly #promptClean: PromptCleanTables;
  readonly #tokenizer: Umt5Tokenizer;

  constructor(assets: WanTokenizerAssets) {
    this.#promptClean = assets.promptClean;
    this.#tokenizer = new Umt5Tokenizer(assets.t5);
  }

  /** The maximum length of the id sequence (including `</s>` — upstream's `max_sequence_length`). */
  get maxLength(): number {
    return this.#tokenizer.maxLength;
  }

  /** Preprocessing only (the same string as the output of upstream `prompt_clean`). */
  clean(prompt: string, label: string = "プロンプト"): string {
    return cleanPrompt(this.#promptClean, prompt, label);
  }

  /**
   * Preprocessing → encoding → accepted-set check. An i32 id sequence (ending in `</s>`, length 2 to
   * `maxLength`).
   *
   * The rejection order is the same as in the recipe's fixture generation: preprocessing → added
   * tokens → `▁` right after whitespace → out of vocabulary / over the limit (whichever is hit first
   * while accumulating from the start) → under the lower limit (after preprocessing it is the shared
   * implementation — `Umt5Tokenizer.tokenize`).
   */
  encode(prompt: string, label: string = "プロンプト"): Int32Array<ArrayBuffer> {
    return this.#tokenizer.tokenize(this.clean(prompt, label), label);
  }
}

/**
 * Converts the asset JSON (already `JSON.parse`d) into tables. It is an external boundary, so the
 * structure is checked before use (`fromPretrained(ref)` can point at any repo — the premise of
 * `src/text/asset-gates.ts`).
 */
export const parseWanTokenizerAsset = (
  raw: unknown,
  label: string = "tokenizer",
): WanTokenizerAssets => {
  const obj = asRecord(raw, label);
  const format = asString(obj["format"], `${label}.format`);
  if (format !== WAN_TOKENIZER_FORMAT) {
    // Do not read an unknown version silently (if a field's meaning changed, there would be no
    // exception and only the id sequence would change).
    throw new Error(`${label}: 形式 ${format} を読めない（${WAN_TOKENIZER_FORMAT} だけを読む）`);
  }
  return {
    t5: parseT5Tables(obj, label),
    promptClean: parsePromptCleanTables(obj["promptClean"], `${label}.promptClean`),
  };
};
