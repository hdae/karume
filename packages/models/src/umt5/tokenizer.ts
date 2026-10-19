/**
 * The umT5-XXL tokenizer — encoding and the accepted-set check (the rules of ADR 0119 decisions 1
 * and 4, in a form used without Wan's preprocessing).
 *
 * The source of truth for the path is the pinned transformers 5.14.1 `AutoTokenizer` (`tokenizer.json`
 * and `tokenizer_config.json` of the original `google/umt5-xxl`), and its shape is exactly the shared
 * T5 layer (`src/text/t5-tokenizer.ts` — no normalization, WhitespaceSplit → Metaspace → Unigram →
 * trailing `</s>`). It departs from upstream only in "never make ids silently", and each case is
 * rejected with a `ModelInputError` (an input error). The rejection order (the same as the fixture
 * generation in the recipe `wan/umt5_xxl_tokenizer.py`):
 *
 * 1. **Added tokens in the text** (`</s>`, `<extra_id_*>` etc.) and 2. **`▁` (U+2581) right after
 *    whitespace** — transformers 5 splits on whitespace first, while `tokenizer.json` replaces
 *    whitespace with `▁` before splitting, so for these two forms the id sequence splits between
 *    versions (`a </s> b` gives `[289, 1, 748, 1]` and `[289, 273, 1, 748, 1]`).
 * 3. **Whitespace other than U+0020, and a trailing U+0020** — transformers 5 drops it as a
 *    separator, while `tokenizer.json` keeps it as a character (`a\tb` gives `[289, 748, 1]` and
 *    `[289, 3, 485, 1]`; `a ` gives `[289, 1]` and `[289, 273, 1]`). The whitespace set is the
 *    asset's `space` (WhitespaceSplit's whitespace set). A leading U+0020 and runs of U+0020 between
 *    words do not split (`tokenizer.json`'s normalizer collapses them), so they are accepted. No trim
 *    or collapsing is added (that would become a second preprocessing).
 * 4. **Out-of-vocabulary characters** (decision 1) and **more than 512 tokens** (decision 4 —
 *    upstream truncates silently) — whichever is hit first while accumulating from the start. The id
 *    of an out-of-vocabulary character splits between transformers 5 (id 2 = `<s>`) and
 *    `tokenizer.json` / transformers 4.x (id 3 = `<unk>`).
 * 5. **Fewer than 2 tokens** (an empty string gives only `</s>`) — the lower limit of the graph's
 *    valid length ({@link UMT5_MIN_TOKENS} in `./relative-position.ts`).
 *
 * The id sequence of an accepted input matches both paths, transformers 5.14.1 and `tokenizer.json`
 * (the recipe confirms this when generating the fixture, and `umt5_tokenizer_test.ts` compares with
 * both).
 *
 * Wan (`WanPromptEncoder` in `src/wan/text/tokenizer.ts`) passes in the string after a mirror of
 * upstream's preprocessing `prompt_clean`. The preprocessing collapses whitespace into a single
 * U+0020 and strips both ends, so rejection 3 cannot happen on Wan's path (the whitespace table of
 * Wan's asset equals the collapsing set — `umt5_tokenizer_test.ts` checks this).
 *
 * The asset (format {@link UMT5_TOKENIZER_FORMAT} — the per-model asset `tokenizer` of the
 * distribution `karume-umt5-xxl`) holds the vocabulary, scores, added tokens, whitespace set and
 * maximum length in one file (written by the recipe `wan/umt5_xxl_tokenizer.py`). It does no I/O
 * (fetching is the hub's job).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { ModelInputError } from "../errors.ts";
import { type AddedTokenMatcher, createAddedTokenMatcher } from "../text/added-tokens.ts";
import { asRecord, asString } from "../text/asset-gates.ts";
import { assertEncodableText } from "../text/code-points.ts";
import { type CodeRanges, inCodeRanges } from "../text/code-ranges.ts";
import {
  METASPACE,
  parseT5Tables,
  type T5Tables,
  T5UnigramTokenizer,
} from "../text/t5-tokenizer.ts";
import { UMT5_MIN_TOKENS } from "./relative-position.ts";

/** The version of the asset format (`ASSET_FORMAT` in the recipe `wan/umt5_xxl_tokenizer.py`). */
export const UMT5_TOKENIZER_FORMAT = "karume-umt5-tokenizer/1";

/** The only whitespace accepted (U+0020). */
const SPACE = " ";

/** A Metaspace right after whitespace (a form whose segmentation splits between the two paths). */
const SPACE_METASPACE = SPACE + METASPACE;

/** Display form of a code point (`U+0009` — the rejection message names the character to fix). */
const codePointLabel = (cp: number): string =>
  `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`;

/** Turns one string into a umT5 id sequence (outside the accepted set: {@link ModelInputError}). */
export class Umt5Tokenizer {
  readonly #t5: T5UnigramTokenizer;
  readonly #added: AddedTokenMatcher;
  readonly #space: CodeRanges;
  readonly #maxLength: number;

  constructor(tables: T5Tables) {
    this.#t5 = new T5UnigramTokenizer(tables, { unknown: "reject", overflow: "reject" });
    this.#added = createAddedTokenMatcher([...tables.addedTokens.keys()]);
    this.#space = tables.space;
    this.#maxLength = tables.maxLength;
  }

  /** The maximum length of the id sequence (including `</s>` — upstream's `max_length`). */
  get maxLength(): number {
    return this.#maxLength;
  }

  /**
   * Encoding → accepted-set check. An i32 id sequence (ending in `</s>`, length
   * {@link UMT5_MIN_TOKENS} to `maxLength`). The rejection order is 1 to 5 in the module doc. `label`
   * is the subject of the rejection message.
   */
  tokenize(text: string, label: string = "text"): Int32Array<ArrayBuffer> {
    assertEncodableText(text, label);
    for (const chunk of this.#added.split(text)) {
      if (chunk.added) {
        throw new ModelInputError(
          `${label}: contains special token ${JSON.stringify(chunk.text)} — special tokens in the text` +
            " are not accepted because the id sequence differs between upstream versions (remove that spelling)",
        );
      }
    }
    if (text.includes(SPACE_METASPACE)) {
      throw new ModelInputError(
        `${label}: ${METASPACE} (U+2581) right after whitespace — it is SentencePiece's whitespace marker,` +
          " not accepted because the segmentation differs between upstream versions (remove the whitespace or the ▁)",
      );
    }
    this.#assertPlainSpaces(text, label);
    const ids = this.#t5.encode(text, label);
    if (ids.length < UMT5_MIN_TOKENS) {
      throw new ModelInputError(
        `${label}: the id sequence has ${ids.length} tokens (minimum ${UMT5_MIN_TOKENS}) — an empty or` +
          " whitespace-only prompt is not accepted. Enter at least one word",
      );
    }
    return Int32Array.from(ids);
  }

  /** Rejects whitespace other than U+0020 and a trailing U+0020 (3 in the module doc). */
  #assertPlainSpaces(text: string, label: string): void {
    for (const ch of text) {
      const cp = ch.codePointAt(0) as number;
      if (ch !== SPACE && inCodeRanges(this.#space, cp)) {
        const found = codePointLabel(cp);
        throw new ModelInputError(
          `${label}: contains whitespace ${found} other than U+0020 — with a tokenizer that does no preprocessing,` +
            " it is not accepted because the id sequence differs between upstream versions (replace it with U+0020)",
        );
      }
    }
    if (text.endsWith(SPACE)) {
      throw new ModelInputError(
        `${label}: ends in whitespace (U+0020) — with a tokenizer that does no preprocessing, it is not accepted because the id sequence differs` +
          " between upstream versions (remove the trailing whitespace)",
      );
    }
  }
}

/**
 * Converts the asset JSON (already `JSON.parse`d — format {@link UMT5_TOKENIZER_FORMAT}) into T5
 * tables. It is an external boundary, so the structure is checked before use (`fromPretrained(ref)`
 * can point at any repo — the premise of `src/text/asset-gates.ts`).
 *
 * The provenance fields (`source` / `versions`) are not read — they are checked by the writer's check
 * (the recipe `wan/umt5_distribution.py`) when it builds the distribution, and the id sequence at run
 * time depends only on the tables (the same treatment as the reader of Wan's asset).
 */
export const parseUmt5TokenizerAsset = (raw: unknown, label: string = "tokenizer"): T5Tables => {
  const obj = asRecord(raw, label);
  const format = asString(obj["format"], `${label}.format`);
  if (format !== UMT5_TOKENIZER_FORMAT) {
    // Do not read an unknown version silently (if a field's meaning changed, there would be no
    // exception and only the id sequence would change).
    throw new Error(
      `${label}: cannot read format ${format} (only ${UMT5_TOKENIZER_FORMAT} is read)`,
    );
  }
  return parseT5Tables(obj, label);
};
