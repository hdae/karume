/**
 * The Qwen3.5 tokenizer — encoding and the length policy, as MOSS-Image's text encoder (the text
 * tower of Qwen3.5-9B) receives its input.
 *
 * Shared across families like `src/umt5/` (which Wan imports), but **no public subpath export yet**:
 * its only consumer, the MOSS-Image pipeline, is not on main (branch `wip/moss-image` imports this
 * module after a rebase). The subpath is added together with the first published consumer.
 *
 * ## Source of truth (measured 2026-10-31)
 *
 * The id sequence is that of transformers 5.14.1 `AutoTokenizer` over the tokenizer MOSS-Image
 * pins and loads (`OpenMOSS-Team/MOSS-Image-SFT@f972bd4b…`, subfolder `tokenizer`, class
 * `Qwen2Tokenizer`), called as upstream's `encode_prompt` calls it:
 * `tokenizer(formatted, truncation=True, max_length=4096)` — no padding, no special tokens added.
 * The encoding itself is the shared byte-level BPE (`src/text/qwen2-tokenizer.ts`, the same engine
 * as Anima's Qwen3 tokenizer): same NFC normalizer, same pre-tokenizer expression, its own
 * vocabulary (248,044 entries), merges (247,587) and 33 added tokens (ids 248044–248076).
 *
 * NOTE: the pre-tokenizer expression is the one transformers 5.14.1 runs: `Qwen2Tokenizer` rebuilds
 * it from its class constant, without `\p{M}`. The official `Qwen/Qwen3.5-9B@c2022362`
 * `tokenizer.json` (and `tokenizer_config.json`'s `pretokenize_regex`) carry `[\p{L}\p{M}]+` instead,
 * which `tokenizers.Tokenizer.from_file` would run — the two split differently on text with
 * combining marks that NFC does not compose (Devanagari vowel signs, for one: `नमस्ते दुनिया` is 10
 * ids here and 6 there). MOSS-Image's embeddings were made through transformers, so this module
 * follows transformers; transformers builds the same backend from either repo's files (compared
 * 2026-10-31). A newer transformers that honours the file's expression would change ids for such
 * text — the recipe's emit gate compares with the pinned version only.
 *
 * ## Special-token spellings in the text
 *
 * They are cut out as control tokens, exactly as upstream (`split_special_tokens=False`, all 33
 * added tokens matched leftmost-longest before normalisation, special or not): a prompt that types
 * `<|im_end|>` or `<think>` yields those ids, not their characters. This differs from the umT5
 * tokenizer (which rejects them) because here the split is the same in every upstream path, so the
 * id sequence is not ambiguous.
 *
 * ## Length (the maintainer's ruling 50, 2026-10-31 — see `Qwen3_5Overflow`)
 *
 * Upstream truncates **silently** at `max_length` = 4096 and so drops the template's tail
 * (`<|im_end|>…<think>\n`). Here the default rejects an over-length input with a
 * {@link ModelInputError} (the repository invariant "unsupported input fails loudly");
 * `overflow: "truncate"` reproduces upstream's ids exactly. The bound is the asset's `maxLength`
 * (4096, written by the recipe from upstream's `max_sequence_length`).
 *
 * The asset (format {@link QWEN3_5_TOKENIZER_FORMAT}) is written by the recipe
 * `tools/export-recipes/qwen3_5/tokenizer.py`; this module does no I/O (fetching is the hub's job).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import { ModelInputError } from "../errors.ts";
import { asRecord, asString } from "../text/asset-gates.ts";
import { assertEncodableText } from "../text/code-points.ts";
import { parseQwen2Tables, type Qwen2Assets, Qwen2Tokenizer } from "../text/qwen2-tokenizer.ts";

/** The version of the asset format (`ASSET_FORMAT` in the recipe `qwen3_5/tokenizer.py`). */
export const QWEN3_5_TOKENIZER_FORMAT = "karume-qwen3_5-tokenizer/1";

/**
 * What to do with an input longer than the asset's `maxLength`.
 *
 * - `"reject"` (default) — throw a {@link ModelInputError}; nothing is encoded past the bound.
 * - `"truncate"` — keep the first `maxLength` ids, as upstream's `truncation=True` does (for a
 *   rendered chat prompt this drops the template's tail, so the encoder sees no `<|im_end|>`).
 */
export type Qwen3_5Overflow = "reject" | "truncate";

export type Qwen3_5TokenizeOptions = {
  /** Over-length policy (default `"reject"`). */
  readonly overflow?: Qwen3_5Overflow;
  /** The subject of the rejection messages (default `"text"`). */
  readonly label?: string;
};

const OVERFLOWS: readonly string[] = ["reject", "truncate"] satisfies readonly Qwen3_5Overflow[];

/** Turns one string into a Qwen3.5 id sequence (outside the accepted set: {@link ModelInputError}). */
export class Qwen3_5Tokenizer {
  readonly #bpe: Qwen2Tokenizer;
  readonly #maxLength: number;

  constructor(assets: Qwen2Assets) {
    this.#bpe = new Qwen2Tokenizer(assets);
    this.#maxLength = assets.maxLength;
  }

  /** The maximum length of the id sequence (upstream's `max_length`). */
  get maxLength(): number {
    return this.#maxLength;
  }

  /**
   * The same id sequence as `tokenizer(text)` upstream, as i32 (graph inputs are i32 — ADR 0009).
   * The text is tokenized as given: a chat prompt is rendered first (`./chat.ts`). An empty text
   * gives an empty sequence (a rendered prompt never is — the template alone is 10 tokens).
   */
  tokenize(text: string, options: Qwen3_5TokenizeOptions = {}): Int32Array<ArrayBuffer> {
    const label = options.label ?? "text";
    const overflow = options.overflow ?? "reject";
    if (!OVERFLOWS.includes(overflow)) {
      throw new ModelInputError(
        `${label}: unknown overflow policy ${JSON.stringify(overflow)} (one of ${OVERFLOWS.join(" / ")})`,
      );
    }
    assertEncodableText(text, label);
    // One id past the bound tells "exactly maxLength" from "longer" without encoding the rest.
    const ids = this.#bpe.encode(text, this.#maxLength + 1);
    if (ids.length <= this.#maxLength) return Int32Array.from(ids);
    if (overflow === "reject") {
      throw new ModelInputError(
        `${label}: the token count exceeds the upper bound ${this.#maxLength} — it is not silently` +
          " truncated (upstream truncates and drops the end of the chat template). Shorten the" +
          ' prompt, or pass overflow: "truncate" to cut it as upstream does',
      );
    }
    return Int32Array.from(ids.slice(0, this.#maxLength));
  }
}

/**
 * Converts the asset JSON (already `JSON.parse`d — format {@link QWEN3_5_TOKENIZER_FORMAT}) into
 * the BPE tables. It is an external boundary, so the structure is checked before use
 * (`fromPretrained(ref)` can point at any repo — the premise of `src/text/asset-gates.ts`).
 *
 * The provenance fields (`source` / `versions`) are not read: the id sequence at run time depends
 * only on the tables, and the recipe checks provenance when it writes the asset (the same treatment
 * as the umT5 tokenizer asset).
 */
export const parseQwen3_5TokenizerAsset = (
  raw: unknown,
  label: string = "tokenizer",
): Qwen2Assets => {
  const obj = asRecord(raw, label);
  const format = asString(obj["format"], `${label}.format`);
  if (format !== QWEN3_5_TOKENIZER_FORMAT) {
    // Do not read an unknown version silently (if a field's meaning changed, there would be no
    // exception and only the id sequence would change).
    throw new Error(
      `${label}: cannot read format ${format} (only ${QWEN3_5_TOKENIZER_FORMAT} is read)`,
    );
  }
  return parseQwen2Tables(obj, label);
};
