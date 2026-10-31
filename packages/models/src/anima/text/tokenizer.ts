/**
 * Anima's prompt layer — interpreting the two tokenizer assets and encoding one prompt.
 *
 * Anima's text layer **passes the same string through two tokenizers independently**:
 *
 *   ① Qwen2 BPE (the manifest's `tokenizer`)    → text_encoder (Qwen3) → source_hidden_states
 *   ② T5 Unigram (the manifest's `tokenizer_2`) → text_conditioner     → target_input_ids
 *
 * Distribution: vocabularies, merges and normalisation tables are JSON baked by the exporter (not
 * the raw `tokenizer.json`, 13.8MB in total, but a 4.6MB form holding only what execution needs).
 * **Fetching is the hub's job**, so this module only takes bytes and has no I/O (the same path in
 * the browser, on Node and on Deno).
 */

import { ModelInputError } from "../../errors.ts";
import { asRecord } from "../../text/asset-gates.ts";
import { assertEncodableText } from "../../text/code-points.ts";
import { parseQwen2Tables, type Qwen2Assets, Qwen2Tokenizer } from "../../text/qwen2-tokenizer.ts";
import { parseT5Tables } from "../../text/t5-tokenizer.ts";
import { parseSpmTables } from "./spm-normalizer.ts";
import { type T5Assets, T5Tokenizer } from "./t5-tokenizer.ts";

/** Encoding result of one prompt (graph inputs are i32 — ADR 0009). */
type AnimaPromptIds = {
  readonly qwenIds: Int32Array<ArrayBuffer>;
  readonly t5Ids: Int32Array<ArrayBuffer>;
};

/**
 * Lower bound of the accepted set. Exactly the exporter's `Dim("Tsrc", min=2)` /
 * `Dim("Ttgt", min=2)` (a lower bound to avoid `torch.export`'s 0/1 specialisation; a one-token
 * input **cannot be fed to the graph**). An empty or whitespace-only prompt becomes length 1 in T5
 * (only `</s>`), and a one-character prompt becomes one token in Qwen2 — both are rejected here.
 */
export const PROMPT_MIN_TOKENS = 2;

/**
 * Checks that the prompt length is in the accepted set (a violation fails loudly).
 *
 * The upper side also checks "truncation is in effect": T5 truncates before appending `</s>`, so
 * getting the order wrong yields `maxLength + 1` (no shape error; silent until the conditioner's
 * 512-padding check). The thrown type also differs between the bounds — the lower bound is
 * input-caused (the caller passes by adding one word), the upper bound is an internal invariant
 * that is reached only when truncation is broken.
 */
export const assertPromptTokenLengths = (
  label: string,
  qwenLength: number,
  t5Length: number,
  maxLength: number,
): void => {
  for (const [which, length] of [
    ["Qwen2", qwenLength],
    ["T5", t5Length],
  ] as const) {
    if (length < PROMPT_MIN_TOKENS) {
      throw new ModelInputError(
        `${label}: the ${which} id sequence is ${length} tokens (minimum ${PROMPT_MIN_TOKENS}) — ` +
          `an empty or whitespace-only prompt is outside the accepted set Dim(min=${PROMPT_MIN_TOKENS}). ` +
          "Enter at least one word.",
      );
    }
    if (length > maxLength) {
      // The upper side is not input-caused (ADR 0107 decision 2) — both tokenizers truncate to
      // `maxLength` before returning, so this is reached only when truncation is broken = a broken
      // internal invariant. Shortening the prompt would not fix it, so a plain `Error` is thrown.
      throw new Error(
        `${label}: the ${which} id sequence is ${length} tokens (upper bound ${maxLength})`,
      );
    }
  }
};

/** The two tokenizers Anima's text layer needs. */
export class AnimaTokenizers {
  readonly #qwen2: Qwen2Tokenizer;
  readonly #t5: T5Tokenizer;
  readonly #maxLength: number;

  constructor(qwen2Assets: Qwen2Assets, t5Assets: T5Assets) {
    if (qwen2Assets.maxLength !== t5Assets.maxLength) {
      // Only one side holds a stale asset. With different truncation lengths, only the id sequence
      // lengths disagree on long prompts, which surfaces no earlier than the conditioner's
      // 512-padding check.
      throw new Error(
        `The maxLength of the tokenizer assets does not match (Qwen2 ${qwen2Assets.maxLength} / ` +
          `T5 ${t5Assets.maxLength})`,
      );
    }
    this.#qwen2 = new Qwen2Tokenizer(qwen2Assets);
    this.#t5 = new T5Tokenizer(t5Assets);
    this.#maxLength = qwen2Assets.maxLength;
  }

  /** Truncation length matching the canonical call (512). */
  get maxLength(): number {
    return this.#maxLength;
  }

  /**
   * Passes one prompt through both tokenizers. A length outside the accepted set fails here
   * (`label` appears in that message — so it tells the positive from the negative prompt).
   */
  encode(prompt: string, label: string = "prompt"): AnimaPromptIds {
    // The acceptance check of the text itself runs **before it is passed to the tokenizers**
    // (input-caused per ADR 0107 decision 2). Parsing the vocabulary assets passes the same
    // condition, but a violation there is an asset inconsistency, so the type differs.
    assertEncodableText(prompt, label);
    const qwenIds = Int32Array.from(this.#qwen2.encode(prompt));
    const t5Ids = Int32Array.from(this.#t5.encode(prompt));
    assertPromptTokenLengths(label, qwenIds.length, t5Ids.length, this.#maxLength);
    return { qwenIds, t5Ids };
  }
}

/**
 * Converts the T5 asset JSON into asset tables (the shared T5 tables + Anima's Precompiled
 * normalisation table).
 */
const parseT5Asset = (raw: unknown, label: string = "tokenizer_2"): T5Assets => ({
  ...parseT5Tables(raw, label),
  normalizer: parseSpmTables(asRecord(raw, label)["normalizer"], `${label}.normalizer`),
});

const decodeJson = (bytes: Uint8Array<ArrayBuffer>, label: string): unknown => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error(`${label}: cannot be read as UTF-8`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`${label}: cannot be read as JSON`, { cause });
  }
};

/**
 * Builds the two tokenizers from already fetched bytes (the manifest's `tokenizer` /
 * `tokenizer_2`). No I/O — fetching is the hub's job.
 */
export const createTokenizers = (
  qwen2Bytes: Uint8Array<ArrayBuffer>,
  t5Bytes: Uint8Array<ArrayBuffer>,
): AnimaTokenizers =>
  new AnimaTokenizers(
    parseQwen2Tables(decodeJson(qwen2Bytes, "tokenizer"), "tokenizer"),
    parseT5Asset(decodeJson(t5Bytes, "tokenizer_2")),
  );
