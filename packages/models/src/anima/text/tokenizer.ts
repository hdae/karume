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
import {
  asNumber,
  asPositiveInteger,
  asRecord,
  assertCodePoint,
  assertUniqueLines,
  asString,
  parseAddedTokens,
  setUnique,
} from "../../text/asset-gates.ts";
import { assertEncodableText } from "../../text/code-points.ts";
import { parseCodeRanges } from "../../text/code-ranges.ts";
import { parseT5Tables } from "../../text/t5-tokenizer.ts";
import {
  type Qwen2Assets,
  type Qwen2CaseFold,
  type Qwen2CharClasses,
  Qwen2Tokenizer,
} from "./qwen2-tokenizer.ts";
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
        `${label}の ${which} id 列が ${length} トークン（最低 ${PROMPT_MIN_TOKENS}）— ` +
          `空文字や空白だけのプロンプトは受理集合 Dim(min=${PROMPT_MIN_TOKENS}) の外。` +
          "1 語以上入れる。",
      );
    }
    if (length > maxLength) {
      // The upper side is not input-caused (ADR 0107 decision 2) — both tokenizers truncate to
      // `maxLength` before returning, so this is reached only when truncation is broken = a broken
      // internal invariant. Shortening the prompt would not fix it, so a plain `Error` is thrown.
      throw new Error(`${label}の ${which} id 列が ${length} トークン（上限 ${maxLength}）`);
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
        `トークナイザ資産の maxLength が食い違う（Qwen2 ${qwen2Assets.maxLength} / ` +
          `T5 ${t5Assets.maxLength}）`,
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
  encode(prompt: string, label: string = "プロンプト"): AnimaPromptIds {
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

const parseClasses = (raw: unknown, label: string): Qwen2CharClasses => {
  const obj = asRecord(raw, label);
  return {
    letter: parseCodeRanges(obj["letter"], `${label}.letter`),
    number: parseCodeRanges(obj["number"], `${label}.number`),
    space: parseCodeRanges(obj["space"], `${label}.space`),
  };
};

/** Equivalence table of `(?i:)` (pairs of `[cp, the character on the suffix side]`). */
const parseCaseFold = (raw: unknown, label: string): Qwen2CaseFold => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<number, number>();
  for (const entry of raw) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "number"
    ) {
      throw new Error(`${label}: [cp, cp] でない`);
    }
    assertCodePoint(entry[0], label);
    assertCodePoint(entry[1], label);
    setUnique(out, entry[0], entry[1], label);
  }
  return out;
};

/**
 * Reads a newline-separated table and checks that the line count matches what the asset declares.
 *
 * MUST: compare the counts. If `mergesText` is missing lines, BPE just "does not merge what it
 * cannot find" and **does not fail** — the split changes and the id sequence silently becomes a
 * different one (the `vocabText` side fails on a token missing from the vocabulary, but this one is
 * a silent wrong value).
 */
const parseLines = (raw: unknown, count: unknown, label: string, countLabel: string): string[] => {
  const lines = asString(raw, label).split("\n");
  const want = asNumber(count, countLabel);
  if (lines.length !== want) {
    throw new Error(
      `${label}: 行数 ${lines.length} が ${countLabel} の ${want} と違う（資産が壊れている）`,
    );
  }
  return lines;
};

/**
 * Converts the Qwen2 asset JSON into asset tables (an external boundary, so the structure is
 * checked before use).
 */
const parseQwen2Asset = (raw: unknown, label: string = "tokenizer"): Qwen2Assets => {
  const obj = asRecord(raw, label);
  // 0-origin line number = id / rank. The byte-level vocabulary contains no newline or space
  // (checked at emit).
  const vocabLines = parseLines(
    obj["vocabText"],
    obj["vocabCount"],
    `${label}.vocabText`,
    `${label}.vocabCount`,
  );
  assertUniqueLines(vocabLines, `${label}.vocabText`);
  const vocab = new Map<string, number>();
  for (const [id, token] of vocabLines.entries()) vocab.set(token, id);
  const mergesLines = parseLines(
    obj["mergesText"],
    obj["mergesCount"],
    `${label}.mergesText`,
    `${label}.mergesCount`,
  );
  assertUniqueLines(mergesLines, `${label}.mergesText`);
  const merges = new Map<string, number>();
  for (const [rank, line] of mergesLines.entries()) merges.set(line, rank);
  return {
    vocab,
    merges,
    addedTokens: parseAddedTokens(obj["addedTokens"], `${label}.addedTokens`),
    classes: parseClasses(obj["classes"], `${label}.classes`),
    caseFold: parseCaseFold(obj["caseFold"], `${label}.caseFold`),
    nfcSegments: parseCodeRanges(obj["nfcSegments"], `${label}.nfcSegments`),
    maxLength: asPositiveInteger(obj["maxLength"], `${label}.maxLength`),
  };
};

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
    throw new Error(`${label}: UTF-8 として読めない`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`${label}: JSON として読めない`, { cause });
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
    parseQwen2Asset(decodeJson(qwen2Bytes, "tokenizer")),
    parseT5Asset(decodeJson(t5Bytes, "tokenizer_2")),
  );
