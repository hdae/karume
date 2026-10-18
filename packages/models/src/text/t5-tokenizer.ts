/**
 * Tokenizer for the T5 family (SentencePiece Unigram assembled with `tokenizers`) — shared across
 * families.
 *
 * The source-of-truth path (anima's T5 and Wan's umT5 have the same shape in transformers 5.14.1):
 *
 *   AddedVocabulary (cuts out special tokens leftmost-longest) → normalization (per family — anima
 *   uses Precompiled, umT5 none) → WhitespaceSplit → Metaspace (replacement=▁ /
 *   prepend_scheme=always / split) → Unigram (Viterbi, no byte_fallback, fuse_unk) →
 *   TemplateProcessing (`</s>` at the end)
 *
 * The tables (vocabulary, added vocabulary, whitespace set) come from the assets, and the three
 * points where families differ come through {@link T5Policy}: whether to normalize, how to treat
 * out-of-vocabulary text (fuse into unk / reject), and how to treat input over the limit (truncate
 * / reject). anima follows upstream with "unk, truncate"; Wan's umT5 uses "reject, reject" for an
 * id sequence that does not depend on the version (ADR 0119 decision 1, decision 4).
 *
 * The Unigram core (Viterbi, tie handling, fusion) lives in `unigram.ts`, and cutting out the added
 * vocabulary in `added-tokens.ts`.
 */

import { ModelInputError } from "../errors.ts";
import { type AddedTokenMatcher, createAddedTokenMatcher } from "./added-tokens.ts";
import {
  asFiniteNumber,
  asPositiveInteger,
  asRecord,
  assertUniqueLines,
  asString,
  asVocabId,
  parseAddedTokens,
} from "./asset-gates.ts";
import { toCodePoints } from "./code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "./code-ranges.ts";
import {
  type UnigramModel,
  unigramSegments,
  unigramTokenize,
  type UnigramVocabEntry,
} from "./unigram.ts";

/** Metaspace replacement character (U+2581). */
export const METASPACE = "▁";

export type T5VocabEntry = UnigramVocabEntry;

/** Tables of the T5-family tokenizer (from the assets). */
export type T5Tables = {
  readonly vocab: ReadonlyMap<string, T5VocabEntry>;
  /**
   * Minimum score of the **whole** vocabulary. Pass the whole-vocabulary value even with a subset
   * (the unknown-node weight).
   */
  readonly minScore: number;
  /** Code-point count of the **whole** vocabulary's longest token (the prefix-search width). */
  readonly maxTokenLength: number;
  readonly unkId: number;
  readonly eosId: number;
  readonly addedTokens: ReadonlyMap<string, number>;
  /**
   * WhitespaceSplit's whitespace set (a table the exporter folded by asking the source of truth one
   * character at a time).
   */
  readonly space: CodeRanges;
  /** Maximum length of the id sequence including `</s>` (upstream `max_length`). */
  readonly maxLength: number;
};

/** The three points where families call upstream differently. */
export type T5Policy = {
  /**
   * Normalization applied to each piece after the added vocabulary is cut out (upstream's
   * normalizer). None if absent.
   */
  readonly normalize?: (text: string) => string;
  /**
   * Out-of-vocabulary (Unigram's unknown node): `"unk"` = fuse into one unk_id / `"reject"`
   * = reject.
   */
  readonly unknown: "unk" | "reject";
  /** Input over `maxLength`: `"truncate"` = cut leaving room for `</s>` / `"reject"` = reject. */
  readonly overflow: "truncate" | "reject";
};

/**
 * WhitespaceSplit → Metaspace. Whitespace is dropped and ▁ is prepended to each piece.
 *
 * MUST: `split=true` is MergedWithNext — the separating ▁ attaches to the head of the **next**
 * piece. Attaching it to "the tail of the previous piece" changes the segmentation.
 */
export const t5PreTokenize = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  for (const word of splitOnSpace(text, space)) {
    let piece = word.replaceAll(" ", METASPACE);
    if (!piece.startsWith(METASPACE)) piece = METASPACE + piece;
    let start = 0;
    for (let idx = 1; idx < piece.length; idx++) {
      if (piece[idx] === METASPACE) {
        out.push(piece.slice(start, idx));
        start = idx;
      }
    }
    out.push(piece.slice(start));
  }
  return out.filter((piece) => piece !== "");
};

const splitOnSpace = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  let buffer = "";
  for (const ch of text) {
    if (inCodeRanges(space, ch.codePointAt(0) as number)) {
      if (buffer !== "") {
        out.push(buffer);
        buffer = "";
      }
    } else {
      buffer += ch;
    }
  }
  if (buffer !== "") out.push(buffer);
  return out;
};

/**
 * Display of an unknown piece (for the out-of-vocabulary rejection message — in a form that lets a
 * person find what to fix).
 */
const describeText = (text: string): string =>
  `${JSON.stringify(text)}（${toCodePoints(text)
    .map((cp) => `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`)
    .join(" ")}）`;

export class T5UnigramTokenizer {
  readonly #tables: T5Tables;
  readonly #policy: T5Policy;
  readonly #added: AddedTokenMatcher;

  constructor(tables: T5Tables, policy: T5Policy) {
    this.#tables = tables;
    this.#policy = policy;
    this.#added = createAddedTokenMatcher([...tables.addedTokens.keys()]);
  }

  /**
   * One piece → id sequence (out-of-vocabulary text is fused into unk or rejected per the policy).
   */
  #tokenize(piece: string, label: string): number[] {
    // T5Tables satisfies the UnigramModel surface as is (no byteBaseId = no byte_fallback).
    const model = this.#tables satisfies UnigramModel;
    const cps = toCodePoints(piece);
    if (this.#policy.unknown === "unk") return unigramTokenize(model, cps);
    const ids: number[] = [];
    for (const segment of unigramSegments(model, cps)) {
      if (segment.entry === undefined) {
        throw new ModelInputError(
          `${label}: 語彙に無い文字列 ${describeText(segment.text)} を含む — 未知トークンへ` +
            "黙って潰さない。その文字を除くか言い換える",
        );
      }
      ids.push(segment.entry.id);
    }
    return ids;
  }

  /**
   * The same id sequence as `tokenizer([text], max_length=maxLength, truncation=True)` (input over
   * the limit per the policy).
   *
   * MUST: truncate after leaving room for `</s>` (the source of truth truncates **before** the
   * post_processor). Appending after `slice(0, maxLength)` gives `maxLength + 1` ids. As a result
   * the id sequence always ends with `</s>` and its length is always at least 1.
   *
   * `label` appears in rejection messages (a form that tells whether it was the positive or the
   * negative prompt).
   */
  encode(text: string, label: string = "入力"): number[] {
    const budget = this.#tables.maxLength - 1;
    const reject = this.#policy.overflow === "reject";
    const ids: number[] = [];
    const overflow = (): ModelInputError =>
      new ModelInputError(
        `${label}: トークン数が上限 ${this.#tables.maxLength}（末尾の </s> を含む）を超える — ` +
          "黙って切り詰めない。プロンプトを短くする",
      );
    /**
     * Called before pushing the next id (added vocabulary and pieces always emit at least one id).
     * When the budget is reached: false (drop the rest) when truncating, throws when rejecting.
     */
    const hasRoom = (): boolean => {
      if (ids.length < budget) return true;
      if (reject) throw overflow();
      return false;
    };
    // Ids are only pushed in order from the start, so dropping the chunks / pieces after the budget
    // is reached does not change the id sequence (the cost of encoding past the truncation point
    // does not grow with the input length — the reject policy stops at the same place).
    for (const chunk of this.#added.split(text)) {
      if (chunk.added) {
        if (!hasRoom()) break;
        ids.push(this.#tables.addedTokens.get(chunk.text) as number);
        continue;
      }
      // NOTE: a non-added chunk emits no id if it is whitespace only, so it is not rejected here
      // (the reject policy decides just before a piece that emits ids, and at the end).
      if (!reject && ids.length >= budget) break;
      const normalized = this.#policy.normalize?.(chunk.text) ?? chunk.text;
      for (const piece of t5PreTokenize(normalized, this.#tables.space)) {
        if (!hasRoom()) break;
        // Do not spread long pieces into arguments (`push(...)` throws RangeError at V8's argument
        // limit).
        for (const id of this.#tokenize(piece, label)) ids.push(id);
      }
    }
    // What the last piece pushed across the budget (the reject policy fails here).
    if (reject && ids.length > budget) throw overflow();
    return [...ids.slice(0, budget), this.#tables.eosId];
  }
}

/**
 * Converts the T5-family asset JSON (`vocabText` / `scores` / `unkId` / `eosId` / `addedTokens` /
 * `space` / `maxLength`) into tables (an external boundary, so the structure is checked before
 * use).
 *
 * MUST: `minScore` / `maxTokenLength` are derived from **the whole vocabulary** (the unknown-node
 * weight and the search width).
 */
export const parseT5Tables = (raw: unknown, label: string): T5Tables => {
  const obj = asRecord(raw, label);
  const tokens = asString(obj["vocabText"], `${label}.vocabText`).split("\n");
  const rawScores = obj["scores"];
  if (!Array.isArray(rawScores) || rawScores.length !== tokens.length) {
    throw new Error(`${label}: scores の長さが語彙数 ${tokens.length} と合わない`);
  }
  assertUniqueLines(tokens, `${label}.vocabText`);
  const vocab = new Map<string, T5VocabEntry>();
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asFiniteNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, Array.from(token).length);
  }
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asVocabId(obj["unkId"], `${label}.unkId`, tokens.length),
    eosId: asVocabId(obj["eosId"], `${label}.eosId`, tokens.length),
    addedTokens: parseAddedTokens(obj["addedTokens"], `${label}.addedTokens`),
    space: parseCodeRanges(obj["space"], `${label}.space`),
    maxLength: asPositiveInteger(obj["maxLength"], `${label}.maxLength`),
  };
};
