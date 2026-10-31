/**
 * Qwen2 tokenizer (byte-level BPE) — the engine transformers' `Qwen2Tokenizer` runs, shared by the
 * families whose text encoder is a Qwen model: Anima (Qwen3, `src/anima/text/tokenizer.ts`) and
 * Qwen3.5 (`src/qwen3_5/tokenizer.ts`). It lives in `src/text/` because a family directory may not
 * import another family's (`packages/models/README.md`); each family keeps its own asset reader,
 * accepted set and template, and only the tables' shape and the encoding are shared here.
 *
 * The canonical path (transformers 5.14.1 `Qwen2Tokenizer`; both families' `tokenizer.json`) is
 *   AddedVocabulary (cuts out the added tokens leftmost-longest — Anima 26, Qwen3.5 33)
 *   → NFC → Split (regular expression, Isolated) → ByteLevel (add_prefix_space=false)
 *   → BPE (ignore_merges=false / no unk) → a post_processor that **adds no special tokens**
 *
 * MUST: this engine applies no chat template. Anima's canonical code calls the plain
 * `tokenizer([text], …)` — going through `apply_chat_template` would prepend `<|im_start|>` and the
 * like, and the id sequence would become an entirely different one. A family whose upstream does
 * render a template (Qwen3.5 for MOSS-Image — `src/qwen3_5/chat.ts`) renders it **before** calling
 * {@link Qwen2Tokenizer.encode}; the special-token spellings in the rendered string are then cut out
 * by the added vocabulary exactly as upstream does.
 *
 * **Do not use the regular-expression classes `\p{L}` / `\p{N}` / `\s`** — the canonical source for
 * classification is the Rust-side Unicode table; the moment it diverges from the JS engine's ICU,
 * the pre-token boundaries change and the id sequence silently becomes a different one.
 * Classification looks up the closed-interval tables the exporter folded by querying the canonical
 * implementation one character at a time (`text/code-ranges.ts`).
 *
 * MUST: **the same rule applies to NFC**. `String.prototype.normalize("NFC")` is not canonical —
 * the canonical implementation (Rust's `unicode-normalization`) has an older Unicode table, and its
 * output differs from ICU / Python on 123 code points (measured: 120 cp are treated as combining
 * class 0 and not reordered, 3 cp do not get newer compositions). Delegating to the plain
 * `normalize("NFC")` gives a different id sequence for exactly the prompts containing those
 * characters, with no exception or warning. Split on the baked segment table first, then apply the
 * plain NFC ({@link normalizeNfc}).
 */

import { type AddedTokenMatcher, createAddedTokenMatcher } from "./added-tokens.ts";
import {
  asNumber,
  asPositiveInteger,
  asRecord,
  assertCodePoint,
  assertUniqueLines,
  asString,
  parseAddedTokens,
  setUnique,
} from "./asset-gates.ts";
import { type BpeMerge, mergeSymbols } from "./bpe.ts";
import { toCodePoints } from "./code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "./code-ranges.ts";

/** Character classes used by the pre_tokenizer regular expression (closed-interval tables). */
export type Qwen2CharClasses = {
  /** `\p{L}` */
  readonly letter: CodeRanges;
  /** `\p{N}` */
  readonly number: CodeRanges;
  /** `\s` */
  readonly space: CodeRanges;
};

/**
 * Code point that `(?i:'s|'t|…)` treats as equivalent → the character on the suffix side (both
 * ASCII lowercase).
 *
 * MUST: do not substitute `toLowerCase()`. **The canonical `(?i:)` is Unicode simple case
 * folding**; measured, U+017F (ſ) is equivalent to `s` (`it'ſs` splits into `'ſ` + `s`). ASCII case
 * flipping and `toLowerCase()` (ſ → ſ) both miss it, and the pre-token boundaries change silently.
 * Trivial pairs (the cp itself) are not included.
 */
export type Qwen2CaseFold = ReadonlyMap<number, number>;

export type Qwen2Assets = {
  /** Token string → id. */
  readonly vocab: ReadonlyMap<string, number>;
  /**
   * `"left right"` → rank (the byte-level vocabulary contains no space, so it can be the
   * separator).
   */
  readonly merges: ReadonlyMap<string, number>;
  /** Added vocabulary (special tokens). Cut out **before** normalisation. */
  readonly addedTokens: ReadonlyMap<string, number>;
  readonly classes: Qwen2CharClasses;
  readonly caseFold: Qwen2CaseFold;
  /**
   * Code points where the canonical NFC and `String.prototype.normalize` differ
   * ({@link normalizeNfc}).
   */
  readonly nfcSegments: CodeRanges;
  /**
   * The `max_length` of the consumer's canonical call (Anima 512, MOSS-Image's Qwen3.5 4096) — the
   * default truncation length of {@link Qwen2Tokenizer.encode}.
   */
  readonly maxLength: number;
};

/** Suffixes of `(?i:'s|'t|'re|'ve|'m|'ll|'d)` (in the same order as the regular expression). */
const APOSTROPHE_SUFFIXES = ["s", "t", "re", "ve", "m", "ll", "d"] as const;

/**
 * The byte → visible character table from GPT-2. A pure algorithm independent of the Unicode
 * version, so it may be built (the only place where no table needs baking).
 *
 * MUST: do not hold it in a module-scope `const` (cross-cutting invariant "zero side effects in
 * every module"). It is built once per instance in the {@link Qwen2Tokenizer} constructor.
 */
const buildByteEncoder = (): readonly string[] => {
  // Ranges that can stay visible as is. The rest are assigned in order from U+0100.
  const isDirect = (byte: number): boolean =>
    (byte >= 0x21 && byte <= 0x7e) ||
    (byte >= 0xa1 && byte <= 0xac) ||
    (byte >= 0xae && byte <= 0xff);
  let extra = 0;
  return Array.from({ length: 256 }, (_, byte) => {
    if (isDirect(byte)) return String.fromCodePoint(byte);
    const mapped = String.fromCodePoint(0x100 + extra);
    extra += 1;
    return mapped;
  });
};

/**
 * The same NFC as the canonical implementation (Rust's `unicode-normalization`).
 *
 * Splits the string at the segment cps, applies `String.prototype.normalize("NFC")` to each segment
 * only, and concatenates.
 * MUST: a segment cp itself goes **into neither segment** — the canonical implementation treats
 * that cp as a starter and involves it in neither reordering nor composition around it, so passing
 * it through is exactly that behaviour. The segment table was baked by probing the canonical
 * implementation; the equivalence is checked at every emit over all code points × 11 contexts +
 * random sampling.
 */
export const normalizeNfc = (text: string, segments: CodeRanges): string => {
  const out: string[] = [];
  let buffer = "";
  for (const ch of text) {
    if (inCodeRanges(segments, ch.codePointAt(0) as number)) {
      out.push(buffer.normalize("NFC"), ch);
      buffer = "";
    } else {
      buffer += ch;
    }
  }
  out.push(buffer.normalize("NFC"));
  return out.join("");
};

/**
 * Hand-written scan of the Split regular expression. Alternatives are tried top to bottom, and the
 * first match wins:
 *   ① `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
 *   ② `[^\r\n\p{L}\p{N}]?\p{L}+`
 *   ③ `\p{N}`
 *   ④ ` ?[^\s\p{L}\p{N}]+[\r\n]*`
 *   ⑤ `\s*[\r\n]+`
 *   ⑥ `\s+(?!\S)`
 *   ⑦ `\s+`
 * No alternative matches empty, so picking one per position and advancing is equivalent to the
 * canonical find_iter. The equivalence itself is pinned by measurement in
 * `verify_qwen_pre_tokenize` (all code points × 11 contexts).
 */
export const qwen2PreTokenize = (
  cps: readonly number[],
  classes: Qwen2CharClasses,
  caseFold: Qwen2CaseFold,
): string[] => {
  const n = cps.length;
  const isLetter = (k: number): boolean => inCodeRanges(classes.letter, cps[k]);
  const isNumber = (k: number): boolean => inCodeRanges(classes.number, cps[k]);
  const isSpace = (k: number): boolean => inCodeRanges(classes.space, cps[k]);
  const isOther = (k: number): boolean => !isSpace(k) && !isLetter(k) && !isNumber(k);
  const isNewline = (k: number): boolean => cps[k] === 0x0d || cps[k] === 0x0a;

  const out: string[] = [];
  let i = 0;
  while (i < n) {
    let end = -1;
    // ① Contractions. Case-insensitivity is looked up in the baked table (`toLowerCase()` is not
    // canonical).
    if (cps[i] === 0x27) {
      for (const suffix of APOSTROPHE_SUFFIXES) {
        if (matchesFolded(cps, i + 1, suffix, caseFold)) {
          end = i + 1 + suffix.length;
          break;
        }
      }
    }
    // ② An optional single character (other than CR/LF, letters, numbers) + a run of letters. The
    // optional part is consumed first, and backed off if that fails.
    if (end < 0) {
      for (const skip of [1, 0]) {
        if (skip === 1 && (isNewline(i) || isLetter(i) || isNumber(i))) continue;
        let k = i + skip;
        const start = k;
        while (k < n && isLetter(k)) k++;
        if (k > start) {
          end = k;
          break;
        }
      }
    }
    // ③ One digit (no `+`, so it always splits one character at a time)
    if (end < 0 && isNumber(i)) end = i + 1;
    // ④ An optional single ASCII space + a run of others + a run of CR/LF
    if (end < 0) {
      for (const skip of [1, 0]) {
        if (skip === 1 && cps[i] !== 0x20) continue;
        let k = i + skip;
        const start = k;
        while (k < n && isOther(k)) k++;
        if (k > start) {
          while (k < n && isNewline(k)) k++;
          end = k;
          break;
        }
      }
    }
    // ⑤ The longest prefix of a whitespace run that ends with CR/LF
    if (end < 0 && isSpace(i)) {
      let k = i;
      while (k < n && isSpace(k)) k++;
      let last = -1;
      for (let m = i; m < k; m++) if (isNewline(m)) last = m;
      if (last >= 0) end = last + 1;
    }
    // ⑥ A whitespace run (leaving the last character if a non-space follows) / ⑦ a whitespace run
    if (end < 0 && isSpace(i)) {
      let k = i;
      while (k < n && isSpace(k)) k++;
      end = k === n ? k : Math.max(i + 1, k - 1);
    }
    if (end <= i) {
      throw new Error(
        `The pre-token scan does not advance (position ${i}, U+${cps[i].toString(16)})`,
      );
    }
    // No argument spreading even for a long pre-token (`fromCodePoint(...)` hits the stack limit).
    let piece = "";
    for (let k = i; k < end; k++) piece += String.fromCodePoint(cps[k]);
    out.push(piece);
    i = end;
  }
  return out;
};

/**
 * Whether `cps[at:]` matches the suffix case-insensitively (equivalence is decided by the baked
 * table only).
 */
const matchesFolded = (
  cps: readonly number[],
  at: number,
  suffix: string,
  caseFold: Qwen2CaseFold,
): boolean => {
  if (at + suffix.length > cps.length) return false;
  for (let k = 0; k < suffix.length; k++) {
    const cp = cps[at + k];
    const want = suffix.codePointAt(k);
    if (cp !== want && caseFold.get(cp) !== want) return false;
  }
  return true;
};

export class Qwen2Tokenizer {
  readonly #assets: Qwen2Assets;
  readonly #added: AddedTokenMatcher;
  readonly #byteEncoder: readonly string[];
  readonly #encoder = new TextEncoder();

  constructor(assets: Qwen2Assets) {
    this.#assets = assets;
    this.#added = createAddedTokenMatcher([...assets.addedTokens.keys()]);
    this.#byteEncoder = buildByteEncoder();
  }

  /**
   * Splits one pre-token (already byte-level encoded) with BPE (the merge is the single
   * implementation shared with Gemma).
   *
   * MUST: do not go back to the quadratic "scan every adjacent pair → splice the lowest rank" —
   * truncation acts only per piece, and a run of letters without separators or Japanese without
   * punctuation makes one piece the whole input (ADR 0084 addendum 2026-10-10).
   *
   * Symbols are held as **numbers local to this piece** (in order of first spelling), and merge
   * rules are looked up by the pair of spellings. They are not vocabulary ids because intermediate
   * spellings of the merge need not be in the vocabulary (only the spellings left at the end must
   * be — the same accepted set as the full adjacent scan). Pair results are kept within the piece:
   * the shared merge expects the same object for the same pair (it matches popped entries by
   * identity), and keeping the cache to the piece's lifetime means it does not grow across inputs.
   */
  #bpe(piece: string): number[] {
    const spellings: string[] = [];
    const numberOf = new Map<string, number>();
    const intern = (spelling: string): number => {
      let symbol = numberOf.get(spelling);
      if (symbol === undefined) {
        symbol = spellings.length;
        spellings.push(spelling);
        numberOf.set(spelling, symbol);
      }
      return symbol;
    };
    const resolved = new Map<number, Map<number, BpeMerge | null>>();
    const mergeOf = (left: number, right: number): BpeMerge | undefined => {
      let row = resolved.get(left);
      if (row === undefined) {
        row = new Map();
        resolved.set(left, row);
      }
      let merge = row.get(right);
      if (merge === undefined) {
        const rank = this.#assets.merges.get(`${spellings[left]} ${spellings[right]}`);
        merge =
          rank === undefined ? null : { rank, newId: intern(spellings[left] + spellings[right]) };
        row.set(right, merge);
      }
      return merge ?? undefined;
    };
    const merged = mergeSymbols(Array.from(piece, intern), mergeOf);
    return merged.map((symbol) => {
      const id = this.#assets.vocab.get(spellings[symbol]);
      // byte-level, so a token missing from the vocabulary cannot occur in principle. If it does,
      // the table is broken.
      if (id === undefined) {
        throw new Error(
          `Token ${JSON.stringify(spellings[symbol])} is not in the Qwen2 vocabulary`,
        );
      }
      return id;
    });
  }

  /**
   * The same id sequence as
   * `tokenizer([text], padding="longest", max_length=limit, truncation=True)`. The post_processor
   * adds no special tokens, so truncation simply keeps the head.
   *
   * `limit` defaults to the asset's `maxLength` (Anima's call). A caller that must tell "exactly
   * `maxLength`" from "longer" passes `maxLength + 1` (the Qwen3.5 tokenizer's overflow check) —
   * encoding still stops at `limit`, so the cost does not grow with the input past it.
   */
  encode(text: string, limit: number = this.#assets.maxLength): number[] {
    const maxLength = limit;
    const ids: number[] = [];
    // Ids are only appended from the start, so dropping chunks / pre-tokens after the limit is
    // reached does not change the id sequence (BPE is not run past the truncation point).
    for (const chunk of this.#added.split(text)) {
      if (ids.length >= maxLength) break;
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      const cps = toCodePoints(normalizeNfc(chunk.text, this.#assets.nfcSegments));
      for (const piece of qwen2PreTokenize(cps, this.#assets.classes, this.#assets.caseFold)) {
        if (ids.length >= maxLength) break;
        // ByteLevel (add_prefix_space=false) — **no leading space is added**.
        let encoded = "";
        for (const byte of this.#encoder.encode(piece)) encoded += this.#byteEncoder[byte];
        for (const id of this.#bpe(encoded)) ids.push(id);
      }
    }
    return ids.slice(0, maxLength);
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
  if (!Array.isArray(raw)) throw new Error(`${label}: not an array`);
  const out = new Map<number, number>();
  for (const entry of raw) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "number"
    ) {
      throw new Error(`${label}: not [cp, cp]`);
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
      `${label}: line count ${lines.length} does not match ${countLabel} ${want} (the asset is broken)`,
    );
  }
  return lines;
};

/**
 * Converts the tables of a Qwen2 asset JSON (`vocabText` / `vocabCount` / `mergesText` /
 * `mergesCount` / `addedTokens` / `classes` / `caseFold` / `nfcSegments` / `maxLength`) into
 * {@link Qwen2Assets}. It is an external boundary, so the structure is checked before use. Each
 * family's reader checks its own envelope (format, provenance) and hands the object here.
 */
export const parseQwen2Tables = (raw: unknown, label: string): Qwen2Assets => {
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
