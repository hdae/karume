/**
 * BPE (merge rank) symbol-sequence construction and merging, plus the byte_fallback expansion.
 *
 * This is **family-independent common processing** (tokenizers that use BPE all go through this
 * path), so it lives in `src/text/` rather than a family directory (the same placement reason as
 * `unigram.ts`). Normalization, pre-tokenization, special tokens and the post_processor are the
 * family's job; this looks only at "one piece → id sequence".
 *
 * This module is a port of the source of truth, `BPE::merge_word` + `Word::merge_all` of `tokenizers` (Rust):
 *
 *   (1) Symbols start as code points. A single character missing from the vocabulary splits by
 *       byte_fallback into one symbol per UTF-8 byte (this port accepts only assets with all 256
 *       byteIds, so the branch that falls to `<unk>` is **unreachable by construction** —
 *       {@link createBpeModel} is the check)
 *   (2) Merging proceeds with a priority queue ordered by **lowest rank → smaller position**. A
 *       popped entry is invalidated by "is the symbol at that position alive" and "is the same pair
 *       still there"
 *
 * MUST: **write it with a merge queue from the start** (ADR 0084 decision 3). The O(n²) "scan every
 * adjacent pair → splice the lowest rank" MUST NOT be ported — Gemma's pre_tokenizer does not split
 * the normalized string at all, so **one piece is the whole input**. Qwen2 too makes one piece of a
 * letter run without separators or Japanese without punctuation, so merging ({@link mergeSymbols})
 * uses the same single implementation (ADR 0084 addendum 2026-10-10).
 */

import { setUnique } from "./asset-gates.ts";
import { toCodePoints } from "./code-points.ts";

/** One merge rule (`rank` is the upstream order, `newId` the id of the concatenated spelling). */
export type BpeMerge = {
  readonly rank: number;
  readonly newId: number;
};

/** A BPE model folded into a lookup-only shape. */
export type BpeModel = {
  /** id → spelling. */
  readonly tokenOf: ReadonlyMap<number, string>;
  /** spelling → id. */
  readonly idOf: ReadonlyMap<string, number>;
  /** {@link pairKey} → merge rule. */
  readonly merges: ReadonlyMap<number, BpeMerge>;
  /** Stride of the composite pair key (= the vocabulary's largest id + 1). */
  readonly pairStride: number;
  /** byte value → id (256 entries). */
  readonly byteIds: readonly number[];
  /** id → byte value (the direction the decoder uses to detect byte runs). */
  readonly byteOf: ReadonlyMap<number, number>;
};

/**
 * The material {@link createBpeModel} accepts (the same entry for a dense vocabulary table or a
 * subset).
 */
export type BpeSource = {
  /** `[id, spelling]`. */
  readonly vocab: readonly (readonly [number, string])[];
  /** `[left id, right id, rank]`. */
  readonly merges: readonly (readonly [number, number, number])[];
  /** ids for bytes 0..255 (**exactly 256 entries**). */
  readonly byteIds: readonly number[];
};

/** Spelling of a byte_fallback vocabulary entry (`format!("<{:#04X}>")` of `tokenizers`). */
const byteTokenSpelling = (byte: number): string =>
  `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;

/** Composite key of two adjacent symbols. */
const pairKey = (model: BpeModel, left: number, right: number): number =>
  left * model.pairStride + right;

/**
 * Builds a BPE model from the asset material (an external boundary, so value ranges and consistency
 * are checked here).
 *
 * MUST: **do not let the asset carry `newId`; look it up again here from the concatenated
 * spellings**. Upstream also builds it by looking up `vocab[a + b]`, so carrying it in the asset
 * would let "an asset whose vocabulary and merge targets disagree" through without an exception
 * (only the segmentation would change quietly).
 */
export const createBpeModel = (source: BpeSource): BpeModel => {
  const tokenOf = new Map<number, string>();
  const idOf = new Map<string, number>();
  let maxId = 0;
  for (const [id, token] of source.vocab) {
    if (!Number.isSafeInteger(id) || id < 0) {
      throw new Error(`Vocabulary id is not a non-negative safe integer (${id})`);
    }
    setUnique(tokenOf, id, token, "vocabulary id");
    setUnique(idOf, token, id, "vocabulary spelling");
    maxId = Math.max(maxId, id);
  }
  // The pair key is one number, `left * stride + right`. The stride is derived from the vocabulary,
  // so as the asset grows the key can leave the safe-integer range — at that point **fail loudly**
  // (never let it silently hit another pair).
  const pairStride = maxId + 1;
  if (pairStride * pairStride > Number.MAX_SAFE_INTEGER) {
    throw new Error(
      `The vocabulary is too large for the pair key to fit in a safe integer (max id ${maxId})`,
    );
  }

  if (source.byteIds.length !== 256) {
    throw new Error(`byteIds does not have 256 entries (${source.byteIds.length} entries)`);
  }
  const byteOf = new Map<number, number>();
  for (const [byte, id] of source.byteIds.entries()) {
    const token = tokenOf.get(id);
    // MUST: check the spelling too. The consecutive `base + byte` numbering is **a measured fact
    // about the real assets**, not a guarantee of the schema (ADR 0084 decision 1) — letting a
    // shifted table through would make the byte expansion point at other rows.
    if (token !== byteTokenSpelling(byte)) {
      throw new Error(
        `byteIds[${byte}]: the spelling of id ${id} is not ${byteTokenSpelling(byte)} (${token})`,
      );
    }
    setUnique(byteOf, id, byte, "byteIds");
  }

  const merges = new Map<number, BpeMerge>();
  const model: BpeModel = {
    tokenOf,
    idOf,
    merges,
    pairStride,
    byteIds: [...source.byteIds],
    byteOf,
  };
  for (const [left, right, rank] of source.merges) {
    const leftToken = tokenOf.get(left);
    const rightToken = tokenOf.get(right);
    if (leftToken === undefined || rightToken === undefined) {
      throw new Error(`Merge pair (${left}, ${right}) has an id outside the vocabulary`);
    }
    const newId = idOf.get(leftToken + rightToken);
    if (newId === undefined) {
      throw new Error(
        `Merge (${left}, ${right}): the concatenated spelling is not in the vocabulary`,
      );
    }
    if (!Number.isSafeInteger(rank) || rank < 0) {
      throw new Error(`Merge rank is not a non-negative safe integer (${rank})`);
    }
    // MUST: forbid later-wins. Upstream is a HashMap too, so a duplicate pair silently loses one
    // side and only the segmentation rule changes (the very reason `setUnique` exists in
    // `asset-gates.ts`).
    setUnique(merges, pairKey(model, left, right), { rank, newId }, "merge pair");
  }
  return model;
};

/**
 * A priority-queue entry (the source of truth's `Merge` — ordered by rank ascending, ties by
 * position ascending).
 */
type QueuedMerge = {
  readonly pos: number;
  readonly rank: number;
  /**
   * The rule at the time of pushing. Compared **by identity** at pop time to discard stale entries.
   */
  readonly merge: BpeMerge;
};

/** Binary heap (min-pop). Single-use, so it is used only inside {@link mergeSymbols}. */
const isBefore = (a: QueuedMerge, b: QueuedMerge): boolean =>
  a.rank !== b.rank ? a.rank < b.rank : a.pos < b.pos;

const heapPush = (heap: QueuedMerge[], item: QueuedMerge): void => {
  heap.push(item);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = (index - 1) >> 1;
    if (!isBefore(heap[index], heap[parent])) break;
    [heap[parent], heap[index]] = [heap[index], heap[parent]];
    index = parent;
  }
};

const heapPop = (heap: QueuedMerge[]): QueuedMerge | undefined => {
  const top = heap[0];
  if (top === undefined) return undefined;
  const last = heap.pop() as QueuedMerge;
  if (heap.length === 0) return top;
  heap[0] = last;
  let index = 0;
  for (;;) {
    const left = index * 2 + 1;
    const right = left + 1;
    let best = index;
    if (left < heap.length && isBefore(heap[left], heap[best])) best = left;
    if (right < heap.length && isBefore(heap[right], heap[best])) best = right;
    if (best === index) break;
    [heap[best], heap[index]] = [heap[index], heap[best]];
    index = best;
  }
  return top;
};

/**
 * One piece (pre-token) → id sequence.
 *
 * The piece is received as an already-normalized string (normalization and pre-tokenization are the
 * family's job). Unpaired surrogates are rejected by {@link toCodePoints} — they are values the
 * source of truth (Rust's `str`) cannot hold, and silently collapsing them to U+FFFD would quietly
 * change the id sequence.
 */
export const bpeEncode = (model: BpeModel, piece: string): number[] =>
  mergeSymbols(initialSymbols(model, piece), (left, right) =>
    model.merges.get(pairKey(model, left, right)),
  );

/**
 * Merges a sequence of symbol ids to the end (the source of truth's `Word::merge_all` — a priority
 * queue by lowest rank → smaller position, and an adjacency list). How the symbol sequence is built
 * (byte_fallback / byte-level) stays with the family; only merging is shared.
 *
 * MUST: `mergeOf` returns **the same object** for the same pair (popped entries are compared by
 * identity — the MUST below). The passed array is rewritten (the caller passes a freshly built
 * symbol sequence).
 */
export const mergeSymbols = (
  symbols: number[],
  mergeOf: (left: number, right: number) => BpeMerge | undefined,
): number[] => {
  const count = symbols.length;
  if (count < 2) return symbols;

  const prev = new Int32Array(count);
  const next = new Int32Array(count);
  const alive = new Uint8Array(count).fill(1);
  for (let index = 0; index < count; index++) {
    prev[index] = index - 1;
    next[index] = index + 1 < count ? index + 1 : -1;
  }

  const heap: QueuedMerge[] = [];
  const enqueue = (pos: number, right: number): void => {
    const merge = mergeOf(symbols[pos], symbols[right]);
    if (merge !== undefined) heapPush(heap, { pos, rank: merge.rank, merge });
  };
  for (let index = 0; index + 1 < count; index++) enqueue(index, index + 1);

  for (;;) {
    const top = heapPop(heap);
    if (top === undefined) break;
    if (alive[top.pos] === 0) continue;
    const right = next[top.pos];
    if (right === -1) continue;
    // MUST: compare **by identity**. The symbol at a position can be replaced by another merge, so
    // comparing only the rank would let "another pair that happens to have the same rank" through
    // (the source of truth checks `new_id`).
    if (mergeOf(symbols[top.pos], symbols[right]) !== top.merge) continue;

    symbols[top.pos] = top.merge.newId;
    alive[right] = 0;
    next[top.pos] = next[right];
    if (next[right] !== -1) prev[next[right]] = top.pos;
    if (prev[top.pos] !== -1) enqueue(prev[top.pos], top.pos);
    if (next[top.pos] !== -1) enqueue(top.pos, next[top.pos]);
  }
  return symbols.filter((_, index) => alive[index] === 1);
};

/**
 * The initial state of the symbol sequence (one code point = one symbol; byte_fallback if not in
 * the vocabulary).
 *
 * NOTE: the branch that falls to `<unk>` is not ported. {@link createBpeModel} checks that all 256
 * byteIds exist, so the source of truth's "unk if even one byte is missing from the vocabulary" is
 * unreachable.
 */
const initialSymbols = (model: BpeModel, piece: string): number[] => {
  const symbols: number[] = [];
  const encoder = new TextEncoder();
  for (const cp of toCodePoints(piece)) {
    const char = String.fromCodePoint(cp);
    const id = model.idOf.get(char);
    if (id !== undefined) {
      symbols.push(id);
      continue;
    }
    for (const byte of encoder.encode(char)) symbols.push(model.byteIds[byte]);
  }
  return symbols;
};
