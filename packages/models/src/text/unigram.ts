/**
 * Unigram lattice search (Viterbi) and id conversion (fuse_unk / byte_fallback).
 *
 * This is **family-independent common processing** (tokenizers that use Unigram all go through this
 * path), so it lives in `src/text/` rather than a family directory (the same placement as
 * `src/audio/wav.ts` / `src/image/png.ts`). Normalization, pre-tokenization, special tokens and the
 * final post-processing are the family's job; this looks only at "one piece → id sequence".
 *
 * This module is a port of the source of truth, `Lattice::viterbi` of `tokenizers` (Rust), at code-point
 * positions. Tie handling follows the source of truth too: nodes ending at the same position are
 * pushed in "ascending start position" order, and on a tie the one pushed first (= the longer
 * piece) wins. If this drifts, a different segmentation occurs (rarely), and the id sequence
 * changes quietly.
 *
 * The input is a code-point sequence rather than a string to separate roles from the string → cp
 * conversion that rejects unpaired surrogates (`toCodePoints` in `src/text/code-points.ts`).
 * Callers always pass through that conversion first.
 */

/**
 * The unknown-node score is "the whole vocabulary's minimum score − 10" (the source of truth's
 * `K_UNK_PENALTY`).
 */
const UNK_PENALTY = 10;

export type UnigramVocabEntry = { readonly id: number; readonly score: number };

export type UnigramModel = {
  readonly vocab: ReadonlyMap<string, UnigramVocabEntry>;
  /**
   * Minimum score of the **whole** vocabulary. Pass the whole-vocabulary value even with a subset
   * (the unknown-node weight).
   */
  readonly minScore: number;
  /** Code-point count of the **whole** vocabulary's longest token (the prefix-search width). */
  readonly maxTokenLength: number;
  readonly unkId: number;
  /**
   * The id of byte_fallback's `<0x00>`. When set, unknown pieces expand into UTF-8 byte sequences
   * and each byte `b` becomes `byteBaseId + b` (when unset, an unknown is one
   * {@link UnigramModel.unkId}). Assumes the vocabulary holds `<0x00>`..`<0xFF>` consecutively (the
   * shape of `tokenizers`' byte_fallback vocabulary).
   */
  readonly byteBaseId?: number;
};

/** A node of the Viterbi path (start position and the slot of the node starting there). */
type Slot = { readonly pos: number; readonly slot: number };

/**
 * One part of a piece's segmentation. A part without `entry` is an unknown piece (consecutive
 * unknown nodes fused).
 */
export type UnigramSegment = { readonly text: string; readonly entry?: UnigramVocabEntry };

/**
 * The Viterbi segmentation of one piece (consecutive unknown nodes already fused into one part).
 *
 * The shape before mapping to ids is public because families differ on unknowns: most map them to
 * unk / byte_fallback ({@link unigramTokenize}), but a family that does not accept
 * out-of-vocabulary text (Wan's umT5 — ADR 0119 decision 1) rejects if there is even one unknown
 * part. Judging unknowns from ids could not tell them apart from an added-vocabulary token in the
 * text that emits the same id (umT5's unk_id 2 is `<s>`).
 *
 * NOTE: it fuses even when `fuse_unk` in `tokenizer.json` is `null` (the Unigram of `tokenizers`
 * fuses by the Rust-side default). Without fusing, input with a run of characters missing from the
 * vocabulary, such as Japanese, would line up one unknown per character and the match against the
 * source of truth would fail.
 */
export const unigramSegments = (model: UnigramModel, cps: readonly number[]): UnigramSegment[] => {
  const out: UnigramSegment[] = [];
  // Buffer accumulating consecutive unknown nodes. Non-empty = unknowns continued up to the
  // previous node.
  let pending = "";
  for (const span of viterbi(model, cps)) {
    let text = "";
    for (let k = span.start; k < span.start + span.length; k++) {
      text += String.fromCodePoint(cps[k]);
    }
    const entry = model.vocab.get(text);
    if (entry === undefined) {
      pending += text;
      continue;
    }
    if (pending !== "") {
      out.push({ text: pending });
      pending = "";
    }
    out.push({ text, entry });
  }
  if (pending !== "") out.push({ text: pending });
  return out;
};

/**
 * One piece → id sequence (an unknown part maps to one unk or to byte_fallback's UTF-8 byte
 * sequence).
 */
export const unigramTokenize = (model: UnigramModel, cps: readonly number[]): number[] => {
  const ids: number[] = [];
  for (const segment of unigramSegments(model, cps)) {
    if (segment.entry === undefined) {
      // One unknown part can grow long through byte expansion — spreading into arguments
      // (`push(...)`) throws RangeError at V8's argument limit (about 130,000), so push one element
      // at a time.
      for (const id of expandUnknown(model, segment.text)) ids.push(id);
    } else ids.push(segment.entry.id);
  }
  return ids;
};

/**
 * Fused unknown piece → id sequence.
 *
 * NOTE: the boundary "fuse first, then split the **whole** piece into bytes" is settled by the
 * parity fixture taken from the real HF tokenizer (`emoji-byte-fallback-pair` in
 * `fixtures/irodori-text/parity.json` — two out-of-vocabulary emoji in a row pin the interaction of
 * fuse × byte_fallback). The expansion point is closed in one function so that **only this** needs
 * replacing if HF's behaviour changes in the future (the fusion loop in {@link unigramSegments} has
 * the same shape with or without byte_fallback).
 */
const expandUnknown = (model: UnigramModel, text: string): number[] => {
  const base = model.byteBaseId;
  if (base === undefined) return [model.unkId];
  const bytes = new TextEncoder().encode(text);
  return Array.from(bytes, (byte) => base + byte);
};

/**
 * The best segmentation of one piece (a port of the source of truth's `Lattice::viterbi` at
 * code-point positions).
 */
const viterbi = (
  model: UnigramModel,
  cps: readonly number[],
): { start: number; length: number }[] => {
  const n = cps.length;
  const unkScore = model.minScore - UNK_PENALTY;
  // begin[pos] = lengths of the nodes starting there (insertion order = ascending length)
  const begin: number[][] = [];
  const scores: number[][] = [];
  for (let pos = 0; pos < n; pos++) {
    const lengths: number[] = [];
    const nodeScores: number[] = [];
    let text = "";
    const limit = Math.min(model.maxTokenLength, n - pos);
    for (let length = 1; length <= limit; length++) {
      text += String.fromCodePoint(cps[pos + length - 1]);
      const entry = model.vocab.get(text);
      if (entry !== undefined) {
        lengths.push(length);
        nodeScores.push(entry.score);
      }
    }
    if (lengths[0] !== 1) {
      lengths.unshift(1);
      nodeScores.unshift(unkScore);
    }
    begin.push(lengths);
    scores.push(nodeScores);
  }
  // end[e] = the nodes ending there. Pushed in ascending begin order (= for the same e, the longer
  // one first).
  const end: Slot[][] = Array.from({ length: n + 1 }, () => []);
  for (let pos = 0; pos < n; pos++) {
    for (const [slot, length] of begin[pos].entries()) end[pos + length].push({ pos, slot });
  }
  const best: number[][] = begin.map((lengths) => Array.from(lengths, () => 0));
  const prev: (Slot | undefined)[][] = begin.map((lengths) =>
    Array.from(lengths, (): Slot | undefined => undefined),
  );

  /** The best node ending at pos. pos=0 is bos (score 0). undefined if unreachable. */
  const bestLeft = (pos: number): { score: number; node: Slot | undefined } | undefined => {
    if (pos === 0) return { score: 0, node: undefined };
    let found: { score: number; node: Slot } | undefined;
    for (const node of end[pos]) {
      const candidate = best[node.pos][node.slot];
      // MUST: strict `>` — on a tie the one pushed first (the longer piece) stays.
      if (found === undefined || candidate > found.score) found = { score: candidate, node };
    }
    return found;
  };

  for (let pos = 0; pos < n; pos++) {
    const left = bestLeft(pos);
    if (left === undefined) return [];
    for (let slot = 0; slot < begin[pos].length; slot++) {
      best[pos][slot] = left.score + scores[pos][slot];
      prev[pos][slot] = left.node;
    }
  }
  const tail = bestLeft(n);
  if (tail === undefined) return [];
  const path: { start: number; length: number }[] = [];
  let node = tail.node;
  while (node !== undefined) {
    path.push({ start: node.pos, length: begin[node.pos][node.slot] });
    node = prev[node.pos][node.slot];
  }
  return path.reverse();
};
