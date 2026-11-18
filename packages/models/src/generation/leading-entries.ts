/**
 * The leading entries of a logits row's ranking from **per-part selections** — the host half of a
 * selection whose GPU half is one small `topk` over the row cut into parts (**pipeline-independent
 * shared code**, next to `sampler.ts`, whose candidate draw reads what this returns).
 *
 * The ranking is the GPU `topk`'s: value descending, the smaller id first on a tie, +0 and −0 equal,
 * NaN above every number. A row of `parts × width` logits is ranked part by part on the GPU, each
 * part returning its first `perPart` entries; the first `count` entries of the whole row are then a
 * merge of the parts' lists — as long as no part can hold more of them than it returned.
 *
 * The parts are taken **by residue**: part `p` holds the ids `≡ p (mod parts)`, its entry `j` the
 * id `j · parts + p`. Tokens that sit next to each other in a vocabulary (digits, single letters)
 * tend to lead a row together, and parts of consecutive ids would put them in one part, which then
 * returns too few; by residue, neighbours land in different parts.
 *
 * Why per part at all: the `topk` kernel keeps a sorted list per lane in workgroup storage, so its
 * time grows steeply with the number of entries selected and its storage with it as well — one
 * `topk` of the 65 entries the distributions' own sampler needs (top-k 64) over a row of 262,144
 * takes several times a decode token, and more workgroup storage than WebGPU's default limit
 * admits. A `topk` of a few entries per part runs on every device, one workgroup per part, in a
 * fraction of the time the whole row takes to read back.
 *
 * SPEC: docs/spec/generation.md#the-merge-and-the-draw-on-the-host
 */

/** The per-part selections of one row, as the GPU returned them. */
export type PartSelections = {
  /** `parts × perPart` raw logits: part `p`'s entries at `p · perPart …`, in the ranking's order. */
  readonly values: Float32Array<ArrayBuffer>;
  /**
   * The entries' places **within their part** (`0 … width − 1`), in the same layout: place `j` of
   * part `p` is the id `j · parts + p`.
   */
  readonly indices: Int32Array<ArrayBuffer>;
  readonly parts: number;
  readonly perPart: number;
  /** The ids per part (`parts × width` is the row's length). */
  readonly width: number;
};

/** The leading entries of a row: `values[i]` is the raw logit of token `indices[i]`. */
export type LeadingEntries = {
  readonly values: Float32Array<ArrayBuffer>;
  readonly indices: Int32Array<ArrayBuffer>;
};

/** "(a, id a) ranks above (b, id b)": NaN above every number, then the value, then the smaller id. */
const ranksAbove = (a: number, aId: number, b: number, bId: number): boolean => {
  if (a !== a) return b === b || aId < bId;
  return b === b && (a > b || (a === b && aId < bId));
};

/**
 * The first `count` entries of the whole row's ranking, merged from the parts' selections — or
 * `undefined` when the selections do not hold them all.
 *
 * The merge takes, `count` times, the best of the parts' current heads (a heap of the parts, keyed
 * by their heads). It is the row's ranking as long as every entry a part did not return ranks below
 * everything taken — which holds for a part whose last returned entry was not taken: its other
 * entries rank below that one. A part **whose whole list was taken before the merge's last entry**
 * may hold an entry that ranks above what the merge took after it (unless it has no more entries
 * than it returned), so the result is not known → `undefined`. A part whose last returned entry is
 * the merge's last entry hides nothing: what it did not return ranks below that entry.
 *
 * Selections that are not what the GPU half promises — a layout of the wrong size, a place outside
 * the part, a list that is not in the ranking's order where the merge walks it — give `undefined`
 * too: the caller then reads the whole row, and the count of such steps is what shows the selection
 * is broken.
 *
 * MUST: it never throws (it runs where a decode run has already gone through — the contract of
 * `SamplerCandidates.next`, which it feeds).
 */
export const leadingEntries = (
  selections: PartSelections,
  count: number,
): LeadingEntries | undefined => {
  const { values, indices, parts, perPart, width } = selections;
  if (
    !Number.isSafeInteger(count) ||
    count < 1 ||
    count > parts * perPart ||
    values.length !== parts * perPart ||
    indices.length !== parts * perPart
  ) {
    return undefined;
  }
  /** How many of each part's entries were taken (its head is the next one). */
  const taken = new Int32Array(parts);
  /** Whether part `a`'s head ranks above part `b`'s (both have one). */
  const headAbove = (a: number, b: number): boolean => {
    const aSlot = a * perPart + taken[a];
    const bSlot = b * perPart + taken[b];
    return ranksAbove(
      values[aSlot],
      indices[aSlot] * parts + a,
      values[bSlot],
      indices[bSlot] * parts + b,
    );
  };
  // A binary heap of the parts that still have a head, the best head at the root.
  const heap = new Int32Array(parts);
  let size = parts;
  const siftDown = (from: number): void => {
    let at = from;
    for (;;) {
      const left = at * 2 + 1;
      if (left >= size) return;
      const right = left + 1;
      const child = right < size && headAbove(heap[right], heap[left]) ? right : left;
      if (!headAbove(heap[child], heap[at])) return;
      const swap = heap[at];
      heap[at] = heap[child];
      heap[child] = swap;
      at = child;
    }
  };
  for (let part = 0; part < parts; part += 1) heap[part] = part;
  for (let at = (parts >> 1) - 1; at >= 0; at -= 1) siftDown(at);

  const outValues = new Float32Array(count);
  const outIndices = new Int32Array(count);
  for (let at = 0; at < count; at += 1) {
    // `count ≤ parts × perPart`, so a part with a head is left.
    const best = heap[0];
    const slot = best * perPart + taken[best];
    const place = indices[slot];
    if (place < 0 || place >= width) return undefined;
    if (taken[best] > 0) {
      // The part's own order where the merge walks it: the entry before ranks above this one
      // (places in a part ascend with their ids).
      const before = indices[slot - 1];
      if (before === place || !ranksAbove(values[slot - 1], before, values[slot], place)) {
        return undefined;
      }
    }
    outValues[at] = values[slot];
    outIndices[at] = place * parts + best;
    taken[best] += 1;
    if (taken[best] >= perPart) {
      // The part's whole list is taken. What it did not return ranks below this entry and may rank
      // above the next one, so only the merge's last entry may be a part's last.
      if (width > perPart && at < count - 1) return undefined;
      size -= 1;
      heap[0] = heap[size];
    }
    siftDown(0);
  }
  return { values: outValues, indices: outIndices };
};
