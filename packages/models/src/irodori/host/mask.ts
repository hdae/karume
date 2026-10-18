/**
 * Building the concatenated mask `[1,1,1,S+Σcaps]` of `dit` (ADR 0047 decisions 1 / 4).
 *
 * The order is **self (latent, all 1) → text → speaker → caption**, exactly the concatenation
 * order of K/V inside the `dit` graph. Each condition segment is "True only for the prefix of the
 * used length", and the rest, the right padding, is False.
 *
 * MUST: do not break the segment order or the offsets. With the same total length, both the shapes
 * and the run go through, and a result that **dropped a different condition** comes out silently
 * (the exporter's `TestSegmentMasks` pins the same values).
 *
 * An uncond variant is "all False for that segment only" — passing cond's state as is and dropping
 * only the mask is bit-identical to upstream's uncond (state 0 + mask 0); that is the measurement
 * behind ADR 0047 decision 1.
 */

/** Name of a condition segment (= the CFG variant name / the K/V concatenation order). */
export type IrodoriSegment = "text" | "speaker" | "caption";

/**
 * Concatenation order. **This order is the source of truth** (same spelling and order as the
 * exporter's `DIT_UNCOND_VARIANTS`).
 */
export const SEGMENT_ORDER: readonly IrodoriSegment[] = ["text", "speaker", "caption"];

/** Length per segment (`used` = rows actually filled / `caps` = the graph's declared length). */
export type SegmentLengths = Readonly<Record<IrodoriSegment, number>>;

/**
 * Builds the concatenated mask. Passing `uncond` makes only that segment all False.
 *
 * The dtype is bool (the IR's boundary convention = u32 0/1 — ADR 0009).
 */
export const buildDitMask = (
  latentLength: number,
  used: SegmentLengths,
  caps: SegmentLengths,
  uncond?: IrodoriSegment,
): Uint32Array<ArrayBuffer> => {
  if (!Number.isInteger(latentLength) || latentLength < 1) {
    throw new RangeError(`latent 長 ${latentLength} が 1 以上の整数でない`);
  }
  let total = latentLength;
  for (const segment of SEGMENT_ORDER) {
    if (used[segment] > caps[segment]) {
      throw new Error(
        `${segment} 区間の使用長 ${used[segment]} が宣言長 ${caps[segment]} を超えている`,
      );
    }
    total += caps[segment];
  }
  const mask = new Uint32Array(total);
  mask.fill(1, 0, latentLength);
  let offset = latentLength;
  for (const segment of SEGMENT_ORDER) {
    const length = uncond === segment ? 0 : used[segment];
    mask.fill(1, offset, offset + length);
    offset += caps[segment];
  }
  return mask;
};
