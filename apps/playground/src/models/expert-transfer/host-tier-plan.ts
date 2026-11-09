/**
 * The pure parts of the host tier measurement (the RAM tier of the expert cache, ADR 0146): the
 * allocation plan for a target size, where whole experts sit in the tier's chunks, the random
 * distinct experts a hit sample brings into the pool, the fill, and the per-expert cost.
 */

import { EXPERT_BYTES } from "./stats";

export const GIB = 2 ** 30;

/**
 * How the tier holds its bytes: `"array"` — plain `ArrayBuffer`s in the page (form A);
 * `"upload-buffer"` — `MAP_WRITE | COPY_SRC` GPUBuffers filled once at creation and never mapped
 * again (form B).
 */
export type HostTierForm = "array" | "upload-buffer";

/**
 * Run order: B first. Its memory is freed deterministically by `destroy()`, so A's allocation does
 * not start on top of a tier the garbage collector has not reclaimed yet.
 */
export const HOST_TIER_FORMS: readonly HostTierForm[] = ["upload-buffer", "array"];

/** Bytes per chunk of each form (B's is also capped by the device's `maxBufferSize`). */
export const HOST_TIER_CHUNK_BYTES: Readonly<Record<HostTierForm, number>> = {
  array: GIB,
  "upload-buffer": GIB / 2,
};

/** How a hit brings an expert from the tier into the pool. */
export type HitMethod = "ring" | "writeBuffer" | "copy";

/** The methods of each form, in run order. */
export const HIT_METHODS: Readonly<Record<HostTierForm, readonly HitMethod[]>> = {
  array: ["ring", "writeBuffer"],
  "upload-buffer": ["copy"],
};

/** Experts per hit sample, in run order. */
export const HIT_BATCHES: readonly number[] = [1, 8, 16, 32];

/** Slots of the VRAM pool a hit writes into (the largest batch). */
export const POOL_EXPERTS = 32;

/** The largest target the form accepts, GiB. */
export const MAX_TARGET_GIB = 64;

/** "13" → 13: an integer from 1 to {@link MAX_TARGET_GIB}; undefined otherwise. */
export const parseTargetGib = (text: string): number | undefined => {
  const value = Number(text.trim());
  return text.trim() !== "" && Number.isSafeInteger(value) && value >= 1 && value <= MAX_TARGET_GIB
    ? value
    : undefined;
};

/**
 * The chunk sizes that add up to `targetBytes`: whole chunks of `chunkBytes`, then the remainder.
 * Throws unless both are positive multiples of 4 (a GPUBuffer's size and every copy offset are).
 */
export const allocationPlan = (targetBytes: number, chunkBytes: number): number[] => {
  for (const [name, value] of [
    ["targetBytes", targetBytes],
    ["chunkBytes", chunkBytes],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 4 || value % 4 !== 0) {
      throw new Error(`${name} must be a positive multiple of 4 (received: ${value})`);
    }
  }
  const whole = Math.floor(targetBytes / chunkBytes);
  const rest = targetBytes - whole * chunkBytes;
  return [...Array.from({ length: whole }, () => chunkBytes), ...(rest === 0 ? [] : [rest])];
};

/** Where one expert of the tier sits: its chunk and the byte offset in it. */
export type ExpertPlace = { readonly chunk: number; readonly offset: number };

/**
 * Every whole expert the chunks hold, packed from the start of each chunk, so none straddles two
 * chunks (the tail of a chunk shorter than an expert holds none). Offsets are multiples of
 * {@link EXPERT_BYTES}, hence of 4.
 */
export const placeExperts = (chunkSizes: readonly number[]): ExpertPlace[] =>
  chunkSizes.flatMap((size, chunk) =>
    Array.from({ length: Math.floor(size / EXPERT_BYTES) }, (_, index) => ({
      chunk,
      offset: index * EXPERT_BYTES,
    })),
  );

/** xorshift32 in [0, 1), so a run's draws are fixed by their seed. */
export const xorshift32 = (seed: number): (() => number) => {
  let state = seed >>> 0 || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
};

/**
 * `count` distinct indices of `[0, total)`, drawn by a partial Fisher–Yates shuffle from `seed`.
 * Throws when `count` exceeds `total`.
 */
export const drawDistinct = (count: number, total: number, seed: number): number[] => {
  if (count > total) throw new Error(`cannot draw ${count} distinct experts out of ${total}`);
  const next = xorshift32(seed);
  // The swapped entries of the virtual array [0, total): only the touched ones are stored.
  const swapped = new Map<number, number>();
  const at = (index: number): number => swapped.get(index) ?? index;
  return Array.from({ length: count }, (_, index) => {
    const pick = index + Math.floor(next() * (total - index));
    const value = at(pick);
    swapped.set(pick, at(index));
    return value;
  });
};

/**
 * Writes every byte of `target` by repeating `pattern` (the last repeat cut short), so every page
 * of a fresh allocation is committed. Throws on an empty pattern.
 */
export const fillByRepeating = (target: Uint8Array, pattern: Uint8Array): void => {
  if (pattern.length === 0) throw new Error("fillByRepeating needs a non-empty pattern");
  let offset = 0;
  for (; offset + pattern.length <= target.length; offset += pattern.length) {
    target.set(pattern, offset);
  }
  target.set(pattern.subarray(0, target.length - offset), offset);
};

/** ms per expert of a batch once the fence is taken off: (batch − fence) / experts. */
export const perExpertMs = (batchMs: number, fenceMs: number, experts: number): number =>
  (batchMs - fenceMs) / experts;

/**
 * The least-squares slope of `y` over `x` (ms per expert when x = experts and y = the batch's
 * median ms: the fixed cost of a batch, such as the fence, falls into the intercept). Undefined with
 * fewer than two distinct x.
 */
export const leastSquaresSlope = (
  points: readonly { readonly x: number; readonly y: number }[],
): number | undefined => {
  if (points.length === 0) return undefined;
  const meanX = points.reduce((sum, { x }) => sum + x, 0) / points.length;
  const meanY = points.reduce((sum, { y }) => sum + y, 0) / points.length;
  const spread = points.reduce((sum, { x }) => sum + (x - meanX) ** 2, 0);
  if (spread === 0) return undefined;
  return points.reduce((sum, { x, y }) => sum + (x - meanX) * (y - meanY), 0) / spread;
};
