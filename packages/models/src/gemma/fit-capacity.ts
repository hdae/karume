/**
 * `capacity: "fit"` — the largest capacity this device can hold right now.
 *
 * WebGPU exposes no total or free VRAM (ADR 0070 decision 5), so each candidate is settled by one
 * trial allocation of what its conversation would add. The result is a fact of that moment (other
 * processes move it), not a reservation, and it is not deterministic across machines or runs; the
 * generated tokens do not depend on it, as capacity does not move tokens (limitations).
 *
 * DECIDED: docs/decisions/0083-generation-api-surface.md#addendum-2026-11-06--capacity-fit
 */

/**
 * What a candidate must leave free on top of its estimate. The sustained VRAM ran up to about
 * 320 MiB over the estimate, and short peaks 430–670 MiB more (research 2026-11-05 §5), so 1 GiB
 * covers both with little to spare.
 */
export const GEMMA4_FIT_MARGIN_BYTES = 1024 ** 3;

/**
 * The capacities tried, largest first: the position limit, the powers of two below it, and the
 * distribution's default as the last one. Nothing below the default is tried: the default is what
 * the distribution promises to work.
 */
export const fitCapacityCandidates = (floor: number, maxPosition: number): readonly number[] => {
  const candidates = [maxPosition];
  const start = 2 ** Math.ceil(Math.log2(maxPosition)) / 2;
  for (let capacity = start; capacity > floor; capacity /= 2) candidates.push(capacity);
  if (floor < maxPosition) candidates.push(floor);
  return candidates;
};

/**
 * The first candidate whose need fits. `needBytes` returns `undefined` for a candidate this device
 * cannot hold whatever is free (a limit on one buffer). `undefined` = none fits.
 */
export const pickFitCapacity = async (
  candidates: readonly number[],
  needBytes: (capacity: number) => number | undefined,
  fits: (bytes: number) => Promise<boolean>,
): Promise<number | undefined> => {
  for (const capacity of candidates) {
    const need = needBytes(capacity);
    if (need !== undefined && (await fits(need + GEMMA4_FIT_MARGIN_BYTES))) return capacity;
  }
  return undefined;
};
