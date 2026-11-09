/** The pure parts of the expert transfer bench: the case plan, the settings parse and the summary. */

/** One Gemma 4 26B A4B expert (gate_up + down at q4_0 with their scales), bytes. */
export const EXPERT_BYTES = 3_345_408;

/** One decode token's router picks read back to the host: 30 layers × 8 u32 ids. */
export const ROUTER_READBACK_BYTES = 30 * 8 * 4;

export type TransferCase =
  | { readonly id: string; readonly kind: "fence-empty"; readonly bytes: 0 }
  | { readonly id: string; readonly kind: "fence-copy"; readonly bytes: number }
  | { readonly id: string; readonly kind: "readback"; readonly bytes: number }
  | {
      readonly id: string;
      readonly kind: "upload";
      readonly method: "ring" | "writeBuffer";
      readonly experts: number;
      readonly bytes: number;
    };

/**
 * The cases in run order: the fence floor first (an empty submit, then a 4-byte copy), the
 * router-pick readback, then each batch of experts by staging ring and by `writeBuffer`.
 */
export const transferCases = (batches: readonly number[]): TransferCase[] => [
  { id: "fence-empty", kind: "fence-empty", bytes: 0 },
  { id: "fence-copy", kind: "fence-copy", bytes: 4 },
  { id: "readback-router", kind: "readback", bytes: ROUTER_READBACK_BYTES },
  ...batches.flatMap((experts): TransferCase[] =>
    (["ring", "writeBuffer"] as const).map((method) => ({
      id: `upload-${method}-${experts}`,
      kind: "upload",
      method,
      experts,
      bytes: experts * EXPERT_BYTES,
    })),
  ),
];

/**
 * "1, 4,8" → [1, 4, 8]: positive integers, unique, in the order given. Undefined for an empty list,
 * a non-integer, a value below 1 or above `max`, or a repeat.
 */
export const parseBatches = (text: string, max: number): number[] | undefined => {
  const parts = text
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
  if (parts.length === 0) return undefined;
  const values = parts.map(Number);
  const valid = values.every((value) => Number.isSafeInteger(value) && value >= 1 && value <= max);
  return valid && new Set(values).size === values.length ? values : undefined;
};

export type Summary = {
  readonly min: number;
  readonly median: number;
  readonly p95: number;
  readonly max: number;
  readonly mean: number;
};

/** Nearest-rank quantile of sorted values (q in [0, 1]). */
const quantile = (sorted: readonly number[], q: number): number =>
  sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))] ?? Number.NaN;

/** Min, median, p95 (nearest rank), max and mean of the samples; throws on an empty list. */
export const summarize = (samples: readonly number[]): Summary => {
  if (samples.length === 0) throw new Error("summarize needs at least one sample");
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    min: sorted[0] ?? Number.NaN,
    median: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted[sorted.length - 1] ?? Number.NaN,
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  };
};

/** GB/s (10⁹ bytes per second) of `bytes` moved in `ms`; undefined when nothing moved. */
export const gigabytesPerSecond = (bytes: number, ms: number): number | undefined =>
  bytes === 0 || ms <= 0 ? undefined : bytes / ms / 1e6;
