/**
 * **Route decision** of linear's integer dot-product path (w8a8 / w4a8) — the single source of truth
 * for "does this linear take the i8a8 path".
 *
 * Recipe derivation (`buildLinear` in recipe-builders/linear.ts) and the fusion pass (the linear
 * prologue in fusion.ts — perf-ledger F-4) call the **same function**: the fusion pass folds a
 * producer into the linear's step ① only when that step ① exists, so writing the decision twice
 * would let one copy move alone, and a linear that runs on the f32 path would silently receive a
 * quantized input.
 *
 * NOTE: the `export` is for recipe derivation and the fusion pass (not part of the public surface of
 * mod.ts — ADR 0008). It imports no GPU module, so the estimate and the GPU-less tests can call it.
 */

import type { WeightStorage } from "../kernels/weight-storage.ts";

/**
 * Whether a linear with this resident weight storage and reduction length `k` takes the i8a8 path.
 *
 * The integer dot-product path is taken only when all four conditions hold: **opt-in × integer
 * residency (i8 / i4) × k > 0 × k % 4 == 0** (ADR 0025 / w4a8 is perf-ledger Q-8). With the default
 * "f32", behaviour does not change by a single byte.
 * MUST: the knob `linearCompute: "a8"` means "**quantize the activations to i8 and compute with
 * integer dot products**"; the weights' storage form is a separate axis (i8-resident → w8a8 /
 * i4-resident → w4a8). Leaving i4 out of the predicate would make it a fake option whose selection
 * changes nothing (i4-resident weights would silently flow to the f32 compute path and lose exactly
 * the dp4a gain — docs/research/2026-08-21-anima-i4-seat-speed.md).
 * MUST: include `k > 0`. `k == 0` is a degenerate shape valid by contract (the linear in
 * src/ops.ts allows in=0), but step ① `quantize_rows` of the i8a8 path requires `dim >= 1`, so
 * picking it up gives an i8a8-specific CodegenError — an empty reduction means there are no
 * activations to quantize at all, so the exception carries no meaning as an i8a8 guard; it goes to
 * the plain path so that the path choice does not change the reason for the failure.
 * NOTE: K=0 fails on the plain path too, because a 0-byte binding is below the minimum binding size
 * (a separate cause unrelated to this predicate).
 */
export const linearTakesI8a8 = (
  linearCompute: "f32" | "a8" | "f16" | undefined,
  weightStorage: WeightStorage,
  k: number,
): boolean =>
  linearCompute === "a8" &&
  (weightStorage === "i8" || weightStorage === "i4") &&
  k > 0 &&
  k % 4 === 0;
