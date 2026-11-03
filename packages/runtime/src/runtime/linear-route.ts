/**
 * **Route decisions** of linear that the fusion pass and recipe derivation must agree on — the single
 * source of truth for "does this linear take the i8a8 path", "is this linear an integer GEMV
 * linear" and "is this linear a byte-plane GEMV linear".
 *
 * Recipe derivation (`buildLinear` in recipe-builders/linear.ts) and the fusion pass (the linear
 * prologue and the packed pairing in fusion.ts, the linear→SRQ rule) call the **same functions**:
 * the fusion pass folds a producer into the linear's step ① only when that step ① exists, and packs a
 * producer SRQ only for consumers that read packed words, so writing a decision twice would let one
 * copy move alone, and a linear would silently receive an input in a form it does not read.
 *
 * NOTE: the `export` is for recipe derivation and the fusion pass (not part of the public surface of
 * mod.ts — ADR 0008). It imports no GPU module, so the estimate and the GPU-less tests can call it.
 */

import { type IntegerGemvVariant, integerGemvVariantFor } from "../kernels/linear-gemv-integer.ts";
import { type PlanesGemvVariant, planesGemvVariantFor } from "../kernels/linear-gemv-planes.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import { LINEAR_OP, numel } from "../ops.ts";
import type { NodePlan } from "./plan.ts";

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

/**
 * The **only pure function** that decides whether a linear is an **integer GEMV linear**
 * (`SessionOptions.integerGemv`) and with which variant: the option is on, the weight is i2-, i4- or
 * i8-resident, M = numel(input shape without its last dimension) is 1..8, and
 * {@link integerGemvVariantFor} names a measured variant for (storage, group, M, n, k).
 *
 * MUST: the packed pairing (`readsPackedActivation` in fusion.ts), the linear→SRQ rule and the recipe
 * builder's guard read this same function. The integer kernel binds the activation as `vec4<u32>`
 * packed codes only, so if one side were wider than the other, f32 words would flow into that
 * binding — a silent wrong value with no exception. The kernel runs only when the activation does
 * arrive packed; a linear this function accepts whose producer was not packed keeps its existing
 * path.
 *
 * @param weightStorage The resident storage of the linear's weight slot (`undefined` = not resident,
 * read as f32).
 * @param groupSize The i4 group length of the slot (`undefined` for any other storage).
 */
export const linearIntegerGemvVariant = (
  integerGemv: boolean | undefined,
  plan: NodePlan,
  weightStorage: WeightStorage | undefined,
  groupSize: number | undefined,
): IntegerGemvVariant | undefined => {
  if (integerGemv !== true || plan.node.op !== LINEAR_OP || weightStorage === undefined) {
    return undefined;
  }
  const [x, weight] = plan.inputShapes;
  const [n, k] = weight;
  return integerGemvVariantFor(weightStorage, groupSize, numel(x.slice(0, -1)), n, k);
};

/**
 * The **only pure function** that decides whether a linear is a **byte-plane GEMV linear**
 * (`SessionOptions.linearGemvPlanes`) and with which variant: the option is on, the weight is
 * i2-resident, M = numel(input shape without its last dimension) is 1..8, and
 * {@link planesGemvVariantFor} names a measured variant for (M, n, k).
 *
 * MUST: the recipe builder (the branch and its guard against packed activations), the packed
 * pairing (`readsPackedActivation` in fusion.ts) and the linear→SRQ rule read this same function.
 * The kernel binds the activation as f32 quads, so a linear it accepts must never be paired with
 * packed int8 words (they would be read as f32 — a silent wrong value), and a fusion rule must not
 * take the linear's window and build another kernel in its place (the option would do nothing,
 * silently).
 *
 * @param weightStorage The resident storage of the linear's weight slot (`undefined` = not resident,
 * read as f32).
 */
export const linearPlanesGemvVariant = (
  linearGemvPlanes: boolean | undefined,
  plan: NodePlan,
  weightStorage: WeightStorage | undefined,
): PlanesGemvVariant | undefined => {
  if (linearGemvPlanes !== true || plan.node.op !== LINEAR_OP || weightStorage === undefined) {
    return undefined;
  }
  const [x, weight] = plan.inputShapes;
  const [n, k] = weight;
  return planesGemvVariantFor(weightStorage, numel(x.slice(0, -1)), n, k);
};
