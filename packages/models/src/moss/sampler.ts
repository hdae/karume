/**
 * The MOSS-Image sampler on the host — upstream's σ schedule (`MOSSImageScheduler.set_timesteps`)
 * and the Euler update of diffusers' `FlowMatchEulerDiscreteScheduler.step` (ADR NNNN decisions 9
 * and 11, plan item 15-14, plan step 4). The CFG combination is `mossClassifierFreeGuidance`
 * (`./dit-stage.ts`).
 *
 * ## The schedule of `n` requested steps ({@link mossSchedule})
 *
 * 1. The requested grid, upstream's `grid(n, shift)`: `v = torch.linspace(1, lower, n)` in f32 with
 *    `lower = shift / (total + shift − 1)` (a Python float, rounded to f32 as linspace's end), then
 *    `shift·v / (1 + (shift − 1)·v)` elementwise in f32 — torch rounds the Python scalar `shift` to
 *    f32 before it multiplies the tensor, while `shift − 1` is a Python subtraction rounded once
 *    ({@link mossRequestedSigmas}).
 * 2. Snapping to the training grid (the asset `training_grid` — upstream's `grid(total,
 *    training_shift).flip(0)`; `./dit-timestep.ts`): `torch.searchsorted` (left) over the ascending
 *    grid, clamped to the last point, then the nearer of the two neighbours by `|σ − grid|` in f32,
 *    a tie going to the lower neighbour — the smaller σ ({@link mossNearestGridIndex}).
 * 3. Consecutive duplicates removed (`torch.unique_consecutive` — the grid is strictly monotonic,
 *    so equal values are equal indices): `k ≤ n` steps run. Upstream's own `steps_taken` shows it
 *    (500 requested → 415, 1,000 → 615).
 * 4. The σ column is the training σ of each index plus the terminal 0, read from the asset by
 *    index, never recomputed. The timestep of a step (`σ × num_train_timesteps` in f32 —
 *    diffusers' `set_timesteps(sigmas=…)` with the shift held at 1) is the asset's too, looked up
 *    by the same index where it is used (`mossTimestep` / `mossTimestepsProj` in
 *    `./dit-timestep.ts`); the schedule does not keep a second copy of it.
 *
 * ### `torch.linspace` on the CPU (torch 2.13 — ATen `linspace_kernel`)
 *
 * `step = (end − start) / (n − 1)` in f32 (the count converted to f32); element `i` is
 * `start + step·i` for `i < ⌊n/2⌋` and `end − step·(n − 1 − i)` from there on (the formula switches
 * so that both ends are exact), with `i` converted to f32. NOTE (implementation behaviour, not a
 * torch guarantee): each element is **one** rounding — the compiled kernel contracts the product
 * and the sum into an FMA. Measured against the stage-0 record (`moss.host_fixture`, torch
 * 2.13.0+cpu, CPU capability AVX512): this transcription is Uint32-identical in all 202 recorded
 * counts (1–200, 500, 1,000); with two roundings (`f32(start + f32(step·i))`) 183 of the 202
 * requested grids differ in some last bit. The snapping absorbs that: for every count 1–1,000 the
 * snapped schedule is the same either way, so a torch build without FMA contraction would give
 * upstream's schedule too — the requested grid is checked bit for bit only because the snapped
 * values alone would hide a slip anywhere but near a midpoint.
 *
 * The schedule walks the requested grid one element at a time and keeps only the snapped indices,
 * so a large step count costs time (linear in `n`, synchronous — about 0.25 s at 10⁷ on the
 * development machine), not a requested-grid array — upstream accepts any count ≥ 1 and so does
 * this (ADR NNNN decision 9).
 *
 * ### Very large step counts (measured on the series' training grid, shift 5)
 *
 * Up to about 2·10⁶ requested steps the snapped indices strictly increase and `k ≤ total` (977
 * steps from 10⁴ to 2·10⁶). Beyond that the f32 requested σ moves up and down by an ULP near the
 * midpoints between neighbouring training points, so the snapping alternates between them and the
 * de-duplication keeps both: at 5·10⁶ steps 989 indices with 6 decreases, at 10⁷ 1,099 indices
 * (more than `total`) with 61 decreases — a σ column that is not monotonic, some Euler steps with
 * `dt > 0`. Upstream's `set_timesteps` gives the same indices (compared at 12 counts from 10⁴ to
 * 2·10⁷), so this is upstream's behaviour, transcribed, not a bound this module keeps.
 *
 * ## Euler ({@link mossEulerStep})
 *
 * `prev = sample + dt · model_output` with `dt = σ_next − σ` — each operation an f32 rounding, as
 * the CPU tensors of `FlowMatchEulerDiscreteScheduler.step` (non-stochastic, no per-token
 * timesteps) compute it (`Math.fround` of a double sum / product of two f32 values is the f32
 * result).
 *
 * The host tests check the schedule against `moss.host_fixture`'s record (requested grid, snapped
 * indices, σ and timesteps) and CFG + Euler against `moss.reference`'s trajectory, all bit for bit.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { MossTrainingGrid } from "./dit-timestep.ts";

const f32 = Math.fround;

/**
 * A `fma(a, b, c)` of three f32 values rounded once to f32 (C's `fmaf`). The scratch word for the
 * f32 neighbour lives in the closure (no module state).
 *
 * `a·b` is exact in double (2 × 24 ≤ 53 significant bits); the sum with `c` is not, so it is taken
 * as `sum + rest` (TwoSum — `sum` the double rounding, `rest` the exact remainder). Rounding `sum`
 * to f32 rounds `sum + rest` correctly unless `sum` lies exactly halfway between two f32 values
 * while `rest ≠ 0` (a double rounding): the exact value is then on `rest`'s side of that midpoint.
 */
const createFmaF32 = (): ((a: number, b: number, c: number) => number) => {
  const bits = new Float32Array(1);
  const word = new Uint32Array(bits.buffer);
  /** The f32 neighbour of the f32 value `value` towards +∞ (`up`) or −∞. */
  const neighbour = (value: number, up: boolean): number => {
    if (value === 0) return up ? 2 ** -149 : -(2 ** -149);
    bits[0] = value;
    // The magnitude grows with the bit pattern: away from zero is +1 on the pattern.
    word[0] += value > 0 === up ? 1 : -1;
    return bits[0];
  };
  return (a, b, c) => {
    const product = a * b;
    const sum = product + c;
    const virtual = sum - product;
    const rest = product - (sum - virtual) + (c - virtual);
    const rounded = f32(sum);
    if (rest === 0 || rounded === sum) return rounded;
    const other = neighbour(rounded, sum > rounded);
    if ((rounded + other) / 2 !== sum) return rounded;
    return rest > 0 === other > rounded ? other : rounded;
  };
};

// Third-party code notice. `requestedGrid`, `mossNearestGridIndex` and `mossSchedule` below are
// adapted from `MOSSImageScheduler.set_timesteps` in `pipeline_moss_image.py` of
// OpenMOSS-Team/MOSS-Image-SFT (the revision pinned in `tools/export-recipes/moss/sources.py`),
// licensed under the Apache License, Version 2.0, with `torch.linspace`, `torch.searchsorted`,
// `torch.where` and `torch.unique_consecutive` written out as the CPU computes them (module doc);
// `mossEulerStep` from the non-stochastic branch of `FlowMatchEulerDiscreteScheduler.step` in
// huggingface/diffusers (`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`,
// `diffusers==0.39.0`), licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice of the diffusers file,
// copied verbatim from its header (the MOSS file carries none):
//
//   Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.

/** The requested grid of `steps` (module doc, step 1): its length and its element at an index. */
type RequestedGrid = {
  readonly steps: number;
  readonly at: (index: number) => number;
};

/**
 * Upstream's `grid(steps, shift)` with `total` training points (module doc, step 1).
 *
 * MUST: a positive integer `steps` and a finite positive `shift` (the acceptance check gives them —
 * a `RangeError` here is a caller's bug).
 */
const requestedGrid = (steps: number, shift: number, total: number): RequestedGrid => {
  if (!Number.isSafeInteger(steps) || steps < 1) {
    throw new RangeError(`steps ${steps} is not a positive integer`);
  }
  if (!(shift > 0) || !Number.isFinite(f32(shift))) {
    throw new RangeError(`shift ${shift} is not a positive finite f32 value`);
  }
  const start = 1;
  const end = f32(shift / (total + shift - 1));
  const step = steps === 1 ? 0 : f32(f32(end - start) / f32(steps - 1));
  const half = Math.floor(steps / 2);
  const scale = f32(shift);
  const slope = f32(shift - 1);
  const fmaF32 = createFmaF32();
  // torch.linspace(start, end, 1) is [start] (ATen fills it without the kernel).
  const linspace = (index: number): number =>
    steps === 1
      ? start
      : index < half
        ? fmaF32(step, f32(index), start)
        : fmaF32(-step, f32(steps - index - 1), end);
  return {
    steps,
    at: (index) => {
      const value = linspace(index);
      return f32(f32(scale * value) / f32(1 + f32(slope * value)));
    },
  };
};

/**
 * Upstream's requested grid before snapping — `[steps]` f32 (module doc, step 1). The schedule does
 * not materialise it; this is the inspection surface the tests compare with upstream's record.
 */
export const mossRequestedSigmas = (
  steps: number,
  shift: number,
  total: number,
): Float32Array<ArrayBuffer> => {
  const grid = requestedGrid(steps, shift, total);
  return Float32Array.from({ length: steps }, (_, index) => grid.at(index));
};

/**
 * The training-grid index (descending — the asset's order) a requested σ snaps to (module doc,
 * step 2). MUST: a finite σ (a NaN would compare false everywhere and snap silently).
 */
export const mossNearestGridIndex = (grid: MossTrainingGrid, sigma: number): number => {
  if (!Number.isFinite(sigma)) throw new RangeError(`requested σ ${sigma} is not finite`);
  const total = grid.points;
  // Ascending position a is descending index total − 1 − a (upstream flips the grid).
  const ascending = (position: number): number => grid.sigmas[total - 1 - position];
  let low = 0;
  let high = total;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (ascending(middle) < sigma) low = middle + 1;
    else high = middle;
  }
  const upper = Math.min(low, total - 1);
  const lower = Math.max(upper - 1, 0);
  const nearest =
    Math.abs(f32(sigma - ascending(lower))) <= Math.abs(f32(ascending(upper) - sigma))
      ? lower
      : upper;
  return total - 1 - nearest;
};

/** The steps that run for a request (module doc, steps 3 and 4). */
export type MossSchedule = {
  /**
   * The training-grid index of each step (`k` entries, `k ≤` the requested count; descending σ
   * up to about 2·10⁶ requested steps — module doc "Very large step counts").
   */
  readonly gridIndices: readonly number[];
  /** σ of each step and the terminal 0 (`[k + 1]`). */
  readonly sigmas: Float32Array<ArrayBuffer>;
};

/**
 * Upstream's `set_timesteps(steps)` on `grid` with `shift` (module doc). A requested σ that is not
 * finite (a shift so small that the grid formula divides by zero) is a `RangeError` naming the
 * index (upstream would snap it silently).
 */
export const mossSchedule = (
  grid: MossTrainingGrid,
  steps: number,
  shift: number,
): MossSchedule => {
  const requested = requestedGrid(steps, shift, grid.points);
  const gridIndices: number[] = [];
  for (let index = 0; index < steps; index += 1) {
    const sigma = requested.at(index);
    if (!Number.isFinite(sigma)) {
      throw new RangeError(`requested σ ${index} of ${steps} is ${sigma} (shift ${shift})`);
    }
    const nearest = mossNearestGridIndex(grid, sigma);
    if (gridIndices.at(-1) !== nearest) gridIndices.push(nearest);
  }
  const sigmas = new Float32Array(gridIndices.length + 1);
  gridIndices.forEach((index, step) => {
    sigmas[step] = grid.sigmas[index];
  });
  return { gridIndices, sigmas };
};

/**
 * One Euler step of the flow (module doc "Euler"): `latents + f32(σ_next − σ) · prediction`, every
 * operation rounded to f32.
 */
export const mossEulerStep = (
  latents: Float32Array<ArrayBuffer>,
  prediction: Float32Array<ArrayBuffer>,
  sigma: number,
  sigmaNext: number,
): Float32Array<ArrayBuffer> => {
  if (latents.length !== prediction.length) {
    throw new Error(`Euler step of ${latents.length} latents and ${prediction.length} predictions`);
  }
  const delta = f32(sigmaNext - sigma);
  const out = new Float32Array(latents.length);
  for (let index = 0; index < latents.length; index += 1) {
    out[index] = f32(latents[index] + f32(delta * prediction[index]));
  }
  return out;
};
