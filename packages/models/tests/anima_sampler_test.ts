// Behaviour tests of Anima's sampler (the stage that is not in the IR). Pure functions that need
// neither a GPU nor assets.
//
// Numerical parity with the reference fixture (comparing the timesteps_proj table) comes back **in
// the same iteration as the E2E with a real GPU and real assets** (P3, iteration 2). This file
// covers shapes and invariants. NOTE: The bit identity of `latents_mean` / `latents_std` became an
// asset-free check (the reference fixture was cut out and committed — `anima_latents_test.ts`).

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { imageToRgba } from "../src/anima/image.ts";
import {
  assertAcceptableSteps,
  cfgEulerStep,
  sigmaSchedule,
  timestepsProj,
} from "../src/anima/sampler.ts";
import { assertPromptTokenLengths } from "../src/anima/text/tokenizer.ts";
import { ModelInputError } from "../src/errors.ts";
import { assertEncodableText } from "../src/text/code-points.ts";
// `needsUncond` is defined on the shared side that does not depend on the update rule (`sampler.ts`
// imports it).
import { needsUncond } from "../src/generation/dpm-solver-multistep.ts";

/**
 * The same values as `pipelineConfig.scheduler` in `models/karume-anima/karume.json` (for the shape
 * checks).
 */
const SHIFT = 3;
const NUM_TRAIN_TIMESTEPS = 1000;

test("sigmaSchedule: length steps+1, ends at 0, strictly decreasing", () => {
  const sigmas = sigmaSchedule(8, SHIFT);
  assertEquals(sigmas.length, 9);
  assertEquals(sigmas[0], 1, "the start is linspace's start=1 (still 3/3 = 1 after the shift)");
  assertEquals(sigmas[8], 0, "one terminal 0 is appended");
  for (let index = 1; index < 8; index += 1) {
    assert(sigmas[index] < sigmas[index - 1], `σ is not strictly decreasing (index=${index})`);
  }
});

test("sigmaSchedule: rejects steps < 2 because the linspace denominator becomes 0", () => {
  assertThrows(() => sigmaSchedule(1, SHIFT), ModelInputError);
  assertThrows(() => sigmaSchedule(2.5, SHIFT), ModelInputError);
});

test("sigmaSchedule: the shift takes the manifest's value (1 gives the plain linspace)", () => {
  // With shift=1 it collapses to `1·σ / (0·σ + 1) = σ`. If the wiring that takes the scheduler
  // values from the manifest were dead and fixed to a constant, this one case would fail.
  const plain = sigmaSchedule(4, 1);
  assertEquals(
    [...plain].slice(0, 4).map((value) => Math.fround(value)),
    [
      Math.fround(1),
      Math.fround(1 + (0.25 - 1) / 3),
      Math.fround(1 + (2 * (0.25 - 1)) / 3),
      Math.fround(0.25),
    ],
  );
  const shifted = sigmaSchedule(4, SHIFT);
  assert(shifted[1] !== plain[1], "the shift has no effect");
});

test("sigmaSchedule: with the representative shifts the ladder starts at exactly 1 (DPM++ 2M's premise)", () => {
  // `dpm-solver-multistep.ts` derives step 0's `alphaS0 = 0` from "the start is exactly 1" and
  // throws no exception itself. An upward drift is rejected by `sigmaSchedule`'s check, and a
  // downward ulp-sized drift is accepted (the describe below). These four representative shifts pin
  // a start of exactly 1.
  for (const shift of [1, 1.5, 3, 12]) {
    assertStrictEquals(sigmaSchedule(8, shift)[0], 1, `start for shift=${shift}`);
  }
});

test("sigmaSchedule: fails loudly outside the region where the ladder is valid (no silently running another trajectory)", () => {
  // The region where the denominator `f32(shift−1)+1` collapses to exactly 0 in f32 → the start is
  // +Inf.
  assertThrows(
    () => sigmaSchedule(8, 1e-8),
    RangeError,
    "is outside the range where the ladder holds",
  );
  // The region where it does not collapse but the start is no longer 1 (measured 0.8388…).
  assertThrows(
    () => sigmaSchedule(8, 1e-7),
    RangeError,
    "is outside the range where the ladder holds",
  );
  // The region where the start stays 1 but the next value collapses to the same f32 step (the side
  // where strict monotonicity breaks).
  assertThrows(() => sigmaSchedule(8, 1e7), RangeError, "is not strictly decreasing");
});

describe("sigmaSchedule: fractional shifts whose start drifts from 1 by an ulp", () => {
  // DPM++ 2M takes `log(1 − σ₀)` at step 0, so if σ₀ exceeds 1 by even one ulp the log of a
  // negative number gives NaN, and it fails late after running every step on NaN. A drift to σ₀ < 1
  // is harmless with `alphaS0 ≈ 6e-8`.
  it("a shift with σ₀ above 1 (1.32 → 1 + 2⁻²³) is rejected with a RangeError before touching the GPU", () => {
    // "Outside the region where the ladder is valid" is also raised by the monotonicity check, so
    // the wording of the start check and the value of σ₀ are matched to pin that the upper branch
    // failed.
    assertThrows(
      () => sigmaSchedule(8, 1.32),
      RangeError,
      `The head of the sigma ladder is ${1 + 2 ** -23} (above 1`,
    );
  });

  it("a shift with σ₀ drifting below 1 (1.18 → 1 − 2⁻²³) is accepted", () => {
    const sigmas = sigmaSchedule(8, 1.18);
    assertStrictEquals(
      sigmas[0],
      1 - 2 ** -23,
      "1.18 drifts down by 2 ulp (f32's step below 1, 2⁻²⁴ × 2)",
    );
  });
});

test("timestepsProj: cos in the first half, sin in the second (flip_sin_to_cos=true)", () => {
  // sigma=0 → timestep=0 → angle=0 at every frequency, so the cos side is 1 and the sin side 0. If
  // flipped, the first half would be 0 and the second 1, and this one case would fail.
  const proj = timestepsProj(0, 8, NUM_TRAIN_TIMESTEPS);
  assertEquals([...proj], [1, 1, 1, 1, 0, 0, 0, 0]);
});

test("timestepsProj: the accepted set of the width and num_train_timesteps", () => {
  assertThrows(() => timestepsProj(0.5, 7, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 0, NUM_TRAIN_TIMESTEPS), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 0), RangeError);
  assertThrows(() => timestepsProj(0.5, 8, 1000.5), RangeError);
});

test("needsUncond: only CFG=1 needs no uncond", () => {
  assertEquals(needsUncond(1), false);
  assertEquals(needsUncond(4), true);
  assertEquals(needsUncond(0), true);
});

test("cfgEulerStep: CFG=1 uses cond as is (does not go through uncond)", () => {
  const previous = Float32Array.from([1, 2, 3]);
  const cond = Float32Array.from([0.25, -0.5, 0.125]);
  const got = cfgEulerStep(previous, cond, undefined, -0.5, 1);
  assertEquals([...got], [1 - 0.125, 2 + 0.25, 3 - 0.0625]);
});

test("cfgEulerStep: rejects a presence of uncond that disagrees with needsUncond", () => {
  const previous = Float32Array.from([0, 0]);
  const cond = Float32Array.from([1, 1]);
  const uncond = Float32Array.from([2, 2]);
  assertThrows(
    () => cfgEulerStep(previous, cond, undefined, -0.1, 4),
    Error,
    "requires the DiT output of the uncond side",
  );
  assertThrows(
    () => cfgEulerStep(previous, cond, uncond, -0.1, 1),
    Error,
    "path should not compute the uncond branch",
  );
});

test("cfgEulerStep: CFG!=1 passes uncond + scale·(cond − uncond) to Euler", () => {
  const previous = Float32Array.from([0]);
  const got = cfgEulerStep(previous, Float32Array.from([1]), Float32Array.from([0.5]), -2, 4);
  // noise = 0.5 + 4·(1 − 0.5) = 2.5 → 0 + (−2)·2.5 = −5
  assertEquals([...got], [-5]);
});

test("cfgEulerStep: a factor not representable in f32 is rounded to f32 before multiplying (matches torch to the last digit)", () => {
  // torch rounds a Python float factor to f32 before the product with an f32 tensor. Multiplying in
  // f64 makes the combination drift by 1 ulp for this pair (0.03789997100830078). The expected
  // value is `uncond + 1.1 * (cond - uncond)` evaluated with f32 tensors on the CPU with torch
  // 2.13.0.
  const got = cfgEulerStep(
    Float32Array.from([0]),
    Float32Array.from([0.011]),
    Float32Array.from([0.5]),
    -1,
    1.1,
  );
  assertStrictEquals(got[0], 0.03790003061294556);
});

// ---- Host-facing error type (ADR 0107) --------------------------------------
// Merged from the former `anima_sampler_test.ts`. Its resolution, seed and prompt-lower-bound
// cases were dropped as literal duplicates of `anima_resolution_test.ts`, `anima_random_test.ts`
// and `anima_tokenizer_test.ts`; what stays here is what no family test checks.
//
// The **other side** is pinned too: if a broken internal invariant were raised with this type, the
// host would return a 500-class failure to the caller as "fix your input". A regression that widens
// the type is not caught by the input-side tests.

describe("when Anima's generation request cannot be accepted", () => {
  it("the steps check and the sigma ladder raise the same ModelInputError with the same diagnostic", () => {
    // What can be observed stops at "the two paths agree on type and diagnostic". That `generate`'s
    // entry calls this check first (the early rejection of ADR 0107 decision 6) is not visible from
    // here — reaching the entry needs real assets and a GPU, so that order is covered by the e2e of
    // the GPU lane.
    const direct = assertThrows(() => assertAcceptableSteps(1), ModelInputError);
    const viaSchedule = assertThrows(() => sigmaSchedule(1, 3), ModelInputError);
    assertEquals(
      direct.message,
      viaSchedule.message,
      "the steps diagnostic differs between the two places",
    );
  });

  it("the upper bound of steps is 1000 (exactly at it the ladder is built; above it, a ModelInputError before the ladder is allocated)", () => {
    assertAcceptableSteps(1000);
    assertEquals(sigmaSchedule(1000, 3).length, 1001, "exactly at the bound is inside");
    const direct = assertThrows(
      () => assertAcceptableSteps(1001),
      ModelInputError,
      "is not at most 1000",
    );
    const viaSchedule = assertThrows(() => sigmaSchedule(1001, 3), ModelInputError);
    assertEquals(
      direct.message,
      viaSchedule.message,
      "the steps diagnostic differs between the two places",
    );
    // Evidence that the check acts before the array is allocated: in code without the bound, `new
    // Float32Array(2⁵³)` throws a `RangeError` (Invalid typed array length) at once, so the type
    // differs and this one line turns red.
    assertThrows(
      () => sigmaSchedule(Number.MAX_SAFE_INTEGER, 3),
      ModelInputError,
      "is not at most 1000",
    );
  });

  it("a text with an unpaired surrogate is raised as a ModelInputError", () => {
    // Kept although `anima_tokenizer_test.ts` covers the rejecting side: the accepting side
    // (a paired surrogate) is checked only here, against the owner directly.
    // The source of truth for the condition stays the one `toCodePoints`. The type is raised only
    // when it passes as the text of a generation request.
    const error = assertThrows(() => assertEncodableText("a\ud800b", "prompt"), ModelInputError);
    assert(error.cause instanceof Error, "the owner's diagnostic is kept in cause");
    assert(!(error.cause instanceof ModelInputError), "cause is the shared helper's plain Error");
    assertEncodableText("a𐀀b", "prompt");
  });
});

describe("when an internal invariant of Anima is broken", () => {
  it("a prompt id sequence over the bound is not a ModelInputError (the truncation broke)", () => {
    // Both tokenizers truncate to `maxLength` before returning, so exceeding the bound can only
    // happen when that breaks = a 500-class failure that the caller cannot fix by shortening the
    // prompt.
    const error = assertThrows(
      () => assertPromptTokenLengths("prompt", 513, 4, 512),
      Error,
      "upper bound 512",
    );
    assert(
      !(error instanceof ModelInputError),
      "a 500-class failure is raised with the input-caused type",
    );
  });

  it("a mismatch between the VAE output's element count and dimensions is not a ModelInputError (the caller cannot fix it)", () => {
    // The dimensions of `imageToRgba` come from the geometry (the latent's full length × scale), so
    // a break here is a broken export or wiring, not something fixing the request repairs = a
    // 500-class failure.
    const error = assertThrows(
      () => imageToRgba(new Float32Array(5), 2, 1),
      Error,
      "element count",
    );
    assert(
      !(error instanceof ModelInputError),
      "a 500-class failure is raised with the input-caused type",
    );
  });
});
