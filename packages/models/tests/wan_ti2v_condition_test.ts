// Checks of the host functions for conditioning Wan2.2's I2V DiT (`wanConditionMask` and
// `withWanFirstFrameCondition` of `src/wan/dit-loop.ts` — ADR 0121 decision 11). Pure functions, so
// neither a GPU nor assets are needed.
//
// Three things are pinned:
//
// (1) The defensive branches of the condition mask (fail loudly) — passing a condition with the input
//     form "t2v", and building I2V with a temporal patch ≠ 1. The current public API cannot reach them
//     (the plan rejects 2.1's image first, and `wanDitPatch` fixes the temporal patch at 1), so this
//     pins that they do not silently pass when the wiring changes.
// (2) The condition mask is true only on the tokens of the first latent frame, `[0, rows·cols)`.
// (3) The replacement computes upstream diffusers' formula `(1 − m)·condition + m·latents` verbatim —
//     giving the same values as upstream down to signed zeros. Simplifying it to "copy the first frame"
//     diverges from upstream on ±0 elements (the expected values below were worked out by hand from
//     upstream's formula in f32). The I2V sha rows also catch a ±0 difference, but a red far away takes
//     effort to attribute, so it is pinned directly here.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { wanConditionMask, withWanFirstFrameCondition } from "../src/wan/dit-loop.ts";
import { type WanPatchGeometry, wanTokenGrid } from "../src/wan/dit-tokens.ts";

const OWNER = "test";

/** The I2V patch (temporal 1, spatial 2×2 — the same stride as `wanDitPatch`). */
const PATCH: WanPatchGeometry = { channels: 4, patchFrames: 1, patchHeight: 2, patchWidth: 2 };

/** Latent `[4, 3, 4, 6]` → token grid 3 × 2 × 3 (the first latent frame has 6 tokens). */
const GRID = wanTokenGrid([4, 3, 4, 6], PATCH);

/** Shows signed zeros distinctly (the diff display of `assertEquals` writes −0 and +0 both as "0"). */
const signed = (value: number): string => (Object.is(value, -0) ? "-0" : String(value));

/** Checks with `Object.is` that the values match down to signed zeros. */
const assertSameValue = (actual: number, expected: number, label: string): void =>
  assert(Object.is(actual, expected), `${label}: ${signed(actual)} (expected ${signed(expected)})`);

describe("wanConditionMask (the I2V condition mask)", () => {
  it("passing a condition latent with the input form 't2v' throws that there is no condition-mask input", () => {
    assertThrows(
      () => wanConditionMask("t2v", GRID, PATCH, true, OWNER),
      Error,
      `${OWNER}: a DiT of input form 't2v' does not accept a condition latent`,
    );
  });

  it("I2V with a temporal patch other than 1 throws that the first latent frame does not align with a token frame", () => {
    const patch: WanPatchGeometry = { ...PATCH, patchFrames: 2 };
    // A grid of the 4 latent frames divided by a temporal patch of 2 (only the patch changes, keeping
    // the patch and the grid consistent).
    const grid = wanTokenGrid([4, 4, 4, 6], patch);
    assertThrows(
      () => wanConditionMask("ti2v", grid, patch, true, OWNER),
      Error,
      `${OWNER}: with temporal patch 2 the first latent frame does not align with the token frames`,
    );
  });

  it("in I2V only the tokens of the first latent frame [0, rows·cols) are true, all others false", () => {
    const mask = wanConditionMask("ti2v", GRID, PATCH, true, OWNER);
    assertEquals(GRID.rows * GRID.cols, 6, "precondition: the first latent frame is 2 × 3 tokens");
    const expected = new Uint32Array(GRID.count);
    expected.fill(1, 0, 6);
    assertEquals(mask, expected);
  });
});

describe("withWanFirstFrameCondition (replacing the first frame of the model input)", () => {
  it("throws when the latent's element count does not match the latent shape", () => {
    assertThrows(
      () => withWanFirstFrameCondition(new Float32Array(7), new Float32Array(4), [1, 2, 1, 4]),
      Error,
      "Condition replacement: latent 7 elements, condition 4 elements do not match the latent shape [1, 2, 1, 4] (the condition is 1 frame)",
    );
  });

  it("throws when the condition is not the element count of one frame (a condition for 2 frames is passed)", () => {
    assertThrows(
      () => withWanFirstFrameCondition(new Float32Array(8), new Float32Array(8), [1, 2, 1, 4]),
      Error,
      "Condition replacement: latent 8 elements, condition 8 elements",
    );
  });

  it("matches upstream's formula down to signed zeros (diverging from values that just copy the first frame)", () => {
    // Latent `[C=1, F=2, H=1, W=4]`, condition `[1, 1, 1, 4]`. The condition is broadcast to every frame.
    const condition = Float32Array.of(-0, -0, -2, 3);
    const latents = Float32Array.of(
      // first frame (m = 0 → 1·cond + 0·lat)
      +0,
      1,
      5,
      7,
      // 2nd frame (m = 1 → 0·cond + 1·lat)
      11,
      13,
      -0,
      -0,
    );
    const out = withWanFirstFrameCondition(latents, condition, [1, 2, 1, 4]);
    // cond = −0, lat = +0 → −0 + 0·(+0) = −0 + +0 = +0 (copying gives −0).
    assertSameValue(out[0], +0, "first frame [0]");
    // cond = −0, lat = 1 → −0 + +0 = +0 (copying gives −0).
    assertSameValue(out[1], +0, "first frame [1]");
    // The non-zero elements are the condition in the first frame and the latent itself in the 2nd frame.
    assertSameValue(out[2], -2, "first frame [2]");
    assertSameValue(out[3], 3, "first frame [3]");
    assertSameValue(out[4], 11, "2nd frame [0]");
    assertSameValue(out[5], 13, "2nd frame [1]");
    // cond = −2 < 0, lat = −0 → 0·(−2) + −0 = −0 + −0 = −0.
    assertSameValue(out[6], -0, "2nd frame [2]");
    // cond = 3 > 0, lat = −0 → 0·3 + −0 = +0 + −0 = +0 (keeping the latent as is gives −0).
    assertSameValue(out[7], +0, "2nd frame [3]");
  });
});
