// Behaviour tests of the approximate RGB preview of an intermediate latent. Pure functions that
// need neither a GPU nor assets.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { approximatePreview } from "../src/anima/preview.ts";

/**
 * A synthetic 16ch 2×2 latent (planar `[1,16,2,2]`). The values `{-1,-0.5,0,0.5,1}` are dealt out
 * shifted by channel × pixel — every channel and every pixel contributes differently, so rewriting
 * any row of the coefficient table moves the output (the premise of the golden below).
 */
const syntheticLatents = (): Float32Array<ArrayBuffer> =>
  Float32Array.from(
    { length: 16 * 4 },
    (_, at) => (((Math.floor(at / 4) + (at % 4)) % 5) - 2) * 0.5,
  );

test("approximatePreview: entry check (axis count / B>1 / channel count / element count)", () => {
  assertThrows(
    () => approximatePreview(new Float32Array(16), [16, 2, 2]),
    Error,
    "does not have 4 axes",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(2 * 16 * 4), [2, 16, 2, 2]),
    Error,
    "assumes batch=1",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(4 * 4), [1, 4, 2, 2]),
    Error,
    "channel count",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(16 * 4 + 1), [1, 16, 2, 2]),
    Error,
    "element count",
  );
});

test("approximatePreview: dimensions are the latent's H/W, alpha 255", () => {
  // Swapping width and height still matches the element count, so a non-square (2×3) shape pins the
  // positions.
  const got = approximatePreview(new Float32Array(16 * 6), [1, 16, 3, 2]);
  assertEquals([got.width, got.height], [2, 3]);
  assertEquals(got.rgba.length, 2 * 3 * 4);
  for (let index = 0; index < 6; index += 1) {
    assertEquals(got.rgba[index * 4 + 3], 255, `alpha of pixel ${index}`);
  }
});

test("approximatePreview: RGBA golden of a synthetic latent (stops rewrites of the coefficient table)", () => {
  // The coefficient table is a private internal constant, so the output is all that can be pinned
  // from outside. This golden is the check of the coefficients.
  //
  // Fault injection (measured 2026-08-24; the injections were reverted at once):
  // ① Rewriting the first row's `-0.1299` of the coefficient table in `preview.ts` to `-0.1199` and
  //   running this one case failed with `FAILED | 2 passed | 1 failed` (one difference, the 4th
  //   byte `78` → `77`).
  // ② Sweeping **all 51** of the 16×3 coefficients + 3 biases with +0.01 one at a time gave
  //   `constants for which delta=0.01 does not move the golden: none` — a rewrite of any constant
  //   fails.
  const got = approximatePreview(syntheticLatents(), [1, 16, 2, 2]);
  assertEquals([got.width, got.height], [2, 2]);
  assertEquals(
    [...got.rgba],
    [133, 173, 84, 255, 78, 127, 32, 255, 102, 120, 114, 255, 48, 4, 25, 255],
  );
});
