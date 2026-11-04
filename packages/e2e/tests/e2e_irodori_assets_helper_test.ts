// Behaviour tests of `worstDifference`, the comparator of `tests/helpers/irodori-assets.ts`.
//
// This comparator is **the only number** the tolerance checks of the real-GPU E2E
// (`e2e_irodori_latent_test.ts` / `e2e_irodori_ab_test.ts`) look at, and it is buried inside
// checks that need both the assets and a GPU. A regression of the comparator itself would
// therefore only show as "the check stays green and detects nothing" — this is the only place it
// can be exercised without assets.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { worstDifference } from "./helpers/irodori-assets.ts";

test("worstDifference: returns the maximum absolute difference and its position", () => {
  const actual = Float32Array.from([1, 2, 3, 4]);
  const golden = Float32Array.from([1, 2.5, 3, 4.25]);
  const { maxAbs, at } = worstDifference(actual, golden);
  assertEquals(at, 1, "position of the worst");
  assertEquals(Math.round(maxAbs * 100) / 100, 0.5);
});

test("worstDifference: fails when the element counts differ", () => {
  assertThrows(
    () => worstDifference(Float32Array.from([1]), Float32Array.from([1, 2])),
    Error,
    "element counts differ",
  );
});

test('worstDifference: does not pass all-NaN elements as "difference 0"', () => {
  // A naive `difference > maxAbs` is always false for NaN, so maxAbs comes back at its initial 0
  // and the upstream `maxAbs > zAtol` is false = **an all-NaN latent slips through the tolerance
  // check**.
  const nan = Float32Array.from([NaN, NaN, NaN]);
  assertThrows(
    () => worstDifference(nan, Float32Array.from([0, 0, 0])),
    Error,
    "3 non-finite samples",
  );
});

test("worstDifference: also fails on a single non-finite value among finite samples (with its position)", () => {
  // The shape "1 finite sample + NaN for the rest" gives a small maxAbs (not caught by the lower
  // bound of the discrimination band).
  const actual = Float32Array.from([0.1, Infinity, 0.2]);
  const error = assertThrows(
    () => worstDifference(actual, Float32Array.from([0.1, 0.1, 0.2])),
    Error,
    "1 non-finite samples",
  );
  assertEquals(error.message.includes("first at 1"), true, error.message);
});
