// Behaviour tests of Anima's deterministic random numbers (the generator that fully determines the
// initial latent). Needs neither a GPU nor assets.
//
// `src/anima/random.ts` is an **intentional duplicate** of sbv2 / irodori (the doc of
// `generation/random.ts`), so the other families' tests (`sbv2_host_test.ts` /
// `irodori_host_test.ts`) do not detect a regression in anima's copy. The PNG sha256 check on a
// real GPU (`e2e_anima_test.ts`) does not run on a machine without assets, so this is the only
// asset-free check.
//
// NOTE: The `u1 === 0` substitution in `normals` of `random.ts` (avoiding `log(0) = −Inf`) happens
// with probability 2⁻⁵³ per draw, and there is no fault-injection point (a replaceable uniform
// source) either, so it **cannot be hit directly** — this one branch is stated as untestable
// (covering it would need a wider generator surface, which would loosen the source of truth for the
// values).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { nextUint64, Randn } from "../src/anima/random.ts";
import { assertAcceptableSeed } from "../src/request-gates.ts";

test("Randn: the same seed gives the same sequence, a different seed a different one", () => {
  const first = new Randn(7).normals(64);
  const second = new Randn(7).normals(64);
  const other = new Randn(8).normals(64);
  assertEquals([...first], [...second]);
  assert(
    [...first].some((value, index) => value !== other[index]),
    "changing the seed gives the same sequence",
  );
});

test("Randn: an odd length does not carry a Box–Muller pair over", () => {
  // Carrying over would make "the same seed gives a different sequence depending on call order",
  // breaking the reproducibility a seed promises. It is pinned by an odd length → next call
  // agreeing with the same two calls on a fresh generator.
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  const fresh = new Randn(5);
  assertEquals([...head], [...fresh.normals(3)]);
  assertEquals([...tail], [...fresh.normals(2)]);
});

test("Randn: an odd-length call drops the rest of the pair (the sin side) and the next call starts a new pair", () => {
  // Comparing only two instances of the same implementation would pass even after a change to carry
  // the rest over (both would carry it the same way). Comparing against a 6-element sequence drawn
  // in one go pins that the one value after the 3rd element (the sin side of the pair) is dropped.
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  const whole = new Randn(5).normals(6);
  assertEquals([...head], [...whole.subarray(0, 3)]);
  assertEquals([...tail], [...whole.subarray(4, 6)]);
});

test("nextUint64: the known splitmix64 sequence (state 0 → 0xe220a8397b1dcdaf …)", () => {
  // The normals (f32) can drift between engines through implementation differences in
  // `Math.cos/sin/log`, so one splitmix64 step (the mixing constants and arithmetic) is pinned on
  // the environment-independent integer part. The mapping from the seed to the internal state and
  // the derivation of the uniforms are not pinned here. The expected values come from evaluating
  // the public splitmix64 reference implementation with Python integers.
  let state = 0n;
  const values: bigint[] = [];
  for (let draw = 0; draw < 4; draw += 1) {
    const next = nextUint64(state);
    state = next.state;
    values.push(next.value);
  }
  assertEquals(values, [
    0xe220a8397b1dcdafn,
    0x6e789e6aa1b965f4n,
    0x06c45d188009454fn,
    0xf88bb8a8724c81ecn,
  ]);
});

test("Randn: scale acts linearly on the same sequence (the sequence itself is determined by the seed)", () => {
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2), `element ${index}`);
  }
});

test("Randn: the accepted set of seeds is the same as assertAcceptableSeed (the generator-side entry)", () => {
  // There is only one accepted set (if the conditions of `generate`'s entry and the generator
  // diverged, only one of them would loosen).
  for (const seed of [-1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assertThrows(
      () => assertAcceptableSeed(seed),
      ModelInputError,
      "not a non-negative safe integer",
    );
    assertThrows(() => new Randn(seed), ModelInputError, "not a non-negative safe integer");
  }
  assertAcceptableSeed(0);
  assertEquals(new Randn(0).normals(2).length, 2);
});
