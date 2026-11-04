// The basic readers of `pipelineConfig` (`src/config/readers.ts`). They are a lower layer shared
// across families, so if they widened or narrowed an accepted set on their own, the parse of eight
// families would shift at once.
//
// It checks three points:
//
// ① Everything **outside** the accepted set (unknown keys, missing fields, wrong types, unmet
//   conditions) is rejected. No silent degrading to a default.
// ② The error message contains the wording the caller passed (`where` / `key` / `requirement` /
//   `why`) — each family's tests assert the wording verbatim, and which field failed how is the
//   family's responsibility.
// ③ Field presence is checked with `Object.hasOwn` (a cross-cutting invariant) — reading an
//   inherited `toString` and the like as "present" would let an undeclared field pass silently with
//   the prototype's value.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { assertAllowedKeys, readChannels, readNumber, readOnly } from "../src/config/readers.ts";

const isPositive = (value: number): boolean => Number.isFinite(value) && value > 0;

test("assertAllowedKeys: passes when everything is inside the allowed set (an empty object passes too)", () => {
  assertAllowedKeys({ a: 1, b: 2 }, ["a", "b", "c"], "here");
  assertAllowedKeys({}, ["a"], "here");
});

test("assertAllowedKeys: rejects an unknown key, and the message contains the label and the allowed list", () => {
  assertThrows(
    () => assertAllowedKeys({ a: 1, image_mean: 2 }, ["a", "b"], "pipelineConfig"),
    Error,
    "pipelineConfig: unknown key 'image_mean' (allowed: a / b)",
  );
});

test("readNumber: returns a number that meets the condition as is", () => {
  assertEquals(readNumber({ n: 1024 }, "n", "here", isPositive, "not positive"), 1024);
});

test("readNumber: rejects a missing field, and the message contains label.key", () => {
  assertThrows(
    () => readNumber({}, "imageWidth", "pipelineConfig", isPositive, "not positive"),
    Error,
    "pipelineConfig.imageWidth: missing",
  );
});

test("readNumber: the field is checked with Object.hasOwn (an inherited prototype value is not picked up)", () => {
  assertThrows(
    () => readNumber({} as Record<string, unknown>, "toString", "here", isPositive, "not positive"),
    Error,
    "here.toString: missing",
  );
});

test("readNumber: a field whose value is undefined is rejected as a wrong type, not as 'missing'", () => {
  // `Object.hasOwn` does not look at the value, so a field explicitly set to `undefined` counts as
  // "present".
  assertThrows(
    () => readNumber({ n: undefined }, "n", "here", isPositive, "not positive"),
    Error,
    "here.n: not positive (undefined)",
  );
});

test("readNumber: rejects a non-number and a number outside the condition with the same message", () => {
  for (const bad of [0, -1, Number.NaN, "1024", null, [1]]) {
    assertThrows(
      () => readNumber({ n: bad }, "n", "here", isPositive, "not a positive finite number"),
      Error,
      "here.n: not a positive finite number",
    );
  }
});

test("readChannels: accepts a numeric array of length 3 and returns three elements separate from the input", () => {
  // Returning the raw array as is would leave a mutable array from the manifest inside the config
  // (to be rewritten later).
  const source = [0.485, 0.456, 0.406];
  const channels = readChannels(
    { c: source },
    "c",
    "here",
    Number.isFinite,
    "has a non-finite element",
  );
  assertEquals(channels, [0.485, 0.456, 0.406]);
  source[0] = 9;
  assertEquals(channels, [0.485, 0.456, 0.406]);
});

test("readChannels: rejects a missing field", () => {
  assertThrows(
    () => readChannels({}, "imageMean", "pipelineConfig", Number.isFinite, "bad"),
    Error,
    "pipelineConfig.imageMean: missing",
  );
});

test("readChannels: rejects a non-array / an array whose length is not 3", () => {
  for (const bad of [0.5, "rgb", { 0: 1, 1: 2, 2: 3 }, [], [1, 2], [1, 2, 3, 4]]) {
    assertThrows(
      () => readChannels({ c: bad }, "c", "here", Number.isFinite, "bad"),
      Error,
      "here.c: not an array of length 3",
    );
  }
});

test("readChannels: rejects a wrong element type or an unmet condition per element, and the message shows the whole array", () => {
  for (const bad of [
    [1, "2", 3],
    [1, 2, Number.NaN],
    [0.2, 0, 0.2],
  ]) {
    assertThrows(
      () =>
        readChannels(
          { c: bad },
          "c",
          "here",
          isPositive,
          "has an element that is not a positive finite number",
        ),
      Error,
      "here.c: has an element that is not a positive finite number",
    );
  }
});

test("readOnly: returns the value when it is the only accepted value", () => {
  assertEquals(
    readOnly({ i: "bilinear" }, "i", "here", "bilinear", "because upstream does so"),
    "bilinear",
  );
});

test("readOnly: rejects a missing field", () => {
  assertThrows(
    () => readOnly({}, "interpolation", "pipelineConfig", "bilinear", "why"),
    Error,
    "pipelineConfig.interpolation: missing",
  );
});

test("readOnly: rejects a misspelling and an unsupported value with the same message, which contains the label and the reason", () => {
  for (const bad of ["bicubic", "BILINEAR", 2, null]) {
    assertThrows(
      () =>
        readOnly({ i: bad }, "i", "pipelineConfig", "bilinear", "upstream bakes it with bilinear"),
      Error,
      "pipelineConfig.i: this implementation supports only 'bilinear'" +
        ` (${String(bad)}) — upstream bakes it with bilinear`,
    );
  }
});
