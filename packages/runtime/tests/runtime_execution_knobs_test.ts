// The check of the execution-knob accepted-set test ({@link assertExecutionKnobs}) — a pure
// function that touches no GPU, so it sits where regressions can be caught even without an adapter.
// Of the GPU tests through `createSessionFromContainer`, only one is kept ("it is also rejected in
// a real build"); the exhaustive sweep of values lives here.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { assertExecutionKnobs } from "../src/runtime/session-options-check.ts";
import { ExecutionError } from "../src/runtime/plan.ts";

/**
 * Reproduces **callers from outside the types** (JS consumers, old builds that kept a pre-rename
 * spelling). TS types guard none of this path, so the accepted-set check can only be exercised
 * here.
 */
const jsCallerAssert = assertExecutionKnobs as unknown as (...knobs: readonly unknown[]) => void;

/**
 * Spellings inside the accepted set (this order is {@link assertExecutionKnobs}'s argument order).
 */
const VALID: readonly [string, string, string, string, string, string] = [
  "f32",
  "f32",
  "f32",
  "f32",
  "parallel-fused",
  "parallel",
];
const NAMES: readonly [string, string, string, string, string, string] = [
  "linearCompute",
  "convCompute",
  "attentionCompute",
  "attentionScoreStorage",
  "stateAttentionReduce",
  "linearGemvReduce",
];

test("spellings inside the accepted set pass for all 6 knobs (proof that the tests below are not 'fails whatever is passed')", () => {
  assertExecutionKnobs("f32", "a8", "f32", "f32", "parallel-fused", "parallel");
});

test("non-string execution knobs fail before the accepted set is looked up, and the diagnostic does not call the caller's conversions", () => {
  for (const [index, name] of NAMES.entries()) {
    let conversions = 0;
    // An object that **can convert** to a spelling inside the accepted set. If `Object.hasOwn`'s
    // key conversion or the diagnostic's stringification ran, it would pass, so the call count
    // itself is checked.
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return VALID[index];
      },
      toJSON(): string {
        conversions++;
        throw new Error("the diagnostic does not call the caller's conversions");
      },
    };
    for (const value of [[VALID[index]], object, false, 0, 1n, Symbol(VALID[index])]) {
      const knobs: unknown[] = [...VALID];
      knobs[index] = value;
      const error = assertThrows(() => jsCallerAssert(...knobs), ExecutionError, name);
      // Only the type name is reported (the return value of the caller's `toString` is not
      // included).
      assertEquals(error.message.includes(`${name}: ${typeof value}`), true, error.message);
      assertEquals(error.message.includes("outside the accepted set: 1"), true, error.message);
    }
    assertEquals(conversions, 0, `${name}: the caller's conversion was called`);
  }
});

test("spellings outside the union are all listed and fail at once", () => {
  // `"i8a8"` is the spelling renamed to `"a8"` in 0.5.0 (ADR 0074 decision 3, no compatibility
  // shim).
  const error = assertThrows(
    () => jsCallerAssert("i8a8", "f32", "f32", "s16", "parallel-fused", "parallel"),
    ExecutionError,
    "outside the accepted set: 2",
  );
  assertEquals(error.message.includes(`linearCompute: "i8a8"`), true, error.message);
  assertEquals(error.message.includes(`attentionScoreStorage: "s16"`), true, error.message);
  assertEquals(error.message.includes("'f32' / 'a8' / 'f16'"), true, error.message);
});
