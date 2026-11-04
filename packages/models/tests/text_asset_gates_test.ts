// Calls directly the **id checks shared across families** among the asset JSON range checks
// (`src/text/asset-gates.ts`). Needs neither a GPU nor real assets.
//
// Why these are pinned on their own: what these checks guard against is "no exception, but pointing
// at a different token / a different row" (silent wrong values), which from the families' e2e tests
// **stays green even when broken**. anima and irodori now use the same function, so the source of
// truth for the rules lives here as well.
//
// NOTE: the plain type checks of strings, numbers and objects (`asString` / `asNumber` /
// `asRecord`) underlie the id checks, so only the minimum is written to show that the id checks see
// only values that passed the type checks.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import {
  asId,
  asNumber,
  asRecord,
  asString,
  asVocabId,
  parseAddedTokens,
} from "../src/text/asset-gates.ts";

test("asRecord / asString / asNumber: reject values of the wrong type without reading them", () => {
  assertEquals(asRecord({ a: 1 }, "table"), { a: 1 });
  // `typeof null === "object"` does not pass (it fails at the check, not as `null.foo`).
  assertThrows(() => asRecord(null, "table"), Error, "table: not an object");
  assertThrows(() => asRecord("{}", "table"), Error, "table: not an object");

  assertEquals(asString("abc", "field"), "abc");
  assertThrows(() => asString(12, "field"), Error, "field: not a string");

  assertEquals(asNumber(12, "field"), 12);
  assertThrows(() => asNumber("12", "field"), Error, "field: not a number");
});

test("asId: rejects every value that would silently become a different id when written to `Int32Array`", () => {
  // `Int32Array` truncates a non-integer = it points at a different token (no exception without the
  // check).
  assertEquals(Int32Array.of(1.5)[0], 1);
  assertThrows(
    () => asId(1.5, "unkId"),
    Error,
    "unkId: token id is not an integer in 0..2147483647",
  );

  // Outside the i32 range it wraps (2147483648 → -2147483648).
  assertEquals(Int32Array.of(2147483648)[0], -2147483648);
  assertThrows(
    () => asId(2147483648, "eosId"),
    Error,
    "token id is not an integer in 0..2147483647",
  );

  assertThrows(() => asId(-1, "eosId"), Error, "token id is not an integer in 0..2147483647");
  assertThrows(() => asId("3", "eosId"), Error, "eosId: not a number");

  // Both ends are inside the accepted set (0 and the i32 maximum).
  assertEquals(asId(0, "unkId"), 0);
  assertEquals(asId(2147483647, "unkId"), 2147483647);
});

test("asVocabId: rejects an id at or above the vocabulary row count (before the gather reads another row)", () => {
  assertEquals(asVocabId(2, "unkId", 3), 2);
  assertThrows(
    () => asVocabId(3, "unkId", 3),
    Error,
    "unkId: token id 3 is not below the vocabulary row count 3",
  );
  // The rules of the underlying {@link asId} apply as well.
  assertThrows(() => asVocabId(1.5, "unkId", 3), Error, "not an integer");
});

test("parseAddedTokens: turns a list of pairs into a map and rejects duplicate keys instead of letting the last win", () => {
  const parsed = parseAddedTokens(
    [
      ["<s>", 1],
      ["</s>", 2],
    ],
    "added",
  );
  assertEquals(
    [...parsed],
    [
      ["<s>", 1],
      ["</s>", 2],
    ],
  );

  // Swallowing a duplicate key with `Map.set`'s last-one-wins would quietly change only the split
  // rules.
  assertThrows(
    () =>
      parseAddedTokens(
        [
          ["<s>", 1],
          ["<s>", 9],
        ],
        "added",
      ),
    Error,
    'added: duplicate key "<s>"',
  );

  assertThrows(() => parseAddedTokens({ "<s>": 1 }, "added"), Error, "added: not an array");
  assertThrows(() => parseAddedTokens([["<s>"]], "added"), Error, "added: not [string, id]");
  assertThrows(() => parseAddedTokens([[1, 1]], "added"), Error, "added: not a string");
  assertThrows(() => parseAddedTokens([["<s>", 1.5]], "added"), Error, "added: token id is");
});

test("parseAddedTokens: added tokens are not bounded by the vocabulary row count (they are numbered beyond the vocabulary table)", () => {
  // The real Qwen2 assets have this shape (a vocabulary of 151643 rows; added tokens
  // 151643..151668).
  const parsed = parseAddedTokens([["<|im_start|>", 151644]], "added");
  assertEquals(parsed.get("<|im_start|>"), 151644);
});
