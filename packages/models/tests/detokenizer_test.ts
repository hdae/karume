// A check of only the **failure classification** of the stop-string filter (`src/text/detokenizer.ts`). Needs
// neither a GPU nor real assets. The behaviour of incremental decoding itself is pinned by `gemma4_chat_test.ts`
// and not repeated here.
//
// It pins the classification of ADR 0107: the stop strings are a value the caller wrote, so both an empty string
// and a duplicate fail with `ModelInputError` (400 for an HTTP server). If the type differed per family, a host
// serving several families could only tell 400 apart by **reading the message text**.

import { assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { createStopStringFilter } from "../src/text/detokenizer.ts";

describe("createStopStringFilter: unacceptable stop strings fail as caused by the input", () => {
  it("empty string (always matches = a setting that can emit not a single character)", () => {
    assertThrows(
      () => createStopStringFilter(["END", ""]),
      ModelInputError,
      "stopStrings[1] is an empty string",
    );
  });

  it("duplicate (cannot mean more than writing the same condition twice)", () => {
    assertThrows(
      () => createStopStringFilter(["END", "END"]),
      ModelInputError,
      'stopStrings contains "END" twice',
    );
  });

  it("an accepted setting passes (the counterpart showing the check does not always fail)", () => {
    createStopStringFilter([]);
    createStopStringFilter(["END", "\n\n"]);
  });
});

describe("createStopStringFilter: a setting of the wrong type fails as caused by the input without implicit conversion", () => {
  /** A setting of the wrong type from JS / JSON (delivers a value the TS type check would not pass to the check). */
  const untyped = (value: unknown): readonly string[] => value as readonly string[];

  it("a string itself (not spread into an array of characters that stops at the first character)", () => {
    // Spreading "END" gives the 3 entries "E" / "N" / "D", which would stop "SENTENCE" at the first E.
    assertThrows(() => createStopStringFilter(untyped("END")), ModelInputError, "stopStrings");
  });

  it("an iterable that is not an array (Set)", () => {
    assertThrows(
      () => createStopStringFilter(untyped(new Set(["END"]))),
      ModelInputError,
      "stopStrings",
    );
  });

  it("an element that is not a string (does not stop at another position through indexOf's implicit conversion)", () => {
    // 123 would be treated as "123" and stop "x123y" at "x".
    for (const element of [123, null, undefined]) {
      assertThrows(
        () => createStopStringFilter(untyped(["END", element])),
        ModelInputError,
        "stopStrings[1]",
        String(element),
      );
    }
  });

  it("a hole in a sparse array (not silently skipped and dropped as forEach does)", () => {
    // Index 0 of [, "END"] is a hole. Skipping holes would accept it as a setting of "END" alone.
    const sparse: (string | undefined)[] = [];
    sparse[1] = "END";
    assertThrows(
      () => createStopStringFilter(untyped(sparse)),
      ModelInputError,
      "stopStrings[0] undefined",
    );
  });
});
