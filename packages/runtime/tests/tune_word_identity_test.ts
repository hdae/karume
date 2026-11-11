// The u32 identity comparison with a NaN policy (src/tune/word-identity.ts) — the one comparison
// loop of the kernel identity checks and the fused-vs-unfused checks. No GPU.

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { wordIdentity } from "../src/tune/word-identity.ts";

const words = (...values: number[]): Uint32Array<ArrayBuffer> => Uint32Array.from(values);

const QUIET_NAN = 0x7fc00000;
const PAYLOAD_NAN = 0x7fc01234;
const NEGATIVE_NAN = 0xff801234;
const SIGNALLING_NAN = 0x7f800001;
const INFINITY = 0x7f800000;
const NEGATIVE_INFINITY = 0xff800000;
const ONE = 0x3f800000;
const NEGATIVE_ZERO = 0x80000000;

describe("wordIdentity", () => {
  describe("with equal words", () => {
    it("compares every word of expected and reports no failure, under either policy", () => {
      const same = words(0, NEGATIVE_ZERO, 1, ONE, INFINITY, PAYLOAD_NAN);
      for (const nan of ["payload", "class"] as const) {
        assertEquals(wordIdentity(same, words(...same), nan), { words: 6, failures: 0 }, nan);
      }
    });
  });

  describe("under the payload policy", () => {
    it("fails a NaN of another payload or sign, like any other word", () => {
      assertEquals(
        wordIdentity(
          words(ONE, QUIET_NAN, NEGATIVE_NAN),
          words(ONE, PAYLOAD_NAN, PAYLOAD_NAN),
          "payload",
        ),
        {
          words: 3,
          failures: 2,
          first: { index: 1, expected: PAYLOAD_NAN, actual: QUIET_NAN },
        },
      );
    });
  });

  describe("under the class policy", () => {
    it("accepts any NaN, of any payload and sign, where expected holds a NaN", () => {
      assertEquals(
        wordIdentity(
          words(QUIET_NAN, NEGATIVE_NAN, SIGNALLING_NAN, PAYLOAD_NAN),
          words(PAYLOAD_NAN, QUIET_NAN, NEGATIVE_NAN, SIGNALLING_NAN),
          "class",
        ),
        { words: 4, failures: 0 },
      );
    });

    it("fails a non-NaN word where expected holds a NaN", () => {
      for (const actual of [ONE, INFINITY, NEGATIVE_INFINITY, 0]) {
        assertEquals(wordIdentity(words(actual), words(PAYLOAD_NAN), "class"), {
          words: 1,
          failures: 1,
          first: { index: 0, expected: PAYLOAD_NAN, actual },
        });
      }
    });

    it("fails an actual NaN against a finite or infinite expected word", () => {
      for (const expected of [ONE, INFINITY, NEGATIVE_INFINITY, 0]) {
        assertEquals(wordIdentity(words(QUIET_NAN), words(expected), "class"), {
          words: 1,
          failures: 1,
          first: { index: 0, expected, actual: QUIET_NAN },
        });
      }
    });

    it("keeps every other word exact: -0 against +0 and infinities of the other sign fail", () => {
      assertEquals(
        wordIdentity(words(0, NEGATIVE_INFINITY), words(NEGATIVE_ZERO, INFINITY), "class"),
        { words: 2, failures: 2, first: { index: 0, expected: NEGATIVE_ZERO, actual: 0 } },
      );
    });
  });

  describe("with failing words", () => {
    it("counts every failing word and reports the first one", () => {
      assertEquals(wordIdentity(words(1, 9, 3, 9, 5), words(1, 2, 3, 4, 5), "payload"), {
        words: 5,
        failures: 2,
        first: { index: 1, expected: 2, actual: 9 },
      });
    });
  });

  describe("with arrays of different lengths", () => {
    it("counts each missing word of a shorter actual as a failure, the first past its end", () => {
      assertEquals(wordIdentity(words(1, 2), words(1, 2, 3, 4), "payload"), {
        words: 4,
        failures: 2,
        first: { index: 2, expected: 3, actual: 0 },
      });
    });

    it("counts each extra word of a longer actual as a failure", () => {
      assertEquals(wordIdentity(words(1, 2, 7), words(1, 2), "class"), {
        words: 2,
        failures: 1,
        first: { index: 2, expected: 0, actual: 7 },
      });
    });

    it("reports a failing word of the common range first, and still counts the missing words", () => {
      assertEquals(wordIdentity(words(1, 8), words(1, 2, 3), "payload"), {
        words: 3,
        failures: 2,
        first: { index: 1, expected: 2, actual: 8 },
      });
    });
  });
});
