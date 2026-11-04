// Behaviour tests of the cross-family input-caused errors (ADR
// [0107](../../../docs/decisions/0107-model-input-error.md)). Needs neither a GPU nor real assets.
//
// Two things are pinned:
//
// - **Branching with `instanceof` works** — it is the only way a host tells 400 from 500, so the two subclasses
//   passing as the parent is itself the spec (re-pointing `extends` would not be caught by the type check).
// - **The accepted set has a single owner** — the boundaries of `assertAcceptableSeed` (0 and
//   `Number.MAX_SAFE_INTEGER` pass, anything outside fails) are pinned by value.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { assertAcceptableSeed } from "../src/request-gates.ts";
import * as sbv2 from "../src/sbv2/errors.ts";
import { GenerationCapacityError } from "../src/generation/sequence.ts";

describe("ModelInputError", () => {
  it("name is spelled as is and readable from logs and err.name branches", () => {
    assertEquals(new ModelInputError("out of range").name, "ModelInputError");
  });

  it("keeps message as is", () => {
    const error = new ModelInputError("seed -1 is not a non-negative safe integer");
    assertEquals(error.message, "seed -1 is not a non-negative safe integer");
  });

  it("passes cause through (the cause can be traced after rewrapping)", () => {
    const cause = new Error("the original failure");
    assertEquals(new ModelInputError("not acceptable", { cause }).cause, cause);
  });

  it("is a subclass of Error (does not break existing catch branches)", () => {
    assert(new ModelInputError("not acceptable") instanceof Error);
  });
});

describe("family input-caused errors as seen from the host", () => {
  it("sbv2.InputError is caught as ModelInputError (name is qualified with the family name)", () => {
    const error = new sbv2.InputError("the style name is not in the distribution");
    assert(error instanceof ModelInputError);
    assertEquals(error.name, "sbv2.InputError");
  });

  it("GenerationCapacityError is caught as ModelInputError (the truncation fields are kept)", () => {
    const error = new GenerationCapacityError("the conversation does not fit the capacity", {
      constraint: "capacity",
      pastLength: 100,
      promptLength: 40,
      requestedNewTokens: 64,
      limit: 128,
      maxNewTokens: -11,
    });
    assert(error instanceof ModelInputError);
    assertEquals(error.name, "GenerationCapacityError");
    assertEquals([error.constraint, error.maxNewTokens], ["capacity", -11]);
  });
});

describe("assertAcceptableSeed (the seed accepted set shared by 3 families)", () => {
  it("passes non-negative safe integers (both ends included)", () => {
    assertAcceptableSeed(0);
    assertAcceptableSeed(Number.MAX_SAFE_INTEGER);
  });

  it("rejects negative, fractional, NaN and beyond-safe-integer values with ModelInputError", () => {
    for (const seed of [-1, 1.5, Number.NaN, 2 ** 53]) {
      assertThrows(
        () => assertAcceptableSeed(seed),
        ModelInputError,
        "is not a non-negative safe integer",
      );
    }
  });
});
