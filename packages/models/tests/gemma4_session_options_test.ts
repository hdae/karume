import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { SessionSpec } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import { DEFAULT_SUBMIT_POLICY } from "@karume/runtime";
import {
  assertGemmaSessionOverrides,
  GEMMA_MAX_CHUNK_SIZE,
  gemmaSubmitPolicy,
  resolveGemmaSessionOptions as resolve,
} from "../src/gemma/session-options.ts";

/** Returns the message if it threw as caused by the input, or `undefined` if it passed (for comparing the verdicts of the 2 paths). */
const inputViolationOf = (run: () => void): string | undefined => {
  try {
    run();
    return undefined;
  } catch (error) {
    assert(error instanceof ModelInputError);
    return error.message;
  }
};

describe("Gemma's quant execution settings", () => {
  /** The shipped `i4-fast` declaration of the q4_0 line. */
  const fast = { numerics: { level: "reorder" } } as const;
  /** The mobile QAT line's `i4-fast`: the same level with the packed activations. */
  const packedFast = { ...fast, packedStaticQuantize: true } as const;
  /** The packed message the runtime gives (`sessionOptionsViolation`). */
  const packedNeedsReorder =
    'packedStaticQuantize needs linear at the level "reorder" (numerics) with linearCompute "f32"';
  it("distinguishes unset, the quant declaration and an explicit override", () => {
    assertEquals(resolve({}, {}, "test"), {});
    assertEquals(resolve(fast, {}, "test"), fast);
    assertEquals(resolve(fast, { numerics: undefined }, "test"), fast);
    assertEquals(resolve(fast, { numerics: { ops: { attention: "exact" } } }, "test"), {
      numerics: { level: "reorder", ops: { attention: "exact" } },
    });
    const reference = { numerics: { level: "exact" } } as const;
    assertEquals(resolve(fast, reference, "test"), reference);
    assertEquals(resolve(reference, fast, "test"), fast);
  });
  it("does not hide an unsupported declaration behind an override, and rejects invalid combinations of the effective settings", () => {
    assertThrows(
      () => resolve({ linearCompute: "f32" }, {}, "test"),
      Error,
      "session.linearCompute",
    );
    assertThrows(
      () => resolve(packedFast, { numerics: { ops: { linear: "exact" } } }, "test"),
      ModelInputError,
      packedNeedsReorder,
    );
    assertEquals(
      resolve(
        packedFast,
        {
          numerics: { ops: { linear: "exact" } },
          packedStaticQuantize: false,
        },
        "test",
      ),
      { numerics: { level: "reorder", ops: { linear: "exact" } }, packedStaticQuantize: false },
    );
    assertThrows(
      () => resolve({}, { packedStaticQuantize: true }, "test"),
      ModelInputError,
      packedNeedsReorder,
    );
    assertEquals(
      resolve(
        { packedStaticQuantize: true },
        {
          numerics: { ops: { linear: "reorder" } },
        },
        "test",
      ),
      { numerics: { ops: { linear: "reorder" } }, packedStaticQuantize: true },
    );
  });
  it("packedStaticQuantize comes from the quant declaration, an explicit false wins, and anything but linear at reorder is rejected", () => {
    // ADR 0105 addendum 2 (promotion into the vocabulary) — the fields of the shipped mobile i4-fast pass through from the declaration.
    assertEquals(resolve(packedFast, {}, "test"), packedFast);
    assertEquals(resolve(packedFast, { packedStaticQuantize: false }, "test"), {
      ...packedFast,
      packedStaticQuantize: false,
    });
    assertEquals(resolve(fast, { packedStaticQuantize: true }, "test"), packedFast);
    // A declaration without linear at reorder is rejected even when it comes from the quant (i4-fast declares the level).
    assertThrows(
      () => resolve({ packedStaticQuantize: true }, {}, "test"),
      Error,
      packedNeedsReorder,
    );
    assertThrows(
      () => resolve({}, { packedStaticQuantize: true }, "test"),
      ModelInputError,
      packedNeedsReorder,
    );
    assertThrows(
      () =>
        resolve(
          fast,
          {
            numerics: { level: "exact" },
            packedStaticQuantize: true,
          },
          "test",
        ),
      ModelInputError,
      packedNeedsReorder,
    );
  });
  it("takes integerGemv explicitly on top of a preset that declares the packed activations", () => {
    // The QAT i4-fast form: the preset declares the packed activations, the caller adds integerGemv.
    assertEquals(resolve(packedFast, { integerGemv: true }, "test"), {
      ...packedFast,
      integerGemv: true,
    });
    assertThrows(
      () => resolve(fast, { integerGemv: true }, "test"),
      ModelInputError,
      "integerGemv requires packedStaticQuantize: true",
    );
    // Both entry points decide alike (fromAssets has no preset to declare the packed activations).
    assertThrows(
      () => assertGemmaSessionOverrides({ integerGemv: true }, "test"),
      ModelInputError,
      "integerGemv requires packedStaticQuantize: true",
    );
    assertGemmaSessionOverrides(
      {
        numerics: { ops: { linear: "reorder" } },
        packedStaticQuantize: true,
        integerGemv: true,
      },
      "test",
    );
  });
  it("numerics resolves by the same rules as the other fields and creates no field without a declaration", () => {
    // There is no family code default (ADR 0104 addendum 2026-09-26). No field = every op kind at `exact` — the `i4`
    // preset with an empty `session` takes this shape.
    assertEquals(resolve({}, {}, "test"), {});
    assertEquals(resolve({ numerics: { ops: { linear: "reorder" } } }, {}, "test"), {
      numerics: { ops: { linear: "reorder" } },
    });
    // The same declaration as the q4_0 line's i4-fast passes through from the declaration.
    assertEquals(resolve(fast, {}, "test"), fast);
    // An explicit value wins over the declaration per field (the way back to the reference path for attention).
    assertEquals(resolve(fast, { numerics: { ops: { attention: "exact" } } }, "test"), {
      numerics: { level: "reorder", ops: { attention: "exact" } },
    });
    assertEquals(resolve({}, { numerics: { ops: { attention: "reorder" } } }, "test"), {
      numerics: { ops: { attention: "reorder" } },
    });
    // The surface without a declaration (fromAssets) accepts both levels, for the model and per op kind.
    for (const level of ["exact", "reorder"] as const) {
      assertGemmaSessionOverrides({ numerics: { level } }, "test");
      assertGemmaSessionOverrides({ numerics: { ops: { attention: level } } }, "test");
    }
  });
  it("every key of SessionSpec appears either in the allow table or on the rejection path", () => {
    // The hub's SESSION_KEYS and the models' WRITERS are exhaustive tables over `Required<SessionSpec>`, so a new knob
    // fails the type check. Gemma accepts 3 of the keys (the numeric requirement and the 2 int8 GEMV booleans), and a
    // missed update of that set shows only as a runtime rejection.
    // Here, from a spec with every key filled (a new key makes this declaration fail the type check), each key is
    // confirmed to pass / fail one at a time.
    const full: Required<SessionSpec> = {
      numerics: { level: "exact" },
      linearCompute: "f16",
      convCompute: "a8",
      attentionCompute: "f16",
      attentionScoreStorage: "f16",
      packedStaticQuantize: false,
      integerGemv: false,
    };
    const accepted = ["numerics", "packedStaticQuantize", "integerGemv"];
    const rejected: string[] = [];
    for (const key of Object.keys(full) as (keyof typeof full)[]) {
      const spec = { [key]: full[key] } satisfies SessionSpec;
      if (accepted.includes(key)) {
        assertEquals(resolve(spec, {}, "test"), spec);
        continue;
      }
      assertThrows(() => resolve(spec, {}, "test"), Error, `session.${key} is not supported`);
      rejected.push(key);
    }
    assertEquals([...accepted, ...rejected].sort(), Object.keys(full).sort());
  });
  it("does not replace an invalid explicit value with the quant's value, and never calls the value's string conversion", () => {
    let conversions = 0;
    const object = {
      toString: () => {
        conversions++;
        return "parallel";
      },
    };
    // The expected messages are pinned too so that a "swapped reason" — failing in the combination check instead of
    // the type check — does not pass (resolve throws for 4 different reasons).
    const invalid = [
      ["packedStaticQuantize", "packedStaticQuantize must be a boolean"],
      ["integerGemv", "integerGemv must be a boolean"],
    ] as const;
    for (const [key, message] of invalid) {
      for (const value of [null, 0, 1, [], object]) {
        const overrides = {};
        // The options real calls pass have only enumerable properties, so the check uses the same shape.
        Object.defineProperty(overrides, key, { value, enumerable: true });
        assertThrows(() => resolve(fast, overrides, "test"), ModelInputError, message);
      }
    }
    // `numerics` is an object, so only its non-object values are invalid as a whole (its fields have their own cases
    // in session_options_test.ts).
    for (const value of [null, 0, 1, []]) {
      const overrides = {};
      Object.defineProperty(overrides, "numerics", { value, enumerable: true });
      assertThrows(
        () => resolve(fast, overrides, "test"),
        ModelInputError,
        "options.numerics must be an object",
      );
    }
    assertEquals(conversions, 0);
  });
  it("a rejection that arises from the manifest declaration alone is not classified as caused by the input", () => {
    // ADR 0107 decision 2: if what fails is the distribution manifest's declaration, fixing the request does not fix
    // it (an asset inconsistency = the equivalent of 500). With empty overrides, the same condition yields a plain Error.
    for (const quant of [{ linearCompute: "f32" }, { packedStaticQuantize: true }] as const) {
      const thrown = assertThrows(() => resolve(quant, {}, "test"));
      assert(thrown instanceof Error);
      assert(!(thrown instanceof ModelInputError));
    }
  });
  it("the same condition is caused by the input once an override is involved", () => {
    // The other side. If a declaration that passes on the quant alone fails once combined with an explicit override, the remedy is "fix the setting".
    const exactLinear = { numerics: { ops: { linear: "exact" } } } as const;
    assertEquals(resolve(exactLinear, {}, "test"), exactLinear);
    assertThrows(
      () => resolve(exactLinear, { packedStaticQuantize: true }, "test"),
      ModelInputError,
      packedNeedsReorder,
    );
    const packedReorder = {
      packedStaticQuantize: true,
      numerics: { ops: { linear: "reorder" } },
    } as const;
    assertEquals(resolve(packedReorder, {}, "test"), packedReorder);
    assertThrows(
      () => resolve(packedReorder, exactLinear, "test"),
      ModelInputError,
      packedNeedsReorder,
    );
  });
});

describe("Gemma's quant execution settings (the check of explicit settings only)", () => {
  it("rejects with the same ranges, types and combinations on a surface without a quant declaration", () => {
    // fromAssets has no manifest, so the verdict is the same as when compared with the default {}.
    assertGemmaSessionOverrides({}, "test");
    assertGemmaSessionOverrides({ numerics: { level: "reorder" } }, "test");
    assertThrows(
      () => assertGemmaSessionOverrides({ packedStaticQuantize: true }, "test"),
      ModelInputError,
      'packedStaticQuantize needs linear at the level "reorder" (numerics) with linearCompute "f32"',
    );
    const bogus = {};
    Object.defineProperty(bogus, "packedStaticQuantize", { value: 1, enumerable: true });
    assertThrows(
      () => assertGemmaSessionOverrides(bogus, "test"),
      ModelInputError,
      "packedStaticQuantize must be a boolean",
    );
  });
  it("agrees with the verdict resolve gives on the same explicit settings", () => {
    // Pins that the 2 entry points go through the same check by the agreement of their verdicts (no loosening of only one side).
    const explicit = [
      {},
      { numerics: { ops: { linear: "reorder" } } },
      { numerics: { ops: { attention: "reorder" } } },
      { packedStaticQuantize: true },
      { packedStaticQuantize: true, numerics: { ops: { linear: "exact" } } },
      { packedStaticQuantize: true, numerics: { ops: { linear: "reorder" } } },
    ] as const;
    for (const overrides of explicit) {
      const viaGate = inputViolationOf(() => assertGemmaSessionOverrides(overrides, "test"));
      const viaResolve = inputViolationOf(() => resolve({}, overrides, "test"));
      assertEquals(viaGate, viaResolve);
    }
  });
});

describe("gemmaSubmitPolicy (perf-ledger H-40)", () => {
  it("caps submits at 128 dispatches and otherwise keeps the runtime default when the caller passes none", () => {
    assertEquals(GEMMA_MAX_CHUNK_SIZE, 128);
    assertEquals(gemmaSubmitPolicy(undefined), { ...DEFAULT_SUBMIT_POLICY, maxChunkSize: 128 });
  });

  it("passes the caller's own policy through unchanged", () => {
    const own = { timeBudgetMs: 50, initialChunkSize: 8, minChunkSize: 2, maxChunkSize: 512 };
    assertEquals(gemmaSubmitPolicy(own), own);
  });
});
