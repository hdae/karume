// The mapping from the manifest's `session` (manifest-owned vocabulary) to runtime
// `SessionOptions`. It is the spelling contract of ADR 0038 §3 itself, and a gap is a **silent
// degradation** (runtime silently ignores unknown keys). All eight families use the same function,
// so its checks are gathered here in one place too (originally they were split into two, in the
// anima / sbv2 pipeline tests, and the other five families had no check that called the mapping
// directly).

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import type { SessionSpec } from "@karume/hub";
import { sessionOptionsViolation } from "@karume/runtime";
import { numericLevels } from "../../runtime/src/runtime/numerics.ts";
import { ANIMA_SESSION_POLICY } from "../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../src/depth-anything/pipeline.ts";
import { EMBEDDINGGEMMA_SESSION_POLICY } from "../src/embeddinggemma/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { GEMMA_SESSION_POLICY } from "../src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../src/irodori/admission.ts";
import { SBV2_SESSION_POLICY } from "../src/sbv2/pipeline.ts";
import {
  assertSessionOverrides,
  type FamilySessionPolicy,
  resolveSessionOptions,
  type SessionOverrides,
  toSessionOptions,
} from "../src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../src/siglip2/pipeline.ts";
import { UMT5_ENCODER_SESSION_POLICY } from "../src/umt5/encoder.ts";
import { WAN_SESSION_POLICY } from "../src/wan/family.ts";

test("toSessionOptions: maps the declared keys one by one (an unspecified key creates no field at all)", () => {
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ numerics: { level: "reorder" } }), {
    numerics: { level: "reorder" },
  });
  assertEquals(toSessionOptions({ numerics: { ops: { linear: "exact" } } }), {
    numerics: { ops: { linear: "exact" } },
  });
  assertEquals(toSessionOptions({ linearCompute: "a8" }), { linearCompute: "a8" });
  assertEquals(toSessionOptions({ attentionCompute: "f16" }), { attentionCompute: "f16" });
  assertEquals(toSessionOptions({ attentionScoreStorage: "f16" }), {
    attentionScoreStorage: "f16",
  });
  // All three keys of the distribution's default quant (f16+dit8-a8-attn8-s16) must pass. Dropping
  // even one is the silent degradation "s16 in name only".
  assertEquals(
    toSessionOptions({
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    }),
    {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    },
  );
});

test("toSessionOptions: does not map a knob the manifest side does not have (submitPolicy)", () => {
  // `SessionOptions.submitPolicy` is the TDR budget = **host policy**, so distributors do not get
  // to write it (reason ③ of ADR 0038 §3). Rewriting the mapping as a pass-through spread could let
  // it slip through here.
  const mapped = toSessionOptions({ linearCompute: "a8" }) as Record<string, unknown>;
  assertEquals(Object.hasOwn(mapped, "submitPolicy"), false);
  assertEquals(Object.keys(mapped), ["linearCompute"]);
});

test("toSessionOptions: maps every key of SessionSpec (detects a key left behind when one is added)", () => {
  // The mapping's coverage is pinned by type through `WRITERS` in `src/session/options.ts` (the
  // exhaustive table of `Required<SessionSpec>`), so a new key is a **compile error**. This is the
  // counterpart that checks at run time that this type check is alive (a change that loosens the
  // type would pass the type check) — it passes a spec with every key filled and confirms that the
  // resulting key set equals the input's.
  const full: Required<SessionSpec> = {
    numerics: { level: "reorder", ops: { attention: "exact" } },
    linearCompute: "f16",
    convCompute: "a8",
    attentionCompute: "f16",
    attentionScoreStorage: "f16",
    packedStaticQuantize: false,
    integerGemv: false,
  };
  const mapped = toSessionOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(mapped).sort(), Object.keys(full).sort());
});

test("toSessionOptions: keeps explicit false and true for the int8 GEMV booleans", () => {
  for (const value of [false, true]) {
    const spec = {
      packedStaticQuantize: value,
      integerGemv: value,
    };
    assertEquals(toSessionOptions(spec), spec);
  }
});

// ---- Composition (explicit > quant declaration > runtime default) — ADR 0058 addendum 2026-09-26
// ----

/**
 * A policy table that accepts all 7 keys (to look at the rules themselves, apart from any family's
 * table).
 */
const ALL: FamilySessionPolicy = {
  numerics: true,
  linearCompute: true,
  convCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  packedStaticQuantize: true,
  integerGemv: true,
};

/** A table that accepts only the linear execution form (the same shape as irodori / sbv2). */
const LINEAR_ONLY: FamilySessionPolicy = {
  ...ALL,
  numerics: false,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  packedStaticQuantize: false,
  integerGemv: false,
};

/** A spec with every key filled (a new key fails this declaration's type check). */
const FULL: Required<SessionSpec> = {
  numerics: { level: "reorder" },
  linearCompute: "f32",
  convCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  packedStaticQuantize: false,
  integerGemv: false,
};

/**
 * Reproduces an explicit setting from a caller outside the types (a JS consumer) — TS types do not
 * guard this path.
 */
const overridesWith = (key: string, value: unknown): SessionOverrides => {
  const overrides: SessionOverrides = {};
  Object.defineProperty(overrides, key, { value, enumerable: true });
  return overrides;
};

describe("resolveSessionOptions: precedence per key", () => {
  it("uses the explicit value if present, otherwise the declaration, and creates no field if neither exists", () => {
    assertEquals(resolveSessionOptions(ALL, {}, {}, "test"), {});
    assertEquals(resolveSessionOptions(ALL, { linearCompute: "a8" }, {}, "test"), {
      linearCompute: "a8",
    });
    assertEquals(
      resolveSessionOptions(
        ALL,
        { linearCompute: "a8", attentionCompute: "a8" },
        {
          linearCompute: "f16",
        },
        "test",
      ),
      { linearCompute: "f16", attentionCompute: "a8" },
    );
    assertEquals(resolveSessionOptions(ALL, {}, { attentionScoreStorage: "f16" }, "test"), {
      attentionScoreStorage: "f16",
    });
  });

  it("an explicit false wins over a declared true, and undefined means 'not specified' so the declaration stays", () => {
    const declared = {
      numerics: { ops: { linear: "reorder" } },
      packedStaticQuantize: true,
    } as const;
    assertEquals(resolveSessionOptions(ALL, declared, { packedStaticQuantize: false }, "test"), {
      numerics: { ops: { linear: "reorder" } },
      packedStaticQuantize: false,
    });
    assertEquals(
      resolveSessionOptions(ALL, declared, { packedStaticQuantize: undefined }, "test"),
      declared,
    );
  });

  it("does not reject values that depend on GPU capabilities (f16 compute) at this layer", () => {
    assertEquals(resolveSessionOptions(ALL, {}, { linearCompute: "f16" }, "test"), {
      linearCompute: "f16",
    });
  });
});

describe("resolveSessionOptions: numerics composes per field", () => {
  it("an explicit level wins over the declared level", () => {
    assertEquals(
      resolveSessionOptions(
        ALL,
        { numerics: { level: "reorder" } },
        { numerics: { level: "exact" } },
        "test",
      ),
      { numerics: { level: "exact" } },
    );
  });

  it("an explicit op kind wins over the same declared op kind and keeps the other declared ones", () => {
    assertEquals(
      resolveSessionOptions(
        ALL,
        { numerics: { ops: { linear: "reorder", attention: "reorder" } } },
        { numerics: { ops: { attention: "exact" } } },
        "test",
      ),
      { numerics: { ops: { linear: "reorder", attention: "exact" } } },
    );
  });

  it("an explicit { ops: { attention: 'exact' } } keeps a declared level 'reorder' for linear", () => {
    const effective = resolveSessionOptions(
      ALL,
      { numerics: { level: "reorder" } },
      { numerics: { ops: { attention: "exact" } } },
      "test",
    );
    assertEquals(effective, { numerics: { level: "reorder", ops: { attention: "exact" } } });
    assertEquals(numericLevels(effective.numerics), {
      linear: "reorder",
      attention: "exact",
      rms_norm: "reorder",
    });
  });

  it("an explicit level keeps the declared op kinds (the narrower scope still wins)", () => {
    const effective = resolveSessionOptions(
      ALL,
      { numerics: { ops: { attention: "exact" } } },
      { numerics: { level: "reorder" } },
      "test",
    );
    assertEquals(effective, { numerics: { level: "reorder", ops: { attention: "exact" } } });
    assertEquals(numericLevels(effective.numerics), {
      linear: "reorder",
      attention: "exact",
      rms_norm: "reorder",
    });
  });

  it("with one side only, that side is the value as given", () => {
    assertEquals(resolveSessionOptions(ALL, { numerics: { level: "reorder" } }, {}, "test"), {
      numerics: { level: "reorder" },
    });
    assertEquals(
      resolveSessionOptions(ALL, {}, { numerics: { ops: { linear: "reorder" } } }, "test"),
      { numerics: { ops: { linear: "reorder" } } },
    );
  });

  it("an explicit null or non-object numerics is a ModelInputError, not a fall-back to the declaration", () => {
    for (const value of [null, 0, "reorder", true, []]) {
      assertThrows(
        () => resolveSessionOptions(ALL, FULL, overridesWith("numerics", value), "test"),
        ModelInputError,
        "test: options.numerics must be an object",
      );
    }
  });

  for (const [numerics, message] of [
    [{ level: "parallel" }, "options.numerics.level"],
    [{ level: null }, "options.numerics.level"],
    [{ ops: null }, "options.numerics.ops must be an object"],
    [{ ops: "exact" }, "options.numerics.ops must be an object"],
    [{ ops: [] }, "options.numerics.ops must be an object"],
    [{ ops: { attention: "parallel" } }, "options.numerics.ops.attention"],
    [{ ops: { gemv: "reorder" } }, "options.numerics.ops.gemv is not an op kind"],
    [{ levle: "exact" }, "options.numerics.levle is not a field"],
  ] as const) {
    it(`an explicit ${JSON.stringify(numerics)} is a ModelInputError, not a fall-back to the declaration`, () => {
      assertThrows(
        () => resolveSessionOptions(ALL, FULL, overridesWith("numerics", numerics), "test"),
        ModelInputError,
        `test: ${message}`,
      );
    });
  }
});

describe("resolveSessionOptions: settings that cannot be accepted", () => {
  it("an invalid explicit value does not fall back to the declaration, throws ModelInputError and never converts the value", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return "f32";
      },
    };
    // `numerics` has its own cases above (its diagnostic is the runtime's, per field).
    const messages: Readonly<Record<Exclude<keyof SessionSpec, "numerics">, string>> = {
      linearCompute: "linearCompute is invalid",
      convCompute: "convCompute is invalid",
      attentionCompute: "attentionCompute is invalid",
      attentionScoreStorage: "attentionScoreStorage is invalid",
      packedStaticQuantize: "packedStaticQuantize must be a boolean",
      integerGemv: "integerGemv must be a boolean",
    };
    for (const [key, message] of Object.entries(messages)) {
      for (const value of [null, 0, "bogus", [], object]) {
        // The declaration is filled with accepted values — a regression that falls back to it with
        // `??` would pass here.
        assertThrows(
          () => resolveSessionOptions(ALL, FULL, overridesWith(key, value), "test"),
          ModelInputError,
          `test: ${message}`,
        );
      }
    }
    assertEquals(conversions, 0);
  });

  it("a declared key the policy table rejects is a plain Error before fetching (even when hidden by an explicit value)", () => {
    for (const key of ["attentionCompute", "numerics", "packedStaticQuantize"] as const) {
      const declared = { [key]: FULL[key] } satisfies SessionSpec;
      const error = assertThrows(
        () => resolveSessionOptions(LINEAR_ONLY, declared, {}, "test"),
        Error,
        `test: session.${key} is not supported`,
      );
      assert(
        !(error instanceof ModelInputError),
        "an asset inconsistency was thrown as an input-error type",
      );
    }
    // An explicit setting does not hide a faulty declaration (a declaration that is not accepted is
    // an asset inconsistency whether or not it is overridden).
    assertThrows(
      () =>
        resolveSessionOptions(
          LINEAR_ONLY,
          { attentionCompute: "a8" },
          {
            linearCompute: "f32",
          },
          "test",
        ),
      Error,
      "session.attentionCompute is not supported",
    );
  });

  it("an explicit key the policy table rejects is a ModelInputError (not silently dropped)", () => {
    assertThrows(
      () => resolveSessionOptions(LINEAR_ONLY, {}, { attentionCompute: "f16" }, "test"),
      ModelInputError,
      "test: attentionCompute cannot be specified for this family",
    );
    // undefined means "not specified", so it passes whatever the policy table says.
    assertEquals(
      resolveSessionOptions(LINEAR_ONLY, {}, { attentionCompute: undefined }, "test"),
      {},
    );
  });

  it("a combination violation uses runtime's verdict and message, with the thrown type chosen by origin", () => {
    // Arising from the declaration alone = an asset inconsistency (the caller cannot fix it by
    // changing settings) → plain Error.
    const declaredOnly = { attentionCompute: "f16", attentionScoreStorage: "f16" } as const;
    const fromAsset = assertThrows(() => resolveSessionOptions(ALL, declaredOnly, {}, "test"));
    assert(fromAsset instanceof Error && !(fromAsset instanceof ModelInputError));
    assertEquals(fromAsset.message, `test: ${sessionOptionsViolation(declaredOnly)}`);
    // Arising only with an explicit setting involved = an input error → ModelInputError (the
    // message comes from the same single runtime function).
    const fromInput = assertThrows(
      () =>
        resolveSessionOptions(
          ALL,
          { attentionScoreStorage: "f16" },
          {
            attentionCompute: "f16",
          },
          "test",
        ),
      ModelInputError,
    );
    assertEquals(fromInput.message, fromAsset.message);
  });

  it("passes when an explicit setting resolves a violation in the declaration", () => {
    // Even a combination runtime rejects from the declaration alone (packed activations × a8)
    // passes if the effective settings are accepted (the accepted set is decided by the effective
    // settings alone).
    const declared = {
      linearCompute: "a8",
      numerics: { ops: { linear: "reorder" } },
      packedStaticQuantize: true,
    } as const;
    assertThrows(
      () => resolveSessionOptions(ALL, declared, {}, "test"),
      Error,
      'packedStaticQuantize needs linear at the level "reorder" (numerics) with linearCompute "f32"',
    );
    assertEquals(resolveSessionOptions(ALL, declared, { linearCompute: "f32" }, "test"), {
      linearCompute: "f32",
      numerics: { ops: { linear: "reorder" } },
      packedStaticQuantize: true,
    });
  });

  it("accepts an explicit integerGemv only on top of the packed activations it requires", () => {
    const packed = {
      numerics: { ops: { linear: "reorder" } },
      packedStaticQuantize: true,
    } as const;
    assertEquals(resolveSessionOptions(ALL, packed, { integerGemv: true }, "test"), {
      ...packed,
      integerGemv: true,
    });
    const requires = `test: ${sessionOptionsViolation({ integerGemv: true })}`;
    assert(requires.includes("integerGemv requires packedStaticQuantize: true"));
    assertThrows(
      () => resolveSessionOptions(ALL, {}, { integerGemv: true }, "test"),
      ModelInputError,
      requires,
    );
    // An explicit false lifts the packed declaration the integer GEMV depends on: input-caused.
    assertThrows(
      () =>
        resolveSessionOptions(
          ALL,
          { ...packed, integerGemv: true },
          { packedStaticQuantize: false },
          "test",
        ),
      ModelInputError,
      requires,
    );
  });
});

describe("assertSessionOverrides", () => {
  it("gives the same verdict as resolveSessionOptions against the declaration {}", () => {
    const cases: readonly SessionOverrides[] = [
      {},
      { linearCompute: "f16" },
      { attentionCompute: "f16", attentionScoreStorage: "f16" },
      { packedStaticQuantize: true },
      { numerics: { level: "reorder" } },
      overridesWith("linearCompute", null),
      overridesWith("numerics", null),
    ];
    const verdict = (run: () => void): string | undefined => {
      try {
        run();
        return undefined;
      } catch (error) {
        assert(
          error instanceof ModelInputError,
          "a violation from explicit settings alone is an input error",
        );
        return error.message;
      }
    };
    for (const overrides of cases) {
      assertEquals(
        verdict(() => assertSessionOverrides(ALL, overrides, "test")),
        verdict(() => resolveSessionOptions(ALL, {}, overrides, "test")),
      );
    }
  });
});

describe("the families' policy tables", () => {
  it("the tables of all 10 families have every key of SessionSpec (the run-time counterpart of the type's exhaustiveness)", () => {
    // The tables' type is the exhaustive `Required<SessionSpec>`, so a new key fails the type
    // check. This is the counterpart for a change that loosens the type (making it `Partial`, for
    // example) and so passes the type check.
    const policies = {
      anima: ANIMA_SESSION_POLICY,
      birefnet: BIREFNET_SESSION_POLICY,
      "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
      embeddinggemma: EMBEDDINGGEMMA_SESSION_POLICY,
      gemma: GEMMA_SESSION_POLICY,
      irodori: IRODORI_SESSION_POLICY,
      sbv2: SBV2_SESSION_POLICY,
      siglip2: SIGLIP2_SESSION_POLICY,
      "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
      wan: WAN_SESSION_POLICY,
    };
    for (const [family, policy] of Object.entries(policies)) {
      assertEquals(Object.keys(policy).sort(), Object.keys(FULL).sort(), family);
    }
  });
});
