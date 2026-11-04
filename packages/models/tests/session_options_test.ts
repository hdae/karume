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
import { ANIMA_SESSION_POLICY } from "../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../src/depth-anything/pipeline.ts";
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
import { VOWEL_DETECTOR_SESSION_POLICY } from "../src/vowel-detector/pipeline.ts";
import { WAN_SESSION_POLICY } from "../src/wan/family.ts";

test("toSessionOptions: maps the declared keys one by one (an unspecified key creates no field at all)", () => {
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ linearGemvReduce: "parallel" }), {
    linearGemvReduce: "parallel",
  });
  assertEquals(toSessionOptions({ linearGemvReduce: "sequential" }), {
    linearGemvReduce: "sequential",
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
    linearCompute: "f16",
    convCompute: "a8",
    attentionCompute: "f16",
    attentionScoreStorage: "f16",
    linearGemvReduce: "sequential",
    stateAttentionReduce: "parallel-fused",
    fuseRmsNormAdd: false,
    fuseLinearStaticQuantize: false,
    fuseGeluMulStaticQuantize: false,
    packedStaticQuantize: false,
    integerGemv: false,
    linearGemvPlanes: false,
  };
  const mapped = toSessionOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(mapped).sort(), Object.keys(full).sort());
});

test("toSessionOptions: keeps explicit false and true for the fusions", () => {
  for (const value of [false, true]) {
    const spec = {
      fuseRmsNormAdd: value,
      fuseLinearStaticQuantize: value,
      fuseGeluMulStaticQuantize: value,
      packedStaticQuantize: value,
      integerGemv: value,
      linearGemvPlanes: value,
    };
    assertEquals(toSessionOptions(spec), spec);
  }
});

// ---- Composition (explicit > quant declaration > runtime default) — ADR 0058 addendum 2026-09-26
// ----

/**
 * A policy table that accepts all 8 keys (to look at the rules themselves, apart from any family's
 * table).
 */
const ALL: FamilySessionPolicy = {
  linearCompute: true,
  convCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: true,
  stateAttentionReduce: true,
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  fuseGeluMulStaticQuantize: true,
  packedStaticQuantize: true,
  integerGemv: true,
  linearGemvPlanes: true,
};

/** A table that accepts only the linear execution form (the same shape as irodori / sbv2). */
const LINEAR_ONLY: FamilySessionPolicy = {
  ...ALL,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
};

/** A spec with every key filled (a new key fails this declaration's type check). */
const FULL: Required<SessionSpec> = {
  linearCompute: "f32",
  convCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  linearGemvReduce: "parallel",
  stateAttentionReduce: "parallel",
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
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
    const declared = { linearGemvReduce: "parallel", fuseRmsNormAdd: true } as const;
    assertEquals(resolveSessionOptions(ALL, declared, { fuseRmsNormAdd: false }, "test"), {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: false,
    });
    assertEquals(
      resolveSessionOptions(ALL, declared, { fuseRmsNormAdd: undefined }, "test"),
      declared,
    );
  });

  it("does not reject values that depend on GPU capabilities (f16 compute, subgroup variants) at this layer", () => {
    assertEquals(resolveSessionOptions(ALL, {}, { linearCompute: "f16" }, "test"), {
      linearCompute: "f16",
    });
    assertEquals(
      resolveSessionOptions(ALL, {}, { linearGemvReduce: "parallel-subgroup32" }, "test"),
      { linearGemvReduce: "parallel-subgroup32" },
    );
  });
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
    const messages: Readonly<Record<keyof SessionSpec, string>> = {
      linearCompute: "linearCompute is invalid",
      convCompute: "convCompute is invalid",
      attentionCompute: "attentionCompute is invalid",
      attentionScoreStorage: "attentionScoreStorage is invalid",
      linearGemvReduce: "linearGemvReduce is invalid",
      stateAttentionReduce: "stateAttentionReduce is invalid",
      fuseRmsNormAdd: "fuseRmsNormAdd must be a boolean",
      fuseLinearStaticQuantize: "fuseLinearStaticQuantize must be a boolean",
      fuseGeluMulStaticQuantize: "fuseGeluMulStaticQuantize must be a boolean",
      packedStaticQuantize: "packedStaticQuantize must be a boolean",
      integerGemv: "integerGemv must be a boolean",
      linearGemvPlanes: "linearGemvPlanes must be a boolean",
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
    for (const key of ["attentionCompute", "linearGemvReduce", "packedStaticQuantize"] as const) {
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
    // Even a combination runtime rejects from the declaration alone (parallel GEMV × a8) passes if
    // the effective settings are accepted (the accepted set is decided by the effective settings
    // alone).
    const declared = { linearCompute: "a8", linearGemvReduce: "parallel" } as const;
    assertThrows(
      () => resolveSessionOptions(ALL, declared, {}, "test"),
      Error,
      "supports only linearCompute: f32",
    );
    assertEquals(resolveSessionOptions(ALL, declared, { linearCompute: "f32" }, "test"), {
      linearCompute: "f32",
      linearGemvReduce: "parallel",
    });
  });

  it("accepts an explicit integerGemv only on top of the packed activations it requires", () => {
    const packed = { linearGemvReduce: "parallel", packedStaticQuantize: true } as const;
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

  it("accepts an explicit linearGemvPlanes on its own: it requires no other option", () => {
    assertEquals(resolveSessionOptions(ALL, {}, { linearGemvPlanes: true }, "test"), {
      linearGemvPlanes: true,
    });
    // An explicit false wins over a declaration, as for every key.
    assertEquals(
      resolveSessionOptions(ALL, { linearGemvPlanes: true }, { linearGemvPlanes: false }, "test"),
      { linearGemvPlanes: false },
    );
  });
});

describe("assertSessionOverrides", () => {
  it("gives the same verdict as resolveSessionOptions against the declaration {}", () => {
    const cases: readonly SessionOverrides[] = [
      {},
      { linearCompute: "f16" },
      { attentionCompute: "f16", attentionScoreStorage: "f16" },
      { fuseLinearStaticQuantize: true },
      overridesWith("linearCompute", null),
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
      gemma: GEMMA_SESSION_POLICY,
      irodori: IRODORI_SESSION_POLICY,
      sbv2: SBV2_SESSION_POLICY,
      siglip2: SIGLIP2_SESSION_POLICY,
      "umt5-encoder": UMT5_ENCODER_SESSION_POLICY,
      "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
      wan: WAN_SESSION_POLICY,
    };
    for (const [family, policy] of Object.entries(policies)) {
      assertEquals(Object.keys(policy).sort(), Object.keys(FULL).sort(), family);
    }
  });
});
