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
  const fast = {
    linearGemvReduce: "parallel",
    fuseRmsNormAdd: true,
    fuseLinearStaticQuantize: true,
  } as const;
  it("distinguishes unset, the quant declaration and an explicit override", () => {
    assertEquals(resolve({}, {}, "test"), {});
    assertEquals(resolve(fast, {}, "test"), fast);
    assertEquals(resolve(fast, { fuseRmsNormAdd: undefined }, "test"), fast);
    assertEquals(resolve(fast, { fuseRmsNormAdd: false }, "test"), {
      ...fast,
      fuseRmsNormAdd: false,
    });
    assertEquals(resolve(fast, { fuseLinearStaticQuantize: false }, "test"), {
      ...fast,
      fuseLinearStaticQuantize: false,
    });
    const reference = {
      linearGemvReduce: "sequential",
      fuseRmsNormAdd: false,
      fuseLinearStaticQuantize: false,
    } as const;
    assertEquals(resolve(fast, reference, "test"), reference);
    assertEquals(resolve({ fuseRmsNormAdd: false }, { fuseRmsNormAdd: true }, "test"), {
      fuseRmsNormAdd: true,
    });
    assertEquals(resolve({}, { linearGemvReduce: "parallel-subgroup32" }, "test"), {
      linearGemvReduce: "parallel-subgroup32",
    });
  });
  it("does not hide an unsupported declaration behind an override, and rejects invalid combinations of the effective settings", () => {
    assertThrows(
      () => resolve({ linearCompute: "f32" }, {}, "test"),
      Error,
      "session.linearCompute",
    );
    for (const linearGemvReduce of ["sequential", "parallel-subgroup32"] as const) {
      assertThrows(
        () => resolve(fast, { linearGemvReduce }, "test"),
        ModelInputError,
        "supports only linearGemvReduce: parallel / linearCompute: f32",
      );
      assertEquals(
        resolve(
          fast,
          {
            linearGemvReduce,
            fuseLinearStaticQuantize: false,
          },
          "test",
        ),
        { ...fast, linearGemvReduce, fuseLinearStaticQuantize: false },
      );
    }
    assertThrows(
      () => resolve({}, { fuseLinearStaticQuantize: true }, "test"),
      ModelInputError,
      "supports only linearGemvReduce: parallel / linearCompute: f32",
    );
    assertEquals(
      resolve(
        { fuseLinearStaticQuantize: true },
        {
          linearGemvReduce: "parallel",
        },
        "test",
      ),
      { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
    );
  });
  it("packedStaticQuantize comes from the quant declaration, an explicit false wins, and anything but parallel is rejected", () => {
    // ADR 0105 addendum 2 (promotion into the vocabulary) — the same 4 fields as the shipped i4-fast pass through from the declaration.
    const packedFast = { ...fast, packedStaticQuantize: true } as const;
    assertEquals(resolve(packedFast, {}, "test"), packedFast);
    assertEquals(resolve(packedFast, { packedStaticQuantize: false }, "test"), {
      ...packedFast,
      packedStaticQuantize: false,
    });
    assertEquals(resolve(fast, { packedStaticQuantize: true }, "test"), packedFast);
    // A declaration without parallel is rejected even when it comes from the quant (i4-fast declares parallel).
    assertThrows(
      () => resolve({ packedStaticQuantize: true }, {}, "test"),
      Error,
      "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
    assertThrows(
      () => resolve({}, { packedStaticQuantize: true }, "test"),
      ModelInputError,
      "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
    assertThrows(
      () =>
        resolve(
          fast,
          {
            linearGemvReduce: "sequential",
            fuseLinearStaticQuantize: false,
            packedStaticQuantize: true,
          },
          "test",
        ),
      ModelInputError,
      "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
  });
  it("takes integerGemv explicitly on top of a preset that declares the packed activations", () => {
    // The QAT i4-fast form: the preset declares the packed activations, the caller adds integerGemv.
    const packedFast = { ...fast, packedStaticQuantize: true } as const;
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
      { linearGemvReduce: "parallel", packedStaticQuantize: true, integerGemv: true },
      "test",
    );
  });
  it("takes linearGemvPlanes explicitly on any preset, alone, at both entry points", () => {
    // The byte-plane GEMV needs no other option: it joins the QAT i4-fast declaration or none.
    assertEquals(resolve(fast, { linearGemvPlanes: true }, "test"), {
      ...fast,
      linearGemvPlanes: true,
    });
    assertEquals(resolve({}, { linearGemvPlanes: true }, "test"), { linearGemvPlanes: true });
    assertGemmaSessionOverrides({ linearGemvPlanes: true }, "test");
  });
  it("stateAttentionReduce resolves by the same rules as the other fields and creates no field without a declaration", () => {
    // There is no family code default (ADR 0104 addendum 2026-09-26). No field = the runtime's reference path
    // `"sequential"` — the `i4` preset with an empty `session` and presets that declare only GEMV take this shape.
    assertEquals(resolve({}, {}, "test"), {});
    assertEquals(resolve({ linearGemvReduce: "parallel" }, {}, "test"), {
      linearGemvReduce: "parallel",
    });
    // The same declaration as regular E2B's i4-fast — all 3 fields pass through from the declaration.
    const fastE2b = {
      linearGemvReduce: "parallel",
      stateAttentionReduce: "parallel-fused",
      fuseRmsNormAdd: true,
    } as const;
    assertEquals(resolve(fastE2b, {}, "test"), fastE2b);
    // An explicit value wins over the declaration (the way back to the reference path).
    assertEquals(resolve(fastE2b, { stateAttentionReduce: "sequential" }, "test"), {
      ...fastE2b,
      stateAttentionReduce: "sequential",
    });
    assertEquals(resolve({}, { stateAttentionReduce: "parallel" }, "test"), {
      stateAttentionReduce: "parallel",
    });
    // The surface without a declaration (fromAssets) accepts all 3 values too.
    for (const stateAttentionReduce of ["sequential", "parallel", "parallel-fused"] as const) {
      assertGemmaSessionOverrides({ stateAttentionReduce }, "test");
    }
  });
  it("every key of SessionSpec appears either in the allow table or on the rejection path", () => {
    // The hub's SESSION_KEYS and the models' WRITERS are exhaustive tables over `Required<SessionSpec>`, so a new knob
    // fails the type check. Only Gemma uses 4 hand-written string comparisons (ADR 0104's 3 fields + ADR 0105
    // addendum 2's packed), so the type check does not fail and a missed update shows only as a runtime rejection.
    // Here, from a spec with every key filled (a new key makes this declaration fail the type check), each key is
    // confirmed to pass / fail one at a time.
    const full: Required<SessionSpec> = {
      linearCompute: "f16",
      convCompute: "a8",
      attentionCompute: "f16",
      attentionScoreStorage: "f16",
      linearGemvReduce: "sequential",
      stateAttentionReduce: "sequential",
      fuseRmsNormAdd: false,
      fuseLinearStaticQuantize: false,
      fuseGeluMulStaticQuantize: false,
      packedStaticQuantize: false,
      integerGemv: false,
      linearGemvPlanes: false,
    };
    const accepted = [
      "linearGemvReduce",
      "stateAttentionReduce",
      "fuseRmsNormAdd",
      "fuseLinearStaticQuantize",
      "fuseGeluMulStaticQuantize",
      "packedStaticQuantize",
      "integerGemv",
      "linearGemvPlanes",
    ];
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
      ["linearGemvReduce", "linearGemvReduce is invalid"],
      ["stateAttentionReduce", "stateAttentionReduce is invalid"],
      ["fuseRmsNormAdd", "fuseRmsNormAdd must be a boolean"],
      ["fuseLinearStaticQuantize", "fuseLinearStaticQuantize must be a boolean"],
      ["fuseGeluMulStaticQuantize", "fuseGeluMulStaticQuantize must be a boolean"],
      ["packedStaticQuantize", "packedStaticQuantize must be a boolean"],
      ["integerGemv", "integerGemv must be a boolean"],
      ["linearGemvPlanes", "linearGemvPlanes must be a boolean"],
    ] as const;
    for (const [key, message] of invalid) {
      for (const value of [null, 0, 1, [], object]) {
        const overrides = {};
        // The options real calls pass have only enumerable properties, so the check uses the same shape.
        Object.defineProperty(overrides, key, { value, enumerable: true });
        assertThrows(() => resolve(fast, overrides, "test"), ModelInputError, message);
      }
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
    assertEquals(resolve({ linearGemvReduce: "sequential" }, {}, "test"), {
      linearGemvReduce: "sequential",
    });
    assertThrows(
      () => resolve({ linearGemvReduce: "sequential" }, { fuseLinearStaticQuantize: true }, "test"),
      ModelInputError,
      "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
    assertEquals(
      resolve({ packedStaticQuantize: true, linearGemvReduce: "parallel" }, {}, "test"),
      {
        packedStaticQuantize: true,
        linearGemvReduce: "parallel",
      },
    );
    assertThrows(
      () =>
        resolve(
          { packedStaticQuantize: true, linearGemvReduce: "parallel" },
          {
            linearGemvReduce: "sequential",
          },
          "test",
        ),
      ModelInputError,
      "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
  });
});

describe("Gemma's quant execution settings (the check of explicit settings only)", () => {
  it("rejects with the same ranges, types and combinations on a surface without a quant declaration", () => {
    // fromAssets has no manifest, so the verdict is the same as when compared with the default {}.
    assertGemmaSessionOverrides({}, "test");
    assertGemmaSessionOverrides({ linearGemvReduce: "parallel-subgroup32" }, "test");
    assertThrows(
      () => assertGemmaSessionOverrides({ fuseLinearStaticQuantize: true }, "test"),
      ModelInputError,
      "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
    );
    const bogus = {};
    Object.defineProperty(bogus, "fuseRmsNormAdd", { value: 1, enumerable: true });
    assertThrows(
      () => assertGemmaSessionOverrides(bogus, "test"),
      ModelInputError,
      "fuseRmsNormAdd must be a boolean",
    );
  });
  it("agrees with the verdict resolve gives on the same explicit settings", () => {
    // Pins that the 2 entry points go through the same check by the agreement of their verdicts (no loosening of only one side).
    const explicit = [
      {},
      { linearGemvReduce: "parallel" },
      { stateAttentionReduce: "parallel-fused" },
      { fuseLinearStaticQuantize: true },
      { packedStaticQuantize: true, linearGemvReduce: "sequential" },
      { packedStaticQuantize: true, linearGemvReduce: "parallel" },
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
