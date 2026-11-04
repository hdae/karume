// The check of `sessionOptionsViolation` (the GPU-independent acceptance conditions of
// SessionOptions) — a pure function that touches no GPU, so it sits where a regression is caught
// even in an environment without an adapter.
//
// Session construction throws this function's violation as an `ExecutionError`, and the models-side
// merge (explicit > declared > runtime default) fails with the same function before fetching the
// weights. What this pins is two things: "it returns the same verdict and the same wording as
// construction" and "it does not mix in conditions that depend on the GPU's capabilities" (mixed
// in, a caller that decides without looking at the device would reject f16 even on a device that
// has shader-f16).

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type SessionOptions, sessionOptionsViolation } from "../mod.ts";

/**
 * Reproduces a field from a caller outside the types (a JS consumer) — TS types do not guard this
 * path.
 */
const withValue = (key: string, value: unknown): SessionOptions => {
  const options: SessionOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

describe("sessionOptionsViolation", () => {
  it("returns undefined for the defaults and accepted combinations", () => {
    assertEquals(sessionOptionsViolation({}), undefined);
    assertEquals(
      sessionOptionsViolation({
        linearGemvReduce: "parallel",
        fuseRmsNormAdd: true,
        fuseLinearStaticQuantize: true,
        packedStaticQuantize: true,
      }),
      undefined,
    );
    assertEquals(
      sessionOptionsViolation({
        linearCompute: "a8",
        attentionCompute: "a8",
        attentionScoreStorage: "f16",
      }),
      undefined,
    );
  });

  it("does not fail on conditions that depend on the GPU's capabilities (shader-f16 / subgroups)", () => {
    // f16 compute and the subgroup variants cannot be decided without the device — that is the job
    // of construction's check.
    assertEquals(sessionOptionsViolation({ linearCompute: "f16" }), undefined);
    assertEquals(sessionOptionsViolation({ attentionCompute: "f16" }), undefined);
    assertEquals(sessionOptionsViolation({ linearGemvReduce: "parallel-subgroup32" }), undefined);
    assertEquals(sessionOptionsViolation({ rmsNormReduce: "subgroup32" }), undefined);
  });

  it("returns a combination violation with the same wording as construction", () => {
    const cases: readonly (readonly [SessionOptions, string])[] = [
      [
        { fuseLinearStaticQuantize: true },
        "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
      ],
      [
        { linearGemvReduce: "parallel", linearCompute: "a8", fuseLinearStaticQuantize: true },
        "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
      ],
      [
        { packedStaticQuantize: true, linearGemvReduce: "sequential" },
        "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32",
      ],
      [
        { linearGemvReduce: "parallel", linearCompute: "a8" },
        "linearGemvReduce: parallel supports only linearCompute: f32",
      ],
    ];
    for (const [options, message] of cases) {
      assertEquals(sessionOptionsViolation(options), message);
    }
    const s16c16 = sessionOptionsViolation({
      attentionCompute: "f16",
      attentionScoreStorage: "f16",
    });
    assert(
      s16c16?.startsWith(
        "attentionScoreStorage 'f16' and attentionCompute 'f16' cannot be given together",
      ),
    );
  });

  it("returns type, spelling and range violations, and the diagnostic calls no user conversion", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return "parallel";
      },
    };
    assertEquals(
      sessionOptionsViolation(withValue("fuseRmsNormAdd", 1)),
      "options.fuseRmsNormAdd must be a boolean",
    );
    assert(
      sessionOptionsViolation(withValue("linearGemvReduce", object))?.includes(
        "linearGemvReduce: object",
      ),
    );
    assert(
      sessionOptionsViolation(withValue("linearCompute", "i8a8"))?.includes(
        'linearCompute: "i8a8"',
      ),
    );
    assert(
      sessionOptionsViolation({ planBackingBudgetBytes: -1 })?.includes(
        "non-negative safe integer",
      ),
    );
    assert(
      sessionOptionsViolation({ linearGemvRowsThreadTarget: 0 })?.includes(
        "safe integer of at least 1",
      ),
    );
    assertEquals(conversions, 0);
  });

  it("attentionRowBlockMaxBytes accepts only positive safe integers (it rejects only the invalid values known at construction)", () => {
    // Whether a budget is smaller than one row cannot be decided without the shapes (it fails when
    // the run derives its plan — ADR 0124 decision 5), so this checks only the range. 1 B is inside
    // the range.
    for (const accepted of [1, 512 * 1024 * 1024, Number.MAX_SAFE_INTEGER]) {
      assertEquals(sessionOptionsViolation({ attentionRowBlockMaxBytes: accepted }), undefined);
    }
    for (const [value, shown] of [
      [0, "0"],
      [-1, "-1"],
      [1.5, "1.5"],
      [Number.NaN, "NaN"],
      [Number.POSITIVE_INFINITY, "Infinity"],
      [Number.MAX_SAFE_INTEGER + 1, String(Number.MAX_SAFE_INTEGER + 1)],
    ] as const) {
      assertEquals(
        sessionOptionsViolation({ attentionRowBlockMaxBytes: value }),
        `options.attentionRowBlockMaxBytes ${shown} must be a positive safe integer`,
      );
    }
    // Non-numeric values (from a JS consumer) are rejected too. A string is shown as spelled, any
    // other value only by its type name.
    assertEquals(
      sessionOptionsViolation(withValue("attentionRowBlockMaxBytes", "512")),
      'options.attentionRowBlockMaxBytes "512" must be a positive safe integer',
    );
    assert(
      sessionOptionsViolation(withValue("attentionRowBlockMaxBytes", Symbol("budget")))?.includes(
        "attentionRowBlockMaxBytes symbol",
      ),
    );
  });

  it("attentionScoreStorage 'none' (the online kernel — ADR 0132) is accepted only with attentionCompute 'a8'", () => {
    assertEquals(
      sessionOptionsViolation({ attentionCompute: "a8", attentionScoreStorage: "none" }),
      undefined,
    );
    assertEquals(
      sessionOptionsViolation({
        linearCompute: "a8",
        attentionCompute: "a8",
        attentionScoreStorage: "none",
      }),
      undefined,
    );
    // An unimplemented combination, not a fallback (ADR 0058 decision 3) — omitted = 'f32' too.
    for (const options of [
      { attentionScoreStorage: "none" },
      { attentionCompute: "f32", attentionScoreStorage: "none" },
      { attentionCompute: "f16", attentionScoreStorage: "none" },
    ] satisfies SessionOptions[]) {
      assertEquals(
        sessionOptionsViolation(options),
        "attentionScoreStorage 'none' (the online kernel) is implemented only with attentionCompute 'a8'",
        JSON.stringify(options),
      );
    }
  });

  it("reads null as the runtime default, like an omitted field (the same reading as construction)", () => {
    // Rejecting "the field is present but its value is invalid" instead of falling back to the
    // default is the job of the upper layer that takes explicit settings (resolveSessionOptions of
    // @karume/models). Turning null into a rejection here would move construction's accepted set —
    // that change shows up in this test.
    assertEquals(sessionOptionsViolation(withValue("linearCompute", null)), undefined);
  });
});
