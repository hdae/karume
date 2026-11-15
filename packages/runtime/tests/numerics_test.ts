// The numeric requirement declaration (`SessionOptions.numerics`): its acceptance, the resolution of
// its two scopes, and the reduction forms a level resolves to. Pure functions — no GPU.

import { assert, assertEquals, assertStrictEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type SessionOptions, sessionOptionsViolation } from "../mod.ts";
import {
  copyNumerics,
  type Numerics,
  numericLevels,
  numericsViolation,
} from "../src/runtime/numerics.ts";
import { executionKnobDefaults } from "../src/runtime/session-options-check.ts";
import { REORDER_STRATEGIES } from "../src/runtime/session-types.ts";

/** A value from a caller outside the types (a JS consumer, a manifest). */
const untypedOptions = (value: unknown): SessionOptions => {
  const options: SessionOptions = {};
  Object.defineProperty(options, "numerics", { value, enumerable: true });
  return options;
};

const untyped = (value: unknown): Numerics | undefined => untypedOptions(value).numerics;

describe("numericLevels", () => {
  it("runs every op kind at exact without a declaration", () => {
    assertEquals(numericLevels(undefined), {
      linear: "exact",
      attention: "exact",
      rms_norm: "exact",
    });
    assertEquals(numericLevels({}), { linear: "exact", attention: "exact", rms_norm: "exact" });
  });

  it("applies the model's level to every op kind", () => {
    assertEquals(numericLevels({ level: "reorder" }), {
      linear: "reorder",
      attention: "reorder",
      rms_norm: "reorder",
    });
  });

  it("lets an op kind's level win over the model's, in both directions", () => {
    assertEquals(numericLevels({ level: "reorder", ops: { attention: "exact" } }), {
      linear: "reorder",
      attention: "exact",
      rms_norm: "reorder",
    });
    assertEquals(numericLevels({ ops: { linear: "reorder" } }), {
      linear: "reorder",
      attention: "exact",
      rms_norm: "exact",
    });
  });
});

describe("numericsViolation", () => {
  it("accepts an omitted declaration and every well-formed one", () => {
    for (const numerics of [
      undefined,
      {},
      { level: "exact" },
      { level: "reorder" },
      { ops: {} },
      { level: "reorder", ops: { linear: "exact", attention: "reorder", rms_norm: "exact" } },
    ] as const) {
      assertEquals(numericsViolation(numerics), undefined);
    }
  });

  it("rejects a value that is not an object", () => {
    for (const value of ["reorder", null, 1, ["reorder"]]) {
      const violation = numericsViolation(value);
      assert(violation?.includes("options.numerics must be an object"), String(violation));
    }
  });

  it("rejects a level outside the two, naming the field", () => {
    assert(numericsViolation({ level: "fast" })?.startsWith('options.numerics.level "fast"'));
    assert(
      numericsViolation({ ops: { linear: "approximation" } })?.startsWith(
        'options.numerics.ops.linear "approximation"',
      ),
    );
  });

  it("rejects an unknown field and an unknown op kind instead of reading them as not declared", () => {
    assertEquals(
      numericsViolation({ levels: "reorder" }),
      "options.numerics.levels is not a field",
    );
    assert(numericsViolation({ ops: { linears: "reorder" } })?.includes("is not an op kind"));
  });

  it("does not call a value's conversions in the diagnostic", () => {
    const hostile = {
      toString: () => {
        throw new Error("toString called");
      },
    };
    assert(numericsViolation({ level: hostile })?.includes("object"));
  });
});

describe("copyNumerics", () => {
  it("returns plain data equal to a well-formed declaration", () => {
    const declared: Numerics = { level: "reorder", ops: { attention: "exact" } };
    const copy = copyNumerics(declared);
    assertEquals(copy, declared);
    assert(copy !== declared && copy?.ops !== declared.ops);
  });

  it("reads each field once, so the checked value is the value used", () => {
    let reads = 0;
    const flipping = {
      get level() {
        reads += 1;
        return reads === 1 ? ("reorder" as const) : ("exact" as const);
      },
    };
    const copy = copyNumerics(flipping);
    assertEquals(numericsViolation(copy), undefined);
    assertEquals(numericLevels(copy).linear, "reorder");
    assertEquals(reads, 1);
  });

  it("keeps a value the check rejects by its type or its keys", () => {
    for (const value of ["reorder", null, { levels: "reorder" }, { ops: { linears: "reorder" } }]) {
      const copy = copyNumerics(untyped(value));
      assert(numericsViolation(copy) !== undefined, JSON.stringify(value));
    }
    assert(numericsViolation(copyNumerics(untyped({ ops: "reorder" }))) !== undefined);
    assert(numericsViolation(copyNumerics(untyped({ level: "fast" }))) !== undefined);
  });
});

describe("the reduction forms a level resolves to", () => {
  it("keeps the reference kernels at exact", () => {
    const knobs = executionKnobDefaults({});
    assertStrictEquals(knobs.linearGemvReduce, "sequential");
    assertStrictEquals(knobs.stateAttentionReduce, "sequential");
  });

  it("runs the parallel GEMV and the fused parallel reductions of states-form attention at reorder", () => {
    const knobs = executionKnobDefaults({ numerics: { level: "reorder" } });
    assertStrictEquals(knobs.linearGemvReduce, "parallel");
    assertStrictEquals(knobs.stateAttentionReduce, "parallel-fused");
  });

  it("moves only the op kind an override names", () => {
    const linearOnly = executionKnobDefaults({ numerics: { ops: { linear: "reorder" } } });
    assertStrictEquals(linearOnly.linearGemvReduce, "parallel");
    assertStrictEquals(linearOnly.stateAttentionReduce, "sequential");
    const attentionExact = executionKnobDefaults({
      numerics: { level: "reorder", ops: { attention: "exact" } },
    });
    assertStrictEquals(attentionExact.linearGemvReduce, "parallel");
    assertStrictEquals(attentionExact.stateAttentionReduce, "sequential");
  });

  it("leaves the approximations at their defaults", () => {
    const knobs = executionKnobDefaults({ numerics: { level: "reorder" } });
    assertStrictEquals(knobs.linearCompute, "f32");
    assertStrictEquals(knobs.attentionCompute, "f32");
    assertStrictEquals(knobs.attentionScoreStorage, "f32");
    assertStrictEquals(knobs.fullStateStorage, "f32");
  });
});

describe("sessionOptionsViolation with numerics", () => {
  it("accepts a declaration alone and beside the approximation options", () => {
    assertEquals(sessionOptionsViolation({ numerics: { level: "reorder" } }), undefined);
    assertEquals(
      sessionOptionsViolation({
        numerics: { ops: { attention: "reorder" } },
        linearCompute: "a8",
        attentionCompute: "a8",
      }),
      undefined,
    );
  });

  it("reports a malformed declaration before any other option", () => {
    const options: SessionOptions = { ...untypedOptions({ level: "fast" }), linearCompute: "f16" };
    assert(sessionOptionsViolation(options)?.startsWith("options.numerics.level"));
  });

  it("holds the packed int8 activations to a linear level of reorder", () => {
    assertEquals(
      sessionOptionsViolation({ numerics: { level: "reorder" }, packedStaticQuantize: true }),
      undefined,
    );
    assert(sessionOptionsViolation({ packedStaticQuantize: true }) !== undefined);
    assert(
      sessionOptionsViolation({
        numerics: { level: "reorder", ops: { linear: "exact" } },
        packedStaticQuantize: true,
      }) !== undefined,
    );
  });

  it("accepts reorder beside a named linear approximation, which keeps its own kernels", () => {
    for (const linearCompute of ["a8", "f16"] as const) {
      const options: SessionOptions = { numerics: { level: "reorder" }, linearCompute };
      assertEquals(sessionOptionsViolation(options), undefined);
      const knobs = executionKnobDefaults(options);
      assertStrictEquals(knobs.linearGemvReduce, "sequential");
      assertStrictEquals(knobs.stateAttentionReduce, "parallel-fused");
    }
  });
});

describe("the adapter-dependent reorder strategies (the private REORDER_STRATEGIES key)", () => {
  const all = { linearGemv: "subgroup32", linearGemvPlanes: true, rmsNorm: "subgroup32" } as const;

  it("selects nothing by itself: a strategy runs only where the level admits it", () => {
    const knobs = executionKnobDefaults({ [REORDER_STRATEGIES]: all });
    assertStrictEquals(knobs.linearGemvReduce, "sequential");
    assertStrictEquals(knobs.linearGemvPlanes, false);
    assertStrictEquals(knobs.rmsNormReduce, "workgroup");
  });

  it("keeps the portable strategies at reorder when none is named", () => {
    const knobs = executionKnobDefaults({ numerics: { level: "reorder" } });
    assertStrictEquals(knobs.linearGemvReduce, "parallel");
    assertStrictEquals(knobs.linearGemvPlanes, false);
    assertStrictEquals(knobs.rmsNormReduce, "workgroup");
  });

  it("runs each named strategy under its own op kind's level", () => {
    const whole = executionKnobDefaults({
      numerics: { level: "reorder" },
      [REORDER_STRATEGIES]: all,
    });
    assertStrictEquals(whole.linearGemvReduce, "parallel-subgroup32");
    assertStrictEquals(whole.linearGemvPlanes, true);
    assertStrictEquals(whole.rmsNormReduce, "subgroup32");
    const linearOnly = executionKnobDefaults({
      numerics: { ops: { linear: "reorder" } },
      [REORDER_STRATEGIES]: all,
    });
    assertStrictEquals(linearOnly.linearGemvReduce, "parallel-subgroup32");
    assertStrictEquals(linearOnly.rmsNormReduce, "workgroup");
    const rmsOnly = executionKnobDefaults({
      numerics: { ops: { rms_norm: "reorder" } },
      [REORDER_STRATEGIES]: all,
    });
    assertStrictEquals(rmsOnly.linearGemvReduce, "sequential");
    assertStrictEquals(rmsOnly.linearGemvPlanes, false);
    assertStrictEquals(rmsOnly.rmsNormReduce, "subgroup32");
  });

  it("keeps the workgroup GEMV for packed int8 activations, which the subgroup form cannot take", () => {
    const options: SessionOptions = {
      numerics: { level: "reorder" },
      packedStaticQuantize: true,
      [REORDER_STRATEGIES]: { linearGemv: "subgroup32" },
    };
    assertEquals(sessionOptionsViolation(options), undefined);
    assertStrictEquals(executionKnobDefaults(options).linearGemvReduce, "parallel");
  });

  it("rejects a value outside the type", () => {
    const bad: SessionOptions = {};
    Object.defineProperty(bad, REORDER_STRATEGIES, {
      value: { linearGemv: "subgroup16" },
      enumerable: true,
    });
    assert(sessionOptionsViolation(bad)?.includes("linearGemv"));
  });
});
