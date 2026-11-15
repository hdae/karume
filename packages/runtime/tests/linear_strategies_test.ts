/**
 * The selection of a linear's strategy (`src/runtime/linear-strategies.ts`): which kernel a linear
 * takes for its shape, the storage of its weight, the Session's options and what the fusion pass
 * declared for it; that the order of the table decides between strategies whose cases overlap; and
 * that a combination nothing implements is refused instead of falling to another path.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import {
  LINEAR_STRATEGIES,
  type LinearFacts,
  type LinearKnobs,
  selectLinearStrategy,
} from "../src/runtime/linear-strategies.ts";
import { ExecutionError, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

/** The plan of one `linear` node `[m, k] × [n, k]ᵀ + [n]`. */
const linearPlan = (m: number, n: number, k: number): NodePlan => {
  const ir = mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  });
  const [plan] = planGraph(ir, {}).nodes;
  return plan;
};

/** The Session's omitted values. */
const DEFAULTS: LinearKnobs = {
  linearCompute: "f32",
  linearGemvReduce: "sequential",
  integerGemv: false,
  linearGemvPlanes: false,
  linearGemvRowsThreadTarget: undefined,
};

const facts = (
  shape: { readonly m: number; readonly n: number; readonly k: number },
  weightStorage: WeightStorage,
  overrides: Partial<Omit<LinearFacts, "plan" | "weightStorage" | "knobs">> & {
    readonly knobs?: Partial<LinearKnobs>;
  } = {},
): LinearFacts => ({
  plan: linearPlan(shape.m, shape.n, shape.k),
  weightStorage,
  groupSize: overrides.groupSize,
  knobs: { ...DEFAULTS, ...overrides.knobs },
  packed: overrides.packed,
  prologue: overrides.prologue,
});

const idOf = (...args: Parameters<typeof facts>): string => selectLinearStrategy(facts(...args)).id;

/** Rows of the measured tables of the kernels (the strategies that take measured shapes only). */
const PARALLEL_I4 = { m: 1, n: 8960, k: 1536 } as const;
const INTEGER_I2 = { m: 1, n: 12288, k: 1536 } as const;
const LM_HEAD_I2 = { m: 1, n: 262144, k: 1536 } as const;

describe("with every option omitted", () => {
  it("takes the default GEMM for many rows, whatever the weight's storage", () => {
    for (const storage of ["f32", "f16", "bf16", "i8"] as const) {
      assertEquals(idOf({ m: 128, n: 64, k: 64 }, storage), "gemm-f32");
    }
    assertEquals(idOf({ m: 128, n: 64, k: 64 }, "i4", { groupSize: 32 }), "gemm-f32");
  });

  it("takes the sequential GEMV for one row of a float-stored weight, and the GEMM for more rows", () => {
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "f32"), "gemv-m1");
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "f16"), "gemv-m1");
    assertEquals(idOf({ m: 2, n: 64, k: 64 }, "f32"), "gemm-f32");
  });

  it("takes the GEMV family for 1 to 64 rows of an integer-stored weight: the column form at M = 1, row blocks above", () => {
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "i8"), "gemv-m1");
    assertEquals(idOf({ m: 8, n: 64, k: 64 }, "i8"), "gemv-rows");
    assertEquals(idOf({ m: 64, n: 64, k: 64 }, "i4", { groupSize: 32 }), "gemv-rows");
    assertEquals(idOf({ m: 65, n: 64, k: 64 }, "i8"), "gemm-f32");
  });

  it("keeps a shape the GEMV's weight binding cannot take on the GEMM", () => {
    // k is not a multiple of the 16 elements of one i8 weight word.
    assertEquals(idOf({ m: 1, n: 64, k: 60 }, "i8"), "gemm-f32");
    // An i4 group shorter than one weight word (32 elements) would need two scales per word.
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "i4", { groupSize: 16 }), "gemm-f32");
    // n is not a multiple of 4: outside the measured (vec4) range.
    assertEquals(idOf({ m: 1, n: 62, k: 64 }, "f32"), "gemm-f32");
  });

  it("never leaves the strategies whose level is exact", () => {
    for (const shape of [PARALLEL_I4, { m: 4, n: 64, k: 64 }, { m: 128, n: 64, k: 64 }]) {
      const { id } = selectLinearStrategy(facts(shape, "i4", { groupSize: 32 }));
      assertEquals(LINEAR_STRATEGIES[id].level, "exact");
    }
  });
});

describe("with an option that asks for another kernel", () => {
  it('takes the parallel GEMV for a measured shape under linearGemvReduce "parallel", and stays sequential elsewhere', () => {
    const parallel = { knobs: { linearGemvReduce: "parallel" }, groupSize: 32 } as const;
    assertEquals(idOf(PARALLEL_I4, "i4", parallel), "gemv-parallel");
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "i4", parallel), "gemv-m1");
  });

  it('takes the subgroup form under "parallel-subgroup32"', () => {
    assertEquals(
      idOf(PARALLEL_I4, "i4", {
        knobs: { linearGemvReduce: "parallel-subgroup32" },
        groupSize: 32,
      }),
      "gemv-parallel-subgroup32",
    );
  });

  it('takes the integer dot products under linearCompute "a8" for an i8 or i4 weight, and not for a float one', () => {
    const a8 = { knobs: { linearCompute: "a8" } } as const;
    assertEquals(idOf({ m: 128, n: 64, k: 64 }, "i8", a8), "i8a8");
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "i4", { ...a8, groupSize: 32 }), "i8a8");
    assertEquals(idOf({ m: 128, n: 64, k: 64 }, "f32", a8), "gemm-f32");
    // k % 4 != 0 has no dot-product words.
    assertEquals(idOf({ m: 128, n: 64, k: 62 }, "i8", a8), "gemm-f32");
  });

  it('takes the f16 GEMM under linearCompute "f16" for a float-stored weight, even at one row', () => {
    const f16 = { knobs: { linearCompute: "f16" } } as const;
    assertEquals(idOf({ m: 128, n: 64, k: 64 }, "f16", f16), "gemm-f16");
    assertEquals(idOf({ m: 1, n: 64, k: 64 }, "f32", f16), "gemm-f16");
  });

  it("takes the byte-plane GEMV for the measured i2 shape under linearGemvPlanes, in place of the i2 GEMV", () => {
    assertEquals(idOf(LM_HEAD_I2, "i2"), "gemv-m1");
    assertEquals(idOf(LM_HEAD_I2, "i2", { knobs: { linearGemvPlanes: true } }), "planes-gemv");
  });
});

describe("with packed activations declared by the fusion pass", () => {
  const packed = { role: "read", scale: 0.5 } as const;
  const parallel = { linearGemvReduce: "parallel" } as const;

  it("takes the integer GEMV before anything else when integerGemv names a variant for the shape", () => {
    const choice = selectLinearStrategy(
      facts(INTEGER_I2, "i2", { knobs: { ...parallel, integerGemv: true }, packed }),
    );
    assertEquals(choice.id, "integer-gemv");
    assertEquals(choice.id === "integer-gemv" && choice.activationScale, 0.5);
  });

  it("keeps the same linear on the parallel GEMV that reads f32 activations when its producer was not packed", () => {
    assertEquals(
      idOf(INTEGER_I2, "i2", { knobs: { ...parallel, integerGemv: true } }),
      "gemv-parallel",
    );
  });

  it("refuses a packed linear that lands on no kernel reading packed words", () => {
    // Sequential reduction: neither the packed parallel GEMV nor the integer GEMV.
    assertThrows(
      () => selectLinearStrategy(facts(INTEGER_I2, "i2", { packed })),
      ExecutionError,
      "lands on neither the parallel GEMV nor the integer GEMV",
    );
  });

  it("refuses a packed linear that is a byte-plane GEMV linear, which reads f32 activations", () => {
    assertThrows(
      () =>
        selectLinearStrategy(
          facts(LM_HEAD_I2, "i2", { knobs: { ...parallel, linearGemvPlanes: true }, packed }),
        ),
      ExecutionError,
      "is a byte-plane GEMV linear",
    );
  });
});

describe("combinations that are not implemented", () => {
  it("refuses an i2 weight under any compute but f32, naming the rule", () => {
    assertThrows(
      () =>
        selectLinearStrategy(
          facts({ m: 1, n: 64, k: 64 }, "i2", { knobs: { linearCompute: "a8" } }),
        ),
      ExecutionError,
      "i2 residency supports only linearCompute 'f32'",
    );
  });

  it("refuses f16 compute on an i8 or i4 weight instead of dropping to f32", () => {
    const f16 = { knobs: { linearCompute: "f16" } } as const;
    assertThrows(
      () => selectLinearStrategy(facts({ m: 128, n: 64, k: 64 }, "i8", f16)),
      ExecutionError,
      "w8a16 is not implemented",
    );
    assertThrows(
      () => selectLinearStrategy(facts({ m: 128, n: 64, k: 64 }, "i4", { ...f16, groupSize: 32 })),
      ExecutionError,
      "w4a16 is not implemented",
    );
  });

  it("refuses a folded producer on a linear that does not take the integer dot products", () => {
    assertThrows(
      () =>
        selectLinearStrategy(
          facts({ m: 128, n: 64, k: 64 }, "i8", { prologue: { op: "gelu_tanh" } }),
        ),
      ExecutionError,
      "declared with a gelu_tanh prologue",
    );
  });
});
