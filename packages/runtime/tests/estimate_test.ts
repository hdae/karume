// Checks for the memory-requirement estimator (ADR 0070 decision 5).
//
// The estimator is a pure function that reports only the "required" side, so what is pinned comes
// down to **each category's formula being derived from the same source as the implementation
// (executor / GenerationContext / RunArena)**. Every expected value is a **hand-computed constant**
// — rebuilding it with the implementation's formula would make the test a tautology that fails not
// once when a formula drifts.
//
// The report has two levels: resident (`resident`) + per run shape (`scenarios`). An estimate
// without generation has the single scenario `"run"`; with it, the two execution shapes of ADR 0066
// decision 4 (`"prefill"` / `"decode"`) are listed separately. `peakAccountedBytes` is resident +
// **`max(budget, largest scenario)`** (not the sum of the scenarios — the retained set of slot
// backings never exceeds this amount: ADR 0095 decisions 1 / 4. Budget 0 = always one, so it is
// the largest scenario as before).
//
// The real-GPU comparison lives in gpu_estimate_test.ts (explicitly skipped without an adapter).
// This file holds only the checks that run without acquiring a GPU.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import type { IrGraph } from "../src/format/ir.ts";
import { numel, OpContractError } from "../src/ops.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import {
  type AdmissionReport,
  type AdmissionScenario,
  estimateGraphMemory,
  type EstimateOptions,
} from "../src/runtime/estimate.ts";
import type { PreparedModel } from "../src/runtime/executor.ts";
import { type ExecStep, FUSION_RULES, planFusions } from "../src/runtime/fusion.ts";
import { countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import { planRecipes, type StepOutput, type StepRecipe } from "../src/runtime/recipe.ts";
import { CORE_TRANSIENT_LIMITS } from "../src/runtime/transient-plan.ts";
import { planWeightBuffers, planWeightResidency } from "../src/runtime/weight-residency.ts";
import { DEFAULT_PLAN_BACKING_BUDGET_BYTES } from "../src/runtime/session-types.ts";
import {
  bothScenarios,
  decomposedAttentionGraph,
  f16Zeros,
  openGraph,
  stateAttentionGraph,
  stateGraph,
  stateModel,
  stateTensors,
  weight,
} from "./helpers/estimate-graphs.ts";
import { mergeGraph, type StorageMap } from "./helpers/merged-graph.ts";
import { type DeclarationJson, f32Bytes, singleOpDeclaration } from "./helpers/model-fixture.ts";

const i32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Int32Array.from(values).buffer);

/**
 * Checks that the scenarios are exactly the single `"run"` and returns it.
 *
 * The only way to read io / workspace of an estimate without generation — it asserts "exactly one"
 * every time it is read (if it grew to two, it would not silently look at the first only).
 */
const runScenario = (report: AdmissionReport): AdmissionScenario => {
  assertEquals(
    report.scenarios.map((scenario) => scenario.name),
    ["run"],
  );
  return report.scenarios[0];
};

// ---------------------------------------------------------------------------
// Weight categories
// ---------------------------------------------------------------------------

/**
 * A graph with uncompressed storage only (f32 matmul weight, f32 embedding table, i32 indices).
 * It has `T`, so the binding dependence of io and the intermediate peak is checked here too.
 */
const plainGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "embedding"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["h", "g"],
  initializers: {
    w: {},
    emb: {},
    idx: {},
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    emb: { dtype: "f32", shape: [5, 3] },
    idx: { dtype: "i32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "embedding", ins: ["emb", "idx"], outs: ["g"], attrs: { padding_idx: -1 } },
  ],
});

const plainModel = (): PreparedModel =>
  openGraph(plainGraph(), [
    weight("w", f32Bytes(new Array(12).fill(1))),
    weight("emb", f32Bytes(new Array(15).fill(1))),
    weight("idx", i32Bytes([0, 1]), { codec: "i32" }),
  ]);

test("uncompressed storage (f32 / i32) is counted at its actual bytes in the uncompressed field", () => {
  const { resident } = plainModel().estimate({ bindings: { T: 7 } });
  // w 4×3×4=48 + emb 5×3×4=60 + idx 2×4=8 (all already 4-byte aligned)
  assertEquals(resident.weights.uncompressedBytes, 116);
  assertEquals(resident.weights.compressedBytes, 0);
  assertEquals(resident.weights.expandedBytes, 0);
  // The sum of the three fields is weights.totalBytes
  assertEquals(resident.weights.totalBytes, 116);
  assertEquals(resident.stateBytes, 0);
});

/**
 * A merged graph with a semantic dtype that cannot run (an i32 matmul weight). It is valid in the
 * IR vocabulary, but matmul accepts only f32, so `createSessionFromContainer` always fails for lack
 * of capability.
 *
 * NOTE: it cannot be built from the storage codec side — every codec in the ledger can run (bf16
 * was opened in ADR 0131).
 */
const unsupportedGraph = (): IrGraph => {
  const declaration = plainGraph();
  declaration.values["w"] = { dtype: "i32", shape: [4, 3] };
  return mergeGraph(declaration, { w: "i32" });
};

test("returns no estimate for a semantic dtype that the real build rejects", () => {
  // Calls the full surface (`estimateGraphMemory`) directly — going through `PreparedModel` would
  // let the check on the `prepareContainer` side fail first, and the test would be green even if
  // the estimator's own check were a no-op.
  const graph = unsupportedGraph();
  assertThrows(
    () => estimateGraphMemory(graph, planWeightResidency(graph), { bindings: { T: 7 } }),
    RuntimeSupportError,
    "Missing runtime capabilities",
  );
});

test("io is the input buffers + the output readback staging (0 elements floor at 4 bytes)", () => {
  const model = plainModel();
  // x[7,4]=112 + h[7,3]=84 + g[2,3]=24
  assertEquals(runScenario(model.estimate({ bindings: { T: 7 } })).ioBytes, 220);
  // At T=0 both x and h have 0 elements → the 4-byte floor each. g stays 24
  assertEquals(runScenario(model.estimate({ bindings: { T: 0 } })).ioBytes, 32);
});

test("with budget 0, peakAccountedBytes is the resident total + the largest scenario (the old accounting)", () => {
  // Budget 0 = always holding only one slot backing (ADR 0095 decision 1). Only in this form does
  // "accounted = resident + the single largest scenario" hold.
  const report = plainModel().estimate({
    bindings: { T: 7 },
    planBackingBudgetBytes: 0,
  });
  const scenario = runScenario(report);
  assertEquals(
    report.peakAccountedBytes,
    report.resident.weights.totalBytes +
      report.resident.stateBytes +
      scenario.ioBytes +
      scenario.workspaceBytes,
  );
  // 116 (uncompressed resident) + 0 (state) + 220 (io) + 108 (workspace — see the generation-less
  // estimate below)
  assertEquals(report.peakAccountedBytes, 444);
});

// ---------------------------------------------------------------------------
// The slot backing retention budget (ADR 0095)
// ---------------------------------------------------------------------------

test("with the default budget, peakAccountedBytes is resident + budget (a form whose scenarios fit in the budget)", () => {
  const report = plainModel().estimate({ bindings: { T: 7 } });
  assertEquals(report.planBackingBudgetBytes, DEFAULT_PLAN_BACKING_BUDGET_BYTES);
  // The largest scenario, 328 (io 220 + workspace 108), fits in the 256 MiB budget, so the budget
  // is what stands on the accounted side (the retained set never exceeds it = an accounted peak
  // that errs on the large side).
  assertEquals(report.peakAccountedBytes, 116 + DEFAULT_PLAN_BACKING_BUDGET_BYTES);
  assertEquals(report.auxiliaryBytes, undefined, "a single Session reports no auxiliary resources");
});

test("the budget does not move the per-scenario numbers (it affects only the peak)", () => {
  const at = (planBackingBudgetBytes: number): AdmissionReport =>
    plainModel().estimate({ bindings: { T: 7 }, planBackingBudgetBytes });
  // The budget is an amount on the "how many are retained" side, so it does not affect what one
  // shape requires (io / workspace / resident).
  assertEquals(at(0).scenarios, at(DEFAULT_PLAN_BACKING_BUDGET_BYTES).scenarios);
  assertEquals(at(0).resident, at(DEFAULT_PLAN_BACKING_BUDGET_BYTES).resident);
});

test("when the budget is smaller than the largest scenario, the largest scenario stands (the max switches)", () => {
  const peak = (planBackingBudgetBytes: number): number =>
    plainModel().estimate({ bindings: { T: 7 }, planBackingBudgetBytes }).peakAccountedBytes;
  // The largest scenario is 328 (io 220 + workspace 108). A new backing that alone exceeds the
  // budget retires all the others and is kept alone, so a scenario larger than the budget stands on
  // the accounted side as it is.
  assertEquals(peak(0), 444);
  assertEquals(peak(327), 444);
  assertEquals(peak(328), 444);
  // From a budget above 328 the budget stands (it switches at the boundary byte).
  assertEquals(peak(329), 116 + 329);
});

test("the report carries the budget it used as is (the caller can read it without recomputing)", () => {
  assertEquals(
    plainModel().estimate({ bindings: { T: 7 }, planBackingBudgetBytes: 4096 })
      .planBackingBudgetBytes,
    4096,
  );
  assertEquals(
    plainModel().estimate({ bindings: { T: 7 }, planBackingBudgetBytes: 0 }).planBackingBudgetBytes,
    0,
  );
});

test("the budget's value range is the same check as createSessionFromContainer (anything other than a non-negative safe integer fails loudly)", () => {
  // MUST: if only the estimate accepted "a budget a Session cannot be built with", it would
  // estimate and then the build would fail.
  for (const budget of [-1, 1.5, Number.NaN, Number.MAX_SAFE_INTEGER + 2]) {
    assertThrows(
      () =>
        plainModel().estimate({
          bindings: { T: 7 },
          planBackingBudgetBytes: budget,
        }),
      ExecutionError,
      "non-negative safe integer",
    );
  }
  // Control: 0 is a valid value (the old capacity 1) — proof that the four above are not "fails
  // whatever is passed".
  assertEquals(
    plainModel().estimate({ bindings: { T: 7 }, planBackingBudgetBytes: 0 }).peakAccountedBytes,
    444,
  );
});

test("the diagnostic for the budget's value range does not call the user's conversion, and shows only the type name for non-numbers", () => {
  let conversions = 0;
  const object = {
    [Symbol.toPrimitive](): number {
      conversions++;
      return 4096;
    },
  };
  for (const value of [object, Symbol("budget"), "4096"]) {
    const options = { bindings: { T: 7 } };
    Object.defineProperty(options, "planBackingBudgetBytes", { value });
    const refused = assertThrows(
      () => plainModel().estimate(options),
      ExecutionError,
      "non-negative safe integer",
    );
    const shown = typeof value === "string" ? JSON.stringify(value) : typeof value;
    assert(refused.message.includes(`planBackingBudgetBytes ${shown} `), refused.message);
  }
  assertEquals(conversions, 0);
});

/**
 * A graph where the **same storage** can be placed on a linear weight (eligible) and on a mul
 * operand (ineligible). An IR v2 declaration carries no storage, so the supply side decides which
 * codec goes where.
 */
const twoPathGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 3] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
    g: {},
  },
  values: {
    w: { dtype: "f32", shape: [3, 3] },
    b: { dtype: "f32", shape: [3] },
    g: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [2, 3] },
    y: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "g"], outs: ["y"], attrs: {} },
  ],
});

test("f16 is resident rounded up to 4 bytes when eligible, and f32-expanded when ineligible", () => {
  const model = openGraph(twoPathGraph(), [
    weight("b", f32Bytes([0, 0, 0])),
    weight("w", f16Zeros(9), { codec: "f16" }),
    weight("g", f16Zeros(3), { codec: "f16" }),
  ]);
  const { weights } = model.estimate().resident;
  // w has numel 9 → 18 bytes, rounded up to 4 bytes = 20
  assertEquals(weights.compressedBytes, 20);
  // b stays f32, 12 (the uncompressed field)
  assertEquals(weights.uncompressedBytes, 12);
  // g is a mul operand, so ineligible → 3×4
  assertEquals(weights.expandedBytes, 12);
});

test("eligible i8 is numel rounded up to 4 bytes + per-channel scale (odd numel)", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 5] }],
    outputs: ["y"],
    initializers: {
      w: {},
      b: {},
    },
    values: {
      w: { dtype: "f32", shape: [3, 5] },
      b: { dtype: "f32", shape: [3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const model = openGraph(graph, [
    weight("b", f32Bytes([0, 0, 0])),
    // Per-channel means groupSize = row length and scale is `[rows, 1]` worth of f32 (the same
    // bytes as v1's keepdim shape, with only the shape dropped).
    weight("w", new Uint8Array(new ArrayBuffer(15)), {
      codec: "int8-sym",
      groupSize: 5,
      scale: { bytes: f32Bytes([1, 1, 1]), dtype: "f32" },
    }),
  ]);
  const report = model.estimate();
  // w numel 15 → 16 (rounded up) + scale 3×4=12
  assertEquals(report.resident.weights.compressedBytes, 28);
  assertEquals(report.resident.weights.uncompressedBytes, 12);
  assertEquals(report.resident.weights.expandedBytes, 0);
  // x[2,5]=40 + y[2,3]=24
  assertEquals(runScenario(report).ioBytes, 64);
});

test("eligible i4 is numel÷2 + group scale (only weight slots with a decode path)", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
    outputs: ["y"],
    initializers: {
      w: {},
      b: {},
    },
    values: {
      w: { dtype: "f32", shape: [2, 16] },
      b: { dtype: "f32", shape: [2] },
      y: { dtype: "f32", shape: [2, 2] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const model = openGraph(graph, [
    weight("b", f32Bytes([0, 0])),
    weight("w", new Uint8Array(new ArrayBuffer(16)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes([1, 1]), dtype: "f32" },
    }),
  ]);
  const { weights } = model.estimate().resident;
  // w numel 32 → 16 bytes + group scale 2×4=8
  assertEquals(weights.compressedBytes, 24);
  // b stays f32, 8
  assertEquals(weights.uncompressedBytes, 8);
  assertEquals(weights.expandedBytes, 0);
});

test("an i4 embedding also stays packed while resident (the embedding addendum to ADR 0069 decision 5)", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y"],
    initializers: {
      w: {},
    },
    values: {
      w: { dtype: "f32", shape: [4, 16] },
      y: { dtype: "f32", shape: [2, 16] },
    },
    nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  const model = openGraph(graph, [
    weight("w", new Uint8Array(new ArrayBuffer(32)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
  ]);
  const { weights } = model.estimate().resident;
  // Vocabulary table numel 64 → 32 bytes + group scale 4×4=16 (the embedding kernel has a decode
  // path too)
  assertEquals(weights.compressedBytes, 48);
  assertEquals(weights.uncompressedBytes, 0);
  assertEquals(weights.expandedBytes, 0);
});

/** A graph with a single `conv1d(x, w, b)` (w is i4 + a rank-2 group scale — step J-5b). */
const i4Conv1dGraph = (groups: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 32, 6] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
  },
  values: {
    // Row length = Cin/groups · K (with groups == 1, 32·2 = 64 = four g16 groups)
    w: { dtype: "f32", shape: [4, 32 / groups, 2] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 5] },
  },
  nodes: [
    {
      op: "conv1d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0, dilation: 1, groups },
    },
  ],
});

/**
 * The storage of `w` in {@link i4Conv1dGraph} (used by the path that builds the merged graph
 * directly).
 */
const I4_CONV1D_STORAGE: StorageMap = { w: { codec: "int4-sym-g", groupSize: 16 } };

test("an i4 conv1d (groups==1) also stays packed while resident (the conv1d addendum to ADR 0069 decision 5)", () => {
  const model = openGraph(i4Conv1dGraph(1), [
    weight("b", f32Bytes([0, 0, 0, 0])),
    // The scale is the rank-independent rank-2 shape `[Cout, (Cin·K)/g]` = [4,4] (the weight is
    // rank 3)
    weight("w", new Uint8Array(new ArrayBuffer(128)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes(new Array(16).fill(1)), dtype: "f32" },
    }),
  ]);
  const { weights } = model.estimate().resident;
  // w numel 256 → 128 bytes + group scale 16×4 = 64
  assertEquals(weights.compressedBytes, 192);
  // b stays f32, 16
  assertEquals(weights.uncompressedBytes, 16);
  assertEquals(weights.expandedBytes, 0);
});

test("an i4 conv1d (groups>1) is ineligible and goes to f32 expansion", () => {
  // The direct kernel has no decode path (groups > 1 does not go to igemm — step J-5b).
  const model = openGraph(i4Conv1dGraph(2), [
    weight("b", f32Bytes([0, 0, 0, 0])),
    weight("w", new Uint8Array(new ArrayBuffer(64)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes(new Array(8).fill(1)), dtype: "f32" },
    }),
  ]);
  const { weights } = model.estimate().resident;
  assertEquals(weights.compressedBytes, 0);
  assertEquals(weights.uncompressedBytes, 16);
  // w numel 128 → 512 bytes with f32 expansion
  assertEquals(weights.expandedBytes, 128 * 4);
});

test("an i4 weight that became a graph output is ineligible and goes to f32 expansion", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y", "w"],
    initializers: {
      w: {},
    },
    values: {
      w: { dtype: "f32", shape: [4, 16] },
      y: { dtype: "f32", shape: [2, 16] },
    },
    nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  const model = openGraph(graph, [
    weight("w", new Uint8Array(new ArrayBuffer(32)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
  ]);
  const { weights } = model.estimate().resident;
  assertEquals(weights.compressedBytes, 0);
  assertEquals(weights.uncompressedBytes, 0);
  assertEquals(weights.expandedBytes, 64 * 4);
});

/**
 * A form where all five `seat` values (`raw` / `f16` / `i8` / `i4` / `expanded`) **live in one
 * graph**.
 *
 * Three linear stages put the three compressed seats on weight slots (the three biases are `raw`),
 * and f16 on the operand of the final `mul` makes `expanded`. Each test above checks the fields
 * with a separate graph per seat, so nobody checks the exhaustiveness claim "every element of
 * `planWeightBuffers` is counted in exactly one of the three fields" = adding a seat and forgetting
 * to fix the branch of `weightEstimate` goes red.
 */
const allSeatsModel = (): PreparedModel => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "mul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
    outputs: ["y"],
    initializers: {
      w4: {},
      b4: {},
      w8: {},
      b8: {},
      w16: {},
      b16: {},
      g: {},
    },
    values: {
      w4: { dtype: "f32", shape: [4, 16] },
      b4: { dtype: "f32", shape: [4] },
      w8: { dtype: "f32", shape: [4, 4] },
      b8: { dtype: "f32", shape: [4] },
      w16: { dtype: "f32", shape: [4, 4] },
      b16: { dtype: "f32", shape: [4] },
      g: { dtype: "f32", shape: [4] },
      h: { dtype: "f32", shape: [2, 4] },
      i: { dtype: "f32", shape: [2, 4] },
      j: { dtype: "f32", shape: [2, 4] },
      y: { dtype: "f32", shape: [2, 4] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w4", "b4"], outs: ["h"], attrs: {} },
      { op: "linear", ins: ["h", "w8", "b8"], outs: ["i"], attrs: {} },
      { op: "linear", ins: ["i", "w16", "b16"], outs: ["j"], attrs: {} },
      // Not a weight-slot consumption, so only this f16 goes to the f32-expanded seat.
      { op: "mul", ins: ["j", "g"], outs: ["y"], attrs: {} },
    ],
  };
  // Mixes four codecs (f32 / f16 / int8-sym / int4-sym-g) into one supply.
  return openGraph(graph, [
    weight("b4", f32Bytes([0, 0, 0, 0])),
    weight("b8", f32Bytes([0, 0, 0, 0])),
    weight("b16", f32Bytes([0, 0, 0, 0])),
    weight("w16", f16Zeros(16), { codec: "f16" }),
    weight("g", f16Zeros(4), { codec: "f16" }),
    weight("w8", new Uint8Array(new ArrayBuffer(16)), {
      codec: "int8-sym",
      groupSize: 4,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
    weight("w4", new Uint8Array(new ArrayBuffer(32)), {
      codec: "int4-sym-g",
      groupSize: 16,
      scale: { bytes: f32Bytes([1, 1, 1, 1]), dtype: "f32" },
    }),
  ]);
};

test("in a graph with all five seats, each resident buffer is counted in exactly one of the three fields", () => {
  const model = allSeatsModel();
  const { weights } = model.estimate().resident;

  // By hand: i4 payload 32 + i4 scale 16 + i8 payload 16 + i8 scale 16 + f16 payload 32
  assertEquals(weights.compressedBytes, 112);
  // Three biases (f32 [4])
  assertEquals(weights.uncompressedBytes, 48);
  // The mul operand f16 [4] f32-expanded = 16 (a field counted in declared bytes before alignment)
  assertEquals(weights.expandedBytes, 16);
  assertEquals(weights.totalBytes, 176);

  // The exhaustiveness check: if even one buffer did not go through the seat branch (the switch in
  // weightEstimate), the total would not match the planner's total. Adding a seat and forgetting to
  // fix the branch goes red.
  const buffers = planWeightBuffers(planWeightResidency(model.graph));
  assertEquals(
    new Set(buffers.map((buffer) => buffer.seat)).size,
    5,
    "the five seats do not coexist",
  );
  assertEquals(
    weights.totalBytes,
    buffers.reduce(
      (sum, buffer) =>
        sum +
        (buffer.seat === "expanded" && buffer.kind === "payload"
          ? buffer.declaredBytes
          : buffer.byteLength),
      0,
    ),
  );
});

// ---------------------------------------------------------------------------
// State slots
// ---------------------------------------------------------------------------

test("state is the slot total with symbolic capacities resolved + the logical-length uniform", () => {
  const report = stateModel().estimate({
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  // k 1×2×8×4=64 elements + v 1×2×6×4=48 elements → (64+48)×4 = 448, logical-length uniform 16
  assertEquals(report.resident.stateBytes, 464);
});

test("the symbolic capacity of state moves only with generation.bindings", () => {
  const model = stateModel();
  const at = (capacity: number): number =>
    model.estimate({
      bindings: { T: 2 },
      generation: { chunkLength: 4, bindings: { C: capacity } },
    }).resident.stateBytes;
  // Only C moves (v's 48 elements + uniform 16 are fixed)
  assertEquals(at(16), (1 * 2 * 16 * 4 + 48) * 4 + 16);
  assertEquals(at(1), (1 * 2 * 1 * 4 + 48) * 4 + 16);
});

test("state is not counted without generation", () => {
  const report = plainModel().estimate({ bindings: { T: 7 } });
  assertEquals(report.resident.stateBytes, 0);
});

// ---------------------------------------------------------------------------
// Scenarios (the two execution shapes of ADR 0066 decision 4)
// ---------------------------------------------------------------------------

/**
 * A graph whose physical chunk rows are **symbolic** (the input `h` of `state_append` is
 * `[1,2,M,4]`).
 *
 * The `chunk` of `stateGraph` is a numeric dimension, so this is the one whose shape moves between
 * prefill / decode. `M` also appears in an input (it is not a states-only symbol), so only this
 * form can observe that the binding point is on the chunkLength side.
 */
const chunkSymbolGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "state_append"] },
  symbols: ["M", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 2, "M", 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [1, 2, "M", 4] },
    y: { dtype: "f32", shape: [1, 2, "M", 4] },
  },
  states: { k: { dtype: "f32", shape: [1, 2, "C", 4] } },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "state_append", ins: ["h"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "neg", ins: ["h"], outs: ["y"], attrs: {} },
  ],
});

test("with generation, the two scenarios prefill / decode are derived automatically (only the chunk symbol moves)", () => {
  const report = openGraph(chunkSymbolGraph()).estimate({
    generation: { chunkLength: 4, bindings: { C: 8 } },
    // Budget 0 = one retained. Only the way scenarios combine (max, not sum) is of interest, so it
    // is measured in a form where the budget term does not stand (the budget itself is checked in
    // the "slot backing retention budget" section).
    planBackingBudgetBytes: 0,
  });
  // prefill is M=4: x / y are both 1×2×4×4=32 elements → io 128+128, intermediates are h and y at
  // 128 each (h is consumed by append and by the second neg, so it has not returned to the pool
  // when y is allocated). decode is M=1: everything is 1/4, 8 elements' worth.
  assertEquals(report.scenarios, [
    { name: "prefill", ioBytes: 256, workspaceBytes: 256 },
    { name: "decode", ioBytes: 64, workspaceBytes: 64 },
  ]);
  // k 1×2×8×4=64 elements ×4 = 256 + logical-length uniform 16
  assertEquals(report.resident.stateBytes, 272);
  assertEquals(report.resident.weights.totalBytes, 0);
  // Resident 272 + max(512, 128) = 784. Adding them up would give 912 (one backing at a time).
  assertEquals(report.peakAccountedBytes, 784);
});

test("in a graph whose chunk rows are a numeric dimension, prefill and decode give the same numbers", () => {
  // The append input of `stateGraph` is the initializer `[1,2,4,4]`. A decode step of a form that
  // does not move symbolically runs with the same physical chunk rows as prefill and
  // queryLength=1 (assertGenerationRun allows rows === chunkLength in decode too), so the two
  // being equal is correct.
  const report = stateModel().estimate({
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  const [prefill, decode] = report.scenarios;
  assertEquals(prefill.name, "prefill");
  assertEquals(decode.name, "decode");
  assertEquals(prefill.ioBytes, decode.ioBytes);
  assertEquals(prefill.workspaceBytes, decode.workspaceBytes);
});

test("a form whose fixed numeric chunk rows are not a row count the context allows is rejected by the estimate too", () => {
  // The physical chunk rows of the append input are fixed at 4. A context with chunkLength 8 allows
  // {1, 8}, so run always rejects it on the first call (assertGenerationRun) — the estimate alone
  // must not return a normal value.
  assertThrows(
    () =>
      stateModel().estimate({
        bindings: { T: 2 },
        generation: { chunkLength: 8, bindings: { C: 8 } },
      }),
    ExecutionError,
    "physical chunk row count 4 of the state node does not match the fixed chunk contract",
  );
  // Control: including 4 in the explicit buckets makes a form run accepts, so the estimate returns
  // numbers too.
  const report = stateModel().estimate({
    bindings: { T: 2 },
    generation: { chunkLength: 8, chunkBuckets: [4], bindings: { C: 8 } },
  });
  assertEquals(report.resident.stateBytes, 464);
});

test("an estimate without generation has a single scenario (io / intermediates derived as before)", () => {
  const report = plainModel().estimate({ bindings: { T: 7 } });
  // x[7,4]=112 + h[7,3]=84 + g[2,3]=24 = 220 / the intermediates h and g live together (both are
  // graph outputs). Behind h, g would sit at offset 256 (256-aligned — ADR 0093 decision 1) and
  // grow h's region by 196 B, not less than g's own 24 B, so g opens a region of its own: 84 + 24 =
  // 108.
  assertEquals(report.scenarios, [{ name: "run", ioBytes: 220, workspaceBytes: 108 }]);
});

test("the chunk-row symbol is not accepted in options.bindings (the binding points are chunkLength and decode's 1)", () => {
  assertThrows(
    () =>
      openGraph(chunkSymbolGraph()).estimate({
        bindings: { M: 4 },
        generation: { chunkLength: 4, bindings: { C: 8 } },
      }),
    ExecutionError,
    "physical chunk-row symbol M",
  );
});

test("a graph with a chunk symbol is not estimated without generation", () => {
  assertThrows(
    () => openGraph(chunkSymbolGraph()).estimate(),
    ExecutionError,
    "bound by options.generation.chunkLength",
  );
});

// ---------------------------------------------------------------------------
// Extra scenarios (`generation.scenarios` — the verify shape of speculative decoding)
// ---------------------------------------------------------------------------

/**
 * The smallest graph with the same form as the verify shape — on a body with physical chunk rows
 * `M`, two outlets carrying **the row-selection symbol `R`** (`logits[R,V]` / `hidden[R,H]`) and an
 * index input `rows[R]` are attached (the distribution has `last_row[R]` → `logits[1,R,V]` +
 * `hidden[1,R,H]` — ADR 0096).
 *
 * `R` is neither a states-only symbol nor the physical chunk-row symbol, so its binding point is
 * on the `options.bindings` side (= the axis an extra scenario overrides with `bindings`). `V=6` /
 * `H=4` have different values so that the growth of the three tensors can be told apart by
 * distinct constants.
 */
const verifyRowsGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "state_append", "embedding"] },
  symbols: ["M", "C", "R"],
  inputs: [
    { name: "x", dtype: "f32", shape: [1, 2, "M", 4] },
    { name: "rows", dtype: "i32", shape: ["R"] },
  ],
  outputs: ["y", "logits", "hidden"],
  initializers: {
    vocab: {},
    hid: {},
  },
  values: {
    vocab: { dtype: "f32", shape: [8, 6] },
    hid: { dtype: "f32", shape: [8, 4] },
    h: { dtype: "f32", shape: [1, 2, "M", 4] },
    y: { dtype: "f32", shape: [1, 2, "M", 4] },
    logits: { dtype: "f32", shape: ["R", 6] },
    hidden: { dtype: "f32", shape: ["R", 4] },
  },
  states: { k: { dtype: "f32", shape: [1, 2, "C", 4] } },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "state_append", ins: ["h"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "neg", ins: ["h"], outs: ["y"], attrs: {} },
    { op: "embedding", ins: ["vocab", "rows"], outs: ["logits"], attrs: { padding_idx: -1 } },
    { op: "embedding", ins: ["hid", "rows"], outs: ["hidden"], attrs: { padding_idx: -1 } },
  ],
});

const verifyRowsModel = (): PreparedModel =>
  openGraph(verifyRowsGraph(), [
    weight("vocab", f32Bytes(new Array(48).fill(0))),
    weight("hid", f32Bytes(new Array(32).fill(0))),
  ]);

/**
 * An estimate with an extra scenario (`R` defaults to 1 = the same row count as non-speculative
 * decode).
 */
const verifyRowsReport = (
  scenarios: readonly { name: string; chunkLength: number; bindings?: { R: number } }[],
  planBackingBudgetBytes?: number,
): AdmissionReport =>
  verifyRowsModel().estimate({
    bindings: { R: 1 },
    generation: { chunkLength: 4, bindings: { C: 8 }, scenarios },
    planBackingBudgetBytes,
  });

test("generation.scenarios follow the two existing shapes in declaration order (same computation path, bindings overridden)", () => {
  const report = verifyRowsReport([
    { name: "verify", chunkLength: 4, bindings: { R: 4 } },
    { name: "verify-tail", chunkLength: 1 },
  ]);
  assertEquals(
    report.scenarios.map((scenario) => scenario.name),
    ["prefill", "decode", "verify", "verify-tail"],
  );
  const [prefill, decode, verify, tail] = report.scenarios;
  // io = x / y at 1×2×M×4 elements each (32M bytes) + rows 4R + logits 6R×4 + hidden 4R×4
  //    = 64M + 44R. prefill is M=4 / R=1, decode is M=1 / R=1.
  assertEquals(prefill.ioBytes, 64 * 4 + 44);
  assertEquals(decode.ioBytes, 64 + 44);
  // Raising R to 4 grows only the three tensors on the row axis: the two outlets
  // (R−1)·(V+H)·4 = 3×40 = 120 and the index input (R−1)·4 = 12. M is the same as prefill, so the
  // body does not move by a single byte.
  assertEquals(verify.ioBytes, prefill.ioBytes + 120 + 12);
  assertEquals(verify.ioBytes, 432);
  // Negative control: without overriding the binding (R stays 1), it does not differ by a single
  // byte from a decode with the same chunk rows.
  assertEquals(tail, { ...decode, name: "verify-tail" });
});

test("an extra scenario's name, physical chunk rows and bindings go through the same checks as the two existing shapes", () => {
  // A collision with a reserved name ("run" included, which is not listed with generation) fails
  // loudly.
  for (const name of ["run", "prefill", "decode"]) {
    assertThrows(
      () => verifyRowsReport([{ name, chunkLength: 2 }]),
      ExecutionError,
      "the reserved names are run / prefill / decode",
    );
  }
  // A duplicate among the extra names goes through the same check (no form where a caller looking
  // up by name sees only the first).
  assertThrows(
    () =>
      verifyRowsReport([
        { name: "verify", chunkLength: 2 },
        { name: "verify", chunkLength: 3 },
      ]),
    ExecutionError,
    "already exists",
  );
  // Physical chunk rows are an integer in 1..generation.chunkLength.
  for (const chunkLength of [0, -1, 1.5, 5]) {
    assertThrows(
      () => verifyRowsReport([{ name: "verify", chunkLength }]),
      ExecutionError,
      "is not an integer in 1..4 (generation.chunkLength)",
    );
  }
  // Both ends exactly pass.
  assertEquals(verifyRowsReport([{ name: "verify", chunkLength: 1 }]).scenarios.length, 3);
  assertEquals(verifyRowsReport([{ name: "verify", chunkLength: 4 }]).scenarios.length, 3);
  // Binding overrides go through the same check as the two existing shapes — the physical
  // chunk-row symbol is not accepted (its binding point is chunkLength).
  assertThrows(
    () =>
      verifyRowsModel().estimate({
        bindings: { R: 1 },
        generation: {
          chunkLength: 4,
          bindings: { C: 8 },
          scenarios: [{ name: "verify", chunkLength: 2, bindings: { M: 2 } }],
        },
      }),
    ExecutionError,
    "physical chunk-row symbol M",
  );
});

test("an extra scenario's physical chunk rows are also limited to the row counts the context allows", () => {
  // A context with chunkLength 4 and no buckets allows {1, 4}. A two-row run is a form execution
  // rejects.
  const at = (chunkBuckets?: readonly number[]): AdmissionReport =>
    verifyRowsModel().estimate({
      bindings: { R: 1 },
      generation: {
        chunkLength: 4,
        chunkBuckets,
        bindings: { C: 8 },
        scenarios: [{ name: "verify", chunkLength: 2 }],
      },
    });
  assertThrows(
    () => at(),
    ExecutionError,
    "physical chunk row count 2 of the state node does not match the fixed chunk contract",
  );
  // Control: with a context that declares 2 as a bucket, run accepts it, so the estimate returns
  // all three too.
  assertEquals(at([2]).scenarios.length, 3);
});

test("extra scenarios also enter the maximum of peakAccountedBytes (budget 0)", () => {
  const without = verifyRowsModel().estimate({
    bindings: { R: 1 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
    planBackingBudgetBytes: 0,
  });
  // Same chunk rows as prefill with only R at 8 rows → io and intermediates both larger than
  // prefill.
  const withVerify = verifyRowsReport([{ name: "verify", chunkLength: 4, bindings: { R: 8 } }], 0);
  // Resident memory (weights, state) does not depend on the run shape, so an extra scenario does
  // not move it by a single byte.
  assertEquals(withVerify.resident, without.resident);
  const verify = withVerify.scenarios[2];
  const resident = withVerify.resident.weights.totalBytes + withVerify.resident.stateBytes;
  // Budget 0 = one retained, so what stands at the peak is resident + the largest scenario = the
  // verify shape.
  assertEquals(withVerify.peakAccountedBytes, resident + verify.ioBytes + verify.workspaceBytes);
  assert(
    withVerify.peakAccountedBytes > without.peakAccountedBytes,
    "the extra scenario does not affect the peak",
  );
});

// ---------------------------------------------------------------------------
// Intermediate (transient) peak
// ---------------------------------------------------------------------------

/**
 * A serial chain of unary ops (`x → t1 → t2 → y`). Replacing `outputs` shows the effect of pinning.
 */
const chainGraph = (outputs: readonly string[]): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: [...outputs],
  initializers: {},
  values: {
    t1: { dtype: "f32", shape: [64] },
    t2: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["t1"], attrs: {} },
    { op: "neg", ins: ["t1"], outs: ["t2"], attrs: {} },
    { op: "neg", ins: ["t2"], outs: ["y"], attrs: {} },
  ],
});

/**
 * A form that fans out into three and folds back (four live at once).
 * A second graph output z (1,024 B) is written after the fan-out: its region is free before z is
 * born, so the fan-out's slots can stack in it — this is where the read / write rule costs a slot.
 */
const fanOutGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "add"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [64] },
    { name: "w", dtype: "f32", shape: [256] },
  ],
  outputs: ["y", "z"],
  initializers: {},
  values: {
    t1: { dtype: "f32", shape: [64] },
    t2: { dtype: "f32", shape: [64] },
    t3: { dtype: "f32", shape: [64] },
    s: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
    z: { dtype: "f32", shape: [256] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["t1"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t2"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t3"], attrs: {} },
    { op: "add", ins: ["t1", "t2"], outs: ["s"], attrs: {} },
    { op: "add", ins: ["s", "t3"], outs: ["y"], attrs: {} },
    { op: "neg", ins: ["w"], outs: ["z"], attrs: {} },
  ],
});

test("the peak of a serial chain is two live at once (an intermediate whose consumers are done is released)", () => {
  // 64 elements = 256 bytes. When t2 is allocated, two are live = 512
  assertEquals(runScenario(openGraph(chainGraph(["y"])).estimate()).workspaceBytes, 512);
});

test("a wide fan-out totals 1,536 B: the read / write rule costs one 256 B slot over the live peak of 1,280 B", () => {
  // Intervals: t1 [0,4) / t2 [1,5) / t3 [2,8) / s [3,7) / y [6,end) / z [9,end), 256 B each but z
  // 1,024 B. When s is allocated, t1 / t2 / t3 / s live together (4 × 256 = 1,024); the live peak
  // is y + z = 1,280. Both orders place z first (region 0, 1,024 B), and t1 / t2 / t3 stack in it
  // without growing it (z is not born yet): at 0 / 256 / 512 in order 1, and with t3 (the longest
  // lived) first at 0, then t1 / t2 at 256 / 512, in order 2. s would fit at 768, but the dispatch
  // that writes s reads t1 / t2, so s opens region 1; y reads s and t3, so it avoids both regions
  // and opens region 2: 1,024 + 256 + 256 = 1,536 in both orders (the tie keeps order 1) — one slot
  // over the live peak (the usage-scope rule, ADR 0093 decision 1; without it s takes 768 and the
  // total is 1,280).
  assertEquals(runScenario(openGraph(fanOutGraph()).estimate()).workspaceBytes, 1536);
});

test("a graph output stays pinned even after its consumers are done (not released)", () => {
  // Putting t1 in graph.outputs makes t1 stay after t2 is consumed, so three overlap
  const report = openGraph(chainGraph(["t1", "y"])).estimate();
  assertEquals(runScenario(report).workspaceBytes, 768);
});

test("initializers and graph inputs are not counted in the intermediate peak (they are accounted on the weights / io side)", () => {
  // Only matmul's h (T=7 → 84) and embedding's g (24, in a region of its own — see the
  // generation-less estimate) are intermediates. x / w / emb / idx are not counted.
  const report = plainModel().estimate({ bindings: { T: 7 } });
  assertEquals(runScenario(report).workspaceBytes, 108);
});

test("reallocation does not depend on size (a 4-byte output takes a released 8-byte range)", () => {
  // Three stages of 8 → 12 → 4 bytes. t2 is written by the dispatch that reads t1, so it is a
  // separate region (12 / 8). y reads t2, so it cannot sit in t2's region and takes t1's range
  // (8 bytes), whose lifetime has ended = 12 + 8 = 20 (it was 24 with the old pool's exact-size
  // reuse — ADR 0093 decision 8).
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
    outputs: ["y"],
    initializers: {
      w1: {},
      w2: {},
      w3: {},
    },
    values: {
      w1: { dtype: "f32", shape: [2, 2] },
      w2: { dtype: "f32", shape: [2, 3] },
      w3: { dtype: "f32", shape: [3, 1] },
      t1: { dtype: "f32", shape: [1, 2] },
      t2: { dtype: "f32", shape: [1, 3] },
      y: { dtype: "f32", shape: [1, 1] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w1"], outs: ["t1"], attrs: {} },
      { op: "matmul", ins: ["t1", "w2"], outs: ["t2"], attrs: {} },
      { op: "matmul", ins: ["t2", "w3"], outs: ["y"], attrs: {} },
    ],
  };
  const model = openGraph(graph, [
    weight("w1", f32Bytes([1, 0, 0, 1])),
    weight("w2", f32Bytes(new Array(6).fill(1))),
    weight("w3", f32Bytes([1, 1, 1])),
  ]);
  assertEquals(runScenario(model.estimate()).workspaceBytes, 20);
});

// ---------------------------------------------------------------------------
// Aliases in the intermediate peak (reshape / identity expand — ADR 0011)
// ---------------------------------------------------------------------------

/**
 * A chain of aliases (`reshape` → `reshape`) and a consumption while the chain's **root** is alive.
 *
 * `h` is consumed twice (the chain's entry and the final `add`), so during execution the root's
 * data does not return to the pool while the chain runs. Allocating the aliases as independent
 * values would grow extra slots by the amount of this overlap.
 */
const aliasChainGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "add", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    r1: { dtype: "f32", shape: [8, 8] },
    r2: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["r1"], attrs: {} },
    { op: "reshape", ins: ["r1"], outs: ["r2"], attrs: {} },
    { op: "add", ins: ["h", "r2"], outs: ["y"], attrs: {} },
  ],
});

test("a chain of aliases counts only the root's single allocation (reshape → reshape)", () => {
  // Only two allocations appear, `h` (256) and `y` (256) = 512. Allocating the two chain values
  // would prevent reuse while `h` is alive, giving 768.
  const report = openGraph(aliasChainGraph()).estimate();
  assertEquals(runScenario(report).workspaceBytes, 512);
});

/** Two nodes, `neg` → `expand`. The output shape switches between identity and non-identity. */
const aliasExpandGraph = (outShape: readonly number[]): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "expand"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [1, 4] },
    y: { dtype: "f32", shape: [...outShape] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "expand", ins: ["h"], outs: ["y"], attrs: {} },
  ],
});

test("an identity expand allocates nothing, and an expand with a replicated axis counts its materialization", () => {
  const workspaceOf = (outShape: readonly number[]): number =>
    runScenario(openGraph(aliasExpandGraph(outShape)).estimate()).workspaceBytes;
  // Identity ([1,4] → [1,4]) is an alias = only `h`'s single 16 bytes.
  assertEquals(workspaceOf([1, 4]), 16);
  // With a replicated axis ([1,4] → [3,4]) it goes back to a strided materializing copy = 16 + 48.
  assertEquals(workspaceOf([3, 4]), 64);
});

/**
 * A form where a graph output is **named by an alias**. `p` is `h`'s data itself, so the pin
 * applies to the root. Even after `h`'s consumers are done the root does not return to the pool,
 * and the later `t` / `y` take new slots.
 */
const pinnedAliasGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: ["p", "y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    p: { dtype: "f32", shape: [8, 8] },
    t: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["p"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t"], attrs: {} },
    { op: "neg", ins: ["t"], outs: ["y"], attrs: {} },
  ],
});

test("even when a graph output is named by an alias, the pin applies to the root's data (it does not return to the pool)", () => {
  // `h` (the pinned root) + `t` + `y` = 768. If the pin did not reach the root, `h` would return to
  // the pool and be reallocated to `t`, shrinking to 512 (= the error of sharing readback-able data
  // with another value).
  const report = openGraph(pinnedAliasGraph()).estimate();
  assertEquals(runScenario(report).workspaceBytes, 768);
});

// ---------------------------------------------------------------------------
// Cross-check against the execution plan (derivePlanSlots)
// ---------------------------------------------------------------------------

/**
 * The device capabilities used for the decision (WebGPU core defaults — 128MiB / 65535). Only the
 * number of row blocks in the row-block split reads them.
 */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/**
 * Plain node list → recipe list for the plan. Copies **only the bookkeeping side** of
 * recipe-builder's `#buildStep` (allocation specs, `uses` / `pinned`, the flattened consumption
 * list, dispatch reads and writes) without touching the GPU. Each node is one dispatch (reading the
 * inputs, writing the outputs), so the usage scope rule applies to the plan in the same direction
 * as in the allocation program the estimator builds.
 *
 * NOTE: the alias source is the first input, as in `#buildStep`. Only for the form pointing at an
 * initializer does `#buildStep` create `{ kind: "resident" }`, but the plan treats any value not
 * in env as "no slot", so from the plan's point of view it is equivalent to `{ kind: "value" }`.
 */
const slotRecipes = (steps: readonly ExecStep[], graph: IrGraph): readonly StepRecipe[] => {
  const uses = countUses(graph);
  const outputNames = new Set(graph.outputs);
  return steps.map((step) => {
    if (step.kind !== "node") throw new Error("copies of fused steps are not built here");
    const outputs: readonly StepOutput[] = step.plan.outputs.map(({ name, shape }) => {
      const bookkeeping = { name, uses: uses.get(name) ?? 0, pinned: outputNames.has(name) };
      return step.aliasesInput
        ? {
            ...bookkeeping,
            kind: "alias" as const,
            source: { kind: "value" as const, name: step.plan.node.ins[0] },
          }
        : { ...bookkeeping, kind: "alloc" as const, byteLength: numel(shape) * 4 };
    });
    const bindings = [
      ...step.plan.node.ins.map((name, index) => ({
        binding: index + 1,
        source: { kind: "value" as const, name },
      })),
      ...outputs.map((output, index) => ({
        binding: step.plan.node.ins.length + index + 1,
        source: { kind: "value" as const, name: output.name },
      })),
    ];
    const dispatch = {
      key: step.plan.node.op,
      bindings,
      roles: {
        reads: new Set(bindings.slice(0, step.plan.node.ins.length).map((b) => b.binding)),
        writes: new Set(bindings.slice(step.plan.node.ins.length).map((b) => b.binding)),
      },
    } as unknown as StepRecipe["dispatches"][number];
    return {
      outputs,
      temps: [],
      dispatches: step.aliasesInput ? [] : [dispatch],
      releases: step.plan.node.ins,
      writesState: false,
    };
  });
};

/**
 * A form mixing three kinds of aliases (chain / identity expand / pinned alias) and a non-identity
 * expand. It is built so that **it matches none of the five rules' leading ops** (`reshape` is the
 * leading op of upsample2x, but the following sequence does not match, so no window is taken — the
 * test body checks this mechanically).
 */
const aliasMixGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "add", "reshape", "expand"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [64] },
    { name: "z", dtype: "f32", shape: [1, 4] },
    { name: "q", dtype: "f32", shape: [5] },
  ],
  outputs: ["v", "yv", "kv", "m", "p1", "p2"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    r1: { dtype: "f32", shape: [8, 8] },
    r2: { dtype: "f32", shape: [64] },
    e1: { dtype: "f32", shape: [64] },
    g: { dtype: "f32", shape: [1, 4] },
    w: { dtype: "f32", shape: [3, 4] },
    v: { dtype: "f32", shape: [3, 4] },
    s: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
    yv: { dtype: "f32", shape: [8, 8] },
    k: { dtype: "f32", shape: [5] },
    kv: { dtype: "f32", shape: [1, 5] },
    m: { dtype: "f32", shape: [5] },
    p1: { dtype: "f32", shape: [64] },
    p2: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    // A chain of aliases (the root is `h`).
    { op: "reshape", ins: ["h"], outs: ["r1"], attrs: {} },
    { op: "reshape", ins: ["r1"], outs: ["r2"], attrs: {} },
    // Identity expand (a third alias pointing at the same root).
    { op: "expand", ins: ["h"], outs: ["e1"], attrs: {} },
    { op: "neg", ins: ["z"], outs: ["g"], attrs: {} },
    // With a replicated axis = a materializing copy (not an alias).
    { op: "expand", ins: ["g"], outs: ["w"], attrs: {} },
    { op: "neg", ins: ["w"], outs: ["v"], attrs: {} },
    { op: "add", ins: ["h", "r2"], outs: ["s"], attrs: {} },
    { op: "add", ins: ["s", "e1"], outs: ["y"], attrs: {} },
    // A graph output named by an alias (the pin applies to the data of the root `y`).
    { op: "reshape", ins: ["y"], outs: ["yv"], attrs: {} },
    // The same form rebuilt with the only size class in this graph (20 bytes). If the pin did not
    // reach the root, `k`'s data would return to the pool and be reallocated to `m`, one slot
    // fewer.
    { op: "neg", ins: ["q"], outs: ["k"], attrs: {} },
    { op: "reshape", ins: ["k"], outs: ["kv"], attrs: {} },
    { op: "neg", ins: ["q"], outs: ["m"], attrs: {} },
    // The two trailing 256-byte values are placed where they **exactly use up** the slots the chain
    // returned. If the alias consumptions were not added to the root, one would be missing and a
    // new slot would grow here.
    { op: "neg", ins: ["x"], outs: ["p1"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["p2"], attrs: {} },
  ],
});

/**
 * Cross-checks the estimator's intermediate total against the region total of the execution plan
 * (`planRecipes` — the same packer, ADR 0093 decision 4).
 *
 * An **exact match** that can be claimed only in a form where no fusion applies — in forms a fusion
 * takes, the execution side removes intermediates and the row-block split adds temporaries, which
 * is the difference the estimator's `unaccounted` admits. If the two implementations split on how
 * aliases are handled or on the direction of dispatch reads and writes, this splits without an
 * exception (that is what issue R6V-2 looked like).
 */
test("in a form where no fusion applies, the estimator's intermediate total equals the execution plan's region total", () => {
  const model = openGraph(aliasMixGraph());
  const plan = planGraph(model.graph, {});
  const fusion = planFusions(plan.nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(model.graph),
    outputNames: new Set(model.graph.outputs),
    limits: TEST_LIMITS,
  });
  // Precondition check: if even one fusion applied, it could not be compared with the estimator's
  // accounting, which "walks the pre-fusion list".
  assertEquals(
    fusion.steps.every((step) => step.kind === "node"),
    true,
    "a fusion applies",
  );
  assertEquals(fusion.counts.identityExpand, 1);

  const transientPlan = planRecipes(slotRecipes(fusion.steps, model.graph), CORE_TRANSIENT_LIMITS);
  // The allocation ranges are the ten of h / g / w / v / s / y / k / m / p1 / p2 (the aliases r1 /
  // r2 / e1 / yv / kv grow none). The region total comes from the same packer as the estimator, so
  // it is an exact match.
  assertEquals(transientPlan.slots.length, 10);
  assertEquals(
    transientPlan.slots.map((slot) => slot.byteLength),
    [256, 16, 48, 48, 256, 256, 20, 20, 256, 256],
  );
  assertEquals(runScenario(model.estimate()).workspaceBytes, transientPlan.totalBytes);
});

// ---------------------------------------------------------------------------
// Decomposed attention (fusion into row-block attention — ADR 0060)
// ---------------------------------------------------------------------------

// The binding limit of a device granted exactly the WebGPU core default. LARGE_SCORES splits into
// two row blocks under it, so the count depends on the limit and the estimate needs it passed
// (ADR 0124 decision 7).
const CORE_LIMIT = CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;

// One score tensor = 4×2048×4200×4 = 137,625,600B > the binding limit of 128 MiB (WebGPU core
// default).
const LARGE_SCORES = { heads: 4, queries: 2048, keys: 4200, headDim: 8 } as const;

describe("intermediates of decomposed attention (counted with the buffers bound after fusion)", () => {
  describe("a form whose pre-fusion scores exceed the binding limit and are fused into row-block attention", () => {
    it("does not throw, and counts intermediates as one row block's temporaries plus the output", () => {
      const scenario = runScenario(
        openGraph(decomposedAttentionGraph(LARGE_SCORES)).estimate({
          maxStorageBufferBindingSize: CORE_LIMIT,
        }),
      );
      // One row = 4×4200×4 = 67,200B → 1,997 rows fit in the limit, so it is split evenly into 2
      // blocks (1024 + 1024 rows). The temporaries S / after mask addition / P are
      // 4×1024×4200×4 = 68,812,800B each, with at most two live at once (the reading and writing
      // sides are the same dispatch, so separate regions — the second block reuses the ranges of
      // the first).
      // The output y = 4×2048×8×4 = 262,144B is written by the dispatch that reads P, so it avoids
      // P's region; behind the masked scores (at 68,812,800B, already 256B-aligned) it would grow
      // that region by its full size, so it opens a region of its own.
      assertEquals(scenario.workspaceBytes, 68_812_800 * 2 + 262_144);
    });
  });

  describe("when intermediates exceed the binding limit in a form where fusion does not apply", () => {
    it("lists every limit overrun and fails loudly as before (not downgraded to a warning)", () => {
      assertThrows(
        () => openGraph(decomposedAttentionGraph(LARGE_SCORES, true)).estimate(),
        ExecutionError,
        "'s3_alias' 137625600B / 'masked' 137625600B / 'p4' 137625600B",
      );
    });
  });
});

// ---------------------------------------------------------------------------
// In-node temporaries of the fused attention op (ADR 0124 decision 7)
// ---------------------------------------------------------------------------

/**
 * A one-node graph of the fused attention op (no states field). q `[1,2,8,4]`, k / v `[1,2,6,4]`
 * (B·H = 2, M = 8, N = 6, D = 4). D % 4 == 0 and N % 4 != 0, so with a8 only ①QK is i8a8, and s16
 * is ineligible (degrades to f32 storage).
 */
const fusedAttentionModel = (mask = false): PreparedModel =>
  openGraph(
    singleOpDeclaration(
      "attention",
      [[1, 2, 8, 4], [1, 2, 6, 4], [1, 2, 6, 4], ...(mask ? [[1, 1, 8, 6]] : [])],
      [[1, 2, 8, 4]],
      { attrs: { scale: 0.5 } },
    ),
  );

describe("counts the in-node temporaries of the fused attention op with the same pure function as execution", () => {
  // Output y = 2×8×4×4 = 256B. ①QK writes S, ② reads S and writes the row statistics, and ③ reads
  // S and the row statistics and writes y, so S, the row statistics and y cannot share a region
  // (usage scope — 3 regions).

  it("f32 adds S 2×8×6×4 = 384B and row statistics 2×8×2×4 = 128B to the 256B output", () => {
    assertEquals(runScenario(fusedAttentionModel().estimate()).workspaceBytes, 256 + 384 + 128);
  });

  it("split by a budget, only one row block's temporaries are counted (1 row is 48B → a 192B budget is 4 rows × 2 blocks, 96B is 2 rows × 4 blocks)", () => {
    const at = (budget: number): number =>
      runScenario(fusedAttentionModel().estimate({ attentionRowBlockMaxBytes: budget }))
        .workspaceBytes;
    // 4 rows: S 2×4×6×4 = 192 / row statistics 64. 2 rows: S 96 / row statistics 32.
    assertEquals(at(192), 256 + 192 + 64);
    assertEquals(at(96), 256 + 96 + 32);
  });

  it("f16 compute counts S at 2B, and s16 in an ineligible form (N % 4 != 0) is counted with f32 storage", () => {
    assertEquals(
      runScenario(fusedAttentionModel().estimate({ attentionCompute: "f16" })).workspaceBytes,
      256 + 192 + 128,
    );
    assertEquals(
      runScenario(fusedAttentionModel().estimate({ attentionScoreStorage: "f16" })).workspaceBytes,
      256 + 384 + 128,
    );
  });

  it("a8 counts the quantization temporaries of ①QK (qq / qs / kq / ks) until the end of the node", () => {
    // qq 2×8×4 = 64, qs 2×8×4 = 64, kq 2×6×4 = 48, ks 2×6×4 = 48 (③PV stays f32 because
    // N % 4 != 0).
    // All seven slots (y, S, row statistics and the four) are live at once (nested lifetimes inside
    // the node), so the live peak is their sum and no placement is smaller; the estimate shows only
    // the total, which meets it:
    // S 384 + y 256 + row statistics 128 + qq 64 + qs 64 + kq 48 + ks 48 = 992 B.
    assertEquals(
      runScenario(fusedAttentionModel().estimate({ attentionCompute: "a8" })).workspaceBytes,
      384 + 256 + 128 + 64 + 64 + 48 + 48,
    );
  });

  it("an additive mask adds only one more read to ①QK and no temporaries", () => {
    assertEquals(runScenario(fusedAttentionModel(true).estimate()).workspaceBytes, 256 + 384 + 128);
  });

  describe("returns no numbers for combinations execution rejects (the same decision and the same message as execution)", () => {
    it("mask × a8", () => {
      assertThrows(
        () => fusedAttentionModel(true).estimate({ attentionCompute: "a8" }),
        ExecutionError,
        "attention with an additive mask cannot be combined with attentionCompute 'a8'",
      );
    });

    it("states form × anything other than f32", () => {
      assertThrows(
        () =>
          openGraph(stateAttentionGraph()).estimate({
            generation: { chunkLength: 4, bindings: { C: 16 } },
            maxStorageBufferBindingSize: 1 << 20,
            attentionScoreStorage: "f16",
          }),
        ExecutionError,
        "states-form attention cannot be combined with attentionScoreStorage 'f16'",
      );
    });

    it("spellings outside the type and c16 × s16 go through the same check as Session build", () => {
      // A spelling from a caller outside the type (a JS consumer) — 'i8a8', the name before the
      // rename.
      const options: EstimateOptions = {};
      Object.defineProperty(options, "attentionCompute", { value: "i8a8", enumerable: true });
      assertThrows(
        () => fusedAttentionModel().estimate(options),
        ExecutionError,
        "attentionCompute",
      );
      assertThrows(
        () =>
          fusedAttentionModel().estimate({
            attentionCompute: "f16",
            attentionScoreStorage: "f16",
          }),
        ExecutionError,
        "cannot be given together",
      );
    });

    it("linearCompute (it decides the linear prologue) goes through the same spelling check", () => {
      const options: EstimateOptions = {};
      Object.defineProperty(options, "linearCompute", { value: "i8a8", enumerable: true });
      assertThrows(() => fusedAttentionModel().estimate(options), ExecutionError, "linearCompute");
    });
  });

  describe("an estimate without a binding limit counts only shapes whose row-block count is the same on every device", () => {
    // q / k / v `[1,1,8192,4]`: one S block = 8,192 × 8,192 × 4 = 256 MiB (does not fit in one
    // block under the core default of 128 MiB), one row = 8,192 × 4 = 32,768B, output y = 8,192 × 4
    // × 4 = 131,072B, row statistics 8B per row.
    const wideAttention = (): PreparedModel =>
      openGraph(
        singleOpDeclaration(
          "attention",
          [
            [1, 1, 8192, 4],
            [1, 1, 8192, 4],
            [1, 1, 8192, 4],
          ],
          [[1, 1, 8192, 4]],
          { attrs: { scale: 0.5 } },
        ),
      );
    const GRANTED = 2_147_483_644;

    it("a shape that does not fit in one block under the core default fails loudly naming the limit (it does not claim a block count split by the core default)", () => {
      assertThrows(
        () => wideAttention().estimate(),
        ExecutionError,
        `do not fit in one block under the WebGPU core default binding limit (${CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize}B), so options.maxStorageBufferBindingSize is needed`,
      );
      // Even with a budget larger than the core default the result is the same, because the limit
      // changes with the device's granted value.
      assertThrows(
        () => wideAttention().estimate({ attentionRowBlockMaxBytes: 512 * 1024 * 1024 }),
        ExecutionError,
        `do not fit in one block under the WebGPU core default binding limit (${CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize}B), so options.maxStorageBufferBindingSize is needed`,
      );
    });

    it("passing the limit counts the same shape (one block with a 2 GiB-class limit)", () => {
      assertEquals(
        runScenario(wideAttention().estimate({ maxStorageBufferBindingSize: GRANTED }))
          .workspaceBytes,
        131_072 + 268_435_456 + 65_536,
      );
    });

    it("a budget at or below the core default is counted even without a limit, with the same value as with the limit (64 MiB → 2,048 rows × 4 blocks)", () => {
      const budget = 64 * 1024 * 1024;
      const without = runScenario(wideAttention().estimate({ attentionRowBlockMaxBytes: budget }));
      const withLimit = runScenario(
        wideAttention().estimate({
          maxStorageBufferBindingSize: GRANTED,
          attentionRowBlockMaxBytes: budget,
        }),
      );
      assertEquals(without.workspaceBytes, 131_072 + 67_108_864 + 16_384);
      assertEquals(without.workspaceBytes, withLimit.workspaceBytes);
    });
  });
});

// ---------------------------------------------------------------------------
// The online kernel (attentionScoreStorage "none" — ADR 0132)
// ---------------------------------------------------------------------------

describe("the online attention kernel counts the a8 quantisation temporaries and no S", () => {
  // q / k / v `[1,2,8,64]` (B·H = 2, M = N = 8, D = 64): eligible for the online kernel.
  const onlineModel = (): PreparedModel =>
    openGraph(
      singleOpDeclaration(
        "attention",
        [
          [1, 2, 8, 64],
          [1, 2, 8, 64],
          [1, 2, 8, 64],
        ],
        [[1, 2, 8, 64]],
        { attrs: { scale: 0.5 } },
      ),
    );
  const NONE = { attentionCompute: "a8", attentionScoreStorage: "none" } as const;
  const SUBGROUPS = { subgroups: true, maxComputeWorkgroupStorageSize: 49152 };

  it("one window: the total is the live peak of y 4,096 B, Vᵀ 4,096 B and the kernel's six quantised inputs", () => {
    // Temporaries: qq 2×8×64 = 1,024 / qs 2×8×4 = 64 / kq 1,024 / ks 64 / vt 2×64×8×4 = 4,096 /
    // vq 2×64×8 = 1,024 / vs 2×64×4 = 512; output y 2×8×64×4 = 4,096. No S, no row statistics.
    // The kernel reads qq / kq / vq / qs / ks / vs and writes y, so those six cannot share y's
    // region; vt is read by Vᵀ's quantisation, which writes vq / vs, so it cannot share theirs.
    // All eight are live at once (nested lifetimes inside the node), so the live peak is their sum
    // and no placement is smaller; the estimate shows only the total, which meets it:
    // y 4,096 + vt 4,096 + qq 1,024 + kq 1,024 + vq 1,024 + vs 512 + qs 64 + ks 64 = 11,904 B.
    assertEquals(
      runScenario(onlineModel().estimate({ ...NONE, attentionDevice: SUBGROUPS })).workspaceBytes,
      4096 + 4096 + 1024 + 1024 + 1024 + 512 + 64 + 64,
    );
  });

  it("a device that admits neither form counts the three a8 dispatches with S in f32", () => {
    // 16 KiB without subgroups: the route falls back, so the estimate equals the plain a8 one.
    assertEquals(
      runScenario(
        onlineModel().estimate({
          ...NONE,
          attentionDevice: { subgroups: false, maxComputeWorkgroupStorageSize: 16384 },
        }),
      ).workspaceBytes,
      runScenario(onlineModel().estimate({ attentionCompute: "a8" })).workspaceBytes,
    );
  });

  it("fails loudly without attentionDevice (the route depends on the device)", () => {
    assertThrows(
      () => onlineModel().estimate(NONE),
      ExecutionError,
      "attentionScoreStorage 'none' needs options.attentionDevice",
    );
  });

  /**
   * `attention → permute [0, 2, 1, 3]` on head-major q `[1, 2, M, 64]` and k / v `[1, 2, N, 64]`,
   * the permute's token-major output `[1, M, 2, 64]` the graph output: on the online route the
   * attention absorbs the permute and writes the token-major output itself (perf-ledger F-3 A4).
   */
  const outputPermuteModel = (rows: number, cols: number): PreparedModel =>
    openGraph({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["attention", "permute"] },
      symbols: [],
      inputs: [
        { name: "q", dtype: "f32", shape: [1, 2, rows, 64] },
        { name: "k", dtype: "f32", shape: [1, 2, cols, 64] },
        { name: "v", dtype: "f32", shape: [1, 2, cols, 64] },
      ],
      outputs: ["ot"],
      initializers: {},
      values: {
        o: { dtype: "f32", shape: [1, 2, rows, 64] },
        ot: { dtype: "f32", shape: [1, rows, 2, 64] },
      },
      nodes: [
        { op: "attention", ins: ["q", "k", "v"], outs: ["o"], attrs: { scale: 0.125 } },
        { op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } },
      ],
    });

  it("with the output permute absorbed, the total is the attention step's live peak (M 64 ≫ N 4: O outgrows the temporaries)", () => {
    // B·H = 2, M = 64, N = 4, D = 64. Temporaries: qq 2×64×64 = 8,192 / qs 2×64×4 = 512 /
    // kq 2×4×64 = 512 / ks 2×4×4 = 32 / vt 2×64×4×4 = 2,048 / vq 2×64×4 = 512 / vs 2×64×4 = 512;
    // the output ot 64×2×64×4 = 32,768, written token-major by the online kernel. As in the case
    // above all eight are live at once, so the total is their sum: 45,088 B.
    // An estimate that does not replay the absorption keeps a head-major O (32,768 B) as the
    // attention's output and adds the permute's step, where O and ot (32,768 B each) are live
    // together — at least 65,536 B.
    assertEquals(
      runScenario(outputPermuteModel(64, 4).estimate({ ...NONE, attentionDevice: SUBGROUPS }))
        .workspaceBytes,
      32_768 + 8192 + 2048 + 512 + 512 + 512 + 512 + 32,
    );
  });

  it("fails loudly without attentionDevice with the estimate's own message whatever the graph (an absorbable output permute, no attention at all)", () => {
    // The check is at the estimate's entry: the fusion pass, which also needs the device facts to
    // decide the output permute, never runs first with a message of its own.
    assertThrows(
      () => outputPermuteModel(64, 4).estimate(NONE),
      ExecutionError,
      "attentionScoreStorage 'none' needs options.attentionDevice",
    );
    assertThrows(
      () => plainModel().estimate({ ...NONE, bindings: { T: 7 } }),
      ExecutionError,
      "attentionScoreStorage 'none' needs options.attentionDevice",
    );
  });

  it("rejects 'none' without a8 with the Session's message, and malformed device facts", () => {
    assertThrows(
      () => onlineModel().estimate({ attentionScoreStorage: "none", attentionDevice: SUBGROUPS }),
      ExecutionError,
      "attentionScoreStorage 'none' (the online kernel) is implemented only with attentionCompute 'a8'",
    );
    assertThrows(
      () =>
        onlineModel().estimate({
          ...NONE,
          attentionDevice: { subgroups: true, maxComputeWorkgroupStorageSize: 0 },
        }),
      ExecutionError,
      "options.attentionDevice.maxComputeWorkgroupStorageSize 0 must be a positive safe integer",
    );
  });
});

// ---------------------------------------------------------------------------
// The budget per row-block temporary (attentionRowBlockMaxBytes — ADR 0124)
// ---------------------------------------------------------------------------

describe("the row-block budget (attentionRowBlockMaxBytes) decides the block count with the same derivation as execution", () => {
  describe("decomposed attention (a 64 MiB budget applied to the example in the ADR 0093 addendum of 2026-10-10)", () => {
    it("the 2 blocks under the 128 MiB binding limit become 3, and one row block's temporaries shrink", () => {
      const scenario = runScenario(
        openGraph(decomposedAttentionGraph(LARGE_SCORES)).estimate({
          attentionRowBlockMaxBytes: 64 * 1024 * 1024,
        }),
      );
      // One row = 4×4200×4 = 67,200B → 998 rows fit in the 67,108,864B budget, so
      // ceil(2048 / 998) = 3 blocks (683 + 683 + 682 rows). The temporaries are at most
      // 4×683×4200×4 = 45,897,600B, with two live at once (2 regions).
      // The output y 262,144B is written by the dispatch that reads P, so it avoids P's region;
      // behind the masked scores it would start at the 256B-aligned 45,897,728B (45,897,600B is not
      // a multiple of 256) and grow that region by 262,272B, more than its own size, so it opens a
      // region of its own (both orders). Down from 68,812,800 × 2 + 262,144 = 137,887,744B without
      // the budget.
      assertEquals(scenario.workspaceBytes, 45_897_600 * 2 + 262_144);
    });

    it("the value without a budget is the same as before the budget field was added (the default is unchanged)", () => {
      const scenario = runScenario(
        openGraph(decomposedAttentionGraph(LARGE_SCORES)).estimate({
          maxStorageBufferBindingSize: CORE_LIMIT,
        }),
      );
      assertEquals(scenario.workspaceBytes, 68_812_800 * 2 + 262_144);
    });
  });

  describe("states-form attention", () => {
    it("prefill (M = 4) splits into 2 blocks under a 512B budget, and decode (M = 1) stays at 1 block", () => {
      const { prefill, decode } = bothScenarios(
        openGraph(stateAttentionGraph()).estimate({
          generation: { chunkLength: 4, bindings: { C: 16 } },
          // A size where the binding limit has no effect (the same value as WIDE_LIMIT in the
          // states-form section below) — only the budget moves the block count.
          maxStorageBufferBindingSize: 1 << 20,
          attentionRowBlockMaxBytes: 512,
        }),
      );
      // B·H = 4, column capacity 16 → one row 4×16×4 = 256B. 2 rows in the 512B budget → 2 blocks.
      //   output o 512 / one S block = 4×2×16×4 = 512 / one row-statistics block = 4×2×2×4 = 64
      //   (1024 / 128 without a budget)
      assertEquals(prefill.workspaceBytes, 512 + 512 + 64);
      // decode has one row of 256B ≤ 512B, so 1 block (the same 128 + 256 + 32 as without a
      // budget).
      assertEquals(decode.workspaceBytes, 128 + 256 + 32);
    });
  });

  it("a budget smaller than one row fails loudly naming the budget, as in the execution plan derivation", () => {
    assertThrows(
      () =>
        openGraph(decomposedAttentionGraph(LARGE_SCORES)).estimate({
          attentionRowBlockMaxBytes: 67_199,
        }),
      ExecutionError,
      "set the budget to at least one row",
    );
  });

  it("the value range is the same check as Session build, and is checked even on graphs without row blocks", () => {
    for (const value of [0, -1, 1.5, Number.NaN]) {
      assertThrows(
        () => plainModel().estimate({ bindings: { T: 7 }, attentionRowBlockMaxBytes: value }),
        ExecutionError,
        `options.attentionRowBlockMaxBytes ${value} must be a positive safe integer`,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// In-node temporaries of states-form attention (score S / row statistics)
// ---------------------------------------------------------------------------

/** A size where the storage binding limit has no effect (always one row block). */
const WIDE_LIMIT = 1 << 20;

const stateAttentionReport = (options: {
  readonly capacity: number;
  readonly chunkLength: number;
  readonly window?: number;
  readonly heads?: number;
  readonly limit?: number;
}): AdmissionReport =>
  openGraph(stateAttentionGraph(options.window, options.heads)).estimate({
    generation: { chunkLength: options.chunkLength, bindings: { C: options.capacity } },
    maxStorageBufferBindingSize: options.limit ?? WIDE_LIMIT,
  });

test("the S / row statistics of states-form attention are added to the intermediates (full variant, one row block)", () => {
  const { prefill, decode } = bothScenarios(stateAttentionReport({ capacity: 16, chunkLength: 4 }));
  // B·H = 4, column capacity = C = 16. prefill (M=4) has one row of 4×16×4=256B, well within the
  // limit, so 1 block.
  //   output o [1,4,4,8]=128 elements → 512 / S = 4×4×16×4 = 1024 / row statistics = 4×4×2×4 = 128
  assertEquals(prefill.workspaceBytes, 512 + 1024 + 128);
  // io is q 512 + k 256 + v 256 + o 512
  assertEquals(prefill.ioBytes, 1536);
  // decode (M=1): o 128 / S = 4×1×16×4 = 256 / row statistics = 4×1×2×4 = 32
  assertEquals(decode.workspaceBytes, 128 + 256 + 32);
  assertEquals(decode.ioBytes, 384);
});

test("S grows in proportion to capacity (in the full variant, column capacity = slot capacity)", () => {
  const at = (capacity: number): number =>
    bothScenarios(stateAttentionReport({ capacity, chunkLength: 4 })).prefill.workspaceBytes;
  // The output 512 + row statistics 128 do not move; only S is proportional to C (1024 → 2048 →
  // 4096).
  assertEquals(at(16), 512 + 1024 + 128);
  assertEquals(at(32), 512 + 2048 + 128);
  assertEquals(at(64), 512 + 4096 + 128);
});

test("the S / row statistics of the readonly variant (a borrower's cross-attention, ins is q only) are counted too", () => {
  // The smallest drafter form (ADR 0096 stage 2): one readonly attention only reads two borrowed
  // external slots. There are no k / v inputs for the current step, so the three dispatches during
  // execution read only q and the temporaries — if this were built assuming "k / v are in ins", the
  // transient plan would fail on an undefined reference.
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention"] },
    symbols: ["C"],
    inputs: [{ name: "q", dtype: "f32", shape: [1, 4, 1, 8] }],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, 4, 1, 8] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 2, "C", 8], external: true },
      vslot: { dtype: "f32", shape: [1, 2, "C", 8], external: true },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q"],
        outs: ["o"],
        attrs: { scale: 0.5, readonly: true },
        states: { k: "kslot", v: "vslot" },
      },
    ],
  };
  const report = openGraph(graph).estimate({
    generation: { chunkLength: 1, bindings: { C: 16 } },
    maxStorageBufferBindingSize: WIDE_LIMIT,
  });
  const { prefill, decode } = bothScenarios(report);
  // A borrower has no slots of its own — state is only lengths (16 bytes).
  assertEquals(report.resident.stateBytes, 16);
  // io is q 128 + o 128 (no k / v inputs). Intermediates are o 128 / S = 4×1×16×4 = 256 / row
  // statistics = 32.
  assertEquals(decode.ioBytes, 256);
  assertEquals(decode.workspaceBytes, 128 + 256 + 32);
  // In a graph without a chunk-row symbol the two shapes give the same numbers (the estimator's
  // contract).
  assertEquals(prefill.workspaceBytes, decode.workspaceBytes);
  assertEquals(prefill.ioBytes, decode.ioBytes);
});

test("S and the row statistics grow in proportion to chunkLength (prefill only — decode is fixed at M=1)", () => {
  const at = (chunkLength: number): AdmissionScenario =>
    bothScenarios(stateAttentionReport({ capacity: 32, chunkLength })).prefill;
  // M=4: o 512 + S 4×4×32×4=2048 + row statistics 4×4×2×4=128
  assertEquals(at(4).workspaceBytes, 512 + 2048 + 128);
  // M=8: all three terms double (the column capacity stays C, so S grows only by the row count)
  assertEquals(at(8).workspaceBytes, 1024 + 4096 + 256);
  // decode is M=1 regardless of chunkLength
  const decode = (chunkLength: number): number =>
    bothScenarios(stateAttentionReport({ capacity: 32, chunkLength })).decode.workspaceBytes;
  assertEquals(decode(4), 128 + 512 + 32);
  assertEquals(decode(8), 128 + 512 + 32);
});

test("the column capacity of the sliding variant is W−1+M (raising capacity does not move S)", () => {
  const at = (capacity: number): AdmissionScenario =>
    bothScenarios(stateAttentionReport({ capacity, chunkLength: 4, window: 6 })).prefill;
  // Column capacity = 6−1+4 = 9 → S = 4×4×9×4 = 576 (independent of C)
  assertEquals(at(16).workspaceBytes, 512 + 576 + 128);
  assertEquals(at(32).workspaceBytes, 512 + 576 + 128);
  // The state slots stay proportional to C (only S is capped by the window)
  assertEquals(
    stateAttentionReport({ capacity: 32, chunkLength: 4, window: 6 }).resident.stateBytes,
    1 * 2 * 32 * 8 * 4 * 2 + 16,
  );
});

test("the row-block count changes with maxStorageBufferBindingSize (only one block's worth is live at once)", () => {
  // MUST: keep the narrowing within the range **the slots can bind** (with H=64 only one row of S
  // is pushed beyond the limit — the same form as the multi-block test on the real GPU). A limit
  // below the slot (1024B) is a device `GenerationContext.create` rejects, so the estimate fails at
  // the same check (ADR 0066 addendum 5).
  const at = (limit: number): number =>
    bothScenarios(stateAttentionReport({ capacity: 16, chunkLength: 4, heads: 64, limit })).prefill
      .workspaceBytes;
  // One row = 64×16×4 = 4096B, output o [1,64,4,8] = 2048 elements → 8192B.
  // With room in the limit, 1 block (4 rows): S 16384 + row statistics 64×4×2×4 = 2048
  assertEquals(at(WIDE_LIMIT), 8192 + 16384 + 2048);
  // 8192B → 2 blocks of 2 rows each. The second block takes the ranges whose lifetime the first one
  // ended, so the total bytes are one block's worth.
  assertEquals(at(8192), 8192 + 8192 + 1024);
  // 4096B → splits into 4 blocks of 1 row each, but the output o (8192B) itself does not fit in the
  // binding limit, so the plan's preflight (ADR 0093 decision 5) fails without returning numbers.
  assertThrows(() => at(4096), ExecutionError, "'o' 8192B");
});

test("even when a block one row narrower is mixed in by the remainder, it reuses the wide block's ranges, so the total is one wide block's worth", () => {
  // M=3, H=64, limit 8192B (4096B per row) → 2 rows per block, 2 blocks (an even split of 2 rows +
  // 1 row). The second, narrower S / row statistics take the ranges whose lifetime the first block
  // ended (reallocation does not depend on size — ADR 0093 decision 8), so the regions are the
  // three of the wide block's S 8192 + row statistics 1024 + output o [1,64,3,8]=1536 elements →
  // 6144.
  const { prefill } = bothScenarios(
    stateAttentionReport({ capacity: 16, chunkLength: 3, heads: 64, limit: 8192 }),
  );
  const wide = 64 * 2 * 16 * 4 + 64 * 2 * 2 * 4; // 8192 + 1024
  assertEquals(prefill.workspaceBytes, 6144 + wide);
});

test("the limit moves only the intermediates (the io, state and weight fields do not move by a single byte)", () => {
  const wide = stateAttentionReport({ capacity: 16, chunkLength: 4, heads: 64 });
  // 8192B = the smallest narrowing at which the output o (8192B) fits in the binding limit (S
  // splits into 2 blocks)
  const narrow = stateAttentionReport({ capacity: 16, chunkLength: 4, heads: 64, limit: 8192 });
  assertEquals(wide.resident, narrow.resident);
  assertEquals(
    wide.scenarios.map((scenario) => scenario.ioBytes),
    narrow.scenarios.map((scenario) => scenario.ioBytes),
  );
  assert(wide.scenarios[0].workspaceBytes > narrow.scenarios[0].workspaceBytes);
});

test("a graph without states-form attention gives the same numbers even when a limit is passed", () => {
  const without = plainModel().estimate({ bindings: { T: 7 } });
  const with_ = plainModel().estimate({
    bindings: { T: 7 },
    maxStorageBufferBindingSize: 256,
  });
  assertEquals(without, with_);
  // The same holds for a graph with only state_append (a node that produces no temporaries).
  const generation = { chunkLength: 4, bindings: { C: 8 } };
  assertEquals(
    stateModel().estimate({ bindings: { T: 2 }, generation }),
    stateModel().estimate({
      bindings: { T: 2 },
      generation,
      maxStorageBufferBindingSize: 256,
    }),
  );
});

test("the limit's value range is checked at the entry regardless of the graph's shape (not silently accepted even by shapes that do not read it)", () => {
  // Placed only inside the reader (stateAttentionTemps), it would never run for this graph without
  // states-form attention, and an invalid value would pass silently as "harmless because unread".
  for (const limit of [-1, 0, 1.5, Number.NaN]) {
    assertThrows(
      () =>
        plainModel().estimate({
          bindings: { T: 7 },
          maxStorageBufferBindingSize: limit,
        }),
      ExecutionError,
      "positive safe integer",
    );
  }
  // Control: valid values pass as before (proof that the four above are not "fails whatever is
  // passed").
  assertEquals(
    plainModel().estimate({
      bindings: { T: 7 },
      maxStorageBufferBindingSize: 256,
    }).peakAccountedBytes,
    plainModel().estimate({ bindings: { T: 7 } }).peakAccountedBytes,
  );
});

test("estimating states-form attention without passing a limit fails loudly", () => {
  assertThrows(
    () =>
      openGraph(stateAttentionGraph()).estimate({
        generation: { chunkLength: 4, bindings: { C: 16 } },
      }),
    ExecutionError,
    "need options.maxStorageBufferBindingSize",
  );
});

test("a shape where even one row does not fit in the limit fails loudly (row blocks cannot divide it)", () => {
  assertThrows(
    // One row = 64×16×4 = 4096B > 2048B (the state slot's 1024B fits, so only the row-block side
    // fails — a shape where a slot breaks the limit is rejected first by the check one step
    // earlier)
    () => stateAttentionReport({ capacity: 16, chunkLength: 4, heads: 64, limit: 2048 }),
    ExecutionError,
    "already exceeds the storage binding limit",
  );
});

// ---------------------------------------------------------------------------
// The peak of prefill bucket shapes (`chunkBuckets`)
// ---------------------------------------------------------------------------

/**
 * A full states attention with B = H = Hkv = D = 1 + two `state_append`. Only one row of S =
 * capacity `C` × 4 bytes moves, so the boundary where the row-block count splits from 1 → 2 can be
 * made with the capacity alone.
 */
const unitStateAttentionGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 1, "M", 1] },
    { name: "k", dtype: "f32", shape: [1, 1, "M", 1] },
    { name: "v", dtype: "f32", shape: [1, 1, "M", 1] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 1, "M", 1] } },
  states: {
    kslot: { dtype: "f32", shape: [1, 1, "C", 1] },
    vslot: { dtype: "f32", shape: [1, 1, "C", 1] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 1 },
      states: { k: "kslot", v: "vslot" },
    },
    { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "kslot" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: {}, states: { slot: "vslot" } },
  ],
});

describe("the peak of an estimate given chunkBuckets", () => {
  const MIB = 1024 * 1024;
  // C = 1,048,576 → one row of S = 4 MiB. Up to 32 rows fit in the 128 MiB binding limit, so
  // M = 32 is 1 block (128 MiB) and M = 33 splits into 2 blocks of 17 + 16 rows (a 68 MiB wide
  // block) — the shape with fewer rows has larger temporaries (the even-split boundary).
  const estimate = (chunkBuckets?: readonly number[]): AdmissionReport =>
    openGraph(unitStateAttentionGraph()).estimate({
      generation: {
        chunkLength: 33,
        bindings: { C: 1_048_576 },
        ...(chunkBuckets === undefined ? {} : { chunkBuckets }),
      },
      maxStorageBufferBindingSize: 128 * MIB,
      maxBufferSize: 256 * MIB,
      // Budget 0 = the peak is the largest scenario itself (a form not hidden by the budget term).
      planBackingBudgetBytes: 0,
    });
  // k / v slots [1,1,C,1], 4,194,304 each + logical-length uniform 16.
  const STATE_BYTES = 8_388_624;

  describe("when the bucket shape fits in one row block and the prefill shape splits into 2", () => {
    it("the io + intermediates of the bucket shape (M = 32) stand at the peak", () => {
      const report = estimate([32]);
      assertEquals(report.resident.stateBytes, STATE_BYTES);
      // M = 32: io = q / k / v / o at 128 each = 512. Intermediates = o 128 + S 32×C×4 =
      // 134,217,728 + row statistics 32×2×4 = 256 (the three are read and written by the same
      // dispatch, so separate regions).
      assertEquals(report.peakAccountedBytes, STATE_BYTES + 512 + 128 + 134_217_728 + 256);
    });

    it("the public scenarios stay the two prefill / decode (bucket shapes are not listed)", () => {
      const { prefill, decode } = bothScenarios(estimate([32]));
      // M = 33: io = 4×132 = 528. Intermediates = o 132 + the wide block's S 17×C×4 = 71,303,168 +
      // row statistics 17×8 = 136 (the narrow block reuses the wide block's ranges).
      assertEquals(prefill.ioBytes + prefill.workspaceBytes, 528 + 132 + 71_303_168 + 136);
      // M = 1: io = 4×4 = 16. Intermediates = o 4 + S C×4 = 4,194,304 + row statistics 8.
      assertEquals(decode.ioBytes + decode.workspaceBytes, 16 + 4 + 4_194_304 + 8);
    });
  });

  describe("when no buckets are declared", () => {
    it("the peak stays the prefill shape (undeclared row counts are not added)", () => {
      assertEquals(estimate().peakAccountedBytes, STATE_BYTES + 528 + 132 + 71_303_168 + 136);
    });
  });
});

// ---------------------------------------------------------------------------
// fail loudly
// ---------------------------------------------------------------------------

test("an unbound symbolic dimension fails loudly (not silently filled with 0)", () => {
  assertThrows(() => plainModel().estimate(), ExecutionError, "Symbol 'T' is not bound");
});

test("a graph that breaks a contract fails at the contract check before the residency plan (naming nodes[i] (op))", () => {
  // The form with `groups` dropped from conv1d's attrs (ADR 0015's "no default filling").
  // If the residency plan (planWeightResidency → i4 eligibility) ran first, the diagnostic would be
  // `nodes (conv1d)` without an index, failing with a different message from the real build
  // (PreparedModel).
  //
  // NOTE: the contract check of the full surface (`estimateGraphMemory`) itself cannot be observed
  // by this claim — `planGraph` inside the same function raises an `OpContractError` with **the
  // same message**, so the assertThrows below stays green even without the estimator's contract
  // check. Only the `prepareContainer`-side assert in the second half pins the order ("before the
  // residency plan"), and that is where the test name's basis lies.
  const graph = i4Conv1dGraph(1);
  const broken = mergeGraph(
    {
      ...graph,
      nodes: [{ ...graph.nodes[0], attrs: { stride: 1, padding: 0, dilation: 1 } }],
    },
    I4_CONV1D_STORAGE,
  );
  // The residency plan is taken from the **unbroken** graph. Calling `planWeightResidency` on the
  // broken graph would fail on the plan side (`nodes (conv1d)` without an index), so it could not
  // show that the estimator runs its own contract check before using the plan.
  const residency = planWeightResidency(mergeGraph(graph, I4_CONV1D_STORAGE));
  const error = assertThrows(
    () => estimateGraphMemory(broken, residency, {}),
    OpContractError,
    "nodes[",
  );
  assert(error.message.includes("conv1d"), error.message);
  // The admission phase (`prepareContainer`) has the same order — if it ran the residency plan
  // first, the same graph would start failing with `nodes (conv1d)` without an index.
  const prepared = assertThrows(
    () =>
      openGraph(
        {
          ...graph,
          nodes: [{ ...graph.nodes[0], attrs: { stride: 1, padding: 0, dilation: 1 } }],
        },
        [
          weight("b", f32Bytes([0, 0, 0, 0])),
          weight("w", new Uint8Array(new ArrayBuffer(128)), {
            codec: "int4-sym-g",
            groupSize: 16,
            scale: { bytes: f32Bytes(new Array(16).fill(1)), dtype: "f32" },
          }),
        ],
      ),
    OpContractError,
    "nodes[",
  );
  assert(prepared.message.includes("conv1d"), prepared.message);
});

test("states-only symbols are not accepted in options.bindings (their binding point is on the generation side)", () => {
  assertThrows(
    () => stateModel().estimate({ bindings: { T: 2, C: 8 } }),
    ExecutionError,
    "states-only symbol",
  );
});

test("a states-form graph is not estimated without generation (fails saying a context is needed)", () => {
  assertThrows(() => stateModel().estimate({ bindings: { T: 2 } }), Error, "GenerationContext");
});

test("passing generation to a graph without states declarations fails loudly", () => {
  assertThrows(
    () =>
      plainModel().estimate({
        bindings: { T: 7 },
        generation: { chunkLength: 4 },
      }),
    ExecutionError,
    "has no states declarations",
  );
});

test("returns no estimate for a chunkLength the real build rejects", () => {
  assertThrows(
    () =>
      stateModel().estimate({
        bindings: { T: 2 },
        generation: { chunkLength: 0, bindings: { C: 8 } },
      }),
    ExecutionError,
    "chunkLength 0 is",
  );
});

test("a disagreement on a symbol appearing in both states and inputs fails loudly (the same check as run)", () => {
  // Make v's capacity T to create a shared symbol (k stays C — the division of states-only symbols
  // is unchanged)
  const graph = stateGraph();
  graph.states = {
    ...graph.states,
    v: { dtype: "f32", shape: [1, 2, "T", 4] },
  };
  const model = openGraph(graph, stateTensors());
  assertThrows(
    () =>
      model.estimate({
        bindings: { T: 2 },
        generation: { chunkLength: 4, bindings: { C: 8, T: 3 } },
      }),
    ExecutionError,
    "Symbol 'T' differs",
  );
});

test("binding a symbol that is not in the graph fails loudly", () => {
  assertThrows(() => plainModel().estimate({ bindings: { T: 7, Z: 3 } }), ExecutionError, "'Z'");
});

// ---------------------------------------------------------------------------
// The same checks as the real build (contracts that cannot be decided without looking at the whole
// graph) ---------------------------------------------------------------------------

// The full surface (`estimateGraphMemory`) does not go through `PreparedModel`, so without running
// the contract checks on the estimator side it would return a complete AdmissionReport for "a model
// on which createSessionFromContainer always fails" (the opposite of admission's purpose). The
// three tests in this section **call that surface directly** — calling through
// `PreparedModel.estimate` would let the `prepareContainer` side fail first, and the test would
// stay green even with the estimator's check removed.

/**
 * A graph where the `kv` slot has only readers and no `state_append` at all (a violation of
 * decision 5b).
 */
const readerOnlyStateGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 8, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 2, "M", 4] },
    { name: "v", dtype: "f32", shape: [1, 2, "M", 4] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 8, "M", 4] } },
  states: {
    "kv.k": { dtype: "f32", shape: [1, 2, "C", 4] },
    "kv.v": { dtype: "f32", shape: [1, 2, "C", 4] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kv.k", v: "kv.v" },
    },
  ],
});

test("returns no estimate for a graph with a reader-only state slot (the same contract check as the real build)", () => {
  const graph = mergeGraph(readerOnlyStateGraph());
  const error = assertThrows(
    () =>
      estimateGraphMemory(graph, planWeightResidency(graph), {
        generation: { chunkLength: 4, bindings: { C: 8 } },
      }),
    ExecutionError,
    "no state_append at all",
  );
  assert(error.message.includes("kv.k"), error.message);
});

/** A cat whose concatenation axis mixes different symbols (the remaining rejection of ADR 0046). */
const mixedSymbolCatGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cat"] },
  symbols: ["T", "U"],
  inputs: [
    { name: "a", dtype: "f32", shape: ["T", 2] },
    { name: "b", dtype: "f32", shape: ["U", 2] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["T", 2] } },
  nodes: [{ op: "cat", ins: ["a", "b"], outs: ["y"], attrs: { dim: 0 } }],
});

test("returns no estimate for a graph whose cat concatenation axis mixes different symbols", () => {
  const graph = mergeGraph(mixedSymbolCatGraph());
  assertThrows(
    () => estimateGraphMemory(graph, planWeightResidency(graph), { bindings: { T: 2, U: 3 } }),
    ExecutionError,
    "different symbols",
  );
});

// ---------------------------------------------------------------------------
// unaccounted
// ---------------------------------------------------------------------------

test("unaccounted states explicitly what is not accounted for (the estimate is not an absolute guarantee)", () => {
  const { unaccounted } = plainModel().estimate({ bindings: { T: 7 } });
  const joined = unaccounted.join("\n");
  assert(joined.includes("fusion"), joined);
  assert(joined.includes("params"), joined);
  assert(joined.includes("writeBuffer"), joined);
});

// The in-node temporaries of attention **are now accounted for** — after the S / row statistics of
// states-form attention, the S, row statistics and i8a8 quantized intermediates of the fused
// attention op are also counted with the same pure function as execution (ADR 0124 decision 7).
// What remains unaccounted is the i8a8 quantized intermediates of linear; unaccounted names them
// and does not list attention's temporaries as unaccounted (naming an accounted item as "not
// included" would make readers of the estimate assume more headroom than there is).
test("unaccounted names the linear i8a8 quantized intermediates and says attention's temporaries (states form, fused op) are accounted for", () => {
  const { unaccounted } = stateModel().estimate({
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  const joined = unaccounted.join("\n");
  assert(!joined.includes("in-node temporaries of attention that is not states-form"), joined);
  assert(joined.includes("Quantized intermediates of linear i8a8"), joined);
  assert(joined.includes("The in-node temporaries of attention are counted"), joined);
  assert(
    joined.includes(
      "the score S, row statistics and i8a8 quantized intermediates of the fused attention op",
    ),
    joined,
  );
});

// ---------------------------------------------------------------------------
// stateAttentionReduce
// ---------------------------------------------------------------------------

test("the parallel-fused temporary estimate removes only the row-statistics allocation", () => {
  const model = openGraph(stateAttentionGraph());
  const options = {
    generation: { chunkLength: 4, bindings: { C: 16 } },
    maxStorageBufferBindingSize: WIDE_LIMIT,
  };
  const plain = model.estimate(options);
  const fused = model.estimate({
    ...options,
    stateAttentionReduce: "parallel-fused",
  });
  const { prefill, decode } = bothScenarios(fused);
  assertEquals(prefill.workspaceBytes, 512 + 1024);
  assertEquals(decode.workspaceBytes, 128 + 256);
  assertEquals(fused.resident, plain.resident);
  assertEquals(prefill.ioBytes, plain.scenarios[0].ioBytes);
  const outside = { ...options, generation: { chunkLength: 16, bindings: { C: 1025 } } };
  assertEquals(
    model.estimate({ ...outside, stateAttentionReduce: "parallel-fused" }),
    model.estimate(outside),
  );
  Object.defineProperty(options, "stateAttentionReduce", { value: "unsupported" });
  assertThrows(() => model.estimate(options), ExecutionError, "stateAttentionReduce");
});

test("attention estimate settings reject wrong types before converting to a string", () => {
  const model = plainModel();
  let conversions = 0;
  const object = {
    [Symbol.toPrimitive](): string {
      conversions++;
      return "parallel-fused";
    },
  };
  for (const value of [["parallel-fused"], object, false, 0, 1n, Symbol("parallel-fused")]) {
    const options = { bindings: { T: 2 } };
    Object.defineProperty(options, "stateAttentionReduce", { value });
    assertThrows(() => model.estimate(options), ExecutionError, "stateAttentionReduce");
  }
  assertEquals(conversions, 0);
});
