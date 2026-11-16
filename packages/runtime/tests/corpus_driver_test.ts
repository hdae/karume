// The corpus driver's inputs and plan checks (tests/corpus/driver.ts), from a declaration alone.
// CPU only — the runs themselves are gpu_corpus_test.ts.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { assertPlanFits, type CorpusPlan, stepInputs } from "./corpus/driver.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

/**
 * ids (an index into a 10-row table) and row (an index into the step's activation rows), an f32
 * feature input, a bool mask; one state slot sized by the states-only symbol C.
 */
const document = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add", "embedding", "mul", "state_append"] },
  symbols: ["C", "M", "R"],
  inputs: [
    { name: "ids", dtype: "i32", shape: [1, "M"] },
    { name: "features", dtype: "f32", shape: [1, "M", 4] },
    { name: "mask", dtype: "bool", shape: [1, "M", 1] },
    { name: "row", dtype: "i32", shape: ["R"] },
  ],
  outputs: ["picked"],
  initializers: { table: {} },
  values: {
    table: { dtype: "f32", shape: [10, 4] },
    e: { dtype: "f32", shape: [1, "M", 4] },
    h: { dtype: "f32", shape: [1, "M", 4] },
    flat: { dtype: "f32", shape: ["M", 4] },
    picked: { dtype: "f32", shape: ["R", 4] },
  },
  states: { "s.k": { dtype: "f32", shape: [1, 1, "C", 4] } },
  nodes: [
    { op: "embedding", ins: ["table", "ids"], outs: ["e"], attrs: {} },
    { op: "add", ins: ["e", "features"], outs: ["h"], attrs: {} },
    { op: "state_append", ins: ["h"], outs: [], attrs: {}, states: { slot: "s.k" } },
    { op: "mul", ins: ["h", "mask"], outs: ["flat"], attrs: {} },
    { op: "embedding", ins: ["flat", "row"], outs: ["picked"], attrs: {} },
  ],
});

const declaration = (): IrDeclaration => parseIrDeclarationValue(document());

const PLAN: CorpusPlan = {
  rowSymbol: "M",
  bindings: { R: 1 },
  capacitySymbol: "C",
  runs: [{ name: "run", steps: [6, 1], capacity: 8 }],
};

describe("the inputs of a step", () => {
  const inputs = stepInputs(declaration(), { M: 6, R: 3 }, 1, "run/0");

  it("evaluate their shapes from the bindings", () => {
    assertEquals(inputs.ids.shape, [1, 6]);
    assertEquals(inputs.features.shape, [1, 6, 4]);
    assertEquals(inputs.row.shape, [3]);
  });

  it("draw an embedding index below the table's row count (an initializer or the step's rows)", () => {
    const ids = inputs.ids.data;
    assert(ids.every((id) => id >= 0 && id < 10));
    assert(inputs.row.data.every((index) => index >= 0 && index < 6));
  });

  it("draw f32 values in [-1, 1) and set bool inputs to ones", () => {
    assert(inputs.features.data.every((value) => value >= -1 && value < 1));
    assert(new Set(inputs.features.data).size > 1);
    assertEquals([...inputs.mask.data], [1, 1, 1, 1, 1, 1]);
  });

  it("are the same for the same seed and label, and differ for another label", () => {
    const again = stepInputs(declaration(), { M: 6, R: 3 }, 1, "run/0");
    const other = stepInputs(declaration(), { M: 6, R: 3 }, 1, "run/1");
    assertEquals(again.features.data, inputs.features.data);
    assert(other.features.data.some((value, i) => value !== inputs.features.data[i]));
  });

  it("throw for an i32 input that is not an embedding index, naming it", () => {
    const changed = document();
    changed.nodes.push({ op: "add", ins: ["e", "ids"], outs: ["odd"], attrs: {} });
    changed.values.odd = { dtype: "f32", shape: [1, "M", 4] };
    changed.outputs.push("odd");
    assertThrows(
      () => stepInputs(parseIrDeclarationValue(changed), { M: 2, R: 1 }, 1, "run/0"),
      Error,
      "input 'ids' (i32) is read by add slot 1: the driver draws i32 inputs only as embedding indices",
    );
  });
});

describe("a plan must bind what the declaration needs", () => {
  it("accepts a plan that binds every input symbol and names the capacity symbol", () => {
    assertPlanFits(declaration(), PLAN);
  });

  it("rejects a row symbol that sizes no input", () => {
    assertThrows(
      () => assertPlanFits(declaration(), { ...PLAN, rowSymbol: "C" }),
      Error,
      "the plan's row symbol 'C' sizes no input",
    );
  });

  it("rejects an input symbol without a value", () => {
    assertThrows(
      () => assertPlanFits(declaration(), { ...PLAN, bindings: {} }),
      Error,
      "input symbol 'R' has no value in the plan",
    );
  });

  it("rejects a capacity symbol other than the graph's states-only symbol", () => {
    assertThrows(
      () => assertPlanFits(declaration(), { ...PLAN, capacitySymbol: undefined }),
      Error,
      "the plan's capacity symbol is absent, but the graph's states-only symbols are C",
    );
  });

  it("rejects a run without a capacity on a graph with a capacity symbol", () => {
    assertThrows(
      () => assertPlanFits(declaration(), { ...PLAN, runs: [{ name: "run", steps: [6, 1] }] }),
      Error,
      "run 'run': a capacity is given exactly when the graph has a capacity symbol",
    );
  });
});
