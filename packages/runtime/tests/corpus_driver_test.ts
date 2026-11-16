// The corpus driver's inputs and plan checks (tests/corpus/driver.ts), from a declaration alone.
// CPU only — the runs themselves are gpu_corpus_test.ts.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import {
  assertPlanFits,
  type CorpusPlan,
  describeStep,
  stepBindings,
  stepInputs,
} from "./corpus/driver.ts";
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

/**
 * A text encoder's relative positions and mask: `pos` [T, T] gathers from the scores' last axis (12
 * buckets) after an expand; `mask` [1, T] is cast to f32 and multiplied in, and `1 − mask` is cast
 * to bool and fills the masked positions.
 */
const textDocument = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["cast", "expand", "gather", "masked_fill", "mul", "reshape", "sub"] },
  symbols: ["T"],
  inputs: [
    { name: "scores", dtype: "f32", shape: [4, "T", 12] },
    { name: "pos", dtype: "i32", shape: ["T", "T"] },
    { name: "mask", dtype: "i32", shape: [1, "T"] },
  ],
  outputs: ["picked", "kept", "filled"],
  initializers: { one: {} },
  values: {
    one: { dtype: "i32", shape: [1] },
    wide: { dtype: "i32", shape: [4, "T", "T"] },
    picked: { dtype: "f32", shape: [4, "T", "T"] },
    column: { dtype: "i32", shape: [1, "T", 1] },
    weight: { dtype: "f32", shape: [1, "T", 1] },
    kept: { dtype: "f32", shape: [4, "T", "T"] },
    flip: { dtype: "i32", shape: [1, "T"] },
    flipped: { dtype: "bool", shape: [1, "T"] },
    filled: { dtype: "f32", shape: [4, "T", "T"] },
  },
  nodes: [
    { op: "expand", ins: ["pos"], outs: ["wide"], attrs: {} },
    { op: "gather", ins: ["scores", "wide"], outs: ["picked"], attrs: {} },
    { op: "reshape", ins: ["mask"], outs: ["column"], attrs: {} },
    { op: "cast", ins: ["column"], outs: ["weight"], attrs: { to: "f32" } },
    { op: "mul", ins: ["picked", "weight"], outs: ["kept"], attrs: {} },
    { op: "sub", ins: ["one", "mask"], outs: ["flip"], attrs: {} },
    { op: "cast", ins: ["flip"], outs: ["flipped"], attrs: { to: "bool" } },
    { op: "masked_fill", ins: ["kept", "flipped"], outs: ["filled"], attrs: { value: 0 } },
  ],
});

const textDeclaration = (): IrDeclaration => parseIrDeclarationValue(textDocument());

/** An image `x` [1, 2, h, w], two symbols, no state slots. */
const spatialDeclaration = (): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add"] },
    symbols: ["h", "w"],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2, "h", "w"] }],
    outputs: ["y"],
    initializers: {},
    values: { y: { dtype: "f32", shape: [1, 2, "h", "w"] } },
    nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
  } satisfies DeclarationJson);

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

  it("draw a gather index read through a layout op below the source's last axis", () => {
    const relative = stepInputs(textDeclaration(), { T: 5 }, 1, "run/0");
    assertEquals(relative.pos.shape, [5, 5]);
    assert(relative.pos.data.every((index) => index >= 0 && index < 12));
    // Drawn over the last axis, not over the source's leading axes (4 and T = 5).
    assert(relative.pos.data.some((index) => index >= 5));
  });

  it("set an i32 mask (a value the graph casts to bool) to all ones", () => {
    const relative = stepInputs(textDeclaration(), { T: 5 }, 1, "run/0");
    assertEquals(relative.mask.dtype, "i32");
    assertEquals([...relative.mask.data], [1, 1, 1, 1, 1]);
  });

  it("throw for an i32 input the graph only casts to f32 (no truth values), naming it", () => {
    const changed = textDocument();
    changed.nodes = changed.nodes.filter((node) => node.op !== "masked_fill");
    changed.nodes = changed.nodes.filter((node) => !node.outs.includes("flip"));
    changed.nodes = changed.nodes.filter((node) => !node.outs.includes("flipped"));
    changed.outputs = changed.outputs.filter((name) => name !== "filled");
    for (const name of ["flip", "flipped", "filled"]) delete changed.values[name];
    delete changed.initializers.one;
    delete changed.values.one;
    changed.requires.ops = changed.requires.ops.filter(
      (op) => op !== "masked_fill" && op !== "sub",
    );
    assertThrows(
      () => stepInputs(parseIrDeclarationValue(changed), { T: 2 }, 1, "run/0"),
      Error,
      "input 'mask' (i32) is read by cast slot 0: the driver draws i32 inputs only as embedding indices, gather indices or masks",
    );
  });

  it("throw for a gather index that is also read as a value, naming it", () => {
    const changed = textDocument();
    changed.nodes.push({ op: "mul", ins: ["pos", "pos"], outs: ["square"], attrs: {} });
    changed.values.square = { dtype: "i32", shape: ["T", "T"] };
    changed.outputs.push("square");
    assertThrows(
      () => stepInputs(parseIrDeclarationValue(changed), { T: 2 }, 1, "run/0"),
      Error,
      "input 'pos' (i32) is read by mul slot 0",
    );
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

  it("rejects a plan without a row symbol on a graph with state slots", () => {
    const { rowSymbol: _, ...withoutRows } = PLAN;
    assertThrows(
      () => assertPlanFits(declaration(), withoutRows),
      Error,
      "the graph has state slots, so the plan needs a row symbol",
    );
  });

  it("rejects a step that binds symbols by name on a plan with a row symbol", () => {
    assertThrows(
      () =>
        assertPlanFits(declaration(), {
          ...PLAN,
          runs: [{ name: "run", steps: [{ M: 6 }], capacity: 8 }],
        }),
      Error,
      "run 'run' step 0 binds symbols by name, but the plan has the row symbol 'M'",
    );
  });
});

describe("a plan without a row symbol binds every input symbol in each step", () => {
  const SPATIAL: CorpusPlan = {
    bindings: {},
    runs: [
      {
        name: "run",
        steps: [
          { h: 3, w: 5 },
          { w: 2, h: 7 },
        ],
      },
    ],
  };

  it("accepts steps that bind several symbols by name, and evaluates the shapes from them", () => {
    assertPlanFits(spatialDeclaration(), SPATIAL);
    const bindings = stepBindings(SPATIAL, { h: 3, w: 5 });
    assertEquals(stepInputs(spatialDeclaration(), bindings, 1, "run/0").x.shape, [1, 2, 3, 5]);
  });

  it("accepts a step that binds nothing when the plan's bindings leave no symbol", () => {
    assertPlanFits(spatialDeclaration(), {
      bindings: { h: 2, w: 2 },
      runs: [{ name: "run", steps: [{}] }],
    });
  });

  it("rejects a step that leaves an input symbol unbound or binds another", () => {
    assertThrows(
      () =>
        assertPlanFits(spatialDeclaration(), {
          bindings: {},
          runs: [{ name: "run", steps: [{ h: 3 }] }],
        }),
      Error,
      "run 'run' step 0 binds [h], but the input symbols the plan's bindings leave are [h, w]",
    );
    assertThrows(
      () =>
        assertPlanFits(spatialDeclaration(), {
          bindings: { w: 2 },
          runs: [{ name: "run", steps: [{ h: 3, w: 4 }] }],
        }),
      Error,
      "run 'run' step 0 binds [h, w], but the input symbols the plan's bindings leave are [h]",
    );
  });

  it("rejects a number of rows without a row symbol, and a value that is not a positive whole number", () => {
    assertThrows(
      () =>
        assertPlanFits(spatialDeclaration(), {
          bindings: { w: 2 },
          runs: [{ name: "run", steps: [3] }],
        }),
      Error,
      "run 'run' step 0 is a number of rows, but the plan has no row symbol",
    );
    assertThrows(
      () =>
        assertPlanFits(spatialDeclaration(), {
          bindings: {},
          runs: [{ name: "run", steps: [{ h: 0, w: 2 }] }],
        }),
      Error,
      "run 'run' step 0: every symbol needs a positive whole number",
    );
  });

  it("describes a step in the census by its rows or by its symbols' values", () => {
    assertEquals(describeStep(48), "rows 48");
    assertEquals(describeStep({ w: 15, h: 9 }), "h 9 w 15");
    assertEquals(describeStep({}), "");
  });
});
