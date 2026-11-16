// The corpus's layer cut (tests/corpus/cut.ts) on a small hand-made layered declaration: what a
// valid cut keeps, how it rewires the residual, and each loud failure. CPU only.

import { describe, it } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CorpusCutError, cutLayers } from "./corpus/cut.ts";
import { canonicalIrJson, type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

type Toy = {
  /** Layer 3 reads layer 1's state slot (a lender, like a shared key/value cache). */
  readonly lendState?: boolean;
  /** Layer 3 reads a value of layer 0 (a skip connection across layers). */
  readonly lendValue?: boolean;
  /** Layer 2 reads two values of layer 1 (no single residual at that boundary). */
  readonly twoHanded?: boolean;
  /** A node of layer 2 reads layer 2's and layer 1's weights. */
  readonly sharedWeight?: boolean;
};

const ROW = [1, "M", 4];

/**
 * x → prelude `h0 = x · embed` → four layers `a_n = h_n · w_n`, `h_{n+1} = h_n + a_n` → exit
 * `y = h_4 · final`. Layer 1 also appends `a_1` to state slot `l1.k`.
 */
const toy = (options: Toy = {}): IrDeclaration => {
  const values: DeclarationJson["values"] = {
    "embed.weight": { dtype: "f32", shape: [4] },
    "final.weight": { dtype: "f32", shape: [4] },
    h0: { dtype: "f32", shape: ROW },
    y: { dtype: "f32", shape: ROW },
  };
  const initializers: DeclarationJson["initializers"] = { "embed.weight": {}, "final.weight": {} };
  const nodes: DeclarationJson["nodes"] = [
    { op: "mul", ins: ["x", "embed.weight"], outs: ["h0"], attrs: {} },
  ];
  const value = (name: string): string => {
    values[name] = { dtype: "f32", shape: ROW };
    return name;
  };
  for (let layer = 0; layer < 4; layer += 1) {
    const weight = `model.layers.${layer}.w`;
    initializers[weight] = {};
    values[weight] = { dtype: "f32", shape: [4] };
    const residual = layer === 0 ? "h0" : `h${layer}`;
    nodes.push({ op: "mul", ins: [residual, weight], outs: [value(`a${layer}`)], attrs: {} });
    let branch = `a${layer}`;
    if (layer === 1) {
      nodes.push({
        op: "state_append",
        ins: ["a1"],
        outs: [],
        attrs: {},
        states: { slot: "l1.k" },
      });
    }
    if (layer === 2 && options.twoHanded) {
      nodes.push({ op: "add", ins: [branch, "a1"], outs: [value("c2")], attrs: {} });
      branch = "c2";
    }
    if (layer === 2 && options.sharedWeight) {
      nodes.push({
        op: "mul",
        ins: [weight, "model.layers.1.w"],
        outs: [value("d2")],
        attrs: {},
      });
      nodes.push({ op: "add", ins: [branch, "d2"], outs: [value("e2")], attrs: {} });
      branch = "e2";
    }
    if (layer === 3 && options.lendValue) {
      nodes.push({ op: "add", ins: [branch, "a0"], outs: [value("b3")], attrs: {} });
      branch = "b3";
    }
    if (layer === 3 && options.lendState) {
      nodes.push({
        op: "attention",
        ins: [branch, branch, branch],
        outs: [value("r3")],
        attrs: {},
        states: { k: "l1.k", v: "l1.k" },
      });
      branch = "r3";
    }
    nodes.push({ op: "add", ins: [residual, branch], outs: [value(`h${layer + 1}`)], attrs: {} });
  }
  nodes.push({ op: "mul", ins: ["h4", "final.weight"], outs: ["y"], attrs: {} });
  const document: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: ["C", "M"],
    inputs: [{ name: "x", dtype: "f32", shape: ROW }],
    outputs: ["y"],
    initializers,
    values,
    states: { "l1.k": { dtype: "f32", shape: [1, 1, "C", 4] } },
    nodes,
  };
  return parseIrDeclarationValue(document);
};

/** Each node as `op(ins) → outs`, in order. */
const listing = (declaration: IrDeclaration): string[] =>
  declaration.nodes.map((node) => `${node.op}(${node.ins.join(", ")}) → ${node.outs.join(", ")}`);

describe("cutting a layered declaration", () => {
  it("keeping every layer gives the declaration back unchanged", () => {
    const declaration = toy({ lendState: true });
    const { declaration: cut, report } = cutLayers(declaration, [3, 1, 0, 2]);
    assertEquals(canonicalIrJson(cut), canonicalIrJson(declaration));
    assertEquals(report.rewired, []);
  });

  it("rewires the residual across each removed range to the last kept producer's hand-off", () => {
    const { declaration: cut, report } = cutLayers(toy({ lendState: true }), [1, 3]);
    assertEquals(report.kept, [1, 3]);
    // Into layer 1: the prelude's h0 replaces layer 0's h1; into layer 3: layer 1's h2 replaces h3.
    assertEquals(report.rewired, [
      ["h1", "h0"],
      ["h3", "h2"],
    ]);
    assertEquals(listing(cut), [
      "mul(x, embed.weight) → h0",
      "mul(h0, model.layers.1.w) → a1",
      "state_append(a1) → ",
      "add(h0, a1) → h2",
      "mul(h2, model.layers.3.w) → a3",
      "attention(a3, a3, a3) → r3",
      "add(h2, r3) → h4",
      "mul(h4, final.weight) → y",
    ]);
  });

  it("removes the dead nodes, values, initializers, state slots and symbols, and recomputes requires.ops", () => {
    const { declaration: cut, report } = cutLayers(toy(), [0, 2, 3]);
    assertEquals(Object.keys(cut.initializers).sort(), [
      "embed.weight",
      "final.weight",
      "model.layers.0.w",
      "model.layers.2.w",
      "model.layers.3.w",
    ]);
    assertEquals(Object.keys(cut.values).includes("a1"), false);
    assertEquals(Object.keys(cut.states), []);
    assertEquals(report.states, []);
    // C sized only the removed slot, so it can no longer be bound and goes with it.
    assertEquals(cut.symbols, ["M"]);
    assertEquals([...cut.requires.ops].sort(), ["add", "mul"]);
  });

  it("gives a declaration the IR parser accepts, and the same one every time", () => {
    const first = cutLayers(toy({ lendState: true }), [1, 3]).declaration;
    const again = cutLayers(toy({ lendState: true }), [3, 1]).declaration;
    const text = canonicalIrJson(first);
    assertEquals(canonicalIrJson(parseIrDeclarationValue(JSON.parse(text))), text);
    assertEquals(canonicalIrJson(again), text);
  });
});

describe("a cut the declaration does not allow fails loudly", () => {
  it("when a kept layer reads a state slot of a removed layer (naming both)", () => {
    assertThrows(
      () => cutLayers(toy({ lendState: true }), [0, 2, 3]),
      CorpusCutError,
      "layer 3 reads state 'l1.k' of removed layer 1",
    );
  });

  it("when a kept layer reads a value of a removed layer (naming both)", () => {
    assertThrows(
      () => cutLayers(toy({ lendValue: true }), [1, 2, 3]),
      CorpusCutError,
      "layer 3 reads value 'a0' of removed layer 0",
    );
  });

  it("when the boundary into a kept layer is not a single residual value", () => {
    assertThrows(
      () => cutLayers(toy({ twoHanded: true }), [2, 3]),
      CorpusCutError,
      "the boundary into layer 2 is not a single residual value: it reads [h2 (layer 1), a1 (layer 1)]",
    );
  });

  it("when a graph output is computed in a removed layer", () => {
    assertThrows(
      () => cutLayers(toy(), [0, 1]),
      CorpusCutError,
      "graph output 'y' is computed in removed layer 3",
    );
  });

  it("when a node reads weights of two layers", () => {
    assertThrows(
      () => cutLayers(toy({ sharedWeight: true }), [0, 1, 2, 3]),
      CorpusCutError,
      "reads weights of layers",
    );
  });

  it("when a layer to keep does not exist, or none is given", () => {
    assertThrows(() => cutLayers(toy(), [5]), CorpusCutError, "the graph has no layer 5");
    assertThrows(() => cutLayers(toy(), []), CorpusCutError, "must be non-empty and distinct");
  });
});
