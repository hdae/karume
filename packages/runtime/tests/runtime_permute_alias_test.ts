import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { planAliases } from "../src/runtime/fusion.ts";
import { planGraph } from "../src/runtime/plan.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  memoryModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { permutedSourceIndices } from "./helpers/permute.ts";

/** Declaration → merged graph (the graphs in this file have no initializers at all). */
const parse = (graph: DeclarationJson): IrGraph =>
  mergedGraph(memoryModel(graph, []).graphs[GRAPH_NAME], GRAPH_NAME);

const permutations = (axes: readonly number[]): number[][] =>
  axes.length === 0
    ? [[]]
    : axes.flatMap((axis, i) =>
        permutations(axes.filter((_, j) => i !== j)).map((rest) => [axis, ...rest]),
      );

describe("aliasing a permute that preserves element order", () => {
  it("checks every axis order of rank 1–4 against a permutation of flat element indices", () => {
    let cases = 0;
    for (const rank of [1, 2, 3, 4]) {
      for (let code = 0; code < 3 ** rank; code++) {
        const shape = Array.from({ length: rank }, (_, i) => 1 + (Math.floor(code / 3 ** i) % 3));
        for (const dims of permutations(Array.from({ length: rank }, (_, i) => i))) {
          const graph = parse(
            singleOpDeclaration("permute", [shape], [dims.map((axis) => shape[axis])], {
              attrs: { dims },
            }),
          );
          // Make the copy source an internal allocation so the ownership condition is verified
          // separately from the index condition.
          const owned = singleOpDeclaration("permute", [shape], [dims.map((axis) => shape[axis])], {
            attrs: { dims },
          });
          owned.requires.ops.push("neg");
          owned.values.h = { dtype: "f32", shape };
          owned.nodes[0].ins = ["h"];
          owned.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
          const plans = planGraph(parse(owned), {}).nodes;
          assertEquals(planAliases(planGraph(graph, {}).nodes).size, 0);
          const expected = permutedSourceIndices(shape, dims).every(
            (source, output) => source === output,
          );
          assertEquals(planAliases(plans).has(plans[1]), expected, JSON.stringify({ shape, dims }));
          cases++;
        }
      }
    }
    assertEquals(cases, 2127);
  });

  it("decides on the shape after symbol binding, and does not confuse M=1 with multiple tokens", () => {
    const source = singleOpDeclaration("permute", [[1, "M", 8, 256]], [[1, 8, "M", 256]], {
      attrs: { dims: [0, 2, 1, 3] },
      symbols: ["M"],
    });
    source.requires.ops.push("neg");
    source.values.h = { dtype: "f32", shape: [1, "M", 8, 256] };
    source.nodes[0].ins = ["h"];
    source.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
    const graph = parse(source);
    for (const M of [1, 4, 8, 1, 32]) {
      const nodes = planGraph(graph, { M }).nodes;
      assertEquals(planAliases(nodes).has(nodes[1]), M === 1);
    }
    // Shapes with no elements are not part of this optimization.
    const empty = singleOpDeclaration("permute", [[0, 1]], [[1, 0]], { attrs: { dims: [1, 0] } });
    empty.requires.ops.push("neg");
    empty.values.h = { dtype: "f32", shape: [0, 1] };
    empty.nodes[0].ins = ["h"];
    empty.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
    assertEquals(planAliases(planGraph(parse(empty), {}).nodes).size, 0);
  });

  it("the estimate uses the same alias decision as execution and counts only the one shared source", () => {
    for (const [dims, expectedBytes] of [
      [[1, 0, 2], 24],
      [[2, 0, 1], 48],
    ] as const) {
      const shape = [1, 2, 3];
      const graph = singleOpDeclaration("permute", [shape], [dims.map((axis) => shape[axis])], {
        attrs: { dims: [...dims] },
      });
      graph.requires.ops.push("neg");
      graph.values.h = { dtype: "f32", shape };
      graph.nodes[0].ins = ["h"];
      graph.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
      const report = openGraph(graph).estimate();
      assertEquals(
        report.scenarios.map((s) => s.name),
        ["run"],
      );
      assertEquals(report.scenarios[0].workspaceBytes, expectedBytes);
    }
  });
});
