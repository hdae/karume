import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { openModel } from "../src/format/container.ts";
import { parseIrGraph } from "../src/format/ir.ts";
import { planAliases } from "../src/runtime/fusion.ts";
import { planGraph } from "../src/runtime/plan.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { permutedSourceIndices } from "./helpers/permute.ts";

const permutations = (axes: readonly number[]): number[][] =>
  axes.length === 0
    ? [[]]
    : axes.flatMap((axis, i) =>
      permutations(axes.filter((_, j) => i !== j)).map((rest) => [axis, ...rest])
    );

describe("要素順を保つpermuteの別名化", () => {
  it("rank 1〜4の全軸順を、平坦な要素番号の並べ替えと照合する", () => {
    let cases = 0;
    for (const rank of [1, 2, 3, 4]) {
      for (let code = 0; code < 3 ** rank; code++) {
        const shape = Array.from({ length: rank }, (_, i) => 1 + Math.floor(code / 3 ** i) % 3);
        for (const dims of permutations(Array.from({ length: rank }, (_, i) => i))) {
          const graph = parseIrGraph(JSON.stringify(
            singleOpGraph("permute", [shape], [dims.map((axis) => shape[axis])], {
              attrs: { dims },
            }),
          ));
          // コピー元を内部確保にし、所有権の条件と添字の条件を分けて検証する。
          const owned = singleOpGraph("permute", [shape], [dims.map((axis) => shape[axis])], {
            attrs: { dims },
          });
          owned.requires.ops.push("neg");
          owned.values.h = { dtype: "f32", shape };
          owned.nodes[0].ins = ["h"];
          owned.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
          const plans = planGraph(parseIrGraph(JSON.stringify(owned)), {}).nodes;
          assertEquals(planAliases(planGraph(graph, {}).nodes).size, 0);
          const expected = permutedSourceIndices(shape, dims).every((source, output) =>
            source === output
          );
          assertEquals(
            planAliases(plans).has(plans[1]),
            expected,
            JSON.stringify({ shape, dims }),
          );
          cases++;
        }
      }
    }
    assertEquals(cases, 2127);
  });

  it("記号束縛後の形で判定し、M=1と複数tokenを混同しない", () => {
    const source = singleOpGraph("permute", [[1, "M", 8, 256]], [[1, 8, "M", 256]], {
      attrs: { dims: [0, 2, 1, 3] },
      symbols: ["M"],
    });
    source.requires.ops.push("neg");
    source.values.h = { dtype: "f32", shape: [1, "M", 8, 256] };
    source.nodes[0].ins = ["h"];
    source.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
    const graph = parseIrGraph(JSON.stringify(source));
    for (const M of [1, 4, 8, 1, 32]) {
      const nodes = planGraph(graph, { M }).nodes;
      assertEquals(planAliases(nodes).has(nodes[1]), M === 1);
    }
    // 要素が無い形は今回の最適化対象に含めない。
    const empty = singleOpGraph("permute", [[0, 1]], [[1, 0]], { attrs: { dims: [1, 0] } });
    empty.requires.ops.push("neg");
    empty.values.h = { dtype: "f32", shape: [0, 1] };
    empty.nodes[0].ins = ["h"];
    empty.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
    assertEquals(planAliases(planGraph(parseIrGraph(JSON.stringify(empty)), {}).nodes).size, 0);
  });

  it("見積りも実行と同じ別名判定を使い、共有元1本だけを数える", () => {
    for (const [dims, expectedBytes] of [[[1, 0, 2], 24], [[2, 0, 1], 48]] as const) {
      const shape = [1, 2, 3];
      const graph = singleOpGraph("permute", [shape], [dims.map((axis) => shape[axis])], {
        attrs: { dims: [...dims] },
      });
      graph.requires.ops.push("neg");
      graph.values.h = { dtype: "f32", shape };
      graph.nodes[0].ins = ["h"];
      graph.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
      const report = estimateSessionMemory(openModel(graphModelBuffer(graph)));
      assertEquals(report.scenarios.map((s) => s.name), ["run"]);
      assertEquals(report.scenarios[0].workspaceBytes, expectedBytes);
    }
  });
});
