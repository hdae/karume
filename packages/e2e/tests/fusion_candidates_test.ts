/**
 * Unit tests of the fusion candidate search (`./helpers/fusion-candidates.ts`) on synthetic IR (no
 * GPU, no real assets — the core lane). The real-asset checks live in the family fusion-count tests
 * (`e2e_<family>_fusion_counts_test.ts`).
 *
 * The graphs use ops no fusion rule grabs (`neg`, `add`, `mul`, `sigmoid` without the SiLU form,
 * `cat`, `reshape`), so every node stays a plain step and the windows are exactly the ones named.
 */

import { describe, expect, it } from "vite-plus/test";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { FUSION_RULES, planFusions } from "../../runtime/src/runtime/fusion.ts";
import { bindSymbols, countUses, planGraph } from "../../runtime/src/runtime/plan.ts";
import { mergeGraph } from "../../runtime/tests/helpers/merged-graph.ts";
import type { DeclarationJson } from "../../runtime/tests/helpers/model-fixture.ts";
import {
  type CandidateScenario,
  candidateRows,
  candidatesJsonl,
  candidatesMarkdown,
  candidateWindows,
  fusionCandidates,
  opSequenceCount,
} from "./helpers/fusion-candidates.ts";

type Node = DeclarationJson["nodes"][number];

/**
 * A graph over one input `x` of `shape`: every value `nodes` write has the shape `values` gives it
 * (default `shape`), and `outputs` are the graph outputs.
 */
const graphOf = (
  shape: readonly number[],
  nodes: readonly Node[],
  outputs: readonly string[],
  values: Readonly<Record<string, readonly number[]>> = {},
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...new Set(nodes.map((node) => node.op))] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [...shape] }],
  outputs: [...outputs],
  initializers: {},
  values: Object.fromEntries(
    nodes.flatMap((node) =>
      node.outs.map((name) => [name, { dtype: "f32", shape: [...(values[name] ?? shape)] }]),
    ),
  ),
  nodes: [...nodes],
});

/** The plan of `graph` under the built-in rules (the steps and the context the search reads). */
const planOf = (graph: DeclarationJson) => {
  const ir: IrGraph = mergeGraph(graph);
  const context = {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  };
  // The graphs here declare no symbol, so the declared input shapes are the binding.
  const shapes = Object.fromEntries(
    graph.inputs.map((input) => [
      input.name,
      input.shape.map((dim) => {
        if (typeof dim !== "number") throw new Error(`${input.name}: symbol '${dim}' unbound`);
        return dim;
      }),
    ]),
  );
  const plan = planFusions(planGraph(ir, bindSymbols(ir, shapes)).nodes, context);
  return { steps: plan.steps, context };
};

const neg = (input: string, output: string): Node => ({
  op: "neg",
  ins: [input],
  outs: [output],
  attrs: {},
});

/** `neg` five times in a row: x → a → b → c → d → e. */
const fiveChain = (): DeclarationJson =>
  graphOf(
    [4, 8],
    [neg("x", "a"), neg("a", "b"), neg("b", "c"), neg("c", "d"), neg("d", "e")],
    ["e"],
  );

describe("maximal windows", () => {
  it("a chain of five nodes gives one row of one window, not its ten n-gram windows", () => {
    const { steps, context } = planOf(fiveChain());
    const rows = fusionCandidates(steps, context);
    expect(rows.map((row) => [row.form, row.count])).toEqual([["neg,neg,neg,neg,neg", 1]]);
    expect(rows[0].nodeIndices).toEqual([0]);
    expect(rows[0].outputName).toBe("e");
  });

  it("a window as wide as the window limit is marked, and the mark goes when the limit is one more", () => {
    const { steps, context } = planOf(fiveChain());
    const at = (maxWindow: number) =>
      candidateWindows(steps, context, maxWindow).map((window) => [window.form, window.hitLimit]);
    expect(at(5)).toEqual([["neg,neg,neg,neg,neg", true]]);
    expect(at(6)).toEqual([["neg,neg,neg,neg,neg", false]]);
    // Below the chain's length the limit cuts it: every window left is as wide as the limit.
    expect(at(3)).toEqual([
      ["neg,neg,neg", true],
      ["neg,neg,neg", true],
      ["neg,neg,neg", true],
    ]);
  });
});

describe("bytes removed", () => {
  it("one f32 internal value of shape [2, 3] is 48 B: a write of 24 B and a read of 24 B", () => {
    const { steps, context } = planOf(graphOf([2, 3], [neg("x", "a"), neg("a", "b")], ["b"]));
    const rows = fusionCandidates(steps, context);
    expect(rows.map((row) => [row.form, row.bytes])).toEqual([["neg,neg", 48]]);
  });

  it("the output of an aliasing node (a reshape) counts 0: its producer's value carries the write and the read", () => {
    const { steps, context } = planOf(
      graphOf(
        [2, 3],
        [neg("x", "a"), { op: "reshape", ins: ["a"], outs: ["r"], attrs: {} }, neg("r", "b")],
        ["b"],
        { r: [3, 2], b: [3, 2] },
      ),
    );
    const aliased = steps.filter((step) => step.kind === "node" && step.aliasesInput);
    expect(aliased.length, "precondition: the plan aliases the reshape").toBe(1);
    const rows = fusionCandidates(steps, context);
    expect(rows.map((row) => [row.form, row.bytes, row.dispatches])).toEqual([
      ["neg,reshape,neg", 48, 1],
    ]);
  });
});

describe("dispatches removed", () => {
  it("are the chain's dispatches less the fused kernel's one, with a cat counted once per input", () => {
    const { steps, context } = planOf(
      graphOf(
        [2, 3],
        [
          neg("x", "a"),
          neg("x", "b"),
          { op: "cat", ins: ["a", "b"], outs: ["c"], attrs: { dim: 0 } },
        ],
        ["c"],
        { c: [4, 3] },
      ),
    );
    const rows = fusionCandidates(steps, context);
    // neg 1 + neg 1 + cat 2 − 1 = 3; the internal values a and b are 48 B each.
    expect(rows.map((row) => [row.form, row.dispatches, row.bytes])).toEqual([
      ["neg,neg,cat", 3, 96],
    ]);
  });
});

/**
 * Two chains with the ops `neg, add, mul`, each with one in-window passthrough (a `neg` of `x` whose
 * output is a graph output, so it cannot join the chain): the first holds it second, the second
 * third.
 */
const passthroughAt = (): DeclarationJson =>
  graphOf(
    [4, 8],
    [
      neg("x", "a"),
      neg("x", "t"),
      { op: "add", ins: ["a", "t"], outs: ["b"], attrs: {} },
      { op: "mul", ins: ["b", "x"], outs: ["c"], attrs: {} },
      neg("x", "a2"),
      { op: "add", ins: ["a2", "x"], outs: ["b2"], attrs: {} },
      neg("x", "t2"),
      { op: "mul", ins: ["b2", "t2"], outs: ["c2"], attrs: {} },
    ],
    ["t", "c", "t2", "c2"],
  );

describe("grouping", () => {
  it("windows with the same chain ops but a passthrough at another position are separate rows", () => {
    const { steps, context } = planOf(passthroughAt());
    const rows = fusionCandidates(steps, context);
    expect(rows.map((row) => [row.form, row.count, row.nodeIndices])).toEqual([
      ["neg,[neg],add,mul", 1, [0]],
      ["neg,add,[neg],mul", 1, [4]],
    ]);
    expect(rows.every((row) => row.ops.join(",") === "neg,add,mul")).toBe(true);
    expect(opSequenceCount(rows, ["neg", "add", "mul"])).toBe(2);
  });

  it("windows of one form add up: count, bytes and dispatches over the instances", () => {
    const { steps, context } = planOf(
      graphOf([2, 3], [neg("x", "a"), neg("a", "b"), neg("x", "c"), neg("c", "d")], ["b", "d"]),
    );
    const rows = fusionCandidates(steps, context);
    expect(
      rows.map((row) => [row.form, row.count, row.bytes, row.dispatches, row.nodeIndices]),
    ).toEqual([["neg,neg", 2, 96, 2, [0, 2]]]);
  });
});

describe("ranking and output", () => {
  /** Two chains of equal bytes and dispatches (`neg, neg` then `sigmoid, neg`), and a larger one. */
  const ranked = (): DeclarationJson =>
    graphOf(
      [2, 3],
      [
        neg("x", "a"),
        neg("a", "b"),
        { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
        neg("s", "n"),
        neg("x", "p"),
        neg("p", "q"),
        neg("q", "r"),
      ],
      ["b", "n", "r"],
    );

  it("ranks by bytes, then dispatches, then the first node index", () => {
    const { steps, context } = planOf(ranked());
    const rows = fusionCandidates(steps, context);
    expect(rows.map((row) => [row.form, row.bytes, row.nodeIndices[0]])).toEqual([
      ["neg,neg,neg", 96, 4],
      ["neg,neg", 48, 0],
      ["sigmoid,neg", 48, 2],
    ]);
  });

  it("two runs write the same Markdown and JSONL, and the table carries its statements", () => {
    const scenario: CandidateScenario = {
      graph: "synthetic",
      shapes: { x: [2, 3] },
      options: {},
      maxWindow: 12,
    };
    const run = () => {
      const { steps, context } = planOf(ranked());
      const rows = candidateRows(candidateWindows(steps, context, scenario.maxWindow));
      return [candidatesMarkdown(rows, scenario), candidatesJsonl(rows, scenario)];
    };
    const [markdown, jsonl] = run();
    expect(run()).toEqual([markdown, jsonl]);
    for (const statement of [
      "**Promises**",
      "**Does not promise**",
      "**Acceptance**",
      "`x` [2, 3]",
    ]) {
      expect(markdown).toContain(statement);
    }
    expect(
      jsonl
        .trimEnd()
        .split("\n")
        .map((line) => JSON.parse(line).rank),
    ).toEqual([1, 2, 3]);
  });
});
