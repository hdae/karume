/**
 * The external entry of the rule definition (`defineExternalRule`, the `defineRule` of
 * `@karume/runtime/kernel`) and the brand every define entry puts on its rule, without a GPU: the
 * name form `<namespace>/<rule>`, the declarations, the brand a hand-written object or a copy lacks,
 * and the checks an external rule meets each time it builds a step — its window starts where it was
 * tried, it binds only its chain's own inputs, it writes the chain's last value, and every dispatch
 * key starts with `ext:<namespace>:`.
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  defineExternalRule,
  defineRule,
  type ExternalRuleName,
  type FusedBuild,
  type FusionMatch,
  type FusionRule,
  ruleOrigin,
} from "../src/runtime/fusion-rule.ts";
import { type FusionPlan, planFusions } from "../src/runtime/fusion.ts";
import {
  bindSymbols,
  countUses,
  ExecutionError,
  type NodePlan,
  planGraph,
} from "../src/runtime/plan.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

const SHAPE = [4, 8];

/** `n2 = neg(neg(z))` beside `y = abs(x)`. */
const NEG_NEG_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["abs", "neg"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: SHAPE },
    { name: "z", dtype: "f32", shape: SHAPE },
  ],
  outputs: ["y", "n2"],
  initializers: {},
  values: {
    y: { dtype: "f32", shape: SHAPE },
    n1: { dtype: "f32", shape: SHAPE },
    n2: { dtype: "f32", shape: SHAPE },
  },
  nodes: [
    { op: "abs", ins: ["x"], outs: ["y"], attrs: {} },
    { op: "neg", ins: ["z"], outs: ["n1"], attrs: {} },
    { op: "neg", ins: ["n1"], outs: ["n2"], attrs: {} },
  ],
};

type NegNegMatch = FusionMatch & { readonly input: string };

/** How a test rule's matcher and build go wrong (each a break of one invariant). */
type Fault = {
  /** The window starts one node after the index the rule was tried at. */
  readonly shiftedWindow?: boolean;
  /** The step binds the graph input `x`, which the chain does not read. */
  readonly strayBind?: boolean;
  /** The step declares the internal value `n1` as its output. */
  readonly wrongOutput?: boolean;
  /** The dispatch key. */
  readonly key?: string;
  /** The step declares no dispatch. */
  readonly noDispatch?: boolean;
};

/** `neg(neg(z))` folded into one dispatch: the definition, with `fault` applied. */
const negNeg = <Name extends string>(name: Name, fault: Fault = {}) => ({
  name,
  heads: ["neg"],
  contract: "E" as const,
  internals: "private" as const,
  match: (nodes: readonly NodePlan[], index: number): NegNegMatch | undefined => {
    const first = nodes[index];
    const second = nodes[index + 1];
    if (first?.node.op !== "neg" || second?.node.op !== "neg") return undefined;
    if (second.node.ins[0] !== first.outputs[0].name) return undefined;
    const chain = [first, second];
    return {
      window: fault.shiftedWindow === true ? [second] : chain,
      chain: fault.shiftedWindow === true ? [second] : chain,
      input: first.node.ins[0],
    };
  },
  build: (matched: NegNegMatch): FusedBuild => {
    const last = matched.chain[matched.chain.length - 1];
    return {
      binds: fault.strayBind === true ? [matched.input, "x"] : [matched.input],
      outputName: fault.wrongOutput === true ? "n1" : last.outputs[0].name,
      outputShape: last.outputs[0].shape,
      temps: [],
      dispatches:
        fault.noDispatch === true
          ? []
          : [
              {
                key: fault.key ?? "ext:test:neg_neg",
                wgsl: () => "",
                params: new Uint32Array(4),
                workgroups: { kind: "gridStride", items: 32, size: 64 },
              },
            ],
    };
  },
});

const plan = (rules: readonly FusionRule[]): FusionPlan<string> => {
  const graph = mergeGraph(NEG_NEG_GRAPH);
  return planFusions(planGraph(graph, bindSymbols(graph, { x: SHAPE, z: SHAPE })).nodes, {
    rules,
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  });
};

/** A JavaScript caller's value outside the type. */
const outsideTheType = <T>(value: unknown): T => value as T;

describe("defineExternalRule", () => {
  it("takes a name '<namespace>/<rule>' of two lowercase identifiers, and refuses every other form", () => {
    assertEquals(defineExternalRule(negNeg("test/neg_neg")).name, "test/neg_neg");
    for (const name of [
      "negNeg",
      "test/neg/neg",
      "Test/neg",
      "test/Neg",
      "test/",
      "/neg",
      "1test/neg",
      "test/neg-neg",
      "test:neg",
    ]) {
      assertThrows(
        () => defineExternalRule(negNeg(outsideTheType<ExternalRuleName>(name))),
        ExecutionError,
        "'<namespace>/<rule>'",
        name,
      );
    }
  });

  it("refuses a contract class or a privacy outside the declared values", () => {
    assertThrows(
      () => defineExternalRule({ ...negNeg("test/neg_neg"), contract: outsideTheType("X") }),
      ExecutionError,
      "contract",
    );
    assertThrows(
      () => defineExternalRule({ ...negNeg("test/neg_neg"), internals: outsideTheType("shared") }),
      ExecutionError,
      "internals",
    );
  });

  it("brands its rule external and frozen, where defineRule brands it built-in; an object written by hand or copied has no brand", () => {
    const external = defineExternalRule(negNeg("test/neg_neg"));
    const builtIn = defineRule(negNeg("testNegNeg"));
    assertEquals(ruleOrigin(external), "external");
    assertEquals(ruleOrigin(builtIn), "built-in");
    assert(Object.isFrozen(external) && Object.isFrozen(external.heads), "the rule is frozen");
    const literal: FusionRule = {
      name: "test/neg_neg",
      heads: ["neg"],
      contract: "E",
      apply: external.apply,
    };
    assertEquals(ruleOrigin(literal), undefined, "a hand-written rule");
    assertEquals(ruleOrigin({ ...external }), undefined, "a spread copy");
    assertEquals(ruleOrigin(undefined), undefined);
  });
});

describe("an external rule building a step", () => {
  it("folds its chain when every check holds", () => {
    const folded = plan([defineExternalRule(negNeg("test/neg_neg"))]);
    assertEquals(
      folded.steps.map((step) =>
        step.kind === "fused" ? `fused:${step.rule}` : step.plan.node.op,
      ),
      ["abs", "fused:test/neg_neg"],
    );
    assertEquals(folded.counts["test/neg_neg"], 1);
  });

  it("fails loudly on a dispatch key outside 'ext:<namespace>:', a key of another namespace included", () => {
    for (const key of ["neg_neg", "test:neg_neg", "ext:other:neg_neg", "ext:test"]) {
      assertThrows(
        () => plan([defineExternalRule(negNeg("test/neg_neg", { key }))]),
        ExecutionError,
        "do not start with 'ext:test:'",
        key,
      );
    }
  });

  it("fails loudly on a window that does not start where it was tried, a bind outside its chain's inputs, and an output other than the chain's last value", () => {
    for (const [label, fault, message] of [
      ["shifted window", { shiftedWindow: true }, "is not the run of nodes that starts at node"],
      ["stray bind", { strayBind: true }, "bind(s) 'x' are not distinct external inputs"],
      ["wrong output", { wrongOutput: true }, "is not the folded chain's last value 'n2'"],
    ] as const) {
      assertThrows(
        () => plan([defineExternalRule(negNeg("test/neg_neg", fault))]),
        ExecutionError,
        message,
        label,
      );
    }
  });

  it("fails loudly on a step that declares no dispatch (its output would never be written)", () => {
    assertThrows(
      () => plan([defineExternalRule(negNeg("test/neg_neg", { noDispatch: true }))]),
      ExecutionError,
      "the step declares no dispatch",
    );
  });

  it("is checked as an external rule only: the same faults through defineRule are the built-in rules' own responsibility", () => {
    // The built-in entry keeps the checks it had; a key without the prefix folds as before.
    const folded = plan([defineRule(negNeg("testNegNeg", { key: "test:neg_neg" }))]);
    assertEquals(folded.counts.testNegNeg, 1);
  });
});
