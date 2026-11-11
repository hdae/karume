/**
 * The external entry of the rule definition (`defineExternalRule`, the `defineRule` of
 * `@karume/runtime/kernel`) and the brand every define entry puts on its rule, without a GPU: the
 * name form `<namespace>/<rule>`, the declarations, the brand a hand-written object or a copy lacks,
 * and the checks an external rule meets each time it builds a step — its window starts where it was
 * tried, it binds only its chain's own inputs, it writes the chain's last value, and every dispatch
 * key starts with `ext:<namespace>:`.
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import {
  defineExternalRule,
  defineRule,
  type ExternalRuleDefinition,
  type ExternalRuleName,
  type FusedBuild,
  type FusedOperand,
  type FusedWorkgroups,
  type FusionMatch,
  type FusionRule,
  type FusionRuleDefinition,
  ruleOrigin,
} from "../src/runtime/fusion-rule.ts";
import {
  type FusionContext,
  type FusionPlan,
  planFusions,
  sessionFusionRules,
} from "../src/runtime/fusion.ts";
import {
  bindSymbols,
  countUses,
  ExecutionError,
  type NodePlan,
  planGraph,
} from "../src/runtime/plan.ts";
import { fakeRecipeGpuContext } from "./helpers/fake-gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
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

/** The same chain with `y = abs(x)` between its two negs. */
const INTERLEAVED_GRAPH: DeclarationJson = {
  ...NEG_NEG_GRAPH,
  nodes: [
    { op: "neg", ins: ["z"], outs: ["n1"], attrs: {} },
    { op: "abs", ins: ["x"], outs: ["y"], attrs: {} },
    { op: "neg", ins: ["n1"], outs: ["n2"], attrs: {} },
  ],
};

type NegNegMatch = FusionMatch & { readonly input: string };

/** How a test rule's matcher and build go wrong (each a break of one invariant). */
type Fault = {
  /** The window starts one node after the index the rule was tried at. */
  readonly shiftedWindow?: boolean;
  /** The chain lists the two negs in reverse order. */
  readonly reversedChain?: boolean;
  /** The second neg is looked for one node further, and the window skips the node between. */
  readonly gappedWindow?: boolean;
  /** The step binds the chain's input twice. */
  readonly duplicateBind?: boolean;
  /** The step declares its output under the right name with the shape transposed. */
  readonly transposedOutput?: boolean;
  /** The step binds the graph input `x`, which the chain does not read. */
  readonly strayBind?: boolean;
  /** The step declares the internal value `n1` as its output. */
  readonly wrongOutput?: boolean;
  /** The dispatch key. */
  readonly key?: string;
  /** The step declares no dispatch. */
  readonly noDispatch?: boolean;
  /** The dispatch's operand list (omitted: the bind, then the output). */
  readonly operands?: readonly FusedOperand[];
  /** The dispatch's WGSL (omitted: none — a plan that is never built needs none). */
  readonly wgsl?: string;
  /** The dispatch binds its params as storage. */
  readonly paramsStorage?: boolean;
  /** The dispatch's workgroups (omitted: 32 items over workgroups of 64). */
  readonly workgroups?: FusedWorkgroups;
};

/** `neg(neg(z))` folded into one dispatch: the definition, with `fault` applied. */
const negNeg = <Name extends string>(name: Name, fault: Fault = {}) => ({
  name,
  heads: ["neg"],
  contract: "E" as const,
  internals: "private" as const,
  match: (nodes: readonly NodePlan[], index: number): NegNegMatch | undefined => {
    const first = nodes[index];
    const second = nodes[index + (fault.gappedWindow === true ? 2 : 1)];
    if (first?.node.op !== "neg" || second?.node.op !== "neg") return undefined;
    if (second.node.ins[0] !== first.outputs[0].name) return undefined;
    const chain = [first, second];
    return {
      window: fault.shiftedWindow === true ? [second] : chain,
      chain:
        fault.shiftedWindow === true
          ? [second]
          : fault.reversedChain === true
            ? [second, first]
            : chain,
      input: first.node.ins[0],
    };
  },
  build: (matched: NegNegMatch): FusedBuild => {
    const last = matched.chain[matched.chain.length - 1];
    return {
      binds:
        fault.strayBind === true
          ? [matched.input, "x"]
          : fault.duplicateBind === true
            ? [matched.input, matched.input]
            : [matched.input],
      outputName: fault.wrongOutput === true ? "n1" : last.outputs[0].name,
      outputShape:
        fault.transposedOutput === true
          ? last.outputs[0].shape.toReversed()
          : last.outputs[0].shape,
      temps: [],
      dispatches:
        fault.noDispatch === true
          ? []
          : [
              {
                key: fault.key ?? "ext:test:neg_neg",
                wgsl: () => fault.wgsl ?? "",
                params: new Uint32Array(4),
                ...(fault.paramsStorage === undefined
                  ? {}
                  : { paramsStorage: fault.paramsStorage }),
                ...(fault.operands === undefined ? {} : { operands: fault.operands }),
                workgroups: fault.workgroups ?? { kind: "gridStride", items: 32, size: 64 },
              },
            ],
    };
  },
});

/**
 * `rewrite` changes the planned nodes (a dtype the graph's declaration cannot carry); `facts` are
 * the Session facts the context carries besides the graph's.
 */
const plan = (
  rules: readonly FusionRule[],
  rewrite: (node: NodePlan) => NodePlan = (node) => node,
  declaration: DeclarationJson = NEG_NEG_GRAPH,
  facts: Pick<FusionContext, "weightsNotF32"> = {},
): FusionPlan<string> => {
  const graph = mergeGraph(declaration);
  const nodes = planGraph(graph, bindSymbols(graph, { x: SHAPE, z: SHAPE })).nodes.map(rewrite);
  return planFusions(nodes, {
    rules,
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    ...facts,
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

  it("refuses, through either define entry, heads that are not a non-empty array of op names and a match or build that is not a function", () => {
    for (const [label, fault, message] of [
      ["heads not an array", { heads: 5 }, "heads must be a non-empty array"],
      ["empty heads", { heads: [] }, "heads must be a non-empty array"],
      ["a head not a string", { heads: ["neg", 7] }, 'got ["neg", 7]'],
      ["an empty head", { heads: [""] }, "heads must be a non-empty array"],
      ["no match", { match: undefined }, "match must be a function"],
      ["build not a function", { build: {} }, "build must be a function"],
    ] as const) {
      assertThrows(
        () =>
          defineExternalRule(
            outsideTheType<ExternalRuleDefinition<NegNegMatch>>({
              ...negNeg("test/neg_neg"),
              ...fault,
            }),
          ),
        ExecutionError,
        message,
        label,
      );
      assertThrows(
        () =>
          defineRule(
            outsideTheType<FusionRuleDefinition<NegNegMatch>>({
              ...negNeg("testNegNeg"),
              ...fault,
            }),
          ),
        ExecutionError,
        message,
        label,
      );
    }
  });

  it("refuses, through either define entry, a definition that is not an object", () => {
    for (const given of [null, undefined, "test/neg_neg"]) {
      assertThrows(
        () => defineExternalRule(outsideTheType<ExternalRuleDefinition<NegNegMatch>>(given)),
        ExecutionError,
        "fusion rule definition: expected an object",
        String(given),
      );
      assertThrows(
        () => defineRule(outsideTheType<FusionRuleDefinition<NegNegMatch>>(given)),
        ExecutionError,
        "fusion rule definition: expected an object",
        String(given),
      );
    }
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

  it("keeps the definition as it was when it ran: editing the caller's object afterwards changes nothing the rule does", () => {
    const definition = negNeg<ExternalRuleName>("test/neg_neg");
    const rule = defineExternalRule(definition);
    definition.name = "test/later";
    definition.internals = outsideTheType("anything");
    definition.match = () => undefined;
    const folded = plan([rule]);
    assertEquals(
      folded.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["abs", "test/neg_neg"],
    );
    assertEquals(folded.counts["test/neg_neg"], 1);
  });
});

describe("an external rule building a step", () => {
  it("gives its matcher only the graph's use counts and outputs, one frozen context per scan", () => {
    const seen: unknown[] = [];
    const definition = negNeg("test/neg_neg");
    plan([
      defineExternalRule({
        ...definition,
        heads: ["abs", "neg"],
        match: (nodes, index, context) => {
          seen.push(context);
          return definition.match(nodes, index);
        },
      }),
    ]);
    // Tried at the abs (index 0) and the first neg (index 1, where it folds both negs).
    assertEquals(seen.length, 2);
    const [context] = seen;
    assert(typeof context === "object" && context !== null);
    assertEquals(Object.keys(context).sort(), ["outputNames", "useCounts"]);
    assert(Object.isFrozen(context), "the context is frozen");
    assert(seen[1] === context, "both tries of one scan read the same context");
  });

  it("is tried only at a node whose op is one of its heads", () => {
    const tried: string[] = [];
    const definition = negNeg("test/neg_neg");
    const folded = plan([
      defineExternalRule({
        ...definition,
        heads: ["abs"],
        match: (nodes, index) => {
          tried.push(nodes[index].node.op);
          return definition.match(nodes, index);
        },
      }),
    ]);
    assertEquals(tried, ["abs"]);
    assertEquals(
      folded.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["abs", "neg", "neg"],
    );
    assertEquals(folded.counts["test/neg_neg"], 0);
  });

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

  it("does not fold a chain with a node that is not f32 (the runtime checks it, not the matcher)", () => {
    const rule = defineExternalRule(negNeg("test/neg_neg"));
    // The second neg reads and writes i32 words; the matcher does not look at dtypes.
    const folded = plan([rule], (node) =>
      node.outputs[0].name === "n2"
        ? {
            ...node,
            inputDtypes: ["i32"],
            outputs: node.outputs.map((out) => ({ ...out, dtype: "i32" })),
          }
        : node,
    );
    assertEquals(
      folded.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["abs", "neg", "neg"],
    );
    assertEquals(folded.counts["test/neg_neg"], 0);
  });

  it("fails loudly on a window that skips a node, a bind given twice, and an output of the right name with another shape", () => {
    assertThrows(
      () =>
        plan(
          [defineExternalRule(negNeg("test/neg_neg", { gappedWindow: true }))],
          undefined,
          INTERLEAVED_GRAPH,
        ),
      ExecutionError,
      "is not the run of nodes that starts at node 0",
      "gapped window",
    );
    assertThrows(
      () => plan([defineExternalRule(negNeg("test/neg_neg", { duplicateBind: true }))]),
      ExecutionError,
      "bind(s) 'z' are not distinct external inputs",
      "duplicate bind",
    );
    assertThrows(
      () => plan([defineExternalRule(negNeg("test/neg_neg", { transposedOutput: true }))]),
      ExecutionError,
      "the declared output 'n2' [8,4] is not the folded chain's last value 'n2' [4,8]",
      "transposed output",
    );
  });

  it("fails loudly on a chain that does not follow the window's node order", () => {
    assertThrows(
      () => plan([defineExternalRule(negNeg("test/neg_neg", { reversedChain: true }))]),
      ExecutionError,
      "the folded chain does not follow the window's node order (its nodes sit at window positions 1, 0)",
    );
  });

  it("fails loudly on a chain that does not start at its window's first node (a leading passthrough would fold ops its heads never name)", () => {
    const definition = negNeg("test/neg_neg");
    // Tried at the abs (its only head), it folds the two negs after it and keeps the abs in front.
    const leading = defineExternalRule({
      ...definition,
      heads: ["abs"],
      match: (nodes, index) => {
        const negs = definition.match(nodes, index + 1);
        return negs === undefined ? undefined : { ...negs, window: [nodes[index], ...negs.window] };
      },
    });
    assertThrows(
      () => plan([leading]),
      ExecutionError,
      "fusion rule 'test/neg_neg': the folded chain starts at window position 1, not at the node it was tried at",
    );
  });

  it("fails loudly on a step whose dispatches never bind its output (its buffer would never be written)", () => {
    assertThrows(
      () =>
        plan([
          defineExternalRule(negNeg("test/neg_neg", { operands: [{ kind: "bind", index: 0 }] })),
        ]),
      ExecutionError,
      "no dispatch binds the step's output 'n2'",
    );
  });

  it("does not fold a chain whose output has no element (the binding sized it, not the rule)", () => {
    const folded = plan([defineExternalRule(negNeg("test/neg_neg"))], (node) =>
      node.outputs[0].name === "n2"
        ? { ...node, outputs: node.outputs.map((out) => ({ ...out, shape: [0, 8] })) }
        : node,
    );
    assertEquals(
      folded.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["abs", "neg", "neg"],
    );
    assertEquals(folded.counts["test/neg_neg"], 0);
  });

  it("does not fold a chain whose step binds a weight not resident as f32 (stored in another form, borrowed or pooled)", () => {
    // The Session lists those weights by name (`sessionFusionContext`); the name is all the pass reads.
    const rule = defineExternalRule(negNeg("test/neg_neg"));
    const kept = plan([rule], undefined, NEG_NEG_GRAPH, { weightsNotF32: new Set(["z"]) });
    assertEquals(
      kept.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["abs", "neg", "neg"],
    );
    assertEquals(kept.counts["test/neg_neg"], 0);
    const folded = plan([rule], undefined, NEG_NEG_GRAPH, { weightsNotF32: new Set(["x"]) });
    assertEquals(folded.counts["test/neg_neg"], 1, "the control: a weight the step does not bind");
  });

  it("fails loudly on a step that binds a weightScale operand (an external kernel reads f32 values only)", () => {
    assertThrows(
      () =>
        plan([
          defineExternalRule(
            negNeg("test/neg_neg", {
              operands: [{ kind: "weightScale", index: 0 }, { kind: "output" }],
            }),
          ),
        ]),
      ExecutionError,
      "fusion rule 'test/neg_neg': dispatch 0 ('ext:test:neg_neg') binds a weightScale operand",
    );
  });

  it("fails loudly on a step that declares no dispatch (its output would never be written)", () => {
    assertThrows(
      () => plan([defineExternalRule(negNeg("test/neg_neg", { noDispatch: true }))]),
      ExecutionError,
      "the step declares no dispatch",
    );
  });

  it("names the rule, keeping what was thrown as the cause, when its match or build throws or returns a value of the wrong shape", () => {
    const definition = negNeg("test/neg_neg");
    const thrown = new RangeError("a bug in the matcher");
    for (const [label, fault] of [
      [
        "a match that throws",
        {
          match: () => {
            throw thrown;
          },
        },
      ],
      [
        "a match that throws a string",
        {
          match: () => {
            throw "no";
          },
        },
      ],
      ["a match result that is not a match", { match: () => 5 }],
      [
        "a window that is not an array",
        {
          match: (nodes: readonly NodePlan[], index: number) => {
            const matched = definition.match(nodes, index);
            return matched === undefined ? undefined : { ...matched, window: "window" };
          },
        },
      ],
      ["no build result", { build: () => undefined }],
      [
        "binds that are not an array",
        { build: (matched: NegNegMatch) => ({ ...definition.build(matched), binds: "z" }) },
      ],
      [
        "no temporaries",
        { build: (matched: NegNegMatch) => ({ ...definition.build(matched), temps: undefined }) },
      ],
    ] as const) {
      const error = assertThrows(
        () =>
          plan([
            defineExternalRule(
              outsideTheType<ExternalRuleDefinition<NegNegMatch>>({ ...definition, ...fault }),
            ),
          ]),
        ExecutionError,
        "fusion rule 'test/neg_neg': its match or build, or the runtime reading what they returned, threw",
        label,
      );
      assert(error.cause !== undefined && !(error.cause instanceof ExecutionError), label);
    }
    const error = assertThrows(
      () =>
        plan([
          defineExternalRule({
            ...definition,
            match: () => {
              throw thrown;
            },
          }),
        ]),
      ExecutionError,
      "threw RangeError: a bug in the matcher",
    );
    assert(error.cause === thrown, "the original is the cause");
  });

  it("calls its match and build as standalone functions: a method that reads `this` fails loudly, naming the rule, instead of changing what it does", () => {
    const definition = negNeg("test/neg_neg");
    for (const [label, methods] of [
      [
        "match",
        {
          fold: true,
          match(nodes: readonly NodePlan[], index: number) {
            return this.fold ? definition.match(nodes, index) : undefined;
          },
        },
      ],
      [
        "build",
        {
          shape: SHAPE,
          build(matched: NegNegMatch) {
            return { ...definition.build(matched), outputShape: this.shape };
          },
        },
      ],
    ] as const) {
      const error = assertThrows(
        () => plan([defineExternalRule({ ...definition, ...methods })]),
        ExecutionError,
        "fusion rule 'test/neg_neg': its match or build, or the runtime reading what they returned, threw TypeError",
        label,
      );
      assert(error.cause instanceof TypeError, label);
    }
  });

  it("is checked as an external rule only: the same faults through defineRule are the built-in rules' own responsibility", () => {
    // The built-in entry keeps the checks it had; a key without the prefix folds as before.
    const folded = plan([defineRule(negNeg("testNegNeg", { key: "test:neg_neg" }))]);
    assertEquals(folded.counts.testNegNeg, 1);
  });
});

describe("an external rule and the packed int8 activations (ADR 0105)", () => {
  const ROWS = 1;
  const K = 12288;
  const N = 1536;
  /** `y = linear(static_quantize(src), w, b)` on an i2 weight: a pair the parallel GEMV takes packed. */
  const PACKED_GRAPH: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "static_quantize"] },
    symbols: [],
    inputs: [{ name: "src", dtype: "f32", shape: [ROWS, K] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [N, K] },
      b: { dtype: "f32", shape: [N] },
      xq: { dtype: "f32", shape: [ROWS, K] },
      y: { dtype: "f32", shape: [ROWS, N] },
    },
    nodes: [
      { op: "static_quantize", ins: ["src"], outs: ["xq"], attrs: { scale: Math.fround(0.00071) } },
      { op: "linear", ins: ["xq", "w", "b"], outs: ["y"], attrs: {} },
    ],
  };

  const planPacked = (rules: readonly FusionRule[]): FusionPlan<string> => {
    const graph = mergeGraph(PACKED_GRAPH, { w: { codec: "int2-off" } });
    return planFusions(planGraph(graph, {}).nodes, {
      rules,
      useCounts: countUses(graph),
      outputNames: new Set(graph.outputs),
      limits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
      packedStaticQuantize: true,
      linearGemvReduce: "parallel",
      linearCompute: "f32",
      weightLayouts: new Map([["w", { storage: "i2" }]]),
    });
  };

  /**
   * A rule folding the one `op` node, from its second try at such a node on: it declines on the
   * first scan, where the pairing is decided, and folds on the second.
   */
  const lateRule = (op: string, name: ExternalRuleName) => {
    let tries = 0;
    return defineExternalRule<FusionMatch, ExternalRuleName>({
      name,
      heads: [op],
      contract: "E",
      internals: "private",
      match: (nodes, index) => {
        const node = nodes[index];
        if (node?.node.op !== op) return undefined;
        tries += 1;
        return tries >= 2 ? { window: [node], chain: [node] } : undefined;
      },
      build: (matched) => ({
        binds: [...new Set(matched.chain[0].node.ins)],
        outputName: matched.chain[0].outputs[0].name,
        outputShape: matched.chain[0].outputs[0].shape,
        temps: [],
        dispatches: [
          {
            key: `ext:${name.split("/")[0]}:late`,
            wgsl: () => "",
            params: new Uint32Array(4),
            workgroups: { kind: "gridStride", items: 32, size: 64 },
          },
        ],
      }),
    });
  };

  it("pairs the static_quantize with the linear without an external rule (the control)", () => {
    const planned = planPacked([]);
    assertEquals(
      planned.steps.map((step) =>
        step.kind === "node" ? `${step.plan.node.op}:${step.packedActivations?.role}` : step.rule,
      ),
      ["static_quantize:write", "linear:read"],
    );
  });

  it("fails loudly, naming the rule, when its step folds the packed producer or the packed consumer on the final scan", () => {
    for (const [op, name, verb] of [
      ["static_quantize", "test/late_producer", "writes"],
      ["linear", "test/late_consumer", "reads"],
    ] as const) {
      assertThrows(
        () => planPacked([lateRule(op, name)]),
        ExecutionError,
        `fusion rule '${name}': its step folds the '${op}' node that ${verb} 'xq', a value handed over as packed int8`,
        op,
      );
    }
  });

  /**
   * `h = linear(src, w0, b0)` before the pair: the linear→SRQ rule can fold `h`'s linear with the
   * static_quantize, which then writes f32 words, not packed codes.
   */
  const LEADING_LINEAR_GRAPH: DeclarationJson = {
    ...PACKED_GRAPH,
    inputs: [{ name: "src", dtype: "f32", shape: [ROWS, N] }],
    initializers: { w0: {}, b0: {}, w: {}, b: {} },
    values: {
      ...PACKED_GRAPH.values,
      w0: { dtype: "f32", shape: [K, N] },
      b0: { dtype: "f32", shape: [K] },
      h: { dtype: "f32", shape: [ROWS, K] },
    },
    nodes: [
      { op: "linear", ins: ["src", "w0", "b0"], outs: ["h"], attrs: {} },
      { op: "static_quantize", ins: ["h"], outs: ["xq"], attrs: { scale: Math.fround(0.00071) } },
      { op: "linear", ins: ["xq", "w", "b"], outs: ["y"], attrs: {} },
    ],
  };

  it("fails loudly, naming the external rules, when a matcher's answer differs between the scans and a built-in rule then takes a paired value", () => {
    const graph = mergeGraph(LEADING_LINEAR_GRAPH, {
      w0: { codec: "int2-off" },
      w: { codec: "int2-off" },
    });
    const planWith = (rules: readonly FusionRule[]): FusionPlan<string> =>
      planFusions(planGraph(graph, {}).nodes, {
        rules,
        useCounts: countUses(graph),
        outputNames: new Set(graph.outputs),
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
          maxComputeWorkgroupsPerDimension: 65535,
        },
        packedStaticQuantize: true,
        fuseLinearStaticQuantize: true,
        linearGemvReduce: "parallel",
        linearCompute: "f32",
        weightLayouts: new Map([
          ["w0", { storage: "i2" }],
          ["w", { storage: "i2" }],
        ]),
      });
    // The controls: the built-in rules alone fold the leading linear with the static_quantize
    // (nothing is paired), and a rule that folds the leading linear on both scans leaves the pair.
    assertEquals(
      planWith(sessionFusionRules(undefined)).steps.map((step) =>
        step.kind === "node" ? `${step.plan.node.op}:${step.packedActivations?.role}` : step.rule,
      ),
      ["linearStaticQuantize", "linear:undefined"],
    );
    const leading = (name: ExternalRuleName, folds: () => boolean) =>
      defineExternalRule<FusionMatch, ExternalRuleName>({
        name,
        heads: ["linear"],
        contract: "E",
        internals: "private",
        match: (nodes, index) => {
          const node = nodes[index];
          return node.node.ins[0] === "src" && folds()
            ? { window: [node], chain: [node] }
            : undefined;
        },
        build: (matched) => ({
          binds: [...matched.chain[0].node.ins],
          outputName: "h",
          outputShape: [ROWS, K],
          temps: [],
          dispatches: [
            {
              key: `ext:${name.split("/")[0]}:leading`,
              wgsl: () => "",
              params: new Uint32Array(4),
              workgroups: { kind: "gridStride", items: 32, size: 64 },
            },
          ],
        }),
      });
    assertEquals(
      planWith(sessionFusionRules([leading("test/always", () => true)])).steps.map((step) =>
        step.kind === "node" ? `${step.plan.node.op}:${step.packedActivations?.role}` : step.rule,
      ),
      ["test/always", "static_quantize:write", "linear:read"],
    );
    // Folds on its first try only: the first scan leaves the static_quantize plain (so `xq` is
    // paired), and on the second the linear→SRQ rule takes it.
    let tries = 0;
    const once = leading("test/once", () => {
      tries += 1;
      return tries === 1;
    });
    assertThrows(
      () => planWith(sessionFusionRules([once])),
      ExecutionError,
      "fusion rules: the plan changed between the two planning scans (the packed int8 activations of ADR 0105" +
        " were planned as 'xq' from the first scan, and the final plan gives none): a rule's matcher is not a" +
        " pure function of its arguments — the external rules in the list are 'test/once'",
    );
  });
});

describe("an external step's recipe (a fake GPU: the run stops at the transient plan, after every recipe)", () => {
  /** The bind group of the neg-neg step: params, the chain's input, the output. */
  const wgslOf = (
    params: string,
    input: "read" | "read_write",
    output: "read" | "read_write" = "read_write",
  ): string =>
    `@group(0) @binding(0) ${params} params: vec4<u32>;
@group(0) @binding(1) var<storage, ${input}> src: array<f32>;
@group(0) @binding(2) var<storage, ${output}> out: array<f32>;
@compute @workgroup_size(64) fn main() {}`;

  const run = async (fault: Fault): Promise<unknown> => {
    const session = await createSessionFromContainer(
      fakeRecipeGpuContext(),
      await openModelBytes(NEG_NEG_GRAPH, []),
      GRAPH_NAME,
      { fusionRules: [defineExternalRule(negNeg("test/neg_neg", fault))] },
    );
    return await session.run({ x: fill(SHAPE, (i) => i), z: fill(SHAPE, (i) => -i) });
  };

  it("builds the recipe of a step that writes only its output (the control: the run reaches the transient plan)", async () => {
    await assertRejects(
      () => run({ wgsl: wgslOf("var<uniform>", "read") }),
      Error,
      "invalid transient plan limit",
    );
  });

  it("fails loudly, naming the rule, on workgroup counts that are not positive integers within the device limit", async () => {
    const wgsl = wgslOf("var<uniform>", "read");
    await assertRejects(
      () => run({ wgsl, workgroups: { kind: "tiled", counts: [1, 1, 1] } }),
      Error,
      "invalid transient plan limit",
      "the control: one workgroup",
    );
    for (const [label, workgroups, message] of [
      [
        "no tile",
        { kind: "tiled", counts: [0, 1, 1] },
        "resolves to the workgroup counts [0, 1, 1]",
      ],
      [
        "over the limit",
        { kind: "tiled", counts: [65536, 1, 1] },
        "resolves to the workgroup counts [65536, 1, 1]",
      ],
      [
        "a fraction",
        { kind: "tiled", counts: [1.5, 1, 1] },
        "resolves to the workgroup counts [1.5, 1, 1]",
      ],
      [
        "no item",
        { kind: "gridStride", items: 0, size: 64 },
        "resolves to the workgroup counts [0, 1, 1]",
      ],
      [
        "no workgroup size",
        { kind: "gridStride", items: 32, size: 0 },
        "declares the grid-stride workgroup size 0",
      ],
    ] as const) {
      await assertRejects(
        () => run({ wgsl, workgroups }),
        ExecutionError,
        `fusion rule 'test/neg_neg': dispatch 0 ('ext:test:neg_neg') ${message}`,
        label,
      );
    }
  });

  it("names the rule, keeping what was thrown as the cause, when resolving a dispatch fails outside its own checks", async () => {
    const wgsl = wgslOf("var<uniform>", "read");
    for (const [label, workgroups, message] of [
      [
        "an element count the grid-stride resolver refuses",
        { kind: "gridStride", items: 2 ** 32, size: 64 },
        "threw DispatchLimitError",
      ],
      [
        "tiled counts missing",
        outsideTheType<FusedWorkgroups>({ kind: "tiled" }),
        "threw TypeError",
      ],
    ] as const) {
      const error = await assertRejects(
        () => run({ wgsl, workgroups }),
        ExecutionError,
        `fusion rule 'test/neg_neg': resolving its step's dispatches ${message}`,
        label,
      );
      assert(error.cause instanceof Error && !(error.cause instanceof ExecutionError), label);
    }
  });

  it("fails loudly, naming the rule, on a step whose dispatches bind the output but never declare it read_write", async () => {
    await assertRejects(
      () => run({ wgsl: wgslOf("var<uniform>", "read", "read") }),
      ExecutionError,
      "fusion rule 'test/neg_neg': no dispatch declares the binding of the step's output 'n2' read_write",
    );
  });

  it("fails loudly, naming the rule and the binding, on a kernel that declares an input or the params read_write", async () => {
    for (const [label, fault, message] of [
      [
        "an input",
        { wgsl: wgslOf("var<uniform>", "read_write") },
        "declares binding 1 read_write, but it binds the input 'z'",
      ],
      [
        "the params",
        { wgsl: wgslOf("var<storage, read_write>", "read"), paramsStorage: true },
        "declares binding 0 read_write, but it binds the params",
      ],
    ] as const) {
      await assertRejects(
        () => run(fault),
        ExecutionError,
        `fusion rule 'test/neg_neg': dispatch 0 ('ext:test:neg_neg') ${message}`,
        label,
      );
    }
  });
});
