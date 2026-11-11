// `SessionOptions.fusionRules` — fusion rules written outside the runtime (ADR 0148) — is checked at
// Session construction, before any GPU work, so a fake device is enough: every entry is a rule the
// external define entry built (a hand-written object, a copy and a built-in rule are refused), no
// name is given twice, and the list the Session plans with puts the given rules before the built-in
// ones, in their order. The test-only DISABLED_FUSION_RULES takes an external rule's name on a
// Session that was given the rule. What the rules do on a real GPU is covered by
// tests/gpu_external_fusion_rule_test.ts.

import { describe, it } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import {
  createSessionFromContainer,
  DISABLED_FUSION_RULES,
  type SessionOptions,
} from "../src/runtime/executor.ts";
import {
  defineExternalRule,
  FUSION_RULES,
  type FusionMatch,
  type FusionRule,
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
import { buildSessionState } from "../src/runtime/session-build.ts";
import { fakeDevice, fakeGpuContext } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import { type DeclarationJson, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

const SHAPE = [8];

/** `y = x * sigmoid(x)` (the built-in silu rule's form). */
const SILU_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["sigmoid", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: SHAPE }],
  outputs: ["y"],
  initializers: {},
  values: { s: { dtype: "f32", shape: SHAPE }, y: { dtype: "f32", shape: SHAPE } },
  nodes: [
    { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["x", "s"], outs: ["y"], attrs: {} },
  ],
};

type SigmoidMulMatch = FusionMatch & { readonly input: string };

/**
 * An external rule that takes the silu form (`sigmoid → mul`, head `sigmoid` like the built-in
 * silu rule). These plans are never executed, so the dispatch only has to be well formed.
 */
const sigmoidMulRule = (name: `${string}/${string}`): FusionRule =>
  defineExternalRule<SigmoidMulMatch, typeof name>({
    name,
    heads: ["sigmoid"],
    contract: "E",
    internals: "private",
    match: (nodes: readonly NodePlan[], index: number) => {
      const sigmoid = nodes[index];
      const mul = nodes[index + 1];
      if (sigmoid?.node.op !== "sigmoid" || mul?.node.op !== "mul") return undefined;
      if (!mul.node.ins.includes(sigmoid.outputs[0].name)) return undefined;
      const chain = [sigmoid, mul];
      return { window: chain, chain, input: sigmoid.node.ins[0] };
    },
    build: (matched) => ({
      binds: [matched.input],
      outputName: matched.chain[1].outputs[0].name,
      outputShape: matched.chain[1].outputs[0].shape,
      temps: [],
      dispatches: [
        {
          key: `ext:${name.split("/")[0]}:sigmoid_mul`,
          wgsl: () => "",
          params: new Uint32Array(4),
          workgroups: { kind: "gridStride", items: 8, size: 64 },
        },
      ],
    }),
  });

const RULE = sigmoidMulRule("test/sigmoid_mul");

/** What the build phase reads once: the offset alignment of a fake device. */
const fakeGpu = () =>
  fakeGpuContext(Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } }));

const construct = async (options: SessionOptions) =>
  await createSessionFromContainer(
    fakeGpu(),
    await openModelBytes(SILU_GRAPH, []),
    GRAPH_NAME,
    options,
  );

const state = async (options: SessionOptions) => {
  async function* noShards(): AsyncGenerator<never> {}
  return await buildSessionState(fakeGpu(), mergeGraph(SILU_GRAPH), new Map(), noShards(), options);
};

/** A JavaScript caller's value outside the type. */
const outsideTheType = (value: unknown): readonly FusionRule[] => value as readonly FusionRule[];

describe("Session construction with fusionRules", () => {
  it("refuses an object written by hand, a copy of a rule and a built-in rule, naming the entry", async () => {
    const literal: FusionRule = {
      name: "test/sigmoid_mul",
      heads: ["sigmoid"],
      contract: "E",
      apply: RULE.apply,
    };
    for (const [label, rules, message] of [
      ["a hand-written rule", [literal], "fusionRules[0] is not a rule built by defineRule"],
      ["a spread copy", [RULE, { ...RULE }], "fusionRules[1] is not a rule built by defineRule"],
      ["a built-in rule", [FUSION_RULES[3]], "fusionRules[0] is a built-in rule"],
    ] as const) {
      await assertRejects(() => construct({ fusionRules: rules }), ExecutionError, message, label);
    }
  });

  it("names both causes of a missing brand: an object defineRule did not make, or a rule of another copy of the runtime", async () => {
    await assertRejects(
      () => construct({ fusionRules: [{ ...RULE }] }),
      ExecutionError,
      "either an object defineRule did not make (a hand-written object or a copy of a rule), or a rule defined by" +
        " another copy of @karume/runtime in the dependency tree",
    );
  });

  it("refuses a value that is not an array, and a rule name given twice", async () => {
    await assertRejects(
      () => construct({ fusionRules: outsideTheType(RULE) }),
      ExecutionError,
      "options.fusionRules must be an array",
    );
    for (const rules of [
      [RULE, RULE],
      [RULE, sigmoidMulRule("test/sigmoid_mul")],
    ]) {
      await assertRejects(
        () => construct({ fusionRules: rules }),
        ExecutionError,
        "the rule 'test/sigmoid_mul' is given twice",
      );
    }
  });

  it("plans with the given rules first, in their order, then every built-in rule", async () => {
    const other = sigmoidMulRule("other/sigmoid_mul");
    const built = await state({ fusionRules: [RULE, other] });
    assertEquals(
      built.fusionRules.map((rule) => rule.name),
      ["test/sigmoid_mul", "other/sigmoid_mul", ...FUSION_RULES.map((rule) => rule.name)],
    );
    assertEquals((await state({})).fusionRules, FUSION_RULES);
  });

  it("takes an external rule's name in DISABLED_FUSION_RULES on a Session that was given the rule, and refuses it on one that was not", async () => {
    const built = await state({
      fusionRules: [RULE],
      [DISABLED_FUSION_RULES]: ["test/sigmoid_mul"],
    });
    assertEquals(built.disabledFusionRules, new Set(["test/sigmoid_mul"]));
    await assertRejects(
      () => construct({ [DISABLED_FUSION_RULES]: ["test/sigmoid_mul"] }),
      ExecutionError,
      "'test/sigmoid_mul'",
    );
  });
});

describe("the Session's rule list in the fusion pass", () => {
  const plan = (rules: readonly FusionRule[]) => {
    const graph = mergeGraph(SILU_GRAPH);
    return planFusions(planGraph(graph, bindSymbols(graph, { x: SHAPE })).nodes, {
      rules,
      useCounts: countUses(graph),
      outputNames: new Set(graph.outputs),
      limits: {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupsPerDimension: 65535,
      },
    });
  };

  it("lets an external rule take a node a built-in rule would take, and counts it under its name", () => {
    const external = plan(sessionFusionRules([RULE]));
    assertEquals(
      external.steps.map((step) => (step.kind === "fused" ? step.rule : step.plan.node.op)),
      ["test/sigmoid_mul"],
    );
    assertEquals([external.counts["test/sigmoid_mul"], external.counts.silu], [1, 0]);
    // The control: without the rule the built-in silu rule takes the same nodes.
    const builtIn = plan(sessionFusionRules(undefined));
    assertEquals([builtIn.counts["test/sigmoid_mul"], builtIn.counts.silu], [undefined, 1]);
  });
});
