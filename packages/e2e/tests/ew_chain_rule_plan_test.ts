/**
 * The rule `models/ew_chain` (`@karume/models/moss`, packages/models/src/moss/ew-chain.ts, which the
 * MOSS-Image pipeline passes by default) as the fusion pass plans it, on small synthetic graphs
 * (no GPU, no model asset — the core lane): planned through the path a Session takes
 * (`planDistributionFusions` with `fusionRules`, `./helpers/fusion-counts.ts`), so what is checked
 * is the plan a Session given the rule would run. It pins the windows the rule declines (a chain
 * that is not f32, an internal value read outside the run or named as a graph output, a node whose
 * output shape is not the last node's, a run of one node, the exact `sigmoid(x) → mul(x, ·)` pair,
 * which the built-in silu rule takes instead), the longest run it chooses (a run whose tail fails
 * falls back to its longest prefix that passes; more than 7 external inputs is such a tail), that
 * one form gives one key and one WGSL text whatever the shapes, and the params words of the
 * broadcast inputs. What the rule computes is checked against the unfused plan on the GPU in
 * `e2e_ew_chain_rule_test.ts`; its hits on the MOSS-Image DiT are pinned in
 * `e2e_moss_fusion_counts_test.ts`.
 */

import { describe, it } from "vite-plus/test";
import { assertEquals, assertNotEquals } from "@std/assert";
import type { ExecStep } from "../../runtime/src/runtime/fusion-rule.ts";
import { mergeGraph } from "../../runtime/tests/helpers/merged-graph.ts";
import type { DeclarationJson } from "../../runtime/tests/helpers/model-fixture.ts";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import { type DistributionFusionPlan, planDistributionFusions } from "./helpers/fusion-counts.ts";

/** A rank-3 activation, and its per-channel broadcast operand. */
const ROWS = 3;
const WIDTH = 8;
const FULL = [1, ROWS, WIDTH];
const CHANNEL = [1, 1, WIDTH];

/** One node: its op, its inputs, its output and the output's shape (and dtype when not f32). */
type NodeSpec = {
  readonly op: string;
  readonly ins: readonly string[];
  readonly out: string;
  readonly shape: readonly number[];
  readonly dtype?: string;
};

type GraphSpec = {
  /** The graph inputs: name → shape (f32 unless `dtypes` says otherwise). */
  readonly inputs: Readonly<Record<string, readonly number[]>>;
  readonly dtypes?: Readonly<Record<string, string>>;
  readonly nodes: readonly NodeSpec[];
  /** The graph outputs (default: the last node's output). */
  readonly outputs?: readonly string[];
};

const node = (
  op: string,
  ins: readonly string[],
  out: string,
  shape: readonly number[] = FULL,
  dtype?: string,
): NodeSpec => ({ op, ins, out, shape, ...(dtype === undefined ? {} : { dtype }) });

const declarationOf = (spec: GraphSpec): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...new Set(spec.nodes.map(({ op }) => op))].sort() },
  symbols: [],
  inputs: Object.entries(spec.inputs).map(([name, shape]) => ({
    name,
    dtype: spec.dtypes?.[name] ?? "f32",
    shape: [...shape],
  })),
  outputs: [...(spec.outputs ?? [spec.nodes[spec.nodes.length - 1].out])],
  initializers: {},
  values: Object.fromEntries(
    spec.nodes.map(({ out, shape, dtype }) => [out, { dtype: dtype ?? "f32", shape: [...shape] }]),
  ),
  nodes: spec.nodes.map(({ op, ins, out }) => ({ op, ins: [...ins], outs: [out], attrs: {} })),
});

/** The plan of `spec` with the rule given, or without it. */
const planOf = (spec: GraphSpec, withRule = true): DistributionFusionPlan =>
  planDistributionFusions(
    mergeGraph(declarationOf(spec)),
    spec.inputs,
    withRule ? { fusionRules: [EW_CHAIN_RULE] } : {},
  );

type FusedStepOf = Extract<ExecStep, { readonly kind: "fused" }>;

/** The rule's steps of a plan. */
const ruleSteps = (planned: DistributionFusionPlan): readonly FusedStepOf[] =>
  planned.plan.steps.flatMap((step) =>
    step.kind === "fused" && step.rule === EW_CHAIN_RULE_NAME ? [step] : [],
  );

/** The rule's one step of a plan (fails when it has none or several). */
const onlyStep = (planned: DistributionFusionPlan, where: string): FusedStepOf => {
  const steps = ruleSteps(planned);
  assertEquals(steps.length, 1, `${where}: the rule's steps`);
  return steps[0];
};

/** The rule declines every window of `spec`: no hit, and the plan of the graph without the rule. */
const assertDeclined = (spec: GraphSpec, where: string): void => {
  const fused = planOf(spec);
  const plain = planOf(spec, false);
  assertEquals(fused.plan.counts[EW_CHAIN_RULE_NAME], 0, `${where}: hits`);
  assertEquals(
    fused.plan.counts,
    { ...plain.plan.counts, [EW_CHAIN_RULE_NAME]: 0 },
    `${where}: the other counters`,
  );
  assertEquals(fused.plan.steps.length, plain.plan.steps.length, `${where}: steps`);
};

/** `mul(x, gate) → add(residual, ·)` — the gated residual's form. */
const GATED: GraphSpec = {
  inputs: { x: FULL, gate: CHANNEL, residual: FULL },
  nodes: [node("mul", ["x", "gate"], "t"), node("add", ["residual", "t"], "y")],
};

describe("models/ew_chain: the windows it declines", () => {
  it("a chain that is not f32", () => {
    assertDeclined(
      {
        inputs: { a: FULL, b: FULL, c: FULL },
        dtypes: { a: "i32", b: "i32", c: "i32" },
        nodes: [
          node("mul", ["a", "b"], "t", FULL, "i32"),
          node("sub", ["t", "c"], "y", FULL, "i32"),
        ],
      },
      "i32",
    );
  });

  it("an internal value read outside the run", () => {
    assertDeclined(
      {
        inputs: { x: FULL, gate: CHANNEL, residual: FULL },
        nodes: [
          node("mul", ["x", "gate"], "t"),
          node("add", ["residual", "t"], "y"),
          // `ge` is not an op of a run, so it ends the run and reads `t` from outside it.
          node("ge", ["t", "residual"], "mask", FULL, "bool"),
        ],
        outputs: ["y", "mask"],
      },
      "read outside",
    );
  });

  it("an internal value that is a graph output", () => {
    assertDeclined({ ...GATED, outputs: ["y", "t"] }, "graph output");
  });

  it("a node whose output shape is not the last node's", () => {
    assertDeclined(
      {
        inputs: { scale: CHANNEL, shift: CHANNEL, x: FULL },
        // `t` is [1, 1, C] and the last node writes [1, R, C].
        nodes: [node("mul", ["scale", "shift"], "t", CHANNEL), node("add", ["t", "x"], "y")],
      },
      "shape",
    );
  });

  it("a run of one node", () => {
    assertDeclined({ inputs: { x: FULL }, nodes: [node("neg", ["x"], "y")] }, "one node");
    assertDeclined(
      {
        inputs: { x: FULL, z: FULL },
        nodes: [node("neg", ["x"], "n"), node("ge", ["n", "z"], "y", FULL, "bool")],
      },
      "one node, then an op that ends the run",
    );
  });

  it("the exact sigmoid(x) → mul(x, ·) pair, which the built-in silu rule takes instead", () => {
    for (const ins of [
      ["x", "s"],
      ["s", "x"],
    ]) {
      const spec: GraphSpec = {
        inputs: { x: FULL },
        nodes: [node("sigmoid", ["x"], "s"), node("mul", ins, "y")],
      };
      assertDeclined(spec, `mul(${ins.join(", ")})`);
      assertEquals(planOf(spec).plan.counts.silu, 1, `mul(${ins.join(", ")}): silu`);
    }
  });

  it("but takes a longer run that holds the pair whole, and silu's counter stays at 0", () => {
    const planned = planOf({
      inputs: { x: FULL, z: FULL },
      nodes: [
        node("sigmoid", ["x"], "s"),
        node("mul", ["x", "s"], "m"),
        node("add", ["m", "z"], "y"),
      ],
    });
    assertEquals(
      [planned.plan.counts[EW_CHAIN_RULE_NAME], planned.plan.counts.silu],
      [1, 0],
      "ew_chain, silu",
    );
    assertEquals(onlyStep(planned, "sigmoid, mul, add").nodeCount, 3);
  });
});

describe("models/ew_chain: the run it takes", () => {
  it("is the longest one: a run whose tail breaks the shape falls back to its longest prefix that passes", () => {
    const planned = planOf({
      inputs: { scale: CHANNEL, shift: CHANNEL, bias: CHANNEL, x: FULL },
      nodes: [
        node("mul", ["scale", "shift"], "a", CHANNEL),
        node("add", ["a", "bias"], "b", CHANNEL),
        // Writes [1, R, C]: the three-node run fails, the two-node prefix (its last value read
        // here, outside it) passes.
        node("add", ["b", "x"], "y"),
      ],
    });
    const step = onlyStep(planned, "prefix");
    assertEquals(
      [step.nodeCount, step.outputName, step.binds],
      [2, "b", ["scale", "shift", "bias"]],
    );
    assertEquals(planned.plan.steps.length, 2, "the fused prefix and the plain tail");
  });

  it("takes at most 7 external inputs: an eighth one ends the run before its node", () => {
    const inputs = Object.fromEntries(
      Array.from({ length: 8 }, (_, k): [string, readonly number[]] => [`x${k}`, FULL]),
    );
    const nodes = Array.from({ length: 7 }, (_, k) =>
      node("add", [k === 0 ? "x0" : `t${k - 1}`, `x${k + 1}`], k === 6 ? "y" : `t${k}`),
    );
    const planned = planOf({ inputs, nodes });
    const step = onlyStep(planned, "eight inputs");
    assertEquals(step.nodeCount, 6, "the six nodes that read seven inputs");
    assertEquals(step.binds, ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]);
    assertEquals(step.outputName, "t5");
    assertEquals(planned.plan.steps.length, 2, "the fused run and the plain seventh add");
  });

  it("folds a value read twice inside the run, and binds a repeated input once", () => {
    const planned = planOf({
      inputs: { x: FULL, z: FULL },
      nodes: [
        node("add", ["x", "z"], "t"),
        node("mul", ["t", "t"], "u"),
        node("sub", ["u", "x"], "y"),
      ],
    });
    const step = onlyStep(planned, "read twice");
    assertEquals([step.nodeCount, step.binds], [3, ["x", "z"]]);
    assertEquals(
      step.dispatches[0].key,
      "ext:models:ew_chain:v1:f.f:add.x0.x1-mul.t0.t0-sub.t1.x0:f32:wg128",
    );
  });
});

describe("models/ew_chain: keys, WGSL and params", () => {
  it("gives one form one key and byte-identical WGSL whatever the shapes, and another form another key", () => {
    const shaped = (rows: number, width: number): GraphSpec => ({
      inputs: { x: [1, rows, width], gate: [1, 1, width], residual: [1, rows, width] },
      nodes: [
        node("mul", ["x", "gate"], "t", [1, rows, width]),
        node("add", ["residual", "t"], "y", [1, rows, width]),
      ],
    });
    const [small, large] = [shaped(3, 8), shaped(4159, 3072)].map(
      (spec) => onlyStep(planOf(spec), "shapes").dispatches[0],
    );
    assertEquals(small.key, "ext:models:ew_chain:v1:f.b001.f:mul.x0.x1-add.x2.t0:f32:wg128");
    assertEquals(large.key, small.key);
    assertEquals(large.wgsl(), small.wgsl());
    assertNotEquals(large.params, small.params);
    // The add's operands swapped: another node list, another key and another text.
    const swapped = onlyStep(
      planOf({
        ...GATED,
        nodes: [node("mul", ["x", "gate"], "t"), node("add", ["t", "residual"], "y")],
      }),
      "swapped",
    ).dispatches[0];
    assertEquals(swapped.key, "ext:models:ew_chain:v1:f.b001.f:mul.x0.x1-add.t0.x2:f32:wg128");
    assertNotEquals(swapped.wgsl(), small.wgsl());
  });

  it("writes the params of a [1, 1, C] input: n, the channel axis's span and length, and its stride", () => {
    const params = (spec: GraphSpec) => [...onlyStep(planOf(spec), "params").dispatches[0].params];
    // Fields n, o2, d2, s1_2: element i reads gate[((i / 1) % C) * 1].
    assertEquals(params(GATED), [ROWS * WIDTH, 1, WIDTH, 1]);
    // A rank-1 operand [C] is right-aligned to [1, 1, C]: the same words.
    assertEquals(params({ ...GATED, inputs: { ...GATED.inputs, gate: [WIDTH] } }), [
      ROWS * WIDTH,
      1,
      WIDTH,
      1,
    ]);
    // Two broadcast inputs on the same axis (adaLN's scale and shift): n, o2, d2, s1_2, s2_2, then
    // three reserved words up to a multiple of four.
    assertEquals(
      params({
        inputs: { x: FULL, scale: CHANNEL, shift: CHANNEL },
        nodes: [node("mul", ["x", "scale"], "t"), node("add", ["t", "shift"], "y")],
      }),
      [ROWS * WIDTH, 1, WIDTH, 1, 1, 0, 0, 0],
    );
  });

  it("writes the params of a [1, S, 1, D] input over [1, S, H, D]: both indexed axes and its strides", () => {
    const [S, H, D] = [5, 3, 4];
    const step = onlyStep(
      planOf({
        inputs: { x: [1, S, H, D], sin: [1, S, 1, D], xc: [1, S, H, D] },
        nodes: [
          node("mul", ["x", "sin"], "t", [1, S, H, D]),
          node("add", ["xc", "t"], "y", [1, S, H, D]),
        ],
      }),
      "rank 4",
    );
    const [dispatch] = step.dispatches;
    assertEquals(dispatch.key, "ext:models:ew_chain:v1:f.b0101.f:mul.x0.x1-add.x2.t0:f32:wg128");
    // Fields n, o1, d1, o3, d3, s1_1, s1_3, reserved0: element i reads
    // sin[((i / (H·D)) % S) · D + ((i / 1) % D) · 1].
    assertEquals([...dispatch.params], [S * H * D, H * D, S, 1, D, D, 1, 0]);
  });
});
