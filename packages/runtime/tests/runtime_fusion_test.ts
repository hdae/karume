/**
 * Pins only the decisions of the fusion pass (src/runtime/fusion.ts), independent of the GPU.
 *
 * Fusion's accepted set is two-tiered — "shapes it matches are fast, shapes it does not match are
 * always correct as the plain sequence" — so **exhaustive counterexamples are themselves the basis
 * of safety**. The real-GPU tests (gpu_*_fusion_test.ts) check bit parity of the values; this file
 * piles up counterexamples in detail without a GPU.
 */

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
  adalnNormParams,
} from "../src/kernels/adaln-norm.ts";
import { bmmKey } from "../src/kernels/bmm.ts";
import { GATED_RESIDUAL_KEY } from "../src/kernels/gated-residual.ts";
import { ROPE_BSHD_KEY, ROPE_KEY, ROPE_PAIR_BSHD_KEY } from "../src/kernels/rope.ts";
import { QUANTIZE_ROWS_PROLOGUE_MAX_DIM } from "../src/kernels/quantize-rows.ts";
import {
  RMS_NORM_ROPE_KEY,
  RMS_NORM_ROPE_WGSL,
  rmsNormRopeParams,
} from "../src/kernels/rms-norm-rope.ts";
import { siluKey } from "../src/kernels/silu.ts";
import { SAFE_SOFTMAX_KEY } from "../src/kernels/softmax.ts";
import { UPSAMPLE_2X_KEY } from "../src/kernels/upsample2x.ts";
import { type AttentionDeviceFacts, attentionRoute } from "../src/runtime/attention-route.ts";
import {
  type AttentionTokenMajor,
  defineRule,
  type ExecStep,
  FUSION_RULES,
  type FusionContext,
  type FusionMatch,
  type FusionPlan,
  type FusionRule,
  fusionMatchContext,
  LINEAR_PROLOGUE_OPS,
  planFusions,
  planRowBlocks,
} from "../src/runtime/fusion.ts";
import {
  bindSymbols,
  countUses,
  ExecutionError,
  type NodePlan,
  planGraph,
} from "../src/runtime/plan.ts";
import { builtInCounts } from "./helpers/fusion-count-record.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import {
  rmsNormRopeGraph,
  type RmsNormRopeGraphOptions,
  rmsNormRopeInputShapes,
} from "./helpers/rms-norm-rope-graph.ts";

const parse = mergeGraph;

/**
 * Device capabilities used for the decision (WebGPU core defaults — 128MiB / 65535). Only the row
 * block split count and the workgroup-count checks of linearStaticQuantize read them (no other rule
 * depends on device capabilities).
 */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/** Graph JSON → fused step sequence (the same input the executor builds per run). */
const fuse = (
  graph: DeclarationJson,
  inputShapes: Readonly<Record<string, readonly number[]>>,
): FusionPlan => {
  const ir = parse(graph);
  const plan = planGraph(ir, bindSymbols(ir, inputShapes));
  return planFusions(plan.nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
  });
};

/** A summary of the step sequence (the rule name, or the op name for plain nodes). */
const outline = (steps: readonly ExecStep[]): readonly string[] =>
  steps.map((step) => (step.kind === "fused" ? `fused:${step.rule}` : step.plan.node.op));

const fusedAt = (plan: FusionPlan, index: number) => {
  const step = plan.steps[index];
  if (step.kind !== "fused") throw new Error(`steps[${index}] is not a fused step`);
  return step;
};

test("the rule table's first ops are disjoint, so the application order does not affect the result", () => {
  assertEquals(
    FUSION_RULES.map((rule) => rule.name),
    [
      "linearStaticQuantize",
      "geluMulStaticQuantize",
      "rmsNormAdd",
      "silu",
      "upsample2x",
      "rope",
      "rmsNormRope",
      "adaln",
      "gatedResidual",
      "rowBlockAttention",
    ],
  );
  const owners = new Map<string, string[]>();
  for (const rule of FUSION_RULES) {
    for (const head of rule.heads) owners.set(head, [...(owners.get(head) ?? []), rule.name]);
  }
  // The one shared head: rmsNormAdd and rmsNormRope never accept the same index ("rmsNormAdd and
  // rmsNormRope share the head rms_norm but never accept the same index" below checks it).
  assertEquals(
    [...owners].filter(([, names]) => names.length > 1),
    [["rms_norm", ["rmsNormAdd", "rmsNormRope"]]],
  );
  const seen = new Set(owners.keys());
  assertEquals([...seen].sort(), [
    "bmm",
    "gelu_tanh",
    "layer_norm",
    "linear",
    "mul",
    "reshape",
    "rms_norm",
    "sigmoid",
    "slice",
    "where",
  ]);
});

test("every built-in rule declares contract class E, the class ADR 0147 decision 2 gives the fusion rules", () => {
  assertEquals(
    FUSION_RULES.filter((rule) => rule.contract !== "E").map((rule) => rule.name),
    [],
  );
});

// ---------------------------------------------------------------- The rule list is an input

type NegNegMatch = FusionMatch & {
  readonly input: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
};

/**
 * A test-local rule: `neg(neg(x))` folded into one dispatch (head `neg`, which no built-in rule
 * has). The plans are never executed, so the dispatch only has to be well formed. Its matcher does
 * not check the privacy of `n1` itself: only its declaration does.
 */
const NEG_NEG_RULE = defineRule<NegNegMatch>({
  name: "testNegNeg",
  heads: ["neg"],
  contract: "E",
  internals: "private",
  match: (nodes, index) => {
    const first = nodes[index];
    const second = nodes[index + 1];
    if (first?.node.op !== "neg" || second?.node.op !== "neg") return undefined;
    if (second.node.ins[0] !== first.outputs[0].name) return undefined;
    const chain = [first, second];
    return {
      window: chain,
      chain,
      input: first.node.ins[0],
      outputName: second.outputs[0].name,
      outputShape: second.outputs[0].shape,
    };
  },
  build: (matched) => ({
    binds: [matched.input],
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    temps: [],
    dispatches: [
      {
        key: "test:neg-neg",
        wgsl: () => "",
        params: new Uint32Array(4),
        workgroups: {
          kind: "gridStride",
          items: matched.outputShape.reduce((count, dim) => count * dim, 1),
          size: 64,
        },
      },
    ],
  }),
});

type RuleListOptions = {
  /** `n1` (the test-local rule's internal value) has a consumer outside the chain. */
  readonly internalConsumer?: boolean;
  /** `n1` is a graph output. */
  readonly internalOutput?: boolean;
};

/** `y = silu(x)` (a built-in rule's form) beside `n2 = neg(neg(z))` (the test-local rule's form). */
const ruleListGraph = (options: RuleListOptions = {}): DeclarationJson => {
  const shape = [4, 8];
  const nodes: DeclarationJson["nodes"] = [
    { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["x", "s"], outs: ["y"], attrs: {} },
    { op: "neg", ins: ["z"], outs: ["n1"], attrs: {} },
    { op: "neg", ins: ["n1"], outs: ["n2"], attrs: {} },
  ];
  const outputs = ["y", "n2"];
  if (options.internalConsumer) {
    nodes.push({ op: "abs", ins: ["n1"], outs: ["n3"], attrs: {} });
    outputs.push("n3");
  }
  if (options.internalOutput) outputs.push("n1");
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [...shape] },
      { name: "z", dtype: "f32", shape: [...shape] },
    ],
    outputs,
    initializers: {},
    values: Object.fromEntries(
      nodes.flatMap((node) => node.outs).map((name) => [name, { dtype: "f32", shape: [...shape] }]),
    ),
    nodes,
  };
};

/** Plans `graph` with the rule list `rules` (and the other facts as {@link fuse} passes them). */
const fuseWithRules = <Name extends string>(
  graph: DeclarationJson,
  rules: readonly FusionRule<Name>[],
  extra: Omit<Partial<FusionContext>, "rules"> = {},
): FusionPlan<Name> => {
  const ir = parse(graph);
  return planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...extra,
    rules,
  });
};

test("the rule list is an input: an extra rule built with defineRule folds and counts under its own name, and the built-in counters stay as they were", () => {
  const graph = ruleListGraph();
  const builtIn = fuseWithRules(graph, FUSION_RULES);
  assertEquals(outline(builtIn.steps), ["fused:silu", "neg", "neg"]);
  const extended = fuseWithRules(graph, [...FUSION_RULES, NEG_NEG_RULE]);
  assertEquals(outline(extended.steps), ["fused:silu", "fused:testNegNeg"]);
  assertEquals(extended.counts, { ...builtIn.counts, testNegNeg: 1 });
});

test("with an empty rule list nothing folds through a rule, while the linear prologue and the attention permutes still apply", () => {
  const annotationsOnly = {
    identityExpand: 0,
    packedStaticQuantize: 0,
    linearPrologue: 0,
    attentionPermute: 0,
  };
  const ruleForms = fuseWithRules(ruleListGraph(), []);
  assertEquals(outline(ruleForms.steps), ["sigmoid", "mul", "neg", "neg"]);
  assertEquals(ruleForms.counts, annotationsOnly);
  const prologue = fuseWithRules(prologuePairGraph(), [], A8_I8);
  assertEquals(outline(prologue.steps), ["linear"]);
  assertEquals(prologue.counts, { ...annotationsOnly, linearPrologue: 1 });
  const permutes = fuseWithRules(attentionPermuteGraph(), [], A8_SEAT);
  assertEquals(outline(permutes.steps), ["attention", "permute"]);
  assertEquals(permutes.counts, { ...annotationsOnly, attentionPermute: 3 });
});

test("defineRule applies the declared privacy: a rule declared private folds only when its internal value has no consumer outside the chain and is not a graph output", () => {
  const rules = [NEG_NEG_RULE];
  const control = fuseWithRules(ruleListGraph(), rules);
  assertEquals(outline(control.steps), ["sigmoid", "mul", "fused:testNegNeg"], "control");
  for (const [label, options] of [
    ["a consumer outside the chain", { internalConsumer: true }],
    ["a graph output", { internalOutput: true }],
  ] as const) {
    const plan = fuseWithRules(ruleListGraph(options), rules);
    assertEquals(plan.counts.testNegNeg, 0, label);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("a rule list whose names do not give one counter each fails loudly (a name twice, a rule named like an annotation counter)", () => {
  const graph = ruleListGraph();
  for (const [label, rules, shared] of [
    ["the same rule twice", [NEG_NEG_RULE, NEG_NEG_RULE], "testNegNeg"],
    [
      "a rule named like a counter",
      [{ ...NEG_NEG_RULE, name: "linearPrologue" }],
      "linearPrologue",
    ],
  ] as const) {
    assertThrows(() => fuseWithRules(graph, rules), ExecutionError, `'${shared}'`, label);
  }
});

test("a hit whose step carries a rule name without a counter fails loudly instead of counting NaN", () => {
  // A rule built outside defineRule can return a step under any name.
  const renaming: FusionRule = {
    name: "testRenaming",
    heads: ["neg"],
    contract: "E",
    apply: (nodes, index, context) => {
      const hit = NEG_NEG_RULE.apply(nodes, index, context);
      return hit === undefined ? undefined : { ...hit, step: { ...hit.step, rule: "elsewhere" } };
    },
  };
  assertThrows(
    () => fuseWithRules(ruleListGraph(), [renaming]),
    ExecutionError,
    "fusion rule 'elsewhere'",
  );
});

// ---------------------------------------------------------------- Disabled rules (test only)

test("disabling 'all' rules leaves no fused step and a complete counter record of zeros, while the annotation folds still apply", () => {
  const plan = fuseWithRules(ruleListGraph(), FUSION_RULES, { disabledRules: "all" });
  assertEquals(outline(plan.steps), ["sigmoid", "mul", "neg", "neg"]);
  assertEquals(plan.counts, builtInCounts());
  const prologue = fuseWithRules(prologuePairGraph(), FUSION_RULES, {
    ...A8_I8,
    disabledRules: "all",
  });
  assertEquals(outline(prologue.steps), ["linear"]);
  assertEquals(prologue.counts, builtInCounts({ linearPrologue: 1 }));
});

test("a disabled rule name leaves only that rule out: another rule of the same graph still folds, and an empty set disables nothing", () => {
  const graph = ruleListGraph();
  const rules = [...FUSION_RULES, NEG_NEG_RULE];
  const withoutSilu = fuseWithRules(graph, rules, { disabledRules: new Set(["silu"]) });
  assertEquals(outline(withoutSilu.steps), ["sigmoid", "mul", "fused:testNegNeg"]);
  assertEquals(withoutSilu.counts, { ...builtInCounts(), testNegNeg: 1 });
  const withoutNegNeg = fuseWithRules(graph, rules, { disabledRules: new Set(["testNegNeg"]) });
  assertEquals(outline(withoutNegNeg.steps), ["fused:silu", "neg", "neg"]);
  assertEquals(withoutNegNeg.counts, { ...builtInCounts({ silu: 1 }), testNegNeg: 0 });
  const none = fuseWithRules(graph, rules, { disabledRules: new Set() });
  assertEquals(outline(none.steps), ["fused:silu", "fused:testNegNeg"]);
});

test("a disabled name that is not a name of the rule list fails loudly instead of leaving its rule on", () => {
  assertThrows(
    () =>
      fuseWithRules(ruleListGraph(), FUSION_RULES, { disabledRules: new Set(["silu", "Silu"]) }),
    ExecutionError,
    "'Silu'",
  );
});

// ---------------------------------------------------------------- SiLU

type SiluOptions = {
  readonly order?: "x-sigmoid" | "sigmoid-x";
  readonly interpose?: boolean;
  readonly sigmoidOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly gateWithOther?: boolean;
  readonly broadcast?: boolean;
};

const siluGraph = (options: SiluOptions = {}): DeclarationJson => {
  const shape = [4, 8];
  const values: DeclarationJson["values"] = {
    s: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: [...shape] }];
  const nodes: DeclarationJson["nodes"] = [{ op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} }];
  let mulSigmoid = "s";
  if (options.interpose) {
    values.s_alias = { dtype: "f32", shape: [...shape] };
    nodes.push({ op: "reshape", ins: ["s"], outs: ["s_alias"], attrs: {} });
    mulSigmoid = "s_alias";
  }
  // broadcast is "an axis of length 1 on one side only": legal for mul but not SiLU.
  if (options.broadcast) inputs[0] = { name: "x", dtype: "f32", shape: [1, 8] };
  if (options.broadcast) values.s = { dtype: "f32", shape: [1, 8] };
  if (options.gateWithOther) inputs.push({ name: "z", dtype: "f32", shape: [...shape] });
  const mulIns = options.gateWithOther
    ? [mulSigmoid, "z"]
    : options.order === "sigmoid-x"
      ? [mulSigmoid, "x"]
      : options.broadcast
        ? ["x2", mulSigmoid]
        : ["x", mulSigmoid];
  if (options.broadcast) inputs.push({ name: "x2", dtype: "f32", shape: [...shape] });
  nodes.push({ op: "mul", ins: mulIns, outs: ["y"], attrs: {} });
  if (options.extraConsumer) {
    values.s_copy = { dtype: "f32", shape: [...(options.broadcast ? [1, 8] : shape)] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [
      ...(options.sigmoidOutput ? ["s"] : []),
      "y",
      ...(options.extraConsumer ? ["s_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

const siluInputs = (options: SiluOptions = {}): Readonly<Record<string, readonly number[]>> => ({
  x: options.broadcast ? [1, 8] : [4, 8],
  ...(options.gateWithOther ? { z: [4, 8] } : {}),
  ...(options.broadcast ? { x2: [4, 8] } : {}),
});

test("SiLU matches both orders of sigmoid→mul and becomes a step that consumes x twice in total", () => {
  for (const order of ["x-sigmoid", "sigmoid-x"] as const) {
    const plan = fuse(siluGraph({ order }), siluInputs());
    assertEquals(outline(plan.steps), ["fused:silu"], order);
    const step = fusedAt(plan, 0);
    assertEquals(step.ins, ["x", "x"], `${order}: external input sequence`);
    assertEquals(step.binds, ["x"], `${order}: bind order`);
    assertEquals(step.nodeCount, 2, order);
    assertEquals(step.outputName, "y", order);
    assertEquals(step.dispatches[0].key, siluKey(order), order);
    assertEquals(plan.counts.silu, 1, order);
  }
});

test("SiLU counterexamples (alias / internal output / another consumer / another gate / broadcast) fall back to the plain sequence", () => {
  const cases: readonly (readonly [string, SiluOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { sigmoidOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["mul(s,z)", { gateWithOther: true }],
    ["broadcast mul", { broadcast: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(siluGraph(options), siluInputs(options));
    assertEquals(plan.counts.silu, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

/**
 * A window containing a node that touches state is matched by **no rule** (ADR 0067 decision 5b).
 *
 * State references draw no tensor data edges, so if fusion moved an in-window passthrough forward
 * and broke the `nodes` array order, neither the shape checks nor reference counting would fail
 * ("reading this step's k/v twice as the past" would happen without an exception).
 *
 * NOTE: in the current op vocabulary only `attention` / `state_append` can have a `states` field,
 * and neither appears in the window spelling of the existing 5 rules — that is, **it cannot be
 * reached from a real graph**. It is still a check kept as an invariant of the window mechanism, so
 * a synthetic node with a `states` field inserted on one side of a window known to match (SiLU)
 * pins that the decision point actually takes effect.
 */
test("no rule matches a window containing a node that touches state (an invariant of the window mechanism)", () => {
  const ir = parse(siluGraph());
  const plan = planGraph(ir, bindSymbols(ir, siluInputs()));
  const context = {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
  };
  // Control: left plain it matches (the flip side showing the 0 below is not "a window that never
  // matches").
  assertEquals(planFusions(plan.nodes, context).counts.silu, 1);

  for (const tainted of [0, 1]) {
    const nodes = plan.nodes.map((node, index) =>
      index === tainted ? { ...node, node: { ...node.node, states: { k: "kv" } } } : node,
    );
    const fused = planFusions(nodes, context);
    assertEquals(fused.counts.silu, 0, `matched although nodes[${tainted}] has a states field`);
    assertEquals(
      fused.steps.every((step) => step.kind === "node"),
      true,
      `nodes[${tainted}]: ${outline(fused.steps).join(",")}`,
    );
  }
});

// ---------------------------------------------------------- upsample2x

type UpsampleOptions = {
  readonly scaleWidth?: number;
  readonly interpose?: boolean;
  readonly internalOutput?: boolean;
  readonly extraConsumer?: boolean;
  readonly dtype?: "f32" | "i32";
};

const upsampleGraph = (options: UpsampleOptions = {}): DeclarationJson => {
  const dtype = options.dtype ?? "f32";
  const scale = options.scaleWidth ?? 2;
  const outW = 7 * scale;
  const values: DeclarationJson["values"] = {
    a: { dtype, shape: [30, 7, 1] },
    wide: { dtype, shape: [30, 7, scale] },
    b: { dtype, shape: [6, 5, outW] },
    c: { dtype, shape: [6, 5, 1, outW] },
    tall: { dtype, shape: [6, 5, 2, outW] },
    y: { dtype, shape: [2, 3, 10, outW] },
  };
  const nodes: DeclarationJson["nodes"] = [{ op: "reshape", ins: ["x"], outs: ["a"], attrs: {} }];
  let expandSource = "a";
  if (options.interpose) {
    values.a_alias = { dtype, shape: [30, 7, 1] };
    nodes.push({ op: "reshape", ins: ["a"], outs: ["a_alias"], attrs: {} });
    expandSource = "a_alias";
  }
  nodes.push(
    { op: "expand", ins: [expandSource], outs: ["wide"], attrs: {} },
    { op: "reshape", ins: ["wide"], outs: ["b"], attrs: {} },
    { op: "reshape", ins: ["b"], outs: ["c"], attrs: {} },
    { op: "expand", ins: ["c"], outs: ["tall"], attrs: {} },
    { op: "reshape", ins: ["tall"], outs: ["y"], attrs: {} },
  );
  if (options.extraConsumer) {
    values.wide_copy = { dtype, shape: [30, 7, scale] };
    nodes.push({ op: "neg", ins: ["wide"], outs: ["wide_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [{ name: "x", dtype, shape: [2, 3, 5, 7] }],
    outputs: [
      ...(options.internalOutput ? ["wide"] : []),
      "y",
      ...(options.extraConsumer ? ["wide_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

test("VAE nearest x2 matches the 6-node chain and consumes only one external input", () => {
  const plan = fuse(upsampleGraph(), { x: [2, 3, 5, 7] });
  assertEquals(outline(plan.steps), ["fused:upsample2x"]);
  const step = fusedAt(plan, 0);
  assertEquals(step.ins, ["x"]);
  assertEquals(step.binds, ["x"]);
  assertEquals(step.nodeCount, 6);
  assertEquals(step.dispatches[0].key, UPSAMPLE_2X_KEY);
  // params are [input element count, width, output width, reserved]
  assertEquals([...step.dispatches[0].params], [210, 7, 14, 0]);
  assertEquals(plan.counts.upsample2x, 1);
});

test("upsample x2 counterexamples (near-shape / alias / internal output / another consumer / dtype) do not match", () => {
  const cases: readonly (readonly [string, UpsampleOptions])[] = [
    ["x3 near-shape", { scaleWidth: 3 }],
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["i32 dtype", { dtype: "i32" }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(upsampleGraph(options), { x: [2, 3, 5, 7] });
    assertEquals(plan.counts.upsample2x, 0, `${label}: fusion counter`);
  }
});

// ---------------------------------------------------------------- RoPE

type RopeOptions = {
  readonly layout?: "bhsd" | "bshd";
  readonly order?: "slice-first" | "direct-first";
  readonly heads?: number;
  readonly interpose?: boolean;
  readonly internalOutput?: boolean;
  readonly extraConsumer?: boolean;
  /**
   * Moves the split point (cut at 1/4 of the head width — a counterexample "with the same formula
   * but not a half split").
   */
  readonly nearSplit?: boolean;
  /** Head width (observed: 128 (Anima) and 256 (the Gemma series). Default 128). */
  readonly headDim?: number;
  /** Sets batch to 2 (the accepted shape assumes B=1). */
  readonly batch2?: boolean;
  /**
   * Swaps add's input order (a counterexample with the same values and a different wiring pattern).
   */
  readonly swappedAdd?: boolean;
  /** Makes the cos / sin tables the same shape as x (a counterexample that is not broadcast). */
  readonly nearTable?: boolean;
  /**
   * Builds the sin table with `sym_prefix_slice` from a Tmax constant and places it between `cat`
   * and the cross mul (the observed shape — the table is built only once per θ family, so only its
   * first occurrence becomes an in-window passthrough).
   */
  readonly prefixSlicedSin?: boolean;
  /**
   * A counterexample that puts something other than `sym_prefix_slice` (reshape) in the same gap.
   */
  readonly gapReshape?: boolean;
};

const ropeGraph = (options: RopeOptions = {}): DeclarationJson => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const split = headDim / (options.nearSplit ? 4 : 2);
  const full = options.layout === "bshd" ? [batch, 5, heads, headDim] : [batch, heads, 5, headDim];
  const low = options.layout === "bshd" ? [batch, 5, heads, split] : [batch, heads, 5, split];
  const high =
    options.layout === "bshd"
      ? [batch, 5, heads, headDim - split]
      : [batch, heads, 5, headDim - split];
  const table = options.nearTable
    ? full
    : options.layout === "bshd"
      ? [1, 5, 1, headDim]
      : [1, 1, 5, headDim];
  const values: DeclarationJson["values"] = {
    first: { dtype: "f32", shape: low },
    second: { dtype: "f32", shape: high },
    negative: { dtype: "f32", shape: high },
    rotated: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const direct: DeclarationJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: DeclarationJson["nodes"] = [];
  if (options.order === "direct-first") nodes.push(direct);
  nodes.push({ op: "slice", ins: ["x"], outs: ["first"], attrs: { dim: 3, start: 0, end: split } });
  let catFirst = "first";
  if (options.interpose) {
    values.first_alias = { dtype: "f32", shape: low };
    nodes.push({ op: "reshape", ins: ["first"], outs: ["first_alias"], attrs: {} });
    catFirst = "first_alias";
  }
  nodes.push(
    { op: "slice", ins: ["x"], outs: ["second"], attrs: { dim: 3, start: split, end: headDim } },
    { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
    { op: "cat", ins: ["negative", catFirst], outs: ["rotated"], attrs: { dim: 3 } },
  );
  // The in-window passthrough sits right after `cat` (the observed shape). The two options separate
  // "sym_prefix_slice still matches" from "any other op breaks the window so it no longer matches".
  if (options.prefixSlicedSin) {
    // Shrinks a constant of the Tmax shape (S=8) to the runtime T=5. sym_prefix_slice's contract is
    // that the declared shape contains no symbol, so the table side is given a static shape.
    values.sin_table = {
      dtype: "f32",
      shape: options.layout === "bshd" ? [1, 8, 1, headDim] : [1, 1, 8, headDim],
    };
    values.sin = {
      dtype: "f32",
      shape: options.layout === "bshd" ? [1, "T", 1, headDim] : [1, 1, "T", headDim],
    };
    nodes.push({
      op: "sym_prefix_slice",
      ins: ["sin_table"],
      outs: ["sin"],
      attrs: {
        sym: "T",
        slices: [{ dim: options.layout === "bshd" ? 1 : 2, coeff: 1, offset: 0 }],
      },
    });
  }
  if (options.gapReshape) {
    values.cos_view = { dtype: "f32", shape: [table[0], table[1] * table[2], table[3]] };
    nodes.push({ op: "reshape", ins: ["cos"], outs: ["cos_view"], attrs: {} });
  }
  if (options.order !== "direct-first") nodes.push(direct);
  nodes.push(
    { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["cross", "direct"] : ["direct", "cross"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.extraConsumer) {
    values.first_copy = { dtype: "f32", shape: low };
    nodes.push({ op: "neg", ins: ["first"], outs: ["first_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    // The symbol is needed only in the form with sym_prefix_slice (T is bound from the auxiliary
    // input `bind`).
    symbols: options.prefixSlicedSin ? ["T"] : [],
    inputs: [
      { name: "x", dtype: "f32", shape: full },
      { name: "cos", dtype: "f32", shape: table },
      ...(options.prefixSlicedSin
        ? [{ name: "bind", dtype: "f32", shape: ["T"] }]
        : [{ name: "sin", dtype: "f32", shape: table }]),
    ],
    outputs: [
      ...(options.internalOutput ? ["first"] : []),
      "y",
      ...(options.gapReshape ? ["cos_view"] : []),
      ...(options.extraConsumer ? ["first_copy"] : []),
    ],
    initializers: options.prefixSlicedSin ? { sin_table: {} } : {},
    values,
    nodes,
  };
};

const ropeInputs = (options: RopeOptions = {}): Readonly<Record<string, readonly number[]>> => {
  const heads = options.heads ?? 1;
  const batch = options.batch2 ? 2 : 1;
  const headDim = options.headDim ?? 128;
  const full = options.layout === "bshd" ? [batch, 5, heads, headDim] : [batch, heads, 5, headDim];
  const table = options.nearTable
    ? full
    : options.layout === "bshd"
      ? [1, 5, 1, headDim]
      : [1, 1, 5, headDim];
  return options.prefixSlicedSin
    ? { x: full, cos: table, bind: [5] }
    : { x: full, cos: table, sin: table };
};

test("half-split RoPE matches both orders and consumes x 3 times in total and cos / sin once each", () => {
  for (const [order, expectedIns] of [
    ["slice-first", ["x", "x", "x", "cos", "sin"]],
    ["direct-first", ["x", "cos", "x", "x", "sin"]],
  ] as const) {
    const plan = fuse(ropeGraph({ order, heads: 4 }), ropeInputs({ heads: 4 }));
    assertEquals(outline(plan.steps), ["fused:rope"], order);
    const step = fusedAt(plan, 0);
    // MUST: the sequence follows **the actual node order** (in direct-first, cos comes second).
    assertEquals(step.ins, expectedIns, `${order}: external input sequence`);
    assertEquals(step.binds, ["x", "cos", "sin"], `${order}: bind order`);
    assertEquals(step.nodeCount, 7, order);
    assertEquals(step.dispatches[0].key, ROPE_KEY, order);
    // params are [total element count, sequence, head_dim, half_dim]
    assertEquals([...step.dispatches[0].params], [1 * 4 * 5 * 128, 5, 128, 64], order);
    assertEquals(plan.counts.rope, 1, order);
  }
});

// The split point is derived from the head width (the kernel takes head_dim / half_dim as uniforms,
// so the WGSL does not change by a single byte when the width changes). Observed widths are Anima's
// 128 and the Gemma series' 256, but a separate width of 64 pins that "matchable widths" have not
// regressed to hard-coding the two observed points.
test("half-split RoPE derives the head width from the split point, and only params change per width", () => {
  for (const headDim of [64, 128, 256]) {
    const options: RopeOptions = { order: "direct-first", heads: 4, headDim };
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(outline(plan.steps), ["fused:rope"], `head width ${headDim}`);
    const step = fusedAt(plan, 0);
    assertEquals(step.dispatches[0].key, ROPE_KEY, `head width ${headDim}: one kernel`);
    assertEquals(
      [...step.dispatches[0].params],
      [4 * 5 * headDim, 5, headDim, headDim / 2],
      `head width ${headDim}: params`,
    );
  }
});

// The observed shape: the cos / sin tables are sliced from a Tmax constant only once per θ family,
// so only their first occurrence falls into the chain's gap (between cat and the cross mul). Unless
// it can be moved **before** the fused step as an in-window passthrough, fusion silently drops out
// at that one place.
test("RoPE crosses a sym_prefix_slice right after cat as an in-window passthrough", () => {
  const options: RopeOptions = { order: "direct-first", heads: 4, prefixSlicedSin: true };
  const plan = fuse(ropeGraph(options), ropeInputs(options));
  // MUST: the passthrough comes first (the fused step takes its output `sin` as input).
  assertEquals(outline(plan.steps), ["sym_prefix_slice", "fused:rope"]);
  const step = fusedAt(plan, 1);
  assertEquals(step.ins, ["x", "cos", "x", "x", "sin"], "external input sequence");
  assertEquals(step.binds, ["x", "cos", "sin"], "bind order");
  // 7 nodes were folded (distinct from the scan width 8 — the passthrough stays a plain node).
  assertEquals(step.nodeCount, 7);
  assertEquals(plan.counts.rope, 1);
});

test("RoPE counterexamples (alias / internal output / another consumer / split point / batch / add order / table shape / another op in the gap) do not match", () => {
  const cases: readonly (readonly [string, RopeOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["32/96 split", { nearSplit: true }],
    ["batch 2", { batch2: true }],
    ["add(cross, direct)", { swappedAdd: true }],
    // A form that passes a separate table per head (the kernel indexes token*head_dim+d, so it
    // would silently give wrong values)
    ["per-head table", { nearTable: true, heads: 4 }],
    // Only sym_prefix_slice may go in the gap (not widened to "cross any single op")
    ["gap reshape", { gapReshape: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(plan.counts.rope, 0, `${label}: fusion counter`);
  }
});

// -------------------------------------------------------- identity expand

const expandGraph = (outShape: readonly number[]): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg", "expand", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [3, 1] }],
  outputs: ["y", "r"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [3, 1] },
    y: { dtype: "f32", shape: [...outShape] },
    r: { dtype: "f32", shape: [1, 3] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "expand", ins: ["h"], outs: ["y"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["r"], attrs: {} },
  ],
});

test("only an identity expand becomes a 0-dispatch alias; with a replicated axis it falls back to a materializing copy", () => {
  const identity = fuse(expandGraph([3, 1]), { x: [3, 1] });
  assertEquals(
    identity.steps.map((step) => step.kind === "node" && step.aliasesInput),
    [
      false, // neg
      true, // identity expand
      true, // reshape is always an alias
    ],
  );
  assertEquals(identity.counts.identityExpand, 1);

  const replicated = fuse(expandGraph([3, 4]), { x: [3, 1] });
  assertEquals(
    replicated.steps.map((step) => step.kind === "node" && step.aliasesInput),
    [
      false,
      false, // a replicated axis, even one, prevents aliasing
      true,
    ],
  );
  assertEquals(replicated.counts.identityExpand, 0);
});

// --------------------------------------------------------------- adaLN

const ROWS = 4;
const DIM = 8;

type AdalnOptions = {
  /**
   * The number of modulation vectors built with `reshape` (= the number of in-window passthroughs).
   * Observed: 3 (shift / scale / gate), and 2 for the last layer only (no gate). Those removed
   * become direct graph inputs.
   */
  readonly gaps?: number;
  /** Adds one more passthrough that consumes the layer_norm output (reordering becomes illegal). */
  readonly gapConsumesNorm?: boolean;
  /** Makes the layer_norm output a graph output. */
  readonly normOutput?: boolean;
  /** Adds another consumer of the layer_norm output. */
  readonly normExtraConsumer?: boolean;
  /** Adds another consumer of the result of `1 + scale`. */
  readonly modulationExtraConsumer?: boolean;
  /** Swaps mul's input order. */
  readonly swappedMul?: boolean;
  /** Swaps the input order of the add that adds shift. */
  readonly swappedAdd?: boolean;
  /** Makes the `1` constant `[dim]` instead of `[1]` (the kernel reads only `one[0]`). */
  readonly wideOne?: boolean;
  /** Makes the modulation per row instead of broadcast (near-shape). */
  readonly perRowScale?: boolean;
  /**
   * Passes the same value name to layer_norm's weight and bias (a form with duplicate bind
   * entries).
   */
  readonly sharedAffine?: boolean;
};

/**
 * The adaLN chain of a real IR (anima DiT). Builds the **non-adjacent** form as is, with the
 * modulation vectors' `reshape` between `layer_norm` and `mul`.
 */
const adalnGraph = (options: AdalnOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const gaps = options.gaps ?? 3;
  const row = [1, ROWS, DIM];
  const modulation = options.perRowScale ? row : [1, 1, DIM];
  const biasName = options.sharedAffine ? "ln_weight" : "ln_bias";
  const values: DeclarationJson["values"] = {
    t: { dtype, shape: [...row] },
    s: { dtype, shape: [...modulation] },
    p: { dtype, shape: [...row] },
    y: { dtype, shape: [...row] },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype, shape: [...row] },
    { name: "ln_weight", dtype, shape: [DIM] },
    ...(options.sharedAffine ? [] : [{ name: "ln_bias", dtype, shape: [DIM] }]),
    { name: "one", dtype, shape: options.wideOne ? [DIM] : [1] },
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", biasName],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
  ];
  const extraOutputs: string[] = [];

  /**
   * One modulation vector. With `viaReshape` it becomes an in-window passthrough, otherwise a
   * direct graph input.
   */
  const declare = (name: string, shape: readonly number[], viaReshape: boolean): void => {
    if (!viaReshape) {
      inputs.push({ name, dtype, shape: [...shape] });
      return;
    }
    inputs.push({ name: `${name}_src`, dtype, shape: [shape[1], shape[2]] });
    values[name] = { dtype, shape: [...shape] };
    nodes.push({ op: "reshape", ins: [`${name}_src`], outs: [name], attrs: {} });
  };

  if (options.gapConsumesNorm) {
    // A form where the in-window passthrough reads a chain value. It cannot be moved before the
    // fused step (the layer_norm output also gets 2 consumers, breaking the private condition on
    // internal values).
    values.norm_alias = { dtype, shape: [...row] };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["norm_alias"], attrs: {} });
    extraOutputs.push("norm_alias");
  }
  // The observed order (shift → scale → gate). gate is consumed outside the chain.
  declare("shift", [1, 1, DIM], gaps >= 1);
  declare("scale", modulation, gaps >= 2);
  for (const [at, name] of ["gate", "spare"].entries()) {
    if (gaps < 3 + at) continue;
    declare(name, [1, 1, DIM], true);
    extraOutputs.push(name);
  }

  nodes.push(
    { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: options.swappedMul ? ["s", "t"] : ["t", "s"], outs: ["p"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["shift", "p"] : ["p", "shift"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.normExtraConsumer) {
    values.t_copy = { dtype, shape: [...row] };
    nodes.push({ op: "neg", ins: ["t"], outs: ["t_copy"], attrs: {} });
    extraOutputs.push("t_copy");
  }
  if (options.modulationExtraConsumer) {
    values.s_copy = { dtype, shape: [...modulation] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
    extraOutputs.push("s_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.normOutput ? ["t"] : []), "y", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

/** The declared shape is the real shape (no symbolic dimensions). */
const adalnInputs = (graph: DeclarationJson): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(graph.inputs.map((spec) => [spec.name, spec.shape as readonly number[]]));

const fuseAdaln = (options: AdalnOptions = {}): FusionPlan => {
  const graph = adalnGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("adaLN matches windows of 7 / 6, and the reshapes stay plain before the fused step", () => {
  for (const gaps of [3, 2] as const) {
    const label = `window ${gaps + 4}`;
    const plan = fuseAdaln({ gaps });
    assertEquals(outline(plan.steps), [...Array(gaps).fill("reshape"), "fused:adaln"], label);
    const step = fusedAt(plan, gaps);
    // The external input sequence comes from **only the 4 folded nodes** (the passthroughs are
    // counted by the plain nodes).
    const operands = ["x", "ln_weight", "ln_bias", "scale", "one", "shift"];
    assertEquals(step.ins, operands, `${label}: external input sequence`);
    assertEquals(step.binds, operands, `${label}: bind order (the kernel's bindings 1–6)`);
    assertEquals(step.nodeCount, 4, `${label}: the folded count is not the window width`);
    assertEquals(step.outputName, "y", label);
    assertEquals(step.outputShape, [1, ROWS, DIM], label);
    assertEquals(step.dispatches[0].key, ADALN_NORM_KEY, label);
    // params are identical to a plain layer_norm (the bit patterns of rows / dim / eps).
    assertEquals([...step.dispatches[0].params], [...adalnNormParams(ROWS, DIM, 1e-6)], label);
    assertEquals(
      step.dispatches[0].workgroups,
      { kind: "gridStride", items: ROWS, size: 1 },
      `${label}: 1 row = 1 workgroup, the row count is the workgroup count`,
    );
    assertEquals(plan.counts.adaln, 1, label);
  }
});

test("adaLN's in-window passthroughs do not take away other rules' match positions", () => {
  const graph = adalnGraph();
  const ir = parse(graph);
  const plan = planGraph(ir, bindSymbols(ir, adalnInputs(graph)));
  const context = {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    packedValues: new Map<string, number>(),
    matchContext: { useCounts: countUses(ir), outputNames: new Set(ir.outputs) },
  };
  // Checks that nothing but adaln matches at **every start position** of the window (layer_norm +
  // reshape×3 + add + mul + add). `reshape` is also upsample2x's first op, so only here can it be
  // seen whether skipping the whole window takes chances away from other rules.
  for (let index = 0; index < 7; index += 1) {
    for (const rule of FUSION_RULES) {
      if (rule.name === "adaln") continue;
      assertEquals(
        rule.apply(plan.nodes, index, context),
        undefined,
        `at position ${index} of the window, '${rule.name}' matches`,
      );
    }
  }
});

// NOTE: a dtype counterexample **cannot be constructed** — layer_norm's contract is f32 only and
// the following add / mul require a uniform dtype, so planGraph fails before the fusion pass.
// `allF32` stays as an eligibility condition shared by every rule (it starts to matter if the
// layers change).
test("adaLN counterexamples (gap count / gap consumes norm / internal output / another consumer / input order / constant shape / near-shape / duplicate bind) do not match", () => {
  const cases: readonly (readonly [string, AdalnOptions])[] = [
    ["no gap (adjacent chain)", { gaps: 0 }],
    ["1 gap", { gaps: 1 }],
    ["4 gaps", { gaps: 4 }],
    ["gap consumes the layer_norm output", { gapConsumesNorm: true }],
    ["layer_norm output is a graph output", { normOutput: true }],
    ["another consumer of the layer_norm output", { normExtraConsumer: true }],
    ["another consumer of 1 + scale", { modulationExtraConsumer: true }],
    ["mul(s, t)", { swappedMul: true }],
    ["add(shift, p)", { swappedAdd: true }],
    ["constant is [dim]", { wideOne: true }],
    ["per-row scale (not broadcast)", { perRowScale: true }],
    ["weight and bias share a value name", { sharedAffine: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseAdaln(options);
    assertEquals(plan.counts.adaln, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

type AdalnTokenOptions = {
  /** The shift `where` reads a different mask from the scale `where`. */
  readonly separateMasks?: boolean;
  /** The mask is `[1,1,1]` (one flag for every row — not one per token). */
  readonly broadcastMask?: boolean;
  /** The mask is `[1,rows,dim]` (per element — not one per token). */
  readonly fullMask?: boolean;
  /** The condition-side scale is `[1,rows,dim]` (per row — not a broadcast vector). */
  readonly perRowVector?: boolean;
  /** `mul(s, t)` instead of `mul(t, s)`. */
  readonly swappedMul?: boolean;
  /** `add(one, sel_scale)` instead of `add(sel_scale, one)`. */
  readonly swappedModulate?: boolean;
  /** `add(sel_shift, p)` instead of `add(p, sel_shift)`. */
  readonly swappedOffset?: boolean;
  /** `one` as `[dim]` (the kernel reads `one[0]` only). */
  readonly wideOne?: boolean;
  /** The layer_norm output is a graph output. */
  readonly normOutput?: boolean;
  /** The selected scale gets a second consumer. */
  readonly selectExtraConsumer?: boolean;
  /** `where(mask, scale_c, scale_c)` (the same value in two bind slots). */
  readonly sharedVector?: boolean;
  /** A zero-dispatch reshape of the selected scale between the `where` and its `add`. */
  readonly gap?: boolean;
};

/**
 * Wan2.2's per-token adaLN: the modulation vectors are picked per token by
 * `where(mask, condition, generation)` and the window is the 6 contiguous nodes.
 */
const adalnTokenGraph = (options: AdalnTokenOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const mask = options.broadcastMask ? [1, 1, 1] : options.fullMask ? row : [1, ROWS, 1];
  const selected = options.broadcastMask ? vector : row;
  const values: DeclarationJson["values"] = {
    t: { dtype, shape: row },
    sel_scale: { dtype, shape: selected },
    s: { dtype, shape: selected },
    p: { dtype, shape: row },
    sel_shift: { dtype, shape: selected },
    y: { dtype, shape: row },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype, shape: row },
    { name: "ln_weight", dtype, shape: [DIM] },
    { name: "ln_bias", dtype, shape: [DIM] },
    { name: "mask", dtype: "bool", shape: mask },
    ...(options.separateMasks ? [{ name: "mask2", dtype: "bool", shape: mask }] : []),
    { name: "one", dtype, shape: options.wideOne ? [DIM] : [1] },
    { name: "scale_c", dtype, shape: options.perRowVector ? row : vector },
    { name: "scale_g", dtype, shape: vector },
    { name: "shift_c", dtype, shape: vector },
    { name: "shift_g", dtype, shape: vector },
  ];
  const extraOutputs: string[] = [];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
    {
      op: "where",
      ins: ["mask", "scale_c", options.sharedVector ? "scale_c" : "scale_g"],
      outs: ["sel_scale"],
      attrs: {},
    },
  ];
  let selectedScale = "sel_scale";
  if (options.gap) {
    values.sel_scale_alias = { dtype, shape: selected };
    nodes.push({ op: "reshape", ins: ["sel_scale"], outs: ["sel_scale_alias"], attrs: {} });
    selectedScale = "sel_scale_alias";
  }
  nodes.push(
    {
      op: "add",
      ins: options.swappedModulate ? ["one", selectedScale] : [selectedScale, "one"],
      outs: ["s"],
      attrs: {},
    },
    { op: "mul", ins: options.swappedMul ? ["s", "t"] : ["t", "s"], outs: ["p"], attrs: {} },
    {
      op: "where",
      ins: [options.separateMasks ? "mask2" : "mask", "shift_c", "shift_g"],
      outs: ["sel_shift"],
      attrs: {},
    },
    {
      op: "add",
      ins: options.swappedOffset ? ["sel_shift", "p"] : ["p", "sel_shift"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.selectExtraConsumer) {
    values.sel_copy = { dtype, shape: selected };
    nodes.push({ op: "neg", ins: ["sel_scale"], outs: ["sel_copy"], attrs: {} });
    extraOutputs.push("sel_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.normOutput ? ["t"] : []), "y", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

const fuseAdalnToken = (options: AdalnTokenOptions = {}): FusionPlan => {
  const graph = adalnTokenGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("per-token adaLN folds the 6 contiguous nodes into a pack dispatch and the shared staged body", () => {
  const plan = fuseAdalnToken();
  assertEquals(outline(plan.steps), ["fused:adaln"]);
  const step = fusedAt(plan, 0);
  // MUST: the total-use list follows the actual node order (the mask is used by both wheres).
  assertEquals(step.ins, [
    "x",
    "ln_weight",
    "ln_bias",
    "mask",
    "scale_c",
    "scale_g",
    "one",
    "mask",
    "shift_c",
    "shift_g",
  ]);
  assertEquals(step.binds, [
    "x",
    "ln_weight",
    "ln_bias",
    "mask",
    "scale_c",
    "scale_g",
    "one",
    "shift_c",
    "shift_g",
  ]);
  assertEquals(step.nodeCount, 6);
  assertEquals(step.outputName, "y");
  // The packed table `[scale_c, scale_g, shift_c, shift_g]` lives across the two dispatches only.
  assertEquals(step.temps, [{ byteLength: 4 * DIM * 4, allocBefore: 0, releaseAfter: 1 }]);
  const [pack, body] = step.dispatches;
  assertEquals(step.dispatches.length, 2);
  assertEquals(pack.key, ADALN_MODULATION_PACK_KEY);
  assertEquals([...pack.params], [DIM, 0, 0, 0]);
  assertEquals(pack.operands, [
    { kind: "bind", index: 4 },
    { kind: "bind", index: 5 },
    { kind: "bind", index: 7 },
    { kind: "bind", index: 8 },
    { kind: "temp", id: 0 },
  ]);
  assertEquals(pack.workgroups, { kind: "gridStride", items: DIM, size: 256 });
  assertEquals(body.key, ADALN_NORM_TOKEN_KEY);
  assertEquals([...body.params], [...adalnNormParams(ROWS, DIM, 1e-6)]);
  // x, weight, bias, mask, table, one, out = 7 storage bindings (the WebGPU default is 8).
  assertEquals(body.operands, [
    { kind: "bind", index: 0 },
    { kind: "bind", index: 1 },
    { kind: "bind", index: 2 },
    { kind: "bind", index: 3 },
    { kind: "temp", id: 0 },
    { kind: "bind", index: 6 },
    { kind: "output" },
  ]);
  assertEquals(body.workgroups, { kind: "gridStride", items: ROWS, size: 1 });
  assertEquals(plan.counts.adaln, 1);
  assertEquals(plan.counts.gatedResidual, 0);
});

test("per-token adaLN counterexamples (two masks / mask and vector shapes / input orders / one's shape / internal output / extra consumer / bind duplicate / gap) are not grabbed", () => {
  const cases: readonly (readonly [string, AdalnTokenOptions])[] = [
    ["two different masks", { separateMasks: true }],
    ["mask [1,1,1]", { broadcastMask: true }],
    ["mask per element", { fullMask: true }],
    ["per-row scale vector", { perRowVector: true }],
    ["mul(s, t)", { swappedMul: true }],
    ["add(one, sel_scale)", { swappedModulate: true }],
    ["add(sel_shift, p)", { swappedOffset: true }],
    ["one as [dim]", { wideOne: true }],
    ["layer_norm output is a graph output", { normOutput: true }],
    ["selected scale has another consumer", { selectExtraConsumer: true }],
    ["where(mask, v, v)", { sharedVector: true }],
    ["reshape between where and add", { gap: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseAdalnToken(options);
    assertEquals(plan.counts.adaln, 0, `${label}: adaln counter`);
    // Neither `where` of the window may be taken by the gated-residual rule instead.
    assertEquals(plan.counts.gatedResidual, 0, `${label}: gatedResidual counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("counters accumulate per rule on a graph with fusions in a row", () => {
  const graph = ropeGraph({ heads: 2 });
  // A SiLU chain is appended right after the RoPE chain (the first ops are disjoint, so they are
  // matched in order).
  graph.values.sig = { dtype: "f32", shape: [1, 2, 5, 128] };
  graph.values.gated = { dtype: "f32", shape: [1, 2, 5, 128] };
  graph.nodes.push(
    { op: "sigmoid", ins: ["y"], outs: ["sig"], attrs: {} },
    { op: "mul", ins: ["y", "sig"], outs: ["gated"], attrs: {} },
  );
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  graph.outputs = ["gated"];

  const plan = fuse(graph, ropeInputs({ heads: 2 }));
  assertEquals(outline(plan.steps), ["fused:rope", "fused:silu"]);
  assertEquals(plan.counts, builtInCounts({ silu: 1, rope: 1 }));
});

// --------------------------------------------- Row-block attention (rowBlockAttention)

type AttentionShape = {
  readonly heads: number;
  readonly queries: number;
  readonly keys: number;
  readonly headDim: number;
};

const ATTENTION: AttentionShape = { heads: 3, queries: 17, keys: 19, headDim: 13 };

/** Knobs for breaking a single point of the window (the default is the observed spelling). */
type AttentionOptions = {
  /** Puts one 0-dispatch alias between bmm and reshape (breaks only the adjacency condition). */
  readonly interpose?: boolean;
  /** Makes the mask `[1,1,M,N]` (per row). */
  readonly rowMask?: boolean;
  /** Makes safe_softmax a plain softmax. */
  readonly plainSoftmax?: boolean;
  /** Makes the V-side expand a broadcast **with a replicated axis** (no longer the identity). */
  readonly broadcastV?: boolean;
  /** Swaps the input order of the last bmm. */
  readonly swapPv?: boolean;
  /** Makes the intermediate S a graph output. */
  readonly scoresOutput?: boolean;
  /** Adds one more consumer of the softmax output. */
  readonly extraConsumer?: boolean;
  /**
   * Makes q and kᵀ the same input (a form with duplicate bind entries — only possible when square).
   */
  readonly sameQk?: boolean;
};

const attentionGraph = (
  shape: AttentionShape = ATTENTION,
  options: AttentionOptions = {},
): DeclarationJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = options.interpose ? "scores3_alias" : "scores3";
  const maskShape = options.rowMask ? [1, 1, queries, keys] : [1, 1, 1, keys];
  // A non-identity expand is built by making one axis of the source 1 so it gets replicated.
  const vSource = options.broadcastV ? [1, 1, keys, headDim] : [1, heads, keys, headDim];
  const values: DeclarationJson["values"] = {
    scores3: { dtype: "f32", shape: scores3 },
    scores4: { dtype: "f32", shape: scores4 },
    masked: { dtype: "f32", shape: scores4 },
    probs: { dtype: "f32", shape: scores4 },
    probsExpanded: { dtype: "f32", shape: scores4 },
    probs3: { dtype: "f32", shape: scores3 },
    vExpanded: { dtype: "f32", shape: [1, heads, keys, headDim] },
    v3: { dtype: "f32", shape: [heads, keys, headDim] },
    y: { dtype: "f32", shape: [heads, queries, headDim] },
  };
  if (options.interpose) values.scores3_alias = { dtype: "f32", shape: scores3 };
  if (options.extraConsumer) values.probsCopy = { dtype: "f32", shape: scores4 };
  const nodes: DeclarationJson["nodes"] = [
    { op: "bmm", ins: ["q", options.sameQk ? "q" : "kt"], outs: [qkOut], attrs: {} },
    ...(options.interpose ? [{ op: "reshape", ins: [qkOut], outs: ["scores3"], attrs: {} }] : []),
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    { op: "add", ins: ["scores4", "mask"], outs: ["masked"], attrs: {} },
    {
      op: options.plainSoftmax ? "softmax" : "safe_softmax",
      ins: ["masked"],
      outs: ["probs"],
      attrs: { dim: 3 },
    },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    {
      op: "bmm",
      ins: options.swapPv ? ["v3", "probs3"] : ["probs3", "v3"],
      outs: ["y"],
      attrs: {},
    },
    ...(options.extraConsumer
      ? [{ op: "neg", ins: ["probs"], outs: ["probsCopy"], attrs: {} }]
      : []),
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      ...(options.sameQk
        ? []
        : [{ name: "kt", dtype: "f32" as const, shape: [heads, headDim, keys] }]),
      { name: "mask", dtype: "f32", shape: maskShape },
      { name: "v", dtype: "f32", shape: vSource },
    ],
    outputs: options.scoresOutput ? ["y", "scores4"] : ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const attentionInputs = (
  shape: AttentionShape = ATTENTION,
  options: AttentionOptions = {},
): Readonly<Record<string, readonly number[]>> => {
  const { heads, queries, keys, headDim } = shape;
  return {
    q: [heads, queries, headDim],
    ...(options.sameQk ? {} : { kt: [heads, headDim, keys] }),
    mask: options.rowMask ? [1, 1, queries, keys] : [1, 1, 1, keys],
    v: options.broadcastV ? [1, 1, keys, headDim] : [1, heads, keys, headDim],
  };
};

test("folds the 9-node window of decomposed attention into one step that consumes external inputs only 4 times in total", () => {
  const plan = fuse(attentionGraph(), attentionInputs());
  assertEquals(outline(plan.steps), ["fused:rowBlockAttention"]);
  const step = fusedAt(plan, 0);
  const operands = ["q", "kt", "mask", "v"];
  assertEquals(
    step.ins,
    operands,
    "external input sequence (exactly the external consumption of the original 9 nodes)",
  );
  assertEquals(step.binds, operands, "bind entries (no duplicates)");
  assertEquals(step.nodeCount, 9);
  assertEquals(step.outputName, "y");
  assertEquals(step.outputShape, [ATTENTION.heads, ATTENTION.queries, ATTENTION.headDim]);
  assertEquals(plan.counts.rowBlockAttention, 1);
  // S is small against the 128MiB limit, so 1 block = the plain 4-dispatch sequence.
  assertEquals(step.dispatches.length, 4, "1 block is 4 dispatches");
  assertEquals(step.temps.length, 3, "the 3 temporaries S / masked S / P");
  assertEquals(
    step.dispatches.map((dispatch) => dispatch.key),
    [
      bmmKey(false, ATTENTION.queries),
      elementwiseKey({ op: "add", rank: 4, dtype: "f32" }),
      SAFE_SOFTMAX_KEY,
      bmmKey(false, ATTENTION.queries),
    ],
    "a single block's keys are the plain bmm (no row-window variant)",
  );
});

test("row blocks are split evenly into the fewest blocks that fit the limit, and temporaries shrink to one block's worth", () => {
  const shape: AttentionShape = { heads: 2, queries: 100, keys: 64, headDim: 8 };
  // 1 row = 2·64·4 = 512B. Limiting to 30 rows' worth gives ceil(100/30) = 4 blocks (25 rows × 4).
  const ir = parse(attentionGraph(shape));
  const plan = planFusions(planGraph(ir, bindSymbols(ir, attentionInputs(shape))).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: { maxStorageBufferBindingSize: 512 * 30, maxComputeWorkgroupsPerDimension: 65535 },
  });
  const step = fusedAt(plan, 0);
  assertEquals(step.dispatches.length, 16, "4 blocks × 4 dispatches");
  assertEquals(step.temps.length, 12, "4 blocks × 3");
  assertEquals(
    [...new Set(step.temps.map((temp) => temp.byteLength))],
    [25 * 512],
    "temporaries are one block's worth (not the full M)",
  );
  assertEquals(
    step.dispatches.map((dispatch) => dispatch.key).slice(0, 4),
    [
      bmmKey(true, 25, "a"),
      elementwiseKey({ op: "add", rank: 4, dtype: "f32" }),
      SAFE_SOFTMAX_KEY,
      bmmKey(true, 25, "c"),
    ],
    "with 2 or more blocks, bmm switches to the row-window variant",
  );
});

test("the attentionRowBlockMaxBytes budget splits decomposed attention by the smaller of it and the binding limit (ADR 0124)", () => {
  const shape: AttentionShape = { heads: 2, queries: 100, keys: 64, headDim: 8 };
  const ir = parse(attentionGraph(shape));
  const nodes = planGraph(ir, bindSymbols(ir, attentionInputs(shape))).nodes;
  const planWith = (budget: number | undefined) =>
    planFusions(nodes, {
      rules: FUSION_RULES,
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      // The binding limit is 128 MiB (1 block without a budget). Only the budget moves the block
      // count.
      limits: TEST_LIMITS,
      ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
    });
  // Without a budget it is the plain 4-dispatch sequence (the default is identical to before the
  // budget field was added).
  assertEquals(fusedAt(planWith(undefined), 0).dispatches.length, 4);
  // 1 row = 2·64·4 = 512B. A budget of 30 rows' worth gives the same 4 blocks (25 rows × 4) as
  // limiting by the binding limit.
  const budgeted = fusedAt(planWith(512 * 30), 0);
  assertEquals(budgeted.dispatches.length, 16, "4 blocks × 4 dispatches");
  assertEquals([...new Set(budgeted.temps.map((temp) => temp.byteLength))], [25 * 512]);
  assertEquals(
    budgeted.dispatches[0].key,
    bmmKey(true, 25, "a"),
    "switches to the row-window variant",
  );
  // A budget at or above the binding limit has no effect (it cannot be used to loosen the limit).
  assertEquals(
    fusedAt(planWith(TEST_LIMITS.maxStorageBufferBindingSize * 2), 0).dispatches.length,
    4,
  );
  // A budget below one row's worth fails loudly naming the budget, instead of dropping a matched
  // window to the plain sequence.
  assertThrows(() => planWith(511), ExecutionError, "set the budget to at least one row");
});

test("the row block split returns the fewest blocks that fit the limit, split evenly, and fails loudly on shapes that do not fit", () => {
  // Exactly at the limit (10 rows × 100B = 1000B) is 1 block.
  assertEquals(planRowBlocks(10, 100, 1000), [{ offset: 0, rows: 10 }]);
  // One element over gives 2 blocks (5 / 5).
  assertEquals(planRowBlocks(10, 100, 999), [
    { offset: 0, rows: 5 },
    { offset: 5, rows: 5 },
  ]);
  // The remainder is handed out one row at a time from the front (3 blocks = 4 / 3 / 3).
  assertEquals(planRowBlocks(10, 100, 400), [
    { offset: 0, rows: 4 },
    { offset: 4, rows: 3 },
    { offset: 7, rows: 3 },
  ]);
  // A forced block count (an internal surface for tests only) is split evenly too.
  assertEquals(planRowBlocks(10, 100, 1000, 3), [
    { offset: 0, rows: 4 },
    { offset: 4, rows: 3 },
    { offset: 7, rows: 3 },
  ]);
  // A shape where even 1 row does not fit fails loudly (silently dropping to the plain sequence
  // would only make the allocation fail).
  assertThrows(
    () => planRowBlocks(10, 1001, 1000),
    ExecutionError,
    "the temporary for one query row",
  );
  // A forced block count that does not fit the limit also fails loudly (it cannot be used to loosen
  // it).
  assertThrows(
    () => planRowBlocks(10, 100, 400, 2),
    ExecutionError,
    "with 2 blocks, one block of 500B",
  );
  // More blocks than rows, or 0 blocks, cannot be taken.
  assertThrows(() => planRowBlocks(10, 100, 1000, 11), ExecutionError);
  assertThrows(() => planRowBlocks(10, 100, 1000, 0), ExecutionError);
});

test("row-block attention counterexamples (adjacency / mask shape / different op / non-identity expand / input order / internal output / another consumer / duplicate bind) do not match", () => {
  // Only a square shape (M = N = D) can build "swap PV's input order" and "make q and kᵀ the same
  // input" without violating the contract. Unless **the base itself matches** is checked first, the
  // two cases below would trivially pass by looking at "a shape that never matched".
  const square: AttentionShape = { heads: 2, queries: 4, keys: 4, headDim: 4 };
  assertEquals(
    fuse(attentionGraph(square), attentionInputs(square)).counts.rowBlockAttention,
    1,
    "the square base does not match (the 2 counterexamples below would trivially pass)",
  );
  const cases: readonly (readonly [
    string,
    DeclarationJson,
    Readonly<Record<string, readonly number[]>>,
  ])[] = [
    ["one alias in between", attentionGraph(ATTENTION, { interpose: true }), attentionInputs()],
    [
      "mask is per row [1,1,M,N]",
      attentionGraph(ATTENTION, { rowMask: true }),
      attentionInputs(ATTENTION, { rowMask: true }),
    ],
    ["plain softmax", attentionGraph(ATTENTION, { plainSoftmax: true }), attentionInputs()],
    [
      "V-side expand has a replicated axis",
      attentionGraph(ATTENTION, { broadcastV: true }),
      attentionInputs(ATTENTION, { broadcastV: true }),
    ],
    ["PV input order reversed", attentionGraph(square, { swapPv: true }), attentionInputs(square)],
    [
      "intermediate S is a graph output",
      attentionGraph(ATTENTION, { scoresOutput: true }),
      attentionInputs(),
    ],
    [
      "another consumer of P",
      attentionGraph(ATTENTION, { extraConsumer: true }),
      attentionInputs(),
    ],
    [
      "q and kᵀ are the same input (duplicate bind)",
      attentionGraph(square, { sameQk: true }),
      attentionInputs(square, { sameQk: true }),
    ],
  ];
  for (const [label, graph, inputs] of cases) {
    const plan = fuse(graph, inputs);
    assertEquals(
      plan.counts.rowBlockAttention,
      0,
      `${label}: matched (${outline(plan.steps).join(",")})`,
    );
  }
});

/**
 * The minimal form of BiRefNet's decomposed attention (window attention). Of the 96 in the observed
 * graph, **0** are matched by `rowBlockAttention`, and three reasons for missing hold at once:
 *
 * ① the mask is `[1,H,S,S]` (a per-head relative-position bias), not the required `[1,1,1,N]`
 * ② a plain `softmax` instead of `safe_softmax`
 * ③ an `embedding` (the relative-position table lookup) and its `reshape` sit mid-window
 *
 * This is the form ACTIVE_DESIGN means by "**for row-block alone, missing costs resources, not
 * performance**": without a match, `bmm`'s output `[nW·H,S,S]` is materialized as **a node output
 * slot** (which in principle cannot be split into row blocks). Widening or narrowing the matcher
 * would make a regression in either direction silent, so the non-match itself and its resource
 * consequence are kept as observation points.
 */
const birefnetAttentionGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["bmm", "reshape", "embedding", "add", "softmax", "expand"] },
  symbols: [],
  inputs: [
    { name: "q", dtype: "f32", shape: [6, 4, 5] },
    { name: "kt", dtype: "f32", shape: [6, 5, 4] },
    { name: "idx", dtype: "i32", shape: [48] },
    { name: "v", dtype: "f32", shape: [2, 3, 4, 5] },
  ],
  outputs: ["y"],
  initializers: { table: {} },
  values: {
    table: { dtype: "f32", shape: [8, 1] },
    scores3: { dtype: "f32", shape: [6, 4, 4] },
    scores4: { dtype: "f32", shape: [2, 3, 4, 4] },
    biasFlat: { dtype: "f32", shape: [48, 1] },
    bias4: { dtype: "f32", shape: [1, 3, 4, 4] },
    masked: { dtype: "f32", shape: [2, 3, 4, 4] },
    probs: { dtype: "f32", shape: [2, 3, 4, 4] },
    probsExpanded: { dtype: "f32", shape: [2, 3, 4, 4] },
    probs3: { dtype: "f32", shape: [6, 4, 4] },
    vExpanded: { dtype: "f32", shape: [2, 3, 4, 5] },
    v3: { dtype: "f32", shape: [6, 4, 5] },
    y: { dtype: "f32", shape: [6, 4, 5] },
  },
  nodes: [
    { op: "bmm", ins: ["q", "kt"], outs: ["scores3"], attrs: {} },
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    // ③ The relative-position table lookup in the middle of the window (2 nodes).
    { op: "embedding", ins: ["table", "idx"], outs: ["biasFlat"], attrs: { padding_idx: -1 } },
    { op: "reshape", ins: ["biasFlat"], outs: ["bias4"], attrs: {} },
    // ① The mask is per head (`[1,H,S,S]`).
    { op: "add", ins: ["scores4", "bias4"], outs: ["masked"], attrs: {} },
    // ② A plain softmax.
    { op: "softmax", ins: ["masked"], outs: ["probs"], attrs: { dim: 3 } },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["probs3", "v3"], outs: ["y"], attrs: {} },
  ],
});

const birefnetInputs: Readonly<Record<string, readonly number[]>> = {
  q: [6, 4, 5],
  kt: [6, 5, 4],
  idx: [48],
  v: [2, 3, 4, 5],
};

/**
 * "Name → bytes" of the materialized node outputs (aliases are excluded since they allocate
 * nothing).
 */
const allocatedOutputs = (plan: FusionPlan): ReadonlyMap<string, number> => {
  const bytes = new Map<string, number>();
  for (const step of plan.steps) {
    if (step.kind !== "node" || step.aliasesInput) continue;
    for (const output of step.plan.outputs) {
      bytes.set(output.name, output.shape.reduce((size, dim) => size * dim, 1) * 4);
    }
  }
  return bytes;
};

test("birefnet-shaped decomposed attention does not match, and S is materialized as a node output slot", () => {
  const plan = fuse(birefnetAttentionGraph(), birefnetInputs);
  assertEquals(plan.counts.rowBlockAttention, 0, outline(plan.steps).join(","));

  // The resource consequence: S = nW·H·S·S·4 = 2·3·4·4·4 appears as **a node output**. A temporary
  // inside a fused step would shrink to one block's worth with row blocks, but a node output slot
  // cannot be split.
  assertEquals(allocatedOutputs(plan).get("scores3"), 2 * 3 * 4 * 4 * 4);

  // Control: the form with mask [1,1,1,N], safe_softmax and a contiguous 9-node window matches, and
  // the same S becomes an in-step temporary (= the side that shrinks with row blocks).
  const fused = fuse(attentionGraph(), attentionInputs());
  assertEquals(fused.counts.rowBlockAttention, 1);
  const scoreBytes = ATTENTION.heads * ATTENTION.queries * ATTENTION.keys * 4;
  assertEquals(
    fusedAt(fused, 0).temps.filter((temp) => temp.byteLength === scoreBytes).length,
    3,
    "S / masked S / P are not in-step temporaries",
  );
  assertEquals(allocatedOutputs(fused).has("scores3"), false, "S remains as a node output");
});

test("BSHD RoPE passes the head count as a uniform in both issue orders and folds only private intermediates", () => {
  for (const order of ["slice-first", "direct-first"] as const) {
    const options: RopeOptions = { layout: "bshd", heads: 4, headDim: 256, order };
    const plan = fuse(ropeGraph(options), ropeInputs(options));
    assertEquals(outline(plan.steps), ["fused:rope"]);
    const step = fusedAt(plan, 0);
    assertEquals(step.binds, ["x", "cos", "sin"]);
    assertEquals(step.nodeCount, 7);
    assertEquals(step.dispatches[0].key, ROPE_BSHD_KEY);
    assertEquals([...step.dispatches[0].params], [4 * 5 * 256, 4, 256, 128]);
    for (const negative of [
      { interpose: true },
      { internalOutput: true },
      { extraConsumer: true },
      { nearSplit: true },
      { batch2: true },
      { swappedAdd: true },
      { nearTable: true },
      { gapReshape: true },
    ]) {
      const rejected = { ...options, ...negative };
      assertEquals(
        fuse(ropeGraph(rejected), ropeInputs(rejected)).counts.rope,
        0,
        JSON.stringify(rejected),
      );
    }
  }
});

test("BSHD RoPE does not match when cos/sin use different axes, and keeps the slice of the shared position table", () => {
  const options: RopeOptions = { layout: "bshd", heads: 4, headDim: 128, order: "direct-first" };
  const graph = ropeGraph(options);
  graph.inputs[1].shape = [1, 1, 4, 128];
  assertEquals(fuse(graph, { ...ropeInputs(options), cos: [1, 1, 4, 128] }).counts.rope, 0);
  const sliced = { ...options, prefixSlicedSin: true };
  const plan = fuse(ropeGraph(sliced), ropeInputs(sliced));
  assertEquals(outline(plan.steps), ["sym_prefix_slice", "fused:rope"]);
  assertEquals(fusedAt(plan, 1).dispatches[0].key, ROPE_BSHD_KEY);
});

// ------------------------------------------------------------ RoPE (pair form)

type RopePairOptions = {
  readonly heads?: number;
  readonly headDim?: number;
  /** `pairs` is a graph input of its own instead of the reshape of `x` (the kernel binds both). */
  readonly separatePairs?: boolean;
  /** A zero-dispatch reshape between `cat` and the reshape back (breaks the exact wiring only). */
  readonly interpose?: boolean;
  /** `odd` is a graph output (an internal value must stay private). */
  readonly internalOutput?: boolean;
  /** `negative` gets a second consumer. */
  readonly extraConsumer?: boolean;
  /** The two slices swap their boundaries (`0..1` is negated — the opposite rotation). */
  readonly swappedSlices?: boolean;
  /** `cat(even, negative)` instead of `cat(negative, even)`. */
  readonly swappedCat?: boolean;
  /** `add(cross, direct)` instead of `add(direct, cross)`. */
  readonly swappedAdd?: boolean;
  /** cos / sin as `[1,1,S,D]` (the BHSD broadcast) while x stays `[1,S,H,D]`. */
  readonly bhsdTables?: boolean;
  /** cos / sin as `[1,S,H,D]` (per-head tables — no broadcast). */
  readonly perHeadTables?: boolean;
  /**
   * Irodori's first use of the shared tables: a `sym_prefix_slice` of each table sits between the
   * reshape back and its `mul` (two in-window nodes the pair form does not accept).
   */
  readonly prefixSlicedTables?: boolean;
  /** Batch 2 (the accepted form is B=1). */
  readonly batch2?: boolean;
};

/**
 * Wan's DiT RoPE (the complex rotation in real numbers): `x` `[1,S,H,D]` is viewed as pairs
 * `[1,S,H·D/2,2]`, the pair is swapped with the second element negated, viewed back and combined
 * with the `[1,S,1,D]` tables. S = 5.
 */
const ropePairGraph = (options: RopePairOptions = {}): DeclarationJson => {
  const heads = options.heads ?? 2;
  const headDim = options.headDim ?? 128;
  const batch = options.batch2 ? 2 : 1;
  const sequence = 5;
  const pairCount = (heads * headDim) / 2;
  const full = [batch, sequence, heads, headDim];
  const pairs = [batch, sequence, pairCount, 2];
  const half = [batch, sequence, pairCount, 1];
  const table = options.perHeadTables
    ? full
    : options.bhsdTables
      ? [1, 1, sequence, headDim]
      : [1, sequence, 1, headDim];
  const values: DeclarationJson["values"] = {
    odd: { dtype: "f32", shape: half },
    negative: { dtype: "f32", shape: half },
    even: { dtype: "f32", shape: half },
    rotated: { dtype: "f32", shape: pairs },
    rot: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype: "f32", shape: full }];
  const nodes: DeclarationJson["nodes"] = [];
  if (options.separatePairs) {
    inputs.push({ name: "pairs", dtype: "f32", shape: pairs });
  } else {
    values.pairs = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["x"], outs: ["pairs"], attrs: {} });
  }
  const [oddRange, evenRange] = options.swappedSlices
    ? [
        { start: 0, end: 1 },
        { start: 1, end: 2 },
      ]
    : [
        { start: 1, end: 2 },
        { start: 0, end: 1 },
      ];
  nodes.push(
    { op: "slice", ins: ["pairs"], outs: ["odd"], attrs: { dim: 3, ...oddRange } },
    { op: "neg", ins: ["odd"], outs: ["negative"], attrs: {} },
    { op: "slice", ins: ["pairs"], outs: ["even"], attrs: { dim: 3, ...evenRange } },
    {
      op: "cat",
      ins: options.swappedCat ? ["even", "negative"] : ["negative", "even"],
      outs: ["rotated"],
      attrs: { dim: 3 },
    },
  );
  let rotatedForView = "rotated";
  if (options.interpose) {
    values.rotated_alias = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["rotated"], outs: ["rotated_alias"], attrs: {} });
    rotatedForView = "rotated_alias";
  }
  nodes.push({ op: "reshape", ins: [rotatedForView], outs: ["rot"], attrs: {} });
  const prefixSlice = (name: string): void => {
    values[`${name}_table`] = { dtype: "f32", shape: [1, 8, 1, headDim] };
    values[name] = { dtype: "f32", shape: [1, "T", 1, headDim] };
    nodes.push({
      op: "sym_prefix_slice",
      ins: [`${name}_table`],
      outs: [name],
      attrs: { sym: "T", slices: [{ dim: 1, coeff: 1, offset: 0 }] },
    });
  };
  if (options.prefixSlicedTables) prefixSlice("cos");
  nodes.push({ op: "mul", ins: ["x", "cos"], outs: ["direct"], attrs: {} });
  if (options.prefixSlicedTables) prefixSlice("sin");
  nodes.push(
    { op: "mul", ins: ["rot", "sin"], outs: ["cross"], attrs: {} },
    {
      op: "add",
      ins: options.swappedAdd ? ["cross", "direct"] : ["direct", "cross"],
      outs: ["y"],
      attrs: {},
    },
  );
  if (options.extraConsumer) {
    values.negative_copy = { dtype: "f32", shape: half };
    nodes.push({ op: "neg", ins: ["negative"], outs: ["negative_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: options.prefixSlicedTables ? ["T"] : [],
    inputs: [
      ...inputs,
      ...(options.prefixSlicedTables
        ? [{ name: "bind", dtype: "f32", shape: ["T"] }]
        : [
            { name: "cos", dtype: "f32", shape: table },
            { name: "sin", dtype: "f32", shape: table },
          ]),
    ],
    outputs: [
      ...(options.internalOutput ? ["odd"] : []),
      "y",
      ...(options.extraConsumer ? ["negative_copy"] : []),
    ],
    initializers: options.prefixSlicedTables ? { cos_table: {}, sin_table: {} } : {},
    values,
    nodes,
  };
};

const ropePairInputs = (graph: DeclarationJson): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [spec.name, spec.name === "bind" ? [5] : (spec.shape as number[])]),
  );

const fuseRopePair = (options: RopePairOptions = {}): FusionPlan => {
  const graph = ropePairGraph(options);
  return fuse(graph, ropePairInputs(graph));
};

test("pair-form RoPE folds the 8-node window into one step that binds x and its pair view separately", () => {
  for (const [heads, headDim] of [
    [2, 128],
    [24, 128],
    [10, 64],
    [3, 6],
  ] as const) {
    const label = `H=${heads} D=${headDim}`;
    const plan = fuseRopePair({ heads, headDim });
    // The reshape x → pairs lies before the window and stays a zero-dispatch alias.
    assertEquals(outline(plan.steps), ["reshape", "fused:rope"], label);
    const step = fusedAt(plan, 1);
    // MUST: the total-use list follows the actual node order (slice, slice, then the direct mul).
    assertEquals(step.ins, ["pairs", "pairs", "x", "cos", "sin"], `${label}: total uses`);
    assertEquals(step.binds, ["x", "pairs", "cos", "sin"], `${label}: bind order`);
    assertEquals(step.nodeCount, 8, label);
    assertEquals(step.outputName, "y", label);
    assertEquals(step.dispatches.length, 1, label);
    assertEquals(step.dispatches[0].key, ROPE_PAIR_BSHD_KEY, label);
    // params are [element count, heads (the table divisor), head_dim, half_dim].
    assertEquals(
      [...step.dispatches[0].params],
      [5 * heads * headDim, heads, headDim, headDim / 2],
      label,
    );
    assertEquals(plan.counts.rope, 1, label);
  }
  // The kernel reads the rotation from `pairs` itself, so a pair view that is not a reshape of x
  // (here a graph input of its own) is folded as well — the window's semantics, not the producer.
  const separate = fuseRopePair({ separatePairs: true });
  assertEquals(outline(separate.steps), ["fused:rope"]);
  assertEquals(fusedAt(separate, 0).binds, ["x", "pairs", "cos", "sin"]);
});

test("pair-form RoPE counterexamples (alias / internal output / extra consumer / slice boundaries / cat order / add order / table shapes / batch / Irodori's first use) are not grabbed", () => {
  const cases: readonly (readonly [string, RopePairOptions])[] = [
    ["interposed alias", { interpose: true }],
    ["internal output", { internalOutput: true }],
    ["extra consumer", { extraConsumer: true }],
    ["swapped slice boundaries", { swappedSlices: true }],
    ["cat(even, negative)", { swappedCat: true }],
    ["add(cross, direct)", { swappedAdd: true }],
    // H = S so that the `[1,1,S,D]` table broadcasts at all (a legal mul, not this rotation).
    ["BHSD tables with BSHD x", { bhsdTables: true, heads: 5 }],
    ["per-head tables", { perHeadTables: true }],
    ["batch 2", { batch2: true }],
    // Irodori's DiT / speaker: the first use of the shared tables slices them inside the window.
    // Only the later uses (exactly the Wan spelling) are folded.
    ["two sym_prefix_slice in the window", { prefixSlicedTables: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseRopePair(options);
    assertEquals(plan.counts.rope, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("pair-form RoPE does not take the half-split window and the half split does not take the pair form", () => {
  // The two forms share the leading `slice`; the second op decides. A half-split window keeps
  // its own key, and a pair window never becomes a half-split step.
  const half = fuse(
    ropeGraph({ layout: "bshd", heads: 4 }),
    ropeInputs({ layout: "bshd", heads: 4 }),
  );
  assertEquals(fusedAt(half, 0).dispatches[0].key, ROPE_BSHD_KEY);
  const pair = fuseRopePair({ heads: 4 });
  assertEquals(fusedAt(pair, 1).dispatches[0].key, ROPE_PAIR_BSHD_KEY);
});

// ------------------------------------------------------------ gated residual (gatedResidual)

type GatedResidualOptions = {
  /** `mul(g, y)` instead of `mul(y, g)`. */
  readonly swappedMul?: boolean;
  /** `add(p, residual)` instead of `add(residual, p)`. */
  readonly swappedAdd?: boolean;
  /** The mask is `[1,1,1]` (one flag for every row). */
  readonly broadcastMask?: boolean;
  /** The mask is `[1,rows,dim]` (per element). */
  readonly fullMask?: boolean;
  /** The condition-side gate is `[1,rows,dim]` (not a broadcast vector). */
  readonly perRowGate?: boolean;
  /** `y` is a broadcast `[1,1,dim]` vector. */
  readonly broadcastY?: boolean;
  /** The selected gate is a graph output. */
  readonly gateOutput?: boolean;
  /** The product gets a second consumer. */
  readonly productExtraConsumer?: boolean;
  /** `where(mask, gate_c, gate_c)`. */
  readonly sharedGate?: boolean;
  /** The residual is `y` itself (`add(y, mul(y, g))` — one value in two bind slots). */
  readonly residualIsY?: boolean;
  /** A zero-dispatch reshape of the selected gate between the `where` and the `mul`. */
  readonly gap?: boolean;
};

/** Wan2.2's gated residual: `add(residual, mul(y, where(mask, gate_c, gate_g)))`. */
const gatedResidualGraph = (options: GatedResidualOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const row = [1, ROWS, DIM];
  const vector = [1, 1, DIM];
  const mask = options.broadcastMask ? [1, 1, 1] : options.fullMask ? row : [1, ROWS, 1];
  const gate = options.broadcastMask && !options.perRowGate ? vector : row;
  const values: DeclarationJson["values"] = {
    g: { dtype, shape: gate },
    p: { dtype, shape: row },
    out: { dtype, shape: row },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "mask", dtype: "bool", shape: mask },
    { name: "gate_c", dtype, shape: options.perRowGate ? row : vector },
    { name: "gate_g", dtype, shape: vector },
    { name: "y", dtype, shape: options.broadcastY ? vector : row },
    ...(options.residualIsY ? [] : [{ name: "residual", dtype, shape: row }]),
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "where",
      ins: ["mask", "gate_c", options.sharedGate ? "gate_c" : "gate_g"],
      outs: ["g"],
      attrs: {},
    },
  ];
  let selected = "g";
  if (options.gap) {
    values.g_alias = { dtype, shape: gate };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
    selected = "g_alias";
  }
  const residual = options.residualIsY ? "y" : "residual";
  nodes.push(
    {
      op: "mul",
      ins: options.swappedMul ? [selected, "y"] : ["y", selected],
      outs: ["p"],
      attrs: {},
    },
    {
      op: "add",
      ins: options.swappedAdd ? ["p", residual] : [residual, "p"],
      outs: ["out"],
      attrs: {},
    },
  );
  const extraOutputs: string[] = [];
  if (options.productExtraConsumer) {
    values.p_copy = { dtype, shape: row };
    nodes.push({ op: "neg", ins: ["p"], outs: ["p_copy"], attrs: {} });
    extraOutputs.push("p_copy");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...(options.gateOutput ? ["g"] : []), "out", ...extraOutputs],
    initializers: {},
    values,
    nodes,
  };
};

const fuseGatedResidual = (options: GatedResidualOptions = {}): FusionPlan => {
  const graph = gatedResidualGraph(options);
  return fuse(graph, adalnInputs(graph));
};

test("the gated residual folds where → mul → add into one step that reads the gate per token", () => {
  const plan = fuseGatedResidual();
  assertEquals(outline(plan.steps), ["fused:gatedResidual"]);
  const step = fusedAt(plan, 0);
  assertEquals(step.ins, ["mask", "gate_c", "gate_g", "y", "residual"]);
  assertEquals(step.binds, ["mask", "gate_c", "gate_g", "y", "residual"]);
  assertEquals(step.nodeCount, 3);
  assertEquals(step.outputName, "out");
  assertEquals(step.temps, []);
  assertEquals(step.dispatches.length, 1);
  assertEquals(step.dispatches[0].key, GATED_RESIDUAL_KEY);
  assertEquals([...step.dispatches[0].params], [ROWS * DIM, DIM, 0, 0]);
  assertEquals(step.dispatches[0].workgroups, {
    kind: "gridStride",
    items: ROWS * DIM,
    size: 256,
  });
  assertEquals(plan.counts.gatedResidual, 1);
});

test("gated residual counterexamples (input orders / mask, gate and y shapes / internal output / extra consumer / bind duplicates / gap) are not grabbed", () => {
  const cases: readonly (readonly [string, GatedResidualOptions])[] = [
    ["mul(g, y)", { swappedMul: true }],
    ["add(p, residual)", { swappedAdd: true }],
    ["mask [1,1,1]", { broadcastMask: true }],
    ["mask per element", { fullMask: true }],
    ["per-row gate", { perRowGate: true }],
    ["broadcast y", { broadcastY: true }],
    ["selected gate is a graph output", { gateOutput: true }],
    ["product has another consumer", { productExtraConsumer: true }],
    ["where(mask, g, g)", { sharedGate: true }],
    ["residual is y", { residualIsY: true }],
    ["reshape between where and mul", { gap: true }],
  ];
  for (const [label, options] of cases) {
    const plan = fuseGatedResidual(options);
    assertEquals(plan.counts.gatedResidual, 0, `${label}: fusion counter`);
    assertEquals(
      plan.steps.every((step) => step.kind === "node"),
      true,
      `${label}: ${outline(plan.steps).join(",")}`,
    );
  }
});

test("a Wan2.2-like block folds the gated residual and the per-token adaLN next to each other", () => {
  // where → mul → add (the attention's gated residual) feeds the next layer_norm of the per-token
  // adaLN (the FFN's modulation), as in Wan2.2's DiT blocks.
  const graph = gatedResidualGraph();
  const token = adalnTokenGraph();
  const rename = (name: string): string =>
    name === "x" ? "out" : name === "mask" ? "mask" : `ffn_${name}`;
  graph.inputs.push(
    ...token.inputs
      .filter((spec) => spec.name !== "x" && spec.name !== "mask")
      .map((spec) => ({ ...spec, name: rename(spec.name) })),
  );
  for (const [name, spec] of Object.entries(token.values)) graph.values[rename(name)] = spec;
  graph.nodes.push(
    ...token.nodes.map((node) => ({
      ...node,
      ins: node.ins.map(rename),
      outs: node.outs.map(rename),
    })),
  );
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  graph.outputs = ["ffn_y"];
  const plan = fuse(graph, adalnInputs(graph));
  assertEquals(outline(plan.steps), ["fused:gatedResidual", "fused:adaln"]);
  assertEquals(plan.counts.gatedResidual, 1);
  assertEquals(plan.counts.adaln, 1);
  // The mask is consumed by both steps (1 + 2 total uses), and the residual output feeds the
  // adaLN step.
  assertEquals(fusedAt(plan, 1).ins.slice(0, 1), ["out"]);
});

// ------------------------------------------------ linear prologue (perf-ledger F-4)

/** Wan2.2's FFN down-projection row (K 14,336). N is shrunk: only K enters the decision. */
const PROLOGUE_K = 14336;
const PROLOGUE_N = 16;
const PROLOGUE_ROWS = 7;

type ProloguePairOptions = {
  /** The producer op (default `gelu_tanh`). */
  readonly op?: string;
  /** The rows S of the activation `[1, S, K]` (default 7). */
  readonly rows?: number;
  /** The reduction length K (default {@link PROLOGUE_K}). */
  readonly k?: number;
  /** The output width N (default {@link PROLOGUE_N}). */
  readonly n?: number;
  /** The producer's output gets a second consumer. */
  readonly extraConsumer?: boolean;
  /** The producer's output is a graph output. */
  readonly producerOutput?: boolean;
  /** The consumer is a `matmul` reading the output as its activation (rank 2). */
  readonly matmulConsumer?: boolean;
  /** The linear reads the producer's output in its bias slot (the activation is another input). */
  readonly biasSlot?: boolean;
  /** An unrelated node between the producer and the linear. */
  readonly gap?: boolean;
};

/** `y = linear(op(x), w, b)` — the shape of a DiT's FFN down projection (default: Wan2.2's). */
const prologuePairGraph = (options: ProloguePairOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const k = options.k ?? PROLOGUE_K;
  const n = options.n ?? PROLOGUE_N;
  const rows = options.rows ?? PROLOGUE_ROWS;
  const matmul = options.matmulConsumer === true;
  const row = matmul ? [rows, k] : [1, rows, k];
  const produced = options.biasSlot ? [n] : row;
  const inputs: DeclarationJson["inputs"] = [{ name: "x", dtype, shape: produced }];
  const values: DeclarationJson["values"] = {
    g: { dtype, shape: produced },
    w: { dtype, shape: matmul ? [k, n] : [n, k] },
    y: { dtype, shape: matmul ? [rows, n] : [1, rows, n] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  const nodes: DeclarationJson["nodes"] = [
    { op: options.op ?? "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} },
  ];
  const outputs = options.producerOutput ? ["g"] : [];
  if (options.gap) {
    inputs.push({ name: "z", dtype, shape: [4] });
    values.z_neg = { dtype, shape: [4] };
    nodes.push({ op: "neg", ins: ["z"], outs: ["z_neg"], attrs: {} });
    outputs.push("z_neg");
  }
  if (matmul) {
    nodes.push({ op: "matmul", ins: ["g", "w"], outs: ["y"], attrs: {} });
  } else if (options.biasSlot) {
    inputs.push({ name: "a", dtype, shape: row });
    nodes.push({ op: "linear", ins: ["a", "w", "g"], outs: ["y"], attrs: {} });
  } else {
    values.b = { dtype, shape: [n] };
    initializers.b = {};
    nodes.push({ op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} });
  }
  outputs.push("y");
  if (options.extraConsumer) {
    values.g_neg = { dtype, shape: produced };
    nodes.push({ op: "neg", ins: ["g"], outs: ["g_neg"], attrs: {} });
    outputs.push("g_neg");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs,
    initializers,
    values,
    nodes,
  };
};

type PrologueContext = Partial<Parameters<typeof planFusions>[1]>;

/** The a8 Session's facts: `linearCompute: "a8"` and the weight resident as i8. */
const A8_I8: PrologueContext = {
  linearCompute: "a8",
  weightLayouts: new Map([["w", { storage: "i8" }]]),
};

const fusePrologue = (
  options: ProloguePairOptions = {},
  context: PrologueContext = A8_I8,
): FusionPlan => {
  const graph = prologuePairGraph(options);
  const ir = parse(graph);
  return planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...context,
  });
};

/** A node step's `prologue` (undefined for fused steps and plain nodes). */
const prologueOf = (step: ExecStep) => (step.kind === "node" ? step.prologue : undefined);

test("the linear prologue folds gelu_tanh and the erf gelu into the a8 linear that reads it: one node step whose input is the producer's input", () => {
  const cases: readonly (readonly [string, ProloguePairOptions, PrologueContext])[] = [
    ["Wan2.2 down projection (K 14,336, i8)", {}, A8_I8],
    [
      "i4 residency (w4a8)",
      {},
      { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "i4", groupSize: 32 }]]) },
    ],
    ["the prologue bound (K 16,384)", { k: QUANTIZE_ROWS_PROLOGUE_MAX_DIM }, A8_I8],
    ["the smallest row (K 4)", { k: 4 }, A8_I8],
    ["one row", { rows: 1 }, A8_I8],
    // anima's DiT FFN down projection: gelu [S, 8192] → linear with the i8 weight [2048, 8192].
    ["erf gelu, anima's down projection", { op: "gelu", rows: 4096, k: 8192, n: 2048 }, A8_I8],
  ];
  for (const [label, options, context] of cases) {
    const plan = fusePrologue(options, context);
    assertEquals(outline(plan.steps), ["linear"], label);
    const [step] = plan.steps;
    assertEquals(prologueOf(step), { op: options.op ?? "gelu_tanh" }, label);
    if (step.kind !== "node") throw new Error(`${label}: not a node step`);
    assertEquals(step.plan.node.ins, ["x", "w", "b"], `${label}: re-pointed input`);
    assertEquals(
      step.plan.outputs.map((out) => out.name),
      ["y"],
      label,
    );
    assertEquals(step.aliasesInput, false, label);
    assertEquals(plan.counts, builtInCounts({ linearPrologue: 1 }), label);
  }
});

test("the linear prologue keeps the total-use accounting exact: the producer's input trades the producer's use for the linear's, and the folded value is never consumed or defined", () => {
  // up projection → gelu_tanh → down projection, with the up projection's output `h` also read by
  // a residual add after the down projection (so `h` has two uses before and after the fold).
  const dtype = "f32";
  const row = [1, 4, 8];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "gelu_tanh", "linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype, shape: row }],
    outputs: ["out"],
    initializers: { w_up: {}, b_up: {}, w: {}, b: {} },
    values: {
      w_up: { dtype, shape: [8, 8] },
      b_up: { dtype, shape: [8] },
      w: { dtype, shape: [8, 8] },
      b: { dtype, shape: [8] },
      h: { dtype, shape: row },
      g: { dtype, shape: row },
      y: { dtype, shape: row },
      out: { dtype, shape: row },
    },
    nodes: [
      { op: "linear", ins: ["x", "w_up", "b_up"], outs: ["h"], attrs: {} },
      { op: "gelu_tanh", ins: ["h"], outs: ["g"], attrs: {} },
      { op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} },
      { op: "add", ins: ["y", "h"], outs: ["out"], attrs: {} },
    ],
  };
  const ir = parse(graph);
  const uses = countUses(ir);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, { x: row })).nodes, {
    rules: FUSION_RULES,
    useCounts: uses,
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    linearCompute: "a8",
    weightLayouts: new Map([
      ["w_up", { storage: "i8" }],
      ["w", { storage: "i8" }],
    ]),
  });
  assertEquals(outline(plan.steps), ["linear", "linear", "add"]);
  assertEquals(plan.steps.map(prologueOf), [undefined, { op: "gelu_tanh" }, undefined]);
  // The release bookkeeping reads `plan.node.ins` of a node step and `ins` of a fused step.
  const consumed = new Map<string, number>();
  const defined = new Set<string>();
  for (const step of plan.steps) {
    const names = step.kind === "node" ? step.plan.node.ins : step.ins;
    for (const name of names) consumed.set(name, (consumed.get(name) ?? 0) + 1);
    if (step.kind === "node") for (const out of step.plan.outputs) defined.add(out.name);
    else defined.add(step.outputName);
  }
  assertEquals(uses.get("g"), 1, "the folded value had exactly one use");
  assertEquals(consumed.get("g"), undefined, "the folded value is never consumed");
  assertEquals(defined.has("g"), false, "the folded value is never defined");
  for (const [name, count] of uses) {
    if (name === "g") continue;
    assertEquals(consumed.get(name), count, `total uses of '${name}'`);
  }
  assertEquals(consumed.get("h"), 2, "h: the down projection and the residual add");
});

test("linear prologue counterexamples (use count 2 / graph output / not a linear / another slot / a gap / K / an op outside the list / no rows) keep both nodes as they are", () => {
  const cases: readonly (readonly [string, ProloguePairOptions])[] = [
    ["the output has a second consumer", { extraConsumer: true }],
    ["the output is a graph output", { producerOutput: true }],
    ["the consumer is a matmul", { matmulConsumer: true }],
    ["the linear reads it in the bias slot", { biasSlot: true }],
    ["an unrelated node between the two", { gap: true }],
    ["K % 4 != 0", { k: PROLOGUE_K - 2 }],
    ["K above the prologue bound", { k: QUANTIZE_ROWS_PROLOGUE_MAX_DIM + 4 }],
    // Elementwise f32 unary ops that are not prologue ops (sigmoid is also the silu rule's head).
    ["relu (not a prologue op)", { op: "relu" }],
    ["sigmoid (not a prologue op)", { op: "sigmoid" }],
    ["no rows", { rows: 0 }],
  ];
  for (const [label, options] of cases) {
    const plan = fusePrologue(options);
    assertEquals(plan.counts.linearPrologue, 0, `${label}: counter`);
    assertEquals(
      outline(plan.steps),
      prologuePairGraph(options).nodes.map((node) => node.op),
      `${label}: plain nodes`,
    );
    assertEquals(
      plan.steps.map(prologueOf),
      plan.steps.map(() => undefined),
      label,
    );
  }
});

test("the linear prologue needs the a8 Session's i8a8 route: f32 / f16 / bf16 weights and linearCompute f32 / f16 / unset do not fold", () => {
  const cases: readonly (readonly [string, PrologueContext])[] = [
    ["f32 weight (not resident compressed)", { linearCompute: "a8", weightLayouts: new Map() }],
    ["no weight layouts", { linearCompute: "a8" }],
    ["f16 weight", { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "f16" }]]) }],
    ["bf16 weight", { linearCompute: "a8", weightLayouts: new Map([["w", { storage: "bf16" }]]) }],
    ["linearCompute f32", { ...A8_I8, linearCompute: "f32" }],
    ["linearCompute f16", { ...A8_I8, linearCompute: "f16" }],
    ["linearCompute unset", { weightLayouts: A8_I8.weightLayouts }],
  ];
  for (const [label, context] of cases) {
    const plan = fusePrologue({}, context);
    assertEquals(plan.counts.linearPrologue, 0, `${label}: counter`);
    assertEquals(outline(plan.steps), ["gelu_tanh", "linear"], label);
  }
});

test("the linear prologue rejects a producer that is not f32 in and out", () => {
  // The contract runs gelu_tanh on f32 only, so a planned graph cannot carry another dtype; the
  // matcher's own check is exercised on a plan with the producer's dtypes replaced.
  const graph = prologuePairGraph();
  const ir = parse(graph);
  const nodes = planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes;
  const [producer, linear] = nodes;
  for (const [label, replaced] of [
    ["input i32", { ...producer, inputDtypes: ["i32"] }],
    ["output i32", { ...producer, outputs: [{ ...producer.outputs[0], dtype: "i32" }] }],
  ] as const) {
    const plan = planFusions([replaced, linear], {
      rules: FUSION_RULES,
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: TEST_LIMITS,
      ...A8_I8,
    });
    assertEquals(plan.counts.linearPrologue, 0, label);
    assertEquals(outline(plan.steps), ["gelu_tanh", "linear"], label);
  }
});

test("the one fusion rule whose head is a prologue op is geluMulStaticQuantize (the rules are tried first at each index; that rule never accepts the linear prologue's index — the next test)", () => {
  const owners = FUSION_RULES.flatMap((rule) =>
    rule.heads
      .filter((head) => LINEAR_PROLOGUE_OPS.some((op) => op === head))
      .map((head) => [head, rule.name] as const),
  );
  assertEquals(owners, [["gelu_tanh", "geluMulStaticQuantize"]]);
  assertEquals(LINEAR_PROLOGUE_OPS, ["gelu_tanh", "gelu"]);
});

test("geluMulStaticQuantize and the linear prologue never accept the same gelu_tanh: each folds its own form in one graph", () => {
  // gelu_tanh → a8 linear (the prologue's form) and gelu_tanh → mul → static_quantize (the rule's
  // form), both enabled. The gelu value has one use either way, so the next node decides which
  // mechanism can accept; a rule that took the prologue's gelu would leave linearPrologue at 0.
  const dtype = "f32";
  const row = [1, 4, 256];
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["gelu_tanh", "linear", "mul", "static_quantize"] },
    symbols: [],
    inputs: [
      { name: "x", dtype, shape: row },
      { name: "gate", dtype, shape: row },
      { name: "ple", dtype, shape: row },
    ],
    outputs: ["y", "q"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype, shape: [8, 256] },
      b: { dtype, shape: [8] },
      g: { dtype, shape: row },
      y: { dtype, shape: [1, 4, 8] },
      h: { dtype, shape: row },
      m: { dtype, shape: row },
      q: { dtype, shape: row },
    },
    nodes: [
      { op: "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} },
      { op: "linear", ins: ["g", "w", "b"], outs: ["y"], attrs: {} },
      { op: "gelu_tanh", ins: ["gate"], outs: ["h"], attrs: {} },
      { op: "mul", ins: ["h", "ple"], outs: ["m"], attrs: {} },
      { op: "static_quantize", ins: ["m"], outs: ["q"], attrs: { scale: Math.fround(0.0371) } },
    ],
  };
  const ir = parse(graph);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, { x: row, gate: row, ple: row })).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...A8_I8,
    fuseGeluMulStaticQuantize: true,
  });
  assertEquals(outline(plan.steps), ["linear", "fused:geluMulStaticQuantize"]);
  assertEquals(plan.counts, builtInCounts({ linearPrologue: 1, geluMulStaticQuantize: 1 }));
});

// ------------------------------- the permutes a fused attention absorbs (perf-ledger F-3 / F-8)

type Operand = "q" | "k" | "v";

type AttentionPermuteOptions = {
  /** The operands that reach the attention through a permute, in node order (default q, k, v). */
  readonly permuted?: readonly Operand[];
  /** O leaves through the output permute (default true). */
  readonly outputPermute?: boolean;
  /** The q permute's dims (default [0, 2, 1, 3]; its input is shaped so the output is still q). */
  readonly qDims?: readonly number[];
  /** The permuted q gets a second consumer. */
  readonly qExtraConsumer?: boolean;
  /** The permuted q is a graph output. */
  readonly qGraphOutput?: boolean;
  /** An unrelated node between the last input permute and the attention. */
  readonly gapBefore?: boolean;
  /** An unrelated node between the first and the second input permute. */
  readonly gapBetween?: boolean;
  /** The additive mask input; `permuted` puts a permute in front of it, `first` / `last` in the run. */
  readonly mask?: "plain" | "first" | "last";
  /** Each token-major input is computed (a `neg`) instead of a graph input — internal storage. */
  readonly internalInputs?: boolean;
  /** Head count H (default 2 — H = 1 makes the permutes element-order-preserving). */
  readonly heads?: number;
  /** Head width D (default 8). */
  readonly d?: number;
  /** Key count N (default 6). */
  readonly n?: number;
  /** The output permute's dims (default [0, 2, 1, 3]; its output is shaped from them). */
  readonly oDims?: readonly number[];
  /** The attention's output gets a second consumer. */
  readonly oExtraConsumer?: boolean;
  /** The attention's output is a graph output too. */
  readonly oGraphOutput?: boolean;
  /** An unrelated node between the attention and the output permute. */
  readonly gapAfter?: boolean;
};

const ATTENTION_PERMUTE = { b: 1, s: 5, n: 6, d: 8 } as const;

/**
 * The Wan DiT form: token-major q / k / v `[B, S, H, D]` permuted to head-major, the attention, and
 * O permuted back — each piece switchable for the counterexamples.
 */
const attentionPermuteGraph = (options: AttentionPermuteOptions = {}): DeclarationJson => {
  const dtype = "f32";
  const { b, s } = ATTENTION_PERMUTE;
  const n = options.n ?? ATTENTION_PERMUTE.n;
  const d = options.d ?? ATTENTION_PERMUTE.d;
  const h = options.heads ?? 2;
  const permuted = options.permuted ?? ["q", "k", "v"];
  const outputPermute = options.outputPermute ?? true;
  const headMajor: Readonly<Record<Operand, readonly number[]>> = {
    q: [b, h, s, d],
    k: [b, h, n, d],
    v: [b, h, n, d],
  };
  const inputs: DeclarationJson["inputs"] = [];
  const values: DeclarationJson["values"] = {};
  const nodes: DeclarationJson["nodes"] = [];
  const permutes: DeclarationJson["nodes"] = [];
  const outputs: string[] = [];
  // The input of a permute with `dims` whose output is `out` (out[i] = in[dims[i]]).
  const permuteInput = (out: readonly number[], dims: readonly number[]): number[] => {
    const shape = new Array<number>(out.length);
    dims.forEach((axis, i) => {
      shape[axis] = out[i];
    });
    return shape;
  };
  const source = (name: string, shape: readonly number[]): string => {
    if (!options.internalInputs) {
      inputs.push({ name, dtype, shape: [...shape] });
      return name;
    }
    inputs.push({ name: `${name}_in`, dtype, shape: [...shape] });
    values[name] = { dtype, shape: [...shape] };
    nodes.push({ op: "neg", ins: [`${name}_in`], outs: [name], attrs: {} });
    return name;
  };
  for (const operand of ["q", "k", "v"] as const) {
    if (!permuted.includes(operand)) {
      inputs.push({ name: operand, dtype, shape: [...headMajor[operand]] });
      continue;
    }
    const dims = operand === "q" ? (options.qDims ?? [0, 2, 1, 3]) : [0, 2, 1, 3];
    const tokenMajor = source(`${operand}t`, permuteInput(headMajor[operand], dims));
    values[operand] = { dtype, shape: [...headMajor[operand]] };
    permutes.push({
      op: "permute",
      ins: [tokenMajor],
      outs: [operand],
      attrs: { dims: [...dims] },
    });
  }
  const ordered = permuted.map((operand) => {
    const found = permutes.find((node) => node.outs[0] === operand);
    if (found === undefined) throw new Error(`no permute for ${operand}`);
    return found;
  });
  const gap = (name: string): DeclarationJson["nodes"][number] => {
    inputs.push({ name: `${name}_in`, dtype, shape: [4] });
    values[name] = { dtype, shape: [4] };
    outputs.push(name);
    return { op: "neg", ins: [`${name}_in`], outs: [name], attrs: {} };
  };
  const attentionIns = ["q", "k", "v"];
  if (options.mask !== undefined) {
    if (options.mask === "plain") {
      inputs.push({ name: "mask", dtype, shape: [1, 1, s, n] });
    } else {
      inputs.push({ name: "mask_t", dtype, shape: [1, s, 1, n] });
      values.mask = { dtype, shape: [1, 1, s, n] };
      const node = {
        op: "permute",
        ins: ["mask_t"],
        outs: ["mask"],
        attrs: { dims: [0, 2, 1, 3] },
      };
      if (options.mask === "first") ordered.unshift(node);
      else ordered.push(node);
    }
    attentionIns.push("mask");
  }
  ordered.forEach((node, index) => {
    if (index === 1 && options.gapBetween) nodes.push(gap("gap_between"));
    nodes.push(node);
  });
  if (options.gapBefore) nodes.push(gap("gap_before"));
  values.o = { dtype, shape: [b, h, s, d] };
  nodes.push({ op: "attention", ins: attentionIns, outs: ["o"], attrs: { scale: 0.125 } });
  if (options.gapAfter) nodes.push(gap("gap_after"));
  if (outputPermute) {
    const oDims = options.oDims ?? [0, 2, 1, 3];
    const headMajorO = [b, h, s, d];
    values.ot = { dtype, shape: oDims.map((axis) => headMajorO[axis]) };
    nodes.push({ op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [...oDims] } });
    outputs.push("ot");
  } else {
    outputs.push("o");
  }
  if (options.oExtraConsumer) {
    values.o_neg = { dtype, shape: [b, h, s, d] };
    nodes.push({ op: "neg", ins: ["o"], outs: ["o_neg"], attrs: {} });
    outputs.push("o_neg");
  }
  if (options.oGraphOutput) outputs.push("o");
  if (options.qExtraConsumer) {
    values.q_neg = { dtype, shape: [...headMajor.q] };
    nodes.push({ op: "neg", ins: ["q"], outs: ["q_neg"], attrs: {} });
    outputs.push("q_neg");
  }
  if (options.qGraphOutput) outputs.push("q");
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [...new Set(outputs)],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * The attention seat the fusion context carries (as the executor passes `attentionCompute`,
 * `attentionScoreStorage` and the Session's device facts).
 */
type AttentionSeat = {
  readonly attentionCompute?: "f32" | "f16" | "a8";
  readonly attentionScoreStorage?: "f32" | "f16" | "none";
  readonly attentionDevice?: AttentionDeviceFacts;
};

/** The a8 attention seat — the only one on which the attention absorbs permutes. */
const A8_SEAT: AttentionSeat = { attentionCompute: "a8" };

const fuseAttentionPermutes = (
  options: AttentionPermuteOptions = {},
  seat: AttentionSeat = A8_SEAT,
): FusionPlan => {
  const graph = attentionPermuteGraph(options);
  const ir = parse(graph);
  return planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...seat,
  });
};

/** A node step's absorbed permutes (undefined for fused steps and plain nodes). */
const tokenMajorOf = (step: ExecStep) => (step.kind === "node" ? step.tokenMajor : undefined);

/** The attention's node step of a plan (exactly one). */
const attentionStepOf = (plan: FusionPlan, label: string) => {
  const found = plan.steps.filter(
    (step) => step.kind === "node" && step.plan.node.op === "attention",
  );
  assertEquals(found.length, 1, `${label}: one attention step`);
  const [step] = found;
  if (step.kind !== "node") throw new Error(`${label}: not a node step`);
  return step;
};

test("on an a8 attention seat a fused attention absorbs the [0, 2, 1, 3] permutes of q / k / v before it (the Wan DiT form): one node step on the token-major inputs, its output head-major as declared", () => {
  const { b, s, n, d } = ATTENTION_PERMUTE;
  const plan = fuseAttentionPermutes();
  assertEquals(outline(plan.steps), ["attention", "permute"]);
  const step = attentionStepOf(plan, "Wan form");
  assertEquals(step.tokenMajor, { q: true, k: true, v: true, o: false });
  assertEquals(step.plan.node.ins, ["qt", "kt", "vt"], "re-pointed to the permutes' inputs");
  assertEquals(step.plan.inputShapes, [
    [b, s, 2, d],
    [b, n, 2, d],
    [b, n, 2, d],
  ]);
  assertEquals(step.plan.outputs, [{ name: "o", shape: [b, 2, s, d], dtype: "f32" }]);
  assertEquals(step.aliasesInput, false);
  assertEquals(plan.counts, builtInCounts({ attentionPermute: 3 }));
});

test("only the a8 attention seat absorbs: on an f32, an f16 or an unset seat every permute stays a node", () => {
  for (const [label, seat] of [
    ["f32", { attentionCompute: "f32" }],
    ["f16", { attentionCompute: "f16" }],
    ["unset", {}],
  ] as const) {
    const plan = fuseAttentionPermutes({}, seat);
    assertEquals(
      outline(plan.steps),
      ["permute", "permute", "permute", "attention", "permute"],
      label,
    );
    assertEquals(
      plan.steps.map(tokenMajorOf),
      plan.steps.map(() => undefined),
      `${label}: no annotation`,
    );
    assertEquals(plan.counts, builtInCounts(), `${label}: counters`);
  }
});

test("on an a8 seat whose route stores S (the three a8 dispatches), the permute after the attention is not absorbed: it stays the next node, reading the attention's head-major output", () => {
  for (const [label, permuted] of [
    ["no input permute", []],
    ["the anima form (v only)", ["v"]],
    ["the Wan form (q / k / v)", ["q", "k", "v"]],
  ] as const) {
    const plan = fuseAttentionPermutes({ permuted });
    const last = plan.steps.at(-1);
    if (last?.kind !== "node") throw new Error(`${label}: the last step is not a node step`);
    assertEquals(last.plan.node.op, "permute", label);
    assertEquals(last.plan.node.ins, ["o"], `${label}: reads the attention's output`);
    assertEquals(last.tokenMajor, undefined, label);
    assertEquals(attentionStepOf(plan, label).plan.outputs[0].name, "o", label);
    assertEquals(plan.counts.attentionPermute, permuted.length, `${label}: counter`);
  }
});

test("the attention absorbs any subset of its input permutes that qualifies: each operand alone, the anima form (v) and two of three", () => {
  const cases: readonly (readonly [string, AttentionPermuteOptions, readonly string[]])[] = [
    ["q alone", { permuted: ["q"] }, ["qt", "k", "v"]],
    ["k alone", { permuted: ["k"] }, ["q", "kt", "v"]],
    ["v alone (the anima form)", { permuted: ["v"] }, ["q", "k", "vt"]],
    ["q and v (k from elsewhere)", { permuted: ["q", "v"] }, ["qt", "k", "vt"]],
    ["q / k / v without an output permute", { outputPermute: false }, ["qt", "kt", "vt"]],
    ["the permutes in another order", { permuted: ["v", "q", "k"] }, ["qt", "kt", "vt"]],
    ["with a mask (never re-pointed)", { mask: "plain" }, ["qt", "kt", "vt", "mask"]],
  ];
  for (const [label, options, ins] of cases) {
    const permuted = options.permuted ?? ["q", "k", "v"];
    const plan = fuseAttentionPermutes(options);
    assertEquals(
      outline(plan.steps),
      (options.outputPermute ?? true) ? ["attention", "permute"] : ["attention"],
      label,
    );
    const step = attentionStepOf(plan, label);
    assertEquals(
      step.tokenMajor,
      {
        q: permuted.includes("q"),
        k: permuted.includes("k"),
        v: permuted.includes("v"),
        o: false,
      },
      label,
    );
    assertEquals(step.plan.node.ins, ins, `${label}: ins`);
    assertEquals(plan.counts.attentionPermute, permuted.length, `${label}: counter`);
  }
});

test("the absorbed permutes keep the total-use accounting exact: each permute's input trades the permute's use for the attention's, and the absorbed values are never consumed or defined", () => {
  // The token-major q is also read by a residual add after the output permute (two uses before and
  // after), and O's permuted form feeds that add.
  const graph = attentionPermuteGraph();
  graph.values.res = { dtype: "f32", shape: [1, 5, 2, 8] };
  graph.nodes.push({ op: "add", ins: ["ot", "qt"], outs: ["res"], attrs: {} });
  graph.outputs = ["res"];
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  const ir = parse(graph);
  const uses = countUses(ir);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    rules: FUSION_RULES,
    useCounts: uses,
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...A8_SEAT,
  });
  assertEquals(outline(plan.steps), ["attention", "permute", "add"]);
  const consumed = new Map<string, number>();
  const defined = new Set<string>();
  for (const step of plan.steps) {
    const names = step.kind === "node" ? step.plan.node.ins : step.ins;
    for (const name of names) consumed.set(name, (consumed.get(name) ?? 0) + 1);
    if (step.kind === "node") for (const out of step.plan.outputs) defined.add(out.name);
    else defined.add(step.outputName);
  }
  for (const absorbed of ["q", "k", "v"]) {
    assertEquals(uses.get(absorbed), 1, `${absorbed}: the absorbed value had exactly one use`);
    assertEquals(consumed.get(absorbed), undefined, `${absorbed}: never consumed`);
    assertEquals(defined.has(absorbed), false, `${absorbed}: never defined`);
  }
  for (const [name, count] of uses) {
    if (["q", "k", "v"].includes(name)) continue;
    assertEquals(consumed.get(name), count, `total uses of '${name}'`);
  }
  assertEquals(consumed.get("qt"), 2, "qt: the attention and the residual add");
  assertEquals(defined.has("o"), true, "the attention step defines its own output");
});

test("attention permute counterexamples keep the permute that does not qualify as a plain node (and only that one)", () => {
  const cases: readonly (readonly [
    string,
    AttentionPermuteOptions,
    readonly string[],
    AttentionTokenMajor | undefined,
  ])[] = [
    [
      "q permuted by [0, 3, 2, 1]",
      { qDims: [0, 3, 2, 1] },
      ["permute", "attention", "permute"],
      { q: false, k: true, v: true, o: false },
    ],
    [
      "the permuted q has a second consumer",
      { qExtraConsumer: true },
      ["permute", "attention", "permute", "neg"],
      { q: false, k: true, v: true, o: false },
    ],
    [
      "the permuted q is a graph output",
      { qGraphOutput: true },
      ["permute", "attention", "permute"],
      { q: false, k: true, v: true, o: false },
    ],
    [
      "a node between the q permute and the rest of the run",
      { gapBetween: true },
      ["permute", "neg", "attention", "permute"],
      { q: false, k: true, v: true, o: false },
    ],
    [
      "a node between the input permutes and the attention",
      { gapBefore: true },
      ["permute", "permute", "permute", "neg", "attention", "permute"],
      undefined,
    ],
    [
      "a permute feeding the mask, first in the run",
      { mask: "first" },
      ["permute", "attention", "permute"],
      { q: true, k: true, v: true, o: false },
    ],
    [
      "a permute feeding the mask, last in the run (it breaks the run before it)",
      { mask: "last" },
      ["permute", "permute", "permute", "permute", "attention", "permute"],
      undefined,
    ],
  ];
  for (const [label, options, steps, tokenMajor] of cases) {
    const plan = fuseAttentionPermutes(options);
    assertEquals(outline(plan.steps), steps, `${label}: steps`);
    assertEquals(attentionStepOf(plan, label).tokenMajor, tokenMajor, label);
    assertEquals(
      plan.counts.attentionPermute,
      Object.values(tokenMajor ?? {}).filter(Boolean).length,
      `${label}: counter`,
    );
  }
});

test("a permute that keeps the element order (H = 1 on internal storage) is an alias, not absorbed; the same graph with H = 2 is", () => {
  const aliased = fuseAttentionPermutes({ internalInputs: true, heads: 1 });
  assertEquals(outline(aliased.steps), [
    "neg",
    "neg",
    "neg",
    "permute",
    "permute",
    "permute",
    "attention",
    "permute",
  ]);
  assertEquals(
    aliased.steps.map(tokenMajorOf),
    aliased.steps.map(() => undefined),
  );
  assertEquals(aliased.counts.attentionPermute, 0);
  const copied = fuseAttentionPermutes({ internalInputs: true });
  assertEquals(outline(copied.steps), ["neg", "neg", "neg", "attention", "permute"]);
  assertEquals(copied.counts.attentionPermute, 3);
});

test("nothing around a states-form attention or a non-f32 permute is absorbed", () => {
  const graph = attentionPermuteGraph();
  const ir = parse(graph);
  const nodes = planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes;
  const context = {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...A8_SEAT,
  };
  // The states form is another kernel family (and the window guard of every rule rejects state).
  const states = nodes.map((plan) =>
    plan.node.op === "attention"
      ? { ...plan, node: { ...plan.node, states: { k: "slot_k", v: "slot_v" } } }
      : plan,
  );
  const withStates = planFusions(states, context);
  assertEquals(outline(withStates.steps), [
    "permute",
    "permute",
    "permute",
    "attention",
    "permute",
  ]);
  assertEquals(withStates.counts.attentionPermute, 0, "states form");
  // A permute is a layout op of any dtype; the planned graph is f32, so the matcher's own check is
  // exercised on a plan with the q permute's dtypes replaced.
  for (const [label, replace] of [
    ["input i32", (plan: (typeof nodes)[number]) => ({ ...plan, inputDtypes: ["i32" as const] })],
    [
      "output i32",
      (plan: (typeof nodes)[number]) => ({
        ...plan,
        outputs: [{ ...plan.outputs[0], dtype: "i32" as const }],
      }),
    ],
  ] as const) {
    const replaced = nodes.map((plan, index) => (index === 0 ? replace(plan) : plan));
    const fused = planFusions(replaced, context);
    assertEquals(outline(fused.steps), ["permute", "attention", "permute"], label);
    assertEquals(
      attentionStepOf(fused, label).tokenMajor,
      { q: false, k: true, v: true, o: false },
      label,
    );
  }
});

// The output permute on the online route (perf-ledger F-3 A4): the online kernel's workgroup storage
// is 18,944 / 31,232 B (sg32) and 23,040 / 35,328 B (wg) at D 64 / 128.

/** Subgroups and 32 KiB of workgroup storage: the sg32 form at D 64 and 128. */
const SG32_DEVICE: AttentionDeviceFacts = {
  subgroups: true,
  maxComputeWorkgroupStorageSize: 32768,
};
/** No subgroups and 48 KiB: the wg form only. */
const WG_DEVICE: AttentionDeviceFacts = { subgroups: false, maxComputeWorkgroupStorageSize: 49152 };
/** No subgroups and 16 KiB: neither form fits, so the route is the three a8 dispatches. */
const NO_ONLINE_DEVICE: AttentionDeviceFacts = {
  subgroups: false,
  maxComputeWorkgroupStorageSize: 16384,
};

/** The online seat (`attentionCompute` "a8" × `attentionScoreStorage` "none") on `device`. */
const onlineSeat = (device: AttentionDeviceFacts): AttentionSeat => ({
  attentionCompute: "a8",
  attentionScoreStorage: "none",
  attentionDevice: device,
});

/** An online-eligible shape: D 64 or 128, N 8 (N % 4 == 0). */
const onlineShape = (d: 64 | 128): AttentionPermuteOptions => ({ d, n: 8 });

test("on the online route (a8 × 'none', D 64 / 128, any N, a device with the sg32 or only the wg form) the attention also absorbs the permute after it: one node step writes the token-major output and the permute has no step", () => {
  const { b, s } = ATTENTION_PERMUTE;
  for (const [label, d, device, reduce, n] of [
    ["D 64, sg32", 64, SG32_DEVICE, "sg32", 8],
    ["D 128, sg32", 128, SG32_DEVICE, "sg32", 8],
    ["D 64, wg only", 64, WG_DEVICE, "wg", 8],
    ["D 128, wg only", 128, WG_DEVICE, "wg", 8],
    // ADR 0132 addendum 2026-10-27: N % 4 != 0 takes the padded key axis (before it, N 6 fell
    // back and was a case of the next test)
    ["D 64, N 6 (N % 4 != 0), sg32", 64, SG32_DEVICE, "sg32", 6],
  ] as const) {
    // Precondition: the case takes the online kernel in the form its label names.
    assertEquals(
      attentionRoute({ attentionCompute: "a8", attentionScoreStorage: "none" }, d, n, device)
        .online,
      reduce,
      `${label}: route`,
    );
    const plan = fuseAttentionPermutes({ d, n }, onlineSeat(device));
    assertEquals(outline(plan.steps), ["attention"], label);
    const step = attentionStepOf(plan, label);
    assertEquals(step.tokenMajor, { q: true, k: true, v: true, o: true }, label);
    assertEquals(step.plan.node.ins, ["qt", "kt", "vt"], `${label}: ins`);
    assertEquals(step.plan.node.outs, ["ot"], `${label}: re-pointed to the permute's output`);
    assertEquals(
      step.plan.outputs,
      [{ name: "ot", shape: [b, s, 2, d], dtype: "f32" }],
      `${label}: token-major output`,
    );
    assertEquals(plan.counts, builtInCounts({ attentionPermute: 4 }), `${label}: counters`);
  }
});

test("on the online route the output permute is absorbed on its own too (no input permute), and with any subset of the input permutes", () => {
  for (const [label, permuted, ins] of [
    ["no input permute", [], ["q", "k", "v"]],
    ["the anima form (v only)", ["v"], ["q", "k", "vt"]],
    ["q and k", ["q", "k"], ["qt", "kt", "v"]],
  ] as const) {
    const plan = fuseAttentionPermutes({ ...onlineShape(128), permuted }, onlineSeat(SG32_DEVICE));
    assertEquals(outline(plan.steps), ["attention"], label);
    const step = attentionStepOf(plan, label);
    assertEquals(
      step.tokenMajor,
      {
        q: permuted.some((operand) => operand === "q"),
        k: permuted.some((operand) => operand === "k"),
        v: permuted.some((operand) => operand === "v"),
        o: true,
      },
      label,
    );
    assertEquals(step.plan.node.ins, ins, `${label}: ins`);
    assertEquals(step.plan.node.outs, ["ot"], `${label}: outs`);
    assertEquals(plan.counts.attentionPermute, permuted.length + 1, `${label}: counter`);
  }
});

test("off the online route the permute after the attention stays a node, and the input permutes are absorbed as before: the three a8 dispatches (S in f32 / f16), the f32 / f16 seats, and an online seat whose shape (D 96) or device (neither form) makes the route fall back", () => {
  const cases: readonly (readonly [string, AttentionPermuteOptions, AttentionSeat, boolean])[] = [
    [
      "a8, S in f32",
      onlineShape(64),
      { attentionCompute: "a8", attentionScoreStorage: "f32", attentionDevice: SG32_DEVICE },
      true,
    ],
    [
      "a8, S in f16 (s16)",
      onlineShape(128),
      { attentionCompute: "a8", attentionScoreStorage: "f16", attentionDevice: SG32_DEVICE },
      true,
    ],
    [
      "f32 seat",
      onlineShape(64),
      { attentionCompute: "f32", attentionScoreStorage: "f32", attentionDevice: SG32_DEVICE },
      false,
    ],
    [
      "f16 seat, S in f16",
      onlineShape(64),
      { attentionCompute: "f16", attentionScoreStorage: "f16", attentionDevice: SG32_DEVICE },
      false,
    ],
    ["online seat, D 96", { d: 96, n: 8 }, onlineSeat(SG32_DEVICE), true],
    [
      "online seat, a device with neither form",
      onlineShape(64),
      onlineSeat(NO_ONLINE_DEVICE),
      true,
    ],
  ];
  for (const [label, options, seat, inputsAbsorbed] of cases) {
    const plan = fuseAttentionPermutes(options, seat);
    assertEquals(
      outline(plan.steps),
      inputsAbsorbed
        ? ["attention", "permute"]
        : ["permute", "permute", "permute", "attention", "permute"],
      label,
    );
    const last = plan.steps.at(-1);
    if (last?.kind !== "node") throw new Error(`${label}: the last step is not a node step`);
    assertEquals(last.plan.node.ins, ["o"], `${label}: the permute reads the attention's output`);
    assertEquals(last.tokenMajor, undefined, label);
    const step = attentionStepOf(plan, label);
    assertEquals(
      step.tokenMajor,
      inputsAbsorbed ? { q: true, k: true, v: true, o: false } : undefined,
      label,
    );
    assertEquals(step.plan.outputs[0].name, "o", `${label}: head-major output`);
    assertEquals(plan.counts.attentionPermute, inputsAbsorbed ? 3 : 0, `${label}: counter`);
  }
});

test("on the online route an output permute that does not match exactly stays a node: other dims, a second use or a graph output of O, a node in between, and a permute that aliases O (H = 1)", () => {
  for (const [label, options, steps] of [
    ["the permute's dims are [0, 3, 2, 1]", { oDims: [0, 3, 2, 1] }, ["attention", "permute"]],
    ["O has a second consumer", { oExtraConsumer: true }, ["attention", "permute", "neg"]],
    ["O is a graph output", { oGraphOutput: true }, ["attention", "permute"]],
    [
      "a node between the attention and the permute",
      { gapAfter: true },
      ["attention", "neg", "permute"],
    ],
    [
      "H = 1: the permute keeps the element order of O (an alias)",
      { heads: 1 },
      ["attention", "permute"],
    ],
  ] as const) {
    const plan = fuseAttentionPermutes({ ...onlineShape(64), ...options }, onlineSeat(SG32_DEVICE));
    assertEquals(outline(plan.steps), steps, label);
    const step = attentionStepOf(plan, label);
    assertEquals(step.tokenMajor, { q: true, k: true, v: true, o: false }, label);
    assertEquals(step.plan.outputs[0].name, "o", `${label}: head-major output`);
    assertEquals(plan.counts.attentionPermute, 3, `${label}: counter`);
  }
});

test("on the online route an absorbable [0, 2, 1, 3] permute right after the attention that does not read O is not absorbed, and O's own permute later stays a node", () => {
  // The unrelated permute has O's head-major shape, so it is the head swap O's permute would be:
  // only the check that it reads O keeps the attention's output from being re-pointed to it.
  const graph = attentionPermuteGraph(onlineShape(64));
  const { b, s } = ATTENTION_PERMUTE;
  graph.inputs.push({ name: "x", dtype: "f32", shape: [b, 2, s, 64] });
  graph.values.xt = { dtype: "f32", shape: [b, s, 2, 64] };
  const attention = graph.nodes.findIndex((node) => node.op === "attention");
  graph.nodes.splice(attention + 1, 0, {
    op: "permute",
    ins: ["x"],
    outs: ["xt"],
    attrs: { dims: [0, 2, 1, 3] },
  });
  graph.outputs.push("xt");
  const ir = parse(graph);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    rules: FUSION_RULES,
    useCounts: countUses(ir),
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...onlineSeat(SG32_DEVICE),
  });
  assertEquals(outline(plan.steps), ["attention", "permute", "permute"]);
  const step = attentionStepOf(plan, "unrelated permute");
  assertEquals(step.tokenMajor, { q: true, k: true, v: true, o: false });
  assertEquals(step.plan.node.outs, ["o"], "head-major output");
  assertEquals(
    plan.steps.slice(1).map((next) => (next.kind === "node" ? next.plan.node.ins : [])),
    [["x"], ["o"]],
    "both permutes stay nodes on their own inputs",
  );
  assertEquals(plan.counts.attentionPermute, 3);
});

test("the absorbed output permute keeps the total-use accounting exact: O is never consumed or defined, and the attention step defines the permute's output", () => {
  const graph = attentionPermuteGraph(onlineShape(64));
  graph.values.res = { dtype: "f32", shape: [1, 5, 2, 64] };
  graph.nodes.push({ op: "add", ins: ["ot", "qt"], outs: ["res"], attrs: {} });
  graph.outputs = ["res"];
  graph.requires.ops = [...new Set(graph.nodes.map((node) => node.op))];
  const ir = parse(graph);
  const uses = countUses(ir);
  const plan = planFusions(planGraph(ir, bindSymbols(ir, adalnInputs(graph))).nodes, {
    rules: FUSION_RULES,
    useCounts: uses,
    outputNames: new Set(ir.outputs),
    limits: TEST_LIMITS,
    ...onlineSeat(SG32_DEVICE),
  });
  assertEquals(outline(plan.steps), ["attention", "add"]);
  const consumed = new Map<string, number>();
  const defined = new Set<string>();
  for (const step of plan.steps) {
    const names = step.kind === "node" ? step.plan.node.ins : step.ins;
    for (const name of names) consumed.set(name, (consumed.get(name) ?? 0) + 1);
    if (step.kind === "node") for (const out of step.plan.outputs) defined.add(out.name);
    else defined.add(step.outputName);
  }
  for (const absorbed of ["q", "k", "v", "o"]) {
    assertEquals(uses.get(absorbed), 1, `${absorbed}: the absorbed value had exactly one use`);
    assertEquals(consumed.get(absorbed), undefined, `${absorbed}: never consumed`);
    assertEquals(defined.has(absorbed), false, `${absorbed}: never defined`);
  }
  for (const [name, count] of uses) {
    if (["q", "k", "v", "o"].includes(name)) continue;
    assertEquals(consumed.get(name), count, `total uses of '${name}'`);
  }
  assertEquals(defined.has("ot"), true, "the attention step defines the permute's output");
});

test("on a8 × 'none' the output permute's decision needs the device facts: without attentionDevice it fails loudly, and without an output permute to decide nothing reads them", () => {
  const seat: AttentionSeat = { attentionCompute: "a8", attentionScoreStorage: "none" };
  assertThrows(
    () => fuseAttentionPermutes(onlineShape(64), seat),
    ExecutionError,
    "attentionScoreStorage 'none' needs attentionDevice",
  );
  const plan = fuseAttentionPermutes({ ...onlineShape(64), outputPermute: false }, seat);
  assertEquals(attentionStepOf(plan, "no output permute").tokenMajor, {
    q: true,
    k: true,
    v: true,
    o: false,
  });
});

test("no fusion rule's head and no prologue op is permute or attention (the rules and the linear prologue are tried first at each index, so the order cannot change the result)", () => {
  const heads = new Set([...FUSION_RULES.flatMap((rule) => rule.heads), ...LINEAR_PROLOGUE_OPS]);
  assertEquals(
    ["permute", "attention"].filter((op) => heads.has(op)),
    [],
  );
});

// ------------------------------- rms_norm → pair RoPE (rmsNormRope — perf-ledger F-3 stage 3)

const RMS_NORM_ROPE_SEQUENCE = 5;

type RmsNormRopeContext = {
  readonly rmsNormReduce?: "workgroup" | "subgroup32";
  readonly fuseRmsNormAdd?: boolean;
};

/** The planned nodes of {@link rmsNormRopeGraph} and the fusion context the executor would pass. */
const planRmsNormRope = (graph: DeclarationJson, context: RmsNormRopeContext = {}) => {
  const ir = parse(graph);
  const nodes = planGraph(
    ir,
    bindSymbols(ir, rmsNormRopeInputShapes(graph, RMS_NORM_ROPE_SEQUENCE)),
  ).nodes;
  return {
    nodes,
    context: {
      rules: FUSION_RULES,
      useCounts: countUses(ir),
      outputNames: new Set(ir.outputs),
      limits: TEST_LIMITS,
      ...context,
    },
  };
};

const fuseRmsNormRope = (
  options: RmsNormRopeGraphOptions = {},
  context: RmsNormRopeContext = {},
): FusionPlan => {
  const planned = planRmsNormRope(rmsNormRopeGraph(options), context);
  return planFusions(planned.nodes, planned.context);
};

/**
 * The step list is a valid execution order and its release bookkeeping is exact: every value a
 * step reads is a graph input or the output of an earlier step, and every value that exists (an
 * input or a step's output) is read by the steps exactly as often as the graph reads it.
 */
const assertStepOrder = (plan: FusionPlan, graph: DeclarationJson, label: string): void => {
  const uses = countUses(parse(graph));
  const defined = new Set(graph.inputs.map((spec) => spec.name));
  const reads = new Map<string, number>();
  for (const step of plan.steps) {
    const ins = step.kind === "fused" ? step.ins : step.plan.node.ins;
    for (const name of ins) {
      assertEquals(defined.has(name), true, `${label}: '${name}' is read before it is defined`);
      reads.set(name, (reads.get(name) ?? 0) + 1);
    }
    const outs = step.kind === "fused" ? [step.outputName] : step.plan.outputs.map((o) => o.name);
    for (const name of outs) defined.add(name);
  }
  for (const name of defined) {
    assertEquals(reads.get(name) ?? 0, uses.get(name) ?? 0, `${label}: uses of '${name}'`);
  }
};

test("rmsNormRope folds the Wan DiT's interleaved q and k prologues into one step each: the deferred nodes follow q's step and are scanned again", () => {
  // Wan2.2 (24 × 128), Wan2.1 (12 × 128), the smallest 256-wide row, a narrower head.
  for (const [heads, headDim] of [
    [24, 128],
    [12, 128],
    [2, 128],
    [3, 64],
  ] as const) {
    const label = `H=${heads} D=${headDim}`;
    const dim = heads * headDim;
    const graph = rmsNormRopeGraph({ heads, headDim });
    const plan = fuseRmsNormRope({ heads, headDim });
    // q's window holds k's rms_norm and reshape and v's reshape; deferred behind q's step, k's
    // rms_norm heads k's window, and v's reshape (deferred again) is the only plain node left.
    assertEquals(
      outline(plan.steps),
      ["fused:rmsNormRope", "fused:rmsNormRope", "reshape", "neg"],
      label,
    );
    const v = plan.steps[2];
    assertEquals(v.kind === "node" && v.plan.outputs[0].name, "v4", `${label}: v's reshape`);
    assertEquals(v.kind === "node" && v.aliasesInput, true, `${label}: still an alias`);
    for (const [at, p] of [
      [0, "q"],
      [1, "k"],
    ] as const) {
      const step = fusedAt(plan, at);
      assertEquals(step.ins, [`${p}_x`, `${p}_w`, "cos", "sin"], `${label} ${p}: total uses`);
      assertEquals(step.binds, [`${p}_x`, `${p}_w`, "cos", "sin"], `${label} ${p}: binds`);
      assertEquals(step.nodeCount, 11, `${label} ${p}: rms_norm + 2 reshapes + 8 RoPE nodes`);
      assertEquals(step.outputName, `${p}_y`, label);
      assertEquals(step.outputShape, [1, RMS_NORM_ROPE_SEQUENCE, heads, headDim], label);
      assertEquals(step.temps, [], label);
      assertEquals(step.dispatches.length, 1, label);
      const [dispatch] = step.dispatches;
      assertEquals(dispatch.key, RMS_NORM_ROPE_KEY, label);
      assertEquals(dispatch.wgsl(), RMS_NORM_ROPE_WGSL, label);
      assertEquals(
        [...dispatch.params],
        [...rmsNormRopeParams(RMS_NORM_ROPE_SEQUENCE, dim, 1e-6, headDim)],
        `${label} ${p}: params {rows, dim, eps, head_dim}`,
      );
      // One workgroup per token row (the plain rms_norm's geometry), operands in bind order.
      assertEquals(dispatch.workgroups, {
        kind: "gridStride",
        items: RMS_NORM_ROPE_SEQUENCE,
        size: 1,
      });
      assertEquals(dispatch.operands, undefined, label);
    }
    assertEquals(plan.counts, builtInCounts({ rmsNormRope: 2 }), label);
    assertStepOrder(plan, graph, label);
  }
  // One chain alone folds the same way (no deferred node).
  const single = fuseRmsNormRope({ form: "single" });
  assertEquals(outline(single.steps), ["fused:rmsNormRope"]);
  assertStepOrder(single, rmsNormRopeGraph({ form: "single" }), "single");
});

test("rmsNormRope counterexamples (bias / per-head norm / half split / interposed alias / values used outside the window or as graph outputs / a foreign op in the window / a table made inside the window) leave the nodes, and the pair RoPE folds alone", () => {
  const cases: readonly (readonly [string, RmsNormRopeGraphOptions])[] = [
    ["rms_norm with a bias", { bias: true }],
    // A row of 256 per head (the 256-wide form), so only the axis differs.
    ["rms_norm over each head", { perHead: true, headDim: 256 }],
    ["an interposed alias of the normalised value", { interpose: true }],
    ["the normalised value used outside the window", { extraConsumer: "normalised" }],
    ["its reshape alias used outside the window", { extraConsumer: "view" }],
    ["the pair view used outside the window", { extraConsumer: "pairs" }],
    ["the normalised value as a graph output", { graphOutput: "normalised" }],
    ["its reshape alias as a graph output", { graphOutput: "view" }],
    ["the pair view as a graph output", { graphOutput: "pairs" }],
    ["a neg between the reshapes", { foreignGap: true }],
    // The chain reads a node it would defer, so the fused step cannot move ahead of it.
    ["cos reshaped inside the window", { tableInWindow: true }],
  ];
  for (const [label, options] of cases) {
    const graph = rmsNormRopeGraph({ form: "single", ...options });
    const plan = fuseRmsNormRope({ form: "single", ...options });
    assertEquals(plan.counts.rmsNormRope, 0, `${label}: rmsNormRope counter`);
    assertEquals(
      plan.steps.flatMap((step) => (step.kind === "fused" ? [step.dispatches[0].key] : [])),
      [ROPE_PAIR_BSHD_KEY],
      `${label}: the pair RoPE alone (${outline(plan.steps).join(",")})`,
    );
    assertStepOrder(plan, graph, label);
  }
  // A half-split RoPE is not the pair form: the half split folds alone.
  const half = fuseRmsNormRope({ form: "single", halfSplit: true });
  assertEquals(half.counts.rmsNormRope, 0, "half split");
  assertEquals(
    half.steps.flatMap((step) => (step.kind === "fused" ? [step.dispatches[0].key] : [])),
    [ROPE_BSHD_KEY],
  );
  // An odd head width (H·D = 258 still pairs up, but a pair would straddle two heads): neither the
  // pair RoPE nor this rule folds.
  const odd = fuseRmsNormRope({ form: "single", heads: 2, headDim: 129 });
  assertEquals([odd.counts.rmsNormRope, odd.counts.rope], [0, 0], "odd head_dim");
});

test("rmsNormRope folds only where the plain path runs the 256-wide workgroup rms_norm: the subgroup reduction and a row of at most 128 keep the rms_norm, and the pair RoPEs fold alone", () => {
  const subgroup = fuseRmsNormRope({}, { rmsNormReduce: "subgroup32" });
  assertEquals(
    [subgroup.counts.rmsNormRope, subgroup.counts.rope],
    [0, 2],
    "rmsNormReduce subgroup32",
  );
  // Nothing defers, so the node order is the declaration order.
  assertEquals(outline(subgroup.steps), [
    "rms_norm",
    "rms_norm",
    "reshape",
    "reshape",
    "reshape",
    "reshape",
    "fused:rope",
    "reshape",
    "fused:rope",
    "neg",
  ]);
  const workgroup = fuseRmsNormRope({}, { rmsNormReduce: "workgroup" });
  assertEquals(workgroup.counts.rmsNormRope, 2, "rmsNormReduce workgroup (the default, explicit)");
  for (const [heads, headDim] of [
    [1, 128],
    [2, 64],
  ] as const) {
    const narrow = fuseRmsNormRope({ heads, headDim });
    assertEquals(
      [narrow.counts.rmsNormRope, narrow.counts.rope],
      [0, 2],
      `H=${heads} D=${headDim}: the 128-wide rms_norm`,
    );
  }
});

test("rmsNormRope rejects a non-f32 rms_norm or reshape in its chain", () => {
  // The contracts run these nodes on f32 only here, so the matcher's own check is exercised on a
  // plan with one node's dtypes replaced.
  const { nodes, context } = planRmsNormRope(rmsNormRopeGraph({ form: "single" }));
  const at = (op: string, out: string): number =>
    nodes.findIndex((plan) => plan.node.op === op && plan.outputs[0].name === out);
  const norm = at("rms_norm", "q_n");
  const view = at("reshape", "q_v");
  const pairs = at("reshape", "q_pairs");
  const i32Output = (plan: NodePlan): NodePlan => ({
    ...plan,
    outputs: [{ ...plan.outputs[0], dtype: "i32" }],
  });
  const i32Input = (plan: NodePlan): NodePlan => ({ ...plan, inputDtypes: ["i32", "f32"] });
  for (const [label, index, replace] of [
    ["rms_norm input i32", norm, i32Input],
    ["rms_norm output i32", norm, i32Output],
    ["first reshape output i32", view, i32Output],
    ["pair reshape output i32", pairs, i32Output],
  ] as const) {
    const replaced = nodes.map((plan, position) => (position === index ? replace(plan) : plan));
    assertEquals(planFusions(replaced, context).counts.rmsNormRope, 0, label);
  }
  assertEquals(planFusions(nodes, context).counts.rmsNormRope, 1, "the f32 control");
});

test("rmsNormAdd and rmsNormRope share the head rms_norm but never accept the same index", () => {
  // Both forms in one graph (with rmsNormAdd opted in), and a normalised value that feeds both an
  // add and the RoPE's reshape. At every index at most one rule of the table accepts, so their
  // order in FUSION_RULES cannot change the result.
  const both = rmsNormRopeGraph({ form: "single" });
  both.inputs.push(
    { name: "a_x", dtype: "f32", shape: [1, "S", 256] },
    { name: "a_w", dtype: "f32", shape: [256] },
    { name: "a_r", dtype: "f32", shape: [1, "S", 256] },
  );
  both.values.a_n = { dtype: "f32", shape: [1, "S", 256] };
  both.values.a_y = { dtype: "f32", shape: [1, "S", 256] };
  both.nodes.unshift(
    { op: "rms_norm", ins: ["a_x", "a_w"], outs: ["a_n"], attrs: { eps: 1e-6 } },
    { op: "add", ins: ["a_n", "a_r"], outs: ["a_y"], attrs: {} },
  );
  both.outputs.push("a_y");
  const shared = rmsNormRopeGraph({ form: "single" });
  shared.inputs.push({ name: "a_r", dtype: "f32", shape: [1, "S", 256] });
  shared.values.a_y = { dtype: "f32", shape: [1, "S", 256] };
  // The add right after the rms_norm (rmsNormAdd's adjacency), the reshape after it.
  shared.nodes.splice(1, 0, { op: "add", ins: ["q_n", "a_r"], outs: ["a_y"], attrs: {} });
  shared.outputs.push("a_y");
  for (const [label, graph, expected] of [
    ["both forms", both, { rmsNormAdd: 1, rmsNormRope: 1, rope: 0 }],
    ["one normalised value, both consumers", shared, { rmsNormAdd: 0, rmsNormRope: 0, rope: 1 }],
  ] as const) {
    graph.requires.ops = [...new Set(graph.nodes.map((step) => step.op))].sort();
    const { nodes, context } = planRmsNormRope(graph, { fuseRmsNormAdd: true });
    const scan = {
      ...context,
      packedValues: new Map<string, number>(),
      matchContext: fusionMatchContext(context),
    };
    for (let index = 0; index < nodes.length; index += 1) {
      const accepting = FUSION_RULES.filter(
        (rule) => rule.apply(nodes, index, scan) !== undefined,
      ).map((rule) => rule.name);
      assertEquals(accepting.length <= 1, true, `${label} @${index}: ${accepting.join(", ")}`);
    }
    const { counts } = planFusions(nodes, context);
    assertEquals(
      { rmsNormAdd: counts.rmsNormAdd, rmsNormRope: counts.rmsNormRope, rope: counts.rope },
      expected,
      label,
    );
  }
});
