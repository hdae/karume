/**
 * A generated fusion rule: a run of consecutive f32 elementwise nodes folded into one dispatch,
 * whatever the ops and their wiring (`models/ew_chain`). Written the way a package outside the
 * runtime writes a rule, against `@karume/runtime/kernel` alone
 * (packages/models/tests/external_fusion_rule_test.ts pins that import list). Adopted on
 * 2026-11-12 (perf-ledger F-13): the MOSS-Image pipeline passes it by default to the Sessions of its
 * image graphs (`./fusion-rules.ts`); no other family passes it.
 *
 * What it takes: the longest run of nodes that starts at the head, whose ops are scalar-free f32
 * unary / binary elementwise ops ({@link ARITY}), in which every node writes the shape of the last
 * one and every value but the last is read only inside the run. The external inputs (at most
 * {@link MAX_INPUTS}) may broadcast to that shape. The test bed is the MOSS-Image DiT, where the
 * runs are passes over whole activations (a nine-node gate over `[S, 8192]`, the `mul → add` tails
 * of the modulation, the gated residual and the rotary embedding), so the saving is the bytes the
 * unfused plan moves through buffers between the nodes.
 *
 * Bit identity with the unfused plan (class E): each node's value is the plain elementwise kernel's
 * own text (`elementwiseValueWgsl`, in the node's operand order), and **every** value the unfused
 * plan stores — every node but the last — is staged through workgroup memory by the rounding
 * barrier (`stagedLoopWgsl`), one round per dependency level, so the compiler sees each node's
 * expression alone, as it does in the plain kernel. Inputs are bound as buffers, never baked. That
 * the barrier holds is measured per backend, not guaranteed by WGSL: the fused-against-unfused check
 * is packages/e2e/tests/e2e_ew_chain_rule_test.ts.
 *
 * It is tried before the built-in rules (every external rule is), so it takes a window a built-in
 * rule with an elementwise head would have taken. The one form it leaves alone is the exact
 * `sigmoid(x) → mul(x, ·)` pair (the built-in silu rule's window); a longer run that contains the
 * pair is taken whole.
 */

import {
  allF32,
  defineRule,
  type ElementwiseValueOp,
  elementwiseValueWgsl,
  type FusionMatch,
  type FusionMatchContext,
  type FusionRule,
  type NodePlan,
  sameShape,
  stagedF32,
  stagedLoopWgsl,
} from "@karume/runtime/kernel";

/** The rule's name, and its counter in `lastRunFusions`. */
export const EW_CHAIN_RULE_NAME = "models/ew_chain";

const WORKGROUP_SIZE = 128;

/**
 * The longest run taken. Each node but the last is one staged value of {@link WORKGROUP_SIZE}
 * words, so 15 of them take 7,680 bytes of the 16,384 bytes of workgroup storage the core limit
 * grants.
 */
const MAX_CHAIN = 16;

/** The external inputs of a run: 8 storage bindings (the core limit) minus the output. */
const MAX_INPUTS = 7;

const U32_MAX = 0xffff_ffff;

/**
 * The ops a run is made of: the value ops of the entry that take no scalar attr (`clamp`,
 * `clamp_min` and `leaky_relu` carry one in the params, which this rule does not lay out).
 */
type ChainOp = Exclude<ElementwiseValueOp, "clamp" | "clamp_min" | "leaky_relu">;

/** The operand count of each op of a run. */
const ARITY = {
  neg: 1,
  abs: 1,
  exp: 1,
  log: 1,
  log1p: 1,
  sqrt: 1,
  sin: 1,
  tanh: 1,
  sigmoid: 1,
  relu: 1,
  gelu: 1,
  gelu_tanh: 1,
  add: 2,
  sub: 2,
  mul: 2,
  div: 2,
} as const satisfies Record<ChainOp, 1 | 2>;

const isChainOp = (op: string): op is ChainOp => Object.hasOwn(ARITY, op);

/** Where a node's operand comes from: an external input (a bind) or an earlier node of the run. */
type Operand =
  | { readonly kind: "input"; readonly index: number }
  | { readonly kind: "node"; readonly index: number };

type FormNode = { readonly op: ChainOp; readonly operands: readonly Operand[] };

/**
 * How an external input is read for output element `i`: `full` = it has the output's shape (element
 * `i`), `scalar` = it has one element, `broadcast` = it has the output's length on `axes` and 1
 * elsewhere (right-aligned, as the plain kernel broadcasts).
 */
type InputClass =
  | { readonly kind: "full" }
  | { readonly kind: "scalar" }
  | { readonly kind: "broadcast"; readonly axes: readonly boolean[] };

/**
 * Everything the WGSL is generated from — and nothing else: no shape value enters it, so one form
 * serves every sequence length. MUST: the key is derived from the whole form ({@link formKey}).
 */
type Form = {
  /** The output rank; it matters only to a `broadcast` input (its `axes` has this length). */
  readonly rank: number;
  readonly inputs: readonly InputClass[];
  readonly nodes: readonly FormNode[];
};

type EwChainMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly form: Form;
  /** The output shape, and each input's shape right-aligned to its rank. */
  readonly shape: readonly number[];
  readonly inputShapes: readonly (readonly number[])[];
};

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const classKey = (input: InputClass): string =>
  input.kind === "full"
    ? "f"
    : input.kind === "scalar"
      ? "s"
      : `b${input.axes.map((taken) => (taken ? "1" : "0")).join("")}`;

const operandKey = (operand: Operand): string =>
  operand.kind === "input" ? `x${operand.index}` : `t${operand.index}`;

/**
 * MUST: every part of the form is in the key (each changes the WGSL); the key is under
 * `ext:models:`. The rank is in it through the `broadcast` classes, the only place it is baked.
 */
const formKey = (form: Form): string => {
  const inputs = form.inputs.map(classKey).join(".");
  const nodes = form.nodes
    .map((node) => [node.op, ...node.operands.map(operandKey)].join("."))
    .join("-");
  return `ext:models:ew_chain:v1:${inputs}:${nodes}:f32:wg${WORKGROUP_SIZE}`;
};

/** The axes some `broadcast` input indexes, ascending: the ones the params carry. */
const indexedAxes = (form: Form): readonly number[] =>
  Array.from({ length: form.rank }, (_, axis) => axis).filter((axis) =>
    form.inputs.some((input) => input.kind === "broadcast" && input.axes[axis]),
  );

/**
 * The fields of the uniform `Params`, in order: the element count `n`; per indexed axis `d` the
 * elements one step of it spans (`o<d>`) and its length (`d<d>`); per `broadcast` input `k` and
 * axis `d` of it the input's stride (`s<k>_<d>`); then `reserved<j>` up to a multiple of four
 * fields (the runtime allocates params in 16-byte units).
 *
 * MUST: {@link wgsl} and {@link paramsOf} both take the layout from here, so the struct and the
 * words cannot disagree.
 */
const paramFields = (form: Form): readonly string[] => {
  const fields = ["n"];
  for (const axis of indexedAxes(form)) fields.push(`o${axis}`, `d${axis}`);
  form.inputs.forEach((input, k) => {
    if (input.kind !== "broadcast") return;
    input.axes.forEach((taken, axis) => {
      if (taken) fields.push(`s${k}_${axis}`);
    });
  });
  for (let pad = 0; fields.length % 4 !== 0; pad += 1) fields.push(`reserved${pad}`);
  return fields;
};

const paramsOf = (match: EwChainMatch): Uint32Array<ArrayBuffer> => {
  const { form, shape, inputShapes } = match;
  // The elements one step of an axis spans in a contiguous buffer of `dims`.
  const span = (dims: readonly number[], axis: number): number => elements(dims.slice(axis + 1));
  const values = new Map<string, number>([["n", elements(shape)]]);
  for (const axis of indexedAxes(form)) {
    values.set(`o${axis}`, span(shape, axis));
    values.set(`d${axis}`, shape[axis]);
  }
  form.inputs.forEach((input, k) => {
    if (input.kind !== "broadcast") return;
    input.axes.forEach((taken, axis) => {
      if (taken) values.set(`s${k}_${axis}`, span(inputShapes[k], axis));
    });
  });
  return Uint32Array.from(paramFields(form), (field) => values.get(field) ?? 0);
};

/** The WGSL text reading input `k` for output element `i`. */
const inputWgsl = (input: InputClass, k: number): string => {
  if (input.kind === "full") return `x${k}[i]`;
  if (input.kind === "scalar") return `x${k}[0]`;
  const terms = input.axes.flatMap((taken, axis) =>
    taken ? [`((i / params.o${axis}) % params.d${axis}) * params.s${k}_${axis}`] : [],
  );
  return `x${k}[${terms.join(" + ")}]`;
};

/** The dependency level of each node: 1 + the deepest node it reads (an input is level 0). */
const levels = (nodes: readonly FormNode[]): readonly number[] => {
  const level: number[] = [];
  for (const node of nodes) {
    const below = node.operands.map((operand) =>
      operand.kind === "node" ? level[operand.index] : 0,
    );
    level.push(1 + Math.max(...below));
  }
  return level;
};

const wgsl = (form: Form): string => {
  const values = form.nodes.map((node) =>
    elementwiseValueWgsl(
      node.op,
      node.operands.map((operand) =>
        operand.kind === "input"
          ? inputWgsl(form.inputs[operand.index], operand.index)
          : stagedF32(`t${operand.index}`),
      ),
    ),
  );
  // Module-scope declarations may come in any order; sorting the distinct bodies fixes it.
  const helpers = [...new Set(values.flatMap((value) => value.helpers))].sort();
  const last = form.nodes.length - 1;
  const level = levels(form.nodes);
  const rounds = Array.from({ length: Math.max(...level.slice(0, last)) }, (_, round) =>
    values.flatMap((value, index) =>
      index < last && level[index] === round + 1 ? [{ name: `t${index}`, value: value.value }] : [],
    ),
  );
  const bindings = form.inputs.map(
    (_, k) => `@group(0) @binding(${k + 1}) var<storage, read> x${k}: array<f32>;`,
  );
  return `// karume rule models/ew_chain (${formKey(form)})
struct Params { ${paramFields(form)
    .map((field) => `${field}: u32`)
    .join(", ")} }
@group(0) @binding(0) var<uniform> params: Params;
${bindings.join("\n")}
@group(0) @binding(${form.inputs.length + 1}) var<storage, read_write> out: array<f32>;

${helpers.join("\n\n")}

${stagedLoopWgsl({
  workgroupSize: WORKGROUP_SIZE,
  count: "params.n",
  rounds,
  finish: [`out[i] = ${values[last].value};`],
})}`;
};

/**
 * How an input of `inputShape` is read against the output `shape`, and its shape right-aligned to
 * the output's rank; `undefined` when it does not broadcast to it.
 */
const classify = (
  inputShape: readonly number[],
  shape: readonly number[],
): { readonly input: InputClass; readonly padded: readonly number[] } | undefined => {
  if (inputShape.length > shape.length) return undefined;
  const padded = [...Array<number>(shape.length - inputShape.length).fill(1), ...inputShape];
  if (padded.some((dim, axis) => dim !== 1 && dim !== shape[axis])) return undefined;
  if (sameShape(padded, shape)) return { input: { kind: "full" }, padded };
  if (elements(padded) === 1) return { input: { kind: "scalar" }, padded };
  return { input: { kind: "broadcast", axes: padded.map((dim) => dim !== 1) }, padded };
};

/**
 * The run `chain` as a match, or `undefined` when it is not one the rule takes.
 *
 * The privacy test repeats what `defineRule` applies for `internals: "stayInChain"` (every use of
 * every value but the last is an operand of a node of the run, there is at least one, and none is a
 * graph output). It is here only to choose the run's length: `defineRule` declines a failing match
 * outright, and a shorter run may pass. The guard is `defineRule`'s.
 */
const matchChain = (
  chain: readonly NodePlan[],
  context: FusionMatchContext,
): EwChainMatch | undefined => {
  const lastIndex = chain.length - 1;
  const shape = chain[lastIndex].outputs[0].shape;
  const count = elements(shape);
  if (count < 1 || count > U32_MAX || !allF32(chain)) return undefined;
  if (chain.some((step) => !sameShape(step.outputs[0].shape, shape))) return undefined;

  const reads = new Map<string, number>();
  for (const step of chain) {
    for (const name of step.node.ins) reads.set(name, (reads.get(name) ?? 0) + 1);
  }
  const internalsStay = chain.slice(0, lastIndex).every((step) => {
    const { name } = step.outputs[0];
    const uses = context.useCounts.get(name) ?? 0;
    return uses > 0 && uses === reads.get(name) && !context.outputNames.has(name);
  });
  if (!internalsStay) return undefined;

  const produced = new Map<string, number>();
  const binds: string[] = [];
  const inputs: InputClass[] = [];
  const inputShapes: (readonly number[])[] = [];
  const nodes: FormNode[] = [];
  for (const [index, step] of chain.entries()) {
    const op = step.node.op;
    if (!isChainOp(op)) return undefined;
    const operands: Operand[] = [];
    for (const [slot, name] of step.node.ins.entries()) {
      const node = produced.get(name);
      if (node !== undefined) {
        operands.push({ kind: "node", index: node });
        continue;
      }
      let bind = binds.indexOf(name);
      if (bind < 0) {
        const classified = classify(step.inputShapes[slot], shape);
        if (classified === undefined) return undefined;
        bind = binds.push(name) - 1;
        inputs.push(classified.input);
        inputShapes.push(classified.padded);
      }
      operands.push({ kind: "input", index: bind });
    }
    nodes.push({ op, operands });
    produced.set(step.outputs[0].name, index);
  }
  if (binds.length > MAX_INPUTS) return undefined;

  // The built-in silu rule's window, left to it: sigmoid(x) → mul of x and the sigmoid. The
  // sigmoid's operand is the first input (index 0), so the pair has exactly one input.
  const [first, second] = nodes;
  if (
    nodes.length === 2 &&
    first.op === "sigmoid" &&
    second.op === "mul" &&
    binds.length === 1 &&
    second.operands.some((operand) => operand.kind === "node")
  ) {
    return undefined;
  }

  return {
    window: chain,
    chain,
    binds,
    form: { rank: shape.length, inputs, nodes },
    shape,
    inputShapes,
  };
};

/**
 * The longest run of at least two nodes that starts at `index` and {@link matchChain} takes.
 *
 * A run stops at the first node that is not a scalar-free unary / binary elementwise op with one
 * output and the op's operand count.
 */
const match = (
  nodes: readonly NodePlan[],
  index: number,
  context: FusionMatchContext,
): EwChainMatch | undefined => {
  const run: NodePlan[] = [];
  for (let at = index; at < nodes.length && run.length < MAX_CHAIN; at += 1) {
    const step = nodes[at];
    const op = step.node.op;
    if (!isChainOp(op) || step.outputs.length !== 1 || step.node.ins.length !== ARITY[op]) break;
    run.push(step);
  }
  for (let length = run.length; length >= 2; length -= 1) {
    const matched = matchChain(run.slice(0, length), context);
    if (matched !== undefined) return matched;
  }
  return undefined;
};

/**
 * The rule (class E). Both module-scope calls are marked pure for the barrel's tree-shaking: a call
 * a bundler cannot prove pure (`Object.keys` included) would keep the module's import of
 * `@karume/runtime/kernel` in every bundle of `@karume/models`.
 */
export const EW_CHAIN_RULE: FusionRule<typeof EW_CHAIN_RULE_NAME> = /* @__PURE__ */ defineRule<
  EwChainMatch,
  typeof EW_CHAIN_RULE_NAME
>({
  name: EW_CHAIN_RULE_NAME,
  heads: /* @__PURE__ */ Object.keys(ARITY),
  contract: "E",
  internals: "stayInChain",
  match,
  build: (matched) => {
    const product = matched.chain[matched.chain.length - 1].outputs[0];
    return {
      binds: matched.binds,
      outputName: product.name,
      outputShape: product.shape,
      temps: [],
      dispatches: [
        {
          key: formKey(matched.form),
          wgsl: () => wgsl(matched.form),
          params: paramsOf(matched),
          workgroups: {
            kind: "gridStride",
            items: elements(matched.shape),
            size: WORKGROUP_SIZE,
          },
        },
      ],
    };
  },
});
