/**
 * The fusion pass (pure functions): planned node list ({@link NodePlan} from plan.ts) → execution
 * step list.
 *
 * It adds nothing to the IR's public vocabulary. It is a peephole that, at run time only, folds
 * **fixed sequences of adjacent nodes** emitted by the exporter into one private kernel; a form it
 * does not match falls through to the plain node list unchanged (the fallback is always the correct
 * existing path). It never touches the GPU, so it can be verified without an adapter.
 *
 * ## Where things live (the entry of the fusion pass)
 *
 * **This file is the entry of the fusion pass**; the implementation is split three ways: the shared
 * types, the eligibility conditions shared by every rule and `defineRule` in
 * {@link "./fusion-rule.ts"}; the eight rules under `fusion-rules/` (one rule = one file); and here
 * the declaration table {@link FUSION_RULES}, aliasing
 * ({@link planAliases}), the scan ({@link planFusions}) and candidate enumeration
 * ({@link enumerateUnfusedWindows}).
 * MUST: consumers **import this spelling (`fusion.ts`)** — ADR 0040 decision 1 names the place of
 * the fusion pass as "a pure-function pass in `src/runtime/fusion.ts`", and the split into three
 * files is only an implementation convenience (this file re-exports the public types and
 * {@link planRowBlocks}).
 * MUST: value dependencies are **one-directional: fusion-rule → fusion-rules → fusion**. Rules do
 * not import `fusion.ts` (the declaration table imports the rules, so that would be a cycle).
 *
 * ## Design commitments
 *
 * - MUST: do not write matchers straight into the executor's scan loop. This is the only decision
 *   point, and the counterexamples (use-count 2 / graph output / near-shape / different dtype /
 *   different order) can be covered exhaustively without a GPU.
 * - MUST: a fused step declares the **total-use list** of its external inputs
 *   ({@link FusedStep.ins}), and the release bookkeeping joins the executor's existing path (the
 *   same as for plain nodes). A hand-written retain/release per fusion would give the arena's
 *   reference counting one separate implementation per fusion, and a single drift would become a
 *   silent wrong value with no exception.
 * - MUST: fusion collapses the op sequence but **does not change values**. Preserving the rounding
 *   points is the kernel's job, and each kernel's docstring states that the means (a rounding
 *   barrier through a workgroup-memory round trip) is not a spec guarantee. The opt-in rmsNormAdd
 *   is limited to the acceptance range of ADR 0099 and keeps the reference path on unverified
 *   GPUs.
 * - MUST: only nodes whose **contract has exactly one output** are grabbed
 *   ({@link windowIsSingleOutput} — ADR 0068 decision 1). {@link FusedStep} stays single-output,
 *   and the output of a chain node is always `outputs[0]` (fusion of multi-output ops is not in
 *   the vocabulary).
 *
 * ## The fold target is not always one dispatch
 *
 * linearStaticQuantize / rmsNormAdd / silu / upsample2x / rope / adaln / gatedResidual are "N nodes
 * → one dispatch of a private kernel" ({@link LINEAR_STATIC_QUANTIZE_RULE} folds the parallel GEMV
 * and the fixed requantization into one dispatch), but
 * {@link ROW_BLOCK_ATTENTION_RULE} folds **the materialised width of an intermediate, not the
 * ops**, so it becomes a list of dispatches with temporaries closed inside the step
 * ({@link FusedStep.temps}) in between. adaln's per-token form is two dispatches as well (a pack
 * of the four modulation vectors into a step temporary, then the body — the body alone would need
 * more storage bindings than the WebGPU default, and fusion does not depend on device limits).
 * Either way one {@link FusedStep} = one execution step, and
 * the point where the release bookkeeping joins does not change.
 *
 * ## A consumer that absorbs its producer (linear prologue)
 *
 * One fusion is not a {@link FusedStep}: an elementwise {@link LINEAR_PROLOGUE_OPS} producer folded
 * into the step ① of the linear that consumes it (perf-ledger F-4 — {@link linearPrologueAt}). The
 * consumer absorbs its producer **through an annotation on its node step** ({@link LinearPrologue},
 * like the packed activations of ADR 0105): the decision is made here with the consumer's own
 * eligibility predicate, and the consumer's recipe executes it (its i8a8 step ① quantizes `op(x)`
 * directly). The linear's input is re-pointed to the producer's input and the producer has no step,
 * so the bookkeeping stays the one path of a plain node. It is tried after the rules at each index
 * and counted as `linearPrologue`.
 *
 * ## Application order
 *
 * The **declaration order** of {@link FUSION_RULES} (linearStaticQuantize → rmsNormAdd → silu →
 * upsample2x → rope → adaln → gatedResidual → rowBlockAttention).
 * The leading ops of the eight rules are `linear` / `rms_norm` / `sigmoid` / `reshape` /
 * `mul|slice` / `layer_norm` / `where` / `bmm`, pairwise disjoint,
 * so the order does not affect the result (the order matters only when leading ops overlap —
 * tests/runtime_fusion_test.ts checks mechanically from {@link FusionRule.heads} that no overlap
 * arises). A form where another rule's leading op appears **inside** a window (rowBlockAttention's
 * window contains five `reshape` / `expand`) cannot fire, because once grabbed the scan advances by
 * the window width.
 * MUST: keep {@link FUSION_RULES} as **one hand-written list** (do not replace it with automatic
 * registration at rule import time). The rule that every module is side-effect free bans running
 * code at import time, and tests/runtime_fusion_test.ts reads **this array** to check the
 * disjointness of the leading ops mechanically.
 *
 * ## In-window passthrough
 *
 * A chain is **not always contiguous**. For adaLN's broadcast form the exporter puts 2–3
 * `reshape`s of the modulation vector between `layer_norm` and `mul` (all 85 measured chains;
 * the per-token form is contiguous). For RoPE's half split, a
 * `sym_prefix_slice` of the cos / sin table sits between `cat` and the following `mul` **only at
 * the first use of the table**.
 * A rule declares a consecutive window
 * (`FusionMatch.window` in {@link "./fusion-rule.ts"}) and places every node of it other than the
 * folded ones (`chain`) as a **passthrough**, a plain node **before** the fused step. The
 * reordering is legal only when the passthrough consumes no value the chain defines, which
 * {@link passthroughIsIndependent} checks mechanically.
 */

import type { PrologueUnaryOp } from "../codegen/elementwise.ts";
import { linearGemvPackedEligible } from "../kernels/linear-gemv.ts";
import { quantizeRowsPrologueEligible } from "../kernels/quantize-rows.ts";
import { numel, permuteDims, STATIC_QUANTIZE_OP, staticQuantizeScale } from "../ops.ts";
import { ADALN_RULE } from "./fusion-rules/adaln.ts";
import { GATED_RESIDUAL_RULE } from "./fusion-rules/gated-residual.ts";
import { LINEAR_STATIC_QUANTIZE_RULE } from "./fusion-rules/linear-static-quantize.ts";
import { RMS_NORM_ADD_RULE } from "./fusion-rules/rms-norm-add.ts";
import { ROPE_RULE } from "./fusion-rules/rope.ts";
import { ROW_BLOCK_ATTENTION_RULE } from "./fusion-rules/row-block-attention.ts";
import { SILU_RULE } from "./fusion-rules/silu.ts";
import { UPSAMPLE_2X_RULE } from "./fusion-rules/upsample2x.ts";
import {
  allF32,
  type ExecStep,
  type FusionContext,
  type FusionCounterName,
  type FusionHit,
  type FusionPlan,
  type FusionRule,
  type FusionScanContext,
  internalsArePrivate,
  type LinearPrologue,
  type PackedActivations,
  passthroughIsIndependent,
  sameShape,
  windowIsSingleOutput,
  windowTouchesState,
} from "./fusion-rule.ts";
import { linearTakesI8a8 } from "./linear-route.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";

/**
 * The public types of the fusion pass (defined in {@link "./fusion-rule.ts"}). MUST: consumers
 * import the `fusion.ts` spelling — a re-export that keeps one entry; the split into three files
 * is an implementation convenience.
 */
export type {
  ExecStep,
  FusedOperand,
  FusedStep,
  FusionCounts,
  FusionLimits,
  FusionPlan,
  FusionWeightLayout,
  LinearPrologue,
  PackedActivations,
} from "./fusion-rule.ts";
/**
 * Row-block splitting and the derivation of its limit (defined in
 * {@link "./fusion-rules/row-block-attention.ts"}).
 */
export {
  planRowBlocks,
  type RowBlockLimit,
  rowBlockLimit,
} from "./fusion-rules/row-block-attention.ts";

/** MUST: the order of this array is the application order (the "Application order" section). */
export const FUSION_RULES: readonly FusionRule[] = [
  LINEAR_STATIC_QUANTIZE_RULE,
  RMS_NORM_ADD_RULE,
  SILU_RULE,
  UPSAMPLE_2X_RULE,
  ROPE_RULE,
  ADALN_RULE,
  GATED_RESIDUAL_RULE,
  ROW_BLOCK_ATTENTION_RULE,
];

/**
 * An axis of length 1 always has coordinate 0, so moving it does not affect the flat element order.
 * Every other axis has to keep its original order. Swapping two axes of the same length is not
 * allowed either. Empty tensors stay on the existing path.
 * DECIDED: docs/decisions/0011-layout-strategy.md#addendum-2026-09-14--permute-that-keeps-element-order
 */
const permuteKeepsElementOrder = (plan: NodePlan): boolean => {
  const shape = plan.inputShapes[0];
  if (shape.some((dim) => dim === 0)) return false;
  let previous = -1;
  for (const axis of permuteDims(plan.node.attrs, "permute alias")) {
    if (shape[axis] === 1) continue;
    if (axis <= previous) return false;
    previous = axis;
  }
  return true;
};

/**
 * The only decision point of the aliasing rule. The newly skipped permute copies are limited to
 * internally allocated storage. Extending the alias to inputs or weights would make the existing
 * copyOutputs reject it as a self-copy.
 */
const aliasesInput = (plan: NodePlan, internal: ReadonlySet<string>): boolean =>
  plan.contract.kind === "reshape" ||
  (plan.contract.kind === "permute" &&
    internal.has(plan.node.ins[0]) &&
    permuteKeepsElementOrder(plan)) ||
  (plan.contract.kind === "expand" && sameShape(plan.inputShapes[0], plan.outputs[0].shape));

/**
 * Derives aliases and the origin of storage in declaration order. Inputs and weights are outside
 * the set; only allocated values and their aliases are internal values. Shared by the execution
 * phase and the estimation phase, so the shape and ownership decisions are not copied separately
 * (ADR 0011).
 */
export const planAliases = (nodes: readonly NodePlan[]): ReadonlySet<NodePlan> => {
  const internal = new Set<string>();
  const aliases = new Set<NodePlan>();
  for (const plan of nodes) {
    const alias = aliasesInput(plan, internal);
    if (alias) aliases.add(plan);
    if (!alias || internal.has(plan.node.ins[0])) {
      for (const output of plan.outputs) internal.add(output.name);
    }
  }
  return aliases;
};

const NO_PACKED: ReadonlyMap<string, number> = new Map();

/**
 * Attaches the packed activation role (ADR 0105) to one plain node. The producer is the fixed SRQ,
 * the consumer a linear that takes that value in its **activation slot**; both follow from the one
 * decision in `packedValues`.
 */
const packedRole = (
  plan: NodePlan,
  packed: ReadonlyMap<string, number>,
): { readonly packedActivations: PackedActivations } | Record<never, never> => {
  if (packed.size === 0) return {};
  if (plan.node.op === STATIC_QUANTIZE_OP) {
    const scale = packed.get(plan.outputs[0].name);
    return scale === undefined ? {} : { packedActivations: { role: "write", scale } };
  }
  if (plan.node.op !== "linear") return {};
  const scale = packed.get(plan.node.ins[0]);
  return scale === undefined ? {} : { packedActivations: { role: "read", scale } };
};

/**
 * The producers a linear's step ① may absorb (perf-ledger F-4 — the `op` of {@link LinearPrologue}).
 *
 * MUST: an op enters here only once its fused form (`quantizeRowsPrologueWgsl` in
 * kernels/quantize-rows.ts) has been measured bit-identical to the pair "elementwise op →
 * quantize_rows" (ADR 0040 decision 2). Both entries were measured so (the ADR 0040 addendum of
 * 2026-10-25 records each): `gelu_tanh` (the Wan DiTs' FFN) and the erf `gelu` (anima's DiT FFN,
 * DeBERTa's intermediate), with 0 differing words in `xq` and `xs` on Vulkan / NVIDIA. A new
 * {@link PrologueUnaryOp} enters with its own measurement and tests.
 */
export const LINEAR_PROLOGUE_OPS: readonly PrologueUnaryOp[] = ["gelu_tanh", "gelu"];

/**
 * The linear prologue (perf-ledger F-4) at `index`: an elementwise {@link LINEAR_PROLOGUE_OPS} node
 * directly followed by the linear that reads its output. The linear's i8a8 step ① then quantizes
 * `op(x)` itself and the producer disappears, so its f32 output is never written or read.
 *
 * The decision is the fusion pass's, the execution the linear recipe's (the same split as packed
 * activations — ADR 0105): the hit is a plain node step of the linear whose first input is
 * re-pointed to the producer's input, annotated with {@link LinearPrologue}.
 *
 * Accepted only on an exact match (ADR 0040 decision 2):
 *
 * 1. `nodes[index]` is an op of {@link LINEAR_PROLOGUE_OPS}, f32 in and out.
 * 2. Its output has exactly one use and is not a graph output ({@link internalsArePrivate}).
 * 3. That use is the activation slot (`ins[0]`) of the **next** node, a `linear` — a two-node window
 *    like every other fusion, so the re-pointed input lives exactly as long as the folded value
 *    would have.
 * 4. The linear takes the i8a8 path by the recipe's own predicate ({@link linearTakesI8a8} on the
 *    weight's resident storage), and its row length fits the prologue kernel
 *    ({@link quantizeRowsPrologueEligible}).
 * 5. At least one row: the prologue kernel binds `x` as `array<vec4<f32>>`, whose 16-byte minimum
 *    binding size the 4-byte buffer of an empty tensor does not meet.
 *
 * Anything else returns `undefined` and both nodes stay as they are (byte-identical behaviour).
 * NOTE: `linearStaticQuantize` (head `linear`) and packed activations require `linearCompute: "f32"`
 * and this requires `"a8"`, so they never compete for the same linear.
 */
const linearPrologueAt = (
  nodes: readonly NodePlan[],
  index: number,
  context: FusionContext,
): { readonly plan: NodePlan; readonly prologue: LinearPrologue } | undefined => {
  const producer = nodes[index];
  const linear = nodes[index + 1];
  if (linear === undefined || linear.node.op !== "linear") return undefined;
  const op = LINEAR_PROLOGUE_OPS.find((accepted) => accepted === producer.node.op);
  if (op === undefined) return undefined;
  const window = [producer, linear];
  if (!windowIsSingleOutput(window) || windowTouchesState(window)) return undefined;
  if (!allF32([producer]) || !internalsArePrivate(window, context)) return undefined;
  if (linear.node.ins[0] !== producer.outputs[0].name) return undefined;
  const [x, weight] = linear.inputShapes;
  const k = weight[1];
  const storage = context.weightLayouts?.get(linear.node.ins[1])?.storage ?? "f32";
  if (!linearTakesI8a8(context.linearCompute, storage, k) || !quantizeRowsPrologueEligible(k)) {
    return undefined;
  }
  if (numel(x.slice(0, -1)) < 1) return undefined;
  // MUST: the re-pointed input is read in the folded value's place, so it has the same shape and
  // dtype (an elementwise unary op changes neither; a mismatch is a broken plan, not a non-match).
  if (!sameShape(producer.inputShapes[0], x) || producer.inputDtypes[0] !== linear.inputDtypes[0]) {
    throw new ExecutionError(
      `linear prologue: ${op} input [${producer.inputShapes[0].join(",")}] (${producer.inputDtypes[0]})` +
        ` does not match the linear's input [${x.join(",")}] (${linear.inputDtypes[0]})`,
    );
  }
  return {
    plan: {
      ...linear,
      node: { ...linear.node, ins: [producer.node.ins[0], ...linear.node.ins.slice(1)] },
    },
    prologue: { op },
  };
};

/** Scans the node list once and folds it into fused steps. Nodes not grabbed stay plain. */
const scanFusions = (
  nodes: readonly NodePlan[],
  context: FusionContext,
  packed: ReadonlyMap<string, number>,
): FusionPlan => {
  const steps: ExecStep[] = [];
  const aliases = planAliases(nodes);
  const scanContext: FusionScanContext = { ...context, packedValues: packed };
  const counts: Record<FusionCounterName, number> = {
    linearStaticQuantize: 0,
    rmsNormAdd: 0,
    silu: 0,
    upsample2x: 0,
    rope: 0,
    adaln: 0,
    gatedResidual: 0,
    rowBlockAttention: 0,
    identityExpand: 0,
    packedStaticQuantize: packed.size,
    linearPrologue: 0,
  };
  const pushNode = (plan: NodePlan): void => {
    const alias = aliases.has(plan);
    if (alias && plan.contract.kind === "expand") counts.identityExpand += 1;
    steps.push({ kind: "node", plan, aliasesInput: alias, ...packedRole(plan, packed) });
  };
  for (let index = 0; index < nodes.length;) {
    const hit = FUSION_RULES.reduce<FusionHit | undefined>(
      (found, rule) => found ?? rule.apply(nodes, index, scanContext),
      undefined,
    );
    if (hit !== undefined) {
      // MUST: passthroughs first (the fused step may take a passthrough's output as input).
      for (const plan of hit.passthrough) pushNode(plan);
      steps.push(hit.step);
      counts[hit.step.rule] += 1;
      index += hit.advance;
      continue;
    }
    const folded = linearPrologueAt(nodes, index, context);
    if (folded !== undefined) {
      // The producer gets no step; the linear's step stands for both nodes. A linear never aliases
      // (planAliases), and packed activations need linearCompute "f32" (never with a prologue).
      steps.push({
        kind: "node",
        plan: folded.plan,
        aliasesInput: false,
        prologue: folded.prologue,
      });
      counts.linearPrologue += 1;
      index += 2;
      continue;
    }
    pushNode(nodes[index]);
    index += 1;
  }
  return { steps, counts };
};

/**
 * Value name → the nodes consuming that value (total uses — a node that takes it twice appears
 * twice).
 */
const consumersByValue = (nodes: readonly NodePlan[]): ReadonlyMap<string, readonly NodePlan[]> => {
  const consumers = new Map<string, NodePlan[]>();
  for (const plan of nodes) {
    for (const name of plan.node.ins) {
      const found = consumers.get(name);
      if (found === undefined) consumers.set(name, [plan]);
      else found.push(plan);
    }
  }
  return consumers;
};

/**
 * Whether this consumer is "a linear that can receive packed activations" (the consumer-side
 * condition of the ADR 0105 pairing).
 *
 * MUST: take it only in the activation slot (`ins[0]`). When the same value arrives as weight /
 * bias, that binding expects f32 words, so switching to packed gives a silent wrong value.
 * MUST: the decision is the single {@link linearGemvPackedEligible} (the same predicate as the
 * recipe-builder's check). Landing on the parallel GEMV is not enough; it is limited to **the rows
 * where packed was measured to help** (ADR 0105 addendum 1) — making forms packed where it does not
 * help is a pure loss, producer SRQ included.
 */
const readsPackedActivation = (plan: NodePlan, name: string, context: FusionContext): boolean => {
  if (plan.node.op !== "linear" || plan.node.ins.length !== 3) return false;
  if (plan.node.ins[0] !== name || plan.node.ins[1] === name || plan.node.ins[2] === name) {
    return false;
  }
  if (!allF32([plan])) return false;
  const weight = context.weightLayouts?.get(plan.node.ins[1]);
  if (weight === undefined || weight.storage === "f16" || weight.storage === "bf16") return false;
  const m = numel(plan.inputShapes[0].slice(0, -1));
  const [n, k] = plan.inputShapes[1];
  const group = weight.storage === "i4" ? weight.groupSize : undefined;
  return linearGemvPackedEligible(weight.storage, m, n, k, group) !== undefined;
};

/**
 * The fixed SRQs handed over as packed int8 activations (ADR 0105) — value name → that SRQ's f32
 * scale.
 *
 * Five conditions for acceptance:
 *
 * 1. The option (`packedStaticQuantize`) is set, with `linearGemvReduce: parallel` /
 *    `linearCompute: f32` (only the parallel GEMV family has packed variants).
 * 2. A `static_quantize` **left as a plain node**. An SRQ swallowed by the linear→SRQ fusion
 *    (ADR 0103) is an output-side epilogue, and packed output is out of scope here, so it does not
 *    qualify — which is why this decision takes the result of the first scan (`steps`) as input.
 * 3. The output is not a graph output, the last dimension k is a multiple of 16 (the binding reads
 *    `vec4<u32>` = units of 16 elements), and the scale is positive and finite (scale 0 is an
 *    identity SRQ and cannot be lowered to int8 codes).
 * 4. There is at least one consumer, and **every one of them** is the activation slot of a
 *    parallel-GEMV linear of a form where packed was measured to help
 *    ({@link readsPackedActivation}). If even one GEMM / sequential GEMV / other op / form measured
 *    not to help is mixed in, it stays f32.
 * 5. No fused step other than {@link LINEAR_STATIC_QUANTIZE_RULE} consumes the value.
 *
 * MUST: 5 is an invariant for **future rules**. Only the linearStaticQuantize fusion has a
 * spelling that reads packed, so the moment another rule puts a linear into its window, that
 * fused kernel reads the packed words as f32 (a silent wrong value with no exception). No window of
 * the current eight rules contains a `linear`, so it does not fire today; but without it on the
 * accepting side, only the added rule would break, silently.
 */
const packedStaticQuantizeValues = (
  nodes: readonly NodePlan[],
  steps: readonly ExecStep[],
  context: FusionContext,
): ReadonlyMap<string, number> => {
  if (
    context.packedStaticQuantize !== true ||
    context.linearGemvReduce !== "parallel" ||
    context.linearCompute !== "f32"
  )
    return NO_PACKED;
  const consumers = consumersByValue(nodes);
  const foldedElsewhere = new Set<string>();
  for (const step of steps) {
    if (step.kind !== "fused" || step.rule === "linearStaticQuantize") continue;
    for (const name of step.ins) foldedElsewhere.add(name);
  }
  const packed = new Map<string, number>();
  for (const step of steps) {
    if (step.kind !== "node" || step.plan.node.op !== STATIC_QUANTIZE_OP) continue;
    const [output] = step.plan.outputs;
    if (context.outputNames.has(output.name) || foldedElsewhere.has(output.name)) continue;
    if (!allF32([step.plan])) continue;
    const k = output.shape[output.shape.length - 1];
    if (k === undefined || k <= 0 || k % 16 !== 0) continue;
    const scale = staticQuantizeScale(step.plan.node.attrs, "packed static_quantize");
    if (!Number.isFinite(scale) || scale <= 0) continue;
    const uses = consumers.get(output.name) ?? [];
    if (uses.length === 0) continue;
    if (!uses.every((use) => readsPackedActivation(use, output.name, context))) continue;
    packed.set(output.name, scale);
  }
  return packed;
};

/**
 * Scans the node list and folds it into fused steps. Nodes not grabbed stay plain.
 *
 * Only when the packed activation option (ADR 0105) is set does it **scan twice**. The first scan
 * obtains the input of the pairing (which SRQs stay plain nodes); the second is the result.
 * MUST: whether packed is on enters no rule's `match` condition (only the key, WGSL and params of
 * the dispatch built by a grabbed linear→SRQ fusion change), so both scans have **the same rule
 * grab the same window** — the pairing's input never moves in the second scan.
 */
export const planFusions = (nodes: readonly NodePlan[], context: FusionContext): FusionPlan => {
  const first = scanFusions(nodes, context, NO_PACKED);
  const packed = packedStaticQuantizeValues(nodes, first.steps, context);
  return packed.size === 0 ? first : scanFusions(nodes, context, packed);
};

/**
 * One candidate window (the return value of {@link enumerateUnfusedWindows}).
 *
 * NOTE: this is **a candidate, not an acceptance**. It stands in for none of the acceptance set of
 * ADR 0040 decision 2 (each rule's spelling, shape conditions, preservation of rounding points) —
 * it is a marker "a rule written here might fold this" that passed only the shared eligibility
 * conditions.
 */
export type UnfusedWindow = {
  /** The op names of the chain that would be folded (in-window passthroughs excluded). */
  readonly ops: readonly string[];
  /** Node count of the consecutive window; the difference from `ops.length` = passthrough count. */
  readonly windowSize: number;
  /** The position of the window's first node (index in node order, the step list expanded). */
  readonly nodeIndex: number;
  /** The output name of the chain's last node (the clue for locating the window in the IR). */
  readonly outputName: string;
};

/**
 * Whether one node in the window may join the chain (= whether its output has "exactly one
 * consumer and is not a graph output").
 *
 * MUST: do not copy the privacy spelling here; let {@link internalsArePrivate} decide as is —
 * treated as a two-node chain headed by that node, only the first node's output is checked
 * (the second is the chain's tail, so it is not looked at). A separate copy would let the
 * candidate table alone keep saying "foldable" by the old criterion when the accepted set changes.
 */
const outputStaysInternal = (step: NodePlan, tail: NodePlan, context: FusionContext): boolean =>
  internalsArePrivate([step, tail], context);

/**
 * Finds the **subsequence to fold** (the chain) of a consecutive window. Of the nodes flowing into
 * the window's last node, only those whose outputs close inside the window form the chain; the rest
 * are in-window passthroughs (the subject of {@link passthroughIsIndependent}).
 *
 * This split matches the measured forms of the current two rules with in-window passthroughs:
 *
 * - RoPE's `sym_prefix_slice` of the cos / sin table flows into the chain's `mul`, but the table
 *   is also read by other layers (two or more consumers), so it does not join the chain.
 * - adaLN's gate `reshape` is consumed outside the chain (it does not flow into the window's last
 *   node), so it does not join the chain.
 *
 * A window whose chain does not start at the window's first node gives `undefined` — a cut-off so
 * that the same chain is not counted again and again from "windows with unrelated nodes added in
 * front"; with it, one chain always corresponds to exactly one window (first = the chain's first,
 * last = the chain's last).
 */
const chainWithinWindow = (
  window: readonly NodePlan[],
  context: FusionContext,
): readonly NodePlan[] | undefined => {
  const last = window[window.length - 1];
  const needed = new Set(last.node.ins);
  const chain = [last];
  for (let at = window.length - 2; at >= 0; at -= 1) {
    const step = window[at];
    if (!step.outputs.some((out) => needed.has(out.name))) continue;
    // A node whose output leaves the window does not join the chain (folding would lose the
    // value). Its inputs are not added to needed, so upstream nodes that reach the last node only
    // through it drop out of the chain automatically.
    if (!outputStaysInternal(step, last, context)) continue;
    chain.push(step);
    for (const name of step.node.ins) needed.add(name);
  }
  chain.reverse();
  return chain[0] === window[0] ? chain : undefined;
};

/**
 * **Candidate enumeration of the consecutive windows fusion did not grab** (the first step of
 * tools/fusion-hints (removed in 0.15)).
 *
 * Walks the consecutive runs of plain nodes that `planFusions` left as `kind: "node"` with windows
 * of length 2–`maxWindow`, and returns only the windows that pass the eligibility conditions
 * shared by every rule. The predicates are the same five as fusion:
 * {@link windowIsSingleOutput} and {@link windowTouchesState} on the window, {@link allF32} on the
 * chain, and on the split between chain and in-window passthroughs itself
 * {@link internalsArePrivate} ({@link chainWithinWindow} applies it node by node) and
 * {@link passthroughIsIndependent}.
 *
 * MUST: do not take the predicates outside the fusion-pass layer (`fusion-rule.ts` /
 * `fusion-rules/` / here) — the "only decision point" of ADR 0040 decision 1 covers not just the
 * fusion decision but also the spelling of the eligibility conditions. Putting the enumerator
 * outside the layer and taking the predicates along would leave the candidate table behind when
 * the accepted set changes (the table alone keeps saying "foldable" by the old criterion).
 * MUST: do not build windows across a fused step or a linear with a folded producer
 * ({@link LinearPrologue}) — a chain already grabbed is not counted twice as a candidate. The scan
 * closes per **consecutive run** of plain nodes.
 *
 * NOTE: what comes back is candidates, not a design. Prefixes of a long window also come out as
 * separate candidates (a property of n-gram enumeration), so read the longest window with the same
 * `nodeIndex`.
 */
export const enumerateUnfusedWindows = (
  steps: readonly ExecStep[],
  context: FusionContext,
  maxWindow: number,
): readonly UnfusedWindow[] => {
  if (!Number.isSafeInteger(maxWindow) || maxWindow < 2) {
    throw new ExecutionError(`window width limit ${maxWindow} must be an integer of at least 2`);
  }
  const found: UnfusedWindow[] = [];
  let run: NodePlan[] = [];
  let runStart = 0;
  let nodeIndex = 0;
  const sweepRun = (): void => {
    for (let start = 0; start < run.length; start += 1) {
      for (let size = 2; size <= maxWindow && start + size <= run.length; size += 1) {
        const window = run.slice(start, start + size);
        if (!windowIsSingleOutput(window) || windowTouchesState(window)) continue;
        const chain = chainWithinWindow(window, context);
        if (chain === undefined || !allF32(chain)) continue;
        const folded = new Set(chain);
        const passthrough = window.filter((step) => !folded.has(step));
        if (!passthroughIsIndependent(chain, passthrough)) continue;
        found.push({
          ops: chain.map((step) => step.node.op),
          windowSize: size,
          nodeIndex: runStart + start,
          outputName: chain[chain.length - 1].outputs[0].name,
        });
      }
    }
    run = [];
  };
  for (const step of steps) {
    if (step.kind === "node" && step.prologue === undefined) {
      if (run.length === 0) runStart = nodeIndex;
      run.push(step.plan);
      nodeIndex += 1;
      continue;
    }
    sweepRun();
    // A linear with a folded producer is a grabbed pair of two original nodes (like a fused step).
    nodeIndex += step.kind === "node" ? 2 : step.nodeCount;
  }
  sweepRun();
  return found;
};
