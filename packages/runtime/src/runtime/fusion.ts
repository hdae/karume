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
 * {@link "./fusion-rule.ts"}; the ten rules under `fusion-rules/` (one rule = one file); and here
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
 * linearStaticQuantize / geluMulStaticQuantize / rmsNormAdd / silu / upsample2x / rope / rmsNormRope /
 * adaln (its broadcast form) / gatedResidual are "N nodes → one dispatch of a private kernel"
 * ({@link LINEAR_STATIC_QUANTIZE_RULE} folds the parallel GEMV and the fixed requantization into one
 * dispatch), but
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
 * The same kind of fusion lets a fused attention on an a8 seat absorb the `[0, 2, 1, 3]` permutes
 * that feed its q / k / v and, on the online route, the one after it (perf-ledger F-3 A4 / F-8 —
 * {@link attentionPermutesAt}): its node step carries {@link AttentionTokenMajor}, its operands and
 * output are re-pointed to the permutes' token-major inputs / output, and the permutes have no step.
 * The input permutes are decided from the seat — the attention recipe honours them on every route an
 * a8 seat can take; the output permute from `attentionRoute`, the recipe's own route decision. It is
 * tried after the linear prologue and counted as `attentionPermute` (one per absorbed permute).
 *
 * ## Application order
 *
 * The **declaration order** of {@link FUSION_RULES} (linearStaticQuantize → geluMulStaticQuantize →
 * rmsNormAdd → silu → upsample2x → rope → rmsNormRope → adaln → gatedResidual → rowBlockAttention).
 * The leading ops of the ten rules are `linear` / `gelu_tanh` / `rms_norm` / `sigmoid` /
 * `reshape` / `mul|slice` / `rms_norm` / `layer_norm` / `where` / `bmm`, pairwise disjoint but
 * for one pair,
 * so the order does not affect the result (the order matters only when two rules can accept the
 * same index — tests/runtime_fusion_test.ts checks mechanically from {@link FusionRule.heads} that
 * no other overlap arises). The one shared head, `rms_norm` of rmsNormAdd and rmsNormRope, cannot
 * accept the same index: rmsNormAdd needs the normalised value's single use to be the next node, an
 * `add`; rmsNormRope needs every use inside its chain, which reads it only in a `reshape` (the test
 * applies both rules at every index of a graph holding both forms). A form where another rule's
 * leading op appears **inside** a window (rowBlockAttention's window contains five `reshape` /
 * `expand`) cannot fire, because once grabbed the scan advances by the window width (a deferred
 * node is the exception — it is scanned again by design, see below).
 * MUST: keep {@link FUSION_RULES} as **one hand-written list** (do not replace it with automatic
 * registration at rule import time). The rule that every module is side-effect free bans running
 * code at import time, and tests/runtime_fusion_test.ts reads **this array** to check the
 * disjointness of the leading ops mechanically. The list a plan uses is **passed in**
 * (`FusionContext.rules`): a Session passes {@link sessionFusionRules} — the external rules it was
 * given (`SessionOptions.fusionRules`, ADR 0148) in their order, then this array — and the scan reads
 * no module-level list. An external rule is tried first at each index, so where it and a built-in
 * rule could both take an index, the external rule takes it; the disjointness above holds among the
 * built-in rules only.
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
 *
 * ## Deferred nodes (a chain interleaved with another)
 *
 * The Wan DiT's q and k prologues interleave (`rms_norm` q, `rms_norm` k, the reshapes of q / k / v,
 * then q's pair reshape and RoPE, then k's), so q's window holds k's `rms_norm` and reshape. As
 * passthroughs they would be emitted as plain nodes before q's fused step and k could never fold. A
 * rule may therefore **defer** its window's other nodes (`FusionMatch.others` — only rmsNormRope
 * does): the fused step takes the window's place, and the deferred nodes are written into the
 * window's last slots of the scanned node list, where the scan resumes — so they come after the
 * fused step and any rule may grab them again. The reordering is legal only when the chain and the
 * deferred nodes are independent both ways ({@link passthroughIsIndependent} and
 * `deferralIsIndependent` in {@link "./fusion-rule.ts"}). The step list is then a valid order of the
 * nodes that differs from the declaration order inside the deferred region only; a consumer that
 * walks it against the declaration order (the estimate's `postFusionSteps`, the node index of
 * {@link enumerateUnfusedWindows}) counts positions in the scanned list.
 */

import type { PrologueUnaryOp } from "../codegen/elementwise.ts";
import { linearGemvPackedEligible } from "../kernels/linear-gemv.ts";
import { quantizeRowsPrologueEligible } from "../kernels/quantize-rows.ts";
import { numel, permuteDims, STATIC_QUANTIZE_OP, staticQuantizeScale } from "../ops.ts";
import { attentionRoute, UNREAD_ATTENTION_DEVICE } from "./attention-route.ts";
import { ADALN_RULE } from "./fusion-rules/adaln.ts";
import { GATED_RESIDUAL_RULE } from "./fusion-rules/gated-residual.ts";
import { GELU_MUL_STATIC_QUANTIZE_RULE } from "./fusion-rules/gelu-mul-static-quantize.ts";
import { LINEAR_STATIC_QUANTIZE_RULE } from "./fusion-rules/linear-static-quantize.ts";
import { RMS_NORM_ADD_RULE } from "./fusion-rules/rms-norm-add.ts";
import { RMS_NORM_ROPE_RULE } from "./fusion-rules/rms-norm-rope.ts";
import { ROPE_RULE } from "./fusion-rules/rope.ts";
import { ROW_BLOCK_ATTENTION_RULE } from "./fusion-rules/row-block-attention.ts";
import { SILU_RULE } from "./fusion-rules/silu.ts";
import { UPSAMPLE_2X_RULE } from "./fusion-rules/upsample2x.ts";
import {
  absorbedPermutes,
  allF32,
  type AnnotationCounterName,
  type AttentionTokenMajor,
  type ExecStep,
  type ExternalRuleName,
  type FusionContext,
  type FusionHit,
  type FusionPlan,
  type FusionRule,
  type FusionRuleName,
  type FusionScanContext,
  fusionMatchContext,
  internalsArePrivate,
  isExternalRule,
  isExternalRuleName,
  ruleOrigin,
  type LinearPrologue,
  nodeStepSpan,
  type PackedActivations,
  passthroughIsIndependent,
  sameShape,
  windowIsSingleOutput,
  windowTouchesState,
} from "./fusion-rule.ts";
import {
  linearIntegerGemvVariant,
  linearPlanesGemvVariant,
  linearTakesI8a8,
} from "./linear-route.ts";
import { describeInputValue, ExecutionError, type NodePlan } from "./plan.ts";

/**
 * The public types of the fusion pass (defined in {@link "./fusion-rule.ts"}). MUST: consumers
 * import the `fusion.ts` spelling — a re-export that keeps one entry; the split into three files
 * is an implementation convenience.
 */
export type {
  AttentionTokenMajor,
  ContractClass,
  ExecStep,
  ExternalRuleDefinition,
  ExternalRuleName,
  FusedBuild,
  FusedDispatch,
  FusedOperand,
  FusedStep,
  FusedTemp,
  FusedWorkgroups,
  FusionContext,
  FusionCounts,
  FusionHit,
  FusionLimits,
  FusionMatch,
  FusionMatchContext,
  FusionPlan,
  FusionRule,
  FusionRuleDefinition,
  FusionRuleName,
  FusionScanContext,
  FusionWeightLayout,
  LinearPrologue,
  PackedActivations,
} from "./fusion-rule.ts";
export {
  absorbedPermutes,
  allF32,
  defineExternalRule,
  defineRule,
  fusionMatchContext,
  isExternalRuleName,
  nodeStepSpan,
  ruleOrigin,
  sameShape,
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

/**
 * The hand-written list of the built-in rules ({@link FUSION_RULES} is its typed view). Its tuple
 * type carries each rule's name, which the compile-time check below reads.
 */
const BUILT_IN_RULES = [
  LINEAR_STATIC_QUANTIZE_RULE,
  GELU_MUL_STATIC_QUANTIZE_RULE,
  RMS_NORM_ADD_RULE,
  SILU_RULE,
  UPSAMPLE_2X_RULE,
  ROPE_RULE,
  RMS_NORM_ROPE_RULE,
  ADALN_RULE,
  GATED_RESIDUAL_RULE,
  ROW_BLOCK_ATTENTION_RULE,
] as const;

/**
 * Compile-time check that {@link FusionRuleName} and the built-in list agree in both directions: the
 * constraint rejects a rule whose name is not in the union, and a name of the union with no rule in
 * the list turns the type into one that names it, which the list does not satisfy.
 */
type ListsEveryBuiltInName<Rules extends readonly FusionRule<FusionRuleName>[]> = [
  Exclude<FusionRuleName, Rules[number]["name"]>,
] extends [never]
  ? Rules
  : { readonly builtInNameWithoutRule: Exclude<FusionRuleName, Rules[number]["name"]> };

/**
 * MUST: the order of this array is the application order (the "Application order" section). It is
 * the list a Session passes to the pass ({@link FusionContext.rules}).
 */
export const FUSION_RULES: readonly FusionRule<FusionRuleName>[] =
  BUILT_IN_RULES satisfies ListsEveryBuiltInName<typeof BUILT_IN_RULES>;

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
 * and this requires `"a8"`, so they never compete for the same linear. `geluMulStaticQuantize`
 * (head `gelu_tanh`, tried first) needs the producer's next node to be a `mul` and this a `linear`,
 * so they never compete for the same producer.
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

/** The permutation an attention absorbs: token-major `[B, S, H, D]` ↔ head-major `[B, H, S, D]`. */
const HEAD_SWAP = [0, 2, 1, 3] as const;

/** The operand slots of a fused attention that may be token-major (the mask, slot 3, never is). */
const ATTENTION_OPERANDS = ["q", "k", "v"] as const;

/**
 * Whether `plan` is a permute an attention may absorb: `[0, 2, 1, 3]` on a rank-4 f32 tensor, and a
 * materialised copy (a permute that aliases its input — an axis of length 1 — costs nothing, and
 * absorbing it would add the copy back on the routes that make the head-major copy themselves).
 */
const isHeadSwap = (plan: NodePlan, aliases: ReadonlySet<NodePlan>): boolean =>
  plan.node.op === "permute" &&
  plan.contract.kind === "permute" &&
  plan.inputShapes[0]?.length === HEAD_SWAP.length &&
  sameShape(permuteDims(plan.node.attrs, "attention permute"), HEAD_SWAP) &&
  allF32([plan]) &&
  !aliases.has(plan);

/**
 * Whether `attention` (the declared head-major node) takes the online kernel: the same
 * `attentionRoute` that `buildAttention` calls, on the Session's seats and device facts, at its D and
 * key count N.
 *
 * MUST: on `"a8"` × `"none"` the route depends on the device, so without
 * {@link FusionContext.attentionDevice} this fails loudly instead of guessing one (the same reason as
 * the estimate's entry check of `EstimateOptions.attentionDevice`).
 */
const takesOnlineRoute = (attention: NodePlan, context: FusionContext): boolean => {
  const seats = {
    attentionCompute: context.attentionCompute ?? "f32",
    attentionScoreStorage: context.attentionScoreStorage ?? "f32",
  } as const;
  const [q, k] = attention.inputShapes;
  let device = context.attentionDevice;
  if (device === undefined) {
    if (seats.attentionCompute === "a8" && seats.attentionScoreStorage === "none") {
      throw new ExecutionError(
        `attention permute: attention [${q.join(",")}] × [${k.join(",")}]: attentionScoreStorage 'none'` +
          " needs attentionDevice (whether the attention takes the online kernel, the one route that" +
          " stores O token-major, depends on the device's subgroups and workgroup-storage limit — ADR 0132)",
      );
    }
    device = UNREAD_ATTENTION_DEVICE;
  }
  return attentionRoute(seats, q[3], k[2], device).online !== undefined;
};

/**
 * The permutes a fused attention absorbs at `index` (perf-ledger F-3 A4 / F-8): the Wan DiTs project
 * q / k / v token-major (`[B, S, H·D]` reshaped to `[B, S, H, D]`), permute each to head-major
 * `[B, H, S, D]` right before the attention, and permute O back right after it. On an a8 seat the
 * attention then reads the token-major tensors itself (the a8 quantisation reads q / k token-major
 * and builds Vᵀ from a token-major V), on the online route it also stores O token-major, and the
 * permutes' copies are never materialised.
 *
 * The decision is the fusion pass's, the execution the attention recipe's (the split of the linear
 * prologue): the hit is a plain node step of the attention whose absorbed operands are re-pointed to
 * the permutes' inputs and whose output, with `o`, is the output permute's, annotated with
 * {@link AttentionTokenMajor}.
 *
 * MUST: accept only on an a8 attention seat ({@link FusionContext.attentionCompute} — the session
 * knob, no device facts). On a dense seat (f32 / f16) every absorbed permute comes back as the same
 * strided copy inside the attention step: no dispatch is saved, and the token-major input stays live
 * next to its copy for the whole step. For the input permutes the gate is the seat, not the route:
 * inside an a8 seat a stage the shape keeps off i8a8 makes the head-major copy itself
 * (`buildAttention` in recipe-builders/attention.ts).
 * MUST: absorb the permute after the attention (O back to token-major) **only where the route is the
 * online kernel** ({@link takesOnlineRoute} — `attentionRoute`, the recipe's own route decision, so
 * the pass and the recipe cannot disagree). That kernel stores O token-major itself: one dispatch
 * less, and under the least-growth transient placement (ADR 0093 addendum of 2026-10-26) the backing
 * does not grow (the Wan2.2 DiT 717.8 / 2,457.2 MiB at S 7,920 / 27,280 either way). MUST NOT absorb
 * it on any other route: the three a8 dispatches and the dense stages store O head-major (③PV), so
 * the permute would come back as an O temporary plus the same strided copy — live bytes added (one
 * O: +92.8 / +319.7 MiB on the Wan2.2 DiT at S 7,920 / 27,280) for no dispatch saved. That path is
 * deliberately not built, and the recipe and the estimate fail loudly on an `o` that reaches it
 * (ADR 0040 addendum of 2026-10-26).
 *
 * Accepted only on an exact match (ADR 0040 decision 2). The window is `nodes[index]` onwards:
 *
 * 1. A run of `p ≥ 0` nodes, **every one** a permute an attention may absorb ({@link isHeadSwap}),
 *    directly followed by an `attention` without a states field (the fused form — the states form is
 *    another kernel family), f32, with one output.
 * 2. Each permute's output has exactly one use and is not a graph output ({@link internalsArePrivate}),
 *    and that use is the attention's q, k or v slot. A run that holds anything else returns
 *    `undefined` here; its tail is tried again from the next index, so the absorbed permutes are the
 *    longest such run that ends at the attention — contiguous, so each re-pointed input lives only
 *    across the other absorbed permutes.
 * 3. The output permute (`o`): the attention's output has exactly one use and is not a graph output,
 *    that use is the **next** node, a permute it may absorb, and the attention takes the online route.
 * 4. At least one permute is absorbed (otherwise both sides stay as they are).
 *
 * Anything else returns `undefined` and every node stays as it is (byte-identical behaviour).
 */
const attentionPermutesAt = (
  nodes: readonly NodePlan[],
  index: number,
  context: FusionContext,
  aliases: ReadonlySet<NodePlan>,
):
  | {
      readonly plan: NodePlan;
      readonly tokenMajor: AttentionTokenMajor;
      readonly advance: number;
    }
  | undefined => {
  if (context.attentionCompute !== "a8") return undefined;
  let at = index;
  while (at - index < ATTENTION_OPERANDS.length && nodes[at]?.node.op === "permute") at += 1;
  const attention = nodes[at];
  if (attention === undefined || attention.contract.kind !== "attention") return undefined;
  if (windowTouchesState([attention]) || !windowIsSingleOutput([attention])) return undefined;
  if (!allF32([attention])) return undefined;
  const sources = new Map<number, NodePlan>();
  for (const permute of nodes.slice(index, at)) {
    if (!isHeadSwap(permute, aliases) || !internalsArePrivate([permute, attention], context)) {
      return undefined;
    }
    const slot = attention.node.ins.indexOf(permute.outputs[0].name);
    if (slot < 0 || slot >= ATTENTION_OPERANDS.length) return undefined;
    sources.set(slot, permute);
  }
  const after = nodes[at + 1];
  const output =
    after !== undefined &&
    isHeadSwap(after, aliases) &&
    after.node.ins[0] === attention.outputs[0].name &&
    internalsArePrivate([attention, after], context) &&
    takesOnlineRoute(attention, context)
      ? after
      : undefined;
  if (sources.size === 0 && output === undefined) return undefined;
  // MUST: a re-pointed operand is the permute's input, which holds the operand's elements with the
  // two middle axes swapped (the permute's shape rule — a mismatch is a broken plan, not a non-match).
  const swapped = (shape: readonly number[]): readonly number[] =>
    HEAD_SWAP.map((axis) => shape[axis]);
  for (const [slot, permute] of sources) {
    if (!sameShape(swapped(permute.inputShapes[0]), attention.inputShapes[slot])) {
      throw new ExecutionError(
        `attention permute: ${ATTENTION_OPERANDS[slot]} input [${permute.inputShapes[0].join(",")}]` +
          ` is not the head swap of the operand [${attention.inputShapes[slot].join(",")}]`,
      );
    }
  }
  if (
    output !== undefined &&
    !sameShape(swapped(attention.outputs[0].shape), output.outputs[0].shape)
  ) {
    throw new ExecutionError(
      `attention permute: output [${output.outputs[0].shape.join(",")}] is not the head swap of` +
        ` the attention's [${attention.outputs[0].shape.join(",")}]`,
    );
  }
  const tokenMajor: AttentionTokenMajor = {
    q: sources.has(0),
    k: sources.has(1),
    v: sources.has(2),
    o: output !== undefined,
  };
  const operand = <T>(slot: number, own: T, absorbed: (permute: NodePlan) => T): T => {
    const permute = sources.get(slot);
    return permute === undefined ? own : absorbed(permute);
  };
  const outputs = output === undefined ? attention.outputs : output.outputs;
  return {
    plan: {
      ...attention,
      node: {
        ...attention.node,
        ins: attention.node.ins.map((name, slot) =>
          operand(slot, name, (permute) => permute.node.ins[0]),
        ),
        outs: outputs.map((out) => out.name),
      },
      inputShapes: attention.inputShapes.map((shape, slot) =>
        operand(slot, shape, (permute) => permute.inputShapes[0]),
      ),
      inputDtypes: attention.inputDtypes.map((dtype, slot) =>
        operand(slot, dtype, (permute) => permute.inputDtypes[0]),
      ),
      outputs,
    },
    tokenMajor,
    advance: at - index + 1 + (output === undefined ? 0 : 1),
  };
};

/** The annotation counters at zero (a record literal, so the type rejects a missing name). */
const ANNOTATION_ZEROS: Readonly<Record<AnnotationCounterName, number>> = {
  identityExpand: 0,
  packedStaticQuantize: 0,
  linearPrologue: 0,
  attentionPermute: 0,
};

/**
 * Whether `counts` — built from the names of `rules` and the annotation counters — has as many keys
 * as there are names, i.e. one counter per name.
 */
const oneCounterPerName = <Name extends string>(
  counts: Record<string, number>,
  rules: readonly FusionRule<Name>[],
): counts is Record<Name | AnnotationCounterName, number> =>
  Object.keys(counts).length === rules.length + Object.keys(ANNOTATION_ZEROS).length;

/**
 * The counter record of a plan at zero: one counter per rule name of `rules` and the annotation
 * counters (the scan starts from it; tests build their expected records from it).
 *
 * MUST: every name has its own counter. Two rules of one name, or a rule named like an annotation
 * counter, would add their hits together silently, so they fail loudly here.
 */
export const zeroFusionCounts = <Name extends string>(
  rules: readonly FusionRule<Name>[],
): Record<Name | AnnotationCounterName, number> => {
  const counts: Record<string, number> = {
    ...Object.fromEntries(rules.map((rule) => [rule.name, 0])),
    ...ANNOTATION_ZEROS,
  };
  if (!oneCounterPerName(counts, rules)) {
    const names = rules.map((rule) => rule.name);
    const shared = names.filter(
      (name, at) => names.indexOf(name) !== at || Object.hasOwn(ANNOTATION_ZEROS, name),
    );
    throw new ExecutionError(
      `fusion rules: the name(s) '${shared.join("', '")}' do not have a counter of their own` +
        " (each rule needs a name of its own, distinct from the annotation counters)",
    );
  }
  return counts;
};

/**
 * Checks that every disabled rule name ({@link FusionContext.disabledRules}) is a name of `rules`.
 * The pass calls it on its context, and Session construction on the Session's rule list (the
 * executor's `DISABLED_FUSION_RULES`), so a bad name fails before the first run.
 *
 * MUST: an unknown name fails loudly (ExecutionError). A misspelt name would leave its rule on, and
 * the side meant to be unfused would fuse — a fused-vs-unfused comparison would then compare a
 * kernel with itself and pass.
 */
export const assertDisabledRulesNamed = (
  rules: readonly FusionRule[],
  disabled: "all" | ReadonlySet<string> | undefined,
): void => {
  if (disabled === undefined || disabled === "all") return;
  const names = new Set(rules.map((rule) => rule.name));
  const unknown = [...disabled].filter((name) => !names.has(name));
  if (unknown.length === 0) return;
  throw new ExecutionError(
    `fusion rules: the disabled name(s) '${unknown.join("', '")}' are not names of the rule list` +
      ` (the rules are ${rules.map((rule) => `'${rule.name}'`).join(", ")})`,
  );
};

/**
 * The rule list a Session plans with: the external rules of `SessionOptions.fusionRules` in the
 * order given, then the built-in {@link FUSION_RULES} — an external rule is tried before every
 * built-in rule at each index (ADR 0148). Session construction calls it once, so the list is fixed
 * for the Session's lifetime (the prepared-plan key is the bindings alone).
 *
 * MUST: every entry is a rule {@link defineExternalRule} of this copy of the runtime built (its brand).
 * A built-in rule is refused as such; a hand-written object, a copy of a rule and a rule defined by
 * another copy of the runtime all lack the brand, and the refusal names both causes (the brand cannot
 * tell them apart). Every name of the list has a counter of its own: the external names are distinct, and no name meets a built-in rule's or an annotation
 * counter's ({@link zeroFusionCounts}, the check the pass applies). Anything else fails loudly
 * (`ExecutionError`), before the Session uploads a byte.
 */
export const sessionFusionRules = (
  fusionRules: unknown,
): readonly FusionRule<FusionRuleName | ExternalRuleName>[] => {
  if (fusionRules === undefined) return FUSION_RULES;
  if (!Array.isArray(fusionRules)) {
    throw new ExecutionError(
      `options.fusionRules must be an array of rules built by defineRule of @karume/runtime/kernel (got ${describeInputValue(fusionRules)})`,
    );
  }
  const external: FusionRule<ExternalRuleName>[] = [];
  for (const [at, rule] of fusionRules.entries()) {
    if (!isExternalRule(rule)) {
      throw new ExecutionError(
        `options.fusionRules[${at}] ${
          ruleOrigin(rule) === "built-in"
            ? "is a built-in rule (the built-in rules are always on)"
            : "is not a rule built by defineRule of @karume/runtime/kernel: either an object defineRule did not make" +
              " (a hand-written object or a copy of a rule), or a rule defined by another copy of @karume/runtime in" +
              " the dependency tree (a rule is bound to the runtime copy that defined it — install one copy)"
        }`,
      );
    }
    if (external.some((given) => given.name === rule.name)) {
      throw new ExecutionError(`options.fusionRules: the rule '${rule.name}' is given twice`);
    }
    external.push(rule);
  }
  const rules: readonly FusionRule<FusionRuleName | ExternalRuleName>[] = [
    ...external,
    ...FUSION_RULES,
  ];
  zeroFusionCounts(rules);
  return rules;
};

/**
 * The rules the scan tries: the context's list without the disabled ones
 * ({@link FusionContext.disabledRules}), in list order.
 */
const triedRules = <Name extends string>(
  context: FusionContext<Name>,
): readonly FusionRule<Name>[] => {
  const disabled = context.disabledRules;
  assertDisabledRulesNamed(context.rules, disabled);
  if (disabled === undefined) return context.rules;
  if (disabled === "all") return [];
  return context.rules.filter((rule) => !disabled.has(rule.name));
};

/**
 * Checks, on the scan that gives the final plan, that an external rule's step leaves every packed
 * activation pair (ADR 0105) as the first scan decided it: no node it folds produces or consumes a
 * packed value. `window` is the hit's window (an external window starts at the index it was tried
 * at, so it is the run the scan covers).
 *
 * MUST: the pairing is decided on the first scan and holds only if the second scan folds the same
 * nodes. An external matcher reads no packed value ({@link FusionMatchContext}), but nothing makes
 * its answer the same on both scans (a matcher that keeps state can differ). Folding a packed
 * producer leaves its reader taking f32 words as packed int8 codes; folding a consumer, or binding
 * the value, makes the external kernel read packed words as f32. Both are silent wrong values, so
 * the final plan fails loudly here, naming the rule.
 */
const assertPackedPairsUntouched = (
  window: readonly NodePlan[],
  hit: FusionHit,
  packed: ReadonlyMap<string, number>,
): void => {
  const kept = new Set([...hit.passthrough, ...hit.deferred]);
  for (const plan of window) {
    if (kept.has(plan)) continue;
    const touched = [...plan.outputs.map((out) => out.name), ...plan.node.ins].find((name) =>
      packed.has(name),
    );
    if (touched !== undefined) {
      throw new ExecutionError(
        `fusion rule '${hit.step.rule}': its step folds the '${plan.node.op}' node that ` +
          `${plan.node.ins.includes(touched) ? "reads" : "writes"} '${touched}', a value handed over as packed int8` +
          " activations (ADR 0105) on the first scan; an external step may neither take nor produce one",
      );
    }
  }
};

/**
 * Scans the node list once and folds it into fused steps. Nodes not grabbed stay plain.
 *
 * `rules` are the rules tried at each index ({@link triedRules}); the counter record has one counter
 * per rule of `context.rules` all the same, so a disabled rule reads 0.
 */
const scanFusions = <Name extends string>(
  nodes: readonly NodePlan[],
  context: FusionContext<Name>,
  rules: readonly FusionRule<Name>[],
  packed: ReadonlyMap<string, number>,
): FusionPlan<Name> => {
  const steps: ExecStep[] = [];
  const aliases = planAliases(nodes);
  const scanContext: FusionScanContext = {
    ...context,
    packedValues: packed,
    matchContext: fusionMatchContext(context),
  };
  const counts = zeroFusionCounts(context.rules);
  counts.packedStaticQuantize = packed.size;
  // The same record keyed by the run-time name a hit carries ({@link FusedStep.rule}).
  const ruleCounts: Record<string, number> = counts;
  const countHit = (rule: string): void => {
    // MUST: a hit counts under a counter of the plan's list. A name without one would count NaN
    // silently (a rule built outside `defineRule` can return a step under any name).
    if (!Object.hasOwn(ruleCounts, rule)) {
      throw new ExecutionError(
        `fusion rule '${rule}': the step's rule name has no counter (the plan's rules are` +
          ` ${context.rules.map((entry) => `'${entry.name}'`).join(", ")})`,
      );
    }
    ruleCounts[rule] += 1;
  };
  const pushNode = (plan: NodePlan): void => {
    const alias = aliases.has(plan);
    if (alias && plan.contract.kind === "expand") counts.identityExpand += 1;
    steps.push({ kind: "node", plan, aliasesInput: alias, ...packedRole(plan, packed) });
  };
  // The node list as scanned: a hit that defers nodes writes them into its window's last slots and
  // the scan resumes at the first of them (the "Deferred nodes" section), so the rules, the linear
  // prologue and the attention permutes read this list. `aliases` may come from `nodes`: a node's
  // alias decision reads only its own input, which precedes it in either order.
  const scanned = [...nodes];
  for (let index = 0; index < scanned.length;) {
    const hit = rules.reduce<FusionHit | undefined>(
      (found, rule) => found ?? rule.apply(scanned, index, scanContext),
      undefined,
    );
    if (hit !== undefined) {
      if (packed.size > 0 && isExternalRuleName(hit.step.rule)) {
        assertPackedPairsUntouched(scanned.slice(index, index + hit.advance), hit, packed);
      }
      // MUST: passthroughs first (the fused step may take a passthrough's output as input).
      for (const plan of hit.passthrough) pushNode(plan);
      steps.push(hit.step);
      countHit(hit.step.rule);
      const resume = index + hit.advance - hit.deferred.length;
      hit.deferred.forEach((plan, offset) => {
        scanned[resume + offset] = plan;
      });
      index = resume;
      continue;
    }
    const folded = linearPrologueAt(scanned, index, context);
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
    const absorbed = attentionPermutesAt(scanned, index, context, aliases);
    if (absorbed !== undefined) {
      // The absorbed permutes get no step; the attention's step stands for all of them. An attention
      // never aliases (planAliases) and is never a packed producer or consumer.
      steps.push({
        kind: "node",
        plan: absorbed.plan,
        aliasesInput: false,
        tokenMajor: absorbed.tokenMajor,
      });
      counts.attentionPermute += absorbedPermutes(absorbed.tokenMajor);
      index += absorbed.advance;
      continue;
    }
    pushNode(scanned[index]);
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
 * Under `integerGemv`, an integer GEMV linear ({@link linearIntegerGemvVariant} — again the one
 * function the recipe builder's guard reads) also reads packed: the integer kernel takes nothing
 * else.
 * Under `linearGemvPlanes`, a byte-plane GEMV linear ({@link linearPlanesGemvVariant} — the function
 * the recipe builder's guard reads) never reads packed, whatever the tables below say: its kernel
 * binds f32 quads only. Checked first, so the option decides the linear's route on its own.
 */
const readsPackedActivation = (plan: NodePlan, name: string, context: FusionContext): boolean => {
  if (plan.node.op !== "linear" || plan.node.ins.length !== 3) return false;
  if (plan.node.ins[0] !== name || plan.node.ins[1] === name || plan.node.ins[2] === name) {
    return false;
  }
  if (!allF32([plan])) return false;
  const weight = context.weightLayouts?.get(plan.node.ins[1]);
  if (weight === undefined || weight.storage === "f16" || weight.storage === "bf16") return false;
  if (linearPlanesGemvVariant(context.linearGemvPlanes, plan, weight.storage) !== undefined) {
    return false;
  }
  const group = weight.storage === "i4" ? weight.groupSize : undefined;
  if (linearIntegerGemvVariant(context.integerGemv, plan, weight.storage, group) !== undefined) {
    return true;
  }
  const m = numel(plan.inputShapes[0].slice(0, -1));
  const [n, k] = plan.inputShapes[1];
  return linearGemvPackedEligible(weight.storage, m, n, k, group) !== undefined;
};

/**
 * The fixed SRQs handed over as packed int8 activations (ADR 0105) — value name → that SRQ's f32
 * scale.
 *
 * Five conditions for acceptance:
 *
 * 1. The option (`packedStaticQuantize`) is set, with `linearGemvReduce: parallel` /
 *    `linearCompute: f32` (only the parallel GEMV family has packed variants; the integer GEMV,
 *    which reads nothing else, is admitted only on top of these settings).
 * 2. A `static_quantize` **left as a plain node**. An SRQ swallowed by the linear→SRQ fusion
 *    (ADR 0103) is an output-side epilogue, and packed output is out of scope here, so it does not
 *    qualify — which is why this decision takes the result of the first scan (`steps`) as input.
 * 3. The output is not a graph output, the last dimension k is a multiple of 16 (the binding reads
 *    `vec4<u32>` = units of 16 elements), and the scale is positive and finite (scale 0 is an
 *    identity SRQ and cannot be lowered to int8 codes).
 * 4. There is at least one consumer, and **every one of them** is the activation slot of a
 *    parallel-GEMV linear of a form where packed was measured to help, or (under `integerGemv`) of
 *    an integer GEMV linear ({@link readsPackedActivation}). If even one GEMM / sequential GEMV /
 *    other op / form measured not to help is mixed in, it stays f32.
 * 5. No fused step other than {@link LINEAR_STATIC_QUANTIZE_RULE} consumes the value.
 *
 * MUST: 5 is an invariant for **future rules**. Only the linearStaticQuantize fusion has a
 * spelling that reads packed, so the moment another rule puts a linear into its window, that
 * fused kernel reads the packed words as f32 (a silent wrong value with no exception). No window of
 * the current ten rules contains a `linear`, so it does not fire today; but without it on the
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
 * grab the same window** — the pairing's input never moves in the second scan. The built-in rules
 * keep this by construction; an external rule's matcher sees no packed value, and its step on the
 * second scan is checked to leave every pair alone ({@link assertPackedPairsUntouched}).
 */
export const planFusions = <Name extends string>(
  nodes: readonly NodePlan[],
  context: FusionContext<Name>,
): FusionPlan<Name> => {
  const rules = triedRules(context);
  const first = scanFusions(nodes, context, rules, NO_PACKED);
  const packed = packedStaticQuantizeValues(nodes, first.steps, context);
  return packed.size === 0 ? first : scanFusions(nodes, context, rules, packed);
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
  /**
   * The chain's nodes themselves, in window order (the nodes of {@link UnfusedWindow.ops}; the
   * in-window passthroughs are the window's other nodes). A consumer reads the shapes of the values
   * a fold would remove from them, instead of re-deriving the chain outside the fusion-pass layer.
   */
  readonly chain: readonly NodePlan[];
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
 * MUST: do not build windows across a fused step, a linear with a folded producer
 * ({@link LinearPrologue}) or an attention that absorbed permutes ({@link AttentionTokenMajor}) — a
 * chain already grabbed is not counted twice as a candidate. The scan closes per **consecutive run**
 * of plain nodes.
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
          chain,
          windowSize: size,
          nodeIndex: runStart + start,
          outputName: chain[chain.length - 1].outputs[0].name,
        });
      }
    }
    run = [];
  };
  for (const step of steps) {
    if (step.kind === "node" && nodeStepSpan(step) === 1) {
      if (run.length === 0) runStart = nodeIndex;
      run.push(step.plan);
      nodeIndex += 1;
      continue;
    }
    sweepRun();
    // A node step that absorbed its neighbours (a folded producer, absorbed permutes) is a grabbed
    // window of several original nodes, like a fused step.
    nodeIndex += step.kind === "node" ? nodeStepSpan(step) : step.nodeCount;
  }
  sweepRun();
  return found;
};
