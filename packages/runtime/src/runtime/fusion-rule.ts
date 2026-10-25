/**
 * The shared surface of the fusion rules: execution step types, **the eligibility conditions shared
 * by every rule**, and `defineRule`.
 *
 * The rules themselves live under `fusion-rules/` (one rule = one file); the declaration table and
 * the scan are in {@link "./fusion.ts"}. MUST: keep the value dependencies **one-directional**:
 * fusion-rule → fusion-rules → fusion (do not import `fusion.ts` from here — the declaration table
 * imports the rules, so that would be a cycle).
 */

import type { PrologueUnaryOp } from "../codegen/elementwise.ts";
import { outputCountOf, WHERE_OP } from "../ops.ts";
import type { LinearGemvReduce } from "./session-types.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";

/**
 * Fusion rule identifiers (one-to-one with the declaration order of `FUSION_RULES` in
 * {@link "./fusion.ts"}).
 */
type FusionRuleName =
  | "linearStaticQuantize"
  | "rmsNormAdd"
  | "silu"
  | "upsample2x"
  | "rope"
  | "adaln"
  | "gatedResidual"
  | "rowBlockAttention";

/**
 * The diagnostic counter names. Besides the fusion rules, they count the identity expands among the
 * zero-dispatch aliases that **can drop out conditionally** (`aliasesInput` of {@link ExecStep}),
 * the fixed SRQs handed over as packed int8 activations (ADR 0105), the producers folded into a
 * linear's step ① ({@link LinearPrologue} — one per folded pair), and the permutes an attention
 * absorbs ({@link AttentionTokenMajor} — one per absorbed permute, so up to four per attention).
 * Reshape aliasing is unconditional and is not counted (it cannot drop out = nothing to observe).
 */
export type FusionCounterName =
  | FusionRuleName
  | "identityExpand"
  | "packedStaticQuantize"
  | "linearPrologue"
  | "attentionPermute";

/**
 * Application count per rule. The only observation point for "a fusion silently dropped out and
 * only performance fell".
 */
export type FusionCounts = Readonly<Record<FusionCounterName, number>>;

/**
 * What a fused step may bind.
 *
 * MUST: only this type can point at a step-internal temporary ({@link FusedStep.temps}); external
 * inputs are pointed at by **index** into {@link FusedStep.binds} (the executor does not re-resolve
 * value names).
 */
export type FusedOperand =
  | { readonly kind: "bind"; readonly index: number }
  | { readonly kind: "temp"; readonly id: number }
  | { readonly kind: "weightScale"; readonly index: number }
  | { readonly kind: "output" };

/**
 * How a dispatch's workgroup count is chosen.
 *
 * - `gridStride` = divide the element / row coverage (elementwise, reduce, single-dispatch
 *   fusions). Beyond the limit it is clamped, and the kernel's grid-stride loop covers the rest.
 * - `tiled` = the GEMM family with **one workgroup = one output tile**. It cannot be clamped onto a
 *   grid-stride loop, so exceeding the limit fails loudly on the declaring side
 *   (`tiledWorkgroups`).
 */
type FusedWorkgroups =
  | { readonly kind: "gridStride"; readonly items: number; readonly size: number }
  | { readonly kind: "tiled"; readonly counts: readonly [number, number, number] };

/** The generation input for one dispatch of a fused kernel (the shape shared by every rule). */
export type FusedDispatch = {
  readonly key: string;
  /**
   * MUST: the same key always yields byte-identical WGSL (the PipelineCache determinism contract).
   * It is a thunk because the body still has to be compared even when cached, while the plan does
   * not need to hold every step's string.
   */
  readonly wgsl: () => string;
  /**
   * Params at binding 0. Uniform by default (each kernel allocates in 16-byte units — the shared
   * form of ADR 0040).
   */
  readonly params: Uint32Array<ArrayBuffer>;
  /**
   * Bind params as **storage** (only the elementwise family with variable-length params — the same
   * as `buildElementwise` for a plain node). Uniform when omitted.
   */
  readonly paramsStorage?: boolean;
  /**
   * The operand list from binding 1 on. **Only single-dispatch rules may omit it**; it then becomes
   * "{@link FusedStep.binds} in declaration order → the output last" (the shape shared by
   * single-dispatch fusions).
   */
  readonly operands?: readonly FusedOperand[];
  readonly workgroups: FusedWorkgroups;
};

/**
 * The allocation spec of a step-internal temporary. Shape and meaning match `TempRecipe` in
 * recipe.ts; **the lifetime is given as dispatch-boundary indices**.
 *
 * MUST: every declared temporary has a release boundary (when `releaseAfter` comes before the
 * allocation, the declaration's entry point (`validateStepRecipe`) rejects it).
 */
export type FusedTemp = {
  readonly byteLength: number;
  /** Allocated **right before** this index of {@link FusedStep.dispatches}. */
  readonly allocBefore: number;
  /** Released **right after** this index of {@link FusedStep.dispatches}. */
  readonly releaseAfter: number;
};

/**
 * A fused step (one run of consecutive original nodes replaced by a list of private-kernel
 * dispatches).
 *
 * A step **may have several** dispatches. A rule whose folded result is not one dispatch
 * (`ROW_BLOCK_ATTENTION_RULE` in {@link "./fusion-rules/row-block-attention.ts"}, and the
 * per-token form of `ADALN_RULE` in {@link "./fusion-rules/adaln.ts"}) lines up several of them
 * with temporaries closed inside the step ({@link FusedStep.temps}) in between. MUST: even then, the release bookkeeping rests on the one total-use list
 * {@link FusedStep.ins}, and not a single temporary leaks past the step boundary.
 */
export type FusedStep = {
  readonly kind: "fused";
  readonly rule: FusionRuleName;
  /**
   * The total-use list (duplicates included) of the **external** inputs the original nodes
   * consumed. Internal values are never materialised, so they are not included. MUST: it never
   * differs from the original total-use count by even one (the only basis of the arena's reference
   * counting).
   */
  readonly ins: readonly string[];
  /** Operand order of the bind group (from binding 1; no duplicates). */
  readonly binds: readonly string[];
  /**
   * The number of folded nodes. **Not the scan width** — for a rule with in-window passthroughs,
   * the window width ({@link FusionHit.advance}) is larger.
   */
  readonly nodeCount: number;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** Temporaries closed inside the step (lifetime = indices into {@link FusedStep.dispatches}). */
  readonly temps: readonly FusedTemp[];
  readonly dispatches: readonly FusedDispatch[];
};

/**
 * Handing over packed int8 activations (ADR 0105) — the declaration of a pair that passes a fixed
 * SRQ's output as one u32 word = four int8 codes. `role` is **which side this node is**, and
 * `scale` is, on either side, **the producer SRQ's f32 scale** (the consumer needs it to restore
 * with `f32(code) * scale`).
 *
 * MUST: acceptance of a pair is decided in one place, `planFusions` in {@link "./fusion.ts"}.
 * Setting only the producer or only the consumer
 * lets f32 words flow into a `vec4<u32>` binding (a silent wrong value with no exception).
 */
export type PackedActivations = {
  readonly role: "write" | "read";
  readonly scale: number;
};

/**
 * A producer folded into a linear's step ① (perf-ledger F-4): the linear's i8a8 path quantizes
 * `op(x)` directly instead of the stored output of an elementwise `op` node, so that output is never
 * written or read.
 *
 * The node step's `plan` is the linear's with its first input **re-pointed to the producer's input**
 * (the same shape and dtype), and the producer has no step of its own. The release bookkeeping
 * therefore stays on the one path of a plain node (`plan.node.ins`): the producer's input loses the
 * producer's use and gains the linear's, and the folded value (one use, not a graph output) is
 * never defined.
 * MUST: acceptance is decided in one place, `planFusions` in {@link "./fusion.ts"}, with the same
 * route predicate as the recipe (`linearTakesI8a8` in {@link "./linear-route.ts"}). A linear that
 * carries it but does not take the i8a8 path would run without the folded op — a silent wrong
 * value — so the recipe checks the same predicate again and fails loudly.
 */
export type LinearPrologue = {
  readonly op: PrologueUnaryOp;
};

/**
 * The operands of a fused attention that are read or written **token-major** (perf-ledger F-3 A4 /
 * F-8): the attention absorbed the `[0, 2, 1, 3]` permute that turned a token-major `[B, S, H, D]`
 * tensor into its head-major `[B, H, S, D]` operand (`q` / `k` / `v`), or the one that turned its
 * head-major output back (`o`), so the permute's copy is never materialised as a node output.
 *
 * The node step's `plan` is the attention's with each absorbed operand **re-pointed to the permute's
 * input** (name, token-major shape and dtype) and, with `o`, its output re-pointed to the output
 * permute's output (name and `[B, M, H, D]` shape). The absorbed permutes have no step, so the release
 * bookkeeping stays the one path of a plain node (`plan.node.ins`): each permute's input trades the
 * permute's use for the attention's, and the absorbed values (one use, not graph outputs) are never
 * defined.
 * MUST: acceptance is decided in one place, `planFusions` in {@link "./fusion.ts"}, and it needs no
 * route: the attention recipe honours every flag on every route (the a8 quantisation reads a
 * token-major q / k / v itself, the online kernel stores O token-major, and any other route makes the
 * head-major copy the permute made, as a node temporary). A node other than a fused attention that
 * carries it would read the operands in the wrong layout — a silent wrong value — so it fails loudly.
 */
export type AttentionTokenMajor = {
  readonly q: boolean;
  readonly k: boolean;
  readonly v: boolean;
  readonly o: boolean;
};

/** The number of permutes an attention absorbed (the `attentionPermute` counter's increment). */
export const absorbedPermutes = (tokenMajor: AttentionTokenMajor): number =>
  [tokenMajor.q, tokenMajor.k, tokenMajor.v, tokenMajor.o].filter(Boolean).length;

type NodeStep = {
  readonly kind: "node";
  readonly plan: NodePlan;
  /**
   * Make the output an alias of the input buffer (zero dispatches — ADR 0011). Always true for
   * reshape; for permute only when the element order does not change and the storage is allocated
   * internally. For expand, true only when the bound input and output shapes match exactly,
   * rank included (= an identity map with no replicated axis).
   */
  readonly aliasesInput: boolean;
  /** The packed int8 activation pair (ADR 0105). `undefined` = handed over as f32, as before. */
  readonly packedActivations?: PackedActivations;
  /**
   * The producer folded into this linear's step ① (perf-ledger F-4). `undefined` = the input is
   * read as stored, as before. One node step then stands for **two** original nodes (the producer
   * and the linear, adjacent in node order).
   */
  readonly prologue?: LinearPrologue;
  /**
   * The permutes this attention absorbed (perf-ledger F-3 A4 / F-8). `undefined` = every operand is
   * read and written head-major as declared, as before. One node step then stands for the absorbed
   * input permutes (the run immediately before it in node order), the attention and, with `o`, the
   * output permute right after it.
   */
  readonly tokenMajor?: AttentionTokenMajor;
};

/**
 * The number of original (pre-fusion) nodes a plain node step stands for: 1, plus the producer a
 * linear folded ({@link LinearPrologue}) or the permutes an attention absorbed
 * ({@link AttentionTokenMajor}). The nodes are contiguous in node order.
 */
export const nodeStepSpan = (step: NodeStep): number =>
  1 +
  (step.prologue === undefined ? 0 : 1) +
  (step.tokenMajor === undefined ? 0 : absorbedPermutes(step.tokenMajor));

export type ExecStep = NodeStep | FusedStep;

export type FusionPlan = {
  readonly steps: readonly ExecStep[];
  readonly counts: FusionCounts;
};

/**
 * The device capabilities the decision needs (**the granted limit values themselves**).
 *
 * MUST: what goes here is "input that decides the plan", not measurements. Measuring at run time
 * and choosing again (autotuning) is banned by ADR 0022 — "the same device and the same bindings
 * always give the same plan" is the premise of the prepared-plan cache and the meaning of its key.
 */
export type FusionLimits = {
  /** The limit of one storage binding. The only device-side input deciding the row-block count. */
  readonly maxStorageBufferBindingSize: number;
  /** The workgroup-count limit per axis (for failing loudly on tiled dispatches). */
  readonly maxComputeWorkgroupsPerDimension: number;
};

/**
 * Resident storage info without a GPU object. The lifetime and ownership of the scale stay with the
 * Session.
 */
export type FusionWeightLayout =
  | { readonly storage: "f16" | "bf16" | "i2" | "i8" }
  | { readonly storage: "i4"; readonly groupSize: number };

/** Whole-graph facts the decision needs (passed from the executor's Session state). */
export type FusionContext = {
  /** Value name → consumer count in the graph (countUses in plan.ts). */
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
  readonly limits: FusionLimits;
  /**
   * Forces the row-block count (**test only** — the executor's `ROW_BLOCK_SPLIT`). Splits into this
   * count instead of the smallest count that fits the limit. A count that does not fit the limit
   * fails loudly.
   */
  readonly rowBlockSplit?: number;
  /**
   * The budget per row-block temporary (`SessionOptions.attentionRowBlockMaxBytes` — ADR 0124).
   * The count's limit is the smaller of {@link FusionLimits.maxStorageBufferBindingSize} and this
   * value (`rowBlockLimit`). `limits` itself is not lowered (decisions other than row blocks keep
   * the granted values). Omitted = no budget.
   */
  readonly attentionRowBlockMaxBytes?: number;
  /** Fuse RMS→add only when explicitly requested (ADR 0099). */
  readonly fuseRmsNormAdd?: boolean;
  readonly fuseLinearStaticQuantize?: boolean;
  /** Hand fixed-SRQ activations to the parallel GEMV as packed int8 (ADR 0105). */
  readonly packedStaticQuantize?: boolean;
  readonly linearGemvReduce?: LinearGemvReduce;
  readonly linearCompute?: "f32" | "f16" | "a8";
  readonly weightLayouts?: ReadonlyMap<string, FusionWeightLayout>;
  readonly rmsNormReduce?: "workgroup" | "subgroup32";
};

/**
 * The context a rule sees = the caller's {@link FusionContext} + the packed activation pairs
 * **decided inside the scan** (value name → the producer SRQ's scale).
 *
 * MUST: the caller does not declare `packedValues` (`planFusions` in {@link "./fusion.ts"} derives
 * it from the result of the first
 * scan — only SRQs left as plain nodes qualify, so it cannot be decided before the scan).
 */
export type FusionScanContext = FusionContext & {
  readonly packedValues: ReadonlyMap<string, number>;
};

export const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((dim, index) => dim === b[index]);

/**
 * Fusion handles only nodes whose **contract has exactly one output** (ADR 0068 decision 1).
 *
 * MUST: check every node of the window, and give up on the whole window if even one fails (if it
 * does not match, the existing list of plain nodes is always correct). {@link FusedStep} has a
 * single output, and both the folding of internal values ({@link externalIns} /
 * {@link internalsArePrivate}) and the hand-over of the last output (`outputName` of `build`) are
 * derived from "one node = one value". A multi-output node in the chain would make the unfolded
 * outputs vanish with nobody defining them, and one in an in-window passthrough would be moved
 * forward as a plain node; the matcher's spelling alone prevents neither.
 * MUST: this is the only decision point ({@link defineRule} runs every rule through it). Leaving it
 * to each rule's match would let only a rule whose window happens to match a future topk / argmax
 * spelling fuse wrongly, silently.
 */
export const windowIsSingleOutput = (window: readonly NodePlan[]): boolean =>
  window.every((step) => outputCountOf(step.contract) === 1);

/**
 * Whether the window contains even one **node that touches state** (ADR 0067 decision 5b).
 *
 * MUST: a state reference draws no tensor data edge, so when fusion moves an in-window passthrough
 * forward and breaks the `nodes` array order, neither the shape checks nor the reference counting
 * fail — the form "read this step's k/v twice as the past" comes out with no exception.
 * MUST: this is the only decision point (the same place as {@link windowIsSingleOutput}). Leaving
 * it to each rule's match would make one omission in a future rule a silent wrong value.
 */
export const windowTouchesState = (window: readonly NodePlan[]): boolean =>
  window.some((step) => Object.keys(step.node.states).length > 0);

/** Whether every node of the chain is f32 only (every fused kernel is fixed to f32). */
export const allF32 = (chain: readonly NodePlan[]): boolean =>
  chain.every(
    (step) =>
      step.outputs.every((out) => out.dtype === "f32") &&
      step.inputDtypes.every((dtype) => dtype === "f32"),
  );

/**
 * Whether a node is a `where` that selects between f32 values under a bool condition — the only
 * non-f32 input a fused kernel reads (the condition as the u32 0 / 1 of ADR 0009, tested `!= 0u`
 * as the plain where does). Every other node of a chain still goes through {@link allF32}.
 */
export const isF32Where = (step: NodePlan): boolean =>
  step.node.op === WHERE_OP &&
  step.inputDtypes.length === 3 &&
  step.inputDtypes[0] === "bool" &&
  step.inputDtypes[1] === "f32" &&
  step.inputDtypes[2] === "f32" &&
  step.outputs.every((out) => out.dtype === "f32");

/**
 * Whether every internal value of the chain (the outputs of every node but the last) has "exactly
 * one consumer and is not a graph output".
 *
 * MUST: after fusion no buffer for an internal value exists, so a single external consumer or
 * readback would lose the value. This is the eligibility condition shared by every rule.
 * MUST: look at **every output** of a node (do not rely on whether {@link windowIsSingleOutput}
 * rejects multi-output nodes first — keep the conclusion independent of the check order).
 */
export const internalsArePrivate = (chain: readonly NodePlan[], context: FusionContext): boolean =>
  chain
    .slice(0, -1)
    .every((step) =>
      step.outputs.every(
        (out) => (context.useCounts.get(out.name) ?? 0) === 1 && !context.outputNames.has(out.name),
      ),
    );

/**
 * The total-use list of the external inputs the chain consumed (duplicates included, in the order
 * of the original node.ins). Internal values are folded away and excluded (every output of those
 * nodes is an internal value).
 */
const externalIns = (chain: readonly NodePlan[]): readonly string[] => {
  const internal = new Set(
    chain.slice(0, -1).flatMap((step) => step.outputs.map((out) => out.name)),
  );
  return chain.flatMap((step) => step.node.ins.filter((name) => !internal.has(name)));
};

/**
 * Whether the in-window passthroughs may move **in front of** the fused step.
 *
 * MUST: a passthrough consumes no value the chain defines (internal values **and the last
 * output**). If it did, a "node that reads a value not yet computed" would be placed first and the
 * reordering would be illegal (the executor only encodes the steps in order, so unless this rejects
 * it, the result is a "no buffer for value 'x'" error or, if the name is reused, a silent wrong
 * value).
 *
 * NOTE: for the two current rules with in-window passthroughs (adaln / rope), the only chain values
 * a passthrough can read are **internal values** (the last output is defined after the passthrough,
 * so it cannot be read), and {@link internalsArePrivate} (exactly one consumer) rejects those first
 * — **a counterexample cannot be built on its own** (confirmed by fault injection in
 * tests/runtime_fusion_test.ts). This check takes effect independently only for a future rule with
 * a window where "the **last** node of the chain comes before the passthrough"; internalsArePrivate
 * does not look at the last output, so it is no substitute. It sits in `defineRule` as an invariant
 * of the window mechanism.
 */
export const passthroughIsIndependent = (
  chain: readonly NodePlan[],
  passthrough: readonly NodePlan[],
): boolean => {
  if (passthrough.length === 0) return true;
  const defined = new Set(chain.flatMap((step) => step.outputs.map((out) => out.name)));
  return passthrough.every((step) => step.node.ins.every((name) => !defined.has(name)));
};

/** The result of applying one fusion rule (fused step + in-window passthroughs + scan width). */
export type FusionHit = {
  /** In-window nodes placed as plain nodes **before** the fused step (original node order). */
  readonly passthrough: readonly NodePlan[];
  readonly step: FusedStep;
  /** How far the scan advances (= the window's node count = passthroughs + the folded chain). */
  readonly advance: number;
};

/** The body of a rule (match grabs, build lowers to a step declaratively). */
export type FusionRule = {
  readonly name: FusionRuleName;
  /**
   * The leading ops this rule can grab (declared so that the pairwise disjointness of the
   * application order is checked mechanically).
   */
  readonly heads: readonly string[];
  readonly apply: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => FusionHit | undefined;
};

/**
 * The window a rule grabbed. `window` is a run of consecutive nodes, `chain` the subsequence of it
 * that is **folded** (actual node order); the difference is the in-window passthroughs.
 */
export type FusionMatch = {
  /** The consecutive window (the only basis of the scan width). */
  readonly window: readonly NodePlan[];
  /** The folded nodes (a subsequence of `window`). */
  readonly chain: readonly NodePlan[];
};

/**
 * Whether the lifetime declarations of step-internal temporaries close inside the dispatch list.
 *
 * MUST: `allocBefore ≤ releaseAfter < dispatch count`. A declaration outside that becomes, in the
 * executor's replay, either "bind a temporary that was never allocated" or "a temporary that is
 * never released"; the former cannot build a bind surface, and the latter goes unnoticed until the
 * plan's closure check. Rather than have each rule write it by hand,
 * it is rejected at the one entry point for declarations.
 */
const assertTempLifetimes = (
  name: FusionRuleName,
  temps: readonly FusedTemp[],
  dispatchCount: number,
): void => {
  temps.forEach((temp, id) => {
    if (
      !Number.isSafeInteger(temp.byteLength) ||
      temp.byteLength < 1 ||
      temp.allocBefore < 0 ||
      temp.releaseAfter < temp.allocBefore ||
      temp.releaseAfter >= dispatchCount
    ) {
      throw new ExecutionError(
        `fusion rule '${name}': temporary ${id} has a declared lifetime [${temp.allocBefore}, ${temp.releaseAfter}] that` +
          ` does not close within the ${dispatchCount} dispatches (${temp.byteLength}B)`,
      );
    }
  });
};

/**
 * Enforces by type the separation of match (grab) and build (declare binds / kernel key / params).
 *
 * MUST: the basis of the release bookkeeping (the total-use list `ins`), the folded count
 * (`nodeCount`), the scan width (`advance`) and the passthroughs are **derived from the grabbed
 * window and chain**. Having each rule declare them would copy the same fact once per
 * rule, and a single drift would raise no exception.
 * MUST: a window containing a multi-output node is rejected here for every rule
 * ({@link windowIsSingleOutput}).
 * MUST: a window containing a node that touches state is rejected at the same place
 * ({@link windowTouchesState}).
 */
export const defineRule = <Matched extends FusionMatch>(rule: {
  readonly name: FusionRuleName;
  readonly heads: readonly string[];
  readonly match: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => Matched | undefined;
  readonly build: (matched: Matched) => Omit<FusedStep, "kind" | "rule" | "ins" | "nodeCount">;
}): FusionRule => ({
  name: rule.name,
  heads: rule.heads,
  apply: (nodes, index, context) => {
    const matched = rule.match(nodes, index, context);
    if (matched === undefined) return undefined;
    // Multi-output nodes (ADR 0068 decision 1) are not fusion candidates. The matcher's op-name
    // spelling only narrows the accepted set and does not guarantee "one output", so there is one
    // place that checks every node of the window against its contract.
    if (!windowIsSingleOutput(matched.window)) return undefined;
    // Nodes that touch state (ADR 0067 decision 5b) are not fusion candidates — the ban on
    // reordering is an invariant of the window mechanism and cannot be expressed by the matcher's
    // op-name spelling.
    if (windowTouchesState(matched.window)) return undefined;
    const folded = new Set(matched.chain);
    const passthrough = matched.window.filter((step) => !folded.has(step));
    // MUST: the chain is a subsequence of the window. Otherwise it folded a node outside the window
    // (= the scan width is short, so it runs twice) or missed one inside it (= never runs); both
    // are silent wrong values with no exception.
    if (passthrough.length + matched.chain.length !== matched.window.length) {
      throw new ExecutionError(
        `fusion rule '${rule.name}': the folded chain of ${matched.chain.length} nodes is not a subsequence of the window of ${matched.window.length} nodes`,
      );
    }
    if (!passthroughIsIndependent(matched.chain, passthrough)) return undefined;
    const built = rule.build(matched);
    assertTempLifetimes(rule.name, built.temps, built.dispatches.length);
    return {
      passthrough,
      advance: matched.window.length,
      step: {
        kind: "fused",
        rule: rule.name,
        ins: externalIns(matched.chain),
        nodeCount: matched.chain.length,
        ...built,
      },
    };
  },
});
