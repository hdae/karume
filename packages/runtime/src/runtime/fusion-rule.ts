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
import type { ScoreStorage } from "../kernels/score-storage.ts";
import { outputCountOf, WHERE_OP } from "../ops.ts";
import type { AttentionDeviceFacts } from "./attention-route.ts";
import type { LinearGemvReduce } from "./session-types.ts";
import { describeInputValue, ExecutionError, type NodePlan } from "./plan.ts";

/**
 * The names of the built-in rules — the type of the built-in list `FUSION_RULES` in
 * {@link "./fusion.ts"}, which checks at compile time that the two agree in both directions (every
 * name has its rule in the list, every rule of the list has a name of this union). A rule's own name
 * is a string ({@link FusionRule.name}): the list a plan uses is an input ({@link FusionContext.rules}).
 */
export type FusionRuleName =
  | "linearStaticQuantize"
  | "geluMulStaticQuantize"
  | "rmsNormAdd"
  | "silu"
  | "upsample2x"
  | "rope"
  | "rmsNormRope"
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
export type FusionCounterName = FusionRuleName | AnnotationCounterName;

/**
 * The counters that do not belong to a rule: each counts an annotation of a plain node step
 * (`aliasesInput` of an expand, {@link PackedActivations}, {@link LinearPrologue},
 * {@link AttentionTokenMajor}). Every plan counts them, whatever its rule list.
 */
export type AnnotationCounterName =
  | "identityExpand"
  | "packedStaticQuantize"
  | "linearPrologue"
  | "attentionPermute";

/**
 * Application count per rule. The only observation point for "a fusion silently dropped out and
 * only performance fell".
 *
 * Every built-in counter is present. A rule written outside the runtime (`SessionOptions.fusionRules`
 * — its name is `<namespace>/<rule>`, {@link ExternalRuleName}) has a counter under its name in the
 * plans of a Session that was given it, and none otherwise, so its key is optional.
 */
export type FusionCounts = Readonly<Record<FusionCounterName, number>> & {
  readonly [rule: ExternalRuleName]: number | undefined;
};

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
export type FusedWorkgroups =
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
  /** The name of the rule that built the step ({@link FusionRule.name}). */
  readonly rule: string;
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
 * MUST: acceptance is decided in one place, `planFusions` in {@link "./fusion.ts"}. `q` / `k` / `v`
 * from the attention seat alone ({@link FusionContext.attentionCompute} `"a8"`), not from the route:
 * the attention recipe honours them on every route an a8 seat can take (the a8 quantisation reads a
 * token-major q / k / v itself, and a stage the shape keeps off i8a8 makes the head-major copy the
 * permute made, as a node temporary). `o` only where `attentionRoute` chooses the online kernel for
 * the node, the one route that stores O token-major itself; any other route that receives `o` fails
 * loudly (no O temporary or copy path exists). A node other than a fused attention that carries the
 * annotation would read the operands in the wrong layout — a silent wrong value — so it fails loudly.
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

/**
 * The result of the fusion pass. `counts` holds one counter per rule name of the plan's rule list
 * (`Name` — {@link FusionContext.rules}) and the {@link AnnotationCounterName}s; with the built-in
 * list (the default) it is {@link FusionCounts}.
 */
export type FusionPlan<Name extends string = FusionRuleName> = {
  readonly steps: readonly ExecStep[];
  readonly counts: Readonly<Record<Name | AnnotationCounterName, number>>;
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
export type FusionContext<Name extends string = string> = {
  /**
   * The rules the scan tries at each index, in this order (the application order). A Session passes
   * the built-in list (`FUSION_RULES` in {@link "./fusion.ts"}); the plan counts one counter per
   * name of this list ({@link FusionPlan.counts}).
   */
  readonly rules: readonly FusionRule<Name>[];
  /**
   * The rules of {@link rules} the scan does not try (**test only** — the executor's
   * `DISABLED_FUSION_RULES`): `"all"`, or the names of the disabled rules. Omitted = every rule is
   * tried. A disabled rule keeps its counter, at 0; the annotation folds
   * ({@link AnnotationCounterName}) are not rules and are unaffected.
   * MUST: every name is a name of {@link rules} (`assertDisabledRulesNamed` in
   * {@link "./fusion.ts"} — a misspelt name fails loudly instead of leaving its rule on).
   */
  readonly disabledRules?: "all" | ReadonlySet<string>;
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
  /** Fuse gelu_tanh → mul → static_quantize only when explicitly requested (perf-ledger K-48). */
  readonly fuseGeluMulStaticQuantize?: boolean;
  /** Hand fixed-SRQ activations to the parallel GEMV as packed int8 (ADR 0105). */
  readonly packedStaticQuantize?: boolean;
  /**
   * Run the eligible decode linears on the integer GEMV (`SessionOptions.integerGemv`); which
   * linears is decided by `linearIntegerGemvVariant` in {@link "./linear-route.ts"} alone.
   */
  readonly integerGemv?: boolean;
  /**
   * Run the eligible i2 linears on the byte-plane GEMV (`SessionOptions.linearGemvPlanes`); which
   * linears is decided by `linearPlanesGemvVariant` in {@link "./linear-route.ts"} alone. Such a
   * linear reads f32 activations and keeps its own dispatch, so the packed pairing and the
   * linear→SRQ rule leave it alone.
   */
  readonly linearGemvPlanes?: boolean;
  readonly linearGemvReduce?: LinearGemvReduce;
  readonly linearCompute?: "f32" | "f16" | "a8";
  /**
   * The Session's attention seat (`SessionOptions.attentionCompute` — a session knob, no device
   * facts). Only `"a8"` lets a fused attention absorb the permutes of its q / k / v
   * ({@link AttentionTokenMajor}); omitted, nothing is absorbed.
   */
  readonly attentionCompute?: "f32" | "f16" | "a8";
  /**
   * The storage form of S (`SessionOptions.attentionScoreStorage` — a session knob; omitted =
   * `"f32"`). With {@link attentionCompute} and {@link attentionDevice} it is what `attentionRoute`
   * reads to decide whether a fused attention takes the online kernel, the only route on which it
   * absorbs the permute after it (`o` of {@link AttentionTokenMajor}).
   */
  readonly attentionScoreStorage?: ScoreStorage;
  /**
   * The device facts of the online kernel's route (ADR 0132 decision 5 — the executor passes the ones
   * it fixes once per Session for `buildAttention`, the estimate `EstimateOptions.attentionDevice`).
   * Device facts in the context have a precedent in {@link limits}: they are inputs that decide the
   * plan, fixed per device, not measurements.
   * MUST: present when the seat is `attentionCompute` `"a8"` × `attentionScoreStorage` `"none"`, the
   * only seat on which the route reads them; a decision that needs them fails loudly without them
   * (guessing a device would decide the output permute for a route the device may not take). Not
   * read otherwise.
   */
  readonly attentionDevice?: AttentionDeviceFacts;
  readonly weightLayouts?: ReadonlyMap<string, FusionWeightLayout>;
  readonly rmsNormReduce?: "workgroup" | "subgroup32";
};

/**
 * What the matcher of a rule written outside the runtime reads (`match` of
 * {@link ExternalRuleDefinition} — the third argument of a matcher written against
 * `@karume/runtime/kernel`): the graph facts a rule is written with, and nothing a plan decides.
 *
 * MUST: no Session knob, no rule list and no packed-activation decision enters it. A matcher that
 * read the knobs would make the public type change with every new planning knob, one that read the
 * rule list could call the built-in rules, and one that read the packed values could take a
 * different window in the second scan than in the first (`planFusions` in {@link "./fusion.ts"}),
 * where the pairing decided on the first scan would then hand packed words to the wrong reader.
 * The scan builds it once ({@link FusionScanContext.matchContext}).
 */
export type FusionMatchContext = {
  /** Value name → consumer count in the graph (each use counted, so a node reading a value twice counts 2). */
  readonly useCounts: ReadonlyMap<string, number>;
  /** The graph outputs (read back after a run, so a fold may not leave them unmaterialised). */
  readonly outputNames: ReadonlySet<string>;
};

/**
 * The context a rule sees = the caller's {@link FusionContext} + the packed activation pairs
 * **decided inside the scan** (value name → the producer SRQ's scale) + the narrowed context the
 * external rules' matchers read.
 *
 * MUST: the caller does not declare `packedValues` (`planFusions` in {@link "./fusion.ts"} derives
 * it from the result of the first
 * scan — only SRQs left as plain nodes qualify, so it cannot be decided before the scan).
 */
export type FusionScanContext = FusionContext & {
  readonly packedValues: ReadonlyMap<string, number>;
  /**
   * What an external rule's matcher receives ({@link FusionMatchContext}, {@link fusionMatchContext}),
   * built once per scan. The built-in rules read the whole context and ignore it.
   */
  readonly matchContext: FusionMatchContext;
};

/** The {@link FusionMatchContext} of a plan's context: its graph facts, frozen. */
export const fusionMatchContext = (context: FusionContext): FusionMatchContext =>
  Object.freeze({ useCounts: context.useCounts, outputNames: context.outputNames });

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
 * readback would lose the value. This is the eligibility condition shared by every rule: a rule
 * declares it (`internals: "private"`) and {@link defineRule} applies it to the matched chain.
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
 * {@link internalsArePrivate} for a chain that reads an internal value **more than once**: every
 * use of every internal value (counted with repeats) is an input slot of a node of the chain, there
 * is at least one, and none is a graph output. The fold loses no value for the same reason (no
 * consumer outside the chain, no readback).
 *
 * MUST: only a rule whose matcher pins **every** slot an internal value may occupy declares this
 * (`internals: "stayInChain"`, applied by {@link defineRule}), and its bind surface holds no internal
 * value. Counted reads alone do not know which slot reads the value:
 * under this predicate `add(n, n)` after `rms_norm → n` would pass, and the rmsNormAdd matcher would
 * bind `n` — a value fusion never materialises — as its residual. That is why it does not replace
 * {@link internalsArePrivate} for every rule (those matchers rely on "one use" to keep an internal
 * value out of their bind surfaces).
 */
export const internalsStayInChain = (
  chain: readonly NodePlan[],
  context: FusionContext,
): boolean => {
  const reads = new Map<string, number>();
  for (const step of chain) {
    for (const name of step.node.ins) reads.set(name, (reads.get(name) ?? 0) + 1);
  }
  return chain.slice(0, -1).every((step) =>
    step.outputs.every((out) => {
      const uses = context.useCounts.get(out.name) ?? 0;
      return uses > 0 && uses === reads.get(out.name) && !context.outputNames.has(out.name);
    }),
  );
};

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
 * NOTE: for the two built-in rules with in-window passthroughs (adaln / rope), the only chain values
 * a passthrough can read are **internal values** (the last output is defined after the passthrough,
 * so it cannot be read), and the privacy both declare (`internals: "private"` —
 * {@link internalsArePrivate}, exactly one consumer), which `defineRule` applies before this check,
 * rejects those first — **a counterexample cannot be built from those rules** (confirmed once by
 * fault injection against tests/runtime_fusion_test.ts; no standing test exercises this predicate
 * alone). This check takes effect independently only for a rule with a window where "the **last**
 * node of the chain comes before the passthrough"; internalsArePrivate does not look at the last
 * output, so it is no substitute. It sits in `defineRule` as an invariant of the window mechanism.
 * The same check holds for the **deferred** nodes ({@link FusionMatch.others}): they run after the
 * fused step, but an internal value is never materialised, so they may read none either (the
 * {@link internalsStayInChain} privacy rmsNormRope declares rejects such a read first, for the same
 * reason).
 */
export const passthroughIsIndependent = (
  chain: readonly NodePlan[],
  passthrough: readonly NodePlan[],
): boolean => {
  if (passthrough.length === 0) return true;
  const defined = new Set(chain.flatMap((step) => step.outputs.map((out) => out.name)));
  return passthrough.every((step) => step.node.ins.every((name) => !defined.has(name)));
};

/**
 * Whether the deferred nodes of a window ({@link FusionMatch.others}) may move **behind** the fused
 * step: no node of the chain consumes a value they define.
 *
 * MUST: the fused step takes the window's place and the deferred nodes follow it, so a chain node
 * that read a deferred node's output would read a value not yet computed (the same failure as
 * {@link passthroughIsIndependent}'s, in the other direction). Together the two checks make the
 * chain and the deferred nodes independent both ways, so their relative order is free.
 */
export const deferralIsIndependent = (
  chain: readonly NodePlan[],
  deferred: readonly NodePlan[],
): boolean => {
  if (deferred.length === 0) return true;
  const defined = new Set(deferred.flatMap((step) => step.outputs.map((out) => out.name)));
  return chain.every((step) => step.node.ins.every((name) => !defined.has(name)));
};

/** The result of applying one fusion rule (fused step + the window's other nodes + scan width). */
export type FusionHit = {
  /** In-window nodes placed as plain nodes **before** the fused step (original node order). */
  readonly passthrough: readonly NodePlan[];
  /**
   * In-window nodes placed **after** the fused step and **scanned again** (original node order) —
   * the scan writes them into the window's last slots and resumes at the first of them, so one of
   * them may head another window ({@link FusionMatch.others}).
   */
  readonly deferred: readonly NodePlan[];
  readonly step: FusedStep;
  /**
   * The window's node count (passthroughs + the folded chain + the deferred nodes). The scan moves
   * past the window, less the deferred nodes it visits again.
   */
  readonly advance: number;
};

/**
 * The contract class a rule declares (ADR 0147 decision 2) — the form of the evidence for its
 * numbers. `"E"`: u32-identical to the unfused plan of the same graph (the class of every built-in
 * rule — the classification table of ADR 0147 decision 2). `"C"`, `"R"` and `"Q"` are the classes
 * of a numerics-changing variant (composition, addition order, approximation) and follow the
 * checks ADR 0147 assigns to them.
 */
export type ContractClass = "E" | "C" | "R" | "Q";

/** The body of a rule (match grabs, build lowers to a step declaratively). */
export type FusionRule<Name extends string = string> = {
  /** The rule's counter name ({@link FusionPlan.counts}) and {@link FusedStep.rule}. */
  readonly name: Name;
  /**
   * The leading ops this rule can grab (declared so that the pairwise disjointness of the
   * application order is checked mechanically among the built-in rules). An external rule is tried
   * only at a node whose op is one of them.
   */
  readonly heads: readonly string[];
  /** The contract class the rule declares ({@link ContractClass}). */
  readonly contract: ContractClass;
  readonly apply: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => FusionHit | undefined;
};

/**
 * The window a rule grabbed. `window` is a run of consecutive nodes, `chain` the subsequence of it
 * that is **folded** (actual node order); the difference is the in-window passthroughs, or the
 * deferred nodes ({@link FusionMatch.others}).
 */
export type FusionMatch = {
  /** The consecutive window (the only basis of the scan width). */
  readonly window: readonly NodePlan[];
  /** The folded nodes (a subsequence of `window`). */
  readonly chain: readonly NodePlan[];
  /**
   * Where the window's other nodes go. Omitted: **before** the fused step, as in-window
   * passthroughs (the chain may read them). `"deferred"`: **after** it, where the scan visits them
   * again ({@link FusionHit.deferred}) — for a chain interleaved with nodes of another chain (the Wan
   * DiT's q and k prologues), whose head would otherwise be emitted as a plain passthrough and never
   * fold. Legal only when the chain and those nodes are independent both ways
   * ({@link passthroughIsIndependent} and {@link deferralIsIndependent}, checked by `defineRule`).
   * MUST: defer only nodes that cannot head a step the estimate keeps in post-fusion form (a
   * row-block attention's `bmm`, a linear-prologue producer, a permute an attention absorbs).
   * `postFusionSteps` in estimate.ts places those steps by the running node count, which inside a
   * deferred region is the position in the scanned list, not in the declaration order.
   */
  readonly others?: "deferred";
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
  name: string,
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
 * What a rule's `build` declares: the bind surface, the output and the dispatches with their
 * temporaries. The rest of a {@link FusedStep} (`ins`, `nodeCount`, the rule's name) is derived from
 * the grabbed window ({@link defineRule}).
 */
export type FusedBuild = Omit<FusedStep, "kind" | "rule" | "ins" | "nodeCount">;

/**
 * What a define entry collects for one rule; `MatchContext` is what its matcher reads
 * ({@link FusionRuleDefinition} inside the runtime, {@link ExternalRuleDefinition} outside it).
 */
type RuleDefinition<Matched extends FusionMatch, Name extends string, MatchContext> = {
  readonly name: Name;
  readonly heads: readonly string[];
  /**
   * The contract class of the rule's numbers ({@link ContractClass}, ADR 0147 decision 2). A rule
   * u32-identical to the unfused plan declares `"E"`.
   */
  readonly contract: ContractClass;
  /**
   * Who may read the chain's internal values: `"private"` = each has exactly one consumer and is
   * not a graph output ({@link internalsArePrivate}); `"stayInChain"` = every use is inside the
   * chain, possibly more than one ({@link internalsStayInChain} — only for a matcher that pins every
   * slot an internal value may occupy).
   */
  readonly internals: "private" | "stayInChain";
  /** Grabs the window that starts at `nodes[index]`, or returns `undefined`. */
  readonly match: (
    nodes: readonly NodePlan[],
    index: number,
    context: MatchContext,
  ) => Matched | undefined;
  /** Lowers a grabbed window to its dispatches. */
  readonly build: (matched: Matched) => FusedBuild;
};

/** What {@link defineRule} collects for a built-in rule: its matcher reads the whole scan context. */
export type FusionRuleDefinition<
  Matched extends FusionMatch,
  Name extends string = string,
> = RuleDefinition<Matched, Name, FusionScanContext>;

/**
 * What {@link defineExternalRule} collects for a rule written outside the runtime (the
 * `FusionRuleDefinition` of `@karume/runtime/kernel`): its matcher reads the narrowed
 * {@link FusionMatchContext}.
 */
export type ExternalRuleDefinition<
  Matched extends FusionMatch,
  Name extends ExternalRuleName = ExternalRuleName,
> = RuleDefinition<Matched, Name, FusionMatchContext>;

/**
 * The brand of a rule built by {@link defineRule} or {@link defineExternalRule}: a property keyed by
 * this module-private symbol, non-enumerable and read-only, on a frozen rule. A Session takes only
 * branded external rules (`sessionFusionRules` in {@link "./fusion.ts"}), so a hand-written object —
 * or a copy of a rule with its `apply` replaced — cannot reach the scan around the checks this file
 * applies. The brand is a value on the rule, not an entry in a module-level table (no global mutable
 * state). It guards against accidents, not against a caller who reads the symbol off a rule on
 * purpose.
 */
const RULE_BRAND: unique symbol = Symbol("karume.fusionRule");

/** Where a branded rule was defined: inside the runtime, or outside it ({@link defineExternalRule}). */
export type RuleOrigin = "built-in" | "external";

/** The brand of `value` ({@link RULE_BRAND}), or `undefined` for anything not built by a define entry. */
export const ruleOrigin = (value: unknown): RuleOrigin | undefined => {
  if (typeof value !== "object" || value === null) return undefined;
  const brand: unknown = Object.getOwnPropertyDescriptor(value, RULE_BRAND)?.value;
  return brand === "built-in" || brand === "external" ? brand : undefined;
};

/**
 * The name of a rule defined outside the runtime: `<namespace>/<rule>`, each segment a lowercase
 * identifier (`[a-z][a-z0-9_]*`). The built-in names have no slash, so the two never meet.
 */
export type ExternalRuleName = `${string}/${string}`;

const EXTERNAL_RULE_NAME = /^([a-z][a-z0-9_]*)\/[a-z][a-z0-9_]*$/;

/** The namespace of an external rule's name, or `undefined` when the name is not of that form. */
export const externalNamespace = (name: unknown): string | undefined =>
  typeof name === "string" ? EXTERNAL_RULE_NAME.exec(name)?.[1] : undefined;

/** Whether `name` is an external rule's name ({@link ExternalRuleName}, the full pattern). */
export const isExternalRuleName = (name: unknown): name is ExternalRuleName =>
  externalNamespace(name) !== undefined;

/**
 * Whether `value` is a rule {@link defineExternalRule} built: branded external, with an external
 * rule's name (the name is read from the frozen rule itself, not through the type).
 */
export const isExternalRule = (value: unknown): value is FusionRule<ExternalRuleName> =>
  ruleOrigin(value) === "external" &&
  typeof value === "object" &&
  value !== null &&
  isExternalRuleName(Object.getOwnPropertyDescriptor(value, "name")?.value);

const CONTRACT_CLASSES: Readonly<Record<ContractClass, true>> = {
  E: true,
  C: true,
  R: true,
  Q: true,
};

/**
 * The checks of a definition's declarations (a JavaScript caller can pass any value; a declaration
 * read wrongly would decide the privacy of every chain the rule folds, and a `match` or `build` that
 * is not a function would fail as an unnamed `TypeError` at the first run instead of here).
 */
const assertDeclarations = (
  rule: Omit<RuleDefinition<FusionMatch, string, never>, "match" | "build"> & {
    readonly match: unknown;
    readonly build: unknown;
  },
): void => {
  const heads: unknown = rule.heads;
  if (
    !Array.isArray(heads) ||
    heads.length === 0 ||
    !heads.every((head) => typeof head === "string" && head.length > 0)
  ) {
    throw new ExecutionError(
      `fusion rule '${String(rule.name)}': heads must be a non-empty array of op names, got ${Array.isArray(heads) ? `[${heads.map(describeInputValue).join(", ")}]` : describeInputValue(heads)}`,
    );
  }
  for (const field of ["match", "build"] as const) {
    if (typeof rule[field] !== "function") {
      throw new ExecutionError(
        `fusion rule '${String(rule.name)}': ${field} must be a function, got ${describeInputValue(rule[field])}`,
      );
    }
  }
  const contract: unknown = rule.contract;
  if (typeof contract !== "string" || !Object.hasOwn(CONTRACT_CLASSES, contract)) {
    throw new ExecutionError(
      `fusion rule '${String(rule.name)}': contract must be "E", "C", "R" or "Q" (ADR 0147 decision 2), got ${String(contract)}`,
    );
  }
  const internals: unknown = rule.internals;
  if (internals !== "private" && internals !== "stayInChain") {
    throw new ExecutionError(
      `fusion rule '${String(rule.name)}': internals must be "private" or "stayInChain", got ${String(internals)}`,
    );
  }
};

/**
 * The window checks of an external rule, after its matcher grabbed `matched` at `index` (a built-in
 * rule is written to the same invariants and pinned by its own tests).
 *
 * MUST: the window is the run of nodes that starts at `nodes[index]` and holds at least the chain's
 * nodes. The scan places the fused step at `index` and moves on by the window's length, so a window
 * that starts elsewhere or skips a node would drop that node or run another twice — silent wrong
 * values with no exception — and an empty one would never move on. MUST: the chain follows the
 * window's node order. The step's inputs and output are derived from the chain in its order (the
 * last node's value is the output, earlier values are internal), so a reversed chain would list
 * its own output among its inputs and never define the value its consumers read.
 */
const assertExternalWindow = (
  name: string,
  nodes: readonly NodePlan[],
  index: number,
  matched: FusionMatch,
): void => {
  const contiguous = matched.window.every((step, offset) => step === nodes[index + offset]);
  if (matched.window.length === 0 || matched.chain.length === 0 || !contiguous) {
    throw new ExecutionError(
      `fusion rule '${name}': the window of ${matched.window.length} nodes (chain ${matched.chain.length}) is not` +
        ` the run of nodes that starts at node ${index} (an external rule's window starts where it was tried,` +
        " holds every node it covers, and folds at least one)",
    );
  }
  const positions = matched.chain.map((step) => matched.window.indexOf(step));
  if (!positions.every((at, offset) => at > (offset === 0 ? -1 : positions[offset - 1]))) {
    throw new ExecutionError(
      `fusion rule '${name}': the folded chain does not follow the window's node order (its nodes sit at` +
        ` window positions ${positions.join(", ")})`,
    );
  }
};

/**
 * The build checks of an external rule: its declared surface agrees with the chain it folds.
 *
 * MUST: every bind is an external input of the chain, once. The release bookkeeping rests on the
 * chain's inputs alone (`ins`), so a bind outside them could be released, and its buffer reused,
 * before the step reads it — a silent wrong value. MUST: the output is the chain's last value with
 * its shape (consumers read that name, and the buffer is sized from the shape), and there is at
 * least one dispatch to write it. MUST: every dispatch key starts with `ext:<namespace>:`. The
 * pipeline cache is one per device and shared by every Session; the prefix keeps an external rule's
 * keys out of the built-in kernels' and another namespace's (two different WGSL bodies under one key
 * still fail as `PipelineKeyConflictError`, the last line of defence).
 */
const assertExternalBuild = (
  name: string,
  namespace: string,
  chain: readonly NodePlan[],
  built: FusedBuild,
): void => {
  const inputs = new Set(externalIns(chain));
  const strayBinds = built.binds.filter(
    (bind, at) => !inputs.has(bind) || built.binds.indexOf(bind) !== at,
  );
  if (strayBinds.length > 0) {
    throw new ExecutionError(
      `fusion rule '${name}': bind(s) '${strayBinds.join("', '")}' are not distinct external inputs of the folded chain` +
        ` (its inputs are ${[...inputs].map((input) => `'${input}'`).join(", ")})`,
    );
  }
  const last = chain[chain.length - 1].outputs[0];
  if (built.outputName !== last.name || !sameShape(built.outputShape, last.shape)) {
    throw new ExecutionError(
      `fusion rule '${name}': the declared output '${built.outputName}' [${built.outputShape.join(",")}] is not the` +
        ` folded chain's last value '${last.name}' [${last.shape.join(",")}]`,
    );
  }
  if (built.dispatches.length === 0) {
    throw new ExecutionError(`fusion rule '${name}': the step declares no dispatch`);
  }
  const prefix = `ext:${namespace}:`;
  const strayKeys = built.dispatches
    .map((dispatch) => dispatch.key)
    .filter((key) => typeof key !== "string" || !key.startsWith(prefix));
  if (strayKeys.length > 0) {
    throw new ExecutionError(
      `fusion rule '${name}': dispatch key(s) '${strayKeys.map(String).join("', '")}' do not start with '${prefix}'` +
        " (an external rule's pipeline keys live in its own namespace)",
    );
  }
};

/**
 * The rest of one try of a rule at `index` ({@link FusionRule.apply} of {@link brandedRule}'s
 * rules), once its matcher grabbed `matched`, on the frozen copy of its definition
 * ({@link snapshotDefinition}).
 */
const applyRule = <Matched extends FusionMatch, Name extends string>(
  rule: RuleDefinition<Matched, Name, never>,
  namespace: string | undefined,
  nodes: readonly NodePlan[],
  index: number,
  context: FusionScanContext,
  matched: Matched | undefined,
): FusionHit | undefined => {
  if (matched === undefined) return undefined;
  if (namespace !== undefined) {
    assertExternalWindow(rule.name, nodes, index, matched);
    // An external rule folds only f32 chains, checked here and not left to its matcher: every
    // fused kernel reads and writes f32 words, so a chain with another dtype that a matcher forgot
    // to refuse would read them as f32 silently. Such a window is not taken (the unfused plan is
    // always correct), as the built-in matchers decline it.
    if (!allF32(matched.chain)) return undefined;
  }
  // After fusion no buffer for an internal value exists, so a consumer outside the chain or a
  // readback would lose it.
  const internalsKept =
    rule.internals === "private"
      ? internalsArePrivate(matched.chain, context)
      : internalsStayInChain(matched.chain, context);
  if (!internalsKept) return undefined;
  // Multi-output nodes (ADR 0068 decision 1) are not fusion candidates. The matcher's op-name
  // spelling only narrows the accepted set and does not guarantee "one output", so there is one
  // place that checks every node of the window against its contract.
  if (!windowIsSingleOutput(matched.window)) return undefined;
  // Nodes that touch state (ADR 0067 decision 5b) are not fusion candidates — the ban on
  // reordering is an invariant of the window mechanism and cannot be expressed by the matcher's
  // op-name spelling.
  if (windowTouchesState(matched.window)) return undefined;
  const folded = new Set(matched.chain);
  const others = matched.window.filter((step) => !folded.has(step));
  // MUST: the chain is a subsequence of the window. Otherwise it folded a node outside the window
  // (= the scan width is short, so it runs twice) or missed one inside it (= never runs); both
  // are silent wrong values with no exception.
  if (others.length + matched.chain.length !== matched.window.length) {
    throw new ExecutionError(
      `fusion rule '${rule.name}': the folded chain of ${matched.chain.length} nodes is not a subsequence of the window of ${matched.window.length} nodes`,
    );
  }
  if (!passthroughIsIndependent(matched.chain, others)) return undefined;
  const deferring = matched.others === "deferred";
  if (deferring && !deferralIsIndependent(matched.chain, others)) return undefined;
  const built = rule.build(matched);
  assertTempLifetimes(rule.name, built.temps, built.dispatches.length);
  if (namespace !== undefined) assertExternalBuild(rule.name, namespace, matched.chain, built);
  return {
    passthrough: deferring ? [] : others,
    deferred: deferring ? others : [],
    advance: matched.window.length,
    step: {
      kind: "fused",
      rule: rule.name,
      ins: externalIns(matched.chain),
      nodeCount: matched.chain.length,
      ...built,
    },
  };
};

/**
 * The definition as it was when a define entry ran: each field read once from the caller's object,
 * checked, and kept in a frozen copy.
 *
 * MUST: the rule reads only this copy. The caller's object stays theirs to edit (deriving a variant
 * by editing a shared definition is ordinary JavaScript), and a rule that read it live would take
 * the later name — crediting its hits to another counter — and the later `internals`, `match` and
 * `build`, past the checks run here.
 */
const snapshotDefinition = <Matched extends FusionMatch, Name extends string, MatchContext>(
  rule: RuleDefinition<Matched, Name, MatchContext>,
): RuleDefinition<Matched, Name, MatchContext> => {
  const { name, heads, contract, internals, match, build } = rule;
  assertDeclarations({ name, heads, contract, internals, match, build });
  return Object.freeze({
    name,
    heads: Object.freeze([...heads]),
    contract,
    internals,
    match,
    build,
  });
};

/**
 * The rule a frozen definition describes, branded built-in or, with `namespace` (its name's
 * namespace), external — the window and build checks of an external rule then apply. `match` runs
 * the definition's matcher on the context it reads.
 */
const brandedRule = <Matched extends FusionMatch, Name extends string>(
  definition: RuleDefinition<Matched, Name, never>,
  namespace: string | undefined,
  match: (
    nodes: readonly NodePlan[],
    index: number,
    context: FusionScanContext,
  ) => Matched | undefined,
): FusionRule<Name> => {
  const defined: FusionRule<Name> = {
    name: definition.name,
    heads: definition.heads,
    contract: definition.contract,
    apply: (nodes, index, context) =>
      applyRule(definition, namespace, nodes, index, context, match(nodes, index, context)),
  };
  const origin: RuleOrigin = namespace === undefined ? "built-in" : "external";
  Object.defineProperty(defined, RULE_BRAND, { value: origin });
  return Object.freeze(defined);
};

/**
 * Enforces by type the separation of match (grab) and build (declare binds / kernel key / params).
 *
 * MUST: the basis of the release bookkeeping (the total-use list `ins`), the folded count
 * (`nodeCount`), the scan width (`advance`) and the passthroughs or deferred nodes are **derived
 * from the grabbed window and chain**. Having each rule declare them would copy the same fact once per
 * rule, and a single drift would raise no exception.
 * MUST: a window containing a multi-output node is rejected here for every rule
 * ({@link windowIsSingleOutput}).
 * MUST: a window containing a node that touches state is rejected at the same place
 * ({@link windowTouchesState}).
 * MUST: the privacy of the folded chain's internal values is applied here, to `matched.chain`, by
 * the rule's declaration (`internals`), so no rule can forget it ({@link internalsArePrivate} /
 * {@link internalsStayInChain}). A matcher may check it earlier as well, where code after the check
 * must not see a chain that is not private (a fail-loudly step, or a matcher shared with a rule that
 * declares the other privacy).
 * The rule comes back frozen and branded ({@link RULE_BRAND}) as a built-in rule. A built-in rule
 * passes its name as `Name` too (`defineRule<Match, "silu">`), so the built-in list can check its
 * names at compile time; any other rule leaves it a string.
 */
export const defineRule = <Matched extends FusionMatch, Name extends string = string>(
  rule: FusionRuleDefinition<Matched, Name>,
): FusionRule<Name> => {
  const definition = snapshotDefinition(rule);
  return brandedRule(definition, undefined, definition.match);
};

/**
 * {@link defineRule} for a rule written outside the runtime — the `defineRule` of
 * `@karume/runtime/kernel`. The name is `<namespace>/<rule>` ({@link ExternalRuleName}), checked
 * here; the rule is branded external, so a Session takes it through `SessionOptions.fusionRules`;
 * it is tried only at a node whose op is one of its `heads`, its matcher reads the narrowed
 * {@link FusionMatchContext}, and a window whose chain is not all f32 is not taken; and besides every
 * check of
 * {@link defineRule}, its window must start where it was tried, its build must bind only the chain's
 * own inputs and write the chain's last value, and every dispatch key must start with
 * `ext:<namespace>:` (checked each time the rule builds a step).
 */
export const defineExternalRule = <
  Matched extends FusionMatch,
  Name extends ExternalRuleName = ExternalRuleName,
>(
  rule: ExternalRuleDefinition<Matched, Name>,
): FusionRule<Name> => {
  const definition = snapshotDefinition(rule);
  const namespace = externalNamespace(definition.name);
  if (namespace === undefined) {
    throw new ExecutionError(
      `fusion rule '${String(definition.name)}': an external rule's name is '<namespace>/<rule>', two lowercase` +
        " identifiers ([a-z][a-z0-9_]*) joined by one slash",
    );
  }
  // MUST: an external rule is tried only at a node whose op it declares in `heads`, so the heads
  // say which built-in rules it can pre-empt (it is tried before them — ADR 0148 decision 3).
  const heads = new Set(definition.heads);
  return brandedRule(definition, namespace, (nodes, index, context) => {
    const op = nodes[index]?.node.op;
    return op !== undefined && heads.has(op)
      ? definition.match(nodes, index, context.matchContext)
      : undefined;
  });
};
