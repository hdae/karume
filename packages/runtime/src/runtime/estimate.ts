/**
 * Memory demand estimator (ADR 0070 decision 5) — a pure function that never touches the GPU and
 * reports the "demand side" bytes by category.
 *
 * MUST: **no comparison with the free side**. WebGPU exposes neither total nor free VRAM, so the
 * moment it became a comparison it would be "guessing where no budget is available" (ADR 0070
 * decision 5 — the same precedent as llama.cpp's exclusion of 0/0 devices). This returns numbers
 * only and makes no go / no-go decision at all.
 * MUST: the estimate is **not an absolute guarantee**. {@link AdmissionReport.unaccounted} admits in
 * its format what is not accounted for, and the real **final check remains the out-of-memory
 * errorScope** (the `pushFailureScopes` pair in src/gpu/error-scope.ts) — the estimator is an
 * advance diagnostic for Session construction / `createGenerationContext` and does not stop
 * execution when exceeded.
 *
 * The report has two levels, **resident + scenarios** (a refinement of ADR 0070 decision 5). What
 * the Session holds over its lifetime (weights, state slots) is the single
 * {@link AdmissionReport.resident} group, and what changes per run shape (prefill / decode) is
 * listed in {@link AdmissionReport.scenarios} as **separate computations**. Folding them into one
 * number would either push a peak of intermediates that only the prefill shape produces onto a
 * caller that only runs decode, or conversely let prefill pass on decode's numbers.
 * Shapes that are neither of the two (the verify run of speculative decoding — ADR 0096) are added
 * by the caller through {@link AdmissionScenarioSpec} **to the same computation path** (the caller
 * never rebuilds the formulas).
 *
 * Correspondence with the measurements ({@link SessionDiagnostics}):
 * - {@link AdmissionReport.resident}`.weights.compressedBytes` ↔ `storage.residentCompressedBytes`
 * - {@link AdmissionReport.resident}`.weights.expandedBytes` ↔ `storage.hostExpandedBytes`
 * - {@link AdmissionReport.resident}`.weights.uncompressedBytes` — no corresponding diagnostic
 *   field (the `storage` diagnostics of ADR 0006 count only low-precision storage. The measured
 *   total is observable only through `weights.allocatedBytes`, which includes params)
 * - {@link AdmissionReport.resident}`.stateBytes` ↔ `stateBacking.residentBytes` (with one context)
 * - {@link AdmissionScenario.workspaceBytes} ↔ `planBacking.residentBytes` /
 *   `lastRun.peakTransientBytes` (of a run executed in that scenario's shape — the diagnostic
 *   `residentBytes` is **the sum over all kept backings**, so it corresponds only for a Session that
 *   ran one shape or when `retainedCount === 1`. Budgeted keeping is ADR 0095 decision 1)
 *
 * MUST: take both the classification and the formulas **from the same sources as the
 * implementation** (the weights' residency class and declared bytes from the pure planner in
 * weight-residency.ts, alignment and size classes from `toSizeClass`, the state's bytes per element
 * and the logical-length uniform from the constants in generation-context.ts, the axis that carries
 * the physical chunk rows `M` from the op contracts in ops/shapes.ts). Rewriting the formulas here
 * would let the estimator keep claiming a different number, with no exception or warning, against
 * an implementation fixed on one side only.
 */

import { parseDim, solveDim } from "../format/dims.ts";
import type { IrDim, IrGraph } from "../format/ir.ts";
import { toSizeClass } from "../gpu/arena.ts";
import { parseStorageRoles } from "../gpu/pipeline-cache.ts";
import {
  groupNormAttrs,
  numel,
  RUNTIME_SUPPORT,
  stateReadonly,
  stateWindow,
  topkK,
} from "../ops.ts";
import { groupNormLayout, groupNormTempBytes } from "../kernels/group-norm.ts";
import { assertRuntimeSupport } from "../ops/support.ts";
import {
  type AttentionTokenMajor,
  type ExecStep,
  type FusedStep,
  FUSION_RULES,
  type FusionWeightLayout,
  nodeStepSpan,
  planAliases,
  planFusions,
  rowBlockLimit,
} from "./fusion.ts";
import {
  allowedChunkRows,
  assertChunkBuckets,
  assertChunkLength,
  assertSlotWithinLimits,
  LENGTHS_BYTES,
  resolveBindings,
  resolveSlotShape,
  type SlotLimits,
  slidingSlotWindows,
  stateSlotBytes,
} from "./generation-context.ts";
import {
  assertGenerationBindings,
  countUses,
  describeInputValue,
  ExecutionError,
  type NodePlan,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
  validateGraphContracts,
} from "./plan.ts";
import {
  CORE_TRANSIENT_LIMITS,
  planTransients,
  type TransientDispatchSpec,
  type TransientOutputSpec,
  type TransientRef,
  type TransientStepSpec,
  type TransientTempSpec,
} from "./transient-plan.ts";
import {
  type ComputePrecision,
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type GenerationContextSpec,
  STATE_ATTENTION_REDUCES,
  type StateAttentionReduce,
} from "./session-types.ts";
import {
  argmaxSplitGroups,
  argmaxSplitPartialBytes,
  topkOneSplitGroups,
} from "../kernels/argmax.ts";
import { assertChunkRowsAllowed } from "./recipe.ts";
import {
  planStateAttention,
  type StateAttentionBlock,
  stateFusesStatsPv,
} from "./state-attention-plan.ts";
import {
  type AttentionDeviceFacts,
  attentionRoute,
  type AttentionSeats,
  UNREAD_ATTENTION_DEVICE,
} from "./attention-route.ts";
import {
  attentionHeadMajorCopies,
  attentionHeadMajorShapes,
  attentionI8a8TempBytes,
  attentionScoreElementBytes,
  fusedAttentionViolation,
  planAttentionBlocks,
  planAttentionOnlineWindows,
  stateAttentionVariantViolation,
} from "./recipe-builders/attention.ts";
import { sessionOptionsViolation } from "./session-options-check.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import { stateSliding } from "../kernels/state-attention.ts";
import { slotStateStorage, type StateStorage } from "../kernels/state-storage.ts";
import { planWeightBuffers, type WeightResidency } from "./weight-residency.ts";
import { planExpertGraph } from "./expert-plan.ts";

/**
 * The **reserved-name** scenarios. An estimate without `generation` has the single `"run"`; with it,
 * the two execution shapes of ADR 0066 decision 4 (`"prefill"` / `"decode"`). Callers cannot use
 * these three for {@link AdmissionScenarioSpec.name}.
 */
export type AdmissionScenarioName = "run" | "prefill" | "decode";

/**
 * The spec of an extra scenario added by the caller ({@link EstimateOptions}`.generation.scenarios`).
 *
 * An entry point that adds one more computation on **the same computation path** as the existing two
 * shapes (prefill / decode), with only the physical chunk rows replaced. It exists so that callers
 * can get the demand of a shape that is neither of the two shapes of ADR 0066 decision 4 — like the
 * verify run of speculative decoding (`queryLength = k+1` rows in one pass — ADR 0096) — without
 * rebuilding the formulas.
 */
export type AdmissionScenarioSpec = {
  /**
   * The name in the report. A duplicate of a reserved name (the three of
   * {@link AdmissionScenarioName}) or of another extra name in the same list fails loudly (no form
   * where a caller looking up by name silently sees only the first).
   */
  readonly name: string;
  /**
   * The physical chunk rows `M` of this shape. An integer in `1..generation.chunkLength` (outside
   * fails loudly — no claiming a row count outside the `queryLength ≤ chunkLength` contract), and
   * also one of the row counts the context allows (1, `generation.chunkBuckets`,
   * `generation.chunkLength`; outside is a shape the run rejects, so it fails loudly too).
   */
  readonly chunkLength: number;
  /**
   * **Overrides** of graph-input symbolic dimensions (laid over a copy of `options.bindings`). If
   * omitted, `options.bindings` as is. The range and binding-point checks are the same as for the
   * existing two shapes (states-only symbols and symbols of the physical chunk rows are not
   * accepted).
   */
  readonly bindings?: SymbolBindings;
};

/**
 * The bytes needed by one run shape (one element of {@link AdmissionReport.scenarios}).
 *
 * Only **what changes with the run shape** goes here — weights and state slots do not depend on the
 * shape, so they are on the {@link AdmissionReport.resident} side.
 */
export type AdmissionScenario = {
  /**
   * A reserved name ({@link AdmissionScenarioName}) or an {@link AdmissionScenarioSpec.name} added
   * by the caller.
   */
  readonly name: string;
  /** Graph input buffers + output readback staging (exact). */
  readonly ioBytes: number;
  /**
   * The bytes needed by the transient slot table = the demand side of the prepared backing
   * (**approximate**).
   *
   * Walks the pre-fusion node list (row-block attention in its post-fusion form) in declaration
   * order, builds slot live ranges with the same allocation rules as the execution phase (use
   * counts from `countUses`; graph outputs are pinned and never released; outputs that can be
   * aliased are not allocated and use the input's backing under another name), and sums the regions
   * assigned by the same placement function as the execution phase (placement by live ranges and
   * offsets — ADR 0093). **The in-node temporaries of states-form attention (scores S and row
   * statistics) are included** — they always appear regardless of whether fusion applies, and in
   * capacity-bound graphs they dominate the intermediates (formula: {@link stateAttentionTemps}).
   * **Row-block attention (ADR 0060) windows are included in their post-fusion form** — a window
   * claimed by the same fusion plan as execution is counted as one row block's temporaries plus the
   * output, not as the full width of the scores, mask-added scores and softmax output (ADR 0093
   * addendum of 2026-10-10). **So is the linear prologue** (perf-ledger F-4 — with
   * {@link EstimateOptions.linearCompute} `"a8"`, a producer folded into an i8a8 linear's step ①
   * has no output, as in execution), **and so are the permutes a fused attention absorbs**
   * (perf-ledger F-3 A4 / F-8 — with {@link EstimateOptions.attentionCompute} `"a8"`, counted as
   * absent, with the head-major copies the attention makes in their place on the stages that need
   * them). Other in-node temporaries and intermediates folded
   * away by other fusions are not accounted for ({@link AdmissionReport.unaccounted}).
   */
  readonly workspaceBytes: number;
};

/**
 * The report of needed bytes (returned by {@link estimateGraphMemory} / `PreparedModel.estimate`).
 */
export type AdmissionReport = {
  /** What the Session holds for its lifetime, regardless of the run shape. */
  readonly resident: {
    readonly weights: {
      /**
       * Bytes of weights GPU-resident in compressed form + companion scales (including alignment
       * padding — exact). Corresponds exactly to the diagnostic `storage.residentCompressedBytes`
       * (the only difference is the 4-byte floor of zero-element tensors — the diagnostic counts
       * the real bytes without the floor).
       */
      readonly compressedBytes: number;
      /**
       * Bytes of weights resident in uncompressed storage (f32 / i32) (exact).
       *
       * MUST: keep it in a field separate from compressed / expanded — the diagnostic `storage`
       * counts only low-precision storage (ADR 0006), so mixing this into `compressedBytes` would
       * silently break "exact agreement with the diagnostics" on models with f32 weights.
       */
      readonly uncompressedBytes: number;
      /**
       * Bytes after expansion of ineligible weights that are expanded to f32 at load and kept
       * resident (exact).
       */
      readonly expandedBytes: number;
      /** The sum of the three fields above. */
      readonly totalBytes: number;
    };
    /**
     * State slots + logical-length uniform of one `GenerationContext` (exact).
     *
     * NOTE: 0 when `options.generation` is omitted. For a graph with non-empty `graph.states`, the
     * plan of {@link AdmissionScenario.workspaceBytes} requires the slots' resolved shapes, so it
     * cannot be estimated without generation at all and fails loudly (there is no window that
     * silently returns 0).
     */
    readonly stateBytes: number;
  };
  /**
   * The bytes needed per run shape (one element or more — names as in
   * {@link AdmissionScenarioName}). When the caller passes
   * {@link EstimateOptions}`.generation.scenarios`, they follow the existing two shapes in
   * declaration order.
   */
  readonly scenarios: readonly AdmissionScenario[];
  /**
   * The slot-backing keep budget this estimate used ({@link EstimateOptions.planBackingBudgetBytes}
   * — `DEFAULT_PLAN_BACKING_BUDGET_BYTES` when omitted). Reported so that one term of
   * {@link AdmissionReport.peakAccountedBytes} can be read directly (callers need not rebuild the
   * same number).
   */
  readonly planBackingBudgetBytes: number;
  /**
   * Extra accounting for auxiliary Sessions and resident inputs / outputs. It does not overlap the
   * resident / scenarios breakdown. Omitted means 0. The runtime's single-Session estimate omits
   * it; models add it when they hold auxiliary resources.
   * DECIDED: docs/decisions/0070-shard-loading-admission.md#auxiliary-resource-breakdown-2026-09-12
   */
  readonly auxiliaryBytes?: number;
  /**
   * Resident total + **the bound of the kept set** + auxiliary resources (=
   * `resident.weights.totalBytes + resident.stateBytes + max(planBackingBudgetBytes, max(ioBytes +
   * workspaceBytes)) + (auxiliaryBytes ?? 0)` — the inner max is over all scenarios and bucket
   * shapes).
   *
   * A field that claims **"the peak of what is accounted for", not a guaranteed upper bound** (hence
   * the name). The scenario side is added as `max(budget, largest scenario)` rather than a sum
   * because the Session's kept set of slot backings **never exceeds that amount** (ADR 0095
   * decisions 1 / 4): a hit run with a new signature retires the oldest until "kept + new ≤ budget"
   * before allocating, and a single new backing that alone exceeds the budget retires every kept one
   * and is kept alone (`Session.#evictBackingsFor` in the executor). This is accounted separately
   * from the plans (`PreparedPlan`), which stay in an LRU of up to `PREPARED_PLAN_CAPACITY`.
   *
   * "The largest scenario" is the maximum over all of {@link AdmissionReport.scenarios} **including
   * the extra scenarios** (if only those escaped the accounting, the peak would be too low when the
   * verify shape is the largest), together with the same computation for each physical row count of
   * `generation.chunkBuckets` (the bucket shapes). Bucket shapes are not listed in
   * {@link AdmissionReport.scenarios} (the public list stays prefill / decode plus the extra
   * scenarios). The largest-M prefill shape is not used as the representative because the equal
   * split into row blocks creates, at the boundary of the block count, "a shape with fewer rows has
   * larger temporaries", so the intermediates are not monotonic in M.
   * When bucket shapes are kept at the same time as other shapes within the budget, the budget term
   * bounds them from above (it errs on the large side — the semantics of "the peak of what is
   * accounted for" stay). The budget counts each backing's regions + the input buffers it owns
   * (the diagnostics' `residentBytes + inputBytes`).
   */
  readonly peakAccountedBytes: number;
  /**
   * What is not accounted for (the field whose format admits that the estimate is not an absolute
   * guarantee — ADR 0070 decision 5).
   */
  readonly unaccounted: readonly string[];
};

export type EstimateOptions = {
  /**
   * The same states attention setting as the Session. When omitted, the temporaries of the
   * reference path are estimated.
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * Values of symbolic dimensions (the graph-input side — the same bindings passed to a run).
   * States-only symbols (capacity) and the symbols of the physical chunk rows `M` are both decided
   * on the {@link EstimateOptions.generation} side, so they are not accepted here (the former from
   * `generation.bindings`, the latter from `generation.chunkLength`).
   */
  readonly bindings?: SymbolBindings;
  /**
   * The spec of the `GenerationContext` to estimate (when omitted, no state is counted).
   *
   * `scenarios` is a list of shapes counted **in addition to** the two shapes of ADR 0066 decision 4
   * ({@link AdmissionScenarioSpec}) — an estimate-only field that has no effect at all on the
   * `GenerationContext`'s shape; the resident side (state slots) is the same for every shape and
   * does not move.
   */
  readonly generation?: GenerationContextSpec & {
    readonly scenarios?: readonly AdmissionScenarioSpec[];
  };
  /**
   * The granted value of `maxStorageBufferBindingSize` (`GPUDevice.limits` / `readAdapterLimits`).
   *
   * Used in three places. ① The row-block count of the in-node temporaries of states-form attention
   * — only **one row block's** temporaries are live at a time, and the count is decided by this
   * limit alone ({@link planStateAttention} — a pure function of the device's granted limit and the
   * bindings, because ADR 0022 bans run-time autotuning). ② The binding limit of intermediate
   * placement for every graph — used for the row-block count of row-block attention and for the
   * limit check of actual slot sizes (packed with the same granted value as the execution phase —
   * ADR 0093 decisions 4 / 5). For ②, the WebGPU core default is used when unspecified, except
   * that the row-block count of row-block attention follows ③'s rule
   * ({@link assertRowBlockWindowCount}).
   * ③ The row-block count of the in-node temporaries of fused attention ops (no states field — ADR
   * 0124 decision 7). When unspecified, only shapes whose count is the same on every device are
   * counted — shapes that fit in one block under the WebGPU core default (128 MiB; a device's
   * granted value is always at least this), and shapes whose `attentionRowBlockMaxBytes` is at most
   * the core default (the limit is then the budget itself on every device). Anything else fails
   * loudly ({@link fusedAttentionBlocks}).
   *
   * MUST: **required** for graphs with states-form attention ({@link stateAttentionTemps} fails
   * loudly). Also required for fused attention ops and row-block attention windows whose count
   * depends on the binding limit (③).
   * Filling a default would make the estimator alone claim a count that no device actually has.
   * **The range (a positive safe integer) is checked at the entry regardless of the graph's shape**
   * (if an invalid value passed silently only for shapes that do not read ①, the location of the
   * range check would be invisible to the caller).
   */
  readonly maxStorageBufferBindingSize?: number;
  /**
   * The granted value of `maxBufferSize` (`GPUDevice.limits` / `readAdapterLimits`).
   *
   * Used for the per-slot limit check of state slots (**paired** with `maxStorageBufferBindingSize`
   * — the real construction checks both: ADR 0066 addendum 5) and for the region limit of
   * intermediate packing (packed with the same granted value as the execution phase — ADR 0093
   * decision 4). The latter can move `workspaceBytes` (how the placement splits regions depends on
   * the limit). There is no field for a device value of the offset alignment, so it packs with the
   * WebGPU core default (256).
   *
   * The field is optional so as not to break callers without a device (estimates that hold only a
   * graph). Only the fields passed take part in the checks, and the missing side is not filled with
   * a default (so that the estimator alone never claims a boundary that no device actually has).
   * Intermediate packing alone needs a region limit even when unspecified, so it uses the larger of
   * the binding limit and the WebGPU core default. The range (a positive safe integer) is checked at
   * the entry regardless of the graph's shape.
   */
  readonly maxBufferSize?: number;
  /**
   * The slot-backing keep budget (the same value as `SessionOptions.planBackingBudgetBytes`; default
   * `DEFAULT_PLAN_BACKING_BUDGET_BYTES` = 256 MiB).
   *
   * MUST: the check is the same as `createContainerSession` (anything but a non-negative safe
   * integer fails loudly). If only the estimate accepted a budget with which no Session can be
   * built, the result would be "estimated, but construction fails".
   * It affects only {@link AdmissionReport.peakAccountedBytes}; the per-scenario numbers do not move
   * (the budget is a quantity on the side of **how many** are kept, not the demand of one shape).
   * Budget 0 = always one, so the peak is the largest scenario, as before.
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * The budget per row-block temporary (the same value as `SessionOptions.attentionRowBlockMaxBytes`;
   * omitted = no budget — ADR 0124).
   *
   * The row-block count is decided by the same derivation as execution (`rowBlockLimit` — the
   * smaller of the binding limit and this value). It affects the accounting split into row blocks
   * (the scores S and row statistics of states-form attention, the window temporaries of row-block
   * attention); the limit check of intermediate placement stays at the binding limit (as in
   * execution).
   * MUST: the check is the same as `createContainerSession` (anything but a positive safe integer
   * fails loudly).
   * MUST: a shape whose temporaries for a single row exceed this value fails loudly with the same
   * message as execution's plan derivation — the way for the estimate to learn of a budget violation
   * in advance (in execution it fails in a run after the weights are uploaded).
   */
  readonly attentionRowBlockMaxBytes?: number;
  /**
   * The compute form of fused attention (the same value as `SessionOptions.attentionCompute`;
   * omitted = `"f32"`, the runtime default — the same precedent as `stateAttentionReduce`).
   *
   * It affects the in-node temporaries of fused attention ops (the width of one S element, whether
   * i8a8 quantisation temporaries exist — ADR 0124 decision 7), and with `"a8"` the estimate sees the
   * same decision as execution on the permutes a fused attention absorbs (perf-ledger F-3 A4 / F-8 —
   * the same `planFusions`). Combinations execution rejects (mask
   * × a8, GQA × a8, states form × anything but f32) fail loudly with the same test and message as
   * execution (the estimate never answers with numbers where execution would reject).
   * MUST: spelling and combinations (c16 × s16 with `attentionScoreStorage`) go through the same
   * check as `sessionOptionsViolation`.
   */
  readonly attentionCompute?: ComputePrecision;
  /**
   * The storage form of fused attention's S (the same value as
   * `SessionOptions.attentionScoreStorage`; omitted = `"f32"`).
   * Eligibility (`attentionRoute`) is the same single function as in execution; ineligible shapes
   * are counted with f32 storage. `"none"` (the online kernel — ADR 0132) counts no S and no row
   * statistics for the nodes whose route takes it, only the a8 quantisation temporaries, and on those
   * nodes the estimate sees the same absorbed output permute as execution (perf-ledger F-3 A4 — the
   * same `planFusions`); it needs {@link attentionDevice}.
   */
  readonly attentionScoreStorage?: ScoreStorage;
  /**
   * The storage of full state slots (the same value as `SessionOptions.fullStateStorage`; omitted =
   * `"f32"`). `"f16"` halves the bytes of the full slots and leaves the sliding slots alone (the same
   * single function as the allocation), and on an f16 slot `stateAttentionReduce: "parallel-fused"`
   * counts the temporaries of the unfused kernels, as execution runs them.
   * MUST: the spelling goes through the same check as `sessionOptionsViolation`.
   */
  readonly fullStateStorage?: StateStorage;
  /**
   * The device facts the online kernel's route reads (ADR 0132 decision 5): whether the device has
   * subgroups (`subgroups` and `subgroup-size-control` enabled and WGSL `subgroup_id` present) and
   * its granted `maxComputeWorkgroupStorageSize`.
   *
   * MUST: required when `attentionScoreStorage` is `"none"`, whatever the graph — without it the
   * estimate fails loudly at its entry instead of guessing a route (a device without subgroups and
   * with a 32 KiB limit runs the three dispatches with S, the others the online kernel without it).
   * Not read otherwise.
   */
  readonly attentionDevice?: {
    readonly subgroups: boolean;
    readonly maxComputeWorkgroupStorageSize: number;
  };
  /**
   * The execution form of linear (the same value as `SessionOptions.linearCompute`; omitted =
   * `"f32"`, the runtime default).
   *
   * With `"a8"` the estimate sees the same linear-prologue decision as execution (perf-ledger F-4 —
   * the same `planFusions`, on the weights' storage from the residency plan): an elementwise producer
   * folded into an i8a8 linear's step ① has no output, so its bytes are not counted. A borrowed
   * weight's storage (`seat: "shared"`) is the lender's and unknown here, so its linear is counted
   * unfolded (on the large side). The i8a8 linear's quantized intermediates stay unaccounted.
   * MUST: the spelling goes through the same check as `sessionOptionsViolation`.
   */
  readonly linearCompute?: "f32" | "a8" | "f16";
  /**
   * The expert cache (the same value as `SessionOptions.expertCache` — ADR 0146). With it the expert
   * weights count as their pools of their layer's `slots` experts (one count, or one per routed
   * layer — the same planner as construction), the slot
   * tables (one i32 per expert and routed layer) count with the uncompressed weights, and every
   * routed `topk` index counts as an output (the routing readback). A graph or a value the
   * construction refuses fails loudly with the same message.
   *
   * Not counted, and named in {@link AdmissionReport.unaccounted} instead: the staging ring (its
   * `MAP_WRITE | COPY_SRC` upload buffers are host-visible upload memory, the class of the
   * `writeBuffer` staging it replaces, which is not counted either) and the host tier of `hostBytes`
   * (RAM, not GPU memory).
   */
  readonly expertCache?: {
    readonly slots: number | readonly number[];
    readonly hostBytes?: number;
  };
};

/**
 * What is not accounted for (a constant).
 *
 * MUST: list every item known to be "not included". Leaving it empty would let the estimate be
 * read as if it were an upper bound (exactly why ADR 0070 decision 5 required the unaccounted
 * field).
 */
const UNACCOUNTED: readonly string[] = Object.freeze([
  "Intermediates that fusions other than row-block attention, the linear prologue and the attention's absorbed permutes fold away, and the in-node temporaries their fusion rules declare (both depend on the device limits and on whether the fusion applies; aliased values and the temporaries of row-block attention are counted, a producer folded into an i8a8 linear is counted as absent when linearCompute is passed, and a permute absorbed by a fused attention is counted as absent with the head-major copies that attention makes in its place when attentionCompute 'a8' is passed)",
  "Quantized intermediates of linear i8a8 (depend on the numeric variant and the device limits). The in-node temporaries of attention are counted — the score S of states-form attention and the row statistics stateAttentionReduce needs, and the score S, row statistics and i8a8 quantized intermediates of the fused attention op (counted with the runtime default f32 unless attentionCompute / attentionScoreStorage is passed; without maxStorageBufferBindingSize only the shapes whose block count is the same on every device are counted [shapes that fit in one block under the WebGPU core default, and shapes whose attentionRowBlockMaxBytes is at most the core default], and no number is returned for the others)",
  "params buffers (kernel constants — resident for the Session's lifetime, content-addressed cache)",
  "The implementation staging of queue.writeBuffer (not released until the submit completes)",
  "The retirement window (slot backings retired by a budget overrun / the plan LRU eviction are not destroyed before the next plan's allocation and live until the cleanup after flush, so that run holds the retired backings and the new ones at the same time)",
]);

/**
 * {@link UNACCOUNTED} under the expert cache (ADR 0146): its staging ring, and its host tier when
 * `hostBytes` is set. Without the option the list is {@link UNACCOUNTED} itself (not a byte of the
 * report changes).
 */
const EXPERT_UNACCOUNTED: readonly string[] = Object.freeze([
  ...UNACCOUNTED,
  "The expert cache's staging ring (MAP_WRITE | COPY_SRC upload buffers of up to 32 experts — host-visible upload memory like the writeBuffer staging it replaces)",
]);
const EXPERT_HOST_UNACCOUNTED: readonly string[] = Object.freeze([
  ...EXPERT_UNACCOUNTED,
  "The expert cache's host tier (expertCache.hostBytes of experts in RAM — host memory, not GPU memory)",
]);

/**
 * Looks up a resolved shape (planGraph lists every value, so a missing one is a bookkeeping break).
 */
const shapeOf = (
  shapes: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] => {
  const shape = shapes.get(name);
  if (shape === undefined) throw new ExecutionError(`Shape of value '${name}' is unresolved`);
  return shape;
};

/**
 * The axis that carries the physical chunk rows `M`.
 *
 * MUST: the source of truth for the position is **the op contract** (ops/shapes.ts). States-form
 * attention has `q[B,H,M,D]` / `k`,`v``[B,Hkv,M,D]` (the contract also checks that M agrees across
 * q / k / v), `state_append` has `x[B,Hkv,M,D]`; both use axis 2. Writing another axis here would
 * make the estimator alone treat a different dimension than execution as the chunk rows and confuse
 * prefill / decode.
 */
const CHUNK_ROW_AXIS = 2;

/**
 * The **dimension declarations** that decide the physical chunk rows `M` (symbolic ones only,
 * deduplicated).
 *
 * The test is "inputs of nodes with a non-empty states field" = the states-form test itself (ADR
 * 0067 decision 4 — the same node set that `buildStateAttention` / `buildStateAppend` in
 * recipe-builders/attention.ts push to `states.chunkRows`). Numeric dimensions are shapes that do not
 * move with a symbol, so they are not collected — the decode step of such a shape runs with the same
 * physical chunk rows as prefill and `queryLength = 1`, so the two scenarios correctly give the same
 * number. Whether a numeric `M` is in the set the context allows is a separate check:
 * {@link stateChunkRows} is matched per scenario (it need not equal `chunkLength`).
 */
const chunkRowDims = (graph: IrGraph): readonly string[] => {
  const declared = new Map<string, readonly IrDim[]>();
  for (const spec of graph.inputs) declared.set(spec.name, spec.shape);
  for (const [name, value] of Object.entries(graph.values)) declared.set(name, value.shape);
  const dims = new Set<string>();
  for (const node of graph.nodes) {
    if (Object.keys(node.states).length === 0) continue;
    // planGraph fails names without a declaration (values that are only initializers and the like)
    // — skip them here.
    for (const name of node.ins) {
      const dim = declared.get(name)?.[CHUNK_ROW_AXIS];
      if (typeof dim === "string") dims.add(dim);
    }
  }
  return [...dims];
};

/**
 * The physical chunk row counts used by the state nodes of a resolved plan (equal values folded).
 *
 * MUST: the nodes and axis collected are the same as execution's collection (`buildStateAttention`
 * in recipe-builders/attention.ts pushes `q[2]`, `buildStateAppend` pushes `x[2]`, to
 * `states.chunkRows`) — both the {@link CHUNK_ROW_AXIS} of the first input. For readonly states-form
 * attention the shape layer guarantees M = 1 for `q`, so collecting it does not change the test
 * against the allowed set (which always contains 1).
 */
const stateChunkRows = (nodes: readonly NodePlan[]): ReadonlySet<number> => {
  const rows = new Set<number>();
  for (const step of nodes) {
    if (Object.keys(step.node.states).length === 0) continue;
    rows.add(step.inputShapes[0][CHUNK_ROW_AXIS]);
  }
  return rows;
};

/**
 * Checks and copies the bindings used to resolve value shapes.
 *
 * MUST: do not accept states-only symbols here (the same division as `bindSymbols` — the source of
 * truth for capacity is on the `createGenerationContext` side, and accepting them here would be a
 * silent wrong answer, "passed a state capacity but got 0"). An unbound symbol fails loudly (no
 * silent 0).
 * MUST: do not accept the chunk-row symbols here either (their binding points are
 * `options.generation.chunkLength` and the decode shape's 1 — a third binding point whose value
 * changes per scenario). Accepting and overriding them would be a silent override, "passed a value
 * fixed for prefill but got decode's number"; accepting and preferring them would collapse the two
 * scenarios into the same shape.
 */
const planBindings = (
  graph: IrGraph,
  bindings: SymbolBindings | undefined,
  chunkSymbols: ReadonlySet<string>,
): SymbolBindings => {
  // resolveBindings owns the checks that the symbol exists and is a non-negative integer, and the
  // null-prototype container.
  const resolved = resolveBindings(graph, bindings);
  const statesOnly = statesOnlySymbols(graph);
  for (const sym of Object.keys(resolved)) {
    if (statesOnly.has(sym)) {
      throw new ExecutionError(
        `Binding '${sym}' is a states-only symbol — give it in options.generation.bindings` +
          " (binding point 2 of ADR 0066 addendum 7)",
      );
    }
    if (chunkSymbols.has(sym)) {
      throw new ExecutionError(
        `Binding '${sym}' is the physical chunk-row symbol M — its values are derived from options.generation.chunkLength` +
          " (prefill shape) and 1 (decode shape) (the two execution shapes of ADR 0066 decision 4)",
      );
    }
  }
  for (const sym of graph.symbols) {
    if (statesOnly.has(sym) || chunkSymbols.has(sym) || Object.hasOwn(resolved, sym)) continue;
    throw new ExecutionError(
      `Symbol '${sym}' is not bound (its value must be given in options.bindings)`,
    );
  }
  return resolved;
};

/**
 * The bindings of a scenario with `rows` chunk rows (adds the chunk symbols to a copy of `base`).
 *
 * MUST: the inverse solve is the same single function as execution ({@link solveDim} — the one
 * `bindSymbols` uses to solve bindings from the actual input sizes). Accepting a derived form
 * (`2M`) with rounding would return numbers estimated for a chunk one row off the declaration,
 * looking as if "it runs in that shape".
 */
const bindChunkRows = (
  base: SymbolBindings,
  dims: readonly string[],
  rows: number,
  scenario: string,
): SymbolBindings => {
  const bindings: Record<string, number> = Object.assign(Object.create(null), base);
  for (const dim of dims) {
    const expr = parseDim(dim);
    const solved = solveDim(expr, rows);
    if (solved === undefined) {
      throw new ExecutionError(
        `Scenario '${scenario}': physical chunk row count ${rows} does not have the declared form '${dim}'` +
          ` (needs a row count divisible by ${expr.coeff} and at least ${expr.offset} — ADR 0066 decision 4)`,
      );
    }
    if (Object.hasOwn(bindings, expr.sym) && bindings[expr.sym] !== solved) {
      throw new ExecutionError(
        `Scenario '${scenario}': conflicting bindings for symbol '${expr.sym}' (${bindings[expr.sym]} and ` +
          `${solved}) — the chunk-row declarations disagree within one graph`,
      );
    }
    bindings[expr.sym] = solved;
  }
  return bindings;
};

/**
 * Range check of a device-limit field (optional; if given, a positive safe integer).
 *
 * MUST: place it at the entry, not inside the readers (`stateAttentionTemps` /
 * `assertSlotWithinLimits`) — if an invalid value passed silently only for graphs that do not read
 * it, the location of the range check would be invisible to the caller.
 */
const assertLimitOption = (name: string, limit: number | undefined): void => {
  if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1)) {
    throw new ExecutionError(`options.${name} ${limit} must be a positive safe integer`);
  }
};

/**
 * The estimate of the state slots (the same formula as {@link GenerationContext}'s allocation — no
 * copied values).
 */
const stateEstimate = (
  graph: IrGraph,
  spec: GenerationContextSpec | undefined,
  limits: SlotLimits,
  fullStateStorage: StateStorage,
): {
  readonly bytes: number;
  readonly shapes: ReadonlyMap<string, readonly number[]> | undefined;
  readonly bindings: SymbolBindings | undefined;
} => {
  if (spec === undefined) return { bytes: 0, shapes: undefined, bindings: undefined };
  const names = Object.keys(graph.states);
  // MUST: do not accept generation for a graph without states declarations (a form
  // `GenerationContext.create` rejects. Letting it through would return an estimate counting "the
  // bytes of a context that cannot be built").
  if (names.length === 0) {
    throw new ExecutionError(
      "This graph has no states declarations (no GenerationContext can be created, so do not pass options.generation)",
    );
  }
  // MUST: return no estimate for a spec the real construction (GenerationContext.create) rejects —
  // ranges go through the same checks (assertChunkLength / assertChunkBuckets), per-slot sizes
  // through the same `assertSlotWithinLimits` (in the loop below).
  assertChunkLength(spec.chunkLength);
  assertChunkBuckets(spec.chunkBuckets, spec.chunkLength);
  const bindings = resolveBindings(graph, spec.bindings);
  const shapes = new Map<string, readonly number[]>();
  const windows = slidingSlotWindows(graph);
  let bytes = 0;
  for (const name of names) {
    const shape = resolveSlotShape(name, graph.states[name].shape, bindings);
    shapes.set(name, shape);
    // A borrowing context does not allocate borrowed slots (external — ADR 0096, addendum of
    // 2026-09-08 on stage 2). The shape is resolved because the temporaries (S / row statistics)
    // need it, but no byte is counted (the same branch as `GenerationContext.create`'s allocation =
    // estimate and real construction claim the same number).
    if (graph.states[name].external) continue;
    const byteLength = stateSlotBytes(name, shape, windows.has(name), fullStateStorage);
    // MUST: check the per-slot limits at the same point and with the same check as the real
    // construction (`GenerationContext.create`) (ADR 0066 addendum 5). Without it, a path where a
    // CLI lets the estimate double as the value check of `--capacity` (examples/shared/gemma-main.ts,
    // removed in 0.15) would "show numbers at startup and fail on the first turn".
    assertSlotWithinLimits(name, shape, byteLength, limits);
    bytes += byteLength;
  }
  // One logical-length uniform per context (independent of the slot count).
  return { bytes: bytes + LENGTHS_BYTES, shapes, bindings };
};

/**
 * The estimate of the weights. Both the classification and the declared bytes come from the result
 * of **the same planner as Session construction** ({@link planWeightResidency}), and the expansion
 * from residency class (`seat`) to real buffers goes through the same single function as
 * construction / the limit check ({@link planWeightBuffers}) — rewriting the classification or
 * expansion here would let the estimator keep claiming a different number, with no exception or
 * warning, against an implementation fixed on one side only (the MUST of the module doc).
 *
 * This only adds the categorisation per residency class. Alignment padding and the buffer floor
 * ({@link toSizeClass}) belong to `WeightBuffer.byteLength` — equal to the 4-byte round-up of
 * `alignF16Payload` / `alignI8Payload` combined with the `Math.max(4, …)` of `allocHostWritten`.
 */
const weightEstimate = (
  residency: ReadonlyMap<string, WeightResidency>,
): { readonly compressed: number; readonly uncompressed: number; readonly expanded: number } => {
  let compressed = 0;
  let uncompressed = 0;
  let expanded = 0;
  for (const buffer of planWeightBuffers(residency)) {
    switch (buffer.seat) {
      case "raw":
        // Uncompressed storage: the raw bytes are the GPU representation as is (the third branch in
        // the executor).
        uncompressed += buffer.byteLength;
        break;
      case "expanded":
        // Ineligible weights are expanded to f32 on the CPU at load (zero VRAM saving).
        // MUST: only this field counts bytes before alignment — the diagnostic
        // `storage.hostExpandedBytes` sums the real bytes passed to writeBuffer and has no 4-byte
        // floor, so switching this to the allocation size would silently break "exact
        // correspondence" for models with zero-element tensors.
        expanded += buffer.declaredBytes;
        break;
      // Compressed residency counts **both** the payload and the companion scale (the same rule as
      // the executor's residentCompressedBytes — the field represents the bytes the GPU actually
      // holds, so leaving the scale out would disagree with the measurement). Both are listed as
      // `WeightBuffer`s.
      case "f16":
      case "bf16":
      case "i8":
      case "i4":
      case "i4f16":
      case "i2":
        compressed += buffer.byteLength;
        break;
      default: {
        // MUST: stop an unhandled residency class here. If the switch fell through, that buffer's
        // bytes would enter none of the three fields and `resident.weights.totalBytes` would keep
        // claiming less than the real load (the estimate does not stop execution, so a drift would
        // not turn anything red). It is unreachable by type; this branch exists to turn a missed
        // wiring red at compile time when a residency class is added.
        const unhandled: never = buffer.seat;
        throw new ExecutionError(`Unhandled residency class: ${JSON.stringify(unhandled)}`);
      }
    }
  }
  return { compressed, uncompressed, expanded };
};

/**
 * Whether it is states-form attention (ADR 0067 decision 4 — **the presence of the field decides
 * the form**).
 *
 * MUST: the test is the same single condition as `#buildStep` in recipe-builder (op is `attention`
 * and the `states` field is non-empty). A different condition would make the estimator alone mix up
 * the temporaries of a separate kernel family that shares not a single byte with fused attention.
 */
const isStateAttention = (node: NodePlan): boolean =>
  node.contract.kind === "attention" && Object.keys(node.node.states).length > 0;

/**
 * The partial-result temporary of the two-phase argmax / topk(k=1) (long rows — the "two-phase
 * split" section of src/kernels/argmax.ts).
 * The path is chosen through the same pure functions as the execution phase, `argmaxSplitGroups` /
 * `topkOneSplitGroups` — rewriting the threshold here would let the estimator keep claiming a
 * different number against an implementation fixed on one side only.
 * The one-dispatch form (short rows) returns 0.
 */
const selectionSplitTempBytes = (node: NodePlan): number => {
  if (
    node.node.op !== "argmax" &&
    !(node.node.op === "topk" && topkK(node.node.attrs, "topk estimate") === 1)
  )
    return 0;
  const input = node.inputShapes[0];
  const dim = input[input.length - 1];
  const groups = node.node.op === "topk" ? topkOneSplitGroups(dim) : argmaxSplitGroups(dim);
  return groups === 0 ? 0 : argmaxSplitPartialBytes(numel(input.slice(0, -1)), groups);
};

/**
 * Lists, in row-block order, the in-node temporaries (scores S and row statistics) of one
 * states-form attention node.
 *
 * They always appear regardless of whether fusion applies, and their size is proportional to the
 * column capacity (full = the slot capacity `C` / sliding = the window's resident width `W−1+M`),
 * so in graphs with variable capacity they dominate the intermediates.
 *
 * MUST: take the formulas only from {@link planStateAttention} (the single function shared with the
 * execution plan's `buildStateAttention`). Rewriting them here would let the estimator keep claiming
 * a different number, with no exception or warning, against an implementation fixed on one side
 * only (the MUST of the module doc). This only translates the estimate-specific material (the
 * symbol-resolved slot shapes) into geometry.
 * MUST: without `maxStorageBufferBindingSize`, fail loudly. The count is decided only by the
 * device's granted limit (and the user's budget), so filling a default would claim "a temporary
 * size that no device produces".
 * MUST: combine the budget (`budget` — `EstimateOptions.attentionRowBlockMaxBytes`) with the binding
 * limit through the same `rowBlockLimit` as execution (ADR 0124 decision 7).
 * NOTE: the range (a positive safe integer) is checked at the entry of {@link estimateGraphMemory} —
 * graphs without states-form attention do not pass through here, so placing it inside would put
 * the check where the caller cannot see it.
 */
const stateAttentionTemps = (
  node: NodePlan,
  stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  limit: number | undefined,
  budget: number | undefined,
  stateAttentionReduce: StateAttentionReduce,
  fullStateStorage: StateStorage,
): readonly StateAttentionBlock[] => {
  const q = node.inputShapes[0];
  const where = `attention (states) [${q.join(",")}]`;
  if (limit === undefined) {
    throw new ExecutionError(
      `${where}: the temporaries of states-form attention need options.maxStorageBufferBindingSize` +
        " (the row-block count is decided by the device's granted limit alone — ADR 0060 / ADR 0022)",
    );
  }
  const [batch, heads, chunkRows] = q;
  // The k / v slots have the same shape (checked by the shape layer), so the capacity is taken from
  // one of them.
  const slotName = node.node.states["k"];
  const slotShape = slotName === undefined ? undefined : stateShapes?.get(slotName);
  if (slotShape === undefined) {
    throw new ExecutionError(
      `${where}: the resolved capacity of state slot 'k' cannot be looked up` +
        " (an estimate without a GenerationContext cannot count states-form attention)",
    );
  }
  const window = stateWindow(node.node.attrs, where) ?? 0;
  // `forced`, which sets the row-block count explicitly (the test-only `ROW_BLOCK_SPLIT`), is not
  // passed — `EstimateOptions` has no field for it and production callers do not use it (the
  // estimate claims only the default equal split).
  return planStateAttention(
    {
      batchHeads: batch * heads,
      chunkRows,
      capacity: slotShape[2],
      window,
      fuseStatsPv:
        node.node.ins.length === 3 &&
        stateFusesStatsPv(
          stateAttentionReduce,
          slotStateStorage(fullStateStorage, stateSliding(window)),
        ),
    },
    rowBlockLimit(limit, budget),
  ).blocks;
};

/**
 * Looks up, among the fusion plan's steps (the same {@link planFusions} as the execution phase),
 * those the estimate counts **in post-fusion form**, as "index of the window's first node → step"
 * (ADR 0093 addendum of 2026-10-10).
 *
 * Row-block attention, the linear prologue and the attention's absorbed permutes. Counted in the
 * pre-fusion list, the binding limit would hit the full width of the scores, mask-added scores and
 * softmax output that fusion never materialises, and only the estimate of shapes that do run would
 * throw (the demand shape of ADR 0060). A linear with a folded producer (perf-ledger F-4) is a plain
 * node step that stands for the producer at the key and the linear right after it; counted in the
 * pre-fusion list, the producer's output (454 MB at the Wan2.2 33-frame FFN) would stay in a peak
 * execution never has. An attention that absorbed its permutes (perf-ledger F-3 A4 / F-8) is a plain
 * node step that stands for the absorbed input permutes at the key, the attention right after them
 * and, with `o`, the output permute after it ({@link nodeStepSpan} nodes); the permutes' copies
 * never exist as node outputs, and the copies the attention makes itself are its own temporaries
 * ({@link fusedAttentionProgram}).
 * Other rules are still counted in the pre-fusion list (the `unaccounted` side).
 * MUST: pass the same `linearCompute` and weight storage as execution (`linear`) — the prologue is
 * decided from them, and the estimate would otherwise count the pair unfolded. Likewise the same
 * seats (`seats`) and device facts (`device`) — the input permutes are decided from the attention
 * seat, the output permute from `attentionRoute` on the seats and the device (only the online
 * kernel absorbs it).
 *
 * MUST: pass the same binding limit as the limit check of the allocation program (the row-block
 * count is decided only by this value and the budget — dividing by another value would make the
 * estimate alone count temporaries with a block count different from execution). The budget
 * (`budget`) is passed as `attentionRowBlockMaxBytes` of the fusion context, as in execution (ADR
 * 0124 decision 7). A forced row-block count (`rowBlockSplit` — test only) is not passed (the same
 * reason as {@link stateAttentionTemps}).
 * NOTE: `maxComputeWorkgroupsPerDimension` only affects the limit check of dispatch counts and does
 * not move bytes. The estimate checks no dispatch counts even for plain nodes (it has no field for
 * it either), so a value that checks nothing is passed here too.
 */
const postFusionSteps = (
  nodes: readonly NodePlan[],
  uses: ReadonlyMap<string, number>,
  outputNames: ReadonlySet<string>,
  bindingLimit: number,
  budget: number | undefined,
  linear: LinearKnobs,
  seats: AttentionSeats,
  device: AttentionDeviceFacts,
): ReadonlyMap<number, ExecStep> => {
  const { steps } = planFusions(nodes, {
    rules: FUSION_RULES,
    useCounts: uses,
    outputNames,
    limits: {
      maxStorageBufferBindingSize: bindingLimit,
      maxComputeWorkgroupsPerDimension: Number.MAX_SAFE_INTEGER,
    },
    ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
    linearCompute: linear.linearCompute,
    attentionCompute: seats.attentionCompute,
    attentionScoreStorage: seats.attentionScoreStorage,
    attentionDevice: device,
    weightLayouts: linear.weightLayouts,
  });
  const found = new Map<number, ExecStep>();
  // The scan lays out "passthrough nodes → fused step" per window and advances by the window width,
  // so the running node count of the step list always points at the start of the next window.
  // Row-block attention windows have no passthrough (window = chain), so this running count is
  // their start in the pre-fusion list as is. A node step that absorbed its neighbours (a folded
  // producer, absorbed permutes) stands for `nodeStepSpan` contiguous nodes. A window that defers
  // nodes (rmsNormRope — "Deferred nodes" in fusion.ts) lays out its fused step first and its
  // deferred nodes after it, so inside that region the count is the position in the scanned list,
  // and it is the pre-fusion position again once they are laid out. No step kept here starts in
  // such a region: the deferred nodes are `rms_norm` / `reshape`, which head none of them (the
  // MUST on `FusionMatch.others`).
  let index = 0;
  for (const step of steps) {
    if (step.kind === "node") {
      const span = nodeStepSpan(step);
      if (span === 1) {
        index += 1;
        continue;
      }
      // MUST: the step's output is the last of its nodes in the unfused list (the linear after its
      // producer, the attention after its input permutes or its output permute) — the same
      // bookkeeping-break check as row-block attention's below.
      if (nodes[index + span - 1]?.outputs[0]?.name !== step.plan.outputs[0].name) {
        throw new ExecutionError(
          `Estimate: the node step '${step.plan.outputs[0].name}' that absorbed ${span - 1} node(s) does not match the unfused node list at index ${index}`,
        );
      }
      found.set(index, step);
      index += span;
      continue;
    }
    if (step.rule === "rowBlockAttention") {
      // MUST: if the end of the chain does not match the fused step's output, bookkeeping is broken
      // (silently replacing another node would make the estimate count a list different from
      // execution).
      if (nodes[index + step.nodeCount - 1]?.outputs[0]?.name !== step.outputName) {
        throw new ExecutionError(
          `Estimate: the window of row-block attention '${step.outputName}' does not match the unfused node list at index ${index}`,
        );
      }
      found.set(index, step);
    }
    index += step.nodeCount;
  }
  return found;
};

/**
 * The allocation program of one fused row-block attention step (a copy of the bookkeeping side of
 * `#buildStep` / `#buildFused` in recipe-builder).
 *
 * The output is a single-output backing (`numel × 4`), the temporaries keep the lifetimes the rule
 * declared, and releases are the external inputs with repeats ({@link FusedStep.ins}). Dispatch reads
 * and writes come from **the WGSL storage declarations**, as in the execution phase
 * ({@link parseStorageRoles} — no hand-written table).
 *
 * MUST: row-block attention only. It copies only the forms this rule uses (every dispatch declares
 * its operands; no resident scale is bound), not the forms of other rules (the default order when
 * operands are omitted, `weightScale`). Passing another rule would make the estimate silently count
 * an allocation program different from execution, so it fails loudly on both the rule name and the
 * form. To extend it to other rules, add the forms to copy here first, then widen the callers'
 * targets.
 */
const fusedTransientStep = (
  step: FusedStep,
  uses: ReadonlyMap<string, number>,
  outputNames: ReadonlySet<string>,
): TransientStepSpec => {
  const where = `Estimate: fusion rule '${step.rule}' (output '${step.outputName}')`;
  if (step.rule !== "rowBlockAttention") {
    throw new ExecutionError(`${where} cannot be counted after fusion (row-block attention only)`);
  }
  const output: TransientRef = { kind: "value", name: step.outputName };
  const bindRef = (index: number): TransientRef => {
    const name = step.binds[index];
    // MUST: an out-of-range index fails loudly (matching the same check in the execution phase's
    // recipe-builder — silently creating a reference without a name would make the estimate alone
    // return a number for a shape on which the execution phase fails).
    if (name === undefined) {
      throw new ExecutionError(
        `${where}: bind index ${index} is outside the ${step.binds.length} declared binds`,
      );
    }
    return { kind: "value", name };
  };
  const dispatches = step.dispatches.map((dispatch): TransientDispatchSpec => {
    if (dispatch.operands === undefined) {
      throw new ExecutionError(
        `${where}: dispatch '${dispatch.key}' does not declare its operands`,
      );
    }
    const roles = parseStorageRoles(dispatch.wgsl());
    const reads: TransientRef[] = [];
    const writes: TransientRef[] = [];
    dispatch.operands.forEach((operand, slot) => {
      if (operand.kind === "weightScale") {
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' has a resident scale bound to it`,
        );
      }
      const ref: TransientRef =
        operand.kind === "temp"
          ? { kind: "temp", id: operand.id }
          : operand.kind === "output"
            ? output
            : bindRef(operand.index);
      // Binding 0 is params, so the operands start at binding 1.
      const binding = slot + 1;
      if (roles.writes.has(binding)) writes.push(ref);
      else if (roles.reads.has(binding)) reads.push(ref);
      else {
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' binding ${binding} has no WGSL` +
            " storage declaration (its read / read-write role cannot be determined)",
        );
      }
    });
    return { reads, writes };
  });
  return {
    outputs: [
      {
        name: step.outputName,
        kind: "alloc",
        byteLength: numel(step.outputShape) * 4,
        uses: uses.get(step.outputName) ?? 0,
        pinned: outputNames.has(step.outputName),
      },
    ],
    temps: step.temps,
    dispatches,
    releases: step.ins,
  };
};

/**
 * The execution options the intermediate accounting reads (the same values as the Session — after
 * the range checks at the entry of {@link estimateGraphMemory}).
 */
type TransientKnobs = {
  readonly stateAttentionReduce: StateAttentionReduce;
  readonly fullStateStorage: StateStorage;
  /** The budget per row-block temporary (`undefined` = no budget — ADR 0124). */
  readonly attentionRowBlockMaxBytes: number | undefined;
  /**
   * The compute form of fused attention and the storage form of S (defaults already filled in). They
   * also decide the absorbed permutes ({@link postFusionSteps}: the compute form the input ones, the
   * route on both and {@link attentionDevice} the output one).
   */
  readonly seats: AttentionSeats;
  /**
   * The device facts of the online kernel's route: `EstimateOptions.attentionDevice`, or
   * {@link UNREAD_ATTENTION_DEVICE} on a seat whose route never reads them (the entry of
   * {@link estimateGraphMemory} rejects `"none"` without the facts).
   */
  readonly attentionDevice: AttentionDeviceFacts;
  /** What the linear-prologue decision reads ({@link postFusionSteps}). */
  readonly linear: LinearKnobs;
};

/** The facts the linear prologue (perf-ledger F-4) is decided from, as in execution. */
type LinearKnobs = {
  /** `EstimateOptions.linearCompute` with the runtime default filled in. */
  readonly linearCompute: "f32" | "a8" | "f16";
  /** The weights' resident storage ({@link fusionWeightLayouts}). */
  readonly weightLayouts: ReadonlyMap<string, FusionWeightLayout>;
};

/**
 * The resident storage per initializer that the fusion decision reads — from the residency plan,
 * the same projection the executor makes from its resident weights (`#planSteps` in executor.ts).
 * Uncompressed (`raw`) and expanded weights are f32 and absent, as in execution. A borrowed weight's
 * storage (`shared`) is the lender's, decided at construction and unknown here, so it is absent too
 * (its linear is counted unfolded — on the large side).
 */
const fusionWeightLayouts = (
  residency: ReadonlyMap<string, WeightResidency>,
): ReadonlyMap<string, FusionWeightLayout> =>
  new Map(
    [...residency].flatMap(([name, weight]): [string, FusionWeightLayout][] => {
      switch (weight.seat) {
        case "i4":
          return [[name, { storage: "i4", groupSize: weight.groupSize }]];
        case "f16":
        case "bf16":
        case "i8":
        case "i2":
          return [[name, { storage: weight.seat }]];
        default:
          return [];
      }
    }),
  );

/**
 * The row blocks of one fused attention op node (the same `planAttentionBlocks` × `rowBlockLimit` as
 * execution).
 *
 * MUST: when no binding limit (`limit` — `EstimateOptions.maxStorageBufferBindingSize`) is passed
 * and the count could change with the actual binding limit, fail loudly (the same reason as the
 * states-form {@link stateAttentionTemps} — dividing by the core default could let the estimate
 * alone claim a count that no device produces; ADR 0124 decision 7). A device's granted binding
 * limit is at least the WebGPU core default (128 MiB), so even without a limit the following two
 * have the same count on every device and may be counted with the core default: ① shapes that fit
 * in one block under the core default (one block on every device) ② shapes whose budget is at most
 * the core default (the limit is then the budget itself on every device).
 */
const fusedAttentionBlocks = (
  extent: Parameters<typeof planAttentionBlocks>[0],
  limit: number | undefined,
  budget: number | undefined,
  where: string,
): ReturnType<typeof planAttentionBlocks> => {
  if (limit !== undefined) return planAttentionBlocks(extent, rowBlockLimit(limit, budget));
  const core = CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;
  if (budget === undefined || budget > core) {
    // S and row statistics with every row in one block, undivided by a limit (only checks whether
    // one block fits — the count comes from the single call below).
    const whole = planAttentionBlocks(extent, Number.MAX_SAFE_INTEGER);
    if (whole.length !== 1 || Math.max(whole[0].scoreBytes, whole[0].statsBytes) > core) {
      throw new ExecutionError(
        `${where}: the temporaries of fused attention do not fit in one block under the WebGPU core default binding limit (${core}B), so` +
          " options.maxStorageBufferBindingSize is needed (the row-block count changes with the device's granted" +
          " limit — ADR 0124 decision 7 / ADR 0022)",
      );
    }
  }
  return planAttentionBlocks(extent, rowBlockLimit(core, budget));
};

/**
 * The row-block count of one decomposed row-block attention window, checked when no binding limit
 * is passed (ADR 0124 decision 7 — the same rule as {@link fusedAttentionBlocks}).
 *
 * MUST: without `limit` (`EstimateOptions.maxStorageBufferBindingSize`) the fusion is planned with
 * the WebGPU core default, so fail loudly when the count could change with the device's granted
 * limit — dividing by the core default would let the estimate alone claim a count that no device
 * produces. A granted limit is at least the core default, so the count is the same on every device
 * in two cases only: ① the window's whole score S (the first bmm's output) fits in one block under
 * the core default ② the budget is at most the core default (the limit is then the budget itself on
 * every device).
 */
const assertRowBlockWindowCount = (
  step: FusedStep,
  scores: NodePlan,
  limit: number | undefined,
  budget: number | undefined,
): void => {
  if (limit !== undefined) return;
  const core = CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;
  if (budget !== undefined && budget <= core) return;
  const scoreBytes = numel(scores.outputs[0].shape) * 4;
  if (scoreBytes <= core) return;
  throw new ExecutionError(
    `Estimate: row-block attention '${step.outputName}': the score S (${scoreBytes}B) does not fit in one block under the WebGPU core default binding limit (${core}B), so` +
      " options.maxStorageBufferBindingSize is needed (the row-block count changes with the device's granted" +
      " limit — ADR 0124 decision 7 / ADR 0060)",
  );
};

/**
 * The in-node temporaries and dispatch reads / writes of one fused attention op node (no states
 * field — ADR 0023) (ADR 0124 decision 7). It copies the allocation and dispatch order of execution
 * (`buildAttention` in recipe-builders/attention.ts):
 *
 * 1. i8a8 ①QK (`route.qk`): allocate `qq` / `qs` / `kq` / `ks`, two dispatches quantising q and k
 * 2. i8a8 ③PV (`route.pv`): allocate `vt` / `vq` / `vs`, two dispatches for v's permute and Vᵀ's
 *    quantisation
 * 3. Per row block, allocate S and the row statistics and release them after the three dispatches
 *    ①QK → ②row statistics → ③PV
 * 4. Release the quantisation temporaries at the end of the node (after the last ③PV)
 *
 * The online route (`route.online` — ADR 0132) replaces step 3 with one dispatch per row window
 * reading every quantisation temporary; it has no S and no row statistics. With an absorbed output
 * permute (`o`) those dispatches write the token-major output itself (the node's re-pointed output —
 * the same bytes); `o` on any other route fails loudly (`attentionHeadMajorCopies`, as in
 * execution).
 *
 * With absorbed permutes (`tokenMajor` — perf-ledger F-3 A4 / F-8) the quantisation dispatches read
 * the token-major inputs (same reads and writes), and the head-major copies of
 * `attentionHeadMajorCopies` come first (one dispatch each, before step 1) and live until the end of
 * the node, as in execution.
 *
 * MUST: take sizes and counts only from the same pure functions as execution (`attentionRoute`,
 * `attentionScoreElementBytes`, `planAttentionBlocks`, `planAttentionOnlineWindows`,
 * `attentionI8a8TempBytes`, `rowBlockLimit`).
 * The combination check is the same `fusedAttentionViolation`, and no number is returned for shapes
 * execution rejects.
 * NOTE: the handling when there is no binding limit (`limit` —
 * `EstimateOptions.maxStorageBufferBindingSize`) is {@link fusedAttentionBlocks} (only shapes whose
 * count does not depend on the device are counted).
 */
const fusedAttentionProgram = (
  node: NodePlan,
  reads: readonly TransientRef[],
  outputRefs: readonly TransientRef[],
  limit: number | undefined,
  knobs: TransientKnobs,
  tokenMajor: AttentionTokenMajor | undefined,
): { readonly temps: TransientTempSpec[]; readonly dispatches: TransientDispatchSpec[] } => {
  const [q, k, v] = attentionHeadMajorShapes(node.inputShapes, tokenMajor);
  const shape = { batch: q[0] * q[1], rows: q[2], cols: k[2], depth: q[3] };
  const where = `attention [${q.join(",")}] × [${k.join(",")}] × [${v.join(",")}]`;
  const { seats } = knobs;
  const route = attentionRoute(seats, shape.depth, shape.cols, knobs.attentionDevice);
  const [qRef, kRef, vRef, maskRef] = reads;
  const violation = fusedAttentionViolation(
    where,
    { heads: q[1], kvHeads: k[1], depth: shape.depth, cols: shape.cols },
    { i8a8: seats.attentionCompute === "a8", mask: maskRef !== undefined, route },
  );
  if (violation !== undefined) throw new ExecutionError(violation);
  const temps: TransientTempSpec[] = [];
  const dispatches: TransientDispatchSpec[] = [];
  const alloc = (byteLength: number, releaseAfter: number): TransientRef => {
    temps.push({ byteLength, allocBefore: dispatches.length, releaseAfter });
    return { kind: "temp", id: temps.length - 1 };
  };
  const quantized = attentionI8a8TempBytes(shape);
  // The i8a8 quantisation of ①QK (q and k) and of ③PV (v's permute and Vᵀ), alive until `last`.
  const quantizeQk = (last: number) => {
    const refs = {
      qq: alloc(quantized.qk.qq, last),
      qs: alloc(quantized.qk.qs, last),
      kq: alloc(quantized.qk.kq, last),
      ks: alloc(quantized.qk.ks, last),
    };
    dispatches.push({ reads: [qRef], writes: [refs.qq, refs.qs] });
    dispatches.push({ reads: [kRef], writes: [refs.kq, refs.ks] });
    return refs;
  };
  const quantizePv = (last: number) => {
    const refs = {
      vt: alloc(quantized.pv.vt, last),
      vq: alloc(quantized.pv.vq, last),
      vs: alloc(quantized.pv.vs, last),
    };
    dispatches.push({ reads: [vRef], writes: [refs.vt] });
    dispatches.push({ reads: [refs.vt], writes: [refs.vq, refs.vs] });
    return refs;
  };
  if (route.online !== undefined) {
    // The online kernel (ADR 0132): the four quantisation dispatches, then one dispatch per row
    // window reading all of them — no S and no row statistics exist.
    const windows = planAttentionOnlineWindows(shape.rows, shape);
    const last = 4 + windows.length - 1;
    const qk = quantizeQk(last);
    const pv = quantizePv(last);
    dispatches.push(
      ...windows.map(() => ({
        reads: [qk.qq, qk.kq, pv.vq, qk.qs, qk.ks, pv.vs],
        writes: outputRefs,
      })),
    );
    return { temps, dispatches };
  }
  const blocks = fusedAttentionBlocks(
    { ...shape, scoreBytes: attentionScoreElementBytes(seats.attentionCompute, route) },
    limit,
    knobs.attentionRowBlockMaxBytes,
    where,
  );
  const qkI8a8 = route.qk === "i8a8";
  const pvI8a8 = route.pv === "i8a8";
  const copies = attentionHeadMajorCopies(tokenMajor, route, node.inputShapes, where);
  const inputCopies = [copies.q, copies.k, copies.v];
  const last =
    inputCopies.filter((bytes) => bytes !== undefined).length +
    (qkI8a8 ? 2 : 0) +
    (pvI8a8 ? 2 : 0) +
    blocks.length * 3 -
    1;
  // The head-major copies of the token-major inputs the stages read as stored (before them).
  const [qOperand, kOperand, vOperand] = [qRef, kRef, vRef].map((ref, slot) => {
    const bytes = inputCopies[slot];
    if (bytes === undefined) return ref;
    const copy = alloc(bytes, last);
    dispatches.push({ reads: [ref], writes: [copy] });
    return copy;
  });
  const qk = qkI8a8 ? quantizeQk(last) : undefined;
  const pv = pvI8a8 ? quantizePv(last) : undefined;
  for (const block of blocks) {
    const end = dispatches.length + 2;
    const scores = alloc(block.scoreBytes, end);
    const rowStats = alloc(block.statsBytes, end);
    dispatches.push({
      reads:
        qk === undefined
          ? maskRef === undefined
            ? [qOperand, kOperand]
            : [qOperand, kOperand, maskRef]
          : [qk.qq, qk.kq, qk.qs, qk.ks],
      writes: [scores],
    });
    dispatches.push({ reads: [scores], writes: [rowStats] });
    dispatches.push({
      reads: pv === undefined ? [scores, vOperand, rowStats] : [scores, pv.vq, rowStats, pv.vs],
      writes: outputRefs,
    });
  }
  return { temps, dispatches };
};

/**
 * Counts the GPU bytes of intermediate buffers (node outputs and fusion-rule temporaries) with **the
 * same planning function** as execution (ADR 0093 decision 4 — sharing {@link planTransients};
 * meets ADR 0070 decision 5's "the same rule as execution" by sharing the function). Returns the sum
 * of regions (`totalBytes`).
 *
 * MUST: build the allocation program from the pre-fusion node list and replace only the windows of
 * row-block attention with the same fused steps as the execution phase (ADR 0093 addendum of
 * 2026-10-10), each linear-prologue pair with its linear's re-pointed plain node (perf-ledger F-4)
 * and each attention with its absorbed permutes with the attention's re-pointed plain node
 * (perf-ledger F-3 A4 / F-8) — all from {@link postFusionSteps}. Intermediates folded by other fusions and fusion-rule
 * temporaries are on the `unaccounted` side. Plain nodes' dispatches are approximated as one per
 * node (reads = `ins`, writes = `outs`), giving the usage-scope rule for separate regions the same
 * counterparts in the same direction as the execution phase. States-form attention and fused
 * attention ops (no states field — {@link fusedAttentionProgram}) copy the execution phase's three
 * dispatches (①qk writes scores → ②stats reads scores and writes row statistics → ③pv reads both
 * and writes the output) per row block — because the overlap between temporaries and the way regions
 * split affect the sum (fused attention also copies the i8a8 quantisation temporaries and dispatches
 * — ADR 0124 decision 7).
 * MUST: the planning function **lists every overrun** (actual slot size > binding limit) and fails
 * (ADR 0093 decision 5). The same check runs at the estimate's entry, so a shape where "this device
 * cannot bind the intermediates" returns no number and fails loudly. The check applies to the
 * buffers actually bound after fusion (one block's temporaries for row-block attention), not to
 * intermediates that fusion never materialises. The limit is `limit`
 * (`EstimateOptions.maxStorageBufferBindingSize` — the WebGPU core default when unspecified), and the
 * region limit is `maxBufferSize` (`EstimateOptions.maxBufferSize`). Only when the latter is
 * unspecified is it filled with the larger of the binding limit and the core default (binding limit
 * ≤ `maxBufferSize` is an invariant of the spec). The granted values passed in are used because
 * how the placement splits regions changes with the limit, and the sum would disagree with the execution
 * phase (which plans with the granted values) (ADR 0093 decision 4).
 * MUST: the bookkeeping unit is **the slot, not the value name** (an alias extends its root's live
 * range).
 */
const transientSlotBytes = (
  graph: IrGraph,
  nodes: readonly NodePlan[],
  stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  limit: number | undefined,
  maxBufferSize: number | undefined,
  knobs: TransientKnobs,
): number => {
  const { stateAttentionReduce, fullStateStorage, attentionRowBlockMaxBytes } = knobs;
  const uses = countUses(graph);
  const outputNames = new Set(graph.outputs);
  const aliases = planAliases(nodes);
  const bindingLimit = limit ?? CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;
  const nodeStep = (node: NodePlan, tokenMajor?: AttentionTokenMajor): TransientStepSpec => {
    // MUST: absorbed permutes reach only a fused attention (the same check as execution's
    // recipe-builder — counting another node with them would return a number for a plan execution
    // rejects).
    if (
      tokenMajor !== undefined &&
      (node.contract.kind !== "attention" || isStateAttention(node))
    ) {
      throw new ExecutionError(
        `Estimate: ${node.node.op} carries absorbed permutes (perf-ledger F-3 / F-8), which only a fused attention executes`,
      );
    }
    const isAlias = aliases.has(node);
    const outputs: TransientOutputSpec[] = node.outputs.map((out) => ({
      name: out.name,
      kind: isAlias ? "alias" : "alloc",
      byteLength: isAlias ? 0 : numel(out.shape) * 4,
      source: isAlias ? node.node.ins[0] : undefined,
      uses: uses.get(out.name) ?? 0,
      pinned: outputNames.has(out.name),
    }));
    const reads: TransientRef[] = node.node.ins.map((name) => ({ kind: "value", name }));
    const outputRefs: TransientRef[] = node.outputs.map((out) => ({
      kind: "value",
      name: out.name,
    }));
    const temps: TransientTempSpec[] = [];
    const dispatches: TransientDispatchSpec[] = [];
    if (isStateAttention(node)) {
      // Copy the execution phase's 2 / 3 dispatches and temporary lifetimes per row block.
      // MUST: scores and stats (when not fused) are both allocated right before QK, in that order,
      // and both are released right after PV — the order of `buildStateAttention` /
      // `buildReadonlyStateAttention`. Allocating stats after QK gives the same plan only while no
      // release falls between QK and the statistics; the copy keeps execution's program instead.
      // The readonly variant (the borrower's cross-attention — ADR 0096; ins is q only) reads K/V
      // from external slots and has no k / v inputs for this step. Read references cover **only the
      // inputs that exist** (referencing a missing name would make the transient plan fail on an
      // undefined reference).
      const [q, k, v] = reads;
      // Execution rejects numeric variants × states form before taking temporaries (same test, same
      // message as execution).
      const [qShape, kShape] = node.inputShapes;
      const readonly = stateReadonly(node.node.attrs, `nodes (${node.node.op})`);
      const variantViolation = stateAttentionVariantViolation(
        readonly
          ? `attention (readonly) [${qShape.join(",")}]`
          : `attention (states) [${qShape.join(",")}] × [${kShape.join(",")}]`,
        readonly ? "readonly" : "states",
        knobs.seats,
      );
      if (variantViolation !== undefined) throw new ExecutionError(variantViolation);
      const blocks = stateAttentionTemps(
        node,
        stateShapes,
        limit,
        attentionRowBlockMaxBytes,
        stateAttentionReduce,
        fullStateStorage,
      );
      for (const temp of blocks) {
        const base = dispatches.length;
        const fused = temp.statsBytes === 0;
        const last = base + (fused ? 1 : 2);
        const scores: TransientRef = { kind: "temp", id: temps.length };
        temps.push({ byteLength: temp.scoreBytes, allocBefore: base, releaseAfter: last });
        dispatches.push({ reads: k === undefined ? [q] : [q, k], writes: [scores] });
        if (fused) {
          dispatches.push({ reads: v === undefined ? [scores] : [scores, v], writes: outputRefs });
        } else {
          const stats: TransientRef = { kind: "temp", id: temps.length };
          temps.push({ byteLength: temp.statsBytes, allocBefore: base, releaseAfter: last });
          dispatches.push({ reads: [scores], writes: [stats] });
          dispatches.push({
            reads: v === undefined ? [scores, stats] : [scores, stats, v],
            writes: outputRefs,
          });
        }
      }
    } else if (node.contract.kind === "attention") {
      // In-node temporaries of a fused attention op (no states field): S, row statistics, i8a8
      // quantisation temporaries.
      const program = fusedAttentionProgram(node, reads, outputRefs, limit, knobs, tokenMajor);
      temps.push(...program.temps);
      dispatches.push(...program.dispatches);
    } else if (node.contract.kind === "groupNorm") {
      // The two in-node temporaries of group_norm (partial sums, per-row mean / rstd) over its
      // five dispatches, with the lifetimes `buildGroupNorm` declares (the same layout function
      // — kernels/group-norm.ts).
      const { groups } = groupNormAttrs(node.node.attrs, `nodes (${node.node.op})`);
      const bytes = groupNormTempBytes(groupNormLayout(node.outputs[0].shape, groups));
      const partials: TransientRef = { kind: "temp", id: temps.length };
      const stats: TransientRef = { kind: "temp", id: temps.length + 1 };
      temps.push({ byteLength: bytes.partials, allocBefore: 0, releaseAfter: 3 });
      temps.push({ byteLength: bytes.stats, allocBefore: 0, releaseAfter: 4 });
      const [x] = reads;
      for (let pass = 0; pass < 2; pass += 1) {
        dispatches.push({ reads: [x, stats], writes: [partials] });
        dispatches.push({ reads: [partials], writes: [stats] });
      }
      dispatches.push({ reads: [x, stats, ...reads.slice(1)], writes: outputRefs });
    } else if (!isAlias && selectionSplitTempBytes(node) > 0) {
      // Reduce from the partial maxima to the smallest index. topk's merge also reads the value
      // bits of the original input.
      const partial: TransientRef = { kind: "temp", id: temps.length };
      temps.push({
        byteLength: selectionSplitTempBytes(node),
        allocBefore: 0,
        releaseAfter: 1,
      });
      dispatches.push({ reads, writes: [partial] });
      dispatches.push({
        reads: node.node.op === "topk" ? [partial, ...reads] : [partial],
        writes: outputRefs,
      });
    } else if (!isAlias) {
      dispatches.push({ reads, writes: outputRefs });
    }
    // Aliases consume their input too (counted in `uses`), so every node releases its inputs with
    // repeats.
    return { outputs, temps, dispatches, releases: node.node.ins };
  };
  const fused = postFusionSteps(
    nodes,
    uses,
    outputNames,
    bindingLimit,
    attentionRowBlockMaxBytes,
    knobs.linear,
    knobs.seats,
    knobs.attentionDevice,
  );
  const program: TransientStepSpec[] = [];
  for (let index = 0; index < nodes.length;) {
    const step = fused.get(index);
    if (step === undefined) {
      program.push(nodeStep(nodes[index]));
      index += 1;
      continue;
    }
    if (step.kind === "node") {
      // The node step that absorbed its neighbours (the linear with its folded producer, the
      // attention with its permutes): its own program on the re-pointed plan (it reads the absorbed
      // nodes' inputs), and no step for the absorbed nodes, whose outputs never exist.
      program.push(nodeStep(step.plan, step.tokenMajor));
      index += nodeStepSpan(step);
      continue;
    }
    assertRowBlockWindowCount(step, nodes[index], limit, attentionRowBlockMaxBytes);
    program.push(fusedTransientStep(step, uses, outputNames));
    index += step.nodeCount;
  }
  return planTransients(program, {
    maxBufferSize: maxBufferSize ?? Math.max(CORE_TRANSIENT_LIMITS.maxBufferSize, bindingLimit),
    maxStorageBufferBindingSize: bindingLimit,
    offsetAlignment: CORE_TRANSIENT_LIMITS.offsetAlignment,
  }).totalBytes;
};

/**
 * Estimates the memory one Session needs (ADR 0070 decision 5 — a pure function that needs neither
 * the GPU nor a device). `PreparedModel.estimate` calls this same function.
 *
 * It makes no decision: it returns only bytes per category and the list of what is not accounted
 * for. The final go / no-go check remains the out-of-memory errorScope, and execution does not stop
 * when this estimate is exceeded.
 *
 * It is also **the per-graph estimate entry**: a caller that has finished loading and let go of the
 * initializer bytes (the models pipelines discard the `PreparedModel` and keep only the `IrGraph`)
 * passes the graph it holds, the plan from `planWeightResidency(graph)` and `{capacity,
 * chunkLength}` in `options.generation`, and gets the same {@link AdmissionReport}.
 *
 * MUST: **a caller that has the residency plan passes it** (`PreparedModel` computes it once at
 * prepare time and shares it with construction). Rebuilding a plan other than the one used for
 * construction would bring back the form where "the residency classes used for the estimate" and
 * "the ones actually uploaded" may be different computation results.
 * A caller without it (a models pipeline that discards the `PreparedModel` and keeps only the
 * `IrGraph`) may compute it with `planWeightResidency(graph)` — for the same graph the plan is the
 * same pure function, so the residency classes cannot disagree (exactly why `mod.ts` exposes these
 * two functions).
 *
 * MUST: take the `graph` passed in from `PreparedModel.graph` (a validated graph that passed the
 * same checks as construction — the capability check and the contract checks). The checks below
 * run inside this function, but `planWeightResidency(graph)` runs **earlier**, during argument
 * evaluation, so with an unvalidated graph the inside of the residency plan (the conv1d attrs and
 * payload length checks) fails first and the failure message does not name `nodes[i] (op)` (failing
 * loudly itself is preserved).
 *
 * @param options.bindings symbolic dimensions on the graph-input side (the same bindings passed to a
 *   run). A shape that needs an unbound symbol fails loudly. Only the symbols of the physical chunk
 *   rows `M` are **not accepted here** (their binding points are `options.generation.chunkLength`
 *   and the decode shape's 1 — {@link planBindings}).
 * @param options.generation the spec of the `GenerationContext` to estimate. When omitted,
 *   `resident.stateBytes` is 0 and there is one scenario, `"run"` (it cannot be omitted for
 *   states-form graphs — the plan of the intermediate peak requires the slots' resolved shapes).
 *   When given, there are two scenarios, `"prefill"` / `"decode"`, and `generation.scenarios`
 *   follows them in declaration order ({@link AdmissionScenarioSpec}). The shapes of
 *   `generation.chunkBuckets` are not listed as scenarios and enter only the maximum of
 *   {@link AdmissionReport.peakAccountedBytes}.
 * @param options.maxStorageBufferBindingSize the device's granted limit. Required for graphs with
 *   states-form attention (the row-block count of the in-node temporaries depends on it alone). Also
 *   required for fused attention ops whose count depends on the binding limit (shapes that do not
 *   fit in one block under the WebGPU core default and whose budget is not at most the core
 *   default). Also used as the binding limit of intermediate placement for every graph (the WebGPU
 *   core default when unspecified).
 * @param options.planBackingBudgetBytes the slot-backing keep budget (the same value as
 *   `SessionOptions`). It becomes one term of {@link AdmissionReport.peakAccountedBytes}.
 * @param options.attentionRowBlockMaxBytes the budget per row-block temporary (the same value as
 *   `SessionOptions` — ADR 0124). The row-block count is decided by the same derivation as
 *   execution, and a budget smaller than a single row fails loudly with the same message as
 *   execution (the way to learn of a budget violation before the weights are loaded).
 * @param options.linearCompute the execution form of linear (the same value as `SessionOptions`).
 *   With `"a8"` a producer folded into an i8a8 linear (perf-ledger F-4) is not counted, as in
 *   execution.
 */
export const estimateGraphMemory = (
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  options: EstimateOptions,
): AdmissionReport => {
  // MUST: run **the same checks** as the real construction (`createSessionFromContainer`) first
  // (before the estimate body — the caller has already computed the residency plan `residency`, so
  // only callers going through `PreparedModel`, or passing the validated `PreparedModel.graph`,
  // guarantee the order "check, then plan").
  // Returning an estimate for a configuration that cannot be built would let the estimator alone
  // claim a plausible total for a model with an unsupported storage dtype or op (e.g. a storage
  // dtype in the IR vocabulary but missing from RUNTIME_SUPPORT makes Session construction always
  // fail — `bf16` was such a case until ADR 0131 added it). Sharing the checks also closes the gap
  // when the vocabulary grows.
  // NOTE: this is a capability check, not a comparison with the free side, so it does not violate
  // the "no comparison" rule of ADR 0070 decision 5 (it touches no GPU and stays a pure function).
  assertRuntimeSupport(graph, RUNTIME_SUPPORT);
  // Contracts that cannot be decided without the whole graph (the count and terminal position of
  // state_append, the Tmax shape, the static axes of slice / flip, cat's concatenation axis, the
  // inputs' semantic dtypes) also run at this point. `PreparedModel.estimate` thus runs them a
  // second time after construction, but both are pure, idempotent and cost one graph scan.
  validateGraphContracts(graph);
  const stateAttentionReduce = options.stateAttentionReduce ?? "sequential";
  if (typeof stateAttentionReduce !== "string") {
    throw new ExecutionError(
      `options.stateAttentionReduce must be a string (${typeof stateAttentionReduce})`,
    );
  }
  if (!Object.hasOwn(STATE_ATTENTION_REDUCES, stateAttentionReduce)) {
    throw new ExecutionError(
      "options.stateAttentionReduce '" + stateAttentionReduce + "' is not supported",
    );
  }
  // MUST: check the ranges here, regardless of the graph's shape. Checking only inside
  // stateAttentionTemps would silently accept -1 / 1.5 / NaN for graphs without states-form
  // attention.
  assertLimitOption("maxStorageBufferBindingSize", options.maxStorageBufferBindingSize);
  assertLimitOption("maxBufferSize", options.maxBufferSize);
  // MUST: check the budget's range **here** too (the same check as `createContainerSession` — ADR
  // 0095 decision 1). Placing it only where it is read (the peak below) would let the estimate alone
  // return numbers for a budget with which no Session can be built.
  const budgetBytes = options.planBackingBudgetBytes ?? DEFAULT_PLAN_BACKING_BUDGET_BYTES;
  if (!Number.isSafeInteger(budgetBytes) || budgetBytes < 0) {
    throw new ExecutionError(
      `options.planBackingBudgetBytes ${describeInputValue(
        budgetBytes,
      )} must be a non-negative safe integer`,
    );
  }
  // The budget per row-block temporary is checked at the same point with the same message (the
  // check of `sessionOptionsViolation` — ADR 0124 decision 5). It is checked even for graphs with no
  // row-block split (the location of the range check must not depend on the graph's shape).
  const attentionRowBlockMaxBytes = options.attentionRowBlockMaxBytes;
  if (
    attentionRowBlockMaxBytes !== undefined &&
    (!Number.isSafeInteger(attentionRowBlockMaxBytes) || attentionRowBlockMaxBytes < 1)
  ) {
    throw new ExecutionError(
      `options.attentionRowBlockMaxBytes ${describeInputValue(
        attentionRowBlockMaxBytes,
      )} must be a positive safe integer`,
    );
  }
  // The compute form of fused attention and the storage form of S go through the same check as
  // Session construction (spelling, the c16 × s16 pair) (ADR 0124 decision 7 — the estimate alone
  // returns no numbers for options with which no Session can be built).
  // The execution form of linear goes through the same check (it decides the linear prologue).
  const seatViolation = sessionOptionsViolation({
    attentionCompute: options.attentionCompute,
    attentionScoreStorage: options.attentionScoreStorage,
    linearCompute: options.linearCompute,
    fullStateStorage: options.fullStateStorage,
  });
  if (seatViolation !== undefined) throw new ExecutionError(seatViolation);
  const seats: AttentionSeats = {
    attentionCompute: options.attentionCompute ?? "f32",
    attentionScoreStorage: options.attentionScoreStorage ?? "f32",
  };
  const linear: LinearKnobs = {
    linearCompute: options.linearCompute ?? "f32",
    weightLayouts: fusionWeightLayouts(residency),
  };
  // The device facts are checked here regardless of the graph's shape (the rule of
  // {@link assertLimitOption}), before the fusion pass reads them.
  // MUST: "none" without them fails loudly with this message whatever the graph — the route (and
  // with it whether S exists and whether the output permute is absorbed) depends on the device, and
  // assuming one would let the estimate claim temporaries the device does not produce (the same
  // reason as the binding-limit requirement of {@link stateAttentionTemps}). Any other seat never
  // reads the facts, so the stand-in passes.
  const passedDevice = options.attentionDevice;
  if (passedDevice === undefined && seats.attentionScoreStorage === "none") {
    throw new ExecutionError(
      "attentionScoreStorage 'none' needs options.attentionDevice" +
        " (the online kernel's route is decided by the device's subgroups and workgroup-storage limit — ADR 0132)",
    );
  }
  if (passedDevice !== undefined) {
    if (typeof passedDevice.subgroups !== "boolean") {
      throw new ExecutionError(
        `options.attentionDevice.subgroups must be a boolean (${typeof passedDevice.subgroups})`,
      );
    }
    assertLimitOption(
      "attentionDevice.maxComputeWorkgroupStorageSize",
      passedDevice.maxComputeWorkgroupStorageSize,
    );
  }
  const attentionDevice = passedDevice ?? UNREAD_ATTENTION_DEVICE;
  const chunkDims = chunkRowDims(graph);
  const chunkSymbols = new Set(chunkDims.map((dim) => parseDim(dim).sym));
  const bindings = planBindings(graph, options.bindings, chunkSymbols);
  const fullStateStorage = options.fullStateStorage ?? "f32";
  const state = stateEstimate(graph, options.generation, options, fullStateStorage);
  // Under the expert cache: the pools in place of the expert weights, and the graph that also reads
  // every routed index back (the same plan as construction — expert-plan.ts).
  const experts =
    options.expertCache === undefined
      ? undefined
      : planExpertGraph(graph, residency, options.expertCache);
  const planned = experts?.graph ?? graph;
  const counted = weightEstimate(experts?.residency ?? residency);
  const weights =
    experts === undefined
      ? counted
      : {
          ...counted,
          uncompressed:
            counted.uncompressed + experts.layers.length * toSizeClass(experts.tableBytes),
        };

  const chunkLength = options.generation?.chunkLength;
  if (chunkLength === undefined && chunkDims.length > 0) {
    // A graph with states-form nodes does not run without a GenerationContext (ADR 0066 decision 1).
    // Without failing here, it would fail in planGraph's dimension evaluation with the chunk symbol
    // unbound, as a DimError that does not name the binding point.
    throw new ExecutionError(
      `The physical chunk-row symbols M [${chunkDims.join(", ")}] can only be bound by options.generation.chunkLength` +
        " (a graph with states-form nodes needs a GenerationContext)",
    );
  }
  // Compute the two execution shapes of ADR 0066 decision 4 independently (the same graph,
  // differing only in chunk rows).
  const plans: [name: string, bindings: SymbolBindings][] =
    chunkLength === undefined
      ? [["run", bindings]]
      : [
          ["prefill", bindChunkRows(bindings, chunkDims, chunkLength, "prefill")],
          ["decode", bindChunkRows(bindings, chunkDims, 1, "decode")],
        ];
  // Shapes added by the caller (ADR 0096's verify run and the like) follow the existing two in
  // declaration order. The computation path is the same single one; only the physical chunk rows
  // and (if passed) the graph-input bindings change.
  const generation = options.generation;
  if (generation?.scenarios !== undefined) {
    // MUST: duplicate names fail loudly (the three reserved names + extra names among themselves).
    // With two entries of the same name, a caller looking up by name silently sees only the first.
    const taken = new Set<string>(["run", "prefill", "decode"] satisfies AdmissionScenarioName[]);
    for (const spec of generation.scenarios) {
      if (taken.has(spec.name)) {
        throw new ExecutionError(
          `Scenario name '${spec.name}' already exists (the reserved names are run / prefill / decode)`,
        );
      }
      taken.add(spec.name);
      // MUST: the range is `1..chunkLength` (the same bound as the `queryLength ≤ chunkLength`
      // contract — no estimate for a row count that does not fit the declared shape).
      if (
        !Number.isSafeInteger(spec.chunkLength) ||
        spec.chunkLength < 1 ||
        spec.chunkLength > generation.chunkLength
      ) {
        throw new ExecutionError(
          `Scenario '${spec.name}': physical chunk row count ${spec.chunkLength} is not an integer in ` +
            `1..${generation.chunkLength} (generation.chunkLength)`,
        );
      }
      // Override a copy of the default bindings, then run the same check as the existing two shapes
      // (`planBindings`).
      const base = planBindings(graph, { ...options.bindings, ...spec.bindings }, chunkSymbols);
      plans.push([spec.name, bindChunkRows(base, chunkDims, spec.chunkLength, spec.name)]);
    }
  }

  // The shapes of `chunkBuckets` (ADR 0066 addendum 10, buckets) are internal plans that enter only
  // the peak and are not listed among the public scenarios (the list callers look up by name does
  // not move — ADR 0095 addendum of 2026-10-10).
  // MUST: do not let the largest-M prefill shape alone represent the peak. The equal split into row
  // blocks (planRowBlocks) makes the shape with fewer rows use more temporaries at the boundary
  // where the block count changes from 1 to 2 (e.g. 128 MiB for one block of 32 rows vs about
  // 68 MiB for 33 rows in two blocks of 17 + 16), so the intermediates are not monotonic in M.
  // Computing every allowed physical row count and taking the maximum keeps the assumption intact
  // even if the split rule changes.
  const buckets: [name: string, bindings: SymbolBindings][] = (generation?.chunkBuckets ?? []).map(
    (rows) => {
      const name = `bucket ${rows}`;
      return [name, bindChunkRows(bindings, chunkDims, rows, name)];
    },
  );

  const allowedRows =
    generation === undefined
      ? undefined
      : allowedChunkRows(generation.chunkLength, generation.chunkBuckets ?? []);
  const estimateScenario = ([name, scenarioBindings]: [
    name: string,
    bindings: SymbolBindings,
  ]): AdmissionScenario => {
    // MUST: a symbol that appears in both states and inputs has the same value at both binding
    // points (the estimate alone returns no normal value for a split a run rejects — ADR 0066
    // addendum 7). The resolved values change per scenario, so the check is per scenario too.
    if (state.bindings !== undefined) assertGenerationBindings(state.bindings, scenarioBindings);
    const plan = planGraph(graph, scenarioBindings, state.shapes);
    // MUST: match the resolved physical chunk rows against the same allowed set as the pre-run
    // check (the same single function — `assertChunkRowsAllowed`). A numerically fixed `M` or an
    // extra scenario's row count outside the set is always rejected by the run's first call, so the
    // estimate alone returns no normal value.
    if (allowedRows !== undefined) assertChunkRowsAllowed(stateChunkRows(plan.nodes), allowedRows);
    // Input buffers and output readback staging use the same floored formula (`#bindInput` /
    // `#activateBacking` / `#stageOutputs` in the executor — `Math.max(4, numel×4)` equals
    // toSizeClass).
    let ioBytes = 0;
    for (const spec of graph.inputs) {
      ioBytes += toSizeClass(numel(shapeOf(plan.shapes, spec.name)) * 4);
    }
    for (const name of planned.outputs) {
      ioBytes += toSizeClass(numel(shapeOf(plan.shapes, name)) * 4);
    }
    return {
      name,
      ioBytes,
      workspaceBytes: transientSlotBytes(
        planned,
        plan.nodes,
        state.shapes,
        options.maxStorageBufferBindingSize,
        options.maxBufferSize,
        {
          stateAttentionReduce,
          fullStateStorage,
          attentionRowBlockMaxBytes,
          seats,
          attentionDevice,
          linear,
        },
      ),
    };
  };
  const scenarios = plans.map(estimateScenario);

  const weightBytes = weights.compressed + weights.uncompressed + weights.expanded;
  const peakScenarioBytes = Math.max(
    ...[...scenarios, ...buckets.map(estimateScenario)].map(
      (scenario) => scenario.ioBytes + scenario.workspaceBytes,
    ),
  );
  return {
    resident: {
      weights: {
        compressedBytes: weights.compressed,
        uncompressedBytes: weights.uncompressed,
        expandedBytes: weights.expanded,
        totalBytes: weightBytes,
      },
      stateBytes: state.bytes,
    },
    scenarios,
    planBackingBudgetBytes: budgetBytes,
    // The kept set never exceeds `max(budget, largest single backing)` (`#evictBackingsFor` in the
    // executor).
    peakAccountedBytes: weightBytes + state.bytes + Math.max(budgetBytes, peakScenarioBytes),
    unaccounted:
      experts === undefined
        ? UNACCOUNTED
        : experts.hostBytes === undefined
          ? EXPERT_UNACCOUNTED
          : EXPERT_HOST_UNACCOUNTED,
  };
};
