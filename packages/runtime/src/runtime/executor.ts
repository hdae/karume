/**
 * Minimal graph executor.
 *
 * Structure: "plan (pure function — plan.ts) → **derive** (recipe building —
 * src/runtime/recipe-builder.ts) → **execute** (encode the recipe list with a generic loop) →
 * submit → read back only the graph outputs".
 * Every output shape is fixed before a single dispatch is issued (static shapes — ADR 0004), and
 * everything unexpected fails loudly (unsupported op / dtype outside the contract / impossible
 * broadcast / unbound symbol / declaration mismatch).
 *
 * MUST: the derivation phase **does not touch run-lifetime state** (it neither allocates a
 * {@link RunArena} nor issues dispatches). Because this holds, the derivation phase's product can
 * be kept resident in the Session keyed by the resolved bindings ({@link PreparedPlan}) — from the
 * second run with the same bindings on, planning, fusion matching and recipe derivation are skipped
 * entirely and the recipe list goes straight to the execution phase.
 *
 * MUST: initialization is an explicit async stage ({@link PreparedModel.createContainerSession}). A
 * synchronous upload loop inside a constructor makes pipelining weight fetch and upload
 * structurally impossible.
 * MUST: unsubmitted encoding is always settled before buffers are destroyed — the success path
 * submits (flush when a wait is needed, `submitPending` for a run that folds its wait into the
 * readback `mapAsync`), the failure path discards (discard-or-flush before destroy). Submitting an
 * encoding that references a destroyed buffer fails the whole command buffer, and even unrelated
 * dispatches riding on the same scheduler stay unexecuted while wrong values silently remain.
 */

import {
  assertTernaryCodes,
  type BoundContainer,
  type InitializerSupply,
  mergedGraph,
} from "../format/container/bind.ts";
import { groupScaleShape } from "../format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";
import { type ArenaStats, RunArena, STORAGE_USAGE } from "../gpu/arena.ts";
import {
  assertDeviceUsable,
  type BatchFailureOrigin,
  type BatchReadSource,
  BatchScopeError,
  type GpuContext,
  ResidentTensor,
  ResidentTensorError,
  RUNTIME_INTERNAL,
} from "../gpu/device.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { MAP_MODE } from "../gpu/webgpu-constants.ts";
/**
 * Storage format of S ({@link SessionOptions.attentionScoreStorage} — re-exported so the public
 * surface can name it).
 */
export type { ScoreStorage } from "../kernels/score-storage.ts";
import { numel, RUNTIME_SUPPORT } from "../ops.ts";
import { assertRuntimeSupport } from "../ops/support.ts";
import { type AdmissionReport, estimateGraphMemory, type EstimateOptions } from "./estimate.ts";
import {
  type ExecStep,
  type FusionCounts,
  type FusionWeightLayout,
  planFusions,
} from "./fusion.ts";
import { GenerationContext, type GenerationContextHost } from "./generation-context.ts";
/**
 * Owner of one generation's mutable state (ADR 0066 decision 1). Public **as a type only** — the
 * entry points are {@link Session.createGenerationContext} and {@link GenerationContext.fork}
 * alone; constructing it directly would bypass the errorScope check (detection of allocation
 * failure) and the capacity guard (addendum 5) (the same convention as `GpuContext` /
 * `ResidentTensor`).
 */
export type { GenerationContext } from "./generation-context.ts";
/**
 * Fusion application counts per rule (the type of {@link SessionDiagnostics.lastRunFusions} —
 * re-exported so the public surface can name it. Treated like `ScoreStorage`: not a pass-through
 * re-export, it only puts **a component of a type that is already public** on the surface).
 *
 * NOTE: the key set is the set of fusion rules itself, so adding one rule adds a key (as a type
 * this can be a breaking change). It is already exposed structurally through `SessionDiagnostics`,
 * so giving it a name does not change the effective compatibility surface.
 */
export type { FusionCounts } from "./fusion.ts";
import {
  assertGenerationBindings,
  bindSymbols,
  describeInputValue,
  ExecutionError,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
  validateGraphContracts,
} from "./plan.ts";
import {
  assertGenerationRun,
  bakeAllBindGroups,
  bakeBindGroups,
  type BakedGroups,
  bakeGenerationBindGroups,
  executeBakedPlan,
  executeBakedPlanPaced,
  type GenerationEncoding,
  type GenerationLimits,
  pipelineCensus,
  planRecipes,
  type StepRecipe,
  type ValueBinding,
} from "./recipe.ts";
import type { TransientPlan } from "./transient-plan.ts";
import { type AddedParams, RecipeBuilder } from "./recipe-builder.ts";
import { planWeightResidency, SharedWeight, type WeightResidency } from "./weight-residency.ts";
import {
  buildSessionState,
  HOST_WRITTEN_USAGE,
  type PreparedPlan,
  type ReadyInitializer,
  type ReadyScale,
  type SessionState,
  type WeightBatch,
} from "./session-build.ts";
import type {
  EnqueueOptions,
  EnqueueRead,
  GenerationContextSpec,
  GenerationRun,
  ParamsCacheStats,
  PreparedPlanStats,
  RunInput,
  RunInputs,
  RunOutputs,
  SessionDiagnostics,
  SessionOptions,
  Tensor,
} from "./session-types.ts";
export type {
  ComputePrecision,
  EnqueueOptions,
  EnqueueRead,
  GenerationContextSpec,
  GenerationRun,
  I8a8Dot,
  LinearGemvReduce,
  ParamsCacheStats,
  PlanBackingStats,
  PreparedPlanStats,
  RmsNormReduce,
  RunInput,
  RunInputs,
  RunOutputs,
  SessionBuildStats,
  SessionDiagnostics,
  SessionOptions,
  StateAttentionReduce,
  StateBackingStats,
  StorageDiagnostics,
  Tensor,
} from "./session-types.ts";
export { I8A8_DOT, ROW_BLOCK_SPLIT } from "./session-types.ts";
/**
 * Opaque reference to a lent weight (ADR 0096 decision 2). Public **as a type only** — the entry
 * point is {@link Session.exportWeight} alone; constructing it directly would bypass the
 * residency-class match and the borrow count (the same convention as `Session` /
 * `GenerationContext`).
 */
export type { SharedWeight } from "./weight-residency.ts";

/** Host-side TypedArray per semantic dtype (used by diagnostics and input checks). */
const HOST_ARRAY: Readonly<
  Record<IrDtype, Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor>
> = {
  f32: Float32Array,
  i32: Int32Array,
  bool: Uint32Array,
};

/**
 * Builds the readback Tensor from the declared dtype (maps discriminant and array type in one
 * place).
 */
const hostTensor = (
  dtype: IrDtype,
  shape: readonly number[],
  buffer: ArrayBuffer,
  count: number,
): Tensor => {
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: new Float32Array(buffer, 0, count) };
    case "i32":
      return { dtype, shape, data: new Int32Array(buffer, 0, count) };
    case "bool":
      return { dtype, shape, data: new Uint32Array(buffer, 0, count) };
  }
};

/**
 * Total bytes of the intermediates alive at the end of a run = the pinned graph outputs
 * (`ArenaStats.transientBytes`).
 */
const pinnedBytes = (plan: TransientPlan): number =>
  [...plan.pinned].reduce((total, slot) => total + plan.slots[slot].byteLength, 0);

/**
 * The bundle of inputs fixed at issue time (the product of {@link captureInputs}). The execution
 * body reads **only this copy** and never looks up the caller's `inputs` Record again.
 */
type CapturedInputs = {
  /**
   * Name → input value. The purpose is to **fix the Record's member identity here**; the container
   * is a `Map` so that even an input named "__proto__" is looked up plainly (indexing
   * `inputs[name]` can pick up a value inherited from the prototype).
   */
  readonly values: ReadonlyMap<string, RunInput>;
  /**
   * A **copy** of the shapes of host-array inputs (the sole input of binding resolution =
   * {@link bindSymbols}).
   */
  readonly inputShapes: Record<string, readonly number[]>;
  /** The table of resident inputs only (binding reservations, plan key, `deferredInputs`). */
  readonly residentInputs: ReadonlyMap<string, ResidentTensor>;
};

/**
 * Copies the input bundle at issue time and splits it into "the shape table of host arrays" and
 * "the table of resident inputs".
 *
 * MUST: called only in the **synchronous section** of {@link Session.run} /
 * {@link Session.enqueue}. The execution body runs behind `#serialize` (one microtask or later), so
 * without a copy here the body would see "a Record / shape array the caller touched right after
 * issuing" — a swapped member is a silent wrong value, a rewritten shape "happens to fail loudly"
 * (both are shapes one walks into naturally with the non-awaited concurrent issue the run JSDoc
 * recommends). Only cheap metadata is copied; `Tensor.data` (the TypedArray itself) **stays
 * borrowed** — the contract is the "input lifetime" section of run.
 * MUST: containers that accumulate keys derived from the caller's inputs have a null prototype (the
 * same reason as bindSymbols in plan.ts). With a plain `{}`, an input named "__proto__" turns its
 * shape array into the [[Prototype]], no own property is created, and the run fails with "input not
 * passed". The resident side is a `Map`, so it has no such hole.
 */
const captureInputs = (inputs: RunInputs): CapturedInputs => {
  const values = new Map<string, RunInput>();
  const inputShapes: Record<string, readonly number[]> = Object.create(null);
  const residentInputs = new Map<string, ResidentTensor>();
  for (const [name, value] of Object.entries(inputs)) {
    values.set(name, value);
    if (value instanceof ResidentTensor) residentInputs.set(name, value);
    // MUST: keep a **copy** of the shape (if it stayed borrowed, a rewrite right after issue would
    // change the input of binding resolution). A resident input is a lifetime class without a
    // host-side shape, so there is nothing to copy.
    else inputShapes[name] = [...value.shape];
  }
  return { values, inputShapes, residentInputs };
};

/**
 * The name set of resident inputs (`deferredInputs` of {@link bindSymbols}). Not passed when empty.
 */
const residentNames = (
  residentInputs: ReadonlyMap<string, ResidentTensor>,
): ReadonlySet<string> | undefined =>
  residentInputs.size === 0 ? undefined : new Set(residentInputs.keys());

/**
 * Guard for whether a resident tensor may be bound (disposed / owned by a different GpuContext).
 *
 * MUST: binding a destroyed buffer is a createBindGroup validation failure; it does not throw, the
 * dispatch becomes a no-op and every output is 0 (this is the only synchronous detection point).
 * MUST: the owning GpuContext is checked too. Resident ids are numbered independently per
 * GpuContext, and only that number goes into the prepared-plan key / backing signature, so a
 * resident of another context with the same id and size **collides on the key**. A hit run (baked
 * backing) never touches the value passed in, so it reads the previous context's stale value with
 * no exception and no warning (only the miss path happens to fail on the device-mismatch
 * validation = detection disappears the moment the cache hits).
 * This is the same check `BatchScope` performs on the owner.
 */
const assertResidentUsable = (resident: ResidentTensor, gpu: GpuContext, where: string): void => {
  if (resident.disposed) {
    throw new ExecutionError(`${where}: 破棄済みの常駐テンソル '${resident.label}' は使えない`);
  }
  if (resident[RUNTIME_INTERNAL].owner !== gpu) {
    throw new ResidentTensorError(
      `${where}: 常駐テンソル '${resident.label}' は別の GpuContext が所有している` +
        "（識別子は GpuContext ごとの独立採番なので、跨いで渡すと導出済み計画のキーが衝突する）",
    );
  }
};

/**
 * Bundles the context-owned buffers and the logical lengths for one run
 * ({@link GenerationEncoding}).
 *
 * MUST: the logical lengths are **the values the caller read once** (`context.pastLength` is not
 * read again here). The value written to the uniform, the dispatch count and the capacity check all
 * coming from one single value is what makes "the range the GPU scans" equal "the workgroup count
 * the host dispatched".
 */
const generationEncoding = (
  context: GenerationContext,
  past: number,
  query: number,
): GenerationEncoding => ({
  slots: new Map([...context[RUNTIME_INTERNAL].slots].map(([name, slot]) => [name, slot.buffer])),
  lengths: context[RUNTIME_INTERNAL].lengths,
  past,
  query,
});

/**
 * Returns every resident-input binding reservation of the run in progress (lifts the hold on
 * {@link ResidentTensor.dispose}).
 *
 * MUST: called exactly once on both the success and the failure path. A missed return leaves that
 * resident tensor undisposable for the whole Session lifetime; a double return fails loudly as a
 * bookkeeping violation.
 */
const releaseBoundResidents = (bound: ResidentTensor[]): void => {
  for (const resident of bound) resident[RUNTIME_INTERNAL].releaseBound();
  bound.length = 0;
};

/**
 * Takes use reservations on resident tensors in the synchronous section of issue (the mechanism is
 * `ResidentInternals.retainUse` in gpu/context.ts).
 *
 * MUST: called only in the **synchronous section** of {@link Session.run} /
 * {@link Session.enqueue}. The binding reservation ({@link Session.#bindInput}) is delayed until
 * the execution body, so on its own it leaves a window, for a resident with `bakedRefs === 0`, in
 * which "a run the API accepted exists, yet a `dispose()` in the same tick goes through".
 * MUST: the targets are **both** the resident input tensors and the `copyOutputs` destinations (an
 * accepted run uses both). When the same tensor appears in two places (a loop that takes the
 * destination as the next input), two reservations are taken and two are returned.
 * MUST: if one of them fails midway, the ones already taken are returned **in reverse order**
 * before rethrowing (releasing in reverse order of acquisition is the lease discipline — a missed
 * return leaves that resident tensor undisposable forever).
 */
const retainUsedResidents = (
  targets: readonly (readonly [where: string, resident: ResidentTensor])[],
  gpu: GpuContext,
): ResidentTensor[] => {
  const used: ResidentTensor[] = [];
  try {
    for (const [where, resident] of targets) {
      // Disposed / foreign-context tensors fail before the reservation is taken (with the same
      // message as the body's check; only the failure point moves forward into the synchronous
      // section of issue).
      assertResidentUsable(resident, gpu, where);
      resident[RUNTIME_INTERNAL].retainUse();
      used.push(resident);
    }
  } catch (cause) {
    releaseUsedResidents(used);
    throw cause;
  }
  return used;
};

/**
 * Returns every use reservation (in reverse order of acquisition).
 *
 * MUST: called exactly once when a run / enqueue settles, on both the success and the failure path
 * (a missed return leaves that resident tensor undisposable; a double return fails loudly as a
 * bookkeeping violation).
 */
const releaseUsedResidents = (used: ResidentTensor[]): void => {
  for (let index = used.length - 1; index >= 0; index -= 1) {
    used[index][RUNTIME_INTERNAL].releaseUse();
  }
  used.length = 0;
};

/**
 * Looks up a resolved shape in the execution plan. plan.shapes holds every input, initializer and
 * node output without exception (an invariant of planGraph), so a missing entry is a violation of a
 * runtime-internal invariant.
 * MUST: do not degrade to the empty shape (scalar) — it would pass as 1 element, and execution
 * would continue with both the allocation size and the element-count check silently wrong.
 */
const resolvedShape = (
  shapes: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] => {
  const shape = shapes.get(name);
  if (shape === undefined) {
    throw new ExecutionError(`値 '${name}' の解決済み shape が実行計画に無い`);
  }
  return shape;
};

/**
 * Byte size of one graph output. **This is the only formula for the size of an output slot**: the
 * readback staging ({@link Session.#stageOutputs}), the source read at settlement
 * ({@link Session.#planReadback}) and the destination check of `copyOutputs`
 * ({@link Session.#planCopyOutputs}) all read it.
 *
 * MUST: even 0 elements are rounded up to the arena's smallest size class (4 bytes). A copy size
 * must not exceed the actual size of either buffer, and the output side is allocated with the same
 * floor.
 */
const outputByteSize = (count: number): number => Math.max(4, count * 4);

const assertGenerationCommit = (capturedGeneration: GenerationRun | undefined): void => {
  // MUST: the value range of `commit` is checked in the synchronous section of issue. Falling back
  // to the default for a misspelling that came from outside the type would let a run issued as
  // deferred silently advance the logical length (a position shift with no exception and no
  // warning).
  if (
    capturedGeneration?.commit !== undefined &&
    capturedGeneration.commit !== "immediate" &&
    capturedGeneration.commit !== "deferred"
  ) {
    throw new ExecutionError(
      `run: generation.commit ${describeInputValue(
        capturedGeneration.commit,
      )} は 'immediate' か 'deferred' のみ`,
    );
  }
  // MUST: a deferred run may write at most **the slack** rows into the sliding ring
  // (`Q ≤ slidingSlack`). A rejected row j (j ≥ m for m committed rows) overwrites logical column
  // `P+j−C` in the physical ring, so it must fall below the lower edge of the readers' window after
  // commit. There are two kinds of readers, and the one with the lower edge is **the borrower (the
  // readonly reader — the drafter)** at `P+m−W` (the readonly section of
  // `src/kernels/state-attention.ts`, `column_base = P − min(P, W)`; a states-form reader sits one
  // column higher at `P+m−(W−1)` because of this step's ins). Satisfying `P+j−C < P+m−W` ⟺
  // `j − m < C − W = slidingSlack` for every `m ≥ 0` (including `commit(0)`) and `j ≤ Q−1` requires
  // `Q ≤ slidingSlack`.
  // MUST: do not branch on whether a borrower exists (a borrower can also be opened **after** a
  // deferred run + `commit(0)`, so relaxing on its presence at run issue leaves a hole for "open
  // the borrower later"). When only the lender's own readers exist this is just one column stricter
  // and creates no broken shape.
  // A run beyond this overwrites past KV with neither an exception nor NaN, so it fails in the
  // synchronous section of issue. An immediate run (prefill / decode) commits every row, so its
  // limit stays `chunkLength`.
  if (capturedGeneration?.commit === "deferred") {
    // A borrowing context has no logical length (advancing and committing belong to the lender —
    // ADR 0096, addendum of 2026-09-08 on stage 2), so a deferred run with nothing to commit to
    // fails in the synchronous section of issue.
    if (capturedGeneration.context[RUNTIME_INTERNAL].borrowing) {
      throw new ExecutionError(
        "run: 借り手 context の run に generation.commit 'deferred' は指定できない" +
          "（借り手は論理長を進めないので確定させる相手が居ない）",
      );
    }
    const slack = capturedGeneration.context.slidingSlack;
    if (slack !== undefined && capturedGeneration.queryLength > slack) {
      throw new ExecutionError(
        `run: deferred な generation run の queryLength ${capturedGeneration.queryLength} が ` +
          `sliding ring の余裕 ${slack} を超える（棄却行が live な過去 KV を潰す）`,
      );
    }
  }
};

/**
 * The product of only the first half of the derivation phase (plan → fusion matching). The second
 * half (recipe derivation) involves the params writeBuffer and pipeline creation and can only be
 * produced inside the run's errorScope section, so until then it is carried as a step list.
 * MUST: told apart from {@link PreparedPlan} by the presence of `recipes` (`in`).
 */
type PlannedSteps = {
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly fusions: FusionCounts;
  readonly steps: readonly ExecStep[];
};

/**
 * The result of preparing a run / enqueue for issue ({@link Session.#prepareInvocation}) — "what
 * this one executes" is all here, and the later branches (backed / arena, single fence / two-stage
 * wait) are the body's concern.
 */
type PreparedInvocation = {
  /** Slot capacities after binding resolution (undefined for an execution without generation). */
  readonly stateShapes: ReadonlyMap<string, readonly number[]> | undefined;
  /** The logical length read exactly once at the head of this execution (0 without generation). */
  readonly pastLength: number;
  /**
   * The context-side face of a generation run. **Context and encoding are bundled into one
   * variable** because the baking path ({@link Session.#generationGroups}) needs both at once —
   * held separately, the type could not exclude the impossible pair "only one of them undefined",
   * and a swallowing branch would grow.
   */
  readonly face:
    | { readonly context: GenerationContext; readonly encoding: GenerationEncoding }
    | undefined;
  readonly preparedKey: string;
  /** Prepared-plan hit (undefined = miss; the diagnostics' `hit` is decided by this). */
  readonly prepared: PreparedPlan | undefined;
  /**
   * On a hit that plan, on a miss the product of the first half of derivation (told apart by
   * `recipes` — `in`).
   */
  readonly derived: PreparedPlan | PlannedSteps;
  readonly shapes: ReadonlyMap<string, readonly number[]>;
};

/**
 * Upper bound on the number of resident prepared plans (LRU).
 *
 * It is a fixed constant because this container only catches the locality of "consecutive runs
 * reuse the same bindings"; in the shape where growing it does not help (the shape changes every
 * run) it does not help from the first one either — as a configuration knob it would turn into
 * tuning that fattens a cache that never hits.
 *
 * 4 → 8: a shape appeared in which the prefill shape + the decode shape + several prefill bucket
 * shapes (ADR 0066 addendum 10, buckets) + the existing one-shot surface are **steady at the same
 * time**. When the steady count exceeds the bound, the oldest drops on every iteration of the
 * generation loop and the decode hot path silently falls back to re-derivation (no exception, no
 * warning; only `diagnostics().lastRunPrepared.hit` sticks at false). The argument above (growing
 * does not help when the shape changes every run) does not depend on the count, so it is unchanged.
 *
 * 8 → 12: the verify shape of speculative decoding (a small `M` of the draft length) becomes steady
 * **in addition to** the decode shape and the prefill buckets. The models side's default buckets
 * are 6 (`[4, 8, 32, 64, 128, 256]`) + the fixed `chunkLength` + the decode shape, exactly filling
 * 8, so at 8 the oldest would drop every step as soon as the verify shape is loaded.
 *
 * The cost of one bucket is one PreparedPlan + two WGSL variants of ①ₜ / ③ₜ when M ≥ 16
 * (`PipelineCache` has no eviction and compiles at first use). Buckets below 16 do not take the
 * tiled path and fall to ① / ③, so they add no variant.
 *
 * MUST: do not copy the number (tests and the estimate's doc refer to this constant).
 */
export const PREPARED_PLAN_CAPACITY = 12;

/**
 * Transient slot backing per signature (prepared-plan key).
 *
 * The Session keeps several **under a byte budget** ({@link SessionOptions.planBackingBudgetBytes},
 * perf-ledger H-15): a slot is the run's intermediate buffer itself (~1 GiB scale for a DiT), so
 * keeping them by count would multiply VRAM by the count, but the prefill bucket shapes / decode
 * shape of generation are a few to a few tens of MiB, and rebuilding on every switch (≈ 40 ms)
 * costs more. A large shape that does not fit the budget is kept alone as before (retiring all
 * others). The premise differs from keeping {@link PREPARED_PLAN_CAPACITY} prepared plans (host
 * objects only).
 */
type ActiveBacking = {
  /** The key of the prepared plan this backing belongs to ({@link Session.#preparedKey}). */
  readonly key: string;
  /**
   * **The generation id of this backing instance** (numbered by `Session.#backingBuilds` —
   * monotonically increasing, never reused).
   *
   * MUST: a backing rebuilt under the same `key` always gets a different value. The bind groups a
   * GenerationContext baked (ADR 0066 decision 5) hold this instance's slot / input buffers, so
   * using the key as the id would make a switch to "a backing retired and rebuilt" undetectable,
   * and an old group binding destroyed buffers would be dispatched as is (no exception, no
   * warning).
   */
  readonly build: number;
  /**
   * Resident bytes of the intermediate buffers = the sum of the plan's regions (**input buffers not
   * included**).
   * MUST: do not change this definition — the footprint-invariance check "backed region sum = the
   * miss run's region allocation" (tests/gpu_plan_backing_test.ts) is this value itself (identical
   * because the same plan is used — ADR 0093 decision 6); mixing in the inputs would change what
   * the check observes.
   */
  readonly bytes: number;
  /**
   * Total of the input buffers the backing owns (resident inputs are not owned and not included).
   * The budget ({@link SessionOptions.planBackingBudgetBytes}) counts `bytes + inputBytes` — the
   * VRAM the retained set actually holds. `bytes` is kept separate because the footprint-invariance
   * check (ADR 0093 decision 6) looks at the region sum alone.
   */
  readonly inputBytes: number;
  /**
   * Graph input name → the buffer bound. A plain input is a resident buffer owned by the backing
   * (`HOST_WRITTEN_USAGE`; the size is fixed by the resolved shape, so there is no reason to
   * rebuild it for the same signature); a resident input is the {@link ResidentTensor}'s buffer
   * itself (**not owned by the backing**).
   */
  readonly inputs: ReadonlyMap<string, GPUBuffer>;
  /**
   * The resident inputs this backing references through baking. Retained at construction, released
   * at retirement (the basis for making {@link ResidentTensor.dispose} fail loudly while
   * referenced).
   */
  readonly residents: readonly ResidentTensor[];
  /**
   * Baked bind groups ({@link bakeBindGroups}). A backed run only dispatches them.
   * The positions that bind state are `undefined` holes, filled by the context-side baking
   * (decision 5).
   */
  readonly groups: BakedGroups;
  /**
   * Plan and region index → resident region buffer. The material **for the context-side baking to
   * reproduce the same value resolution**; the Session-side baking (once at construction) no longer
   * needs it.
   * MUST: keep exactly what the baking used (rebuilding and passing it could make the two bakings
   * bind different instances).
   */
  readonly plan: TransientPlan;
  readonly regions: readonly GPUBuffer[];
  /**
   * Graph output name → binding (the readback source — fixed at construction; a backed run builds
   * no env). Readback eligibility (pinned slot / alias of an input) lives in
   * {@link ValueBinding.readable}.
   */
  readonly outputs: ReadonlyMap<string, ValueBinding>;
  /**
   * Every buffer the backing owns (regions + **plain inputs** — used for destruction at
   * retirement).
   * MUST: resident input buffers are not included — the owner is the GpuContext side and the
   * lifetime spans Sessions.
   */
  readonly owned: ReadonlySet<GPUBuffer>;
};

/**
 * What one execution unit (one run / one enqueue) **newly added** to the Session's caches and
 * retained set.
 *
 * A WebGPU allocation failure (non-fatal OOM) does not throw; params, plans and backings are
 * published as soon as they were "created" synchronously. The failure only shows in a later
 * errorScope (a run's own pop, the batch's `finish` for an enqueue), so "what this unit added" is
 * carried until that settlement. What is rolled back depends on the kind:
 * - a backing is retired when the execution unit fails ({@link Session.#retireBuiltBackings} — the
 *   existing discipline of ADR 0095 decision 2. A GPU-side failure of a batch is unknown until
 *   `finish`, so that one is retired at the span's settlement).
 * - params and plans are removed only when a GPU-side failure shows
 *   ({@link Session.#forgetDerived}). On a host-side failure what was created is valid; removing it
 *   would, on every failure that recurs identically (wrong input and the like), allocate params
 *   anew, forget them, and let them pile up until `Session.dispose`.
 * Only what is recorded here is rolled back, and existing healthy entries stay (no rebuild
 * thrashing on unrelated failures).
 *
 * MUST: matching is by identity (params by instance, plans by object, backings by generation number
 * {@link ActiveBacking.build}). The result differs from key matching only when a later run rebuilt
 * or re-registered the same key before the batch's failure settled (a race whose order between the
 * release of the section lock and the finalizer is decided by the number of microtask hops); key
 * matching would roll back the valid other generation that run created. A rebuild inside the same
 * failed batch rolls back the rebuilder's record too, so the result does not change.
 * In other words this is a defence against a hard-to-reach race.
 */
type CacheAdditions = {
  readonly params: AddedParams[];
  readonly prepared: { readonly key: string; readonly plan: PreparedPlan }[];
  readonly backings: { readonly key: string; readonly build: number }[];
};

const emptyAdditions = (): CacheAdditions => ({ params: [], prepared: [], backings: [] });

/**
 * The staging of one readback and the declaration data needed to rebuild it into a host
 * {@link Tensor}. Kept in the same form whatever it is queued on (the run body's command list / a
 * readback-only encoder).
 */
type StagedOutput = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly count: number;
  /** `COPY_DST | MAP_READ` ({@link RunArena.allocHostRead}). Owned by the arena. */
  readonly staging: GPUBuffer;
};

/**
 * One checked `copyOutputs` entry ({@link Session.#planCopyOutputs} — the buffer is not resolved
 * yet).
 */
type PlannedCopy = {
  readonly name: string;
  readonly target: ResidentTensor;
  readonly size: number;
};

/** One `copyOutputs` entry resolved down to its source ({@link Session.#resolveCopyOutputs}). */
type ResolvedCopy = {
  readonly source: ValueBinding;
  readonly target: GPUBuffer;
  readonly size: number;
};

/**
 * One graph output resolved to "a readable buffer + size" ({@link Session.#resolveOutput}).
 * Both readback paths (copy into staging / copy at settlement) take it from the same single rule.
 */
type ResolvedOutput = {
  readonly shape: readonly number[];
  readonly count: number;
  readonly buffer: GPUBuffer;
  /** First byte when the binding is a sub-range (slot). Falls back to 0 for forms without one. */
  readonly offset: number;
  readonly size: number;
};

/**
 * One graph output read back at settlement ({@link Session.#planReadback} — the source is the slot
 * backing).
 */
type PlannedRead = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly count: number;
  readonly source: BatchReadSource;
};

export class Session {
  readonly #state: SessionState;
  #lastRun: ArenaStats | undefined;
  #lastRunFusions: FusionCounts | undefined;
  #lastRunParams: ParamsCacheStats | undefined;
  #lastRunPrepared: PreparedPlanStats | undefined;
  /**
   * The recipe list the last run queued (the source of `diagnostics().lastRunPipelines`). It keeps
   * the list itself, not a tally, and derives the table only when read, with {@link pipelineCensus}
   * (no tallying on the hot path; no independently held copy of a derivable table).
   */
  #lastRunRecipes: readonly StepRecipe[] | undefined;
  /**
   * The derivation phase (step list → recipe list — {@link RecipeBuilder}). The Session holds
   * exactly one, and passes its own {@link Session.#state} as the state (no run-lifetime container
   * on its surface).
   */
  readonly #recipeBuilder: RecipeBuilder;
  /**
   * The retained slot backings (plan key → instance — {@link ActiveBacking}). **Insertion order =
   * oldest first**; a hit key is moved to newest by delete → re-insert (LRU). The budget is
   * {@link Session.#evictBackingsFor}.
   */
  readonly #backings = new Map<string, ActiveBacking>();
  #backingBuilds = 0;
  /**
   * Slot buffers waiting for destruction. Switching, eviction and dispose only push here; the
   * actual `destroy()` happens **in one place after flush** ({@link Session.#destroyRetired}).
   */
  readonly #retired: GPUBuffer[] = [];
  /**
   * The live {@link GenerationContext}s (the **source** from which the diagnostic
   * `stateBacking.residentBytes` is derived).
   *
   * MUST: do not keep the resident byte count in an independently updated counter — if creation and
   * dispose added and subtracted on separate paths, a failure path that skips one of them (an
   * exception mid-allocation) would skew it permanently.
   */
  readonly #contexts = new Set<GenerationContext>();
  #contextCount = 0;
  /**
   * Cumulative count of re-baking bind groups that contain state (the diagnostic
   * `stateBacking.rebindCount`).
   *
   * MUST: counted per Session (not split per context). What is to be observed is precisely the
   * shape "alternating contexts re-bakes on every switch", which is visible only as a count over
   * the Session's run sequence.
   */
  #stateRebinds = 0;
  /**
   * The serialization chain of running / waiting runs and dispose. It passes only the settlement
   * (success or failure) on, so it never rejects itself.
   */
  #chain: Promise<void> = Promise.resolve();
  /**
   * Number of issued and unsettled {@link Session.run} calls (the observation point of the batch
   * self-deadlock detector).
   *
   * MUST: both increment and decrement happen in a **synchronous section** (+1 in the call section
   * of `run`, −1 in the body's finally). Call order = the order of `#chain`, so if this is positive
   * in the synchronous section of `enqueue`, "an unsettled run sits ahead of the enqueue" = exactly
   * the condition for the cycle ({@link Session.enqueue}).
   * MUST: only `run` is counted. The only thing that requests the errorScope section lock is the
   * run body; the dispose family (`Session.dispose` / `GenerationContext.dispose`) waits only on
   * flush and cannot take part in the cycle (mixing them in would make an overly strict guard that
   * also rejects sequences that do settle).
   */
  #pendingRuns = 0;
  /**
   * The batch that has an `enqueueRead` queued (later enqueues from the same Session are rejected
   * until it settles — a later one overwriting the output slot read at settlement would be a silent
   * wrong value).
   */
  #readBatch: EnqueueOptions["batch"] | undefined;
  /**
   * Number of weights lent to other Sessions (the borrow count of {@link Session.exportWeight} —
   * ADR 0096 decision 2 and its stage 2 addendum). While it is not 0, {@link Session.dispose} is rejected.
   */
  #lentWeights = 0;
  #disposal: Promise<void> | undefined;

  private constructor(state: SessionState) {
    this.#state = state;
    this.#recipeBuilder = new RecipeBuilder(state);
  }

  /**
   * The common construction path (one path whatever the source — the supply unit is the container's
   * part). Per part it steps through "upload blocks one at a time in an errorScope synchronous
   * section → explicit submit + fence" (ADR 0108 decision 9). Completeness of declarations and
   * supply is decided in the **merge phase** (`bindDeclarations`' "no supply in the binding table"
   * / `mergedGraph`'s "no weight supply"), so there is no completeness check after reading here.
   * The body is `buildSessionState` in {@link "./session-build.ts"} (returns the state only); this
   * is the thin facade on the side that owns the private constructor.
   *
   * MUST: the single entry point of construction is {@link PreparedModel.createContainerSession}
   * (an explicit async stage that includes the weight upload). It is callable from outside the
   * class only because PreparedModel is a separate class; it is not public surface — mod.ts exports
   * Session as a type only.
   * MUST: the residency plan (residency classes) is the one the prepare phase computed —
   * recomputing it on the construction side would bring back the shape in which the classes the
   * estimate saw and the classes actually uploaded can be results of different computations.
   *
   * NOTE: a `submitPolicy` misconfiguration (gpuTiming on × a `maxChunkSize` beyond the querySet
   * capacity) becomes a `SubmitPolicyError` here (construction of the SubmitScheduler), not
   * mid-run.
   * NOTE: since 89c1ccd moved the body out to `buildSessionState`, the completion of construction
   * is signalled one microtask later (accepted — before the added boundary the state is unreachable
   * from every caller, and the GPU wait position and fence order are unchanged).
   */
  static async build(
    gpu: GpuContext,
    graph: IrGraph,
    residency: ReadonlyMap<string, WeightResidency>,
    batches: AsyncIterable<WeightBatch>,
    options: SessionOptions,
  ): Promise<Session> {
    return new Session(await buildSessionState(gpu, graph, residency, batches, options));
  }

  /**
   * Binds the inputs, executes the graph and reads back **only the graph outputs**.
   * `bindings` explicitly sets symbolic dimensions and fails loudly when it disagrees with the
   * bindings derived from the input shapes.
   *
   * MUST: runs of the same Session are **serialized** in call order (even when issued concurrently
   * without await, they execute one at a time). errorScope is a per-device LIFO stack and a run
   * keeps its scope open across awaits, so overlapping runs would miss their own validation failure
   * as null while an unrelated run fails instead. A preceding run's failure does not propagate to
   * later runs.
   * NOTE: overlap with **another Session** on the same device is not prevented by this
   * serialization (each Session has its own chain). That is handled by the GpuContext's errorScope
   * section lock.
   *
   * A run given `generation` becomes **one step of a state-referencing graph** (prefill-chunk or
   * decode of ADR 0066 decision 4). A run without it does not change by a single byte of behaviour.
   *
   * ## input lifetime (snapshot at issue and borrowed data)
   *
   * The execution body runs behind `#serialize` (one microtask or later), so the pair "non-awaited
   * concurrent issue" and "rewrite right after issue" is easy to walk into. Therefore **metadata is
   * fixed in the synchronous section of issue** ({@link captureInputs}) — the member set of the
   * `inputs` Record, each input's shape, `bindings` and the two fields of `generation` (`context` /
   * `queryLength`) are copied, so even if the caller rewrites them without awaiting the returned
   * Promise, this run executes in **the form it had at issue**.
   *
   * MUST: **at most one unsettled run per context** (a second one fails loudly in the synchronous
   * section of issue). The second would build its uniform and dispatches from the logical length
   * the first advanced, while the position inputs the caller built at issue keep the old logical
   * length — a position shift with no exception and no warning. **Concurrent issue to different
   * contexts** and non-awaited concurrent issue of runs without generation still go through as
   * before.
   *
   * MUST NOT: `Tensor.data` (the TypedArray itself) is not copied but **borrowed**, so do not
   * rewrite it until the returned Promise settles. The GPU `writeBuffer` happens microtasks later,
   * so a rewrite is a silent wrong value with no exception and no warning. This is a deliberate
   * line so as not to pay a GiB-scale copy every run, and "borrowing" is itself the contract (the
   * same discipline that use reservations, binding reservations and the disposed check enforce for
   * resident inputs {@link ResidentTensor} is kept by the caller for host arrays. For resident
   * inputs too, **rewriting the contents** is the same borrowed-side contract —
   * {@link ResidentTensor.write}).
   */
  run(
    inputs: RunInputs,
    bindings: SymbolBindings = {},
    generation?: GenerationRun,
  ): Promise<RunOutputs> {
    // The disposed check happens at call time (checked inside the chain, runs issued before dispose
    // would fail along with it).
    if (this.#disposal !== undefined) {
      return Promise.reject(new ExecutionError("dispose 済みの Session では実行できない"));
    }
    // MUST: the context lease is acquired **before** pushing onto `#serialize` (the same reason as
    // `batch.enter` in `enqueue` — the body sits one microtask later, so acquiring it in the body
    // would let "`context.rewind()` right after a non-awaited run" judge "no run in progress", and
    // the captured P and the uniform would split).
    // A failed check is returned as a rejection of the returned Promise as before (not turned into
    // a synchronous throw).
    //
    // MUST: the third argument is also **copied in the synchronous section of issue** (the same
    // discipline as `inputs` / `bindings`). The body sits one microtask later and moreover reads
    // `context` / `queryLength` at several points across awaits, so carrying the reference around
    // would turn a rewrite right after issue into silent wrong values: "the source of the dispatch
    // count and the value in the uniform split" or "the lease stands on A but B's KV gets written".
    // The lease is also taken from this copy (the acquisition in the synchronous section and the
    // body's reads coming from one single value is the basis).
    let capturedGeneration: GenerationRun | undefined;
    let lease: GenerationContext[typeof RUNTIME_INTERNAL] | undefined;
    let captured: CapturedInputs;
    let capturedBindings: SymbolBindings;
    let used: ResidentTensor[];
    try {
      // MUST: copying the third argument, the value-range check and taking `[RUNTIME_INTERNAL]`
      // also happen inside this try (outside it, a getter's throw or a context from outside the
      // type would leak as a synchronous throw — the same discipline as {@link Session.#admit}).
      // The order is copy → value range → take → copy inputs (the lease is acquired after the
      // reservations).
      capturedGeneration =
        generation === undefined
          ? undefined
          : {
              context: generation.context,
              queryLength: generation.queryLength,
              commit: generation.commit,
            };
      assertGenerationCommit(capturedGeneration);
      lease = capturedGeneration?.context[RUNTIME_INTERNAL];
      // MUST: the input copy comes **before** the lease acquisition (a lease left behind with
      // nobody to return it after a failed copy would reject every later `rewind()` forever).
      captured = captureInputs(inputs);
      // Bindings are also fixed at issue (read in the body, a rewrite right after issue would turn
      // into an unrelated failure, "symbol bindings conflict").
      capturedBindings = { ...bindings };
      // MUST: check the entry once more after the copy. `captureInputs` and the spread run the
      // caller's getters synchronously, and a `dispose()` of the same Session can be issued inside
      // them. If that slips past the call-time check, the body runs after disposal completed,
      // reallocates the slot backing, and leaves a GPU buffer with no path to reclaim it (the same
      // discipline as "once more after allocation" in {@link Session.#adopt} — the final check sits
      // right before resource reservation and joining the serialization chain).
      if (this.#disposal !== undefined) {
        throw new ExecutionError("dispose 済みの Session では実行できない");
      }
      // MUST: device liveness is also checked in the synchronous section of acceptance
      // ({@link assertDeviceUsable}). The body's failure detection relies on errorScope and the
      // readback wait, so a run with 0 outputs and no readback can settle as a success even on a
      // lost device or one whose destruction was requested. It sits after the copy because a
      // caller's getter can issue `destroy()` synchronously (the same position as the dispose
      // re-check).
      assertDeviceUsable(this.#state.gpu, "run の受付");
      // MUST: use reservations on resident inputs are also taken in the **synchronous section of
      // issue** (the body's binding reservation alone lets a `dispose()` in the same tick through
      // while an accepted run exists — {@link retainUsedResidents}).
      const targets: [where: string, resident: ResidentTensor][] = [];
      for (const [name, resident] of captured.residentInputs) {
        targets.push([`入力 '${name}'`, resident]);
      }
      used = retainUsedResidents(targets, this.#state.gpu);
    } catch (cause) {
      return Promise.reject(cause);
    }
    try {
      lease?.acquireRun();
    } catch (cause) {
      // Return the use reservations already taken before failing (the body will not run = nobody
      // would return them).
      releaseUsedResidents(used);
      return Promise.reject(cause);
    }
    // MUST: counted in the synchronous section (the body sits one microtask later, so counting in
    // the body would hide "the unsettled run" from an `enqueue` queued in the same tick).
    this.#pendingRuns += 1;
    return this.#serialize(async () => {
      try {
        return await this.#runOnce(captured, capturedBindings, capturedGeneration);
      } finally {
        // MUST: always return them on success and failure alike (a missed return rejects every
        // later rewind forever / leaves that resident tensor undisposable). At this point every GPU
        // operation has been submitted, so a later `dispose()` is safe in WebGPU terms (the same
        // basis as `releaseBoundResidents`).
        releaseUsedResidents(used);
        lease?.releaseRun();
        this.#pendingRuns -= 1;
      }
    });
  }

  /**
   * Queues one execution **without a fence** (H-5 — the surface that removes the run boundaries of
   * a generation loop).
   *
   * Only three differences from `run`: ① it issues no `flush` / `onSubmittedWorkDone` / readback
   * (the wait is folded into the single {@link BatchScope.finish}) ② outputs are not returned to
   * the host but written out to resident tensors through `copyOutputs` ③ it rides on the errorScope
   * the batch opened (no per-run push / pop). The command list it queues is **identical** to a
   * backed run's.
   *
   * MUST: the only path is "prepared plan + slot backing". The first call does the derivation and
   * the backing construction here (unlike run it builds a backing **from the first call** — without
   * one the next call would also be non-backed and enqueue would not hold). **Not silently falling
   * back** to the arena path / readback path is the premise of this surface; for shapes that need a
   * fallback use `run`, and to merely receive the outputs on the host use
   * {@link Session.enqueueRead}.
   * MUST: it always submits eagerly at the end (`SubmitScheduler.submitPending` in
   * {@link "../gpu/submit.ts"}). This is the basis of "the next enqueue / `writeBuffer` does not
   * overtake preceding dispatches".
   * MUST: run / enqueue / dispose of the same Session are serialized in call order (the same
   * {@link Session.#chain} as `run`).
   * MUST: **a Session with an unsettled `run` ahead cannot enqueue** (fails loudly).
   * That sequence is a deterministic self-deadlock: the batch span holds the errorScope section
   * lock while waiting for the in-flight leases to be returned → a lease is returned only in the
   * enqueue body's finally → the enqueue body waits for the preceding run on `#chain` to settle →
   * the run body waits for that section lock; the four edges close.
   * `withScopeLock` is designed so that it cannot tell "a legitimate queue" from "re-entry" and
   * cannot host a re-entry detector, so undetected this **hangs forever with no exception and no
   * diagnostic** — the hang is turned into a typed exception.
   * NOTE: calling {@link BatchScope.finish} without awaiting the returned Promise loses nothing.
   * The batch's in-flight lease is taken in this **synchronous section**, and finish does not
   * proceed to the fence until every outstanding lease has been returned (the mechanism is
   * `BatchInternals.enter`).
   * MUST: in the non-awaited form **host-side failures are also attributed to `finish()`** (the
   * first one in the span — `BatchInternals.leave`). errorScope catches only GPU-side failures, so
   * without this "the span settles as a success with as many dispatches missing as enqueues that
   * failed in the body". The same failure also appears on the returned Promise, so **to discard it,
   * take it with `Promise.all`, not `void p.catch(...)`** (swallowed by `catch`, the failure of
   * that one is visible only from finish).
   *
   * Input lifetime is **the same as** `run` (the "input lifetime" section of {@link Session.run}) —
   * the member set of the `inputs` Record, shapes and the form of `bindings` / `copyOutputs` are
   * fixed in the synchronous section of issue, and `Tensor.data` stays borrowed until the returned
   * Promise settles (MUST NOT: do not rewrite it). This surface assumes non-awaited issue, so it is
   * easier to walk into than with run.
   */
  enqueue(inputs: RunInputs, options: EnqueueOptions): Promise<void> {
    return this.#admit(inputs, options, undefined);
  }

  /**
   * Queues the same command list as {@link Session.enqueue} and **reads the graph outputs back to
   * the host at the batch's closing fence** (ADR 0054, addendum of 2026-09-20 — batch readback of
   * graph outputs; the surface that keeps an ordinary run queued in the same span as GPU-resident
   * side inputs from paying one extra fence).
   *
   * The outputs are copied from the slot backing; the copies are queued at the span's settlement,
   * concatenated into one staging (the same staging and the same map as
   * {@link BatchScope.finishAndRead} = one fence). Because it reads at settlement rather than
   * copying the value at queue time, **later enqueues from the same Session are rejected until that
   * batch settles** (a later one overwriting the same slot would change the value read — a silent
   * wrong value).
   *
   * MUST: await `outputs` only after calling `finish` / `finishAndRead` (it does not resolve until
   * the span closes). A failed acceptance shows on both `admitted` and `outputs` with the same
   * reason.
   * NOTE: `lastRun` (the arena's record) stays undefined, as with enqueue (no arena is created).
   */
  enqueueRead(inputs: RunInputs, options: EnqueueOptions): EnqueueRead {
    const read = Promise.withResolvers<RunOutputs>();
    // Not an unhandled rejection until the caller takes the settlement (the same failure also shows
    // on admitted).
    void read.promise.catch(() => undefined);
    const admitted = this.#admit(inputs, options, read);
    // An enqueue that failed acceptance has queued nothing to read back — outputs fails for the
    // same reason.
    admitted.catch((cause) => read.reject(cause));
    return { admitted, outputs: read.promise };
  }

  /**
   * The entry check of `enqueue` / `enqueueRead` ({@link Session.#admit} calls it twice, before and
   * after the copy).
   *
   * MUST: the second call comes **after** `captureInputs` and the options copy. Both run the
   * caller's getters synchronously, and a `dispose()` / `enqueueRead` / `run` of the same Session
   * can be issued inside them. With only the call-time check, ① the body of a disposed Session runs
   * ② a later enqueue gets into a batch holding an `enqueueRead` and overwrites the slot read at
   * settlement (silent wrong value) ③ the cycle between a non-awaited run and the section lock is
   * not converted into a {@link BatchScopeError} and becomes a hang with no diagnostic.
   * MUST: checked **before** the lease is taken (failing after taking it would leave a lease with
   * nobody to return it, and then `finish()` really would wait forever).
   */
  #assertEnqueueAdmissible(batch: EnqueueOptions["batch"]): void {
    if (this.#disposal !== undefined) {
      throw new ExecutionError("dispose 済みの Session では実行できない");
    }
    // MUST: device liveness is checked too (the same reason as the acceptance of
    // {@link Session.run} — an enqueue with 0 outputs settles its returned Promise as a success
    // even on a device whose destruction was requested. The span's finish fails, but with the
    // enqueue's own settlement left as a success, one cannot read which enqueue hit the loss).
    assertDeviceUsable(this.#state.gpu, "enqueue の受付");
    if (this.#readBatch === batch) {
      throw new BatchScopeError(
        "enqueueRead を積んだ batch には、その決着まで同じ Session から enqueue できない" +
          "（決着時に読む出力 slot を後続の enqueue が上書きすると沈黙誤値になる）",
      );
    }
    if (this.#pendingRuns > 0) {
      throw new BatchScopeError(
        `未決着の run が ${this.#pendingRuns} 本ある Session には enqueue できない` +
          "（batch 区間は errorScope 区間ロックを握ったまま enqueue の決着を待ち、その " +
          "enqueue は先行 run を待ち、run はそのロックを待つ = 自己デッドロック）。" +
          "run を await してから batch を開くか、区間中は enqueue だけを使うこと",
      );
    }
  }

  /**
   * The common body of `enqueue` / `enqueueRead` (entry check, copy, lease, registration with the
   * span).
   */
  #admit(
    inputs: RunInputs,
    options: EnqueueOptions,
    read: PromiseWithResolvers<RunOutputs> | undefined,
  ): Promise<void> {
    // MUST: `options.batch` is read **here, exactly once**. The caller can plant a getter, so each
    // read may return a different span, and if the span checked, the span `enter`ed and the span
    // remembered in `#readBatch` disagree, ① the self-deadlock and disposed checks are bypassed ②
    // the `enqueueRead` guard "reject later enqueues until settlement" lands on a different span
    // and the slot to read gets overwritten (silent wrong value). From here on only this `scope` is
    // used.
    // MUST: taking `[RUNTIME_INTERNAL]` also happens inside this try (outside it, a getter's throw
    // would leak as a synchronous throw and break "acceptance failures are returned as a rejection
    // of the returned Promise").
    let scope: EnqueueOptions["batch"];
    let batch: EnqueueOptions["batch"][typeof RUNTIME_INTERNAL];
    try {
      scope = options.batch;
      this.#assertEnqueueAdmissible(scope);
      batch = scope[RUNTIME_INTERNAL];
    } catch (cause) {
      return Promise.reject(cause);
    }
    // MUST: the entry check and the lease acquisition are done **before** pushing onto
    // `#serialize`. The body sits one microtask later, so acquiring in the body would let "finish()
    // right after queuing a non-awaited enqueue" settle finish first, and the span would end as a
    // success with none of the queued work dispatched.
    // A failed check is returned as a rejection of the returned Promise as before (not turned into
    // a synchronous throw).
    let captured: CapturedInputs;
    let capturedOptions: EnqueueOptions;
    let used: ResidentTensor[];
    let lease: GenerationContext[typeof RUNTIME_INTERNAL] | undefined;
    try {
      // MUST: the input copy comes **before** the lease acquisition (a lease left behind with
      // nobody to return it after a failed copy makes `finish()` wait forever). `batch` is held as
      // the instance itself (it is not something to copy).
      captured = captureInputs(inputs);
      const generation = options.generation;
      capturedOptions = {
        batch: scope,
        ...(generation === undefined
          ? {}
          : {
              generation: {
                context: generation.context,
                queryLength: generation.queryLength,
                commit: generation.commit,
              },
            }),
        bindings: { ...options.bindings },
        ...(options.copyOutputs === undefined
          ? {}
          : // The destination Record also gets its member swaps fixed (the same reason as for
            // inputs — a swap getting through is the silent wrong value "written, but to another
            // resident tensor"). The instances stay borrowed.
            { copyOutputs: { ...options.copyOutputs } }),
      };
      // MUST: once more after the copy ({@link Session.#assertEnqueueAdmissible}) — the final check
      // sits right before resource reservation and joining the serialization chain.
      this.#assertEnqueueAdmissible(capturedOptions.batch);
      // MUST: use reservations, including the destinations, are taken in the **synchronous section
      // of issue** (the same reason as run — {@link retainUsedResidents}). The copy targets are
      // also instances an accepted enqueue uses, so they need the same lifetime protection as the
      // inputs.
      assertGenerationCommit(capturedOptions.generation);
      // A reference to an invalid context from outside the type is also resolved before the use
      // reservations on resident outputs.
      lease = capturedOptions.generation?.context[RUNTIME_INTERNAL];
      const targets: [where: string, resident: ResidentTensor][] = [];
      for (const [name, resident] of captured.residentInputs) {
        targets.push([`入力 '${name}'`, resident]);
      }
      for (const [name, target] of Object.entries(capturedOptions.copyOutputs ?? {})) {
        targets.push([`copyOutputs '${name}'`, target]);
      }
      used = retainUsedResidents(targets, this.#state.gpu);
    } catch (cause) {
      return Promise.reject(cause);
    }
    let acquired = false;
    try {
      lease?.acquireRun("batch");
      acquired = lease !== undefined;
      batch.enter(this.#state.gpu);
    } catch (cause) {
      if (acquired) lease?.releaseRun();
      // Return the use reservations already taken before failing (this enqueue does not enter the
      // span).
      releaseUsedResidents(used);
      return Promise.reject(cause);
    }
    // Separates the completion of the enqueue's encoding from the success of the span. Fixing the
    // length and returning the reservation both happen before finish settles.
    // DECIDED: docs/decisions/0066-generation-context-state-slots.md#addendum-2026-09-12--generationcontext-in-batched-execution
    let finalize: { complete(): void; fail(cause: unknown): void } | undefined;
    if (lease !== undefined) {
      batch.onSettled({
        complete: () => finalize?.complete(),
        fail: (cause) => finalize?.fail(cause),
        release: () => lease.releaseRun(),
      });
    }
    // The params, plans and backings this enqueue added (filled in by the body). A GPU-side failure
    // first shows not in the enqueue's return but in `finish()`'s errorScope, so the rollback is
    // wired to the span's failed settlement (the same discipline as run's catch — ADR 0095 decision
    // 2).
    // MUST: registration happens in the **synchronous section of acceptance** whether or not there
    // is a lease (a finalizer can only be registered at acceptance — registered in the body,
    // "non-awaited enqueue → finish()" would reject it).
    const additions = emptyAdditions();
    batch.onSettled({
      complete: () => undefined,
      fail: (_cause, origin) => this.#rollbackSettled(additions, origin),
      release: () => undefined,
    });
    if (read !== undefined) {
      // MUST: later enqueues from the same Session are rejected until settlement (blocks
      // overwriting the slot to read). It is lifted at the span's settlement (release, whatever the
      // outcome) — the same point as the lease finalizer.
      // MUST: lift it only when the field still holds this span. Clearing it unconditionally would
      // make correctness depend solely on the microtask count of "this finalizer runs before the
      // next span's enqueueRead", and a single await inserted before the finalizer of
      // `#resolveFinish` would clear the next span's field and remove the guard.
      this.#readBatch = scope;
      batch.onSettled({
        complete: () => undefined,
        fail: () => undefined,
        release: () => {
          if (this.#readBatch === scope) this.#readBatch = undefined;
        },
      });
    }
    return this.#serialize(async () => {
      // A host-side failure shows only on the returned Promise (errorScope catches only GPU-side
      // failures), so it is passed to the span as well and becomes what `finish()` attributes. The
      // wrapper tells "no failure" apart from "undefined was thrown".
      let failure: { readonly cause: unknown } | undefined;
      try {
        finalize = await this.#enqueueOnce(captured, capturedOptions, read, additions);
      } catch (cause) {
        failure = { cause };
        throw cause;
      } finally {
        // MUST: always return them on success and failure alike (a missed return hangs finish /
        // leaves that resident tensor undisposable).
        releaseUsedResidents(used);
        batch.leave(failure);
      }
    });
  }

  /**
   * Creates a {@link GenerationContext} that owns one generation's mutable state (ADR 0066
   * decisions 1 / 6).
   *
   * It fixes the slot capacities of `graph.states` (symbolic dimensions resolved through
   * `spec.bindings`) and `spec.chunkLength` and allocates them physically. The Session side
   * (immutable weights, plan cache, slot backing) **does not change at all** — by the ownership
   * split of decision 5, recipes and plan keys do not know the context.
   *
   * Declaring `spec.chunkBuckets` also allows those values as physical chunk row counts of the
   * prefill shape in addition to `chunkLength` (ADR 0066 addendum 10, buckets) — an additional
   * execution shape for running short prompts without padding; the caller picks one per chunk.
   *
   * MUST: state slots go neither into the Session's arena nor into a {@link ResidentTensor} (the
   * lifetime granularity differs — details in the module doc of generation-context.ts).
   * MUST: allocation happens inside an out-of-memory / validation errorScope section, and a failure
   * fails loudly (decision 6). A slot whose capacity exceeds `maxStorageBufferBindingSize` fails
   * before allocation (addendum 5).
   * NOTE: a context can outlive the Session (`dispose` touches only the injected flush and its own
   * buffers). Runs that use state go through the Session, though, so actual use stays within the
   * Session's lifetime.
   * NOTE: the call form of prefill / decode (the part ADR 0066 decision 6 left to the
   * implementation design) ended up as **the single third argument of `Session.run`**
   * ({@link GenerationRun}). Only `queryLength` and the physical chunk row count of the input tell
   * the two execution shapes apart; no methods are added.
   */
  createGenerationContext(spec: GenerationContextSpec): Promise<GenerationContext> {
    return this.#adopt(
      () => GenerationContext.create(this.#generationHost(), spec),
      "dispose 済みの Session では GenerationContext を作れない",
    );
  }

  /**
   * The face a context borrows from the Session ({@link GenerationContextHost} — only the fields it
   * needs).
   */
  #generationHost(): GenerationContextHost {
    return {
      gpu: this.#state.gpu,
      graph: this.#state.graph,
      flush: () => this.#state.scheduler.flush(),
      serialize: <T>(body: () => Promise<T>): Promise<T> => this.#serialize(body),
      forget: (context: GenerationContext): void => {
        this.#contexts.delete(context);
      },
      adopt: (allocate, refusal) => this.#adopt(allocate, refusal),
    };
  }

  /**
   * Allocation of a context and its registration in the live set (**the one registration path**
   * shared by {@link Session.createGenerationContext} and `GenerationContext.fork`).
   *
   * MUST: the disposed check happens **at call time** (the same discipline as run / enqueue), and
   * `allocate` is called **synchronously** right after it (no await in between). An async
   * function's body runs synchronously up to its first await, so the check and the `allocate()`
   * call happen inside the caller's synchronous section — fork relies on finishing the submit of
   * the copy within this section and takes no reservation on the source (the MUST of
   * `GenerationContextHost.adopt`).
   * MUST: registration in the live set comes **after the allocation settled** (counting a failed
   * context would make the diagnostics' resident byte count keep claiming memory that does not
   * exist).
   * MUST: the dispose check is enforced **once more after** the allocation. `#disposal` is set in
   * the synchronous section of `dispose()`, so a dispose issued across the allocation's await slips
   * past the call-time check and the context would be registered in a disposed Session's live set
   * (the diagnostic `stateBacking.residentBytes` would keep claiming a non-zero value, and those
   * GPU buffers would stay until the caller calls `context.dispose()`). An allocated context is
   * disposed by this method before failing.
   * NOTE: pushing the whole allocation onto {@link Session.#serialize} is **not done** — as a
   * member of `#chain`, create / fork would be drawn into the same kind of cycle as an unsettled
   * run in a batch span (BatchScope → enqueue → run → errorScope section lock). Staying off the
   * chain is the safe side of the wait graph.
   *
   * @param refusal the message when disposed (the caller's context — so create and fork read
   *   differently).
   */
  async #adopt(
    allocate: () => Promise<GenerationContext>,
    refusal: string,
  ): Promise<GenerationContext> {
    if (this.#disposal !== undefined) {
      throw new ExecutionError(refusal);
    }
    const context = await allocate();
    if (this.#disposal !== undefined) {
      // Return what was allocated first, then fail loudly (do not silently leak it outside the live
      // set).
      await context.dispose();
      throw new ExecutionError(refusal);
    }
    this.#contexts.add(context);
    this.#contextCount += 1;
    return context;
  }

  /**
   * **Lends** one resident weight (ADR 0096 decision 2 and its stage 2 addendum). The return value is an opaque reference,
   * passed to the borrowing Session's
   * `createContainerSession(gpu, { sharedWeights: { <borrower's name>: this } })`. Not a single
   * byte is copied (the bind groups of both bind the same GPU buffer).
   *
   * MUST: only weights **this Session actually allocated** can be lent. Re-exporting a shared
   * declaration (a borrowed weight) is rejected — allowing chains could make lifetime dependencies
   * cyclic across Sessions.
   * MUST: while a borrower is alive, {@link Session.dispose} of this Session fails loudly (borrow
   * count). This is a deliberate departure from the idempotent, non-throwing contract of `dispose`;
   * disposing would make the borrower's bind group read destroyed buffers (a silent wrong value
   * with no exception and no warning).
   */
  exportWeight(initializerName: string): SharedWeight {
    if (this.#disposal !== undefined) {
      throw new ExecutionError("dispose 済みの Session では重みを貸し出せない");
    }
    const { graph, residency, weightBuffers, residentWeights, gpu } = this.#state;
    if (!Object.hasOwn(graph.initializers, initializerName)) {
      throw new ExecutionError(`exportWeight: initializer '${initializerName}' がグラフに無い`);
    }
    const seat = residency.get(initializerName);
    if (seat === undefined) {
      throw new ExecutionError(`exportWeight: initializer '${initializerName}' の常駐分類が無い`);
    }
    if (seat.seat === "shared") {
      throw new ExecutionError(
        `exportWeight: initializer '${initializerName}' はこの Session も借りている` +
          "（借り物の再輸出はしない — 貸し手から直接借りること）",
      );
    }
    const buffer = weightBuffers.get(initializerName);
    if (buffer === undefined) {
      throw new ExecutionError(
        `exportWeight: initializer '${initializerName}' の重みバッファが台帳に無い`,
      );
    }
    // The lender's codec (the borrower's expected residency class is derived from it and the
    // borrower-side consumption). Shared declarations were rejected above, so storage always
    // exists.
    const lenderCodec = graph.initializers[initializerName].storage?.codec;
    if (lenderCodec === undefined) {
      throw new ExecutionError(
        `exportWeight: initializer '${initializerName}' の格納が無い（簿記の破れ）`,
      );
    }
    return new SharedWeight({
      initializer: initializerName,
      gpu,
      buffer,
      resident: residentWeights.get(initializerName),
      seat: seat.seat,
      rowAxis: seat.seat === "i8" || seat.seat === "i2" ? seat.rowAxis : undefined,
      codec: lenderCodec,
      shape: graph.values[initializerName].shape.map(Number),
      dtype: graph.values[initializerName].dtype,
      retain: (): void => {
        if (this.#disposal !== undefined) {
          throw new ExecutionError("dispose 済みの Session の重みは借りられない");
        }
        this.#lentWeights += 1;
      },
      release: (): void => {
        if (this.#lentWeights < 1) {
          throw new ExecutionError(
            `Session: 重みの借用計数の解放が過多（ランタイム内部の簿記の破れ）`,
          );
        }
        this.#lentWeights -= 1;
      },
    });
  }

  /**
   * Releases the weights. Ordinary runs are waited for. If a context is reserved by an unsettled
   * batch, or a batch on which this Session's `enqueueRead` waits for a readback is unsettled, the
   * call is rejected before acceptance closes (it can be retried after finish).
   */
  dispose(): Promise<void> {
    try {
      for (const context of this.#contexts) context[RUNTIME_INTERNAL].assertCanDispose();
      // MUST: the source of the settlement-time readback (`readAtFinish`) is the slot backing's raw
      // GPUBuffer, with no borrow count. Letting this through would queue finish's copy after the
      // disposal body destroyed the backing, and the whole span (including contexts of other
      // Sessions riding on it) would fail on a validation far from the real cause.
      if (this.#readBatch !== undefined) {
        throw new ExecutionError(
          "dispose: この Session は enqueueRead の読み戻しで batch の最終決着待ち（finish 後に破棄すること）",
        );
      }
    } catch (cause) {
      return Promise.reject(cause);
    }

    // MUST: not disposed while a borrower is alive (ADR 0096 decision 2 and its stage 2 addendum). The borrower's bind group
    // holds this Session's weight buffers. The same departure as `GenerationContext.dispose`; it
    // returns before acceptance closes (the assignment of `#disposal`).
    if (this.#lentWeights > 0 && this.#disposal === undefined) {
      return Promise.reject(
        new ExecutionError(
          `dispose: この Session の重みを ${this.#lentWeights} 本貸し出している` +
            "（借り手 Session の bind group が掴んでいる）。借り手を先に dispose すること",
        ),
      );
    }
    return this.#disposeOnce();
  }

  /**
   * The body of {@link Session.dispose} (the one and only disposal, after passing the borrow
   * check).
   */
  #disposeOnce(): Promise<void> {
    // MUST: the second and later calls return the same completion. Returning earlier would let the
    // caller treat it as "disposed" and go on to device.destroy(), breaking flush-before-destroy.
    // MUST: the slot backing's destruction also **rides on this one** (no additional owner of the
    // disposal path — ADR 0004). weights.destroy() waits for flush to complete, so the destroy
    // after it satisfies flush-before-destroy. Destruction always happens even on failure (mainly
    // device loss).
    this.#disposal ??= this.#serialize(async () => {
      try {
        await this.#state.weights.destroy();
      } finally {
        this.#retireAllBackings();
        this.#destroyRetired();
        // MUST: the borrow count of borrowed weights is returned **always** (regardless of whether
        // disposal succeeded — a missed return leaves the lending Session undisposable forever).
        for (const weight of this.#state.sharedWeights) weight[RUNTIME_INTERNAL].release();
      }
    });
    return this.#disposal;
  }

  diagnostics(): SessionDiagnostics {
    return {
      pipelineCount: this.#state.cache.usedCount,
      devicePipelineCount: this.#state.cache.deviceCount,
      submit: this.#state.scheduler.stats,
      weights: this.#state.weights.stats,
      storage: this.#state.storage,
      buildStats: this.#state.buildStats,
      geometryProfile: this.#state.geometryProfile.id,
      lastRun: this.#lastRun,
      lastRunTiming: this.#state.scheduler.timing,
      lastRunFusions: this.#lastRunFusions,
      lastRunParams: this.#lastRunParams,
      lastRunPrepared: this.#lastRunPrepared,
      lastRunPipelines:
        this.#lastRunRecipes === undefined ? undefined : pipelineCensus(this.#lastRunRecipes),
      planBacking: {
        // MUST: derived from the retained set every time (not kept in a counter added to and
        // subtracted from independently — the same discipline as #contexts).
        residentBytes: [...this.#backings.values()].reduce(
          (total, backing) => total + backing.bytes,
          0,
        ),
        inputBytes: [...this.#backings.values()].reduce(
          (total, backing) => total + backing.inputBytes,
          0,
        ),
        retainedCount: this.#backings.size,
        buildCount: this.#backingBuilds,
      },
      stateBacking: {
        // MUST: derived from the live set every time (the doc of {@link Session.#contexts}).
        residentBytes: [...this.#contexts].reduce(
          (total, context) => total + context[RUNTIME_INTERNAL].bytes,
          0,
        ),
        contextCount: this.#contextCount,
        rebindCount: this.#stateRebinds,
      },
    };
  }

  /**
   * Runs `body` after the previous execution settles. Returns this call's own result / exception.
   */
  #serialize<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#chain.then(body);
    this.#chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /**
   * Prepares a run / enqueue for issue (binding resolution → plan key → derivation → clearing the
   * diagnostics fields). **The one sequence both heads share**, kept as a single function so an
   * edit that fixes only one side cannot be made.
   *
   * MUST: keep it **synchronous** (not a single await). If the discipline of copying and reserving
   * in the synchronous section of issue (ADR 0054 / 0066) breaks here, a logical length and
   * bindings read one microtask late enter the body.
   * MUST: do not reorder the statements. Downstream poison decisions rely on failures occurring in
   * the order: reading the logical length (this read rejects poisoned, disposed and device-lost
   * contexts) → binding resolution → plan key → derivation.
   */
  #prepareInvocation(
    generation: GenerationRun | undefined,
    captured: CapturedInputs,
    bindings: SymbolBindings,
  ): PreparedInvocation {
    // Slot capacities after binding resolution. **Only these capacities** go into recipes and plan
    // keys; not a single byte of the context's identity does (ADR 0066 decision 5 —
    // {@link Session.#preparedKey}).
    const stateShapes =
      generation === undefined
        ? undefined
        : new Map(
            [...generation.context[RUNTIME_INTERNAL].slots].map(([name, slot]) => [
              name,
              slot.shape,
            ]),
          );
    // MUST: the logical length is read exactly once at the head of the execution (this read rejects
    // poisoned, disposed and device-lost contexts). The value written to the uniform, the dispatch
    // count, the capacity check and the advance all coming from one single value is the basis for
    // "the range the GPU scans" equalling "the workgroup count the host dispatched" (downstream
    // `writeLengths` / `advance` take this captured value and check it against the context's
    // current value).
    // MUST: read the **internal face** (not the user-facing `pastLength`). Dispose has two steps,
    // and the contract is that this accepted execution runs to completion even after the first step
    // (acceptance closed).
    const pastLength =
      generation === undefined ? 0 : generation.context[RUNTIME_INTERNAL].pastLength();
    const face =
      generation === undefined
        ? undefined
        : {
            context: generation.context,
            encoding: generationEncoding(generation.context, pastLength, generation.queryLength),
          };
    const { inputShapes, residentInputs } = captured;
    // MUST: binding resolution (= validation of the input shapes) **runs every time**, hit or miss.
    // Skipping it would let only executions that hit the cache slip past declared input-shape
    // mismatches.
    const resolved = bindSymbols(
      this.#state.graph,
      inputShapes,
      bindings,
      residentNames(residentInputs),
    );
    // State references without a context are rejected by the derivation. The bindings of a given
    // context are checked here.
    if (generation !== undefined) {
      assertGenerationBindings(generation.context[RUNTIME_INTERNAL].bindings, resolved);
    }
    const preparedKey = this.#preparedKey(resolved, residentInputs, stateShapes);
    const prepared = this.#takePrepared(preparedKey);
    // Planning (planGraph) and fusion matching (planFusions) are both pure functions that do not
    // touch the GPU, skipped entirely on a hit (the basis is {@link Session.#preparedKey}). Fusion
    // leaves nodes it could not match in the step list as they are, so this step only decides "does
    // it get faster" and has no part in correctness.
    const derived = prepared ?? this.#planSteps(resolved, stateShapes);
    const shapes = derived.shapes;
    // MUST: a hit execution also reports its fusion counts (the permanent contract of ADR 0040
    // decision 3 — if the observation point disappeared depending on the cache, a state in which
    // fusion stopped matching would hide behind hit runs).
    this.#lastRunFusions = derived.fusions;
    this.#lastRunPrepared = undefined;
    this.#lastRunRecipes = undefined;
    // MUST: the "last run" fields are reset **together** at the entry of derivation. Leaving the
    // fields that are assigned only on the success path (#lastRun / #lastRunParams) would make the
    // diagnostics right after a failed execution describe "the fusion counts of the failed
    // execution" and "the arena / params record of the previous successful run" as one run (the
    // diagnostics are the only observation point of fusion misses and params cache misses, so a
    // mix-up makes the reader reason from a different run).
    // NOTE: enqueue never creates an arena, so for enqueue `#lastRun` correctly **stays undefined
    // even on success** (keeping the previous run's value would make diagnostics read after an
    // enqueue talk about another execution).
    this.#lastRun = undefined;
    this.#lastRunParams = undefined;
    this.#recipeBuilder.resetParamsStats();
    return { stateShapes, pastLength, face, preparedKey, prepared, derived, shapes };
  }

  /**
   * The body of one run. **Inputs arrive as the copy taken at issue** ({@link CapturedInputs}) —
   * this runs one microtask or later, so looking up the caller's Record again would read rewrites
   * made after issue. `generation` arrives **as the copy taken at issue** for the same reason (the
   * two fields `Session.run` built).
   */
  async #runOnce(
    captured: CapturedInputs,
    bindings: SymbolBindings,
    generation: GenerationRun | undefined,
  ): Promise<RunOutputs> {
    const { gpu, graph, scheduler } = this.#state;

    // MUST: the context's provenance is checked **before entering encoding** (a failure at this
    // step does not reach state, so it does not poison). A mix-up with another Session's or a
    // disposed context would, if let through, become a silent wrong value "running with another
    // generation's KV bound", visible only in the values.
    if (generation !== undefined && !this.#contexts.has(generation.context)) {
      throw new ExecutionError(
        "run の GenerationContext がこの Session の生存集合に無い" +
          "（別の Session が作った context か、dispose 済み）",
      );
    }
    const {
      stateShapes,
      pastLength,
      face: generationFace,
      preparedKey,
      prepared,
      derived,
      shapes,
    } = this.#prepareInvocation(generation, captured, bindings);
    const encoding = generationFace?.encoding;
    /**
     * The submit counter at the moment the first state-writing dispatch was queued (undefined = not
     * queued yet). The poison decision on failure (ADR 0066 addendum 3) is a single comparison with
     * this value.
     */
    let stateWriteSubmits: number | undefined;
    const { residentInputs } = captured;

    const device = gpu.device;
    /**
     * Whether this run's fences are folded into **a single mapAsync** (H-1). This is the only
     * branch point; everything downstream (arena cleanup, how submits are issued, where readbacks
     * are queued) follows from this boolean.
     *
     * Two conditions keep the two-stage wait:
     * - **gpuTiming on**: both the timestamp collection and the measurement window hang off
     *   `flush`'s `onSubmittedWorkDone` (ADR 0021 — collection has to happen after completion, or
     *   mapAsync turns into the wait). Measurement is a diagnostic mode that captures a breakdown;
     *   there is no reason to save one fence there.
     * - **0 graph outputs**: no copy to queue = not a single mapAsync. Returning without a fence
     *   would let the run return with unfinished GPU work outstanding, so only that run falls back
     *   to the existing wait (the IR accepts an empty `graph.outputs` — format/ir.ts).
     */
    const singleFence = !gpu.gpuTimingEnabled && graph.outputs.length > 0;
    // MUST: cleanup on the single-fence path goes as far as "leave no unsubmitted encoding" and
    // opens no completion wait (mapAsync has already done the completion part of
    // flush-before-destroy, and a submitted command referencing a destroyed buffer is safe in
    // WebGPU terms — the implementation delays the actual release until completion). The default
    // (two-stage wait) keeps the fenced flush as before.
    const arena = new RunArena(
      device,
      singleFence
        ? () => {
            scheduler.submitPending();
            return Promise.resolve();
          }
        : () => scheduler.flush(),
    );
    // MUST: env holds **run-lifetime buffers only** (graph inputs and node outputs).
    // Session-resident weights / per-channel scales are folded into direct references in the
    // derivation phase, so there is no need to copy them per run — copying would leave
    // "Session-resident state mixed into the run's container", and recipe reuse would no longer
    // hold.
    const env = new Map<string, GPUBuffer>();
    // Resident inputs on which this run took a binding reservation ({@link Session.#bindInput}).
    // Always returned once encoding and submit are done (in exactly two places, the success path
    // and the failure path, one of which always runs).
    const boundResidents: ResidentTensor[] = [];
    // MUST: **a run issues GPU operations only inside its own lock section** (from encoding through
    // flush, pop and readback to arena destruction). errorScope is a per-device LIFO and an
    // operation's failure is attributed to the top of the stack at issue time, so "an operation
    // that opens no scope may run outside the lock" does not hold — the failures of operations that
    // open no scope are exactly the ones that land in someone else's scope. Within a Session #chain
    // already serializes, but runs of **another Session** on the same device can overlap (see the
    // GpuContext invariant).
    // MUST NOT: do not re-acquire the lock from inside this section (self-deadlock — the inner
    // layers use only scopes that complete within a synchronous section).
    const encode = async (): Promise<RunOutputs> =>
      await gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
        // The GPU time breakdown lives for **the last run** (ADR 0021). Discarding it after taking
        // the lock makes the table cover "only the run about to be queued" (the preceding run's
        // collection is already done).
        scheduler.resetTiming();
        let outputs: RunOutputs;
        // The activated slot backing (stays undefined on a miss run). Carried out of the encoding
        // section because readback eligibility checks "is that buffer a pinned slot".
        let backing: ActiveBacking | undefined;
        // Value name → binding (readback source) of a miss run. A backing run uses backing.outputs.
        let missValues: ReadonlyMap<string, ValueBinding> | undefined;
        // The params, plans and backings this run added (the failure-recovery discipline — read by
        // the catch below).
        const additions = emptyAdditions();
        // Whether the run's errorScope reported a failure or the pop itself failed (= the
        // possibility that an allocation was invalid through a non-fatal OOM cannot be excluded).
        // The catch below uses this to decide whether to remove params and plans
        // ({@link CacheAdditions}).
        let scopeFailed = false;
        // Readbacks queued on the run body's command list on the single-fence path (undefined = the
        // two-stage-wait path).
        let staged: readonly StagedOutput[] | undefined;
        try {
          // An invalid bindGroup / dispatch does not throw but fails together with its submit, a
          // silent failure that leaves pool leftovers in the outputs. Allocation of intermediate
          // buffers is watched for out-of-memory in the same section.
          pushFailureScopes(device);
          let popped = false;
          try {
            // Derivation phase → execution phase. Both inside the run's errorScope section with no
            // submit in between (the params writeBuffer is issued in the derivation phase, so it is
            // always issued before the submit of the dispatches that read it). A hit run skips the
            // whole derivation phase — no params writeBuffer and no pipeline creation, so the GPU
            // operations are only those of executing the recipes.
            let recipes: readonly StepRecipe[];
            let limits: GenerationLimits;
            if ("recipes" in derived) {
              recipes = derived.recipes;
              limits = derived.generation;
              // MUST: the input check (value-dependent — every run) comes before backing
              // construction. A run that fails here does not pay for building slots (~GiB scale for
              // a DiT).
              const data = graph.inputs.map((spec) =>
                this.#checkInput(spec.name, captured.values.get(spec.name), shapes),
              );
              // MUST: backings are built **only by hit runs**. This is the only guard that keeps
              // one-shot runs (workloads that run once) from paying slot memory, and it leaves the
              // behaviour and ArenaStats of miss runs completely unchanged.
              // MUST: generation runs also **ride on the same backing** (ADR 0066 decision 5). What
              // rides on it are only the dispatches that bind Session-owned buffers; dispatches
              // that bind context-owned buffers stay unbaked and `#generationGroups` below fills
              // them per context. Precisely because they are separated, the backing does not have
              // to be rebuilt when the context switches.
              const activated = this.#activateBacking(preparedKey, recipes, shapes, residentInputs);
              backing = activated.backing;
              if (activated.built) {
                additions.backings.push({ key: preparedKey, build: activated.backing.build });
              }
              graph.inputs.forEach((spec, index) => {
                // Resident inputs issue no writeBuffer (the instance itself is baked in).
                const values = data[index];
                if (values !== undefined) this.#writeInput(activated.backing, spec.name, values);
              });
            } else {
              this.#bindInputs(captured, shapes, arena, env, boundResidents);
              const plan = await this.#derivePrepared(preparedKey, derived, stateShapes, additions);
              recipes = plan.recipes;
              limits = plan.generation;
            }
            // MUST: filled at the same position as `#lastRunPrepared` (both fields describe the
            // same settlement point of the same run — enqueue does the same).
            this.#lastRunRecipes = recipes;
            this.#lastRunPrepared = {
              hit: prepared !== undefined,
              cachedPlans: this.#state.prepared.size,
            };
            if (generation !== undefined) {
              // MUST: the logical-length check and its transport happen **before a single dispatch
              // is queued** (failures up to here do not reach state, so they do not poison).
              // `queue.writeBuffer` lands on the queue timeline in issue order, so it does not
              // overtake preceding submits (ADR 0004 invariant ④ / ADR 0066 addendum 4).
              assertGenerationRun(
                limits,
                generation.context[RUNTIME_INTERNAL].allowedRows,
                pastLength,
                generation.queryLength,
              );
              generation.context[RUNTIME_INTERNAL].writeLengths(pastLength, generation.queryLength);
            }
            // MUST: the snapshot is taken **before queuing** the writing dispatch
            // (`SubmitScheduler.dispatch` auto-submits mid-run on the chunk limit and the time
            // budget — src/gpu/submit.ts). Taken afterwards, it would miss an auto-submit at the
            // moment of queuing and "submitted yet did not poison" = silent corruption. The error
            // in the other direction (counting a time-budget submit before queuing, over-poisoning)
            // is the safe side, recoverable with a new context.
            // MUST: called from **the same position** on miss runs and hit runs. Shifting it would
            // make the same failure of the same graph poison on a miss run and not on a hit run (or
            // the reverse).
            const noteStateWrite = (recipe: StepRecipe): void => {
              if (recipe.writesState && stateWriteSubmits === undefined) {
                stateWriteSubmits = scheduler.submitCount;
              }
            };
            if (backing === undefined) {
              // A miss run also runs on the same plan as the backing (ADR 0093 decision 3): regions
              // are allocated with run lifetime and every dispatch's bind group is built here.
              const plan = planRecipes(recipes, this.#state.transientLimits);
              const regions = plan.regions.map((bytes) => arena.allocRegion(bytes));
              const baked = bakeAllBindGroups(
                recipes,
                { device, plan, regions, inputs: env },
                encoding,
              );
              missValues = baked.values;
              arena.recordTransients({
                reuseCount: plan.sharedSlots,
                transientBytes: pinnedBytes(plan),
                peakTransientBytes: plan.peakLiveBytes,
              });
              const paced = executeBakedPlanPaced(
                recipes,
                baked.groups,
                scheduler,
                generationFace === undefined
                  ? undefined
                  : {
                      groups: baked.groups,
                      encoding: generationFace.encoding,
                      onStep: noteStateWrite,
                    },
              );
              // undefined for a run that does not reach the bound (adds neither a wait nor a yield
              // — ADR 0123).
              if (paced !== undefined) await paced;
            } else {
              // The slot path. The command list it queues is identical to the arena path's; only
              // the bound buffers stay fixed across runs (the basis for tolerating leftovers of the
              // previous run is full-write — ADR 0014).
              // The bind groups were baked at construction, so this only queues dispatches
              // (a generation run checks only the context-side bundle per run — decision 5).
              const paced = executeBakedPlanPaced(
                recipes,
                backing.groups,
                scheduler,
                generationFace === undefined
                  ? undefined
                  : {
                      groups: this.#generationGroups(generationFace, backing, recipes),
                      encoding: generationFace.encoding,
                      onStep: noteStateWrite,
                    },
              );
              if (paced !== undefined) await paced;
            }

            if (singleFence) {
              // MUST: the readback copies are queued on **the same command list as the dispatches**
              // (scheduler) (FIFO). Correctness rests on the queuing order — at this point the
              // scheduler still holds unsubmitted dispatches, so submitting a copy from another
              // encoder first would read outputs not yet written (the same as the MUST of
              // SubmitScheduler.copyBuffer). The submit count is not the basis — submits on the
              // same queue execute in issue order, so even split into a separate submit after all
              // dispatches are out, the resolution of mapAsync implies completion of the preceding
              // dispatches. Queuing on the same list makes one submit and one fence enough, and the
              // flush completion wait and the re-flush disappear (ADR 0054 decision 6 — a side
              // benefit).
              staged = this.#stageOutputs(
                this.#runValues(backing, missValues),
                shapes,
                arena,
                (source, offset, size, staging) =>
                  scheduler.copyBuffer(source, offset, staging, 0, size),
              );
              // Flush everything out without a fence. The wait is folded into the single mapAsync
              // below.
              scheduler.submitPending();
            } else {
              await scheduler.flush();
            }
            // The errorScope net does not change with the path (on the single-fence path the
            // readback copy and the staging allocation also ride on the run body's two scopes — the
            // same `out-of-memory` + `validation` pair that `#readOutputs` opens on the
            // two-stage-wait path).
            const pending = popFailureScopes(
              device,
              singleFence ? "run のエンコードと readback" : "run のエンコード",
            );
            popped = true;
            const failure = await pending.catch((cause: unknown) => {
              scopeFailed = true;
              throw cause;
            });
            if (failure !== undefined) {
              scopeFailed = true;
              throw failure;
            }
          } catch (cause) {
            // MUST: the leftover pending dispatches of a failed run are discarded, not submitted (a
            // throw mid-encoding leaves the earlier nodes' dispatches behind). There is no reason
            // to execute them, and issuing them through the cleanup arena.destroy() → flush would
            // submit while referencing buffers about to be destroyed. discard is synchronous, so no
            // window for a submit opens before they are discarded.
            scheduler.discard();
            // A run that failed before the pop (mainly host-side failures) still looks at the
            // scope's contents — if just discarding fell to the removing side, every recurring
            // identical host-side failure (a miss run's assertGenerationRun and the like) would
            // allocate params anew and let them pile up until dispose.
            if (!popped) scopeFailed = await discardFailureScopes(device);
            throw cause;
          }

          // A backed run builds no env (the readback sources are the mapping fixed at baking).
          outputs =
            staged === undefined
              ? await this.#readOutputs(this.#runValues(backing, missValues), shapes, arena)
              : await this.#finishStagedRead(staged);
          this.#lastRun = arena.stats;
          this.#lastRunParams = this.#recipeBuilder.paramsStats;
        } catch (cause) {
          // MUST: a cleanup failure does not overwrite the body's exception. The cause lies in the
          // body, and replacing it with destroy's rejection (mainly device loss) would erase the
          // starting point of the investigation.
          await arena.destroy().catch(() => undefined);
          // MUST: retire only the backings **this run newly built**. A transient failure
          // (out-of-memory at construction and the like) recovers by rebuilding on the next hit of
          // the same signature, and a broken backing does not stay resident for later runs. A
          // failed run on an existing backing does not retire it — that would force ~GiB-scale
          // rebuild thrashing on every unrelated failure (the recovery means there are signature
          // switches and LRU eviction).
          this.#retireBuiltBackings(additions);
          // MUST: the params and plans this run added are removed only when a GPU-side failure
          // cannot be excluded — the errorScope reported a failure (including the contents of a
          // scope discarded after failing before the pop), the pop itself failed, or the device was
          // lost (popErrorScope after loss resolves with null, so the check result is no evidence).
          // On the single-fence path the staging allocation and the copy ride on the run's scope,
          // so their failure falls on the removing side here. On a host-side failure (a throw in
          // derivation or the pre-run checks), and on a readback failure after passing the run's
          // scope without failure (mapAsync, or a failure that `#readOutputs` on the two-stage-wait
          // path reports in its own scope), the created params and plans are valid — removing them
          // would allocate params anew on every identical failure, forget them, and let them pile
          // up until dispose.
          if (scopeFailed || gpu.destroyRequested || gpu.lost !== undefined) {
            this.#forgetDerived(additions);
          }
          // MUST: slots waiting for destruction are returned **after** arena.destroy (= flushed /
          // discarded).
          this.#destroyRetired();
          throw cause;
        } finally {
          // MUST: binding reservations are always returned here on both the success and the failure
          // path (a missed return leaves that resident tensor undisposable for the whole Session
          // lifetime). At this point every GPU operation, including the readback, has been
          // submitted, so a later `dispose()` is "destroying a buffer referenced by submitted
          // commands" = safe in WebGPU terms (the implementation delays the actual release until
          // completion).
          releaseBoundResidents(boundResidents);
        }
        // When the body succeeded, a cleanup failure is propagated as is (never return with the
        // flush unfinished).
        await arena.destroy();
        // The only cleanup point that returns old backings freed by switching / eviction (after
        // flush — flush-before-destroy of ADR 0004). dispose does the same, so leftovers of runs
        // that failed without passing here are not lost either.
        this.#destroyRetired();
        return outputs;
      });

    try {
      const outputs = await encode();
      // MUST: the logical length advances only when the run **returned without an exception**
      // (ADR 0066 decision 6 — the "success" in "the logical length advances on a run's success"
      // means this). A run that failed in readback or cleanup has advanced only the physical ring, so it
      // does not advance and falls to the poison below. Deferred uses the same "success" boundary,
      // but instead of advancing it **hands the right to advance over to context.commit** (the
      // number of accepted rows is decided only by reading this run's outputs, so the runtime
      // cannot decide it). A failed run creates no pending commit either.
      if (generation !== undefined) {
        const internals = generation.context[RUNTIME_INTERNAL];
        // A borrower's run does not write a single row of state (0 appends — ADR 0096 decision 1),
        // so there is nothing to advance. Only the lender's own runs move the lender's logical
        // length.
        if (internals.borrowing) {
          // no-op (the right to advance belongs to the lender)
        } else if (generation.commit === "deferred") {
          internals.defer(pastLength, generation.queryLength);
        } else {
          internals.advance(pastLength, generation.queryLength);
        }
      }
      return outputs;
    } catch (cause) {
      this.#poisonOnStateWrite(generation, stateWriteSubmits, cause);
      throw cause;
    }
  }

  /**
   * Cleanup of a failed generation run ("failure atomicity" of ADR 0066 addendum 3).
   *
   * If a state-changing dispatch was **submitted**, the physical ring may have been overwritten
   * while the logical length did not advance, so the context is poisoned (a design without rollback
   * / staging — recovery is a new context + rebuilding the state on the host). If nothing was
   * submitted, `scheduler.discard()` has dropped the pending work and the ring is intact, so it is
   * not poisoned.
   *
   * MUST: paths that cannot be decided fall to poison (over-poisoning is recoverable with a new
   * context, while under-poisoning is the silent corruption "generation continues on broken KV").
   * That is why the snapshot is taken **before** the writing dispatch.
   */
  #poisonOnStateWrite(
    generation: GenerationRun | undefined,
    snapshot: number | undefined,
    cause: unknown,
  ): void {
    if (generation === undefined || snapshot === undefined) return;
    if (this.#state.scheduler.submitCount === snapshot) return;
    generation.context[RUNTIME_INTERNAL].poison(
      `state 変更 dispatch を submit した run が失敗した（${
        cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)
      }）`,
    );
  }

  /**
   * Queues the arena-path input bindings for all graph inputs at once (the caller of
   * {@link Session.#bindInput}). The one function shared by non-backed miss runs and generation
   * runs.
   */
  #bindInputs(
    captured: CapturedInputs,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    env: Map<string, GPUBuffer>,
    bound: ResidentTensor[],
  ): void {
    for (const spec of this.#state.graph.inputs) {
      const value = captured.values.get(spec.name);
      env.set(spec.name, this.#bindInput(spec.name, value, shapes, arena, bound));
    }
  }

  /**
   * One enqueue (the body of {@link Session.enqueue}).
   *
   * The structure is the content of a backed run with every part "that involves a fence and
   * readback" removed:
   *
   * - **No errorScope is opened**. The batch keeps the two scopes `out-of-memory` + `validation`
   *   open for the span, and pushing here would deepen the LIFO nesting by the number of enqueues
   *   (and since the pop spans awaits it would need the section lock = held by the batch, a
   *   self-deadlock).
   * - **The `GpuContext`'s `withScopeLock` is not taken**. The batch holds the section lock (taking
   *   it would self-deadlock — the GpuContext invariant).
   * - **No {@link RunArena} is created**. The only path is the slot backing, and not a single byte
   *   of run-lifetime allocation is issued (not even readback staging).
   *
   * Like run, inputs arrive as **the copy taken at issue** ({@link CapturedInputs}) — this too runs
   * one microtask or later, so looking up the caller's Record again would read rewrites made after
   * issue.
   *
   * MUST: a backing is built even on the first call (no prepared plan). run "builds a backing only
   * on hit runs" — so that one-shot runs do not pay slot memory — but enqueue is a surface that
   * assumes repetition from the start, so bringing that guard in would drop the first call to
   * non-backed (= the arena path with a fence). Not silently falling back is this surface's
   * contract, so the first call pays here.
   */
  async #enqueueOnce(
    captured: CapturedInputs,
    options: EnqueueOptions,
    read: PromiseWithResolvers<RunOutputs> | undefined,
    additions: CacheAdditions,
  ): Promise<{ complete(): void; fail(cause: unknown): void } | undefined> {
    const { graph, scheduler } = this.#state;
    // The entry check and the lease acquisition are already done in the synchronous section of
    // {@link Session.enqueue} (doing them in the body would not close the race with finish). Here
    // it only registers as a party to the settlement.
    // MUST: register as the party to "flush out the unsubmitted work" and "close the measurement
    // window" at the span's settlement.
    options.batch[RUNTIME_INTERNAL].join(scheduler);

    const generation = options.generation;
    if (generation !== undefined && !this.#contexts.has(generation.context)) {
      throw new ExecutionError("enqueue の GenerationContext がこの Session の生存集合に無い");
    }
    const { stateShapes, pastLength, face, preparedKey, prepared, derived, shapes } =
      this.#prepareInvocation(generation, captured, options.bindings ?? {});
    let stateWriteSubmits: number | undefined;
    const { residentInputs } = captured;

    let reads: readonly PlannedRead[] | undefined;
    try {
      let recipes: readonly StepRecipe[];
      let limits: GenerationLimits;
      if ("recipes" in derived) {
        recipes = derived.recipes;
        limits = derived.generation;
      } else {
        const plan = await this.#derivePrepared(preparedKey, derived, stateShapes, additions);
        recipes = plan.recipes;
        limits = plan.generation;
      }
      // MUST: the checks of inputs and destinations (value-dependent) come before backing
      // construction. An enqueue that fails here does not pay for building slots.
      const data = graph.inputs.map((spec) =>
        this.#checkInput(spec.name, captured.values.get(spec.name), shapes),
      );
      const copies = this.#planCopyOutputs(options.copyOutputs, shapes);
      const activated = this.#activateBacking(preparedKey, recipes, shapes, residentInputs);
      if (activated.built) {
        additions.backings.push({ key: preparedKey, build: activated.backing.build });
      }
      // MUST: resolving the copy sources (a check that depends on the instances) is done before a
      // single dispatch is queued.
      const writes = this.#resolveCopyOutputs(copies, activated.backing);
      // MUST: resolving the outputs to read back (a check that depends on the instances) is also
      // done before a single dispatch is queued.
      reads =
        read === undefined
          ? undefined
          : this.#planReadback(activated.backing, shapes, residentInputs);
      // MUST: the total readback limit is also checked before the dispatches (registration happens
      // after the submit, so failing there for the first time would be a failure after state was
      // written, and it would poison the context).
      if (reads !== undefined) {
        options.batch[RUNTIME_INTERNAL].assertReadAtFinish(reads.map((item) => item.source));
      }
      graph.inputs.forEach((spec, index) => {
        const values = data[index];
        if (values !== undefined) this.#writeInput(activated.backing, spec.name, values);
      });
      this.#lastRunRecipes = recipes;
      this.#lastRunPrepared = {
        hit: prepared !== undefined,
        cachedPlans: this.#state.prepared.size,
      };
      if (generation !== undefined) {
        assertGenerationRun(
          limits,
          generation.context[RUNTIME_INTERNAL].allowedRows,
          pastLength,
          generation.queryLength,
        );
        generation.context[RUNTIME_INTERNAL].writeLengths(pastLength, generation.queryLength);
      }
      executeBakedPlan(
        recipes,
        activated.backing.groups,
        scheduler,
        face === undefined
          ? undefined
          : {
              groups: this.#generationGroups(face, activated.backing, recipes),
              encoding: face.encoding,
              onStep: (recipe) => {
                if (recipe.writesState && stateWriteSubmits === undefined) {
                  stateWriteSubmits = scheduler.submitCount;
                }
              },
            },
      );
      // MUST: the copies are queued **after** the dispatch list (the FIFO of the same command list
      // is the basis of "reading a slot that has finished being written"). The shape in which the
      // destination doubles as a resident input of the same enqueue (a loop's state update) is also
      // correct, since every dispatch that reads it is queued first.
      for (const write of writes) {
        scheduler.copyBuffer(
          write.source.buffer,
          write.source.offset ?? 0,
          write.target,
          0,
          write.size,
        );
      }
      // MUST: always submit at the end. A later `queue.writeBuffer` (the next enqueue's input
      // write) lands on the queue timeline in issue order and does not overtake preceding
      // dispatches — only **unsubmitted** encoding is overtaken (ADR 0004 invariant ④). A path that
      // leaves pending work would be a silent wrong value in which the next input is read by the
      // previous enqueue.
      scheduler.submitPending();
    } catch (cause) {
      // MUST: the leftover pending work of a failed enqueue is discarded, not submitted (the same
      // discipline as run).
      scheduler.discard();
      this.#poisonOnStateWrite(generation, stateWriteSubmits, cause);
      // A backing this enqueue newly built is retired on the spot even on a host-side failure (the
      // same discipline as run — already discarded, so destruction may proceed). Params and plans
      // are not removed here (valid on a host-side failure — {@link CacheAdditions}). If a GPU-side
      // failure also shows in the same span, the span's failed settlement
      // ({@link Session.#rollbackSettled}) removes them from the same record.
      this.#retireBuiltBackings(additions);
      this.#destroyRetired();
      throw cause;
    }
    // MUST: buffers waiting for destruction may be returned only **after** submit / discard (no
    // unsubmitted encoding may reference a destroyed buffer). A reference from submitted commands
    // is safe in WebGPU terms; the implementation delays the actual release until completion.
    try {
      this.#destroyRetired();
      this.#lastRunParams = this.#recipeBuilder.paramsStats;
      if (read !== undefined && reads !== undefined) {
        // Registers as a party to be put on the settlement staging (inside the in-flight lease —
        // finish waits for every lease to be returned, so nothing is lost). Resolution happens
        // after the span's outcome is decided (on the BatchScope side).
        const planned = reads;
        options.batch[RUNTIME_INTERNAL].readAtFinish(planned.map((item) => item.source)).then(
          (buffers) => {
            try {
              read.resolve(this.#collectRead(planned, buffers));
            } catch (cause) {
              read.reject(cause);
            }
          },
          (cause) => read.reject(cause),
        );
      }
    } catch (cause) {
      this.#poisonOnStateWrite(generation, stateWriteSubmits, cause);
      throw cause;
    }
    if (generation === undefined) return undefined;
    return {
      complete: (): void => {
        const internals = generation.context[RUNTIME_INTERNAL];
        if (!internals.borrowing) {
          if (generation.commit === "deferred") internals.defer(pastLength, generation.queryLength);
          else internals.advance(pastLength, generation.queryLength);
        }
      },
      fail: (cause): void => this.#poisonOnStateWrite(generation, stateWriteSubmits, cause),
    };
  }

  /**
   * Checks `copyOutputs` (done **before backing construction**). It checks that each is a graph
   * output, that the resident destination is alive, and that its size matches the declared shape
   * exactly.
   */
  #planCopyOutputs(
    copyOutputs: Readonly<Record<string, ResidentTensor>> | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): readonly PlannedCopy[] {
    if (copyOutputs === undefined) return [];
    return Object.entries(copyOutputs).map(([name, target]) => {
      if (!this.#state.outputNames.has(name)) {
        throw new ExecutionError(
          `copyOutputs '${name}' はグラフ出力ではない（[${[...this.#state.outputNames].join(
            ", ",
          )}]）`,
        );
      }
      assertResidentUsable(target, this.#state.gpu, `copyOutputs '${name}'`);
      const shape = resolvedShape(shapes, name);
      const size = outputByteSize(numel(shape));
      if (target.byteLength !== size) {
        throw new ExecutionError(
          `copyOutputs '${name}': 常駐テンソル '${target.label}' の ${target.byteLength} バイトが shape [${shape.join(
            ",",
          )}] の ${size} バイトと合わない`,
        );
      }
      return { name, target, size };
    });
  }

  /**
   * Resolves the sources of `copyOutputs` (the graph outputs' buffers). It is a check that depends
   * on the instances, so it comes after backing activation, but it is done **before a single
   * dispatch is queued**.
   *
   * MUST: the shape in which source and destination are the same buffer fails here. In a degenerate
   * graph whose output is an alias of an input (`reshape` / identity `expand`), the output's buffer
   * is the resident input itself and a self-copy gets queued. WebGPU validation catches it, but
   * **it is caught at `batch.finish()`**, so ① the enqueue at fault cannot be identified ②
   * unrelated enqueues of the same span fail along with it (failures are attributed per batch — an
   * accepted consequence of BatchScope's design). A one-line check here returns the real cause, by
   * name, to the enqueue's call site.
   */
  #resolveCopyOutputs(
    copies: readonly PlannedCopy[],
    backing: ActiveBacking,
  ): readonly ResolvedCopy[] {
    return copies.map((copy) => {
      const source = backing.outputs.get(copy.name);
      // The IR allows initializer names in graph.outputs, and such a value is not in the baked
      // value mapping (on the run side the weight fallback of `#readOutputs` handles it). There is
      // no point in making it a copy source (it would only copy the same constant on the GPU every
      // execution), so it fails.
      if (source === undefined) {
        throw new ExecutionError(
          `copyOutputs '${copy.name}': ノード出力ではない（initializer をそのままグラフ出力に` +
            "した値は写せない — その値は実行に依らず不変)",
        );
      }
      const target = copy.target[RUNTIME_INTERNAL].buffer;
      if (source.buffer === target) {
        throw new ExecutionError(
          `copyOutputs '${copy.name}': 写し元と写し先が同じバッファ（常駐テンソル ` +
            `'${copy.target.label}'）— グラフ出力がその常駐入力の別名（reshape / 恒等 expand）に` +
            "なっているため、写しは自己コピーになる（計算内容がゼロの構成）",
        );
      }
      return { source, target, size: copy.size };
    });
  }

  /**
   * The first half of the derivation phase (plan → fusion matching). Closed over pure functions
   * that do not touch the GPU.
   *
   * The device limits are passed **as values** (so that fusion rules can statically decide "plans
   * determined by the machine's capability", such as the number of row blocks, without holding a
   * GPUDevice — {@link FusionLimits} in src/runtime/fusion.ts).
   */
  #planSteps(
    bindings: SymbolBindings,
    stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  ): PlannedSteps {
    const plan = planGraph(this.#state.graph, bindings, stateShapes);
    const { maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension } =
      this.#state.gpu.limits;
    // The instances are not passed to the fusion pure function; only at plan derivation are they
    // projected to storage descriptions.
    // The pairing of packed activations (ADR 0105) also looks at the storage format of the
    // consuming linear, so the projection is made when either option is on.
    const weightLayouts =
      this.#state.fuseLinearStaticQuantize || this.#state.packedStaticQuantize
        ? new Map(
            [...this.#state.residentWeights].map(([name, weight]): [string, FusionWeightLayout] => [
              name,
              weight.storage === "i4"
                ? { storage: weight.storage, groupSize: weight.groupSize }
                : { storage: weight.storage },
            ]),
          )
        : undefined;
    const fusion = planFusions(plan.nodes, {
      useCounts: this.#state.useCounts,
      fuseRmsNormAdd: this.#state.fuseRmsNormAdd,
      fuseLinearStaticQuantize: this.#state.fuseLinearStaticQuantize,
      packedStaticQuantize: this.#state.packedStaticQuantize,
      linearGemvReduce: this.#state.linearGemvReduce,
      linearCompute: this.#state.linearCompute,
      ...(weightLayouts === undefined ? {} : { weightLayouts }),
      rmsNormReduce: this.#state.rmsNormReduce,
      outputNames: this.#state.outputNames,
      limits: { maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension },
      ...(this.#state.rowBlockSplit === undefined
        ? {}
        : { rowBlockSplit: this.#state.rowBlockSplit }),
      ...(this.#state.attentionRowBlockMaxBytes === undefined
        ? {}
        : { attentionRowBlockMaxBytes: this.#state.attentionRowBlockMaxBytes }),
    });
    return { shapes: plan.shapes, fusions: fusion.counts, steps: fusion.steps };
  }

  /**
   * The prepared-plan key — the concatenation of the resolved bindings' values in the **declaration
   * order** of `graph.symbols`. A graph without symbols has the single key `""`.
   *
   * MUST: this is the whole basis of "the derivation phase may be skipped entirely". The graph is
   * fixed at Session construction, and planning, fusion matching and recipe derivation are pure
   * functions of graph and bindings only, so **if the key is an exact match of the resolved
   * bindings, the derivation result is structurally identical**, and planGraph's contract checks
   * (matching against declared shapes, dtype resolution, unbound symbols) reach the same verdict.
   * Hence a hit run skipping the checks never loosens fail-loudly (it only omits re-running the
   * same checks on the same input). Validation of the input shapes themselves is done by
   * {@link bindSymbols} on every run.
   * MUST: bindSymbols guarantees that every symbol is bound (it throws otherwise), so missing
   * entries need no attention here.
   * MUST: **the ids of resident inputs are included too**. The derivation result (recipes) itself
   * is a function of bindings only, but this key is also the slot backing's signature, and the
   * backing folds the resident inputs' buffers into bind groups at baking — without reflecting a
   * swap in the key, it keeps running on **bind groups that still bind the previous resident
   * tensor** (not a single exception, only the values are stale). The separator `|` never appears
   * in the bindings part (digits and commas only), so the key of a run without resident inputs
   * stays as it was.
   * MUST: **the {@link GenerationContext}'s identity is not included** (ADR 0066 decision 5).
   * Including it would wipe out planning, fusion matching, recipe derivation and slot backing
   * construction on every context switch (re-derivation for every sequence on the decode hot path).
   * It is treated the opposite way from resident inputs because context-owned buffers (state slots
   * and the logical-length uniform) are not baked into Session-owned bind groups — the price of the
   * separation is re-binding only "the bind groups that contain state" on the context side
   * ({@link Session.#generationGroups}).
   * tests/gpu_generation_context_test.ts pins this invariant as a check in the form "creating two
   * contexts does not increase the prepared plans and the params cache".
   * MUST: what is included is the **resolved slot capacity** (ADR 0066 decision 3, "the key is the
   * capacity", and addendum 7). Creating contexts with C=512 and C=131072 on the same Session
   * correctly gives state-referencing plans a separate key per capacity (recipes bake the capacity
   * into params and the allocation size of S). The separator `#` does not appear in the two
   * upstream parts (digits, commas, `|`, `t`/`r<id>`), so **the key of a run without generation
   * does not move by a single character**.
   */
  #preparedKey(
    bindings: SymbolBindings,
    residents: ReadonlyMap<string, ResidentTensor>,
    stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  ): string {
    const graph = this.#state.graph;
    // States-only symbols (the KV capacity `C` and the like) are **not put** in the dims part —
    // bindSymbols neither requires nor accepts them (ADR 0066 addendum 7), so they have no value,
    // and the capacities part below fully carries the capacity information.
    const statesOnly = statesOnlySymbols(graph);
    const dims = graph.symbols
      .filter((sym) => !statesOnly.has(sym))
      .map((sym) => bindings[sym])
      .join(",");
    const bound =
      residents.size === 0
        ? ""
        : `|${graph.inputs
            .map((spec) => {
              const resident = residents.get(spec.name);
              return resident === undefined ? "t" : `r${resident[RUNTIME_INTERNAL].id}`;
            })
            .join(",")}`;
    if (stateShapes === undefined) return `${dims}${bound}`;
    // MUST: the order is the declaration order of `graph.states` (not dependent on Map iteration
    // order — if two contexts with the same slot set in a different order got different keys,
    // re-derivation would happen on every switch).
    const capacities = Object.keys(graph.states)
      .map((name) => (stateShapes.get(name) ?? []).join("x"))
      .join(";");
    return `${dims}${bound}#${capacities}`;
  }

  /**
   * Looks up the cache and, on a hit, moves it to most recently used (Map insertion order is the
   * LRU order itself).
   */
  #takePrepared(key: string): PreparedPlan | undefined {
    const cached = this.#state.prepared.get(key);
    if (cached === undefined) return undefined;
    this.#state.prepared.delete(key);
    this.#state.prepared.set(key, cached);
    return cached;
  }

  /**
   * Stores a prepared plan and drops the oldest when the bound is exceeded.
   *
   * NOTE: eviction discards **host-side objects only** and destroys no GPU resources. The instances
   * recipes reference directly (params buffers in the weights arena held by paramsCache = Session
   * resident; pipelines / layouts in the GpuContext-owned cache = device resident) all outlive this
   * plan, and destruction is centralized in `Session.dispose` / `GpuContext.destroy` (ADR 0004,
   * "allocation and destruction in one place"). Destroying here would make other plans pointing at
   * the same instances, or the next derivation, hold destroyed buffers.
   */
  #registerPrepared(key: string, plan: PreparedPlan): void {
    this.#state.prepared.set(key, plan);
    // Only one is added per run, so the excess is always exactly one.
    if (this.#state.prepared.size > PREPARED_PLAN_CAPACITY) {
      const oldest = this.#state.prepared.keys().next().value;
      if (oldest !== undefined) {
        this.#state.prepared.delete(oldest);
        // MUST: the slot backing of an evicted plan is left dangling (when those bindings come
        // again it is a miss run and the backing is not used). Keeping it would hold the VRAM of
        // the intermediate buffers of a signature that will never hit again for the whole Session
        // lifetime.
        this.#retireBacking(oldest);
      }
    }
  }

  /**
   * Derivation and registration on the miss path (the one function shared by run / enqueue).
   * Records the plans and params it added in `additions` ({@link Session.#forgetDerived} removes
   * them once a GPU-side failure shows).
   *
   * MUST: registration happens only **after `RecipeBuilder.buildRecipes` ran to completion and
   * returned**. Storing the partial recipes of an execution that threw midway would make the next
   * execution with the same bindings run a step list with gaps and return wrong values without an
   * exception.
   * MUST: params are recorded even when derivation fails midway (finally). Params stored before the
   * failure, if their allocation hit a non-fatal OOM, keep being hit by the next derivation while
   * invalid.
   */
  async #derivePrepared(
    key: string,
    derived: PlannedSteps,
    stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
    additions: CacheAdditions,
  ): Promise<PreparedPlan> {
    let built: Awaited<ReturnType<RecipeBuilder["buildRecipes"]>>;
    try {
      built = await this.#recipeBuilder.buildRecipes(derived.steps, stateShapes);
    } finally {
      additions.params.push(...this.#recipeBuilder.addedParams);
    }
    const plan: PreparedPlan = {
      shapes: derived.shapes,
      recipes: built.recipes,
      fusions: derived.fusions,
      generation: built.generation,
    };
    this.#registerPrepared(key, plan);
    additions.prepared.push({ key, plan });
    return plan;
  }

  /**
   * Moves the backings that a failed execution unit newly built to the destruction queue (ADR 0095
   * decision 2). The destruction ({@link Session.#destroyRetired}) is done by the caller at a safe
   * point.
   *
   * MUST: retire only when the generation number matches (the matching of {@link CacheAdditions} —
   * so that another generation rebuilt under the same key in a hard-to-reach race is not retired by
   * mistake).
   */
  #retireBuiltBackings(additions: CacheAdditions): void {
    for (const { key, build } of additions.backings) {
      if (this.#backings.get(key)?.build === build) this.#retireBacking(key);
    }
  }

  /**
   * Removes the params and plans added by an execution unit whose GPU-side failure has shown
   * (params are only removed from the cache, not destroyed — the weights arena owns them). Not
   * called on a host-side failure ({@link CacheAdditions}).
   *
   * MUST: matching by identity ({@link CacheAdditions}).
   * MUST: when a plan is removed, that key's backing is retired too (the same discipline as
   * eviction). The backing has baked that plan's recipes (= the instances of the params being
   * removed), so letting it outlive the plan would make a hit on the re-derived plan reuse the old
   * bundle and repeat the same failure.
   * NOTE: on the ordinary path, if the key of a removed plan has a backing, the same failed
   * execution unit built it and {@link Session.#retireBuiltBackings} has already retired it (run
   * derives only on miss runs, and miss runs build no backing; enqueue derives and builds in the
   * same unit). The retirement here takes effect only in the race where a later run hit that plan
   * and built a backing before the batch's failure settled — a defence against a hard-to-reach
   * race.
   */
  #forgetDerived(additions: CacheAdditions): void {
    this.#recipeBuilder.forgetParams(additions.params);
    for (const { key, plan } of additions.prepared) {
      if (this.#state.prepared.get(key) !== plan) continue;
      this.#state.prepared.delete(key);
      this.#retireBacking(key);
    }
  }

  /**
   * Rollback at a batch's failed settlement (the `fail` of the finalizer an enqueue registers at
   * acceptance).
   *
   * A non-fatal OOM leaves the enqueue successful and first shows in `finish()`'s errorScope, so
   * the enqueue body's catch (which looks only at host-side failures) cannot roll it back. Rollback
   * happens only when the span body failed (`origin === "gpu"` — {@link BatchFailureOrigin}). In a
   * batch that settled with host-side failures only, the created backings, params and plans are
   * valid (a backing newly built by the enqueue that failed on the host side has already been
   * retired by its catch).
   * MUST: do not destroy here. At settlement, a later run of the same Session may already have
   * received the released section lock and started encoding (whether settlement or the later start
   * comes first is decided only by the number of microtask hops), and destroying a backing that run
   * holds while unsubmitted would break flush-before-destroy of ADR 0004. Destruction is pushed as
   * one entry onto the Session's serialization chain — every body on the chain leaves after
   * finishing its submit / discard, so destruction on the chain is "a point with no unsubmitted
   * encoding". The owner of destruction stays the single {@link Session.#destroyRetired} as before.
   * NOTE: when a batch's settlement races with the start of a later run, a backing may remain for a
   * key with no plan, which is harmless. If the later run grabbed the plan before it was removed
   * and built a backing, that backing bakes in the removed params. If those are invalid, that run
   * itself fails on its errorScope and retires the backing it built. If valid, the weights arena
   * keeps the instances until dispose and their contents are the same, so the values are correct
   * even when a hit after re-derivation reuses it.
   */
  #rollbackSettled(additions: CacheAdditions, origin: BatchFailureOrigin): void {
    if (origin !== "gpu") return;
    this.#retireBuiltBackings(additions);
    this.#forgetDerived(additions);
    if (this.#retired.length === 0) return;
    void this.#serialize(() => {
      this.#destroyRetired();
      return Promise.resolve();
    });
  }

  /**
   * Activates the slot backing of a hit run (the retained one if present; otherwise built after
   * fitting into the budget — a different signature is **added** if within budget, and when over
   * budget the oldest are retired before allocation).
   * `built` is "whether this run / enqueue newly built it" — read by the failure-recovery
   * discipline (run's catch, enqueue's catch and the batch's failed settlement —
   * {@link Session.#retireBuiltBackings}).
   *
   * MUST: called only inside a run's `withScopeLock` / errorScope section, or inside the span of
   * the batch an enqueue belongs to (the section lock / errorScope the batch holds). createBuffer
   * returns an invalid buffer instead of throwing synchronously when over a limit, and a
   * createBindGroup validation failure does not throw either, so without the enclosure this becomes
   * the silent failure "dispatches write into invalid slots / bind groups".
   * MUST: the span from allocation to registration in `#backings` (= establishing ownership) is
   * wrapped in try/catch, and on a synchronous exception midway what was allocated goes to
   * `#retired`. Instances leaked in this window are reachable neither from `#retireBacking(key)`
   * nor from `Session.dispose()`, and their volume is the largest in this Session (the region sum)
   * (ADR 0004 "allocation and destruction in one place" holds on the failure path too).
   * NOTE: on the run path the recipe list has always gone once through the same plan
   * (`planRecipes`) on a miss run (run builds a backing **only on hit runs**). `#enqueueOnce`
   * builds a backing **from the first call** as a MUST, but in both cases violations of reference
   * counting are handled by the plan's own closure check (it throws immediately on negative counts
   * / missed releases).
   */
  #activateBacking(
    key: string,
    recipes: readonly StepRecipe[],
    shapes: ReadonlyMap<string, readonly number[]>,
    residentInputs: ReadonlyMap<string, ResidentTensor>,
  ): { readonly backing: ActiveBacking; readonly built: boolean } {
    const current = this.#backings.get(key);
    if (current !== undefined) {
      // LRU: move the hit key to newest (Map insertion order is oldest first).
      this.#backings.delete(key);
      this.#backings.set(key, current);
      return { backing: current, built: false };
    }
    const graph = this.#state.graph;
    const device = this.#state.gpu.device;
    const plan = planRecipes(recipes, this.#state.transientLimits);
    // Sizes of the owned input buffers (resident inputs are not owned, so they are not listed).
    // MUST: the size uses the same formula as `#bindInput` on the arena path (including the 4-byte
    // floor — no 0-size buffer is bound for a 0-element input). **Both** the budget accounting and
    // the allocation **read this one table** (with two formulas, one would move alone and the
    // budget accounting would silently drift).
    const ownedInputSizes = new Map<string, number>(
      graph.inputs
        .filter((spec) => !residentInputs.has(spec.name))
        .map((spec) => [spec.name, Math.max(4, numel(resolvedShape(shapes, spec.name)) * 4)]),
    );
    const inputBytes = [...ownedInputSizes.values()].reduce((total, size) => total + size, 0);
    // MUST: fit into the budget **before** allocating (retirement does not destroy, it only pushes
    // onto the destruction queue — returned only after this run's flush). Retiring before
    // allocation guarantees only the upper bound of the retained set's accounting (retained + new ≤
    // max(budget, new) — a new one that alone exceeds the budget retires every retained one and is
    // kept alone; ADR 0095 decision 1) — the retired old backings stay in VRAM until after flush,
    // so at the moment the new one is allocated, old and new physically overlap (this overlap
    // window is outside the accounting — ADR 0095 decision 4).
    this.#evictBackingsFor(plan.totalBytes + inputBytes);
    // What has been allocated and the retained resident inputs until ownership is established
    // (registration in `#backings`). If a synchronous exception occurs midway, the catch reclaims
    // them from here.
    const regions: GPUBuffer[] = [];
    const ownedInputs: GPUBuffer[] = [];
    const retained: ResidentTensor[] = [];
    try {
      for (const size of plan.regions) {
        regions.push(device.createBuffer({ size, usage: STORAGE_USAGE }));
      }
      // Plain input buffers are owned by the backing (removes the per-run allocation and the
      // changing writeBuffer target). Resident inputs bind the GpuContext-owned instance as is and
      // are **not owned**.
      // The sizes come from `ownedInputSizes` above (the same table as the budget accounting).
      // Unchanged for the same signature.
      const inputs = new Map<string, GPUBuffer>(
        graph.inputs.map((spec) => {
          const resident = residentInputs.get(spec.name);
          if (resident !== undefined) return [spec.name, resident[RUNTIME_INTERNAL].buffer];
          const size = ownedInputSizes.get(spec.name);
          if (size === undefined) {
            throw new ExecutionError(`入力 '${spec.name}' の大きさが表に無い（簿記の破れ）`);
          }
          const buffer = device.createBuffer({ size, usage: HOST_WRITTEN_USAGE });
          ownedInputs.push(buffer);
          return [spec.name, buffer];
        }),
      );
      const baked = bakeBindGroups(recipes, { device, plan, regions, inputs });
      // The readback sources are fixed here. The shape in which an initializer is a graph output
      // (the IR allows it) is not in the value-name mapping, so the weight fallback on the
      // `#readOutputs` side handles it as is.
      const outputs = new Map<string, ValueBinding>();
      for (const name of graph.outputs) {
        const value = baked.values.get(name);
        if (value !== undefined) outputs.set(name, value);
      }
      const residents = [...residentInputs.values()];
      // MUST: the baking references stay held for as long as the backing lives (returned at
      // retirement). This is the only basis for "making ResidentTensor.dispose fail loudly while
      // referenced".
      for (const resident of residents) {
        resident[RUNTIME_INTERNAL].retainBaked();
        retained.push(resident);
      }
      // The generation id is the monotonic "order of creation" counter itself (the same numbering
      // as the diagnostic `buildCount`).
      // MUST: the id and the cumulative count are taken from **one single expression** (incremented
      // separately, a path that skips one would skew them permanently into "rebuilt yet the same
      // generation").
      const build = this.#backingBuilds + 1;
      const backing: ActiveBacking = {
        key,
        build,
        bytes: plan.totalBytes,
        inputBytes,
        inputs,
        residents,
        groups: baked.groups,
        plan,
        regions,
        outputs,
        owned: new Set([...regions, ...ownedInputs]),
      };
      this.#backings.set(key, backing);
      this.#backingBuilds = build;
      return { backing, built: true };
    } catch (cause) {
      // MUST: allocated buffers are only put on the existing destruction path (`#retired` →
      // `#destroyRetired`). Destroying here would give destruction two owners (ADR 0004). At this
      // point not a single dispatch using the bind groups has been queued (`#activateBacking`
      // precedes the dispatches), so destruction at the cleanup point cannot break submitted
      // commands.
      for (const buffer of regions) this.#retired.push(buffer);
      for (const buffer of ownedInputs) this.#retired.push(buffer);
      for (const resident of retained) resident[RUNTIME_INTERNAL].releaseBaked();
      throw cause;
    }
  }

  /**
   * Prepares the **context-side** bind groups of a generation run (the split baking of ADR 0066
   * decision 5).
   *
   * The bundle's lifetime is **the pair (context, backing instance)**, so the backing's generation
   * id alone is enough to tell them apart (the context side looks only at its own cache).
   * Consecutive runs on the same context bake 0 times; only a context switch or a backing rebuild
   * bakes once, and that count becomes the diagnostic `stateBacking.rebindCount`.
   *
   * MUST: the order "match the generation id → re-bake → dispatch" is **closed inside this one
   * function**. Creating another way to read the cache without the match would create the shape
   * that dispatches with an old group binding a retired backing's instances. Retired instances
   * **live until that run's cleanup** (`#retired` → `#destroyRetired` after flush), so the old
   * bundle passes validation while binding another plan's instances, and **only the values change,
   * silently** (measured by fault injection with the match removed — all 64 output elements
   * mismatched, zero exceptions).
   * MUST: called only inside a run's errorScope section (a createBindGroup validation failure does
   * not throw).
   * NOTE: only the **buffers** of `encoding` go into the bundle (`past` / `query` are the uniform's
   * contents). That is why the bundle stays valid even though the logical length changes every run,
   * and the only re-bake condition is the backing's generation.
   */
  #generationGroups(
    face: { readonly context: GenerationContext; readonly encoding: GenerationEncoding },
    backing: ActiveBacking,
    recipes: readonly StepRecipe[],
  ): BakedGroups {
    const internals = face.context[RUNTIME_INTERNAL];
    const cached = internals.bakedGroups(backing.build);
    if (cached !== undefined) return cached;
    const groups = bakeGenerationBindGroups(recipes, {
      device: this.#state.gpu.device,
      // MUST: the same plan, the same regions and the same inputs as the Session-side baking (not
      // rebuilt).
      plan: backing.plan,
      regions: backing.regions,
      inputs: backing.inputs,
      generation: face.encoding,
    });
    internals.setBakedGroups(backing.build, groups);
    this.#stateRebinds += 1;
    return groups;
  }

  /**
   * Moves one retained backing to the destruction queue (the actual `destroy()` happens once, at
   * the cleanup point after flush / submit). A key not retained is a no-op.
   *
   * The baking references of resident inputs are returned here. Returning them is allowed because
   * not a single dispatch using the retired backing's bind groups is queued from here on (only
   * retained backings are queued).
   * MUST: the bundles live contexts baked ({@link Session.#generationGroups}) are also dropped
   * **here** — a bundle holds the retired instances, so even though it is matched by the generation
   * token, its references extend their lifetime.
   */
  #retireBacking(key: string): void {
    const current = this.#backings.get(key);
    if (current === undefined) return;
    this.#backings.delete(key);
    for (const buffer of current.owned) this.#retired.push(buffer);
    for (const resident of current.residents) resident[RUNTIME_INTERNAL].releaseBaked();
    for (const context of this.#contexts) {
      context[RUNTIME_INTERNAL].dropBakedGroups(current.build);
    }
  }

  /** Moves every retained backing to the destruction queue (dispose). */
  #retireAllBackings(): void {
    for (const key of [...this.#backings.keys()]) this.#retireBacking(key);
  }

  /**
   * Retires backings **oldest first** until a new backing (regions of `bytes`) fits into the
   * budget.
   *
   * Budget = {@link SessionOptions.planBackingBudgetBytes}. A new one that alone exceeds it retires
   * every retained one and is kept alone (the same shape as the former capacity of 1), so residency
   * never exceeds `max(budget, the largest single one)` — exactly the amount the estimate
   * (src/runtime/estimate.ts) puts on its account.
   */
  #evictBackingsFor(bytes: number): void {
    const budget = this.#state.planBackingBudgetBytes;
    let resident = [...this.#backings.values()].reduce(
      (total, backing) => total + backing.bytes + backing.inputBytes,
      0,
    );
    // Retirement shrinks the map during iteration, so iterate over a copy (oldest-first order
    // kept).
    for (const [key, backing] of [...this.#backings]) {
      if (resident + bytes <= budget) break;
      this.#retireBacking(key);
      resident -= backing.bytes + backing.inputBytes;
    }
  }

  /**
   * Actually destroys the slot buffers waiting for destruction.
   *
   * MUST: called only after flush (or discard) completed — submitting an encoding that references a
   * destroyed buffer fails the whole command buffer, and even unrelated dispatches stay unexecuted
   * while wrong values silently remain (ADR 0004).
   * NOTE: `destroy()` after device loss is a harmless no-op.
   */
  #destroyRetired(): void {
    for (const buffer of this.#retired) buffer.destroy();
    this.#retired.length = 0;
  }

  /**
   * A value's declared dtype. A miss is a violation of a runtime-internal invariant (declaredDtypes
   * holds every value).
   */
  #declaredDtype(name: string): IrDtype {
    const dtype = this.#state.dtypes.get(name);
    if (dtype === undefined) {
      throw new ExecutionError(`値 '${name}' の宣言 dtype が無い`);
    }
    return dtype;
  }

  /**
   * Input check (**value-dependent, so needed every run** — not skippable even on a hit run). If it
   * passes, returns the host array to write to the GPU, and **`undefined` for a resident input**
   * (nothing to write).
   */
  #checkInput(
    name: string,
    value: RunInput | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): Tensor["data"] | undefined {
    if (value instanceof ResidentTensor) {
      this.#checkResidentInput(name, value, shapes);
      return undefined;
    }
    return this.#checkTensorInput(name, value, shapes);
  }

  /** Input check of a host array. If it passes, returns the array to write to the GPU as is. */
  #checkTensorInput(
    name: string,
    tensor: Tensor | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): Tensor["data"] {
    // Missing inputs and shape mismatches are already handled by bindSymbols. Only dtype and
    // element count are checked here.
    if (tensor === undefined) throw new ExecutionError(`入力 '${name}' が渡されていない`);
    // MUST: both the discriminant and the actual data are checked against the declared dtype. Every
    // element type is 4 bytes, so a mix-up would pass through writeBuffer and become a
    // reinterpretation of the bit pattern (no exception).
    const dtype = this.#declaredDtype(name);
    const expected = HOST_ARRAY[dtype];
    if (tensor.dtype !== dtype || !(tensor.data instanceof expected)) {
      throw new ExecutionError(
        `入力 '${name}': 宣言 dtype '${dtype}'（${expected.name}）に対し渡されたのは '${tensor.dtype}' / ${tensor.data.constructor.name}`,
      );
    }
    const shape = resolvedShape(shapes, name);
    const count = numel(shape);
    if (tensor.data.length !== count) {
      throw new ExecutionError(
        `入力 '${name}': 要素数 ${tensor.data.length} が shape [${shape.join(
          ",",
        )}] の ${count} と合わない`,
      );
    }
    return tensor.data;
  }

  /**
   * Input check of a resident input.
   *
   * MUST: a resident tensor has no dtype (a lifetime class of bytes and a size only), so only the
   * size can be checked. **An exact match** is required — allowing a large resident tensor to be
   * bound by its head only would silently change the element count on the WGSL side, where
   * `arrayLength()` is derived from the byte size of the bound range (the same reason the arena
   * "does not hand out a buffer larger than requested").
   */
  #checkResidentInput(
    name: string,
    resident: ResidentTensor,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): void {
    assertResidentUsable(resident, this.#state.gpu, `入力 '${name}'`);
    const shape = resolvedShape(shapes, name);
    const wanted = Math.max(4, numel(shape) * 4);
    if (resident.byteLength !== wanted) {
      throw new ExecutionError(
        `入力 '${name}': 常駐テンソル '${resident.label}' の ${resident.byteLength} バイトが shape [${shape.join(
          ",",
        )}] の ${wanted} バイトと合わない`,
      );
    }
  }

  /**
   * Input binding on the arena path. A plain input allocates one run-lifetime buffer and writes it;
   * a resident input returns the GpuContext-owned instance as is (**neither allocation nor
   * writeBuffer is issued**).
   *
   * MUST: a resident input takes one reservation at the moment of binding and is put on `bound`.
   * This path (miss run) proceeds in the order "bind the raw buffer into env → await `buildRecipes`
   * → encode", and there is not a single baking reference yet. A `dispose()` arriving in the await
   * window would slip through, and the resumed encoding would hold a destroyed buffer (the run
   * fails, caught by errorScope, but far from the real cause). With the reservation, the mistake
   * fails immediately at the dispose call site as a {@link ResidentTensorError}.
   */
  #bindInput(
    name: string,
    value: RunInput | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    bound: ResidentTensor[],
  ): GPUBuffer {
    if (value instanceof ResidentTensor) {
      this.#checkResidentInput(name, value, shapes);
      value[RUNTIME_INTERNAL].retainBound();
      bound.push(value);
      return value[RUNTIME_INTERNAL].buffer;
    }
    const data = this.#checkTensorInput(name, value, shapes);
    const buffer = arena.allocHostWritten(Math.max(4, data.length * 4), HOST_WRITTEN_USAGE);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * Input write of a backed run (**overwrites** the resident buffer the backing owns).
   *
   * MUST: there is no unsubmitted encoding this `writeBuffer` could overtake. Three grounds —
   * ① run / enqueue / dispose of the same Session are serialized by {@link Session.#chain}
   * ② a preceding execution leaves no unsubmitted encoding when it returns: a run returns only
   * after the readback has completed (on the two-stage-wait path also flush's
   * `onSubmittedWorkDone`), and **an enqueue always submits eagerly at the end**
   * (`SubmitScheduler.submitPending` in {@link "../gpu/submit.ts"}) ③ within an execution, input
   * writes precede all encoding. `queue.writeBuffer` lands on the queue timeline in issue order, so
   * it does not overtake **submitted** preceding dispatches; only unsubmitted encoding is overtaken
   * (ADR 0004 invariant ④). If any of these breaks, a previous execution's dispatches read the new
   * input — a silent wrong value. Therefore no "path that leaves pending work without submitting
   * after an enqueue" may be created.
   */
  #writeInput(backing: ActiveBacking, name: string, data: Tensor["data"]): void {
    const buffer = backing.inputs.get(name);
    if (buffer === undefined) throw new ExecutionError(`入力 '${name}' の常駐バッファが無い`);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
  }

  /**
   * The table of readback sources (a backing run's output mapping fixed at construction; a miss
   * run's bindings it built). Whether reading back is allowed lives in
   * {@link ValueBinding.readable} (only pinned slots and non-slot values), so miss runs and hit
   * runs do not have two separate checks.
   */
  #runValues(
    backing: ActiveBacking | undefined,
    missValues: ReadonlyMap<string, ValueBinding> | undefined,
  ): ReadonlyMap<string, ValueBinding> {
    const values = backing?.outputs ?? missValues;
    if (values === undefined) throw new ExecutionError("run の束縛先が組まれていない");
    return values;
  }

  /**
   * Resolves one graph output to its source buffer and size (**the only readback rule** — the
   * source at settlement and the copy into staging both go through here).
   *
   * MUST: initializers are looked up too. The IR allows initializer names in graph.outputs (the
   * defined-name check in format/ir.ts), and such values are not among the run-lifetime bindings.
   * MUST: whether reading back is allowed is decided by {@link ValueBinding.readable} alone (only
   * pinned slots and non-slot values). With an eligibility check per path, an edit that loosens
   * only one of them would let through "reading back an unpinned slot" = returning contents swapped
   * by reassignment.
   */
  #resolveOutput(
    values: ReadonlyMap<string, ValueBinding>,
    shapes: ReadonlyMap<string, readonly number[]>,
    name: string,
  ): ResolvedOutput {
    const weight = this.#state.weightBuffers.get(name);
    const binding =
      values.get(name) ?? (weight === undefined ? undefined : { buffer: weight, readable: true });
    if (binding === undefined) throw new ExecutionError(`グラフ出力 '${name}' のバッファが無い`);
    if (!binding.readable) {
      throw new ExecutionError(`グラフ出力 '${name}' がピン留めされておらず読み戻せない`);
    }
    const shape = resolvedShape(shapes, name);
    const count = numel(shape);
    return {
      shape,
      count,
      buffer: binding.buffer,
      offset: binding.offset ?? 0,
      size: outputByteSize(count),
    };
  }

  /**
   * Resolves, from the slot backing, the sources of the graph outputs read back at settlement
   * ({@link Session.enqueueRead}). The eligibility check and the handling of initializers are the
   * one and only {@link Session.#resolveOutput} (pinned slots only; an initializer is the weight
   * buffer itself).
   *
   * MUST: when the source is the resident input's instance itself (a graph output that aliases a
   * resident input — a degenerate graph of reshape / identity expand), acceptance fails. The copy
   * runs **at settlement**, not at queue time, so a `ResidentTensor.write` after `admitted` and
   * before settlement (which does not look at use reservations — the borrowed contract) would get
   * mixed into the readback, and a value different from the one at queue time would be returned
   * silently. The counterpart of the self-copy guard of {@link Session.#resolveCopyOutputs}.
   */
  #planReadback(
    backing: ActiveBacking,
    shapes: ReadonlyMap<string, readonly number[]>,
    residentInputs: ReadonlyMap<string, ResidentTensor>,
  ): readonly PlannedRead[] {
    return this.#state.graph.outputs.map((name) => {
      const { shape, count, buffer, offset, size } = this.#resolveOutput(
        backing.outputs,
        shapes,
        name,
      );
      for (const [input, resident] of residentInputs) {
        if (resident[RUNTIME_INTERNAL].buffer === buffer) {
          throw new ExecutionError(
            `enqueueRead: 出力 '${name}' は常駐入力 '${input}' の別名（reshape / 恒等 expand）— ` +
              "読み戻しは決着時に走るので、決着前の write が読み戻しに混ざる",
          );
        }
      }
      return { name, shape, count, source: { buffer, offset, size } };
    });
  }

  /**
   * Rebuilds the ArrayBuffers returned at settlement into host {@link Tensor}s (the same rule as
   * {@link Session.#collectStaged}).
   */
  #collectRead(reads: readonly PlannedRead[], buffers: readonly ArrayBuffer[]): RunOutputs {
    const outputs: Record<string, Tensor> = Object.create(null);
    reads.forEach((item, index) => {
      outputs[item.name] = hostTensor(
        this.#declaredDtype(item.name),
        item.shape,
        buffers[index],
        item.count,
      );
    });
    return outputs;
  }

  /**
   * Allocates a staging per graph output and has `copy` queue the copy into it (**synchronous**).
   *
   * MUST: only graph outputs are read back. Intermediate values have had their contents swapped by
   * reassignment.
   * NOTE: the queue target is a parameter because the single-fence path (the run body's command
   * list — `SubmitScheduler.copyBuffer` in {@link "../gpu/submit.ts"}) and the two-stage-wait path
   * (a readback-only encoder) differ **only in where they queue**. How staging is built and the
   * readback eligibility check do not exist twice per path (eligibility, size and source are
   * {@link Session.#resolveOutput}).
   */
  #stageOutputs(
    env: ReadonlyMap<string, ValueBinding>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    copy: (source: GPUBuffer, offset: number, size: number, staging: GPUBuffer) => void,
  ): readonly StagedOutput[] {
    const staged: StagedOutput[] = [];
    for (const name of this.#state.graph.outputs) {
      const { shape, count, buffer, offset, size } = this.#resolveOutput(env, shapes, name);
      const staging = arena.allocHostRead(size);
      staged.push({ name, shape, count, staging });
      copy(buffer, offset, size, staging);
    }
    return staged;
  }

  /**
   * Waits for the staging maps to complete (the resolution condition is **the last use of this
   * buffer = completion of the queued copy**).
   *
   * MUST: multiple outputs are awaited **in parallel**. Sequential awaits would pay the fixed cost
   * of the wait once per output.
   * MUST: some implementations may never resolve mapAsync after loss (the measurement is in the doc
   * of raceCanaryDeviceLost), so it is raced — insurance that turns a hang into a failure.
   */
  async #awaitStaged(staged: readonly StagedOutput[]): Promise<void> {
    await this.#state.gpu[RUNTIME_INTERNAL].raceDeviceLost(
      Promise.all(staged.map((item) => item.staging.mapAsync(MAP_MODE.READ))),
      "readback",
    );
  }

  /** Rebuilds mapped staging into host {@link Tensor}s (synchronous, including unmap). */
  #collectStaged(staged: readonly StagedOutput[]): RunOutputs {
    // MUST: containers that accumulate keys derived from graph output names have a null prototype.
    // With a plain `{}`, an output named "__proto__" turns its Tensor into the [[Prototype]], and
    // that output is returned without appearing as an own property (missing from Object.keys) — a
    // silent omission.
    const outputs: Record<string, Tensor> = Object.create(null);
    for (const item of staged) {
      const copy = item.staging.getMappedRange().slice(0);
      item.staging.unmap();
      // MUST: the TypedArray read back is chosen from the declared dtype (every element type is 4
      // bytes, so fixing it to f32 would silently reinterpret the bit patterns of i32 / bool
      // outputs).
      outputs[item.name] = hostTensor(this.#declaredDtype(item.name), item.shape, copy, item.count);
    }
    return outputs;
  }

  /**
   * Readback on the single-fence path (H-1). The copies queued on the run body's command list are
   * the only thing waited on, and **this mapAsync is the run's fence itself**.
   */
  async #finishStagedRead(staged: readonly StagedOutput[]): Promise<RunOutputs> {
    await this.#awaitStaged(staged);
    // MUST: the measurement window is closed **after** the fence (closing it before would make the
    // measurement too small and the chunks grow, the dangerous side heading into TDR territory —
    // src/gpu/submit.ts). The window includes host cleanup, so the estimate errs large, but the
    // direction is the safe side and it is kept.
    this.#state.scheduler.closeMeasurementWindowAfterFence();
    return this.#collectStaged(staged);
  }

  /**
   * Readback on the two-stage-wait path (gpuTiming on / 0 graph outputs — `singleFence` of
   * {@link Session.#runOnce}). It is **a separate submit from a separate encoder** from the run
   * body's flush, and its fence is the second one after `onSubmittedWorkDone`.
   */
  async #readOutputs(
    env: ReadonlyMap<string, ValueBinding>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
  ): Promise<RunOutputs> {
    const device = this.#state.gpu.device;
    // MUST: this copy → submit also goes inside an errorScope (the run body's scope has been
    // popped, and this is a separate submit from its own encoder). A validation failure such as a
    // missing COPY_SRC does not throw, and the readback silently returns all zeros. Allocating
    // staging can exceed the device's headroom, so out-of-memory is opened alongside.
    // MUST NOT: no await between the push and issuing the pop. Keeping this section synchronous is
    // what prevents interleaving of the LIFO without taking the per-device lock.
    pushFailureScopes(device);
    let popped = false;
    try {
      const encoder = device.createCommandEncoder();
      const staged = this.#stageOutputs(env, shapes, arena, (source, offset, size, staging) =>
        encoder.copyBufferToBuffer(source, offset, staging, 0, size),
      );
      device.queue.submit([encoder.finish()]);

      // MUST: the pop is issued before mapAsync (so that no other operation is absorbed while
      // waiting on mapAsync). mapAsync itself enters no scope, but this is inside #runOnce's lock
      // section, so its failure is never attributed to someone else's scope (every GPU operation of
      // a run is inside the lock section — the invariant above).
      const pending = popFailureScopes(device, "出力の readback");
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;

      await this.#awaitStaged(staged);
      return this.#collectStaged(staged);
    } finally {
      // `!popped` holds only when the body left with an exception. Cleanup does not hide that
      // exception.
      // Destroying staging belongs to RunArena (#runOnce always proceeds to arena.destroy() on both
      // the success and the failure path — ADR 0004 "allocation and destruction in one place").
      if (!popped) await discardFailureScopes(device);
    }
  }
}

/**
 * The source of a {@link PreparedModel} — one graph of an opened container (`krm` / in-memory
 * container).
 */
type PreparedSource = { readonly opened: BoundContainer; readonly graphName: string };

/**
 * The product of the admission phase before weights are fetched (ADR 0070 decision 5 / graph-first)
 * — it decides "can this be executed" from the graph declaration alone, estimates the required
 * memory, and is the entry point of Session construction as is.
 *
 * It holds only three things: ① the source (the opened container and the target graph name)
 * ② the `IrGraph` ③ the residency plan (residency classes), and **holds no GPU resources at all**
 * (`estimate` stays a pure function — "GPU-independent" of decision 5). It holds not a single byte
 * of weights: {@link PreparedModel.createContainerSession} fetches blocks one at a time and lets go
 * of each once uploaded to the GPU (one fence per part — ADR 0108 decision 9).
 *
 * MUST: construction has {@link PreparedModel.createContainerSession} as its only entry point, so
 * mod.ts exports it **as a type only** (the same convention as `Session` / `GpuContext` —
 * constructing it directly would allow "a Session of a model that cannot execute" bypassing the
 * capability check and the contract checks).
 */
export class PreparedModel {
  readonly #source: PreparedSource;
  readonly #graph: IrGraph;
  /**
   * Residency classes and byte counts (decided by the graph alone) — the one instance shared by the
   * estimate and construction.
   */
  readonly #residency: ReadonlyMap<string, WeightResidency>;

  private constructor(source: PreparedSource, graph: IrGraph) {
    // Unsupported ops / storage are listed in full and then fail (failing one at a time would not
    // tell how many are missing).
    // MUST: these two checks pass **before weights are touched** (the very reason the two-step
    // boundary exists — the weights of a model that cannot execute are no longer downloaded only to
    // fail).
    // NOTE: missing capability and contract violations are properties of the model (graph), not of
    // the source, so no attribution is added — the message stays the same whichever source it was
    // opened from.
    assertRuntimeSupport(graph, RUNTIME_SUPPORT);
    validateGraphContracts(graph);
    this.#source = source;
    this.#graph = graph;
    this.#residency = planWeightResidency(graph);
  }

  /**
   * Prepares from one graph of an opened container (`openContainer` in
   * {@link "../format/container/open.ts"} for a `krm` / `openMemoryContainer` in
   * {@link "../format/container/memory.ts"} for an in-memory container).
   * The merge (graph × binding table) was done when the container was opened, so this is admission
   * only.
   */
  static fromContainer(opened: BoundContainer, graphName: string): PreparedModel {
    const bound = opened.graphs[graphName];
    if (bound === undefined) {
      throw new ExecutionError(
        `コンテナにグラフ '${graphName}' が無い（在るのは ${Object.keys(opened.graphs).join(
          " / ",
        )}）`,
      );
    }
    return new PreparedModel({ opened, graphName }, mergedGraph(bound, graphName));
  }

  /**
   * The graph's **declaration** (inputs / outputs, symbolic dimensions, value shapes).
   *
   * Callers (the models pipelines) want to finish "matching the graph declaration against their own
   * configuration" in the admission phase — this is the only thing readable at a stage that fetches
   * no weight block at all, so the reading point of the declaration is closed into this one getter.
   */
  get graph(): IrGraph {
    return this.#graph;
  }

  /**
   * Estimate of the required memory (ADR 0070 decision 5) — a pure function that does not touch the
   * GPU and returns only byte counts per category and the list of what is not accounted for. It
   * makes no admissibility decision (the final guard remains the out-of-memory errorScope). **The
   * same single implementation** as the graph-level entry `estimateGraphMemory`.
   */
  estimate(options: EstimateOptions = {}): AdmissionReport {
    return estimateGraphMemory(this.#graph, this.#residency, options);
  }

  /**
   * Creates a Session by taking the container's blocks one at a time (ADR 0108 decision 9 —
   * errorScope per block, fence per part). A block's sha256 is verified only when the source is
   * unverified (`BlockSource.verified` is false — full bytes, a part list, a local directory).
   * Through HF, the fetch layer has already verified the whole file (ADR 0109 decision 7 /
   * container-v1 §7 — verification is the job of the source's `readBlock`).
   */
  async createContainerSession(gpu: GpuContext, options: SessionOptions = {}): Promise<Session> {
    const source = this.#source;
    return await Session.build(
      gpu,
      this.#graph,
      this.#residency,
      containerBatches(source.opened, source.graphName),
      options,
    );
  }
}

/**
 * Streams the container's supply plan as one {@link WeightBatch} per part. Within a part, in the
 * declaration order of the initializers (= the order of the binding table), and pieces in index
 * order. The scale is in the same part as piece 1 (rule ③), so it rides along with piece 1's item.
 *
 * Blocks are not handed over after reading the whole part; one block is read each time one `items`
 * entry is pulled (lazy — {@link WeightBatch.items}).
 *
 * It is exported for package-internal use by tests (items can be pulled without touching the GPU,
 * so the scale shapes passed to Session construction can be checked even in an environment without
 * an adapter). It is not exposed on the public surface of `mod.ts` (ADR 0008).
 */
export const containerBatches = async function* (
  opened: BoundContainer,
  graphName: string,
): AsyncGenerator<WeightBatch, void, unknown> {
  const bound = opened.graphs[graphName];
  if (bound === undefined) throw new ExecutionError(`コンテナにグラフ '${graphName}' が無い`);
  const byPart = new Map<
    number,
    { readonly name: string; readonly supply: InitializerSupply; readonly index: number }[]
  >();
  for (const [name, supply] of bound.supplies) {
    supply.blocks.forEach((block, index) => {
      const list = byPart.get(block.part) ?? [];
      list.push({ name, supply, index });
      byPart.set(block.part, list);
    });
  }
  /** Builds the items of one part, reading one block each time one is pulled. */
  const partItems = async function* (
    part: number,
  ): AsyncGenerator<ReadyInitializer, void, unknown> {
    for (const { name, supply, index } of byPart.get(part) ?? []) {
      const block = supply.blocks[index];
      const where = `graph '${graphName}' initializer '${name}'`;
      const payload = (await opened.readBlock(block.id)).subarray(0, block.payloadBytes);
      if (supply.encoding.codec === "ternary") assertTernaryCodes(payload, where);
      let scale: ReadyScale | undefined;
      if (index === 0 && supply.scale !== undefined) {
        const shape = bound.declaration.values[name].shape.map(Number);
        scale = {
          bytes: (await opened.readBlock(supply.scale.id)).subarray(0, supply.scale.payloadBytes),
          shape: groupScaleShape(
            shape,
            supply.encoding.rowAxis ?? 0,
            supply.encoding.groupSize ?? 1,
          ),
        };
      }
      const piece =
        supply.blocks.length === 1
          ? undefined
          : {
              rowOffset: block.rows[0],
              rows: block.rows[1] - block.rows[0],
              first: index === 0,
              last: index === supply.blocks.length - 1,
            };
      yield {
        name,
        payload,
        ...(scale === undefined ? {} : { scale }),
        ...(piece === undefined ? {} : { piece }),
      };
    }
  };
  for (const part of [...byPart.keys()].sort((a, b) => a - b)) {
    yield { origin: `part ${part}`, items: partItems(part) };
  }
};

/**
 * The entry of the two-step boundary that finishes admission on one graph of an opened container
 * (ADR 0070 decision 5 / graph-first). The merge (graph × binding table) was done when the
 * container was opened, so the only failures here are missing capability and IR contract violations
 * — "cannot execute" is known **before a single weight block is fetched**.
 *
 * Then {@link PreparedModel.estimate} gives the bytes required and
 * {@link PreparedModel.createContainerSession} gives the Session.
 */
export const prepareContainer = (opened: BoundContainer, graphName: string): PreparedModel =>
  PreparedModel.fromContainer(opened, graphName);

/**
 * Creates a Session from one graph of an opened container (a thin composition of
 * {@link prepareContainer} + {@link PreparedModel.createContainerSession}). Blocks are fetched one
 * at a time (a block's sha256 is verified only when the source is unverified — container-v1 §7).
 */
export const createSessionFromContainer = async (
  gpu: GpuContext,
  opened: BoundContainer,
  graphName: string,
  options: SessionOptions = {},
): Promise<Session> =>
  await prepareContainer(opened, graphName).createContainerSession(gpu, options);
