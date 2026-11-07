/**
 * Execution recipes — the intermediate representation of the encode layer and its generic execution
 * loop.
 *
 * Structure: "**derivation** (the executor lowers the step list → the recipe list) → **planning**
 * ({@link planRecipes} decides the regions and offsets of intermediate buffers — ADR 0093) →
 * **execution** (builds bind groups and queues dispatches)". The derivation phase issues not a single
 * GPU command and touches no run-lifetime object — it folds in only Session-resident objects (weights,
 * per-channel scales, params) as direct references.
 *
 * MUST: a recipe holds no `GPUBindGroup`. The plan hands out the binding targets (node outputs,
 * temporaries) to region offsets, so all a recipe can hold is the **procedure** "what is bound at
 * which position". The baked bind groups ({@link bakeBindGroups}) are held by the **backing side**;
 * the recipe itself stays independent of the objects and is shared by several backings.
 * MUST: the lifetime of a temporary buffer is expressed by dispatch boundaries ({@link TempRecipe}).
 * Flattening it to allocate-all / release-all changes the plan's simultaneous liveness for shapes
 * with nested live intervals (i8a8 attention).
 *
 * The execution phase is **two paths sharing one plan**. A miss run allocates run-lifetime regions
 * and builds the bind groups of every dispatch with {@link bakeAllBindGroups}; a hit run uses the
 * groups baked by {@link bakeBindGroups} into Session-resident regions (backing). Both only queue the
 * dispatches of baked execution (run: {@link executeBakedPlanPaced}, enqueue:
 * {@link executeBakedPlan}); allocation, reference counting and release are gone from the execution
 * phase (the plan has replayed them). Bindings are cut to the exact size with
 * `{ buffer: region, offset, size }` (ADR 0093 decision 2).
 *
 * MUST: **the unit of baking is split by the owner of the binding targets** (ADR 0066 decision 5). A
 * dispatch that binds only Session-owned objects (slots / resident inputs / weights) is baked once by
 * {@link bakeBindGroups} when the backing is built; a dispatch that binds context-owned objects (state
 * slots, the logical-length uniform) is baked **per GenerationContext** by
 * {@link bakeGenerationBindGroups}. Without the split it is a choice between "running with the
 * previous context's KV still bound", a silent stale read, and thrashing that rebuilds everything on
 * every context switch. The discriminator is the single {@link bindsGeneration}, and the two bakes
 * fill each other's holes (`undefined`) ({@link BakedGroups}).
 */

import type { StorageRoles } from "../gpu/pipeline-cache.ts";
import type { SubmitScheduler } from "../gpu/submit.ts";
import { ExecutionError } from "./plan.ts";
import {
  planTransients,
  type TransientLimits,
  type TransientPlan,
  type TransientProgram,
  type TransientRef,
} from "./transient-plan.ts";

/**
 * A source that does not depend on the run lifetime (a direct reference to a Session-resident object,
 * or a value name). It can also be the origin of an alias.
 * `resident` is a weight / per-channel scale / params — it may be resolved down to the object in the
 * derivation phase.
 */
export type ValueSource =
  | { readonly kind: "resident"; readonly buffer: GPUBuffer }
  | { readonly kind: "value"; readonly name: string };

/** A reference to a temporary region closed within the same step (`id` indexes {@link StepRecipe.temps}). */
export type TempSource = { readonly kind: "temp"; readonly id: number };

/**
 * Objects owned by the {@link GenerationContext} (ADR 0066 decision 5 — separation of ownership).
 *
 * `state` is a named slot and `lengths` is the logical-length uniform (the carrier of the 2026-08-17
 * addendum, item 4); **neither is owned by the Session**. A recipe holds only the name and kind, and the objects are
 * handed out per run by {@link GenerationEncoding} — this is the object-side basis of "a recipe does
 * not know the context's identity" (the context is not part of the plan key).
 */
export type GenerationSource =
  | { readonly kind: "state"; readonly name: string }
  | { readonly kind: "lengths" };

/** The source of the object one bind group slot points at. */
export type BindingSource = ValueSource | TempSource | GenerationSource;

/** One bind group entry. MUST: the binding number matches the kernel's declaration. */
export type BindingRecipe = {
  readonly binding: number;
  readonly source: BindingSource;
};

/**
 * The workgroup count of a dispatch. Either a **static triple** (every existing builder) or a **pure
 * function computing it from the logical lengths** (states form — ADR 0067 decision 4, "the host
 * computes dispatch counts from the logical lengths").
 *
 * MUST: held as a discriminated union (a dispatch never carries both a static value and a function).
 * A form that can hold both splits "which one is the source of truth" per dispatch, and an
 * implementation where only one was fixed runs a different amount of work without an exception.
 */
export type DispatchWorkgroups =
  | readonly [number, number, number]
  | ((past: number, query: number) => readonly [number, number, number]);

/**
 * The recipe of one dispatch.
 *
 * MUST: `params` is **binding 0 for every kernel** (bind groups have this form without exception), so
 * it is kept apart from the list of {@link BindingRecipe}.
 */
type DispatchRecipe = {
  /** Pipeline key (identical to the diagnostic label). */
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  /** The params at binding 0 (uniform or storage). Session-resident (content-addressed cache). */
  readonly params: GPUBuffer;
  readonly bindings: readonly BindingRecipe[];
  readonly workgroups: DispatchWorkgroups;
  /**
   * The role of each binding number (read / read-write), taken from the kernel WGSL's storage
   * declarations. The material the plan uses to keep "a value read and a value written in the same
   * dispatch cannot share a buffer" ({@link StorageRoles} in src/gpu/pipeline-cache.ts).
   */
  readonly roles: StorageRoles;
};

/**
 * The allocation spec of a node temporary. **Its lifetime is expressed by dispatch-boundary
 * indices** (the MUST of the module doc).
 *
 * In the plan it becomes alive right before `allocBefore`, and when several share a boundary they
 * end **in reverse allocation order** (the same order each `#build*` returns them).
 */
export type TempRecipe = {
  readonly byteLength: number;
  /** Allocated right **before** this index of {@link StepRecipe.dispatches}. */
  readonly allocBefore: number;
  /** Released right **after** this index of {@link StepRecipe.dispatches}. */
  readonly releaseAfter: number;
};

/**
 * The allocation spec of a step output. `alias` makes the input's object the output's object as is
 * (reshape and identity expand — ADR 0011), and issues neither allocation nor dispatch. Combined with
 * the bookkeeping, {@link StepOutput} is the outward form (this discriminated union alone has no
 * consumer).
 */
type OutputRecipe =
  | { readonly kind: "alloc"; readonly byteLength: number }
  | { readonly kind: "alias"; readonly source: ValueSource };

/** The allocation spec and bookkeeping of one step output (an element of {@link StepRecipe.outputs}). */
export type StepOutput = OutputRecipe & {
  readonly name: string;
  /** Future consumption count of the output value (the `uses` of `retain`). */
  readonly uses: number;
  /** Graph output (its lifetime extends to the end so it stays readable for readback). */
  readonly pinned: boolean;
};

/** The recipe of one execution step (a plain node or a fused step). */
export type StepRecipe = {
  /**
   * Allocation specs and bookkeeping in ascending output slot (ADR 0068 decision 1 — for a plain node
   * the same order and length as `node.outs`; a fused step always has 1).
   *
   * MUST: **allocate → retain → register in env → release the definition's own count, in ascending
   * order of this list**. Four places share this order — the plan replay
   * (src/runtime/transient-plan.ts), the allocation program ({@link buildTransientProgram}), baking
   * (one implementation, `bakeGroups` — entries {@link bakeBindGroups} /
   * {@link bakeGenerationBindGroups}) and the list assembly (`#buildStep` in recipe-builder). If the
   * order splits in even one place, no exception is raised; only the reassignment partners (= the
   * pairs of values holding the same bytes, the alias decisions) differ between baking and the plan.
   */
  readonly outputs: readonly StepOutput[];
  readonly temps: readonly TempRecipe[];
  readonly dispatches: readonly DispatchRecipe[];
  /** The **release list** of inputs released at the end of the step (same order as the current bookkeeping). */
  readonly releases: readonly string[];
  /**
   * Whether this step **writes to a state slot** (contains `state_append` — ADR 0067 decision 5).
   *
   * MUST: the only discriminator for the poison decision on failure (ADR 0066 addendum of 2026-08-17,
   * item 3). The physical ring can be overwritten only by a run that submitted a writing dispatch, so
   * the executor uses this flag to decide "where to snapshot the submit counter". States-form attention that only
   * reads is false.
   */
  readonly writesState: boolean;
};

/** The mutable temporary spec under construction (the mutable form of {@link TempRecipe}). */
type MutableTemp = {
  readonly byteLength: number;
  readonly allocBefore: number;
  releaseAfter: number;
};

/**
 * The builder of one step's recipe. It derives the temporaries' lifetimes from **the position where
 * dispatches were queued**, so each `#build*` only needs to call
 * {@link StepRecipeBuilder.allocTemp} / {@link StepRecipeBuilder.releaseTemp} where a temporary is
 * needed and where it has been read for the last time.
 */
export class StepRecipeBuilder {
  readonly #temps: MutableTemp[] = [];
  readonly #dispatches: DispatchRecipe[] = [];

  /** Declares one node temporary (in the plan, one live interval bounded by dispatch boundaries). */
  allocTemp(byteLength: number): TempSource {
    const id = this.#temps.length;
    // releaseTemp fixes the release position. A declaration with no release is caught by the
    // declaration's entry point (validateStepRecipe) (it does not leak silently).
    this.#temps.push({ byteLength, allocBefore: this.#dispatches.length, releaseAfter: -1 });
    return { kind: "temp", id };
  }

  /** Fixes the temporary's release position to "right after the last dispatch queued so far". */
  releaseTemp(temp: TempSource): void {
    this.#temps[temp.id].releaseAfter = this.#dispatches.length - 1;
  }

  dispatch(recipe: DispatchRecipe): void {
    this.#dispatches.push(recipe);
  }

  get temps(): readonly TempRecipe[] {
    return this.#temps;
  }

  get dispatches(): readonly DispatchRecipe[] {
    return this.#dispatches;
  }
}

/**
 * Checks that the **declaration** of an assembled step recipe closes within its own dispatch list (a
 * static check that does not touch the GPU).
 *
 * MUST: a temporary's lifetime is `0 ≤ allocBefore ≤ releaseAfter < dispatch count`. A declaration
 * outside this turns, in the execution phase, into "binding an unallocated temporary" (`temps[id]`
 * reaches createBindGroup as `undefined` and becomes a validation error far from the real cause) or
 * "a temporary that is never released" (unnoticed until the plan's closure check).
 * MUST: lifetime boundaries and the temporary indices a binding points at are integers
 * (`Number.isSafeInteger`). A fraction appears to be inside the interval, and NaN makes every
 * comparison false, so it slips past comparison-only checks.
 * MUST: a temporary index a binding points at is within the declared range. Out of range is the same
 * `undefined` binding as above, and "which dispatch failed to read which temporary" vanishes from the
 * diagnostic.
 * MUST: the index of a dispatch that binds a temporary is inside that temporary's
 * `[allocBefore, releaseAfter]`. The plan reassigns bytes outside the interval to other temporaries,
 * so binding outside it reads, without an exception, the values written by the owner before or after
 * (a silently wrong value).
 * MUST: the caller is **the single entry point for recipe declarations** (the return of `#buildStep`
 * in recipe-builder). Both plain nodes and fusion replays pass there, so no hand-written check per
 * path is needed (the fusion rule's declaration itself is caught at its entry by
 * `assertTempLifetimes` in src/runtime/fusion-rule.ts; this checks **the result of the replay** with
 * the same strength).
 */
export const validateStepRecipe = (recipe: StepRecipe): void => {
  const where = `step '${recipe.outputs.map((output) => output.name).join(", ")}'`;
  const dispatchCount = recipe.dispatches.length;
  recipe.temps.forEach((temp, id) => {
    if (
      !Number.isSafeInteger(temp.byteLength) ||
      temp.byteLength < 1 ||
      !Number.isSafeInteger(temp.allocBefore) ||
      !Number.isSafeInteger(temp.releaseAfter) ||
      temp.allocBefore < 0 ||
      temp.releaseAfter < temp.allocBefore ||
      temp.releaseAfter >= dispatchCount
    ) {
      throw new ExecutionError(
        `${where}: temporary ${id} has a declared lifetime [${temp.allocBefore}, ${temp.releaseAfter}] that` +
          ` does not close within the ${dispatchCount} dispatches (${temp.byteLength}B)`,
      );
    }
  });
  recipe.dispatches.forEach((dispatch, index) => {
    for (const entry of dispatch.bindings) {
      const { source } = entry;
      if (source.kind !== "temp") continue;
      // A non-integer index fails regardless of the declared count, so name the cause with a text
      // separate from out-of-range.
      if (!Number.isSafeInteger(source.id)) {
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' binding ${entry.binding} has a temporary index ` +
            `${source.id} that is not an integer`,
        );
      }
      if (source.id < 0 || source.id >= recipe.temps.length) {
        throw new ExecutionError(
          `${where}: dispatch '${dispatch.key}' binding ${entry.binding} points to temporary ${source.id},` +
            ` but ${recipe.temps.length} are declared`,
        );
      }
      const temp = recipe.temps[source.id];
      if (index < temp.allocBefore || index > temp.releaseAfter) {
        throw new ExecutionError(
          `${where}: dispatch ${index} '${dispatch.key}' binding ${entry.binding} references temporary ` +
            `${source.id} outside its live range [${temp.allocBefore}, ${temp.releaseAfter}]`,
        );
      }
    }
  });
};

/**
 * The context-side objects and logical lengths of one generation run (ADR 0066 decision 5 / addendum
 * of 2026-08-17, item 4).
 *
 * MUST: the logical lengths are **the values at encode time** (the same pair written into the
 * `lengths` uniform). Dispatch counts are computed from them ({@link DispatchWorkgroups}), so a
 * mismatch with the uniform shifts "the range the GPU scans" against "the workgroup count the host
 * dispatched", and part of S is read without having been written.
 */
export type GenerationEncoding = {
  /** Slot name → context-owned buffer. */
  readonly slots: ReadonlyMap<string, GPUBuffer>;
  /** The logical-length uniform (16 bytes: `{past, query}` and the bidirectional block — ADR 0144). */
  readonly lengths: GPUBuffer;
  readonly past: number;
  readonly query: number;
};

/**
 * What a generation run accepts **with respect to the logical lengths** (collected from state nodes
 * by the derivation phase — the implementation is in src/runtime/recipe-builder.ts). For a graph
 * without state nodes both are empty.
 *
 * MUST: held with the same lifetime as the prepared plan (the executor's `PreparedPlan`). A hit run
 * runs neither planning nor derivation, so unless this is folded together with the recipe list, "the
 * check disappears only from the second run on".
 */
export type GenerationLimits = {
  /**
   * The physical chunk rows `M` of state nodes (equal values folded into a set). `queryLength` is
   * allowed up to the **minimum** of these — rows beyond it are not in the declared shape (the
   * fixed-length chunk + pad of ADR 0066 decision 4).
   */
  readonly chunkRows: ReadonlySet<number>;
  /**
   * Name of a full slot (no attrs `window` declared) → row capacity `C`. Sliding slots are rings and
   * are excluded (ADR 0067 decision 4 ④ — "for full slots, `pastLength + queryLength ≤ C` at run
   * time").
   */
  readonly fullCapacities: ReadonlyMap<string, number>;
};

/**
 * Checks that the physical chunk rows `M` of state nodes are in the set the context allows
 * (`allowedRows` — produced by `allowedChunkRows` in src/runtime/generation-context.ts) (a pure
 * function independent of the GPU).
 *
 * MUST: the pre-run check ({@link assertGenerationRun}) and the estimate (estimate.ts) go through
 * this same function. If the estimate had its own decision, the estimate alone would return normal
 * values for a shape the run always rejects on its first call (such as an `M` fixed as a number that
 * is not in the set).
 */
export const assertChunkRowsAllowed = (
  chunkRows: Iterable<number>,
  allowedRows: ReadonlySet<number>,
): void => {
  for (const rows of chunkRows) {
    if (allowedRows.has(rows)) continue;
    const prefill = [...allowedRows].filter((allowed) => allowed !== 1);
    throw new ExecutionError(
      `the physical chunk row count ${rows} of the state node does not match the fixed chunk contract` +
        ` (only the prefill form's {${prefill.join(", ")}} or the decode form's 1 is allowed — ` +
        "ADR 0066 decision 4 / addendum 10, buckets)",
    );
  }
};

/**
 * The **pre-encode** check of a generation run (ADR 0066 decision 4 / ADR 0067 decision 4 ④).
 *
 * MUST: fail before queuing a single dispatch. `queryLength > M` reads and writes outside the
 * declared shape of q / ins, and for a full slot with `pastLength + queryLength > C` the write of
 * `state_append` falls out of range (discarded by robustness = **silently not written**) — neither
 * raises an exception or a warning, and the next step's reader consumes the debris as past KV.
 * MUST: **the physical chunk rows `M` only within the set the context allows (`allowedRows`)** —
 * the run-time enforcement of `{1}` (decode shape) ∪ `chunkBuckets` ∪ `{chunkLength}` (prefill
 * shapes; ADR 0066 decision 4's "two execution shapes" widened by the addendum item on buckets (item
 * 10 of the 2026-08-17 addendum) to "as many as declared"). Letting an arbitrary `M` through adds plans with distinct keys per kind of
 * `M`, pollutes the PreparedPlan LRU, and the decode hot path silently falls back to re-derivation —
 * the set is a plan-time constant of the context to cap the number of plans at the declared count.
 * This one place is enough for the decision.
 * MUST: `queryLength ≤ chunkLength` and the u32 range are not checked again here (`writeLengths` /
 * `advance` of `GenerationContext` own them — no double bookkeeping). With `M = 1`, `Q ≤ M` implies
 * `Q = 1`, so the decode shape needs no separate `Q` check either.
 */
export const assertGenerationRun = (
  limits: GenerationLimits,
  allowedRows: ReadonlySet<number>,
  pastLength: number,
  queryLength: number,
): void => {
  assertChunkRowsAllowed(limits.chunkRows, allowedRows);
  for (const rows of limits.chunkRows) {
    if (queryLength > rows) {
      throw new ExecutionError(
        `queryLength ${queryLength} exceeds the physical chunk row count ${rows} of the state node` +
          " (prefill is a fixed-length chunk + pad, decode is 1 — ADR 0066 decision 4)",
      );
    }
  }
  for (const [name, capacity] of limits.fullCapacities) {
    if (pastLength + queryLength > capacity) {
      throw new ExecutionError(
        `state '${name}': pastLength ${pastLength} + queryLength ${queryLength} exceeds the full slot's` +
          ` capacity ${capacity} (ADR 0067 decision 4 ④ — a slot that is not sliding does not wrap)`,
      );
    }
  }
};

/**
 * One binding target. A slot handed out by the plan is the `(offset, size)` range of a region buffer;
 * anything else (graph inputs, resident objects) is the whole buffer.
 *
 * `readable` is the eligibility for readback — true only for values that are not slots (inputs /
 * weights) and for pinned slots. Other slots are reassigned to other values during the run, and the
 * next run finds the previous run's debris there (the rule that rejects readback of intermediates —
 * the same as the former `RunArena.isReadable`).
 */
export type ValueBinding = {
  readonly buffer: GPUBuffer;
  readonly offset?: number;
  readonly size?: number;
  readonly readable: boolean;
};

/** Lowers a {@link ValueBinding} to a bind group resource (without keys holding undefined). */
const toResource = (binding: ValueBinding): GPUBufferBinding =>
  binding.offset === undefined
    ? { buffer: binding.buffer }
    : { buffer: binding.buffer, offset: binding.offset, size: binding.size };

const resolveValue = (
  source: ValueSource,
  env: ReadonlyMap<string, ValueBinding>,
): ValueBinding => {
  if (source.kind === "resident") return { buffer: source.buffer, readable: true };
  const binding = env.get(source.name);
  if (binding === undefined) throw new ExecutionError(`no buffer for value '${source.name}'`);
  return binding;
};

/**
 * Looks up a context-owned object.
 *
 * MUST: binding state / logical lengths in an execution without generation information fails loudly
 * (a breakwater that is not reached because the planning layer fails first — letting it through
 * passes `undefined` to the bind group, which fails validation with a diagnostic far from the real
 * cause).
 */
const resolveGeneration = (
  source: GenerationSource,
  generation: GenerationEncoding | undefined,
): GPUBuffer => {
  if (generation === undefined) {
    throw new ExecutionError(
      `${
        source.kind === "lengths" ? "logical-length uniform" : `state slot '${source.name}'`
      }: tried to run a dispatch that binds it without a GenerationContext`,
    );
  }
  if (source.kind === "lengths") return generation.lengths;
  const buffer = generation.slots.get(source.name);
  if (buffer === undefined) {
    throw new ExecutionError(`state slot '${source.name}' has no buffer in the GenerationContext`);
  }
  return buffer;
};

const resolveBinding = (
  source: BindingSource,
  env: ReadonlyMap<string, ValueBinding>,
  temps: readonly ValueBinding[],
  generation: GenerationEncoding | undefined,
): GPUBufferBinding => {
  switch (source.kind) {
    case "temp":
      return toResource(temps[source.id]);
    case "state":
    case "lengths":
      return { buffer: resolveGeneration(source, generation) };
    default:
      return toResource(resolveValue(source, env));
  }
};

/**
 * Lowers the form computed from the logical lengths ({@link DispatchWorkgroups}) to actual numbers.
 *
 * MUST: compute **at every encode** (the logical lengths change every run). Folding the result here
 * and carrying it around breaks "a recipe is a pure function of the bindings" and runs with the
 * previous step's workgroup count.
 */
const resolveWorkgroups = (
  recipe: DispatchRecipe,
  generation: GenerationEncoding | undefined,
): readonly [number, number, number] => {
  if (typeof recipe.workgroups !== "function") return recipe.workgroups;
  if (generation === undefined) {
    throw new ExecutionError(
      `dispatch '${recipe.key}': tried to run a dispatch whose workgroup count is derived from the logical length ` +
        "without a GenerationContext",
    );
  }
  return recipe.workgroups(generation.past, generation.query);
};

/**
 * Queues one dispatch. **It is skipped only when a workgroup count computed from the logical lengths
 * has an axis of 0.**
 *
 * A 0 in a dynamic shape means "this row block is entirely pad rows = no valid row in this step"
 * (states form — `stateEffectiveRows` in src/kernels/state-attention.ts). For WebGPU a 0 is a legal
 * no-op too, but queuing it lines up zero-work dispatches in the diagnostics (GPU time breakdown,
 * submit count) and blurs the observation that "work is proportional to queryLength".
 *
 * MUST: **a static triple is queued even when it is 0** (the existing contract — the path for
 * zero-element tensors goes through without "silently skipping" zero-element dispatches, and
 * tests/e2e_public_api_test.ts pins that dispatch count). Making the skip condition "the value is 0"
 * would drag along the behavior of non-generation paths.
 * MUST: miss runs and hit runs share **this one function**. Skipping on one side only would make the
 * same recipe queue different dispatch lists per path.
 */
const dispatchWithWork = (
  scheduler: SubmitScheduler,
  recipe: DispatchRecipe,
  bindGroup: GPUBindGroup,
  workgroups: readonly [number, number, number],
): void => {
  const derived = typeof recipe.workgroups === "function";
  if (derived && (workgroups[0] === 0 || workgroups[1] === 0 || workgroups[2] === 0)) return;
  scheduler.dispatch(recipe.pipeline, bindGroup, workgroups, recipe.key);
};

/**
 * Whether this dispatch binds objects owned by the {@link GenerationContext} — **the single
 * discriminator of the baking unit** (ADR 0066 decision 5).
 *
 * MUST: this is the only discrimination point. If the Session side ({@link bakeBindGroups}) and the
 * context side ({@link bakeGenerationBindGroups}) had separate conditions, they would produce
 * dispatches baked by neither (fail loudly at run time = still safe) or dispatches **baked by both**
 * (the Session-side group holds context-owned objects across runs and keeps reading the previous
 * context's KV after a switch — a silent stale read).
 * NOTE: dynamic workgroups (the function form of {@link DispatchWorkgroups}) are **not** part of the
 * discrimination — that axis is orthogonal to the bound objects, and the dispatch count is resolved
 * from the logical lengths at every encode whichever side baked the group ({@link resolveWorkgroups}).
 */
const bindsGeneration = (dispatch: DispatchRecipe): boolean =>
  dispatch.bindings.some(
    (entry) => entry.source.kind === "state" || entry.source.kind === "lengths",
  );

/**
 * Builds the bind group of one dispatch.
 *
 * MUST: the miss run ({@link bakeAllBindGroups}) and baking ({@link bakeBindGroups}) share **this one
 * function**. If the entry layout split into two implementations, a shift of binding numbers or of
 * the params position on one side would still pass validation (it only has to satisfy the layout),
 * and only the values would change silently.
 */
const createBindGroup = (
  device: GPUDevice,
  recipe: DispatchRecipe,
  resolve: (source: BindingSource) => GPUBufferBinding,
): GPUBindGroup =>
  device.createBindGroup({
    layout: recipe.layout,
    entries: [
      { binding: 0, resource: { buffer: recipe.params } },
      ...recipe.bindings.map((entry) => ({
        binding: entry.binding,
        resource: resolve(entry.source),
      })),
    ],
  });

/**
 * Builds the allocation program of intermediate buffers from the recipe list (the input of
 * {@link planTransients}).
 *
 * MUST: a pure function (creates no GPU resources and does not modify the recipes).
 * MUST: reads / writes per dispatch come from {@link DispatchRecipe.roles} (the WGSL declarations).
 * A binding of a value name / temporary without a role fails loudly — filling it by guess makes only
 * the wrong dispatch fail usage-scope validation, and the output stays all 0 without an exception.
 * NOTE: state slots and the logical-length uniform **do not appear** in the plan (ADR 0066 decision
 * 5 — their owner is the GenerationContext). Neither do resident objects (weights / scales).
 */
export const buildTransientProgram = (recipes: readonly StepRecipe[]): TransientProgram =>
  recipes.map((recipe) => ({
    outputs: recipe.outputs.map((output) => ({
      name: output.name,
      kind: output.kind,
      byteLength: output.kind === "alloc" ? output.byteLength : 0,
      source:
        output.kind === "alias" && output.source.kind === "value" ? output.source.name : undefined,
      uses: output.uses,
      pinned: output.pinned,
    })),
    temps: recipe.temps,
    dispatches: recipe.dispatches.map((dispatch) => {
      const reads: TransientRef[] = [];
      const writes: TransientRef[] = [];
      for (const entry of dispatch.bindings) {
        const { source } = entry;
        if (source.kind !== "value" && source.kind !== "temp") continue;
        const ref: TransientRef =
          source.kind === "value"
            ? { kind: "value", name: source.name }
            : { kind: "temp", id: source.id };
        if (dispatch.roles.writes.has(entry.binding)) writes.push(ref);
        else if (dispatch.roles.reads.has(entry.binding)) reads.push(ref);
        else {
          throw new ExecutionError(
            `dispatch '${dispatch.key}' binding ${entry.binding} has no WGSL storage declaration` +
              " (the read / read-write role cannot be taken from it)",
          );
        }
      }
      return { reads, writes };
    }),
    releases: recipe.releases,
  }));

/** The intermediate-buffer plan of a recipe list ({@link buildTransientProgram} → {@link planTransients}). */
export const planRecipes = (
  recipes: readonly StepRecipe[],
  limits: TransientLimits,
): TransientPlan => planTransients(buildTransientProgram(recipes), limits);

/** Slot index → binding on a region. A failed lookup is a mismatch between plan and recipe list (a broken internal invariant). */
const resolveSlot = (
  slot: number | undefined,
  plan: TransientPlan,
  regions: readonly GPUBuffer[],
): ValueBinding => {
  const placed = slot === undefined ? undefined : plan.slots[slot];
  const buffer = placed === undefined ? undefined : regions[placed.region];
  if (slot === undefined || placed === undefined || buffer === undefined) {
    throw new ExecutionError(`no region buffer for slot ${slot}`);
  }
  return {
    buffer,
    offset: placed.offset,
    size: placed.byteLength,
    readable: plan.pinned.has(slot),
  };
};

/**
 * The table of baked bind groups. The outer level has the same order and length as the
 * {@link StepRecipe} list, the inner level as {@link StepRecipe.dispatches}.
 *
 * `undefined` marks "this position is baked by **the other owner**" (the separation of the baking
 * unit in ADR 0066 decision 5). The Session-side table and the context-side table are complementary
 * through {@link bindsGeneration}, and execution ({@link executeBakedPlan}) fills them in position by
 * position.
 */
export type BakedGroups = readonly (readonly (GPUBindGroup | undefined)[])[];

/** The objects baking needs (both bakes receive the same bundle). */
export type BakeContext = {
  readonly device: GPUDevice;
  /** The intermediate-buffer plan ({@link planRecipes}). */
  readonly plan: TransientPlan;
  /** Region index → region buffer (same order and length as {@link TransientPlan.regions}). */
  readonly regions: readonly GPUBuffer[];
  /** Graph input name → object (resident buffer owned by the backing / run-lifetime input buffer / resident input). */
  readonly inputs: ReadonlyMap<string, GPUBuffer>;
};

/** The result of baking ({@link bakeBindGroups} / {@link bakeAllBindGroups}). */
export type BakedPlan = {
  /** The bind groups of the selected dispatches (the rest are `undefined` holes). */
  readonly groups: BakedGroups;
  /**
   * Value name → binding target once every step has been expanded. The readback targets of graph
   * outputs are looked up here (only values whose `readable` is true can be read).
   */
  readonly values: ReadonlyMap<string, ValueBinding>;
};

/**
 * **The only implementation** of baking (the Session side, the context side and the miss run all go
 * through here). Builds bind groups only for dispatches where `select` is true and returns the rest
 * as `undefined` holes.
 *
 * MUST: the value name → binding target mapping is expanded **in the same order** as the plan replay
 * ({@link planTransients}) — if the order "put the outputs in env first, then resolve the step's
 * dispatches" breaks, exactly the steps that also bind their output as their own input grab a
 * different object. With several outputs the order goes down to **ascending output slot** (the
 * ordering rule of {@link StepRecipe.outputs}).
 * MUST: do not split how baking walks. If only the context side had its own implementation, the same
 * binding of the same dispatch could point at different objects in "the group baked by the Session
 * side" and "the group baked by the context side" (both satisfy the layout, so validation passes and
 * only the values change silently).
 */
const bakeGroups = (
  recipes: readonly StepRecipe[],
  context: BakeContext,
  select: (dispatch: DispatchRecipe) => boolean,
  generation: GenerationEncoding | undefined,
): BakedPlan => {
  const { device, plan, regions } = context;
  const values = new Map<string, ValueBinding>(
    [...context.inputs].map(([name, buffer]) => [name, { buffer, readable: true }]),
  );
  const groups: (readonly (GPUBindGroup | undefined)[])[] = [];
  recipes.forEach((recipe, index) => {
    const step = plan.steps[index];
    recipe.outputs.forEach((output, slot) => {
      values.set(
        output.name,
        output.kind === "alias"
          ? resolveValue(output.source, values)
          : resolveSlot(step.outputs[slot], plan, regions),
      );
    });
    const temps = step.temps.map((slot) => resolveSlot(slot, plan, regions));
    groups.push(
      recipe.dispatches.map((dispatch) =>
        select(dispatch)
          ? createBindGroup(device, dispatch, (source) =>
              resolveBinding(source, values, temps, generation),
            )
          : undefined,
      ),
    );
  });
  return { groups, values };
};

/**
 * Bakes the bind groups of a slot backing (once, at construction).
 *
 * They can be baked because the binding targets are fixed across runs: params / weights /
 * per-channel scales are `resident` direct references, node outputs and temporaries are ranges of
 * resident regions handed out by the plan, and graph inputs are resident buffers owned by the backing
 * (`inputs`). What changes per run is **only the contents of the input buffers**; not a single object
 * the bind groups point at moves.
 *
 * MUST: dispatches that bind context-owned objects are **not baked** (ADR 0066 decision 5). The
 * backing is Session-owned (several are kept per shape within the budget — ADR 0095), and which
 * GenerationContext it pairs with changes every run, so baking here would keep binding the previous
 * context's KV after a switch. Those positions are filled by {@link bakeGenerationBindGroups}.
 * MUST: call it only inside a run's errorScope interval (a validation failure of createBindGroup does
 * not become an exception).
 */
export const bakeBindGroups = (recipes: readonly StepRecipe[], context: BakeContext): BakedPlan =>
  bakeGroups(recipes, context, (dispatch) => !bindsGeneration(dispatch), undefined);

/**
 * Builds the bind groups of a miss run (**every dispatch**, against run-lifetime regions). A run with
 * generation binds that run's {@link GenerationEncoding} directly (it is run-lifetime, so no separate
 * baking is needed).
 *
 * MUST: call it only inside a run's errorScope interval (same reason as {@link bakeBindGroups}).
 */
export const bakeAllBindGroups = (
  recipes: readonly StepRecipe[],
  context: BakeContext,
  generation: GenerationEncoding | undefined,
): BakedPlan => bakeGroups(recipes, context, () => true, generation);

/**
 * Bakes the **context-side** bind groups of a generation run (the separate baking of ADR 0066
 * decision 5).
 *
 * The binding targets are of two kinds: backing-owned objects (slots / inputs / weights — reproduced
 * through **the same value resolution** as {@link bakeBindGroups}) and context-owned objects (state
 * slots, the logical-length uniform). So the lifetime of this bundle is **the pair (backing,
 * context)**, and replacing either requires re-baking (the backing side is detected by a generation
 * token — the executor's `#generationGroups`).
 *
 * MUST: call it only inside a run's errorScope interval (same reason as {@link bakeBindGroups}).
 */
export const bakeGenerationBindGroups = (
  recipes: readonly StepRecipe[],
  context: BakeContext & { readonly generation: GenerationEncoding },
): BakedGroups => bakeGroups(recipes, context, bindsGeneration, context.generation).groups;

/**
 * The baked-execution surface of a generation run (the 4th argument of {@link executeBakedPlan}).
 */
export type BakedGeneration = {
  /** The bind groups baked on the context side (they fill the Session side's `undefined` holes). */
  readonly groups: BakedGroups;
  /** The logical lengths and context-owned objects (dispatch counts are computed from these values). */
  readonly encoding: GenerationEncoding;
  /**
   * Called **before** queuing a step's dispatches (the snapshot point of the poison decision — ADR
   * 0066 addendum of 2026-08-17, item 3).
   *
   * MUST: called at the same position as in a miss run, "right before each step".
   * Shifting it makes the same failure of the same graph poison or not depending on the path (miss
   * run / hit run).
   */
  onStep(recipe: StepRecipe): void;
};

/**
 * Counts the **planned** dispatches of a recipe list per pipeline key (in key order — code-unit
 * comparison, locale-independent). The implementation of `SessionDiagnostics.lastRunPipelines`.
 *
 * MUST: the source is the run's recipe list itself (no counter incremented and decremented on its
 * own). The Session-cumulative set of used keys (src/gpu/pipeline-cache.ts) is not updated by runs
 * that hit a prepared plan, so it cannot be the source of "what this run queued".
 * NOTE: dispatches whose workgroup count is computed from the logical lengths (states form) are not
 * actually queued in zero-work steps ({@link dispatchWithWork}), but this still counts them as one —
 * it is a census of the plan, not of issuance. The argument is a structural type with only
 * `dispatches[].key` because counting reads nothing else (tests can build the list without GPU
 * objects).
 */
export const pipelineCensus = (
  recipes: readonly { readonly dispatches: readonly { readonly key: string }[] }[],
): readonly { readonly key: string; readonly dispatchCount: number }[] => {
  const counts = new Map<string, number>();
  for (const recipe of recipes) {
    for (const { key } of recipe.dispatches) counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([key, dispatchCount]) => ({ key, dispatchCount }));
};

/**
 * Executes the baked bind groups (the enqueue path — run uses {@link executeBakedPlanPaced}). The
 * only GPU operations issued are these dispatches; neither allocation, release nor createBindGroup
 * (a miss run passes the groups built by {@link bakeAllBindGroups}).
 *
 * MUST: being correct even with the previous run's debris left in the regions rests on full-write
 * (ADR 0014 — every node writes every byte of its binding range), the same single invariant as the
 * safety of reassignment.
 * MUST: a position that stays unfilled fails loudly (a dispatch binding state reached without the
 * generation surface). Letting it through passes `undefined` to `setBindGroup`, a diagnostic far from
 * the real cause.
 */
export const executeBakedPlan = (
  recipes: readonly StepRecipe[],
  groups: BakedGroups,
  scheduler: SubmitScheduler,
  generation?: BakedGeneration,
): void => {
  recipes.forEach((recipe, index) =>
    dispatchBakedStep(recipe, index, groups, scheduler, generation),
  );
};

/**
 * Queues the same dispatch list as {@link executeBakedPlan} and waits for the in-flight submit bound
 * between steps (`SubmitScheduler.backpressure` — ADR 0123). The content and order of the queued
 * commands are the same; only the waiting points are added.
 *
 * Returns a Promise only when a wait was needed (undefined means every step was queued
 * synchronously). MUST: a run that does not reach the bound gets not a single extra microtask — as
 * an async function the caller's await would yield even with no wait, letting other tasks run while
 * the last step is still unsubmitted (and losing the synchronicity of the section under the
 * errorScope).
 *
 * MUST: called only from run (inside the errorScope interval lock). The lock keeps other runs and
 * batches out, so nobody else's dispatches cut in during the wait. MUST NOT: called from enqueue (use
 * the synchronous {@link executeBakedPlan} and do not wait). Enqueues of other Sessions in the same
 * batch are not serialised by the interval lock, so waiting would let an enqueue called later reach
 * the queue first (ADR 0123 decision 4).
 */
export const executeBakedPlanPaced = (
  recipes: readonly StepRecipe[],
  groups: BakedGroups,
  scheduler: SubmitScheduler,
  generation?: BakedGeneration,
): Promise<void> | undefined => {
  const continueFrom = (start: number): Promise<void> | undefined => {
    for (let index = start; index < recipes.length; index += 1) {
      dispatchBakedStep(recipes[index], index, groups, scheduler, generation);
      const wait = scheduler.backpressure();
      if (wait !== undefined) return wait.then(() => continueFrom(index + 1));
    }
    return undefined;
  };
  return continueFrom(0);
};

/** Queues one step's dispatches (the single function shared by {@link executeBakedPlan} and {@link executeBakedPlanPaced}). */
const dispatchBakedStep = (
  recipe: StepRecipe,
  index: number,
  groups: BakedGroups,
  scheduler: SubmitScheduler,
  generation: BakedGeneration | undefined,
): void => {
  generation?.onStep(recipe);
  const stepGroups = groups[index];
  const contextGroups = generation?.groups[index];
  recipe.dispatches.forEach((dispatch, id) => {
    const bindGroup = stepGroups[id] ?? contextGroups?.[id];
    if (bindGroup === undefined) {
      throw new ExecutionError(
        `dispatch '${dispatch.key}': the bind group that binds objects owned by the GenerationContext` +
          " is not baked (the split baking of ADR 0066 decision 5 — reached the baked path without" +
          " the context-side baking)",
      );
    }
    dispatchWithWork(
      scheduler,
      dispatch,
      bindGroup,
      resolveWorkgroups(dispatch, generation?.encoding),
    );
  });
};
