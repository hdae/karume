/**
 * Derivation phase — lowers the execution step list to the recipe list ({@link RecipeBuilder}).
 *
 * Structure: "planning (pure functions, plan.ts) → **derivation** (here) → **execution** (recipe
 * types and the generic loop — src/runtime/recipe.ts)". For each step it looks up the pipeline /
 * layout, writes the params, decides the bind surface and workgroup count, and builds a
 * {@link StepRecipe}. Per-op branching (kernel-variant selection, replay of fusion rules, the i8a8 /
 * f16 opt-in paths) is closed within this layer.
 *
 * ## Layout (entry point of the derivation phase)
 *
 * **This file is the entry point of the derivation phase**; the derivation per operator family is
 * split into 6 files under `recipe-builders/` (elementwise / layout / linear / norm / attention /
 * conv). What this file holds is name resolution and settling every step
 * ({@link RecipeBuilder.buildRecipes}), the replay of fused steps, the dispatch on contract kind, the
 * params cache and its counts, and weight-slot resolution; the family functions receive one narrow
 * surface, {@link RecipeBuildFace}.
 * MUST: consumers **import it by this name (`recipe-builder.ts`)**. The split into 6 family files is
 * an implementation convenience only; {@link RecipeBuilder} stays the single entry point of the
 * derivation phase.
 * MUST: value dependencies stay **one-way, recipe-builder → recipe-builders**. The family side
 * refers to this file only with `import type` (erased, so no reverse edge appears in the run-time
 * import graph), and **values** both sides need (the params usage) live in
 * `recipe-builders/params-usage.ts`.
 *
 * MUST: the derivation phase **does not touch run-lifetime state** (it neither allocates from a
 * {@link RunArena} nor issues dispatches). It touches only objects whose lifetime is the Session or
 * longer (weights, per-channel scales, the params cache = Session-resident; pipelines =
 * device-resident {@link SessionPipelines}), and this is the very basis of "the products of the
 * derivation phase may be kept resident in the Session keyed by the resolved bindings" (executor.ts's
 * PreparedPlan).
 * MUST: dependencies run **one way, executor.ts → here** (no reverse edge, type imports included).
 * Session state is received only through the structural surface {@link RecipeBuilderContext}.
 */

import { gridStrideWorkgroups } from "../codegen/dispatch.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import type { AttentionDeviceFacts } from "./attention-route.ts";
import type { IrGraph } from "../format/ir.ts";
import type { RunArena } from "../gpu/arena.ts";
import type { GpuContext } from "../gpu/device.ts";
import type { SessionPipelines } from "../gpu/pipeline-cache.ts";
import { CAST_OP, numel, stateReadonly, WEIGHT_SLOTS, WHERE_OP } from "../ops.ts";
import type {
  AttentionTokenMajor,
  ExecStep,
  FusedOperand,
  FusedStep,
  LinearPrologue,
  PackedActivations,
} from "./fusion.ts";
import { ExecutionError, type NodePlan } from "./plan.ts";
import {
  type BindingRecipe,
  type BindingSource,
  type GenerationLimits,
  type StepOutput,
  type StepRecipe,
  StepRecipeBuilder,
  type TempSource,
  validateStepRecipe,
  type ValueSource,
} from "./recipe.ts";
import type {
  ComputePrecision,
  I8a8Dot,
  LinearGemvReduce,
  ParamsCacheStats,
  RmsNormReduce,
  StateAttentionReduce,
} from "./session-types.ts";
import type { ResidentWeight } from "./weight-residency.ts";
import {
  buildArgmax,
  buildCumsum,
  buildElementwise,
  buildMaskedFill,
  buildRowReduce,
  buildSoftmax,
  buildTopk,
} from "./recipe-builders/elementwise.ts";
import {
  buildCat,
  buildFlip,
  buildGather,
  buildPad,
  buildStridedCopy,
  buildUpsampleBilinear2d,
} from "./recipe-builders/layout.ts";
import { buildBmm, buildEmbedding, buildLinear, buildMatmul } from "./recipe-builders/linear.ts";
import {
  buildGroupNorm,
  buildLayerNorm,
  buildRmsNorm,
  buildStaticQuantize,
} from "./recipe-builders/norm.ts";
import {
  buildAttention,
  buildReadonlyStateAttention,
  buildStateAppend,
  buildStateAttention,
} from "./recipe-builders/attention.ts";
import {
  buildConv1d,
  buildConv2d,
  buildConv3d,
  buildConvTranspose1d,
  buildDeformConv2d,
  buildGatedDeltaScan,
  buildGruScan,
} from "./recipe-builders/conv.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./recipe-builders/params-usage.ts";

/**
 * The Session state the derivation phase reads (a structural surface with **only the fields it
 * needs**).
 *
 * MUST: do not import executor.ts's `SessionState` — to keep the one-way import Session →
 * RecipeBuilder intact at the type level too. The Session passes its own state as is (it fits
 * structurally).
 * MUST: run-lifetime containers (the run arena of {@link RunArena}, env, the scheduler) are not
 * carried. The moment they are, "the derivation phase does not touch run-lifetime state" becomes a
 * mere statement in the module doc, and the basis for keeping prepared plans resident in the Session
 * disappears from the types.
 */
export type RecipeBuilderContext = {
  readonly gpu: GpuContext;
  readonly graph: IrGraph;
  /**
   * Where pipelines are looked up (the device-lifetime cache + this Session's usage record —
   * {@link SessionPipelines}). The objects live on the device side, so they survive across Sessions
   * regardless of the rule against carrying run-lifetime containers (the MUST below).
   */
  readonly cache: SessionPipelines;
  /** Where params are allocated (the **Session-resident** weights arena — the MUST of `#writeParams`). */
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  readonly paramsCache: Map<string, GPUBuffer>;
  /**
   * Weights resident in compressed form (residency class, scale, group length — the source for
   * kernel-variant selection and extra bindings. A value not in the table is read as f32).
   */
  readonly residentWeights: ReadonlyMap<string, ResidentWeight>;
  readonly linearCompute: "f32" | "a8" | "f16";
  readonly convCompute: "f32" | "a8";
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
  /**
   * The device facts the online attention kernel's route reads (ADR 0132 — fixed once at Session
   * construction, like {@link geometryProfile}).
   */
  readonly attentionDevice: AttentionDeviceFacts;
  /** The reduction form of states-form attention ③PV (the executor's {@link SessionState} decides the default). */
  readonly stateAttentionReduce: StateAttentionReduce;
  readonly linearGemvReduce: LinearGemvReduce;
  /**
   * Whether the eligible decode linears run on the integer GEMV (`SessionOptions.integerGemv`;
   * the linear's guard reads it through `linearIntegerGemvVariant`, the same function as the
   * packed pairing).
   */
  readonly integerGemv: boolean;
  readonly rmsNormReduce: RmsNormReduce;
  /**
   * The parallelism target of the row-block GEMV (`SessionOptions.linearGemvRowsThreadTarget` — a
   * static knob fixed at Session creation. `undefined` is the kernel-side default).
   */
  readonly linearGemvRowsThreadTarget: number | undefined;
  /**
   * The i8a8 integer dot-product variant (**a separate field per family** — split because "both
   * variants are bit-identical" was once thought refuted on real hardware for attention only; that
   * inference was withdrawn on 2026-08-29, and the fields stay — `SessionState.attentionI8a8Dot` in
   * session-build.ts. The executor's {@link SessionState} decides the default).
   */
  readonly linearI8a8Dot: I8a8Dot;
  readonly attentionI8a8Dot: I8a8Dot;
  /**
   * Forcing the number of row blocks (**test only** — the executor's `ROW_BLOCK_SPLIT`). The
   * decomposed path receives it through the fusion rules, but the row blocks of the states form (ADR
   * 0067 decision 7) are split directly by the derivation phase, so the same knob is needed here too
   * (the forced-split parity is not driven by a different knob per path).
   */
  readonly rowBlockSplit: number | undefined;
  /**
   * The budget per row-block temporary (`SessionOptions.attentionRowBlockMaxBytes` — ADR 0124).
   * Fused attention and the states form are split directly by the derivation phase, so the same value
   * as in the fusion rules' context is held here too (same reason as {@link rowBlockSplit}).
   * `undefined` = no budget.
   */
  readonly attentionRowBlockMaxBytes: number | undefined;
  /**
   * The GEMM geometry profile (chosen once at construction by the executor's {@link SessionState} —
   * DECIDED: ADR 0115). Where the derivation of matmul / bmm / linear, fused attention, conv2d and
   * i8a8 looks up geometry.
   */
  readonly geometryProfile: GeometryProfile;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly outputNames: ReadonlySet<string>;
};

/**
 * The context the build of state nodes needs (slot capacities with bindings resolved, and where the
 * pre-run checks are collected).
 *
 * MUST: collected once, in the derivation phase ({@link GenerationLimits} has the same lifetime as
 * the prepared plan, and a hit run does not run this). Recounting per run would either drop the
 * check from hit runs only or split it into two implementations.
 */
export type StateBuildContext = {
  /** Slot name → concrete shape including capacity, with bindings resolved (passed by `GenerationContext`). */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly chunkRows: Set<number>;
  readonly fullCapacities: Map<string, number>;
};

/**
 * The surface of shared services the family derivations (under `recipe-builders/`) may touch.
 *
 * MUST: **only this surface** is passed to the family modules, never the {@link RecipeBuilder}
 * itself. Its job is to keep one owner of the params cache and one counter of params counts, and
 * `writeParams` encloses both (the moment the surface is widened to expose `paramsCache` directly,
 * the counting scatters across families).
 */
export type RecipeBuildFace = {
  /**
   * The parts of the derivation-phase state that the family derivations **actually read**.
   *
   * MUST: the mutable `paramsCache` / `weights` ({@link RecipeBuilderContext}) are not carried here
   * — `readonly` applies only to the property reference, and the contents of a Map or an arena stay
   * writable. Passing the whole state turns "keep one owner of the params cache" above into a mere
   * statement in the doc.
   * Fields are added only when a reader actually appears.
   */
  readonly state: Pick<
    RecipeBuilderContext,
    | "gpu"
    | "cache"
    | "linearCompute"
    | "convCompute"
    | "attentionCompute"
    | "attentionScoreStorage"
    | "attentionDevice"
    | "stateAttentionReduce"
    | "linearGemvReduce"
    | "integerGemv"
    | "rmsNormReduce"
    | "linearGemvRowsThreadTarget"
    | "linearI8a8Dot"
    | "attentionI8a8Dot"
    | "rowBlockSplit"
    | "attentionRowBlockMaxBytes"
    | "geometryProfile"
  >;
  readonly writeParams: (params: Uint32Array<ArrayBuffer>, usage: number) => GPUBuffer;
  readonly weightStorage: (step: NodePlan) => WeightStorage;
  /** The upload-time layout flag of the node's weight (`"conv3d-tap-major"` — ADR 0133 decision 4), else undefined. */
  readonly weightLayout: (step: NodePlan) => "conv3d-tap-major" | undefined;
  readonly weightScaleBindings: (step: NodePlan, binding: number) => readonly BindingRecipe[];
  readonly weightGroupSize: (step: NodePlan) => number;
};

/**
 * One entry newly put into the params cache ({@link RecipeBuilder.addedParams} — the unit that
 * {@link RecipeBuilder.forgetParams} rolls back when a run settles with a failure).
 */
export type AddedParams = { readonly key: string; readonly buffer: GPUBuffer };

/**
 * The body of the derivation phase. A Session holds exactly one and calls
 * {@link RecipeBuilder.buildRecipes} from the miss path of run / enqueue.
 */
export class RecipeBuilder {
  readonly #state: RecipeBuilderContext;
  /** The surface passed to the family derivations (one object enclosing the private helpers, built once in the constructor). */
  readonly #face: RecipeBuildFace;
  /**
   * The params counts of the run in progress (reset at the head of a run, moved to the Session's
   * diagnostics when it settles). New allocations are kept as **the list of objects put in**, not as
   * a count — the rollback of a failed execution ({@link RecipeBuilder.forgetParams}) needs this
   * list, and the count (`allocCount`) is derived from its length (counting independently would let
   * the two drift on a path that updates only one).
   */
  #paramsAdded: AddedParams[] = [];
  #paramsReuseCount = 0;

  constructor(state: RecipeBuilderContext) {
    this.#state = state;
    this.#face = {
      state,
      writeParams: (params, usage) => this.#writeParams(params, usage),
      weightStorage: (step) => this.#weightStorage(step),
      weightLayout: (step) => this.#weightLayout(step),
      weightScaleBindings: (step, binding) => this.#weightScaleBindings(step, binding),
      weightGroupSize: (step) => this.#weightGroupSize(step),
    };
  }

  /** Resets the params counts at the head of a run (the Session calls it once at the entry of run / enqueue). */
  resetParamsStats(): void {
    this.#paramsAdded = [];
    this.#paramsReuseCount = 0;
  }

  /** The params counts accumulated since the reset (the Session's `lastRunParams` is exactly this value). */
  get paramsStats(): ParamsCacheStats {
    return { allocCount: this.#paramsAdded.length, reuseCount: this.#paramsReuseCount };
  }

  /** What was newly put into the params cache since the reset (a copy — it can be carried until the failure settles). */
  get addedParams(): readonly AddedParams[] {
    return [...this.#paramsAdded];
  }

  /**
   * Removes from the cache the params put in by an execution that settled with a failure.
   *
   * An allocation that hits a non-fatal OOM does not throw; it returns an invalid buffer, and the
   * failure is found only later by the errorScope. Without removal, re-derivation after memory frees
   * up keeps hitting the invalid buffer under the same content key.
   * MUST: remove an entry only when its key points at **the same object**. Removing by key would also
   * remove another object put back in with the same content before the rollback.
   * MUST: do not destroy it. The weights arena owns it, and destruction rides on `Session.dispose`
   * (the MUST of {@link RecipeBuilder.#writeParams} — do not add another party responsible for
   * destruction). A removed object stays until dispose, referenced by no plan (each one is tens of
   * bytes).
   */
  forgetParams(added: readonly AddedParams[]): void {
    for (const { key, buffer } of added) {
      if (this.#state.paramsCache.get(key) === buffer) this.#state.paramsCache.delete(key);
    }
  }

  /**
   * Derivation phase — lowers the step list to the recipe list. Issues not a single GPU command and
   * touches no run-lifetime object (the buffers of {@link RunArena}) (the MUST of the module doc).
   *
   * @param stateShapes The state-slot shapes with bindings resolved (slot name → concrete shape
   *   including capacity).
   *   MUST: a recipe stays a pure function of "bindings ∪ capacities" — the context's **identity is
   *   not passed** (ADR 0066 decision 5; the objects are handed out per run by `GenerationEncoding`).
   *   A derivation without it that meets a state node fails loudly (it does not silently build the
   *   older form).
   */
  async buildRecipes(
    steps: readonly ExecStep[],
    stateShapes?: ReadonlyMap<string, readonly number[]>,
  ): Promise<{
    readonly recipes: readonly StepRecipe[];
    readonly generation: GenerationLimits;
  }> {
    // The objects are not fixed until the execution phase, so the derivation phase tracks only
    // "whether the value name is already defined" (a missing binding fails loudly here instead of
    // being carried into the execution phase).
    const defined = new Set(this.#state.graph.inputs.map((spec) => spec.name));
    const states: StateBuildContext = {
      shapes: stateShapes ?? new Map(),
      chunkRows: new Set(),
      fullCapacities: new Map(),
    };
    const recipes: StepRecipe[] = [];
    // Names are resolved in the original order; only the compilation waits of the steps overlap.
    // DECIDED: docs/decisions/0042-prepared-execution-plan.md#addendum-2026-09-11--asynchronous-compilation
    const pending = steps.map((step) => this.#buildStep(step, defined, states));
    // Leaving early on a failure in the middle would let later params writes overtake the run's
    // clean-up. Wait for every step, and report failures in the original step order. The recipe list
    // keeps the order of GPU commands.
    const settled = await Promise.allSettled(pending);
    for (const result of settled) {
      if (result.status === "rejected") throw result.reason;
      recipes.push(result.value);
    }
    return {
      recipes,
      generation: { chunkRows: states.chunkRows, fullCapacities: states.fullCapacities },
    };
  }

  /**
   * Value name → source on the bind surface. Weights are Session-resident, so they are resolved down
   * to the object; everything else (graph inputs, node outputs, aliases) stays a value name. It can
   * never be a node temporary, so the return type is {@link ValueSource} (usable as is as an alias
   * origin too).
   */
  #bindingSource(name: string, defined: ReadonlySet<string>): ValueSource {
    const resident = this.#state.weightBuffers.get(name);
    if (resident !== undefined) return { kind: "resident", buffer: resident };
    if (!defined.has(name)) throw new ExecutionError(`no buffer for value '${name}'`);
    return { kind: "value", name };
  }

  /**
   * The recipe of one execution step (a plain node or a fused step).
   *
   * MUST: the bookkeeping allocate → retain → body → release of inputs (the release list) → release
   * of the definition's own count is closed in **one path** for both (the replay side is
   * src/runtime/transient-plan.ts). Hand-written release bookkeeping per fusion would make the plan's
   * reference counting a separate implementation per fusion, and a single drift becomes a silently
   * wrong value with no exception (a release too early corrupts the value through reassignment; too
   * many and the peak does not drop).
   * MUST: the output list is built in **ascending output slot** (the ordering rule of
   * {@link StepRecipe.outputs} — one order shared with execution, planning and baking). A fused step
   * has a single output (the window guard of fusion.ts).
   */
  async #buildStep(
    step: ExecStep,
    defined: Set<string>,
    states: StateBuildContext,
  ): Promise<StepRecipe> {
    // A plain node's output list has the same order and length as `node.outs` ({@link NodePlan.outputs} in plan.ts).
    const outSlots: readonly { readonly name: string; readonly shape: readonly number[] }[] =
      step.kind === "node"
        ? step.plan.outputs
        : [{ name: step.outputName, shape: step.outputShape }];
    // The operand order of the bind surface (no duplicates) and the release list of the bookkeeping
    // are different things. For a plain node both equal node.ins; only a fused step declares the two
    // separately.
    const bindNames = step.kind === "node" ? step.plan.node.ins : step.binds;
    const consumedNames = step.kind === "node" ? step.plan.node.ins : step.ins;
    const binds = bindNames.map((name) => this.#bindingSource(name, defined));
    // An alias that keeps the element order **makes the input buffer the output's object as is**
    // (alias — ADR 0011). It issues neither a dispatch nor an allocation. planGraph has already
    // checked that the element counts match, so the real buffer behind the alias always covers the
    // size of the declared shape.
    // MUST: the alias origin is the head of the bind surface (it can never be a temp).
    // MUST: the ops that can alias (reshape / expand / permute) have a **single output**, so the
    // output list of an alias is always exactly one. This form cannot express aliasing a
    // multi-output op (adding it would require declaring "which input" per slot for the alias).
    const aliasesInput = step.kind === "node" && step.aliasesInput;
    const outputs: readonly StepOutput[] = outSlots.map(({ name, shape }): StepOutput => {
      const bookkeeping = {
        name,
        uses: this.#state.useCounts.get(name) ?? 0,
        pinned: this.#state.outputNames.has(name),
      };
      return aliasesInput
        ? { ...bookkeeping, kind: "alias", source: binds[0] }
        : { ...bookkeeping, kind: "alloc", byteLength: numel(shape) * 4 };
    });
    // Outputs are bound as "that value name". The execution phase puts them in env before the
    // dispatches, so binds within the same step can be resolved through this list too.
    const outs: readonly BindingSource[] = outputs.map((output) => ({
      kind: "value",
      name: output.name,
    }));

    // Advance the definitions after checking the input references, keeping name resolution the same
    // as the execution order.
    for (const output of outputs) defined.add(output.name);
    const builder = new StepRecipeBuilder();
    if (step.kind === "node") {
      await this.#buildNode(
        step.plan,
        step.aliasesInput,
        binds,
        outs,
        builder,
        states,
        step.packedActivations,
        step.prologue,
        step.tokenMajor,
      );
    } else {
      await this.#buildFused(step, binds, outs, builder);
    }

    const recipe: StepRecipe = {
      outputs,
      temps: builder.temps,
      dispatches: builder.dispatches,
      releases: consumedNames,
      // MUST: the discrimination is by contract (only `state_append` writes — ADR 0067 decision 5).
      // Fused steps never contain nodes that touch state (the window guard of fusion.ts), so they are
      // always false.
      writesState: step.kind === "node" && step.plan.contract.kind === "stateAppend",
    };
    // MUST: the static check of the declaration happens in this one place (the only exit through
    // which a recipe leaves). Each `#build*` of plain nodes and the fusion replay declare into the
    // same builder ({@link StepRecipeBuilder}), so a check per path would let a newly added path pass
    // unchecked.
    validateStepRecipe(recipe);
    return recipe;
  }

  /**
   * The dispatch list of a fused step.
   *
   * The default bind surface is "params, inputs…, output" (the common form of fusion rules that do
   * not declare `operands`); only rules that declare `operands` take an order mixing in step
   * temporaries. **Allocation and release of temporaries are replayed into
   * {@link StepRecipeBuilder}** (no second point that derives lifetimes), so the plan's bookkeeping
   * stays closed in the same single path as plain nodes (the replay of
   * src/runtime/transient-plan.ts).
   *
   * MUST: a {@link FusedStep} has a **single output** (the window guard of fusion.ts requires the
   * contract's output count to be 1), so `{ kind: "output" }` always points at `outs[0]`. Adding
   * multi-output fusion means redesigning from a form where operands carry an output slot
   * (`{ kind: "output", slot }`).
   */
  async #buildFused(
    step: FusedStep,
    binds: readonly BindingSource[],
    outs: readonly BindingSource[],
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const temps: TempSource[] = [];
    const resolve = (operand: FusedOperand): BindingSource => {
      if (operand.kind === "output") return outs[0];
      if (operand.kind === "bind") {
        const source = binds[operand.index];
        if (source === undefined) {
          throw new ExecutionError(
            `fusion rule '${step.rule}': bind index ${operand.index} is outside the ${binds.length} declared binds`,
          );
        }
        return source;
      }
      if (operand.kind === "weightScale") {
        const name = step.binds[operand.index];
        const resident = this.#state.residentWeights.get(name);
        if (resident === undefined || resident.storage === "f16" || resident.storage === "bf16") {
          throw new ExecutionError(
            `fusion rule '${step.rule}': bind ${operand.index} has no resident scale`,
          );
        }
        // Borrow the existing weight scale. The owner and the sharedWeights lease stay with the Session.
        return { kind: "resident", buffer: resident.scale };
      }
      const temp = temps[operand.id];
      // Binding an unallocated temporary breaks the lifetime declaration (it is read from a dispatch
      // before its allocation).
      if (temp === undefined) {
        throw new ExecutionError(`fusion rule '${step.rule}': unallocated temporary ${operand.id}`);
      }
      return temp;
    };
    for (const [index, dispatch] of step.dispatches.entries()) {
      for (const [id, temp] of step.temps.entries()) {
        if (temp.allocBefore === index) temps[id] = builder.allocTemp(temp.byteLength);
      }
      const { pipeline, layout, roles } = await this.#state.cache.get(
        dispatch.key,
        dispatch.wgsl(),
      );
      const params = this.#writeParams(
        dispatch.params,
        dispatch.paramsStorage === true ? PARAMS_STORAGE_USAGE : PARAMS_UNIFORM_USAGE,
      );
      const operands = dispatch.operands ?? [
        ...binds.map((_, at): FusedOperand => ({ kind: "bind", index: at })),
        { kind: "output" } as const,
      ];
      const { workgroups } = dispatch;
      builder.dispatch({
        key: dispatch.key,
        pipeline,
        layout,
        roles,
        params,
        bindings: operands.map((operand, slot) => ({
          binding: slot + 1,
          source: resolve(operand),
        })),
        workgroups:
          workgroups.kind === "tiled"
            ? workgroups.counts
            : [gridStrideWorkgroups(workgroups.items, workgroups.size, limit), 1, 1],
      });
      // MUST: releases at the same boundary go in reverse allocation order (the same order as the plan replay).
      for (let id = step.temps.length - 1; id >= 0; id -= 1) {
        if (step.temps[id].releaseAfter === index) builder.releaseTemp(temps[id]);
      }
    }
  }

  /**
   * The body of one plain node (allocation, retain and release are done by the plan replay).
   *
   * MUST: each per-op builder receives the **output list** (`outs` — ascending output slot), and a
   * single-output kernel writes only `outs[0]` (ADR 0068 decision 1). A multi-output op writes its own
   * list with explicit slots. The **allocation** order of outputs is the ascending order of the list,
   * one order shared with execution, slot derivation and baking (the ordering rule of
   * {@link StepRecipe.outputs}).
   */
  async #buildNode(
    step: NodePlan,
    aliasesInput: boolean,
    binds: readonly BindingSource[],
    outs: readonly BindingSource[],
    builder: StepRecipeBuilder,
    states: StateBuildContext,
    /** The role of packed int8 activations (ADR 0105 — acceptance was settled in one place, fusion.ts). */
    packed: PackedActivations | undefined,
    /** The producer folded into a linear's step ① (perf-ledger F-4 — accepted in one place, fusion.ts). */
    prologue: LinearPrologue | undefined,
    /** The permutes a fused attention absorbed (perf-ledger F-3 A4 / F-8 — accepted in one place, fusion.ts). */
    tokenMajor: AttentionTokenMajor | undefined,
  ): Promise<void> {
    // MUST: only a linear executes a folded producer. Any other kind would run without the folded op
    // (its input is the producer's input) — a silent wrong value, so it fails loudly.
    if (prologue !== undefined && step.contract.kind !== "linear") {
      throw new ExecutionError(
        `${step.node.op}: carries a ${prologue.op} prologue (perf-ledger F-4), which only a linear executes`,
      );
    }
    // MUST: only a fused attention (no states field) reads and writes token-major operands. Any other
    // node would read the re-pointed inputs in the wrong layout — a silent wrong value, so it fails
    // loudly.
    if (
      tokenMajor !== undefined &&
      (step.contract.kind !== "attention" || Object.keys(step.node.states).length > 0)
    ) {
      throw new ExecutionError(
        `${step.node.op}: carries absorbed permutes (perf-ledger F-3 / F-8), which only a fused attention executes`,
      );
    }
    switch (step.contract.kind) {
      case "unary":
      case "binary":
        await buildElementwise(
          this.#face,
          step,
          { op: step.contract.name, dtype: step.inputDtypes[0] },
          binds,
          outs,
          builder,
        );
        break;
      case "cast":
        await buildElementwise(
          this.#face,
          step,
          { op: CAST_OP, dtype: step.inputDtypes[0], to: step.outputs[0].dtype },
          binds,
          outs,
          builder,
        );
        break;
      case "where":
        // The generated input is the dtype of the **value slot** (slot 0 is the condition, always bool).
        await buildElementwise(
          this.#face,
          step,
          { op: WHERE_OP, dtype: step.inputDtypes[1] },
          binds,
          outs,
          builder,
        );
        break;
      case "cumsum":
        await buildCumsum(this.#face, step, binds, outs, builder);
        break;
      case "matmul":
        await buildMatmul(this.#face, step, binds, outs, builder);
        break;
      case "bmm":
        await buildBmm(this.#face, step, binds, outs, builder);
        break;
      case "gather":
        await buildGather(this.#face, step, binds, outs, builder);
        break;
      case "rowReduce":
        await buildRowReduce(this.#face, step, step.contract.name, binds, outs, builder);
        break;
      case "argmax":
        await buildArgmax(this.#face, step, binds, outs, builder);
        break;
      case "topk":
        await buildTopk(this.#face, step, binds, outs, builder);
        break;
      case "permute":
        // When the element order does not change, the existing alias bookkeeping is all it takes (ADR 0011).
        if (!aliasesInput) {
          await buildStridedCopy(this.#face, step, "permute", binds, outs, builder);
        }
        break;
      case "slice":
      case "symPrefixSlice":
        await buildStridedCopy(this.#face, step, step.contract.kind, binds, outs, builder);
        break;
      case "expand":
        // An identity expand is already aliased (0 dispatches). With even one replicated axis it is a materialising copy.
        if (!aliasesInput) {
          await buildStridedCopy(this.#face, step, step.contract.kind, binds, outs, builder);
        }
        break;
      case "cat":
        await buildCat(this.#face, step, binds, outs, builder);
        break;
      case "pad":
        await buildPad(this.#face, step, binds, outs, builder);
        break;
      case "flip":
        await buildFlip(this.#face, step, binds, outs, builder);
        break;
      case "linear":
        await buildLinear(this.#face, step, binds, outs, builder, packed, prologue);
        break;
      case "layerNorm":
        await buildLayerNorm(this.#face, step, binds, outs, builder);
        break;
      case "staticQuantize":
        await buildStaticQuantize(this.#face, step, binds, outs, builder, packed);
        break;
      case "rmsNorm":
        await buildRmsNorm(this.#face, step, binds, outs, builder);
        break;
      case "softmax":
        await buildSoftmax(this.#face, step, false, binds, outs, builder);
        break;
      case "safeSoftmax":
        await buildSoftmax(this.#face, step, true, binds, outs, builder);
        break;
      case "attention":
        // The presence of the field decides the form (ADR 0067 decision 4). The states form is a
        // separate kernel family (src/kernels/state-attention.ts) and shares not a single byte with
        // fused attention.
        // The states form splits further in two: the older form that also reads this step's k/v, and
        // the **readonly** form that reads only the past (ADR 0096 stage 2 addendum — ins is q alone,
        // a separate kernel with the bindings packed one by one).
        if (Object.keys(step.node.states).length === 0) {
          await buildAttention(this.#face, step, binds, outs, builder, tokenMajor);
        } else if (stateReadonly(step.node.attrs, `nodes (${step.node.op})`)) {
          await buildReadonlyStateAttention(this.#face, step, binds, outs, builder, states);
        } else {
          await buildStateAttention(this.#face, step, binds, outs, builder, states);
        }
        break;
      case "embedding":
        await buildEmbedding(this.#face, step, binds, outs, builder);
        break;
      case "maskedFill":
        await buildMaskedFill(this.#face, step, binds, outs, builder);
        break;
      case "conv1d":
        await buildConv1d(this.#face, step, binds, outs, builder);
        break;
      case "conv2d":
        await buildConv2d(this.#face, step, binds, outs, builder);
        break;
      case "conv3d":
        await buildConv3d(this.#face, step, binds, outs, builder);
        break;
      case "convTranspose1d":
        await buildConvTranspose1d(this.#face, step, binds, outs, builder);
        break;
      case "deformConv2d":
        await buildDeformConv2d(this.#face, step, binds, outs, builder);
        break;
      case "upsampleBilinear2d":
        await buildUpsampleBilinear2d(this.#face, step, binds, outs, builder);
        break;
      case "groupNorm":
        await buildGroupNorm(this.#face, step, binds, outs, builder);
        break;
      case "gruScan":
        await buildGruScan(this.#face, step, binds, outs, builder);
        break;
      case "gatedDeltaScan":
        await buildGatedDeltaScan(this.#face, step, binds, outs, builder);
        break;
      case "reshape":
        // Aliasing was done in #buildStep (this op issues not a single dispatch).
        break;
      case "stateAppend":
        await buildStateAppend(this.#face, step, binds, builder, states);
        break;
      default: {
        // MUST: stop an unhandled kind here. If the switch falls through, the next node runs with
        // not a single byte written to the output buffer, and the debris of reassignment becomes the
        // value (a break of the full-write invariant — ADR 0014). It is unreachable by type; the job
        // of this branch is to turn a missing wiring red at compile time when an op is added.
        const unhandled: never = step.contract;
        throw new ExecutionError(`unhandled op kind: ${JSON.stringify(unhandled)}`);
      }
    }
  }

  /**
   * The weight resident at this node's weight slot (residency class, scale, group length).
   * `undefined` for a value not resident in compressed form (= read as f32).
   *
   * MUST: the slot position is looked up in **the same table** as the eligibility decision
   * ({@link WEIGHT_SLOTS}). If only one side drifted, an f32 kernel would read a weight uploaded in
   * compressed form (or the reverse) — a reinterpretation of the bit pattern, with not a single
   * exception.
   */
  #residentWeight(step: NodePlan): ResidentWeight | undefined {
    return this.#state.residentWeights.get(this.#weightSlotName(step));
  }

  /** The layout flag the upload set on this node's weight (only i8 records carry one). */
  #weightLayout(step: NodePlan): "conv3d-tap-major" | undefined {
    const resident = this.#residentWeight(step);
    return resident?.storage === "i8" ? resident.layout : undefined;
  }

  /** The value name bound to this node's weight slot ({@link WEIGHT_SLOTS} is the source of truth). */
  #weightSlotName(step: NodePlan): string {
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' has no weight slot definition`);
    }
    return step.node.ins[slot];
  }

  /** The storage form of this node's weight slot (kernel-variant selection — ADR 0018). */
  #weightStorage(step: NodePlan): WeightStorage {
    return this.#residentWeight(step)?.storage ?? "f32";
  }

  /**
   * The extra bindings of the i8 / i4 variants (per-channel / group scale). For f32 / f16 this is an
   * empty array, and the bind group stays as before (ADR 0019 / 0069).
   *
   * MUST: the binding number comes from the kernel-side constant (`*_SCALE_BINDING`). If the WGSL
   * declaration and the executor each held their own number, adding a variant would shift only one.
   * NOTE: no run-time check for "i8 / i4 resident but no scale buffer" is needed —
   * {@link ResidentWeight} requires a scale paired with the residency class, and the type guarantees
   * it.
   */
  #weightScaleBindings(step: NodePlan, binding: number): readonly BindingRecipe[] {
    const resident = this.#residentWeight(step);
    if (resident === undefined || resident.storage === "f16" || resident.storage === "bf16") {
      return [];
    }
    return [{ binding, source: { kind: "resident", buffer: resident.scale } }];
  }

  /**
   * The group length of an i4-resident weight (the value the executor copied from the declaration —
   * ADR 0069).
   *
   * It is called only on paths where the residency class is known to be i4, so anything else fails
   * loudly instead of silently falling back to a default — filling in a default would be the silently
   * wrong value "an asset with group 64 runs on a group-32 pipeline".
   */
  #weightGroupSize(step: NodePlan): number {
    const resident = this.#residentWeight(step);
    if (resident?.storage !== "i4") {
      throw new ExecutionError(
        `initializer '${this.#weightSlotName(step)}': i4-resident but has no group_size`,
      );
    }
    return resident.groupSize;
  }

  /**
   * Allocates and writes a dispatch's params uniform / storage. It goes through **a Session-resident
   * cache keyed by the content itself**, so params with the same content are allocated and uploaded
   * only once in the Session's life (every byte of params is a pure function of the graph,
   * node.attrs and the resolved shapes, and does not depend on tensor values at run time). The key is
   * the string concatenating the usage and every element — not a hash, so collisions are impossible
   * in principle.
   *
   * MUST: allocate from the **weights arena (Session-resident)**, not from a RunArena. A per-run arena
   * is destroyed at the end of the run, so handing out from it would leave the cache pointing at
   * destroyed buffers.
   * MUST: a cached buffer is **written once and never rewritten**. ADR 0004 excludes
   * `allocHostWritten` from pooling because "`queue.writeBuffer` overtakes earlier unsubmitted
   * encodes", but here no second writeBuffer to the same buffer exists, so the overtaking hazard
   * cannot arise in principle (sharing by identity of content, a layer separate from reassignment).
   * MUST: destruction rides on dispose of the weights arena (`Session.dispose` → the
   * flush-before-destroy of `RunArena.destroy`). A destruction path dedicated to the cache would give
   * flush two owners, and depending on which runs first a submit referencing a destroyed buffer would
   * appear.
   * NOTE: after a failed run (the `scheduler.discard` path), entries that **already existed** stay
   * valid. Buffer contents are immutable and discard drops only unsubmitted encodes, so params values
   * are unaffected. Only entries **newly put in** by the failed execution are removed by the Session
   * through {@link RecipeBuilder.forgetParams} (the allocation itself may have been invalid due to a
   * non-fatal OOM).
   * NOTE: **this cache is unbounded for the lifetime of the Session** (by-design —
   * `docs/limitations.md`). Unlike the other caches of the same scope (`MEASURED_HISTORY` in
   * gpu/submit.ts / `PREPARED_PLAN_CAPACITY`), which have bounds, this one has no eviction —
   * evicting params means destroying them, but live prepared plans **fold the objects in as direct
   * references**, so doing it safely needs separate bookkeeping, reference counting. Uses that run
   * many variable shapes in one Session (variable-length TTS / embeddings of variable sequence length)
   * pile up small buffers, one per node kind per run, so watch the growth of
   * `diagnostics().weights.allocCount` and recreate the Session. Being unbounded is itself pinned by a
   * check in tests/gpu_params_cache_test.ts.
   */
  #writeParams(params: Uint32Array<ArrayBuffer>, usage: number): GPUBuffer {
    const key = `u${usage}:${params.join(",")}`;
    const cached = this.#state.paramsCache.get(key);
    if (cached !== undefined) {
      this.#paramsReuseCount += 1;
      return cached;
    }
    const buffer = this.#state.weights.allocHostWritten(params.byteLength, usage);
    this.#state.gpu.device.queue.writeBuffer(buffer, 0, params);
    this.#state.paramsCache.set(key, buffer);
    this.#paramsAdded.push({ key, buffer });
    return buffer;
  }
}
