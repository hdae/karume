/**
 * The construction phase of a Session ({@link buildSessionState} = the body of `Session.build`, and
 * the parts only construction uses — companion scales / handling of supply units / the canary /
 * {@link SessionState}). The acceptance conditions and defaults of the knobs are in
 * session-options-check.ts (pure functions shared with the estimate).
 *
 * MUST: no import of executor.ts (no circular imports — the same rule as session-types.ts). The
 * construction phase returns only state ({@link SessionState}); `Session` has a private constructor,
 * so the object is created on the facade side. Dependencies run **one way**:
 * {@link "./executor.ts"} → here.
 */

import { codecLayout } from "../format/container/codecs.ts";
import { decodeBf16 } from "../format/bf16.ts";
import { alignF16Payload, decodeF16 } from "../format/f16.ts";
import { decodeI2 } from "../format/i2.ts";
import { decodeI4 } from "../format/i4.ts";
import { alignI8Payload, decodeI8 } from "../format/i8.ts";
import { reorderConv3dWeightTapMajor } from "../kernels/conv3d-i8a8.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";
import { RunArena } from "../gpu/arena.ts";
import {
  type AttentionI8a8Decision,
  decideAttentionI8a8Dot,
  formatAttentionI8a8Decision,
} from "../gpu/attention-dp4a-canary.ts";
import { assertDeviceUsable, type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import {
  discardFailureScopes,
  type GpuOutOfMemoryError,
  type GpuValidationError,
  popFailureScopes,
  pushFailureScopes,
} from "../gpu/error-scope.ts";
import { SessionPipelines } from "../gpu/pipeline-cache.ts";
import { copySubmitPolicy, SubmitScheduler } from "../gpu/submit.ts";
import { BUFFER_USAGE } from "../gpu/webgpu-constants.ts";
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { dp4aAvailable } from "../kernels/linear-i8a8.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import { A8_EMULATION_WARNING_KEY, a8EmulationWarning } from "./a8-emulation-warning.ts";
import { type AttentionDeviceFacts, attentionDeviceFacts } from "./attention-route.ts";
import type { FusionCounts } from "./fusion.ts";
import { sessionGeometryProfile } from "./geometry-selection.ts";
import { conv3dTapMajorInitializers, countUses, declaredDtypes, ExecutionError } from "./plan.ts";
import type { GenerationLimits, StepRecipe } from "./recipe.ts";
import type { TransientLimits } from "./transient-plan.ts";
import {
  assertWeightsWithinLimits,
  type ResidentWeight,
  resolveSharedWeights,
  type SharedWeight,
  type WeightResidency,
} from "./weight-residency.ts";
import {
  type ComputePrecision,
  I8A8_DOT,
  type I8a8Dot,
  type LinearGemvReduce,
  type RmsNormReduce,
  ROW_BLOCK_SPLIT,
  type SessionBuildStats,
  type SessionOptions,
  type StateAttentionReduce,
  type StorageDiagnostics,
} from "./session-types.ts";
import { executionKnobDefaults, sessionOptionsViolation } from "./session-options-check.ts";

/** MUST: buffers written with `queue.writeBuffer` are outside the pool (the arena's invariant). */
export const HOST_WRITTEN_USAGE =
  BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;

/**
 * The object of a companion scale (quantized codecs only; for a piece list, piece 1 only).
 *
 * `shape` is the **rank-2 group form** `[shape[rowAxis], row length / groupSize]` (container-v1
 * §6.1). Per-channel (i8 / i2) is `[rows, 1]` with one group, so consumers handle a single form.
 */
export type ReadyScale = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
};

/**
 * The object of one initializer passed from the supply side (the container's supply plan) to Session
 * construction. Passed as **bytes** (neither a container block nor the caller's container — the
 * supply side's form does not leak to the consumer).
 */
export type ReadyInitializer = {
  readonly name: string;
  /** The stored payload (whole / that piece only). Raw bytes without alignment padding. */
  readonly payload: Uint8Array<ArrayBuffer>;
  readonly scale?: ReadyScale;
  /**
   * The position within a split tensor (undefined when it arrives whole — container-v1 §5).
   *
   * The consumer allocates the buffer and uploads the scale on `first`, applies the tail-alignment
   * padding only on `last`, and writes each piece at the byte position given by `rowOffset` (padding
   * a middle piece would overwrite the first bytes of the next piece — a silently wrong value).
   */
  readonly piece?: {
    /** The row (leading dimension) where this piece starts. */
    readonly rowOffset: number;
    /** Rows in this piece. */
    readonly rows: number;
    /** Piece 1 (the one responsible for allocating the buffer and uploading the companion scale). */
    readonly first: boolean;
    /** Piece n (the one responsible for the tail-alignment padding). */
    readonly last: boolean;
  };
};

/**
 * The companion scale of quantized storage (ADR 0019 / 0069). Presence, F32 and shape
 * (normalisation to the rank-2 group form) are already handled by the supply side (the container's
 * merge layer), so this only builds a view.
 *
 * MUST: the `Float32Array` view is created without copying (the scale is small compared with the
 * weight body, but an unconditional copy here would double-allocate on the "resident as raw bytes"
 * path). The 4-byte alignment of the byte position is guaranteed by the supply side (the container's
 * 64 B-aligned blocks / the `byteOffset % 4` check on scales of the in-memory container).
 */
const scaleTensor = (
  item: ReadyInitializer,
  layout: string,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
} => {
  const scale = item.scale;
  if (scale === undefined) {
    // Presence is optional only in the type, so restate the check rather than skip silently (fail loudly).
    throw new ExecutionError(`initializer '${item.name}': storage ${layout} but no scale`);
  }
  return {
    bytes: scale.bytes,
    values: new Float32Array(
      scale.bytes.buffer,
      scale.bytes.byteOffset,
      scale.bytes.byteLength / 4,
    ),
    shape: scale.shape,
  };
};

/**
 * The slice of the companion scale read by the CPU decode of a piece (a row range of the leading
 * dimension).
 *
 * The scale is in the rank-2 group form `[rows, groups]`, and splitting into pieces is allowed only
 * for initializers whose row axis is 0 (container-v1 §5 rule ④ — guaranteed by the supply side). A
 * shape whose row axis is not 0 (conv_transpose1d's `[Cin,Cout,K]`) only arrives whole, so it never
 * appears here.
 */
const scaleForPiece = (
  scale: { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] },
  declaredRows: number,
  rowOffset: number,
  rows: number,
): { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] } => {
  if (scale.shape[0] !== declaredRows) return scale;
  const stride = scale.shape[1];
  return {
    values: scale.values.subarray(rowOffset * stride, (rowOffset + rows) * stride),
    shape: [rows, stride],
  };
};

/**
 * Maps a per-channel scale (rank 2 `[rows, 1]`) to the form read by the CPU decode (`decodeI8` /
 * `decodeI2` — they index with the strides of the keepdim broadcast form). The bytes are the same;
 * only the shape is reinterpreted.
 */
const keepdimScaleShape = (
  weightShape: readonly number[],
  rows: number,
  rowAxis: number,
): readonly number[] => weightShape.map((_, axis) => (axis === rowAxis ? rows : 1));

/**
 * Checks that the per-channel scale of the GPU-resident path is **in a form indexable by a flat
 * index** (ADR 0019).
 *
 * The kernel reads `wscale[output channel]`, so the scale must be rank 2 `[rows, 1]` stretched only
 * along the row axis (rows = the weight's `shape[rowAxis]`). Forms with more than one group (the
 * group form of i4) never reach the i8 / i2 residency classes, but a mismatch between declaration and
 * supply side becomes a silently wrong value, so **this is the only check**.
 */
const assertRowScale = (
  name: string,
  weightShape: readonly number[],
  scaleShape: readonly [number, number],
  rowAxis: number,
): void => {
  if (scaleShape[0] !== weightShape[rowAxis] || scaleShape[1] !== 1) {
    throw new ExecutionError(
      `initializer '${name}': scale [${scaleShape.join(",")}] is not, for weight [${weightShape.join(
        ",",
      )}] on axis ${rowAxis}, the per-channel shape [${weightShape[rowAxis]},1]`,
    );
  }
};

/**
 * The **supply unit** consumed by Session construction (one container part = the list of objects of
 * the blocks in that part). The fence (an empty submit + waiting for completion) happens once per
 * unit; an errorScope is set per item (block).
 */
export type WeightBatch = {
  readonly origin: string | undefined;
  /**
   * The list of objects of the part's blocks. Each pull hands over one item as soon as its blocks are
   * read (lazy), so the consumer uploads them one at a time and lets go — `queue.writeBuffer` copies
   * the bytes at the call (the WebGPU spec), so they are not held until the fence. The container
   * supply reads up to 2 blocks ahead of the item, across part boundaries (`READ_AHEAD_BLOCKS` in
   * {@link "./executor.ts"} — ADR 0108 addendum 14). The weight bytes alive on the JS side are
   * therefore not one part but one item (one block + on piece 1 the scale block that rides along + on
   * the expanded class its f32 decode result) plus the 2 blocks read ahead, plus the copy of the
   * carried scale that lives only during a piece list of the expanded class (the whole scale of that
   * initializer — copied on piece 1 and released on the last piece). With a scan-type source (hub)
   * the block and the riding scale are views into the retention slot's container, so one retention
   * slot + containers waiting for GC stack on top — while the read-ahead crosses a part boundary,
   * the parts its blocks fall in (up to 3) are alive (the carried scale copy and the decode result
   * are values, so even with scan-type sources they sit outside the retention slot). The number of
   * containers waiting for GC is not bounded by the declaration — measured external maximum was
   * about 1.5 to 4.4 times the largest part (container-v1 §11).
   *
   * MUST: iterate once, to the end, before pulling the next batch. A scan-type source (hub) keeps
   * only one part on the premise of reading parts sequentially in order, so pulling the next part
   * midway makes the whole part be read again. Iterating twice may silently yield 0 items (there is
   * no run-time detection — the only consumer is the internal `buildSessionState`, so the MUST in the
   * doc protects it).
   */
  readonly items: AsyncIterable<ReadyInitializer>;
};

/**
 * The label of the weight-upload interval (what the errorScope and the fence are attributed to).
 * Derived from `origin` — keeping it separately on the batch side would let the same failure appear
 * under two names when only one side changes how it names itself.
 */
const uploadLabel = (origin: string | undefined): string =>
  origin === undefined ? "weight upload" : `${origin} weight upload`;

/**
 * Adds the attribution (`part N`) to a failure that came from a supply unit and **returns the same
 * error** (passed through when there is no `origin` — the text does not change by a single
 * character).
 *
 * MUST: do not wrap it in a new Error. Callers branch on the class (a declaration violation =
 * `ContainerFormatError`); wrapping breaks the branch and cuts the stack.
 */
const attributeToOrigin = (origin: string | undefined, cause: unknown): unknown => {
  if (origin !== undefined && cause instanceof Error) {
    cause.message = `${origin}: ${cause.message}`;
  }
  return cause;
};

/**
 * The settled result of one initializer's error-scope pair: a failure (a typed scope failure, or
 * the pop's own rejection as is) or none.
 */
type ScopeOutcome = { readonly failed: false } | { readonly failed: true; readonly error: unknown };

/**
 * Turns one initializer's issued pop pair into a promise that never rejects, and calls `onFailed`
 * when the pair settles as a failure (what the build's early stop reads — the part loop in
 * {@link buildSessionState}).
 *
 * MUST: attach the handlers when the pop is issued. The outcomes are awaited later and in item
 * order ({@link firstScopeFailure}); a later item's rejection that settles while an earlier
 * failure is being thrown would otherwise surface as an unhandled rejection.
 */
const settleScopes = (
  pop: Promise<GpuValidationError | GpuOutOfMemoryError | undefined>,
  onFailed: () => void,
): Promise<ScopeOutcome> =>
  pop.then(
    (failure): ScopeOutcome => {
      if (failure === undefined) return { failed: false };
      onFailed();
      return { failed: true, error: failure };
    },
    (cause: unknown): ScopeOutcome => {
      onFailed();
      return { failed: true, error: cause };
    },
  );

/**
 * The first failure **in item order** among the outcomes (not the first to settle).
 *
 * MUST: await in item order. The contract (ADR 0108 addendum 13) is the error the per-item await
 * used to report — the first failing item — so a later item that settles earlier MUST NOT win.
 * Failures after the first are discarded: the per-item await never reached them (the build stopped
 * at the first), and a later one (another allocation on a device that has already run out, for
 * example) adds nothing to the first.
 */
const firstScopeFailure = async (
  outcomes: readonly Promise<ScopeOutcome>[],
): Promise<ScopeOutcome> => {
  for (const outcome of outcomes) {
    const settled = await outcome;
    if (settled.failed) return settled;
  }
  return { failed: false };
};

/**
 * The whole product of the derivation phase (Session-resident — the key is
 * `Session.#preparedKey` in {@link "./executor.ts"}).
 *
 * MUST: hold only what later phases actually reference (`GraphPlan.nodes` / `GraphPlan.bindings`
 * are read by nobody once recipe derivation is over). Holding derived products nobody reads detaches
 * the cache's lifetime from "facts that affect the run's inputs and outputs", and what is being
 * reused becomes unreadable.
 */
export type PreparedPlan = {
  /** The resolved shapes of inputs, initializers and every node output (read by `#bindInputs` / `#readOutputs`). */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly recipes: readonly StepRecipe[];
  /** The fusion counts decided at planning (hit runs report this value to the permanent diagnostics too — ADR 0040 decision 3). */
  readonly fusions: FusionCounts;
  /**
   * The plan facts the pre-run check of a generation run needs (`assertGenerationRun` in
   * {@link "./recipe.ts"}). Empty for a graph without state nodes, and such a run passes no check
   * (there is nothing to look at).
   */
  readonly generation: GenerationLimits;
};

/**
 * Decides the integer dot-product variant of fused attention (the entry of
 * {@link SessionState.attentionI8a8Dot}).
 *
 * The canary (src/gpu/attention-dp4a-canary.ts) runs only when "the extension is advertised and a8
 * was requested":
 *
 * - Not when `I8A8_DOT` is given — inserting an environment decision while a test forces a variant
 *   would erase from both the diagnostics and the numbers what was measured.
 * - Not advertised → the dp4a variant is not even generated, so there is nothing to decide (straight
 *   to emu as before).
 * - Not a8 → not a single i8a8 attention kernel is emitted, so this field's value is never read. It
 *   is the guard that keeps "the first-use cost of a feature not used" from being handed to every
 *   Session; the decision is paid by the first Session that requests a8, and memoised per device
 *   afterwards.
 *
 * When the canary decided by "not an exact match with the known answer, but within the sanity
 * band", it does **not pass silently** — the warning is printed inside the memoised body, so it
 * appears once per device (printing it per Session would repeat the same single fact just by lining
 * up a8 Sessions).
 *
 * The canary fires with the i8a8 attention geometry of `profile` (the geometry production's ①QK /
 * ③PV actually run). The memo can stay per device because the profile is either the table injected
 * into the same {@link GpuContext} (`acquireGpu({ geometryProfile })`) or, absent that, a pure
 * function of `adapterInfo` (`selectGeometryProfile`); both are invariant over the device's
 * lifetime — the same device gives the same geometry for every Session.
 */
const resolveAttentionI8a8Dot = async (
  gpu: GpuContext,
  forced: I8a8Dot | undefined,
  attentionCompute: ComputePrecision,
  dp4a: boolean,
  profile: GeometryProfile,
): Promise<I8a8Dot> => {
  if (forced !== undefined) return forced;
  if (!dp4a) return "emu";
  if (attentionCompute !== "a8") return "dp4a";
  const decision = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(async () => {
    const decided = await decideAttentionI8a8Dot(gpu, {
      qk: profile.i8a8.attentionQk,
      pv: profile.i8a8.attentionPv,
    });
    if (!decided.exact) warnInexactAttentionCanary(decided);
    return decided;
  });
  return decision.dot;
};

/**
 * Reports once that the canary decided within the band without an exact match.
 *
 * It does not stop because this case is a real, healthy device (Apple M2 — the rounding of the shared
 * f32 epilogue is only a few ULP off the known answer), and it does not stay silent because "passed
 * because within the band" is **a fact that affects measurement** (the a8 output on this device is
 * not bit-identical with other machines). The wording follows the main-tracking warning of
 * @karume/hub — what happened and what makes it go away, in one console.warn.
 */
const warnInexactAttentionCanary = (decision: AttentionI8a8Decision): void => {
  console.warn(
    `@karume/runtime: the i8a8 canary of fused attention did not match the known answer exactly.\n` +
      `  ${formatAttentionI8a8Decision(decision)}\n` +
      `a8 attention on this device is not bit-identical with other machines (the difference is inside the sanity band,\n` +
      `at the level of rounding differences in the shared f32 epilogue). If you need bit identity, set attentionCompute to 'f32' or\n` +
      `'f16'.`,
  );
};

export type SessionState = {
  readonly gpu: GpuContext;
  /**
   * The graph to execute. MUST: do not hold the supply side (the opened container) as a whole —
   * holding the source's bytes pins an ArrayBuffer of the whole distribution file for the Session's
   * lifetime, and the "let go of references" contract of per-part sequential consumption (ADR 0108
   * decision 9) no longer holds. After construction only the graph is needed.
   */
  readonly graph: IrGraph;
  /**
   * This Session's surface, with a usage record, onto the device-lifetime pipeline cache (owned by
   * GpuContext). **The cache itself is not Session-resident** — Sessions on the same device share one.
   */
  readonly cache: SessionPipelines;
  readonly scheduler: SubmitScheduler;
  /**
   * The limits of intermediate-buffer planning (the device's granted values — ADR 0093 decision 1).
   * The plan looks only at values that do not change after Session construction, so they are fixed
   * here once.
   */
  readonly transientLimits: TransientLimits;
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  /**
   * The residency classes of the weights (the result of the prepare phase's pure function —
   * `planWeightResidency` in {@link "./weight-residency.ts"}).
   * Kept after construction so that `Session.exportWeight` in {@link "./executor.ts"} can name the
   * residency class.
   */
  readonly residency: ReadonlyMap<string, WeightResidency>;
  /**
   * The weights this Session borrows (the objects of {@link SessionOptions.sharedWeights}).
   * Where borrows are returned on dispose; **the lender's survival is guaranteed by this count**.
   */
  readonly sharedWeights: readonly SharedWeight[];
  /**
   * The content-addressed cache of params buffers (key = usage + all elements concatenated —
   * `RecipeBuilder.#writeParams`). The objects are Session-resident buffers owned by the weights
   * arena; this holds only the index "content → buffer already uploaded".
   * MUST: not at module scope (the zero-side-effect invariant — each Session has its own device and
   * buffers).
   */
  readonly paramsCache: Map<string, GPUBuffer>;
  /**
   * Resolved bindings → derived execution plan (LRU, bounded by `PREPARED_PLAN_CAPACITY` in
   * {@link "./executor.ts"}).
   * MUST: not at module scope (the zero-side-effect invariant — each Session has its own graph and
   * resident buffers, and the recipes fold those objects in as direct references).
   */
  readonly prepared: Map<string, PreparedPlan>;
  /**
   * Weights resident in compressed form (residency class and attached objects — ADR 0018 / 0019 /
   * 0069). A value not in it is read as f32 — kernel-variant selection and extra bindings are both
   * decided by this one table.
   *
   * MUST: do not split residency class, scale and group length into parallel Maps (the doc of
   * {@link ResidentWeight}). It is filled in one place only, Session construction, and the group
   * length is copied from the declaration (graph) — a form that allows passing another value becomes
   * the silently wrong value "an asset with group 64 runs on a group-32 pipeline".
   */
  readonly residentWeights: ReadonlyMap<string, ResidentWeight>;
  readonly storage: StorageDiagnostics;
  /** The cost breakdown of the construction phase ({@link SessionBuildStats}). Fixed when construction settles, immutable afterwards. */
  readonly buildStats: SessionBuildStats;
  /** The execution form of linear (opt-in — {@link SessionOptions.linearCompute}). */
  readonly linearCompute: "f32" | "a8" | "f16";
  /** The execution form of conv3d (opt-in — {@link SessionOptions.convCompute}). */
  readonly convCompute: "f32" | "a8";
  /** The budget for holding slot backings at the same time ({@link SessionOptions.planBackingBudgetBytes}). */
  readonly planBackingBudgetBytes: number;
  /** The execution form of fused attention (opt-in — {@link SessionOptions.attentionCompute}). */
  readonly attentionCompute: ComputePrecision;
  /** The storage form of S (opt-in — {@link SessionOptions.attentionScoreStorage}). An axis orthogonal to the compute form. */
  readonly attentionScoreStorage: ScoreStorage;
  /**
   * The device facts the online attention kernel's route reads (subgroups and the workgroup-storage
   * limit — ADR 0132 decision 5). Read once here from the device's enabled features and granted
   * limits; `"none"` adds no construction-time rejection (the route falls back per node).
   * MUST: not re-read after construction (the prepared plans' keys assume one route per shape — the
   * same Session-level invariant as {@link geometryProfile}).
   */
  readonly attentionDevice: AttentionDeviceFacts;
  /**
   * The reduction form of states-form attention ①QK / ③PV (opt-in —
   * {@link SessionOptions.stateAttentionReduce}). Even with `"parallel"`, **①' is chosen only for
   * plans with M ≤ 8** (decode and speculative verify — prefill plans stay on ①), and **③' only for
   * plans with M < 16** (M ≥ 16 takes the tiled path, bit-identical with ③ₜ = ③ regardless of the
   * option — a single option value).
   */
  readonly stateAttentionReduce: StateAttentionReduce;
  readonly linearGemvReduce: LinearGemvReduce;
  readonly rmsNormReduce: RmsNormReduce;
  /**
   * The parallelism target of the row-block GEMV (opt-in —
   * {@link SessionOptions.linearGemvRowsThreadTarget}). Omitted (`undefined`) is the kernel-side
   * default = the saturation point of the reference device.
   */
  readonly linearGemvRowsThreadTarget: number | undefined;
  /**
   * The i8a8 integer dot-product variant **of linear**. The default is decided from the enumeration
   * of `navigator.gpu.wgslLanguageFeatures`, and tests can force it with {@link I8A8_DOT}. **Either
   * way the numbers do not change by a single bit** (for linear this has not been refuted on real
   * hardware, Metal included — docs/known-issues.md).
   */
  readonly linearI8a8Dot: I8a8Dot;
  /**
   * The i8a8 integer dot-product variant **of fused attention** (①QK / ③PV). The default is decided
   * by a per-device run-time canary (src/gpu/attention-dp4a-canary.ts — the primary decision is the
   * comparison with the known answer), and tests can force it with {@link I8A8_DOT}. It was split
   * from linear because it was initially inferred that "both variants are bit-identical" broke for
   * attention only, on Apple M2; that inference was withdrawn on 2026-08-29 (on M2 too both variants
   * are bit-identical — what differed was 1 ULP of the shared f32 epilogue, docs/known-issues.md).
   * The field and the canary stay (the rationale is in the canary's head doc).
   */
  readonly attentionI8a8Dot: I8a8Dot;
  /** Forcing the number of row blocks (test only — {@link ROW_BLOCK_SPLIT}). */
  readonly rowBlockSplit: number | undefined;
  /**
   * The budget per row-block temporary ({@link SessionOptions.attentionRowBlockMaxBytes} — ADR 0124).
   * `undefined` = no budget (the binding limit stays).
   * MUST: not changed after construction (the key of a prepared plan is only the resolved bindings,
   * and plans are looked up assuming the block count implied by this value — a Session-level
   * invariant like {@link rowBlockSplit}).
   */
  readonly attentionRowBlockMaxBytes: number | undefined;
  /**
   * The GEMM geometry profile (the table injected through `acquireGpu`, or absent that a static table
   * chosen once from the adapter at construction — DECIDED: ADR 0115).
   * MUST: not re-chosen after construction (that the prepared plans and pipeline keys of one Session
   * correspond to a single table is the basis for looking up the plan cache without the profile).
   */
  readonly geometryProfile: GeometryProfile;
  readonly fuseRmsNormAdd: boolean;
  readonly fuseLinearStaticQuantize: boolean;
  /** Fuses gelu_tanh → mul → static_quantize into one dispatch (perf-ledger K-48). */
  readonly fuseGeluMulStaticQuantize: boolean;
  /** Passes the activations of fixed SRQ to the parallel GEMV as packed int8 (ADR 0105). */
  readonly packedStaticQuantize: boolean;
  /**
   * Runs the eligible decode linears on the integer GEMV (`SessionOptions.integerGemv`; true only
   * together with {@link packedStaticQuantize} — the entry check).
   */
  readonly integerGemv: boolean;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly dtypes: ReadonlyMap<string, IrDtype>;
  readonly outputNames: ReadonlySet<string>;
};

/**
 * The copy of {@link SessionOptions} taken at the entry of construction (the MUST at the head of
 * {@link buildSessionState}).
 *
 * MUST: fields are copied one by one with ordinary property reads (no spread). Spread copies only own
 * enumerable fields, so non-enumerable fields and fields from the prototype drop silently, and a
 * value that would have been rejected before the copy turns into the default and passes. The type is
 * a mapping with every field required, so adding a field to SessionOptions makes this a type error
 * (no form where a forgotten field silently becomes the default).
 * submitPolicy is copied with its contents (the scheduler is built after an await). The objects of
 * sharedWeights are borrowed (they are read in the synchronous section before the await).
 */
const snapshotSessionOptions = (
  requested: SessionOptions,
): { readonly [K in keyof Required<SessionOptions>]: SessionOptions[K] } => ({
  submitPolicy:
    requested.submitPolicy === undefined ? undefined : copySubmitPolicy(requested.submitPolicy),
  linearCompute: requested.linearCompute,
  convCompute: requested.convCompute,
  attentionCompute: requested.attentionCompute,
  attentionScoreStorage: requested.attentionScoreStorage,
  stateAttentionReduce: requested.stateAttentionReduce,
  linearGemvReduce: requested.linearGemvReduce,
  fuseRmsNormAdd: requested.fuseRmsNormAdd,
  fuseLinearStaticQuantize: requested.fuseLinearStaticQuantize,
  fuseGeluMulStaticQuantize: requested.fuseGeluMulStaticQuantize,
  packedStaticQuantize: requested.packedStaticQuantize,
  integerGemv: requested.integerGemv,
  rmsNormReduce: requested.rmsNormReduce,
  linearGemvRowsThreadTarget: requested.linearGemvRowsThreadTarget,
  planBackingBudgetBytes: requested.planBackingBudgetBytes,
  attentionRowBlockMaxBytes: requested.attentionRowBlockMaxBytes,
  sharedWeights: requested.sharedWeights,
  [I8A8_DOT]: requested[I8A8_DOT],
  [ROW_BLOCK_SPLIT]: requested[ROW_BLOCK_SPLIT],
});

/**
 * The body of `Session.build` in {@link "./executor.ts"} (the construction transaction itself). The
 * contract and the MUSTs are owned by the doc of `Session.build` on the facade side.
 *
 * It returns only the {@link SessionState} passed to `new Session(state)` — `Session` has a private
 * constructor, so the object is not created here.
 */
export const buildSessionState = async (
  gpu: GpuContext,
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  batches: AsyncIterable<WeightBatch>,
  requested: SessionOptions,
): Promise<SessionState> => {
  // MUST: options are copied once at the entry, and every later check, capability check and state
  // creation reads from the copy. Construction spans awaits (the canary, the weight supply), so
  // re-reading the caller's object would let a modification made during the wait reach the state
  // without passing the combination check at the entry (ADR 0105 decision 1) (how it is copied:
  // {@link snapshotSessionOptions}).
  const options: SessionOptions = snapshotSessionOptions(requested);
  // MUST: construction itself checks that the device is alive twice, at the entry and right before
  // settling (the end of the try below). Waiting for the weight upload to complete turns a loss into
  // an exception, but a graph that uploads no weight (no initializers, or all shared) never reaches
  // that wait, and a Session tied to a lost device would return as a success.
  assertDeviceUsable(gpu, "Session construction");
  // MUST: every check that does not depend on GPU capabilities (type, spelling, combination, range)
  // runs first through the single {@link sessionOptionsViolation}. The models-side composition
  // (explicit > declared > default) borrows the same decision before fetching weights, so adding a
  // condition here would split the accepted sets of the two entries.
  const violation = sessionOptionsViolation(options);
  if (violation !== undefined) throw new ExecutionError(violation);
  const {
    rmsNormReduce,
    linearCompute,
    convCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
    planBackingBudgetBytes,
  } = executionKnobDefaults(options);
  const linearGemvRowsThreadTarget = options.linearGemvRowsThreadTarget;
  // From here on, only checks that look at GPU capabilities (features / WGSL language features).
  if (
    rmsNormReduce === "subgroup32" &&
    (!gpu.features.has("subgroups") ||
      !gpu.features.has("subgroup-size-control") ||
      !gpu.wgslLanguageFeatures.has("subgroup_id"))
  ) {
    throw new ExecutionError("rmsNormReduce: subgroup32 requires acquireGpu({ subgroups: true })");
  }
  if (
    linearGemvReduce === "parallel-subgroup32" &&
    (!gpu.features.has("subgroups") ||
      !gpu.features.has("subgroup-size-control") ||
      !gpu.wgslLanguageFeatures.has("subgroup_id"))
  ) {
    throw new ExecutionError(
      "linearGemvReduce: parallel-subgroup32 requires acquireGpu({ subgroups: true })",
    );
  }
  // MUST: when f16 compute is requested on a device without the feature, **fail here**. Falling back
  // silently to the f32 path would make the distinction between the default path and the opt-in
  // invisible in both the diagnostics and the numbers (the same reason as ADR 0025 decision 1).
  if ((linearCompute === "f16" || attentionCompute === "f16") && !gpu.shaderF16Enabled) {
    throw new ExecutionError(
      "an f16 compute variant was requested, but the device does not enable 'shader-f16'" +
        ` (linearCompute: ${linearCompute} / attentionCompute: ${attentionCompute}). ` +
        "Acquire the device again with acquireGpu({ shaderF16: true })" +
        " (a feature can only be requested when the device is created)",
    );
  }
  // The GEMM geometry profile (DECIDED: ADR 0115). If a table was injected through
  // `acquireGpu({ geometryProfile })`, that one (its check already passed before the device was
  // created; `match` is not looked at); otherwise it is decided by the adapter's (vendor,
  // architecture, description) and the embedded static tables (`adapterInfo` is passed as is, so
  // `description` takes part in matching too; an embedded table without `match` is used only by
  // injection). Neither is re-chosen during execution. A broken embedded table or a tie of equal rank
  // fails **before a single byte of weights is uploaded** (not in a form that goes unnoticed until a
  // run that hits that op).
  // MUST: the selection happens in this one place. The dp4a canary (`resolveAttentionI8a8Dot`) and
  // the diagnostic id (`SessionState.geometryProfile`) use this return value — choosing separately
  // could make the geometry the canary verified differ from the geometry actually run.
  const geometryProfile = sessionGeometryProfile(gpu);

  // MUST: before allocating weights, compare the allocation size of each residency class with the
  // device's absolute limits (before the batch loop = before a single byte is uploaded). Detecting
  // allocation failure is the job of the per-item (block) errorScope (ADR 0108 decision 9), but that
  // depends on the implementation's reporting quality (devices whose out-of-memory scope stays silent
  // exist — the Metal section of docs/known-issues.md), and even when caught it shows only after
  // several GiB were transferred. The sizes are fixed by the declaration alone (the residency plan is
  // a pure function of the prepare phase), so what can fail deterministically fails here (the
  // "explicit size check" named by the same known-issues entry).
  // NOTE: only absolute limits are looked at, never free VRAM (the rule of ADR 0070 decision 5 — the
  // check stays a pure comparison, and the final check of whether the total fits stays with the
  // errorScope).
  assertWeightsWithinLimits(residency, gpu.limits);

  // Matching of shared initializers (borrowed weights — ADR 0096 decision 2 and its stage 2
  // addendum). **Before a single byte is uploaded**, check the residency class, declared shape,
  // semantic dtype, the lender's codec and the device (the check itself is `resolveSharedWeights`).
  const shared = resolveSharedWeights(graph, residency, gpu, options.sharedWeights);

  // The integer dot-product variant is **a separate field for linear and attention**
  // ({@link SessionState}). For both, `I8A8_DOT` takes precedence, and without it each family falls
  // to its own default.
  const dp4a = dp4aAvailable(gpu.wgslLanguageFeatures);
  // When a8 is requested on a device without dp4a, report it (once per device). Do not fail, and do
  // not silently fall to f32 or to a sibling preset without a8 (DECIDED: ADR 0128 decision 5) — the
  // numbers are the same and the user loses only speed, so show the fact and let them choose again.
  const emulationWarning = a8EmulationWarning(
    linearCompute,
    convCompute,
    attentionCompute,
    gpu.wgslLanguageFeatures,
  );
  if (emulationWarning !== undefined) {
    gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, emulationWarning);
  }
  const attentionI8a8Dot = await resolveAttentionI8a8Dot(
    gpu,
    options[I8A8_DOT],
    attentionCompute,
    dp4a,
    geometryProfile,
  );

  const scheduler = new SubmitScheduler(gpu, options.submitPolicy);
  const weights = new RunArena(gpu.device, () => scheduler.flush());
  const weightBuffers = new Map<string, GPUBuffer>();
  const residentWeights = new Map<string, ResidentWeight>();
  // The i8 weights the conv3d w8a8 kernel reads tap-major (ADR 0133 decision 4) — reordered at
  // upload, flagged on the resident record, and never read by the f32 kernel.
  const tapMajorWeights =
    convCompute === "a8" ? conv3dTapMajorInitializers(graph) : new Set<string>();
  /**
   * The **copy** of the companion scale carried along a piece list of the expanded class (key =
   * initializer name).
   *
   * MUST: hold a copy of the values, not a view. The scale's object exists only in the same part as
   * piece 1 (rule ③), and what the source returns may be a view into a container (the scale block for
   * seek-type sources, the part's retention slot for hub's scan type). Held as a view, that container
   * would survive until the end of the list (the last piece), breaking the block-granularity bound
   * that keeps the host RAM of weights to one item. Only the scale is copied; not a single byte of the
   * weight body.
   */
  const carriedScales = new Map<
    string,
    { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] }
  >();
  let residentCompressedBytes = 0;
  let hostExpandedBytes = 0;
  // The cost breakdown of the construction phase ({@link SessionBuildStats}). An aggregator ticking
  // with the host clock only;
  // MUST NOT: add GPU fences for measurement or move the position of submits (breaking the contract
  // of one submit per batch, ADR 0108 decision 9, pushes the momentary peak up by one weight). So the
  // real transfer time of writeBuffer stays absorbed in uploadFenceMs.
  let partCount = 0;
  let supplyWaitMs = 0;
  let decodeMs = 0;
  let bufferCreateMs = 0;
  let writeBufferIssueMs = 0;
  let uploadedBytes = 0;
  let uploadFenceMs = 0;
  // The measurement wrapping has several call sites for each of the 3 paths (decode / createBuffer /
  // writeBuffer), so it is folded into local helpers. **Not a single call order or argument changes**
  // (measurement is a thin pass-through layer).
  const timedDecode = (decode: () => Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
    const start = performance.now();
    const expanded = decode();
    decodeMs += performance.now() - start;
    return expanded;
  };
  /** The host-side reorder of a conv3d w8a8 weight counts as decode time (CPU work on the bytes). */
  const timedReorder = (reorder: () => Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
    const start = performance.now();
    const reordered = reorder();
    decodeMs += performance.now() - start;
    return reordered;
  };
  const timedAlloc = (bytes: number): GPUBuffer => {
    const start = performance.now();
    const buffer = weights.allocHostWritten(bytes, HOST_WRITTEN_USAGE);
    bufferCreateMs += performance.now() - start;
    return buffer;
  };
  // The destination offset exists for piece lists (split tensors) — the whole path always passes 0,
  // and a piece passes "row offset × byte length of one row" to append into one buffer.
  const timedWrite = (
    buffer: GPUBuffer,
    data: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    offset: number,
  ): void => {
    const start = performance.now();
    gpu.device.queue.writeBuffer(buffer, offset, data);
    writeBufferIssueMs += performance.now() - start;
    uploadedBytes += data.byteLength;
  };
  /** Shared initializers whose borrow has been added (on construction failure they are returned one by one from here). */
  const borrowed: SharedWeight[] = [];
  try {
    // Borrowed weights enter the ledger without reading a single block (the lender has already
    // uploaded the bytes to the GPU). The borrow count is added first to close the window in which
    // the lender could be disposed during construction.
    for (const { name, shared: weight } of shared) {
      const internals = weight[RUNTIME_INTERNAL];
      internals.retain();
      borrowed.push(weight);
      weightBuffers.set(name, internals.buffer);
      // Compressed classes (f16 / i8 / i4) take over the lender's attached objects (scale, group
      // length) as well. A name not entered here is read as f32 (the default of the weight ledger),
      // so the residency-class match acts as the check.
      if (internals.resident !== undefined) residentWeights.set(name, internals.resident);
    }
    // The iteration wait (= the cost of the supply side) is hidden by for await, so it is measured as
    // the difference from **the time the previous one finished processing** (the time until the next
    // arrives lies only between these two points). There are two levels of waiting — the next element
    // of the part list (batch) and the next block within a part (item — reading and verifying the
    // block fall here). Both are added on the same clock ({@link SessionBuildStats.supplyWaitMs}).
    let supplyBoundary = performance.now();
    for await (const batch of batches) {
      supplyWaitMs += performance.now() - supplyBoundary;
      partCount += 1;
      // MUST: the errorScope and the fence use the same label (built separately, a failure of the
      // same upload interval would appear under two names).
      const label = uploadLabel(batch.origin);
      /** Decode and upload of one initializer (whole / one piece). Called inside the synchronous section. */
      const uploadItem = (item: ReadyInitializer): void => {
        const name = item.name;
        const initializer = graph.initializers[name];
        const raw = item.payload;
        // The planner is the source of truth for the residency class (the contract lists every
        // initializer — a gap is a broken bookkeeping).
        const seat = residency.get(name);
        if (seat === undefined) {
          throw new ExecutionError(`initializer '${name}': no residency class`);
        }
        // MUST: an object arriving for a borrowed class fails. Shared initializers are outside the
        // matched set (supply side), so they never appear in `items` — if one does, the bookkeeping is
        // broken, and letting it through would overwrite with other bytes a name that points at the
        // lender's buffer.
        if (seat.seat === "shared" || initializer.storage === undefined) {
          throw new ExecutionError(`initializer '${name}': declared shared but has a tensor`);
        }
        const storage = initializer.storage;
        const layout = codecLayout(storage.codec);
        // An initializer's declared shape is numeric only (guaranteed by the parser — symbolic dimensions are rejected).
        const declaredShape = graph.values[name].shape.map(Number);
        const declaredRows = declaredShape[0];
        // A split tensor (piece list) arrives as "contiguous ranges of the leading dimension". The
        // shape passed to decoding is the piece's shape, and the byte position and length follow from
        // a proportional share of **the bytes per row derived from the declaration** (the length per
        // row divides out of the declaration — the supply side has matched the remaining dimensions of
        // the shape, so only the row count changes per piece).
        const piece = item.piece;
        const pieceShape =
          piece === undefined ? declaredShape : [piece.rows, ...declaredShape.slice(1)];
        const rows = pieceShape[0];
        // MUST: fail when the byte length derived from the declaration and the real object disagree.
        // The planner (and the estimate) count from the declaration alone without looking at real
        // tensors, so this is the only point that actually verifies "declaration = real object" (it
        // never fires when the supply side's matching check holds — a second net).
        const expectedBytes =
          piece === undefined ? seat.payloadBytes : rows * (seat.payloadBytes / declaredRows);
        if (raw.byteLength !== expectedBytes) {
          throw new ExecutionError(
            `initializer '${name}': ${expectedBytes} bytes from the declaration, but the actual tensor has ${raw.byteLength} bytes`,
          );
        }
        // The destination byte position (always 0 when whole). The raw-byte classes put the stored
        // bytes on the GPU and the expanded class the bytes after f32 decoding, so the total length
        // that the proportional share is based on differs by class.
        const wholeBytes = seat.seat === "expanded" ? seat.expandedBytes : seat.payloadBytes;
        const byteOffset = piece === undefined ? 0 : piece.rowOffset * (wholeBytes / declaredRows);
        // The trailing zero padding may be applied only "whole" and to "the last piece of a list".
        // Applied to a middle piece, the padding overwrites the first bytes of the next piece with 0
        // (that a middle piece is 4-byte aligned is the supply side's job — this side keeps the
        // invariant by not padding).
        const tailAligned = piece === undefined || piece.last;
        /**
         * The scale read by the expanded class (f32 on the CPU) (rank-2 group form) — for a piece list
         * only the piece's row range is returned. The object exists only on piece 1, so its values are
         * copied there and carried to the end of the list (the MUST of {@link carriedScales}).
         */
        const expandedScale = (): {
          readonly values: Float32Array<ArrayBuffer>;
          readonly shape: readonly [number, number];
        } => {
          if (piece === undefined) return scaleTensor(item, layout);
          if (piece.first) {
            const scale = scaleTensor(item, layout);
            carriedScales.set(name, { values: new Float32Array(scale.values), shape: scale.shape });
          }
          const carried = carriedScales.get(name);
          if (carried === undefined) {
            throw new ExecutionError(
              `initializer '${name}': the piece scale was not carried over from piece 1`,
            );
          }
          return scaleForPiece(carried, declaredRows, piece.rowOffset, rows);
        };
        // Only stored f16 / bf16 / i8 / i4 / i2 split into two paths (ADR 0018 / 0131 / 0019 / 0069 / 0097).
        // Eligible ones stay resident as raw bytes with dequantization inside the kernel (this is
        // what makes the VRAM saving happen at all); ineligible ones are decoded to f32 here
        // (correctness is kept, VRAM saving is zero). For other storage the raw bytes are the GPU
        // representation as is.
        let payload: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer> = raw;
        if (layout === "f16" || layout === "bf16") {
          if (seat.seat === layout) {
            // MUST: an odd element count is padded with 2 trailing zero bytes to 4-byte alignment.
            // writeBuffer rejects sizes that are not a multiple of 4 in validation (= execution
            // starts with empty weights). bf16 is also 2 bytes per element, so it uses the same
            // padding.
            payload = tailAligned ? alignF16Payload(raw) : raw;
            residentWeights.set(name, { storage: layout });
            residentCompressedBytes += payload.byteLength;
          } else {
            payload = timedDecode(() => (layout === "f16" ? decodeF16(raw) : decodeBf16(raw)));
            hostExpandedBytes += payload.byteLength;
          }
        }
        if (layout === "i8" || layout === "i2") {
          const rowAxis = storage.rowAxis ?? 0;
          if (seat.seat === "i8" || seat.seat === "i2") {
            // The scale is a single one covering the **whole** tensor before splitting, so its shape
            // match, allocation and upload happen only on piece 1 (the only object when whole). The
            // match is given the declared shape, not the piece's shape.
            if (piece === undefined || piece.first) {
              const scale = scaleTensor(item, layout);
              assertRowScale(name, declaredShape, scale.shape, seat.rowAxis);
              // MUST: the scale buffer also counts as "GPU-resident compressed" (the bytes actually held).
              residentCompressedBytes += scale.bytes.byteLength;
              const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
              if (scale.bytes.byteLength > 0) {
                timedWrite(scaleBuffer, scale.bytes, 0);
              }
              residentWeights.set(name, {
                storage: layout,
                scale: scaleBuffer,
                ...(tapMajorWeights.has(name) ? { layout: "conv3d-tap-major" } : {}),
              });
            }
            // The conv3d w8a8 layout is a permutation inside each row (= output channel), so a
            // piece of a row-split list is reordered on its own (declaredShape = [Cout, Cin, Kt, Kh, Kw]).
            const stored = tapMajorWeights.has(name)
              ? timedReorder(() =>
                  reorderConv3dWeightTapMajor(
                    raw,
                    rows,
                    declaredShape[1],
                    declaredShape[2] * declaredShape[3] * declaredShape[4],
                  ),
                )
              : raw;
            // MUST: a weight whose element count is not a multiple of 4 is padded with trailing
            // zeros to 4-byte alignment (the same reason as the 2-byte padding of f16 — writeBuffer
            // fails validation).
            payload = tailAligned ? alignI8Payload(stored) : stored;
            residentCompressedBytes += payload.byteLength;
          } else {
            const scale = expandedScale();
            const scaleShape = keepdimScaleShape(pieceShape, scale.shape[0], rowAxis);
            payload = timedDecode(() =>
              layout === "i2"
                ? decodeI2(raw, pieceShape, scale.values, scaleShape)
                : decodeI8(raw, pieceShape, scale.values, scaleShape),
            );
            hostExpandedBytes += payload.byteLength;
          }
        }
        if (layout === "i4") {
          // Eligibility is narrower than for f16 / i8: "only weight slots consumed by linear /
          // embedding / conv1d(groups==1)" (ADR 0069 decision 5 and its addendum — decode paths exist
          // only in the tile reads of the GEMM skeleton (B side for linear, A side for conv1d
          // igemm) and in the embedding kernel). Weight slots without a decode path (conv2d /
          // conv_transpose1d / conv1d with groups > 1) and shared i4 go to the CPU-decode fallback
          // (correctness is kept, VRAM saving is zero — the same design as the ineligible i8 case).
          // The planner has already decided.
          if (seat.seat === "i4") {
            // The payload needs no padding and is always 4-byte aligned — byte length = numel / 2,
            // and numel is a multiple of group_size (a power of 2 ≥ 16), so the byte length is a
            // multiple of 8 (ADR 0069 decision 2). A piece's length per row is a multiple of 8 for the
            // same reason.
            residentCompressedBytes += payload.byteLength;
            if (piece === undefined || piece.first) {
              const scale = scaleTensor(item, layout);
              // MUST: the scale buffer also counts as "GPU-resident compressed" (as for i8 — the bytes
              // actually held; keeps the meaning in line with the exporter's storage_breakdown and the
              // diagnostics).
              residentCompressedBytes += scale.bytes.byteLength;
              const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
              if (scale.bytes.byteLength > 0) {
                timedWrite(scaleBuffer, scale.bytes, 0);
              }
              // The group length is decided only in the one place that copies it from the declaration
              // (the planner) — a form that can pass it through another path becomes the silently
              // wrong value "an asset with group 64 runs on a group-32 pipeline".
              residentWeights.set(name, {
                storage: "i4",
                scale: scaleBuffer,
                groupSize: seat.groupSize,
              });
            }
          } else {
            // The range (power of 2 ≥ 16, divisibility) is guaranteed by the merge layer. Presence is
            // optional only in the type, so restate the check rather than skip silently (the same
            // convention as "stored i8 but no scale").
            const groupSize = storage.groupSize;
            if (groupSize === undefined) {
              throw new ExecutionError(`initializer '${name}': storage i4 but no groupSize`);
            }
            const scale = expandedScale();
            payload = timedDecode(() =>
              decodeI4(raw, pieceShape, scale.values, scale.shape, groupSize),
            );
            hostExpandedBytes += payload.byteLength;
          }
        }
        // The buffer is allocated once when whole / once at the head of a piece list. Even for pieces
        // the size is computed from the declaration for **the whole tensor** (splitting does not change
        // the GPU-side layout by a single byte — for the raw-byte classes the stored byte length
        // rounded up to 4 bytes = the trailing padding; for the expanded class the byte length after
        // f32 decoding).
        if (piece === undefined) {
          weightBuffers.set(name, timedAlloc(Math.max(4, payload.byteLength)));
        } else if (piece.first) {
          const aligned =
            seat.seat === "expanded"
              ? seat.expandedBytes
              : seat.payloadBytes + ((4 - (seat.payloadBytes % 4)) % 4);
          weightBuffers.set(name, timedAlloc(Math.max(4, aligned)));
        }
        const buffer = weightBuffers.get(name);
        if (buffer === undefined) {
          throw new ExecutionError(`initializer '${name}': piece 1 buffer is not in the ledger`);
        }
        if (payload.byteLength > 0) timedWrite(buffer, payload, byteOffset);
        // The carried scale is dropped once the list has been read to the end (only one list's worth is alive).
        if (piece?.last === true) carriedScales.delete(name);
      };
      // MUST: weight upload is also wrapped in an errorScope (the "errorScope always on" of ADR
      // 0004). A createBuffer over the limit returns an invalid buffer without a synchronous
      // exception, and writeBuffer into an invalid buffer / with a misalignment becomes a no-op
      // without even a warning, so without the wrapping execution starts with empty weights.
      // MUST NOT: await inside this interval. Keeping push through the issue of pop in one
      // synchronous section is the basis for preventing LIFO interleaving without taking a per-device
      // lock.
      // Interval granularity: per block (item) (ADR 0108 decision 9 — push / pop costs 1.81 µs each,
      // nearly free. The main cost is the fence, so the fence stays at once per batch = part). The
      // read of the next item (the iteration wait of `for await`) comes outside the interval.
      // Each item's pop pair is issued inside its interval but **awaited at the part fence**, in item
      // order (ADR 0108 addendum 13). Awaiting it per item serialises the supply behind a GPU-process
      // round trip on Dawn / Chromium (about −1.5 to −3 s per Wan2.2 request); the attribution stays
      // per item because the scope itself is still per item.
      // MUST: the error reported is the first failing item in item order — the one the per-item
      // await reported — and it wins over a host-side or supply failure of a later item, which the
      // per-item await never reached.
      const scopeOutcomes: Promise<ScopeOutcome>[] = [];
      // Early stop: once any item's scopes have settled as a failure the part is lost, so the loop
      // leaves before uploading the next item instead of allocating and writing the rest of the part
      // after an OOM. The failure is still reported in item order, after the loop.
      let failureSettled = false;
      const onScopeFailure = (): void => {
        failureSettled = true;
      };
      supplyBoundary = performance.now();
      try {
        for await (const item of batch.items) {
          // This item's pull wait is counted with the wait for the end of the list, below.
          if (failureSettled) break;
          supplyWaitMs += performance.now() - supplyBoundary;
          pushFailureScopes(gpu.device);
          try {
            uploadItem(item);
          } catch (cause) {
            // MUST: always pop the two pushed scopes and leave none behind (leftovers would swallow
            // later validation results into the wrong scope and make errors permanently invisible).
            // Disposal is held in one place by the outer transaction boundary.
            await discardFailureScopes(gpu.device);
            throw attributeToOrigin(batch.origin, cause);
          }
          scopeOutcomes.push(
            settleScopes(
              popFailureScopes(gpu.device, `${label} for initializer '${item.name}'`),
              onScopeFailure,
            ),
          );
          // The reference to the item ends with this iteration (releasing the CPU-side bytes does
          // not wait for the fence — the NOTE on the fence below).
          supplyBoundary = performance.now();
        }
      } catch (cause) {
        const earlier = await firstScopeFailure(scopeOutcomes);
        throw earlier.failed ? earlier.error : cause;
      }
      // The wait until the end of the list is known (the iteration after the last item) is also a
      // cost of the supply side.
      supplyWaitMs += performance.now() - supplyBoundary;
      // MUST: every item's scopes are settled before the fence and before the part counts as
      // uploaded — a failure that settles late still fails the build (no failure is dropped).
      // The wait is a wait for the GPU at the part's end, so it is counted with the fence.
      const scopesStart = performance.now();
      const failure = await firstScopeFailure(scopeOutcomes);
      uploadFenceMs += performance.now() - scopesStart;
      if (failure.failed) throw failure.error;

      // MUST: per batch (container part), issue **one real submit** and wait for completion (ADR 0108
      // decision 9). queue.writeBuffer allocates and accumulates staging and does not release it
      // until the submit completes — right after uploading several GiB of weights the VRAM enters the
      // first run still double-counted, and the first peak rises by one weight (measured +2.7GiB with
      // the f16 preset; docs/research/2026-08-08-vram-oom-misreport.md §4). With sequential
      // consumption this release is the very condition for the staging peak O(largest batch).
      // NOTE: the fence governs only the staging, not the CPU-side bytes. WebGPU's `writeBuffer`
      // copies `dataContents` at the call (the spec's content timeline — "a copy of the bytes held by
      // the buffer source"), so rewriting the original bytes after it returns does not change the
      // values on the GPU. Therefore an item is let go in the iteration that uploaded it and not held
      // until the fence (ADR 0070 decision 3 / ADR 0108 decision 9, addendum of 2026-09-24; pinned on
      // hardware by tests/gpu_write_buffer_copy_test.ts).
      // MUST NOT: substitute scheduler.flush(). With no pending dispatch, flush issues no submit (the
      // early return covers only the submit; waiting for completion, closing the measurement window
      // and collecting timestamps still happen). A completion wait without a submit does not release
      // staging (measured in research 2026-08-08 §4), so it cannot replace the explicit submit([]).
      // NOTE: what forbids onSubmittedWorkDone per submit is the run's hot path (the "Attribution of
      // measurements" section of submit.ts); this is once per batch and outside the window, so it
      // enters neither the estimate nor the wall clock.
      // NOTE: it is not wrapped in an errorScope because an empty submit involves neither allocation
      // nor validation (the pair wraps "intervals that allocate" — pushFailureScopes in
      // gpu/error-scope.ts). Moreover Session construction is outside GpuContext's scope lock, so a
      // scope spanning an await here would open a door to misattributing failures of concurrent
      // Sessions.
      gpu.device.queue.submit([]);
      // MUST: some implementations may never resolve onSubmittedWorkDone after a loss (measured in the
      // doc of raceCanaryDeviceLost), so it is raced — insurance that turns a hang into a failure.
      const fenceStart = performance.now();
      await gpu[RUNTIME_INTERNAL].raceDeviceLost(gpu.device.queue.onSubmittedWorkDone(), label);
      uploadFenceMs += performance.now() - fenceStart;
      supplyBoundary = performance.now();
    }
    // The check right before settling (the MUST at the entry). It sits inside the try so that a
    // failure lets the catch below destroy the owned weights and return the added borrows.
    assertDeviceUsable(gpu, "Session construction");
  } catch (cause) {
    // Transaction boundary (ADR 0108 decision 9): when a batch fails midway (a declaration violation,
    // an exception from the input sequence or a GPU error alike), the weights arena is destroyed
    // along with the uploaded weights and no partial Session is published.
    // MUST: a failure of the clean-up does not overwrite the main exception (the same rule as on the
    // run side). The cause lies on the main side, and replacing it with destroy's rejection (mainly a
    // device loss) erases the starting point of the investigation.
    await weights.destroy().catch(() => undefined);
    // MUST: always return the added borrows (failing to return them leaves the lending Session unable
    // to dispose forever).
    for (const weight of borrowed) weight[RUNTIME_INTERNAL].release();
    throw cause;
  }

  return {
    gpu,
    graph,
    // MUST: the pipeline cache borrows the single one owned by GpuContext (device lifetime —
    // `GpuContextInternals.pipelines`). Allocating a new one here would make every Session on the
    // same device pay again for compiling the same WGSL and resolving getBindGroupLayout.
    // MUST: it is borrowed **only at the settling point of construction**, and not a single pipeline
    // is looked up during the construction phase. The construction phase is outside GpuContext's
    // scope lock, so creating pipelines here would misattribute the errorScope of a concurrent
    // construction (the "Invariants of errorScope sections" at the head of GpuContext in
    // gpu/context.ts). All actual creation happens on the miss path of the first run (RecipeBuilder)
    // = inside the lock.
    cache: new SessionPipelines(gpu[RUNTIME_INTERNAL].pipelines()),
    scheduler,
    transientLimits: {
      maxBufferSize: gpu.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
      offsetAlignment: gpu.device.limits.minStorageBufferOffsetAlignment,
    },
    weights,
    weightBuffers,
    residency,
    sharedWeights: borrowed,
    paramsCache: new Map(),
    prepared: new Map(),
    residentWeights,
    storage: { residentCompressedBytes, hostExpandedBytes },
    buildStats: {
      partCount,
      supplyWaitMs,
      decodeMs,
      bufferCreateMs,
      writeBufferIssueMs,
      uploadedBytes,
      uploadFenceMs,
    },
    linearCompute,
    convCompute,
    attentionCompute,
    attentionScoreStorage,
    attentionDevice: attentionDeviceFacts(gpu),
    stateAttentionReduce,
    linearGemvReduce,
    linearGemvRowsThreadTarget,
    planBackingBudgetBytes,
    // Whether linear has the extension **affects only speed** (both variants return the same
    // integers), so it is decided once here as a path choice rather than a feature detection (the
    // docstring of src/kernels/linear-i8a8.ts).
    linearI8a8Dot: options[I8A8_DOT] ?? (dp4a ? "dp4a" : "emu"),
    // Attention decides not by enumeration but by **the run-time canary's verdict** (`attentionI8a8Dot`
    // above — the comparison with the known answer) (the original rationale, a mismatch between
    // variants on M2, has been withdrawn — the doc of SessionState.attentionI8a8Dot).
    attentionI8a8Dot,
    rowBlockSplit: options[ROW_BLOCK_SPLIT],
    attentionRowBlockMaxBytes: options.attentionRowBlockMaxBytes,
    geometryProfile,
    fuseRmsNormAdd: options.fuseRmsNormAdd ?? false,
    fuseLinearStaticQuantize: options.fuseLinearStaticQuantize ?? false,
    fuseGeluMulStaticQuantize: options.fuseGeluMulStaticQuantize ?? false,
    packedStaticQuantize: options.packedStaticQuantize ?? false,
    integerGemv: options.integerGemv ?? false,
    rmsNormReduce,
    useCounts: countUses(graph),
    dtypes: declaredDtypes(graph),
    outputNames: new Set(graph.outputs),
  };
};
