/**
 * The public API of Karume.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — acquire a GPU / open a model / query
 * capabilities / create a session / run / release / get diagnostics / handle failures by type.
 *
 * Only error types for "failures caused by the user's input or environment" are public. A broken
 * internal invariant of the runtime (arena reference counting, codegen determinism and the like) is
 * a bug in Karume, not something the user should branch on, so it arrives as a plain `Error` (with
 * its `name` kept).
 */

/**
 * The reader of Karume's own container (`krm` / `krg` — ADR 0108, docs/container-v1.md).
 * `openContainer` verifies the two documents against the outer expected hash before parsing them,
 * and joins the graph with the binding table (weight blocks are fetched when needed). `codecLayout`
 * maps a registered codec name to its decode path (`ternary` takes the same i2 path as `int2-off`),
 * and is the only reader models use when they branch on the nature of the storage.
 */
export { openContainer } from "./src/format/container/open.ts";
export type {
  AssetReader,
  BlockSource,
  ContainerInput,
  DescriptorExpectation,
  OpenedContainer,
} from "./src/format/container/open.ts";
/**
 * **The in-memory container** — a supply surface that passes bytes at hand (synthesised weights,
 * data read from another container) to Session construction without writing a container. It
 * returns the same two readers as `openContainer` ({@link BoundContainer}), so
 * `prepareContainer` / `createSessionFromContainer` go through the same single path for either
 * source.
 *
 * A `MemoryTensor` is either one whole tensor (`bytes`) or split into rows (`pieces` — a sequence of
 * readers), exclusively; the latter holds no bytes at all (references run out at the fence of each
 * part = RAM for one container).
 */
export { openMemoryContainer } from "./src/format/container/memory.ts";
export type {
  MemoryContainerInput,
  MemoryEncoding,
  MemoryPiece,
  MemoryTensor,
} from "./src/format/container/memory.ts";
/**
 * The common surface met by both sources (`krm` / in-memory container). It lives in the join layer,
 * not in a source's implementation.
 */
export type { BoundContainer } from "./src/format/container/bind.ts";
export { ContainerFormatError } from "./src/format/container/header.ts";
export { codecLayout } from "./src/format/container/codecs.ts";
export type { CodecLayout, CodecName } from "./src/format/container/codecs.ts";
export { DimError } from "./src/format/dims.ts";
export { IrError } from "./src/format/ir.ts";
/**
 * **The entry point for IR v2 declarations built on the host** (docs/ir-v2.md). A caller that
 * passes a small graph built on the spot without the exporter (CFG composition, an Euler update,
 * max selection, PLE gather) to {@link openMemoryContainer} runs **the rules decided by the graph
 * alone** through this — SSA, forward references, topological order, agreement of `requires.ops`
 * with the ops actually used, unknown keys, the canonical spelling of symbol names, orphan
 * declarations.
 *
 * This check is the same single check that the path opening a `krm` (`openContainer`) applies to
 * the container's declaration. Without it on the public surface, only callers of the in-memory
 * container could build a Session from a declaration that bypassed the checks, and a broken graph
 * would not fail until the join (declaration × supply) or execution.
 *
 * `initializers[name]` has exactly two forms: `{}` (has its own data) or `{ "shared": true }`
 * (borrows a weight of the lending Session) (`"shared": false` cannot be written — it declares the
 * same thing as an absent field, which would give two canonical forms). `states` and a node's
 * `states` may be omitted (an empty table is the default).
 *
 * NOTE: it takes **a plain value already passed through JSON.parse**; non-finite numbers from
 * `1e999` and nesting depth are not checked here — on the `krm` path the reader of the descriptor
 * document (`openContainer`) owns those two, and for a caller building on the host, the side
 * writing the literals owns them. A caller loading a JSON string from outside MUST check those two
 * itself before passing it (passing the plain return of `JSON.parse` straight through skips exactly
 * the checks that the `krm` path has).
 */
export { parseIrDeclarationValue } from "./src/format/ir.ts";
/**
 * The shape {@link parseIrDeclarationValue} returns (= a value of
 * `MemoryContainerInput["graphs"]`).
 */
export type { IrDeclaration } from "./src/format/ir.ts";
/**
 * Strict safetensors reader (with coverage, alignment and dtype checks). A surface for models to
 * read accompanying assets that are not Karume containers (for example Anima's rope base table —
 * the payload of the asset `rope_base` of the `transformer` container, role `rope-base` — ADR 0109
 * decision 4) through the same checks; providing a general-purpose loader is not the goal
 * (DECIDED: removing a duplicate implementation — ADR 0008 addendum 2026-08-05).
 */
export { parseSafetensors, SafetensorsError } from "./src/format/safetensors.ts";
export type { SafetensorsDtype, SafetensorsFile, TensorView } from "./src/format/safetensors.ts";

export {
  acquireGpu,
  BatchScopeError,
  GpuDeviceLostError,
  GpuFeatureError,
  GpuLimitError,
  GpuUnavailableError,
  readAdapterLimits,
  ResidentTensorError,
} from "./src/gpu/device.ts";
/**
 * Typed errors raised by the errorScope discipline (they live in `src/gpu/error-scope.ts` — the
 * public surface is unchanged).
 */
export {
  GpuInternalError,
  GpuOutOfMemoryError,
  GpuValidationError,
} from "./src/gpu/error-scope.ts";
/**
 * A trial allocation of free VRAM (uses the non-fatal OOM of `createBuffer` to see "whether this
 * amount fits right now"). Input for deciding, before a step with a large allocation, whether to
 * enter a path that loses the device on OOM (the staging of writeBuffer).
 */
export { fitsHeadroom } from "./src/gpu/headroom.ts";
/**
 * {@link acquireGpu} is the only entry point that constructs a GpuContext, so it is public as a type
 * only (the same form as Session). Public as a value, `new GpuContext(...)` could pass a device that
 * bypassed planRequiredLimits / assertLimitsGranted, and the net that detects missing limit
 * requests would have a hole.
 *
 * `ResidentTensor` (the fourth lifetime class, owned by the GpuContext) and `BatchScope` (an
 * enqueue section closed by a single fence) are public as types only for the same reason — the
 * only entry points are `GpuContext.createResident` / `GpuContext.beginBatch`, and constructing them
 * directly could bypass the errorScope checks (detection of allocation failures, holding the
 * section lock).
 */
export type {
  AcquireGpuOptions,
  BatchScope,
  DeviceLostHandler,
  GpuContext,
  RequiredLimits,
  ResidentData,
  ResidentTensor,
} from "./src/gpu/device.ts";

/**
 * GEMM geometry profiles (ADR 0115) expose their type and **the list of embedded profiles** — the
 * type is the shape of the table passed to `AcquireGpuOptions.geometryProfile` (the shape the
 * output of `tools/geometry-sweep` (removed in 0.15) satisfied). The list
 * (`BUILTIN_GEOMETRY_PROFILES`) holds every embedded table — the tables selected automatically by
 * the adapter's (vendor, architecture, description), and the injection-only tables without `match`
 * (never selected automatically) — as values an application looks up by id and passes to
 * `acquireGpu({ geometryProfile })` (ADR 0115 addendum decision 7). The list and each table are
 * deeply frozen (modifying them would silently change automatic selection for every Session) — to
 * inject a modified table, `structuredClone` it first and modify the clone.
 * Instead of a table, `geometryProfile` also accepts a synchronous callback
 * `(adapterInfo) => GeometryProfile | undefined` (ADR 0117 decision 6) — called once with the
 * information of the adapter the runtime actually obtained, it checks a saved table with
 * `geometryProfileMismatch` of `@karume/runtime/tune` and returns it (`undefined` means automatic
 * selection).
 * The default table and the selection function are not exported as values (they are internal
 * codegen vocabulary; tools import them from `src/`).
 */
export type { GemmRowsRule, GeometryProfile } from "./src/kernels/geometry-profile.ts";
export { BUILTIN_GEOMETRY_PROFILES } from "./src/kernels/geometry-profiles/index.ts";
export type { GemmGeometry } from "./src/kernels/gemm-geometry.ts";
export type { I8a8Geometry } from "./src/kernels/i8a8-geometry.ts";

export type { ArenaStats } from "./src/gpu/arena.ts";
export { DEFAULT_SUBMIT_POLICY, SubmitPolicyError } from "./src/gpu/submit.ts";
/**
 * The GPU time breakdown (ADR 0021) is public as types only. Aggregation happens inside the
 * SubmitScheduler, and the user-facing surface is the single field
 * `Session.diagnostics().lastRunTiming` (undefined when measurement is disabled).
 */
export type {
  ChunkBudgetStats,
  GpuTimingEntry,
  GpuTimingStats,
  SubmitPolicy,
  SubmitStats,
} from "./src/gpu/submit.ts";

export { capabilities, OpContractError } from "./src/ops.ts";
export type { RuntimeCapabilities } from "./src/ops.ts";
/**
 * Missing capability (an op / semantic dtype / attrs / storage that cannot run). The type that
 * `prepareContainer` throws, listing every case, **before fetching a single weight block** (the
 * two-step boundary of ADR 0070 decision 5).
 */
export { RuntimeSupportError } from "./src/ops/support.ts";

export { DispatchLimitError } from "./src/codegen/errors.ts";

export { createSessionFromContainer, prepareContainer } from "./src/runtime/executor.ts";
export { DEFAULT_PLAN_BACKING_BUDGET_BYTES } from "./src/runtime/session-types.ts";
/**
 * A pure function that decides the GPU-independent acceptance conditions of `SessionOptions` (types,
 * spellings, combinations, value ranges) with **the same single check** as Session construction. It
 * returns the violation message and leaves the error type to the caller — the entry point for upper
 * layers that compose a distribution's declarations with explicit options to fail, before fetching
 * weights, with the same check as construction (ADR 0058 addendum 2026-09-26).
 */
export { sessionOptionsViolation } from "./src/runtime/session-options-check.ts";
/**
 * The entry point of admission before fetching weights (ADR 0070 decision 5 / graph-first). A
 * two-step boundary that rejects "cannot run" from the graph declaration of the opened container
 * alone, estimates the memory needed, and turns it into a Session:
 * `prepareContainer(opened, graphName) → estimate() → createContainerSession(gpu)`.
 *
 * `PreparedModel` is public **as a type only** — the only entry point is {@link prepareContainer};
 * constructing it directly could create "a Session of a model that cannot run" that bypassed the
 * capability check and the IR contract checks (the same convention as `Session` / `GpuContext`).
 */
export type { PreparedModel } from "./src/runtime/executor.ts";
/**
 * The memory requirement estimator (ADR 0070 decision 5). A GPU-independent pure function that
 * reports only the "demand side" bytes per category — it does not compare with the free side or
 * decide feasibility (the final check is the out-of-memory errorScope).
 */
export { estimateGraphMemory } from "./src/runtime/estimate.ts";
export type {
  AdmissionReport,
  AdmissionScenario,
  AdmissionScenarioName,
  AdmissionScenarioSpec,
  EstimateOptions,
} from "./src/runtime/estimate.ts";
/**
 * The device facts the online attention kernel's route reads (ADR 0132 decision 5), from a
 * `GpuContext` — the same function Session construction uses, including the key-axis form the
 * device's geometry profile chooses (ADR 0132 addendum 2026-11-12). An estimate of a graph with
 * `attentionScoreStorage: "none"` requires them (`EstimateOptions.attentionDevice`); deriving them
 * any other way could estimate a route the Session does not take.
 */
export { attentionDeviceFacts } from "./src/runtime/attention-device.ts";
export type { AttentionDeviceFacts } from "./src/runtime/attention-route.ts";
/**
 * The weight residency plan — a pure function determined by the graph alone, and the second
 * argument of {@link estimateGraphMemory}.
 *
 * A caller that has finished loading (a models pipeline) keeps only the `IrGraph`, not the
 * `PreparedModel`, by discipline, so the `PreparedModel.estimate` entry point is not available. These
 * two (`estimateGraphMemory(graph, planWeightResidency(graph), options)`) are the only way back to
 * the same estimate from there; the same graph gives the same plan (a pure function).
 *
 * Take the graph from `PreparedModel.graph` (a validated graph that went through the same checks as
 * construction) — called this way, the residency plan runs before the estimate's checks, so with an
 * unvalidated graph the failure message does not name the node.
 */
export { planWeightResidency } from "./src/runtime/weight-residency.ts";
export type { WeightResidency } from "./src/runtime/weight-residency.ts";
/**
 * Session construction has only two entry points, `createContainerSession` of
 * {@link prepareContainer} and its composition {@link createSessionFromContainer}, so it is public
 * as a type only.
 *
 * `Tensor` is a union discriminated by semantic dtype (ADR 0009's partial revision of ADR 0008):
 * `{ dtype: "f32", data: Float32Array }` / `{ dtype: "i32", data: Int32Array }` /
 * `{ dtype: "bool", data: Uint32Array }` (bool is a u32 holding 0 / 1). Symmetric for inputs and
 * outputs.
 *
 * Input lifetime is part of the surface: `run` / `enqueue` **copy, in the synchronous section of the
 * call**, the member set of the `inputs` Record, the shape of each input and `bindings`, but they do
 * not copy `Tensor.data` — they **borrow** it.
 * MUST NOT: modify `data` until the returned Promise settles (it produces silently wrong values).
 * Details in the "input lifetime" section of `Session.run`.
 *
 * `GenerationContext` (the owner of the mutable state of one generation — ADR 0066) is type-only for
 * the same reason: the only entry points are `Session.createGenerationContext` and
 * `GenerationContext.fork`; constructing it directly could bypass the allocation errorScope and the
 * capacity check (the same convention as `ResidentTensor` / `BatchScope`).
 *
 * `SharedWeight` (an opaque reference to a lent weight — ADR 0096 decision 2 and its stage 2
 * addendum) is type-only for the same reason: the only entry point is `Session.exportWeight`;
 * constructing it directly could bypass the residency-class match and the borrow counting.
 *
 * `ExpertCache` (the exact expert cache of a Session built with `SessionOptions.expertCache` — ADR
 * 0146) is type-only too: the only entry point is `Session.expertCache`; its run, load and routing
 * types come with it.
 */
export type {
  ComputePrecision,
  EnqueueOptions,
  EnqueueRead,
  BidirectionalRows,
  ExpertCache,
  ExpertCacheStats,
  ExpertLoad,
  ExpertRouting,
  ExpertRun,
  FusionCounts,
  GenerationContext,
  GenerationContextSpec,
  GenerationRun,
  LinearGemvReduce,
  ParamsCacheStats,
  PlanBackingStats,
  PreparedPlanStats,
  RmsNormReduce,
  RunInput,
  RunInputs,
  RunOutputs,
  ScoreStorage,
  Session,
  SessionBuildStats,
  SessionDiagnostics,
  SessionOptions,
  SharedWeight,
  StateAttentionReduce,
  StateStorage,
  StateBackingStats,
  StorageDiagnostics,
  Tensor,
} from "./src/runtime/executor.ts";
/**
 * The range and order check of `GenerationContextSpec.chunkBuckets` (ADR 0066 addendum
 * 2026-08-17, item 10, buckets).
 *
 * A pure function that needs neither a GPU nor a Session is **deliberately** on the public surface
 * so that the side choosing the physical row count of the prefill shape (a models pipeline) does not
 * keep its own copy of the rule — with a copy, the set the context allows and the set the caller
 * chooses grow apart, and a mismatch surfaces far from its true cause, in the pre-run check of
 * `Session.run`. There is only one source of truth for the accepted set.
 */
export { assertChunkBuckets } from "./src/runtime/generation-context.ts";
export { ExecutionError } from "./src/runtime/plan.ts";
export type { SymbolBindings } from "./src/runtime/plan.ts";
