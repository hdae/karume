/**
 * The public type surface of the Session. The declaration-only public API types that used to sit
 * in the first half of executor.ts, and the unique symbols that serve as keys of
 * {@link SessionOptions}, were moved here (executor.ts re-exports them to keep its public surface).
 * MUST: do not import executor.ts from here (no circular imports).
 */

import type { IrDtype } from "../format/ir.ts";
import type { ArenaStats } from "../gpu/arena.ts";
import type { BatchScope, ResidentTensor } from "../gpu/device.ts";
import type { GpuTimingStats, SubmitPolicy, SubmitStats } from "../gpu/submit.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import type { FusionCounts } from "./fusion.ts";
import type { GenerationContext } from "./generation-context.ts";
import type { SymbolBindings } from "./plan.ts";
import type { SharedWeight } from "./weight-residency.ts";

type TensorOf<D extends IrDtype, A> = {
  readonly dtype: D;
  /** Concrete values, with every symbol already bound. */
  readonly shape: readonly number[];
  readonly data: A;
};

/**
 * A tensor discriminated by its semantic dtype (ADR 0009 — a partial revision of ADR 0008's public
 * surface).
 *
 * Every element type is 4 bytes, and **bool is a u32 holding 0 / 1** (WebGPU storage buffers have
 * no 1-bit type; this is the same convention as the GPU-side storage). Inputs and outputs use the
 * same shape.
 *
 * MUST NOT: **do not modify the `data` passed as an input until the Promise returned by `run` /
 * `enqueue` settles** (it is borrowed — the runtime does not copy it). The `writeBuffer` to the GPU
 * is issued after a microtask, not inside the synchronous section of the call, so a modification
 * produces a silently wrong value with no exception and no warning. `shape`, on the other hand, is
 * copied when the call is issued, so modifying it does not affect a run in flight (the full
 * contract is the "input lifetime" section of `Session.run`). A `Tensor` received as an output is
 * owned by the caller and is not subject to this constraint.
 */
export type Tensor =
  | TensorOf<"f32", Float32Array<ArrayBuffer>>
  | TensorOf<"i32", Int32Array<ArrayBuffer>>
  | TensorOf<"bool", Uint32Array<ArrayBuffer>>;

/**
 * One input of a run: a host array ({@link Tensor}), or a {@link ResidentTensor} bound while it
 * stays resident on the GPU (the fourth lifetime class).
 *
 * MUST: a resident input **never issues a `writeBuffer`** — its buffer is baked into the bind group
 * as is. The host therefore has neither its shape nor its dtype, and the only thing that can be
 * checked is **its size** (exactly the size of the declared shape). Symbolic dimensions are not
 * bound from a resident input, so unless another input or `bindings` determines them, the run
 * fails loudly ({@link bindSymbols}' `deferredInputs`).
 */
export type RunInput = Tensor | ResidentTensor;
/**
 * The inputs of a run (name → one input).
 *
 * **The member set of this Record is copied when `run` / `enqueue` is issued**, so replacing
 * members without awaiting the returned Promise leaves that run with the members it had at issue
 * time (shapes and `bindings` are fixed the same way). Only the metadata is copied; the `data` of a
 * {@link Tensor} stays borrowed — the full contract is the "input lifetime" section of
 * `Session.run`.
 */
export type RunInputs = Readonly<Record<string, RunInput>>;
export type RunOutputs = Readonly<Record<string, Tensor>>;

/**
 * The specification of one generation run (the third argument of Session.run, and
 * EnqueueOptions.generation).
 *
 * `queryLength` is the number of real tokens in this step (`1..chunkLength` for prefill, 1 for
 * decode). **`pastLength` is not passed** — the context owns the advance of the logical length,
 * which advances only on the final success of a run or a batch (ADR 0066 decision 6: no double
 * bookkeeping).
 */
export type GenerationRun = {
  readonly context: GenerationContext;
  readonly queryLength: number;
  /**
   * When the logical length advances (default `"immediate"` = the original behaviour — it
   * advances by `queryLength` rows when the run returns without an exception). With enqueue it
   * does not advance until the batch completes without an exception.
   *
   * `"deferred"` holds the advance back, and the **number of accepted rows** is committed later
   * with `GenerationContext.commit(rows)` (the verify shape of speculative decoding — how many
   * draft rows are accepted is known only after reading that run's output). While a commit is
   * pending, the next run and `rewind` are rejected, so there is still only one path that moves
   * the logical length (ADR 0066 decision 6: no double bookkeeping).
   */
  readonly commit?: "immediate" | "deferred";
};

/** The specification of {@link Session.enqueue}. */
export type EnqueueOptions = {
  /**
   * The batch to join ({@link GpuContext.beginBatch}). Its only fence is the settlement of this
   * batch.
   */
  readonly batch: BatchScope;
  /**
   * A run that uses conversation state. The usage reservation is held from issue until the final
   * settlement of the batch. The logical length is advanced or deferred on the success of
   * finish / finishAndRead, not when enqueue returns. The same context cannot be used again while
   * it is still uncommitted. Dispose the context / Session after finish.
   */
  readonly generation?: GenerationRun;
  /**
   * Explicit values of symbolic dimensions. A resident input is not a binding source, so symbols
   * that **only** a resident input carries are given here (the same meaning as `run`'s second
   * argument).
   */
  readonly bindings?: SymbolBindings;
  /**
   * Graph output → the resident tensor it is written to. A `copyBufferToBuffer` is recorded in the
   * same command stream **after** the dispatch sequence (no readback, no fence).
   *
   * MUST: the size must exactly match the declared shape (fail loudly). `enqueue` does not read
   * back, so an output not listed here is overwritten with its slot by the next enqueue of the same
   * signature and is lost.
   */
  readonly copyOutputs?: Readonly<Record<string, ResidentTensor>>;
};

/**
 * The return value of {@link Session.enqueueRead} — two settlements: admission and graph outputs.
 *
 * `admitted` has the same meaning as the return value of {@link Session.enqueue} (acceptance or
 * failure of the encode shows here too). `outputs` resolves only **after the batch settles** (a
 * failed batch rejects it with the same reason).
 * MUST NOT: await `outputs` before calling `finish` / `finishAndRead` (it does not resolve until the
 * batch closes, so that await never returns).
 */
export type EnqueueRead = {
  readonly admitted: Promise<void>;
  readonly outputs: Promise<RunOutputs>;
};

/**
 * The specification of {@link Session.createGenerationContext} (ADR 0066 decision 6).
 *
 * Fixes the slot capacity (the symbolic dimensions of `graph.states`) and `chunkLength`, and
 * allocates the physical storage. A context has no inputs at all, so its symbolic dimensions are
 * determined **only by the bindings given here** (states are not a binding source — ADR 0066
 * decision 2).
 */
export type GenerationContextSpec = {
  /**
   * Values of the symbolic dimensions that appear in the shapes of the state slots. A name not in
   * `graph.symbols` fails loudly. May be omitted for a graph whose capacity is determined by
   * numeric dimensions alone.
   */
  readonly bindings?: SymbolBindings;
  /**
   * The row count of the fixed-length prefill chunk (ADR 0066 decision 4 — a plan-time constant;
   * the last chunk is filled with padding). Decode has the fixed shape `queryLength = 1` and runs
   * independently of this value.
   *
   * Additional physical shapes that run short chunks without padding are
   * {@link GenerationContextSpec.chunkBuckets}.
   */
  readonly chunkLength: number;
  /**
   * Physical chunk row counts allowed as prefill shapes **in addition to** `chunkLength` (ADR 0066
   * decision 4 / addendum 2026-08-17, item 10, buckets). Each element is an integer from 2 to below
   * `chunkLength`, in **strictly ascending** order. Omitted or empty = nothing added (as before: one
   * prefill shape plus one decode shape).
   *
   * Padding a short prompt to `chunkLength` rows wastes the work of the pad rows (row-local
   * linear / pointwise / norm cost grows with the physical row count). With buckets declared, the
   * caller can pick, per chunk, the smallest bucket at or above `queryLength` as the physical row
   * count. The default is not lowered instead because a long prompt is fastest in one large chunk.
   *
   * MUST: the count is also "the number of PreparedPlans allowed to stay warm at once" (one
   * execution shape = one plan under its own key — the LRU of ADR 0042 decision 2). Too many, and
   * the decode hot path is silently evicted and falls back to re-derivation.
   */
  readonly chunkBuckets?: readonly number[];
  /**
   * **The context to borrow from** (ADR 0096 stage 2 addendum, borrowing context — the target's
   * generation context that the drafter reads).
   *
   * Allowed only when "every slot of the graph is external", and the converse is a MUST too (a
   * graph with external slots and no `borrow` fails loudly). The borrower allocates none of its
   * own slots and binds the external slots **by name** to the lender's slots. `bindings` are
   * inherited from the lender and **cannot be passed**, `chunkLength` is exactly 1 and
   * `chunkBuckets` cannot be declared (the borrower has a single execution shape, decode).
   *
   * A borrower's run takes the lender's run lease, so it is serialised with the lender's run /
   * commit / rewind, and it is rejected while the lender has a `pendingCommit` (the draft comes
   * after the commit).
   */
  readonly borrow?: GenerationContext;
};

/**
 * The integer dot-product variant (the w8a8 path). **Both return the same integers**, so this is a
 * choice of speed only (src/kernels/linear-i8a8.ts).
 */
export type I8a8Dot = "dp4a" | "emu";

/**
 * **A private, test-only surface** (not exported from mod.ts — it does not pollute the "thin
 * surface" of ADR 0008).
 *
 * **Forces** the i8a8 integer dot-product variant (**common to every i8a8 kernel** of linear and
 * fused attention — the given variant applies equally to both families). Running the
 * `dot4I8Packed` version and the emulated version on a machine with the extension and comparing
 * them at atol=0 is the only mechanical detector for the claim "the emulation is numerically
 * identical"; it is a per-Session knob rather than an environment variable so that one process can
 * run both.
 *
 * MUST: forcing has one meaning across families (even though the defaults differ per family, this
 * knob is not split). **The default** is decided separately per family — linear by enumerating the
 * language features, attention by a per-device run-time canary
 * (src/gpu/attention-dp4a-canary.ts). When this is given, the canary does not run (the test says
 * "run with this variant"; inserting an environment check would erase what was measured).
 */
export const I8A8_DOT: unique symbol = Symbol("karume.i8a8Dot");

/**
 * **A private, test-only surface** (not exported from mod.ts — the same convention as
 * {@link I8A8_DOT}).
 *
 * Forces the row-block count of decomposed attention (`rowBlockAttention` in
 * src/runtime/fusion.ts). The default count is determined statically from the device's
 * `maxStorageBufferBindingSize`, so **on a machine with headroom under the limit it is always 1**,
 * and the paths with two or more blocks (the row-window kernels, the full-write across blocks)
 * never run. A forced split is the only way to run those paths on real hardware and match a
 * one-block run at Uint32 level; it is a per-Session knob rather than an environment variable so
 * that one process can run both.
 *
 * MUST: a count that does not fit the limit fails loudly (it cannot be used to loosen it). When
 * {@link SessionOptions.attentionRowBlockMaxBytes} is given, the limit is the smaller of that and
 * the binding limit, and the forced count takes effect only within that range (ADR 0124 decision 6).
 */
export const ROW_BLOCK_SPLIT: unique symbol = Symbol("karume.rowBlockSplit");

/**
 * The compute precision knob per op family (ADR 0028 / attention i8a8: design §9.2). **A separate
 * axis from f16 weight storage (ADR 0018)**: `"f16"` selects the variant that drops the shared
 * tiles to f16 for the dot product (accumulating in f32), and `"a8"` selects the variant that
 * quantizes activations to per-token i8 and uses an integer dot product.
 *
 * MUST: the three values are **mutually exclusive** (not a product). The q/k/v of attention are all
 * activations with no storage axis, so the combination "f16 and a8" has nothing to describe.
 * MUST: `"f16"` comes with `acquireGpu({ shaderF16: true })` (fails loudly at Session construction).
 * **`"a8"` does not require `shader-f16`** (do not mix it into the feature check).
 *
 * NOTE: the value was renamed from `"i8a8"` to `"a8"` in 0.5.0 (ADR 0074 decision 3) — this knob
 * decides **only how activations are handled**; the storage form of the weights is decided by the
 * asset header. The kernel-side internal identifiers (`linear-i8a8.ts` / `:i8a8:` in pipeline keys)
 * name an execution variant and stay unchanged.
 */
export type ComputePrecision = "f32" | "f16" | "a8";

/**
 * The reduction form of ①QK / ③PV of states-form attention ({@link
 * SessionOptions.stateAttentionReduce}).
 */
export type StateAttentionReduce = "sequential" | "parallel" | "parallel-fused";

/** The value set shared by execution and the memory estimate. */
export const STATE_ATTENTION_REDUCES: Readonly<Record<StateAttentionReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-fused": true,
};

/** The K summation order of the quantized GEMV (ADR 0098). */
export type LinearGemvReduce = "sequential" | "parallel" | "parallel-subgroup32";

/**
 * The reduction method of RMS normalization. subgroup32 sums in a different order from the
 * reference (ADR 0100).
 */
export type RmsNormReduce = "workgroup" | "subgroup32";

export type SessionOptions = {
  /**
   * The submit time-budget policy (against TDR / watchdogs — ADR 0004). Default:
   * DEFAULT_SUBMIT_POLICY.
   */
  readonly submitPolicy?: SubmitPolicy;
  /**
   * The execution form of linear (default `"f32"` = the original behaviour).
   *
   * `"a8"` applies to **linears with integer-resident (i8 / i4) weights**: activations are
   * quantized to per-token i8 and an integer dot product is used. The knob means only "i8
   * activations + integer dot product"; **the storage form of the weights is a separate axis** — the
   * storage form splits it into two variants with different numeric contracts:
   * - i8 resident → **w8a8**: the whole reduction is one i32 (zero roundings in the integer part),
   *   and dequantization first folds `xs · wscale` into one f32 and then multiplies it into
   *   `f32(acc)` — **two roundings: that product and the final fma** (design =
   *   docs/research/2026-08-03-dp4a-w8a8-design.md. The rounding count is a property of an
   *   fma-fusing implementation — not a specification guarantee. See the MUST in
   *   src/kernels/linear-i8a8.ts).
   * - i4 resident → **w4a8**: an i32 dot product per group, flushed to f32 at each group boundary,
   *   so there are `k/g + 1` roundings (counted for the same fusing implementation). wscale changes
   *   per group and cannot be folded, so `xs` goes into **the final fma alone**. The numeric
   *   contracts differ in two ways — ① the reduction granularity (whole vs partial reductions per
   *   group = the rounding count depends on k and g) ② where `xs` is multiplied (ADR 0076 decision
   *   2). Filed as Q-8 in docs/perf-ledger.md; the contract is the "w4a8 variant" section of
   *   src/kernels/linear-i8a8.ts.
   *
   * `"f16"` is the compute variant that drops the shared tiles to f16 (ADR 0028) and applies to
   * linears with f32 / f16 weight storage (**it cannot be combined with i8 / i4 resident weights** —
   * w8a16 / w4a16 are not implemented, so this fails loudly).
   * MUST: the default is `"f32"` — running i8 / i4 / f16 assets in low precision automatically would
   * silently change the existing PNG sha256 checks and the E2E tolerances. Only opt-in is possible.
   */
  readonly linearCompute?: "f32" | "a8" | "f16";
  /**
   * The execution form of fused attention (the three kernels of ADR 0023; default `"f32"` = the
   * original behaviour).
   *
   * `"f16"` makes the shared tiles of ①QK / ②row statistics / ③PV f16 and **passes S in f16 as
   * well** (① writes it, ② and ③ read it — the transient is halved). It is **a separate axis** from
   * `linearCompute` because at 1024 px the split is attention 46% / linear 42%, and there are real
   * cases that want only one of them in f16.
   *
   * `"a8"` is the variant that quantizes q / k / v to i8 and uses an integer dot product
   * (design = docs/research/2026-08-04-attention-a8-design.md). **Its current semantics are "①QK
   * and ③PV are i8a8, ②row statistics stay f32"** (the A side of ③, P̃, has a scale that collapses
   * to 1/127 by construction, so it does not go through the quantization kernel; only V becomes a
   * per-column i8 through Vᵀ). Unlike `"f16"`, it does not require `shader-f16` and applies
   * independently of the storage form of the assets (f32 / f16 / i8) — attention's inputs are all
   * activations.
   * **Eligibility is decided per stage, and only a stage that does not qualify falls back silently
   * to the f32 path** (① needs `D % 4 == 0`, ③ needs `N % 4 == 0` — a word-boundary condition of
   * the i8 payload, whose packing direction differs per stage). So **with `D % 4 == 0` and
   * `N % 4 != 0` the result is the mix "①QK in i8a8, ③PV in f32"**. Same convention as linear's
   * `k % 4`: the fallback shows only in the diagnostic pipeline keys.
   * MUST: the default is `"f32"`.
   */
  readonly attentionCompute?: ComputePrecision;
  /**
   * **The storage form of the intermediate buffer S (scores)** of fused attention (default
   * `"f32"` = the original behaviour). **A second axis orthogonal to** `attentionCompute` (the
   * compute form): S is an intermediate buffer, so "which precision to compute in" and "which
   * precision to store in" can be chosen separately.
   *
   * `"f16"` packs S into `array<u32>` **with `pack2x16float`, two elements per word** (core WGSL,
   * **does not require `shader-f16`** — it combines with `attentionCompute: "a8"` while keeping ADR
   * 0030 decision 1, "i8a8 does not require shader-f16"; this combination is the main target). The
   * only rounding is the one at storage, and `unpack2x16float` on the read side is exact. The output
   * therefore matches, bit for bit, "**the f32 variant with S rounded to f16 on the host**".
   *
   * MUST: the default is `"f32"` — dropping the storage form of S automatically would silently
   * change the existing PNG sha256 checks and the E2E tolerances (the same reason ADR 0028 decision
   * 1 forbids an auto mode).
   * MUST: combining it with `attentionCompute: "f16"` (`:c16`) **fails loudly**. That option holds S
   * as `array<f16>`, **a different storage form**, so the combination is redundant and
   * contradictory.
   * **Eligibility is `D % 4 == 0 && N % 4 == 0`** (the condition for the writer ①QK to take the v4
   * path — one thread writes four consecutive columns = exactly two words, exclusively); other
   * shapes fall back **silently** to f32 storage (same convention as linear's `k % 4` and ADR 0030
   * decision 5: the fallback shows only in the diagnostic pipeline keys).
   */
  readonly attentionScoreStorage?: ScoreStorage;
  /**
   * The **reduction form of ①QK and ③PV** of states-form attention (ADR 0067 decision 4 — the form
   * that reads the KV slots of a generation context; default `"sequential"` = the reference path).
   * **One knob switches both stages together** (ruling of 2026-09-06 — no separate option per
   * stage).
   *
   * `"parallel"` is the variant in which **both stages split the reduction across 16 lanes of a
   * workgroup and fold it in a fixed-order tree**:
   * - ③PV splits along the KV length (perf-ledger K-12 — the "③' KV-parallel reduction variant"
   *   section of src/kernels/state-attention.ts). It removes the shape in which, for decode (M=1),
   *   one thread's sequential length grows with the KV length.
   * - ①QK splits along D of the dot product (perf-ledger K-14 — the "①' D-parallel reduction
   *   variant" section of the same file). It shortens the latency of one invocation summing D
   *   products sequentially.
   *
   * **The two stages apply to different shapes** (still one option): **①' is chosen only for plans
   * whose `M` (physical chunk row count) is 8 or less** (decode's M=1 and speculative verify's
   * M ≤ 8); a prefill plan (M > 8) does not choose ①' even when `"parallel"` is given. In prefill,
   * ① already fills rows × columns, and switching to ①' was measured to be slower (2026-09-06 — the
   * conditions and measurements are in the WHY of `stateQkParallelEligible`). The QK path splits
   * three ways by M: M ≤ 8 is ①', 9 to 15 is ① (sequential), and M ≥ 16 is ①ₜ regardless of the
   * option (K-row tile sharing on the GEMM skeleton, bit-identical to ① — `stateQkTiledEligible`).
   * **③' is chosen only for plans with `M < 16`**; `M ≥ 16` takes ③ₜ regardless of the option (V-row
   * tile sharing on the GEMM skeleton — perf-ledger K-13 step 2). ③ₜ is **bit-identical** to ③, so
   * even when the option is `"parallel"`, a prefill plan produces the values of the reference path
   * (a decode plan, M = 1, uses ③' as before).
   *
   * Both change the reduction order, so they are **not bit-identical to the reference path**
   * (determinism is kept). They are a separate family from fused attention (`attentionCompute`) and
   * orthogonal to it.
   * MUST: the default is `"sequential"` (ADR 0058 decision 2 — no automatic selection of a path that
   * changes the numbers).
   *
   * parallel-fused keeps parallel's summation order and fuses the row statistics with PV only for
   * states shapes with M<=8 and a column limit <=1024. readonly and other shapes use the parallel
   * path. ADR 0102.
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * The summation order of the quantized GEMV (default sequential). parallel is an optional setting
   * limited to the measured INT2/4/8 shapes and M=1..8. f32 compute only; other shapes and storage
   * use the original path (distinguished by the diagnostic key).
   * The summation order changes, so the generated sequence may differ from the default, including
   * with QAT re-quantization. M=1 and M=4/8 use the same summation order. No extra weight copy.
   * ADR 0098.
   * parallel-subgroup32 runs the same partition and summation tree as parallel through value
   * exchange within 32 lanes (ADR 0101).
   * Requires acquireGpu({ subgroups: true }). Without it the option is rejected, with no automatic
   * fallback to parallel.
   */
  readonly linearGemvReduce?: LinearGemvReduce;
  /**
   * Fuses an adjacent f32 rms_norm→add (default false, ADR 0099).
   * Applies only to last dimensions 256/1536/2560, identical shapes, and intermediate values with a
   * single consumer.
   * Rounding preservation has been measured, but bit identity with the reference is not guaranteed
   * on unverified GPUs.
   * submitPolicy is an independent setting; this option alone does not change the submission limit.
   */
  readonly fuseRmsNormAdd?: boolean;
  /**
   * Fuses an adjacent linear→static_quantize (default false, ADR 0103).
   * Supported only together with linearGemvReduce: parallel / linearCompute: f32.
   * Only the accepted INT2/4/8 shapes, M=1..8, and intermediate values with a single consumer. It
   * borrows the existing scale.
   * Bit identity on unverified GPUs is not guaranteed; false returns to the unfused reference.
   */
  readonly fuseLinearStaticQuantize?: boolean;
  /**
   * Passes fixed-SRQ activations to the parallel GEMV as packed int8 (four int8 codes in one u32
   * word) (default false, ADR 0105).
   * Supported only together with linearGemvReduce: parallel / linearCompute: f32.
   * It applies only to "linear activations from a plain static_quantize node whose consumers all go
   * to the parallel GEMV"; a shape with even one other consumer, a shape whose k is not a multiple
   * of 16, and scale 0 (identity) stay in f32 as before.
   * Activation loads per weight word drop from 16 to 4 for i2, 8 to 2 for i4 and 4 to 1 for i8
   * (research 2026-09-19 §14).
   * Decoding is `f32(code) * scale`, and matches the current SRQ output per element at u32 level
   * (ADR 0105).
   * The exceptions are two values with no int8 representation — -0.0 becomes +0.0 (the
   * multiply-add result does not move) and NaN saturates.
   */
  readonly packedStaticQuantize?: boolean;
  /**
   * The RMS reduction method (default workgroup). subgroup32 applies to RMS wider than 128 and to
   * the optional RMS→add fusion.
   * Requires acquireGpu({ subgroups: true }). Without it the option is rejected, with no automatic
   * fallback to the reference.
   * The summation order changes, so the generated sequence may differ from the reference. Adoption
   * on M2 and for speculative generation is accepted separately.
   */
  readonly rmsNormReduce?: RmsNormReduce;
  /**
   * The **parallelism target** of the row-block gemv (the GEMV family of linear, M ≥ 2) (thread
   * count = output columns n × y tiles). Default 16384 = the saturation point of the reference
   * device (RTX 3080 Ti).
   *
   * On GPUs with a smaller saturation point (integrated GPUs, Apple M series), lowering it raises
   * `rows` (rows per thread) and reduces weight re-reads (the mechanism is
   * `linearGemvRowsForShape` in src/kernels/linear-gemv.ts). A **static** knob — fixed at Session
   * creation, not changed at run time, and not selected automatically from the device (the ban on
   * run-time autotuning, ADR 0022).
   * A different target changes the chosen `rows`, and `rows` is part of the pipeline key, so
   * "same key → byte-identical WGSL" holds.
   * MUST: anything other than a safe integer of at least 1 fails loudly.
   */
  readonly linearGemvRowsThreadTarget?: number;
  /**
   * The **budget for slot backings held at once** (a slot backing is the set of intermediate
   * buffers used by runs that hit a prepared plan) (bytes; default
   * {@link DEFAULT_PLAN_BACKING_BUDGET_BYTES} = 256 MiB).
   *
   * There is one backing per run shape (signature), and generation switches between the prefill
   * and decode shapes every turn. Within the budget, switching keeps both instead of rebuilding
   * (removing a rebuild of ≈ 40 ms per switch — perf-ledger H-15); beyond it, the oldest are
   * retired. **A shape whose single new backing alone exceeds the budget retires all the others and
   * keeps only itself** (= the original capacity-1 behaviour), so residency never exceeds
   * `max(budget, largest single backing)`. A budget of 0 always keeps one, as before.
   * What is counted is the VRAM each backing holds = **the sum of its regions + the input buffers
   * it owns** ({@link PlanBackingStats.residentBytes} + {@link PlanBackingStats.inputBytes}).
   * MUST: anything other than a non-negative safe integer fails loudly.
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * The **limit per row-block temporary** of attention (bytes; no default — ADR 0124).
   *
   * The three attention forms that materialise the scores S (fused attention, the states form, and
   * decomposed attention's `rowBlockAttention`) split S into groups of query rows (row blocks) and
   * compute them in turn. The count is "the smallest count at which one S block and one row-stats
   * block each fit the limit", and the limit is the **smaller** of the device's
   * `maxStorageBufferBindingSize` and this value. A value below the binding limit raises the count
   * and shrinks the per-block temporaries (S and row statistics). The peak of all intermediates
   * (the VRAM peak) can drop below the no-budget case, but a smaller value does not always lower it
   * (it depends on how the intermediate layout splits — the measurement of one Wan2.2 shape is in
   * the Results section of ADR 0124, step 4b, and in
   * `docs/research/2026-10-11-row-block-budget-stage-4b.md`).
   * In exchange, dispatches grow by "added blocks × dispatches per block"
   * (3 per block for fused attention, 4 for decomposed attention, 2 to 3 for the states form — the
   * effect on speed depends on the shape and the machine).
   *
   * - **Not a total**: one S block and one row-stats block are each at most this value, and the
   *   total alive at once can exceed it. The temporaries of i8a8 quantization (outside the
   *   row-block loop) and other intermediates are not bounded by this value.
   * - **Does not change the numbers**: row blocks only change the assignment of rows (ADR 0060
   *   decision 3), so the output matches the no-budget case bit for bit (the parity of forced
   *   splits binds this across variants). It is a resource policy kept out of the numeric settings
   *   (quant presets, the manifest's `session`), handled like `submitPolicy`.
   * - **The default (omitted) stays at the binding limit**: the count, pipeline keys, WGSL,
   *   dispatch sequence and backing are byte-for-byte what they were before this field was added.
   * - **When it fails**: a shape whose temporary for a single query row exceeds this value fails
   *   loudly, naming the budget, in the run that first derives that shape's plan (at construction
   *   the shape is still symbolic and the count is unknown). The way to know in advance is the
   *   estimate (pass the same value to `EstimateOptions.attentionRowBlockMaxBytes`). Construction
   *   fails only for values outside the range (0, negative, non-integer, NaN, non-numbers).
   *
   * MUST: anything other than a positive safe integer fails loudly.
   */
  readonly attentionRowBlockMaxBytes?: number;
  /**
   * **The backing of shared initializers** (ADR 0096 decision 2 and its stage 2 addendum —
   * borrower's initializer name → the return value of the lender's `Session.exportWeight()`).
   *
   * MUST be given for **every** `shared` declaration of the graph, no more and no fewer. There are
   * six checks (same device, declared shape, semantic dtype, consuming residency class (derived
   * from the lender's codec), row axis — `resolveSharedWeights`); any violation would surface as a
   * different value rather than an exception, so each fails loudly.
   *
   * Lifetime: while a borrowing Session is alive, `dispose()` of the lending Session fails loudly
   * (borrow counting — a deliberate departure from the idempotent, non-throwing contract of
   * `dispose`).
   */
  readonly sharedWeights?: Readonly<Record<string, SharedWeight>>;
  /**
   * Test only ({@link I8A8_DOT}). The default is decided from the enumeration of
   * wgslLanguageFeatures.
   */
  readonly [I8A8_DOT]?: I8a8Dot;
  /**
   * Test only ({@link ROW_BLOCK_SPLIT}). The default is the count determined statically from the
   * device's limit.
   */
  readonly [ROW_BLOCK_SPLIT]?: number;
};

/**
 * What low-precision storage (f16 — ADR 0018 / i8 — ADR 0019) achieved. **The permanent diagnostic
 * required by ADR 0006**, and the only observation point that keeps "f16 / i8 requested but 0 MB
 * eligible" from going unnoticed.
 *
 * Only initializers with compressed storage are counted (f32 / i32 storage counts in neither). Both
 * 0 means "the model has no low-precision storage at all"; `resident` 0 with a large
 * `hostExpanded` means "low precision was declared but everything failed the eligibility check" —
 * the shape lets the two be told apart.
 */
export type StorageDiagnostics = {
  /**
   * The **bytes in GPU buffers** of weights resident on the GPU in compressed form (including the
   * alignment zero padding; for i8 / i2 / i4 **the scale buffers (per-channel for i8 / i2, per
   * group for i4) are added too** — the field represents the bytes the GPU actually holds, so
   * leaving out the scales would disagree with the actual VRAM). The point of this path is 1/2 (f16)
   * and about 1/4 (i8) of holding them in f32.
   */
  readonly residentCompressedBytes: number;
  /**
   * The bytes, **after decoding**, of weights that were not eligible and were decoded to f32 on the
   * CPU at load (= the bytes the GPU actually holds). They save no VRAM; only the download size
   * shrank.
   */
  readonly hostExpandedBytes: number;
};

/**
 * The cost breakdown of the Session construction phase (`Session.build` = the explicit async stage
 * that uploads the weights). A permanent diagnostic that breaks "construction took 2.50s" down
 * into **where the time went**; the values do not change over the Session's lifetime (they are
 * fixed when construction finishes, and no run moves any of them).
 *
 * ## Limits of attribution (the furthest this breakdown can go)
 *
 * Only the host clock (`performance.now()`) is used, and **no GPU fence is added** (adding one
 * would break the contract of one submit per part and push the instantaneous peak up by one weight
 * — ADR 0108 decision 9). **The real GPU transfer time of `queue.writeBuffer` therefore cannot be
 * separated with the host clock**, and is absorbed whole into the fence wait at the end of each
 * part ({@link SessionBuildStats.uploadFenceMs}).
 * MUST: do not read "the speed of the transfer itself" from these seven fields. The breakdown can
 * go no further than **two categories, "time the host spent"** (supplyWait / decode / bufferCreate
 * / writeBufferIssue) **and "time spent waiting for the GPU to complete"** (uploadFence); the
 * breakdown of the latter (real transfer / queue wait / driver behaviour) is not visible to the
 * runtime.
 *
 * NOTE: byte and count fields **reuse the existing diagnostics** (they are not repeated here) —
 * bytes after CPU decoding are {@link StorageDiagnostics.hostExpandedBytes}, compressed resident
 * bytes are {@link StorageDiagnostics.residentCompressedBytes}, and the number of allocated buffers
 * is `allocCount` of {@link SessionDiagnostics.weights}.
 * NOTE: the cost of graph validation and pipeline creation is **not included** — the former is
 * outside the construction phase (the `prepareContainer` phase), and the latter does not exist in
 * the construction phase at all (pipelines are created on the first run; `pipelineCount` right
 * after construction is 0). Mixing them in would blur the definition of "construction cost".
 */
export type SessionBuildStats = {
  /**
   * The number of supply units (container parts) consumed.
   */
  readonly partCount: number;
  /**
   * The total time spent **waiting for the next item** of the supply (the iteration wait of
   * `for await`). There are two levels of wait, and both are added — the next part in the part
   * sequence, and the next block within a part. Blocks are read one at a time as they are pulled
   * (lazy), so nearly all of the cost of fetching and verification (sha256 for an unverified
   * source, the sign check of ternary values) lands in the block-level wait. Network / disk cost
   * collects here, so this field is the only point that separates that attribution from the
   * construction cost (if it is large, the slow part is the supply side, not Karume).
   */
  readonly supplyWaitMs: number;
  /**
   * The total time spent decoding non-eligible weights to f32 on the CPU (decode of f16 / i8 / i2 /
   * i4). **0 if every weight passed the eligibility check** — a non-zero value goes together with
   * lost VRAM savings ({@link StorageDiagnostics.hostExpandedBytes} is positive).
   */
  readonly decodeMs: number;
  /** The total time spent in `createBuffer` (weight bodies + scales). */
  readonly bufferCreateMs: number;
  /**
   * The total time spent **issuing** `queue.writeBuffer`. The call copies into staging but does not
   * wait for the GPU transfer to complete, so this is "the host-side copy cost", not the transfer
   * time ("Limits of attribution" above).
   */
  readonly writeBufferIssueMs: number;
  /**
   * The total bytes actually passed to `queue.writeBuffer` (weight bodies + scales, including the
   * alignment zero padding). It does not match the sum of the two fields of
   * {@link StorageDiagnostics} — those are the bytes the GPU holds, and this is **the bytes streamed
   * from the host** (non-eligible weights stream their decoded f32, so this one is larger).
   */
  readonly uploadedBytes: number;
  /**
   * The total time spent waiting for the explicit submit per supply unit (part) to complete (the
   * fence of ADR 0108 decision 9). The real GPU transfer time is absorbed here ("Limits of
   * attribution" above).
   */
  readonly uploadFenceMs: number;
};

/**
 * What the content-addressed cache of params buffers achieved (for one run). params do not depend
 * on tensor values at run time, so as long as shapes do not change, `allocCount` drops to 0 from
 * the second run on.
 *
 * MUST: emit this as a permanent diagnostic. A cache miss keeps the values correct (it only falls
 * back to allocating per dispatch), with no exception and no warning — this is the only
 * observation point.
 *
 * NOTE: in a run that hit a prepared plan ({@link PreparedPlanStats}) **the derivation phase does
 * not run at all**, so `allocCount` and `reuseCount` are both 0. The meaning of the values does not
 * change; it reports the fact that "that run performed zero GPU operations on params".
 */
export type ParamsCacheStats = {
  /** The number of params newly allocated + written with writeBuffer in this run. */
  readonly allocCount: number;
  /** The number of params handed out again from the cache in this run (zero GPU operations). */
  readonly reuseCount: number;
};

/**
 * What the prepared execution plans (resident in the Session) achieved. A run repeated with the
 * same bindings skips planning, fusion decisions and recipe derivation entirely, and hands the
 * recipe sequence straight to the execution phase.
 *
 * MUST: emit this as a permanent diagnostic. A cache miss keeps the values correct (it only falls
 * back to deriving on every run), with no exception and no warning — only the speed quietly drops.
 * This is the only observation point.
 */
export type PreparedPlanStats = {
  /** Whether this run hit a prepared plan. */
  readonly hit: boolean;
  /** The number of prepared plans the Session holds when this run settles (bounded). */
  readonly cachedPlans: number;
};

/**
 * What the GPU backing of intermediate buffers (the region buffers resident in the Session — ADR
 * 0093) achieved. A run that hits a prepared plan hands out its intermediate buffers from here, so
 * per-run region allocation and createBuffer / destroy drop to zero.
 *
 * MUST: emit this as a permanent diagnostic. Shapes that fit the budget
 * ({@link SessionOptions.planBackingBudgetBytes}) keep and reuse both backings across signature
 * round trips (ADR 0095 decision 1), but **for shapes that do not fit and cause retirement, every
 * switch rebuilds** (with budget 0, every change of signature rebuilds), and createBuffer /
 * destroy of hundreds of MiB per run returns while the values stay correct (no exception, no
 * warning). `buildCount` not growing in proportion to the run count is the only observation point
 * for that silent degradation.
 */
export type PlanBackingStats = {
  /**
   * **The sum of the regions** kept resident by all retained backings (0 if not built or
   * destroyed).
   * MUST: the definition is "the sum of the plan's regions" — the input buffers a backing also keeps
   * resident are not included (the reason and the check are in {@link ActiveBacking.bytes}).
   * What {@link SessionOptions.planBackingBudgetBytes} counts is this plus
   * {@link PlanBackingStats.inputBytes}, `residentBytes + inputBytes`.
   */
  readonly residentBytes: number;
  /**
   * The total of the input buffers owned by the retained backings (resident inputs are not owned,
   * so they are not included). What the budget ({@link SessionOptions.planBackingBudgetBytes})
   * counts is `residentBytes + inputBytes`.
   */
  readonly inputBytes: number;
  /**
   * The number of retained backings (several are retained within the budget — {@link
   * SessionOptions.planBackingBudgetBytes}).
   */
  readonly retainedCount: number;
  /**
   * The cumulative number of backings built over the Session's lifetime (cumulative, not per run).
   */
  readonly buildCount: number;
};

/**
 * The default of {@link SessionOptions.planBackingBudgetBytes} (256 MiB). For gemma4 E2B the decode
 * shape (3 MiB) and the prefill bucket 32 / 64 / 128 shapes (23 / 45 / 89 MiB at capacity 16K) fit,
 * and the chunk 768 shape (≈ 500 MiB at capacity 16K) falls to the keep-only-one side.
 */
export const DEFAULT_PLAN_BACKING_BUDGET_BYTES = 256 * 1024 * 1024;

/**
 * What the state backing (the buffers owned by a {@link GenerationContext}) achieved. The
 * diagnostic field that ADR 0066 decision 5 places on a par with {@link PlanBackingStats}, and the
 * only observation point for KV residency and the silent degradation of re-baking.
 *
 * MUST: emit this as a permanent diagnostic. **The context identity is not part of the plan key**
 * (a MUST of decision 5), so switching contexts does not re-derive recipes — instead only "re-baking
 * the bind groups that contain state" happens, and if that runs away there is no exception and no
 * warning.
 */
export type StateBackingStats = {
  /**
   * The total bytes kept resident by the live contexts (0 if none were created or all are
   * disposed).
   * MUST: the definition is "the total bytes of the state slots + the logical-length uniform" = the
   * bytes a context actually holds on the GPU (the same discipline as
   * {@link StorageDiagnostics.residentCompressedBytes} counting the scales). It is derived from the
   * live set each time; no independently updated counter is kept.
   */
  readonly residentBytes: number;
  /**
   * The cumulative number of contexts created over the Session's lifetime (cumulative, not
   * current).
   */
  readonly contextCount: number;
  /**
   * The cumulative number of times the bind groups containing state were re-baked (the separate
   * bake units of ADR 0066 decision 5).
   *
   * Re-baking happens only **when the (context, backing instance) pair changes** — the first time,
   * a context switch, and a rebuild of the backing (a switch to another signature / LRU eviction /
   * recovery from a failed build).
   * MUST: **this not growing in proportion to the run count** is the only observation point that
   * decision 5's separation is working. Proportional growth is exactly the state of "rebuilding
   * everything on every switch", in which createBindGroup returns to the decode hot path while the
   * values stay correct (no exception, no warning).
   */
  readonly rebindCount: number;
};

export type SessionDiagnostics = {
  /**
   * The number of pipeline keys **this Session used** (how many kernels were set up for the
   * combination of graph and opt-ins).
   *
   * MUST: the definition is "the number of keys used", not "the number this Session caused to be
   * created". The pipeline cache lives as long as the device (owned by the GpuContext), so if an
   * earlier Session on the same device already created them, this Session creates none — with a
   * creation-side definition, Sessions of the same graph would report "3 for the first, 0 for the
   * second", useless as an observation point for the kernel count.
   * NOTE: pipelines are created on the first run, so this is 0 right after construction
   * ({@link SessionBuildStats}).
   */
  readonly pipelineCount: number;
  /**
   * The total number of pipeline keys held by the cache on this device (the union over all
   * Sessions).
   *
   * Keys used by disposed Sessions stay here too (the cache lives as long as the GpuContext and is
   * released only by `GpuContext.destroy`). The difference from
   * {@link SessionDiagnostics.pipelineCount} is "the share of other Sessions on the same device",
   * and steady growth across runs is the only observation point for a break of codegen determinism
   * (a value that should not be in the key is in it).
   */
  readonly devicePipelineCount: number;
  readonly submit: SubmitStats;
  /**
   * What the weight (initializer) arena achieved. **It also owns the backing of the params cache
   * (resident in the Session)**, so `allocCount` is the initializer count + the number of params
   * created.
   */
  readonly weights: ArenaStats;
  /**
   * The eligible / non-eligible breakdown of low-precision storage (the permanent diagnostic of ADR
   * 0006).
   */
  readonly storage: StorageDiagnostics;
  /**
   * The cost breakdown of the construction phase (unchanged over the Session's lifetime — fixed
   * when construction finishes).
   * The limits of attribution are in the docstring of {@link SessionBuildStats}.
   */
  readonly buildStats: SessionBuildStats;
  /**
   * The name of this Session's GEMM geometry profile (a static table selected once at construction
   * from the adapter's vendor / architecture — `"default"` on a machine that matches none; on a
   * device into which a table was injected with `acquireGpu({ geometryProfile })`, that table's
   * `id`). Unchanged over the Session's lifetime.
   *
   * Geometry does not move the numeric contract (only the work assignment), so it does not show in
   * the values, but it does show in speed and in the geometry discriminator of the pipeline keys
   * (`reg64x32r4x4w8` and the like). The only observation point that names which table ran.
   * NOTE: the name is a diagnostic label outside semver (it can change without notice as tables are
   * added or renamed).
   */
  readonly geometryProfile: string;
  /**
   * The intermediate-buffer figures of the latest run. undefined if nothing has run.
   *
   * NOTE: in a run on a slot backing ({@link PlanBackingStats}), neither intermediate buffers nor
   * input buffers go through the arena, so only the readback staging remains here, and the three
   * fields copied from the plan (`reuseCount` / `transientBytes` / `peakTransientBytes`) are 0 (the
   * meaning of the values is unchanged — "what that run allocated from the arena" and "that run's
   * plan").
   */
  readonly lastRun: ArenaStats | undefined;
  /**
   * The **per-op GPU time breakdown** of the latest run (per pipeline key — ADR 0021).
   * undefined on a device where measurement is disabled (`gpuTiming` of `acquireGpu` / the feature
   * is absent). Same lifetime as `lastRun`; reset at the start of a run.
   */
  readonly lastRunTiming: GpuTimingStats | undefined;
  /**
   * The number of fusions / aliasings decided **at planning time** for the latest run (per rule —
   * src/runtime/fusion.ts). undefined if nothing has run, and replaced wholesale by every run.
   *
   * MUST: emit this as a permanent diagnostic. A fusion silently stops applying when a single node
   * of the exporter's emission order changes, and only the speed drops while the values stay
   * correct (no exception, no warning). This is the only observation point.
   */
  readonly lastRunFusions: FusionCounts | undefined;
  /**
   * The params cache figures of the latest run. undefined if nothing has run; replaced by every run.
   * The backing (the buffers) is resident in the Session, so this is "how many that run created and
   * how many it reused".
   */
  readonly lastRunParams: ParamsCacheStats | undefined;
  /**
   * The prepared-plan cache figures of the latest run. Reset at the start of a run and filled when
   * the derivation phase settles (undefined if nothing has run, and for a run that failed during
   * the derivation phase).
   */
  readonly lastRunPrepared: PreparedPlanStats | undefined;
  /**
   * The **planned** dispatch counts of the latest run (per pipeline key, in lexicographic key
   * order). Replaced wholesale by every run; the content is decided at planning time (like
   * {@link SessionDiagnostics.lastRunFusions}). It is filled at the same point as
   * {@link SessionDiagnostics.lastRunPrepared}, and is undefined for a Session that has not run and
   * for a run that failed during the derivation phase.
   *
   * It is filled regardless of measurement (`gpuTiming` of `acquireGpu`), and also on the resident
   * path (`enqueue` / `enqueueRead`). Fallbacks show in the pipeline keys (ADR 0058 decision 3), but
   * if the list of keys were only in {@link SessionDiagnostics.lastRunTiming}, it could be read only
   * on devices that requested measurement, and fallbacks could not be observed on the resident path,
   * which cannot use measurement (ADR 0054) — this field closes that gap.
   * NOTE: it counts planned dispatches. A states-form dispatch whose workgroup count is computed
   * from the logical length is not actually recorded when that step has zero work, so this can be
   * larger than `dispatchCount` of `lastRunTiming` (every other dispatch corresponds 1:1 to an
   * actual issue).
   * NOTE: the spelling of the keys is a diagnostic label outside semver (it can change without notice
   * as variants are added or renamed).
   */
  readonly lastRunPipelines:
    | readonly { readonly key: string; readonly dispatchCount: number }[]
    | undefined;
  /**
   * What the GPU backing of transient slots achieved (the Session's current state + cumulative
   * counts, not per run).
   * For a Session that has not built one:
   * `{ residentBytes: 0, inputBytes: 0, retainedCount: 0, buildCount: 0 }`.
   */
  readonly planBacking: PlanBackingStats;
  /**
   * What the state backing (owned by a {@link GenerationContext}) achieved (the Session's current
   * state + cumulative counts — ADR 0066 decision 5). For a Session that has created no context:
   * `{ residentBytes: 0, contextCount: 0, rebindCount: 0 }`.
   */
  readonly stateBacking: StateBackingStats;
};
