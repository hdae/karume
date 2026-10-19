/**
 * Splitting submits by a time budget.
 *
 * MUST: do not put the whole dispatch sequence into one command buffer. A long submit hits Windows
 * TDR (2 seconds by default) and the Chromium watchdog and shows up as device lost. Submit per
 * chunk, and decide where chunks end from **the estimated time per workgroup derived from
 * measurements**.
 *
 * ## Invariants (the execution model of ADR 0004)
 *
 * 1. **Growth rests only on measured evidence**. A measurement of 0 (below timer resolution) means
 *    "no information", not "fast" — the estimate is not updated. Without evidence the upper limit
 *    stays at {@link SubmitPolicy.initialChunkSize}.
 * 2. **One chunk's estimated time does not exceed the time budget**. Before adding, decide "adding
 *    this dispatch would exceed it" and close the chunk first (not shrinking after an overrun). A
 *    dispatch that exceeds the budget on its own cannot be split and is the exception
 *    ({@link SubmitPolicy.minChunkSize} also acts as a lower bound).
 * 3. **Feed-forward**. The unit of the estimate is the **workgroup count**, not the dispatch count;
 *    when the input length T changes, the workgroups per dispatch change too. Estimating per
 *    dispatch would queue a long input with the value measured on a short one and overrun the
 *    budget badly once (with graphs that have symbolic dimensions this always happens within the
 *    same Session).
 *
 * ## Attribution of measurements (why the estimate comes from the one flush window)
 *
 * MUST: do **not call** `onSubmittedWorkDone` **per submit**. On some implementations the
 * synchronous part of this call blocks the host until "every piece of work submitted so far has
 * completed", fully serializing CPU encoding and GPU execution (wall time becomes **the sum of GPU
 * + host** instead of `max(GPU, host)`; measurements and the isolation experiment on plain WebGPU
 * are in `docs/research/2026-08-04-host-overhead-recon.md` §4.1). It is called only once, in
 * {@link SubmitScheduler.flush} — that wait is needed anyway for flush-before-destroy, so the extra
 * cost of measuring is zero.
 * The only exception is the completion marker of a run's in-flight submit bound
 * ({@link SubmitScheduler.backpressure} — ADR 0123): when a run asks between steps, one marker is
 * placed if 64 submits have been reached since the previous marker (enqueue does not place one).
 *
 * NOTE (following H-1): ordinary runs no longer set `onSubmittedWorkDone` either (the readback copy
 * is queued in the same command stream as the dispatches and **the readback `mapAsync` is the only
 * fence** — `singleFence` in src/runtime/executor.ts). On this path the window is also closed by
 * {@link SubmitScheduler.closeMeasurementWindowAfterFence}, not through `flush`. The character of
 * the window (1 run = 1 window, inflated by host time) does not change.
 *
 * The estimate comes from the **window** closed by that one call. Window = "the time of the first
 * submit in the window → the time the flush's `onSubmittedWorkDone` resolved", work = the workgroup
 * total of every chunk in the window, and the estimate is **measured time ÷ total workgroups**.
 * Per-chunk attribution was never trustworthy (completion notifications of overlapping submits
 * arrive almost together, the first one takes all the time and the rest get 0), so coarsening to
 * the window loses no information.
 *
 * NOTE: the window also contains host-side encoding time, so the estimate comes out **larger** than
 * the real GPU time. Larger = fewer workgroups fit in the same budget = **smaller chunks**, which
 * is the safe side for TDR. Two stops keep chunks from getting too thin: {@link
 * SubmitPolicy.minChunkSize} (the budget cuts only after at least this many are queued) and {@link
 * SubmitPolicy.maxChunkSize} (a hard upper limit).
 *
 * NOTE (following H-5): in an enqueue section without fences (`GpuContext.beginBatch`) the window
 * is **one section rather than one run** (closed once by `BatchScope.finish` —
 * {@link SubmitScheduler.closeMeasurementWindowAfterFence}). With N executions in one window the
 * estimate gets coarser and larger still, but in the same safe direction as above, so the TDR stop
 * still holds. The adaptive control uses only the latest window, so after leaving the section the
 * next window returns to the normal granularity.
 *
 * ## Breakdown of real GPU time (timestamp-query — ADR 0021)
 *
 * The wall-clock measurement above is per chunk and its attribution is not trustworthy (it is only
 * material for the adaptive control). A per-op breakdown can only be taken from **timestamps at
 * pass boundaries**, so only on a device with measurement enabled ({@link
 * GpuContext.gpuTimingEnabled}) the chunk is opened as **1 dispatch = 1 pass** and
 * `timestampWrites` are written. Consecutive dispatches in one pass already have an implicit
 * dependency order through the visibility guarantee of storage, so splitting passes does not change
 * the execution semantics.
 *
 * MUST: when measurement is disabled, not a single line of this path's code runs (1 chunk = 1 pass
 * as before).
 */

import { type GpuContext, RUNTIME_INTERNAL } from "./device.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** The SubmitPolicy values form a configuration that cannot run (including a hang). */
export class SubmitPolicyError extends Error {
  override readonly name = "SubmitPolicyError";
}

export type SubmitPolicy = {
  /**
   * Chunk boundaries are set from the estimate so that one chunk's GPU execution stays under this.
   */
  readonly timeBudgetMs: number;
  /**
   * The dispatch-count limit of one chunk **while there is no measured evidence** (bootstrap).
   * MUST: no growth from here even if measurements of 0 continue (invariant 1).
   */
  readonly initialChunkSize: number;
  /**
   * A lower bound so that cutting by the time budget never goes below this dispatch count (a stop
   * on the number of submits). Only when a dispatch exceeds the budget on its own does a chunk
   * whose estimated time exceeds the budget come out.
   */
  readonly minChunkSize: number;
  /** A hard limit on the dispatch count that is not exceeded even with measured evidence. */
  readonly maxChunkSize: number;
};

/**
 * The default policy. Frozen because rewriting the public value would spread to every Session that
 * uses the default (the scheduler copies it at construction, but the default itself would change
 * before the copy).
 */
export const DEFAULT_SUBMIT_POLICY: SubmitPolicy = Object.freeze({
  timeBudgetMs: 100,
  initialChunkSize: 16,
  minChunkSize: 1,
  maxChunkSize: 1024,
});

/**
 * A copy of the policy (held by the entry points that construct a {@link SubmitScheduler} and a
 * Session).
 *
 * MUST: copy the fields one by one with ordinary property reads (no spread). Spread copies only own
 * enumerable fields, so inherited or non-enumerable fields are dropped, and a policy that passed
 * before the copy turns into a rejection for 0.
 */
export const copySubmitPolicy = (policy: SubmitPolicy): SubmitPolicy => ({
  timeBudgetMs: policy.timeBudgetMs,
  initialChunkSize: policy.initialChunkSize,
  minChunkSize: policy.minChunkSize,
  maxChunkSize: policy.maxChunkSize,
});

/**
 * Safety factor applied to the estimated time. A chunk is estimated with the **average cost**
 * measured in workgroups of grid-stride kernels, so this keeps headroom for heavy dispatches
 * clustering in one chunk (a skew of up to 2× stays within the budget).
 */
const CHUNK_TIME_SAFETY = 0.5;

/**
 * The in-flight submit bound (DECIDED: ADR 0123): when a run asks
 * {@link SubmitScheduler.backpressure} between steps, one completion marker (`onSubmittedWorkDone`)
 * is placed if the submits since the previous marker within the measurement window have reached
 * this count.
 *
 * NOTE: the count is per measurement window because the window closes right after a fence ("the
 * number of submits since a fence last confirmed GPU completion" is exactly the window's chunk
 * count). A run whose submits between fences never reach this count (repeated short runs such as
 * decode) places no marker at all.
 */
export const IN_FLIGHT_MARKER_INTERVAL = 64;

/**
 * The number of completion markers that may be pending (ADR 0123). Beyond it, wait for the oldest
 * marker to complete (pending returns to this count) — the host runs ahead by at most about
 * (this value + 1) × {@link IN_FLIGHT_MARKER_INTERVAL} submits.
 */
export const IN_FLIGHT_MAX_MARKERS = 2;

/** One completion marker ({@link IN_FLIGHT_MARKER_INTERVAL}). */
type InFlightMarker = {
  readonly work: Promise<void>;
  /**
   * Whether it has settled (including rejection). Completion is FIFO, so it is set front to back.
   */
  settled: boolean;
};

/**
 * Policy check at construction.
 *
 * MUST: chunk sizes are at least 1. With 0, cutting chunks keeps returning an empty array and the
 * loop in `flush()` never exits (a hang rather than a wrong value, so it fails at construction
 * rather than at run time).
 */
const assertPolicy = (policy: SubmitPolicy): void => {
  for (const key of ["initialChunkSize", "minChunkSize", "maxChunkSize"] as const) {
    const value = policy[key];
    if (!Number.isInteger(value) || value < 1) {
      throw new SubmitPolicyError(`SubmitPolicy.${key} must be an integer >= 1: ${value}`);
    }
  }
  if (policy.minChunkSize > policy.maxChunkSize) {
    throw new SubmitPolicyError(
      `SubmitPolicy.minChunkSize (${policy.minChunkSize}) exceeds maxChunkSize (${policy.maxChunkSize})`,
    );
  }
  if (
    policy.initialChunkSize < policy.minChunkSize ||
    policy.initialChunkSize > policy.maxChunkSize
  ) {
    throw new SubmitPolicyError(
      `SubmitPolicy.initialChunkSize (${policy.initialChunkSize}) is outside [${policy.minChunkSize}, ${policy.maxChunkSize}]`,
    );
  }
  if (!(policy.timeBudgetMs > 0) || !Number.isFinite(policy.timeBudgetMs)) {
    throw new SubmitPolicyError(
      `SubmitPolicy.timeBudgetMs must be a positive finite number: ${policy.timeBudgetMs}`,
    );
  }
};

/**
 * The number of recent measurements kept in `measuredMs`.
 *
 * MUST: it is bounded. The adaptive control uses only the most recently closed window; the history
 * is for diagnostics only. Unbounded, it would eat memory monotonically during long inference, and
 * the copy cost of `stats` would grow with the number of flushes (the more you read diagnostics,
 * the slower it gets). 32 entries are enough to see the trend of the adaptation.
 */
export const MEASURED_HISTORY = 32;

export type SubmitStats = {
  readonly submitCount: number;
  readonly dispatchCount: number;
  /**
   * The measured times (ms) of closed **measurement windows**, **up to the latest
   * {@link MEASURED_HISTORY}**. For the count over the whole lifetime see
   * {@link SubmitStats.measuredCount}.
   *
   * NOTE: 1 entry = 1 window = one {@link SubmitScheduler.flush} (in most cases one run), not a
   * per-chunk value. The window also contains host-side encoding time (module doc).
   */
  readonly measuredMs: readonly number[];
  /**
   * Number of measurement windows closed over the whole lifetime (including those that overflowed
   * the `measuredMs` history).
   */
  readonly measuredCount: number;
  /**
   * Cumulative number of dispatches that {@link SubmitScheduler.discard} threw away without
   * submitting. They are the remains of a failed run, so normally 0. A non-zero value records "a
   * run at that time failed midway".
   */
  readonly discardedDispatches: number;
  /**
   * The current **dispatch-count limit** of one chunk. {@link SubmitPolicy.initialChunkSize} while
   * there is no measured evidence, {@link SubmitPolicy.maxChunkSize} once there is.
   *
   * NOTE: no longer the main actor of the adaptation (meaning changed at M1-P2). The real
   * boundaries are usually set first by the time budget derived from {@link
   * SubmitStats.msPerWorkgroup}; this value is the hard limit "never queue more than this".
   */
  readonly currentChunkSize: number;
  /**
   * The estimated time per workgroup (ms) derived from the most recently closed window. **undefined
   * while there is no evidence yet** (it stays empty while measurements are 0 = no growth; this is
   * the observation point of invariant 1).
   * The workgroup budget of one chunk is `timeBudgetMs * CHUNK_TIME_SAFETY / msPerWorkgroup`.
   */
  readonly msPerWorkgroup?: number;
  /**
   * Observation field for how much of the time budget the chunks use ({@link ChunkBudgetStats}).
   */
  readonly chunkBudget: ChunkBudgetStats;
};

/**
 * How much of the chunk time budget is consumed (**diagnostics only** — not one byte of it is used
 * for control).
 *
 * WHY: the proportional allocation of the time budget rests on the homogeneity assumption
 * "workgroup count = amount of work", and that assumption breaks by up to 3 orders of magnitude or
 * more between kernels (the real cost per workgroup differs by orders of magnitude). Dropping the
 * assumption and feeding a per-op cost proxy into the estimator is **a separate perf-ledger
 * entry**, and the ruling is that it starts "**once a chunk over the budget is actually observed**"
 * (in measurements of the 3 shipped pipelines there was no overrun; it stayed within the safety
 * factor {@link CHUNK_TIME_SAFETY}). This is **the observation point for deciding that start
 * condition**, not the budget rule itself.
 *
 * MUST: do not bring per-chunk **wall-clock** attribution in here (ADR 0004 adaptive-control
 * invariant ④ — completion notifications of overlapping submits concentrate on the first one, and a
 * per-chunk wall clock has no meaning). What is carried here: ① the distribution on the estimate
 * side (how the cutting side estimated the budget) and ② the per-window average — two views that
 * **do not depend on attribution** — and ③ per-chunk values of the **GPU-side clock**
 * (timestamp-query), filled only on a device with measurement enabled
 * ({@link ChunkBudgetStats.submitGpuTime}). ③ comes from timestamps written by the chunk's own
 * passes, so its attribution holds (no completion notifications are used).
 */
export type ChunkBudgetStats = {
  /**
   * The reference time (ms) for cutting one chunk = `timeBudgetMs * CHUNK_TIME_SAFETY`. The point
   * of reference when reading the values below (repeated here only so they can be compared without
   * re-deriving the policy and the safety factor).
   */
  readonly budgetMs: number;
  /**
   * The maximum **estimated time** (`workgroup total × msPerWorkgroup`) of any chunk submitted over
   * the whole lifetime. Chunks issued while there was no measured evidence
   * (`msPerWorkgroup === undefined`) cannot be estimated and are not counted — undefined if the
   * evidence never arrived.
   */
  readonly maxEstimatedMs?: number;
  /**
   * Cumulative number of chunks submitted while their estimated time exceeded
   * {@link ChunkBudgetStats.budgetMs}.
   *
   * NOTE: as long as invariant 2 (cut before adding) holds, this grows only when there was no way
   * to cut — a dispatch that exceeds the budget on its own, or hitting the lower bound
   * {@link SubmitPolicy.minChunkSize}. Once it is non-zero, the start condition of the cost-proxy
   * entry is met.
   */
  readonly overBudgetChunks: number;
  /**
   * The maximum over the whole lifetime of "measured time ÷ chunk count of that window" for closed
   * windows (ms).
   *
   * WHY: looking only at the estimate side, a failure where **the work proxy itself is off** is
   * invisible in principle (it only compares a value estimated with the same proxy against the same
   * budget). The window mean is obtained independently of attribution, so it is kept as a
   * **conservative signal** that catches that drift.
   *
   * NOTE: the window is the wall clock "first submit → after the flush" and contains host-side
   * encoding, stalls and waits (module doc). It rises when the host stalls even if the GPU is fast,
   * so it is not a lower bound of the GPU chunk time, and exceeding {@link
   * ChunkBudgetStats.budgetMs} does not prove a GPU-side budget overrun. A GPU-side overrun is
   * judged by the GPU time per submit ({@link ChunkBudgetStats.submitGpuTime}).
   */
  readonly maxWindowMeanMs?: number;
  /**
   * Observation over the whole lifetime of the GPU execution time per submit (chunk). Filled only
   * on a device with measurement enabled ({@link GpuContext.gpuTimingEnabled}); undefined on a
   * disabled device and until the first one has been collected.
   *
   * WHY: {@link ChunkBudgetStats.maxWindowMeanMs} is a window average that includes host time; when
   * one heavy chunk and a few light ones share a window it is buried in the average, and conversely
   * a host stall alone raises it. Against an OS timeout per submit (`job_timeout_ms` of Linux xe
   * and the like — ADR 0118 decision 6), only the maximum time of individual chunks says anything.
   *
   * How it is measured: measurement mode writes the begin / end of each pass with 1 dispatch = 1
   * pass, so the span of one chunk's GPU execution = minimum → maximum of that chunk's timestamps
   * (including the gaps between passes). Collection piggybacks after the existing wait of
   * {@link SubmitScheduler.flush} and **adds no new wait** (waiting for completion per submit
   * serializes CPU encoding and GPU execution — the MUST in the module doc).
   *
   * NOTE: the unit is the timestamp value itself (the same as {@link GpuTimingEntry.ns} — ns per
   * the specification. Deno returns wgpu's raw ticks without conversion; on the B570 1 tick = 52.08
   * ns — the "Intel Arc B570" section of docs/known-issues.md). **Not used for control** (chunk
   * boundaries do not change with this value).
   */
  readonly submitGpuTime?: {
    /**
     * Cumulative number of chunks collected (chunks with only copies carry no timestamps and are
     * not counted).
     */
    readonly submits: number;
    /** The maximum span of one chunk's GPU execution. */
    readonly maxNs: number;
    /**
     * Of those, the number of chunks issued before measured evidence existed (while
     * `msPerWorkgroup` is undefined — limited to {@link SubmitPolicy.initialChunkSize} and not cut
     * by the time budget). Every chunk of the first run lands here.
     */
    readonly unbackedSubmits: number;
    /** The maximum span of chunks issued before evidence (undefined if there were none). */
    readonly maxUnbackedNs?: number;
    /**
     * The maximum GPU time of a single dispatch (1 pass). A gauge of the weight of the unit that
     * cannot be split.
     */
    readonly maxDispatchNs: number;
    /**
     * The pipeline key of that dispatch (the same vocabulary as {@link GpuTimingEntry.key}).
     * undefined if no positive difference has ever been observed (no dispatch can claim "maximum"
     * — filling it with the key of the first dispatch would show a non-maximum as the maximum).
     */
    readonly maxDispatchKey?: string;
  };
};

/** The breakdown of real GPU time for one pipeline key (ADR 0021). */
export type GpuTimingEntry = {
  /**
   * The unit of aggregation is the **pipeline key** (a vocabulary that already uniquely identifies
   * op kind + variant).
   */
  readonly key: string;
  /** Total real GPU time (nanoseconds). The sum of the pass begin/end differences. */
  readonly ns: number;
  readonly dispatchCount: number;
  readonly workgroupCount: number;
};

/**
 * The breakdown of real GPU time of the latest run. On a device with measurement **disabled** the
 * whole value is undefined (returning an empty table could not be told apart from "measured, but
 * all 0").
 * NOTE: when enabled, the table is empty before a run (that nothing has run yet is expressed by
 * `lastRun === undefined`).
 */
export type GpuTimingStats = {
  /** Descending by `ns` (ties in lexicographic key order). */
  readonly entries: readonly GpuTimingEntry[];
  readonly totalNs: number;
  /**
   * Total number of dispatches in the breakdown. Used for cross-checking against {@link
   * SubmitStats.dispatchCount}.
   */
  readonly dispatchCount: number;
  /**
   * Number of samples **clamped to 0** because begin > end (the driver's timestamps are not
   * monotonic).
   * MUST: do not drop them silently — a non-zero value tells the reader that the breakdown itself
   * is questionable.
   */
  readonly clampedNegativeSamples: number;
};

type PendingDispatch = {
  readonly kind: "dispatch";
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroups: readonly [number, number, number];
  /**
   * The amount of work of this dispatch (workgroup count). The time budget is allocated
   * proportionally in this unit.
   */
  readonly work: number;
  /** Where the breakdown is attributed (the {@link PipelineCache} key itself). */
  readonly key: string;
};

/**
 * A buffer-to-buffer copy ({@link SubmitScheduler.copyBuffer}). It can only be encoded **outside a
 * compute pass**, so inside a chunk the pass is closed first and the copy is queued after it.
 */
type PendingCopy = {
  readonly kind: "copy";
  readonly source: GPUBuffer;
  readonly sourceOffset: number;
  readonly destination: GPUBuffer;
  readonly destinationOffset: number;
  readonly size: number;
};

/**
 * One unsubmitted command.
 * MUST: commands run in the order they were queued (FIFO). This order is the only reason a copy can
 * read the results of the dispatches queued before it; an optimization that reorders them is not
 * allowed.
 */
type PendingCommand = PendingDispatch | PendingCopy;

/** Bytes of one timestamp (u64 ns). */
const TIMESTAMP_BYTES = 8;

/**
 * The capacity limit of the querySet created for one chunk.
 *
 * The need is "the chunk's dispatch count × 2". 2,048 is the design allowance of ADR 0021 (twice
 * the default policy's `maxChunkSize` of 1,024), set more conservatively than WebGPU's querySet
 * limit `maxQueryCount` of 4,096. A `maxChunkSize` of 1,025 to 2,048, valid as far as WebGPU is
 * concerned, is also rejected by this allowance. A policy beyond it fails explicitly before
 * encoding (at construction and at querySet creation).
 */
const MAX_TIMESTAMP_QUERIES = 2048;

/** Submitted, not yet collected timestamp resources (one per chunk). */
type PendingTiming = {
  readonly querySet: GPUQuerySet;
  readonly resolveBuffer: GPUBuffer;
  readonly readBuffer: GPUBuffer;
  /** The attribution targets, 1:1 with the sequence of query pairs (begin, end). */
  readonly entries: readonly { readonly key: string; readonly work: number }[];
  /**
   * Whether this chunk was issued after measured evidence existed ({@link
   * ChunkBudgetStats.submitGpuTime}).
   */
  readonly backed: boolean;
};

/** Per-key totals (the mutable version of {@link GpuTimingEntry}). */
type TimingTotal = { ns: number; dispatchCount: number; workgroupCount: number };

/**
 * The mutable version of {@link ChunkBudgetStats.submitGpuTime} (lifetime totals — not reset per
 * run).
 */
type SubmitGpuTime = {
  submits: number;
  maxNs: number;
  unbackedSubmits: number;
  maxUnbackedNs: number | undefined;
  maxDispatchNs: number;
  maxDispatchKey: string | undefined;
};

/**
 * Releases timestamp resources. Runs **always** exactly once regardless of whether collection
 * succeeded (no VRAM left behind for diagnostics). The targets are referenced only by already
 * submitted commands, so destroying them during execution is also safe in WebGPU terms (the
 * implementation defers the real release until completion).
 */
const destroyTiming = (timing: PendingTiming): void => {
  timing.readBuffer.destroy();
  timing.resolveBuffer.destroy();
  timing.querySet.destroy();
};

export class SubmitScheduler {
  readonly #gpu: GpuContext;
  readonly #device: GPUDevice;
  readonly #policy: SubmitPolicy;
  /** Time source for measurement (ms, monotonic). Defaults to `performance.now()`. */
  readonly #now: () => number;
  #pending: PendingCommand[] = [];
  /** Total work of unsubmitted encodes (the sum of `work` over `#pending`). */
  #pendingWork = 0;
  #submitCount = 0;
  #dispatchCount = 0;
  /** A ring of the latest {@link MEASURED_HISTORY} entries (the oldest are dropped first). */
  readonly #measuredMs: number[] = [];
  #measuredCount = 0;
  #discardedDispatches = 0;
  /**
   * The start of the open measurement window (the time of the window's first submit). undefined =
   * no window is open (nothing has been submitted since the last flush).
   */
  #windowStartedAt: number | undefined;
  /** Total workgroups queued in the open window. */
  #windowWork = 0;
  /**
   * Number of chunks submitted in the open window (the denominator of the window mean — {@link
   * ChunkBudgetStats}).
   */
  #windowChunks = 0;
  /**
   * `#windowChunks` at the last completion marker in the open window (reset to 0 when the window is
   * closed or discarded).
   */
  #markedChunks = 0;
  /**
   * Completion markers placed and not yet dropped from the front ({@link
   * SubmitScheduler.backpressure}).
   */
  readonly #markers: InFlightMarker[] = [];
  /**
   * The chunk with the largest estimated time (diagnostics only). undefined = no chunk with
   * evidence yet.
   */
  #maxEstimatedChunkMs: number | undefined;
  /**
   * Cumulative number of chunks issued while their estimated time exceeded the budget (diagnostics
   * only).
   */
  #overBudgetChunks = 0;
  /**
   * The lifetime maximum of the window-mean chunk time (diagnostics only). undefined = no window
   * closed yet.
   */
  #maxWindowMeanMs: number | undefined;
  /** The estimated time per workgroup backed by measurement. undefined = no evidence yet. */
  #msPerWorkgroup: number | undefined;
  /** Whether GPU-side time measurement is enabled (decided by the device's features — ADR 0021). */
  readonly #timingEnabled: boolean;
  /** Submitted, not yet collected timestamp resources. */
  #pendingTimings: PendingTiming[] = [];
  /** Per-pipeline-key totals (lifetime = the latest run — {@link SubmitScheduler.resetTiming}). */
  readonly #timingTotals = new Map<string, TimingTotal>();
  #clampedNegativeSamples = 0;
  /**
   * Observation of the GPU time per submit (diagnostics only, lifetime — not cleared by
   * {@link SubmitScheduler.resetTiming}). undefined = nothing collected yet (always undefined on a
   * device with measurement disabled).
   */
  #submitGpuTime: SubmitGpuTime | undefined;

  /**
   * @param now Time source (ms). Injectable so that tests can pin the rules of differential timing.
   */
  constructor(
    gpu: GpuContext,
    policy: SubmitPolicy = DEFAULT_SUBMIT_POLICY,
    now: () => number = () => performance.now(),
  ) {
    // MUST: copy the policy, then check it, and keep only the copy. Holding the caller's object by
    // reference would let a rewrite after the check (e.g. initialChunkSize to Infinity) remove both
    // the chunk limit (the TDR guard) and the rejection of 0 (the hang guard).
    const owned = copySubmitPolicy(policy);
    assertPolicy(owned);
    this.#gpu = gpu;
    this.#device = gpu.device;
    this.#policy = owned;
    this.#now = now;
    this.#timingEnabled = gpu.gpuTimingEnabled;
    // MUST: check the querySet capacity with measurement enabled (the chunk's dispatch count × 2)
    // at construction. With only the encode-time guard ({@link SubmitScheduler.#encodeTimedChunk}),
    // #chunkSizeLimit returns initialChunkSize until measured evidence exists, so a policy with
    // maxChunkSize > 1024 does not fail on the first run and fails only on a **later run** (a
    // misconfiguration of the public knob `SessionOptions.submitPolicy` turns up mid-operation).
    // The default policy (1024 → 2048 queries = exactly the limit) passes.
    if (this.#timingEnabled && owned.maxChunkSize * 2 > MAX_TIMESTAMP_QUERIES) {
      throw new SubmitPolicyError(
        `On a device with GPU timing enabled, SubmitPolicy.maxChunkSize (${owned.maxChunkSize}) ` +
          `exceeds the querySet capacity (${
            owned.maxChunkSize * 2
          } queries > ${MAX_TIMESTAMP_QUERIES}). ` +
          `Set maxChunkSize to ${MAX_TIMESTAMP_QUERIES / 2} or less, or set gpuTiming: false` +
          " to turn timing off",
      );
    }
  }

  /**
   * Queues one dispatch. When the time budget or the dispatch-count limit is reached, it submits on
   * the spot.
   *
   * MUST: decide the budget **before adding** (invariant 2). With "added it, it exceeded, shrink
   * from next time", the overrunning chunk is already submitted, and overrunning the budget once
   * every time becomes the steady state.
   *
   * @param key The pipeline key of this dispatch (where the GPU time breakdown is attributed — ADR
   *   0021). MUST: pass exactly the key used to fetch `pipeline`. Building it separately would
   *   silently shift only the breakdown to another op.
   */
  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: readonly [number, number, number],
    key: string,
  ): void {
    const work = workgroups[0] * workgroups[1] * workgroups[2];
    if (this.#exceedsTimeBudget(this.#pendingWork + work)) {
      this.#submitChunk();
    }
    this.#pending.push({ kind: "dispatch", pipeline, bindGroup, workgroups, work, key });
    this.#pendingWork += work;
    this.#dispatchCount += 1;
    if (this.#pending.length >= this.#chunkSizeLimit()) {
      this.#submitChunk();
    }
  }

  /**
   * Queues one buffer-to-buffer copy (`copyOutputs` of {@link Session.enqueue} — output slot →
   * {@link ResidentTensor}).
   *
   * MUST: queue it in **the same command stream as the dispatches**. Submitting it first on a
   * separate encoder would read a slot that has not been written yet. The stream is FIFO, so only
   * the results of dispatches queued before it become the copy source.
   * NOTE: its work is 0 (GPU execution time is negligible). It does not affect the time-budget
   * estimate and counts as one item only toward the chunk's dispatch-count limit.
   */
  copyBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void {
    this.#pending.push({
      kind: "copy",
      source,
      sourceOffset,
      destination,
      destinationOffset,
      size,
    });
    if (this.#pending.length >= this.#chunkSizeLimit()) {
      this.#submitChunk();
    }
  }

  /**
   * Submits the unsubmitted encodes (**without a fence** — {@link BatchMember}).
   *
   * There are two callers — `enqueue` (the section's fence is the one in `BatchScope.finish`) and a
   * single-fence run (the fence is the one readback `mapAsync` — H-1).
   *
   * MUST: `enqueue` always calls this at its end. A later `queue.writeBuffer` lands on the queue
   * timeline in issue order, so it **does not overtake earlier dispatches** (only unsubmitted
   * encodes are overtaken — ADR 0004 buffer-management invariant ④). A path that "leaves pending
   * work after enqueue" would break this invariant, and the next input write would be read by the
   * previous enqueue's dispatches — a silent wrong value.
   */
  submitPending(): void {
    this.#submitChunk();
  }

  /**
   * Paces a run's submissions (ADR 0123). If the submits since the previous marker have reached
   * {@link IN_FLIGHT_MARKER_INTERVAL}, one completion marker is placed; if pending markers exceed
   * {@link IN_FLIGHT_MAX_MARKERS}, a Promise that waits for the oldest marker is returned.
   * Otherwise it returns undefined and creates no wait (not even a microtask).
   *
   * Markers are placed only here (one per call). As long as the caller awaits the wait, pending
   * markers at the time of waiting are the limit + 1, and once the oldest completes they are back
   * to the limit.
   *
   * MUST: drain unsubmitted work before returning a wait. A `queue.writeBuffer` issued during the
   * wait then does not overtake an unsubmitted dispatch (keeping ADR 0004 buffer-management
   * invariant ④ during the wait too), and the GPU receives everything queued before we wait.
   * MUST: the caller calls this at a point where **no one else's dispatch can cut in** during the
   * wait — today only between steps inside a run's section lock (`executeBakedPlanPaced`). enqueue
   * does not call it (ADR 0123 decision 4 — waiting would let another Session's enqueue overtake
   * the call order. It places no marker either — on Deno a marker stops the host until GPU
   * completion, slowing a batch's generation loop).
   * MUST: turn a loss into {@link GpuDeviceLostError} (an implementation may never resolve
   * `onSubmittedWorkDone` after a loss — the same reason as flush).
   */
  backpressure(): Promise<void> | undefined {
    if (this.#windowChunks - this.#markedChunks >= IN_FLIGHT_MARKER_INTERVAL) this.#placeMarker();
    while (this.#markers[0]?.settled === true) this.#markers.shift();
    const oldest = this.#markers[0];
    if (oldest === undefined || this.#markers.length <= IN_FLIGHT_MAX_MARKERS) return undefined;
    this.#submitChunk();
    return this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(oldest.work, "in-flight submit bound wait");
  }

  /**
   * Closes the measurement window (paths that do not go through `flush` — settling a batch, and the
   * readback of a single-fence run).
   *
   * MUST: call it **after that path's fence** (`onSubmittedWorkDone` for a batch, the readback
   * `mapAsync` for a single-fence run). Calling it before would close the window mid-GPU-execution,
   * the measurement would come out too small, and chunks would swell (the dangerous side, toward
   * TDR territory).
   */
  closeMeasurementWindowAfterFence(): void {
    this.#closeMeasurementWindow();
  }

  /**
   * Discards the open measurement window without recording it (paths that never reached the fence —
   * when settling a batch fails at the errorScope check before `mapAsync`).
   *
   * MUST: call this instead of closing before the fence. Closing mid-GPU-execution would put a
   * too-small measurement into `msPerWorkgroup` and switch the chunk limit to `maxChunkSize`
   * permanently (the dangerous side, toward TDR territory — the flip side of the MUST of
   * {@link SubmitScheduler.closeMeasurementWindowAfterFence}). A discarded window only lacks
   * information, so the existing estimate is left alone (treated like a window measuring 0 —
   * invariant 1).
   */
  discardMeasurementWindow(): void {
    this.#windowStartedAt = undefined;
    this.#windowWork = 0;
    this.#windowChunks = 0;
    this.#markedChunks = 0;
  }

  /**
   * Discards unsubmitted encodes without submitting them.
   *
   * MUST: cleanup after a failed run goes through this path. There is no reason to execute the
   * remains, and a submit on the failure path happens outside the errorScope (after the caller has
   * already folded its scopes), so its validation error is attributed to **someone else's scope**
   * at the top of the per-device LIFO.
   * The list of measurement resources awaiting collection also holds those of a chunk that failed
   * midway through encoding (they are listed right after allocation), so flushing without going
   * through here would collect 0s for measurements that were never submitted.
   * Discarding also satisfies, together with {@link SubmitScheduler.flush}, "discard-or-flush
   * before destroy" — no encode that refers to a destroyed buffer remains to be submitted.
   *
   * NOTE: the open measurement window is not touched. The discarded part is not in its work (only
   * submitted chunks are counted), so the window closed by the cleanup flush comes out **too
   * large** by the time of the failure's unwinding — chunks shrink = the safe side, and the next
   * window restores the right value.
   */
  discard(): void {
    // Count only dispatches (a copy is not a record of "a dispatch that did not run").
    this.#discardedDispatches += this.#pending.filter((item) => item.kind === "dispatch").length;
    this.#pending.length = 0;
    this.#pendingWork = 0;
    // MUST: discard timestamp resources awaiting collection on the same path (the query version of
    // discard-or-flush before destroy). Left behind, a later cleanup flush would try to map "the
    // remains of a failed run".
    for (const timing of this.#pendingTimings) destroyTiming(timing);
    this.#pendingTimings.length = 0;
  }

  /**
   * Submits every unsubmitted encode and waits for GPU-side completion.
   *
   * MUST: always drain unsubmitted encodes before destroying buffers (success path — here when a
   * wait is needed, {@link SubmitScheduler.submitPending} when the wait was consolidated onto
   * another fence). If an unsubmitted encode refers to a destroyed buffer, the submit fails for the
   * whole command buffer, and even unrelated dispatches riding on the same scheduler never run,
   * silently leaving wrong values behind. On a failure path, {@link SubmitScheduler.discard} avoids
   * draining the remains.
   *
   * MUST: within this scheduler, `onSubmittedWorkDone` is called as a fence **only here** (the
   * attribution of measurements in the module doc — the in-flight bound's completion markers do not
   * close the window. The fence that settles a batch is set by `BatchScope`, and that window is
   * closed by {@link SubmitScheduler.closeMeasurementWindowAfterFence}).
   * The observation point of the adaptive control piggybacks on the same wait — the window is
   * closed right after the wait, and timestamp collection (`mapAsync`) is placed outside the window
   * (the cost of diagnostics is not mixed into the estimate).
   */
  async flush(): Promise<void> {
    // The rest is always within the budget and the limit (boundaries were decided at dispatch), so
    // one submit drains it.
    this.#submitChunk();
    // MUST: an implementation may never resolve onSubmittedWorkDone after a loss (measurements in
    // the doc of raceCanaryDeviceLost), so race it against the loss and turn it into an exception
    // (the subscription is always removed when it settles — GpuContext's responsibility).
    await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
      this.#device.queue.onSubmittedWorkDone(),
      "flush",
    );
    this.#closeMeasurementWindow();
    await this.#collectTimings();
  }

  /**
   * The GPU time breakdown of the latest run. undefined on a device with measurement disabled.
   *
   * NOTE: the table is built on every read (the only aggregate state is the per-key total Map). The
   * total fields are derived too — holding an independently updated denormalized field invites
   * mismatches such as the clamp count growing while the total does not move.
   */
  get timing(): GpuTimingStats | undefined {
    if (!this.#timingEnabled) return undefined;
    const entries: GpuTimingEntry[] = [...this.#timingTotals]
      .map(([key, total]) => ({ key, ...total }))
      // A table read in descending order. Ties are ordered by key to make it deterministic
      // (locale-independent comparison).
      .sort((a, b) => b.ns - a.ns || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    return {
      entries,
      totalNs: entries.reduce((total, entry) => total + entry.ns, 0),
      dispatchCount: entries.reduce((total, entry) => total + entry.dispatchCount, 0),
      clampedNegativeSamples: this.#clampedNegativeSamples,
    };
  }

  /**
   * Empties the breakdown. **Its lifetime is the latest run** (ADR 0021), so it is called at the
   * start of a run. Uncollected timestamp resources are not touched (the previous run always
   * drained them with flush / discard).
   */
  resetTiming(): void {
    this.#timingTotals.clear();
    this.#clampedNegativeSamples = 0;
  }

  /**
   * Cumulative number of real `queue.submit` calls issued (monotonically increasing; the same value
   * as {@link SubmitStats.submitCount}).
   *
   * Used to decide whether a failed generation run reached the state's physical ring (ADR 0066
   * addendum of 2026-08-17, item 3 — the executor's poison wiring). There is a separate accessor
   * although `stats` holds the same value because the decision reads it twice on the run's hot path
   * (`stats` copies the `measuredMs` array).
   * MUST: do not add another independently updated counter — only the one place in `#submitChunk`
   * moves it.
   */
  get submitCount(): number {
    return this.#submitCount;
  }

  get stats(): SubmitStats {
    return {
      submitCount: this.#submitCount,
      dispatchCount: this.#dispatchCount,
      measuredMs: [...this.#measuredMs],
      measuredCount: this.#measuredCount,
      discardedDispatches: this.#discardedDispatches,
      currentChunkSize: this.#chunkSizeLimit(),
      msPerWorkgroup: this.#msPerWorkgroup,
      chunkBudget: {
        // No independently updated field; derived from the policy and the safety factor on every
        // read.
        budgetMs: this.#chunkBudgetMs(),
        maxEstimatedMs: this.#maxEstimatedChunkMs,
        overBudgetChunks: this.#overBudgetChunks,
        maxWindowMeanMs: this.#maxWindowMeanMs,
        // Copied on every read so readers do not grab mutable internal state (undefined fields are
        // not written).
        ...(this.#submitGpuTime === undefined
          ? {}
          : {
              submitGpuTime: {
                submits: this.#submitGpuTime.submits,
                maxNs: this.#submitGpuTime.maxNs,
                unbackedSubmits: this.#submitGpuTime.unbackedSubmits,
                ...(this.#submitGpuTime.maxUnbackedNs === undefined
                  ? {}
                  : { maxUnbackedNs: this.#submitGpuTime.maxUnbackedNs }),
                maxDispatchNs: this.#submitGpuTime.maxDispatchNs,
                ...(this.#submitGpuTime.maxDispatchKey === undefined
                  ? {}
                  : { maxDispatchKey: this.#submitGpuTime.maxDispatchKey }),
              },
            }),
      },
    };
  }

  /**
   * The reference time for cutting one chunk (the same value {@link #exceedsTimeBudget} compares
   * against).
   */
  #chunkBudgetMs(): number {
    return this.#policy.timeBudgetMs * CHUNK_TIME_SAFETY;
  }

  /**
   * The current dispatch-count limit. While there is no measured evidence it stays at the bootstrap
   * value (invariant 1 — a measurement of 0 is no ground for growth).
   */
  #chunkSizeLimit(): number {
    return this.#msPerWorkgroup === undefined
      ? this.#policy.initialChunkSize
      : this.#policy.maxChunkSize;
  }

  /**
   * Whether a chunk with work `work` is expected to exceed the time budget. Without evidence no
   * decision is made (false = no cut by budget — the limit is held by the dispatch-count side).
   */
  #exceedsTimeBudget(work: number): boolean {
    const perWorkgroup = this.#msPerWorkgroup;
    if (perWorkgroup === undefined) return false;
    // Cut by budget only after at least minChunkSize are queued (a stop on the number of submits).
    // A dispatch that exceeds the budget on its own cannot be split here — it has to go as one
    // chunk.
    if (this.#pending.length < this.#policy.minChunkSize) return false;
    return work * perWorkgroup > this.#chunkBudgetMs();
  }

  #submitChunk(): void {
    const chunk = this.#pending;
    if (chunk.length === 0) {
      return;
    }
    this.#pending = [];
    const chunkWork = this.#pendingWork;
    this.#pendingWork = 0;
    const encoder = this.#device.createCommandEncoder();
    if (this.#timingEnabled) {
      // Whether there is evidence is the state when the boundary was decided (the estimate moves
      // only when a window closes, and the closing paths drain unsubmitted work first, so it does
      // not change while a chunk is being assembled).
      this.#encodeTimedChunk(encoder, chunk, this.#msPerWorkgroup !== undefined);
    } else {
      this.#encodePlainChunk(encoder, chunk);
    }
    const submittedAt = this.#now();
    this.#device.queue.submit([encoder.finish()]);
    this.#submitCount += 1;
    // MUST NOT: call onSubmittedWorkDone here (module doc — serialization of host and GPU). Only
    // the start of the measurement is kept; the flush side closes it.
    this.#windowStartedAt ??= submittedAt;
    this.#windowWork += chunkWork;
    this.#windowChunks += 1;
    this.#observeChunkBudget(chunkWork);
  }

  /**
   * Places one completion marker (ADR 0123).
   *
   * NOTE: on Deno 2.9.6 the synchronous part of `onSubmittedWorkDone` stops the host until "every
   * piece of work submitted so far has completed" (research 2026-08-04 §4.1). On that
   * implementation the GPU catches up when the marker is placed, and meanwhile the GPU idles
   * waiting for the next chunk (the wait's Promise only waits for something already resolved). If
   * the loss happens in this synchronous part, Deno panics instead of throwing (same as flush — the
   * Deno device-lost entry in docs/known-issues.md).
   * NOTE: a rejection is only counted as settled here and is not thrown. The waiting side awaits
   * the same Promise through `raceDeviceLost`, so the rejection surfaces there. The rejection of a
   * marker dropped without waiting is failed by the fence of that run / section on the same device
   * state.
   */
  #placeMarker(): void {
    this.#markedChunks = this.#windowChunks;
    const marker: InFlightMarker = {
      work: this.#device.queue.onSubmittedWorkDone(),
      settled: false,
    };
    const settle = (): void => {
      marker.settled = true;
    };
    marker.work.then(settle, settle);
    this.#markers.push(marker);
  }

  /**
   * Records one issued chunk in the diagnostics (**observation only** — no line leads from here
   * back to control).
   *
   * It uses "the estimate in effect at the time of cutting" as is. If the estimate moves in a later
   * window, it is not overwritten retroactively — what we want to see is "how the side that decided
   * that boundary estimated the budget", and repainting with a later estimate would erase that.
   */
  #observeChunkBudget(chunkWork: number): void {
    const perWorkgroup = this.#msPerWorkgroup;
    // Chunks from the time without evidence have no estimated time (nor were they cut by budget),
    // so they are not counted.
    if (perWorkgroup === undefined) return;
    const estimatedMs = chunkWork * perWorkgroup;
    if (this.#maxEstimatedChunkMs === undefined || estimatedMs > this.#maxEstimatedChunkMs) {
      this.#maxEstimatedChunkMs = estimatedMs;
    }
    // The comparison runs in the same direction as control (an overrun is "larger than the budget"
    // — a chunk exactly at the budget passes uncut).
    if (estimatedMs > this.#chunkBudgetMs()) this.#overBudgetChunks += 1;
  }

  /**
   * The default (non-measuring) encode. Still **1 chunk = 1 compute pass**, but a copy can only go
   * outside a pass, so when a copy comes the pass is closed and reopened by the next dispatch.
   *
   * MUST: reopening does not change the execution semantics (on the same queue execution follows
   * submit order and pass order, and storage visibility holds across pass boundaries). Not
   * reordering is the only invariant.
   */
  #encodePlainChunk(encoder: GPUCommandEncoder, chunk: readonly PendingCommand[]): void {
    let pass: GPUComputePassEncoder | undefined;
    for (const item of chunk) {
      if (item.kind === "copy") {
        pass?.end();
        pass = undefined;
        encoder.copyBufferToBuffer(
          item.source,
          item.sourceOffset,
          item.destination,
          item.destinationOffset,
          item.size,
        );
        continue;
      }
      pass ??= encoder.beginComputePass();
      pass.setPipeline(item.pipeline);
      pass.setBindGroup(0, item.bindGroup);
      pass.dispatchWorkgroups(...item.workgroups);
    }
    pass?.end();
  }

  /**
   * The measurement-mode encode (**1 dispatch = 1 pass**). The begin/end of each pass is written to
   * a querySet for one chunk, and `resolveQuerySet` → COPY_SRC buffer → MAP_READ buffer is queued
   * in the same command buffer (collection happens in {@link SubmitScheduler.flush} after
   * completion).
   *
   * MUST: do not use `writeTimestamp` (not in standard WebGPU — pass boundaries are the only
   * portable measurement points).
   */
  #encodeTimedChunk(
    encoder: GPUCommandEncoder,
    chunk: readonly PendingCommand[],
    backed: boolean,
  ): void {
    // Only dispatches are measured (a copy has no pass, so there is nowhere to write a timestamp).
    // NOTE: a batch cannot be opened on a device with measurement enabled (the
    // GpuContext.beginBatch guard), so a chunk with copies never takes this path — the stream order
    // is kept anyway.
    const dispatches = chunk.filter((item): item is PendingDispatch => item.kind === "dispatch");
    // Do not create a querySet for a chunk with nothing to measure (copies only) (no point creating
    // a zero-capacity resource only to collect it at once, and no reason to touch a surface that
    // differs between implementations).
    if (dispatches.length === 0) {
      this.#encodePlainChunk(encoder, chunk);
      return;
    }
    const count = dispatches.length * 2;
    if (count > MAX_TIMESTAMP_QUERIES) {
      throw new SubmitPolicyError(
        `GPU timing querySet capacity exceeded (${dispatches.length} dispatches = ${count} queries > ` +
          `${MAX_TIMESTAMP_QUERIES}). Set SubmitPolicy.maxChunkSize to ${
            MAX_TIMESTAMP_QUERIES / 2
          } or less, or set gpuTiming: false to turn timing off`,
      );
    }
    const byteLength = count * TIMESTAMP_BYTES;
    // MUST: put the 3 resources on the collection list (#pendingTimings) right after they are
    // allocated, and encode only after that. Then the resources of a chunk that exits with a
    // synchronous exception midway through encoding can also be destroyed from the same list by the
    // failure path's discard (exiting before they are listed puts them out of reach of both discard
    // and flush). If an allocation itself fails midway, nothing is listed yet, so the allocated
    // ones are destroyed here (they are not in any command yet, so destroying them on the spot
    // breaks neither submitted nor unsubmitted encodes).
    const allocated: { destroy(): void }[] = [];
    let timing: PendingTiming;
    try {
      const querySet = this.#device.createQuerySet({ type: "timestamp", count });
      allocated.push(querySet);
      const resolveBuffer = this.#device.createBuffer({
        size: byteLength,
        usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
      });
      allocated.push(resolveBuffer);
      const readBuffer = this.#device.createBuffer({
        size: byteLength,
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      timing = {
        querySet,
        resolveBuffer,
        readBuffer,
        entries: dispatches.map((item) => ({ key: item.key, work: item.work })),
        backed,
      };
    } catch (cause) {
      for (const resource of allocated) resource.destroy();
      throw cause;
    }
    this.#pendingTimings.push(timing);
    const { querySet, resolveBuffer, readBuffer } = timing;
    let index = 0;
    for (const item of chunk) {
      if (item.kind === "copy") {
        encoder.copyBufferToBuffer(
          item.source,
          item.sourceOffset,
          item.destination,
          item.destinationOffset,
          item.size,
        );
        continue;
      }
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: index * 2,
          endOfPassWriteIndex: index * 2 + 1,
        },
      });
      pass.setPipeline(item.pipeline);
      pass.setBindGroup(0, item.bindGroup);
      pass.dispatchWorkgroups(...item.workgroups);
      pass.end();
      index += 1;
    }
    encoder.resolveQuerySet(querySet, 0, count, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, byteLength);
  }

  /**
   * Collects the timestamps of completed chunks and aggregates them per key.
   *
   * MUST: call only **after** `onSubmittedWorkDone` (after the GPU execution of those chunks has
   * completed).
   * MUST: do not put the `mapAsync` wait on the completion-callback side (fire-and-forget). The
   * wait would escape the run's lock section, and GPU operations would be issued outside the
   * errorScope serialization (misattribution in the per-device LIFO). Waiting inside flush keeps
   * every GPU operation of the run inside the lock section.
   */
  async #collectTimings(): Promise<void> {
    if (this.#pendingTimings.length === 0) return;
    const pending = this.#pendingTimings;
    this.#pendingTimings = [];
    try {
      // MUST: an implementation may never resolve mapAsync after a loss (raced for the same reason
      // as readback).
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
        Promise.all(pending.map((item) => item.readBuffer.mapAsync(MAP_MODE.READ))),
        "timestamp collection",
      );
      for (const item of pending) {
        // Read entirely within the synchronous section, so no copy is needed (the view is valid
        // until unmap).
        const stamps = new BigUint64Array(item.readBuffer.getMappedRange());
        this.#accumulate(item.entries, stamps);
        this.#observeSubmitGpuTime(item, stamps);
        item.readBuffer.unmap();
      }
    } finally {
      // MUST: release regardless of the outcome. Even on failure no resources remain (destroy
      // implies unmap).
      for (const item of pending) destroyTiming(item);
    }
  }

  /**
   * Adds the differences of the query pairs to the per-key totals (order 1:1 with {@link
   * PendingTiming.entries}).
   */
  #accumulate(
    entries: readonly { readonly key: string; readonly work: number }[],
    stamps: BigUint64Array<ArrayBuffer>,
  ): void {
    for (const [index, entry] of entries.entries()) {
      const delta = stamps[index * 2 + 1] - stamps[index * 2];
      // MUST: clamp negatives to 0 and **keep the count** (no silent dropping — ADR 0021).
      if (delta < 0n) this.#clampedNegativeSamples += 1;
      const total = this.#timingTotals.get(entry.key) ?? {
        ns: 0,
        dispatchCount: 0,
        workgroupCount: 0,
      };
      total.ns += delta < 0n ? 0 : Number(delta);
      total.dispatchCount += 1;
      total.workgroupCount += entry.work;
      this.#timingTotals.set(entry.key, total);
    }
  }

  /**
   * Adds one chunk's GPU execution span and the maximum single dispatch to the lifetime observation
   * (**observation only** — no line leads from here back to control).
   *
   * The span is query minimum → maximum. Passes run in queued order, so this equals "begin of the
   * first pass → end of the last pass" and includes the gaps between passes (the same interval an
   * OS per-submit limit counts). Minimum / maximum are used so that non-monotonic samples never
   * produce a negative span (the clamp count is kept by {@link SubmitScheduler.#accumulate}).
   */
  #observeSubmitGpuTime(timing: PendingTiming, stamps: BigUint64Array<ArrayBuffer>): void {
    let first = stamps[0];
    let last = stamps[0];
    for (const stamp of stamps) {
      if (stamp < first) first = stamp;
      if (stamp > last) last = stamp;
    }
    const spanNs = Number(last - first);
    const observed = (this.#submitGpuTime ??= {
      submits: 0,
      maxNs: 0,
      unbackedSubmits: 0,
      maxUnbackedNs: undefined,
      maxDispatchNs: 0,
      maxDispatchKey: undefined,
    });
    observed.submits += 1;
    observed.maxNs = Math.max(observed.maxNs, spanNs);
    if (!timing.backed) {
      observed.unbackedSubmits += 1;
      observed.maxUnbackedNs = Math.max(observed.maxUnbackedNs ?? 0, spanNs);
    }
    for (const [index, entry] of timing.entries.entries()) {
      const delta = stamps[index * 2 + 1] - stamps[index * 2];
      const deltaNs = delta < 0n ? 0 : Number(delta);
      if (deltaNs > observed.maxDispatchNs) {
        observed.maxDispatchNs = deltaNs;
        observed.maxDispatchKey = entry.key;
      }
    }
  }

  /**
   * Closes the measurement window and updates the estimate (called right after flush's wait for
   * `onSubmittedWorkDone`).
   *
   * MUST: do **not update** for a window whose measurement is 0 (the whole window below timer
   * resolution). 0 means "no information", not "fast"; treating it as ground for growth would make
   * chunks swell unconditionally in environments with coarse resolution, ending in everything in
   * one submit (= TDR territory). A window with 0 work has no information for the same reason
   * (grabbing Infinity from a division by 0 would instead pin the size at minChunkSize forever).
   */
  #closeMeasurementWindow(): void {
    const startedAt = this.#windowStartedAt;
    // Window not open = nothing submitted since the last flush. There is nothing to record
    // (counting an empty flush as "a window measuring 0" would fill the diagnostic windows with
    // empty entries).
    if (startedAt === undefined) return;
    const measured = this.#now() - startedAt;
    const work = this.#windowWork;
    // The window start and the chunk count are set together in #submitChunk, so with an open window
    // the count is always at least 1 (no division by 0).
    const meanChunkMs = measured / this.#windowChunks;
    if (this.#maxWindowMeanMs === undefined || meanChunkMs > this.#maxWindowMeanMs) {
      this.#maxWindowMeanMs = meanChunkMs;
    }
    this.#windowStartedAt = undefined;
    this.#windowWork = 0;
    this.#windowChunks = 0;
    this.#markedChunks = 0;
    this.#measuredMs.push(measured);
    if (this.#measuredMs.length > MEASURED_HISTORY) {
      this.#measuredMs.shift();
    }
    this.#measuredCount += 1;
    if (measured > 0 && work > 0) {
      this.#msPerWorkgroup = measured / work;
    }
  }
}
