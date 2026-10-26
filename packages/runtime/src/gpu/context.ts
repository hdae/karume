/**
 * The container of an acquired device ({@link GpuContext}), and the resident tensors / batch
 * sections it owns.
 *
 * MUST: this file does not **value-import** `acquire.ts`. The layer's dependency is one-way,
 * acquire → context, and only the `acquireGpu` side constructs a {@link GpuContext} (the reverse
 * reference is `import type` only — it is erased, so the runtime import graph stays one-way).
 * The entry point of the layer is `device.ts`, which re-exports the public names of these 2 files.
 *
 * This layer also has three more containers **owned by GpuContext with the same lifetime as the
 * device**: {@link ResidentTensor} (a 4th lifetime class that can be shared across Sessions),
 * {@link BatchScope} (an enqueue section closed by one fence), and
 * {@link GpuContextInternals.pipelines} (the cache of compiled compute pipelines).
 * The first two are GPU-buffer containers but live here because the discipline of errorScope
 * sections and the loss race applies to them as is.
 */

// MUST: take types only (a value import would create the dependency this layer (context.ts) →
// canary → kernels / reference, breaking the rule "this layer creates no import into kernels /
// reference" — {@link GpuContextInternals.attentionI8a8Dot}). `import type` is erased, so the
// runtime import graph stays one-way as before.
import type { AttentionI8a8Decision } from "./attention-dp4a-canary.ts";
// MUST: take only types from acquire.ts too (taking a value would break "acquire → context,
// one-way" at the top). The reference is erased, so the runtime import graph stays acquire →
// context.
import type { DeviceLostHandler, RequiredLimits } from "./acquire.ts";
// MUST: take only types from kernels too (the same reason as attention-dp4a-canary above — no value
// import from this layer into kernels).
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { STORAGE_USAGE } from "./arena.ts";
import {
  discardFailureScopes,
  GpuOutOfMemoryError,
  popFailureScopes,
  pushFailureScopes,
} from "./error-scope.ts";
import { PipelineCache } from "./pipeline-cache.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** An operation on a device that has been lost. Always thrown instead of waiting forever. */
export class GpuDeviceLostError extends Error {
  override readonly name = "GpuDeviceLostError";
}

/**
 * A suffix that adds the loss reason to the message of {@link GpuDeviceLostError} (an empty string
 * when not lost or without information).
 *
 * MUST: every path that turns a loss into an exception (the wait of `#raceDeviceLost`, the
 * synchronous check of {@link assertDeviceUsable}, `raceCanaryDeviceLost` of the shader-f16 canary,
 * the check of the same name in `runtime/generation-context.ts`) goes through this.
 * `GPUDeviceLostInfo.message` is **the only survivor of the real cause the backend put in** (the
 * equivalent of Metal's `Device::create_query_set: ...`) and does not reach the errorScope — errors
 * of a lost device are dropped at the entrance of Deno's error handler, so dropping it here leaves
 * the real cause nowhere in karume (the Metal timestamp section of `docs/limitations.md`).
 * The **prefix** does not change: existing callers and docs assume the form that lets the reader
 * see "where it was lost" first (the reason is extra information appended after it).
 */
export const describeDeviceLoss = (info: GPUDeviceLostInfo | undefined): string => {
  if (info === undefined) return "";
  const message = info.message.trim();
  return ` — reason: ${info.reason}${message === "" ? "" : ` / ${message}`}`;
};

/**
 * A breach of the lifetime discipline of a resident tensor ({@link ResidentTensor}) (use after
 * dispose, dispose while referenced).
 */
export class ResidentTensorError extends Error {
  override readonly name = "ResidentTensorError";
}

/**
 * A breach of how a batch section ({@link BatchScope}) is used (enqueue after settlement, combining
 * with measurement, and the like).
 */
export class BatchScopeError extends Error {
  override readonly name = "BatchScopeError";
}

/** The feature needed for GPU-side time measurement (timestamps at pass boundaries) — ADR 0021. */
export const TIMESTAMP_QUERY_FEATURE: GPUFeatureName = "timestamp-query";

/**
 * The feature needed by the f16 **compute** variants (GEMM with f16 shared tiles — `enable f16`).
 *
 * MUST: do not confuse it with f16 weight **storage** (ADR 0018). The storage side works with core
 * WGSL's `unpack2x16float` alone and never requests this feature. This side uses WGSL's `f16` type
 * itself, so without the feature the shader does not compile.
 */
export const SHADER_F16_FEATURE: GPUFeatureName = "shader-f16";

/**
 * The key of **the surface that only the runtime touches** (not exported from mod.ts — it does not
 * pollute the "thin surface" of ADR 0008). {@link GpuContext} / {@link ResidentTensor} /
 * {@link BatchScope} carry only the few methods for users under plain names, and fold what the
 * executor needs (serialization primitives, GPUBuffer, baked reference counts, member registration)
 * under this key. The same style as the test-only knob (`I8A8_DOT`).
 *
 * MUST: declare it **before** {@link GpuContext}. Computed property names in a class body are
 * evaluated when the class is defined, so placing it after would fail in the TDZ.
 */
export const RUNTIME_INTERNAL: unique symbol = Symbol("karume.runtimeInternal");

/**
 * The runtime-internal surface of {@link GpuContext} (serialization primitives that do not map to a
 * user story).
 */
type GpuContextInternals = {
  /**
   * Races the settling of `work` against device loss, and turns it into {@link GpuDeviceLostError}
   * if the loss comes first.
   *
   * MUST: a wait can become a hang rather than a failure (whether `onSubmittedWorkDone` /
   * `mapAsync` resolve after a loss is a surface that differs between implementations —
   * measurements are recorded in the doc of `raceCanaryDeviceLost` in {@link "./acquire.ts"}).
   * On settling, the subscription is always removed (no reactions are left behind).
   */
  raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T>;
  /**
   * Runs a section that keeps errorScopes pushed across `await`s, serialized per device.
   * For the full rules see "Invariants of errorScope sections" at the top of {@link GpuContext}.
   *
   * Trade-off: the held section spans the whole GPU work of one run (encoding to the wait for
   * `flush()` completion to readback to arena destruction), so runs of several Sessions on the same
   * device are serialized whole. The section is not narrowed to keep the invariant "a run issues
   * GPU operations only inside its own lock section" simple (see #runOnce in the executor). The
   * design is still taken because the consequence of misattribution is "an unrelated run fails, and
   * the run that should fail silently returns all 0" — a silent wrong value has no means of
   * detection and does not weigh against the lost throughput.
   *
   * MUST: do not expose it on the user surface. Published under a plain name,
   * `gpu.withScopeLock(() => session.run(...))` could be written, and run would go for the same
   * lock and **self-deadlock with neither diagnostics nor exception** (no reentrancy detector can
   * be put in — see the doc below).
   */
  withScopeLock<T>(body: () => Promise<T>): Promise<T>;
  /**
   * Decides the integer-dot-product variant of fused attention **only once per device** (lazy,
   * memoized).
   *
   * `run` is shaped so that the caller passes the actual decision (`decideAttentionI8a8Dot` of
   * {@link "./attention-dp4a-canary.ts"}) — so that this layer creates no import into the kernels /
   * reference layers (the import in the other direction is made by the canary side). MUST: there is
   * only one call site, in Session construction (a shape that accepts several implementations would
   * turn the memo into a field that merely means "the decision passed first").
   *
   * MUST: memoize **the Promise itself** (not the value). When Sessions with attentionCompute "a8"
   * are built concurrently, memoizing the value would run the canary twice and break the contract
   * of once per device. A failure (both arms outside the sanity band / device loss) is also handed
   * out as the same Promise — a device property does not change by rerunning, so a retry would only
   * pay 1 submit to reach the same conclusion.
   * MUST: but a decision that failed with {@link GpuOutOfMemoryError} is removed from the memo (a
   * request after the settlement reruns it). A failed allocation of the canary's temporary buffers
   * is not a device property but a transient state of the moment when other Sessions filled VRAM
   * near the ceiling, so keeping it would make it impossible to create an a8 Session on the same
   * GpuContext even after memory frees up. It is removed only when the held Promise is the very one
   * that failed (sharing among in-flight requests is not broken).
   *
   * The slot holds **the whole decision** ({@link AttentionI8a8Decision}), not just the one variant
   * value. The fact "it did not match the known answer exactly but was inside the band, so it was
   * let through" has the same lifetime as the decision; collapsing it to a value would leave the
   * caller unable to warn.
   */
  attentionI8a8Dot(run: () => Promise<AttentionI8a8Decision>): Promise<AttentionI8a8Decision>;
  /**
   * The cache of compiled compute pipelines (**one per device** — created on first reference).
   *
   * Its lifetime is per device because whether a pipeline can be reused is decided per device —
   * Sessions on the same device merely hit the browser's implicit cache (keyed by the WGSL string),
   * and splitting an explicit cache per Session would pay the `createShaderModule` /
   * `createComputePipeline` calls and the resolution of `getBindGroupLayout` once per Session.
   *
   * MUST: do not drop it explicitly on destroy or loss (**it lives and dies with the GpuContext
   * instance** — the rebuild discipline of ADR 0004). It is created lazily, so resetting it to
   * undefined on destroy would make the next reference build "the cache of a dead device" as if
   * new. The only way to rebuild is reconstructing the GpuContext from `acquireGpu()`, which gives
   * a new, empty cache at that point.
   * MUST: the caller (executor) **does not touch it from the Session construction phase** — the
   * construction phase is outside the scope lock, and creating pipelines there would misattribute
   * the errorScopes of concurrent constructions ("Invariants of errorScope sections" at the top of
   * {@link GpuContext}).
   */
  pipelines(): PipelineCache;
  /**
   * Emits a warning with the same `key` to `console.warn` **only once per device**.
   *
   * A fact that is a property of the device itself, such as "this device has no dp4a", needs saying
   * only once however many Sessions are created — this avoids the same message flowing once per
   * Session in families that line up a8 Sessions per component. The caller builds the message (so
   * that this layer creates no import into the kernels / runtime layers).
   */
  warnOnce(key: string, message: string): void;
  /**
   * Whether GPU timing is suspended — true while a batch holds the device's section (one at a time:
   * the scope lock). `SubmitScheduler` then encodes plain chunks, so enqueues are not timed and no
   * per-pass timestamps wait for a flush the batch never runs. Runs never overlap a batch on the
   * same device, so they stay timed (ADR 0021 addendum of 2026-10-24).
   */
  timingSuspended(): boolean;
  /**
   * Marks the start of a batch section and returns the function that marks its end.
   * MUST: called only by `BatchScope` inside its scope-lock section, with the end in a `finally`
   * before the lock is returned — a section left open would leave every later run untimed.
   */
  openBatchSection(): () => void;
  /**
   * The GEMM geometry profile injected with `acquireGpu({ geometryProfile })` (DECIDED: ADR 0115).
   * undefined without an injection, in which case Session construction picks an embedded table from
   * the adapter.
   *
   * The only reader is the selection in Session construction (the geometry of the dp4a canary and
   * the `geometryProfile` of the diagnostics take the result of that selection). It is not on the
   * user surface because whoever passed it knows the value, and the table a Session actually used
   * can be observed through `SessionDiagnostics.geometryProfile`.
   * MUST: unchanged for the device's lifetime (that the Sessions and the canary's decision memo on
   * the same device correspond to one table is the ground for memoizing the decision per device).
   */
  readonly geometryProfile?: GeometryProfile;
};

/**
 * A synchronous check of whether the device is usable (the admission guard of the resident-tensor
 * path — ADR 0054). Session construction (`runtime/session-build.ts`), the admission of run /
 * enqueue (`runtime/executor.ts`) and the shut-off surface of GenerationContext
 * (`runtime/generation-context.ts` — ADR 0066 decision 7) borrow the same check (the check lives
 * here alone and is runtime-internal — not exposed from mod.ts).
 *
 * MUST: look at `destroyRequested` as well as `lost`. `destroy()` sets the flag synchronously, but
 * the reaction of `device.lost` runs in a later task, so with `lost` alone operations pass in that
 * window. The reason this exists is that passing shows up silently — {@link ResidentTensor.write}
 * and GenerationContext's write of the logical-length uniform become **silent no-ops** on destroyed
 * buffers (not even a warning), and {@link GpuContext.createResident} **returns as a success while
 * holding an invalid buffer**, because `popErrorScope` resolves with null after a loss
 * (`docs/research/2026-08-16-device-lost-wait-settlement.md`). Either way it becomes loud only at
 * the next fence, and the diagnostics in between mislead.
 * MUST: the type is {@link GpuDeviceLostError} (GPU resources from a lost device are unrecoverable
 * per the WebGPU specification, and the recovery is the same for an intentional destroy and an
 * unexpected loss, so the types are not split).
 */
export const assertDeviceUsable = (gpu: GpuContext, where: string): void => {
  if (gpu.destroyRequested || gpu.lost !== undefined) {
    throw new GpuDeviceLostError(
      `${where}: device was lost (re-acquire the device and rebuild)` +
        describeDeviceLoss(gpu.lost),
    );
  }
};

/**
 * A bundle of an acquired device and its normalized capabilities.
 *
 * The subscription to `device.lost` is made in the constructor **unconditionally and only once**.
 * Leaving a loss unhandled makes `mapAsync` / `onSubmittedWorkDone` never resolve, showing up as a
 * hang rather than a failure. Because the subscription is not optional, a loss is always recorded
 * in {@link GpuContext.lost}, and the waits of this layer are turned into exceptions.
 *
 * ## Invariants of errorScope sections (per device, enforced by this layer)
 *
 * errorScopes are a **per-device LIFO stack**, and `popErrorScope()` unconditionally takes the top
 * of the stack at the time it is called. That is, there is no notion of "whose scope". Three rules
 * follow:
 *
 * - MUST: a section that keeps errorScopes pushed across `await`s runs inside `withScopeLock`
 *   ({@link GpuContextInternals}). If sections overlap, ① one's errors land in another's scope and
 *   ② one's pop takes another's scope, so a failure is attributed to an unrelated call and the call
 *   that should fail silently returns all 0.
 * - MUST NOT: re-acquire `withScopeLock` inside `withScopeLock` (self-deadlock). The lock is not
 *   reentrant. No detector is put in — a "held" flag cannot tell **a legitimate queue** (another
 *   Session waiting in line for an earlier section to complete) from reentry, and telling them
 *   apart needs execution-context tracking across async calls. Under the constraint of Web standard
 *   APIs only (ADR 0002) there is no pure means, so it is kept by a layering rule: the acquisition
 *   point is limited to one place in the executor, and the layers inside it (PipelineCache /
 *   SubmitScheduler / RunArena) use only scopes that complete within a synchronous section.
 * - Scopes that complete within a synchronous section (`withPipelineScope` /
 *   {@link pushFailureScopes} to {@link popFailureScopes} of {@link "./error-scope.ts"}) need no
 *   lock. No other task can cut in, and the LIFO nesting always balances. Layers called from inside
 *   the lock (PipelineCache and the like) have to take this form.
 */
export class GpuContext {
  /** The runtime-internal surface (not a surface users touch). */
  readonly [RUNTIME_INTERNAL]: GpuContextInternals;
  /**
   * The raw device (a surface touched outside the runtime's management). Calling `destroy()` from
   * here bypasses the suppression of `AcquireGpuOptions.onDeviceLost` in {@link "./acquire.ts"} and
   * is reported as an unexpected loss, and pushing `pushErrorScope` / `popErrorScope` here does not
   * take part in the errorScope section lock (LIFO exclusion). Use {@link GpuContext.destroy} for
   * cleanup.
   */
  readonly device: GPUDevice;
  readonly adapterInfo: GPUAdapterInfo;
  /** Features the device actually enabled (not including features the adapter merely has). */
  readonly features: ReadonlySet<string>;
  /**
   * WGSL language features (an empty set on implementations without them). On their own they are
   * not proof of GPU support. They may be used only for choosing between numerically identical
   * variants (dot4I8Packed / emulation of w8a8 — ADR 0025) and for checking a necessary condition
   * when explicitly requested (subgroup_id — the device feature and a canary run are required too).
   * The source of truth for the line is `readWgslLanguageFeatures` in {@link "./acquire.ts"}.
   */
  readonly wgslLanguageFeatures: ReadonlySet<string>;
  /**
   * Whether the device maps `local_invocation_index` linearly onto subgroups (lane = index % 32,
   * subgroup = index / 32), measured by `acquireGpu` when the subgroup features were requested
   * (false otherwise). The online attention's subgroup form relies on it (ADR 0132 decision 5).
   */
  readonly subgroupLanesLinear: boolean;
  readonly limits: RequiredLimits;
  #lost: GPUDeviceLostInfo | undefined;
  /** A batch holds the section ({@link GpuContextInternals.timingSuspended}). */
  #batchSectionOpen = false;
  #destroyRequested = false;
  readonly #lostListeners = new Set<DeviceLostHandler>();
  /**
   * The serialization chain of errorScope sections. It passes on only the settlement (success or
   * failure), so it never rejects itself (one failure does not drag every later section down).
   */
  #scopeChain: Promise<void> = Promise.resolve();
  /**
   * The result of the dot-product variant canary of fused attention
   * ({@link GpuContextInternals.attentionI8a8Dot}). Stays undefined unless requested = not a single
   * dispatch is issued (users who do not use a8 pay nothing).
   */
  #attentionI8a8Dot: Promise<AttentionI8a8Decision> | undefined;
  /**
   * Issues the identifiers of {@link ResidentTensor}. **Not placed at module scope** (the
   * zero-side-effect invariant) — a separate space per GpuContext is enough (a resident does not
   * cross devices).
   */
  #residentSeq = 0;
  /**
   * The device-lifetime pipeline cache ({@link GpuContextInternals.pipelines}).
   * Stays undefined until referenced = none is created (users who create no Session pay nothing).
   */
  #pipelines: PipelineCache | undefined;
  /** Keys that {@link GpuContextInternals.warnOnce} has already emitted. */
  readonly #warned = new Set<string>();

  constructor(
    device: GPUDevice,
    adapterInfo: GPUAdapterInfo,
    limits: RequiredLimits,
    wgslLanguageFeatures: ReadonlySet<string>,
    onDeviceLost?: DeviceLostHandler,
    geometryProfile?: GeometryProfile,
    subgroupLanesLinear = false,
  ) {
    this.device = device;
    this.subgroupLanesLinear = subgroupLanesLinear;
    this.adapterInfo = adapterInfo;
    this.limits = limits;
    this.features = new Set(device.features);
    this.wgslLanguageFeatures = wgslLanguageFeatures;
    this[RUNTIME_INTERNAL] = {
      raceDeviceLost: <T>(work: Promise<T>, where: string): Promise<T> =>
        this.#raceDeviceLost(work, where),
      withScopeLock: <T>(body: () => Promise<T>): Promise<T> => this.#withScopeLock(body),
      attentionI8a8Dot: (
        run: () => Promise<AttentionI8a8Decision>,
      ): Promise<AttentionI8a8Decision> => {
        if (this.#attentionI8a8Dot !== undefined) return this.#attentionI8a8Dot;
        const decision = run();
        this.#attentionI8a8Dot = decision;
        // Registered before the caller's reaction, so the memo is already empty by the time the
        // caller receives the rejection. The caller gets decision itself, so the rejection reaches
        // the caller as is.
        decision.catch((cause: unknown) => {
          if (cause instanceof GpuOutOfMemoryError && this.#attentionI8a8Dot === decision) {
            this.#attentionI8a8Dot = undefined;
          }
        });
        return decision;
      },
      pipelines: (): PipelineCache => {
        this.#pipelines ??= new PipelineCache(this.device);
        return this.#pipelines;
      },
      warnOnce: (key: string, message: string): void => {
        if (this.#warned.has(key)) return;
        this.#warned.add(key);
        console.warn(message);
      },
      geometryProfile,
      timingSuspended: (): boolean => this.#batchSectionOpen,
      openBatchSection: (): (() => void) => {
        if (this.#batchSectionOpen) {
          throw new Error(
            "A batch section is already open on this device (broken scope-lock discipline)",
          );
        }
        this.#batchSectionOpen = true;
        return () => {
          this.#batchSectionOpen = false;
        };
      },
    };
    if (onDeviceLost !== undefined) {
      this.onLost((info) => {
        if (!this.#destroyRequested) {
          onDeviceLost(info);
        }
      });
    }
    void device.lost.then((info) => {
      this.#lost = info;
      // Copy before calling so that unsubscribing during notification does not break the
      // iteration. A loss happens only once, so after notifying, the subscriptions are emptied and
      // later onLost calls fall to the immediate-notification path.
      const listeners = [...this.#lostListeners];
      this.#lostListeners.clear();
      for (const listener of listeners) {
        // MUST: a listener's exception does not stop the notification fan-out. The subscriptions
        // were cleared above and there is no renotification, so if a throw from the public
        // onDeviceLost (first in insertion order) kept the notification from reaching the later
        // internal subscriptions (`raceDeviceLost`), the loss would become a hang instead of an
        // exception. The caught exception is not swallowed; it is detached from the loss's control
        // path and rethrown asynchronously.
        try {
          listener(info);
        } catch (cause) {
          queueMicrotask(() => {
            throw cause;
          });
        }
      }
    });
  }

  /** The information if lost; undefined if not. Also recorded after `destroy()`. */
  get lost(): GPUDeviceLostInfo | undefined {
    return this.#lost;
  }

  /**
   * Whether an intentional destroy has been requested (to tell it apart from an unexpected loss).
   */
  get destroyRequested(): boolean {
    return this.#destroyRequested;
  }

  /**
   * Whether GPU-side time measurement is enabled (ADR 0021).
   *
   * MUST: do not copy the requested value into another field; derive it from **the features
   * actually enabled**. The device's feature set is exactly what was requested, so a copy would let
   * the diagnostics alone claim it is enabled when "requested but not granted" (an empty breakdown
   * that looks like "measuring").
   */
  get gpuTimingEnabled(): boolean {
    return this.features.has(TIMESTAMP_QUERY_FEATURE);
  }

  /**
   * Whether the f16 compute variants are usable (ADR 0028).
   *
   * MUST: by the same discipline as {@link GpuContext.gpuTimingEnabled}, derived from **the
   * features actually enabled**, not from a copy of the requested value. This returns true only
   * when `acquireGpu({shaderF16: true})` passed both the feature request and the canary run.
   */
  get shaderF16Enabled(): boolean {
    return this.features.has(SHADER_F16_FEATURE);
  }

  /**
   * The number of loss subscriptions not yet removed (diagnostics).
   *
   * `raceDeviceLost` ({@link GpuContextInternals}) always unsubscribes on settling, so when every
   * wait has settled the count returns to its baseline. The baseline is not 0 and depends on the
   * configuration: when constructed with `onDeviceLost`, it includes the resident one set by the
   * constructor (baseline 1), and subscriptions a user set with {@link GpuContext.onLost} and has
   * not removed also count. Not returning to the baseline is what the leak "a reaction is left
   * behind per flush / readback" looks like, and it is not observable from behavior (no hang, no
   * wrong value — it only grows monotonically in a long-lived Session).
   * A surface only for showing an invisible residue honestly as a number.
   */
  get pendingLostListeners(): number {
    return this.#lostListeners.size;
  }

  /**
   * Subscribes to device loss. The returned function unsubscribes. If the device is already lost at
   * subscription time, it notifies **immediately (synchronously)** so that nothing is missed.
   *
   * MUST: do not set a new `device.lost.then(...)` for every wait. `lost` stays unresolved for the
   * device's lifetime, so each `.then` piles up on the promise as a reaction with no way to remove
   * it, growing monotonically per flush / readback (a leak in a long-lived Session). Subscription
   * is unified here, and waiting sides use `raceDeviceLost` ({@link GpuContextInternals}), which
   * always unsubscribes on settling.
   */
  onLost(listener: DeviceLostHandler): () => void {
    const info = this.#lost;
    if (info !== undefined) {
      listener(info);
      return () => {};
    }
    this.#lostListeners.add(listener);
    return () => {
      this.#lostListeners.delete(listener);
    };
  }

  /**
   * The implementation of {@link GpuContextInternals.raceDeviceLost} (exposed only under
   * `RUNTIME_INTERNAL`).
   */
  async #raceDeviceLost<T>(work: Promise<T>, where: string): Promise<T> {
    let unsubscribe: () => void = () => {};
    const lost = new Promise<never>((_resolve, reject) => {
      unsubscribe = this.onLost((info) => {
        reject(
          new GpuDeviceLostError(
            `The device was lost during ${where} (rebuild required)${describeDeviceLoss(info)}`,
          ),
        );
      });
    });
    try {
      return await Promise.race([work, lost]);
    } finally {
      unsubscribe();
    }
  }

  /**
   * The implementation of {@link GpuContextInternals.withScopeLock} (exposed only under
   * `RUNTIME_INTERNAL`).
   */
  #withScopeLock<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#scopeChain.then(body);
    this.#scopeChain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /**
   * Creates a resident buffer that can be shared across Sessions (**the 4th lifetime class** — it
   * belongs to none of the weight arena / slot backing / run arena).
   *
   * Its use is "values to keep on the GPU throughout a generation loop" — condition tensors (put in
   * once with {@link ResidentTensor.write} before the loop) and latents passed between steps
   * (written by `copyOutputs` of {@link Session.enqueue} and used as the input of the next
   * enqueue). Neither goes through the host even once, so the fences at run boundaries can be
   * removed.
   *
   * MUST: it is async in order to wrap it in errorScopes. A `createBuffer` over a limit / out of
   * memory throws no synchronous exception and **returns an invalid buffer**, so without the scopes
   * later `writeBuffer`s become no-ops without even a warning, and the generation loop runs with an
   * empty resident tensor.
   * MUST: `byteLength` is a multiple of 4 (every element type is 4 bytes — the semantic dtypes of
   * ADR 0009).
   * MUST: a lost device is not accepted ({@link assertDeviceUsable} — errorScopes do not catch a
   * loss, so even with the scopes an invalid buffer comes back as a success). The check runs twice:
   * at the entrance and after the pop settles (a loss during the wait is not visible at the
   * entrance).
   */
  async createResident(byteLength: number, label = "resident"): Promise<ResidentTensor> {
    assertDeviceUsable(this, `resident '${label}' allocation`);
    if (!Number.isInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0) {
      throw new ResidentTensorError(
        `resident '${label}': byteLength must be a positive integer multiple of 4: ${byteLength}`,
      );
    }
    // MUST: no await between the push and the issue of the pop (the ground for preventing
    // interleaving in the per-device LIFO — the 3rd of "Invariants of errorScope sections" at the
    // top of the class).
    pushFailureScopes(this.device);
    let buffer: GPUBuffer;
    try {
      buffer = this.device.createBuffer({ label, size: byteLength, usage: STORAGE_USAGE });
    } catch (cause) {
      await discardFailureScopes(this.device);
      throw cause;
    }
    const failure = await popFailureScopes(this.device, `resident '${label}' allocation`);
    if (failure !== undefined) {
      buffer.destroy();
      throw failure;
    }
    // MUST: do not return a success even for a device lost while waiting for the pop. A pop after a
    // loss settles with null (docs/research/2026-08-16-device-lost-wait-settlement.md), so with
    // only the entrance check an unusable resident tensor would circulate as a successful
    // allocation, and the failure point would shift to the first write / read (the same shape as
    // the check after the pop in `BatchScope.finish`).
    try {
      assertDeviceUsable(this, `resident '${label}' allocation`);
    } catch (cause) {
      buffer.destroy();
      throw cause;
    }
    this.#residentSeq += 1;
    return new ResidentTensor(this, this.#residentSeq, buffer, byteLength, label);
  }

  /**
   * Opens a section that bundles enqueues without fences ({@link Session.enqueue} — H-5).
   *
   * For the duration of the section it **keeps holding the per-device errorScope section lock**
   * (`withScopeLock` — {@link GpuContextInternals}). This is the ground for "every GPU operation
   * issued inside the batch is attributed to the batch's errorScope" and for "a pop does not take
   * the wrong top of the stack".
   *
   * MUST: **do not issue** {@link Session.run} / `dispose` on the same device during the section
   * (with or without await). run goes for the section lock and does not settle until
   * {@link BatchScope.finish}; if that run sits before an `enqueue` of the same Session, the 4
   * edges "finish → in-flight lease → enqueue body → earlier run → section lock" close into a
   * certain self-deadlock (the same for an un-awaited run issued **before** the section was opened
   * — `beginBatch` grabs the lock synchronously in the constructor, so within the same tick the
   * batch is always first). `Session.enqueue` turns this shape into a {@link BatchScopeError}.
   * NOTE: even in shapes that do not form a cycle, **any Session operation awaited during the
   * section** (`session.dispose()` / `context.dispose()`) does not return until `finish()` if an
   * unsettled run is ahead, which looks like a hang to the user. During a batch, use only `enqueue`
   * and `finish`. **The same goes for Session construction** — the first construction with
   * `attentionCompute: "a8"` goes for the section lock in the canary
   * ({@link GpuContextInternals.attentionI8a8Dot}), so constructing during the section does not
   * return until `finish()`.
   * NOTE: on a device with measurement enabled the section is **not timed** — its enqueues are
   * encoded as plain chunks ({@link GpuContextInternals.timingSuspended}), because per-pass
   * timestamps are collected only by a run's flush and would pile up uncollected during the batch.
   * Runs before and after the section keep their breakdown (ADR 0021 addendum of 2026-10-24).
   * NOTE: trying to open a second one on the same device waits here until the first returns the
   * lock in {@link BatchScope.finish} (sections are exclusive per device — they do not nest).
   * MUST: do not open on a lost or destroy-requested device ({@link assertDeviceUsable}). Opening
   * would take the section lock and keep accepting enqueues, delaying the failure until `finish()`.
   * Detecting a loss in the middle of a section at finish is an accepted part of the design, but a
   * device already unusable when opening can be judged synchronously.
   * The check runs twice: at the entrance and right after the section lock is acquired (a loss
   * while waiting for an earlier section to release the lock is not visible at the entrance).
   */
  async beginBatch(): Promise<BatchScope> {
    assertDeviceUsable(this, "batch start");
    const batch = new BatchScope(this);
    await batch[RUNTIME_INTERNAL].entered;
    return batch;
  }

  /**
   * Destroys the device (the last path that returns all the VRAM the device holds — individual
   * buffers return with `destroy()` + poll).
   *
   * MUST: flush / destroy the layers holding unsubmitted encodes and live buffers (RunArena) before
   * calling this. Continuing after destroy means rebuilding with `acquireGpu()`.
   */
  destroy(): void {
    this.#destroyRequested = true;
    this.device.destroy();
  }
}

/** The runtime-internal surface of {@link ResidentTensor}. */
type ResidentInternals = {
  /** The backing object that becomes the target of baking / aliasing. */
  readonly buffer: GPUBuffer;
  /**
   * An identifier unique within the GpuContext. **It goes into the key of derived plans and into
   * the backing signature**, so replacing a resident gives another signature (= rebaking), and
   * putting it back hits the original backing.
   */
  readonly id: number;
  /**
   * The GpuContext that allocated this backing.
   *
   * MUST: carried so that the binding side (executor) can check "is this a resident of my device".
   * {@link id} is numbered independently per GpuContext, so a resident from another context with
   * the same id and size collides on the key of the derived plan — in a hit run (a baked backing)
   * the passed object is never referenced, so **the old value of the previous context is read with
   * neither exception nor warning**. Only the miss path fails, by chance, with a device-mismatch
   * validation error, and the detection vanishes the moment the cache hits.
   * The same guard as the owner check in {@link BatchScope} is placed here too.
   */
  readonly owner: GpuContext;
  /** Adds one reference from a baked bind group (when a backing is built). */
  retainBaked(): void;
  /** Returns one baked reference (when a backing is retired). */
  releaseBaked(): void;
  /**
   * Adds one record that **an in-flight run bound this as an input** (separate from baked
   * references).
   *
   * A baked reference is a static reference while the backing lives, whereas this is a reservation
   * that closes the window of a miss run, "bind the raw buffer into env → await pipeline creation →
   * encode". In this window there is not yet a single baked reference, so reference counts alone
   * would let dispose through.
   */
  retainBound(): void;
  /**
   * Returns one binding reservation of an in-flight run (once the run's encoding and submit are
   * done).
   */
  releaseBound(): void;
  /**
   * Adds one record that **an issued (unsettled) run / enqueue uses this** (separate from baked
   * references and bindings).
   *
   * MUST: it is taken in **the synchronous section of the issue** of `Session.run` /
   * `Session.enqueue`. The execution body comes one microtask later, so taking it in the body
   * (binding reservation {@link retainBound}) would open, for a resident with `bakedRefs === 0`
   * where neither baking nor binding has happened yet, a window where "a `dispose()` in the same
   * tick passes although a run the API accepted exists". That run fails loudly later, but the
   * failure point shifts from the dispose call site to the run's settlement, and the promise in the
   * doc of {@link ResidentTensor.dispose}, "a mistake fails at the dispose call site with its real
   * cause", would not hold for an accepted run. Covers the destination of `copyOutputs` as well as
   * inputs.
   */
  retainUse(): void;
  /**
   * Returns one use reservation (when the run / enqueue settles — always, on both the success and
   * the failure path).
   */
  releaseUse(): void;
};

/**
 * Arrays the host can write into a resident tensor (every element type is 4 bytes — the same
 * convention as ADR 0009).
 */
export type ResidentData =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>;

/**
 * A resident buffer owned by GpuContext (**the 4th lifetime class**).
 *
 * The existing 3 classes (weight arena / slot backing / run arena) are all closed inside a Session,
 * and any hand-over across Sessions always goes through the host (readback → writeBuffer). In a
 * shape that cycles values between graphs, such as a generation loop, each round trip raises 2
 * fences. This container exists only to remove that round trip and **has neither dtype nor shape**
 * (only bytes and a size).
 *
 * MUST: destroy only after confirming "it is not referenced" ({@link ResidentTensor.dispose}).
 * flush-before-destroy (ADR 0004) is met by these two: ① destroying is refused while even one
 * reference from a baked bind group exists ② `enqueue` always submits eagerly at its end, so when
 * it returns there is no **unsubmitted** encode referring to this backing (destroying a buffer
 * referenced by already-submitted commands is safe in WebGPU terms — the implementation defers the
 * real release until completion).
 */
export class ResidentTensor {
  /** The number of bytes allocated (the requested value itself — a multiple of 4). */
  readonly byteLength: number;
  /** A name for diagnostics (the same as the GPUBuffer's label). */
  readonly label: string;
  /** The runtime-internal surface (not a surface users touch). */
  readonly [RUNTIME_INTERNAL]: ResidentInternals;
  readonly #gpu: GpuContext;
  readonly #buffer: GPUBuffer;
  #disposed = false;
  #bakedRefs = 0;
  #boundRefs = 0;
  #useRefs = 0;

  /**
   * MUST: the only entry point for construction is {@link GpuContext.createResident} (the
   * errorScope guard is not bypassed).
   */
  constructor(gpu: GpuContext, id: number, buffer: GPUBuffer, byteLength: number, label: string) {
    this.#gpu = gpu;
    this.#buffer = buffer;
    this.byteLength = byteLength;
    this.label = label;
    this[RUNTIME_INTERNAL] = {
      buffer,
      id,
      owner: gpu,
      retainBaked: () => {
        this.#assertUsable("baking");
        this.#bakedRefs += 1;
      },
      releaseBaked: () => {
        if (this.#bakedRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': too many releases of baked references (the runtime's internal bookkeeping is broken)`,
          );
        }
        this.#bakedRefs -= 1;
      },
      retainBound: () => {
        this.#assertUsable("binding");
        this.#boundRefs += 1;
      },
      releaseBound: () => {
        if (this.#boundRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': too many releases of binding reservations (the runtime's internal bookkeeping is broken)`,
          );
        }
        this.#boundRefs -= 1;
      },
      retainUse: () => {
        this.#assertUsable("use");
        this.#useRefs += 1;
      },
      releaseUse: () => {
        if (this.#useRefs === 0) {
          throw new ResidentTensorError(
            `resident '${this.label}': too many releases of use reservations (the runtime's internal bookkeeping is broken)`,
          );
        }
        this.#useRefs -= 1;
      },
    };
  }

  /** Whether it has been disposed. */
  get disposed(): boolean {
    return this.#disposed;
  }

  /** The number of references from baked bind groups (it cannot be disposed unless 0). */
  get bakedReferences(): number {
    return this.#bakedRefs;
  }

  /** The number of in-flight runs that bound it as an input (it cannot be disposed unless 0). */
  get boundReferences(): number {
    return this.#boundRefs;
  }

  /**
   * The number of issued (unsettled) runs / enqueues using it as an input or as the destination of
   * `copyOutputs` (it cannot be disposed unless 0).
   */
  get useReferences(): number {
    return this.#useRefs;
  }

  /**
   * Writes the whole range from the host (`queue.writeBuffer`). For putting in condition tensors
   * **before** a generation loop starts.
   *
   * MUST: the size matches exactly. Allowing partial writes would leave the remaining bytes with
   * their previous contents, and the loop would run with stale conditions with neither exception
   * nor warning (the same thinking as full-write — ADR 0014).
   * MUST NOT: **write while an issued (unsettled) run / enqueue takes this backing as an input**.
   * The execution body runs one microtask or more later, so a write in the same tick right after
   * issue lands on the queue before that dispatch and makes the dispatch read **the value written
   * later, not the value intended at issue** (with neither exception nor warning). Writing is
   * allowed after the returned Promise settles — for an `enqueue` run without await, after {@link
   * BatchScope.finish}. It is **the same borrowed-side contract** as the host input `Tensor.data`
   * (the MUST NOT of the "input lifetime" section of `Session.run`), and no mechanical guard is put
   * in (the line drawn to avoid paying for a copy of the value every time is the same too).
   * MUST: this `writeBuffer` lands on the queue timeline in issue order, so it **does not overtake
   * dispatches already submitted**. Only unsubmitted encodes are overtaken (ADR 0004
   * buffer-management invariant ④), and `enqueue` always submits at its end before returning, so it
   * does not conflict with **settled** enqueues. This is a claim about the order of the queue
   * timeline and does not protect the window the MUST NOT above speaks of, "issued but the body has
   * not run yet" (that is the caller's contract).
   * MUST: do not write on a lost device ({@link assertDeviceUsable}). `queue.writeBuffer` is a
   * no-op on a destroyed buffer with neither exception nor warning, so without stopping here the
   * generation loop would run with empty conditions.
   */
  write(data: ResidentData): void {
    this.#assertUsable("write");
    assertDeviceUsable(this.#gpu, `resident '${this.label}' write`);
    if (data.byteLength !== this.byteLength) {
      throw new ResidentTensorError(
        `resident '${this.label}': write byte count ${data.byteLength} does not match the allocation ${this.byteLength}`,
      );
    }
    this.#gpu.device.queue.writeBuffer(this.#buffer, 0, data);
  }

  /**
   * Sets the whole range to zero on the GPU (its own command encoder → `clearBuffer` → submit). For
   * resetting a state carried between runs, such as a causal cache at the head of a tile, without a
   * host write: no host array and no `writeBuffer` staging.
   *
   * It is a method of the already-exported `ResidentTensor` type, so the public surface gains no new
   * export (mod.ts stays thin — ADR 0008).
   *
   * MUST: the clear lands on the queue timeline in issue order, so it **does not overtake
   * dispatches already submitted**, and later submits do not overtake it. Only unsubmitted encodes
   * would be overtaken, and a settled run / enqueue leaves none (`enqueue` always submits at its end
   * — ADR 0054 decision 2, which extends ADR 0004 buffer-management invariant ④). It is the same
   * queue-order claim as {@link ResidentTensor.write}.
   * MUST: **fail loudly while a use reservation is held** ({@link ResidentTensor.useReferences} >
   * 0): an issued (unsettled) run / enqueue that uses this backing as an input or as the destination
   * of `copyOutputs`, or a pending {@link BatchScope.finishAndRead} that has it as a member. The
   * execution body runs one microtask or more later, so a clear in the same tick right after issue
   * would land before that run's dispatches and zero its input, or its copy target before the copy;
   * a clear before the batch's readback copy would read back zeros. Clear after the returned Promise
   * settles — for an `enqueue` run without await, after {@link BatchScope.finish}. NOTE: this is an
   * asymmetry with `write`, which leaves the same window as a caller contract (its MUST NOT): a
   * clear carries no host data, so the guard is an O(1) counter check with nothing to copy.
   * MUST: the same guards as `write` — a disposed resident and a lost device fail loudly ({@link
   * assertDeviceUsable}: commands on a destroyed device are silent no-ops). The size needs no
   * runtime check: the clear covers exactly {@link ResidentTensor.byteLength}, which
   * `createResident` keeps a positive multiple of 4 (`clearBuffer`'s alignment), and the backing is
   * always created with `COPY_DST`.
   *
   * Unscoped like `write` (no errorScope pair): with the guards above no validation failure is
   * reachable, and staying synchronous avoids one await per resident. As with `write`, the submit
   * can land while another operation's errorScope is pushed (a run waiting inside its section lock,
   * or an open batch), and a failure would then be attributed to it — the window ADR 0123 names for
   * unscoped operations. Each clear is one small submit of its own, outside the Session scheduler's
   * in-flight window (ADR 0123 bounds only run / enqueue dispatches), so a loop over many residents
   * without a fence is not bounded by it.
   */
  clear(): void {
    this.#assertUsable("clear");
    assertDeviceUsable(this.#gpu, `resident '${this.label}' clear`);
    if (this.#useRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': in use by an issued run / enqueue (an input or a copy destination)` +
          ` or a pending finishAndRead (${this.#useRefs} uses), so it cannot be cleared. ` +
          "Await the settlement of that run / enqueue / batch before clearing it",
      );
    }
    const device = this.#gpu.device;
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(this.#buffer, 0, this.byteLength);
    device.queue.submit([encoder.finish()]);
  }

  /**
   * Reads the whole range back to the host (copy to staging → `mapAsync`). **This is the only
   * fence**, placed to take the latent out once at the end of a generation loop.
   *
   * MUST: call it **after** {@link BatchScope.finish}. Calling it inside the batch adds one more
   * fence and also reads a value from before the section settled (without the section's remaining
   * enqueues).
   *
   * MUST: do not read on a lost device ({@link assertDeviceUsable} — the same guard as `write` /
   * {@link GpuContext.createResident}). In the window right after `destroy()` (the flag is set
   * synchronously but the reaction of `device.lost` runs in the next task), staging would be
   * allocated and the copy submitted on a destroyed device, and since `popErrorScope` resolves with
   * null after a loss, the failure would not be caught.
   */
  async read(): Promise<ArrayBuffer> {
    this.#assertUsable("read");
    const device = this.#gpu.device;
    const where = `resident '${this.label}' readback`;
    assertDeviceUsable(this.#gpu, where);
    // MUST: wrap copy → submit in the errorScope pair as well. Neither a validation failure such as
    // a missing COPY_SRC nor a failed staging allocation becomes an exception, and the readback
    // would quietly return all 0 (the same as #readOutputs).
    // MUST NOT: put an await between the push and the issue of the pop (it completes within a
    // synchronous section, so no lock is needed).
    pushFailureScopes(device);
    let popped = false;
    let staging: GPUBuffer | undefined;
    try {
      staging = device.createBuffer({
        size: this.byteLength,
        usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.#buffer, 0, staging, 0, this.byteLength);
      device.queue.submit([encoder.finish()]);
      const pending = popFailureScopes(device, where);
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;
      // MUST: an implementation may never resolve mapAsync after a loss (measurements in the doc of
      // raceCanaryDeviceLost), so race it — insurance that turns a hang into a failure.
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(staging.mapAsync(MAP_MODE.READ), where);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return copy;
    } finally {
      if (!popped) await discardFailureScopes(device);
      // MUST: always return staging regardless of the outcome (destroy implies unmap).
      staging?.destroy();
    }
  }

  /**
   * Disposes (no-op from the second time on).
   *
   * MUST: **fail loudly** while referenced from a baked bind group. Destroying silently would let
   * that backing's dispatches be submitted with the destroyed buffer bound, failing the whole
   * command buffer, and **even unrelated dispatches never run, silently leaving wrong values
   * behind** (ADR 0004). To remove the reference, dispose that Session or switch the backing with a
   * run / enqueue of another signature.
   * MUST: likewise fail loudly **while an in-flight run binds it as an input**. Baked references
   * appear only when a backing is built (= from the hit run on), so the window of a miss run, "bind
   * → await pipeline creation → encode", cannot be protected by baked references alone (destroying
   * in that window makes the resumed encode grab the destroyed buffer and the run fails with
   * validation). With this reservation, the mistake fails at the dispose call site with its real
   * cause.
   * MUST: likewise fail loudly **while an issued (unsettled) run / enqueue uses it as an input or
   * as the destination of `copyOutputs`**. The binding reservation appears in the execution body (a
   * microtask or more later), so in "the same tick right after the API accepted it" there are 0
   * baked references and 0 bindings — the only thing that closes that window is the use reservation
   * taken in the synchronous section of the issue ({@link ResidentInternals.retainUse}), and that
   * is also the ground for the guarantee above holding for accepted runs / enqueues.
   */
  dispose(): void {
    if (this.#disposed) return;
    if (this.#bakedRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': referenced by baked bind groups (${this.#bakedRefs} references), so it cannot be disposed. ` +
          "Dispose the Session that references it, or switch the backing with a run / enqueue of another signature",
      );
    }
    if (this.#boundRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': bound as an input by an in-progress run (${this.#boundRefs} bindings), so it cannot be disposed. ` +
          "Await the completion of that run before disposing it",
      );
    }
    // Checked after baking and binding (both name a window narrower than this one, so as a
    // diagnostic it is closer to the cause for them to come out first).
    if (this.#useRefs > 0) {
      throw new ResidentTensorError(
        `resident '${this.label}': in use as an input or a copy destination by an issued run / enqueue` +
          ` (${this.#useRefs} uses), so it cannot be disposed. ` +
          "Await the settlement of that run / enqueue before disposing it",
      );
    }
    this.#disposed = true;
    this.#buffer.destroy();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  #assertUsable(where: string): void {
    if (this.#disposed) {
      throw new ResidentTensorError(
        `resident '${this.label}': a disposed resident tensor cannot be used (${where})`,
      );
    }
  }
}

/**
 * The parties a batch gathers at settlement (the implementation is {@link SubmitScheduler}).
 * It is a structural type so that the device layer does not depend on the submit layer.
 */
export type BatchMember = {
  /** Submits the unsubmitted encodes (**without a fence**). */
  submitPending(): void;
  /** Closes the measurement window after the batch's fence has completed. */
  closeMeasurementWindowAfterFence(): void;
  /**
   * Discards, without recording, the measurement window of a batch that never reached its fence.
   */
  discardMeasurementWindow(): void;
};

/**
 * Where a batch failure came from (the second argument of {@link BatchFinalizer.fail}).
 *
 * - `"gpu"`: the body of the section (the errorScope check, the completion fence, the readback)
 *   failed. Besides the errorScope reporting a GPU-side failure, this includes failing before the
 *   check and discarding the scopes unchecked, and device loss — in none of them has it been
 *   confirmed that "the resources made in the section were valid".
 * - `"host"`: the body of the section passed without reporting a failure, and the settlement came
 *   from a host-side failure of an enqueue body ({@link BatchInternals.leave}) or of a finalizer.
 *   The resources made in the section are valid.
 *
 * They are split because the rollback of resources whose allocation was invalid due to a non-fatal
 * OOM (`#rollbackSettled` of `Session`) is needed only for `"gpu"`. Rolling back on a host-side
 * failure too would rebuild every time the same failure recurs, however often.
 */
export type BatchFailureOrigin = "gpu" | "host";

/**
 * After the batch's success is decided, commits the state, and returns the use reservations
 * regardless of the outcome.
 */
type BatchFinalizer = {
  complete(): void;
  fail(cause: unknown, origin: BatchFailureOrigin): void;
  release(): void;
};

/**
 * A copy source read back at section settlement by concatenated copy into one staging ({@link
 * BatchInternals.readAtFinish}).
 */
export type BatchReadSource = {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
};

/**
 * One readback of graph outputs registered with {@link BatchInternals.readAtFinish} (kept until
 * settlement).
 */
type GraphRead = {
  readonly sources: readonly BatchReadSource[];
  readonly resolve: (buffers: readonly ArrayBuffer[]) => void;
  readonly reject: (cause: unknown) => void;
  /**
   * Filled by the settlement's readback (not returned to the caller until the section's outcome is
   * decided).
   */
  result?: readonly ArrayBuffer[];
};

/** The runtime-internal surface of {@link BatchScope}. */
type BatchInternals = {
  /**
   * The wait until the errorScope section actually opens ({@link GpuContext.beginBatch} awaits it).
   */
  readonly entered: Promise<void>;
  /** Registers a party to gather at settlement (registering the same party twice is harmless). */
  join(member: BatchMember): void;
  /**
   * Checks that it can accept an enqueue, and **takes one in-flight lease** (settled / another
   * device fails loudly and takes no lease).
   *
   * MUST: call it in **the synchronous section** of `Session.enqueue` (before queuing on `#chain`).
   * The enqueue body comes one microtask later, so taking the lease in the body would let
   * `finish()` right after queuing un-awaited enqueues settle first, and the section would look
   * like a normal end with not one of the queued enqueues dispatched (a silent no-op).
   */
  enter(owner: GpuContext): void;
  /**
   * Returns one in-flight lease (always, on both the success and the failure path of the enqueue
   * body).
   *
   * Passing `failure` records **only the first one** of the section as a host-side failure, and
   * {@link BatchScope.finish} becomes where it is attributed (passing nothing records nothing, as
   * before). It needs recording because when an un-awaited `enqueue` fails in its body, the section
   * would otherwise settle as "success, with 0 dispatches fewer" — only GPU-side failures land in
   * the errorScope, and a host-side throw shows only on the returned Promise (escaping as an
   * unhandled rejection if nobody holds it).
   * MUST: drop the second and later ones. A row of derived failures dragged along by the first
   * would make the root cause unreadable (the same judgement as the errorScope preferring internal
   * / out-of-memory).
   */
  leave(failure?: { readonly cause: unknown }): void;
  onSettled(finalizer: BatchFinalizer): void;
  /**
   * Registers graph-output slots as parties to read back at section settlement
   * (`Session.enqueueRead`). They are concatenated into the same one staging as the readback of
   * resident tensors ({@link BatchScope.finishAndRead}), and its map is the only fence. The return
   * value resolves only after the section settles without an exception, and rejects on failure (GPU
   * side or host side).
   *
   * MUST: call it from the enqueue body (while holding an in-flight lease). `finish` proceeds to
   * the readback only after every lease has been returned, so parties registered inside a lease are
   * not missed. Registration after the readback has started fails loudly (no Promise that never
   * resolves is returned).
   */
  readAtFinish(sources: readonly BatchReadSource[]): Promise<readonly ArrayBuffer[]>;
  /**
   * Performs only the admission checks that {@link BatchInternals.readAtFinish} makes at
   * registration (rejection after the readback has started, and the limit on the registered total +
   * this request), without registering.
   *
   * MUST: `Session.enqueueRead` calls this before queuing a single dispatch — registration
   * (readAtFinish) happens after the submit, so failing there for the first time would turn a
   * recoverable failure (the readback is merely too large) into a failure after the state write,
   * poisoning the context. The check is implemented once, shared with readAtFinish.
   * The check on the readAtFinish side is not removed (another Session's registration can come
   * between this check and the registration).
   */
  assertReadAtFinish(sources: readonly BatchReadSource[]): void;
};

/**
 * A section that bundles enqueues without fences ({@link GpuContext.beginBatch}).
 *
 * For the duration of the section it holds the device's errorScope section lock and keeps the two
 * scopes `out-of-memory` + `validation` pushed. {@link BatchScope.finish} ① drains the unsubmitted
 * work of every member ② waits for `onSubmittedWorkDone` **only once** ③ pops the scopes and turns
 * a failure into a typed exception.
 *
 * Trade-offs (accepted by design):
 *
 * - **Failures are attributed per batch**. N enqueues ride in one section, so even when validation
 *   / out-of-memory comes out it cannot be narrowed to "which enqueue". When isolation is needed,
 *   run them one by one with ordinary `run`s (run is attributed per run as before).
 * - **Detection of device loss is delayed until finish**. Naturally so, since there is not a single
 *   wait during the section; the loss is turned into an exception by finish's `raceDeviceLost`
 *   (the enqueue side does not hang — it does not wait).
 */
export class BatchScope {
  /** The runtime-internal surface (not a surface users touch). */
  readonly [RUNTIME_INTERNAL]: BatchInternals;
  readonly #gpu: GpuContext;
  readonly #members = new Set<BatchMember>();
  readonly #finalizers: BatchFinalizer[] = [];
  readonly #completion: Promise<Readonly<Record<string, ArrayBuffer>> | undefined>;
  readonly #release: () => void;
  /**
   * Notification that every outstanding in-flight lease has been returned ({@link
   * BatchInternals.enter}).
   */
  readonly #drained = Promise.withResolvers<void>();
  #leases = 0;
  #readback: readonly (readonly [string, ResidentTensor])[] | undefined;
  /** Graph outputs read back at settlement ({@link BatchInternals.readAtFinish}). */
  readonly #graphReads: GraphRead[] = [];
  /** Whether the settlement's readback has started (registrations after it fail loudly). */
  #readbackStarted = false;
  #finished = false;
  /**
   * Whether {@link BatchScope.settle} is in progress (enqueues during it are rejected — protecting
   * the window's attribution).
   */
  #settling = false;
  /**
   * The first host-side failure in the section ({@link BatchInternals.leave}). The second and later
   * are dropped.
   */
  #failure: { readonly cause: unknown } | undefined;
  /**
   * The settlement {@link BatchScope.finish} returns (the errorScope result merged with {@link
   * #failure}).
   */
  #settled: Promise<void> | undefined;

  /**
   * MUST: the only entry point for construction is {@link GpuContext.beginBatch} (the guard against
   * combining with measurement lives there).
   */
  constructor(gpu: GpuContext) {
    this.#gpu = gpu;
    const hold = Promise.withResolvers<void>();
    const entered = Promise.withResolvers<void>();
    this.#release = hold.resolve;
    this.#completion = gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
      // MUST: the section is untimed from before the first enqueue until the lock is returned.
      const closeSection = gpu[RUNTIME_INTERNAL].openBatchSection();
      try {
        try {
          // MUST: recheck that the device is alive once the lock is acquired (before the push). The
          // check at the entrance of `beginBatch` comes before waiting for an earlier section to
          // release the lock, so it has not seen a loss during the wait. Opening would keep holding
          // the section lock on a lost device and make later sections wait until the user's finish. A
          // throw makes the catch below reject the beginBatch wait and return the lock without
          // waiting for hold.
          assertDeviceUsable(gpu, "batch start");
          pushFailureScopes(gpu.device);
        } catch (cause) {
          // MUST: always tell the beginBatch wait that opening failed (otherwise it hangs).
          entered.reject(cause);
          throw cause;
        }
        entered.resolve();
        await hold.promise;
        // MUST: settle every in-flight enqueue before setting the fence. `finish()` only sets
        // `#finished` synchronously, so without this, with "finish() right after queuing un-awaited
        // enqueues", ① every enqueue whose body has not run yet rejects and the section **settles as
        // a success** with 0 dispatches ② enqueues that had started submit after the fence and the
        // pop, and finish returns with GPU execution still incomplete (a read right after returns a
        // stale value).
        await this.#drained.promise;
        const device = gpu.device;
        let outputs: Readonly<Record<string, ArrayBuffer>> | undefined;
        let popped = false;
        /**
         * Whether the completion fence (the readback's `mapAsync` or `onSubmittedWorkDone`) was
         * reached.
         */
        let fenced = false;
        const checkFailureScopes = async (): Promise<void> => {
          const pending = popFailureScopes(
            device,
            this.#readbackBytes() === 0 ? "batch encoding" : "batch encoding and readback",
          );
          popped = true;
          const failure = await pending;
          if (failure !== undefined) throw failure;
        };
        try {
          // enqueue always submits at its end, so normally this is a no-op. It drains anyway to close
          // "no unsubmitted encode exists once the batch closes" as the section's responsibility (no
          // extra party carrying the disposal path — ADR 0004).
          for (const member of this.#members) member.submitPending();
          // MUST: an implementation may never resolve onSubmittedWorkDone after a loss (measurements
          // in the doc of raceCanaryDeviceLost), so race it.
          // MUST: accept no registrations from here on (every lease has been returned, so none should
          // come, but if one does, failing loudly beats returning a Promise that never resolves).
          this.#readbackStarted = true;
          if (this.#readbackBytes() > 0) {
            outputs = await this.#readOutputs(
              this.#readback ?? [],
              this.#graphReads,
              checkFailureScopes,
              () => {
                fenced = true;
              },
            );
          } else {
            // A registration with no copy source at all (0 graph outputs) closes with the usual
            // completion fence.
            for (const read of this.#graphReads) read.result = [];
            await gpu[RUNTIME_INTERNAL].raceDeviceLost(
              device.queue.onSubmittedWorkDone(),
              "batch completion",
            );
            fenced = true;
          }
        } catch (cause) {
          if (!popped) await discardFailureScopes(device);
          throw cause;
        } finally {
          // The measurement window closes with the batch's single fence. With N enqueues in the
          // window the estimate comes out coarse (too large), but too large = smaller chunks = the
          // safe side for TDR ("attribution of measurements" in src/gpu/submit.ts).
          // MUST: close only on paths that reached the fence. A window whose readback failed at the
          // errorScope check before `mapAsync` is in the middle of GPU execution; closing it would
          // put a too-small measurement into the estimate and open the chunk limit permanently (the
          // dangerous side). That window is discarded and the existing estimate is kept.
          for (const member of this.#members) {
            if (fenced) member.closeMeasurementWindowAfterFence();
            else member.discardMeasurementWindow();
          }
        }
        if (!popped) {
          await checkFailureScopes();
          // MUST: a loss while waiting for the pop is not caught by the fence race (`raceDeviceLost`
          // unsubscribes as soon as its wait ends). A pop after a loss settles with null
          // (docs/research 2026-08-16-device-lost-wait-settlement), so without checking here a finish
          // that crossed a loss would return a success and be misread as "the section closed
          // cleanly".
          assertDeviceUsable(gpu, "batch completion");
        }
        return outputs;
      } finally {
        closeSection();
      }
    });
    // Do not let the settlement become an unhandled rejection until finish receives it (finish
    // returns the rejection as is).
    void this.#completion.catch(() => undefined);
    this[RUNTIME_INTERNAL] = {
      entered: entered.promise,
      join: (member) => {
        this.#members.add(member);
      },
      onSettled: (finalizer) => {
        if (this.#finished) {
          throw new BatchScopeError("batch finalizer must be registered at admission");
        }
        this.#finalizers.push(finalizer);
      },
      readAtFinish: (sources) => {
        this.#assertGraphReadAcceptable(sources);
        const { promise, resolve, reject } = Promise.withResolvers<readonly ArrayBuffer[]>();
        // Do not let the settlement become an unhandled rejection until the caller receives it
        // (finish returns the rejection as is).
        void promise.catch(() => undefined);
        this.#graphReads.push({ sources, resolve, reject });
        return promise;
      },
      assertReadAtFinish: (sources) => this.#assertGraphReadAcceptable(sources),
      enter: (owner) => {
        if (owner !== this.#gpu) {
          throw new BatchScopeError("Cannot enqueue into a batch opened on another GpuContext");
        }
        if (this.#finished) {
          throw new BatchScopeError("Cannot enqueue into a batch after finish()");
        }
        if (this.#settling) {
          throw new BatchScopeError(
            "Cannot enqueue into a batch in the middle of settle() (a submit mixed in before the window closes " +
              "puts dispatches the fence did not wait for into the observed time, and the estimate comes out too small = the chunks swell into the TDR range)",
          );
        }
        this.#leases += 1;
      },
      leave: (failure) => {
        if (this.#leases === 0) {
          throw new BatchScopeError(
            "Too many returns of batch in-flight leases (internal bookkeeping is broken)",
          );
        }
        // MUST: record before resolving `#drained` (the section body joins the pop after that
        // resolution).
        this.#failure ??= failure;
        this.#leases -= 1;
        if (this.#leases === 0 && this.#finished) this.#drained.resolve();
      },
    };
  }

  /** Whether it has settled (whether {@link BatchScope.finish} has been called even once). */
  get finished(): boolean {
    return this.#finished;
  }

  /**
   * Sets a single fence **in the middle** of the section and backs the estimate of
   * {@link SubmitScheduler} with a measurement of the dispatches issued so far (perf-ledger P-2).
   *
   * The section's measurement window closes only once, at {@link BatchScope.finish}, so throughout
   * the section the measurement is "0" = chunks stay at `initialChunkSize` (invariant 1 of
   * submit.ts — a measurement of 0 is no ground for growth). In a section of tens of thousands of
   * dispatches (irodori's DiT loop) submits run into the thousands, and a host fixed cost of ≈0.5
   * ms each adds up in wall time. Calling this once right after awaiting the first execution lets
   * the rest of the section grow its chunks with a backed budget.
   *
   * It does only finish's ① drain unsubmitted work ② wait once for `onSubmittedWorkDone` ③ close
   * the window (it neither waits for leases to settle nor pops the errorScope — the section goes
   * on).
   *
   * MUST: call it when no enqueue is in flight (right after awaiting an enqueue). If an un-awaited
   * enqueue remains, it fails loudly — if that enqueue's submit slipped in after the fence and
   * before the window closes, dispatches the fence did not wait for would enter the measurement and
   * the estimate would come out too small (too small = chunks swell = the dangerous side, toward
   * TDR territory). New enqueues while waiting are rejected for the same reason.
   * MUST: fails loudly on a finished section. Device loss is turned into an exception, as in
   * finish.
   */
  async settle(): Promise<void> {
    if (this.#finished) throw new BatchScopeError("Cannot settle() a batch after finish()");
    if (this.#settling) throw new BatchScopeError("settle() calls cannot overlap");
    if (this.#leases > 0) {
      throw new BatchScopeError(
        `Cannot settle() a batch with ${this.#leases} in-flight enqueues` +
          " (await the enqueues before calling it)",
      );
    }
    this.#settling = true;
    try {
      for (const member of this.#members) member.submitPending();
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
        this.#gpu.device.queue.onSubmittedWorkDone(),
        "batch intermediate settlement",
      );
      for (const member of this.#members) member.closeMeasurementWindowAfterFence();
    } finally {
      this.#settling = false;
    }
  }

  /**
   * Closes the section. It proceeds in the order **reject new enqueues → wait for every in-flight
   * enqueue to settle → drain unsubmitted work → wait for every enqueue's completion with one fence
   * → pop the errorScopes and turn a failure into a typed exception**.
   * With `finishAndRead`, the errorScopes are checked after the copy, and the map is the only
   * completion fence.
   *
   * MUST: return the same completion from the second call on (returning earlier would let the
   * caller proceed to destroy with the lock and errorScopes still open).
   * MUST: **the first host-side failure** in the section is also attributed here
   * ({@link BatchInternals.leave}) — if the errorScopes caught nothing, that failure is thrown;
   * when both exist, the errorScope side is thrown with the record in its `cause`. It is **a
   * breaking change of behavior** (up to 0.7.0 a host-side failure never came out of `finish()` as
   * `ExecutionError` or the like), but without it "a section whose un-awaited `enqueue` failed in
   * its body settles as a success with one dispatch missing" — and if nobody holds the returned
   * Promise, it merely escapes as an unhandled rejection. The same failure still shows on the
   * enqueue's returned Promise as before (one fact visible on two paths, the same as `run`).
   * MUST: it cannot be closed while {@link BatchScope.settle} is in progress (fails loudly). The
   * section's transitions go one way, open → settling → open → finished; allowing a skip would let
   * the settle side's fence resolve after the errorScope pop, and an un-awaited `settle()` would
   * escape as an unhandled rejection (finish has already returned its settlement, so one fact
   * disappears). The rejection is returned as **a rejected Promise, not a synchronous throw** —
   * finish has always followed the rule of never throwing synchronously, which avoids an exception
   * being replaced during cleanup on the `await using` / `Symbol.asyncDispose` path.
   * NOTE: it waits for in-flight work, so even calling `finish()` without awaiting the Promise
   * returned by `enqueue()`, everything queued always enters the section and completes (the lease
   * mechanism is {@link BatchInternals.enter}).
   */
  finish(): Promise<void> {
    if (this.#settling) {
      // MUST: do not store the settlement (filling `#settled` with a rejection would keep returning
      // the same rejection even to a legitimate finish() after awaiting settle).
      return Promise.reject(
        new BatchScopeError(
          "Cannot finish() a batch in the middle of settle() (await settle() before closing it)",
        ),
      );
    }
    if (this.#settled === undefined) {
      this.#finished = true;
      // If nothing is in flight, settle it here (sections that issued no enqueue, or whose enqueues
      // were all awaited, get no extra wait, as before).
      if (this.#leases === 0) this.#drained.resolve();
      this.#release();
      this.#settled = this.#resolveFinish();
      // Do not let the settlement become an unhandled rejection until the caller receives it
      // (finish returns its contents as is).
      void this.#settled.catch(() => undefined);
    }
    return this.#settled;
  }

  /**
   * Reads back the given resident outputs together and makes that map the batch's completion fence.
   * Avoids the double wait of a read after the existing finish. An empty selection closes with the
   * same fence as finish.
   *
   * The composition is fixed in the synchronous section, and every output is held under a use
   * reservation until settlement. The data is borrowed and is not written from the call until
   * settlement. The returned ArrayBuffers belong to the caller.
   * It can be specified only once, on a batch that is neither finished nor settling. Later finish
   * calls wait for the same settlement.
   * The outputs total at most the device's maxBufferSize; another device's or disposed outputs are
   * not accepted.
   * DECIDED: docs/decisions/0054-resident-loop-and-fence.md#addendum-2026-09-12--batch-end-readback-of-resident-tensors
   */
  finishAndRead(
    outputs: Readonly<Record<string, ResidentTensor>>,
  ): Promise<Readonly<Record<string, ArrayBuffer>>> {
    const retained: ResidentTensor[] = [];
    try {
      this.#assertReadbackAcceptable();
      assertDeviceUsable(this.#gpu, "batch readback");
      const entries = Object.entries(outputs);
      // MUST: check again after the copy. `Object.entries` runs the user's getters synchronously,
      // and inside them a `finishAndRead` / `finish` of the same batch can settle it first. Without
      // rejecting here, the outer call would overwrite `#readback`, and the use reservations the
      // inner call took on residents would never be returned (they could never be disposed again).
      this.#assertReadbackAcceptable();
      let totalBytes = 0;
      for (const [name, resident] of entries) {
        if (
          !(resident instanceof ResidentTensor) ||
          resident[RUNTIME_INTERNAL].owner !== this.#gpu
        ) {
          throw new BatchScopeError(
            `finishAndRead '${name}': requires a ResidentTensor of the same GpuContext`,
          );
        }
        resident[RUNTIME_INTERNAL].retainUse();
        retained.push(resident);
        totalBytes += resident.byteLength;
      }
      this.#assertReadbackBytes(totalBytes + this.#readbackBytes(), "finishAndRead");
      this.#readback = entries;
    } catch (cause) {
      for (const resident of retained) resident[RUNTIME_INTERNAL].releaseUse();
      return Promise.reject(cause);
    }
    return this.finish().then(async () => (await this.#completion) ?? {});
  }

  /**
   * The admission of `finishAndRead` (only once, on a batch that is neither finished nor settling).
   */
  #assertReadbackAcceptable(): void {
    if (this.#finished || this.#settling) {
      throw new BatchScopeError(
        "finishAndRead can be specified only once, on a batch that is neither finished nor settling",
      );
    }
  }

  /**
   * Whether a graph-output readback can be registered (the admission of {@link
   * BatchInternals.readAtFinish}).
   */
  #assertGraphReadAcceptable(sources: readonly BatchReadSource[]): void {
    if (this.#readbackStarted) {
      throw new BatchScopeError(
        "Cannot register graph outputs on a batch whose settlement readback has started",
      );
    }
    const total = sources.reduce((sum, source) => sum + source.size, 0);
    this.#assertReadbackBytes(this.#readbackBytes() + total, "graph output readback");
  }

  /**
   * The total bytes on the settlement's staging (the resident selection + registered graph
   * outputs).
   */
  #readbackBytes(): number {
    const residents = (this.#readback ?? []).reduce(
      (sum, [, resident]) => sum + resident.byteLength,
      0,
    );
    return this.#graphReads.reduce(
      (sum, read) => read.sources.reduce((inner, source) => inner + source.size, sum),
      residents,
    );
  }

  /**
   * Whether it fits in one staging (no silent switch to split staging or several maps — ADR 0054).
   */
  #assertReadbackBytes(total: number, where: string): void {
    if (!Number.isSafeInteger(total) || total > this.#gpu.limits.maxBufferSize) {
      throw new BatchScopeError(
        `${where}: total ${total} bytes exceeds maxBufferSize ${this.#gpu.limits.maxBufferSize}`,
      );
    }
  }

  /**
   * Reads back the resident-tensor selection and the registered graph outputs by concatenated copy
   * into one staging.
   * Returns the map of residents and puts graph outputs in the registrant's `result` (they are
   * returned to the caller only after the section's outcome is decided —
   * {@link BatchScope.#resolveFinish}).
   */
  async #readOutputs(
    residents: readonly (readonly [string, ResidentTensor])[],
    graphReads: readonly GraphRead[],
    checkFailureScopes: () => Promise<void>,
    markFenced: () => void,
  ): Promise<Readonly<Record<string, ArrayBuffer>>> {
    const device = this.#gpu.device;
    const sources: readonly BatchReadSource[] = [
      ...residents.map(([, resident]) => ({
        buffer: resident[RUNTIME_INTERNAL].buffer,
        offset: 0,
        size: resident.byteLength,
      })),
      ...graphReads.flatMap((read) => read.sources),
    ];
    const staging = device.createBuffer({
      label: "batch-readback",
      size: sources.reduce((sum, source) => sum + source.size, 0),
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      let offset = 0;
      for (const source of sources) {
        encoder.copyBufferToBuffer(source.buffer, source.offset, staging, offset, source.size);
        offset += source.size;
      }
      device.queue.submit([encoder.finish()]);
      await checkFailureScopes();
      await this.#gpu[RUNTIME_INTERNAL].raceDeviceLost(
        staging.mapAsync(MAP_MODE.READ),
        "batch readback",
      );
      markFenced();
      const mapped = staging.getMappedRange();
      offset = 0;
      const take = (size: number): ArrayBuffer => {
        const copy = mapped.slice(offset, offset + size);
        offset += size;
        return copy;
      };
      const outputs = Object.fromEntries(
        residents.map(([name, resident]) => [name, take(resident.byteLength)]),
      );
      for (const read of graphReads) read.result = read.sources.map((source) => take(source.size));
      return outputs;
    } finally {
      staging.destroy();
    }
  }

  /**
   * `await using` support (Explicit Resource Management) — an alias of {@link BatchScope.finish}.
   *
   * The section keeps holding the per-device errorScope section lock, so leaving without going
   * through `finish()` holds the lock forever, and from then on {@link Session.run} on that device
   * keeps waiting with neither exception nor diagnostics. This surface "enables a way of writing
   * (`await using`) that makes that slip less likely"; it does not close the undiagnosed hang
   * itself when a plain `try`/`finally` is forgotten.
   */
  [Symbol.asyncDispose](): Promise<void> {
    return this.finish();
  }

  /**
   * Merges the result of the errorScope pop and the recorded host-side failure into one settlement
   * (the contents of {@link BatchScope.finish} — it is different from the public
   * {@link BatchScope.settle}, which settles **midway** through the section, so this name avoids
   * the same name).
   */
  async #resolveFinish(): Promise<void> {
    let failure: { readonly cause: unknown } | undefined;
    // "gpu" only when the section body failed ({@link BatchFailureOrigin}). Failures decided later
    // (a recorded host-side failure, a finalizer failure) stay "host".
    let origin: BatchFailureOrigin = "host";
    try {
      try {
        await this.#completion;
      } catch (cause) {
        if (this.#failure !== undefined && cause instanceof Error && cause.cause === undefined) {
          cause.cause = this.#failure.cause;
        }
        failure = { cause };
        origin = "gpu";
      }
      failure ??= this.#failure;
      if (failure === undefined) {
        try {
          for (const finalizer of this.#finalizers) finalizer.complete();
        } catch (cause) {
          failure = { cause };
        }
      }
      const notifyFailure = (): void => {
        if (failure === undefined) return;
        for (const finalizer of this.#finalizers) {
          try {
            finalizer.fail(failure.cause, origin);
          } catch {
            /* Keep the original failure and return the remaining use reservations too. */
          }
        }
      };
      if (failure !== undefined) notifyFailure();
      const hadFailure = failure !== undefined;
      for (const finalizer of this.#finalizers) {
        try {
          finalizer.release();
        } catch (cause) {
          failure ??= { cause };
        }
      }
      if (!hadFailure && failure !== undefined) notifyFailure();
      if (failure !== undefined) throw failure.cause;
    } finally {
      this.#finalizers.length = 0;
      for (const [, resident] of this.#readback ?? []) resident[RUNTIME_INTERNAL].releaseUse();
      this.#readback = undefined;
      // Graph-output readbacks are returned only after the section's outcome is decided ("after the
      // merge", the same as finishAndRead — no shape where finish fails while holding success
      // data).
      for (const read of this.#graphReads) {
        if (failure === undefined && read.result !== undefined) read.resolve(read.result);
        else {
          read.reject(
            failure?.cause ??
              new BatchScopeError("The batch settled without reading back the graph outputs"),
          );
        }
      }
      this.#graphReads.length = 0;
    }
  }
}
