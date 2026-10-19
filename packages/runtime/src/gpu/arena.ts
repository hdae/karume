/**
 * The buffer arena of one execution unit (1 run). It **owns** every buffer it allocated and carries
 * flush-before-destroy.
 *
 * The placement of intermediate buffers (node outputs, node-internal temporaries) is not decided
 * here — the plan (the static liveness packing of src/runtime/transient-plan.ts, ADR 0093) decides
 * the offsets into region buffers from live intervals, and the arena only owns those **regions**
 * ({@link RunArena.allocRegion}) for the lifetime of the run. The old "size-class pools + release
 * at the last consumer" (ADR 0004) was retired by ADR 0093 decision 8.
 *
 * Why handing bytes out again is safe (resting on the WebGPU specification):
 * - The usage scope of a compute pass is per dispatch; on the same queue a preceding dispatch's
 *   writes are visible to later dispatches, and execution order is submit order. The plan hands
 *   later intervals only bytes whose live interval ended "after the reading dispatch was encoded",
 *   so a write never overtakes a read.
 * - MUST: buffers written by `queue.writeBuffer` (uniforms / inputs / weights) do not go into a
 *   region. writeBuffer overtakes earlier encodes that are not yet submitted in queue order. Only
 *   output storage and temporaries written by dispatches go into regions.
 * - MUST: bytes handed out again are not zero-initialized (zeroing is guaranteed only for a fresh
 *   createBuffer). The invariant that keeps us from depending on it is **full-write** (ADR 0014):
 *   **every node writes every byte of its bound range**. Even when one node issues several
 *   dispatches (cat), the caller ensures that the whole range is covered per node.
 *   MUST: do **not add** a path that "hands a fresh buffer around the region for ops that write
 *   only part of their output" (rejected in ADR 0014). It would make the invariant depend on
 *   knowledge of kernel implementations and force the arena to know op kinds. An op that needs zero
 *   fill (pad) is implemented as a kernel that **writes** 0 outside the range too.
 *   This is pinned by fault injection (tests/gpu_full_write_test.ts — writes poison values into the
 *   bytes handed out again, runs cat / pad, and checks on a real GPU that not one poison word
 *   remains in the output).
 */

import { BUFFER_USAGE } from "./webgpu-constants.ts";

/** Misuse of the arena (allocation after destroy, an invalid size, a missing usage). */
export class ArenaError extends Error {
  override readonly name = "ArenaError";
}

/**
 * The usage of output storage. Dispatches write it, and it carries COPY_SRC for readback when
 * needed.
 * MUST: the transient region buffers (both run-lifetime ones and Session-resident backings) and
 * {@link ResidentTensor} (src/gpu/context.ts) are also created with **this constant**. A separate
 * constant would give buffers of the same role two different usages.
 */
export const STORAGE_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;

/**
 * The size class is "the real byte count aligned to 4 bytes", nothing more.
 *
 * MUST: never make the bound range larger than requested. When a runtime-sized array is bound,
 * `arrayLength()` is determined by the byte count of the bound range, so binding a rounded-up size
 * silently changes the element count and wrong values come out without an exception (the region
 * buffer itself may be larger — the bound `size` is cut to this value. ADR 0093 decision 2).
 * MUST: the plan (`transient-plan.ts`) and the estimate (`estimate.ts`) both use **this function**.
 * If the definition of the size class split in two, slot sizes would drift from the bindings at run
 * time, giving silent wrong values for the same reason.
 * MUST: the domain is **safe integers** (`Number.isInteger` is not enough). For a request beyond
 * 2^53, `Math.ceil(bytes / 4) * 4` returns a different value that has dropped the remainder, and a
 * buffer smaller than requested would be handed out looking "rounded up" (the MUST above breaks
 * without an exception).
 */
export const toSizeClass = (bytes: number): number => {
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new ArenaError(`Allocation size must be a safe integer >= 0: ${bytes}`);
  }
  return Math.max(4, Math.ceil(bytes / 4) * 4);
};

/**
 * GPU buffer figures of one run. `allocCount` / `allocatedBytes` are the real numbers the arena
 * passed to createBuffer (regions + host writes + readback staging). The other 3 fields are copied
 * by the executor from the plan (ADR 0093):
 * `reuseCount` = number of slots that share a byte range with an earlier slot
 * (`TransientPlan.sharedSlots`), `peakTransientBytes` = the maximum of intermediate bytes live at
 * once (`TransientPlan.peakLiveBytes`), `transientBytes` = intermediates still live at the end of
 * the run = total bytes of the pinned graph outputs.
 * In a run on a slot backing (intermediates do not go through the arena) all 3 fields are 0 (the
 * NOTE in session-types).
 */
export type ArenaStats = {
  readonly allocCount: number;
  readonly reuseCount: number;
  /** Total bytes actually passed to createBuffer. */
  readonly allocatedBytes: number;
  readonly transientBytes: number;
  readonly peakTransientBytes: number;
};

/** The 3 fields the executor copies from the plan (the rest of {@link ArenaStats}). */
export type TransientStats = Pick<
  ArenaStats,
  "reuseCount" | "transientBytes" | "peakTransientBytes"
>;

/**
 * The means of **draining** encodes that are not yet submitted (`SubmitScheduler.flush` /
 * `submitPending`).
 *
 * MUST: when it returns, not one unsubmitted encode remains. **Whether it waits for GPU completion
 * is up to the injecting side** — all {@link RunArena.destroy} requires is that "no encode that
 * refers to a destroyed buffer remains to be submitted"; references from already-submitted commands
 * are safe in WebGPU terms (the implementation defers the real release until completion). The
 * default is the waiting side (`flush`); only a path that consolidated its wait onto another fence
 * passes the non-waiting side.
 */
export type ArenaFlush = () => Promise<void>;

export class RunArena {
  readonly #device: GPUDevice;
  readonly #flush: ArenaFlush;
  readonly #owned: GPUBuffer[] = [];
  #allocCount = 0;
  #allocatedBytes = 0;
  #transients: TransientStats = { reuseCount: 0, transientBytes: 0, peakTransientBytes: 0 };
  #destroying: Promise<void> | undefined;

  constructor(device: GPUDevice, flush: ArenaFlush) {
    this.#device = device;
    this.#flush = flush;
  }

  /**
   * Allocates a region buffer decided by the plan (the backing of intermediate buffers). Cutting it
   * by offset / size and binding is done by the plan side (src/runtime/recipe.ts); the arena only
   * owns and destroys.
   * MUST: allocate before the encode of any dispatch of the run (the ADR 0004 invariant — the plan
   * allocates every region at the start of the run).
   */
  allocRegion(bytes: number): GPUBuffer {
    this.#assertUsable();
    const buffer = this.#device.createBuffer({ size: toSizeClass(bytes), usage: STORAGE_USAGE });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * Allocates a buffer the host writes with `queue.writeBuffer` (uniforms / inputs / weights).
   * MUST: buffers on this path do not go into a region (the writeBuffer overtaking reason in the
   * header).
   */
  allocHostWritten(bytes: number, usage: number): GPUBuffer {
    this.#assertUsable();
    if ((usage & BUFFER_USAGE.COPY_DST) === 0) {
      throw new ArenaError("A buffer written with writeBuffer requires COPY_DST");
    }
    const buffer = this.#device.createBuffer({ size: toSizeClass(bytes), usage });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * Allocates staging (`COPY_DST | MAP_READ`) the host reads with `mapAsync`.
   * MUST: buffers on this path are outside the regions too — MAP_READ cannot be combined with
   * STORAGE.
   * **Not reusing** them (single use per readback) is a design choice, not something the
   * specification forbids (reuse after `unmap()` works — confirmed on a real GPU on 2026-08-29).
   * Single use makes the arena's bulk destroy the only cleanup point, so even if a `mapAsync` is
   * left pending on a failure path, no contamination carries over to the next run. The arena still
   * owns them to keep "allocation and destruction in one place" (ADR 0004) — destruction rides on
   * flush-before-destroy automatically.
   * NOTE: adopting a shared staging (perf-ledger H-9) requires a recovery discipline that destroys
   * and recreates a pending staging on the failure path (without it every later readback fails
   * permanently with OperationError).
   */
  allocHostRead(bytes: number): GPUBuffer {
    this.#assertUsable();
    const buffer = this.#device.createBuffer({
      size: toSizeClass(bytes),
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    });
    this.#owned.push(buffer);
    this.#allocCount += 1;
    this.#allocatedBytes += buffer.size;
    return buffer;
  }

  /**
   * Stores the 3 diagnostic fields copied from the plan (doc of {@link ArenaStats} — once, after
   * the run's intermediates went through the regions).
   */
  recordTransients(stats: TransientStats): void {
    this.#transients = stats;
  }

  get stats(): ArenaStats {
    return {
      allocCount: this.#allocCount,
      allocatedBytes: this.#allocatedBytes,
      ...this.#transients,
    };
  }

  /**
   * Destroys every owned buffer.
   *
   * MUST: clear unsubmitted encodes before destroying (discard-or-flush before destroy). This
   * method waits only for the injected {@link ArenaFlush} to settle, so **the discarding side
   * (`SubmitScheduler.discard` on a failure path) is the caller's responsibility**. Submitting an
   * encode that refers to a destroyed buffer fails the whole command buffer, and even unrelated
   * dispatches riding on the same scheduler never run, silently leaving wrong values behind.
   *
   * NOTE: destroying here ends ownership. The VRAM of a destroyed buffer returns to the budget at
   * the next device poll (`onSubmittedWorkDone` and the like) (measured on the B570 —
   * docs/research/2026-09-27-h35-oom-device-lost.md §5). Returning everything the device holds is
   * `device.destroy()`.
   */
  destroy(): Promise<void> {
    // MUST: a second destroy waits for the same completion. Returning before the earlier flush
    // completes would let the caller move on to the next allocation believing "destroyed".
    this.#destroying ??= this.#destroyOnce();
    return this.#destroying;
  }

  async #destroyOnce(): Promise<void> {
    // MUST: destroy the buffers and clear the state even if the flush fails (mainly device loss).
    // Skipping this would leave `#destroying` stuck as rejected with not one buffer destroyed, and
    // no retry possible. The flush failure itself is not swallowed; it propagates after cleanup.
    try {
      await this.#flush();
    } finally {
      for (const buffer of this.#owned) {
        buffer.destroy();
      }
      this.#owned.length = 0;
    }
  }

  #assertUsable(): void {
    // Allocation after destroy creates "a buffer that is never destroyed" (it escapes the owned
    // list), so it is refused.
    if (this.#destroying !== undefined) {
      throw new ArenaError("A destroyed RunArena cannot be used (create a new one for each run)");
    }
  }
}
