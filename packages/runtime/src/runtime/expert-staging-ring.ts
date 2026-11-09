/**
 * The staging ring of the exact expert cache (ADR 0146 decision 5): the `MAP_WRITE | COPY_SRC`
 * buffers a load writes missing experts into and copies out of with `copyBufferToBuffer`, about 2×
 * the rate of `writeBuffer` in Chrome (the browser measurement of 2026-11-09: 0.52 ms per 26B expert
 * in rounds of 32, against 1.12).
 *
 * ## A slot's life
 *
 * Each slot is one buffer that carries one expert's entry per round:
 *
 * | state      | how it is entered                                           | what may touch it          |
 * | ---------- | ----------------------------------------------------------- | -------------------------- |
 * | mapped     | created `mappedAtCreation`, or its `mapAsync` resolved      | {@link stage} (the host)   |
 * | in flight  | {@link stage} wrote and unmapped it; the round's copies are submitted | the GPU's copies  |
 * | mapping    | {@link remap} asked `mapAsync(WRITE)` right after the submit | nothing until it resolves |
 *
 * `mapAsync` resolves only once the GPU has finished every command submitted before it that uses the
 * buffer (the WebGPU mapping rule), so a slot's memory is written by the host only after the copies
 * that read it completed — {@link acquire} waits for exactly that. The order "submit, then
 * `mapAsync`" is required too: a submit that uses a buffer with a pending map fails validation, so a
 * bookkeeping slip shows as a validation failure of the load, not as overwritten bytes.
 *
 * ## The concurrency argument (JS: the ring's state changes only in synchronous sections)
 *
 * - **No lost wakeup**: a slot's wait is a promise kept per slot ({@link #mapped}), replaced only in
 *   {@link stage} / {@link remap}; a promise keeps its settlement, so a mapping that completed before
 *   the next load awaits it is not missed. It never rejects (a failure is its value), so a mapping
 *   that fails with no load left to see it — the Session disposed — is no unhandled rejection.
 * - **No slot reused before its copy finished**: {@link stage} requires every slot it writes to have
 *   settled as mapped, and only {@link acquire} — the wait for those settlements — comes before it in
 *   a round; within a round each slot carries one expert.
 * - **No interleaving**: the ring belongs to one Session's cache, whose loads run one at a time on
 *   the Session's serialization chain (with its runs and its dispose), and stage → copies → submit →
 *   remap is one synchronous section. `dispose` runs after the loads queued before it and destroys
 *   the buffers (an aborted `mapAsync` settles as a failure nobody reads); a load issued after it is
 *   refused before it reaches the ring.
 * - **A broken round fails loudly**: {@link stage} marks its slots as unusable until {@link remap}
 *   replaces the mark, so a round that failed between them (the cache is poisoned then anyway) cannot
 *   leave a slot that a later acquire would take as mapped.
 */

import { type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import { MAP_MODE } from "../gpu/webgpu-constants.ts";
import { ExecutionError } from "./plan.ts";

/** A slot's settlement: undefined = mapped, otherwise why it cannot be written. */
type SlotState = Error | undefined;

export class ExpertStagingRing {
  readonly slots: number;
  readonly slotBytes: number;
  readonly #buffers: readonly GPUBuffer[];
  /** Per slot, the settlement of its latest mapping (module doc). */
  readonly #mapped: Promise<SlotState>[];

  /** @param buffers `slotBytes` each, `MAP_WRITE | COPY_SRC`, created `mappedAtCreation` */
  constructor(buffers: readonly GPUBuffer[], slotBytes: number) {
    if (buffers.length === 0) {
      throw new ExecutionError("expert staging ring: no slot (broken bookkeeping)");
    }
    this.slots = buffers.length;
    this.slotBytes = slotBytes;
    this.#buffers = buffers;
    this.#mapped = buffers.map(() => Promise.resolve(undefined));
  }

  /**
   * Waits until slots `[0, count)` are mapped — after every copy submitted from them completed.
   * MUST: raced with the device's loss (an implementation may never resolve `mapAsync` after one).
   * Fails loudly when a slot's mapping failed: the ring cannot upload any more.
   */
  async acquire(count: number, gpu: GpuContext): Promise<void> {
    const states = await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      Promise.all(this.#mapped.slice(0, this.#count(count))),
      "expert load: staging ring",
    );
    const failed = states.find((state) => state !== undefined);
    if (failed !== undefined) {
      throw new ExecutionError(
        `expert staging ring: a slot cannot be written (${failed.name}: ${failed.message})`,
      );
    }
  }

  /**
   * Writes `entries[i]` into slot `i` and unmaps it (synchronous — the first step of a round's GPU
   * section, after {@link acquire} settled). Returns the slot buffers the copies read.
   */
  stage(entries: readonly Uint8Array<ArrayBuffer>[]): readonly GPUBuffer[] {
    this.#count(entries.length);
    return entries.map((entry, slot) => {
      if (entry.byteLength > this.slotBytes) {
        throw new ExecutionError(
          `expert staging ring: an entry of ${entry.byteLength} bytes does not fit a ${this.slotBytes}-byte slot (broken bookkeeping)`,
        );
      }
      const buffer = this.#buffers[slot];
      this.#mapped[slot] = Promise.resolve(
        new ExecutionError(
          `slot ${slot} was staged but not mapped again (a round failed between them)`,
        ),
      );
      new Uint8Array(buffer.getMappedRange(0, this.slotBytes), 0, entry.byteLength).set(entry);
      buffer.unmap();
      return buffer;
    });
  }

  /**
   * Asks the mapping of slots `[0, count)` again. MUST: called right after the submit of the copies
   * that read them (before it, the submit would fail validation).
   */
  remap(count: number): void {
    for (let slot = 0; slot < this.#count(count); slot += 1) {
      this.#mapped[slot] = this.#buffers[slot].mapAsync(MAP_MODE.WRITE).then(
        () => undefined,
        (cause: unknown) => (cause instanceof Error ? cause : new Error(String(cause))),
      );
    }
  }

  #count(count: number): number {
    if (!Number.isSafeInteger(count) || count < 1 || count > this.slots) {
      throw new ExecutionError(
        `expert staging ring: a round of ${count} experts in ${this.slots} slots (broken bookkeeping)`,
      );
    }
    return count;
  }
}
