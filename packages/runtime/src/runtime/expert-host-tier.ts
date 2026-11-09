/**
 * The host tier of the exact expert cache (ADR 0146 decision 5 — `SessionOptions.expertCache.hostBytes`):
 * expert bytes held in RAM in front of the container, so that a load reads an expert the tier holds
 * from memory instead of the container's range reader (on the hub's cached source, a `slice` of a
 * CacheStorage Blob — about 3.8 ms per 26B expert in Chrome on the RTX 5070 Ti, where the staging
 * ring uploads it in about 0.5).
 *
 * ## Form
 *
 * One `Uint8Array<ArrayBuffer>` per (layer, expert): the expert's **entry**, its ranges of every
 * pooled weight of the layer contiguous in the layer's layout (`ExpertWeightPlan.entryCodesAt` /
 * `entryStepsAt` in expert-plan.ts) — the form a staging-ring slot carries, so a held expert is one
 * copy into a slot. Plain `ArrayBuffer`s were chosen over GPU upload buffers held mapped by the
 * browser measurement of 2026-11-09 (about twice the RAM in Chrome and a `mapAsync` per refill for
 * the other form). The load sees only {@link ExpertHostStore} — a narrow interface, so that another
 * form (upload buffers, or a tier over CacheStorage in front of a smaller RAM tier) can replace this
 * one without touching the load.
 *
 * ## Why every held byte is a verified byte
 *
 * Construction {@link ExpertHostTier.fill}s the tier from the block bytes it reads and verifies anyway
 * (each expert's ranges are copied out — no block buffer is retained), and a load
 * {@link ExpertHostTier.put}s only entries it assembled from container reads that passed the same
 * checks as an upload (the digest of construction on an unverified source). Nothing else writes an
 * entry, and an entry is never changed once held.
 *
 * ## Recency
 *
 * One least-recently-used order over every held expert of every layer, by bytes: construction fills
 * in its read order (the first filled is the least recent), a `get` makes an expert the most recent,
 * a `put` adds the most recent and evicts the least recent until the held bytes fit the budget.
 *
 * MUST: no import that touches the GPU (the tier is host memory only).
 */

import { ExecutionError } from "./plan.ts";

/** What a load reads a missing expert from before the container (the narrow face of a host tier). */
export type ExpertHostStore = {
  /**
   * The entry of `expert` of `layer`, which becomes the most recently used — undefined when it is not
   * held. MUST NOT: the caller changes the returned bytes (the store keeps them).
   */
  get(layer: number, expert: number): Uint8Array<ArrayBuffer> | undefined;
  /**
   * Holds the entry of an expert the load read from the container (the store keeps `entry` — the
   * caller MUST NOT change it afterwards) as the most recently used, evicting the least recently used
   * over the budget. An entry is never put while held (a load reads only what `get` did not return).
   */
  put(layer: number, expert: number, entry: Uint8Array<ArrayBuffer>): void;
  /** Drops every entry (the Session's dispose — the memory goes back with it, not with GC of it). */
  clear(): void;
  /** Experts held now. */
  readonly heldExperts: number;
  /** Bytes held now (at most the budget). */
  readonly heldBytes: number;
};

/** The `ArrayBuffer` form of {@link ExpertHostStore}, with the construction-time fill. */
export class ExpertHostTier implements ExpertHostStore {
  readonly #budget: number;
  readonly #experts: number;
  readonly #entryBytes: readonly number[];
  /** Key `layer · E + expert` → entry, least recently used first (Map insertion order). */
  readonly #held = new Map<number, Uint8Array<ArrayBuffer>>();
  #heldBytes = 0;
  /**
   * During construction, the bytes filled into each held entry so far (checked by {@link seal});
   * undefined once sealed.
   */
  #filled: Map<number, number> | undefined = new Map();

  /**
   * @param budget bytes of entries the tier may hold (a positive safe integer)
   * @param experts experts per layer (E)
   * @param entryBytes bytes of one entry, per layer
   */
  constructor(budget: number, experts: number, entryBytes: readonly number[]) {
    if (!Number.isSafeInteger(budget) || budget < 1) {
      throw new ExecutionError(`expert host tier: budget ${budget} is not a positive safe integer`);
    }
    this.#budget = budget;
    this.#experts = experts;
    this.#entryBytes = entryBytes;
  }

  get heldExperts(): number {
    return this.#held.size;
  }

  get heldBytes(): number {
    return this.#heldBytes;
  }

  /**
   * Construction only: copies `bytes` (one range of a verified block) into `[at, at + length)` of
   * the entry of `expert` of `layer`. An expert not held yet is held from its first range on when
   * its whole entry fits what is left of the budget, and never otherwise (the held bytes only grow
   * while filling, so an expert refused at its first range is refused at every later one).
   */
  fill(layer: number, expert: number, at: number, bytes: Uint8Array<ArrayBuffer>): void {
    const filled = this.#filled;
    if (filled === undefined) {
      throw new ExecutionError("expert host tier: filled after construction (broken bookkeeping)");
    }
    const size = this.#entrySize(layer);
    if (at < 0 || at + bytes.byteLength > size) {
      throw new ExecutionError(
        `expert host tier: a range [${at}, ${at + bytes.byteLength}) of expert ${expert} of layer ${layer} lies outside its ${size}-byte entry (broken bookkeeping)`,
      );
    }
    const key = this.#key(layer, expert);
    let entry = this.#held.get(key);
    if (entry === undefined) {
      if (this.#heldBytes + size > this.#budget) return;
      entry = new Uint8Array(size);
      this.#held.set(key, entry);
      this.#heldBytes += size;
    }
    entry.set(bytes, at);
    filled.set(key, (filled.get(key) ?? 0) + bytes.byteLength);
  }

  /**
   * Ends construction: every held entry was filled exactly once over (a gap would hold zeros that no
   * block carried — a broken supply walk fails here, not as a wrong expert later).
   */
  seal(): void {
    const filled = this.#filled;
    if (filled === undefined) return;
    for (const [key, entry] of this.#held) {
      if (filled.get(key) !== entry.byteLength) {
        throw new ExecutionError(
          `expert host tier: expert ${key % this.#experts} of layer ${Math.floor(key / this.#experts)} was filled with ${filled.get(key) ?? 0} of its ${entry.byteLength} bytes (broken bookkeeping)`,
        );
      }
    }
    this.#filled = undefined;
  }

  get(layer: number, expert: number): Uint8Array<ArrayBuffer> | undefined {
    this.#assertSealed();
    const key = this.#key(layer, expert);
    const entry = this.#held.get(key);
    if (entry === undefined) return undefined;
    this.#held.delete(key);
    this.#held.set(key, entry);
    return entry;
  }

  put(layer: number, expert: number, entry: Uint8Array<ArrayBuffer>): void {
    this.#assertSealed();
    const key = this.#key(layer, expert);
    const size = this.#entrySize(layer);
    if (entry.byteLength !== size || this.#held.has(key)) {
      throw new ExecutionError(
        `expert host tier: expert ${expert} of layer ${layer} put ${
          this.#held.has(key) ? "while held" : `as ${entry.byteLength} bytes, not ${size}`
        } (broken bookkeeping)`,
      );
    }
    if (size > this.#budget) return;
    this.#held.set(key, entry);
    this.#heldBytes += size;
    // The entry just put is the most recent and fits the budget alone, so it is never its own victim.
    for (const [victim, bytes] of this.#held) {
      if (this.#heldBytes <= this.#budget) break;
      this.#held.delete(victim);
      this.#heldBytes -= bytes.byteLength;
    }
  }

  clear(): void {
    this.#held.clear();
    this.#heldBytes = 0;
    this.#filled = undefined;
  }

  #assertSealed(): void {
    if (this.#filled !== undefined) {
      throw new ExecutionError(
        "expert host tier: used before construction sealed it (broken bookkeeping)",
      );
    }
  }

  #entrySize(layer: number): number {
    const size = this.#entryBytes[layer];
    if (size === undefined) {
      throw new ExecutionError(`expert host tier: no layer ${layer} (broken bookkeeping)`);
    }
    return size;
  }

  #key(layer: number, expert: number): number {
    if (!Number.isSafeInteger(expert) || expert < 0 || expert >= this.#experts) {
      throw new ExecutionError(
        `expert host tier: expert ${expert} is not in [0, ${this.#experts}) (broken bookkeeping)`,
      );
    }
    return layer * this.#experts + expert;
  }
}
