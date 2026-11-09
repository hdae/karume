/**
 * The Session side of the exact expert cache (ADR 0146): the pools and slot tables construction
 * allocates ({@link ExpertPool}), the host mirror of the tables, the per-layer recency, and the
 * {@link ExpertCache} a Session built with `SessionOptions.expertCache` hands out.
 *
 * ## Why the mirror equals the GPU tables
 *
 * The tables are written in one place only, {@link SessionExpertCache} `load`, in the same
 * synchronous section that updates the mirror, and every load runs on the Session's serialization
 * chain with its runs. A run's miss count is read from the mirror inside the run's own serialized
 * body, so no load can come between the GPU reading the tables and the host reading the mirror.
 * `touch` changes host recency only (no GPU state); a load that is already running has chosen its
 * victims, so a concurrent `touch` takes effect from the next load on.
 *
 * MUST: no import of executor.ts (dependencies run one way: executor.ts / session-build.ts → here).
 */

import type { BlockRangeReader } from "../format/container/bind.ts";
import { ContainerFormatError } from "../format/container/header.ts";
import { assertDeviceUsable, type GpuContext } from "../gpu/device.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import {
  assertRouting,
  type ExpertByteRange,
  type ExpertWeightPlan,
  type ExpertWeightRanges,
  planExpertLoad,
  routedExperts,
} from "./expert-plan.ts";
import { ExecutionError, type SymbolBindings } from "./plan.ts";
import type {
  ExpertCache,
  ExpertCacheStats,
  ExpertLoad,
  ExpertRouting,
  ExpertRun,
  GenerationRun,
  RunInputs,
  RunOutputs,
  Tensor,
} from "./session-types.ts";

/** One pooled weight on the GPU: `slots` experts of codes and, for i4 / i4f16, of steps. */
export type ExpertPoolWeight = ExpertWeightPlan & {
  readonly codes: GPUBuffer;
  readonly steps: GPUBuffer | undefined;
  readonly ranges: ExpertWeightRanges;
};

/** One routed layer on the GPU: its slot table (i32 per expert, −1 = not resident) and weights. */
export type ExpertPoolLayer = {
  readonly index: string;
  readonly table: GPUBuffer;
  readonly weights: readonly ExpertPoolWeight[];
};

/**
 * What Session construction allocates for the cache (owned by the weights arena, so `dispose`
 * releases it with the weights).
 */
export type ExpertPool = {
  readonly slots: number;
  readonly experts: number;
  readonly picks: number;
  readonly layers: readonly ExpertPoolLayer[];
  /** Index value → slot table (what an `expert_linear` recipe binds). */
  readonly tables: ReadonlyMap<string, GPUBuffer>;
  /** The graph's declared outputs (the Session's graph also reads every index back). */
  readonly declaredOutputs: readonly string[];
  readonly reader: BlockRangeReader;
  /**
   * The SHA-256 of every expert range, taken at construction from the verified block bytes —
   * present when the reader does not verify (`reader.verified` false), keyed by {@link rangeKey}.
   */
  readonly digests: ReadonlyMap<string, Uint8Array<ArrayBuffer>> | undefined;
};

/** The key of one range in the digest table (a block holds one range per offset). */
export const rangeKey = (range: ExpertByteRange): string => `${range.block}@${range.offset}`;

const sha256 = async (bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> =>
  new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));

/**
 * Records the SHA-256 of the expert ranges one supply item carries (construction, unverified source
 * only): the codes of the experts in the item's rows and, with the item that carries the scale, the
 * steps of every expert. The item's bytes are what `readBlock` verified, so a later range read that
 * matches the digest reads exactly the verified bytes.
 *
 * @param rows the item's rows of the leading (expert) dimension (`[0, E)` when it arrives whole)
 * @param steps the whole scale (joined when split), with the item that carries it
 */
export const digestExpertItem = async (
  digests: Map<string, Uint8Array<ArrayBuffer>>,
  weight: ExpertPoolWeight,
  rows: { readonly first: number; readonly count: number },
  payload: Uint8Array<ArrayBuffer>,
  steps: Uint8Array<ArrayBuffer> | undefined,
): Promise<void> => {
  const work: Promise<void>[] = [];
  const take = (range: ExpertByteRange, bytes: Uint8Array<ArrayBuffer>): void => {
    work.push(
      sha256(bytes).then((digest) => {
        digests.set(rangeKey(range), digest);
      }),
    );
  };
  for (let expert = rows.first; expert < rows.first + rows.count; expert += 1) {
    const range = weight.ranges.codes[expert];
    const at = (expert - rows.first) * weight.codesBytes;
    // The ranges and the supply walk the same rows (both from the block list) — a drift would
    // digest one expert's bytes under another's key.
    if (range === undefined || range.offset !== at || at + range.length > payload.byteLength) {
      throw new ExecutionError(
        `expert cache: the codes of expert ${expert} of '${weight.name}' do not sit where the item's rows put them (broken bookkeeping)`,
      );
    }
    take(range, payload.subarray(at, at + range.length));
  }
  if (steps !== undefined && weight.ranges.steps !== undefined) {
    weight.ranges.steps.forEach((range, expert) => {
      const at = expert * weight.stepsBytes;
      if (at + range.length > steps.byteLength) {
        throw new ExecutionError(
          `expert cache: the steps of '${weight.name}' are shorter than its experts (broken bookkeeping)`,
        );
      }
      take(range, steps.subarray(at, at + range.length));
    });
  }
  await Promise.all(work);
};

/** What the Session lends its cache (no other access to the Session's state). */
export type ExpertCacheHost = {
  readonly gpu: GpuContext;
  /**
   * `Session.run`'s body: the same acceptance, lock, validation and readback. `admit` runs at the
   * head of the serialized body (before any GPU work) and `settle` right after the readback, both
   * inside the body — no load can come between them.
   */
  readonly run: <T>(
    inputs: RunInputs,
    bindings: SymbolBindings,
    generation: GenerationRun | undefined,
    hooks: { readonly admit: () => void; readonly settle: (outputs: RunOutputs) => T },
  ) => Promise<T>;
  /** The Session's serialization chain (the run lock). */
  readonly serialize: <T>(body: () => Promise<T>) => Promise<T>;
  readonly disposed: () => boolean;
};

type MutableStats = { -readonly [K in keyof ExpertCacheStats]: number };

/** The {@link ExpertCache} of one Session (ADR 0146 decisions 2 to 6). */
export class SessionExpertCache implements ExpertCache {
  readonly slots: number;
  readonly experts: number;
  readonly layers: number;
  readonly picks: number;
  readonly #pool: ExpertPool;
  readonly #host: ExpertCacheHost;
  /** Per layer, expert → slot (−1 = not resident): the mirror of the layer's GPU table. */
  readonly #tables: readonly Int32Array<ArrayBuffer>[];
  /** Per layer, the resident experts, least recently used first (Map insertion order). */
  readonly #recency: readonly Map<number, true>[];
  readonly #stats: MutableStats = {
    runs: 0,
    missedRuns: 0,
    loads: 0,
    loadedExperts: 0,
    evictedExperts: 0,
    bytesRead: 0,
    loadMs: 0,
  };
  /** Why the cache is unusable (a load failed after its GPU writes were issued). */
  #poisoned: string | undefined;

  constructor(pool: ExpertPool, host: ExpertCacheHost) {
    this.#pool = pool;
    this.#host = host;
    this.slots = pool.slots;
    this.experts = pool.experts;
    this.layers = pool.layers.length;
    this.picks = pool.picks;
    // The GPU tables start as −1 (construction writes them so).
    this.#tables = pool.layers.map(() => new Int32Array(pool.experts).fill(-1));
    this.#recency = pool.layers.map(() => new Map<number, true>());
  }

  run(
    inputs: RunInputs,
    bindings: SymbolBindings = {},
    generation?: GenerationRun,
  ): Promise<ExpertRun> {
    let issued: GenerationRun | undefined;
    try {
      this.#assertUsable("run");
      // Read once: the checks and the run see the same values (the Session copies this copy again).
      issued =
        generation === undefined
          ? undefined
          : {
              context: generation.context,
              queryLength: generation.queryLength,
              commit: generation.commit,
              chain: generation.chain,
              bidirectionalRows: generation.bidirectionalRows,
            };
      // MUST: a run with a miss is not exact, so its rows must stay uncommitted until a re-run.
      if (issued !== undefined && issued.commit !== "deferred") {
        throw new ExecutionError(
          "expertCache.run: a generation run needs commit 'deferred' (a run that missed is run again after commit(0))",
        );
      }
      if (issued?.chain === true) {
        throw new ExecutionError(
          "expertCache.run: a generation run cannot be chained (a chained run's rows cannot be run again alone)",
        );
      }
    } catch (cause) {
      return Promise.reject(cause);
    }
    return this.#host.run(inputs, bindings, issued, {
      admit: () => this.#assertUsable("run"),
      settle: (outputs) => this.#settle(outputs, issued),
    });
  }

  load(routing: ExpertRouting): Promise<ExpertLoad> {
    let copy: ExpertRouting;
    try {
      this.#assertUsable("load");
      // Copied at issue, then checked: the load runs later, behind the runs before it, and every
      // field is read once.
      const ids = routing.ids;
      copy = {
        layers: routing.layers,
        rows: routing.rows,
        picks: routing.picks,
        ids: ids instanceof Int32Array ? ids.slice() : ids,
      };
      assertRouting(copy, this, "expertCache.load");
    } catch (cause) {
      return Promise.reject(cause);
    }
    return this.#host.serialize(() => this.#load(copy));
  }

  touch(routing: ExpertRouting, rows: number): void {
    this.#assertUsable("touch");
    // Every field is read once (the checks and the walk see the same values).
    const read: ExpertRouting = {
      layers: routing.layers,
      rows: routing.rows,
      picks: routing.picks,
      ids: routing.ids,
    };
    assertRouting(read, this, "expertCache.touch");
    if (!Number.isSafeInteger(rows) || rows < 0 || rows > read.rows) {
      throw new ExecutionError(`expertCache.touch: rows ${rows} is not in [0, ${read.rows}]`);
    }
    const picks = (layer: number): Int32Array<ArrayBuffer> =>
      read.ids.subarray(layer * read.rows * this.picks, (layer * read.rows + rows) * this.picks);
    // Every touched expert is checked before any recency moves, so a failed touch changes nothing.
    this.#tables.forEach((table, layer) => {
      for (const expert of picks(layer)) {
        if (table[expert] < 0) {
          throw new ExecutionError(
            `expertCache.touch: expert ${expert} of layer ${layer} is not resident (touch the routing of a run with no miss, before the next load)`,
          );
        }
      }
    });
    this.#recency.forEach((recency, layer) => {
      for (const expert of picks(layer)) {
        recency.delete(expert);
        recency.set(expert, true);
      }
    });
  }

  stats(): ExpertCacheStats {
    return { ...this.#stats };
  }

  #assertUsable(what: string): void {
    if (this.#poisoned !== undefined) {
      throw new ExecutionError(`expertCache.${what}: the cache is unusable — ${this.#poisoned}`);
    }
    if (this.#host.disposed()) {
      throw new ExecutionError(`expertCache.${what}: the Session is disposed`);
    }
  }

  /**
   * The routing of the run's real rows and its miss count, read inside the run's serialized body
   * (the mirror equals the tables the run read — module doc).
   */
  #settle(outputs: RunOutputs, generation: GenerationRun | undefined): ExpertRun {
    const indices = this.#pool.layers.map((layer) => {
      const index = outputs[layer.index];
      if (
        index === undefined ||
        index.dtype !== "i32" ||
        index.shape.length !== 2 ||
        index.shape[1] !== this.picks
      ) {
        throw new ExecutionError(
          `expert cache: the routing '${layer.index}' did not come back as [rows, ${this.picks}] i32 (broken bookkeeping)`,
        );
      }
      return { tokens: index.shape[0], ids: index.data };
    });
    const tokens = indices[0].tokens;
    if (indices.some((index) => index.tokens !== tokens)) {
      throw new ExecutionError(
        "expert cache: the routed layers came back with different row counts",
      );
    }
    // A prefill bucket's padded rows follow the real ones and are not routing of the sequence.
    const rows = generation === undefined ? tokens : generation.queryLength;
    if (rows > tokens) {
      throw new ExecutionError(
        `expert cache: ${rows} real rows but the routing has ${tokens} (broken bookkeeping)`,
      );
    }
    const span = rows * this.picks;
    const ids = new Int32Array(this.layers * span);
    indices.forEach((index, layer) => ids.set(index.ids.subarray(0, span), layer * span));
    const routing: ExpertRouting = { layers: this.layers, rows, picks: this.picks, ids };
    assertRouting(routing, this, "expert cache");
    let missing = 0;
    this.#tables.forEach((table, layer) => {
      for (const expert of routedExperts(routing, layer, rows)) {
        if (table[expert] < 0) missing += 1;
      }
    });
    this.#stats.runs += 1;
    if (missing > 0) this.#stats.missedRuns += 1;
    // Only the declared outputs reach the caller (the index readback is the cache's).
    const declared: Record<string, Tensor> = Object.create(null);
    for (const name of this.#pool.declaredOutputs) {
      const tensor = outputs[name];
      if (tensor === undefined) {
        throw new ExecutionError(`expert cache: output '${name}' did not come back`);
      }
      declared[name] = tensor;
    }
    return { outputs: declared, routing, missing };
  }

  /** The body of `load` (on the serialization chain). */
  async #load(routing: ExpertRouting): Promise<ExpertLoad> {
    // A load queued behind a failed one, or a lost device, stops before reading a byte.
    this.#assertUsable("load");
    assertDeviceUsable(this.#host.gpu, "expert load");
    const plans = this.#tables.map((table, layer) =>
      planExpertLoad(
        this.slots,
        table,
        this.#recency[layer].keys(),
        routedExperts(routing, layer, routing.rows),
      ),
    );
    if (plans.every((plan) => plan.load.length === 0)) {
      return { loaded: 0, evicted: 0, bytes: 0, ms: 0 };
    }
    const start = performance.now();
    // Every range of every missing expert, with the pool position it is written to.
    const writes = plans.flatMap((plan, layer) =>
      plan.load.flatMap(({ expert, slot }) =>
        this.#pool.layers[layer].weights.flatMap((weight) => [
          {
            buffer: weight.codes,
            at: slot * weight.codesBytes,
            range: weight.ranges.codes[expert],
          },
          ...(weight.steps === undefined || weight.ranges.steps === undefined
            ? []
            : [
                {
                  buffer: weight.steps,
                  at: slot * weight.stepsBytes,
                  range: weight.ranges.steps[expert],
                },
              ]),
        ]),
      ),
    );
    // A failed read changes nothing (no GPU write has been issued yet).
    const bytes = await Promise.all(writes.map((write) => this.#read(write.range)));
    const device = this.#host.gpu.device;
    // MUST NOT: await between the push and the issue of the pop (a section that completes
    // synchronously needs no section lock — the errorScope invariants of gpu/context.ts).
    pushFailureScopes(device);
    try {
      writes.forEach((write, at) => device.queue.writeBuffer(write.buffer, write.at, bytes[at]));
      // The tables after the pools: the next run reads a slot only once its bytes are queued ahead
      // of it (queue order), and the mirror moves in the same section as its table.
      plans.forEach((plan, layer) => {
        if (plan.load.length === 0) return;
        const table = this.#tables[layer];
        const recency = this.#recency[layer];
        for (const { expert } of plan.evict) {
          table[expert] = -1;
          recency.delete(expert);
        }
        for (const { expert, slot } of plan.load) {
          table[expert] = slot;
          recency.set(expert, true);
        }
        device.queue.writeBuffer(this.#pool.layers[layer].table, 0, table);
      });
    } catch (cause) {
      this.#poison(cause);
      await discardFailureScopes(device);
      throw cause;
    }
    const failure = await popFailureScopes(device, "expert load").catch((cause: unknown) => {
      this.#poison(cause);
      throw cause;
    });
    if (failure !== undefined) {
      this.#poison(failure);
      throw failure;
    }
    const ms = performance.now() - start;
    const loaded = plans.reduce((total, plan) => total + plan.load.length, 0);
    const evicted = plans.reduce((total, plan) => total + plan.evict.length, 0);
    const read = writes.reduce((total, write) => total + write.range.length, 0);
    this.#stats.loads += 1;
    this.#stats.loadedExperts += loaded;
    this.#stats.evictedExperts += evicted;
    this.#stats.bytesRead += read;
    this.#stats.loadMs += ms;
    return { loaded, evicted, bytes: read, ms };
  }

  /** One range from the container, checked against its construction digest on an unverified source. */
  async #read(range: ExpertByteRange): Promise<Uint8Array<ArrayBuffer>> {
    const bytes = await this.#pool.reader.read(range.block, range.offset, range.length);
    if (bytes.byteLength !== range.length) {
      throw new ContainerFormatError(
        `block '${range.block}': read ${bytes.byteLength} bytes of a ${range.length}-byte expert range`,
      );
    }
    const digests = this.#pool.digests;
    if (digests === undefined) return bytes;
    const expected = digests.get(rangeKey(range));
    if (expected === undefined) {
      throw new ExecutionError(
        `expert cache: no digest of block '${range.block}' at ${range.offset} was taken at construction (broken bookkeeping)`,
      );
    }
    const actual = await sha256(bytes);
    if (actual.some((byte, at) => byte !== expected[at])) {
      throw new ContainerFormatError(
        `block '${range.block}': bytes [${range.offset}, ${range.offset + range.length}) differ from the bytes verified at construction (SHA-256 mismatch — the source changed under the Session)`,
      );
    }
    return bytes;
  }

  #poison(cause: unknown): void {
    this.#poisoned ??= `a load failed after its GPU writes were issued (${
      cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)
    }), so the slot tables can no longer be trusted — build a new Session`;
  }
}
