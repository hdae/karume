/**
 * The **host-side gather** of Gemma 4's PLE (per-layer embeddings) (ADR 0085).
 *
 * The codec and constants of the index (the container asset `ple_index`) live in `./ple-index.ts`;
 * this file holds only the **owner** — whole blocks stay at or below the budget and kept rows sit on
 * top of it as headroom (a budget of 0 keeps neither); that relation between blocks, rows and the
 * budget cannot be split, so one function owns it.
 *
 * ## Why the host gathers
 *
 * PLE is a pure row lookup whose **only** argument is `input_ids`; for E2B it is 35 i8 tables =
 * 2,240 MiB. The runtime contract gives every initializer GPU residency at Session construction, so
 * as long as PLE stays in the graph it cannot be lazy (ADR 0085 Context). Taking it out of the
 * graph and supplying `per_layer_inputs[1,M,35,256]` as an **ordinary graph input** brings
 * residency from 3.70 to 1.51 GiB. The runtime contract does not change by a single character
 * (ADR 0085 decision 6 — no "pageable initializer" slot is created).
 *
 * The precedents are SBV2's relative position tables (`sbv2/relattn-tables.ts`) and Anima's rope
 * base tables — in both, "how to build a model-specific input" is knowledge of models, not runtime
 * vocabulary (ADR 0008).
 *
 * ## Distribution (assets of the model container — ADR 0109 decision 4)
 *
 * `values` (**token-major** `[tokens, layers, dim]`) and `scales` (`[tokens, layers]` f32 — schema 4:
 * `[tokens, layers, dim / groupSize]`, one per group of a layer's values) are each cut into a list
 * of blocks at or below the block limit and **in whole rows**, and stored in the
 * container as assets with the roles `ple-values` / `ple-scales`. A bf16 index (schema 5) has the
 * `values` table alone — the checkpoint's bf16 values, no scales. A block that needs range reads
 * sits alone in its own part (container-v1 §4.2), so reading one row does not drag in the
 * neighbouring weights. The block boundaries of the two tables are **independent** (one row =
 * 8,960 B of values / 140 B of scales, so the same 32 MiB spans a different number of tokens).
 *
 * ## MUST: dequantization is bit-identical to the GPU-side `embedding`
 *
 * While PLE was in the graph, `embedding` computed `f32(i8) × per-row scale` and the `mul` right
 * after it multiplied by `embed_scale` (= `hidden_size_per_layer_input ** 0.5` = 16.0). This file
 * builds it in **the same order** (`(q × scale) × embedScale`). `embedScale` is a power of two, so
 * the f32 multiplication is exact and matching the order is enough (ADR 0085 decision 4). A bf16 row
 * has no first step: the value is the bf16 bits widened to f32 (`bits << 16` — exact, the host
 * mirror of the GPU `embedding`'s `bitcast<f32>`, runtime `format/bf16.ts`), then `× embedScale`. If it
 * diverged, token-sequence parity would break and the proof of "unchanged behaviour" would be
 * unusable. The check is the `ple.probe.safetensors` comparison in
 * `packages/e2e/tests/e2e_gemma4_product_test.ts` (**exact equality** with the values torch
 * computed on the 35-table path).
 *
 * ## How to read (whole block ↔ rows) and the reader's lifetime
 *
 * The reader of a block the index points at (the runtime's `AssetReader`) is opened with
 * `open(MODEL).asset(name)`. One gather groups ids by block per table and picks the policy from the
 * **unique row count** touching that block:
 *
 * | State | Policy |
 * | --- | --- |
 * | Already resident | hit (moved to the LRU tail) |
 * | Unique rows ≥ {@link FULL_BLOCK_ROWS} (or the block's row count when smaller) and it fits the budget **without eviction** | whole read + LRU |
 * | Otherwise | row reads (`read(row offset, rowBytes)`) |
 *
 * MUST: **do not hold a reader longer than one read**. The `AssetReader` of an unverified source
 * (in-memory bytes / a local directory) verifies the block once and keeps it **while the reader is
 * alive** (the contract in the runtime's `AssetReader` doc), so holding readers lets host RAM grow
 * outside {@link Gemma4PleOptions.maxResidentBytes}. Residency is represented only by "the
 * ArrayBuffer that was read", and the reader is dropped on the spot.
 *
 * Quantized rows that were looked up are reused through an LRU of at most {@link ROW_CACHE_CAPACITY}
 * rows. A row taken from a resident whole block is **copied** into its own buffer first, so a kept
 * row never keeps a block alive. The row cache is headroom **outside** the block budget (blocks ≤
 * budget, plus at most {@link ROW_CACHE_CAPACITY} rows); a budget of 0 keeps no rows either (ADR
 * 0085 addenda 2026-09-12 and 2026-10-27). The same bytes go through the same two-step rounding, so
 * decision 4's bit identity does not depend on the path.
 *
 * ## Gather order (hits first) and duplicate ids
 *
 * One gather **touches the row-cache and resident-block hits first**. Reading the non-resident
 * blocks first would let that read's LRU eviction drop "a hit this gather has not touched yet",
 * forcing a re-read within the same gather. Touching hits first moves them to the LRU tail, so the
 * eviction of a later miss leans towards "blocks this gather is done with". The data taken during
 * the run is held by local variables, so values stay consistent even if eviction happens midway.
 *
 * For a sequence with repeated ids (typically prefill's pad row id 0 — most of 768 rows share one
 * id), **only the first position is dequantized and its f32 bytes are copied to the rest**. A copy,
 * not a recomputation, is bit-identical to the two-step rounding, so the
 * `(q × scale) × embedScale` contract holds as is.
 *
 * ## MUST: cross-check the id spaces (ADR 0085 decision 5)
 *
 * Compare the index's row count, the main embedding's vocab row count and the ids actually looked
 * up. A mismatch here reads **not out of bounds but "a valid row of another token"** (silent
 * corruption without an exception), so this is the only place for the fail-loudly check. The
 * cross-check of the index against the container's assets (blocks exist, lengths) is listed in full
 * by the entry admission (`assertGemma4PleAssets` in `./ple-index.ts`) **before a single weight
 * byte is fetched**.
 */

import type { AssetReader, Tensor } from "@karume/runtime";
import { ModelInputError } from "../errors.ts";
import { readAssetRange } from "../hub/asset-readers.ts";
import {
  BF16_BYTES,
  defaultGemma4PleResidentBytes,
  type Gemma4PleBlock,
  gemma4PleBlockBytes,
  gemma4PleBlockOf,
  type Gemma4PleIndex,
  type Gemma4PleTable,
  gemma4PleTables,
  largestBlockBytes,
  packFactor,
  scalesPerLayer,
} from "./ple-index.ts";

/**
 * Options passed through to one gather.
 *
 * MUST: the contract is **best-effort** — abort is checked only **between steps** (per row read and
 * right before a whole read). The runtime's `AssetReader.read` takes no signal, so there is no way
 * to stop a read once it has started. The generation side checks its own `signal` before issuing a
 * run (`generation/sequence.ts`), so it does not rely on this for the correctness of abort.
 */
export type Gemma4PleReadOptions = {
  /** Abort of this gather (one generation's `signal`, passed down as is). */
  readonly signal?: AbortSignal;
};

export type Gemma4PleOptions = {
  readonly index: Gemma4PleIndex;
  /**
   * Opens the reader of one block the index points at (`open(MODEL).asset` itself — the caller's
   * responsibility).
   *
   * MUST: it may return a **new reader** on every call (in fact that is the default). This
   * implementation does not hold an opened reader longer than one read (the MUST in the module
   * doc), so even with a source that keeps the block on the reader side, host RAM stays within
   * {@link Gemma4PleOptions.maxResidentBytes}.
   */
  readonly openBlock: (asset: string) => AssetReader;
  /**
   * The main embedding's vocab row count (cross-check of the id spaces — ADR 0085 decision 5).
   *
   * MUST: do not make it optional. A combination whose index and main embedding were baked with
   * different vocabularies reads **another token's row** with shape and dtype still matching.
   */
  readonly vocabSize: number;
  /**
   * The **host RAM limit (bytes)** for resident whole blocks (LRU — ADR 0085 decision 3).
   *
   * The row cache sits on top of it: at most {@link ROW_CACHE_CAPACITY} quantized rows (E2B:
   * 2,329,600 B regular / 1,182,720 B QAT), kept only while the budget is above 0 (ADR 0085
   * addendum 2026-10-27).
   *
   * Defaults to {@link defaultGemma4PleResidentBytes} (= two of the largest blocks). `0` means
   * "keep nothing resident" = the gather drops each block as soon as it is done with it, and is a
   * valid setting (in exchange for re-reading every time, this index's host RAM stays at a peak of
   * one block).
   *
   * MUST: take bytes, not a count. The block width changes with the asset generation (the writer's
   * block limit), so "two blocks" would mean different RAM in each generation (ADR 0085 addendum
   * 2026-09-02).
   */
  readonly maxResidentBytes?: number;
};

/**
 * Measurements of lazy loading (fields that let a check see "read only touched blocks"
 * non-vacuously).
 */
export type Gemma4PleStats = {
  /** Number of **whole** block fetches (cache misses that were not routed to row reads). */
  readonly loads: number;
  /**
   * Number of ranges read by row reads (pairs with {@link Gemma4PleStats.loads}).
   *
   * This field lets "no whole read happened" be checked non-vacuously — zero whole reads alone also
   * holds when "nothing was read at all", so only together with the number of ranges read does it
   * show the row-read path was taken. One row is read from the two tables `values` and `scales`, so
   * a row read from both by row reads counts twice.
   */
  readonly rowReads: number;
  /** Number of blocks currently resident (both tables combined). */
  readonly resident: number;
  /**
   * Host RAM taken by resident blocks and the row cache (the same unit as
   * {@link Gemma4PleOptions.maxResidentBytes}).
   *
   * A whole block is reserved when its read **starts**; a row adds the bytes of the buffers it keeps
   * once kept (measured from the kept arrays, so a row that aliased a block would show up here).
   * Blocks stay at or below the budget and rows at or below {@link ROW_CACHE_CAPACITY} × one row, so
   * the sum is at most `budget + ROW_CACHE_CAPACITY × rowBytes` (0 when the budget is 0). Temporary
   * buffers during reads and bookkeeping objects such as Maps are not counted.
   */
  readonly residentBytes: number;
};

export type Gemma4Ple = {
  /**
   * token id sequence → `per_layer_inputs` `[1, ids.length, layers, dim]` f32.
   *
   * Prefill pad rows also **pass their id as is** (the host fills the pad of `input_ids` with 0, so
   * the PLE of row 0 goes in here too) — the values equal those of the in-graph lookup, and the
   * value contract of pad rows (ADR 0066 addendum of 2026-08-17, item 6) holds.
   */
  gather(ids: readonly number[], options?: Gemma4PleReadOptions): Promise<Tensor>;
  stats(): Gemma4PleStats;
  /**
   * Releases the resident blocks (up to {@link Gemma4PleOptions.maxResidentBytes} of host RAM).
   *
   * MUST: have a release surface. GPU-side residency is returned by `Session.dispose`, but this is
   * a **host RAM cache**, so without a surface "disposing the pipeline does not return the resident
   * bytes" (one reference left holding the data keeps it for the life of the process).
   *
   * Any later {@link Gemma4Ple.gather} fails loudly — if a released owner silently started
   * re-reading, the "RAM does not come back after dispose" shape would return. Idempotent.
   */
  dispose(): void;
};

/**
 * The unique-row lower bound for routing to a whole read (**a provisional value set from
 * measurements**).
 *
 * One range read takes 0.1–0.6 ms, so **for this one gather** row reads are cheaper than a whole
 * read at any row count. A whole read pays off only when "loading it into the free budget for later
 * gathers' hits", and this threshold is the line of the expectation that "a block touched this much
 * in one gather will be touched again". Too low and residency churns; too high and residency never
 * warms up (`SEEK_FULL_ROWS` of ADR 0085 addendum 2026-09-07, carried over to blocks).
 *
 * The actual lower bound is **the smaller of this and the block's row count**
 * ({@link fullBlockRows}) — for a gather that touches every row of a block, one whole read is
 * always cheaper than `rows × 2` row reads. Without this rule, blocks with few rows (the trailing
 * remainder, small distributions) would never become resident however often they are touched.
 */
const FULL_BLOCK_ROWS = 32;

/**
 * The unique-row lower bound for routing that block to a whole read (all rows when the block is
 * narrow).
 */
const fullBlockRows = (block: Gemma4PleBlock): number =>
  Math.min(FULL_BLOCK_ROWS, block.stop - block.start);

/**
 * The **concurrency limit of row reads** (in-flight reads one gather holds across both tables).
 *
 * MUST: have a limit. Some sources use **1 read = 1 fd** (the local directory sources open the file
 * on every read — `Deno.open` in `denoDirectory`, `open` in `nodeDirectory`), so without a limit
 * one gather holds as many fds at once as **the number of ranges it reads**. Streaming a whole
 * 768-row prefill chunk through row reads (no free resident budget) is the realistic upper bound
 * and eats the whole remainder of `ulimit -n 1024` (the Linux default). Measured (2026-09-07, the
 * version without a limit): a peak of 768 at 768 rows, `Too many open files (os error 24)` at 1,200
 * rows.
 *
 * MUST: apply the limit to **the whole gather**, not per table or per block — applied separately, a
 * prefill spread over several blocks multiplies the limit by their number (that is where fds run
 * out).
 */
const ROW_READ_CONCURRENCY = 16;

/**
 * Row count of the LRU that skips re-reads of the same token (ADR 0085 addenda 2026-09-12 and
 * 2026-10-27).
 *
 * MUST: it is headroom **outside** the block budget, not the budget's free space. With the default
 * budget (two of the largest blocks) one resident values block and the scales block leave less than
 * one row of free space, so a row cache bounded by that space keeps nothing and decode re-reads
 * every row (perf-ledger H-41).
 */
const ROW_CACHE_CAPACITY = 256;

/**
 * One row that was looked up (kept as stored bytes, so the dequantization order does not
 * change). Each array occupies its own buffer of exactly one row; a bf16 row has no scales.
 */
type CachedRow = {
  readonly values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer> | undefined;
};

/**
 * The lookup result of one table (a view of a whole block, or one row obtained by a row read).
 *
 * `from` is the row start in **elements** (bytes for `values`, the f32 index for `scales`), the
 * form that `writeRow` in {@link createGemma4Ple} takes as is.
 */
type Supply<T> = {
  readonly array: T;
  readonly from: number;
  /**
   * true when obtained by a row read (= it occupies its own buffer of one row). Otherwise
   * `array` is a view of a whole block, and the row cache keeps a copy of the row instead.
   */
  readonly owned: boolean;
};

/**
 * The shared failure slot of one gather (the first cause).
 *
 * The row-read workers and the wait for whole blocks see the same one — so that a failure on one
 * side stops new issues, and the gather rejects with the first cause only after every started read
 * has settled (keeping ADR 0085 addendum 2026-09-11's "wait for both to settle" in the per-job
 * structure as well).
 */
type GatherFailure = { cause?: { readonly error: unknown } };

/** The plan of one table (blocks routed to whole reads, and ids routed to row reads). */
type TablePlan = {
  readonly table: Gemma4PleTable;
  /** block index → the ids read from that block. */
  readonly grouped: Map<number, number[]>;
  /** block index → whole read (already started). */
  readonly full: Map<number, Promise<ArrayBuffer>>;
  /** id routed to a row read → block index. */
  readonly rows: Map<number, number>;
  /** Row-read results (id → one row occupying its own buffer). */
  readonly read: Map<number, ArrayBuffer>;
};

/**
 * Builds the gather that reads **only what it touches** of the blocks the index points at (ADR 0085
 * decision 3 + 0109 decision 4).
 *
 * MUST: zero module side effects (nothing happens until this function is called).
 */
export const createGemma4Ple = (options: Gemma4PleOptions): Gemma4Ple => {
  const { index, openBlock, vocabSize } = options;
  const budget = options.maxResidentBytes ?? defaultGemma4PleResidentBytes(index);
  // The budget is the caller's `maxResidentPleBytes` itself, so it is input-caused (ADR 0107
  // decision 2).
  if (!Number.isSafeInteger(budget) || budget < 0) {
    throw new ModelInputError(`maxResidentBytes ${budget} is not a non-negative integer`);
  }
  const largest = largestBlockBytes(index);
  // MUST: "a budget that cannot hold even one block" fails loudly. Silently exceeding it voids the
  // budget, silently honouring it leaves the gather unable to read — both betray the caller's
  // setting. 0 is the exception: valid as "keep nothing resident" (drop each block as soon as it is
  // read). It is compared with the asset sizes, but what is rejected is the budget the caller
  // passed, so it is input-caused.
  if (budget > 0 && budget < largest) {
    throw new ModelInputError(
      `maxResidentBytes ${budget} is less than one PLE block, ${largest} bytes` +
        ` (this index has ${gemma4PleTables(index)
          .map(([name, table]) => `${table.blocks.length} ${name} blocks`)
          .join(" / ")} — pass 0 for no residency)`,
    );
  }
  // (1) the index's row count and (2) the main embedding's vocab row count (ADR 0085 decision 5's
  // cross-check). This is a mismatch of the baked combination, so it stays a plain `Error` (the
  // caller cannot fix it by changing a setting).
  if (index.tokens !== vocabSize) {
    throw new Error(
      `The PLE index's row count ${index.tokens} differs from the main embedding's vocab row count ${vocabSize}` +
        ` (a combination baked with a different vocabulary — the shapes match while the ids that can be looked up disagree)`,
    );
  }
  const stride = index.layers * index.dim;
  const rowValueBytes = index.values.rowBytes;
  /** The scale table (absent for a bf16 index — its values are not quantized). */
  const scaleTable = index.storage === "bf16" ? undefined : index.scales;
  /** Values per byte (0 for bf16, whose values are wider than a byte — {@link writeRow}). */
  const factor = index.storage === "bf16" ? 0 : packFactor(index);
  /** Scales of one layer of a row (1 = one per row — schema 3; 0 for bf16). */
  const groups = index.storage === "bf16" ? 0 : scalesPerLayer(index);
  /** **Element count** of one row of `scales` (f32 × layers × groups). */
  const scaleStride = index.layers * groups;
  /** Values that share one scale (unused for bf16). */
  const groupSize = index.storage === "bf16" ? index.dim : index.groupSize;
  /**
   * One f32 and its bits: a bf16 value is widened by writing its bits into the upper half
   * (`bits << 16` — exact, no rounding; the definition of the runtime's `format/bf16.ts`, which the
   * GPU `embedding` follows with `bitcast<f32>`). Owned by this gather, written and read inside one
   * synchronous `writeRow`.
   */
  const widened = new Float32Array(1);
  const widenedBits = new Uint32Array(widened.buffer);
  /**
   * Rows are kept as quantized bytes too, so the dequantization order does not change. Insertion
   * order = LRU.
   */
  const rowCache = new Map<number, CachedRow>();
  /**
   * Insertion order = LRU (a touched block is moved to the tail; overflow is dropped from the
   * head).
   */
  const resident = new Map<
    string,
    { readonly bytes: number; readonly data: Promise<ArrayBuffer> }
  >();
  /**
   * Total bytes of resident blocks (including reads already started). Always moves together with
   * `resident`; the row cache is outside it.
   */
  let residentBytes = 0;
  let loads = 0;
  let rowReads = 0;
  let disposed = false;

  const release = (asset: string): void => {
    const entry = resident.get(asset);
    if (entry === undefined) return;
    resident.delete(asset);
    residentBytes -= entry.bytes;
  };

  /**
   * Takes one block from the resident cache (reads it **whole** when not resident).
   *
   * MUST: keep it synchronous (no await between looking up `resident` and registering) — with one,
   * two concurrent requests for the same block would both judge it a miss and read it twice.
   */
  const acquire = (table: Gemma4PleTable, position: number): Promise<ArrayBuffer> => {
    const block = table.blocks[position];
    const cached = resident.get(block.asset);
    if (cached !== undefined) {
      // Touched, so move it to the tail (Map iteration order = insertion order).
      resident.delete(block.asset);
      resident.set(block.asset, cached);
      return cached.data;
    }
    const bytes = gemma4PleBlockBytes(table, block);
    loads += 1;
    // The reader lives only for this one read (the MUST in the module doc).
    const data = readAssetRange(openBlock(block.asset), 0, bytes);
    // MUST: do not keep a failed fetch resident (the next gather would keep taking the same
    // rejected Promise).
    data.catch(() => {
      if (resident.get(block.asset)?.data === data) release(block.asset);
    });
    // A gather already running may complete during dispose, but nothing goes back into the owner's
    // cache.
    if (disposed) return data;
    resident.set(block.asset, { bytes, data });
    residentBytes += bytes;
    // The budget is measured in bytes (not a count). With a budget of 0 the block just added is
    // dropped here too, but the running gather holds the resolved data itself, so values stay
    // consistent.
    for (const oldest of resident.keys()) {
      if (residentBytes <= budget) break;
      release(oldest);
    }
    return data;
  };

  /**
   * Dequantizes one row and writes it at position `target` of `data`.
   *
   * MUST: the whole-block path and the row-read path share **this one implementation** (with two
   * formulas, a rounding point drifting in only one of them goes unnoticed because the values
   * "roughly match").
   * MUST: **two steps**, dequantization → embed scale (the same order and rounding points as the
   * GPU-side `embedding` and the `mul` right after it). JS arithmetic is f64, so the first step is
   * brought down to f32 with `Math.fround` before multiplying — rounding once in f64 can give a
   * different result from the two-step rounding in the subnormal range (ADR 0085 decision 4).
   * A bf16 row (`scales` undefined) widens each value to f32 exactly (`widened`) and multiplies by
   * the embed scale — the GPU's `embedding` output is that same exact f32.
   */
  const writeRow = (
    data: Float32Array<ArrayBuffer>,
    target: number,
    values: Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer>,
    valuesFrom: number,
    scales: Float32Array<ArrayBuffer> | undefined,
    scalesFrom: number,
  ): void => {
    let cursor = target;
    if (scales === undefined) {
      const end = cursor + stride;
      for (let byte = valuesFrom; cursor < end; byte += BF16_BYTES, cursor += 1) {
        // Little-endian bf16 bits (the writer's int16 view — ADR 0131 decision 2).
        widenedBits[0] = (values[byte] | (values[byte + 1] << 8)) << 16;
        data[cursor] = widened[0] * index.embedScale;
      }
      return;
    }
    for (let layer = 0; layer < index.layers; layer += 1) {
      // One scale per group of `groupSize` values (schema 3: one group = the whole layer).
      const scaleBase = scalesFrom + layer * groups;
      const base = valuesFrom + (layer * index.dim) / factor;
      if (factor === 1) {
        for (let column = 0; column < index.dim; column += 1) {
          const scale = scales[scaleBase + Math.floor(column / groupSize)];
          data[cursor + column] = Math.fround(values[base + column] * scale) * index.embedScale;
        }
      } else {
        const bits = 8 / factor;
        const mask = (1 << bits) - 1;
        const offset = 1 << (bits - 1);
        for (let column = 0; column < index.dim; column += 1) {
          const scale = scales[scaleBase + Math.floor(column / groupSize)];
          const value =
            ((values[base + Math.floor(column / factor)] >>> (bits * (column % factor))) & mask) -
            offset;
          data[cursor + column] = Math.fround(value * scale) * index.embedScale;
        }
      }
      cursor += index.dim;
    }
  };

  /**
   * **Copies the computed f32 bytes** to the remaining positions with the same id.
   *
   * A copy, not a recomputation, so bit identity with the two-step rounding is guaranteed by
   * construction.
   */
  const copyDuplicates = (
    data: Float32Array<ArrayBuffer>,
    positions: readonly number[],
    first: number,
  ): void => {
    for (let rest = 1; rest < positions.length; rest += 1) {
      data.copyWithin(positions[rest] * stride, first, first + stride);
    }
  };

  /** Groups the ids to read by block (the policy is not decided yet). */
  const groupTable = (table: Gemma4PleTable, missing: readonly number[]): TablePlan => {
    const grouped = new Map<number, number[]>();
    for (const id of missing) {
      const position = gemma4PleBlockOf(table, id);
      const ids = grouped.get(position);
      if (ids === undefined) grouped.set(position, [id]);
      else ids.push(id);
    }
    return { table, grouped, full: new Map(), rows: new Map(), read: new Map() };
  };

  /**
   * Touches the blocks that are already resident, moving them to the LRU tail, and adds them to
   * `planned`.
   *
   * MUST: run **both tables first, together** (before the whole reads that follow) — finishing
   * "touch → load" per table lets the first table's whole read evict **the other table's hit**
   * (eviction happens in one LRU spanning both tables).
   */
  const touchHits = (plan: TablePlan, planned: { value: number }): void => {
    for (const position of plan.grouped.keys()) {
      const block = plan.table.blocks[position];
      if (!resident.has(block.asset)) continue;
      planned.value += gemma4PleBlockBytes(plan.table, block);
      plan.full.set(position, acquire(plan.table, position));
    }
  };

  /**
   * Routes the non-resident blocks to "whole read" or "row reads".
   *
   * `planned` is **the resident amount this gather holds** (hits + what it is about to load), one
   * value shared across both tables. It stops short of exceeding the budget, so **one gather never
   * evicts its own hits** — while blocks loaded by earlier gathers are evicted, so the working set
   * rotates through the LRU.
   */
  const decideTable = (plan: TablePlan, planned: { value: number }): void => {
    for (const [position, ids] of plan.grouped) {
      if (plan.full.has(position)) continue;
      const block = plan.table.blocks[position];
      const bytes = gemma4PleBlockBytes(plan.table, block);
      if (ids.length >= fullBlockRows(block) && planned.value + bytes <= budget) {
        planned.value += bytes;
        plan.full.set(position, acquire(plan.table, position));
        continue;
      }
      for (const id of ids) plan.rows.set(id, position);
    }
  };

  /** One row read (reads `[row offset, +rowBytes)` into its own buffer). */
  const readRow = async (plan: TablePlan, id: number): Promise<void> => {
    const position = plan.rows.get(id) as number;
    const block: Gemma4PleBlock = plan.table.blocks[position];
    rowReads += 1;
    plan.read.set(
      id,
      await readAssetRange(
        openBlock(block.asset),
        (id - block.start) * plan.table.rowBytes,
        plan.table.rowBytes,
      ),
    );
  };

  /**
   * Folds the row reads of both tables into one list and runs it through a pool of
   * {@link ROW_READ_CONCURRENCY} workers.
   *
   * A failure does not reject; the first one is recorded in `failure` (the caller waits for every
   * path).
   */
  const gatherRows = async (
    plans: readonly TablePlan[],
    options: Gemma4PleReadOptions,
    failure: GatherFailure,
  ): Promise<void> => {
    const jobs: { readonly plan: TablePlan; readonly id: number }[] = [];
    for (const plan of plans) {
      for (const id of plan.rows.keys()) jobs.push({ plan, id });
    }
    if (jobs.length === 0) return;
    let next = 0;
    const worker = async (): Promise<void> => {
      // MUST: no await between taking a job and advancing `next` (with one, two workers read the
      // same row).
      // MUST: take no new job after a failure — continuing would overlap the caller's retry after
      // the failure with this gather's remaining reads, and the concurrency limit would no longer
      // hold across the whole gather.
      while (failure.cause === undefined && next < jobs.length) {
        const job = jobs[next];
        next += 1;
        try {
          // Abort is checked between steps (`AssetReader` has no way to stop one read midway).
          options.signal?.throwIfAborted();
          await readRow(job.plan, job.id);
        } catch (error) {
          failure.cause ??= { error };
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(ROW_READ_CONCURRENCY, jobs.length) }, worker));
  };

  /**
   * Looks up an id's row in one table (the row-read result → otherwise a view of the whole block).
   *
   * `rowUnits` is the **element count** of one row (bytes for `values`, the f32 count for
   * `scales`).
   */
  const supplyOf = <T>(
    plan: TablePlan,
    id: number,
    blocks: ReadonlyMap<number, ArrayBuffer>,
    view: (buffer: ArrayBuffer) => T,
    rowUnits: number,
  ): Supply<T> => {
    const row = plan.read.get(id);
    if (row !== undefined) return { array: view(row), from: 0, owned: true };
    const position = gemma4PleBlockOf(plan.table, id);
    const buffer = blocks.get(position);
    if (buffer === undefined) {
      // If this happens, the plan and the lookup disagree (= a bug in this implementation), so do
      // not silently hand out a zero row.
      throw new Error(`PLE gather: token ${id}: block ${position} was not fetched`);
    }
    return {
      array: view(buffer),
      from: (id - plan.table.blocks[position].start) * rowUnits,
      owned: false,
    };
  };

  /**
   * Waits for the whole blocks (block index → bytes).
   *
   * Does not reject; failures are recorded in `failure` — leaving at the first failure would return
   * control to the caller before the remaining blocks and row reads have settled.
   */
  const awaitFull = async (
    plan: TablePlan,
    failure: GatherFailure,
  ): Promise<ReadonlyMap<number, ArrayBuffer>> => {
    const blocks = new Map<number, ArrayBuffer>();
    await Promise.all(
      [...plan.full].map(async ([position, data]) => {
        try {
          blocks.set(position, await data);
        } catch (error) {
          failure.cause ??= { error };
        }
      }),
    );
    return blocks;
  };

  // bf16 values are read as bytes (`writeRow` assembles the little-endian 16-bit words — a
  // `Uint16Array` view of a block would need 2-byte alignment the row reads do not guarantee).
  const valuesView = (buffer: ArrayBuffer): Int8Array<ArrayBuffer> | Uint8Array<ArrayBuffer> =>
    index.storage === "i8" ? new Int8Array(buffer) : new Uint8Array(buffer);
  const scalesView = (buffer: ArrayBuffer): Float32Array<ArrayBuffer> => new Float32Array(buffer);

  /**
   * The row of a supply in its own buffer: a row read as is, a view of a whole block copied out.
   *
   * MUST: copy (`slice`, not `subarray`) — keeping a view would keep the whole block alive outside
   * the budget after its LRU eviction.
   */
  const ownedRow = <T extends { slice(start: number, end: number): T }>(
    supply: Supply<T>,
    rowUnits: number,
  ): T => (supply.owned ? supply.array : supply.array.slice(supply.from, supply.from + rowUnits));

  return {
    async gather(ids: readonly number[], options: Gemma4PleReadOptions = {}): Promise<Tensor> {
      if (disposed) {
        throw new Error(
          "PLE gather: a disposed index cannot be looked up (its residency is released)",
        );
      }
      if (ids.length < 1) throw new Error("The token sequence of the PLE gather is empty");
      // Called already aborted: return without touching a single asset byte (the first step
      // boundary).
      options.signal?.throwIfAborted();
      ids.forEach((id, position) => {
        // (3) the ids actually looked up (including the special ids the tokenizer can produce) —
        // out of range is not OOB but "a valid row of another token", so this is the only
        // fail-loudly point.
        if (!Number.isSafeInteger(id) || id < 0 || id >= index.tokens) {
          throw new Error(
            `token id[${position}] ${id} is outside the PLE index's 0..${index.tokens - 1}`,
          );
        }
      });
      const data = new Float32Array(ids.length * stride);
      // Group the positions of each id (dequantization then happens once per id).
      const rows = new Map<number, number[]>();
      ids.forEach((id, position) => {
        const positions = rows.get(id);
        if (positions === undefined) rows.set(id, [position]);
        else positions.push(position);
      });
      // (1) Write the row-cache hits first (touch them before a later read's eviction drops them).
      const missing: number[] = [];
      for (const [id, positions] of rows) {
        const cached = rowCache.get(id);
        if (cached === undefined) {
          missing.push(id);
          continue;
        }
        rowCache.delete(id);
        rowCache.set(id, cached);
        const first = positions[0] * stride;
        writeRow(data, first, cached.values, 0, cached.scales, 0);
        copyDuplicates(data, positions, first);
      }
      if (missing.length > 0) {
        // (2) Group by block per table, and decide the policy **after touching the hits of both
        // tables** (a bf16 index has the values table alone).
        const planned = { value: 0 };
        const valuePlan = groupTable(index.values, missing);
        const scalePlan = scaleTable === undefined ? undefined : groupTable(scaleTable, missing);
        const plans = scalePlan === undefined ? [valuePlan] : [valuePlan, scalePlan];
        for (const plan of plans) touchHits(plan, planned);
        for (const plan of plans) decideTable(plan, planned);
        // (3) Run the row reads first (an extension of hits-first — running them after waiting for
        // the whole reads would let that eviction drop this gather's row-cache hits). The waits (=
        // the failure record) are set up before the row reads, so that a whole-block failure also
        // stops new row reads.
        const failure: GatherFailure = {};
        const valueFull = awaitFull(valuePlan, failure);
        const scaleFull = scalePlan === undefined ? undefined : awaitFull(scalePlan, failure);
        await gatherRows(plans, options, failure);
        const valueBlocks = await valueFull;
        const scaleBlocks = await scaleFull;
        // MUST: reject only after every started read has settled (do not overlap old reads with a
        // retry or dispose after the failure).
        if (failure.cause !== undefined) throw failure.cause.error;
        // (4) Match the two tables and write. Only the last ROW_CACHE_CAPACITY missing ids are
        // admitted to the row cache: earlier ones would be evicted later in this same loop, so
        // copying them is wasted work (a 768-row prefill would copy ≈ 7 MB), and the final cache
        // contents are the same as admitting every id in order.
        const admitFrom = missing.length - ROW_CACHE_CAPACITY;
        for (const [at, id] of missing.entries()) {
          const positions = rows.get(id) as number[];
          const values = supplyOf(valuePlan, id, valueBlocks, valuesView, rowValueBytes);
          const scales =
            scalePlan === undefined || scaleBlocks === undefined
              ? undefined
              : supplyOf(scalePlan, id, scaleBlocks, scalesView, scaleStride);
          const first = positions[0] * stride;
          writeRow(data, first, values.array, values.from, scales?.array, scales?.from ?? 0);
          copyDuplicates(data, positions, first);
          // Every admitted row goes into the row cache, whichever path supplied it. Nothing is
          // kept with a budget of 0, and nothing is revived after abort or dispose.
          if (at >= admitFrom && budget > 0 && !disposed && options.signal?.aborted !== true) {
            rowCache.delete(id);
            rowCache.set(id, {
              values: ownedRow(values, rowValueBytes),
              scales: scales === undefined ? undefined : ownedRow(scales, scaleStride),
            });
            for (const oldest of rowCache.keys()) {
              if (rowCache.size <= ROW_CACHE_CAPACITY) break;
              rowCache.delete(oldest);
            }
          }
        }
      }
      return { dtype: "f32", shape: [1, ids.length, index.layers, index.dim], data };
    },
    stats(): Gemma4PleStats {
      let rowCacheBytes = 0;
      for (const row of rowCache.values()) {
        rowCacheBytes += row.values.buffer.byteLength + (row.scales?.buffer.byteLength ?? 0);
      }
      return {
        loads,
        rowReads,
        resident: resident.size,
        residentBytes: residentBytes + rowCacheBytes,
      };
    },
    dispose(): void {
      disposed = true;
      // Reads in flight are not stopped (the buffers they return are held by nobody and get
      // collected).
      resident.clear();
      residentBytes = 0;
      rowCache.clear();
    },
  };
};
