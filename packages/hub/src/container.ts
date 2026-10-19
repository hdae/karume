/**
 * The **fetch surface** of one container (`krm`) (ADR 0109 decision 7). It takes the manifest's
 * `container` field and returns a "part → byte range" interface that the reader (`openContainer` of
 * `@karume/runtime`) consumes as is.
 *
 * This surface holds **range reads only**: it pulls `[offset, offset+length)` from the warmed
 * persistent cache (or the local file) and picks the path by cost class (`AssetRangeReader.cost`).
 *
 * MUST: **warming (phase 1) stays out of this surface**. Not one byte of the weights of a model
 * that cannot run is downloaded (ADR 0108 decision 19 / ADR 0070 decision 5), so callers keep this
 * order:
 *
 * 1. fetch only part 0, which holds the descriptor, and pass admission
 * 2. once it passes, warm the weight parts ({@link ../mod.ts prefetchAssets} — with progress,
 *    cancellation and 4-way concurrency)
 * 3. open this surface and read
 *
 * Warming every part inside the surface would break this order from the inside. Opening without
 * warming still reads (the HF source's range reader inserts the same warm-up as phase 1 exactly
 * once when the file is not in the inventory — `source.ts` (8)), but that warm-up shows neither in
 * progress nor in cancellation.
 *
 * MUST: hub does not import `@karume/runtime` (dependencies go one way: models → hub / runtime).
 * {@link ContainerBlockSource} is a type spelt on this side that is **structurally compatible**
 * with the runtime's `BlockSource`; only the `verified` field is this surface's addition ("verified
 * is carried along" of ADR 0109 decision 7).
 *
 * MUST: **do not open** the descriptor's contents. Verifying the two documents and matching the
 * block index are `openContainer`'s job; hub keeps only the checks that close on the declaration
 * (`parts[].size`) alone (ADR 0109 decision 6).
 */

import { createFetchContext } from "./context.ts";
import { HubError } from "./errors.ts";
import { assertTightView } from "./fetch.ts";
import type { ContainerRef, FileRef } from "./manifest.ts";
import { type LoadedManifest, type LoadManifestOptions, pinnedSourceOf } from "./session.ts";
import { type AssetRangeReader, type PinnedSource, sourceForRef } from "./source.ts";

/**
 * The part → byte range interface. Structurally compatible with the `BlockSource` of
 * `@karume/runtime`, so it can be passed as is to `openContainer({ kind: "source", source })`.
 *
 * MUST: **callers do not modify** the returned bytes (depending on the source's implementation,
 * they may be the very buffer the source keeps holding). On the scan path the return is a view that
 * holds the whole buffer of the part, so a caller that keeps it longer than the range copies it
 * (holding it keeps the whole part alive, not just the range).
 */
export type ContainerBlockSource = {
  /** The number of declared parts (zero-length parts count too — the index is the part's id). */
  readonly partCount: number;
  /**
   * Whether the source has **verified the whole bytes** (ADR 0109 decision 7). For the HF source
   * the fetch layer compares sha256 while streaming the whole part, so `true`; a local directory
   * does not compare, so `false`. Readers use it to decide whether to digest each block (avoiding
   * the double digest on a cold read — container-v1 §7).
   */
  readonly verified: boolean;
  /**
   * The declared part length (synchronous; answers without fetching a byte). An out-of-range index
   * fails loudly.
   */
  partLength(index: number): number;
  /**
   * Returns `[offset, offset + length)`. The seek path returns a tight view of the range; the scan
   * path returns a view into the buffer of the part in the holding slot (`byteOffset` = start of
   * the range, `buffer` = the whole part).
   */
  read(part: number, offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * The range-read path. Decided once per part (it splits on the source's capabilities and cost
 * class).
 */
type ReadPlan =
  /** A source with cheap positional reads (browser Blob, local range reads) — read per block. */
  | { readonly kind: "seek"; readonly reader: AssetRangeReader }
  /**
   * A source whose skipping is proportional to the offset (the default under Deno and Node via HF)
   * and a source without range reads — read the part fully once and slice it (reading the blocks of
   * one part in order would make the skipping quadratic).
   */
  | { readonly kind: "scan" };

/** One holding slot of the scan path (an in-flight full read is held in the same slot). */
type HeldPart = {
  readonly part: number;
  readonly bytes: Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * Opens the fetch surface of one container. It **fetches nothing**, so it returns synchronously —
 * the files are touched only when {@link ContainerBlockSource.read} is called (the range reader of
 * each part is opened on its first read).
 *
 * This surface has no progress and no `signal` option (it does no warming, so there is nothing to
 * report). Cancellation is checked once before the interface is handed out and does not apply to
 * reads after opening — consumers hold a reader for its whole lifetime, so holding the signal from
 * open would take every later read down with it (`source.ts` (8)).
 *
 * Failures are handled in two ways: resolving the source (an unmapped cross-repo target) is missing
 * fetch configuration, so it is wrapped in a `HubFetchError`; a `read` failure passes the source's
 * bare `Error` through as is (as `openAsset` does — a read is not a fetch).
 */
export const openContainerSource = (
  loaded: LoadedManifest,
  container: ContainerRef,
  options: LoadManifestOptions = {},
): ContainerBlockSource => {
  // Check cancellation before building the interface (so source capabilities do not change how
  // cancellation looks — a convention of the common layer).
  options.signal?.throwIfAborted();
  const session = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, session.origin);
  const { parts } = container;

  const partAt = (index: number): FileRef => {
    if (!Number.isInteger(index) || index < 0 || index >= parts.length) {
      throw new Error(
        `hub: part ${index} is not in this container (${parts.length} declared — the index is the part's id)`,
      );
    }
    return parts[index];
  };

  // Cross-repo references are **per container** (parsing checks that they are uniform), so there is
  // exactly one source per container. Resolve and bind it once at open — looking it up per read
  // would rebuild the target's source on every read of a cross-repo container.
  let origin: PinnedSource;
  try {
    origin = sourceForRef(session, parts[0]);
  } catch (error) {
    // An unmapped cross-repo target (the bare Error of `sources/local.ts`) is missing configuration
    // on the caller's side. Fail the same way `fetchAssets` does for the same manifest (with path /
    // repo / revisionSha / available).
    if (error instanceof HubError) throw error;
    throw context.fetchFailure(parts[0], "fetch", error);
  }
  const verified = origin.origin.integrity === "network";

  const plans = new Map<number, Promise<ReadPlan>>();
  // MUST: look up by index only (the ref is derived with `partAt`) — letting callers pass an
  // (index, ref) pair would let a mismatched pair grab "another part's reader".
  const planFor = (index: number): Promise<ReadPlan> => {
    const existing = plans.get(index);
    if (existing !== undefined) return existing;
    const { openFile } = origin;
    const opened: Promise<ReadPlan> =
      openFile === undefined
        ? Promise.resolve({ kind: "scan" })
        : openFile(partAt(index), {}).then((reader): ReadPlan =>
            reader.cost === "seek" ? { kind: "seek", reader } : { kind: "scan" },
          );
    // MUST: do not remember a failed decision — remembering it would make a part that failed to
    // open on a transient error return the same reject on every later read (no way left to reopen).
    const decided = opened.catch((error: unknown) => {
      plans.delete(index);
      throw error;
    });
    plans.set(index, decided);
    return decided;
  };

  /**
   * The holding slot of the scan path. **It holds one part at a time** (reading another part
   * releases the previous one). Slices are views into the buffer, so no host RAM is added beyond
   * the buffer of the part the slot holds — the weight reader (the runtime's `containerBatches`)
   * pulls blocks one at a time and releases each in the iteration that uploaded it (the views it
   * pulled also point into this buffer). At a part boundary, two parts are alive for a while until
   * the released buffer is garbage-collected.
   *
   * MUST: **put the in-flight full read in the same slot as well** — if the slot held only "settled
   * bytes", concurrent reads of the same part would all enter readFile, and part length × count
   * would be alive at that moment.
   * NOTE: there is one slot, so reading **different** parts concurrently makes the later read evict
   *       the earlier one from the slot, and the evicted full read still runs to the end. The parts
   *       alive at once grow up to "the number of different parts read concurrently", and hub does
   *       not cap that number (reading an evicted part again repeats the full read). The buffer of
   *       an evicted part survives while the caller holds views sliced from it.
   *       The weight reader reads parts serially in order, so it never takes this shape. Row reads
   *       of assets (the PLE in models — reading different parts concurrently) do, and during
   *       prefill full reads of parts are alive at once up to the limit of concurrently issued row
   *       reads (`ROW_READ_CONCURRENCY` = 16 in models). `readAssetRange` copies the range, so
   *       finished buffers are not held (an asset exactly one part long is held without copying,
   *       but the amount held equals the copy).
   */
  let held: HeldPart | undefined;
  const readWhole = (index: number, ref: FileRef): Promise<Uint8Array<ArrayBuffer>> => {
    const current = held;
    if (current !== undefined && current.part === index) return current.bytes;
    const pending = origin
      .readFile(ref, {
        // This surface is not a fetch, so it emits no progress (the caller already warmed with
        // prefetchAssets).
        onProgress: () => {},
        sizeViolation: context.sizeViolation(ref),
      })
      .then((bytes) => assertTightView(bytes, ref.path));
    const entry: HeldPart = {
      part: index,
      // MUST: do not leave a failed full read in the slot (later reads would keep returning the
      // same reject).
      bytes: pending.catch((error: unknown) => {
        if (held === entry) held = undefined;
        throw error;
      }),
    };
    held = entry;
    return entry.bytes;
  };

  return {
    partCount: parts.length,
    verified,
    partLength: (index) => partAt(index).size,
    read: async (part, offset, length) => {
      const ref = partAt(part);
      if (ref.size === 0) {
        throw new Error(
          `hub: part ${part} (${ref.path}) cannot be read because it is a zero-length part` +
            ` (a reader does not read once partLength returns 0)`,
        );
      }
      if (
        !Number.isInteger(offset) ||
        !Number.isInteger(length) ||
        offset < 0 ||
        length < 0 ||
        offset + length > ref.size
      ) {
        throw new Error(
          `hub: part ${part} (${ref.path}) range [${offset}, ${offset + length}) is invalid` +
            ` (declared size ${ref.size} — only ranges of non-negative integers within size can be read)`,
        );
      }
      const plan = await planFor(part);
      if (plan.kind === "seek") {
        return assertTightView(await plan.reader.read(offset, length), ref.path);
      }
      // MUST: do not copy the slice (copying would hold the part length twice per part, buffer and
      // copy). The return is a view into the buffer of the part in the holding slot, and consumers
      // can read it as is because two things hold:
      //  - Alignment: the buffer is tight (`assertTightView` above), so byteOffset = block.offset,
      //    and block starts are 64 B aligned (enforced by descriptor parsing). That satisfies the
      //    4 B alignment a scale Float32Array view needs — removing `assertTightView` would leave
      //    this alignment to chance.
      //  - Lifetime: the weight reader uses up a batch's views by the fence, and only after that
      //    starts reading the next part (= the slot releases the buffer), so the views stay within
      //    the period the slot holds the buffer. The asset reader keeps a range only after
      //    `readAssetRange` (models) copied it (an asset exactly one part long shares the buffer
      //    without copying, but the amount held equals the copy).
      return (await readWhole(part, ref)).subarray(offset, offset + length);
    },
  };
};
