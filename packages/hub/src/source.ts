/**
 * The internal contract of sources ({@link SourceDriver} / {@link PinnedSource}) and the **opaque
 * handle** the public surface carries around ({@link DistributionSource}). The contract itself is
 * not public — the public surface stays `mod.ts` (the thinness of ADR 0008); this is the boundary
 * between the common layer (`fetch.ts`) and the adapters (`sources/`).
 *
 * Folding the current fetch paths, a source actually answers only five questions:
 *
 * 1. mutable ref → immutable generation identifier ({@link SourceDriver.resolveGeneration})
 * 2. the full bytes of the one `karume.json` ({@link PinnedSource.readManifest} — capped at
 *    {@link ../manifest.ts MAX_MANIFEST_BYTES}; **no** expected sha256 can exist)
 * 3. the full bytes of a `FileRef` ({@link PinnedSource.readFile} — with expected sha256 / size)
 * 4. "put a `FileRef` into a state where 3 is cheap later, without holding it in RAM"
 *    ({@link PinnedSource.prefetchFile} = phase 1 of the warm-up surface `prefetchAssets`. An
 *    **optional capability**, and a source without it is legitimate — it is an optimisation
 *    specific to HTTP + a persistent cache and means nothing for a source that reads directly)
 * 5. cross-repo (repo, revision) → another source ({@link PinnedSource.originFor})
 *
 * Outside the fetch paths, two more sit on top: the cache-inventory management surfaces (query and
 * delete in `inventory.ts`) — (6) {@link PinnedSource.inventory} / (7) {@link PinnedSource.evict}.
 * Both are **optional capabilities**, questions only a source that "keeps what it fetched" can
 * really answer.
 *
 * In addition, (8) reads **only a range** of a `FileRef` ({@link PinnedSource.openFile} =
 * {@link AssetRangeReader}). Also an **optional capability**, offered only by sources where
 * "pulling a few KB without reading the whole file" actually works (the only difference from (3) is
 * that it returns just `[offset, offset+length)` instead of the whole declared size).
 *
 * MUST: progress, concurrency (the in-flight byte budget), passing cancellation through, the tight
 * view check and the error context stay in `fetch.ts` as **conventions of the common layer**, not
 * capabilities of a source. Pushing them down into sources would mean rewriting the same invariants
 * for every new source.
 */

import type { IntegritySource } from "./errors.ts";
import { crossRefOf, type FileRef } from "./manifest.ts";
import type { LoadManifestOptions } from "./session.ts";

/**
 * The source's identity that diagnostics name. **The common layer assembles errors**
 * (`context.ts`), so a source holds only this value, which says "who it is".
 *
 * MUST: do not synthesise an identity the source lacks — a local source naming a repo / commit SHA
 * produces diagnostics pointing at a repo that does not exist (guidance to go and look on HF). The
 * design lets a source omit fields it lacks; instead {@link label} **always** names the place
 * actually fetched from.
 */
export type SourceOrigin = {
  /**
   * The one-line naming shown in diagnostic messages (HF: `repo owner/name @ <commit SHA>` / local:
   * the directory label as `localDirectory` formats it). The only field whose vocabulary may differ
   * per source.
   */
  readonly label: string;
  /** Where an integrity check failed (the `source` of {@link ../errors.ts IntegrityError}). */
  readonly integrity: IntegritySource;
  /**
   * A structured field of the HF vocabulary. **Omitted by sources without it** (as in
   * {@link ../errors.ts HubFetchError}).
   */
  readonly repo?: string;
  /**
   * The resolved generation identifier (commit SHA). Omitted by sources without the notion of
   * generations.
   */
  readonly revisionSha?: string;
};

/**
 * Assembles the error thrown when a byte count disagrees with the declaration. **Sources do not
 * assemble it** — the common layer holds the diagnostic context (the source's naming, the available
 * labels), so a source passes only "what it was" and **its own failure origin**
 * ({@link SourceOrigin.integrity}) (the single assembly point is `context.ts`).
 *
 * MUST: `integrity` is the calling source's own (`origin.integrity` — do not write a constant). A
 * cross-repo reference comes from a source other than the session's (a local session + a remote
 * cross-repo target is a valid setup), so the common layer cannot infer the failure origin from the
 * declaration — inferring it would slip one lying field in, of the form "bytes that came from the
 * network name `"local"` (= retrying is useless)".
 */
export type SizeViolation = (actual: number, integrity: IntegritySource) => Error;

/**
 * Options for reading one asset ({@link PinnedSource.readFile} /
 * {@link PinnedSource.prefetchFile}).
 */
export type FileReadOptions = {
  readonly signal?: AbortSignal;
  /**
   * Cumulative bytes received so far. **A source that reads directly may never call it** (treated
   * like a cache hit — the common layer can close the progress with the single `complete`).
   */
  readonly onProgress: (loaded: number) => void;
  readonly sizeViolation: SizeViolation;
};

/**
 * The **range reader** of one asset (capability (8), returned by {@link PinnedSource.openFile}).
 * Unlike a full read ({@link PinnedSource.readFile}), it returns only the wanted
 * `[offset, offset + length)` of the declared size — a surface for consumers that pull rows of a
 * few KB out of a part of hundreds of MiB (decoding the per-layer embedding table).
 *
 * {@link cost} names **a cost class** because consumers use it to change "the maximum row count for
 * which switching to row reads pays off": with a `"scan"` source each row involves skipping
 * proportional to the offset, so beyond some row count one full read is cheaper. It is named as a
 * class rather than a number (ms / bytes per second) because measured values move by two orders of
 * magnitude across environments (browser / runtime / disk), while **which class** is fixed by the
 * source's implementation.
 *
 * A reader **may reopen on every read** (it holds no fd or handle — `denoDirectory` calls
 * `Deno.open` on each read). Measured at 9,100 B × 2,000 reads: 46 µs/read against 23 µs/read with
 * a held fd, a difference negligible next to the wall of decoding one token (tens of ms).
 * So **there is no close surface** — nothing stays open, so callers carry no release duty.
 */
export type AssetRangeReader = {
  /**
   * `"seek"` = small regardless of the offset (positional file reads, `slice` of a lazy browser
   * Blob) / `"scan"` = proportional to the offset (skipping through a body stream).
   */
  readonly cost: "seek" | "scan";
  /**
   * Returns `[offset, offset + length)`.
   *
   * MUST: return exactly `length` bytes (never short). A request that runs past the file, or a file
   * shorter than declared that cannot fill it, fails loudly either way — passing a short return
   * silently would make the consumer read "zero-filled rows" as valid values.
   * MUST: return a view that spans its whole buffer (a tight view) — some asset readers use
   * `bytes.buffer` as the range itself without copying (callers of `openAsset`, `readAssetRange` in
   * models). Consumers of the container surface (`container.ts`) honour byteOffset, but on its seek
   * path too, byteOffset 0 directly gives the block alignment (the 4 B a scale Float32Array view
   * needs).
   */
  readonly read: (
    offset: number,
    length: number,
    options?: { readonly signal?: AbortSignal },
  ) => Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * The bounds check of one range read against the **declared** `size` ({@link AssetRangeReader}):
 * `offset` and `length` must be non-negative integers and the range must end within `size`.
 * `subject` names what is read in the message (a path, or a part and its path).
 *
 * MUST: the bound is the declared size, never the length of the file at hand — a file can be
 * longer than its declaration (a mixed-up quant, a copy still being written), and checking against
 * it would silently read bytes outside the manifest. Every surface that reads a range calls this
 * one helper so the rule and the message cannot drift apart between them.
 */
export const assertRangeWithinDeclaredSize = (
  subject: string,
  size: number,
  offset: number,
  length: number,
): void => {
  if (
    !Number.isInteger(offset) ||
    !Number.isInteger(length) ||
    offset < 0 ||
    length < 0 ||
    offset + length > size
  ) {
    throw new Error(
      `hub: ${subject} range [${offset}, ${offset + length}) is invalid` +
        ` (declared size ${size} — only ranges of non-negative integers within size can be read)`,
    );
  }
};

/** Options for reading one manifest. */
export type ManifestReadOptions = {
  readonly signal?: AbortSignal;
  readonly sizeViolation: SizeViolation;
  /**
   * The single conversion point bytes → `Manifest`.
   *
   * MUST: a source calls this **inside its own integrity check** (calling it outside would leave a
   * broken entry unevicted, returning the same `ManifestFormatError` every time). A throw from this
   * function means "the fetched bytes are broken", and the source may treat it as corruption.
   * MUST NOT: mix a cancellation check in here (it would lead to evicting a healthy entry).
   */
  readonly parse: (bytes: Uint8Array<ArrayBufferLike>) => void;
};

/**
 * A source pinned to a generation ((2)–(5)). The generation is pinned **once per session**, and
 * every later fetch stays in that one generation (resolving a mutable ref several times would let
 * the manifest and the weights come from different generations).
 */
export type PinnedSource = {
  /** The identity diagnostics name (for this generation and these coordinates). */
  readonly origin: SourceOrigin;
  /**
   * (2) Reads `karume.json` and runs `parse`. The bytes themselves are not handed to the common
   * layer.
   */
  readonly readManifest: (options: ManifestReadOptions) => Promise<void>;
  /**
   * (3) The full bytes of one file. Verification belongs to the source (the common layer only
   * checks that it spans the whole buffer — the tight view check in `fetch.ts`). **What can be
   * verified differs per source** — HF compares up to sha256, the local source checks only an exact
   * size match (it trusts the sha256).
   */
  readonly readFile: (
    ref: FileRef,
    options: FileReadOptions,
  ) => Promise<Uint8Array<ArrayBufferLike>>;
  /**
   * (4) Phase 1 (optional capability) — puts the file into a state where a later {@link readFile}
   * is cheap, without holding it in RAM. **A source without it is legitimate**; then the warm-up
   * surface emits only `complete` for that ref without warming it (for a source that reads
   * directly, "a later read is cheap" holds from the start).
   */
  readonly prefetchFile?: (ref: FileRef, options: FileReadOptions) => Promise<void>;
  /**
   * (8) Opens a range reader (**optional capability**) — in the same style as {@link prefetchFile},
   * **a source without it is legitimate** (a local source given a directory adapter without
   * positional reads has that shape). The common layer then returns `undefined` from `openAsset`,
   * and the consumer falls back to a full read.
   *
   * MUST NOT: fetch bytes for anything other than warming — the range's bytes are touched only when
   * {@link AssetRangeReader.read} is called. However, an HTTP + persistent-cache source may insert
   * **the phase-1 warm-up ({@link prefetchFile}) exactly once** for a ref that is not in the
   * inventory — on every path, pulling the first range needs one full fetch, so the opening side
   * does it rather than making callers "go through phase 1 first" (after opening, range reads alone
   * suffice, with nothing kept resident).
   * MAY: use the `signal` that open received **to cancel that warm-up**.
   * MUST NOT: let the opened handle keep the `signal` given at open — consumers hold a reader for
   * its whole lifetime, so holding the cancellation from open would take every later read down with
   * it (cancellation is carried per read by the `signal` of {@link AssetRangeReader.read}).
   */
  readonly openFile?: (
    ref: FileRef,
    options: { readonly signal?: AbortSignal },
  ) => Promise<AssetRangeReader>;
  /**
   * (5) The source for a cross-repo reference (`repo` / `revision` of a `FileRef` — ADR 0038 §7).
   * The target must be pinned to a generation identifier, so no generation resolution happens at
   * the target.
   *
   * **A source that cannot determine the target may throw** (the local source has only explicit
   * mappings, and an unmapped target fails loudly — it does not guess an adjacent directory of the
   * same name). The common layer wraps that throw with context as a fetch failure.
   */
  readonly originFor: (repo: string, revision: string) => PinnedSource;
  /**
   * (6) Inventory query (**optional capability**) — returns the set of
   * {@link ../manifest.ts fileRefKey} of the given refs that **can be read without going to the
   * network**.
   *
   * MUST: every ref passed in belongs to **this source's coordinates** (the common layer splits
   * cross-repo refs per origin with `originFor` before passing them). Re-dispatching cross-repo
   * refs on the source side would count the same ref under two keys.
   */
  readonly inventory?: (refs: readonly FileRef[]) => Promise<ReadonlySet<string>>;
  /**
   * (7) Inventory deletion (**optional capability**) — deletes the cached copies of the given refs
   * and returns **what was actually deleted**.
   *
   * A source without it is legitimate (the contents of a local directory are the user's assets, not
   * fetched copies, and nothing there is hub's to delete). The common layer then refuses with a
   * fail loud.
   */
  readonly evict?: (refs: readonly FileRef[]) => Promise<readonly FileRef[]>;
};

/**
 * A source before generation resolution ((1)). The implementation that the public
 * {@link DistributionSource} wraps.
 *
 * Per-surface options (`fetch` / `caches` / `headers` / `onCacheError`) are passed **on each
 * {@link pin} call, not when the source is created**. A source holds only "where to fetch from";
 * "with which options" comes from the surface's options — otherwise the `fetch` passed to
 * `loadManifest` would silently keep applying to later `fetchAssets` calls, and per-surface
 * replacement would stop working.
 */
export type SourceDriver = {
  /**
   * The naming before generation resolution (for HF it includes the **requested** ref, as in `repo
   * owner/name @ main`).
   */
  readonly origin: SourceOrigin;
  /**
   * (1) mutable ref → immutable generation identifier. **The session's single resolution point**;
   * the return value pins every later fetch. A source without the notion of generations may return
   * a fixed value (the empty string).
   */
  readonly resolveGeneration: (options: LoadManifestOptions) => Promise<string>;
  /** Opens the source pinned to the resolved generation, with the options of that call. */
  readonly pin: (generation: string, options: LoadManifestOptions) => PinnedSource;
};

/**
 * Public handle → implementation. It is not a private field of the handle (`#driver`) because
 * reading the field out of the class would need a `static` block as a bridge (it runs at class
 * evaluation = import time) — bundlers treat that as a side effect and keep even unused classes in
 * the bundle (what the cross-cutting invariant "no execution at import time" protects). Building an
 * empty `WeakMap` is dropped as a side-effect-free expression.
 *
 * The key is the handle itself (a weak reference), and this table is invisible from outside the
 * module — as with a private field, holding the handle does not let anyone read the implementation.
 */
const drivers = new WeakMap<DistributionSource, SourceDriver>();

/**
 * The **public handle** of a source. The value `loadManifest(source, …)` / `fromPretrained(source)`
 * take; its contents ({@link SourceDriver}) exist only inside hub.
 *
 * MUST: grow no public members — if fetch implementation details (generation resolution, pin,
 * cross-repo references) leaked into the public surface, every new source would raise public
 * compatibility concerns. Discrimination is by **identity** (`instanceof`) too: a brand field could
 * be forged by users, and structural discrimination (`"repo" in value`) would silently let a
 * misspelt `HubRepoRef` through as a source.
 */
export class DistributionSource {
  /**
   * A type-only nominal brand (absent at run time — discrimination is `instanceof`, per the MUST
   * above). A class without members is structurally the same type as `{}`, and any object
   * (including a misspelt `HubRepoRef`) would type-check as a `DistributionSource`.
   */
  declare private readonly brand: never;

  /**
   * MUST: only the factories of the source adapters (`sources/`) call this. `mod.ts` does not
   * export `SourceDriver`, so users cannot spell this argument as a type.
   */
  constructor(driver: SourceDriver) {
    drivers.set(this, driver);
  }
}

/** Takes the implementation out of a public handle (called only inside hub). */
export const driverOf = (source: DistributionSource): SourceDriver => {
  const driver = drivers.get(source);
  if (driver === undefined) {
    // A value created without going through the constructor
    // (`Object.create(DistributionSource.prototype)`, …).
    throw new Error(
      "hub: no implementation is bound to the source handle (not a value created by a hub factory)",
    );
  }
  return driver;
};

/**
 * Checks whether a value is a source handle (**the only discrimination point** for the `ref |
 * source` union).
 *
 * `mod.ts` exports {@link DistributionSource} **only as a type** (with no members to grow,
 * exporting the class value would only let users create broken sources), so this predicate is the
 * only way to spell an identity check outside hub. As the MUST above says, leaving the check to
 * users drifts toward structural discrimination (`"repo" in value`), which silently lets a misspelt
 * `HubRepoRef` through as a source.
 */
export const isDistributionSource = (value: unknown): value is DistributionSource =>
  value instanceof DistributionSource;

/**
 * Decides the source that fetches one `FileRef`. **Only a cross-repo reference (a ref with both
 * `repo` and `revision`)** comes from the declared (repo, revision) instead of the session's
 * source.
 *
 * MUST: the branch lives in this one place only — writing it per surface would let a mix-up of the
 * form "fetch another repo's same-named path from our own repo" pass silently when one side lets a
 * cross-repo reference slip through.
 */
export const sourceForRef = (source: PinnedSource, ref: FileRef): PinnedSource => {
  const cross = crossRefOf(ref);
  return cross === undefined ? source : source.originFor(cross.repo, cross.revision);
};
