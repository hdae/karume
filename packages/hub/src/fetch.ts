/**
 * The common fetch layer. **It holds no source itself** — the source surface is the contract in
 * `source.ts`, implemented in `sources/` (HF and local directory). What remains here are only the
 * conventions that must be the same whatever the source is:
 *
 * - The generation is resolved **only once per session**, and the manifest and every file are
 *   fetched pinned to that one generation (resolving a mutable ref several times would let the
 *   manifest and the weights come from different generations). The only exception is a **cross-repo
 *   reference** the manifest declares explicitly (`repo` / `revision` of a `FileRef` — ADR 0038
 *   §7); that one file comes from the declared (repo, revision), not from the session's generation.
 * - Deduplication of fetches and of the progress total ({@link fileRefKey}), and progress
 *   aggregation (`progress.ts`).
 * - Pacing of concurrent fetches differs per surface (a byte budget on the whole-file surface,
 *   4 files on the warm-up surface).
 * - `AbortSignal` is passed through to every fetch and is checked explicitly at the surface
 *   boundaries, **even in stretches where the source does not go to the network**.
 * - The check that handed-over bytes span the whole buffer (a tight view).
 * - Adding context to failures (`context.ts` — including restoring the real first failure).
 */

import { purgeLegacyCaches } from "./cache.ts";
import { createByteAdmission } from "./concurrency.ts";
import {
  createFetchContext,
  type FetchContext,
  manifestFetchFailure,
  manifestOversize,
  revisionResolutionFailure,
} from "./context.ts";
import { HubError, ManifestFormatError, ManifestReferenceError } from "./errors.ts";
import {
  type FileRef,
  fileRefKey,
  isRepoName,
  type Manifest,
  MANIFEST_FILENAME,
  parseRepoManifest,
} from "./manifest.ts";
import { createProgressEmitter, type ProgressEmitter } from "./progress.ts";
import {
  type FetchAssetsOptions,
  type HubRepoRef,
  LoadedManifest,
  type LoadManifestOptions,
  pinnedSourceOf,
} from "./session.ts";
import {
  type AssetRangeReader,
  type DistributionSource,
  driverOf,
  isDistributionSource,
  type PinnedSource,
  sourceForRef,
} from "./source.ts";
import { createHfSource } from "./sources/hf.ts";

/**
 * The number of concurrent fetches of the warm-up surface {@link prefetchAssets} (phase 1), so that
 * a manifest with dozens of components does not break the connections. Phase 1 streams the body
 * straight into the cache, with no pre-allocated receive buffer = the count alone does not
 * determine RAM.
 */
const CONCURRENCY = 4;

/**
 * The in-flight byte budget of the whole-file surface {@link fetchAssets} (1.5GiB =
 * 1,610,612,736 bytes).
 *
 * With `karume/5`, container parts (the weights) go through the warm-up surface (phase 1) and the
 * container surface (`openContainerSource`); only the manifest's `assets` (tokenizer, style
 * vectors, …) go through this surface (the caller in `@karume/models` is the single place that
 * fetches assets). The "count 2, total = largest part length × 2" that ADR 0109 decision 7 set for
 * parts is not applied to this surface, and the budget stays at 1.5GiB (there is no basis for
 * deriving the budget of a surface that parts do not pass through from the part length). The review
 * is recorded in ADR 0109 addendum 2.
 *
 * The whole-file surface allocates a receive buffer of `ref.size` per file **before receiving**,
 * so pacing by count would make the RAM peak "concurrent count × the largest file at that moment"
 * (measured when `karume/4` shards were fetched through this surface: the first 4 anima turbo i4
 * files pre-allocated 2.503GiB at once — on an 8 GB target machine the whole browser goes down).
 * The budget caps the sum of the pre-allocations; on top of it come the retained finished files and
 * the temporary verification copy (Chrome's `crypto.subtle.digest` copies the whole input into
 * Blink), so it sits clearly below the single-ArrayBuffer limit (Chromium cuts off at 2,145,386,496
 * bytes).
 *
 * NOTE: that digest happens inside the source (verification of the network fetch) and cannot be
 * serialised from the common layer — with verification delegated, this budget alone decides how
 * many run at once. A cache hit needs only a string comparison of the recorded hash, so no digest
 * happens at all (this peak does not appear on the second and later starts).
 *
 * Not a public knob — the value has no use other than "lower it to be safe", and raising it brings
 * the peak above straight back. If a distribution does not fit, the constant is ruled on again.
 */
const BYTE_BUDGET = 1.5 * 1024 * 1024 * 1024;

/**
 * Whether this is a cancellation (the caller's `AbortSignal`). A cancellation is not a fetch
 * failure, so it passes through without being wrapped in a `HubFetchError` — wrapping it would
 * break the caller's ability to tell "I stopped it" from "it failed".
 *
 * MUST: do not look only at the DOMException the default `abort()` creates — with a custom reason,
 * `abort(reason)` makes `fetch` reject with **the reason itself**, so any value can be the
 * cancellation. The test is "the signal actually passed to the fetch is aborted, and the caught
 * value is identical to its reason" (for a combined signal, the combined one — the upstream reason
 * propagates as is).
 */
const isAborted = (error: unknown, signal?: AbortSignal): boolean =>
  (error instanceof DOMException && error.name === "AbortError") ||
  (signal?.aborted === true && error === signal.reason);

/**
 * The bytes the fetch layer returns must span the whole buffer. Three reasons, one per surface:
 *
 * - Full-download surface (`fetchAssets`): callers hand `bytes.buffer` to parsers as a full
 *   `ArrayBuffer` (compensating with slice would double the RAM peak).
 * - Range readers (`openAsset`, and the seek path of `openContainerSource`): per the MUST of
 *   `AssetRangeReader.read` (`source.ts`) they return a buffer of exactly the range — callers can
 *   use `bytes.buffer` as the range itself without copying, and with byteOffset 0, views with
 *   alignment requirements can be created directly.
 * - Container surface (the scan path of `openContainerSource`): blocks are sliced as views into
 *   this buffer, so the buffer being tight is the condition for "byteOffset = block.offset (a
 *   multiple of 64)" — the 4 B alignment a scale Float32Array view needs.
 *
 * A SharedArrayBuffer backing is rejected here (so that the `Uint8Array<ArrayBuffer>` the predicate
 * claims is not a lie at the type level either).
 */
const isTightView = (bytes: Uint8Array<ArrayBufferLike>): bytes is Uint8Array<ArrayBuffer> =>
  bytes.buffer instanceof ArrayBuffer &&
  bytes.byteOffset === 0 &&
  bytes.byteLength === bytes.buffer.byteLength;

export const assertTightView = (
  bytes: Uint8Array<ArrayBufferLike>,
  path: string,
): Uint8Array<ArrayBuffer> => {
  if (!isTightView(bytes)) {
    throw new Error(
      `hub: ${path} の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes;
};

/**
 * The single conversion point bytes → `Manifest` (called inside the source's verification hook).
 * `ownRepo` is the repo the manifest was loaded from (the source's naming `SourceOrigin.repo` —
 * `undefined` for a source without it).
 */
const decodeManifest = (
  bytes: Uint8Array<ArrayBufferLike>,
  ownRepo: string | undefined,
): Manifest => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new ManifestFormatError(`manifest: ${MANIFEST_FILENAME} が UTF-8 として読めない`, {
      cause: error,
    });
  }
  return parseRepoManifest(text, ownRepo);
};

/**
 * Fetches and parses `karume.json`. The generation is resolved here, once, and **the source
 * itself** is carried in the return value ({@link fetchAssets} and later fetch from that source and
 * that generation).
 *
 * The first argument is an HF repo reference ({@link HubRepoRef}) or a source itself (the opaque
 * handle returned by `localDirectory(...)` and the like). The former is shorthand for the HF
 * source.
 *
 * This is also the session's entry point, so the legacy namespaces (the `karume/1` family) are
 * reclaimed here, once (regardless of the source — copies left by older hub versions are unneeded
 * whatever source is in use now).
 *
 * NOTE: unlike assets, the manifest **cannot have an expected sha256 in advance** (it is the root
 * of the source of truth). So its key stays the SHA-pinned resolve URL, and `parse` = UTF-8
 * decoding + parsing remains the single conversion point bytes → `Manifest` (only the verification
 * of assets moved into the source).
 */
export const loadManifest = async (
  ref: HubRepoRef | DistributionSource,
  options: LoadManifestOptions = {},
): Promise<LoadedManifest> => {
  // A source is recognised by **identity** (`source.ts` — no brand field, no structural check).
  const isSource = isDistributionSource(ref);
  // MUST: check the spelling of an HF repo name before touching either the fetch or the reclaiming
  // of legacy namespaces — the fetch layer writes the repo into the URL without percent-encoding
  // (`/` is structural), so `?` `#` push `/resolve/<SHA>/...` into the query / fragment, and `../`
  // or three or more segments point at another path. The accepted set is the same single one as the
  // cross-repo reference check ({@link isRepoName}). The common layer assembles errors, so the
  // check lives here, not in the source adapter (`sources/hf.ts`).
  // A ref without a repo arriving from outside the types (a call from JS, …) would make the split
  // in `isRepoName` fail with a bare TypeError that never reaches users who branch on HubError, so
  // non-strings are refused by the same branch.
  if (!isSource && (typeof ref.repo !== "string" || !isRepoName(ref.repo))) {
    throw new HubError(
      `loadManifest: repo '${ref.repo}' が HF の 'owner/name' でない` +
        `（2 セグメントちょうど・許可文字 [A-Za-z0-9._-]・先頭ドット禁止）`,
    );
  }
  await purgeLegacyCaches(options);
  const driver = driverOf(isSource ? ref : createHfSource(ref));
  let generation: string;
  try {
    generation = await driver.resolveGeneration(options);
  } catch (error) {
    if (isAborted(error, options.signal)) throw error;
    throw revisionResolutionFailure(driver.origin, error);
  }
  // MUST: a cache hit on a SHA-pinned URL does not go to the network = the source's signal watch
  // has no effect, so check cancellation explicitly before and after the fetch (otherwise a call
  // with an already-aborted signal still returns the manifest, and a load that was cancelled
  // carries on).
  options.signal?.throwIfAborted();
  const pinned = driver.pin(generation, options);
  let manifest: Manifest | undefined;
  try {
    await pinned.readManifest({
      parse: (bytes) => {
        manifest = decodeManifest(bytes, pinned.origin.repo);
      },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      sizeViolation: manifestOversize(pinned.origin),
    });
  } catch (error) {
    if (error instanceof HubError || isAborted(error, options.signal)) throw error;
    throw manifestFetchFailure(pinned.origin, error);
  }
  // Check again right after the fetch (only synchronous assembly follows, so this is the last check
  // before returning).
  options.signal?.throwIfAborted();
  if (manifest === undefined) {
    throw new Error(`hub: ${MANIFEST_FILENAME} の検証フックが走っていない（取得層の不変条件破れ）`);
  }
  // **This value carries** the source (it is not rebuilt from the identity fields — a source with
  // no way to restore it could not be added otherwise).
  return new LoadedManifest(manifest, { driver, generation }, pinned.origin);
};

/**
 * Fetches a "key → file reference" table. Fetches and the progress total are **deduplicated by
 * {@link fileRefKey}**, and several keys pointing at the same file get the same bytes. **The caller
 * decides** the spelling of the keys — hub takes the table and returns it with the same keys,
 * without interpreting them.
 *
 * Verification (size / sha256) is delegated to the source — the manifest's `sha256` / `size` are
 * passed as expected values, so a network fetch is checked while receiving, and an entry that
 * passes gets the hash recorded. Later hits need only a string comparison with the record (zero
 * full-file hashes); an entry whose record disagrees, or which has no record and whose real hash
 * does not match, is evicted and fetched again (self-heal).
 */
export const fetchAssets = (
  loaded: LoadedManifest,
  files: Readonly<Record<string, FileRef>>,
  options: FetchAssetsOptions = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> =>
  fetchAssetsWithBudget(loaded, files, options, BYTE_BUDGET);

/**
 * The body of {@link fetchAssets}. Only the in-flight byte budget is taken as an argument.
 *
 * MUST: not exported from the barrel (`mod.ts`) — the budget is not a public knob (the note on
 * {@link BYTE_BUDGET}). This entry exists only for tests that exercise the pacing wiring (waiting
 * for the budget, no overtaking) with a budget of tens of bytes instead of GB-sized data.
 */
export const fetchAssetsWithBudget = async (
  loaded: LoadedManifest,
  files: Readonly<Record<string, FileRef>>,
  options: FetchAssetsOptions,
  byteBudget: number,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const source = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, source.origin);

  const keys = Object.keys(files);
  // MUST: deduplicate by {@link fileRefKey}, not by path — with cross-repo references, a same-named
  // path in another repo is different bytes, and folding by path would hand one's bytes to the
  // other.
  const unique = new Map<string, FileRef>();
  for (const key of keys) {
    const ref = files[key];
    const refKey = fileRefKey(ref);
    if (!unique.has(refKey)) unique.set(refKey, ref);
  }
  const targets = [...unique.values()];
  const progress = createProgressEmitter(unique, context.originOf, options.onProgress);

  const failure = new AbortController();
  const signal =
    options.signal === undefined
      ? failure.signal
      : AbortSignal.any([failure.signal, options.signal]);
  const bytesByRef = new Map<string, Uint8Array<ArrayBuffer>>();

  const fetchOne = async (ref: FileRef): Promise<void> => {
    // MUST: a cache hit does not go to the network = a stretch where the source's signal watch has
    // no effect, so check cancellation explicitly before and after the fetch (cancellation not
    // working while several GB are being read is the same as broken cancellation pass-through — the
    // same check as on the warm-up surface prefetchAssets).
    signal.throwIfAborted();
    let bytes: Uint8Array<ArrayBufferLike>;
    try {
      bytes = await sourceForRef(source, ref).readFile(ref, {
        signal,
        onProgress: (received) => progress.downloading(ref, received),
        sizeViolation: context.sizeViolation(ref),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, signal)) throw error;
      throw context.fetchFailure(ref, "取得", error);
    }
    // Check again right after the fetch — with only the earlier check, "cancelled in the middle of
    // a cache read" goes unobserved, and a file that was cancelled advances to complete.
    signal.throwIfAborted();
    bytesByRef.set(fileRefKey(ref), assertTightView(bytes, ref.path));
    progress.complete(ref);
  };

  // Dispatch is paced by "the in-flight sum of `ref.size`", not by count ({@link BYTE_BUDGET}).
  // Only this one loop waits, so head-of-line blocking over `targets` (= the order of the given
  // table) is the dispatch order — later small files never overtake, so the same manifest
  // dispatches in the same order.
  const admission = createByteAdmission(byteBudget);
  const running: Promise<void>[] = [];
  const run = async (ref: FileRef): Promise<void> => {
    try {
      await fetchOne(ref);
    } catch (error) {
      // If even one fails, stop the rest (fail loud — there is no point continuing downloads when
      // the whole call rejects).
      // MUST: do not let the reject escape here — settlement is awaited only after every dispatch,
      // so an escaped reject would become an unhandled rejection while dispatch is pending. The
      // real failure reason is held by the reason of `failure`.
      failure.abort(error);
    } finally {
      // Return the slots whatever the outcome (otherwise later fetches wait forever).
      admission.release(ref.size);
    }
  };
  for (const ref of targets) {
    // Start no new fetch after a failure or cancellation (stop before the check at the top of
    // `fetchOne`).
    if (signal.aborted) break;
    await admission.admit(ref.size);
    running.push(run(ref));
  }
  // MUST: even on failure, return only after every dispatched fetch has settled (an early reject
  // would leave fetches running in the background after the caller caught it, making abort
  // meaningless). `run` never rejects, so this always resolves.
  await Promise.all(running);
  // MUST: raise **the real first failure**, not a collateral one. Only the first `failure.abort()`
  // sets the reason, so `failure.signal.reason` is always the reason of the first fetch that
  // failed. Picking up a worker's reject directly would surface a collateral failure when it
  // settled first — and a collateral failure comes from the cancellation, so `isAborted` keeps it
  // from being wrapped in a `HubFetchError` (a raw AbortError), and in Chrome even the message is
  // replaced by a fixed text ("BodyStreamBuffer was aborted"), erasing the real cause.
  // A cancellation through the external signal the caller passed also puts its reason into
  // `failure` as is, so it still passes through (a cancellation does not turn into a fetch
  // failure).
  if (failure.signal.aborted) throw failure.signal.reason;
  // MUST: a start with every file cached never goes to the network, so check cancellation after
  // settlement too (only synchronous assembly follows, so this is the last check before returning).
  signal.throwIfAborted();

  // Build the table once. fromEntries makes even __proto__ an ordinary own property.
  return Object.fromEntries(
    keys.map((key): readonly [string, Uint8Array<ArrayBuffer>] => {
      const bytes = bytesByRef.get(fileRefKey(files[key]));
      if (bytes === undefined) {
        throw new Error(`hub: ${files[key].path} の bytes が揃っていない（取得層の不変条件破れ）`);
      }
      return [key, bytes];
    }),
  );
};

/**
 * Input checks and progress setup for phase 1. Done **before touching the source** (phase 1
 * downloads every ref given, so noticing a caller error after going to the network does not give
 * the bandwidth back).
 */
const preparePhase = (
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
): ProgressEmitter => {
  const available = context.available;
  if (refs.length === 0) {
    throw new ManifestReferenceError(
      `prefetchAssets: 取得対象が 1 つも無い（${context.session}）`,
      { available },
    );
  }
  // The duplicate-check table doubles as the per-file lookup for progress (`fileTotal`). Identity
  // is {@link fileRefKey} (a cross-repo reference counts a same-named path of another repo as a
  // separate file).
  const declared = new Map<string, FileRef>();
  for (const ref of refs) {
    const refKey = fileRefKey(ref);
    if (declared.has(refKey)) {
      throw new ManifestReferenceError(
        `prefetchAssets: 参照 '${refKey}' が重複している（この面は渡された列をそのまま扱う —` +
          ` 同じ参照を 2 回渡すのは呼び出し側の誤り。全量面 fetchAssets は一意化する）`,
        { available },
      );
    }
    declared.set(refKey, ref);
  }
  return createProgressEmitter(declared, context.originOf, options.onProgress);
};

/**
 * **Phase 1**, which puts every ref into "a state where a later full read is cheap, without holding
 * it in RAM" (the body of {@link prefetchAssets}).
 *
 * MUST: check the phase-1 capability **per ref** (`source.ts` (4) is an optional per-source
 * capability, and a cross-repo reference comes from a source other than the session's). Deciding
 * by the session's source alone would, in the valid setup of a local session + an HF cross-repo
 * target, skip the warm-up of the cross-repo files entirely, and the real fetch would run silently
 * inside a later read (the re-warm of `openFile`) with neither progress nor `signal` reaching it.
 */
const runPrefetchPhase = async (
  source: PinnedSource,
  context: FetchContext,
  refs: readonly FileRef[],
  options: FetchAssetsOptions,
  progress: ProgressEmitter,
): Promise<void> => {
  const failure = new AbortController();
  const prefetchSignal =
    options.signal === undefined
      ? failure.signal
      : AbortSignal.any([failure.signal, options.signal]);

  const prefetchOne = async (ref: FileRef): Promise<void> => {
    // MUST: an entry whose recorded hash matches does not go to the network = a stretch where the
    // source's signal watch has no effect, so check cancellation explicitly per file (otherwise
    // neither cancellation nor the first failure takes effect for already-warm files, and the call
    // settles only after running through every remaining ref). Spelt the same as fetchOne of the
    // whole-file surface.
    prefetchSignal.throwIfAborted();
    let origin: PinnedSource;
    try {
      origin = sourceForRef(source, ref);
    } catch (error) {
      // An unmapped cross-repo target (the bare Error of `sources/local.ts`) is missing
      // configuration on the caller's side, so make it the same failure with context as on the
      // whole-file surface (the surfaces do not differ in how it looks).
      if (error instanceof HubError) throw error;
      throw context.fetchFailure(ref, "事前取得", error);
    }
    const prefetchFile = origin.prefetchFile;
    if (prefetchFile === undefined) {
      // The session having phase 1 while only the cross-repo target lacks it breaks the source
      // contract (`originFor` returns other coordinates of the same source; it is not a way to drop
      // capabilities).
      if (source.prefetchFile !== undefined) {
        throw new Error(
          `hub: 越境先の取得元が相 1 を持たない（${fileRefKey(ref)} — 取得元契約の不変条件破れ）`,
        );
      }
      // A ref of a source without phase 1 (a local directory) is not warmed; only one complete is
      // emitted (ADR 0086 addendum 2026-10-11) — for a source that reads directly, "a later read is
      // cheap" holds from the start, so this ref's warm-up is done. Without it, the final value
      // would fall short of the total by this ref, which the caller put in the progress
      // denominator, and the rule would split when sources are mixed.
      progress.complete(ref);
      return;
    }
    try {
      await prefetchFile(ref, {
        signal: prefetchSignal,
        onProgress: (received) => progress.downloading(ref, received),
        sizeViolation: context.sizeViolation(ref),
      });
    } catch (error) {
      if (error instanceof HubError || isAborted(error, prefetchSignal)) throw error;
      // MUST: do not judge cancellation by the identity of the caught value alone — phase 1 is a
      // surface that does not hold the bytes, so an interrupted transfer also shows up as a put
      // reject, wrapped and buried in `cause`. If the signal is aborted, pass its reason through
      // (for a collateral failure, the first failure itself).
      // NOTE: the shape that passes the `isAborted` above first (a raw AbortError) never gets here,
      //       so this alone does not restore the real cause. The final outcome is taken from the
      //       reason after phase 1's allSettled.
      if (prefetchSignal.aborted) throw prefetchSignal.reason;
      throw context.fetchFailure(ref, "事前取得", error);
    }
    // Check again right after the warm-up (the same before-and-after checks as fetchOne of the
    // whole-file surface) — with only the earlier check, files already in flight before the
    // cancellation (at most {@link CONCURRENCY}) would emit complete after it, and a cancelled
    // load's progress would appear to advance.
    prefetchSignal.throwIfAborted();
    // This surface emits `complete` (this is the only end — the contract of `AssetPhase`).
    progress.complete(ref);
  };

  let next = 0;
  const worker = async (): Promise<void> => {
    try {
      while (next < refs.length) await prefetchOne(refs[next++]);
    } catch (error) {
      // If even one fails, stop the rest (fail loud — there is no point continuing downloads when
      // the whole call rejects).
      failure.abort(error);
      throw error;
    }
  };
  // MUST: even on failure, return only after every worker has settled (the same reason as on the
  // whole-file surface — an early reject would leave fetches running in the background after the
  // caller caught it).
  await Promise.allSettled(
    Array.from({ length: Math.min(CONCURRENCY, refs.length) }, () => worker()),
  );
  // MUST: raise **the real first failure**, not a collateral one (the same reason as on the
  // whole-file surface). Picking up worker rejects in array order would surface a collateral
  // failure when the real culprit was not worker[0].
  if (failure.signal.aborted) throw failure.signal.reason;
};

/**
 * A surface that **only warms assets ahead of time** (phase 1). It downloads the weights first —
 * later {@link fetchAssets} calls and the container surface (`openContainerSource`) do not go to
 * the network.
 *
 * Use it "at load time of a pipeline that builds its Session lazily" — if building waits until the
 * first run, the weight download waits too and does not show in the load progress. Downloading here
 * first makes the progress complete during the load, and building needs only cache reads.
 *
 * {@link CONCURRENCY} at a time, `sha256` checked in transit, a failure restores the real cause as
 * a `HubFetchError`. No bytes are returned (this surface keeps nothing in RAM; to get them, read
 * again with {@link fetchAssets} or a range read ({@link openAsset}) — it will be a cache hit).
 *
 * Progress emits `downloading`* followed by **one `complete` per file** (this surface is the end —
 * the contract of `AssetPhase`; a cached file emits only the single `complete`). Refs that need no
 * fetch (next paragraph) are no exception — every ref passed in emits one `complete`, so if the
 * caller uses the sum of the sizes of the refs it passed as the denominator, the final `loaded`
 * reaches it even with mixed sources (ADR 0086 addendum 2026-10-11).
 *
 * **Refs of a source without phase 1 (a local directory) get the input checks only and are not
 * warmed** — their progress is the single `complete` (the same shape as a cached file). The
 * decision is **per ref**, so with a local session + an HF cross-repo target only the cross-repo
 * files are warmed.
 * It does not fail loudly: this surface promises "a state where later reads are cheap", which for a
 * source that reads directly **holds from the start** (having no cache to warm is not a failure).
 * Failing would force apps, which should be able to swap sources, to branch per source.
 *
 * NOTE: with the HF source, an environment without `caches` and a failed cache write (quota
 * exceeded, …) **fail loudly** (this surface does not hold the bytes, so there is no room to fall
 * back to a plain fetch).
 */
export const prefetchAssets = async (
  loaded: LoadedManifest,
  refs: readonly FileRef[],
  options: FetchAssetsOptions = {},
): Promise<void> => {
  const source = pinnedSourceOf(loaded, options);
  const context = createFetchContext(loaded, source.origin);
  const progress = preparePhase(context, refs, options);
  await runPrefetchPhase(source, context, refs, options, progress);
  // MUST: a call with every file cached never goes to the network = the source's signal watch has
  // no effect, so check cancellation after settlement too (the last check before returning).
  options.signal?.throwIfAborted();
};

/**
 * Opens the **range reader** of one asset (`source.ts` (8)). Where the whole-file surface
 * {@link fetchAssets} works in units of "one whole file of the declared size", this surface pulls
 * only `[offset, offset + length)` out of the same file — a surface for consumers that need only
 * rows of a few KB out of a table of hundreds of MiB.
 *
 * **`undefined` when the source lacks that capability** (a local source given a directory adapter
 * without positional reads has that shape). It does not fail loudly for the same reason as phase 1
 * ({@link prefetchAssets}): lacking it is a difference in capability, not a failure — the caller
 * sees `undefined` and falls back to a full read (one branch is enough).
 *
 * The reader names its **cost class** ({@link AssetRangeReader.cost}). Callers use it to change
 * "the maximum row count for which switching to row reads pays off" — with a `"scan"` source each
 * row carries skipping proportional to the offset, so as the row count grows one full read becomes
 * cheaper.
 *
 * The ref's source is resolved as on the other surfaces (a cross-repo reference uses the source of
 * the declared (repo, revision)). **What opening touches depends on the source** (`source.ts` (8))
 * — the local source only creates the reader and writes neither to the network nor to a cache. The
 * HF source, only for a ref not in the inventory, inserts the same warm-up as phase 1 (full
 * download → persistent cache) exactly once before opening (this call's `signal` cancels that
 * warm-up). No byte fetch other than warming happens, and the range's bytes are touched only when
 * {@link AssetRangeReader.read} is called. The fetch context (re-wrapping into the {@link HubError}
 * family) is not applied: this surface is not a fetch, so failures surface as the source's bare
 * `Error`.
 *
 * This surface checks **only the bounds of the declared `size`**; neither the whole-file size
 * check (`sizeViolation` → `IntegrityError`) nor the sha256 comparison applies (reading only a
 * range, there is no whole file at hand to compare with the declaration). Corruption of the file
 * shows up as the values of the rows read.
 *
 * As on the other surfaces, the per-surface options (`fetch` / `caches` / `headers` /
 * `onCacheError` / `onRetry`) apply **only as passed to this call** (ADR 0086 decision 1 — a source
 * takes its options per `pin`, not at creation). None of them applies to the local source (they are
 * vocabulary of the HTTP source only).
 */
export const openAsset = async (
  loaded: LoadedManifest,
  ref: FileRef,
  options: LoadManifestOptions = {},
): Promise<AssetRangeReader | undefined> => {
  // Check cancellation before checking whether a reader exists (so source capabilities do not
  // change how cancellation looks — a convention of the common layer).
  options.signal?.throwIfAborted();
  const source = sourceForRef(pinnedSourceOf(loaded, options), ref);
  const openFile = source.openFile;
  if (openFile === undefined) return undefined;
  const reader = await openFile(ref, {
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  // MUST: check again right after opening — opening a warm reader does not go to the network = a
  // stretch where the source's signal watch has no effect, so otherwise a cancellation during the
  // open goes unobserved and a reader that was cancelled is returned (the same before-and-after
  // checks as loadManifest / prefetchAssets). The design of not carrying the open-time signal into
  // the reader (the MUST NOT of `source.ts` (8)) is unchanged — this checks only the settlement of
  // the one open.
  options.signal?.throwIfAborted();
  // The tight view check is also a convention of the common layer (the MUST in `source.ts`) —
  // placing it per source would mean rewriting the same invariant for every new source.
  return {
    cost: reader.cost,
    read: async (offset, length, readOptions) => {
      // MUST: check the bounds of the declared size here too (this surface's promise — the doc
      // above). Left to the source, the HF fetch layer does not use the declared size as the read
      // bound, so a ref declaring a shorter size for the same sha256 could, on opening a warm
      // entry, read past the declaration (the rest of the file), diverging from the local source,
      // which rejects by the declaration.
      if (
        !Number.isInteger(offset) ||
        !Number.isInteger(length) ||
        offset < 0 ||
        length < 0 ||
        offset + length > ref.size
      ) {
        throw new Error(
          `hub: ${ref.path} の区間 [${offset}, ${offset + length}) が不正` +
            `（宣言 size ${ref.size} — 非負整数で size に収まる区間だけを読める）`,
        );
      }
      return assertTightView(await reader.read(offset, length, readOptions), ref.path);
    },
  };
};
