/**
 * The Hugging Face source adapter (ADR 0038 §5, "hub's fetch layer"). Built on `@hdae/fetch-cache`
 * (zero run-time dependencies, Web standard APIs only). One implementation of the source contract
 * `source.ts` (the other is `local.ts` — only this one has generations, a cache and the network).
 *
 * What this adapter holds (= what is specific to HF / HTTP + a persistent cache):
 * - resolving a mutable ref → commit SHA, and the pinning guidance for an implicit `main`
 * - ownership of the cache (since 0.5.0 there is a single, internally fixed namespace). The asset
 *   key is the content key `["hf", kind, repo, path, sha256]`, so a file whose bytes do not change
 *   stays a hit even when the revision moves. Only the manifest has no expected sha in advance, so
 *   its key is the SHA-pinned resolve URL.
 * - **asset integrity verification** (the spec's `sha256` / `expectedBytes` are delegated to the
 *   fetch layer). It verifies on fetch and records the hash in the entry; later hits need only a
 *   string comparison with the record (zero full-file hashes). An entry whose record disagrees is
 *   evicted and fetched again automatically (self-heal). The same `expectedBytes` also caps the
 *   bytes received — the fetch layer stops receiving once the declaration is exceeded (the fetch
 *   layer's ADR 0011).
 * - phase 1 (streaming prefetch) — an HTTP-specific optimisation that **writes to the persistent
 *   cache without holding the bytes in RAM**.
 * - querying and deleting the cache inventory (`source.ts` (6) (7)) — this source is the one that
 *   keeps things, so only it can answer "what is here" and "can this be deleted".
 * - range reads (`source.ts` (8)) — opens only `[offset, offset + length)` inside a stored entry
 *   (`openHfFile` of fetch layer 0.8.0). Only the cache contents can be read, so a ref not fetched
 *   yet gets the same warm-up as phase 1 once and is then reopened.
 *
 * MUST NOT: assemble errors here (the common layer holds the diagnostic context — `context.ts`).
 * The only exception is a bare `Error` that reports "an invariant of the fetch layer is broken".
 */

import { evict as evictKey, listKeys } from "@hdae/fetch-cache";
import {
  fetchHfFile,
  isCommitSha,
  openHfFile,
  prefetchHfFile,
  resolveHfRevision,
} from "@hdae/fetch-cache/hf";
import { type FileRef, fileRefKey, MANIFEST_FILENAME } from "../manifest.ts";
import type { HubRepoRef, LoadManifestOptions } from "../session.ts";
import {
  type AssetRangeReader,
  DistributionSource,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../source.ts";

/**
 * One coordinate on HF (generation resolved). `hubUrl` is **the choice of host**, so it applies to
 * cross-repo targets too.
 */
type HfTarget = { readonly repo: string; readonly revision: string; readonly hubUrl?: string };

const hfTarget = (repo: string, revision: string, hubUrl?: string): HfTarget => ({
  repo,
  revision,
  ...(hubUrl === undefined ? {} : { hubUrl }),
});

const requestInit = (headers?: HeadersInit, signal?: AbortSignal): RequestInit => ({
  ...(headers === undefined ? {} : { headers }),
  ...(signal === undefined ? {} : { signal }),
});

/**
 * Prints the guidance once when `main` was resolved implicitly because no revision was passed.
 *
 * Only when the caller wrote no revision; not for an **explicit** `"main"`, a tag or a SHA (an
 * explicit value is the caller saying "a mutable ref is fine", and a warning would be a false
 * positive). The resolved SHA is printed so that copying that one line completes the pin.
 */
const warnImplicitMain = (repo: string, revisionSha: string): void => {
  console.warn(
    `@karume/hub: resolved 'main' because no revision was given (repo ${repo} → ${revisionSha}).\n` +
      `main can be moved, so the same code may read different weights on the next start. Pin it in one of these ways:\n` +
      `  ① pass revision: "${revisionSha}" (copying this one line completes the pin)\n` +
      `  ② use the SOURCES table of the family's @karume/models subpath (e.g. SOURCES["sbv2-jvnv"] from @karume/models/sbv2), or KARUME_SOURCES (the tables of published distribution repos — pins verified by the package)`,
  );
};

/**
 * The cache key of one asset (the **content key** of the fetch layer's HF layer). hub does not pass
 * `kind`, so the second element is always `"model"`.
 *
 * MUST: keep the spelling in this one place — if the inventory query (prefix `["hf", "model",
 * repo]`) and deletion (the full five-element key) drift from the same expression, entries believed
 * deleted remain.
 */
const contentKey = (repo: string, ref: FileRef): readonly string[] => [
  "hf",
  "model",
  repo,
  ref.path,
  ref.sha256,
];

/**
 * The inventory matching key. The repo is already narrowed by the prefix, so the (path, sha256)
 * pair suffices.
 */
const stockKey = (path: string, sha256: string): string => `${path} ${sha256}`;

/**
 * Read strategy for the fetch layer. Its own default is "stream" only on Deno (a Blob loads the
 * whole body into memory there); Node's Blob is in-memory as well, so Node gets "stream" too, and
 * browsers keep the layer's default ("blob": a lazy slice of a disk-backed Blob). The runtime is
 * probed through optional globals only, not through any runtime API (ADR 0002).
 */
const defaultReadStrategy = (): "stream" | undefined => {
  const host = globalThis as {
    readonly Deno?: unknown;
    readonly process?: { readonly versions?: { readonly node?: unknown } };
  };
  return host.Deno !== undefined || typeof host.process?.versions?.node === "string"
    ? "stream"
    : undefined;
};

const pinnedHfSource = (
  repo: string,
  generation: string,
  hubUrl: string | undefined,
  options: LoadManifestOptions,
): PinnedSource => {
  const target = hfTarget(repo, generation, hubUrl);
  const baseFetch = options.fetch ?? globalThis.fetch;
  const shared = {
    ...(options.caches === undefined ? {} : { caches: options.caches }),
  };
  // The naming for diagnostics (HF is a source with a repo and a commit SHA — integrity checks are
  // on the network-fetch side).
  const origin: SourceOrigin = {
    label: `repo ${repo} @ ${generation}`,
    integrity: "network",
    repo,
    revisionSha: generation,
  };
  /**
   * The body of phase 1 (shared by (4) `prefetchFile` and the re-warm of (8) `openFile`). A surface
   * that only writes to the persistent cache without holding the bytes, where `expectedBytes` acts
   * as **the receive cap** (the fetch layer's ADR 0011) — it stops once the declaration is
   * exceeded, so the cap holds on this surface too.
   */
  const warmFile = async (
    ref: FileRef,
    {
      signal,
      onProgress,
    }: {
      readonly signal?: AbortSignal;
      readonly onProgress?: (loaded: number) => void;
    },
  ): Promise<void> => {
    await prefetchHfFile(
      target,
      { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size },
      {
        init: requestInit(options.headers, signal),
        fetch: baseFetch,
        ...(onProgress === undefined
          ? {}
          : { onProgress: (progress) => onProgress(progress.loaded) }),
        ...shared,
        ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
      },
    );
  };
  return {
    origin,

    readManifest: async ({ parse, signal }) => {
      // The manifest is the root of the source of truth, so it **can have neither an expected
      // sha256 nor an expected byte count in advance**. The fetch layer has no "cap without exact
      // match", so the 1 MiB limit is checked by `parse` (`parseManifest`) after receiving
      // everything (there is no check here that stops receiving midway).
      // MUST: UTF-8 decoding and parsing happen inside the fetch layer's `validate` hook — doing
      // them outside the fetch would leave a corrupt cache entry unevicted, returning the same
      // ManifestFormatError every time until `clearHubCache` is called by hand (this aligns with
      // the same self-heal path as assets).
      // MUST NOT: mix a cancellation check into this hook — a throw from the hook is read as
      // "corrupt" by the lower layer and leads to evicting and refetching a healthy cache entry.
      await fetchHfFile(
        target,
        { path: MANIFEST_FILENAME, validate: parse },
        {
          init: requestInit(options.headers, signal),
          fetch: baseFetch,
          ...shared,
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
          ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
        },
      );
    },

    readFile: async (ref, { signal, onProgress }) => {
      return await fetchHfFile(
        target,
        {
          path: ref.path,
          // Verification belongs to the fetch layer (hashing while receiving / comparing the
          // recorded hash / self-heal on mismatch).
          sha256: ref.sha256,
          // The byte-count check (receive cap + exact match after the full receive), and at the
          // same time the pre-allocation size of the receive buffer. A size whose allocation itself
          // fails (above Chromium's single-ArrayBuffer limit) throws before receiving, so it never
          // fails only after several GB have been downloaded.
          expectedBytes: ref.size,
        },
        {
          init: requestInit(options.headers, signal),
          fetch: baseFetch,
          // Fires only on the network side (a cache hit advances with the single complete).
          // The fetch layer's receive cap (`expectedBytes`) guarantees `loaded` never exceeds
          // `size`.
          onProgress: (progress) => onProgress(progress.loaded),
          ...shared,
          ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
          ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
        },
      );
    },

    // Phase 1: a surface that does not hold the bytes. An existing entry is handled by comparing
    // the recorded hash (fetch layer 0.5.0) — if the record matches the expected sha256 it is kept
    // as is without going to the network; an entry without a record or with a mismatching one is
    // re-warmed with verification. A missing `caches` or a failed put fails loudly (there is no
    // room to fall back to a plain fetch — a fallback would break the RAM-peak target).
    prefetchFile: async (ref, { signal, onProgress }) => {
      await warmFile(ref, { ...(signal === undefined ? {} : { signal }), onProgress });
    },

    // (8) Range reads. Only the cache contents are opened, and `openHfFile` **does not go to the
    // network** (nor resolve a revision), so not fetched, no record, and a mismatching record
    // (which the opening side deletes by self-heal) all come back as `undefined`. So it
    // inserts **the same warm-up as phase 1** exactly once and reopens — so that callers do not
    // carry the step "go through prefetchAssets first" (consumers only need to request rows).
    openFile: async (ref, { signal }): Promise<AssetRangeReader> => {
      const spec = { path: ref.path, sha256: ref.sha256, expectedBytes: ref.size };
      const strategy = defaultReadStrategy();
      const openOptions = {
        ...shared,
        ...(strategy === undefined ? {} : { strategy }),
        ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      };
      let opened = await openHfFile(target, spec, openOptions);
      if (opened === undefined) {
        await warmFile(ref, { ...(signal === undefined ? {} : { signal }) });
        opened = await openHfFile(target, spec, openOptions);
      }
      if (opened === undefined) {
        // Failing to open right after a successful warm-up is either a broken invariant of the
        // fetch layer (phase 1 records the hash with verification, so after it the same content key
        // should open) or the inventory being deleted between the warm-up and the reopen (a race
        // with inventory deletion (7)), so fail naming every candidate.
        throw new Error(
          `@karume/hub: the range reader for ${ref.path} does not open even right after re-warming` +
            ` (possible causes: a cache that does not keep the recorded hash / CacheStorage is unavailable / the inventory was` +
            ` deleted between warming and reopening <a race with evictCachedAssets or clearHubCache>)`,
        );
      }
      const entry = opened;
      return {
        // The cost class is the fetch layer's read strategy itself: "blob" is a slice of a lazy
        // Blob (independent of the offset), "stream" skips through the body (proportional to the
        // offset). Deno and Node use "stream" (`blob()` puts the whole body on the heap there —
        // `defaultReadStrategy`), so "scan"; browsers get "seek".
        cost: entry.strategy === "blob" ? "seek" : "scan",
        read: async (offset, length, readOptions) => {
          const bytes = await entry.read(offset, length, readOptions);
          const { buffer } = bytes;
          // The fetch layer's return type does not carry the buffer kind
          // (`Uint8Array<ArrayBufferLike>`). Consumers read the returned bytes as a TypedArray as
          // is, so a SharedArrayBuffer is not let through silently (squashing it with `as` would
          // make a view over shared memory look sound by type alone).
          if (!(buffer instanceof ArrayBuffer)) {
            throw new Error(
              `@karume/hub: the range read of ${ref.path} returned a buffer that is not an ArrayBuffer`,
            );
          }
          // The length check (exactly `length`?) and the tight view check belong to the fetch layer
          // and the common layer, so they are not repeated here.
          return new Uint8Array(buffer, bytes.byteOffset, bytes.byteLength);
        },
      };
    },

    // Cross-repo targets use the same adapter (the target must be pinned to a commit SHA, so no
    // resolution happens on the cross-repo side). The asset cache key is the content key (without
    // `hubUrl`), so identical bytes share one entry even across mirrors. A same-named path in
    // another repo is a separate entry, since the key includes `repo`.
    originFor: (crossRepo, crossRevision) =>
      pinnedHfSource(crossRepo, crossRevision, hubUrl, options),

    // (6) Inventory. **listKeys is called once with the repo prefix** — calling it per ref would
    // enumerate the whole cache once per ref (a difference that matters for a manifest with dozens
    // of components).
    inventory: async (refs) => {
      const keys = await listKeys(["hf", "model", repo], shared);
      // A content key always has five elements (`contentKey`). Any other shape was not written by
      // this source.
      const stock = new Set(
        keys.filter((key) => key.length === 5).map((key) => stockKey(`${key[3]}`, `${key[4]}`)),
      );
      return new Set(
        refs.filter((ref) => stock.has(stockKey(ref.path, ref.sha256))).map(fileRefKey),
      );
    },

    // (7) Deletion. The semantics are by prefix, but the full key is passed, so the target is that
    // one entry only (count > 0 = it existed and was deleted).
    evict: async (refs) => {
      const removed: FileRef[] = [];
      for (const ref of refs) {
        const count = await evictKey(contentKey(repo, ref), shared);
        if (count > 0) removed.push(ref);
      }
      return removed;
    },
  };
};

/**
 * Creates an HF source. `ref.revision` may be a mutable ref ({@link SourceDriver.resolveGeneration}
 * resolves it to a commit SHA).
 *
 * The fetch options (`fetch` / `caches` / `headers`) come **per surface** (the arguments of `pin`)
 * — this factory holds only "where to fetch from" (repo / requested ref / mirror).
 */
export const createHfSource = (ref: HubRepoRef): DistributionSource => {
  // MUST: copy the input object's values once at the entry and never read them again — otherwise,
  // if the caller modified the same object after the load (or while revision resolution is
  // pending), the naming (`loaded.repo`) would keep the old value while fetches go to another repo
  // / host with the pinned SHA.
  const { repo, hubUrl, revision } = ref;
  const requested = revision ?? "main";
  const driver: SourceDriver = {
    // The naming before generation resolution. When resolution fails, the diagnostic must say "what
    // was looked up where" with this alone, so it includes the requested ref (still mutable).
    origin: { label: `repo ${repo} @ ${requested}`, integrity: "network", repo },

    resolveGeneration: async (options) => {
      if (isCommitSha(requested)) return requested;
      const revisionSha = await resolveHfRevision(hfTarget(repo, requested, hubUrl), {
        ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
        init: requestInit(options.headers, options.signal),
        ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
      });
      // MUST: check the response as a commit SHA (the same 40-lowercase-hex rule as the `revision`
      // of a cross-repo reference) — the fetch layer checks only for a non-empty string, so if a
      // non-conforming mirror or a custom fetch returned `main` or a short form, it would be
      // resolved again on every later fetch as "the pinned generation", and the contract of pinning
      // to one commit would silently break. The common layer wraps the bare Error with context as a
      // generation-resolution failure.
      if (!isCommitSha(revisionSha)) {
        throw new Error(
          `@karume/hub: the resolve response for revision '${requested}' is not a 40-digit lowercase hex commit SHA` +
            ` (response '${revisionSha}' — suspect a non-conforming mirror or a replaced fetch)`,
        );
      }
      // Emit it **after** resolution — this is where the SHA to print is settled, and when
      // resolution fails, the failure itself should be reported, not a warning (fail loudly comes
      // first).
      if (revision === undefined) warnImplicitMain(repo, revisionSha);
      return revisionSha;
    },

    pin: (generation, options) => pinnedHfSource(repo, generation, hubUrl, options),
  };
  return new DistributionSource(driver);
};
