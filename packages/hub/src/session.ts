/**
 * The session vocabulary — "which source, which generation, read with which call options". Only the
 * types handed between surfaces live here; they belong neither to the source implementations
 * (`sources/`) nor to the common layer (`fetch.ts`).
 *
 * NOTE: `fetch` / `caches` / `headers` / `onRetry` are the HTTP source's vocabulary exposed as is
 * on the public surface. Sources that read directly (a local directory, …) **ignore them** (they do
 * not go through `caches` and do not call `fetch`). The options common to all sources are `signal`
 * / `onProgress` / `onCacheError`; per-source settings are taken by the factory that creates the
 * source (`localDirectory(...)`, …).
 */

import type { Manifest } from "./manifest.ts";
import type { AssetProgress } from "./progress.ts";
import type { PinnedSource, SourceDriver, SourceOrigin } from "./source.ts";

/**
 * The repository to fetch from. `hubUrl` applies **only when the app sets it explicitly** (it never
 * comes from a manifest).
 */
export type HubRepoRef = {
  /** `"owner/name"`. */
  readonly repo: string;
  /** Branch / tag / commit SHA. Default `"main"`. Passing a SHA issues no resolution request. */
  readonly revision?: string;
  /** For mirrors. Default `https://huggingface.co`. */
  readonly hubUrl?: string;
};

/**
 * Diagnostic for a cache I/O failure (quota exceeded, …). **Delivered to the app** instead of left
 * to console.warn.
 */
export type CacheDiagnostic = {
  readonly op: "open" | "match" | "put" | "delete";
  readonly url: string;
  readonly error: unknown;
};

/**
 * Notification for each retry the fetch layer makes on 429 / 503 following `Retry-After` (sent
 * before the wait).
 */
export type RetryDiagnostic = {
  readonly url: string;
  readonly status: number;
  /** Which retry this is (1-based). */
  readonly attempt: number;
  /** The ms about to be waited. */
  readonly delayMs: number;
  /** The response's `Retry-After` header (undefined if absent). */
  readonly retryAfter?: string;
};

export type LoadManifestOptions = {
  readonly signal?: AbortSignal;
  /**
   * `Authorization` and the like. Passed through as is to the fetches (revision resolution, files).
   *
   * NOTE: the cache is not split by credential (by design) — headers are not part of the key, so
   * bytes fetched with authentication are also hit by later unauthenticated calls. With no plans to
   * operate gated assets, isolation was excessive defence (details in `docs/limitations.md`).
   */
  readonly headers?: HeadersInit;
  /**
   * Receiver of cache I/O failures. Without it, "a full re-download on every start silently becomes
   * the norm", so apps take it and use it, for example, to suggest `navigator.storage.persist()`.
   */
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /**
   * Receiver of each retry. Emitted so that an app can show "rate limited, waiting N more seconds"
   * (instead of appearing stuck in silence for tens of seconds).
   *
   * Called **only by the HTTP source** (the local source does not go through the fetch layer, so it
   * never calls it). It covers every fetch of that surface — revision resolution, `karume.json`,
   * assets (the whole-file and the warm-up surfaces) and cross-repo files. The retry policy
   * (default 429 / 503, at most 5 times, `Retry-After` first) stays the fetch layer's default; hub
   * does not change it.
   *
   * NOTE: a throw from this listener does not fail the fetch (the fetch layer isolates it and
   * warns).
   */
  readonly onRetry?: (diagnostic: RetryDiagnostic) => void;
  /** Replacement for `fetch` (for tests and custom transports). */
  readonly fetch?: typeof globalThis.fetch;
  /** Replacement for `CacheStorage` (for tests). */
  readonly caches?: CacheStorage;
};

/** A resolved session (the source implementation + the generation identifier pinned there). */
type SessionSource = {
  readonly driver: SourceDriver;
  /**
   * The return value of {@link SourceDriver.resolveGeneration} (the empty string for a source
   * without generations).
   */
  readonly generation: string;
};

/**
 * manifest → its source. It lives in a module-level table instead of a private field (`#session`)
 * for the same reason as `drivers` in `source.ts` (a `static` block that reads the field out runs
 * at import time, and bundlers then keep even unused classes). The key is the value itself (a weak
 * reference), invisible from outside the module.
 */
const sessions = new WeakMap<LoadedManifest, SessionSource>();

/**
 * A manifest pinned to a resolved generation. Every later fetch uses this generation.
 *
 * **It carries the source itself as an internal field** — the surfaces that need a source
 * (`fetchAssets` / `prefetchAssets` / `openAsset` / `openContainerSource` / the inventory surfaces
 * `listCachedAssets` / `evictCachedAssets`), that is, every surface that goes through
 * {@link pinnedSourceOf}, open the source this value holds instead of rebuilding a source from the
 * identity fields.
 * Rebuilding would make every surface lean on the HF-specific property "for HF it can be restored
 * from repo and SHA", and a source with no way to restore it (a local directory) could not be
 * added.
 *
 * The public identity fields ({@link repo} / {@link revisionSha}) are **information for diagnostics
 * and display**, never used to reconstruct a fetch. For a source that lacks them (a local
 * distribution with neither generations nor a repo) the fields do not appear at all — no synthetic
 * repo or fake SHA is named (that would produce diagnostics pointing at a repo that does not
 * exist).
 *
 * MUST: this type cannot be constructed outside hub (`mod.ts` exports only the type). The source is
 * closed inside the module-level table (`sessions`); a rebuilt value loses its source altogether
 * ({@link pinnedSourceOf} refuses it with a fail loudly).
 */
export class LoadedManifest {
  /** Only when the source has the notion of a repo (HF: `"owner/name"`). */
  readonly repo?: string;
  /** The resolved commit SHA (40 digits). Only for sources with the notion of generations. */
  readonly revisionSha?: string;
  readonly manifest: Manifest;
  /**
   * A type-only nominal brand (absent at run time). A class with only public fields is a structural
   * type, and a hand-built value such as `{ manifest }` would type-check as a `LoadedManifest`.
   */
  declare private readonly brand: never;

  constructor(manifest: Manifest, session: SessionSource, origin: SourceOrigin) {
    this.manifest = manifest;
    sessions.set(this, session);
    if (origin.repo !== undefined) this.repo = origin.repo;
    if (origin.revisionSha !== undefined) this.revisionSha = origin.revisionSha;
  }
}

/**
 * Opens the session's source **with the options of that call**. `fetch` / `caches` / `headers` can
 * be replaced per surface because the source takes its options here, not at creation
 * ({@link SourceDriver.pin} in `source.ts`).
 */
export const pinnedSourceOf = (
  loaded: LoadedManifest,
  options: LoadManifestOptions,
): PinnedSource => {
  const session = sessions.get(loaded);
  if (session === undefined) {
    throw new Error(
      "hub: LoadedManifest に取得元が結び付いていない（loadManifest の返り値を作り替えた値は渡せない）",
    );
  }
  return session.driver.pin(session.generation, options);
};

export type FetchAssetsOptions = LoadManifestOptions & {
  /**
   * Receiver of per-asset progress.
   *
   * NOTE: a throw from this listener does not fail the fetch (hub isolates it and warns for a
   * notification of any phase — progress is incidental information, not a correctness requirement).
   */
  readonly onProgress?: (progress: AssetProgress) => void;
};
