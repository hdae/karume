/**
 * hub's error types (ADR 0038 §5, "error shape"; the vocabulary updated to v2 by ADR 0041).
 *
 * The classification axis is **where the user branches to**:
 * - {@link ManifestFormatError} — JSON / structure / size (the manifest is broken as a shape)
 * - {@link ManifestReferenceError} — references and vocabulary (defaultModel / defaultQuant, the
 *   weights mapping, unknown keys)
 * - {@link ManifestPathError} — path allowlist violation (the breakwater against traversal outside
 *   the SHA pin)
 * - {@link IntegrityError} — the byte count of a file read by the local source disagrees with the
 *   manifest's `size` (a size / sha256 mismatch from the HF source is detected by the fetch layer
 *   and carried in the `cause` of a {@link HubFetchError})
 * - {@link HubFetchError} — pass-through with context of failures from the fetch layer (404, auth,
 *   revision resolution failure)
 *
 * MUST: every error carries **the list of available model / quant / dtype labels**
 * ({@link AvailableLabels}) — it replaces the information GGUF users get from the quant table in a
 * README, and returning "then what does work" as primary information on failure is part of why the
 * manifest was introduced (ADR 0038 §5 / 0041 decision 8). It is empty only in the two cases where
 * there is nothing to list — (1) the manifest's structure is broken and cannot be enumerated, (2)
 * the manifest has not been fetched yet (revision resolution failure, a 404 on `karume.json`, …).
 */

/**
 * The list of "what this repo offers right now" shown on failure.
 *
 * NOTE: `quants` / `dtypes` belong to **the model in context** (v2 has several models per manifest,
 * so mixing across models would mislead by recommending "a quant name of another model"). Where the
 * model context is not settled (top-level checks, the fetch layer) they are empty and only `models`
 * is filled.
 */
export type AvailableLabels = {
  /**
   * Model names that exist in the manifest (an enumeration first machine-readable in v2 — ADR 0041
   * decision 8).
   */
  readonly models: readonly string[];
  /** Quant names that exist for the model in context. */
  readonly quants: readonly string[];
  /** Weights name of the model in context → its list of dtype labels. */
  readonly dtypes: Readonly<Record<string, readonly string[]>>;
};

/** The value used when the manifest's structure is broken and labels cannot be enumerated. */
export const NO_LABELS: AvailableLabels = { models: [], quants: [], dtypes: {} };

type HubErrorOptions = {
  readonly available?: AvailableLabels;
  readonly cause?: unknown;
};

/**
 * The base of the error types hub **publishes** (published so that `instanceof HubError` handles
 * them all at once). Only "failures caused by the user's input or environment" are published (the
 * policy of `mod.ts`), so failures of surfaces that are not fetches — the range reads that
 * `openAsset` returns — surface as the source's bare `Error` and do not go through this base.
 *
 * It is thrown directly only for "a failure the user should branch on that is none of the five
 * classes below" — currently two places: `evictCachedAssets` refusing a source that lacks the
 * capability (a source without a cache), and `loadManifest` refusing, before any fetch, a repo name
 * that is not an HF `owner/name`.
 */
export class HubError extends Error {
  /** The available model / quant / dtype labels known at the time of the failure. */
  readonly available: AvailableLabels;

  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = "HubError";
    this.available = options.available ?? NO_LABELS;
  }
}

/**
 * Invalid JSON, structure violation, size limit exceeded (ADR 0041 decision 2 "top level", decision
 * 7 "size limits").
 */
export class ManifestFormatError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestFormatError";
  }
}

/**
 * Violations of references and vocabulary (ADR 0041 decisions 2 / 3). A dangling `defaultModel` /
 * `defaultQuant`, a `weights` mapping with missing or extra entries, unknown keys, a `session`
 * value not in the allowlist, a mismatch of the triple for a duplicated path.
 */
export class ManifestReferenceError extends HubError {
  constructor(message: string, options: HubErrorOptions = {}) {
    super(message, options);
    this.name = "ManifestReferenceError";
  }
}

type ManifestPathErrorOptions = HubErrorOptions & { readonly path: string };

/** Path allowlist violation (ADR 0038 §2). */
export class ManifestPathError extends HubError {
  /** The raw path string that violated it (before the URL is built). */
  readonly path: string;

  constructor(message: string, options: ManifestPathErrorOptions) {
    super(message, options);
    this.name = "ManifestPathError";
    this.path = options.path;
  }
}

/**
 * Where an integrity check failed (cache read / network fetch / reading a local file).
 *
 * `"local"` means bytes read by a source that goes through neither CacheStorage nor the network (a
 * local directory). Fetching again returns the same bytes, so unlike `"network"`, a retry is not a
 * way to recover.
 */
export type IntegritySource = "cache" | "network" | "local";

type IntegrityErrorOptions = HubErrorOptions & {
  readonly repo?: string;
  readonly revisionSha?: string;
  readonly path: string;
  readonly expected: string;
  readonly actual: string;
  readonly source: IntegritySource;
};

/**
 * A fetched file disagreed with the manifest's triple (ADR 0038 §5).
 *
 * NOTE: what surfaces is a mismatch in a file read by the local source (`source: "local"`). A
 * mismatch from the HF source is, on the cache-read side, used by the fetch layer for self-heal
 * (evict → fetch again, at most one round trip), and for a network fetch it is carried as the fetch
 * layer's verification failure in the `cause` of a {@link HubFetchError}.
 */
export class IntegrityError extends HubError {
  /** Only when the source has the notion of a repo (HF). The local source **does not name one**. */
  readonly repo?: string;
  /**
   * The commit SHA (40 digits) the fetch was pinned to. Only for sources with the notion of
   * generations.
   */
  readonly revisionSha?: string;
  readonly path: string;
  /** The value the manifest declared (sha256 / byte count). */
  readonly expected: string;
  /** The value actually obtained. */
  readonly actual: string;
  readonly source: IntegritySource;

  constructor(message: string, options: IntegrityErrorOptions) {
    super(message, options);
    this.name = "IntegrityError";
    // An identity the source lacks does not appear as a field at all (no synthetic repo, no fake
    // SHA — `source.ts` SourceOrigin).
    if (options.repo !== undefined) this.repo = options.repo;
    if (options.revisionSha !== undefined) this.revisionSha = options.revisionSha;
    this.path = options.path;
    this.expected = options.expected;
    this.actual = options.actual;
    this.source = options.source;
  }
}

type HubFetchErrorOptions = HubErrorOptions & {
  readonly repo?: string;
  readonly revisionSha?: string;
  readonly path?: string;
};

/**
 * A container that **passes through with context** the failures from the fetch layer (404, auth,
 * revision resolution failure) (ADR 0038 §5). The original error stays as `cause`.
 */
export class HubFetchError extends HubError {
  /**
   * Only when the source has the notion of a repo (HF). The local source **does not name one** —
   * the naming in `message` (directory / path) is the primary information on what was read.
   */
  readonly repo?: string;
  /**
   * The resolved commit SHA. undefined when revision resolution itself failed, or for a source
   * without generations.
   */
  readonly revisionSha?: string;
  /**
   * The path of the target file. undefined before the manifest fetch / on revision resolution
   * failure.
   */
  readonly path?: string;

  constructor(message: string, options: HubFetchErrorOptions) {
    super(message, options);
    this.name = "HubFetchError";
    if (options.repo !== undefined) this.repo = options.repo;
    if (options.revisionSha !== undefined) this.revisionSha = options.revisionSha;
    if (options.path !== undefined) this.path = options.path;
  }
}
