# 0094: Hub cache inventory and per-selection eviction (reference counting) — resolves ADR 0080 decision 5

- Status: accepted (2026-09-05 — the maintainer's ruling: a file is protected by the other fully
  cached selections of the same manifest; no counting across manifests, a missing file is fetched
  again on the next load); decision 3's "no counting across manifests" re-graded 2026-10-18 to MAY
- Date: 2026-09-05
- Related: ADR [0080](0080-hub-fetch-cache-050.md) (decision 5 deferred the per-repo clean-up
  `pruneHubCache` to a cache maintenance phase; **this ADR is that phase**) /
  [0086](0086-distribution-source.md) (optional capabilities of the source contract; this ADR adds ⑥
  and ⑦) / [0038](0038-manifest-v1.md) §7 (cross-repo references: the referenced repo owns the file) /
  ADR 0010 (429 / 503 retries) and 0011 (HF receive limit) of the fetch layer `@hdae/fetch-cache`

## Context

anima-web built a model manager (download separate from load). To tell whether a model × quant was
downloaded, it depended on the fetch layer `@hdae/fetch-cache` directly and matched the content keys
`["hf", kind, repo, path, sha256]` from `listKeys` against the manifest's `FileRef`s. It used only
public API and the documented key format, but **the fetch layer's key design leaked into the app**:
the app re-implemented rules the hub already owns (repo resolution of cross-repo references,
de-duplication by `fileRefKey`). The same request asked to delete the cache per model, which needs
reference counting: quants and models share files (switching the quant keeps the same tokenizer and
f16 text encoder). The hub's only clean-up API was `clearHubCache` (the whole namespace).

The third item (more requests after shard splitting, more HF 429s; follow `Retry-After`) belongs to the
fetch layer and was done there (its ADR 0010), with the useful part of the hub's receive-byte guard
`transport.ts` (abort once the body exceeds the declared size; its ADR 0011).

## Decision

### 1. Two public functions — the unit is the same "model / quant pair" as fetching

```ts
listCachedAssets(loaded: LoadedManifest, selection?: ResolveOptions, options?: { caches? })
  → { cached: FileRef[]; missing: FileRef[] }
evictCachedAssets(loaded: LoadedManifest, selection?: ResolveOptions, options?: { caches? })
  → { evicted: FileRef[]; kept: { ref; reason: "shared" | "cross-repo"; sharedWith: string[] }[] }
```

The first argument is a `LoadedManifest`: only it carries the repo and the source (HF or local)
(`packages/hub/src/session.ts`); a bare `Manifest` would make the caller re-implement the cross-repo
rule. The refs are the selection's refs de-duplicated by `fileRefKey` (`selectionRefs` in
`packages/hub/src/resolve.ts`; same rule and order as `fetchAssets`). No boolean helper.

### 2. Inventory and eviction are optional source capabilities (⑥ `inventory` / ⑦ `evict`)

The spelling of a cache key belongs to the fetch layer; if the hub's common layer built keys, a
fetch-layer release could leave entries that look deleted. `PinnedSource` gains two optional
capabilities (`packages/hub/src/source.ts`); the common layer only groups refs by origin (the
(repo, revision) pair of `crossRefOf`, else the session's source) and asks each origin.

- **HF source**: calls `listKeys(["hf", "model", repo])` **once** per repo and matches by
  (path, sha256) (one call per ref would list the whole cache per ref). Eviction calls `evict` with the
  full five-part key (prefix semantics, but a full key matches one entry). Only
  `packages/hub/src/sources/hf.ts` spells the key.
- **Local source**: `inventory` answers "everything passed in is present" (as with not implementing
  phase 1: a directly readable source is already cheap to read; a missing file fails when read). No
  `evict`: the directory holds the user's assets, not fetched copies; the common layer refuses with
  `HubError`.

### 3. Reference counting stays within one manifest; only fully cached other selections protect

- A file used by another (model, quant) of the same manifest **whose refs are all cached** is kept
  (`kept: "shared"`, `sharedWith` lists `"<model>/<quant>"`), matching what the app shows as
  "downloaded".
- **A partly cached selection does not protect**: it fetches the rest on its next use anyway, and
  protecting it would leave files that cannot be deleted yet cannot be used.
- **A cross-repo reference is not deleted from the referring side** (`kept: "cross-repo"`): the
  referenced repo owns the file; delete it through that repo's manifest.
- This record does not count references across manifests. Example: anima-extra references anima's
  text stack, so evicting an anima selection leaves extra partly cached and its next load fetches the
  missing files again (the cache is an optimisation, not a correctness requirement). A later version
  MAY add counting across manifests (several manifests, or explicitly protected selections).
  (Re-graded 2026-10-18 to MAY: this was the scope of the first implementation, not a prohibition.)
- Refs that were never cached do not appear in the result. The manifest file (`karume.json`) is out of
  scope (URL key, small; `clearHubCache` drops it).

### 4. 429 / 503 retries and the receive limit live in the fetch layer

fetch-cache retries by default (`Retry-After` first, else 1 / 2 / 4 / 8 / 16 s; at most 5 attempts;
`onRetry` notification; `signal` aborts; `retry: false` disables). The hub adopted it with fetch-cache
0.7.0 (2026-09-05); the dependency is now `^0.8.0` (`pnpm-workspace.yaml`).

- `LoadManifestOptions.onRetry` reaches every fetch: revision resolution, `karume.json`, assets
  (phase 1 / phase 2) and cross-repo targets. The notification type is the hub-owned `RetryDiagnostic`
  (like `CacheDiagnostic`, structurally equal to the fetch layer's type, not re-exported). The retry
  policy is the fetch layer's default; `retry` is not on the hub's public surface.
- `transport.ts` (the hub's `fetch` wrapper: content-length pre-check + abort on excess bytes) was
  removed. Aborting on excess bytes moved to the HF layer's `expectedBytes` limit. The content-length
  pre-check is not carried over (false positives through Content-Encoding in a general library). This
  overrides ADR 0038 §2 ("abort when `content-length` disagrees with `size`", "abort the manifest
  fetch above 1 MiB"): the fetch layer has no limit without an exact match, so the 1 MiB limit of
  `karume.json` is **checked after the whole body is received** (`parseManifest`; recorded in
  `docs/limitations.md`).
- Excess bytes from the HF source raise `HubFetchError` (`cause` = the fetch layer's error), as a
  sha256 mismatch does. Only the local source throws `IntegrityError`; `SizeViolation` lost its
  argument naming where the excess was found (content-length / body).

## Alternatives considered

- **Keep anima-web's direct dependency on fetch-cache**: it re-implements rules the hub now holds.
- **A boolean `isSelectionCached`**: only more API surface (`missing.length === 0`).
- **"Any selection with one cached file protects" / "every selection protects"**: the first frees less
  space; with the second a shared file is never deleted (fails "delete it once nothing shares it").
- **Retry 429 in the hub's transport wrapper**: no help to other consumers (yomi / sbv2-web), and it
  duplicates the fetch layer's HTTP error handling (ruled for fetch-cache).

## Consequences

- The hub's public API only grows (next minor). The source contract is internal (not in `mod.ts`).
- Without `CacheStorage`, a caching source (HF) reports everything as `missing` and evicts nothing (the
  fetch layer's contract; no special case in the hub). The local source bypasses the cache (decision
  2) and reports "all cached" either way. On a runtime without `Cache.keys()` (Deno 2.8 and earlier)
  the fetch layer throws loudly; on Node, `nodeCacheStorage` implements `keys` (`packages/hub/node.ts`).
- anima-web can drop its direct dependency on `@hdae/fetch-cache`.
- Future options: counting across manifests / eviction for the referenced repo only / a dedicated
  "source lacks the capability" error (today the base `HubError`) / a real existence check for the
  local inventory (directory I/O per query).

## Addendum (2026-09-06) — sibling presets with the same reference set, `protect` / `alsoEvicted`

anima-web saw "0 files evicted whichever preset is deleted". In the distributed anima manifest,
`f16`, `f16-c16` and four `f16+dit8` presets point at **the same reference set** (same weights,
different session settings); under decision 3 the siblings protect each other and nothing is ever
freed. Ruling (2026-09-06):

- **By default, selections whose reference set equals the target's do not protect.** At cache
  granularity they cannot be told apart and "delete only one" is undefined; counting them would delete
  nothing and still claim protection. Proper subsets and supersets still protect when fully cached.
- **`CacheInventoryOptions.protect?: readonly ResolveOptions[]`** (the future option "name protected
  selections explicitly"): when given, only this list protects and the same-set exclusion is off —
  naming a sibling protects it (`kept: "shared"`). The policy (e.g. curated presets × all catalog
  models − the target) is the app's. The target is ignored; an unknown model / quant / weights raises
  `ManifestReferenceError`. No effect on `listCachedAssets`.
- **`EvictedAssets.alsoEvicted: readonly string[]`**: labels of selections that became partly cached
  as a side effect (fully cached before the call and using at least one ref actually evicted, judged on
  the set the source reported as evicted). The app withdraws their "downloaded" mark.
- Effect: only manifests with same-set presets change their default result (0.11.0 freed 0 files
  there). The public API only gains fields (next minor).
