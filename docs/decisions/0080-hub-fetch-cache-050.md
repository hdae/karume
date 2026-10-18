# 0080: Hub asset cache moves to the content keys and recorded-hash trust of fetch-cache 0.5.0

- Status: accepted (2026-08-28 — the maintainer's ruling: hand verification to the fetch layer; drop
  the per-credential isolation, since no gated hosting is planned); partly superseded by
  [0086](0086-distribution-source.md) (2026-08-31: verification is decided per source); decision 3's
  re-isolation path narrowed on 2026-10-10; decision 5's purge re-graded 2026-10-18 to MAY.
- Date: 2026-08-28
- Related: ADR [0038](0038-manifest-v1.md) (fetch-cache connection contract — this ADR updates its
  decision 5) / [0070](0070-shard-loading-admission.md) (this ADR **changes** decision 2's "the sha256
  check also runs on cache hits (non-negotiable)": a full re-hash in the hub becomes a comparison of
  the recorded hash in fetch-cache). Replaces an unmerged draft of 0080 (`272c0e9`), whose rulings
  carry over; only the key changed (a hub-made synthetic URL key → the fetch-cache content key).

## Context

fetch-cache 0.5.0 (breaking — upstream ADRs 0006 to 0008) **verifies bytes when storing them,
records the hash on the entry, and decides later hits by a string comparison**, so the full re-hash
of multi-GB assets on every start disappears. The HF content key `["hf", kind, repo, path, sha256]`
does not depend on the revision: a revision bump with unchanged bytes does not download again (the
property the draft built with `https://karume.invalid/sha256/<hex>` keys). 0.5.0 also removed
`cacheName` (the namespace is fixed to `"fetch-cache"`), so the hub's per-credential namespaces
(`karume/1:auth:<hex16>`, isolating gated assets between tokens) lost their place.

## Decision

### 1. Hand verification to fetch-cache (always trust, no knob)

> Narrowed by ADR [0086](0086-distribution-source.md) on 2026-08-31: verification is per source
> (addendum below).

Asset fetching always passes `sha256` (the manifest `FileRef`) and `expectedBytes` (= `size`), and
the hub's own validate hook (full sha256 and size on every hit) is removed. A miss is verified while
receiving (a mismatch leaves no entry); a hit compares the recorded hash; an entry without a record
is backfilled by a verifying first read; a record mismatch evicts and refetches (self-heal).
`recheck` is not passed: there is no opt-in / opt-out knob. The manifest fetch keeps its parse-only
check (no expected sha256 exists in advance).

**For ADR 0070 decision 2**: the purpose (detecting cache corruption) passes to the recorded hash,
but **silent bit rot after storage is not detected** (record and content can rot independently).
The recovery is `clearHubCache`, or `evictCachedAssets` for one selection (ADR
[0094](0094-hub-cache-inventory-and-eviction.md)). This cost was accepted in the draft.

### 2. Keys are the library's content keys

The hub builds no keys: with `sha256` the key is `["hf", kind, repo, path, sha256]`; without it (the
manifest) it is the resolve URL. Same-byte hits across revisions and coexisting revisions follow from
the key design. The draft's `casKeyFor` and `karume-cache/2` namespace are not adopted.

### 3. Per-credential cache isolation is removed (the maintainer's ruling)

The per-token namespaces of `cacheNameFor` are not restored: no gated assets are planned and the
isolation was excessive defence (2026-08-28). The authorization header passes through as
`RequestInit`, but the cache is shared, so another token's user on the same device can reach a gated
asset's copy; `docs/limitations.md` records this as by-design. If gated hosting starts, a custom
`caches` wrapper remapping `open()` was named as the way back — **not enough, see the 2026-10-10
addendum**.

### 4. `AssetPhase` loses `verifying`

Verification happens inside fetch-cache; the hub does not fake a phase it cannot observe (breaking,
pre-release). `fileLoaded` / `fileTotal` still agree at `complete`.

### 5. Migration and what it brings

- **Purge of the old namespaces**: the hub MAY keep deleting `karume/1` and `karume/1:auth:*` at the
  entry of `loadManifest` (the load continues on failure); `clearHubCache` collects them too
  (`packages/hub/src/cache.ts`). The key shape changed, so old entries miss and download once more.
  (Re-graded 2026-10-18 to MAY: the purge stays, its cost is small.)
- **`clearHubCache`** now clears the fixed namespace `"fetch-cache"`, including what an application
  on the same origin cached with fetch-cache directly (stated in the docstring). Fine-grained cleanup
  became `listCachedAssets` / `evictCachedAssets` (ADR 0094).
- **Error before download**: a failed allocation of `expectedBytes` throws before receiving (`cause`
  = `RangeError`), closing the known issue that the Chromium single-`ArrayBuffer` limit surfaced only
  after the download. The hub reports it as the first root cause.

## Consequences

- Breaking (hub 0.7.0): `AssetPhase` is `"downloading" | "complete"`
  (`packages/hub/src/progress.ts`), `clearHubCache` changed target, verification moved internally.
- From the second load on, the full re-hash of multi-GB assets disappears.
- `sha256Hex` and the digest serialization guard leave the hub, but **Chrome's full copy of the
  digest input disappears only on the warming path (phase 1, now `prefetchAssets`)**, which hashes
  while receiving. On the network path of `fetchAssets` fetch-cache digests the finished bytes, so
  Blink's +N peak remains and only `BYTE_BUDGET` (1.5 GiB, `packages/hub/src/fetch.ts`) limits
  concurrency. Cache hits run no digest; prefetch records the hash and skips matching entries.

## Addendum (2026-08-31) — distribution sources (ADR 0086)

Verification in decision 1 is per source. The HF source works as above. **The local source (a
directory on disk) checks only an exact `size` match, not sha256** (ADR 0086 decision 2): files on
disk are the user's assets, not fetched data, so no full hash is paid on every start. It catches a
truncated copy or a mixed-up quant and **does not detect tampering** (by-design,
`docs/limitations.md`). Decision 3 concerns HTTP sources only.

## Addendum (2026-09-25) — container path (ADR 0109 decision 7)

"Does not detect tampering" now means only that **the hub does not check a whole file's sha256**.
For container parts (`krm`) the local source is unverified and the runtime's `openContainer` checks
the two descriptor documents and the sha256 of every block it reads (ADR 0086 decision 2, addendum).
The `size` match remains the only check for files read through a manifest `assets` `FileRef`.

## Addendum (2026-10-10) — merged concurrent fetches

The fetch layer merges concurrent fetches in the process by content key (single-flight) and gives
later callers the first fetch's bytes; their `caches`, authorization header and `fetch` are unused,
and they re-check the full bytes against their own sha256 (by-design in `@hdae/fetch-cache` 0.8.0).
This narrows two statements:

- Decision 3: a custom `caches` wrapper does not isolate. An unauthenticated request made during an
  authenticated fetch joins it and receives the authenticated bytes. Isolation needs merge keys split
  by credential (a fetch-layer change) or separate origins / processes. No gated hosting is planned,
  so users are not affected (`docs/limitations.md`).
- Consequences: a later caller merged into a concurrent fetch recomputes the full SHA-256 even on a
  cache hit. Correctness is unaffected; CPU time grows for parallel large fetches (not measured).
  Changing that is a redesign of the fetch layer's trust model, not a hub fix.
