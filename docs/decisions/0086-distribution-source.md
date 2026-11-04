# 0086: Distribution source abstraction — `DistributionSource` and local distributions

- Status: accepted (2026-08-31 — the public-surface review's ruling "local loader = b)
  DistributionSource abstraction" and the second-round design rulings: skeleton, three
  confirmations, unmapped cross-repo references; implemented in stages ① to ③; ruling records in
  `.claude/reviews/2026-08-31_182ced7/SUMMARY.md`, local and untracked); partly superseded by
  [0127](0127-node-dawn-toolchain.md) (decision 4's third carve-out condition, by the Node
  `CacheStorage` on the same subpath, 2026-10-17); decision 5's `prefetchAssets` no-op revised by the
  2026-10-11 addendum
- Date: 2026-08-31
- Scope: `packages/hub/` (`src/source.ts` = internal contract, `src/sources/{hf,local}.ts` =
  adapters, `deno.ts` = subpath, `src/session.ts`, `src/errors.ts`), `packages/models/` (the eight
  families' `fromPretrained`, `src/hub/repo-ref.ts`). **IR, runtime, distribution (`karume.json`)
  and exporter unchanged.**
- Related: [0038](0038-manifest-v1.md) §5 (fetch-layer contract — demoted here to "the HF adapter's
  contract") / [0080](0080-hub-fetch-cache-050.md) decision 1 (verification is per source) /
  [0070](0070-shard-loading-admission.md) decision 2 (two-phase loading — phase 1 becomes optional) /
  [0073](0073-models-source-pin.md) (`ref` is required — carried over to source handles) /
  [0008](0008-public-api.md) (thin public surface)

## Context

A distribution already sits locally as one directory (`models/karume-gemma4-e2b`, a 3.8 GiB mirror —
`docs/assets-layout.md`), yet **the public surface could not read it** — the largest blocker of the
public-surface review (2026-08-31), agreed by every leg. The only way was:

- Start `examples/shared/local-dist-server.ts` (**a private 196-line fake HF server**) and fetch over
  HTTP from a real port, under a fake repo name and commit SHA, **copying the 3.8 GiB again** into
  CacheStorage (`DENO_DIR` on Deno). A first-time consumer read six files to find this.
- `fromPretrained("./models/karume-gemma4-e2b")` **looks like an HF `owner/name`** (a legal repo name,
  so no spelling check helps), slid into the fetch layer and returned 401 / 404 — a diagnosis far from
  the cause.

Hub's fetch path was built from HF vocabulary (`repo`, commit SHA, `hubUrl`, CacheStorage, `fetch`);
"the source is not HF" did not exist in the types.

## Decision

### 1. Skeleton: an opaque handle + an internal driver contract (the public surface is one handle)

The public surface exposes only **a handle with no content**, `DistributionSource`
(`src/source.ts`), and its factories (`localDirectory`; `denoDirectory` in `@karume/hub/deno`).
Fetching sits behind the internal contract `SourceDriver` / `PinnedSource`; `mod.ts` exports
`DistributionSource` **as a type only**. The contract is **the five questions** left after folding
the existing fetch path:

1. Mutable ref → immutable generation identifier (`SourceDriver.resolveGeneration` — **the session's
   single resolution point**).
2. The full bytes of one `karume.json` (`PinnedSource.readManifest`).
3. The full bytes of a `FileRef` (`PinnedSource.readFile`, with expected sha256 / size).
4. Make 3 cheap later without holding the bytes in RAM (`PinnedSource.prefetchFile` = phase 1 of ADR
   0070 decision 2; **optional**).
5. Cross-repo (repo, revision) → another source (`PinnedSource.originFor` — ADR 0038 §7).

Addendum (2026-09-07): three more questions. ⑥⑦ = listing and deleting the cache inventory
(`PinnedSource.inventory` / `evict`, public as `listCachedAssets` / `evictCachedAssets` — ADR
[0094](0094-hub-cache-inventory-and-eviction.md); optional). ⑧ = **reading a range of a `FileRef`**
(`PinnedSource.openFile` → `AssetRangeReader { cost, read(offset, length) }`; optional; public as
`openAsset(loaded, ref, options)`). Motivation: gemma4's PLE (then 9 token-major shards of 253 MB) —
re-reading a whole shard per decode token cost 137 reads, about 42 s, for 400 tokens (measured
2026-09-07), so only the row (8,960 B + 140 B) must be read. The reader names its **cost type**:
`"seek"` = small regardless of offset (positional file reads; `slice` of a lazy Blob in the browser,
0.1–0.3 ms on Chrome 152) / `"scan"` = proportional to offset (skipping through a body stream — Deno's
CacheStorage reads the whole body in `blob()`; 17–76 ms). ⑧ checks only the bounds of the declared
size, not the full-read size check or sha256 (corruption shows up as row values). The common-layer
conventions (abort, tight view) live in `src/fetch.ts` as for ③. `denoDirectory` implements it with
positional reads (seek); the HF source with `openHfFile` of `@hdae/fetch-cache` 0.8.0 (strategy blob →
seek, stream → scan; an entry not in the cache is prefetched as in phase 1 first).

MUST: progress, concurrency (the in-flight byte budget), abort propagation, the tight-view check and
error context stay in `src/fetch.ts` / `src/context.ts` as **common-layer conventions**; pushed into
sources, every new source would rewrite the same invariants.

MUST: **the handle grows no public members.** It is identified by **identity** (the type predicate
`isDistributionSource`, wrapping `instanceof`) — a brand field can be forged, and structural detection
(`"repo" in value`) would let a misspelt `HubRepoRef` pass as a source. The models side's `ref |
source` branches also use this one predicate (`toManifestSource` in `src/hub/repo-ref.ts`).

MUST: per-call conventions (`fetch`, `caches`, `headers`, `onCacheError`) are passed **at each `pin`
call, not when the source is created**. A source holds only "where from"; "how" comes from each
call's options — otherwise a `fetch` given to `loadManifest` would silently apply to later
`fetchAssets`.

The local source delegates reading to `DirectoryAdapter` (one required method,
`readFile(path, {signal})`, plus the optional `readFileRange` for ⑧), and `localDirectory` stays
**pure Web standard** (no runtime-specific API). Future backends (OPFS, IndexedDB, File System Access)
plug into this interface.

### 2. The local source verifies the exact size only (sha256 is trusted from the record)

`localDirectory`'s `readFile` checks only **exact equality with `FileRef.size`**, **not sha256** —
as ADR 0080 decision 1 removed re-hashing by "verify on store, trust the record". Local files are
**the user's own assets**, not fetched from the distributor, and no threat justifies hashing several
GiB on every start. The size is free once reading ends, so it stays as a check (a truncated copy or
another quant fails here). **No tamper detection** is recorded as by-design in
`docs/limitations.md`.

A local manifest is a format check on a local file, so it MAY fail **after a full read**, and a
`parse` throw propagates as is (with no cache to evict, a broken manifest correctly fails with the
same `ManifestFormatError` every time).

Addendum (2026-09-25 — the container path): "no sha256, no tamper detection" still holds for **whole
files in the host (hub)**. For the container (`krm` parts — ADR [0108](0108-container-format.md) /
[0109](0109-manifest-v5-container.md) decision 7), a local source is **unverified** (`integrity:
"local"`, so `verified = integrity === "network"` in `packages/hub/src/container.ts` is false): the
runtime's `openContainer` always checks the two descriptor documents against the manifest's length and
sha256 and checks each block's sha256 on every read (`readBlock` in
`packages/runtime/src/format/container/open.ts`). **Tampering in blocks that are read is detected**;
exact size remains the only check for files read through a `FileRef` in the manifest's `assets`.

Addendum (2026-10-10 — the manifest limit): since fetch layer 0.7.0 (ADR
[0094](0094-hub-cache-inventory-and-eviction.md) decision 4) HF's `MAX_MANIFEST_BYTES` no longer
stops receiving midway; `parseManifest` checks the 1 MiB limit **after the whole body is received**,
for HF as for local sources (`docs/limitations.md`, "hub: the receive limit of the HF source is the
fetch layer's `expectedBytes`").

### 3. Cross-repo references: explicit mapping, otherwise fail loudly (no implicit guessing, no implicit demotion)

Cross-repo references (`FileRef`'s `repo` + `revision` — ADR 0038 §7) resolve **without guessing**:

1. The **explicit mapping** `LocalDirectoryOptions.crossRepo` (key = the manifest's `"owner/name"`;
   value = a `DistributionSource` providing that whole repo).
2. Otherwise **fail loudly** (the message says what to put into `crossRepo`).

Addendum (2026-09-25): the original second step, an explicit delegate `LocalDirectoryOptions.fallback`,
worked with no value the public surface could build (`fallback: hfHub()` always failed — review
finding E-HB1-1 of 2026-09-24) and was removed in 0.13.0; cross-repo targets go into `crossRepo`.

MUST NOT: **guess an adjacent directory with the same name** ("`../karume-anima` if it exists") —
the wrong bytes pass if the size matches (decision 2).

MUST NOT: **demote implicitly to remote** (an offline distribution would go onto the network).

`originFor` returns another coordinate of the same source and **does not drop capabilities** — a
cross-repo target lacking phase 1 is a breach of the source contract, and the common layer fails
loudly.

### 4. The `@karume/hub/deno` subpath is an opt-in carve-out from a cross-cutting invariant

The adapter that reads Deno's file system (`Deno.readFile`) lives **only in this subpath**, an
**explicit carve-out** from the `CLAUDE.md` invariant "runtime dependencies are Web standard APIs
only", under three conditions:

- **`Deno` never appears through `mod.ts`** — the carve-out is confined to one subpath.
- **Browser code does not import it** (browsers pass their own `DirectoryAdapter`); JSR `exports`
  separate `.` and `./deno`, fixing the direction of dependency.
- The subpath holds only "how to read local bytes"; **source semantics** (generation, verification,
  cross-repo, progress) stay with `localDirectory` (superseded by ADR 0127 decision 2 and its
  resolved open point on 2026-10-17: Node has no Cache API, so `@karume/hub/node` also holds
  `nodeCacheStorage`, a file-backed `CacheStorage` for the fetch layer that writes local bytes —
  `packages/hub/node.ts`; source and cache semantics stay in the main module and the fetch layer.
  ADR 0127 decision 2 itself still reads "holds only how to read local bytes" *(corrected
  2026-11-04: ADR 0127's addendum of 2026-10-29 amends decision 2 to "holds only how to read and
  write local bytes" — the directory source and, on Node, `nodeCacheStorage`)*).

Another runtime gets the same carve-out under these conditions (`@karume/hub/<runtime>`; main module
and models unchanged) — `@karume/hub/node` (`nodeDirectory`, ADR 0127 decision 2).

### 5. The local source bypasses CacheStorage (zero byte copies; `prefetchAssets` was a no-op — revised by the 2026-10-11 addendum)

A local source has nothing to warm, so it **does not implement phase 1 (`prefetchFile`)**; the
common layer sees `prefetchFile === undefined` and returns. This is **the local source's main
advantage**: copying 3.8 GiB into CacheStorage doubles disk use and adds a read for nothing. `headers`
/ `fetch` / `caches` **have no effect on calls given a source handle** (HTTP vocabulary, as the eight
families' `*FromPretrainedOptions` docs state).

Addendum (2026-09-25): the sequential surface `streamAssets` was retired with the old distribution
readers (2026-09-24, stage 3d); local reads go through container range reads (ADR
[0109](0109-manifest-v5-container.md) decision 7) and `fetchAssets`.

### 6. Generalise the vocabulary of errors and identity fields (never claim an identity the source does not have)

HF vocabulary was a **required field** of diagnostics; now each source names only what it has:

- New `SourceOrigin` ("who the source is"): a required one-line `label` (HF:
  `repo owner/name @ <commit SHA>`; local: the directory label) and `integrity`; `repo` /
  `revisionSha` are **omitted by sources without them**.
- `IntegritySource` gains `"local"` (`"cache" | "network" | "local"`): a local re-read returns the
  same bytes, so unlike `"network"` **a retry is no remedy**.
- `repo` / `revisionSha` of `IntegrityError` / `HubFetchError` become optional. Addendum (2026-09-05):
  for a cross-repo reference, `IntegrityError.source` names **the cross-repo target's source class**,
  not the session's — a local session with a remote target is valid, so the source calling
  `SizeViolation` passes it.
- `LoadedManifest` becomes a **class** with optional `repo` / `revisionSha`; `hubUrl` is **removed**.
  The source is carried internally (a module-level `WeakMap` since the 2026-10-11 addendum), and the
  asset surfaces (`fetchAssets`, `prefetchAssets`; `streamAssets` until 2026-09-24) **do not rebuild
  the source from identity fields** — that leans on HF's "repo and SHA restore it" and excludes other
  sources.

MUST: **never synthesise an identity the source does not have.** A local source claiming a repo /
SHA produces diagnostics pointing at a non-existent repo ("go look on HF"); the fake HF server
papered over exactly this hole.

## Future work (not implemented in this step)

- **OPFS / IndexedDB / File System Access picker:** each a `DirectoryAdapter`; no hub change expected.
- **A cache other than the Cache API** (for example OPFS): a **fetch-layer** question, not a source one.
- `pruneHubCache` (per-repo cleanup) stays with the cache-maintenance work (ADR 0080 decision 5).

## Alternatives considered

- **a) `fromDirectory(path)`, a Deno-only entry in models:** smallest, but ① all of `@karume/models`
  would depend on Deno APIs, and ② browser OPFS / pickers would need yet another surface. Rejected.
- **c) Ship the fake HF server** (productise `examples/shared/local-dist-server.ts`): **fixes the
  workaround in the product** (fake repo / SHA, a 3.8 GiB cache copy, a real port) — what decisions 5
  and 6 remove. Rejected.
- **A brand field on the handle** (`{ __karumeSource: true }`): forgeable, and lets a misspelt
  `HubRepoRef` through (decision 1's MUST). Rejected.
- **Check sha256 locally too:** tamper detection, but a full 3.8 GiB hash per start **buys back
  locally the cost ADR 0080 decision 1 removed**. Rejected (decision 2; deployments that need it use
  the HF path — limitations).

## Consequences

- **Hub's public surface gains** `localDirectory`, `DirectoryAdapter` + `LocalDirectoryOptions`,
  `isDistributionSource` and the type `DistributionSource`, plus `denoDirectory` on
  `@karume/hub/deno`. Existing surfaces (`loadManifest`, `resolveFiles`, `fetchAssets`,
  `prefetchAssets`, `clearHubCache`; `streamAssets` retired 2026-09-24) keep **names and kinds**; only
  the first argument widens to `HubRepoRef | DistributionSource`.
- **Breaking (unreleased):** `LoadedManifest`'s `repo` / `revisionSha` optional, `hubUrl` removed;
  code relying on identity fields other than `AssetPhase` has to follow.
- Stages and checks: ① fold HF into one adapter (check: **existing tests green unchanged**) → ②
  `localDirectory` + `denoDirectory` + fixture tests → ③ union entries for the eight families + an
  e2e reading the real mirror (temperature-0 chat equals the golden **verbatim**; fault injection
  and a call count of 0 prove CacheStorage and the network are untouched). One fake-HF-server e2e
  **stays as the HTTP connectivity check**. In the 0.15 tree Deno tests are not maintained (ADR 0127
  decision 3): `packages/hub/tests/local_test.ts` and `node_directory_test.ts` cover the local
  source, and `packages/e2e/tests/e2e_gemma4_pretrained_test.ts` reads the mirror both through the
  HF-shaped server (`packages/test-support/src/local-dist-server.ts`) and through `nodeDirectory`.
- Consumers pass `denoDirectory("./models/karume-gemma4-e2b")` to `fromPretrained`: no network, no
  CacheStorage, only `--allow-read`.
- The HF-specific parts of ADR 0038 §5 (revision resolution, cache namespace, `hubUrl`) are read as
  **the HF adapter's contract**; source-independent conventions (progress, abort, concurrency, tight
  view, error shape) remain the common-layer contract in §5.

## Addendum (2026-10-11)

**Decision 5's "`prefetchAssets` is a no-op" is revised:** refs of a source without phase 1 (for
example a local directory) also emit `complete` once on the warm-up surface. Nothing is warmed (no
bytes read, no CacheStorage). Progress looks like an already-cached ref (one `complete`,
`fileLoaded === fileTotal`).

- Reason: `@karume/models` loading uses the total size of all files as the progress denominator; refs
  that emit nothing keep the last `loaded` short of it, and a distribution without assets emits
  nothing at all. With HF and local sources mixed (component replacement, cross-repo references), the
  one rule "every ref passed to the warm-up surface emits `complete` once" reaches 100% without a way
  to query source capabilities.
- Scope: still per ref; a cross-repo target alone lacking phase 1 still fails loudly. The statement
  "`prefetchAssets` is a no-op for a local source" in ADR [0070](0070-shard-loading-admission.md)'s
  2026-08-31 addendum is read through this one.

**Decision 6's internal field `#session` (and the handle's implementation field) were replaced by
module-level `WeakMap`s** (`sessions` in `session.ts`, `drivers` in `source.ts`): a `static` block
exposing a private field runs at import time, and bundlers keep even unused classes (confirmed with
`deno bundle`), against the invariant "no execution at import time". Carrying the source in the
value, the opaque handle and `instanceof` identification are unchanged; nominal typing comes from a
type-only brand (`declare private readonly brand: never`).
