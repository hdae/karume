/**
 * `@karume/hub` — resolves distribution manifest v5 (`karume.json` / `karume/5`) and fetches from
 * Hugging Face or a local directory.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — read a manifest ({@link parseManifest} /
 * {@link loadManifest}) / use a local distribution as the source ({@link localDirectory} — the
 * per-runtime readers are subpaths such as `@karume/hub/deno`) / choose a model and an execution
 * configuration ({@link resolveSelection} — the flat FileRef list is {@link selectionRefs}) / read
 * one container with range reads ({@link openContainerSource} — warm it first with
 * {@link prefetchAssets}; the result can be passed as is to `openContainer` of `@karume/runtime`) /
 * fetch assets ({@link fetchAssets} — the surface that loads the whole selection into RAM; the RAM
 * peak is "the byte budget of concurrent fetches + the total of completed ones", and a model that
 * cannot be held whole is read through the container surface ({@link openContainerSource}) with
 * range reads, only the blocks needed, when they are needed) / download assets into the persistent
 * cache ahead of time ({@link prefetchAssets} — a warm-only surface that does not load bytes into
 * RAM) / read only a range of one asset ({@link openAsset} — `undefined` when the source does not
 * support it) / handle failures by type ({@link HubError} and its subclasses) / receive cache
 * diagnostics ({@link CacheDiagnostic}) / receive notifications of the fetch layer's retries
 * (following `Retry-After` on 429 / 503) ({@link RetryDiagnostic}) / clear the cache to free space
 * ({@link clearHubCache}) / ask whether a selection is already downloaded ({@link listCachedAssets}
 * — it does not fetch) / remove the stored files of one selection ({@link evictCachedAssets} — files
 * used by other selections and cross-repository references are kept).
 *
 * The specification is `docs/decisions/0109-manifest-v5-container.md` (the physical container
 * format is `docs/container-v1.md`, the fetch layer is `0038-manifest-v1.md` §5).
 *
 * ## Versions and manifests
 *
 * **Manifests of older versions are not read** (a different major fails as an unsupported format —
 * ADR 0109 decision 1). The three JSR packages and PyPI `karume` (the side that writes manifests)
 * move in lockstep, so the table below applies as is to `@karume/runtime` / `@karume/models` / the
 * exporter. A table for planning ahead when a distribution (`karume.json` of a Hugging Face
 * repository) has to be rebuilt:
 *
 * | Package version | `format` | Main changes |
 * | --- | --- | --- |
 * | 0.1.x | `karume/1` | First version |
 * | 0.2.x – 0.3.x | `karume/2` | Two axes, model / quant (ADR 0041) |
 * | 0.4.x | `karume/3` | shard field of dtype entries (ADR 0071) |
 * | 0.5.x – 0.12.x | `karume/4` | quant display fields + `requiredLimits` (ADR 0075 / 0038 §7),
 * cross-repository fields of file references (`repo` / `revision`), compute knob value `i8a8` →
 * `a8` in `session` (ADR 0074) |
 * | 0.13.x | `karume/5` | The distribution becomes containers (`krm`) — dtype entries become
 * `{ container }` (the descriptor's expected values + the FileRef list of parts); `shards` / `extras`
 * retired (ADR 0109) |
 *
 * The procedure for re-publishing a distribution is `docs/release-runbook.md`.
 */

export { MANIFEST_FILENAME, parseManifest } from "./src/manifest.ts";
export type {
  AttentionCompute,
  ContainerRef,
  DocumentExpectation,
  FileRef,
  GpuFeaturesSpec,
  LinearCompute,
  Manifest,
  ModelEntry,
  PipelineId,
  Quant,
  RequiredLimitName,
  RequiredLimitsSpec,
  ScoreStorage,
  SessionSpec,
  WeightContainer,
  WeightEntry,
} from "./src/manifest.ts";

export { resolveSelection, selectionRefs } from "./src/resolve.ts";
export type { ResolvedSelection, ResolveOptions } from "./src/resolve.ts";

/**
 * The fetch surface for one container (ADR 0109 decision 7). The {@link ContainerBlockSource}
 * returned by `openContainerSource` is structurally compatible with `BlockSource` of
 * `@karume/runtime`, so it can be passed as is to `openContainer({ kind: "source", source })`.
 *
 * This surface **does not warm** (opening it fetches no bytes at all) — warm first, with progress
 * and cancellation, through {@link prefetchAssets} (so that the caller can keep the order of passing
 * admission on the descriptor before downloading weights — ADR 0108 decision 19).
 */
export { openContainerSource } from "./src/container.ts";
export type { ContainerBlockSource } from "./src/container.ts";

/**
 * Sources. `loadManifest` / `fromPretrained` accept either a Hugging Face repository reference
 * ({@link HubRepoRef}) or a source handle created here ({@link DistributionSource} — **its contents
 * are opaque**). A surface that handles the union itself (`fromPretrained` of `@karume/models`)
 * tells them apart with {@link isDistributionSource} — telling an opaque handle apart by its
 * structure would let a misspelling pass as a source.
 */
export { localDirectory } from "./src/sources/local.ts";
export type { DirectoryAdapter, LocalDirectoryOptions } from "./src/sources/local.ts";
export { isDistributionSource } from "./src/source.ts";
export type { DistributionSource } from "./src/source.ts";

export { clearHubCache } from "./src/cache.ts";
export { evictCachedAssets, listCachedAssets } from "./src/inventory.ts";
export type {
  CachedAssets,
  CacheInventoryOptions,
  EvictedAssets,
  KeptAsset,
} from "./src/inventory.ts";
export { fetchAssets, loadManifest, prefetchAssets } from "./src/fetch.ts";
/**
 * Range reads of one asset ({@link openAsset} — reads only `[offset, offset + length)`, not the
 * whole asset. It is an **optional capability**, so a source without it returns `undefined` and the
 * caller falls back to a whole read). As with the other surfaces, only the options
 * ({@link LoadManifestOptions}) passed to this call take effect.
 */
export { openAsset } from "./src/fetch.ts";
export type { AssetRangeReader } from "./src/source.ts";
export type { AssetPhase, AssetProgress } from "./src/progress.ts";
export type {
  CacheDiagnostic,
  FetchAssetsOptions,
  HubRepoRef,
  LoadedManifest,
  LoadManifestOptions,
  RetryDiagnostic,
} from "./src/session.ts";

/**
 * Only error types for "failures caused by the user's input or environment" are public. A broken
 * invariant of the fetch layer (bytes not covering the whole buffer, and the like) is a bug in the
 * hub, not something the user should branch on, so it arrives as a plain `Error`.
 */
export {
  HubError,
  HubFetchError,
  IntegrityError,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
} from "./src/errors.ts";
export type { AvailableLabels, IntegritySource } from "./src/errors.ts";
