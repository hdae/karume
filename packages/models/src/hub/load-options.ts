/**
 * The options every family's `fromPretrained` has in common — **one type and one mapping**. Two
 * parts: the knobs that pass through to the fetch layer ({@link FromPretrainedHubOptions}) and the
 * component-replacement option ({@link FromPretrainedComponentOptions}).
 *
 * MUST: not duplicated per family. The same 5 fields and the same spread used to be hand-written in
 * 8 places, and when hub added a field ({@link FromPretrainedHubOptions.onRetry} in 0.11.0) all 8
 * missed it — the shape "a new knob silently does not arrive" recurs as long as copies exist.
 *
 * NOTE: `signal` is not included here — where it lives differs per family (families that pass it to
 * **both** fetching and construction keep it on their `PipelineOptions`; families where it affects
 * only fetching keep it on their `FromPretrainedOptions`). {@link hubLoadOptions} only takes it as
 * an argument and maps it; it does not touch where it lives.
 */

import type {
  AssetProgress,
  CacheDiagnostic,
  LoadManifestOptions,
  RetryDiagnostic,
} from "@karume/hub";
import type { ComponentSource } from "./components.ts";

/**
 * The fetch-layer knobs that every family's `FromPretrainedOptions` shares
 * (`sbv2.FromPretrainedOptions` and so on through the barrel; passed through to hub as is).
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` are knobs **for the HTTP source only**; none of
 * them has any effect on a call that passes a source handle (`localDirectory` / `denoDirectory` /
 * `nodeDirectory`) — a local distribution goes through neither the network nor CacheStorage.
 */
export type FromPretrainedHubOptions = {
  /**
   * `Authorization` etc. Passed through as is to fetching (revision resolution, files).
   *
   * NOTE: **the cache is not partitioned by credential** (by design — headers are not part of the
   * key, so bytes fetched with authentication also hit for later unauthenticated calls). The
   * canonical source is ADR 0080 decision 3 / `docs/limitations.md` "hub: the cache is not isolated
   * by credential".
   */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /**
   * Notified on each retry of a 429 / 503 by the fetch layer (delivered **before** the wait
   * starts). The option exists so an app can show "rate limited, waiting N more seconds"; without
   * it, it looks like tens of seconds of silence.
   *
   * It covers **both** the manifest fetch (`loadManifest`) and the asset fetches (the sequential
   * surface of container parts, prefetch, the whole-bytes surface) — this one listener reaches every
   * fetch of `fromPretrained`. A throwing listener does not fail the fetch (the fetch layer isolates
   * it and warns).
   */
  readonly onRetry?: (diagnostic: RetryDiagnostic) => void;
  /** Replacement for `fetch` (for tests and custom transports). */
  readonly fetch?: typeof globalThis.fetch;
  /** Replacement for `CacheStorage` (for tests). */
  readonly caches?: CacheStorage;
};

/**
 * The **component-replacement option** that every family's `FromPretrainedOptions` shares (ADR 0108
 * decision 19 / ADR 0109 decision 10).
 */
export type FromPretrainedComponentOptions = {
  /**
   * Role (the manifest's weights name) → **the same role** in another `karume/5` repo. Only the
   * components of the given roles come from that source; the rest are built from the original repo.
   *
   * MUST: only components with an **identical graph descriptor (sha256 of `descriptor.graph`)** can
   * be replaced. Identity is judged **before a single byte of weights is fetched**, from the two
   * manifests' declarations alone, and a difference is rejected (zero shortage / excess in the
   * binding table is also confirmed from the descriptor alone). A role name this family does not
   * have fails loudly — a misspelling is not silently folded into "replace nothing".
   */
  readonly components?: Readonly<Record<string, ComponentSource>>;
};

/**
 * `fromPretrained` options → hub options. Maps **only the defined fields** (no `key: undefined` is
 * created — an explicit `undefined` can affect hub's branching differently from "not given").
 *
 * `onProgress` is not mapped — whether progress needs aggregation differs per family, and the side
 * that passes it to `loadContainerComponents` attaches it separately (the manifest fetch has no
 * progress). `components` is not mapped either — it is not a fetch-layer knob but an option the
 * seam (`hub/components.ts`) receives.
 */
export const hubLoadOptions = (
  options: FromPretrainedHubOptions & { readonly signal?: AbortSignal },
): LoadManifestOptions => ({
  ...(options.signal === undefined ? {} : { signal: options.signal }),
  ...(options.headers === undefined ? {} : { headers: options.headers }),
  ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
  ...(options.onRetry === undefined ? {} : { onRetry: options.onRetry }),
  ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
  ...(options.caches === undefined ? {} : { caches: options.caches }),
});
