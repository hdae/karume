/**
 * Clean-up of cache namespaces. Not the source itself but **hub's session housekeeping**
 * (reclaiming the namespaces that older hub versions left behind, and the bulk delete offered to
 * users).
 *
 * NOTE: the target is the Cache Storage namespace owned by the fetch layer (`@hdae/fetch-cache`).
 * Once a source that does not go through CacheStorage arrives, there is a "source with no cache to
 * clear", and that is the place to move the fail loud of {@link clearHubCache} (below) onto the
 * source's capabilities.
 */

import { clearCache } from "@hdae/fetch-cache";
import type { CacheDiagnostic } from "./session.ts";

/**
 * The namespace hub used before the fetch layer owned its namespaces itself (`@hdae/fetch-cache`
 * 0.4 and earlier). Covers both the main `karume/1` and the `karume/1:*` names that isolated
 * authenticated entries.
 */
const LEGACY_CACHE_NAMESPACE = "karume/1";

/**
 * Lists the legacy namespace names (the main `karume/1` and the former auth-isolated `karume/1:*`).
 */
const legacyCacheNames = async (storage: CacheStorage): Promise<string[]> =>
  (await storage.keys()).filter(
    (name) => name === LEGACY_CACHE_NAMESPACE || name.startsWith(`${LEGACY_CACHE_NAMESPACE}:`),
  );

/**
 * Reclaims the legacy namespaces. New code never reads them again, so keeping them only takes space
 * (their contents use the format of fetch layer 0.4 and earlier; neither the keys nor the recorded
 * hashes are compatible with the current ones).
 *
 * The cache is an optimisation, not a correctness requirement, so this never fails a load —
 * environments without `caches` (Node, for example) pass through silently, and failures to list or
 * delete go to the diagnostics and the load continues.
 */
export const purgeLegacyCaches = async (options: {
  readonly caches?: CacheStorage;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
}): Promise<void> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) return;
  let names: readonly string[];
  try {
    names = await legacyCacheNames(storage);
  } catch (error) {
    options.onCacheError?.({ op: "delete", url: LEGACY_CACHE_NAMESPACE, error });
    return;
  }
  for (const name of names) {
    try {
      await storage.delete(name);
    } catch (error) {
      // Name the namespace that failed (this diagnostic's `url` is a cache name — not a source).
      options.onCacheError?.({ op: "delete", url: name, error });
    }
  }
};

/**
 * Deletes downloaded assets to free space. The target is **the fetch layer's whole namespace** and,
 * if still present, the legacy namespaces (the `karume/1` family). This is also the recovery path
 * when the default of trusting the recorded hash is in doubt (there is no knob equivalent to
 * `recheck` — DECIDED: `docs/decisions/0080-hub-fetch-cache-050.md`).
 *
 * NOTE: fetch layer 0.5.0 has a single, internally fixed namespace, and its keys do not record
 * "which app warmed it". So **whatever an app on the same origin warmed by using
 * `@hdae/fetch-cache` directly is deleted as well** (the library offers no way to separate them).
 * Fine-grained clean-up per repo is left to the cache-maintenance phase; this stays the single
 * choice "delete everything".
 *
 * @returns `true` if something existed and was deleted (`false` if there was nothing to begin
 * with).
 */
export const clearHubCache = async (
  options: { readonly caches?: CacheStorage } = {},
): Promise<boolean> => {
  const storage = options.caches ?? globalThis.caches;
  if (storage === undefined) {
    // Do not make this a silent no-op. The worst outcome is "believing it was cleared" in an
    // environment without Cache Storage (a non-secure origin, for example) — the app would treat
    // copies that are still there as deleted.
    throw new Error(
      "hub: cannot clear the cache because this environment has no CacheStorage" +
        " (pass one explicitly with options.caches)",
    );
  }
  // Reclaim the legacy namespaces here too (a caller may clean up without ever going through
  // {@link ../fetch.ts loadManifest}). Here the clean-up is the job itself, so failures propagate
  // instead of being swallowed.
  const legacy = await legacyCacheNames(storage);
  const deleted = await Promise.all(legacy.map((name) => storage.delete(name)));
  // The fetch layer owns the namespace names, so hub does not hard-code their spelling and leaves
  // it to the clean-up API.
  const cleared = await clearCache({ caches: storage });
  return cleared || deleted.includes(true);
};
