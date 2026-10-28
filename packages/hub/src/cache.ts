/**
 * Clean-up of cache namespaces. Not the source itself but **hub's session housekeeping** (the bulk
 * delete offered to users).
 *
 * NOTE: the target is the Cache Storage namespace owned by the fetch layer (`@hdae/fetch-cache`).
 * Once a source that does not go through CacheStorage arrives, there is a "source with no cache to
 * clear", and that is the place to move the fail loud of {@link clearHubCache} (below) onto the
 * source's capabilities.
 */

import { clearCache } from "@hdae/fetch-cache";

/**
 * Deletes downloaded assets to free space. The target is **the fetch layer's whole namespace**.
 * This is also the recovery path when the default of trusting the recorded hash is in doubt (there
 * is no knob equivalent to `recheck` — DECIDED: `docs/decisions/0080-hub-fetch-cache-050.md`).
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
  // The fetch layer owns the namespace names, so hub does not hard-code their spelling and leaves
  // it to the clean-up API.
  return await clearCache({ caches: storage });
};
