/**
 * What the hub's CacheStorage holds of a distribution read through the HF-compatible route. Only the
 * hub's public functions touch the cache (`listCachedAssets` here, `clearHubCache` on the page);
 * this module adds the union over every preset of the manifest.
 */

import {
  type CachedAssets,
  type FileRef,
  type HubRepoRef,
  listCachedAssets,
  loadManifest,
} from "@karume/hub";

/** Files and bytes of a distribution, over every (model, preset) pair; a shared file counts once. */
export type CacheInventory = {
  readonly cachedFiles: number;
  readonly cachedBytes: number;
  readonly totalFiles: number;
  readonly totalBytes: number;
};

/** One file however many presets list it (cross-repo files are told apart by their repo). */
const fileKey = (ref: FileRef): string => `${ref.repo ?? ""}\n${ref.path}\n${ref.sha256}`;

/** The union of several presets' inventories (pure). */
export const unionInventory = (inventories: readonly CachedAssets[]): CacheInventory => {
  const cached = new Map<string, number>();
  const total = new Map<string, number>();
  for (const { cached: present, missing } of inventories) {
    for (const ref of present) cached.set(fileKey(ref), ref.size);
    for (const ref of [...present, ...missing]) total.set(fileKey(ref), ref.size);
  }
  const sum = (sizes: ReadonlyMap<string, number>): number =>
    [...sizes.values()].reduce((bytes, size) => bytes + size, 0);
  return {
    cachedFiles: cached.size,
    cachedBytes: sum(cached),
    totalFiles: total.size,
    totalBytes: sum(total),
  };
};

/** Queries the cache for every preset of the distribution at `ref` (fetches only the manifest). */
export const queryCacheInventory = async (
  ref: HubRepoRef,
  signal?: AbortSignal,
): Promise<CacheInventory> => {
  const loaded = await loadManifest(ref, signal === undefined ? {} : { signal });
  const inventories: CachedAssets[] = [];
  for (const [model, entry] of Object.entries(loaded.manifest.models)) {
    for (const quant of Object.keys(entry.quants)) {
      inventories.push(await listCachedAssets(loaded, { model, quant }));
    }
  }
  return unionInventory(inventories);
};
