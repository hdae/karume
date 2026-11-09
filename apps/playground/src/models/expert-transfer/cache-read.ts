/**
 * Reading experts back from the cached 26B distribution: what an expert cache pays on a miss before
 * the upload, when the host store is the hub's CacheStorage (a `slice` of the cached Blob) rather
 * than RAM. An expert is four ranges of the container (gate_up codes and steps, down codes and
 * steps), so a sample reads four ranges per expert, all at once. The ranges are drawn at random
 * positions of the model's parts — the time of a read depends on where the bytes sit on the disk and
 * in the OS cache, not on which expert they hold — and each sample reads its ranges a second time to
 * time the same bytes once the OS has them.
 */

import {
  type AssetRangeReader,
  type FileRef,
  type HubRepoRef,
  listCachedAssets,
  loadManifest,
  openAsset,
  resolveSelection,
} from "@karume/hub";

import { EXPERT_BYTES, type Summary, summarize } from "./stats";

/** The four ranges of one expert, bytes: gate_up codes, gate_up steps, down codes, down steps. */
export const EXPERT_RANGES = [1_982_464, 247_808, 991_232, 123_904] as const;

export type CacheReadResult = {
  readonly id: string;
  readonly experts: number;
  readonly bytes: number;
  /** ms of the first read of each sample's ranges, in run order (warm-up excluded). */
  readonly samples: readonly number[];
  readonly ms: Summary;
  /** ms of reading the same ranges again right after. */
  readonly warmSamples: readonly number[];
  readonly warmMs: Summary;
};

export type CacheReadOptions = {
  /** The model's container parts to draw from (each larger than every range). */
  readonly parts: readonly FileRef[];
  /** Opens a reader over one part (the hub's `openAsset`); called once per part. */
  readonly open: (part: FileRef) => Promise<AssetRangeReader>;
  readonly batches: readonly number[];
  readonly samples: number;
  readonly warmup: number;
  readonly signal: AbortSignal;
  readonly onProgress: (progress: {
    readonly caseId: string;
    readonly done: number;
    readonly total: number;
  }) => void;
};

type Range = { readonly part: FileRef; readonly offset: number; readonly length: number };

/** The model and weights role whose parts the reads draw from. */
const SELECTION = { model: "26b", weights: ["model"] } as const;

/**
 * The parts of 26B's model container and a reader factory over the hub's cache at `ref`. Fails,
 * without downloading anything, when any file of the selection is not in the cache, and when the
 * source's range reads are not `"seek"` (a scan per read would time the source, not the disk).
 */
export const openCached26b = async (
  ref: HubRepoRef,
  signal: AbortSignal,
): Promise<Pick<CacheReadOptions, "parts" | "open">> => {
  const loaded = await loadManifest(ref, { signal });
  const { missing } = await listCachedAssets(loaded, SELECTION);
  if (missing.length > 0) {
    throw new Error(
      `26B is not fully in the hub's cache (${missing.length} files missing): load it once with the cached source on the Gemma 4 chat page`,
    );
  }
  const container = resolveSelection(loaded.manifest, SELECTION).containers["model"];
  if (container === undefined) throw new Error("26B's selection has no 'model' container");
  const largest = Math.max(...EXPERT_RANGES);
  return {
    parts: container.parts.filter((part) => part.size > largest),
    open: async (part) => {
      const reader = await openAsset(loaded, part, { signal });
      if (reader === undefined) throw new Error("the cached source offers no range reads");
      if (reader.cost !== "seek") {
        throw new Error(`${part.path}: range reads cost "${reader.cost}" here, expected "seek"`);
      }
      return reader;
    },
  };
};

/** xorshift32, so a run's positions are fixed by its sample index (the contents do not matter). */
const generator = (seed: number): (() => number) => {
  let state = seed >>> 0 || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
};

/** The ranges of `experts` experts, each of the four at a random 4-byte-aligned offset of a random part. */
export const drawRanges = (parts: readonly FileRef[], experts: number, seed: number): Range[] => {
  const next = generator(seed);
  return Array.from({ length: experts }, () =>
    EXPERT_RANGES.map((length) => {
      const part = parts[Math.floor(next() * parts.length)];
      if (part === undefined) throw new Error("drawRanges needs at least one part");
      const slots = Math.floor((part.size - length) / 4);
      if (slots < 0) throw new Error(`part ${part.path} is smaller than a ${length}-byte range`);
      return { part, offset: Math.floor(next() * (slots + 1)) * 4, length };
    }),
  ).flat();
};

/**
 * Times `warmup` + `samples` samples per batch: each draws fresh ranges, reads them all at once
 * (first read), then reads the same ranges again (second read). A read that returns another length
 * than asked fails (the reader's own contract, checked again here).
 */
export const runCacheRead = async (options: CacheReadOptions): Promise<CacheReadResult[]> => {
  const { parts, open, batches, samples, warmup, signal, onProgress } = options;
  const readers = new Map<string, Promise<AssetRangeReader>>();
  const readerOf = (part: FileRef): Promise<AssetRangeReader> => {
    const known = readers.get(part.path);
    if (known !== undefined) return known;
    const opened = open(part);
    readers.set(part.path, opened);
    return opened;
  };
  const readAll = async (ranges: readonly Range[]): Promise<number> => {
    const started = performance.now();
    await Promise.all(
      ranges.map(async ({ part, offset, length }) => {
        const bytes = await (await readerOf(part)).read(offset, length);
        if (bytes.byteLength !== length) {
          throw new Error(
            `${part.path}: read ${offset}+${length} returned ${bytes.byteLength} bytes`,
          );
        }
      }),
    );
    return performance.now() - started;
  };
  const results: CacheReadResult[] = [];
  let seed = 1;
  for (const experts of batches) {
    const id = `cache-read-${experts}`;
    const cold: number[] = [];
    const warm: number[] = [];
    for (let index = 0; index < warmup + samples; index += 1) {
      signal.throwIfAborted();
      const ranges = drawRanges(parts, experts, seed);
      seed += 1;
      const first = await readAll(ranges);
      const second = await readAll(ranges);
      if (index >= warmup) {
        cold.push(first);
        warm.push(second);
      }
      onProgress({ caseId: id, done: index + 1, total: warmup + samples });
    }
    results.push({
      id,
      experts,
      bytes: experts * EXPERT_BYTES,
      samples: cold,
      ms: summarize(cold),
      warmSamples: warm,
      warmMs: summarize(warm),
    });
  }
  return results;
};
