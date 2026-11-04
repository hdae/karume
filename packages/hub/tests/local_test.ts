/**
 * Behaviour of the local source (`localDirectory`). Its difference from the HF source lies in "the
 * capabilities it does not have", so the tests mostly **observe absences** — it names no
 * generation / does not open CacheStorage / has no phase 1 / does not compare sha256 (looks only
 * at size) / does not guess the referenced repository.
 *
 * To exercise a remote referenced repository, a fake source (directly implementing the internal
 * contract of `DistributionSource`) is injected into crossRepo — there is no public factory that
 * builds an HF source handle.
 */

import { beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  fetchAssets,
  HubFetchError,
  IntegrityError,
  loadManifest,
  localDirectory,
  ManifestFormatError,
  ManifestReferenceError,
  openAsset,
  prefetchAssets,
} from "../mod.ts";
import type {
  AssetProgress,
  DirectoryAdapter,
  LoadedManifest,
  LocalDirectoryOptions,
} from "../mod.ts";
import { type FileRef, MANIFEST_FILENAME } from "../src/manifest.ts";
import {
  DistributionSource,
  driverOf,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../src/source.ts";
import {
  buildLocalDist,
  CROSS_PATH,
  CROSS_REPO,
  CROSS_REVISION,
  FETCHED_PART_PATHS,
  memoryDirectory,
  type MemoryDirectory,
  PART_KEYS,
  TOKENIZER_PATH,
} from "./helpers/local.ts";
import { selectionFiles } from "./helpers/selection.ts";
import { MemoryCacheStorage, payloadFor } from "./helpers/mock.ts";

const LABEL = "./models/karume-test";

/**
 * A CacheStorage that fails the moment it is touched. The check that the local source never opens
 * it.
 */
class HostileCacheStorage implements CacheStorage {
  open(): Promise<Cache> {
    throw new Error("test: the local source opened CacheStorage");
  }
  match(): Promise<Response | undefined> {
    throw new Error("test: the local source looked up CacheStorage");
  }
  has(): Promise<boolean> {
    throw new Error("test: the local source looked up CacheStorage");
  }
  delete(): Promise<boolean> {
    throw new Error("test: the local source deleted from CacheStorage");
  }
  keys(): Promise<string[]> {
    throw new Error("test: the local source looked up CacheStorage");
  }
}

const openLocal = async (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: Parameters<typeof localDirectory>[1] = {},
) => {
  const directory = memoryDirectory(files);
  const loaded = await loadManifest(
    localDirectory(directory.adapter, { label: LABEL, ...options }),
    { caches: new HostileCacheStorage() },
  );
  return { directory, loaded };
};

/**
 * A progress record (phase and path only — the semantics of the numbers belong to `stream_test` /
 * `fetch_test`).
 */
const recordProgress = (): { events: AssetProgress[]; onProgress: (p: AssetProgress) => void } => {
  const events: AssetProgress[] = [];
  return { events, onProgress: (progress) => events.push(progress) };
};

/**
 * Orders [phase, path] pairs element-wise in UTF-16 code-unit order (only a total order is needed).
 */
const comparePairs = (a: readonly string[], b: readonly string[]): number => {
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] < b[index]) return -1;
    if (a[index] > b[index]) return 1;
  }
  return a.length - b.length;
};

test("localDirectory: manifest → resolveSelection → fetchAssets runs to the end with the actual bytes", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);

  // The source has neither a generation nor a repo, so the identifying fields **do not appear at
  // all** (no fake SHA).
  assertEquals(loaded.repo, undefined);
  assertEquals(loaded.revisionSha, undefined);
  assertEquals(loaded.manifest.available.models, ["m"]);

  const files = selectionFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files);
  assertEquals(Object.keys(assets).sort(), [...PART_KEYS, "tokenizer"].sort());
  PART_KEYS.forEach((key, index) => {
    const part = FETCHED_PART_PATHS[index];
    assertEquals(assets[key], payloadFor(part), `the actual bytes of ${part} were not delivered`);
  });
  assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
  // Only the manifest + 3 assets were read (zero-length parts are not read; the same path is not
  // read twice).
  assertEquals(directory.reads, [MANIFEST_FILENAME, ...FETCHED_PART_PATHS, TOKENIZER_PATH]);
});

test("localDirectory: an abort reaches the asset reads", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);
  await fetchAssets(loaded, selectionFiles(loaded.manifest), {
    signal: AbortSignal.timeout(60_000),
  });
  // The manifest comes from loadManifest's call without a signal, the assets from the surface's
  // call with a signal.
  assertEquals(directory.signals, [false, true, true, true]);
});

test("localDirectory: fetchAssets reads each file only once and never opens CacheStorage", async () => {
  const dist = await buildLocalDist();
  const caches = new MemoryCacheStorage();
  const directory = memoryDirectory(dist.files);
  const loaded = await loadManifest(localDirectory(directory.adapter, { label: LABEL }), {
    caches,
  });
  const selected = selectionFiles(loaded.manifest);
  const files = Object.fromEntries(PART_KEYS.map((key): [string, FileRef] => [key, selected[key]]));
  const progress = recordProgress();

  const assets = await fetchAssets(loaded, files, { onProgress: progress.onProgress });

  PART_KEYS.forEach((key, index) => {
    assertEquals(assets[key], payloadFor(FETCHED_PART_PATHS[index]));
  });
  // Warming is skipped entirely, so each file is read only once (not warmed and then read again).
  assertEquals(directory.reads, [MANIFEST_FILENAME, ...FETCHED_PART_PATHS]);
  // There is no "in the middle of receiving" state, so progress is a single complete per file.
  assertEquals(
    progress.events.map((event) => event.phase),
    ["complete", "complete"],
  );
  // Files of a source with neither a repo nor a generation do not emit the fetch-origin fields
  // **at all** (no synthesized name or fake SHA — the same rule as `loaded.repo` / `revisionSha`).
  for (const event of progress.events) {
    assertEquals(Object.hasOwn(event, "repo"), false, `${event.path} names a repo`);
    assertEquals(Object.hasOwn(event, "revision"), false, `${event.path} names a revision`);
  }
  assertEquals(
    caches.namespaces.size,
    0,
    "the local source creates a cache namespace (duplicating the bytes)",
  );
});

describe("prefetchAssets: a source without phase 1 emits only complete without warming (the input check still applies)", () => {
  let directory: MemoryDirectory;
  let loaded: LoadedManifest;
  let refs: FileRef[];

  beforeAll(async () => {
    const dist = await buildLocalDist();
    ({ directory, loaded } = await openLocal(dist.files));
    const files = selectionFiles(loaded.manifest);
    refs = PART_KEYS.map((key): FileRef => files[key]);
  });

  it("having no cache to warm is not a failure (returns without reading the files)", async () => {
    const progress = recordProgress();
    directory.reads.length = 0;
    await prefetchAssets(loaded, refs, { onProgress: progress.onProgress });
    assertEquals(directory.reads, [], "reads the files although there is no phase 1");
    // A ref that needs no fetch also emits complete once each (ADR 0086 addendum 2026-10-11) — a
    // caller whose progress uses the size total of the refs passed as denominator reaches the total
    // regardless of the source.
    assertEquals(
      progress.events.map((event) => [event.phase, event.path]).sort(comparePairs),
      refs.map((ref) => ["complete", ref.path]).sort(comparePairs),
      "a ref that needs no warming does not emit complete once each",
    );
    const total = refs.reduce((sum, ref) => sum + ref.size, 0);
    assertEquals(
      progress.events.at(-1)?.loaded,
      total,
      "the last notification does not reach the total",
    );
    for (const event of progress.events) assertEquals(event.fileLoaded, event.fileTotal);
  });

  it("a caller error (empty refs) fails before touching the source", async () => {
    await assertRejects(() => prefetchAssets(loaded, []), ManifestReferenceError, LABEL);
  });
});

test("prefetchAssets: phase 1 is per ref — when the referenced repository has it, only the cross-repository files are warmed", async () => {
  // Deciding by the session's source alone would skip warming the cross-repository files entirely
  // in the valid setup of a local distribution form + a remote referenced repository (neither
  // progress nor signal arrives, and the actual fetch runs later inside the read).
  const dist = await buildLocalDist({ cross: true });
  const warmed: string[] = [];
  const { directory, loaded } = await openLocal(dist.files, {
    crossRepo: { [CROSS_REPO]: fakeRemote(dist.crossFiles, { warmed }) },
  });
  const files = selectionFiles(loaded.manifest);
  const refs = [...PART_KEYS.map((key): FileRef => files[key]), files["text_encoder"]];
  directory.reads.length = 0;
  const progress = recordProgress();

  await prefetchAssets(loaded, refs, { onProgress: progress.onProgress });

  assertEquals(warmed, [CROSS_PATH], "the cross-repository ref was not warmed");
  // Progress follows one rule (every ref emits complete once) — the warmed cross-repository ref and
  // the local ref that needs no warming both emit it (ADR 0086 addendum 2026-10-11).
  assertEquals(
    progress.events.map((event) => [event.phase, event.path]).sort(comparePairs),
    refs.map((ref) => ["complete", ref.path]).sort(comparePairs),
    "the complete events per ref are not all there",
  );
  // Even in a local session, the cross-repository files name the declared referenced repository
  // (if the fields dropped because the session names nothing, they could not be told apart from the
  // same path in another repository). The local refs do not emit the fetch-origin fields at all.
  assertEquals(
    progress.events
      .map((event) => [event.path, event.repo, event.revision])
      .filter(([, repo]) => repo !== undefined),
    [[CROSS_PATH, CROSS_REPO, CROSS_REVISION]],
  );
  for (const event of progress.events) {
    if (event.path !== CROSS_PATH) {
      assertEquals(Object.hasOwn(event, "repo"), false, `${event.path} names a repo`);
    }
  }
  const total = refs.reduce((sum, ref) => sum + ref.size, 0);
  assertEquals(
    progress.events.at(-1)?.loaded,
    total,
    "the last notification does not reach the total",
  );
  assertEquals(directory.reads, [], "reads a ref of a source without phase 1");
});

test("localDirectory: a size mismatch fails loudly (naming the path and the directory)", async () => {
  const dist = await buildLocalDist();
  // Replace only the file (the manifest's size / sha256 unchanged = the shape of a truncated copy).
  const tampered = new Map(dist.files);
  tampered.set(TOKENIZER_PATH, payloadFor(`${TOKENIZER_PATH}:truncated`));
  const { loaded } = await openLocal(tampered);

  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
    IntegrityError,
  );
  assertEquals(error.path, TOKENIZER_PATH);
  assertEquals(error.source, "local", "the local file read names network / cache");
  assertEquals(error.repo, undefined, "names a repo that does not exist");
  assertEquals(error.revisionSha, undefined, "names a fake SHA");
  assert(error.message.includes(TOKENIZER_PATH), `${error.message} does not name the path`);
  assert(error.message.includes(LABEL), `${error.message} does not name the directory`);
});

test("localDirectory: sha256 is not compared (a matching size reads to the end)", async () => {
  const dist = await buildLocalDist();
  const original = payloadFor(TOKENIZER_PATH);
  // Change only the content and keep the length (the shape that would fail the sha256 check with
  // the HF source).
  const swapped = new Uint8Array(original);
  swapped[swapped.length - 1] ^= 0xff;
  const tampered = new Map(dist.files);
  tampered.set(TOKENIZER_PATH, swapped);
  const { loaded } = await openLocal(tampered);

  const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest));
  assertEquals(
    assets["tokenizer"],
    swapped,
    "departs from the design that trusts the local files (size is the only check)",
  );
});

test("localDirectory: a missing file is a HubFetchError keeping the actual path in cause", async () => {
  const dist = await buildLocalDist();
  const missing = new Map(dist.files);
  missing.delete(FETCHED_PART_PATHS[1]);
  const { loaded } = await openLocal(missing);

  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
    HubFetchError,
  );
  assertEquals(error.path, FETCHED_PART_PATHS[1]);
  assertEquals(error.repo, undefined);
  assert(error.message.includes(LABEL), `${error.message} does not name the directory`);
  assert(error.cause instanceof Error, "the adapter's error is not kept as cause");
  assert(
    error.cause.message.includes(FETCHED_PART_PATHS[1]),
    `${error.cause.message} does not name the file that could not be read`,
  );
});

test("localDirectory: a karume.json over the limit is a ManifestFormatError", async () => {
  const oversized = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_FILENAME, new TextEncoder().encode(" ".repeat(2 * 1024 * 1024))],
  ]);
  const directory = memoryDirectory(oversized);
  const error = await assertRejects(
    () => loadManifest(localDirectory(directory.adapter, { label: LABEL })),
    ManifestFormatError,
  );
  assert(error.message.includes(LABEL), `${error.message} does not name the directory`);
});

test("localDirectory: a broken karume.json fails with the same ManifestFormatError every time", async () => {
  const broken = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_FILENAME, new TextEncoder().encode('{"format": "karume/5"')],
  ]);
  const directory = memoryDirectory(broken);
  const source = localDirectory(directory.adapter, { label: LABEL });
  // Locally there is no cache to evict, so there is neither self-heal nor "fetch again once".
  for (const _attempt of [0, 1]) {
    await assertRejects(() => loadManifest(source), ManifestFormatError);
  }
  assertEquals(directory.reads, [MANIFEST_FILENAME, MANIFEST_FILENAME]);
});

// ---- Cross-repository references (ADR 0038 §7): the 2 shapes, explicit mapping / not in the
// mapping (there is no field for a delegate).

/**
 * A fake source that only provides the referenced repository (standing in for the HF source).
 * Passing `shortBy` makes it a **remote** source that reports fewer bytes than the declared `size`
 * (the shape that exercises a failed integrity check in the referenced repository). Passing
 * `warmed` makes it a source **with phase 1** that records the warmed paths there.
 */
const fakeRemote = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: { readonly shortBy?: number; readonly warmed?: string[] } = {},
): DistributionSource => {
  const pinnedFor = (repo: string, revision: string): PinnedSource => {
    const origin: SourceOrigin = {
      label: `fake ${repo} @ ${revision}`,
      integrity: "network",
      repo,
      revisionSha: revision,
    };
    return {
      origin,
      readManifest: () => Promise.reject(new Error("fake: the manifest is not read")),
      readFile: (ref, { sizeViolation }) => {
        const bytes = files.get(ref.path);
        if (bytes === undefined) return Promise.reject(new Error(`fake: ${ref.path} missing`));
        const { shortBy } = options;
        // The source returns to the shared layer naming its own failure origin (`origin.integrity`)
        // — assembling the error is the shared layer's job (`context.ts`).
        if (shortBy !== undefined) {
          return Promise.reject(sizeViolation(bytes.byteLength - shortBy, origin.integrity));
        }
        return Promise.resolve(new Uint8Array(bytes));
      },
      // Phase 1 is an optional capability per source (`source.ts` ④). Only a source that has it
      // grows this member.
      ...(options.warmed === undefined
        ? {}
        : {
            prefetchFile: (ref: FileRef): Promise<void> => {
              options.warmed?.push(ref.path);
              return Promise.resolve();
            },
          }),
      originFor: (crossRepo, crossRevision) => pinnedFor(crossRepo, crossRevision),
    };
  };
  const driver: SourceDriver = {
    origin: { label: "fake remote", integrity: "network" },
    resolveGeneration: () => Promise.resolve("fake"),
    pin: (generation) => pinnedFor("fake/session", generation),
  };
  return new DistributionSource(driver);
};

const crossAssets = async (
  options: Parameters<typeof localDirectory>[1],
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const dist = await buildLocalDist({ cross: true });
  const { loaded } = await openLocal(dist.files, options);
  return await fetchAssets(loaded, selectionFiles(loaded.manifest));
};

test("localDirectory: a cross-repository reference is fetched from the source of the explicit mapping", async () => {
  const dist = await buildLocalDist({ cross: true });
  const crossDirectory = memoryDirectory(dist.crossFiles);
  const assets = await crossAssets({
    crossRepo: {
      [CROSS_REPO]: localDirectory(crossDirectory.adapter, { label: "./models/shared" }),
    },
  });
  assertEquals(assets["text_encoder"], payloadFor(`${CROSS_REPO}/${CROSS_PATH}`));
  assertEquals(crossDirectory.reads, [CROSS_PATH], "not fetched from the referenced repository");
});

test("localDirectory: a delegate (fallback) field for cross-repository references is not accepted — passing one fails without delegating", async () => {
  const dist = await buildLocalDist({ cross: true });
  const crossDirectory = memoryDirectory(dist.crossFiles);
  // The type check: if the field came back to the public surface, the `@ts-expect-error` would be
  // unused and `deno check` would turn red (no value buildable through the public surface works as
  // a delegate — the NOTE on `LocalDirectoryOptions.crossRepo`).
  const options: LocalDirectoryOptions = {
    // @ts-expect-error there is no delegate field in LocalDirectoryOptions
    fallback: localDirectory(crossDirectory.adapter, { label: "./models/shared" }),
  };

  // Even when a caller that bypasses the types (JS) passes it, it does not delegate silently and
  // fails as a missing mapping.
  const error = await assertRejects(() => crossAssets(options), HubFetchError);
  assert(error.cause instanceof Error, "the reason for the missing setting is not kept as cause");
  assert(
    !error.cause.message.includes("fallback"),
    `${error.cause.message} points to a field that does not exist`,
  );
  assertEquals(crossDirectory.reads, [], "read the delegate that should not be accepted");
});

test("localDirectory: a cross-repository reference not in the mapping fails without guessing", async () => {
  const error = await assertRejects(() => crossAssets({}), HubFetchError);
  assertEquals(error.path, CROSS_PATH);
  // The diagnostic names **the declared referenced repository** (not the session's directory).
  assertEquals(error.repo, CROSS_REPO);
  assertEquals(error.revisionSha, CROSS_REVISION);
  assert(error.cause instanceof Error, "the reason for the missing setting is not kept as cause");
  assert(
    error.cause.message.includes("crossRepo") && error.cause.message.includes(CROSS_REPO),
    `${error.cause.message} does not name the missing setting`,
  );
});

test("localDirectory: a failed integrity check in the referenced repository names the referenced repository's failure origin", async () => {
  const dist = await buildLocalDist({ cross: true });

  const error = await assertRejects(
    () => crossAssets({ crossRepo: { [CROSS_REPO]: fakeRemote(dist.crossFiles, { shortBy: 1 }) } }),
    IntegrityError,
  );

  // The diagnostic names the declared referenced repository (label / repo / revisionSha as in the
  // existing checks).
  assertEquals(error.path, CROSS_PATH);
  assertEquals(error.repo, CROSS_REPO);
  assertEquals(error.revisionSha, CROSS_REVISION);
  assertEquals(error.actual, String(payloadFor(`${CROSS_REPO}/${CROSS_PATH}`).byteLength - 1));
  // A local session + a remote referenced repository (a setup `crossRepo` officially accepts). The
  // bytes came from the network, so retrying makes sense — inheriting the session's classification
  // "local" (fetching again gives the same result) would tell the app the lie that "there is no way
  // to recover".
  assertEquals(error.source, "network");
});

test("localDirectory: sources of referenced repositories do not mix even when the same handle is reused", async () => {
  const dist = await buildLocalDist({ cross: true });
  const crossDirectory = memoryDirectory(dist.crossFiles);
  const shared = localDirectory(crossDirectory.adapter, { label: "./models/shared" });
  const first = await crossAssets({ crossRepo: { [CROSS_REPO]: shared } });
  const second = await crossAssets({ crossRepo: { [CROSS_REPO]: shared } });
  assertEquals(first["text_encoder"], second["text_encoder"]);
  // It does not look up the session's path of the same name (`text_encoder/...` does not exist in
  // this repository).
  assertEquals(crossDirectory.reads, [CROSS_PATH, CROSS_PATH]);
});

test("localDirectory: the source handle is opaque (loadManifest tells it apart by identity)", async () => {
  const dist = await buildLocalDist();
  const directory = memoryDirectory(dist.files);
  const source = localDirectory(directory.adapter, { label: LABEL });
  // No member is exposed on the public surface (the implementation is confined to a table inside
  // the hub — `source.ts`).
  assertEquals(Object.keys(source), []);
  assertEquals(Reflect.ownKeys(source).length, 0);
  const first = await loadManifest(source);
  const second = await loadManifest(source);
  // However many sessions are opened from the same handle, the source's state is not shared (it
  // reads again every time).
  assertStrictEquals(first === second, false);
  assertEquals(directory.reads, [MANIFEST_FILENAME, MANIFEST_FILENAME]);
});

// ---- Range reads (`openAsset` — `source.ts` ⑧). With the local source, a reader exists **only
// when the adapter has positional reads**. The source's responsibilities are two, "checking
// against the declared size" and "not letting a short return through"; how the file is read
// (`Deno.open` → `seek`) belongs to `deno_directory_test.ts`.

/**
 * A recording positional read. Passing `short` makes an adapter that returns 1 byte less than
 * requested; passing `loose` makes one that returns a view into the middle of a padded buffer
 * (a non-tight view) with correct content.
 */
type RangeDirectoryOptions = { readonly short?: boolean; readonly loose?: boolean };

const rangeDirectory = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: RangeDirectoryOptions = {},
) => {
  const base = memoryDirectory(files);
  /** The requests that reached `readFileRange` (path / offset / length). */
  const ranges: [string, number, number][] = [];
  const adapter: DirectoryAdapter = {
    ...base.adapter,
    readFileRange: (path, offset, length) => {
      ranges.push([path, offset, length]);
      const bytes = files.get(path);
      if (bytes === undefined) {
        return Promise.reject(new Error(`test-directory: cannot read ${path}`));
      }
      const served = options.short === true ? Math.max(length - 1, 0) : length;
      const slice = bytes.subarray(offset, offset + served);
      if (options.loose !== true) return Promise.resolve(new Uint8Array(slice));
      // A view copied into the middle of a padded buffer (content and length are correct — the only
      // reason to fail is the shape of the view).
      const padded = new Uint8Array(new ArrayBuffer(slice.byteLength + 8));
      padded.set(slice, 4);
      return Promise.resolve(padded.subarray(4, 4 + slice.byteLength));
    },
  };
  return { ...base, adapter, ranges };
};

const openRangeLocal = async (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: Parameters<typeof localDirectory>[1] = {},
  directoryOptions: RangeDirectoryOptions = {},
) => {
  const directory = rangeDirectory(files, directoryOptions);
  const loaded = await loadManifest(
    localDirectory(directory.adapter, { label: LABEL, ...options }),
    { caches: new HostileCacheStorage() },
  );
  return { directory, loaded };
};

test("openAsset: undefined for an adapter without positional reads (falls back to whole reads)", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);
  const ref = selectionFiles(loaded.manifest)["tokenizer"];
  directory.reads.length = 0;

  // This is a difference in capability, not a failure (failing loudly on a missing reader would
  // force an app that should be able to swap sources to branch per source).
  assertEquals(await openAsset(loaded, ref), undefined);
  assertEquals(directory.reads, [], "opening a reader reads the file");
});

test("openAsset: an adapter with positional reads opens a seek reader, and only the requested range goes down", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openRangeLocal(dist.files);
  const ref = selectionFiles(loaded.manifest)["tokenizer"];
  directory.reads.length = 0;

  const reader = await openAsset(loaded, ref);
  assert(
    reader !== undefined,
    "the reader does not open although the adapter has positional reads",
  );
  assertEquals(reader.cost, "seek");
  assertEquals(directory.ranges, [], "opening alone reads the file");

  const bytes = await reader.read(4, 3);
  assertEquals(bytes, payloadFor(TOKENIZER_PATH).subarray(4, 7).slice());
  // Not falling back to whole reads (`readFile`) = the surface that reads only rows works.
  assertEquals(directory.ranges, [[TOKENIZER_PATH, 4, 3]]);
  assertEquals(directory.reads, [], "range reads fell back to whole reads");
});

test("openAsset: a range outside the declared size fails without calling the adapter once", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openRangeLocal(dist.files);
  const ref = selectionFiles(loaded.manifest)["tokenizer"];
  const reader = await openAsset(loaded, ref);
  assert(reader !== undefined, "the reader does not open");

  // The file can be longer than declared (a mixed-up quant), so the check uses the declared size,
  // not the file length.
  const error = await assertRejects(() => reader.read(ref.size - 1, 2), Error);
  assert(error.message.includes(TOKENIZER_PATH), `${error.message} does not name the path`);
  assert(
    error.message.includes(String(ref.size)),
    `${error.message} does not name the declared size`,
  );
  assertEquals(directory.ranges, [], "an out-of-range request reached the adapter");
});

test("local source range reader: checks the declared size itself, before the adapter", async () => {
  // Driven through the source directly — through `openAsset`, the common layer's check (`fetch.ts`)
  // rejects first, so this source's own check would never be reached.
  const dist = await buildLocalDist();
  const directory = rangeDirectory(dist.files);
  const source = localDirectory(directory.adapter, { label: LABEL });
  const loaded = await loadManifest(source, { caches: new HostileCacheStorage() });
  // A declaration shorter than the file: the bound is the declared size, not the file length.
  const declared = { ...selectionFiles(loaded.manifest)["tokenizer"] };
  const ref: FileRef = { ...declared, size: declared.size - 1 };
  const openFile = driverOf(source).pin("", {}).openFile;
  assert(openFile !== undefined, "an adapter with positional reads opens no range reader");
  const reader = await openFile(ref, {});

  await assertRejects(
    () => reader.read(ref.size - 1, 2),
    Error,
    `hub: ${TOKENIZER_PATH} range [${ref.size - 1}, ${ref.size + 1}) is invalid` +
      ` (declared size ${ref.size} — only ranges of non-negative integers within size can be read)`,
  );
  for (const [offset, length] of [
    [ref.size, 1],
    [-1, 1],
    [0, 0.5],
    [Number.NaN, 1],
    [0, Number.POSITIVE_INFINITY],
  ]) {
    await assertRejects(() => reader.read(offset, length), Error, "declared size");
  }
  assertEquals(directory.ranges, [], "an out-of-range request reached the adapter");
  // The whole declared range is still readable.
  assertEquals(await reader.read(0, ref.size), payloadFor(TOKENIZER_PATH).slice(0, ref.size));
});

test("openAsset: a short return throws (zero-filled rows are not delivered as valid values)", async () => {
  const dist = await buildLocalDist();
  const { loaded } = await openRangeLocal(dist.files, {}, { short: true });
  const ref = selectionFiles(loaded.manifest)["tokenizer"];
  const reader = await openAsset(loaded, ref);
  assert(reader !== undefined, "the reader does not open");

  const error = await assertRejects(() => reader.read(0, 8), Error);
  assert(error.message.includes(TOKENIZER_PATH), `${error.message} does not name the path`);
  assert(error.message.includes("7"), `${error.message} does not name the actual byte count`);
});

test("openAsset: an adapter that returns a non-tight view fails loudly (padded views are not delivered)", async () => {
  const dist = await buildLocalDist();
  const { loaded } = await openRangeLocal(dist.files, {}, { loose: true });
  const ref = selectionFiles(loaded.manifest)["tokenizer"];
  const reader = await openAsset(loaded, ref);
  assert(reader !== undefined, "the reader does not open");

  // The consumer reads the returned bytes as a TypedArray as is, so a padded view shifts the
  // values. The check lives in the shared layer (`fetch.ts`) — placing it per source would multiply
  // copies of the same invariant.
  const error = await assertRejects(() => reader.read(0, 8), Error);
  assert(
    error.message.includes("do not span the whole buffer"),
    `${error.message} does not name the tight-view violation`,
  );
  assert(error.message.includes(TOKENIZER_PATH), `${error.message} does not name the path`);
});

test("openAsset: a cross-repository reference resolves through the referenced source's reader", async () => {
  const dist = await buildLocalDist({ cross: true });
  // The session has no positional reads and only the referenced repository does (capabilities
  // differ per source).
  const cross = rangeDirectory(dist.crossFiles);
  const { loaded } = await openLocal(dist.files, {
    crossRepo: { [CROSS_REPO]: localDirectory(cross.adapter, { label: "./models/shared" }) },
  });
  const files = selectionFiles(loaded.manifest);

  assertEquals(
    await openAsset(loaded, files["tokenizer"]),
    undefined,
    "this repository's reader exists",
  );
  const reader = await openAsset(loaded, files["text_encoder"]);
  assert(reader !== undefined, "the referenced repository's reader does not open");
  assertEquals(reader.cost, "seek");

  // Read from the referenced repository's file (the same path string, but bytes different from the
  // session's).
  const bytes = await reader.read(0, 5);
  assertEquals(bytes, payloadFor(`${CROSS_REPO}/${CROSS_PATH}`).subarray(0, 5).slice());
  assertEquals(cross.ranges, [[CROSS_PATH, 0, 5]]);
});
