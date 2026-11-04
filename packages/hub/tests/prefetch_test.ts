/**
 * Behaviour of the warming surface (`prefetchAssets` — phase 1 of `source.ts` ④). This surface does
 * not hold the bytes, so only three things can be observed: "how many times it went to the
 * network", "how progress was reported" and "how failures came up"; the correctness of the fetched
 * bytes is checked by **the whole-file surface that follows** (`fetchAssets`) being able to read
 * them.
 *
 * The contract of the whole-file surface itself (progress numbers, rate limiting, self-heal,
 * cross-repository references) belongs to `fetch_test.ts` and is not duplicated here. The no-op on
 * the local source side (a source without phase 1) is in `local_test.ts`.
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  fetchAssets,
  type FileRef,
  HubFetchError,
  type LoadedManifest,
  loadManifest,
  ManifestReferenceError,
  prefetchAssets,
  resolveSelection,
  type RetryDiagnostic,
  selectionRefs,
} from "../mod.ts";
import { FETCH_MANIFEST_BYTES, fetchManifest, TOKENIZER } from "./helpers/fixture.ts";
import { sha256Hex } from "./helpers/local.ts";
import { declaredRefs } from "./helpers/selection.ts";
import {
  abortWhileAwaitingResponse,
  createMockFetch,
  hasEntry,
  HUB_URL,
  hubCache,
  MemoryCacheStorage,
  type MockRoutes,
  payloadFor,
  REPO,
  SHA,
} from "./helpers/mock.ts";

const MANIFEST_PATH = "karume.json";

/**
 * The generation a cross-repository reference names (the shape that reads the same path of this
 * repository from another generation — ADR 0038 §7).
 */
const CROSS_REVISION = "89abcdef0123456789abcdef0123456789abcdef";

/** Every file the fixture declares (zero-length parts have no content and are not served). */
const ASSET_PATHS = declaredRefs(fetchManifest).map((ref) => ref.path);

const serveAll = (
  overrides: ReadonlyMap<string, Uint8Array<ArrayBuffer>> = new Map(),
): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, FETCH_MANIFEST_BYTES]]);
  for (const path of ASSET_PATHS) files.set(path, payloadFor(path));
  for (const [path, bytes] of overrides) files.set(path, bytes);
  return files;
};

const resolveUrl = (path: string): string => `${HUB_URL}/${REPO}/resolve/${SHA}/${path}`;

const countCalls = (calls: readonly string[], url: string): number =>
  calls.filter((call) => call === url).length;

/**
 * Changes only the content and keeps the length (so that the sha256 check fires, not the size
 * check).
 */
const tamper = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const copy = new Uint8Array(bytes);
  copy[copy.length - 1] ^= 0xff;
  return copy;
};

/**
 * The file list (= the input of the warming surface). `selectionRefs` drops zero-length parts and
 * returns the list with the same file (an alias such as `style_alias`) folded, in declaration
 * order, so it can be passed as is.
 */
const assetRefs = (loaded: LoadedManifest): FileRef[] => [
  ...selectionRefs(resolveSelection(loaded.manifest)),
];

/**
 * The warming surface's input as a table for the whole-file surface (the key is the path — matched
 * one to one).
 */
const filesOf = (refs: readonly FileRef[]): Record<string, FileRef> =>
  Object.fromEntries(refs.map((ref) => [ref.path, ref]));

/**
 * The manifest is read with a separate mock, and a mock with an empty call record is passed for
 * observing the warming surface.
 */
const prepare = async (
  routes: MockRoutes,
  caches: MemoryCacheStorage,
): Promise<{
  loaded: LoadedManifest;
  refs: FileRef[];
  mock: ReturnType<typeof createMockFetch>;
}> => {
  const loader = createMockFetch(routes);
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: loader.fetch,
      caches,
    },
  );
  return { loaded, refs: assetRefs(loaded), mock: createMockFetch(routes) };
};

/**
 * A wrapper that replaces **only the first** fetch matching a URL predicate with
 * `429 Too Many Requests` (`retry-after: 0`) and delegates to the original `fetch` afterwards (it
 * makes the fetch layer retry exactly once).
 *
 * It always calls the original `fetch` once before replacing — otherwise the request that became a
 * 429 would not stay in the mock's call record, and "whether it fetched again" could not be seen
 * from the request count.
 */
const rateLimitOnce = (
  base: typeof globalThis.fetch,
  matches: (url: string) => boolean,
): typeof globalThis.fetch => {
  let fired = false;
  return async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const response = await base(input, init);
    if (fired || !matches(url)) return response;
    fired = true;
    await response.body?.cancel().catch(() => {});
    return new Response(null, {
      status: 429,
      statusText: "Too Many Requests",
      headers: { "retry-after": "0" },
    });
  };
};

test("prefetchAssets: emits complete once per file, and loaded lands on the total", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const events: AssetProgress[] = [];

  await prefetchAssets(loaded, refs, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  let total = 0;
  for (const ref of refs) total += ref.size;
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(
    completes.length,
    refs.length,
    "the number of complete events does not match the file count",
  );
  assertEquals(
    new Set(completes.map((event) => event.path)),
    new Set(refs.map((ref) => ref.path)),
    "some file never emitted complete",
  );
  for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(
    events[events.length - 1].loaded,
    total,
    "the last event does not land on the total",
  );
  // The phases of one file are downloading* → complete, and nothing comes after complete.
  const lastPhase = new Map<string, AssetPhase>();
  for (const event of events) {
    if (lastPhase.get(event.path) === "complete") {
      throw new Error(`${event.phase} after complete (${event.path})`);
    }
    lastPhase.set(event.path, event.phase);
  }
});

test("prefetchAssets: fetchAssets after the prefetch never goes to the network", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });
  const warmed = mock.calls.length;
  assert(warmed > 0, "prefetch fetched nothing");

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, filesOf(refs), { fetch: second.fetch, caches });

  assertEquals(
    Object.keys(assets),
    refs.map((ref) => ref.path),
  );
  for (const ref of refs) assertEquals(assets[ref.path], payloadFor(ref.path));
  assertEquals(second.calls, [], "the whole-file surface goes to the network after warming");
  assertEquals(mock.calls.length, warmed, "the first mock was called again as well");
});

test("prefetchAssets: a cached file emits a single complete", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  await prefetchAssets(loaded, refs, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  assertEquals(second.calls, [], "goes to the network although the cache is warm");
  assertEquals(
    events.length,
    refs.length,
    "downloading events are mixed in (the list of cache hits)",
  );
  assertEquals(new Set(events.map((event) => event.phase)), new Set<AssetPhase>(["complete"]));
});

test("prefetchAssets: an abort takes effect per file even when every file is cached", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const controller = new AbortController();
  const reason = new Error("app: the user cancelled the load");
  const completes: string[] = [];

  const error = await assertRejects(() =>
    prefetchAssets(loaded, refs, {
      fetch: second.fetch,
      caches,
      signal: controller.signal,
      // Every file is warm, so no downloading appears (a list of single complete events). Cancel
      // right after the first file finishes.
      onProgress: (progress) => {
        completes.push(progress.path);
        controller.abort(reason);
      },
    }),
  );

  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assertEquals(
    second.calls,
    [],
    "goes to the network although the cache is warm (hiding a gap in the abort)",
  );
  // If the abort were checked only in the lower layer (network), it would run through every
  // remaining ref from the cache before settling. The 4 concurrent ones are already in flight and
  // run to the end, so the test checks "does not run through all", not an upper bound.
  assert(
    refs.length > 4,
    `not tried with a list longer than the concurrency (refs ${refs.length})`,
  );
  assert(
    completes.length < refs.length,
    `runs through the rest after the cancel (complete ${completes.length} / ${refs.length})`,
  );
});

test("prefetchAssets: emits no complete after an abort (ends the same way as the whole-file surface)", async () => {
  // The test above only checks "does not run through all", so it lets through the shape where files
  // already in flight before the abort (up to 4) emit complete after it. The whole-file surface
  // fetchAssets checks the abort right after leaving the fetch, so 0 — the two surfaces end their
  // notifications the same way.
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const controller = new AbortController();
  const reason = new Error("app: the user cancelled the load");
  const afterAbort: string[] = [];

  const error = await assertRejects(() =>
    prefetchAssets(loaded, refs, {
      fetch: createMockFetch({ files: serveAll() }).fetch,
      caches,
      signal: controller.signal,
      onProgress: (progress) => {
        if (controller.signal.aborted) afterAbort.push(`${progress.phase} ${progress.path}`);
        else controller.abort(reason);
      },
    }),
  );

  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assert(
    refs.length > 4,
    `not tried with a list longer than the concurrency (refs ${refs.length})`,
  );
  assertEquals(afterAbort, [], "notifications still arrive after the abort");
});

test("prefetchAssets: an abort during a fetch passes through without being wrapped in HubFetchError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const reason = new Error("app: the user cancelled the load");

  const error = await assertRejects(() =>
    prefetchAssets(loaded, refs, {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      // Cancel in the middle of the first file whose reception started (the window on the side that
      // is not the list of cache hits).
      onProgress: (progress) => {
        if (progress.phase === "downloading") controller.abort(reason);
      },
    }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
});

test("prefetchAssets: one failure comes up as HubFetchError with the root cause restored", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { loaded, refs, mock } = await prepare({ files: served, delayMs: 5 }, caches);
  // Fails only one file **other than the first**. One failure aborts all the rest, so the
  // collateral ones settle as bare AbortErrors — picking the error by worker array position would
  // surface a collateral one when the real culprit is not worker[0], and the root cause (which file
  // failed and why) would be lost.
  const victim = refs[1].path;
  assert(refs[0].path !== victim, "the failing file is the first one");
  served.delete(victim);

  const error = await assertRejects(
    () =>
      prefetchAssets(loaded, refs, {
        fetch: abortWhileAwaitingResponse(mock.fetch, victim),
        caches,
      }),
    HubFetchError,
  );
  assertEquals(error.path, victim, "a file other than the one that failed is reported");
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assert(error.message.includes(victim), `${error.message} does not name the failed path`);
});

test("prefetchAssets: a 429 arrives through onRetry, and the re-warmed bytes are correct in the whole-file surface that follows", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const target = resolveUrl(refs[0].path);
  const retries: RetryDiagnostic[] = [];

  await prefetchAssets(loaded, refs, {
    fetch: rateLimitOnce(mock.fetch, (url) => url === target),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });

  assertEquals(retries.length, 1, "the retry notification did not arrive exactly once");
  assertEquals(retries[0].url, target, "the notification names the URL of another file");
  assertEquals(retries[0].status, 429);
  assertEquals(countCalls(mock.calls, target), 2, "did not warm again after the 429");

  // That the re-warmed content is correct is checked by reading it through the whole-file surface
  // without going to the network.
  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, filesOf(refs), { fetch: second.fetch, caches });
  assertEquals(assets[refs[0].path], payloadFor(refs[0].path), "the re-warmed content differs");
  assertEquals(second.calls, [], "not fully warmed (the whole-file surface fetches again)");
});

test("prefetchAssets: a sha256 mismatch fails loudly and leaves no entry in the cache", async () => {
  const caches = new MemoryCacheStorage();
  const path = TOKENIZER;
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[path, tamper(payloadFor(path))]])) },
    caches,
  );
  // Rules out collateral failures of parallel workers and looks only at "the one that failed".
  const only = refs.filter((ref) => ref.path === path);
  assertEquals(only.length, 1);

  const error = await assertRejects(
    () => prefetchAssets(loaded, only, { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assert(error.cause instanceof Error, "keeps the lower layer's mismatch as cause");
  assertEquals(
    hasEntry(hubCache(caches), tamper(payloadFor(path))),
    false,
    "the mismatching bytes remain in the cache",
  );
});

test("prefetchAssets: where the cache cannot be opened, fails loudly instead of degrading to plain fetch", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  // This surface does not hold the bytes, so when the cache is unusable it cannot silently degrade
  // into "thinks it warmed, but nothing is left" (degrading would make the following reads go to
  // the network silently).
  const broken = {
    open: () => Promise.reject(new Error("cache storage open failure")),
  } as unknown as CacheStorage;

  const error = await assertRejects(
    () => prefetchAssets(loaded, refs, { fetch: mock.fetch, caches: broken }),
    HubFetchError,
  );
  assert(error.cause instanceof Error, "keeps the lower layer's failure as cause");
  assertEquals(mock.calls, [], "the main download runs although the cache cannot be opened");
});

test("prefetchAssets: empty or duplicate input is a ManifestReferenceError before going to the network", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  const empty = await assertRejects(
    () => prefetchAssets(loaded, [], { fetch: mock.fetch, caches }),
    ManifestReferenceError,
  );
  assert(empty.message.includes("prefetchAssets"), `${empty.message} does not name the surface`);
  assertEquals(empty.available.models, ["anima-turbo", "anima-lite"]);

  const duplicated = await assertRejects(
    () => prefetchAssets(loaded, [refs[0], refs[0]], { fetch: mock.fetch, caches }),
    ManifestReferenceError,
  );
  assert(
    duplicated.message.includes(refs[0].path),
    `${duplicated.message} does not name the duplicate path`,
  );
  assertEquals(mock.calls, [], "goes to the network before the input check");
});

test("prefetchAssets: duplicates are judged by fileRefKey (the same path with a different revision is a separate file)", async () => {
  const caches = new MemoryCacheStorage();
  const path = ASSET_PATHS[0];
  // Serves the same path from another generation too. The content is changed — with identical
  // bytes, one of them would be a cache hit through the fetch layer's content key, and "whether it
  // fetched both" could not be seen from the request count.
  const crossBytes = payloadFor(`${CROSS_REVISION}/${path}`);
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[`${REPO}@${CROSS_REVISION}/${path}`, crossBytes]])) },
    caches,
  );
  const own = refs.find((ref) => ref.path === path);
  assert(own !== undefined, `${path} is not in the selection's declaration`);
  const cross: FileRef = {
    path,
    size: crossBytes.byteLength,
    sha256: await sha256Hex(crossBytes),
    repo: REPO,
    revision: CROSS_REVISION,
  };

  // An implementation that folds duplicates by path would hit a ManifestReferenceError here.
  await prefetchAssets(loaded, [own, cross], { fetch: mock.fetch, caches });

  assertEquals(countCalls(mock.calls, resolveUrl(path)), 1);
  assertEquals(
    countCalls(mock.calls, `${HUB_URL}/${REPO}/resolve/${CROSS_REVISION}/${path}`),
    1,
    "the cross-repository reference was folded into the same-repository reference and not warmed",
  );
});
