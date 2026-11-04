import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  type CacheDiagnostic,
  clearHubCache,
  type DirectoryAdapter,
  fetchAssets,
  type FileRef,
  HubError,
  HubFetchError,
  type HubRepoRef,
  type LoadedManifest,
  loadManifest,
  localDirectory,
  ManifestFormatError,
  openAsset,
  prefetchAssets,
  type RetryDiagnostic,
} from "../mod.ts";
import { fetchAssetsWithBudget } from "../src/fetch.ts";
import {
  FETCH_MANIFEST_BYTES,
  fetchManifest,
  STYLE_VECTORS,
  TEXT_CONDITIONER_PARTS,
  TEXT_ENCODER_PARTS,
  TOKENIZER,
  TRANSFORMER_F16_PARTS,
  TRANSFORMER_I8_FETCHED,
  VAE_DECODER_PARTS,
} from "./helpers/fixture.ts";
import { declaredRefs, selectionFiles } from "./helpers/selection.ts";
import {
  createMockFetch,
  hasEntry,
  HUB_URL,
  hubCache,
  MemoryCacheStorage,
  type MockFetch,
  type MockRoutes,
  overwriteEntry,
  payloadFor,
  REPO,
  SHA,
  swapRecords,
  withChromeAbortShape,
} from "./helpers/mock.ts";

const MANIFEST_PATH = "karume.json";

/** Every file the fixture declares (zero-length parts have no content and are not served). */
const ASSET_PATHS = declaredRefs(fetchManifest).map((ref) => ref.path);

const manifestBytes = FETCH_MANIFEST_BYTES;

const serveAll = (
  overrides: ReadonlyMap<string, Uint8Array<ArrayBuffer>> = new Map(),
): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, manifestBytes]]);
  for (const path of ASSET_PATHS) files.set(path, payloadFor(path));
  for (const [path, bytes] of overrides) files.set(path, bytes);
  return files;
};

const resolveUrl = (path: string): string => `${HUB_URL}/${REPO}/resolve/${SHA}/${path}`;
const revisionUrl = (ref: string): string => `${HUB_URL}/api/models/${REPO}/revision/${ref}`;

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

/**
 * Replaces `console.warn`, runs `body` and always restores it (returns the emitted messages as is).
 */
const captureWarnings = async (body: () => Promise<void>): Promise<string[]> => {
  const original = console.warn;
  const warnings: string[] = [];
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map((arg) => String(arg)).join(" "));
  };
  try {
    await body();
  } finally {
    console.warn = original;
  }
  return warnings;
};

test("loadManifest: a mutable ref is resolved only once, and later fetches are pinned to the same SHA", async () => {
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  const caches = new MemoryCacheStorage();
  // This path passes no revision, so the pin hint is emitted (its wording is checked by a dedicated
  // test).
  let loaded!: LoadedManifest;
  await captureWarnings(async () => {
    loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL }, { fetch: mock.fetch, caches });
  });
  assertEquals(loaded.revisionSha, SHA);
  assertEquals(loaded.repo, REPO);
  assertEquals(mock.calls[0], revisionUrl("main"));
  assertEquals(mock.calls[1], resolveUrl(MANIFEST_PATH));

  await fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches });
  assertEquals(
    countCalls(mock.calls, revisionUrl("main")),
    1,
    "resolution happens once per session",
  );
  for (const call of mock.calls.slice(1)) {
    assert(call.includes(`/resolve/${SHA}/`), `${call} is not pinned to the resolved SHA`);
  }
});

describe("loadManifest: warns once only for resolving main without a revision", () => {
  const load = async (revision?: string): Promise<string[]> => {
    const mock = createMockFetch({ sha: SHA, files: serveAll() });
    const caches = new MemoryCacheStorage();
    return await captureWarnings(async () => {
      await loadManifest(
        {
          repo: REPO,
          hubUrl: HUB_URL,
          ...(revision === undefined ? {} : { revision }),
        },
        { fetch: mock.fetch, caches },
      );
    });
  };

  it("without a revision it is emitted once, with the resolved SHA and the two options", async () => {
    const warnings = await load();
    assertEquals(warnings.length, 1, "the warning is missing or emitted more than once");
    const [warning] = warnings;
    // A form that becomes a pin when copied as is (without the SHA in the text the hint could not
    // be followed).
    assert(warning.includes(SHA), `${warning} does not print the resolved commit SHA`);
    assert(warning.includes(REPO), `${warning} does not say which repository it is about`);
    assert(warning.includes(`revision: "${SHA}"`), `${warning} does not show how to write the pin`);
    // The bare table name on a subpath (0.15 spelling), not a prefixed `<FAMILY>_SOURCES`.
    assert(
      /(?<![\w.])SOURCES\[/.test(warning) && warning.includes("@karume/models/"),
      `${warning} does not point to the models table`,
    );
  });

  it("not emitted for an explicit 'main' (a statement that a mutable ref is fine)", async () => {
    assertEquals(await load("main"), []);
  });

  it("not emitted for a SHA", async () => {
    assertEquals(await load(SHA), []);
  });

  it("not emitted for an explicit tag or branch either", async () => {
    assertEquals(await load("v1.0"), []);
  });

  it("when resolution fails, the failure itself comes up instead of a warning", async () => {
    const mock = createMockFetch({ files: serveAll() }); // no sha = the resolution API returns 404
    const warnings = await captureWarnings(async () => {
      await assertRejects(
        () =>
          loadManifest(
            { repo: REPO, hubUrl: HUB_URL },
            {
              fetch: mock.fetch,
              caches: new MemoryCacheStorage(),
            },
          ),
        HubFetchError,
      );
    });
    assertEquals(warnings, [], "recommends a pin although resolution failed");
  });
});

test("loadManifest: passing a SHA as revision issues no resolution request", async () => {
  const mock = createMockFetch({ files: serveAll() }); // no sha = the resolution API returns 404
  const caches = new MemoryCacheStorage();
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: mock.fetch,
      caches,
    },
  );
  assertEquals(loaded.revisionSha, SHA);
  assertEquals(mock.calls, [resolveUrl(MANIFEST_PATH)]);
});

test("loadManifest: a failed resolution of a mutable ref is reported as HubFetchError saying it cannot work offline", async () => {
  const mock = createMockFetch({ files: serveAll() });
  const caches = new MemoryCacheStorage();
  const error = await assertRejects(
    () => loadManifest({ repo: REPO, hubUrl: HUB_URL }, { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, undefined);
  assert(error.message.includes("offline"), "states that it cannot work offline");
});

describe("loadManifest: a repo name that breaks the URL structure is refused with HubError before any fetch", () => {
  // The fetch layer writes repo into the URL without percent-encoding (`/` is a structural
  // element), so `?` and `#` push `/resolve/<SHA>/...` into the query / fragment, and `../` or 3+
  // segments point at another path.
  const cases: readonly [string, string][] = [
    ["? of a query", "someone/anima?untrusted=yes"],
    ["# of a fragment", "someone/anima#frag"],
    ["% of percent-encoding", "someone/anima%2Fother"],
    ["parent directory ../", "../anima"],
    ["3 segments", "someone/anima/extra"],
    ["1 segment", "anima"],
  ];
  for (const [label, repo] of cases) {
    it(label, async () => {
      for (const revision of [undefined, SHA]) {
        const mock = createMockFetch({ sha: SHA, files: serveAll() });
        const error = await assertRejects(
          () =>
            loadManifest(
              { repo, hubUrl: HUB_URL, ...(revision === undefined ? {} : { revision }) },
              {
                fetch: mock.fetch,
                caches: new MemoryCacheStorage(),
              },
            ),
          HubError,
        );
        assert(error.message.includes(repo), `${error.message} does not name the repo it received`);
        assertEquals(mock.calls, [], "goes to fetch with an invalid repo");
      }
    });
  }
});

test("loadManifest: a reference whose repo is not a string is refused with HubError, not a plain TypeError", async () => {
  // The types are deliberately bypassed to reproduce what comes from outside them (calls from JS,
  // unvalidated settings).
  const ref = { hubUrl: HUB_URL } as unknown as HubRepoRef;
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  await assertRejects(
    () => loadManifest(ref, { fetch: mock.fetch, caches: new MemoryCacheStorage() }),
    HubError,
  );
  assertEquals(mock.calls, [], "goes to fetch with a reference without a repo");
});

describe("loadManifest: a revision resolution response that is not a commit SHA fails without pinning", () => {
  // If a non-conforming mirror or a custom fetch returned a mutable ref or a short form, every
  // later
  // fetch would resolve it again as "the pinned generation", and the contract of pinning to one
  // commit would not hold.
  const cases: readonly [string, string][] = [
    ["branch name", "main"],
    ["short SHA", SHA.slice(0, 7)],
    ["hex shorter than 40 digits", SHA.slice(0, 39)],
    ["uppercase hex", SHA.toUpperCase()],
  ];
  for (const [label, sha] of cases) {
    it(label, async () => {
      const mock = createMockFetch({ sha, files: serveAll() });
      const error = await assertRejects(
        () =>
          loadManifest(
            { repo: REPO, hubUrl: HUB_URL, revision: "main" },
            {
              fetch: mock.fetch,
              caches: new MemoryCacheStorage(),
            },
          ),
        HubFetchError,
      );
      assertEquals(error.repo, REPO);
      assert(
        error.cause instanceof Error && error.cause.message.includes(sha),
        "the diagnostic does not carry the value of the response it received",
      );
      assertEquals(
        mock.calls,
        [revisionUrl("main")],
        "proceeds to fetch with a generation that could not be pinned",
      );
    });
  }
});

test("loadManifest: a karume.json over 1MiB is rejected with ManifestFormatError (judged after the whole body is received)", async () => {
  const oversized = new TextEncoder().encode(" ".repeat(2 * 1024 * 1024));
  const mock = createMockFetch({
    sha: SHA,
    files: serveAll(new Map([[MANIFEST_PATH, oversized]])),
  });
  await captureWarnings(async () => {
    await assertRejects(
      () =>
        loadManifest(
          { repo: REPO, hubUrl: HUB_URL },
          {
            fetch: mock.fetch,
            caches: new MemoryCacheStorage(),
          },
        ),
      ManifestFormatError,
    );
  });
});

test("loadManifest: a 404 from the fetch layer passes through with the repo / SHA / path context added", async () => {
  const files = serveAll();
  files.delete(MANIFEST_PATH);
  const mock = createMockFetch({ sha: SHA, files });
  let error!: HubFetchError;
  await captureWarnings(async () => {
    error = await assertRejects(
      () =>
        loadManifest(
          { repo: REPO, hubUrl: HUB_URL },
          {
            fetch: mock.fetch,
            caches: new MemoryCacheStorage(),
          },
        ),
      HubFetchError,
    );
  });
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, MANIFEST_PATH);
  assert(error.cause instanceof Error, "keeps the fetch layer's error as cause");
});

test("loadManifest: a 429 on revision resolution arrives through onRetry, and the manifest arrives after the retry", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  const retries: RetryDiagnostic[] = [];
  let loaded!: LoadedManifest;
  await captureWarnings(async () => {
    loaded = await loadManifest(
      { repo: REPO, hubUrl: HUB_URL },
      {
        fetch: rateLimitOnce(mock.fetch, (url) => url === revisionUrl("main")),
        caches,
        onRetry: (diagnostic) => retries.push(diagnostic),
      },
    );
  });
  assertEquals(loaded.revisionSha, SHA, "not resolved after the 429");
  assertEquals(
    retries,
    [{ url: revisionUrl("main"), status: 429, attempt: 1, delayMs: 0, retryAfter: "0" }],
    "the retry notification did not arrive or lacks content",
  );
});

test("loadManifest: a 429 on karume.json also arrives through onRetry", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const target = resolveUrl(MANIFEST_PATH);
  const retries: RetryDiagnostic[] = [];
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: rateLimitOnce(mock.fetch, (url) => url === target),
      caches,
      onRetry: (diagnostic) => retries.push(diagnostic),
    },
  );
  assertEquals(loaded.manifest.available.models, ["anima-turbo", "anima-lite"]);
  assertEquals(retries.length, 1, "the retry notification did not arrive exactly once");
  assertEquals(retries[0].url, target, "the notification names a URL other than karume.json");
  assertEquals(retries[0].status, 429);
  assertEquals(countCalls(mock.calls, target), 2, "not fetched again after the 429");
});

test("loadManifest: a corrupted cached karume.json self-heals by fetching again exactly once", async () => {
  const caches = new MemoryCacheStorage();
  const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
  const first = createMockFetch({ files: serveAll() });
  await loadManifest(ref, { fetch: first.fetch, caches });

  // ① Bytes that cannot be read as UTF-8 (0xff never appears in UTF-8).
  overwriteEntry(hubCache(caches), manifestBytes, new Uint8Array([0xff, 0xfe, 0xff]), {
    keepRecord: true,
  });
  const healedDecode = createMockFetch({ files: serveAll() });
  const afterDecodeBreak = await loadManifest(ref, { fetch: healedDecode.fetch, caches });
  assertEquals(
    afterDecodeBreak.manifest.available.models,
    ["anima-turbo", "anima-lite"],
    "did not recover from the corrupted cache",
  );
  assertEquals(
    countCalls(healedDecode.calls, resolveUrl(MANIFEST_PATH)),
    1,
    "self-heal takes exactly one round trip",
  );

  // ② Decodes, but is broken as JSON.
  overwriteEntry(
    hubCache(caches),
    manifestBytes,
    new TextEncoder().encode('{"format": "karume/5"'),
    { keepRecord: true },
  );
  const healedParse = createMockFetch({ files: serveAll() });
  const afterParseBreak = await loadManifest(ref, { fetch: healedParse.fetch, caches });
  assertEquals(afterParseBreak.manifest.available.models, ["anima-turbo", "anima-lite"]);
  assertEquals(
    countCalls(healedParse.calls, resolveUrl(MANIFEST_PATH)),
    1,
    "self-heal takes exactly one round trip",
  );
});

test("loadManifest: a broken karume.json at the source of truth is a ManifestFormatError (and is not kept in the cache)", async () => {
  const caches = new MemoryCacheStorage();
  const broken = new TextEncoder().encode('{"format": "karume/5"');
  const mock = createMockFetch({ files: serveAll(new Map([[MANIFEST_PATH, broken]])) });
  await assertRejects(
    () =>
      loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, { fetch: mock.fetch, caches }),
    ManifestFormatError,
  );
  assertEquals(hubCache(caches).entries.size, 0, "keeps the broken manifest in the cache");
});

test("loadManifest: even when fully cached, an already aborted signal returns no manifest", async () => {
  const caches = new MemoryCacheStorage();
  const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
  const first = createMockFetch({ files: serveAll() });
  await loadManifest(ref, { fetch: first.fetch, caches });

  const controller = new AbortController();
  const reason = new Error("app: the startup was cancelled");
  controller.abort(reason);
  const second = createMockFetch({ files: serveAll() });
  const error = await assertRejects(() =>
    loadManifest(ref, { fetch: second.fetch, caches, signal: controller.signal }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assertEquals(second.calls, [], "goes to the network although already aborted");
  assertEquals(
    hubCache(caches).entries.size,
    1,
    "mistakes the abort for corruption and discards the cache",
  );
});

describe("loadManifest: rewriting the passed HubRepoRef afterwards does not move the fetch target", () => {
  // An implementation that rereads the caller's value would keep the old value in the displayed
  // `loaded.repo` while fetching with the old SHA from the new repo and host (what it names and
  // where it fetches would disagree).
  const OTHER_HUB = "https://another.test";
  const OTHER_REPO = "another/model";
  const assertPinnedCalls = (calls: readonly string[]): void => {
    for (const call of calls) {
      assert(call.startsWith(`${HUB_URL}/`), `${call} goes to the rewritten host`);
      assert(!call.includes(OTHER_REPO), `${call} fetches from the rewritten repo`);
    }
  };

  it("rewriting after the load", async () => {
    const mock = createMockFetch({ files: serveAll() });
    const caches = new MemoryCacheStorage();
    const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
    const loaded = await loadManifest(ref, { fetch: mock.fetch, caches });
    ref.repo = OTHER_REPO;
    ref.hubUrl = OTHER_HUB;

    await fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches });
    assertPinnedCalls(mock.calls);
    assertEquals(loaded.repo, REPO);
  });

  it("rewriting while the revision resolution is pending", async () => {
    const mock = createMockFetch({ sha: SHA, files: serveAll() });
    const caches = new MemoryCacheStorage();
    const ref = { repo: REPO, hubUrl: HUB_URL, revision: "main" };
    // Rewrite the moment the resolution request goes out (before the resolution returns = before
    // the pin).
    const mutating: typeof globalThis.fetch = (input, init) => {
      ref.repo = OTHER_REPO;
      ref.hubUrl = OTHER_HUB;
      ref.revision = "other-branch";
      return mock.fetch(input, init);
    };
    const loaded = await loadManifest(ref, { fetch: mutating, caches });
    assertEquals(loaded.repo, REPO);
    assertEquals(loaded.revisionSha, SHA);
    await fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches });
    assertPinnedCalls(mock.calls);
  });
});

const load = async (routes: MockRoutes, caches: MemoryCacheStorage) => {
  const mock = createMockFetch(routes);
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: mock.fetch,
      caches,
    },
  );
  return { mock, loaded };
};

test("fetchAssets: returns every key and fetches the same path only once", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(Object.keys(assets), Object.keys(files));
  assertEquals(
    countCalls(mock.calls, resolveUrl(STYLE_VECTORS)),
    1,
    "a duplicate path is fetched only once",
  );
  assert(
    assets["style_alias"] === assets["style_vectors"],
    "keys of the same path share the same bytes",
  );
  assertEquals(assets["transformer[0]"], payloadFor(TRANSFORMER_I8_FETCHED[0]));
  // The containers of dtypes that were not chosen are not fetched, not even one part.
  for (const path of TRANSFORMER_F16_PARTS) {
    assertEquals(countCalls(mock.calls, resolveUrl(path)), 0, `${path} is fetched`);
  }
  for (const bytes of Object.values(assets)) {
    assertEquals(bytes.byteOffset, 0);
    assertEquals(bytes.byteLength, bytes.buffer.byteLength);
  }
});

test("fetchAssets: the progress total is the sum of the manifest sizes (unique paths)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  const uniqueSizes = new Map<string, number>();
  for (const ref of Object.values(files)) uniqueSizes.set(ref.path, ref.size);
  let expectedTotal = 0;
  for (const size of uniqueSizes.values()) expectedTotal += size;

  assert(events.length > 0, "no progress at all");
  for (const event of events) {
    assertEquals(event.total, expectedTotal, "the total is the deduplicated sum of sizes");
    assert(event.loaded <= event.total, "loaded does not exceed the total");
    assertEquals(
      event.fileTotal,
      uniqueSizes.get(event.path),
      `${event.path}: fileTotal is the file's own manifest size`,
    );
    assert(
      event.fileLoaded <= event.fileTotal,
      `${event.path}: fileLoaded exceeded the file's size`,
    );
    if (event.phase !== "downloading") {
      assertEquals(
        event.fileLoaded,
        event.fileTotal,
        `${event.path}: ${event.phase} is the point where the whole file is present`,
      );
    }
  }
  assertEquals(events[events.length - 1].loaded, expectedTotal, "the last event reaches the total");
});

test("fetchAssets: per-file progress represents one file, separately from the overall total", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  // A downloading event in mid-reception (below size) appears = fileLoaded is that file's measured
  // reception.
  assert(
    events.some((event) => event.phase === "downloading" && event.fileLoaded < event.fileTotal),
    "fileLoaded in mid-reception is never observed",
  );
  // A downloading event after an earlier finished file was added to loaded is always larger than
  // the fileLoaded of that one file (if they stayed equal, the per-file field would be a copy of
  // the
  // overall total).
  assert(
    events.some((event) => event.phase === "downloading" && event.loaded > event.fileLoaded),
    "overall loaded and per-file fileLoaded never differ although several files are downloaded",
  );
});

/** The direction phases advance in (larger is later). */
const PHASE_RANK: Record<AssetPhase, number> = { downloading: 0, complete: 1 };

/**
 * The phase list per path (progress for several files arrives interleaved, so it is regrouped by
 * path).
 */
const phasesByPath = (events: readonly AssetProgress[]): Map<string, AssetPhase[]> => {
  const byPath = new Map<string, AssetPhase[]>();
  for (const event of events) {
    const phases = byPath.get(event.path) ?? [];
    phases.push(event.phase);
    byPath.set(event.path, phases);
  }
  return byPath;
};

const assertMonotonic = (phases: readonly AssetPhase[], path: string): void => {
  for (let index = 1; index < phases.length; index += 1) {
    assert(
      PHASE_RANK[phases[index]] >= PHASE_RANK[phases[index - 1]],
      `${path}: ${phases[index - 1]} → ${phases[index]} goes backwards`,
    );
  }
  assertEquals(phases[phases.length - 1], "complete", `${path}: does not end with complete`);
  assertEquals(
    phases.filter((phase) => phase === "complete").length,
    1,
    `${path}: complete exactly once per file`,
  );
};

test("fetchAssets: the phases of a network fetch advance monotonically downloading → complete", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  const byPath = phasesByPath(events);
  assertEquals(
    new Set(byPath.keys()),
    new Set(Object.values(files).map((ref) => ref.path)),
    "every fetched path appears in the progress",
  );
  for (const [path, phases] of byPath) {
    assert(
      phases.includes("downloading"),
      `${path}: no downloading although fetched from the network`,
    );
    assertMonotonic(phases, path);
  }
});

test("fetchAssets: the phase list of a cache hit is a single complete", async () => {
  const caches = new MemoryCacheStorage();
  const first = await load({ files: serveAll() }, caches);
  const files = selectionFiles(first.loaded.manifest);
  await fetchAssets(first.loaded, files, { fetch: first.mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  await fetchAssets(first.loaded, files, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  assertEquals(second.calls, [], "a cache hit does not go to the network");
  const byPath = phasesByPath(events);
  assert(byPath.size > 0, "no progress at all");
  for (const [path, phases] of byPath) {
    assertEquals(
      phases,
      ["complete"],
      `${path}: downloading appeared without a download / claims a verification that cannot be observed`,
    );
  }
  for (const event of events) {
    // Even on a path where downloading never appears, complete is the point where the whole file is
    // present, so this holds.
    assertEquals(
      event.fileLoaded,
      event.fileTotal,
      `${event.path}: fileLoaded does not reach size at ${event.phase} of a cache hit`,
    );
    // Overall loaded is "the sum of the bytes fetched", so it always includes the fileLoaded of the
    // same event.
    assert(
      event.loaded >= event.fileLoaded,
      `${event.path}: the overall loaded at ${event.phase} does not count this file`,
    );
  }
});

// Rate limiting is by "the size total in flight" (a byte budget), not by "count". The budget's own
// boundary behaviour is frozen on its own by tests/concurrency_test.ts; this is the observation as
// a surface — that nothing is throttled by count as long as it fits the budget, and that the
// dispatch order is fixed to the selection's declaration order.
test("fetchAssets: nothing is throttled by count as long as it fits the budget (rate limiting is the byte budget)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const files = selectionFiles(loaded.manifest);
  // All the fixture's files together are far below the budget (1.5GiB), so all of them run at once.
  const uniquePaths = new Set(Object.values(files).map((ref) => ref.path));
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  assertEquals(
    mock.peakConcurrency(),
    uniquePaths.size,
    `capped at ${mock.peakConcurrency()} at once although it fits the budget`,
  );
});

test("fetchAssets: the dispatch order is the selection's declaration order (no overtaking of a request waiting for budget)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const files = selectionFiles(loaded.manifest);
  // After path dedupe, once each in declaration order.
  const expected: string[] = [];
  for (const ref of Object.values(files)) {
    const url = resolveUrl(ref.path);
    if (!expected.includes(url)) expected.push(url);
  }
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  // calls[0] is the manifest fetch issued by load().
  assertEquals(mock.calls.slice(1), expected);
});

// In the 2 tests above, the fixture's total does not reach the budget (1.5GiB), so no waiting ever
// happens — they stay green even with the production rate limiting (admission) removed or with
// overtaking allowed. Here the budget is squeezed to a few dozen bytes through the internal entry
// point, and 3 requests, "an earlier unfinished request, a large request waiting for budget and a
// later small request", exercise the production wiring.
test("fetchAssets: a small later request does not overtake a large request waiting for budget, and in-flight stays within the budget", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const unique = new Map<string, FileRef>();
  for (const ref of Object.values(selectionFiles(loaded.manifest))) {
    if (ref.size > 0) unique.set(ref.path, ref);
  }
  const bySize = [...unique.values()].sort((x, y) => x.size - y.size);
  const small = bySize[0];
  const first = bySize[1];
  const big = bySize[bySize.length - 1];
  // A budget where the earlier (first) and the small (small) fit together, and the earlier and the
  // large (big) do not.
  const budget = first.size + small.size;
  assert(big.size > first.size && big.size > small.size, "the large request is not the largest");
  assert(
    big.size <= budget,
    "the large request alone exceeds the budget (the within-budget check would not hold)",
  );

  // Counts the bytes being received (released once the response body is fully read — the budget
  // slot is returned after that, so with correct rate limiting this value never exceeds the
  // budget).
  const sizeByUrl = new Map([small, first, big].map((ref) => [resolveUrl(ref.path), ref.size]));
  let inFlightBytes = 0;
  let peakBytes = 0;
  const tracked: typeof globalThis.fetch = async (input, init) => {
    const response = await mock.fetch(input, init);
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const size = sizeByUrl.get(href);
    if (size === undefined || response.body === null) return response;
    inFlightBytes += size;
    peakBytes = Math.max(peakBytes, inFlightBytes);
    const body = response.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        flush: () => {
          inFlightBytes -= size;
        },
      }),
    );
    return new Response(body, { status: response.status, headers: response.headers });
  };

  const files = { first, big, small };
  const assets = await fetchAssetsWithBudget(loaded, files, { fetch: tracked, caches }, budget);
  assertEquals(Object.keys(assets), ["first", "big", "small"]);
  // calls[0] is the manifest fetch issued by load(). The small request fits together with the
  // earlier one, but does not overtake the large request waiting for budget (head-of-line blocking
  // in declaration order).
  assertEquals(
    mock.calls.slice(1),
    [first, big, small].map((ref) => resolveUrl(ref.path)),
  );
  assert(
    peakBytes <= budget,
    `in-flight exceeded the budget (peak ${peakBytes}B / budget ${budget}B)`,
  );
  assertEquals(
    mock.peakConcurrency(),
    1,
    "a request that does not fit ran together with the earlier one",
  );
});

/** Lists the fetched paths in dispatch order (= the order of the table, with paths deduped). */
const dispatchOrder = (files: Readonly<Record<string, { path: string }>>): string[] => {
  const order: string[] = [];
  for (const ref of Object.values(files)) {
    if (!order.includes(ref.path)) order.push(ref.path);
  }
  return order;
};

test("fetchAssets: the real first failure surfaces, not a collateral one", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { mock, loaded } = await load({ files: served, delayMs: 5 }, caches);
  const files = selectionFiles(loaded.manifest);
  const order = dispatchOrder(files);
  // Makes only one file **other than the first** return 404. One failure aborts all the rest, so
  // the collateral ones settle as bare AbortErrors (which Chrome further replaces with a fixed
  // message) — picking the error by settle order or by worker array position would surface one of
  // those that is not the real culprit, and the root cause would be lost.
  const victim = order[order.length - 1];
  assert(order.indexOf(victim) > 0, "the failing file is the first one");
  served.delete(victim);

  const error = await assertRejects(
    () => fetchAssets(loaded, files, { fetch: withChromeAbortShape(mock.fetch), caches }),
    HubFetchError,
  );
  assertEquals(error.path, victim, "a file other than the one that failed is reported");
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assert(error.message.includes(victim), `${error.message} does not name the failed path`);
});

test("fetchAssets: the second time returns from the cache without going to the network", async () => {
  const caches = new MemoryCacheStorage();
  const first = await load({ files: serveAll() }, caches);
  const files = selectionFiles(first.loaded.manifest);
  await fetchAssets(first.loaded, files, { fetch: first.mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  const assets = await fetchAssets(first.loaded, files, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  assertEquals(second.calls, [], "a cache hit does not go to the network");
  assertEquals(assets["tokenizer"], payloadFor(TOKENIZER));
  assertEquals(
    events.filter((event) => event.phase === "complete").length,
    new Set(Object.values(files).map((ref) => ref.path)).size,
    "complete is emitted for every file even on a cache hit",
  );
});

test("fetchAssets: a hit whose recorded hash matches is trusted without reading the content again", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // Keep the record (the headers written at fetch time) and replace only the content. The fetch
  // layer's default is "trust the entry when the record matches the expectation" (trusting local
  // storage — a ruling without a knob), so no full hash runs and these broken bytes come back as
  // is. **This check makes the scope of the trust explicit**: if it turns red, rehashing came back
  // (= the full hash on every startup returned).
  const path = VAE_DECODER_PARTS[1];
  const tampered = tamper(payloadFor(path));
  overwriteEntry(hubCache(caches), payloadFor(path), tampered, { keepRecord: true });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });
  assertEquals(
    assets["vae_decoder[1]"],
    tampered,
    "a full hash runs on a hit whose record matches",
  );
  assertEquals(second.calls, [], "fetches again although the record matches");
});

test("fetchAssets: a corrupted entry without a record is caught by the actual hash and heals in one round trip", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // Dropping the record too = the same shape as an entry from an older version / an unverified
  // prefetch. The read side checks it against its actual hash, so a broken one is evicted and
  // fetched again (self-heal).
  const path = VAE_DECODER_PARTS[1];
  overwriteEntry(hubCache(caches), payloadFor(path), tamper(payloadFor(path)), {
    keepRecord: false,
  });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });

  assertEquals(assets["vae_decoder[1]"], payloadFor(path), "the corrupted cache passes through");
  assertEquals(
    countCalls(second.calls, resolveUrl(path)),
    1,
    "self-heal takes exactly one round trip",
  );
  assertEquals(second.calls.length, 1, "files that are not broken are fetched again too");
});

test("fetchAssets: even with a matching record, a byte count that differs from the manifest is fetched again", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // The record matches = no full hash runs, and only the length disagrees with the manifest. Only
  // the `expectedBytes` check (the verification in the fetch layer's HF layer) can catch this.
  const path = TEXT_ENCODER_PARTS[1];
  const truncated = payloadFor(path).slice(0, payloadFor(path).byteLength - 1);
  overwriteEntry(hubCache(caches), payloadFor(path), truncated, { keepRecord: true });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });
  assertEquals(
    assets["text_encoder[1]"],
    payloadFor(path),
    "an entry with a different length passes through",
  );
  assertEquals(
    countCalls(second.calls, resolveUrl(path)),
    1,
    "self-heal takes exactly one round trip",
  );
});

test("fetchAssets: a fetch whose content-length is right but whose body is short fails loudly", async () => {
  const caches = new MemoryCacheStorage();
  const path = TEXT_CONDITIONER_PARTS[1];
  const full = payloadFor(path);
  const short = full.slice(0, full.byteLength - 2);
  // content-length claims the manifest's size while only the body is streamed short (the fetch
  // layer's limit looks only at excess, so only the fetch layer's verification stops this).
  const { mock, loaded } = await load(
    {
      files: serveAll(new Map([[path, short]])),
      contentLength: (target) => (target === path ? full.byteLength : undefined),
    },
    caches,
  );
  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assert(error.cause instanceof Error, "keeps the fetch layer's verification failure as cause");
  assertEquals(
    hasEntry(hubCache(caches), short),
    false,
    "stores bytes that did not pass verification",
  );
});

test("fetchAssets: a 429 on an asset arrives through onRetry, and the refetched bytes are returned", async () => {
  const caches = new MemoryCacheStorage();
  const path = VAE_DECODER_PARTS[1];
  const target = resolveUrl(path);
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const retries: RetryDiagnostic[] = [];
  const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest), {
    fetch: rateLimitOnce(mock.fetch, (url) => url === target),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });
  assertEquals(retries.length, 1, "the retry notification did not arrive exactly once");
  assertEquals(retries[0].url, target, "the notification names the URL of another asset");
  assertEquals(retries[0].status, 429);
  assertEquals(assets["vae_decoder[1]"], payloadFor(path), "the refetched bytes are not returned");
  assertEquals(countCalls(mock.calls, target), 2, "not fetched again after the 429");
});

test("fetchAssets: even when fully cached, an already aborted signal returns no assets", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const cached = hubCache(caches).entries.size;

  const controller = new AbortController();
  const reason = new Error("app: the startup was cancelled");
  controller.abort(reason);
  const second = createMockFetch({ files: serveAll() });
  const error = await assertRejects(() =>
    fetchAssets(loaded, files, { fetch: second.fetch, caches, signal: controller.signal }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assertEquals(second.calls, [], "goes to the network although already aborted");
  assertEquals(
    hubCache(caches).entries.size,
    cached,
    "mistakes the abort for corruption and discards the cache",
  );
});

test("fetchAssets: an abort while reading the cache also passes through without returning assets", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const uniquePaths = new Set(Object.values(files).map((ref) => ref.path)).size;

  const controller = new AbortController();
  const reason = new Error("app: cancelled while reading");
  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  // Everything is cached, so no downloading appears. Cancelling on the complete of the first file
  // that is ready stops the rest at "the check right after leaving the fetch" (= nothing advances
  // to
  // complete after the cancel).
  const error = await assertRejects(() =>
    fetchAssets(loaded, files, {
      fetch: second.fetch,
      caches,
      signal: controller.signal,
      onProgress: (progress) => {
        events.push(progress);
        controller.abort(reason);
      },
    }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assert(uniquePaths > 1, "not enough files to observe the effect of the cancel");
  assert(
    events.filter((event) => event.phase === "complete").length < uniquePaths,
    "advances every file to complete although cancelled",
  );
});

test("fetchAssets: the cache is not split by authentication (the headers pass through to the fetch)", async () => {
  const caches = new MemoryCacheStorage();
  const authed = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: authed.fetch,
      caches,
      headers: { authorization: "Bearer hf_token" },
    },
  );
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, {
    fetch: authed.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  assert(authed.calls.length > 0, "nothing was fetched in the first place");
  assertEquals(
    new Set(authed.authorizations),
    new Set(["Bearer hf_token"]),
    "Authorization does not reach the fetch",
  );
  assertEquals(caches.namespaces.size, 1, "namespaces are split by authentication");

  // An unauthenticated request again hits the same entry (no isolation per credential — by design).
  const anonymous = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, { fetch: anonymous.fetch, caches });
  assertEquals(anonymous.calls, [], "the cache is split by authentication");
});

test("fetchAssets: a sha256 mismatch fails loudly (nothing is kept when the source of truth is broken)", async () => {
  const caches = new MemoryCacheStorage();
  const path = VAE_DECODER_PARTS[1];
  // The length is as declared, only the content differs (so the sha256 check fires, not the size
  // check).
  const corrupt = new TextEncoder().encode(
    "karume-test:tampered-".padEnd(payloadFor(path).byteLength, "X"),
  );
  assertEquals(
    corrupt.byteLength,
    payloadFor(path).byteLength,
    "the length matches; only sha256 is off",
  );
  const { mock, loaded } = await load({ files: serveAll(new Map([[path, corrupt]])) }, caches);
  // The comparison is done by the fetch layer (hashing while receiving), so to the hub it comes up
  // as a fetch failure.
  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, path);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assert(error.cause instanceof Error, "keeps the fetch layer's mismatch as cause");
  assertEquals(hasEntry(hubCache(caches), corrupt), false, "stores the mismatching bytes");
});

test("fetchAssets: received bytes over size fail loudly at the fetch layer's limit (nothing kept in the cache)", async () => {
  const caches = new MemoryCacheStorage();
  const path = TEXT_ENCODER_PARTS[1];
  const declared = payloadFor(path).byteLength;
  const bloated = new TextEncoder().encode(`karume-test:${path}${"!".repeat(64)}`);
  const { mock, loaded } = await load(
    {
      files: serveAll(new Map([[path, bloated]])),
      // content-length claims the right value while only the body is streamed longer.
      contentLength: (target) => (target === path ? declared : undefined),
    },
    caches,
  );
  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assert(error.cause instanceof Error, "keeps the fetch layer's error as cause");
  // The wording is set by the fetch layer's ADR 0011 (`受信が申告 N バイトを超えた`, "received more
  // than the declared N bytes"). This one word tells that it failed at **the receive limit**, not
  // on a sha256 mismatch.
  assert(
    error.cause.message.includes("超えた"),
    `${error.cause.message} does not show that it was cut off at the receive limit`,
  );
  assertEquals(hasEntry(hubCache(caches), bloated), false, "stores bytes over the limit");
});

test("fetchAssets: the AbortSignal reaches every fetch", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const error = await assertRejects(() =>
    fetchAssets(loaded, selectionFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort(),
    }),
  );
  assert(error instanceof DOMException && error.name === "AbortError", `${error} is not an abort`);
});

test("fetchAssets: a custom Error passed to abort(reason) propagates as is (it does not turn into a fetch failure)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const reason = new Error("app: the user cancelled the load");
  const error = await assertRejects(() =>
    fetchAssets(loaded, selectionFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort(reason),
    }),
  );
  assertStrictEquals(error, reason, "the reason the caller passed is wrapped in another error");
});

test("fetchAssets: abort(reason) propagates as is even when it is a primitive", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const error = await assertRejects(() =>
    fetchAssets(loaded, selectionFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort("app:cancelled"),
    }),
  );
  assertStrictEquals(error, "app:cancelled");
});

describe("an exception in onProgress does not fail the fetch whatever the notification stage (it arrives as a warning)", () => {
  // For downloading, the fetch layer isolates it and continues with only a warning. Unless complete
  // gets the same treatment, the same UI-side exception would change meaning by stage: "harmless
  // while receiving, a failure of the whole load at the end".
  const failure = new Error("ui: rendering bug");
  const throwing = (): void => {
    throw failure;
  };

  it("fetchAssets (network fetch = downloading and complete)", async () => {
    const caches = new MemoryCacheStorage();
    const { mock, loaded } = await load({ files: serveAll() }, caches);
    const files = selectionFiles(loaded.manifest);
    let bytes: Record<string, Uint8Array> = {};
    const warnings = await captureWarnings(async () => {
      bytes = await fetchAssets(loaded, files, { fetch: mock.fetch, caches, onProgress: throwing });
    });
    assertEquals(Object.keys(bytes).sort(), Object.keys(files).sort());
    assert(warnings.length > 0, "the listener's exception is silently discarded");
    assert(
      warnings.some((warning) => warning.includes("onProgress") && warning.includes("complete")),
      `the exception at the complete stage does not arrive as a warning: ${JSON.stringify(warnings)}`,
    );
  });

  it("prefetchAssets (cache hit = complete only)", async () => {
    const caches = new MemoryCacheStorage();
    const { mock, loaded } = await load({ files: serveAll() }, caches);
    const refs = Object.values(selectionFiles(loaded.manifest)).filter(
      (ref, index, all) => all.findIndex((other) => other.path === ref.path) === index,
    );
    await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });
    const warnings = await captureWarnings(async () => {
      await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches, onProgress: throwing });
    });
    assertEquals(warnings.length, refs.length, "one warning per complete did not arrive");
  });
});

test("fetchAssets: a cache I/O failure becomes a diagnostic that reaches the app (the fetch does not fail)", async () => {
  const caches = new MemoryCacheStorage({ failPut: true });

  const mock = createMockFetch({ files: serveAll() });
  const diagnostics: CacheDiagnostic[] = [];
  const loaded: LoadedManifest = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    { fetch: mock.fetch, caches, onCacheError: (entry) => diagnostics.push(entry) },
  );
  const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest), {
    fetch: mock.fetch,
    caches,
    onCacheError: (entry) => diagnostics.push(entry),
  });
  assertEquals(
    Object.keys(assets).length,
    Object.keys(selectionFiles(loaded.manifest)).length,
    "the fetch succeeds even when the cache is dead",
  );
  assert(diagnostics.length > 0, "the quota failure is silently swallowed");
  assertEquals(new Set(diagnostics.map((entry) => entry.op)), new Set(["put"]));
});

/**
 * Replaces `globalThis.caches`, runs `body` and always restores it (the real cache is never
 * touched).
 */
const withGlobalCaches = async (
  value: CacheStorage | undefined,
  body: () => Promise<void>,
): Promise<void> => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "caches");
  // Node has no CacheStorage: "absent" needs no swap, and a value is installed temporarily.
  if (value === undefined && original === undefined) {
    await body();
    return;
  }
  Object.defineProperty(globalThis, "caches", { value, configurable: true });
  try {
    await body();
  } finally {
    if (original === undefined) Reflect.deleteProperty(globalThis, "caches");
    else Object.defineProperty(globalThis, "caches", original);
  }
};

const populated = async (...names: readonly string[]): Promise<MemoryCacheStorage> => {
  const caches = new MemoryCacheStorage();
  for (const name of names) await caches.open(name);
  return caches;
};

// ---- Cross-repository references (`repo` / `revision` of `FileRef` — ADR 0038 §7).
//
// Two points are pinned: "fetch from the declared (repo, revision), not from the session's resolved
// SHA", and **the same path string in a different repository is different bytes** (folding by path
// would mix them up).

const FOREIGN_REPO = "someone/text-stack";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";
const CROSS_PATH = "text_encoder/model-00001-of-00002.krm";
const OWN_PART1 = "own/model-00002-of-00002.krm";
const BORROWED_PART1 = "borrowed/model-00002-of-00002.krm";

const digestOf = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const localBytes = new TextEncoder().encode("karume-test:local-text-encoder");
const foreignBytes = new TextEncoder().encode("karume-test:foreign-text-encoder-payload");

/** The declaration of part 0 (exactly `24 + graph + model` — container-v1 §8). */
const descriptorFor = (part0Bytes: number) => ({
  graph: { length: part0Bytes - 24 - 5, sha256: "f2".repeat(32) },
  model: { length: 5, sha256: "a3".repeat(32) },
});

const crossPart1 = async (path: string, mark: string) => ({
  path,
  size: payloadFor(path).byteLength,
  sha256: await digestOf(payloadFor(path)),
  ...(mark === "" ? {} : { repo: FOREIGN_REPO, revision: FOREIGN_SHA }),
});

/**
 * A manifest in which this repository and the referenced one claim **the same path** for part 0
 * (a detector for mix-ups). A cross-repository reference is **per container** (ADR 0109 decision
 * 3), so on the borrowing side every part names the referenced coordinates.
 */
const crossRepoManifest = JSON.stringify({
  format: "karume/5",
  generator: "karume/0.1.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        own: {
          i8: {
            container: {
              descriptor: descriptorFor(localBytes.byteLength),
              parts: [
                {
                  path: CROSS_PATH,
                  size: localBytes.byteLength,
                  sha256: await digestOf(localBytes),
                },
                await crossPart1(OWN_PART1, ""),
              ],
            },
          },
        },
        borrowed: {
          i8: {
            container: {
              descriptor: descriptorFor(foreignBytes.byteLength),
              parts: [
                {
                  path: CROSS_PATH,
                  size: foreignBytes.byteLength,
                  sha256: await digestOf(foreignBytes),
                  repo: FOREIGN_REPO,
                  revision: FOREIGN_SHA,
                },
                await crossPart1(BORROWED_PART1, "cross"),
              ],
            },
          },
        },
      },
      assets: {},
      quants: { i8: { weights: { own: "i8", borrowed: "i8" }, session: {} } },
      defaultQuant: "i8",
      pipelineConfig: {},
    },
  },
});

const crossRepoFiles = (
  overrides: ReadonlyMap<string, Uint8Array<ArrayBuffer>> = new Map(),
): Map<string, Uint8Array<ArrayBuffer>> =>
  new Map([
    [MANIFEST_PATH, new TextEncoder().encode(crossRepoManifest)],
    [CROSS_PATH, localBytes],
    [OWN_PART1, payloadFor(OWN_PART1)],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, foreignBytes],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${BORROWED_PART1}`, payloadFor(BORROWED_PART1)],
    ...overrides,
  ]);

const foreignUrl = `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${CROSS_PATH}`;

test("fetchAssets: a cross-repository reference is fetched from the declared (repo, revision)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const files = selectionFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(
    assets["own[0]"],
    localBytes,
    "this repository's file turned into the referenced repository's bytes",
  );
  assertEquals(
    assets["borrowed[0]"],
    foreignBytes,
    "the cross-repository file turned into this repository's bytes",
  );
  assertEquals(
    countCalls(mock.calls, foreignUrl),
    1,
    "the referenced repository's URL was not requested",
  );
  assertEquals(
    countCalls(mock.calls, resolveUrl(CROSS_PATH)),
    1,
    "this repository's URL was not requested",
  );
  // It is a content key, so it cannot be looked up by the source URL. That paths of the same name
  // in different repositories coexist as separate entries (= the key includes the repo) is checked
  // by the fact that both byte arrays remain at the same time.
  assert(hasEntry(hubCache(caches), localBytes), "this repository's entry is missing");
  assert(hasEntry(hubCache(caches), foreignBytes), "the cross-repository entry is missing");
});

test("fetchAssets: a failed verification of a cross-repository reference names the referenced repo / SHA", async () => {
  const caches = new MemoryCacheStorage();
  const corrupt = tamper(foreignBytes);
  const { mock, loaded } = await load(
    {
      files: crossRepoFiles(new Map([[`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, corrupt]])),
    },
    caches,
  );
  const error = await assertRejects(
    () => fetchAssets(loaded, selectionFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  // Naming the session's repo would make the diagnostic point at "a path that does not exist in
  // that repository".
  assertEquals(error.repo, FOREIGN_REPO);
  assertEquals(error.revisionSha, FOREIGN_SHA);
  assertEquals(error.path, CROSS_PATH);
});

test("fetchAssets: a 429 from the referenced repository also arrives through onRetry (naming the referenced URL)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const retries: RetryDiagnostic[] = [];
  const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest), {
    fetch: rateLimitOnce(mock.fetch, (url) => url === foreignUrl),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });
  assertEquals(
    retries.map((diagnostic) => diagnostic.url),
    [foreignUrl],
  );
  assertEquals(
    assets["borrowed[0]"],
    foreignBytes,
    "the refetched cross-repository file is not returned",
  );
  assertEquals(countCalls(mock.calls, foreignUrl), 2, "not fetched again after the 429");
});

test("fetchAssets: a same-repository and a cross-repository file with the same path count as two in the progress too", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const files = selectionFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  let expectedTotal = 0;
  for (const ref of Object.values(files)) expectedTotal += ref.size;
  for (const event of events) assertEquals(event.total, expectedTotal, "the total is folded");
  assertEquals(events[events.length - 1].loaded, expectedTotal, "the last event reaches the total");
  assertEquals(
    events.filter((event) => event.phase === "complete").length,
    Object.keys(files).length,
    "the 2 files with the same path are folded into one",
  );
  assertEquals(
    events.filter((event) => event.phase === "complete" && event.path === CROSS_PATH).length,
    2,
    "the same-repository and cross-repository files with the same path are folded into one",
  );
});

test("fetchAssets: progress names the (repo, revision) it actually fetched from", async () => {
  // The consumer must be able to tell the 2 files with the same path apart — keying per-file
  // progress by path alone mixes the 2 into one (if the public event did not carry the identifier,
  // the fetch layer's dedupe would break down on the consumer side).
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, selectionFiles(loaded.manifest), {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  const origins = (path: string): string[] =>
    events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => `${event.repo}@${event.revision}`)
      .sort();
  assertEquals(
    origins(CROSS_PATH),
    [`${FOREIGN_REPO}@${FOREIGN_SHA}`, `${REPO}@${SHA}`].sort(),
    "the same-repository and cross-repository files with the same path cannot be told apart by fetch origin",
  );
  // The reference is per container, so the remaining parts of the borrowing side name the
  // referenced repository too (this repository's parts name this repository).
  assertEquals(origins(BORROWED_PART1), [`${FOREIGN_REPO}@${FOREIGN_SHA}`]);
  assertEquals(origins(OWN_PART1), [`${REPO}@${SHA}`]);
  // downloading carries the same fields too (if only complete named them, an in-progress bar could
  // not be matched).
  for (const event of events) {
    assert(
      event.repo !== undefined && event.revision !== undefined,
      `${event.path} does not name the fetch origin`,
    );
  }
});

describe("loadManifest: a cross-repository reference to the loaded repo itself is refused with ManifestFormatError", () => {
  // A cross-repository reference to another revision of this repository makes the identity of the
  // reference (which includes repo and revision) and the cached file (which on HF does not include
  // revision) diverge — protecting one deletes the shared file along with it.
  for (const [label, revision] of [
    ["same revision", SHA],
    ["different revision", FOREIGN_SHA],
  ]) {
    it(label, async () => {
      const selfCross = crossRepoManifest
        .replaceAll(FOREIGN_REPO, REPO)
        .replaceAll(FOREIGN_SHA, revision);
      const mock = createMockFetch({
        files: crossRepoFiles(new Map([[MANIFEST_PATH, new TextEncoder().encode(selfCross)]])),
      });
      const error = await assertRejects(
        () =>
          loadManifest(
            { repo: REPO, hubUrl: HUB_URL, revision: SHA },
            {
              fetch: mock.fetch,
              caches: new MemoryCacheStorage(),
            },
          ),
        ManifestFormatError,
      );
      assert(error.message.includes(REPO), `${error.message} does not name this repository`);
      assert(
        error.message.includes("models.m.weights.borrowed"),
        `${error.message} does not name where the violation is`,
      );
    });
  }
});

// ---- Cross-repository containers (ADR 0038 §7 / ADR 0109 decision 3 — "cross-repository
// references are per container"). The exporter writes shared components in the form **one part =
// one reference**, so the receiving side must be able to fetch every part of the container from
// the referenced URL (referencing only the first one / folding the list would make the remaining
// parts fail because they are not in the session's repo).

const SPLIT_PATHS = [
  "text_encoder/model-00001-of-00002.krm",
  "text_encoder/model-00002-of-00002.krm",
];

const splitBytes = SPLIT_PATHS.map((path) =>
  new TextEncoder().encode(`karume-test:foreign-part:${path}`),
);

const splitShardManifest = JSON.stringify({
  format: "karume/5",
  generator: "karume/0.1.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        borrowed: {
          f16: {
            container: {
              descriptor: descriptorFor(splitBytes[0].byteLength),
              parts: await Promise.all(
                SPLIT_PATHS.map(async (path, index) => ({
                  path,
                  size: splitBytes[index].byteLength,
                  sha256: await digestOf(splitBytes[index]),
                  repo: FOREIGN_REPO,
                  revision: FOREIGN_SHA,
                })),
              ),
            },
          },
        },
      },
      assets: {},
      quants: { f16: { weights: { borrowed: "f16" }, session: {} } },
      defaultQuant: "f16",
      pipelineConfig: {},
    },
  },
});

const splitShardFiles = (): Map<string, Uint8Array<ArrayBuffer>> =>
  new Map([
    [MANIFEST_PATH, new TextEncoder().encode(splitShardManifest)],
    ...SPLIT_PATHS.map(
      (path, index) => [`${FOREIGN_REPO}@${FOREIGN_SHA}/${path}`, splitBytes[index]] as const,
    ),
  ]);

test("fetchAssets: a cross-repository container fetches each part from the referenced URL", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: splitShardFiles() }, caches);
  const files = selectionFiles(loaded.manifest);

  assertEquals(Object.keys(files), ["borrowed[0]", "borrowed[1]"]);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(assets["borrowed[0]"], splitBytes[0], "the bytes of part 0 differ");
  assertEquals(assets["borrowed[1]"], splitBytes[1], "the bytes of part 1 differ");
  for (const path of SPLIT_PATHS) {
    assertEquals(
      countCalls(mock.calls, `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${path}`),
      1,
      `${path} was not fetched from the referenced repository`,
    );
    assertEquals(
      countCalls(mock.calls, resolveUrl(path)),
      0,
      `${path} is fetched from the session's repo (it is not there)`,
    );
  }
});

test("clearHubCache: deletes the warmed assets (the next fetch goes to the network)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = selectionFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const cached = hubCache(caches).entries.size;
  assert(cached > 1, "no assets accumulated besides the manifest");

  assertEquals(await clearHubCache({ caches }), true);
  assertEquals(
    [...caches.namespaces.keys()],
    [],
    "the fetch layer's namespace was not deleted as a whole",
  );

  const after = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, { fetch: after.fetch, caches });
  assert(after.calls.length > 0, "the fetch after the deletion does not go to the network");
});

test("clearHubCache: false when there is nothing to delete", async () => {
  const caches = await populated("other/1");
  assertEquals(await clearHubCache({ caches }), false);
  assertEquals([...caches.namespaces.keys()], ["other/1"]);
});

test("loadManifest / clearHubCache: leave the namespaces of hub 0.6 and earlier (karume/1 family) alone (reclaiming them is manual)", async () => {
  const untouched = ["karume/1", "karume/1:auth:0123456789abcdef", "other/1"];
  const caches = await populated(...untouched);
  await load({ files: serveAll() }, caches);
  assertEquals([...caches.namespaces.keys()].slice(0, untouched.length), untouched);

  assertEquals(await clearHubCache({ caches }), true);
  assertEquals(
    [...caches.namespaces.keys()],
    untouched,
    "cleared more than the fetch layer namespace",
  );
  assertEquals(
    await clearHubCache({ caches }),
    false,
    "counted the old namespaces as something to clear",
  );
});

test("clearHubCache: with caches passed, deletes only those (does not touch globalThis)", async () => {
  const injected = new MemoryCacheStorage();
  await load({ files: serveAll() }, injected);
  const global = new MemoryCacheStorage();
  await load({ files: serveAll() }, global);
  const warmed = [...global.namespaces.keys()];
  assertEquals(warmed.length, 1);
  await withGlobalCaches(global, async () => {
    assertEquals(await clearHubCache({ caches: injected }), true);
  });
  assertEquals([...injected.namespaces.keys()], []);
  assertEquals([...global.namespaces.keys()], warmed);
});

test("clearHubCache: without caches, deletes globalThis.caches", async () => {
  const global = new MemoryCacheStorage();
  await load({ files: serveAll() }, global);
  await withGlobalCaches(global, async () => {
    assertEquals(await clearHubCache(), true);
  });
  assertEquals([...global.namespaces.keys()], []);
});

test("clearHubCache: an environment without CacheStorage fails loudly (not a silent no-op)", async () => {
  await withGlobalCaches(undefined, async () => {
    const error = await assertRejects(() => clearHubCache(), Error);
    assert(error.message.includes("CacheStorage"), `${error.message} does not name the cause`);
  });
});

// ---- Range reads (`openAsset` — `source.ts` ⑧). The HF source's reader can open **only the
// cache's content** (`openHfFile` does not go to the network), so two points are pinned here: "when
// warm, it reads with zero fetches" and "when not warm, the opening side warms once".

/** The setup shared by the range-read tests (only the manifest read + the one target file). */
const openable = async (): Promise<{
  mock: MockFetch;
  caches: MemoryCacheStorage;
  loaded: LoadedManifest;
  ref: FileRef;
}> => {
  const mock = createMockFetch({ files: serveAll() });
  const caches = new MemoryCacheStorage();
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: mock.fetch,
      caches,
    },
  );
  return { mock, caches, loaded, ref: selectionFiles(loaded.manifest)["tokenizer"] };
};

test("openAsset: a warmed reference returns only the range without causing a fetch", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  await prefetchAssets(loaded, [ref], access);
  const payload = payloadFor(ref.path);
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "the range reader does not open although warm");
  // Deno has `globalThis.Deno`, so the fetch layer's default strategy is "stream" (because `blob()`
  // loads the whole file onto the heap) = skipping costs in proportion to the offset.
  assertEquals(reader.cost, "scan");

  const middle = Math.floor(ref.size / 2);
  for (const [offset, length] of [
    [0, 4],
    [middle, 3],
    [ref.size - 5, 5],
  ]) {
    const bytes = await reader.read(offset, length);
    assertEquals(
      bytes,
      new Uint8Array(payload.subarray(offset, offset + length)),
      `the content of [${offset}, ${offset + length}) is corrupted`,
    );
    // The consumer reads it as a TypedArray as is, so it must occupy its whole buffer.
    assertEquals(bytes.byteOffset, 0);
    assertEquals(bytes.buffer.byteLength, length);
  }
  assertEquals(mock.calls.length, calls, "a fetch happens although warm");
});

test("openAsset: an unfetched reference is warmed once by the opening side before it opens", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  const calls = mock.calls.length;

  // The caller is not burdened with the step "run prefetchAssets first" (the consumer only requests
  // rows).
  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "the range reader does not open for an unfetched reference");
  assertEquals(mock.calls.length - calls, 1, "the re-warm did not take a single round trip");

  assertEquals(await reader.read(0, ref.size), payloadFor(ref.path));
  // Reads after opening complete inside the cache (no network per read).
  assertEquals(mock.calls.length - calls, 1, "a read causes a fetch");
});

test("openAsset: an entry without a recorded hash is fetched again without being read", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  await prefetchAssets(loaded, [ref], access);
  const payload = payloadFor(ref.path);
  // Drop the record and break the content (there is nothing to compare against, so the opening side
  // cannot open it).
  overwriteEntry(hubCache(caches), payload, new Uint8Array(ref.size), { keepRecord: false });
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "the range reader does not open although it was fetched again");
  assertEquals(mock.calls.length - calls, 1, "the broken entry was not fetched again");
  assertEquals(await reader.read(0, ref.size), payload, "the broken bytes are read as is");
});

test("openAsset: an entry whose recorded hash disagrees with the declaration is also fetched again without being read", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  const other = selectionFiles(loaded.manifest)["vae_decoder[0]"];
  await prefetchAssets(loaded, [ref, other], access);
  const payload = payloadFor(ref.path);
  // Swap the records of 2 entries = both get "recorded hash ≠ the manifest's declaration". The
  // opening side sees this as "the content changed" and evicts, so it falls into the same refetch
  // path as an entry without a record.
  swapRecords(hubCache(caches), payload, payloadFor(other.path));
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "the range reader does not open although it was fetched again");
  // Only the one that was opened is fetched again (the other one whose record was broken is not
  // touched).
  assertEquals(
    mock.calls.length - calls,
    1,
    "the disagreeing entry was not fetched again in one round trip",
  );
  assertEquals(await reader.read(0, ref.size), payload, "the refetched content cannot be read");
});

test("openAsset: a cross-repository reference opens through the reader of the declared (repo, revision)", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const access = { fetch: mock.fetch, caches };
  // The manifest has this repository and the referenced one claim **the same path**, so folding by
  // path would read this repository's bytes (a detector that the range reader also comes from the
  // referenced repository through `originFor`).
  const ref = selectionFiles(loaded.manifest)["borrowed[0]"];
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(
    reader !== undefined,
    "the reader does not open although the capability should carry over to the referenced repository",
  );
  assertEquals(
    countCalls(mock.calls, foreignUrl),
    1,
    "the referenced repository's URL was not warmed",
  );
  assertEquals(
    mock.calls.length - calls,
    1,
    "warming the cross-repository file did not take a single round trip",
  );
  assertEquals(await reader.read(0, ref.size), foreignBytes, "this repository's bytes are read");
});

describe("openAsset: nothing outside the declared size can be read, regardless of the source and of what is cached", () => {
  // The boundary the range reader checks is **the declared size** (the doc of `openAsset`). HF's
  // fetch layer does not use the declared size as the read boundary, so a reference that declares a
  // shorter size for the same sha256 and opens a warm entry could read outside the declaration (the
  // rest of the file) — the local source rejects by the declared size, so the behaviour would split
  // by source.
  const served = serveAll();
  const lookup = (path: string): Uint8Array<ArrayBuffer> => {
    const bytes = served.get(path);
    if (bytes === undefined) throw new Error(`test-directory: cannot read ${path}`);
    return bytes;
  };
  const adapter: DirectoryAdapter = {
    readFile: (path) => Promise.resolve(new Uint8Array(lookup(path))),
    readFileRange: (path, offset, length) =>
      Promise.resolve(new Uint8Array(lookup(path).subarray(offset, offset + length))),
  };
  /**
   * Ranges that go outside the declaration (1 byte from exactly the declared size, a negative
   * offset, a non-integer length).
   */
  const outside = (size: number): readonly (readonly [number, number])[] => [
    [size, 1],
    [size - 1, 2],
    [-1, 1],
    [0, 0.5],
  ];

  for (const warm of [false, true]) {
    it(`HF, ${warm ? "warm" : "cold"}, a reference as declared`, async () => {
      const { mock, caches, loaded, ref } = await openable();
      const access = { fetch: mock.fetch, caches };
      if (warm) await prefetchAssets(loaded, [ref], access);
      const reader = await openAsset(loaded, ref, access);
      assert(reader !== undefined);
      for (const [offset, length] of outside(ref.size)) {
        await assertRejects(() => reader.read(offset, length), Error, "declared size");
      }
    });
  }

  it("HF, warm, a reference that declares a shorter size for the same sha256", async () => {
    const { mock, caches, loaded, ref } = await openable();
    const access = { fetch: mock.fetch, caches };
    await prefetchAssets(loaded, [ref], access);
    const shorter: FileRef = { ...ref, size: ref.size - 1 };
    const reader = await openAsset(loaded, shorter, access);
    assert(reader !== undefined);
    await assertRejects(() => reader.read(shorter.size, 1), Error, "declared size");
  });

  it("HF: the rejection names the path, the requested range and the declared size", async () => {
    // The HF source has no range check of its own, so this message comes from the common layer.
    const { mock, caches, loaded, ref } = await openable();
    const access = { fetch: mock.fetch, caches };
    await prefetchAssets(loaded, [ref], access);
    const reader = await openAsset(loaded, ref, access);
    assert(reader !== undefined);
    await assertRejects(
      () => reader.read(ref.size - 1, 2),
      Error,
      `hub: ${ref.path} range [${ref.size - 1}, ${ref.size + 1}) is invalid` +
        ` (declared size ${ref.size} — only ranges of non-negative integers within size can be read)`,
    );
    for (const [offset, length] of [
      [Number.NaN, 1],
      [0, Number.POSITIVE_INFINITY],
    ]) {
      await assertRejects(() => reader.read(offset, length), Error, "declared size");
    }
    // The boundary itself is inside: the whole declared range and an empty range at its end.
    assertEquals(await reader.read(0, ref.size), payloadFor(ref.path));
    assertEquals((await reader.read(ref.size, 0)).byteLength, 0);
  });

  for (const shrink of [0, 1]) {
    it(`local, a reference ${shrink === 0 ? "as declared" : "declaring a shorter size"}`, async () => {
      const local = await loadManifest(localDirectory(adapter, { label: "./models/test" }));
      const ref = selectionFiles(local.manifest)["tokenizer"];
      const declared: FileRef = { ...ref, size: ref.size - shrink };
      const reader = await openAsset(local, declared);
      assert(reader !== undefined);
      for (const [offset, length] of outside(declared.size)) {
        await assertRejects(() => reader.read(offset, length), Error, "declared size");
      }
    });
  }
});

test("openAsset: an abort while opening a warmed reader also returns no reader", async () => {
  // A warm-cache hit does not go to the network = a stretch where the source's signal watch has no
  // effect, so without checking the abort after opening, "a reader comes back although cancelled"
  // (inconsistent with loadManifest / prefetchAssets).
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  await prefetchAssets(loaded, [ref], access);
  const controller = new AbortController();
  const reason = new Error("app: cancelled while opening");
  hubCache(caches).onMatch = () => controller.abort(reason);
  const calls = mock.calls.length;

  const error = await assertRejects(() =>
    openAsset(loaded, ref, { ...access, signal: controller.signal }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assertEquals(mock.calls.length, calls, "a fetch happens although warm");
});

test("openAsset: an already aborted signal raises the reason as is, whether or not a reader exists", async () => {
  const mock = createMockFetch({ files: serveAll() });
  const remote = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: mock.fetch,
      caches: new MemoryCacheStorage(),
    },
  );
  const files = selectionFiles(remote.manifest);
  const ref = files[Object.keys(files)[0]];

  // Build one source without a reader (an adapter without positional reads) from the same manifest.
  const served = serveAll();
  const lookup = (path: string): Uint8Array<ArrayBuffer> => {
    const bytes = served.get(path);
    if (bytes === undefined) throw new Error(`test-directory: cannot read ${path}`);
    return bytes;
  };
  const adapter: DirectoryAdapter = {
    readFile: (path) => Promise.resolve(new Uint8Array(lookup(path))),
  };
  const local = await loadManifest(localDirectory(adapter, { label: "./models/test" }), {
    caches: new MemoryCacheStorage(),
  });

  const controller = new AbortController();
  const reason = new Error("test: aborted by the caller");
  controller.abort(reason);

  // The abort is checked before the difference in capability — even a source without a reader fails
  // with the reason, not `undefined`.
  for (const loadedSource of [remote, local]) {
    const error = await assertRejects(
      () => openAsset(loadedSource, ref, { signal: controller.signal }),
      Error,
    );
    assertStrictEquals(error, reason);
  }
});

test("fetchAssets: special keys are also returned as own properties, sharing the bytes of the same reference", async () => {
  const served = serveAll();
  const reads: string[] = [];
  const loaded = await loadManifest(
    localDirectory(
      {
        readFile: (path) => {
          reads.push(path);
          const bytes = served.get(path);
          if (bytes === undefined) throw new Error(`undefined asset ${path}`);
          return Promise.resolve(bytes);
        },
      },
      { label: "synthetic" },
    ),
  );
  const ref = Object.values(selectionFiles(loaded.manifest))[0];
  const files = Object.fromEntries(
    ["__proto__", "constructor", "2", "toString"].map((key) => [key, ref]),
  );
  const assets = await fetchAssets(loaded, files);
  assertEquals(Object.keys(assets), Object.keys(files));
  assertStrictEquals(Object.getPrototypeOf(assets), Object.prototype);
  for (const key of Object.keys(files)) {
    assert(Object.hasOwn(assets, key));
    assertStrictEquals(assets[key], assets["2"]);
  }
  assertEquals(reads.filter((path) => path === ref.path).length, 1);
});
