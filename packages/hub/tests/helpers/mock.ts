/**
 * Test doubles for the hub (they never touch the real network or the real Cache Storage).
 *
 * The fetch layer (`@hdae/fetch-cache`) takes `fetch` and `caches` as injection points, so the hub
 * tests plug these mocks in there. MUST: always pass `caches` — without it the fetch layer writes
 * to
 * Deno's real cache (on disk).
 */

/**
 * One cache entry. **It keeps the headers exactly as they were put** — the fetch layer stores the
 * recorded hash in the response headers, so a mock that drops headers becomes "a cache whose record
 * vanishes every time", and the tests would only exercise the full rehash path that does not happen
 * in a real environment.
 *
 * MUST: the tests do not interpret the header names (they do not lean on the fetch layer's internal
 * format). The mock only returns what was put; a test that wants "with record / without record"
 * carries `headers` over as a whole or empties it.
 */
export type CacheEntry = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly headers: Headers;
};

/**
 * An in-memory Cache (implements only the match / put / delete / keys of `Cache` that the hub
 * uses).
 */
export class MemoryCache implements Cache {
  readonly entries = new Map<string, CacheEntry>();
  /** Makes put fail (simulates exceeding the quota). */
  failPut = false;
  /**
   * A hook called at the moment a read happens (used to inject an abort in the middle of a read).
   */
  onMatch?: (key: string) => void;

  match(request: RequestInfo | URL): Promise<Response | undefined> {
    const key = urlOf(request);
    this.onMatch?.(key);
    const entry = this.entries.get(key);
    if (entry === undefined) return Promise.resolve(undefined);
    return Promise.resolve(new Response(entry.bytes, { headers: entry.headers }));
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    if (this.failPut) throw new Error("mock: quota exceeded");
    // The real Cache API stores the headers too, so they are kept for the same lifetime as the
    // entry.
    const headers = new Headers(response.headers);
    const bytes = new Uint8Array(await response.arrayBuffer());
    this.entries.set(urlOf(request), { bytes, headers });
  }

  delete(request: RequestInfo | URL): Promise<boolean> {
    return Promise.resolve(this.entries.delete(urlOf(request)));
  }

  keys(request?: RequestInfo | URL): Promise<readonly Request[]> {
    const urls = request === undefined ? [...this.entries.keys()] : [urlOf(request)];
    return Promise.resolve(urls.map((url) => new Request(url)));
  }

  add(): Promise<void> {
    throw new Error("mock: add is not used");
  }

  addAll(): Promise<void> {
    throw new Error("mock: addAll is not used");
  }

  matchAll(): Promise<readonly Response[]> {
    throw new Error("mock: matchAll is not used");
  }
}

/**
 * An in-memory CacheStorage. The namespace name is **owned by the fetch layer** (fixed internally
 * to
 * one since 0.5.0), so tests do not name it and look it up with {@link hubCache}. The names are
 * kept
 * as is in order to observe that "the namespaces of other code are not touched".
 */
export class MemoryCacheStorage implements CacheStorage {
  readonly namespaces = new Map<string, MemoryCache>();

  /**
   * `failPut` applies to every namespace opened afterwards (simulates a quota failure without
   * knowing the name).
   */
  constructor(private readonly defaults: { readonly failPut?: boolean } = {}) {}

  open(cacheName: string): Promise<Cache> {
    const existing = this.namespaces.get(cacheName);
    if (existing !== undefined) return Promise.resolve(existing);
    const created = new MemoryCache();
    created.failPut = this.defaults.failPut ?? false;
    this.namespaces.set(cacheName, created);
    return Promise.resolve(created);
  }

  has(cacheName: string): Promise<boolean> {
    return Promise.resolve(this.namespaces.has(cacheName));
  }

  delete(cacheName: string): Promise<boolean> {
    return Promise.resolve(this.namespaces.delete(cacheName));
  }

  keys(): Promise<string[]> {
    return Promise.resolve([...this.namespaces.keys()]);
  }

  match(): Promise<Response | undefined> {
    throw new Error("mock: CacheStorage.match is not used");
  }
}

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;

/**
 * Returns the one namespace the hub uses. The namespace **name is owned by the fetch layer**, so
 * the
 * tests do not name it and look it up by the observable fact that "the hub touched exactly this one
 * namespace".
 */
export const hubCache = (caches: MemoryCacheStorage): MemoryCache => {
  const found = [...caches.namespaces.values()];
  if (found.length !== 1) {
    throw new Error(
      `test: the hub's namespace is not exactly one (${JSON.stringify([...caches.namespaces.keys()])})`,
    );
  }
  return found[0];
};

/**
 * Looks up the key of the entry that stores `bytes`. **The key's spelling is owned by the fetch
 * layer** (a content key is an opaque URL folded under a reserved origin), so the tests find it
 * from
 * the content.
 */
const keyOf = (cache: MemoryCache, bytes: Uint8Array): string | undefined => {
  for (const [key, entry] of cache.entries) {
    if (entry.bytes.byteLength !== bytes.byteLength) continue;
    if (entry.bytes.every((byte, index) => byte === bytes[index])) return key;
  }
  return undefined;
};

/** Whether an entry stores `bytes`. */
export const hasEntry = (cache: MemoryCache, bytes: Uint8Array): boolean =>
  keyOf(cache, bytes) !== undefined;

/**
 * Replaces the cached `original` with `replacement` (simulates cache corruption).
 *
 * With `keepRecord` true, the headers the fetch layer wrote are kept (= an entry whose record still
 * matches but whose content is broken); with false they are dropped (= an entry without a record,
 * the shape left by an unverified prefetch or an older version). The read side branches completely
 * differently on the two (the former is trusted, the latter is checked against its actual hash).
 */
export const overwriteEntry = (
  cache: MemoryCache,
  original: Uint8Array,
  replacement: Uint8Array<ArrayBuffer>,
  options: { readonly keepRecord: boolean },
): void => {
  const key = keyOf(cache, original);
  if (key === undefined) throw new Error("test: no cache entry holds the given bytes");
  const entry = cache.entries.get(key);
  if (entry === undefined) throw new Error(`test: the entry ${key} is gone`);
  cache.entries.set(key, {
    bytes: replacement,
    headers: options.keepRecord ? entry.headers : new Headers(),
  });
};

/**
 * **Swaps only the records (headers)** of two entries — simulates the shape where both have
 * "recorded
 * hash ≠ the manifest's declaration". {@link overwriteEntry} can only make two shapes, "record
 * matches but content broken" (trusted) and "no record" (checked against its actual hash); the
 * third
 * shape, where **the record disagrees** (the read side evicts and fetches again), can only be made
 * here.
 *
 * MUST: the headers are carried over as a whole — the tests do not interpret their names (the MUST
 * of {@link CacheEntry}).
 */
export const swapRecords = (cache: MemoryCache, left: Uint8Array, right: Uint8Array): void => {
  const keys = [left, right].map((bytes) => {
    const key = keyOf(cache, bytes);
    if (key === undefined) throw new Error("test: no cache entry holds the given bytes");
    return key;
  });
  const entries = keys.map((key) => {
    const entry = cache.entries.get(key);
    if (entry === undefined) throw new Error(`test: the entry ${key} is gone`);
    return entry;
  });
  cache.entries.set(keys[0], { bytes: entries[0].bytes, headers: entries[1].headers });
  cache.entries.set(keys[1], { bytes: entries[1].bytes, headers: entries[0].headers });
};

export const HUB_URL = "https://hub.test";
export const REPO = "someone/anima";
export const SHA = "0123456789abcdef0123456789abcdef01234567";

/**
 * payload(path) — the size / sha256 of tests/fixtures/manifest-fetch.json are derived from this
 * rule.
 */
export const payloadFor = (path: string): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(`karume-test:${path}`);

export type MockRoutes = {
  /** The sha the revision resolution API returns. undefined returns 404. */
  readonly sha?: string;
  /**
   * path → the bytes to return. A path that is not present returns 404.
   *
   * Files of a cross-repository reference (another repository) are registered under the qualified
   * key
   * `"<repo>@<revision>/<path>"` (the same path string points at different content per repository —
   * without a qualified key the bare path is looked up).
   */
  readonly files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>;
  /** Overrides the content-length header (simulates a mismatch). */
  readonly contentLength?: (path: string) => number | undefined;
  /**
   * A delay inserted before each chunk resolves (for observing the number of concurrent fetches).
   */
  readonly delayMs?: number;
};

export type MockFetch = {
  readonly fetch: typeof globalThis.fetch;
  /** The URLs requested (in order). */
  readonly calls: string[];
  /** The `Authorization` headers sent (in the same order as `calls`; `null` when absent). */
  readonly authorizations: (string | null)[];
  /** The maximum number of fetches that ran at the same time. */
  readonly peakConcurrency: () => number;
};

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * A `fetch` that speaks only HF's two endpoints (the revision resolution API and the resolve URL).
 * The body is always a ReadableStream split into 2 chunks (so that the checks that stop in the
 * middle of receiving are actually exercised).
 */
export const createMockFetch = (routes: MockRoutes): MockFetch => {
  const calls: string[] = [];
  const authorizations: (string | null)[] = [];
  let inFlight = 0;
  let peak = 0;
  const fetchImpl = (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const href = urlOf(input);
    calls.push(href);
    authorizations.push(new Headers(init?.headers).get("authorization"));
    const url = new URL(href);
    const revision = REVISION_RE.exec(url.pathname);
    if (revision !== null) {
      if (routes.sha === undefined) return Promise.resolve(notFound());
      return Promise.resolve(
        new Response(JSON.stringify({ sha: routes.sha }), {
          headers: { "content-type": "application/json" },
        }),
      );
    }
    const resolved = RESOLVE_RE.exec(url.pathname);
    if (resolved === null) return Promise.resolve(notFound());
    const repo = resolved[1];
    const pinned = decodeURIComponent(resolved[2]);
    const path = decodeURIComponent(resolved[3]);
    const bytes = routes.files.get(`${repo}@${pinned}/${path}`) ?? routes.files.get(path);
    if (bytes === undefined) return Promise.resolve(notFound());
    inFlight += 1;
    peak = Math.max(peak, inFlight);
    const declared = routes.contentLength?.(path) ?? bytes.length;
    const signal = init?.signal;
    const half = Math.ceil(bytes.length / 2);
    const chunks = [bytes.subarray(0, half), bytes.subarray(half)];
    let index = 0;
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      inFlight -= 1;
    };
    const body = new ReadableStream<Uint8Array>({
      pull: async (controller) => {
        if (routes.delayMs !== undefined) {
          await new Promise((resolve) => setTimeout(resolve, routes.delayMs));
        }
        if (signal?.aborted === true) {
          finish();
          controller.error(signal.reason);
          return;
        }
        if (index >= chunks.length) {
          finish();
          controller.close();
          return;
        }
        controller.enqueue(new Uint8Array(chunks[index++]));
      },
      cancel: finish,
    });
    return Promise.resolve(new Response(body, { headers: { "content-length": String(declared) } }));
  };
  return { fetch: fetchImpl, calls, authorizations, peakConcurrency: () => peak };
};

const notFound = (): Response =>
  new Response("not found", { status: 404, statusText: "Not Found" });

/**
 * A `fetch` wrapper that causes the same Chrome substitution at the stage of **waiting for the
 * response headers**. Only `victimPath` passes straight through (= can fail first); every other
 * request returns no response until it is aborted, and once aborted rejects with a bare
 * `AbortError` with a fixed message.
 *
 * It has to differ from `withChromeAbortShape` (during the body read) because the fetch layer wraps
 * errors differently depending on the stage — `prefetchUrl` of the warming surface (phase 1)
 * rewraps a transfer abort via the rejection of `cache.put`, so the signal's reason is restored,
 * while an abort during the wait for the response comes up as the bare `AbortError`.
 */
export const abortWhileAwaitingResponse =
  (base: typeof globalThis.fetch, victimPath: string): typeof globalThis.fetch =>
  async (input, init) => {
    const signal = init?.signal;
    if (urlOf(input).endsWith(victimPath) || signal === undefined || signal === null) {
      return await base(input, init);
    }
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 50);
      const done = (): void => {
        clearTimeout(timer);
        resolve();
      };
      if (signal.aborted) done();
      else signal.addEventListener("abort", done, { once: true });
    });
    if (signal.aborted) {
      throw new DOMException("BodyStreamBuffer was aborted", "AbortError");
    }
    return await base(input, init);
  };

/**
 * A `fetch` wrapper that applies Chrome's observed behaviour. **A fetch aborted during the body
 * read fails with a bare `AbortError` with a fixed message instead of `signal.reason`** (Chrome
 * replaces the reason with "BodyStreamBuffer was aborted" — the flag that prevents the replacement
 * is OFF by default on stable).
 *
 * The plain `createMockFetch` passes the reason through as the spec says, so even a fetch that
 * fails
 * as collateral settles carrying "the first failure itself", and the shape in which the root cause
 * is lost cannot be reproduced. Only when fetches other than the real culprit fail with this fixed
 * message does an implementation that picks the error by settle order or by worker array position
 * get exposed.
 */
export const withChromeAbortShape =
  (base: typeof globalThis.fetch): typeof globalThis.fetch =>
  async (input, init) => {
    const response = await base(input, init);
    const signal = init?.signal;
    const source = response.body;
    if (!response.ok || source === null || signal === undefined || signal === null) {
      return response;
    }
    const reader = source.getReader();
    const body = new ReadableStream<Uint8Array>({
      pull: async (controller) => {
        try {
          const { done, value } = await reader.read();
          if (done) controller.close();
          else controller.enqueue(value);
        } catch (error) {
          controller.error(
            signal.aborted ? new DOMException("BodyStreamBuffer was aborted", "AbortError") : error,
          );
        }
      },
      cancel: (reason) => reader.cancel(reason),
    });
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
