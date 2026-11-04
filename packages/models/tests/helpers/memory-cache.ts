/**
 * An in-memory Cache Storage (the replacement that keeps e2e from **polluting the real Cache
 * Storage**).
 *
 * MUST: writing several GB into the real cache not only eats the machine's disk but also turns the
 * next run into a different path — "a cache hit that never touches the network" (the `source` of
 * the integrity check also changes from `network` to `cache`, so what was checked changes from run
 * to run).
 *
 * NOTE: the same shape as the hub test helper (`packages/hub/tests/helpers/mock.ts`) but not
 * imported — when a package's tests depend on another package's test internals, that side's
 * concerns leak into this one.
 */
export class MemoryCache implements Cache {
  readonly entries = new Map<string, { bytes: Uint8Array<ArrayBuffer>; headers: Headers }>();

  match(request: RequestInfo | URL): Promise<Response | undefined> {
    const entry = this.entries.get(urlOf(request));
    if (entry === undefined) return Promise.resolve(undefined);
    return Promise.resolve(new Response(entry.bytes, { headers: entry.headers }));
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    // MUST: keep the headers — the fetch layer puts the recorded hash in the response headers, so
    // dropping them gives "a cache whose record vanishes every time", and we would measure a
    // full-rehash path that never happens in a real environment.
    const headers = new Headers(response.headers);
    this.entries.set(urlOf(request), {
      bytes: new Uint8Array(await response.arrayBuffer()),
      headers,
    });
  }

  delete(request: RequestInfo | URL): Promise<boolean> {
    return Promise.resolve(this.entries.delete(urlOf(request)));
  }

  keys(request?: RequestInfo | URL): Promise<readonly Request[]> {
    const urls = request === undefined ? [...this.entries.keys()] : [urlOf(request)];
    return Promise.resolve(urls.map((url) => new Request(url)));
  }

  add(): Promise<void> {
    throw new Error("memory-cache: add is not used");
  }

  addAll(): Promise<void> {
    throw new Error("memory-cache: addAll is not used");
  }

  matchAll(): Promise<readonly Response[]> {
    throw new Error("memory-cache: matchAll is not used");
  }
}

export class MemoryCacheStorage implements CacheStorage {
  readonly namespaces = new Map<string, MemoryCache>();

  open(cacheName: string): Promise<Cache> {
    const existing = this.namespaces.get(cacheName);
    if (existing !== undefined) return Promise.resolve(existing);
    const created = new MemoryCache();
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
    throw new Error("memory-cache: CacheStorage.match is not used");
  }
}

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;
