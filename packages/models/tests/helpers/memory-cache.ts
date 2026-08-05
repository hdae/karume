/**
 * メモリ上の Cache Storage（e2e が**実 Cache Storage を汚さない**ための差し替え）。
 *
 * MUST: 実キャッシュへ数 GB を書くと、端末のディスクを食うだけでなく、次の run が
 * 「キャッシュヒットで network を叩かない」別経路に化ける（integrity 検証の `source` も
 * `network` → `cache` に変わり、何を検証したのかが run ごとに変わる）。
 *
 * NOTE: hub のテスト helper（`packages/hub/tests/helpers/mock.ts`）と同型だが import しない —
 * パッケージのテストが他パッケージのテスト内部に依存すると、向こうの都合がこちらへ漏れる。
 */
export class MemoryCache implements Cache {
  readonly entries = new Map<string, Uint8Array<ArrayBuffer>>();

  match(request: RequestInfo | URL): Promise<Response | undefined> {
    const bytes = this.entries.get(urlOf(request));
    return Promise.resolve(bytes === undefined ? undefined : new Response(bytes));
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    this.entries.set(urlOf(request), new Uint8Array(await response.arrayBuffer()));
  }

  delete(request: RequestInfo | URL): Promise<boolean> {
    return Promise.resolve(this.entries.delete(urlOf(request)));
  }

  keys(request?: RequestInfo | URL): Promise<readonly Request[]> {
    const urls = request === undefined ? [...this.entries.keys()] : [urlOf(request)];
    return Promise.resolve(urls.map((url) => new Request(url)));
  }

  add(): Promise<void> {
    throw new Error("memory-cache: add は使わない");
  }

  addAll(): Promise<void> {
    throw new Error("memory-cache: addAll は使わない");
  }

  matchAll(): Promise<readonly Response[]> {
    throw new Error("memory-cache: matchAll は使わない");
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
    throw new Error("memory-cache: CacheStorage.match は使わない");
  }
}

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;
