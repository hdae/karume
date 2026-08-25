/**
 * hub のテスト用差し替え（実網・実 Cache Storage を一切叩かない）。
 *
 * 取得層（`@hdae/fetch-cache`）は `fetch` と `caches` の注入口を持つので、hub のテストは
 * そこへこのモックを差し込む。MUST: `caches` は必ず渡す — 渡さないと Deno の実キャッシュ
 * （ディスク）に書きに行く。
 */

/** メモリ上の Cache（`Cache` の実装は hub が使う match / put / delete / keys だけ）。 */
export class MemoryCache implements Cache {
  readonly entries = new Map<string, Uint8Array<ArrayBuffer>>();
  /** put を失敗させる（quota 超過の模擬）。 */
  failPut = false;

  match(request: RequestInfo | URL): Promise<Response | undefined> {
    const key = urlOf(request);
    const bytes = this.entries.get(key);
    return Promise.resolve(bytes === undefined ? undefined : new Response(bytes));
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    if (this.failPut) throw new Error("mock: quota exceeded");
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
    throw new Error("mock: add は使わない");
  }

  addAll(): Promise<void> {
    throw new Error("mock: addAll は使わない");
  }

  matchAll(): Promise<readonly Response[]> {
    throw new Error("mock: matchAll は使わない");
  }
}

/** メモリ上の CacheStorage。名前空間ごとの分離を観測するために名前をそのまま持つ。 */
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
    throw new Error("mock: CacheStorage.match は使わない");
  }
}

const urlOf = (request: RequestInfo | URL): string =>
  typeof request === "string" ? request : request instanceof URL ? request.href : request.url;

export const HUB_URL = "https://hub.test";
export const REPO = "someone/anima";
export const SHA = "0123456789abcdef0123456789abcdef01234567";

/** payload(path) — tests/fixtures/manifest-fetch.json の size / sha256 はこの規則から導出済み。 */
export const payloadFor = (path: string): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(`karume-test:${path}`);

export type MockRoutes = {
  /** revision 解決 API が返す sha。undefined なら 404 を返す。 */
  readonly sha?: string;
  /**
   * path → 返すバイト列。無い path は 404。
   *
   * 越境参照（別リポ）のファイルは `"<repo>@<revision>/<path>"` の修飾キーで登録する
   * （同じ path 文字列が repo ごとに別の実体を指すため — 修飾キーが無ければ素の path を引く）。
   */
  readonly files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>;
  /** content-length ヘッダの上書き（食い違いの模擬）。 */
  readonly contentLength?: (path: string) => number | undefined;
  /** 1 チャンクごとに解決を挟むためのディレイ（同時実行数の観測用）。 */
  readonly delayMs?: number;
};

export type MockFetch = {
  readonly fetch: typeof globalThis.fetch;
  /** 発行された URL（順序どおり）。 */
  readonly calls: string[];
  /** 同時に走った取得の最大数。 */
  readonly peakConcurrency: () => number;
};

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * HF の 2 経路（revision 解決 API・resolve URL）だけを喋る `fetch`。body は必ず
 * ReadableStream で 2 チャンクに割って返す（受信途中で止める門を実際に踏ませるため）。
 */
export const createMockFetch = (routes: MockRoutes): MockFetch => {
  const calls: string[] = [];
  let inFlight = 0;
  let peak = 0;
  const fetchImpl = (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    const href = urlOf(input);
    calls.push(href);
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
    return Promise.resolve(
      new Response(body, { headers: { "content-length": String(declared) } }),
    );
  };
  return { fetch: fetchImpl, calls, peakConcurrency: () => peak };
};

const notFound = (): Response =>
  new Response("not found", { status: 404, statusText: "Not Found" });

/**
 * 同じ Chrome の差し替えを**応答ヘッダ待ち**の段で起こす `fetch` ラッパ。`victimPath` だけは
 * 素通し（＝真っ先に失敗できる）で、それ以外は abort されるまで応答を返さず、abort されたら
 * 固定文言の生 `AbortError` で reject する。
 *
 * `withChromeAbortShape`（body 読み取り中）との違いが要るのは、取得層が段によって別の包み方を
 * するため — 逐次面 相 1 の `prefetchUrl` は転送中断を `cache.put` の reject 経由で包み直すので
 * signal の reason が復元されるが、応答待ちの中断は生の `AbortError` のまま上がってくる。
 */
export const abortWhileAwaitingResponse = (
  base: typeof globalThis.fetch,
  victimPath: string,
): typeof globalThis.fetch =>
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
 * Chrome の観測挙動を被せる `fetch` ラッパ。**body 読み取り中に abort された取得を、
 * `signal.reason` ではなく固定文言の生 `AbortError` で落とす**（Chrome は reason を
 * "BodyStreamBuffer was aborted" に差し替える — 差し替えを防ぐフラグは stable 既定 OFF）。
 *
 * 素の `createMockFetch` は仕様どおり reason をそのまま伝えるため、巻き添えで落ちた取得も
 * 「最初の失敗そのもの」を持って決着してしまい、真因が失われる形を再現できない。真犯人以外が
 * この固定文言で落ちる状況を作って初めて、決着順やワーカーの配列位置で拾う実装が露見する。
 */
export const withChromeAbortShape = (
  base: typeof globalThis.fetch,
): typeof globalThis.fetch =>
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
