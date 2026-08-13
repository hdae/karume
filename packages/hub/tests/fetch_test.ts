import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  type CacheDiagnostic,
  clearHubCache,
  fetchAssets,
  HubFetchError,
  IntegrityError,
  type LoadedManifest,
  loadManifest,
  ManifestFormatError,
  resolveFiles,
} from "../mod.ts";
import {
  createMockFetch,
  HUB_URL,
  MemoryCache,
  MemoryCacheStorage,
  type MockRoutes,
  payloadFor,
  REPO,
  SHA,
} from "./helpers/mock.ts";

const MANIFEST_PATH = "karume.json";

const ASSET_PATHS = [
  "text_encoder/model.safetensors",
  "text_conditioner/model.safetensors",
  "transformer/model.f16.safetensors",
  "transformer/model.i8.safetensors",
  "transformer/rope_base.safetensors",
  "vae_decoder/model.safetensors",
  "tokenizer/qwen2-tokenizer.json",
];

const manifestBytes = new TextEncoder().encode(
  await Deno.readTextFile(new URL("./fixtures/manifest-fetch.json", import.meta.url)),
);

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

Deno.test("loadManifest: 可変 ref は 1 回だけ解決し、以降は同一 SHA に固定される", async () => {
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  const caches = new MemoryCacheStorage();
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
    fetch: mock.fetch,
    caches,
  });
  assertEquals(loaded.revisionSha, SHA);
  assertEquals(loaded.repo, REPO);
  assertEquals(mock.calls[0], revisionUrl("main"));
  assertEquals(mock.calls[1], resolveUrl(MANIFEST_PATH));

  await fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches });
  assertEquals(countCalls(mock.calls, revisionUrl("main")), 1, "解決はセッション 1 回だけ");
  for (const call of mock.calls.slice(1)) {
    assert(call.includes(`/resolve/${SHA}/`), `${call} が解決済み SHA に固定されていない`);
  }
});

Deno.test("loadManifest: revision に SHA を渡すと解決リクエストが発生しない", async () => {
  const mock = createMockFetch({ files: serveAll() }); // sha 無し = 解決 API は 404
  const caches = new MemoryCacheStorage();
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: mock.fetch,
    caches,
  });
  assertEquals(loaded.revisionSha, SHA);
  assertEquals(mock.calls, [resolveUrl(MANIFEST_PATH)]);
});

Deno.test("loadManifest: 可変 ref の解決失敗はオフライン不可として HubFetchError で報告する", async () => {
  const mock = createMockFetch({ files: serveAll() });
  const caches = new MemoryCacheStorage();
  const error = await assertRejects(
    () => loadManifest({ repo: REPO, hubUrl: HUB_URL }, { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, undefined);
  assert(error.message.includes("オフライン"), "オフライン不可であることを明示する");
});

Deno.test("loadManifest: 1MiB を超える karume.json は受信前に弾く", async () => {
  const oversized = new TextEncoder().encode(" ".repeat(2 * 1024 * 1024));
  const mock = createMockFetch({
    sha: SHA,
    files: serveAll(new Map([[MANIFEST_PATH, oversized]])),
  });
  await assertRejects(
    () =>
      loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
        fetch: mock.fetch,
        caches: new MemoryCacheStorage(),
      }),
    ManifestFormatError,
  );
});

Deno.test("loadManifest: 取得層の 404 は repo / SHA / path の文脈を付けて透過する", async () => {
  const files = serveAll();
  files.delete(MANIFEST_PATH);
  const mock = createMockFetch({ sha: SHA, files });
  const error = await assertRejects(
    () =>
      loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
        fetch: mock.fetch,
        caches: new MemoryCacheStorage(),
      }),
    HubFetchError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, MANIFEST_PATH);
  assert(error.cause instanceof Error, "取得層のエラーを cause に残す");
});

const load = async (routes: MockRoutes, caches: MemoryCacheStorage) => {
  const mock = createMockFetch(routes);
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: mock.fetch,
    caches,
  });
  return { mock, loaded };
};

Deno.test("fetchAssets: 全キーを返し、同一 path は 1 回しか取りに行かない", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(Object.keys(assets), Object.keys(files));
  assertEquals(
    countCalls(mock.calls, resolveUrl("transformer/rope_base.safetensors")),
    1,
    "重複 path は 1 回だけ取得する",
  );
  assert(
    assets["rope_alias"] === assets["transformer.rope_base"],
    "同一 path のキーは同じバイト列を共有する",
  );
  assertEquals(
    assets["transformer"],
    payloadFor("transformer/model.i8.safetensors"),
  );
  // 選ばれなかった variant は取りに行かない。
  assertEquals(countCalls(mock.calls, resolveUrl("transformer/model.f16.safetensors")), 0);
  for (const bytes of Object.values(assets)) {
    assertEquals(bytes.byteOffset, 0);
    assertEquals(bytes.byteLength, bytes.buffer.byteLength);
  }
});

Deno.test("fetchAssets: 進捗総量は manifest の size 合計（path 一意化）で verifying を挟む", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
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

  assert(events.length > 0, "進捗が 1 度も出ない");
  for (const event of events) {
    assertEquals(event.total, expectedTotal, "総量は重複排除された size 合計");
    assert(event.loaded <= event.total, "合計が総量を超えない");
  }
  assert(
    events.some((event) => event.phase === "verifying"),
    "sha256 照合中は verifying フェーズを出す",
  );
  assertEquals(events[events.length - 1].loaded, expectedTotal, "最後は総量に到達する");
});

Deno.test("fetchAssets: 同時取得は 4 本までに絞る", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  assert(mock.peakConcurrency() > 1, "そもそも並行していない");
  assert(mock.peakConcurrency() <= 4, `同時 ${mock.peakConcurrency()} 本まで走った`);
});

Deno.test("fetchAssets: 2 回目はキャッシュから返り network に出ない", async () => {
  const caches = new MemoryCacheStorage();
  const first = await load({ files: serveAll() }, caches);
  const files = resolveFiles(first.loaded.manifest);
  await fetchAssets(first.loaded, files, { fetch: first.mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  const assets = await fetchAssets(first.loaded, files, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  assertEquals(second.calls, [], "キャッシュヒットは network に出ない");
  assertEquals(assets["tokenizer"], payloadFor("tokenizer/qwen2-tokenizer.json"));
  assert(
    events.some((event) => event.phase === "verifying"),
    "キャッシュヒットでも sha256 検証は走る",
  );
});

Deno.test("fetchAssets: Authorization 付き取得は別のキャッシュ名前空間へ隔離される", async () => {
  const caches = new MemoryCacheStorage();
  const anonymous = await load({ files: serveAll() }, caches);
  const files = resolveFiles(anonymous.loaded.manifest);
  await fetchAssets(anonymous.loaded, files, { fetch: anonymous.mock.fetch, caches });
  assertEquals([...caches.namespaces.keys()], ["karume/1"]);

  const authed = createMockFetch({ files: serveAll() });
  await fetchAssets(anonymous.loaded, files, {
    fetch: authed.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  const names = [...caches.namespaces.keys()];
  assertEquals(names.length, 2);
  assertEquals(names[0], "karume/1");
  assert(names[1].startsWith("karume/1:auth"), `${names[1]} が認証隔離の名前空間でない`);
  assert(!names[1].includes("hf_token"), `${names[1]} に生の credential が出ている`);
  assert(authed.calls.length > 0, "無認証キャッシュのヒットに供されてはならない");
});

Deno.test("fetchAssets: credential が違えば同じ URL でもキャッシュを共有しない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);

  const tokenA = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, {
    fetch: tokenA.fetch,
    caches,
    headers: { authorization: "Bearer token-A" },
  });
  assert(tokenA.calls.length > 0, "そもそも取得していない");

  const tokenB = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, {
    fetch: tokenB.fetch,
    caches,
    headers: { authorization: "Bearer token-B" },
  });
  assertEquals(
    tokenB.calls.length,
    tokenA.calls.length,
    "別 credential が token A の写しにヒットしてはならない",
  );

  const again = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, {
    fetch: again.fetch,
    caches,
    headers: { authorization: "Bearer token-A" },
  });
  assertEquals(again.calls, [], "同一 credential の再要求はキャッシュに当たる");
});

Deno.test("fetchAssets: sha256 の食い違いは IntegrityError（文脈と利用可能ラベルつき）", async () => {
  const caches = new MemoryCacheStorage();
  const corrupt = new TextEncoder().encode("karume-test:tampered-payload-XXXXXXXXXXXX");
  const path = "vae_decoder/model.safetensors";
  assertEquals(corrupt.byteLength, payloadFor(path).byteLength, "長さは合わせ sha256 だけ外す");
  const { mock, loaded } = await load({ files: serveAll(new Map([[path, corrupt]])) }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    IntegrityError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, path);
  assertEquals(error.source, "network");
  assertEquals(error.expected.length, 64);
  assert(error.expected !== error.actual);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

Deno.test("fetchAssets: content-length が size と食い違えば受信前に止める", async () => {
  const caches = new MemoryCacheStorage();
  const path = "tokenizer/qwen2-tokenizer.json";
  const { mock, loaded } = await load({
    files: serveAll(),
    contentLength: (target) => target === path ? payloadFor(path).byteLength + 1 : undefined,
  }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    IntegrityError,
  );
  assertEquals(error.path, path);
  assertEquals(error.actual, String(payloadFor(path).byteLength + 1));
  assert(error.message.includes("content-length"));
});

Deno.test("fetchAssets: 受信バイトが size を超えた時点で abort する", async () => {
  const caches = new MemoryCacheStorage();
  const path = "text_encoder/model.safetensors";
  const declared = payloadFor(path).byteLength;
  const bloated = new TextEncoder().encode(`karume-test:${path}${"!".repeat(64)}`);
  const { mock, loaded } = await load({
    files: serveAll(new Map([[path, bloated]])),
    // content-length は正しい値を主張しつつ、body だけ多く流す。
    contentLength: (target) => target === path ? declared : undefined,
  }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    IntegrityError,
  );
  assertEquals(error.path, path);
  assertEquals(error.expected, String(declared));
  assert(Number(error.actual) > declared);
});

Deno.test("fetchAssets: AbortSignal は全取得へ透過する", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const error = await assertRejects(() =>
    fetchAssets(loaded, resolveFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort(),
    })
  );
  assert(error instanceof DOMException && error.name === "AbortError", `${error} が中断でない`);
});

Deno.test("fetchAssets: abort(reason) の custom Error はそのまま伝播する（取得失敗に化けない）", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const reason = new Error("app: ユーザーがロードを取り消した");
  const error = await assertRejects(() =>
    fetchAssets(loaded, resolveFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort(reason),
    })
  );
  assertStrictEquals(error, reason, "呼び出し側が渡した reason が別のエラーに包まれている");
});

Deno.test("fetchAssets: abort(reason) が primitive でもそのまま伝播する", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const error = await assertRejects(() =>
    fetchAssets(loaded, resolveFiles(loaded.manifest), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      onProgress: () => controller.abort("app:cancelled"),
    })
  );
  assertStrictEquals(error, "app:cancelled");
});

Deno.test("fetchAssets: cache I/O の失敗はアプリへ届く診断になる（取得は落とさない）", async () => {
  const caches = new MemoryCacheStorage();
  const namespace = new MemoryCache();
  namespace.failPut = true;
  caches.namespaces.set("karume/1", namespace);

  const mock = createMockFetch({ files: serveAll() });
  const diagnostics: CacheDiagnostic[] = [];
  const loaded: LoadedManifest = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    { fetch: mock.fetch, caches, onCacheError: (entry) => diagnostics.push(entry) },
  );
  const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest), {
    fetch: mock.fetch,
    caches,
    onCacheError: (entry) => diagnostics.push(entry),
  });
  assertEquals(Object.keys(assets).length, 7, "cache が死んでも取得は成立する");
  assert(diagnostics.length > 0, "quota 失敗が黙って握り潰されている");
  assertEquals(new Set(diagnostics.map((entry) => entry.op)), new Set(["put"]));
});

/** `globalThis.caches` を差し替えて `body` を走らせ、必ず元へ戻す（実キャッシュを触らせない）。 */
const withGlobalCaches = async (
  value: CacheStorage | undefined,
  body: () => Promise<void>,
): Promise<void> => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "caches");
  if (original === undefined) throw new Error("この環境に globalThis.caches が無い");
  Object.defineProperty(globalThis, "caches", { value, configurable: true });
  try {
    await body();
  } finally {
    Object.defineProperty(globalThis, "caches", original);
  }
};

const populated = async (...names: readonly string[]): Promise<MemoryCacheStorage> => {
  const caches = new MemoryCacheStorage();
  for (const name of names) await caches.open(name);
  return caches;
};

Deno.test("clearHubCache: karume の 2 名前空間だけを消す（他コードの名前空間は残す）", async () => {
  const caches = await populated("karume/1", "karume/1:auth", "other/1");
  assertEquals(await clearHubCache({ caches }), true);
  assertEquals([...caches.namespaces.keys()], ["other/1"]);
});

Deno.test("clearHubCache: 認証側だけ残っていても消して true を返す", async () => {
  const caches = await populated("karume/1:auth");
  assertEquals(await clearHubCache({ caches }), true, "gated 資産の写しを残さない");
  assertEquals([...caches.namespaces.keys()], []);
});

Deno.test("clearHubCache: credential ごとの認証名前空間も残さず消す", async () => {
  const caches = await populated(
    "karume/1",
    "karume/1:auth:0123456789abcdef",
    "karume/1:auth:fedcba9876543210",
    "other/1",
  );
  assertEquals(await clearHubCache({ caches }), true);
  assertEquals([...caches.namespaces.keys()], ["other/1"]);
});

Deno.test("clearHubCache: 実際に埋まった無認証 / 認証の名前空間を両方消す", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  assertEquals(caches.namespaces.size, 2, "無認証と認証で 2 つ埋まっている");

  assertEquals(await clearHubCache({ caches }), true);
  assertEquals([...caches.namespaces.keys()], []);

  const after = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, {
    fetch: after.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  assert(after.calls.length > 0, "消した後の認証取得が network に出ていない");
});

Deno.test("clearHubCache: 消すものが 1 つも無ければ false", async () => {
  const caches = await populated("other/1");
  assertEquals(await clearHubCache({ caches }), false);
  assertEquals([...caches.namespaces.keys()], ["other/1"]);
});

Deno.test("clearHubCache: caches を渡すとそちらだけを消す（globalThis には触らない）", async () => {
  const injected = await populated("karume/1");
  const global = await populated("karume/1");
  await withGlobalCaches(global, async () => {
    assertEquals(await clearHubCache({ caches: injected }), true);
  });
  assertEquals([...injected.namespaces.keys()], []);
  assertEquals([...global.namespaces.keys()], ["karume/1"]);
});

Deno.test("clearHubCache: caches 省略時は globalThis.caches を消す", async () => {
  const global = await populated("karume/1", "karume/1:auth");
  await withGlobalCaches(global, async () => {
    assertEquals(await clearHubCache(), true);
  });
  assertEquals([...global.namespaces.keys()], []);
});

Deno.test("clearHubCache: CacheStorage が無い環境は fail loudly（黙って no-op にしない）", async () => {
  await withGlobalCaches(undefined, async () => {
    const error = await assertRejects(() => clearHubCache(), Error);
    assert(error.message.includes("CacheStorage"), `${error.message} が原因を名指ししていない`);
  });
});
