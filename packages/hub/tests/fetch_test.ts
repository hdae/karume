import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type AssetProgress,
  type CacheDiagnostic,
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
  assertEquals([...caches.namespaces.keys()], ["karume/1", "karume/1:auth"]);
  assert(authed.calls.length > 0, "無認証キャッシュのヒットに供されてはならない");
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
  assertEquals(error.available.presets, ["f16", "w8a8-s16", "f16-c16"]);
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
