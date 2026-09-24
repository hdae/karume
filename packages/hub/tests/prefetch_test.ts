/**
 * 温め面（`prefetchAssets` — `source.ts` ④の相 1）の振る舞い。バイト列を手元に持たない面なので、
 * 観測できるのは「何回 network へ出たか」「進捗がどう出たか」「失敗がどう上がったか」の 3 つだけ
 * で、取れたバイト列の正しさは**後続の全量面**（`fetchAssets`）が読めることで見る。
 *
 * 全量面そのものの契約（進捗の数値・律速・self-heal・越境）は `fetch_test.ts` の担当で、ここでは
 * 重複させない。ローカル取得元（相 1 を持たない取得元）側の no-op は `local_test.ts`。
 */

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

/** 越境参照が名乗る世代（自リポの同じ path を別世代から引く形 — ADR 0038 §7）。 */
const CROSS_REVISION = "89abcdef0123456789abcdef0123456789abcdef";

/** fixture が宣言する全ファイル（長さ 0 の part は中身が無いので配信しない）。 */
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

/** 長さは保ったまま中身だけ変える（size ではなく sha256 の門を踏ませる）。 */
const tamper = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const copy = new Uint8Array(bytes);
  copy[copy.length - 1] ^= 0xff;
  return copy;
};

/**
 * ファイル列（= 温め面の入力）。`selectionRefs` は長さ 0 の part を落とし、同一実体
 * （`style_alias` のような別名）を畳んだ列を宣言順で返すので、そのまま渡せる。
 */
const assetRefs = (loaded: LoadedManifest): FileRef[] => [
  ...selectionRefs(resolveSelection(loaded.manifest)),
];

/** 温め面の入力をそのまま全量面の表へ（キーは path — 1 対 1 で突き合わせる）。 */
const filesOf = (refs: readonly FileRef[]): Record<string, FileRef> =>
  Object.fromEntries(refs.map((ref) => [ref.path, ref]));

/** manifest は別 mock で読み、温め面の観測用に呼び出し記録が空の mock を渡す。 */
const prepare = async (
  routes: MockRoutes,
  caches: MemoryCacheStorage,
): Promise<
  { loaded: LoadedManifest; refs: FileRef[]; mock: ReturnType<typeof createMockFetch> }
> => {
  const loader = createMockFetch(routes);
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: loader.fetch,
    caches,
  });
  return { loaded, refs: assetRefs(loaded), mock: createMockFetch(routes) };
};

/**
 * URL 述語に一致する取得の**最初の 1 回だけ**を `429 Too Many Requests`（`retry-after: 0`）へ
 * 差し替え、以後は元の `fetch` へ委譲するラッパ（取得層の再試行を 1 回だけ踏ませる）。
 *
 * 差し替えの前に元の `fetch` を必ず 1 回呼ぶ — こうしないと 429 になった要求が mock の
 * 呼び出し記録に残らず、「取り直したか」を要求回数で見られない。
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

Deno.test("prefetchAssets: complete をファイル 1 本につき 1 回出し、loaded は合計へ着地する", async () => {
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
  assertEquals(completes.length, refs.length, "complete がファイル数と一致しない");
  assertEquals(
    new Set(completes.map((event) => event.path)),
    new Set(refs.map((ref) => ref.path)),
    "complete を出していないファイルがある",
  );
  for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total, "最後のイベントが合計に着地していない");
  // 1 ファイルの phase は downloading* → complete の順で、complete の後は出ない。
  const lastPhase = new Map<string, AssetPhase>();
  for (const event of events) {
    if (lastPhase.get(event.path) === "complete") {
      throw new Error(`complete の後に ${event.phase} が出ている（${event.path}）`);
    }
    lastPhase.set(event.path, event.phase);
  }
});

Deno.test("prefetchAssets: 落とした後の fetchAssets は 1 度も network に出ない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });
  const warmed = mock.calls.length;
  assert(warmed > 0, "prefetch が 1 本も取得していない");

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, filesOf(refs), { fetch: second.fetch, caches });

  assertEquals(Object.keys(assets), refs.map((ref) => ref.path));
  for (const ref of refs) assertEquals(assets[ref.path], payloadFor(ref.path));
  assertEquals(second.calls, [], "温めた後の全量面が network に出ている");
  assertEquals(mock.calls.length, warmed, "1 回目の mock も追加で呼ばれている");
});

Deno.test("prefetchAssets: キャッシュ済みのファイルは complete 1 点だけを出す", async () => {
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

  assertEquals(second.calls, [], "温まっているのに network へ出ている");
  assertEquals(events.length, refs.length, "downloading が混ざっている（キャッシュヒットの列）");
  assertEquals(new Set(events.map((event) => event.phase)), new Set<AssetPhase>(["complete"]));
});

Deno.test("prefetchAssets: 全ファイルがキャッシュ済みでも中断は 1 ファイルごとに効く", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const controller = new AbortController();
  const reason = new Error("app: ユーザーがロードを取り消した");
  const completes: string[] = [];

  const error = await assertRejects(() =>
    prefetchAssets(loaded, refs, {
      fetch: second.fetch,
      caches,
      signal: controller.signal,
      // 全ファイルが温まっているので downloading は出ない（complete 1 点だけの列）。最初の
      // 1 本を終えた直後に取り消す。
      onProgress: (progress) => {
        completes.push(progress.path);
        controller.abort(reason);
      },
    })
  );

  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(second.calls, [], "温まっているのに network へ出ている（中断の穴を隠している）");
  // 中断確認が下層（network）にしか無いと、残り全 ref をキャッシュから舐め切ってから決着する。
  // 同時 4 本ぶんは既に飛び込み済みで最後まで進むので、上限ではなく「舐め切らない」を見る。
  assert(refs.length > 4, `同時本数より多い列で試していない（refs ${refs.length}）`);
  assert(
    completes.length < refs.length,
    `取り消し後も残りを舐め切っている（complete ${completes.length} / ${refs.length}）`,
  );
});

Deno.test("prefetchAssets: 取得中の中断は HubFetchError に包まれず素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const reason = new Error("app: ユーザーがロードを取り消した");

  const error = await assertRejects(() =>
    prefetchAssets(loaded, refs, {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      // 受信が始まった最初の 1 本の途中で取り消す（キャッシュヒットの列ではない側の窓）。
      onProgress: (progress) => {
        if (progress.phase === "downloading") controller.abort(reason);
      },
    })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
});

Deno.test("prefetchAssets: 1 本の失敗は真因を復元して HubFetchError で上がる", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { loaded, refs, mock } = await prepare({ files: served, delayMs: 5 }, caches);
  // **先頭以外**の 1 本だけを落とす。1 本の失敗は残り全部を abort するので、巻き添え側は生の
  // AbortError として決着する — ワーカーの配列位置で拾うと、真犯人が worker[0] 以外のときに
  // 巻き添えが表面化して真因（どのファイルがなぜ落ちたか）が消える。
  const victim = refs[1].path;
  assert(refs[0].path !== victim, "先頭以外が落ちる形になっていない");
  served.delete(victim);

  const error = await assertRejects(
    () =>
      prefetchAssets(loaded, refs, {
        fetch: abortWhileAwaitingResponse(mock.fetch, victim),
        caches,
      }),
    HubFetchError,
  );
  assertEquals(error.path, victim, "落ちたのとは別のファイルが報告されている");
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assert(error.message.includes(victim), `${error.message} が落ちた path を名乗っていない`);
});

Deno.test("prefetchAssets: 429 は onRetry で届き、温め直したバイト列が後続の全量面で揃う", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const target = resolveUrl(refs[0].path);
  const retries: RetryDiagnostic[] = [];

  await prefetchAssets(loaded, refs, {
    fetch: rateLimitOnce(mock.fetch, (url) => url === target),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });

  assertEquals(retries.length, 1, "再試行の通知が 1 回だけ届いていない");
  assertEquals(retries[0].url, target, "通知が別のファイルの URL を名乗っている");
  assertEquals(retries[0].status, 429);
  assertEquals(countCalls(mock.calls, target), 2, "429 の後に温め直していない");

  // 温め直した中身が正しいことは、network に出ない全量面で読めることで見る。
  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, filesOf(refs), { fetch: second.fetch, caches });
  assertEquals(assets[refs[0].path], payloadFor(refs[0].path), "温め直した中身が違う");
  assertEquals(second.calls, [], "温めきれていない（全量面が取り直している）");
});

Deno.test("prefetchAssets: sha256 不一致は fail loud で、キャッシュにエントリを残さない", async () => {
  const caches = new MemoryCacheStorage();
  const path = TOKENIZER;
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[path, tamper(payloadFor(path))]])) },
    caches,
  );
  // 並行ワーカーの巻き添えを排して「落ちた 1 本」だけを見る。
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
  assert(error.cause instanceof Error, "下層の不一致を cause に残す");
  assertEquals(
    hasEntry(hubCache(caches), tamper(payloadFor(path))),
    false,
    "不一致のバイト列がキャッシュに残っている",
  );
});

Deno.test("prefetchAssets: キャッシュを開けない環境では素 fetch へ縮退せず fail loud", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  // この面はバイト列を手元に持たないので、キャッシュが使えないなら「温めたつもりで何も残って
  // いない」への沈黙縮退はできない（縮退させると後続の読みが黙って network に出る）。
  const broken = {
    open: () => Promise.reject(new Error("cache storage open failure")),
  } as unknown as CacheStorage;

  const error = await assertRejects(
    () => prefetchAssets(loaded, refs, { fetch: mock.fetch, caches: broken }),
    HubFetchError,
  );
  assert(error.cause instanceof Error, "下層の失敗を cause に残す");
  assertEquals(mock.calls, [], "キャッシュを開けないのに本体の DL が走っている");
});

Deno.test("prefetchAssets: 空・重複は network に出る前に ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  const empty = await assertRejects(
    () => prefetchAssets(loaded, [], { fetch: mock.fetch, caches }),
    ManifestReferenceError,
  );
  assert(empty.message.includes("prefetchAssets"), `${empty.message} が面の名前を名乗っていない`);
  assertEquals(empty.available.models, ["anima-turbo", "anima-lite"]);

  const duplicated = await assertRejects(
    () => prefetchAssets(loaded, [refs[0], refs[0]], { fetch: mock.fetch, caches }),
    ManifestReferenceError,
  );
  assert(
    duplicated.message.includes(refs[0].path),
    `${duplicated.message} が重複 path を名指ししていない`,
  );
  assertEquals(mock.calls, [], "入力検査より先に network へ出ている");
});

Deno.test("prefetchAssets: 重複判定は fileRefKey で見る（同じ path でも revision が違えば別の 1 本）", async () => {
  const caches = new MemoryCacheStorage();
  const path = ASSET_PATHS[0];
  // 同じ path を別世代からも配る。中身は変えておく — 同一バイト列だと取得層の内容キーで
  // 片方がキャッシュヒットになり、「2 本取りに行ったか」を要求回数で見られない。
  const crossBytes = payloadFor(`${CROSS_REVISION}/${path}`);
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[`${REPO}@${CROSS_REVISION}/${path}`, crossBytes]])) },
    caches,
  );
  const own = refs.find((ref) => ref.path === path);
  assert(own !== undefined, `${path} が選択の宣言に無い`);
  const cross: FileRef = {
    path,
    size: crossBytes.byteLength,
    sha256: await sha256Hex(crossBytes),
    repo: REPO,
    revision: CROSS_REVISION,
  };

  // 重複を path で畳む実装なら、ここが ManifestReferenceError になる。
  await prefetchAssets(loaded, [own, cross], { fetch: mock.fetch, caches });

  assertEquals(countCalls(mock.calls, resolveUrl(path)), 1);
  assertEquals(
    countCalls(mock.calls, `${HUB_URL}/${REPO}/resolve/${CROSS_REVISION}/${path}`),
    1,
    "越境参照ぶんが自リポ参照へ畳まれて温められていない",
  );
});
