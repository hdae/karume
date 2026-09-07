import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  type CacheDiagnostic,
  clearHubCache,
  type DirectoryAdapter,
  fetchAssets,
  type FileRef,
  HubFetchError,
  type LoadedManifest,
  loadManifest,
  localDirectory,
  ManifestFormatError,
  openAsset,
  prefetchAssets,
  resolveFiles,
  type RetryDiagnostic,
} from "../mod.ts";
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

/** 長さは保ったまま中身だけ変える（size ではなく sha256 の門を踏ませる）。 */
const tamper = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const copy = new Uint8Array(bytes);
  copy[copy.length - 1] ^= 0xff;
  return copy;
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

/** `console.warn` を差し替えて `body` を走らせ、必ず元へ戻す（出た文言をそのまま返す）。 */
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

Deno.test("loadManifest: 可変 ref は 1 回だけ解決し、以降は同一 SHA に固定される", async () => {
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  const caches = new MemoryCacheStorage();
  // revision を渡さない経路なので pin の案内が出る（文言の検査は専用テスト側）。
  let loaded!: LoadedManifest;
  await captureWarnings(async () => {
    loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL }, { fetch: mock.fetch, caches });
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

Deno.test("loadManifest: revision 未指定の main 解決だけを 1 回 warn する", async (t) => {
  const load = async (revision?: string): Promise<string[]> => {
    const mock = createMockFetch({ sha: SHA, files: serveAll() });
    const caches = new MemoryCacheStorage();
    return await captureWarnings(async () => {
      await loadManifest({
        repo: REPO,
        hubUrl: HUB_URL,
        ...(revision === undefined ? {} : { revision }),
      }, { fetch: mock.fetch, caches });
    });
  };

  await t.step("revision 省略なら 1 回だけ出て、解決した SHA と 2 択の案内を載せる", async () => {
    const warnings = await load();
    assertEquals(warnings.length, 1, "警告が出ない / 複数回出ている");
    const [warning] = warnings;
    // コピーでそのまま pin になる形（SHA が本文に無いと案内が実行不能になる）。
    assert(warning.includes(SHA), `${warning} が解決した commit SHA を印字していない`);
    assert(warning.includes(REPO), `${warning} がどのリポの話か示していない`);
    assert(warning.includes(`revision: "${SHA}"`), `${warning} が pin の書き方を出していない`);
    assert(warning.includes("_SOURCES"), `${warning} が models の対応表へ誘導していない`);
  });

  await t.step("'main' の明示指定では出さない（可変 ref でよいという意思表示）", async () => {
    assertEquals(await load("main"), []);
  });

  await t.step("SHA 指定では出さない", async () => {
    assertEquals(await load(SHA), []);
  });

  await t.step("タグ・ブランチの明示指定でも出さない", async () => {
    assertEquals(await load("v1.0"), []);
  });

  await t.step("解決に失敗したときは warn ではなく失敗そのものが上がる", async () => {
    const mock = createMockFetch({ files: serveAll() }); // sha 無し = 解決 API は 404
    const warnings = await captureWarnings(async () => {
      await assertRejects(
        () =>
          loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
            fetch: mock.fetch,
            caches: new MemoryCacheStorage(),
          }),
        HubFetchError,
      );
    });
    assertEquals(warnings, [], "解決できていないのに pin を勧めている");
  });
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

Deno.test("loadManifest: 1MiB を超える karume.json は ManifestFormatError で弾く（判定は全量受信後）", async () => {
  const oversized = new TextEncoder().encode(" ".repeat(2 * 1024 * 1024));
  const mock = createMockFetch({
    sha: SHA,
    files: serveAll(new Map([[MANIFEST_PATH, oversized]])),
  });
  await captureWarnings(async () => {
    await assertRejects(
      () =>
        loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
          fetch: mock.fetch,
          caches: new MemoryCacheStorage(),
        }),
      ManifestFormatError,
    );
  });
});

Deno.test("loadManifest: 取得層の 404 は repo / SHA / path の文脈を付けて透過する", async () => {
  const files = serveAll();
  files.delete(MANIFEST_PATH);
  const mock = createMockFetch({ sha: SHA, files });
  let error!: HubFetchError;
  await captureWarnings(async () => {
    error = await assertRejects(
      () =>
        loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
          fetch: mock.fetch,
          caches: new MemoryCacheStorage(),
        }),
      HubFetchError,
    );
  });
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, MANIFEST_PATH);
  assert(error.cause instanceof Error, "取得層のエラーを cause に残す");
});

Deno.test("loadManifest: revision 解決の 429 は onRetry で届き、取り直して manifest が揃う", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ sha: SHA, files: serveAll() });
  const retries: RetryDiagnostic[] = [];
  let loaded!: LoadedManifest;
  await captureWarnings(async () => {
    loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL }, {
      fetch: rateLimitOnce(mock.fetch, (url) => url === revisionUrl("main")),
      caches,
      onRetry: (diagnostic) => retries.push(diagnostic),
    });
  });
  assertEquals(loaded.revisionSha, SHA, "429 の後に解決できていない");
  assertEquals(
    retries,
    [{ url: revisionUrl("main"), status: 429, attempt: 1, delayMs: 0, retryAfter: "0" }],
    "再試行の通知が届いていない / 中身が欠けている",
  );
});

Deno.test("loadManifest: karume.json の 429 も onRetry で届く", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const target = resolveUrl(MANIFEST_PATH);
  const retries: RetryDiagnostic[] = [];
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: rateLimitOnce(mock.fetch, (url) => url === target),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });
  assertEquals(loaded.manifest.available.models, ["anima-turbo", "anima-lite"]);
  assertEquals(retries.length, 1, "再試行の通知が 1 回だけ届いていない");
  assertEquals(retries[0].url, target, "通知が karume.json 以外の URL を名乗っている");
  assertEquals(retries[0].status, 429);
  assertEquals(countCalls(mock.calls, target), 2, "429 の後に取り直していない");
});

Deno.test("loadManifest: 破損した cached karume.json は self-heal で 1 往復だけ取り直す", async () => {
  const caches = new MemoryCacheStorage();
  const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
  const first = createMockFetch({ files: serveAll() });
  await loadManifest(ref, { fetch: first.fetch, caches });

  // ① UTF-8 として読めないバイト列（0xff は UTF-8 に現れない）。
  overwriteEntry(hubCache(caches), manifestBytes, new Uint8Array([0xff, 0xfe, 0xff]), {
    keepRecord: true,
  });
  const healedDecode = createMockFetch({ files: serveAll() });
  const afterDecodeBreak = await loadManifest(ref, { fetch: healedDecode.fetch, caches });
  assertEquals(
    afterDecodeBreak.manifest.available.models,
    ["anima-turbo", "anima-lite"],
    "破損キャッシュから復帰できていない",
  );
  assertEquals(
    countCalls(healedDecode.calls, resolveUrl(MANIFEST_PATH)),
    1,
    "self-heal は 1 往復だけ",
  );

  // ② decode は通るが JSON として壊れている。
  overwriteEntry(
    hubCache(caches),
    manifestBytes,
    new TextEncoder().encode('{"format": "karume/4"'),
    { keepRecord: true },
  );
  const healedParse = createMockFetch({ files: serveAll() });
  const afterParseBreak = await loadManifest(ref, { fetch: healedParse.fetch, caches });
  assertEquals(afterParseBreak.manifest.available.models, ["anima-turbo", "anima-lite"]);
  assertEquals(
    countCalls(healedParse.calls, resolveUrl(MANIFEST_PATH)),
    1,
    "self-heal は 1 往復だけ",
  );
});

Deno.test("loadManifest: 真実源の karume.json が壊れていれば ManifestFormatError（キャッシュにも残さない）", async () => {
  const caches = new MemoryCacheStorage();
  const broken = new TextEncoder().encode('{"format": "karume/4"');
  const mock = createMockFetch({ files: serveAll(new Map([[MANIFEST_PATH, broken]])) });
  await assertRejects(
    () =>
      loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, { fetch: mock.fetch, caches }),
    ManifestFormatError,
  );
  assertEquals(
    hubCache(caches).entries.size,
    0,
    "壊れた manifest をキャッシュに残している",
  );
});

Deno.test("loadManifest: 完全キャッシュ済みでも中断済み signal なら manifest を返さない", async () => {
  const caches = new MemoryCacheStorage();
  const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
  const first = createMockFetch({ files: serveAll() });
  await loadManifest(ref, { fetch: first.fetch, caches });

  const controller = new AbortController();
  const reason = new Error("app: 起動を取り消した");
  controller.abort(reason);
  const second = createMockFetch({ files: serveAll() });
  const error = await assertRejects(() =>
    loadManifest(ref, { fetch: second.fetch, caches, signal: controller.signal })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(second.calls, [], "中断済みなのに network へ出ている");
  assertEquals(
    hubCache(caches).entries.size,
    1,
    "中断を破損と取り違えてキャッシュを捨てている",
  );
});

Deno.test("loadManifest: 旧名前空間（karume/1 系）を入口で回収する", async () => {
  const caches = new MemoryCacheStorage();
  // 取得層 0.4 以前が残した写し。新コードは二度と読まないので容量を占めるだけ。
  await caches.open("karume/1");
  await caches.open("karume/1:auth:0123456789abcdef");
  await caches.open("other/1");

  const mock = createMockFetch({ files: serveAll() });
  await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: mock.fetch,
    caches,
  });

  const names = [...caches.namespaces.keys()];
  assertEquals(names.includes("karume/1"), false, "旧名前空間が残っている");
  assertEquals(
    names.some((name) => name.startsWith("karume/1:")),
    false,
    "旧認証隔離の名前空間が残っている",
  );
  assertEquals(names.includes("other/1"), true, "他コードの名前空間まで消している");
  // 残るのは他コードの 1 つと、取得層が今の manifest を入れた 1 つ。
  assertEquals(names.length, 2, `名前空間が想定外の構成: ${JSON.stringify(names)}`);
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

Deno.test("fetchAssets: 進捗総量は manifest の size 合計（path 一意化）", async () => {
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
    assertEquals(
      event.fileTotal,
      uniqueSizes.get(event.path),
      `${event.path}: fileTotal はそのファイル自身の manifest size`,
    );
    assert(
      event.fileLoaded <= event.fileTotal,
      `${event.path}: fileLoaded がそのファイルの size を超えた`,
    );
    if (event.phase !== "downloading") {
      assertEquals(
        event.fileLoaded,
        event.fileTotal,
        `${event.path}: ${event.phase} は全量が揃った点`,
      );
    }
  }
  assertEquals(events[events.length - 1].loaded, expectedTotal, "最後は総量に到達する");
});

Deno.test("fetchAssets: ファイル別の進捗は全体合計とは別に 1 ファイルぶんを表す", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });

  // 受信途中（size 未満）の downloading が出ている = fileLoaded がそのファイルの受信実測である。
  assert(
    events.some((event) => event.phase === "downloading" && event.fileLoaded < event.fileTotal),
    "受信途中の fileLoaded が 1 度も観測できない",
  );
  // 先に終わったファイルのぶんが loaded に積まれた後の downloading は、そのファイル 1 本ぶんの
  // fileLoaded より必ず大きい（同値のままなら per-file 欄が全体合計の写しになっている）。
  assert(
    events.some((event) => event.phase === "downloading" && event.loaded > event.fileLoaded),
    "複数ファイルを落としているのに全体 loaded とファイル別 fileLoaded が食い違わない",
  );
});

/** phase の進む向き（大きいほど後）。 */
const PHASE_RANK: Record<AssetPhase, number> = { downloading: 0, complete: 1 };

/** path ごとの phase 列（複数ファイルの進捗は交錯して届くので path で束ね直す）。 */
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
      `${path}: ${phases[index - 1]} → ${phases[index]} は phase の逆行`,
    );
  }
  assertEquals(phases[phases.length - 1], "complete", `${path}: 終端が complete でない`);
  assertEquals(
    phases.filter((phase) => phase === "complete").length,
    1,
    `${path}: complete はファイルごとに 1 回だけ`,
  );
};

Deno.test("fetchAssets: network 取得の phase は downloading → complete と単調に進む", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
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
    "取得した全 path が進捗に現れる",
  );
  for (const [path, phases] of byPath) {
    assert(phases.includes("downloading"), `${path}: network 取得なのに downloading が無い`);
    assertMonotonic(phases, path);
  }
});

Deno.test("fetchAssets: キャッシュヒットの phase 列は complete の 1 点だけ", async () => {
  const caches = new MemoryCacheStorage();
  const first = await load({ files: serveAll() }, caches);
  const files = resolveFiles(first.loaded.manifest);
  await fetchAssets(first.loaded, files, { fetch: first.mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  await fetchAssets(first.loaded, files, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  assertEquals(second.calls, [], "キャッシュヒットは network に出ない");
  const byPath = phasesByPath(events);
  assert(byPath.size > 0, "進捗が 1 度も出ない");
  for (const [path, phases] of byPath) {
    assertEquals(
      phases,
      ["complete"],
      `${path}: DL していないのに downloading が出た / 観測できない照合を名乗っている`,
    );
  }
  for (const event of events) {
    // downloading が 1 度も出ない経路でも、complete は全量が揃った点なので満たされる。
    assertEquals(
      event.fileLoaded,
      event.fileTotal,
      `${event.path}: キャッシュヒットの ${event.phase} で fileLoaded が size に届いていない`,
    );
    // 全体 loaded は「取得済みバイトの合計」なので、同一イベントの fileLoaded を必ず含む。
    assert(
      event.loaded >= event.fileLoaded,
      `${event.path}: ${event.phase} の全体 loaded がこのファイルぶんを数えていない`,
    );
  }
});

// 律速は「本数」ではなく「in-flight の size 合計」（バイト予算）。予算そのものの境界挙動は
// tests/concurrency_test.ts が単体で凍結し、ここは面としての観測 — 予算に収まる限り本数では
// 絞らないこと・送出順が resolveFiles の順に決まることを見る。
Deno.test("fetchAssets: 予算に収まる限り本数では絞らない（律速はバイト予算）", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const files = resolveFiles(loaded.manifest);
  // fixture の全ファイルを足しても予算（1.5GiB）に遠く及ばないので、全本が同時に走る。
  const uniquePaths = new Set(Object.values(files).map((ref) => ref.path));
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  assertEquals(
    mock.peakConcurrency(),
    uniquePaths.size,
    `予算に収まるのに同時 ${mock.peakConcurrency()} 本で頭打ちになった`,
  );
});

Deno.test("fetchAssets: 送出順は resolveFiles の順（予算待ちを後続に追い越させない）", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll(), delayMs: 5 }, caches);
  const files = resolveFiles(loaded.manifest);
  // path 一意化の後、宣言順に 1 回ずつ。
  const expected: string[] = [];
  for (const ref of Object.values(files)) {
    const url = resolveUrl(ref.path);
    if (!expected.includes(url)) expected.push(url);
  }
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  // calls[0] は load() が出した manifest 取得。
  assertEquals(mock.calls.slice(1), expected);
});

/** 取得対象の path を送出順（= resolveFiles 順の path 一意化）に並べる。 */
const dispatchOrder = (files: Readonly<Record<string, { path: string }>>): string[] => {
  const order: string[] = [];
  for (const ref of Object.values(files)) {
    if (!order.includes(ref.path)) order.push(ref.path);
  }
  return order;
};

Deno.test("fetchAssets: 巻き添えではなく真の第一失敗が表面化する", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { mock, loaded } = await load({ files: served, delayMs: 5 }, caches);
  const files = resolveFiles(loaded.manifest);
  const order = dispatchOrder(files);
  // **先頭以外**の 1 本だけを 404 にする。1 本の失敗は残り全部を abort するので、巻き添え側は
  // 生の AbortError（Chrome ではさらに固定文言へ差し替えられる）として決着する — 決着順や
  // ワーカーの配列位置で拾うと、真犯人ではないその 1 つが表面化して真因が消える。
  const victim = order[order.length - 1];
  assert(order.indexOf(victim) > 0, "先頭以外が落ちる形になっていない");
  served.delete(victim);

  const error = await assertRejects(
    () => fetchAssets(loaded, files, { fetch: withChromeAbortShape(mock.fetch), caches }),
    HubFetchError,
  );
  assertEquals(error.path, victim, "落ちたのとは別のファイルが報告されている");
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assert(error.message.includes(victim), `${error.message} が落ちた path を名乗っていない`);
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
  assertEquals(
    events.filter((event) => event.phase === "complete").length,
    new Set(Object.values(files).map((ref) => ref.path)).size,
    "キャッシュヒットでも complete は全ファイルぶん出る",
  );
});

Deno.test("fetchAssets: 記録ハッシュが一致するヒットは中身を読み直さずに信じる", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // 記録（取得時に焼かれたヘッダ）はそのままに、中身だけ差し替える。取得層の既定は
  // 「記録が期待と一致すれば信じる」（ローカル格納の信頼 — knob なしの裁定）なので、全量
  // ハッシュは走らず、この壊れたバイト列がそのまま返る。**トラストの範囲を明示する門**で、
  // ここが赤くなったなら再ハッシュが復活した（＝毎起動の全量ハッシュが戻った）ということ。
  const path = "vae_decoder/model.safetensors";
  const tampered = tamper(payloadFor(path));
  overwriteEntry(hubCache(caches), payloadFor(path), tampered, { keepRecord: true });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });
  assertEquals(assets["vae_decoder"], tampered, "記録一致のヒットで全量ハッシュが走っている");
  assertEquals(second.calls, [], "記録が一致しているのに取り直している");
});

Deno.test("fetchAssets: 記録の無い破損エントリは実ハッシュが捕まえ、1 往復で治る", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // 記録ごと落とす = 旧版 / 無検証 prefetch 由来のエントリと同じ形。読出し側は実ハッシュで
  // 突合するので、壊れていれば evict → 取り直し（self-heal）になる。
  const path = "vae_decoder/model.safetensors";
  overwriteEntry(hubCache(caches), payloadFor(path), tamper(payloadFor(path)), {
    keepRecord: false,
  });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });

  assertEquals(assets["vae_decoder"], payloadFor(path), "破損キャッシュが素通りしている");
  assertEquals(countCalls(second.calls, resolveUrl(path)), 1, "self-heal は 1 往復だけ");
  assertEquals(second.calls.length, 1, "壊れていないファイルまで取り直している");
});

Deno.test("fetchAssets: 記録が一致してもバイト数が manifest と違えば取り直す", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  // 記録は一致 = 全量ハッシュは走らない状態で、長さだけが manifest と食い違うエントリ。
  // `expectedBytes` の門（取得層 HF 層の検証）だけがこれを捕まえられる。
  const path = "text_encoder/model.safetensors";
  const truncated = payloadFor(path).slice(0, payloadFor(path).byteLength - 1);
  overwriteEntry(hubCache(caches), payloadFor(path), truncated, { keepRecord: true });

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });
  assertEquals(assets["text_encoder"], payloadFor(path), "長さの違うエントリが素通りしている");
  assertEquals(countCalls(second.calls, resolveUrl(path)), 1, "self-heal は 1 往復だけ");
});

Deno.test("fetchAssets: content-length は正しいのに body が足りない取得は fail loudly", async () => {
  const caches = new MemoryCacheStorage();
  const path = "text_conditioner/model.safetensors";
  const full = payloadFor(path);
  const short = full.slice(0, full.byteLength - 2);
  // content-length は manifest の size を主張しつつ、body だけ短く流す（取得層の上限は
  // 超過しか見ないので、ここを止めるのは取得層の検証だけ）。
  const { mock, loaded } = await load({
    files: serveAll(new Map([[path, short]])),
    contentLength: (target) => target === path ? full.byteLength : undefined,
  }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assert(error.cause instanceof Error, "取得層の検証失敗を cause に残す");
  assertEquals(hasEntry(hubCache(caches), short), false, "検証を通らないバイト列を格納している");
});

Deno.test("fetchAssets: 資産の 429 は onRetry で届き、取り直したバイト列が返る", async () => {
  const caches = new MemoryCacheStorage();
  const path = "vae_decoder/model.safetensors";
  const target = resolveUrl(path);
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const retries: RetryDiagnostic[] = [];
  const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest), {
    fetch: rateLimitOnce(mock.fetch, (url) => url === target),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });
  assertEquals(retries.length, 1, "再試行の通知が 1 回だけ届いていない");
  assertEquals(retries[0].url, target, "通知が別の資産の URL を名乗っている");
  assertEquals(retries[0].status, 429);
  assertEquals(assets["vae_decoder"], payloadFor(path), "取り直したバイト列が返っていない");
  assertEquals(countCalls(mock.calls, target), 2, "429 の後に取り直していない");
});

Deno.test("fetchAssets: 完全キャッシュ済みでも中断済み signal なら資産を返さない", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const cached = hubCache(caches).entries.size;

  const controller = new AbortController();
  const reason = new Error("app: 起動を取り消した");
  controller.abort(reason);
  const second = createMockFetch({ files: serveAll() });
  const error = await assertRejects(() =>
    fetchAssets(loaded, files, { fetch: second.fetch, caches, signal: controller.signal })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(second.calls, [], "中断済みなのに network へ出ている");
  assertEquals(
    hubCache(caches).entries.size,
    cached,
    "中断を破損と取り違えてキャッシュを捨てている",
  );
});

Deno.test("fetchAssets: キャッシュ読出し中の中断でも資産を返さずに素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const uniquePaths = new Set(Object.values(files).map((ref) => ref.path)).size;

  const controller = new AbortController();
  const reason = new Error("app: 読出し中に取り消した");
  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  // 全キャッシュ済みなので downloading は出ない。最初に揃った 1 本の complete で取り消すと、
  // 残りは「取得を抜けた直後の確認」で止まる（＝取り消し後に complete まで進むものが無い）。
  const error = await assertRejects(() =>
    fetchAssets(loaded, files, {
      fetch: second.fetch,
      caches,
      signal: controller.signal,
      onProgress: (progress) => {
        events.push(progress);
        controller.abort(reason);
      },
    })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assert(uniquePaths > 1, "取り消しの効き目を観測できる本数になっていない");
  assert(
    events.filter((event) => event.phase === "complete").length < uniquePaths,
    "取り消したのに全ファイルを complete まで進めている",
  );
});

Deno.test("fetchAssets: 認証の有無でキャッシュを分けない（ヘッダは取得へ透過する）", async () => {
  const caches = new MemoryCacheStorage();
  const authed = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: authed.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, {
    fetch: authed.fetch,
    caches,
    headers: { authorization: "Bearer hf_token" },
  });
  assert(authed.calls.length > 0, "そもそも取得していない");
  assertEquals(
    new Set(authed.authorizations),
    new Set(["Bearer hf_token"]),
    "Authorization が取得へ届いていない",
  );
  assertEquals(caches.namespaces.size, 1, "認証の有無で名前空間を分けている");

  // 無認証の再要求は同じエントリに当たる（credential ごとの隔離はしない — by-design）。
  const anonymous = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, { fetch: anonymous.fetch, caches });
  assertEquals(anonymous.calls, [], "認証の有無でキャッシュが分かれている");
});

Deno.test("fetchAssets: sha256 の食い違いは fail loudly（真実源が壊れていれば残さない）", async () => {
  const caches = new MemoryCacheStorage();
  const corrupt = new TextEncoder().encode("karume-test:tampered-payload-XXXXXXXXXXXX");
  const path = "vae_decoder/model.safetensors";
  assertEquals(corrupt.byteLength, payloadFor(path).byteLength, "長さは合わせ sha256 だけ外す");
  const { mock, loaded } = await load({ files: serveAll(new Map([[path, corrupt]])) }, caches);
  // 照合は取得層（受信中のハッシュ）が行うので、hub からは取得の失敗として上がる。
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.path, path);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assert(error.cause instanceof Error, "取得層の不一致を cause に残す");
  assertEquals(hasEntry(hubCache(caches), corrupt), false, "不一致のバイト列を格納している");
});

Deno.test("fetchAssets: 受信バイトが size を超えれば取得層の上限で fail loudly（キャッシュに残さない）", async () => {
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
    HubFetchError,
  );
  assertEquals(error.path, path);
  assert(error.cause instanceof Error, "取得層のエラーを cause に残す");
  // 文言は取得層 ADR 0011 が定めている（`受信が申告 N バイトを超えた`）。sha256 不一致ではなく
  // **受信の上限**で落ちたことを、この 1 語で見分ける。
  assert(
    error.cause.message.includes("超えた"),
    `${error.cause.message} が受信上限での打ち切りを示していない`,
  );
  assertEquals(hasEntry(hubCache(caches), bloated), false, "上限を超えたバイト列を格納している");
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
  const caches = new MemoryCacheStorage({ failPut: true });

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

// ---- 越境参照（`FileRef` の repo / revision — ADR 0038 §7）。
//
// 押さえるのは「セッションの解決済み SHA ではなく、宣言された (repo, revision) から取る」ことと、
// **同じ path 文字列でもリポが違えば別のバイト列**（path で畳むと取り違えが起きる）の 2 点。

const FOREIGN_REPO = "someone/text-stack";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";
const CROSS_PATH = "text_encoder/model.safetensors";

const digestOf = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const localBytes = new TextEncoder().encode("karume-test:local-text-encoder");
const foreignBytes = new TextEncoder().encode("karume-test:foreign-text-encoder-payload");

/** 自リポと越境先が**同じ path**を主張する manifest（取り違えの検出器）。 */
const crossRepoManifest = JSON.stringify({
  format: "karume/4",
  generator: "karume/0.1.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        own: {
          i8: {
            shards: [{
              path: CROSS_PATH,
              size: localBytes.byteLength,
              sha256: await digestOf(localBytes),
            }],
          },
        },
        borrowed: {
          i8: {
            shards: [{
              path: CROSS_PATH,
              size: foreignBytes.byteLength,
              sha256: await digestOf(foreignBytes),
              repo: FOREIGN_REPO,
              revision: FOREIGN_SHA,
            }],
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
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, foreignBytes],
    ...overrides,
  ]);

const foreignUrl = `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${CROSS_PATH}`;

Deno.test("fetchAssets: 越境参照は宣言された (repo, revision) から取る", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const files = resolveFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(assets["own"], localBytes, "自リポぶんが越境先のバイト列に化けている");
  assertEquals(assets["borrowed"], foreignBytes, "越境ぶんが自リポのバイト列に化けている");
  assertEquals(countCalls(mock.calls, foreignUrl), 1, "越境先の URL を叩いていない");
  assertEquals(countCalls(mock.calls, resolveUrl(CROSS_PATH)), 1, "自リポの URL を叩いていない");
  // 内容キーなので取得元 URL では引けない。別リポの同名 path が別エントリで共存すること
  // （＝キーが repo を含むこと）を、両方のバイト列が同時に残っている事実で見る。
  assert(hasEntry(hubCache(caches), localBytes), "自リポぶんのエントリが無い");
  assert(hasEntry(hubCache(caches), foreignBytes), "越境ぶんのエントリが無い");
});

Deno.test("fetchAssets: 越境参照の検証失敗は越境先の repo / SHA を名乗る", async () => {
  const caches = new MemoryCacheStorage();
  const corrupt = tamper(foreignBytes);
  const { mock, loaded } = await load({
    files: crossRepoFiles(new Map([[`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, corrupt]])),
  }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    HubFetchError,
  );
  // セッションの repo を名乗ると「そのリポには無い path」を指す診断になる。
  assertEquals(error.repo, FOREIGN_REPO);
  assertEquals(error.revisionSha, FOREIGN_SHA);
  assertEquals(error.path, CROSS_PATH);
});

Deno.test("fetchAssets: 越境先の 429 も onRetry で届く（越境先の URL を名乗る）", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const retries: RetryDiagnostic[] = [];
  const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest), {
    fetch: rateLimitOnce(mock.fetch, (url) => url === foreignUrl),
    caches,
    onRetry: (diagnostic) => retries.push(diagnostic),
  });
  assertEquals(retries.map((diagnostic) => diagnostic.url), [foreignUrl]);
  assertEquals(assets["borrowed"], foreignBytes, "取り直した越境ぶんが返っていない");
  assertEquals(countCalls(mock.calls, foreignUrl), 2, "429 の後に取り直していない");
});

Deno.test("fetchAssets: 同じ path の自リポ / 越境は進捗でも別の 1 本として数える", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const files = resolveFiles(loaded.manifest);
  const events: AssetProgress[] = [];
  await fetchAssets(loaded, files, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  });
  const expectedTotal = localBytes.byteLength + foreignBytes.byteLength;
  for (const event of events) assertEquals(event.total, expectedTotal, "総量が畳まれている");
  assertEquals(events[events.length - 1].loaded, expectedTotal, "最後は総量に到達する");
  assertEquals(
    events.filter((event) => event.phase === "complete").length,
    2,
    "同じ path の 2 本が 1 本に畳まれている",
  );
});

// ---- 分割されたコンポーネントへの越境参照（ADR 0038 §7 / ADR 0071 決定 2 —「shards の各要素は
// 従来の FileRef 検査をそのまま通す」）。exporter は 1GiB 超の共有コンポーネントを
// **shard 1 本 = 参照 1 つ**の形で焼くので、受け側は列の全要素を参照先の URL から取れなければ
// ならない（先頭だけ越境する / 列を畳むと、残りの shard がセッションの repo に無くて落ちる）。

const SPLIT_PATHS = [
  "text_encoder/model-00001-of-00002.safetensors",
  "text_encoder/model-00002-of-00002.safetensors",
];

const splitBytes = SPLIT_PATHS.map((path) =>
  new TextEncoder().encode(`karume-test:foreign-shard:${path}`)
);

const splitShardManifest = JSON.stringify({
  format: "karume/4",
  generator: "karume/0.1.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        borrowed: {
          f16: {
            shards: await Promise.all(SPLIT_PATHS.map(async (path, index) => ({
              path,
              size: splitBytes[index].byteLength,
              sha256: await digestOf(splitBytes[index]),
              repo: FOREIGN_REPO,
              revision: FOREIGN_SHA,
            }))),
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
    ...SPLIT_PATHS.map((path, index) =>
      [`${FOREIGN_REPO}@${FOREIGN_SHA}/${path}`, splitBytes[index]] as const
    ),
  ]);

Deno.test("fetchAssets: 分割コンポーネントは shard ごとに越境先の URL から取る", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: splitShardFiles() }, caches);
  const files = resolveFiles(loaded.manifest);

  // 取得キーは shard の位置つき（列の位置が shard の id — ADR 0071 決定 2）。
  assertEquals(Object.keys(files), ["borrowed[0]", "borrowed[1]"]);
  const assets = await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  assertEquals(assets["borrowed[0]"], splitBytes[0], "先頭 shard のバイト列が違う");
  assertEquals(assets["borrowed[1]"], splitBytes[1], "後続 shard のバイト列が違う");
  for (const path of SPLIT_PATHS) {
    assertEquals(
      countCalls(mock.calls, `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${path}`),
      1,
      `${path} を越境先から取っていない`,
    );
    assertEquals(
      countCalls(mock.calls, resolveUrl(path)),
      0,
      `${path} をセッションの repo へ取りに行っている（そこには無い）`,
    );
  }
});

Deno.test("clearHubCache: 温めた資産を消す（次の取得は network に出る）", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });
  const cached = hubCache(caches).entries.size;
  assert(cached > 1, "manifest 以外に資産が溜まっていない");

  assertEquals(await clearHubCache({ caches }), true);
  assertEquals([...caches.namespaces.keys()], [], "取得層の名前空間ごと消えていない");

  const after = createMockFetch({ files: serveAll() });
  await fetchAssets(loaded, files, { fetch: after.fetch, caches });
  assert(after.calls.length > 0, "消した後の取得が network に出ていない");
});

Deno.test("clearHubCache: 旧名前空間（karume/1 系）も残さず消す", async () => {
  const caches = await populated("karume/1", "karume/1:auth:0123456789abcdef", "other/1");
  assertEquals(await clearHubCache({ caches }), true);
  assertEquals([...caches.namespaces.keys()], ["other/1"], "他コードの名前空間まで消している");
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

// ---- 区間読み（`openAsset` — `source.ts` ⑧）。HF 取得元の口は**キャッシュの中身**しか開けない
// （`openHfFile` は network に出ない）ので、ここで固定するのは「温まっていれば取得ゼロで読める」
// 「温まっていなければ開く側が 1 度だけ温める」の 2 点。

/** 区間読みのテストが共有する下ごしらえ（manifest だけ読んだ状態 + 対象の 1 本）。 */
const openable = async (): Promise<{
  mock: MockFetch;
  caches: MemoryCacheStorage;
  loaded: LoadedManifest;
  ref: FileRef;
}> => {
  const mock = createMockFetch({ files: serveAll() });
  const caches = new MemoryCacheStorage();
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: mock.fetch,
    caches,
  });
  return { mock, caches, loaded, ref: resolveFiles(loaded.manifest)["tokenizer"] };
};

Deno.test("openAsset: 温め済みの参照は取得を起こさずに区間だけを返す", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  await prefetchAssets(loaded, [ref], access);
  const payload = payloadFor(ref.path);
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "温め済みなのに区間読み口が開かない");
  // Deno には `globalThis.Deno` があるので取得層の既定戦略は "stream"（`blob()` が全量を
  // ヒープへ載せるため）= 読み飛ばしが offset に比例する費用。
  assertEquals(reader.cost, "scan");

  const middle = Math.floor(ref.size / 2);
  for (const [offset, length] of [[0, 4], [middle, 3], [ref.size - 5, 5]]) {
    const bytes = await reader.read(offset, length);
    assertEquals(
      bytes,
      new Uint8Array(payload.subarray(offset, offset + length)),
      `[${offset}, ${offset + length}) の中身が化けている`,
    );
    // 消費側はそのまま TypedArray として読むので、buffer 全体を占めている必要がある。
    assertEquals(bytes.byteOffset, 0);
    assertEquals(bytes.buffer.byteLength, length);
  }
  assertEquals(mock.calls.length, calls, "温め済みなのに取得が起きている");
});

Deno.test("openAsset: 未取得の参照は開く側が 1 度だけ温めてから開く", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  const calls = mock.calls.length;

  // 呼び手に「先に prefetchAssets を通せ」という手順を負わせない（消費側は行を要求するだけ）。
  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "未取得の参照で区間読み口が開かない");
  assertEquals(mock.calls.length - calls, 1, "温め直しが 1 往復で済んでいない");

  assertEquals(await reader.read(0, ref.size), payloadFor(ref.path));
  // 開いた後の読みはキャッシュの中だけで完結する（読みごとに network へ出ない）。
  assertEquals(mock.calls.length - calls, 1, "読みが取得を起こしている");
});

Deno.test("openAsset: 記録ハッシュを持たないエントリは読まずに取り直す", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  await prefetchAssets(loaded, [ref], access);
  const payload = payloadFor(ref.path);
  // 記録を落としたうえで中身を壊す（照合するものが無いので開く側は開けない）。
  overwriteEntry(hubCache(caches), payload, new Uint8Array(ref.size), { keepRecord: false });
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "取り直したのに区間読み口が開かない");
  assertEquals(mock.calls.length - calls, 1, "壊れたエントリを取り直していない");
  assertEquals(await reader.read(0, ref.size), payload, "壊れたバイト列がそのまま読めている");
});

Deno.test("openAsset: 記録ハッシュが宣言と食い違うエントリも読まずに取り直す", async () => {
  const { mock, caches, loaded, ref } = await openable();
  const access = { fetch: mock.fetch, caches };
  const other = resolveFiles(loaded.manifest)["vae_decoder"];
  await prefetchAssets(loaded, [ref, other], access);
  const payload = payloadFor(ref.path);
  // 2 件の記録を入れ替える = どちらも「記録ハッシュ ≠ manifest の宣言」になる。開く側はこれを
  // 「内容が変わった」と見て evict するので、記録なしと同じ取り直しの経路へ倒れる。
  swapRecords(hubCache(caches), payload, payloadFor(other.path));
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "取り直したのに区間読み口が開かない");
  // 取り直すのは開いた 1 本だけ（記録を壊したもう 1 本は触らない）。
  assertEquals(mock.calls.length - calls, 1, "食い違うエントリを 1 往復で取り直していない");
  assertEquals(await reader.read(0, ref.size), payload, "取り直した中身が読めていない");
});

Deno.test("openAsset: 越境参照は宣言された (repo, revision) の口で開く", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: crossRepoFiles() }, caches);
  const access = { fetch: mock.fetch, caches };
  // 自リポと越境先が**同じ path** を主張する manifest なので、path で畳んでいれば自リポの
  // バイト列が読めてしまう（区間読みの口も `originFor` 経由で越境先から生えることの検出器）。
  const ref = resolveFiles(loaded.manifest)["borrowed"];
  const calls = mock.calls.length;

  const reader = await openAsset(loaded, ref, access);
  assert(reader !== undefined, "越境先でも能力は継承されるはずなのに口が開かない");
  assertEquals(countCalls(mock.calls, foreignUrl), 1, "越境先の URL を温めていない");
  assertEquals(mock.calls.length - calls, 1, "越境の温めが 1 往復で済んでいない");
  assertEquals(await reader.read(0, ref.size), foreignBytes, "自リポのバイト列が読めている");
});

Deno.test("openAsset: abort 済み signal は口の有無に依らず reason をそのまま上げる", async () => {
  const mock = createMockFetch({ files: serveAll() });
  const remote = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: mock.fetch,
    caches: new MemoryCacheStorage(),
  });
  const files = resolveFiles(remote.manifest);
  const ref = files[Object.keys(files)[0]];

  // 口を持たない取得元（位置読みを持たないアダプター）を同じ manifest で 1 本作る。
  const served = serveAll();
  const lookup = (path: string): Uint8Array<ArrayBuffer> => {
    const bytes = served.get(path);
    if (bytes === undefined) throw new Error(`test-directory: ${path} を読めない`);
    return bytes;
  };
  const adapter: DirectoryAdapter = {
    readFile: (path) => Promise.resolve(new Uint8Array(lookup(path))),
  };
  const local = await loadManifest(localDirectory(adapter, { label: "./models/test" }), {
    caches: new MemoryCacheStorage(),
  });

  const controller = new AbortController();
  const reason = new Error("test: 呼び出し側の中断");
  controller.abort(reason);

  // 中断は能力の差より先に見る — 口を持たない取得元でも `undefined` ではなく reason で落ちる。
  for (const loadedSource of [remote, local]) {
    const error = await assertRejects(
      () => openAsset(loadedSource, ref, { signal: controller.signal }),
      Error,
    );
    assertStrictEquals(error, reason);
  }
});
