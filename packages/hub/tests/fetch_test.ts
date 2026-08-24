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

/** 長さは保ったまま中身だけ変える（size ではなく sha256 の門を踏ませる）。 */
const tamper = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const copy = new Uint8Array(bytes);
  copy[copy.length - 1] ^= 0xff;
  return copy;
};

/** 無認証名前空間の中身（キャッシュを直に壊して self-heal を観測するため）。 */
const hubCache = (caches: MemoryCacheStorage): MemoryCache => {
  const namespace = caches.namespaces.get("karume/1");
  if (namespace === undefined) throw new Error("test: karume/1 の名前空間がまだ無い");
  return namespace;
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
    assert(warning.includes("_CURRENT"), `${warning} が models の pin 定数へ誘導していない`);
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

Deno.test("loadManifest: 1MiB を超える karume.json は受信前に弾く", async () => {
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

Deno.test("loadManifest: 破損した cached karume.json は self-heal で 1 往復だけ取り直す", async () => {
  const caches = new MemoryCacheStorage();
  const ref = { repo: REPO, hubUrl: HUB_URL, revision: SHA };
  const first = createMockFetch({ files: serveAll() });
  await loadManifest(ref, { fetch: first.fetch, caches });

  // ① UTF-8 として読めないバイト列（0xff は UTF-8 に現れない）。
  hubCache(caches).entries.set(resolveUrl(MANIFEST_PATH), new Uint8Array([0xff, 0xfe, 0xff]));
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
  hubCache(caches).entries.set(
    resolveUrl(MANIFEST_PATH),
    new TextEncoder().encode('{"format": "karume/4"'),
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
    hubCache(caches).entries.has(resolveUrl(MANIFEST_PATH)),
    false,
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
    hubCache(caches).entries.has(resolveUrl(MANIFEST_PATH)),
    true,
    "中断を破損と取り違えてキャッシュを捨てている",
  );
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
  assert(
    events.some((event) => event.phase === "verifying"),
    "sha256 照合中は verifying フェーズを出す",
  );
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
const PHASE_RANK: Record<AssetPhase, number> = { downloading: 0, verifying: 1, complete: 2 };

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

Deno.test("fetchAssets: network 取得の phase は downloading → verifying → complete と単調に進む", async () => {
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

Deno.test("fetchAssets: キャッシュヒットの phase 列は verifying → complete の 2 点だけ", async () => {
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
      ["verifying", "complete"],
      `${path}: DL していないのに downloading が出た`,
    );
  }
  for (const event of events) {
    // downloading が 1 度も出ない経路でも、この 2 相は全量が揃った点なので満たされる。
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

Deno.test("fetchAssets: 破損したキャッシュエントリは照合が捕まえ、1 往復で治る", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  const path = "vae_decoder/model.safetensors";
  hubCache(caches).entries.set(resolveUrl(path), tamper(payloadFor(path)));

  const second = createMockFetch({ files: serveAll() });
  const assets = await fetchAssets(loaded, files, { fetch: second.fetch, caches });

  assertEquals(assets["vae_decoder"], payloadFor(path), "破損キャッシュが素通りしている");
  assertEquals(countCalls(second.calls, resolveUrl(path)), 1, "self-heal は 1 往復だけ");
  assertEquals(second.calls.length, 1, "壊れていないファイルまで取り直している");
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

Deno.test("fetchAssets: キャッシュ検証中の中断でも資産を返さずに素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { mock, loaded } = await load({ files: serveAll() }, caches);
  const files = resolveFiles(loaded.manifest);
  await fetchAssets(loaded, files, { fetch: mock.fetch, caches });

  const controller = new AbortController();
  const reason = new Error("app: 検証中に取り消した");
  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  const error = await assertRejects(() =>
    fetchAssets(loaded, files, {
      fetch: second.fetch,
      caches,
      signal: controller.signal,
      // 全キャッシュ済みなので downloading は出ない（verifying が唯一の観測点）。
      onProgress: (progress) => {
        events.push(progress);
        if (progress.phase === "verifying") controller.abort(reason);
      },
    })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assert(
    events.every((event) => event.phase !== "complete"),
    "取り消したのに検証済みファイルを complete まで進めている",
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
  // キャッシュキーは URL のまま（SHA が URL に載るので不変性はそれで成立する）。
  assertEquals(hubCache(caches).entries.has(foreignUrl), true);
  assertEquals(hubCache(caches).entries.has(resolveUrl(CROSS_PATH)), true);
});

Deno.test("fetchAssets: 越境参照の検証失敗は越境先の repo / SHA を名乗る", async () => {
  const caches = new MemoryCacheStorage();
  const corrupt = tamper(foreignBytes);
  const { mock, loaded } = await load({
    files: crossRepoFiles(new Map([[`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, corrupt]])),
  }, caches);
  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest), { fetch: mock.fetch, caches }),
    IntegrityError,
  );
  // セッションの repo を名乗ると「そのリポには無い path」を指す診断になる。
  assertEquals(error.repo, FOREIGN_REPO);
  assertEquals(error.revisionSha, FOREIGN_SHA);
  assertEquals(error.path, CROSS_PATH);
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
