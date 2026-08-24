import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  type FileRef,
  HubFetchError,
  IntegrityError,
  type LoadedManifest,
  loadManifest,
  ManifestReferenceError,
  resolveFiles,
  streamAssets,
  type StreamedAsset,
} from "../mod.ts";
import {
  createMockFetch,
  HUB_URL,
  type MemoryCache,
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

const countCalls = (calls: readonly string[], url: string): number =>
  calls.filter((call) => call === url).length;

/** 長さは保ったまま中身だけ変える（size ではなく sha256 の門を踏ませる）。 */
const tamper = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const copy = new Uint8Array(bytes);
  copy[copy.length - 1] ^= 0xff;
  return copy;
};

/**
 * shard 列（= 逐次面の入力）。全量面と違い path の一意化は呼び出し側の責務なので、
 * ここで manifest 由来の重複（`rope_alias`）を畳んでから渡す。
 */
const shardRefs = (loaded: LoadedManifest): FileRef[] => {
  const files = resolveFiles(loaded.manifest);
  const byPath = new Map<string, FileRef>();
  for (const key of Object.keys(files)) {
    const ref = files[key];
    if (!byPath.has(ref.path)) byPath.set(ref.path, ref);
  }
  return [...byPath.values()];
};

/** manifest は別 mock で読み、逐次面の観測用に呼び出し記録が空の mock を渡す。 */
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
  return { loaded, refs: shardRefs(loaded), mock: createMockFetch(routes) };
};

const hubCache = (caches: MemoryCacheStorage): MemoryCache => {
  const namespace = caches.namespaces.get("karume/1");
  if (namespace === undefined) throw new Error("test: karume/1 の名前空間がまだ無い");
  return namespace;
};

const drain = async (
  stream: AsyncGenerator<StreamedAsset, void, unknown>,
): Promise<StreamedAsset[]> => {
  const seen: StreamedAsset[] = [];
  for await (const asset of stream) seen.push(asset);
  return seen;
};

Deno.test("streamAssets: 呼んだだけでは何も起きず、相 1 は最初の next() で始まる", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  const stream = streamAssets(loaded, refs, { fetch: mock.fetch, caches });
  assertEquals(mock.calls, [], "generator を作っただけで DL が始まっている");

  const first = await stream.next();
  assertEquals(first.done, false);
  assertEquals(
    new Set(mock.calls),
    new Set(refs.map((ref) => resolveUrl(ref.path))),
    "最初の yield の前に全 shard を落とし終えていない（相 1 が 2 相になっていない）",
  );
  const afterPhase1 = mock.calls.length;

  const rest = await drain(stream);
  assertEquals(rest.map((asset) => asset.path), refs.slice(1).map((ref) => ref.path));
  assertEquals(mock.calls.length, afterPhase1, "相 2 は温めたキャッシュだけを読む");
});

Deno.test("streamAssets: yield は refs の入力順で、bytes は宣言どおりの中身になる", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  // manifest 順ではなく「渡した列の順」で届くこと。
  const order = [...refs].reverse();

  const seen = await drain(streamAssets(loaded, order, { fetch: mock.fetch, caches }));

  assertEquals(seen.map((asset) => asset.path), order.map((ref) => ref.path));
  for (const asset of seen) {
    assertEquals(asset.bytes, payloadFor(asset.path));
    assertEquals(asset.bytes.byteOffset, 0);
    assertEquals(asset.bytes.byteLength, asset.bytes.buffer.byteLength);
  }
});

Deno.test("streamAssets: 破損したキャッシュエントリは相 2 の照合が捕まえ、1 往復で治る", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const ref = refs[2];
  // 相 1 は既存エントリを見て network に出ない（＝相 2 の照合だけが最後の門になる）。
  hubCache(caches).entries.set(resolveUrl(ref.path), tamper(payloadFor(ref.path)));

  const seen = await drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches }));

  const healed = seen.find((asset) => asset.path === ref.path);
  assertEquals(healed?.bytes, payloadFor(ref.path), "破損キャッシュが素通りしている");
  assertEquals(
    countCalls(mock.calls, resolveUrl(ref.path)),
    1,
    "self-heal は 1 往復だけ（相 1 は温め済みとして飛ばす）",
  );
});

Deno.test("streamAssets: キャッシュも真実源も壊れていれば IntegrityError（source は network）", async () => {
  const caches = new MemoryCacheStorage();
  const path = "vae_decoder/model.safetensors";
  const corrupt = tamper(payloadFor(path));
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[path, corrupt]])) },
    caches,
  );
  hubCache(caches).entries.set(resolveUrl(path), corrupt);

  const error = await assertRejects(
    () => drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches })),
    IntegrityError,
  );
  assertEquals(error.path, path);
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.source, "network", "self-heal の取り直し先は network");
  assertEquals(error.expected.length, 64);
  assert(error.expected !== error.actual);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
});

Deno.test("streamAssets: 相 1 の sha256 不一致は fail loud で、キャッシュにエントリを残さない", async () => {
  const caches = new MemoryCacheStorage();
  const path = "tokenizer/qwen2-tokenizer.json";
  const { loaded, refs, mock } = await prepare(
    { files: serveAll(new Map([[path, tamper(payloadFor(path))]])) },
    caches,
  );
  // 並行ワーカーの巻き添えを排して「相 1 で落ちた 1 本」だけを見る。
  const only = refs.filter((ref) => ref.path === path);
  assertEquals(only.length, 1);

  const error = await assertRejects(
    () => drain(streamAssets(loaded, only, { fetch: mock.fetch, caches })),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assert(error.cause instanceof Error, "下層の不一致を cause に残す");
  assertEquals(
    hubCache(caches).entries.has(resolveUrl(path)),
    false,
    "不一致のバイト列がキャッシュに残っている",
  );
});

Deno.test("streamAssets: 取得対象が空なら network に出る前に ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await prepare({ files: serveAll() }, caches);

  const error = await assertRejects(
    () => drain(streamAssets(loaded, [], { fetch: mock.fetch, caches })),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assertEquals(mock.calls, [], "入力検査より先に network へ出ている");
});

Deno.test("streamAssets: 同一 path の重複は network に出る前に ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  const error = await assertRejects(
    () => drain(streamAssets(loaded, [refs[0], refs[1], refs[0]], { fetch: mock.fetch, caches })),
    ManifestReferenceError,
  );
  assert(error.message.includes(refs[0].path), `${error.message} が重複 path を名指ししていない`);
  assertEquals(mock.calls, [], "入力検査より先に network へ出ている");
});

Deno.test("streamAssets: キャッシュを開けない環境では素 fetch へ縮退せず fail loud", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  // 相 1 はバイト列を手元に持たない面なので、キャッシュが使えないなら「全量面で動いている
  // ように見えるが RAM 目標が壊れている」への沈黙縮退はできない（ADR 0070 追記）。
  const broken = {
    open: () => Promise.reject(new Error("cache storage open failure")),
  } as unknown as CacheStorage;

  const error = await assertRejects(
    () => drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches: broken })),
    HubFetchError,
  );
  assert(error.cause instanceof Error, "下層の失敗を cause に残す");
  assertEquals(mock.calls, [], "キャッシュを開けないのに本体の DL が走っている");
});

Deno.test("streamAssets: 相 1 中の中断は HubFetchError に包まれず素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll(), delayMs: 5 }, caches);
  const controller = new AbortController();
  const reason = new Error("app: ユーザーがロードを取り消した");

  const error = await assertRejects(() =>
    drain(streamAssets(loaded, refs, {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
      // downloading が出るのは相 1 だけ（prefetch に verifying は無い）。
      onProgress: (progress) => {
        if (progress.phase === "downloading") controller.abort(reason);
      },
    }))
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
});

Deno.test("streamAssets: 相 2 中の中断は shard の切れ目で素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const controller = new AbortController();
  const reason = new Error("app: 途中でやめた");
  const seen: string[] = [];

  const error = await assertRejects(async () => {
    for await (
      const asset of streamAssets(loaded, refs, {
        fetch: mock.fetch,
        caches,
        signal: controller.signal,
        // verifying が出るのは相 2 だけ（相 1 は照合を通過中に済ませる）。
        onProgress: (progress) => {
          if (progress.phase === "verifying") controller.abort(reason);
        },
      })
    ) seen.push(asset.path);
  });
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assert(seen.length < refs.length, "中断したのに全 shard を配り切っている");
});

Deno.test("streamAssets: 最終 shard の検証中の中断でも配り切らずに素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const controller = new AbortController();
  const reason = new Error("app: 検証中に取り消した");
  // 1 shard だけの列 = その shard が最終 shard。冒頭の中断確認は通過済みなので、
  // verifying 中の中断は yield 直前の確認だけが観測できる。
  const only = [refs[0]];
  const seen: string[] = [];

  const error = await assertRejects(async () => {
    for await (
      const asset of streamAssets(loaded, only, {
        fetch: mock.fetch,
        caches,
        signal: controller.signal,
        onProgress: (progress) => {
          if (progress.phase === "verifying") controller.abort(reason);
        },
      })
    ) seen.push(asset.path);
  });
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(seen, [], "取り消したのに検証済みバイトを配っている");
});

/** phase の進む向き（大きいほど後）。 */
const PHASE_RANK: Record<AssetPhase, number> = { downloading: 0, verifying: 1, complete: 2 };

/** path ごとの phase 列（相 1 の進捗は交錯して届くので path で束ね直す）。 */
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

Deno.test("streamAssets: 進捗は phase 単調・total は size 合計・complete はファイルごと 1 回", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const events: AssetProgress[] = [];

  await drain(streamAssets(loaded, refs, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  }));

  let expectedTotal = 0;
  for (const ref of refs) expectedTotal += ref.size;
  const sizeByPath = new Map(refs.map((ref) => [ref.path, ref.size] as const));
  assert(events.length > 0, "進捗が 1 度も出ない");
  for (const event of events) {
    assertEquals(event.total, expectedTotal, "総量は refs の size 合計");
    assert(event.loaded <= event.total, "合計が総量を超えない");
    assertEquals(
      event.fileTotal,
      sizeByPath.get(event.path),
      `${event.path}: fileTotal はその shard 自身の manifest size`,
    );
    assert(
      event.fileLoaded <= event.fileTotal,
      `${event.path}: fileLoaded がその shard の size を超えた`,
    );
    if (event.phase !== "downloading") {
      assertEquals(
        event.fileLoaded,
        event.fileTotal,
        `${event.path}: ${event.phase} は全量が揃った点`,
      );
    }
  }

  const byPath = phasesByPath(events);
  assertEquals(
    new Set(byPath.keys()),
    new Set(refs.map((ref) => ref.path)),
    "渡した全 shard が進捗に現れる",
  );
  for (const [path, phases] of byPath) {
    assert(phases.includes("downloading"), `${path}: 相 1 の downloading が無い`);
    assertMonotonic(phases, path);
  }
  assertEquals(events[events.length - 1].loaded, expectedTotal, "最後は総量に到達する");
});

Deno.test("streamAssets: ファイル別の進捗は全体合計とは別に 1 shard ぶんを表す", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const events: AssetProgress[] = [];

  await drain(streamAssets(loaded, refs, {
    fetch: mock.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  }));

  // 受信途中（size 未満）の downloading が出ている = fileLoaded がその shard の受信実測である。
  assert(
    events.some((event) => event.phase === "downloading" && event.fileLoaded < event.fileTotal),
    "受信途中の fileLoaded が 1 度も観測できない",
  );
  // 相 1 で先に終わった shard のぶんが loaded に積まれた後の downloading は、その shard 1 本ぶんの
  // fileLoaded より必ず大きい（同値のままなら per-file 欄が全体合計の写しになっている）。
  assert(
    events.some((event) => event.phase === "downloading" && event.loaded > event.fileLoaded),
    "複数 shard を落としているのに全体 loaded とファイル別 fileLoaded が食い違わない",
  );
});

Deno.test("streamAssets: 2 回目は network に出ないが sha256 照合は走る", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches }));
  const before = mock.calls.length;

  const second = createMockFetch({ files: serveAll() });
  const events: AssetProgress[] = [];
  const seen = await drain(streamAssets(loaded, refs, {
    fetch: second.fetch,
    caches,
    onProgress: (progress) => events.push(progress),
  }));

  assertEquals(second.calls, [], "全キャッシュ済みなのに network へ出ている");
  assertEquals(mock.calls.length, before, "1 回目の mock も追加で呼ばれていない");
  assertEquals(seen.map((asset) => asset.path), refs.map((ref) => ref.path));
  const byPath = phasesByPath(events);
  assertEquals(byPath.size, refs.length);
  for (const [path, phases] of byPath) {
    assertEquals(
      phases,
      ["verifying", "complete"],
      `${path}: 相 1 が温め済みを取り直している / 照合が省かれている`,
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
      `${event.path}: ${event.phase} の全体 loaded がこの shard ぶんを数えていない`,
    );
  }
});
