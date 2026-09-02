import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  type AssetPhase,
  type AssetProgress,
  type FileRef,
  HubFetchError,
  type LoadedManifest,
  loadManifest,
  ManifestReferenceError,
  prefetchAssets,
  resolveFiles,
  streamAssets,
  type StreamedAsset,
} from "../mod.ts";
import {
  abortWhileAwaitingResponse,
  createMockFetch,
  hasEntry,
  HUB_URL,
  hubCache,
  MemoryCacheStorage,
  type MockRoutes,
  overwriteEntry,
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

/**
 * 全 shard を溜めて後から見る（順序・中身の検査用）。
 *
 * MUST: bytes は yield ごとに写す — 逐次面の契約は「次の `next()` までに使い終える」で、器を使う
 * 取得元（ディレクトリ / HF）では次の反復が同じ buffer を上書きする。view を溜めると最後の
 * shard の中身に化ける。
 */
const drain = async (
  stream: AsyncGenerator<StreamedAsset, void, unknown>,
): Promise<StreamedAsset[]> => {
  const seen: StreamedAsset[] = [];
  for await (const asset of stream) seen.push({ ...asset, bytes: asset.bytes.slice() });
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
  assertEquals(rest.map((asset) => asset.id), refs.slice(1).map((ref) => ref.path));
  assertEquals(mock.calls.length, afterPhase1, "相 2 は温めたキャッシュだけを読む");
});

Deno.test("streamAssets: yield は refs の入力順で、bytes は宣言どおりの中身になる", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  // manifest 順ではなく「渡した列の順」で届くこと。
  const order = [...refs].reverse();

  // 器の形（同じ buffer の prefix view）を見るので、写しを取る drain を通さず yield 中に検査する。
  const ids: string[] = [];
  const buffers = new Set<ArrayBufferLike>();
  for await (const asset of streamAssets(loaded, order, { fetch: mock.fetch, caches })) {
    ids.push(asset.id);
    buffers.add(asset.bytes.buffer);
    assertEquals(asset.bytes, payloadFor(asset.id));
    assertEquals(asset.bytes.byteOffset, 0);
    assertEquals(asset.bytes.byteLength, payloadFor(asset.id).byteLength);
  }

  assertEquals(ids, order.map((ref) => ref.path));
  // HF 取得元も逐次面の器を使う（取得層の `into`）— 全 shard が 1 本の buffer に届く。
  assertEquals(buffers.size, 1, "HF 取得元が器を使っていない（shard ごとに別 buffer）");
});

Deno.test("streamAssets: 記録の無い破損エントリは相 2 が捕まえ、1 往復で治る", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const ref = refs[2];
  // まず正規経路で温めてから、記録ごと落として中身を壊す（旧版 / 無検証 prefetch 由来の形）。
  // 記録が無いエントリは相 1 が「陳腐化」として温め直そうとするので、network に出るのは
  // その 1 往復だけになる。
  await drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches }));
  const before = mock.calls.length;
  overwriteEntry(hubCache(caches), payloadFor(ref.path), tamper(payloadFor(ref.path)), {
    keepRecord: false,
  });

  const second = createMockFetch({ files: serveAll() });
  const seen = await drain(streamAssets(loaded, refs, { fetch: second.fetch, caches }));

  const healed = seen.find((asset) => asset.id === ref.path);
  assertEquals(healed?.bytes, payloadFor(ref.path), "破損キャッシュが素通りしている");
  assertEquals(countCalls(second.calls, resolveUrl(ref.path)), 1, "self-heal は 1 往復だけ");
  assertEquals(second.calls.length, 1, "壊れていない shard まで取り直している");
  assertEquals(mock.calls.length, before, "1 回目の mock も追加で呼ばれていない");
});

Deno.test("streamAssets: キャッシュも真実源も壊れていれば fail loudly", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  await drain(streamAssets(loaded, refs, { fetch: mock.fetch, caches }));

  // 記録を落として中身を壊す = 相 1 が「陳腐化」と見て温め直しに行く形。その取り直し先
  // （真実源）も壊れているので、通過中の照合で落ちてエントリは成立しない。
  const path = "vae_decoder/model.safetensors";
  const corrupt = tamper(payloadFor(path));
  overwriteEntry(hubCache(caches), payloadFor(path), corrupt, { keepRecord: false });

  const broken = createMockFetch({ files: serveAll(new Map([[path, corrupt]])) });
  const error = await assertRejects(
    () => drain(streamAssets(loaded, refs, { fetch: broken.fetch, caches })),
    HubFetchError,
  );
  assertEquals(error.path, path);
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assertEquals(error.available.models, ["anima-turbo", "anima-lite"]);
  assert(error.cause instanceof Error, "取得層の不一致を cause に残す");
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
    hasEntry(hubCache(caches), tamper(payloadFor(path))),
    false,
    "不一致のバイト列がキャッシュに残っている",
  );
});

Deno.test("streamAssets: 相 1 は巻き添えではなく真の第一失敗を表面化する", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { loaded, refs, mock } = await prepare({ files: served, delayMs: 5 }, caches);
  // **先頭以外**の 1 本だけを 404 にする。1 本の失敗は残り全部を abort するので、巻き添え側は
  // 生の AbortError として決着する — ワーカーの配列位置で拾うと、真犯人が worker[0] 以外の
  // ときに巻き添えが表面化して真因（どのファイルがなぜ落ちたか）が消える。
  const victim = refs[1].path;
  assert(refs[0].path !== victim, "先頭以外が落ちる形になっていない");
  served.delete(victim);

  const error = await assertRejects(
    () =>
      drain(
        streamAssets(loaded, refs, {
          fetch: abortWhileAwaitingResponse(mock.fetch, victim),
          caches,
        }),
      ),
    HubFetchError,
  );
  assertEquals(error.path, victim, "落ちたのとは別の shard が報告されている");
  assertEquals(error.repo, REPO);
  assertEquals(error.revisionSha, SHA);
  assert(error.message.includes(victim), `${error.message} が落ちた path を名乗っていない`);
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
      // downloading が出るのは相 1 だけ（相 2 はキャッシュヒットなので complete しか出ない）。
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
        // complete が出るのは相 2 だけ（相 1 は downloading しか出さない）。1 本目を配った
        // 直後に取り消すと、次の shard は冒頭の確認で止まる。
        onProgress: (progress) => {
          if (progress.phase === "complete") controller.abort(reason);
        },
      })
    ) seen.push(asset.id);
  });
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assert(seen.length < refs.length, "中断したのに全 shard を配り切っている");
});

Deno.test("streamAssets: 最終 shard の読出し中の中断でも配り切らずに素通しする", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);
  const controller = new AbortController();
  const reason = new Error("app: 読出し中に取り消した");
  // 1 shard だけの列 = その shard が最終 shard。冒頭の中断確認は通過済みなので、読出し中の
  // 中断を捉えられるのは yield 直前の確認だけになる。
  const only = [refs[0]];
  const seen: string[] = [];

  // キャッシュ読出しの瞬間に取り消す。1 回目の match は相 1 の既存エントリ検査、2 回目が
  // 相 2 の読出し — その内側で落とすことで「冒頭の確認は通過済み」の窓を狙う。
  let matches = 0;
  hubCache(caches).onMatch = () => {
    matches += 1;
    if (matches === 2) controller.abort(reason);
  };

  const error = await assertRejects(async () => {
    for await (
      const asset of streamAssets(loaded, only, {
        fetch: mock.fetch,
        caches,
        signal: controller.signal,
      })
    ) seen.push(asset.id);
  });
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(matches, 2, "相 2 の読出しに到達していない（窓を狙えていない）");
  assertEquals(seen, [], "取り消したのに検証済みバイトを配っている");
});

/** phase の進む向き（大きいほど後）。 */
const PHASE_RANK: Record<AssetPhase, number> = { downloading: 0, complete: 1 };

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

Deno.test("streamAssets: 2 回目は相 1 も相 2 も network に出ない（記録ハッシュのヒット）", async () => {
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
  assertEquals(seen.map((asset) => asset.id), refs.map((ref) => ref.path));
  const byPath = phasesByPath(events);
  assertEquals(byPath.size, refs.length);
  for (const [path, phases] of byPath) {
    assertEquals(
      phases,
      ["complete"],
      `${path}: 相 1 が温め済みを取り直している / 観測できない照合を名乗っている`,
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
      `${event.path}: ${event.phase} の全体 loaded がこの shard ぶんを数えていない`,
    );
  }
});

// ---- 越境参照（`FileRef` の repo / revision — ADR 0038 §7）。逐次面も宣言された
// (repo, revision) から取り、同じ path 文字列の自リポ / 越境を別の 1 本として扱う。

const FOREIGN_REPO = "someone/text-stack";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";
const CROSS_PATH = "text_encoder/model.safetensors";

const digestOf = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const localShard = new TextEncoder().encode("karume-test:local-shard");
const foreignShard = new TextEncoder().encode("karume-test:foreign-shard-payload");

const crossRepoRefs: FileRef[] = [
  { path: CROSS_PATH, size: localShard.byteLength, sha256: await digestOf(localShard) },
  {
    path: CROSS_PATH,
    size: foreignShard.byteLength,
    sha256: await digestOf(foreignShard),
    repo: FOREIGN_REPO,
    revision: FOREIGN_SHA,
  },
];

const crossRepoServe = (): Map<string, Uint8Array<ArrayBuffer>> =>
  new Map([
    [MANIFEST_PATH, manifestBytes],
    [CROSS_PATH, localShard],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, foreignShard],
  ]);

Deno.test("streamAssets: 越境参照は宣言された (repo, revision) から取り、path が同じでも混ざらない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await prepare({ files: crossRepoServe() }, caches);

  const seen = await drain(streamAssets(loaded, crossRepoRefs, { fetch: mock.fetch, caches }));

  assertEquals(seen.map((asset) => asset.id), [CROSS_PATH, CROSS_PATH]);
  assertEquals(seen[0].bytes, localShard, "自リポぶんが越境先のバイト列に化けている");
  assertEquals(seen[1].bytes, foreignShard, "越境ぶんが自リポのバイト列に化けている");
  assertEquals(
    countCalls(mock.calls, `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${CROSS_PATH}`),
    1,
    "越境先の URL を叩いていない",
  );
  assertEquals(countCalls(mock.calls, resolveUrl(CROSS_PATH)), 1, "自リポの URL を叩いていない");
});

Deno.test("streamAssets: 同一の越境参照を 2 回渡すのは呼び出し側の誤りとして拒否する", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await prepare({ files: crossRepoServe() }, caches);
  const duplicated = [crossRepoRefs[1], crossRepoRefs[1]];
  await assertRejects(
    () => drain(streamAssets(loaded, duplicated, { fetch: mock.fetch, caches })),
    ManifestReferenceError,
  );
  assertEquals(mock.calls, [], "重複検査より先に network へ出ている");
});

// ---- 相 1 単体の面（prefetchAssets）。逐次面と機構を共有するので、ここで見るのは共有部の
// 挙動ではなく**この面だけの契約** — 「complete の発行者はここ」「落とした後の逐次面は
// network に出ない」の 2 点。

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

Deno.test("prefetchAssets: 落とした後の streamAssets は 1 度も network に出ない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });
  const warmed = mock.calls.length;
  assert(warmed > 0, "prefetch が 1 本も取得していない");

  const second = createMockFetch({ files: serveAll() });
  const seen = await drain(streamAssets(loaded, refs, { fetch: second.fetch, caches }));

  assertEquals(seen.map((asset) => asset.id), refs.map((ref) => ref.path));
  for (const asset of seen) assertEquals(asset.bytes, payloadFor(asset.id));
  assertEquals(second.calls, [], "相 1 / 相 2 のどちらかが network に出ている");
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

Deno.test("prefetchAssets: 1 本の失敗は真因を復元して HubFetchError で上がる", async () => {
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const { loaded, refs, mock } = await prepare({ files: served, delayMs: 5 }, caches);
  // 逐次面と同じく、先頭以外を落として巻き添え（生の AbortError）が表面化しないことを見る。
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
});

Deno.test("prefetchAssets: 空・重複は network に出る前に ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, refs, mock } = await prepare({ files: serveAll() }, caches);

  const empty = await assertRejects(
    () => prefetchAssets(loaded, [], { fetch: mock.fetch, caches }),
    ManifestReferenceError,
  );
  assert(empty.message.includes("prefetchAssets"), `${empty.message} が面の名前を名乗っていない`);

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
