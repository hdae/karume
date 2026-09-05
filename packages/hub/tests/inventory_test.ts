/**
 * キャッシュ在庫の照会（`listCachedAssets`）と選択単位の削除（`evictCachedAssets`）。
 *
 * 押さえるのは 3 点 — ①在庫の同一性が**内容キー**（別リポの同名 path は別の 1 本）②削除が
 * 他の選択を壊さない（全在庫の選択が使うファイルは残す・部分在庫の選択は守らない）③キャッシュ
 * を持たない取得元は fail loud で断る。
 *
 * MUST: `caches` は必ず渡す（helpers/mock.ts の MUST — 渡さないと Deno の実キャッシュに書く）。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  evictCachedAssets,
  type FileRef,
  HubError,
  listCachedAssets,
  type LoadedManifest,
  loadManifest,
  localDirectory,
  ManifestReferenceError,
  prefetchAssets,
  resolveFiles,
  type ResolveOptions,
} from "../mod.ts";
import { fileRefKey } from "../src/manifest.ts";
import { DistributionSource, driverOf } from "../src/source.ts";
import { createHfSource } from "../src/sources/hf.ts";
import { buildLocalDist, memoryDirectory, sha256Hex } from "./helpers/local.ts";
import {
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

const TEXT_ENCODER = "text_encoder/model.safetensors";
const TEXT_CONDITIONER = "text_conditioner/model.safetensors";
const TRANSFORMER_F16 = "transformer/model.f16.safetensors";
const TRANSFORMER_I8 = "transformer/model.i8.safetensors";
const ROPE_BASE = "transformer/rope_base.safetensors";
const VAE_DECODER = "vae_decoder/model.safetensors";
const TOKENIZER = "tokenizer/qwen2-tokenizer.json";

const ASSET_PATHS = [
  TEXT_ENCODER,
  TEXT_CONDITIONER,
  TRANSFORMER_F16,
  TRANSFORMER_I8,
  ROPE_BASE,
  VAE_DECODER,
  TOKENIZER,
];

/** fixture の既定選択（anima-turbo / w8a8-s16）と、その部分集合になる別モデルの選択。 */
const TURBO: ResolveOptions = {};
const LITE: ResolveOptions = { model: "anima-lite", quant: "w8" };

const manifestBytes = new TextEncoder().encode(
  await Deno.readTextFile(new URL("./fixtures/manifest-fetch.json", import.meta.url)),
);

const serveAll = (): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, manifestBytes]]);
  for (const path of ASSET_PATHS) files.set(path, payloadFor(path));
  return files;
};

const load = async (
  caches: MemoryCacheStorage,
  routes: MockRoutes = { files: serveAll() },
): Promise<{ loaded: LoadedManifest; mock: ReturnType<typeof createMockFetch> }> => {
  const loader = createMockFetch(routes);
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: loader.fetch,
    caches,
  });
  return { loaded, mock: createMockFetch(routes) };
};

/** 選択の参照列（`fileRefKey` で一意化 — `prefetchAssets` は重複を受け付けない）。 */
const refsOf = (loaded: LoadedManifest, selection: ResolveOptions = {}): FileRef[] => {
  const unique = new Map<string, FileRef>();
  for (const ref of Object.values(resolveFiles(loaded.manifest, selection))) {
    if (!unique.has(fileRefKey(ref))) unique.set(fileRefKey(ref), ref);
  }
  return [...unique.values()];
};

const paths = (refs: readonly FileRef[]): string[] => refs.map((ref) => ref.path);

Deno.test("listCachedAssets: 未取得なら全て missing（取りには行かない）", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);

  const inventory = await listCachedAssets(loaded, TURBO, { caches });

  assertEquals(inventory.cached, []);
  // 順は resolveFiles（weights の宣言順 → assets の順）そのまま。
  assertEquals(paths(inventory.missing), [
    TEXT_ENCODER,
    TEXT_CONDITIONER,
    TRANSFORMER_I8,
    ROPE_BASE,
    VAE_DECODER,
    TOKENIZER,
  ]);
  assertEquals(mock.calls, [], "在庫の照会が network に出ている");
});

Deno.test("listCachedAssets: 落とした選択は全て cached・共有分だけの別選択は部分 cached", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);

  await prefetchAssets(loaded, refsOf(loaded, LITE), { fetch: mock.fetch, caches });

  const lite = await listCachedAssets(loaded, LITE, { caches });
  assertEquals(lite.missing, [], "落とした選択に欠けがある");
  assertEquals(paths(lite.cached), [TEXT_ENCODER, TRANSFORMER_I8, ROPE_BASE, TOKENIZER]);

  // 既定選択は anima-lite と 4 本を共有し、固有の 2 本（text_conditioner / vae_decoder）を欠く。
  const turbo = await listCachedAssets(loaded, TURBO, { caches });
  assertEquals(paths(turbo.cached), [TEXT_ENCODER, TRANSFORMER_I8, ROPE_BASE, TOKENIZER]);
  assertEquals(paths(turbo.missing), [TEXT_CONDITIONER, VAE_DECODER]);
});

Deno.test("listCachedAssets: 存在しない quant は ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => listCachedAssets(loaded, { quant: "w4" }, { caches }),
    ManifestReferenceError,
    "w4",
  );
  await assertRejects(
    () => evictCachedAssets(loaded, { quant: "w4" }, { caches }),
    ManifestReferenceError,
    "w4",
  );
});

Deno.test("evictCachedAssets: 他の選択が全在庫なら共有ファイルを残し、固有分だけ消す", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // 既定選択（6 本）を落とすと、その部分集合である anima-lite/w8（4 本）も全在庫になる。
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);

  const result = await evictCachedAssets(loaded, TURBO, { caches });

  assertEquals(paths(result.evicted), [TEXT_CONDITIONER, VAE_DECODER]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    TEXT_ENCODER,
    TRANSFORMER_I8,
    ROPE_BASE,
    TOKENIZER,
  ]);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `${entry.ref.path} の理由が違う`);
    // anima-turbo の f16 / f16-c16 は transformer の f16 shard を欠くので守らない。
    assertEquals(entry.sharedWith, ["anima-lite/w8"], `${entry.ref.path} を守る選択が違う`);
  }

  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), [
    TEXT_CONDITIONER,
    VAE_DECODER,
  ]);
  assertEquals(
    (await listCachedAssets(loaded, LITE, { caches })).missing,
    [],
    "守ったはずの選択が部分在庫に落ちている",
  );
});

Deno.test("evictCachedAssets: 部分在庫の選択は守らない（共有ファイルも消える）", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // 既定選択から固有の 1 本（vae_decoder）だけを落とさない = 部分在庫。
  const partial = refsOf(loaded, TURBO).filter((ref) => ref.path !== VAE_DECODER);
  await prefetchAssets(loaded, partial, { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, LITE, { caches });

  assertEquals(result.kept, []);
  assertEquals(paths(result.evicted), [TEXT_ENCODER, TRANSFORMER_I8, ROPE_BASE, TOKENIZER]);
  // 対象の選択に入っていないファイルまで巻き込まない。
  assert(
    hasEntry(hubCache(caches), payloadFor(TEXT_CONDITIONER)),
    "対象外の text_conditioner まで消えている",
  );
});

// ---- `evicted` は**取得元が「消えた」と名乗ったもの**だけ（消せる候補をそのまま返さない）。
// 組み込みの HF 取得元は候補と実際に消えたものが常に一致するので、差が出る取得元を被せないと
// この契約は観測できない（実装が `evicted: 候補` に退化しても既存テストは全て緑のまま）。

/** 渡された候補のうち**先頭 1 本だけ**を消す HF 取得元（残りは消さず、名乗りもしない）。 */
const firstOnlyEvictSource = (): DistributionSource => {
  const base = driverOf(createHfSource({ repo: REPO, hubUrl: HUB_URL, revision: SHA }));
  return new DistributionSource({
    ...base,
    pin: (generation, options) => {
      const pinned = base.pin(generation, options);
      return {
        ...pinned,
        evict: (refs) => {
          if (pinned.evict === undefined) throw new Error("HF 取得元が ⑦ 削除を持たない");
          return pinned.evict(refs.slice(0, 1));
        },
      };
    },
  });
};

Deno.test("evictCachedAssets: evicted は取得元が消したと名乗ったものだけ", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(firstOnlyEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, TURBO, { caches });

  // 消せる候補は固有の 2 本（残り 4 本は全在庫の anima-lite/w8 が守る）。取得元はその先頭だけ消す。
  assertEquals(paths(result.evicted), [TEXT_CONDITIONER]);
  // 消えなかった候補（vae_decoder）は evicted にも kept にも載らない — kept は守った 4 本だけ。
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    TEXT_ENCODER,
    TRANSFORMER_I8,
    ROPE_BASE,
    TOKENIZER,
  ]);
  assert(
    hasEntry(hubCache(caches), payloadFor(VAE_DECODER)),
    "取得元が消していない参照のエントリが消えている",
  );
});

// ---- 内容キーの同一性。在庫は (path, sha256) の組で突合する — path だけで畳むと、リポの
// 更新で中身が変わったファイルの**古いエントリ**を「在庫あり」と答えてしまう（取得は内容キーの
// ミスで再 DL になるので、在庫の表示だけが嘘になる）。

const NEXT_SHA = "abcdef0123456789abcdef0123456789abcdef01";

/** 1 weights だけの最小 manifest（path は固定・sha256 は渡したバイト列から導出）。 */
const singleFileManifest = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  JSON.stringify({
    format: "karume/4",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            i8: {
              shards: [{
                path: TEXT_ENCODER,
                size: bytes.byteLength,
                sha256: await sha256Hex(bytes),
              }],
            },
          },
        },
        assets: {},
        quants: { i8: { weights: { net: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
      },
    },
  });

Deno.test("listCachedAssets: 同じ path でも sha256 が違えば在庫にならない", async () => {
  const caches = new MemoryCacheStorage();
  const oldBytes = new TextEncoder().encode("karume-test:weights-old");
  const newBytes = new TextEncoder().encode("karume-test:weights-new-payload");
  const encoder = new TextEncoder();
  // 同じ repo の 2 世代（manifest は revision 固定 URL キーなので世代ごとに別エントリ）。
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [`${REPO}@${SHA}/${MANIFEST_PATH}`, encoder.encode(await singleFileManifest(oldBytes))],
    [`${REPO}@${SHA}/${TEXT_ENCODER}`, oldBytes],
    [`${REPO}@${NEXT_SHA}/${MANIFEST_PATH}`, encoder.encode(await singleFileManifest(newBytes))],
    [`${REPO}@${NEXT_SHA}/${TEXT_ENCODER}`, newBytes],
  ]);
  const previous = createMockFetch({ files });
  const before = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: previous.fetch,
    caches,
  });
  await prefetchAssets(before, refsOf(before), { fetch: previous.fetch, caches });
  assertEquals((await listCachedAssets(before, {}, { caches })).missing, []);

  const next = createMockFetch({ files });
  const updated = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: NEXT_SHA }, {
    fetch: next.fetch,
    caches,
  });

  const inventory = await listCachedAssets(updated, {}, { caches });
  assertEquals(inventory.cached, [], "中身が変わった path の古いエントリを在庫と答えている");
  assertEquals(paths(inventory.missing), [TEXT_ENCODER]);
});

// ---- 越境参照（`FileRef` の repo / revision — ADR 0038 §7）。在庫のキーは**参照先 repo** の
// 内容キーなので、参照元 repo に同名 path を温めても越境ぶんは在庫にならない。

const FOREIGN_REPO = "someone/text-stack";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";
const CROSS_PATH = TEXT_ENCODER;

const ownBytes = new TextEncoder().encode("karume-test:own-text-encoder");
const foreignBytes = new TextEncoder().encode("karume-test:foreign-text-encoder-payload");

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
              size: ownBytes.byteLength,
              sha256: await sha256Hex(ownBytes),
            }],
          },
        },
        borrowed: {
          i8: {
            shards: [{
              path: CROSS_PATH,
              size: foreignBytes.byteLength,
              sha256: await sha256Hex(foreignBytes),
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

const crossRepoRoutes = (): MockRoutes => ({
  sha: SHA,
  files: new Map([
    [MANIFEST_PATH, new TextEncoder().encode(crossRepoManifest)],
    [CROSS_PATH, ownBytes],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, foreignBytes],
  ]),
});

Deno.test("listCachedAssets: 越境参照の在庫は参照先 repo のキーで引く", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches, crossRepoRoutes());
  const [own, borrowed] = refsOf(loaded);

  // 参照元 repo にだけ同名 path を温める（path で畳む実装ならここで越境ぶんも cached になる）。
  await prefetchAssets(loaded, [own], { fetch: mock.fetch, caches });
  const partial = await listCachedAssets(loaded, {}, { caches });
  assertEquals(partial.cached.map(fileRefKey), [fileRefKey(own)]);
  assertEquals(
    partial.missing.map(fileRefKey),
    [fileRefKey(borrowed)],
    "越境参照が参照元 repo のキーで在庫になっている",
  );

  await prefetchAssets(loaded, [borrowed], { fetch: mock.fetch, caches });
  assertEquals(
    (await listCachedAssets(loaded, {}, { caches })).missing,
    [],
    "参照先 repo の内容キーで在庫が引けていない",
  );
});

Deno.test("evictCachedAssets: 越境参照は cross-repo として残す（エントリも消えない）", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches, crossRepoRoutes());
  const refs = refsOf(loaded);
  const [own, borrowed] = refs;
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, {}, { caches });

  assertEquals(result.evicted.map(fileRefKey), [fileRefKey(own)]);
  assertEquals(result.kept.length, 1);
  assertEquals(result.kept[0].ref, borrowed);
  assertEquals(result.kept[0].reason, "cross-repo");
  assertEquals(result.kept[0].sharedWith, []);
  assert(hasEntry(hubCache(caches), foreignBytes), "越境ぶんのエントリが消えている");
  assert(!hasEntry(hubCache(caches), ownBytes), "対象のエントリが残っている");
});

// ---- ローカル取得元（`localDirectory`）— キャッシュを持たない取得元の 2 つの答え。

const LOCAL_LABEL = "./models/karume-test";

const openLocal = async (caches: MemoryCacheStorage): Promise<LoadedManifest> => {
  const dist = await buildLocalDist();
  const directory = memoryDirectory(dist.files);
  return await loadManifest(localDirectory(directory.adapter, { label: LOCAL_LABEL }), { caches });
};

Deno.test("localDirectory: 在庫は常に全て cached（温める対象が無い）", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await openLocal(caches);

  const inventory = await listCachedAssets(loaded, {}, { caches });

  assertEquals(inventory.missing, []);
  assertEquals(inventory.cached.length, refsOf(loaded).length);
});

Deno.test("evictCachedAssets: キャッシュを持たない取得元は HubError で断る", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await openLocal(caches);

  const error = await assertRejects(
    () => evictCachedAssets(loaded, {}, { caches }),
    HubError,
  );
  assert(error.message.includes(LOCAL_LABEL), `${error.message} がどの取得元か名乗っていない`);
});
