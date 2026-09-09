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

// ---- 参照集合が**まったく同じ**選択（fixture の anima-turbo/f16 と f16-c16 — 重みファイルは
// 同じで session の設定だけ違う）。守る側に数えると互いに守り合って 1 本も消えず、どの席を
// どの順で消しても evicted が 0 件になる（下流 anima-web の報告）。

/** 同一の参照集合を指す 2 席と、その巻き添え表示に出るラベル。 */
const F16: ResolveOptions = { quant: "f16" };
const F16_C16: ResolveOptions = { quant: "f16-c16" };
const F16_C16_LABEL = "anima-turbo/f16-c16";
const W8A8_LABEL = "anima-turbo/w8a8-s16";
const LITE_LABEL = "anima-lite/w8";

/** f16 の 6 本（`resolveFiles` の順 — transformer は f16 shard）。 */
const F16_PATHS = [
  TEXT_ENCODER,
  TEXT_CONDITIONER,
  TRANSFORMER_F16,
  ROPE_BASE,
  VAE_DECODER,
  TOKENIZER,
];

/** 既定選択（w8a8-s16）と f16 の両方を温める = fixture の 4 席すべてが全在庫。 */
const prefetchAll = async (
  loaded: LoadedManifest,
  mock: ReturnType<typeof createMockFetch>,
  caches: MemoryCacheStorage,
): Promise<void> => {
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });
};

Deno.test("evictCachedAssets: 参照集合が同一の選択は守る側に数えない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  const result = await evictCachedAssets(loaded, F16, { caches });

  // f16 固有の 1 本は消える（f16-c16 が守ると 0 件になり、兄弟同士でどの順でも解けない）。
  assertEquals(paths(result.evicted), [TRANSFORMER_F16]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    TEXT_ENCODER,
    TEXT_CONDITIONER,
    ROPE_BASE,
    VAE_DECODER,
    TOKENIZER,
  ]);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `${entry.ref.path} の理由が違う`);
    assert(
      !entry.sharedWith.includes(F16_C16_LABEL),
      `${entry.ref.path} を同一集合の兄弟が守っている`,
    );
  }
  // 守るのは真部分集合 / 別集合の席だけ（順は manifest の宣言順）。
  assertEquals(result.kept[0].sharedWith, [W8A8_LABEL, LITE_LABEL], "text_encoder を守る選択");
  assertEquals(result.kept[1].sharedWith, [W8A8_LABEL], "text_conditioner を守る選択");
  assertEquals(result.alsoEvicted, [F16_C16_LABEL]);

  // 巻き添えの中身 — 兄弟は部分在庫に落ちる（次のロードで足りない 1 本だけ取り直す）。
  assertEquals(paths((await listCachedAssets(loaded, F16_C16, { caches })).missing), [
    TRANSFORMER_F16,
  ]);
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);
});

Deno.test("evictCachedAssets: 同一集合の兄弟しか居なければ全部消える", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // f16 だけを温める（w8a8-s16 と anima-lite は i8 shard を欠くので部分在庫）。
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  // 部分在庫の 2 席は巻き添えに数えない（もともと「落とし済み」ではない）。
  assertEquals(result.alsoEvicted, [F16_C16_LABEL]);
});

Deno.test("evictCachedAssets: protect の一覧だけが守る（同一集合の兄弟も明示すれば守れる）", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches, protect: [F16_C16] });

  assertEquals(result.evicted, []);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), F16_PATHS);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `${entry.ref.path} の理由が違う`);
    assertEquals(entry.sharedWith, [F16_C16_LABEL], `${entry.ref.path} を守る選択が違う`);
  }
  assertEquals(result.alsoEvicted, [], "1 本も消えていないのに巻き添えを名乗っている");
});

Deno.test("evictCachedAssets: protect が空なら全在庫の他の選択も守らない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  const result = await evictCachedAssets(loaded, F16, { caches, protect: [] });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  // 守らなかった全在庫の 3 席が巻き添え（manifest の宣言順）。
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL, LITE_LABEL]);
});

Deno.test("evictCachedAssets: protect に対象自身を混ぜても無視される", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  // 対象を自分から守ることはできない — `protect: []` と同じ結果になる。
  const result = await evictCachedAssets(loaded, F16, { caches, protect: [F16, F16] });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL, LITE_LABEL]);
});

Deno.test("evictCachedAssets: protect の存在しない quant は ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => evictCachedAssets(loaded, F16, { caches, protect: [{ quant: "w4" }] }),
    ManifestReferenceError,
    "w4",
  );
});

// ---- weights の絞り込み（`ResolveOptions.weights`）が在庫の勘定に効く形。選択の参照集合が
// 小さくなるだけなので、照会も削除も「絞ったぶん」を数える。同じ (model, quant) の残りは
// 守る側の候補にならない（label が同じものは候補から外れる）ので、**本体を残して 1 役割だけ
// 消す**が書ける — gemma4 の `drafter` を投機を使わなくなった後に落とす席。ただし共通 assets
// は守られない（`docs/known-issues.md` — この下のテストは別 (model, quant) の在庫が守っている）。

/** 既定選択の 1 役割ぶん（絞った選択も `ResolveOptions` そのもの）。 */
const TURBO_VAE: ResolveOptions = { weights: ["vae_decoder"] };

Deno.test("listCachedAssets: weights を絞ると絞ったぶんだけを数える", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, LITE), { fetch: mock.fetch, caches });

  // 既定選択は 2 本欠けているが、text_encoder に絞れば「落とし済み」と答える。
  const encoder = await listCachedAssets(loaded, { weights: ["text_encoder"] }, { caches });
  assertEquals(encoder.missing, [], "絞った役割の在庫に欠けがある");
  assertEquals(paths(encoder.cached), [TEXT_ENCODER, TOKENIZER, ROPE_BASE]);

  // 欠けている役割に絞れば、その 1 本だけが missing に出る（assets は絞りの対象外）。
  const vae = await listCachedAssets(loaded, TURBO_VAE, { caches });
  assertEquals(paths(vae.cached), [TOKENIZER, ROPE_BASE]);
  assertEquals(paths(vae.missing), [VAE_DECODER]);

  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), [
    TEXT_CONDITIONER,
    VAE_DECODER,
  ]);
});

Deno.test("listCachedAssets: 存在しない weights は ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => listCachedAssets(loaded, { weights: ["drafter"] }, { caches }),
    ManifestReferenceError,
    "drafter",
  );
  await assertRejects(
    () => evictCachedAssets(loaded, { weights: ["drafter"] }, { caches }),
    ManifestReferenceError,
    "drafter",
  );
});

Deno.test("evictCachedAssets: weights を絞ると同じ選択の残りは消えない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, TURBO_VAE, { caches });

  // 消えるのは絞った役割の固有ファイルだけ。assets 2 本は全在庫の anima-lite/w8 が守る。
  assertEquals(paths(result.evicted), [VAE_DECODER]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [TOKENIZER, ROPE_BASE]);
  for (const entry of result.kept) {
    assertEquals(entry.sharedWith, [LITE_LABEL], `${entry.ref.path} を守る選択が違う`);
  }
  // 同じ (model, quant) の残りは対象ですらない（守る側にも巻き添えにも出ない）。
  assertEquals(result.alsoEvicted, []);
  assert(
    hasEntry(hubCache(caches), payloadFor(TEXT_CONDITIONER)),
    "絞りの外にある text_conditioner まで消えている",
  );
  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), [VAE_DECODER]);
});

Deno.test("evictCachedAssets: protect は同じ label の部分集合を 2 つとも守る", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  // 同じ席（w8a8-s16）を 2 つの部分集合で守る。label で畳む実装だと後勝ちで text_encoder が消える。
  const result = await evictCachedAssets(loaded, F16, {
    caches,
    protect: [
      { quant: "w8a8-s16", weights: ["text_encoder"] },
      { quant: "w8a8-s16", weights: ["vae_decoder"] },
    ],
  });

  assertEquals(paths(result.evicted), [TEXT_CONDITIONER, TRANSFORMER_F16]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    TEXT_ENCODER,
    ROPE_BASE,
    VAE_DECODER,
    TOKENIZER,
  ]);
  for (const entry of result.kept) {
    // 同じ label が 2 度出ない（2 つの部分集合が守っていても名乗りは 1 つ）。
    assertEquals(entry.sharedWith, [W8A8_LABEL], `${entry.ref.path} を守る選択が違う`);
  }
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL]);
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

/** 候補を 1 本も消さない HF 取得元（「誰かが先に消していた」= 件数 0 の形）。 */
const noEvictSource = (): DistributionSource => {
  const base = driverOf(createHfSource({ repo: REPO, hubUrl: HUB_URL, revision: SHA }));
  return new DistributionSource({
    ...base,
    pin: (generation, options) => ({
      ...base.pin(generation, options),
      evict: () => Promise.resolve([]),
    }),
  });
};

Deno.test("evictCachedAssets: alsoEvicted は実際に消えた参照を使う選択だけ", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(firstOnlyEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAll(loaded, mock, caches);

  // anima-lite/w8 だけを守る = 候補は turbo 固有の 3 本。取得元はその先頭（text_conditioner）だけ消す。
  const result = await evictCachedAssets(loaded, F16, { caches, protect: [LITE] });

  assertEquals(paths(result.evicted), [TEXT_CONDITIONER]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [TEXT_ENCODER, ROPE_BASE, TOKENIZER]);
  // anima-lite/w8 は全在庫のまま text_encoder を共有しているが、その 1 本は消えていないので
  // 巻き添えには載らない（候補で数えると載ってしまう）。
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL]);
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);
  assert(
    hasEntry(hubCache(caches), payloadFor(VAE_DECODER)),
    "取得元が消していない候補のエントリが消えている",
  );
});

Deno.test("evictCachedAssets: 1 本も消えなければ alsoEvicted は空", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(noEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches });

  // 候補は 6 本（同一集合の f16-c16 は守らない）だが、取得元は 1 本も消したと名乗らない。
  assertEquals(result.evicted, []);
  assertEquals(result.kept, []);
  assertEquals(result.alsoEvicted, [], "誰も部分在庫に落ちていないのに巻き添えを名乗っている");
  assertEquals((await listCachedAssets(loaded, F16_C16, { caches })).missing, []);
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
