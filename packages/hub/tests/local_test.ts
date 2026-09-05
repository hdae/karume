/**
 * ローカル取得元（`localDirectory`）の振る舞い。HF 取得元との差は「無い能力」に集中している
 * ので、テストも**無いことの観測**が中心になる — 世代を名乗らない / CacheStorage を開かない /
 * 相 1 を持たない / sha256 を照合しない（size だけ見る）/ 越境先を推測しない。
 *
 * 取得元の差し替えを踏むために、fallback には fake の取得元（`DistributionSource` の内部契約を
 * 直接実装したもの）を注入する — 公開のリモート factory は段③なので、ここでは席だけを踏む。
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  fetchAssets,
  HubFetchError,
  IntegrityError,
  loadManifest,
  localDirectory,
  ManifestFormatError,
  ManifestReferenceError,
  prefetchAssets,
  resolveFiles,
  streamAssets,
} from "../mod.ts";
import type { AssetProgress } from "../mod.ts";
import { type FileRef, MANIFEST_FILENAME } from "../src/manifest.ts";
import {
  DistributionSource,
  type PinnedSource,
  type SourceDriver,
  type SourceOrigin,
} from "../src/source.ts";
import {
  buildLocalDist,
  CROSS_PATH,
  CROSS_REPO,
  CROSS_REVISION,
  memoryDirectory,
  SHARD_PATHS,
  TOKENIZER_PATH,
} from "./helpers/local.ts";
import { MemoryCacheStorage, payloadFor } from "./helpers/mock.ts";

const LABEL = "./models/karume-test";

/** 触れた瞬間に落ちる CacheStorage。ローカル取得元が 1 度も開かないことの門。 */
class HostileCacheStorage implements CacheStorage {
  open(): Promise<Cache> {
    throw new Error("test: ローカル取得元が CacheStorage を開いた");
  }
  match(): Promise<Response | undefined> {
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  has(): Promise<boolean> {
    throw new Error("test: ローカル取得元が CacheStorage を引いた");
  }
  delete(): Promise<boolean> {
    throw new Error("test: ローカル取得元が CacheStorage を消した");
  }
  /** 旧名前空間の回収（hub のセッション入口）だけは通す — 消すものが無いことを答える。 */
  keys(): Promise<string[]> {
    return Promise.resolve([]);
  }
}

const openLocal = async (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  options: Parameters<typeof localDirectory>[1] = {},
) => {
  const directory = memoryDirectory(files);
  const loaded = await loadManifest(
    localDirectory(directory.adapter, { label: LABEL, ...options }),
    { caches: new HostileCacheStorage() },
  );
  return { directory, loaded };
};

/** 進捗の記録（phase と path だけ — 数値の意味論は `stream_test` / `fetch_test` の担当）。 */
const recordProgress = (): { events: AssetProgress[]; onProgress: (p: AssetProgress) => void } => {
  const events: AssetProgress[] = [];
  return { events, onProgress: (progress) => events.push(progress) };
};

Deno.test("localDirectory: manifest → resolveFiles → fetchAssets が実体のバイト列で完走する", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);

  // 世代も repo も持たない取得元なので、識別欄は**欄ごと現れない**（偽の SHA を名乗らない）。
  assertEquals(loaded.repo, undefined);
  assertEquals(loaded.revisionSha, undefined);
  assertEquals(loaded.manifest.available.models, ["m"]);

  const files = resolveFiles(loaded.manifest);
  const assets = await fetchAssets(loaded, files);
  assertEquals(Object.keys(assets).sort(), ["net[0]", "net[1]", "tokenizer"]);
  SHARD_PATHS.forEach((path, index) => {
    assertEquals(assets[`net[${index}]`], payloadFor(path), `${path} の実体が配られていない`);
  });
  assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
  // 読んだのは manifest 1 本 + 資産 3 本だけ（同じ path を 2 度読まない）。
  assertEquals(directory.reads, [MANIFEST_FILENAME, ...SHARD_PATHS, TOKENIZER_PATH]);
});

Deno.test("localDirectory: 中断は資産の読みへ透過する", async () => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);
  await fetchAssets(loaded, resolveFiles(loaded.manifest), { signal: AbortSignal.timeout(60_000) });
  // manifest は loadManifest の signal 無し呼び出し、資産は面の signal つき。
  assertEquals(directory.signals, [false, true, true, true]);
});

Deno.test("localDirectory: streamAssets は相 2 だけで完走し、CacheStorage を 1 度も開かない", async () => {
  const dist = await buildLocalDist();
  const caches = new MemoryCacheStorage();
  const directory = memoryDirectory(dist.files);
  const loaded = await loadManifest(localDirectory(directory.adapter, { label: LABEL }), {
    caches,
  });
  const files = resolveFiles(loaded.manifest);
  const refs = SHARD_PATHS.map((_path, index): FileRef => files[`net[${index}]`]);
  const progress = recordProgress();

  const received: string[] = [];
  for await (const asset of streamAssets(loaded, refs, { onProgress: progress.onProgress })) {
    received.push(asset.id);
    assertEquals(asset.bytes, payloadFor(asset.id));
  }

  assertEquals(received, [...SHARD_PATHS], "宣言順に届いていない");
  // 相 1 が丸ごと省かれるので、1 本につき読みは 1 回だけ（温めてから読み直す形にならない）。
  assertEquals(directory.reads, [MANIFEST_FILENAME, ...SHARD_PATHS]);
  // 受信の途中という状態が無いので、進捗はファイルごとの complete 1 点だけ。
  assertEquals(progress.events.map((event) => event.phase), ["complete", "complete"]);
  assertEquals(
    caches.namespaces.size,
    0,
    "ローカル取得元がキャッシュ名前空間を作っている（バイト列を複製している）",
  );
});

Deno.test("prefetchAssets: 相 1 を持たない取得元では何もしない（入力検査だけは効く）", async (t) => {
  const dist = await buildLocalDist();
  const { directory, loaded } = await openLocal(dist.files);
  const files = resolveFiles(loaded.manifest);
  const refs = SHARD_PATHS.map((_path, index): FileRef => files[`net[${index}]`]);

  await t.step("温めるべきキャッシュが無いのは失敗ではない（no-op で返る）", async () => {
    const progress = recordProgress();
    directory.reads.length = 0;
    await prefetchAssets(loaded, refs, { onProgress: progress.onProgress });
    assertEquals(directory.reads, [], "相 1 が無いのに実体を読んでいる");
    assertEquals(progress.events, [], "何もしていないのに進捗を出している");
  });

  await t.step("呼び出し側の誤り（空の refs）は取得元に触れる前に落とす", async () => {
    await assertRejects(() => prefetchAssets(loaded, []), ManifestReferenceError, LABEL);
  });
});

Deno.test("localDirectory: size 不一致は fail loudly（path とディレクトリを名乗る）", async () => {
  const dist = await buildLocalDist();
  // 実体だけを差し替える（manifest の size / sha256 は元のまま = 途中で切れたコピーの形）。
  const tampered = new Map(dist.files);
  tampered.set(TOKENIZER_PATH, payloadFor(`${TOKENIZER_PATH}:truncated`));
  const { loaded } = await openLocal(tampered);

  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest)),
    IntegrityError,
  );
  assertEquals(error.path, TOKENIZER_PATH);
  assertEquals(error.source, "local", "ローカルの実体読みが network / cache を名乗っている");
  assertEquals(error.repo, undefined, "存在しない repo を名乗っている");
  assertEquals(error.revisionSha, undefined, "偽の SHA を名乗っている");
  assert(error.message.includes(TOKENIZER_PATH), `${error.message} が path を名乗っていない`);
  assert(error.message.includes(LABEL), `${error.message} がディレクトリを名乗っていない`);
});

Deno.test("localDirectory: sha256 は照合しない（size が合えば読み切る）", async () => {
  const dist = await buildLocalDist();
  const original = payloadFor(TOKENIZER_PATH);
  // 長さは保ったまま中身だけ変える（HF 取得元なら sha256 の門で落ちる形）。
  const swapped = new Uint8Array(original);
  swapped[swapped.length - 1] ^= 0xff;
  const tampered = new Map(dist.files);
  tampered.set(TOKENIZER_PATH, swapped);
  const { loaded } = await openLocal(tampered);

  const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest));
  assertEquals(
    assets["tokenizer"],
    swapped,
    "手元の実体を信じる設計（size だけが門）から外れている",
  );
});

Deno.test("localDirectory: 実体が無いファイルは実パスを cause に残して HubFetchError", async () => {
  const dist = await buildLocalDist();
  const missing = new Map(dist.files);
  missing.delete(SHARD_PATHS[1]);
  const { loaded } = await openLocal(missing);

  const error = await assertRejects(
    () => fetchAssets(loaded, resolveFiles(loaded.manifest)),
    HubFetchError,
  );
  assertEquals(error.path, SHARD_PATHS[1]);
  assertEquals(error.repo, undefined);
  assert(error.message.includes(LABEL), `${error.message} がディレクトリを名乗っていない`);
  assert(error.cause instanceof Error, "アダプターのエラーを cause に残していない");
  assert(
    error.cause.message.includes(SHARD_PATHS[1]),
    `${error.cause.message} が読めなかった実体を名乗っていない`,
  );
});

Deno.test("localDirectory: 上限を超えた karume.json は ManifestFormatError", async () => {
  const oversized = new Map<string, Uint8Array<ArrayBuffer>>([[
    MANIFEST_FILENAME,
    new TextEncoder().encode(" ".repeat(2 * 1024 * 1024)),
  ]]);
  const directory = memoryDirectory(oversized);
  const error = await assertRejects(
    () => loadManifest(localDirectory(directory.adapter, { label: LABEL })),
    ManifestFormatError,
  );
  assert(error.message.includes(LABEL), `${error.message} がディレクトリを名乗っていない`);
});

Deno.test("localDirectory: 壊れた karume.json は毎回同じ ManifestFormatError で落ちる", async () => {
  const broken = new Map<string, Uint8Array<ArrayBuffer>>([[
    MANIFEST_FILENAME,
    new TextEncoder().encode('{"format": "karume/4"'),
  ]]);
  const directory = memoryDirectory(broken);
  const source = localDirectory(directory.adapter, { label: LABEL });
  // ローカルには evict すべきキャッシュが無いので、self-heal も「1 往復だけ取り直す」も無い。
  for (const _attempt of [0, 1]) {
    await assertRejects(() => loadManifest(source), ManifestFormatError);
  }
  assertEquals(directory.reads, [MANIFEST_FILENAME, MANIFEST_FILENAME]);
});

// ---- 越境（ADR 0038 §7）: 明示 mapping / 明示 fallback / どちらも無い の 3 形。

/**
 * 越境先だけを提供する fake の取得元（段③の公開 factory の代役）。`shortBy` を渡すと、宣言
 * `size` より短いバイト数を名乗る**リモート**取得元になる（越境先の完全性検証の失敗を踏む形）。
 */
const fakeRemote = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  calls: [string, string][],
  options: { readonly shortBy?: number } = {},
): DistributionSource => {
  const pinnedFor = (repo: string, revision: string): PinnedSource => {
    const origin: SourceOrigin = {
      label: `fake ${repo} @ ${revision}`,
      integrity: "network",
      repo,
      revisionSha: revision,
    };
    return {
      origin,
      readManifest: () => Promise.reject(new Error("fake: manifest は読まれない")),
      readFile: (ref, { sizeViolation }) => {
        const bytes = files.get(ref.path);
        if (bytes === undefined) return Promise.reject(new Error(`fake: ${ref.path} が無い`));
        const { shortBy } = options;
        // 取得元は自分の失敗元（`origin.integrity`）を名乗って共通層へ返す — 組み立てるのは
        // 共通層（`context.ts`）。
        if (shortBy !== undefined) {
          return Promise.reject(
            sizeViolation(bytes.byteLength - shortBy, "body", origin.integrity),
          );
        }
        return Promise.resolve(new Uint8Array(bytes));
      },
      originFor: (crossRepo, crossRevision) => {
        calls.push([crossRepo, crossRevision]);
        return pinnedFor(crossRepo, crossRevision);
      },
    };
  };
  const driver: SourceDriver = {
    origin: { label: "fake remote", integrity: "network" },
    resolveGeneration: () => Promise.resolve("fake"),
    pin: (generation) => pinnedFor("fake/session", generation),
  };
  return new DistributionSource(driver);
};

const crossAssets = async (
  options: Parameters<typeof localDirectory>[1],
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const dist = await buildLocalDist({ cross: true });
  const { loaded } = await openLocal(dist.files, options);
  return await fetchAssets(loaded, resolveFiles(loaded.manifest));
};

Deno.test("localDirectory: 越境参照は明示 mapping の取得元から取る", async () => {
  const dist = await buildLocalDist({ cross: true });
  const crossDirectory = memoryDirectory(dist.crossFiles);
  const assets = await crossAssets({
    crossRepo: { [CROSS_REPO]: localDirectory(crossDirectory.adapter, { label: "./models/共有" }) },
  });
  assertEquals(assets["text_encoder"], payloadFor(`${CROSS_REPO}/${CROSS_PATH}`));
  assertEquals(crossDirectory.reads, [CROSS_PATH], "越境先から取っていない");
});

Deno.test("localDirectory: mapping に無い越境は明示 fallback へ宣言座標のまま委譲する", async () => {
  const dist = await buildLocalDist({ cross: true });
  const calls: [string, string][] = [];
  const assets = await crossAssets({
    fallback: fakeRemote(dist.crossFiles, calls),
  });
  assertEquals(assets["text_encoder"], payloadFor(`${CROSS_REPO}/${CROSS_PATH}`));
  assertEquals(calls, [[CROSS_REPO, CROSS_REVISION]], "宣言された座標で委譲していない");
});

Deno.test("localDirectory: mapping も fallback も無い越境は推測せずに落ちる", async () => {
  const error = await assertRejects(() => crossAssets({}), HubFetchError);
  assertEquals(error.path, CROSS_PATH);
  // 診断は**宣言された越境先**を名乗る（セッションのディレクトリではない）。
  assertEquals(error.repo, CROSS_REPO);
  assertEquals(error.revisionSha, CROSS_REVISION);
  assert(error.cause instanceof Error, "設定不足の理由を cause に残していない");
  assert(
    error.cause.message.includes("crossRepo") && error.cause.message.includes(CROSS_REPO),
    `${error.cause.message} が足りない設定を名乗っていない`,
  );
});

Deno.test("localDirectory: 越境先の完全性検証の失敗は越境先の失敗元を名乗る", async () => {
  const dist = await buildLocalDist({ cross: true });
  const calls: [string, string][] = [];

  const error = await assertRejects(
    () => crossAssets({ fallback: fakeRemote(dist.crossFiles, calls, { shortBy: 1 }) }),
    IntegrityError,
  );

  // 診断は宣言された越境先を名乗る（label / repo / revisionSha は既存の門と同じ）。
  assertEquals(error.path, CROSS_PATH);
  assertEquals(error.repo, CROSS_REPO);
  assertEquals(error.revisionSha, CROSS_REVISION);
  assertEquals(error.actual, String(payloadFor(`${CROSS_REPO}/${CROSS_PATH}`).byteLength - 1));
  // ローカルセッション + リモート越境（`fallback` が正式に受ける構成）。バイト列は network から
  // 来たので再試行に意味がある — セッションの分類 "local"（取り直しても同じ）を継ぐと、アプリに
  // 「回復手段は無い」と嘘を伝える。
  assertEquals(error.source, "network");
});

Deno.test("localDirectory: 越境先の取得元は同じハンドルを使い回しても混ざらない", async () => {
  const dist = await buildLocalDist({ cross: true });
  const crossDirectory = memoryDirectory(dist.crossFiles);
  const shared = localDirectory(crossDirectory.adapter, { label: "./models/共有" });
  const first = await crossAssets({ crossRepo: { [CROSS_REPO]: shared } });
  const second = await crossAssets({ crossRepo: { [CROSS_REPO]: shared } });
  assertEquals(first["text_encoder"], second["text_encoder"]);
  // セッション側の同名 path（`text_encoder/...` は自リポに存在しない）を引きに行っていない。
  assertEquals(crossDirectory.reads, [CROSS_PATH, CROSS_PATH]);
});

Deno.test("localDirectory: 取得元ハンドルは不透明（loadManifest が同一性で判別する）", async () => {
  const dist = await buildLocalDist();
  const directory = memoryDirectory(dist.files);
  const source = localDirectory(directory.adapter, { label: LABEL });
  // 公開面に生えているメンバは 1 つも無い（実装は `#driver` に閉じている）。
  assertEquals(Object.keys(source), []);
  assertEquals(Reflect.ownKeys(source).length, 0);
  const first = await loadManifest(source);
  const second = await loadManifest(source);
  // 同じハンドルから何セッション開いても、取得元の状態は共有されない（毎回読み直す）。
  assertStrictEquals(first === second, false);
  assertEquals(directory.reads, [MANIFEST_FILENAME, MANIFEST_FILENAME]);
});
