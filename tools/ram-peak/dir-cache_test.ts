// ディレクトリ固定 CacheStorage の門（GPU 不要）。
//
// 固定するのは harness が寄りかかっている 3 点だけ（`dir-cache.ts` のモジュール doc の①②③）:
//
//  ・`match` の body が**ファイルのストリーム**である（全量をヒープに載せない）
//  ・**ヘッダごと**往復する（取得層の記録ハッシュが warm まで生き残る）
//  ・キー → ファイル名の写像が `crypto.subtle.digest` を**呼ばない**（digest の計数を汚さない）
//
// 加えて「別プロセスで開き直しても同じキーが同じ在庫を指す」（cold → warm の前提）を、
// 同じディレクトリに `directoryCaches` を 2 度作ることで観測する。

import { assert, assertEquals } from "@std/assert";
import { directoryCaches, resetCacheDirectory } from "./dir-cache.ts";

const KEY = "https://hub.test/karume-local/dist/resolve/main/dit/model-00001-of-00002.krm";
const SHA_HEADER = "x-fetch-cache-sha256";

const withTempDir = async (body: (root: string) => Promise<void>): Promise<void> => {
  const root = await Deno.makeTempDir({ prefix: "karume-dir-cache-" });
  try {
    await body(root);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

Deno.test("dir-cache: put → match がヘッダごと往復し、実績を数える", async () => {
  await withTempDir(async (root) => {
    const caches = directoryCaches(root);
    const cache = await caches.open("fetch-cache");
    const payload = new Uint8Array([1, 2, 3, 4, 5]);
    await cache.put(
      KEY,
      new Response(payload, { headers: { [SHA_HEADER]: "a".repeat(64) } }),
    );

    const hit = await cache.match(KEY);
    assert(hit !== undefined, "put した在庫が引けない");
    assertEquals(hit.headers.get(SHA_HEADER), "a".repeat(64), "記録ハッシュが落ちている");
    assertEquals(new Uint8Array(await hit.arrayBuffer()), payload);

    const miss = await cache.match(`${KEY}.other`);
    assertEquals(miss, undefined, "置いていないキーが引けてしまう");

    const stats = caches.stats();
    assertEquals(stats.puts, 1);
    assertEquals(stats.putBytes, payload.byteLength);
    assertEquals(stats.hits, 1);
    assertEquals(stats.misses, 1);
    assertEquals(stats.matches, 2);
  });
});

Deno.test("dir-cache: match の body はファイルのストリーム（全量をヒープに載せない）", async () => {
  await withTempDir(async (root) => {
    const caches = directoryCaches(root);
    const cache = await caches.open("fetch-cache");
    await cache.put(KEY, new Response(new Uint8Array(1024)));
    const hit = await cache.match(KEY);
    assert(hit !== undefined);
    assert(hit.body !== null, "body が無い（全量を抱えた Response になっている）");
    // 1 チャンク目だけ読んで打ち切れる = 取得層の "stream" 戦略が成立する形。
    const reader = hit.body.getReader();
    const first = await reader.read();
    assert(first.done === false && first.value.byteLength > 0);
    await reader.cancel();
  });
});

Deno.test("dir-cache: 同じディレクトリを開き直すと在庫が見える（cold → warm の前提）", async () => {
  await withTempDir(async (root) => {
    const writer = await directoryCaches(root).open("fetch-cache");
    await cachePut(writer);
    // 別プロセスの warm に相当する（Map の中身を引き継がない新しい CacheStorage）。
    const reader = directoryCaches(root);
    const cache = await reader.open("fetch-cache");
    assert(await cache.match(KEY) !== undefined, "開き直すと在庫が見えない");
    assertEquals(reader.stats().puts, 0, "warm 側が書いている");
    assertEquals(reader.stats().hits, 1);
  });
});

Deno.test("dir-cache: キー → ファイル名の写像は crypto.subtle.digest を呼ばない", async () => {
  await withTempDir(async (root) => {
    const subtle = crypto.subtle;
    const original = subtle.digest.bind(subtle);
    let calls = 0;
    Object.defineProperty(subtle, "digest", {
      value: ((algorithm, data) => {
        calls += 1;
        return original(algorithm, data);
      }) as SubtleCrypto["digest"],
      configurable: true,
      writable: true,
    });
    try {
      const cache = await directoryCaches(root).open("fetch-cache");
      await cachePut(cache);
      await cache.match(KEY);
      await cache.keys();
      await cache.delete(KEY);
    } finally {
      Object.defineProperty(subtle, "digest", {
        value: original,
        configurable: true,
        writable: true,
      });
    }
    assertEquals(calls, 0, "キャッシュ実装が digest を呼んでいる（検収③の計数が汚れる）");
  });
});

Deno.test("dir-cache: keys は元の URL を返し、delete は在庫の有無を答える", async () => {
  await withTempDir(async (root) => {
    const cache = await directoryCaches(root).open("fetch-cache");
    await cachePut(cache);
    assertEquals((await cache.keys()).map((request) => request.url), [KEY]);
    assertEquals(await cache.delete(KEY), true);
    assertEquals(await cache.delete(KEY), false);
    assertEquals(await cache.keys(), []);
  });
});

Deno.test("dir-cache: 上書きの put が決着するまで、古い添え状は在庫として見えない", async () => {
  await withTempDir(async (root) => {
    const cache = await directoryCaches(root).open("fetch-cache");
    await cache.put(
      KEY,
      new Response(new Uint8Array([1, 1]), { headers: { [SHA_HEADER]: "a".repeat(64) } }),
    );

    // 本文を流している最中で止められる応答（highWaterMark 0 = 読まれるまで pull しない）。
    let release = (): void => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    let streaming = (): void => {};
    const started = new Promise<void>((resolve) => (streaming = resolve));
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        streaming();
        await gate;
        controller.enqueue(new Uint8Array([2, 2, 2]));
        controller.close();
      },
    }, { highWaterMark: 0 });
    const putting = cache.put(
      KEY,
      new Response(body, { headers: { [SHA_HEADER]: "b".repeat(64) } }),
    );
    await started;
    assertEquals(await cache.match(KEY), undefined, "決着前の put で古い添え状が見えている");

    release();
    await putting;
    const hit = await cache.match(KEY);
    assert(hit !== undefined, "上書きした在庫が引けない");
    assertEquals(hit.headers.get(SHA_HEADER), "b".repeat(64), "添え状が古いまま");
    assertEquals(new Uint8Array(await hit.arrayBuffer()), new Uint8Array([2, 2, 2]));
    const names: string[] = [];
    for await (const entry of Deno.readDir(`${root}/fetch-cache`)) names.push(entry.name);
    assertEquals(names.filter((name) => name.endsWith(".partial")), [], "一時名が残っている");
  });
});

Deno.test("dir-cache: resetCacheDirectory は置き場を空にする", async () => {
  await withTempDir(async (root) => {
    const cache = await directoryCaches(root).open("fetch-cache");
    await cachePut(cache);
    await resetCacheDirectory(root);
    const reopened = await directoryCaches(root).open("fetch-cache");
    assertEquals(await reopened.match(KEY), undefined, "消したはずの在庫が残っている");
  });
});

const cachePut = (cache: Cache): Promise<void> =>
  cache.put(KEY, new Response(new Uint8Array([9, 9, 9])));
