/**
 * `nodeCacheStorage` (`@karume/hub/node`): the Cache API subset the fetch layer (`@hdae/fetch-cache`)
 * uses, on the real file system. The in-memory `MemoryCacheStorage` (`helpers/mock.ts`) stands in for
 * it everywhere else in hub's tests; what only the file-backed one can get wrong is covered here —
 * the entry file format, the atomic replace, the handle lifetime of a streamed body, and the
 * namespace directories.
 *
 * The second half drives hub's Hugging Face source through it against the HF-shaped local server
 * (`@karume/test-support/local-dist-server`): the real fetch layer warms entries by streaming into
 * `put`, reads them back with the "stream" strategy (`match` per range read) and clears them through
 * `clearHubCache`. "No network" is observed with a `fetch` that rejects every call.
 *
 * The last case talks to huggingface.co and is opt-in (`KARUME_NETWORK=1`).
 */

import { makeTempDir, mkdir, readDirSync, remove, writeFile } from "@karume/test-support/fs";
import { type LocalDistServer, serveLocalDist } from "@karume/test-support/local-dist-server";
import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import {
  clearHubCache,
  fetchAssets,
  type FileRef,
  HubFetchError,
  loadManifest,
  openAsset,
  prefetchAssets,
} from "../mod.ts";
import { nodeCacheStorage } from "../node.ts";
import {
  buildLocalDist,
  FETCHED_PART_PATHS,
  PART_KEYS,
  sha256Hex,
  TOKENIZER_PATH,
} from "./helpers/local.ts";
import { payloadFor } from "./helpers/mock.ts";
import { declaredRefs, selectionFiles } from "./helpers/selection.ts";

const NAMESPACE = "ns";
const URL_A = "https://example.com/assets/a.bin";
const URL_B = "https://example.com/assets/b.bin";

const bytesOf = (text: string): Uint8Array<ArrayBuffer> => new TextEncoder().encode(text);

/** A `fetch` that stands for "the network is gone": every call rejects. */
const closedFetch: typeof globalThis.fetch = () =>
  Promise.reject(new Error("test: the network is closed"));

const withStorage = async (
  body: (root: string, caches: CacheStorage) => Promise<void>,
): Promise<void> => {
  const root = await makeTempDir({ prefix: "karume-hub-cache-" });
  try {
    await body(root, nodeCacheStorage(root));
  } finally {
    await remove(root, { recursive: true });
  }
};

/** A body that delivers `head` and then fails — an aborted transfer, or a hash mismatch the fetch layer turns into a stream error. */
const failingBody = (
  head: Uint8Array<ArrayBuffer>,
  reason: Error,
): ReadableStream<Uint8Array<ArrayBuffer>> => {
  let delivered = false;
  return new ReadableStream({
    pull: (controller) => {
      if (delivered) {
        controller.error(reason);
        return;
      }
      delivered = true;
      controller.enqueue(head);
    },
  });
};

/** Bytes that cross the 1 MiB read size a few times, with a pattern that catches a shifted read. */
const largeBody = (): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(new ArrayBuffer(3 * (1 << 20) + 7));
  for (let index = 0; index < bytes.length; index++)
    bytes[index] = (index * 7 + (index >> 12)) & 0xff;
  return bytes;
};

const assertSameBytes = (actual: Uint8Array, expected: Uint8Array, what: string): void => {
  assertEquals(actual.byteLength, expected.byteLength, `${what}: length`);
  for (let index = 0; index < expected.length; index++) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${what}: byte ${index} is ${actual[index]}, expected ${expected[index]}`);
    }
  }
};

/** Everything under the namespace directory (temporary files included — they must not linger). */
const entryFiles = (root: string): string[] =>
  readDirSync(`${root}/cache-${NAMESPACE}`)
    .map((entry) => entry.name)
    .sort();

describe("nodeCacheStorage: the Cache API subset the fetch layer uses", () => {
  it("put / match round-trips bytes, status and headers; keys lists the key URL", async () => {
    await withStorage(async (_root, caches) => {
      const cache = await caches.open(NAMESPACE);
      await cache.put(
        URL_A,
        new Response(bytesOf("hello"), {
          status: 200,
          statusText: "OK",
          headers: { "x-record": "abc", "content-type": "application/octet-stream" },
        }),
      );
      const matched = await cache.match(URL_A);
      assert(matched !== undefined, "put entry does not match");
      assertEquals(matched.status, 200);
      assertEquals(matched.statusText, "OK");
      assertEquals(matched.headers.get("x-record"), "abc");
      assertEquals(matched.headers.get("content-type"), "application/octet-stream");
      assertEquals(new Uint8Array(await matched.arrayBuffer()), bytesOf("hello"));
      assertEquals(
        (await cache.keys()).map((request) => request.url),
        [URL_A],
      );
      assertEquals(
        (await cache.keys(URL_A)).map((request) => request.url),
        [URL_A],
      );
      assertEquals(await cache.keys(URL_B), []);
      assertEquals(await cache.match(URL_B), undefined);
      assertStrictEquals(await caches.has(NAMESPACE), true);
      assertEquals(await caches.keys(), [NAMESPACE]);
    });
  });

  it("keys are normalised like the Cache API: fragment dropped, host lower-cased, Request / URL accepted", async () => {
    await withStorage(async (_root, caches) => {
      const cache = await caches.open(NAMESPACE);
      await cache.put("https://Example.com/x?q=1#frag", new Response(bytesOf("x")));
      const spellings: Record<string, RequestInfo | URL> = {
        string: "https://example.com/x?q=1",
        URL: new URL("https://example.com/x?q=1#other"),
        Request: new Request("https://EXAMPLE.com/x?q=1"),
      };
      for (const [form, spelling] of Object.entries(spellings)) {
        const matched = await cache.match(spelling);
        assert(matched !== undefined, `the ${form} spelling does not match`);
        await matched.body?.cancel();
      }
      assertEquals(
        (await cache.keys()).map((request) => request.url),
        ["https://example.com/x?q=1"],
      );
      assertStrictEquals(await cache.delete(new URL("https://example.com/x?q=1")), true);
      assertEquals(await cache.keys(), []);
    });
  });

  it("a body that fails mid-stream leaves no entry and no temporary file behind", async () => {
    await withStorage(async (root, caches) => {
      const cache = await caches.open(NAMESPACE);
      const reason = new Error("test: transfer aborted");
      const error = await assertRejects(() =>
        cache.put(URL_A, new Response(failingBody(bytesOf("partial"), reason))),
      );
      assertStrictEquals(error, reason, "put must reject with the stream's own error");
      assertEquals(await cache.match(URL_A), undefined, "a failed put left an entry");
      assertEquals(entryFiles(root), [], "a failed put left a file");
    });
  });

  it("put replaces an entry atomically: a reader of the old file keeps it, a failed replacement keeps the current one", async () => {
    await withStorage(async (root, caches) => {
      const cache = await caches.open(NAMESPACE);
      const first = largeBody();
      const second = bytesOf("second");
      await cache.put(URL_A, new Response(first));
      const reader = await cache.match(URL_A);
      assert(reader !== undefined);

      await cache.put(URL_A, new Response(second));
      assertSameBytes(new Uint8Array(await reader.arrayBuffer()), first, "old reader");
      const current = await cache.match(URL_A);
      assert(current !== undefined);
      assertEquals(new Uint8Array(await current.arrayBuffer()), second);

      await assertRejects(() =>
        cache.put(URL_A, new Response(failingBody(bytesOf("third"), new Error("test: aborted")))),
      );
      const kept = await cache.match(URL_A);
      assert(kept !== undefined, "a failed replacement removed the entry");
      assertEquals(new Uint8Array(await kept.arrayBuffer()), second);
      assertEquals(entryFiles(root).length, 1, "a failed replacement left a file");
    });
  });

  it("a streamed body is read in chunks and can be cancelled unread", async () => {
    await withStorage(async (_root, caches) => {
      const cache = await caches.open(NAMESPACE);
      const bytes = largeBody();
      await cache.put(URL_A, new Response(bytes));
      const streamed = await cache.match(URL_A);
      assert(streamed !== undefined && streamed.body !== null);
      const chunks: Uint8Array[] = [];
      for await (const chunk of streamed.body) chunks.push(chunk);
      assert(chunks.length >= 3, `expected several reads, got ${chunks.length}`);
      const joined = new Uint8Array(bytes.byteLength);
      let offset = 0;
      for (const chunk of chunks) {
        joined.set(chunk, offset);
        offset += chunk.byteLength;
      }
      assertSameBytes(joined, bytes, "streamed body");

      const unread = await cache.match(URL_A);
      assert(unread !== undefined && unread.body !== null);
      await unread.body.cancel();
      assertStrictEquals(await caches.delete(NAMESPACE), true);
    });
  });

  it("delete answers true once, then false", async () => {
    await withStorage(async (_root, caches) => {
      const cache = await caches.open(NAMESPACE);
      await cache.put(URL_A, new Response(bytesOf("a")));
      assertStrictEquals(await cache.delete(URL_A), true);
      assertEquals(await cache.match(URL_A), undefined);
      assertStrictEquals(await cache.delete(URL_A), false);
    });
  });

  it("namespaces are directories under the root, whatever their name", async () => {
    await withStorage(async (root, caches) => {
      const names = ["fetch-cache", "karume/1:auth:0123", "..", ""];
      for (const name of names) await caches.open(name);
      assertEquals((await caches.keys()).sort(), [...names].sort());
      assertEquals(
        readDirSync(root)
          .map((entry) => entry.name)
          .sort(),
        ["cache-", "cache-..", "cache-fetch-cache", "cache-karume%2F1%3Aauth%3A0123"],
        "a namespace escaped the root or was spelled differently",
      );
      assertStrictEquals(await caches.has("karume/1:auth:0123"), true);
      assertStrictEquals(await caches.delete("karume/1:auth:0123"), true);
      assertStrictEquals(await caches.has("karume/1:auth:0123"), false);
      assertStrictEquals(await caches.delete("karume/1:auth:0123"), false);
      assertStrictEquals(await caches.has("never-opened"), false);
    });
  });

  it("what the fetch layer never calls rejects instead of pretending", async () => {
    await withStorage(async (_root, caches) => {
      const cache = await caches.open(NAMESPACE);
      for (const call of [
        () => cache.add(URL_A),
        () => cache.addAll([URL_A]),
        () => cache.matchAll(),
        () => caches.match(URL_A),
        () => cache.match(URL_A, { ignoreSearch: true }),
        () => cache.keys(undefined, { ignoreVary: true }),
        () => cache.delete(URL_A, { ignoreMethod: true }),
      ]) {
        const error = await assertRejects(call, Error);
        assert(error.message.includes("not supported"), error.message);
      }
    });
  });

  it("a file that is not an entry fails loudly on match instead of reading as a miss", async () => {
    await withStorage(async (root, caches) => {
      const cache = await caches.open(NAMESPACE);
      const path = `${root}/cache-${NAMESPACE}/${await sha256Hex(bytesOf(URL_A))}`;
      await writeFile(path, bytesOf("no header line at all"));
      const noHeader = await assertRejects(() => cache.match(URL_A), Error);
      assert(noHeader.message.includes("not a cache entry"), noHeader.message);
      await writeFile(path, bytesOf("{not json}\nbody"));
      const badHeader = await assertRejects(() => cache.match(URL_A), Error);
      assert(badHeader.message.includes("corrupt header"), badHeader.message);
      await writeFile(path, bytesOf('{"url":"x"}\nbody'));
      const incomplete = await assertRejects(() => cache.match(URL_A), Error);
      assert(incomplete.message.includes("corrupt header"), incomplete.message);
    });
  });
});

/** Write a synthetic distribution to a directory the local server can serve. */
const materialize = async (
  directory: string,
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): Promise<void> => {
  for (const [path, bytes] of files) {
    const at = `${directory}/${path}`;
    await mkdir(at.slice(0, at.lastIndexOf("/")), { recursive: true });
    await writeFile(at, bytes);
  }
};

describe("nodeCacheStorage: hub's Hugging Face source on Node, against the local HF-shaped server", () => {
  let root: string | undefined;
  let server: LocalDistServer | undefined;
  let caches: CacheStorage | undefined;
  beforeAll(async () => {
    root = await makeTempDir({ prefix: "karume-hub-cache-e2e-" });
    const distribution = `${root}/dist`;
    await materialize(distribution, (await buildLocalDist()).files);
    server = await serveLocalDist(distribution);
    caches = nodeCacheStorage(`${root}/cache`);
  });
  afterAll(async () => {
    await server?.close();
    if (root !== undefined) await remove(root, { recursive: true });
  });
  const setup = (): {
    readonly source: LocalDistServer["source"];
    readonly caches: CacheStorage;
  } => {
    assert(server !== undefined && caches !== undefined, "setup did not run");
    return { source: server.source, caches };
  };

  it("prefetchAssets warms the cache on disk; fetchAssets reads it back with the network closed", async () => {
    const { source, caches } = setup();
    const loaded = await loadManifest(source, { caches });
    const files = selectionFiles(loaded.manifest);
    await prefetchAssets(loaded, Object.values(files), { caches });
    const assets = await fetchAssets(loaded, files, { caches, fetch: closedFetch });
    PART_KEYS.forEach((key, index) => {
      assertEquals(assets[key], payloadFor(FETCHED_PART_PATHS[index]));
    });
    assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
  });

  it("openAsset reads a range from the cached entry (stream strategy) with the network closed", async () => {
    const { source, caches } = setup();
    const loaded = await loadManifest(source, { caches });
    const tokenizer = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, tokenizer, { caches, fetch: closedFetch });
    assert(reader !== undefined, "the warmed asset did not open");
    assertEquals(reader.cost, "scan");
    const expected = payloadFor(TOKENIZER_PATH);
    assertEquals(Array.from(await reader.read(3, 5)), Array.from(expected.subarray(3, 8)));
    assertEquals(Array.from(await reader.read(0, expected.byteLength)), Array.from(expected));
  });

  it("clearHubCache removes the warmed entries; the next load warms them again", async () => {
    const { source, caches } = setup();
    assertStrictEquals(await clearHubCache({ caches }), true);
    const loaded = await loadManifest(source, { caches });
    const files = selectionFiles(loaded.manifest);
    await assertRejects(
      () => fetchAssets(loaded, files, { caches, fetch: closedFetch }),
      HubFetchError,
    );
    const assets = await fetchAssets(loaded, files, { caches });
    assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
    assertStrictEquals(await clearHubCache({ caches }), true);
    assertStrictEquals(await clearHubCache({ caches }), false, "nothing was left to clear");
  });
});

describe.skipIf(process.env.KARUME_NETWORK !== "1")(
  "nodeCacheStorage: Hugging Face over the network (opt-in KARUME_NETWORK=1)",
  () => {
    // A public distribution with a 324-byte part; pinned so the case does not move with `main`.
    const source = {
      repo: "hdae/karume-depth-anything-v2",
      revision: "b3969c2e64dba345a4ec996231abd642c06a8efd",
    };

    test("warms the smallest declared file from huggingface.co and serves it from disk afterwards", async () => {
      await withStorage(async (_root, caches) => {
        const loaded = await loadManifest(source, { caches });
        const smallest = declaredRefs(loaded.manifest)
          .filter((ref) => ref.size > 0)
          .reduce((best: FileRef, ref) => (ref.size < best.size ? ref : best));
        await prefetchAssets(loaded, [smallest], { caches });
        const assets = await fetchAssets(
          loaded,
          { file: smallest },
          { caches, fetch: closedFetch },
        );
        assertEquals(assets["file"].byteLength, smallest.size);
      });
    });
  },
);
