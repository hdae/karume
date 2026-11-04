/**
 * The fetch surface for one container (`openContainerSource` — ADR 0109 decision 7).
 *
 * Nine points are pinned here:
 *  ① Opening alone **fetches nothing** (warming is done first by the caller with `prefetchAssets`).
 *  ② Opening without warming still reads (the HF source warms again once on the first read). When
 *     already warm, 0 network requests.
 *  ③ `read` returns exactly the range (for the scan type, a view of the buffer of the part in the
 *     holding slot — no copy; reading the next part does not corrupt the earlier view), and an
 *     out-of-range read, a zero-length part or an unknown index fails loudly.
 *  ④ `verified` says **whether the source verified the whole file** (HF = true / a local directory
 *     = false; for a cross-repository reference the referenced source decides).
 *  ⑤ The scan type reads the whole part **once** however many times the same part is read (once
 *     even for parallel reads).
 *  ⑥ The seek type passes only the requested range down to the adapter.
 *  ⑦ A cross-repository container is read from the referenced source. Without a mapping, opening
 *     fails with `HubFetchError`.
 *  ⑧ A read failure is the source's plain `Error` (this surface is not a fetch).
 *  ⑨ Failures are not remembered — a part whose reader could not be opened and a part whose whole
 *     read failed are both fetched again on the next read.
 */

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import {
  type DirectoryAdapter,
  HubFetchError,
  type LoadedManifest,
  loadManifest,
  localDirectory,
  openContainerSource,
  prefetchAssets,
  resolveSelection,
} from "../mod.ts";
import { createHfSource } from "../src/sources/hf.ts";
import { memoryDirectory } from "./helpers/local.ts";
import {
  createMockFetch,
  HUB_URL,
  hubCache,
  MemoryCacheStorage,
  type MockFetch,
  payloadFor,
  REPO,
  SHA,
} from "./helpers/mock.ts";

const MANIFEST_PATH = "karume.json";
const HEADER_BYTES = 24;
const MODEL_DOC_BYTES = 5;
const EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

const FOREIGN_REPO = "someone/shared";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";

/**
 * A container with a zero-length part 1 (the shape of ADR 0109 decision 3). The numbering is the
 * writer's spelling (starting at 1 — `container_paths` of
 * `tools/exporter/src/karume/container.py`).
 */
const NET_PARTS = [
  "net/model-00001-of-00003.krm",
  "net/model-00002-of-00003.krm",
  "net/model-00003-of-00003.krm",
];
const EMPTY_INDEX = 1;
/** The parts that are fetched and read (the 2 without the zero-length one). */
const NET_FETCHED = NET_PARTS.filter((_path, index) => index !== EMPTY_INDEX);

const CROSS_PARTS = ["shared/model-00001-of-00002.krm", "shared/model-00002-of-00002.krm"];

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const partRef = async (path: string, cross: boolean) => ({
  path,
  size: payloadFor(path).byteLength,
  sha256: await sha256Hex(payloadFor(path)),
  ...(cross ? { repo: FOREIGN_REPO, revision: FOREIGN_SHA } : {}),
});

/**
 * `descriptor` is derived from the length of part 0 (= header + the 2 documents) (container-v1 §8).
 */
const containerFor = async (
  paths: readonly string[],
  options: { readonly emptyIndex?: number; readonly cross?: boolean } = {},
) => {
  const parts = await Promise.all(
    paths.map(async (path, index) =>
      index === options.emptyIndex
        ? {
            path,
            size: 0,
            sha256: EMPTY_SHA,
            ...(options.cross === true ? { repo: FOREIGN_REPO, revision: FOREIGN_SHA } : {}),
          }
        : await partRef(path, options.cross === true),
    ),
  );
  return {
    descriptor: {
      graph: {
        length: parts[0].size - HEADER_BYTES - MODEL_DOC_BYTES,
        sha256: await sha256Hex(payloadFor(`graph:${paths[0]}`)),
      },
      model: {
        length: MODEL_DOC_BYTES,
        sha256: await sha256Hex(payloadFor(`model:${paths[0]}`)),
      },
    },
    parts,
  };
};

const manifestText = JSON.stringify({
  format: "karume/5",
  generator: "karume/0.13.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        net: { f16: { container: await containerFor(NET_PARTS, { emptyIndex: EMPTY_INDEX }) } },
        shared: { f16: { container: await containerFor(CROSS_PARTS, { cross: true }) } },
      },
      assets: {},
      quants: { f16: { weights: { net: "f16", shared: "f16" }, session: {} } },
      defaultQuant: "f16",
      pipelineConfig: {},
    },
  },
});

const manifestBytes = new TextEncoder().encode(manifestText);

/**
 * The file table the HF source serves (cross-repository files are registered under qualified keys).
 */
const serveAll = (
  overrides: ReadonlyMap<string, Uint8Array<ArrayBuffer>> = new Map(),
): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, manifestBytes]]);
  for (const path of NET_FETCHED) files.set(path, payloadFor(path));
  for (const path of CROSS_PARTS) {
    files.set(`${FOREIGN_REPO}@${FOREIGN_SHA}/${path}`, payloadFor(path));
  }
  for (const [path, bytes] of overrides) files.set(path, bytes);
  return files;
};

/**
 * The directory contents (the zero-length part is placed as a 0-byte file too — container-v1 §8).
 */
const localFiles = (): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, manifestBytes]]);
  NET_PARTS.forEach((path, index) => {
    files.set(path, index === EMPTY_INDEX ? new Uint8Array(new ArrayBuffer(0)) : payloadFor(path));
  });
  return files;
};

const resolveUrl = (path: string): string => `${HUB_URL}/${REPO}/resolve/${SHA}/${path}`;
const foreignUrl = (path: string): string =>
  `${HUB_URL}/${FOREIGN_REPO}/resolve/${FOREIGN_SHA}/${path}`;

const countCalls = (calls: readonly string[], url: string): number =>
  calls.filter((call) => call === url).length;

const openRemote = async (
  caches: MemoryCacheStorage,
  files: Map<string, Uint8Array<ArrayBuffer>> = serveAll(),
): Promise<{ loaded: LoadedManifest; mock: MockFetch }> => {
  const loader = createMockFetch({ files });
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: loader.fetch,
      caches,
    },
  );
  // From here on a mock with an empty record is passed, so that the manifest fetch is not counted.
  return { loaded, mock: createMockFetch({ files }) };
};

const containerOf = (loaded: LoadedManifest, name: string) =>
  resolveSelection(loaded.manifest).containers[name];

/**
 * Phase 1 (warming) is **the caller's responsibility** — this surface does not warm, so the tests
 * call the step that runs first explicitly too.
 */
const warmParts = async (
  loaded: LoadedManifest,
  name: string,
  options: { readonly fetch: typeof globalThis.fetch; readonly caches: CacheStorage },
): Promise<void> => {
  await prefetchAssets(
    loaded,
    containerOf(loaded, name).parts.filter((ref) => ref.size > 0),
    {
      fetch: options.fetch,
      caches: options.caches,
    },
  );
};

/** A local directory session (only the referenced repository can be swapped in). */
const openLocal = async (
  options: Parameters<typeof localDirectory>[1] = {},
): Promise<{ loaded: LoadedManifest; directory: ReturnType<typeof memoryDirectory> }> => {
  const directory = memoryDirectory(localFiles());
  const loaded = await loadManifest(
    localDirectory(directory.adapter, { label: "./models/test", ...options }),
    { caches: new MemoryCacheStorage() },
  );
  return { loaded, directory };
};

test("openContainerSource: opening alone fetches not a single byte", async () => {
  // Warming belongs to the caller's order (admission on the descriptor → prefetchAssets → open).
  // Warming every part inside the surface would download the weights of a model that cannot run
  // first (ADR 0108 decision 19 / 0070 decision 5).
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);

  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(mock.calls, [], "opening alone goes to the network");
  assertEquals(source.partCount, NET_PARTS.length);
});

test("openContainerSource: without warming, the first read warms only the part it touched", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(await source.read(0, 0, 4), payloadFor(NET_PARTS[0]).subarray(0, 4));
  assertEquals(
    countCalls(mock.calls, resolveUrl(NET_PARTS[0])),
    1,
    "the re-warm did not happen exactly once",
  );
  // A part that was not touched is not fetched at all (the observation point that the surface does
  // not warm every part).
  for (const path of [NET_PARTS[1], NET_PARTS[2]]) {
    assertEquals(countCalls(mock.calls, resolveUrl(path)), 0, `${path} is fetched ahead of time`);
  }

  const warmed = mock.calls.length;
  await source.read(0, 4, 4);
  assertEquals(mock.calls.length, warmed, "the second read of the same part goes to the network");
});

test("openContainerSource: when warmed beforehand, reads never go to the network", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  await warmParts(loaded, "net", { fetch: mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: second.fetch,
    caches,
  });
  for (const index of [0, 2]) await source.read(index, 0, source.partLength(index));
  assertEquals(second.calls, [], "goes to the network although already warm");
});

test("openContainerSource: partCount / partLength answer the declaration as is", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  // The zero-length part **is counted** too (the index is the part id, so closing the gap would
  // shift the following ones).
  assertEquals(source.partCount, NET_PARTS.length);
  assertEquals(source.partLength(EMPTY_INDEX), 0);
  assertEquals(source.partLength(0), payloadFor(NET_PARTS[0]).byteLength);
  assertEquals(source.partLength(2), payloadFor(NET_PARTS[2]).byteLength);
});

test("openContainerSource: a scan-type read returns a view of the holding slot's part buffer without copying", async () => {
  // A copy would hold the part length twice per part, in the buffer and in the copy
  // (container-v1 §11). The consumer reads honouring byteOffset (alignment holds because the buffer
  // is tight + blocks start 64 B aligned — the MUST of container.ts).
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const payload = payloadFor(NET_PARTS[2]);
  const middle = Math.floor(payload.byteLength / 2);

  const views: Uint8Array<ArrayBuffer>[] = [];
  for (const [offset, length] of [
    [0, 4],
    [middle, 3],
    [payload.byteLength - 5, 5],
    [2, 0],
  ]) {
    const bytes = await source.read(2, offset, length);
    assertEquals(
      bytes,
      new Uint8Array(payload.subarray(offset, offset + length)),
      `the content of [${offset}, ${offset + length}) is corrupted`,
    );
    // The range's position shows in byteOffset and the whole part in buffer (a copy would have
    // byteOffset 0 and buffer = range length).
    assertEquals(
      bytes.byteOffset,
      offset,
      `the position of [${offset}, ${offset + length}) is off`,
    );
    assertEquals(bytes.buffer.byteLength, source.partLength(2), "buffer is not the part's buffer");
    views.push(bytes);
  }
  // Every range of the same part points at the same buffer (no copy per range).
  for (const view of views.slice(1)) {
    assertStrictEquals(
      view.buffer,
      views[0].buffer,
      "ranges of the same part are copied into different buffers",
    );
  }
});

test("openContainerSource: with the scan type, reading the next part does not corrupt the view of the earlier part", async () => {
  // When the holding slot moves to another part, the released buffer is not overwritten. An
  // implementation that reuses the buffer across parts (allocating one receiver for whole reads and
  // reading the next part into it) would silently turn the content of a view the caller still holds
  // into the next part — this check turns that red.
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const lengthOf = (index: number): number => source.partLength(index);

  const earlier = await source.read(2, 0, lengthOf(2));
  const snapshot = earlier.slice();
  assertEquals(snapshot, payloadFor(NET_PARTS[2]));

  // Read another part to move the holding slot, then read the original part again (the buffer is
  // allocated twice).
  const other = await source.read(0, 0, lengthOf(0));
  const again = await source.read(2, 0, lengthOf(2));

  assertEquals(earlier, snapshot, "the content of the released part's view was rewritten");
  assertEquals(other, payloadFor(NET_PARTS[0]));
  assertEquals(again, snapshot);
});

test("openContainerSource: an out-of-range read, a zero-length part and an unknown index fail loudly", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const size = source.partLength(2);

  for (const [offset, length] of [
    [size - 1, 2],
    [size, 1],
    [-1, 2],
    [0, 1.5],
    [0, size + 1],
  ]) {
    const error = await assertRejects(() => source.read(2, offset, length), Error);
    assert(
      error.message.includes(String(size)),
      `${error.message} does not name the declared size`,
    );
  }

  // A zero-length part cannot be read (a reader does not read once partLength answers 0).
  const empty = await assertRejects(() => source.read(EMPTY_INDEX, 0, 0), Error);
  assert(
    empty.message.includes("zero-length"),
    `${empty.message} does not name the zero-length part`,
  );

  for (const index of [NET_PARTS.length, -1, 1.5]) {
    const error = await assertRejects(() => source.read(index, 0, 1), Error);
    assert(error.message.includes(`part ${index}`), `${error.message} does not name the index`);
  }
  // The synchronous methods fail by the same rule.
  for (const index of [NET_PARTS.length, -1]) {
    try {
      source.partLength(index);
      throw new Error(`partLength(${index}) passed`);
    } catch (error) {
      assert(
        error instanceof Error && error.message.includes(`part ${index}`),
        `${error} does not name the index`,
      );
    }
  }
  // The checks are closed over the declaration, so it fails without fetching anything.
  assertEquals(mock.calls, [], "a read of an invalid range goes to the network");
});

test("openContainerSource: a bad range names the part, its path and its declared size", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const size = source.partLength(2);

  await assertRejects(
    () => source.read(2, size - 1, 2),
    Error,
    `hub: part 2 (${NET_PARTS[2]}) range [${size - 1}, ${size + 1}) is invalid` +
      ` (declared size ${size} — only ranges of non-negative integers within size can be read)`,
  );
  for (const [offset, length] of [
    [Number.NaN, 1],
    [0, Number.POSITIVE_INFINITY],
  ]) {
    await assertRejects(() => source.read(2, offset, length), Error, "declared size");
  }
  assertEquals(mock.calls, [], "a bad range reached the network");
  // The whole declared part is still readable (warm first — this surface does not warm).
  await warmParts(loaded, "net", { fetch: mock.fetch, caches });
  assertEquals(await source.read(2, 0, size), payloadFor(NET_PARTS[2]));
});

describe("openContainerSource: verified says whether the source verified the whole file", () => {
  it("an HF session compares while the fetch layer streams the whole part", async () => {
    const caches = new MemoryCacheStorage();
    const { loaded, mock } = await openRemote(caches);
    const remote = openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
    });
    assertEquals(remote.verified, true);
  });

  it("a local directory does not compare sha256 (ADR 0086 decision 2)", async () => {
    const { loaded } = await openLocal();
    assertEquals(openContainerSource(loaded, containerOf(loaded, "net")).verified, false);
  });

  it("for a cross-repository reference **the referenced** source decides (local session + HF reference)", async () => {
    // The combination that would be false for code that looks only at the session's source (local).
    const caches = new MemoryCacheStorage();
    const mock = createMockFetch({ files: serveAll() });
    const { loaded } = await openLocal({
      crossRepo: {
        [FOREIGN_REPO]: createHfSource({
          repo: FOREIGN_REPO,
          hubUrl: HUB_URL,
          revision: FOREIGN_SHA,
        }),
      },
    });
    const source = openContainerSource(loaded, containerOf(loaded, "shared"), {
      fetch: mock.fetch,
      caches,
    });

    assertEquals(
      source.verified,
      true,
      "the referenced source's verified claim is not carried over",
    );
    // The file comes from the referenced repository too (the session's directory has no shared
    // parts).
    assertEquals(await source.read(1, 0, source.partLength(1)), payloadFor(CROSS_PARTS[1]));
    assertEquals(countCalls(mock.calls, foreignUrl(CROSS_PARTS[1])), 1);
  });
});

test("openContainerSource: the scan type reads the whole part once however many times the same part is read", async () => {
  // An adapter without range reads (`readFileRange`) = the path that falls back to whole reads.
  // Reading the whole part again for every block of the same part read in order would make
  // skipping quadratic (ADR 0109 decision 7).
  const { loaded, directory } = await openLocal();
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  directory.reads.length = 0;

  const payload = payloadFor(NET_PARTS[2]);
  for (const [offset, length] of [
    [0, 4],
    [4, 4],
    [8, 4],
  ]) {
    assertEquals(
      await source.read(2, offset, length),
      new Uint8Array(payload.subarray(offset, offset + length)),
    );
  }
  assertEquals(directory.reads, [NET_PARTS[2]], "the same part is read again");

  // Moving to another part releases the earlier whole read (the holding slot holds one part).
  await source.read(0, 0, 4);
  await source.read(2, 0, 4);
  assertEquals(directory.reads, [NET_PARTS[2], NET_PARTS[0], NET_PARTS[2]]);
});

test("openContainerSource: the scan type folds parallel reads of the same part into one whole read", async () => {
  // Putting only settled bytes into the slot would let every parallel read enter readFile, and for
  // that moment "part length × count" would be alive at once (breaking the RAM contract of one
  // part in the holding slot — container-v1 §11).
  const { loaded, directory } = await openLocal();
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  directory.reads.length = 0;

  const payload = payloadFor(NET_PARTS[2]);
  const ranges: readonly (readonly [number, number])[] = [
    [0, 4],
    [4, 4],
    [8, 4],
  ];
  const bytes = await Promise.all(ranges.map(([offset, length]) => source.read(2, offset, length)));

  ranges.forEach(([offset, length], index) => {
    assertEquals(bytes[index], new Uint8Array(payload.subarray(offset, offset + length)));
  });
  assertEquals(directory.reads, [NET_PARTS[2]], "parallel reads stack whole reads");
});

test("openContainerSource: the HF scan path does not hit the fetch layer again for the second range", async () => {
  // Deno's HF read path has cost "scan" (the fetch layer's strategy "stream"), so this is the path
  // used in practice. Counting only network calls would stay green even for an implementation
  // without the holding slot (both are 0 when warm), so the number of cache reads is observed.
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  await warmParts(loaded, "net", { fetch: mock.fetch, caches });
  const cache = hubCache(caches);

  const matchesFor = async (ranges: readonly (readonly [number, number])[]): Promise<number> => {
    const source = openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
    });
    let matches = 0;
    cache.onMatch = () => (matches += 1);
    try {
      for (const [offset, length] of ranges) await source.read(2, offset, length);
    } finally {
      cache.onMatch = undefined;
    }
    return matches;
  };

  const once = await matchesFor([[0, 4]]);
  const thrice = await matchesFor([
    [0, 4],
    [4, 4],
    [8, 4],
  ]);
  assertEquals(
    thrice,
    once,
    "the second and later ranges of the same part cause reads in the fetch layer",
  );
});

test("openContainerSource: the seek type passes only the requested range down to the adapter", async () => {
  // An adapter with positional reads = the path that reads per block (does not fall back to whole
  // reads).
  const base = memoryDirectory(localFiles());
  const ranges: [string, number, number][] = [];
  const files = localFiles();
  const adapter: DirectoryAdapter = {
    ...base.adapter,
    readFileRange: (path, offset, length) => {
      ranges.push([path, offset, length]);
      const bytes = files.get(path);
      if (bytes === undefined) return Promise.reject(new Error(`test-directory: ${path}`));
      return Promise.resolve(new Uint8Array(bytes.subarray(offset, offset + length)));
    },
  };
  const loaded = await loadManifest(localDirectory(adapter, { label: "./models/test" }), {
    caches: new MemoryCacheStorage(),
  });
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  base.reads.length = 0;

  const payload = payloadFor(NET_PARTS[2]);
  assertEquals(await source.read(2, 4, 3), new Uint8Array(payload.subarray(4, 7)));
  assertEquals(await source.read(2, 0, 2), new Uint8Array(payload.subarray(0, 2)));
  assertEquals(ranges, [
    [NET_PARTS[2], 4, 3],
    [NET_PARTS[2], 0, 2],
  ]);
  assertEquals(base.reads, [], "range reads fell back to whole reads");
});

test("openContainerSource: a cross-repository container is read from the referenced source", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);

  const source = openContainerSource(loaded, containerOf(loaded, "shared"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(
    await source.read(1, 0, source.partLength(1)),
    payloadFor(CROSS_PARTS[1]),
    "the referenced file could not be read",
  );
  assertEquals(
    countCalls(mock.calls, foreignUrl(CROSS_PARTS[1])),
    1,
    "not fetched from the referenced repository",
  );
  assertEquals(
    countCalls(mock.calls, resolveUrl(CROSS_PARTS[1])),
    0,
    "fetches from the session's repo (the file is not there)",
  );
  assertEquals(
    source.verified,
    true,
    "the verified claim is not carried over for the referenced repository either",
  );
});

test("openContainerSource: a cross-repository reference without a mapping fails with HubFetchError on open", async () => {
  // If the same missing setting looked different per surface (a plain Error / a silent success),
  // the caller could not get from the diagnostic to "pass crossRepo". It names the same structured
  // fields as the whole-file surface fetchAssets.
  const { loaded } = await openLocal();

  const error = assertThrows(
    () => openContainerSource(loaded, containerOf(loaded, "shared")),
    HubFetchError,
  );
  assertEquals(error.path, CROSS_PARTS[0]);
  assertEquals(error.repo, FOREIGN_REPO);
  assertEquals(error.revisionSha, FOREIGN_SHA);
  assert(error.cause instanceof Error, "the reason for the missing setting is not kept as cause");
  assert(
    error.cause.message.includes("crossRepo"),
    `${error.cause.message} does not name the missing setting`,
  );
});

test("openContainerSource: a read failure is the source's plain Error (not wrapped in a fetch context)", async () => {
  // This surface is not a fetch, so it is treated like `openAsset` (wrapping covers only the source
  // resolution; reads pass through).
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  served.delete(NET_FETCHED[1]);
  const { loaded, mock } = await openRemote(caches, served);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  const error = await assertRejects(() => source.read(2, 0, 4), Error);
  assert(!(error instanceof HubFetchError), `${error.name} is wrapped in a fetch context`);
});

test("openContainerSource: a part whose reader could not be opened is opened again on the next read", async () => {
  // Through HF on Deno, the first read of each part opens a range reader (with a re-warm).
  // Remembering that failure as the path decision would make a part that could not be opened
  // because of a transient 404 or disconnect keep returning the same rejection on later reads, and
  // the Session build would fail permanently (the MUST of planFor in container.ts).
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  const path = NET_PARTS[2];
  served.delete(path);
  const { loaded, mock } = await openRemote(caches, served);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  await assertRejects(() => source.read(2, 0, 4), Error);
  // Once the source is back (the transient failure cleared), the read succeeds on the same surface.
  served.set(path, payloadFor(path));
  assertEquals(
    await source.read(2, 0, 4),
    new Uint8Array(payloadFor(path).subarray(0, 4)),
    "a part that failed to open once is not opened again",
  );
});

test("openContainerSource: the scan type reads again a part whose whole read failed", async () => {
  // Keeping a failed whole read in the holding slot would make a part that failed on a transient
  // read error keep returning the same rejection on later reads (the MUST of readWhole in
  // container.ts). An adapter without range reads = the whole-read path.
  const base = memoryDirectory(localFiles());
  const path = NET_PARTS[2];
  let failures = 1;
  const adapter: DirectoryAdapter = {
    readFile: (target, options) => {
      if (target === path && failures > 0) {
        failures -= 1;
        return Promise.reject(
          new Error(`test-directory: the read of ${target} was cut off temporarily`),
        );
      }
      return base.adapter.readFile(target, options);
    },
  };
  const loaded = await loadManifest(localDirectory(adapter, { label: "./models/test" }), {
    caches: new MemoryCacheStorage(),
  });
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  base.reads.length = 0;

  await assertRejects(() => source.read(2, 0, 4), Error, "cut off temporarily");
  assertEquals(
    await source.read(2, 0, 4),
    new Uint8Array(payloadFor(path).subarray(0, 4)),
    "the failed whole read remains in the holding slot",
  );
  assertEquals(base.reads, [path], "the second read did not redo the whole read");
});

test("openContainerSource: returns no reader for an already aborted signal", async () => {
  // It has no effect on reads after opening (the consumer holds the reader for its lifetime —
  // `source.ts` ⑧), but the surface boundary checks it explicitly.
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const controller = new AbortController();
  const reason = new Error("app: the load was cancelled");
  controller.abort(reason);

  const error = assertThrows(() =>
    openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
    }),
  );
  assertStrictEquals(error, reason, "the abort is wrapped in another error");
  assertEquals(mock.calls, [], "goes to the network although already aborted");
});
