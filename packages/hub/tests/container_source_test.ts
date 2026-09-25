/**
 * コンテナ 1 本の取得面（`openContainerSource` — ADR 0109 決定 7）。
 *
 * 押さえるのは 9 つ:
 *  ① 開くだけでは**何も取りに行かない**（温めは呼び手が `prefetchAssets` で先に済ませる）。
 *  ② 温めずに開いても読める（HF 取得元が初回の読みで 1 度だけ温め直す）。温め済みなら network 0。
 *  ③ `read` は区間ちょうどを返し（scan 型は保持枠の part の器の view — 写さない・次の part を
 *     読んでも前の view は化けない）、範囲外 / 長さ 0 の part / 未知の添字は fail loudly。
 *  ④ `verified` は**取得元が全量を検証したか**（HF = true / ローカルディレクトリ = false・
 *     越境は越境先の取得元で決まる）。
 *  ⑤ scan 型は同じ part を何度読んでも**全量読みが 1 回**（並行に読んでも 1 回）。
 *  ⑥ seek 型は要求した区間だけをアダプターへ降ろす。
 *  ⑦ 越境した容器は越境先の取得元から読む。未 mapping は開くときに `HubFetchError`。
 *  ⑧ 読みの失敗は取得元の素の `Error`（この面は取得ではない）。
 *  ⑨ 失敗は覚えない — 読み口を開けなかった part も、全量読みに失敗した part も、次の読みで
 *     取り直す。
 */

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
 * 長さ 0 の part 1 を持つ容器（ADR 0109 決定 3 の形）。連番は書き手の綴り（1 始まり —
 * `tools/exporter/src/karume/container.py` の `container_paths`）。
 */
const NET_PARTS = [
  "net/model-00001-of-00003.krm",
  "net/model-00002-of-00003.krm",
  "net/model-00003-of-00003.krm",
];
const EMPTY_INDEX = 1;
/** 取得も読みも起きる part（長さ 0 を除いた 2 本）。 */
const NET_FETCHED = NET_PARTS.filter((_path, index) => index !== EMPTY_INDEX);

const CROSS_PARTS = [
  "shared/model-00001-of-00002.krm",
  "shared/model-00002-of-00002.krm",
];

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

/** `descriptor` は part 0 の長さ（= ヘッダ + 2 文書）から導く（container-v1 §8）。 */
const containerFor = async (
  paths: readonly string[],
  options: { readonly emptyIndex?: number; readonly cross?: boolean } = {},
) => {
  const parts = await Promise.all(paths.map(async (path, index) =>
    index === options.emptyIndex
      ? {
        path,
        size: 0,
        sha256: EMPTY_SHA,
        ...(options.cross === true ? { repo: FOREIGN_REPO, revision: FOREIGN_SHA } : {}),
      }
      : await partRef(path, options.cross === true)
  ));
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

/** HF 取得元が配信するファイル表（越境ぶんは修飾キーで登録する）。 */
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

/** ディレクトリの中身（長さ 0 の part も 0 バイトのファイルとして置く — container-v1 §8）。 */
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
  const loaded = await loadManifest({ repo: REPO, hubUrl: HUB_URL, revision: SHA }, {
    fetch: loader.fetch,
    caches,
  });
  // manifest の取得を数えないよう、以降は記録の空な mock を渡す。
  return { loaded, mock: createMockFetch({ files }) };
};

const containerOf = (loaded: LoadedManifest, name: string) =>
  resolveSelection(loaded.manifest).containers[name];

/** 相 1（温め）は**呼び手の責務** — この面は温めないので、先に通す側をテストも明示的に呼ぶ。 */
const warmParts = async (
  loaded: LoadedManifest,
  name: string,
  options: { readonly fetch: typeof globalThis.fetch; readonly caches: CacheStorage },
): Promise<void> => {
  await prefetchAssets(loaded, containerOf(loaded, name).parts.filter((ref) => ref.size > 0), {
    fetch: options.fetch,
    caches: options.caches,
  });
};

/** ローカルディレクトリのセッション（越境先だけ差し替えられる）。 */
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

Deno.test("openContainerSource: 開くだけでは 1 バイトも取りに行かない", async () => {
  // 温めは呼び手の順序（descriptor で admission → prefetchAssets → open）に属する。面の中で
  // 全 part を温めると、実行できないモデルの重みが先に落ちる（ADR 0108 決定 19 / 0070 決定 5）。
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);

  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(mock.calls, [], "開いただけで network に出ている");
  assertEquals(source.partCount, NET_PARTS.length);
});

Deno.test("openContainerSource: 温めていなければ初回の読みが触れた part だけを温める", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(await source.read(0, 0, 4), payloadFor(NET_PARTS[0]).subarray(0, 4));
  assertEquals(countCalls(mock.calls, resolveUrl(NET_PARTS[0])), 1, "温め直しが 1 回でない");
  // 触っていない part は 1 バイトも取りに行かない（面が全 part を温めないことの観測点）。
  for (const path of [NET_PARTS[1], NET_PARTS[2]]) {
    assertEquals(countCalls(mock.calls, resolveUrl(path)), 0, `${path} を先回りで取っている`);
  }

  const warmed = mock.calls.length;
  await source.read(0, 4, 4);
  assertEquals(mock.calls.length, warmed, "同じ part の 2 度目の読みが network を起こしている");
});

Deno.test("openContainerSource: 先に温めてあれば読みは 1 度も network に出ない", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  await warmParts(loaded, "net", { fetch: mock.fetch, caches });

  const second = createMockFetch({ files: serveAll() });
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: second.fetch,
    caches,
  });
  for (const index of [0, 2]) await source.read(index, 0, source.partLength(index));
  assertEquals(second.calls, [], "温め済みなのに network へ出ている");
});

Deno.test("openContainerSource: partCount / partLength は宣言をそのまま答える", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  // 長さ 0 の part も**本数には数える**（添字が part の id なので、詰めると後続がずれる）。
  assertEquals(source.partCount, NET_PARTS.length);
  assertEquals(source.partLength(EMPTY_INDEX), 0);
  assertEquals(source.partLength(0), payloadFor(NET_PARTS[0]).byteLength);
  assertEquals(source.partLength(2), payloadFor(NET_PARTS[2]).byteLength);
});

Deno.test("openContainerSource: scan 型の read は保持枠の part の器を写さずに view で返す", async () => {
  // 写すと part ごとに器と写しで part 長を 2 重に持つ（container-v1 §11）。消費側は byteOffset を
  // 尊重して読む（整列は器が tight + block 開始の 64 B 整列で成り立つ — container.ts の MUST）。
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const payload = payloadFor(NET_PARTS[2]);
  const middle = Math.floor(payload.byteLength / 2);

  const views: Uint8Array<ArrayBuffer>[] = [];
  for (const [offset, length] of [[0, 4], [middle, 3], [payload.byteLength - 5, 5], [2, 0]]) {
    const bytes = await source.read(2, offset, length);
    assertEquals(
      bytes,
      new Uint8Array(payload.subarray(offset, offset + length)),
      `[${offset}, ${offset + length}) の中身が化けている`,
    );
    // 区間の位置は byteOffset に、part 全体は buffer に出る（写しなら byteOffset 0・buffer = 区間長）。
    assertEquals(bytes.byteOffset, offset, `[${offset}, ${offset + length}) の位置がずれている`);
    assertEquals(bytes.buffer.byteLength, source.partLength(2), "buffer が part の器でない");
    views.push(bytes);
  }
  // 同じ part の区間はどれも同じ器を指す（区間ごとに写していない）。
  for (const view of views.slice(1)) {
    assertStrictEquals(
      view.buffer,
      views[0].buffer,
      "同じ part の区間が別の buffer に写されている",
    );
  }
});

Deno.test("openContainerSource: scan 型は次の part を読んでも前の part の view が化けない", async () => {
  // 保持枠が別の part へ移っても、手放した器は上書きしない。器を part 間で使い回す実装
  // （全量読みの受け皿を 1 本だけ確保して次の part を読み込む形）を入れると、呼び手がまだ握って
  // いる view の中身が黙って次の part に化ける — この門で赤にする。
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

  // 別の part を読んで保持枠を移し、さらに元の part を読み直す（器の確保が 2 度起きる）。
  const other = await source.read(0, 0, lengthOf(0));
  const again = await source.read(2, 0, lengthOf(2));

  assertEquals(earlier, snapshot, "手放した part の view の中身が書き換わっている");
  assertEquals(other, payloadFor(NET_PARTS[0]));
  assertEquals(again, snapshot);
});

Deno.test("openContainerSource: 範囲外・長さ 0 の part・未知の添字は fail loudly", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });
  const size = source.partLength(2);

  for (const [offset, length] of [[size - 1, 2], [size, 1], [-1, 2], [0, 1.5], [0, size + 1]]) {
    const error = await assertRejects(() => source.read(2, offset, length), Error);
    assert(error.message.includes(String(size)), `${error.message} が宣言 size を名乗っていない`);
  }

  // 長さ 0 の part は読めない（読み手は partLength が 0 を答えた時点で読みに行かない）。
  const empty = await assertRejects(() => source.read(EMPTY_INDEX, 0, 0), Error);
  assert(empty.message.includes("長さ 0"), `${empty.message} が長さ 0 の part を名乗っていない`);

  for (const index of [NET_PARTS.length, -1, 1.5]) {
    const error = await assertRejects(() => source.read(index, 0, 1), Error);
    assert(error.message.includes(`part ${index}`), `${error.message} が添字を名乗っていない`);
  }
  // 同期の面も同じ規則で落ちる。
  for (const index of [NET_PARTS.length, -1]) {
    try {
      source.partLength(index);
      throw new Error(`partLength(${index}) が通ってしまった`);
    } catch (error) {
      assert(
        error instanceof Error && error.message.includes(`part ${index}`),
        `${error} が添字を名乗っていない`,
      );
    }
  }
  // 宣言だけで閉じる検査なので、1 本も取りに行かないまま落ちている。
  assertEquals(mock.calls, [], "不正な区間の読みが network に出ている");
});

Deno.test("openContainerSource: verified は取得元が全量を検証したかを答える", async (t) => {
  await t.step("HF セッションは取得層が part 全量を流しながら照合する", async () => {
    const caches = new MemoryCacheStorage();
    const { loaded, mock } = await openRemote(caches);
    const remote = openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
    });
    assertEquals(remote.verified, true);
  });

  await t.step("ローカルディレクトリは sha256 を照合しない（ADR 0086 決定 2）", async () => {
    const { loaded } = await openLocal();
    assertEquals(openContainerSource(loaded, containerOf(loaded, "net")).verified, false);
  });

  await t.step("越境は**越境先**の取得元で決まる（ローカルセッション + HF 越境）", async () => {
    // セッションの取得元（ローカル）だけを見る綴りなら false になる組み合わせ。
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

    assertEquals(source.verified, true, "越境先の検証済みの名乗りが継がれていない");
    // 実体も越境先から来る（セッションのディレクトリに shared の part は置いていない）。
    assertEquals(await source.read(1, 0, source.partLength(1)), payloadFor(CROSS_PARTS[1]));
    assertEquals(countCalls(mock.calls, foreignUrl(CROSS_PARTS[1])), 1);
  });
});

Deno.test("openContainerSource: scan 型は同じ part を何度読んでも全量読みが 1 回", async () => {
  // 区間読み（`readFileRange`）を持たないアダプター = 全量読みへ倒れる経路。同じ part の block を
  // 順に読むたびに全量を読み直すと、読み飛ばしが二次になる（ADR 0109 決定 7）。
  const { loaded, directory } = await openLocal();
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  directory.reads.length = 0;

  const payload = payloadFor(NET_PARTS[2]);
  for (const [offset, length] of [[0, 4], [4, 4], [8, 4]]) {
    assertEquals(
      await source.read(2, offset, length),
      new Uint8Array(payload.subarray(offset, offset + length)),
    );
  }
  assertEquals(directory.reads, [NET_PARTS[2]], "同じ part を読み直している");

  // 別の part へ移ると前の全量は手放される（保持枠は 1 part ぶん）。
  await source.read(0, 0, 4);
  await source.read(2, 0, 4);
  assertEquals(directory.reads, [NET_PARTS[2], NET_PARTS[0], NET_PARTS[2]]);
});

Deno.test("openContainerSource: scan 型は同じ part への並行 read も全量読み 1 回に畳む", async () => {
  // 決着したバイト列だけを枠に置くと、並行 read が全員 readFile へ入り、その瞬間だけ
  // 「part 長 × 本数」が同時に生きる（保持枠 1 part ぶんの RAM 契約 — container-v1 §11 — が破れる）。
  const { loaded, directory } = await openLocal();
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  directory.reads.length = 0;

  const payload = payloadFor(NET_PARTS[2]);
  const ranges: readonly (readonly [number, number])[] = [[0, 4], [4, 4], [8, 4]];
  const bytes = await Promise.all(ranges.map(([offset, length]) => source.read(2, offset, length)));

  ranges.forEach(([offset, length], index) => {
    assertEquals(bytes[index], new Uint8Array(payload.subarray(offset, offset + length)));
  });
  assertEquals(directory.reads, [NET_PARTS[2]], "並行 read が全量読みを重ねている");
});

Deno.test("openContainerSource: HF の scan 経路も 2 区間目で取得層を叩き直さない", async () => {
  // Deno の HF 読み口は cost "scan"（取得層の戦略 "stream"）なので、実運用の経路はこちら。
  // network 呼び出し数だけを数えると、保持枠を外した実装でも緑のまま通る（温め済みなら
  // どちらも 0 回）ので、キャッシュ読出しの回数で観測する。
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  await warmParts(loaded, "net", { fetch: mock.fetch, caches });
  const cache = hubCache(caches);

  const matchesFor = async (
    ranges: readonly (readonly [number, number])[],
  ): Promise<number> => {
    const source = openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
    });
    let matches = 0;
    cache.onMatch = () => matches += 1;
    try {
      for (const [offset, length] of ranges) await source.read(2, offset, length);
    } finally {
      cache.onMatch = undefined;
    }
    return matches;
  };

  const once = await matchesFor([[0, 4]]);
  const thrice = await matchesFor([[0, 4], [4, 4], [8, 4]]);
  assertEquals(thrice, once, "同じ part の 2 区間目以降が取得層の読出しを起こしている");
});

Deno.test("openContainerSource: seek 型は要求した区間だけをアダプターへ降ろす", async () => {
  // 位置読みを持つアダプター = block ごとに引く経路（全量読みへ降りない）。
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
  assertEquals(ranges, [[NET_PARTS[2], 4, 3], [NET_PARTS[2], 0, 2]]);
  assertEquals(base.reads, [], "区間読みが全量読みへ降りている");
});

Deno.test("openContainerSource: 越境した容器は越境先の取得元から読む", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);

  const source = openContainerSource(loaded, containerOf(loaded, "shared"), {
    fetch: mock.fetch,
    caches,
  });

  assertEquals(
    await source.read(1, 0, source.partLength(1)),
    payloadFor(CROSS_PARTS[1]),
    "越境先の実体が読めていない",
  );
  assertEquals(countCalls(mock.calls, foreignUrl(CROSS_PARTS[1])), 1, "越境先から取っていない");
  assertEquals(
    countCalls(mock.calls, resolveUrl(CROSS_PARTS[1])),
    0,
    "セッションの repo へ取りに行っている（そこには無い）",
  );
  assertEquals(source.verified, true, "越境先でも検証済みの名乗りが継がれていない");
});

Deno.test("openContainerSource: 未 mapping の越境は開くときに HubFetchError で落ちる", async () => {
  // 同じ設定不足が面ごとに別の見え方（素の Error / 無言の成功）になると、呼び手は診断から
  // 「crossRepo を渡せ」へ辿れない。全量面 fetchAssets と同じ構造化欄を名乗る。
  const { loaded } = await openLocal();

  const error = assertThrows(
    () => openContainerSource(loaded, containerOf(loaded, "shared")),
    HubFetchError,
  );
  assertEquals(error.path, CROSS_PARTS[0]);
  assertEquals(error.repo, FOREIGN_REPO);
  assertEquals(error.revisionSha, FOREIGN_SHA);
  assert(error.cause instanceof Error, "設定不足の理由を cause に残していない");
  assert(
    error.cause.message.includes("crossRepo"),
    `${error.cause.message} が足りない設定を名乗っていない`,
  );
});

Deno.test("openContainerSource: 読みの失敗は取得元の素の Error（取得の文脈に包まない）", async () => {
  // この面は取得ではないので `openAsset` と同じ扱い（包むのは取得元の解決までで、読みは素通し）。
  const caches = new MemoryCacheStorage();
  const served = serveAll();
  served.delete(NET_FETCHED[1]);
  const { loaded, mock } = await openRemote(caches, served);
  const source = openContainerSource(loaded, containerOf(loaded, "net"), {
    fetch: mock.fetch,
    caches,
  });

  const error = await assertRejects(() => source.read(2, 0, 4), Error);
  assert(!(error instanceof HubFetchError), `${error.name} が取得の文脈に包まれている`);
});

Deno.test("openContainerSource: 読み口を開けなかった part も次の読みで開き直す", async () => {
  // Deno の HF 経由は、part ごとの初回の読みで区間読み口を開く（温め直しつき）。その失敗を
  // 経路の決定として覚えると、一過性の 404・切断で開けなかった part が以後の読みで同じ reject を
  // 返し続け、Session 構築が恒久的に失敗する（container.ts の planFor の MUST）。
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
  // 取得元が戻った（一過性の失敗が解けた）後の読みは、同じ面のまま成功する。
  served.set(path, payloadFor(path));
  assertEquals(
    await source.read(2, 0, 4),
    new Uint8Array(payloadFor(path).subarray(0, 4)),
    "1 度開けなかった part を開き直していない",
  );
});

Deno.test("openContainerSource: scan 型は全量読みに失敗した part を次の読みで読み直す", async () => {
  // 失敗した全量読みを保持枠に残すと、一過性の読み失敗で落ちた part が以後の読みで同じ reject を
  // 返し続ける（container.ts の readWhole の MUST）。区間読みを持たないアダプター = 全量読みの経路。
  const base = memoryDirectory(localFiles());
  const path = NET_PARTS[2];
  let failures = 1;
  const adapter: DirectoryAdapter = {
    readFile: (target, options) => {
      if (target === path && failures > 0) {
        failures -= 1;
        return Promise.reject(new Error(`test-directory: ${target} の読みが一時的に切れた`));
      }
      return base.adapter.readFile(target, options);
    },
  };
  const loaded = await loadManifest(localDirectory(adapter, { label: "./models/test" }), {
    caches: new MemoryCacheStorage(),
  });
  const source = openContainerSource(loaded, containerOf(loaded, "net"));
  base.reads.length = 0;

  await assertRejects(() => source.read(2, 0, 4), Error, "一時的に切れた");
  assertEquals(
    await source.read(2, 0, 4),
    new Uint8Array(payloadFor(path).subarray(0, 4)),
    "失敗した全量読みが保持枠に残っている",
  );
  assertEquals(base.reads, [path], "2 度目の読みが全量読みをやり直していない");
});

Deno.test("openContainerSource: 中断済みの signal では口を返さない", async () => {
  // 開いた後の読みには効かない（読み口は消費側が寿命ぶん掴む — `source.ts` ⑧）が、面の境界では
  // 明示的に見る。
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await openRemote(caches);
  const controller = new AbortController();
  const reason = new Error("app: ロードを取り消した");
  controller.abort(reason);

  const error = assertThrows(() =>
    openContainerSource(loaded, containerOf(loaded, "net"), {
      fetch: mock.fetch,
      caches,
      signal: controller.signal,
    })
  );
  assertStrictEquals(error, reason, "中断が別のエラーに包まれている");
  assertEquals(mock.calls, [], "中断済みなのに network へ出ている");
});
