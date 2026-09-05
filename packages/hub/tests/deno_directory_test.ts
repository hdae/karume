/**
 * `@karume/hub/deno` の実 fs テスト。メモリ上のアダプター（`local_test.ts`）では踏めないのは
 * 「ランタイムの読み口そのもの」— path の継ぎ方（末尾 `/` の有無・`file:` URL）と、`Deno` が
 * 投げるエラーを**実体のパスを名乗ったまま**共通層へ渡せているか。
 *
 * 取得元としての意味論（世代・検証・越境・進捗）は本体側（`local_test.ts`）の担当なので、
 * ここでは重複させずに end-to-end が 1 本通ることだけを見る。
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import {
  fetchAssets,
  type FileRef,
  HubFetchError,
  IntegrityError,
  type LoadedManifest,
  loadManifest,
  resolveFiles,
  streamAssets,
  type StreamedAsset,
} from "../mod.ts";
import { denoDirectory } from "../deno.ts";
import { buildLocalDist, SHARD_PATHS, TOKENIZER_PATH } from "./helpers/local.ts";
import { payloadFor } from "./helpers/mock.ts";

/** 合成した配布形を実ディレクトリへ書き出す（呼び手が消す）。 */
const materialize = async (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): Promise<string> => {
  const root = await Deno.makeTempDir({ prefix: "karume-hub-local-" });
  for (const [path, bytes] of files) {
    const at = `${root}/${path}`;
    await Deno.mkdir(at.slice(0, at.lastIndexOf("/")), { recursive: true });
    await Deno.writeFile(at, bytes);
  }
  return root;
};

const withDistribution = async (
  body: (root: string, files: Map<string, Uint8Array<ArrayBuffer>>) => Promise<void>,
  mutate: (files: Map<string, Uint8Array<ArrayBuffer>>) => void = () => {},
): Promise<void> => {
  const dist = await buildLocalDist();
  mutate(dist.files);
  const root = await materialize(dist.files);
  try {
    await body(root, dist.files);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

/** 逐次面の入力（この配布形は path が全て別なので resolveFiles の出力をそのまま渡せる）。 */
const streamRefs = (loaded: LoadedManifest): FileRef[] => {
  const files = resolveFiles(loaded.manifest);
  return Object.keys(files).map((key) => files[key]);
};

/** 到着した id だけを集める（bytes は次の反復で器が上書きするので溜めない）。 */
const drain = async (
  stream: AsyncGenerator<StreamedAsset, void, unknown>,
): Promise<string[]> => {
  const ids: string[] = [];
  for await (const asset of stream) ids.push(asset.id);
  return ids;
};

Deno.test("denoDirectory: 実ディレクトリの配布形が末尾 / の有無に関わらず読める", async (t) => {
  await withDistribution(async (root) => {
    for (const spelling of [root, `${root}/`]) {
      await t.step(`root = ${spelling === root ? "末尾なし" : "末尾あり"}`, async () => {
        const loaded = await loadManifest(denoDirectory(spelling));
        const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest));
        SHARD_PATHS.forEach((path, index) => {
          assertEquals(assets[`net[${index}]`], payloadFor(path));
        });
        assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
      });
    }
  });
});

Deno.test("denoDirectory: file: URL の root でも同じ配布形を指す", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(new URL(`file://${root}`)));
    const assets = await fetchAssets(loaded, resolveFiles(loaded.manifest));
    assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
  });
});

Deno.test("denoDirectory: 隣のディレクトリを読みに行かない（root の外へ出ない）", async () => {
  await withDistribution(async (root) => {
    // 末尾 `/` を落として継いだ場合に**ちょうど当たる**位置へ、読めない karume.json を置く
    // （`<root>` + `karume.json`）。継ぎ方が壊れればここが読まれて manifest ごと落ちる。
    const sibling = `${root}karume.json`;
    await Deno.writeFile(sibling, new TextEncoder().encode("{}"));
    try {
      const loaded = await loadManifest(denoDirectory(root));
      assertEquals(loaded.manifest.available.models, ["m"], "兄弟の karume.json を読んでいる");
    } finally {
      await Deno.remove(sibling);
    }
  });
});

Deno.test("denoDirectory: 欠損は実体のパスを名乗って fail loudly", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    // manifest を読んだ**後**に消える形（配布形の一部が欠けたディレクトリ）。
    await Deno.remove(`${root}/${SHARD_PATHS[1]}`);

    const error = await assertRejects(
      () => fetchAssets(loaded, resolveFiles(loaded.manifest)),
      HubFetchError,
    );
    assertEquals(error.path, SHARD_PATHS[1]);
    assert(error.message.includes(root), `${error.message} がディレクトリを名乗っていない`);
    assert(error.cause instanceof Error, "Deno のエラーを cause に残していない");
    assert(
      error.cause.message.includes(`${root}/${SHARD_PATHS[1]}`),
      `${error.cause.message} が読めなかった実体の絶対パスを名乗っていない`,
    );
  });
});

Deno.test("denoDirectory: manifest に無い karume.json（root ごと不在）も同じ経路で落ちる", async () => {
  const root = await Deno.makeTempDir({ prefix: "karume-hub-empty-" });
  try {
    const error = await assertRejects(() => loadManifest(denoDirectory(root)), HubFetchError);
    assertEquals(error.path, "karume.json");
    assertEquals(error.repo, undefined, "存在しない repo を名乗っている");
    assert(error.message.includes(root), `${error.message} がディレクトリを名乗っていない`);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("denoDirectory: 実体の size が manifest と食い違えば IntegrityError", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    // 途中で切れたコピー（size だけが狂う — sha256 は信頼するので、これが唯一の門）。
    const truncated = payloadFor(TOKENIZER_PATH).subarray(0, 3);
    await Deno.writeFile(`${root}/${TOKENIZER_PATH}`, truncated);

    const error = await assertRejects(
      () => fetchAssets(loaded, resolveFiles(loaded.manifest)),
      IntegrityError,
    );
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.source, "local");
    assertEquals(error.actual, "3");
  });
});

// ---- 逐次面（`streamAssets`）の器経路。`readFile` しか通らない上の面と違い、ここだけが
// `readFileInto`（`Deno.open` → `stat` → 分割読みループ → `close`）を踏む。ホスト RAM ピークの
// 係数 1 化（ADR 0070 追記）の実装は実 fs 側のこの 1 本しかないので、fake アダプターでは
// 代替できない。

Deno.test("denoDirectory: 逐次面は全 shard を 1 本の器の先頭へ読む", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const refs = streamRefs(loaded);
    const largest = refs.reduce((max, ref) => Math.max(max, ref.size), 0);

    const ids: string[] = [];
    const buffers = new Set<ArrayBufferLike>();
    for await (const asset of streamAssets(loaded, refs)) {
      ids.push(asset.id);
      buffers.add(asset.bytes.buffer);
      assertEquals(asset.bytes, payloadFor(asset.id), `${asset.id} の中身が化けている`);
      assertEquals(asset.bytes.byteOffset, 0);
      assertEquals(asset.bytes.buffer.byteLength, largest, "器が最大 shard 長で作られていない");
    }

    assertEquals(ids, refs.map((ref) => ref.path));
    // 器を使わずに shard ごとの buffer を返していれば、ここが shard 本数になる。
    assertEquals(buffers.size, 1, "shard ごとに別の buffer が確保されている");
  });
});

Deno.test("denoDirectory: 器へ読む経路でも途中で切れたコピーは IntegrityError", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const refs = streamRefs(loaded);
    await Deno.writeFile(`${root}/${TOKENIZER_PATH}`, payloadFor(TOKENIZER_PATH).subarray(0, 3));

    const error = await assertRejects(() => drain(streamAssets(loaded, refs)), IntegrityError);
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.actual, "3");
    assertEquals(error.source, "local");
  });
});

Deno.test("denoDirectory: 器に収まらない長いコピーは読まずに実長を名乗って落ちる", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const refs = streamRefs(loaded);
    const tokenizer = refs.find((ref) => ref.path === TOKENIZER_PATH);
    assert(tokenizer !== undefined, "配布形に tokenizer が無い");
    // 器（= 宣言 size ちょうどに切った target）に収まらない側。実長がそのまま `actual` に出るのが
    // 「1 バイトも読まずに stat の値だけ返した」証拠（読んでから数えると器から溢れる）。
    await Deno.writeFile(`${root}/${TOKENIZER_PATH}`, new Uint8Array(tokenizer.size + 1));

    const error = await assertRejects(() => drain(streamAssets(loaded, refs)), IntegrityError);
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.actual, String(tokenizer.size + 1));
    assertEquals(error.source, "local");
  });
});

Deno.test("denoDirectory: 逐次面の中断は reason を素通しし、以降の shard を渡さない", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const refs = streamRefs(loaded);
    assert(refs.length >= 3, "中断の観測に 3 本以上の配布形が要る");
    const controller = new AbortController();
    const reason = new Error("test: 呼び出し側の中断");
    const ids: string[] = [];

    const error = await assertRejects(async () => {
      for await (const asset of streamAssets(loaded, refs, { signal: controller.signal })) {
        ids.push(asset.id);
        controller.abort(reason);
      }
    }, Error);

    // 中断が取得失敗（HubFetchError）に化けない — 呼び手が渡した reason がそのまま上がる。
    assertStrictEquals(error, reason);
    assertEquals(ids, [refs[0].path], "中断後の shard が渡されている");
  });
});
