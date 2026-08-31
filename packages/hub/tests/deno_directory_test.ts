/**
 * `@karume/hub/deno` の実 fs テスト。メモリ上のアダプター（`local_test.ts`）では踏めないのは
 * 「ランタイムの読み口そのもの」— path の継ぎ方（末尾 `/` の有無・`file:` URL）と、`Deno` が
 * 投げるエラーを**実体のパスを名乗ったまま**共通層へ渡せているか。
 *
 * 取得元としての意味論（世代・検証・越境・進捗）は本体側（`local_test.ts`）の担当なので、
 * ここでは重複させずに end-to-end が 1 本通ることだけを見る。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { fetchAssets, HubFetchError, IntegrityError, loadManifest, resolveFiles } from "../mod.ts";
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
