/**
 * `@karume/hub/deno` の実 fs テスト。メモリ上のアダプター（`local_test.ts`）では踏めないのは
 * 「ランタイムの読み口そのもの」— path の継ぎ方（末尾 `/` の有無・`file:` URL）と、`Deno` が
 * 投げるエラーを**実体のパスを名乗ったまま**共通層へ渡せているか。
 *
 * 取得元としての意味論（世代・検証・越境・進捗）は本体側（`local_test.ts`）の担当なので、
 * ここでは重複させずに end-to-end が 1 本通ることと、ランタイムの読み口そのものに閉じた性質を見る。
 *
 * 後者のうち「中断 signal が `Deno` の読みまで透過するか」は公開面（`fetchAssets`）から踏めない
 * （中断が送出ループの手前で捕まる）ので、内部の `driverOf` で取得元ドライバを直に叩く
 * — `inventory_test.ts` が取得元の内部挙動を見るときと同じ流儀。
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { fetchAssets, HubFetchError, IntegrityError, loadManifest, openAsset } from "../mod.ts";
import { denoDirectory } from "../deno.ts";
import { driverOf } from "../src/source.ts";
import { buildLocalDist, FETCHED_PART_PATHS, PART_KEYS, TOKENIZER_PATH } from "./helpers/local.ts";
import { payloadFor } from "./helpers/mock.ts";
import { selectionFiles } from "./helpers/selection.ts";

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
        const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest));
        PART_KEYS.forEach((key, index) => {
          assertEquals(assets[key], payloadFor(FETCHED_PART_PATHS[index]));
        });
        assertEquals(assets["tokenizer"], payloadFor(TOKENIZER_PATH));
      });
    }
  });
});

Deno.test("denoDirectory: file: URL の root でも同じ配布形を指す", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(new URL(`file://${root}`)));
    const assets = await fetchAssets(loaded, selectionFiles(loaded.manifest));
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
    await Deno.remove(`${root}/${FETCHED_PART_PATHS[1]}`);

    const error = await assertRejects(
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      HubFetchError,
    );
    assertEquals(error.path, FETCHED_PART_PATHS[1]);
    assert(error.message.includes(root), `${error.message} がディレクトリを名乗っていない`);
    assert(error.cause instanceof Error, "Deno のエラーを cause に残していない");
    assert(
      error.cause.message.includes(`${root}/${FETCHED_PART_PATHS[1]}`),
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
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      IntegrityError,
    );
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.source, "local");
    assertEquals(error.actual, "3");
  });
});

// ---- 全量面の size 門（実 fs 側）。宣言より**長い**実体は上の 3 本が踏んでいない形で、
// `Deno.readFile` が実体を丸ごと返した後に取得元（`sources/local.ts`）が実長を名乗って落とす。

Deno.test("denoDirectory: 宣言より長い実体も size 門で落ちる（実長を名乗る）", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    // 別 quant の取り違え・書きかけのコピーの形（manifest の size / sha256 は元のまま）。
    await Deno.writeFile(`${root}/${TOKENIZER_PATH}`, new Uint8Array(ref.size + 1));

    const error = await assertRejects(
      () => fetchAssets(loaded, selectionFiles(loaded.manifest)),
      IntegrityError,
    );
    assertEquals(error.path, TOKENIZER_PATH);
    assertEquals(error.actual, String(ref.size + 1));
    assertEquals(error.source, "local");
  });
});

Deno.test("denoDirectory: 全量読みは abort 済み signal で 1 バイトも返さない", async () => {
  await withDistribution(async (root) => {
    const source = denoDirectory(root);
    const loaded = await loadManifest(source);
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const controller = new AbortController();
    const reason = new Error("test: 呼び出し側の中断");
    controller.abort(reason);

    // 取得元（`sources/local.ts` ③）を**直接**叩く。共通層（`fetchAssets`）を通すと、中断は
    // 送出ループの手前にあるアドミッション席の取得で捕まり、実 fs の読みまで降りない
    // （= `Deno.readFile` へ signal が透過するかを見られない）。
    const pinned = driverOf(source).pin("", {});
    const error = await assertRejects(
      () =>
        pinned.readFile(ref, {
          signal: controller.signal,
          // ローカル取得元は契約上 `onProgress` を 1 度も呼ばない（`source.ts` の
          // `FileReadOptions`）ので、ここに中断の見張りを置いても恒真になる — 必須欄を
          // 埋めるだけの no-op。中断が size 違反に化けていないことは `sizeViolation` が見る。
          onProgress: () => {},
          sizeViolation: (actual) => new Error(`size 門が反応した（actual ${actual}）`),
        }),
      Error,
    );
    // 中断が取得失敗にも size 違反にも化けない — 呼び手が渡した reason がそのまま上がる。
    assertStrictEquals(error, reason);
  });
});

// ---- 区間読み（`openAsset` — `source.ts` ⑧）。`Deno.open` → `seek` → 読みループの実装は実 fs
// 側のここにしかないので、fake アダプター（`local_test.ts`）では代替できない。

Deno.test("denoDirectory: 区間読みは実体の同区間と一致する（先頭 / 中央 / 末尾ちょうど）", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const whole = await Deno.readFile(`${root}/${ref.path}`);
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "位置読みを持つアダプターなのに読み口が開かない");
    // ディレクトリの実体は位置読みなので offset に依らない費用。
    assertEquals(reader.cost, "seek");

    const middle = Math.floor(ref.size / 2);
    for (const [offset, length] of [[0, 4], [middle, 3], [ref.size - 5, 5]]) {
      const bytes = await reader.read(offset, length);
      assertEquals(
        bytes,
        new Uint8Array(whole.subarray(offset, offset + length)),
        `[${offset}, ${offset + length}) の中身が化けている`,
      );
      // 消費側はそのまま TypedArray として読むので、buffer 全体を占めている必要がある。
      assertEquals(bytes.byteOffset, 0);
      assertEquals(bytes.buffer.byteLength, length);
    }
  });
});

Deno.test("denoDirectory: 長さ 0 の区間は空のバイト列（呼び手の誤りにしない）", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "読み口が開かない");
    // 0 バイト要求は「読むものが無い」であって誤りではない（末尾ちょうどの位置も含めて許す）。
    for (const offset of [0, ref.size]) {
      assertEquals((await reader.read(offset, 0)).byteLength, 0);
    }
  });
});

Deno.test("denoDirectory: 宣言 size の外はアダプターへ降ろす前に落ちる", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "読み口が開かない");

    for (const [offset, length] of [[ref.size - 1, 2], [ref.size, 1], [-1, 2], [0, 1.5]]) {
      const error = await assertRejects(() => reader.read(offset, length), Error);
      assert(error.message.includes(ref.path), `${error.message} が path を名乗っていない`);
      assert(
        error.message.includes(String(ref.size)),
        `${error.message} が宣言 size を名乗っていない`,
      );
    }
  });
});

Deno.test("denoDirectory: 実体が宣言より短ければ埋まらない区間で落ちる（0 埋めを返さない）", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "読み口が開かない");
    // 途中で切れたコピー（manifest の size はそのまま = 全量読みの size 門と同じ形）。
    await Deno.writeFile(`${root}/${ref.path}`, payloadFor(ref.path).subarray(0, 3));

    const error = await assertRejects(() => reader.read(0, ref.size), Error);
    assert(
      error.message.includes(`${root}/${ref.path}`),
      `${error.message} が読めなかった実体を名乗っていない`,
    );
  });
});

Deno.test("denoDirectory: 区間読みは abort 済み signal で 1 バイトも返さない", async () => {
  await withDistribution(async (root) => {
    const loaded = await loadManifest(denoDirectory(root));
    const ref = selectionFiles(loaded.manifest)["tokenizer"];
    const reader = await openAsset(loaded, ref);
    assert(reader !== undefined, "読み口が開かない");
    const controller = new AbortController();
    const reason = new Error("test: 呼び出し側の中断");
    controller.abort(reason);

    // 長さ 0 も同じ（読みループが 1 度も回らない形でも、中断は実体に触れる前に見る）。
    for (const length of [4, 0]) {
      // 中断は取得失敗に化けない（呼び手が渡した reason がそのまま上がる）。
      assertStrictEquals(
        await assertRejects(() => reader.read(0, length, { signal: controller.signal }), Error),
        reason,
      );
    }
  });
});
