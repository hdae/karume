// デモの `--source` / `--source-map` → 取得元の写像の門（GPU も実資産も要らない）。
//
// 見るのは 2 つ: ①ローカルの配布形と HF リポジトリ名の分岐 ②綴りの崩れを黙って捨てないこと
// （効かないノブが静かに残ると、取り違えた取得元から焼いた出力が「モデルの揺れ」に見える）。

import { assertEquals, assertRejects } from "@std/assert";
import { distributionSource } from "./local-source.ts";
import { MANIFEST_FILE } from "./local-assets.ts";

/** 一時ディレクトリを 1 つ作って渡す（終わったら消す）。 */
const withDir = async (fn: (dir: string) => Promise<void>): Promise<void> => {
  const dir = await Deno.makeTempDir({ prefix: "karume-local-source-test-" });
  try {
    await fn(dir);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
};

/** `karume.json` を置くだけ（`isLocalDist` は存在しか見ない）。 */
const markAsDist = async (dir: string): Promise<void> => {
  await Deno.writeTextFile(`${dir}/${MANIFEST_FILE}`, "{}");
};

Deno.test("distributionSource: karume.json を持つディレクトリは取得元ハンドルになる", async () => {
  await withDir(async (dir) => {
    await markAsDist(dir);
    const source = await distributionSource(dir);
    assertEquals(typeof source === "string", false);
  });
});

Deno.test("distributionSource: karume.json が無ければ HF リポジトリ名としてそのまま返す", async () => {
  await withDir(async (dir) => {
    assertEquals(await distributionSource(`${dir}/nothing-here`), `${dir}/nothing-here`);
  });
});

Deno.test("distributionSource: ローカル配布形 + mapping 1 件は通る", async () => {
  await withDir(async (dir) => {
    await markAsDist(dir);
    const source = await distributionSource(dir, [`owner/name=${dir}`]);
    assertEquals(typeof source === "string", false);
  });
});

Deno.test("distributionSource: HF リポジトリ名に mapping を渡すと落ちる（効かないノブを残さない）", async () => {
  await withDir(async (dir) => {
    await assertRejects(
      () => distributionSource(`${dir}/nothing-here`, ["owner/name=/tmp/x"]),
      Error,
      "--source-map はローカルの配布形にだけ効く",
    );
  });
});

Deno.test("distributionSource: mapping の綴りが owner/name=<パス> でなければ落ちる", async () => {
  await withDir(async (dir) => {
    await markAsDist(dir);
    for (const spelled of ["ownername", "=path", "owner/name="]) {
      await assertRejects(
        () => distributionSource(dir, [spelled]),
        Error,
        "owner/name=<パス> の形でない",
      );
    }
  });
});

Deno.test("distributionSource: 同じ repo を 2 度名指しすると落ちる", async () => {
  await withDir(async (dir) => {
    await markAsDist(dir);
    await assertRejects(
      () => distributionSource(dir, ["owner/name=/tmp/a", "owner/name=/tmp/b"]),
      Error,
      "repo owner/name が重複",
    );
  });
});
