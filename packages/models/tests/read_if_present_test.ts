// 「在れば読む」読み口（`helpers/read-if-present.ts`）の挙動テスト。GPU も実資産も要らない。
//
// この読み口は資産系テストの SKIP 判定の源なので、「無い」と「読めない」を取り違えると
// 実行されていない検証が SKIP として静かに緑になる。NotFound だけを `undefined` に畳み、
// それ以外（ここではディレクトリを指す path）は投げることを縛る。
//
// NOTE: ディレクトリ読込の例外型は OS で違う（`Deno.errors.IsADirectory` が無い環境もある）
// ので、型ではなく「`undefined` を返さずに投げた」ことで観測する。

import { assertEquals, assertRejects } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "@std/testing/bdd";
import { readFileIfPresent, readTextIfPresent } from "./helpers/read-if-present.ts";

describe("readTextIfPresent / readFileIfPresent", () => {
  let root: URL;
  beforeAll(async () => {
    const dir = await Deno.makeTempDir({ prefix: "karume-read-if-present-" });
    root = new URL(`file://${dir}/`);
    await Deno.writeTextFile(new URL("present.txt", root), "中身");
    await Deno.mkdir(new URL("directory/", root));
  });
  afterAll(async () => {
    await Deno.remove(root, { recursive: true });
  });

  describe("path が存在しないとき", () => {
    it("テキスト読みは undefined を返す", async () => {
      assertEquals(await readTextIfPresent(new URL("absent.txt", root)), undefined);
    });
    it("バイト読みは undefined を返す", async () => {
      assertEquals(await readFileIfPresent(new URL("absent.bin", root)), undefined);
    });
  });

  describe("path がディレクトリを指すとき（NotFound 以外の失敗）", () => {
    it("テキスト読みは undefined へ畳まずに投げる", async () => {
      await assertRejects(() => readTextIfPresent(new URL("directory", root)));
    });
    it("バイト読みは undefined へ畳まずに投げる", async () => {
      await assertRejects(() => readFileIfPresent(new URL("directory", root)));
    });
  });

  describe("path にファイルが在るとき", () => {
    it("テキスト読みは中身を返す", async () => {
      assertEquals(await readTextIfPresent(new URL("present.txt", root)), "中身");
    });
    it("バイト読みは中身のバイト列を返す", async () => {
      assertEquals(
        await readFileIfPresent(new URL("present.txt", root)),
        new TextEncoder().encode("中身"),
      );
    });
  });
});
