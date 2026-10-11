// A/B の道具の書き出し先の判定（`render.ts` の `resolveOutputDirectory` — GPU 不要・資産不要）。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { resolveOutputDirectory } from "./render.ts";

const ROOT = "/work/karume";

describe("resolveOutputDirectory", () => {
  it("リポの根の outputs/ の下なら絶対パスにして返す（cwd からの相対も受ける）", () => {
    assertEquals(
      resolveOutputDirectory("outputs/birefnet-ab/run", ROOT, ROOT),
      "/work/karume/outputs/birefnet-ab/run",
    );
    assertEquals(
      resolveOutputDirectory("../outputs/x", `${ROOT}/tools`, `${ROOT}/`),
      "/work/karume/outputs/x",
    );
    assertEquals(
      resolveOutputDirectory("/work/karume/outputs/./a//b", "/elsewhere", ROOT),
      "/work/karume/outputs/a/b",
    );
  });

  it("outputs/ の外は落とす（.. で抜ける・outputs/ そのもの・接頭辞だけ同じ名前）", () => {
    for (const out of ["outputs/../docs", "outputs", "outputs-old/x", "/tmp/ab", "docs/ab"]) {
      assertThrows(() => resolveOutputDirectory(out, ROOT, ROOT), Error, "の下でない", out);
    }
  });

  it("cwd がリポの外なら、相対の outputs/ も外として落とす（根は cwd でなく main.ts から導く）", () => {
    assertThrows(
      () => resolveOutputDirectory("outputs/ab", "/home/someone", ROOT),
      Error,
      "/work/karume/outputs/ の下でない",
    );
  });
});
