// examples/wan の台本の出力先の名前の画像の印（`output-label.ts`）。名前はファイルのバイト列の sha256 の先頭 8 桁と、
// 明示した `--fit` だけで決まる（README の Output 節 — 同じ中身なら同じ名前・fit を省いた回は名前に fit を入れない）。

import { assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { outputImageLabel } from "./output-label.ts";

/** "abc" の sha256 は ba7816bf…（FIPS 180-2 の例 — 実装と独立な期待値）。 */
const ABC = new TextEncoder().encode("abc");

describe("outputImageLabel（出力先の名前の画像の印）", () => {
  it("--fit を省いた回は -img-<sha256 の先頭 8 桁> だけ（既定の fit を名前に写さない）", async () => {
    assertEquals(await outputImageLabel(ABC, undefined), "-img-ba7816bf");
  });

  it("--fit を明示した回は印の後ろに -<fit> を足す", async () => {
    assertEquals(await outputImageLabel(ABC, "crop"), "-img-ba7816bf-crop");
    assertEquals(await outputImageLabel(ABC, "stretch"), "-img-ba7816bf-stretch");
  });

  it("名前は中身だけで決まる（同じバイト列は同じ印・1 バイト違えば別の印）", async () => {
    assertEquals(
      await outputImageLabel(new TextEncoder().encode("abc"), undefined),
      await outputImageLabel(ABC, undefined),
    );
    assertNotEquals(
      await outputImageLabel(new TextEncoder().encode("abd"), undefined),
      await outputImageLabel(ABC, undefined),
    );
  });
});
