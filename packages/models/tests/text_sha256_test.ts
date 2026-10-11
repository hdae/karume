// 同期の SHA-256（`src/text/sha256.ts`）— Web 標準の `crypto.subtle.digest` と全ビット一致すること。
//
// パディングの境界（1 ブロックに長さが収まる最後の 55 バイト・収まらない 56 バイト・ブロック境界の
// 63 / 64 / 65 バイト・2 ブロック目の境界）を全て踏む。

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { sha256HexSync } from "../src/text/sha256.ts";

const subtleHex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

describe("sha256HexSync", () => {
  it("FIPS 180-4 の例 'abc' と空のバイト列の digest を返す", () => {
    const encoder = new TextEncoder();
    assertEquals(
      sha256HexSync(encoder.encode("abc")),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    assertEquals(
      sha256HexSync(new Uint8Array(0)),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("パディングの境界の全ての長さで crypto.subtle.digest と一致する", async () => {
    for (const length of [1, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000, 70_000]) {
      // 値は長さごとに変える（全ゼロだとバイトの取り違えが digest に出にくい）。
      const bytes = new Uint8Array(length).map((_, index) => (index * 131 + length) & 0xff);
      assertEquals(sha256HexSync(bytes), await subtleHex(bytes), `${length} バイト`);
    }
  });
});
