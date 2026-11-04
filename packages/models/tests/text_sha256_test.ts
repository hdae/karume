// The synchronous SHA-256 (`src/text/sha256.ts`) — must agree bit for bit with the Web-standard
// `crypto.subtle.digest`.
//
// Exercises every padding boundary (55 bytes, the last length that fits the length into one block;
// 56 bytes, which does not; 63 / 64 / 65 bytes around the block boundary; and the second block's
// boundary).

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { sha256HexSync } from "../src/text/sha256.ts";

const subtleHex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

describe("sha256HexSync", () => {
  it("returns the digests of the FIPS 180-4 example 'abc' and of empty bytes", () => {
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

  it("matches crypto.subtle.digest at every length around the padding boundaries", async () => {
    for (const length of [1, 55, 56, 57, 63, 64, 65, 119, 120, 128, 1000, 70_000]) {
      // The values vary with the length (with all zeros a byte mix-up hardly shows in the digest).
      const bytes = new Uint8Array(length).map((_, index) => (index * 131 + length) & 0xff);
      assertEquals(sha256HexSync(bytes), await subtleHex(bytes), `${length} bytes`);
    }
  });
});
