// Behaviour tests of the shared image layer (RGBA → PNG). Pure functions; no GPU and no assets.
//
// Three claims: ① CRC-32 matches known vectors ② the chunk structure follows the PNG spec ③
// inflating IDAT gives back the original pixels **without a row shift** (a missing filter byte is
// not caught by a length check). The compressed bytes themselves are not pinned (they change with
// the zlib version — the round trip guarantees the content instead).

import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import { crc32, encodePng } from "../src/image/png.ts";

test("crc32: known vectors (the CRC-32/ISO-HDLC check value and the IEND chunk)", () => {
  const ascii = (text: string) => Uint8Array.from(text, (c) => c.charCodeAt(0));
  // The standard check value of CRC-32/ISO-HDLC.
  assertEquals(crc32(ascii("123456789")), 0xcbf43926);
  // By the spec, the CRC of PNG's IEND chunk (4 type bytes only, data length 0) is always this
  // value.
  assertEquals(crc32(ascii("IEND")), 0xae426082);
  assertEquals(crc32(new Uint8Array(0)), 0);
});

/** Walks the PNG chunks from the start (checking length, type and CRC of each). */
const walkChunks = (
  png: Uint8Array<ArrayBuffer>,
): readonly { type: string; data: Uint8Array; crcOk: boolean }[] => {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const chunks: { type: string; data: Uint8Array; crcOk: boolean }[] = [];
  let at = 8;
  while (at < png.length) {
    const length = view.getUint32(at);
    const type = String.fromCharCode(...png.subarray(at + 4, at + 8));
    const data = png.subarray(at + 8, at + 8 + length);
    const stored = view.getUint32(at + 8 + length);
    chunks.push({ type, data, crcOk: crc32(png.subarray(at + 4, at + 8 + length)) === stored });
    at += 12 + length;
  }
  return chunks;
};

test("encodePng: signature + IHDR/IDAT/IEND in that order, and every chunk's CRC matches", async () => {
  const width = 3;
  const height = 2;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    rgba[index * 4] = index * 10;
    rgba[index * 4 + 1] = 255 - index * 10;
    rgba[index * 4 + 2] = index;
    rgba[index * 4 + 3] = 255;
  }
  const png = await encodePng(rgba, width, height);

  assertEquals([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const chunks = walkChunks(png);
  assertEquals(
    chunks.map((chunk) => chunk.type),
    ["IHDR", "IDAT", "IEND"],
  );
  for (const chunk of chunks) assert(chunk.crcOk, `${chunk.type} CRC does not match`);

  const ihdr = new DataView(
    chunks[0].data.buffer,
    chunks[0].data.byteOffset,
    chunks[0].data.byteLength,
  );
  assertEquals(ihdr.getUint32(0), width, "IHDR width");
  assertEquals(ihdr.getUint32(4), height, "IHDR height");
  assertEquals(
    [...chunks[0].data.subarray(8)],
    [8, 2, 0, 0, 0],
    "bit depth 8 / truecolor / non-interlaced",
  );
  assertEquals(chunks[2].data.length, 0, "IEND has no data");
});

test("encodePng: inflating IDAT with DecompressionStream gives back the original pixels", async () => {
  const width = 5;
  const height = 4;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    // Give every neighbouring pixel a different value so that a row shift (a missing filter byte)
    // always changes the result.
    rgba[index * 4] = (index * 37) % 256;
    rgba[index * 4 + 1] = (index * 91 + 7) % 256;
    rgba[index * 4 + 2] = (index * 13 + 200) % 256;
    rgba[index * 4 + 3] = 255;
  }
  const png = await encodePng(rgba, width, height);
  const idat = walkChunks(png).find((chunk) => chunk.type === "IDAT");
  assert(idat !== undefined, "no IDAT");

  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(idat.data.slice());
      controller.close();
    },
  });
  const raw = new Uint8Array(
    await new Response(source.pipeThrough(new DecompressionStream("deflate"))).arrayBuffer(),
  );

  const stride = 1 + width * 3;
  assertEquals(raw.length, height * stride, "inflated length (including filter bytes)");
  for (let y = 0; y < height; y += 1) {
    assertEquals(raw[y * stride], 0, `filter type of row ${y} (None)`);
    for (let x = 0; x < width; x += 1) {
      const source = (y * width + x) * 4;
      const target = y * stride + 1 + x * 3;
      assertEquals(
        [raw[target], raw[target + 1], raw[target + 2]],
        [rgba[source], rgba[source + 1], rgba[source + 2]],
        `pixel (${x}, ${y})`,
      );
    }
  }
});

test("encodePng: rejects a size that is not a positive integer (before the length check)", async () => {
  // `imageToRgba` has a test of the same check (`anima_latents_test.ts`), so only this side was
  // uncovered. 0, negative and fractional sizes break the row-stride calculation, so they are
  // rejected even when the length happens to match.
  for (const bad of [0, -1, 1.5]) {
    await assertRejects(
      () => encodePng(new Uint8ClampedArray(4), bad, 1),
      ModelInputError,
      "is not a positive integer",
    );
    await assertRejects(
      () => encodePng(new Uint8ClampedArray(4), 1, bad),
      ModelInputError,
      "is not a positive integer",
    );
  }
  // Check order: even for a pair whose `width × height × 4` happens to equal the RGBA length, the
  // size check fails before the length check (checked later, it would be a path that produces a
  // broken PNG whose length is correct).
  await assertRejects(
    () => encodePng(new Uint8ClampedArray(4), 2, 0.5),
    ModelInputError,
    "is not a positive integer",
  );
});

test("encodePng: rejects non-opaque pixels (does not silently make them opaque)", async () => {
  const rgba = new Uint8ClampedArray([1, 2, 3, 254]);
  await assertRejects(() => encodePng(rgba, 1, 1), ModelInputError, "Alpha");
  await assertRejects(
    () => encodePng(new Uint8ClampedArray(3), 1, 1),
    ModelInputError,
    "RGBA length",
  );
});

// ---- Classifying whether a failure is caused by the input (ADR 0107) --------

describe("when a host sorts encodePng failures into 400 / 500", () => {
  it("a bad size passed by the caller is caught as ModelInputError", async () => {
    await assertRejects(() => encodePng(new Uint8ClampedArray(4), 0, 1), ModelInputError);
  });

  it("a bad RGBA length passed by the caller is also caught as ModelInputError", async () => {
    await assertRejects(() => encodePng(new Uint8ClampedArray(3), 1, 1), ModelInputError);
  });

  it("bad RGBA content (alpha) passed by the caller is caught as the same type", async () => {
    // If the same function split its error types over the same argument, a host would have to
    // write separate branches for size and content.
    await assertRejects(
      () => encodePng(new Uint8ClampedArray([1, 2, 3, 254]), 1, 1),
      ModelInputError,
    );
  });
});

test("encodePng: deflate is effective (a uniform image is much smaller than the raw bytes)", async () => {
  const width = 64;
  const height = 64;
  const rgba = new Uint8ClampedArray(width * height * 4).fill(255);
  const png = await encodePng(rgba, width, height);
  assert(
    png.length < (width * height * 3) / 10,
    `PNG ${png.length}B is not under 1/10 of the uncompressed ${width * height * 3}B`,
  );
});

test("encodePng: the same input gives a byte-identical PNG (determinism)", async () => {
  const width = 7;
  const height = 5;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    rgba[index * 4] = (index * 53) % 256;
    rgba[index * 4 + 1] = (index * 17 + 3) % 256;
    rgba[index * 4 + 2] = (index * 199 + 41) % 256;
    rgba[index * 4 + 3] = 255;
  }
  const [first, second] = await Promise.all([
    encodePng(rgba, width, height),
    encodePng(rgba, width, height),
  ]);
  assertEquals(first, second);
  // Checking generated images for identity by sha256 (ADR 0037's P3 check) rests on this
  // determinism.
  const digest = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
    Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  assertEquals(await digest(first), await digest(second));
});
