// 共通 image 層（RGBA → PNG）の挙動テスト。GPU も資産も要らない純関数。
//
// 主張は 3 つ: ①CRC-32 が既知ベクタと合う ②チャンク構造が PNG 仕様どおり ③IDAT を展開すると
// 元の画素列に**行ずれなく**戻る（filter バイトの挿入漏れは長さ検査では捕まらない）。
// 圧縮結果のバイト列そのものは固定しない（zlib のバージョンで変わるため — 代わりに往復一致が
// 中身を保証する）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { crc32, encodePng } from "../src/image/png.ts";

Deno.test("crc32: 既知ベクタ（CRC-32/ISO-HDLC の check 値と IEND チャンク）", () => {
  const ascii = (text: string) => Uint8Array.from(text, (c) => c.charCodeAt(0));
  // CRC-32/ISO-HDLC の標準 check 値。
  assertEquals(crc32(ascii("123456789")), 0xcbf43926);
  // PNG の IEND チャンク（型 4 バイトのみ・データ長 0）の CRC は仕様上いつもこの値。
  assertEquals(crc32(ascii("IEND")), 0xae426082);
  assertEquals(crc32(new Uint8Array(0)), 0);
});

/** PNG のチャンクを頭から辿る（長さ・型・CRC を全部見る）。 */
const walkChunks = (
  png: Uint8Array,
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

Deno.test("encodePng: シグネチャ + IHDR/IDAT/IEND の順で、全チャンクの CRC が合う", async () => {
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
  assertEquals(chunks.map((chunk) => chunk.type), ["IHDR", "IDAT", "IEND"]);
  for (const chunk of chunks) assert(chunk.crcOk, `${chunk.type} の CRC が合わない`);

  const ihdr = new DataView(
    chunks[0].data.buffer,
    chunks[0].data.byteOffset,
    chunks[0].data.byteLength,
  );
  assertEquals(ihdr.getUint32(0), width, "IHDR の幅");
  assertEquals(ihdr.getUint32(4), height, "IHDR の高さ");
  assertEquals(
    [...chunks[0].data.subarray(8)],
    [8, 2, 0, 0, 0],
    "深度8 / truecolor / 非インタレース",
  );
  assertEquals(chunks[2].data.length, 0, "IEND はデータを持たない");
});

Deno.test("encodePng: IDAT を DecompressionStream で戻すと元の画素列に一致する", async () => {
  const width = 5;
  const height = 4;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    // 行ずれ（filter バイトの挿入漏れ）が起きたら必ず変わるよう、隣接画素を全て別値にする。
    rgba[index * 4] = (index * 37) % 256;
    rgba[index * 4 + 1] = (index * 91 + 7) % 256;
    rgba[index * 4 + 2] = (index * 13 + 200) % 256;
    rgba[index * 4 + 3] = 255;
  }
  const png = await encodePng(rgba, width, height);
  const idat = walkChunks(png).find((chunk) => chunk.type === "IDAT");
  assert(idat !== undefined, "IDAT が無い");

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
  assertEquals(raw.length, height * stride, "展開後の長さ（filter バイト込み）");
  for (let y = 0; y < height; y += 1) {
    assertEquals(raw[y * stride], 0, `行 ${y} の filter type（None）`);
    for (let x = 0; x < width; x += 1) {
      const source = (y * width + x) * 4;
      const target = y * stride + 1 + x * 3;
      assertEquals(
        [raw[target], raw[target + 1], raw[target + 2]],
        [rgba[source], rgba[source + 1], rgba[source + 2]],
        `画素 (${x}, ${y})`,
      );
    }
  }
});

Deno.test("encodePng: サイズが正の整数でなければ落とす（長さ検査より先）", async () => {
  // `imageToRgba` 側には同型の門のテストがある（`anima_latents_test.ts`）ので、こちら側だけが
  // 裸だった。0 / 負 / 端数は行ストライドの計算が破綻するので、長さが合っていても通さない。
  for (const bad of [0, -1, 1.5]) {
    await assertRejects(
      () => encodePng(new Uint8ClampedArray(4), bad, 1),
      RangeError,
      "正の整数でない",
    );
    await assertRejects(
      () => encodePng(new Uint8ClampedArray(4), 1, bad),
      RangeError,
      "正の整数でない",
    );
  }
  // 門の順序: `width × height × 4` が偶然 RGBA 長と一致する組でも、長さ検査より先に
  // RangeError で落ちる（後段だと「長さは合っているのに壊れた PNG」を作る経路になる）。
  await assertRejects(
    () => encodePng(new Uint8ClampedArray(4), 2, 0.5),
    RangeError,
    "正の整数でない",
  );
});

Deno.test("encodePng: 不透明でない画素は落とす（黙って不透明化しない）", async () => {
  const rgba = new Uint8ClampedArray([1, 2, 3, 254]);
  await assertRejects(() => encodePng(rgba, 1, 1), Error, "アルファ");
  await assertRejects(() => encodePng(new Uint8ClampedArray(3), 1, 1), Error, "RGBA の長さ");
});

Deno.test("encodePng: deflate が効いている（一様画像が生バイト列より十分小さい）", async () => {
  const width = 64;
  const height = 64;
  const rgba = new Uint8ClampedArray(width * height * 4).fill(255);
  const png = await encodePng(rgba, width, height);
  assert(
    png.length < width * height * 3 / 10,
    `PNG ${png.length}B が無圧縮 ${width * height * 3}B の 1/10 未満でない`,
  );
});

Deno.test("encodePng: 同じ入力からバイト単位で同じ PNG が出る（決定性）", async () => {
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
  // 生成画像の同一性を sha256 で門にする運用（ADR 0037 の P3 門）は、この決定性の上に立つ。
  const digest = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
    Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
  assertEquals(await digest(first), await digest(second));
});
