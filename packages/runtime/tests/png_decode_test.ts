// テスト専用 PNG デコーダ（tests/helpers/png-decode.ts）の**受理域外 fail loudly** の門。
//
// あのデコーダは「8bit / truecolor RGB(2) / 非インタレース / 全スキャンラインの filter type が
// None の 1 形だけを受け、外れたら例外にする」と宣言している（対応していない経路を黙って
// 近似したまま実画像 e2e が緑になるのを防ぐため）。ところが唯一の利用者（models 側 3 本）は
// `encodePng` が書いた正常な PNG しか渡さないので、**拒否経路が 1 本も実行されていなかった**。
//
// MUST: フィクスチャは models の `encodePng` を import して作らない（依存方向が逆）。ここは
// `CompressionStream("deflate")` で最小の encoder を持ち、正しい形を 1 箇所に置いて異常系は
// 1 点だけ壊す（helpers/format.ts と同じ規律）。
// NOTE: チャンク CRC はデコーダが検証しないので 0 を書く。

import { assertEquals, assertRejects } from "@std/assert";
import { decodePng } from "./helpers/png-decode.ts";

const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** zlib（RFC1950）へ圧縮する — デコーダ側の `DecompressionStream("deflate")` の逆。 */
const deflate = async (bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> => {
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const packed = source.pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(packed).arrayBuffer());
};

/** `[u32 長][型 4][データ][crc 4]` の 1 チャンク（crc は 0 固定）。 */
const chunk = (type: string, data: Uint8Array): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(12 + data.length);
  new DataView(out.buffer).setUint32(0, data.length);
  for (let i = 0; i < 4; i += 1) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  return out;
};

const concat = (parts: readonly Uint8Array[]): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
};

const WIDTH = 4;
const HEIGHT = 3;

/** 位置ごとに違う既知画素（行 / 列 / チャネルの取り違えが必ず値に出る）。 */
const PIXELS: Uint8Array<ArrayBuffer> = (() => {
  const data = new Uint8Array(HEIGHT * WIDTH * 3);
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const at = (y * WIDTH + x) * 3;
      data[at] = 10 * y + x;
      data[at + 1] = 100 + 10 * y + x;
      data[at + 2] = 200 + 10 * y + x;
    }
  }
  return data;
})();

/** IHDR の 13 バイト（欄はケースごとに差し替える）。 */
const ihdrBytes = (
  overrides: {
    readonly height?: number;
    readonly colorType?: number;
    readonly interlace?: number;
  } = {},
): Uint8Array<ArrayBuffer> => {
  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, WIDTH);
  view.setUint32(4, overrides.height ?? HEIGHT);
  ihdr[8] = 8; // bit depth
  ihdr[9] = overrides.colorType ?? 2; // truecolor RGB
  ihdr[10] = 0; // compression = deflate
  ihdr[11] = 0; // filter method = adaptive
  ihdr[12] = overrides.interlace ?? 0;
  return ihdr;
};

/** 全行の filter type を `filter` にした生スキャンライン。 */
const scanlines = (filter: number): Uint8Array<ArrayBuffer> => {
  const stride = 1 + WIDTH * 3;
  const raw = new Uint8Array(HEIGHT * stride);
  for (let y = 0; y < HEIGHT; y += 1) {
    raw[y * stride] = filter;
    raw.set(PIXELS.subarray(y * WIDTH * 3, (y + 1) * WIDTH * 3), y * stride + 1);
  }
  return raw;
};

type PngOptions = {
  readonly filter?: number;
  readonly height?: number;
  readonly colorType?: number;
  readonly interlace?: number;
  readonly omitIhdr?: boolean;
  readonly omitIdat?: boolean;
  /** IDAT を 2 チャンクへ割る（PNG 仕様が許す形 — 連結経路）。 */
  readonly splitIdat?: boolean;
};

const buildPng = async (options: PngOptions = {}): Promise<Uint8Array<ArrayBuffer>> => {
  const compressed = await deflate(scanlines(options.filter ?? 0));
  const idat = options.splitIdat
    ? [
      chunk("IDAT", compressed.subarray(0, 3)),
      chunk("IDAT", compressed.subarray(3)),
    ]
    : [chunk("IDAT", compressed)];
  return concat([
    SIGNATURE,
    ...(options.omitIhdr ? [] : [chunk("IHDR", ihdrBytes(options))]),
    ...(options.omitIdat ? [] : idat),
    chunk("IEND", new Uint8Array(0)),
  ]);
};

Deno.test("decodePng: encodePng が書く 1 形（8bit RGB / 非インタレース / filter None）を読める", async () => {
  const decoded = await decodePng(await buildPng(), "正常系");
  assertEquals(decoded.width, WIDTH);
  assertEquals(decoded.height, HEIGHT);
  assertEquals([...decoded.data], [...PIXELS]);
});

Deno.test("decodePng: IDAT が複数チャンクに割れていても連結して同じ画素を返す", async () => {
  const decoded = await decodePng(await buildPng({ splitIdat: true }), "複数 IDAT");
  assertEquals([...decoded.data], [...PIXELS]);
});

Deno.test("decodePng: PNG でないバイト列をシグネチャで落とす", async () => {
  const broken = await buildPng();
  broken[3] = 0x00;
  await assertRejects(() => decodePng(broken, "壊れたシグネチャ"), Error, "PNG シグネチャでない");
});

Deno.test("decodePng: 受理しない IHDR（color type 6 / interlace 1）を落とす", async () => {
  const rgba = await buildPng({ colorType: 6 });
  await assertRejects(
    () => decodePng(rgba, "RGBA"),
    Error,
    "IHDR が [depth,color,compression,filter,interlace]",
  );
  const interlaced = await buildPng({ interlace: 1 });
  await assertRejects(
    () => decodePng(interlaced, "Adam7"),
    Error,
    "IHDR が [depth,color,compression,filter,interlace]",
  );
});

Deno.test("decodePng: None 以外の行 filter を落とす（黙って近似しない）", async () => {
  const subFiltered = await buildPng({ filter: 1 });
  const error = await assertRejects(
    () => decodePng(subFiltered, "Sub フィルタ"),
    Error,
    "filter type が 1",
  );
  assertEquals(error.message.includes("None(0)"), true, error.message);
});

Deno.test("decodePng: IHDR / IDAT の欠落をそれぞれの帰属で落とす", async () => {
  const noIhdr = await buildPng({ omitIhdr: true });
  await assertRejects(() => decodePng(noIhdr, "IHDR 欠落"), Error, "IHDR が無い");
  const noIdat = await buildPng({ omitIdat: true });
  await assertRejects(() => decodePng(noIdat, "IDAT 欠落"), Error, "IDAT が無い");
});

Deno.test("decodePng: チャンク宣言長がファイル末尾を越える形を落とす", async () => {
  const bytes = await buildPng();
  // IDAT は シグネチャ 8 + IHDR チャンク 25 の直後。その長さ欄を巨大な値へ書き換える。
  new DataView(bytes.buffer).setUint32(8 + 25, 0xffff);
  await assertRejects(() => decodePng(bytes, "長さ超過"), Error, "ファイル末尾を越える");
});

Deno.test("decodePng: 展開後のバイト数が宣言した高さと食い違う形を落とす", async () => {
  // IHDR の height だけを +1 する（画素は 3 行ぶんしか無い）
  const tallerHeader = await buildPng({ height: HEIGHT + 1 });
  await assertRejects(() => decodePng(tallerHeader, "高さ不一致"), Error, "展開後が");
});
