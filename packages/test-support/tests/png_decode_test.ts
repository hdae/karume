// Checks that the test-only PNG decoder (`@karume/test-support/png-decode`) **fails loudly outside
// the accepted range**.
//
// That decoder declares "it accepts only the one form of 8-bit / truecolor RGB(2) /
// non-interlaced / filter type None on every scanline, and throws on anything else" (so that
// real-image e2e cannot go green while an unsupported path is silently approximated). Yet its only
// users (the 3 real-image e2e tests and verify-diff) pass only well-formed PNGs written by
// `encodePng`, so **not one rejection path was being executed**.
//
// MUST: the fixtures are not built by importing `encodePng` from models (the dependency would point
// the wrong way). This file keeps a minimal encoder on `CompressionStream("deflate")`, puts the
// correct form in one place, and each failure case breaks exactly one point (the same discipline as
// helpers/safetensors.ts).
// NOTE: the decoder does not verify chunk CRCs, so 0 is written.

import { test } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import { decodePng } from "@karume/test-support/png-decode";

const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Compresses to zlib (RFC1950) — inverse of the decoder's `DecompressionStream("deflate")`. */
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

/** One `[u32 length][type 4][data][crc 4]` chunk (crc fixed at 0). */
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

/**
 * Known pixels that differ per position (a row / column / channel mix-up always shows in the
 * values).
 */
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

/** The 13 bytes of IHDR (fields are overridden per case). */
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

/** Raw scanlines with the filter type of every row set to `filter`. */
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
  /** Splits IDAT into 2 chunks (a form the PNG spec allows — the concatenation path). */
  readonly splitIdat?: boolean;
};

const buildPng = async (options: PngOptions = {}): Promise<Uint8Array<ArrayBuffer>> => {
  const compressed = await deflate(scanlines(options.filter ?? 0));
  const idat = options.splitIdat
    ? [chunk("IDAT", compressed.subarray(0, 3)), chunk("IDAT", compressed.subarray(3))]
    : [chunk("IDAT", compressed)];
  return concat([
    SIGNATURE,
    ...(options.omitIhdr ? [] : [chunk("IHDR", ihdrBytes(options))]),
    ...(options.omitIdat ? [] : idat),
    chunk("IEND", new Uint8Array(0)),
  ]);
};

test("decodePng: reads the one form encodePng writes (8-bit RGB / non-interlaced / filter None)", async () => {
  const decoded = await decodePng(await buildPng(), "well-formed");
  assertEquals(decoded.width, WIDTH);
  assertEquals(decoded.height, HEIGHT);
  assertEquals([...decoded.data], [...PIXELS]);
});

test("decodePng: concatenates IDAT split over several chunks and returns the same pixels", async () => {
  const decoded = await decodePng(await buildPng({ splitIdat: true }), "multiple IDAT");
  assertEquals([...decoded.data], [...PIXELS]);
});

test("decodePng: rejects bytes that are not a PNG by the signature", async () => {
  const broken = await buildPng();
  broken[3] = 0x00;
  await assertRejects(() => decodePng(broken, "broken signature"), Error, "not a PNG signature");
});

test("decodePng: rejects an IHDR it does not accept (color type 6 / interlace 1)", async () => {
  const rgba = await buildPng({ colorType: 6 });
  await assertRejects(
    () => decodePng(rgba, "RGBA"),
    Error,
    "IHDR has [depth,color,compression,filter,interlace]",
  );
  const interlaced = await buildPng({ interlace: 1 });
  await assertRejects(
    () => decodePng(interlaced, "Adam7"),
    Error,
    "IHDR has [depth,color,compression,filter,interlace]",
  );
});

test("decodePng: rejects a row filter other than None (does not silently approximate)", async () => {
  const subFiltered = await buildPng({ filter: 1 });
  const error = await assertRejects(
    () => decodePng(subFiltered, "Sub filter"),
    Error,
    "filter type of row 0 is 1",
  );
  assertEquals(error.message.includes("None(0)"), true, error.message);
});

test("decodePng: rejects a missing IHDR / IDAT, each with its own attribution", async () => {
  const noIhdr = await buildPng({ omitIhdr: true });
  await assertRejects(() => decodePng(noIhdr, "missing IHDR"), Error, "no IHDR");
  const noIdat = await buildPng({ omitIdat: true });
  await assertRejects(() => decodePng(noIdat, "missing IDAT"), Error, "no IDAT");
});

test("decodePng: rejects a chunk whose declared length runs past the end of the file", async () => {
  const bytes = await buildPng();
  // IDAT comes right after signature 8 + IHDR chunk 25. Overwrite its length field with a huge
  // value.
  new DataView(bytes.buffer).setUint32(8 + 25, 0xffff);
  await assertRejects(
    () => decodePng(bytes, "length overrun"),
    Error,
    "runs past the end of the file",
  );
});

test("decodePng: rejects a byte count after inflating that disagrees with the declared height", async () => {
  // Only IHDR's height is +1 (there are pixels for only 3 rows)
  const tallerHeader = await buildPng({ height: HEIGHT + 1 });
  await assertRejects(
    () => decodePng(tallerHeader, "height mismatch"),
    Error,
    "bytes after inflating",
  );
});
