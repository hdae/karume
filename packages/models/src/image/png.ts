/**
 * RGBA bytes → PNG (8-bit truecolor RGB, non-interlaced).
 *
 * **Pipeline-independent shared processing** (image generation models generally write a PNG at the
 * end). Runtime dependencies are **Web standard APIs only** (ADR 0002); compression is left to
 * `CompressionStream("deflate")`.
 *
 * MUST: `"deflate"`, not `"deflate-raw"`. Compression Streams' `"deflate"` emits **the RFC1950 zlib
 * wrapper** (a 2-byte header + a trailing Adler-32), which is exactly the form PNG's IDAT requires.
 * With `"deflate-raw"` the zlib header is missing and decoders fail with "invalid compression
 * method".
 *
 * MUST: insert the filter type byte `0x00` (None) at the start of each scanline before compressing.
 * Forgetting it shifts each row by 1 byte, and **the image opens but the colours smear
 * diagonally** (a length check on the `Uint8Array` does not catch it).
 *
 * CRC32 is not in the Web standard, so it is implemented here (a 256-entry table, the polynomial of
 * the PNG specification's Annex D).
 */

import { ModelInputError } from "../errors.ts";

/**
 * PNG signature (8 bytes).
 *
 * MUST: keep it an array literal and build the `Uint8Array` inside {@link encodePng} (do not run
 * `Uint8Array.from` at module scope — the same cross-cutting invariant as {@link buildCrcTable}).
 */
const SIGNATURE: readonly number[] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** 8 bits / channel. */
const BIT_DEPTH = 8;
/** color type 2 = truecolor (RGB, no alpha). */
const COLOR_TYPE_RGB = 2;
const COMPRESSION_DEFLATE = 0;
const FILTER_ADAPTIVE = 0;
const INTERLACE_NONE = 0;
/** The filter type at the start of a scanline (None). */
const FILTER_NONE = 0;

/**
 * Builds the CRC-32 table.
 *
 * MUST: not held in a module-scope `const` (the cross-cutting invariant "zero side effects in every
 * module" = no import-time execution and no global mutable state — the condition for tree-shaking
 * through the barrel). It is rebuilt on every call, but one PNG makes only 3 chunk calls, and
 * building the table is 2,048 integer operations = negligible next to the CRC itself over an IDAT
 * of several MB.
 */
const buildCrcTable = (): Uint32Array<ArrayBuffer> => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let bit = 0; bit < 8; bit += 1) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
};

/** CRC-32 of a PNG chunk (ISO 3309 / ITU-T V.42 — inverted initial value, inverted output). */
export const crc32 = (bytes: Uint8Array<ArrayBuffer>): number => {
  const table = buildCrcTable();
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = table[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
};

/** One chunk (length 4 + type 4 + data + CRC 4). The CRC covers the range **including the type**. */
const chunk = (type: string, data: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let index = 0; index < 4; index += 1) out[4 + index] = type.charCodeAt(index);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
};

/**
 * Compresses into zlib (RFC1950) form.
 * NOTE: `Uint8Array<ArrayBuffer>` is required to reject `SharedArrayBuffer`-backed input
 * (`CompressionStream`'s `BufferSource` does not accept shared buffers).
 */
const deflate = async (bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> => {
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const compressed = source.pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(compressed).arrayBuffer());
};

/**
 * Turns RGBA (4 bytes / pixel, row-major) into PNG bytes.
 *
 * Alpha is **dropped** (written as truecolor RGB). Generated images are opaque, and giving the PNG an
 * alpha plane would only add a quarter of dead capacity holding the constant 255.
 * MUST: check that it is 255 before dropping it (do not silently make translucency opaque).
 */
export const encodePng = async (
  rgba: Uint8ClampedArray<ArrayBufferLike> | Uint8Array<ArrayBufferLike>,
  width: number,
  height: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  // NOTE: all 3 arguments belong to the caller (this function is only exported from the barrel and
  // is not called from inside a pipeline). The only remedy is "fix the size / bytes you pass", hence
  // {@link ModelInputError} (ADR 0107 decision 2).
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new ModelInputError(`画像サイズ ${width}×${height} が正の整数でない`);
  }
  if (rgba.length !== width * height * 4) {
    throw new ModelInputError(`RGBA の長さ ${rgba.length} が 4×${width}×${height} と違う`);
  }

  const stride = 1 + width * 3;
  const raw = new Uint8Array(height * stride);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride;
    raw[rowStart] = FILTER_NONE;
    for (let x = 0; x < width; x += 1) {
      const source = (y * width + x) * 4;
      if (rgba[source + 3] !== 255) {
        // Alpha is also the content of the `rgba` the caller passed, so it is input-caused like the
        // size and the length (the only remedy is "make it 255" — ADR 0107 decision 2).
        throw new ModelInputError(
          `画素 (${x}, ${y}) のアルファが ${rgba[source + 3]}（255 でない）`,
        );
      }
      const target = rowStart + 1 + x * 3;
      raw[target] = rgba[source];
      raw[target + 1] = rgba[source + 1];
      raw[target + 2] = rgba[source + 2];
    }
  }

  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, width);
  ihdrView.setUint32(4, height);
  ihdr[8] = BIT_DEPTH;
  ihdr[9] = COLOR_TYPE_RGB;
  ihdr[10] = COMPRESSION_DEFLATE;
  ihdr[11] = FILTER_ADAPTIVE;
  ihdr[12] = INTERLACE_NONE;

  const parts = [
    Uint8Array.from(SIGNATURE),
    chunk("IHDR", ihdr),
    chunk("IDAT", await deflate(raw)),
    chunk("IEND", new Uint8Array(0)),
  ];
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const png = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    png.set(part, at);
    at += part.length;
  }
  return png;
};
