/**
 * A **minimal PNG → RGB8 decoder (tests only)**.
 *
 * Karume itself has no decoder (the module doc of `packages/models/src/image/preprocess.ts` — the
 * entry point is "an RGB8 pixel array + width + height", by design without a PNG / JPEG decoder).
 * This is a tool only for letting **tests read real images** without changing that design, and it
 * lives at the subpath `./png-decode` of the private package `@karume/test-support` (it is not
 * shipped in a public package).
 *
 * ## Accepts only the form `encodePng` writes (anything else fails loudly)
 *
 * It reads only PNGs baked by `packages/models/src/image/png.ts` (the outputs of the image pipelines), so
 * it is not made a general decoder — it accepts only the one form **8-bit / truecolor RGB (color
 * type 2) / non-interlaced / filter type None on every scanline**, and throws on anything else.
 * Letting row filters such as Paeth, or palette / gray / alpha, "pass just in case" could keep an
 * unsupported path silently approximated while the real-image e2e goes green.
 *
 * NOTE: chunk CRCs are not verified (it is a tool against mix-ups, not tampering, and a broken
 * byte sequence fails in the zlib inflation or the length check). IDAT split across several chunks
 * is accepted by concatenating them (a form the PNG spec allows — `encodePng` writes only one, but
 * the concatenation adds no branch).
 */

/** Decode result (structurally compatible with `Rgb8Image` — row-major, 3 bytes per pixel). */
export type DecodedRgb8 = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** PNG signature (8 bytes). */
const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** The accepted IHDR (exactly the values `encodePng` writes). */
const BIT_DEPTH = 8;
/** color type 2 = truecolor (RGB, no alpha). */
const COLOR_TYPE_RGB = 2;
const COMPRESSION_DEFLATE = 0;
const FILTER_ADAPTIVE = 0;
const INTERLACE_NONE = 0;
/** The accepted filter type at the start of a scanline (None only). */
const FILTER_NONE = 0;

/** Channel count (RGB). */
const CHANNELS = 3;

/** Inflates zlib (RFC1950). The inverse of `encodePng`'s `CompressionStream("deflate")`. */
const inflate = async (bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> => {
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const expanded = source.pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(expanded).arrayBuffer());
};

/** Reads a chunk type (4 ASCII bytes). */
const chunkType = (bytes: Uint8Array<ArrayBuffer>, at: number): string =>
  String.fromCharCode(bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]);

/** Checks the 13 bytes of IHDR and returns `[width, height]`. */
const parseHeader = (data: Uint8Array<ArrayBuffer>, where: string): [number, number] => {
  if (data.length !== 13) throw new Error(`${where}: IHDR is ${data.length} bytes (13 required)`);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const width = view.getUint32(0);
  const height = view.getUint32(4);
  if (width <= 0 || height <= 0) throw new Error(`${where}: image size ${width}×${height}`);
  const declared = [data[8], data[9], data[10], data[11], data[12]];
  const expected = [
    BIT_DEPTH,
    COLOR_TYPE_RGB,
    COMPRESSION_DEFLATE,
    FILTER_ADAPTIVE,
    INTERLACE_NONE,
  ];
  if (declared.join(",") !== expected.join(",")) {
    throw new Error(
      `${where}: IHDR has [depth,color,compression,filter,interlace] = [${declared.join(",")}]` +
        ` (this decoder accepts only [${expected.join(",")}] — see the module doc)`,
    );
  }
  return [width, height];
};

/**
 * Expands PNG bytes into RGB8.
 *
 * @param bytes the contents of the PNG file
 * @param where the origin named on failure (a file name, for example)
 */
export const decodePng = async (
  bytes: Uint8Array<ArrayBuffer>,
  where: string,
): Promise<DecodedRgb8> => {
  for (let index = 0; index < SIGNATURE.length; index += 1) {
    if (bytes[index] !== SIGNATURE[index]) throw new Error(`${where}: not a PNG signature`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  let size: [number, number] | undefined;
  const parts: Uint8Array<ArrayBuffer>[] = [];
  let at = SIGNATURE.length;
  while (at + 8 <= bytes.length) {
    const length = view.getUint32(at);
    const type = chunkType(bytes, at + 4);
    const begin = at + 8;
    if (begin + length + 4 > bytes.length) {
      throw new Error(`${where}: chunk '${type}' runs past the end of the file`);
    }
    const data = bytes.slice(begin, begin + length);
    if (type === "IHDR") size = parseHeader(data, where);
    if (type === "IDAT") parts.push(data);
    if (type === "IEND") break;
    at = begin + length + 4;
  }
  if (size === undefined) throw new Error(`${where}: no IHDR`);
  if (parts.length === 0) throw new Error(`${where}: no IDAT`);
  const [width, height] = size;

  const compressed = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    compressed.set(part, offset);
    offset += part.length;
  }

  const raw = await inflate(compressed);
  const stride = 1 + width * CHANNELS;
  if (raw.length !== height * stride) {
    throw new Error(`${where}: ${raw.length} bytes after inflating (${height * stride} required)`);
  }

  const data = new Uint8Array(height * width * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride;
    if (raw[rowStart] !== FILTER_NONE) {
      throw new Error(
        `${where}: the filter type of row ${y} is ${raw[rowStart]}` +
          ` (this decoder accepts only None(${FILTER_NONE}) — see the module doc)`,
      );
    }
    data.set(raw.subarray(rowStart + 1, rowStart + stride), y * width * CHANNELS);
  }
  return { data, width, height };
};
