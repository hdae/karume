/**
 * PNG → RGB8 の**最小デコーダ（テスト専用）**。
 *
 * karume 本体は decode を持たない（`packages/models/src/image/preprocess.ts` のモジュール
 * doc — 入口は「RGB8 の画素列 + 幅 + 高さ」で、PNG / JPEG のデコーダは持たない設計）。
 * ここはその設計を変えずに**テストが実画像を読む**ためだけの道具で、`mod.ts` にも
 * サブパス面にも出さない（publish は `tests/` を除外する）。
 *
 * ## 受理するのは `encodePng` が書く形だけ（それ以外は fail loudly）
 *
 * 読む相手は `packages/models/src/image/png.ts` が焼いた PNG（`outputs/demo/*.png`）だけなので、
 * 汎用デコーダにはしない — **8bit / truecolor RGB（color type 2）/ 非インタレース / 全
 * スキャンラインの filter type が None** の 1 形のみを受け、外れたら例外にする。Paeth などの
 * 行フィルタや palette / gray / アルファ付きを「一応通す」形にすると、対応していない経路を
 * 黙って近似したまま実画像 e2e が緑になりうる。
 *
 * NOTE: チャンク CRC は検証しない（改竄ではなく取り違えを見る道具で、壊れたバイト列は
 * zlib 展開か長さ検査で落ちる）。IDAT は複数チャンクに分かれていても連結して受ける
 * （PNG 仕様が許す形 — `encodePng` は 1 本しか書かないが、連結は分岐を増やさない）。
 */

/** デコード結果（`Rgb8Image` と構造互換 — 行優先・画素あたり 3 バイト）。 */
export type DecodedRgb8 = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** PNG シグネチャ（8 バイト）。 */
const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** 受理する IHDR（`encodePng` が書く値そのもの）。 */
const BIT_DEPTH = 8;
/** color type 2 = truecolor（RGB・アルファ無し）。 */
const COLOR_TYPE_RGB = 2;
const COMPRESSION_DEFLATE = 0;
const FILTER_ADAPTIVE = 0;
const INTERLACE_NONE = 0;
/** 受理するスキャンライン先頭の filter type（None のみ）。 */
const FILTER_NONE = 0;

/** チャネル数（RGB）。 */
const CHANNELS = 3;

/** zlib（RFC1950）を展開する。`encodePng` の `CompressionStream("deflate")` の逆。 */
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

/** チャンクの型（4 バイトの ASCII）を読む。 */
const chunkType = (bytes: Uint8Array, at: number): string =>
  String.fromCharCode(bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]);

/** IHDR の 13 バイトを検査して `[width, height]` を返す。 */
const parseHeader = (data: Uint8Array<ArrayBuffer>, where: string): [number, number] => {
  if (data.length !== 13) throw new Error(`${where}: IHDR が ${data.length} バイト（13 が必要）`);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const width = view.getUint32(0);
  const height = view.getUint32(4);
  if (width <= 0 || height <= 0) throw new Error(`${where}: 画像サイズ ${width}×${height}`);
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
      `${where}: IHDR が [depth,color,compression,filter,interlace] = [${declared.join(",")}]` +
        `（このデコーダが受けるのは [${expected.join(",")}] だけ — モジュール doc）`,
    );
  }
  return [width, height];
};

/**
 * PNG バイト列を RGB8 へ展開する。
 *
 * @param bytes PNG ファイルの中身
 * @param where 失敗時に名指しする出所（ファイル名など）
 */
export const decodePng = async (
  bytes: Uint8Array<ArrayBuffer>,
  where: string,
): Promise<DecodedRgb8> => {
  for (let index = 0; index < SIGNATURE.length; index += 1) {
    if (bytes[index] !== SIGNATURE[index]) throw new Error(`${where}: PNG シグネチャでない`);
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
      throw new Error(`${where}: チャンク '${type}' がファイル末尾を越える`);
    }
    const data = bytes.slice(begin, begin + length);
    if (type === "IHDR") size = parseHeader(data, where);
    if (type === "IDAT") parts.push(data);
    if (type === "IEND") break;
    at = begin + length + 4;
  }
  if (size === undefined) throw new Error(`${where}: IHDR が無い`);
  if (parts.length === 0) throw new Error(`${where}: IDAT が無い`);
  const [width, height] = size;

  const compressed = new Uint8Array(
    parts.reduce((sum, part) => sum + part.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    compressed.set(part, offset);
    offset += part.length;
  }

  const raw = await inflate(compressed);
  const stride = 1 + width * CHANNELS;
  if (raw.length !== height * stride) {
    throw new Error(`${where}: 展開後が ${raw.length} バイト（${height * stride} が必要）`);
  }

  const data = new Uint8Array(height * width * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride;
    if (raw[rowStart] !== FILTER_NONE) {
      throw new Error(
        `${where}: 行 ${y} の filter type が ${raw[rowStart]}` +
          `（このデコーダが受けるのは None(${FILTER_NONE}) だけ — モジュール doc）`,
      );
    }
    data.set(raw.subarray(rowStart + 1, rowStart + stride), y * width * CHANNELS);
  }
  return { data, width, height };
};
