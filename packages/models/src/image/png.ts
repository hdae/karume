/**
 * RGBA バイト列 → PNG（8bit truecolor RGB・非インタレース）。
 *
 * **パイプライン非依存の共通処理**（画像生成モデルは総じて最後に PNG を書く）。ランタイム
 * 依存は **Web 標準 API のみ**（ADR 0002）で、圧縮は `CompressionStream("deflate")` に任せる。
 *
 * MUST: `"deflate"` であって `"deflate-raw"` ではない。Compression Streams の `"deflate"` は
 * **RFC1950 の zlib ラッパ付き**（2 バイトのヘッダ + 末尾 Adler-32）を出し、これは PNG の
 * IDAT が要求する形そのもの。`"deflate-raw"` を使うと zlib ヘッダが無く、デコーダは
 * 「不正な圧縮方式」で落ちる。
 *
 * MUST: 各スキャンラインの先頭に filter type バイト `0x00`（None）を挿入してから圧縮する。
 * 挿入を忘れると行が 1 バイトずつずれ、**画像は開けるが色が斜めに流れる**（`Uint8Array` の
 * 長さ検査では捕まらない）。
 *
 * CRC32 は Web 標準に無いので自前で持つ（表 256 エントリ・PNG 仕様 Annex D の多項式）。
 */

/** PNG シグネチャ（8 バイト）。 */
const SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** 8bit / チャネル。 */
const BIT_DEPTH = 8;
/** color type 2 = truecolor（RGB・アルファ無し）。 */
const COLOR_TYPE_RGB = 2;
const COMPRESSION_DEFLATE = 0;
const FILTER_ADAPTIVE = 0;
const INTERLACE_NONE = 0;
/** スキャンライン先頭の filter type（None）。 */
const FILTER_NONE = 0;

/**
 * CRC-32 の表を組む。
 *
 * MUST: モジュールスコープの `const` に持たない（横断不変条件「全モジュール副作用ゼロ」=
 * import 時実行もグローバル可変状態も禁止 — barrel 経由 tree-shaking の成立条件）。
 * 呼び出しごとに組み直すが、PNG 1 枚あたりの呼び出しは 3 チャンクだけで、表の構築は
 * 2,048 回の整数演算 = 数 MB の IDAT に掛かる CRC 本体に対して無視できる。
 */
const buildCrcTable = (): Uint32Array => {
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

/** PNG チャンクの CRC-32（ISO 3309 / ITU-T V.42 — 反転初期値・反転出力）。 */
export const crc32 = (bytes: Uint8Array): number => {
  const table = buildCrcTable();
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = table[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
};

/** 1 チャンク（長さ 4 + 型 4 + データ + CRC 4）。CRC は**型を含む**範囲に掛かる。 */
const chunk = (type: string, data: Uint8Array): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let index = 0; index < 4; index += 1) out[4 + index] = type.charCodeAt(index);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
};

/**
 * zlib（RFC1950）形式へ圧縮する。
 * NOTE: `Uint8Array<ArrayBuffer>` を要求するのは `SharedArrayBuffer` 由来を弾くため
 * （`CompressionStream` の `BufferSource` は共有バッファを受け付けない）。
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
 * RGBA（4 バイト / 画素・行優先）を PNG バイト列にする。
 *
 * アルファは**落とす**（truecolor RGB で書く）。生成画像は不透明で、PNG にアルファ面を
 * 持たせると 1/4 が定数 255 の死に容量になるだけだから。
 * MUST: 落とす前に 255 であることを検査する（半透明を黙って不透明化しない）。
 */
export const encodePng = async (
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`画像サイズ ${width}×${height} が正の整数でない`);
  }
  if (rgba.length !== width * height * 4) {
    throw new Error(`RGBA の長さ ${rgba.length} が 4×${width}×${height} と違う`);
  }

  const stride = 1 + width * 3;
  const raw = new Uint8Array(height * stride);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride;
    raw[rowStart] = FILTER_NONE;
    for (let x = 0; x < width; x += 1) {
      const source = (y * width + x) * 4;
      if (rgba[source + 3] !== 255) {
        throw new Error(`画素 (${x}, ${y}) のアルファが ${rgba[source + 3]}（255 でない）`);
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
    SIGNATURE,
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
