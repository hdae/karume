/**
 * I2V の条件画像のデコード（PNG / JPEG のバイト列 → `Rgb8Image`）。examples 専用 — Karume のパッケージは画像の
 * デコードを持たず（ランタイム依存ゼロ）、呼び手がデコードして `Rgb8Image` を渡す（`@karume/models/wan` の
 * `Rgb8Image` の doc）。ここはその呼び手の 1 例で、広く使われている pure-JS のデコーダ（PNG = fast-png・
 * JPEG = jpeg-js）を examples だけの依存として使う。MUST: `packages/` からこのモジュールや 2 つの依存を
 * import しない（パッケージのランタイム依存ゼロの不変条件 — CLAUDE.md）。
 *
 * 目標は Pillow の `Image.open(path).convert("RGB")` と同じ RGB8（上流 Wan2.2 の `generate.py` が条件画像を読む形
 * — commit 1ea34ff4 の 376 行目 `Image.open(args.image).convert("RGB")`）:
 *
 * - 形式は拡張子ではなく先頭のマジックバイトで決める。PNG / JPEG 以外は対応形式を名指しして落とす。
 * - アルファは合成せずに捨てる（`convert("RGB")` と同じ — 背景色を決めない）。
 * - EXIF の向き（orientation）は適用しない。上流の `generate.py` も `ImageOps.exif_transpose` を呼ばずに
 *   `convert("RGB")` だけで読む（上の行）ので、同じファイルから同じ向きの画素になる。
 * - PNG は 8 bit の RGB / RGBA / Gray / Gray+Alpha と、パレット（索引の深度 1 / 2 / 4 / 8 bit）を受けて Pillow と
 *   画素単位で一致する。16 bit・パレット以外の 8 bit 未満・インターレース（Adam7）の 8 bit 未満のパレットは落とす
 *   （黙って近似しない — 16 bit を 8 bit へ丸める規則は Pillow と合わせる根拠がまだ無い）。
 * - JPEG は jpeg-js で復号する。参照の Pillow（libjpeg-turbo）と IDCT・色差の補間（jpeg-js は最近傍）が違うので
 *   画素は少しずれる（テストが最大絶対差の上限で縛る）。グレースケールは 3 チャネルへ複製、CMYK（Adobe）は
 *   jpeg-js の式で RGB にする（`(255 − K)·(1 − C / 255)` — Pillow の `cmyk2rgb` と同じ式で、差は丸めの範囲）。
 *   3 成分の色空間（YCbCr か RGB 符号か）は libjpeg の規則でヘッダから決めて jpeg-js へ明示する（`readJpegHeader`）。
 *   標本の精度が 8 bit でない JPEG（12 bit など）は落とす。
 */

import { convertIndexedToRgb, decode as decodePngRaw } from "fast-png";
import { decode as decodeJpegRaw } from "jpeg-js";
import type { Rgb8Image } from "../../packages/models/wan.ts";

/** 対応形式の名指し（拒否の文言に入れる）。 */
const SUPPORTED =
  "対応形式は PNG（8 bit の RGB / RGBA / Gray / Gray+Alpha・パレット）と 8 bit の JPEG";

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
/** JPEG の SOI（FF D8）+ 次のマーカーの先頭（FF）。 */
const JPEG_SIGNATURE = [0xff, 0xd8, 0xff];

const startsWith = (bytes: Uint8Array, signature: readonly number[]): boolean =>
  bytes.length >= signature.length && signature.every((byte, at) => bytes[at] === byte);

/** PNG の色の種類（IHDR — PNG 仕様 11.2.2）→ 1 画素のチャネル数（索引の 3 は別扱い）。 */
const PNG_COLOR_TYPES: Readonly<
  Record<number, { readonly name: string; readonly channels: number }>
> = {
  0: { name: "Gray", channels: 1 },
  2: { name: "RGB", channels: 3 },
  3: { name: "パレット", channels: 1 },
  4: { name: "Gray+Alpha", channels: 2 },
  6: { name: "RGBA", channels: 4 },
};
const PNG_INDEXED = 3;

/**
 * IHDR の深度・色の種類・インターレースを読む。fast-png の復号結果は色の種類を持たない（パレットの有無では
 * RGB に付く推奨パレット〈PLTE〉と索引を区別できない）ので、ヘッダから直接読む。
 */
const readPngHeader = (bytes: Uint8Array) => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // 署名 8 バイト → 先頭チャンクの長さ 4 + 型 4 → IHDR の本体 13 バイト（MUST: IHDR は先頭のチャンク — 仕様 5.6）。
  if (bytes.length < 33 || String.fromCharCode(...bytes.subarray(12, 16)) !== "IHDR") {
    throw new Error(
      `PNG の先頭のチャンクが IHDR でない（壊れているか途中で切れている — ${SUPPORTED}）`,
    );
  }
  return {
    width: view.getUint32(16),
    height: view.getUint32(20),
    depth: bytes[24],
    colorType: bytes[25],
    interlaced: bytes[28] !== 0,
  };
};

const decodePng = (bytes: Uint8Array): Rgb8Image => {
  const header = readPngHeader(bytes);
  const color = Object.hasOwn(PNG_COLOR_TYPES, header.colorType)
    ? PNG_COLOR_TYPES[header.colorType]
    : undefined;
  if (color === undefined) {
    throw new Error(`PNG の色の種類 ${header.colorType} を知らない（${SUPPORTED}）`);
  }
  const indexed = header.colorType === PNG_INDEXED;
  if (header.depth === 16 || (!indexed && header.depth !== 8)) {
    throw new Error(
      `PNG の ${color.name} の ${header.depth} bit は受けない（${SUPPORTED} — 8 bit 未満はパレットだけ）`,
    );
  }
  // fast-png の Adam7 の経路は 1 画素 = 1 バイト以上を前提に行のバイト数を数える（8 bit 未満の詰めた行を読み違える）。
  if (indexed && header.interlaced && header.depth < 8) {
    throw new Error(
      `インターレース（Adam7）の ${header.depth} bit のパレットの PNG は受けない（${SUPPORTED} — ` +
        `インターレースを外すか 8 bit のパレットで保存し直す）`,
    );
  }
  let decoded: ReturnType<typeof decodePngRaw>;
  let expanded: Uint8Array | undefined;
  try {
    // checkCrc: チャンクの CRC を検査する（壊れたデータを黙って画素にしない — 既定は検査しない）。
    decoded = decodePngRaw(bytes, { checkCrc: true });
    // パレットの色は [r, g, b]、tRNS があると [r, g, b, a]（fast-png の decodetRNS が全色にアルファを足す）。
    // 展開も壊れたデータで投げる（PLTE が無い・索引がパレットの外）ので同じ文言で包む。
    expanded = indexed ? convertIndexedToRgb(decoded) : undefined;
  } catch (cause) {
    throw new Error(
      `PNG を復号できない（壊れているか途中で切れている — ${SUPPORTED}）: ${String(cause)}`,
      { cause },
    );
  }
  const { width, height } = decoded;
  const pixels = width * height;
  const rgb = new Uint8Array(pixels * 3);
  if (expanded !== undefined) {
    const stride = expanded.length / pixels;
    if (stride !== 3 && stride !== 4) {
      throw new Error(
        `パレットの展開の長さ ${expanded.length} が ${pixels} 画素の 3 / 4 倍でない（壊れている — ${SUPPORTED}）`,
      );
    }
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      rgb.set(expanded.subarray(pixel * stride, pixel * stride + 3), pixel * 3);
    }
    return { data: rgb, width, height };
  }
  const { data } = decoded;
  if (!(data instanceof Uint8Array) || data.length !== pixels * color.channels) {
    throw new Error(
      `PNG の画素の長さ ${data.length} が ${width}×${height}×${color.channels} と違う（壊れている — ${SUPPORTED}）`,
    );
  }
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const from = pixel * color.channels;
    // Gray / Gray+Alpha は灰を 3 チャネルへ複製、RGB / RGBA は先頭の 3 つ（アルファは捨てる）。
    const gray = color.channels <= 2;
    rgb[pixel * 3] = data[from];
    rgb[pixel * 3 + 1] = data[gray ? from : from + 1];
    rgb[pixel * 3 + 2] = data[gray ? from : from + 2];
  }
  return { data: rgb, width, height };
};

/** JPEG のヘッダ（SOI から最初の SOS まで）から読む、色空間と精度の判定に要る事実。 */
type JpegHeader = {
  /** SOF の標本の精度（bit）。 */
  readonly precision: number;
  /** SOF の成分 ID の並び。 */
  readonly componentIds: readonly number[];
  /** JFIF の APP0 があるか。 */
  readonly jfif: boolean;
  /** Adobe の APP14 の transform（無ければ undefined）。 */
  readonly adobeTransform?: number;
};

const ascii = (bytes: Uint8Array, length: number): string =>
  String.fromCharCode(...bytes.subarray(0, length));

/** SOF のマーカー（C0〜CF のうち DHT〈C4〉・JPG〈C8〉・DAC〈CC〉以外 — JPEG 仕様 B.1.1.3）。 */
const isStartOfFrame = (marker: number): boolean =>
  marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;

/**
 * SOI の後のマーカーの区画を最初の SOS まで辿る（libjpeg が色空間を決めるのもこの範囲 — `jpeg_read_header`）。
 * 区画が途中で切れていれば落とす。
 */
const readJpegHeader = (bytes: Uint8Array): JpegHeader => {
  const broken = (why: string): Error =>
    new Error(`JPEG を復号できない（${why} — 壊れているか途中で切れている — ${SUPPORTED}）`);
  let jfif = false;
  let adobeTransform: number | undefined;
  let frame: { readonly precision: number; readonly componentIds: number[] } | undefined;
  // SOI（FF D8 の 2 バイト）の直後から。
  let at = 2;
  for (;;) {
    if (at >= bytes.length || bytes[at] !== 0xff) throw broken("マーカーが続かない");
    // マーカーの前の詰め物の FF は飛ばす（仕様 B.1.1.2）。
    while (at < bytes.length && bytes[at] === 0xff) at += 1;
    if (at >= bytes.length) throw broken("SOS の前で終わる");
    const marker = bytes[at];
    at += 1;
    if (marker === 0xda) break;
    // 長さを持たない単独のマーカー（TEM・RSTn）。
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (at + 2 > bytes.length) throw broken("区画の長さが切れている");
    const length = (bytes[at] << 8) | bytes[at + 1];
    if (length < 2 || at + length > bytes.length) throw broken("区画が途中で切れている");
    const segment = bytes.subarray(at + 2, at + length);
    if (marker === 0xe0 && segment.length >= 5 && ascii(segment, 5) === "JFIF\0") jfif = true;
    // libjpeg と同じく本体 12 バイト以上の "Adobe" だけを Adobe の区画とみなす（`examine_app14`）。
    if (marker === 0xee && segment.length >= 12 && ascii(segment, 5) === "Adobe") {
      adobeTransform = segment[11];
    }
    if (isStartOfFrame(marker)) {
      const count = segment[5];
      if (segment.length < 6 + count * 3) throw broken("SOF が切れている");
      frame = {
        precision: segment[0],
        componentIds: Array.from({ length: count }, (_, index) => segment[6 + index * 3]),
      };
    }
    at += length;
  }
  if (frame === undefined) throw broken("SOS の前に SOF が無い");
  return { ...frame, jfif, ...(adobeTransform === undefined ? {} : { adobeTransform }) };
};

/**
 * 3 成分の JPEG が YCbCr か（false なら RGB 符号 — 色の変換をしない）。libjpeg の `default_decompress_parms` と
 * 同じ順で決める: JFIF があれば YCbCr、無くて Adobe があれば transform が 0 のときだけ RGB、どちらも無ければ
 * 成分 ID が 'R' 'G' 'B' のときだけ RGB。jpeg-js は既定で 3 成分を常に YCbCr とみなすので、Adobe の transform = 0
 * （Pillow の `keep_rgb=True` が書く形）を黙って誤った色にする — 判定をここで済ませて `colorTransform` で明示する。
 * Pillow 12.3.0 で 3 つの場合（Adobe 0・ID だけ・JFIF + Adobe 0）の判定を実測し、テストが縛る。
 */
const isYCbCr = (header: JpegHeader): boolean => {
  if (header.jfif) return true;
  if (header.adobeTransform !== undefined) return header.adobeTransform !== 0;
  const [r, g, b] = header.componentIds;
  return !(r === 0x52 && g === 0x47 && b === 0x42);
};

const decodeJpeg = (bytes: Uint8Array): Rgb8Image => {
  const header = readJpegHeader(bytes);
  // jpeg-js は精度を読むが、出力は常に 8 bit へ clamp する（12 bit の値域を縮めない）ので、黙って白飛びさせない。
  if (header.precision !== 8) {
    throw new Error(`JPEG の標本の精度 ${header.precision} bit は受けない（${SUPPORTED}）`);
  }
  // 4 成分（CMYK / YCCK）は jpeg-js の既定（Adobe の transform で決まる）に任せる — `colorTransform` を明示すると
  // 4 成分の判定まで上書きする（`copyToImageData` の case 4）。
  const colorSpace = header.componentIds.length === 3 ? { colorTransform: isYCbCr(header) } : {};
  try {
    // useTArray: Buffer ではなく Uint8Array を返す。formatAsRGBA: false で 1 画素 3 バイト（グレースケールは
    // Y を 3 チャネルへ複製・CMYK は jpeg-js の式で RGB — `copyToImageData`）。
    // NOTE: tolerantDecoding は既定（true）のまま — 効くのはフレームの外へはみ出すブロックを書かずに飛ばす所だけで
    // （`decodeMcu` / `decodeBlock`）、見える画素は変わらない。false にすると Pillow が読める規格外の JPEG まで拒む。
    // 途中で切れたデータは真偽に関わらず jpeg-js が投げる（テストが縛る）。
    const { data, width, height } = decodeJpegRaw(bytes, {
      useTArray: true,
      formatAsRGBA: false,
      ...colorSpace,
    });
    return { data, width, height };
  } catch (cause) {
    throw new Error(
      `JPEG を復号できない（壊れているか途中で切れているか、対応外の JPEG — ${SUPPORTED}）: ${
        String(cause)
      }`,
      { cause },
    );
  }
};

/**
 * 画像ファイルのバイト列 → RGB8（モジュール doc の規則）。形式は先頭のマジックバイトで決め、対応外・壊れた
 * データは対応形式を名指しする `Error` で落とす。
 */
export const decodeImage = (bytes: Uint8Array): Rgb8Image => {
  if (startsWith(bytes, PNG_SIGNATURE)) return decodePng(bytes);
  if (startsWith(bytes, JPEG_SIGNATURE)) return decodeJpeg(bytes);
  throw new Error(`画像の先頭のバイト列が PNG でも JPEG でもない（${SUPPORTED}）`);
};
