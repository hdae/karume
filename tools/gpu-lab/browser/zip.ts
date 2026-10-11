/**
 * 無圧縮（stored）の zip の書き手 — Wan のタブが連番 PNG を 1 つのファイルで保存するため。
 *
 * 中身は既に圧縮済みの PNG なので deflate は掛けない（縮まないうえ、書き手が小さく保てる）。形は APPNOTE の
 * 素の zip だけ: 各項目のローカルヘッダ + データ、中央ディレクトリ、終端レコード（EOCD）。名前は UTF-8 で書き、
 * 汎用フラグの bit 11 を立てる。ZIP64 は書かない — 項目数・オフセット・大きさが素の欄に収まらない入力は
 * 黙って壊れた zip にせず投げる（{@link zipStoredLength}）。
 *
 * CRC-32 は PNG の書き手と同じもの（`packages/models/src/image/png.ts` — 多項式は zip と PNG で同じ
 * ISO 3309）を使い、2 本持たない。
 */
import { crc32 } from "../../../packages/models/src/image/png.ts";

export type ZipEntry = {
  /** zip の中の名前（`/` 区切り・空や重複は不可）。 */
  readonly name: string;
  readonly data: Uint8Array<ArrayBuffer>;
};

const LOCAL_HEADER_BYTES = 30;
const CENTRAL_HEADER_BYTES = 46;
const END_RECORD_BYTES = 22;

const LOCAL_HEADER_SIGNATURE = 0x04034b50;
const CENTRAL_HEADER_SIGNATURE = 0x02014b50;
const END_RECORD_SIGNATURE = 0x06054b50;

/** version made by / needed to extract = 2.0（MS-DOS の属性 — 上位バイト 0）。 */
const ZIP_VERSION = 20;
/** 汎用フラグの bit 11 = 名前とコメントが UTF-8。 */
const FLAG_UTF8 = 0x0800;
const METHOD_STORED = 0;

/**
 * 素の欄に入る最大の値。`0xFFFF` / `0xFFFFFFFF` そのものは ZIP64 の目印として読む読み手があるので使わない。
 */
export const ZIP_MAX_ENTRIES = 0xfffe;
export const ZIP_MAX_OFFSET = 0xfffffffe;
const MAX_NAME_BYTES = 0xffff;

/**
 * zip 全体のバイト数（ローカルヘッダ + データ + 中央ディレクトリ + 終端レコード）。ZIP64 が要る入力
 * （項目数が {@link ZIP_MAX_ENTRIES} 超・中央ディレクトリのオフセットか大きさが {@link ZIP_MAX_OFFSET} 超・
 * 名前が 65,535 バイト超）は投げる。各項目のオフセットと大きさは中央ディレクトリのオフセット以下なので、
 * その 2 つを見れば足りる。
 */
export const zipStoredLength = (
  entries: readonly { readonly nameBytes: number; readonly dataBytes: number }[],
): number => {
  if (entries.length > ZIP_MAX_ENTRIES) {
    throw RangeError(
      `zip の項目数 ${entries.length} が ZIP64 なしの上限 ${ZIP_MAX_ENTRIES} を超える`,
    );
  }
  let centralOffset = 0;
  let centralBytes = 0;
  for (const { nameBytes, dataBytes } of entries) {
    if (nameBytes > MAX_NAME_BYTES) {
      throw RangeError(`zip の名前 ${nameBytes} バイトが上限 ${MAX_NAME_BYTES} を超える`);
    }
    centralOffset += LOCAL_HEADER_BYTES + nameBytes + dataBytes;
    centralBytes += CENTRAL_HEADER_BYTES + nameBytes;
  }
  if (centralOffset > ZIP_MAX_OFFSET || centralBytes > ZIP_MAX_OFFSET) {
    throw RangeError(
      `zip の中央ディレクトリ（オフセット ${centralOffset}・大きさ ${centralBytes} バイト）が ZIP64 なしの上限 ${ZIP_MAX_OFFSET} を超える`,
    );
  }
  return centralOffset + centralBytes + END_RECORD_BYTES;
};

/** MS-DOS の日付と時刻（地方時・2 秒単位）。表せない年（1980〜2107 の外）は投げる。 */
const dosDateTime = (at: Date): { readonly date: number; readonly time: number } => {
  const year = at.getFullYear();
  if (year < 1980 || year > 2107) {
    throw RangeError(`zip の日時 ${at.toISOString()} は MS-DOS の日付（1980〜2107 年）で表せない`);
  }
  return {
    date: ((year - 1980) << 9) | ((at.getMonth() + 1) << 5) | at.getDate(),
    time: (at.getHours() << 11) | (at.getMinutes() << 5) | (at.getSeconds() >> 1),
  };
};

/**
 * 無圧縮の zip を組む。返り値は順に並べると zip になる断片の列（データは写さずにそのまま入れる — 数百 MB の
 * 連番 PNG を 2 重に持たないため。`new Blob(parts)` にそのまま渡せる）。全項目の日時は `modified`。
 */
export const zipStored = (
  entries: readonly ZipEntry[],
  modified: Date,
): Uint8Array<ArrayBuffer>[] => {
  const encoder = new TextEncoder();
  const names = entries.map(({ name }) => encoder.encode(name));
  const seen = new Set<string>();
  for (const { name } of entries) {
    if (name === "") throw Error("zip の項目の名前が空");
    // 重複は解凍で黙って上書きになる。
    if (seen.has(name)) throw Error(`zip の項目の名前 '${name}' が重複`);
    seen.add(name);
  }
  // 大きさの検査を CRC の計算より先に行う（上限を超える入力に CRC を回さない）。
  zipStoredLength(
    entries.map(({ data }, index) => ({ nameBytes: names[index].length, dataBytes: data.length })),
  );
  const { date, time } = dosDateTime(modified);

  const parts: Uint8Array<ArrayBuffer>[] = [];
  const centrals: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;
  for (const [index, { data }] of entries.entries()) {
    const name = names[index];
    const crc = crc32(data);

    const local = new Uint8Array(LOCAL_HEADER_BYTES + name.length);
    const localView = new DataView(local.buffer);
    localView.setUint32(0, LOCAL_HEADER_SIGNATURE, true);
    localView.setUint16(4, ZIP_VERSION, true);
    localView.setUint16(6, FLAG_UTF8, true);
    localView.setUint16(8, METHOD_STORED, true);
    localView.setUint16(10, time, true);
    localView.setUint16(12, date, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, data.length, true);
    localView.setUint32(22, data.length, true);
    localView.setUint16(26, name.length, true);
    localView.setUint16(28, 0, true);
    local.set(name, LOCAL_HEADER_BYTES);

    const central = new Uint8Array(CENTRAL_HEADER_BYTES + name.length);
    const centralView = new DataView(central.buffer);
    centralView.setUint32(0, CENTRAL_HEADER_SIGNATURE, true);
    centralView.setUint16(4, ZIP_VERSION, true);
    centralView.setUint16(6, ZIP_VERSION, true);
    centralView.setUint16(8, FLAG_UTF8, true);
    centralView.setUint16(10, METHOD_STORED, true);
    centralView.setUint16(12, time, true);
    centralView.setUint16(14, date, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, data.length, true);
    centralView.setUint32(24, data.length, true);
    centralView.setUint16(28, name.length, true);
    // extra 長・コメント長・開始ディスク・内部属性・外部属性は 0（30〜41）
    centralView.setUint32(42, offset, true);
    central.set(name, CENTRAL_HEADER_BYTES);

    parts.push(local, data);
    centrals.push(central);
    offset += local.length + data.length;
  }

  const centralBytes = centrals.reduce((sum, central) => sum + central.length, 0);
  const end = new Uint8Array(END_RECORD_BYTES);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, END_RECORD_SIGNATURE, true);
  // ディスク番号と中央ディレクトリの開始ディスクは 0（4〜7）
  endView.setUint16(8, entries.length, true);
  endView.setUint16(10, entries.length, true);
  endView.setUint32(12, centralBytes, true);
  endView.setUint32(16, offset, true);
  // コメント長は 0（20〜21）
  return [...parts, ...centrals, end];
};
