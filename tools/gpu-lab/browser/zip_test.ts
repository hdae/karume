import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { ZIP_MAX_ENTRIES, ZIP_MAX_OFFSET, zipStored, zipStoredLength } from "./zip.ts";

const ascii = (text: string): Uint8Array<ArrayBuffer> => new TextEncoder().encode(text);

const concat = (parts: readonly Uint8Array<ArrayBuffer>[]): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
};

type ReadEntry = {
  readonly name: string;
  readonly data: Uint8Array<ArrayBuffer>;
  readonly crc: number;
  readonly flags: number;
  readonly method: number;
  readonly date: number;
  readonly time: number;
};

/**
 * テストの中の小さな読み手: 終端レコード → 中央ディレクトリ → 各ローカルヘッダの順に辿り、
 * 中央とローカルの欄が食い違えば落とす（書き手の写し間違いをここで捕まえる）。
 */
const readZip = (bytes: Uint8Array<ArrayBuffer>): ReadEntry[] => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const end = bytes.length - 22;
  assertEquals(view.getUint32(end, true), 0x06054b50, "終端レコードの署名");
  assertEquals(view.getUint16(end + 20, true), 0, "コメント長");
  const count = view.getUint16(end + 10, true);
  assertEquals(view.getUint16(end + 8, true), count, "このディスクの項目数");
  const centralBytes = view.getUint32(end + 12, true);
  const centralOffset = view.getUint32(end + 16, true);
  assertEquals(centralOffset + centralBytes, end, "中央ディレクトリは終端レコードの直前で終わる");
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const entries: ReadEntry[] = [];
  let at = centralOffset;
  let expectedLocal = 0;
  for (let index = 0; index < count; index += 1) {
    assertEquals(view.getUint32(at, true), 0x02014b50, "中央ヘッダの署名");
    const flags = view.getUint16(at + 8, true);
    const method = view.getUint16(at + 10, true);
    const time = view.getUint16(at + 12, true);
    const date = view.getUint16(at + 14, true);
    const crc = view.getUint32(at + 16, true);
    const compressed = view.getUint32(at + 20, true);
    const size = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    assertEquals(
      view.getUint16(at + 30, true) + view.getUint16(at + 32, true),
      0,
      "extra・コメント",
    );
    const localOffset = view.getUint32(at + 42, true);
    const name = decoder.decode(bytes.subarray(at + 46, at + 46 + nameLength));
    assertEquals(compressed, size, "stored は圧縮後 = 圧縮前");
    assertEquals(localOffset, expectedLocal, "ローカルヘッダは隙間なく並ぶ");

    assertEquals(view.getUint32(localOffset, true), 0x04034b50, "ローカルヘッダの署名");
    assertEquals(
      [6, 8, 10, 12].map((field) => view.getUint16(localOffset + field, true)),
      [flags, method, time, date],
      "ローカルと中央の flags / method / 時刻 / 日付",
    );
    assertEquals(
      [14, 18, 22].map((field) => view.getUint32(localOffset + field, true)),
      [crc, compressed, size],
      "ローカルと中央の CRC / 大きさ",
    );
    assertEquals(view.getUint16(localOffset + 26, true), nameLength, "ローカルの名前長");
    const extraLength = view.getUint16(localOffset + 28, true);
    const localName = decoder.decode(
      bytes.subarray(localOffset + 30, localOffset + 30 + nameLength),
    );
    assertEquals(localName, name, "ローカルと中央の名前");
    const dataStart = localOffset + 30 + nameLength + extraLength;
    entries.push({
      name,
      data: bytes.slice(dataStart, dataStart + size),
      crc,
      flags,
      method,
      date,
      time,
    });
    expectedLocal = dataStart + size;
    at += 46 + nameLength;
  }
  assertEquals(expectedLocal, centralOffset, "データの直後に中央ディレクトリが始まる");
  assertEquals(at, end, "中央ディレクトリの大きさが項目の和と一致");
  return entries;
};

const MODIFIED = new Date(2026, 9, 11, 13, 45, 31);

describe("zipStored", () => {
  describe("項目を入れたとき", () => {
    const entries = [
      { name: "frame-0001.png", data: ascii("123456789") },
      { name: "frame-0002.png", data: new Uint8Array([0, 255, 1, 254]) },
      { name: "フレーム.png", data: new Uint8Array(0) },
    ];
    const bytes = concat(zipStored(entries, MODIFIED));

    it("読み戻すと名前と中身が入れた順のまま一致する", () => {
      assertEquals(
        readZip(bytes).map(({ name, data }) => ({ name, data })),
        entries.map(({ name, data }) => ({ name, data: new Uint8Array(data) })),
      );
    });

    it('CRC-32 の欄が既知の値になる（"123456789" → 0xCBF43926・空 → 0）', () => {
      const read = readZip(bytes);
      assertEquals(read[0].crc, 0xcbf43926);
      assertEquals(read[2].crc, 0);
    });

    it("全項目が無圧縮（method 0）で UTF-8 の名前のフラグ（bit 11）を立てる", () => {
      for (const entry of readZip(bytes)) {
        assertEquals(entry.method, 0);
        assertEquals(entry.flags, 0x0800);
      }
    });

    it("日時は MS-DOS の形（地方時・秒は 2 秒単位に切り捨て）", () => {
      const { date, time } = readZip(bytes)[0];
      assertEquals(
        [date >> 9, (date >> 5) & 0xf, date & 0x1f],
        [2026 - 1980, 10, 11],
      );
      assertEquals([time >> 11, (time >> 5) & 0x3f, (time & 0x1f) * 2], [13, 45, 30]);
    });

    it("全体の長さが zipStoredLength の値と一致する", () => {
      const encoder = new TextEncoder();
      assertEquals(
        bytes.length,
        zipStoredLength(
          entries.map(({ name, data }) => ({
            nameBytes: encoder.encode(name).length,
            dataBytes: data.length,
          })),
        ),
      );
    });

    it("データは写さずに断片としてそのまま並ぶ", () => {
      const parts = zipStored(entries, MODIFIED);
      for (const { data } of entries) {
        assertEquals(parts.includes(data), true);
      }
    });
  });

  describe("項目が無いとき", () => {
    it("終端レコードだけの 22 バイトになり、読み戻すと空", () => {
      const bytes = concat(zipStored([], MODIFIED));
      assertEquals(bytes.length, 22);
      assertEquals(readZip(bytes), []);
    });
  });

  describe("受けない入力", () => {
    it("名前が重複すると投げる（解凍で黙って上書きになる）", () => {
      assertThrows(
        () =>
          zipStored([
            { name: "a.png", data: new Uint8Array(1) },
            { name: "a.png", data: new Uint8Array(2) },
          ], MODIFIED),
        Error,
        "重複",
      );
    });

    it("名前が空だと投げる", () => {
      assertThrows(() => zipStored([{ name: "", data: new Uint8Array(1) }], MODIFIED), Error, "空");
    });

    it("項目数が ZIP64 なしの上限を超えると投げる", () => {
      const entries = Array.from(
        { length: ZIP_MAX_ENTRIES + 1 },
        (_, index) => ({ name: `${index}`, data: new Uint8Array(0) }),
      );
      assertThrows(() => zipStored(entries, MODIFIED), RangeError, "項目数");
    });

    it("MS-DOS の日付で表せない日時（1980 年より前）は投げる", () => {
      assertThrows(
        () => zipStored([{ name: "a", data: new Uint8Array(0) }], new Date(1979, 11, 31)),
        RangeError,
        "MS-DOS",
      );
    });
  });
});

describe("zipStoredLength", () => {
  it("上限ちょうどの項目数は受ける", () => {
    assertEquals(
      zipStoredLength(
        Array.from({ length: ZIP_MAX_ENTRIES }, () => ({ nameBytes: 0, dataBytes: 0 })),
      ),
      ZIP_MAX_ENTRIES * (30 + 46) + 22,
    );
  });

  it("中央ディレクトリのオフセットが上限ちょうどなら受け、1 バイト超えると投げる", () => {
    // 1 項目・名前 1 バイト: 中央ディレクトリのオフセット = 30 + 1 + データ
    const fits = ZIP_MAX_OFFSET - 31;
    assertEquals(zipStoredLength([{ nameBytes: 1, dataBytes: fits }]), ZIP_MAX_OFFSET + 47 + 22);
    assertThrows(
      () => zipStoredLength([{ nameBytes: 1, dataBytes: fits + 1 }]),
      RangeError,
      "ZIP64",
    );
  });

  it("項目の和で 4 GiB を超えると投げる（1 つずつは収まっても）", () => {
    const half = 0x80000000;
    assertThrows(
      () => zipStoredLength([{ nameBytes: 1, dataBytes: half }, { nameBytes: 1, dataBytes: half }]),
      RangeError,
      "ZIP64",
    );
  });

  it("名前が 65,535 バイトを超えると投げる", () => {
    assertThrows(() => zipStoredLength([{ nameBytes: 0x10000, dataBytes: 0 }]), RangeError, "名前");
  });
});
