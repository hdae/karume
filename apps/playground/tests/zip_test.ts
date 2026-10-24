import { crc32 as nodeCrc32 } from "node:zlib";
import { describe, expect, it } from "vite-plus/test";

import { crc32, type ZipEntry, zipStored, zipStoredLength } from "../src/lib/zip.ts";

const concat = (parts: readonly Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
};

type Read = { name: string; data: Uint8Array; crc: number; flags: number };

/**
 * A minimal independent reader: EOCD → central directory → local headers. Checks the fields that
 * must agree between the two headers and returns the stored data.
 */
const readZip = (zip: Uint8Array): Read[] => {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const end = zip.length - 22;
  expect(view.getUint32(end, true)).toBe(0x06054b50);
  const count = view.getUint16(end + 10, true);
  const centralBytes = view.getUint32(end + 12, true);
  let cursor = view.getUint32(end + 16, true);
  expect(cursor + centralBytes).toBe(end);
  const decoder = new TextDecoder();
  const entries: Read[] = [];
  for (let index = 0; index < count; index += 1) {
    expect(view.getUint32(cursor, true)).toBe(0x02014b50);
    const flags = view.getUint16(cursor + 8, true);
    expect(view.getUint16(cursor + 10, true)).toBe(0); // stored
    const crc = view.getUint32(cursor + 16, true);
    const size = view.getUint32(cursor + 20, true);
    expect(view.getUint32(cursor + 24, true)).toBe(size);
    const nameLength = view.getUint16(cursor + 28, true);
    const local = view.getUint32(cursor + 42, true);
    const name = decoder.decode(zip.subarray(cursor + 46, cursor + 46 + nameLength));
    expect(view.getUint32(local, true)).toBe(0x04034b50);
    expect(view.getUint32(local + 14, true)).toBe(crc);
    expect(view.getUint16(local + 26, true)).toBe(nameLength);
    const start = local + 30 + nameLength + view.getUint16(local + 28, true);
    entries.push({ name, data: zip.subarray(start, start + size), crc, flags });
    cursor += 46 + nameLength;
  }
  return entries;
};

describe("crc32", () => {
  it("matches zlib's CRC-32", () => {
    const data = new TextEncoder().encode("The quick brown fox jumps over the lazy dog");
    expect(crc32(data)).toBe(nodeCrc32(data));
    expect(crc32(data)).toBe(0x414fa339);
  });
});

describe("zipStored", () => {
  const entries: ZipEntry[] = [
    { name: "frame_000.png", data: Uint8Array.of(1, 2, 3, 4, 5) },
    { name: "empty.bin", data: new Uint8Array(0) },
    { name: "dir/フレーム.txt", data: new TextEncoder().encode("utf-8 name") },
  ];
  const zip = concat(zipStored(entries, new Date(2026, 9, 24, 12, 30, 10)));

  it("round-trips every entry through the central directory", () => {
    const read = readZip(zip);
    expect(read.map(({ name }) => name)).toEqual(entries.map(({ name }) => name));
    for (const [index, entry] of read.entries()) {
      expect(Array.from(entry.data)).toEqual(Array.from(entries[index]?.data ?? []));
      expect(entry.crc).toBe(nodeCrc32(entry.data));
    }
  });

  it("flags only the non-ASCII name as UTF-8", () => {
    expect(readZip(zip).map(({ flags }) => flags & 0x0800)).toEqual([0, 0, 0x0800]);
  });

  it("has the length zipStoredLength predicts", () => {
    const encoder = new TextEncoder();
    expect(zip.length).toBe(
      zipStoredLength(
        entries.map(({ name, data }) => ({
          nameBytes: encoder.encode(name).length,
          dataBytes: data.length,
        })),
      ),
    );
  });

  it("rejects empty and repeated names", () => {
    const at = new Date(2026, 0, 1);
    expect(() => zipStored([{ name: "", data: new Uint8Array() }], at)).toThrow(/empty/);
    const twice = { name: "a", data: new Uint8Array() };
    expect(() => zipStored([twice, twice], at)).toThrow(/repeated/);
  });

  it("refuses inputs that would need ZIP64", () => {
    expect(() => zipStoredLength([{ nameBytes: 1, dataBytes: 0xffffffff }])).toThrow(/ZIP64/);
  });
});
