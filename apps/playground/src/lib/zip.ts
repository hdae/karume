/**
 * Store-only (uncompressed) zip writer, used to save a video's frames as one file of PNGs.
 *
 * The entries are PNGs, which are compressed already, so deflate would not shrink them and is left
 * out to keep the writer small. The layout is plain APPNOTE zip: local header + data per entry, the
 * central directory, the end-of-central-directory record. Names are UTF-8; only entries with a
 * non-ASCII name set general-purpose bit 11 and "version needed" 6.3 (bit 11 came with APPNOTE 6.3 —
 * an ASCII name is the same bytes in CP437 and UTF-8, so it stays 2.0 for older readers). No ZIP64:
 * an input whose count, offsets or sizes do not fit the plain fields throws instead of producing a
 * broken archive ({@link zipStoredLength}).
 */

export type ZipEntry = {
  /** The name inside the archive (`/`-separated; not empty, not repeated). */
  readonly name: string;
  readonly data: Uint8Array<ArrayBuffer>;
};

const LOCAL_HEADER_BYTES = 30;
const CENTRAL_HEADER_BYTES = 46;
const END_RECORD_BYTES = 22;

const LOCAL_HEADER_SIGNATURE = 0x04034b50;
const CENTRAL_HEADER_SIGNATURE = 0x02014b50;
const END_RECORD_SIGNATURE = 0x06054b50;

/** "version made by" / "needed to extract" = 2.0 (MS-DOS attributes — high byte 0). */
const ZIP_VERSION = 20;
/** The version for entries with a UTF-8 name (bit 11) = 6.3, the APPNOTE edition that defined it. */
const ZIP_VERSION_UTF8 = 63;
/** General-purpose flag bit 11: name and comment are UTF-8. */
const FLAG_UTF8 = 0x0800;
const METHOD_STORED = 0;

/** The largest values of the plain fields; `0xFFFF` / `0xFFFFFFFF` themselves read as ZIP64 markers. */
export const ZIP_MAX_ENTRIES = 0xfffe;
export const ZIP_MAX_OFFSET = 0xfffffffe;
const MAX_NAME_BYTES = 0xffff;

/** The CRC-32 lookup table (ISO 3309, reflected polynomial 0xEDB88320 — the one zip and PNG use). */
const crcTable = (): Uint32Array<ArrayBuffer> => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
};

/**
 * CRC-32 of `data`. Pass `table` when hashing many buffers (the table is built per call otherwise —
 * no module-level cache, the module stays free of mutable state).
 */
export const crc32 = (data: Uint8Array, table: Uint32Array = crcTable()): number => {
  let crc = 0xffffffff;
  for (const byte of data) crc = (table[(crc ^ byte) & 0xff] ?? 0) ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
};

/**
 * The byte length of the whole archive. Throws for an input that would need ZIP64 (more than
 * {@link ZIP_MAX_ENTRIES} entries, a central directory offset or size above {@link ZIP_MAX_OFFSET},
 * a name above 65,535 bytes). Every entry offset and size is at most the central directory offset,
 * so checking those two is enough.
 */
export const zipStoredLength = (
  entries: readonly { readonly nameBytes: number; readonly dataBytes: number }[],
): number => {
  if (entries.length > ZIP_MAX_ENTRIES) {
    throw new RangeError(`${entries.length} zip entries exceed ${ZIP_MAX_ENTRIES} (no ZIP64)`);
  }
  let centralOffset = 0;
  let centralBytes = 0;
  for (const { nameBytes, dataBytes } of entries) {
    if (nameBytes > MAX_NAME_BYTES) {
      throw new RangeError(`A zip entry name of ${nameBytes} bytes exceeds ${MAX_NAME_BYTES}`);
    }
    centralOffset += LOCAL_HEADER_BYTES + nameBytes + dataBytes;
    centralBytes += CENTRAL_HEADER_BYTES + nameBytes;
  }
  if (centralOffset > ZIP_MAX_OFFSET || centralBytes > ZIP_MAX_OFFSET) {
    throw new RangeError(
      `The zip central directory (offset ${centralOffset}, ${centralBytes} bytes) exceeds ${ZIP_MAX_OFFSET} (no ZIP64)`,
    );
  }
  return centralOffset + centralBytes + END_RECORD_BYTES;
};

/** MS-DOS date and time (local time, 2-second resolution); years outside 1980–2107 throw. */
const dosDateTime = (at: Date): { readonly date: number; readonly time: number } => {
  const year = at.getFullYear();
  if (year < 1980 || year > 2107) {
    throw new RangeError(`${at.toISOString()} cannot be written as an MS-DOS date (1980–2107)`);
  }
  return {
    date: ((year - 1980) << 9) | ((at.getMonth() + 1) << 5) | at.getDate(),
    time: (at.getHours() << 11) | (at.getMinutes() << 5) | (at.getSeconds() >> 1),
  };
};

/**
 * Builds a store-only zip. Returns the pieces that, concatenated in order, form the archive (the
 * entry data is not copied — a few hundred MB of PNGs are not held twice; pass the pieces straight
 * to `new Blob(parts)`). Every entry gets the timestamp `modified`.
 */
export const zipStored = (
  entries: readonly ZipEntry[],
  modified: Date,
): Uint8Array<ArrayBuffer>[] => {
  const encoder = new TextEncoder();
  const names = entries.map(({ name }) => encoder.encode(name));
  const seen = new Set<string>();
  for (const { name } of entries) {
    if (name === "") throw new Error("A zip entry name is empty");
    // A repeated name silently overwrites on extraction.
    if (seen.has(name)) throw new Error(`The zip entry name '${name}' is repeated`);
    seen.add(name);
  }
  // Check the sizes before computing any CRC (no CRC pass over an input that cannot be written).
  zipStoredLength(
    entries.map(({ data }, index) => ({
      nameBytes: names[index]?.length ?? 0,
      dataBytes: data.length,
    })),
  );
  const { date, time } = dosDateTime(modified);
  const table = crcTable();

  const parts: Uint8Array<ArrayBuffer>[] = [];
  const centrals: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;
  for (const [index, { data }] of entries.entries()) {
    const name = names[index] ?? new Uint8Array();
    const crc = crc32(data, table);
    const utf8 = name.some((byte) => byte >= 0x80);
    const version = utf8 ? ZIP_VERSION_UTF8 : ZIP_VERSION;
    const flags = utf8 ? FLAG_UTF8 : 0;

    const local = new Uint8Array(LOCAL_HEADER_BYTES + name.length);
    const localView = new DataView(local.buffer);
    localView.setUint32(0, LOCAL_HEADER_SIGNATURE, true);
    localView.setUint16(4, version, true);
    localView.setUint16(6, flags, true);
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
    centralView.setUint16(4, version, true);
    centralView.setUint16(6, version, true);
    centralView.setUint16(8, flags, true);
    centralView.setUint16(10, METHOD_STORED, true);
    centralView.setUint16(12, time, true);
    centralView.setUint16(14, date, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, data.length, true);
    centralView.setUint32(24, data.length, true);
    centralView.setUint16(28, name.length, true);
    // Extra length, comment length, start disk, internal and external attributes stay 0 (30–41).
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
  // Disk numbers stay 0 (4–7).
  endView.setUint16(8, entries.length, true);
  endView.setUint16(10, entries.length, true);
  endView.setUint32(12, centralBytes, true);
  endView.setUint32(16, offset, true);
  // Comment length stays 0 (20–21).
  return [...parts, ...centrals, end];
};
