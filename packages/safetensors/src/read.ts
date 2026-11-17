/**
 * The safetensors reader: a file opened from its 8-byte length and its header alone, each tensor
 * read later with one ranged read of exactly its bytes. The input is a `Blob` (a browser `File` is
 * one) or any ranged reader, so a file larger than one ArrayBuffer can hold is never read whole.
 *
 * Layout: [u64 LE header length N][N bytes of UTF-8 JSON][data section]. `data_offsets` are
 * relative to the start of the data section.
 *
 * NOTE: the checks are those of the runtime's whole-buffer reader
 * (`packages/runtime/src/format/safetensors.ts`), stated here again until the two readers are one;
 * this reader adds the reference implementation's bound on the header length.
 *
 * SPEC: docs/spec/safetensors.md#the-checks
 */

/**
 * The element types accepted: the subset of the safetensors vocabulary the runtime's reader
 * accepts. Any other is refused.
 */
export type SafetensorsDtype =
  | "F32"
  | "F16"
  | "BF16"
  | "I8"
  | "U8"
  | "I32"
  | "U32"
  | "I64"
  | "BOOL";

/**
 * Bytes per element. A tensor's byte range is exactly `numel × this` long, and its first byte's
 * position in the file is a multiple of it (so that a typed-array view of the bytes needs no copy).
 */
const ELEMENT_BYTES: Readonly<Record<SafetensorsDtype, number>> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  I8: 1,
  U8: 1,
  I32: 4,
  U32: 4,
  I64: 8,
  BOOL: 1,
};

/** The largest header accepted, in bytes (the bound of the reference implementation). */
const MAX_HEADER_BYTES = 100_000_000;

const LENGTH_BYTES = 8;

/** A file read by ranges. */
export type RangedReader = {
  /** The file's length in bytes. */
  readonly size: number;
  /** Bytes `[offset, offset + length)` of the file. */
  read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/** What a safetensors file is opened from. */
export type SafetensorsSource = Blob | RangedReader;

/** One tensor of the file, as its header declares it. */
export type SafetensorsTensor = {
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  /** `numel × element size`. */
  readonly byteLength: number;
  /**
   * The tensor's bytes as stored (little endian, row-major): one ranged read of exactly
   * `byteLength` bytes, on every call. Nothing is kept.
   */
  readonly read: () => Promise<Uint8Array<ArrayBuffer>>;
};

export type SafetensorsFile = {
  /** `__metadata__` (empty when the header has none). */
  readonly metadata: ReadonlyMap<string, string>;
  /** Tensor name → tensor, in the order of the data section. */
  readonly tensors: ReadonlyMap<string, SafetensorsTensor>;
};

/** Every refusal of the reader. A refusal about one tensor names it. */
export class SafetensorsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SafetensorsError";
  }
}

type Declared = {
  readonly name: string;
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  readonly begin: number;
  readonly end: number;
};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isKnownDtype = (dtype: string): dtype is SafetensorsDtype =>
  Object.hasOwn(ELEMENT_BYTES, dtype);

const asIndex = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new SafetensorsError(`${where}: ${what} ${String(value)} is not a non-negative integer`);
  }
  return value;
};

const elementCount = (shape: readonly number[], where: string): number => {
  let count = 1;
  for (const dim of shape) {
    count *= dim;
    if (!Number.isSafeInteger(count)) {
      throw new SafetensorsError(`${where}: element count exceeds the safe integer range`);
    }
  }
  return count;
};

const parseDeclaration = (name: string, raw: unknown): Declared => {
  const where = `tensor '${name}'`;
  if (!isPlainObject(raw)) throw new SafetensorsError(`${where}: header entry is not an object`);

  const dtype = raw["dtype"];
  if (typeof dtype !== "string" || !isKnownDtype(dtype)) {
    throw new SafetensorsError(`${where}: unsupported dtype ${JSON.stringify(dtype)}`);
  }

  if (!Array.isArray(raw["shape"])) throw new SafetensorsError(`${where}: shape is not an array`);
  const rawShape: readonly unknown[] = raw["shape"];
  const shape = rawShape.map((dim) => asIndex(dim, where, "shape element"));

  if (!Array.isArray(raw["data_offsets"])) {
    throw new SafetensorsError(`${where}: data_offsets is not an array`);
  }
  const rawOffsets: readonly unknown[] = raw["data_offsets"];
  if (rawOffsets.length !== 2) {
    throw new SafetensorsError(
      `${where}: data_offsets has ${rawOffsets.length} elements (2 required)`,
    );
  }
  const begin = asIndex(rawOffsets[0], where, "data_offsets");
  const end = asIndex(rawOffsets[1], where, "data_offsets");
  if (end < begin) {
    throw new SafetensorsError(`${where}: data_offsets are reversed [${begin}, ${end})`);
  }

  const expected = elementCount(shape, where) * ELEMENT_BYTES[dtype];
  if (end - begin !== expected) {
    throw new SafetensorsError(
      `${where}: size mismatch offsets=${end - begin} expected=${expected} (${dtype} [${shape.join(
        ",",
      )}])`,
    );
  }
  return { name, dtype, shape, begin, end };
};

const decodeHeader = (bytes: Uint8Array<ArrayBuffer>): Record<string, unknown> => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new SafetensorsError("Header is invalid UTF-8");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new SafetensorsError(`Cannot parse the header JSON: ${String(cause)}`);
  }
  if (!isPlainObject(parsed)) throw new SafetensorsError("Header JSON is not an object");
  return parsed;
};

/**
 * A `Blob` is read with `slice(…).arrayBuffer()`; anything with a `read` is a ranged reader (told
 * apart by shape, not `instanceof`, so a `Blob` of another realm is still a `Blob`).
 */
const rangedReader = (source: SafetensorsSource): RangedReader => {
  if ("read" in source) {
    if (!Number.isSafeInteger(source.size) || source.size < 0) {
      throw new SafetensorsError(`File length ${source.size} is not a non-negative integer`);
    }
    return source;
  }
  return {
    size: source.size,
    read: async (offset, length) =>
      new Uint8Array(await source.slice(offset, offset + length).arrayBuffer()),
  };
};

const readExactly = async (
  reader: RangedReader,
  offset: number,
  length: number,
  what: string,
): Promise<Uint8Array<ArrayBuffer>> => {
  const bytes = await reader.read(offset, length);
  if (bytes.byteLength !== length) {
    throw new SafetensorsError(
      `${what}: the read of [${offset}, ${offset + length}) gave ${bytes.byteLength} bytes`,
    );
  }
  return bytes;
};

/**
 * Opens a safetensors file: two reads (the 8-byte length, then the header) and every check of the
 * tensor table, which needs only the declarations and the file length. No byte of the data
 * section is read until a tensor's `read` is called.
 *
 * SPEC: docs/spec/safetensors.md#opening-a-file
 */
export const openSafetensors = async (source: SafetensorsSource): Promise<SafetensorsFile> => {
  const reader = rangedReader(source);
  const fileLength = reader.size;
  if (fileLength < LENGTH_BYTES) {
    throw new SafetensorsError(`File is too short: ${fileLength} bytes (not even a header length)`);
  }
  const prefix = await readExactly(reader, 0, LENGTH_BYTES, "The header length");
  const declaredLength = new DataView(prefix.buffer, prefix.byteOffset, LENGTH_BYTES).getBigUint64(
    0,
    true,
  );
  // MUST: bounded before it is used as a read length — a corrupt length would otherwise fail as an
  // allocation error instead of a refusal.
  if (declaredLength > BigInt(MAX_HEADER_BYTES)) {
    throw new SafetensorsError(
      `Header length ${declaredLength} exceeds the limit of ${MAX_HEADER_BYTES} bytes`,
    );
  }
  const headerLength = Number(declaredLength);
  const dataStart = LENGTH_BYTES + headerLength;
  if (dataStart > fileLength) {
    throw new SafetensorsError(
      `Header length ${headerLength} exceeds the file length ${fileLength}`,
    );
  }
  const header = decodeHeader(await readExactly(reader, LENGTH_BYTES, headerLength, "The header"));

  const metadata = new Map<string, string>();
  const declared: Declared[] = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") {
      if (!isPlainObject(value)) throw new SafetensorsError("__metadata__ is not an object");
      for (const [key, item] of Object.entries(value)) {
        if (typeof item !== "string") {
          throw new SafetensorsError(`__metadata__.${key} is not a string`);
        }
        metadata.set(key, item);
      }
      continue;
    }
    declared.push(parseDeclaration(name, value));
  }

  const dataLength = fileLength - dataStart;
  const ordered = [...declared].sort((a, b) => a.begin - b.begin || a.end - b.end);
  const tensors = new Map<string, SafetensorsTensor>();
  // The declarations MUST cover the data section without gaps or overlaps: an overlap gives the
  // same bytes two meanings, and a gap or unused tail is a sign that a declaration was not read.
  let previous: Declared | undefined;
  let cursor = 0;
  for (const entry of ordered) {
    const where = `tensor '${entry.name}'`;
    if (entry.begin < cursor && previous !== undefined) {
      throw new SafetensorsError(
        `${where}: range [${entry.begin}, ${entry.end}) overlaps tensor '${previous.name}' [${previous.begin}, ${previous.end})`,
      );
    }
    if (entry.begin > cursor) {
      throw new SafetensorsError(
        `${where}: unused range [${cursor}, ${entry.begin}) in the data section before it${
          previous === undefined ? "" : ` (after tensor '${previous.name}')`
        }`,
      );
    }
    if (entry.end > dataLength) {
      throw new SafetensorsError(
        `${where}: outside the data section [${entry.begin}, ${entry.end}) data section length=${dataLength}`,
      );
    }
    const start = dataStart + entry.begin;
    const align = ELEMENT_BYTES[entry.dtype];
    if (start % align !== 0) {
      throw new SafetensorsError(
        `${where}: absolute offset ${start} is not aligned to the alignment unit of ${entry.dtype}, ${align} bytes`,
      );
    }
    previous = entry;
    cursor = entry.end;
    const byteLength = entry.end - entry.begin;
    tensors.set(entry.name, {
      dtype: entry.dtype,
      shape: entry.shape,
      byteLength,
      read: () => readExactly(reader, start, byteLength, where),
    });
  }
  if (cursor !== dataLength) {
    throw new SafetensorsError(
      `Unused range of ${dataLength - cursor} bytes at the end of the data section${
        previous === undefined
          ? " (the header declares no tensor)"
          : ` after tensor '${previous.name}'`
      }`,
    );
  }
  return { metadata, tensors };
};
