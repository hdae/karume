// safetensors reader (read-only, no I/O — the caller passes an ArrayBuffer).
// Layout: [u64 LE header length][header JSON][data section]. data_offsets are relative to the
// start of the data section.

/**
 * Storage dtypes an auxiliary-asset safetensors file may have (a subset of the official safetensors
 * vocabulary). Unknown ones fail loudly — **dialect dtypes such as packed 4-bit / 2-bit are not
 * accepted** (the distribution has moved to the `krm` container, and quantized storage is handled by
 * the container's codecs).
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
 * dtype → bytes per element. The byte-length check is an exact match of `numel × bytes`, and the
 * byte alignment required at a tensor's start is this value too (on the premise that a typed array
 * view is created without a copy — a view's element alignment = the element size).
 *
 * NOTE: the auxiliary-asset vocabulary has no packed types (the 4-bit / 2-bit packed kind, where the
 * element size and the alignment requirement split), so one table suffices. The day packed types are
 * accepted, the alignment table is split again (the three-surface split of ADR 0069 decision 2 —
 * superseded for weights by ADR 0108 decisions 10–13).
 */
const DTYPE_BYTES: Readonly<Record<SafetensorsDtype, number>> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  I8: 1,
  U8: 1,
  I32: 4,
  // U32 is the physical form of semantic bool (u32 0/1 — ADR 0009). Golden io is written in this
  // form.
  U32: 4,
  I64: 8,
  BOOL: 1,
};

export type TensorView = {
  readonly name: string;
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  /** Absolute byte offset from the start of the file (not relative to the data section). */
  readonly byteOffset: number;
  readonly byteLength: number;
};

export type SafetensorsFile = {
  readonly buffer: ArrayBuffer;
  readonly metadata: ReadonlyMap<string, string>;
  readonly tensors: ReadonlyMap<string, TensorView>;
};

/**
 * The result of parsing only the header (it holds no bytes of the data section).
 *
 * The checks of the tensor table (coverage, alignment, tail) need only the declarations and the file
 * length, so a caller that reads ranges can get this table without holding the whole file.
 */
export type SafetensorsHeader = {
  /**
   * Start of the data section (absolute byte offset from the start of the file = 8 + header length).
   */
  readonly dataStart: number;
  readonly metadata: ReadonlyMap<string, string>;
  /** byteOffset is absolute from the start of the file, as in {@link TensorView}. */
  readonly tensors: ReadonlyMap<string, TensorView>;
};

export class SafetensorsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SafetensorsError";
  }
}

const HEADER_LENGTH_BYTES = 8;

type DeclaredTensor = {
  readonly name: string;
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  readonly begin: number;
  readonly end: number;
};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isKnownDtype = (dtype: string): dtype is SafetensorsDtype =>
  Object.hasOwn(DTYPE_BYTES, dtype);

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

/**
 * Byte length of a tensor determined by its declaration (dtype + element count) (used by the "size
 * mismatch" check below).
 */
const declaredByteLength = (dtype: SafetensorsDtype, count: number): number =>
  count * DTYPE_BYTES[dtype];

const parseDeclaration = (name: string, raw: unknown): DeclaredTensor => {
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

  const count = elementCount(shape, where);
  const expected = declaredByteLength(dtype, count);
  if (end - begin !== expected) {
    throw new SafetensorsError(
      `${where}: size mismatch offsets=${end - begin} expected=${expected} (${dtype} [${shape.join(
        ",",
      )}])`,
    );
  }
  return { name, dtype, shape, begin, end };
};

const decodeHeader = (
  prefix: Uint8Array<ArrayBuffer>,
  headerLength: number,
): Record<string, unknown> => {
  const bytes = prefix.subarray(HEADER_LENGTH_BYTES, HEADER_LENGTH_BYTES + headerLength);
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

// Distinguished with `ArrayBuffer.isView` (`instanceof ArrayBuffer` is false across realms, so an
// ArrayBuffer from another realm would be treated as a view and fail with a TypeError instead of a
// `SafetensorsError`).
const asPrefixBytes = (prefix: Uint8Array<ArrayBuffer> | ArrayBuffer): Uint8Array<ArrayBuffer> =>
  ArrayBuffer.isView(prefix) ? prefix : new Uint8Array(prefix);

/**
 * Reads the header length (the byte count of the header JSON) from the first 8 bytes.
 *
 * The entry point for a range-reading caller to take the two steps "read 8 bytes → learn N → read
 * 8+N bytes again". prefix is a contiguous range from the start of the file and needs at least 8
 * bytes.
 *
 * MUST: the second read is **clamped to the known file length** — this function does not receive the
 * file length, so it can only see values beyond safe integers; {@link parseSafetensorsHeader} is the
 * only place that checks whether the length is plausible. Using a corrupt header length (e.g. 1 TiB)
 * as the read length makes the allocation or the read fail first with a RangeError / OOM, never
 * reaching `SafetensorsError`. The caller always has the file length (a required argument of
 * {@link parseSafetensorsHeader}).
 */
export const safetensorsHeaderLength = (prefix: Uint8Array<ArrayBuffer> | ArrayBuffer): number => {
  const bytes = asPrefixBytes(prefix);
  if (bytes.byteLength < HEADER_LENGTH_BYTES) {
    throw new SafetensorsError(
      `Reading the header length needs the first ${HEADER_LENGTH_BYTES} bytes (prefix is ${bytes.byteLength} bytes)`,
    );
  }
  const raw = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true);
  if (raw > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new SafetensorsError(`Header length ${raw} exceeds the safe integer range`);
  }
  return Number(raw);
};

/**
 * Parses the header from the range at the start of the file (at least 8 + header length bytes) and
 * **the total file length**.
 *
 * No byte of the data section needs to be read because the coverage, alignment and tail checks close
 * over just "the set of declarations" and "the file length". A caller that reads an asset by ranges
 * can pass this table's byteOffset / byteLength straight to its reader.
 *
 * When the prefix is too short it fails loudly with the needed length in the message — the caller
 * MUST be able to decide from the error how much more to read (zero-filling on its own and parsing
 * would silently accept a corrupt header).
 */
export const parseSafetensorsHeader = (
  prefix: Uint8Array<ArrayBuffer> | ArrayBuffer,
  fileLength: number,
): SafetensorsHeader => {
  if (!Number.isInteger(fileLength) || fileLength < 0) {
    throw new SafetensorsError(`File length ${fileLength} is not a non-negative integer`);
  }
  if (fileLength < HEADER_LENGTH_BYTES) {
    throw new SafetensorsError(`File is too short: ${fileLength} bytes (not even a header length)`);
  }
  const bytes = asPrefixBytes(prefix);
  const headerLength = safetensorsHeaderLength(bytes);
  const dataStart = HEADER_LENGTH_BYTES + headerLength;
  if (dataStart > fileLength) {
    throw new SafetensorsError(
      `Header length ${headerLength} exceeds the file length ${fileLength}`,
    );
  }
  if (bytes.byteLength < dataStart) {
    throw new SafetensorsError(
      `Decoding the header needs the first ${dataStart} bytes (prefix is ${bytes.byteLength} bytes)`,
    );
  }
  const header = decodeHeader(bytes, headerLength);
  const metadata = new Map<string, string>();
  const declared: DeclaredTensor[] = [];
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
  const tensors = new Map<string, TensorView>();
  // The data section MUST be covered by the set of declarations without gaps — an overlap gives the
  // same bytes two meanings, and gaps and unused space at the tail are signs that "some declaration
  // was not read", so none of them is let through silently.
  let cursor = 0;
  for (const entry of ordered) {
    const where = `tensor '${entry.name}'`;
    if (entry.begin < cursor) {
      throw new SafetensorsError(
        `${where}: range overlaps a preceding tensor [${entry.begin}, ${entry.end}) used end=${cursor}`,
      );
    }
    if (entry.begin > cursor) {
      throw new SafetensorsError(
        `${where}: unused range [${cursor}, ${entry.begin}) in the data section`,
      );
    }
    if (entry.end > dataLength) {
      throw new SafetensorsError(
        `${where}: outside the data section [${entry.begin}, ${entry.end}) data section length=${dataLength}`,
      );
    }
    const align = DTYPE_BYTES[entry.dtype];
    if ((dataStart + entry.begin) % align !== 0) {
      // Not accepted: it breaks the premise that a typed array view is created without a copy.
      throw new SafetensorsError(
        `${where}: absolute offset ${
          dataStart + entry.begin
        } is not aligned to the alignment unit of ${entry.dtype}, ${align} bytes`,
      );
    }
    cursor = entry.end;
    tensors.set(entry.name, {
      name: entry.name,
      dtype: entry.dtype,
      shape: entry.shape,
      byteOffset: dataStart + entry.begin,
      byteLength: entry.end - entry.begin,
    });
  }
  if (cursor !== dataLength) {
    throw new SafetensorsError(
      `Unused range of ${dataLength - cursor} bytes at the end of the data section`,
    );
  }
  return { dataStart, metadata, tensors };
};

/**
 * Takes the whole file as one ArrayBuffer, checks the tensor table strictly and returns views.
 * The views make no copy; they refer to byteOffset / byteLength on the buffer.
 *
 * The readers of this reader are auxiliary assets (rope tables, style tables, mel bases, golden io
 * and so on); Session weights are read by the `krm` container path.
 *
 * The whole buffer is treated as the actual file length — slack at the tail of a buffer longer than
 * the file is rejected as "unused space at the tail of the data section".
 *
 * The checks themselves MUST live only in the one implementation, {@link parseSafetensorsHeader} —
 * so that the whole-file path and the range-read path accept the same forms.
 */
export const parseSafetensors = (buffer: ArrayBuffer): SafetensorsFile => {
  const { metadata, tensors } = parseSafetensorsHeader(buffer, buffer.byteLength);
  return { buffer, metadata, tensors };
};

/**
 * The raw bytes of a tensor (no copy).
 *
 * NOTE: no per-dtype TypedArray is returned. The form a caller wants (reinterpreting as f32 / u32 /
 * bf16) differs per asset, so this function hands over "raw bytes + dtype + logical shape
 * ({@link TensorView})" and leaves creating the view to the caller.
 */
export const tensorBytes = (file: SafetensorsFile, view: TensorView): Uint8Array<ArrayBuffer> =>
  new Uint8Array(file.buffer, view.byteOffset, view.byteLength);
