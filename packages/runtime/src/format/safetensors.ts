// safetensors リーダ（読み取り専用・I/O を持たない — 呼び出し側が ArrayBuffer を渡す）。
// レイアウト: [u64 LE ヘッダ長][ヘッダ JSON][データ節]。data_offsets はデータ節先頭からの相対。

/** M0 で扱う格納 dtype と、エクスポータが出しうる整数／真偽テンソルのみ。未知は fail loudly。 */
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

const DTYPE_BYTES: Readonly<Record<SafetensorsDtype, number>> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  I8: 1,
  U8: 1,
  I32: 4,
  // U32 は意味論 bool の実表現（u32 の 0/1 — ADR 0009）。golden の io がこの形で書かれる。
  U32: 4,
  I64: 8,
  BOOL: 1,
};

export type TensorView = {
  readonly name: string;
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  /** buffer 先頭からの絶対 byte offset（データ節相対ではない）。 */
  readonly byteOffset: number;
  readonly byteLength: number;
};

export type SafetensorsFile = {
  readonly buffer: ArrayBuffer;
  readonly metadata: ReadonlyMap<string, string>;
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
    throw new SafetensorsError(`${where}: ${what} ${String(value)} が非負整数でない`);
  }
  return value;
};

const elementCount = (shape: readonly number[], where: string): number => {
  let count = 1;
  for (const dim of shape) {
    count *= dim;
    if (!Number.isSafeInteger(count)) {
      throw new SafetensorsError(`${where}: 要素数が安全整数を超える`);
    }
  }
  return count;
};

const parseDeclaration = (name: string, raw: unknown): DeclaredTensor => {
  const where = `tensor '${name}'`;
  if (!isPlainObject(raw)) throw new SafetensorsError(`${where}: ヘッダ項目がオブジェクトでない`);

  const dtype = raw["dtype"];
  if (typeof dtype !== "string" || !isKnownDtype(dtype)) {
    throw new SafetensorsError(`${where}: 未対応の dtype ${JSON.stringify(dtype)}`);
  }

  if (!Array.isArray(raw["shape"])) throw new SafetensorsError(`${where}: shape が配列でない`);
  const rawShape: readonly unknown[] = raw["shape"];
  const shape = rawShape.map((dim) => asIndex(dim, where, "shape 要素"));

  if (!Array.isArray(raw["data_offsets"])) {
    throw new SafetensorsError(`${where}: data_offsets が配列でない`);
  }
  const rawOffsets: readonly unknown[] = raw["data_offsets"];
  if (rawOffsets.length !== 2) {
    throw new SafetensorsError(
      `${where}: data_offsets の要素数が ${rawOffsets.length}（2 が必要）`,
    );
  }
  const begin = asIndex(rawOffsets[0], where, "data_offsets");
  const end = asIndex(rawOffsets[1], where, "data_offsets");
  if (end < begin) {
    throw new SafetensorsError(`${where}: data_offsets が逆転している [${begin}, ${end})`);
  }

  const expected = elementCount(shape, where) * DTYPE_BYTES[dtype];
  if (end - begin !== expected) {
    throw new SafetensorsError(
      `${where}: サイズ不一致 offsets=${end - begin} 期待=${expected}（${dtype} [${
        shape.join(",")
      }]）`,
    );
  }
  return { name, dtype, shape, begin, end };
};

const decodeHeader = (buffer: ArrayBuffer, headerLength: number): Record<string, unknown> => {
  const bytes = new Uint8Array(buffer, HEADER_LENGTH_BYTES, headerLength);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new SafetensorsError("ヘッダが UTF-8 として不正");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new SafetensorsError(`ヘッダ JSON を解析できない: ${String(cause)}`);
  }
  if (!isPlainObject(parsed)) throw new SafetensorsError("ヘッダ JSON がオブジェクトでない");
  return parsed;
};

/**
 * ファイル全体を 1 本の ArrayBuffer で受け取り、テンソル表を厳密に検査して view を返す。
 * view はコピーを作らず buffer 上の byteOffset / byteLength で参照する。
 */
export const parseSafetensors = (buffer: ArrayBuffer): SafetensorsFile => {
  if (buffer.byteLength < HEADER_LENGTH_BYTES) {
    throw new SafetensorsError(
      `ファイルが短すぎる: ${buffer.byteLength} バイト（ヘッダ長すら無い）`,
    );
  }
  const rawHeaderLength = new DataView(buffer).getBigUint64(0, true);
  if (rawHeaderLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new SafetensorsError(`ヘッダ長 ${rawHeaderLength} が安全整数を超える`);
  }
  const headerLength = Number(rawHeaderLength);
  const dataStart = HEADER_LENGTH_BYTES + headerLength;
  if (dataStart > buffer.byteLength) {
    throw new SafetensorsError(
      `ヘッダ長 ${headerLength} がファイル長 ${buffer.byteLength} を超える`,
    );
  }

  const header = decodeHeader(buffer, headerLength);
  const metadata = new Map<string, string>();
  const declared: DeclaredTensor[] = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") {
      if (!isPlainObject(value)) throw new SafetensorsError("__metadata__ がオブジェクトでない");
      for (const [key, item] of Object.entries(value)) {
        if (typeof item !== "string") {
          throw new SafetensorsError(`__metadata__.${key} が文字列でない`);
        }
        metadata.set(key, item);
      }
      continue;
    }
    declared.push(parseDeclaration(name, value));
  }

  const dataLength = buffer.byteLength - dataStart;
  const ordered = [...declared].sort((a, b) => a.begin - b.begin || a.end - b.end);
  const tensors = new Map<string, TensorView>();
  // データ節は宣言の集合で隙間なく覆われる MUST — 重複は同一バイトの二重意味、隙間と
  // 末尾の未使用領域は「読めていない宣言がある」ことの徴候なので、いずれも黙って通さない。
  let cursor = 0;
  for (const entry of ordered) {
    const where = `tensor '${entry.name}'`;
    if (entry.begin < cursor) {
      throw new SafetensorsError(
        `${where}: 先行テンソルと領域が重複 [${entry.begin}, ${entry.end}) 使用済み末尾=${cursor}`,
      );
    }
    if (entry.begin > cursor) {
      throw new SafetensorsError(
        `${where}: データ節に未使用領域 [${cursor}, ${entry.begin}) がある`,
      );
    }
    if (entry.end > dataLength) {
      throw new SafetensorsError(
        `${where}: データ節の範囲外 [${entry.begin}, ${entry.end}) データ節長=${dataLength}`,
      );
    }
    const align = DTYPE_BYTES[entry.dtype];
    if ((dataStart + entry.begin) % align !== 0) {
      // コピーを作らず typed array view を張る前提が崩れるため受理しない。
      throw new SafetensorsError(
        `${where}: 絶対 offset ${
          dataStart + entry.begin
        } が ${entry.dtype} の要素サイズ ${align} に整列していない`,
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
    throw new SafetensorsError(`データ節末尾に未使用領域が ${dataLength - cursor} バイトある`);
  }
  return { buffer, metadata, tensors };
};

/** テンソルの生バイト（コピーしない）。 */
export const tensorBytes = (file: SafetensorsFile, view: TensorView): Uint8Array<ArrayBuffer> =>
  new Uint8Array(file.buffer, view.byteOffset, view.byteLength);
