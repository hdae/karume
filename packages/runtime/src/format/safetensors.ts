// safetensors リーダ（読み取り専用・I/O を持たない — 呼び出し側が ArrayBuffer を渡す）。
// レイアウト: [u64 LE ヘッダ長][ヘッダ JSON][データ節]。data_offsets はデータ節先頭からの相対。

/** M0 で扱う格納 dtype と、エクスポータが出しうる整数／真偽テンソルのみ。未知は fail loudly。 */
export type SafetensorsDtype =
  | "F32"
  | "F16"
  | "BF16"
  | "I8"
  | "I4"
  | "U8"
  | "I32"
  | "U32"
  | "I64"
  | "BOOL";

/**
 * dtype → 1 要素の **bit** 数（サイズ表）。バイト長の検証は `numel × bits / 8` の厳密一致で、
 * 8 で割り切れない bit 総量は受理しない。
 *
 * MUST: 整列表（{@link DTYPE_ALIGN}）と分けて持つ（ADR 0069 決定 2 の 3 面分離）。I4 は
 * 1 バイトに 2 要素を詰めるので「要素サイズ = 整列」が成り立たず、1 本の表で両方を賄うと
 * 4bit 格納でどちらかが必ず壊れる。
 */
const DTYPE_BITS: Readonly<Record<SafetensorsDtype, number>> = {
  F32: 32,
  F16: 16,
  BF16: 16,
  I8: 8,
  // packed 4bit（ADR 0069 決定 2）。shape は論理形のままで、バイト数だけが bit 幅から決まる。
  I4: 4,
  U8: 8,
  I32: 32,
  // U32 は意味論 bool の実表現（u32 の 0/1 — ADR 0009）。golden の io がこの形で書かれる。
  U32: 32,
  I64: 64,
  BOOL: 8,
};

/**
 * dtype → テンソル**先頭**に要求する byte 整列（整列表）。要素サイズと一致するのは要素整列の
 * 概念を持つ dtype だけで、**I4 は 4**（要素境界ではなく、展開カーネルが `array<u32>` として
 * 束縛する都合 — ADR 0069 決定 2）。
 */
const DTYPE_ALIGN: Readonly<Record<SafetensorsDtype, number>> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  I8: 1,
  I4: 4,
  U8: 1,
  I32: 4,
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

const isKnownDtype = (dtype: string): dtype is SafetensorsDtype => Object.hasOwn(DTYPE_BITS, dtype);

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

  const count = elementCount(shape, where);
  const bits = count * DTYPE_BITS[dtype];
  // MUST: bit 総量が byte 境界に乗らない形（I4 の要素数が奇数）は fail loudly。末尾要素が
  // 半バイトだけ突き出すので、テンソルの長さが宣言から一意に決まらない。
  if (bits % 8 !== 0) {
    throw new SafetensorsError(
      `${where}: ${dtype}（1 要素 ${
        DTYPE_BITS[dtype]
      }bit）の要素数 ${count} が奇数で byte 境界に乗らない`,
    );
  }
  const expected = bits / 8;
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
    const align = DTYPE_ALIGN[entry.dtype];
    if ((dataStart + entry.begin) % align !== 0) {
      // コピーを作らず typed array view を張る前提（I4 は u32 として束縛する前提）が
      // 崩れるため受理しない。
      throw new SafetensorsError(
        `${where}: 絶対 offset ${
          dataStart + entry.begin
        } が ${entry.dtype} の整列単位 ${align} バイトに整列していない`,
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

/**
 * テンソルの生バイト（コピーしない）。
 *
 * NOTE: dtype 別の TypedArray は返さない。I4 に対応する TypedArray は存在しないが、この面は
 * 最初から「raw バイト + 論理 numel（{@link TensorView.shape}）」なので 4bit 格納でも
 * 表現が足りている（ADR 0069 決定 2 の 3 面目）。
 */
export const tensorBytes = (file: SafetensorsFile, view: TensorView): Uint8Array<ArrayBuffer> =>
  new Uint8Array(file.buffer, view.byteOffset, view.byteLength);
