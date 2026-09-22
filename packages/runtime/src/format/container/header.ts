/**
 * コンテナ先頭の固定ヘッダと、単一形の part 配置 — docs/container-v1.md §1 / §8 / §9。
 *
 * ```text
 *  0.. 4  magic          "KRMC" = モデル容器（krm） / "KRGC" = グラフ容器（krg）
 *  4.. 8  u32 LE         コンテナ版
 *  8..16  u64 LE         グラフ記述のバイト長
 * 16..24  u64 LE         モデル記述のバイト長（krg では 0 MUST）
 * ```
 *
 * 種別を持つのは magic だけで、descriptor の中に `kind` 欄は無い（同じ事実を 2 か所に持たせない —
 * ADR 0108 決定 1）。JSON を 1 バイトも読む前に種別で弾ける。
 */

import {
  BLOCK_START_ALIGN,
  CONTAINER_VERSION,
  HEADER_BYTES,
  MAGIC_GRAPH,
  MAGIC_MODEL,
  MAX_DESCRIPTOR_BYTES,
  PAD_BYTE,
} from "./limits.ts";

/** コンテナ形式の違反（未対応・不整合・破損）を 1 本に集める例外。黙って近似しないための入口。 */
export class ContainerFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContainerFormatError";
  }
}

export type ContainerKind = "model" | "graph";

export type ContainerHeader = {
  readonly kind: ContainerKind;
  readonly version: number;
  readonly graphDescriptorLength: number;
  readonly modelDescriptorLength: number;
};

/** `value` を `align` の倍数へ切り上げる。 */
export const alignUp = (value: number, align: number): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ContainerFormatError(`整列対象が非負の安全整数でない: ${value}`);
  }
  return Math.ceil(value / align) * align;
};

const magicBytes = (magic: string): Uint8Array<ArrayBuffer> => new TextEncoder().encode(magic);

const matchesMagic = (bytes: Uint8Array<ArrayBuffer>, magic: string): boolean =>
  magicBytes(magic).every((byte, index) => bytes[index] === byte);

const readSafeUint64 = (view: DataView, offset: number, label: string): number => {
  const raw = view.getBigUint64(offset, true);
  if (raw > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new ContainerFormatError(`${label}が安全整数を超える: ${raw}`);
  }
  return Number(raw);
};

/** ヘッダを書き出す。`modelDescriptorLength` が 0 のときだけ `krg` を名乗れる（§1）。 */
export const writeHeader = (header: ContainerHeader): Uint8Array<ArrayBuffer> => {
  if (header.kind === "graph" && header.modelDescriptorLength !== 0) {
    throw new ContainerFormatError("krg はモデル記述を持てない（長さは 0 MUST）");
  }
  if (header.kind === "model" && header.modelDescriptorLength === 0) {
    throw new ContainerFormatError("krm はモデル記述を必ず持つ（長さ 0 は krg の形）");
  }
  if (header.graphDescriptorLength === 0) {
    throw new ContainerFormatError("グラフ記述の長さが 0");
  }
  const bytes = new Uint8Array(new ArrayBuffer(HEADER_BYTES));
  bytes.set(magicBytes(header.kind === "model" ? MAGIC_MODEL : MAGIC_GRAPH), 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(4, header.version, true);
  view.setBigUint64(8, BigInt(header.graphDescriptorLength), true);
  view.setBigUint64(16, BigInt(header.modelDescriptorLength), true);
  return bytes;
};

/** 先頭 24 バイトからヘッダを読む。未知の magic / 版・安全整数超過・種別と長さの矛盾は拒否（§1）。 */
export const readHeader = (bytes: Uint8Array<ArrayBuffer>): ContainerHeader => {
  if (bytes.byteLength < HEADER_BYTES) {
    throw new ContainerFormatError(
      `ヘッダに ${HEADER_BYTES} バイト必要だが ${bytes.byteLength} バイトしかない`,
    );
  }
  const kind: ContainerKind | undefined = matchesMagic(bytes, MAGIC_MODEL)
    ? "model"
    : matchesMagic(bytes, MAGIC_GRAPH)
    ? "graph"
    : undefined;
  if (kind === undefined) {
    const seen = Array.from(bytes.subarray(0, 4), (b) => b.toString(16).padStart(2, "0")).join(" ");
    throw new ContainerFormatError(
      `未知の magic: ${seen}（Karume のコンテナは '${MAGIC_MODEL}' / '${MAGIC_GRAPH}'）`,
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== CONTAINER_VERSION) {
    throw new ContainerFormatError(`未対応のコンテナ版 ${version}（対応は ${CONTAINER_VERSION}）`);
  }
  const graphDescriptorLength = readSafeUint64(view, 8, "グラフ記述の長さ");
  const modelDescriptorLength = readSafeUint64(view, 16, "モデル記述の長さ");
  if (graphDescriptorLength === 0) throw new ContainerFormatError("グラフ記述の長さが 0");
  for (
    const [label, length] of [["グラフ記述", graphDescriptorLength], [
      "モデル記述",
      modelDescriptorLength,
    ]] as const
  ) {
    if (length > MAX_DESCRIPTOR_BYTES) {
      throw new ContainerFormatError(
        `${label}の長さ ${length} が上限 ${MAX_DESCRIPTOR_BYTES} を超える`,
      );
    }
  }
  if (kind === "graph" && modelDescriptorLength !== 0) {
    throw new ContainerFormatError(`krg なのにモデル記述の長さが ${modelDescriptorLength}`);
  }
  if (kind === "model" && modelDescriptorLength === 0) {
    throw new ContainerFormatError("krm なのにモデル記述の長さが 0");
  }
  return { kind, version, graphDescriptorLength, modelDescriptorLength };
};

/** part 0（ヘッダ + 2 文書）の長さ。単一形ではこの直後を 64 B 整列まで詰める（§1 / §8）。 */
export const part0Length = (header: ContainerHeader): number =>
  HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength;

/**
 * 単一形で各 part が置かれる絶対 offset（§8）。
 *
 * 書き手と読み手がここを共有するので、単一形にだけ存在する part 間の詰め物の規則が 1 箇所に
 * 閉じる。**長さ 0 の part の前に詰め物を挟まない** MUST — 挟むと、const が空の `krm` から抜いた
 * `krg` と直接書いた `krg` がバイトでずれる（§3 / §9）。
 */
export const derivePartOffsets = (
  part0Bytes: number,
  partLengths: readonly number[],
): { readonly offsets: readonly number[]; readonly totalLength: number } => {
  const offsets: number[] = [0];
  let cursor = part0Bytes;
  for (const length of partLengths) {
    const start = length === 0 ? cursor : alignUp(cursor, BLOCK_START_ALIGN);
    offsets.push(start);
    cursor = start + length;
  }
  return { offsets, totalLength: cursor };
};

/**
 * `krg` = `[ヘッダ'][グラフ記述][詰め物][const 領域]` を組み立てる（§9）。
 *
 * `krg` を直接書く側と、`krm` から抜き出す側の**両方**がこれを呼ぶ。同じ関数を通るので
 * 「抽出結果 = 直接書いた krg」がバイト単位で成り立つ（`krg` の同一性を内容ハッシュで判定する
 * 条件 — ADR 0108 決定 4）。
 */
export const assembleGraphContainer = (
  graphDescriptorBytes: Uint8Array<ArrayBuffer>,
  constRegionBytes: Uint8Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => {
  const header = writeHeader({
    kind: "graph",
    version: CONTAINER_VERSION,
    graphDescriptorLength: graphDescriptorBytes.byteLength,
    modelDescriptorLength: 0,
  });
  const layout = derivePartOffsets(HEADER_BYTES + graphDescriptorBytes.byteLength, [
    constRegionBytes.byteLength,
  ]);
  const out = new Uint8Array(new ArrayBuffer(layout.totalLength));
  out.fill(PAD_BYTE);
  out.set(header, 0);
  out.set(graphDescriptorBytes, HEADER_BYTES);
  out.set(constRegionBytes, layout.offsets[1]);
  return out;
};
