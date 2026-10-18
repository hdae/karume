/**
 * The fixed header at the start of a container, and the part layout of the single form —
 * docs/container-v1.md §1 / §8 / §9.
 *
 * ```text
 *  0.. 4  magic          "KRMC" = model container (krm) / "KRGC" = graph container (krg)
 *  4.. 8  u32 LE         container version
 *  8..16  u64 LE         byte length of the graph descriptor
 * 16..24  u64 LE         byte length of the model descriptor (MUST be 0 in a krg)
 * ```
 *
 * Only the magic carries the kind; the descriptor has no `kind` field (the same fact is not kept in
 * two places — ADR 0108 decision 1). The kind can be rejected before a single byte of JSON is read.
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

/**
 * The one error for container-format violations (unsupported, inconsistent, corrupt). The entry
 * point for not approximating silently.
 */
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

/** Rounds `value` up to a multiple of `align`. */
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

/** Writes the header. Only a `modelDescriptorLength` of 0 may claim to be a `krg` (§1). */
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

/**
 * Reads the header from the first 24 bytes. Unknown magic / version, values beyond safe integers
 * and kind/length contradictions are rejected (§1).
 */
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
  for (const [label, length] of [
    ["グラフ記述", graphDescriptorLength],
    ["モデル記述", modelDescriptorLength],
  ] as const) {
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

/**
 * Length of part 0 (header + 2 documents). In the single form, unless the next part has length 0,
 * the bytes right after it are padded up to 64 B alignment (§1 / §8 — the rule is
 * {@link derivePartOffsets}).
 */
export const part0Length = (header: ContainerHeader): number =>
  HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength;

/**
 * Absolute offset of each part in the single form (§8).
 *
 * Writer and reader share this, so the rule for the padding between parts, which exists only in the
 * single form, lives in one place. **Padding MUST NOT be inserted before a part of length 0** —
 * otherwise a `krg` extracted from a `krm` with an empty const region and a `krg` written directly
 * differ in bytes (§3 / §9).
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
 * Assembles `krg` = `[header'][graph descriptor][padding][const region]` (§9).
 *
 * **Both** the side that writes a `krg` directly and the side that extracts it from a `krm` call
 * this. Going through the same function makes "extracted result = directly written krg" hold byte
 * for byte (the condition for deciding `krg` identity by content hash — ADR 0108 decision 4).
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
