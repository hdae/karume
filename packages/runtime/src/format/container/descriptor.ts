/**
 * 2 文書 descriptor（グラフ記述 / モデル記述）の型・parse・宣言検査・正準直列化 —
 * docs/container-v1.md §2 / §4 / §5 / §6 / §7。
 *
 * - **グラフ記述** = `graphs`（IR v2）+ const 領域の目次と束縛 + `capabilities`。`krm` と `krg` で
 *   **バイト単位に同一**（`krg` のバイトコピー抽出の条件 — §9）。
 * - **モデル記述** = 束縛表 + 重み / 資産の目次 + parts の内部配置 + provenance。
 *
 * ここで掛けるのは**宣言だけで決まる検査**（重みを 1 バイトも取る前に効く — §7）。宣言 shape と
 * encoding から決まる payload 長・piece の被覆・group 長の規則は合流層（`bind.ts`）が持つ。
 *
 * descriptor は自分の正しさを証明できない: 2 文書のバイト列は**外側の期待 hash + 長さ**で先に
 * 検証してから parse する（`open.ts`）。
 */

import {
  CODEC_NAMES,
  codecEntry,
  type CodecName,
  isCodecName,
  SCALE_DTYPES,
  type ScaleDtype,
} from "./codecs.ts";
import { alignUp, ContainerFormatError } from "./header.ts";
import {
  decodeJsonDocument,
  encodeJsonBytes,
  isJsonObject,
  type JsonObject,
  sortedByCodePoints,
  sortedObject,
} from "./json.ts";
import {
  BLOCK_ID_PATTERN,
  BLOCK_MAX_BYTES,
  BLOCK_START_ALIGN,
  BLOCK_TAIL_ALIGN,
  GRAPH_NAME_PATTERN,
  MAX_BLOCKS,
  MAX_DESCRIPTOR_BYTES,
  MAX_GRAPHS,
  MAX_PARTS,
  PART_MAX_BYTES,
  SHA256_HEX_PATTERN,
} from "./limits.ts";
import {
  canonicalIrDocument,
  type IrDeclaration,
  IrError,
  parseIrDeclarationValue,
} from "../ir.ts";

// ---------------------------------------------------------------------------
// 型
// ---------------------------------------------------------------------------

declare const sha256HexBrand: unique symbol;

/** 小文字 16 進 64 文字の sha256（§0）。 */
export type Sha256Hex = string & { readonly [sha256HexBrand]: true };

export type EncodingPacking = {
  readonly blockElements: number;
  readonly blockBytes: number;
  readonly alignBytes: number;
};

/**
 * 格納の宣言（§6.1）。bit 数は `packing` からの派生値なので欄を持たない。`rowAxis` / `groupSize` /
 * `scale` は量子化 codec（台帳の `scale: "required"`）でのみ書き、非量子化では書けない。
 */
export type Encoding = {
  readonly codec: CodecName;
  readonly packing: EncodingPacking;
  readonly rowAxis?: 0 | 1;
  readonly groupSize?: number;
  readonly scale?: { readonly block: string; readonly dtype: ScaleDtype };
  readonly zeroPoint?: { readonly block: string };
};

/** const 領域の block（§2.1 — offset は const 領域の先頭からの相対）。 */
export type ConstBlockRecord = {
  readonly id: string;
  readonly offset: number;
  readonly length: number;
  readonly sha256: Sha256Hex;
};

/** const block → initializer の束縛（グラフの所有物なのでグラフ記述側 — §3）。 */
export type ConstantBinding = {
  readonly graph: string;
  readonly initializer: string;
  readonly block: string;
  readonly encoding: Encoding;
};

/** グラフ記述（§2.1）。 */
export type GraphDescriptor = {
  readonly format: "karume-container";
  readonly version: 1;
  readonly capabilities: {
    readonly ops: readonly string[];
    readonly features: readonly string[];
  };
  readonly graphs: Readonly<Record<string, IrDeclaration>>;
  readonly const: {
    readonly length: number;
    readonly blocks: readonly ConstBlockRecord[];
    readonly constants: readonly ConstantBinding[];
  };
};

export type BlockRole = "weight" | "scale" | "zero-point" | "asset";
const BLOCK_ROLES: readonly BlockRole[] = ["weight", "scale", "zero-point", "asset"];

/** part 2 以降の block（§2.2 — offset は所属 part の先頭からの相対）。 */
export type DataBlockRecord = ConstBlockRecord & {
  readonly part: number;
  readonly role: BlockRole;
};

/** part の内部配置（§2.2 — 添字 1 以上だけ。part 0 の完全性は外側の期待 hash が張る）。 */
export type PartRecord = {
  readonly index: number;
  readonly length: number;
  readonly sha256: Sha256Hex;
};

/** piece 列の 1 本（§5 — `rows` は先頭次元の半開区間）。 */
export type WeightPiece = {
  readonly block: string;
  readonly rows: readonly [number, number];
};

/** 供給形（§5）。`block`（丸ごと 1 本）と `pieces`（2 本以上の行分割）は排他。 */
export type WeightSupply =
  | { readonly block: string; readonly pieces?: undefined; readonly encoding: Encoding }
  | {
    readonly block?: undefined;
    readonly pieces: readonly WeightPiece[];
    readonly encoding: Encoding;
  };

/** 資産（§2.2 — runtime は `role` を解釈しない）。 */
export type AssetBinding = {
  readonly block: string;
  readonly role: string;
  /** payload のバイト数（論理長）。block 長はこれを 4 の倍数へ切り上げた値 MUST（末尾は 0x00 詰め）。 */
  readonly length: number;
};

/** 出所（§2.3 — 本文は載せない）。 */
export type Provenance = {
  readonly license: string;
  readonly notice?: string;
  readonly upstreamRevision?: string;
  readonly writer?: string;
};

/** モデル記述（§2.2）。 */
export type ModelDescriptor = {
  readonly format: "karume-model";
  readonly version: 1;
  /**
   * 束縛表が使う codec 登録名の集合（§2.1 の `capabilities.codecs` の置き場をこちらへ移した —
   * グラフ記述は `krm` / `krg` でバイト同一 MUST なので、束縛表に依存する欄を持てない）。
   */
  readonly codecs: readonly CodecName[];
  readonly parts: readonly PartRecord[];
  readonly blocks: readonly DataBlockRecord[];
  /** グラフ名 → initializer 名 → 供給。 */
  readonly binding: Readonly<Record<string, Readonly<Record<string, WeightSupply>>>>;
  /** 資産名 → `{ block, role }`。 */
  readonly assets: Readonly<Record<string, AssetBinding>>;
  readonly provenance: Provenance;
};

// ---------------------------------------------------------------------------
// 小さな読み取り器（全て path 付きで落ちる）
// ---------------------------------------------------------------------------

// NOTE: 関数宣言にするのは、`never` 戻りの呼び出しを TS が制御フローの打ち切りとして扱う条件が
// 「宣言型が明示された識別子」であるため（const の arrow では narrowing が効かない）。
function fail(message: string): never {
  throw new ContainerFormatError(message);
}

const requireObject = (value: unknown, path: string): JsonObject => {
  if (!isJsonObject(value)) fail(`${path} がオブジェクトでない`);
  return value;
};

const requireArray = (value: unknown, path: string): readonly unknown[] => {
  if (!Array.isArray(value)) fail(`${path} が配列でない`);
  return value;
};

const requireString = (value: unknown, path: string): string => {
  if (typeof value !== "string" || value.length === 0) fail(`${path} が非空文字列でない`);
  return value;
};

const requireIndex = (value: unknown, path: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(`${path} が非負の安全整数でない: ${String(value)}`);
  }
  return value;
};

const requirePositive = (value: unknown, path: string): number => {
  const n = requireIndex(value, path);
  if (n === 0) fail(`${path} が 0`);
  return n;
};

/** 既知キー以外が 1 つでもあれば拒否し、必須キーの欠けも拒否する（§0 — 未知のキーは fail loudly）。 */
const requireKeys = (
  object: JsonObject,
  required: readonly string[],
  optional: readonly string[],
  path: string,
): void => {
  for (const key of required) {
    if (!Object.hasOwn(object, key)) fail(`${path}.${key} が無い`);
  }
  for (const key of Object.keys(object)) {
    if (!required.includes(key) && !optional.includes(key)) fail(`${path}: 未知のキー '${key}'`);
  }
};

const requireStrings = (value: unknown, path: string): string[] =>
  requireArray(value, path).map((item, i) => requireString(item, `${path}[${i}]`));

/** 集合として扱う欄（重複は集合等価の判定を曖昧にする）。 */
const requireUniqueStrings = (value: unknown, path: string): string[] => {
  const list = requireStrings(value, path);
  const seen = new Set<string>();
  for (const item of list) {
    if (seen.has(item)) fail(`${path}: '${item}' が重複している`);
    seen.add(item);
  }
  return list;
};

export const asSha256Hex = (value: unknown, path: string): Sha256Hex => {
  if (typeof value !== "string" || !SHA256_HEX_PATTERN.test(value)) {
    fail(`${path} が sha256（小文字 16 進 64 文字）でない: ${String(value)}`);
  }
  return value as Sha256Hex;
};

/** バイト列の sha256 を 16 進で返す。`subarray` をそのまま渡せる（複製しない）。 */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<Sha256Hex> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join(
    "",
  ) as Sha256Hex;
};

const requireBlockId = (value: unknown, path: string): string => {
  const id = requireString(value, path);
  if (!BLOCK_ID_PATTERN.test(id)) fail(`${path}: block id '${id}' が語彙外（${BLOCK_ID_PATTERN}）`);
  return id;
};

const sameSet = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((item, i) => item === b[i]);

// ---------------------------------------------------------------------------
// encoding（§6.1）
// ---------------------------------------------------------------------------

const PACKING_KEYS = ["blockElements", "blockBytes", "alignBytes"] as const;

export const parseEncoding = (value: unknown, path: string): Encoding => {
  const object = requireObject(value, path);
  requireKeys(object, ["codec", "packing"], ["rowAxis", "groupSize", "scale", "zeroPoint"], path);
  const codec = requireString(object["codec"], `${path}.codec`);
  if (!isCodecName(codec)) {
    fail(`${path}.codec: 台帳に無い codec '${codec}'（台帳: ${CODEC_NAMES.join(" / ")}）`);
  }
  const entry = codecEntry(codec);
  const packingObject = requireObject(object["packing"], `${path}.packing`);
  requireKeys(packingObject, PACKING_KEYS, [], `${path}.packing`);
  const packing = {
    blockElements: requirePositive(packingObject["blockElements"], `${path}.packing.blockElements`),
    blockBytes: requirePositive(packingObject["blockBytes"], `${path}.packing.blockBytes`),
    alignBytes: requirePositive(packingObject["alignBytes"], `${path}.packing.alignBytes`),
  };
  // 宣言の packing は台帳の写し — 食い違いは「別の版の台帳で書かれた資産」（§6.1）。
  for (const key of PACKING_KEYS) {
    if (packing[key] !== entry.packing[key]) {
      fail(
        `${path}.packing.${key}: codec '${codec}' の台帳は ${entry.packing[key]} だが宣言は ${
          packing[key]
        }（別の版の台帳で書かれた資産）`,
      );
    }
  }
  const quantized = entry.scale === "required";
  for (const key of ["rowAxis", "groupSize", "scale"] as const) {
    if (quantized && !Object.hasOwn(object, key)) {
      fail(`${path}.${key}: codec '${codec}' は量子化なので必須`);
    }
    if (!quantized && Object.hasOwn(object, key)) {
      fail(`${path}.${key}: codec '${codec}' は量子化でないので書けない`);
    }
  }
  if (entry.zeroPoint === "forbidden" && Object.hasOwn(object, "zeroPoint")) {
    fail(`${path}.zeroPoint: codec '${codec}' は zeroPoint を持てない`);
  }
  if (!quantized) return { codec, packing };
  const rowAxis = requireIndex(object["rowAxis"], `${path}.rowAxis`);
  if (rowAxis !== 0 && rowAxis !== 1) fail(`${path}.rowAxis は 0 か 1（宣言は ${rowAxis}）`);
  const groupSize = requirePositive(object["groupSize"], `${path}.groupSize`);
  const scaleObject = requireObject(object["scale"], `${path}.scale`);
  requireKeys(scaleObject, ["block", "dtype"], [], `${path}.scale`);
  const scaleDtype = requireString(scaleObject["dtype"], `${path}.scale.dtype`);
  if (!SCALE_DTYPES.some((dtype) => dtype === scaleDtype)) {
    fail(`${path}.scale.dtype が受理集合（${SCALE_DTYPES.join(" / ")}）に無い: ${scaleDtype}`);
  }
  const encoding: Encoding = {
    codec,
    packing,
    rowAxis,
    groupSize,
    scale: {
      block: requireBlockId(scaleObject["block"], `${path}.scale.block`),
      dtype: scaleDtype as ScaleDtype,
    },
  };
  if (!Object.hasOwn(object, "zeroPoint")) return encoding;
  const zeroObject = requireObject(object["zeroPoint"], `${path}.zeroPoint`);
  requireKeys(zeroObject, ["block"], [], `${path}.zeroPoint`);
  return {
    ...encoding,
    zeroPoint: { block: requireBlockId(zeroObject["block"], `${path}.zeroPoint.block`) },
  };
};

const canonicalEncoding = (encoding: Encoding): Record<string, unknown> => ({
  codec: encoding.codec,
  packing: {
    blockElements: encoding.packing.blockElements,
    blockBytes: encoding.packing.blockBytes,
    alignBytes: encoding.packing.alignBytes,
  },
  ...(encoding.rowAxis === undefined ? {} : { rowAxis: encoding.rowAxis }),
  ...(encoding.groupSize === undefined ? {} : { groupSize: encoding.groupSize }),
  ...(encoding.scale === undefined
    ? {}
    : { scale: { block: encoding.scale.block, dtype: encoding.scale.dtype } }),
  ...(encoding.zeroPoint === undefined ? {} : { zeroPoint: { block: encoding.zeroPoint.block } }),
});

// ---------------------------------------------------------------------------
// block の形（§4.1）
// ---------------------------------------------------------------------------

const assertBlockShape = (block: ConstBlockRecord, path: string): void => {
  if (block.offset % BLOCK_START_ALIGN !== 0) {
    fail(`${path}: offset ${block.offset} が ${BLOCK_START_ALIGN} の倍数でない`);
  }
  if (block.length === 0) fail(`${path}: length が 0`);
  if (block.length % BLOCK_TAIL_ALIGN !== 0) {
    fail(`${path}: length ${block.length} が ${BLOCK_TAIL_ALIGN} の倍数でない`);
  }
  if (block.length > BLOCK_MAX_BYTES) {
    fail(`${path}: length ${block.length} が block 上限 ${BLOCK_MAX_BYTES} を超える`);
  }
};

/** 同じ領域（const 領域 / 1 つの part）の block が互いに重ならず領域長に収まること。 */
const assertNoOverlap = (
  blocks: readonly ConstBlockRecord[],
  regionLength: number,
  path: string,
): void => {
  const sorted = [...blocks].sort((a, b) => a.offset - b.offset);
  let cursor = 0;
  for (const block of sorted) {
    if (block.offset < cursor) fail(`${path}: block '${block.id}' が直前の block と重なる`);
    cursor = block.offset + block.length;
    if (cursor > regionLength) {
      fail(`${path}: block '${block.id}' が領域長 ${regionLength} をはみ出す（末尾 ${cursor}）`);
    }
  }
};

const parseConstBlock = (value: unknown, path: string): ConstBlockRecord => {
  const object = requireObject(value, path);
  requireKeys(object, ["id", "offset", "length", "sha256"], [], path);
  return {
    id: requireBlockId(object["id"], `${path}.id`),
    offset: requireIndex(object["offset"], `${path}.offset`),
    length: requireIndex(object["length"], `${path}.length`),
    sha256: asSha256Hex(object["sha256"], `${path}.sha256`),
  };
};

const parseDataBlock = (value: unknown, path: string): DataBlockRecord => {
  const object = requireObject(value, path);
  requireKeys(object, ["id", "part", "offset", "length", "sha256", "role"], [], path);
  const role = requireString(object["role"], `${path}.role`);
  if (!BLOCK_ROLES.some((known) => known === role)) {
    fail(`${path}.role が ${BLOCK_ROLES.join(" / ")} のいずれでもない: ${role}`);
  }
  return {
    id: requireBlockId(object["id"], `${path}.id`),
    part: requireIndex(object["part"], `${path}.part`),
    offset: requireIndex(object["offset"], `${path}.offset`),
    length: requireIndex(object["length"], `${path}.length`),
    sha256: asSha256Hex(object["sha256"], `${path}.sha256`),
    role: role as BlockRole,
  };
};

// ---------------------------------------------------------------------------
// グラフ記述（§2.1）
// ---------------------------------------------------------------------------

const GRAPH_DESCRIPTOR_KEYS = ["format", "version", "capabilities", "graphs", "const"] as const;

const requireFormat = (object: JsonObject, expected: string, path: string): void => {
  const format = requireString(object["format"], `${path}.format`);
  if (format !== expected) fail(`${path}.format が '${expected}' でない: ${format}`);
  if (object["version"] !== 1) fail(`${path}.version が 1 でない: ${String(object["version"])}`);
};

/** グラフ記述のバイト列を検査つきで読む（宣言だけで決まる検査まで済ませて返す）。 */
export const parseGraphDescriptor = (bytes: Uint8Array<ArrayBuffer>): GraphDescriptor => {
  const path = "graphDescriptor";
  const object = requireObject(decodeJsonDocument(bytes, path, MAX_DESCRIPTOR_BYTES, fail), path);
  requireKeys(object, GRAPH_DESCRIPTOR_KEYS, [], path);
  requireFormat(object, "karume-container", path);

  const capabilitiesObject = requireObject(object["capabilities"], `${path}.capabilities`);
  requireKeys(capabilitiesObject, ["ops", "features"], [], `${path}.capabilities`);
  const ops = requireUniqueStrings(capabilitiesObject["ops"], `${path}.capabilities.ops`);
  const features = requireUniqueStrings(
    capabilitiesObject["features"],
    `${path}.capabilities.features`,
  );
  if (features.length > 0) {
    // 初版は拡張点を 1 つも定義していない（§2.1）— 知らない機能を要求する資産を黙って読まない。
    fail(`${path}.capabilities.features: 未知の機能 [${features.join(", ")}]（初版は空配列のみ）`);
  }

  const graphsObject = requireObject(object["graphs"], `${path}.graphs`);
  const graphNames = Object.keys(graphsObject);
  if (graphNames.length === 0) fail(`${path}.graphs が空`);
  if (graphNames.length > MAX_GRAPHS) {
    fail(`${path}.graphs の件数 ${graphNames.length} が上限 ${MAX_GRAPHS} を超える`);
  }
  const graphs: Record<string, IrDeclaration> = {};
  for (const name of graphNames) {
    if (!GRAPH_NAME_PATTERN.test(name)) {
      fail(`${path}.graphs: グラフ名 '${name}' が語彙外（${GRAPH_NAME_PATTERN}）`);
    }
    try {
      graphs[name] = parseIrDeclarationValue(graphsObject[name]);
    } catch (cause) {
      if (cause instanceof IrError) fail(`${path}.graphs['${name}']: ${cause.message}`);
      throw cause;
    }
  }
  // capabilities.ops は全グラフの requires.ops の和集合と完全一致（§2.1）。
  const union = sortedByCodePoints(
    new Set(graphNames.flatMap((name) => graphs[name].requires.ops)),
  );
  if (!sameSet(sortedByCodePoints(ops), union)) {
    fail(
      `${path}.capabilities.ops [${ops.join(", ")}] が graphs の requires.ops の和集合 [${
        union.join(", ")
      }] と一致しない`,
    );
  }

  const constObject = requireObject(object["const"], `${path}.const`);
  requireKeys(constObject, ["length", "blocks", "constants"], [], `${path}.const`);
  const length = requireIndex(constObject["length"], `${path}.const.length`);
  const blocks = requireArray(constObject["blocks"], `${path}.const.blocks`).map((block, i) =>
    parseConstBlock(block, `${path}.const.blocks[${i}]`)
  );
  const constants = requireArray(constObject["constants"], `${path}.const.constants`).map(
    (entry, i) => {
      const entryPath = `${path}.const.constants[${i}]`;
      const entryObject = requireObject(entry, entryPath);
      requireKeys(entryObject, ["graph", "initializer", "block", "encoding"], [], entryPath);
      return {
        graph: requireString(entryObject["graph"], `${entryPath}.graph`),
        initializer: requireString(entryObject["initializer"], `${entryPath}.initializer`),
        block: requireBlockId(entryObject["block"], `${entryPath}.block`),
        encoding: parseEncoding(entryObject["encoding"], `${entryPath}.encoding`),
      };
    },
  );

  const descriptor: GraphDescriptor = {
    format: "karume-container",
    version: 1,
    capabilities: { ops, features },
    graphs,
    const: { length, blocks, constants },
  };
  validateGraphDescriptor(descriptor);
  return descriptor;
};

/** グラフ記述の宣言検査（const 目次と束縛の整合）。 */
export const validateGraphDescriptor = (descriptor: GraphDescriptor): void => {
  const path = "graphDescriptor.const";
  const region = descriptor.const;
  if (region.blocks.length > MAX_BLOCKS) fail(`${path}.blocks の件数が上限 ${MAX_BLOCKS} を超える`);
  const ids = new Set<string>();
  let previousOffset = -1;
  for (const [i, block] of region.blocks.entries()) {
    const blockPath = `${path}.blocks[${i}]`;
    if (ids.has(block.id)) fail(`${blockPath}: block id '${block.id}' が重複`);
    ids.add(block.id);
    assertBlockShape(block, blockPath);
    // const.blocks は offset 昇順で並ぶ（§2.1 — 読み手が並べ直さずに済む）。
    if (block.offset <= previousOffset) fail(`${blockPath}: offset ${block.offset} が昇順でない`);
    previousOffset = block.offset;
  }
  assertNoOverlap(region.blocks, region.length, path);

  // 規則①: 1 block ≤ 1 参照。scale / zeroPoint の block も同様。
  const referenced = new Set<string>();
  const claim = (id: string, by: string): void => {
    if (!ids.has(id)) fail(`${by}: const 目次に無い block '${id}'`);
    if (referenced.has(id)) {
      fail(`${by}: block '${id}' が二重に束縛されている（1 block ≤ 1 binding）`);
    }
    referenced.add(id);
  };
  const seen = new Set<string>();
  for (const [i, entry] of region.constants.entries()) {
    const entryPath = `${path}.constants[${i}]`;
    const graph = descriptor.graphs[entry.graph];
    if (graph === undefined) fail(`${entryPath}.graph: 未宣言のグラフ '${entry.graph}'`);
    const declared = graph.initializers[entry.initializer];
    if (declared === undefined) {
      fail(
        `${entryPath}.initializer: グラフ '${entry.graph}' に initializer '${entry.initializer}' が無い`,
      );
    }
    if (declared.shared) {
      fail(
        `${entryPath}: '${entry.initializer}' は shared 宣言（貸し手の重みを借りる）なので const では供給できない`,
      );
    }
    const key = `${entry.graph}/${entry.initializer}`;
    if (seen.has(key)) {
      fail(
        `${entryPath}: (graph, initializer) = ('${entry.graph}', '${entry.initializer}') が重複`,
      );
    }
    seen.add(key);
    claim(entry.block, entryPath);
    if (entry.encoding.scale !== undefined) {
      claim(entry.encoding.scale.block, `${entryPath}.encoding.scale`);
    }
    if (entry.encoding.zeroPoint !== undefined) {
      claim(entry.encoding.zeroPoint.block, `${entryPath}.encoding.zeroPoint`);
    }
  }
  for (const id of ids) {
    // 規則⑤: どこからも参照されない block は余剰（黙って太った配布形を受理しない）。
    if (!referenced.has(id)) fail(`${path}: block '${id}' がどこからも参照されていない（余剰）`);
  }
};

/** グラフ記述の正準直列化（docs/ir-v2.md「正準直列化」の規則をグラフ記述全体へ広げたもの）。 */
export const serializeGraphDescriptor = (descriptor: GraphDescriptor): Uint8Array<ArrayBuffer> => {
  validateGraphDescriptor(descriptor);
  const bytes = encodeJsonBytes({
    format: "karume-container",
    version: 1,
    capabilities: {
      ops: sortedByCodePoints(descriptor.capabilities.ops),
      features: sortedByCodePoints(descriptor.capabilities.features),
    },
    graphs: sortedObject(descriptor.graphs, canonicalIrDocument),
    const: {
      length: descriptor.const.length,
      blocks: [...descriptor.const.blocks]
        .sort((a, b) => a.offset - b.offset)
        .map((block) => ({
          id: block.id,
          offset: block.offset,
          length: block.length,
          sha256: block.sha256,
        })),
      constants: sortConstants(descriptor.const.constants).map((entry) => ({
        graph: entry.graph,
        initializer: entry.initializer,
        block: entry.block,
        encoding: canonicalEncoding(entry.encoding),
      })),
    },
  });
  if (bytes.byteLength > MAX_DESCRIPTOR_BYTES) {
    fail(`グラフ記述が上限 ${MAX_DESCRIPTOR_BYTES} バイトを超える: ${bytes.byteLength}`);
  }
  return bytes;
};

const sortConstants = (constants: readonly ConstantBinding[]): ConstantBinding[] =>
  [...constants].sort((a, b) => {
    const byGraph = a.graph < b.graph ? -1 : a.graph > b.graph ? 1 : 0;
    if (byGraph !== 0) return byGraph;
    return a.initializer < b.initializer ? -1 : a.initializer > b.initializer ? 1 : 0;
  });

// ---------------------------------------------------------------------------
// モデル記述（§2.2）
// ---------------------------------------------------------------------------

const MODEL_DESCRIPTOR_KEYS = [
  "format",
  "version",
  "codecs",
  "parts",
  "blocks",
  "binding",
  "assets",
  "provenance",
] as const;
const PROVENANCE_KEYS = ["notice", "upstreamRevision", "writer"] as const;

const parseSupply = (value: unknown, path: string): WeightSupply => {
  const object = requireObject(value, path);
  requireKeys(object, ["encoding"], ["block", "pieces"], path);
  const encoding = parseEncoding(object["encoding"], `${path}.encoding`);
  const hasBlock = Object.hasOwn(object, "block");
  const hasPieces = Object.hasOwn(object, "pieces");
  if (hasBlock === hasPieces) {
    fail(`${path}: 'block'（丸ごと 1 本）と 'pieces'（行分割）のどちらか一方だけを書く`);
  }
  if (hasBlock) return { block: requireBlockId(object["block"], `${path}.block`), encoding };
  const pieces = requireArray(object["pieces"], `${path}.pieces`).map((piece, i) => {
    const piecePath = `${path}.pieces[${i}]`;
    const pieceObject = requireObject(piece, piecePath);
    requireKeys(pieceObject, ["block", "rows"], [], piecePath);
    const rows = requireArray(pieceObject["rows"], `${piecePath}.rows`);
    if (rows.length !== 2) fail(`${piecePath}.rows の長さが 2 でない`);
    const begin = requireIndex(rows[0], `${piecePath}.rows[0]`);
    const end = requireIndex(rows[1], `${piecePath}.rows[1]`);
    if (end <= begin) fail(`${piecePath}.rows [${begin}, ${end}) が空区間`);
    return {
      block: requireBlockId(pieceObject["block"], `${piecePath}.block`),
      rows: [begin, end] as const,
    };
  });
  if (pieces.length < 2) fail(`${path}.pieces は 2 本以上（1 本なら 'block' で書く）`);
  // 規則④: 行範囲を隙間なく被覆（末尾 = shape[0] は宣言 shape を知る合流層が見る）。
  let cursor = 0;
  for (const [i, piece] of pieces.entries()) {
    if (piece.rows[0] !== cursor) {
      fail(`${path}.pieces[${i}]: 行 ${cursor} から続かない（宣言は ${piece.rows[0]}）`);
    }
    cursor = piece.rows[1];
  }
  return { pieces, encoding };
};

/** モデル記述のバイト列を検査つきで読む。 */
export const parseModelDescriptor = (bytes: Uint8Array<ArrayBuffer>): ModelDescriptor => {
  const path = "modelDescriptor";
  const object = requireObject(decodeJsonDocument(bytes, path, MAX_DESCRIPTOR_BYTES, fail), path);
  requireKeys(object, MODEL_DESCRIPTOR_KEYS, [], path);
  requireFormat(object, "karume-model", path);

  const codecs = requireUniqueStrings(object["codecs"], `${path}.codecs`).map((codec) => {
    if (!isCodecName(codec)) {
      fail(`${path}.codecs: 台帳に無い codec '${codec}'（台帳: ${CODEC_NAMES.join(" / ")}）`);
    }
    return codec;
  });

  const parts = requireArray(object["parts"], `${path}.parts`).map((part, i) => {
    const partPath = `${path}.parts[${i}]`;
    const partObject = requireObject(part, partPath);
    requireKeys(partObject, ["index", "length", "sha256"], [], partPath);
    return {
      index: requireIndex(partObject["index"], `${partPath}.index`),
      length: requireIndex(partObject["length"], `${partPath}.length`),
      sha256: asSha256Hex(partObject["sha256"], `${partPath}.sha256`),
    };
  });

  const blocks = requireArray(object["blocks"], `${path}.blocks`).map((block, i) =>
    parseDataBlock(block, `${path}.blocks[${i}]`)
  );

  const bindingObject = requireObject(object["binding"], `${path}.binding`);
  const binding: Record<string, Record<string, WeightSupply>> = {};
  for (const graphName of Object.keys(bindingObject)) {
    const graphPath = `${path}.binding['${graphName}']`;
    const supplies: Record<string, WeightSupply> = {};
    for (const [name, raw] of Object.entries(requireObject(bindingObject[graphName], graphPath))) {
      supplies[requireString(name, `${graphPath} のキー`)] = parseSupply(
        raw,
        `${graphPath}['${name}']`,
      );
    }
    binding[graphName] = supplies;
  }

  const assetsObject = requireObject(object["assets"], `${path}.assets`);
  const assets: Record<string, AssetBinding> = {};
  for (const [name, raw] of Object.entries(assetsObject)) {
    const assetPath = `${path}.assets['${name}']`;
    requireString(name, `${path}.assets のキー`);
    const assetObject = requireObject(raw, assetPath);
    requireKeys(assetObject, ["block", "role", "length"], [], assetPath);
    assets[name] = {
      block: requireBlockId(assetObject["block"], `${assetPath}.block`),
      role: requireString(assetObject["role"], `${assetPath}.role`),
      length: requireIndex(assetObject["length"], `${assetPath}.length`),
    };
  }

  const provenanceObject = requireObject(object["provenance"], `${path}.provenance`);
  requireKeys(provenanceObject, ["license"], PROVENANCE_KEYS, `${path}.provenance`);
  const provenance: Provenance = {
    license: requireString(provenanceObject["license"], `${path}.provenance.license`),
    ...Object.fromEntries(
      PROVENANCE_KEYS.filter((key) => Object.hasOwn(provenanceObject, key)).map((key) => [
        key,
        requireString(provenanceObject[key], `${path}.provenance.${key}`),
      ]),
    ),
  };

  const descriptor: ModelDescriptor = {
    format: "karume-model",
    version: 1,
    codecs,
    parts,
    blocks,
    binding,
    assets,
    provenance,
  };
  validateModelDescriptor(descriptor);
  return descriptor;
};

/** 供給形の実体 block と付随 block（scale / zeroPoint）— 参照検査の共通経路。 */
const supplyBlocks = (supply: WeightSupply): {
  readonly weights: readonly string[];
  readonly scale: string | undefined;
  readonly zeroPoint: string | undefined;
} => ({
  weights: supply.block !== undefined ? [supply.block] : supply.pieces.map((piece) => piece.block),
  scale: supply.encoding.scale?.block,
  zeroPoint: supply.encoding.zeroPoint?.block,
});

/** モデル記述の宣言検査（グラフ記述との突合は {@link validateAgainstGraph}）。 */
export const validateModelDescriptor = (descriptor: ModelDescriptor): void => {
  const path = "modelDescriptor";
  if (descriptor.parts.length === 0) {
    fail(`${path}.parts が空（part 1 = const 領域は長さ 0 でも必ず宣言する）`);
  }
  if (descriptor.parts.length > MAX_PARTS) fail(`${path}.parts の件数が上限 ${MAX_PARTS} を超える`);
  if (descriptor.blocks.length > MAX_BLOCKS) {
    fail(`${path}.blocks の件数が上限 ${MAX_BLOCKS} を超える`);
  }

  // parts は添字 1 から昇順に隙間なく（§2.2 — part 0 は自己参照になるので載せない）。
  const partLength = new Map<number, number>();
  for (const [i, part] of descriptor.parts.entries()) {
    const partPath = `${path}.parts[${i}]`;
    if (part.index !== i + 1) fail(`${partPath}.index が ${i + 1} でない: ${part.index}`);
    if (part.length > PART_MAX_BYTES) {
      fail(`${partPath}.length ${part.length} が part 長の天井 ${PART_MAX_BYTES} を超える`);
    }
    partLength.set(part.index, part.length);
  }

  const byId = new Map<string, DataBlockRecord>();
  const byPart = new Map<number, DataBlockRecord[]>();
  for (const [i, block] of descriptor.blocks.entries()) {
    const blockPath = `${path}.blocks[${i}]`;
    if (byId.has(block.id)) fail(`${blockPath}: block id '${block.id}' が重複`);
    byId.set(block.id, block);
    if (block.part < 2) {
      fail(`${blockPath}.part が 2 未満（part 0 は descriptor・part 1 は const 領域の専有）`);
    }
    if (!partLength.has(block.part)) fail(`${blockPath}.part: 未宣言の part ${block.part}`);
    assertBlockShape(block, blockPath);
    const list = byPart.get(block.part);
    if (list === undefined) byPart.set(block.part, [block]);
    else list.push(block);
  }
  for (const [part, blocks] of byPart) {
    assertNoOverlap(blocks, partLength.get(part) ?? 0, `${path} part ${part}`);
  }

  // 規則①: 1 block ≤ 1 binding。role は参照する側が決める（実体 = weight・scale・zero-point・asset）。
  const referenced = new Set<string>();
  const claim = (id: string, role: BlockRole, by: string): DataBlockRecord => {
    const block = byId.get(id);
    if (block === undefined) fail(`${by}: 目次に無い block '${id}'`);
    if (block.role !== role) {
      fail(`${by}: block '${id}' の role は '${block.role}'（'${role}' が要る）`);
    }
    if (referenced.has(id)) {
      fail(`${by}: block '${id}' が二重に束縛されている（1 block ≤ 1 binding）`);
    }
    referenced.add(id);
    return block;
  };
  const usedCodecs = new Set<CodecName>();
  for (const [graphName, supplies] of Object.entries(descriptor.binding)) {
    for (const [name, supply] of Object.entries(supplies)) {
      const entryPath = `${path}.binding['${graphName}']['${name}']`;
      usedCodecs.add(supply.encoding.codec);
      const refs = supplyBlocks(supply);
      const parts = refs.weights.map((id, i) =>
        claim(id, "weight", `${entryPath}${supply.pieces === undefined ? "" : `.pieces[${i}]`}`)
          .part
      );
      const firstPart = parts[0];
      // 規則③: companion scale / zeroPoint の block は実体（piece 列なら piece 1）と同一 part。
      if (refs.scale !== undefined) {
        const scalePart = claim(refs.scale, "scale", `${entryPath}.encoding.scale`).part;
        if (scalePart !== firstPart) {
          fail(
            `${entryPath}.encoding.scale: scale が part ${scalePart}・実体の先頭が part ${firstPart}（同一 part MUST）`,
          );
        }
      }
      if (refs.zeroPoint !== undefined) {
        const zeroPart =
          claim(refs.zeroPoint, "zero-point", `${entryPath}.encoding.zeroPoint`).part;
        if (zeroPart !== firstPart) {
          fail(
            `${entryPath}.encoding.zeroPoint: zeroPoint が part ${zeroPart}・実体の先頭が part ${firstPart}（同一 part MUST）`,
          );
        }
      }
    }
  }
  for (const [name, asset] of Object.entries(descriptor.assets)) {
    const assetPath = `${path}.assets['${name}']`;
    const block = claim(asset.block, "asset", assetPath);
    // 資産は shape を持たないので論理長を自分で宣言する（§2.2）。block 長 = 論理長の 4 の倍数への
    // 切り上げ MUST — 詰め物の量まで宣言で閉じる（消費側が末尾の 0x00 を推測で剥がない）。
    if (alignUp(asset.length, BLOCK_TAIL_ALIGN) !== block.length) {
      fail(
        `${assetPath}.length: 論理長 ${asset.length} を ${BLOCK_TAIL_ALIGN} の倍数へ切り上げた値が block '${asset.block}' の長さ ${block.length} と違う`,
      );
    }
  }
  for (const id of byId.keys()) {
    if (!referenced.has(id)) fail(`${path}: block '${id}' がどこからも参照されていない（余剰）`);
  }
  // `codecs` は束縛表が使う codec の集合と完全一致（宣言と実際の突合 — 重みを取る前に台帳と照合）。
  const declared = sortedByCodePoints(descriptor.codecs);
  const actual = sortedByCodePoints(usedCodecs);
  if (!sameSet(declared, actual)) {
    fail(
      `${path}.codecs [${declared.join(", ")}] が束縛表の codec 集合 [${
        actual.join(", ")
      }] と一致しない`,
    );
  }
};

/** 2 文書のあいだの突合（§2 / §5 の規則⑤ — 束縛表のキー集合と initializer 集合の完全一致）。 */
export const validateAgainstGraph = (model: ModelDescriptor, graph: GraphDescriptor): void => {
  const constPart = model.parts[0];
  if (constPart.length !== graph.const.length) {
    fail(
      `part 1 の長さ ${constPart.length} がグラフ記述の const.length ${graph.const.length} と違う`,
    );
  }
  if (graph.const.blocks.length + model.blocks.length > MAX_BLOCKS) {
    fail(
      `block 件数（const ${graph.const.blocks.length} + 重み ${model.blocks.length}）が上限 ${MAX_BLOCKS} を超える`,
    );
  }
  const constIds = new Set(graph.const.blocks.map((block) => block.id));
  for (const block of model.blocks) {
    if (constIds.has(block.id)) {
      fail(`block id '${block.id}' が const 目次とモデル目次で衝突（素集合 MUST）`);
    }
  }
  const constSupplied = new Set(
    graph.const.constants.map((entry) => `${entry.graph}/${entry.initializer}`),
  );
  for (const graphName of Object.keys(model.binding)) {
    if (!Object.hasOwn(graph.graphs, graphName)) {
      fail(`modelDescriptor.binding: 未宣言のグラフ '${graphName}'`);
    }
  }
  for (const [graphName, declaration] of Object.entries(graph.graphs)) {
    const supplies = model.binding[graphName] ?? {};
    const expected = sortedByCodePoints(
      Object.entries(declaration.initializers)
        .filter(([name, init]) => !init.shared && !constSupplied.has(`${graphName}/${name}`))
        .map(([name]) => name),
    );
    const bound = sortedByCodePoints(Object.keys(supplies));
    const missing = expected.filter((name) => !supplies[name]);
    const surplus = bound.filter((name) => !expected.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      fail(
        `グラフ '${graphName}' の束縛表が initializer 宣言と一致しない: 不足 [${
          missing.join(", ")
        }] / 余剰 [${surplus.join(", ")}]（shared 宣言と const 供給は束縛表に載せない）`,
      );
    }
  }
};

/** モデル記述の正準直列化。 */
export const serializeModelDescriptor = (descriptor: ModelDescriptor): Uint8Array<ArrayBuffer> => {
  validateModelDescriptor(descriptor);
  const bytes = encodeJsonBytes({
    format: "karume-model",
    version: 1,
    codecs: sortedByCodePoints(descriptor.codecs),
    parts: [...descriptor.parts]
      .sort((a, b) => a.index - b.index)
      .map((part) => ({ index: part.index, length: part.length, sha256: part.sha256 })),
    blocks: [...descriptor.blocks]
      .sort((a, b) => a.part - b.part || a.offset - b.offset)
      .map((block) => ({
        id: block.id,
        part: block.part,
        offset: block.offset,
        length: block.length,
        sha256: block.sha256,
        role: block.role,
      })),
    binding: sortedObject(descriptor.binding, (supplies) =>
      sortedObject(supplies, (supply) => ({
        ...(supply.block === undefined
          ? {
            pieces: supply.pieces.map((piece) => ({
              block: piece.block,
              rows: [piece.rows[0], piece.rows[1]],
            })),
          }
          : { block: supply.block }),
        encoding: canonicalEncoding(supply.encoding),
      }))),
    assets: sortedObject(descriptor.assets, (asset) => ({
      block: asset.block,
      role: asset.role,
      length: asset.length,
    })),

    provenance: {
      license: descriptor.provenance.license,
      ...Object.fromEntries(
        PROVENANCE_KEYS.filter((key) => descriptor.provenance[key] !== undefined).map((key) => [
          key,
          descriptor.provenance[key],
        ]),
      ),
    },
  });
  if (bytes.byteLength > MAX_DESCRIPTOR_BYTES) {
    fail(`モデル記述が上限 ${MAX_DESCRIPTOR_BYTES} バイトを超える: ${bytes.byteLength}`);
  }
  return bytes;
};
