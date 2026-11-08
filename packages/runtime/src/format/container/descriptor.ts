/**
 * Types, parsing, declaration checks and canonical serialization of the two-document descriptor
 * (graph descriptor / model descriptor) — docs/container-v1.md §2 / §4 / §5 / §6 / §7.
 *
 * - **Graph descriptor** = `graphs` (IR v2) + the const region's index and bindings +
 *   `capabilities`. **Byte-identical** in a `krm` and a `krg` (the condition for byte-copy extraction
 *   of a `krg` — §9).
 * - **Model descriptor** = binding table + weight / asset index + the internal layout of parts +
 *   provenance.
 *
 * Applied here are **the checks determined by declarations alone** (effective before fetching a
 * single byte of weights — §7). The rules for payload length, piece coverage and group length,
 * which are determined by the declared shape and the encoding, belong to the merge layer
 * (`bind.ts`).
 *
 * A descriptor cannot prove its own correctness: the bytes of the 2 documents are verified with
 * **the outer expected hash + length** first and parsed after that (`open.ts`).
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
  compareCodePoints,
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
// Types
// ---------------------------------------------------------------------------

declare const sha256HexBrand: unique symbol;

/** sha256 as 64 lowercase hex characters (§0). */
export type Sha256Hex = string & { readonly [sha256HexBrand]: true };

export type EncodingPacking = {
  readonly blockElements: number;
  readonly blockBytes: number;
  readonly alignBytes: number;
};

/**
 * The companion scale of a quantized supply (§6.1): one block, or — when one block cannot hold it —
 * a piece list split on the weight's pieces (§5 rule ③, ADR 0108 addendum 15). `block` and `pieces`
 * are exclusive; `pieces` has 2 or more entries, each the element of the weight's piece list.
 */
export type ScaleRef =
  | { readonly block: string; readonly pieces?: undefined; readonly dtype: ScaleDtype }
  | {
      readonly block?: undefined;
      readonly pieces: readonly WeightPiece[];
      readonly dtype: ScaleDtype;
    };

/**
 * Storage declaration (§6.1). The bit count is derived from `packing`, so it has no field.
 * `rowAxis` / `groupSize` / `scale` are written only for quantized codecs (`scale: "required"` in
 * the ledger) and cannot be written for non-quantized ones.
 */
export type Encoding = {
  readonly codec: CodecName;
  readonly packing: EncodingPacking;
  readonly rowAxis?: 0 | 1;
  readonly groupSize?: number;
  readonly scale?: ScaleRef;
  readonly zeroPoint?: { readonly block: string };
};

/** A block of the const region (§2.1 — offset is relative to the start of the const region). */
export type ConstBlockRecord = {
  readonly id: string;
  readonly offset: number;
  readonly length: number;
  readonly sha256: Sha256Hex;
};

/**
 * Binding of a const block → an initializer (owned by the graph, so on the graph descriptor side —
 * §3).
 */
export type ConstantBinding = {
  readonly graph: string;
  readonly initializer: string;
  readonly block: string;
  readonly encoding: Encoding;
};

/** Graph descriptor (§2.1). */
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

/** A block in part 2 onward (§2.2 — offset is relative to the start of its part). */
export type DataBlockRecord = ConstBlockRecord & {
  readonly part: number;
  readonly role: BlockRole;
};

/**
 * Internal layout of a part (§2.2 — index 1 and above only. The integrity of part 0 is covered by
 * the outer expected hash).
 */
export type PartRecord = {
  readonly index: number;
  readonly length: number;
  readonly sha256: Sha256Hex;
};

/** One entry of a piece list (§5 — `rows` is a half-open range of the first dimension). */
export type WeightPiece = {
  readonly block: string;
  readonly rows: readonly [number, number];
};

/** Supply form (§5). `block` (whole) and `pieces` (split by rows into 2 or more) are exclusive. */
export type WeightSupply =
  | { readonly block: string; readonly pieces?: undefined; readonly encoding: Encoding }
  | {
      readonly block?: undefined;
      readonly pieces: readonly WeightPiece[];
      readonly encoding: Encoding;
    };

/** Asset (§2.2 — the runtime does not interpret `role`). */
export type AssetBinding = {
  readonly block: string;
  readonly role: string;
  /**
   * Payload byte count (logical length). The block length MUST be this rounded up to a multiple of
   * 4 (tail padded with 0x00).
   */
  readonly length: number;
};

/** Provenance (§2.3 — no full texts). */
export type Provenance = {
  readonly license: string;
  readonly notice?: string;
  readonly upstreamRevision?: string;
  readonly writer?: string;
};

/** Model descriptor (§2.2). */
export type ModelDescriptor = {
  readonly format: "karume-model";
  readonly version: 1;
  /**
   * The set of registered codec names the binding table uses (moved here from `capabilities.codecs`
   * of §2.1 — the graph descriptor MUST be byte-identical in `krm` / `krg`, so it cannot have a
   * field that depends on the binding table).
   */
  readonly codecs: readonly CodecName[];
  readonly parts: readonly PartRecord[];
  readonly blocks: readonly DataBlockRecord[];
  /** Graph name → initializer name → supply. */
  readonly binding: Readonly<Record<string, Readonly<Record<string, WeightSupply>>>>;
  /** Asset name → `{ block, role }`. */
  readonly assets: Readonly<Record<string, AssetBinding>>;
  readonly provenance: Provenance;
};

// ---------------------------------------------------------------------------
// Small readers (all fail with a path)
// ---------------------------------------------------------------------------

// NOTE: a function declaration, because TS treats a call returning `never` as ending control flow
// only for "an identifier with an explicitly declared type" (narrowing does not work with a const
// arrow).
function fail(message: string): never {
  throw new ContainerFormatError(message);
}

const requireObject = (value: unknown, path: string): JsonObject => {
  if (!isJsonObject(value)) fail(`${path} is not an object`);
  return value;
};

const requireArray = (value: unknown, path: string): readonly unknown[] => {
  if (!Array.isArray(value)) fail(`${path} is not an array`);
  return value;
};

const requireString = (value: unknown, path: string): string => {
  if (typeof value !== "string" || value.length === 0) fail(`${path} is not a non-empty string`);
  return value;
};

const requireIndex = (value: unknown, path: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(`${path} is not a non-negative safe integer: ${String(value)}`);
  }
  return value;
};

const requirePositive = (value: unknown, path: string): number => {
  const n = requireIndex(value, path);
  if (n === 0) fail(`${path} is 0`);
  return n;
};

/**
 * Rejects any key that is not known, and rejects missing required keys (§0 — unknown keys fail
 * loudly).
 */
const requireKeys = (
  object: JsonObject,
  required: readonly string[],
  optional: readonly string[],
  path: string,
): void => {
  for (const key of required) {
    if (!Object.hasOwn(object, key)) fail(`${path}.${key} is missing`);
  }
  for (const key of Object.keys(object)) {
    if (!required.includes(key) && !optional.includes(key)) fail(`${path}: unknown key '${key}'`);
  }
};

const requireStrings = (value: unknown, path: string): string[] =>
  requireArray(value, path).map((item, i) => requireString(item, `${path}[${i}]`));

/** A field treated as a set (duplicates would make the set-equality check ambiguous). */
const requireUniqueStrings = (value: unknown, path: string): string[] => {
  const list = requireStrings(value, path);
  const seen = new Set<string>();
  for (const item of list) {
    if (seen.has(item)) fail(`${path}: '${item}' is duplicated`);
    seen.add(item);
  }
  return list;
};

export const asSha256Hex = (value: unknown, path: string): Sha256Hex => {
  if (typeof value !== "string" || !SHA256_HEX_PATTERN.test(value)) {
    fail(`${path} is not a sha256 (64 lowercase hex characters): ${String(value)}`);
  }
  return value as Sha256Hex;
};

/** Returns the sha256 of bytes in hex. A `subarray` can be passed as is (no copy). */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<Sha256Hex> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join(
    "",
  ) as Sha256Hex;
};

const requireBlockId = (value: unknown, path: string): string => {
  const id = requireString(value, path);
  if (!BLOCK_ID_PATTERN.test(id))
    fail(`${path}: block id '${id}' is outside the vocabulary (${BLOCK_ID_PATTERN})`);
  return id;
};

const sameSet = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((item, i) => item === b[i]);

// ---------------------------------------------------------------------------
// encoding（§6.1）
// ---------------------------------------------------------------------------

/**
 * A piece list (§5): 2 or more `{ block, rows: [begin, end) }`, each non-empty, continuing from row
 * 0 without gaps. The one reader of both the weight's `pieces` and a split scale's `scale.pieces`.
 */
const parsePieces = (value: unknown, path: string): readonly WeightPiece[] => {
  const pieces = requireArray(value, path).map((piece, i) => {
    const piecePath = `${path}[${i}]`;
    const pieceObject = requireObject(piece, piecePath);
    requireKeys(pieceObject, ["block", "rows"], [], piecePath);
    const rows = requireArray(pieceObject["rows"], `${piecePath}.rows`);
    if (rows.length !== 2) fail(`${piecePath}.rows length is not 2`);
    const begin = requireIndex(rows[0], `${piecePath}.rows[0]`);
    const end = requireIndex(rows[1], `${piecePath}.rows[1]`);
    if (end <= begin) fail(`${piecePath}.rows [${begin}, ${end}) is an empty range`);
    return {
      block: requireBlockId(pieceObject["block"], `${piecePath}.block`),
      rows: [begin, end] as const,
    };
  });
  if (pieces.length < 2) fail(`${path} must have 2 or more entries (write 'block' for one)`);
  // Rule ④: the row range is covered without gaps (end = shape[0] is checked by the merge layer,
  // which knows the declared shape).
  let cursor = 0;
  for (const [i, piece] of pieces.entries()) {
    if (piece.rows[0] !== cursor) {
      fail(`${path}[${i}]: does not continue from row ${cursor} (declared ${piece.rows[0]})`);
    }
    cursor = piece.rows[1];
  }
  return pieces;
};

const canonicalPieces = (pieces: readonly WeightPiece[]): Record<string, unknown>[] =>
  pieces.map((piece) => ({ block: piece.block, rows: [piece.rows[0], piece.rows[1]] }));

const PACKING_KEYS = ["blockElements", "blockBytes", "alignBytes"] as const;

export const parseEncoding = (value: unknown, path: string): Encoding => {
  const object = requireObject(value, path);
  requireKeys(object, ["codec", "packing"], ["rowAxis", "groupSize", "scale", "zeroPoint"], path);
  const codec = requireString(object["codec"], `${path}.codec`);
  if (!isCodecName(codec)) {
    fail(
      `${path}.codec: codec '${codec}' is not in the ledger (ledger: ${CODEC_NAMES.join(" / ")})`,
    );
  }
  const entry = codecEntry(codec);
  const packingObject = requireObject(object["packing"], `${path}.packing`);
  requireKeys(packingObject, PACKING_KEYS, [], `${path}.packing`);
  const packing = {
    blockElements: requirePositive(packingObject["blockElements"], `${path}.packing.blockElements`),
    blockBytes: requirePositive(packingObject["blockBytes"], `${path}.packing.blockBytes`),
    alignBytes: requirePositive(packingObject["alignBytes"], `${path}.packing.alignBytes`),
  };
  // The declared packing is a copy of the ledger — a mismatch means "an asset written with a
  // different version of the ledger" (§6.1).
  for (const key of PACKING_KEYS) {
    if (packing[key] !== entry.packing[key]) {
      fail(
        `${path}.packing.${key}: the ledger of codec '${codec}' has ${entry.packing[key]} but the declaration has ${
          packing[key]
        } (an asset written with a different version of the ledger)`,
      );
    }
  }
  const quantized = entry.scale === "required";
  for (const key of ["rowAxis", "groupSize", "scale"] as const) {
    if (quantized && !Object.hasOwn(object, key)) {
      fail(`${path}.${key}: codec '${codec}' is quantized, so it is required`);
    }
    if (!quantized && Object.hasOwn(object, key)) {
      fail(`${path}.${key}: codec '${codec}' is not quantized, so it cannot be written`);
    }
  }
  if (entry.zeroPoint === "forbidden" && Object.hasOwn(object, "zeroPoint")) {
    fail(`${path}.zeroPoint: codec '${codec}' cannot have a zeroPoint`);
  }
  if (!quantized) return { codec, packing };
  const rowAxis = requireIndex(object["rowAxis"], `${path}.rowAxis`);
  if (rowAxis !== 0 && rowAxis !== 1) fail(`${path}.rowAxis must be 0 or 1 (declared ${rowAxis})`);
  const groupSize = requirePositive(object["groupSize"], `${path}.groupSize`);
  const scaleObject = requireObject(object["scale"], `${path}.scale`);
  requireKeys(scaleObject, ["dtype"], ["block", "pieces"], `${path}.scale`);
  const scaleDtype = requireString(scaleObject["dtype"], `${path}.scale.dtype`);
  if (!SCALE_DTYPES.some((dtype) => dtype === scaleDtype)) {
    fail(
      `${path}.scale.dtype is not in the accepted set (${SCALE_DTYPES.join(" / ")}): ${scaleDtype}`,
    );
  }
  // The codec implies the step dtype (the ledger's `scaleDtype`) — a mismatch would have the decode
  // read f16 steps as f32 or the reverse, a silent wrong value with a plausible byte count.
  if (scaleDtype !== entry.scaleDtype) {
    fail(
      `${path}.scale.dtype is not in the accepted set of codec '${codec}' (${String(entry.scaleDtype)} steps): ${scaleDtype}`,
    );
  }
  const hasBlock = Object.hasOwn(scaleObject, "block");
  if (hasBlock === Object.hasOwn(scaleObject, "pieces")) {
    fail(`${path}.scale: write exactly one of 'block' (one block) and 'pieces' (split scale)`);
  }
  const encoding: Encoding = {
    codec,
    packing,
    rowAxis,
    groupSize,
    scale: hasBlock
      ? {
          block: requireBlockId(scaleObject["block"], `${path}.scale.block`),
          dtype: scaleDtype as ScaleDtype,
        }
      : {
          pieces: parsePieces(scaleObject["pieces"], `${path}.scale.pieces`),
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
    : {
        scale:
          encoding.scale.pieces === undefined
            ? { block: encoding.scale.block, dtype: encoding.scale.dtype }
            : { pieces: canonicalPieces(encoding.scale.pieces), dtype: encoding.scale.dtype },
      }),
  ...(encoding.zeroPoint === undefined ? {} : { zeroPoint: { block: encoding.zeroPoint.block } }),
});

// ---------------------------------------------------------------------------
// Block shape (§4.1)
// ---------------------------------------------------------------------------

const assertBlockShape = (block: ConstBlockRecord, path: string): void => {
  if (block.offset % BLOCK_START_ALIGN !== 0) {
    fail(`${path}: offset ${block.offset} is not a multiple of ${BLOCK_START_ALIGN}`);
  }
  if (block.length === 0) fail(`${path}: length is 0`);
  if (block.length % BLOCK_TAIL_ALIGN !== 0) {
    fail(`${path}: length ${block.length} is not a multiple of ${BLOCK_TAIL_ALIGN}`);
  }
  if (block.length > BLOCK_MAX_BYTES) {
    fail(`${path}: length ${block.length} exceeds the block limit ${BLOCK_MAX_BYTES}`);
  }
};

/**
 * The blocks of one region (the const region / one part) do not overlap and fit in the region length.
 */
const assertNoOverlap = (
  blocks: readonly ConstBlockRecord[],
  regionLength: number,
  path: string,
): void => {
  const sorted = [...blocks].sort((a, b) => a.offset - b.offset);
  let cursor = 0;
  for (const block of sorted) {
    if (block.offset < cursor) fail(`${path}: block '${block.id}' overlaps the previous block`);
    cursor = block.offset + block.length;
    if (cursor > regionLength) {
      fail(
        `${path}: block '${block.id}' runs past the region length ${regionLength} (end ${cursor})`,
      );
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
    fail(`${path}.role is none of ${BLOCK_ROLES.join(" / ")}: ${role}`);
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
// Graph descriptor (§2.1)
// ---------------------------------------------------------------------------

const GRAPH_DESCRIPTOR_KEYS = ["format", "version", "capabilities", "graphs", "const"] as const;

const requireFormat = (object: JsonObject, expected: string, path: string): void => {
  const format = requireString(object["format"], `${path}.format`);
  if (format !== expected) fail(`${path}.format is not '${expected}': ${format}`);
  if (object["version"] !== 1) fail(`${path}.version is not 1: ${String(object["version"])}`);
};

/**
 * Reads graph descriptor bytes with checks (returns after the checks determined by declarations
 * alone).
 */
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
    // The first edition defines no extension point (§2.1) — assets that require unknown features
    // are not read silently.
    fail(
      `${path}.capabilities.features: unknown features [${features.join(", ")}] (the first edition allows only an empty array)`,
    );
  }

  const graphsObject = requireObject(object["graphs"], `${path}.graphs`);
  const graphNames = Object.keys(graphsObject);
  if (graphNames.length === 0) fail(`${path}.graphs is empty`);
  if (graphNames.length > MAX_GRAPHS) {
    fail(`${path}.graphs count ${graphNames.length} exceeds the limit ${MAX_GRAPHS}`);
  }
  const graphs: Record<string, IrDeclaration> = Object.create(null);
  for (const name of graphNames) {
    if (!GRAPH_NAME_PATTERN.test(name)) {
      fail(
        `${path}.graphs: graph name '${name}' is outside the vocabulary (${GRAPH_NAME_PATTERN})`,
      );
    }
    try {
      graphs[name] = parseIrDeclarationValue(graphsObject[name]);
    } catch (cause) {
      if (cause instanceof IrError) fail(`${path}.graphs['${name}']: ${cause.message}`);
      throw cause;
    }
  }
  // capabilities.ops equals exactly the union of requires.ops of all graphs (§2.1).
  const union = sortedByCodePoints(
    new Set(graphNames.flatMap((name) => graphs[name].requires.ops)),
  );
  if (!sameSet(sortedByCodePoints(ops), union)) {
    fail(
      `${path}.capabilities.ops [${ops.join(", ")}] does not match the union of requires.ops over graphs [${union.join(
        ", ",
      )}]`,
    );
  }

  const constObject = requireObject(object["const"], `${path}.const`);
  requireKeys(constObject, ["length", "blocks", "constants"], [], `${path}.const`);
  const length = requireIndex(constObject["length"], `${path}.const.length`);
  const blocks = requireArray(constObject["blocks"], `${path}.const.blocks`).map((block, i) =>
    parseConstBlock(block, `${path}.const.blocks[${i}]`),
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

/** Declaration checks of the graph descriptor (consistency of the const index and bindings). */
export const validateGraphDescriptor = (descriptor: GraphDescriptor): void => {
  const path = "graphDescriptor.const";
  const region = descriptor.const;
  // The ceiling of §10 applies to part 1 as well. In a krg this field is the only one holding part
  // 1's size, so unless the descriptor check stops it, it passes the reader's length check and
  // reaches a huge bulk read (a krm also checks it on the model descriptor's parts side, but the
  // duplication is harmless).
  if (region.length > PART_MAX_BYTES) {
    fail(`${path}.length ${region.length} exceeds the part length ceiling ${PART_MAX_BYTES}`);
  }
  if (region.blocks.length > MAX_BLOCKS)
    fail(`${path}.blocks count exceeds the limit ${MAX_BLOCKS}`);
  const ids = new Set<string>();
  let previousOffset = -1;
  for (const [i, block] of region.blocks.entries()) {
    const blockPath = `${path}.blocks[${i}]`;
    if (ids.has(block.id)) fail(`${blockPath}: block id '${block.id}' is duplicated`);
    ids.add(block.id);
    assertBlockShape(block, blockPath);
    // const.blocks are in ascending offset order (§2.1 — the reader need not reorder them).
    if (block.offset <= previousOffset)
      fail(`${blockPath}: offset ${block.offset} is not in ascending order`);
    previousOffset = block.offset;
  }
  assertNoOverlap(region.blocks, region.length, path);

  // Rule ①: 1 block ≤ 1 reference. The same for scale / zeroPoint blocks.
  const referenced = new Set<string>();
  const claim = (id: string, by: string): void => {
    if (!ids.has(id)) fail(`${by}: block '${id}' is not in the const index`);
    if (referenced.has(id)) {
      fail(`${by}: block '${id}' is bound twice (1 block ≤ 1 binding)`);
    }
    referenced.add(id);
  };
  const seen = new Set<string>();
  for (const [i, entry] of region.constants.entries()) {
    const entryPath = `${path}.constants[${i}]`;
    const graph = Object.hasOwn(descriptor.graphs, entry.graph)
      ? descriptor.graphs[entry.graph]
      : undefined;
    if (graph === undefined) fail(`${entryPath}.graph: undeclared graph '${entry.graph}'`);
    const declared = graph.initializers[entry.initializer];
    if (declared === undefined) {
      fail(
        `${entryPath}.initializer: graph '${entry.graph}' has no initializer '${entry.initializer}'`,
      );
    }
    if (declared.shared) {
      fail(
        `${entryPath}: '${entry.initializer}' is a shared declaration (it borrows the lender's weights), so it cannot be supplied as a const`,
      );
    }
    const key = `${entry.graph}/${entry.initializer}`;
    if (seen.has(key)) {
      fail(
        `${entryPath}: (graph, initializer) = ('${entry.graph}', '${entry.initializer}') is duplicated`,
      );
    }
    seen.add(key);
    claim(entry.block, entryPath);
    if (entry.encoding.scale !== undefined) {
      // A const is one block (it has no piece list — §3), so there are no piece boundaries to
      // split its scale on (rule ③).
      if (entry.encoding.scale.pieces !== undefined) {
        fail(
          `${entryPath}.encoding.scale: a const cannot have a split scale (a const has no pieces — rule ③)`,
        );
      }
      claim(entry.encoding.scale.block, `${entryPath}.encoding.scale`);
    }
    if (entry.encoding.zeroPoint !== undefined) {
      claim(entry.encoding.zeroPoint.block, `${entryPath}.encoding.zeroPoint`);
    }
  }
  for (const id of ids) {
    // Rule ⑤: a block referenced from nowhere is surplus (a bloated distribution is not accepted
    // silently).
    if (!referenced.has(id))
      fail(`${path}: block '${id}' is not referenced from anywhere (surplus)`);
  }
};

/**
 * Canonical serialization of the graph descriptor (the rules of docs/ir-v2.md "Canonical
 * serialization" extended to the whole graph descriptor).
 */
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
    fail(
      `Graph descriptor exceeds the limit of ${MAX_DESCRIPTOR_BYTES} bytes: ${bytes.byteLength}`,
    );
  }
  return bytes;
};

const sortConstants = (constants: readonly ConstantBinding[]): ConstantBinding[] =>
  [...constants].sort((a, b) => {
    const byGraph = compareCodePoints(a.graph, b.graph);
    if (byGraph !== 0) return byGraph;
    return compareCodePoints(a.initializer, b.initializer);
  });

// ---------------------------------------------------------------------------
// Model descriptor (§2.2)
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
    fail(`${path}: write exactly one of 'block' (one whole block) and 'pieces' (row split)`);
  }
  if (hasBlock) return { block: requireBlockId(object["block"], `${path}.block`), encoding };
  return { pieces: parsePieces(object["pieces"], `${path}.pieces`), encoding };
};

/** Reads model descriptor bytes with checks. */
export const parseModelDescriptor = (bytes: Uint8Array<ArrayBuffer>): ModelDescriptor => {
  const path = "modelDescriptor";
  const object = requireObject(decodeJsonDocument(bytes, path, MAX_DESCRIPTOR_BYTES, fail), path);
  requireKeys(object, MODEL_DESCRIPTOR_KEYS, [], path);
  requireFormat(object, "karume-model", path);

  const codecs = requireUniqueStrings(object["codecs"], `${path}.codecs`).map((codec) => {
    if (!isCodecName(codec)) {
      fail(
        `${path}.codecs: codec '${codec}' is not in the ledger (ledger: ${CODEC_NAMES.join(" / ")})`,
      );
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
    parseDataBlock(block, `${path}.blocks[${i}]`),
  );

  const bindingObject = requireObject(object["binding"], `${path}.binding`);
  const binding: Record<string, Record<string, WeightSupply>> = Object.create(null);
  for (const graphName of Object.keys(bindingObject)) {
    const graphPath = `${path}.binding['${graphName}']`;
    const supplies: Record<string, WeightSupply> = Object.create(null);
    for (const [name, raw] of Object.entries(requireObject(bindingObject[graphName], graphPath))) {
      supplies[requireString(name, `${graphPath} key`)] = parseSupply(
        raw,
        `${graphPath}['${name}']`,
      );
    }
    binding[graphName] = supplies;
  }

  const assetsObject = requireObject(object["assets"], `${path}.assets`);
  const assets: Record<string, AssetBinding> = Object.create(null);
  for (const [name, raw] of Object.entries(assetsObject)) {
    const assetPath = `${path}.assets['${name}']`;
    requireString(name, `${path}.assets key`);
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

/**
 * The data blocks and companion blocks (scale / zeroPoint) of a supply — the shared path of the
 * reference checks.
 */
const supplyBlocks = (
  supply: WeightSupply,
): {
  readonly weights: readonly string[];
  /** The scale's blocks (one, or the pieces of a split scale); empty when not quantized. */
  readonly scales: readonly string[];
  readonly zeroPoint: string | undefined;
} => {
  const scale = supply.encoding.scale;
  return {
    weights:
      supply.block !== undefined ? [supply.block] : supply.pieces.map((piece) => piece.block),
    scales:
      scale === undefined
        ? []
        : scale.pieces === undefined
          ? [scale.block]
          : scale.pieces.map((piece) => piece.block),
    zeroPoint: supply.encoding.zeroPoint?.block,
  };
};

/**
 * Declaration checks of the model descriptor (the check against the graph descriptor is
 * {@link validateAgainstGraph}).
 */
export const validateModelDescriptor = (descriptor: ModelDescriptor): void => {
  const path = "modelDescriptor";
  if (descriptor.parts.length === 0) {
    fail(
      `${path}.parts is empty (part 1 = the const region is always declared, even with length 0)`,
    );
  }
  if (descriptor.parts.length > MAX_PARTS)
    fail(`${path}.parts count exceeds the limit ${MAX_PARTS}`);
  if (descriptor.blocks.length > MAX_BLOCKS) {
    fail(`${path}.blocks count exceeds the limit ${MAX_BLOCKS}`);
  }

  // parts run from index 1 in ascending order without gaps (§2.2 — part 0 is not listed, as it would
  // be self-referential).
  const partLength = new Map<number, number>();
  for (const [i, part] of descriptor.parts.entries()) {
    const partPath = `${path}.parts[${i}]`;
    if (part.index !== i + 1) fail(`${partPath}.index is not ${i + 1}: ${part.index}`);
    if (part.length > PART_MAX_BYTES) {
      fail(`${partPath}.length ${part.length} exceeds the part length ceiling ${PART_MAX_BYTES}`);
    }
    partLength.set(part.index, part.length);
  }

  const byId = new Map<string, DataBlockRecord>();
  const byPart = new Map<number, DataBlockRecord[]>();
  for (const [i, block] of descriptor.blocks.entries()) {
    const blockPath = `${path}.blocks[${i}]`;
    if (byId.has(block.id)) fail(`${blockPath}: block id '${block.id}' is duplicated`);
    byId.set(block.id, block);
    if (block.part < 2) {
      fail(
        `${blockPath}.part is less than 2 (part 0 is reserved for the descriptors, part 1 for the const region)`,
      );
    }
    if (!partLength.has(block.part)) fail(`${blockPath}.part: undeclared part ${block.part}`);
    assertBlockShape(block, blockPath);
    const list = byPart.get(block.part);
    if (list === undefined) byPart.set(block.part, [block]);
    else list.push(block);
  }
  for (const [part, blocks] of byPart) {
    assertNoOverlap(blocks, partLength.get(part) ?? 0, `${path} part ${part}`);
  }

  // Rule ①: 1 block ≤ 1 binding. The referencing side decides the role (data = weight, scale,
  // zero-point, asset).
  const referenced = new Set<string>();
  const claim = (id: string, role: BlockRole, by: string): DataBlockRecord => {
    const block = byId.get(id);
    if (block === undefined) fail(`${by}: block '${id}' is not in the index`);
    if (block.role !== role) {
      fail(`${by}: block '${id}' has role '${block.role}' ('${role}' is required)`);
    }
    if (referenced.has(id)) {
      fail(`${by}: block '${id}' is bound twice (1 block ≤ 1 binding)`);
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
      const parts = refs.weights.map(
        (id, i) =>
          claim(id, "weight", `${entryPath}${supply.pieces === undefined ? "" : `.pieces[${i}]`}`)
            .part,
      );
      const firstPart = parts[0];
      // Rule ③: the companion scale / zeroPoint block is in the same part as the data (piece 1 for a
      // piece list) — every block of a split scale too (construction reads the whole scale with
      // piece 1).
      for (const [i, scale] of refs.scales.entries()) {
        const scalePath = `${entryPath}.encoding.scale${
          supply.encoding.scale?.pieces === undefined ? "" : `.pieces[${i}]`
        }`;
        const scalePart = claim(scale, "scale", scalePath).part;
        if (scalePart !== firstPart) {
          fail(
            `${scalePath}: scale is in part ${scalePart}, the first data block in part ${firstPart} (same part MUST)`,
          );
        }
      }
      if (refs.zeroPoint !== undefined) {
        const zeroPart = claim(
          refs.zeroPoint,
          "zero-point",
          `${entryPath}.encoding.zeroPoint`,
        ).part;
        if (zeroPart !== firstPart) {
          fail(
            `${entryPath}.encoding.zeroPoint: zeroPoint is in part ${zeroPart}, the first data block in part ${firstPart} (same part MUST)`,
          );
        }
      }
    }
  }
  for (const [name, asset] of Object.entries(descriptor.assets)) {
    const assetPath = `${path}.assets['${name}']`;
    const block = claim(asset.block, "asset", assetPath);
    // An asset has no shape, so it declares its logical length itself (§2.2). Block length MUST be
    // the logical length rounded up to a multiple of 4 — even the amount of padding is fixed by the
    // declaration (consumers do not strip trailing 0x00 by guessing).
    if (alignUp(asset.length, BLOCK_TAIL_ALIGN) !== block.length) {
      fail(
        `${assetPath}.length: logical length ${asset.length} rounded up to a multiple of ${BLOCK_TAIL_ALIGN} differs from block '${asset.block}' length ${block.length}`,
      );
    }
  }
  for (const id of byId.keys()) {
    if (!referenced.has(id))
      fail(`${path}: block '${id}' is not referenced from anywhere (surplus)`);
  }
  // `codecs` equals exactly the set of codecs the binding table uses (declaration checked against
  // actual use — matched against the ledger before fetching weights).
  const declared = sortedByCodePoints(descriptor.codecs);
  const actual = sortedByCodePoints(usedCodecs);
  if (!sameSet(declared, actual)) {
    fail(
      `${path}.codecs [${declared.join(", ")}] does not match the codec set of the binding table [${actual.join(
        ", ",
      )}]`,
    );
  }
};

/**
 * Check between the 2 documents (§2 / rule ⑤ of §5 — the binding table's key set exactly equals the
 * initializer set).
 */
export const validateAgainstGraph = (model: ModelDescriptor, graph: GraphDescriptor): void => {
  const constPart = model.parts[0];
  if (constPart.length !== graph.const.length) {
    fail(
      `part 1 length ${constPart.length} differs from the graph descriptor const.length ${graph.const.length}`,
    );
  }
  if (graph.const.blocks.length + model.blocks.length > MAX_BLOCKS) {
    fail(
      `block count (const ${graph.const.blocks.length} + weights ${model.blocks.length}) exceeds the limit ${MAX_BLOCKS}`,
    );
  }
  const constIds = new Set(graph.const.blocks.map((block) => block.id));
  for (const block of model.blocks) {
    if (constIds.has(block.id)) {
      fail(
        `block id '${block.id}' collides between the const index and the model index (disjoint sets MUST)`,
      );
    }
  }
  const constSupplied = new Set(
    graph.const.constants.map((entry) => `${entry.graph}/${entry.initializer}`),
  );
  for (const graphName of Object.keys(model.binding)) {
    if (!Object.hasOwn(graph.graphs, graphName)) {
      fail(`modelDescriptor.binding: undeclared graph '${graphName}'`);
    }
  }
  for (const [graphName, declaration] of Object.entries(graph.graphs)) {
    const supplies = Object.hasOwn(model.binding, graphName) ? model.binding[graphName] : {};
    const expected = sortedByCodePoints(
      Object.entries(declaration.initializers)
        .filter(([name, init]) => !init.shared && !constSupplied.has(`${graphName}/${name}`))
        .map(([name]) => name),
    );
    const bound = sortedByCodePoints(Object.keys(supplies));
    const missing = expected.filter((name) => !Object.hasOwn(supplies, name));
    const surplus = bound.filter((name) => !expected.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      fail(
        `Binding table of graph '${graphName}' does not match the initializer declarations: missing [${missing.join(
          ", ",
        )}] / surplus [${surplus.join(", ")}] (shared declarations and const supplies are not listed in the binding table)`,
      );
    }
  }
};

/** Canonical serialization of the model descriptor. */
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
          ? { pieces: canonicalPieces(supply.pieces) }
          : { block: supply.block }),
        encoding: canonicalEncoding(supply.encoding),
      })),
    ),
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
    fail(
      `Model descriptor exceeds the limit of ${MAX_DESCRIPTOR_BYTES} bytes: ${bytes.byteLength}`,
    );
  }
  return bytes;
};
