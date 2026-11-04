/**
 * Test-only container writer (single form / split form / `krg`) — docs/container-v1.md §3 / §4 /
 * §8 / §9.
 *
 * The only production writer is Python (the exporter's `karume.container`); this is **a helper for
 * the reader's tests to build synthetic assets**. It follows the same rules as the production
 * writer:
 *
 * - A block never crosses a part; 1 block ≤ the limit (32 MiB by default — tests can lower it).
 * - A block starts 64 B aligned and its stored length is a multiple of 4 (**the writer bakes in the
 *   tail padding**). Intermediate pieces get no padding (it would overwrite the next piece's
 *   start), so the row step is rounded to `4 / gcd(rowBytes, 4)` when cutting.
 * - A companion scale is in the same part as piece 1 (or as the only block when the tensor is
 *   whole).
 * - The descriptor bytes are **identical in the single and the split form** (no absolute part
 *   offsets are written).
 * - An asset that needs range reads (`dedicatedPart`) is placed alone in a part of its own.
 */

import { codecEntry, type CodecName, payloadBytes } from "../../src/format/container/codecs.ts";
import {
  type AssetBinding,
  type ConstantBinding,
  type ConstBlockRecord,
  type DataBlockRecord,
  type Encoding,
  type GraphDescriptor,
  type ModelDescriptor,
  type PartRecord,
  type Provenance,
  type ScaleRef,
  serializeGraphDescriptor,
  serializeModelDescriptor,
  sha256Hex,
  validateAgainstGraph,
  type WeightSupply,
} from "../../src/format/container/descriptor.ts";
import {
  alignUp,
  assembleGraphContainer,
  ContainerFormatError,
  derivePartOffsets,
  writeHeader,
} from "../../src/format/container/header.ts";
import {
  BLOCK_MAX_BYTES,
  BLOCK_START_ALIGN,
  BLOCK_TAIL_ALIGN,
  CONTAINER_VERSION,
  DEFAULT_PART_BYTES,
  HEADER_BYTES,
  PAD_BYTE,
} from "../../src/format/container/limits.ts";
import type { IrDeclaration } from "../../src/format/ir.ts";

export type EncodingInput = {
  readonly codec: CodecName;
  /** Quantized codecs only. Defaults to 0. */
  readonly rowAxis?: 0 | 1;
  /** Quantized codecs only. The row length for per-channel. */
  readonly groupSize?: number;
  readonly scale?: { readonly bytes: Uint8Array<ArrayBuffer>; readonly dtype: "f32" };
};

export type TensorInput = {
  readonly graph: string;
  readonly initializer: string;
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly encoding: EncodingInput;
};

export type AssetInput = {
  readonly name: string;
  readonly role: string;
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly dedicatedPart?: boolean;
};

export type ModelInput = {
  readonly graphs: Readonly<Record<string, IrDeclaration>>;
  /**
   * Constants owned by the graph (they go to the const region = part 1; never split into
   * pieces).
   */
  readonly consts: readonly TensorInput[];
  readonly weights: readonly TensorInput[];
  readonly assets: readonly AssetInput[];
  readonly provenance: Provenance;
};

export type WriteOptions = {
  /** Part length. 256 MiB by default (tests may make it smaller). */
  readonly partBytes?: number;
  /** Block limit. 32 MiB by default (tests can lower it, not raise it). */
  readonly blockBytes?: number;
};

export type WrittenContainer = {
  readonly graph: GraphDescriptor;
  readonly model: ModelDescriptor;
  readonly graphDescriptorBytes: Uint8Array<ArrayBuffer>;
  readonly modelDescriptorBytes: Uint8Array<ArrayBuffer>;
  /** Split form. Index 0 is `[header][graph descriptor][model descriptor]`. */
  readonly parts: readonly Uint8Array<ArrayBuffer>[];
  /** Single form. */
  readonly single: Uint8Array<ArrayBuffer>;
  /** The `krg` taken from parts 0..1. */
  readonly graphContainer: Uint8Array<ArrayBuffer>;
};

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

const resolveEncoding = (
  input: EncodingInput,
  scaleBlock: string | undefined,
  scalePieces?: readonly { readonly block: string; readonly rows: readonly [number, number] }[],
): Encoding => {
  const entry = codecEntry(input.codec);
  if (entry.scale === "forbidden") {
    if (input.scale !== undefined || input.groupSize !== undefined || input.rowAxis !== undefined) {
      throw new ContainerFormatError(
        `codec '${input.codec}' cannot have scale / groupSize / rowAxis`,
      );
    }
    return { codec: input.codec, packing: entry.packing };
  }
  if (input.scale === undefined || input.groupSize === undefined) {
    throw new ContainerFormatError(`codec '${input.codec}' needs scale and groupSize`);
  }
  const dtype = input.scale.dtype;
  let scale: ScaleRef;
  if (scalePieces !== undefined) scale = { pieces: scalePieces, dtype };
  else if (scaleBlock !== undefined) scale = { block: scaleBlock, dtype };
  else throw new ContainerFormatError(`codec '${input.codec}' needs a scale block`);
  return {
    codec: input.codec,
    packing: entry.packing,
    rowAxis: input.rowAxis ?? 0,
    groupSize: input.groupSize,
    scale,
  };
};

/**
 * The stored bytes with the tail zero padding baked in (`tailPad = false`: only the length
 * check).
 */
const storedPayload = (
  bytes: Uint8Array<ArrayBuffer>,
  tailPad: boolean,
  path: string,
): Uint8Array<ArrayBuffer> => {
  if (bytes.byteLength === 0)
    throw new ContainerFormatError(`${path}: no block of length 0 is made`);
  if (!tailPad) {
    if (bytes.byteLength % BLOCK_TAIL_ALIGN !== 0) {
      throw new ContainerFormatError(
        `${path}: intermediate piece length ${bytes.byteLength} is not a multiple of ${BLOCK_TAIL_ALIGN}`,
      );
    }
    return bytes;
  }
  const length = alignUp(bytes.byteLength, BLOCK_TAIL_ALIGN);
  if (length === bytes.byteLength) return bytes;
  const padded = new Uint8Array(new ArrayBuffer(length));
  padded.fill(PAD_BYTE);
  padded.set(bytes, 0);
  return padded;
};

type PlacedBlock = {
  readonly id: string;
  readonly offset: number;
  readonly length: number;
  readonly payload: Uint8Array<ArrayBuffer>;
};

/** Packer for one part. Offsets are relative to the part's start. */
class PartBuilder {
  #cursor = 0;
  readonly #blocks: PlacedBlock[] = [];

  probe(lengths: readonly number[]): number {
    let cursor = this.#cursor;
    for (const length of lengths) cursor = alignUp(cursor, BLOCK_START_ALIGN) + length;
    return cursor;
  }

  push(id: string, payload: Uint8Array<ArrayBuffer>): void {
    const offset = alignUp(this.#cursor, BLOCK_START_ALIGN);
    this.#blocks.push({ id, offset, length: payload.byteLength, payload });
    this.#cursor = offset + payload.byteLength;
  }

  get isEmpty(): boolean {
    return this.#blocks.length === 0;
  }

  get blocks(): readonly PlacedBlock[] {
    return this.#blocks;
  }

  materialize(): Uint8Array<ArrayBuffer> {
    const out = new Uint8Array(new ArrayBuffer(this.#cursor));
    out.fill(PAD_BYTE);
    for (const block of this.#blocks) out.set(block.payload, block.offset);
    return out;
  }
}

const shapeOf = (input: ModelInput, tensor: TensorInput, path: string): readonly number[] => {
  const graph = input.graphs[tensor.graph];
  if (graph === undefined) {
    throw new ContainerFormatError(`${path}: no graph '${tensor.graph}'`);
  }
  const value = graph.values[tensor.initializer];
  if (value === undefined) {
    throw new ContainerFormatError(`${path}: initializer '${tensor.initializer}' is not declared`);
  }
  return value.shape.map(Number);
};

const assertPayloadLength = (input: ModelInput, tensor: TensorInput, path: string): number => {
  const shape = shapeOf(input, tensor, path);
  const numel = shape.reduce((count, dim) => count * dim, 1);
  const expected = payloadBytes(tensor.encoding.codec, numel, path);
  if (tensor.bytes.byteLength !== expected) {
    throw new ContainerFormatError(
      `${path}: payload ${tensor.bytes.byteLength} バイトが宣言から決まる ${expected} と違う`,
    );
  }
  return shape.length === 0 ? 1 : shape[0];
};

type ConstRegion = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly blocks: readonly ConstBlockRecord[];
  readonly constants: readonly ConstantBinding[];
};

const buildConstRegion = async (input: ModelInput, blockBytes: number): Promise<ConstRegion> => {
  const builder = new PartBuilder();
  const constants: ConstantBinding[] = [];
  let serial = 0;
  const nextId = (prefix: string): string => `${prefix}${String(++serial).padStart(4, "0")}`;
  for (const tensor of input.consts) {
    const path = `const ${tensor.graph}/${tensor.initializer}`;
    assertPayloadLength(input, tensor, path);
    const payload = storedPayload(tensor.bytes, true, path);
    if (payload.byteLength > blockBytes) {
      // Consts have no piece-splitting mechanism (§3 — the binding table is not in the krg, so a
      // piece list cannot be expressed).
      throw new ContainerFormatError(
        `${path}: a const must fit in one block but is ${payload.byteLength} bytes (limit ${blockBytes})`,
      );
    }
    const id = nextId("c");
    builder.push(id, payload);
    let scaleId: string | undefined;
    if (tensor.encoding.scale !== undefined) {
      scaleId = nextId("cs");
      builder.push(scaleId, storedPayload(tensor.encoding.scale.bytes, true, `${path} scale`));
    }
    constants.push({
      graph: tensor.graph,
      initializer: tensor.initializer,
      block: id,
      encoding: resolveEncoding(tensor.encoding, scaleId),
    });
  }
  const blocks: ConstBlockRecord[] = [];
  for (const block of builder.blocks) {
    blocks.push({
      id: block.id,
      offset: block.offset,
      length: block.length,
      sha256: await sha256Hex(block.payload),
    });
  }
  return { bytes: builder.materialize(), blocks, constants };
};

type PieceCut = {
  readonly rows: readonly [number, number];
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly last: boolean;
};

/**
 * Cuts a tensor at row boundaries into pieces within the block limit. Returns one piece when no
 * cut is needed.
 */
const cutPieces = (
  bytes: Uint8Array<ArrayBuffer>,
  rows: number,
  blockBytes: number,
  path: string,
): readonly PieceCut[] => {
  const total = bytes.byteLength;
  if (alignUp(total, BLOCK_TAIL_ALIGN) <= blockBytes) {
    return [{ rows: [0, rows], bytes, last: true }];
  }
  if (rows <= 1)
    throw new ContainerFormatError(`${path}: cannot cut — one row exceeds the block limit`);
  if (total % rows !== 0) {
    throw new ContainerFormatError(
      `${path}: ${total} bytes do not divide evenly into ${rows} rows`,
    );
  }
  const rowBytes = total / rows;
  const step = BLOCK_TAIL_ALIGN / gcd(rowBytes, BLOCK_TAIL_ALIGN);
  const capacity = Math.floor(blockBytes / rowBytes);
  const rowsPerPiece = capacity - (capacity % step);
  if (rowsPerPiece < step || rowsPerPiece === 0) {
    throw new ContainerFormatError(
      `${path}: with ${rowBytes} bytes per row, no 4-byte-aligned group of rows fits the block limit ${blockBytes}`,
    );
  }
  const cuts: PieceCut[] = [];
  for (let begin = 0; begin < rows; begin += rowsPerPiece) {
    const end = Math.min(begin + rowsPerPiece, rows);
    cuts.push({
      rows: [begin, end],
      bytes: bytes.subarray(begin * rowBytes, end * rowBytes),
      last: end === rows,
    });
  }
  return cuts;
};

/**
 * The scale's blocks (mirrors `_cut_scale` of the Python writer): one when it fits the block limit,
 * otherwise one per data piece covering the same rows (rule ③ — ADR 0108 addendum 15).
 */
const cutScale = (
  bytes: Uint8Array<ArrayBuffer>,
  rows: number,
  cuts: readonly PieceCut[],
  blockBytes: number,
  path: string,
): readonly PieceCut[] => {
  if (alignUp(bytes.byteLength, BLOCK_TAIL_ALIGN) <= blockBytes) {
    return [{ rows: [0, rows], bytes, last: true }];
  }
  if (cuts.length === 1) {
    throw new ContainerFormatError(
      `${path}: the scale exceeds the block limit and the data is one block (no piece boundaries)`,
    );
  }
  const rowBytes = bytes.byteLength / rows;
  return cuts.map((cut, i) => {
    const piece = bytes.subarray(cut.rows[0] * rowBytes, cut.rows[1] * rowBytes);
    if (alignUp(piece.byteLength, BLOCK_TAIL_ALIGN) > blockBytes) {
      throw new ContainerFormatError(`${path} scale ${i}: over the block limit ${blockBytes}`);
    }
    return { rows: cut.rows, bytes: piece, last: cut.last };
  });
};

type DataPlan = {
  readonly parts: readonly PartBuilder[];
  readonly roles: ReadonlyMap<string, DataBlockRecord["role"]>;
  readonly binding: Record<string, Record<string, WeightSupply>>;
  readonly assets: Record<string, AssetBinding>;
};

const planDataParts = (input: ModelInput, partBytes: number, blockBytes: number): DataPlan => {
  const parts: PartBuilder[] = [new PartBuilder()];
  const roles = new Map<string, DataBlockRecord["role"]>();
  const binding: Record<string, Record<string, WeightSupply>> = {};
  const assets: Record<string, AssetBinding> = {};
  let serial = 0;
  const nextId = (prefix: string, role: DataBlockRecord["role"]): string => {
    const id = `${prefix}${String(++serial).padStart(4, "0")}`;
    roles.set(id, role);
    return id;
  };
  const current = (): PartBuilder => parts[parts.length - 1];
  const openPart = (): PartBuilder => {
    const next = new PartBuilder();
    parts.push(next);
    return next;
  };
  /**
   * Places a group that "must be in the same part" together. Opens a new part when it does not
   * fit.
   */
  const placeGroup = (
    items: readonly { readonly id: string; readonly payload: Uint8Array<ArrayBuffer> }[],
    path: string,
  ): void => {
    const lengths = items.map((item) => item.payload.byteLength);
    if (new PartBuilder().probe(lengths) > partBytes) {
      throw new ContainerFormatError(
        `${path}: a group that must be in one part exceeds the part length ${partBytes}`,
      );
    }
    let target = current();
    if (target.probe(lengths) > partBytes) target = openPart();
    for (const item of items) target.push(item.id, item.payload);
  };

  for (const tensor of input.weights) {
    const path = `weight ${tensor.graph}/${tensor.initializer}`;
    const rows = assertPayloadLength(input, tensor, path);
    const cuts = cutPieces(tensor.bytes, rows, blockBytes, path);
    const scaleCuts =
      tensor.encoding.scale === undefined
        ? undefined
        : cutScale(tensor.encoding.scale.bytes, rows, cuts, blockBytes, path);
    const pieceIds = cuts.map(() => nextId("w", "weight"));
    const head = [
      {
        id: pieceIds[0],
        payload: storedPayload(cuts[0].bytes, cuts[0].last, `${path} piece 0`),
      },
    ];
    const scaleIds: string[] = [];
    for (const [i, cut] of (scaleCuts ?? []).entries()) {
      scaleIds.push(nextId("s", "scale"));
      head.push({
        id: scaleIds[i],
        payload: storedPayload(cut.bytes, cut.last, `${path} scale ${i}`),
      });
    }
    // Rule ③: piece 1 and the scale (every piece of a split scale) share a part. The remaining
    // pieces can be placed independently.
    placeGroup(head, path);
    for (const [i, cut] of cuts.entries()) {
      if (i === 0) continue;
      placeGroup(
        [
          {
            id: pieceIds[i],
            payload: storedPayload(cut.bytes, cut.last, `${path} piece ${i}`),
          },
        ],
        `${path} piece ${i}`,
      );
    }
    const encoding = resolveEncoding(
      tensor.encoding,
      scaleIds.length === 1 ? scaleIds[0] : undefined,
      scaleCuts !== undefined && scaleCuts.length > 1
        ? scaleCuts.map((cut, i) => ({ block: scaleIds[i], rows: cut.rows }))
        : undefined,
    );
    const supply: WeightSupply =
      cuts.length === 1
        ? { block: pieceIds[0], encoding }
        : { pieces: cuts.map((cut, i) => ({ block: pieceIds[i], rows: cut.rows })), encoding };
    binding[tensor.graph] = { ...binding[tensor.graph], [tensor.initializer]: supply };
  }

  for (const asset of input.assets) {
    const path = `asset ${asset.name}`;
    const payload = storedPayload(asset.bytes, true, path);
    if (payload.byteLength > blockBytes) {
      throw new ContainerFormatError(
        `${path}: ${payload.byteLength} bytes exceed the block limit ${blockBytes}`,
      );
    }
    const id = nextId("a", "asset");
    if (asset.dedicatedPart === true) {
      // Isolated in a part of its own for sources where a range read scans in proportion to the
      // offset (§4.2).
      if (!current().isEmpty) openPart();
      current().push(id, payload);
      openPart();
    } else {
      placeGroup([{ id, payload }], path);
    }
    assets[asset.name] = { block: id, role: asset.role, length: asset.bytes.byteLength };
  }

  return { parts: parts.filter((part) => !part.isEmpty), roles, binding, assets };
};

const buildGraphDescriptor = (input: ModelInput, region: ConstRegion): GraphDescriptor => ({
  format: "karume-container",
  version: 1,
  capabilities: {
    ops: [...new Set(Object.values(input.graphs).flatMap((graph) => graph.requires.ops))].sort(),
    features: [],
  },
  graphs: input.graphs,
  const: { length: region.bytes.byteLength, blocks: region.blocks, constants: region.constants },
});

/**
 * Writes only a `krg` (graph container). The graph side is built by **the same function** as
 * parts 0..1 of a `krm`.
 */
export const writeGraphContainer = async (
  input: ModelInput,
  options: WriteOptions = {},
): Promise<{
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly descriptorBytes: Uint8Array<ArrayBuffer>;
}> => {
  const region = await buildConstRegion(input, options.blockBytes ?? BLOCK_MAX_BYTES);
  const descriptorBytes = serializeGraphDescriptor(buildGraphDescriptor(input, region));
  return { bytes: assembleGraphContainer(descriptorBytes, region.bytes), descriptorBytes };
};

/** Writes a `krm` (model container) in both the single and the split form. */
export const writeModelContainer = async (
  input: ModelInput,
  options: WriteOptions = {},
): Promise<WrittenContainer> => {
  const partBytes = options.partBytes ?? DEFAULT_PART_BYTES;
  const blockBytes = options.blockBytes ?? BLOCK_MAX_BYTES;
  if (blockBytes > BLOCK_MAX_BYTES) {
    throw new ContainerFormatError(
      `block limit ${blockBytes} exceeds the spec's ${BLOCK_MAX_BYTES}`,
    );
  }

  const region = await buildConstRegion(input, blockBytes);
  const graph = buildGraphDescriptor(input, region);
  const graphDescriptorBytes = serializeGraphDescriptor(graph);

  const plan = planDataParts(input, partBytes, blockBytes);
  const dataParts = plan.parts.map((part) => part.materialize());
  const blocks: DataBlockRecord[] = [];
  for (const [i, builder] of plan.parts.entries()) {
    for (const block of builder.blocks) {
      blocks.push({
        id: block.id,
        part: i + 2,
        offset: block.offset,
        length: block.length,
        sha256: await sha256Hex(block.payload),
        role: plan.roles.get(block.id) ?? "weight",
      });
    }
  }
  const partRecords: PartRecord[] = [
    {
      index: 1,
      length: region.bytes.byteLength,
      sha256: await sha256Hex(region.bytes),
    },
  ];
  for (const [i, bytes] of dataParts.entries()) {
    partRecords.push({ index: i + 2, length: bytes.byteLength, sha256: await sha256Hex(bytes) });
  }
  const model: ModelDescriptor = {
    format: "karume-model",
    version: 1,
    codecs: [...new Set(input.weights.map((tensor) => tensor.encoding.codec))].sort(),
    parts: partRecords,
    blocks,
    binding: plan.binding,
    assets: plan.assets,
    provenance: input.provenance,
  };
  validateAgainstGraph(model, graph);
  const modelDescriptorBytes = serializeModelDescriptor(model);

  const header = writeHeader({
    kind: "model",
    version: CONTAINER_VERSION,
    graphDescriptorLength: graphDescriptorBytes.byteLength,
    modelDescriptorLength: modelDescriptorBytes.byteLength,
  });
  const part0 = new Uint8Array(
    new ArrayBuffer(
      HEADER_BYTES + graphDescriptorBytes.byteLength + modelDescriptorBytes.byteLength,
    ),
  );
  part0.set(header, 0);
  part0.set(graphDescriptorBytes, HEADER_BYTES);
  part0.set(modelDescriptorBytes, HEADER_BYTES + graphDescriptorBytes.byteLength);

  const parts = [part0, region.bytes, ...dataParts];
  const layout = derivePartOffsets(
    part0.byteLength,
    parts.slice(1).map((part) => part.byteLength),
  );
  const single = new Uint8Array(new ArrayBuffer(layout.totalLength));
  single.fill(PAD_BYTE);
  for (const [i, part] of parts.entries()) single.set(part, layout.offsets[i]);

  return {
    graph,
    model,
    graphDescriptorBytes,
    modelDescriptorBytes,
    parts,
    single,
    graphContainer: assembleGraphContainer(graphDescriptorBytes, region.bytes),
  };
};
