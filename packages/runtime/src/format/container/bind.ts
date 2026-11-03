/**
 * Merge layer — merges the graph (IR v2 declarations) and the binding table **before the weights are
 * fetched** and decides the supply plan per initializer. docs/container-v1.md §5 / §6 / §13.3,
 * docs/ir-v2.md "Values and types" (the storage and merge-layer rules).
 *
 * The rules placed here are all those determined by "declared shape × encoding × block index"
 * (**they live in one place** — no second implementation):
 *
 * - Pairs of semantic dtype and codec (f32 takes the encodings of f32 / i32 takes raw int32 — cross
 *   pairings fail loudly)
 * - Payload length = the value determined by the declared shape and the packing. The only
 *   difference from the block length is padding (at least 0, less than 4)
 * - A piece list covers the row range of the first dimension without gaps, ending at shape[0].
 *   Middle pieces have no padding
 * - The parts of pieces are non-decreasing in index order (construction streams in ascending part
 *   order and allocates at piece 1)
 * - An initializer with `rowAxis != 0` cannot be split into pieces (the scale's row range does not
 *   correspond to the pieces' row ranges)
 * - `rowAxis` only within the codec ledger's `rowAxes` (axis 1 is for per-channel i8 only — decode
 *   is fixed to axis 0)
 * - Group granularity: for a per-channel codec `groupSize` = row length; for the group codec a power
 *   of two ≥ 16 that divides the row length
 * - Scale block length = `shape[rowAxis] · (row length / groupSize) · 4` (+ padding)
 * - A split scale goes with a piece list only: one scale piece per data piece, each covering the
 *   same rows, middle pieces without padding (rule ③ — ADR 0108 addendum 15)
 * - The declared shape for the i2 path (`int2-off` / `ternary`) is positive rank 2 with a row length
 *   that is a multiple of 16
 */

import {
  type CodecEntry,
  codecEntry,
  type CodecName,
  groupScaleShape,
  MIN_GROUP_SIZE,
  payloadBytes,
  perChannelGroupSize,
  quantizedRowLength,
  scaleBytes,
} from "./codecs.ts";
import type {
  ConstantBinding,
  Encoding,
  GraphDescriptor,
  ModelDescriptor,
  WeightPiece,
  WeightSupply,
} from "./descriptor.ts";
import { ContainerFormatError } from "./header.ts";
import { BLOCK_TAIL_ALIGN } from "./limits.ts";
import { isI2Shape } from "../i2.ts";
import type { IrDeclaration, IrDtype, IrGraph, IrInitializer } from "../ir.ts";

/**
 * **The fields the merge layer needs** from one row of the block index (location and length). No
 * sha256 is required — the integrity of fetched bytes is the fetch layer's job (`readBlock` in
 * `open.ts`), and the merge looks only at "declared shape × encoding × index". A container's index
 * (`ConstBlockRecord` / `DataBlockRecord`) satisfies this structurally, and the in-memory container
 * (`memory.ts`), which supplies its own bytes, fills only these 3 fields.
 */
export type SupplyRecord = {
  readonly id: string;
  readonly offset: number;
  readonly length: number;
};

/** block id → location (part and index row). Undeclared ids fail loudly. */
export type Locator = (
  id: string,
  by: string,
) => { readonly part: number; readonly record: SupplyRecord };

/** The block for one piece of data (one piece for a piece list). */
export type SupplyBlock = {
  readonly id: string;
  readonly part: number;
  readonly offset: number;
  readonly length: number;
  /**
   * Row range of the first dimension this block carries (`[0, shape[0]]` when whole; `[0, 1]` for
   * rank 0).
   */
  readonly rows: readonly [number, number];
  /** Payload byte length of this block (the block length minus padding). */
  readonly payloadBytes: number;
};

/**
 * The companion scale of a quantized supply, handed to construction as **one contiguous scale**
 * whether or not it is split.
 *
 * Unsplit, it is the one declared block and `pieces` is absent. Split (container-v1 §5 rule ③ —
 * ADR 0108 addendum 15), `pieces` holds the declared blocks in row order, piece k covering the rows
 * of the data's piece k, and the fields above describe the **joined read**: `id` is
 * {@link joinedScaleId} of the pieces (not a declared block — the key under which the opened
 * container's `readBlock` returns the pieces concatenated, each verified), `length` is the byte
 * count that read returns, `offset` is the first piece's, and `part` is the part every piece is in.
 */
export type ScaleSupply = SupplyBlock & { readonly pieces?: readonly SupplyBlock[] };

/**
 * The id under which a split scale is read whole: its pieces' block ids joined with `+`.
 *
 * `+` is outside the block-id vocabulary (§0), so a joined id never names a declared block, and the
 * spelling is the same on every reader that derives it from the same descriptor (the Python mirror
 * is `joined_scale_id` in `tools/exporter/src/karume/container.py`).
 */
export const joinedScaleId = (pieceIds: readonly string[]): string => pieceIds.join("+");

/** The supply plan of one initializer (which blocks the data comes from). */
export type InitializerSupply = {
  readonly encoding: Encoding;
  /** The data's block list (one when whole). */
  readonly blocks: readonly SupplyBlock[];
  /** The companion scale (quantized codecs only) — one contiguous scale even when split. */
  readonly scale?: ScaleSupply;
  readonly zeroPoint?: SupplyBlock;
  /**
   * Supply origin: the const region (part 1, owned by the graph) or the weight side (model
   * descriptor).
   */
  readonly origin: "const" | "model";
};

/** Merge result of one graph. */
export type BoundGraph = {
  readonly declaration: IrDeclaration;
  /** Supply plans of every non-shared initializer (declaration order). */
  readonly supplies: ReadonlyMap<string, InitializerSupply>;
};

/**
 * **The supply-source surface** `PreparedModel` needs (both the `OpenedContainer` of an opened `krm`
 * and the return of the in-memory container (`memory.ts`), which passes its own bytes, satisfy it
 * structurally). It lives in the merge layer because the contents of this surface (`graphs`) are
 * the merge result itself — placing it in one supply source's implementation file would make
 * callers that use only the other source depend on that file for types.
 *
 * MUST: callers **do not modify** the bytes `readBlock` returns. The in-memory container returns the
 * buffer the caller passed as is (no copy), so modifying it corrupts the source's data.
 */
export type BoundContainer = {
  readonly graphs: Readonly<Record<string, BoundGraph>>;
  readBlock(id: string): Promise<Uint8Array<ArrayBuffer>>;
};

// NOTE: a function declaration, because TS treats a call returning `never` as ending control flow
// only for "an identifier with an explicitly declared type" (narrowing does not work with a const
// arrow).
function fail(message: string): never {
  throw new ContainerFormatError(message);
}

/**
 * Semantic dtype → the codec decode paths allowed (the merge-layer rules in docs/ir-v2.md "Values
 * and types").
 */
const allowedLayouts = (dtype: IrDtype): readonly CodecEntry["layout"][] =>
  dtype === "f32" ? ["f32", "f16", "bf16", "i8", "i4", "i2"] : dtype === "i32" ? ["i32"] : [];

const isPowerOfTwo = (value: number): boolean => 2 ** Math.round(Math.log2(value)) === value;

const numelOf = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/**
 * Relation of block length and payload length (§4.1 — padding is at least 0 and less than 4, baked
 * by the writer).
 */
const assertPadded = (record: SupplyRecord, payload: number, by: string): void => {
  if (record.length < payload || record.length - payload >= BLOCK_TAIL_ALIGN) {
    fail(
      `${by}: block '${record.id}' length ${record.length} is not payload ${payload} bytes + padding (less than ${BLOCK_TAIL_ALIGN})`,
    );
  }
};

/**
 * Payload byte length from the declared shape (rethrows the bare error of {@link payloadBytes} as
 * this layer's type).
 */
const declaredPayloadBytes = (codec: CodecName, numel: number, where: string): number => {
  try {
    return payloadBytes(codec, numel, where);
  } catch (cause) {
    return fail(cause instanceof Error ? cause.message : String(cause));
  }
};

/**
 * Companion scale byte length from the declaration (rethrows the bare error of {@link scaleBytes}
 * as this layer's type).
 */
const declaredScaleBytes = (count: number, where: string): number => {
  try {
    return scaleBytes(count, where);
  } catch (cause) {
    return fail(cause instanceof Error ? cause.message : String(cause));
  }
};

/**
 * Bytes per row for piece splitting (payload ÷ first dimension).
 *
 * A piece's block length is determined from this value, so **a source that synthesizes its own
 * block index** (`memory.ts`) goes through the same single function — with two formulas, the
 * synthesized length and the length the merge layer expects quietly split (and when they do, the
 * check `found.record.length === pieceBytes` is trivially true, since both carry the same error).
 */
export const pieceRowBytes = (codec: CodecName, shape: readonly number[], where: string): number =>
  rowBytesOfPayload(declaredPayloadBytes(codec, numelOf(shape), where), shape, where);

/**
 * The body of {@link pieceRowBytes} (for callers that already have the payload, to avoid
 * recomputing it).
 */
const rowBytesOfPayload = (payload: number, shape: readonly number[], where: string): number => {
  const rows = shape.length === 0 ? 1 : shape[0];
  if (shape.length === 0 || payload % rows !== 0) {
    fail(
      `${where}: payload ${payload} bytes is not divisible by the leading dimension of ${rows} rows, so it cannot be split into pieces`,
    );
  }
  return payload / rows;
};

/**
 * The plan of a split scale (§5 rule ③ — ADR 0108 addendum 15): one scale block per data piece,
 * piece k covering exactly the rows of data piece k, its length `rows · (rowLength / groupSize) · 4`
 * (middle pieces exactly, the last one with padding under 4). The descriptor parse has already
 * checked that the list has 2 or more entries continuing from row 0 without gaps.
 */
const planSplitScale = (
  where: string,
  dataIsPieced: boolean,
  data: readonly SupplyBlock[],
  pieces: readonly WeightPiece[],
  scale: { readonly rowCount: number; readonly payload: number; readonly locate: Locator },
): ScaleSupply => {
  const by = `${where} scale`;
  // A split scale follows the data's pieces, so data in one block has no boundaries to split it on
  // (the writer fails loudly instead — `_cut_scale` in `karume/container.py`).
  if (!dataIsPieced) {
    fail(`${by}: a split scale needs a piece list (the scale is split on the data's pieces)`);
  }
  if (pieces.length !== data.length) {
    fail(
      `${by}: ${pieces.length} scale pieces for ${data.length} data pieces (one scale piece per data piece — rule ③)`,
    );
  }
  // The data's piece list exists only for rowAxis 0 (rule ④), so the scale's rows are the data's
  // leading dimension and one scale row is `payload / rowCount` bytes.
  const rowBytes = scale.payload / scale.rowCount;
  const planned = pieces.map((piece, i): SupplyBlock => {
    const pieceBy = `${by} piece[${i}]`;
    const rows = data[i].rows;
    if (piece.rows[0] !== rows[0] || piece.rows[1] !== rows[1]) {
      fail(
        `${pieceBy}: rows [${piece.rows[0]}, ${piece.rows[1]}) differ from the rows [${rows[0]}, ${rows[1]}) of data piece ${i} (rule ③)`,
      );
    }
    const found = scale.locate(piece.block, pieceBy);
    const pieceBytes = (rows[1] - rows[0]) * rowBytes;
    if (i < pieces.length - 1) {
      // A middle piece carries no padding (§4.1 — the same rule as the data's middle pieces).
      if (found.record.length !== pieceBytes) {
        fail(
          `${pieceBy}: block '${found.record.id}' length ${found.record.length} differs from the ${pieceBytes} bytes of its rows (a middle piece has no padding)`,
        );
      }
    } else {
      assertPadded(found.record, pieceBytes, pieceBy);
    }
    return {
      id: found.record.id,
      part: found.part,
      offset: found.record.offset,
      length: found.record.length,
      rows: [rows[0], rows[1]],
      payloadBytes: pieceBytes,
    };
  });
  return {
    id: joinedScaleId(planned.map((piece) => piece.id)),
    part: planned[0].part,
    offset: planned[0].offset,
    length: planned.reduce((total, piece) => total + piece.length, 0),
    rows: [0, scale.rowCount],
    payloadBytes: scale.payload,
    pieces: planned,
  };
};

const planSupply = (
  where: string,
  dtype: IrDtype,
  shape: readonly number[],
  supply: { readonly encoding: Encoding } & (
    | { readonly block: string; readonly pieces?: undefined }
    | {
        readonly block?: undefined;
        readonly pieces: WeightSupply extends { pieces?: infer P } ? NonNullable<P> : never;
      }
  ),
  locate: Locator,
  origin: InitializerSupply["origin"],
): InitializerSupply => {
  const { encoding } = supply;
  const entry = codecEntry(encoding.codec);
  if (!allowedLayouts(dtype).includes(entry.layout)) {
    fail(`${where}: semantic dtype '${dtype}' cannot be paired with codec '${encoding.codec}'`);
  }
  const numel = numelOf(shape);
  if (entry.layout === "i2" && !isI2Shape(shape)) {
    fail(
      `${where}: codec '${encoding.codec}' needs a positive rank-2 declared shape with a row length that is a multiple of 16 ([${shape.join(
        ",",
      )}])`,
    );
  }
  const payload = declaredPayloadBytes(encoding.codec, numel, where);
  const rows = shape.length === 0 ? 1 : shape[0];

  // The data's block list.
  const blocks: SupplyBlock[] = [];
  if (supply.pieces === undefined) {
    const found = locate(supply.block, where);
    assertPadded(found.record, payload, where);
    blocks.push({
      id: found.record.id,
      part: found.part,
      offset: found.record.offset,
      length: found.record.length,
      rows: [0, rows],
      payloadBytes: payload,
    });
  } else {
    if (encoding.rowAxis !== undefined && encoding.rowAxis !== 0) {
      fail(
        `${where}: an initializer with rowAxis ${encoding.rowAxis} cannot be split into pieces (rule ④)`,
      );
    }
    const rowBytes = rowBytesOfPayload(payload, shape, where);
    const last = supply.pieces[supply.pieces.length - 1];
    if (last.rows[1] !== rows) {
      fail(
        `${where}: the piece list ends at row ${
          last.rows[1]
        }, which falls short of / exceeds the leading dimension of the declared shape, ${rows} rows (rule ④)`,
      );
    }
    // Rule ④: the row ranges continue from 0 without gaps. Holes, overlaps and empty ranges all
    // become silent wrong values where "which bytes fill that row" depends on the transfer order.
    // Through a container the descriptor parse rejects the same forms first, but **this layer owns
    // the binding rules**, so it checks here too regardless of the source (the in-memory container,
    // which supplies its own bytes, does not go through a descriptor).
    let cursor = 0;
    for (const [i, piece] of supply.pieces.entries()) {
      const by = `${where} piece[${i}]`;
      // Rule ④: row numbers are non-negative safe integers. A fractional row boundary slips past
      // every coverage check (empty range, continuity, end), turns `rowOffset · (total length /
      // rows)` into a byte position that is not a multiple of 4, and delays the failure until
      // GPU writeBuffer validation. Through a container, `requireIndex` in the descriptor rejects
      // the same form first.
      for (const [edge, value] of piece.rows.entries()) {
        if (!Number.isSafeInteger(value) || value < 0) {
          fail(
            `${by}: row range ${
              edge === 0 ? "start" : "end"
            } ${value} is not a non-negative safe integer (rule ④)`,
          );
        }
      }
      if (piece.rows[1] <= piece.rows[0]) {
        fail(`${by}: row range [${piece.rows[0]}, ${piece.rows[1]}) is an empty range (rule ④)`);
      }
      if (piece.rows[0] !== cursor) {
        fail(
          `${by}: row range [${piece.rows[0]}, ${piece.rows[1]}) does not continue from row ${cursor} (rule ④)`,
        );
      }
      cursor = piece.rows[1];
      const found = locate(piece.block, by);
      // Rule ④: the parts of pieces are non-decreasing in index order. Construction
      // (`containerBatches` → `buildSessionState`) streams blocks in ascending part order, allocates
      // the buffer at piece 1 and carries the scale over (by rule ③ the scale is in the same part as
      // piece 1), so a container in reverse order would fail with a message about broken internal
      // bookkeeping. The check of the accepted set sits in the merge layer and fails here regardless
      // of the source.
      const previous = blocks.at(-1);
      if (previous !== undefined && found.part < previous.part) {
        fail(
          `${by}: part ${found.part} comes before part ${previous.part} of the previous piece (the parts of pieces are non-decreasing in index order MUST — rule ④)`,
        );
      }
      const pieceBytes = (piece.rows[1] - piece.rows[0]) * rowBytes;
      if (i < supply.pieces.length - 1) {
        // A middle piece cannot be padded (it would overwrite the start of the next piece) — its
        // length MUST be a multiple of 4 to begin with. In a container the writer and the index
        // parse reject this first, but **this layer owns the binding rules**, so it checks here too
        // regardless of the source (otherwise a piece split of a codec whose row length is not a
        // multiple of 4 would not fail until GPU writeBuffer validation, with a transfer-layer
        // message).
        if (pieceBytes % BLOCK_TAIL_ALIGN !== 0) {
          fail(
            `${by}: byte count ${pieceBytes} of a middle piece's row range is not a multiple of ${BLOCK_TAIL_ALIGN} (no padding allowed — rule ④)`,
          );
        }
        if (found.record.length !== pieceBytes) {
          fail(
            `${by}: block length ${found.record.length} of a middle piece differs from the byte count ${pieceBytes} of its row range (no padding allowed)`,
          );
        }
      } else {
        assertPadded(found.record, pieceBytes, by);
      }
      blocks.push({
        id: found.record.id,
        part: found.part,
        offset: found.record.offset,
        length: found.record.length,
        rows: [piece.rows[0], piece.rows[1]],
        payloadBytes: pieceBytes,
      });
    }
  }

  if (entry.scale === "forbidden") return { encoding, blocks, origin };

  // Quantized: parseEncoding has guaranteed that rowAxis / groupSize / scale exist.
  const rowAxis = encoding.rowAxis ?? 0;
  const groupSize = encoding.groupSize ?? 0;
  if (!entry.rowAxes.some((axis) => axis === rowAxis)) {
    fail(
      `${where}: rowAxis of codec '${encoding.codec}' must be ${entry.rowAxes.join(
        " / ",
      )} only (declared ${rowAxis} — container-v1 §6.3)`,
    );
  }
  if (shape.length === 0 || (rowAxis === 1 && shape.length < 2)) {
    fail(
      `${where}: rowAxis ${rowAxis} needs a higher rank than the declared shape [${shape.join(",")}]`,
    );
  }
  const rowCount = shape[rowAxis];
  // Quantized weights with 0 rows have no use in real models, so they are not in the accepted set
  // (an empty weight is not created as an initializer at all). Accepting them would make a
  // downstream that has its own row-length formula (`numel / shape[0]` in the CPU decode `decodeI4`)
  // fail with 0 / 0, a "row length NaN", in a different layer with a different message. It fails
  // here regardless of the source.
  if (rowCount === 0) {
    fail(
      `${where}: quantized weights with 0 rows (shape[${rowAxis}] = 0) are not accepted (declared shape [${shape.join(
        ",",
      )}] — there is not a single quantized row)`,
    );
  }
  const rowLength = quantizedRowLength(shape, rowAxis);
  if (entry.grouping === "channel" && groupSize !== perChannelGroupSize(rowLength)) {
    fail(
      `${where}: codec '${encoding.codec}' is per-channel, so groupSize MUST equal the row length ${perChannelGroupSize(
        rowLength,
      )} (declared ${groupSize})`,
    );
  }
  if (entry.grouping === "group") {
    if (!isPowerOfTwo(groupSize) || groupSize < MIN_GROUP_SIZE) {
      fail(
        `${where}: groupSize ${groupSize} is not a power of two that is at least ${MIN_GROUP_SIZE} (ADR 0069 decision 2)`,
      );
    }
    if (rowLength % groupSize !== 0) {
      fail(
        `${where}: row length ${rowLength} (= numel / shape[${rowAxis}]) is not divisible by groupSize ${groupSize} (ADR 0069 decision 2)`,
      );
    }
  }
  const scaleRef = encoding.scale;
  if (scaleRef === undefined) fail(`${where}: codec '${encoding.codec}' requires a scale`);
  const scalePayload = declaredScaleBytes(
    numelOf(groupScaleShape(shape, rowAxis, groupSize)),
    `${where} scale`,
  );
  let scale: ScaleSupply;
  if (scaleRef.pieces === undefined) {
    const scaleFound = locate(scaleRef.block, `${where} scale`);
    assertPadded(scaleFound.record, scalePayload, `${where} scale`);
    scale = {
      id: scaleFound.record.id,
      part: scaleFound.part,
      offset: scaleFound.record.offset,
      length: scaleFound.record.length,
      rows: [0, rowCount],
      payloadBytes: scalePayload,
    };
  } else {
    scale = planSplitScale(where, supply.pieces !== undefined, blocks, scaleRef.pieces, {
      rowCount,
      payload: scalePayload,
      locate,
    });
  }
  if (encoding.zeroPoint === undefined) return { encoding, blocks, scale, origin };
  const zeroFound = locate(encoding.zeroPoint.block, `${where} zeroPoint`);
  const zeroPoint: SupplyBlock = {
    id: zeroFound.record.id,
    part: zeroFound.part,
    offset: zeroFound.record.offset,
    length: zeroFound.record.length,
    rows: [0, rowCount],
    payloadBytes: zeroFound.record.length,
  };
  return { encoding, blocks, scale, zeroPoint, origin };
};

/**
 * Merge result → the graph the runtime executes (an {@link IrGraph} with the storage of every
 * initializer settled). Copies `codec` / `groupSize` / `rowAxis` from the binding table's
 * `encoding` and passes shared declarations through. A `krg` (no weight supply) can be built if it
 * has only const supplies and shared declarations — an initializer that needs weights has no supply
 * and cannot become an `IrGraph` (fails loudly).
 */
export const mergedGraph = (bound: BoundGraph, graphName: string): IrGraph => {
  const initializers: Record<string, IrInitializer> = Object.create(null);
  for (const [name, init] of Object.entries(bound.declaration.initializers)) {
    if (init.shared) {
      initializers[name] = { shared: true };
      continue;
    }
    const supply = bound.supplies.get(name);
    if (supply === undefined) {
      fail(
        `graph '${graphName}' initializer '${name}': no weight supply (a krg alone cannot build a Session)`,
      );
    }
    const { codec, groupSize, rowAxis } = supply.encoding;
    initializers[name] = {
      storage: {
        codec,
        ...(groupSize === undefined ? {} : { groupSize }),
        ...(rowAxis === undefined ? {} : { rowAxis }),
      },
    };
  }
  return { ...bound.declaration, initializers };
};

/**
 * The extra condition of a `ternary` declaration (container-v1 §6.3): every 2-bit code of the
 * payload is in `{1, 2, 3}` (code 0 = q − 2 is outside the ternary range). The checkable content of
 * the claim "this is ternary".
 */
export const assertTernaryCodes = (payload: Uint8Array<ArrayBuffer>, where: string): void => {
  for (let i = 0; i < payload.byteLength; i += 1) {
    const byte = payload[i];
    if ((byte & 3) === 0 || (byte & 12) === 0 || (byte & 48) === 0 || (byte & 192) === 0) {
      fail(`${where}: ternary payload has code 0 (outside the ternary range) (byte ${i})`);
    }
  }
};

/**
 * The body that merges declarations and supplies — **independent of the shape of the 2 documents**
 * (a container's index and the in-memory container, which supplies its own bytes, go through the
 * same function). Its inputs are only four: graph declarations, const bindings, the binding table
 * and the block locator; the rules placed here are exactly the list in the module doc.
 *
 * When `binding` is `undefined` it is the form without a binding table (`krg`) — initializers other
 * than const supplies and shared ones stay as "needs weights" declarations without a supply plan (no
 * Session can be built).
 */
export const bindDeclarations = (input: {
  readonly graphs: Readonly<Record<string, IrDeclaration>>;
  readonly constants: readonly ConstantBinding[];
  readonly binding: ModelDescriptor["binding"] | undefined;
  readonly locate: Locator;
}): Readonly<Record<string, BoundGraph>> => {
  const { graphs, constants, binding, locate } = input;
  const constSupply = new Map<string, ConstantBinding>();
  for (const entry of constants) {
    constSupply.set(`${entry.graph}/${entry.initializer}`, entry);
  }

  // null prototype MUST: keeps assigning the graph name `__proto__` from replacing the prototype
  // (on Node and in browsers).
  const out: Record<string, BoundGraph> = Object.create(null);
  for (const [graphName, declaration] of Object.entries(graphs)) {
    const supplies = new Map<string, InitializerSupply>();
    for (const [name, init] of Object.entries(declaration.initializers)) {
      if (init.shared) continue;
      const where = `graph '${graphName}' initializer '${name}'`;
      const value = declaration.values[name];
      const shape = value.shape.map(Number);
      const constant = constSupply.get(`${graphName}/${name}`);
      if (constant !== undefined) {
        supplies.set(
          name,
          planSupply(
            where,
            value.dtype,
            shape,
            { block: constant.block, encoding: constant.encoding },
            locate,
            "const",
          ),
        );
        continue;
      }
      const graphBinding =
        binding !== undefined && Object.hasOwn(binding, graphName) ? binding[graphName] : undefined;
      const supply =
        graphBinding !== undefined && Object.hasOwn(graphBinding, name)
          ? graphBinding[name]
          : undefined;
      if (supply === undefined) {
        // In a krg (no binding table) a non-const initializer is a "needs weights" declaration — no
        // supply plan.
        if (binding === undefined) continue;
        return fail(`${where}: no supply in the binding table`);
      }
      supplies.set(name, planSupply(where, value.dtype, shape, supply, locate, "model"));
    }
    out[graphName] = { declaration, supplies };
  }
  return out;
};

/**
 * Merges every graph (the surface over the 2 documents). Assumes the structural checks
 * (`descriptor.ts`) are done; this is a thin wrapper that builds the const bindings, the binding
 * table and the block locator from the 2 documents and passes them to {@link bindDeclarations}.
 */
export const bindGraphs = (
  graph: GraphDescriptor,
  model: ModelDescriptor | undefined,
): Readonly<Record<string, BoundGraph>> => {
  const constById = new Map(graph.const.blocks.map((block) => [block.id, block]));
  const dataById = new Map((model?.blocks ?? []).map((block) => [block.id, block]));
  const locate: Locator = (id, by) => {
    const constBlock = constById.get(id);
    if (constBlock !== undefined) return { part: 1, record: constBlock };
    const dataBlock = dataById.get(id);
    if (dataBlock !== undefined) return { part: dataBlock.part, record: dataBlock };
    return fail(`${by}: undeclared block '${id}'`);
  };
  return bindDeclarations({
    graphs: graph.graphs,
    constants: graph.const.constants,
    binding: model?.binding,
    locate,
  });
};
