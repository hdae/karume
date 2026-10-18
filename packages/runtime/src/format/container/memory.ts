/**
 * In-memory container — the supply surface that hands bytes already at hand (synthesized weights,
 * data read out of another buffer) to Session construction **without writing a container**.
 *
 * The surface is the same two members as the `OpenedContainer` of {@link "./open.ts"} for an opened
 * `krm` (`graphs` and `readBlock` = `BoundContainer`), and the binding rules **still have one
 * owner**, the merge layer ({@link bindDeclarations}). What this module does is check the
 * "declaration → supply" correspondence and **synthesize** the block index the merge layer needs;
 * rules ①–④, codec × semantic dtype, payload lengths and piece coverage are all checked by the merge
 * layer (no second implementation — if only one side were fixed, the accepted sets would split).
 *
 * However, **the checks confined to the synthesis itself belong to this layer** (the path that opens
 * a container has no counterpart, and the merge layer cannot see them) — five of them: the
 * correspondence with the declarations (missing / surplus non-shared initializers, unknown graph
 * names), `pieces` having 2 or more entries, collisions of synthesized ids (the equivalent of rule
 * ①), the alignment of the companion scale's byte position, and the check of the length `readBlock`
 * returns against the synthesized declared length ({@link openMemoryContainer}).
 *
 * **No sha256 is computed**: the bytes the caller passed went through neither the network nor a
 * disk, so there is no "mismatch between declaration and actual data" to verify (the per-block
 * digest of a `krm` is a check that presumes an unverified source — container-v1 §7).
 */

import {
  bindDeclarations,
  type BoundContainer,
  type Locator,
  pieceRowBytes,
  type SupplyRecord,
} from "./bind.ts";
import { codecEntry, type CodecName } from "./codecs.ts";
import type { Encoding, WeightSupply } from "./descriptor.ts";
import { ContainerFormatError } from "./header.ts";
import { DEFAULT_PART_BYTES } from "./limits.ts";
import type { IrDeclaration } from "../ir.ts";

/**
 * Storage declaration (the fields of `Encoding` the caller decides — `packing` is copied from the
 * codec ledger).
 *
 * NOTE: there is **no** `zeroPoint` field (all 4 codecs of the ledger are `zeroPoint: "forbidden"`,
 * so today no form exists that cannot be expressed). When a codec for asymmetric quantization is
 * added to the ledger, look at this too — otherwise a form that a `krm` can express but the
 * in-memory container cannot quietly appears.
 */
export type MemoryEncoding = {
  readonly codec: CodecName;
  /**
   * Quantized codecs only; 0 when omitted. Only `int8-sym` may have 1 (`int4-sym-g` / `int2-off` /
   * `ternary` allow 0 only — the merge layer rejects others).
   */
  readonly rowAxis?: 0 | 1;
  /** Quantized codecs only; the row length for per-channel. */
  readonly groupSize?: number;
  /** Required for quantized codecs (f32, laid out as rank 2 `[rows, groups]`). */
  readonly scale?: Uint8Array<ArrayBuffer>;
};

/**
 * One piece of a split supply (a half-open range of the first dimension and a reader for that
 * range's payload).
 */
export type MemoryPiece = {
  /** Half-open range of the first dimension. */
  readonly rows: readonly [number, number];
  /** The payload of that piece (no padding). */
  read(): Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * The supply of one initializer. Either whole (`bytes`) or split by rows (`pieces` — 2 or more),
 * exclusively.
 *
 * Split values are "readers", not data: {@link openMemoryContainer} holds no bytes and calls
 * `read()` on every `readBlock`. A supply that uses only one buffer's worth of RAM (a caller that
 * reads a file partially, for example) is passed in this form.
 */
export type MemoryTensor =
  | {
      readonly encoding: MemoryEncoding;
      readonly bytes: Uint8Array<ArrayBuffer>;
      readonly pieces?: undefined;
    }
  | {
      readonly encoding: MemoryEncoding;
      readonly bytes?: undefined;
      readonly pieces: readonly MemoryPiece[];
    };

export type MemoryContainerInput = {
  /**
   * **Every** graph passed here needs a supply (missing even one non-shared initializer fails
   * loudly). A caller that wants a Session for only one graph of a multi-graph model narrows
   * `graphs` itself to that one — a graph with declarations only (the equivalent of a `krg`) cannot
   * be made with the in-memory container.
   */
  readonly graphs: Readonly<Record<string, IrDeclaration>>;
  /** Graph name → initializer name → supply. Every non-shared initializer is required. */
  readonly tensors: Readonly<Record<string, Readonly<Record<string, MemoryTensor>>>>;
};

/**
 * The part where data supplied whole (`bytes`) starts to be stacked.
 *
 * The part split matters because Session construction (`containerBatches` → `buildSessionState`)
 * **issues one submit per part and waits for it to complete**. `queue.writeBuffer` allocates and
 * accumulates staging and does not release it until the submit completes (the MUST of the batch
 * loop in `session-build.ts` — uploading several GiB at once enters the first run with VRAM still
 * double-counted, and the first peak rises by one weight's worth). The part split is therefore
 * exactly **the upper bound of staging VRAM during construction**, an axis separate from whether the
 * host already holds the bytes.
 *
 * - Whole supplies are stacked in declaration order and move to the next part where the running
 *   total exceeds {@link DEFAULT_PART_BYTES} (the same step as a container writer cutting by part
 *   length). The companion scale goes in the same part as the data.
 * - `pieces` are **one piece = one part = one fence**. This split keeps staging to one piece's
 *   worth; host RAM does not depend on the part split (Session construction uploads and drops one
 *   item at a time — container-v1 §11). As rule ③ says, the scale goes in the same part as piece 1.
 *
 * The running total of whole supplies is shared across graphs (`containerBatches` consumes only one
 * graph's worth, so in a multi-graph container this only adds extra fences — no under-splitting that
 * pushes staging over the threshold happens).
 */
const WHOLE_PART = 2;

/** dtype of the companion scale (only f32 in the first edition — container-v1 §6.1). */
const SCALE_DTYPE = "f32" as const;

/**
 * Alignment of the scale's byte position (required by the side that creates the f32 view —
 * `scaleTensor` in `session-build.ts`).
 */
const SCALE_ALIGN = 4;

/**
 * The caller's declaration → `Encoding` (`packing` is a copy of the codec ledger). The rules are the
 * same as the container writer's: a non-quantized codec may not have scale / groupSize / rowAxis,
 * and a quantized codec needs scale and groupSize.
 */
const resolveEncoding = (
  input: MemoryEncoding,
  scaleBlock: string | undefined,
  where: string,
): Encoding => {
  const entry = codecEntry(input.codec);
  if (entry.scale === "forbidden") {
    if (input.scale !== undefined || input.groupSize !== undefined || input.rowAxis !== undefined) {
      throw new ContainerFormatError(
        `${where}: codec '${input.codec}' は scale / groupSize / rowAxis を持てない`,
      );
    }
    return { codec: input.codec, packing: entry.packing };
  }
  if (input.scale === undefined || input.groupSize === undefined || scaleBlock === undefined) {
    throw new ContainerFormatError(`${where}: codec '${input.codec}' は scale と groupSize が要る`);
  }
  return {
    codec: input.codec,
    packing: entry.packing,
    rowAxis: input.rowAxis ?? 0,
    groupSize: input.groupSize,
    scale: { block: scaleBlock, dtype: SCALE_DTYPE },
  };
};

/**
 * Builds the supply surface from bytes at hand (synchronous; reads are lazy).
 *
 * The block index is synthesized: the id is `<graph>/<initializer>`, a piece is
 * `<graph>/<initializer>#piece<k>` (k starts at 1), the scale is `<graph>/<initializer>#scale`.
 * `length` is the payload (no padding) and `offset` is 0 — in memory there is no "contiguous region
 * to cut ranges from".
 *
 * Only **what this layer synthesized itself** is checked here — the correspondence with the
 * declarations (missing / surplus non-shared initializers, unknown graph names), `pieces` having 2
 * or more entries, collisions of synthesized ids, the alignment of the scale's byte position, and the
 * check of the length `readBlock` returns against the index's declared length. The binding rules
 * themselves are checked by the merge layer (module doc).
 */
export const openMemoryContainer = (input: MemoryContainerInput): BoundContainer => {
  const located = new Map<
    string,
    {
      readonly part: number;
      readonly record: SupplyRecord;
      readonly read: () => Promise<Uint8Array<ArrayBuffer>>;
    }
  >();
  const register = (
    id: string,
    part: number,
    length: number,
    read: () => Promise<Uint8Array<ArrayBuffer>>,
  ): void => {
    // The equivalent of rule ①: a collision of synthesized ids fails loudly (overwriting with the
    // last one wins would dequantize one initializer's data with another's scale, a silent wrong
    // value). Collisions come from names that contain the separators used to build the id (`#` or
    // `/` in an initializer name / graph name); the naming rules themselves are not changed.
    if (located.has(id)) {
      throw new ContainerFormatError(
        `合成した block '${id}' が二重に束縛されている（1 block ≤ 1 binding — initializer 名 / グラフ名が id の区切り '#' / '/' を含んでいる）`,
      );
    }
    located.set(id, { part, record: { id, offset: 0, length }, read });
  };

  for (const graphName of Object.keys(input.tensors)) {
    if (!Object.hasOwn(input.graphs, graphName)) {
      throw new ContainerFormatError(`供給に未宣言のグラフ '${graphName}' がある`);
    }
  }

  // Name maps MUST have a null prototype (on Node and in browsers, assigning the name `__proto__`
  // replaces the prototype and the supply silently disappears — the same practice as the readers
  // descriptor.ts / ir.ts).
  const binding: Record<string, Record<string, WeightSupply>> = Object.create(null);
  // Handing out parts (see the doc of {@link WHOLE_PART}). Whole supplies are stacked on
  // `wholePart`, and each of the `pieces` takes one part from `nextPart`.
  let wholePart = WHOLE_PART;
  let wholeBytes = 0;
  let nextPart = WHOLE_PART + 1;
  /** The part for one whole supply (data + companion scale). On overflow, the next part. */
  const takeWholePart = (bytes: number): number => {
    if (wholeBytes > 0 && wholeBytes + bytes > DEFAULT_PART_BYTES) {
      wholePart = nextPart;
      nextPart += 1;
      wholeBytes = 0;
    }
    wholeBytes += bytes;
    return wholePart;
  };
  for (const [graphName, declaration] of Object.entries(input.graphs)) {
    const supplied = Object.hasOwn(input.tensors, graphName) ? input.tensors[graphName] : {};
    // The set checked is "non-shared initializers" — a shared one is a declaration without bytes
    // (it borrows the lender's weight), so supplying it fails as surplus.
    const expected = Object.entries(declaration.initializers)
      .filter(([, init]) => !init.shared)
      .map(([name]) => name);
    // Both missing and surplus items are **enumerated in full** (failing one at a time would leave
    // the side building the supply unaware of how many are missing).
    const missing = expected.filter((name) => !Object.hasOwn(supplied, name));
    const surplus = Object.keys(supplied).filter((name) => !expected.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      throw new ContainerFormatError(
        `グラフ '${graphName}' の供給が initializer 宣言と一致しない: 不足 [${missing.join(
          ", ",
        )}] / 余剰 [${surplus.join(", ")}]（shared 宣言は供給しない）`,
      );
    }
    const supplies: Record<string, WeightSupply> = Object.create(null);
    for (const name of expected) {
      const tensor = supplied[name];
      const where = `graph '${graphName}' initializer '${name}'`;
      const id = `${graphName}/${name}`;
      const scale = tensor.encoding.scale;
      const encoding = resolveEncoding(
        tensor.encoding,
        scale === undefined ? undefined : `${id}#scale`,
        where,
      );
      if (scale !== undefined && scale.byteOffset % SCALE_ALIGN !== 0) {
        // GPU construction creates the scale's `Float32Array` view without a copy (the MUST of
        // `scaleTensor` in `session-build.ts`: "the supplier guarantees 4-byte alignment of the byte
        // position"). Letting an unaligned view through raises a bare RangeError there, disguised
        // as a message from the transfer layer.
        throw new ContainerFormatError(
          `${where}: scale の byteOffset ${scale.byteOffset} が ${SCALE_ALIGN} バイト整列でない（${SCALE_ALIGN} バイト整列した view を渡す）`,
        );
      }
      let scalePart: number;
      if (tensor.pieces === undefined) {
        // Data and companion scale share a part (the staging of one whole supply is counted
        // together).
        scalePart = takeWholePart(tensor.bytes.byteLength + (scale?.byteLength ?? 0));
        register(id, scalePart, tensor.bytes.byteLength, () => Promise.resolve(tensor.bytes));
        supplies[name] = { block: id, encoding };
      } else {
        if (tensor.pieces.length < 2) {
          throw new ContainerFormatError(`${where}: pieces は 2 本以上（1 本なら bytes で渡す）`);
        }
        const shape = declaration.values[name].shape.map(Number);
        const rowBytes = pieceRowBytes(encoding.codec, shape, where);
        // Rule ③: the companion scale is in the same part as piece 1.
        const firstPart = nextPart;
        scalePart = firstPart;
        const pieces = tensor.pieces.map((piece, index) => {
          const block = `${id}#piece${index + 1}`;
          register(block, firstPart + index, (piece.rows[1] - piece.rows[0]) * rowBytes, () =>
            piece.read(),
          );
          return { block, rows: piece.rows };
        });
        nextPart += tensor.pieces.length;
        supplies[name] = { pieces, encoding };
      }
      if (scale !== undefined) {
        register(`${id}#scale`, scalePart, scale.byteLength, () => Promise.resolve(scale));
      }
    }
    binding[graphName] = supplies;
  }

  const locate: Locator = (id, by) => {
    const found = located.get(id);
    if (found === undefined) throw new ContainerFormatError(`${by}: 未宣言の block '${id}'`);
    return found;
  };

  return {
    graphs: bindDeclarations({
      graphs: input.graphs,
      // The const region is a concept of graph containers only (`krg` / part 1 of a `krm`); the
      // in-memory container has none — constants are supplied from `tensors` like weights.
      constants: [],
      binding,
      locate,
    }),
    readBlock: async (id) => {
      const found = located.get(id);
      if (found === undefined) throw new ContainerFormatError(`未宣言の block '${id}'`);
      // Whole data and scales return **the passed buffer as is** (no copy — the MUST of
      // `BoundContainer`). A piece calls `read()` again on every call (nothing kept = one buffer's
      // worth of RAM).
      const bytes = await found.read();
      // MUST: fetched length = the declared length of the synthesized index (the same check as
      // `readBlock` in `open.ts`). A piece's length is derived from the declaration, so even if
      // `read()` returns longer bytes the merge layer's check is trivially true, and the
      // `subarray(0, payloadBytes)` of `containerBatches` silently truncates — a caller's mix-up of
      // row range and reader would land on the GPU as a silent wrong value.
      if (bytes.byteLength !== found.record.length) {
        throw new ContainerFormatError(
          `block '${id}': 取得長 ${bytes.byteLength} が宣言 ${found.record.length} と違う`,
        );
      }
      return bytes;
    },
  };
};
