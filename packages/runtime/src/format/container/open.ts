/**
 * Opens a container — `BlockSource` → header → verify the 2 documents → parse → merge → fetch blocks
 * → extract `krg`. docs/container-v1.md §7 / §8 / §9 / §11.
 *
 * Order ("fetch only the graph first, admission, then weights"):
 * 1. Read the header from the first 24 B of part 0.
 * 2. Verify the graph descriptor / model descriptor against **the outer expected hash + length**
 *    first, then parse them (values held by the manifest's FileRef or the caller's pin. A
 *    descriptor cannot hold its own hash — §7).
 * 3. Pass the checks determined by declarations alone (consistency of the 2 documents, binding
 *    checks, payload lengths).
 * 4. Fetch blocks when needed. When the source is unverified ({@link BlockSource.verified} is false),
 *    verify the sha256 of each fetched block (for a verified source the fetch layer has verified the
 *    whole file — ADR 0109 decision 7).
 *
 * `BlockSource` puts "the whole bytes of the single form" and "the bytes per part" behind one
 * interface. The absolute part offsets of the single form are derived by {@link derivePartOffsets},
 * shared with the writer (the descriptor does not record absolute offsets, so the descriptor is
 * byte-identical in the single and split forms — §8).
 */

import {
  assertRangeInBlock,
  bindGraphs,
  type BlockRangeReader,
  type BoundGraph,
  type SupplyBlock,
} from "./bind.ts";
import {
  asSha256Hex,
  type ConstBlockRecord,
  type DataBlockRecord,
  type GraphDescriptor,
  type ModelDescriptor,
  parseGraphDescriptor,
  parseModelDescriptor,
  sha256Hex,
  validateAgainstGraph,
} from "./descriptor.ts";
import {
  assembleGraphContainer,
  ContainerFormatError,
  type ContainerHeader,
  derivePartOffsets,
  part0Length,
  readHeader,
} from "./header.ts";
import { HEADER_BYTES, MAX_SINGLE_CONTAINER_BYTES } from "./limits.ts";

/**
 * The accessor from part → byte range. The fetch layer (hub / local file / memory) implements only
 * this interface. `read` may return **copied bytes** or a view of the source's buffer — callers do
 * not modify the returned bytes.
 *
 * MUST: `read` takes concurrent calls, and bytes it returned stay unchanged after later calls (do
 * not reuse one buffer across calls). Session construction keeps up to 3 block reads in flight,
 * across part boundaries (`READ_AHEAD_BLOCKS` in `runtime/executor.ts` — ADR 0108 addendum 14).
 */
export type BlockSource = {
  readonly partCount: number;
  /**
   * Whether the source has **already verified** the part bytes (ADR 0109 decision 7 / container-v1
   * §7). When true, {@link OpenedContainer.readBlock} does not compute block sha256 (the path where
   * the fetch layer verified the whole file and compared it with the recorded hash — avoiding the
   * double digest on cold and the digest on warm). A source nobody verified, such as whole bytes / a
   * part list / a local directory, MUST be false — silently claiming true lets tampering through.
   */
  readonly verified: boolean;
  partLength(index: number): number;
  read(part: number, offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * Input forms. The single form (whole bytes) and the split form (part list) share one entry point.
 */
export type ContainerInput =
  | { readonly kind: "bytes"; readonly bytes: Uint8Array<ArrayBuffer> }
  | { readonly kind: "parts"; readonly parts: readonly Uint8Array<ArrayBuffer>[] }
  | { readonly kind: "source"; readonly source: BlockSource };

/**
 * Expected values of the 2 documents held outside (`container.descriptor` of manifest `karume/5` /
 * the caller's pin) (§7 step 1). `sha256` is 64 lowercase hex characters (the format is checked
 * here — callers are not required to pass a branded type).
 */
export type DescriptorExpectation = {
  readonly graph: { readonly length: number; readonly sha256: string };
  /**
   * Required for a `krm`, must not be passed for a `krg` (`openContainer` fails with a
   * ContainerFormatError in both cases — the container kind is unknown until the header is read, so
   * this is checked when opening, not by the type).
   */
  readonly model?: { readonly length: number; readonly sha256: string };
};

/**
 * Reader for one asset (models interprets the role — container-v1 §2.2. The runtime knows only
 * name → block).
 */
export type AssetReader = {
  readonly role: string;
  /** Payload byte count (the logical length — the declared value; block padding not included). */
  readonly length: number;
  /**
   * Returns `[offset, offset + length)`. A verified source fetches only the range (the block is not
   * read whole — the interface for consumers that pull a few KB of rows from a table of hundreds of
   * MiB). An unverified source fetches the block once, verifies its sha256, keeps it while this
   * reader is alive and slices from it (how many are kept is decided by the caller through the
   * readers' lifetimes).
   *
   * MUST: the result is not modified. A verified source returns a view of the source's buffer as is
   * (on the hub's scan path, a view holding the whole part), so a caller that keeps it longer than
   * the range copies it — holding on keeps the whole buffer alive, not just the range. With an
   * unverified source the result is also a view (`subarray`) of the kept block, and that block can
   * itself be a view of the source's buffer (a local directory whose adapter has no positional reads
   * (`readFileRange`) is on the hub's scan path, so it holds the whole part's buffer) — copy for the
   * same reason.
   */
  read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/** An opened container. */
export type OpenedContainer = {
  readonly header: ContainerHeader;
  readonly graph: GraphDescriptor;
  /** undefined for a `krg`. */
  readonly model: ModelDescriptor | undefined;
  /** Graph name → merge result (IR v2 declaration + the supply plan per initializer). */
  readonly graphs: Readonly<Record<string, BoundGraph>>;
  readonly source: BlockSource;
  /** Location of one block (part and index row). */
  locate(id: string): {
    readonly part: number;
    readonly record: ConstBlockRecord | DataBlockRecord;
  };
  /**
   * Fetches and returns one block. When the source is not verified ({@link BlockSource.verified} is
   * false), its sha256 is verified (§7 — only unverified sources get a single digest per block).
   *
   * The id of a split scale (`InitializerSupply.scale.id` with `scale.pieces` set — ADR 0108
   * addendum 15) returns its pieces concatenated in row order into one new buffer, each piece read
   * and verified like a block (one at a time, so the read holds the whole scale + one piece).
   */
  readBlock(id: string): Promise<Uint8Array<ArrayBuffer>>;
  /**
   * Range reads inside one declared block, straight from the source (ADR 0146 decision 5): no block
   * digest, so on an unverified source `verified` is false and the caller checks the bytes itself.
   * The cost class is `"seek"` for the bytes / parts input (slices of buffers in hand) and unknown
   * for a `BlockSource` passed in (its type carries no cost class).
   */
  readonly ranges: BlockRangeReader;
  /**
   * Opens the reader for one asset (a `krg` and undeclared names fail loudly). Opening fetches no
   * byte.
   */
  asset(name: string): AssetReader;
  /**
   * Assembles and returns `[header'][graph descriptor][padding][const region]` (byte-copy
   * extraction of a `krg` — §9). The only extraction source is a `krm`; a container opened from a
   * `krg` fails loudly (there is no hash of the whole const region).
   *
   * NOTE: this function holds the graph descriptor bytes for the whole lifetime of the opened
   * container. When the source returns views of its buffer (the hub's scan path), it holds part 0's
   * whole buffer, but part 0 is exactly the header and the 2 documents, so the extra amount stays
   * around the length of the model descriptor.
   */
  extractGraph(): Promise<Uint8Array<ArrayBuffer>>;
};

const sliceExact = (
  bytes: Uint8Array<ArrayBuffer>,
  offset: number,
  length: number,
  path: string,
): Uint8Array<ArrayBuffer> => {
  if (offset + length > bytes.byteLength) {
    throw new ContainerFormatError(
      `${path}: ${offset}..${offset + length} runs past the length ${bytes.byteLength}`,
    );
  }
  return bytes.subarray(offset, offset + length);
};

/**
 * `BlockSource` of the single form (whole bytes). Absolute part offsets come from the declared part
 * lengths + the alignment rule (§8).
 */
export const bytesSource = (
  bytes: Uint8Array<ArrayBuffer>,
  part0Bytes: number,
  partLengths: readonly number[],
): BlockSource => {
  const layout = derivePartOffsets(part0Bytes, partLengths);
  if (bytes.byteLength !== layout.totalLength) {
    throw new ContainerFormatError(
      `Single-form length ${bytes.byteLength} differs from ${layout.totalLength} derived from the declarations`,
    );
  }
  const lengths = [part0Bytes, ...partLengths];
  const partLength = (index: number): number => {
    const length = lengths[index];
    if (length === undefined) throw new ContainerFormatError(`part ${index} does not exist`);
    return length;
  };
  return {
    partCount: lengths.length,
    verified: false,
    partLength,
    read: (part, offset, length) => {
      if (offset + length > partLength(part)) {
        throw new ContainerFormatError(
          `part ${part}: ${offset}..${offset + length} runs past the part length`,
        );
      }
      return Promise.resolve(
        sliceExact(bytes, layout.offsets[part] + offset, length, `part ${part}`),
      );
    },
  };
};

/** `BlockSource` of the split form (bytes per part). */
export const partsSource = (parts: readonly Uint8Array<ArrayBuffer>[]): BlockSource => {
  const partOf = (index: number): Uint8Array<ArrayBuffer> => {
    const part = parts[index];
    if (part === undefined) throw new ContainerFormatError(`part ${index} does not exist`);
    return part;
  };
  return {
    partCount: parts.length,
    verified: false,
    partLength: (index) => partOf(index).byteLength,
    read: (part, offset, length) =>
      Promise.resolve(sliceExact(partOf(part), offset, length, `part ${part}`)),
  };
};

const verifyDocument = async (
  bytes: Uint8Array<ArrayBuffer>,
  expected: { readonly length: number; readonly sha256: string },
  path: string,
): Promise<void> => {
  const expectedHash = asSha256Hex(expected.sha256, `${path} expected sha256`);
  if (bytes.byteLength !== expected.length) {
    throw new ContainerFormatError(
      `${path} length ${bytes.byteLength} differs from the expected ${expected.length}`,
    );
  }
  const actual = await sha256Hex(bytes);
  if (actual !== expectedHash) {
    throw new ContainerFormatError(
      `${path} sha256 differs from the expected value (expected ${expected.sha256} / actual ${actual})`,
    );
  }
};

const initialSource = (input: ContainerInput): BlockSource => {
  if (input.kind === "source") return input.source;
  if (input.kind === "parts") return partsSource(input.parts);
  if (input.bytes.byteLength > MAX_SINGLE_CONTAINER_BYTES) {
    throw new ContainerFormatError(
      `Single-form total of ${input.bytes.byteLength} bytes exceeds the limit ${MAX_SINGLE_CONTAINER_BYTES} (read it in split form)`,
    );
  }
  // In the single form the part lengths are unknown until the descriptor is read. Build a
  // provisional source that can read only the start.
  return {
    partCount: 1,
    verified: false,
    partLength: () => input.bytes.byteLength,
    read: (_part, offset, length) =>
      Promise.resolve(sliceExact(input.bytes, offset, length, "part 0")),
  };
};

/**
 * Opens a container.
 *
 * `expect` holds the expected values from outside (§7). When omitted, **nobody guarantees** the
 * integrity of the 2 documents, so pass it everywhere except tests and local loading.
 */
export const openContainer = async (
  input: ContainerInput,
  expect?: DescriptorExpectation,
): Promise<OpenedContainer> => {
  const head = initialSource(input);
  if (head.partLength(0) < HEADER_BYTES) {
    throw new ContainerFormatError(
      `part 0 is shorter than ${HEADER_BYTES} bytes: ${head.partLength(0)}`,
    );
  }
  const header = readHeader(await head.read(0, 0, HEADER_BYTES));
  const part0Bytes = part0Length(header);
  if (head.partLength(0) < part0Bytes) {
    throw new ContainerFormatError(
      `part 0 is shorter than ${part0Bytes} bytes: ${head.partLength(0)}`,
    );
  }
  const graphBytes = await head.read(0, HEADER_BYTES, header.graphDescriptorLength);
  if (expect !== undefined) await verifyDocument(graphBytes, expect.graph, "graph descriptor");
  const graph = parseGraphDescriptor(graphBytes);

  let model: ModelDescriptor | undefined;
  if (header.kind === "model") {
    // MUST: when expected values are passed, pass them for both documents (§7 step 1). Passing only
    // graph would parse the model descriptor, which holds the binding table and the block sha256s,
    // unverified (symmetric with the krg-side check).
    if (expect !== undefined && expect.model === undefined) {
      throw new ContainerFormatError(
        "krm but no expected values for the model descriptor were passed",
      );
    }
    const modelBytes = await head.read(
      0,
      HEADER_BYTES + header.graphDescriptorLength,
      header.modelDescriptorLength,
    );
    if (expect?.model !== undefined)
      await verifyDocument(modelBytes, expect.model, "model descriptor");
    model = parseModelDescriptor(modelBytes);
    validateAgainstGraph(model, graph);
  } else if (expect?.model !== undefined) {
    throw new ContainerFormatError("krg but expected values for the model descriptor were passed");
  }
  const graphs = bindGraphs(graph, model);

  // Part 1 of a krg is the const region itself (§3). For a krm, the declared lengths of parts.
  const partLengths =
    model === undefined ? [graph.const.length] : model.parts.map((part) => part.length);
  const source = input.kind === "bytes" ? bytesSource(input.bytes, part0Bytes, partLengths) : head;
  if (source.partCount !== partLengths.length + 1) {
    throw new ContainerFormatError(
      `part count is ${source.partCount} but ${partLengths.length + 1} are declared`,
    );
  }
  if (input.kind !== "bytes") {
    // Split form / fetch layer: each part's actual length must match its declaration (for the
    // single form bytesSource checks the total length).
    if (source.partLength(0) !== part0Bytes) {
      throw new ContainerFormatError(
        `part 0 length ${source.partLength(0)} differs from the declared ${part0Bytes}`,
      );
    }
    for (const [i, length] of partLengths.entries()) {
      if (source.partLength(i + 1) !== length) {
        throw new ContainerFormatError(
          `part ${i + 1} length ${source.partLength(i + 1)} differs from the declared ${length}`,
        );
      }
    }
  }

  const located = new Map<
    string,
    { readonly part: number; readonly record: ConstBlockRecord | DataBlockRecord }
  >();
  for (const block of graph.const.blocks) located.set(block.id, { part: 1, record: block });
  for (const block of model?.blocks ?? []) {
    located.set(block.id, { part: block.part, record: block });
  }

  const locate: OpenedContainer["locate"] = (id) => {
    const found = located.get(id);
    if (found === undefined) throw new ContainerFormatError(`Undeclared block '${id}'`);
    return found;
  };

  const readDeclared = async (id: string): Promise<Uint8Array<ArrayBuffer>> => {
    const found = locate(id);
    const bytes = await source.read(found.part, found.record.offset, found.record.length);
    if (bytes.byteLength !== found.record.length) {
      throw new ContainerFormatError(
        `block '${id}': fetched length ${bytes.byteLength} differs from the declared ${found.record.length}`,
      );
    }
    if (source.verified) return bytes;
    const actual = await sha256Hex(bytes);
    if (actual !== found.record.sha256) {
      throw new ContainerFormatError(
        `block '${id}' sha256 differs from the declared value (declared ${found.record.sha256} / actual ${actual})`,
      );
    }
    return bytes;
  };

  // Split scales are read whole under their joined id (ADR 0108 addendum 15): construction takes
  // one contiguous scale per binding, exactly as for an unsplit one.
  const joinedScales = new Map<string, readonly SupplyBlock[]>();
  for (const bound of Object.values(graphs)) {
    for (const supply of bound.supplies.values()) {
      if (supply.scale?.pieces !== undefined)
        joinedScales.set(supply.scale.id, supply.scale.pieces);
    }
  }

  const readBlock = async (id: string): Promise<Uint8Array<ArrayBuffer>> => {
    const pieces = joinedScales.get(id);
    if (pieces === undefined) return readDeclared(id);
    const joined = new Uint8Array(
      new ArrayBuffer(pieces.reduce((total, piece) => total + piece.length, 0)),
    );
    let cursor = 0;
    // One piece at a time: the read holds the joined scale + one piece (each within the block
    // limit), and the source sees one read at a time per joined scale.
    for (const piece of pieces) {
      const bytes = await readDeclared(piece.id);
      joined.set(bytes, cursor);
      cursor += bytes.byteLength;
    }
    return joined;
  };

  // The bytes and parts inputs read slices of buffers already in hand; a caller's source says nothing.
  const rangeCost = input.kind === "source" ? undefined : "seek";
  const ranges: BlockRangeReader = {
    verified: source.verified,
    cost: () => rangeCost,
    read: async (id, offset, length) => {
      const found = locate(id);
      assertRangeInBlock(id, found.record.length, offset, length);
      const bytes = await source.read(found.part, found.record.offset + offset, length);
      if (bytes.byteLength !== length) {
        throw new ContainerFormatError(
          `block '${id}': fetched range length ${bytes.byteLength} differs from the requested ${length}`,
        );
      }
      return bytes;
    },
  };

  const asset: OpenedContainer["asset"] = (name) => {
    const binding =
      model !== undefined && Object.hasOwn(model.assets, name) ? model.assets[name] : undefined;
    if (binding === undefined) throw new ContainerFormatError(`Undeclared asset '${name}'`);
    const found = locate(binding.block);
    let verifiedBlock: Promise<Uint8Array<ArrayBuffer>> | undefined;
    return {
      role: binding.role,
      length: binding.length,
      read: async (offset, length) => {
        // NaN makes every comparison false and slips past the range check, and subarray silently
        // truncates fractions.
        if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)) {
          throw new ContainerFormatError(
            `asset '${name}': range (offset ${offset}, length ${length}) is not a safe integer`,
          );
        }
        if (offset < 0 || length < 0 || offset + length > binding.length) {
          throw new ContainerFormatError(
            `asset '${name}': ${offset}..${offset + length} runs past the logical length ${binding.length}`,
          );
        }
        if (source.verified) {
          const bytes = await source.read(found.part, found.record.offset + offset, length);
          if (bytes.byteLength !== length) {
            throw new ContainerFormatError(
              `asset '${name}': fetched length ${bytes.byteLength} differs from the requested ${length}`,
            );
          }
          return bytes;
        }
        verifiedBlock ??= readDeclared(binding.block);
        return (await verifiedBlock).subarray(offset, offset + length);
      },
    };
  };

  const extractGraph = async (): Promise<Uint8Array<ArrayBuffer>> => {
    // §9 defines only a krm as the extraction source. A krg has no document holding the hash of the
    // whole const region, so the integrity of the extracted bytes cannot be checked (readBlock checks
    // per-block hashes, but nobody checks the padding outside the blocks).
    if (model === undefined) {
      throw new ContainerFormatError(
        "krg extraction works only from a krm (a krg input cannot be an extraction source — §9)",
      );
    }
    const regionLength = graph.const.length;
    const region =
      regionLength === 0
        ? new Uint8Array(new ArrayBuffer(0))
        : await source.read(1, 0, regionLength);
    if (region.byteLength !== regionLength) {
      throw new ContainerFormatError(
        `const region: fetched length ${region.byteLength} differs from the declared ${regionLength}`,
      );
    }
    // The integrity of part 1 is held by parts[0] of the model descriptor. Verify it before
    // extracting.
    const actual = await sha256Hex(region);
    if (actual !== model.parts[0].sha256) {
      throw new ContainerFormatError(
        `const region sha256 differs from the declared value (declared ${model.parts[0].sha256} / actual ${actual})`,
      );
    }
    return assembleGraphContainer(graphBytes.slice(), region.slice());
  };

  return { header, graph, model, graphs, source, locate, readBlock, ranges, asset, extractGraph };
};
