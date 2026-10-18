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

import { bindGraphs, type BoundGraph } from "./bind.ts";
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
   */
  readBlock(id: string): Promise<Uint8Array<ArrayBuffer>>;
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
      `${path}: ${offset}..${offset + length} が長さ ${bytes.byteLength} をはみ出す`,
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
      `単一形の長さ ${bytes.byteLength} が宣言から導いた ${layout.totalLength} と違う`,
    );
  }
  const lengths = [part0Bytes, ...partLengths];
  const partLength = (index: number): number => {
    const length = lengths[index];
    if (length === undefined) throw new ContainerFormatError(`part ${index} は無い`);
    return length;
  };
  return {
    partCount: lengths.length,
    verified: false,
    partLength,
    read: (part, offset, length) => {
      if (offset + length > partLength(part)) {
        throw new ContainerFormatError(
          `part ${part}: ${offset}..${offset + length} が part 長をはみ出す`,
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
    if (part === undefined) throw new ContainerFormatError(`part ${index} は無い`);
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
  const expectedHash = asSha256Hex(expected.sha256, `${path}の期待 sha256`);
  if (bytes.byteLength !== expected.length) {
    throw new ContainerFormatError(
      `${path}の長さ ${bytes.byteLength} が期待 ${expected.length} と違う`,
    );
  }
  const actual = await sha256Hex(bytes);
  if (actual !== expectedHash) {
    throw new ContainerFormatError(
      `${path}の sha256 が期待と違う（期待 ${expected.sha256} / 実物 ${actual}）`,
    );
  }
};

const initialSource = (input: ContainerInput): BlockSource => {
  if (input.kind === "source") return input.source;
  if (input.kind === "parts") return partsSource(input.parts);
  if (input.bytes.byteLength > MAX_SINGLE_CONTAINER_BYTES) {
    throw new ContainerFormatError(
      `単一形の全量 ${input.bytes.byteLength} バイトが上限 ${MAX_SINGLE_CONTAINER_BYTES} を超える（分割形で読む）`,
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
      `part 0 が ${HEADER_BYTES} バイトに足りない: ${head.partLength(0)}`,
    );
  }
  const header = readHeader(await head.read(0, 0, HEADER_BYTES));
  const part0Bytes = part0Length(header);
  if (head.partLength(0) < part0Bytes) {
    throw new ContainerFormatError(
      `part 0 が ${part0Bytes} バイトに足りない: ${head.partLength(0)}`,
    );
  }
  const graphBytes = await head.read(0, HEADER_BYTES, header.graphDescriptorLength);
  if (expect !== undefined) await verifyDocument(graphBytes, expect.graph, "グラフ記述");
  const graph = parseGraphDescriptor(graphBytes);

  let model: ModelDescriptor | undefined;
  if (header.kind === "model") {
    // MUST: when expected values are passed, pass them for both documents (§7 step 1). Passing only
    // graph would parse the model descriptor, which holds the binding table and the block sha256s,
    // unverified (symmetric with the krg-side check).
    if (expect !== undefined && expect.model === undefined) {
      throw new ContainerFormatError("krm なのにモデル記述の期待値が渡されていない");
    }
    const modelBytes = await head.read(
      0,
      HEADER_BYTES + header.graphDescriptorLength,
      header.modelDescriptorLength,
    );
    if (expect?.model !== undefined) await verifyDocument(modelBytes, expect.model, "モデル記述");
    model = parseModelDescriptor(modelBytes);
    validateAgainstGraph(model, graph);
  } else if (expect?.model !== undefined) {
    throw new ContainerFormatError("krg なのにモデル記述の期待値が渡された");
  }
  const graphs = bindGraphs(graph, model);

  // Part 1 of a krg is the const region itself (§3). For a krm, the declared lengths of parts.
  const partLengths =
    model === undefined ? [graph.const.length] : model.parts.map((part) => part.length);
  const source = input.kind === "bytes" ? bytesSource(input.bytes, part0Bytes, partLengths) : head;
  if (source.partCount !== partLengths.length + 1) {
    throw new ContainerFormatError(
      `part が ${source.partCount} 本だが宣言は ${partLengths.length + 1} 本`,
    );
  }
  if (input.kind !== "bytes") {
    // Split form / fetch layer: each part's actual length must match its declaration (for the
    // single form bytesSource checks the total length).
    if (source.partLength(0) !== part0Bytes) {
      throw new ContainerFormatError(
        `part 0 の長さ ${source.partLength(0)} が宣言 ${part0Bytes} と違う`,
      );
    }
    for (const [i, length] of partLengths.entries()) {
      if (source.partLength(i + 1) !== length) {
        throw new ContainerFormatError(
          `part ${i + 1} の長さ ${source.partLength(i + 1)} が宣言 ${length} と違う`,
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
    if (found === undefined) throw new ContainerFormatError(`未宣言の block '${id}'`);
    return found;
  };

  const readBlock = async (id: string): Promise<Uint8Array<ArrayBuffer>> => {
    const found = locate(id);
    const bytes = await source.read(found.part, found.record.offset, found.record.length);
    if (bytes.byteLength !== found.record.length) {
      throw new ContainerFormatError(
        `block '${id}': 取得長 ${bytes.byteLength} が宣言 ${found.record.length} と違う`,
      );
    }
    if (source.verified) return bytes;
    const actual = await sha256Hex(bytes);
    if (actual !== found.record.sha256) {
      throw new ContainerFormatError(
        `block '${id}' の sha256 が宣言と違う（宣言 ${found.record.sha256} / 実物 ${actual}）`,
      );
    }
    return bytes;
  };

  const asset: OpenedContainer["asset"] = (name) => {
    const binding =
      model !== undefined && Object.hasOwn(model.assets, name) ? model.assets[name] : undefined;
    if (binding === undefined) throw new ContainerFormatError(`未宣言の資産 '${name}'`);
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
            `資産 '${name}': 区間 (offset ${offset}, length ${length}) が安全整数でない`,
          );
        }
        if (offset < 0 || length < 0 || offset + length > binding.length) {
          throw new ContainerFormatError(
            `資産 '${name}': ${offset}..${offset + length} が論理長 ${binding.length} をはみ出す`,
          );
        }
        if (source.verified) {
          const bytes = await source.read(found.part, found.record.offset + offset, length);
          if (bytes.byteLength !== length) {
            throw new ContainerFormatError(
              `資産 '${name}': 取得長 ${bytes.byteLength} が要求 ${length} と違う`,
            );
          }
          return bytes;
        }
        verifiedBlock ??= readBlock(binding.block);
        return (await verifiedBlock).subarray(offset, offset + length);
      },
    };
  };

  const extractGraph = async (): Promise<Uint8Array<ArrayBuffer>> => {
    // §9 defines only a krm as the extraction source. A krg has no document holding the hash of the
    // whole const region, so the integrity of the extracted bytes cannot be checked (readBlock checks
    // per-block hashes, but nobody checks the padding outside the blocks).
    if (model === undefined) {
      throw new ContainerFormatError("krg 抽出は krm からだけ（krg 入力は抽出元にならない — §9）");
    }
    const regionLength = graph.const.length;
    const region =
      regionLength === 0
        ? new Uint8Array(new ArrayBuffer(0))
        : await source.read(1, 0, regionLength);
    if (region.byteLength !== regionLength) {
      throw new ContainerFormatError(
        `const 領域: 取得長 ${region.byteLength} が宣言 ${regionLength} と違う`,
      );
    }
    // The integrity of part 1 is held by parts[0] of the model descriptor. Verify it before
    // extracting.
    const actual = await sha256Hex(region);
    if (actual !== model.parts[0].sha256) {
      throw new ContainerFormatError(
        `const 領域の sha256 が宣言と違う（宣言 ${model.parts[0].sha256} / 実物 ${actual}）`,
      );
    }
    return assembleGraphContainer(graphBytes.slice(), region.slice());
  };

  return { header, graph, model, graphs, source, locate, readBlock, asset, extractGraph };
};
