/**
 * テスト専用の容器の書き手（単一形 / 分割形 / `krg`）— docs/container-v1.md §3 / §4 / §8 / §9。
 *
 * 製品の書き手は Python（exporter の `karume.container`）だけで、これは**読み手のテストが合成資産を
 * 作るための補助**である。守る規則は製品の書き手と同じ:
 *
 * - block は part をまたがない・1 block ≤ 上限（既定 32 MiB — テストでは下げられる）。
 * - block の先頭は 64 B 整列、格納長は 4 の倍数（**末尾の詰め物を書き手が焼く**）。中間 piece には
 *   詰め物を掛けない（次の piece の先頭を潰す）ので、行の刻みを `4 / gcd(rowBytes, 4)` に丸めて切る。
 * - companion scale は piece 1（丸ごとなら唯一の実体）と同じ part。
 * - descriptor のバイト列は**単一形と分割形で同一**（part の絶対 offset を書かない）。
 * - 区間読みを要する資産（`dedicatedPart`）は専用 part に単独で置く。
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
  /** 量子化 codec のみ。省略時 0。 */
  readonly rowAxis?: 0 | 1;
  /** 量子化 codec のみ。per-channel なら行長。 */
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
  /** グラフ所有の定数（const 領域 = part 1 行き。piece 分割はしない）。 */
  readonly consts: readonly TensorInput[];
  readonly weights: readonly TensorInput[];
  readonly assets: readonly AssetInput[];
  readonly provenance: Provenance;
};

export type WriteOptions = {
  /** part 長。既定 256 MiB（テストでは小さくしてよい）。 */
  readonly partBytes?: number;
  /** block 上限。既定 32 MiB（テストのために下げられる・上げられない）。 */
  readonly blockBytes?: number;
};

export type WrittenContainer = {
  readonly graph: GraphDescriptor;
  readonly model: ModelDescriptor;
  readonly graphDescriptorBytes: Uint8Array<ArrayBuffer>;
  readonly modelDescriptorBytes: Uint8Array<ArrayBuffer>;
  /** 分割形。index 0 が `[ヘッダ][グラフ記述][モデル記述]`。 */
  readonly parts: readonly Uint8Array<ArrayBuffer>[];
  /** 単一形。 */
  readonly single: Uint8Array<ArrayBuffer>;
  /** part 1 までを抜いた `krg`。 */
  readonly graphContainer: Uint8Array<ArrayBuffer>;
};

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

const resolveEncoding = (input: EncodingInput, scaleBlock: string | undefined): Encoding => {
  const entry = codecEntry(input.codec);
  if (entry.scale === "forbidden") {
    if (input.scale !== undefined || input.groupSize !== undefined || input.rowAxis !== undefined) {
      throw new ContainerFormatError(
        `codec '${input.codec}' は scale / groupSize / rowAxis を持てない`,
      );
    }
    return { codec: input.codec, packing: entry.packing };
  }
  if (input.scale === undefined || input.groupSize === undefined || scaleBlock === undefined) {
    throw new ContainerFormatError(`codec '${input.codec}' は scale と groupSize が要る`);
  }
  return {
    codec: input.codec,
    packing: entry.packing,
    rowAxis: input.rowAxis ?? 0,
    groupSize: input.groupSize,
    scale: { block: scaleBlock, dtype: input.scale.dtype },
  };
};

/** 末尾のゼロ詰めを焼いた格納バイト列（`tailPad = false` なら長さ検査だけ）。 */
const storedPayload = (
  bytes: Uint8Array<ArrayBuffer>,
  tailPad: boolean,
  path: string,
): Uint8Array<ArrayBuffer> => {
  if (bytes.byteLength === 0) throw new ContainerFormatError(`${path}: 長さ 0 の block は作らない`);
  if (!tailPad) {
    if (bytes.byteLength % BLOCK_TAIL_ALIGN !== 0) {
      throw new ContainerFormatError(
        `${path}: 中間 piece の長さ ${bytes.byteLength} が ${BLOCK_TAIL_ALIGN} の倍数でない`,
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

/** 1 part ぶんの詰め込み器。offset は part 先頭からの相対。 */
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
    throw new ContainerFormatError(`${path}: グラフ '${tensor.graph}' が無い`);
  }
  const value = graph.values[tensor.initializer];
  if (value === undefined) {
    throw new ContainerFormatError(`${path}: initializer '${tensor.initializer}' の宣言が無い`);
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
      // const は piece 分割の機構を持たない（§3 — 束縛表が krg に無いので piece 列を表せない）。
      throw new ContainerFormatError(
        `${path}: const は 1 block に収める必要があるが ${payload.byteLength} バイト（上限 ${blockBytes}）`,
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

/** テンソルを block 上限以下の piece へ行境界で切る。切る必要が無ければ 1 本返す。 */
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
  if (rows <= 1) throw new ContainerFormatError(`${path}: 1 行で block 上限を超えるので切れない`);
  if (total % rows !== 0) {
    throw new ContainerFormatError(`${path}: ${total} バイトが ${rows} 行で割り切れない`);
  }
  const rowBytes = total / rows;
  const step = BLOCK_TAIL_ALIGN / gcd(rowBytes, BLOCK_TAIL_ALIGN);
  const capacity = Math.floor(blockBytes / rowBytes);
  const rowsPerPiece = capacity - (capacity % step);
  if (rowsPerPiece < step || rowsPerPiece === 0) {
    throw new ContainerFormatError(
      `${path}: 1 行 ${rowBytes} バイトでは block 上限 ${blockBytes} に 4 バイト整列した行束を収められない`,
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
  /** 「同じ part に置く」必要がある一群をまとめて置く。入らなければ新しい part を開く。 */
  const placeGroup = (
    items: readonly { readonly id: string; readonly payload: Uint8Array<ArrayBuffer> }[],
    path: string,
  ): void => {
    const lengths = items.map((item) => item.payload.byteLength);
    if (new PartBuilder().probe(lengths) > partBytes) {
      throw new ContainerFormatError(
        `${path}: 同一 part に置く必要がある一群が part 長 ${partBytes} を超える`,
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
    const pieceIds = cuts.map(() => nextId("w", "weight"));
    const head = [{
      id: pieceIds[0],
      payload: storedPayload(cuts[0].bytes, cuts[0].last, `${path} piece 0`),
    }];
    let scaleId: string | undefined;
    if (tensor.encoding.scale !== undefined) {
      scaleId = nextId("s", "scale");
      head.push({
        id: scaleId,
        payload: storedPayload(tensor.encoding.scale.bytes, true, `${path} scale`),
      });
    }
    // 規則③: piece 1 と scale は同一 part。残りの piece は独立に置ける。
    placeGroup(head, path);
    for (const [i, cut] of cuts.entries()) {
      if (i === 0) continue;
      placeGroup([{
        id: pieceIds[i],
        payload: storedPayload(cut.bytes, cut.last, `${path} piece ${i}`),
      }], `${path} piece ${i}`);
    }
    const encoding = resolveEncoding(tensor.encoding, scaleId);
    const supply: WeightSupply = cuts.length === 1
      ? { block: pieceIds[0], encoding }
      : { pieces: cuts.map((cut, i) => ({ block: pieceIds[i], rows: cut.rows })), encoding };
    binding[tensor.graph] = { ...(binding[tensor.graph] ?? {}), [tensor.initializer]: supply };
  }

  for (const asset of input.assets) {
    const path = `asset ${asset.name}`;
    const payload = storedPayload(asset.bytes, true, path);
    if (payload.byteLength > blockBytes) {
      throw new ContainerFormatError(
        `${path}: ${payload.byteLength} バイトが block 上限 ${blockBytes} を超える`,
      );
    }
    const id = nextId("a", "asset");
    if (asset.dedicatedPart === true) {
      // 区間読みが offset 比例で走査になる取得元のために、単独 part へ隔離する（§4.2）。
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

/** `krg`（グラフ容器）だけを書く。`krm` の part 0..1 と**同じ関数**でグラフ側を作る。 */
export const writeGraphContainer = async (
  input: ModelInput,
  options: WriteOptions = {},
): Promise<
  { readonly bytes: Uint8Array<ArrayBuffer>; readonly descriptorBytes: Uint8Array<ArrayBuffer> }
> => {
  const region = await buildConstRegion(input, options.blockBytes ?? BLOCK_MAX_BYTES);
  const descriptorBytes = serializeGraphDescriptor(buildGraphDescriptor(input, region));
  return { bytes: assembleGraphContainer(descriptorBytes, region.bytes), descriptorBytes };
};

/** `krm`（モデル容器）を単一形・分割形の両方で書く。 */
export const writeModelContainer = async (
  input: ModelInput,
  options: WriteOptions = {},
): Promise<WrittenContainer> => {
  const partBytes = options.partBytes ?? DEFAULT_PART_BYTES;
  const blockBytes = options.blockBytes ?? BLOCK_MAX_BYTES;
  if (blockBytes > BLOCK_MAX_BYTES) {
    throw new ContainerFormatError(`block 上限 ${blockBytes} が仕様の ${BLOCK_MAX_BYTES} を超える`);
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
  const partRecords: PartRecord[] = [{
    index: 1,
    length: region.bytes.byteLength,
    sha256: await sha256Hex(region.bytes),
  }];
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
  const layout = derivePartOffsets(part0.byteLength, parts.slice(1).map((part) => part.byteLength));
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
