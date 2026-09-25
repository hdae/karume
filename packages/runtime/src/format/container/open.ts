/**
 * コンテナを開く — `BlockSource` → ヘッダ → 2 文書の検証 → parse → 合流 → block 取得 → `krg` 抽出。
 * docs/container-v1.md §7 / §8 / §9 / §11。
 *
 * 順序（「先にグラフだけ取って admission → 重み」）:
 * 1. part 0 の先頭 24 B からヘッダを読む。
 * 2. グラフ記述 / モデル記述を**外側の期待 hash + 長さ**で先に検証してから parse する
 *    （manifest の FileRef や呼び手の pin が持つ値。descriptor 自身は自分の hash を持てない — §7）。
 * 3. 宣言だけで決まる検査（2 文書の整合・束縛の突合・payload 長）を通す。
 * 4. block は要るときに取る。取得元が未検証（{@link BlockSource.verified} が false）なら、取った
 *    block ごとに sha256 を検証する（検証済みの取得元は取得層がファイル全体を検証済み — ADR 0109
 *    決定 7）。
 *
 * `BlockSource` は「単一形の全量バイト」と「part ごとのバイト列」を同じ面に揃える。単一形の part
 * 絶対 offset は書き手と共有する {@link derivePartOffsets} が導く（descriptor に絶対 offset を
 * 書かないので、単一形と分割形で descriptor がバイト同一になる — §8）。
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
 * part → バイト区間の引き出し口。取得層（hub / ローカルファイル / メモリ）はこの面だけを埋める。
 * `read` は**複製したバイト列**を返してよいし、取得元の器の view を返してもよい — 呼び手は返った
 * バイト列を書き換えない。
 */
export type BlockSource = {
  readonly partCount: number;
  /**
   * 取得元が part のバイト列を**検証済み**か（ADR 0109 決定 7 / container-v1 §7）。true なら
   * {@link OpenedContainer.readBlock} は block の sha256 を掛けない（取得層がファイル全体を検証して
   * 記録ハッシュと突き合わせた経路 — cold の 2 重 digest と warm の digest を避ける）。全量バイト /
   * part 列 / ローカルディレクトリのように誰も検証していない取得元は false MUST — 黙って true を
   * 名乗ると改ざんが素通りする。
   */
  readonly verified: boolean;
  partLength(index: number): number;
  read(part: number, offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/** 入力の形。単一形（全量バイト）と分割形（part 列）を同じ入口で受ける。 */
export type ContainerInput =
  | { readonly kind: "bytes"; readonly bytes: Uint8Array<ArrayBuffer> }
  | { readonly kind: "parts"; readonly parts: readonly Uint8Array<ArrayBuffer>[] }
  | { readonly kind: "source"; readonly source: BlockSource };

/**
 * 外側（manifest `karume/5` の `container.descriptor` / 呼び手の pin）が持つ 2 文書の期待値（§7 の①）。
 * `sha256` は小文字 16 進 64 文字（形式はここで検査する — 呼び手にブランド型を要求しない）。
 */
export type DescriptorExpectation = {
  readonly graph: { readonly length: number; readonly sha256: string };
  /** `krm` のときだけ。`krg` では渡してはならない。 */
  readonly model?: { readonly length: number; readonly sha256: string };
};

/**
 * 資産 1 本の読み口（役割の解釈は models — container-v1 §2.2。runtime は名前 → block までしか知らない）。
 */
export type AssetReader = {
  readonly role: string;
  /** payload のバイト数（論理長 — 宣言値。block の詰め物は含まない）。 */
  readonly length: number;
  /**
   * `[offset, offset + length)` を返す。検証済みの取得元では区間だけを取る（block 全体は読まない —
   * 数百 MiB の表から数 KB の行だけを引く消費側のための面）。未検証の取得元では block を 1 度取って
   * sha256 を検証し、この読み口が生きている間は保持してそこから切る（保持する本数は呼び手が読み口の
   * 寿命で決める）。
   *
   * MUST: 返りは書き換えない。検証済みの取得元では取得元の器の view がそのまま出る（hub の scan
   * 経路なら part 全体を握る view）ので、区間より長く持つ呼び手は写す — 握り続けると区間ではなく
   * 器全体が生き残る。未検証の取得元でも返りは保持した block の view（`subarray`）で、その block
   * 自体が取得元の器の view でありうる（hub で位置読みを持たないローカルディレクトリは scan 経路
   * なので、握るのは part 全体の器）— 写す理由は同じ。
   */
  read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/** 開いたコンテナ。 */
export type OpenedContainer = {
  readonly header: ContainerHeader;
  readonly graph: GraphDescriptor;
  /** `krg` では undefined。 */
  readonly model: ModelDescriptor | undefined;
  /** グラフ名 → 合流結果（IR v2 宣言 + initializer ごとの供給計画）。 */
  readonly graphs: Readonly<Record<string, BoundGraph>>;
  readonly source: BlockSource;
  /** block 1 本の所在（part と目次の行）。 */
  locate(
    id: string,
  ): { readonly part: number; readonly record: ConstBlockRecord | DataBlockRecord };
  /**
   * block を 1 本取って返す。取得元が検証済みでなければ（{@link BlockSource.verified} が false）
   * sha256 を検証する（§7 — 未検証の取得元だけが block ごとに一括 digest）。
   */
  readBlock(id: string): Promise<Uint8Array<ArrayBuffer>>;
  /** 資産 1 本の読み口を開く（`krg` と未宣言の名前は fail loudly）。開くだけでは 1 バイトも取らない。 */
  asset(name: string): AssetReader;
  /**
   * `[ヘッダ'][グラフ記述][詰め物][const 領域]` を組み立てて返す（`krg` のバイトコピー抽出 — §9）。
   *
   * NOTE: この口はグラフ記述のバイト列を開いた容器の寿命いっぱい握る。取得元が器の view を返す形
   * （hub の scan 経路）では part 0 の器ごと握るが、part 0 はヘッダと 2 文書ちょうどなので、増える
   * 量はモデル記述の長さ程度に収まる。
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

/** 単一形（全量バイト）の `BlockSource`。part 絶対 offset は宣言 part 長 + 整列規則から導く（§8）。 */
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

/** 分割形（part ごとのバイト列）の `BlockSource`。 */
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
  // 単一形は part 長が descriptor を読むまで分からない。先頭だけ読める暫定の面を作る。
  return {
    partCount: 1,
    verified: false,
    partLength: () => input.bytes.byteLength,
    read: (_part, offset, length) =>
      Promise.resolve(sliceExact(input.bytes, offset, length, "part 0")),
  };
};

/**
 * コンテナを開く。
 *
 * `expect` は外側が持つ期待値（§7）。省略すると 2 文書の完全性は**誰も保証しない**ので、テストと
 * ローカル読み込み以外では渡す。
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

  // krg の part 1 は const 領域そのもの（§3）。krm は parts の宣言長。
  const partLengths = model === undefined
    ? [graph.const.length]
    : model.parts.map((part) => part.length);
  const source = input.kind === "bytes" ? bytesSource(input.bytes, part0Bytes, partLengths) : head;
  if (source.partCount !== partLengths.length + 1) {
    throw new ContainerFormatError(
      `part が ${source.partCount} 本だが宣言は ${partLengths.length + 1} 本`,
    );
  }
  if (input.kind !== "bytes") {
    // 分割形 / 取得層: 各 part の実長が宣言と一致すること（単一形は bytesSource が全長で見る）。
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
    const binding = model !== undefined && Object.hasOwn(model.assets, name)
      ? model.assets[name]
      : undefined;
    if (binding === undefined) throw new ContainerFormatError(`未宣言の資産 '${name}'`);
    const found = locate(binding.block);
    let verifiedBlock: Promise<Uint8Array<ArrayBuffer>> | undefined;
    return {
      role: binding.role,
      length: binding.length,
      read: async (offset, length) => {
        // NaN は全ての比較が偽になって範囲検査を素通りし、小数は subarray が黙って切り捨てる。
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
    const regionLength = graph.const.length;
    const region = regionLength === 0
      ? new Uint8Array(new ArrayBuffer(0))
      : await source.read(1, 0, regionLength);
    if (model !== undefined) {
      // part 1 の完全性はモデル記述の parts[0] が持つ。抜き出す前に検証する。
      const actual = await sha256Hex(region);
      if (actual !== model.parts[0].sha256) {
        throw new ContainerFormatError(
          `const 領域の sha256 が宣言と違う（宣言 ${model.parts[0].sha256} / 実物 ${actual}）`,
        );
      }
    }
    return assembleGraphContainer(graphBytes.slice(), region.slice());
  };

  return { header, graph, model, graphs, source, locate, readBlock, asset, extractGraph };
};
