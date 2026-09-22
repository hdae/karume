/**
 * 合流層 — グラフ（IR v2 の宣言）と束縛表を**重み取得前に**合流し、initializer ごとの供給計画を
 * 決める。docs/container-v1.md §5 / §6 / §13.3、docs/ir-v2.md「格納」。
 *
 * ここに置く規則は「宣言 shape × encoding × block 目次」で決まるもの全部である（**置き場は 1 箇所** —
 * 二重実装しない）:
 *
 * - 意味論 dtype と codec の組（f32 の符号化 / i32 は生の int32 — 交差は fail loudly）
 * - payload 長 = 宣言 shape と packing から決まる値。block 長との差は詰め物（0 以上 4 未満）だけ
 * - piece 列は先頭次元の行範囲を隙間なく被覆し、末尾 = shape[0]。中間 piece に詰め物は無い
 * - `rowAxis != 0` の initializer は piece 分割不可（scale の行範囲が piece の行範囲に対応しない）
 * - group の刻み: per-channel codec は `groupSize` = 行長、group codec は 2 冪 ≥ 16 で行長を割る
 * - scale block の長さ = `shape[rowAxis] · (行長 / groupSize) · 4`（+ 詰め物）
 * - i2 経路（`int2-off` / `ternary`）の宣言 shape は正の rank 2 で行長は 16 の倍数
 */

import { type CodecEntry, codecEntry, MIN_GROUP_SIZE, payloadBytes } from "./codecs.ts";
import type {
  ConstBlockRecord,
  DataBlockRecord,
  Encoding,
  GraphDescriptor,
  ModelDescriptor,
  WeightSupply,
} from "./descriptor.ts";
import { ContainerFormatError } from "./header.ts";
import { BLOCK_TAIL_ALIGN } from "./limits.ts";
import { isI2Shape } from "../i2.ts";
import type { IrDeclaration, IrDtype } from "../ir.ts";

/** 実体 1 本ぶんの block（piece 列なら 1 piece）。 */
export type SupplyBlock = {
  readonly id: string;
  readonly part: number;
  readonly offset: number;
  readonly length: number;
  /** この block が運ぶ先頭次元の行範囲（丸ごと 1 本なら `[0, shape[0]]`・rank 0 は `[0, 1]`）。 */
  readonly rows: readonly [number, number];
  /** この block の payload バイト長（block 長から詰め物を除いたもの）。 */
  readonly payloadBytes: number;
};

/** initializer 1 本の供給計画（実体をどの block から取るか）。 */
export type InitializerSupply = {
  readonly encoding: Encoding;
  /** 実体の block 列（丸ごとなら 1 本）。 */
  readonly blocks: readonly SupplyBlock[];
  /** companion scale の block（量子化 codec のみ）。 */
  readonly scale?: SupplyBlock;
  readonly zeroPoint?: SupplyBlock;
  /** 供給元。const 領域（part 1・グラフの所有）か重み側（モデル記述）か。 */
  readonly origin: "const" | "model";
};

/** グラフ 1 本の合流結果。 */
export type BoundGraph = {
  readonly declaration: IrDeclaration;
  /** shared でない initializer 全部の供給計画（宣言順）。 */
  readonly supplies: ReadonlyMap<string, InitializerSupply>;
};

// NOTE: 関数宣言にするのは、`never` 戻りの呼び出しを TS が制御フローの打ち切りとして扱う条件が
// 「宣言型が明示された識別子」であるため（const の arrow では narrowing が効かない）。
function fail(message: string): never {
  throw new ContainerFormatError(message);
}

/** 意味論 dtype → 許される codec の展開経路（docs/ir-v2.md「格納」の合流層の規則）。 */
const allowedLayouts = (dtype: IrDtype): readonly CodecEntry["layout"][] =>
  dtype === "f32" ? ["f32", "f16", "bf16", "i8", "i4", "i2"] : dtype === "i32" ? ["i32"] : [];

const isPowerOfTwo = (value: number): boolean => 2 ** Math.round(Math.log2(value)) === value;

const numelOf = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

type Locator = (
  id: string,
  by: string,
) => { readonly part: number; readonly record: ConstBlockRecord | DataBlockRecord };

/** block 長と payload 長の関係（§4.1 — 詰め物は 0 以上 4 未満・書き手が焼く）。 */
const assertPadded = (record: ConstBlockRecord, payload: number, by: string): void => {
  if (record.length < payload || record.length - payload >= BLOCK_TAIL_ALIGN) {
    fail(
      `${by}: block '${record.id}' の長さ ${record.length} が payload ${payload} バイト + 詰め物（${BLOCK_TAIL_ALIGN} 未満）でない`,
    );
  }
};

const planSupply = (
  where: string,
  dtype: IrDtype,
  shape: readonly number[],
  supply:
    & { readonly encoding: Encoding }
    & (
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
    fail(`${where}: 意味論 dtype '${dtype}' に codec '${encoding.codec}' は組めない`);
  }
  const numel = numelOf(shape);
  if (entry.layout === "i2" && !isI2Shape(shape)) {
    fail(
      `${where}: codec '${encoding.codec}' は正の rank 2・行長 16 の倍数の宣言 shape が要る（[${
        shape.join(",")
      }]）`,
    );
  }
  let payload: number;
  try {
    payload = payloadBytes(encoding.codec, numel, where);
  } catch (cause) {
    fail(cause instanceof Error ? cause.message : String(cause));
  }
  const rows = shape.length === 0 ? 1 : shape[0];

  // 実体の block 列。
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
      fail(`${where}: rowAxis ${encoding.rowAxis} の initializer は piece 分割できない（規則④）`);
    }
    if (shape.length === 0 || payload % rows !== 0) {
      fail(
        `${where}: payload ${payload} バイトが先頭次元 ${rows} 行で割り切れないので piece 分割できない`,
      );
    }
    const rowBytes = payload / rows;
    const last = supply.pieces[supply.pieces.length - 1];
    if (last.rows[1] !== rows) {
      fail(
        `${where}: piece 列の末尾 ${
          last.rows[1]
        } 行が宣言 shape の先頭次元 ${rows} 行に届かない / 超える（規則④）`,
      );
    }
    for (const [i, piece] of supply.pieces.entries()) {
      const by = `${where} piece[${i}]`;
      const found = locate(piece.block, by);
      const pieceBytes = (piece.rows[1] - piece.rows[0]) * rowBytes;
      if (i < supply.pieces.length - 1) {
        // 中間 piece に詰め物は掛けられない（次の piece の先頭を潰す）— 長さは元から 4 の倍数 MUST。
        if (found.record.length !== pieceBytes) {
          fail(
            `${by}: 中間 piece の block 長 ${found.record.length} が行範囲のバイト数 ${pieceBytes} と違う（詰め物不可）`,
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

  // 量子化: rowAxis / groupSize / scale は parseEncoding が存在を保証済み。
  const rowAxis = encoding.rowAxis ?? 0;
  const groupSize = encoding.groupSize ?? 0;
  if (shape.length === 0 || (rowAxis === 1 && shape.length < 2)) {
    fail(`${where}: rowAxis ${rowAxis} に対して宣言 shape [${shape.join(",")}] の rank が足りない`);
  }
  const rowCount = shape[rowAxis];
  const rowLength = rowCount === 0 ? 0 : numel / rowCount;
  if (entry.grouping === "channel" && groupSize !== rowLength) {
    fail(
      `${where}: codec '${encoding.codec}' は per-channel なので groupSize は行長 ${rowLength} に等しい MUST（宣言は ${groupSize}）`,
    );
  }
  if (entry.grouping === "group") {
    if (!isPowerOfTwo(groupSize) || groupSize < MIN_GROUP_SIZE) {
      fail(
        `${where}: groupSize ${groupSize} が 2 冪かつ ${MIN_GROUP_SIZE} 以上でない（ADR 0069 決定 2）`,
      );
    }
    if (rowLength % groupSize !== 0) {
      fail(
        `${where}: 行長 ${rowLength}（= numel / shape[${rowAxis}]）が groupSize ${groupSize} で割り切れない（ADR 0069 決定 2）`,
      );
    }
  }
  const scaleRef = encoding.scale;
  if (scaleRef === undefined) fail(`${where}: codec '${encoding.codec}' は scale 必須`);
  const scaleFound = locate(scaleRef.block, `${where} scale`);
  const scaleBytes = rowCount * (rowLength / groupSize) * 4;
  assertPadded(scaleFound.record, scaleBytes, `${where} scale`);
  const scale: SupplyBlock = {
    id: scaleFound.record.id,
    part: scaleFound.part,
    offset: scaleFound.record.offset,
    length: scaleFound.record.length,
    rows: [0, rowCount],
    payloadBytes: scaleBytes,
  };
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
 * 全グラフを合流する。2 文書の構造検査（`descriptor.ts`）は済んでいる前提で、ここは宣言 shape を
 * 要する規則だけを掛ける。
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
    return fail(`${by}: 未宣言の block '${id}'`);
  };
  const constSupply = new Map<string, GraphDescriptor["const"]["constants"][number]>();
  for (const entry of graph.const.constants) {
    constSupply.set(`${entry.graph}/${entry.initializer}`, entry);
  }

  const out: Record<string, BoundGraph> = {};
  for (const [graphName, declaration] of Object.entries(graph.graphs)) {
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
      const supply = model?.binding[graphName]?.[name];
      if (supply === undefined) {
        // krg（束縛表を持たない）で const 以外の initializer は「重みが要る」宣言 — 供給計画は無い。
        if (model === undefined) continue;
        return fail(`${where}: 束縛表に供給が無い`);
      }
      supplies.set(name, planSupply(where, value.dtype, shape, supply, locate, "model"));
    }
    out[graphName] = { declaration, supplies };
  }
  return out;
};
