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
 * - piece の part は添字順に非減少（構築は part 昇順に流し、piece 1 で確保する）
 * - `rowAxis != 0` の initializer は piece 分割不可（scale の行範囲が piece の行範囲に対応しない）
 * - `rowAxis` は codec 台帳の `rowAxes` の中だけ（軸 1 は per-channel i8 のみ — 展開は軸 0 固定）
 * - group の刻み: per-channel codec は `groupSize` = 行長、group codec は 2 冪 ≥ 16 で行長を割る
 * - scale block の長さ = `shape[rowAxis] · (行長 / groupSize) · 4`（+ 詰め物）
 * - i2 経路（`int2-off` / `ternary`）の宣言 shape は正の rank 2 で行長は 16 の倍数
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
  WeightSupply,
} from "./descriptor.ts";
import { ContainerFormatError } from "./header.ts";
import { BLOCK_TAIL_ALIGN } from "./limits.ts";
import { isI2Shape } from "../i2.ts";
import type { IrDeclaration, IrDtype, IrGraph, IrInitializer } from "../ir.ts";

/**
 * block 目次の 1 行のうち**合流層が要る欄**（所在と長さ）。sha256 は要求しない — 取得したバイト列の
 * 完全性は取得層（`open.ts` の `readBlock`）の担当で、合流は「宣言 shape × encoding × 目次」しか
 * 見ないため。容器の目次（`ConstBlockRecord` / `DataBlockRecord`）は構造的にこれを満たし、自前の
 * バイト列を供給するメモリ内容器（`memory.ts`）はこの 3 欄だけを埋める。
 */
export type SupplyRecord = {
  readonly id: string;
  readonly offset: number;
  readonly length: number;
};

/** block id → 所在（part と目次の行）。未宣言の id は fail loudly。 */
export type Locator = (
  id: string,
  by: string,
) => { readonly part: number; readonly record: SupplyRecord };

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

/**
 * `PreparedModel` が要る**供給元の面**（`krm` を開いた `OpenedContainer` も、自前のバイト列を
 * 渡すメモリ内容器（`memory.ts`）の戻りも、構造的にこれを満たす）。置き場が合流層なのは、
 * この面の中身（`graphs`）が合流結果そのものだから — 供給元 1 つ 1 つの実装ファイルに置くと、
 * 他方の供給元しか使わない呼び手までそのファイルへ型依存する。
 *
 * MUST: `readBlock` が返したバイト列を呼び手は**書き換えない**。メモリ内容器は呼び手が渡した
 * 器をそのまま返す（複製しない）ので、書き換えは供給元の実体を壊す。
 */
export type BoundContainer = {
  readonly graphs: Readonly<Record<string, BoundGraph>>;
  readBlock(id: string): Promise<Uint8Array<ArrayBuffer>>;
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

/** block 長と payload 長の関係（§4.1 — 詰め物は 0 以上 4 未満・書き手が焼く）。 */
const assertPadded = (record: SupplyRecord, payload: number, by: string): void => {
  if (record.length < payload || record.length - payload >= BLOCK_TAIL_ALIGN) {
    fail(
      `${by}: block '${record.id}' の長さ ${record.length} が payload ${payload} バイト + 詰め物（${BLOCK_TAIL_ALIGN} 未満）でない`,
    );
  }
};

/** 宣言 shape から payload バイト長（{@link payloadBytes} の素の例外をこの層の型へ言い直す）。 */
const declaredPayloadBytes = (codec: CodecName, numel: number, where: string): number => {
  try {
    return payloadBytes(codec, numel, where);
  } catch (cause) {
    return fail(cause instanceof Error ? cause.message : String(cause));
  }
};

/** 宣言から companion scale のバイト長（{@link scaleBytes} の素の例外をこの層の型へ言い直す）。 */
const declaredScaleBytes = (count: number, where: string): number => {
  try {
    return scaleBytes(count, where);
  } catch (cause) {
    return fail(cause instanceof Error ? cause.message : String(cause));
  }
};

/**
 * piece 分割の 1 行あたりバイト数（payload ÷ 先頭次元）。
 *
 * piece の block 長はこの値から決まるので、**自前の block 目次を合成する供給元**
 * （`memory.ts`）も同じ 1 本を通す — 式が 2 つあると、合成した長さと合流層が期待する長さが
 * 静かに割れる（割れても `found.record.length === pieceBytes` の突合は両方が同じ誤りを
 * 持つぶん恒真になる）。
 */
export const pieceRowBytes = (
  codec: CodecName,
  shape: readonly number[],
  where: string,
): number => rowBytesOfPayload(declaredPayloadBytes(codec, numelOf(shape), where), shape, where);

/** {@link pieceRowBytes} の本体（payload を既に持っている呼び手が再計算を避けるための口）。 */
const rowBytesOfPayload = (payload: number, shape: readonly number[], where: string): number => {
  const rows = shape.length === 0 ? 1 : shape[0];
  if (shape.length === 0 || payload % rows !== 0) {
    fail(
      `${where}: payload ${payload} バイトが先頭次元 ${rows} 行で割り切れないので piece 分割できない`,
    );
  }
  return payload / rows;
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
  const payload = declaredPayloadBytes(encoding.codec, numel, where);
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
    const rowBytes = rowBytesOfPayload(payload, shape, where);
    const last = supply.pieces[supply.pieces.length - 1];
    if (last.rows[1] !== rows) {
      fail(
        `${where}: piece 列の末尾 ${
          last.rows[1]
        } 行が宣言 shape の先頭次元 ${rows} 行に届かない / 超える（規則④）`,
      );
    }
    // 規則④: 行範囲は 0 から隙間なく続く。穴・重なり・空区間はどれも「どのバイトがその行を
    // 埋めるか」が転送順で決まる沈黙誤値になる。容器経由では descriptor の parse が同じ形を
    // 先に落とすが、**束縛規則の所有者はこの層**なので供給元に依らずここでも見る
    // （自前のバイト列を供給するメモリ内容器は descriptor を通らない）。
    let cursor = 0;
    for (const [i, piece] of supply.pieces.entries()) {
      const by = `${where} piece[${i}]`;
      // 規則④: 行番号は非負の安全整数。小数の行境界は被覆検査（空区間・連続性・末尾）を
      // どれも素通りしたうえで、`rowOffset · (全体長 / 行数)` を 4 の倍数でないバイト位置に
      // し、GPU の writeBuffer validation まで失敗が遅れる。容器経由では descriptor の
      // `requireIndex` が同じ形を先に落とす。
      for (const [edge, value] of piece.rows.entries()) {
        if (!Number.isSafeInteger(value) || value < 0) {
          fail(
            `${by}: 行範囲の ${
              edge === 0 ? "開始" : "終端"
            } ${value} が非負の安全整数でない（規則④）`,
          );
        }
      }
      if (piece.rows[1] <= piece.rows[0]) {
        fail(`${by}: 行範囲 [${piece.rows[0]}, ${piece.rows[1]}) が空区間（規則④）`);
      }
      if (piece.rows[0] !== cursor) {
        fail(
          `${by}: 行範囲 [${piece.rows[0]}, ${piece.rows[1]}) が行 ${cursor} から続かない（規則④）`,
        );
      }
      cursor = piece.rows[1];
      const found = locate(piece.block, by);
      // 規則④: piece の part は添字順に非減少。構築（`containerBatches` → `buildSessionState`）は
      // block を part 昇順に流し、piece 1 でバッファを確保して scale を持ち越す（規則③で scale は
      // piece 1 と同じ part）ので、逆順の容器は内部簿記の破れの文言で落ちる。受理集合の門を
      // 合流相に置き、供給元に依らずここで落とす。
      const previous = blocks.at(-1);
      if (previous !== undefined && found.part < previous.part) {
        fail(
          `${by}: part ${found.part} が直前の piece の part ${previous.part} より前にある（piece の part は添字順に非減少 MUST — 規則④）`,
        );
      }
      const pieceBytes = (piece.rows[1] - piece.rows[0]) * rowBytes;
      if (i < supply.pieces.length - 1) {
        // 中間 piece に詰め物は掛けられない（次の piece の先頭を潰す）— 長さは元から 4 の倍数 MUST。
        // 容器では書き手と目次の parse が先に落とすが、**束縛規則の所有者はこの層**なので
        // 供給元に依らずここでも見る（見ないと、4 の倍数でない行長の codec を piece 分割した
        // 形が GPU の writeBuffer validation まで落ちず、転送層の文言で出る）。
        if (pieceBytes % BLOCK_TAIL_ALIGN !== 0) {
          fail(
            `${by}: 中間 piece の行範囲のバイト数 ${pieceBytes} が ${BLOCK_TAIL_ALIGN} の倍数でない（詰め物不可 — 規則④）`,
          );
        }
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
  if (!entry.rowAxes.some((axis) => axis === rowAxis)) {
    fail(
      `${where}: codec '${encoding.codec}' の rowAxis は ${
        entry.rowAxes.join(" / ")
      } だけ（宣言は ${rowAxis} — container-v1 §6.3）`,
    );
  }
  if (shape.length === 0 || (rowAxis === 1 && shape.length < 2)) {
    fail(`${where}: rowAxis ${rowAxis} に対して宣言 shape [${shape.join(",")}] の rank が足りない`);
  }
  const rowCount = shape[rowAxis];
  const rowLength = quantizedRowLength(shape, rowAxis);
  if (entry.grouping === "channel" && groupSize !== perChannelGroupSize(rowLength)) {
    fail(
      `${where}: codec '${encoding.codec}' は per-channel なので groupSize は行長 ${
        perChannelGroupSize(rowLength)
      } に等しい MUST（宣言は ${groupSize}）`,
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
  const scalePayload = declaredScaleBytes(
    numelOf(groupScaleShape(shape, rowAxis, groupSize)),
    `${where} scale`,
  );
  assertPadded(scaleFound.record, scalePayload, `${where} scale`);
  const scale: SupplyBlock = {
    id: scaleFound.record.id,
    part: scaleFound.part,
    offset: scaleFound.record.offset,
    length: scaleFound.record.length,
    rows: [0, rowCount],
    payloadBytes: scalePayload,
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
 * 合流結果 → ランタイムが実行するグラフ（initializer ごとの格納が確定した {@link IrGraph}）。
 * 束縛表の `encoding` から `codec` / `groupSize` / `rowAxis` を写し、shared 宣言はそのまま渡す。
 * `krg`（重みの供給が無い）でも const 供給と shared だけなら組める — 重みが要る initializer は
 * 供給が無いので `IrGraph` にできない（fail loudly）。
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
        `graph '${graphName}' initializer '${name}': 重みの供給が無い（krg だけでは Session を組めない）`,
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
 * `ternary` の宣言の追加条件（container-v1 §6.3）: payload の全 2 bit コードが `{1, 2, 3}`
 * （コード 0 = q − 2 は三値の値域外）。「三値である」という主張の検査可能な中身。
 */
export const assertTernaryCodes = (payload: Uint8Array<ArrayBuffer>, where: string): void => {
  for (let i = 0; i < payload.byteLength; i += 1) {
    const byte = payload[i];
    if ((byte & 3) === 0 || (byte & 12) === 0 || (byte & 48) === 0 || (byte & 192) === 0) {
      fail(`${where}: ternary の payload にコード 0（三値の値域外）がある（バイト ${i}）`);
    }
  }
};

/**
 * 宣言と供給を合流する本体 — **2 文書の形に依存しない**（容器の目次でも、自前のバイト列を
 * 供給するメモリ内容器でも同じ 1 本を通る）。入口はグラフの宣言・const 束縛・束縛表・block の
 * 引き当て口の 4 つだけで、ここに置く規則がモジュール doc の一覧そのものである。
 *
 * `binding` が `undefined` のときは束縛表を持たない形（`krg`）— const 供給と shared 以外の
 * initializer は「重みが要る」宣言として供給計画を持たないまま残る（Session は組めない）。
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

  const out: Record<string, BoundGraph> = {};
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
      const graphBinding = binding !== undefined && Object.hasOwn(binding, graphName)
        ? binding[graphName]
        : undefined;
      const supply = graphBinding !== undefined && Object.hasOwn(graphBinding, name)
        ? graphBinding[name]
        : undefined;
      if (supply === undefined) {
        // krg（束縛表を持たない）で const 以外の initializer は「重みが要る」宣言 — 供給計画は無い。
        if (binding === undefined) continue;
        return fail(`${where}: 束縛表に供給が無い`);
      }
      supplies.set(name, planSupply(where, value.dtype, shape, supply, locate, "model"));
    }
    out[graphName] = { declaration, supplies };
  }
  return out;
};

/**
 * 全グラフを合流する（2 文書の面）。構造検査（`descriptor.ts`）は済んでいる前提で、ここは
 * 2 文書から const 束縛・束縛表・block の引き当て口を組んで {@link bindDeclarations} へ渡す
 * 薄い外皮である。
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
  return bindDeclarations({
    graphs: graph.graphs,
    constants: graph.const.constants,
    binding: model?.binding,
    locate,
  });
};
