/**
 * 読み込んだ IR グラフを**計画に渡せる形**にする（融合候補列挙の入力側）。
 *
 * 資産の見つけ方と IR の読み出しは tools/_shared/assets.ts が、束縛が正しいかの判定は
 * tools/_shared/scenario.ts が持つ（どちらも census と同じ 1 本）。ここに残るのは
 * 「解けた記号表から具体 shape をどう起こすか」だけ。
 */

import type { IrGraph } from "../../packages/runtime/src/format/ir.ts";
import { resolveShape, type SymbolBindings } from "../../packages/runtime/src/runtime/plan.ts";

/** シンボル束縛から導いた、計画に渡す具体 shape。 */
export type BoundShapes = {
  readonly symbols: SymbolBindings;
  readonly inputShapes: Readonly<Record<string, readonly number[]>>;
  readonly stateShapes: ReadonlyMap<string, readonly number[]>;
};

/**
 * グラフのシンボルを束縛して入力 / state の具体 shape を作る。
 *
 * @param symbols このグラフに効く記号表（`resolveComponentBindings` が解いたもの）。未束縛の
 *   記号が 1 つでもあればそちらが先に打ち切っている — 既定値で計画を進めないため
 *   （`M`（chunk 行数）のような**ヒット数を左右する**記号が知らない値で解かれた表が出る:
 *   gemma4 decode の rope は M=1 で 15・M=32 で 0）。
 */
export const bindGraphSymbols = (
  graph: IrGraph,
  symbols: Readonly<Record<string, number>>,
): BoundShapes => ({
  symbols,
  inputShapes: Object.fromEntries(
    graph.inputs.map((spec) => [spec.name, resolveShape(spec.shape, symbols)]),
  ),
  stateShapes: new Map(
    Object.entries(graph.states).map(([name, slot]) => [name, resolveShape(slot.shape, symbols)]),
  ),
});
