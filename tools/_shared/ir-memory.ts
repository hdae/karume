/**
 * tools 共有: **IR v2 の宣言 → メモリ内容器**（`krm` を書かずに合成グラフを扱う 1 本）。
 *
 * 道具側で合成グラフが要る場面は 2 つある:
 *
 * - **opbench single** — 加重表の 1 行を 1 ノードのグラフに組んで実 GPU で測る。実体は
 *   その場で合成したバイト列なので、容器を書いて読み直す意味が無い（{@link memoryContainer}）
 * - **道具側のテスト** — 束縛の門や候補列挙を合成 IR で見る。初期化子を持たないグラフが
 *   大半で、要るのは合流後の `IrGraph` だけ（{@link memoryGraph}）
 *
 * 宣言は runtime の公開面の型（`IrDeclaration`）で受ける — 呼び手は書いた JSON を
 * `parseIrDeclarationValue` に通してから渡す。宣言の写しをここで別に定義すると、道具ごとに
 * v2 の綴りが割れる（格納 codec は宣言でなく供給側にしか無い — docs/ir-v2.md「格納」）。
 *
 * MUST: `parseIrDeclarationValue` は「非有限数と入れ子の深さは呼び手が検査済み」を前提にした
 * 読み（descriptor の読み手向け — packages/runtime/src/format/ir.ts）なので、通してよいのは
 * **その場で書いたリテラル**（または census 由来の値）だけ。外から来た JSON を載せるなら、
 * 文字列で受けて `parseIrDeclaration` を通すこと。
 */

import type { BoundContainer, IrDeclaration, MemoryTensor } from "../../packages/runtime/mod.ts";
import { openMemoryContainer } from "../../packages/runtime/mod.ts";
import { mergedGraph } from "../../packages/runtime/src/format/container/bind.ts";
import type { IrGraph } from "../../packages/runtime/src/format/ir.ts";

/** 宣言 1 本を載せたメモリ内容器（供給は initializer 名 → 丸ごと 1 本のバイト列）。 */
export const memoryContainer = (
  name: string,
  declaration: IrDeclaration,
  tensors: Readonly<Record<string, MemoryTensor>> = {},
): BoundContainer =>
  openMemoryContainer({
    graphs: { [name]: declaration },
    tensors: { [name]: tensors },
  });

/** 合成グラフ名（1 本きりなので外から選ばせない）。 */
const GRAPH_NAME = "g";

/**
 * 宣言 + 供給を合流した `IrGraph`（= 格納が確定したグラフ）。initializer を持たない宣言なら
 * 供給は空でよい。
 */
export const memoryGraph = (
  declaration: IrDeclaration,
  tensors: Readonly<Record<string, MemoryTensor>> = {},
): IrGraph =>
  mergedGraph(memoryContainer(GRAPH_NAME, declaration, tensors).graphs[GRAPH_NAME], GRAPH_NAME);
