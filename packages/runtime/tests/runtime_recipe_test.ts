/**
 * レシピの **slot 導出**（src/runtime/recipe.ts の `derivePlanSlots`）の構造テスト。GPU を
 * 1 つも触らないので、アダプタ無し環境でも走る。
 *
 * 見ているのは「{@link StepRecipe.outputs} の列から、確保・retain・解放の簿記が
 * {@link RunArena} のサイズクラス LIFO どおりに再生されるか」— 実 GPU の footprint 門
 * （tests/gpu_plan_backing_test.ts が「slot 表の総バイト数 = 非 backed run のプール確保」を
 * 突き合わせる）とは別の角度で、**期待値を手計算で固定する**。
 *
 * ADR 0068 決定 1 の受入条件「単一出力ノードのレシピ表現は列化前と同値」がここの 1 本目。
 */

import { assertEquals } from "@std/assert";
import { derivePlanSlots, type StepOutput, type StepRecipe } from "../src/runtime/recipe.ts";

/** 出力列だけのステップ（dispatch と一時は slot 導出の別軸なので空で足りる）。 */
const step = (
  outputs: readonly StepOutput[],
  releases: readonly string[] = [],
): StepRecipe => ({ outputs, temps: [], dispatches: [], releases, writesState: false });

/** プールから確保する出力（`byteLength` は 4 の倍数 = サイズクラスと同値にしておく）。 */
const alloc = (
  name: string,
  byteLength: number,
  uses: number,
  pinned = false,
): StepOutput => ({ kind: "alloc", byteLength, name, uses, pinned });

/** 別名の出力（入力実体をそのまま出力にする reshape / 恒等 expand — ADR 0011）。 */
const alias = (name: string, source: string, uses: number): StepOutput => ({
  kind: "alias",
  source: { kind: "value", name: source },
  name,
  uses,
  pinned: false,
});

Deno.test("単一出力ステップの slot 割当は alloc / alias / pin / LIFO 再利用で決まる", () => {
  const slots = derivePlanSlots([
    // 消費者 1 本ある中間値 → ステップ末尾では refs が残り、プールへ返らない。
    step([alloc("a", 64, 1)]),
    // 消費者ゼロの中間値（到達不能な値）。定義ぶんの解放でそのままプールへ戻る。
    step([alloc("b", 64, 0)], []),
    // 直前の 2 本がここで返っているので、LIFO の頂点（"b" の slot 1）を掴む。
    step([alloc("c", 64, 0, true)], ["a"]),
    // 別名は slot を増やさない（実体は元の値のもの）。
    step([alias("d", "c", 0)]),
    // 別名元が slot を持たない値（グラフ入力）なら、この出力も slot を持たない。
    step([alias("e", "x", 0)]),
  ]);

  // 新規確保は 2 本だけ（3 本目は "b" の slot を再利用し、別名 2 本は確保を出さない）。
  assertEquals(slots.bytes, [64, 64]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [
    [0],
    [1],
    [1],
    [undefined],
    [undefined],
  ]);
  // pin した slot 1 はグラフ出力 "c" の実体。以後どの値にも配り直されない。
  assertEquals(slots.pinned, new Set([1]));
});

/**
 * 多出力ステップ（ADR 0068 決定 1 — 実装済みの多出力 op はまだ無いので、レシピ表現の側だけを
 * 純関数で固定する）。**出力 slot 昇順**の確保と、slot ごとに独立した `uses` / `pinned` の
 * 簿記が要点。
 */
/**
 * `topk` の実形（ADR 0068 決定 3 の最初の入居者）。実 GPU 側の寿命門
 * （tests/runtime_executor_test.ts の「多出力ノードの slot ごとの uses / pin」）と**同じグラフ**
 * の slot 割当を、GPU に触らない純関数の側から手計算で固定する:
 *
 * ```
 * topk(x[3,8], k=2) → v[3,2] f32（neg が 1 度消費）+ i[3,2] i32（グラフ出力）
 * neg(v) → nv[3,2]（exp が 1 度消費）
 * exp(nv) → e[3,2]（グラフ出力）
 * ```
 *
 * MUST: 2 本の出力は**同じバイト数**（どちらも 3·2 語の 4 バイト要素 = 24）— サイズクラスが
 * 同じなので、slot を取り違えても総バイト数では気づけない。だから **どの slot が再利用されるか**
 * まで期待値に置く（`e` が掴むのは消費済みの `v` の slot 0 で、pin された添字の slot 1 では
 * ない）。
 */
Deno.test("topk の実形（値 + 添字・同一サイズクラス）の slot 割当が手計算どおり", () => {
  // rows·k·4 = 3·2·4（実 GPU 側の門と同じ形から導く）
  const bytes = 3 * 2 * 4;
  const slots = derivePlanSlots([
    // topk: 値（消費者 1 本）+ 添字（グラフ出力 = pin）を出力 slot 昇順で
    step([alloc("v", bytes, 1), alloc("i", bytes, 0, true)]),
    // neg: `v` を消費し、`nv`（消費者 1 本）を定義する。確保の時点でプールは空
    // （`v` は refs 1・`i` は pin）なので新しい slot が生える
    step([alloc("nv", bytes, 1)], ["v"]),
    // exp: `nv` を消費し、`e`（グラフ出力）を定義する。ここで掴むのは**返ってきた `v` の
    // slot 0**（LIFO の頂点）— pin された添字の slot 1 は候補にならない
    step([alloc("e", bytes, 0, true)], ["nv"]),
  ]);

  assertEquals(slots.bytes, [bytes, bytes, bytes]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [[0, 1], [2], [0]]);
  // pin は添字（slot 1）と `e`（slot 0）。実 GPU 側の生存ピーク 72 バイト = 3 slot ぶんと一致する。
  assertEquals(slots.pinned, new Set([1, 0]));
});

Deno.test("多出力ステップは出力 slot 昇順に確保し、slot ごとに独立して retain / pin する", () => {
  const slots = derivePlanSlots([
    // 値 [64B・消費者 1 本] と添字 [32B・グラフ出力] の 2 本を 1 ステップで定義する形。
    step([alloc("v", 64, 1), alloc("i", 32, 0, true)]),
    // "v" を消費するステップ。確保の時点でプールは空（"v" は refs 1・"i" は pin）なので
    // 新しい slot が生える。
    step([alloc("y", 64, 0, true)], ["v"]),
  ]);

  assertEquals(slots.bytes, [64, 32, 64]);
  assertEquals(slots.steps.map((assigned) => assigned.outputs), [[0, 1], [2]]);
  assertEquals(slots.pinned, new Set([1, 2]));
});
