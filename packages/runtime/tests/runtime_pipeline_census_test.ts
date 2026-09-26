// `pipelineCensus`（`SessionDiagnostics.lastRunPipelines` の実体）の純関数としての振る舞い。
// GPU は使わない — 数えるのは `dispatches[].key` だけなので、偽のレシピ列で決定論的に固定できる。
// 実 GPU の発行（`lastRunTiming`）との突合は gpu_pipeline_census_test.ts。

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { pipelineCensus } from "../src/runtime/recipe.ts";

/** キー列から偽のステップ 1 本を組む（census が読むのは `dispatches[].key` だけ）。 */
const step = (...keys: readonly string[]): { readonly dispatches: { readonly key: string }[] } => ({
  dispatches: keys.map((key) => ({ key })),
});

describe("pipelineCensus", () => {
  describe("レシピ列が空のとき", () => {
    it("空の表を返す", () => {
      assertEquals(pipelineCensus([]), []);
    });
  });

  describe("dispatch を持たないステップ（別名化 — reshape / 恒等 expand）だけのとき", () => {
    it("空の表を返す（ステップ数ではなく dispatch 数を数える）", () => {
      assertEquals(pipelineCensus([step(), step()]), []);
    });
  });

  describe("同じキーがステップを跨いで・ステップ内で繰り返されるとき", () => {
    it("キーごとの延べ本数を返す", () => {
      const census = pipelineCensus([
        step("relu:1:f32"),
        step("matmul:a", "matmul:b", "matmul:a"),
        step(),
        step("relu:1:f32"),
      ]);
      assertEquals(census, [
        { key: "matmul:a", dispatchCount: 2 },
        { key: "matmul:b", dispatchCount: 1 },
        { key: "relu:1:f32", dispatchCount: 2 },
      ]);
    });

    it("本数の合計はレシピ列の dispatch 総数と一致する（取りこぼし・二重計上が無い）", () => {
      const recipes = [step("c", "a"), step("b", "a", "c", "c"), step("a")];
      const total = recipes.reduce((sum, recipe) => sum + recipe.dispatches.length, 0);
      assertEquals(
        pipelineCensus(recipes).reduce((sum, entry) => sum + entry.dispatchCount, 0),
        total,
      );
    });
  });

  describe("並び", () => {
    it("出現順ではなくキーの辞書順（コード単位比較 — 大文字が小文字より前）", () => {
      const census = pipelineCensus([step("b"), step("a:2"), step("B"), step("a:10"), step("a")]);
      assertEquals(census.map((entry) => entry.key), ["B", "a", "a:10", "a:2", "b"]);
    });

    it("入力の並びを入れ替えても同じ表になる（計画の並びに依らない決定性）", () => {
      const forward = [step("x", "y"), step("z"), step("y")];
      const backward = [step("y"), step("z"), step("y", "x")];
      assertEquals(pipelineCensus(backward), pipelineCensus(forward));
    });
  });
});
