// The behavior of `pipelineCensus` (the implementation of `SessionDiagnostics.lastRunPipelines`) as
// a pure function. No GPU — it counts only `dispatches[].key`, so a fake recipe sequence pins it
// deterministically. The cross-check against real GPU issuance (`lastRunTiming`) is in
// gpu_pipeline_census_test.ts.

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { pipelineCensus } from "../src/runtime/recipe.ts";

/** Builds one fake step from a key list (the census reads only `dispatches[].key`). */
const step = (...keys: readonly string[]): { readonly dispatches: { readonly key: string }[] } => ({
  dispatches: keys.map((key) => ({ key })),
});

describe("pipelineCensus", () => {
  describe("when the recipe sequence is empty", () => {
    it("returns an empty table", () => {
      assertEquals(pipelineCensus([]), []);
    });
  });

  describe("when there are only steps without dispatches (aliasing — reshape / identity expand)", () => {
    it("returns an empty table (it counts dispatches, not steps)", () => {
      assertEquals(pipelineCensus([step(), step()]), []);
    });
  });

  describe("when the same key repeats across steps and within a step", () => {
    it("returns the total count per key", () => {
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

    it("the counts sum to the recipe sequence's total dispatch count (nothing missed, nothing double counted)", () => {
      const recipes = [step("c", "a"), step("b", "a", "c", "c"), step("a")];
      const total = recipes.reduce((sum, recipe) => sum + recipe.dispatches.length, 0);
      assertEquals(
        pipelineCensus(recipes).reduce((sum, entry) => sum + entry.dispatchCount, 0),
        total,
      );
    });
  });

  describe("ordering", () => {
    it("keys are in lexicographic order, not order of appearance (code-unit comparison — uppercase before lowercase)", () => {
      const census = pipelineCensus([step("b"), step("a:2"), step("B"), step("a:10"), step("a")]);
      assertEquals(
        census.map((entry) => entry.key),
        ["B", "a", "a:10", "a:2", "b"],
      );
    });

    it("reordering the input gives the same table (deterministic regardless of plan order)", () => {
      const forward = [step("x", "y"), step("z"), step("y")];
      const backward = [step("y"), step("z"), step("y", "x")];
      assertEquals(pipelineCensus(backward), pipelineCensus(forward));
    });
  });
});
