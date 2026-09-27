import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmUsesVec4 } from "../../packages/runtime/src/kernels/gemm.ts";
import { LINEAR_I8A8_MAX_K } from "../../packages/runtime/src/kernels/linear-i8a8.ts";
import { attentionScoreUsesF16 } from "../../packages/runtime/src/kernels/score-storage.ts";
import { conv2dIgemmMTile, conv2dUsesVec4 } from "../../packages/runtime/src/kernels/conv2d.ts";
import { caseFlops, SWEEP_CASES, SWEEP_OPS } from "./cases.ts";

describe("形状表", () => {
  it("id は一意で、全 op 族が 1 ケース以上ある", () => {
    const ids = SWEEP_CASES.map((sweepCase) => sweepCase.id);
    assertEquals(new Set(ids).size, ids.length);
    for (const op of SWEEP_OPS) {
      assert(SWEEP_CASES.some((sweepCase) => sweepCase.op === op), op);
    }
  });

  it("census 上の本数は正の整数で、演算数は正", () => {
    for (const sweepCase of SWEEP_CASES) {
      assert(Number.isInteger(sweepCase.censusCount) && sweepCase.censusCount > 0, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
    }
  });

  it("本番と同じ vec4 経路に乗る形だけを持つ（掃引の対象は本番の生成物）", () => {
    for (const sweepCase of SWEEP_CASES) {
      switch (sweepCase.op) {
        case "linear":
          assert(gemmUsesVec4(sweepCase.k, sweepCase.n), sweepCase.id);
          break;
        case "i8a8-linear":
          // i8a8 の適格判定（k % 4 == 0）と i32 縮約の門（recipe-builders/linear.ts の述語）
          assertEquals(sweepCase.k % 4, 0, sweepCase.id);
          assert(sweepCase.k <= LINEAR_I8A8_MAX_K, sweepCase.id);
          assertEquals(sweepCase.n % 4, 0, sweepCase.id);
          break;
        case "attention":
          assert(gemmUsesVec4(sweepCase.d, sweepCase.n), sweepCase.id);
          assert(gemmUsesVec4(sweepCase.n, sweepCase.d), sweepCase.id);
          assertEquals(sweepCase.score, "f32", sweepCase.id);
          break;
        case "i8a8-attention":
          // ①は D % 4・③は N % 4 が適格条件、s16 は ①が v4 を取る形だけ
          assertEquals(sweepCase.d % 4, 0, sweepCase.id);
          assertEquals(sweepCase.n % 4, 0, sweepCase.id);
          assertEquals(sweepCase.score, "f16", sweepCase.id);
          assert(attentionScoreUsesF16(sweepCase.d, sweepCase.n), sweepCase.id);
          break;
        case "conv2d":
          assert(conv2dUsesVec4(sweepCase.channelsIn * 9, sweepCase.width, 1), sweepCase.id);
          break;
      }
    }
  });

  it("conv2d は m タイル 32 行と 64 行の両方の本番幾何を 1 行以上持つ", () => {
    const mTiles = new Set(
      SWEEP_CASES.flatMap((sweepCase) =>
        sweepCase.op === "conv2d" ? [conv2dIgemmMTile(sweepCase.channelsOut)] : []
      ),
    );
    assertEquals([...mTiles].sort(), [32, 64]);
  });

  it("attention は ①QK と ③PV が同じ形の対で並ぶ", () => {
    for (const op of ["attention", "i8a8-attention"] as const) {
      const shapes = (stage: "qk" | "pv"): string[] =>
        SWEEP_CASES.flatMap((sweepCase) =>
          sweepCase.op === op && sweepCase.stage === stage
            ? [`${sweepCase.batchHeads}/${sweepCase.m}/${sweepCase.n}/${sweepCase.d}`]
            : []
        );
      assertEquals(shapes("qk"), shapes("pv"), op);
    }
  });
});
