import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmUsesVec4 } from "../../packages/runtime/src/kernels/gemm.ts";
import { GEMM_ROWS_BUCKETS } from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import { LINEAR_I8A8_MAX_K } from "../../packages/runtime/src/kernels/linear-i8a8.ts";
import { attentionScoreUsesF16 } from "../../packages/runtime/src/kernels/score-storage.ts";
import { conv2dIgemmMTile, conv2dUsesVec4 } from "../../packages/runtime/src/kernels/conv2d.ts";
import { caseFlops, PROFILE_GEMM_ROWS_BOUNDS, SWEEP_CASES, SWEEP_OPS } from "./cases.ts";

/** M が入るプロファイルの段の添字（`rows <= maxRows` で最初に当たる段 — runtime の表引きと同じ）。 */
const profileSegmentOf = (rows: number): number =>
  PROFILE_GEMM_ROWS_BOUNDS.findIndex((maxRows) => rows <= maxRows);

/** gemmRows を引く 3 経路（linear / matmul / bmm）のケースの M。bmm は行列 1 枚の M。 */
const gemmRowsCases = SWEEP_CASES.flatMap((sweepCase) =>
  sweepCase.op === "linear" || sweepCase.op === "matmul" || sweepCase.op === "bmm"
    ? [{ op: sweepCase.op, m: sweepCase.m }]
    : []
);

describe("プロファイルの gemmRows の段の境界", () => {
  it("末尾の Infinity を除いた境界は、linear / matmul / bmm の掃引ケースの M（≤ 512）の集合と一致する", () => {
    const sweptRows = [
      ...new Set(gemmRowsCases.map(({ m }) => m).filter((m) => m <= 512)),
    ].sort((a, b) => a - b);
    assertEquals(PROFILE_GEMM_ROWS_BOUNDS.slice(0, -1), sweptRows);
    assertEquals(PROFILE_GEMM_ROWS_BOUNDS.at(-1), Number.POSITIVE_INFINITY);
  });

  it("狭義昇順で、既定の表（GEMM_ROWS_BUCKETS）の境界を全て含む細分になっている", () => {
    for (let index = 1; index < PROFILE_GEMM_ROWS_BOUNDS.length; index++) {
      assert(PROFILE_GEMM_ROWS_BOUNDS[index - 1] < PROFILE_GEMM_ROWS_BOUNDS[index], `${index}`);
    }
    for (const bucket of GEMM_ROWS_BUCKETS) {
      assert(PROFILE_GEMM_ROWS_BOUNDS.includes(bucket.maxRows), `${bucket.maxRows}`);
    }
  });

  it("どの段にも linear / matmul / bmm のケースが 1 本以上ある（> 512 の段の bmm を除く）", () => {
    PROFILE_GEMM_ROWS_BOUNDS.forEach((maxRows, segment) => {
      // > 512 の段の bmm は census に無く、ADR 0116 決定 4 の対象外（足すのは linear しか無かった段だけ）
      const ops = maxRows === Number.POSITIVE_INFINITY
        ? (["linear", "matmul"] as const)
        : (["linear", "matmul", "bmm"] as const);
      for (const op of ops) {
        assert(
          gemmRowsCases.some((sweepCase) =>
            sweepCase.op === op && profileSegmentOf(sweepCase.m) === segment
          ),
          `段 ≤ ${maxRows} に ${op} が無い`,
        );
      }
    });
  });
});

describe("形状表", () => {
  it("id は一意で、全 op 族が 1 ケース以上ある", () => {
    const ids = SWEEP_CASES.map((sweepCase) => sweepCase.id);
    assertEquals(new Set(ids).size, ids.length);
    for (const op of SWEEP_OPS) {
      assert(SWEEP_CASES.some((sweepCase) => sweepCase.op === op), op);
    }
  });

  it("census 由来のケースは census 上の本数が正の整数で、演算数は正", () => {
    for (const sweepCase of SWEEP_CASES) {
      if (sweepCase.op === "matmul") continue;
      assert(Number.isInteger(sweepCase.censusCount) && sweepCase.censusCount > 0, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
    }
  });

  it("matmul は census に無い鏡像で、本数 0・鏡像元は同じ M / N / K の linear のケース", () => {
    const matmuls = SWEEP_CASES.flatMap((sweepCase) =>
      sweepCase.op === "matmul" ? [sweepCase] : []
    );
    assert(matmuls.length > 0);
    for (const sweepCase of matmuls) {
      assertEquals(sweepCase.censusCount, 0, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
      assertStringIncludes(sweepCase.source, sweepCase.mirrorOf, sweepCase.id);
      const mirror = SWEEP_CASES.find((other) => other.id === sweepCase.mirrorOf);
      assert(
        mirror !== undefined && mirror.op === "linear",
        `${sweepCase.id}: 鏡像元が linear に無い`,
      );
      assertEquals(
        [mirror.m, mirror.n, mirror.k],
        [sweepCase.m, sweepCase.n, sweepCase.k],
        sweepCase.id,
      );
    }
  });

  it("matmul はプロファイルの gemmRows の 7 段を 1 本ずつ持ち、既定の 3 バケット（≤ 64 / 65〜512 / > 512）も全て覆う", () => {
    const matmuls = SWEEP_CASES.flatMap((sweepCase) =>
      sweepCase.op === "matmul" ? [sweepCase] : []
    );
    assertEquals(
      matmuls.map((sweepCase) => profileSegmentOf(sweepCase.m)).sort((a, b) => a - b),
      PROFILE_GEMM_ROWS_BOUNDS.map((_, index) => index),
    );
    const defaultBuckets = new Set(
      matmuls.map((sweepCase) =>
        GEMM_ROWS_BUCKETS.findIndex((bucket) => sweepCase.m <= bucket.maxRows)
      ),
    );
    assertEquals([...defaultBuckets].sort(), [0, 1, 2]);
  });

  it("本番と同じ vec4 経路に乗る形だけを持つ（掃引の対象は本番の生成物）", () => {
    for (const sweepCase of SWEEP_CASES) {
      switch (sweepCase.op) {
        case "linear":
        case "matmul":
        case "bmm":
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
