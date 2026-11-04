import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { GEMM_ROWS_BUCKETS } from "../src/kernels/gemm-geometry.ts";
import { LINEAR_I8A8_MAX_K } from "../src/kernels/linear-i8a8.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { conv2dIgemmMTile, conv2dUsesVec4 } from "../src/kernels/conv2d.ts";
import { conv3dI8a8EligibleShape, conv3dI8a8UsesVec4 } from "../src/kernels/conv3d-i8a8.ts";
import {
  caseFlops,
  PROFILE_GEMM_ROWS_BOUNDS,
  SWEEP_CASES,
  SWEEP_OPS,
  sweepCaseEntries,
} from "../src/tune/cases.ts";

/**
 * The index of the profile segment that M falls into (the first segment with `rows <= maxRows` —
 * the same lookup as the runtime's).
 */
const profileSegmentOf = (rows: number): number =>
  PROFILE_GEMM_ROWS_BOUNDS.findIndex((maxRows) => rows <= maxRows);

/**
 * The M of the cases of the three paths that look up gemmRows (linear / matmul / bmm). For bmm it
 * is the M of one matrix.
 */
const gemmRowsCases = SWEEP_CASES.flatMap((sweepCase) =>
  sweepCase.op === "linear" || sweepCase.op === "matmul" || sweepCase.op === "bmm"
    ? [{ op: sweepCase.op, m: sweepCase.m }]
    : [],
);

describe("the boundaries of the profile's gemmRows segments", () => {
  it("the boundaries without the trailing Infinity equal the set of M (≤ 512) of the linear / matmul / bmm sweep cases", () => {
    const sweptRows = [...new Set(gemmRowsCases.map(({ m }) => m).filter((m) => m <= 512))].sort(
      (a, b) => a - b,
    );
    assertEquals(PROFILE_GEMM_ROWS_BOUNDS.slice(0, -1), sweptRows);
    assertEquals(PROFILE_GEMM_ROWS_BOUNDS.at(-1), Number.POSITIVE_INFINITY);
  });

  it("are strictly ascending and refine the default table (GEMM_ROWS_BUCKETS), containing all its boundaries", () => {
    for (let index = 1; index < PROFILE_GEMM_ROWS_BOUNDS.length; index++) {
      assert(PROFILE_GEMM_ROWS_BOUNDS[index - 1] < PROFILE_GEMM_ROWS_BOUNDS[index], `${index}`);
    }
    for (const bucket of GEMM_ROWS_BUCKETS) {
      assert(PROFILE_GEMM_ROWS_BOUNDS.includes(bucket.maxRows), `${bucket.maxRows}`);
    }
  });

  it("every segment has at least one linear / matmul / bmm case (except bmm in the > 512 segment)", () => {
    PROFILE_GEMM_ROWS_BOUNDS.forEach((maxRows, segment) => {
      // bmm in the > 512 segment is not in the census and is outside ADR 0116 decision 4 (cases are
      // added only to segments that had nothing but linear)
      const ops =
        maxRows === Number.POSITIVE_INFINITY
          ? (["linear", "matmul"] as const)
          : (["linear", "matmul", "bmm"] as const);
      for (const op of ops) {
        assert(
          gemmRowsCases.some(
            (sweepCase) => sweepCase.op === op && profileSegmentOf(sweepCase.m) === segment,
          ),
          `segment ≤ ${maxRows} has no ${op}`,
        );
      }
    });
  });
});

describe("shape table", () => {
  it("ids are unique and every op family has at least one case", () => {
    const ids = SWEEP_CASES.map((sweepCase) => sweepCase.id);
    assertEquals(new Set(ids).size, ids.length);
    for (const op of SWEEP_OPS) {
      assert(
        SWEEP_CASES.some((sweepCase) => sweepCase.op === op),
        op,
      );
    }
  });

  it("census-derived cases have a positive integer census count and a positive operation count", () => {
    for (const sweepCase of SWEEP_CASES) {
      if (sweepCase.op === "matmul") continue;
      assert(Number.isInteger(sweepCase.censusCount) && sweepCase.censusCount > 0, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
    }
  });

  it("matmul is a mirror absent from the census: count 0, mirrored from the linear case with the same M / N / K", () => {
    const matmuls = SWEEP_CASES.flatMap((sweepCase) =>
      sweepCase.op === "matmul" ? [sweepCase] : [],
    );
    assert(matmuls.length > 0);
    for (const sweepCase of matmuls) {
      assertEquals(sweepCase.censusCount, 0, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
      assertStringIncludes(sweepCase.source, sweepCase.mirrorOf, sweepCase.id);
      const mirror = SWEEP_CASES.find((other) => other.id === sweepCase.mirrorOf);
      assert(
        mirror !== undefined && mirror.op === "linear",
        `${sweepCase.id}: the mirrored case is not among the linear cases`,
      );
      assertEquals(
        [mirror.m, mirror.n, mirror.k],
        [sweepCase.m, sweepCase.n, sweepCase.k],
        sweepCase.id,
      );
    }
  });

  it("matmul has one case per each of the profile's 7 gemmRows segments and also covers all 3 default buckets (≤ 64 / 65–512 / > 512)", () => {
    const matmuls = SWEEP_CASES.flatMap((sweepCase) =>
      sweepCase.op === "matmul" ? [sweepCase] : [],
    );
    assertEquals(
      matmuls.map((sweepCase) => profileSegmentOf(sweepCase.m)).sort((a, b) => a - b),
      PROFILE_GEMM_ROWS_BOUNDS.map((_, index) => index),
    );
    const defaultBuckets = new Set(
      matmuls.map((sweepCase) =>
        GEMM_ROWS_BUCKETS.findIndex((bucket) => sweepCase.m <= bucket.maxRows),
      ),
    );
    assertEquals(
      [...defaultBuckets].sort((a, b) => a - b),
      [0, 1, 2],
    );
  });

  it("holds only shapes that take the same vec4 path as production (the sweep targets what production generates)", () => {
    for (const sweepCase of SWEEP_CASES) {
      switch (sweepCase.op) {
        case "linear":
        case "matmul":
        case "bmm":
          assert(gemmUsesVec4(sweepCase.k, sweepCase.n), sweepCase.id);
          break;
        case "i8a8-linear":
          // the i8a8 eligibility test (k % 4 == 0) and the i32 reduction check (the predicates of
          // recipe-builders/linear.ts)
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
          // ① is eligible with D % 4 and ③ with N % 4; s16 only for shapes where ① takes v4
          assertEquals(sweepCase.d % 4, 0, sweepCase.id);
          assertEquals(sweepCase.n % 4, 0, sweepCase.id);
          assertEquals(sweepCase.score, "f16", sweepCase.id);
          assert(attentionScoreUsesF16(sweepCase.d, sweepCase.n), sweepCase.id);
          break;
        case "conv2d":
          assert(conv2dUsesVec4(sweepCase.channelsIn * 9, sweepCase.width, 1), sweepCase.id);
          break;
        case "i8a8-conv3d": {
          // The w8a8 path's eligibility (Cin % 4, the i32 bound, groups 1) and the v4 store
          const { dims } = sweepCase;
          assert(conv3dI8a8EligibleShape(dims.channelsIn, dims.groups), sweepCase.id);
          assert(conv3dI8a8UsesVec4(dims.timeOut * dims.heightOut * dims.widthOut), sweepCase.id);
          break;
        }
      }
    }
  });

  it("conv3d w8a8 operation count is 2 · Cout · Cin · taps · P_out (a 3×3×3 case and the 1×1×1 case, exact)", () => {
    const flopsOf = (id: string): number => {
      const sweepCase = SWEEP_CASES.find((candidate) => candidate.id === id);
      assert(sweepCase !== undefined, id);
      return caseFlops(sweepCase);
    };
    // Cout 1024 · Cin 1024 · taps 3·3·3 = 27 · P_out = 1 (T: 3 in, no padding) · 16 · 16
    assertEquals(flopsOf("i8a8-conv3d-cin1024-cout1024-k333-in3x16x16"), 14_495_514_624);
    // Cout 512 · Cin 1024 · taps 1 · P_out = 4 · 64 · 64 = 16,384
    assertEquals(flopsOf("i8a8-conv3d-cin1024-cout512-k111-in4x64x64"), 17_179_869_184);
  });

  it("conv3d w8a8: the six heavy 3×3×3 shapes of tile 16, conv_out and one 1×1×1 shortcut, and the six heavy 3×3×3 shapes of tile 32 (ADR 0115 addendum 2026-10-26)", () => {
    const conv3d = SWEEP_CASES.flatMap((sweepCase) =>
      sweepCase.op === "i8a8-conv3d" ? [sweepCase] : [],
    );
    const heavy = (tile: number): string[] => [
      `i8a8-conv3d-cin256-cout256-k333-in6x${8 * tile}x${8 * tile}`,
      `i8a8-conv3d-cin512-cout512-k333-in6x${4 * tile}x${4 * tile}`,
      `i8a8-conv3d-cin1024-cout1024-k333-in4x${2 * tile}x${2 * tile}`,
      `i8a8-conv3d-cin1024-cout1024-k333-in3x${tile}x${tile}`,
      `i8a8-conv3d-cin1024-cout512-k333-in6x${4 * tile}x${4 * tile}`,
      `i8a8-conv3d-cin512-cout256-k333-in6x${8 * tile}x${8 * tile}`,
    ];
    assertEquals(
      conv3d.map((sweepCase) => sweepCase.id),
      [
        ...heavy(16),
        "i8a8-conv3d-cin256-cout12-k333-in6x128x128",
        "i8a8-conv3d-cin1024-cout512-k111-in4x64x64",
        ...heavy(32),
      ],
    );
    // The decoder's attrs: stride 1, dilation 1, groups 1, no time padding (the cache frames are
    // in the input), spatial padding (k - 1) / 2; the output extents follow the conv formula
    const extent = (input: number, kernel: number, padding: number): number =>
      input + 2 * padding - (kernel - 1);
    for (const { id, dims } of conv3d) {
      assertEquals([dims.strideT, dims.strideH, dims.strideW], [1, 1, 1], id);
      assertEquals([dims.dilationT, dims.dilationH, dims.dilationW], [1, 1, 1], id);
      assertEquals(dims.groups, 1, id);
      assertEquals(dims.kernelT, dims.kernelH, id);
      assertEquals(dims.kernelH, dims.kernelW, id);
      assertEquals(dims.paddingT, 0, id);
      assertEquals([dims.paddingH, dims.paddingW], Array(2).fill((dims.kernelH - 1) / 2), id);
      assertEquals(dims.timeOut, extent(dims.timeIn, dims.kernelT, dims.paddingT), id);
      assertEquals(dims.heightOut, extent(dims.heightIn, dims.kernelH, dims.paddingH), id);
      assertEquals(dims.widthOut, extent(dims.widthIn, dims.kernelW, dims.paddingW), id);
    }
  });

  it("conv2d has at least one row for each of the production geometries with m tiles of 32 and 64 rows", () => {
    const mTiles = new Set(
      SWEEP_CASES.flatMap((sweepCase) =>
        sweepCase.op === "conv2d" ? [conv2dIgemmMTile(sweepCase.channelsOut)] : [],
      ),
    );
    assertEquals(
      [...mTiles].sort((a, b) => a - b),
      [32, 64],
    );
  });

  it("attention lists ①QK and ③PV in pairs of the same shape", () => {
    for (const op of ["attention", "i8a8-attention"] as const) {
      const shapes = (stage: "qk" | "pv"): string[] =>
        SWEEP_CASES.flatMap((sweepCase) =>
          sweepCase.op === op && sweepCase.stage === stage
            ? [`${sweepCase.batchHeads}/${sweepCase.m}/${sweepCase.n}/${sweepCase.d}`]
            : [],
        );
      assertEquals(shapes("qk"), shapes("pv"), op);
    }
  });
});

describe("sweepCaseEntries (the public case list, geometrySweepCases of ./tune)", () => {
  it("lists every case of the table as its id and op, in the table's order", () => {
    assertEquals(
      sweepCaseEntries(),
      SWEEP_CASES.map(({ id, op }) => ({ id, op })),
    );
  });

  describe("when a caller changes the returned list", () => {
    it("leaves the table and the next call untouched", () => {
      const before = SWEEP_CASES.length;
      const entries = sweepCaseEntries();
      entries.push({ id: "extra", op: "linear" });
      const [first] = entries;
      assert(first !== undefined);
      Object.assign(first, { id: "renamed" });
      assertEquals(SWEEP_CASES.length, before);
      assertEquals(sweepCaseEntries()[0]?.id, SWEEP_CASES[0]?.id);
      assert(SWEEP_CASES[0]?.id !== "renamed");
    });
  });
});
