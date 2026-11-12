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
  sweepCasesFor,
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

describe("the case table of the evaluation set k85 (perf-ledger K-85)", () => {
  const k85 = sweepCasesFor("k85");

  it("is the Wan2.2 TI2V-5B DiT linears at 33 frames (S = 7,920) and a small-M guard, all i8a8 linear, outside the sweep's case set", () => {
    assertEquals(
      k85.map((sweepCase) =>
        sweepCase.op === "i8a8-linear"
          ? [sweepCase.m, sweepCase.n, sweepCase.k, sweepCase.censusCount]
          : [],
      ),
      [
        [7920, 3072, 3072, 180],
        [7920, 14336, 3072, 30],
        [7920, 3072, 14336, 30],
        [512, 3072, 3072, 61],
        [1, 18432, 3072, 2],
        [185, 1280, 1280, 0],
      ],
    );
    const ids = new Set(SWEEP_CASES.map(({ id }) => id));
    for (const sweepCase of k85) {
      assert(!ids.has(sweepCase.id), `${sweepCase.id} is also a case of the sweep's table`);
      assert(sweepCase.id.startsWith("k85-i8a8-linear-"), sweepCase.id);
    }
    assertEquals(new Set(k85.map(({ id }) => id)).size, k85.length);
    // 303 of the graph's 310 linears are weighted (the 7 left out are 1–2 nodes each)
    assertEquals(
      k85.reduce((total, sweepCase) => total + sweepCase.censusCount, 0),
      303,
    );
  });

  it("holds shapes the vec4 fill and the i8a8 path take (k % 16 == 0, n % 4 == 0, k within the i32 guard)", () => {
    for (const sweepCase of k85) {
      assert(sweepCase.op === "i8a8-linear", sweepCase.id);
      assertEquals(sweepCase.k % 16, 0, sweepCase.id);
      assertEquals(sweepCase.n % 4, 0, sweepCase.id);
      assert(sweepCase.k <= LINEAR_I8A8_MAX_K, sweepCase.id);
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
    }
  });

  it("every other set sweeps the sweep's table, and the public list gives the k85 cases with their census count", () => {
    for (const set of ["quick", "quick+", "full"] as const)
      assertEquals(sweepCasesFor(set), SWEEP_CASES);
    assertEquals(
      sweepCaseEntries("k85"),
      k85.map(({ id, op, censusCount }) => ({ id, op, censusCount })),
    );
    assertEquals(sweepCaseEntries("full"), sweepCaseEntries());
  });
});

describe("the case table of the evaluation set k88 (perf-ledger K-88)", () => {
  const k88 = sweepCasesFor("k88");
  /** One row per case: op, the dispatch shape, the weight storage, the count and its unit. */
  const rows = k88.map((sweepCase) => {
    switch (sweepCase.op) {
      case "linear":
        return [
          "linear",
          `${sweepCase.m}x${sweepCase.n}x${sweepCase.k}`,
          `${sweepCase.weight ?? ""}${sweepCase.groupSize ?? ""}`,
          sweepCase.censusCount,
          sweepCase.censusGroup ?? "",
        ];
      case "matmul":
        return ["matmul", `${sweepCase.m}x${sweepCase.n}x${sweepCase.k}`, "", 0, ""];
      case "conv2d":
        return [
          "conv2d",
          `b${sweepCase.batch ?? 1} c${sweepCase.channelsIn}>${sweepCase.channelsOut} ${sweepCase.height}x${sweepCase.width}`,
          sweepCase.weight ?? "",
          sweepCase.censusCount,
          sweepCase.censusGroup ?? "",
        ];
      default:
        return [sweepCase.op];
    }
  });

  it("is the gemma4 E2B prefill, umT5, siglip2 and guard linears, the square matmul, and the Wan2.2 and anima VAE conv2d, with their storage and counts, outside every other table", () => {
    const gemma = "one gemma4 E2B prefill chunk (M = 768)";
    const next = "one Wan2.2 TI2V-5B VAE decoder next chunk (tile 16)";
    const first = "one Wan2.2 TI2V-5B VAE decoder first chunk (tile 16)";
    const anima = "one anima VAE decode (the graph's 64×64 latent)";
    assertEquals(rows, [
      ["linear", "768x6144x1536", "i432", 30, gemma],
      ["linear", "768x12288x1536", "i432", 40, gemma],
      ["linear", "768x1536x6144", "i432", 15, gemma],
      ["linear", "768x1536x12288", "i432", 20, gemma],
      ["linear", "768x2048x1536", "i432", 28, gemma],
      ["linear", "768x4096x1536", "i432", 7, gemma],
      ["linear", "512x10240x4096", "i8", 48, "one umT5-XXL encoder forward (512 tokens)"],
      ["linear", "729x4304x1152", "f32", 27, "one siglip2 so400m vision forward (729 tokens)"],
      ["linear", "64x4096x4096", "f32", 0, ""],
      ["linear", "4096x4096x4096", "f32", 0, ""],
      ["matmul", "4096x4096x4096", "", 0, ""],
      ["conv2d", "b2 c1024>1024 32x32", "i8", 1, next],
      ["conv2d", "b4 c1024>1024 64x64", "i8", 1, next],
      ["conv2d", "b4 c512>512 128x128", "i8", 1, next],
      ["conv2d", "b1 c1024>1024 32x32", "i8", 1, first],
      ["conv2d", "b1 c1024>1024 64x64", "i8", 1, first],
      ["conv2d", "b1 c512>512 128x128", "i8", 1, first],
      ["conv2d", "b1 c192>192 256x256", "f16", 6, anima],
      ["conv2d", "b1 c384>384 128x128", "f16", 5, anima],
    ]);
    const others = new Set([...SWEEP_CASES, ...sweepCasesFor("k85")].map(({ id }) => id));
    for (const sweepCase of k88) {
      assert(!others.has(sweepCase.id), `${sweepCase.id} is also a case of another table`);
      assert(sweepCase.id.startsWith("k88-"), sweepCase.id);
    }
    assertEquals(new Set(k88.map(({ id }) => id)).size, k88.length);
    // The gemma4 cases weight 140 of the prefill chunk's 277 linears
    assertEquals(
      k88.reduce(
        (total, sweepCase) =>
          total +
          ("censusGroup" in sweepCase && sweepCase.censusGroup === gemma
            ? sweepCase.censusCount
            : 0),
        0,
      ),
      140,
    );
  });

  it("holds shapes the forms and the decode take: the vec4 kernels (k % 4, n % 4; conv2d's K, Wout and stride), whole i4 groups, conv2d's flops per batch", () => {
    for (const sweepCase of k88) {
      assert(caseFlops(sweepCase) > 0, sweepCase.id);
      switch (sweepCase.op) {
        case "linear":
        case "matmul":
          assert(gemmUsesVec4(sweepCase.k, sweepCase.n), sweepCase.id);
          if (sweepCase.op === "linear" && sweepCase.weight === "i4") {
            assertEquals(sweepCase.k % (sweepCase.groupSize ?? 0), 0, sweepCase.id);
          }
          break;
        case "conv2d":
          assert(conv2dUsesVec4(sweepCase.channelsIn * 9, sweepCase.width, 1), sweepCase.id);
          assertEquals(
            caseFlops(sweepCase),
            2 *
              (sweepCase.batch ?? 1) *
              sweepCase.channelsOut *
              sweepCase.height *
              sweepCase.width *
              sweepCase.channelsIn *
              9,
          );
          break;
        default:
          assert(false, `${sweepCase.id}: an op outside the forms`);
      }
    }
  });

  it("the public list gives the k88 cases with their count and its unit, and the other sets' lists are unchanged", () => {
    assertEquals(
      sweepCaseEntries("k88"),
      k88.map((sweepCase) => ({
        id: sweepCase.id,
        op: sweepCase.op,
        censusCount: sweepCase.censusCount,
        ...("censusGroup" in sweepCase && sweepCase.censusGroup !== undefined
          ? { censusGroup: sweepCase.censusGroup }
          : {}),
      })),
    );
    assertEquals(
      sweepCaseEntries("k88").filter((entry) => entry.censusGroup === undefined).length,
      3,
    );
    for (const set of ["quick", "quick+", "full"] as const) {
      assertEquals(sweepCasesFor(set), SWEEP_CASES);
    }
    assertEquals(sweepCaseEntries("full"), sweepCaseEntries());
  });
});
