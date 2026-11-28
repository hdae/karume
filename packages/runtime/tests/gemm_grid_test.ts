// The grid-fill rule of the GEMM of linear / matmul / bmm (`gemmGrid` and `gemmShapeGeometry` in
// src/kernels/geometry-profile.ts), without a GPU:
//
// 1. The walk down the ladder: the largest tile on a full grid with K at its floor, a smaller tile
//    where the grid is under-filled, the last tile when none fills, the padding bound and the K floor
//    at their edges, the batch counted on the z axis.
// 2. On the recon's 125 shapes of 2026-11-28 (the default presets' GEMM-path linears and bmms,
//    outputs/misc/gemm-geometry-recon-2026-11-28/shapes.ts), through an injected test profile with
//    the fitted integers (256, 48), the rule picks the tile the recon's own fitting script (fit.ts,
//    `two(256, 48, 1.5, 256)`) picked — an oracle written apart from this implementation.
// 3. Without the field nothing moves: on every registered table and the default profile, for every
//    shape of a grid, the lookup is the `gemmRows` rule of M (the call sites' keys follow it).
// 4. A broken rule fails loudly at the check and at the lookup (fault injection: a ladder out of
//    order); the JSON of a table with the rule reads back to the same rule.
// 5. The linear's dispatch program takes the rule's tile in its GEMM key, and the split-K partial
//    takes the tile of the reference GEMM of the same shape (its one-slice identity rests on it).

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { type GemmGeometry, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import {
  assertGeometryProfile,
  assertGeometryProfileWithinLimits,
  DEFAULT_GEOMETRY_PROFILE,
  type GemmGridRule,
  gemmGridPadding,
  gemmGridStep,
  gemmGridWorkgroups,
  gemmRowsGeometry,
  type GemmShape,
  gemmShapeGeometry,
  type GeometryProfile,
  selectGeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { linearSplitKGemmKey } from "../src/kernels/linear-splitk.ts";
import { linearProgramFor } from "../src/runtime/linear-program.ts";
import { geometryProfileJson } from "../src/tune/derive.ts";
import { GeometryProfileParseError, parseGeometryProfileJson } from "../src/tune/profile-json.ts";

/** The recon's ladder (largest first): 128×128, 64×64 / 256, 64×32 / 128, 32×32 / 128, 16×16 / 64. */
const LADDER: readonly GemmGeometry[] = [
  { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16 },
  { regM: 2, regN: 4, wgX: 8, wgY: 16 },
  { regM: 1, regN: 4, wgX: 4, wgY: 16 },
];

/** The rule the recon fitted on the RTX 3080 Ti (keep 128×128 at 256 workgroups and K ≥ 256, else 48). */
const FITTED: GemmGridRule = {
  ladder: LADDER,
  keepLargest: { minWorkgroups: 256, minK: 256 },
  minWorkgroups: 48,
  maxPadding: 1.5,
};

/** The default profile with a grid-fill rule (an injected test table). */
const withGrid = (rule: GemmGridRule, id = "grid-test"): GeometryProfile => ({
  ...DEFAULT_GEOMETRY_PROFILE,
  id,
  gemmGrid: rule,
});

const tile = (geometry: GemmGeometry): string => `${gemmTileM(geometry)}x${gemmTileN(geometry)}`;

const shape = (m: number, n: number, k: number, batch = 1): GemmShape => ({ m, n, k, batch });

/**
 * The recon's 125 shape classes and the tile its fitting script picked under `two(256, 48, 1.5,
 * 256)` (outputs/misc/gemm-geometry-recon-2026-11-28/fit.ts over final.tsv): family, op, M, N, K,
 * batch, tileM×tileN.
 */
const RECON_SHAPES: readonly (readonly [
  string,
  "linear" | "bmm",
  number,
  number,
  number,
  number,
  string,
])[] = [
  ["depth", "linear", 1370, 1536, 384, 1, "64x64"],
  ["depth", "linear", 1370, 384, 1536, 1, "64x64"],
  ["depth", "linear", 1370, 384, 384, 1, "64x64"],
  ["siglip2-base", "linear", 196, 3072, 768, 1, "64x64"],
  ["siglip2-base", "linear", 196, 768, 3072, 1, "64x64"],
  ["siglip2-base", "bmm", 196, 64, 196, 12, "64x64"],
  ["siglip2-base", "bmm", 196, 196, 64, 12, "64x32"],
  ["so400m", "linear", 729, 4304, 1152, 1, "64x64"],
  ["so400m", "bmm", 729, 72, 729, 16, "64x32"],
  ["lucida", "linear", 1296, 2304, 768, 1, "64x64"],
  ["lucida", "linear", 4096, 3072, 768, 1, "128x128"],
  ["lucida", "linear", 1024, 3072, 768, 1, "64x64"],
  ["lucida", "linear", 5184, 768, 768, 1, "64x64"],
  ["lucida", "linear", 1296, 768, 768, 1, "64x64"],
  ["lucida", "linear", 1024, 1536, 6144, 1, "64x64"],
  ["lucida", "linear", 1296, 4608, 1536, 1, "128x128"],
  ["lucida", "linear", 1296, 1536, 1536, 1, "64x64"],
  ["lucida", "linear", 17424, 384, 384, 1, "128x128"],
  ["lucida", "linear", 256, 6144, 1536, 1, "64x64"],
  ["lucida", "bmm", 144, 32, 144, 216, "64x32"],
  ["lucida", "bmm", 144, 144, 32, 216, "64x32"],
  ["lucida", "bmm", 144, 32, 144, 864, "64x32"],
  ["birefnet", "linear", 4096, 768, 3072, 1, "64x64"],
  ["birefnet", "linear", 1024, 768, 3072, 1, "64x64"],
  ["birefnet", "linear", 1296, 768, 768, 1, "64x64"],
  ["eg2", "linear", 256, 512, 512, 1, "64x32"],
  ["eg2", "linear", 256, 512, 2048, 1, "64x32"],
  ["eg2", "linear", 256, 1024, 512, 1, "64x64"],
  ["eg2", "linear", 256, 512, 1024, 1, "64x32"],
  ["eg2", "linear", 64, 2048, 512, 1, "64x32"],
  ["eg2", "linear", 64, 512, 512, 1, "16x16"],
  ["g4-e2b-bf16", "linear", 768, 12288, 1536, 1, "128x128"],
  ["g4-e2b-bf16", "linear", 768, 1536, 12288, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 6144, 1536, 1, "128x128"],
  ["g4-e2b-bf16", "linear", 768, 1536, 2048, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 256, 1536, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 1536, 256, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 512, 1536, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 12288, 1536, 1, "128x128"],
  ["g4-e2b-i4", "linear", 768, 1536, 12288, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 6144, 1536, 1, "128x128"],
  ["g4-e2b-i4", "linear", 768, 1536, 6144, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 2048, 1536, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 1536, 2048, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 256, 1536, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 10240, 2560, 1, "128x128"],
  ["g4-e4b-i4", "linear", 768, 2560, 10240, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 2560, 2048, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 512, 2560, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 256, 2560, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 2560, 256, 1, "64x64"],
  ["g4-12b-i4", "linear", 768, 2048, 3840, 1, "64x64"],
  ["g4-12b-i4", "linear", 768, 512, 3840, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 2112, 2816, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 2816, 2112, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 2048, 2816, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 2816, 4096, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 4096, 2816, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 128, 2816, 1, "64x32"],
  ["g4m-e2b", "linear", 768, 1536, 12288, 1, "64x64"],
  ["g4m-e2b", "linear", 768, 256, 1536, 1, "64x64"],
  ["g4m-e2b", "linear", 768, 1536, 256, 1, "64x64"],
  ["anima-text", "linear", 64, 3072, 1024, 1, "64x64"],
  ["anima-text", "linear", 64, 1024, 1024, 1, "32x32"],
  ["anima-text", "linear", 64, 1024, 3072, 1, "32x32"],
  ["anima-text", "linear", 64, 2048, 1024, 1, "64x32"],
  ["anima-text", "linear", 512, 1024, 1024, 1, "64x64"],
  ["anima-text", "linear", 512, 4096, 1024, 1, "64x64"],
  ["anima-text", "bmm", 64, 64, 128, 16, "32x32"],
  ["anima-text", "bmm", 64, 128, 64, 16, "64x32"],
  ["anima-text", "bmm", 512, 512, 64, 16, "64x64"],
  ["sbv2", "linear", 512, 1024, 1024, 1, "64x64"],
  ["sbv2", "linear", 128, 1024, 1024, 1, "64x32"],
  ["sbv2", "linear", 128, 4096, 1024, 1, "64x64"],
  ["sbv2", "linear", 128, 1024, 4096, 1, "64x32"],
  ["sbv2", "bmm", 128, 512, 64, 16, "64x64"],
  ["sbv2", "bmm", 128, 64, 128, 16, "64x32"],
  ["umt5", "linear", 512, 10240, 4096, 1, "128x128"],
  ["umt5", "bmm", 512, 64, 512, 64, "64x64"],
  ["irodori-mf", "linear", 185, 1280, 1280, 1, "64x64"],
  ["irodori-mf", "linear", 185, 3680, 1280, 1, "64x64"],
  ["irodori-mf", "linear", 185, 1280, 3680, 1, "64x64"],
  ["irodori-mf", "bmm", 185, 1704, 64, 20, "64x64"],
  ["irodori-mf", "bmm", 185, 64, 1704, 20, "64x64"],
  ["sweep", "bmm", 64, 64, 128, 16, "32x32"],
  ["sweep", "bmm", 512, 64, 64, 16, "64x64"],
  ["sweep", "linear", 64, 3072, 1024, 1, "64x64"],
  ["sweep", "linear", 512, 2048, 1024, 1, "64x64"],
  ["depth", "bmm", 1370, 64, 1370, 6, "64x64"],
  ["depth", "bmm", 1370, 1370, 64, 6, "64x64"],
  ["siglip2-base", "linear", 196, 768, 768, 1, "64x64"],
  ["so400m", "linear", 729, 1152, 1152, 1, "64x64"],
  ["so400m", "linear", 729, 1152, 4304, 1, "64x64"],
  ["so400m", "bmm", 729, 729, 72, 16, "64x64"],
  ["lucida", "linear", 5184, 2304, 768, 1, "128x128"],
  ["lucida", "linear", 4096, 768, 3072, 1, "64x64"],
  ["lucida", "linear", 1024, 768, 3072, 1, "64x64"],
  ["lucida", "linear", 65536, 768, 192, 1, "64x64"],
  ["lucida", "linear", 16384, 1536, 384, 1, "128x128"],
  ["lucida", "linear", 1024, 6144, 1536, 1, "128x128"],
  ["lucida", "linear", 576, 4608, 1536, 1, "64x64"],
  ["lucida", "linear", 256, 1536, 6144, 1, "64x64"],
  ["lucida", "bmm", 144, 144, 32, 864, "64x32"],
  ["lucida", "bmm", 144, 32, 144, 2904, "64x32"],
  ["birefnet", "linear", 5184, 2304, 768, 1, "128x128"],
  ["eg2", "linear", 256, 2048, 512, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 1536, 6144, 1, "64x64"],
  ["g4-e2b-bf16", "linear", 768, 2048, 1536, 1, "64x64"],
  ["g4-e2b-i4", "linear", 768, 1536, 256, 1, "64x64"],
  ["g4-e4b-i4", "linear", 768, 2048, 2560, 1, "64x64"],
  ["g4-12b-i4", "linear", 768, 15360, 3840, 1, "128x128"],
  ["g4-12b-i4", "linear", 768, 3840, 15360, 1, "64x64"],
  ["g4-12b-i4", "linear", 768, 3840, 4096, 1, "64x64"],
  ["g4-26b-i4", "linear", 768, 1024, 2816, 1, "64x64"],
  ["g4m-e2b", "linear", 768, 12288, 1536, 1, "128x128"],
  ["anima-text", "bmm", 512, 64, 512, 16, "64x64"],
  ["umt5", "linear", 512, 4096, 10240, 1, "64x64"],
  ["umt5", "linear", 512, 4096, 4096, 1, "64x64"],
  ["umt5", "bmm", 512, 512, 64, 64, "64x64"],
  ["irodori-mf", "linear", 512, 1280, 512, 1, "64x64"],
  ["irodori-mf", "linear", 256, 1280, 512, 1, "64x64"],
  ["sweep", "linear", 4096, 8192, 2048, 1, "128x128"],
  ["lucida", "linear", 17424, 1152, 384, 1, "128x128"],
  ["lucida", "linear", 65536, 192, 768, 1, "128x128"],
  ["g4-12b-i4", "linear", 768, 4096, 3840, 1, "64x64"],
];

describe("the walk down the grid-fill ladder", () => {
  const profile = withGrid(FITTED);

  it("keeps the largest tile where its grid is full and K reaches the floor, and steps down where the grid is under-filled", () => {
    // Gemma 4 E2B's gate / up at a 768-row chunk: 576 workgroups of 128×128
    assertEquals(tile(gemmShapeGeometry(profile, shape(768, 12288, 1536))), "128x128");
    // Its k / v projection: 12 workgroups of 128×128, 48 of 64×64
    assertEquals(tile(gemmShapeGeometry(profile, shape(768, 256, 1536))), "64x64");
    // 6 workgroups of 128×128, 24 of 64×64, 48 of 64×32 (the 26B router, N 128)
    assertEquals(tile(gemmShapeGeometry(profile, shape(768, 128, 2816))), "64x32");
  });

  it("takes the last tile when no step fills its grid", () => {
    assertEquals(tile(gemmShapeGeometry(profile, shape(16, 64, 512))), "16x16");
    assertEquals(gemmGridStep(FITTED, shape(16, 64, 512)), LADDER.length - 1);
  });

  it("keeps the largest tile at K = the floor and leaves it at one below", () => {
    // 16 × 16 = 256 workgroups of 128×128 at no padding
    const full = (k: number) => shape(2048, 2048, k);
    assertEquals(gemmGridWorkgroups(LADDER[0], full(256)), 256);
    assertEquals(gemmGridStep(FITTED, full(256)), 0);
    assertEquals(gemmGridStep(FITTED, full(255)), 1);
  });

  it("keeps the largest tile at exactly its least workgroup count and leaves it one row of tiles below", () => {
    assertEquals(gemmGridStep(FITTED, shape(2048, 2048, 512)), 0);
    // 2048 × 1920: 15 × 16 = 240 workgroups
    assertEquals(gemmGridStep(FITTED, shape(2048, 1920, 512)), 1);
  });

  it("takes a tile at a padding of exactly the bound and refuses it just past it", () => {
    // N 192 against 128×128: two column tiles over 192 columns = 256 / 192 = 4 / 3; eight matrices
    // of 2048 rows make 2 · 16 · 8 = 256 workgroups
    const atBound = { ...FITTED, maxPadding: 4 / 3 };
    const wide = shape(2048, 192, 512, 8);
    assertEquals(gemmGridPadding(LADDER[0], wide), 4 / 3);
    assertEquals(gemmGridWorkgroups(LADDER[0], wide), 256);
    assertEquals(gemmGridStep(atBound, wide), 0);
    assertEquals(gemmGridStep({ ...FITTED, maxPadding: 4 / 3 - 1e-9 }, wide), 1);
    // An N of 32 pads 128×128 by 4 at any M: the largest tile never fits
    assertEquals(gemmGridPadding(LADDER[0], shape(4096, 32, 512)), 4);
    assertNotEquals(gemmGridStep(FITTED, shape(4096, 32, 512, 64)), 0);
  });

  it("counts the batch on the grid's z axis", () => {
    // bmm 16 × [64, 128] × [128, 64]: 16 workgroups of 64×64 at batch 1 would fill nothing
    assertEquals(gemmGridWorkgroups(LADDER[1], shape(64, 64, 128, 16)), 16);
    assertEquals(gemmGridWorkgroups(LADDER[2], shape(64, 64, 128, 16)), 32);
    assertEquals(gemmGridWorkgroups(LADDER[3], shape(64, 64, 128, 16)), 64);
    assertEquals(tile(gemmShapeGeometry(profile, shape(64, 64, 128, 16))), "32x32");
    assertEquals(tile(gemmShapeGeometry(profile, shape(64, 64, 128, 1))), "16x16");
  });
});

describe("the grid-fill rule on the recon's 125 shapes (RTX 3080 Ti, 2026-11-28)", () => {
  it("picks, through an injected test profile, the tile the recon's fitting script picked on every shape", () => {
    assertEquals(RECON_SHAPES.length, 125);
    const profile = withGrid(FITTED);
    const differing = RECON_SHAPES.flatMap(([family, op, m, n, k, batch, expected]) => {
      const picked = tile(gemmShapeGeometry(profile, shape(m, n, k, batch)));
      return picked === expected
        ? []
        : [`${family} ${op} ${m}×${n}×${k} b${batch}: ${picked} (fit.ts ${expected})`];
    });
    assertEquals(differing, []);
    // Every step of the ladder is reached on these shapes
    const reached = new Set(
      RECON_SHAPES.map(([, , m, n, k, batch]) => gemmGridStep(FITTED, shape(m, n, k, batch))),
    );
    assertEquals(
      [...reached].toSorted((a, b) => a - b),
      [0, 1, 2, 3, 4],
    );
  });
});

describe("without the field, the lookup is the gemmRows rule of M", () => {
  it("gives, on the default profile and every registered table, the gemmRows geometry of M for every shape of a grid", () => {
    const rows = [1, 16, 17, 64, 65, 300, 512, 513, 768, 4096, 65536];
    for (const profile of [DEFAULT_GEOMETRY_PROFILE, ...BUILTIN_GEOMETRY_PROFILES]) {
      assertEquals(profile.gemmGrid, undefined, `${profile.id} carries no grid-fill rule`);
      for (const m of rows) {
        for (const [n, k, batch] of [
          [32, 64, 1],
          [384, 1536, 1],
          [12288, 1536, 1],
          [64, 128, 216],
        ] as const) {
          // The same object as the bucket lookup (the call sites pass it to key, WGSL and grid)
          assert(
            gemmShapeGeometry(profile, shape(m, n, k, batch)) === gemmRowsGeometry(profile, m),
            `${profile.id} M ${m} N ${n} K ${k} batch ${batch}`,
          );
        }
      }
    }
  });

  it("a table that carries the field still selects by its match like any other", () => {
    const table = { ...withGrid(FITTED, "grid-matched"), match: { vendor: "test-vendor" } };
    assertEquals(
      selectGeometryProfile({ vendor: "test-vendor", architecture: "" }, [table]),
      table,
    );
  });
});

describe("a broken grid-fill rule fails loudly", () => {
  const broken: readonly (readonly [string, unknown, string])[] = [
    [
      "a ladder out of order",
      { ...FITTED, ladder: [LADDER[1], LADDER[0], ...LADDER.slice(2)] },
      "strictly decreasing tile area",
    ],
    [
      "two tiles of one area",
      { ...FITTED, ladder: [LADDER[0], LADDER[2], { regM: 2, regN: 8, wgX: 8, wgY: 16 }] },
      "strictly decreasing tile area",
    ],
    ["a single tile", { ...FITTED, ladder: [LADDER[0]] }, "two or more"],
    [
      "a tile with the counter decode",
      { ...FITTED, ladder: [LADDER[0], { ...LADDER[1], decode: "counter" }] },
      "counter decode",
    ],
    [
      "a tile that fails the geometry check",
      { ...FITTED, ladder: [LADDER[0], { regM: 3, regN: 4, wgX: 8, wgY: 8 }] },
      "gemmGrid.ladder[1]",
    ],
    [
      "a count that is no integer",
      { ...FITTED, minWorkgroups: 47.5 },
      "minWorkgroups must be a positive integer",
    ],
    [
      "a K floor of 0",
      { ...FITTED, keepLargest: { minWorkgroups: 256, minK: 0 } },
      "keepLargest.minK must be a positive integer",
    ],
    ["a padding bound below 1", { ...FITTED, maxPadding: 0.9 }, "at least 1"],
    [
      "a missing padding bound",
      { ladder: LADDER, keepLargest: FITTED.keepLargest, minWorkgroups: 48 },
      "maxPadding is not a number",
    ],
  ];

  it("at the profile check, at selection and at the lookup", () => {
    for (const [name, rule, message] of broken) {
      const profile = { ...DEFAULT_GEOMETRY_PROFILE, id: "broken-grid", gemmGrid: rule };
      assertThrows(() => assertGeometryProfile(profile), CodegenError, message, name);
      assertThrows(
        () =>
          selectGeometryProfile({ vendor: "test", architecture: "" }, [profile as GeometryProfile]),
        CodegenError,
        message,
        name,
      );
      if (name !== "a missing padding bound") {
        assertThrows(
          () => gemmShapeGeometry(profile as GeometryProfile, shape(768, 256, 1536)),
          CodegenError,
          message,
          name,
        );
      }
    }
  });

  it("refuses a shape without outputs or terms at the lookup", () => {
    for (const [name, value] of [
      ["M", shape(0, 64, 64)],
      ["N", shape(64, 0, 64)],
      ["K", shape(64, 64, 0)],
      ["batch", shape(64, 64, 64, 0)],
    ] as const) {
      assertThrows(() => gemmShapeGeometry(withGrid(FITTED), value), CodegenError, `${name} is 0`);
    }
  });

  it("names a ladder tile past the device's workgroup limits", () => {
    const limits = {
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 16384,
    };
    const small = withGrid({ ...FITTED, ladder: LADDER.slice(2) });
    const profile = {
      ...small,
      gemmRows: [{ maxRows: Number.POSITIVE_INFINITY, geometry: LADDER[2] }],
      attention: { qk: LADDER[2], pv: LADDER[2] },
      conv2d: { rows64: LADDER[2], rows32: LADDER[2] },
    };
    assertGeometryProfileWithinLimits(profile, limits);
    assertThrows(
      () => assertGeometryProfileWithinLimits({ ...profile, gemmGrid: FITTED }, limits),
      CodegenError,
      "gemmGrid.ladder[0]",
    );
  });
});

describe("the JSON of a table that carries the rule", () => {
  it("reads back to the same rule (skeleton forms included)", () => {
    const rule = {
      ...FITTED,
      ladder: [{ ...LADDER[0], form: "tb-un" as const }, ...LADDER.slice(1)],
    };
    const table = withGrid(rule);
    assertEquals(parseGeometryProfileJson(geometryProfileJson(table)), table);
  });

  it("refuses an unknown field inside the rule and a padding bound that is no finite number", () => {
    const text = (rule: unknown): string =>
      geometryProfileJson({
        ...DEFAULT_GEOMETRY_PROFILE,
        id: "json",
        gemmGrid: rule as GemmGridRule,
      });
    assertThrows(
      () => parseGeometryProfileJson(text({ ...FITTED, minK: 256 })),
      GeometryProfileParseError,
      "unknown fields minK",
    );
    assertThrows(
      () => parseGeometryProfileJson(text({ ...FITTED, maxPadding: "1.5" })),
      GeometryProfileParseError,
      "gemmGrid.maxPadding is not a finite number",
    );
    assertThrows(
      () => parseGeometryProfileJson(text({ ...FITTED, ladder: [LADDER[1], LADDER[0]] })),
      GeometryProfileParseError,
      "strictly decreasing tile area",
    );
  });
});

describe("the linear's dispatch program under the rule", () => {
  const facts = (m: number, n: number, k: number, profile: GeometryProfile) => ({
    x: [m, k],
    weight: [n, k],
    m,
    n,
    k,
    weightStorage: "f32" as const,
    groupSize: undefined,
    packedScale: undefined,
    prologue: undefined,
    geometryProfile: profile,
    linearI8a8Dot: "dp4a" as const,
  });

  it("names the rule's tile in the GEMM key, a tile the M bucket would not give", () => {
    const profile = withGrid(FITTED);
    const [m, n, k] = [768, 256, 1536];
    const v4 = gemmUsesVec4(k, n);
    const program = linearProgramFor(
      { id: "gemm-f32", compute: "f32", v4, groupSize: undefined },
      facts(m, n, k, profile),
    );
    const geometry = gemmShapeGeometry(profile, shape(m, n, k));
    assertNotEquals(tile(geometry), tile(gemmRowsGeometry(profile, m)));
    assertEquals(
      program.dispatches.map((dispatch) => dispatch.key),
      [linearKey("f32", v4, "f32", m, undefined, geometry)],
    );
  });

  it("gives the split-K partial the tile of the reference GEMM of the same shape at every step", () => {
    const profile = withGrid({
      ...FITTED,
      keepLargest: { minWorkgroups: 2, minK: 256 },
      minWorkgroups: 2,
    });
    for (const [m, n, k] of [
      [512, 512, 1536],
      [128, 64, 1536],
      [64, 64, 1536],
      [32, 64, 1536],
      [16, 16, 1536],
    ] as const) {
      const v4 = gemmUsesVec4(k, n);
      const reference = linearProgramFor(
        { id: "gemm-f32", compute: "f32", v4, groupSize: undefined },
        facts(m, n, k, profile),
      );
      const split = linearProgramFor(
        { id: "gemm-splitk", slices: 4, v4, groupSize: undefined },
        facts(m, n, k, profile),
      );
      const geometry = gemmShapeGeometry(profile, shape(m, n, k));
      assertEquals(
        reference.dispatches[0].key,
        linearKey("f32", v4, "f32", m, undefined, geometry),
      );
      assertEquals(
        split.dispatches[0].key,
        linearSplitKGemmKey("f32", v4, 4, m, undefined, geometry),
      );
    }
    const steps = [
      [512, 512, 1536],
      [128, 64, 1536],
      [64, 64, 1536],
      [32, 64, 1536],
      [16, 16, 1536],
    ].map(([m, n, k]) => gemmGridStep(profile.gemmGrid as GemmGridRule, shape(m, n, k)));
    assertEquals(steps, [0, 1, 2, 3, 4]);
  });
});
