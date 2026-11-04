import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { GEMM_TILE, gemmTileN } from "../src/kernels/gemm-geometry.ts";

const LIMIT = 65535;

/**
 * The actual tile edge conv2d's implicit GEMM uses along n (shared by the m-tile 64 / 32 variants).
 */
const CONV2D_N_TILE = gemmTileN(gemmMTileGeometry(GEMM_TILE));

test("grid-stride kernels clamp an over-limit count (the kernel loops over the rest)", () => {
  assertEquals(gridStrideWorkgroups(1024, 256, LIMIT), 4);
  assertEquals(gridStrideWorkgroups(1025, 256, LIMIT), 5);
  assertEquals(gridStrideWorkgroups(0, 256, LIMIT), 0);
  // 16 heads × 4096 tokens = 65536 rows (a shape hit by a real model)
  assertEquals(gridStrideWorkgroups(65536, 1, LIMIT), LIMIT);
  assertEquals(gridStrideWorkgroups(1_000_000_000, 256, LIMIT), LIMIT);
});

test("grid-stride fails loudly on element counts where the u32 loop index wraps on the last add", () => {
  // The kernel runs `i = i + stride` (stride = workgroup count × workgroup size) in u32 and tests
  // `i < n`, so when `n + stride > 2^32` the last add wraps and the loop never ends.
  const stride = LIMIT * 256;
  const edge = 2 ** 32 - stride;
  assertEquals(gridStrideWorkgroups(edge, 256, LIMIT), LIMIT, "n + stride = 2^32 passes");
  assertThrows(() => gridStrideWorkgroups(edge + 1, 256, LIMIT), DispatchLimitError, "2^32");
  // 4294901760 + 65535·256 wraps to 16711424 in u32 (the value from the review's reproduction)
  assertThrows(() => gridStrideWorkgroups(4294901760, 256, LIMIT), DispatchLimitError, "2^32");
  // Row-wise grid-stride (workgroup size 1 = rows advanced nwg.x at a time) has the same condition
  assertThrows(() => gridStrideWorkgroups(2 ** 32 - LIMIT + 1, 1, LIMIT), DispatchLimitError);
  assertEquals(gridStrideWorkgroups(2 ** 32 - LIMIT, 1, LIMIT), LIMIT);
});

test("tiled kernels fail loudly on an over-limit count (a missing tile is a silent wrong value)", () => {
  assertEquals(tiledWorkgroups(1024, 16, LIMIT, "matmul"), 64);
  assertEquals(tiledWorkgroups(1, 16, LIMIT, "matmul"), 1);
  assertEquals(tiledWorkgroups(0, 16, LIMIT, "matmul"), 0);
  assertThrows(() => tiledWorkgroups(LIMIT * 16 + 1, 16, LIMIT, "matmul"), DispatchLimitError);
  // conv2d's implicit GEMM (ADR 0024) is in the same category. The edge is **derived from the
  // geometry**, so first pin with a literal that the geometry has not moved, then pin the counts
  // for real-model shapes.
  assertEquals(CONV2D_N_TILE, 128, "if the n edge moves, the premise of the counts below changes");
  // A 2048px VAE has n = Hout·Wout = 4,194,304 (inside the limit with the current geometry)
  assertEquals(tiledWorkgroups(2048 * 2048, CONV2D_N_TILE, LIMIT, "conv2d"), 32_768);
  // One tile over the limit throws (pins that it fails rather than giving a silent wrong value)
  assertEquals(tiledWorkgroups(LIMIT * CONV2D_N_TILE, CONV2D_N_TILE, LIMIT, "conv2d"), LIMIT);
  assertThrows(
    () => tiledWorkgroups(LIMIT * CONV2D_N_TILE + 1, CONV2D_N_TILE, LIMIT, "conv2d"),
    DispatchLimitError,
  );
});
