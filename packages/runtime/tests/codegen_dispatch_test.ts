import { assertEquals, assertThrows } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { GEMM_TILE, gemmTileN } from "../src/kernels/gemm-geometry.ts";

const LIMIT = 65535;

/** conv2d の implicit GEMM が n 方向に使う実タイル辺（m タイル 64 / 32 の変種で共通）。 */
const CONV2D_N_TILE = gemmTileN(gemmMTileGeometry(GEMM_TILE));

Deno.test("grid-stride カーネルは上限超過を縮退させる（カーネル側が残りを回す）", () => {
  assertEquals(gridStrideWorkgroups(1024, 256, LIMIT), 4);
  assertEquals(gridStrideWorkgroups(1025, 256, LIMIT), 5);
  assertEquals(gridStrideWorkgroups(0, 256, LIMIT), 0);
  // 16 head × 4096 token = 65536 行（実モデルで踏んだ形）
  assertEquals(gridStrideWorkgroups(65536, 1, LIMIT), LIMIT);
  assertEquals(gridStrideWorkgroups(1_000_000_000, 256, LIMIT), LIMIT);
});

Deno.test("タイル分割カーネルは上限超過を fail loudly にする（タイル欠落は沈黙誤値になる）", () => {
  assertEquals(tiledWorkgroups(1024, 16, LIMIT, "matmul"), 64);
  assertEquals(tiledWorkgroups(1, 16, LIMIT, "matmul"), 1);
  assertEquals(tiledWorkgroups(0, 16, LIMIT, "matmul"), 0);
  assertThrows(
    () => tiledWorkgroups(LIMIT * 16 + 1, 16, LIMIT, "matmul"),
    DispatchLimitError,
  );
  // conv2d の implicit GEMM（ADR 0024）も同じカテゴリ。辺は**幾何から導く**ので、まず幾何が
  // 動いていないことを期待値リテラルで固定してから、実モデル形の枚数を固定する。
  assertEquals(CONV2D_N_TILE, 128, "n の辺が動いたら以下の枚数の前提が変わる");
  // 2048px の VAE は n = Hout·Wout = 4,194,304（現行幾何では上限の内側に収まる）
  assertEquals(tiledWorkgroups(2048 * 2048, CONV2D_N_TILE, LIMIT, "conv2d"), 32_768);
  // 上限は 1 タイル超えただけで例外（沈黙誤値ではなく落ちることの固定）
  assertEquals(tiledWorkgroups(LIMIT * CONV2D_N_TILE, CONV2D_N_TILE, LIMIT, "conv2d"), LIMIT);
  assertThrows(
    () => tiledWorkgroups(LIMIT * CONV2D_N_TILE + 1, CONV2D_N_TILE, LIMIT, "conv2d"),
    DispatchLimitError,
  );
});
