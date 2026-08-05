import { assertEquals, assertThrows } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { GEMM_TILE } from "../src/kernels/gemm.ts";

const LIMIT = 65535;

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
  // conv2d の implicit GEMM（ADR 0024）も同じカテゴリ。2048px の VAE は n = Hout·Wout =
  // 4,194,304 → 65,536 タイルで上限を 1 だけ超える（沈黙誤値ではなく例外になることの固定）。
  assertEquals(tiledWorkgroups(1024 * 1024, GEMM_TILE, LIMIT, "conv2d"), 16_384);
  assertThrows(
    () => tiledWorkgroups(2048 * 2048, GEMM_TILE, LIMIT, "conv2d"),
    DispatchLimitError,
  );
});
