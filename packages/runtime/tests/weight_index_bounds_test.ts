/**
 * 重みの論理要素添字が u32 に収まらない形を params が拒否すること（GPU 不要）。
 *
 * linear の GEMM 経路・w4a8 経路と embedding は重みを「論理要素の平坦添字」で引き、ローダ
 * （weight-storage.ts）が i2 なら `>> 4`・i4 なら `>> 3` して語位置へ変える。積
 * `行 × 行長` が先に u32 で折り返すと上位ビットは戻らず、別の行の重みを例外なしに読む。
 * i2 では格納が約 1 GiB で届くので binding 上限の検査では止まらない。巨大な重みは確保せず、
 * params の受理集合をここで固定する。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { CodegenError } from "../src/codegen/errors.ts";
import { embeddingParams } from "../src/kernels/embedding.ts";
import { linearParams } from "../src/kernels/linear.ts";
import { linearI8a8Params } from "../src/kernels/linear-i8a8.ts";

describe("linearParams: 重み [N, K] の論理要素数", () => {
  it("N·K ≥ 2^32 の形（M 65・N 131076・K 32768 で行 131072 が語 0 を指す）を拒否する", () => {
    assertThrows(() => linearParams(65, 131076, 32768), CodegenError, "n * k");
  });

  it("N·K = 2^32 ちょうどは拒否し、2^32 − 16 は受理する", () => {
    assertThrows(() => linearParams(65, 131072, 32768), CodegenError, "n * k");
    assertEquals([...linearParams(65, 268435455, 16)], [65, 268435455, 16, 0]);
  });
});

describe("linearI8a8Params: w4a8（i4 重み × i8 活性）の重み [N, K] の論理要素数", () => {
  it("N·K = 2^32 の形（行頭 wcol·dims.k が折り返す）を拒否し、2^32 − 32 は受理する", () => {
    assertThrows(() => linearI8a8Params(65, 131072, 32768, 32), CodegenError, "n * k");
    assertEquals([...linearI8a8Params(65, 134217727, 32, 32)], [65, 134217727, 32, 0]);
  });

  it("i8 変種は行頭を語数で作るので、同じ N·K = 2^32 を論理要素数の門では落とさない", () => {
    assertEquals([...linearI8a8Params(65, 131072, 32768)], [65, 131072, 32768, 0]);
  });
});

describe("embeddingParams: 重み [V, H] の論理要素数", () => {
  it("V·H = 2^32 の形（最後の行が先頭行の整数コードを読む）を拒否する", () => {
    assertThrows(() => embeddingParams(32768, 32768, 131072), CodegenError, "vocab * hidden");
  });

  it("V·H = 2^32 − 1 は受理する", () => {
    assertEquals(embeddingParams(1, 1, 0xffff_ffff)[2], 0xffff_ffff);
  });
});
