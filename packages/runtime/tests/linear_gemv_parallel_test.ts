import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelWgsl,
} from "../src/kernels/linear-gemv.ts";

describe("GEMV並列加算の選択とコード生成", () => {
  it("decodeと少数行の検証で同じ幾何を選び、prefillと未計測形へ拡張しない", () => {
    for (const m of [1, 4, 8]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), 32);
      assertEquals(linearGemvParallelLanes("i4", m, 6144, 1536, 512), 4);
      assertEquals(linearGemvParallelLanes("i4", m, 1536, 4096, 4096), 32);
      assertEquals(linearGemvParallelLanes("i2", m, 12288, 1536), 2);
      assertEquals(linearGemvParallelLanes("i8", m, 262144, 1536), 16);
    }
    for (const m of [0, 9, 64, 65]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), undefined);
    }
    assertEquals(linearGemvParallelLanes("i2", 1, 262144, 1536), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 256, 1536, 64), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 260, 1536, 32), undefined);
    assertEquals(linearGemvParallelLanes("f16", 1, 256, 1536), undefined);
  });
  it("キーが加算形・格納・group・laneを区別する", () => {
    const keys = new Map<string, string>();
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const key = linearGemvParallelKey(storage, group, lanes);
          const shader = linearGemvParallelWgsl(storage, group, lanes);
          assertNotEquals(key, linearGemvKey(storage, group));
          assertEquals(keys.has(key), false);
          keys.set(key, shader);
          assertEquals(linearGemvParallelWgsl(storage, group, lanes), shader);
        }
      }
    }
    assertEquals(keys.size, 30);
  });
  it("未対応格納と不正なgroupをコード生成前に拒否する", () => {
    assertThrows(() => linearGemvParallelWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("i4", 16, 4));
    assertThrows(() => linearGemvParallelWgsl("i8", 32, 4));
  });
});
