import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelWgsl,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
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
          const subgroupKey = linearGemvSubgroupKey(storage, group, lanes);
          const subgroupShader = linearGemvSubgroupWgsl(storage, group, lanes);
          assertNotEquals(subgroupKey, key);
          assertEquals(keys.has(subgroupKey), false);
          keys.set(subgroupKey, subgroupShader);
          assertEquals(linearGemvSubgroupWgsl(storage, group, lanes), subgroupShader);
        }
      }
    }
    assertEquals(keys.size, 60);
  });
  it("未対応格納と不正なgroupをコード生成前に拒否する", () => {
    assertThrows(() => linearGemvParallelWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("i4", 16, 4));
    assertThrows(() => linearGemvParallelWgsl("i8", 32, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i4", 16, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i8", 32, 4));
  });
});

describe("subgroup32変種の積和はparallelと同じ綴り", () => {
  // ADR 0101 決定 3（積和は parallel と同一）を綴りで縛る。parallel 族だけ fma 化して subgroup が
  // `acc + x * d` のまま残ると、Metal では縮約の入れ方が式形で変わり u32 一致が崩れる（ADR 0105 追記 4）。
  // GPU 実走の u32 一致門（helpers/gemv-subgroup-check.ts）は Deno で SKIP になるので、この門が常設側。
  const macLines = (wgsl: string): string[] =>
    wgsl.split("\n").map((line) => line.trim()).filter((line) => line.startsWith("acc = "));
  it("全60形でfma行の列が一致し、積和にacc + x * dが残らない", () => {
    let compared = 0;
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const parallel = macLines(linearGemvParallelWgsl(storage, group, lanes));
          const subgroup = macLines(linearGemvSubgroupWgsl(storage, group, lanes));
          const fma = (lines: string[]): string[] =>
            lines.filter((l) => l.startsWith("acc = fma("));
          assertEquals(fma(subgroup), fma(parallel), `${storage} g${group} l${lanes}`);
          assertEquals(fma(subgroup).length > 0, true);
          // subgroup の値交換（`if (lane < width) { acc = acc + other; }`）は if の行なので、
          // `acc = ` で始まる行に `acc + x * d` の綴りが 1 本も無いことが積和の門になる。
          assertEquals(
            subgroup.filter((l) => l.startsWith("acc = acc +")),
            [],
            `${storage} g${group} l${lanes}: 積和に acc + x * d が残っている`,
          );
          compared++;
        }
      }
    }
    assertEquals(compared, 30);
  });
});
