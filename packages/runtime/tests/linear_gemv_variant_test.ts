import { assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  defaultLinearGemvVariant,
  linearGemvKey,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";

describe("M=1 GEMVの形状選択", () => {
  it("実測した大語彙INT8をc16へ選び、別キーでWGSLを識別する", () => {
    const selected = defaultLinearGemvVariant({ storage: "i8", n: 262144, k: 1536 });
    assertEquals(selected, { cols: 16, unroll: 4 });
    assertEquals(linearGemvKey("i8", undefined, selected), "linear_gemv:v1:f32:c16u4:wi8");
    assertNotEquals(linearGemvKey("i8", undefined, selected), linearGemvKey("i8"));
    assertEquals(
      linearGemvWgsl("i8", undefined, selected),
      linearGemvWgsl("i8").replace("32 列 / wg", "16 列 / wg").replace(
        "@workgroup_size(32)",
        "@workgroup_size(16)",
      ),
    );
  });

  it("格納型や行列形が異なる場合と形状未指定の生成は従来の変種を保つ", () => {
    assertEquals(defaultLinearGemvVariant(), { cols: 32, unroll: 4 });
    const storageTypes: readonly WeightStorage[] = ["i2", "i4", "f16", "f32"];
    for (const storage of storageTypes) {
      assertEquals(defaultLinearGemvVariant({ storage, n: 262144, k: 1536 }), {
        cols: 32,
        unroll: 4,
      });
    }
    for (
      const [n, k] of [[262140, 1536], [262148, 1536], [262144, 1520], [262144, 1552], [6144, 1536]]
    ) {
      assertEquals(defaultLinearGemvVariant({ storage: "i8", n, k }), { cols: 32, unroll: 4 });
    }
  });
});
