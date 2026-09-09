/**
 * argmax の 2 相分割（src/kernels/argmax.ts の「2 相分割」— MTP 段 4-B ③）の純関数側の門。
 *
 * 見るのは 2 点: ①経路の選択が形の純関数で、閾値の両側で決定的（同じ形は常に同じ経路 = 同じ
 * キー）②params が分割の形と食い違う呼びを落とす（`groups` を別の値で渡すと partial の書き先と
 * merge の読み元がずれ、沈黙誤値になる）。GPU 上の値（ビット同一・タイブレーク・NaN・全 −inf 行）は
 * gpu_ops_test.ts の argmax 門が 2 相形のケースも含めて見る。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  ARGMAX_SPLIT_MIN_DIM,
  ARGMAX_SPLIT_SPAN,
  argmaxSplitGroups,
  argmaxSplitParams,
  argmaxSplitPartialBytes,
} from "../src/kernels/argmax.ts";

Deno.test("argmax 2 相分割の経路選択は行長の純関数", async (t) => {
  await t.step("閾値未満は 1 dispatch 形（区間 0）", () => {
    for (const dim of [1, 256, 4096, ARGMAX_SPLIT_MIN_DIM - 1]) {
      assertEquals(argmaxSplitGroups(dim), 0, `dim=${dim}`);
    }
  });

  await t.step("閾値以上は区間 ceil(dim / span) 本（端数の区間も 1 本に数える）", () => {
    assertEquals(argmaxSplitGroups(ARGMAX_SPLIT_MIN_DIM), 4);
    assertEquals(argmaxSplitGroups(ARGMAX_SPLIT_MIN_DIM + 1), 5);
    // gemma4 の語彙長（drafter の lm_head 出口）
    assertEquals(argmaxSplitGroups(262144), 262144 / ARGMAX_SPLIT_SPAN);
  });

  await t.step("一時は [rows, groups] × (値 + index) の 8 バイト", () => {
    assertEquals(argmaxSplitPartialBytes(3, 64), 3 * 64 * 8);
  });
});

Deno.test("argmax 2 相分割の params は分割の形と一致しないと落ちる", async (t) => {
  await t.step("正しい形は 4 語（rows, dim, groups, −inf）", () => {
    const params = argmaxSplitParams(2, 262144, 64);
    assertEquals([...params.slice(0, 3)], [2, 262144, 64]);
    assertEquals(params[3], 0xff800000);
  });

  await t.step("groups が純関数の値と違えば落ちる（1 dispatch 形の dim も）", () => {
    assertThrows(() => argmaxSplitParams(1, 262144, 63), CodegenError, "分割の形に合わない");
    assertThrows(() => argmaxSplitParams(1, 4096, 1), CodegenError, "分割の形に合わない");
    assertThrows(() => argmaxSplitParams(1, 262144, 0), CodegenError, "分割の形に合わない");
  });
});
