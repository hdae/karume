// 閉区間表の二分探索（`src/text/code-ranges.ts`）を単体で縛る。GPU も実資産も要らない。
//
// ここを単体で縛る理由: この 1 本が anima（Qwen2 / T5 / spm）と sbv2（DeBERTa）の文字分類の
// **唯一の判定器**で、外すと例外を出さずに「別の文字分類」になる — pre-token の切れ目が変わり
// id 列が静かに別物になる形なので、家族側の e2e からは壊しても緑のまま見える位置にある。

import { assertEquals } from "@std/assert";
import { type CodeRanges, inCodeRanges } from "../src/text/code-ranges.ts";

Deno.test("inCodeRanges: 区間の両端は含み、その 1 つ外は含まない", () => {
  const ranges: CodeRanges = [[0x41, 0x5A]];
  assertEquals(inCodeRanges(ranges, 0x40), false);
  assertEquals(inCodeRanges(ranges, 0x41), true);
  assertEquals(inCodeRanges(ranges, 0x4D), true);
  assertEquals(inCodeRanges(ranges, 0x5A), true);
  assertEquals(inCodeRanges(ranges, 0x5B), false);
});

Deno.test("inCodeRanges: 1 点区間と表の両端の区間も落とさない", () => {
  // 先頭 / 末尾の区間は二分探索の縮小が片側へ寄るので、間の区間とは別に見る。
  const ranges: CodeRanges = [[0, 0], [0x10, 0x12], [0x10FFFF, 0x10FFFF]];
  assertEquals(inCodeRanges(ranges, 0), true);
  assertEquals(inCodeRanges(ranges, 1), false);
  assertEquals(inCodeRanges(ranges, 0x11), true);
  assertEquals(inCodeRanges(ranges, 0x10FFFE), false);
  assertEquals(inCodeRanges(ranges, 0x10FFFF), true);
});

Deno.test("inCodeRanges: 空表は何も含まない（表が落ちても素通りで気づける形にしない）", () => {
  assertEquals(inCodeRanges([], 0), false);
  assertEquals(inCodeRanges([], 0x41), false);
});

Deno.test("inCodeRanges: 区間数を振っても線形走査と一致する", () => {
  // 二分探索の分岐（偶数 / 奇数長・mid の切り捨て）を表の長さで振り、素朴な走査を正解に取る。
  // 表の区間は [4k, 4k+1] なので、隙間（4k+2 / 4k+3）も同時に見ている。
  for (let count = 1; count <= 16; count++) {
    const ranges: CodeRanges = Array.from({ length: count }, (_, k) => [4 * k, 4 * k + 1]);
    const naive = (cp: number): boolean => ranges.some(([start, end]) => cp >= start && cp <= end);
    for (let cp = 0; cp < 4 * count + 2; cp++) {
      assertEquals(inCodeRanges(ranges, cp), naive(cp), `count=${count} cp=${cp}`);
    }
  }
});
