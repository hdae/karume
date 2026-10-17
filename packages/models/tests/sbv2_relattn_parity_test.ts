// Asset-free tests of the host relative-attention table generator (`buildRelattnTables`):
// fault injection, the in/out-of-window semantics, and input validation. They always run.
//
// The byte-parity cases against the Python-generated golden tables and the baked window-width
// check read outputs/series/sbv2-F1, so they live in
// packages/e2e/tests/e2e_sbv2_relattn_parity_test.ts (ADR 0129). The fault-injection test below
// guards that parity check against being vacuous.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { buildRelattnTables, RELATTN_WINDOW_SIZE } from "../src/sbv2/relattn-tables.ts";

test("相対位置表: 窓幅が違えば表も違う（パリティ検査が恒真でない）", () => {
  // 故障注入 — 上のパリティ検査は「両側が同じ定数を見ている」ことに依存する。窓幅が
  // 1 ずれても表が同じなら、検査は誤りを通す形になっている。
  const correct = buildRelattnTables(9, RELATTN_WINDOW_SIZE);
  const shifted = buildRelattnTables(9, RELATTN_WINDOW_SIZE + 1);

  assertEquals(correct.idxK.shape, shifted.idxK.shape, "shape は同じ（= 検出できない形）");
  assert(
    correct.idxK.data.some((value, index) => value !== shifted.idxK.data[index]),
    "窓幅を 1 ずらしても idx_k が変わらない",
  );
  assert(
    correct.valid.data.some((value, index) => value !== shifted.valid.data[index]),
    "窓幅を 1 ずらしても valid が変わらない",
  );
});

test("相対位置表: 窓の内外で idx_k と valid が対で効く", () => {
  // 表の意味そのものの固定（Python 側の写しではなく、gather の前提を式で書く）:
  // 窓内は `w + (j − i)` を素直に指し valid=1、窓外は clamp で端へ張り付き valid=0。
  const length = 12;
  const { idxK, valid } = buildRelattnTables(length, RELATTN_WINDOW_SIZE);
  const upper = 2 * RELATTN_WINDOW_SIZE;

  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const at = i * length + j;
      const inside = Math.abs(j - i) <= RELATTN_WINDOW_SIZE;
      assertEquals(valid.data[at], inside ? 1 : 0, `valid[${i}][${j}]`);
      // clamp 済みなので窓外でも常に埋め込みの範囲内（gather の範囲外規約に頼らない）。
      assert(idxK.data[at] >= 0 && idxK.data[at] <= upper, `idx_k[${i}][${j}] が範囲外`);
      if (inside) {
        assertEquals(idxK.data[at], RELATTN_WINDOW_SIZE + j - i, `idx_k[${i}][${j}]`);
      }
    }
  }
});

test("相対位置表: 不正な長さ・窓幅は落とす", () => {
  assertThrows(() => buildRelattnTables(0), RangeError, "長さ");
  assertThrows(() => buildRelattnTables(1.5), RangeError, "長さ");
  assertThrows(() => buildRelattnTables(4, -1), RangeError, "窓幅");
});
