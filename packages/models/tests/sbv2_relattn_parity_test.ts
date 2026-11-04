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

test("relative-position tables: a different window width gives different tables (the parity check is not vacuous)", () => {
  // Fault injection — the parity check above depends on both sides reading the same constant. If
  // the tables stayed the same with the window width off by one, the check would let the error
  // through.
  const correct = buildRelattnTables(9, RELATTN_WINDOW_SIZE);
  const shifted = buildRelattnTables(9, RELATTN_WINDOW_SIZE + 1);

  assertEquals(
    correct.idxK.shape,
    shifted.idxK.shape,
    "the shapes are equal (= a fault the shape cannot detect)",
  );
  assert(
    correct.idxK.data.some((value, index) => value !== shifted.idxK.data[index]),
    "idx_k does not change with the window width off by one",
  );
  assert(
    correct.valid.data.some((value, index) => value !== shifted.valid.data[index]),
    "valid does not change with the window width off by one",
  );
});

test("relative-position tables: idx_k and valid act as a pair inside and outside the window", () => {
  // Pinning the meaning of the tables itself (the gather's premise written as a formula, not a copy
  // of the Python side): inside the window the index points plainly at `w + (j − i)` with valid=1;
  // outside, the clamp pins it to the edge with valid=0.
  const length = 12;
  const { idxK, valid } = buildRelattnTables(length, RELATTN_WINDOW_SIZE);
  const upper = 2 * RELATTN_WINDOW_SIZE;

  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const at = i * length + j;
      const inside = Math.abs(j - i) <= RELATTN_WINDOW_SIZE;
      assertEquals(valid.data[at], inside ? 1 : 0, `valid[${i}][${j}]`);
      // Already clamped, so always within the embedding even outside the window (no reliance on the
      // gather's out-of-range convention).
      assert(idxK.data[at] >= 0 && idxK.data[at] <= upper, `idx_k[${i}][${j}] is out of range`);
      if (inside) {
        assertEquals(idxK.data[at], RELATTN_WINDOW_SIZE + j - i, `idx_k[${i}][${j}]`);
      }
    }
  }
});

test("relative-position tables: rejects an invalid length or window width", () => {
  assertThrows(() => buildRelattnTables(0), RangeError, "length");
  assertThrows(() => buildRelattnTables(1.5), RangeError, "length");
  assertThrows(() => buildRelattnTables(4, -1), RangeError, "Window size");
});
