// Asset-free invariants of the host-side DeBERTa relative-position table builder
// (buildRelPosTables). The byte parity against the Python-generated golden tables and the
// "no table baked into the container" check read the series assets, so they live in
// packages/e2e/tests/e2e_sbv2_rel_pos_parity_test.ts.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { buildRelPosTables } from "../src/sbv2/text/rel-pos-tables.ts";

// ---- Checks that need no assets (the parity above runs zero cases without the golden) ----
//
// The counterpart `sbv2_relattn_parity_test.ts` has two asset-free tests, but on this side
// `PARITY_CASES` is empty without the golden and **not one case runs** (green with neither the
// argument checks nor the mathematical properties verified). Byte parity with the Python side can
// only live in the parity above, but the structure the formula must satisfy (accepted arguments,
// transpose, Toeplitz) can be pinned without assets.

test("buildRelPosTables: accepted arguments (values below the lower bound the formula needs are rejected)", () => {
  // Length 0 not only empties the table; the diagonal array `2·length-1` gets a negative length.
  assertThrows(() => buildRelPosTables(0, 256, 512), RangeError, "is not an integer >= 1");
  // bucketSize 1 gives mid = 0 and breaks the log denominator (log((maxPosition-1)/mid)).
  assertThrows(
    () => buildRelPosTables(4, 1, 512),
    RangeError,
    "bucketSize 1 is not an integer >= 2",
  );
  // maxPosition 1 produces -Infinity through log(0/…).
  assertThrows(
    () => buildRelPosTables(4, 256, 1),
    RangeError,
    "maxPosition 1 is not an integer >= 2",
  );
});

test("buildRelPosTables: a pair for which the log-bucket formula does not hold fails instead of producing wrong indices through NaN", () => {
  // (bucket 2, maxPosition 2): mid = 1 gives the log denominator log(1) = 0 and the coefficient
  // mid − 1 = 0 → `Infinity × 0 = NaN`. NaN slips through the clamp comparisons and becomes 0 in an
  // Int32Array, a wrong gather index of the right shape (the position with |d| = 2 turns into 0).
  assertThrows(() => buildRelPosTables(4, 2, 2), RangeError, "log-bucket");
  // Even with mid ≥ 2, maxPosition − 1 ≤ mid makes the denominator ≤ 0 and breaks the far
  // direction.
  assertThrows(() => buildRelPosTables(4, 4, 3), RangeError, "log-bucket");
  assertThrows(() => buildRelPosTables(4, 3, 512), RangeError, "log-bucket");
});

test("buildRelPosTables: near positions at the distributed values (256 / 512) stay linear at d + 256", () => {
  // |d| < mid = 128 is not log-compressed (the near branch of `make_log_bucket_position`). The
  // counterpart showing that adding the entry checks did not change one element of the table for
  // the distributed values.
  const length = 7;
  const { c2pPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      assertEquals(c2pPos.data[i * length + j], i - j + 256, `c2p[${i}][${j}]`);
    }
  }
});

test("buildRelPosTables: p2c is the transpose of c2p (a consequence of the bucketing being an odd function)", () => {
  const length = 7;
  const { c2pPos, p2cPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      assertEquals(
        p2cPos.data[i * length + j],
        c2pPos.data[j * length + i],
        `p2c[${i}][${j}] differs from c2p[${j}][${i}]`,
      );
    }
  }
});

test("buildRelPosTables: the table depends only on i−j (Toeplitz)", () => {
  const length = 7;
  const { c2pPos } = buildRelPosTables(length, 256, 512);
  for (let i = 0; i + 1 < length; i += 1) {
    for (let j = 0; j + 1 < length; j += 1) {
      assertEquals(
        c2pPos.data[i * length + j],
        c2pPos.data[(i + 1) * length + (j + 1)],
        `the value of diagonal ${i - j} changes with position`,
      );
    }
  }
  // The counterpart showing the two tests above are not vacuous: `maxPosition` really moves the far
  // (logarithmic) values. A table with near positions only (length 7 against bucketSize 256) never
  // enters the logarithmic range, so this uses a shape that reaches it (|i−j| above 4 against mid =
  // 4).
  const wide = buildRelPosTables(16, 8, 512);
  const narrow = buildRelPosTables(16, 8, 64);
  assert(
    [...wide.c2pPos.data].some((value, index) => value !== narrow.c2pPos.data[index]),
    "the table is the same for a different maxPosition (the bucket rule is not read)",
  );
  // Near positions (the linear range) do not depend on maxPosition — pinning that only far
  // positions differ.
  // oxlint-disable-next-line oxc/erasing-op -- `row * width + column` spelled out for row 0 keeps the table coordinates readable
  assertEquals(wide.c2pPos.data[0 * 16 + 1], narrow.c2pPos.data[0 * 16 + 1]);
});
