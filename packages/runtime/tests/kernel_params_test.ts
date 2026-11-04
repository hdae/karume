// The u32 range check of uniforms (`assertU32Params` in src/codegen/params.ts) and the kernel
// params entry points that go through it (no GPU needed).
//
// The point of the checks is twofold:
//
// 1. **The edges of the range** (0 and 0xffff_ffff) pass, and anything outside it (overflow /
//    negative / non-integer / NaN) always fails. Assigning into a `Uint32Array` silently takes the
//    value modulo 2^32, so letting an overflow through gives a silent wrong value: the dispatch
//    runs but with a different element count.
// 2. **The message carries the parameter name and the value.** An exception that does not say
//    which field overflowed is no starting point for diagnosis in a kernel with several uniform
//    words.
//
// On the kernel side, gather is taken as the representative (it accepted the element count with a
// bare safe-integer check — before the check was unified, a u32 overflow was silently narrowed).

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { assertU32Params } from "../src/codegen/params.ts";
import { GATHER_OOB_BITS, gatherParams } from "../src/kernels/gather.ts";

const U32_MAX = 0xffff_ffff;

test("assertU32Params passes the u32 range (0 to 0xffff_ffff) unchanged", () => {
  assertU32Params("where", { zero: 0, max: U32_MAX, mid: 1 });
});

test("assertU32Params fails loudly outside the range (overflow / negative / non-integer / NaN / non-finite)", () => {
  const cases: readonly (readonly [string, number])[] = [
    ["overflow (0xffff_ffff + 1)", U32_MAX + 1],
    ["negative", -1],
    ["non-integer", 1.5],
    ["NaN", NaN],
    ["+Infinity", Infinity],
  ];
  for (const [label, value] of cases) {
    assertThrows(
      () => assertU32Params("gather params", { count: value }),
      CodegenError,
      "count",
      label,
    );
  }
});

test("the assertU32Params message includes the parameter name and the value (shows which field overflowed)", () => {
  const error = assertThrows(
    () => assertU32Params("rope params", { n: 4, sequence: U32_MAX + 1 }),
    CodegenError,
  );
  assertEquals(
    error.message,
    "rope params: sequence must be a non-negative u32 integer (4294967296)",
  );
});

test("gatherParams rejects an element count above u32 instead of narrowing it", () => {
  // Inside the range, nothing changes (unifying the check does not change a single byte for valid
  // input)
  assertEquals([...gatherParams(12, 3, 9)], [12, 3, 9, GATHER_OOB_BITS]);
  // 2^32 turns into 0 in a Uint32Array (= every element counts as out of range and the output is
  // all NaN)
  assertThrows(() => gatherParams(U32_MAX + 1, 3, 9), CodegenError, "count");
});
