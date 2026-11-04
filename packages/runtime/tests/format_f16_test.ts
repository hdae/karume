// CPU-side decoding, rounding and alignment of f16 (IEEE 754 binary16) storage (format/f16.ts —
// ADR 0018 / 0028).
//
// Until now f16.ts was exercised only by `gpu_f16_i8_weights_test.ts` / `gpu_f16_compute_test.ts`
// and was skipped entirely on machines without an adapter (an asymmetry: of the three sibling
// modules only i4 had a CPU-only check). This file makes the pinning of the boundaries (±0 /
// subnormal / largest finite / ±Inf / NaN / rounding ties) independent of whether a GPU exists.
//
// MUST: expected values are written out from the IEEE 754 binary16 definition, not from the
// implementation's formulas (subnormals in steps of `2**-24`, 65504 as the largest finite value,
// 65520 as the overflow boundary). Deriving them from the implementation makes the test a
// tautology.

import { test } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { alignF16Payload, decodeF16, f16BitsToF32, roundToF16 } from "../src/format/f16.ts";

test("f16BitsToF32: exponent 0 is ±0 and subnormals (in steps of 2**-24)", () => {
  // ±0 keeps its sign (a form `Object.is` can tell apart = the sign of -1 * 0 is not lost)
  assertEquals(Object.is(f16BitsToF32(0x0000), 0), true);
  assertEquals(Object.is(f16BitsToF32(0x8000), -0), true);
  // Smallest and largest subnormal
  assertStrictEquals(f16BitsToF32(0x0001), 2 ** -24);
  assertStrictEquals(f16BitsToF32(0x03ff), 1023 * 2 ** -24);
  // Right above the subnormals is the smallest normal (the step size changes at the boundary)
  assertStrictEquals(f16BitsToF32(0x0400), 2 ** -14);
});

test("f16BitsToF32: exponent 31 is ±Inf and NaN, everything else is a known normal value", () => {
  assertStrictEquals(f16BitsToF32(0x7bff), 65504);
  assertStrictEquals(f16BitsToF32(0x7c00), Number.POSITIVE_INFINITY);
  assertStrictEquals(f16BitsToF32(0xfc00), Number.NEGATIVE_INFINITY);
  assertEquals(Number.isNaN(f16BitsToF32(0x7e00)), true);
  assertStrictEquals(f16BitsToF32(0x3c00), 1);
  assertStrictEquals(f16BitsToF32(0xc000), -2);
});

test("f16BitsToF32: the upper 16 bits are ignored (only the lower bits are read)", () => {
  assertStrictEquals(f16BitsToF32(0xffff0000 | 0x3c00), 1);
});

test("roundToF16: the overflow boundary and subnormal ties round to even", () => {
  // 65520, the midpoint of 65504 (largest finite) and 65536 (the next grid point), rounds up to Inf
  assertStrictEquals(roundToF16(65520), Number.POSITIVE_INFINITY);
  assertStrictEquals(roundToF16(65519.999), 65504);
  // The subnormal step is 2**-24. The midpoint of 0 and 1 step (0.5 step) goes to the even side = 0
  assertStrictEquals(roundToF16(2 ** -25), 0);
  // The midpoint of 1 and 2 steps (1.5 steps) also goes up to the even side = 2 steps (= 2**-23).
  // NOTE: the review's draft spec said 2**-24, but round-to-even lands on 2 steps (measured).
  assertStrictEquals(roundToF16(3 * 2 ** -25), 2 ** -23);
  // The midpoint of 2 and 3 steps (2.5 steps) goes down to the even side = 2 steps (a pair of ties
  // that round in opposite directions)
  assertStrictEquals(roundToF16(5 * 2 ** -25), 2 ** -23);
});

test("decodeF16: reads correctly from a view whose byteOffset is not 2-aligned", () => {
  // MUST detector: an implementation that creates a `Uint16Array` view throws RangeError here.
  const buffer = new ArrayBuffer(8);
  const bytes = new Uint8Array(buffer);
  // Put 0x3c00 (= 1) and 0xc000 (= −2) in little endian at a position shifted by one byte
  bytes.set([0x00, 0x3c, 0x00, 0xc0], 1);
  const decoded = decodeF16(new Uint8Array(buffer, 1, 4));
  assertEquals([...decoded], [1, -2]);
});

test("alignF16Payload: a 4-aligned payload is not copied, and a trailing 2 bytes are zero-padded", () => {
  const aligned = new Uint8Array([1, 2, 3, 4]);
  // Weights are GB-sized, so an aligned payload comes back as the **same reference** (an
  // unconditional copy shows up in measurements)
  assertStrictEquals(alignF16Payload(aligned), aligned);

  const odd = new Uint8Array([1, 2, 3, 4, 5, 6]);
  const padded = alignF16Payload(odd);
  assertEquals(padded.byteLength, 8);
  assertEquals([...padded], [1, 2, 3, 4, 5, 6, 0, 0]);
});

// The shapes decoding receives are narrowed by the safetensors declaration check (element count →
// byte length). This only checks that "bytes as declared give the right values" (f16 → f32 is
// always exact).
test("decodeF16: decodes a contiguous pattern into exactly the declared number of values", () => {
  const bytes = new Uint8Array([0x00, 0x00, 0x00, 0x80, 0x00, 0x3c, 0xff, 0x7b]);
  const decoded = decodeF16(bytes);
  assertEquals(decoded.length, 4);
  assertEquals(Object.is(decoded[0], 0), true);
  assertEquals(Object.is(decoded[1], -0), true);
  assertEquals([decoded[2], decoded[3]], [1, 65504]);
});

test("decodeF16: an odd byte count fails when the last read runs past the end (a RangeError of unclear origin)", () => {
  // Today this shape is unreachable because the safetensors declaration check (the bit-boundary
  // check) blocks it upstream. This records that f16 has no dedicated Error (i4 / i8 do — an
  // asymmetry in the defences) and that the failure is a diagnostic of unclear origin, "DataView
  // out of range".
  // NOTE: the review's draft spec said "building the TypedArray fails on a non-integer length", but
  // `new Float32Array(1.5)` truncates to length 1 (measured) — what actually fails is the read.
  assertThrows(
    () => decodeF16(new Uint8Array([1, 2, 3])),
    RangeError,
    "outside the bounds of the DataView",
  );
});
