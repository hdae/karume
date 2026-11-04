// CPU-side decoding and alignment of i8 (per-channel symmetric int8) storage (format/i8.ts — ADR
// 0019).
//
// Until now i8.ts was exercised only by `gpu_f16_i8_weights_test.ts` and was skipped entirely on
// machines without an adapter (an asymmetry: of the three sibling modules only i4 had a CPU-only
// check). This file makes the carry of multi-axis keepdim scales, the rejection of broadcasts and
// the alignment padding independent of whether a GPU exists.
//
// MUST: expected values are kept as hand-written "per-axis coordinate → scale index" mappings, not
// as the implementation's traversal formula (copying the same carry formula would let an index bug
// cancel out on both sides).

import { test } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { alignI8Payload, decodeI8, I8Error } from "../src/format/i8.ts";

/** Bytes of signed 8-bit values (negative values in two's complement). */
const i8Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Int8Array.from(values).buffer);

test("decodeI8: a 3-axis keepdim scale [2,1,1] carries on the leading axis", () => {
  // The 24 elements of shape [2,3,4]. q differs at every position so that a shifted index always
  // moves a value.
  const q = Array.from({ length: 24 }, (_, i) => i - 12);
  const scale = Float32Array.from([0.5, 0.25]);
  const decoded = decodeI8(i8Bytes(q), [2, 3, 4], scale, [2, 1, 1]);

  // The first 12 elements use scale[0], the last 12 use scale[1] (it switches every 12 elements of
  // axis 0)
  const expected = q.map((value, i) => Math.fround(value * (i < 12 ? 0.5 : 0.25)));
  assertEquals(decoded.length, 24);
  for (const [i, want] of expected.entries()) {
    assertStrictEquals(decoded[i], want, `element ${i}`);
  }
});

test("decodeI8: a middle-axis keepdim scale [1,3,1] is handed out across a stride-0 axis", () => {
  const q = Array.from({ length: 24 }, (_, i) => i - 12);
  const scale = Float32Array.from([1, 2, 4]);
  const decoded = decodeI8(i8Bytes(q), [2, 3, 4], scale, [1, 3, 1]);

  // The axis-1 coordinate of flat index i is (i / 4) % 3 (axis 0 has length 1, so its stride 0
  // contributes nothing)
  for (const [i, value] of q.entries()) {
    assertStrictEquals(
      decoded[i],
      Math.fround(value * scale[Math.floor(i / 4) % 3]),
      `element ${i}`,
    );
  }
});

test("decodeI8: decoding rounds to f32 exactly once (equal to the GPU's f32 multiply)", () => {
  // 0.1 is not representable in f32, so the low bits differ from the value multiplied in f64
  const scale = Float32Array.from([0.1]);
  const decoded = decodeI8(i8Bytes([127]), [1, 1], scale, [1, 1]);
  assertStrictEquals(decoded[0], Math.fround(127 * scale[0]));
});

test("decodeI8: rejects violations of byte length, rank, broadcast and scale element count", () => {
  const scale = Float32Array.from([1, 1]);
  // The actual byte count differs from the shape's element count
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3]), [2, 2], Float32Array.from([1, 1]), [2, 1]),
    I8Error,
    "does not match shape",
  );
  // A scaleShape of a different rank (not a keepdim shape)
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], scale, [2]),
    I8Error,
    "a keepdim shape is required",
  );
  // A scaleShape whose axis length is neither 1 nor equal
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], Float32Array.from([1, 1, 1]), [3, 1]),
    I8Error,
    "cannot be broadcast",
  );
  // The actual element count of scale does not match scaleShape
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], Float32Array.from([1]), [2, 1]),
    I8Error,
    "scale element count",
  );
});

test("alignI8Payload: zero-pads a remainder of 1 / 2 / 3 up to a multiple of 4 and does not copy an aligned payload", () => {
  for (const [length, padded] of [
    [5, 8],
    [6, 8],
    [7, 8],
  ] as const) {
    const bytes = new Uint8Array(Array.from({ length }, (_, i) => i + 1));
    const result = alignI8Payload(bytes);
    assertEquals(result.byteLength, padded, `padded size of ${length} bytes`);
    assertEquals([...result.subarray(0, length)], [...bytes], `head of ${length} bytes`);
    assertEquals(
      [...result.subarray(length)].every((byte) => byte === 0),
      true,
      `padding of ${length} bytes`,
    );
  }
  // Weights are GB-sized, so an aligned payload comes back as the **same reference** (an
  // unconditional copy shows up in measurements)
  const aligned = new Uint8Array([1, 2, 3, 4]);
  assertStrictEquals(alignI8Payload(aligned), aligned);
});
