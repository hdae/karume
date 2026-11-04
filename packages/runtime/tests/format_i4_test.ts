// CPU-side decoding of i4 (group-symmetric packed 4-bit along K) (format/i4.ts — ADR 0069).
//
// The expected bytes of the pack order are **written out by hand** (the spec of the exporter's
// `karume/emit.py: pack_int4` — the same value as `q=[1,2] → 0xA9` in
// tools/exporter/tests/test_emit.py pins both languages as a pair). Deriving the expected values
// from the implementation makes the round trip a tautology.

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { decodeI4, I4Error } from "../src/format/i4.ts";

test("decodeI4: pack order is element 2i = low nibble / 2i+1 = high nibble, u = q + 8", () => {
  // q = [1, 2] → u = [9, 10] → one byte 0xA9 (reading the nibbles swapped gives [2, 1]).
  // The same value as TestPackedFourBitOrder in test_emit.py on the Python side — pins the packing
  // correspondence between the two languages.
  const decoded = decodeI4(Uint8Array.from([0xa9]), [1, 2], Float32Array.from([1]), [1, 1], 2);
  assertEquals([...decoded], [1, 2]);
});

test("decodeI4: positions match exactly for an asymmetric pattern where all neighbours differ", () => {
  // [1,-2,3,-4,...] (within ±7) — with a symmetric pattern, swapped nibbles would not show in the
  // values (the CPU side of ADR 0069 decision 4 ①).
  const q = Array.from({ length: 16 }, (_, i) => ((i % 7) + 1) * (i % 2 === 0 ? 1 : -1));
  const bytes = new Uint8Array(8);
  for (const [i, value] of q.entries()) {
    const u = value + 8;
    bytes[i >> 1] |= (i & 1) === 1 ? u << 4 : u;
  }
  const decoded = decodeI4(bytes, [1, 16], Float32Array.from([0.5]), [1, 1], 16);
  assertEquals(
    [...decoded],
    q.map((value) => value * 0.5),
  );
});

test("decodeI4: scale is looked up per group (flat row × group order)", () => {
  // [2, 8], group 4 → group shape [2, 2]. Every element is q = 1 (u = 9 = 0x99), so value
  // differences come only from the scale index — an index error (rows and groups swapped) goes red
  // directly.
  const bytes = new Uint8Array(8).fill(0x99);
  const scale = Float32Array.from([1, 2, 4, 8]);
  const decoded = decodeI4(bytes, [2, 8], scale, [2, 2], 4);
  assertEquals([...decoded], [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8]);
});

test("decodeI4: an embedding table [V,D] decodes as is even when not eligible (group 32, row × group order)", () => {
  // The embedding weight `[V,D]` has an identity `channel_rows`, so groups are cut along the D axis
  // (the embedding addendum to ADR 0069 decision 5). This decoder is what handles an ineligible i4
  // embedding. V = 3 / D = 64 / group 32 → group shape [3, 2]. q varies by row and column (a form
  // where swapping rows / groups shows in the values).
  const vocab = 3;
  const hidden = 64;
  const groupSize = 32;
  const q = Array.from(
    { length: vocab * hidden },
    (_, i) => ((Math.floor(i / hidden) + i) % 15) - 7,
  );
  const bytes = new Uint8Array(q.length / 2);
  for (const [i, value] of q.entries()) {
    const u = value + 8;
    bytes[i >> 1] |= (i & 1) === 1 ? u << 4 : u;
  }
  const scale = Float32Array.from([0.5, 0.25, 1, 2, 4, 8]);
  const decoded = decodeI4(bytes, [vocab, hidden], scale, [vocab, 2], groupSize);

  assertEquals(decoded.length, vocab * hidden);
  const expected = q.map((value, i) => {
    const row = Math.floor(i / hidden);
    const group = Math.floor((i % hidden) / groupSize);
    return Math.fround(value * scale[row * 2 + group]);
  });
  assertEquals([...decoded], expected);
});

test("decodeI4: a rank-3 conv1d weight decodes with a rank-2 scale (row = Cout, row length = Cin·K)", () => {
  // Step J-5b: every eligible op's `channel_rows` is "leading dimension = row, the rest flattened",
  // and the scale is rank 2 `[shape[0], (numel / shape[0]) / g]` whatever the rank. `[2,4,4]` (row
  // length 16) × g8 → scale `[2,2]`. Nibbles are in flat memory order (`[O,Cin,K]` row-major =
  // `[O, Cin·K]` flat), so the form is chosen so that swapping rows / groups shows directly as a
  // difference in scale values.
  const shape = [2, 4, 4];
  const groupSize = 8;
  const count = 2 * 4 * 4;
  const q = Array.from({ length: count }, (_, i) => (i % 15) - 7);
  const bytes = new Uint8Array(count / 2);
  for (const [i, value] of q.entries()) {
    const u = value + 8;
    bytes[i >> 1] |= (i & 1) === 1 ? u << 4 : u;
  }
  const scale = Float32Array.from([0.5, 0.25, 1, 2]);
  const decoded = decodeI4(bytes, shape, scale, [2, 2], groupSize);
  const expected = q.map((value, i) => {
    const row = Math.floor(i / 16);
    const group = Math.floor((i % 16) / groupSize);
    return Math.fround(value * scale[row * 2 + group]);
  });
  assertEquals([...decoded], expected);
  // The old rule's shape (same rank as the weight, only the last dimension is the group count =
  // `[2,4,1]`) is not accepted
  assertThrows(
    () => decodeI4(bytes, shape, Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1]), [2, 4, 1], 4),
    I4Error,
    "group shape",
  );
});

test("decodeI4: decoding rounds to f32 exactly once (equal to the GPU's f32 multiply)", () => {
  // 0.1 is a value that f32 rounds (the same argument as the matching test for format/i8.ts — q is
  // an integer and exact in f32, and the product is exact in f64, so the single rounding matches
  // the correct rounding of an f32 multiply).
  const scale = Float32Array.from([0.1]);
  // q = [7, -7] → u = [15, 1] → 0x1F
  const decoded = decodeI4(Uint8Array.from([0x1f]), [1, 2], scale, [1, 1], 2);
  assertEquals(decoded[0], Math.fround(7 * scale[0]));
  assertEquals(decoded[1], Math.fround(-7 * scale[0]));
});

test("decodeI4: rejects violations of byte length, group-shaped scale and divisibility", () => {
  const bytes = new Uint8Array(8);
  // byte length ≠ numel / 2
  assertThrows(
    () => decodeI4(new Uint8Array(7), [2, 8], Float32Array.from([1, 1]), [2, 1], 8),
    I4Error,
    "bytes",
  );
  // The keepdim broadcast shape (the i8 shape) is rejected as a group shape — letting it through
  // silently would hand out the group scales as one value per channel
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1, 1]), [2, 1], 4),
    I4Error,
    "group shape",
  );
  // The group does not divide the quantization axis
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1, 1]), [2, 1], 3),
    I4Error,
    "does not divide",
  );
  // The element count of the scale data does not match its shape
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1]), [2, 2], 4),
    I4Error,
    "element count",
  );
  // rank 0 (no quantization axis)
  assertThrows(
    () => decodeI4(new Uint8Array(0), [], Float32Array.from([]), [], 16),
    I4Error,
    "rank 0",
  );
});
