/**
 * The readers of the CPU decode read a view pointing into the middle of the source buffer
 * (byteOffset ≠ 0) the same as a tight copy.
 *
 * The hub's scan path returns a block as a view into the part's buffer (no copy — container-v1
 * §11). A reader that reads `bytes.buffer` without the byteOffset silently reads the head of the
 * buffer (the neighbouring block). The buffer's slack is filled with **poison (0xFF)** so that a
 * misaligned read does not happen to produce plausible values — 0xFF is NaN in f16 / f32, −1 in i8
 * and an edge-of-range code in i4 / i2.
 *
 * The view's byteOffset is placed at a multiple of 64 (blocks start 64 B aligned — enforced by the
 * descriptor parse). The scale's Float32Array view is built on this alignment (the same way as
 * `scaleTensor` in session-build).
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { decodeF16 } from "../src/format/f16.ts";
import { decodeI2 } from "../src/format/i2.ts";
import { decodeI4 } from "../src/format/i4.ts";
import { decodeI8 } from "../src/format/i8.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { f32Bytes } from "./helpers/model-fixture.ts";

const POISON = 0xff;
const BLOCK_ALIGN = 64;
/** Where the view is placed (64 B aligned like a block start, non-zero). */
const OFFSETS = [BLOCK_ALIGN, 3 * BLOCK_ALIGN] as const;

/**
 * Places `payload` at `offset` in a poison-filled buffer and returns a view of that range (poison
 * is left behind it too).
 */
const embed = (payload: Uint8Array<ArrayBuffer>, offset: number): Uint8Array<ArrayBuffer> => {
  const vessel = new Uint8Array(offset + payload.byteLength + BLOCK_ALIGN).fill(POISON);
  vessel.set(payload, offset);
  return vessel.subarray(offset, offset + payload.byteLength);
};

/** A scale view built the same way as `scaleTensor` (session-build.ts). */
const scaleView = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

const bitsOf = (values: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

const values = (count: number): number[] =>
  Array.from({ length: count }, (_, i) => Math.sin(i * 0.37) * 0.75);

describe("passing a view into the middle of a buffer to the readers of the CPU decode", () => {
  it("a scale Float32Array view reads from byteOffset (not the poison at the buffer head)", () => {
    const scale = Float32Array.of(0.5, -1.25, 3.0, 0.0078125);
    for (const offset of OFFSETS) {
      const view = scaleView(embed(f32Bytes(scale), offset));
      assertEquals(view.byteOffset, offset);
      assertEquals(bitsOf(view), bitsOf(scale), `the scale at offset ${offset} is corrupted`);
    }
  });

  it("decodeF16 produces the same bits for the view and a tight copy", () => {
    const quantized = quantizeF16(values(40));
    for (const offset of OFFSETS) {
      const view = embed(quantized.bytes, offset);
      assertNotEquals(view.byteOffset, 0);
      assertEquals(bitsOf(decodeF16(view)), bitsOf(decodeF16(view.slice())));
      assertEquals(bitsOf(decodeF16(view)), bitsOf(quantized.values));
    }
  });

  it("decodeI8 reads view weights and a view scale the same as a tight copy", () => {
    const shape = [6, 8];
    const quantized = quantizeI8(values(48), shape, 0);
    const scaleBytes = f32Bytes(quantized.scale);
    for (const offset of OFFSETS) {
      const view = embed(quantized.bytes, offset);
      const scale = scaleView(embed(scaleBytes, offset));
      const actual = decodeI8(view, shape, scale, quantized.scaleShape);
      const expected = decodeI8(
        view.slice(),
        shape,
        new Float32Array(quantized.scale),
        quantized.scaleShape,
      );
      assertEquals(bitsOf(actual), bitsOf(expected));
      assertEquals(bitsOf(actual), bitsOf(quantized.values));
    }
  });

  it("decodeI4 reads view weights and a view group scale the same as a tight copy", () => {
    const shape = [4, 32];
    const groupSize = 16;
    const quantized = quantizeI4(values(128), shape, groupSize);
    const scaleBytes = f32Bytes(quantized.scale);
    for (const offset of OFFSETS) {
      const view = embed(quantized.bytes, offset);
      const scale = scaleView(embed(scaleBytes, offset));
      const actual = decodeI4(view, shape, scale, quantized.scaleShape, groupSize);
      const expected = decodeI4(
        view.slice(),
        shape,
        new Float32Array(quantized.scale),
        quantized.scaleShape,
        groupSize,
      );
      assertEquals(bitsOf(actual), bitsOf(expected));
      assertEquals(bitsOf(actual), bitsOf(quantized.values));
    }
  });

  it("decodeI2 reads view weights and a view row scale the same as a tight copy", () => {
    const shape = [2, 16];
    const bytes = Uint8Array.of(0xe4, 0x1b, 0x72, 0x8d, 0x1b, 0xe4, 0x8d, 0x72);
    const scale = Float32Array.of(0.375, 1.25);
    const expected = decodeI2(bytes, shape, scale, [2, 1]);
    for (const offset of OFFSETS) {
      const view = embed(bytes, offset);
      const actual = decodeI2(view, shape, scaleView(embed(f32Bytes(scale), offset)), [2, 1]);
      assertEquals(bitsOf(actual), bitsOf(expected));
    }
  });
});
