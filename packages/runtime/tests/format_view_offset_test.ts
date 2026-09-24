/**
 * CPU 展開の読み手は、取得元の器の中程を指す view（byteOffset ≠ 0）を tight な写しと同じに読む。
 *
 * hub の scan 経路は block を part の器の view として返す（写さない — container-v1 §11）。
 * 読み手が `bytes.buffer` を byteOffset 抜きで読むと、器の先頭（隣の block）を黙って読む。
 * 器の余白を**毒（0xFF）**で埋めるのは、ずれた読みが「偶然それらしい値」にならないようにするため —
 * 0xFF は f16 / f32 で NaN、i8 で −1、i4 / i2 で範囲端のコードになる。
 *
 * view の byteOffset は 64 の倍数に置く（block 開始は 64 B 整列 — descriptor の parse が強制）。
 * scale の Float32Array view はこの整列の上で張る（session-build の `scaleTensor` と同じ組み方）。
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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
/** view を置く位置（block 開始と同じ 64 B 整列・0 以外）。 */
const OFFSETS = [BLOCK_ALIGN, 3 * BLOCK_ALIGN] as const;

/** 毒で埋めた器の `offset` に `payload` を置き、その区間の view を返す（後ろにも毒を残す）。 */
const embed = (payload: Uint8Array<ArrayBuffer>, offset: number): Uint8Array<ArrayBuffer> => {
  const vessel = new Uint8Array(offset + payload.byteLength + BLOCK_ALIGN).fill(POISON);
  vessel.set(payload, offset);
  return vessel.subarray(offset, offset + payload.byteLength);
};

/** `scaleTensor`（session-build.ts）と同じ組み方の scale view。 */
const scaleView = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

const bitsOf = (values: Float32Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

const values = (count: number): number[] =>
  Array.from({ length: count }, (_, i) => Math.sin(i * 0.37) * 0.75);

describe("CPU 展開の読み手に器の中程を指す view を渡す", () => {
  it("scale の Float32Array view は byteOffset から読む（器の先頭の毒を読まない）", () => {
    const scale = Float32Array.of(0.5, -1.25, 3.0, 0.0078125);
    for (const offset of OFFSETS) {
      const view = scaleView(embed(f32Bytes(scale), offset));
      assertEquals(view.byteOffset, offset);
      assertEquals(bitsOf(view), bitsOf(scale), `offset ${offset} の scale が化けている`);
    }
  });

  it("decodeF16 は view と tight な写しで同じビット列を出す", () => {
    const quantized = quantizeF16(values(40));
    for (const offset of OFFSETS) {
      const view = embed(quantized.bytes, offset);
      assertNotEquals(view.byteOffset, 0);
      assertEquals(bitsOf(decodeF16(view)), bitsOf(decodeF16(view.slice())));
      assertEquals(bitsOf(decodeF16(view)), bitsOf(quantized.values));
    }
  });

  it("decodeI8 は view の重みと view の scale を tight な写しと同じに読む", () => {
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

  it("decodeI4 は view の重みと view の group scale を tight な写しと同じに読む", () => {
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

  it("decodeI2 は view の重みと view の行 scale を tight な写しと同じに読む", () => {
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
