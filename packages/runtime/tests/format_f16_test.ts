// f16（IEEE 754 binary16）格納の CPU 側展開・丸め・整列（format/f16.ts — ADR 0018 / 0028）。
//
// これまで f16.ts は `gpu_f16_weights_test.ts` / `gpu_f16_compute_test.ts` からしか踏まれず、
// アダプタ無しの環境では丸ごと SKIP されていた（3 兄弟のうち i4 だけが CPU 単体の門を持つ
// 非対称）。境界（±0 / subnormal / 最大有限 / ±Inf / NaN / 丸めの同点）の固定を GPU の有無から
// 独立させるのがこのファイルの役目。
//
// MUST: 期待値は実装式ではなく IEEE 754 binary16 の定義から書き下す（`2**-24` 刻みの
// subnormal・65504 の最大有限・65520 のオーバフロー境界）。実装から引くと恒真になる。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { alignF16Payload, decodeF16, f16BitsToF32, roundToF16 } from "../src/format/f16.ts";

Deno.test("f16BitsToF32: 指数 0 は ±0 と subnormal（2**-24 刻み）", () => {
  // ±0 は符号まで保つ（`Object.is` で区別できる形 = -1 * 0 の符号が消えていない）
  assertEquals(Object.is(f16BitsToF32(0x0000), 0), true);
  assertEquals(Object.is(f16BitsToF32(0x8000), -0), true);
  // 最小 subnormal と最大 subnormal
  assertStrictEquals(f16BitsToF32(0x0001), 2 ** -24);
  assertStrictEquals(f16BitsToF32(0x03ff), 1023 * 2 ** -24);
  // subnormal の直上が最小 normal（境界で刻みが切り替わる）
  assertStrictEquals(f16BitsToF32(0x0400), 2 ** -14);
});

Deno.test("f16BitsToF32: 指数 31 は ±Inf と NaN、それ以外は normal の既知値", () => {
  assertStrictEquals(f16BitsToF32(0x7bff), 65504);
  assertStrictEquals(f16BitsToF32(0x7c00), Number.POSITIVE_INFINITY);
  assertStrictEquals(f16BitsToF32(0xfc00), Number.NEGATIVE_INFINITY);
  assertEquals(Number.isNaN(f16BitsToF32(0x7e00)), true);
  assertStrictEquals(f16BitsToF32(0x3c00), 1);
  assertStrictEquals(f16BitsToF32(0xc000), -2);
});

Deno.test("f16BitsToF32: 上位 16bit は無視する（下位だけを見る）", () => {
  assertStrictEquals(f16BitsToF32(0xffff0000 | 0x3c00), 1);
});

Deno.test("roundToF16: オーバフロー境界と subnormal の同点は偶数丸め", () => {
  // 65504（最大有限）と 65536（次の格子）の中点 65520 は上へ倒れて Inf になる
  assertStrictEquals(roundToF16(65520), Number.POSITIVE_INFINITY);
  assertStrictEquals(roundToF16(65519.999), 65504);
  // subnormal の刻みは 2**-24。0 と 1 刻みの中点（0.5 刻み）は偶数側 = 0 へ
  assertStrictEquals(roundToF16(2 ** -25), 0);
  // 1 刻みと 2 刻みの中点（1.5 刻み）も偶数側 = 2 刻み（= 2**-23）へ上がる。
  // NOTE: レビューの仕様案は 2**-24 と書いていたが、偶数丸めの行き先は 2 刻み側（実測）。
  assertStrictEquals(roundToF16(3 * 2 ** -25), 2 ** -23);
  // 2 刻みと 3 刻みの中点（2.5 刻み）は下の偶数側 = 2 刻みへ（同点が両方向へ倒れる対）
  assertStrictEquals(roundToF16(5 * 2 ** -25), 2 ** -23);
});

Deno.test("decodeF16: byteOffset が 2 整列していない view でも正しく読める", () => {
  // MUST の検出器: `Uint16Array` の view を張る実装はここで RangeError になる。
  const buffer = new ArrayBuffer(8);
  const bytes = new Uint8Array(buffer);
  // 1 バイトずらした位置に 0x3c00（= 1）と 0xc000（= −2）をリトルエンディアンで置く
  bytes.set([0x00, 0x3c, 0x00, 0xc0], 1);
  const decoded = decodeF16(new Uint8Array(buffer, 1, 4));
  assertEquals([...decoded], [1, -2]);
});

Deno.test("alignF16Payload: 4 整列済みはコピーせず、端数 2 バイトはゼロ詰めする", () => {
  const aligned = new Uint8Array([1, 2, 3, 4]);
  // 重みは GB 級なので、整列済みは**同一参照**で返る（無条件コピーは実測に響く）
  assertStrictEquals(alignF16Payload(aligned), aligned);

  const odd = new Uint8Array([1, 2, 3, 4, 5, 6]);
  const padded = alignF16Payload(odd);
  assertEquals(padded.byteLength, 8);
  assertEquals([...padded], [1, 2, 3, 4, 5, 6, 0, 0]);
});

// 展開が受ける形は safetensors の宣言検査が絞っている（要素数 → バイト長）。ここは
// 「宣言どおりのバイト列なら値が合う」ことだけを見る（f16 → f32 は常に厳密）。
Deno.test("decodeF16: 連続したパターンを宣言どおりの本数だけ展開する", () => {
  const bytes = new Uint8Array([0x00, 0x00, 0x00, 0x80, 0x00, 0x3c, 0xff, 0x7b]);
  const decoded = decodeF16(bytes);
  assertEquals(decoded.length, 4);
  assertEquals(Object.is(decoded[0], 0), true);
  assertEquals(Object.is(decoded[1], -0), true);
  assertEquals([decoded[2], decoded[3]], [1, 65504]);
});

Deno.test("decodeF16: 奇数バイトは末尾の読み出しがはみ出して落ちる（出所不明の RangeError）", () => {
  // 現状は safetensors の宣言検査（bit 境界門）が上流で塞いでいるので到達しない形。専用 Error
  // を持たない（i4 / i8 は持つ — 防御の非対称）ことと、落ち方が「DataView が範囲外」という
  // 出所の分からない診断になることを記録しておく。
  // NOTE: レビューの仕様案は「非整数長で TypedArray 構築が落ちる」としていたが、
  // `new Float32Array(1.5)` は長さ 1 へ切り捨てられる（実測）— 実際に落ちるのは読み出し側。
  assertThrows(
    () => decodeF16(new Uint8Array([1, 2, 3])),
    RangeError,
    "outside the bounds of the DataView",
  );
});
