// i8（per-channel symmetric int8）格納の CPU 側展開・整列（format/i8.ts — ADR 0019）。
//
// これまで i8.ts は `gpu_i8_weights_test.ts` からしか踏まれず、アダプタ無しの環境では丸ごと
// SKIP されていた（3 兄弟のうち i4 だけが CPU 単体の門を持つ非対称）。多軸 keepdim scale の
// 桁上がり・broadcast の拒否・整列の詰めを GPU の有無から独立させるのがこのファイルの役目。
//
// MUST: 期待値は実装の走査式ではなく「軸ごとの座標 → scale の添字」を手で書き下したもので
// 持つ（同じ桁上がり式を写すと添字バグが両側で相殺する）。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { alignI8Payload, decodeI8, I8Error } from "../src/format/i8.ts";

/** 符号付き 8bit のバイト列（負値は 2 の補数）。 */
const i8Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Int8Array.from(values).buffer);

Deno.test("decodeI8: 3 軸の keepdim scale [2,1,1] は先頭軸で桁上がりする", () => {
  // shape [2,3,4] の 24 要素。q は位置ごとに違う値にして、添字がずれたら必ず値が動く形にする。
  const q = Array.from({ length: 24 }, (_, i) => i - 12);
  const scale = Float32Array.from([0.5, 0.25]);
  const decoded = decodeI8(i8Bytes(q), [2, 3, 4], scale, [2, 1, 1]);

  // 先頭 12 要素が scale[0]、後半 12 要素が scale[1]（軸 0 の 12 要素ごとに切り替わる）
  const expected = q.map((value, i) => Math.fround(value * (i < 12 ? 0.5 : 0.25)));
  assertEquals(decoded.length, 24);
  for (const [i, want] of expected.entries()) {
    assertStrictEquals(decoded[i], want, `要素 ${i}`);
  }
});

Deno.test("decodeI8: 中央軸の keepdim scale [1,3,1] は stride 0 の軸を跨いで配られる", () => {
  const q = Array.from({ length: 24 }, (_, i) => i - 12);
  const scale = Float32Array.from([1, 2, 4]);
  const decoded = decodeI8(i8Bytes(q), [2, 3, 4], scale, [1, 3, 1]);

  // 平坦添字 i の軸 1 の座標は (i / 4) % 3（軸 0 は長さ 1 なので stride 0 で寄与しない）
  for (const [i, value] of q.entries()) {
    assertStrictEquals(decoded[i], Math.fround(value * scale[Math.floor(i / 4) % 3]), `要素 ${i}`);
  }
});

Deno.test("decodeI8: 展開の丸めは f32 の 1 回だけ（GPU の f32 乗算と同値）", () => {
  // 0.1 は f32 で表せないので、f64 のまま掛けた値とは下位ビットが割れる
  const scale = Float32Array.from([0.1]);
  const decoded = decodeI8(i8Bytes([127]), [1, 1], scale, [1, 1]);
  assertStrictEquals(decoded[0], Math.fround(127 * scale[0]));
});

Deno.test("decodeI8: バイト長・rank・broadcast・scale 要素数の違反を拒否する", () => {
  const scale = Float32Array.from([1, 1]);
  // 実バイト数が shape の要素数と違う
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3]), [2, 2], Float32Array.from([1, 1]), [2, 1]),
    I8Error,
    "が shape",
  );
  // rank 違いの scaleShape（keepdim 形でない）
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], scale, [2]),
    I8Error,
    "keepdim 形が要る",
  );
  // 軸長が 1 でも同値でもない scaleShape
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], Float32Array.from([1, 1, 1]), [3, 1]),
    I8Error,
    "broadcast できない",
  );
  // scale の実要素数が scaleShape と合わない
  assertThrows(
    () => decodeI8(i8Bytes([1, 2, 3, 4]), [2, 2], Float32Array.from([1]), [2, 1]),
    I8Error,
    "scale の要素数",
  );
});

Deno.test("alignI8Payload: 端数 1 / 2 / 3 を 4 の倍数までゼロ詰めし、整列済みはコピーしない", () => {
  for (const [length, padded] of [[5, 8], [6, 8], [7, 8]] as const) {
    const bytes = new Uint8Array(Array.from({ length }, (_, i) => i + 1));
    const result = alignI8Payload(bytes);
    assertEquals(result.byteLength, padded, `${length} バイトの詰め先`);
    assertEquals([...result.subarray(0, length)], [...bytes], `${length} バイトの先頭`);
    assertEquals(
      [...result.subarray(length)].every((byte) => byte === 0),
      true,
      `${length} バイトの詰め物`,
    );
  }
  // 重みは GB 級なので、整列済みは**同一参照**で返る（無条件コピーは実測に響く）
  const aligned = new Uint8Array([1, 2, 3, 4]);
  assertStrictEquals(alignI8Payload(aligned), aligned);
});
