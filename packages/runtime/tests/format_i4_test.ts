// i4（K 方向 group symmetric packed 4bit）の CPU 側展開（format/i4.ts — ADR 0069）。
//
// pack 順の期待バイトは**手で書き下す**（エクスポータ `karume/emit.py: pack_int4` の仕様 —
// tests 側 tools/exporter/tests/test_emit.py の `q=[1,2] → 0xA9` と同じ値で両言語を対に固定
// する）。実装から期待値を引くと往復が恒真化する。

import { assertEquals, assertThrows } from "@std/assert";
import { decodeI4, I4Error } from "../src/format/i4.ts";

Deno.test("decodeI4: pack 順は要素 2i = 下位 nibble / 2i+1 = 上位・u = q + 8", () => {
  // q = [1, 2] → u = [9, 10] → 1 バイト 0xA9（上下を逆に読むと [2, 1] になる）。
  // Python 側 test_emit.py の TestPackedFourBitOrder と同じ値 — 両言語の pack 対応を固定する。
  const decoded = decodeI4(Uint8Array.from([0xa9]), [1, 2], Float32Array.from([1]), [1, 1], 2);
  assertEquals([...decoded], [1, 2]);
});

Deno.test("decodeI4: 隣接要素が全て異なる非対称パターンの位置が完全一致する", () => {
  // [1,-2,3,-4,...]（±7 内）— 対称なパターンだと nibble の取り違えが値に出ない
  // （ADR 0069 決定 4 ① の CPU 側）。
  const q = Array.from({ length: 16 }, (_, i) => ((i % 7) + 1) * (i % 2 === 0 ? 1 : -1));
  const bytes = new Uint8Array(8);
  for (const [i, value] of q.entries()) {
    const u = value + 8;
    bytes[i >> 1] |= (i & 1) === 1 ? u << 4 : u;
  }
  const decoded = decodeI4(bytes, [1, 16], Float32Array.from([0.5]), [1, 1], 16);
  assertEquals([...decoded], q.map((value) => value * 0.5));
});

Deno.test("decodeI4: scale は group ごとに引かれる（行 × group の平坦順）", () => {
  // [2, 8]・group 4 → group 形 [2, 2]。全要素 q = 1（u = 9 = 0x99）で、値の違いは
  // scale の添字だけから出る — 添字の誤り（行と group の取り違え）が直接赤になる。
  const bytes = new Uint8Array(8).fill(0x99);
  const scale = Float32Array.from([1, 2, 4, 8]);
  const decoded = decodeI4(bytes, [2, 8], scale, [2, 2], 4);
  assertEquals([...decoded], [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8]);
});

Deno.test("decodeI4: 展開の丸めは f32 の 1 回だけ（GPU の f32 乗算と同値）", () => {
  // 0.1 は f32 で丸められる値（format/i8.ts の同型テストと同じ論証 — q は整数で f32 厳密、
  // 積は f64 で厳密なので 1 回丸めが f32 乗算の正しい丸めと一致する）。
  const scale = Float32Array.from([0.1]);
  // q = [7, -7] → u = [15, 1] → 0x1F
  const decoded = decodeI4(Uint8Array.from([0x1f]), [1, 2], scale, [1, 1], 2);
  assertEquals(decoded[0], Math.fround(7 * scale[0]));
  assertEquals(decoded[1], Math.fround(-7 * scale[0]));
});

Deno.test("decodeI4: バイト長・group 形 scale・整除の違反を拒否する", () => {
  const bytes = new Uint8Array(8);
  // バイト長 ≠ numel / 2
  assertThrows(
    () => decodeI4(new Uint8Array(7), [2, 8], Float32Array.from([1, 1]), [2, 1], 8),
    I4Error,
    "バイト",
  );
  // keepdim broadcast 形（i8 の形）は group 形として拒否 — 黙って通すと group scale が
  // 1 チャネル 1 値として配られる
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1, 1]), [2, 1], 4),
    I4Error,
    "group 形",
  );
  // group が量子化軸を割り切らない
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1, 1]), [2, 1], 3),
    I4Error,
    "割り切らない",
  );
  // scale 実体の要素数が shape と合わない
  assertThrows(
    () => decodeI4(bytes, [2, 8], Float32Array.from([1]), [2, 2], 4),
    I4Error,
    "要素数",
  );
  // rank 0（量子化軸が無い）
  assertThrows(
    () => decodeI4(new Uint8Array(0), [], Float32Array.from([]), [], 16),
    I4Error,
    "rank 0",
  );
});
