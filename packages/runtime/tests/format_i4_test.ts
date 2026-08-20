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

Deno.test("decodeI4: 語彙表 [V,D] は適格外でもそのまま展開できる（group 32・行 × group 順）", () => {
  // embedding の重み `[V,D]` は `channel_rows` が恒等なので、group は D 軸に沿って切られる
  // （ADR 0069 決定 5 の embedding 追補）。適格外の i4 embedding はこの展開器が受け皿になる。
  // V = 3 / D = 64 / group 32 → group 形 [3, 2]。q は行と列で変える（行 / group の取り違えが
  // 値に出る形）。
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

Deno.test("decodeI4: conv1d の rank 3 重みは rank 2 scale で展開できる（行 = Cout・行長 = Cin·K）", () => {
  // 波 J-5b: 適格 op の `channel_rows` はどれも「先頭次元 = 行・残りを平坦化」で、scale は
  // rank に依らず rank 2 `[shape[0], (numel / shape[0]) / g]`。`[2,4,4]`（行長 16）× g8 →
  // scale `[2,2]`。nibble の並びは平坦メモリ順（`[O,Cin,K]` row-major = `[O, Cin·K]` 平坦）
  // なので、行 / group の取り違えが scale の値差でそのまま出る形にする。
  const shape = [2, 4, 4];
  const groupSize = 8;
  const count = 2 * 4 * 4;
  const q = Array.from({ length: count }, (_, i) => ((i % 15) - 7));
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
  // 旧規則の形（重みと同 rank・最終次元だけ group 数 = `[2,4,1]`）は受理しない
  assertThrows(
    () => decodeI4(bytes, shape, Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1]), [2, 4, 1], 4),
    I4Error,
    "group 形",
  );
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
