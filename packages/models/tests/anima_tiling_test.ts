// タイル decode の幾何と貼り合わせ（VAE tiling — ADR 0033）。
//
// 実 GPU / 実重みでの参照突合は E2E（P3 波 2）。ここは GPU も資産も要らない純関数側で、
// **幾何の不変条件**（固定形の decoder が食える配置であること）と**貼り合わせの解析解**を
// 押さえる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assembleTiles,
  blendExtent,
  decodeTiled,
  latentTile,
  planTileAxis,
  planVaeTiling,
  tileCount,
} from "../src/anima/tiling.ts";

Deno.test("planTileAxis: 1024px（latent 128）は 3 タイル・stride 32・重なり latent 32", () => {
  const axis = planTileAxis(128, 64, 8);
  assertEquals([...axis.starts], [0, 32, 64]);
  assertEquals(axis.stride, 32);
  assertEquals(blendExtent(axis, 8), 256);
});

Deno.test("planTileAxis: latent 64（512px）は 1 タイルに縮退しブレンド幅 0", () => {
  const axis = planTileAxis(64, 64, 8);
  assertEquals([...axis.starts], [0]);
  assertEquals(axis.stride, 64);
  assertEquals(blendExtent(axis, 8), 0);
});

Deno.test("planTileAxis: どの latent 全長でも配置の不変条件が成り立つ", () => {
  for (const extent of [64, 80, 96, 112, 128, 160, 192, 256, 384, 512]) {
    const axis = planTileAxis(extent, 64, 8);
    const where = `extent=${extent}`;
    assertEquals(axis.starts[0], 0, where);
    // 末端へのスナップ = 固定形の decoder が最後のタイルも食えることの条件。
    assertEquals(axis.starts.at(-1), extent - 64, `${where}: 末端へスナップ`);
    assertEquals((extent - 64) % axis.stride, 0, `${where}: stride の割り切れ`);
    for (let index = 1; index < axis.starts.length; index += 1) {
      assertEquals(axis.starts[index] - axis.starts[index - 1], axis.stride, `${where}: 等間隔`);
    }
    assert(
      axis.starts.length === 1 || axis.tile - axis.stride >= 8,
      `${where}: 重なりが下限 8 未満`,
    );
    // 本数は最小（1 本減らせるなら下限を割るか割り切れない）。安全側に倒して常に
    // stride 1 にする実装は、この 1 本が無いと緑のまま通る。
    for (let count = 2; count < axis.starts.length; count += 1) {
      const span = extent - 64;
      assert(
        span % (count - 1) !== 0 || 64 - span / (count - 1) < 8,
        `${where}: ${count} 本で足りるのに ${axis.starts.length} 本`,
      );
    }
  }
});

Deno.test("planTileAxis: タイルより短い latent と広すぎる重なりは落とす", () => {
  assertThrows(() => planTileAxis(32, 64, 8), Error, "タイル幅");
  assertThrows(() => planTileAxis(128, 64, 64), Error, "重なり");
});

Deno.test("planVaeTiling: 縮尺は資産の入出力形から割り出し、軸は独立に配置する", () => {
  const geometry = planVaeTiling([1, 16, 128, 64], [1, 16, 64, 64], [1, 3, 512, 512]);
  assertEquals(geometry.scale, 8);
  assertEquals(geometry.channels, 16);
  assertEquals(geometry.rows.starts.length, 3);
  assertEquals(geometry.cols.starts.length, 1, "H = W でなくても軸ごとに独立");
  assertEquals(tileCount(geometry), 3);
  // 入口の受理集合（チャネル不一致 / 軸で揃わない縮尺 / B>1）。
  assertThrows(
    () => planVaeTiling([1, 8, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "チャネル数",
  );
  assertThrows(
    () => planVaeTiling([1, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 256]),
    Error,
    "縮尺",
  );
  assertThrows(
    () => planVaeTiling([2, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "batch=1",
  );
});

Deno.test("latentTile: 平面ごとに正しい矩形を切り出す（軸とチャネルの取り違え検出）", () => {
  // [1,2,4,4]。値は `channel*100 + y*10 + x` で全要素が識別可能。
  const geometry = planVaeTiling([1, 2, 4, 4], [1, 2, 2, 2], [1, 1, 2, 2], 1);
  assertEquals([...geometry.rows.starts], [0, 1, 2]);
  const latents = new Float32Array(2 * 16);
  for (let channel = 0; channel < 2; channel += 1) {
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) latents[channel * 16 + y * 4 + x] = channel * 100 + y * 10 + x;
    }
  }
  assertEquals([...latentTile(latents, geometry, 0, 0)], [0, 1, 10, 11, 100, 101, 110, 111]);
  assertEquals([...latentTile(latents, geometry, 2, 1)], [21, 22, 31, 32, 121, 122, 131, 132]);
  assertThrows(() => latentTile(latents, geometry, 3, 0), RangeError);
});

Deno.test("assembleTiles: 1 枚（縮退）は decode 出力の素の写し（ビット同一）", () => {
  const geometry = planVaeTiling([1, 2, 8, 8], [1, 2, 8, 8], [1, 3, 8, 8], 2);
  assertEquals(tileCount(geometry), 1);
  const decoded = new Float32Array(3 * 64);
  for (let index = 0; index < decoded.length; index += 1) decoded[index] = Math.sin(index) * 3;
  const assembled = assembleTiles([decoded], geometry);
  assertEquals(
    new Uint32Array(assembled.buffer, assembled.byteOffset, assembled.length),
    new Uint32Array(decoded.buffer, decoded.byteOffset, decoded.length),
  );
});

Deno.test("assembleTiles: 位置依存の decode 出力を線形ランプが解析解どおりに畳む", () => {
  // extent 16 / tile 8 / stride 4（重なり 4）で開始位置は 0,4,8。タイル内の値が行番号
  // `0..7` の傾斜のとき、貼り合わせ後は `0,1,2,3` → 中間は全て 4 → 末尾が `4,5,6,7`。
  // 重なりの値がタイルごとに**違う**ので、ブレンドの向き反転 / stride の off-by-one /
  // 末端タイルのスナップ落とし / 担当領域の取り違えが全てここで割れる（同じ値が重なる
  // 作りだと向きを反転しても緑のまま通る）。
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  assertEquals([...geometry.rows.starts], [0, 4, 8]);
  const rowRamp = new Float32Array(64);
  const colRamp = new Float32Array(64);
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      rowRamp[y * 8 + x] = y;
      colRamp[y * 8 + x] = x;
    }
  }
  const expected = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7];
  const rows = assembleTiles(Array.from({ length: 9 }, () => rowRamp), geometry);
  assertEquals([...Array.from({ length: 16 }, (_, y) => rows[y * 16])], expected, "行方向");
  const cols = assembleTiles(Array.from({ length: 9 }, () => colRamp), geometry);
  assertEquals([...cols.subarray(0, 16)], expected, "列方向");
});

Deno.test("assembleTiles: 末端まで覆う（stride 幅で切り詰めるだけだと欠ける）", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64).fill(7);
  const image = assembleTiles(Array.from({ length: 9 }, () => flat), geometry);
  assertEquals(image.length, 16 * 16);
  assert(image.every((value) => value === 7), "覆われていない画素が残っている");
});

Deno.test("assembleTiles: 渡されたタイルを破壊しない（in-place ブレンドは写しの上）", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const tiles = Array.from({ length: 9 }, (_, index) => new Float32Array(64).fill(index + 1));
  const before = tiles.map((tile) => [...tile]);
  assembleTiles(tiles, geometry);
  assertEquals(tiles.map((tile) => [...tile]), before);
});

Deno.test("assembleTiles: タイル枚数と要素数の食い違いを落とす", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64);
  assertThrows(() => assembleTiles([flat], geometry), Error, "幾何の");
  assertThrows(
    () =>
      assembleTiles(
        [...Array.from({ length: 8 }, () => flat), new Float32Array(63)],
        geometry,
      ),
    Error,
    "1 枚目",
  );
});

Deno.test("decodeTiled: 行優先で全タイルを 1 度ずつ回す", async () => {
  const geometry = planVaeTiling([1, 1, 16, 16], [1, 1, 8, 8], [1, 1, 8, 8], 2);
  const visited: string[] = [];
  await decodeTiled(new Float32Array(256), geometry, (_tile, row, col) => {
    visited.push(`${row},${col}`);
    return Promise.resolve(new Float32Array(64));
  });
  assertEquals(visited, ["0,0", "0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2"]);
});
