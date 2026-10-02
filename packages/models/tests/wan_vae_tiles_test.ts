// Wan2.1 の動画 VAE のタイル decode のホスト側（ADR 0118 決定 2・段 5 — GPU も資産も要らない純関数）。
//
// 実 GPU での参照照合と縮退門は `e2e_wan_vae_tiles_test.ts`。ここは**幾何の凍結表**（Python 側と
// 同じ値 — ADR 0033 追記 9a の二重凍結）・配置の不変条件・貼り合わせの解析解を押さえる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assembleWanVaeTiles,
  clampWanVaeFrames,
  planWanVaeTileAxis,
  planWanVaeTiles,
  WAN_VAE_MIN_TILE_OVERLAP,
  wanVaeBlendExtentAt,
  wanVaeLatentTile,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../src/wan/vae-tiles.ts";

/** 実寸のタイル辺（潜在 — chunk グラフの既定の入力形）と縮尺。 */
const TILE = 32;
const SCALE = 8;
const LAYOUT = { latentChannels: 16, tile: TILE, sampleTile: TILE * SCALE };

/**
 * タイル辺 32・重なりの下限 8 の開始位置（潜在の全長 → 開始位置）。Python 側
 * （`tools/export-recipes/wan/tests/test_vae_tiling.py` の `MIRRORED_STARTS`）と**同じ値**で凍結する。
 *
 * - 32: 縮退（1 枚 — 縮退門の形）
 * - 60 / 104: 受理集合（832×480 / 480×832 — ADR 0118 決定 7）の 2 辺。ADR の値そのもの
 * - 57: `i·span/(本数−1)` がちょうど半分になる辺（span 25 / 3 本 → 12.5）。0.5 の寄せ方が
 *   切り上げでなくなる退行（`Math.floor` や偶数丸め）はこの 1 行だけが捕まえる — 下の不変条件
 *   ループは寄せ方を決めない
 */
const AXIS_STARTS: readonly (readonly [number, readonly number[]])[] = [
  [32, [0]],
  [57, [0, 13, 25]],
  [60, [0, 14, 28]],
  [104, [0, 24, 48, 72]],
];

Deno.test("planWanVaeTileAxis: 開始位置を値で凍結する（Python 側と同じ表）", () => {
  for (const [extent, starts] of AXIS_STARTS) {
    assertEquals([...planWanVaeTileAxis(extent, TILE).starts], [...starts], `潜在 ${extent}`);
  }
});

Deno.test("planWanVaeTiles: 832×480（潜在 60×104）は 3×4 = 12 枚・ブレンド 144 / 64 px", () => {
  const plan = planWanVaeTiles(LAYOUT, 60, 104);
  assertEquals(plan.scale, SCALE);
  assertEquals(wanVaeTileCount(plan), 12);
  assertEquals([...plan.rows.starts], [0, 14, 28]);
  assertEquals([...plan.cols.starts], [0, 24, 48, 72]);
  assertEquals([1, 2].map((index) => wanVaeBlendExtentAt(plan.rows, SCALE, index)), [144, 144]);
  assertEquals(
    [1, 2, 3].map((index) => wanVaeBlendExtentAt(plan.cols, SCALE, index)),
    [64, 64, 64],
  );
  // 480×832 は転置（軸ごとに独立）。
  const portrait = planWanVaeTiles(LAYOUT, 104, 60);
  assertEquals([...portrait.rows.starts], [0, 24, 48, 72]);
  assertEquals([...portrait.cols.starts], [0, 14, 28]);
});

Deno.test("planWanVaeTileAxis: どの全長でも配置の不変条件が成り立つ", () => {
  for (const extent of [32, 33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256]) {
    const axis = planWanVaeTileAxis(extent, TILE);
    const where = `extent=${extent}`;
    const span = extent - TILE;
    assertEquals(axis.starts[0], 0, where);
    // 末端へのスナップ = 固定形のグラフが最後のタイルも食えることの条件。
    assertEquals(axis.starts.at(-1), span, `${where}: 末端へスナップ`);
    // 本数は重なりの下限だけを制約にした最小（安全側に倒した実装はタイル数が跳ねる）。
    assertEquals(
      axis.starts.length,
      span === 0 ? 1 : Math.ceil(span / (TILE - WAN_VAE_MIN_TILE_OVERLAP)) + 1,
      `${where}: 本数`,
    );
    const gaps = axis.starts.slice(1).map((start, index) => start - axis.starts[index]);
    for (const [index, gap] of gaps.entries()) {
      assert(TILE - gap >= WAN_VAE_MIN_TILE_OVERLAP, `${where}: 対 ${index} の重なり`);
      assertEquals(wanVaeBlendExtentAt(axis, SCALE, index + 1), (TILE - gap) * SCALE, where);
    }
    // 丸め等間隔の実体 = 間隔の差は高々 1 潜在。
    if (gaps.length > 0) assert(Math.max(...gaps) - Math.min(...gaps) <= 1, `${where}: ${gaps}`);
  }
});

Deno.test("planWanVaeTileAxis / planWanVaeTiles: 受理できない形は落とす", () => {
  assertThrows(() => planWanVaeTileAxis(31, TILE), Error, "タイル幅");
  assertThrows(() => planWanVaeTileAxis(60, TILE, TILE), Error, "重なり");
  assertThrows(() => wanVaeBlendExtentAt(planWanVaeTileAxis(32, TILE), SCALE, 1), RangeError);
  // 縮尺は資産の宣言（sampleTile / tile）から — 割り切れない宣言は fail loudly。
  assertThrows(
    () => planWanVaeTiles({ latentChannels: 16, tile: 32, sampleTile: 250 }, 60, 104),
    Error,
    "縮尺",
  );
});

/** 小さな計画（タイル 8・重なりの下限 2・縮尺 `scale`）。 */
const smallPlan = (height: number, width: number, scale: number, channels = 2): WanVaeTilePlan =>
  planWanVaeTiles({ latentChannels: channels, tile: 8, sampleTile: 8 * scale }, height, width, 2);

Deno.test("wanVaeLatentTile: [C,F,H,W] の各平面から同じ矩形を切り出す（軸とフレームの取り違え検出）", () => {
  // [2,3,10,12]・値は `平面*1000 + y*12 + x`（平面 = c*3 + f）で全要素が識別できる。
  const plan = smallPlan(10, 12, 1);
  assertEquals([...plan.rows.starts], [0, 2]);
  assertEquals([...plan.cols.starts], [0, 4]);
  const latents = new Float32Array(2 * 3 * 10 * 12);
  latents.forEach((_, index) => {
    const plane = Math.floor(index / 120);
    const y = Math.floor((index % 120) / 12);
    const x = index % 12;
    latents[index] = plane * 1000 + y * 12 + x;
  });
  const tile = wanVaeLatentTile(plan, latents, 1, 1);
  assertEquals(tile.length, 2 * 3 * 64);
  for (let plane = 0; plane < 6; plane += 1) {
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        assertEquals(tile[plane * 64 + y * 8 + x], plane * 1000 + (2 + y) * 12 + (4 + x));
      }
    }
  }
  assertThrows(() => wanVaeLatentTile(plan, latents, 2, 0), RangeError);
  assertThrows(() => wanVaeLatentTile(plan, latents.subarray(1), 0, 0), Error, "要素数");
});

const bits = (values: Float32Array): Uint32Array =>
  new Uint32Array(values.buffer, values.byteOffset, values.length);

Deno.test("assembleWanVaeTiles: 1 枚（縮退）はタイルの素の写し（ビット同一）", () => {
  const plan = smallPlan(8, 8, 1);
  assertEquals(wanVaeTileCount(plan), 1);
  const tile = new Float32Array(3 * 5 * 64).map((_, index) => Math.sin(index) * 3);
  assertEquals(bits(assembleWanVaeTiles([tile], plan)), bits(tile));
});

/** タイル内の位置（行 / 列番号）を値に持つ `[3, 2, 8, 8]`（重なりの値が隣のタイルと**違う**）。 */
const rampTile = (axis: "rows" | "cols"): Float32Array =>
  new Float32Array(3 * 2 * 64).map((_, index) => {
    const offset = index % 64;
    return axis === "rows" ? Math.floor(offset / 8) : offset % 8;
  });

Deno.test("assembleWanVaeTiles: 位置依存のタイルを線形ランプが解析解どおりに畳む（全平面）", () => {
  // 全長 16・タイル 8・開始 0 / 4 / 8（重なり 4）。傾斜 0..7 は貼り合わせ後 0,1,2,3 → 中間は
  // 全て 4 → 末尾 4,5,6,7。ブレンドの向き反転・間隔の off-by-one・末端のスナップ落とし・
  // 担当領域の取り違えは全てここで割れる（同じ値が重なる作りだと向きの反転が緑のまま通る）。
  const plan = smallPlan(16, 16, 1);
  assertEquals([...plan.rows.starts], [0, 4, 8]);
  const expected = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7];
  for (const axis of ["rows", "cols"] as const) {
    const out = assembleWanVaeTiles(Array.from({ length: 9 }, () => rampTile(axis)), plan);
    assertEquals(out.length, 3 * 2 * 256);
    for (let plane = 0; plane < 6; plane += 1) {
      const line = Array.from(
        { length: 16 },
        (_, at) => out[plane * 256 + (axis === "rows" ? at * 16 : at)],
      );
      assertEquals(line, expected, `${axis} の平面 ${plane}`);
    }
  }
});

Deno.test("assembleWanVaeTiles: 角は縦 → 横の順で畳む（上流と同じ）", () => {
  // 全長 6・タイル 4・開始 0 / 2（ブレンド幅 2）・定数タイル a=1（左上）/ b=2 / c=3 / d=4。
  // 右下タイルの 2 行目（出力の行 3）: 左下は上と縦ブレンド済みで行 1 = (a + c) / 2 = 2、
  // 右上は左と横ブレンド済みで行 3 = [1, 1.5, 2, 2]。右下は縦が先で行 1 = 0.5·右上の行 3 + 0.5·d
  // = [2.5, 2.75, 3, 3] → 横で列 0 = 2・列 1 = 0.5·2 + 0.5·2.75 = 2.375。出力の行 3 は
  // 左下の担当 [2, 2] + 右下の担当 [2, 2.375, 3, 3]（横 → 縦なら [2, 2, 1.5, 2.25, 3, 3]）。
  const plan = planWanVaeTiles({ latentChannels: 1, tile: 4, sampleTile: 4 }, 6, 6, 2);
  assertEquals([...plan.rows.starts], [0, 2]);
  const tiles = [1, 2, 3, 4].map((value) => new Float32Array(3 * 16).fill(value));
  const out = assembleWanVaeTiles(tiles, plan);
  for (let plane = 0; plane < 3; plane += 1) {
    assertEquals([...out.subarray(plane * 36 + 18, plane * 36 + 24)], [2, 2, 2, 2.375, 3, 3]);
  }
});

Deno.test("assembleWanVaeTiles: 末端まで覆い、渡したタイルを破壊しない", () => {
  const plan = smallPlan(16, 26, 2);
  assertEquals(wanVaeTileCount(plan), 12);
  const tiles = Array.from({ length: 12 }, () => new Float32Array(3 * 2 * 256).fill(7));
  tiles[5].fill(9);
  const before = tiles.map((tile) => [...tile]);
  const out = assembleWanVaeTiles(tiles, plan);
  assertEquals(out.length, 3 * 2 * 32 * 52);
  assert(out.every((value) => value >= 7 && value <= 9), "覆われていない画素が残っている");
  assertEquals(tiles.map((tile) => [...tile]), before, "渡したタイルが書き換わった");
});

Deno.test("assembleWanVaeTiles: タイル枚数と要素数の食い違いを落とす", () => {
  const plan = smallPlan(16, 16, 1);
  const tile = new Float32Array(3 * 64);
  assertThrows(() => assembleWanVaeTiles([tile], plan), Error, "計画");
  assertThrows(
    () => assembleWanVaeTiles([...Array.from({ length: 8 }, () => tile), tile.subarray(1)], plan),
    Error,
    "1 枚目",
  );
  assertThrows(() => assembleWanVaeTiles(Array(9).fill(new Float32Array(64)), plan), Error, "3,F");
});

Deno.test("clampWanVaeFrames: [-1, 1] へ in-place（NaN と -0 はそのまま — torch.clamp と同じ）", () => {
  const frames = new Float32Array([-3, -1, -0, 0, 0.5, 1, 1.25, Number.NaN]);
  clampWanVaeFrames(frames);
  assertEquals([...frames.subarray(0, 7)], [-1, -1, -0, 0, 0.5, 1, 1]);
  assert(Object.is(frames[2], -0), "-0 が保たれていない");
  assert(Number.isNaN(frames[7]), "NaN が数に化けた");
});
