// Wan2.1 の動画 VAE のタイル decode のホスト側（ADR 0118 決定 2・段 5 — GPU も資産も要らない純関数）。
//
// 実 GPU での参照照合と縮退門は `e2e_wan_vae_tiles_test.ts`。ここは**幾何の凍結表**（Python 側と
// 同じ値 — ADR 0033 追記 9a の二重凍結）・配置の不変条件・貼り合わせの解析解を押さえる。

import {
  assert,
  assertEquals,
  assertNotEquals,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import {
  assembleWanVaeTiles,
  clampWanVaeFrames,
  planWanVaeTileAxis,
  planWanVaeTiles,
  unpatchifyWanVaeFrames,
  wanVaeBlendExtentAt,
  wanVaeLatentTile,
  wanVaeMinTileOverlap,
  wanVaeSpatialCompression,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../src/wan/vae-tiles.ts";
import { WAN21_GENERATION } from "../src/wan/descriptor.ts";
import { finishWanVaeFrames } from "../src/wan/tile-decode.ts";

/** 実寸のタイル辺（潜在 — chunk グラフの既定の入力形）と縮尺。 */
const TILE = 32;
const SCALE = 8;
const LAYOUT = { latentChannels: 16, tile: TILE, sampleTile: TILE * SCALE, sampleChannels: 3 };
/** Wan2.1 の重なりの下限（潜在）= 64 px ÷ 空間の圧縮 8（縮尺 8 × unpatchify 1）。 */
const MIN_OVERLAP = wanVaeMinTileOverlap(SCALE, WAN21_GENERATION.vaePatchSize);

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
    assertEquals(
      [...planWanVaeTileAxis(extent, TILE, MIN_OVERLAP).starts],
      [...starts],
      `潜在 ${extent}`,
    );
  }
});

Deno.test("planWanVaeTiles: 832×480（潜在 60×104）は 3×4 = 12 枚・ブレンド 144 / 64 px", () => {
  const plan = planWanVaeTiles(LAYOUT, 60, 104, MIN_OVERLAP);
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
  const portrait = planWanVaeTiles(LAYOUT, 104, 60, MIN_OVERLAP);
  assertEquals([...portrait.rows.starts], [0, 24, 48, 72]);
  assertEquals([...portrait.cols.starts], [0, 14, 28]);
});

Deno.test("planWanVaeTileAxis: どの全長でも配置の不変条件が成り立つ", () => {
  for (const extent of [32, 33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256]) {
    const axis = planWanVaeTileAxis(extent, TILE, MIN_OVERLAP);
    const where = `extent=${extent}`;
    const span = extent - TILE;
    assertEquals(axis.starts[0], 0, where);
    // 末端へのスナップ = 固定形のグラフが最後のタイルも食えることの条件。
    assertEquals(axis.starts.at(-1), span, `${where}: 末端へスナップ`);
    // 本数は重なりの下限だけを制約にした最小（安全側に倒した実装はタイル数が跳ねる）。
    assertEquals(
      axis.starts.length,
      span === 0 ? 1 : Math.ceil(span / (TILE - MIN_OVERLAP)) + 1,
      `${where}: 本数`,
    );
    const gaps = axis.starts.slice(1).map((start, index) => start - axis.starts[index]);
    for (const [index, gap] of gaps.entries()) {
      assert(TILE - gap >= MIN_OVERLAP, `${where}: 対 ${index} の重なり`);
      assertEquals(wanVaeBlendExtentAt(axis, SCALE, index + 1), (TILE - gap) * SCALE, where);
    }
    // 丸め等間隔の実体 = 間隔の差は高々 1 潜在。
    if (gaps.length > 0) assert(Math.max(...gaps) - Math.min(...gaps) <= 1, `${where}: ${gaps}`);
  }
});

Deno.test("planWanVaeTileAxis / planWanVaeTiles: 受理できない形は落とす", () => {
  assertThrows(() => planWanVaeTileAxis(31, TILE, MIN_OVERLAP), Error, "タイル幅");
  assertThrows(() => planWanVaeTileAxis(60, TILE, TILE), Error, "重なり");
  assertThrows(
    () => wanVaeBlendExtentAt(planWanVaeTileAxis(32, TILE, MIN_OVERLAP), SCALE, 1),
    RangeError,
  );
  // 縮尺は資産の宣言（sampleTile / tile）から — 割り切れない宣言は fail loudly。
  assertThrows(
    () =>
      planWanVaeTiles(
        { latentChannels: 16, tile: 32, sampleTile: 250, sampleChannels: 3 },
        60,
        104,
        MIN_OVERLAP,
      ),
    Error,
    "縮尺",
  );
});

Deno.test("wanVaeMinTileOverlap: 重なり = 64 px ÷ 空間の圧縮（グラフの比 × unpatchify — Python 側と同じ式）", () => {
  // 2 点で式を縛る（Python 側 test_vae_tiling.py と同じ行）。unpatchify の倍率を無視する退行は
  // (8, 2) で割れる（グラフの比だけなら 8）。
  for (const [scale, patchSize, expected] of [[8, 1, 8], [8, 2, 4]] as const) {
    assertEquals(wanVaeMinTileOverlap(scale, patchSize), expected, `(${scale}, ${patchSize})`);
  }
  assertEquals(wanVaeSpatialCompression(8, 2), 16);
});

Deno.test("wanVaeMinTileOverlap: 64 px を割り切れない圧縮と、正の整数でない因子は丸めずに落とす", () => {
  assertThrows(
    () => wanVaeMinTileOverlap(8, 3),
    Error,
    "重なり 64 px が空間の圧縮 24（グラフの比 8 × unpatchify 3）で割り切れない",
  );
  for (const [scale, patchSize] of [[7.8125, 1], [8, 1.5], [0, 1], [8, 0], [-8, 1]] as const) {
    assertThrows(
      () => wanVaeMinTileOverlap(scale, patchSize),
      Error,
      `空間の圧縮の因子が正の整数でない（グラフの比 ${scale}・unpatchify ${patchSize}）`,
    );
  }
});

/** 小さな計画（タイル 8・重なりの下限 2・縮尺 `scale`・出口 3 チャネル）。 */
const smallPlan = (height: number, width: number, scale: number, channels = 2): WanVaeTilePlan =>
  planWanVaeTiles(
    { latentChannels: channels, tile: 8, sampleTile: 8 * scale, sampleChannels: 3 },
    height,
    width,
    2,
  );

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
  const plan = planWanVaeTiles(
    { latentChannels: 1, tile: 4, sampleTile: 4, sampleChannels: 3 },
    6,
    6,
    2,
  );
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

Deno.test("assembleWanVaeTiles: 平面はグラフの出口のチャネル数（計画の sampleChannels）で割る — 3 を仮定しない", () => {
  // 出口 12 チャネルの計画（合成の値 — 縮退の 1 枚なので貼り付けは素の写し）。12 × 1 フレームは
  // そのまま通り、3 × 1 フレーム（RGB 1 枚ぶん）は出口の形に合わないので落ちる。
  const plan = planWanVaeTiles(
    { latentChannels: 2, tile: 8, sampleTile: 8, sampleChannels: 12 },
    8,
    8,
    2,
  );
  const tile = new Float32Array(12 * 64).map((_, index) => index);
  assertEquals(bits(assembleWanVaeTiles([tile], plan)), bits(tile));
  assertThrows(() => assembleWanVaeTiles([new Float32Array(3 * 64)], plan), Error, "[12,F,8,8]");
});

Deno.test("clampWanVaeFrames: [-1, 1] へ in-place（NaN と -0 はそのまま — torch.clamp と同じ）", () => {
  const frames = new Float32Array([-3, -1, -0, 0, 0.5, 1, 1.25, Number.NaN]);
  clampWanVaeFrames(frames);
  assertEquals([...frames.subarray(0, 7)], [-1, -1, -0, 0, 0.5, 1, 1]);
  assert(Object.is(frames[2], -0), "-0 が保たれていない");
  assert(Number.isNaN(frames[7]), "NaN が数に化けた");
});

/**
 * 本番の計画で「絶対位置を値に持つタイル」を貼り合わせたときの許容差（出力 − 絶対位置・絶対値）。
 *
 * 値は sample の px 座標（0〜831）で、f32 の 1 ulp は [512, 1024) で 2⁻¹⁴ ≈ 6.1e-5。1 回のランプ合成
 * `f32(f32(a·w₁) + f32(b·w₂))` の誤差は、丸め 3 回（各 0.5 ulp）と、重み 2 本を f32 へ丸めたことによる
 * 和の 1 からのずれ（|δ| ≤ 2⁻²⁴ → 値 1024 未満で 1 ulp 以下）で 2.5 ulp 以下。合成は凸結合なので入力の
 * 誤差を増やさず、誤差は合成の連鎖の深さに比例する。タイル (r, c) は (r−1, c) の最終値との縦の合成と
 * (r, c−1) の最終値との横の合成を経る（in-place — 入れ子ではブレンド済みの行をさらに読む）ので、深さは
 * 高々 2r + c（3×4 枚で 7・4×3 枚で 8）→ 8 × 2.5 = 20 ulp = 1.22e-3 を上限にする。実測（2026-10-03・
 * CPU）の最大は 3.05e-5〜1.22e-4（0.5〜2 ulp）。ブレンド幅の 1 潜在のずれ・担当領域や開始位置のずれは
 * 1 px（= 1.0）以上の差になる。
 */
const ABSOLUTE_POSITION_TOLERANCE = 20 * 2 ** -14;

Deno.test("assembleWanVaeTiles: 本番の計画（潜在 60×104 / 104×60）で絶対位置のタイルが絶対位置に戻る（入れ子のブレンド）", () => {
  // 行の重なり 18 潜在（144 px）はタイル幅の半分（16 潜在）を超える。タイル 1 の「上と縦ブレンドした
  // 行」[0, 144) px と「下のタイルが読む行」[112, 256) px が重なり、出力の一部は 3 枚のタイルの寄与に
  // なる（上流の既定〈ブレンド 64 px < stride 192 px〉では起きない入れ子）。既存の解析解のテストは
  // 重なりがちょうど半分なので、この状態を通らない。値が位置の線形関数なら、重みの和が 1 で、合成する
  // 2 行（列）が同じ絶対位置を指す限り位置に戻る — 縛るのは入れ子の配置でのブレンド幅（対ごと）・開始位置
  // × 縮尺・担当領域の対応。重みの向きと縦 → 横の順は、隣のタイルが同じ値を持つこの形では差が出ないので
  // 上の解析解のテスト（傾斜・角）が持つ。
  for (const [height, width] of [[60, 104], [104, 60]] as const) {
    const plan = planWanVaeTiles(
      { latentChannels: 16, tile: TILE, sampleTile: TILE * SCALE, sampleChannels: 3 },
      height,
      width,
      MIN_OVERLAP,
    );
    const nested = height === 60 ? plan.rows : plan.cols;
    assert(
      wanVaeBlendExtentAt(nested, SCALE, 1) > (TILE * SCALE) / 2,
      "入れ子の前提（重なり > タイル幅の半分）が崩れた",
    );
    const side = TILE * SCALE;
    const plane = side * side;
    for (const axis of ["rows", "cols"] as const) {
      const tiles: Float32Array[] = [];
      for (const top of plan.rows.starts) {
        for (const left of plan.cols.starts) {
          // [3, 1, 256, 256]・値 = 出力の絶対位置（行なら y・列なら x の px）。
          tiles.push(
            new Float32Array(3 * plane).map((_, index) => {
              const offset = index % plane;
              return axis === "rows"
                ? top * SCALE + Math.floor(offset / side)
                : left * SCALE + (offset % side);
            }),
          );
        }
      }
      const out = assembleWanVaeTiles(tiles, plan);
      const outHeight = height * SCALE;
      const outWidth = width * SCALE;
      assertEquals(out.length, 3 * outHeight * outWidth);
      let worst = 0;
      for (let index = 0; index < out.length; index += 1) {
        const offset = index % (outHeight * outWidth);
        const expected = axis === "rows" ? Math.floor(offset / outWidth) : offset % outWidth;
        worst = Math.max(worst, Math.abs(out[index] - expected));
      }
      assert(
        worst <= ABSOLUTE_POSITION_TOLERANCE,
        `潜在 ${height}×${width} の ${axis}: 最大差 ${worst} が ${ABSOLUTE_POSITION_TOLERANCE} を超える`,
      );
    }
  }
});

// ---- ホストの unpatchify と VAE 段の末尾（ADR 0121 段 5 — Wan2.2 は patchify 空間で貼り合わせてから戻す）----

type Shape4 = readonly [number, number, number, number];

/**
 * unpatchify の並びの表（p = 2・`[dy][dx]` → チャネル内のずれ）。上流 diffusers `unpatchify` の
 * `view(c, r, q, f, h, w).permute(c, f, h, q, w, r)` で、r = dx（幅方向のずれ）が上位・q = dy（高さ方向の
 * ずれ）が下位。Python 側（`tools/export-recipes/wan/tests/test_vae_tiling_ti2v.py` の
 * `UNPATCHIFY_SOURCE_CHANNEL`）と同じ値で凍結し、そちらが上流の関数そのものと照合する（二重凍結）。
 */
const UNPATCHIFY_SOURCE_CHANNEL: readonly (readonly number[])[] = [[0, 2], [1, 3]];

/** 表から組んだ unpatchify（p = 2 — 出力の座標から入力を引く。実装の入れ子のループとは別の組み立て）。 */
const unpatchifyByTable = (
  input: Float32Array,
  [channels, count, height, width]: Shape4,
  table: readonly (readonly number[])[],
): Float32Array<ArrayBuffer> => {
  const outHeight = height * 2;
  const outWidth = width * 2;
  const out = new Float32Array(input.length);
  for (let channel = 0; channel < channels / 4; channel += 1) {
    for (let frame = 0; frame < count; frame += 1) {
      for (let y = 0; y < outHeight; y += 1) {
        for (let x = 0; x < outWidth; x += 1) {
          const source = channel * 4 + table[y % 2][x % 2];
          out[((channel * count + frame) * outHeight + y) * outWidth + x] = input[
            ((source * count + frame) * height + Math.floor(y / 2)) * width + Math.floor(x / 2)
          ];
        }
      }
    }
  }
  return out;
};

const arange = (length: number): Float32Array<ArrayBuffer> =>
  new Float32Array(length).map((_, index) => index);

/** patchify 空間の `[12, 2, 3, 5]`（RGB にすると `[3, 2, 6, 10]`）。 */
const PATCH_SHAPE: Shape4 = [12, 2, 3, 5];

Deno.test("unpatchifyWanVaeFrames: p = 1 は入力の同じインスタンスを返す（写さない — unpatchify の無い世代の経路）", () => {
  const frames = arange(3 * 2 * 3 * 5);
  assertStrictEquals(unpatchifyWanVaeFrames(frames, [3, 2, 3, 5], 1), frames);
});

Deno.test("unpatchifyWanVaeFrames: p = 2 は並びの表どおりに写す・dx と dy を入れ替えた表とは一致しない", () => {
  const input = arange(12 * 2 * 3 * 5);
  const got = unpatchifyWanVaeFrames(input, PATCH_SHAPE, 2);
  assertEquals(got.length, input.length);
  assertEquals([...got], [...unpatchifyByTable(input, PATCH_SHAPE, UNPATCHIFY_SOURCE_CHANNEL)]);
  // 上流の式の 2 点: 出力 (c 0, f 0, y 0, x 1) = 入力チャネル 2 の (0, 0)・(y 1, x 0) = 入力チャネル 1 の (0, 0)
  // （入力チャネル k の先頭 = k · 2·3·5）。
  assertEquals([got[1], got[10]], [2 * 30, 1 * 30]);
  // 対照: arange が並びの誤り（幅と高さのずれの取り違え）を区別できること。
  assertNotEquals([...got], [...unpatchifyByTable(input, PATCH_SHAPE, [[0, 1], [2, 3]])]);
  assertEquals([...input], [...arange(input.length)], "入力を書き換えた");
});

Deno.test("unpatchifyWanVaeFrames: 倍率・形・チャネル数・長さの食い違いは落とす（p = 1 でも）", () => {
  const input = arange(12 * 2 * 3 * 5);
  assertThrows(
    () => unpatchifyWanVaeFrames(input, PATCH_SHAPE, 0),
    Error,
    "倍率 0 が正の整数でない",
  );
  assertThrows(() => unpatchifyWanVaeFrames(input, PATCH_SHAPE, 1.5), Error, "倍率 1.5");
  assertThrows(
    () => unpatchifyWanVaeFrames(input, [12, 2, 3, 0], 2),
    Error,
    "[12, 2, 3, 0] の次元が正の整数でない",
  );
  assertThrows(
    () => unpatchifyWanVaeFrames(arange(10 * 2 * 3 * 5), [10, 2, 3, 5], 2),
    Error,
    "チャネル数 10 が倍率 2² = 4 で割り切れない",
  );
  assertThrows(
    () => unpatchifyWanVaeFrames(arange(359), PATCH_SHAPE, 2),
    Error,
    "要素数 359 が [12, 2, 3, 5]（360）と違う",
  );
  assertThrows(() => unpatchifyWanVaeFrames(arange(89), [3, 2, 3, 5], 1), Error, "要素数 89");
});

const OWNER = "WanPipeline";

/**
 * 縮退 1 枚のタイル計画（縮尺 2・出口 `sampleChannels`）。末尾の検査が計画から読むのは出口のチャネル数と
 * 潜在 × 縮尺だけ。
 */
const finishTiles = (
  sampleChannels: number,
  latentHeight: number,
  latentWidth: number,
): WanVaeTilePlan => ({
  latentChannels: 48,
  sampleChannels,
  scale: 2,
  rows: { extent: latentHeight, tile: latentHeight, starts: [0] },
  cols: { extent: latentWidth, tile: latentWidth, starts: [0] },
});

/** RGB `[3, 2, 8, 12]` の要求（patchify 空間は p = 2 で `[12, 2, 4, 6]`）。 */
const RGB = { frames: 2, height: 8, width: 12 } as const;
const PATCHED = { ...RGB, tiles: finishTiles(12, 2, 3) };

Deno.test("finishWanVaeFrames: unpatchify の無い世代（p = 1）は旧来の末尾（有限性 + クランプ）と同じ値・同じインスタンス", () => {
  const assembled = new Float32Array(3 * 2 * 8 * 12).map((_, index) => Math.sin(index) * 2);
  assembled[5] = -0;
  const legacy = Float32Array.from(assembled);
  clampWanVaeFrames(legacy);
  const got = finishWanVaeFrames(
    assembled,
    { ...RGB, tiles: finishTiles(3, 4, 6) },
    WAN21_GENERATION,
    OWNER,
  );
  assertStrictEquals(got, assembled);
  assertEquals(bits(got), bits(legacy));
});

Deno.test("finishWanVaeFrames: p = 2 は clamp(表で組んだ unpatchify)（ビット一致）", () => {
  const assembled = new Float32Array(12 * 2 * 4 * 6).map((_, index) => (index - 288) / 160);
  const expected = unpatchifyByTable(assembled, [12, 2, 4, 6], UNPATCHIFY_SOURCE_CHANNEL);
  clampWanVaeFrames(expected);
  const got = finishWanVaeFrames(assembled, PATCHED, { vaePatchSize: 2 }, OWNER);
  assertEquals(bits(got), bits(expected));
  assert(got.includes(1) && got.includes(-1), "クランプが 1 度も効いていない（入力の値域が狭い）");
});

Deno.test("finishWanVaeFrames: 非有限はクランプの前に RGB の座標で名指す（patchify 空間の座標ではない）", () => {
  // 入力チャネル 6 = c 1・dx 1・dy 0、フレーム 1・(y 2, x 3) → RGB の channel 1・フレーム 1・(x 7, y 4)。
  const assembled = new Float32Array(12 * 2 * 4 * 6);
  assembled[((6 * 2 + 1) * 4 + 2) * 6 + 3] = Number.POSITIVE_INFINITY;
  assertThrows(
    () => finishWanVaeFrames(assembled, PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE の出力（クランプ前）の channel 1・フレーム 1・画素 (x=7, y=4) が非有限（Infinity）`,
  );
});

Deno.test("finishWanVaeFrames: 形は次元ごとに見る（要素数が同じでも unpatchify の倍率・フレーム数の食い違いを落とす）", () => {
  const assembled = new Float32Array(12 * 2 * 4 * 6);
  // 出口 12 ch の資産を unpatchify の無い記述子で閉じる（要素数は RGB の [3, 2, 8, 12] と同じ 576）。
  assertThrows(
    () => finishWanVaeFrames(assembled, PATCHED, WAN21_GENERATION, OWNER),
    Error,
    `${OWNER}: VAE の出力の形 [12, 2, 4, 6]（unpatchify 1 で [12, 2, 4, 6]）が [3, 2, 8, 12] と違う`,
  );
  // 実フレーム数は長さから導く（要求の 3 フレームに対して 2 フレームぶん）。
  assertThrows(
    () => finishWanVaeFrames(assembled, { ...PATCHED, frames: 3 }, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE の出力の形 [12, 2, 4, 6]（unpatchify 2 で [3, 2, 8, 12]）が [3, 3, 8, 12] と違う`,
  );
  assertThrows(
    () => finishWanVaeFrames(assembled.subarray(1), PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE の出力 575 要素が [12, F, 4, 6] にならない（1 フレーム 288 要素で割り切れない）`,
  );
});

Deno.test("finishWanVaeFrames: 0 要素と、出口のチャネル数が p² で割り切れない計画は、その理由で落とす", () => {
  assertThrows(
    () => finishWanVaeFrames(new Float32Array(0), PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE の出力が 0 要素（フレームが 1 枚も無い）`,
  );
  // 出口 3 ch の計画を p = 2 で閉じる（形の比較に小数のチャネル数 0.75 を出さない）。
  assertThrows(
    () =>
      finishWanVaeFrames(
        new Float32Array(3 * 2 * 4 * 6),
        { ...RGB, tiles: finishTiles(3, 2, 3) },
        { vaePatchSize: 2 },
        OWNER,
      ),
    Error,
    `${OWNER}: VAE の出口 3 チャネルが unpatchify の倍率 2² = 4 で割り切れない`,
  );
});
