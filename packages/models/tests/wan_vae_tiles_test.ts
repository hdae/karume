// The host side of the tiled decode of the Wan2.1 video VAE (ADR 0118 decision 2, stage 5 — pure
// functions that need neither a GPU nor assets).
//
// The reference comparison and the degenerate check on a real GPU are in `e2e_wan_vae_tiles_test.ts`.
// This file pins the **frozen geometry table** (the same values as the Python side — frozen twice, ADR
// 0033 addendum 9a), the layout invariants and the analytic solutions of the stitching.

import { test } from "vite-plus/test";
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

/** The real tile edge (latent — the chunk graph's default input shape) and the scale. */
const TILE = 32;
const SCALE = 8;
const LAYOUT = { latentChannels: 16, tile: TILE, sampleTile: TILE * SCALE, sampleChannels: 3 };
/** Wan2.1's minimum overlap (latent) = 64 px ÷ spatial compression 8 (scale 8 × unpatchify 1). */
const MIN_OVERLAP = wanVaeMinTileOverlap(SCALE, WAN21_GENERATION.vaePatchSize);

/**
 * Start positions for tile edge 32 and minimum overlap 8 (latent extent → start positions), frozen with
 * **the same values** as the Python side (`MIRRORED_STARTS` in
 * `tools/export-recipes/wan/tests/test_vae_tiling.py`).
 *
 * - 32: degenerate (one tile — the shape of the degenerate check)
 * - 60 / 104: the two edges of the accepted set (832×480 / 480×832 — ADR 0118 decision 7). The ADR's
 *   values as they are
 * - 57: an edge where `i·span/(count−1)` lands exactly on a half (span 25 / 3 tiles → 12.5). Only this
 *   row catches a regression where 0.5 is no longer rounded up (`Math.floor` or round-half-to-even) —
 *   the invariant loop below does not decide the rounding
 */
const AXIS_STARTS: readonly (readonly [number, readonly number[]])[] = [
  [32, [0]],
  [57, [0, 13, 25]],
  [60, [0, 14, 28]],
  [104, [0, 24, 48, 72]],
];

test("planWanVaeTileAxis: freezes the start positions by value (the same table as the Python side)", () => {
  for (const [extent, starts] of AXIS_STARTS) {
    assertEquals(
      [...planWanVaeTileAxis(extent, TILE, MIN_OVERLAP).starts],
      [...starts],
      `latent ${extent}`,
    );
  }
});

test("planWanVaeTiles: 832×480 (latent 60×104) takes 3×4 = 12 tiles with blends of 144 / 64 px", () => {
  const plan = planWanVaeTiles(LAYOUT, 60, 104, MIN_OVERLAP);
  assertEquals(plan.scale, SCALE);
  assertEquals(wanVaeTileCount(plan), 12);
  assertEquals([...plan.rows.starts], [0, 14, 28]);
  assertEquals([...plan.cols.starts], [0, 24, 48, 72]);
  assertEquals(
    [1, 2].map((index) => wanVaeBlendExtentAt(plan.rows, SCALE, index)),
    [144, 144],
  );
  assertEquals(
    [1, 2, 3].map((index) => wanVaeBlendExtentAt(plan.cols, SCALE, index)),
    [64, 64, 64],
  );
  // 480×832 is the transpose (the axes are independent).
  const portrait = planWanVaeTiles(LAYOUT, 104, 60, MIN_OVERLAP);
  assertEquals([...portrait.rows.starts], [0, 24, 48, 72]);
  assertEquals([...portrait.cols.starts], [0, 14, 28]);
});

test("planWanVaeTileAxis: the layout invariants hold for every extent", () => {
  for (const extent of [32, 33, 40, 56, 57, 60, 64, 80, 104, 128, 135, 256]) {
    const axis = planWanVaeTileAxis(extent, TILE, MIN_OVERLAP);
    const where = `extent=${extent}`;
    const span = extent - TILE;
    assertEquals(axis.starts[0], 0, where);
    // Snapping to the end = the condition for the fixed-shape graph to take the last tile too.
    assertEquals(axis.starts.at(-1), span, `${where}: snaps to the end`);
    // The count is the minimum constrained only by the minimum overlap (an implementation that errs on the
    // safe side makes the tile count jump).
    assertEquals(
      axis.starts.length,
      span === 0 ? 1 : Math.ceil(span / (TILE - MIN_OVERLAP)) + 1,
      `${where}: count`,
    );
    const gaps = axis.starts.slice(1).map((start, index) => start - axis.starts[index]);
    for (const [index, gap] of gaps.entries()) {
      assert(TILE - gap >= MIN_OVERLAP, `${where}: overlap of pair ${index}`);
      assertEquals(wanVaeBlendExtentAt(axis, SCALE, index + 1), (TILE - gap) * SCALE, where);
    }
    // What rounded equal spacing means = the gaps differ by at most 1 latent.
    if (gaps.length > 0) assert(Math.max(...gaps) - Math.min(...gaps) <= 1, `${where}: ${gaps}`);
  }
});

test("planWanVaeTileAxis / planWanVaeTiles: rejects shapes they cannot accept", () => {
  assertThrows(() => planWanVaeTileAxis(31, TILE, MIN_OVERLAP), Error, "tile width");
  assertThrows(() => planWanVaeTileAxis(60, TILE, TILE), Error, "overlap");
  assertThrows(
    () => wanVaeBlendExtentAt(planWanVaeTileAxis(32, TILE, MIN_OVERLAP), SCALE, 1),
    RangeError,
  );
  // The scale comes from the asset's declaration (sampleTile / tile) — a declaration that does not divide
  // fails loudly.
  assertThrows(
    () =>
      planWanVaeTiles(
        { latentChannels: 16, tile: 32, sampleTile: 250, sampleChannels: 3 },
        60,
        104,
        MIN_OVERLAP,
      ),
    Error,
    "scale",
  );
});

test("wanVaeMinTileOverlap: overlap = 64 px ÷ spatial compression (graph ratio × unpatchify — the same formula as the Python side)", () => {
  // Two points pin the formula (the same rows as the Python side's test_vae_tiling.py). A regression that
  // ignores the unpatchify factor breaks at (8, 2) (the graph ratio alone gives 8).
  for (const [scale, patchSize, expected] of [
    [8, 1, 8],
    [8, 2, 4],
  ] as const) {
    assertEquals(wanVaeMinTileOverlap(scale, patchSize), expected, `(${scale}, ${patchSize})`);
  }
  assertEquals(wanVaeSpatialCompression(8, 2), 16);
});

test("wanVaeMinTileOverlap: rejects, without rounding, a compression that does not divide 64 px and factors that are not positive integers", () => {
  assertThrows(
    () => wanVaeMinTileOverlap(8, 3),
    Error,
    "Overlap 64 px is not divisible by the spatial compression 24 (graph ratio 8 × unpatchify 3)",
  );
  for (const [scale, patchSize] of [
    [7.8125, 1],
    [8, 1.5],
    [0, 1],
    [8, 0],
    [-8, 1],
  ] as const) {
    assertThrows(
      () => wanVaeMinTileOverlap(scale, patchSize),
      Error,
      `Spatial compression factors are not positive integers (graph ratio ${scale}, unpatchify ${patchSize})`,
    );
  }
});

/** A small plan (tile 8, minimum overlap 2, scale `scale`, 3 output channels). */
const smallPlan = (height: number, width: number, scale: number, channels = 2): WanVaeTilePlan =>
  planWanVaeTiles(
    { latentChannels: channels, tile: 8, sampleTile: 8 * scale, sampleChannels: 3 },
    height,
    width,
    2,
  );

test("wanVaeLatentTile: cuts the same rectangle from every plane of [C,F,H,W] (detects mixed-up axes and frames)", () => {
  // [2,3,10,12]; the value `plane*1000 + y*12 + x` (plane = c*3 + f) identifies every element.
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
  assertThrows(() => wanVaeLatentTile(plan, latents.subarray(1), 0, 0), Error, "element count");
});

test("wanVaeLatentTile: rejects non-integer indices (0.5, NaN) instead of returning a tile of zeros", () => {
  // A range check made only of comparisons lets 0.5 and NaN through, `starts[0.5]` is undefined → a tile of
  // zeros comes back silently (the same hole as the chunk index of `wanVaeLatentChunk`).
  const plan = smallPlan(10, 12, 1);
  const latents = new Float32Array(2 * 3 * 10 * 12).fill(1);
  for (const [row, col] of [
    [0.5, 0],
    [0, 0.5],
    [Number.NaN, 0],
    [0, Number.NaN],
  ]) {
    assertThrows(
      () => wanVaeLatentTile(plan, latents, row, col),
      RangeError,
      "out of range",
      `(${row}, ${col})`,
    );
  }
});

const bits = (values: Float32Array): Uint32Array =>
  new Uint32Array(values.buffer, values.byteOffset, values.length);

test("assembleWanVaeTiles: a single tile (degenerate) is a plain copy of the tile (bit-identical)", () => {
  const plan = smallPlan(8, 8, 1);
  assertEquals(wanVaeTileCount(plan), 1);
  const tile = new Float32Array(3 * 5 * 64).map((_, index) => Math.sin(index) * 3);
  assertEquals(bits(assembleWanVaeTiles([tile], plan)), bits(tile));
});

/**
 * `[3, 2, 8, 8]` whose values are the position inside the tile (row / column index; the overlap
 * values **differ** from the neighbouring tile's).
 */
const rampTile = (axis: "rows" | "cols"): Float32Array<ArrayBuffer> =>
  new Float32Array(3 * 2 * 64).map((_, index) => {
    const offset = index % 64;
    return axis === "rows" ? Math.floor(offset / 8) : offset % 8;
  });

test("assembleWanVaeTiles: the linear ramp folds position-dependent tiles as the analytic solution says (every plane)", () => {
  // Extent 16, tile 8, starts 0 / 4 / 8 (overlap 4). After stitching, the slope 0..7 becomes 0,1,2,3 → all 4
  // in the middle → 4,5,6,7 at the end. A reversed blend direction, an off-by-one in the spacing, a dropped
  // snap to the end and mixed-up owned regions all break here (with overlaps of equal values, a reversed
  // direction would stay green).
  const plan = smallPlan(16, 16, 1);
  assertEquals([...plan.rows.starts], [0, 4, 8]);
  const expected = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7];
  for (const axis of ["rows", "cols"] as const) {
    const out = assembleWanVaeTiles(
      Array.from({ length: 9 }, () => rampTile(axis)),
      plan,
    );
    assertEquals(out.length, 3 * 2 * 256);
    for (let plane = 0; plane < 6; plane += 1) {
      const line = Array.from(
        { length: 16 },
        (_, at) => out[plane * 256 + (axis === "rows" ? at * 16 : at)],
      );
      assertEquals(line, expected, `${axis}, plane ${plane}`);
    }
  }
});

test("assembleWanVaeTiles: corners fold vertical → horizontal (the same as upstream)", () => {
  // Extent 6, tile 4, starts 0 / 2 (blend width 2), constant tiles a=1 (top left) / b=2 / c=3 / d=4.
  // Second row of the bottom-right tile (output row 3): bottom left is already blended vertically with the
  // tile above, so its row 1 = (a + c) / 2 = 2; top right is already blended horizontally with the tile to
  // its left, so its row 3 = [1, 1.5, 2, 2]. Bottom right goes vertical first: row 1 = 0.5·(top right's row
  // 3) + 0.5·d = [2.5, 2.75, 3, 3] → then horizontal: column 0 = 2, column 1 = 0.5·2 + 0.5·2.75 = 2.375.
  // Output row 3 is bottom left's owned part [2, 2] + bottom right's owned part [2, 2.375, 3, 3]
  // (horizontal → vertical would give [2, 2, 1.5, 2.25, 3, 3]).
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

test("assembleWanVaeTiles: covers up to the end and does not destroy the tiles passed in", () => {
  const plan = smallPlan(16, 26, 2);
  assertEquals(wanVaeTileCount(plan), 12);
  const tiles = Array.from({ length: 12 }, () => new Float32Array(3 * 2 * 256).fill(7));
  tiles[5].fill(9);
  const before = tiles.map((tile) => [...tile]);
  const out = assembleWanVaeTiles(tiles, plan);
  assertEquals(out.length, 3 * 2 * 32 * 52);
  assert(
    out.every((value) => value >= 7 && value <= 9),
    "some pixels are left uncovered",
  );
  assertEquals(
    tiles.map((tile) => [...tile]),
    before,
    "the tiles passed in were rewritten",
  );
});

test("assembleWanVaeTiles: rejects mismatched tile counts and element counts", () => {
  const plan = smallPlan(16, 16, 1);
  const tile = new Float32Array(3 * 64);
  assertThrows(() => assembleWanVaeTiles([tile], plan), Error, "plan");
  assertThrows(
    () => assembleWanVaeTiles([...Array.from({ length: 8 }, () => tile), tile.subarray(1)], plan),
    Error,
    "the first tile's",
  );
  assertThrows(() => assembleWanVaeTiles(Array(9).fill(new Float32Array(64)), plan), Error, "3,F");
});

test("assembleWanVaeTiles: splits planes by the graph's output channel count (the plan's sampleChannels) — does not assume 3", () => {
  // A plan with 12 output channels (synthetic values — one degenerate tile, so pasting is a plain copy).
  // 12 × 1 frame passes as is; 3 × 1 frame (one RGB frame) does not fit the output shape and fails.
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

test("clampWanVaeFrames: clamps to [-1, 1] in place (NaN and -0 stay as they are — the same as torch.clamp)", () => {
  const frames = new Float32Array([-3, -1, -0, 0, 0.5, 1, 1.25, Number.NaN]);
  clampWanVaeFrames(frames);
  assertEquals([...frames.subarray(0, 7)], [-1, -1, -0, 0, 0.5, 1, 1]);
  assert(Object.is(frames[2], -0), "-0 was not kept");
  assert(Number.isNaN(frames[7]), "NaN turned into a number");
});

/**
 * The tolerance when stitching "tiles whose values are their absolute position" with the production plan
 * (|output − absolute position|).
 *
 * The values are sample px coordinates (0..831), and one f32 ulp on [512, 1024) is 2⁻¹⁴ ≈ 6.1e-5. The error
 * of one ramp blend `f32(f32(a·w₁) + f32(b·w₂))` is at most 2.5 ulp: three roundings (0.5 ulp each) plus the
 * deviation of the sum from 1 caused by rounding the two weights to f32 (|δ| ≤ 2⁻²⁴ → at most 1 ulp for
 * values below 1024). A blend is a convex combination, so it does not amplify the input error, and the error
 * grows with the depth of the blend chain. Tile (r, c) goes through a vertical blend with the final values of
 * (r−1, c) and a horizontal blend with the final values of (r, c−1) (in place — when nested, already blended
 * rows are read again), so the depth is at most 2r + c (7 for 3×4 tiles, 8 for 4×3) → the bound is
 * 8 × 2.5 = 20 ulp = 1.22e-3. The measured maximum (2026-10-03, CPU) is 3.05e-5..1.22e-4 (0.5..2 ulp). A blend
 * width off by 1 latent, or a shifted owned region or start position, gives a difference of 1 px (= 1.0) or
 * more.
 */
const ABSOLUTE_POSITION_TOLERANCE = 20 * 2 ** -14;

test("assembleWanVaeTiles: with the production plan (latent 60×104 / 104×60), absolute-position tiles come back to their absolute positions (nested blends)", () => {
  // The row overlap of 18 latents (144 px) exceeds half the tile width (16 latents). Tile 1's "rows blended
  // vertically with the tile above" [0, 144) px and "rows the tile below reads" [112, 256) px overlap, so part
  // of the output gets contributions from three tiles (a nesting that upstream's defaults — blend 64 px <
  // stride 192 px — never produce). The existing analytic-solution tests have an overlap of exactly half, so
  // they never reach this state. When the values are a linear function of the position, they come back to the
  // position as long as the weights sum to 1 and the two blended rows (columns) point at the same absolute
  // position — what this pins is the blend width per pair in the nested layout, start position × scale, and
  // the mapping of owned regions. The weight direction and the vertical → horizontal order make no difference
  // in this form, where neighbouring tiles hold the same values, so the analytic-solution tests above (slope,
  // corner) cover them.
  for (const [height, width] of [
    [60, 104],
    [104, 60],
  ] as const) {
    const plan = planWanVaeTiles(
      { latentChannels: 16, tile: TILE, sampleTile: TILE * SCALE, sampleChannels: 3 },
      height,
      width,
      MIN_OVERLAP,
    );
    const nested = height === 60 ? plan.rows : plan.cols;
    assert(
      wanVaeBlendExtentAt(nested, SCALE, 1) > (TILE * SCALE) / 2,
      "the nesting premise (overlap > half the tile width) no longer holds",
    );
    const side = TILE * SCALE;
    const plane = side * side;
    for (const axis of ["rows", "cols"] as const) {
      const tiles: Float32Array<ArrayBuffer>[] = [];
      for (const top of plan.rows.starts) {
        for (const left of plan.cols.starts) {
          // [3, 1, 256, 256]; value = the absolute position in the output (y px for rows, x px for columns).
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
        `latent ${height}×${width}, ${axis}: the largest difference ${worst} exceeds ${ABSOLUTE_POSITION_TOLERANCE}`,
      );
    }
  }
});

// ---- Host unpatchify and the end of the VAE stage (ADR 0121 stage 5 — Wan2.2 stitches in patchify
// space, then unpatchifies) ----

type Shape4 = readonly [number, number, number, number];

/**
 * The unpatchify ordering table (p = 2, `[dy][dx]` → offset within the channel). In upstream diffusers'
 * `unpatchify`, `view(c, r, q, f, h, w).permute(c, f, h, q, w, r)`, r = dx (the offset along the width) is
 * the major one and q = dy (the offset along the height) the minor one. Frozen with the same values as the
 * Python side (`UNPATCHIFY_SOURCE_CHANNEL` in `tools/export-recipes/wan/tests/test_vae_tiling_ti2v.py`),
 * which compares them with the upstream function itself (frozen twice).
 */
const UNPATCHIFY_SOURCE_CHANNEL: readonly (readonly number[])[] = [
  [0, 2],
  [1, 3],
];

/**
 * An unpatchify built from the table (p = 2 — looks up the input from the output coordinates; built
 * differently from the implementation's nested loops).
 */
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
          out[((channel * count + frame) * outHeight + y) * outWidth + x] =
            input[
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

/** `[12, 2, 3, 5]` in patchify space (`[3, 2, 6, 10]` as RGB). */
const PATCH_SHAPE: Shape4 = [12, 2, 3, 5];

test("unpatchifyWanVaeFrames: p = 1 returns the same instance as the input (no copy — the path of generations without unpatchify)", () => {
  const frames = arange(3 * 2 * 3 * 5);
  assertStrictEquals(unpatchifyWanVaeFrames(frames, [3, 2, 3, 5], 1), frames);
});

test("unpatchifyWanVaeFrames: p = 2 copies as the ordering table says and does not match the table with dx and dy swapped", () => {
  const input = arange(12 * 2 * 3 * 5);
  const got = unpatchifyWanVaeFrames(input, PATCH_SHAPE, 2);
  assertEquals(got.length, input.length);
  assertEquals([...got], [...unpatchifyByTable(input, PATCH_SHAPE, UNPATCHIFY_SOURCE_CHANNEL)]);
  // Two points of the upstream formula: output (c 0, f 0, y 0, x 1) = input channel 2 at (0, 0), and (y 1, x 0)
  // = input channel 1 at (0, 0) (input channel k starts at k · 2·3·5).
  assertEquals([got[1], got[10]], [2 * 30, 1 * 30]);
  // Control: arange can tell an ordering error (width and height offsets mixed up) apart.
  assertNotEquals(
    [...got],
    [
      ...unpatchifyByTable(input, PATCH_SHAPE, [
        [0, 1],
        [2, 3],
      ]),
    ],
  );
  assertEquals([...input], [...arange(input.length)], "the input was rewritten");
});

test("unpatchifyWanVaeFrames: rejects mismatched factors, shapes, channel counts and lengths (with p = 1 too)", () => {
  const input = arange(12 * 2 * 3 * 5);
  assertThrows(
    () => unpatchifyWanVaeFrames(input, PATCH_SHAPE, 0),
    Error,
    "patch size 0 is not a positive integer",
  );
  assertThrows(() => unpatchifyWanVaeFrames(input, PATCH_SHAPE, 1.5), Error, "patch size 1.5");
  assertThrows(
    () => unpatchifyWanVaeFrames(input, [12, 2, 3, 0], 2),
    Error,
    "[12, 2, 3, 0] has a dimension that is not a positive integer",
  );
  assertThrows(
    () => unpatchifyWanVaeFrames(arange(10 * 2 * 3 * 5), [10, 2, 3, 5], 2),
    Error,
    "channel count 10 is not divisible by patch size 2² = 4",
  );
  assertThrows(
    () => unpatchifyWanVaeFrames(arange(359), PATCH_SHAPE, 2),
    Error,
    "element count 359 does not match [12, 2, 3, 5] (360)",
  );
  assertThrows(
    () => unpatchifyWanVaeFrames(arange(89), [3, 2, 3, 5], 1),
    Error,
    "element count 89",
  );
});

const OWNER = "wan.Pipeline";

/**
 * A degenerate one-tile plan (scale 2, `sampleChannels` output channels). The end-of-stage checks read only
 * the output channel count and latent × scale from the plan.
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

/** A request for RGB `[3, 2, 8, 12]` (`[12, 2, 4, 6]` in patchify space with p = 2). */
const RGB = { frames: 2, height: 8, width: 12 } as const;
const PATCHED = { ...RGB, tiles: finishTiles(12, 2, 3) };

test("finishWanVaeFrames: generations without unpatchify (p = 1) give the same values and the same instance as the old end of stage (finiteness + clamp)", () => {
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

test("finishWanVaeFrames: p = 2 is clamp(the table-built unpatchify) (bit-identical)", () => {
  const assembled = new Float32Array(12 * 2 * 4 * 6).map((_, index) => (index - 288) / 160);
  const expected = unpatchifyByTable(assembled, [12, 2, 4, 6], UNPATCHIFY_SOURCE_CHANNEL);
  clampWanVaeFrames(expected);
  const got = finishWanVaeFrames(assembled, PATCHED, { vaePatchSize: 2 }, OWNER);
  assertEquals(bits(got), bits(expected));
  assert(
    got.includes(1) && got.includes(-1),
    "the clamp never took effect (the input range is too narrow)",
  );
});

test("finishWanVaeFrames: names a non-finite value before the clamp by RGB coordinates (not patchify-space coordinates)", () => {
  // Input channel 6 = c 1, dx 1, dy 0, frame 1, (y 2, x 3) → RGB channel 1, frame 1, (x 7, y 4).
  const assembled = new Float32Array(12 * 2 * 4 * 6);
  assembled[((6 * 2 + 1) * 4 + 2) * 6 + 3] = Number.POSITIVE_INFINITY;
  assertThrows(
    () => finishWanVaeFrames(assembled, PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE output (before clamping) channel 1, frame 1, pixel (x=7, y=4) is non-finite (Infinity)`,
  );
});

test("finishWanVaeFrames: checks the shape per dimension (rejects a mismatched unpatchify factor or frame count even when the element count matches)", () => {
  const assembled = new Float32Array(12 * 2 * 4 * 6);
  // Finishes a 12-channel output asset with a descriptor without unpatchify (576 elements, the same as RGB
  // [3, 2, 8, 12]).
  assertThrows(
    () => finishWanVaeFrames(assembled, PATCHED, WAN21_GENERATION, OWNER),
    Error,
    `${OWNER}: VAE output shape [12, 2, 4, 6] (unpatchify 1 gives [12, 2, 4, 6]) differs from [3, 2, 8, 12]`,
  );
  // The actual frame count is derived from the length (2 frames' worth against the requested 3).
  assertThrows(
    () => finishWanVaeFrames(assembled, { ...PATCHED, frames: 3 }, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: VAE output shape [12, 2, 4, 6] (unpatchify 2 gives [3, 2, 8, 12]) differs from [3, 3, 8, 12]`,
  );
  assertThrows(
    () => finishWanVaeFrames(assembled.subarray(1), PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: the VAE output's 575 elements do not form [12, F, 4, 6] (not divisible by 288 elements per frame)`,
  );
});

test("finishWanVaeFrames: rejects 0 elements, and a plan whose output channel count is not divisible by p², with that reason", () => {
  assertThrows(
    () => finishWanVaeFrames(new Float32Array(0), PATCHED, { vaePatchSize: 2 }, OWNER),
    Error,
    `${OWNER}: the VAE output has 0 elements (no frames at all)`,
  );
  // Finishes a 3-channel output plan with p = 2 (the shape comparison must not produce a fractional channel
  // count of 0.75).
  assertThrows(
    () =>
      finishWanVaeFrames(
        new Float32Array(3 * 2 * 4 * 6),
        { ...RGB, tiles: finishTiles(3, 2, 3) },
        { vaePatchSize: 2 },
        OWNER,
      ),
    Error,
    `${OWNER}: VAE output 3 channels are not divisible by the unpatchify patch size 2² = 4`,
  );
});
