// The geometry and the stitching of the tiled decode (VAE tiling — ADR 0033).
//
// The comparison with the reference on a real GPU / real weights is the E2E test (P3, iteration 2).
// This file is the pure-function side that needs neither a GPU nor assets, and covers the
// **geometric invariants** (a layout the fixed-shape decoder can consume) and the **analytic
// solution of the stitching**.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ANIMA_SPATIAL_COMPRESSION } from "../src/anima/dit-tokens.ts";
import {
  MAX_RESOLUTION_SIDE,
  MIN_RESOLUTION_SIDE,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";
import {
  assembleTiles,
  blendExtentAt,
  decodeTiled,
  latentTile,
  MIN_TILE_OVERLAP_LATENT,
  planTileAxis,
  planVaeTiling,
  tileCount,
} from "../src/anima/tiling.ts";

test("planTileAxis: 1024px (latent 128) is 3 tiles, starts [0,32,64], blend 256px", () => {
  // A side that gets the same layout as the old rule (divisor snapping). **Byte-unchanged output at
  // the shipped resolutions** rests on this, so the values are frozen as literals (so that the
  // tile-count change of 1824×1248 does not drag it along).
  const axis = planTileAxis(128, 64, 8);
  assertEquals([...axis.starts], [0, 32, 64]);
  assertEquals([blendExtentAt(axis, 8, 1), blendExtentAt(axis, 8, 2)], [256, 256]);
});

/**
 * The start positions (latent) of all 97 sides of the accepted set (512–2048px, step 16px) —
 * `[side, starts]`.
 *
 * The invariant loop below does not decide the rounding **direction** (with `Math.round` replaced
 * by `Math.floor`, start 0 / snap at the end / count / overlap lower bound / spacing spread ≤ 1 all
 * still hold). A layout shifted by 1 latent throws nothing; it only moves the blend position of the
 * seams and blurs the image, so it is frozen **by value**. The e2e PNG check cannot catch it either
 * — the 1024 family a real GPU runs consists of sides where the span divides by count−1 and the
 * rounding is the identity.
 *
 * Origin: an exhaustive comparison with the old rule (divisor snapping): the count changes on only
 * 24 sides, which match `RECOUNTED_SIDES` in `anima_resolution_test.ts`, and the remaining 73 sides
 * match the old rule down to the start positions (= shipped bytes unchanged). `i·span/(count−1)`
 * lands exactly on a half on six sides, 1872 / 1904 / 1936 / 1968 / 2000 / 2032px, and there it
 * rounds half up (1872px = `(0,43,85,128,170)`) — if the integer formula on the Python side
 * regressed to round-half- even it would give (0,42,85,128,170), and the mirrored comparison
 * (`MIRRORED_STARTS` in `tools/export-recipes/anima/tests/test_tiling.py`) would break.
 */
const AXIS_STARTS: readonly (readonly [number, readonly number[]])[] = [
  [512, [0]],
  [528, [0, 2]],
  [544, [0, 4]],
  [560, [0, 6]],
  [576, [0, 8]],
  [592, [0, 10]],
  [608, [0, 12]],
  [624, [0, 14]],
  [640, [0, 16]],
  [656, [0, 18]],
  [672, [0, 20]],
  [688, [0, 22]],
  [704, [0, 24]],
  [720, [0, 26]],
  [736, [0, 28]],
  [752, [0, 30]],
  [768, [0, 32]],
  [784, [0, 34]],
  [800, [0, 36]],
  [816, [0, 38]],
  [832, [0, 40]],
  [848, [0, 42]],
  [864, [0, 44]],
  [880, [0, 46]],
  [896, [0, 48]],
  [912, [0, 50]],
  [928, [0, 52]],
  [944, [0, 54]],
  [960, [0, 56]],
  [976, [0, 29, 58]],
  [992, [0, 30, 60]],
  [1008, [0, 31, 62]],
  [1024, [0, 32, 64]],
  [1040, [0, 33, 66]],
  [1056, [0, 34, 68]],
  [1072, [0, 35, 70]],
  [1088, [0, 36, 72]],
  [1104, [0, 37, 74]],
  [1120, [0, 38, 76]],
  [1136, [0, 39, 78]],
  [1152, [0, 40, 80]],
  [1168, [0, 41, 82]],
  [1184, [0, 42, 84]],
  [1200, [0, 43, 86]],
  [1216, [0, 44, 88]],
  [1232, [0, 45, 90]],
  [1248, [0, 46, 92]],
  [1264, [0, 47, 94]],
  [1280, [0, 48, 96]],
  [1296, [0, 49, 98]],
  [1312, [0, 50, 100]],
  [1328, [0, 51, 102]],
  [1344, [0, 52, 104]],
  [1360, [0, 53, 106]],
  [1376, [0, 54, 108]],
  [1392, [0, 55, 110]],
  [1408, [0, 56, 112]],
  [1424, [0, 38, 76, 114]],
  [1440, [0, 39, 77, 116]],
  [1456, [0, 39, 79, 118]],
  [1472, [0, 40, 80, 120]],
  [1488, [0, 41, 81, 122]],
  [1504, [0, 41, 83, 124]],
  [1520, [0, 42, 84, 126]],
  [1536, [0, 43, 85, 128]],
  [1552, [0, 43, 87, 130]],
  [1568, [0, 44, 88, 132]],
  [1584, [0, 45, 89, 134]],
  [1600, [0, 45, 91, 136]],
  [1616, [0, 46, 92, 138]],
  [1632, [0, 47, 93, 140]],
  [1648, [0, 47, 95, 142]],
  [1664, [0, 48, 96, 144]],
  [1680, [0, 49, 97, 146]],
  [1696, [0, 49, 99, 148]],
  [1712, [0, 50, 100, 150]],
  [1728, [0, 51, 101, 152]],
  [1744, [0, 51, 103, 154]],
  [1760, [0, 52, 104, 156]],
  [1776, [0, 53, 105, 158]],
  [1792, [0, 53, 107, 160]],
  [1808, [0, 54, 108, 162]],
  [1824, [0, 55, 109, 164]],
  [1840, [0, 55, 111, 166]],
  [1856, [0, 56, 112, 168]],
  [1872, [0, 43, 85, 128, 170]],
  [1888, [0, 43, 86, 129, 172]],
  [1904, [0, 44, 87, 131, 174]],
  [1920, [0, 44, 88, 132, 176]],
  [1936, [0, 45, 89, 134, 178]],
  [1952, [0, 45, 90, 135, 180]],
  [1968, [0, 46, 91, 137, 182]],
  [1984, [0, 46, 92, 138, 184]],
  [2000, [0, 47, 93, 140, 186]],
  [2016, [0, 47, 94, 141, 188]],
  [2032, [0, 48, 95, 143, 190]],
  [2048, [0, 48, 96, 144, 192]],
];

test("planTileAxis: freezes the start positions of all 97 sides of the accepted set by value (down to the rounding direction)", () => {
  // First checks that the table **covers** the accepted set. When the side lower bound / ceiling /
  // step moves, it breaks here instead of the table silently becoming a partial cover.
  const sides: number[] = [];
  for (let side = MIN_RESOLUTION_SIDE; side <= MAX_RESOLUTION_SIDE; side += RESOLUTION_GRANULARITY)
    sides.push(side);
  assertEquals(
    AXIS_STARTS.map(([side]) => side),
    sides,
    "sides the table covers",
  );

  const tile = MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION;
  for (const [side, starts] of AXIS_STARTS) {
    const axis = planTileAxis(side / ANIMA_SPATIAL_COMPRESSION, tile, MIN_TILE_OVERLAP_LATENT);
    assertEquals([...axis.starts], [...starts], `start positions of ${side}px`);
  }
});

test("planTileAxis: latent 64 (512px) degenerates to one tile with no blend pair", () => {
  const axis = planTileAxis(64, 64, 8);
  assertEquals([...axis.starts], [0]);
  assertThrows(() => blendExtentAt(axis, 8, 1), RangeError, "out of range");
});

test("planTileAxis: the layout invariants hold for every latent length", () => {
  // 182 / 228 / 242 are the sides whose count jumped under the old rule (1456 / 1824 / 1936px) —
  // the very target of the removal.
  for (const extent of [64, 80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512]) {
    const axis = planTileAxis(extent, 64, 8);
    const where = `extent=${extent}`;
    const span = extent - 64;
    assertEquals(axis.starts[0], 0, where);
    // Snapping to the end = the condition for the fixed-shape decoder to consume the last tile too.
    assertEquals(axis.starts.at(-1), span, `${where}: snaps to the end`);
    // The count is exactly the analytic formula "the minimum constrained only by the overlap lower
    // bound". An implementation that plays safe with spacing 1, and the old rule's divisor search,
    // both break here.
    assertEquals(
      axis.starts.length,
      span === 0 ? 1 : Math.ceil(span / (64 - 8)) + 1,
      `${where}: count`,
    );
    const gaps = axis.starts.slice(1).map((start, index) => start - axis.starts[index]);
    for (const [index, gap] of gaps.entries()) {
      assert(
        64 - gap >= 8,
        `${where}: pair ${index} has overlap ${64 - gap}, below the lower bound 8`,
      );
      assertEquals(blendExtentAt(axis, 8, index + 1), (64 - gap) * 8, `${where}: pair ${index}`);
    }
    // The substance of rounded equal spacing = the spacing (and so the blend width) differs by at
    // most 1 latent. A regression to the upstream-style "fixed stride + snap only the last one"
    // opens only the last pair wide and breaks here.
    if (gaps.length > 0) {
      assert(Math.max(...gaps) - Math.min(...gaps) <= 1, `${where}: spacing spread ${gaps}`);
    }
  }
});

test("planTileAxis: rejects a latent shorter than a tile and an overlap that is too wide", () => {
  assertThrows(() => planTileAxis(32, 64, 8), Error, "tile width");
  assertThrows(() => planTileAxis(128, 64, 64), Error, "overlap");
});

test("planVaeTiling: derives the scale from the asset's input and output shapes and lays out each axis independently", () => {
  const geometry = planVaeTiling([1, 16, 128, 64], [1, 16, 64, 64], [1, 3, 512, 512]);
  assertEquals(geometry.scale, 8);
  assertEquals(geometry.channels, 16);
  assertEquals(geometry.rows.starts.length, 3);
  assertEquals(geometry.cols.starts.length, 1, "each axis is independent even when H ≠ W");
  assertEquals(tileCount(geometry), 3);
  // The entry's accepted set (channel mismatch / scale that differs between axes / B>1).
  assertThrows(
    () => planVaeTiling([1, 8, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "channel count",
  );
  assertThrows(
    () => planVaeTiling([1, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 256]),
    Error,
    "scale",
  );
  assertThrows(
    () => planVaeTiling([2, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "batch=1",
  );
});

test("latentTile: cuts the right rectangle out of each plane (detects axis and channel mix-ups)", () => {
  // [1,2,4,4]. The values are `channel*100 + y*10 + x`, so every element is identifiable.
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

test("latentTile: a non-integer index (0.5, NaN) is rejected instead of returning a zero tile", () => {
  // A range check made only of comparisons lets 0.5 and NaN through, and `starts[0.5]` is undefined
  // → an empty subarray silently returns a zero tile.
  const geometry = planVaeTiling([1, 2, 4, 4], [1, 2, 2, 2], [1, 1, 2, 2], 1);
  const latents = new Float32Array(2 * 16).fill(1);
  for (const [row, col] of [
    [0.5, 0],
    [0, 0.5],
    [Number.NaN, 0],
    [0, Number.NaN],
  ]) {
    assertThrows(
      () => latentTile(latents, geometry, row, col),
      RangeError,
      "out of range",
      `(${row}, ${col})`,
    );
  }
});

test("assembleTiles: a single tile (degenerate) is a plain copy of the decode output (bit-identical)", () => {
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

test("assembleTiles: linear ramps fold a position-dependent decode output exactly as the analytic solution", () => {
  // extent 16 / tile 8 / stride 4 (overlap 4) gives start positions 0,4,8. When the values inside a
  // tile are the row-number slope `0..7`, after stitching it is `0,1,2,3` → all 4 in the middle →
  // `4,5,6,7` at the end. The overlap values **differ** per tile, so a reversed blend direction /
  // an off-by-one stride / a dropped snap of the last tile / a mix-up of the owned regions all
  // break here (with equal values overlapping, a reversed direction would still pass green).
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
  const rows = assembleTiles(
    Array.from({ length: 9 }, () => rowRamp),
    geometry,
  );
  assertEquals([...Array.from({ length: 16 }, (_, y) => rows[y * 16])], expected, "row direction");
  const cols = assembleTiles(
    Array.from({ length: 9 }, () => colRamp),
    geometry,
  );
  assertEquals([...cols.subarray(0, 16)], expected, "column direction");
});

test("assembleTiles: covers up to the end (truncating at the stride width alone would leave a gap)", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64).fill(7);
  const image = assembleTiles(
    Array.from({ length: 9 }, () => flat),
    geometry,
  );
  assertEquals(image.length, 16 * 16);
  assert(
    image.every((value) => value === 7),
    "uncovered pixels remain",
  );
});

test("assembleTiles: does not destroy the tiles passed in (the in-place blend works on a copy)", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const tiles = Array.from({ length: 9 }, (_, index) => new Float32Array(64).fill(index + 1));
  const before = tiles.map((tile) => [...tile]);
  assembleTiles(tiles, geometry);
  assertEquals(
    tiles.map((tile) => [...tile]),
    before,
  );
});

test("assembleTiles: rejects a mismatch between the tile count and the element count", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64);
  assertThrows(() => assembleTiles([flat], geometry), Error, "geometry's");
  assertThrows(
    () => assembleTiles([...Array.from({ length: 8 }, () => flat), new Float32Array(63)], geometry),
    Error,
    "first tile's",
  );
});

test("decodeTiled: runs every tile once in row-major order", async () => {
  const geometry = planVaeTiling([1, 1, 16, 16], [1, 1, 8, 8], [1, 1, 8, 8], 2);
  const visited: string[] = [];
  await decodeTiled(new Float32Array(256), geometry, (_tile, row, col) => {
    visited.push(`${row},${col}`);
    return Promise.resolve(new Float32Array(64));
  });
  assertEquals(visited, ["0,0", "0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2"]);
});

test("decodeTiled: takes ownership of the decode return values and folds them without an intervening copy", async () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const colCount = geometry.cols.starts.length;
  // The values differ per tile and within a tile (if both sides of an overlap were equal the blend
  // would be the identity, hiding both whether a copy exists and differences in how it folds).
  const makeTiles = () =>
    Array.from({ length: 9 }, (_, index) => {
      const tile = new Float32Array(64);
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) tile[y * 8 + x] = index * 100 + y * 10 + x;
      }
      return tile;
    });
  const handed = makeTiles();
  const pristine = makeTiles();

  const image = await decodeTiled(new Float32Array(2 * 256), geometry, (_tile, row, col) =>
    Promise.resolve(handed[row * colCount + col]),
  );

  // Taking ownership does not change the folding by a single bit compared with the public surface.
  const expected = assembleTiles(pristine, geometry);
  assertEquals(
    new Uint32Array(image.buffer, image.byteOffset, image.length),
    new Uint32Array(expected.buffer, expected.byteOffset, expected.length),
  );
  // The array passed in is the one that was folded = the ownership-transfer contract. If a copy
  // layer came back inside, this breaks.
  assert(
    handed.some((tile, index) => tile.some((value, at) => value !== pristine[index][at])),
    "the decode return value was not folded in place (a copy sits in between)",
  );
});
