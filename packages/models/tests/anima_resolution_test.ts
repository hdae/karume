// The resolution spelling and its accepted set.
//
// **A violation of the accepted set fails only deep inside execution** (rope base tables out of
// rows / an impossible tile layout / the upper bound of Dim("S")), so this file pins that the entry
// names each condition. Pure functions that need neither assets nor a GPU.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ANIMA_SPATIAL_COMPRESSION } from "../src/anima/dit-tokens.ts";
import {
  assertAcceptableResolution,
  formatResolution,
  MAX_DIT_TOKENS,
  MAX_RESOLUTION_SIDE,
  MIN_RESOLUTION_SIDE,
  parseResolution,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";
import { MIN_TILE_OVERLAP_LATENT, planTileAxis } from "../src/anima/tiling.ts";
import { ModelInputError } from "../src/errors.ts";

/** The candidate sides on the step grid (from the lower bound to the ceiling). */
const CANDIDATE_SIDES = ((): readonly number[] => {
  const sides: number[] = [];
  for (let side = MIN_RESOLUTION_SIDE; side <= MAX_RESOLUTION_SIDE; side += RESOLUTION_GRANULARITY)
    sides.push(side);
  return sides;
})();

/**
 * The measured **sides whose tile count changes from the old rule** under rounded equal-spacing
 * placement (ADR 0033 addendum 2026-09-02) (P-3 spike). `[side, old count, new count]`. The old
 * rule picked the count from the divisors of `extent − tile`, so only sides with few divisors blew
 * up to 60 or more tiles, and the eight sides above 10 (1456 / 1488 / 1584 / 1648 / 1680 / 1776 /
 * 1840 / 1936) were rejected at the entry.
 *
 * The 73 sides not in this table keep both the count and the placement of the old rule = **shipped
 * bytes unchanged** (the placements of the 1024² / 512² / 1344×768 family themselves are frozen as
 * literals by `anima_tiling_test.ts`).
 */
const RECOUNTED_SIDES: readonly (readonly [number, number, number])[] = [
  [1440, 5, 4],
  [1456, 60, 4],
  [1488, 62, 4],
  [1504, 5, 4],
  [1536, 5, 4],
  [1552, 6, 4],
  [1584, 68, 4],
  [1600, 5, 4],
  [1632, 5, 4],
  [1648, 72, 4],
  [1680, 74, 4],
  [1696, 5, 4],
  [1728, 5, 4],
  [1744, 8, 4],
  [1776, 80, 4],
  [1792, 5, 4],
  [1824, 5, 4],
  [1840, 84, 4],
  [1872, 6, 5],
  [1904, 7, 5],
  [1936, 90, 5],
  [1968, 8, 5],
  [2000, 7, 5],
  [2032, 6, 5],
];

/** The number of VAE tiles that cover the side (counted by running the split plan itself). */
const tilesOf = (side: number): number =>
  planTileAxis(
    side / ANIMA_SPATIAL_COMPRESSION,
    MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION,
    MIN_TILE_OVERLAP_LATENT,
  ).starts.length;

test("parseResolution: accepts WxH and the square shorthand (the round trip is the identity)", () => {
  assertEquals(parseResolution("1344x768"), { width: 1344, height: 768 });
  assertEquals(parseResolution("768x1344"), { width: 768, height: 1344 });
  assertEquals(parseResolution("512"), { width: 512, height: 512 });
  assertEquals(parseResolution("1024X1024"), { width: 1024, height: 1024 });
  for (const raw of ["512", "1024", "1344x768", "896x1152"]) {
    assertEquals(formatResolution(parseResolution(raw)), raw, `round trip: ${raw}`);
  }
});

test("parseResolution: rejects input that is not the spelling (no degrading to a default)", () => {
  for (const raw of ["", "1344*768", "1344x", "x768", "1344x768x1", "-512", "51 2", "1e3"]) {
    assertThrows(
      () => parseResolution(raw),
      ModelInputError,
      "is neither WxH nor",
      `spelling: ${JSON.stringify(raw)}`,
    );
  }
});

test("accepted set: rejects each of the four conditions by name", () => {
  // ① The step (multiple of 16) — with a multiple of 8 the latent is not divisible by patch 2.
  assertThrows(
    () => parseResolution("1352x768"),
    ModelInputError,
    `is not a multiple of ${RESOLUTION_GRANULARITY}`,
  );
  // ② The lower bound (the VAE tile decoder's latent 64 does not fit).
  assertThrows(
    () => parseResolution(`1024x${MIN_RESOLUTION_SIDE - 16}`),
    ModelInputError,
    "lower bound",
  );
  // ③ The ceiling (the row count of the rope base tables).
  assertThrows(
    () => parseResolution(`${MAX_RESOLUTION_SIDE + 16}x512`),
    ModelInputError,
    "upper bound",
  );
  // ④ The upper bound of S. With the current constants each side's ceiling implies S's upper bound,
  //   so
  // **a shape that exceeds it cannot be built** (2048×2048 = 16,384 is exactly the bound — the
  // relation between the constants itself is pinned by the check below).
  assertEquals(parseResolution("2048x2048"), { width: 2048, height: 2048 });
});

test("assertAcceptableResolution: the path that passes a struct is checked against the same four conditions", () => {
  // The manifest's `pipelineConfig.defaults.resolution` and the `generate` request go through here
  // (they bypass the spelling, so a check only on the spelling side would let them through).
  assertAcceptableResolution({ width: 1024, height: 1024 });
  assertThrows(
    () => assertAcceptableResolution({ width: 1024, height: 1000 }),
    ModelInputError,
    "multiple",
  );
  assertThrows(
    () => assertAcceptableResolution({ width: 256, height: 1024 }),
    ModelInputError,
    "lower bound",
  );
  assertThrows(
    () => assertAcceptableResolution({ width: 4096, height: 512 }),
    ModelInputError,
    "upper bound",
  );
  assertThrows(
    () => assertAcceptableResolution({ width: 1024.5, height: 1024 }),
    ModelInputError,
    "integer",
  );
});

test("accepted set: all four patterns (16:9 / 3:4, both orientations) are accepted with S=4,032", () => {
  // Pins with numbers, without literals, that **both orientations of each pair** are accepted and
  // that S lies inside Dim("S").
  const patterns = ["1344x768", "768x1344", "1152x896", "896x1152"];
  for (const raw of patterns) {
    const size = parseResolution(raw);
    const tokens = (size.width / RESOLUTION_GRANULARITY) * (size.height / RESOLUTION_GRANULARITY);
    assertEquals(tokens, 4032, `token length S of ${raw}`);
    assert(tokens <= MAX_DIT_TOKENS, `${raw}: S exceeds the upper bound of Dim("S")`);
  }
});

test("accepted set: rejects none of the sides on the step grid (the tile-count check was removed)", () => {
  // The old rule rejected at the entry the eight sides whose count jumps (a symptomatic check —
  // fixed at the root by P-3). The rejected set is checked to be empty on both axes, so if a
  // rejection came back anywhere it would break here.
  for (const axis of ["width", "height"] as const) {
    const rejected = CANDIDATE_SIDES.filter((side) => {
      try {
        assertAcceptableResolution(
          axis === "width" ? { width: side, height: 1024 } : { width: 1024, height: side },
        );
        return false;
      } catch {
        return true;
      }
    });
    assertEquals(rejected, [], `sides rejected on the ${axis} axis`);
  }
});

test("tile count: at most 5 on all 97 sides (no side whose count jumps exists any more)", () => {
  // This is the very reason the count check could be removed from the entry. `decodeTiled` holds
  // the decode output of every tile, so the bound on the count is the bound on host RAM — if a
  // jumping side came back (= the placement rule regressed), it would show up as an OOM rather than
  // a rejection. This makes it break here first.
  const worst = Math.max(...CANDIDATE_SIDES.map(tilesOf));
  assertEquals(
    worst,
    5,
    `maximum tile count per side (${worst * worst} tiles even when both sides are worst)`,
  );
});

test("tile count: freezes the measured new counts of the 24 sides that change from the old rule", () => {
  // The 73 unchanged sides match the old rule down to the placement = shipped bytes unchanged. The
  // changed sides are frozen **by value** (written as a formula it would copy `tilesOf` and the
  // check would be empty).
  const recounted = new Map(RECOUNTED_SIDES.map(([side, , after]) => [side, after]));
  for (const side of CANDIDATE_SIDES) {
    const expected = recounted.get(side);
    if (expected !== undefined) assertEquals(tilesOf(side), expected, `new count of ${side}px`);
  }
  // All eight sides whose old count exceeded the bound of 10 (= the sides the old rule rejected at
  // the entry) came back.
  const restored = RECOUNTED_SIDES.filter(([, before]) => before > 10).map(([side]) => side);
  assertEquals(restored, [1456, 1488, 1584, 1648, 1680, 1776, 1840, 1936], "restored sides");
  for (const side of restored) assertAcceptableResolution({ width: side, height: 1024 });
});

test("tile count: 1824×1248 is 4×3 = 12 tiles (the old rule gave 5×3 = 15)", () => {
  // The very target of the P-3 measurement. The tile count goes straight into the VAE decode time,
  // so the premise of the extrapolation is pinned with numbers (the A/B on a real GPU is on the e2e
  // side).
  assertAcceptableResolution({ width: 1824, height: 1248 });
  assertEquals(tilesOf(1824), 4, "width 1824px");
  assertEquals(tilesOf(1248), 3, "height 1248px");
});

test('constant relation: even using every side\'s full ceiling, S stays within the upper bound of Dim("S")', () => {
  // The S-bound branch of `resolution.ts` is **unreachable** with the current constants (each side
  // 2048 = grid 128 → S ≤ 128² = 16,384). The branch is kept as insurance for when the constants
  // move, so this pins "unreachable today" — it turns red the moment the side ceiling, the step or
  // the S bound moves.
  assert(
    (MAX_RESOLUTION_SIDE / RESOLUTION_GRANULARITY) ** 2 <= MAX_DIT_TOKENS,
    `the maximum S from the side ceiling, ${(MAX_RESOLUTION_SIDE / RESOLUTION_GRANULARITY) ** 2}, ` +
      `exceeds ${MAX_DIT_TOKENS} (the S-bound branch became reachable)`,
  );
});
