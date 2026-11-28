/**
 * The code side of the grid-fill rule of the GEMM (ADR 0157 — the table field `gemmGrid` of
 * src/kernels/geometry-profile.ts): the ladder the rule walks, the two shape constants a derivation
 * writes into a table's field beside the device's two integers, and the geometry sweep's `grid` set
 * that times the ladder (cases.ts's `GRID_CASES`, harness.ts's `sweepGridCase`).
 *
 * The ladder, the K floor and the padding bound are the recon's of 2026-11-28 (125 shape classes of
 * the default presets on the RTX 3080 Ti — docs/research/2026-11-28-gemm-grid-fill.md): five tiles
 * that each measured fastest somewhere in the recon, in strictly decreasing area; the K floor and the
 * padding bound that the fit kept best on every device integer pair it tried. A table carries them in
 * its field, so a change here does not change a registered table; it changes what the next derivation
 * writes and what the next grid sweep times.
 *
 * MUST: nothing is computed at import time (constants of plain data only).
 * SPEC: docs/spec/geometry-profiles.md#the-grid-fill-rule-of-the-gemm
 */

import type { GemmGeometry } from "../kernels/gemm-geometry.ts";

/** The candidate set of the geometry sweep that times the ladder on the grid cases. */
export const GRID_SET = "grid";

/**
 * The ladder, largest first: 128 × 128 (r8 × 8, 16 × 16 threads — the default geometry), 64 × 64
 * (r4 × 4, 16 × 16), 64 × 32 (r4 × 4, 8 × 16 — the default's medium-M bucket), 32 × 32 (r2 × 4,
 * 8 × 16) and 16 × 16 (r1 × 4, 4 × 16 — the default's small-M bucket). The default profile's three
 * bucket tiles are on it, so a grid case's default row is one of its rows.
 */
export const GEMM_GRID_LADDER: readonly GemmGeometry[] = [
  { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16 },
  { regM: 2, regN: 4, wgX: 8, wgY: 16 },
  { regM: 1, regN: 4, wgX: 4, wgY: 16 },
];

/**
 * The least K at which the rule keeps the largest tile. Below it 128 × 128's per-workgroup fill of
 * its shared tiles is not amortised over the K loop (the recon's short-K bmm, K 32 to 72, took ×0.36
 * to ×0.62 of 128 × 128's time with smaller tiles even at 432 to 5,808 workgroups).
 */
export const GEMM_GRID_MIN_K = 256;

/**
 * The largest padding (tile area over output area) at which the rule takes a tile: a shape that pads
 * M or N by 2 to 4 times wastes that share of the tile's work.
 */
export const GEMM_GRID_MAX_PADDING = 1.5;
