/**
 * **Tiled decode** of Wan's video VAE (host side — ADR 0118 decision 2, step 5).
 *
 * The space of the chunk graphs (`vae-chunks.ts`) is a fixed latent `t×t` tile (the input shape of the
 * opened asset), so for the full frame (832×480 = latent 60×104) the host cuts tiles, runs the chunk
 * sequence on each, blends the overlaps and pastes them together (always tiled — the video version of
 * ADR 0033 / 0038 §4). The Python-side source of truth is `tools/export-recipes/wan/vae_tiling.py`
 * (which also writes the reference fixtures that the GPU comparison reads).
 *
 * ## Tile geometry = rounded evenly spaced snap placement (an intentional deviation from upstream)
 *
 * Upstream (diffusers `AutoencoderKLWan.tiled_decode`) scans with `range(0, H, stride)`, so the last
 * tile is short, which a fixed-shape chunk graph cannot take. The start positions **divide the range
 * between 0 and `extent − tile` evenly, rounded** (the same rule as Anima — ADR 0033 addendum P-3). The
 * count is the smallest that satisfies "every adjacent pair overlaps at least the lower bound"
 * ({@link planWanVaeTileAxis}). The source of truth for the lower bound is 64 px of output, derived in
 * latents by dividing by the spatial compression ({@link wanVaeMinTileOverlap} — ADR 0121 decision 6).
 * For Wan2.1 that is 64 ÷ 8 = 8, and 832×480 gives 12 tiles at rows 0 / 14 / 28 × columns 0 / 24 / 48
 * / 72. The tile edge and the scale are taken from the asset's declaration ({@link planWanVaeTiles} —
 * so a different tile edge needs only a new distribution. Decision 2).
 *
 * ## The blend has the same form as upstream's linear ramp; pasting assigns regions
 *
 * The same formula and order as `blend_v` / `blend_h` (**vertical blend with the tile above first,
 * horizontal blend with the tile to the left second**), in the same sample space, applied to every
 * frame and every channel. Upstream rewrites the tile array in place, so what affects a neighbour is
 * the already-blended tile — the same here. Pasting assigns "tile i owns
 * `[starts[i], starts[i+1])`, the last one up to the end" (the snap version of upstream's "trim to the
 * stride width + crop the whole" — trimming to the stride width would lose the end. ADR 0033 decision
 * 3).
 *
 * ## Tiles outside, chunks inside
 *
 * The cache is rebuilt from zero per tile ({@link decodeWanVaeTile} writes zeros at the head of each
 * tile — the same as `clear_cache` in upstream `tiled_decode`). The reverse nesting would hold as many
 * caches as there are tiles at once (decision 2). The one tile = one batch, read-back-frames-only
 * procedure is in the doc of `vae-chunks.ts`.
 *
 * ## MUST: the degenerate case (one tile) is bit-identical to the untiled chunk sequence
 *
 * When the latent is exactly one tile (`extent === tile`) there is one tile, no blend ever runs, and the
 * owned region is the whole. If this path differs from the chunk sequence ({@link decodeWanVaeTile}) by
 * even one bit, cutting or pasting is wrong — the real-GPU e2e checks it in Uint32 (ADR 0033 decision
 * 4).
 *
 * ## Clamp
 *
 * {@link decodeWanVaeTiled} returns values **before the clamp** (comparisons are made before the clamp
 * — differences in saturated elements are not hidden). The final frames are the values passed through
 * {@link clampWanVaeFrames} (after pasting, as upstream). For a generation with unpatchify (Wan2.2) the
 * order is pasting (in patchify space, before the clamp) → {@link unpatchifyWanVaeFrames}
 * → clamp (the same as upstream's `_decode` / `tiled_decode` — blending happens in patchify space).
 */

import type { GpuContext, Session } from "@karume/runtime";
import { decodeWanVaeTile, type WanVaeChunkCaches, type WanVaeChunkLayout } from "./vae-chunks.ts";

const f32 = Math.fround;

/**
 * The minimum overlap of adjacent tiles in output pixels. Equal to upstream's default blend width
 * `tile_sample_min − tile_sample_stride` (256 − 192 = 64 px). The same value as the Python side's
 * `vae_tiling.MIN_OVERLAP_PX` (the latent overlap is derived from it by {@link wanVaeMinTileOverlap} —
 * no latent value is held per generation).
 */
const MIN_OVERLAP_PX = 64;

/**
 * The spatial compression (final output pixels per latent) = the graph's input / output spatial ratio
 * `scale` × the host unpatchify factor `patchSize` (ADR 0121 decision 6). The graph's ratio comes from
 * the chunk graph declaration (`sampleTile / tile`), the unpatchify factor from the generation
 * descriptor (`vaePatchSize`).
 *
 * MUST: the latent size, the check that tiles cover it, and the overlap formula take the compression
 * from this one function (`wanSpatialCompression` in `tile-decode.ts` and {@link wanVaeMinTileOverlap}).
 * Taking the compression from the graph's ratio alone doubles the latent size for a generation with
 * unpatchify, and doubles the overlap too.
 */
export const wanVaeSpatialCompression = (scale: number, patchSize: number): number => {
  if (!Number.isInteger(scale) || scale < 1 || !Number.isInteger(patchSize) || patchSize < 1) {
    throw new Error(
      `Spatial compression factors are not positive integers (graph ratio ${scale}, unpatchify ${patchSize})`,
    );
  }
  return scale * patchSize;
};

/**
 * The minimum width by which adjacent tiles overlap in latents = {@link MIN_OVERLAP_PX} ÷ the spatial
 * compression ({@link wanVaeSpatialCompression}). The same formula and the same message as the Python
 * side's `vae_tiling.min_overlap_latent` (otherwise the frozen tile-plan tables of the two sides
 * disagree).
 *
 * MUST: a compression that does not divide evenly fails without rounding (rounding would silently move
 * the overlap away from 64 px).
 */
export const wanVaeMinTileOverlap = (scale: number, patchSize: number): number => {
  const compression = wanVaeSpatialCompression(scale, patchSize);
  if (MIN_OVERLAP_PX % compression !== 0) {
    throw new Error(
      `Overlap ${MIN_OVERLAP_PX} px is not divisible by the spatial compression ${compression}` +
        ` (graph ratio ${scale} × unpatchify ${patchSize})`,
    );
  }
  return MIN_OVERLAP_PX / compression;
};

/** The tile placement along one latent axis. */
export type WanVaeTileAxis = {
  /** The full latent length along this axis. */
  readonly extent: number;
  /** The latent width of one tile (= the input shape of the chunk graph). */
  readonly tile: number;
  /**
   * The start position of each tile (latent, ascending, the last exactly `extent − tile`).
   *
   * MUST: do not hold the difference of start positions (the spacing) as a field — it moves by up to
   * one latent per pair, so folding it into one number silently disagrees with the placement. Derive
   * it from `starts` where needed ({@link wanVaeBlendExtentAt}).
   */
  readonly starts: readonly number[];
};

/** The tile placement along both axes of a latent `[C, F, H, W]`, and the latent ↔ sample scale. */
export type WanVaeTilePlan = {
  /** The latent channel count (decides the number of planes to cut). */
  readonly latentChannels: number;
  /**
   * The channel count at the chunk graph's exit (used in pasting to check that the number of tile
   * planes is a multiple of it — blending and pasting loop over planes, without separating channels
   * from frames).
   */
  readonly sampleChannels: number;
  /** Sample pixels per latent. */
  readonly scale: number;
  /** The height axis. */
  readonly rows: WanVaeTileAxis;
  /** The width axis. */
  readonly cols: WanVaeTileAxis;
};

/**
 * Rounded evenly spaced snap placement along one axis (the same rule as the Python side's
 * `vae_tiling.plan_tile_axis`).
 *
 * The count is the smallest value that satisfies "overlap at least `minOverlap`",
 * `ceil(span / (tile − minOverlap)) + 1` (`span = extent − tile`); the start positions are
 * `round(i · span / (count − 1))` (`Math.round` = 0.5 rounds up — the Python side uses an integer
 * formula that rounds the same way). The rounding error stays within ±0.5, so the spacing takes only
 * two values.
 *
 * `minOverlap` is required, in latent units (no default — each generation's value is derived by
 * {@link wanVaeMinTileOverlap}, so a forgotten argument is not silently planned with Wan2.1's value).
 */
export const planWanVaeTileAxis = (
  extent: number,
  tile: number,
  minOverlap: number,
): WanVaeTileAxis => {
  if (!Number.isInteger(extent) || !Number.isInteger(tile) || !Number.isInteger(minOverlap)) {
    throw new Error(
      `Tile placement must use integers (extent=${extent} tile=${tile} overlap=${minOverlap})`,
    );
  }
  if (tile < 1) throw new Error(`Tile width ${tile} is less than 1`);
  if (extent < tile) {
    throw new Error(
      `Latent extent ${extent} is smaller than tile width ${tile} (cannot be fed to the fixed-shape chunk graph)`,
    );
  }
  if (minOverlap < 0 || minOverlap >= tile) {
    throw new Error(
      `Minimum overlap ${minOverlap} is outside [0, ${tile}) (the overlap must be less than the tile width)`,
    );
  }
  const span = extent - tile;
  if (span === 0) {
    // Degenerate: one tile covers it. With no pair no blend runs, and pasting is a plain copy (the
    // module doc's MUST — bit-identical to the untiled chunk sequence).
    return { extent, tile, starts: [0] };
  }
  const count = Math.ceil(span / (tile - minOverlap)) + 1;
  const starts: number[] = [];
  for (let index = 0; index < count; index += 1) {
    starts.push(Math.round((index * span) / (count - 1)));
  }
  // The overlap lower bound follows from the count formula, but the derivation relies on bounding the
  // rounding error and cannot be followed by eye. If it broke, the ramp would no longer hide the seams
  // (a silent error that shows only in the picture), so it is enforced structurally.
  for (let index = 1; index < count; index += 1) {
    const overlap = tile - (starts[index] - starts[index - 1]);
    if (overlap < minOverlap) {
      throw new Error(
        `Tiles ${index - 1}/${index} overlap by ${overlap}, below the minimum ${minOverlap}` +
          ` (latent ${extent} / tile ${tile} / start positions ${starts})`,
      );
    }
  }
  return { extent, tile, starts };
};

/**
 * The blend width of the pair `(index − 1, index)` (**sample space** — the same unit as upstream's
 * `blend_height` / `blend_width`). The spacing is rounded-even, so it moves by up to one latent per
 * pair.
 */
export const wanVaeBlendExtentAt = (axis: WanVaeTileAxis, scale: number, index: number): number => {
  if (!Number.isInteger(index) || index < 1 || index >= axis.starts.length) {
    throw new RangeError(
      `Blend pair ${index} is out of range (${axis.starts.length} start positions = ${
        axis.starts.length - 1
      } pairs)`,
    );
  }
  return (axis.tile - (axis.starts[index] - axis.starts[index - 1])) * scale;
};

/** Length of tile `index`'s owned region (latent) = up to the next tile's start; the last to the end. */
const regionOf = (axis: WanVaeTileAxis, index: number): number =>
  (index + 1 < axis.starts.length ? axis.starts[index + 1] : axis.extent) - axis.starts[index];

/**
 * The tile plan for the latent space `height × width`. The tile edge, scale and channel counts are
 * taken from the chunk graph declaration (`layout.tile` and `layout.sampleTile / layout.tile` — no 32
 * or 8 as a literal in the caller). `minOverlap` is required in latent units, as in
 * {@link planWanVaeTileAxis} (plans for a generation request go through `planWanGenerationTiles` in
 * `tile-decode.ts`).
 */
export const planWanVaeTiles = (
  layout: Pick<WanVaeChunkLayout, "latentChannels" | "tile" | "sampleTile" | "sampleChannels">,
  height: number,
  width: number,
  minOverlap: number,
): WanVaeTilePlan => {
  const scale = layout.sampleTile / layout.tile;
  if (!Number.isInteger(scale) || scale < 1) {
    throw new Error(`Tile scale ${layout.sampleTile} / ${layout.tile} is not a positive integer`);
  }
  return {
    latentChannels: layout.latentChannels,
    sampleChannels: layout.sampleChannels,
    scale,
    rows: planWanVaeTileAxis(height, layout.tile, minOverlap),
    cols: planWanVaeTileAxis(width, layout.tile, minOverlap),
  };
};

/** The total number of tiles. */
export const wanVaeTileCount = (plan: WanVaeTilePlan): number =>
  plan.rows.starts.length * plan.cols.starts.length;

/** The frame count F of a latent `[C, F, H, W]` (fails loudly if the length does not match the plan). */
const latentFrames = (plan: WanVaeTilePlan, latents: Float32Array<ArrayBuffer>): number => {
  const plane = plan.latentChannels * plan.rows.extent * plan.cols.extent;
  const frames = latents.length / plane;
  if (!Number.isInteger(frames) || frames < 1) {
    throw new Error(
      `Latent element count ${latents.length} does not match [${plan.latentChannels},F,${plan.rows.extent},${plan.cols.extent}]`,
    );
  }
  return frames;
};

/**
 * Cuts tile `(row, col)` `[C, F, t, t]` out of the latent `[C, F, H, W]` (a copy per plane — every one
 * of the `C·F` planes is `H × W`).
 */
export const wanVaeLatentTile = (
  plan: WanVaeTilePlan,
  latents: Float32Array<ArrayBuffer>,
  row: number,
  col: number,
): Float32Array<ArrayBuffer> => {
  const { rows, cols } = plan;
  // Integers are checked too — a range comparison alone lets 0.5 / NaN through, `starts[0.5]` becomes
  // undefined, and a zero tile is silently returned.
  if (
    !Number.isInteger(row) ||
    !Number.isInteger(col) ||
    row < 0 ||
    row >= rows.starts.length ||
    col < 0 ||
    col >= cols.starts.length
  ) {
    throw new RangeError(
      `Tile (${row}, ${col}) is out of range (${rows.starts.length}×${cols.starts.length})`,
    );
  }
  const planes = plan.latentChannels * latentFrames(plan, latents);
  const sourcePlane = rows.extent * cols.extent;
  const tilePlane = rows.tile * cols.tile;
  const top = rows.starts[row];
  const left = cols.starts[col];
  const out = new Float32Array(planes * tilePlane);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let y = 0; y < rows.tile; y += 1) {
      const from = plane * sourcePlane + (top + y) * cols.extent + left;
      out.set(latents.subarray(from, from + cols.tile), plane * tilePlane + y * cols.tile);
    }
  }
  return out;
};

// Third-party code notice. `blendVertical`, `blendHorizontal` and `assembleOwnedTiles` below are
// adapted from `AutoencoderKLWan.blend_v`, `AutoencoderKLWan.blend_h` and
// `AutoencoderKLWan.tiled_decode` in huggingface/diffusers
// (`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`, `diffusers==0.39.0`). The blend
// formulas and the blend order (above, then left) are carried over with each f32 operation written
// out; `assembleOwnedTiles` replaces the stride crop and the final crop by the region assignment
// (see the module doc). License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
// header of that file:
//
//   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.

/**
 * Linear-ramp blend with the tile above (the same form as upstream `blend_v`, **in place**).
 *
 * `b[y] = a[H−blend+y]·(1 − y/blend) + b[y]·(y/blend)`. MUST: round the two weights to f32 first, then
 * multiply (torch casts a Python float scalar to the tensor dtype before the operation) — moving the
 * rounding point disagrees with the reference in the last digit.
 */
const blendVertical = (
  above: Float32Array<ArrayBuffer>,
  current: Float32Array<ArrayBuffer>,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let y = 0; y < blend; y += 1) {
    const weightAbove = f32(1 - y / blend);
    const weightCurrent = f32(y / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      const from = base + (height - blend + y) * width;
      const to = base + y * width;
      for (let x = 0; x < width; x += 1) {
        current[to + x] = f32(
          f32(above[from + x] * weightAbove) + f32(current[to + x] * weightCurrent),
        );
      }
    }
  }
};

/** Linear-ramp blend with the tile to the left (the same form as upstream `blend_h`, **in place**). */
const blendHorizontal = (
  left: Float32Array<ArrayBuffer>,
  current: Float32Array<ArrayBuffer>,
  planes: number,
  height: number,
  width: number,
  blend: number,
): void => {
  for (let x = 0; x < blend; x += 1) {
    const weightLeft = f32(1 - x / blend);
    const weightCurrent = f32(x / blend);
    for (let plane = 0; plane < planes; plane += 1) {
      const base = plane * height * width;
      for (let y = 0; y < height; y += 1) {
        const from = base + y * width + (width - blend + x);
        const to = base + y * width + x;
        current[to] = f32(f32(left[from] * weightLeft) + f32(current[to] * weightCurrent));
      }
    }
  }
};

/**
 * The body of pasting (**blends in place on the given arrays** — assumes the caller owns them). Returns
 * `[Cs, F', H·s, W·s]` (Cs is the plan's `sampleChannels`, F' the tiles' frame count).
 */
const assembleOwnedTiles = (
  working: readonly Float32Array<ArrayBuffer>[],
  plan: WanVaeTilePlan,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, scale } = plan;
  const rowCount = rows.starts.length;
  const colCount = cols.starts.length;
  if (working.length !== rowCount * colCount) {
    throw new Error(`${working.length} tiles do not match the plan's ${rowCount}×${colCount}`);
  }
  const tileHeight = rows.tile * scale;
  const tileWidth = cols.tile * scale;
  const tilePlane = tileHeight * tileWidth;
  const planes = working[0].length / tilePlane;
  if (!Number.isInteger(planes) || planes < 1 || planes % plan.sampleChannels !== 0) {
    throw new Error(
      `Tile element count ${
        working[0].length
      } does not match [${plan.sampleChannels},F,${tileHeight},${tileWidth}]`,
    );
  }
  for (const [index, tile] of working.entries()) {
    if (tile.length !== planes * tilePlane) {
      throw new Error(
        `Tile ${index} element count ${tile.length} differs from the first tile's ${planes * tilePlane}`,
      );
    }
  }

  // MUST: vertical (above) → horizontal (left), the order of upstream `tiled_decode`. Swapping them
  // changes the order of the coefficient products where the four corner tiles overlap (the unit test's
  // analytic corner solution breaks).
  // MUST: take the blend width per pair (the spacing is rounded-even, so pairs differ by up to one
  // latent).
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < colCount; col += 1) {
      const current = working[row * colCount + col];
      if (row > 0) {
        blendVertical(
          working[(row - 1) * colCount + col],
          current,
          planes,
          tileHeight,
          tileWidth,
          wanVaeBlendExtentAt(rows, scale, row),
        );
      }
      if (col > 0) {
        blendHorizontal(
          working[row * colCount + col - 1],
          current,
          planes,
          tileHeight,
          tileWidth,
          wanVaeBlendExtentAt(cols, scale, col),
        );
      }
    }
  }

  const height = rows.extent * scale;
  const width = cols.extent * scale;
  const out = new Float32Array(planes * height * width);
  for (let row = 0; row < rowCount; row += 1) {
    // MUST: the width is exactly the owned region. Too large (always the full tile width) still leaves
    // the values unchanged under a row-major scan because later tiles overwrite, so no numeric check
    // can detect it (ADR 0033 detection limits, item 1) — that harmlessness depends on the scan order,
    // so it turns into silently wrong values the moment pasting is parallelized or reversed.
    const top = rows.starts[row] * scale;
    const spanRows = regionOf(rows, row) * scale;
    for (let col = 0; col < colCount; col += 1) {
      const left = cols.starts[col] * scale;
      const spanCols = regionOf(cols, col) * scale;
      const tile = working[row * colCount + col];
      for (let plane = 0; plane < planes; plane += 1) {
        for (let y = 0; y < spanRows; y += 1) {
          const from = plane * tilePlane + y * tileWidth;
          out.set(
            tile.subarray(from, from + spanCols),
            plane * height * width + (top + y) * width + left,
          );
        }
      }
    }
  }
  return out;
};

/**
 * Blends the decoded tiles (row-major, each `[Cs, F', s, s]`) and pastes them into
 * `[Cs, F', H·s, W·s]` (still before the clamp; Cs is the plan's `sampleChannels`).
 *
 * MUST: do not destroy the given arrays (blending is in place, so it runs on copies).
 * {@link decodeWanVaeTiled}, which owns its arrays, calls the body without copying.
 */
export const assembleWanVaeTiles = (
  tiles: readonly Float32Array<ArrayBuffer>[],
  plan: WanVaeTilePlan,
): Float32Array<ArrayBuffer> =>
  assembleOwnedTiles(
    tiles.map((tile) => Float32Array.from(tile)),
    plan,
  );

/**
 * Decodes every tile in row-major order (**tiles outside, chunks inside** — per tile,
 * {@link decodeWanVaeTile} runs the cache from zero). Returns each tile's pre-clamp
 * `[Cs, 1 + 4(F−1), s, s]` (new arrays owned by the caller).
 *
 * `latents` is the de-normalized `[C, F, H, W]`. If the plan's tile edge, scale or channel counts
 * (latent and exit) disagree with the opened asset (`caches.layout`), it fails loudly.
 *
 * `onTile` is awaited each time the decode of one tile settles (`tile` is 1-based, row-major — the
 * hook for the pipeline's progress and diagnostics. If it throws, the remaining tiles are not run and
 * the error is rethrown).
 *
 * MUST: do not issue another batch or run on the same device concurrently (each tile's batch holds the
 * device's span lock — `decodeWanVaeTile`).
 */
export const decodeWanVaeTiles = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  plan: WanVaeTilePlan,
  latents: Float32Array<ArrayBuffer>,
  onTile?: (tile: number) => void | Promise<void>,
): Promise<Float32Array<ArrayBuffer>[]> => {
  const { layout } = caches;
  if (
    plan.latentChannels !== layout.latentChannels ||
    plan.rows.tile !== layout.tile ||
    plan.cols.tile !== layout.tile ||
    plan.scale * layout.tile !== layout.sampleTile ||
    plan.sampleChannels !== layout.sampleChannels
  ) {
    throw new Error(
      `Tile plan (C ${plan.latentChannels}, tile ${plan.rows.tile}×${plan.cols.tile}, scale ${plan.scale}` +
        `, output ${plan.sampleChannels} channels) differs from the asset (C ${layout.latentChannels}, tile ${layout.tile}` +
        `, sample ${layout.sampleTile}, output ${layout.sampleChannels} channels)`,
    );
  }
  // The latent length is checked before running the GPU (cutting each tile would also fail, but only
  // after paying for the decode of the earlier tiles).
  latentFrames(plan, latents);
  const decoded: Float32Array<ArrayBuffer>[] = [];
  for (let row = 0; row < plan.rows.starts.length; row += 1) {
    for (let col = 0; col < plan.cols.starts.length; col += 1) {
      decoded.push(
        await decodeWanVaeTile(gpu, sessions, caches, wanVaeLatentTile(plan, latents, row, col)),
      );
      await onTile?.(decoded.length);
    }
  }
  return decoded;
};

/**
 * The body of the tiled decode: decodes every tile, blends and pastes them, giving the **pre-clamp**
 * `[Cs, 1 + 4(F−1), H·s, W·s]`. The final frames go through {@link clampWanVaeFrames}. `onTile` is the
 * same as in {@link decodeWanVaeTiles}.
 */
export const decodeWanVaeTiled = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  plan: WanVaeTilePlan,
  latents: Float32Array<ArrayBuffer>,
  onTile?: (tile: number) => void | Promise<void>,
): Promise<Float32Array<ArrayBuffer>> =>
  assembleOwnedTiles(await decodeWanVaeTiles(gpu, sessions, caches, plan, latents, onTile), plan);

/**
 * Patchify space `[Cs, F, h, w]` → `[Cs/p², F, h·p, w·p]` (the same ordering as upstream diffusers
 * `unpatchify` — `view(c, r, q, f, h, w).permute(c, f, h, q, w, r)`, r = q = p). The index mapping is
 * `out[c, f, y·p + dy, x·p + dx] = in[c·p² + dx·p + dy, f, y, x]` (within a channel group the width
 * offset dx is the major index and the height offset dy the minor).
 *
 * p = 1 **returns the input as is** (no copy — the same as upstream's `if patch_size == 1: return x`).
 * p > 1 only copies into a new array with no floating-point arithmetic (values are kept bit for bit).
 *
 * Mismatches of shape, factor or length fail loudly (checked even for p = 1).
 */
export const unpatchifyWanVaeFrames = (
  frames: Float32Array<ArrayBuffer>,
  shape: readonly [number, number, number, number],
  patchSize: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(patchSize) || patchSize < 1) {
    throw new Error(`unpatchify patch size ${patchSize} is not a positive integer`);
  }
  if (shape.some((dim) => !Number.isInteger(dim) || dim < 1)) {
    throw new Error(
      `unpatchify input shape [${shape.join(", ")}] has a dimension that is not a positive integer`,
    );
  }
  const [channels, count, height, width] = shape;
  const area = patchSize * patchSize;
  if (channels % area !== 0) {
    throw new Error(
      `unpatchify input channel count ${channels} is not divisible by patch size ${patchSize}² = ${area}`,
    );
  }
  const elements = channels * count * height * width;
  if (frames.length !== elements) {
    throw new Error(
      `unpatchify input element count ${frames.length} does not match [${shape.join(", ")}] (${elements})`,
    );
  }
  if (patchSize === 1) return frames;
  const outChannels = channels / area;
  const plane = height * width;
  const out = new Float32Array(elements);
  // Loop in the order that makes output rows contiguous (c → f → y → dy → x → dx), so the write
  // position advances by one.
  let to = 0;
  for (let channel = 0; channel < outChannels; channel += 1) {
    for (let frame = 0; frame < count; frame += 1) {
      for (let y = 0; y < height; y += 1) {
        for (let dy = 0; dy < patchSize; dy += 1) {
          for (let x = 0; x < width; x += 1) {
            for (let dx = 0; dx < patchSize; dx += 1) {
              const source = channel * area + dx * patchSize + dy;
              out[to] = frames[(source * count + frame) * plane + y * width + x];
              to += 1;
            }
          }
        }
      }
    }
  }
  return out;
};

/**
 * Clamps the frames to `[-1, 1]` in place (the same as upstream `torch.clamp(min=-1, max=1)` — NaN
 * stays NaN, `-0` stays `-0`).
 */
export const clampWanVaeFrames = (frames: Float32Array<ArrayBuffer>): void => {
  for (let index = 0; index < frames.length; index += 1) {
    const value = frames[index];
    if (value > 1) frames[index] = 1;
    else if (value < -1) frames[index] = -1;
  }
};
