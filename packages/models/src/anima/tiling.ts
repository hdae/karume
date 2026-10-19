/**
 * **Fixed-tile** VAE decode (host glue only — the runtime is not involved by a single line).
 * ADR 0033.
 *
 * ## Why the host alone is enough
 *
 * The VAE decoder graph is **structurally invariant to the resolution** (measured: the weight
 * payloads for 512px and 1024px are sha256-identical; only the declared shapes of intermediate
 * values differ — measurement 2 of ADR 0038 §4). So tiling a tile decoder (input `[1,16,64,64]`)
 * covers every resolution. All that is needed is cutting out, blending and pasting tiles, which is
 * this module.
 *
 * ## Tile geometry = rounded equal-spacing layout (a deliberate deviation from diffusers)
 *
 * Upstream (`tiled_decode` in `autoencoder_kl_qwenimage.py`) scans with `range(0, H, stride)`, so
 * **the last tile is shorter**. Our tile decoder has **a fixed shape** and cannot take a short
 * tile. So the scan is changed to "snap the last tile's start to `extent − tile`". The starts
 * **divide the range between 0 and `extent − tile` equally, rounded**, and the spacing need not
 * divide evenly ({@link planTileAxis}) — the only constraint is "the overlap of each adjacent pair
 * is at least the lower bound".
 *
 * As a consequence, upstream's "truncate each tile to the spacing and concatenate → crop the whole"
 * **does not reach the end of the image** as is (the sum of spacings = `extent − tile < extent`).
 * The correct counterpart for snapped placement is the **region assignment** "tile i owns
 * `[starts[i], starts[i+1])`, only the last one `[starts[n−1], extent)`", which gives the same
 * split as upstream's crop + whole crop while covering up to the end ({@link assembleTiles}).
 *
 * ## The blend has the same form as upstream's linear ramp
 *
 * Same formula as `blend_v` / `blend_h`, same order (**the vertical blend with the tile above
 * first, the horizontal blend with the tile on the left after**), same **sample space**. Upstream
 * rewrites the tile array in place, so what affects a neighbour is **the already blended tile** —
 * the same here. There is one in-place body: {@link assembleTiles} (public surface) copies and then
 * calls it, and {@link decodeTiled} (the internal path that owns its arrays) calls it directly
 * without copying.
 *
 * ## The degenerate case (one tile) is **bit-identical** to the non-tiled path — MUST
 *
 * With `extent === tile` (512px = latent 64) there is one tile, no blend runs and the owned region
 * is the whole image. If this path differs from the non-tiled decode by even one bit, geometry or
 * pasting is wrong — a degenerate point that pins the identity **by structure**, not by numbers; a
 * unit test (the degenerate case of `assembleTiles` in `anima_tiling_test.ts`) checks it
 * Uint32-exactly.
 */

const f32 = Math.fround;

/**
 * Minimum overlap of adjacent tiles in latent units (= 64px in sample space).
 *
 * Matched to the same order as upstream's default blend width
 * (`tile_sample_min − tile_sample_stride` = 256−192 = 64px). The decoder has 12 3×3 convs + 3
 * upsample stages, so its effective receptive field in sample space is a few tens of px; below this
 * the ramp can no longer hide the seams.
 *
 * NOTE: `export` is so that the test (the table counting tile layouts on every side of the accepted
 * set) runs the split plan **with the same constant** (copying the default would let the test and
 * the actual split drift apart silently when only one side moves).
 */
export const MIN_TILE_OVERLAP_LATENT = 8;

/** Tile layout of one latent axis. */
type TileAxis = {
  /** Latent length of this axis. */
  readonly extent: number;
  /** Latent width of one tile (= the tile decoder's input shape). */
  readonly tile: number;
  /**
   * Start of each tile (latent, ascending, the last exactly `extent − tile`).
   *
   * MUST: do not hold the difference of starts (the spacing) as a field — it varies by up to 1
   * latent per pair, so folding it into one number silently disagrees with the layout. Derive it
   * from `starts` where needed ({@link blendExtentAt}).
   */
  readonly starts: readonly number[];
};

/** Tile layout of both axes and the latent ↔ sample scale. */
type TileGeometry = {
  /** Sample pixels per latent. */
  readonly scale: number;
  /** Latent channel count (sets the plane width of the cut-out). */
  readonly channels: number;
  /** Height axis. */
  readonly rows: TileAxis;
  /** Width axis. */
  readonly cols: TileAxis;
};

/**
 * Decides the rounded equal-spacing snapped layout of one axis.
 *
 * The count is the smallest value meeting "overlap at least `minOverlap`",
 * `ceil(span / (tile − minOverlap)) + 1` (`span = extent − tile`), and the starts are
 * `round(i · span / (count − 1))`. The rounding error stays within ±0.5, so **the spacing takes
 * only two values (`floor` and `ceil`)**, and the overlap of every pair satisfies
 * `tile − ceil(span / (count − 1)) ≥ minOverlap`.
 *
 * NOTE: the old rule (ADR 0033 decision 2) also required "the spacing divides evenly" and chose the
 * count from the divisors of `span`. On sides with few divisors the count jumped (1456px = latent
 * 182, span 118 = 2·59 → 60 tiles), and infeasible shapes had to be rejected at the entry. Dropping
 * divisibility breaks only "the same blend width for every pair", which is handled by holding it
 * per pair ({@link blendExtentAt}) — what uniformity really means is "overlap at least the lower
 * bound, per-pair difference at most 1 latent".
 */
export const planTileAxis = (extent: number, tile: number, minOverlap: number): TileAxis => {
  if (!Number.isInteger(extent) || !Number.isInteger(tile) || !Number.isInteger(minOverlap)) {
    throw new Error(
      `The tile layout is built from integers (extent=${extent} tile=${tile} overlap=${minOverlap})`,
    );
  }
  if (tile < 1) throw new Error(`Tile width ${tile} is below 1`);
  if (extent < tile) {
    throw new Error(
      `Latent length ${extent} is smaller than the tile width ${tile} (cannot be fed to the fixed-shape decoder)`,
    );
  }
  if (minOverlap < 0 || minOverlap >= tile) {
    throw new Error(
      `Minimum overlap ${minOverlap} is outside [0, ${tile}) (the overlap must be less than the tile width)`,
    );
  }
  const span = extent - tile;
  if (span === 0) {
    // Degenerate: one tile covers it. There is no pair, so no blend runs and pasting is a plain
    // copy (bit-identical to the non-tiled path — the MUST in the module doc).
    return { extent, tile, starts: [0] };
  }
  const count = Math.ceil(span / (tile - minOverlap)) + 1;
  const starts: number[] = [];
  for (let index = 0; index < count; index += 1) {
    starts.push(Math.round((index * span) / (count - 1)));
  }
  // The overlap bound follows from the count formula, but the derivation rests on a rounding-error
  // estimate that cannot be followed by eye. If it breaks, the ramp no longer hides the seams (a
  // silent error visible only in the picture), so it fails structurally.
  for (let index = 1; index < count; index += 1) {
    const overlap = tile - (starts[index] - starts[index - 1]);
    if (overlap < minOverlap) {
      throw new Error(
        `Tiles ${index - 1}/${index}: overlap ${overlap} is below the lower bound ${minOverlap}` +
          ` (latent ${extent} / tile ${tile} / start positions ${starts})`,
      );
    }
  }
  return { extent, tile, starts };
};

/**
 * Blend width of the pair `(index − 1, index)` (**sample space** — the same unit as upstream's
 * `blend_height` / `blend_width`). With rounded equal spacing it varies by up to 1 latent per pair.
 */
export const blendExtentAt = (axis: TileAxis, scale: number, index: number): number => {
  if (!Number.isInteger(index) || index < 1 || index >= axis.starts.length) {
    throw new RangeError(
      `Blend pair ${index} is out of range (${axis.starts.length} start positions = ${
        axis.starts.length - 1
      } pairs)`,
    );
  }
  return (axis.tile - (axis.starts[index] - axis.starts[index - 1])) * scale;
};

/**
 * Builds the geometry from the latent shape and the tile decoder's input/output shapes.
 *
 * `latentShape` is `[1,C,H,W]` (the latent the DiT emits), `tileShape` is the tile decoder's
 * **graph input** `[1,C,th,tw]`, and `sampleShape` its **graph output** `[1,3,th·s,tw·s]`. The
 * scale `s` is derived from the output shape so that callers do not put `8` as a literal.
 */
export const planVaeTiling = (
  latentShape: readonly number[],
  tileShape: readonly number[],
  sampleShape: readonly number[],
  minOverlap: number = MIN_TILE_OVERLAP_LATENT,
): TileGeometry => {
  if (latentShape.length !== 4 || tileShape.length !== 4 || sampleShape.length !== 4) {
    throw new Error(
      `Tiling assumes rank 4 (latent [${latentShape}] / tile input [${tileShape}] / tile output [${sampleShape}])`,
    );
  }
  if (latentShape[0] !== 1 || tileShape[0] !== 1) {
    throw new Error(`Tiling assumes batch=1 (latent B=${latentShape[0]} / tile B=${tileShape[0]})`);
  }
  if (latentShape[1] !== tileShape[1]) {
    throw new Error(
      `Latent channel count ${latentShape[1]} does not match the tile decoder's input ${tileShape[1]}`,
    );
  }
  const scale = sampleShape[2] / tileShape[2];
  if (!Number.isInteger(scale) || scale < 1 || sampleShape[3] !== tileShape[3] * scale) {
    throw new Error(
      `The tile decoder's scale differs between axes (input ${tileShape[2]}×${tileShape[3]} → output ${
        sampleShape[2]
      }×${sampleShape[3]})`,
    );
  }
  return {
    scale,
    channels: latentShape[1],
    rows: planTileAxis(latentShape[2], tileShape[2], minOverlap),
    cols: planTileAxis(latentShape[3], tileShape[3], minOverlap),
  };
};

/** Total number of tiles. */
export const tileCount = (geometry: TileGeometry): number =>
  geometry.rows.starts.length * geometry.cols.starts.length;

/**
 * Cuts the tile `[1,C,th,tw]` at `(row, col)` out of the latent `[1,C,H,W]` (a copy per plane).
 */
export const latentTile = (
  latents: Float32Array<ArrayBuffer>,
  geometry: TileGeometry,
  row: number,
  col: number,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, channels } = geometry;
  // Check integrality too — comparisons alone let 0.5 / NaN through, `starts[0.5]` becomes
  // undefined and a zero tile is returned silently.
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
  const sourcePlane = rows.extent * cols.extent;
  if (latents.length !== channels * sourcePlane) {
    throw new Error(
      `Latent element count ${latents.length} does not match [1,${channels},${rows.extent},${cols.extent}]`,
    );
  }
  const top = rows.starts[row];
  const left = cols.starts[col];
  const tilePlane = rows.tile * cols.tile;
  const out = new Float32Array(channels * tilePlane);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let y = 0; y < rows.tile; y += 1) {
      const from = channel * sourcePlane + (top + y) * cols.extent + left;
      out.set(latents.subarray(from, from + cols.tile), channel * tilePlane + y * cols.tile);
    }
  }
  return out;
};

// Third-party code notice. `blendVertical`, `blendHorizontal` and `assembleOwnedTiles` below are
// adapted from `AutoencoderKLQwenImage.blend_v`, `AutoencoderKLQwenImage.blend_h` and
// `AutoencoderKLQwenImage.tiled_decode` in huggingface/diffusers
// (`src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`, `diffusers==0.39.0`). The blend
// formulas and the blend order (above, then left) are carried over with each f32 operation written
// out; `assembleOwnedTiles` replaces the stride crop and the final crop by the region assignment
// (see the module doc). License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
// header of that file:
//
//   Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.

/**
 * Linear-ramp blend with the tile above (same form as upstream `blend_v`, **in place**).
 *
 * `b[y] = a[H−blend+y]·(1 − y/blend) + b[y]·(y/blend)`. MUST: round both weights to f32 before
 * multiplying (torch drops a Python float scalar to the tensor dtype before operating) — moving the
 * rounding point makes the last digit differ from the reference.
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

/** Linear-ramp blend with the tile on the left (same form as upstream `blend_h`, **in place**). */
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
 * The body of the assembly (**blends in place on the arrays passed in** — assumes the caller owns
 * them). The shape and count checks live here too, so the copying public surface
 * {@link assembleTiles} and the owning internal path {@link decodeTiled} pass the same checks.
 */
const assembleOwnedTiles = (
  working: Float32Array<ArrayBuffer>[],
  geometry: TileGeometry,
): Float32Array<ArrayBuffer> => {
  const { rows, cols, scale } = geometry;
  const rowCount = rows.starts.length;
  const colCount = cols.starts.length;
  if (working.length !== rowCount * colCount) {
    throw new Error(`${working.length} tiles do not match the geometry's ${rowCount}×${colCount}`);
  }
  const tileHeight = rows.tile * scale;
  const tileWidth = cols.tile * scale;
  const tilePlane = tileHeight * tileWidth;
  const channels = working[0].length / tilePlane;
  if (!Number.isInteger(channels) || channels < 1) {
    throw new Error(
      `Decode output element count ${working[0].length} is not divisible by the ${tileHeight}×${tileWidth} plane`,
    );
  }
  for (const [index, tile] of working.entries()) {
    if (tile.length !== channels * tilePlane) {
      throw new Error(
        `Tile ${index}: element count ${tile.length} does not match the first tile's ${channels * tilePlane}`,
      );
    }
  }

  // MUST: vertical (above) → horizontal (left) order (the same as upstream `tiled_decode`).
  // Swapping the order changes the order of the coefficient products where four tiles overlap at a
  // corner, and a cross-shaped streak appears at the seams.
  // MUST: look up the blend width **per pair** (with rounded equal spacing pairs differ by up to 1
  // latent). Reusing one number makes the ramp's edge stick out of the owned region at the shifted
  // seams.
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < colCount; col += 1) {
      const current = working[row * colCount + col];
      if (row > 0) {
        blendVertical(
          working[(row - 1) * colCount + col],
          current,
          channels,
          tileHeight,
          tileWidth,
          blendExtentAt(rows, scale, row),
        );
      }
      if (col > 0) {
        blendHorizontal(
          working[row * colCount + col - 1],
          current,
          channels,
          tileHeight,
          tileWidth,
          blendExtentAt(cols, scale, col),
        );
      }
    }
  }

  const height = rows.extent * scale;
  const width = cols.extent * scale;
  const image = new Float32Array(channels * height * width);
  for (let row = 0; row < rowCount; row += 1) {
    // Owned region = up to the next tile's start (only the last one up to the end). The
    // equal-spacing snapped version of upstream's "truncate to stride width + crop the whole",
    // covering up to the end (module doc).
    //
    // MUST: make the width exactly the owned region. Even if it is **too large** (e.g. always the
    // full tile width), later tiles overwrite it, so the result does not change by a bit and
    // numeric checks cannot detect it (measured: fault injection stays green). This "harmlessness"
    // depends on the row-major scan order, so it turns into a silent wrong value the moment pasting
    // is parallelised or reversed. The **too small** side (always the stride width) loses the end,
    // so it is detectable.
    const top = rows.starts[row] * scale;
    const spanRows =
      ((row + 1 < rowCount ? rows.starts[row + 1] : rows.extent) - rows.starts[row]) * scale;
    for (let col = 0; col < colCount; col += 1) {
      const left = cols.starts[col] * scale;
      const spanCols =
        ((col + 1 < colCount ? cols.starts[col + 1] : cols.extent) - cols.starts[col]) * scale;
      const tile = working[row * colCount + col];
      for (let channel = 0; channel < channels; channel += 1) {
        for (let y = 0; y < spanRows; y += 1) {
          const from = channel * tilePlane + y * tileWidth;
          image.set(
            tile.subarray(from, from + spanCols),
            channel * height * width + (top + y) * width + left,
          );
        }
      }
    }
  }
  return image;
};

/**
 * Blends decoded tiles (row-major) and assembles them into one image `[1,C,H·s,W·s]`.
 *
 * MUST: do not destroy the arrays passed in. The blend runs in place, so this public surface copies
 * before passing them to the body — to avoid the trap of the caller's Tensor contents changing
 * silently. The copy exists **only for this public contract**; the internal path that owns its
 * arrays ({@link decodeTiled}) calls the body directly without copying.
 */
export const assembleTiles = (
  decoded: readonly Float32Array<ArrayBuffer>[],
  geometry: TileGeometry,
): Float32Array<ArrayBuffer> =>
  assembleOwnedTiles(
    decoded.map((tile) => Float32Array.from(tile)),
    geometry,
  );

/**
 * Drives the tiled decode (the only impure seam = the `decode` callback).
 *
 * MUST: `decode` **reuses one session** (reopening per tile loads the weights once per tile, and
 * only the time jumps, against the purpose of tiling (keeping the memory peak at the 512px level)).
 *
 * MUST: the arrays `decode` returns are **handed over with ownership** — the caller neither keeps
 * nor reuses them (they are blended in place here as they are). Skipping one copy is directly a
 * peak saving, for all tiles (27MiB at 1024px / 75MiB at 2048px — docs/perf-ledger.md H-6).
 * `Session.run` outputs are freshly allocated per readback (`executor.ts` passes
 * `getMappedRange().slice(0)` to `hostTensor`), so the anima pipeline's callback meets this
 * contract as is.
 */
export const decodeTiled = async (
  latents: Float32Array<ArrayBuffer>,
  geometry: TileGeometry,
  decode: (
    tile: Float32Array<ArrayBuffer>,
    row: number,
    col: number,
  ) => Promise<Float32Array<ArrayBuffer>>,
): Promise<Float32Array<ArrayBuffer>> => {
  const decoded: Float32Array<ArrayBuffer>[] = [];
  for (let row = 0; row < geometry.rows.starts.length; row += 1) {
    for (let col = 0; col < geometry.cols.starts.length; col += 1) {
      decoded.push(await decode(latentTile(latents, geometry, row, col), row, col));
    }
  }
  return assembleOwnedTiles(decoded, geometry);
};
