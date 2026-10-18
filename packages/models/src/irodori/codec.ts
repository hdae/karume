/**
 * **Tiling** of the codec decoder (host glue only — not a single line of the runtime is involved).
 *
 * ## Why tiles are needed
 *
 * The decoder decodes the latent `[1,S,32]` into `[1,1,1920S]`. At S = 750 (30 s) the intermediate
 * tensors reach 553MB, and **on a machine with the default 128MiB `maxStorageBufferBindingSize`,
 * allocation fails at S ≥ 183**. Tiles exist for this portability; on a machine with a high
 * allocation ceiling a single decode goes through as is (`tileFrames` is a performance knob, not
 * something that changes the result).
 *
 * ## Why discarding the halo gives bit identity
 *
 * The decoder's main path has no causal layer at all; every layer is either symmetric padding or a
 * `conv_transpose1d` with exactly `L·stride`, so it is **translation equivariant**. The measured
 * receptive field per side is 13,793 samples (= 7.19 latent frames), and the 8 frames of
 * {@link IrodoriPipelineConfig.codecHaloFrames} (15,360 samples) cover it. Karume's `conv1d` /
 * `conv_transpose1d` are gather-form kernels that reduce each output element in a fixed order
 * (`(ic,k)` ascending), so the same input window gives **the same reduction order too**, and the
 * kept span does not differ by a single bit from a full-length run. The real-GPU check
 * (`packages/e2e/tests/e2e_irodori_codec_test.ts`) measures this in Uint32 every time.
 *
 * The left end of the first tile and the right end of the last tile are **true boundaries**, so
 * they need no halo (the graph's zero padding plays the same role as at the ends of a full-length
 * run).
 *
 * ## Placement is evenly spaced + a snapped last tile (the same convention as `anima/tiling.ts`)
 *
 * To make **every tile's decode length `tileFrames`**, the start of the last tile snaps to
 * `frames − tileFrames`. With identical shapes the Session's `PreparedExecutionPlan` (ADR 0042)
 * keeps applying across tiles — a single tile with a different shape redoes the derivation for
 * that tile.
 */

import { ModelInputError } from "../errors.ts";

/**
 * Default of `tileFrames` (= the maximum number of latent frames sent through one decode).
 *
 * A size that goes through even on a machine whose `maxStorageBufferBindingSize` is only the
 * default 128MiB (one below the measured S = 183 where allocation fails). On a machine with a high
 * allocation ceiling S ≤ 750 goes through in one pass, so there passing a large value as
 * {@link IrodoriGenerateRequest.codecTileFrames} collapses the tiles to one.
 */
export const DEFAULT_CODEC_TILE_FRAMES = 182;

/**
 * One tile (start and length in latent frames — the 1920-sample grid lines up through every layer).
 */
export type CodecTile = {
  /** Start frame of the decoded span. */
  readonly start: number;
  /** Length of the decoded span (frames). */
  readonly length: number;
  /** Start of the kept span (frames relative to the tile start — the width of the left halo). */
  readonly offset: number;
  /** Length of the kept span (frames). The sum over all tiles is S. */
  readonly take: number;
};

/** Input of {@link planCodecTiles} (both in latent frames). */
type CodecTilePlanOptions = {
  /** Maximum number of frames sent through one decode. */
  readonly tileFrames: number;
  /** Context added on both sides of the kept span ({@link IrodoriPipelineConfig.codecHaloFrames}). */
  readonly haloFrames: number;
};

/**
 * Acceptance set of `tileFrames` (an integer larger than 2 halos).
 *
 * NOTE: `export` so that **the entry of generation** (the request check in `pipeline.ts`) rejects
 * with the same set first. {@link planCodecTiles} is called after latent generation, so with the
 * check only here, it would fail after loading 1.26GB of weights and running every DiT step. The
 * acceptance set exists only once (writing the condition on both sides always ends up diverging).
 */
export const assertCodecTileFrames = (tileFrames: number, haloFrames: number): void => {
  if (!Number.isInteger(haloFrames) || haloFrames < 0) {
    throw new Error(`planCodecTiles: haloFrames ${haloFrames} が非負整数でない`);
  }
  if (!Number.isInteger(tileFrames) || tileFrames <= 2 * haloFrames) {
    // `tileFrames` is the caller's request knob (`IrodoriGenerateRequest.codecTileFrames`), so
    // fixing the input makes it pass = the 400 side in ADR 0107's classification. The `haloFrames`
    // just above is a number the distribution declares, which the caller cannot fix, so it stays a
    // plain `Error`.
    throw new ModelInputError(
      `planCodecTiles: tileFrames ${tileFrames} が halo 2 枚ぶん（${2 * haloFrames}）` +
        "より大きい整数でない（採用できるフレームが 1 枚も残らない）",
    );
  }
};

/**
 * Splits the latent length S into a tile sequence whose kept spans line up without gaps.
 *
 * `frames <= tileFrames` gives one tile (= the same run as a single decode). Otherwise
 * `step = tileFrames − 2·haloFrames` frames are kept at a time, and the window with a halo added on
 * both sides is decoded. At an end where the window would stick out of the latent the halo
 * shrinks, but that is a true boundary, so the context is not missing but **not needed**.
 */
export const planCodecTiles = (
  frames: number,
  options: CodecTilePlanOptions,
): readonly CodecTile[] => {
  const { tileFrames, haloFrames } = options;
  if (!Number.isInteger(frames) || frames <= 0) {
    throw new Error(`planCodecTiles: frames ${frames} が正の整数でない`);
  }
  assertCodecTileFrames(tileFrames, haloFrames);
  if (frames <= tileFrames) return [{ start: 0, length: frames, offset: 0, take: frames }];

  const step = tileFrames - 2 * haloFrames;
  const tiles: CodecTile[] = [];
  for (let takeStart = 0; takeStart < frames; takeStart += step) {
    const takeEnd = Math.min(frames, takeStart + step);
    // Move left by the halo, and snap inward only when it would stick out of either end of the
    // latent.
    const start = Math.min(Math.max(0, takeStart - haloFrames), frames - tileFrames);
    tiles.push({
      start,
      length: tileFrames,
      offset: takeStart - start,
      take: takeEnd - takeStart,
    });
  }
  assertTileContext(tiles, frames, haloFrames);
  return tiles;
};

/** Decodes one tile (`[1,length,latentDim]` → a raw waveform `[1,1,hopLength·length]`). */
type CodecTileRunner = (
  latent: Float32Array<ArrayBuffer>,
  frames: number,
) => Promise<Float32Array<ArrayBuffer>>;

/** The geometry {@link decodeTiles} needs (2 fields of `pipelineConfig` + the plan). */
type CodecDecodeOptions = {
  readonly latentDim: number;
  readonly hopLength: number;
  readonly tiles: readonly CodecTile[];
};

/**
 * Decodes the tiles in order and stitches them into a waveform of **the full length**.
 *
 * `run` is injected so that the real-GPU check can measure "this stitching is bit-identical to a
 * single decode" on **this very function** (a test that copied the same loop would be green when
 * only the copy is right and the real one is wrong).
 */
export const decodeTiles = async (
  latent: Float32Array<ArrayBuffer>,
  options: CodecDecodeOptions,
  run: CodecTileRunner,
): Promise<Float32Array<ArrayBuffer>> => {
  const { latentDim, hopLength, tiles } = options;
  if (latent.length % latentDim !== 0) {
    throw new Error(
      `decodeTiles: latent の要素数 ${latent.length} が latentDim ${latentDim} で割れない`,
    );
  }
  const frames = latent.length / latentDim;
  const waveform = new Float32Array(frames * hopLength) as Float32Array<ArrayBuffer>;
  for (const tile of tiles) {
    // MUST: pass a copy (passing the view breaks the hand-off contract, which assumes the array
    // occupies its whole buffer).
    const slice = latent.slice(
      tile.start * latentDim,
      (tile.start + tile.length) * latentDim,
    ) as Float32Array<ArrayBuffer>;
    const decoded = await run(slice, tile.length);
    if (decoded.length !== tile.length * hopLength) {
      throw new Error(
        `decodeTiles: decoder の出力が ${decoded.length} サンプル` +
          `（${tile.length}×${hopLength} = ${tile.length * hopLength} のはず）`,
      );
    }
    waveform.set(
      decoded.subarray(tile.offset * hopLength, (tile.offset + tile.take) * hopLength),
      (tile.start + tile.offset) * hopLength,
    );
  }
  return waveform;
};

/**
 * Confirms "each side of a kept span either has a halo's worth of context or is a true boundary".
 *
 * MUST: do not remove this check. This condition alone is the ground for the bit identity, and when
 * it breaks the values still come out **looking plausible** (only a few LSBs differ near the seams).
 * This is the only place that notices when someone touches the geometry.
 */
const assertTileContext = (
  tiles: readonly CodecTile[],
  frames: number,
  haloFrames: number,
): void => {
  let covered = 0;
  for (const tile of tiles) {
    const takeStart = tile.start + tile.offset;
    const takeEnd = takeStart + tile.take;
    if (takeStart !== covered) {
      throw new Error(`planCodecTiles: 採用区間が ${covered} で途切れた（次は ${takeStart}）`);
    }
    if (tile.start + tile.length > frames) {
      throw new Error(
        `planCodecTiles: タイル [${tile.start}, ${
          tile.start + tile.length
        }) が S ${frames} を超えた`,
      );
    }
    if (tile.start > 0 && tile.offset < haloFrames) {
      throw new Error(
        `planCodecTiles: 左の文脈が ${tile.offset} フレームしかない（halo ${haloFrames} が要る）`,
      );
    }
    const right = tile.start + tile.length - takeEnd;
    if (tile.start + tile.length < frames && right < haloFrames) {
      throw new Error(
        `planCodecTiles: 右の文脈が ${right} フレームしかない（halo ${haloFrames} が要る）`,
      );
    }
    covered = takeEnd;
  }
  if (covered !== frames) {
    throw new Error(`planCodecTiles: 採用区間の合計 ${covered} が S ${frames} と違う`);
  }
};
