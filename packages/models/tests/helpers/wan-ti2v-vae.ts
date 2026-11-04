/**
 * What the tests of Wan2.2 TI2V-5B's VAE decoder chunk graphs share (ADR 0121 stage 4): the goldens of the f16 series
 * `wan2.2-ti2v-5b-f16-dyn` (fixtures, in the test-asset root — ADR 0139 decision 2), the table of fixture cases, the
 * expected contract values (including the cache table) and the fixtures' presence. The containers are the mirror's
 * (`models/karume-wan2.2/`, opened through the manifest with `openWanTi2vComponent` of `wan-ti2v-pipeline.ts` —
 * decision 1).
 *
 * The GPU-free cross-check (`e2e_wan_ti2v_vae_chunks_host_test.ts`) and the real-GPU chunk-sequence comparison
 * (`e2e_wan_ti2v_vae_tiles_test.ts`) read the same table — growing the table in 2 places would let a case added to only
 * one side silently escape the other side's check.
 *
 * The source of truth for the cases is `TI2V_FIXTURE_CASES` of `tools/export-recipes/wan/export_vae.py`. The source of
 * truth for the cache table is the forward order of the upstream decoder, and `TestCacheTableRealWeights.TABLE` of the
 * recipe's `wan/tests/test_vae_patch_ti2v.py` freezes the same values (frozen twice — changing only one side fails the
 * other side's test). **Fixed here rather than enumerated** (so that tests do not silently vanish in an environment
 * where only part of the generation was run).
 *
 * The stage-5 tile references `vae_tiles.*` (the writer is `tools/export-recipes/wan/vae_tiling.py --model ti2v-5b` —
 * the source of truth for the cases is its `TI2V_FIXTURE_CASES`) are **a separate presence set** from the chunk-graph
 * set ({@link ti2vVaeTileAssets} — none of the 3 is an explicit SKIP; only some is a FAIL), because their writer is a
 * different script (`vae_tiling.py`) and they are written at a different time from the chunk set.
 */

import { goldenDir } from "@karume/test-support/test-assets";
import { filePresent } from "./wan-ti2v-dit.ts";

/** The series the fixtures were exported with (their folder under `goldens/` of the test-asset root). */
const WAN_TI2V_VAE_SERIES = "wan2.2-ti2v-5b-f16-dyn";
/** The fixtures `vae_chunks.*` / `vae_tiles.*` / `vae_encoder.*` (the series' relative paths — decision 2). */
export const WAN_TI2V_VAE_GOLDEN_DIR = goldenDir(WAN_TI2V_VAE_SERIES);
/** Component names: the manifest's weights keys, which are also the containers' graph names (container-v1 §2.1). */
export const WAN_TI2V_VAE_FIRST = "vae_decoder_first";
export const WAN_TI2V_VAE_NEXT = "vae_decoder_next";

/** One fixture (1 that decides the band + 2 for acceptance — built the same way as 2.1). */
export type Ti2vVaeCase = {
  readonly name: "band" | "accept" | "long";
  readonly role: "band" | "accept";
  readonly chunks: number;
};

/** `long` is an 81-frame chunk sequence (1 + 4·20 — long enough to carry the cache over 20 times). */
export const WAN_TI2V_VAE_CASES: readonly Ti2vVaeCase[] = [
  { name: "band", role: "band", chunks: 9 },
  { name: "accept", role: "accept", chunks: 5 },
  { name: "long", role: "accept", chunks: 21 },
];

/** The edge of a latent tile (ADR 0121 decision 6 — 256 px of output ÷ spatial compression 16). */
export const WAN_TI2V_VAE_TILE = 16;
/** The edge of a frame (patchify space — 8 per latent). */
export const WAN_TI2V_VAE_SAMPLE_TILE = 128;
/** The number of latent channels (`z_dim` of the upstream config). */
export const WAN_TI2V_VAE_LATENT_CHANNELS = 48;
/** The number of output channels (patchify space `3·p²` — unpatchify is outside the graph). */
export const WAN_TI2V_VAE_SAMPLE_CHANNELS = 12;

/** One row of the cache table (name = `cache_` + upstream's `feat_idx` in 2 digits; shapes are for tile 16). */
export type Ti2vVaeCacheRow = {
  readonly name: string;
  readonly shape: readonly number[];
  /** The cache of upsample3d's `time_conv` (first does not have it — it does not run on the first chunk). */
  readonly timeConv: boolean;
};

const cacheRows = (
  from: number,
  to: number,
  shape: readonly number[],
  timeConv = false,
): Ti2vVaeCacheRow[] =>
  Array.from({ length: to - from }, (_, offset) => ({
    name: `cache_${String(from + offset).padStart(2, "0")}`,
    shape,
    timeConv,
  }));

/** Every cache of next (in input order — the same order and the same values as the recipe's table). */
export const WAN_TI2V_VAE_CACHE_TABLE: readonly Ti2vVaeCacheRow[] = [
  ...cacheRows(0, 1, [48, 2, 16, 16]),
  ...cacheRows(1, 11, [1024, 2, 16, 16]),
  ...cacheRows(11, 12, [1024, 2, 16, 16], true),
  ...cacheRows(12, 18, [1024, 2, 32, 32]),
  ...cacheRows(18, 19, [1024, 2, 32, 32], true),
  ...cacheRows(19, 20, [1024, 2, 64, 64]),
  ...cacheRows(20, 25, [512, 2, 64, 64]),
  ...cacheRows(25, 26, [512, 2, 128, 128]),
  ...cacheRows(26, 32, [256, 2, 128, 128]),
];

/** The `time_conv` caches (only next has them — from the table's `timeConv`). */
export const WAN_TI2V_VAE_TIME_CONV_CACHES: readonly string[] = WAN_TI2V_VAE_CACHE_TABLE.filter(
  ({ timeConv }) => timeConv,
).map(({ name }) => name);

/**
 * The total resident bytes of the caches (f32 — 116,940,800 elements × 4. The same value as the element count of the
 * recipe's real-weight test, written independently of the table — this catches a mistake in building the table).
 */
export const WAN_TI2V_VAE_CACHE_BYTES = 467_763_200;
/**
 * The total bytes of first's caches (f32 — 114,319,360 elements × 4. The same value as first's element count in the
 * same recipe test, written independently of the table).
 */
export const WAN_TI2V_VAE_FIRST_CACHE_BYTES = 457_277_440;

export const ti2vVaeFixtureUrl = (name: Ti2vVaeCase["name"]): URL =>
  new URL(`vae_chunks.${name}.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/** The presence of the chunk fixtures `names` (all three by default). Checked on each call, never at module evaluation. */
export const ti2vVaeChunkFixtures = (
  names: readonly Ti2vVaeCase["name"][] = WAN_TI2V_VAE_CASES.map(({ name }) => name),
): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  names.map((name) => {
    const url = ti2vVaeFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });

// ---- Tile references (stage 5 — `vae_tiles.<case>.safetensors`) ----------------------------------------

/** One tile reference (1 that decides the band + 2 for acceptance — built the same way as 2.1's tile references). */
export type Ti2vVaeTileCase = {
  readonly name: "band" | "accept" | "wide";
  readonly role: "band" | "accept";
  readonly chunks: number;
  /** The height of the latent space (output pixels = × 16). */
  readonly height: number;
  /** The width of the latent space (output pixels = × 16). */
  readonly width: number;
  /** Holds the non-tiled reference `frames_full` (patchify space, before the clamp — observation). */
  readonly full: boolean;
  /**
   * Holds the RGB `frames_rgb [3, F, H, W]` with upstream's unpatchify → clamp applied (the comparison at the end of
   * the VAE stage).
   */
  readonly rgb: boolean;
};

/**
 * - `band`: 832×480×81 (latent 30×52, 21 chunks) — the one that decides the band (stage 5 of ADR 0121's acceptance
 *   table).
 * - `accept`: portrait 480×832×9 (latent 52×30, 3 chunks) — both the tile positions and the chunk boundaries differ
 *   from band.
 * - `wide`: 1280×704×5 (latent 44×80, 2 chunks) — the only size whose blend width differs per pair.
 */
export const WAN_TI2V_VAE_TILE_CASES: readonly Ti2vVaeTileCase[] = [
  { name: "band", role: "band", chunks: 21, height: 30, width: 52, full: false, rgb: false },
  { name: "accept", role: "accept", chunks: 3, height: 52, width: 30, full: true, rgb: true },
  { name: "wide", role: "accept", chunks: 2, height: 44, width: 80, full: false, rgb: false },
];

export const ti2vVaeTileFixtureUrl = (name: Ti2vVaeTileCase["name"]): URL =>
  new URL(`vae_tiles.${name}.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/**
 * The presence of the 3 tile references (a set separate from the chunk-graph set). Checked on every call (module
 * evaluation does not touch files).
 */
export const ti2vVaeTileAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  WAN_TI2V_VAE_TILE_CASES.map(({ name }) => {
    const url = ti2vVaeTileFixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });

// ---- The 32-latent pair (ADR 0134 — weightless containers that borrow the 16-latent pair) ---------

/** The 32-latent pair's components (= component directories and graph names). */
export const WAN_TI2V_VAE_FIRST_32 = "vae_decoder_first_32";
export const WAN_TI2V_VAE_NEXT_32 = "vae_decoder_next_32";
/** The 32-latent pair's latent tile edge and frame edge (patchify space — 8 per latent). */
export const WAN_TI2V_VAE_TILE_32 = 32;
export const WAN_TI2V_VAE_SAMPLE_TILE_32 = 256;

/**
 * The tile references at tile 32 (ADR 0134 decision 6 — `vae_tiling.py --tile 32`). Only `wide`
 * (1280×704×5, latent 44×80 → 2×3 tiles) fits a 32-latent tile: `band` and `accept` are 30 latents
 * on one axis, so the writer skips them.
 */
export const WAN_TI2V_VAE_TILE32_CASES: readonly Ti2vVaeTileCase[] = WAN_TI2V_VAE_TILE_CASES.filter(
  ({ height, width }) => height >= WAN_TI2V_VAE_TILE_32 && width >= WAN_TI2V_VAE_TILE_32,
);

export const ti2vVaeTile32FixtureUrl = (name: Ti2vVaeTileCase["name"]): URL =>
  new URL(`vae_tiles.${name}.t32.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/**
 * Presence of the tile-32 references (none → explicit SKIP, some → FAIL — the convention of
 * {@link ti2vVaeTileAssets}). Checked on each call, never at module evaluation.
 */
export const ti2vVaeTile32TileAssets = (): readonly {
  readonly path: string;
  readonly present: boolean;
}[] =>
  WAN_TI2V_VAE_TILE32_CASES.map(({ name }) => {
    const url = ti2vVaeTile32FixtureUrl(name);
    return { path: url.pathname, present: filePresent(url) };
  });
