/**
 * The reference comparison of the Wan2.2 TI2V-5B video VAE's **tiled decode** (real GPU — the acceptance of ADR 0121 step 5).
 *
 * The two f16 chunk graphs of the distribution mirror `models/karume-wan2.2/` (stage 4 — latent tile 16, 12 output
 * channels in patchify space; opened through the manifest — ADR 0139 decision 1; the fixtures are in the test-asset
 * root — decision 2) are tiled over the whole frame with the production plan (`planWanGenerationTiles(…,
 * WAN22_TI2V_GENERATION)` — overlap of 4 latents) and run (tiles outside, chunks inside, one tile = one batch — the same
 * `decodeWanVaeTiled` as 2.1). The comparison target is the recipe's tile reference (`tools/export-recipes/wan/
 * vae_tiling.py --model ti2v-5b` — the same geometry, upstream's blend verbatim in patchify space, CPU f32, weights
 * rounded to f16) **before unpatchify and clamp**. The skeleton is 2.1's `e2e_wan_vae_tiles_test.ts`; the handling
 * while the band is not derived yet is the same as the step-4 chunk-sequence comparison (section "Chunk sequence" below).
 *
 * ## Checks
 *
 * 1. **Plan**: the TS plan (from the assets' input shapes) matches the fixtures' metadata (the Python plan — tile / scale
 *    / start positions / blend widths / patch_size). band has 12 tiles (rows 0,7,14, columns 0,12,24,36), wide has 28.
 * 2. **Degenerate check**: for the latent `[48,9,16,16]` of the step-4 chunk-sequence fixture (exactly one tile), the
 *    tiled path (`decodeWanVaeTiled`) is **Uint32-identical** to the untiled chunk sequence (`decodeWanVaeTile`).
 * 3. **Band**: `allclose(atol = BAND, rtol = 0)` on `band` (832×480, 81 frames, 12 tiles × 21 chunks). The acceptance
 *    cases are `accept` (480×832, 9 frames — portrait, different chunk boundaries) and `wide` (1280×704, 5 frames —
 *    the blend width differs per pair). Further, the RGB from passing the assembled `accept` through the end of the VAE
 *    stage (`finishWanVaeFrames` — shape check → unpatchify → finiteness → clamp) is compared with `frames_rgb` from
 *    upstream's `unpatchify` → `clamp` under **the same band** (unpatchify is a permutation and clamp is 1-Lipschitz,
 *    so the band carries over unchanged — this checks the host unpatchify against the upstream function itself).
 *    It is also checked independently of the band (a test that needs no GPU): the RGB from applying
 *    `finishWanVaeFrames` to the fixture's `frames` (the upstream tile reference itself) is **Uint32-identical** to
 *    the `frames_rgb` upstream made from the same `frames` (unpatchify is bit-identical to upstream's order `(c, r,
 *    q)` — step 5 of ADR 0121's acceptance table; green even while the band is not derived).
 * 4. **Poison values and non-finite values**: the resident frames are refilled with NaN before each tile's decode (at
 *    the start of a case, and in `onTile` before the next tile — so finite values left by the previous tile do not
 *    hide a missed copy), and every case has 0 non-finite values and no device lost.
 * 5. **Fault injection** (host side — the decoded tiles of `accept` are assembled with a fault): swapping rows and
 *    columns and shifting a start position by one latent land outside the band. 2.1's floor of "1e3 × the band" is not
 *    carried over (the effect of 2.2's injections was not measured — the same judgement as step 4).
 *
 * ## How the band is set
 *
 * - The band is the measured maxAbs of a plain comparison on the single band-setting case (`band`) × 5, rounded up to
 *   2 significant digits. The acceptance cases (accept / accept RGB / wide) are not used to set the band.
 * - `undefined` = not derived yet: each case records its maxAbs and stops red, the fault injections record their maxAbs
 *   and stop red too, and the last step prints a band candidate ({@link bandCandidate}). The plan, the degenerate check,
 *   0 non-finite values and no device lost are green even then (each is its own step). The value the first real run
 *   derives is written into the constant together with the table of measurements.
 *
 * ## Assets
 *
 * The tile references `vae_tiles.*` are a presence group separate from the chunk-graph group: all 3 absent means an
 * explicit SKIP, only some present is a FAIL (this file's GPU-free tests). When any tile reference is present but the
 * chunk-graph group (2 containers + `vae_chunks.*` — the comparison's prerequisites) is incomplete, that is a FAIL too
 * (no lane may end green without the comparison ever running). A machine with only the chunk group (tile references
 * not written yet) stays SKIP. Gaps inside the chunk group are also failed by the step-4 host test
 * (`wan_ti2v_vae_chunks_host_test.ts`).
 *
 * ## Chunk sequence (ADR 0121 stage 4 acceptance)
 *
 * The GPU tests placed before the tile decode tests check the chunk graphs without a tile plan. They keep
 * their own band ({@link CHUNK_BAND}), cases (`WAN_TI2V_VAE_CASES`) and results family (`wan-ti2v-vae-chunks`),
 * separate from the tile decode's.
 *
 * The **reference comparison of the chunk sequence** of the Wan2.2 TI2V-5B video VAE (real GPU — the acceptance of ADR 0121 step 4).
 *
 * The mirror's two f16 chunk graphs (`vae_decoder_first` / `vae_decoder_next` — latent tile 16, 12 output channels in
 * patchify space) run first → next × (F−1) through the production path (`decodeWanVaeTile` — the same function as
 * 2.1, one tile = one batch, only the frames read back), and the output is compared with upstream's untiled `_decode`
 * chunk loop (diffusers, CPU f32, weights rounded to f16) **before unpatchify and clamp**. The references and fixtures
 * are written by `tools/export-recipes/wan/export_vae.py --model ti2v-5b` and read from the test-asset root (the table
 * is in `helpers/wan-ti2v-vae.ts`). The skeleton is the same as 2.1's `e2e_wan_vae_tiles_test.ts`, and the
 * fault-injection loop is the shared `helpers/wan-vae-chunk-loop.ts`.
 *
 * ### How the band is set
 *
 * - The band is set from the measured maxAbs of a plain comparison (`rtol = 0`) on the `band` case (9 chunks = 33
 *   frames) × 5 (the same derivation as 2.1's `BAND` — {@link CHUNK_BAND}). The acceptance cases, `accept` with a
 *   different latent and different chunk boundaries (5 chunks = 17 frames) and `long` (21 chunks = 81 frames — long
 *   enough to carry the caches over 20 times), are checked to be inside the band. The band is not changed after seeing
 *   the long result.
 * - `undefined` = not derived yet: each case records its maxAbs and stops red, and the last step prints a band candidate
 *   (band's maxAbs × 5 rounded up to 2 significant digits) ({@link chunkBandCandidate} — the same shape as the step-2 DiT
 *   e2e; no placeholder value that passes, which would make a check with no detection power). The value the first real
 *   run derives is written into the constant together with the table of measurements.
 *
 * ### Fault injection
 *
 * - A cache not updated, the 2 frames of a cache reversed, and the zeroing at the tile head skipped are checked to land
 *   **outside the band** on `accept` (2.1's floor of "1e3 × the band" came from 2.1's measurements, and the effect of
 *   2.2's injections was not measured, so it is not carried over). While the band is not derived, they record their
 *   maxAbs and stop red (the injection targets are fixed in the same commit that writes the band — changing a target
 *   after it misses would be the same as fitting the band afterwards).
 * - Using first for chunk 2 onwards is rejected by the size of `copyOutputs` before any value is compared.
 * - The injection loop without a fault is also checked to be Uint32-identical to the product path (ensuring the
 *   injected results represent the product path).
 * - A missing leading slice of DupUp3D is not injected on the GPU (it is a node inside the graph, and a graph missing it
 *   fails on shape in both export and eager — the checks are the recipe's pytest and the TS host test
 *   `wan_vae_chunks_test.ts`, which rejects a declaration with 4 first frames).
 *
 * ### The mid attention (D = 1024)
 *
 * - The plans of both graphs' last runs are checked to queue the 3 stages of fused attention (`attention_qk` /
 *   `attention_stats` / `attention_pv`) at least once each (ADR 0121 decision 6 left the D = 1024 measurement to step 4
 *   with no check of the upper limit of D — the band sees whether the values are correct, but not whether the fused
 *   path was taken). A run without a plan census is red rather than counted as 0.
 *
 * An environment with incomplete assets or without a GPU SKIPs explicitly. Asset completeness (only some present is a
 * FAIL) is checked by the GPU-free host test (`wan_ti2v_vae_chunks_host_test.ts`).
 *
 * ## Default run and opt-in upstream parity (maintainer ruling 2026-10-17)
 *
 * Every comparison with an upstream reference under a band (the chunk sequence against the diffusers
 * `_decode`, the tile decode and the accept RGB against the recipe's upstream tiled reference), the
 * fault injections that only prove those bands are live (the cache faults and the seam faults), and
 * the band-candidate steps are opt-in behind `KARUME_UPSTREAM_PARITY=1` (`helpers/upstream-parity.ts`).
 * The default run keeps the checks that do not depend on a band: the asset gates, the GPU-free
 * unpatchify Uint32 identity with `frames_rgb`, the chunk layout and cache size, the fused-attention
 * census, the fault loop's Uint32 identity with the product path, the `copyOutputs` rejection, the
 * TS/Python plan equality, the degenerate Uint32 gate, and every case decoded from NaN-poisoned
 * frames with 0 non-finite and no device lost. The record-only cases (the measured-mode submit GPU
 * time, the VRAM and submit statistics, the untiled `frames_full` observation) are removed.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import {
  ExecutionError,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  type WanVaeChunkLayout,
  wanVaeFrameCount,
} from "../../models/src/wan/vae-chunks.ts";
import type { PlanLayout } from "../../models/src/wan/plan.ts";
import {
  assembleWanVaeTiles,
  decodeWanVaeTiled,
  decodeWanVaeTiles,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../../models/src/wan/vae-tiles.ts";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import {
  assertWanVaeMatchesGeneration,
  finishWanVaeFrames,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../../models/src/wan/tile-decode.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  bitsEqual,
  decodeWithFault,
  type Fault,
  type Fixture,
  readFixture,
} from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { TEST_ASSETS_FETCH as FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  openWanTi2vComponent,
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import {
  type Ti2vVaeCase,
  ti2vVaeChunkFixtures,
  ti2vVaeFixtureUrl,
  ti2vVaeTileAssets,
  type Ti2vVaeTileCase,
  ti2vVaeTileFixtureUrl,
  WAN_TI2V_VAE_CACHE_BYTES as CACHE_BYTES,
  WAN_TI2V_VAE_CACHE_TABLE as CACHE_TABLE,
  WAN_TI2V_VAE_CASES as CHUNK_CASES,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_GOLDEN_DIR as GOLDEN_DIR,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
  WAN_TI2V_VAE_TILE as TILE,
  WAN_TI2V_VAE_TILE_CASES as CASES,
  WAN_TI2V_VAE_TIME_CONV_CACHES as TIME_CONV_CACHES,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  openResults,
  type RecordedCase,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The band of the tiled decode (absolute — the output is pre-clamp frames in patchify space). `undefined` = not derived
 * yet (module doc "How the band is set").
 *
 * Derivation (2026-10-05, B570): maxAbs of the band-setting band (12 tiles × 21 chunks) 9.805e-6 × 5 = 4.902e-5 → rounded
 * up to 2 significant digits, 5.0e-5. Acceptance: accept 7.339e-6, accept RGB 7.339e-6, wide 6.348e-6. The maxAbs of the
 * 2 fault injections are 2.18 / 2.26.
 *
 * MUST: neither this value nor the band-setting case is changed after seeing the acceptance results (accept / accept RGB
 * / wide). When an acceptance case leaves the band, investigate the cause instead of widening the band.
 */
const BAND: number | undefined = 5.0e-5;
/** The band for the verdict and the record (while not derived, it is recorded as unbounded and the verdict is red). */
const TOLERANCE: Tolerance = { atol: BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

/** The owner named in the messages of the VAE stage's end (2.2's pipeline name — the same as in the host test). */
const OWNER = "wan.Ti2vPipeline";
/** The number of RGB channels (the leading dimension of `frames_rgb` after unpatchify). */
const RGB_CHANNELS = 3;

type CaseName = Ti2vVaeTileCase["name"];

/** The latent of the degenerate check (the step-4 chunk-sequence fixture `vae_chunks.band` — latent `[48,9,16,16]`, exactly one tile). */
const DEGENERATE_CASE = "band";

const TILE_ASSETS = ti2vVaeTileAssets();
const ANY_TILES = TILE_ASSETS.some(({ present }) => present);
const ALL_TILES = TILE_ASSETS.every(({ present }) => present);
const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing fixture a FAIL (the completeness tests below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_FIXTURES = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;
const CHUNK_FIXTURES = ti2vVaeChunkFixtures();
/** The chunk-sequence checks read the mirror's two chunk graphs and the three chunk fixtures. */
const CHUNK_ASSETS_PRESENT = MIRROR_PRESENT && CHUNK_FIXTURES.every(({ present }) => present);
/** The tile checks read the tile references, the mirror and the degenerate gate's chunk fixture only. */
const TILE_RUN_PRESENT =
  ALL_TILES &&
  MIRROR_PRESENT &&
  ti2vVaeChunkFixtures([DEGENERATE_CASE]).every(({ present }) => present);
if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the real-GPU checks of Wan2.2 TI2V's VAE decoder are SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
}
if (!ANY_TILES) {
  console.warn(
    `[karume] no tile reference (vae_tiles.*) under ${GOLDEN_DIR.pathname}, so the tile-decode checks are ` +
      `SKIPPED (${MISSING_FIXTURES})`,
  );
}

/** The tile reference's metadata `weights` (the provenance of the weights — spelled as by the writer `vae_tiling.write_fixture`). */
const FIXTURE_WEIGHTS = "f16-rounded";
/** The tile reference's metadata `reference` (the provenance of the reference — spelled as `vae_tiling.FIXTURE_REFERENCE_IN_PATCHIFY_SPACE`). */
const FIXTURE_REFERENCE =
  "snapped-tile decode with upstream blend_v / blend_h in patchify space before unpatchify and clamp (CPU f32)";

// An environment with none "has not generated them", so it SKIPs. With any one present a missing one is a FAIL (a group
// separate from the chunk graphs) — and so is any missing one in a complete fetch of the pinned dataset.
test.skipIf(!ANY_TILES && !FETCHED)(
  "Wan2.2 TI2V VAE tile references: the 3 vae_tiles files (band / accept / wide) are present",
  () => {
    assertEquals(
      TILE_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `missing tile references (${MISSING_FIXTURES})`,
    );
  },
);

// The tile references and the chunk fixtures are written by different recipes, so a root may hold one group without
// the other. A SKIP would end the lane green without the GPU check ever running. (The mirror's completeness is the
// distribution gate's.)
test.skipIf(!ANY_TILES && !FETCHED)(
  "Wan2.2 TI2V VAE tile references: with tile references present, the chunk fixtures the check needs (vae_chunks.*) " +
    "are present too",
  () => {
    assertEquals(
      CHUNK_FIXTURES.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `chunk fixtures missing (${MISSING_FIXTURES})`,
    );
  },
);

/** One tile reference (F32 tensors and metadata — written by `vae_tiling.write_fixture`). */
type TileFixture = {
  readonly tensor: (key: string) => {
    readonly data: Float32Array<ArrayBuffer>;
    readonly shape: readonly number[];
  };
  readonly has: (key: string) => boolean;
  readonly meta: (key: string) => string;
};

const readTileFixture = async (url: URL): Promise<TileFixture> => {
  const bytes = await readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    tensor: (key) => {
      const tensor = file.tensors.get(key);
      assert(
        tensor !== undefined && tensor.dtype === "F32",
        `${url.pathname}: '${key}' is not F32`,
      );
      return {
        data: new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4),
        shape: tensor.shape,
      };
    },
    has: (key) => file.tensors.has(key),
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: no metadata '${key}'`);
      return value;
    },
  };
};

/**
 * Renders the plan's geometry in the same spelling as the fixtures' metadata (`vae_tiling.TilePlan.meta` +
 * `patch_size`) (the TS ↔ Python comparison — sequences in the form `"0,7,14"`).
 */
const planMeta = (plan: WanVaeTilePlan): Record<string, string> => {
  assertEquals(plan.rows.tile, plan.cols.tile, "tiles are square");
  const blends = (axis: WanVaeTilePlan["rows"]) =>
    axis.starts.slice(1).map((_, index) => wanVaeBlendExtentAt(axis, plan.scale, index + 1));
  return {
    tile: String(plan.rows.tile),
    scale: String(plan.scale),
    rows_starts: plan.rows.starts.join(","),
    cols_starts: plan.cols.starts.join(","),
    rows_blend: blends(plan.rows).join(","),
    cols_blend: blends(plan.cols).join(","),
    patch_size: String(WAN22_TI2V_GENERATION.vaePatchSize),
  };
};

/**
 * Plans from a latent `[C,F,H,W]` through the same entry point as production (it goes through
 * `planWanGenerationTiles`, so the comparison with the fixtures' metadata pins the production plan).
 */
const planFor = (layout: PlanLayout, latentShape: readonly number[]): WanVaeTilePlan => {
  assertEquals(latentShape.length, 4, `latent shape [${latentShape}]`);
  return planWanGenerationTiles(layout, latentShape[2], latentShape[3], WAN22_TI2V_GENERATION);
};

/**
 * A literal of the plan's layout (for the GPU-free test — that the real assets' layout matches it is checked by the
 * plan step of the GPU tests and by the step-5 host test).
 */
const LITERAL_LAYOUT: PlanLayout = {
  latentChannels: LATENT_CHANNELS,
  tile: TILE,
  sampleTile: SAMPLE_TILE,
  sampleChannels: SAMPLE_CHANNELS,
};

// A comparison independent of the band (unpatchify is a permutation and clamp only maps f32 values to f32 values — the
// same input gives no rounding difference). A partial group is failed by the presence test, so this runs only on a
// machine with all 3.
test.skipIf(!ALL_TILES)(
  "Wan2.2 TI2V VAE tile reference (no GPU needed): the RGB from applying the end of the VAE stage (finishWanVaeFrames) " +
    "to accept's frames is Uint32-identical to frames_rgb from upstream's unpatchify → clamp",
  async () => {
    const spec = CASES.find(({ name }) => name === "accept");
    assert(spec !== undefined && spec.rgb, "the table has no accept case (with frames_rgb)");
    const fixture = await readTileFixture(ti2vVaeTileFixtureUrl("accept"));
    const latents = fixture.tensor("latents");
    const plan = planFor(LITERAL_LAYOUT, latents.shape);
    const compression = wanSpatialCompression(LITERAL_LAYOUT, WAN22_TI2V_GENERATION);
    const want = fixture.tensor("frames_rgb");
    const frames = wanVaeFrameCount(latents.shape[1]);
    const height = spec.height * compression;
    const width = spec.width * compression;
    assertEquals(want.shape, [RGB_CHANNELS, frames, height, width], "frames_rgb shape");
    const got = finishWanVaeFrames(
      // Pass a copy (keeps the fixture's values whether or not the clamp writes in place).
      Float32Array.from(fixture.tensor("frames").data),
      { frames, width, height, tiles: plan },
      WAN22_TI2V_GENERATION,
      OWNER,
    );
    assertEquals(got.length, want.data.length, "element count");
    assert(
      bitsEqual(got, want.data),
      "the TS unpatchify → clamp is not bit-identical to upstream's frames_rgb",
    );
  },
);

const absMax = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** A summary of the difference (ratio = max absolute difference ÷ the reference's max absolute value — the metric of decision 8). */
const compare = (got: Float32Array, want: Float32Array, tolerance: Tolerance) => {
  assertEquals(got.length, want.length, "element count");
  const report = allclose(got, want, tolerance);
  const referenceMax = absMax(want);
  return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
};

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/** Fills the resident frames with NaN (a frame that was not copied stays NaN — a poison value). */
const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

/**
 * The `onTile` hook: except after the last tile, refills the resident frames with poison values before the next tile's
 * decode (a tile's output has already been read back by `decodeWanVaeTile`, so finished tiles' values are not damaged).
 */
const poisonBetweenTiles =
  (caches: WanVaeChunkCaches, plan: WanVaeTilePlan, chunks: number) => async (tile: number) => {
    if (tile < wanVaeTileCount(plan)) await poisonFrames(caches, chunks);
  };

/** Rounding up to 2 significant digits (the same rule as the band candidates of the step 2 / 4 e2e). */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

const results = openResults("wan-ti2v-vae-tiles");

/**
 * The body of the step that prints a candidate for the not-yet-derived band (`undefined`) and stops red (built from band
 * only; the maxAbs of the acceptance cases and the fault injections are listed but not used for the candidate. When band
 * failed before producing a maxAbs, no candidate is given). Once the band is derived the step SKIPs (`it.skipIf` where
 * it is registered).
 */
const bandCandidate = async (
  observed: ReadonlyMap<string, number>,
  faults: ReadonlyMap<string, number>,
): Promise<void> => {
  const band = observed.get("band");
  const candidate =
    band === undefined
      ? "band failed before producing a maxAbs — no candidate (investigate the failed case first)"
      : band > 0
        ? `band's maxAbs ${band.toExponential(3)} × 5 = ${(band * 5).toExponential(3)}` +
          ` (rounded up to 2 significant digits: ${roundUpTwoDigits(band * 5)})`
        : "band's maxAbs is 0 — × 5 cannot derive a band (investigate the comparison path first)";
  const format = (value: number | undefined): string => value?.toExponential(3) ?? "—";
  const accepted = ["accept", "accept-rgb", "wide"].map(
    (label) => `${label} ${format(observed.get(label))}`,
  );
  const injected = [...faults].map(([label, value]) => `${label} ${format(value)}`);
  const message = `band not derived yet — ${candidate}. Acceptance maxAbs (not used to set the band): ${accepted.join(
    " / ",
  )}. Fault-injection maxAbs (C4 judges whether they are outside the band): ${
    injected.length === 0 ? "—" : injected.join(" / ")
  }`;
  await runRecordedCase(results, { id: "band-candidate", failureNote: () => message }, () => {
    console.log(`[wan-ti2v-vae-tiles] ${message}`);
    throw new Error(message);
  });
};

// Chunk sequence (module doc "Chunk sequence"): one tile = one batch, no tile plan.

/**
 * The band of the chunk sequence (absolute — the output is pre-clamp frames in patchify space). `undefined` = not
 * derived yet (module doc "How the band is set").
 *
 * Derivation (2026-10-05, B570): maxAbs of the band-setting band (9 chunks) 9.477e-6 × 5 = 4.739e-5 → rounded up to 2
 * significant digits, 4.8e-5. Acceptance: accept (5 chunks) 7.421e-6, long (21 chunks) 1.153e-5. The maxAbs of the 3
 * fault injections are 0.66 / 1.37 / 1.30.
 *
 * MUST: neither this value nor the band-setting case is changed after seeing the acceptance results (accept / long).
 * When an acceptance case leaves the band, investigate the cause instead of widening the band.
 */
const CHUNK_BAND: number | undefined = 4.8e-5;
/** The band for the verdict and the record (while not derived, it is recorded as unbounded and the verdict is red). */
const CHUNK_TOLERANCE: Tolerance = { atol: CHUNK_BAND ?? Number.POSITIVE_INFINITY, rtol: 0 };

type ChunkCaseName = Ti2vVaeCase["name"];

/** The cache the fault injection drops (the head's conv — the least influential one, passing through a single layer; the same position as 2.1). */
const DROPPED_CACHE = "cache_31";

/** The 3 stages of fused attention to record (ADR 0023 — the pipeline key prefixes of ①QK, ②row statistics, ③PV). */
const ATTENTION_KERNELS = ["attention_qk", "attention_stats", "attention_pv"] as const;

if (MIRROR_PRESENT && !CHUNK_ASSETS_PRESENT) {
  console.warn(
    `[karume] the chunk fixtures under ${GOLDEN_DIR.pathname} are not all present, so the chunk-sequence ` +
      `checks are SKIPPED (the host test reports the gap; ${MISSING_FIXTURES})`,
  );
}

const chunkResults = openResults("wan-ti2v-vae-chunks");

/** Opens and prepares the mirror's two f16 chunk graphs (the graph names are the component keys). */
const prepareModels = async () => {
  const [firstOpened, nextOpened] = await Promise.all([
    openWanTi2vComponent(FIRST, "f16"),
    openWanTi2vComponent(NEXT, "f16"),
  ]);
  return {
    first: prepareContainer(firstOpened, FIRST),
    next: prepareContainer(nextOpened, NEXT),
  };
};

/**
 * The body of the step that prints a candidate for the not-yet-derived band (`undefined`) and stops red (built from band
 * only; the acceptance maxAbs are listed but not used for the candidate. When band failed before producing a maxAbs, no
 * candidate is given). Once the band is derived the step SKIPs (`it.skipIf` where it is registered).
 */
const chunkBandCandidate = async (observed: ReadonlyMap<ChunkCaseName, number>): Promise<void> => {
  const band = observed.get("band");
  const candidate =
    band === undefined
      ? "band failed before producing a maxAbs — no candidate (investigate the failed case first)"
      : band > 0
        ? `band's maxAbs ${band.toExponential(3)} × 5 = ${(band * 5).toExponential(3)}` +
          ` (rounded up to 2 significant digits: ${roundUpTwoDigits(band * 5)})`
        : "band's maxAbs is 0 — × 5 cannot derive a band (investigate the comparison path first)";
  const accepted = CHUNK_CASES.filter(({ role }) => role === "accept").map(
    ({ name }) => `${name} ${observed.get(name)?.toExponential(3) ?? "—"}`,
  );
  const message = `band not derived yet — ${candidate}. Acceptance maxAbs (not used to set the band): ${accepted.join(
    " / ",
  )}`;
  await runRecordedCase(chunkResults, { id: "band-candidate", failureNote: () => message }, () => {
    console.log(`[wan-ti2v-vae] ${message}`);
    throw new Error(message);
  });
};

/**
 * The dispatch counts of the 3 fused-attention stages and the list of pipeline keys starting with `attention` (from
 * the last run's plan).
 *
 * MUST: a run without a census is not counted as 0 dispatches (the fusion check would pass vacuously) — it throws.
 */
const attentionPipelines = (graph: string, diagnostics: SessionDiagnostics) => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) throw new Error(`${graph}: the last run has no pipeline census`);
  const dispatches = (kernel: (typeof ATTENTION_KERNELS)[number]): number =>
    pipelines
      .filter(({ key }) => key.split(":")[0] === kernel)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  return {
    dispatches: {
      attention_qk: dispatches("attention_qk"),
      attention_stats: dispatches("attention_stats"),
      attention_pv: dispatches("attention_pv"),
    },
    keys: pipelines.filter(({ key }) => key.startsWith("attention")).map(({ key }) => key),
  };
};

/** The two product Sessions, the cache residents and the fixtures one chunk-sequence case runs on. */
type ChunkRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
  readonly fixtures: Readonly<Record<ChunkCaseName, Fixture>>;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and the chunk
 * fixtures and builds both product Sessions and the cache residents on one device, and an afterAll
 * that tears them down, so the default suite and the opt-in parity suite share one lifecycle and one
 * teardown order. Returns the accessor the its read the run through.
 */
const registerChunkSessions = (): (() => ChunkRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: ChunkRun | undefined;
  beforeAll(async () => {
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    const fixtures = {
      band: await readFixture(ti2vVaeFixtureUrl("band")),
      accept: await readFixture(ti2vVaeFixtureUrl("accept")),
      long: await readFixture(ti2vVaeFixtureUrl("long")),
    };

    const device = verifyRunningGpu(await acquireTestGpu());
    gpu = device;
    first = await models.first.createContainerSession(device);
    next = await models.next.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = { gpu: device, layout, sessions: { first, next }, caches, fixtures };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: tear down in the order Session → resident tensors → device (the same order and reason as the 2.1 e2e — the
    // resident tensors may be referenced by what the Session baked in, so the Session goes first. Even when one step
    // fails, the remaining steps always run). Vitest reports an afterAll failure
    // apart from the its, so a teardown failure cannot overwrite a test failure.
    await disposeSteps([
      () => first?.dispose(),
      () => next?.dispose(),
      () => caches?.dispose(),
      // Wait for the releases before discarding the device (leaves budget for the following tests — {@link settleReleases}).
      () => (device === undefined ? undefined : settleReleases(device)),
      () => device?.destroy(),
    ]);
  });
  return () => {
    if (run === undefined) throw new Error("chunk-sequence setup (beforeAll) did not complete");
    return run;
  };
};

// Default run: the chunk layout and cache size, every case decoded from NaN-poisoned frames with 0
// non-finite, the fused-attention census, the fault loop's Uint32 identity with the product path and
// the copyOutputs rejection. It owns the per-case results.json ids; the band comparison with the
// diffusers `_decode` is the opt-in suite below.
describe.skipIf(!CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE chunk sequence: the layout of first → next with resident caches, 0 non-finite from poisoned frames, " +
    "the 3 fused-attention stages, Uint32 identity of the fault-injection loop with the product path (real GPU)",
  () => {
    const chunkRun = registerChunkSessions();

    it("layout: latent 48 ch, tile 16, output 12 ch, the 32 caches of the table (first has 30, without the 2 time_conv ones)", () => {
      const { layout } = chunkRun();
      assertEquals(
        [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
        [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      );
      assertEquals(
        layout.next.caches,
        CACHE_TABLE.map(({ name }) => name),
      );
      assertEquals(
        layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
        [...TIME_CONV_CACHES],
      );
      assertEquals(layout.first.frameShape, [SAMPLE_CHANNELS, 1, SAMPLE_TILE, SAMPLE_TILE]);
      assertEquals(layout.next.frameShape, [SAMPLE_CHANNELS, 4, SAMPLE_TILE, SAMPLE_TILE]);
    });

    it("observation: bytes and host time of zeroing the caches at the tile head (GPU-side clear)", () => {
      const { caches: liveCaches } = chunkRun();
      // The same record as 2.1 (`ResidentTensor.clear` per cache — `vae-chunks.ts` doc). The tiled
      // decode pays it once per tile.
      const started = performance.now();
      liveCaches.zero();
      const hostMs = performance.now() - started;
      console.log(
        `[wan-ti2v-vae] zero: bytes=${liveCaches.cacheBytes} hostMs=${hostMs.toFixed(1)}`,
      );
      assertEquals(liveCaches.cacheBytes, CACHE_BYTES);
    });

    for (const { name, chunks: expectedChunks } of CHUNK_CASES) {
      it(`${name}: the chunk sequence run from poisoned frames has 0 non-finite (chunk count, frame shape, element count)`, async () => {
        const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
        await runRecordedCase(chunkResults, { id: name }, async () => {
          const fixture = fixtures[name];
          const chunks = wanVaeChunkCount(layout, fixture.latents);
          assertEquals(chunks, expectedChunks, `chunk count of ${name}`);
          assertEquals(fixture.frameShape, [
            SAMPLE_CHANNELS,
            wanVaeFrameCount(chunks),
            SAMPLE_TILE,
            SAMPLE_TILE,
          ]);
          await poisonFrames(liveCaches, chunks);
          const started = performance.now();
          const got = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
          const elapsedMs = performance.now() - started;
          assertEquals(got.length, fixture.frames.length, `${name}: element count`);
          const nonFinite = countNonFinite(got);
          console.log(
            `[wan-ti2v-vae] ${name}: chunks=${chunks} nonFinite=${nonFinite} ` +
              `decodeMs=${elapsedMs.toFixed(0)}`,
          );
          assertEquals(nonFinite, 0, `${name}: non-finite (frames that were not copied)`);
          return undefined;
        });
      });
    }

    it("the mid attention (D = 1024) goes through the 3 fused-attention stages in both graphs", () => {
      const { sessions } = chunkRun();
      // From the last tile's plan (even with the band not derived, every case runs to completion before the comparison,
      // so the census is filled).
      for (const [graph, session] of Object.entries(sessions)) {
        const { dispatches } = attentionPipelines(graph, session.diagnostics());
        for (const kernel of ATTENTION_KERNELS) {
          assert(
            dispatches[kernel] > 0,
            `${graph}: ${kernel} has 0 dispatches (${JSON.stringify(dispatches)})`,
          );
        }
      }
    });

    it("the fault-injection loop without a fault is Uint32-identical to the product path", async () => {
      const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
      const fixture = fixtures.accept;
      const chunks = wanVaeChunkCount(layout, fixture.latents);
      // The poison is refilled before each decode (rules out the second decode reading back the frames the first one
      // wrote and faking a match).
      await poisonFrames(liveCaches, chunks);
      const product = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
      await poisonFrames(liveCaches, chunks);
      const looped = await decodeWithFault(gpu, sessions, liveCaches, fixture.latents, {
        kind: "none",
      });
      assertEquals(
        looped.filter((value) => !Number.isFinite(value)).length,
        0,
        "non-finite (frames the loop did not copy)",
      );
      assert(bitsEqual(product, looped), "the test's loop differs from the product path");
    });

    it("fault injection: using first for chunk 2 onwards → the enqueue is rejected by the size of copyOutputs", async () => {
      const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
      // first's output frames are [12,1,…] and cannot be copied into the resident frames for next ([12,4,…]) — the
      // runtime's size check of `copyOutputs` fails before any dispatch is queued (red before any value is compared).
      const error = await assertRejects(
        () =>
          decodeWithFault(gpu, sessions, liveCaches, fixtures.accept.latents, {
            kind: "first-for-all",
          }),
        ExecutionError,
        "copyOutputs",
      );
      // The rejection must be the mismatch between chunk 1's resident frames and first's frame shape (not to be confused
      // with a rejection of another copy target).
      assertStringIncludes(error.message, "'frame 1'");
      assertStringIncludes(error.message, `[${layout.first.frameShape.join(",")}]`);
    });
  },
);

// Upstream parity (the chunk sequence against the diffusers `_decode` under the absolute band), so it
// is opt-in, together with the cache fault injections that only prove the band is live and the
// band-candidate step. It writes the fault/* and band-candidate results.json ids; the per-case ids
// belong to the default suite above, and writing them here would duplicate its settlements.
describe.skipIf(!CHUNK_ASSETS_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan2.2 TI2V VAE chunk sequence: first → next with resident caches matches upstream's _decode (before unpatchify) within the band (real GPU)" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const chunkRun = registerChunkSessions();
    /** The maxAbs per case (material for the band candidate — recorded before the verdict, so runs outside the band are kept too). */
    const observed = new Map<ChunkCaseName, number>();

    const compareChunks = (name: ChunkCaseName, got: Float32Array) => {
      const fixture = chunkRun().fixtures[name];
      assertEquals(got.length, fixture.frames.length, `${name}: element count`);
      const report = allclose(got, fixture.frames, CHUNK_TOLERANCE);
      const referenceMax = fixture.frames.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
    };

    for (const { name, chunks: expectedChunks } of CHUNK_CASES) {
      it(`${name}: the chunk sequence matches upstream within the band (the frames start as poison values)`, async () => {
        const { gpu, layout, sessions, caches: liveCaches, fixtures } = chunkRun();
        const fixture = fixtures[name];
        const chunks = wanVaeChunkCount(layout, fixture.latents);
        assertEquals(chunks, expectedChunks, `chunk count of ${name}`);
        await poisonFrames(liveCaches, chunks);
        const got = await decodeWanVaeTile(gpu, sessions, liveCaches, fixture.latents);
        const { report, referenceMax, ratio } = compareChunks(name, got);
        console.log(
          `[wan-ti2v-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
            `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, `${name}: non-finite (frames that were not copied)`);
        observed.set(name, report.maxAbsError);
        if (CHUNK_BAND === undefined) {
          throw new Error(
            `${name}: band not derived yet (maxAbs ${report.maxAbsError}) — write band's maxAbs × 5 into the band ` +
              "constant (the last step prints the band candidate)",
          );
        }
        assert(
          report.pass,
          `${name}: maxAbs ${report.maxAbsError} is outside the band ${CHUNK_BAND}`,
        );
      });
    }

    const faults: readonly { readonly label: string; readonly fault: Fault }[] = [
      {
        label: "a cache not updated (one dropped from copyOutputs)",
        fault: { kind: "drop-cache", cache: DROPPED_CACHE },
      },
      { label: "the 2 frames of a cache reversed", fault: { kind: "swap-cache-frames" } },
      { label: "the caches not zeroed at the tile head", fault: { kind: "skip-zero" } },
    ];
    for (const { label, fault } of faults) {
      it(`fault injection: ${label} → accept is outside the band`, async () => {
        const { gpu, sessions, caches: liveCaches, fixtures } = chunkRun();
        await runRecordedCase(
          chunkResults,
          { id: `fault/${fault.kind}` },
          async ({ measurements }) => {
            if (fault.kind === "skip-zero") {
              // Run accept with the previous tile's (band's) caches left in place.
              await decodeWanVaeTile(gpu, sessions, liveCaches, fixtures.band.latents);
            }
            const got = await decodeWithFault(
              gpu,
              sessions,
              liveCaches,
              fixtures.accept.latents,
              fault,
            );
            const { report, ratio } = compareChunks("accept", got);
            measurements.push({
              output: `frames(fault ${fault.kind})`,
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance: CHUNK_TOLERANCE,
              stage: "karume",
            });
            console.log(
              `[wan-ti2v-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}` +
                (CHUNK_BAND === undefined
                  ? ""
                  : ` band×${(report.maxAbsError / CHUNK_BAND).toFixed(1)}`),
            );
            if (CHUNK_BAND === undefined) {
              throw new Error(
                `${label}: band not derived yet (fault-injection maxAbs ${report.maxAbsError})`,
              );
            }
            assert(
              !report.pass,
              `${label}: stayed inside the band ${CHUNK_BAND} (a sign the band is too wide)`,
            );
            return undefined;
          },
        );
      });
    }

    it.skipIf(CHUNK_BAND !== undefined)("band candidate (not derived yet)", () =>
      chunkBandCandidate(observed),
    );
  },
);

/** The two product Sessions, the cache residents and the tile fixtures one tile-decode case runs on. */
type TileRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  /** Sample pixels per latent pixel (VAE scale × patch size). */
  readonly compression: number;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
  readonly fixtures: Readonly<Record<CaseName, TileFixture>>;
  /** The device-lost reason so far (`undefined` while the device is alive). */
  readonly deviceLost: () => string | undefined;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and the tile
 * fixtures and builds both product Sessions and the cache residents on one device that records a
 * device loss, and an afterAll that tears them down, so the default suite and the opt-in parity suite
 * share one lifecycle and one teardown order. Returns the accessor the its read the run through.
 */
const registerTileSessions = (): (() => TileRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: TileRun | undefined;
  beforeAll(async () => {
    const models = await prepareModels();
    const layout = wanVaeChunkLayout(models.first, models.next);
    const compression = wanSpatialCompression(layout, WAN22_TI2V_GENERATION);
    const fixtures: Readonly<Record<CaseName, TileFixture>> = {
      band: await readTileFixture(ti2vVaeTileFixtureUrl("band")),
      accept: await readTileFixture(ti2vVaeTileFixtureUrl("accept")),
      wide: await readTileFixture(ti2vVaeTileFixtureUrl("wide")),
    };

    let deviceLost: string | undefined;
    const device = verifyRunningGpu(
      await acquireTestGpu({
        onDeviceLost: (info) => {
          deviceLost = `${info.reason}: ${info.message}`;
        },
      }),
    );
    gpu = device;
    first = await models.first.createContainerSession(device);
    next = await models.next.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = {
      gpu: device,
      layout,
      compression,
      sessions: { first, next },
      caches,
      fixtures,
      deviceLost: () => deviceLost,
    };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: tear down in the order Session → resident tensors → device (the same as the 2.1 / step-4 e2e — even when one
    // step fails, the remaining steps always run).
    // Vitest reports an afterAll failure apart from the its, so a teardown failure cannot overwrite a test
    // failure.
    await disposeSteps([
      () => first?.dispose(),
      () => next?.dispose(),
      () => caches?.dispose(),
      // Wait for the releases before discarding the device (leaves budget for the following tests — {@link settleReleases}).
      () => (device === undefined ? undefined : settleReleases(device)),
      () => device?.destroy(),
    ]);
  });
  return () => {
    if (run === undefined) throw new Error("tile-decode setup (beforeAll) did not complete");
    return run;
  };
};

/**
 * One tile case through the product tile decode (`decodeWanVaeTiled`), starting from NaN-poisoned
 * frames and poisoning them again between tiles.
 */
const decodeTiledCase = async (run: TileRun, name: CaseName) => {
  const latents = run.fixtures[name].tensor("latents");
  const plan = planFor(run.layout, latents.shape);
  const chunks = latents.shape[1];
  await poisonFrames(run.caches, chunks);
  const started = performance.now();
  const got = await decodeWanVaeTiled(
    run.gpu,
    run.sessions,
    run.caches,
    plan,
    latents.data,
    poisonBetweenTiles(run.caches, plan, chunks),
  );
  return { got, plan, chunks, elapsedMs: performance.now() - started };
};

/**
 * The accept case decoded tile by tile and assembled on the host, starting from NaN-poisoned frames
 * (the decoded tiles are returned for the host-side seam fault injections).
 */
const decodeAcceptCase = async (run: TileRun) => {
  const latents = run.fixtures.accept.tensor("latents");
  const plan = planFor(run.layout, latents.shape);
  const chunks = latents.shape[1];
  await poisonFrames(run.caches, chunks);
  const decodeStarted = performance.now();
  const tiles = await decodeWanVaeTiles(
    run.gpu,
    run.sessions,
    run.caches,
    plan,
    latents.data,
    poisonBetweenTiles(run.caches, plan, chunks),
  );
  const decodeMs = performance.now() - decodeStarted;
  const assembleStarted = performance.now();
  const got = assembleWanVaeTiles(tiles, plan);
  const assembleMs = performance.now() - assembleStarted;
  return { tiles, got, plan, chunks, decodeMs, assembleMs };
};

// Default run: the TS/Python plan equality, the degenerate Uint32 gate, and band / accept / wide
// decoded from NaN-poisoned frames with 0 non-finite and no device lost. It owns the per-case
// results.json ids; the band comparisons with the upstream tiled reference are the opt-in suite below.
describe.skipIf(!TILE_RUN_PRESENT || !GPU_AVAILABLE)(
  "Wan2.2 TI2V VAE tiled decode: the plan matches Python, degenerate check (Uint32), band / accept / wide run from " +
    "poisoned frames with 0 non-finite and no device lost (real GPU)",
  () => {
    let degenerate: Fixture | undefined;
    beforeAll(async () => {
      degenerate = await readFixture(ti2vVaeFixtureUrl(DEGENERATE_CASE));
    });
    const tileRun = registerTileSessions();
    /** Case name → non-finite count (check 4 — recorded before the verdict). */
    const nonFinite = new Map<string, number>();

    /** Records the non-finite count of one case, then fails when it is not 0. */
    const assertFinite = (label: string, got: Float32Array, want: Float32Array, extra: string) => {
      assertEquals(got.length, want.length, "element count");
      const count = countNonFinite(got);
      nonFinite.set(label, count);
      console.log(`[wan-ti2v-vae-tiles] ${label}: nonFinite=${count}${extra}`);
      if (count > 0) {
        throw new Error(`${label}: ${count} non-finite values (frames that were not copied)`);
      }
    };

    it("plan: the TS plan (from the assets' input shapes, 2.2's descriptor) matches the fixtures' metadata (Python)", () => {
      const { layout, compression, fixtures } = tileRun();
      assertEquals(
        [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
        [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      );
      assertWanVaeMatchesGeneration(layout, WAN22_TI2V_GENERATION, OWNER);
      for (const { name, role, chunks, height, width, full, rgb } of CASES) {
        const fixture = fixtures[name];
        const latents = fixture.tensor("latents");
        assertEquals(
          latents.shape,
          [LATENT_CHANNELS, chunks, height, width],
          `latent shape of ${name}`,
        );
        assertEquals(
          {
            role: fixture.meta("role"),
            weights: fixture.meta("weights"),
            reference: fixture.meta("reference"),
          },
          { role, weights: FIXTURE_WEIGHTS, reference: FIXTURE_REFERENCE },
          `role and provenance of ${name} (f16 rounding, reference in patchify space)`,
        );
        const plan = planFor(layout, latents.shape);
        const ours = planMeta(plan);
        const theirs = Object.fromEntries(Object.keys(ours).map((key) => [key, fixture.meta(key)]));
        assertEquals(ours, theirs, `${name} geometry`);
        const frames = wanVaeFrameCount(chunks);
        assertEquals(
          fixture.tensor("frames").shape,
          [SAMPLE_CHANNELS, frames, height * plan.scale, width * plan.scale],
          `frames shape of ${name} (patchify space)`,
        );
        assertEquals(fixture.has("frames_full"), full, `presence of frames_full in ${name}`);
        assertEquals(fixture.has("frames_rgb"), rgb, `presence of frames_rgb in ${name}`);
        if (rgb) {
          assertEquals(
            fixture.tensor("frames_rgb").shape,
            [RGB_CHANNELS, frames, height * compression, width * compression],
            `frames_rgb shape of ${name}`,
          );
        }
      }
      const band = planFor(layout, fixtures.band.tensor("latents").shape);
      assertEquals(wanVaeTileCount(band), 12);
      assertEquals([...band.rows.starts], [0, 7, 14]);
      assertEquals([...band.cols.starts], [0, 12, 24, 36]);
      assertEquals(wanVaeTileCount(planFor(layout, fixtures.wide.tensor("latents").shape)), 28);
    });

    it("degenerate check: for a 16×16 latent (one tile) the tiled path ≡ the untiled chunk sequence (Uint32)", async () => {
      const { gpu, layout, sessions, caches: liveCaches } = tileRun();
      assert(degenerate !== undefined, "the degenerate check's fixture (beforeAll) was not read");
      const expectedChunks = CHUNK_CASES.find(({ name }) => name === DEGENERATE_CASE)?.chunks;
      assert(
        expectedChunks !== undefined,
        `the table has no chunk-sequence case '${DEGENERATE_CASE}'`,
      );
      assertEquals(
        degenerate.latentShape,
        [LATENT_CHANNELS, expectedChunks, TILE, TILE],
        "latent shape of the degenerate check",
      );
      const plan = planFor(layout, degenerate.latentShape);
      assertEquals(wanVaeTileCount(plan), 1);
      const chunks = wanVaeChunkCount(layout, degenerate.latents);
      await poisonFrames(liveCaches, chunks);
      const tiled = await decodeWanVaeTiled(gpu, sessions, liveCaches, plan, degenerate.latents);
      // The poison is refilled before each decode (rules out the second decode reading back the frames the first one
      // wrote and faking a match).
      await poisonFrames(liveCaches, chunks);
      const direct = await decodeWanVaeTile(gpu, sessions, liveCaches, degenerate.latents);
      assertEquals(
        tiled.length,
        SAMPLE_CHANNELS * wanVaeFrameCount(chunks) * SAMPLE_TILE * SAMPLE_TILE,
      );
      assertEquals(countNonFinite(tiled), 0, "non-finite (frames that were not copied)");
      assert(bitsEqual(tiled, direct), "the tiled path is not bit-identical to the chunk sequence");
    });

    for (const name of ["band", "wide"] as const) {
      it(`${name}: run from poisoned frames, 0 non-finite (tiled path decodeWanVaeTiled)`, async () => {
        const run = tileRun();
        await runRecordedCase(results, { id: name }, async () => {
          const { got, plan, chunks, elapsedMs } = await decodeTiledCase(run, name);
          assertFinite(
            name,
            got,
            run.fixtures[name].tensor("frames").data,
            ` tiles=${wanVaeTileCount(plan)} chunks=${chunks} decodeMs=${elapsedMs.toFixed(0)}`,
          );
          return undefined;
        });
      });
    }

    it("accept: run from poisoned frames, 0 non-finite (per-tile decode + host assembly)", async () => {
      const run = tileRun();
      await runRecordedCase(results, { id: "accept" }, async () => {
        const { got, plan, chunks, decodeMs, assembleMs } = await decodeAcceptCase(run);
        assertFinite(
          "accept",
          got,
          run.fixtures.accept.tensor("frames").data,
          ` tiles=${wanVaeTileCount(plan)} chunks=${chunks} decodeMs=${decodeMs.toFixed(
            0,
          )} assembleMs=${assembleMs.toFixed(0)}`,
        );
        return undefined;
      });
    });

    it("poison values and device lost: 0 non-finite and no device lost in every case (band / accept / wide)", () => {
      const run = tileRun();
      const names = CASES.map(({ name }) => name);
      assertEquals(
        names.filter((name) => !nonFinite.has(name)),
        [],
        "cases that failed before their non-finite values were counted",
      );
      assertEquals(
        names.map((name) => nonFinite.get(name)),
        names.map(() => 0),
        `non-finite counts (${names.join(" / ")})`,
      );
      assertEquals(run.deviceLost(), undefined, "device lost");
    });
  },
);

// Upstream parity (band / accept / wide and the accept RGB against the upstream tiled reference under
// the absolute band), so it is opt-in, together with the seam fault injections that only prove the
// band is live and the band-candidate step. It writes the accept-rgb, fault/* and band-candidate
// results.json ids; the per-case ids belong to the default suite above, and writing them here would
// duplicate its settlements.
describe.skipIf(!TILE_RUN_PRESENT || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan2.2 TI2V VAE tiled decode: degenerate check + 832×480, 81 frames match the upstream tile reference (before " +
    "unpatchify) within the band, and accept's RGB matches upstream's unpatchify → clamp within the band (real GPU)" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    const tileRun = registerTileSessions();
    /** Label (case name / `accept-rgb`) → maxAbs (material for the band candidate — recorded before the verdict, so runs outside the band are kept too). */
    const observed = new Map<string, number>();
    /** Fault-injection label → maxAbs (a record — judged only once the band is derived). */
    const faultObserved = new Map<string, number>();

    /**
     * The verdict of one case (after recording its maxAbs — so a run with the band not derived still leaves material for
     * the candidate). Fails in the order non-finite → band not derived → outside the band.
     */
    const judge = (
      label: string,
      got: Float32Array,
      want: Float32Array,
      extra = "",
      recorded?: {
        readonly measurements: RecordedCase["measurements"];
        readonly output: string;
      },
    ): void => {
      const { report, referenceMax, ratio } = compare(got, want, TOLERANCE);
      recorded?.measurements.push({
        output: recorded.output,
        maxAbs: report.maxAbsError,
        maxRel: report.maxRelError,
        tolerance: TOLERANCE,
        stage: "karume",
      });
      observed.set(label, report.maxAbsError);
      console.log(
        `[wan-ti2v-vae-tiles] ${label}: maxAbs=${report.maxAbsError} refMax=${referenceMax} ` +
          `ratio=${ratio} nonFinite=${report.nonFiniteCount}${extra}`,
      );
      if (report.nonFiniteCount > 0) {
        throw new Error(
          `${label}: ${report.nonFiniteCount} non-finite values (frames that were not copied)`,
        );
      }
      if (BAND === undefined) {
        throw new Error(
          `${label}: band not derived yet (maxAbs ${report.maxAbsError}) — write band's maxAbs × 5 into the band constant` +
            " (the last step prints the band candidate)",
        );
      }
      assert(report.pass, `${label}: maxAbs ${report.maxAbsError} is outside the band ${BAND}`);
    };

    it("band: 832×480, 81 frames (12 tiles × 21 chunks) match the reference within the band (the frames start as poison values)", async () => {
      const run = tileRun();
      const { got, elapsedMs } = await decodeTiledCase(run, "band");
      judge(
        "band",
        got,
        run.fixtures.band.tensor("frames").data,
        ` decodeMs=${elapsedMs.toFixed(0)}`,
      );
    });

    // Acceptance: keep the decoded tiles and reuse them for the comparison of the VAE stage's end and the fault
    // injections (host-side assembly).
    /** The accept fixture, its reference frames, plan and chunk count (derived from the run's fixtures). */
    const acceptCase = () => {
      const { layout, fixtures } = tileRun();
      const accept = fixtures.accept;
      const latents = accept.tensor("latents");
      return {
        accept,
        want: accept.tensor("frames").data,
        plan: planFor(layout, latents.shape),
        chunks: latents.shape[1],
      };
    };
    let acceptTiles: Float32Array<ArrayBuffer>[] = [];
    let acceptAssembled: Float32Array<ArrayBuffer> | undefined;

    it("accept: 480×832, 9 frames (different latent, chunk boundaries, portrait) are inside the band", async () => {
      const { want } = acceptCase();
      const { tiles, got, decodeMs, assembleMs } = await decodeAcceptCase(tileRun());
      acceptTiles = tiles;
      acceptAssembled = got;
      judge(
        "accept",
        got,
        want,
        ` decodeMs=${decodeMs.toFixed(0)} assembleMs=${assembleMs.toFixed(0)}`,
      );
    });

    it("accept RGB: the end of the VAE stage (finishWanVaeFrames — unpatchify → clamp) matches upstream's unpatchify → clamp within the band", async () => {
      const { compression } = tileRun();
      const { accept, plan: acceptPlan, chunks: acceptChunks } = acceptCase();
      await runRecordedCase(results, { id: "accept-rgb" }, ({ measurements }) => {
        assert(acceptAssembled !== undefined, "accept was not assembled");
        const spec = CASES.find(({ name }) => name === "accept");
        assert(spec !== undefined, "the table has no accept case");
        const video = finishWanVaeFrames(
          acceptAssembled,
          {
            frames: wanVaeFrameCount(acceptChunks),
            width: spec.width * compression,
            height: spec.height * compression,
            tiles: acceptPlan,
          },
          WAN22_TI2V_GENERATION,
          OWNER,
        );
        judge("accept-rgb", video, accept.tensor("frames_rgb").data, "", {
          measurements,
          output: `frames_rgb(${acceptChunks} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
        });
        return Promise.resolve(undefined);
      });
    });

    it("wide: 1280×704, 5 frames (28 tiles, a different blend width per pair) are inside the band", async () => {
      const run = tileRun();
      const { got, elapsedMs } = await decodeTiledCase(run, "wide");
      judge(
        "wide",
        got,
        run.fixtures.wide.tensor("frames").data,
        ` decodeMs=${elapsedMs.toFixed(0)}`,
      );
    });

    const faults: readonly {
      readonly id: string;
      readonly label: string;
      readonly plan: (acceptPlan: WanVaeTilePlan) => WanVaeTilePlan;
    }[] = [
      {
        // Shows in the values only for portrait frames (an involution for square ones). The tile counts agree as
        // 4×3 = 3×4, so it slips past the shape checks.
        id: "swap-axes",
        label: "rows and columns swapped",
        plan: (acceptPlan) => ({ ...acceptPlan, rows: acceptPlan.cols, cols: acceptPlan.rows }),
      },
      {
        // What a rounding disagreement between TS and Python looks like (the decoded position and the pasted position
        // differ by one latent).
        id: "shift-start",
        label: "start position off by one latent",
        plan: (acceptPlan) => ({
          ...acceptPlan,
          rows: {
            ...acceptPlan.rows,
            starts: acceptPlan.rows.starts.map((start, index) => (index === 1 ? start + 1 : start)),
          },
        }),
      },
    ];
    for (const { id, label, plan } of faults) {
      it(`fault injection: ${label} → accept is outside the band`, async () => {
        const { want: acceptWant, plan: acceptPlan } = acceptCase();
        await runRecordedCase(results, { id: `fault/${id}` }, ({ measurements }) => {
          assert(acceptTiles.length > 0, "the accept tiles were not decoded");
          const got = assembleWanVaeTiles(acceptTiles, plan(acceptPlan));
          const { report, ratio } = compare(got, acceptWant, TOLERANCE);
          measurements.push({
            output: `frames(fault ${id})`,
            maxAbs: report.maxAbsError,
            maxRel: report.maxRelError,
            tolerance: TOLERANCE,
            stage: "karume",
          });
          faultObserved.set(label, report.maxAbsError);
          console.log(
            `[wan-ti2v-vae-tiles] fault ${id}: maxAbs=${report.maxAbsError} ratio=${ratio}` +
              (BAND === undefined ? "" : ` band×${(report.maxAbsError / BAND).toFixed(1)}`),
          );
          if (BAND === undefined) {
            throw new Error(
              `${label}: band not derived yet (fault-injection maxAbs ${report.maxAbsError})`,
            );
          }
          assert(
            !report.pass,
            `${label}: stayed inside the band ${BAND} (a sign the band is too wide)`,
          );
          return Promise.resolve(undefined);
        });
      });
    }

    it.skipIf(BAND !== undefined)("band candidate (not derived yet)", () =>
      bandCandidate(observed, faultObserved),
    );
  },
);
