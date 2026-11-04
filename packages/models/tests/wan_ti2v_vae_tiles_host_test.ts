// The host side of the tiled VAE decode of Wan2.2 TI2V-5B (ADR 0121 stage 5 — no GPU needed).
//
// Four things are checked:
//
// - **Frozen tile plan**: the start positions for tile 16 and overlap = 64 px ÷ spatial compression 16 (graph ratio 8 ×
//   unpatchify 2) = 4 latents are frozen by value ({@link AXIS_STARTS_TI2V} — frozen twice with the same values as
//   `MIRRORED_STARTS_TI2V` in the Python side's `tools/export-recipes/wan/tests/test_vae_tiling_ti2v.py`). The plan
//   goes through the production entry point `planWanGenerationTiles`. The 832×480 family is not in the descriptor's
//   accepted set (outside the official supported sizes), but this function does not look at the accepted set, so it
//   is frozen through the same entry point (stage 5's tile references use it as a numeric comparison that does not
//   depend on the accepted sizes).
// - **Request check**: passes the 2.2 descriptor to `planWanRequest` and pins by value the accepted sizes
//   (1280×704 / 704×1280), the frame-count range (4n+1 in 5..121) and the defaults (1280×704, 33 frames).
// - **Copy of the denormalization statistics**: `WAN22_LATENTS_MEAN` / `WAN22_LATENTS_STD` are bit-identical to the
//   fixture `fixtures/wan-latents/wan22-ti2v.json` (on the other side, the recipe's
//   `wan/tests/test_ti2v_generation_stats.py` compares it with the pinned config).
// - **Real-asset layout** (only on machines with the two chunk graphs): the layout derived from the declarations
//   matches the plan's literal, passes the 2.2 descriptor's admission, and fails with the 2.1 descriptor.
//
// The real-asset case reads the two f16 chunk graphs of the distribution mirror `models/karume-wan2.2/` (through its
// manifest — ADR 0139 decision 1) and SKIPs explicitly without the mirror (ADR 0005; the mirror's completeness is the
// distribution gate's). The presence of the tile references `vae_tiles.*` is not checked here (the GPU check's group).

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { prepareContainer, type PreparedModel } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import type { WanPipelineConfig } from "../src/wan/config.ts";
import { WAN21_GENERATION, WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import { WAN22_LATENTS_MEAN, WAN22_LATENTS_STD } from "../src/wan/latents.ts";
import { planWanRequest, type PromptGate } from "../src/wan/plan.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../src/wan/tile-decode.ts";
import { wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import { wanVaeBlendExtentAt, wanVaeTileCount } from "../src/wan/vae-tiles.ts";
import { mirrorPresent } from "./helpers/mirror.ts";
import { WAN_TI2V_VAE_FIRST as FIRST, WAN_TI2V_VAE_NEXT as NEXT } from "./helpers/wan-ti2v-vae.ts";
import {
  openWanTi2vComponent,
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "./helpers/wan-ti2v-pipeline.ts";

const OWNER = "wan.Ti2vPipeline";

/**
 * Literal of the 2.2 chunk graphs' declarations (tile 16, frame edge 128 in patchify space, latent 48 ch,
 * output 12 ch).
 */
const LAYOUT = { latentChannels: 48, tile: 16, sampleTile: 128, sampleChannels: 12 } as const;
/**
 * Spatial ratio of the graph's output to its input (`sampleTile / tile`) — blend widths count pixels in
 * patchify space.
 */
const GRAPH_SCALE = 8;

/**
 * Start positions for tile edge 16 and minimum overlap 4 (latent extent → start positions). **The same values** as the
 * Python side's `MIRRORED_STARTS_TI2V`.
 *
 * - 16: degenerate (one tile)
 * - 30 / 52: the two edges of 832×480 / 480×832
 * - 44 / 80: the two edges of 1280×704 / 704×1280 (a layout whose blend width differs per pair)
 */
const AXIS_STARTS_TI2V: readonly (readonly [number, readonly number[]])[] = [
  [16, [0]],
  [30, [0, 7, 14]],
  [44, [0, 9, 19, 28]],
  [52, [0, 12, 24, 36]],
  [80, [0, 11, 21, 32, 43, 53, 64]],
];

/** Plans one axis (extent × degenerate 16) through the production entry point and takes the row start positions. */
const rowStarts = (extent: number, generation = WAN22_TI2V_GENERATION): readonly number[] =>
  planWanGenerationTiles(LAYOUT, extent, LAYOUT.tile, generation).rows.starts;

test("Wan2.2 tile plan: freezes the start positions by value (the same table as the Python side, through the production entry point)", () => {
  for (const [extent, starts] of AXIS_STARTS_TI2V) {
    assertEquals([...rowStarts(extent)], [...starts], `latent ${extent}`);
  }
});

test("Wan2.2 tile plan: the spatial compression is 16 (graph ratio 8 × unpatchify 2) and tiles cover the two accepted sizes", () => {
  assertEquals(wanSpatialCompression(LAYOUT, WAN22_TI2V_GENERATION), 16);
  assertWanVaeTilesCover(LAYOUT, WAN22_TI2V_GENERATION, OWNER);
});

test("Wan2.2 request check: passes 1280×704 / 704×1280 × up to 121 frames, and rejects 832×480 and 125 frames with the accepted-set message", () => {
  const gate: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };
  const config: WanPipelineConfig = {
    scheduler: { shift: 5 },
    defaults: { steps: 50, guidance: 5 },
  };
  const planOf = (size: { width?: number; height?: number; frames?: number }) =>
    planWanRequest({ prompt: "p", ...size }, gate, LAYOUT, config, WAN22_TI2V_GENERATION);

  // The defaults (1280×704, 33 frames) and portrait pass; the latent shape and the tile count follow from the size.
  const byDefault = planOf({});
  assertEquals([byDefault.width, byDefault.height, byDefault.frames], [1280, 704, 33]);
  assertEquals(byDefault.latentShape, [48, 9, 44, 80]);
  assertEquals(wanVaeTileCount(byDefault.tiles), 28);
  assertEquals(planOf({ width: 704, height: 1280, frames: 33 }).latentShape, [48, 9, 80, 44]);

  // The upper limit is the official default 121 (the value confirmed to complete 50 steps on the development machine's
  // RTX 3080 Ti). 49 and 121 pass in both orientations.
  assertEquals(planOf({ frames: 49 }).latentShape, [48, 13, 44, 80]);
  assertEquals(planOf({ width: 704, height: 1280, frames: 49 }).latentShape, [48, 13, 80, 44]);
  assertEquals(planOf({ frames: 121 }).latentShape, [48, 31, 44, 80]);
  assertEquals(planOf({ width: 704, height: 1280, frames: 121 }).latentShape, [48, 31, 80, 44]);

  // Sizes outside the official supported ones (832×480 / 480×832) and frame counts above the limit fail as
  // input errors.
  for (const [width, height] of [
    [832, 480],
    [480, 832],
  ]) {
    assertThrows(
      () => planOf({ width, height }),
      ModelInputError,
      `${width}×${height} is not in the accepted set (1280×704 / 704×1280)`,
    );
  }
  assertThrows(
    () => planOf({ frames: 125 }),
    ModelInputError,
    "frames 125 is not in the accepted set (4n+1 in 5..121)",
  );

  // The lower limit is 5 (5 passes; 1 fails even though it is 4n+1). Values that are not 4n+1 fail too.
  assertEquals(planOf({ frames: 5 }).latentShape, [48, 2, 44, 80]);
  for (const frames of [1, 31]) {
    assertThrows(
      () => planOf({ frames }),
      ModelInputError,
      `frames ${frames} is not in the accepted set (4n+1 in 5..121)`,
    );
  }
});

/** Plans of the four sizes (output pixels → tile count, start positions, blend widths in patchify-space pixels). */
const SIZES: readonly {
  readonly width: number;
  readonly height: number;
  readonly count: number;
  readonly rows: readonly number[];
  readonly cols: readonly number[];
  readonly rowBlend: readonly number[];
  readonly colBlend: readonly number[];
}[] = [
  {
    width: 832,
    height: 480,
    count: 12,
    rows: [0, 7, 14],
    cols: [0, 12, 24, 36],
    rowBlend: [72, 72],
    colBlend: [32, 32, 32],
  },
  {
    width: 480,
    height: 832,
    count: 12,
    rows: [0, 12, 24, 36],
    cols: [0, 7, 14],
    rowBlend: [32, 32, 32],
    colBlend: [72, 72],
  },
  {
    width: 1280,
    height: 704,
    count: 28,
    rows: [0, 9, 19, 28],
    cols: [0, 11, 21, 32, 43, 53, 64],
    rowBlend: [56, 48, 56],
    colBlend: [40, 48, 40, 40, 48, 40],
  },
  {
    width: 704,
    height: 1280,
    count: 28,
    rows: [0, 11, 21, 32, 43, 53, 64],
    cols: [0, 9, 19, 28],
    rowBlend: [40, 48, 40, 40, 48, 40],
    colBlend: [56, 48, 56],
  },
];

test("Wan2.2 tile plan: 832×480 / 480×832 take 12 tiles and 1280×704 / 704×1280 take 28 (count, start positions, blend width per pair)", () => {
  const compression = wanSpatialCompression(LAYOUT, WAN22_TI2V_GENERATION);
  for (const { width, height, count, rows, cols, rowBlend, colBlend } of SIZES) {
    const where = `${width}×${height}`;
    const plan = planWanGenerationTiles(
      LAYOUT,
      height / compression,
      width / compression,
      WAN22_TI2V_GENERATION,
    );
    assertEquals(
      [plan.latentChannels, plan.sampleChannels, plan.scale],
      [48, 12, GRAPH_SCALE],
      where,
    );
    assertEquals(wanVaeTileCount(plan), count, `${where}: tile count`);
    assertEquals([...plan.rows.starts], [...rows], `${where}: rows`);
    assertEquals([...plan.cols.starts], [...cols], `${where}: columns`);
    assertEquals(
      rowBlend.map((_, index) => wanVaeBlendExtentAt(plan.rows, GRAPH_SCALE, index + 1)),
      [...rowBlend],
      `${where}: row blend widths`,
    );
    assertEquals(
      colBlend.map((_, index) => wanVaeBlendExtentAt(plan.cols, GRAPH_SCALE, index + 1)),
      [...colBlend],
      `${where}: column blend widths`,
    );
  }
});

test("Wan2.2 tile plan fault injection: deriving the overlap from the graph ratio 8 alone (ignoring unpatchify, so 8) breaks the rows of latents 44 / 52 / 80", () => {
  // A descriptor that drops the unpatchify factor = compression 8, overlap 8. Latents 16 / 30 get the same start
  // positions with overlap 4 or 8, so only the rows of 44 / 52 / 80 tell this injection apart (they are named).
  const graphScaleOnly = { ...WAN22_TI2V_GENERATION, vaePatchSize: 1 };
  const broken = AXIS_STARTS_TI2V.filter(([extent, starts]) => {
    const injected = rowStarts(extent, graphScaleOnly);
    return (
      injected.length !== starts.length || injected.some((start, index) => start !== starts[index])
    );
  }).map(([extent]) => extent);
  assertEquals(broken, [44, 52, 80]);
});

// ---- Copy of the denormalization statistics (checked from both sides of the fixture) -------------------------

type LatentStatsFixture = {
  readonly source: { readonly repo: string; readonly revision: string; readonly reference: string };
  readonly mean: readonly number[];
  readonly std: readonly number[];
};

test("WAN22_LATENTS_MEAN / STD: bit-identical to the fixture wan22-ti2v.json (48 values each; the fixture's values are exact f32)", async () => {
  const fixture = JSON.parse(
    await readTextFile(new URL("./fixtures/wan-latents/wan22-ti2v.json", import.meta.url)),
  ) as LatentStatsFixture;
  assertEquals(fixture.source.repo, "Wan-AI/Wan2.2-TI2V-5B-Diffusers");
  for (const [name, constant, values] of [
    ["mean", WAN22_LATENTS_MEAN, fixture.mean],
    ["std", WAN22_LATENTS_STD, fixture.std],
  ] as const) {
    assertEquals(constant.length, LAYOUT.latentChannels, `${name}: count`);
    assertEquals(
      [...constant],
      [...Float32Array.from(values)],
      `${name}: the constant and the fixture (f32) disagree — only one side was rewritten`,
    );
    // The fixture itself is in shortest f32 notation (a value that changes when rounded to f32 means the config was
    // copied wrongly).
    assertEquals([...values], [...Float32Array.from(values)], `${name}: the fixture is not f32`);
  }
  assert(
    WAN22_TI2V_GENERATION.latents.mean === WAN22_LATENTS_MEAN,
    "the descriptor holds a different mean",
  );
  assert(
    WAN22_TI2V_GENERATION.latents.std === WAN22_LATENTS_STD,
    "the descriptor holds a different std",
  );
});

// ---- Real-asset layout (the mirror's two f16 chunk graphs) ----------------------------------------------

const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the real-asset layout check of Wan2.2 TI2V's VAE tiles is SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
}

/** Opens and prepares the mirror's f16 chunk graph (the graph name is the component key). */
const prepared = async (component: string): Promise<PreparedModel> =>
  prepareContainer(await openWanTi2vComponent(component, "f16"), component);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE tiles on the real assets (no GPU needed): the layout matches the plan's literal, passes the 2.2 " +
    "descriptor's admission, and fails with the 2.1 descriptor",
  async () => {
    const layout = wanVaeChunkLayout(await prepared(FIRST), await prepared(NEXT));
    assertEquals(
      {
        latentChannels: layout.latentChannels,
        tile: layout.tile,
        sampleTile: layout.sampleTile,
        sampleChannels: layout.sampleChannels,
      },
      { ...LAYOUT },
    );
    assertWanVaeMatchesGeneration(layout, WAN22_TI2V_GENERATION, OWNER);
    assertWanVaeTilesCover(layout, WAN22_TI2V_GENERATION, OWNER);
    assertThrows(
      () => assertWanVaeMatchesGeneration(layout, WAN21_GENERATION, OWNER),
      Error,
      `${OWNER}: the denormalization statistics (mean 16, std 16) do not match the VAE latent's 48 channels`,
    );
    // It fails on the output channel count too, not only on the number of statistics (a descriptor with 2.1's factor
    // and 48 statistics).
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layout,
          { ...WAN21_GENERATION, latents: WAN22_TI2V_GENERATION.latents },
          OWNER,
        ),
      Error,
      `${OWNER}: VAE output 12 channels differ from RGB 3 × unpatchify 1² = 3`,
    );
  },
);
