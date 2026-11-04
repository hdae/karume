/**
 * The reference comparison of Wan2.1's video VAE **tiled decode** (real GPU — the acceptance of ADR 0118 step 5).
 *
 * The two chunk graphs (stage 4) of the local distribution mirror `models/karume-wan2.1/` (opened through its manifest —
 * ADR 0139 decision 1; the fixtures are in the test-asset root `goldens/wan2.1-t2v-1.3b-f16-dyn/` — decision 2) are
 * tiled over the whole frame with the host tile plan (`vae-tiles.ts`) and run — tiles outside, chunks inside, one tile
 * = one batch, only the frames read back (`decodeWanVaeTile`). The comparison target is the recipe's tile reference
 * (`tools/export-recipes/wan/vae_tiling.py` — the same geometry, upstream's blend verbatim, CPU f32, weights rounded
 * to f16 — ADR 0006) **before the clamp**.
 *
 * ## Checks
 *
 * - **Plan**: the TS plan (derived from the assets' input shapes) matches the fixtures' metadata (the Python plan).
 * - **Degenerate check**: for a 32×32 latent (a single tile — the latent of the step-4 fixture) the tiled path
 *   (`decodeWanVaeTiled`) is **Uint32-identical** to the untiled chunk sequence (`decodeWanVaeTile`)
 *   (ADR 0033 decision 4 — pins the identity of cropping and pasting structurally).
 * - **Band**: the band is set from the measured worst case of a plain comparison with `atol = rtol = 0` on
 *   `band` (832×480, 33 frames, 12 tiles) ({@link BAND}), and acceptance is judged on `accept` with **a different
 *   latent, different chunk boundaries and portrait orientation** (480×832, 9 frames) (ADR 0118 addendum 2026-10-02).
 * - **Poison values**: the resident frames are filled with NaN before the run, and the non-finite count must be 0 (a
 *   frame that was not copied stays NaN).
 * - **Fault injection** (host side — the decoded tiles of `accept` are assembled with a fault): swapping rows and
 *   columns and shifting a start position by one latent must land outside the band (if not, the band is too wide).
 *
 * Upstream parity is opt-in (`KARUME_UPSTREAM_PARITY=1` — helpers/upstream-parity.ts): the `band` /
 * `accept` frames against the recipe's tiled reference under {@link BAND}, and in the chunk section
 * the chunk sequences against the untiled `_decode` with the cache fault injections that prove that
 * band is live. The plan, the degenerate gate, the NaN poison and the seam fault injections (the
 * host-side assembly against the tiled reference) run by default.
 *
 * An environment with no assets at all SKIPs explicitly; one with only some is a FAIL (the same rule as the step-4 e2e).
 *
 * This file also holds the step-4 check (single-tile chunk sequence vs the untiled upstream
 * `_decode`, with its own band, cache fault injections and the 81-frame `long` case) — see the
 * "Chunk sequence" section below. Both checks share the series and the two chunk graphs.
 */

import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  ExecutionError,
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type Session,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  type WanVaeChunkLayout,
  wanVaeChunkLayout,
  wanVaeFrameCount,
} from "../../models/src/wan/vae-chunks.ts";
import {
  assembleWanVaeTiles,
  decodeWanVaeTiled,
  decodeWanVaeTiles,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../../models/src/wan/vae-tiles.ts";
import { WAN21_GENERATION } from "../../models/src/wan/descriptor.ts";
import { planWanGenerationTiles } from "../../models/src/wan/tile-decode.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import {
  bitsEqual,
  decodeWithFault,
  type Fault,
  readFixture as readChunkFixture,
} from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import { allclose, type Tolerance } from "../../runtime/src/reference/allclose.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import {
  openWanComponent,
  WAN_ASSEMBLE_COMMAND,
  WAN_DIST_ROOT,
} from "../../models/tests/helpers/wan-distribution.ts";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The band of the tiled decode (absolute — the output is the pre-clamp frames, in a range of about ±1.5).
 *
 * Measured (B570, Deno 2.9.6, plain comparison with `atol = rtol = 0`, 2026-10-02):
 *
 * | Case   | Role       | Latent            | Tiles × chunks | maxAbs  | Reference max | Ratio (maxAbs ÷ reference max) |
 * | ------ | ---------- | ----------------- | -------------- | ------- | ------------- | ------------------------------ |
 * | band   | band       | `[16,9,60,104]`   | 12 × 9         | 3.87e-6 | 1.462         | 2.65e-6                        |
 * | accept | acceptance | `[16,3,104,60]`   | 12 × 3         | 4.16e-6 | 1.319         | 3.15e-6                        |
 *
 * The band 2e-5 is about 5.2 × the measured worst of `band` (the same order as the step-4 chunk-sequence band 2.5e-5 —
 * the blend is a convex combination, so it does not amplify the GPU error). The ratio is at most 1/32 of the prior
 * guideline (1e-4 — decision 8). The acceptance case slightly exceeding the band case's measured worst (4.16e-6 >
 * 3.87e-6) is due to the different latent — it is inside the band, and the band does not move (MUST: the band is not
 * re-derived from the acceptance result — addendum 2026-10-02). The fault injections (rows and columns swapped 2.08, a
 * start position off by one latent 1.96) are 5 orders of magnitude outside the band.
 */
const BAND = 2e-5;
const TOLERANCE: Tolerance = { atol: BAND, rtol: 0 };

/** The fixtures' place: the test-asset root's folder of the f16 series (ADR 0139 decision 2). */
const GOLDEN_ROOT = goldenDir("wan2.1-t2v-1.3b-f16-dyn");
/** Component names: the manifest's weights keys, which are also the containers' graph names (container-v1 §2.1). */
const FIRST_COMPONENT = "vae_decoder_first";
const NEXT_COMPONENT = "vae_decoder_next";
const CASES = ["band", "accept"] as const;
type CaseName = (typeof CASES)[number];

/** The latent of the degenerate check (the step-4 chunk-sequence fixture — latent `[16,9,32,32]`, exactly one tile). */
const DEGENERATE_FIXTURE = "vae_chunks.band.safetensors";

/** The command that fetches the fixtures into the local test-asset root (ADR 0139 decision 3). */
const FETCH = "node tools/test-assets/fetch.ts";
/**
 * A complete fetch of the pinned test-asset dataset makes a missing fixture a FAIL (the completeness tests below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_FIXTURES = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;

const fixtureUrl = (name: CaseName): URL => new URL(`vae_tiles.${name}.safetensors`, GOLDEN_ROOT);
const degenerateUrl = new URL(DEGENERATE_FIXTURE, GOLDEN_ROOT);

const fileExists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MIRROR_PRESENT = mirrorPresent(WAN_DIST_ROOT);
if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${WAN_DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so Wan's VAE tile-decode and chunk-sequence checks are SKIPPED. Assemble it: ${WAN_ASSEMBLE_COMMAND}`,
  );
}
const PRESENT = [fileExists(degenerateUrl), ...CASES.map((name) => fileExists(fixtureUrl(name)))];
const ANY_PRESENT = PRESENT.some(Boolean);
/** The tile checks read the mirror's two chunk graphs and these three fixtures. */
const TILES_RUNNABLE = MIRROR_PRESENT && ANY_PRESENT;
if (!ANY_PRESENT) {
  console.warn(
    `[karume] no VAE tile reference under ${GOLDEN_ROOT.pathname}, so Wan's VAE tile-decode checks are SKIPPED ` +
      `(${MISSING_FIXTURES})`,
  );
}

// An environment with none "has not generated them", so it SKIPs. With any one present a missing one is a FAIL (so is any
// missing one in a complete fetch of the pinned dataset; the mirror's completeness is the distribution gate's).
test.skipIf(!ANY_PRESENT && !FETCHED)(
  "Wan VAE tile fixtures: the three fixtures are present",
  () => {
    assertEquals(
      PRESENT,
      PRESENT.map(() => true),
      `missing: ${[degenerateUrl, ...CASES.map(fixtureUrl)]
        .filter((_, index) => !PRESENT[index])
        .map(({ pathname }) => pathname)
        .join(", ")} (${MISSING_FIXTURES})`,
    );
  },
);

/** One safetensors file (F32 tensors and metadata). */
type Fixture = {
  readonly tensor: (key: string) => {
    readonly data: Float32Array<ArrayBuffer>;
    readonly shape: readonly number[];
  };
  readonly has: (key: string) => boolean;
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
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
 * Renders the plan's geometry in the same spelling as the fixtures' metadata (`vae_tiling.TilePlan.meta`) (the TS ↔
 * Python comparison — sequences in the form `"0,14,28"`).
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
  };
};

const absMax = (values: Float32Array): number =>
  values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** A summary of the difference (ratio = max absolute difference ÷ the reference's max absolute value — the metric of decision 8). */
const compare = (got: Float32Array, want: Float32Array, tolerance: Tolerance) => {
  assertEquals(got.length, want.length, "element count");
  const report = allclose(got, want, tolerance);
  const referenceMax = absMax(want);
  return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
};

/** Fills the resident frames with NaN (a frame that was not copied stays NaN — a poison value). */
const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

const nonFiniteCount = (values: Float32Array): number =>
  values.reduce((count, value) => (Number.isFinite(value) ? count : count + 1), 0);

// ---- GPU setup shared by the default cases and the opt-in cases ------------------------------

/** The chunk layout, the two product Sessions and the resident caches one GPU case runs on. */
type VaeRun = {
  readonly gpu: GpuContext;
  readonly layout: WanVaeChunkLayout;
  readonly sessions: { readonly first: Session; readonly next: Session };
  readonly caches: WanVaeChunkCaches;
};

/**
 * Registers, in the enclosing describe, a beforeAll that opens the two chunk graphs and builds their
 * Sessions and the resident caches on one device, and an afterAll that tears them down, so every GPU
 * suite of this file shares one lifecycle and one teardown order. Returns the accessor the its read
 * the run through.
 */
const registerVaeSessions = (): (() => VaeRun) => {
  let gpu: GpuContext | undefined;
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let run: VaeRun | undefined;
  beforeAll(async () => {
    const [firstOpened, nextOpened] = await Promise.all([
      openWanComponent(FIRST_COMPONENT, "f16"),
      openWanComponent(NEXT_COMPONENT, "f16"),
    ]);
    const firstModel = prepareContainer(firstOpened, FIRST_COMPONENT);
    const nextModel = prepareContainer(nextOpened, NEXT_COMPONENT);
    const layout = wanVaeChunkLayout(firstModel, nextModel);

    const device: GpuContext = verifyRunningGpu(await acquireTestGpu());
    gpu = device;
    first = await firstModel.createContainerSession(device);
    next = await nextModel.createContainerSession(device);
    caches = await WanVaeChunkCaches.create(device, layout);
    run = { gpu: device, layout, sessions: { first, next }, caches };
  });
  afterAll(async () => {
    const device = gpu;
    // MUST: tear down in the order Session → resident tensors → device (the resident tensors may be referenced by what
    // the Session baked in, so the Session is torn down before they are released). Even when one step fails, the
    // remaining steps (up to the device) always run (the same shape as the VAE stage of the product `Pipeline`). Vitest reports an afterAll failure apart from the its, so a
    // teardown failure cannot overwrite a test failure.
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
    if (run === undefined) throw new Error("VAE session setup (beforeAll) did not complete");
    return run;
  };
};

const results = openResults("wan-vae-tiles");

/**
 * Plans from a fixture with latent `[C,F,H,W]` (the tile side and scale from the assets' declaration, the overlap by
 * the same derivation as production — it goes through `planWanGenerationTiles`, so the comparison with the fixtures'
 * metadata pins the production plan).
 */
const planFor = (layout: WanVaeChunkLayout, latentShape: readonly number[]): WanVaeTilePlan => {
  assertEquals(latentShape.length, 4, `latent shape [${latentShape}]`);
  return planWanGenerationTiles(layout, latentShape[2], latentShape[3], WAN21_GENERATION);
};

/** The band / accept tile fixtures one tile suite reads (`vae_tiles.*`). */
type TileFixtures = { readonly band: Fixture; readonly accept: Fixture };

/** Registers a beforeAll that reads the band / accept tile fixtures; returns their accessor. */
const registerTileFixtures = (): (() => TileFixtures) => {
  let fixtures: TileFixtures | undefined;
  beforeAll(async () => {
    fixtures = {
      band: await readFixture(fixtureUrl("band")),
      accept: await readFixture(fixtureUrl("accept")),
    };
  });
  return () => {
    if (fixtures === undefined) throw new Error("tile fixture setup (beforeAll) did not complete");
    return fixtures;
  };
};

// Default run: the plan against the Python plan, the degenerate Uint32 gate, the NaN-poisoned
// multi-tile decodes and the seam fault injections. The band comparisons of the band / accept
// frames with the recipe's tiled reference (and their results.json ids) are the opt-in suite below.
describe.skipIf(!TILES_RUNNABLE || !GPU_AVAILABLE)(
  "Wan VAE tiled decode: plan, degenerate check, poison values, assembly fault injection (real GPU)",
  () => {
    const tileFixtures = registerTileFixtures();
    let degenerate: ReturnType<Fixture["tensor"]> | undefined;
    beforeAll(async () => {
      degenerate = (await readFixture(degenerateUrl)).tensor("latents");
    });
    const vaeRun = registerVaeSessions();

    it("plan: the TS plan (from the assets' input shapes) matches the fixtures' metadata (Python)", () => {
      const fixtures = tileFixtures();
      const { layout } = vaeRun();
      for (const name of CASES) {
        const fixture = fixtures[name];
        const ours = planMeta(planFor(layout, fixture.tensor("latents").shape));
        const theirs = Object.fromEntries(Object.keys(ours).map((key) => [key, fixture.meta(key)]));
        assertEquals(ours, theirs, `${name} geometry`);
      }
      const band = planFor(layout, fixtures.band.tensor("latents").shape);
      assertEquals(wanVaeTileCount(band), 12);
      assertEquals([...band.rows.starts], [0, 14, 28]);
      assertEquals([...band.cols.starts], [0, 24, 48, 72]);
    });

    it("degenerate check: for a 32×32 latent (one tile) the tiled path ≡ the untiled chunk sequence (Uint32)", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      assert(degenerate !== undefined, "the degenerate check's latent was not read");
      const plan = planFor(layout, degenerate.shape);
      assertEquals(wanVaeTileCount(plan), 1);
      const chunks = wanVaeChunkCount(layout, degenerate.data);
      await poisonFrames(caches, chunks);
      const tiled = await decodeWanVaeTiled(gpu, sessions, caches, plan, degenerate.data);
      const direct = await decodeWanVaeTile(gpu, sessions, caches, degenerate.data);
      assertEquals(tiled.length, 3 * wanVaeFrameCount(chunks) * 256 * 256);
      assert(tiled.every(Number.isFinite), "non-finite (frames that were not copied)");
      assert(bitsEqual(tiled, direct), "the tiled path is not bit-identical to the chunk sequence");
    });

    it("band: the tiled decode of 832×480, 33 frames writes every frame (the frames start as poison values)", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      const fixture = tileFixtures().band;
      const latents = fixture.tensor("latents");
      const want = fixture.tensor("frames");
      const plan = planFor(layout, latents.shape);
      const chunks = latents.shape[1];
      assertEquals(latents.shape, [layout.latentChannels, chunks, 60, 104]);
      assertEquals(want.shape, [3, wanVaeFrameCount(chunks), 480, 832]);
      await poisonFrames(caches, chunks);
      const got = await decodeWanVaeTiled(gpu, sessions, caches, plan, latents.data);
      assertEquals(got.length, want.data.length, "element count");
      assertEquals(nonFiniteCount(got), 0, "band: non-finite (frames that were not copied)");
    });

    // Acceptance: keep the decoded tiles and reuse them for the fault injections (host-side assembly).
    let acceptTiles: Float32Array<ArrayBuffer>[] = [];

    it("accept: the frames assembled from the tiles of 480×832, 9 frames (different latent, chunk boundaries, portrait) are finite", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      const accept = tileFixtures().accept;
      const acceptLatents = accept.tensor("latents");
      const acceptWant = accept.tensor("frames").data;
      const acceptPlan = planFor(layout, acceptLatents.shape);
      acceptTiles = await decodeWanVaeTiles(gpu, sessions, caches, acceptPlan, acceptLatents.data);
      const got = assembleWanVaeTiles(acceptTiles, acceptPlan);
      assertEquals(got.length, acceptWant.length, "element count");
      assertEquals(nonFiniteCount(got), 0, "accept: non-finite");
    });

    const faults: readonly {
      readonly label: string;
      readonly plan: (acceptPlan: WanVaeTilePlan) => WanVaeTilePlan;
    }[] = [
      {
        // Shows in the values only for portrait frames (an involution for square ones). The tile counts agree as
        // 4×3 = 3×4, so it slips past the shape checks.
        label: "rows and columns swapped",
        plan: (acceptPlan) => ({ ...acceptPlan, rows: acceptPlan.cols, cols: acceptPlan.rows }),
      },
      {
        // What a rounding disagreement between TS and Python looks like (the decoded position and the pasted position
        // differ by one latent).
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
    for (const { label, plan } of faults) {
      it(`fault injection: ${label} → accept is outside the band`, () => {
        assert(acceptTiles.length > 0, "the accept tiles were not decoded");
        const { layout } = vaeRun();
        const accept = tileFixtures().accept;
        const acceptWant = accept.tensor("frames").data;
        const acceptPlan = planFor(layout, accept.tensor("latents").shape);
        const got = assembleWanVaeTiles(acceptTiles, plan(acceptPlan));
        const { report, ratio } = compare(got, acceptWant, TOLERANCE);
        console.log(`[wan-vae-tiles] fault ${label}: maxAbs=${report.maxAbsError} ratio=${ratio}`);
        assert(!report.pass, `${label}: stayed inside the band (a sign the band is too wide)`);
        assert(report.maxAbsError > 1e3 * BAND, `${label}: maxAbs ${report.maxAbsError}`);
      });
    }
  },
);

// Upstream parity: the band / accept frames against the recipe's tiled reference (the upstream
// blend, CPU f32) under the absolute band, so it is opt-in. It owns the band / accept results.json
// ids and their measurements.
describe.skipIf(!TILES_RUNNABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  `Wan VAE tiled decode: 832×480, 33 frames match the upstream reference within the band (real GPU)${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const tileFixtures = registerTileFixtures();
    const vaeRun = registerVaeSessions();

    it("band: 832×480, 33 frames match the reference within the band (the frames start as poison values)", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      await runRecordedCase(results, { id: "band" }, async ({ measurements }) => {
        const fixture = tileFixtures().band;
        const latents = fixture.tensor("latents");
        const want = fixture.tensor("frames");
        const plan = planFor(layout, latents.shape);
        const chunks = latents.shape[1];
        assertEquals(latents.shape, [layout.latentChannels, chunks, 60, 104]);
        assertEquals(want.shape, [3, wanVaeFrameCount(chunks), 480, 832]);
        await poisonFrames(caches, chunks);
        const got = await decodeWanVaeTiled(gpu, sessions, caches, plan, latents.data);
        const { report, referenceMax, ratio } = compare(got, want.data, TOLERANCE);
        measurements.push({
          output: `frames(${chunks} chunks × ${wanVaeTileCount(plan)} tiles)`,
          maxAbs: report.maxAbsError,
          maxRel: report.maxRelError,
          tolerance: TOLERANCE,
          stage: "karume",
        });
        console.log(
          `[wan-vae-tiles] band: tiles=${wanVaeTileCount(plan)} chunks=${chunks} ` +
            `maxAbs=${report.maxAbsError} refMax=${referenceMax} ratio=${ratio} ` +
            `nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, "band: non-finite (frames that were not copied)");
        assert(report.pass, `band: maxAbs ${report.maxAbsError} is outside the band ${BAND}`);
        return undefined;
      });
    });

    it("accept: 480×832, 9 frames (different latent, chunk boundaries, portrait) are inside the band", async () => {
      const { gpu, layout, sessions, caches } = vaeRun();
      await runRecordedCase(results, { id: "accept" }, async ({ measurements }) => {
        const accept = tileFixtures().accept;
        const acceptLatents = accept.tensor("latents");
        const acceptWant = accept.tensor("frames").data;
        const acceptPlan = planFor(layout, acceptLatents.shape);
        const acceptTiles = await decodeWanVaeTiles(
          gpu,
          sessions,
          caches,
          acceptPlan,
          acceptLatents.data,
        );
        const got = assembleWanVaeTiles(acceptTiles, acceptPlan);
        const { report, referenceMax, ratio } = compare(got, acceptWant, TOLERANCE);
        measurements.push({
          output: `frames(${acceptLatents.shape[1]} chunks × ${wanVaeTileCount(acceptPlan)} tiles)`,
          maxAbs: report.maxAbsError,
          maxRel: report.maxRelError,
          tolerance: TOLERANCE,
          stage: "karume",
        });
        console.log(
          `[wan-vae-tiles] accept: maxAbs=${report.maxAbsError} refMax=${referenceMax} ` +
            `ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
        );
        assertEquals(report.nonFiniteCount, 0, "accept: non-finite");
        assert(report.pass, `accept: maxAbs ${report.maxAbsError} is outside the band ${BAND}`);
        return undefined;
      });
    });
  },
);

// ---- Chunk sequence (ADR 0118 step 4) --------------------------------------------------------

/*
 * The **reference comparison of the chunk sequence** of Wan2.1's video VAE (real GPU — the acceptance of ADR 0118 step 4).
 *
 * The mirror's two chunk graphs (`vae_decoder_first` / `vae_decoder_next` — f16) run first → next × (F−1) with
 * resident causal caches (`decodeWanVaeTile` — one tile = one batch, only the frames read back), and the output is
 * compared with upstream's untiled `_decode` chunk loop (diffusers, CPU f32, weights rounded to f16 — ADR 0006)
 * **before the clamp**.
 * The reference and the fixtures are written by `tools/export-recipes/wan/export_vae.py` (read from the test-asset root — fetch with
 * `node tools/test-assets/fetch.ts`).
 *
 * ## How the band is set (decision 8 and addendum 2026-10-02)
 *
 * - The band is set from the measured worst case of a plain comparison with `atol = rtol = 0` on the `band` case
 *   (a latent of 9 chunks = 33 frames, tile 32) ({@link CHUNK_BAND}).
 * - Acceptance is judged on the `accept` case with **a different latent and different chunk boundaries** (5 chunks =
 *   17 frames).
 * - The `long` case (21 chunks = 81 frames — ADR 0118 step 8) is an acceptance case too, checked to be inside the same
 *   band. With the actual chunk count of 81 frames it shows that the error does not accumulate through the carried-over
 *   caches (without relying on extrapolating from the 9 / 5 chunks of band / accept). The band stays the value derived
 *   from `band` (it is not changed after seeing the long result).
 * - The fault injections (a cache not updated, the 2 frames of a cache reversed, the zeroing at the tile head skipped,
 *   first used for chunk 2 onwards) are checked to land outside the band on `accept` (if not, the band is too wide).
 *   The fault injections run in this test's loop, built from the same parts as the product path, and that loop without
 *   a fault is also checked to be Uint32-identical to the product path (ensuring the injected results represent the
 *   product path).
 *
 * An environment with no assets at all SKIPs explicitly; one with only some is a FAIL (the same rule as the dacvae e2e).
 */

/**
 * The band of the chunk sequence (absolute — the output is the pre-clamp frames, in a range of about ±1.3).
 *
 * Measured (B570, Deno 2.9.6, plain comparison with `atol = rtol = 0`; band / accept on 2026-10-02, long on 2026-10-03
 * — first run after the band was set):
 *
 * | Case   | Role       | Chunks | Frames | maxAbs  | Reference max | Ratio (maxAbs ÷ reference max) |
 * | ------ | ---------- | ------ | ------ | ------- | ------------- | ------------------------------ |
 * | band   | band       | 9      | 33     | 4.65e-6 | 1.297         | 3.58e-6                        |
 * | accept | acceptance | 5      | 17     | 3.55e-6 | 1.282         | 2.77e-6                        |
 * | long   | acceptance | 21     | 81     | 4.12e-6 | 1.380         | 2.98e-6                        |
 *
 * The band 2.5e-5 is about 5.4 × the measured worst of `band`. The ratio is 1/28 of the prior guideline (1e-4 —
 * decision 8), the same order as the difference of the eager rewrite (2.79e-6 on the CPU — the recipe's `--verify`) —
 * the GPU's reduction-order differences are buried in it. Stacking 9 or 21 chunks stays at the same order as 5 (the
 * error does not accumulate through the carried-over caches). The fault injections (cache not updated 0.545, 2 frames
 * reversed 1.12, zeroing skipped 0.918) are at least 4 orders of magnitude outside the band.
 */
const CHUNK_BAND = 2.5e-5;
const CHUNK_TOLERANCE: Tolerance = { atol: CHUNK_BAND, rtol: 0 };

const CHUNK_CASES = ["band", "accept", "long"] as const;
type ChunkCaseName = (typeof CHUNK_CASES)[number];

/** The chunk count of `long` (81 frames = 1 + 4·20 — the chunk sequence of the largest accepted frame count). */
const LONG_CHUNKS = 21;

/** The cache the fault injection drops (the head's conv — the least influential one, passing through a single layer). */
const DROPPED_CACHE = "cache_31";

const chunkFixtureUrl = (name: ChunkCaseName): URL =>
  new URL(`vae_chunks.${name}.safetensors`, GOLDEN_ROOT);

const CHUNK_PRESENT = CHUNK_CASES.map((name) => fileExists(chunkFixtureUrl(name)));
const CHUNK_ANY_PRESENT = CHUNK_PRESENT.some(Boolean);
/** The chunk-sequence checks read the mirror's two chunk graphs and these three fixtures. */
const CHUNK_RUNNABLE = MIRROR_PRESENT && CHUNK_ANY_PRESENT;
if (!CHUNK_ANY_PRESENT) {
  console.warn(
    `[karume] no VAE chunk fixture under ${GOLDEN_ROOT.pathname}, so Wan's VAE chunk-sequence checks are ` +
      `SKIPPED (${MISSING_FIXTURES})`,
  );
}

// An environment with none "has not generated them", so it SKIPs. With any one present a missing one is a FAIL (so is any
// missing one in a complete fetch of the pinned dataset; the mirror's completeness is the distribution gate's).
test.skipIf(!CHUNK_ANY_PRESENT && !FETCHED)(
  "Wan VAE chunk fixtures: the three fixtures are present",
  () => {
    assertEquals(
      CHUNK_PRESENT,
      CHUNK_PRESENT.map(() => true),
      `missing: ${CHUNK_CASES.filter((_, index) => !CHUNK_PRESENT[index])
        .map((name) => chunkFixtureUrl(name).pathname)
        .join(", ")} (${MISSING_FIXTURES})`,
    );
  },
);

const chunkResults = openResults("wan-vae-chunks");

const readChunkFixtures = async () => ({
  band: await readChunkFixture(chunkFixtureUrl("band")),
  accept: await readChunkFixture(chunkFixtureUrl("accept")),
  long: await readChunkFixture(chunkFixtureUrl("long")),
});

type ChunkFixtures = Awaited<ReturnType<typeof readChunkFixtures>>;

/** Registers a beforeAll that reads the band / accept / long chunk fixtures; returns their accessor. */
const registerChunkFixtures = (): (() => ChunkFixtures) => {
  let fixtures: ChunkFixtures | undefined;
  beforeAll(async () => {
    fixtures = await readChunkFixtures();
  });
  return () => {
    if (fixtures === undefined) throw new Error("chunk fixture setup (beforeAll) did not complete");
    return fixtures;
  };
};

// Default run: the layout declared by the assets, the cache-zeroing size, the NaN-poisoned chunk
// sequences, the Uint32 identity of the fault-injection loop with the product path and the rejected
// first-for-all enqueue. The band comparisons with the untiled upstream `_decode`, the fault
// injections that prove that band is live and their results.json ids are the opt-in suite below.
describe.skipIf(!CHUNK_RUNNABLE || !GPU_AVAILABLE)(
  "Wan VAE chunk sequence: layout, poison values, Uint32 identity of the fault-injection loop, rejection of reusing first (real GPU)",
  () => {
    const chunkFixtures = registerChunkFixtures();
    const vaeRun = registerVaeSessions();

    it("layout: the tile side and the caches come from the assets' declaration (first 30, next 32)", () => {
      const { layout } = vaeRun();
      assertEquals(layout.tile, 32);
      assertEquals(layout.sampleTile, 256);
      assertEquals(layout.first.caches.length, 30);
      assertEquals(layout.next.caches.length, 32);
      assertEquals(layout.first.frameShape, [3, 1, 256, 256]);
      assertEquals(layout.next.frameShape, [3, 4, 256, 256]);
      assertEquals(
        layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
        ["cache_11", "cache_18"],
      );
    });

    it("observation: bytes and host time of zeroing the caches at the tile head (GPU-side clear)", () => {
      const { caches } = vaeRun();
      // Zeroing is `ResidentTensor.clear` (one clearBuffer submit per cache — no host memcpy or
      // staging). A record, not a gate — the tiled decode pays it once per tile.
      const started = performance.now();
      caches.zero();
      const hostMs = performance.now() - started;
      console.log(`[wan-vae] zero: bytes=${caches.cacheBytes} hostMs=${hostMs.toFixed(1)}`);
      assertEquals(caches.cacheBytes, 154_959_872 * 4);
    });

    for (const name of CHUNK_CASES) {
      it(`${name}: the chunk sequence writes every frame (the frames start as poison values)`, async () => {
        const { gpu, layout, sessions, caches } = vaeRun();
        const fixture = chunkFixtures()[name];
        const chunks = wanVaeChunkCount(layout, fixture.latents);
        assertEquals(fixture.frameShape, [3, wanVaeFrameCount(chunks), 256, 256]);
        if (name === "long") assertEquals(chunks, LONG_CHUNKS, "chunk count of long");
        // Poison values: fill the resident frames with NaN before the run (a frame that was not copied stays NaN).
        await poisonFrames(caches, chunks);
        const got = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
        assertEquals(got.length, fixture.frames.length, `${name}: element count`);
        assertEquals(nonFiniteCount(got), 0, `${name}: non-finite (frames that were not copied)`);
      });
    }

    it("the fault-injection loop without a fault is Uint32-identical to the product path", async () => {
      const { gpu, sessions, caches } = vaeRun();
      const fixture = chunkFixtures().accept;
      const product = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
      const looped = await decodeWithFault(gpu, sessions, caches, fixture.latents, {
        kind: "none",
      });
      assert(bitsEqual(product, looped), "the test's loop differs from the product path");
    });

    it("fault injection: using first for chunk 2 onwards → the enqueue is rejected by the size of copyOutputs", async () => {
      const { gpu, sessions, caches } = vaeRun();
      const fixtures = chunkFixtures();
      // first's output frames are [3,1,…] and cannot be copied into the resident frames for next ([3,4,…]) — the
      // runtime's size check of `copyOutputs` fails before any dispatch is queued (red before any value is compared).
      await assertRejects(
        () =>
          decodeWithFault(gpu, sessions, caches, fixtures.accept.latents, {
            kind: "first-for-all",
          }),
        ExecutionError,
        "copyOutputs",
      );
    });
  },
);

// Upstream parity: the chunk sequences against the untiled upstream `_decode` under the absolute
// band, with the cache fault injections that only prove that band is live, so it is opt-in. It owns
// the band / accept / long results.json ids and their measurements.
describe.skipIf(!CHUNK_RUNNABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  `Wan VAE chunk sequence: first → next with resident caches matches the upstream _decode within the band (real GPU)${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const chunkFixtures = registerChunkFixtures();
    const vaeRun = registerVaeSessions();

    const compareChunks = (name: ChunkCaseName, got: Float32Array) => {
      const fixture = chunkFixtures()[name];
      assertEquals(got.length, fixture.frames.length, `${name}: element count`);
      const report = allclose(got, fixture.frames, CHUNK_TOLERANCE);
      const referenceMax = fixture.frames.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      return { report, referenceMax, ratio: report.maxAbsError / referenceMax };
    };

    for (const name of CHUNK_CASES) {
      it(`${name}: the chunk sequence matches upstream within the band (the frames start as poison values)`, async () => {
        const { gpu, layout, sessions, caches } = vaeRun();
        await runRecordedCase(chunkResults, { id: name }, async ({ measurements }) => {
          const fixture = chunkFixtures()[name];
          const chunks = wanVaeChunkCount(layout, fixture.latents);
          assertEquals(fixture.frameShape, [3, wanVaeFrameCount(chunks), 256, 256]);
          if (name === "long") assertEquals(chunks, LONG_CHUNKS, "chunk count of long");
          // Poison values: fill the resident frames with NaN before the run (a frame that was not copied stays NaN).
          await poisonFrames(caches, chunks);
          const got = await decodeWanVaeTile(gpu, sessions, caches, fixture.latents);
          const { report, referenceMax, ratio } = compareChunks(name, got);
          measurements.push({
            output: `frames(${chunks} chunks)`,
            maxAbs: report.maxAbsError,
            maxRel: report.maxRelError,
            tolerance: CHUNK_TOLERANCE,
            stage: "karume",
          });
          console.log(
            `[wan-vae] ${name}: chunks=${chunks} maxAbs=${report.maxAbsError} ` +
              `refMax=${referenceMax} ratio=${ratio} nonFinite=${report.nonFiniteCount}`,
          );
          assertEquals(
            report.nonFiniteCount,
            0,
            `${name}: non-finite (frames that were not copied)`,
          );
          assert(
            report.pass,
            `${name}: maxAbs ${report.maxAbsError} is outside the band ${CHUNK_BAND}`,
          );
          return undefined;
        });
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
        const { gpu, sessions, caches } = vaeRun();
        const fixtures = chunkFixtures();
        if (fault.kind === "skip-zero") {
          // Run accept with the previous tile's (band's) caches left in place.
          await decodeWanVaeTile(gpu, sessions, caches, fixtures.band.latents);
        }
        const got = await decodeWithFault(gpu, sessions, caches, fixtures.accept.latents, fault);
        const { report, ratio } = compareChunks("accept", got);
        console.log(`[wan-vae] fault ${fault.kind}: maxAbs=${report.maxAbsError} ratio=${ratio}`);
        assert(!report.pass, `${label}: stayed inside the band (a sign the band is too wide)`);
        assert(report.maxAbsError > 1e3 * CHUNK_BAND, `${label}: maxAbs ${report.maxAbsError}`);
      });
    }
  },
);
