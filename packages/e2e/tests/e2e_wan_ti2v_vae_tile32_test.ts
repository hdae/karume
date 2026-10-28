/**
 * The Wan2.2 TI2V-5B VAE decoder at a **32-latent tile** (ADR 0134; real GPU): the second chunk-graph
 * pair (`vae_decoder_first_32` / `vae_decoder_next_32` — weightless containers written by
 * `wan.export_vae --tile 32 --shared`) borrowing every weight of the 16-latent pair of the same series.
 *
 * ## Gates (ADR 0134 decision 7)
 *
 * Three runs, each on its own device: the f16 series with `{}`, the int8 series with `{}` and the int8
 * series with `{ convCompute: "a8" }` (the conv8 preset's decoder — ADR 0133).
 *
 * 1. **Shared weights**: the 16-latent Sessions are built with the run's options as lenders; the
 *    32-latent Sessions are built with the same options plus `sharedWeights` over every initializer the
 *    production admission check names (`wanVaeTile32Contract`). The borrowers building at all is the
 *    runtime's six shared-weight checks passing (device, shape, dtype, residency class, row axis, the
 *    exact set). A borrower built without `sharedWeights` fails loudly.
 * 2. **Degenerate gate**: a 32×32 latent (exactly one tile) through the tiled path
 *    (`decodeWanVaeTiled` with the 32-latent layout) ≡ the untiled chunk sequence on the borrowers
 *    (`decodeWanVaeTile`) — Uint32 identical, both decoded from NaN-poisoned frames, 0 non-finite values.
 *    The latent is the f16 series' `vae_chunks.accept` latent (`[48, 5, 16, 16]` — decoder-range values)
 *    repeated 2×2 into `[48, 5, 32, 32]`: an identity gate needs no tile-32 reference.
 * 3. **conv8 census** (the `a8` run): every conv3d dispatch of both borrowers runs the w8a8 key (33 / 35
 *    nodes, the counts at tile 16) with its `quantize_rows` prelude, and no f32 conv3d key remains; the
 *    other runs run no w8a8 key.
 *
 * The tile-32 references (`vae_tiles.<case>.t32.safetensors` in the f16 series root —
 * `vae_tiling.py --model ti2v-5b --tile 32`; only `wide` fits a 32-latent tile) add, on the f16 run:
 *
 * 4. **Plan**: the TS plan (from the 32-latent layout) equals the fixture's meta (the Python plan).
 * 5. **No non-finite values**: `wide` decoded through the borrowers from NaN-poisoned frames.
 * 6. **Band** (opt-in upstream parity — `KARUME_UPSTREAM_PARITY=1`): `wide` against the Python tile
 *    reference under {@link BAND_32}. `undefined` = not derived: the step records maxAbs and fails with
 *    the candidate (maxAbs × 5, rounded up to two significant digits — the rule of the 16-latent band).
 *    With `wide` the only case that fits the tile, it is both the deciding and the accepted case.
 *
 * No tile-32 pair in either series → explicit SKIP with the generation commands; a pair in only some of
 * the places → FAIL. No tile-32 reference → the reference steps SKIP with the command; some → FAIL.
 */

import { readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  prepareContainer,
  type Session,
  type SessionOptions,
  type SharedWeight,
} from "@karume/runtime";
import {
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeChunkLayout,
  wanVaeFrameCount,
} from "../../models/src/wan/vae-chunks.ts";
import {
  decodeWanVaeTiled,
  wanVaeBlendExtentAt,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "../../models/src/wan/vae-tiles.ts";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import {
  planWanGenerationTiles,
  type WanVaeTile32Contract,
  wanVaeTile32Contract,
} from "../../models/src/wan/tile-decode.ts";
import { disposeSteps } from "../../models/src/session/dispose-steps.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { bitsEqual, readFixture } from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import {
  ti2vVaeAssets,
  ti2vVaeFixtureUrl,
  ti2vVaeI8Assets,
  ti2vVaeTile32Assets,
  ti2vVaeTile32FixtureUrl,
  ti2vVaeTile32TileAssets,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_FIRST_32 as FIRST_32,
  WAN_TI2V_VAE_GENERATE as GENERATE_CHUNKS,
  WAN_TI2V_VAE_I8_GENERATE as GENERATE_I8,
  WAN_TI2V_VAE_I8_ROOT as I8_ROOT,
  WAN_TI2V_VAE_I8_SERIES as I8_SERIES,
  WAN_TI2V_VAE_I8_TILE32_GENERATE as GENERATE_I8_TILE32,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_MODEL_FILE as MODEL_FILE,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_NEXT_32 as NEXT_32,
  WAN_TI2V_VAE_ROOT as ROOT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE_32 as SAMPLE_TILE_32,
  WAN_TI2V_VAE_SERIES as SERIES,
  WAN_TI2V_VAE_TILE_32 as TILE_32,
  WAN_TI2V_VAE_TILE32_CASES as T32_CASES,
  WAN_TI2V_VAE_TILE32_GENERATE as GENERATE_TILE32,
  WAN_TI2V_VAE_TILES_T32_GENERATE as GENERATE_T32_TILES,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";
import { allclose } from "../../runtime/src/reference/allclose.ts";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import { conv3dCensus } from "./helpers/conv3d-census.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The band of the tile decode at tile 32 against the Python tile reference (absolute — pre-clamp
 * patchify-space frames). `undefined` = not derived (module doc, gate 6).
 *
 * Derived 2026-10-23 (RTX 3080 Ti, Node + Dawn, f16 series): `wide` (1280×704×5, 2×3 tiles) maxAbs
 * 1.001e-5 × 5 = 5.007e-5 → rounded up to two significant digits 5.1e-5 (the 16-latent band is 5.0e-5;
 * `wide` measured 6.348e-6 there). `wide` is the only case that fits a 32-latent tile, so it both
 * derives and meets the band.
 *
 * MUST: do not widen it to pass a later run — investigate the cause instead.
 */
const BAND_32: number | undefined = 5.1e-5;

/** The conv3d node counts of the two chunk graphs (the same graphs at either tile). */
const CONV3D_NODES = { first: 33, next: 35 } as const;

/** The message prefix of the production checks (the 2.2 pipeline's name). */
const OWNER = "wan.Ti2vPipeline";

/** The tile reference's meta `weights` / `reference` (the writer's spelling — `vae_tiling.write_fixture`). */
const FIXTURE_WEIGHTS = "f16-rounded";
const FIXTURE_REFERENCE =
  "snapped-tile decode with upstream blend_v / blend_h in patchify space before unpatchify and clamp (CPU f32)";

/** The chunk fixture whose latent the degenerate gate repeats 2×2 (`[48, 5, 16, 16]`). */
const DEGENERATE_SOURCE = "accept";

type Run = {
  readonly label: "f16" | "i8" | "i8-a8";
  readonly root: URL;
  readonly series: string;
  readonly options: SessionOptions;
};

const RUNS: readonly Run[] = [
  { label: "f16", root: ROOT, series: SERIES, options: {} },
  { label: "i8", root: I8_ROOT, series: I8_SERIES, options: {} },
  { label: "i8-a8", root: I8_ROOT, series: I8_SERIES, options: { convCompute: "a8" } },
];

const TILE32_ASSETS = [...ti2vVaeTile32Assets(ROOT), ...ti2vVaeTile32Assets(I8_ROOT)];
const ANY_TILE32 = TILE32_ASSETS.some(({ present }) => present);
const ALL_TILE32 = TILE32_ASSETS.every(({ present }) => present);
/** The lenders and the chunk fixtures (the f16 series' set) and the int8 lenders. */
const F16_LENDERS = ti2vVaeAssets().every(({ present }) => present);
const I8_LENDERS = ti2vVaeI8Assets().every(({ present }) => present);
const T32_ASSETS = ti2vVaeTile32TileAssets();
const ANY_T32 = T32_ASSETS.some(({ present }) => present);
const ALL_T32 = T32_ASSETS.every(({ present }) => present);

if (!ANY_TILE32) {
  console.warn(
    `[karume] the Wan2.2 TI2V tile-32 VAE decoder pair is in neither ${ROOT.pathname} nor ${I8_ROOT.pathname} — ` +
      `the tile-32 gates SKIP. Generate: ${GENERATE_TILE32} / ${GENERATE_I8_TILE32}`,
  );
}
if (!ANY_T32) {
  console.warn(
    `[karume] no tile-32 VAE tile reference (vae_tiles.*.t32) in ${ROOT.pathname} — the plan, wide and band ` +
      `steps at tile 32 SKIP. Generate: ${GENERATE_T32_TILES}`,
  );
}

// None → "not generated" (SKIP); some → the missing ones FAIL (the writer writes both series).
test.skipIf(!ANY_TILE32)(
  "Wan2.2 TI2V VAE tile 32: the 32-latent pair is in both series (f16 and i8)",
  () => {
    assertEquals(
      TILE32_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `the tile-32 pair is missing (generate: ${GENERATE_TILE32} / ${GENERATE_I8_TILE32})`,
    );
  },
);

// The pair borrows the lenders' weights, so a pair without its lenders cannot run — FAIL, not SKIP.
test.skipIf(!ANY_TILE32)(
  "Wan2.2 TI2V VAE tile 32: the lenders (the 16-latent pair of both series) and the chunk fixtures are present",
  () => {
    assert(
      F16_LENDERS,
      `the f16 series' 16-latent pair or chunk fixtures are missing (${GENERATE_CHUNKS})`,
    );
    assert(I8_LENDERS, `the int8 series' 16-latent pair is missing (${GENERATE_I8})`);
  },
);

test.skipIf(!ANY_T32)("Wan2.2 TI2V VAE tile 32: the tile-32 references are all present", () => {
  assertEquals(
    T32_ASSETS.filter(({ present }) => !present).map(({ path }) => path),
    [],
    `tile-32 references missing (generate: ${GENERATE_T32_TILES})`,
  );
});

/** One tile reference (F32 tensors and meta — the writer is `vae_tiling.write_fixture`). */
const readTileFixture = async (url: URL) => {
  const bytes = await readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    tensor: (key: string) => {
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
    meta: (key: string): string => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: meta '${key}' is missing`);
      return value;
    },
  };
};

/** The plan's geometry in the fixture meta's spelling (`vae_tiling.TilePlan.meta` + `patch_size`). */
const planMeta = (plan: WanVaeTilePlan): Record<string, string> => {
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

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/** Fills the frame residents with NaN (a frame that is never written stays NaN — the poison). */
const poisonFrames = async (caches: WanVaeChunkCaches, chunks: number): Promise<void> => {
  for (const resident of await caches.frames(chunks)) {
    resident.write(new Float32Array(resident.byteLength / 4).fill(Number.NaN));
  }
};

/** A latent `[C, F, h, w]` repeated 2×2 along both spatial axes (`[C, F, 2h, 2w]`). */
const repeat2x2 = (latents: Float32Array, shape: readonly number[]): Float32Array<ArrayBuffer> => {
  const [channels, frames, rows, cols] = shape;
  const out = new Float32Array(channels * frames * rows * 2 * cols * 2);
  for (let sheet = 0; sheet < channels * frames; sheet += 1) {
    for (let y = 0; y < rows * 2; y += 1) {
      for (let x = 0; x < cols * 2; x += 1) {
        out[(sheet * rows * 2 + y) * cols * 2 + x] =
          latents[(sheet * rows + (y % rows)) * cols + (x % cols)];
      }
    }
  }
  return out;
};

/** Rounding up to two significant digits (the rule of the 16-latent band candidates). */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(value / unit) * unit).toPrecision(2));
};

/** Opens one series' component and prepares its graph (graph names from the series table). */
const prepareComponent = async (run: Run, component: string) =>
  prepareContainer(
    await openSeriesContainer(new URL(`${component}/${MODEL_FILE}`, run.root)),
    seriesGraph(run.series, component),
  );

/** The lender's weights the borrower takes, by name (`Session.exportWeight`). */
const lend = (lender: Session, names: readonly string[]): Record<string, SharedWeight> =>
  Object.fromEntries(names.map((name) => [name, lender.exportWeight(name)]));

for (const run of RUNS) {
  const lendersPresent = run.label === "f16" ? F16_LENDERS : I8_LENDERS && F16_LENDERS;
  describe.skipIf(!ALL_TILE32 || !lendersPresent || !GPU_AVAILABLE)(
    `Wan2.2 TI2V VAE tile 32 (${run.label}${
      run.options.convCompute === undefined ? "" : ` — convCompute '${run.options.convCompute}'`
    }): the 32-latent pair borrows the 16-latent pair's weights; the degenerate gate (Uint32), 0 non-finite` +
      `${run.options.convCompute === "a8" ? ", the conv8 census" : ""} (real GPU)`,
    () => {
      let gpu: GpuContext | undefined;
      let deviceLost: string | undefined;
      let lenders: { first: Session; next: Session } | undefined;
      let borrowers: { first: Session; next: Session } | undefined;
      let caches: WanVaeChunkCaches | undefined;
      let contract: WanVaeTile32Contract | undefined;
      let models:
        | {
            readonly first32: Awaited<ReturnType<typeof prepareComponent>>;
            readonly next32: Awaited<ReturnType<typeof prepareComponent>>;
          }
        | undefined;
      const opened: Session[] = [];
      const current = () => {
        assert(
          gpu !== undefined &&
            lenders !== undefined &&
            borrowers !== undefined &&
            caches !== undefined &&
            contract !== undefined &&
            models !== undefined,
          "setup (beforeAll) did not complete",
        );
        return { gpu, lenders, borrowers, caches, contract, models };
      };

      beforeAll(async () => {
        const first = await prepareComponent(run, FIRST);
        const next = await prepareComponent(run, NEXT);
        const first32 = await prepareComponent(run, FIRST_32);
        const next32 = await prepareComponent(run, NEXT_32);
        models = { first32, next32 };
        // The production admission check (ADR 0134): layout at tile 32, the generation's channels and
        // coverage, every initializer borrowed from the lender of its role.
        contract = wanVaeTile32Contract(
          { first: first32, next: next32 },
          { first, next },
          wanVaeChunkLayout(first, next),
          WAN22_TI2V_GENERATION,
          OWNER,
        );
        const device = verifyRunningGpu(
          await acquireTestGpu({
            onDeviceLost: (info) => {
              deviceLost = `${info.reason}: ${info.message}`;
            },
          }),
        );
        gpu = device;
        const lenderFirst = await first.createContainerSession(device, run.options);
        opened.push(lenderFirst);
        const lenderNext = await next.createContainerSession(device, run.options);
        opened.push(lenderNext);
        lenders = { first: lenderFirst, next: lenderNext };
        const borrowerFirst = await first32.createContainerSession(device, {
          ...run.options,
          sharedWeights: lend(lenderFirst, contract.borrowed.first),
        });
        opened.push(borrowerFirst);
        const borrowerNext = await next32.createContainerSession(device, {
          ...run.options,
          sharedWeights: lend(lenderNext, contract.borrowed.next),
        });
        opened.push(borrowerNext);
        borrowers = { first: borrowerFirst, next: borrowerNext };
        caches = await WanVaeChunkCaches.create(device, contract.layout);
      });
      afterAll(async () => {
        const device = gpu;
        // MUST: borrowers → lenders → residents → device (the runtime refuses to dispose a lender while
        // a borrower is alive; every step runs even when one fails).
        await disposeSteps([
          ...[...opened].reverse().map((session) => () => session.dispose()),
          () => caches?.dispose(),
          () => (device === undefined ? undefined : settleReleases(device)),
          () => device?.destroy(),
        ]);
      });

      it("shared weights: the borrowers built (the runtime's checks passed) over every initializer, the pair's layout is tile 32", () => {
        const { contract: settled, lenders: lent, borrowers: borrowing } = current();
        assertEquals(
          [settled.layout.latentChannels, settled.layout.tile, settled.layout.sampleTile],
          [LATENT_CHANNELS, TILE_32, SAMPLE_TILE_32],
        );
        assertEquals(settled.layout.sampleChannels, SAMPLE_CHANNELS);
        // The containers are weightless: the contract requires every initializer shared, so an empty
        // set here would mean the pair declares no weight at all.
        assert(
          settled.borrowed.first.length > 0 && settled.borrowed.next.length > 0,
          "nothing borrowed",
        );
        const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(3)} GiB`;
        console.log(
          `[wan-ti2v-vae-tile32] ${run.label}: borrowed ${settled.borrowed.first.length} / ${settled.borrowed.next.length}` +
            ` initializers; weights lender first ${gib(lent.first.diagnostics().weights.allocatedBytes)}` +
            ` next ${gib(lent.next.diagnostics().weights.allocatedBytes)} / borrower first` +
            ` ${gib(borrowing.first.diagnostics().weights.allocatedBytes)} next` +
            ` ${gib(borrowing.next.diagnostics().weights.allocatedBytes)}`,
        );
      });

      it("a borrower built without sharedWeights fails loudly (every shared declaration needs a lent weight)", async () => {
        const { gpu: device, models: pair } = current();
        await assertRejects(
          () => pair.first32.createContainerSession(device, run.options),
          Error,
          "options.sharedWeights does not match the graph's shared declarations: missing [",
        );
      });

      it("degenerate gate: one 32×32 latent tile through the tiled path ≡ the untiled chunk sequence (Uint32), 0 non-finite", async () => {
        const { gpu: device, borrowers: sessions, caches: live, contract: settled } = current();
        const source = await readFixture(ti2vVaeFixtureUrl(DEGENERATE_SOURCE));
        assertEquals(source.latentShape, [LATENT_CHANNELS, 5, 16, 16], "the source latent's shape");
        const latents = repeat2x2(source.latents, source.latentShape);
        const plan = planWanGenerationTiles(
          settled.layout,
          TILE_32,
          TILE_32,
          WAN22_TI2V_GENERATION,
        );
        assertEquals(wanVaeTileCount(plan), 1);
        const chunks = wanVaeChunkCount(settled.layout, latents);
        await poisonFrames(live, chunks);
        const started = performance.now();
        const tiled = await decodeWanVaeTiled(device, sessions, live, plan, latents);
        const tiledMs = performance.now() - started;
        // The poison is refilled before each decode (the second must not read back the first's frames).
        await poisonFrames(live, chunks);
        const direct = await decodeWanVaeTile(device, sessions, live, latents);
        assertEquals(
          tiled.length,
          SAMPLE_CHANNELS * wanVaeFrameCount(chunks) * SAMPLE_TILE_32 * SAMPLE_TILE_32,
        );
        const nonFinite = countNonFinite(tiled);
        console.log(
          `[wan-ti2v-vae-tile32] ${run.label}: degenerate chunks=${chunks} nonFinite=${nonFinite} decodeMs=${tiledMs.toFixed(0)}`,
        );
        assertEquals(nonFinite, 0, "non-finite values (frames that were not written)");
        assert(bitsEqual(tiled, direct), "the tiled path differs from the chunk sequence");
        assertEquals(deviceLost, undefined, "device lost");
      });

      it(
        run.options.convCompute === "a8"
          ? "conv8 census: every conv3d dispatch of both borrowers runs the w8a8 key (none on f32)"
          : "conv census: no conv3d dispatch runs the w8a8 key (the knob is off)",
        () => {
          const { borrowers: sessions } = current();
          for (const [graph, expected] of [
            ["first", CONV3D_NODES.first],
            ["next", CONV3D_NODES.next],
          ] as const) {
            const census = conv3dCensus(`${graph}_32`, sessions[graph].diagnostics());
            console.log(
              `[wan-ti2v-vae-tile32] ${run.label}: ${graph}_32 conv3d w8a8=${census.w8a8} f32=${census.f32}` +
                ` quantize_rows=${census.preludes}`,
            );
            if (run.options.convCompute === "a8") {
              assertEquals(census.w8a8, expected, `${graph}_32: w8a8 conv3d dispatches`);
              assertEquals(census.f32, 0, `${graph}_32: an f32 conv3d remained`);
              assertEquals(census.preludes, expected, `${graph}_32: quantize_rows dispatches`);
            } else {
              assertEquals(census.w8a8, 0, `${graph}_32: a w8a8 conv3d ran with the knob off`);
            }
          }
        },
      );

      if (run.label === "f16") {
        it.skipIf(!ALL_T32)(
          "tile reference: the TS plan at tile 32 equals the fixture meta (Python), and wide decodes from poisoned frames with 0 non-finite",
          async () => {
            const { gpu: device, borrowers: sessions, caches: live, contract: settled } = current();
            for (const { name, chunks, height, width } of T32_CASES) {
              const fixture = await readTileFixture(ti2vVaeTile32FixtureUrl(name));
              const latents = fixture.tensor("latents");
              assertEquals(
                latents.shape,
                [LATENT_CHANNELS, chunks, height, width],
                `${name}: latent shape`,
              );
              assertEquals(
                { weights: fixture.meta("weights"), reference: fixture.meta("reference") },
                { weights: FIXTURE_WEIGHTS, reference: FIXTURE_REFERENCE },
                `${name}: provenance`,
              );
              const plan = planWanGenerationTiles(
                settled.layout,
                height,
                width,
                WAN22_TI2V_GENERATION,
              );
              const ours = planMeta(plan);
              const theirs = Object.fromEntries(
                Object.keys(ours).map((key) => [key, fixture.meta(key)]),
              );
              assertEquals(ours, theirs, `${name}: plan geometry`);
              await poisonFrames(live, chunks);
              const got = await decodeWanVaeTiled(
                device,
                sessions,
                live,
                plan,
                latents.data,
                async (tile) => {
                  if (tile < wanVaeTileCount(plan)) await poisonFrames(live, chunks);
                },
              );
              const want = fixture.tensor("frames").data;
              assertEquals(got.length, want.length, `${name}: element count`);
              const nonFinite = countNonFinite(got);
              console.log(
                `[wan-ti2v-vae-tile32] ${name}.t32: tiles=${wanVaeTileCount(plan)} nonFinite=${nonFinite}`,
              );
              assertEquals(nonFinite, 0, `${name}: non-finite values`);
            }
          },
        );

        it.skipIf(!ALL_T32 || !UPSTREAM_PARITY)(
          `tile reference: wide at tile 32 is inside the band of the Python tile reference${UPSTREAM_PARITY_OPT_IN}`,
          async () => {
            const { gpu: device, borrowers: sessions, caches: live, contract: settled } = current();
            const observed: string[] = [];
            let worst = 0;
            for (const { name, chunks, height, width } of T32_CASES) {
              const fixture = await readTileFixture(ti2vVaeTile32FixtureUrl(name));
              const plan = planWanGenerationTiles(
                settled.layout,
                height,
                width,
                WAN22_TI2V_GENERATION,
              );
              await poisonFrames(live, chunks);
              const got = await decodeWanVaeTiled(
                device,
                sessions,
                live,
                plan,
                fixture.tensor("latents").data,
              );
              const report = allclose(got, fixture.tensor("frames").data, {
                atol: BAND_32 ?? Number.POSITIVE_INFINITY,
                rtol: 0,
              });
              worst = Math.max(worst, report.maxAbsError);
              observed.push(`${name} maxAbs ${report.maxAbsError.toExponential(3)}`);
              console.log(
                `[wan-ti2v-vae-tile32] ${name}.t32: maxAbs=${report.maxAbsError} nonFinite=${report.nonFiniteCount}`,
              );
              assertEquals(report.nonFiniteCount, 0, `${name}: non-finite values`);
              if (BAND_32 !== undefined) {
                assert(
                  report.pass,
                  `${name}: maxAbs ${report.maxAbsError} is outside the band ${BAND_32}`,
                );
              }
            }
            if (BAND_32 === undefined) {
              throw new Error(
                `the tile-32 band is not derived — candidate: ${worst.toExponential(3)} × 5 = ` +
                  `${(worst * 5).toExponential(3)} (rounded up ${roundUpTwoDigits(worst * 5)}); ${observed.join(" / ")}`,
              );
            }
          },
        );
      }
    },
  );
}
