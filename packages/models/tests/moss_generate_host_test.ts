/**
 * The MOSS-Image generate path on the CPU (ADR 0138 decision 9, stage
 * 4): the one acceptance check of a request, and the sampler loop of the generate path driven
 * by upstream's recorded DiT outputs.
 *
 * - **Acceptance** ({@link planMossGenerate}): every rejection is a `ModelInputError` naming the
 *   limit or the missing input, and nothing outside them is refused (any step count, upstream's
 *   whole guidance range, any area under the device's VAE limit). The device is a stand-in with the
 *   RTX 3080 Ti's 2 GiB binding and the decoder geometry of the MOSS VAE decoder (49,152 B per
 *   VAE cell, 8 px per cell — `vae-contract.ts`); the training grid is synthetic (its values are not
 *   the subject here).
 * - **The trajectory** (the Wan check — host CFG + Euler bit-identical from upstream's DiT outputs):
 *   the stage-0 DiT records (`moss.reference` — upstream's own `pipe(...)` at 256² and 256×384,
 *   2 steps, CFG 3 with the skip-22 negative, injected `latents.init`) go through the acceptance
 *   check with the record's own inputs, and the generate path's loop (`mossDenoise`) runs on a
 *   `predict` that returns upstream's recorded samples of each branch. Each forward must receive
 *   upstream's latents and timestep of that step, and the loop's result must be upstream's final
 *   latents — bit for bit. The training grid is the asset of the distribution's `dit` container
 *   (`models/karume-moss-image/` — ADR 0139; the product's source).
 * - **The loop's own rules** (no record needed): guidance 1 runs the positive branch alone through
 *   the plain Euler chain, a `predict` whose branches disagree with the guidance fails loudly, and
 *   the seed path's initial latents are the first `C · N` normals of the seed's sequence.
 *
 * Each group skips explicitly, naming the command that writes what it lacks (ADR 0005).
 */

import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { readMossDitAssets } from "../src/moss/dit-contract.ts";
import { mossTimestep } from "../src/moss/dit-timestep.ts";
import {
  type MossGenerateLimits,
  type MossGenerateRequest,
  mossDenoise,
  mossInitialLatents,
  planMossGenerate,
} from "../src/moss/generate.ts";
import { MossRandn } from "../src/moss/random.ts";
import { mossEulerStep } from "../src/moss/sampler.ts";
import type { MossTextEmbeddings } from "../src/moss/request.ts";
import {
  filePresent,
  firstBitMismatch,
  MOSS_GENERATE_STAGE0,
  MOSS_GET_MIRROR,
  mossMirrorPresent,
  openMossDit,
  readRecordTensors,
  stage0Dir,
  syntheticTrainingGrid,
} from "./helpers/moss-dit.ts";

const skipped = (present: boolean, what: string, command: string): string =>
  present ? "" : ` (SKIP: no ${what} — ${command})`;

const embeddings = (tokens: number): MossTextEmbeddings => ({
  data: new Float32Array(tokens * 4096).fill(0.25),
  shape: [tokens, 4096],
});

/** The decoder geometry of the MOSS VAE decoder and a 2 GiB binding (module doc). */
const LIMITS: MossGenerateLimits = {
  dit: { latentChannels: 128, trainingGrid: syntheticTrainingGrid(1000) },
  vae: { bytesPerCell: 49_152, spatialScale: 8 },
  limits: { maxStorageBufferBindingSize: 2_147_483_644, maxBufferSize: 2_147_483_644 },
};

/** ⌊2,147,483,644 / 49,152⌋ × 64 (`mossVaeAreaLimit`). */
const AREA_LIMIT = 2_796_160;

const BASE: MossGenerateRequest = {
  width: 256,
  height: 256,
  positive: embeddings(5),
  negative: embeddings(10),
  steps: 2,
  guidance: 3,
  shift: 5,
  seed: 42,
};

/**
 * A request as a JavaScript caller can pass it: any field replaced by any value, outside the types
 * too (the acceptance check's job is to refuse such values loudly).
 */
type UntypedOverrides = Partial<Record<keyof MossGenerateRequest, unknown>>;

/**
 * NOTE: the one cast of this file — it stands for a caller outside the types, which is what the
 * rejection cases exercise (a missing `positive`, a `Float64Array` of latents).
 */
const untypedRequest = (overrides: UntypedOverrides): MossGenerateRequest =>
  ({ ...BASE, ...overrides }) as MossGenerateRequest;

const plan = (overrides: UntypedOverrides) => planMossGenerate(untypedRequest(overrides), LIMITS);

describe("MOSS-Image generate acceptance (one check — ADR 0138 decisions 8 and 9)", () => {
  it("plans every stage of an accepted request: cells, the VAE latent, the schedule and the noise", () => {
    const accepted = plan({ width: 384, height: 256 });

    expect(accepted.dit.cells).toEqual({ rows: 16, cols: 24 });
    expect(accepted.vae.latent).toEqual({ height: 32, width: 48 });
    expect(accepted.dit.texts.positive.tokens).toBe(5);
    expect(accepted.dit.texts.negative?.tokens).toBe(10);
    expect(accepted.schedule.gridIndices.length).toBe(2);
    expect(accepted.guidance).toBe(3);
    expect(accepted.initial).toEqual({ kind: "seed", seed: 42 });
  });

  it("rejects each input outside the accepted set with a ModelInputError naming the limit or the missing input", () => {
    const latents = new Float32Array(128 * 256);
    const broken = new Float32Array(128 * 256);
    broken[5] = Number.NaN;
    const cases: readonly [UntypedOverrides, string][] = [
      [{ width: 1000 }, "width 1000 px is not a positive multiple of 16"],
      [{ width: 65_552, height: 16 }, "the longest side 65552 px exceeds the accepted 65536 px"],
      [
        { width: 2048, height: 2048 },
        `2048×2048 px (4194304 px) exceeds this device's area limit of ${AREA_LIMIT} px`,
      ],
      [{ positive: undefined }, "the positive prompt's embeddings are missing"],
      [{ guidance: -0.5 }, "guidance -0.5 is not a finite number ≥ 0 that fits in f32"],
      [{ guidance: Number.NaN }, "guidance NaN is not a finite number ≥ 0"],
      [{ guidance: 1e39 }, "guidance 1e+39 is not a finite number ≥ 0 that fits in f32"],
      [
        { negative: undefined },
        "guidance 3 runs the negative branch (CFG with block 22 skipped), so the negative prompt's" +
          " embeddings are required",
      ],
      [
        { guidance: 1 },
        "guidance 1 runs no negative branch, so the negative prompt's embeddings would have no effect",
      ],
      [
        { positive: { data: new Float32Array(4095), shape: [1, 4095] } },
        "the positive text embeddings have shape [1, 4095]",
      ],
      [
        { negative: embeddings(4097) },
        "the negative text embeddings have 4097 tokens, over the accepted 4096",
      ],
      [{ steps: 0 }, "steps 0 is not a positive integer"],
      [{ steps: 2.5 }, "steps 2.5 is not a positive integer"],
      [{ shift: 0 }, "shift 0 is not a finite number > 0 that fits in f32"],
      [{ shift: Number.POSITIVE_INFINITY }, "shift Infinity is not a finite number > 0"],
      [{ shift: 1e39 }, "shift 1e+39 is not a finite number > 0 that fits in f32"],
      [{ shift: 1e-45 }, "steps 2 with shift 1e-45 give no finite σ schedule"],
      [{ latents }, "seed and latents are mutually exclusive"],
      [{ seed: undefined }, "the initial noise needs a seed or the latents"],
      [{ seed: -1 }, "MOSS-Image: seed -1 is not a non-negative safe integer"],
      [
        { seed: undefined, latents: new Float32Array(10) },
        "latents are not a Float32Array of 32768 values",
      ],
      [
        { seed: undefined, latents: new Float64Array(128 * 256) },
        "latents are not a Float32Array of 32768 values",
      ],
      [{ seed: undefined, latents: broken }, "latents element 5 is non-finite (NaN)"],
    ];
    for (const [overrides, message] of cases) {
      assertThrows(() => plan(overrides), ModelInputError, message);
    }
  });

  it("refuses nothing else: any step count, upstream's whole guidance range, any area under the limit", () => {
    expect(plan({ width: 16, height: 16 }).dit.cells).toEqual({ rows: 1, cols: 1 });
    // 1696 × 1648 = 2,795,008 px, under the limit.
    expect(plan({ width: 1696, height: 1648 }).vae.width).toBe(1696);
    expect(plan({ steps: 1 }).schedule.gridIndices).toEqual([0]);
    const many = plan({ steps: 5000 }).schedule;
    assert(many.gridIndices.length > 0 && many.gridIndices.length <= 1000);
    for (const guidance of [0, 0.5, 7.5]) {
      expect(plan({ guidance }).dit.texts.negative?.tokens).toBe(10);
    }
    expect(plan({ shift: 2.5 }).schedule.gridIndices.length).toBe(2);
    expect(plan({ positive: embeddings(4096) }).dit.texts.positive.tokens).toBe(4096);
    const latents = new Float32Array(128 * 256);
    expect(plan({ seed: undefined, latents }).initial).toEqual({ kind: "latents", data: latents });
  });

  it("plans no negative branch at guidance 1 (no negative given)", () => {
    const accepted = plan({ guidance: 1, negative: undefined });

    expect(accepted.dit.texts.negative).toBeUndefined();
    expect(accepted.guidance).toBe(1);
  });
});

describe("MOSS-Image sampler loop rules (no record needed)", () => {
  const CHANNELS = 128;
  /** The 256² request's latent count (128 × 16 × 16). */
  const COUNT = CHANNELS * 256;
  const latentsOf = (seed: number): Float32Array<ArrayBuffer> => new MossRandn(seed).normals(COUNT);

  it("runs guidance 1 as the positive branch alone through the plain Euler chain, bit for bit", async () => {
    const accepted = plan({ guidance: 1, negative: undefined, steps: 3 });
    const initial = latentsOf(5);
    const velocities = [latentsOf(6), latentsOf(7), latentsOf(8)];
    const { gridIndices, sigmas } = accepted.schedule;
    assertEquals(gridIndices.length, 3);
    const seen: number[] = [];

    const final = await mossDenoise(
      async (_latents, gridIndex) => {
        seen.push(gridIndex);
        return { positive: velocities[seen.length - 1], negative: undefined };
      },
      accepted,
      initial,
    );

    const expected = velocities.reduce(
      (latents, velocity, step) => mossEulerStep(latents, velocity, sigmas[step], sigmas[step + 1]),
      initial,
    );
    expect(seen).toEqual([...gridIndices]);
    assertEquals(firstBitMismatch(final, expected), -1);
  });

  it("fails loudly when the branches predict returns disagree with the guidance (no silent fall back)", async () => {
    const velocity = latentsOf(6);

    await assertRejects(
      () =>
        mossDenoise(
          async () => ({ positive: velocity, negative: undefined }),
          plan({}),
          latentsOf(5),
        ),
      Error,
      "MOSS-Image: step 0 at guidance 3 returned no negative branch",
    );
    await assertRejects(
      () =>
        mossDenoise(
          async () => ({ positive: velocity, negative: velocity }),
          plan({ guidance: 1, negative: undefined }),
          latentsOf(5),
        ),
      Error,
      "MOSS-Image: step 0 at guidance 1 returned a negative branch",
    );
  });

  it("starts the seed path from the first C · N normals of the seed's sequence, the latents path from the latents", () => {
    const seeded = mossInitialLatents(plan({ seed: 42 }), CHANNELS);
    const injected = latentsOf(9);

    assertEquals(seeded.length, COUNT);
    assertEquals(firstBitMismatch(seeded, new MossRandn(42).normals(COUNT)), -1);
    expect(mossInitialLatents(plan({ seed: undefined, latents: injected }), CHANNELS)).toBe(
      injected,
    );
  });
});

const STAGE0 = stage0Dir();
const MIRROR_PRESENT = mossMirrorPresent();

/** The stage-0 DiT records (`moss.reference.SIZES`, height × width) and their latent grids. */
const DIT_RECORDS = [
  { file: "dit-256x256.safetensors", cells: { rows: 16, cols: 16 }, tokens: [63, 10] },
  { file: "dit-256x384.safetensors", cells: { rows: 16, cols: 24 }, tokens: [63, 10] },
] as const;

for (const { file, cells, tokens } of DIT_RECORDS) {
  const url = STAGE0 === undefined ? undefined : new URL(file, STAGE0);
  const present = url !== undefined && filePresent(url);
  describe.skipIf(!present || !MIRROR_PRESENT)(
    `MOSS-Image sampler loop against upstream's trajectory (${file})${skipped(present, "stage-0 DiT record", MOSS_GENERATE_STAGE0)}${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
    () => {
      it("reproduces upstream's latents of both steps bit for bit from upstream's DiT outputs (CFG 3, shift 5)", async () => {
        assert(url !== undefined);
        const { trainingGrid } = await readMossDitAssets((await openMossDit()).components.dit);
        const steps = [0, 1];
        const record = await readRecordTensors(url, [
          "latents.init",
          "text_embeds.positive",
          "text_embeds.negative",
          "sigmas",
          "timesteps",
          ...steps.flatMap((step) => [
            `step${step}.latents_in`,
            `step${step}.timestep`,
            `step${step}.positive.sample`,
            `step${step}.negative.sample`,
          ]),
          "step1.latents_out",
        ]);
        // The record's own request (its metadata: 2 steps, guidance 3; upstream's shift 5).
        const accepted = planMossGenerate(
          {
            width: cells.cols * 16,
            height: cells.rows * 16,
            positive: { data: record["text_embeds.positive"], shape: [tokens[0], 4096] },
            negative: { data: record["text_embeds.negative"], shape: [tokens[1], 4096] },
            steps: 2,
            guidance: 3,
            shift: 5,
            latents: record["latents.init"],
          },
          { ...LIMITS, dit: { latentChannels: 128, trainingGrid } },
        );
        assertEquals(firstBitMismatch(accepted.schedule.sigmas, record["sigmas"]), -1, "σ");
        assertEquals(
          firstBitMismatch(
            Float32Array.from(accepted.schedule.gridIndices, (index) =>
              mossTimestep(trainingGrid, index),
            ),
            record["timesteps"],
          ),
          -1,
          "timesteps",
        );
        assert(accepted.initial.kind === "latents");
        const seen: number[] = [];

        const final = await mossDenoise(
          async (latents, gridIndex) => {
            const step = seen.length;
            seen.push(gridIndex);
            assertEquals(
              firstBitMismatch(latents, record[`step${step}.latents_in`]),
              -1,
              `step ${step}: the latents the forwards receive`,
            );
            assertEquals(
              firstBitMismatch(
                Float32Array.of(mossTimestep(trainingGrid, gridIndex)),
                record[`step${step}.timestep`],
              ),
              -1,
              `step ${step}: the timestep`,
            );
            return {
              positive: record[`step${step}.positive.sample`],
              negative: record[`step${step}.negative.sample`],
            };
          },
          accepted,
          accepted.initial.data,
        );

        expect(seen).toEqual([...accepted.schedule.gridIndices]);
        assertEquals(firstBitMismatch(final, record["step1.latents_out"]), -1, "final latents");
      });
    },
  );
}
