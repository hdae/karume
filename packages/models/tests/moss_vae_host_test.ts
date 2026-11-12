/**
 * The host side of the MOSS-Image VAE stage on the CPU (ADR 0138 decisions 5, 8 and 9, stage 3):
 * the BatchNorm restoration, the unpatchify and the clamp against upstream's recorded values
 * (Uint32-identical), the graph contract of the distribution's VAE containers with its fault
 * injections, the per-device area limit of the acceptance check, and the headroom arithmetic.
 *
 * Sources of the expected values:
 *
 * - the stage-0 VAE records (`goldens/moss-image-sft-stage0-reference/vae-<H>x<W>.safetensors`
 *   of the test assets — `moss.vae_reference`): upstream's `denormalize`, `unpatchify`,
 *   decode and clamp of one seeded latent per size (256² and the non-square 512×768);
 * - the VAE decoder containers of the distribution mirror `models/karume-moss-image/` (read through
 *   its manifest — ADR 0139): the graph declarations and the asset `latent_batch_norm`;
 * - the goldens of `moss.export_vae write` (read from the test-asset root
 *   `goldenDir("moss-image-sft-f32-dyn")`, fetched with `node tools/test-assets/fetch.ts`): their
 *   `latents` / `z` / `decoded` / `image` (upstream's values for the other seeds).
 *
 * Each group skips explicitly, naming the command that gets what it lacks (ADR 0005).
 */

import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { HEADROOM_MARGIN_BYTES } from "../src/session/headroom-margin.ts";
import { mossFusionRules } from "../src/moss/fusion-rules.ts";
import { assertMossImageArea } from "../src/moss/request.ts";
import {
  MOSS_LATENT_BATCH_NORM,
  mossVaeAreaLimit,
  mossVaeGraphContract,
  type MossVaeGraphs,
  parseMossLatentBatchNorm,
} from "../src/moss/vae-contract.ts";
import {
  mossClampImage,
  mossDenormalizeLatents,
  mossUnpatchify,
  mossVaeInput,
} from "../src/moss/vae-host.ts";
import {
  MOSS_VAE_ATTENTION_ROW_BLOCK_BYTES,
  mossVaeStageNeeds,
  planMossVae,
} from "../src/moss/vae-stage.ts";
import {
  assertTestAssetsPresent,
  firstBitMismatch,
  MOSS_FETCH_GOLDENS,
  MOSS_GET_MIRROR,
  mossMirrorPresent,
  readRecordTensors,
  stage0Record,
  testAssetsRunnable,
} from "./helpers/moss-dit.ts";
import {
  loadVaeGolden,
  MOSS_VAE_CASES,
  MOSS_VAE_STAGE0_SEED,
  openMossVae,
  vaeGoldenPresent,
  vaeGoldenUrl,
} from "./helpers/moss-vae.ts";

const stage0Vae = (height: number, width: number): URL =>
  stage0Record(`vae-${height}x${width}.safetensors`);
const STAGE0_URLS = [
  [256, 256],
  [512, 768],
].map(([height, width]) => stage0Vae(height, width));
/** Present, or absent from a complete fetch of the pin (then the group FAILs naming them). */
const STAGE0_RUNNABLE = testAssetsRunnable(STAGE0_URLS);
const MIRROR_PRESENT = mossMirrorPresent();
const GOLDEN_URLS = MOSS_VAE_CASES.map(vaeGoldenUrl);
/** Present, or absent from a complete fetch of the pin (then the group FAILs naming them). */
const GOLDENS_RUNNABLE = testAssetsRunnable(GOLDEN_URLS);
const MISSING_GOLDENS = MOSS_VAE_CASES.filter((testCase) => !vaeGoldenPresent(testCase))
  .map((testCase) => vaeGoldenUrl(testCase).pathname)
  .join(", ");

/**
 * A device whose binding limit is exactly 2 GiB (the RTX 3080 Ti class — ADR 0124), with the dispatch
 * limit the stage's estimates need (they plan with the family's rule list).
 */
const LIMITS_2GIB = {
  maxStorageBufferBindingSize: 2 ** 31,
  maxBufferSize: 2 ** 32,
  maxComputeWorkgroupsPerDimension: 65_535,
};
/** The WebGPU core default binding (128 MiB). */
const LIMITS_CORE = { maxStorageBufferBindingSize: 2 ** 27, maxBufferSize: 2 ** 28 };

/** The suffix that names what a skipped group lacks and the command that gets it. */
const skipped = (present: boolean, what: string, command: string): string =>
  present ? "" : ` (SKIP: no ${what} — ${command})`;

const SIZES = [
  [256, 256],
  [512, 768],
] as const;

describe.skipIf(!STAGE0_RUNNABLE || !MIRROR_PRESENT)(
  `MOSS VAE host functions are Uint32-identical to upstream's (stage-0 records and the mirror)${skipped(STAGE0_RUNNABLE, "stage-0 VAE records", MOSS_FETCH_GOLDENS)}${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    for (const [height, width] of SIZES) {
      it(`${height}x${width}: BatchNorm restoration, unpatchify and clamp`, async () => {
        assertTestAssetsPresent(STAGE0_URLS);
        const { contract } = await openMossVae();
        const record = await readRecordTensors(stage0Vae(height, width), [
          "latents",
          "denormalized",
          "vae.z",
          "vae.decoded",
          "image",
        ]);
        const cells = { rows: height / 16, cols: width / 16 };
        const denormalized = mossDenormalizeLatents(record["latents"], contract.batchNorm, cells);
        assertEquals(firstBitMismatch(denormalized, record["denormalized"]), -1, "denormalize");
        const z = mossUnpatchify(denormalized, contract.ditChannels, cells);
        assertEquals(firstBitMismatch(z, record["vae.z"]), -1, "unpatchify");
        assertEquals(
          firstBitMismatch(
            mossVaeInput(record["latents"], contract.batchNorm, cells),
            record["vae.z"],
          ),
          -1,
          "the decode input",
        );
        assertEquals(firstBitMismatch(mossClampImage(record["vae.decoded"]), record["image"]), -1);
      });
    }
  },
);

describe.skipIf(!MIRROR_PRESENT || !GOLDENS_RUNNABLE)(
  `MOSS VAE host functions on every golden of the VAE series${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}${skipped(GOLDENS_RUNNABLE, `goldens ${MISSING_GOLDENS}`, MOSS_FETCH_GOLDENS)}`,
  () => {
    it("reproduce the goldens' decode input and clamp", async () => {
      assertTestAssetsPresent(GOLDEN_URLS);
      const { contract } = await openMossVae();
      for (const testCase of MOSS_VAE_CASES) {
        const golden = await loadVaeGolden(testCase);
        const cells = { rows: testCase.height / 16, cols: testCase.width / 16 };
        const z = mossVaeInput(golden.latents, contract.batchNorm, cells);
        assertEquals(firstBitMismatch(z, golden.z), -1, `${testCase.name} z`);
        assertEquals(
          firstBitMismatch(mossClampImage(golden.decoded), golden.image),
          -1,
          `${testCase.name} image`,
        );
      }
      // The stage-0 seed's goldens are the stage-0 records (the recipe's gate 3 checks the bits).
      assert(MOSS_VAE_CASES.some((testCase) => testCase.seed === MOSS_VAE_STAGE0_SEED));
    });
  },
);

describe("MOSS VAE host functions (no records)", () => {
  it("unpatchify puts channel 4c + 2a + b at row 2i + a, column 2j + b", () => {
    // [8, 1, 2] → [2, 2, 4]: input channel k holds the value 10·k + j.
    const latents = Float32Array.from(
      { length: 16 },
      (_, at) => 10 * Math.floor(at / 2) + (at % 2),
    );
    const out = mossUnpatchify(latents, 8, { rows: 1, cols: 2 });
    assertEquals([...out.slice(0, 8)], [0, 10, 1, 11, 20, 30, 21, 31]);
    assertEquals([...out.slice(8)], [40, 50, 41, 51, 60, 70, 61, 71]);
    assertThrows(() => mossUnpatchify(latents, 6, { rows: 1, cols: 2 }), Error, "unpatchify");
  });

  it("the clamp keeps NaN (the stage then fails loudly on it)", () => {
    const out = mossClampImage(Float32Array.from([-3, -1, 0.5, 1, 7, Number.NaN]));
    assertEquals([...out.slice(0, 5)], [-1, -1, 0.5, 1, 1]);
    assert(Number.isNaN(out[5]));
  });

  it("denormalize rounds the product before the sum (upstream's two f32 operations)", () => {
    // (1 + 2^-23)·(1 + 2^-22) = 1 + 3·2^-23 + 2^-45: rounding the product to f32 drops the 2^-45,
    // so adding −1 gives 3·2^-23. A single rounding of the whole expression keeps 3·2^-23 + 2^-45
    // (24 significant bits — an f32), which this value tells apart.
    const batchNorm = {
      mean: Float32Array.from([-1]),
      std: Float32Array.from([1 + 2 ** -22]),
    };
    const out = mossDenormalizeLatents(Float32Array.from([1 + 2 ** -23]), batchNorm, {
      rows: 1,
      cols: 1,
    });
    assertEquals(Math.fround(3 * 2 ** -23 + 2 ** -45), 3 * 2 ** -23 + 2 ** -45);
    assertEquals(out[0], 3 * 2 ** -23);
    assertThrows(
      () => mossDenormalizeLatents(Float32Array.from([1, 2]), batchNorm, { rows: 1, cols: 1 }),
      Error,
      "latents",
    );
  });
});

describe.skipIf(!MIRROR_PRESENT)(
  `MOSS VAE graph contract of the mirror's VAE containers${skipped(MIRROR_PRESENT, "mirror models/karume-moss-image/", MOSS_GET_MIRROR)}`,
  () => {
    it("reads the widths, the scale and the largest tensor per pixel from the graphs", async () => {
      const { contract } = await openMossVae();
      expect({
        latentChannels: contract.latentChannels,
        hiddenChannels: contract.hiddenChannels,
        spatialScale: contract.spatialScale,
        ditChannels: contract.ditChannels,
        bytesPerCell: contract.bytesPerCell,
      }).toEqual({
        latentChannels: 32,
        hiddenChannels: 384,
        spatialScale: 8,
        ditChannels: 128,
        // [1, 192, 8h, 8w] f32: 192 · 64 · 4 B per VAE cell = 768 B per pixel (ADR 0138 decision 8)
        bytesPerCell: 49_152,
      });
      expect(contract.batchNorm.std.every((value) => value > 0)).toBe(true);
    });

    it("refuses declarations the host cannot build or whose area limit is not the area's", async () => {
      const { components } = await openMossVae();
      const withValue = (shape: readonly (number | string)[]): MossVaeGraphs => ({
        ...components,
        post: {
          graph: {
            ...components.post.graph,
            values: { ...components.post.graph.values, extra: { dtype: "f32", shape } },
          },
        },
      });
      assertThrows(() => mossVaeGraphContract(withValue([1, 4, "h+1", "w"]), "t"), Error, "coeff");
      assertThrows(() => mossVaeGraphContract(withValue([1, 4, "h", 7]), "t"), Error, "area");
      const noAsset: MossVaeGraphs = { ...components, pre: { ...components.pre, assets: {} } };
      assertThrows(() => mossVaeGraphContract(noAsset, "t"), Error, MOSS_LATENT_BATCH_NORM);
      const swapped: MossVaeGraphs = { ...components, attn: components.post };
      assertThrows(() => mossVaeGraphContract(swapped, "t"), Error, "vae_decoder_attn");
      assertThrows(() => parseMossLatentBatchNorm(new ArrayBuffer(16)), Error);
    });

    it("caps the area per device and names the limit; the cap fits the 2 GiB binding", async () => {
      const { contract } = await openMossVae();
      const limit = mossVaeAreaLimit(contract, LIMITS_2GIB);
      // floor(2^31 / 49,152) = 43,690 VAE cells = 2,796,160 px
      expect(limit).toEqual({ maxPixels: 2_796_160, bytesPerPixel: 768, bindingBytes: 2 ** 31 });
      // 1696×1648 = 2,795,008 px is accepted; 1712×1648 = 2,821,376 px is not.
      planMossVae({ width: 1648, height: 1696 }, contract, LIMITS_2GIB);
      const error = assertThrows(
        () => planMossVae({ width: 1648, height: 1712 }, contract, LIMITS_2GIB),
        ModelInputError,
      );
      expect(error.message).toContain("area limit of 2796160 px");
      expect(error.message).toContain("768 B per pixel");
      expect(error.message).toContain(`${2 ** 31} B`);
      // The limit follows the device: at the core default binding 512² no longer fits.
      expect(mossVaeAreaLimit(contract, LIMITS_CORE).maxPixels).toBe(174_720);
      assertThrows(() => assertMossImageArea(512, 512, mossVaeAreaLimit(contract, LIMITS_CORE)));
      planMossVae({ width: 256, height: 256 }, contract, LIMITS_CORE);
      // The binding over the buffer: the smaller one decides.
      expect(
        mossVaeAreaLimit(contract, { maxStorageBufferBindingSize: 2 ** 32, maxBufferSize: 2 ** 31 })
          .bindingBytes,
      ).toBe(2 ** 31);
      // The size rules still run first (a multiple of 16).
      assertThrows(
        () => planMossVae({ width: 250, height: 256 }, contract, LIMITS_2GIB),
        ModelInputError,
      );
    });

    it("needs the three estimates (the attention's with its row-block budget), the residents, the largest part and the margin", async () => {
      const { components, contract } = await openMossVae();
      const plan = planMossVae({ width: 1648, height: 1696 }, contract, LIMITS_2GIB);
      const bindings = { h: 212, w: 206 };
      const estimate = (
        component: typeof components.pre,
        values: Record<string, number>,
        attentionRowBlockMaxBytes?: number,
      ) =>
        component.estimate({
          ...LIMITS_2GIB,
          bindings: values,
          ...(attentionRowBlockMaxBytes === undefined ? {} : { attentionRowBlockMaxBytes }),
          // The stage's Sessions and estimates plan with the family's rule list.
          fusionRules: mossFusionRules(undefined),
        }).peakAccountedBytes;
      const post = estimate(components.post, bindings);
      const attention = estimate(
        components.attn,
        { S: 212 * 206 },
        MOSS_VAE_ATTENTION_ROW_BLOCK_BYTES,
      );
      // Precondition: at the cap the budget is reached (the attention's estimate is smaller with it).
      expect(attention).toBeLessThan(estimate(components.attn, { S: 212 * 206 }));
      const expected =
        estimate(components.pre, bindings) +
        attention +
        post +
        2 * 384 * 212 * 206 * 4 +
        3 * 1696 * 1648 * 4 +
        Math.max(
          components.pre.maxPartBytes,
          components.attn.maxPartBytes,
          components.post.maxPartBytes,
        ) +
        HEADROOM_MARGIN_BYTES;
      expect(mossVaeStageNeeds({ limits: LIMITS_2GIB, components, contract }, plan)).toBe(expected);
      // The design's hand count (docs/research/2026-10-28-moss-image-dit-vae.md §3.4): at the
      // full-resolution resnet 192 → 96 its input, its SiLU output and its conv output (480
      // channels × 4 B per pixel) are alive together, so the post graph's accounted peak cannot be
      // below 1,920 B per pixel.
      expect(post).toBeGreaterThanOrEqual(1_920 * 1696 * 1648);
      console.info(
        `[moss vae] estimate at 1696×1648: post ${post} B (${(post / (1696 * 1648)).toFixed(0)} B/px),` +
          ` stage ${mossVaeStageNeeds({ limits: LIMITS_2GIB, components, contract }, plan)} B`,
      );
    });
  },
);
