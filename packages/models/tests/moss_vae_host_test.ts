/**
 * The host side of the MOSS-Image VAE stage on the CPU (ADR NNNN decisions 5, 8 and 9, plan step 3):
 * the BatchNorm restoration, the unpatchify and the clamp against upstream's recorded values
 * (Uint32-identical), the graph contract of the written VAE series with its fault injections, the
 * per-device area limit of the acceptance check, and the headroom arithmetic.
 *
 * Sources of the expected values:
 *
 * - the stage-0 VAE records (`outputs/bench/karume-moss-image/<date>_stage0-reference/
 *   vae-<H>x<W>.safetensors` — `moss.vae_reference`): upstream's `denormalize`, `unpatchify`,
 *   decode and clamp of one seeded latent per size (256² and the non-square 512×768);
 * - the VAE series (`outputs/series/moss-image-sft-f32-dyn/` — `moss.export_vae write`): the graph
 *   declarations, the asset `latent_batch_norm` and the goldens' `latents` / `z` / `decoded` /
 *   `image` (upstream's values for the other seeds).
 *
 * Each group skips explicitly, naming the command that writes what it lacks (ADR 0005).
 */

import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { HEADROOM_MARGIN_BYTES } from "../src/anima/residency.ts";
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
import { firstBitMismatch, readRecordTensors, stage0Dir } from "./helpers/moss-dit.ts";
import {
  loadVaeGolden,
  MOSS_GENERATE_VAE_SERIES,
  MOSS_GENERATE_VAE_STAGE0,
  MOSS_VAE_CASES,
  MOSS_VAE_STAGE0_SEED,
  mossVaeSeriesPresent,
  openMossVaeSeries,
  vaeGoldenPresent,
} from "./helpers/moss-vae.ts";

const STAGE0 = stage0Dir();
const SERIES_PRESENT = mossVaeSeriesPresent();
const GOLDENS_PRESENT = MOSS_VAE_CASES.every(vaeGoldenPresent);

/** A device whose binding limit is exactly 2 GiB (the RTX 3080 Ti class — ADR 0124). */
const LIMITS_2GIB = { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 32 };
/** The WebGPU core default binding (128 MiB). */
const LIMITS_CORE = { maxStorageBufferBindingSize: 2 ** 27, maxBufferSize: 2 ** 28 };

/** The suffix that names what a skipped group lacks and the command that writes it. */
const skipped = (present: boolean, what: string, command: string): string =>
  present ? "" : ` (SKIP: no ${what} — ${command})`;

const SIZES = [
  [256, 256],
  [512, 768],
] as const;

describe.skipIf(STAGE0 === undefined || !SERIES_PRESENT)(
  `MOSS VAE host functions are Uint32-identical to upstream's (stage-0 records and the series)${skipped(STAGE0 !== undefined, "stage-0 VAE record", MOSS_GENERATE_VAE_STAGE0)}${skipped(SERIES_PRESENT, "VAE series", MOSS_GENERATE_VAE_SERIES)}`,
  () => {
    for (const [height, width] of SIZES) {
      it(`${height}x${width}: BatchNorm restoration, unpatchify and clamp`, async () => {
        const { contract } = await openMossVaeSeries();
        const record = await readRecordTensors(
          new URL(`vae-${height}x${width}.safetensors`, STAGE0),
          ["latents", "denormalized", "vae.z", "vae.decoded", "image"],
        );
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

describe.skipIf(!SERIES_PRESENT || !GOLDENS_PRESENT)(
  `MOSS VAE host functions on every golden of the series${skipped(SERIES_PRESENT && GOLDENS_PRESENT, "VAE series or goldens", MOSS_GENERATE_VAE_SERIES)}`,
  () => {
    it("reproduce the goldens' decode input and clamp", async () => {
      const { contract } = await openMossVaeSeries();
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

describe.skipIf(!SERIES_PRESENT)(
  `MOSS VAE graph contract of the written series${skipped(SERIES_PRESENT, "VAE series", MOSS_GENERATE_VAE_SERIES)}`,
  () => {
    it("reads the widths, the scale and the largest tensor per pixel from the graphs", async () => {
      const { contract } = await openMossVaeSeries();
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
        // [1, 192, 8h, 8w] f32: 192 · 64 · 4 B per VAE cell = 768 B per pixel (plan §1.5)
        bytesPerCell: 49_152,
      });
      expect(contract.batchNorm.std.every((value) => value > 0)).toBe(true);
    });

    it("refuses declarations the host cannot build or whose area limit is not the area's", async () => {
      const { components } = await openMossVaeSeries();
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
      const { contract } = await openMossVaeSeries();
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
      const { components, contract } = await openMossVaeSeries();
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
      // The plan's hand count (§1.5): at the full-resolution resnet 192 → 96 its input, its SiLU
      // output and its conv output (480 channels × 4 B per pixel) are alive together, so the post
      // graph's accounted peak cannot be below 1,920 B per pixel.
      expect(post).toBeGreaterThanOrEqual(1_920 * 1696 * 1648);
      console.info(
        `[moss vae] estimate at 1696×1648: post ${post} B (${(post / (1696 * 1648)).toFixed(0)} B/px),` +
          ` stage ${mossVaeStageNeeds({ limits: LIMITS_2GIB, components, contract }, plan)} B`,
      );
    });
  },
);
