/**
 * What the tests of the Wan2.2 TI2V I2V condition-image preprocessing (ADR 0121 step 9a) share: the place and shape of
 * the LANCZOS fixture of synthetic images, the place, case table and keys of the real-image goldens, and the tolerance
 * band of the encoder's output ({@link WAN_I2V_ENCODER_BAND}). The spellings live **in this one place** (when the
 * writer's spelling changes, only this file is updated).
 *
 * ## The LANCZOS fixture (git-tracked — missing is red, not SKIP)
 *
 * The writer is the recipe's `tools/export-recipes/wan/lanczos_fixture.py`. `tests/fixtures/wan-i2v/lanczos.json` is
 * the case table (`source.pillow` (version), `tensors` (the name of the pixel safetensors), `cases[]` = `name` / `why`
 * / `height` / `width` / `outHeight` / `outWidth`), and `lanczos.safetensors` holds the pixels (U8 `<name>.input`
 * `[height, width, 3]` and `<name>.resized` `[outHeight, outWidth, 3]` — RGB8, row-major).
 *
 * ## The real-image goldens (not git-tracked — none at all is an explicit SKIP; only some is a FAIL)
 *
 * Written by the recipe `tools/export-recipes/wan/export_vae_encoder.py` (`GOLDEN_CASES` — goldens shared with the
 * encoder's GPU gate) and read from `goldens/wan2.2-ti2v-5b-f16-dyn/vae_encoder.<case>.safetensors` of the test-asset
 * root (ADR 0139 decision 2). The case name is `<image>-<width>x<height>` (fit = crop) /
 * `<image>-<width>x<height>-stretch`. Content (what the tests read):
 *
 * - `source` U8 `[480, 832, 3]` — the test image's pixels (the chain runs without decoding the PNG)
 * - `rgb8` U8 `[H, W, 3]` — the output of the fit preprocessing (Pillow's LANCZOS; for crop, after the center crop)
 * - crop only: `encoder_input` F32 `[12, 1, H/2, W/2]` (`[-1, 1]` → upstream's `patchify`, no batch axis) and `mu` /
 *   `latent` F32 `[48, 1, h, w]` (the mu of upstream's non-tiled encode and its normalization — verbatim from the
 *   upstream I2V pipeline)
 * - `__metadata__`: `image`, `image_sha256`, `width`, `height`, `fit`, `role`, `version_pillow` and others
 */

import type { I2vFit } from "../../src/wan/i2v-preprocess.ts";
import { WAN_TI2V_VAE_GOLDEN_DIR } from "./wan-ti2v-vae.ts";

/**
 * The LANCZOS fixture's case table (git-tracked — the pixel safetensors is the file named by the table's `tensors`, in
 * the same directory).
 */
export const WAN_I2V_LANCZOS_FIXTURE = new URL("../fixtures/wan-i2v/lanczos.json", import.meta.url);

/**
 * The test images (the sha256 of the PNG files `corpus/wan-i2v/<name>-832x480.png` of the test-asset root — ADR 0139
 * decision 2; the same values as the recipe's `IMAGES`, frozen twice).
 * Checked against a golden's `image_sha256` (that the golden was taken from this image).
 */
export const WAN_I2V_IMAGE_SHA256: Readonly<Record<string, string>> = {
  "boxing-cats": "4689f815503859b4d08b464f21239b0ab5066b4da06e7130aa32a60ef1fe5888",
  "cat-dog-baking": "75d6d9c514190806baa4fc5a9d0877e04d4c50ba940057d737915bfb354f3c0e",
  ferret: "a6b740a2ec936d2a102839441f9caabd787fc1d4e27126bbed4be429aef7bd54",
};

/** One golden (the recipe's `GoldenCase`). */
export type WanI2vGoldenCase = {
  readonly image: string;
  readonly width: number;
  readonly height: number;
  readonly fit: I2vFit;
};

/**
 * The golden cases (the same set as the recipe's `GOLDEN_CASES` — **fixed here rather than enumerated**, so that cases
 * do not silently vanish in an environment where only some were written). Crop is 3 sizes (the 2 accepted sizes —
 * 704×1280 is portrait with a landscape image given explicitly — and 256×160 on the downscaling path (outside the
 * accepted set)) × 3 images; stretch is 1280×704 × 3 images.
 */
export const WAN_I2V_GOLDEN_CASES: readonly WanI2vGoldenCase[] = [
  ...(
    [
      [1280, 704],
      [704, 1280],
      [256, 160],
    ] as const
  ).flatMap(([width, height]) =>
    Object.keys(WAN_I2V_IMAGE_SHA256).map((image) => ({
      image,
      width,
      height,
      fit: "crop" as const,
    })),
  ),
  ...Object.keys(WAN_I2V_IMAGE_SHA256).map((image) => ({
    image,
    width: 1280,
    height: 704,
    fit: "stretch" as const,
  })),
];

/** The case name (the recipe's `GoldenCase.name`). */
export const wanI2vGoldenName = ({ image, width, height, fit }: WanI2vGoldenCase): string =>
  `${image}-${width}x${height}${fit === "crop" ? "" : `-${fit}`}`;

export const wanI2vGoldenUrl = (testCase: WanI2vGoldenCase): URL =>
  new URL(`vae_encoder.${wanI2vGoldenName(testCase)}.safetensors`, WAN_TI2V_VAE_GOLDEN_DIR);

/**
 * The tolerance band (maximum absolute difference, absolute) of the VAE encoder's output (after the mu normalization —
 * the DiT's condition latent). Shared by the encoder's GPU check (`e2e_wan_ti2v_vae_encoder_test.ts` — compared with
 * the golden's `latent`) and the condition-latent check of the I2V end-to-end e2e (`e2e_wan_ti2v_i2v_pipeline_test.ts`
 * — compared with the end-to-end reference's `condition_latents`; the reference's writer checks bit equality with the
 * 9a golden, so it is the same quantity).
 *
 * Derivation (2026-10-09, RTX 3080 Ti, Deno, all 9 cases — the encoder's GPU check): the worst deciding case,
 * boxing-cats 704×1280, 1.550e-5 × 5 = 7.749e-5 → rounded up to 2 significant digits, 7.8e-5. The same day, with the 9
 * cases moved into the default lane, it was re-run twice and every case's maxAbs equalled the table below (the number
 * of runs that derived the band is not recorded — the records were overwritten — so only the runs that can be confirmed
 * are written).
 *
 * | size | boxing-cats (deciding) | cat-dog-baking (deciding) | ferret (acceptance) | reference max abs |
 * |---|---|---|---|---|
 * | 1280×704 | 1.150e-5 | 1.264e-5 | 5.394e-6 | 2.95–3.34 |
 * | 704×1280 | 1.550e-5 | 1.073e-5 | 6.557e-6 | 2.89–3.68 |
 * | 256×160 | 5.186e-6 | 4.768e-6 | 4.813e-6 | 3.01–4.04 |
 *
 * maxAbs under fault injection (ferret 1280×704): patchify's sub-index mixed up 2.44, std and 1/std mixed up 2.54, h
 * and w of the post binding swapped 1.18, logvar's 48 channels 67.3, AvgDown3D's even/odd mixed up 2.77 (each over
 * 15,000 times the band).
 *
 * MUST: do not change this value or the deciding cases after looking at the acceptance (ferret) result. If the
 * acceptance case falls outside the band, investigate the cause instead of widening the band.
 */
export const WAN_I2V_ENCODER_BAND = 7.8e-5;

/** The golden's tensor keys. */
export const WAN_I2V_GOLDEN_KEYS = {
  source: "source",
  rgb8: "rgb8",
  encoderInput: "encoder_input",
  mu: "mu",
  latent: "latent",
} as const;

/** The fields of the golden's `__metadata__`. */
export const WAN_I2V_GOLDEN_METADATA = {
  image: "image",
  imageSha256: "image_sha256",
  width: "width",
  height: "height",
  fit: "fit",
  /** The role in the encoder's band (crop is `band` / `accept`, stretch is `host` — the recipe's `GoldenCase.role`). */
  role: "role",
  pillow: "version_pillow",
} as const;
