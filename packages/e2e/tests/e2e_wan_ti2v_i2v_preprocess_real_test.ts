// Preprocessing of the Wan2.2 TI2V I2V conditioning image — the **real-image** part (ADR 0121 stage 9a, no GPU).
//
// The synthetic-image side (`wan_ti2v_i2v_preprocess_test.ts`) pins the formulas, the ordering and Pillow's LANCZOS
// piece by piece; this file compares the real pixels of the test images (the first frames of three Wan2.1 visual-check
// clips, 832×480) **bit for bit** against goldens the recipe took with the upstream Python (Pillow's LANCZOS, the
// official center crop / diffusers' direct resize, diffusers' `patchify`, the I2V pipeline's normalization). No
// tolerance band (every stage is integer or a single f32 rounding and gives upstream's value — a mismatch is a copy
// mistake).
//
// Three places are compared: the resized RGB8 (`rgb8` — LANCZOS and the crop position), the encoder input
// (`encoder_input` — `[-1, 1]` and the patchify ordering), and the normalization at the encoder output (`mu` →
// `latent`). The sizes are the golden's (explicit), and the downscale path outside the accepted set (256×160) runs
// too — the accepted-set check lives in `selectWanI2vSize`; this file looks at the preprocessing values. So the chain
// runs as the composition of the pieces (`fitWanI2vImage` → `wanI2vPixels` → `patchifyWanImage`) (a host test pins
// that `preprocessWanI2vImage` is the same composition).
//
// ## Environments without the assets
//
// The goldens (in the test-asset root `outputs/test-assets/` — ADR 0139 decision 2; written together with the encoder's
// three graphs) are not tracked by git. **None present → explicit SKIP**, **some → FAIL** (a set that broke off while
// being fetched is not rounded to a SKIP). Their place and form are in `helpers/wan-i2v-image.ts`.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import {
  fitWanI2vImage,
  patchifyWanImage,
  wanI2vPixels,
} from "../../models/src/wan/i2v-preprocess.ts";
import { normalizeWanLatents } from "../../models/src/wan/latents.ts";
import {
  WAN_I2V_GOLDEN_CASES,
  WAN_I2V_GOLDEN_KEYS,
  WAN_I2V_GOLDEN_METADATA,
  WAN_I2V_IMAGE_SHA256,
  wanI2vGoldenName,
  wanI2vGoldenUrl,
} from "../../models/tests/helpers/wan-i2v-image.ts";
import {
  filePresent,
  firstBitMismatch,
  TEST_ASSETS_FETCH,
} from "../../models/tests/helpers/wan-ti2v-dit.ts";
import { readBuffer, viewOf } from "../../models/tests/helpers/safetensors.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";

/** Counted synchronously because it is needed at registration time (it decides `ignore`). */
const present = WAN_I2V_GOLDEN_CASES.filter((testCase) =>
  filePresent(wanI2vGoldenUrl(testCase)),
).length;
/**
 * A complete fetch of the pinned test-asset dataset makes a missing golden a FAIL (the completeness case below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${TEST_ASSETS_FETCH}`;

if (present === 0) {
  console.warn(
    `[karume] the real-image I2V preprocessing check is SKIPPED (goldens 0/${WAN_I2V_GOLDEN_CASES.length}). ` +
      `(${MISSING_GOLDENS})`,
  );
}

const metadataOf = (file: SafetensorsFile, key: string, where: string): string => {
  const value = file.metadata.get(key);
  if (value === undefined) throw new Error(`${where}: __metadata__.${key} is missing`);
  return value;
};

/** Reads a U8 tensor, checking its shape. */
const bytesOf = (
  file: SafetensorsFile,
  key: string,
  shape: readonly number[],
  where: string,
): Uint8Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, "U8", `${where}: ${key} dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} shape`);
  return new Uint8Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

/**
 * Reads an F32 tensor, checking its shape. It copies before reading (with U8 tensors placed before it, the F32 start
 * is not guaranteed to fall on a 4-byte boundary).
 */
const floatsOf = (
  file: SafetensorsFile,
  key: string,
  shape: readonly number[],
  where: string,
): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, "F32", `${where}: ${key} dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} shape`);
  return new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

describe("I2V real-image preprocessing and encoder-output normalization × upstream goldens (bit-identical)", () => {
  it.skipIf(present === 0 && !FETCHED)(
    `all ${WAN_I2V_GOLDEN_CASES.length} goldens are present (none → SKIP, only some → FAIL)`,
    () => {
      const missing = WAN_I2V_GOLDEN_CASES.map(wanI2vGoldenUrl)
        .filter((url) => !filePresent(url))
        .map((url) => url.pathname);
      assertEquals(missing, [], `missing goldens (${MISSING_GOLDENS})`);
    },
  );

  for (const testCase of WAN_I2V_GOLDEN_CASES) {
    const where = wanI2vGoldenName(testCase);
    it.skipIf(present === 0 && !FETCHED)(where, async () => {
      const url = wanI2vGoldenUrl(testCase);
      assert(filePresent(url), `${where}: golden ${url.pathname} is missing`);
      const file = parseSafetensors(await readBuffer(url));
      const meta = WAN_I2V_GOLDEN_METADATA;
      const { width, height, fit } = testCase;
      assertEquals(metadataOf(file, meta.image, where), testCase.image, `${where}: image`);
      assertEquals(
        metadataOf(file, meta.imageSha256, where),
        WAN_I2V_IMAGE_SHA256[testCase.image],
        `${where}: image sha256 differs from the table`,
      );
      assertEquals(metadataOf(file, meta.fit, where), fit, `${where}: fit`);
      assertEquals(
        [metadataOf(file, meta.width, where), metadataOf(file, meta.height, where)],
        [String(width), String(height)],
        `${where}: output size`,
      );
      assert(metadataOf(file, meta.pillow, where).length > 0, `${where}: Pillow version is empty`);

      // Resize (Pillow's RGB8).
      const source = {
        data: bytesOf(file, WAN_I2V_GOLDEN_KEYS.source, [480, 832, 3], where),
        width: 832,
        height: 480,
      };
      const expectedRgb = bytesOf(file, WAN_I2V_GOLDEN_KEYS.rgb8, [height, width, 3], where);
      const fitted = fitWanI2vImage(source, { width, height }, fit);
      const firstRgbMismatch = expectedRgb.findIndex((value, at) => value !== fitted.data[at]);
      assertEquals(
        firstRgbMismatch,
        -1,
        `${where}: RGB8 differs from Pillow (first at ${firstRgbMismatch}: ${
          fitted.data[firstRgbMismatch]
        } vs ${expectedRgb[firstRgbMismatch]})`,
      );
      if (fit !== "crop") return; // A stretch golden holds only the preprocessed RGB8 (the encoder is not run).

      // Encoder input (`[-1, 1]` → patchify).
      const patchSize = WAN22_TI2V_GENERATION.vaePatchSize;
      const expectedInput = floatsOf(
        file,
        WAN_I2V_GOLDEN_KEYS.encoderInput,
        [3 * patchSize * patchSize, 1, height / patchSize, width / patchSize],
        where,
      );
      const input = patchifyWanImage(wanI2vPixels(fitted), [3, height, width], patchSize);
      const firstInputMismatch = firstBitMismatch(input, expectedInput);
      assertEquals(
        firstInputMismatch,
        -1,
        `${where}: encoder input differs from upstream (first at ${firstInputMismatch}: ${
          input[firstInputMismatch]
        } vs ${expectedInput[firstInputMismatch]})`,
      );

      // Normalization at the encoder output (upstream's mu → the conditioning latent).
      // Spatial compression 16 = the encoder graph's ratio 8 × patchify 2 (the same value as `wanSpatialCompression`).
      const compression = 8 * patchSize;
      const latentShape = [
        WAN22_TI2V_GENERATION.latents.mean.length,
        1,
        height / compression,
        width / compression,
      ];
      const mu = floatsOf(file, WAN_I2V_GOLDEN_KEYS.mu, latentShape, where);
      const expectedLatent = floatsOf(file, WAN_I2V_GOLDEN_KEYS.latent, latentShape, where);
      const latent = normalizeWanLatents(mu, WAN22_TI2V_GENERATION.latents);
      const firstLatentMismatch = firstBitMismatch(latent, expectedLatent);
      assertEquals(
        firstLatentMismatch,
        -1,
        `${where}: normalization differs from upstream (first at ${firstLatentMismatch}: ${
          latent[firstLatentMismatch]
        } vs ${expectedLatent[firstLatentMismatch]})`,
      );
    });
  }
});
