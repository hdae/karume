/**
 * Where the CPU reference of the Wan2.2 TI2V I2V end-to-end run (`pipeline_steps_i2v.<case>.safetensors` — ADR 0121
 * step 9b) lives, and the shape of the source image embedded in it. Shared by the I2V end-to-end e2e
 * (`e2e_wan_ti2v_i2v_pipeline_test.ts`) and the I2V case of the local A/B check that reads its condition image from
 * this reference (`e2e_wan_ti2v_ab_test.ts`). The spelling lives **in this one place** (so that when the writer's
 * output name or module name changes, one side does not keep looking at the old place and the opt-in check does not
 * silently keep SKIPping).
 *
 * The writer is the recipe's `tools/export-recipes/wan/ti2v_i2v_few_step_ref.py` (the contents' conventions are in the
 * module doc of the I2V end-to-end e2e).
 */

import { goldenDir } from "@karume/test-support/test-assets";
import { WAN_TI2V_SERIES_NAME } from "./wan-ti2v-dit.ts";

/**
 * The references' place (goldens no distribution carries — the test-asset root's folder of the int8 DiT's series,
 * ADR 0139 decision 2; the same folder as the T2V references, another prefix).
 */
export const WAN_I2V_STEPS_ROOT = goldenDir(WAN_TI2V_SERIES_NAME);

/** Where one reference lives (`name` is a name of the recipe's `FIXTURE_CASES` — e.g. `band-boxing-cats`). */
export const wanI2vStepsFixtureUrl = (name: string): URL =>
  new URL(`pipeline_steps_i2v.${name}.safetensors`, WAN_I2V_STEPS_ROOT);

/**
 * The shape of the source image `source` embedded in the reference (the pixels of the corpus image
 * `corpus/wan-i2v/<name>-832x480.png` of the test-asset root — HWC).
 */
export const WAN_I2V_SOURCE_SHAPE: readonly number[] = [480, 832, 3];
