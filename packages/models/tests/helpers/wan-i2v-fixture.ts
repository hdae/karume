/**
 * Wan2.2 TI2V の I2V の通しの CPU の参照（`pipeline_steps_i2v.<case>.safetensors` — ADR 0121 段 9b）の置き場と、埋め込まれた
 * 元画像の形。I2V の通しの e2e（`e2e_wan_ti2v_i2v_pipeline_test.ts`）と、条件画像をこの参照から
 * 読む自機 A/B 門の I2V のケース（`e2e_wan_ti2v_ab_test.ts`）が共有する。綴りは**ここ 1 か所**に置く（書き手の出力名や
 * モジュール名が変わったときに片方だけが古い置き場を見て、opt-in の門が黙って SKIP し続けないように）。
 *
 * 書き手は recipe の `tools/export-recipes/wan/ti2v_i2v_few_step_ref.py`（中身の規約は I2V の通しの e2e のモジュール doc）。
 */

import { goldenDir } from "@karume/test-support/test-assets";
import { WAN_TI2V_SERIES_NAME } from "./wan-ti2v-dit.ts";

/**
 * The references' place (goldens no distribution carries — the test-asset root's folder of the int8 DiT's series,
 * ADR 0139 decision 2; the same folder as the T2V references, another prefix).
 */
export const WAN_I2V_STEPS_ROOT = goldenDir(WAN_TI2V_SERIES_NAME);

/** 参照 1 本の置き場（`name` は recipe の `FIXTURE_CASES` の名前 — 例 `band-boxing-cats`）。 */
export const wanI2vStepsFixtureUrl = (name: string): URL =>
  new URL(`pipeline_steps_i2v.${name}.safetensors`, WAN_I2V_STEPS_ROOT);

/**
 * The shape of the source image `source` embedded in the reference (the pixels of the corpus image
 * `corpus/wan-i2v/<name>-832x480.png` of the test-asset root — HWC).
 */
export const WAN_I2V_SOURCE_SHAPE: readonly number[] = [480, 832, 3];
