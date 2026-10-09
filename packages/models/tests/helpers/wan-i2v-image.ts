/**
 * Wan2.2 TI2V の I2V の条件画像の前処理（ADR 0121 段 9a）のテストが共有するもの: 合成画像の LANCZOS の fixture の
 * 置き場と形、実画像の golden の置き場・ケースの表・キー。綴りは**ここ 1 か所**に置く（書き手の綴りが変わったら、
 * ここだけを合わせる）。
 *
 * ## LANCZOS の fixture（git 追跡 — 無ければ SKIP ではなく赤）
 *
 * 書き手は recipe の `tools/export-recipes/wan/lanczos_fixture.py`。`tests/fixtures/wan-i2v/lanczos.json` がケース表
 * （`source.pillow`〈版〉・`tensors`〈画素の safetensors の名前〉・`cases[]` = `name` / `why` / `height` / `width` /
 * `outHeight` / `outWidth`）、`lanczos.safetensors` が画素（U8 `<name>.input` `[height, width, 3]` と
 * `<name>.resized` `[outHeight, outWidth, 3]` — RGB8 の行優先）。
 *
 * ## 実画像の golden（git 追跡外 — 1 本も無ければ明示 SKIP・一部だけなら FAIL）
 *
 * 書き手は recipe の `tools/export-recipes/wan/export_vae_encoder.py`（`GOLDEN_CASES` — encoder の GPU の門と共有の
 * golden）。置き場は `outputs/series/wan2.2-ti2v-5b-f16-dyn/vae_encoder.<ケース名>.safetensors`、ケース名は
 * `<画像>-<幅>x<高さ>`（fit = crop）/ `<画像>-<幅>x<高さ>-stretch`。中身（このテストが読む分）:
 *
 * - `source` U8 `[480, 832, 3]` — テスト画像の画素（PNG を decode せずに鎖を回せる）
 * - `rgb8` U8 `[H, W, 3]` — fit の前処理（Pillow の LANCZOS・crop は中央クロップの後）の出力
 * - crop だけ: `encoder_input` F32 `[12, 1, H/2, W/2]`（`[-1, 1]` → 上流の `patchify`・バッチ軸無し）・
 *   `mu` / `latent` F32 `[48, 1, h, w]`（上流の非タイル encode の mu と、その正規化 — 上流の I2V パイプラインの逐語）
 * - `__metadata__`: `image`・`image_sha256`・`width`・`height`・`fit`・`role`・`version_pillow` ほか
 */

import type { WanI2vFit } from "../../src/wan/i2v-preprocess.ts";
import { WAN_TI2V_VAE_ROOT } from "./wan-ti2v-vae.ts";

/** LANCZOS の fixture のケース表（git 追跡 — 画素の safetensors は表の `tensors` が名指す同じディレクトリのファイル）。 */
export const WAN_I2V_LANCZOS_FIXTURE = new URL(
  "../fixtures/wan-i2v/lanczos.json",
  import.meta.url,
);

/**
 * テスト画像（`inputs/wan-i2v/<名前>-832x480.png` の PNG ファイルの sha256 — recipe の `IMAGES` と同じ値の二重凍結）。
 * golden の `image_sha256` と照合する（golden がこの画像から採られたこと）。
 */
export const WAN_I2V_IMAGE_SHA256: Readonly<Record<string, string>> = {
  "boxing-cats": "4689f815503859b4d08b464f21239b0ab5066b4da06e7130aa32a60ef1fe5888",
  "cat-dog-baking": "75d6d9c514190806baa4fc5a9d0877e04d4c50ba940057d737915bfb354f3c0e",
  "ferret": "a6b740a2ec936d2a102839441f9caabd787fc1d4e27126bbed4be429aef7bd54",
};

/** golden 1 本（recipe の `GoldenCase`）。 */
export type WanI2vGoldenCase = {
  readonly image: string;
  readonly width: number;
  readonly height: number;
  readonly fit: WanI2vFit;
};

/**
 * golden のケース（recipe の `GOLDEN_CASES` と同じ組 — **列挙結果ではなくここで固定する**: 一部だけ書いた環境で
 * ケースが黙って消えないように）。crop は 3 寸法（受理の 2 寸法 — 704×1280 は横長の画像を明示した縦長 — と縮小の経路の
 * 256×160〈受理の外〉）× 3 枚、stretch は 1280×704 × 3 枚。
 */
export const WAN_I2V_GOLDEN_CASES: readonly WanI2vGoldenCase[] = [
  ...([[1280, 704], [704, 1280], [256, 160]] as const).flatMap(([width, height]) =>
    Object.keys(WAN_I2V_IMAGE_SHA256).map((image) => ({
      image,
      width,
      height,
      fit: "crop" as const,
    }))
  ),
  ...Object.keys(WAN_I2V_IMAGE_SHA256).map((image) => ({
    image,
    width: 1280,
    height: 704,
    fit: "stretch" as const,
  })),
];

/** ケース名（recipe の `GoldenCase.name`）。 */
export const wanI2vGoldenName = ({ image, width, height, fit }: WanI2vGoldenCase): string =>
  `${image}-${width}x${height}${fit === "crop" ? "" : `-${fit}`}`;

export const wanI2vGoldenUrl = (testCase: WanI2vGoldenCase): URL =>
  new URL(`vae_encoder.${wanI2vGoldenName(testCase)}.safetensors`, WAN_TI2V_VAE_ROOT);

/** SKIP 時にそのまま貼れる golden の生成コマンド（encoder の 3 グラフと一組で書く）。 */
export const WAN_I2V_GOLDEN_GENERATE = "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.export_vae_encoder";

/** golden のテンソルのキー。 */
export const WAN_I2V_GOLDEN_KEYS = {
  source: "source",
  rgb8: "rgb8",
  encoderInput: "encoder_input",
  mu: "mu",
  latent: "latent",
} as const;

/** golden の `__metadata__` の欄。 */
export const WAN_I2V_GOLDEN_METADATA = {
  image: "image",
  imageSha256: "image_sha256",
  width: "width",
  height: "height",
  fit: "fit",
  /** encoder の帯での役割（crop は `band` / `accept`・stretch は `host` — recipe の `GoldenCase.role`）。 */
  role: "role",
  pillow: "version_pillow",
} as const;
