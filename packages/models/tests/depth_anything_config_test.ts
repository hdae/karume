// `depth-anything/1` の `pipelineConfig` スキーマ（`src/depth-anything/config.ts`）。**焼く側**は
// `tools/export-recipes/depth_anything/distribution.py` で、そちらの pytest が同じ欄名を別方向から
// 押さえている（欄名が片側だけ動くと、配布形はできるのにロードが parse で落ちる）。
//
// 見るのは 3 点:
//
// ① 受理集合の**外側**を全部落とす（未知キー・欠落・値域外）。既定へ黙って縮退しない。
// ② `imageStd` の 0 を落とす — 0 除算は例外を出さず `±Infinity` の `pixel_values` を作り、
//    グラフは NaN を吐きながら shape だけ合う。
// ③ `interpolation` の受理集合は **`bicubic` の 1 値**（BiRefNet / SigLIP2 の `bilinear` と
//    **逆向き**）。DA-V2 の上流は `resample: 3` = PIL の BICUBIC で、bilinear を宣言した
//    配布形を通すと `pixel_values` が uint8 1 LSB の 34 倍ずれる。受理集合がファミリごとに
//    違うことが、スキーマを共有しない理由そのものなので、両向きに固定する。

import { assertEquals, assertThrows } from "@std/assert";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";

/** `models/karume-depth-anything-v2-small/karume.json` の `pipelineConfig` 実物（5 欄）。 */
const CONFIG: Record<string, unknown> = {
  imageWidth: 518,
  imageHeight: 518,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
};

const withPatch = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...CONFIG,
  ...patch,
});

const without = (key: string): Record<string, unknown> => {
  const copy = { ...CONFIG };
  delete copy[key];
  return copy;
};

Deno.test("配布形の pipelineConfig をそのまま読める（5 欄・値はそのまま）", () => {
  const config = parseDepthAnythingPipelineConfig(CONFIG);
  assertEquals(config.imageWidth, 518);
  assertEquals(config.imageHeight, 518);
  assertEquals(config.imageMean, [0.485, 0.456, 0.406]);
  assertEquals(config.imageStd, [0.229, 0.224, 0.225]);
  assertEquals(config.interpolation, "bicubic");
});

Deno.test("未知キーは落とす（綴り違いが黙って既定へ縮退しない）", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ image_mean: [0.5, 0.5, 0.5] })),
    Error,
    "未知キー 'image_mean'",
  );
});

Deno.test("上流 config の余った欄も落とす（keep_aspect_ratio / ensure_multiple_of）", () => {
  // 上流 `preprocessor_config.json` にはこの 2 欄があるが、焼かれたグラフは正方 1 点しか
  // 受け取らないので宣言に席は無い。素通しで写した配布形はここで止まる。
  for (const key of ["keepAspectRatio", "ensureMultipleOf"]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ [key]: 14 })),
      Error,
      `未知キー '${key}'`,
    );
  }
});

for (const key of Object.keys(CONFIG)) {
  Deno.test(`欄の欠落は落とす: ${key}`, () => {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(without(key)),
      Error,
      `pipelineConfig.${key}`,
    );
  });
}

Deno.test("寸法は正の整数だけ（0 / 小数 / 文字列を落とす）", () => {
  for (const bad of [0, -518, 518.5, "518"]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ imageWidth: bad })),
      Error,
      "imageWidth",
    );
  }
});

Deno.test("imageStd の 0 を落とす（0 除算は ±Infinity を静かに作る）", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ imageStd: [0.229, 0, 0.225] })),
    Error,
    "imageStd",
  );
});

Deno.test("チャネル定数は長さ 3 の数の配列だけ", () => {
  for (const bad of [[0.485, 0.456], [0.485, 0.456, 0.406, 0.5], [0.485, "0.456", 0.406], 0.485]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ imageMean: bad })),
      Error,
      "imageMean",
    );
  }
});

Deno.test("imageMean は負も受ける（平行移動なので値域を狭めない）", () => {
  assertEquals(
    parseDepthAnythingPipelineConfig(withPatch({ imageMean: [-0.5, 0, 0.5] })).imageMean,
    [-0.5, 0, 0.5],
  );
});

Deno.test("非有限な定数は落とす（全画素が NaN になる）", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ imageMean: [0.485, Number.NaN, 0.406] })),
    Error,
    "imageMean",
  );
  assertThrows(
    () =>
      parseDepthAnythingPipelineConfig(
        withPatch({ imageStd: [0.229, Number.POSITIVE_INFINITY, 0.2] }),
      ),
    Error,
    "imageStd",
  );
});

Deno.test("bilinear を宣言した配布形は受理しない（BiRefNet / SigLIP2 と逆向き）", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ interpolation: "bilinear" })),
    Error,
    "'bicubic' だけ",
  );
});
