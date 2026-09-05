// `birefnet/1` の `pipelineConfig` スキーマ（`src/birefnet/config.ts`）。**焼く側**は
// `tools/export-recipes/birefnet/distribution.py` で、そちらの pytest が同じ欄名を別方向から
// 押さえている（欄名が片側だけ動くと、配布形はできるのにロードが parse で落ちる）。
//
// 見るのは 3 点:
//
// ① 受理集合の**外側**を全部落とす（未知キー・欠落・値域外）。既定へ黙って縮退しない。
// ② `imageStd` の 0 を落とす — 0 除算は例外を出さず `±Infinity` の `pixel_values` を作り、
//    グラフは NaN を吐きながら shape だけ合う。
// ③ `interpolation` は分岐ではなく**宣言**で、受理集合は 1 値だけ（上流 `handler.py` の
//    torchvision Resize が既定の補間 = bilinear で通すから — 前処理層は bicubic も持っている）。

import { assertEquals, assertThrows } from "@std/assert";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";

/** `models/karume-birefnet-hr/karume.json` の `pipelineConfig` 実物（5 欄）。 */
const CONFIG: Record<string, unknown> = {
  imageWidth: 1024,
  imageHeight: 1024,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
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
  const config = parseBirefnetPipelineConfig(CONFIG);
  assertEquals(config.imageWidth, 1024);
  assertEquals(config.imageHeight, 1024);
  assertEquals(config.imageMean, [0.485, 0.456, 0.406]);
  assertEquals(config.imageStd, [0.229, 0.224, 0.225]);
  assertEquals(config.interpolation, "bilinear");
});

Deno.test("未知キーは落とす（綴り違いが黙って既定へ縮退しない）", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ image_mean: [0.5, 0.5, 0.5] })),
    Error,
    "未知キー 'image_mean'",
  );
});

for (const key of Object.keys(CONFIG)) {
  Deno.test(`欄の欠落は落とす: ${key}`, () => {
    assertThrows(() => parseBirefnetPipelineConfig(without(key)), Error, `pipelineConfig.${key}`);
  });
}

Deno.test("寸法は正の整数だけ（0 / 小数 / 文字列を落とす）", () => {
  for (const bad of [0, -1024, 1024.5, "1024"]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ imageWidth: bad })),
      Error,
      "imageWidth",
    );
  }
});

Deno.test("imageStd の 0 を落とす（0 除算は ±Infinity を静かに作る）", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ imageStd: [0.229, 0, 0.225] })),
    Error,
    "imageStd",
  );
});

Deno.test("チャネル定数は長さ 3 の数の配列だけ", () => {
  for (const bad of [[0.485, 0.456], [0.485, 0.456, 0.406, 0.5], [0.485, "0.456", 0.406], 0.485]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ imageMean: bad })),
      Error,
      "imageMean",
    );
  }
});

Deno.test("imageMean は負も受ける（平行移動なので値域を狭めない）", () => {
  assertEquals(parseBirefnetPipelineConfig(withPatch({ imageMean: [-0.5, 0, 0.5] })).imageMean, [
    -0.5,
    0,
    0.5,
  ]);
});

Deno.test("非有限な定数は落とす（全画素が NaN になる）", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ imageMean: [0.485, Number.NaN, 0.406] })),
    Error,
    "imageMean",
  );
  assertThrows(
    () =>
      parseBirefnetPipelineConfig(withPatch({ imageStd: [0.229, Number.POSITIVE_INFINITY, 0.2] })),
    Error,
    "imageStd",
  );
});

Deno.test("対応していない補間は受理しない（黙って bilinear で通さない）", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ interpolation: "bicubic" })),
    Error,
    "'bilinear' だけ",
  );
});
