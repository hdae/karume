// `birefnet/2` の `pipelineConfig` スキーマ（`src/birefnet/config.ts`）。**焼く側**は
// `tools/export-recipes/birefnet/distribution.py` で、そちらの pytest が同じ欄名を別方向から
// 押さえている（欄名が片側だけ動くと、配布形はできるのにロードが parse で落ちる）。
//
// 見るのは 4 点:
//
// ① 受理集合の**外側**を全部落とす（未知キー・欠落・値域外）。既定へ黙って縮退しない。
// ② `imageStd` の 0 を落とす — 0 除算は例外を出さず `±Infinity` の `pixel_values` を作り、
//    グラフは NaN を吐きながら shape だけ合う。
// ③ `interpolation` は分岐ではなく**宣言**で、受理集合は 1 値だけ（上流 `handler.py` の
//    torchvision Resize が既定の補間 = bilinear で通すから — 前処理層は bicubic も持っている）。
// ④ `postprocess` は閉じた集合（`MATTE_PROCEDURES`）の宣言で、必須・既定なし（ADR 0125）。
//    集合の外は綴り違いも Object.prototype の綴りも落とす。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";
import { MATTE_PROCEDURES } from "../src/birefnet/matte.ts";

/** `models/karume-birefnet-hr/karume.json` の `pipelineConfig`（`birefnet/2` で組み直した形・6 欄）。 */
const CONFIG: Record<string, unknown> = {
  imageWidth: 1024,
  imageHeight: 1024,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
  postprocess: "u8-pil-bicubic",
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

Deno.test("配布形の pipelineConfig をそのまま読める（6 欄・値はそのまま）", () => {
  const config = parseBirefnetPipelineConfig(CONFIG);
  assertEquals(config.imageWidth, 1024);
  assertEquals(config.imageHeight, 1024);
  assertEquals(config.imageMean, [0.485, 0.456, 0.406]);
  assertEquals(config.imageStd, [0.229, 0.224, 0.225]);
  assertEquals(config.interpolation, "bilinear");
  assertEquals(config.postprocess, "u8-pil-bicubic");
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

Deno.test("imageMean / imageStd の組で正規化後の値が f32 で溢れる配布形は資産の不正として落とす", () => {
  // 個々の欄は有限でも商が f32 で溢れる。構築時に落とさないと、生成のたびに入力起因の型で落ちる。
  const error = assertThrows(
    () => parseBirefnetPipelineConfig({ ...CONFIG, imageStd: [1e-50, 0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean / imageStd: チャネル 0 の正規化後の値域",
  );
  assert(!(error instanceof ModelInputError), "資産の不正が入力起因に分類された");
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

Deno.test("後処理の手順は閉じた集合の全てを宣言どおりに受ける", () => {
  // 両系列の配布形（HR = u8-pil-bicubic・Lucida = float-bilinear）が同じ読み手で読める。
  for (const procedure of MATTE_PROCEDURES) {
    assertEquals(
      parseBirefnetPipelineConfig(withPatch({ postprocess: procedure })).postprocess,
      procedure,
    );
  }
});

Deno.test("集合の外の後処理の手順は落とす（綴り違い・大小文字・Object.prototype の綴り・文字列でない値）", () => {
  // 黙って片方の系列の手順へ縮退すると、境界の α が上流と違う手順で出たまま気づけない。
  for (const bad of ["bicubic", "U8-PIL-BICUBIC", "float_bilinear", "", "toString", 1, null]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ postprocess: bad })),
      Error,
      "pipelineConfig.postprocess: 未知の後処理の手順",
    );
  }
});
