// Wan2.2 TI2V の I2V の条件画像の前処理 —— **実画像**編（ADR 0121 段 9a・GPU 不要）。
//
// 合成画像の側（`wan_ti2v_i2v_preprocess_test.ts`）が式と並びと Pillow の LANCZOS を部品ごとに縛るのに対し、こちらは
// テスト画像（Wan2.1 の視認の先頭フレーム 3 枚・832×480）の実画素を、recipe が上流の Python（Pillow の LANCZOS・公式の
// 中央クロップ / diffusers の直接の伸縮・diffusers の `patchify`・I2V パイプラインの正規化）で採った golden と
// **ビット一致**で比べる。帯は置かない（どの段も整数か f32 の 1 回の丸めで、上流と同じ値になる — 割れれば写し間違い）。
//
// 比べるのは 3 か所: 寸法合わせの RGB8（`rgb8` — LANCZOS と切り出しの位置）、encoder の入力（`encoder_input` —
// `[-1, 1]` と patchify の並び）、encoder の出口の正規化（`mu` → `latent`）。寸法は golden のもの（明示）で、受理集合の
// 外の縮小の経路（256×160）も通す — 受理集合の門は `selectWanI2vSize` の側で、ここが見るのは前処理の値。そのため鎖は
// 部品の合成（`fitWanI2vImage` → `wanI2vPixels` → `patchifyWanImage`）で回す（`preprocessWanI2vImage` が同じ合成で
// あることはホストテストが縛る）。
//
// ## 資産が無い環境
//
// golden（`outputs/series/` — encoder の 3 グラフと一組で書かれる）は git 追跡外。**1 本も無ければ明示 SKIP**、
// **一部だけなら FAIL**（採り直しの途中で落ちた資産を SKIP に丸めない）。置き場と形の正本は `helpers/wan-i2v-image.ts`。

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import { fitWanI2vImage, patchifyWanImage, wanI2vPixels } from "../src/wan/i2v-preprocess.ts";
import { normalizeWanLatents } from "../src/wan/latents.ts";
import {
  WAN_I2V_GOLDEN_CASES,
  WAN_I2V_GOLDEN_GENERATE,
  WAN_I2V_GOLDEN_KEYS,
  WAN_I2V_GOLDEN_METADATA,
  WAN_I2V_IMAGE_SHA256,
  wanI2vGoldenName,
  wanI2vGoldenUrl,
} from "./helpers/wan-i2v-image.ts";
import { filePresent, firstBitMismatch, readBuffer, viewOf } from "./helpers/wan-ti2v-dit.ts";

/** 登録時点で必要なので同期で数える（`ignore` の判定に使う）。 */
const present = WAN_I2V_GOLDEN_CASES.filter((testCase) => filePresent(wanI2vGoldenUrl(testCase)))
  .length;

if (present === 0) {
  console.warn(
    `[karume] I2V の実画像の前処理の照合を SKIP する（golden 0/${WAN_I2V_GOLDEN_CASES.length} 本）。` +
      `golden の生成: ${WAN_I2V_GOLDEN_GENERATE}`,
  );
}

const metadataOf = (file: SafetensorsFile, key: string, where: string): string => {
  const value = file.metadata.get(key);
  if (value === undefined) throw new Error(`${where}: __metadata__.${key} が無い`);
  return value;
};

/** U8 のテンソルを形の検査つきで読む。 */
const bytesOf = (
  file: SafetensorsFile,
  key: string,
  shape: readonly number[],
  where: string,
): Uint8Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, "U8", `${where}: ${key} の dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} の形`);
  return new Uint8Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

/**
 * F32 のテンソルを形の検査つきで読む。写してから読む（U8 のテンソルが前に並ぶと、F32 の開始位置が 4 バイト境界に
 * 載る保証が無い）。
 */
const floatsOf = (
  file: SafetensorsFile,
  key: string,
  shape: readonly number[],
  where: string,
): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  assertEquals(view.dtype, "F32", `${where}: ${key} の dtype`);
  assertEquals([...view.shape], [...shape], `${where}: ${key} の形`);
  return new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

describe("I2V の実画像の前処理と encoder の出口の正規化 × 上流の golden（ビット一致）", () => {
  it({
    name:
      `golden が ${WAN_I2V_GOLDEN_CASES.length} 本揃っている（1 本も無ければ SKIP・一部だけなら FAIL）`,
    ignore: present === 0,
    fn: () => {
      const missing = WAN_I2V_GOLDEN_CASES.map(wanI2vGoldenUrl)
        .filter((url) => !filePresent(url))
        .map((url) => url.pathname);
      assertEquals(missing, [], `golden が欠けている（採り直す: ${WAN_I2V_GOLDEN_GENERATE}）`);
    },
  });

  for (const testCase of WAN_I2V_GOLDEN_CASES) {
    const where = wanI2vGoldenName(testCase);
    it({
      name: where,
      ignore: present === 0,
      fn: async () => {
        const url = wanI2vGoldenUrl(testCase);
        assert(filePresent(url), `${where}: golden ${url.pathname} が無い`);
        const file = parseSafetensors(await readBuffer(url));
        const meta = WAN_I2V_GOLDEN_METADATA;
        const { width, height, fit } = testCase;
        assertEquals(metadataOf(file, meta.image, where), testCase.image, `${where}: 画像`);
        assertEquals(
          metadataOf(file, meta.imageSha256, where),
          WAN_I2V_IMAGE_SHA256[testCase.image],
          `${where}: 画像の sha256 が表と違う`,
        );
        assertEquals(metadataOf(file, meta.fit, where), fit, `${where}: fit`);
        assertEquals(
          [metadataOf(file, meta.width, where), metadataOf(file, meta.height, where)],
          [String(width), String(height)],
          `${where}: 出力寸法`,
        );
        assert(metadataOf(file, meta.pillow, where).length > 0, `${where}: Pillow の版が空`);

        // 寸法合わせ（Pillow の RGB8）。
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
          `${where}: RGB8 が Pillow と割れる（最初は ${firstRgbMismatch}: ${
            fitted.data[firstRgbMismatch]
          } 対 ${expectedRgb[firstRgbMismatch]}）`,
        );
        if (fit !== "crop") return; // stretch の golden は前処理の RGB8 だけを持つ（encoder を回さない）。

        // encoder の入力（`[-1, 1]` → patchify）。
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
          `${where}: encoder の入力が上流と割れる（最初は ${firstInputMismatch}: ${
            input[firstInputMismatch]
          } 対 ${expectedInput[firstInputMismatch]}）`,
        );

        // encoder の出口の正規化（上流の mu → 条件の潜在）。
        // 空間の圧縮 16 = encoder のグラフの比 8 × patchify 2（`wanSpatialCompression` と同じ値）。
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
          `${where}: 正規化が上流と割れる（最初は ${firstLatentMismatch}: ${
            latent[firstLatentMismatch]
          } 対 ${expectedLatent[firstLatentMismatch]}）`,
        );
      },
    });
  }
});
