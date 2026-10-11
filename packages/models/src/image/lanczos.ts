/**
 * RGB8 の LANCZOS リサンプル — Pillow の `Image.resize(size, Image.LANCZOS)` の 8bit 経路
 * （`libImaging/Resample.c` の `ImagingResampleInner` と `*_8bpc`）の逐語の移植。本体は Pillow の 8bit の
 * 分離型リサンプルの共通の核（`pillow-resample.ts` — フィルタ `PILLOW_LANCZOS`・3 チャネル）で、
 * ここは RGB8 の入口の検査と結線だけを持つ（Pillow の逐語の写しを 1 本に寄せる — ADR 0125）。
 *
 * Wan2.2 の I2V の条件画像は、公式実装（`textimage2video.py`）も diffusers（`VaeImageProcessor` の
 * `resample="lanczos"`）も Pillow の LANCZOS で作る。その uint8 をビット単位で再現するために置く
 * （ADR 0121 決定 11）。逐語に写している点・`preprocess.ts` の `resizeRgb8` と経路を分けている理由・
 * 窓だけを計算する理由の正本は `pillow-resample.ts` のモジュール doc。版は fixture
 * `tests/fixtures/wan-i2v/lanczos.json`（画素は `lanczos.safetensors`）が記録する。
 *
 * NOTE: `export` は Wan の I2V の前処理とテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */

import { assertRgb8, type Rgb8Image } from "./preprocess.ts";
import { PILLOW_LANCZOS, type PixelWindow, resampleU8Window } from "./pillow-resample.ts";

/** チャネル数（RGB）。 */
const CHANNELS = 3;

/** 伸縮後の画像から切り出す矩形（左上 `left`, `top` から `width × height`）。 */
export type Rgb8Window = PixelWindow;

/**
 * RGB8 を `width × height` へ LANCZOS でリサンプルし（Pillow の `Image.resize(..., Image.LANCZOS)` と uint8 で一致 —
 * モジュール doc）、その矩形 `window` を切り出したものを返す。計算するのは窓の出力だけ（全体を伸縮してから切り出した
 * 値と同じ）。縦横比は保たない。寸法が変わらない軸はパスを飛ばし（その軸は入力の写しを切り出す）、両軸とも変わらなければ
 * 写しを返す（Pillow の `ImagingCopy`）。
 *
 * 画像の検査は `assertRgb8`（呼び手が渡した画像 — `ModelInputError`）。出力寸法と窓は内部の事前条件（呼び手は受理
 * 集合とそこから導いた寸法しか渡さない）なので `RangeError`（共通の核が投げる）。
 */
export const resizeRgb8LanczosWindow = (
  image: Rgb8Image,
  width: number,
  height: number,
  window: Rgb8Window,
): Rgb8Image => {
  assertRgb8(image);
  const resized = resampleU8Window(
    { data: image.data, width: image.width, height: image.height, channels: CHANNELS },
    width,
    height,
    window,
    PILLOW_LANCZOS,
  );
  return { data: resized.data, width: resized.width, height: resized.height };
};

/**
 * RGB8 を `width × height` へ LANCZOS でリサンプルする（{@link resizeRgb8LanczosWindow} の窓 = 全体）。
 */
export const resizeRgb8Lanczos = (image: Rgb8Image, width: number, height: number): Rgb8Image =>
  resizeRgb8LanczosWindow(image, width, height, { left: 0, top: 0, width, height });
