// latent / 条件テンソルの整形と RGBA 化の挙動テスト。GPU も資産も要らない純関数。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
  ANIMA_LATENTS_MEAN,
  ANIMA_LATENTS_STD,
  denormalizeLatents,
  padSequence,
} from "../src/anima/latents.ts";
import { imageToRgba } from "../src/anima/image.ts";

Deno.test("denormalizeLatents: B>1 と channel 数の食い違いを落とす", () => {
  const latents = new Float32Array(2 * 2 * 1 * 1);
  const mean = Float32Array.from([0, 0]);
  const std = Float32Array.from([1, 1]);
  assertThrows(() => denormalizeLatents(latents, [2, 2, 1, 1], mean, std), Error, "batch=1 前提");
  assertThrows(
    () => denormalizeLatents(new Float32Array(3), [1, 3, 1, 1], mean, std),
    Error,
    "チャネル数",
  );
  assertThrows(
    () => denormalizeLatents(new Float32Array(5), [1, 2, 1, 1], mean, std),
    Error,
    "要素数",
  );
});

Deno.test("denormalizeLatents: std の逆数で割る（掛け算に直すと最終桁が変わる）", () => {
  const latents = Float32Array.from([1, 1]);
  const mean = Float32Array.from([0, 0]);
  const std = Float32Array.from([3, 7]);
  const got = denormalizeLatents(latents, [1, 2, 1, 1], mean, std);
  for (const [index, scale] of [3, 7].entries()) {
    assertEquals(got[index], Math.fround(1 / Math.fround(1 / scale)), `channel ${index}`);
  }
  // 掛け算版と**実際に割れる**ことを見せる（見せられないなら MUST に意味が無い）。
  assert(
    got[0] !== Math.fround(1 * 3) || got[1] !== Math.fround(1 * 7),
    "逆数で割る形と掛ける形が全チャネルで一致してしまい、この不変条件を固定できていない",
  );
});

Deno.test("デモ定数: latents_mean / latents_std は 16 チャネル対で揃っている", () => {
  // 値そのものの正しさ（参照フィクスチャとのビット一致）は実資産の波で戻す。ここは
  // 「片方だけ差し替えた」形の取りこぼしを構造で止める。
  assertEquals(ANIMA_LATENTS_MEAN.length, 16);
  assertEquals(ANIMA_LATENTS_STD.length, ANIMA_LATENTS_MEAN.length);
  assert(ANIMA_LATENTS_STD.every((value) => value > 0), "std に非正の値がある");
});

Deno.test("padSequence: 余白はゼロ・B>1 と行数超過は落とす", () => {
  const hidden = { dtype: "f32", shape: [1, 2, 3], data: Float32Array.from([1, 2, 3, 4, 5, 6]) };
  const padded = padSequence(hidden as never, 4);
  assertEquals([...padded], [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]);
  assertThrows(() => padSequence(hidden as never, 1), Error, "上限");
  assertThrows(
    () => padSequence({ dtype: "f32", shape: [2, 1, 3], data: new Float32Array(6) } as never, 4),
    Error,
    "batch=1 前提",
  );
  // i32 / bool は `set` が黙って数値変換する（shape も長さも合ったまま値だけ別物）。
  assertThrows(
    () => padSequence({ dtype: "i32", shape: [1, 2, 3], data: new Int32Array(6) } as never, 4),
    Error,
    "f32 前提",
  );
});

Deno.test("imageToRgba: [-1,1] → [0,255]・planar から interleave・アルファ 255", () => {
  // 1×2 の画像。R 面 / G 面 / B 面がこの順に並ぶ。
  const image = Float32Array.from([
    -1,
    1, // R
    0,
    -2, // G（-2 は clamp 対象）
    2,
    0.5, // B（2 は clamp 対象）
  ]);
  const rgba = imageToRgba(image, 2, 1);
  assertEquals([...rgba], [0, 128, 255, 255, 255, 0, 191, 255]);
});

Deno.test("imageToRgba: 要素数とサイズの食い違いを落とす", () => {
  assertThrows(() => imageToRgba(new Float32Array(5), 2, 1), Error, "要素数");
  assertThrows(() => imageToRgba(new Float32Array(6), 0, 1), RangeError);
});

Deno.test("imageToRgba: 非有限値は座標付きで落とす（黒画素へ沈黙変換しない）", () => {
  // 2×2 の 3 面。土台は全て有限（clamp 対象の範囲外値も混ぜる）で、1 要素だけ壊す。
  const base = new Float32Array(2 * 2 * 3);
  base.set([-2, -1, 0, 1], 0); // R
  base.set([1, 0, -1, -2], 4); // G
  base.set([0, 1, -2, -1], 8); // B
  const spoil = (offset: number, value: number): Float32Array<ArrayBuffer> => {
    const image = base.slice();
    image[offset] = value;
    return image;
  };
  // 土台は落ちない — 検査するのは「有限か」であって「範囲内か」ではない。
  assertEquals([...imageToRgba(base, 2, 2).slice(0, 4)], [0, 255, 128, 255]);

  // 壊す位置を面ごとにずらし、座標が固定文言でなく実際の画素を指すことを見る。
  const nan = assertThrows(() => imageToRgba(spoil(3, NaN), 2, 2), Error);
  assertStringIncludes(nan.message, "x=1");
  assertStringIncludes(nan.message, "y=1");
  assertStringIncludes(nan.message, "channel 0");

  const positiveInfinity = assertThrows(() => imageToRgba(spoil(5, Infinity), 2, 2), Error);
  assertStringIncludes(positiveInfinity.message, "x=1");
  assertStringIncludes(positiveInfinity.message, "y=0");
  assertStringIncludes(positiveInfinity.message, "channel 1");

  const negativeInfinity = assertThrows(() => imageToRgba(spoil(10, -Infinity), 2, 2), Error);
  assertStringIncludes(negativeInfinity.message, "x=0");
  assertStringIncludes(negativeInfinity.message, "y=1");
  assertStringIncludes(negativeInfinity.message, "channel 2");
});

Deno.test("imageToRgba: 非正方（H≠W）でも行優先で走る", () => {
  // 2×3（幅 2 / 高さ 3）。転置した実装だと同じ長さのまま画素の並びだけが変わるので、
  // **面ごとに違う値**を配って位置で捕まえる（正方では検出できない取り違え）。
  const plane = Float32Array.from([-1, 1, 1, -1, -1, 1]);
  const image = new Float32Array(18);
  image.set(plane, 0); // R
  image.set(plane, 6); // G
  image.set(plane, 12); // B
  const rgba = imageToRgba(image, 2, 3);
  assertEquals(rgba.length, 2 * 3 * 4);
  // 画素 (x=1, y=0) は plane[1] = 1 → 255、画素 (x=0, y=1) は plane[2] = 1 → 255。
  assertEquals([...rgba.slice(4, 8)], [255, 255, 255, 255]);
  assertEquals([...rgba.slice(8, 12)], [255, 255, 255, 255]);
  // 画素 (x=0, y=0) は plane[0] = -1 → 0。
  assertEquals([...rgba.slice(0, 4)], [0, 0, 0, 255]);
});
