// 解像度の綴りと受理集合。
//
// **受理集合の破れは実行時の深い場所でしか落ちない**（rope 素表の行数超過 / タイル配置の
// 不成立 / Dim("S") の上限）ので、入口で条件を名指しできていることをここで固定する。
// 資産も GPU も要らない純関数。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ANIMA_SPATIAL_COMPRESSION } from "../src/anima/dit-tokens.ts";
import {
  assertAcceptableResolution,
  formatResolution,
  MAX_DIT_TOKENS,
  MAX_RESOLUTION_SIDE,
  MAX_TILES_PER_AXIS,
  MIN_RESOLUTION_SIDE,
  parseResolution,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";
import { MIN_TILE_OVERLAP_LATENT, planTileAxis } from "../src/anima/tiling.ts";

/** 刻みの上に並ぶ辺の候補（下限から天井まで）。 */
const CANDIDATE_SIDES = ((): readonly number[] => {
  const sides: number[] = [];
  for (
    let side = MIN_RESOLUTION_SIDE;
    side <= MAX_RESOLUTION_SIDE;
    side += RESOLUTION_GRANULARITY
  ) sides.push(side);
  return sides;
})();

/**
 * VAE タイルの本数が跳ねる辺（実測 — `planTileAxis` は「重なりを満たす最小本数」を
 * `extent − tile` の**約数**から選ぶので、約数の乏しい辺だけが 60 本以上に化ける）。
 * この一覧は受理集合の実測そのものなので、判定式ではなく**値**で置く。
 */
const EXPLODING_SIDES: readonly number[] = [1456, 1488, 1584, 1648, 1680, 1776, 1840, 1936];

/** その辺を VAE タイルで覆う本数（分割計画そのものを回して数える）。 */
const tilesOf = (side: number): number =>
  planTileAxis(
    side / ANIMA_SPATIAL_COMPRESSION,
    MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION,
    MIN_TILE_OVERLAP_LATENT,
  ).starts.length;

Deno.test("parseResolution: WxH と正方の略記を受ける（往復は恒等）", () => {
  assertEquals(parseResolution("1344x768"), { width: 1344, height: 768 });
  assertEquals(parseResolution("768x1344"), { width: 768, height: 1344 });
  assertEquals(parseResolution("512"), { width: 512, height: 512 });
  assertEquals(parseResolution("1024X1024"), { width: 1024, height: 1024 });
  for (const raw of ["512", "1024", "1344x768", "896x1152"]) {
    assertEquals(formatResolution(parseResolution(raw)), raw, `往復: ${raw}`);
  }
});

Deno.test("parseResolution: 綴りでない入力を落とす（既定へ縮退させない）", () => {
  for (const raw of ["", "1344*768", "1344x", "x768", "1344x768x1", "-512", "51 2", "1e3"]) {
    assertThrows(() => parseResolution(raw), Error, "が WxH でも", `綴り: ${JSON.stringify(raw)}`);
  }
});

Deno.test("受理集合: 5 条件をそれぞれ名指しで落とす", () => {
  // ①刻み（16 の倍数）— 8 の倍数でも latent が patch 2 で割り切れない。
  assertThrows(() => parseResolution("1352x768"), Error, `${RESOLUTION_GRANULARITY} の倍数でない`);
  // ②下限（VAE タイル decoder の latent 64 が入らない）。
  assertThrows(() => parseResolution(`1024x${MIN_RESOLUTION_SIDE - 16}`), Error, "下限");
  // ③天井（rope 素表の行数）。
  assertThrows(() => parseResolution(`${MAX_RESOLUTION_SIDE + 16}x512`), Error, "上限");
  // ④S の上限。現行定数では各辺の天井が S の上限を含意するので**超える形は作れない**
  // （2048×2048 = 16,384 がちょうど上限 — 定数関係そのものは下の門が固定する）。
  assertEquals(parseResolution("2048x2048"), { width: 2048, height: 2048 });
  // ⑤VAE タイルの本数（内訳は下の 3 本）。
  assertThrows(() => parseResolution("1456x1024"), Error, "VAE タイルが");
});

Deno.test("assertAcceptableResolution: 構造体で渡す経路も同じ 5 条件で見る", () => {
  // manifest の `pipelineConfig.defaults.resolution` と `generate` の request はここを通る
  // （綴りを経由しないので、綴り側だけに検査があると素通しになる）。
  assertAcceptableResolution({ width: 1024, height: 1024 });
  assertThrows(() => assertAcceptableResolution({ width: 1024, height: 1000 }), Error, "倍数");
  assertThrows(() => assertAcceptableResolution({ width: 256, height: 1024 }), Error, "下限");
  assertThrows(() => assertAcceptableResolution({ width: 4096, height: 512 }), Error, "上限");
  assertThrows(() => assertAcceptableResolution({ width: 1024.5, height: 1024 }), Error, "整数");
});

Deno.test("受理集合: 4 パターン（16:9 / 3:4 の縦横）が全て S=4,032 で受理される", () => {
  // **縦横の対を両方**受けることと、S が Dim("S") の内側であることを、literal を置かずに
  // 数で押さえる。
  const patterns = ["1344x768", "768x1344", "1152x896", "896x1152"];
  for (const raw of patterns) {
    const size = parseResolution(raw);
    const tokens = (size.width / RESOLUTION_GRANULARITY) * (size.height / RESOLUTION_GRANULARITY);
    assertEquals(tokens, 4032, `${raw} のトークン長 S`);
    assert(tokens <= MAX_DIT_TOKENS, `${raw}: S が Dim("S") の上限を超えている`);
  }
});

Deno.test("タイル本数: 跳ねる 8 通りだけを落とし、他の辺は 1 つも落とさない", () => {
  // 本数の条件を足したことで**既存の受理値が巻き添えで落ちない**ことが要点。落ちる集合を
  // 判定式ではなく実測の値で固定するので、閾や `planTileAxis` を触れば赤くなる。
  for (const axis of ["width", "height"] as const) {
    const rejected = CANDIDATE_SIDES.filter((side) => {
      try {
        assertAcceptableResolution(
          axis === "width" ? { width: side, height: 1024 } : { width: 1024, height: side },
        );
        return false;
      } catch {
        return true;
      }
    });
    assertEquals(rejected, [...EXPLODING_SIDES], `${axis} 軸で落ちた辺`);
  }
});

Deno.test("タイル本数: 本数を名指しし、近傍の受理値を添えて落ちる", () => {
  // 「1 つ下の刻み」を機械的に案内すると、また跳ねる形を勧めうる（跳ねる辺は飛び飛びに現れる）。
  // 実際に通る値が診断に載ることを、両側の近傍で見る。
  for (
    const [raw, tiles, nearby] of [
      ["1456x1024", 60, "1440 / 1472"],
      ["1024x1680", 74, "1664 / 1696"],
      ["1936x1024", 90, "1920 / 1952"],
    ] as const
  ) {
    const error = assertThrows(
      () => parseResolution(raw),
      Error,
      `VAE タイルが ${tiles} 本`,
      `${raw} の本数`,
    );
    assert(
      error.message.includes(`近傍の受理値: ${nearby}`),
      `${raw} の診断に近傍の受理値が無い: ${error.message}`,
    );
    // 案内された値は本当に受理される（診断が行き止まりを勧めていない）。
    for (const side of nearby.split(" / ")) {
      assertAcceptableResolution({ width: Number(side), height: Number(side) });
    }
  }
});

Deno.test("MAX_TILES_PER_AXIS: 閾が実測の谷間（受理側の最大と拒否側の最小の間）にある", () => {
  // 上限の置き所そのものの門。受理側が 8 本・拒否側が 60 本で大きく離れているという実測が
  // 閾 10 の根拠なので、その谷間が消えたら（= 分割計画が変わったら）ここで気づく。
  const accepted = CANDIDATE_SIDES.filter((side) => !EXPLODING_SIDES.includes(side));
  const worstAccepted = Math.max(...accepted.map(tilesOf));
  const bestRejected = Math.min(...EXPLODING_SIDES.map(tilesOf));
  assertEquals(worstAccepted, 8, "受理側の最大本数");
  assertEquals(bestRejected, 60, "拒否側の最小本数");
  assert(
    worstAccepted <= MAX_TILES_PER_AXIS && MAX_TILES_PER_AXIS < bestRejected,
    `閾 ${MAX_TILES_PER_AXIS} が谷間 (${worstAccepted}, ${bestRejected}) の外にある`,
  );
});

Deno.test('定数関係: 各辺の天井を使い切っても S は Dim("S") の上限に収まる', () => {
  // `resolution.ts` の S 上限分岐は現行定数では**到達しない**（各辺 2048 = 格子 128 →
  // S ≤ 128² = 16,384）。分岐は定数を動かしたときの保険として残してあるので、「今は到達しない」
  // ことをここで固定する — 辺の天井 / 刻み / S の上限のどれかを動かした瞬間に赤くなる。
  assert(
    (MAX_RESOLUTION_SIDE / RESOLUTION_GRANULARITY) ** 2 <= MAX_DIT_TOKENS,
    `辺の天井から出る S の最大 ${(MAX_RESOLUTION_SIDE / RESOLUTION_GRANULARITY) ** 2} が ` +
      `${MAX_DIT_TOKENS} を超えている（S 上限分岐が到達可能になった）`,
  );
});
