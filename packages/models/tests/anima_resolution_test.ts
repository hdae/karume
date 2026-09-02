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
 * 丸め等間隔配置（ADR 0033 追記 2026-09-02）で**本数が旧規則から変わる辺**の実測
 * （P-3 スパイク）。`[辺, 旧本数, 新本数]`。旧規則は本数を `extent − tile` の約数から
 * 選んでいたので、約数の乏しい辺だけが 60 本以上に化けており、10 本超の 8 通り
 * （1456 / 1488 / 1584 / 1648 / 1680 / 1776 / 1840 / 1936）は入口で拒否していた。
 *
 * この表に無い 73 辺は本数も配置も旧規則と同一 = **出荷バイト不変**（1024² / 512² /
 * 1344×768 系の配置そのものは `anima_tiling_test.ts` が literal で凍結する）。
 */
const RECOUNTED_SIDES: readonly (readonly [number, number, number])[] = [
  [1440, 5, 4],
  [1456, 60, 4],
  [1488, 62, 4],
  [1504, 5, 4],
  [1536, 5, 4],
  [1552, 6, 4],
  [1584, 68, 4],
  [1600, 5, 4],
  [1632, 5, 4],
  [1648, 72, 4],
  [1680, 74, 4],
  [1696, 5, 4],
  [1728, 5, 4],
  [1744, 8, 4],
  [1776, 80, 4],
  [1792, 5, 4],
  [1824, 5, 4],
  [1840, 84, 4],
  [1872, 6, 5],
  [1904, 7, 5],
  [1936, 90, 5],
  [1968, 8, 5],
  [2000, 7, 5],
  [2032, 6, 5],
];

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

Deno.test("受理集合: 4 条件をそれぞれ名指しで落とす", () => {
  // ①刻み（16 の倍数）— 8 の倍数でも latent が patch 2 で割り切れない。
  assertThrows(() => parseResolution("1352x768"), Error, `${RESOLUTION_GRANULARITY} の倍数でない`);
  // ②下限（VAE タイル decoder の latent 64 が入らない）。
  assertThrows(() => parseResolution(`1024x${MIN_RESOLUTION_SIDE - 16}`), Error, "下限");
  // ③天井（rope 素表の行数）。
  assertThrows(() => parseResolution(`${MAX_RESOLUTION_SIDE + 16}x512`), Error, "上限");
  // ④S の上限。現行定数では各辺の天井が S の上限を含意するので**超える形は作れない**
  // （2048×2048 = 16,384 がちょうど上限 — 定数関係そのものは下の門が固定する）。
  assertEquals(parseResolution("2048x2048"), { width: 2048, height: 2048 });
});

Deno.test("assertAcceptableResolution: 構造体で渡す経路も同じ 4 条件で見る", () => {
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

Deno.test("受理集合: 刻みの上に並ぶ辺を 1 つも落とさない（タイル本数の門は撤廃）", () => {
  // 旧規則では本数が跳ねる 8 通りを入口で拒否していた（対症の門 — P-3 で根治）。落ちる集合が
  // 空であることを両軸で見るので、拒否がどこかへ復活すればここで割れる。
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
    assertEquals(rejected, [], `${axis} 軸で落ちた辺`);
  }
});

Deno.test("タイル本数: 全 97 辺で 5 本以下（本数が跳ねる辺はもう存在しない）", () => {
  // 入口から本数の門を外せる根拠そのもの。`decodeTiled` は decode 出力を全枚数ぶん抱えるので
  // 本数の上限がホスト RAM の上限になる — 跳ねる辺が復活したら（= 配置規則が退行したら）
  // 拒否ではなく OOM で出る。ここで先に割れるようにしておく。
  const worst = Math.max(...CANDIDATE_SIDES.map(tilesOf));
  assertEquals(worst, 5, `辺あたりの最大タイル本数（両辺最悪でも ${worst * worst} 枚）`);
});

Deno.test("タイル本数: 旧規則から変わる 24 辺の新しい本数を実測で凍結する", () => {
  // 変わらない 73 辺は配置まで旧規則と同一 = 出荷バイト不変。変わる辺は**値で**凍結する
  // （式で書くと `tilesOf` の写しになって門が空虚になる）。
  const recounted = new Map(RECOUNTED_SIDES.map(([side, , after]) => [side, after]));
  for (const side of CANDIDATE_SIDES) {
    const expected = recounted.get(side);
    if (expected !== undefined) assertEquals(tilesOf(side), expected, `${side}px の新本数`);
  }
  // 旧本数が上限 10 本を超えていた 8 通り（= 旧規則が入口で拒否していた辺）が全て復帰した。
  const restored = RECOUNTED_SIDES.filter(([, before]) => before > 10).map(([side]) => side);
  assertEquals(restored, [1456, 1488, 1584, 1648, 1680, 1776, 1840, 1936], "復帰した辺");
  for (const side of restored) assertAcceptableResolution({ width: side, height: 1024 });
});

Deno.test("タイル枚数: 1824×1248 は 4×3 = 12 枚（旧規則は 5×3 = 15 枚）", () => {
  // P-3 の測定対象そのもの。枚数がそのまま VAE decode の実行時間に効くので、外挿の前提を
  // 数で押さえる（実 GPU の A/B は e2e 側）。
  assertAcceptableResolution({ width: 1824, height: 1248 });
  assertEquals(tilesOf(1824), 4, "幅 1824px");
  assertEquals(tilesOf(1248), 3, "高さ 1248px");
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
