// A/B の道具の純粋な計算（`analysis.ts`）と素材の表（`corpus.ts`）の振る舞い（GPU 不要・資産不要）。

import { assert, assertAlmostEquals, assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  BAND_RADIUS,
  blindAssignment,
  composite,
  contourBand,
  errorAgainst,
  gradientMagnitude,
  kernelDifference,
  largestDifferenceWindow,
  type Plane,
  pseudoTruth,
  tile,
  zoomNearest,
} from "./analysis.ts";
import { animaCommand, animaFile, CORPUS } from "./corpus.ts";

/** 全画素 `fill` の平面に、`(x, y) → 値` の点を打ったもの。 */
const plane = (
  width: number,
  height: number,
  fill: number,
  points: readonly (readonly [number, number, number])[] = [],
): Plane => {
  const data = new Uint8Array(width * height).fill(fill);
  for (const [x, y, value] of points) data[y * width + x] = value;
  return { data, width, height };
};

const ones = (band: Uint8Array): number => band.reduce((sum, value) => sum + value, 0);

describe("輪郭の帯", () => {
  it("中間の α の画素 1 つは、半径の正方形に膨らむ", () => {
    const soft = plane(11, 11, 0, [[5, 5, 128]]);
    const band = contourBand(soft, plane(11, 11, 0));
    const side = 2 * BAND_RADIUS + 1;
    assertEquals(ones(band), side * side);
    assertEquals(band[(5 - BAND_RADIUS) * 11 + 5 - BAND_RADIUS], 1);
    assertEquals(band[(5 - BAND_RADIUS - 1) * 11 + 5], 0);
  });

  it("0 と 255 だけの平面は帯を持たず、どちらかの手順の中間値で帯ができる", () => {
    const hard = plane(9, 9, 255, [[0, 0, 0]]);
    assertEquals(ones(contourBand(hard, hard)), 0);
    assert(ones(contourBand(hard, plane(9, 9, 255, [[8, 8, 200]]))) > 0);
  });
});

describe("K 列（帯の中の 2 手順の差）", () => {
  it("帯の外の差は数えず、帯の中の差の平均と最大を段で返す", () => {
    const first = plane(5, 1, 0, [[1, 0, 10], [4, 0, 9]]);
    const second = plane(5, 1, 0, [[1, 0, 13], [4, 0, 100]]);
    const band = Uint8Array.of(1, 1, 1, 0, 0);
    assertEquals(kernelDifference(first, second, band), {
      bandPixels: 3,
      differing: 1,
      meanAbs: 1,
      maxAbs: 3,
    });
  });
});

describe("総合列（擬似正解に対する誤差）", () => {
  it("擬似正解と同じ α なら SAD・MSE・Gradient error は 0", () => {
    const alpha = plane(8, 8, 0, [[3, 3, 255], [4, 3, 51], [3, 4, 102]]);
    const truth = pseudoTruth(Float64Array.from(alpha.data, (value) => value / 255), 8, 8);
    const band = new Uint8Array(64).fill(1);
    assertEquals(errorAgainst(alpha, truth, band), { sad: 0, mse: 0, gradient: 0 });
  });

  it("SAD は帯の中の |Δα|（α ∈ [0, 1]）の和で、MSE は帯の画素数で割った二乗の平均", () => {
    const alpha = plane(4, 1, 255);
    const truth = pseudoTruth(Float64Array.of(1, 0.5, 1, 0), 4, 1);
    const error = errorAgainst(alpha, truth, Uint8Array.of(1, 1, 1, 0));
    assertAlmostEquals(error.sad, 0.5, 1e-12);
    assertAlmostEquals(error.mse, 0.25 / 3, 1e-12);
  });

  it("勾配の大きさは一様な平面で 0、斜面では軸を入れ替えても同じ値", () => {
    const flat = gradientMagnitude(new Float64Array(64).fill(0.7), 8, 8);
    assert(flat.every((value) => Math.abs(value) < 1e-12), "一様な平面に勾配が出た");
    const size = 16;
    const alongX = Float64Array.from({ length: size * size }, (_, i) => (i % size) / size);
    const alongY = Float64Array.from(
      { length: size * size },
      (_, i) => Math.floor(i / size) / size,
    );
    const x = gradientMagnitude(alongX, size, size);
    const y = gradientMagnitude(alongY, size, size);
    assert(x[8 * size + 8] > 0, "斜面に勾配が出ない");
    assertAlmostEquals(x[8 * size + 8], y[8 * size + 8], 1e-12);
  });
});

describe("差の大きい窓", () => {
  it("2 手順の差が集まった場所を含む窓を選ぶ", () => {
    const first = plane(40, 30, 0);
    const second = plane(40, 30, 0, [[30, 20, 50], [31, 21, 50], [33, 22, 50]]);
    const window = largestDifferenceWindow(first, second, 8);
    assertEquals([window.width, window.height], [8, 8]);
    for (const [x, y] of [[30, 20], [31, 21], [33, 22]]) {
      assert(
        x >= window.left && x < window.left + 8 && y >= window.top && y < window.top + 8,
        `(${x}, ${y}) が窓 ${JSON.stringify(window)} の外`,
      );
    }
  });

  it("画像が窓より小さい軸は全体を窓にする", () => {
    const window = largestDifferenceWindow(plane(5, 100, 0), plane(5, 100, 1), 64);
    assertEquals([window.left, window.width, window.height], [0, 5, 64]);
  });
});

describe("盲検の割り付け", () => {
  it("同じ seed と対の数なら同じ割り付けを返す（対応表を作り直せる）", () => {
    assertEquals(blindAssignment(8, 20261011), blindAssignment(8, 20261011));
    assertEquals(blindAssignment(8, 20261011).length, 8);
  });

  it("seed を変えると割り付けが変わる", () => {
    assertNotEquals(blindAssignment(32, 1), blindAssignment(32, 2));
  });
});

describe("目視用の合成", () => {
  it("α 255 は画像そのもの、α 0 は地（黒・白・市松）", () => {
    const image = { data: Uint8Array.of(10, 20, 30, 10, 20, 30), width: 2, height: 1 };
    const alpha = plane(2, 1, 255, [[1, 0, 0]]);
    assertEquals([...composite(image, alpha, "black").data], [10, 20, 30, 0, 0, 0]);
    assertEquals([...composite(image, alpha, "white").data], [10, 20, 30, 255, 255, 255]);
    const checker = composite(image, alpha, "checker").data;
    assertEquals(checker[3], checker[4]);
    assert(checker[3] > 0 && checker[3] < 255, "市松の地が黒か白に潰れた");
  });

  it("最近傍の拡大は 1 画素を倍率の正方形にし、並べると隙間を挟む", () => {
    const image = { data: Uint8Array.of(1, 2, 3, 4, 5, 6), width: 2, height: 1 };
    const zoom = zoomNearest(image, { left: 1, top: 0, width: 1, height: 1 }, 3);
    assertEquals([zoom.width, zoom.height], [3, 3]);
    assert(
      Array.from({ length: 9 }, (_, i) => zoom.data.subarray(i * 3, i * 3 + 3).join()).every((
        pixel,
      ) => pixel === "4,5,6"),
    );
    const tiled = tile([[zoom, zoom], [zoom, zoom]], 2);
    assertEquals([tiled.width, tiled.height], [3 * 2 + 2, 3 * 2 + 2]);
  });
});

describe("素材の表（裁定 31 — anima で生成）", () => {
  it("拡大 4 枚は 2048²、縮小 4 枚は 832×512 の上下 16 画素を落とした 832×480", () => {
    const up = CORPUS.filter((entry) => entry.direction === "up");
    const down = CORPUS.filter((entry) => entry.direction === "down");
    assertEquals(up.length, 4);
    assertEquals(down.length, 4);
    assert(up.every((entry) => entry.width === 2048 && entry.height === 2048 && !entry.crop));
    for (const entry of down) {
      assertEquals([entry.width, entry.height], [832, 512]);
      assertEquals(entry.crop, { left: 0, top: 16, width: 832, height: 480 });
    }
  });

  it("名前と生成物のファイル名は重複しない（同じ寸法・同じ seed の上書きを起こさない）", () => {
    assertEquals(new Set(CORPUS.map((entry) => entry.name)).size, CORPUS.length);
    assertEquals(new Set(CORPUS.map(animaFile)).size, CORPUS.length);
  });

  it("生成のコマンドはプロンプトを 1 つの単一引用符で包める（引用符を含まない）", () => {
    for (const entry of CORPUS) {
      assert(!entry.prompt.includes("'"), `${entry.name} のプロンプトに単一引用符がある`);
      assert(animaCommand(entry).includes(`--seed ${entry.seed}`));
    }
  });
});
