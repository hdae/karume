// `pipelineConfig` を読む基本 reader（`src/config/readers.ts`）。family 横断の下位層なので、
// ここが受理集合を勝手に広げたり狭めたりすると 8 family の parse が一斉にずれる。
//
// 見るのは 3 点:
//
// ① 受理集合の**外側**（未知キー・欠落・型違い・条件外）を全部落とす。既定へ黙って縮退しない。
// ② エラー文言に呼び手が渡した文言（`where` / `key` / `requirement` / `why`）が入る — 各 family
//    のテストは文言を逐語 assert しており、どの欄でどう外れたかは family 側の責任で決まる。
// ③ 欄の有無は `Object.hasOwn` で見る（横断不変条件）— 継承した `toString` などを「有る」と
//    読むと、未宣言の欄が prototype の値で黙って通る。

import { assertEquals, assertThrows } from "@std/assert";
import { assertAllowedKeys, readChannels, readNumber, readOnly } from "../src/config/readers.ts";

const isPositive = (value: number): boolean => Number.isFinite(value) && value > 0;

Deno.test("assertAllowedKeys: 許可集合の内側だけなら通る（空オブジェクトも通る）", () => {
  assertAllowedKeys({ a: 1, b: 2 }, ["a", "b", "c"], "here");
  assertAllowedKeys({}, ["a"], "here");
});

Deno.test("assertAllowedKeys: 未知キーは落とし、文言に label と許可一覧が入る", () => {
  assertThrows(
    () => assertAllowedKeys({ a: 1, image_mean: 2 }, ["a", "b"], "pipelineConfig"),
    Error,
    "pipelineConfig: 未知キー 'image_mean'（許可: a / b）",
  );
});

Deno.test("readNumber: 条件を満たす数はそのまま返す", () => {
  assertEquals(readNumber({ n: 1024 }, "n", "here", isPositive, "正でない"), 1024);
});

Deno.test("readNumber: 欄が無ければ落とし、文言に label.key が入る", () => {
  assertThrows(
    () => readNumber({}, "imageWidth", "pipelineConfig", isPositive, "正でない"),
    Error,
    "pipelineConfig.imageWidth: 無い",
  );
});

Deno.test("readNumber: 欄は Object.hasOwn で見る（継承した prototype の値を拾わない）", () => {
  assertThrows(
    () => readNumber({} as Record<string, unknown>, "toString", "here", isPositive, "正でない"),
    Error,
    "here.toString: 無い",
  );
});

Deno.test("readNumber: 値が undefined の欄は『無い』ではなく型違いとして落とす", () => {
  // `Object.hasOwn` は値を見ないので、明示的に `undefined` を置いた欄は「有る」側に落ちる。
  assertThrows(
    () => readNumber({ n: undefined }, "n", "here", isPositive, "正でない"),
    Error,
    "here.n: 正でない（undefined）",
  );
});

Deno.test("readNumber: 数でない値と条件外の数を同じ文言で落とす", () => {
  for (const bad of [0, -1, Number.NaN, "1024", null, [1]]) {
    assertThrows(
      () => readNumber({ n: bad }, "n", "here", isPositive, "正の有限数でない"),
      Error,
      "here.n: 正の有限数でない",
    );
  }
});

Deno.test("readChannels: 長さ 3 の数配列を受け、入力とは別の 3 要素として返す", () => {
  // 生の配列をそのまま返すと、manifest 由来の可変配列が config に居座る（後から書き換わる）。
  const source = [0.485, 0.456, 0.406];
  const channels = readChannels(
    { c: source },
    "c",
    "here",
    Number.isFinite,
    "有限でない要素がある",
  );
  assertEquals(channels, [0.485, 0.456, 0.406]);
  source[0] = 9;
  assertEquals(channels, [0.485, 0.456, 0.406]);
});

Deno.test("readChannels: 欄が無ければ落とす", () => {
  assertThrows(
    () => readChannels({}, "imageMean", "pipelineConfig", Number.isFinite, "だめ"),
    Error,
    "pipelineConfig.imageMean: 無い",
  );
});

Deno.test("readChannels: 配列でない / 長さが 3 でないものを落とす", () => {
  for (const bad of [0.5, "rgb", { 0: 1, 1: 2, 2: 3 }, [], [1, 2], [1, 2, 3, 4]]) {
    assertThrows(
      () => readChannels({ c: bad }, "c", "here", Number.isFinite, "だめ"),
      Error,
      "here.c: 長さ 3 の配列でない",
    );
  }
});

Deno.test("readChannels: 要素の型違い・条件外は要素ごとに落とし、文言に配列全体を出す", () => {
  for (const bad of [[1, "2", 3], [1, 2, Number.NaN], [0.2, 0, 0.2]]) {
    assertThrows(
      () => readChannels({ c: bad }, "c", "here", isPositive, "正の有限数でない要素がある"),
      Error,
      "here.c: 正の有限数でない要素がある",
    );
  }
});

Deno.test("readOnly: 受理する唯一の値ならその値を返す", () => {
  assertEquals(
    readOnly({ i: "bilinear" }, "i", "here", "bilinear", "上流がそうだから"),
    "bilinear",
  );
});

Deno.test("readOnly: 欄が無ければ落とす", () => {
  assertThrows(
    () => readOnly({}, "interpolation", "pipelineConfig", "bilinear", "why"),
    Error,
    "pipelineConfig.interpolation: 無い",
  );
});

Deno.test("readOnly: 綴り違いも対応外も同じ文言で落とし、label と理由が入る", () => {
  for (const bad of ["bicubic", "BILINEAR", 2, null]) {
    assertThrows(
      () => readOnly({ i: bad }, "i", "pipelineConfig", "bilinear", "上流が bilinear で焼いている"),
      Error,
      "pipelineConfig.i: この実装が対応するのは 'bilinear' だけ" +
        `（${String(bad)}）— 上流が bilinear で焼いている`,
    );
  }
});
