// 画像前処理層（src/image/preprocess.ts）の Python 正本とのパリティ検証。
//
// フィクスチャ `image-preprocess/parity.json` は exporter（tools/exporter/siglip2_preprocess.py）が
// 生成する。正本は transformers 5.14.1 の `SiglipImageProcessor`（`TorchvisionBackend` — v5 の
// `AutoImageProcessor` が既定で返す fast 側）で、中身は 7 ケースぶんの
//  ・`input`       合成画像の RGB8（行優先・画素あたり 3 バイト）
//  ・`resized`     resize 直後の RGB8。**フルパイプラインの中間そのもの**であることは
//                  emit 時に実測してある（融合正規化に通して `pixelValues` とビット同一）
//  ・`pixelValues` 正規化済み `[3, outHeight, outWidth]` の f32
//
// 実資産（outputs/ 配下）にも GPU にも依存しないので常時走る。
//
// 縛るのは振る舞い: 「同じ RGB8 を入れたら Python と同じ `pixel_values` が出る」。重みの
// 持ち方や 2 パスの回し方は縛らない。
//
// ## tolerance の導出（実測）
//
// resize は**完全一致にならない** — 参照側は f32 で積算し、こちらは f64 で積算するので、
// 丸め境界に載った標本だけが 1 LSB ずれる（片軸だけを動かす切り分けで、参照の f32 経路との
// 丸め**前**の差が 1e-4〜2.4e-3 級であることを実測した）。実測:
//
//  ・差は**必ず 1 LSB 以内**（13 幾何 × 2 種の画像 = 1,286,856 標本で最大 1）
//  ・ずれる標本は 0.28%（同上）。本フィクスチャの 7 ケース 3,063 標本では 7 件 = 0.23%
//  ・丸めは half-up が正しい。同じ実測で half-down は相違 3.1 倍・half-even は 2.0 倍に増え、
//    どちらも最大 2 LSB へ悪化する（フィクスチャ内の同点が偶然 half-down 側に見えるのに
//    引きずられないこと）
//
// したがって門は 3 段に分ける:
//  ① 正規化単体は**完全一致**（参照 `resized` から作った f32 が `pixelValues` とビット同一）
//  ② resize は 1 LSB 以内 + 相違率の上限（実測 0.23% に対し {@link MISMATCH_RATE_CAP}）
//  ③ 全経路は ② が f32 へ写った幅 = {@link PIXEL_ATOL}
//
// ①と②を分けるのは、tolerance 付きの全経路比較だけだと、正規化の定数取り違え（1 LSB より
// 小さい差にしかならない組み合わせがある）が素通りするため。
//
// ## bicubic（`resample: 3`）の扱い
//
// フィクスチャの正本は SigLIP2 = **bilinear** なので、bicubic 枝の Python 正本との突合は
// ここには無い（実測は Depth Anything V2 の実画像パリティ門 =
// `e2e_depth_anything_real_test.ts` が持つ — 1024² の実画像を 518² へ縮小した
// `pixel_values` が `DPTImageProcessor` と **1 LSB 以内 / 相違率 0.5%** で一致することを、
// 資産のある環境で毎回実測する）。ここが受け持つのは資産の要らない**カーネルの署名**だけ:
// 恒等寸法で恒等になること（台の半径 2 が正しく組めている）と、階段エッジで**入力の値域を
// 越える overshoot** が出ること（負のローブを持つ三次カーネルでしか起きない）。

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../src/image/preprocess.ts";

type Case = {
  readonly name: string;
  readonly why: string;
  readonly height: number;
  readonly width: number;
  readonly outHeight: number;
  readonly outWidth: number;
  readonly input: number[];
  readonly resized: number[];
  readonly pixelValues: number[];
};

type Fixture = {
  readonly constants: {
    readonly imageMean: [number, number, number];
    readonly imageStd: [number, number, number];
    readonly rescaleFactor: number;
  };
  readonly checkpoints: Record<string, { readonly height: number; readonly width: number }>;
  readonly cases: Case[];
};

const FIXTURE_PATH = new URL("./fixtures/image-preprocess/parity.json", import.meta.url);
const fixture = JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture;

const { imageMean, imageStd } = fixture.constants;

/** resize の 1 標本あたりの許容差（uint8 の 1 LSB）。導出はファイル冒頭。 */
const RESIZE_MAX_DELTA = 1;

/** 相違する標本の割合の上限。実測 0.23% に対し 4 倍強を取る。 */
const MISMATCH_RATE_CAP = 0.01;

/**
 * 全経路の許容差。uint8 の 1 LSB は正規化後 `1 / 127.5 = 7.8431e-3` で、f32 へ写す丸めが
 * 乗って実測最大 7.84314e-3。その上に僅かな余裕を取った値。
 */
const PIXEL_ATOL = 7.85e-3;

const imageOf = (testCase: Case): Rgb8Image => ({
  data: Uint8Array.from(testCase.input),
  width: testCase.width,
  height: testCase.height,
});

const resizedOf = (testCase: Case): Rgb8Image => ({
  data: Uint8Array.from(testCase.resized),
  width: testCase.outWidth,
  height: testCase.outHeight,
});

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でない（取り違えで全ケース素通しになっていない）", () => {
  // 1 本目に置く。読み違い / 生成失敗でケースが空になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assertEquals(fixture.cases.length, 7);
  assertEquals(imageMean, [0.5, 0.5, 0.5]);
  assertEquals(imageStd, [0.5, 0.5, 0.5]);
  assertEquals(fixture.constants.rescaleFactor, 1 / 255);
  // 前処理定数は 224 / 384 の両チェックポイントで共有される（size だけが違うことは emit 時に
  // 実測済み）。片方しか無いフィクスチャは、その主張が検査されないまま作られたもの。
  assert(Object.keys(fixture.checkpoints).length >= 2, "チェックポイントが 1 つしかない");
  for (const testCase of fixture.cases) {
    assertEquals(
      testCase.input.length,
      testCase.width * testCase.height * 3,
      `[${testCase.name}] input の長さ`,
    );
    assertEquals(
      testCase.resized.length,
      testCase.outWidth * testCase.outHeight * 3,
      `[${testCase.name}] resized の長さ`,
    );
    assertEquals(
      testCase.pixelValues.length,
      testCase.resized.length,
      `[${testCase.name}] pixelValues の長さ`,
    );
  }
});

// ---- ① 正規化（resize と切り分けて完全一致で縛る）--------------------------

for (const testCase of fixture.cases) {
  Deno.test(`正規化パリティ [${testCase.name}] 参照 resized → pixelValues がビット同一`, () => {
    const got = normalizeToNchw(resizedOf(testCase), imageMean, imageStd);
    assertEquals(Array.from(got), testCase.pixelValues);
  });
}

// ---- ② resize（1 LSB 以内）--------------------------------------------------

for (const testCase of fixture.cases) {
  Deno.test(`resize パリティ [${testCase.name}] ${testCase.why}`, () => {
    const got = resizeRgb8(imageOf(testCase), testCase.outWidth, testCase.outHeight);
    assertEquals(got.width, testCase.outWidth);
    assertEquals(got.height, testCase.outHeight);
    for (let index = 0; index < testCase.resized.length; index += 1) {
      const delta = Math.abs(got.data[index] - testCase.resized[index]);
      assert(
        delta <= RESIZE_MAX_DELTA,
        `標本 ${index} が ${got.data[index]}（参照 ${testCase.resized[index]}）`,
      );
    }
  });
}

Deno.test("resize の相違率が実測の幅に収まる（全ケース合計）", () => {
  // 1 LSB 以内という上限だけだと、全画素が 1 ずつずれる実装（例: 丸めの取り違え）が通る。
  let mismatched = 0;
  let total = 0;
  for (const testCase of fixture.cases) {
    const got = resizeRgb8(imageOf(testCase), testCase.outWidth, testCase.outHeight);
    for (let index = 0; index < testCase.resized.length; index += 1) {
      if (got.data[index] !== testCase.resized[index]) mismatched += 1;
    }
    total += testCase.resized.length;
  }
  const rate = mismatched / total;
  assert(rate <= MISMATCH_RATE_CAP, `相違率 ${rate}（${mismatched}/${total}）が上限を超えた`);
});

// ---- ③ 全経路（RGB8 → pixel_values）----------------------------------------

for (const testCase of fixture.cases) {
  Deno.test(`全経路パリティ [${testCase.name}] ${testCase.why}`, () => {
    const resized = resizeRgb8(imageOf(testCase), testCase.outWidth, testCase.outHeight);
    const got = normalizeToNchw(resized, imageMean, imageStd);
    assertEquals(got.length, testCase.pixelValues.length);
    for (let index = 0; index < got.length; index += 1) {
      assertAlmostEquals(got[index], testCase.pixelValues[index], PIXEL_ATOL, `標本 ${index}`);
    }
  });
}

// ---- bicubic（DA-V2 の `resample: 3`）のカーネル署名 ------------------------

Deno.test("bicubic は同寸で恒等（台の半径 2 が正しく組めている）", () => {
  // 縮尺 1 では三次カーネルが整数位置で `[…, 0, 1, 0, …]` に落ちる。台の半径を 1 のまま
  // （bilinear の値）にしたり、カーネルの分岐境界を取り違えたりすると、ここで高周波の
  // 画像が滲む。
  for (const testCase of fixture.cases) {
    const image = imageOf(testCase);
    const got = resizeRgb8(image, testCase.width, testCase.height, "bicubic");
    assertEquals(Array.from(got.data), Array.from(image.data), `[${testCase.name}]`);
  }
});

Deno.test("bicubic は階段エッジで入力の値域を越える（bilinear は越えない）", () => {
  // 負のローブの直接の署名。フィルタ指定が黙って無視されて bilinear へ落ちると、overshoot が
  // 消えてここが落ちる（恒等の検査だけでは 2 つのフィルタを区別できない）。
  const low = 40;
  const high = 200;
  const width = 8;
  const data = new Uint8Array(width * 3);
  for (let x = 0; x < width; x += 1) {
    const value = x < width / 2 ? low : high;
    data[x * 3] = value;
    data[x * 3 + 1] = value;
    data[x * 3 + 2] = value;
  }
  const step: Rgb8Image = { data, width, height: 1 };

  const bicubic = resizeRgb8(step, width * 4, 1, "bicubic");
  const bilinear = resizeRgb8(step, width * 4, 1, "bilinear");

  assert(Math.min(...bicubic.data) < low, "bicubic の undershoot が無い");
  assert(Math.max(...bicubic.data) > high, "bicubic の overshoot が無い");
  assertEquals(Math.min(...bilinear.data), low, "bilinear が値域を下へ外れた");
  assertEquals(Math.max(...bilinear.data), high, "bilinear が値域を上へ外れた");
});

Deno.test("bicubic の overshoot は巻き戻らず飽和する", () => {
  // MUST: `Uint8Array` への代入は範囲外を mod 256 で巻き戻す（黒白反転が静かに通る）。
  // 端の値を振り切らせて、飽和側に張り付くことを直に見る。
  const width = 8;
  const data = new Uint8Array(width * 3);
  for (let x = 0; x < width; x += 1) {
    const value = x < width / 2 ? 0 : 255;
    data[x * 3] = value;
    data[x * 3 + 1] = value;
    data[x * 3 + 2] = value;
  }
  const got = resizeRgb8({ data, width, height: 1 }, width * 4, 1, "bicubic");

  assertEquals(Math.min(...got.data), 0);
  assertEquals(Math.max(...got.data), 255);
});

// ---- 入口の契約（想定外は fail loudly）--------------------------------------

Deno.test("画素列の長さが寸法と合わなければ落ちる", () => {
  const image: Rgb8Image = { data: new Uint8Array(11), width: 2, height: 2 };
  assertThrows(() => resizeRgb8(image, 4, 4), Error, "RGB8 の長さ 11");
  assertThrows(() => normalizeToNchw(image, imageMean, imageStd), Error, "RGB8 の長さ 11");
});

Deno.test("寸法が正の整数でなければ落ちる", () => {
  const empty: Rgb8Image = { data: new Uint8Array(0), width: 0, height: 3 };
  assertThrows(() => resizeRgb8(empty, 4, 4), RangeError, "画像サイズ 0×3");
  const image: Rgb8Image = { data: new Uint8Array(12), width: 2, height: 2 };
  assertThrows(() => resizeRgb8(image, 0, 4), RangeError, "出力サイズ 0×4");
  assertThrows(() => resizeRgb8(image, 4, 2.5), RangeError, "出力サイズ 4×2.5");
});
