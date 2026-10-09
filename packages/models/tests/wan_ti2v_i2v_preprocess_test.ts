// Wan2.2 TI2V の I2V の条件画像の前処理のホスト側（ADR 0121 段 9a — GPU 不要・資産不要）。
//
// 見るのは 6 つ:
//
// - **LANCZOS**（`src/image/lanczos.ts`）: Pillow の `Image.resize(..., LANCZOS)` と uint8 で全一致する（git 追跡の
//   fixture `fixtures/wan-i2v/lanczos.{json,safetensors}` — 書き手は recipe の `wan/lanczos_fixture.py`・無ければ SKIP
//   ではなく赤）。加えて fixture の作り方に依らない 2 点（恒等寸法は写し・段差の拡大は 0 / 255 に飽和して mod 256 で
//   巻き戻らない）。
// - **出力寸法の選択**: 受理集合の中から公式の比較式 `max(r/rc, rc/r)` で選び、同点（正方形）は横長。差の絶対値で
//   測る形とは 1100×1000 で結果が割れる（表の行がその取り違えを赤にする）。
// - **縮尺後の寸法**: Python の `round`（.5 は偶数側）— 752.5 → 752・757.5 → 758。
// - **寸法合わせ**: crop は覆う寸法へ伸縮してから `(rw − ow) // 2` で中央を切り出す（床関数）。stretch は全体を収める。
// - **`[-1, 1]` と patchify**: 公式の `to_tensor → sub(0.5) → div(0.5)` と f32 で一致し、patchify は上流の
//   `permute(0,1,6,4,2,3,5)` と同じ並び（高さと幅の副添字の取り違えを故障注入で赤）。
// - **encoder の出口の正規化**: `f32(f32(mu − mean) · f32(1 / f32(std)))`（割り算の形・f64 の逆数の形とは割れる）。
//
// 実画像の鎖（PNG → … → encoder の入力）の golden とのビット一致は `wan_ti2v_i2v_preprocess_real_test.ts`。

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import {
  resizeRgb8Lanczos,
  resizeRgb8LanczosWindow,
  type Rgb8Window,
} from "../src/image/lanczos.ts";
import { normalizeToNchw, type Rgb8Image } from "../src/image/preprocess.ts";
import { WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import {
  fitWanI2vImage,
  patchifyWanImage,
  preprocessWanI2vImage,
  selectWanI2vSize,
  wanI2vCoverSize,
  type WanI2vFit,
  wanI2vPixels,
} from "../src/wan/i2v-preprocess.ts";
import {
  denormalizeWanLatents,
  normalizeWanLatents,
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
} from "../src/wan/latents.ts";
import { unpatchifyWanVaeFrames } from "../src/wan/vae-tiles.ts";
import { WAN_I2V_LANCZOS_FIXTURE } from "./helpers/wan-i2v-image.ts";
import { firstBitMismatch, readBuffer, viewOf } from "./helpers/wan-ti2v-dit.ts";

const CHANNELS = 3;

/** 画素ごとに値を決めて RGB8 を作る（`pixel(x, y)` は [r, g, b]）。 */
const synthetic = (
  width: number,
  height: number,
  pixel: (x: number, y: number) => readonly [number, number, number],
): Rgb8Image => {
  const data = new Uint8Array(width * height * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data.set(pixel(x, y), (y * width + x) * CHANNELS);
    }
  }
  return { data, width, height };
};

/** 座標を画素値に埋めた画像（切り出しの位置を値から読み戻せる）。 */
const coordinateImage = (width: number, height: number): Rgb8Image =>
  synthetic(width, height, (x, y) => [x & 255, y & 255, (x >> 8) | ((y >> 8) << 4)]);

const pixelAt = (image: Rgb8Image, x: number, y: number): readonly number[] => {
  const at = (y * image.width + x) * CHANNELS;
  return [...image.data.subarray(at, at + CHANNELS)];
};

/** 矩形の切り出し（Pillow の `crop` の写し — 窓の計算と独立な参照）。 */
const cropOf = (image: Rgb8Image, { left, top, width, height }: Rgb8Window): Rgb8Image => {
  const data = new Uint8Array(width * height * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data.set(pixelAt(image, left + x, top + y), (y * width + x) * CHANNELS);
    }
  }
  return { data, width, height };
};

// ---- LANCZOS ---------------------------------------------------------------------------------------------

/** fixture のケース表の 1 行（`lanczos.json` の `cases` — 画素は safetensors 側）。 */
type LanczosCase = {
  readonly name: string;
  readonly why: string;
  readonly height: number;
  readonly width: number;
  readonly outHeight: number;
  readonly outWidth: number;
};

type LanczosFixture = {
  readonly pillow: string;
  readonly cases: readonly LanczosCase[];
  readonly tensors: SafetensorsFile;
};

const isLanczosCase = (value: unknown): value is LanczosCase => {
  if (typeof value !== "object" || value === null) return false;
  const row = value as Record<string, unknown>;
  return typeof row.name === "string" && typeof row.why === "string" &&
    [row.height, row.width, row.outHeight, row.outWidth].every((size) =>
      Number.isInteger(size) && (size as number) > 0
    );
};

/**
 * fixture（ケース表の JSON + 画素の safetensors）を読む。形が `helpers/wan-i2v-image.ts` の doc から外れたら fail
 * loudly。MUST: 無ければ投げる（git 追跡の fixture なので、無いのは取り違えか書き損じ — SKIP に丸めると Pillow との
 * 一致を誰も見ていない状態が緑になる）。
 */
const readLanczosFixture = async (): Promise<LanczosFixture> => {
  const raw = JSON.parse(await Deno.readTextFile(WAN_I2V_LANCZOS_FIXTURE)) as Record<
    string,
    unknown
  >;
  const source = raw.source as Record<string, unknown> | undefined;
  if (
    typeof source?.pillow !== "string" || typeof raw.tensors !== "string" ||
    !Array.isArray(raw.cases) || !raw.cases.every(isLanczosCase)
  ) {
    throw new Error(
      `${WAN_I2V_LANCZOS_FIXTURE.pathname}: ケース表の形が {source.pillow, tensors, cases} でない`,
    );
  }
  const tensors = parseSafetensors(await readBuffer(new URL(raw.tensors, WAN_I2V_LANCZOS_FIXTURE)));
  return { pillow: source.pillow, cases: raw.cases, tensors };
};

/** ケースの画素（U8 `[高さ, 幅, 3]` — 形の食い違いは fail loudly）。 */
const pixelsOf = (
  fixture: LanczosFixture,
  key: string,
  height: number,
  width: number,
): Uint8Array<ArrayBuffer> => {
  const view = viewOf(fixture.tensors, key, "LANCZOS の fixture");
  assertEquals(view.dtype, "U8", `${key} の dtype`);
  assertEquals([...view.shape], [height, width, CHANNELS], `${key} の形`);
  return new Uint8Array(
    fixture.tensors.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength),
  );
};

describe("LANCZOS（Pillow の 8bit 経路の逐語の移植）", () => {
  it("fixture の全ケースが Pillow と uint8 で全一致する", async () => {
    const fixture = await readLanczosFixture();
    const failures: string[] = [];
    for (const { name, height, width, outHeight, outWidth } of fixture.cases) {
      const input: Rgb8Image = {
        data: pixelsOf(fixture, `${name}.input`, height, width),
        width,
        height,
      };
      const expected = pixelsOf(fixture, `${name}.resized`, outHeight, outWidth);
      const actual = resizeRgb8Lanczos(input, outWidth, outHeight);
      let mismatches = 0;
      let first = -1;
      for (let index = 0; index < expected.length; index += 1) {
        if (actual.data[index] === expected[index]) continue;
        mismatches += 1;
        if (first < 0) first = index;
      }
      if (mismatches > 0) {
        failures.push(
          `${name}: ${mismatches} / ${expected.length} 標本が違う` +
            `（最初は ${first}: ${actual.data[first]} 対 ${expected[first]}）`,
        );
      }
    }
    assertEquals(
      failures,
      [],
      `Pillow ${fixture.pillow} と ${fixture.cases.length} ケース中 ${failures.length} ケースが違う`,
    );
  });

  it("fixture は縮小（台が伸びる経路）・拡大・片軸だけの変化を含み、Pillow の版を記録している", async () => {
    const fixture = await readLanczosFixture();
    const { cases } = fixture;
    assert(
      cases.some((row) => row.outWidth < row.width || row.outHeight < row.height),
      "縮小のケースが無い（実際の入力に多い経路が Pillow と照合されない）",
    );
    assert(
      cases.some((row) => row.outWidth > row.width || row.outHeight > row.height),
      "拡大のケースが無い",
    );
    // NOTE: 寸法の変わらない軸のパスを飛ばすこと自体は出力では区別できない（n → n の係数は中心が 2^22・他が 0 で、
    // パスを通しても写しと同じ値）。このケースが縛るのは片軸だけの伸縮。
    assert(
      cases.some((row) => (row.outWidth === row.width) !== (row.outHeight === row.height)),
      "片軸だけの伸縮（横だけ・縦だけ）のケースが無い（片軸だけの伸縮が Pillow と照合されない）",
    );
    assertEquals(new Set(cases.map(({ name }) => name)).size, cases.length, "ケース名の重複");
    assert(fixture.pillow.length > 0, "Pillow の版が空");
  });

  it("寸法が変わらなければパスを飛ばして入力の写しを返す（同じ値・別の配列）", () => {
    const image = coordinateImage(23, 17);
    const same = resizeRgb8Lanczos(image, 23, 17);
    assertEquals([...same.data], [...image.data]);
    assert(
      same.data.buffer !== image.data.buffer,
      "入力と同じ配列を返した（呼び手の画像と共有する）",
    );
  });

  it("窓だけを計算した値は、全体を伸縮してから切り出した値と uint8 で同じ（両軸・片軸・恒等・窓の端）", () => {
    const table: readonly (readonly [number, number, number, number, Rgb8Window])[] = [
      // [入力の幅, 高さ, 伸縮後の幅, 高さ, 窓]
      [13, 9, 50, 31, { left: 7, top: 4, width: 30, height: 20 }], // 両軸の拡大
      [73, 41, 29, 17, { left: 3, top: 5, width: 20, height: 9 }], // 両軸の縮小（台が伸びる）
      [40, 12, 23, 12, { left: 2, top: 3, width: 15, height: 6 }], // 横だけ（縦は行の切り出し）
      [12, 40, 12, 57, { left: 4, top: 10, width: 5, height: 30 }], // 縦だけ（横は列の切り出し）
      [21, 13, 21, 13, { left: 3, top: 2, width: 9, height: 7 }], // 恒等（両方とも切り出し）
      [9, 50, 31, 13, { left: 30, top: 12, width: 1, height: 1 }], // 右下の角の 1 画素
      [50, 9, 13, 31, { left: 0, top: 0, width: 13, height: 31 }], // 窓 = 全体
    ];
    for (const [width, height, outWidth, outHeight, window] of table) {
      const image = synthetic(
        width,
        height,
        (x, y) => [(x * 37 + y * 11) & 255, (x * y) & 255, (x ^ y) * 5 & 255],
      );
      const whole = resizeRgb8Lanczos(image, outWidth, outHeight);
      const windowed = resizeRgb8LanczosWindow(image, outWidth, outHeight, window);
      assertEquals([windowed.width, windowed.height], [window.width, window.height]);
      assertEquals(
        [...windowed.data],
        [...cropOf(whole, window).data],
        `${width}×${height} → ${outWidth}×${outHeight} の窓 ${JSON.stringify(window)}`,
      );
    }
  });

  it("窓が出力の外にはみ出せば RangeError（内部の事前条件）", () => {
    const image = coordinateImage(4, 4);
    for (
      const window of [
        { left: 5, top: 0, width: 4, height: 8 },
        { left: 0, top: -1, width: 8, height: 8 },
        { left: 0, top: 0, width: 0, height: 8 },
        { left: 0.5, top: 0, width: 4, height: 8 },
      ]
    ) {
      assertThrows(() => resizeRgb8LanczosWindow(image, 8, 8, window), RangeError, "窓");
    }
  });

  it("段差の拡大は入力の値域（5〜250）を越えて 0 / 255 に飽和し、mod 256 で巻き戻らない", () => {
    // 期待値は Pillow 12.3.0 の `Image.resize((32, 1), Image.LANCZOS)` の出力の写し（負のローブの谷が 5 の側で 0 に、
    // 山が 250 の側で 255 に張り付く）。飽和を外して `Uint8Array` へ入れると、谷が 250 前後に巻き戻る。
    const image = synthetic(8, 1, (x) => {
      const value = x < 4 ? 5 : 250;
      return [value, value, value];
    });
    const resized = resizeRgb8Lanczos(image, 32, 1);
    const red = Array.from({ length: 32 }, (_, x) => resized.data[x * CHANNELS]);
    assertEquals(red, [
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      9,
      13,
      10,
      0,
      0,
      0,
      0,
      28,
      91,
      164,
      227,
      255,
      255,
      255,
      255,
      245,
      242,
      246,
      250,
      250,
      250,
      250,
      250,
      250,
      250,
    ]);
  });

  it("出力寸法が正の整数でなければ RangeError（内部の事前条件）・画像の長さの食い違いは ModelInputError", () => {
    const image = coordinateImage(4, 4);
    assertThrows(() => resizeRgb8Lanczos(image, 0, 4), RangeError);
    assertThrows(() => resizeRgb8Lanczos(image, 4.5, 4), RangeError);
    assertThrows(
      () => resizeRgb8Lanczos({ data: new Uint8Array(10), width: 4, height: 4 }, 8, 8),
      ModelInputError,
    );
  });
});

// ---- 出力寸法の選択 ---------------------------------------------------------------------------------------

describe("出力寸法の選択（受理集合 × 公式の比較式 max(r/rc, rc/r)・同点は横長）", () => {
  const blank = (width: number, height: number): Rgb8Image => ({
    data: new Uint8Array(width * height * CHANNELS),
    width,
    height,
  });

  it("画像の縦横比に最も近い受理寸法を選ぶ", () => {
    const table: readonly (readonly [number, number, number, number])[] = [
      [832, 480, 1280, 704],
      [480, 832, 704, 1280],
      [1920, 1080, 1280, 704],
      [1080, 1920, 704, 1280],
      // 正方形は 2 候補が厳密に同点 → 横長。
      [512, 512, 1280, 704],
      // r = 1.1: 比の比較式は横長（1.65 対 2.0）、差の絶対値なら縦長（0.72 対 0.55）— 物差しの取り違えを赤にする行。
      [1100, 1000, 1280, 704],
      [1000, 1001, 704, 1280],
    ];
    for (const [width, height, expectedWidth, expectedHeight] of table) {
      assertEquals(
        selectWanI2vSize(blank(width, height), {}, WAN22_TI2V_GENERATION),
        { width: expectedWidth, height: expectedHeight },
        `${width}×${height}`,
      );
    }
  });

  it("同点の横長は受理集合の並びに依らない（縦長を先に並べても正方形は横長）", () => {
    const reversed = { acceptedSizes: [...WAN22_TI2V_GENERATION.acceptedSizes].reverse() };
    assertEquals(selectWanI2vSize(blank(512, 512), {}, reversed), { width: 1280, height: 704 });
  });

  it("明示した寸法は縦横比より優先し、片方だけならその欄が合う候補から選ぶ", () => {
    const landscape = blank(832, 480);
    assertEquals(
      selectWanI2vSize(landscape, { width: 704, height: 1280 }, WAN22_TI2V_GENERATION),
      { width: 704, height: 1280 },
    );
    assertEquals(
      selectWanI2vSize(landscape, { width: 704 }, WAN22_TI2V_GENERATION),
      { width: 704, height: 1280 },
    );
    assertEquals(
      selectWanI2vSize(blank(480, 832), { height: 704 }, WAN22_TI2V_GENERATION),
      { width: 1280, height: 704 },
    );
  });

  it("明示した寸法が受理集合の外なら ModelInputError", () => {
    const image = blank(832, 480);
    for (
      const [requested, message] of [
        [{ width: 832, height: 480 }, "832×480 が受理集合（1280×704 / 704×1280）に無い"],
        [{ width: 1280, height: 1280 }, "1280×1280 が受理集合"],
        [{ width: 1000 }, "width 1000 が受理集合"],
        [{ height: 1000 }, "height 1000 が受理集合"],
      ] as const
    ) {
      assertThrows(
        () => selectWanI2vSize(image, requested, WAN22_TI2V_GENERATION),
        ModelInputError,
        message,
      );
    }
  });

  it("画像そのものの食い違い（長さ）は ModelInputError", () => {
    assertThrows(
      () =>
        selectWanI2vSize(
          { data: new Uint8Array(5), width: 4, height: 4 },
          {},
          WAN22_TI2V_GENERATION,
        ),
      ModelInputError,
    );
  });
});

// ---- 縮尺後の寸法（Python の round）----------------------------------------------------------------------

describe("crop の縮尺後の寸法（scale = max(ow/iw, oh/ih) → Python の round）", () => {
  const landscape = { width: 1280, height: 704 };

  it(".5 ちょうどは偶数側へ丸める（752.5 → 752・757.5 → 758）", () => {
    // JS の Math.round なら 753 になる（.5 は +∞ 側）— この行が丸めの取り違えを赤にする。
    assertEquals(Math.round(301 * 2.5), 753);
    assertEquals(wanI2vCoverSize({ width: 512, height: 301 }, landscape), {
      width: 1280,
      height: 752,
    });
    assertEquals(wanI2vCoverSize({ width: 512, height: 303 }, landscape), {
      width: 1280,
      height: 758,
    });
  });

  it("覆う側の縮尺を取る（縮尺を決めた軸は目標ちょうど・もう片方は目標以上）", () => {
    assertEquals(wanI2vCoverSize({ width: 832, height: 480 }, landscape), {
      width: 1280,
      height: 738,
    });
    assertEquals(wanI2vCoverSize({ width: 480, height: 832 }, { width: 704, height: 1280 }), {
      width: 738,
      height: 1280,
    });
    assertEquals(wanI2vCoverSize({ width: 1280, height: 800 }, landscape), {
      width: 1280,
      height: 800,
    });
    assertEquals(wanI2vCoverSize({ width: 3000, height: 1000 }, landscape), {
      width: 2112,
      height: 704,
    });
  });
});

// ---- 寸法合わせ ------------------------------------------------------------------------------------------

describe("寸法合わせ（crop = 覆ってから中央を切り出す / stretch = 全体を直接伸縮）", () => {
  const landscape = { width: 1280, height: 704 };

  it("crop の切り出し位置は (rw − ow) // 2（余りは右・下へ — 床関数）", () => {
    // 縮尺 1 の軸は LANCZOS のパスを飛ばすので、出力の画素は入力の画素そのもの（座標から位置を読み戻せる）。
    for (
      const [width, height, left, top] of [
        [1280, 800, 0, 48],
        [1280, 805, 0, 50], // (805 − 704) / 2 = 50.5 → 50
        [1400, 704, 60, 0],
        [1401, 704, 60, 0], // 60.5 → 60
      ] as const
    ) {
      const image = coordinateImage(width, height);
      const fitted = fitWanI2vImage(image, landscape, "crop");
      assertEquals([fitted.width, fitted.height], [1280, 704]);
      for (const [x, y] of [[0, 0], [1279, 0], [0, 703], [1279, 703], [640, 352]] as const) {
        assertEquals(
          pixelAt(fitted, x, y),
          pixelAt(image, x + left, y + top),
          `${width}×${height} の (${x}, ${y})`,
        );
      }
    }
  });

  it("crop は覆う寸法へ全体を伸縮してから中央を切り出した値と uint8 で同じ（縦横比の極端な 64×4 → 覆う寸法 11,264×704）", () => {
    const image = synthetic(
      64,
      4,
      (x, y) => [(x * 13 + y * 71) & 255, (x * 7) & 255, (y * 97) & 255],
    );
    const cover = wanI2vCoverSize(image, landscape);
    assertEquals(cover, { width: 11264, height: 704 });
    const whole = resizeRgb8Lanczos(image, cover.width, cover.height);
    const expected = cropOf(whole, {
      left: (cover.width - landscape.width) / 2,
      top: 0,
      width: landscape.width,
      height: landscape.height,
    });
    assertEquals([...fitWanI2vImage(image, landscape, "crop").data], [...expected.data]);
  });

  it("縦横比の極端な画像（4096×4 → 覆う寸法 720,896×704）も出力の窓だけを計算して切り出す", () => {
    // NOTE: 手間の上限そのものは門にできない（覆う寸法の全体を作る形へ戻しても、この it は数分と 1.5 GB を使った後に
    // 緑で終わる — 赤にはならない）。ここで縛るのは寸法と、窓の経路が端の窓（覆う寸法の中央）でも走ること。値は
    // 上の 64×4 の it が全体の伸縮と突き合わせる。
    const image = synthetic(4096, 4, (x, y) => [x & 255, (x >> 4) & 255, y * 60]);
    const fitted = fitWanI2vImage(image, landscape, "crop");
    assertEquals([fitted.width, fitted.height], [1280, 704]);
    assertEquals(fitted.data.length, 1280 * 704 * CHANNELS);
  });

  it("stretch は縦横比を保たずに全体を収め、crop が落とす上端の帯を残す", () => {
    // 1280×800 の上端 8 行だけ赤。crop は上下 48 行ずつ落とすので赤は残らず、stretch は 800 → 704 に縮めて残す。
    const image = synthetic(1280, 800, (_, y) => (y < 8 ? [255, 0, 0] : [128, 128, 128]));
    const cropped = fitWanI2vImage(image, landscape, "crop");
    const stretched = fitWanI2vImage(image, landscape, "stretch");
    assertEquals([stretched.width, stretched.height], [1280, 704]);
    const reddish = (fitted: Rgb8Image): boolean =>
      Array.from({ length: fitted.width * fitted.height }, (_, at) => at * CHANNELS)
        .some((at) => fitted.data[at] > 200 && fitted.data[at + 1] < 60);
    assert(!reddish(cropped), "crop に上端の帯が残った");
    assert(reddish(stretched), "stretch で上端の帯が消えた");
    const [r, g] = pixelAt(stretched, 640, 0);
    assert(r > 200 && g < 60, `stretch の上端が赤でない（${r}, ${g}）`);
  });

  it("未知の fit は fail loudly（素の Error — 名前の綴り違い）", () => {
    assertThrows(
      () => fitWanI2vImage(coordinateImage(4, 4), landscape, "cover" as never),
      Error,
      "未対応",
    );
  });
});

// ---- [-1, 1] と patchify ---------------------------------------------------------------------------------

describe("[-1, 1] の画素（f32(2·f32(x/255) − 1)・planar [3, H, W]）", () => {
  // x = 0..255 を R に、255 − x を G に、x を B に並べた 256×1 の画像。
  const ramp = synthetic(256, 1, (x) => [x, 255 - x, x]);

  it("公式の to_tensor → sub(0.5) → div(0.5) と diffusers の 2·(x/255) − 1 の両方と f32 で一致し、チャネルは planar に並ぶ", () => {
    const pixels = wanI2vPixels(ramp);
    assertEquals(pixels.length, 3 * 256);
    const official = (value: number): number =>
      Math.fround(Math.fround(Math.fround(value / 255) - 0.5) / 0.5);
    // diffusers（stretch の正本）: `np.float32(x) / 255.0` → `2.0 * images - 1.0`（どちらも f32 の演算）。
    const diffusers = (value: number): number =>
      Math.fround(Math.fround(2 * Math.fround(value / 255)) - 1);
    for (let x = 0; x < 256; x += 1) {
      assertEquals(pixels[x], official(x), `R ${x}`);
      assertEquals(pixels[x], diffusers(x), `R ${x}（diffusers の形）`);
      assertEquals(pixels[256 + x], official(255 - x), `G ${x}`);
      assertEquals(pixels[512 + x], official(x), `B ${x}`);
    }
    assertEquals([pixels[0], pixels[255]], [-1, 1]);
  });

  it("畳んだ形の normalizeToNchw（mean = std = 0.5）とは最終ビットが割れる（使い回しの取り違えを赤にする）", () => {
    const pixels = wanI2vPixels(ramp);
    const folded = normalizeToNchw(ramp, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    const differs = Array.from({ length: 256 }, (_, x) => x).filter((x) => pixels[x] !== folded[x]);
    assert(differs.length > 0, "畳んだ形と全値一致した（式の取り違えを縛れていない）");
  });
});

/**
 * torch の `view` → `permute` → `contiguous` を strides で写す（実装の入れ子のループとは独立な定式化）。
 * `viewShape` は行優先の形、`order` は permute の軸の並び。
 */
const viewPermute = (
  data: Float32Array,
  viewShape: readonly number[],
  order: readonly number[],
): Float32Array<ArrayBuffer> => {
  const strides = viewShape.map((_, axis) =>
    viewShape.slice(axis + 1).reduce((product, dim) => product * dim, 1)
  );
  const outShape = order.map((axis) => viewShape[axis]);
  const out = new Float32Array(data.length);
  const index = outShape.map(() => 0);
  for (let to = 0; to < out.length; to += 1) {
    out[to] = data[order.reduce((from, axis, k) => from + index[k] * strides[axis], 0)];
    for (let k = outShape.length - 1; k >= 0; k -= 1) {
      index[k] += 1;
      if (index[k] < outShape[k]) break;
      index[k] = 0;
    }
  }
  return out;
};

describe("patchify（上流 diffusers の patchify — permute(0,1,6,4,2,3,5)・p = 2）", () => {
  // 高さと幅が違い、全要素が別の値の [3, 6, 10]（取り違えが必ず値に出る）。
  const height = 6;
  const width = 10;
  const pixels = Float32Array.from({ length: 3 * height * width }, (_, index) => index);
  const p = 2;
  /** 上流: `x [1, c, 1, H, W]` → `view(1, c, 1, H/p, p, W/p, p)` → `permute(0, 1, 6, 4, 2, 3, 5)`。 */
  const viewShape = [1, 3, 1, height / p, p, width / p, p];

  it("上流の view → permute と同じ並び（channel = c·p² + 幅の副添字·p + 高さの副添字）", () => {
    const actual = patchifyWanImage(pixels, [3, height, width], p);
    assertEquals([...actual], [...viewPermute(pixels, viewShape, [0, 1, 6, 4, 2, 3, 5])]);
  });

  it("故障注入: 高さと幅の副添字を取り違えた並び（permute(0,1,4,6,…)）とは割れる", () => {
    const actual = patchifyWanImage(pixels, [3, height, width], p);
    const swapped = viewPermute(pixels, viewShape, [0, 1, 4, 6, 2, 3, 5]);
    assertNotEquals([...actual], [...swapped]);
  });

  it("unpatchifyWanVaeFrames（VAE の出口の逆変換）で元に戻る", () => {
    const patched = patchifyWanImage(pixels, [3, height, width], p);
    const restored = unpatchifyWanVaeFrames(patched, [12, 1, height / p, width / p], p);
    assertEquals([...restored], [...pixels]);
  });

  it("倍率で割り切れない寸法・長さの食い違いは fail loudly", () => {
    assertThrows(
      () => patchifyWanImage(new Float32Array(3 * 5 * 4), [3, 5, 4], 2),
      Error,
      "割り切れない",
    );
    assertThrows(() => patchifyWanImage(new Float32Array(7), [3, 2, 2], 2), Error, "要素数");
  });
});

// ---- 前処理の鎖 ------------------------------------------------------------------------------------------

describe("前処理の鎖（寸法の選択 → 寸法合わせ → [-1, 1] → patchify）", () => {
  it("encoder の入力は [12, 1, H/2, W/2]・寸法は縦横比で選ぶ・fit の既定は crop", () => {
    const image = coordinateImage(832, 480);
    const input = preprocessWanI2vImage(image, {}, WAN22_TI2V_GENERATION);
    assertEquals([input.width, input.height], [1280, 704]);
    assertEquals(input.shape, [12, 1, 352, 640]);
    assertEquals(input.pixels.length, 12 * 352 * 640);
    const crop = preprocessWanI2vImage(image, { fit: "crop" }, WAN22_TI2V_GENERATION);
    assertEquals(firstBitMismatch(input.pixels, crop.pixels), -1);
    const expected = patchifyWanImage(
      wanI2vPixels(fitWanI2vImage(image, { width: 1280, height: 704 }, "crop")),
      [3, 704, 1280],
      2,
    );
    assertEquals(firstBitMismatch(input.pixels, expected), -1);

    const portrait = preprocessWanI2vImage(coordinateImage(480, 832), {}, WAN22_TI2V_GENERATION);
    assertEquals(portrait.shape, [12, 1, 640, 352]);
  });

  it("fit = stretch は crop と別の画素を出す（欄が鎖の中で効いている）", () => {
    const image = coordinateImage(832, 480);
    const fits: readonly WanI2vFit[] = ["crop", "stretch"];
    const [crop, stretch] = fits.map((fit) =>
      preprocessWanI2vImage(image, { fit }, WAN22_TI2V_GENERATION)
    );
    assertEquals(crop.shape, stretch.shape);
    assertNotEquals(firstBitMismatch(crop.pixels, stretch.pixels), -1);
  });
});

// ---- encoder の出口の正規化 ------------------------------------------------------------------------------

describe("encoder の出口の正規化（上流 (mu − mean) · f32(1 / f32(std))）", () => {
  const stats = WAN22_TI2V_GENERATION.latents;
  const perChannel = 4096;
  const channels = WAN22_LATENTS_MEAN.length;
  const mu = Float32Array.from(
    { length: channels * perChannel },
    (_, index) => Math.fround(Math.sin(index) * 2),
  );

  it("値は f32(f32(mu − mean) · f32(1 / f32(std)))（チャネルは軸 0）", () => {
    const got = normalizeWanLatents(mu, stats);
    for (let channel = 0; channel < channels; channel += 1) {
      const inverse = Math.fround(1 / Math.fround(WAN22_LATENTS_STD[channel]));
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        assertEquals(
          got[at],
          Math.fround(Math.fround(mu[at] - mean) * inverse),
          `${channel}/${index}`,
        );
      }
    }
  });

  it("故障注入: std で割る形・config の 10 進値から f64 で作った逆数の形とは割れる", () => {
    const got = normalizeWanLatents(mu, stats);
    let differsFromDivision = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        const divided = Math.fround(Math.fround(mu[at] - mean) / WAN22_LATENTS_STD[channel]);
        if (divided !== got[at]) differsFromDivision += 1;
      }
    }
    assert(differsFromDivision > 0, "std で割る形と全値一致した（掛ける順を縛れていない）");
    // config の値は 4 桁以下の小数（`latents.ts` の規約）なので、toFixed(4) で 10 進値へ戻せる。
    let differsFromDecimalInverse = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      const inverse = Math.fround(1 / Number(WAN22_LATENTS_STD[channel].toFixed(4)));
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        const fromDecimal = Math.fround(Math.fround(mu[at] - mean) * inverse);
        if (fromDecimal !== got[at]) differsFromDecimalInverse += 1;
      }
    }
    assert(
      differsFromDecimalInverse > 0,
      "f64 の 10 進の逆数の形と全値一致した（逆数の作り方を縛れていない）",
    );
  });

  it("逆正規化と同じ統計の表・同じ逆数で往復する（往復の差は丸めの数 ulp 以内）", () => {
    const restored = denormalizeWanLatents(normalizeWanLatents(mu, stats), stats);
    for (let index = 0; index < mu.length; index += 1) {
      assert(Math.abs(restored[index] - mu[index]) <= 1e-6 * Math.max(1, Math.abs(mu[index])));
    }
  });

  it("チャネル数で割り切れない長さ・本数の違う統計は fail loudly", () => {
    assertThrows(
      () => normalizeWanLatents(new Float32Array(49), stats),
      Error,
      "正規化: 要素数 49",
    );
    assertThrows(
      () =>
        normalizeWanLatents(new Float32Array(48), { mean: stats.mean, std: stats.std.slice(1) }),
      Error,
      "正規化: mean 48 本と std 47 本の数が違う",
    );
  });
});
