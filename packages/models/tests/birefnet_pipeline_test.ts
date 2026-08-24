// `BirefnetPipeline` の**構築ガード**と**前後処理の結線**。GPU も実資産も要らない範囲だけを
// 見る（実 GPU の突合は `packages/runtime/tests/e2e_birefnet_test.ts`〈golden 入力〉と
// `e2e_birefnet_real_test.ts`〈実画像の前処理パリティと判別〉が持つ — 重複させない）。
//
// 押さえるのは 4 点:
//
// ① `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を取りに行く前**に落とす
//    （`src/birefnet/pipeline.ts` の `openBirefnetState` が掲げる MUST）。観測の仕掛けは
//    SigLIP2 側と同じ — **全ケースで `assets` は空**にしておき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産解析より前に落ちている
//     - 正しい manifest + 空 assets が `資産 'matte' が無い` で落ちる = 契約検査が全部済んだ
//       後に初めて資産へ触る（上の対偶）
//    の 2 つで門の順序そのものを縛る。
//
// ② グラフ宣言との突合（`assertStaticDim` / `assertMatteShape`）の**拒否経路**。ここは
//    `fromAssets` の中では実 GPU と実資産が揃わないと踏めないので、門を直接叩く
//    （`tests/helpers/stub-model.ts` が宣言だけの `KarumeModel` を組む）。門自身の軸番号や
//    期待形がずれても、正常系だけを走らせている限り緑のまま通るため、**壊れた宣言を名指しで
//    落とすこと**を毎回踏む。
//
// ③ 前処理の結線（`preprocessPixelValues`）が `pipelineConfig` の宣言どおりに
//    `resizeRgb8` → `normalizeToNchw` を通す。**参照は Python 正本のフィクスチャ**
//    （`fixtures/image-preprocess/parity.json`）なので、ここが見るのは「宣言の 4 欄が正しい
//    引数の位置へ届いているか」だけ。非正方のケースを使うのは、`imageWidth` / `imageHeight` の
//    取り違えが正方形では**原理的に検出できない**ため。
//
// ④ 後処理（`matteFromLogits`）の**段の順序と量子化**。sigmoid → resize → 8bit の順序は
//    上流の逐語で、入れ替えても shape も値域も合ったまま通る（順序違いは α が数十の差で
//    ずれるだけ）。順序が入れ替わったら落ちる非対称なケースを置く。

import { assert, assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";
import {
  assertMatteShape,
  assertStaticDim,
  BirefnetPipeline,
  matteFromLogits,
  preprocessPixelValues,
} from "../src/birefnet/pipeline.ts";
import { resizePlaneF32, type Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";

const FILE = {
  path: "hr/matte/model.f32.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/karume-birefnet-hr/karume.json` の `pipelineConfig` 実物（5 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 1024,
  imageHeight: 1024,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["hr"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/3",
    generator: "karume/0.2.2",
    defaultModel: "hr",
    models: {
      hr: {
        pipeline: "birefnet/1",
        weights: { matte: { f32: { shards: [FILE] } } },
        // 実行に要るのはグラフ 1 本だけ（tokenizer も表も無い）。
        assets: {},
        quants: { f32: { weights: { matte: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が birefnet でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "siglip2/1" }));
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "manifest の pipeline が 'siglip2/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "birefnet/2" }));
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`imageMean` に対する `image_mean`）が黙って既定へ縮退する経路を作らない。
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, image_mean: [0.5, 0.5, 0.5] } }),
  );
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'image_mean'",
  );
});

Deno.test("fromAssets: bilinear 以外の補間を宣言した配布形は受理しない", async () => {
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, interpolation: "bicubic" } }),
  );
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.interpolation: この実装が対応するのは 'bilinear' だけ",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 6 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、`matte` の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => BirefnetPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'matte' が無い",
  );
});

// ---- グラフ宣言との突合（拒否経路）------------------------------------------

/** 実配布形（hr）と同じ宣言。`patch` で 1 点だけ壊す。 */
const matteGraph = (
  patch: {
    readonly inputShape?: readonly (number | string)[];
    readonly outputShape?: readonly (number | string)[];
  } = {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 1024, 1024] }],
    outputs: ["matte"],
    values: { matte: patch.outputShape ?? [1, 1, 1024, 1024] },
  });

const birefnetConfig = parseBirefnetPipelineConfig(PIPELINE_CONFIG);

Deno.test("assertStaticDim: 宣言どおりのグラフは通り、解像度違いは名指しで落ちる", () => {
  // 2048² の系列を 1024² の席へ置いた形。前処理は宣言の寸法へ resize するのでホスト側は
  // 最後まで通り、落ちるのは Session の shape 検査 = どちらが正しいか読めない。
  assertStaticDim(matteGraph(), "pixel_values", 3, birefnetConfig.imageWidth, "imageWidth");
  const error = assertThrows(
    () =>
      assertStaticDim(
        matteGraph({ inputShape: [1, 3, 2048, 2048] }),
        "pixel_values",
        3,
        birefnetConfig.imageWidth,
        "imageWidth",
      ),
    Error,
    "birefnet: imageWidth — グラフ入力 'pixel_values' の軸 3 が 2048",
  );
  assert(error.message.includes("pipelineConfig は 1024"), error.message);
});

Deno.test("assertStaticDim: 入力名そのものが無ければ落とす", () => {
  assertThrows(
    () => assertStaticDim(matteGraph(), "pixel", 3, birefnetConfig.imageWidth, "imageWidth"),
    Error,
    "birefnet: グラフ入力 'pixel' が無い（imageWidth）",
  );
});

Deno.test("assertMatteShape: multi-scale で焼かれた出力は要素数でなく形で落ちる", () => {
  // `[1, 3, S, S]` は中間予測込みの export。要素数だけを見る門だと**別の値を α として**通る。
  assertMatteShape(matteGraph(), birefnetConfig, "マットの形");
  assertThrows(
    () =>
      assertMatteShape(
        matteGraph({ outputShape: [1, 3, 1024, 1024] }),
        birefnetConfig,
        "マットの形",
      ),
    Error,
    "期待は [1, 1, 1024, 1024]",
  );
});

Deno.test("assertMatteShape: 先頭が一致する短い宣言も落とす（軸ごとの比較だけでは通る）", () => {
  // `[1, 1, 1024]` は宣言されている全軸が期待と一致するので、`shape.length` を見ない門は
  // 素通しする（階数まで見る理由そのもの）。
  assertThrows(
    () => assertMatteShape(matteGraph({ outputShape: [1, 1, 1024] }), birefnetConfig, "マットの形"),
    Error,
    "グラフ出力 'matte' の形が [1, 1, 1024]",
  );
});

Deno.test("assertMatteShape: 出力名の宣言が values に無ければ落とす", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 1024, 1024] }],
    outputs: ["matte"],
    values: {},
  });
  assertThrows(
    () => assertMatteShape(model, birefnetConfig, "マットの形"),
    Error,
    "birefnet: グラフ出力 'matte' の宣言が無い（マットの形）",
  );
});

// ---- 前処理の結線（Python 正本のフィクスチャに対して）------------------------

type ParityCase = {
  readonly name: string;
  readonly height: number;
  readonly width: number;
  readonly outHeight: number;
  readonly outWidth: number;
  readonly input: number[];
  readonly pixelValues: number[];
};

type Fixture = {
  readonly constants: {
    readonly imageMean: [number, number, number];
    readonly imageStd: [number, number, number];
  };
  readonly cases: ParityCase[];
};

const FIXTURE_PATH = new URL("./fixtures/image-preprocess/parity.json", import.meta.url);
const fixture = JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture;

/** 縦は 4 倍縮小・横は 3 倍拡大の非正方ケース（24×5 → 6×15）。 */
const MIXED_AXIS = "mixed-axis";

/** 全経路の許容差。導出は `image_preprocess_test.ts` 冒頭（uint8 1 LSB の f32 換算）。 */
const PIXEL_ATOL = 7.85e-3;

const parityCase = (name: string): ParityCase => {
  const found = fixture.cases.find((entry) => entry.name === name);
  assert(found !== undefined, `フィクスチャに '${name}' が無い（生成台本が変わった）`);
  return found;
};

const imageOf = (entry: ParityCase): Rgb8Image => ({
  data: Uint8Array.from(entry.input),
  width: entry.width,
  height: entry.height,
});

/** そのケースの寸法・定数をそのまま宣言した `pipelineConfig`。 */
const configFor = (entry: ParityCase, swapped = false) =>
  parseBirefnetPipelineConfig({
    imageWidth: swapped ? entry.outHeight : entry.outWidth,
    imageHeight: swapped ? entry.outWidth : entry.outHeight,
    imageMean: fixture.constants.imageMean,
    imageStd: fixture.constants.imageStd,
    interpolation: "bilinear",
  });

Deno.test("前処理の結線: 宣言の寸法・mean・std で Python 正本と一致する（非正方）", () => {
  const entry = parityCase(MIXED_AXIS);
  const got = preprocessPixelValues(configFor(entry), imageOf(entry));
  assertEquals(got.length, entry.pixelValues.length);
  for (let index = 0; index < got.length; index += 1) {
    assertAlmostEquals(got[index], entry.pixelValues[index], PIXEL_ATOL, `標本 ${index}`);
  }
});

Deno.test("前処理の結線: imageWidth / imageHeight を入れ替えると別の結果になる", () => {
  // 上のケースが恒真でないことの担保 — 非正方の宣言を転置すると値が変わる
  // （`resizeRgb8` は `(width, height)` の順で受ける）。
  const entry = parityCase(MIXED_AXIS);
  assert(entry.outWidth !== entry.outHeight, "非正方のケースでないと転置が検出できない");
  const swapped = preprocessPixelValues(configFor(entry, true), imageOf(entry));
  assertEquals(swapped.length, entry.pixelValues.length, "要素数は転置しても同じ");
  let matched = 0;
  for (let index = 0; index < swapped.length; index += 1) {
    if (Math.abs(swapped[index] - entry.pixelValues[index]) <= PIXEL_ATOL) matched += 1;
  }
  assert(matched < swapped.length, "転置しても全標本が一致した（門が効いていない）");
});

// ---- 後処理（logit → sigmoid → 元解像度 → 8bit α）---------------------------

/** 8bit へ落ちたあとの 1 LSB。 */
const ALPHA_LSB = 1;

Deno.test("後処理: 恒等 resize では sigmoid をそのまま 8bit へ量子化する", () => {
  // logit 0 → 0.5 → 127.5 → 最近傍へ丸めて 128（切り捨てだと 127 になる — 上流の
  // `astype("uint8")` ではなく丸めを採る決定が値に出る唯一の点）。
  const matte = matteFromLogits(Float32Array.from([0, 40, -40, 1]), 2, 2, 2, 2);
  assertEquals(matte.width, 2);
  assertEquals(matte.height, 2);
  assertEquals([...matte.data], [128, 255, 0, Math.round((1 / (1 + Math.E ** -1)) * 255)]);
});

Deno.test("後処理: 返るのは**入力画像の**寸法（グラフの寸法ではない）", () => {
  const matte = matteFromLogits(new Float32Array(4 * 4), 4, 4, 7, 3);
  assertEquals(matte.width, 7);
  assertEquals(matte.height, 3);
  assertEquals(matte.data.length, 21);
});

Deno.test("後処理: sigmoid は resize の**前**に掛かる（順序を入れ替えると落ちる）", () => {
  // 2×1 → 1×1 の縮小は台が [0.5, 0.5] に縮むので、resize は 2 点の平均そのもの。
  //   sigmoid 先: (σ(0) + σ(10)) / 2 = (0.5 + 0.9999546) / 2 = 0.7499773 → 191
  //   resize 先 : σ((0 + 10) / 2) = σ(5) = 0.9933 → 253
  // 62 も離れるので、順序を取り違えたまま「それらしい」マットが出ることはない。
  const matte = matteFromLogits(Float32Array.from([0, 10]), 2, 1, 1, 1);
  assertEquals([...matte.data], [191]);
});

Deno.test("resizePlaneF32: 同寸法なら値をそのまま通す（丸めも縮退も挟まない）", () => {
  const plane = Float32Array.from([0.1, 0.25, 0.5, 0.75, 0.9, 1]);
  const out = resizePlaneF32(plane, 3, 2, 3, 2);
  assertEquals([...out], [...plane]);
});

Deno.test("resizePlaneF32: 一様な平面は寸法を変えても一様（重み総和が 1）", () => {
  const plane = new Float32Array(5 * 7).fill(0.375);
  for (const [width, height] of [[3, 2], [11, 13], [5, 7]] as const) {
    const out = resizePlaneF32(plane, 5, 7, width, height);
    assertEquals(out.length, width * height);
    for (const value of out) assertAlmostEquals(value, 0.375, 1e-6);
  }
});

Deno.test("resizePlaneF32: 縮小の重みが Pillow の台と一致する（手計算との突合）", () => {
  // 4 → 2 の縮小: scale = 2 なので台が 2 倍に伸び、出力 0 の重みは
  // triangle(-0.25) / triangle(0.25) / triangle(0.75) = 0.75 / 0.75 / 0.25 を総和 1.75 で
  // 正規化したもの。出力 1 は左右反転した並び。**非対称な入力**でないと反転を検出できない。
  const out = resizePlaneF32(Float32Array.from([0, 0, 0, 1]), 4, 1, 2, 1);
  assertAlmostEquals(out[0], 0, 1e-7);
  assertAlmostEquals(out[1], 0.75 / 1.75, 1e-7);
});

Deno.test("resizePlaneF32: 横方向だけの勾配は resize しても横方向だけ（軸の取り違え検出）", () => {
  const width = 6;
  const height = 4;
  const plane = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) plane[y * width + x] = x / (width - 1);
  }
  const out = resizePlaneF32(plane, width, height, 3, 9);
  for (let y = 1; y < 9; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      assertAlmostEquals(out[y * 3 + x], out[x], 1e-7, `行 ${y} 列 ${x}`);
    }
  }
  assert(out[0] < out[2], "横方向の単調性が失われた（軸が入れ替わっている）");
});

Deno.test("resizePlaneF32: 長さの食い違いは落とす（黙って途中まで読まない）", () => {
  assertThrows(() => resizePlaneF32(new Float32Array(5), 3, 2, 3, 2), Error, "平面の長さ 5");
  assertThrows(() => resizePlaneF32(new Float32Array(6), 3, 2, 0, 2), RangeError, "出力幅");
});

Deno.test("後処理: α は 8bit の全域を使う（飽和側が 1 LSB で潰れていない）", () => {
  // 恒真化の防止 — 上のケースが全て 128 付近でも通ってしまう形を避ける。
  const matte = matteFromLogits(Float32Array.from([-20, -2, 2, 20]), 4, 1, 4, 1);
  assertEquals(matte.data[0], 0);
  assertEquals(matte.data[3], 255);
  assert(matte.data[1] > 0 && matte.data[1] < 128 - ALPHA_LSB, "σ(−2) が中間に来ていない");
  assert(matte.data[2] > 128 + ALPHA_LSB && matte.data[2] < 255, "σ(2) が中間に来ていない");
});

Deno.test("後処理: 非有限の logit は画素座標つきで落ちる（NaN が透明画素に化けない）", () => {
  // `Uint8Array` 代入は NaN を 0 = 完全に透明な画素へ黙って丸めるので、検査が無いと
  // 「正常に見えるマット」が返る。±Inf は σ を通れば 0/1 に落ちるが、そこまで来ている時点で
  // グラフが壊れているので同じく落とす（他 3 家族の量子化出口と同じ綴り）。
  assertThrows(
    () => matteFromLogits(Float32Array.of(0, NaN, 0, 0), 2, 2, 2, 2),
    Error,
    "マットの logit (x=1, y=0) が非有限値",
  );
  assertThrows(
    () => matteFromLogits(Float32Array.of(0, 0, 0, Infinity), 2, 2, 2, 2),
    Error,
    "マットの logit (x=1, y=1) が非有限値",
  );
  assertThrows(
    () => matteFromLogits(Float32Array.of(-Infinity, 0, 0, 0), 2, 2, 2, 2),
    Error,
    "マットの logit (x=0, y=0) が非有限値",
  );
});
