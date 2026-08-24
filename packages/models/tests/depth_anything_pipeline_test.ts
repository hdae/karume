// `DepthAnythingPipeline` の**構築ガード**と**前後処理の結線**。GPU も実資産も要らない範囲だけ
// を見る（実 GPU の突合は `packages/runtime/tests/e2e_depth_anything_test.ts`〈golden 入力〉と
// `e2e_depth_anything_real_test.ts`〈実画像の前処理パリティと判別〉が持つ — 重複させない）。
//
// 押さえるのは 4 点:
//
// ① `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を取りに行く前**に落とす
//    （`src/depth-anything/pipeline.ts` の `openDepthAnythingState` が掲げる MUST）。観測の
//    仕掛けは SigLIP2 / BiRefNet 側と同じ — **全ケースで `assets` は空**にしておき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産解析より前に落ちている
//     - 正しい manifest + 空 assets が `資産 'depth' が無い` で落ちる = 契約検査が全部済んだ
//       後に初めて資産へ触る（上の対偶）
//    の 2 つで門の順序そのものを縛る。
//
// ② グラフ宣言との突合（`assertStaticDim` / `assertDepthShape`）の**拒否経路**。ここは
//    `fromAssets` の中では実 GPU と実資産が揃わないと踏めないので、門を直接叩く
//    （`tests/helpers/stub-model.ts` が宣言だけの `KarumeModel` を組む）。門自身の軸番号や
//    期待形がずれても、正常系だけを走らせている限り緑のまま通るため、**壊れた宣言を名指しで
//    落とすこと**を毎回踏む。
//
// ③ 前処理の結線（`preprocessPixelValues`）が `pipelineConfig` の宣言どおりに
//    `resizeRgb8`（**bicubic**）→ `normalizeToNchw` を通す。**Python 正本のフィクスチャは
//    bicubic 枝にまだ無い**（bilinear 枝の `fixtures/image-preprocess/parity.json` は
//    フィルタが違うので参照にならない）ので、ここは資産不要で**落ちうる**不変条件だけを見る:
//     - 一様な画像は「宣言の定数がそのチャネルへ届いているか」を閉じた式で言える
//     - 非正方の宣言を転置すると値が変わる
//     - **bilinear で通した結果と一致しない**（既定へ落ちていれば一致してしまう）
//    bicubic そのものの正しさは、①`image_preprocess_test.ts` のカーネル署名テストと
//    ②実資産のある環境で毎回 `DPTImageProcessor` と突き合わせる E2E が持つ。
//
// ④ 後処理（`resampleDepth`）が**値に触らない**こと。深度を `[0, 1]` へ畳まない決定
//    （`pipeline.ts` のモジュール doc）は、畳んでも shape も非負性も合ったまま通るので、
//    「畳んだら落ちる」形の検査を置く。

import { assert, assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";
import {
  assertDepthShape,
  assertStaticDim,
  DepthAnythingPipeline,
  preprocessPixelValues,
  resampleDepth,
} from "../src/depth-anything/pipeline.ts";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";

const FILE = {
  path: "small/depth/model.f32.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/karume-depth-anything-v2-small/karume.json` の `pipelineConfig` 実物（5 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 518,
  imageHeight: 518,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["small"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/4",
    generator: "karume/0.2.2",
    defaultModel: "small",
    models: {
      small: {
        pipeline: "depth-anything/1",
        weights: { depth: { f32: { shards: [FILE] } } },
        // 実行に要るのはグラフ 1 本だけ（tokenizer も表も無い）。
        assets: {},
        quants: { f32: { weights: { depth: "f32" }, session: {} } },
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
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が depth-anything でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "birefnet/1" }));
  await assertRejects(
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "manifest の pipeline が 'birefnet/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "depth-anything/2" }));
  await assertRejects(
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "nope" }),
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
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'image_mean'",
  );
});

Deno.test("fromAssets: bicubic 以外の補間を宣言した配布形は受理しない", async () => {
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, interpolation: "bilinear" } }),
  );
  await assertRejects(
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.interpolation: この実装が対応するのは 'bicubic' だけ",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 6 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、`depth` の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => DepthAnythingPipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'depth' が無い",
  );
});

// ---- グラフ宣言との突合（拒否経路）------------------------------------------

/** 実配布形（small）と同じ宣言。`patch` で 1 点だけ壊す。 */
const depthGraph = (
  patch: {
    readonly inputShape?: readonly (number | string)[];
    readonly outputShape?: readonly (number | string)[];
  } = {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 518, 518] }],
    outputs: ["depth"],
    values: { depth: patch.outputShape ?? [1, 518, 518] },
  });

const depthConfig = parseDepthAnythingPipelineConfig(PIPELINE_CONFIG);

Deno.test("assertStaticDim: 宣言どおりのグラフは通り、事前学習解像度違いは名指しで落ちる", () => {
  // DINOv2 の位置埋め込みはパッチ数に紐づくので、ここで落ちるのは常に**資産の取り違え**。
  assertStaticDim(depthGraph(), "pixel_values", 2, depthConfig.imageHeight, "imageHeight");
  const error = assertThrows(
    () =>
      assertStaticDim(
        depthGraph({ inputShape: [1, 3, 384, 384] }),
        "pixel_values",
        2,
        depthConfig.imageHeight,
        "imageHeight",
      ),
    Error,
    "depth-anything: imageHeight — グラフ入力 'pixel_values' の軸 2 が 384",
  );
  assert(error.message.includes("pipelineConfig は 518"), error.message);
});

Deno.test("assertStaticDim: 入力名そのものが無ければ落とす", () => {
  assertThrows(
    () => assertStaticDim(depthGraph(), "pixel", 2, depthConfig.imageHeight, "imageHeight"),
    Error,
    "depth-anything: グラフ入力 'pixel' が無い（imageHeight）",
  );
});

Deno.test("assertDepthShape: 宣言どおりなら通り、チャネル軸を持ったままの出力は落ちる", () => {
  // `[1, 1, S, S]` は中間段まで出す別 export の形で、要素数では `[1, S, S]` と区別できない。
  assertDepthShape(depthGraph(), depthConfig, "深度地図の形");
  assertThrows(
    () =>
      assertDepthShape(depthGraph({ outputShape: [1, 1, 518, 518] }), depthConfig, "深度地図の形"),
    Error,
    "期待は [1, 518, 518]",
  );
});

Deno.test("assertDepthShape: 先頭が一致する短い宣言も落とす（軸ごとの比較だけでは通る）", () => {
  // `[1, 518]` は宣言されている全軸が期待と一致するので、`shape.length` を見ない門は素通しする。
  assertThrows(
    () => assertDepthShape(depthGraph({ outputShape: [1, 518] }), depthConfig, "深度地図の形"),
    Error,
    "グラフ出力 'depth' の形が [1, 518]",
  );
});

Deno.test("assertDepthShape: 出力名の宣言が values に無ければ落とす", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 518, 518] }],
    outputs: ["depth"],
    values: {},
  });
  assertThrows(
    () => assertDepthShape(model, depthConfig, "深度地図の形"),
    Error,
    "depth-anything: グラフ出力 'depth' の宣言が無い（深度地図の形）",
  );
});

// ---- 前処理の結線（bicubic + ImageNet 統計）----------------------------------

const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/** そのケースの寸法をそのまま宣言した `pipelineConfig`（定数は配布形の実物）。 */
const configFor = (width: number, height: number) =>
  parseDepthAnythingPipelineConfig({
    imageWidth: width,
    imageHeight: height,
    imageMean: IMAGE_MEAN,
    imageStd: IMAGE_STD,
    interpolation: "bicubic",
  });

/** チャネルごとに違う値で塗りつぶした RGB8（統計の順序違いが値に出る）。 */
const solid = (
  width: number,
  height: number,
  color: readonly [number, number, number],
): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let index = 0; index < width * height; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) data[index * 3 + channel] = color[channel];
  }
  return { data, width, height };
};

/** 市松 + 斜めランプ（bicubic と bilinear が別の値を出すだけの構造を持つ）。 */
const textured = (width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 3;
      data[index] = (x + y) % 2 === 0 ? 250 : 5;
      data[index + 1] = Math.round((255 * x) / Math.max(1, width - 1));
      data[index + 2] = Math.round((255 * y) / Math.max(1, height - 1));
    }
  }
  return { data, width, height };
};

Deno.test("前処理の結線: 一様な画像は宣言の mean / std がチャネルごとに掛かる", () => {
  // resize の重みは総和 1 なので一様入力は一様のまま通り、残るのは正規化だけ。RGB を別々の
  // 値にしてあるので、統計の順序違い（BGR / mean と std の入れ替え）はここで落ちる。
  const color = [10, 200, 90] as const;
  const got = preprocessPixelValues(configFor(4, 2), solid(3, 5, color));
  assertEquals(got.length, 3 * 4 * 2);
  for (let channel = 0; channel < 3; channel += 1) {
    const expected = (color[channel] - IMAGE_MEAN[channel] * 255) / (IMAGE_STD[channel] * 255);
    for (let index = 0; index < 4 * 2; index += 1) {
      assertAlmostEquals(got[channel * 8 + index], expected, 1e-5, `チャネル ${channel}`);
    }
  }
});

Deno.test("前処理の結線: imageWidth / imageHeight を入れ替えると別の結果になる", () => {
  // 宣言の 2 欄が正しい引数の位置へ届いていることの担保（`resizeRgb8` は `(width, height)`
  // の順）。正方形では原理的に検出できないので非正方で見る。
  const image = textured(12, 9);
  const got = preprocessPixelValues(configFor(6, 4), image);
  const swapped = preprocessPixelValues(configFor(4, 6), image);
  assertEquals(got.length, swapped.length, "要素数は転置しても同じ");
  let matched = 0;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] === swapped[index]) matched += 1;
  }
  assert(matched < got.length, "転置しても全標本が一致した（門が効いていない）");
});

Deno.test("前処理の結線: bicubic で通る（既定の bilinear へ落ちていない）", () => {
  // `resizeRgb8` の `filter` 引数の既定は bilinear（既存 2 ファミリの値）なので、渡し忘れは
  // 型検査を通り、shape も値域も合ったまま `pixel_values` がずれる（実測 0.59 = uint8 1 LSB
  // の 34 倍）。同じ経路を bilinear で組んだ結果と**一致しない**ことがフィルタ選択の証拠。
  const image = textured(12, 9);
  const config = configFor(6, 4);
  const got = preprocessPixelValues(config, image);
  const bilinear = normalizeToNchw(resizeRgb8(image, 6, 4, "bilinear"), IMAGE_MEAN, IMAGE_STD);
  assertEquals(got.length, bilinear.length);
  let differing = 0;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] !== bilinear[index]) differing += 1;
  }
  assert(differing > 0, "bicubic 指定が bilinear と同じ値を出した（既定へ落ちている）");
});

// ---- 後処理（深度地図 → 元解像度・値には触らない）----------------------------

Deno.test("後処理: 恒等寸法では値をそのまま通す（丸めも正規化も挟まない）", () => {
  const depth = Float32Array.from([0, 0.25, 1.5, 4.75, 2, 0]);
  const map = resampleDepth(depth, 3, 2, 3, 2);
  assertEquals(map.width, 3);
  assertEquals(map.height, 2);
  assertEquals([...map.data], [...depth]);
});

Deno.test("後処理: 返るのは**入力画像の**寸法（グラフの寸法ではない）", () => {
  const map = resampleDepth(new Float32Array(4 * 4), 4, 4, 7, 3);
  assertEquals(map.width, 7);
  assertEquals(map.height, 3);
  assertEquals(map.data.length, 21);
});

Deno.test("後処理: 相対深度の尺度を保つ（[0, 1] へ畳まない — 公開 API の決定）", () => {
  // 正規化を足しても shape も非負性も合ったまま通るので、「畳んだら落ちる」形で固定する。
  // min-max で畳んだ地図は必ず 0 と 1 を含み、上端 4.75 は残らない。
  const depth = Float32Array.from([2, 2.5, 3, 4.75]);
  const map = resampleDepth(depth, 4, 1, 4, 1);
  assertEquals(map.data[0], 2, "下端が 0 へ落ちている（min-max 正規化が掛かっている）");
  assertEquals(map.data[3], 4.75, "上端が 1 へ落ちている（min-max 正規化が掛かっている）");
});

Deno.test("後処理: 拡大しても非負のまま（重みが非負・総和 1）", () => {
  // head 末尾の ReLU が出す「遠景は厳密に 0」という性質が、後処理で負へ潜らないこと。
  // 負のローブを持つカーネル（bicubic）へ差し替えるとここが落ちる — モジュール doc の
  // 「元解像度へ戻す段は bilinear」が値に出る唯一の点。
  const depth = Float32Array.from([0, 0, 5, 5]);
  const map = resampleDepth(depth, 4, 1, 16, 1);
  for (const value of map.data) assert(value >= 0, `負の深度 ${value} が出た`);
  assert(map.data[15] > map.data[0], "単調性が失われた（左右が入れ替わっている）");
});

Deno.test("後処理: 横方向だけの勾配は拡大しても横方向だけ（軸の取り違え検出）", () => {
  const width = 6;
  const height = 4;
  const plane = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) plane[y * width + x] = x / (width - 1);
  }
  const map = resampleDepth(plane, width, height, 3, 9);
  for (let y = 1; y < 9; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      assertAlmostEquals(map.data[y * 3 + x], map.data[x], 1e-7, `行 ${y} 列 ${x}`);
    }
  }
  assert(map.data[0] < map.data[2], "横方向の単調性が失われた（軸が入れ替わっている）");
});
