// `Siglip2Pipeline` の**構築ガード**と**前処理の結線**。GPU も実資産も要らない範囲だけを見る
// （実 GPU の突合は `packages/runtime/tests/e2e_siglip2_test.ts` が持つ — 重複させない）。
//
// 押さえるのは 3 点:
//
// ① `fromAssets` は **manifest の契約違反を、資産を開く前・GPU を取りに行く前**に落とす
//    （`src/siglip2/pipeline.ts` の `openSiglip2State` が掲げる MUST）。観測の仕掛けは
//    irodori 側と同じ — **全ケースで `assets` は空**にしておき、
//     - 契約違反ケースが「その違反の文言」で落ちる = 資産解析より前に落ちている
//     - 正しい manifest + 空 assets が `資産 'vision' が無い` で落ちる = 契約検査が全部済んだ
//       後に初めて資産へ触る（上の対偶）
//    の 2 つで門の順序そのものを縛る。
//
// ② グラフ宣言との突合（`assertStaticDim` / `assertOutputDim`）の**拒否経路**。ここは
//    `fromAssets` の中では実 GPU と実資産が揃わないと踏めないので、門を直接叩く
//    （`tests/helpers/stub-model.ts` が宣言だけの `KarumeModel` を組む）。門自身の綴りや
//    軸番号がずれても、正常系だけを走らせている限り緑のまま通るため、**壊れた宣言を名指しで
//    落とすこと**を毎回踏む。
//
// ③ 前処理の結線（`preprocessPixelValues`）が `pipelineConfig` の宣言どおりに
//    `resizeRgb8` → `normalizeToNchw` を通す。**参照は Python 正本のフィクスチャ**
//    （`fixtures/image-preprocess/parity.json` — 前処理層そのものの門は
//    `image_preprocess_test.ts`）なので、ここが見るのは「宣言の 6 欄が正しい引数の位置へ
//    届いているか」だけ。非正方のケースを使うのは、`imageWidth` / `imageHeight` の取り違えが
//    正方形では**原理的に検出できない**ため。

import { assert, assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseSiglip2PipelineConfig } from "../src/siglip2/config.ts";
import {
  assertOutputDim,
  assertStaticDim,
  preprocessPixelValues,
  Siglip2Pipeline,
} from "../src/siglip2/pipeline.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";

const FILE = {
  path: "vision/model.f32.safetensors",
  size: 16,
  sha256: "a".repeat(64),
};

/** `models/karume-siglip2-base/karume.json` の `pipelineConfig` 実物（6 欄）。 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 224,
  imageHeight: 224,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  hiddenDim: 768,
  interpolation: "bilinear",
};

/** 配布形の骨格（検査に要る欄だけ）。`patch` は `models["base"]` の中身を上書きする。 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/4",
    generator: "karume/0.1.0",
    defaultModel: "base",
    models: {
      base: {
        pipeline: "siglip2/1",
        weights: { vision: { f32: { shards: [FILE] } } },
        // 実行に要るのはグラフ 1 本だけ（tokenizer も表も無い）。
        assets: {},
        quants: { f32: { weights: { vision: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/** `pipelineConfig` だけを差し替えた manifest（残りは骨格のまま）。 */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

Deno.test("fromAssets: 存在しない model は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }, { model: "nope" }),
    Error,
    "model 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipeline の契約名が siglip2 でない manifest を落とす", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/1" }));
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "manifest の pipeline が 'anima/1'",
  );
});

Deno.test("fromAssets: 未知 major は fail loudly（検査責務は models 側 — ADR 0038 §1）", async () => {
  // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門。hub は major を検査しない。
  const manifest = parseManifest(manifestText({ pipeline: "siglip2/2" }));
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "major に未対応",
  );
});

Deno.test("fromAssets: 存在しない quant は利用可能な一覧を添えて落とす", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }, { quant: "nope" }),
    Error,
    "quant 'nope' は manifest に無い",
  );
});

Deno.test("fromAssets: pipelineConfig の未知キーは構築時に落ちる", async () => {
  // 綴り違い（`imageMean` に対する `image_mean`）が黙って既定へ縮退する経路を作らない。
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, image_mean: [0.5, 0.5, 0.5] }));
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig: 未知キー 'image_mean'",
  );
});

Deno.test("fromAssets: pipelineConfig の欄が欠けていれば構築時に落ちる", async () => {
  // 欠けた欄が既定で埋まると、ホストだけが別の数を持ったまま前処理が通る（config.ts の MUST）。
  const { hiddenDim: _dropped, ...missing } = PIPELINE_CONFIG;
  const manifest = parseManifest(withConfig(missing));
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.hiddenDim: 無い",
  );
});

Deno.test("fromAssets: bilinear 以外の補間を宣言した配布形は受理しない", async () => {
  // 前処理層が持つのは antialias 付き bilinear だけ。bicubic を黙って bilinear で通すと
  // resize の値が最大 47/255 ずれたまま実行される（config.ts の MUST）。
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, interpolation: "bicubic" }));
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "pipelineConfig.interpolation: この実装が対応するのは 'bilinear' だけ",
  );
});

Deno.test("fromAssets: manifest 契約を全て満たして初めて資産へ触る（門の順序の対偶）", async () => {
  // 上の 7 ケースが「資産が空でも manifest の文言で落ちる」ことの裏返し。正しい manifest なら
  // 検査は資産まで進み、`vision` の不在で落ちる（= 契約検査は全て資産より前）。
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Siglip2Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "資産 'vision' が無い",
  );
});

// ---- pipelineConfig のスキーマ（門を直接叩く）--------------------------------

Deno.test("parseSiglip2PipelineConfig: 実物の 6 欄をそのまま読む", () => {
  const config = parseSiglip2PipelineConfig(PIPELINE_CONFIG);
  assertEquals(config.imageWidth, 224);
  assertEquals(config.imageHeight, 224);
  assertEquals(config.imageMean, [0.5, 0.5, 0.5]);
  assertEquals(config.imageStd, [0.5, 0.5, 0.5]);
  assertEquals(config.hiddenDim, 768);
  assertEquals(config.interpolation, "bilinear");
});

Deno.test("parseSiglip2PipelineConfig: std に 0 があれば落ちる（0 除算は例外を出さない）", () => {
  // `(x − mean·255) / (std·255)` の除数なので、0 は ±Infinity の pixel_values を静かに作る。
  assertThrows(
    () => parseSiglip2PipelineConfig({ ...PIPELINE_CONFIG, imageStd: [0.5, 0, 0.5] }),
    Error,
    "pipelineConfig.imageStd: 正の有限数でない要素がある",
  );
});

Deno.test("parseSiglip2PipelineConfig: mean / std の要素数が 3 でなければ落ちる", () => {
  assertThrows(
    () => parseSiglip2PipelineConfig({ ...PIPELINE_CONFIG, imageMean: [0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean: 長さ 3 の配列でない",
  );
});

// ---- グラフ宣言との突合（拒否経路）------------------------------------------

/** 実配布形（base）と同じ宣言。`patch` で 1 点だけ壊す。 */
const visionGraph = (
  patch: { readonly inputShape?: readonly (number | string)[]; readonly outputShape?: number[] } =
    {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 224, 224] }],
    outputs: ["pooler_output"],
    values: { pooler_output: patch.outputShape ?? [1, 768] },
  });

Deno.test("assertStaticDim: 宣言どおりのグラフは通り、解像度違いは名指しで落ちる", () => {
  // so400m（384²）の資産を base（224²）の席へ置いた形。前処理は宣言の寸法へ resize するので
  // ホスト側は最後まで通り、落ちるのは Session の shape 検査 = どちらが正しいか読めない。
  assertStaticDim(visionGraph(), "pixel_values", 2, 224, "imageHeight");
  const error = assertThrows(
    () =>
      assertStaticDim(
        visionGraph({ inputShape: [1, 3, 384, 384] }),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "siglip2: imageHeight — グラフ入力 'pixel_values' の軸 2 が 384",
  );
  assert(error.message.includes("pipelineConfig は 224"), error.message);
});

Deno.test("assertStaticDim: 入力名そのものが無ければ落とす", () => {
  assertThrows(
    () => assertStaticDim(visionGraph(), "pixel", 2, 224, "imageHeight"),
    Error,
    "siglip2: グラフ入力 'pixel' が無い（imageHeight）",
  );
});

Deno.test("assertOutputDim: 埋め込み次元の宣言違いは名指しで落ちる", () => {
  // hiddenDim は前処理にも実行にも使われないので、ずれても埋め込みは何事もなく出る。
  assertOutputDim(visionGraph(), 1, 768, "hiddenDim");
  assertThrows(
    () => assertOutputDim(visionGraph({ outputShape: [1, 1152] }), 1, 768, "hiddenDim"),
    Error,
    "siglip2: hiddenDim — グラフ出力 'pooler_output' の軸 1 が 1152",
  );
});

Deno.test("assertOutputDim: 出力名の宣言が values に無ければ落とす", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 224, 224] }],
    outputs: ["pooler_output"],
    values: {},
  });
  assertThrows(
    () => assertOutputDim(model, 1, 768, "hiddenDim"),
    Error,
    "siglip2: グラフ出力 'pooler_output' の宣言が無い（hiddenDim）",
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

/** そのケースの寸法・定数をそのまま宣言した `pipelineConfig`（hiddenDim は結線に効かない）。 */
const configFor = (entry: ParityCase, swapped = false) =>
  parseSiglip2PipelineConfig({
    imageWidth: swapped ? entry.outHeight : entry.outWidth,
    imageHeight: swapped ? entry.outWidth : entry.outHeight,
    imageMean: fixture.constants.imageMean,
    imageStd: fixture.constants.imageStd,
    hiddenDim: 768,
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
  // 上のケースが恒真でないことの担保 — 非正方の宣言を転置すると要素数から変わる
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
