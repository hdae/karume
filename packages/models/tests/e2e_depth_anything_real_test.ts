// Depth Anything V2 の**実画像**の門 —— TS 前処理層（`src/image/preprocess.ts`・**bicubic**）と
// PNG 符号化（`src/image/png.ts`）を通す 2 つの主張を持つ。
//
// ## この門が受け持つ主張と、runtime 側との分担
//
// もともとは `packages/runtime/tests/e2e_depth_anything_test.ts` に同居していた（TS 前処理を
// 通した `pixel_values` で推論し、入力側と出力側を 1 本のテストで見ていた）。それを 2 つに
// 割った —— SigLIP2 で先に踏んだのと同じ形（`image_preprocess_real_test.ts` 冒頭）:
//
//  ・**入力側（ここ）**: 「同じ PNG から `DPTImageProcessor` と同じ `pixel_values` が出る」——
//    GPU もモデル本体（99MB）も要らない、前処理層だけの純粋なパリティ。
//  ・**出力側（runtime）**: golden に焼かれた `pixel_values` を入力にした実 GPU 忠実度。
//    あちらの golden 突合は**合成 4 + 実画像 4 の全 8 ケース**を同じ `GOLDEN_TOLERANCE` で
//    回しており、実画像の分布点での忠実度はそこが持つ。
//
// 「PNG を渡したら同じ深度地図が返る」という鎖の主張は、この 2 つの**合成**で持つ（ここが
// 入力差 ≤ {@link PIXEL_ATOL} = 1 LSB を保証し、runtime 側がその入力での忠実度を保証する）。
// 割った理由は依存方向 —— runtime のテストが models の実装を相対 import していた（逆向き）。
//
// MUST: DA-V2 の resize は **bicubic**（`preprocessor_config.json` の `"resample": 3`）。
// SigLIP2 / BiRefNet の bilinear（2）と**違う**ので、{@link IMAGE_FILTER} を渡さずに既定へ
// 落ちると `pixel_values` が最大 0.59 ずれる（1 LSB = 0.0175 の 34 倍 — 実測）。取り違えは
// 下のパリティ門が名指しで落とす。
//
// ## 意味の判別と成果物 PNG はここに残る（golden を経由しない門だから）
//
// 構図から言える近い領域 > 遠い領域、という**判別**は golden のテンソルを 1 本も読まない
// （読むのはモデル本体と PNG だけ）ので、割りようが無い。TS 前処理を通した入力のまま**逐語で**
// こちらへ移した —— 「PNG を渡したら意味のある深度が返る」という鎖の主張は、合成に頼らず
// ここが直接持つ。深度 PNG を書き出すのも同じ理由でここ（`encodePng` は models の実装）。
//
// ## 資産が無い環境
//
// golden（`outputs/series/`）も入力の実画像（`outputs/misc/corpus/`）も深度 PNG の書き出し先
// （`outputs/bench/`）もリポジトリ管理外。ただし性格は割れていて、**読み**の corpus はホスト
// 資産（消すと台本での焼き直しと凍結コピーが要る）、**書き**の bench は `rm -rf` で常に安全に
// 消せる席である。**1 件も無ければ明示 SKIP**、**golden が中途半端に欠けていれば FAIL**
// （欠けの FAIL は runtime 側の「資産の完全性」テストが名指しで出す —— 系列ディレクトリの
// 列挙はあちらが持つ）。
// ADR 0005 の「全 SKIP は明示 FAIL」門番は *GPU アダプタの有無* だけを見ており、この SKIP とは
// 独立。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, type PreparedModel, prepareModel } from "@karume/runtime";
import { normalizeToNchw, type ResampleFilter, resizeRgb8 } from "../src/image/preprocess.ts";
import { encodePng } from "../src/image/png.ts";
// NOTE: PNG デコーダは karume 本体が持たない設計（preprocess.ts のモジュール doc —— 入口は
// RGB8 の画素列）。テストが実画像を読むためだけの道具を runtime のテスト配下から借りる。
// models → runtime は**正方向**の依存で、かつ両者とも `tests/` は publish から除外される
// （packages/*/deno.json の `publish.exclude`）ので、配布物には影響しない。
import { decodePng } from "../../runtime/tests/helpers/png-decode.ts";
import { readShard, resolveShards, streamShards } from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * 実画像ケースの**入力側**（TS 前処理 対 `DPTImageProcessor`）の許容誤差。
 *
 * 実画像は 1024² で系列は 518² なので、BiRefNet と違って **resize が恒等にならない**。残るのは
 * ①縮小の重み積算の精度（参照は f32 / こちらは f64）②パス間 uint8 丸めの境界、の 2 つで、
 * どちらも **uint8 の 1 LSB** に収まる。正規化後の 1 LSB は `1 / (std · 255)` で、最大は
 * 緑チャネルの `1 / (0.224 · 255) = 1.7507e-2`。実測は 4 ケースとも maxAbs **1.7507e-2**
 * ちょうど（= 1 LSB を 1 度も超えない）。
 *
 * atol 1.8e-2 は 1 LSB の約 1.03 倍。**2 LSB（3.5e-2）は通さない**幅に留めてある。
 *
 * 上限だけだと「全画素が 1 ずつずれる実装」（丸めの取り違え・フィルタの取り違えで偶然近い
 * 場合）が通るので、{@link PIXEL_MISMATCH_RATE_CAP} と対で使う。
 *
 * この突合を出力側と分けて持つのは、落ちたときに前処理と推論のどちらが動いたのかを分ける
 * ため（出力側だけだと、統計の取り違えと実装バグが同じ「大きい maxAbs」に見える）。
 * 統計の取り違え（SigLIP2 の mean = std = 0.5 を当てる / RGB の順序違い）も、フィルタの
 * 取り違え（bilinear を当てる — 実測 maxAbs 0.593）も、ここで 1 桁以上上に出る。
 *
 * NOTE: 判定は `|got − expected| ≤ atol`（rtol は 0 だった）に**非有限の不合格**を足したもの
 * —— runtime の `compareTensors` が rtol 0 で行う判定と同じ式で、NaN / ±Inf が「差 0」に化けて
 * 通る穴も同じように塞いである。
 */
const PIXEL_ATOL = 1.8e-2;

/**
 * `pixel_values` が参照と食い違ってよい標本の割合（{@link PIXEL_ATOL} と対）。
 *
 * 「食い違い」は差が {@link PIXEL_MISMATCH_EPSILON} を超えた標本。実測は 4 ケースで
 * 0.425% / 0.448% / 0.474% / 0.504%（各 804,972 標本）。上限 2% は実測最悪の約 4 倍。
 */
const PIXEL_MISMATCH_RATE_CAP = 0.02;

/**
 * 「食い違い」と数える差の下限。uint8 の 1 LSB（1.75e-2）の 1/17 で、f32 の 1 ULP
 * （\|x\| ≈ 2 で 2.4e-7）の 4 桁上 — 丸めの最終ビットだけが違う標本を数えないための境目。
 * 参照は f32 で、こちらは f64 で積んでから f32 へ落とすので、値が一致していても最終ビットは
 * 半数近くで動く（実測: この下限を 0 にすると相違率が 50〜67% になる）。
 */
const PIXEL_MISMATCH_EPSILON = 1e-3;

/** `outputs/series/` 直下のディレクトリ名（`depth_anything.export.default_out_dir` の綴り）。 */
const SERIES_NAME = "depth-anything-v2-small-hf";

/** SKIP 時にそのまま貼れる生成コマンド（実画像 golden まで含む形）。 */
const GENERATE = "cd tools/export-recipes && uv run --group depth-anything-preprocess" +
  " python -m depth_anything.export --real-images";

/**
 * 実画像そのものを焼き直すコマンド（プロンプト / seed の正本は台本側）。台本は
 * `outputs/bench/<model>/<日付>_eval-images/` へ焼くので、採用分は {@link CORPUS_DIR} へ
 * **人手で凍結コピー**する。
 */
const IMAGE_COMMAND = "deno task demo:eval-images --source <Anima 配布形のパス>";

/** 実行日（モジュールロード時に 1 回だけ確定 — 書き出し先の日付ディレクトリに使う）。 */
const TODAY = new Date().toISOString().slice(0, 10);

const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
/** 入力の実画像コーパス（凍結コピー — ホスト資産なので消すと焼き直しが要る）。 */
const CORPUS_DIR = new URL("../../../outputs/misc/corpus/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** 深度 PNG（目視確認用の成果物）の置き場。`outputs/bench/` は消して安全な席。 */
const ARTIFACT_DIR = new URL(
  `../../../outputs/bench/depth-anything/${TODAY}_e2e-mismatch/`,
  import.meta.url,
);

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。ケース名とファイル名の正本は
 * `depth_anything/export.py` の `REAL_CASES`、画像そのもの（プロンプト / seed / 解像度）の
 * 正本は `examples/anima/eval-images.ts`。**列挙結果ではなくここで固定する** — 列挙に頼ると、
 * 一部だけ生成した環境でケースが黙って消えて「緑だが未検証」になる。
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/** 深度地図の相対矩形（0〜1 の比・上端 / 左端が 0）。 */
type Region = {
  readonly label: string;
  readonly top: number;
  readonly bottom: number;
  readonly left: number;
  readonly right: number;
};

/**
 * 実画像の判別で見る「近い領域 / 遠い領域」の対。正本は `depth_anything/export.py` の
 * `REAL_REGIONS` で、**あちらは torch 出力に、こちらは実 GPU 出力に同じ矩形を掛ける**
 * （矩形を変えるなら両方 — 片方だけ動かすと「両者一致」の主張が黙って別物になる）。
 *
 * 置いてあるのは構図から一意に言える対だけ:
 *
 * - `photo-corridor` — 画面下端中央の**床**（カメラの足元）と、廊下の**消失点**
 * - `photo-street` — 中央の**人物**（胴〜脚）と、建物の上に抜ける**空**
 * - `photo-landscape` — 手前の**草地**と、奥の**山と空**
 * - `photo-portrait` — 画面下部の**被写体**（肩）と、左端の**背景**（遠景の並木）
 */
const REAL_REGIONS: Readonly<Record<string, readonly [Region, Region]>> = {
  "photo-portrait": [
    { label: "subject", top: 0.85, bottom: 1.00, left: 0.35, right: 0.65 },
    { label: "background", top: 0.30, bottom: 0.45, left: 0.02, right: 0.12 },
  ],
  "photo-landscape": [
    { label: "foreground-ground", top: 0.92, bottom: 1.00, left: 0.30, right: 0.70 },
    { label: "mountain-sky", top: 0.05, bottom: 0.20, left: 0.35, right: 0.65 },
  ],
  "photo-corridor": [
    { label: "floor", top: 0.92, bottom: 1.00, left: 0.40, right: 0.60 },
    { label: "vanishing-point", top: 0.64, bottom: 0.76, left: 0.40, right: 0.56 },
  ],
  "photo-street": [
    { label: "person", top: 0.55, bottom: 0.85, left: 0.44, right: 0.56 },
    { label: "sky", top: 0.02, bottom: 0.12, left: 0.40, right: 0.60 },
  ],
};

/** golden の入力テンソルのキー（正本は `depth_anything/export.py` の `INPUT_NAME`）。 */
const PIXEL_INPUT_KEY = "input.pixel_values";

/** 実画像 golden の `__metadata__` の欄（正本は `depth_anything/export.py` の同名定数）。 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * 前処理の正規化定数（ImageNet 統計）。正本は同梱 `preprocessor_config.json` の `image_mean` /
 * `image_std` で、写しは `depth_anything/export.py` の `IMAGENET_MEAN` / `IMAGENET_STD`。
 * **SigLIP2（mean = std = 0.5）とは別の統計**なので共有しない。
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * 補間フィルタ。`preprocessor_config.json` の `"resample": 3` = **PIL の定数で BICUBIC**
 * （`depth_anything.export.check_processor` が emit のたびに現物から実測する）。BiRefNet /
 * SigLIP2 の bilinear とは違う — モジュール docstring の MUST。
 */
const IMAGE_FILTER: ResampleFilter = "bicubic";

/**
 * ファイルの存在。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const goldenUrl = (caseName: string): URL =>
  new URL(`${IO_PREFIX}${caseName}${IO_SUFFIX}`, SERIES_ROOT);

/** 登録時点で必要なので同期で数える（`Deno.test` の ignore 判定に使う）。 */
const GOLDEN_COUNT = REAL_CASES.filter((entry) => exists(goldenUrl(entry.name))).length;

/**
 * 実画像が 4 枚とも揃っているか（`outputs/misc/corpus/` は人手で凍結コピーする席）。
 * MUST: NotFound 以外は伝播させる（`exists` と同じ理由）。
 */
const IMAGES_PRESENT: boolean = REAL_CASES.every((entry) =>
  exists(new URL(entry.file, CORPUS_DIR))
);

/** 実画像の群。golden と画像の**両方**が揃ってはじめて実走する。 */
const REAL_AVAILABLE = GOLDEN_COUNT > 0 && IMAGES_PRESENT;

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** 実画像 1 枚のバイト列（decode 前 — sha256 の突合にも使うので生のまま持つ）。 */
const readImage = (file: string): Promise<Uint8Array<ArrayBuffer>> =>
  Deno.readFile(new URL(file, CORPUS_DIR));

/** グラフ入力の静的次元（記号次元は無い — `depth_anything/export.py` の `symbol_names=()`）。 */
const staticDim = (parsed: PreparedModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/**
 * 実画像 1 枚を **TS 側の経路**で `pixel_values` にする（PNG decode → bicubic resize →
 * normalize）。通すのは models パッケージの前処理層そのもの — この鎖こそがこの門の主張。
 */
const preprocessImage = async (
  png: Uint8Array<ArrayBuffer>,
  file: string,
  width: number,
  height: number,
): Promise<Float32Array<ArrayBuffer>> =>
  normalizeToNchw(
    resizeRgb8(await decodePng(png, file), width, height, IMAGE_FILTER),
    IMAGE_MEAN,
    IMAGE_STD,
  );

/** 参照と食い違う（差が {@link PIXEL_MISMATCH_EPSILON} を超える）標本の割合。 */
const mismatchRate = (got: Float32Array, expected: Float32Array): number => {
  assertEquals(got.length, expected.length, "pixel_values 長");
  let differing = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (Math.abs(got[index] - expected[index]) > PIXEL_MISMATCH_EPSILON) differing += 1;
  }
  return differing / expected.length;
};

/** 相対矩形で切った深度の平均（判別はこの量の順序だけを見る）。 */
const regionMean = (depth: Float32Array, size: number, region: Region): number => {
  const top = Math.trunc(region.top * size);
  const bottom = Math.trunc(region.bottom * size);
  const left = Math.trunc(region.left * size);
  const right = Math.trunc(region.right * size);
  let sum = 0;
  let count = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      sum += depth[y * size + x];
      count += 1;
    }
  }
  assert(count > 0, `領域 ${region.label} が ${size}×${size} の地図で空になった`);
  return sum / count;
};

/**
 * 深度用の疑似カラーマップ（遠 = 濃紺 → 近 = 淡黄）の節点。**グレースケールを採らない**のは、
 * 相対深度が遠景側に密集するため（実測: 空や消失点の平地は厳密に 0 で、値域の上端は被写体
 * だけが取る）。明度 1 本では遠景が黒へ潰れて「遠景に構造がある」のか「出力が死んでいる」の
 * かが目視で分けられない。門ではなく成果物なので、色の正確さより読みやすさを採る。
 */
const DEPTH_COLORS: readonly (readonly [number, number, number])[] = [
  [12, 16, 48],
  [58, 26, 110],
  [150, 44, 96],
  [232, 108, 48],
  [252, 236, 160],
];

/** 深度地図（min-max 正規化 → {@link DEPTH_COLORS} で着色）の PNG バイト列。 */
const depthToPng = (
  depth: Float32Array,
  size: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of depth) {
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  const span = maximum - minimum;
  const last = DEPTH_COLORS.length - 1;
  const rgba = new Uint8Array(size * size * 4);
  for (let index = 0; index < depth.length; index += 1) {
    const normalized = span > 0 ? (depth[index] - minimum) / span : 0;
    const position = normalized * last;
    const anchor = Math.min(last - 1, Math.floor(position));
    const weight = position - anchor;
    for (let channel = 0; channel < 3; channel += 1) {
      const low = DEPTH_COLORS[anchor][channel];
      const high = DEPTH_COLORS[anchor + 1][channel];
      rgba[index * 4 + channel] = Math.round(low + (high - low) * weight);
    }
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, size, size);
};

if (!REAL_AVAILABLE) {
  console.warn(
    `[karume] Depth Anything V2 の実画像ケースを SKIP する（golden ${GOLDEN_COUNT}/` +
      `${REAL_CASES.length} 本・画像 ${IMAGES_PRESENT ? "有" : "無"}）。` +
      `画像の生成: ${IMAGE_COMMAND} / golden の生成: ${GENERATE}`,
  );
}

for (const entry of REAL_CASES) {
  Deno.test({
    name: `Depth Anything 実画像前処理パリティ: ${entry.name}` +
      "（TS の bicubic + 畳んだ正規化 対 DPTImageProcessor）",
    ignore: !REAL_AVAILABLE,
    fn: async () => {
      const [goldenBytes, png] = await Promise.all([
        readBuffer(goldenUrl(entry.name)),
        readImage(entry.file),
      ]);
      const io = parseSafetensors(goldenBytes);

      // ① golden を採った画像と、いま読んでいる画像が同一であること。**tolerance では
      // 吸収されない差**（生成台本を回し直して golden を採り直していない）を、突合の前に
      // 名指しで落とす。ここが緩むと「別の画像どうしを比べて緑」が成立しうる。
      assertEquals(io.metadata.get(SOURCE_IMAGE_KEY), entry.file, "golden が指す元画像");
      assertEquals(
        io.metadata.get(SOURCE_SHA256_KEY),
        await sha256Hex(png),
        `${entry.file} が golden を採った画像と違う（採り直す: ${GENERATE}）`,
      );

      const view = io.tensors.get(PIXEL_INPUT_KEY);
      assert(view !== undefined, `${PIXEL_INPUT_KEY} が golden に無い`);
      assertEquals(view.dtype, "F32", `${PIXEL_INPUT_KEY} の格納 dtype`);
      assertEquals(view.shape.length, 4, `${PIXEL_INPUT_KEY} の階数`);
      const [batch, channels, height, width] = view.shape;
      assertEquals([batch, channels], [1, 3], `${PIXEL_INPUT_KEY} の先頭 2 軸`);
      const expected = new Float32Array(io.buffer, view.byteOffset, view.byteLength / 4);

      // ② TS 側の経路（PNG decode → bicubic resize → 融合正規化）。
      const got = await preprocessImage(png, entry.file, width, height);

      assertEquals(got.length, expected.length, `${entry.name}: pixel_values の長さ`);
      let maxAbs = 0;
      let worst = 0;
      let nonFinite = 0;
      for (let index = 0; index < got.length; index += 1) {
        if (!Number.isFinite(got[index]) || !Number.isFinite(expected[index])) {
          nonFinite += 1;
          continue;
        }
        const delta = Math.abs(got[index] - expected[index]);
        if (delta > maxAbs) {
          maxAbs = delta;
          worst = index;
        }
      }
      // MUST: NaN / ±Inf は不合格。素朴な差分判定だと NaN が「差 0」に化けて通る。
      assertEquals(nonFinite, 0, `${entry.name}: pixel_values に非有限の標本`);
      assert(
        maxAbs <= PIXEL_ATOL,
        `${entry.name}: pixel_values の maxAbs=${maxAbs}` +
          `（標本 ${worst}: ${got[worst]} 対 ${expected[worst]}）が ${PIXEL_ATOL} を超えた`,
      );

      // ③ 上限だけだと「全画素が 1 ずつずれる実装」が通るので、相違率とも対で見る。
      const rate = mismatchRate(got, expected);
      assert(
        rate <= PIXEL_MISMATCH_RATE_CAP,
        `${entry.name} pixel_values の相違率 ${(rate * 100).toFixed(3)}%` +
          `（上限 ${PIXEL_MISMATCH_RATE_CAP * 100}%）— 1 LSB 以内でも全面がずれるのは` +
          "フィルタか丸めの取り違え",
      );
    },
  });
}

Deno.test({
  name: "Depth Anything 実画像 判別: 構図から言える近い領域の深度が遠い領域を上回る" +
    "（深度 PNG も書く）",
  ignore: !REAL_AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    // 判別は**領域平均の順序**だけを見る（閾値は置かない — `depth_anything/export.py` の
    // `_real_sanity` と同じ形で、あちらは torch 側に掛かっている）。一様に潰れた出力も
    // 入力非依存の出力も、上下反転も、両領域の平均が並ぶか逆転するので落ちる。実測は
    // corridor 3.8544 / 0.2055・street 3.1659 / 0.0044・landscape 3.5228 / 0.0855・
    // portrait 4.9632 / 0.2736（torch 側と 3 桁一致）。
    //
    // 入力は**実画像を TS 前処理で通したもの**（golden の入力ではない）— 「PNG を渡したら
    // 意味のある深度が返る」ところまでを検査にする。ついでに深度地図を PNG で書き出す
    // （数値の門だけでは形が見えないため — 目視確認用の成果物であって、門ではない）。
    const shards = resolveShards(new URL(MODEL_FILE, SERIES_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const [outputName] = parsed.graph.outputs;
    const width = staticDim(parsed, 3);
    const height = staticDim(parsed, 2);
    // 相対矩形も PNG も一辺だけで索く（正方でなければ黙って別の場所を測ってしまう）。
    assertEquals(width, height, "領域の判別は正方形の地図を前提にする");
    const inputName = parsed.graph.inputs[0].name;
    await Deno.mkdir(ARTIFACT_DIR, { recursive: true });

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    const means = new Map<string, { near: number; far: number }>();
    try {
      // 4 枚を 1 Session で回す（重みは 99MB — 画像ごとに組み直す理由が無い）。
      for (const real of REAL_CASES) {
        const pixels = await preprocessImage(
          await readImage(real.file),
          real.file,
          width,
          height,
        );
        const output = (await session.run({
          [inputName]: { dtype: "f32", shape: [1, 3, height, width], data: pixels },
        }))[outputName];
        // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
        assert(output.dtype === "f32", `${real.name}: 深度の dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, height, width], `${real.name}: 深度地図の形`);
        const [near, far] = REAL_REGIONS[real.name];
        means.set(real.name, {
          near: regionMean(output.data, width, near),
          far: regionMean(output.data, width, far),
        });
        await Deno.writeFile(
          new URL(`${real.name}-depth.png`, ARTIFACT_DIR),
          await depthToPng(output.data, width),
        );
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }

    for (const real of REAL_CASES) {
      const measured = means.get(real.name);
      assert(measured !== undefined, `${real.name} の領域平均が無い`);
      const [near, far] = REAL_REGIONS[real.name];
      assert(
        measured.near > measured.far,
        `${real.name} の近側 ${near.label} の深度平均 ${measured.near} が` +
          ` 遠側 ${far.label} の ${measured.far} 以下 — 構図の遠近を当てられていない`,
      );
    }
    console.log(
      `[karume] Depth Anything の深度 PNG を ${ARTIFACT_DIR.pathname} へ書いた` +
        "（<ケース>-depth.png）",
    );
  },
});
