// 実重みの Depth Anything V2（単一画像の相対深度推定）の実 GPU golden E2E（ADR 0005 の段 3）。
//
// BiRefNet（tests/e2e_birefnet_test.ts）が「画素ごとの出力を持つ画像系」の**二値の地図**を
// 受け持つのに対し、こちらは同じ画素ごとの出力でも**連続値の地図**（`[1, S, S]` の相対深度）
// を受け持つ。相対深度には単位も原点も無く、意味を持つのは**大小の順序だけ**なので、門の
// 立て方も「値が合っている」（golden 突合）と「順序が構図と合っている」（判別）に分かれる。
//
// 対象は `outputs/series/depth-anything-v2-small-hf/`（重み + 焼いた定数で 99MB のためリポジ
// トリ管理外 — `.gitignore` の `outputs/`）。生成は `tools/exporter/export_depth_anything.py`
// （コマンドは {@link GENERATE} がそのまま正本）。
//
// 系列は **1 本だけ**（Small）。台本は `--model-dir` でサイズ軸を受けるが、Base / Large は
// 上流のライセンスが CC BY-NC 4.0 で重みを取得していない（Apache-2.0 は Small のみ）。
//
// 解像度も**軸ではない** — 518²（patch 14 × 37）の 1 点固定で、外れると DINOv2 の位置埋め込みが
// bicubic 補間へ落ちる（`patch_depth_anything` の ② が fail loudly にしてある）。
//
// ## golden の 2 群（合成画像 + 実画像）— どちらも残す
//
// - **合成画像 4 ケース**（`checker` / `disc` / `noise` / `ramp`）: 入力は golden に焼かれた
//   `pixel_values` そのもの（ビット同一）。`ramp` は単調な奥行き手掛かりを持つ唯一の 1 枚で、
//   幾何の判別（対角ランプとの相関が `ramp` で最大）の土台になっている。
// - **実画像 4 ケース**（`photo-*` — `outputs/demo/` の PNG）: TS 側は PNG を decode して
//   `packages/models/src/image/preprocess.ts` を通し、Python 側は同じ画像を
//   `AutoImageProcessor`（= `DPTImageProcessor`・fast 側）の**現物**に通す。つまりこの門は
//   「**TS 前処理 + karume 推論**」対「**transformers 前処理 + torch 推論**」の突合で、
//   前処理を含めた鎖が意味のある深度を出すかを見る。
//
// MUST: DA-V2 の resize は **bicubic**（`preprocessor_config.json` の `"resample": 3`）。
// SigLIP2 / BiRefNet の bilinear（2）と**違う**ので、{@link IMAGE_FILTER} を渡さずに既定へ
// 落ちると `pixel_values` が最大 0.59 ずれる（1 LSB = 0.0175 の 34 倍 — 実測）。取り違えは
// {@link PIXEL_TOLERANCE} の門が名指しで落とす。
//
// 意味の判別（構図から言える近い領域 > 遠い領域）は**実画像側だけ**が持ち、そのとき作った
// 深度地図を **PNG として `outputs/demo/depth-anything/` へ書く**（{@link ARTIFACT_DIR}）—
// 数値の門だけでは「それらしい形の深度が出ているか」が分からないため。
//
// 資産が無い環境では**明示 SKIP** する。実画像の群も独立に SKIP する（`outputs/demo/` は
// `rm -rf` で消せる席なので、画像だけ無い環境がありうる）。ADR 0005 の「全 SKIP は明示
// FAIL」門番（tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは
// 独立。逆に資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする
// （下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  type KarumeModel,
  openModel,
  parseSafetensors,
  type SafetensorsFile,
  type Tensor,
} from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { decodePng } from "./helpers/png-decode.ts";
// NOTE: 実画像ケースは models パッケージの前処理層 / PNG 符号化**そのもの**を通すのが主張の
// 中身なので、テストからだけ相対で読む（どちらも依存ゼロの純関数で、runtime の実装は何も
// 引き込まない。publish は `tests/` を除外する — packages/runtime/deno.json）。
import {
  normalizeToNchw,
  type ResampleFilter,
  resizeRgb8,
} from "../../models/src/image/preprocess.ts";
import { encodePng } from "../../models/src/image/png.ts";

/**
 * **golden の入力で回した**ときの許容誤差（合成 4 + 実画像 4 の全 8 ケース共通）。
 *
 * 実測（`atol=rtol=0` の素の突合、出力 1 本 `[1,518,518]`）:
 *
 * | ケース          | maxAbs  | maxRel  | \|ref\| 上端 |
 * | --------------- | ------- | ------- | ------------ |
 * | checker         | 4.53e-6 | 1.98e-6 | 4.048        |
 * | disc            | 6.62e-6 | 7.25e-6 | 4.438        |
 * | noise           | 3.58e-6 | 1.23e-6 | 4.212        |
 * | ramp            | 1.86e-5 | 8.78e-4 | 4.896        |
 * | photo-portrait  | 2.69e-5 | —       | 5.333        |
 * | photo-landscape | 3.96e-5 | —       | 5.143        |
 * | photo-corridor  | 8.35e-6 | —       | 5.411        |
 * | photo-street    | 3.82e-5 | —       | 4.925        |
 *
 * atol 2e-4 は実測最悪 3.96e-5（photo-landscape）の約 5.1 倍。
 *
 * **rtol は 0**。head 末尾が ReLU なので、遠景は**厳密に 0** の広い平地になる（実測の最小は
 * 8 ケースとも 0.000）。相対量を主役にすると、その平地で分母 0 の判定が発散する — 実測
 * maxRel は実画像側で 45.7 まで出るが、その要素の絶対誤差は 3.8e-5 でしかない。
 *
 * 誤差の出所は SigLIP2 / BiRefNet と同じ（fma 融合・linear / conv の縮約順序が torch と違う・
 * 超越関数の実装差）。相対量で見ると 3.96e-5 / 5.14 ≈ 7.7e-6 で、BiRefNet（≈2.8e-5）より
 * 1 桁小さい — DINOv2 + DPT は BiRefNet の decoder ほど深い飽和を持たない。
 *
 * 実装バグ（reassemble の並べ替え取り違え・DPT fusion の段順違い・upsample の軸違い）の誤差は
 * 出力の値域と同じ O(1)〜O(5) で、この閾値の 4 桁上に出る。
 *
 * **{@link CHAIN_TOLERANCE} と共有しない** — あちらは前処理の 1 LSB 差を含んだ別の主張で、
 * 共有すると片方を測り直したときにもう片方が黙って 3 桁緩む。
 */
const GOLDEN_TOLERANCE: Tolerance = { atol: 2e-4, rtol: 0 };

/**
 * **TS 前処理で作った `pixel_values` で回した**ときの許容誤差（実画像ケースのみ）。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,518,518]`）:
 *
 * | ケース          | maxAbs  | \|ref\| 上端 | 同ケースを golden 入力で回した maxAbs |
 * | --------------- | ------- | ------------ | ------------------------------------- |
 * | photo-portrait  | 2.09e-2 | 5.333        | 2.69e-5                               |
 * | photo-landscape | 2.78e-2 | 5.143        | 3.96e-5                               |
 * | photo-corridor  | 9.00e-3 | 5.411        | 8.35e-6                               |
 * | photo-street    | 1.92e-2 | 4.925        | 3.82e-5                               |
 *
 * atol 1e-1 は実測最悪 2.78e-2（photo-landscape）の約 3.6 倍。
 *
 * 右端の列が**帰属の証拠**: 同じグラフ・同じ実行経路でも、入力を golden の `pixel_values` に
 * 差し替えると誤差は 3 桁落ちる。つまりここで効いているのはランタイムの数値誤差ではなく
 * **前処理の 1 LSB 差**（{@link PIXEL_TOLERANCE}）が推論を通って増幅したぶんで、BiRefNet の
 * 実画像門が 1024² の恒等 resize でこの成分を持たなかったのとは事情が違う（DA-V2 は
 * 1024 → 518 の縮小なので、resize の丸めが必ず入る）。
 *
 * この門が緩いぶんは {@link GOLDEN_TOLERANCE} の 8 ケースが押さえている。ここが受け持つのは
 * 「PNG を渡したら**同じ深度地図**が返る」という鎖の主張で、前処理の取り違え（フィルタ /
 * 統計 / チャネル順）は O(0.1)〜O(1) の差になり、この閾値の上に出る。
 */
const CHAIN_TOLERANCE: Tolerance = { atol: 1e-1, rtol: 0 };

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
 */
const PIXEL_TOLERANCE: Tolerance = { atol: 1.8e-2, rtol: 0 };

/**
 * `pixel_values` が参照と食い違ってよい標本の割合（{@link PIXEL_TOLERANCE} と対）。
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

/** `outputs/series/` 直下のディレクトリ名（`export_depth_anything.default_out_dir` の綴り）。 */
const SERIES_NAME = "depth-anything-v2-small-hf";

/** SKIP 時にそのまま貼れる生成コマンド（実画像 golden まで含む形）。 */
const GENERATE = "cd tools/exporter && uv run --group depth-anything-preprocess" +
  " python export_depth_anything.py --real-images";

/** 実画像そのものを焼き直すコマンド（プロンプト / seed の正本は台本側）。 */
const IMAGE_COMMAND = "deno task demo:eval-images --source <Anima 配布形のパス>";

const SERIES_ROOT = new URL(`../../../outputs/series/${SERIES_NAME}/`, import.meta.url);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** 深度 PNG（目視確認用の成果物）の置き場。`outputs/` 配下なので git 追跡外。 */
const ARTIFACT_DIR = new URL("depth-anything/", DEMO_DIR);

/**
 * 生成されているはずの**合成画像**ケース。正本は `tools/exporter/export_depth_anything.py` の
 * `build_cases`（モデル軸に依らず同じ 4 枚）。
 */
const SYNTHETIC_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。ケース名とファイル名の正本は
 * `export_depth_anything.py` の `REAL_CASES`、画像そのもの（プロンプト / seed / 解像度）の
 * 正本は `examples/anima/eval-images.ts`。
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
 * 実画像の判別で見る「近い領域 / 遠い領域」の対。正本は `export_depth_anything.py` の
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

/** 実画像 golden の `__metadata__` の欄（正本は `export_depth_anything.py` の同名定数）。 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * 前処理の正規化定数（ImageNet 統計）。正本は同梱 `preprocessor_config.json` の `image_mean` /
 * `image_std` で、写しは `export_depth_anything.py` の `IMAGENET_MEAN` / `IMAGENET_STD`。
 * **SigLIP2（mean = std = 0.5）とは別の統計**なので共有しない。
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * 補間フィルタ。`preprocessor_config.json` の `"resample": 3` = **PIL の定数で BICUBIC**
 * （`export_depth_anything.check_processor` が emit のたびに現物から実測する）。BiRefNet /
 * SigLIP2 の bilinear とは違う — モジュール docstring の MUST。
 */
const IMAGE_FILTER: ResampleFilter = "bicubic";

/**
 * 幾何の判別に使うケース（対角ランプとの相関がここで最大になる）。正本は
 * `export_depth_anything.py` の `RAMP_CASE`。
 */
const RAMP_CASE = "ramp";

/**
 * 資産ディレクトリの列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, SERIES_ROOT));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** golden の入力を宣言 dtype の view で組む（記号次元が無いので明示 bindings も不要）。 */
const goldenInputs = (parsed: KarumeModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が golden に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** 実画像 1 枚のバイト列（decode 前 — sha256 の突合にも使うので生のまま持つ）。 */
const readImage = (file: string): Promise<Uint8Array<ArrayBuffer>> =>
  Deno.readFile(new URL(file, DEMO_DIR));

/**
 * 実画像が 4 枚とも揃っているか（`outputs/demo/` は `rm -rf` で消せる席）。
 * MUST: NotFound 以外は伝播させる（`listDir` と同じ理由）。
 */
const IMAGES_PRESENT: boolean = REAL_CASES.every((entry) => {
  try {
    return Deno.statSync(new URL(entry.file, DEMO_DIR)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
});

/** グラフ入力の静的次元（記号次元は無い — `export_depth_anything.py` の `symbol_names=()`）。 */
const staticDim = (parsed: KarumeModel, axis: number): number => {
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

/** 深度地図と対角ランプ座標のピアソン相関（`export_depth_anything.py` の `_ramp_correlation`）。 */
const rampCorrelation = (depth: Float32Array, size: number): number => {
  const plane = new Float64Array(depth.length);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      plane[y * size + x] = (y / (size - 1) + x / (size - 1)) / 2;
    }
  }
  let depthMean = 0;
  let planeMean = 0;
  for (let index = 0; index < depth.length; index += 1) {
    depthMean += depth[index];
    planeMean += plane[index];
  }
  depthMean /= depth.length;
  planeMean /= plane.length;
  let covariance = 0;
  let depthNorm = 0;
  let planeNorm = 0;
  for (let index = 0; index < depth.length; index += 1) {
    const left = depth[index] - depthMean;
    const right = plane[index] - planeMean;
    covariance += left * right;
    depthNorm += left * left;
    planeNorm += right * right;
  }
  return covariance / Math.sqrt(depthNorm * planeNorm);
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

const DISCOVERED = discoverCases(SERIES_ROOT);
const realNames = new Set<string>(REAL_CASES.map((entry) => entry.name));
const FOUND_SYNTHETIC = DISCOVERED.filter((name) => !realNames.has(name));
const FOUND_REAL = DISCOVERED.filter((name) => realNames.has(name));
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const AVAILABLE = DISCOVERED.length > 0;
/** 実画像の群。golden と画像の**両方**が揃ってはじめて実走する。 */
const REAL_AVAILABLE = AVAILABLE && FOUND_REAL.length > 0 && IMAGES_PRESENT;

if (!AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み Depth Anything V2 ` +
      `E2E を SKIP する（重みがリポジトリ管理外）。生成: ${GENERATE}`,
  );
} else if (!REAL_AVAILABLE) {
  console.warn(
    `[karume] Depth Anything V2 の実画像ケースを SKIP する（golden ${FOUND_REAL.length}/` +
      `${REAL_CASES.length} 本・画像 ${IMAGES_PRESENT ? "有" : "無"}）。` +
      `画像の生成: ${IMAGE_COMMAND} / golden の生成: ${GENERATE}`,
  );
}

Deno.test({
  name: "Depth Anything 資産: 期待するケースとモデル本体が揃っている",
  // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
  ignore: !AVAILABLE,
  fn: () => {
    assertEquals(
      FOUND_SYNTHETIC,
      [...SYNTHETIC_CASES],
      `${SERIES_ROOT.pathname} の合成画像 golden ケース`,
    );
    // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
    // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
    assert(
      FOUND_REAL.length === 0 || FOUND_REAL.length === REAL_CASES.length,
      `${SERIES_ROOT.pathname} の実画像 golden が ${FOUND_REAL.length}/${REAL_CASES.length} 本` +
        `（採り直す: ${GENERATE}）`,
    );
    assert(Deno.statSync(new URL(MODEL_FILE, SERIES_ROOT)).isFile, `${MODEL_FILE} が無い`);
  },
});

for (const caseName of DISCOVERED) {
  Deno.test({
    name: `Depth Anything golden 突合: ${caseName}（golden 入力 / 実 GPU 対 torch CPU 期待値）`,
    ignore: !AVAILABLE || !GPU_AVAILABLE,
    fn: async () => {
      const [modelBytes, ioBytes] = await Promise.all([
        readBuffer(MODEL_FILE),
        readBuffer(`${IO_PREFIX}${caseName}${IO_SUFFIX}`),
      ]);
      const parsed = openModel(modelBytes);
      const io = parseSafetensors(ioBytes);

      // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
      const expectedKeys = [
        ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
        ...parsed.graph.outputs.map((_, index) => `output.${index}`),
      ].sort();
      assertEquals([...io.tensors.keys()].sort(), expectedKeys, "io.safetensors のテンソルキー");

      const inputs = goldenInputs(parsed, io);

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      try {
        const outputs = await session.run(inputs);
        assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

        const [name] = parsed.graph.outputs;
        const view = io.tensors.get("output.0");
        assert(view !== undefined, "output.0 が golden に無い");
        const where = `${caseName} output.0 ('${name}')`;
        const declared = parsed.graph.values[name].dtype;
        assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
        assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
        const expected = ioTensor(io, view, declared);
        const report = compareTensors(outputs[name], expected, GOLDEN_TOLERANCE);
        assert(report.pass, `${where}: ${formatAllclose(report)}`);
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}

Deno.test({
  name: "Depth Anything 幾何判別: 対角ランプとの相関が ramp ケースで最大になる",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    // golden 突合だけだと「期待値と合っている」ことしか言えず、深度として意味のある出力かは
    // 別問題（一様に潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは**単調な
    // 奥行き手掛かりを持つ 1 枚だけが立つ**という順序を見るので、出力が入力の幾何を追えて
    // いなければ落ちる。閾値は置かない（順序そのものが検査対象 — `export_depth_anything.py`
    // の `_sanity` と同じ形で、あちらは torch 側に掛かっている）。実測は
    // ramp 0.7705 / noise 0.4128 / checker 0.0414 / disc −0.1837（torch 側と 4 桁一致）。
    const modelBytes = await readBuffer(MODEL_FILE);
    const parsed = openModel(modelBytes);
    const size = staticDim(parsed, 3);
    assertEquals(size, staticDim(parsed, 2), "相関は正方形の入力を前提にする");
    const [name] = parsed.graph.outputs;

    const gpu = await acquireGpu();
    const session = await createSession(gpu, parsed);
    const correlations = new Map<string, number>();
    try {
      for (const caseName of SYNTHETIC_CASES) {
        const io = parseSafetensors(await readBuffer(`${IO_PREFIX}${caseName}${IO_SUFFIX}`));
        const output = (await session.run(goldenInputs(parsed, io)))[name];
        assert(output.dtype === "f32", `${caseName}: 深度の dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, size, size], `${caseName}: 深度地図の形`);
        correlations.set(caseName, rampCorrelation(output.data, size));
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }

    const rampValue = correlations.get(RAMP_CASE);
    assert(rampValue !== undefined, `${RAMP_CASE} の相関が無い`);
    for (const [caseName, value] of correlations) {
      if (caseName === RAMP_CASE) continue;
      assert(
        rampValue > value,
        `対角ランプとの相関が ${caseName}=${value} で ${RAMP_CASE}=${rampValue} 以上 —` +
          " 出力が入力の幾何を追えていない",
      );
    }
  },
});

for (const real of REAL_CASES) {
  Deno.test({
    name: `Depth Anything 実画像 golden 突合: ${real.name}` +
      "（TS 前処理 + 実 GPU / DPTImageProcessor + torch CPU）",
    ignore: !REAL_AVAILABLE || !GPU_AVAILABLE,
    fn: async () => {
      const [modelBytes, ioBytes, png] = await Promise.all([
        readBuffer(MODEL_FILE),
        readBuffer(`${IO_PREFIX}${real.name}${IO_SUFFIX}`),
        readImage(real.file),
      ]);
      const parsed = openModel(modelBytes);
      const io = parseSafetensors(ioBytes);

      // ① golden を採った画像と、いま読んでいる画像が同一であること。**tolerance では
      // 吸収されない差**（生成台本を回し直して golden を採り直していない）を、実行の前に
      // 名指しで落とす。
      assertEquals(io.metadata.get(SOURCE_IMAGE_KEY), real.file, "golden が指す元画像");
      assertEquals(
        io.metadata.get(SOURCE_SHA256_KEY),
        await sha256Hex(png),
        `${real.file} が golden を採った画像と違う（採り直す: ${GENERATE}）`,
      );

      const width = staticDim(parsed, 3);
      const height = staticDim(parsed, 2);
      const pixels = await preprocessImage(png, real.file, width, height);

      // ② 前処理の突合（TS の bicubic + 畳んだ正規化 対 DPTImageProcessor）。出力側と
      // 分けて持つ理由は PIXEL_TOLERANCE の docstring。
      const inputView = io.tensors.get(`input.${parsed.graph.inputs[0].name}`);
      assert(inputView !== undefined, "golden に入力が無い");
      const expectedPixels = ioTensor(io, inputView, "f32");
      const inputReport = compareTensors(
        { dtype: "f32", data: pixels },
        expectedPixels,
        PIXEL_TOLERANCE,
      );
      assert(inputReport.pass, `${real.name} pixel_values: ${formatAllclose(inputReport)}`);
      assert(expectedPixels.dtype === "f32", "golden の pixel_values が f32 でない");
      const rate = mismatchRate(pixels, expectedPixels.data);
      assert(
        rate <= PIXEL_MISMATCH_RATE_CAP,
        `${real.name} pixel_values の相違率 ${(rate * 100).toFixed(3)}%` +
          `（上限 ${PIXEL_MISMATCH_RATE_CAP * 100}%）— 1 LSB 以内でも全面がずれるのは` +
          "フィルタか丸めの取り違え",
      );

      // ③ TS 側で作った pixel_values で回した出力を、torch の期待値と突き合わせる。
      const inputs: Record<string, Tensor> = {
        [parsed.graph.inputs[0].name]: {
          dtype: "f32",
          shape: [1, 3, height, width],
          data: pixels,
        },
      };
      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      try {
        const outputs = await session.run(inputs);
        const [name] = parsed.graph.outputs;
        const view = io.tensors.get("output.0");
        assert(view !== undefined, "output.0 が golden に無い");
        const where = `${real.name} output.0 ('${name}')`;
        const declared = parsed.graph.values[name].dtype;
        assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
        assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
        const expected = ioTensor(io, view, declared);
        const report = compareTensors(outputs[name], expected, CHAIN_TOLERANCE);
        assert(report.pass, `${where}: ${formatAllclose(report)}`);
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}

Deno.test({
  name: "Depth Anything 実画像 判別: 構図から言える近い領域の深度が遠い領域を上回る" +
    "（深度 PNG も書く）",
  ignore: !REAL_AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    // 判別は**領域平均の順序**だけを見る（閾値は置かない — `export_depth_anything.py` の
    // `_real_sanity` と同じ形で、あちらは torch 側に掛かっている）。一様に潰れた出力も
    // 入力非依存の出力も、上下反転も、両領域の平均が並ぶか逆転するので落ちる。実測は
    // corridor 3.8544 / 0.2055・street 3.1659 / 0.0044・landscape 3.5228 / 0.0855・
    // portrait 4.9632 / 0.2736（torch 側と 3 桁一致）。
    //
    // 入力は**実画像を TS 前処理で通したもの**（golden の入力ではない）— 「PNG を渡したら
    // 意味のある深度が返る」ところまでを検査にする。ついでに深度地図を PNG で書き出す
    // （数値の門だけでは形が見えないため — 目視確認用の成果物であって、門ではない）。
    const modelBytes = await readBuffer(MODEL_FILE);
    const parsed = openModel(modelBytes);
    const [outputName] = parsed.graph.outputs;
    const width = staticDim(parsed, 3);
    const height = staticDim(parsed, 2);
    // 相対矩形も PNG も一辺だけで索く（正方でなければ黙って別の場所を測ってしまう）。
    assertEquals(width, height, "領域の判別は正方形の地図を前提にする");
    const inputName = parsed.graph.inputs[0].name;
    await Deno.mkdir(ARTIFACT_DIR, { recursive: true });

    const gpu = await acquireGpu();
    const session = await createSession(gpu, parsed);
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
