// 実重み SigLIP2 の **vision tower** の実 GPU golden E2E（ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、実重み SBV2 / Irodori が
// 「音響チェーン側の実重み」を、EmbeddingGemma（tests/e2e_embeddinggemma_test.ts）が
// 「単一ベクトル出力のテキスト系」を受け持つのに対し、こちらは**単一ベクトル出力の画像系**
// を受け持つ。対象は `outputs/series/<系列>/`（重み 350MB〜1.7GB のためリポジトリ管理外 —
// `.gitignore` の `outputs/`）。生成は `tools/exporter/export_siglip2.py`（コマンドは下の
// generateCommand がそのまま正本）。
//
// 系列は **モデルごとに 1 本**（下の SERIES）。base（`patch16-224` — hidden 768 / 12 層 /
// 196 パッチ）と so400m（`patch14-384` — hidden 1152 / 27 層 / 729 パッチ）は同じ経路の
// 大小 2 点で、グラフの形も op 表も同じ。**tolerance だけは系列ごとに実測から導く**
// （片方の実測で他方を通さない — 桁が違えば誤差も違う）。格納 dtype 系列は f32 の 1 本
// のみなので、そちらの軸は持たない。
//
// グラフは vision tower 1 本で、出力も **pooler_output（MAP head 経由の `[1,hidden]`）
// 1 本だけ**（text tower も `last_hidden_state` も載っていない — export_siglip2.py の
// docstring）。入力は**正規化済みの** `pixel_values f32 [1,3,解像度,解像度]` 1 本で、記号
// 次元は無い（解像度もパッチ数も固定）。
//
// ## golden の 2 群（合成画像 + 実画像）— どちらも残す
//
// - **合成画像 4 ケース**（`ramp` / `ramp-dim` / `checker` / `noise`）: 入力は golden に
//   焼かれた `pixel_values` そのもの（ビット同一）なので、突合に出るのは**ランタイムの
//   数値誤差だけ**。値域の端や勾配を踏むぶん数値回帰の検出が鋭い（実測 1e-5 級）。
// - **実画像 4 ケース**（`photo-*` — `outputs/demo/` の PNG）: TS 側は PNG を decode して
//   `packages/models/src/image/preprocess.ts` を通し、Python 側は同じ画像を transformers の
//   `TorchvisionBackend` に通す。つまりこの門は「**TS 前処理 + karume 推論**」対「**Python
//   前処理 + torch 推論**」の突合で、前処理を含めた鎖が意味のある出力を出すかを見る。
//   数値は穏やか（1 LSB の入力差が伝播するので tolerance は 3 桁緩い —
//   {@link REAL_BASE_TOLERANCE}）で、鋭さは合成側が受け持つ。
//
// 判別（cosine の順序）は**実画像側だけ**が持つ（合成画像の ramp / checker より意味のある
// 判別になる — {@link REAL_PERSON_CASES}）。
//
// 資産が無い環境では**明示 SKIP**する（系列ごとに独立 — 片方だけ生成した環境でも、ある方は
// 実走する）。実画像の群も独立に SKIP する（`outputs/demo/` は `rm -rf` で消せる席なので、
// 画像だけ無い環境がありうる）。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 逆に資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする（下の
// 「資産の完全性」テスト）— そこは無音の見かけ成功になる。

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
// NOTE: 実画像ケースは models パッケージの前処理層**そのもの**を通すのが主張の中身なので、
// テストからだけ相対で読む（`preprocess.ts` は依存ゼロの純関数 2 本で、runtime の実装は
// 何も引き込まない。publish は `tests/` を除外する — packages/runtime/deno.json）。
import { normalizeToNchw, resizeRgb8 } from "../../models/src/image/preprocess.ts";

/**
 * 実重み SigLIP2 **base**（`patch16-224`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,768]`）:
 *
 * | ケース   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | ------- | ------- | ------------ | ------------------ |
 * | checker  | 1.05e-5 | 7.45e-3 | 6.842        | 2.04e-4            |
 * | noise    | 9.54e-6 | 6.11e-3 | 7.147        | 1.02e-4            |
 * | ramp     | 6.44e-6 | 4.13e-3 | 6.909        | 1.70e-4            |
 * | ramp-dim | 1.05e-5 | 3.03e-3 | 6.950        | 2.36e-4            |
 *
 * **判定は atol が主導する**（`|x−y| ≤ atol + rtol·|ref|`）: pooler_output は L2 正規化されて
 * いない生のベクトルで、値域は 0 を跨いで \|ref\| 最小非ゼロ 1.02e-4 まで薄く広がるため、
 * rtol を主役にすると 0 近傍で発散する（実測 maxRel 7.45e-3 の要素も絶対誤差は
 * `7.45e-3 × 1.4e-3 ≈ 1.0e-5` でしかない）。rtol 1e-6 の寄与は \|ref\| 上端 7.147 でも
 * 7.1e-6 で、atol の 1/7 と判定を主導しない。
 *
 * atol 5e-5 は実測最悪 1.05e-5（checker / ramp-dim）の約 4.8 倍。**EmbeddingGemma の
 * `EMBEDDINGGEMMA_TOLERANCE`（atol 1e-6）より 2 桁緩いのは値域の違いだけ** — あちらの出力は
 * L2 正規化済みの単位ベクトル（\|ref\| 上端 0.25）で、値域で割った相対量はどちらも 1.5e-6 級で
 * 揃っている（1.05e-5 / 7.15 ≈ 1.5e-6 対 2.8e-7 / 0.25 ≈ 1.1e-6）。誤差の出所も同じ（fma 融合・
 * linear / attention の縮約順序が torch と違う・超越関数（tanh / exp / rsqrt）の実装差）で、
 * SDPA は分解経路（mul×2 + bmm + softmax + bmm）を 13 本とも通っている。
 *
 * NOTE: MAP head の q/k/v 明示化パッチ由来の差（`export_siglip2.py --verify` の実測
 * 7.75e-7〜2.38e-6）は**ここには入らない** — golden の期待値は差し替え**後**のモジュールから
 * 採っており、この突合はランタイム側の誤差だけを見ている。
 *
 * 実装バグ（パッチ埋め込みのチャネル取り違え・位置埋め込みの並び違い・attention の head 分割
 * 誤り・MAP head の probe 取り違え）の誤差は出力の値域と同じ O(1)〜O(7) で、この閾値の
 * 5 桁以上上に出る。
 */
const BASE_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * 実重み SigLIP2 **so400m**（`patch14-384`）の torch CPU 期待値との突合に使う許容誤差。
 *
 * MUST: base と**独立に**実測から導く（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本
 * `[1,1152]`）。同じ経路でも hidden 1152 / 27 層 / 729 パッチと縮約の長さが違い、誤差の
 * 積み上がり方も違う:
 *
 * | ケース   | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | -------- | ------- | ------- | ------------ | ------------------ |
 * | checker  | 8.46e-6 | 4.64e-3 | 10.389       | 1.00e-4            |
 * | noise    | 1.05e-5 | 9.76e-3 | 8.451        | 1.28e-4            |
 * | ramp     | 6.68e-6 | 2.55e-2 | 10.753       | 4.68e-6            |
 * | ramp-dim | 6.08e-6 | 8.35e-3 | 10.874       | 1.89e-4            |
 *
 * atol 5e-5 は実測最悪 1.05e-5（noise）の約 4.8 倍。**base と同じ値になったのは実測の帰結**
 * で、共有はしない（片方を測り直したときにもう片方が黙って緩む）。判定を atol が主導する
 * 理由も base と同じ — \|ref\| 最小非ゼロが 4.68e-6（ramp）まで薄く広がり、maxRel 2.55e-2 の
 * 要素も絶対誤差は 1.2e-7 でしかない。rtol 1e-6 の寄与は \|ref\| 上端 10.874 でも 1.1e-5 で、
 * atol の 1/5 と判定を主導しない。
 *
 * NOTE: MAP head の q/k/v 明示化パッチ由来の差（`--verify` の so400m 実測
 * 1.07e-6〜2.86e-6・形の畳み込みは 4 ケースともビット同一）は base と同じくここには入らない。
 */
const SO400M_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * **実画像**ケース（`photo-*`）の突合に使う許容誤差 — SigLIP2 **base**。
 *
 * 合成画像より 3 桁緩いのは、この門が**前処理を含めた鎖**の突合だから。TS 側は
 * `resizeRgb8`（f64 で積算）、Python 側は torchvision（f32 で積算）で、丸め境界に載った標本
 * だけが **1 LSB** ずれる（実測: 差は必ず 1 LSB 以内 = 正規化後 7.8432e-3・ずれる標本は
 * 150,528 中 0.09〜0.15% — 導出は `packages/models/tests/image_preprocess_test.ts` 冒頭）。
 * その入力差が 12 層を通って増幅した量が、そのままこの tolerance の中身になる。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,768]`）:
 *
 * | ケース          | maxAbs  | maxRel | \|ref\| 上端 | 参照入力で回した場合の maxAbs |
 * | --------------- | ------- | ------ | ------------ | ---------------------------- |
 * | photo-portrait  | 1.35e-3 | 0.74   | 5.314        | 3.34e-6                      |
 * | photo-landscape | 2.52e-2 | 16.5   | 4.550        | 1.34e-5                      |
 * | photo-corridor  | 6.89e-3 | 9.94   | 6.727        | 3.82e-6                      |
 * | photo-street    | 7.46e-4 | 1.79   | 5.319        | 5.72e-6                      |
 *
 * 右端の列が**帰属の証拠**: golden に焼かれた Python 側の `pixel_values` をそのまま入力に
 * すると誤差は合成画像と同じ 1e-5 級に戻る（= 大きい側の 2.52e-2 は**ランタイムではなく
 * 1 LSB の入力差の増幅**）。増幅倍率はケースごとに 3 桁ばらつくので、tolerance を実測から
 * 導く以外に決めようが無い。
 *
 * atol 5e-2 は実測最悪 2.52e-2（photo-landscape）の約 2.0 倍。合成画像側より margin が薄いのは
 * 誤差の出所が**決定的**だから — 入力差は f64 の CPU 前処理で環境に依らず同じで、GPU 側の
 * 揺れは右端の列の 1e-5 級しか乗らない。
 *
 * rtol は **0**。atol が 5e-2 もある以上、\|ref\| 上端 6.7 に掛けて意味を持つ rtol は 1e-2 級で、
 * それは「判定しない」に等しい。maxRel が 16.5 まで出るのも 0 近傍の要素の見かけで、その
 * 絶対誤差は 3e-3 でしかない（合成画像側の docstring と同じ理屈）。
 *
 * 実装バグ（解像度の取り違え・チャネル順の反転・補間の取り違え〈bicubic は最大 47/255〉・
 * パッチ埋め込みの誤り）の誤差は出力の値域と同じ O(1)〜O(7) で、この閾値の 2 桁以上上に出る。
 * **画像の差し替え**（生成台本を回し直して golden を採り直していない）は tolerance では
 * 吸収されず、その手前の sha256 突合が名指しで落とす（実測: 1 バイト書き換えで落ちる）。
 * 一方 **resize の 2 軸の取り違えはこの門では捕まらない**（実画像は 4 枚とも正方形で、出力も
 * 正方形）— そこは `image_preprocess_test.ts` の `mixed-axis` ケースが受け持つ。
 */
const REAL_BASE_TOLERANCE: Tolerance = { atol: 5e-2, rtol: 0 };

/**
 * **実画像**ケースの許容誤差 — SigLIP2 **so400m**。MUST: base と独立に実測から導く。
 *
 * | ケース          | maxAbs  | maxRel | \|ref\| 上端 | 参照入力で回した場合の maxAbs |
 * | --------------- | ------- | ------ | ------------ | ---------------------------- |
 * | photo-portrait  | 2.64e-3 | 1.90   | 5.845        | 9.06e-6                      |
 * | photo-landscape | 1.13e-2 | 8.01   | 5.883        | 7.63e-6                      |
 * | photo-corridor  | 6.17e-3 | 9.05   | 6.584        | 9.54e-6                      |
 * | photo-street    | 8.11e-3 | 2.34   | 5.329        | 3.34e-5                      |
 *
 * atol 2.5e-2 は実測最悪 1.13e-2（photo-landscape）の約 2.2 倍。**base より厳しい値になったのは
 * 実測の帰結**で、共有はしない。1024×1024 から 384 への縮小は 224 への縮小より丸め境界に
 * 載る標本が多い（相違率 0.79〜0.90% 対 0.09〜0.15%）のに、出力の増幅は小さい — 層が深い
 * （27 層）ほど誤差が積み上がるとは限らないという実測で、片方の値をもう片方へ流用できない
 * 理由そのもの。
 */
const REAL_SO400M_TOLERANCE: Tolerance = { atol: 2.5e-2, rtol: 0 };

/**
 * 実画像ケースの**入力側**（TS 前処理 対 Python 前処理）の許容誤差。
 *
 * uint8 の 1 LSB は正規化後 `1 / 127.5 = 7.8431e-3` で、f32 へ写す丸めが乗って実測
 * 7.8432e-3（8 ケースとも同値 = ずれは常にちょうど 1 LSB）。
 * `packages/models/tests/image_preprocess_test.ts` の `PIXEL_ATOL` と同じ導出だが、あちらは
 * 22×33 級の合成画像・こちらは 1024×1024 の実画像で、**縮尺も画の性質も違う**ので値は
 * 共有せずここでも実測している。
 *
 * この突合を出力側と分けて持つのは、落ちたときに前処理と推論のどちらが動いたのかを
 * 分けるため（出力側だけだと、1 LSB の増幅と実装バグが同じ「大きい maxAbs」に見える）。
 */
const REAL_PIXEL_TOLERANCE: Tolerance = { atol: 7.85e-3, rtol: 0 };

/** 系列（= モデル）1 本の宣言。系列名は exporter の `--model-dir` のディレクトリ名。 */
type Series = {
  readonly name: string;
  readonly tolerance: Tolerance;
  /** 実画像ケースの許容誤差（合成画像とは 3 桁違う — {@link REAL_BASE_TOLERANCE}）。 */
  readonly realTolerance: Tolerance;
};

/**
 * 実走する系列。**列挙ではなくここで固定する** — 生成済みのものを拾う形にすると、
 * 系列ごと生成し忘れた環境で「緑だが未検証」になる。
 */
const SERIES: readonly Series[] = [
  {
    name: "siglip2-base-patch16-224",
    tolerance: BASE_TOLERANCE,
    realTolerance: REAL_BASE_TOLERANCE,
  },
  {
    name: "siglip2-so400m-patch14-384",
    tolerance: SO400M_TOLERANCE,
    realTolerance: REAL_SO400M_TOLERANCE,
  },
];

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const seriesRoot = (series: Series): URL => new URL(`${series.name}/`, SERIES_PARENT);

/**
 * SKIP 時にそのまま貼れる生成コマンド（tools/exporter/export_siglip2.py の docstring）。
 * 系列名は実重みのディレクトリ名でもある（どちらも HF のリポジトリ名）。
 */
const generateCommand = (series: Series): string =>
  "cd tools/exporter && uv run --group siglip2 python export_siglip2.py" +
  ` --model-dir ../../inputs/siglip2/${series.name}`;

/** 実画像ケースまで含めて採り直すコマンド（`--real-images` はグループが違う）。 */
const realGenerateCommand = (series: Series): string =>
  "cd tools/exporter && uv run --group siglip2-preprocess python export_siglip2.py" +
  ` --real-images --model-dir ../../inputs/siglip2/${series.name}`;

/** 実画像そのものを焼き直すコマンド（プロンプト / seed の正本は台本側）。 */
const IMAGE_COMMAND = "deno task demo:eval-images --source <Anima 配布形のパス>";

/**
 * 生成されているはずの**合成画像**ケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_siglip2.py` の `build_cases`。
 */
const EXPECTED_CASES = ["checker", "noise", "ramp", "ramp-dim"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。ケース名とファイル名の正本は
 * `export_siglip2.py` の `REAL_CASES`、画像そのもの（プロンプト / seed / 解像度）の正本は
 * `examples/anima/eval-images.ts`。
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/**
 * 実画像の判別で見る 2 群（人物が写っている 2 枚 / 写っていない 2 枚）。正本は
 * `export_siglip2.py` の `REAL_PERSON_CASES` / `REAL_SCENE_CASES` で、あちらは torch 出力に、
 * こちらは**実 GPU 出力**に掛ける。
 *
 * NOTE: 合成画像の対（`ramp`×`ramp-dim` 対 `ramp`×`checker`）はここには無い — 判別は実画像
 * 側へ**置き直した**（同じ構造の強弱より「人が写っているか」の方が意味のある判別で、合成
 * 画像側は数値回帰の門として残っている）。torch 側の順序検査は emit のたびに
 * `export_siglip2.py` の `_sanity` が合成・実画像の両方に掛けている。
 */
const REAL_PERSON_CASES = ["photo-portrait", "photo-street"] as const;
const REAL_SCENE_CASES = ["photo-landscape", "photo-corridor"] as const;

/** 実画像 golden の `__metadata__` の欄（正本は `export_siglip2.py` の同名定数）。 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * 前処理の正規化定数。正本は重みと同じ場所の `preprocessor_config.json` で、2 モデルとも
 * mean = std = 0.5（`siglip2_preprocess.py` の `check_processor_shape` が emit のたびに実測し、
 * 外れていたら golden が 1 バイトも書かれない）。解像度は系列ごとに違うので**グラフ入力の
 * 宣言から取る**（ここに 224 / 384 を書くともう片方で黙って別の画を作る）。
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.5, 0.5, 0.5];
const IMAGE_STD: readonly [number, number, number] = [0.5, 0.5, 0.5];

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

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
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

const cosine = (first: Float32Array, second: Float32Array): number => {
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let i = 0; i < first.length; i += 1) {
    dot += first[i] * second[i];
    firstNorm += first[i] * first[i];
    secondNorm += second[i] * second[i];
  }
  return dot / Math.sqrt(firstNorm * secondNorm);
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
 * MUST: NotFound 以外は伝播させる（`listDir` と同じ理由 — 権限エラーを「無い」と読み替えない）。
 */
const IMAGES_PRESENT: boolean = REAL_CASES.every((entry) => {
  try {
    return Deno.statSync(new URL(entry.file, DEMO_DIR)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
});

/** グラフ入力の静的次元（記号次元は無い — `export_siglip2.py` の `symbol_names=()`）。 */
const staticDim = (parsed: KarumeModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/**
 * 実画像 1 枚を **TS 側の経路**で `pixel_values` にする（PNG decode → resize → normalize）。
 * 通すのは models パッケージの前処理層そのもの — この鎖こそがこの門の主張。
 */
const preprocessImage = async (
  png: Uint8Array<ArrayBuffer>,
  file: string,
  width: number,
  height: number,
): Promise<Float32Array<ArrayBuffer>> =>
  normalizeToNchw(resizeRgb8(await decodePng(png, file), width, height), IMAGE_MEAN, IMAGE_STD);

for (const series of SERIES) {
  const root = seriesRoot(series);
  /** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
  const discovered = discoverCases(root);
  const realNames = new Set<string>(REAL_CASES.map((entry) => entry.name));
  const cases = discovered.filter((name) => !realNames.has(name));
  const realCases = discovered.filter((name) => realNames.has(name));
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  const available = discovered.length > 0;
  /** 実画像の群。golden と画像の**両方**が揃ってはじめて実走する。 */
  const realAvailable = available && realCases.length > 0 && IMAGES_PRESENT;

  if (!available) {
    console.warn(
      `[karume] ${root.pathname} に export 済み資産が無いため実重み SigLIP2 vision E2E ` +
        `(${series.name}) を SKIP する（重みがリポジトリ管理外）。` +
        `生成: ${generateCommand(series)}`,
    );
  } else if (!realAvailable) {
    console.warn(
      `[karume] 実画像ケース (${series.name}) を SKIP する（golden ${realCases.length}/` +
        `${REAL_CASES.length} 本・画像 ${IMAGES_PRESENT ? "有" : "無"}）。` +
        `画像の生成: ${IMAGE_COMMAND} / golden の生成: ${realGenerateCommand(series)}`,
    );
  }

  Deno.test({
    name: `SigLIP2 資産: ${series.name} — 期待するケースとモデル本体が揃っている`,
    // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
    ignore: !available,
    fn: () => {
      assertEquals(cases, [...EXPECTED_CASES], `${root.pathname} の合成画像 golden ケース`);
      // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
      // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
      assert(
        realCases.length === 0 || realCases.length === REAL_CASES.length,
        `${root.pathname} の実画像 golden が ${realCases.length}/${REAL_CASES.length} 本` +
          `（採り直す: ${realGenerateCommand(series)}）`,
      );
      const model = new URL(MODEL_FILE, root);
      assert(Deno.statSync(model).isFile, `${MODEL_FILE} が無い`);
    },
  });

  for (const caseName of cases) {
    Deno.test({
      name: `SigLIP2 golden 突合: ${series.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: !available || !GPU_AVAILABLE,
      fn: async () => {
        const [modelBytes, ioBytes] = await Promise.all([
          readBuffer(root, MODEL_FILE),
          readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
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

          parsed.graph.outputs.forEach((name, index) => {
            const view = io.tensors.get(`output.${index}`);
            assert(view !== undefined, `output.${index} が golden に無い`);
            const where = `${series.name} / ${caseName} output.${index} ('${name}')`;
            const declared = parsed.graph.values[name].dtype;
            assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
            assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
            const report = compareTensors(
              outputs[name],
              ioTensor(io, view, declared),
              series.tolerance,
            );
            assert(report.pass, `${where}: ${formatAllclose(report)}`);
          });
        } finally {
          await session.dispose();
          gpu.destroy();
        }
      },
    });
  }

  for (const real of REAL_CASES) {
    Deno.test({
      name: `SigLIP2 実画像 golden 突合: ${series.name} / ${real.name}` +
        "（TS 前処理 + 実 GPU / Python 前処理 + torch CPU）",
      ignore: !realAvailable || !GPU_AVAILABLE,
      fn: async () => {
        const [modelBytes, ioBytes, png] = await Promise.all([
          readBuffer(root, MODEL_FILE),
          readBuffer(root, `${IO_PREFIX}${real.name}${IO_SUFFIX}`),
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
          `${real.file} が golden を採った画像と違う（採り直す: ${realGenerateCommand(series)}）`,
        );

        const width = staticDim(parsed, 3);
        const height = staticDim(parsed, 2);
        const pixels = await preprocessImage(png, real.file, width, height);

        // ② 前処理の突合（TS の f64 経路 対 torchvision の f32 経路）。出力側と分けて持つ理由は
        // REAL_PIXEL_TOLERANCE の docstring。
        const inputView = io.tensors.get(`input.${parsed.graph.inputs[0].name}`);
        assert(inputView !== undefined, "golden に入力が無い");
        const inputReport = compareTensors(
          { dtype: "f32", data: pixels },
          ioTensor(io, inputView, "f32"),
          REAL_PIXEL_TOLERANCE,
        );
        assert(
          inputReport.pass,
          `${series.name} / ${real.name} pixel_values: ${formatAllclose(inputReport)}`,
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
          const where = `${series.name} / ${real.name} output.0 ('${name}')`;
          const declared = parsed.graph.values[name].dtype;
          assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
          assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
          const report = compareTensors(
            outputs[name],
            ioTensor(io, view, declared),
            series.realTolerance,
          );
          assert(report.pass, `${where}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
          gpu.destroy();
        }
      },
    });
  }

  Deno.test({
    name: `SigLIP2 判別: ${series.name} — 人物 2 枚の cosine が人物と風景のどの対よりも高い`,
    ignore: !realAvailable || !GPU_AVAILABLE,
    fn: async () => {
      // golden 突合だけだと「期待値と合っている」ことしか言えず、埋め込みとして意味のある
      // 出力かは別問題（1 点へ潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは
      // **別々の入力どうしの順序**を見るので、出力がケース間で定数なら cosine が全て 1 に
      // なって落ちる。閾値は置かない（順序そのものが検査対象 — export_siglip2.py の
      // `_real_sanity` と同じ形で、あちらは torch 側に掛かっている）。
      //
      // 入力は**実画像を TS 前処理で通したもの**（golden の入力ではない）— 判別まで含めて
      // 「PNG を渡したら意味のある埋め込みが返る」ことの検査にする。
      const modelBytes = await readBuffer(root, MODEL_FILE);
      const parsed = openModel(modelBytes);
      const [outputName] = parsed.graph.outputs;
      const width = staticDim(parsed, 3);
      const height = staticDim(parsed, 2);
      const inputName = parsed.graph.inputs[0].name;

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      const pooled = new Map<string, Float32Array>();
      try {
        // 4 枚を 1 Session で回す（重みは 350MB〜1.7GB — 画像ごとに組み直す理由が無い）。
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
          assert(output.dtype === "f32", `${real.name}: pooler_output の dtype が ${output.dtype}`);
          pooled.set(real.name, output.data);
        }
      } finally {
        await session.dispose();
        gpu.destroy();
      }

      const cosineOf = (first: string, second: string): number => {
        const [left, right] = [pooled.get(first), pooled.get(second)];
        assert(left !== undefined && right !== undefined, `${first}×${second} の出力が無い`);
        return cosine(left, right);
      };
      const person = cosineOf(REAL_PERSON_CASES[0], REAL_PERSON_CASES[1]);
      for (const personCase of REAL_PERSON_CASES) {
        for (const sceneCase of REAL_SCENE_CASES) {
          const cross = cosineOf(personCase, sceneCase);
          assert(
            person > cross,
            `cosine の順序が逆: ${REAL_PERSON_CASES.join("×")}=${person} <=` +
              ` ${personCase}×${sceneCase}=${cross}`,
          );
        }
      }
    },
  });
}
