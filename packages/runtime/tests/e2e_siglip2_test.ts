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
// 入力は**どちらの群も golden に焼かれた `pixel_values` そのもの**（ビット同一）なので、
// 突合に出るのは**ランタイムの数値誤差だけ**。2 群に分けて持つのは、踏む分布が違うから:
//
// - **合成画像 4 ケース**（`ramp` / `ramp-dim` / `checker` / `noise`）: 値域の端や勾配を
//   踏むぶん数値回帰の検出が鋭い（実測 6.4e-6〜1.05e-5）。
// - **実画像 4 ケース**（`photo-*`）: 自然画像の分布点でのランタイム忠実度と、出力が埋め込み
//   として意味を持つかの**判別**（cosine の順序 — {@link REAL_PERSON_CASES}）を受け持つ。
//   誤差の桁は合成側と同じだが、tolerance は群ごと・系列ごとに独立に実測から導く
//   （{@link REAL_BASE_TOLERANCE}）。
//
// ## TS 前処理を含む鎖は、2 つの門の**合成**で持つ
//
// 「PNG を渡したら Python と同じ埋め込みが返る」という鎖は、このテスト単独ではなく
// `packages/models/tests/image_preprocess_real_test.ts` の**入力側 parity 門**（同じ PNG から
// Python と同じ `pixel_values` が出る — 入力差 ≤ 7.85e-3 = uint8 の 1 LSB 級）と、本テストの
// **golden 入力での忠実度**の合成で持つ。前処理をここで通さないのは依存方向のため
// （runtime のテストから models の実装を相対 import するのは逆向き）。分けた副産物として、
// 落ちたときに前処理と推論のどちらが動いたのかがテストの名前で分かる。
//
// 資産が無い環境では**明示 SKIP**する（系列ごとに独立 — 片方だけ生成した環境でも、ある方は
// 実走する）。実画像の群も独立に SKIP する（`--real-images` を付けずに emit した資産では
// 合成 4 ケースしか無い）。ADR 0005 の「全 SKIP は明示 FAIL」門番
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
 * 入力は合成画像と同じく golden に焼かれた `pixel_values` そのものなので、出るのは
 * ランタイム誤差だけで桁も合成側と同じ。それでも {@link BASE_TOLERANCE} と**共有しない**のは、
 * 踏む分布が違って誤差の出方も違うから（片方を測り直したときにもう片方が黙って緩む形にしない
 * のは、系列間で共有しないのと同じ理由）。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,768]`）:
 *
 * | ケース          | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | --------------- | ------- | ------- | ------------ | ------------------ |
 * | photo-portrait  | 3.34e-6 | 2.16e-3 | 5.314        | 8.77e-5            |
 * | photo-landscape | 1.34e-5 | 6.30e-3 | 4.550        | 1.82e-4            |
 * | photo-corridor  | 3.82e-6 | 1.73e-2 | 6.727        | 7.22e-5            |
 * | photo-street    | 5.72e-6 | 2.49e-3 | 5.319        | 1.55e-4            |
 *
 * atol 5e-5 は実測最悪 1.34e-5（photo-landscape）の約 3.7 倍。合成画像側と同じ値になったのは
 * 実測の帰結で、共有の根拠ではない。
 *
 * rtol は **0**（合成画像側の 1e-6 も入れない）。実測最悪 1.34e-5 は atol 単独で収まり、
 * rtol 1e-6 を足しても \|ref\| 上端 6.727 で寄与は 6.7e-6 = atol の 1/7 と判定を動かさない —
 * 判定を動かさないものは置かない。maxRel が 1.73e-2 まで出るのは 0 近傍の要素の見かけで、
 * その絶対誤差は 3.8e-6 でしかない（合成画像側の docstring と同じ理屈）。
 *
 * ランタイム側の実装バグ（{@link BASE_TOLERANCE} が列挙するもの）の誤差は出力の値域と同じ
 * O(1)〜O(7) で、この閾値の 5 桁以上上に出る。一方 **TS 前処理側の誤り**（解像度の取り違え・
 * チャネル順の反転・補間の取り違え）**はこの門では捕まらない** — 入力が golden の
 * `pixel_values` だから。そこは
 * `packages/models/tests/image_preprocess_real_test.ts` が受け持つ（ファイル冒頭の「2 つの門の
 * 合成」）。
 */
const REAL_BASE_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 0 };

/**
 * **実画像**ケースの許容誤差 — SigLIP2 **so400m**。MUST: base と独立に実測から導く。
 *
 * | ケース          | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | --------------- | ------- | ------- | ------------ | ------------------ |
 * | photo-portrait  | 9.06e-6 | 2.85e-3 | 5.845        | 3.24e-4            |
 * | photo-landscape | 7.63e-6 | 1.20e-2 | 5.883        | 3.19e-4            |
 * | photo-corridor  | 9.54e-6 | 1.42e-3 | 6.584        | 1.47e-4            |
 * | photo-street    | 3.34e-5 | 3.95e-3 | 5.329        | 4.22e-4            |
 *
 * atol 1e-4 は実測最悪 3.34e-5（photo-street）の約 3.0 倍。**合成画像側の
 * {@link SO400M_TOLERANCE}（5e-5）より緩いのは実測の帰結** — photo-street だけが合成 4 ケースの
 * 最悪 1.05e-5 の 3 倍出ており、自然画像の分布点が縮約の長い経路（hidden 1152 / 27 層 /
 * 729 パッチ）で合成画像より悪い側を踏むことがある、という実測そのもの。base（実測最悪
 * 1.34e-5）へ流用できない理由でもある。
 *
 * rtol が 0 な理由は base と同じ（rtol 1e-6 の寄与は \|ref\| 上端 6.584 でも 6.6e-6 で、
 * atol の 1/15）。
 */
const REAL_SO400M_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 0 };

/** 系列（= モデル）1 本の宣言。系列名は exporter の `--model-dir` のディレクトリ名。 */
type Series = {
  readonly name: string;
  readonly tolerance: Tolerance;
  /** 実画像ケースの許容誤差（合成画像とは独立に実測から導く — {@link REAL_BASE_TOLERANCE}）。 */
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

/**
 * 生成されているはずの**合成画像**ケース。**列挙結果ではなくここで固定する** — 列挙だけに
 * 頼ると生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。正本は
 * `tools/exporter/export_siglip2.py` の `build_cases`。
 */
const EXPECTED_CASES = ["checker", "noise", "ramp", "ramp-dim"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。正本は `export_siglip2.py` の
 * `REAL_CASES`。ここが要るのは golden のケース名だけで、元になった PNG との対応は
 * `packages/models/tests/image_preprocess_real_test.ts` が持つ（このテストは PNG を読まない —
 * ファイル冒頭の「2 つの門の合成」）。
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

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

/** ファイルの有無。MUST: NotFound 以外は伝播させる（`listDir` と同じ理由）。 */
const fileExists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
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

for (const series of SERIES) {
  const root = seriesRoot(series);
  /** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
  const discovered = discoverCases(root);
  const realNames = new Set<string>(REAL_CASES);
  const cases = discovered.filter((name) => !realNames.has(name));
  const realCases = discovered.filter((name) => realNames.has(name));
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  const available = discovered.length > 0;
  /** 実画像の群（`--real-images` を付けずに emit した資産には無い）。 */
  const realAvailable = available && realCases.length > 0;
  /**
   * **何か 1 つでも**残っているか（完全性テストの SKIP 述語 — Codex 波 H 指摘 H-02）。
   * golden が全滅してモデルだけ残った欠損は `available` では偽になり、`ignore: !available`
   * だと完全性テスト自身が SKIP される — 欠損を FAIL にする述語は「完全に空」でだけ寝てよい。
   */
  const anyPresent = available || fileExists(new URL(MODEL_FILE, root));

  if (!available) {
    console.warn(
      `[karume] ${root.pathname} に export 済み資産が無いため実重み SigLIP2 vision E2E ` +
        `(${series.name}) を SKIP する（重みがリポジトリ管理外）。` +
        `生成: ${generateCommand(series)}`,
    );
  } else if (!realAvailable) {
    console.warn(
      `[karume] 実画像ケース (${series.name}) を SKIP する（golden ${realCases.length}/` +
        `${REAL_CASES.length} 本）。生成: ${realGenerateCommand(series)}`,
    );
  }

  Deno.test({
    name: `SigLIP2 資産: ${series.name} — 期待するケースとモデル本体が揃っている`,
    // 完全に空の環境だけ「生成していない」として SKIP。**何か 1 つでも**あれば欠けは FAIL
    //（モデルだけ残って golden が全滅した欠損も拾う — `anyPresent` の JSDoc）。
    ignore: !anyPresent,
    fn: () => {
      assertEquals(cases, [...EXPECTED_CASES], `${root.pathname} の合成画像 golden ケース`);
      // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
      // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
      assert(
        realCases.length === 0 || realCases.length === REAL_CASES.length,
        `${root.pathname} の実画像 golden が ${realCases.length}/${REAL_CASES.length} 本` +
          `（採り直す: ${realGenerateCommand(series)}）`,
      );
      assert(fileExists(new URL(MODEL_FILE, root)), `${MODEL_FILE} が無い`);
    },
  });

  /**
   * 突合を回すケース。合成と実画像で**入力の作り方は同じ**（どちらも golden の
   * `pixel_values`）で、違うのは踏む分布と、そこから独立に導いた tolerance だけ。
   */
  const goldenCases = [
    ...cases.map((name) => ({
      name,
      label: "golden 突合",
      tolerance: series.tolerance,
      ignore: !available,
    })),
    ...realCases.map((name) => ({
      name,
      label: "実画像 golden 突合",
      tolerance: series.realTolerance,
      ignore: !realAvailable,
    })),
  ];

  for (const entry of goldenCases) {
    const caseName = entry.name;
    Deno.test({
      name: `SigLIP2 ${entry.label}: ${series.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
      ignore: entry.ignore || !GPU_AVAILABLE,
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
              entry.tolerance,
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
      // 入力は golden に焼かれた `pixel_values`（= 突合ケースと同じもの）。TS 前処理を通した
      // 入力での判別ではなくなったが、主張は落ちない — 前処理側の parity は
      // `packages/models/tests/image_preprocess_real_test.ts` が入力差 1 LSB 級で縛っており、
      // その幅では順序が反転しないことを実測してある（2026-08-14: 人物対と最悪の交差対の差は
      // base 0.1196 / so400m 0.0747 なのに対し、入力を TS 前処理へ差し替えたときの各 cosine の
      // 動きは最大 1.6e-3 —— 2 桁小さい）。
      const modelBytes = await readBuffer(root, MODEL_FILE);
      const parsed = openModel(modelBytes);
      const [outputName] = parsed.graph.outputs;

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      const pooled = new Map<string, Float32Array>();
      try {
        // 4 ケースを 1 Session で回す（重みは 350MB〜1.7GB — ケースごとに組み直す理由が無い）。
        for (const real of REAL_CASES) {
          const io = parseSafetensors(await readBuffer(root, `${IO_PREFIX}${real}${IO_SUFFIX}`));
          const output = (await session.run(goldenInputs(parsed, io)))[outputName];
          // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
          assert(output.dtype === "f32", `${real}: pooler_output の dtype が ${output.dtype}`);
          pooled.set(real, output.data);
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
