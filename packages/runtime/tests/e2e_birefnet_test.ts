// 実重みの BiRefNet 系（背景抜き / salient object segmentation）の実 GPU golden E2E
// （ADR 0005 の段 3）。
//
// tiny golden（tests/e2e_golden_test.ts）が「op 契約の被覆」を、SigLIP2
// （tests/e2e_siglip2_test.ts）が「単一ベクトル出力の画像系」を受け持つのに対し、こちらは
// **画素ごとの出力を持つ画像系**（`[1,1,S,S]` の 1,048,576 要素マップ）を受け持つ。対象は
// `outputs/series/<系列>/`（重み + 焼いた定数で 1 系列 964MB のためリポジトリ管理外 —
// `.gitignore` の `outputs/`）。生成は `tools/export-recipes/birefnet/export.py`（コマンドは
// 各系列の {@link Series.generate} がそのまま正本）。
//
// 系列は**モデル × 解像度ごとに 1 本**（下の {@link SERIES}）。BiRefNet_HR（上流の
// 高解像度チェックポイント）と Lucida（その fine-tune — 構造は完全に同一で重みだけが違う）を
// 別系列として実走する。shifted-window マスクも H/W padding のゼロ定数も解像度依存の定数と
// して焼かれるので、解像度が変われば別のグラフになる。ここで実走するのはどちらも **1024²** で、
// 2048²（本家 handler の General-HR）は実行段が未実測（conv2d の dispatch 上限と中間 3.22GB
// — birefnet/export.py の docstring）。
//
// **許容誤差は系列ごとに独立して実測する**（`SERIES` の各行が自分の tolerance を持つ）。
// 共有すると、片方を測り直したときにもう片方が黙って緩む — 2 系列は同じ構造でも logit の
// 値域が桁で違う（実測: BiRefNet_HR の \|ref\| 上端 64.1 に対し Lucida は 1078.1）。
//
// グラフは 1 本で、出力も **sigmoid 前の logit `[1,1,S,S]`** 1 本だけ（マットの α は
// `sigmoid` を掛けたホスト側の値）。入力は**正規化済みの** `pixel_values f32 [1,3,S,S]`
// 1 本で、記号次元は無い。
//
// ## golden の 2 群（合成画像 + 実画像）— どちらも残す
//
// 入力は**どちらの群も golden に焼かれた `pixel_values` そのもの**（ビット同一）なので、
// 突合に出るのは**ランタイムの数値誤差だけ**。2 群に分けて持つのは、踏む分布が違うから:
//
// - **合成画像 4 ケース**（`checker` / `disc` / `noise` / `ramp`）: 値域の端や勾配を踏むぶん
//   数値回帰の検出が鋭い。`disc` は暗い背景に明るい円を置いた顕著物体で、幾何の判別
//   （円内 logit 平均 > 円外）を実 GPU 出力に掛ける土台にもなっている。
// - **実画像 4 ケース**（`photo-*`）: 自然画像の分布点でのランタイム忠実度。tolerance は群
//   ごと・系列ごとに独立に実測から導く（{@link HR_REAL_TOLERANCE}）。
//
// ## TS 前処理を含む鎖は、2 つの門の**合成**で持つ
//
// 「PNG を渡したら Python と同じマットが返る」という鎖は、このテスト単独ではなく
// `packages/models/tests/e2e_birefnet_real_test.ts` の**入力側 parity 門**（同じ PNG から
// Python と同じ `pixel_values` が出る — 入力差 ≤ 1e-6）と、本テストの**golden 入力での
// 忠実度**の合成で持つ。前処理をここで通さないのは依存方向のため（runtime のテストから
// models の実装を相対 import するのは逆向き）。分けた副産物として、落ちたときに前処理と推論の
// どちらが動いたのかがテストの名前で分かる。
//
// 意味の判別のうち**実画像側**（顕著物体のある 2 枚の前景比が無い 2 枚を上回る）と、その
// ついでに書くマット PNG も同じ理由で models 側にある（`encodePng` が models の実装で、
// 入力も TS 前処理を通したものだから）。**合成画像側の幾何判別**（`disc`）は golden の入力
// だけで完結するのでここに残る。
//
// 資産が無い環境では**明示 SKIP** する（系列ごとに独立）。実画像の群も独立に SKIP する
// （`--real-images` を付けずに emit した資産では合成 4 ケースしか無い）。ADR 0005 の
// 「全 SKIP は明示 FAIL」門番（tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見て
// おり、この SKIP とは独立。逆に資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく
// FAIL にする（下の「資産の完全性」テスト）— そこは無音の見かけ成功になる。

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
 * **BiRefNet_HR / 合成画像**ケース（入力が golden とビット同一）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 2.38e-5 | 3.04e-6 | 11.677       | 3.50               |
 * | disc    | 1.26e-4 | 7.69e-3 | 23.123       | 5.36e-4            |
 * | noise   | 8.30e-5 | 1.00e-5 | 15.302       | 3.16               |
 * | ramp    | 2.11e-4 | 6.73e-1 | 10.765       | 2.58e-4            |
 *
 * atol 1e-3 は実測最悪 2.11e-4（ramp）の約 4.7 倍。
 *
 * **rtol は 0**。この出力は 0 を跨ぐ logit の地図で、`disc` / `ramp` のように前景と背景の
 * 境界を持つケースでは \|ref\| が 2.6e-4 まで薄く落ちる（境界画素は定義上 logit ≈ 0）。
 * rtol を主役にすると境界のところで判定が発散する — 実測 maxRel 0.673（ramp）の要素も、
 * 絶対誤差は 1.7e-4 でしかない。
 *
 * 誤差の出所は SigLIP2 と同じ（fma 融合・linear / conv の縮約順序が torch と違う・超越関数の
 * 実装差）だが、値域が広い（\|ref\| 上端 23.1）ぶん絶対量は 1 桁大きい。相対量で見ると
 * 2.11e-4 / 10.8 ≈ 2.0e-5 で、実画像側（1.09e-3 / 64.1 ≈ 1.7e-5 — {@link HR_REAL_TOLERANCE}）と
 * 同じ桁に揃う。
 *
 * 実装バグ（deform_conv2d のオフセット取り違え・窓マスクの位相ずれ・upsample の軸違い・
 * BatchNorm の per-channel 定数の並び違い）の誤差は出力の値域と同じ O(1)〜O(20) で、この
 * 閾値の 4 桁以上上に出る。
 */
const HR_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 0 };

/**
 * **BiRefNet_HR / 実画像**ケース（`photo-*`）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース          | TS 前処理入力の maxAbs | \|ref\| 上端 | **golden 入力の maxAbs** |
 * | --------------- | ---------------------- | ------------ | ------------------------ |
 * | photo-corridor  | 4.77e-5                | 12.729       | 4.01e-5                  |
 * | photo-landscape | 5.25e-5                | 12.662       | 3.24e-5                  |
 * | photo-portrait  | 1.78e-3                | 64.094       | 1.09e-3                  |
 * | photo-street    | 8.58e-5                | 16.069       | 9.63e-5                  |
 *
 * この門が入力に使うのは**右端の列と同じ golden の `pixel_values`**（TS 前処理を通す鎖は
 * models 側へ分かれた — モジュール docstring の「2 つの門の合成」）。実効の実測最悪は
 * 1.09e-3（photo-portrait）で、**atol 5e-3 は据え置き**（約 4.6 倍の margin。左の列で導いた
 * ときの 2.8 倍より広くなっただけで、締め直すには実資産のある環境で測り直す必要がある）。
 *
 * 左右の列がほぼ同じ桁なのが**帰属の証拠**: 実画像側で誤差が大きいのは前処理の差ではなく
 * **値域**で、photo-portrait は logit が 64.1 まで振れる（顕著物体が画面の 55% を占め、内部の
 * 飽和が深い）。相対量で見ると 1.09e-3 / 64.1 ≈ 1.7e-5 で、合成画像側（≈2.0e-5）と同じ桁。
 * **それでも定数を共有しない** — 片方を測り直したときにもう片方が黙って緩む。
 *
 * **rtol は 0**（理由は {@link HR_SYNTHETIC_TOLERANCE} と同じ — 0 を跨ぐ logit の地図）。
 *
 * **画像の差し替え**（生成台本を回し直して golden を採り直していない）はこの門では見えない —
 * その sha256 突合は `packages/models/tests/e2e_birefnet_real_test.ts` が名指しで持つ。
 */
const HR_REAL_TOLERANCE: Tolerance = { atol: 5e-3, rtol: 0 };

/**
 * **Lucida / 合成画像**ケースの突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース  | maxAbs  | maxRel  | \|ref\| 上端 | \|ref\| 最小非ゼロ |
 * | ------- | ------- | ------- | ------------ | ------------------ |
 * | checker | 1.53e-5 | 2.68e-6 | 13.554       | 3.34               |
 * | disc    | 1.11e-4 | 2.83e-3 | 17.400       | 1.19e-3            |
 * | noise   | 5.29e-5 | 1.85e-2 | 13.574       | 1.76e-4            |
 * | ramp    | 4.41e-5 | 2.89e-3 | 13.210       | 3.04e-4            |
 *
 * atol 5e-4 は実測最悪 1.11e-4（disc）の約 4.5 倍。**rtol は 0**（理由は
 * {@link HR_SYNTHETIC_TOLERANCE} と同じ — 0 を跨ぐ logit の地図）。
 *
 * BiRefNet_HR より 1 段小さいのは、合成画像に対する Lucida の応答が浅い（顕著物体を見つけ
 * られず値域が広がらない）ため。**両系列で同じ定数を使わない**のはこの非対称のためでもある。
 */
const LUCIDA_SYNTHETIC_TOLERANCE: Tolerance = { atol: 5e-4, rtol: 0 };

/**
 * **Lucida / 実画像**ケース（`photo-*`）の突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合、4 ケース × 出力 1 本 `[1,1,1024,1024]`）:
 *
 * | ケース          | TS 前処理入力の maxAbs | \|ref\| 上端 | **golden 入力の maxAbs** |
 * | --------------- | ---------------------- | ------------ | ------------------------ |
 * | photo-corridor  | 1.25e-4                | 12.950       | 5.34e-5                  |
 * | photo-landscape | 1.00e-4                | 13.711       | 8.77e-5                  |
 * | photo-portrait  | 9.80e-3                | **1078.080** | 4.30e-3                  |
 * | photo-street    | 8.77e-4                | 210.791      | 1.05e-3                  |
 *
 * この門が入力に使うのは右端の列と同じ golden の `pixel_values`。実効の実測最悪は 4.30e-3
 * （photo-portrait）で、**atol 3e-2 は据え置き**（約 7.0 倍の margin）。
 *
 * BiRefNet_HR の実画像門（5e-3）より 1 桁緩いのは**値域の差がそのまま出ている**だけで、
 * 精度が落ちているわけではない — 相対量は 4.30e-3 / 1078.1 ≈ 4.0e-6 で、BiRefNet_HR の
 * 1.7e-5 より**小さい**。fine-tune された Lucida は前景の確信が桁で強く出る（logit 1078 =
 * sigmoid で 1 との差が f64 でも表せない飽和）。
 */
const LUCIDA_REAL_TOLERANCE: Tolerance = { atol: 3e-2, rtol: 0 };

/**
 * 二値マスク（`logit > 0` = `sigmoid > 0.5`）が torch と食い違ってよい画素の割合。
 *
 * この門が出力側の tolerance と別に要るのは、**成果物がマスクだから**。tolerance は「値が
 * 近い」しか言わず、境界の画素（logit ≈ 0）は近いままいくらでも符号が反転しうる。逆にここ
 * だけでは値の回帰を捉えられない（飽和域の誤差は符号を変えない）ので、両方を持つ。
 *
 * 実測は **2 系列 × 8 ケースとも 0 / 1,048,576**（合成 4 + 実画像 4）。0 をそのまま門に
 * しないのは、境界画素の符号が別のバックエンドやドライバで動きうるため — 1e-4 は 1024² で
 * 104 画素に相当し、マットの見た目には出ない量。実装バグ側は数万〜数十万画素が反転するので、
 * この閾値の 3 桁以上上に出る。**系列で共有する**のは、これが数値の量ではなく「マスクとしての
 * 判断が一致する」という同じ 1 つの主張だから（両系列とも実測 0 で、緩める理由が片方にも
 * 無い）。
 */
const MASK_DISAGREEMENT_LIMIT = 1e-4;

/** 実走する 1 系列（モデル × 解像度）。 */
type Series = {
  /** `outputs/series/` 直下のディレクトリ名（`birefnet.export.default_out_dir` の綴り）。 */
  readonly name: string;
  /** SKIP 時にそのまま貼れる生成コマンド。 */
  readonly generate: string;
  /** 合成画像ケースの許容誤差（**系列ごとに独立実測** — モジュール docstring）。 */
  readonly tolerance: Tolerance;
  /** 実画像ケースの許容誤差（同上）。 */
  readonly realTolerance: Tolerance;
};

const EXPORT_PREFIX =
  "cd tools/export-recipes && uv run --group birefnet python -m birefnet.export";

/**
 * 実走する系列（どちらも 1024²）。**列挙結果ではなくここで固定する** — 列挙だけに頼ると
 * 生成を一部だけ流した環境でテストが黙って消え、「緑だが未検証」になる。
 */
const SERIES: readonly Series[] = [
  {
    name: "birefnet-hr-1024",
    generate: `${EXPORT_PREFIX} --real-images`,
    tolerance: HR_SYNTHETIC_TOLERANCE,
    realTolerance: HR_REAL_TOLERANCE,
  },
  {
    name: "lucida-1024",
    generate: `${EXPORT_PREFIX} --model-dir <リポ>/inputs/birefnet/lucida --real-images`,
    tolerance: LUCIDA_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_REAL_TOLERANCE,
  },
];

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const seriesRoot = (series: Series): URL => new URL(`${series.name}/`, SERIES_PARENT);

/**
 * 生成されているはずの**合成画像**ケース。正本は `birefnet/export.py` の
 * `build_cases`（モデル軸に依らず同じ 4 枚）。
 */
const EXPECTED_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * **実画像**ケース（`--real-images` を付けた emit だけが持つ）。正本は `birefnet/export.py` の
 * `REAL_CASES`。ここが要るのは golden のケース名だけで、元になった PNG との対応は
 * `packages/models/tests/e2e_birefnet_real_test.ts` が持つ（このテストは PNG を読まない —
 * モジュール docstring の「2 つの門の合成」）。
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

/**
 * 幾何の判別に使う合成ケースと、その円の半径（画像の短辺を 1 とした比）。正本は
 * `birefnet/export.py` の `DISC_CASE` / `DISC_RADIUS`（あちらは torch 出力に同じ式を掛ける）。
 */
const DISC_CASE = "disc";
const DISC_RADIUS = 0.3;

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

/** グラフ入力の静的次元（記号次元は無い — `birefnet/export.py` の `symbol_names=()`）。 */
const staticDim = (parsed: KarumeModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(typeof dim === "number", `pixel_values の軸 ${axis} が記号次元 '${String(dim)}'`);
  return dim;
};

/** 二値マスク（前景 = `logit > 0`）が食い違う画素の割合（{@link MASK_DISAGREEMENT_LIMIT}）。 */
const maskDisagreement = (got: Float32Array, expected: Float32Array): number => {
  assertEquals(got.length, expected.length, "マット長");
  let differing = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (got[index] > 0 !== expected[index] > 0) differing += 1;
  }
  return differing / expected.length;
};

/**
 * `disc` ケースの円内（`[S, S]` の bool）。`birefnet/export.py` の `disc_mask` と**同じ式**
 * （あちらが画像を作り、こちらは実 GPU 出力を同じ円で切る）。
 */
const discMask = (size: number): Uint8Array => {
  const axis = new Float64Array(size);
  for (let index = 0; index < size; index += 1) axis[index] = ((index + 0.5) / size) * 2 - 1;
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      mask[y * size + x] = axis[y] ** 2 + axis[x] ** 2 <= DISC_RADIUS ** 2 ? 1 : 0;
    }
  }
  return mask;
};

/** 円の内 / 外それぞれの logit 平均（{@link discMask} で切る）。 */
const discMeans = (logits: Float32Array, size: number): { inside: number; outside: number } => {
  const mask = discMask(size);
  let inside = 0;
  let insideCount = 0;
  let outside = 0;
  let outsideCount = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (mask[index] === 1) {
      inside += logits[index];
      insideCount += 1;
    } else {
      outside += logits[index];
      outsideCount += 1;
    }
  }
  return { inside: inside / insideCount, outside: outside / outsideCount };
};

/** 1 系列ぶんの資産の状態（登録時点で必要なので同期列挙する）。 */
type Discovery = {
  readonly cases: readonly string[];
  readonly realCases: readonly string[];
  /** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
  readonly available: boolean;
  /** 実画像の群（`--real-images` を付けずに emit した資産には無い）。 */
  readonly realAvailable: boolean;
};

const realNames = new Set<string>(REAL_CASES);

const discover = (series: Series): Discovery => {
  const discovered = discoverCases(seriesRoot(series));
  const realCases = discovered.filter((name) => realNames.has(name));
  const available = discovered.length > 0;
  return {
    cases: discovered.filter((name) => !realNames.has(name)),
    realCases,
    available,
    realAvailable: available && realCases.length > 0,
  };
};

const DISCOVERED: ReadonlyMap<string, Discovery> = new Map(
  SERIES.map((series) => [series.name, discover(series)]),
);

const discoveryOf = (series: Series): Discovery => {
  const found = DISCOVERED.get(series.name);
  if (found === undefined) throw new Error(`系列 ${series.name} の列挙が無い`);
  return found;
};

for (const series of SERIES) {
  const found = discoveryOf(series);
  if (!found.available) {
    console.warn(
      `[karume] ${seriesRoot(series).pathname} に export 済み資産が無いため実重み BiRefNet ` +
        `E2E（${series.name}）を SKIP する（重みがリポジトリ管理外）。生成: ${series.generate}`,
    );
  } else if (!found.realAvailable) {
    console.warn(
      `[karume] ${series.name} の実画像ケースを SKIP する（golden ${found.realCases.length}/` +
        `${REAL_CASES.length} 本）。生成: ${series.generate}`,
    );
  }
}

for (const series of SERIES) {
  const found = discoveryOf(series);
  const root = seriesRoot(series);

  Deno.test({
    name: `BiRefNet 資産: ${series.name} — 期待するケースとモデル本体が揃っている`,
    // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
    ignore: !found.available,
    fn: () => {
      assertEquals(found.cases, [...EXPECTED_CASES], `${root.pathname} の合成画像 golden ケース`);
      // 実画像は**任意だが全部か 0 か**（`--real-images` を付けた emit は 4 本まとめて書く）。
      // 部分的な欠けを SKIP に丸めると、採り直しの途中で落ちた資産が黙って通る。
      assert(
        found.realCases.length === 0 || found.realCases.length === REAL_CASES.length,
        `${root.pathname} の実画像 golden が ${found.realCases.length}/${REAL_CASES.length} 本` +
          `（採り直す: ${series.generate}）`,
      );
      const model = new URL(MODEL_FILE, root);
      assert(Deno.statSync(model).isFile, `${MODEL_FILE} が無い`);
    },
  });

  /**
   * 突合を回すケース。合成と実画像で**入力の作り方は同じ**（どちらも golden の
   * `pixel_values`）で、違うのは踏む分布と、そこから独立に導いた tolerance だけ。
   */
  const goldenCases = [
    ...found.cases.map((name) => ({
      name,
      label: "golden 突合",
      tolerance: series.tolerance,
      ignore: !found.available,
    })),
    ...found.realCases.map((name) => ({
      name,
      label: "実画像 golden 突合",
      tolerance: series.realTolerance,
      ignore: !found.realAvailable,
    })),
  ];

  for (const entry of goldenCases) {
    const caseName = entry.name;
    Deno.test({
      name: `BiRefNet ${entry.label}: ${series.name} / ${caseName}（実 GPU / torch CPU 期待値）`,
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

          const [name] = parsed.graph.outputs;
          const view = io.tensors.get("output.0");
          assert(view !== undefined, "output.0 が golden に無い");
          const where = `${series.name} / ${caseName} output.0 ('${name}')`;
          const declared = parsed.graph.values[name].dtype;
          assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
          assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
          const expected = ioTensor(io, view, declared);
          const report = compareTensors(outputs[name], expected, entry.tolerance);
          assert(report.pass, `${where}: ${formatAllclose(report)}`);

          // 値の近さとは別に、**マスクとしての判断**が torch と一致すること。
          assert(outputs[name].dtype === "f32" && expected.dtype === "f32", `${where}: f32 でない`);
          const disagreement = maskDisagreement(outputs[name].data, expected.data);
          assert(
            disagreement <= MASK_DISAGREEMENT_LIMIT,
            `${where}: 二値マスクの不一致 ${(disagreement * 100).toFixed(4)}%` +
              `（上限 ${MASK_DISAGREEMENT_LIMIT * 100}%）`,
          );
        } finally {
          await session.dispose();
          gpu.destroy();
        }
      },
    });
  }

  Deno.test({
    name: `BiRefNet 幾何判別: ${series.name} — disc の円内 logit 平均が円外を上回る`,
    ignore: !found.available || !GPU_AVAILABLE,
    fn: async () => {
      // golden 突合だけだと「期待値と合っている」ことしか言えず、マットとして意味のある出力かは
      // 別問題（一様に潰れた出力は期待値も同じく潰れていれば通ってしまう）。ここは**画像の中の
      // 既知の幾何**（円）で切るので、出力が一様なら平均が並んで落ちる。閾値は置かない
      // （順序そのものが検査対象 — `birefnet/export.py` の `_sanity` と同じ形で、あちらは
      // torch 側に掛かっている）。実測は円内 / 円外が BiRefNet_HR で +10.96 / −9.64、
      // Lucida で +3.37 / −9.69（合成画像に対する応答の深さが系列で違う）。
      const modelBytes = await readBuffer(root, MODEL_FILE);
      const ioBytes = await readBuffer(root, `${IO_PREFIX}${DISC_CASE}${IO_SUFFIX}`);
      const parsed = openModel(modelBytes);
      const io = parseSafetensors(ioBytes);
      const size = staticDim(parsed, 3);
      assertEquals(size, staticDim(parsed, 2), "円の判別は正方形の入力を前提にする");

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      try {
        const [name] = parsed.graph.outputs;
        const output = (await session.run(goldenInputs(parsed, io)))[name];
        assert(output.dtype === "f32", `disc のマットの dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, 1, size, size], "マットの形");
        const { inside, outside } = discMeans(output.data, size);
        assert(
          inside > outside,
          `disc の円内 logit 平均 ${inside} が円外 ${outside} 以下 — 顕著物体を分離できていない`,
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}
