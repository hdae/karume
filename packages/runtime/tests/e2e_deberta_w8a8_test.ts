// 実重み DeBERTa-v2 の **w8a8 鏡像門**（実 GPU・ADR 0026 決定 3）。
//
// `e2e_deberta_test.ts` が w8（i8 格納・f32 計算）を torch CPU の通常 golden `io.<case>` と突き
// 合わせるのに対し、こちらは w8a8（`SessionOptions.linearCompute: "a8"` — 活性を per-token i8
// へ落として整数内積で計算する）を、同じ入力で採った **torch 鏡像 golden `io-i8a8.<case>`**
// （`--act-quant` — 数値の正本は exporter の `karume.act_quant`）と突き合わせる。2 つを別
// ファイルに分けるのは期待値の系列が違うからで、片方の golden をもう片方の門で読むと活性量子化
// ごと汚染される（鏡像側の prefix を分けてある理由と同じ — `deberta/export.py` の
// `ACT_IO_PREFIX`）。
//
// 対象は `outputs/series/deberta-i8/full-24layer/`（リポジトリ管理外 — `.gitignore` の
// `outputs/`）。鏡像 io は同じ export の `--act-quant` で同じ席に書かれる。
//
// ## 検出力の置き場（ADR 0026「検出限界」）
//
// 活性の丸めは不連続関数で、上流の 1e-5 級の差が丸め境界の ±1 段飛びを起こし、数層で飽和する。
// GPU と torch 鏡像は深い層では「同じ分布の別標本」になり、末端層では f32 経路 vs 鏡像と
// 区別がつかない（歴史値: i8a8 1.46 / f32 経路 1.33）。素直な「実測の 5〜10 倍」を全出力に
// 掛けると f32 経路まで通して恒真化する。そこで検出力を 3 つに分けて置く:
//
// 1. **output.0 / output.1 の厳密 tolerance**（{@link STRICT_TOLERANCE}）— 判別帯が残るのは
//    1 層目の出口だけ（歴史値: i8a8 5.72e-6 vs f32 経路 5.76e-2 = 10,058 倍）。
//    `linearCompute` が黙って f32 経路へ落ちると差は 5.76e-2 級に**開く**ので、ここで落ちる。
// 2. **パイプラインキーの census**（{@link EXPECTED_CENSUS}）— run が実際に i8a8 GEMM と
//    `quantize_rows` を linear の本数ぶん回し、それ以外の linear カーネルを 1 回も回していない
//    こと。1 が分布の話であるのに対し、こちらは実行そのものの直接観測。
// 3. **output.2 以降の崩壊上限**（{@link COLLAPSE_CEILINGS}・判定は {@link judgeCollapse}）—
//    数値パリティではなく、崩壊の検出だけを受け持つ。指標は出力ごとの相対 RMS 誤差
//    ‖gpu − golden‖₂ / ‖golden‖₂ ≤ その出力の上限（output.2 の 0.020 〜 output.24 の 0.33）と、
//    非有限 0（両側・全要素）。
//    - 捕まえるもの（実 GPU の dump を壊して同じ判定に通した故障注入 — 4 ケース × output.2〜24
//      の 92 出力の全てで赤）: 全要素 0（1.0）・符号反転（≈ 2.0）・1 要素だけの NaN / +Inf・
//      全要素 × 2（1.00〜1.02）・全要素 × 1.5（0.50〜0.53）。
//    - 捕まえないもの: 深い出力の × 1.25 の拡大（output.22〜24 の 12 出力のうち 11 が緑 — 92 出力の
//      うち 81 が赤）、中間層の 1 要素の飛び、f32 経路への沈黙フォールバック（GPU f32 経路の dump は
//      92 出力の全てで上限の内 — 0.016〜0.120）。これらは 1・2 と
//      `tests/gpu_i8a8_test.ts` の atol=0 の数値契約が受け持つ（24 層は同じ形の同じパイプラインを
//      回るので、浅い層の欠陥は 1 の output.1 にも出る）。
//    - 上限の導き方: 出力ごとに、正当な標本（GPU a8・この機の CPU で採り直した torch 鏡像の
//      別標本 6 本・鏡像の埋め込み出口に相対 ±2^-23 の乱数を入れた摂動アンサンブル各ケース 8 本）
//      の最悪 × 2 を有効数字 2 桁で切り上げた値。標本の内訳と出力ごとの最悪・上限の表は ADR 0026 の
//      追記（2026-10-07）。
//    - maxAbs の atol 3 をやめた理由: atol 3 は ADR 0026 の**末端層（output.24）だけ**の歴史値
//      1.46 の約 2 倍で、それを中間層まで maxAbs で掛けていた。中間層は外れ値チャネル
//      （channel 686・\|ref\| 最大 28.8）の符号の分岐点で活性量子化の段の反転が 1 要素を大きく
//      動かし、CPU の torch 鏡像の別標本どうしでも同じ 1 要素が 7.17 動く（RTX 3080 Ti の case2
//      output.19 が 3.54 で赤になったのはこの形 — GPU の欠陥ではない）。
//
// ## 値の出どころ
//
// 1 の数値は ADR 0026（2026-08-03・RTX 3080 Ti・torch 鏡像との素の突合）の**歴史値**で、
// この環境で導き直していない。3 の数値は 2026-10-07 にこの環境（RTX 3080 Ti・Ryzen 5 7600）で
// 実測から導き直した値。この門は output.0 / 1 の maxAbs（`measurements`）と output.2 以降の
// 相対 RMS 誤差・maxAbs（`comparisons`）を `outputs/verify/` の results.json に残す（落ちた回も
// 全出力を測り終えてから落とす）ので、導き直しはその実測を読む。
//
// ## 鏡像 golden を採り直さない
//
// 鏡像 golden（`io-i8a8.<case>`）は 2026-09-05 に換装前の機の CPU で採った。1 の厳密 tolerance は
// golden を採った CPU の第 0 層の量子化の段の丸めの向きに依存する: この機の CPU で採り直すと
// case0 / padded の output.1 が保存版から 1.285e-3 動き（GPU は旧機の CPU と同じ側）、atol 5e-5 の
// 1 が赤になる。採り直しは解決にならないので、別の機の CPU で鏡像 golden を作り直さない。
//
// ## 宣言しない variant
//
// `deberta-i8/sbv2-22layer` は最終層 1 本出しで、出力が飽和域（1 の判別帯の外）にしか無い —
// 3 の崩壊上限と 2 の census しか掛けられず、しかも崩壊上限は full-24layer の標本からしか
// 導いていない。`deberta-i8/dev-2layer` も実測が無い。足すときは `atol=rtol=0` の素の突合と
// 別標本で実測してから宣言する（値を発明しない）。
//
// 資産が無い環境では SKIP する（GPU アダプタの有無を見る ADR 0005 の門番とは独立）。鏡像が
// **一部だけ**ある場合は SKIP ではなく FAIL にする（下の「資産の完全性」テスト）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type SessionDiagnostics,
  type Tensor,
} from "../mod.ts";
import {
  AllcloseError,
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus, SEAT_SIGNATURES } from "./helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { modelPresent, openSeriesContainer } from "./helpers/container-files.ts";
import { seriesGraph } from "./helpers/series-graphs.ts";

/**
 * output.0 / output.1 に掛ける厳密 tolerance（**歴史値** — ADR 0026 決定 3）。
 *
 * ADR 0026 の実測（full-24layer・GPU i8a8 vs torch 鏡像）: output.1 の maxAbs 5.72e-6。
 * f32 経路 vs 鏡像は同じ output.1 で 5.76e-2。atol 5e-5 は前者の約 8.7 倍・後者の 1/1,150 で、
 * 活性量子化が走らなかった run（f32 経路への沈黙フォールバック）はここで落ちる。
 *
 * output.0 は埋め込み層の出口で linear を通らない（w8 門の実測 9.54e-7 と同じ経路）ので、
 * 同じ値で足りる。
 *
 * **rtol は 0**（w8 門と同じ理由 — \|ref\| の最小非ゼロが 1e-7 級まで薄く広がり、rtol は
 * 0 近傍要素の見かけに引きずられる）。
 */
const STRICT_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 0 };

/** 厳密 tolerance を掛ける出力の本数（output.0 と output.1 — 判別帯が残る 1 層目の出口まで）。 */
const STRICT_OUTPUTS = 2;

/**
 * output.2 以降に掛ける**崩壊上限**の表 = 出力名 → 相対 RMS 誤差 ‖gpu − golden‖₂ / ‖golden‖₂ の
 * 上限（**この環境で実測から導いた宣言値** — 2026-10-07・RTX 3080 Ti / Ryzen 5 7600・ADR 0026
 * 追記 2026-10-07）。宣言であって環境キー別の行ではない（ADR 0110 決定 4）。
 *
 * 導き方: 出力ごと（観測点 = output.k）に、正当な標本 60 本（4 ケース × 〈GPU a8 1 本 + この機の
 * CPU で採り直した torch 鏡像の別標本 6 本〈AVX512 / AVX2 / DEFAULT × 1 / 6 スレッド〉+ 鏡像の
 * 埋め込み出口に相対 ±2^-23 の乱数を入れた摂動アンサンブル 8 本〉）の最悪 × 2 を、有効数字 2 桁で
 * 切り上げた値。この門は ADR 0110 決定 5 の A/B 門ではない（比較相手は同じ機の参照層ではなく torch
 * 鏡像 golden）が、崩壊上限の導き方だけを決定 5-3（E2E の崩壊上限 = 観測点の実測 × 2 程度）に揃え、
 * 決定 4 の MUST NOT（「実測の 5〜10 倍」を E2E の帯の導出に使う）を踏まない。5-3 の「量子化席なら
 * 理論値との整合を残す」は当てはめない: 活性の丸めは不連続で、上流の 1e-5 級の差が段の ±1 飛びを
 * 起こして数層で飽和する（ADR 0026「検出限界」）ので、output.k の相対 RMS 誤差に使える理論上界が
 * 無い — a8 の数値の正しさは `tests/gpu_i8a8_test.ts` の atol=0 の数値契約が受け持つ。GPU f32 経路の
 * dump は a8 の標本ではないので入れない。最悪は浅い出力ほど小さい（output.2 0.0099 → output.24 0.1620）ので、
 * 1 つのスカラーで掛けると浅い出力の上限が桁違いに緩む — 表にするのはそのため。
 *
 * 材料と再現: `outputs/diag/deberta-w8a8-2026-10-07/d1-derive/`（git 追跡外）。tools/export-recipes
 * から `HF_HUB_OFFLINE=1 uv run --with 'transformers==5.14.1' python
 * ../../outputs/diag/deberta-w8a8-2026-10-07/d1-derive/derive.py` が `derive.json` の `ceilings` に
 * この表を書く（標本の作り方は ADR 0026 追記の「導き直しの手順」）。故障注入は同じ置き場の
 * `fault_inject.ts`（{@link collapseCeilingOf} と {@link judgeCollapse} をそのまま import する）。
 *
 * 数値パリティではない: 受け持つのは非有限（{@link judgeCollapse} が両側・全要素を見る）と、出力
 * 全体が golden と別物になる崩壊（全要素 0 = 1.0・符号反転 ≈ 2.0・全要素 × 2 ≈ 1.0・× 1.5 ≈ 0.5）
 * だけ。中間層の 1 要素の飛びや f32 経路への沈黙フォールバックは捕まえない — 細かな欠陥は
 * output.0 / 1 の厳密 tolerance・census・`tests/gpu_i8a8_test.ts` の atol=0 の数値契約の受け持ち。
 * 上限を広げて通すのではなく、先に i8a8 の scale / accumulator / 適格判定のどれが動いたかを確かめる。
 */
const COLLAPSE_CEILINGS: Readonly<Record<string, number>> = {
  "output.2": 0.020,
  "output.3": 0.037,
  "output.4": 0.051,
  "output.5": 0.062,
  "output.6": 0.071,
  "output.7": 0.085,
  "output.8": 0.095,
  "output.9": 0.11,
  "output.10": 0.12,
  "output.11": 0.13,
  "output.12": 0.14,
  "output.13": 0.15,
  "output.14": 0.15,
  "output.15": 0.17,
  "output.16": 0.17,
  "output.17": 0.19,
  "output.18": 0.21,
  "output.19": 0.22,
  "output.20": 0.24,
  "output.21": 0.26,
  "output.22": 0.28,
  "output.23": 0.31,
  "output.24": 0.33,
};

/**
 * io の出力名（`output.<k>`）の崩壊上限。表に無い出力は投げる（上限の無い出力を黙って通さない）。
 */
export const collapseCeilingOf = (output: string): number => {
  if (!Object.hasOwn(COLLAPSE_CEILINGS, output)) {
    throw new Error(`${output} の崩壊上限が宣言されていない（COLLAPSE_CEILINGS）`);
  }
  return COLLAPSE_CEILINGS[output];
};

/** 崩壊上限の判定 1 出力ぶん（{@link judgeCollapse}）。 */
export type CollapseReport = {
  readonly pass: boolean;
  /** ‖actual − expected‖₂ / ‖expected‖₂（f64 で積む）。非有限があれば +Inf。 */
  readonly relRms: number;
  /** 記録だけ — 判定には使わない。非有限があれば +Inf。 */
  readonly maxAbs: number;
  /** どちらかの側が NaN / ±Inf だった要素数（全要素を見る）。 */
  readonly nonFiniteCount: number;
};

/**
 * output.2 以降の 1 出力の判定: 非有限 0（両側・全要素）かつ相対 RMS 誤差 ≤ `ceiling`（その出力の
 * 上限 — {@link collapseCeilingOf}）。
 *
 * maxAbs で判定しない理由: 中間層は外れ値チャネル（channel 686・\|ref\| 最大 28.8）の符号の
 * 分岐点で、活性量子化の段の反転が 1 要素を大きく動かす。CPU の torch 鏡像の別標本どうしでも
 * その 1 要素が 7.17 動く（ADR 0026 追記 2026-10-07）。
 */
export const judgeCollapse = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  ceiling: number,
): CollapseReport => {
  if (actual.length !== expected.length) {
    throw new AllcloseError(`長さ不一致: actual ${actual.length} vs expected ${expected.length}`);
  }
  let difference = 0;
  let norm = 0;
  let maxAbs = 0;
  let nonFiniteCount = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const x = actual[index];
    const y = expected[index];
    // MUST: 非有限は相対 RMS の大小に任せず数えて落とす — 合否を比較の向き（NaN はどの比較も
    // false）に依存させず、報告にも何が起きたかを出す。
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      nonFiniteCount += 1;
      continue;
    }
    const delta = x - y;
    difference += delta * delta;
    norm += y * y;
    maxAbs = Math.max(maxAbs, Math.abs(delta));
  }
  if (nonFiniteCount > 0) {
    return {
      pass: false,
      relRms: Number.POSITIVE_INFINITY,
      maxAbs: Number.POSITIVE_INFINITY,
      nonFiniteCount,
    };
  }
  const relRms = Math.sqrt(difference) / Math.sqrt(norm);
  // ‖expected‖₂ = 0 は 0/0 = NaN か +Inf になり、`<=` が false を返して落ちる。
  return { pass: relRms <= ceiling, relRms, maxAbs, nonFiniteCount };
};

/**
 * 1 run あたりの dispatch の内訳の期待値（**グラフと実装から導いた値** — ADR 0026 の診断キー
 * 192 本と一致）。
 *
 * - `i8a8Linear` 192 = 24 層 × 8。DeBERTa-v2 は `share_att_key` で query_proj / key_proj を
 *   相対位置射影にも使うので、1 層あたりモジュール 6 本に対しグラフの linear ノードは 8 本
 *   （full-24layer の IR を読んで確認: linear 192 本・重みは全て i8 格納・k ∈ {1024, 4096} で
 *   全て a8 適格〈k % 4 == 0〉）。
 * - `quantizeRows` 192 = 同じ本数。a8 の linear は「活性を per-token i8 へ落とす
 *   `quantize_rows` → 整数内積の GEMM」の対で降りる（`recipe-builders/linear.ts` の
 *   `buildLinearI8a8`）。入力を共有する linear どうしでも量子化は束ねない。
 * - `otherLinear` 0 = i8a8 以外の linear カーネル（GEMM / GEMV 族のどれでも）。1 本でも
 *   走れば、適格判定が外れて一部が黙って f32 経路へ落ちている。
 */
const EXPECTED_CENSUS: KeyCensus = { i8a8Linear: 192, quantizeRows: 192, otherLinear: 0 };

const MODEL_FILE = "model.krm";
const GRAPH_OUTPUTS = 25;
/** 鏡像 io の prefix（`deberta/export.py` の `ACT_IO_PREFIX` と同じ綴り MUST）。 */
const ACT_IO_PREFIX = "io-i8a8.";
const IO_SUFFIX = ".safetensors";

const ROOT = new URL("../../../outputs/series/deberta-i8/full-24layer/", import.meta.url);
const GRAPH = seriesGraph("deberta-i8", "full-24layer");
const VARIANT = "i8/full-24layer";
/** SKIP 時にそのまま貼れる生成コマンド（tools/export-recipes/deberta/README.md）。 */
const GENERATE = "cd tools/export-recipes && uv run --with 'transformers==5.14.1' " +
  "python -m deberta.export --dtype i8 --layers 24 --act-quant";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する** — 生成を一部だけ流した
 * 環境でテストが黙って消えないため（正本は `deberta/export.py` の `GOLDEN_SENTENCES` +
 * `padded`）。
 */
const EXPECTED_CASES = ["case0", "case1", "case2", "padded"] as const;

/** 1 run ぶんの分類済み dispatch 数。 */
type KeyCensus = {
  readonly i8a8Linear: number;
  readonly quantizeRows: number;
  readonly otherLinear: number;
};

/**
 * パイプラインキーの分類（綴りの正本は `src/kernels/linear{,-i8a8,-gemv}.ts` /
 * `quantize-rows.ts`）。
 *
 * MUST: linear 側は `"linear"` の前方一致で数える（`"linear:"` ではなく）。この門の T は
 * 11〜35 で、f32 経路へ落ちた linear は GEMM（`linear:`）ではなく行ブロック GEMV
 * （`linear_gemv:` — M ≤ 64）へ降りる。`"linear:"` で数えると、そのフォールバックが
 * `otherLinear` に入らず 0 のまま通る。
 */
const isLinearKey = (key: string): boolean => key.startsWith("linear");
const isI8a8 = SEAT_SIGNATURES.linearCompute.a8;
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

const censusOf = (diagnostics: SessionDiagnostics): KeyCensus => {
  // MUST: census の無い run を「0 本」として数えない — `otherLinear: 0` の検査が黙って空振り
  // する（requireCensus が undefined / 空を落とす）。
  const entries = requireCensus(diagnostics.lastRunPipelines, "deberta w8a8");
  let i8a8Linear = 0;
  let quantizeRows = 0;
  let otherLinear = 0;
  for (const entry of entries) {
    if (isLinearKey(entry.key)) {
      if (isI8a8(entry.key)) i8a8Linear += entry.dispatchCount;
      else otherLinear += entry.dispatchCount;
    } else if (isQuantizeRows(entry.key)) {
      quantizeRows += entry.dispatchCount;
    }
  }
  return { i8a8Linear, quantizeRows, otherLinear };
};

/**
 * 鏡像 io の列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、実行されて
 * いない検証が SKIP として静かに緑になる。
 */
const mirrorFiles = (root: URL): readonly string[] => {
  let entries: readonly Deno.DirEntry[];
  try {
    entries = [...Deno.readDirSync(root)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
  return entries
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(ACT_IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name)
    .sort();
};

const caseNameOf = (file: string): string =>
  file.slice(ACT_IO_PREFIX.length, file.length - IO_SUFFIX.length);

const readBuffer = async (file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, ROOT));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** 登録時点で必要なので同期列挙する（Deno.test の ignore 判定と同じ理由）。 */
const FILES = mirrorFiles(ROOT);
/**
 * 鏡像の有無。1 件も無い = `--act-quant` なしで生成した（または生成していない）環境なので
 * 全 SKIP。export は出力席を丸ごと据え替える（`karume.artifacts.staged_publication`）ので、
 * 古い鏡像が新しいモデルの横に残る形は作られない — 無いことは正当な状態として読んでよい。
 */
const AVAILABLE = FILES.length > 0;
const RUNNABLE = AVAILABLE && GPU_AVAILABLE;

if (!AVAILABLE) {
  console.warn(
    `[karume] ${ROOT.pathname} に w8a8 鏡像 io（${ACT_IO_PREFIX}<case>）が無いため ` +
      `DeBERTa の w8a8 鏡像門（${VARIANT}）を SKIP する。生成: ${GENERATE}`,
  );
}

describe("崩壊上限の判定（output.2 以降 — GPU も資産も要らない）", () => {
  // 固定の合成データ: 1 チャネルだけ桁の大きい外れ値チャネルを持つ golden（実物の channel 686 の形）。
  const SIZE = 1024;
  const OUTLIER = 686;
  const golden = Float32Array.from(
    { length: SIZE },
    (_, index) => index === OUTLIER ? 28.8 : Math.sin(index * 0.37 + 0.1) * (1 + (index % 7) * 0.3),
  );
  const normOf = (values: ArrayLike<number>): number =>
    Math.sqrt(Array.from(values, (value) => value * value).reduce((sum, value) => sum + value, 0));
  /** golden に、相対 RMS 誤差がちょうど `target` になる揺れ（golden と別周期）を足した別標本。 */
  const sampleAt = (target: number): Float32Array => {
    const noise = Array.from({ length: SIZE }, (_, index) => Math.cos(index * 1.13 + 0.7));
    const scale = target * normOf(golden) / normOf(noise);
    return Float32Array.from(golden, (value, index) => value + noise[index] * scale);
  };
  const map = (values: Float32Array, f: (value: number) => number): Float32Array =>
    Float32Array.from(values, f);
  const withElement = (values: Float32Array, index: number, value: number): Float32Array => {
    const copy = values.slice();
    copy[index] = value;
    return copy;
  };
  /** 表の全出力（output.2〜24）と、その上限。 */
  const ceilings = Object.entries(COLLAPSE_CEILINGS);
  /**
   * 各出力の上限の半分の相対 RMS を持つ別標本。上限は導出に使った最悪の × 2 以上なので、
   * これは導出に使った正当な最悪以上の揺れを持つ（正当な標本の代表）。
   */
  const legitimateFor = (ceiling: number): Float32Array => sampleAt(ceiling / 2);
  const deepest = collapseCeilingOf("output.24");
  /**
   * 上限が output.24 の正当な揺れ（上限の半分 ≈ 0.165 — 導出の最悪 0.1620 程度）より狭い最後の
   * 出力（ADR 0026 追記 2026-10-07 の表: output.14 の上限 0.15 < 0.165 ≤ output.15 の 0.17）。
   */
  const LAST_TIGHTER_THAN_DEEPEST = 14;
  const indexOf = (output: string): number => Number(output.slice("output.".length));

  it("上限の表は output.2〜24 をちょうど覆い、表に無い出力は投げる", () => {
    assertEquals(
      ceilings.map(([output]) => output),
      Array.from(
        { length: GRAPH_OUTPUTS - STRICT_OUTPUTS },
        (_, k) => `output.${k + STRICT_OUTPUTS}`,
      ),
    );
    for (const output of ["output.1", `output.${GRAPH_OUTPUTS}`]) {
      assertThrows(() => collapseCeilingOf(output), Error, output);
    }
  });

  it("上限は出力の順に非減少（活性量子化の誤差は層を下るほど積もる — 導出の最悪も単調）", () => {
    ceilings.slice(1).forEach(([output, ceiling], position) => {
      const [previousOutput, previous] = ceilings[position];
      assert(previous <= ceiling, `${previousOutput} の ${previous} > ${output} の ${ceiling}`);
    });
  });

  it("output.24 の正当な揺れの大きさの別標本は、output.2〜14 では赤・output.15〜24 では通す（上限は出力ごと）", () => {
    const deepLegitimate = legitimateFor(deepest);
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(deepLegitimate, golden, ceiling);
      assertEquals(
        report.pass,
        indexOf(output) > LAST_TIGHTER_THAN_DEEPEST,
        `${output}: relRms=${report.relRms} 上限 ${ceiling}`,
      );
    }
  });

  it("各出力で、上限の半分（導出の最悪以上）の別標本は通し、上限を 1% 超えた別標本は赤にする", () => {
    for (const [output, ceiling] of ceilings) {
      const inside = judgeCollapse(legitimateFor(ceiling), golden, ceiling);
      assert(inside.pass, `${output}: relRms=${inside.relRms} 上限 ${ceiling}`);
      const outside = judgeCollapse(sampleAt(ceiling * 1.01), golden, ceiling);
      assertEquals(outside.pass, false, `${output}: relRms=${outside.relRms} 上限 ${ceiling}`);
    }
  });

  it("外れ値チャネルの 1 要素が 7 動いても（maxAbs では落ちた形）全体が近ければ output.19 の上限で通す", () => {
    const ceiling = collapseCeilingOf("output.19");
    const report = judgeCollapse(withElement(golden, OUTLIER, 28.8 - 7.17), golden, ceiling);
    assert(report.pass, `relRms=${report.relRms} 上限 ${ceiling}`);
    assert(report.maxAbs > 7, `maxAbs=${report.maxAbs}`);
  });

  it("全要素 0 を全出力で赤にする（相対 RMS 1.0）", () => {
    for (const [output, ceiling] of ceilings) {
      const report = judgeCollapse(new Float32Array(SIZE), golden, ceiling);
      assertEquals([report.pass, report.relRms], [false, 1], output);
    }
  });

  it("符号反転を全出力で赤にする", () => {
    for (const [output, ceiling] of ceilings) {
      const flipped = map(legitimateFor(ceiling), (value) => -value);
      assertEquals(judgeCollapse(flipped, golden, ceiling).pass, false, output);
    }
  });

  it("全要素 × 2 と × 1.5 の拡大を全出力で赤にする（相対 RMS 1.0 / 0.5）", () => {
    for (const factor of [2, 1.5]) {
      const scaled = map(golden, (value) => value * factor);
      for (const [output, ceiling] of ceilings) {
        const report = judgeCollapse(scaled, golden, ceiling);
        assertEquals(
          report.pass,
          false,
          `× ${factor} ${output}: relRms=${report.relRms} 上限 ${ceiling}`,
        );
      }
    }
  });

  it("1 要素だけの NaN / +Inf / -Inf を、実行側でも期待値側でも赤にする", () => {
    const legitimate = legitimateFor(deepest);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const onActual = judgeCollapse(withElement(legitimate, 3, bad), golden, deepest);
      assertEquals([onActual.pass, onActual.nonFiniteCount], [false, 1], `actual に ${bad}`);
      const onExpected = judgeCollapse(legitimate, withElement(golden, 3, bad), deepest);
      assertEquals([onExpected.pass, onExpected.nonFiniteCount], [false, 1], `expected に ${bad}`);
    }
  });

  it("長さが違えば判定せずに投げる（取り違え — 誤差の問題ではない）", () => {
    assertThrows(() => judgeCollapse(golden.subarray(1), golden, deepest), AllcloseError);
  });
});

Deno.test({
  name: `DeBERTa w8a8 資産（${VARIANT}）: 鏡像 io の全ケースとモデル本体が揃っている`,
  // 鏡像が 1 つでもあれば欠けは FAIL（生成を途中で止めた席を「未生成」と読まない）。
  ignore: !AVAILABLE,
  fn: () => {
    assertEquals(FILES.map(caseNameOf), [...EXPECTED_CASES], `${ROOT.pathname} の鏡像ケース`);
    assert(modelPresent(new URL(MODEL_FILE, ROOT)), `${MODEL_FILE} が無い`);
  },
});

/** Session を a8 で組み、鏡像 io の入力で 1 回 run する（数値門と census の共通部）。 */
const openCase = async (file: string) => {
  const [opened, ioBytes] = await Promise.all([
    openSeriesContainer(new URL(MODEL_FILE, ROOT)),
    readBuffer(file),
  ]);
  const parsed = prepareContainer(opened, GRAPH);
  const io = parseSafetensors(ioBytes);
  // 全層出し（25 本）であることを先に見る。配布形（1 本出し）の資産を取り違えると厳密
  // tolerance を掛ける output.1 がそもそも無く、門の中身が崩壊上限だけに縮む。
  assertEquals(parsed.graph.outputs.length, GRAPH_OUTPUTS, `${VARIANT}: graph.outputs の本数`);
  // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
  const expectedKeys = [
    ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
    ...parsed.graph.outputs.map((_, index) => `output.${index}`),
  ].sort();
  assertEquals([...io.tensors.keys()].sort(), expectedKeys, `${file} のテンソルキー`);
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が ${file} に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return { parsed, io, inputs };
};

/**
 * 決着と実測の置き場（`outputs/verify/<環境キー>/<日付>_deberta-w8a8/` — 消して安全）。
 * w8 門（`deberta-golden`）と席を分けるのは、同じ席に 2 モジュールが書くと互いの `cases` を
 * 上書きするため。
 */
const results = openResults("deberta-w8a8");

for (const file of FILES) {
  const caseName = caseNameOf(file);
  Deno.test({
    name: `DeBERTa w8a8 鏡像突合: ${VARIANT} / ${caseName}（実 GPU / torch 鏡像期待値）`,
    ignore: !RUNNABLE,
    fn: async () => {
      await runRecordedCase(results, { id: `${VARIANT}/${caseName}` }, async (recorded) => {
        const { measurements, comparisons } = recorded;
        const { parsed, io, inputs } = await openCase(file);
        const gpu = await acquireGpu();
        /** 帯を外れた出力（全出力を測り終えてから落とす — 導き直しに全出力の実測が要る）。 */
        const failures: string[] = [];
        try {
          // 結果をこの機の行として残す経路なので、キーを採ったアダプタと実行アダプタの同一性を
          // 見る（複数 GPU の機で取り違えると、別の機の帯で測ることになる）。
          assertAdapterMatchesEnvironment(gpu);
          // Session の構築が落ちた回も device を手放す（構築の失敗を try の外に置くと漏れる）。
          const session = await parsed.createContainerSession(gpu, { linearCompute: "a8" });
          try {
            const outputs = await session.run(inputs);
            assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
            parsed.graph.outputs.forEach((name, index) => {
              // MUST: io の引きと崩壊上限の引きは同じ名前で行う（鍵の取り違えで別の出力の上限を掛けない）。
              const ioName = `output.${index}`;
              const view = io.tensors.get(ioName);
              assert(view !== undefined, `${ioName} が ${file} に無い`);
              const where = `${VARIANT}/${caseName} ${ioName} ('${name}')`;
              const declared = parsed.graph.values[name].dtype;
              assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
              assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
              const expected = ioTensor(io, view, declared);
              if (index < STRICT_OUTPUTS) {
                const report = compareTensors(outputs[name], expected, STRICT_TOLERANCE);
                measurements.push({
                  output: name,
                  maxAbs: report.maxAbsError,
                  maxRel: report.maxRelError,
                  tolerance: STRICT_TOLERANCE,
                  stage: "karume",
                });
                if (!report.pass) failures.push(`${where}: ${formatAllclose(report)}`);
                return;
              }
              // 崩壊上限の出力は `comparisons` に積む（相対 RMS と maxAbs を 1 本で表せる既存の
              // 形 — 参照側 = torch 鏡像 golden・実行側 = a8。床は 0 = 床なし）。
              const ceiling = collapseCeilingOf(ioName);
              const report = judgeCollapse(outputs[name].data, expected.data, ceiling);
              comparisons.push({
                output: name,
                reference: `${ACT_IO_PREFIX}${caseName}`,
                practical: "a8",
                relRms: report.relRms,
                maxAbs: report.maxAbs,
                band: { metric: "relRms", floor: 0, ceiling },
              });
              if (!report.pass) {
                failures.push(
                  `${where}: relRms=${report.relRms} (上限 ${ceiling}) ` +
                    `nonFinite=${report.nonFiniteCount} maxAbs=${report.maxAbs}`,
                );
              }
            });
          } finally {
            await session.dispose();
          }
        } finally {
          gpu.destroy();
        }
        assertEquals(failures, [], "帯を外れた出力（厳密: output.0/1・崩壊上限: output.2〜）");
      });
    },
  });
}

Deno.test({
  name:
    `DeBERTa w8a8（${VARIANT}）: run が i8a8 GEMM と quantize_rows だけで linear を回す（キー census）`,
  ignore: !RUNNABLE,
  fn: async () => {
    // 内訳は T に依らない（計画は記号次元の束縛で形が変わるだけで、dispatch の並びは同じ）ので
    // 1 ケースで足りる。`padded` を使うのはマスク経路まで同じ run に乗せるため。
    const file = `${ACT_IO_PREFIX}padded${IO_SUFFIX}`;
    const { parsed, inputs } = await openCase(file);
    const gpu = await acquireGpu();
    try {
      const session = await parsed.createContainerSession(gpu, { linearCompute: "a8" });
      try {
        await session.run(inputs);
        const census = censusOf(session.diagnostics());
        console.log(
          `[e2e] deberta ${VARIANT} w8a8 census: i8a8 linear ${census.i8a8Linear} 本 / ` +
            `quantize_rows ${census.quantizeRows} 本 / それ以外の linear ${census.otherLinear} 本`,
        );
        assertEquals(
          census,
          EXPECTED_CENSUS,
          "dispatch の内訳が期待と違う（otherLinear > 0 は f32 経路への沈黙フォールバック・" +
            "本数の違いは linear の本数か降ろし方が動いた）",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});
