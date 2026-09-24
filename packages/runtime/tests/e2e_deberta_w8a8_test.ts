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
// 3. **output.2 以降の崩壊上限**（{@link COLLAPSE_TOLERANCE}）— 数値パリティではなく、NaN /
//    発散 / 桁外れの値の検出だけを受け持つ。
//
// ## 値はすべて歴史値
//
// 1 と 3 の数値は ADR 0026（2026-08-03・RTX 3080 Ti・torch 鏡像との素の突合）の値を起点に
// 置いたもので、**この環境で実測から導き直していない**。この門は全出力の maxAbs を
// `outputs/verify/` の results.json に残す（落ちた回も全出力を測り終えてから落とす）ので、
// 導き直しはその実測を読む。
//
// ## 宣言しない variant
//
// `deberta-i8/sbv2-22layer` は最終層 1 本出しで、出力が飽和域（1 の判別帯の外）にしか無い —
// 3 の崩壊上限と 2 の census しか掛けられず、しかもその歴史値が無い。`deberta-i8/dev-2layer`
// も歴史値が無い。足すときは `atol=rtol=0` の素の突合で実測してから宣言する（値を発明しない）。
//
// 資産が無い環境では SKIP する（GPU アダプタの有無を見る ADR 0005 の門番とは独立）。鏡像が
// **一部だけ**ある場合は SKIP ではなく FAIL にする（下の「資産の完全性」テスト）。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type SessionDiagnostics,
  type Tensor,
} from "../mod.ts";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment } from "./helpers/environment.ts";
import { ioTensor } from "./helpers/golden-io.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";
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
 * output.2 以降に掛ける**崩壊上限**（**歴史値からの仮置き** — ADR 0026 の末端層の実測 maxAbs
 * 1.46 の約 2 倍）。
 *
 * 数値パリティではない: 飽和域では f32 経路（1.33）もこの上限を通る。受け持つのは NaN / ±Inf
 * （`compareTensors` はどちらの側の非有限値も不合格にする）と、\|ref\| 上端 28.9 と同じ桁まで
 * 開く発散だけ。上限を広げて通すのではなく、先に i8a8 の scale / accumulator / 適格判定の
 * どれが動いたかを確かめる（数値契約の正本は `tests/gpu_i8a8_test.ts` の atol=0）。
 */
const COLLAPSE_TOLERANCE: Tolerance = { atol: 3, rtol: 0 };

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
const isI8a8 = (key: string): boolean => key.includes(":i8a8:");
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

const censusOf = (diagnostics: SessionDiagnostics): KeyCensus => {
  const entries = diagnostics.lastRunTiming?.entries;
  // MUST: 計測が無効な run を「0 本」として数えない — `otherLinear: 0` の検査が黙って空振り
  // する（`acquireGpu({ gpuTiming: true })` を渡し忘れた形が緑のまま通る）。
  if (entries === undefined) {
    throw new Error(
      "run に lastRunTiming が無い（計測が有効な device で走っていない — この門は " +
        "acquireGpu({ gpuTiming: true }) を前提にしている）",
    );
  }
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

/** 出力 index に掛ける帯（{@link STRICT_OUTPUTS} 未満が厳密・以降が崩壊上限）。 */
const toleranceOf = (index: number): Tolerance =>
  index < STRICT_OUTPUTS ? STRICT_TOLERANCE : COLLAPSE_TOLERANCE;

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
      await runRecordedCase(results, { id: `${VARIANT}/${caseName}` }, async ({ measurements }) => {
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
              const view = io.tensors.get(`output.${index}`);
              assert(view !== undefined, `output.${index} が ${file} に無い`);
              const where = `${VARIANT}/${caseName} output.${index} ('${name}')`;
              const declared = parsed.graph.values[name].dtype;
              assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
              assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
              const tolerance = toleranceOf(index);
              const report = compareTensors(outputs[name], ioTensor(io, view, declared), tolerance);
              measurements.push({
                output: name,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance,
                stage: "karume",
              });
              if (!report.pass) failures.push(`${where}: ${formatAllclose(report)}`);
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

if (RUNNABLE && !TIMESTAMP_QUERY_AVAILABLE) {
  console.warn(
    "[karume] アダプタが 'timestamp-query' を列挙しないため DeBERTa w8a8 のキー census を SKIP " +
      "する（数値門は残る — ADR 0021 の計測は device 作成時にしか要求できない）",
  );
}

Deno.test({
  name:
    `DeBERTa w8a8（${VARIANT}）: run が i8a8 GEMM と quantize_rows だけで linear を回す（キー census）`,
  ignore: !(RUNNABLE && TIMESTAMP_QUERY_AVAILABLE),
  fn: async () => {
    // 内訳は T に依らない（計画は記号次元の束縛で形が変わるだけで、dispatch の並びは同じ）ので
    // 1 ケースで足りる。`padded` を使うのはマスク経路まで同じ run に乗せるため。
    const file = `${ACT_IO_PREFIX}padded${IO_SUFFIX}`;
    const { parsed, inputs } = await openCase(file);
    // MUST: 計測は device 作成時の opt-in（既定は要求しない）。
    const gpu = await acquireGpu({ gpuTiming: true });
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
