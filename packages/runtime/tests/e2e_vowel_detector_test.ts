// 実重み母音認識 CRNN（音声 → リップシンク用の母音系列 `.lab`）の実 GPU golden E2E
// （ADR 0005 の段 3）。
//
// ## 門は 2 段（どちらも残す・置き場はパッケージで分かれる）
//
// - **① ロジットの数値突合**（合成 golden 4 ケース・torch CPU 期待値）= **ここ**。実測は
//   1e-5 級で、{@link LOGIT_TOLERANCE} がその 4.4 倍。
// - **② `.lab` の完全一致**（実音声・全鎖）= `packages/models/tests/
//   e2e_vowel_detector_chain_test.ts`。離散のラベル列なので tolerance を持たない（PNG / WAV の
//   sha256 門と同じ哲学 — 出口の成果物そのものを固定する）。
//
// ②「WAV → 特徴抽出 → グラフ → 後処理」の鎖は両端が models の実装（`src/audio/wav.ts` /
// `src/vowel-detector/*`）なので、models 側に置く — runtime のテストから models の実装を
// 相対 import するのは逆向きの依存。ここは**グラフ 1 本の数値**だけを見る。`.lab` だけだと
// 「ロジットが動いてもラベルが変わらなかった」変更が素通りするので、①が②の下敷きになる。
//
// ## 系列は 1 本（グラフが 1 本・長さは記号）
//
// 時間軸は記号 `T`（20ms 格子）で、グラフ入力は `[1, 2T, 83]`（ADR 0056 / 0057）。合成 golden
// は T10=200 の 1 長だが、**同じグラフ**が実長 4 通りで回ることは②の側が実測で踏む
// （`2T` の派生次元からの束縛 = ADR 0057 の経路）。
//
// ## 資産
//
// - グラフと合成 golden: `outputs/series/vowel-detector-crnn-epoch3/`（`.gitignore` の
//   `outputs/`）。生成コマンドは {@link GENERATE_COMMAND} がそのまま正本。
//
// 資産が無い環境では**明示 SKIP** する。ADR 0005 の「全 SKIP は明示 FAIL」門番
// （tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP とは独立。
// 資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする。

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
 * 合成 golden（torch CPU 期待値）との突合に使う許容誤差。
 *
 * 実測（`atol=rtol=0` の素の突合・golden 4 ケース @ T10=200・出力 `[1, 100, 8]`）:
 *
 * | ケース  | maxAbs  | \|ref\| 上端 |
 * | ------- | ------- | ------------ |
 * | noise   | 3.82e-6 | 7.50         |
 * | ramp    | 3.93e-6 | 9.66         |
 * | silence | 5.01e-6 | 11.40        |
 * | voiced  | 6.68e-6 | 11.38        |
 *
 * 実音声側（`.lab` 門・T10 284〜1236）は torch 期待値を持たないが、**バケット時代の展開グラフ
 * との差は 0**（4 本とも f32 のビット一致 — 移行時の実測）。長さを変えても数値経路は同じ
 * `gru_scan` ノードなので、誤差が T と共に増える構造は残る（展開列だった頃の実測は t284 の
 * 4.17e-6 → t1236 の 1.14e-5）。MUST: **今より長い golden を足すときは測り直す**（外挿ではない）。
 *
 * atol 5e-5 は展開列時代の実測最悪 1.14e-5（T10=1236 / silence）の約 4.4 倍で、記号長へ移った
 * 後も同じ値を据え置いている（数値がビット一致なので、緩める理由も締める理由も無い）。判定は
 * atol が主導する（rtol 1e-6 の寄与は \|ref\| 上端 11.4 でも 1.1e-5 で、atol の 1/4）。
 *
 * 誤差の出所は他の実重み E2E と同じ（fma 融合・縮約順序・超越関数の実装差）。
 *
 * 実装バグ（GRU のゲート順の取り違え・reset ゲートの掛け先の誤り・時間軸と特徴軸の転置）の
 * 誤差はロジットの値域と同じ O(1)〜O(10) で、この閾値の 5 桁以上上に出る（bias を外へ出す
 * 誤り形の実測が 0.196）。
 */
const LOGIT_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * 特徴の次元（80 log-mel + 3 DSP）。正本は
 * `packages/models/src/vowel-detector/features.ts` の `FEATURE_DIM` で、ここは**写し**
 * （runtime のテストから models の実装を import するのは逆向きの依存）。
 */
const FEATURE_DIM = 83;

/** 8 クラス（`LIPSYNC_CLASSES` の本数 — 出力の最終軸）。 */
const CLASS_COUNT = 8;

/**
 * 生成されているはずの合成 golden ケース。**列挙結果ではなくここで固定する**（正本は
 * `tools/export-recipes/vowel_detector/export.py` の `build_cases`）。
 */
const EXPECTED_GOLDEN = ["noise", "ramp", "silence", "voiced"] as const;

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";
const INPUT_NAME = "features";

const SERIES_NAME = "vowel-detector-crnn-epoch3";
const SERIES_ROOT = new URL(`${SERIES_NAME}/`, SERIES_PARENT);

/** SKIP 時にそのまま貼れる生成コマンド（グラフは 1 本 — 長さの指定は要らない）。 */
const GENERATE_COMMAND = "cd tools/export-recipes && uv run python -m vowel_detector.export";

/**
 * 資産ディレクトリの列挙。存在しない場合だけ空に縮退する。
 * MUST: NotFound 以外は伝播させる — 権限エラー等を「資産が無い」と読み替えると、
 * 実行されていない検証が SKIP として静かに緑になる。
 */
const listDir = (url: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(url)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
};

const discoverGolden = (root: URL): readonly string[] =>
  listDir(root)
    .filter((entry) =>
      entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX)
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

/** ファイルの有無。MUST: NotFound 以外は伝播させる（`listDir` と同じ理由）。 */
const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * golden の入力を宣言 dtype の view で組む。
 *
 * MUST: 明示 bindings を渡さない — この門は `2T` の派生次元から `T` を解く経路（ADR 0057）
 * そのものを踏む席で、seed を渡すと解かせる相手が消える。
 */
const goldenInputs = (parsed: KarumeModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} が golden に無い`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

/**
 * グラフの時間軸が**記号**（入力 `2T` / 出力 `T`）で焼かれていることを見る。
 *
 * MUST: 落とさない。長さを固定して焼いた古い形は入出力の名前も階数も同じなので、置き換わって
 * いても「その 1 長のケースだけ」は緑のまま通る。
 */
const assertSymbolicTimeAxis = (parsed: KarumeModel): void => {
  assertEquals(parsed.graph.symbols, ["T"], `${SERIES_NAME} の記号次元`);
  assertEquals(
    parsed.graph.inputs[0].shape,
    [1, "2T", FEATURE_DIM],
    `${SERIES_NAME} の入力 '${INPUT_NAME}' の宣言`,
  );
  assertEquals(
    parsed.graph.values[parsed.graph.outputs[0]].shape,
    [1, "T", CLASS_COUNT],
    `${SERIES_NAME} の出力の宣言`,
  );
};

/** 登録時点で必要なので同期列挙する（`Deno.test` の ignore 判定と同じ理由）。 */
const golden = discoverGolden(SERIES_ROOT);
/** 資産の有無。1 件も無い = 生成していない環境なので全 SKIP（部分的な欠けは FAIL 側）。 */
const available = golden.length > 0;

if (!available) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に export 済み資産が無いため実重み母音検出 E2E を ` +
      `SKIP する（重みがリポジトリ管理外）。生成: ${GENERATE_COMMAND}`,
  );
}

Deno.test({
  name: `母音検出 資産: ${SERIES_NAME} — 期待するケースとモデル本体が揃っている`,
  // 1 件も無い環境は「生成していない」なので SKIP。1 件でもあるなら欠けは FAIL。
  ignore: !available,
  fn: () => {
    assertEquals(golden, [...EXPECTED_GOLDEN], `${SERIES_ROOT.pathname} の golden ケース`);
    assert(exists(new URL(MODEL_FILE, SERIES_ROOT)), `${MODEL_FILE} が無い`);
  },
});

for (const caseName of golden) {
  Deno.test({
    name: `母音検出 golden 突合: ${caseName}（実 GPU / torch CPU 期待値）`,
    ignore: !available || !GPU_AVAILABLE,
    fn: async () => {
      const [modelBytes, ioBytes] = await Promise.all([
        readBuffer(SERIES_ROOT, MODEL_FILE),
        readBuffer(SERIES_ROOT, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
      ]);
      const parsed = openModel(modelBytes);
      const io = parseSafetensors(ioBytes);

      // io の全テンソルがグラフの入出力とちょうど対応する（余りも欠けも無い）。
      const expectedKeys = [
        ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
        ...parsed.graph.outputs.map((_, index) => `output.${index}`),
      ].sort();
      assertEquals(
        [...io.tensors.keys()].sort(),
        expectedKeys,
        `${caseName} の io.safetensors のテンソルキー`,
      );
      assertSymbolicTimeAxis(parsed);

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      try {
        const outputs = await session.run(goldenInputs(parsed, io));
        assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

        parsed.graph.outputs.forEach((name, index) => {
          const view = io.tensors.get(`output.${index}`);
          assert(view !== undefined, `output.${index} が golden に無い`);
          const where = `${SERIES_NAME} / ${caseName} output.${index} ('${name}')`;
          const declared = parsed.graph.values[name].dtype;
          assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
          assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
          const report = compareTensors(
            outputs[name],
            ioTensor(io, view, declared),
            LOGIT_TOLERANCE,
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
