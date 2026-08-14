// 実重み母音認識 CRNN（音声 → リップシンク用の母音系列 `.lab`）の実 GPU E2E（ADR 0005 の段 3）。
//
// 他の実重み E2E が「グラフ 1 本の数値」を見るのに対し、こちらは **WAV から `.lab` までの
// 全鎖**（decode → 特徴抽出 → グラフ → 後処理）を通す唯一の門。鎖の途中（ホスト側の特徴抽出と
// 後処理）は `packages/models/tests/vowel_detector_host_test.ts` が Python 正本とのパリティで
// 押さえており、ここはその 2 つを実 GPU 推論で繋いだときに**意味のある `.lab` が出る**ことを
// 見る。
//
// ## 門は 2 段（どちらも残す）
//
// - **① `.lab` の完全一致**（実音声・全鎖）: 離散のラベル列なので tolerance を持たない
//   （PNG / WAV の sha256 門と同じ哲学 — 出口の成果物そのものを固定する）。
// - **② ロジットの数値突合**（合成 golden 4 ケース・torch CPU 期待値）: `.lab` だけだと
//   「ロジットが動いてもラベルが変わらなかった」変更が素通りする。実測は 1e-5 級で、
//   {@link LOGIT_TOLERANCE} がその 4.4 倍。
//
// ## 系列は 1 本（グラフが 1 本・長さは記号）
//
// 時間軸は記号 `T`（20ms 格子）で、グラフ入力は `[1, 2T, 83]`（ADR 0056 / 0057）。したがって
// **4 本の音声はすべて同じグラフ**を、それぞれの実長で回る。束縛は明示 seed を渡さず
// **入力 shape から解かせる**（`2T` の派生次元からの束縛 = ADR 0057 の経路を、この門が実測で
// 踏む唯一の席。配布形パイプラインの側は逆に明示 seed を渡す）。
//
// **pad は 1 要素も無い**（奇数フレームの端数 1 本だけを切り捨てる）。以前は配布形だけが
// 長さバケット + 右ゼロ pad を採っていて、pad が末尾のロジットを O(1) 動かすせいで `.lab` が
// この門と割れていた（実測: pad 2 フレーム = 40ms でも max abs diff 2.2〜3.2）。バケットが
// 消えた今は**配布形の `.lab` もここと完全一致**する（`packages/models/tests/
// e2e_vowel_detector_lab_test.ts` が同じ 4 本で実測する）。
//
// ## 資産
//
// - グラフ: `outputs/series/vowel-detector-crnn-epoch3/`（`.gitignore` の `outputs/`）。
//   生成コマンドは {@link GENERATE_COMMAND} がそのまま正本。
// - 実音声: `outputs/demo/vowel-<ケース>.wav`（16kHz mono）。生成台本は
//   `examples/irodori/eval-audio.ts`（テキスト / seed / リサンプルの正本はあちら）で、日本語
//   TTS（Irodori）が焼いた 48kHz を 1/3 に間引いたもの。**素材の同一性は sha256 で固定する** —
//   焼き直したら期待 `.lab` も採り直しになる。
// - mel 基底: `packages/models/tests/fixtures/vowel-detector/parity.json`（git 追跡・上流
//   `assets/feature_config.json` そのまま）。配布形は同じ行列を `assets` 席の safetensors で
//   配る（`karume dist --pipeline vowel-detector`）が、この門は配布形を経由しない。
//
// 資産が無い環境では**明示 SKIP** する（系列ごと・音声ごとに独立）。ADR 0005 の「全 SKIP は
// 明示 FAIL」門番（tests/gpu_gate_test.ts）は *GPU アダプタの有無* だけを見ており、この SKIP
// とは独立。資産が**中途半端に**（ケース欠け）存在する場合は SKIP ではなく FAIL にする。

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
// NOTE: 実音声ケースは models パッケージのホスト層**そのもの**を通すのが主張の中身なので、
// テストからだけ相対で読む（どれも依存ゼロの純関数で、runtime の実装は何も引き込まない。
// publish は `tests/` を除外する — packages/runtime/deno.json）。
import { decodeWav } from "../../models/src/audio/wav.ts";
import {
  extractFeatures,
  FEATURE_DIM,
  SAMPLE_RATE,
} from "../../models/src/vowel-detector/features.ts";
import { logitsToSegments, toLab } from "../../models/src/vowel-detector/postprocess.ts";

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
 * 実音声 1 本の宣言（グラフは 4 本で共有 — 違うのは長さだけ）。
 *
 * `frames` / `length` / `sha256` / `lab` は**全て実測値**で、音声を焼き直したら 4 つとも
 * 採り直しになる（{@link AUDIO_COMMAND} → {@link GENERATE_COMMAND} の順）。
 */
type Case = {
  /** 音声ケース名（`examples/irodori/eval-audio.ts` の `CASES` と 1:1）。 */
  readonly name: string;
  /** 何を測る素材か（人が読むための欄 — 台本側の `why` の写し）。 */
  readonly why: string;
  /** `extractFeatures` が返す 10ms フレーム数（波形長から決まる）。 */
  readonly frames: number;
  /** グラフへ渡す入力長 T10（= `frames` を偶数へ落としたもの）。 */
  readonly length: number;
  /** 16kHz WAV の sha256（golden を採った素材と同一であることを実行前に見る）。 */
  readonly sha256: string;
  /** 期待する `.lab` 全文（実 GPU 出力・torch CPU 出力の両方と一致することを確認済み）。 */
  readonly lab: string;
};

/**
 * 実走するケース。**列挙ではなくここで固定する** — 生成済みのものを拾う形にすると、
 * 一部だけ生成し忘れた環境で「緑だが未検証」になる。
 *
 * 長さは 2.88s / 3.48s / 8.04s / 12.40s に散らしてある（**同じ 18 ノードのグラフ**を 4 通りの
 * 束縛で回す — 展開列だった頃はここが 10,812 / 13,092 / 30,420 / 46,988 ノードの別グラフ
 * 4 本だった）。`vowels` だけは内容も検査に使う（{@link VOWEL_SEQUENCE}）。
 */
const CASES: readonly Case[] = [
  {
    name: "short",
    why: "2 秒級・1 文（「ありがとうございます。」）",
    frames: 285,
    length: 284,
    sha256: "0b48165fd11bf34c1f2b7904e4d53b5efa3f52c9baefc6a54566f850a9a30dd8",
    lab: `0.0000000 0.2000000 a
0.2000000 0.5000000 o
0.5000000 0.6400000 a
0.6400000 0.7000000 i
0.7000000 1.7400000 a
1.7400000 2.3800000 u
2.3800000 2.8000000 i
2.8000000 2.8400000 pau
`,
  },
  {
    name: "vowels",
    why: "3 秒級・5 母音を区切って読む（「あ、い、う、え、お。」）",
    frames: 345,
    length: 344,
    sha256: "ae3712e01da6906a214b46c52490d106e1dfa598528e611e8f0f85ce6207a829",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.4000000 a
0.4000000 0.9600000 i
0.9600000 1.6000000 u
1.6000000 2.3000000 e
2.3000000 3.4400000 o
`,
  },
  {
    name: "mid",
    why: "8 秒級・3 文",
    frames: 801,
    length: 800,
    sha256: "b63d35c12d9d60af44754db209b907f03e3fd5f46f25e6bb0a7ccb5195df734a",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.1600000 o
0.1600000 0.2200000 N
0.2200000 0.5400000 i
0.5400000 0.8800000 a
0.8800000 0.9200000 pau
0.9200000 1.8000000 o
1.8000000 2.0200000 a
2.0200000 2.1800000 o
2.1800000 2.3200000 e
2.3200000 2.5200000 o
2.5200000 2.7200000 i
2.7200000 2.9000000 e
2.9000000 2.9600000 N
2.9600000 3.0800000 i
3.0800000 3.2000000 e
3.2000000 3.3400000 u
3.3400000 3.7400000 e
3.7400000 4.6000000 a
4.6000000 4.6600000 N
4.6600000 4.8200000 o
4.8200000 4.9600000 i
4.9600000 5.0800000 e
5.0800000 5.2000000 a
5.2000000 5.3400000 e
5.3400000 5.4400000 u
5.4400000 5.5600000 i
5.5600000 5.8600000 a
5.8600000 5.9400000 pau
5.9400000 6.5600000 o
6.5600000 6.9000000 i
6.9000000 7.0200000 e
7.0200000 7.1400000 u
7.1400000 7.2600000 a
7.2600000 7.6000000 o
7.6000000 7.6600000 i
7.6600000 7.9000000 a
7.9000000 8.0000000 pau
`,
  },
  {
    name: "long",
    why: "12 秒級・2 文（この門で最長の束縛 T=618）",
    frames: 1237,
    length: 1236,
    sha256: "28013dab4551c353d3f95e51733e69d71de43a719bf3c4f120eaa1082013f59b",
    lab: `0.0000000 0.0400000 pau
0.0400000 0.2400000 o
0.2400000 0.5600000 e
0.5600000 0.9400000 a
0.9400000 1.0000000 pau
1.0000000 1.3800000 o
1.3800000 1.5000000 i
1.5000000 1.6000000 N
1.6000000 1.7400000 o
1.7400000 2.0200000 a
2.0200000 2.1600000 i
2.1600000 2.3800000 o
2.3800000 2.4800000 i
2.4800000 2.6200000 a
2.6200000 2.7400000 i
2.7400000 3.0200000 e
3.0200000 3.0800000 pau
3.0800000 3.4000000 u
3.4000000 3.5400000 i
3.5400000 3.6600000 o
3.6600000 3.9200000 a
3.9200000 4.0400000 i
4.0400000 4.3400000 a
4.3400000 4.4000000 i
4.4000000 4.5200000 u
4.5200000 4.6600000 a
4.6600000 4.7600000 i
4.7600000 4.9000000 o
4.9000000 5.1200000 u
5.1200000 5.2200000 i
5.2200000 5.3800000 a
5.3800000 7.1800000 i
7.1800000 7.3400000 a
7.3400000 7.4400000 i
7.4400000 7.6000000 u
7.6000000 7.6600000 N
7.6600000 7.8000000 e
7.8000000 7.9800000 o
7.9800000 8.3600000 a
8.3600000 8.4600000 i
8.4600000 8.6000000 u
8.6000000 8.6800000 N
8.6800000 8.8200000 e
8.8200000 9.1000000 o
9.1000000 9.1600000 pau
9.1600000 9.5400000 o
9.5400000 9.6800000 a
9.6800000 9.8200000 i
9.8200000 9.9600000 e
9.9600000 10.1400000 u
10.1400000 10.1800000 N
10.1800000 10.3400000 e
10.3400000 10.5000000 o
10.5000000 10.6000000 i
10.6000000 10.7400000 e
10.7400000 10.8400000 i
10.8400000 10.9400000 u
10.9400000 11.3400000 o
11.3400000 11.4800000 a
11.4800000 11.5800000 i
11.5800000 11.7000000 a
11.7000000 11.9400000 e
11.9400000 12.0000000 i
12.0000000 12.2600000 a
12.2600000 12.3600000 pau
`,
  },
];

/**
 * `vowels` ケースの `.lab` から無音を除いたラベル列。
 *
 * 期待 `.lab` の完全一致とは**主張が違う** — あちらは「前回と同じ値が出る」で、こちらは
 * 「5 母音を順に読ませたら 5 母音が順に出る」= 鎖が意味を持つことの主張。音声を焼き直して
 * 期待 `.lab` を採り直すとき、壊れた出力をそのまま固定してしまう事故はこちらが止める。
 */
const VOWEL_SEQUENCE: readonly string[] = ["a", "i", "u", "e", "o"];
const VOWEL_CASE = "vowels";

/** 8 クラス（`LIPSYNC_CLASSES` の本数 — 出力の最終軸）。 */
const CLASS_COUNT = 8;

/**
 * 生成されているはずの合成 golden ケース。**列挙結果ではなくここで固定する**（正本は
 * `tools/exporter/export_vowel_detector.py` の `build_cases`）。
 */
const EXPECTED_GOLDEN = ["noise", "ramp", "silence", "voiced"] as const;

const SERIES_PARENT = new URL("../../../outputs/series/", import.meta.url);
const DEMO_DIR = new URL("../../../outputs/demo/", import.meta.url);
const FIXTURE_PATH = new URL(
  "../../models/tests/fixtures/vowel-detector/parity.json",
  import.meta.url,
);
const MODEL_FILE = "model.safetensors";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";
const INPUT_NAME = "features";

const SERIES_NAME = "vowel-detector-crnn-epoch3";
const SERIES_ROOT = new URL(`${SERIES_NAME}/`, SERIES_PARENT);
const audioFile = (entry: Case): string => `vowel-${entry.name}.wav`;

/** SKIP 時にそのまま貼れる生成コマンド（グラフは 1 本 — 長さの指定は要らない）。 */
const GENERATE_COMMAND = "cd tools/exporter && uv run --frozen python export_vowel_detector.py";

/** 実音声そのものを焼き直すコマンド（テキスト / seed の正本は台本側）。 */
const AUDIO_COMMAND = "deno task demo:eval-audio --source <Irodori 配布形のパス>";

/** mel 基底のフィクスチャ（`melBasis` 以外の欄はこの門では使わない）。 */
type Fixture = { readonly melBasis: number[][] };

const melBasis: Float32Array = Float32Array.from(
  (JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture).melBasis.flat(),
);

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

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

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
 * いても「その 1 長のケースだけ」は緑のまま通る（残り 3 本が別の理由で落ちたように見える）。
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

for (const entry of CASES) {
  /** 実音声の門。系列と WAV の**両方**が揃ってはじめて実走する。 */
  const audioAvailable = available && exists(new URL(audioFile(entry), DEMO_DIR));

  if (available && !audioAvailable) {
    console.warn(
      `[karume] 実音声ケース (${entry.name}) を SKIP する（${audioFile(entry)} が無い）。` +
        `生成: ${AUDIO_COMMAND}`,
    );
  }

  Deno.test({
    name: `母音検出 実音声の全鎖: ${entry.name} — WAV → 特徴 → 実 GPU → .lab（${entry.why}）`,
    ignore: !audioAvailable || !GPU_AVAILABLE,
    fn: async () => {
      const wavBytes = await Deno.readFile(new URL(audioFile(entry), DEMO_DIR));
      // ① 期待 `.lab` を採った音声と、いま読んでいる音声が同一であること。**tolerance では
      // 吸収されない差**（台本を回し直して期待値を採り直していない）を、実行の前に名指しで落とす。
      assertEquals(
        await sha256Hex(wavBytes),
        entry.sha256,
        `${audioFile(entry)} が期待 .lab を採った音声と違う（採り直す: ${AUDIO_COMMAND}）`,
      );

      const wav = decodeWav(wavBytes);
      assertEquals(wav.sampleRate, SAMPLE_RATE, `${audioFile(entry)} の周波数`);
      const features = extractFeatures(wav.data, melBasis);
      assertEquals(features.frames, entry.frames, `${entry.name} の 10ms フレーム数`);
      // 出力は 20ms 格子なので、奇数フレームの端数 1 本は落として偶数長で回す（切り捨てで
      // あって pad ではない — 冒頭の「pad は 1 要素も無い」）。
      const usable = features.frames - (features.frames % 2);
      assertEquals(usable, entry.length, `${entry.name} の偶数化フレーム数`);

      const parsed = openModel(await readBuffer(SERIES_ROOT, MODEL_FILE));
      assertSymbolicTimeAxis(parsed);
      const [outputName] = parsed.graph.outputs;

      const gpu = await acquireGpu();
      const session = await createSession(gpu, parsed);
      let logits: Float32Array;
      try {
        const output = (await session.run({
          [INPUT_NAME]: {
            dtype: "f32",
            shape: [1, usable, FEATURE_DIM],
            data: features.data.slice(0, usable * FEATURE_DIM),
          },
        }))[outputName];
        // 判別子で絞る（Float32Array へのキャストは dtype がずれたときに黙って通る）。
        assert(output.dtype === "f32", `${entry.name}: ロジットの dtype が ${output.dtype}`);
        assertEquals(output.shape, [1, usable / 2, CLASS_COUNT], `${entry.name}: 出力の形`);
        logits = output.data;
      } finally {
        await session.dispose();
        gpu.destroy();
      }

      // ② `.lab` の完全一致（tolerance を持たない離散の門）。
      const segments = logitsToSegments(logits, usable / 2);
      assertEquals(toLab(segments), entry.lab, `${entry.name} の .lab`);

      if (entry.name === VOWEL_CASE) {
        // 期待 `.lab` の一致とは別の主張（{@link VOWEL_SEQUENCE}）。
        assertEquals(
          segments.map((segment) => segment.label).filter((label) => label !== "pau"),
          [...VOWEL_SEQUENCE],
          "「あ、い、う、え、お。」から 5 母音が順に出ていない",
        );
      }
    },
  });
}
